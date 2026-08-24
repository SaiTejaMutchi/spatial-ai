"""Filesystem-backed scan sessions and the orchestration of one processing run.

Everything here is a call into `pipeline`. If a line of this file starts to look
like geometry, it belongs in `pipeline` instead — the plan is explicit that the
API must not duplicate connector, geometry, rendering, benchmark, AI, or
loss-preview logic, and keeping that boundary honest is what makes the final
Stray run a rerun of the same code rather than a second implementation.

A local SQLite catalog (`_results.db` under the state directory) stores
completed AI and opening results so the same room is not sent to Groq again.
"""

from __future__ import annotations

import json
import shutil
import threading
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from pipeline.ai.verifier import (
    VerifierResult, load_ai_config, run_verifier, validate_assessment,
)
from pipeline.connectors.base import ConnectorError
from pipeline.connectors.detect import CONNECTORS, detect_source
from pipeline.contracts.normalized_capture import NormalizedCapture
from pipeline.contracts.validate import validate
from pipeline.contracts.validate_model import validate_model
from pipeline.evidence.select import select_evidence_views
from pipeline.geometry.config import load_geometry_config
from pipeline.geometry.model import write_model
from pipeline.geometry.run import run_geometry
from pipeline.rendering.floorplan import render_floorplan
from pipeline.rendering.model_3d import write_model_3d

from .models import STAGES, ArtifactRecord, FailureClass, ScanStatus, StageState
from .results_db import (
    ResultsDB,
    ai_cache_key,
    opening_cache_key,
)
from .export_public_results import EVIDENCE_DIR, PUBLIC_RESULTS_DIR


def clean_source_path(source_path: str) -> Path:
    """Accept a pasted path that still carries wrapping quotes or spaces.

    The Add-capture field is a plain text box. Pasting a quoted shell path is
    a common miss; those quotes are not part of the filesystem path.
    """
    text = (source_path or "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return Path(text).expanduser()


def safe_upload_relative(relative: str) -> Path:
    """Keep a browser folder listing inside the incoming directory."""
    text = (relative or "").replace("\\", "/").strip()
    if not text or text.startswith("/") or text.startswith("~"):
        raise ValueError(f"invalid upload path '{relative}'")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"invalid upload path '{relative}'")
    return Path(*parts)


def extract_zip_upload(handle, dest: Path) -> int:
    """Unpack a capture zip onto this machine. Paths stay inside dest."""
    if hasattr(handle, "seek"):
        try:
            handle.seek(0)
        except (OSError, ValueError):
            pass
    if isinstance(handle, (bytes, bytearray)):
        handle = BytesIO(handle)
    written = 0
    with zipfile.ZipFile(handle) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename.endswith("/"):
                continue
            dest_file = dest / safe_upload_relative(info.filename)
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, dest_file.open("wb") as out:
                shutil.copyfileobj(source, out)
            written += 1
    if written == 0:
        raise ValueError("the zip did not contain any files")
    return written


def unwrap_single_root(incoming: Path) -> Path:
    children = [path for path in incoming.iterdir() if path.name != ".DS_Store"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return incoming


def describe_authenticity(state: dict) -> dict:
    """Cite where a capture came from. No accuracy number is invented."""
    classification = state.get("classification") or ""
    source_type = state.get("sourceType") or ""
    source_path = str(state.get("sourcePath") or "").replace("\\", "/")
    in_samples = "/samples/" in source_path
    dataset = {
        "arkitscenes": "ARKitScenes",
        "stray_scanner": "Stray Scanner",
        "unity_obj": "Unity OBJ",
    }.get(source_type)

    if classification == "public_development_fixture":
        who = dataset or "Published dataset"
        return {
            "kind": "published_dataset",
            "label": "Published dataset",
            "cite": (
                f"{who} public fixture. Development only — not a tape-measure "
                "result. FARO correspondence to this room was not established, "
                "so no accuracy is written."
            ),
        }
    if classification == "public_review_example":
        return {
            "kind": "public_review_example",
            "label": "Sanitized example",
            "cite": (
                "Published review example. The original private capture is not "
                "in this repository. No tape accuracy is written."
            ),
        }
    if classification == "public_sample":
        return {
            "kind": "public_sample",
            "label": "Public sample",
            "cite": (
                "Public Stray interoperability sample. No independent ground "
                "truth is available, so accuracy is not written."
            ),
        }
    if classification == "final_private_capture":
        return {
            "kind": "local_capture",
            "label": "This machine",
            "cite": "Folder processed on this machine. Authenticity is the capture itself.",
        }
    if in_samples:
        who = dataset or "Repository sample"
        return {
            "kind": "public_sample",
            "label": "Public sample",
            "cite": f"{who} in this repository. Not a private iPhone acceptance capture.",
        }
    return {
        "kind": "unclassified",
        "label": "Unclassified source",
        "cite": "The source has not been classified.",
    }


def geometry_stride_for_source(source_type: str, requested: int) -> int:
    """Keep frozen geometry, but do not starve a short Stray capture.

    The public Stray export is already connector-strided to ~70 frames.
    Frozen estimators recover floor, ceiling and walls at geometry stride 1
    or 2, and lose the ceiling at stride 4. ARKit development fixtures keep
    the existing default.
    """
    if source_type == "stray_scanner":
        return 1
    return requested

ARTIFACT_SPEC = [
    ("spatial_model.json", "development-only", "Canonical insurance-facing model"),
    ("floorplan.svg", "development-only", "Dimensioned 2D plan, generated from the model"),
    ("room_model.obj", "development-only", "Semantic 3D surfaces with stable IDs"),
    ("room_model.mtl", "development-only", "Materials encoding observation state"),
    ("room_model_entity_map.json", "development-only", "OBJ group to surface ID map"),
    ("evidence_manifest.json", "development-only", "Registered RGB evidence views"),
    ("ai_assessment.json", "experimental", "Spatial AI Verifier assessment"),
    ("geometry_diagnostics.json", "development-only", "Fit diagnostics and provenance"),
    ("benchmark.csv", "pending", "Model versus tape comparison (final data only)"),
    ("benchmark.md", "pending", "Benchmark report (final data only)"),
    ("loss_preview.json", "experimental", "Experimental P&C grounding preview"),
    ("visible_condition.json", "experimental", "Model-generated visible condition grounding"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ScanManager:
    """Holds scan sessions on disk plus a local results catalog. No accounts."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self.db = ResultsDB(self.root / "_results.db")

    # -- session state ----------------------------------------------------

    def _dir(self, scan_id: str) -> Path:
        local = self.root / scan_id
        if (local / "state.json").is_file():
            return local
        published = PUBLIC_RESULTS_DIR / scan_id
        if (published / "state.json").is_file():
            return published
        return local

    def is_published_example(self, scan_id: str) -> bool:
        published = PUBLIC_RESULTS_DIR / scan_id / "state.json"
        local = self.root / scan_id / "state.json"
        return published.is_file() and not local.is_file()

    def _state_path(self, scan_id: str) -> Path:
        return self._dir(scan_id) / "state.json"

    def _read(self, scan_id: str) -> dict:
        path = self._state_path(scan_id)
        if not path.is_file():
            raise KeyError(scan_id)
        return json.loads(path.read_text())

    def _write(self, state: dict) -> None:
        state["updatedAt"] = _now()
        self._state_path(state["scanId"]).write_text(json.dumps(state, indent=2) + "\n")
        self.db.upsert_scan(state)

    def create(self, source_path: str, label: str | None,
               classification: str) -> dict:
        source = clean_source_path(source_path)
        scan_id = f"scan-{uuid.uuid4().hex[:12]}"
        directory = self._dir(scan_id)
        (directory / "output").mkdir(parents=True, exist_ok=True)

        state = {
            "scanId": scan_id,
            "status": ScanStatus.created.value,
            "label": label or source.name,
            "classification": classification,
            "sourcePath": str(source),
            "sourceType": None,
            "connector": None,
            "stages": [{"name": name, "state": StageState.pending.value,
                        "detail": None, "startedAt": None, "finishedAt": None}
                       for name in STAGES],
            "currentStage": None,
            "failureClass": None,
            "error": None,
            "createdAt": _now(),
            "updatedAt": _now(),
            "summary": {},
        }

        if not source.exists():
            self._set_stage(state, "upload_received", StageState.failed,
                            f"'{source}' does not exist on this machine")
            state["status"] = ScanStatus.failed.value
            state["failureClass"] = FailureClass.connector.value
            state["error"] = f"the capture path '{source}' does not exist"
            self._write(state)
            return state

        self._set_stage(state, "upload_received", StageState.complete,
                        f"accepted {source.name} without preprocessing")

        # Source detection happens at intake so the UI can name the connector
        # before any processing is asked for.
        try:
            connector_cls = detect_source(source)
            state["sourceType"] = connector_cls.source_type
            state["connector"] = connector_cls.__name__
            self._set_stage(state, "source_detected", StageState.complete,
                            f"recognised as {connector_cls.source_type}")
        except ConnectorError as exc:
            self._set_stage(state, "source_detected", StageState.failed, str(exc))
            state["status"] = ScanStatus.failed.value
            state["failureClass"] = FailureClass.connector.value
            state["error"] = str(exc)

        self._write(state)
        return state

    def create_from_upload(self, files, label: str | None,
                           classification: str) -> dict:
        """Copy a browser-chosen folder onto this machine, then run normal intake.

        The bytes never leave the host. The native Mac/Windows folder picker
        cannot hand us a filesystem path, so the service writes a local copy
        and the existing connector path continues from there.
        """
        incoming = self.root / "_incoming" / uuid.uuid4().hex[:12]
        incoming.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            pairs = list(files)
            only = pairs[0] if len(pairs) == 1 else None
            if only and Path(only[0]).suffix.lower() == ".zip":
                written = extract_zip_upload(only[1], incoming)
            else:
                for relative, handle in pairs:
                    dest = incoming / safe_upload_relative(relative)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if hasattr(handle, "seek"):
                        try:
                            handle.seek(0)
                        except (OSError, ValueError):
                            pass
                    with dest.open("wb") as out:
                        if hasattr(handle, "read"):
                            shutil.copyfileobj(handle, out)
                        else:
                            out.write(handle)
                    written += 1
            if written == 0:
                raise ValueError("the folder did not contain any files")
            source = unwrap_single_root(incoming)
            state = self.create(str(source), label or source.name, classification)
            if state.get("failureClass") is None:
                self._set_stage(
                    state, "upload_received", StageState.complete,
                    f"copied {written} files onto this machine as {source.name}")
                self._write(state)
            return state
        except Exception:
            if written == 0:
                shutil.rmtree(incoming, ignore_errors=True)
            raise

    def get(self, scan_id: str) -> dict:
        state = self._read(scan_id)
        self._hydrate_results(scan_id)
        return state

    def list_scans(self) -> list[dict]:
        scans = []
        seen: set[str] = set()
        paths = list(sorted(self.root.glob("scan-*/state.json")))
        paths.extend(sorted(PUBLIC_RESULTS_DIR.glob("*/state.json")))
        for path in paths:
            state = json.loads(path.read_text())
            scan_id = state.get("scanId")
            if not scan_id or scan_id in seen:
                continue
            seen.add(scan_id)
            scans.append(state)
        return sorted(scans, key=lambda s: s["createdAt"], reverse=True)

    def rename(self, scan_id: str, name: str) -> dict:
        """Rename a saved space. The only mutation the library screen needs."""
        if self.is_published_example(scan_id):
            raise PermissionError("published review examples cannot be renamed")
        state = self._read(scan_id)
        cleaned = " ".join(name.split())[:80]
        if not cleaned:
            raise ValueError("a name cannot be empty")
        state["label"] = cleaned
        self._write(state)
        return state

    def library_summary(self, state: dict) -> dict:
        """A user-facing record for the library, derived from real artifacts.

        Counts are read back from the generated files rather than cached at
        process time, so a scan produced before these fields existed still
        shows its findings instead of silently reporting none.
        """
        summary = dict(state.get("summary") or {})
        output = self._dir(state["scanId"]) / "output"

        findings = 0
        needs_review = 0
        ai_status = summary.get("aiStatus")
        assessment_path = output / "ai_assessment.json"
        if assessment_path.is_file():
            try:
                assessment = json.loads(assessment_path.read_text())["assessment"]
                ai_status = assessment.get("status", ai_status)
                for finding in assessment.get("findings", []):
                    findings += 1
                    if finding.get("status") == "review_recommended":
                        needs_review += 1
            except (json.JSONDecodeError, KeyError):
                pass

        condition_path = output / "visible_condition.json"
        condition_status = None
        if condition_path.is_file():
            try:
                condition = json.loads(condition_path.read_text())
                proposal = condition.get("proposal")
                condition_status = ("supported" if proposal else
                                    condition.get("status") or "no_supported_condition")
                if proposal:
                    findings += 1
                    if proposal.get("reviewStatus") == "human_review_required":
                        needs_review += 1
            except json.JSONDecodeError:
                pass

        openings_unresolved = summary.get("unresolvedOpenings")
        confirmed_openings = summary.get("confirmedOpenings")
        model_path = output / "spatial_model.json"
        if model_path.is_file() and (
                openings_unresolved is None or confirmed_openings is None):
            try:
                model = json.loads(model_path.read_text())
                openings_unresolved = sum(
                    1 for o in model["openings"]
                    if o["observationState"] == "unresolved")
                confirmed_openings = sum(
                    1 for o in model["openings"]
                    if o["observationState"] != "unresolved" and o.get("dimensions"))
            except (json.JSONDecodeError, KeyError):
                openings_unresolved = openings_unresolved
                confirmed_openings = confirmed_openings

        return {
            "length_m": summary.get("roomLength_m"),
            "width_m": summary.get("roomWidth_m"),
            "height_m": summary.get("roomHeight_m"),
            "floorArea_m2": summary.get("floorArea_m2"),
            "surfaceCount": summary.get("surfaceCount"),
            "evidenceViews": summary.get("evidenceViews"),
            "observedPerimeterFraction": summary.get("observedPerimeterFraction"),
            "aiStatus": ai_status,
            "aiFindingCount": findings,
            "needsReviewCount": needs_review,
            "conditionStatus": condition_status,
            "unresolvedOpenings": openings_unresolved,
            "confirmedOpenings": confirmed_openings,
            "lossPreviewStatus": summary.get("lossPreviewStatus"),
        }

    def library_record(self, state: dict) -> dict:
        output = self._dir(state["scanId"]) / "output"
        thumbnail = "floorplan.svg" if (output / "floorplan.svg").is_file() else None
        return {
            "id": state["scanId"],
            "name": state.get("label") or state["scanId"],
            "createdAt": state["createdAt"],
            "updatedAt": state["updatedAt"],
            "status": state["status"],
            "classification": state["classification"],
            "sourceType": state.get("sourceType"),
            "connector": state.get("connector"),
            "failureClass": state.get("failureClass"),
            "error": state.get("error"),
            "thumbnailArtifact": thumbnail,
            "authenticity": describe_authenticity(state),
            "summary": self.library_summary(state),
        }

    def delete(self, scan_id: str) -> None:
        if self.is_published_example(scan_id):
            raise PermissionError("published review examples cannot be deleted")
        directory = self.root / scan_id
        if not (directory / "state.json").is_file():
            raise KeyError(scan_id)
        shutil.rmtree(directory)
        self.db.delete_scan(scan_id)

    # -- stage bookkeeping -------------------------------------------------

    @staticmethod
    def _set_stage(state: dict, name: str, value: StageState,
                   detail: str | None = None) -> None:
        for stage in state["stages"]:
            if stage["name"] != name:
                continue
            if value == StageState.running:
                stage["startedAt"] = _now()
            if value in (StageState.complete, StageState.failed,
                         StageState.skipped, StageState.not_applicable):
                stage["finishedAt"] = _now()
            stage["state"] = value.value
            stage["detail"] = detail
            return

    def _ai_key(self, model: dict) -> tuple[str, object]:
        config = load_ai_config()
        return ai_cache_key(model, config.sha256, config.model), config

    def _opening_key(self, model: dict) -> tuple[str, object]:
        config = load_ai_config()
        return opening_cache_key(model, config.sha256, config.model), config

    def _verifier_from_cache(self, cached: dict) -> VerifierResult:
        diagnostics = dict(cached.get("diagnostics") or {})
        diagnostics["cacheHit"] = True
        return VerifierResult(
            assessment=cached["assessment"],
            rejected_findings=cached.get("rejectedFindings") or [],
            diagnostics=diagnostics,
        )

    def _run_or_load_openings(self, scan_id: str, model: dict, output: Path,
                              **resolver_kwargs) -> dict:
        from pipeline.ai.opening_pipeline import resolve_scan_openings

        key, _config = self._opening_key(model)
        cached = self.db.get_openings(key)
        current_ids = {item.get("id") for item in model.get("openings") or []}
        if cached and current_ids == {item.get("id") for item in cached["openings"]}:
            model["openings"] = cached["openings"]
            report = dict(cached["report"])
            report.setdefault("diagnostics", {})["cacheHit"] = True
            (output / "opening_resolutions.json").write_text(
                json.dumps(report, indent=2) + "\n")
            return report
        report = resolve_scan_openings(model, output, **resolver_kwargs)
        self.db.put_openings(key, scan_id, model.get("openings") or [], report)
        return report

    def _run_or_load_verifier(self, scan_id: str, model: dict,
                              evidence_dir: Path) -> VerifierResult:
        key, _config = self._ai_key(model)
        cached = self.db.get_completed_ai(key)
        if cached:
            return self._verifier_from_cache(cached)
        verifier = run_verifier(model, evidence_dir)
        self.db.put_ai(
            key, scan_id, verifier.assessment,
            verifier.rejected_findings, verifier.diagnostics)
        return verifier

    def _hydrate_results(self, scan_id: str) -> None:
        """Record a completed on-disk review so a later retry does not call Groq."""
        output = self._dir(scan_id) / "output"
        model_path = output / "spatial_model.json"
        if not model_path.is_file():
            return
        try:
            model = json.loads(model_path.read_text())
        except json.JSONDecodeError:
            return
        assessment_path = output / "ai_assessment.json"
        if assessment_path.is_file():
            try:
                payload = json.loads(assessment_path.read_text())
                key, _config = self._ai_key(model)
                self.db.put_ai(
                    key, scan_id, payload.get("assessment") or {},
                    payload.get("rejectedFindings") or [],
                    payload.get("diagnostics") or {})
            except json.JSONDecodeError:
                pass
        openings_path = output / "opening_resolutions.json"
        if openings_path.is_file() and model.get("openings") is not None:
            try:
                report = json.loads(openings_path.read_text())
                key, _config = self._opening_key(model)
                self.db.put_openings(key, scan_id, model["openings"], report)
            except json.JSONDecodeError:
                pass

    # -- processing --------------------------------------------------------

    def process(self, scan_id: str, frame_stride: int = 6,
                geometry_stride: int = 4) -> dict:
        lock = self._locks.setdefault(scan_id, threading.Lock())
        with lock:
            state = self._read(scan_id)
            state["status"] = ScanStatus.processing.value
            state["error"] = None
            state["failureClass"] = None
            self._write(state)
            try:
                self._run(state, frame_stride, geometry_stride)
            except Exception as exc:  # noqa: BLE001 - surfaced to the operator
                state["status"] = ScanStatus.failed.value
                state["error"] = f"{type(exc).__name__}: {exc}"
                state.setdefault("summary", {})["traceback"] = \
                    traceback.format_exc(limit=6)
                if state["failureClass"] is None:
                    state["failureClass"] = FailureClass.output.value
            self._write(state)
            return state

    def _run(self, state: dict, frame_stride: int, geometry_stride: int) -> None:
        scan_id = state["scanId"]
        directory = self._dir(scan_id)
        output = directory / "output"
        normalized = directory / "normalized_capture"
        source = Path(state["sourcePath"])
        geometry_stride = geometry_stride_for_source(
            state.get("sourceType") or "", geometry_stride)

        if state["sourceType"] is None:
            state["failureClass"] = FailureClass.connector.value
            raise ConnectorError("no connector recognised this source at intake")

        # ---- connector -------------------------------------------------
        self._set_stage(state, "connector_validation", StageState.running)
        self._write(state)
        connector_cls = CONNECTORS[state["sourceType"]]
        supported = connector_cls.accepted_options()
        options = {k: v for k, v in {
            "classification": state["classification"],
            "stride": frame_stride,
        }.items() if k in supported}
        if normalized.exists():
            shutil.rmtree(normalized)
        try:
            capture = connector_cls(source, **options).normalize(normalized)
        except ConnectorError as exc:
            state["failureClass"] = FailureClass.connector.value
            self._set_stage(state, "connector_validation", StageState.failed, str(exc))
            raise
        self._set_stage(state, "connector_validation", StageState.complete,
                        f"{len(capture.frames)} frames retained, "
                        f"{len(capture.excluded_frames)} excluded")

        # ---- frame alignment and contract validation --------------------
        self._set_stage(state, "frame_alignment", StageState.complete,
                        f"matched by timestamp within "
                        f"{capture.frame_selection.get('pose_tolerance_s', 0) * 1000:.0f} ms")
        self._set_stage(state, "normalized_capture", StageState.running)
        self._write(state)
        issues = validate(normalized)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            state["failureClass"] = FailureClass.connector.value
            detail = "; ".join(i.message for i in errors[:3])
            self._set_stage(state, "normalized_capture", StageState.failed, detail)
            raise ConnectorError(f"NORMALIZED_CAPTURE_INVALID: {detail}")
        warnings = [i for i in issues if i.severity == "warning"]
        self._set_stage(state, "normalized_capture", StageState.complete,
                        "NORMALIZED_CAPTURE_VALID"
                        + (f" with {len(warnings)} warnings" if warnings else ""))

        # ---- geometry ---------------------------------------------------
        self._set_stage(state, "geometry", StageState.running)
        self._write(state)
        try:
            result = run_geometry(normalized, frame_stride=geometry_stride)
        except ValueError as exc:
            state["failureClass"] = FailureClass.geometry.value
            self._set_stage(state, "geometry", StageState.failed, str(exc))
            raise
        planes = result.planes
        if planes.ceiling is None:
            state["failureClass"] = FailureClass.geometry.value
            self._set_stage(
                state, "geometry", StageState.failed,
                f"{len(planes.walls)} wall candidates, floor="
                f"{'yes' if planes.floor else 'no'}, ceiling=no")
            raise ValueError(
                "GEOMETRY_GENERALIZATION_FAILURE: a floor was found but no "
                "ceiling, so room height cannot be measured")
        self._set_stage(state, "geometry", StageState.complete,
                        f"{len(planes.walls)} wall candidates, "
                        f"floor={'yes' if planes.floor else 'no'}, "
                        f"ceiling={'yes' if planes.ceiling else 'no'}")

        # ---- evidence, then the canonical model -------------------------
        views, evidence_diagnostics = select_evidence_views(
            capture_root=normalized, capture=result.capture, cloud=result.cloud,
            model=result.model, planes_by_id=result.planes_by_id,
            config=load_geometry_config(), output_dir=output)
        result.model["evidence"] = [v.to_record() for v in views]
        result.model["provenance"]["evidenceSelection"] = evidence_diagnostics

        self._set_stage(state, "canonical_model", StageState.running)
        self._write(state)
        problems = validate_model(result.model)
        if problems:
            state["failureClass"] = FailureClass.output.value
            self._set_stage(state, "canonical_model", StageState.failed,
                            "; ".join(problems[:3]))
            raise ValueError(f"SPATIAL_MODEL_INVALID: {problems[:3]}")
        self._set_stage(state, "canonical_model", StageState.complete,
                        f"{len(result.model['surfaces'])} surfaces, "
                        f"{len(result.model['measurements'])} measurements")

        # ---- opening resolver (crop of each geometry gap) ----------------
        opening_report = self._run_or_load_openings(
            scan_id, result.model, output,
            capture=result.capture, capture_root=normalized,
            cloud=result.cloud, planes_by_id=result.planes_by_id,
            config=load_geometry_config())
        result.model["provenance"]["openingResolution"] = opening_report["diagnostics"]
        result.model["provenance"]["openingResolution"]["promotedCount"] = (
            opening_report["promotedCount"])
        result.model["provenance"]["openingResolution"]["classified"] = (
            opening_report["classified"])

        # ---- renderers ---------------------------------------------------
        (output / "floorplan.svg").write_text(render_floorplan(result.model))
        self._set_stage(state, "floorplan", StageState.complete,
                        "generated from spatial_model.json")
        entity_map = write_model_3d(result.model, output)
        self._set_stage(state, "model_3d", StageState.complete,
                        f"{entity_map['surfaceCount']} named surfaces")

        # ---- benchmark ----------------------------------------------------
        self._set_stage(
            state, "benchmark", StageState.not_applicable,
            "No tape ground truth exists for a development fixture. The public "
            "reference comparison is a separate artifact and is not this benchmark.")

        # ---- AI review -----------------------------------------------------
        self._set_stage(state, "ai_review", StageState.running)
        self._write(state)
        verifier = self._run_or_load_verifier(
            scan_id, result.model, output / "evidence")
        result.model["aiAssessments"] = [verifier.assessment]
        result.model["provenance"]["aiReview"] = verifier.diagnostics
        (output / "ai_assessment.json").write_text(json.dumps(
            {"assessment": verifier.assessment,
             "rejectedFindings": verifier.rejected_findings,
             "diagnostics": verifier.diagnostics}, indent=2) + "\n")
        assessment = verifier.assessment
        opening_note = ""
        if opening_report["candidates"]:
            opening_note = (
                f", {opening_report['promotedCount']} of "
                f"{opening_report['candidates']} opening crops confirmed")
        self._set_stage(
            state, "ai_review",
            StageState.skipped if assessment["status"] == "not_run"
            else StageState.complete,
            (assessment.get("notRunReason") or f"{len(assessment['findings'])} findings")
            + opening_note)

        # ---- loss preview ---------------------------------------------------
        from pipeline.loss_preview.preview import build_loss_preview

        preview = build_loss_preview(result.model, output)
        self._set_stage(
            state, "loss_preview",
            StageState.complete if preview.get("status") != "not_applicable"
            else StageState.not_applicable,
            preview.get("statusReason", ""))

        # ---- persist ---------------------------------------------------------
        digest = write_model(result.model, output / "spatial_model.json")
        (output / "geometry_diagnostics.json").write_text(
            json.dumps(result.diagnostics, indent=2) + "\n")
        (output / "evidence_manifest.json").write_text(json.dumps({
            "scanId": result.model["scan"]["id"],
            "classification": result.model["scan"]["classification"],
            "views": [v.to_record() for v in views],
            "diagnostics": evidence_diagnostics,
        }, indent=2) + "\n")

        measurements = {m["type"]: m for m in result.model["measurements"]}
        state["summary"] = {
            "modelSha256": digest,
            "surfaceCount": len(result.model["surfaces"]),
            "evidenceViews": len(views),
            "roomLength_m": measurements.get("room_length", {}).get("value_m"),
            "roomWidth_m": measurements.get("room_width", {}).get("value_m"),
            "roomHeight_m": measurements.get("room_height", {}).get("value_m"),
            "floorArea_m2": measurements.get("floor_area", {}).get("value_m"),
            "observedPerimeterFraction":
                result.envelope.diagnostics["observedPerimeterFraction"],
            "aiStatus": assessment["status"],
            "lossPreviewStatus": preview.get("status"),
            "unresolvedOpenings": sum(
                1 for o in result.model["openings"]
                if o["observationState"] == "unresolved"),
            "confirmedOpenings": sum(
                1 for o in result.model["openings"]
                if o["observationState"] != "unresolved" and o.get("dimensions")),
        }
        self._set_stage(state, "complete", StageState.complete, "all stages finished")
        state["status"] = ScanStatus.complete.value
        state["currentStage"] = "complete"

    def rerun_ai_review(self, scan_id: str) -> dict:
        """Retry Groq review on the saved model. Geometry is not rebuilt."""
        lock = self._locks.setdefault(scan_id, threading.Lock())
        with lock:
            state = self._read(scan_id)
            output = self._dir(scan_id) / "output"
            model_path = output / "spatial_model.json"
            if not model_path.is_file():
                raise FileNotFoundError("this scan has no spatial model to review")
            model = json.loads(model_path.read_text())
            length_before = next(
                (m.get("value_m") for m in model.get("measurements") or []
                 if m.get("type") == "room_length"), None)
            self._set_stage(state, "ai_review", StageState.running)
            state["status"] = ScanStatus.processing.value
            self._write(state)
            try:
                self._hydrate_results(scan_id)
                verifier = self._run_or_load_verifier(
                    scan_id, model, output / "evidence")
                model["aiAssessments"] = [verifier.assessment]
                model.setdefault("provenance", {})["aiReview"] = verifier.diagnostics
                (output / "ai_assessment.json").write_text(json.dumps(
                    {"assessment": verifier.assessment,
                     "rejectedFindings": verifier.rejected_findings,
                     "diagnostics": verifier.diagnostics}, indent=2) + "\n")
                digest = write_model(model, model_path)
                assessment = verifier.assessment
                length_after = next(
                    (m.get("value_m") for m in model.get("measurements") or []
                     if m.get("type") == "room_length"), None)
                if length_before != length_after:
                    raise RuntimeError("AI review changed a geometry measurement")
                state.setdefault("summary", {})["modelSha256"] = digest
                state["summary"]["aiStatus"] = assessment["status"]
                self._set_stage(
                    state, "ai_review",
                    StageState.skipped if assessment["status"] == "not_run"
                    else StageState.complete,
                    assessment.get("notRunReason")
                    or f"{len(assessment['findings'])} findings")
                state["status"] = ScanStatus.complete.value
                state["currentStage"] = "complete"
            except Exception as exc:  # noqa: BLE001
                state["status"] = ScanStatus.complete.value
                self._set_stage(
                    state, "ai_review", StageState.skipped,
                    f"{type(exc).__name__}: {exc}")
                state.setdefault("summary", {})["aiStatus"] = "not_run"
            self._write(state)
            return state

    def import_ai_review(self, scan_id: str, raw: dict) -> dict:
        """Accept an offline Qwen JSON review. Geometry is not rebuilt or sent."""
        lock = self._locks.setdefault(scan_id, threading.Lock())
        with lock:
            state = self._read(scan_id)
            output = self._dir(scan_id) / "output"
            model_path = output / "spatial_model.json"
            if not model_path.is_file():
                raise FileNotFoundError("this scan has no spatial model to review")
            model = json.loads(model_path.read_text())
            length_before = next(
                (m.get("value_m") for m in model.get("measurements") or []
                 if m.get("type") == "room_length"), None)
            config = load_ai_config()
            verifier = validate_assessment(
                raw, model, config, "spatial_verifier_v0.1",
                {"importSource": "offline_qwen_paste", "approved": config.approved})
            if verifier.assessment.get("status") != "completed":
                raise ValueError(
                    verifier.assessment.get("notRunReason")
                    or "the pasted review was rejected")
            if raw.get("provider"):
                verifier.assessment["provider"] = raw["provider"]
            if raw.get("model"):
                verifier.assessment["model"] = raw["model"]
            verifier.diagnostics["importSource"] = "offline_qwen_paste"
            verifier.diagnostics.setdefault("cacheHit", False)
            model["aiAssessments"] = [verifier.assessment]
            model.setdefault("provenance", {})["aiReview"] = verifier.diagnostics
            (output / "ai_assessment.json").write_text(json.dumps(
                {"assessment": verifier.assessment,
                 "rejectedFindings": verifier.rejected_findings,
                 "diagnostics": verifier.diagnostics}, indent=2) + "\n")
            digest = write_model(model, model_path)
            length_after = next(
                (m.get("value_m") for m in model.get("measurements") or []
                 if m.get("type") == "room_length"), None)
            if length_before != length_after:
                raise RuntimeError("AI review changed a geometry measurement")
            key, _cfg = self._ai_key(model)
            self.db.put_ai(
                key, scan_id, verifier.assessment,
                verifier.rejected_findings, verifier.diagnostics)
            state.setdefault("summary", {})["modelSha256"] = digest
            state["summary"]["aiStatus"] = "completed"
            self._set_stage(
                state, "ai_review", StageState.complete,
                f"{len(verifier.assessment.get('findings') or [])} findings "
                "(offline Qwen import)")
            state["status"] = ScanStatus.complete.value
            state["currentStage"] = "complete"
            self._write(state)
            return state

    # -- artifacts ----------------------------------------------------------

    def artifacts(self, scan_id: str) -> list[ArtifactRecord]:
        output = self._dir(scan_id) / "output"
        state = self._read(scan_id)
        development = state["classification"] == "public_development_fixture"
        records: list[ArtifactRecord] = []
        for name, status, description in ARTIFACT_SPEC:
            path = output / name
            available = path.is_file()
            resolved = status
            if not available:
                resolved = "pending" if status == "pending" else "unavailable"
            elif status == "development-only" and not development:
                resolved = "final-verified"
            records.append(ArtifactRecord(
                name=name, available=available, status=resolved,
                path=str(path) if available else None,
                bytes=path.stat().st_size if available else None,
                description=description))
        return records

    def artifact_path(self, scan_id: str, name: str) -> Path:
        allowed = {spec[0] for spec in ARTIFACT_SPEC}
        if name not in allowed:
            raise KeyError(name)
        path = self._dir(scan_id) / "output" / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def evidence_path(self, scan_id: str, name: str) -> Path:
        if "/" in name or ".." in name:
            raise ValueError("invalid evidence name")
        candidates = [
            self._dir(scan_id) / "output" / "evidence" / name,
            EVIDENCE_DIR / name,
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(name)

    def model(self, scan_id: str) -> dict[str, Any]:
        path = self._dir(scan_id) / "output" / "spatial_model.json"
        if not path.is_file():
            raise FileNotFoundError("spatial_model.json")
        return json.loads(path.read_text())
