"""Execute the validation ledger and write its evidence.

The ledger's own rule is that a pipeline output is not a validation result
unless an independent expected value exists. This runner therefore does two
different jobs and never confuses them: it *executes* what can be executed, and
it *records why* everything else cannot be. A dataset that is not on this
machine produces a `BLOCKED` row with the reason, never an absent row and never
an invented number.

Each run emits a record in the template of ledger section 8, so a reader can
tell what was frozen before what was read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOT = REPO_ROOT / "validation"

NOT_STARTED = "NOT STARTED"
PASS = "PASS"
FAIL = "FAIL"
PARTIAL = "PARTIAL"
NOT_COMPARABLE = "NOT COMPARABLE"
BLOCKED = "BLOCKED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(args, cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "note": "This repository is not committed; hashes below identify the "
                "configuration and modules instead.",
    }


def frozen_state() -> dict[str, Any]:
    manifest_path = REPO_ROOT / "output" / "config_freeze_manifest.json"
    if not manifest_path.is_file():
        return {"state": "NO_FREEZE_MANIFEST"}
    manifest = json.loads(manifest_path.read_text())
    configs = manifest.get("configurations", {})
    geometry = configs.get("config/geometry_config_v0.1.json")
    return {
        "state": manifest.get("state"),
        "geometryConfigHash": geometry if isinstance(geometry, str)
                              else (geometry or {}).get("sha256"),
        "allChecksPassed": manifest.get("allChecksPassed"),
        "frozenAt": manifest.get("frozenAt"),
    }


@dataclass
class RunRecord:
    """Ledger section 8, as data."""

    validation_id: str
    dataset: str
    dataset_role: str
    source_modality: str
    scene_id: str | None = None
    held_out: str = "no"
    reference_source: str = "none attached"
    reference_extraction_version: str | None = None
    prediction_freeze_artifact: str | None = None
    reference_loaded_after_prediction_freeze: str = "n/a"
    metrics_evaluated: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    verdict: str = NOT_STARTED
    known_ambiguity: list[str] = field(default_factory=list)
    files_generated: list[str] = field(default_factory=list)
    parameter_changes_after_run: str = "none"
    conclusion: str = ""
    date: str = field(default_factory=utc_now)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["pipeline"] = {"git": git_state(), "frozen": frozen_state(),
                             "python": platform.python_version()}
        return record


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_structure() -> None:
    for relative in ("manifests", "references/public", "references/final_tape",
                     "predictions/public", "predictions/final", "results", "reports"):
        (VALIDATION_ROOT / relative).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# V01 - synthetic geometry regression (exact ground truth)
# --------------------------------------------------------------------------

def run_v01() -> RunRecord:
    """Exact-GT regression. Establishes implementation correctness only."""
    record = RunRecord(
        validation_id="V01",
        dataset="Locally authored synthetic room fixtures",
        dataset_role="Exact implementation correctness",
        source_modality="synthetic depth + exact poses",
        held_out="n/a (ground truth known before execution)",
        reference_source="exact, authored before the run",
        metrics_evaluated=["room_length", "room_width", "room_height",
                           "floor_area", "wall plane positions", "yaw invariance"],
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "pipeline/tests/test_geometry_model.py",
         "pipeline/tests/test_geometry_planes.py",
         "pipeline/tests/test_geometry_frame.py",
         "pipeline/tests/test_frame_resolution.py"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    record.results = {"pytestSummary": tail, "returnCode": completed.returncode}
    record.verdict = PASS if completed.returncode == 0 else FAIL
    record.conclusion = (
        "Regression only. Exact synthetic ground truth evidences that the "
        "implementation recovers what it was given; it evidences nothing about "
        "real-world accuracy or generalization.")
    return record


# --------------------------------------------------------------------------
# V02 - ARKitScenes against FARO laser reference
# --------------------------------------------------------------------------

def run_v02(models: dict[str, Path], visit_dir: Path) -> RunRecord:
    """Height only, and only as a development reference.

    ARKitScenes publishes no transform between the ARKit session frame and the
    FARO visit frame. Floor-to-ceiling height survives that gap because both
    frames are gravity-aligned; every horizontal quantity does not, and is
    reported NOT COMPARABLE rather than registered by guesswork.
    """
    record = RunRecord(
        validation_id="V02",
        dataset="ARKitScenes",
        dataset_role="Public Apple mobile RGB-D geometry validation",
        source_modality="mobile RGB-D + intrinsics + trajectory",
        scene_id="47333462",
        held_out="no - 47333462 is the primary tuning scene",
        reference_source="ARKitScenes FARO laser_scanner_point_clouds visit 467138",
        reference_extraction_version="reference_extraction_v0.1",
    )

    primary = models.get("47333462")
    if primary is None or not primary.is_file():
        record.verdict = BLOCKED
        record.conclusion = "No spatial_model.json for 47333462; nothing to compare."
        return record

    record.prediction_freeze_artifact = _relative(primary)
    record.reference_loaded_after_prediction_freeze = (
        "yes - the model is written and hashed before the FARO clouds are read")

    output_dir = VALIDATION_ROOT / "results" / "v02_arkitscenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pipeline.benchmark.run_reference",
               str(primary), str(visit_dir), str(output_dir),
               "--scene-id", "47333462", "--scene-role", "PRIMARY_TUNING"]
    secondary = [f"{scene}={path}" for scene, path in models.items()
                 if scene != "47333462" and path.is_file()]
    if secondary:
        command += ["--secondary", *secondary]

    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        record.verdict = BLOCKED
        record.results = {"stderr": completed.stderr.strip()[-800:]}
        record.conclusion = "The reference comparison did not run."
        return record

    report = output_dir / "development_reference_benchmark.json"
    if not report.is_file():
        record.verdict = BLOCKED
        record.conclusion = "The reference comparison produced no report."
        return record

    payload = json.loads(report.read_text())
    comparisons = payload.get("comparisons", payload if isinstance(payload, list) else [])
    if isinstance(comparisons, dict):
        comparisons = comparisons.get("comparisons", [])

    rows = {}
    for item in comparisons:
        rows[item["measurementId"]] = {
            "type": item.get("type"),
            "model_m": item.get("model_m"),
            "reference_m": item.get("reference_m"),
            "absoluteError_cm": item.get("absoluteError_cm"),
            "assignmentGate": item.get("assignmentGate"),
            "result": item.get("result"),
        }
    record.metrics_evaluated = sorted({r["type"] for r in rows.values() if r["type"]})
    record.results = rows
    record.files_generated = [_relative(p) for p in sorted(output_dir.iterdir())]

    height = next((r for r in rows.values() if r["type"] == "room_height"), None)
    not_comparable = [r for r in rows.values() if r["result"] == "not_comparable"]

    correspondence = payload.get("referenceCorrespondence") or {}
    pairings = [p.get("separation_m") for p in correspondence.get("plausiblePairings", [])
                if p.get("separation_m") is not None]
    record.results["referenceCorrespondence"] = {
        "established": correspondence.get("established"),
        "plausibleSeparationRange_m": ([min(pairings), max(pairings)] if pairings else None),
        "plausiblePairingCount": len(pairings) or None,
    }

    record.verdict = PARTIAL
    record.known_ambiguity = [
        "Reference correspondence between the ARKit session frame and the FARO "
        "visit frame is NOT ESTABLISHED. Only gravity-invariant height is compared.",
        "47333462 is the primary tuning scene, so this is a development reference "
        "and not held-out evidence.",
    ]
    if pairings and height and height.get("model_m") is not None:
        low, high = min(pairings), max(pairings)
        if low <= height["model_m"] <= high:
            record.known_ambiguity.append(
                f"The reference itself is not a single number. {len(pairings)} floor and "
                f"ceiling pairings in the FARO cloud are plausible, separated by "
                f"{low:.3f} m to {high:.3f} m, and the model's {height['model_m']} m falls "
                f"inside that range. The recorded error is against one chosen pairing, so "
                f"it is a bound on disagreement rather than a measured error.")

    # Cross-scene execution, which is generalization evidence and not accuracy.
    for scene, path in models.items():
        if scene == "47333462" or not path.is_file():
            continue
        other = json.loads(path.read_text())
        record.results.setdefault("crossSceneExecution", {})[scene] = {
            "surfaces": len(other.get("surfaces", [])),
            "measurements_m": {m["type"]: m.get("value_m", m.get("value"))
                               for m in other.get("measurements", [])},
            "referenceAvailable": False,
            "note": "The frozen configuration executes on a second scene. No laser "
                    "reference is published locally for it, so this is evidence "
                    "against overfitting, not evidence of accuracy.",
        }
    record.conclusion = (
        f"Development reference only - reference correspondence not established. "
        f"Height {height['model_m']} m against {height['reference_m']} m reference, "
        f"error {height['absoluteError_cm']} cm against a {height['assignmentGate']} "
        f"gate: {height['result']}. "
        f"{len(not_comparable)} room-dimension metrics remain NOT COMPARABLE and no "
        f"room accuracy percentage is derived from unmatched FARO data."
    ) if height else "No height comparison was produced."
    return record


# --------------------------------------------------------------------------
# V06 - public Stray sample, interoperability only
# --------------------------------------------------------------------------

def run_v06(model_path: Path, capture_dir: Path) -> RunRecord:
    """Interoperability. No reference is attached, so no accuracy is claimed."""
    record = RunRecord(
        validation_id="V06",
        dataset="Public Stray sample 8653a2142b",
        dataset_role="Real Stray-format interoperability and reconstruction",
        source_modality="Stray Scanner export (depth, confidence, rgb.mp4, odometry)",
        scene_id="8653a2142b",
        held_out="n/a",
        reference_source="none found / none attached",
        reference_loaded_after_prediction_freeze="n/a - no reference exists",
        metrics_evaluated=["reconstruction completeness", "frame resolution outcome"],
    )
    if not model_path.is_file():
        record.verdict = FAIL
        record.conclusion = ("The capture produced no spatial_model.json, so Stray "
                             "interoperability is not demonstrated.")
        return record

    record.prediction_freeze_artifact = _relative(model_path)
    model = json.loads(model_path.read_text())
    measurements = {m["type"]: m.get("value_m", m.get("value"))
                    for m in model.get("measurements", [])}
    surfaces = model.get("surfaces", [])
    kinds: dict[str, int] = {}
    for surface in surfaces:
        kinds[surface["type"]] = kinds.get(surface["type"], 0) + 1

    resolution_path = capture_dir / "frame_resolution.json"
    resolution = json.loads(resolution_path.read_text()) if resolution_path.is_file() else {}

    record.results = {
        "surfaces": len(surfaces),
        "surfacesByType": kinds,
        "inferredSurfaces": sum(1 for s in surfaces
                                if s.get("observationState") == "inferred"),
        "measurements_m": measurements,
        "frameResolution": {
            "outcome": resolution.get("outcome"),
            "axis": resolution.get("axis"),
            "basis": resolution.get("basis"),
        },
        "modelSha256": sha256_file(model_path),
    }
    has_room = kinds.get("floor", 0) >= 1 and kinds.get("ceiling", 0) >= 1
    record.verdict = PASS if has_room else FAIL
    record.known_ambiguity = [
        "No independent measurement exists for this capture, so every dimension "
        "above is unvalidated output, not a validated quantity.",
    ]
    record.conclusion = (
        "PASS for interoperability: a real Stray export parses, normalizes, resolves "
        "its vertical axis, and reconstructs a closed room. NOT COMPARABLE for "
        "accuracy, because no independent reference is attached. "
        "The dimensions recorded here must never be quoted as accuracy evidence.")
    return record


# --------------------------------------------------------------------------
# Everything the ledger asks for that this machine cannot supply
# --------------------------------------------------------------------------

BLOCKED_ROWS = [
    ("V03", "Redwood Indoor LiDAR-RGBD", "High-end laser reference",
     "The dataset is not present on this machine."),
    ("V04", "TUM RGB-D", "Trajectory ground truth",
     "The dataset is not present on this machine. This is the validation that would "
     "test pose and transform handling directly, which is where the repaired frame "
     "defect lived."),
    ("V05", "ScanNet", "Reconstruction reference",
     "Optional, and the ledger says not to spend final-submission time here before "
     "V07 exists."),
    ("V07", "Final self-captured iPhone Stray + tape", "Independent tape measurements",
     "No self-captured iPhone export and no tape measurements exist yet. This is the "
     "only end-to-end Track A accuracy benchmark, and it is P0."),
]


def run_v02b() -> RunRecord:
    """CA-1M held-out validation against laser-registered ground truth."""
    record = RunRecord(
        validation_id="V02B",
        dataset="Apple CA-1M / Cubify Anything (val split, held out)",
        dataset_role="Public mobile RGB-D geometry accuracy in registered laser space",
        source_modality="ARKitScenes mobile RGB-D and odometry; CA-1M laser reference",
        held_out="yes - ARKitScenes Validation split; the configuration was frozen "
                 "against Training scene 47333462",
        reference_source="CA-1M FARO-rendered depth and laser-registered poses",
    )
    results = REPO_ROOT / "validation" / "results" / "ca1m" / "ca1m_multiscene.json"
    manifest = REPO_ROOT / "validation" / "manifests" / "ca1m_heldout_manifest.json"
    if not results.is_file():
        record.verdict = NOT_STARTED
        record.conclusion = ("No CA-1M results exist. Run "
                             "python3 -m pipeline.validation.ca1m_benchmark.")
        return record

    payload = json.loads(results.read_text())
    aggregate = payload["aggregate"]
    record.prediction_freeze_artifact = _relative(results)
    record.reference_loaded_after_prediction_freeze = (
        "yes - each prediction is hashed before any gt/ asset is opened, and a test "
        "enforces that ordering")
    record.metrics_evaluated = ["room_height", "floor/ceiling/wall surface distance",
                                "coverage", "alignment residual"]
    record.results = {
        "capturesEvaluated": aggregate["capturesEvaluated"],
        "roomHeightAbsoluteError_cm": aggregate["roomHeightAbsoluteError_cm"],
        "roomHeightSignedError_cm": aggregate["roomHeightSignedError_cm"],
        "roomHeightGate": aggregate["roomHeightGate"],
        "surfaceMedianDistance_cm_byType": aggregate["surfaceMedianDistance_cm_byType"],
        "failures": aggregate["failures"],
    }
    record.files_generated = [_relative(results), _relative(manifest)]
    record.known_ambiguity = [
        "CA-1M publishes no mobile camera pose, so the trajectory comes from "
        "ARKitScenes Validation for the same capture. CA-1M supplies ground truth only.",
        "Surface distances need the ARKit-to-laser transform, which holds on only 1 of "
        "5 captures because CA-1M orients frames upright and the camera roll can change "
        "mid-capture. They are withheld on the rest rather than reported.",
        "Room height is measured inside the laser frame and compared with an internal "
        "model scalar, so it carries none of that alignment error.",
    ]
    gate = aggregate["roomHeightGate"]
    signed = aggregate.get("roomHeightSignedError_cm") or {}
    record.verdict = PARTIAL
    record.conclusion = (
        f"Room height validated on {aggregate['capturesEvaluated']} held-out captures "
        f"against laser-derived reference: median "
        f"{aggregate['roomHeightAbsoluteError_cm']['median']} cm, worst "
        f"{aggregate['roomHeightAbsoluteError_cm']['worst']} cm, "
        f"{gate['passed']} of {gate['passed'] + gate['failed']} within the 1.5 cm gate. "
        + ("The signed error is the same sign on every capture (median "
           f"{signed.get('median')} cm), which is a systematic under-measurement rather "
           "than scatter, and it agrees in sign with the ARKitScenes FARO result. "
           if signed.get("allSameSign") else "")
        + "Surface distances remain NOT COMPARABLE on most captures. No geometry "
          "parameter was changed in response to any of it.")
    return record


def blocked_records() -> list[RunRecord]:
    records = []
    for validation_id, dataset, reference, reason in BLOCKED_ROWS:
        records.append(RunRecord(
            validation_id=validation_id,
            dataset=dataset,
            dataset_role="see ledger section 3",
            source_modality="not available",
            reference_source=reference,
            verdict=BLOCKED,
            conclusion=reason,
        ))
    return records


# --------------------------------------------------------------------------
# A01 / A03 - AI evaluation against labelled fixtures
# --------------------------------------------------------------------------

def run_a01() -> RunRecord:
    """Semantic classification and binding against the labelled opening fixture."""
    record = RunRecord(
        validation_id="A01",
        dataset="Synthetic opening semantic fixture",
        dataset_role="AI classification + grounding",
        source_modality="rendered frames with authored labels",
        reference_source="samples/ai_semantic_eval/manifest.json labels",
        metrics_evaluated=["semantic class accuracy", "surface binding correctness"],
    )
    report = REPO_ROOT / "output" / "ai_semantic_benchmark.json"
    if not report.is_file():
        record.verdict = NOT_STARTED
        record.conclusion = ("No semantic benchmark output exists. Regenerate with "
                             "python3 -m pipeline.ai.semantic_benchmark.")
        return record

    payload = json.loads(report.read_text())
    record.prediction_freeze_artifact = _relative(report)
    record.results = {
        key: payload[key] for key in
        ("model", "provider", "generatedUtc", "cases", "summary", "results")
        if key in payload
    }
    record.known_ambiguity = [
        "The fixture is rendered, not photographed, so this scores the model on "
        "synthetic imagery only.",
        payload.get("interpretationRule", ""),
    ]
    record.verdict = PARTIAL
    record.conclusion = payload.get("conclusion", "")[:600]
    return record


def run_a03() -> RunRecord:
    """Condition grounding: does an AI region bind to the right surface."""
    record = RunRecord(
        validation_id="A03",
        dataset="Development condition fixture",
        dataset_role="AI to spatial registration",
        source_modality="development fixture with an overlaid synthetic stain",
        reference_source="samples/ai_condition_eval/labels.json",
        metrics_evaluated=["condition class", "bound surface id", "registration state"],
    )
    labels_path = REPO_ROOT / "samples" / "ai_condition_eval" / "labels.json"
    produced = REPO_ROOT / "output" / "visible_condition.json"
    if not labels_path.is_file() or not produced.is_file():
        record.verdict = NOT_STARTED
        record.conclusion = "The condition fixture or its produced proposal is missing."
        return record

    labels = json.loads(labels_path.read_text())
    proposal = json.loads(produced.read_text())
    record.prediction_freeze_artifact = _relative(produced)
    body = proposal.get("proposal") or {}
    registration = body.get("registration") or {}
    expected_surface = labels.get("surfaceId") or labels.get("expectedSurfaceId")
    bound_surface = body.get("surfaceId")

    record.results = {
        "expectedSurfaceId": expected_surface,
        "boundSurfaceId": bound_surface,
        "bindingCorrect": (expected_surface == bound_surface
                           if expected_surface and bound_surface else None),
        "registrationStatus": registration.get("status"),
        "quantityProducer": registration.get("producer"),
        "isRealDamageEvidence": proposal.get("isRealDamageEvidence"),
    }
    record.known_ambiguity = [
        "The stain is overlaid, not real damage, so this evaluates grounding "
        "mechanics and not condition-detection accuracy.",
    ]
    record.verdict = PARTIAL
    record.conclusion = (
        "Grounding mechanics only: the check is that an AI region binds to the "
        "correct surface and that any physical quantity stays geometry-produced. "
        "It is not evidence that the model detects real damage.")
    return record


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

DASHBOARD_ORDER = ["V01", "V02", "V02B", "V03", "V04", "V05", "V06", "V07",
                   "A01", "A02", "A03"]


def _first_sentence(text: str, limit: int = 150) -> str:
    """One whole sentence, or a clean ellipsis. Never a word cut in half."""
    if not text:
        return ""
    sentence = text.split(". ")[0].rstrip(".")
    if len(sentence) <= limit:
        return sentence
    clipped = sentence[:limit].rsplit(" ", 1)[0]
    return clipped + "..."


def write_summary(records: list[RunRecord]) -> Path:
    by_id = {r.validation_id: r for r in records}
    lines = [
        "# Validation summary",
        "",
        f"Generated {utc_now()}.",
        "",
        "Produced by `python3 -m pipeline.validation.run_ledger`. Every row is either "
        "an executed run or a recorded reason it could not run. No row is an estimate.",
        "",
        "A pipeline output is not a validation result unless an independent expected "
        "value exists. Rows marked `NOT COMPARABLE` have output but no admissible "
        "reference, and their numbers must not be quoted as accuracy.",
        "",
        "| ID | Dataset | Independent reference | Verdict | What it establishes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for validation_id in DASHBOARD_ORDER:
        record = by_id.get(validation_id)
        if record is None:
            continue
        establishes = _first_sentence(record.conclusion)
        lines.append(
            f"| {record.validation_id} | {record.dataset} | {record.reference_source} "
            f"| `{record.verdict}` | {establishes} |")

    blocked = [r for r in records if r.verdict == BLOCKED]
    lines += ["", "## What is blocked, and why it matters", ""]
    for record in blocked:
        lines.append(f"- **{record.validation_id} — {record.dataset}.** {record.conclusion}")

    lines += [
        "",
        "## Claim boundary",
        "",
        "- No room-dimension accuracy is validated on any real capture. The only "
        "independent reference attached anywhere is the ARKitScenes FARO height on "
        "the primary tuning scene, whose mobile-to-reference correspondence is not "
        "established, and which fails its gate.",
        "- The Stray sample evidences interoperability only. It has no reference.",
        "- The end-to-end accuracy benchmark (V07) does not exist yet, so no "
        "end-to-end accuracy claim is available to make.",
        "",
    ]
    path = VALIDATION_ROOT / "reports" / "validation_summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arkitscenes-model", type=Path,
                        help="spatial_model.json for scene 47333462")
    parser.add_argument("--arkitscenes-secondary", type=Path,
                        help="spatial_model.json for scene 41418135")
    parser.add_argument("--stray-model", type=Path,
                        help="spatial_model.json for the Stray sample")
    parser.add_argument("--stray-capture", type=Path,
                        help="normalized capture directory for the Stray sample")
    parser.add_argument("--visit-dir", type=Path,
                        default=REPO_ROOT / "samples" / "arkitscenes"
                                / "laser_scanner_point_clouds" / "467138")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="validation ids to skip")
    args = parser.parse_args(argv)

    ensure_structure()
    records: list[RunRecord] = []
    skip = set(args.skip)

    if "V01" not in skip:
        records.append(run_v01())

    if "V02" not in skip:
        models = {}
        if args.arkitscenes_model:
            models["47333462"] = args.arkitscenes_model
        if args.arkitscenes_secondary:
            models["41418135"] = args.arkitscenes_secondary
        if models and args.visit_dir.is_dir():
            records.append(run_v02(models, args.visit_dir))
        else:
            record = RunRecord("V02", "ARKitScenes", "Public geometry validation",
                               "mobile RGB-D", scene_id="47333462",
                               reference_source="ARKitScenes FARO visit 467138")
            record.verdict = BLOCKED
            record.conclusion = ("No reconstructed model was supplied, or the FARO "
                                 "visit directory is absent.")
            records.append(record)

    if "V02B" not in skip:
        records.append(run_v02b())

    if "V06" not in skip:
        if args.stray_model and args.stray_capture:
            records.append(run_v06(args.stray_model, args.stray_capture))
        else:
            record = RunRecord("V06", "Public Stray sample 8653a2142b",
                               "Interoperability", "Stray export")
            record.verdict = BLOCKED
            record.conclusion = "No reconstructed Stray model was supplied."
            records.append(record)

    if "A01" not in skip:
        records.append(run_a01())
    if "A03" not in skip:
        records.append(run_a03())

    a02 = RunRecord("A02", "ARKitScenes live AI review",
                    "Real-image AI execution", "real RGB frames",
                    reference_source="none - the frames carry no semantic labels")
    a02.verdict = NOT_COMPARABLE
    a02.conclusion = ("The model runs on real frames and produces findings, but no "
                      "labels exist for them, so nothing is scored. Execution is not "
                      "accuracy.")
    records.append(a02)

    records.extend(blocked_records())

    results_path = VALIDATION_ROOT / "results" / "validation_runs.json"
    results_path.write_text(json.dumps(
        {"generatedUtc": utc_now(),
         "runs": [r.to_record() for r in records]}, indent=2) + "\n")
    summary_path = write_summary(records)

    print(f"validation runs -> {_relative(results_path)}")
    print(f"summary         -> {_relative(summary_path)}\n")
    width = max(len(r.verdict) for r in records)
    for validation_id in DASHBOARD_ORDER:
        record = next((r for r in records if r.validation_id == validation_id), None)
        if record:
            print(f"  {record.validation_id:5s} {record.verdict:{width}s}  {record.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
