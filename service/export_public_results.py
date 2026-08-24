"""Build a privacy-safe public results catalog from local working state.

The working database and private capture stay on this machine. This module
writes `samples/public_results.db` plus a few reviewed evidence JPEGs.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DB = REPO_ROOT / "outputs" / "scans" / "_results.db"
DEFAULT_OUTPUT_DB = REPO_ROOT / "samples" / "public_results.db"
PUBLIC_RESULTS_DIR = REPO_ROOT / "samples" / "public_results"
EVIDENCE_DIR = REPO_ROOT / "samples" / "evidence"

# Stable public IDs. The privacy test allow-list must match this set.
PUBLIC_STRAY_ID = "public-stray-8653a2142b"
PUBLIC_IPHONE_ID = "public-iphone-e30fe3cae4"
ALLOWLISTED_SCAN_IDS = (PUBLIC_STRAY_ID, PUBLIC_IPHONE_ID)

LOCAL_STRAY_SCAN = "scan-30fb4864ea40"
LOCAL_IPHONE_SCAN = "scan-050cb25df8d1"

# Tight crops on the three registered iPhone frames after visual inspection.
# Pixel boxes are (left, top, right, bottom) on the 1920×1440 source PNG.
# Each crop is meant to keep wall / opening structure and drop bed, keyboard,
# bicycle, screens, and other personal possessions.
IPHONE_EVIDENCE = (
    {
        "source": "evidence-001.png",
        "evidence_id": "frame-000128",
        "filename": "evidence_wall_01.jpg",
        "crop": (220, 0, 980, 540),
        "visible": ["ceiling-001", "wall-001", "wall-006", "wall-007", "wall-008"],
    },
    {
        "source": "evidence-003.png",
        "evidence_id": "frame-000881",
        "filename": "evidence_wall_02.jpg",
        "crop": (640, 0, 1180, 520),
        "visible": ["wall-002", "wall-003", "wall-004", "wall-005"],
    },
)

LONG_EDGE_PX = 1280
JPEG_QUALITY = 80

FORBIDDEN = re.compile(
    r"(/Users/|/home/[a-z]|saiteja|GROQ_API_KEY|sk-[A-Za-z0-9]{8,}"
    r"|iphone-actual|rgb\.mp4|\.mov\b|odometry\.csv|confidence/"
    r"|/depth/|outputs/scans|_incoming)",
    re.IGNORECASE,
)

PUBLIC_SCHEMA = """
CREATE TABLE scans (
  scan_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  classification TEXT NOT NULL,
  source_type TEXT,
  source_path TEXT,
  status TEXT NOT NULL,
  created_at TEXT,
  updated_at TEXT,
  summary_json TEXT NOT NULL,
  state_json TEXT NOT NULL
);
CREATE TABLE ai_reviews (
  scan_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  model_id TEXT,
  prompt_version TEXT,
  assessment_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE artifacts (
  scan_id TEXT NOT NULL,
  name TEXT NOT NULL,
  rel_path TEXT NOT NULL,
  status TEXT NOT NULL,
  description TEXT,
  PRIMARY KEY (scan_id, name)
);
CREATE TABLE evidence (
  scan_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL,
  rel_path TEXT NOT NULL,
  PRIMARY KEY (scan_id, evidence_id)
);
CREATE TABLE validation_records (
  record_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  capture_id TEXT,
  payload_json TEXT NOT NULL
);
"""

ARTIFACT_FILES = (
    ("spatial_model.json", "final-verified", "Canonical insurance-facing model"),
    ("floorplan.svg", "final-verified", "Dimensioned 2D plan, generated from the model"),
    ("room_model.obj", "final-verified", "Semantic 3D surfaces with stable IDs"),
    ("room_model.mtl", "final-verified", "Materials encoding observation state"),
    ("room_model_entity_map.json", "final-verified", "OBJ group to surface ID map"),
    ("ai_assessment.json", "experimental", "Spatial AI Verifier assessment"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def sanitize(value: Any) -> Any:
    """Drop secrets and absolute/private paths. Do not invent replacements."""
    if isinstance(value, str):
        if FORBIDDEN.search(value):
            return "redacted"
        return value
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    return value


def _redact_openings(model: dict) -> None:
    for opening in model.get("openings") or []:
        provenance = opening.setdefault("provenance", {})
        ai = provenance.get("aiResolution") or {}
        ai.pop("cropPath", None)
        if "crop" in (ai.get("diagnostics") or {}):
            ai["diagnostics"].pop("crop", None)


def _empty_business_arrays(model: dict) -> None:
    model["damage"] = []
    model["scope"] = []


def write_jpeg(source: Path, dest: Path, crop: tuple[int, int, int, int]) -> None:
    image = Image.open(source).convert("RGB")
    image = image.crop(crop)
    width, height = image.size
    long_edge = max(width, height)
    if long_edge > LONG_EDGE_PX:
        scale = LONG_EDGE_PX / long_edge
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="JPEG", quality=JPEG_QUALITY, optimize=True,
               exif=b"", icc_profile=None)


def _copy_text_artifact(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".json"}:
        dest.write_text(json.dumps(sanitize(_load(src)), indent=2) + "\n")
    else:
        text = src.read_text()
        if FORBIDDEN.search(text):
            raise ValueError(f"refusing to publish '{src}' — private path leaked")
        dest.write_text(text)


def _iphone_model(src_model: dict) -> dict:
    model = sanitize(json.loads(json.dumps(src_model)))
    _empty_business_arrays(model)
    _redact_openings(model)
    by_id = {item["filename"]: item for item in IPHONE_EVIDENCE}
    views = []
    for view in model.get("evidence") or []:
        name = Path(view.get("path") or "").name
        spec = next((item for item in IPHONE_EVIDENCE
                     if item["source"] == name or item["evidence_id"] == view.get("id")),
                    None)
        if spec is None:
            continue
        view = dict(view)
        view["id"] = spec["evidence_id"]
        view["path"] = f"samples/evidence/{spec['filename']}"
        view["visibleSurfaceIds"] = list(spec["visible"])
        views.append(view)
        by_id.pop(spec["filename"], None)
    model["evidence"] = views
    scan = dict(model.get("scan") or {})
    scan["id"] = PUBLIC_IPHONE_ID
    scan["classification"] = "public_review_example"
    model["scan"] = scan
    return model


def _stray_model(src_model: dict) -> dict:
    model = sanitize(json.loads(json.dumps(src_model)))
    _empty_business_arrays(model)
    _redact_openings(model)
    model["evidence"] = []
    scan = dict(model.get("scan") or {})
    scan["id"] = PUBLIC_STRAY_ID
    scan["classification"] = "public_sample"
    model["scan"] = scan
    assessments = []
    for assessment in model.get("aiAssessments") or []:
        item = dict(assessment)
        item["findings"] = []
        assessments.append(item)
    model["aiAssessments"] = assessments
    return model


def _state(*, scan_id: str, name: str, classification: str, source_type: str,
           source_path: str | None, src_state: dict, summary: dict) -> dict:
    stages = []
    for stage in src_state.get("stages") or []:
        if stage.get("name") in {"benchmark", "loss_preview"}:
            stages.append({
                **{key: stage.get(key) for key in
                   ("name", "state", "detail", "startedAt", "finishedAt")},
                "detail": "Not published in the public catalog.",
            })
        else:
            stages.append({key: stage.get(key) for key in
                           ("name", "state", "detail", "startedAt", "finishedAt")})
    return {
        "scanId": scan_id,
        "status": "complete",
        "label": name,
        "classification": classification,
        "sourcePath": source_path,
        "sourceType": source_type,
        "connector": src_state.get("connector"),
        "stages": stages,
        "currentStage": "complete",
        "failureClass": None,
        "error": None,
        "createdAt": src_state.get("createdAt"),
        "updatedAt": src_state.get("updatedAt") or _now(),
        "summary": sanitize(summary),
        "publishedExample": True,
    }


def _ca1m_records() -> list[dict]:
    multi = _load(REPO_ROOT / "validation/results/ca1m/ca1m_multiscene.json")
    gate = _load(REPO_ROOT / "validation/results/ca1m/gate_45261179.json")
    records = [{
        "record_id": "ca1m-heldout-aggregate",
        "kind": "ca1m_heldout_aggregate",
        "capture_id": None,
        "payload_json": json.dumps({
            "manifestId": multi["manifestId"],
            "geometryConfigHash": multi["geometryConfigHash"],
            "heldOut": True,
            "claimBoundary": multi["claimBoundary"],
            "aggregate": {
                "capturesEvaluated": multi["aggregate"]["capturesEvaluated"],
                "roomHeightAbsoluteError_cm": multi["aggregate"]["roomHeightAbsoluteError_cm"],
                "roomHeightSignedError_cm": {
                    key: multi["aggregate"]["roomHeightSignedError_cm"][key]
                    for key in ("n", "mean", "median", "p90", "worst", "allSameSign")
                },
                "roomHeightGate": multi["aggregate"]["roomHeightGate"],
            },
        }),
    }]
    for capture in multi["captures"]:
        error = capture["roomHeightError"]
        records.append({
            "record_id": f"ca1m-heldout-{capture['captureId']}",
            "kind": "ca1m_heldout_capture",
            "capture_id": capture["captureId"],
            "payload_json": json.dumps({
                "captureId": capture["captureId"],
                "heldOut": True,
                "geometryConfigHash": capture["geometryConfigHash"],
                "modelHeight_m": error["model_m"],
                "laserReferenceHeight_m": error["laserReference_m"],
                "absoluteError_cm": error["absoluteError_cm"],
                "signedError_cm": error["signedError_cm"],
                "gate": error["assignmentGate"],
                "gateResult": error["result"],
                "surfaceDistancesComparable": capture["surfaceDistancesComparable"],
            }),
        })
    error = gate["roomHeightError"]
    model_m = error["model_m"]
    laser_m = error["laserReference_m"]
    records.append({
        "record_id": "ca1m-acceptance-45261179",
        "kind": "ca1m_one_sample_acceptance",
        "capture_id": gate["captureId"],
        "payload_json": json.dumps({
            "captureId": gate["captureId"],
            "heldOut": True,
            "geometryConfigHash": gate["geometryConfigHash"],
            "modelHeight_m": model_m,
            "laserReferenceHeight_m": laser_m,
            "absoluteError_cm": error["absoluteError_cm"],
            "signedError_cm": round((model_m - laser_m) * 100.0, 3),
            "gate": error["assignmentGate"],
            "gateResult": error["result"],
            "note": "One-sample acceptance gate. Not a tape-measure result.",
        }),
    })
    return records


def _write_scan_bundle(scan_id: str, state: dict, model: dict, src_output: Path,
                       assessment: dict | None, include_evidence: bool) -> None:
    dest = PUBLIC_RESULTS_DIR / scan_id
    output = dest / "output"
    output.mkdir(parents=True, exist_ok=True)
    (dest / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    (output / "spatial_model.json").write_text(json.dumps(model, indent=2) + "\n")
    for name, _status, _desc in ARTIFACT_FILES:
        if name in {"spatial_model.json", "ai_assessment.json"}:
            continue
        src = src_output / name
        if src.is_file():
            _copy_text_artifact(src, output / name)
    if assessment is not None:
        (output / "ai_assessment.json").write_text(json.dumps({
            "assessment": assessment,
            "rejectedFindings": [],
            "diagnostics": {"importSource": "public_catalog"},
        }, indent=2) + "\n")


def export_public_results(source_db: Path, output_db: Path) -> dict:
    """Rebuild the public catalog. Existing output is replaced."""
    del source_db  # Working DB is not copied; scans are read from allow-listed folders.
    if output_db.exists():
        output_db.unlink()
    for leftover in (output_db.with_name(output_db.name + "-wal"),
                     output_db.with_name(output_db.name + "-shm")):
        if leftover.exists():
            leftover.unlink()

    stray_root = REPO_ROOT / "outputs" / "scans" / LOCAL_STRAY_SCAN
    iphone_root = REPO_ROOT / "outputs" / "scans" / LOCAL_IPHONE_SCAN
    if not (stray_root / "state.json").is_file():
        raise FileNotFoundError(f"missing local Stray result {LOCAL_STRAY_SCAN}")
    if not (iphone_root / "state.json").is_file():
        raise FileNotFoundError(f"missing local iPhone result {LOCAL_IPHONE_SCAN}")

    stray_state = _load(stray_root / "state.json")
    iphone_state = _load(iphone_root / "state.json")
    stray_model = _stray_model(_load(stray_root / "output" / "spatial_model.json"))
    iphone_model = _iphone_model(_load(iphone_root / "output" / "spatial_model.json"))

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in EVIDENCE_DIR.glob("evidence_*.jpg"):
        stale.unlink()
    written_evidence: list[str] = []
    for spec in IPHONE_EVIDENCE:
        dest = EVIDENCE_DIR / spec["filename"]
        write_jpeg(iphone_root / "output" / "evidence" / spec["source"], dest, spec["crop"])
        written_evidence.append(f"samples/evidence/{spec['filename']}")

    stray_summary = dict(stray_state.get("summary") or {})
    stray_summary.update({
        "normalizedFrameCount": (stray_model.get("provenance") or {})
        .get("capture", {}).get("frameCount"),
        "accuracyClaim": "none",
        "accuracyReason": "No independent tape or laser ground truth is available for this public Stray sample.",
        "aiStatus": (stray_model.get("aiAssessments") or [{}])[0].get("status"),
        "aiFindingCount": 0,
    })
    iphone_summary = dict(iphone_state.get("summary") or {})
    iphone_summary.pop("lossPreviewStatus", None)
    iphone_summary.update({
        "normalizedFrameCount": (iphone_model.get("provenance") or {})
        .get("capture", {}).get("frameCount"),
        "accuracyClaim": "none",
        "accuracyReason": "Tape comparison is not published. Geometry numbers are the model, not a scored accuracy result.",
        "aiStatus": "completed",
        "aiFindingCount": len((iphone_model.get("aiAssessments") or [{}])[0].get("findings") or []),
    })

    stray_pub = _state(
        scan_id=PUBLIC_STRAY_ID,
        name="Public Stray sample 8653a2142b",
        classification="public_sample",
        source_type="stray_scanner",
        source_path="samples/stray/raw/8653a2142b",
        src_state=stray_state,
        summary=stray_summary,
    )
    iphone_pub = _state(
        scan_id=PUBLIC_IPHONE_ID,
        name="Sanitized iPhone review example",
        classification="public_review_example",
        source_type="stray_scanner",
        source_path=None,
        src_state=iphone_state,
        summary=iphone_summary,
    )

    stray_assessment = (stray_model.get("aiAssessments") or [None])[0]
    iphone_assessment = (iphone_model.get("aiAssessments") or [None])[0]

    _write_scan_bundle(PUBLIC_STRAY_ID, stray_pub, stray_model,
                       stray_root / "output", stray_assessment, False)
    _write_scan_bundle(PUBLIC_IPHONE_ID, iphone_pub, iphone_model,
                       iphone_root / "output", iphone_assessment, True)

    output_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_db)
    conn.executescript(PUBLIC_SCHEMA)

    def insert_scan(state: dict) -> None:
        conn.execute(
            """INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (state["scanId"], state["label"], state["classification"],
             state["sourceType"], state["sourcePath"], state["status"],
             state["createdAt"], state["updatedAt"],
             json.dumps(state["summary"]), json.dumps(state)))

    insert_scan(stray_pub)
    insert_scan(iphone_pub)

    if stray_assessment:
        conn.execute(
            "INSERT INTO ai_reviews VALUES (?,?,?,?,?,?)",
            (PUBLIC_STRAY_ID, stray_assessment.get("status"),
             stray_assessment.get("model"), stray_assessment.get("promptVersion"),
             json.dumps(stray_assessment), stray_assessment.get("generatedAt") or _now()))
    if iphone_assessment:
        conn.execute(
            "INSERT INTO ai_reviews VALUES (?,?,?,?,?,?)",
            (PUBLIC_IPHONE_ID, iphone_assessment.get("status"),
             iphone_assessment.get("model"), iphone_assessment.get("promptVersion"),
             json.dumps(iphone_assessment), iphone_assessment.get("generatedAt") or _now()))

    for scan_id in ALLOWLISTED_SCAN_IDS:
        for name, status, description in ARTIFACT_FILES:
            rel = f"samples/public_results/{scan_id}/output/{name}"
            if (REPO_ROOT / rel).is_file():
                conn.execute(
                    "INSERT INTO artifacts VALUES (?,?,?,?,?)",
                    (scan_id, name, rel, status, description))
    for spec in IPHONE_EVIDENCE:
        conn.execute(
            "INSERT INTO evidence VALUES (?,?,?)",
            (PUBLIC_IPHONE_ID, spec["evidence_id"],
             f"samples/evidence/{spec['filename']}"))

    for record in _ca1m_records():
        conn.execute(
            "INSERT INTO validation_records VALUES (?,?,?,?)",
            (record["record_id"], record["kind"], record["capture_id"],
             record["payload_json"]))
    conn.commit()
    conn.close()

    resolved = Path(output_db).resolve()
    try:
        public_db = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        public_db = str(output_db)
    return {
        "publicDb": public_db,
        "bytes": resolved.stat().st_size,
        "scanIds": list(ALLOWLISTED_SCAN_IDS),
        "evidence": written_evidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DB)
    args = parser.parse_args(argv)
    result = export_public_results(args.source, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
