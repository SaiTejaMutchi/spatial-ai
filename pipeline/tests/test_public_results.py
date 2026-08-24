"""Privacy and allow-list checks for the curated public results catalog."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from service.export_public_results import (
    ALLOWLISTED_SCAN_IDS,
    EVIDENCE_DIR,
    FORBIDDEN,
    PUBLIC_IPHONE_ID,
    REPO_ROOT,
)

PUBLIC_DB = REPO_ROOT / "samples" / "public_results.db"


def _blob() -> str:
    parts = [PUBLIC_DB.read_bytes()]
    for path in (REPO_ROOT / "samples" / "public_results").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".svg", ".obj", ".mtl", ".txt"}:
            parts.append(path.read_bytes())
    return b"\n".join(parts).decode("utf-8", errors="ignore")


def test_public_db_exists_and_is_small():
    assert PUBLIC_DB.is_file()
    assert PUBLIC_DB.stat().st_size < 2_000_000


def test_public_db_contains_only_allowlisted_scan_ids():
    conn = sqlite3.connect(PUBLIC_DB)
    ids = {row[0] for row in conn.execute("SELECT scan_id FROM scans")}
    assert ids == set(ALLOWLISTED_SCAN_IDS)
    ai_ids = {row[0] for row in conn.execute("SELECT scan_id FROM ai_reviews")}
    assert ai_ids <= set(ALLOWLISTED_SCAN_IDS)
    ev_ids = {row[0] for row in conn.execute("SELECT scan_id FROM evidence")}
    assert ev_ids <= {PUBLIC_IPHONE_ID}


def test_no_absolute_paths_env_keys_or_private_capture_roots():
    blob = _blob().lower()
    assert "/users/" not in blob
    assert "saiteja" not in blob
    assert "groq_api_key" not in blob
    assert "sk-" not in blob
    assert ".env" not in blob
    assert "iphone-actual" not in blob
    assert "rgb.mp4" not in blob
    assert ".mov" not in blob
    assert "odometry.csv" not in blob
    assert "/depth/" not in blob
    assert "confidence/" not in blob
    assert "outputs/scans" not in blob


def test_no_videos_or_raw_depth_files_are_published():
    published = list((REPO_ROOT / "samples" / "public_results").rglob("*"))
    published.extend(EVIDENCE_DIR.glob("*"))
    suffixes = {path.suffix.lower() for path in published if path.is_file()}
    assert ".mp4" not in suffixes
    assert ".mov" not in suffixes
    assert ".ply" not in suffixes


def test_production_damage_and_scope_stay_empty():
    conn = sqlite3.connect(PUBLIC_DB)
    for (state_json,) in conn.execute("SELECT state_json FROM scans"):
        state = json.loads(state_json)
        assert not FORBIDDEN.search(json.dumps(state))
    for scan_id in ALLOWLISTED_SCAN_IDS:
        model = json.loads((
            REPO_ROOT / "samples" / "public_results" / scan_id / "output"
            / "spatial_model.json").read_text())
        assert model.get("damage") == []
        assert model.get("scope") == []


def test_referenced_evidence_files_exist_and_are_few():
    conn = sqlite3.connect(PUBLIC_DB)
    rows = list(conn.execute("SELECT scan_id, evidence_id, rel_path FROM evidence"))
    assert rows
    assert all(row[0] == PUBLIC_IPHONE_ID for row in rows)
    assert len(rows) <= 5
    for _scan_id, _evidence_id, rel in rows:
        assert not rel.startswith("/")
        assert rel.startswith("samples/evidence/")
        assert (REPO_ROOT / rel).is_file()
        assert (REPO_ROOT / rel).stat().st_size < 500_000


def test_ca1m_validation_records_are_numeric_only():
    conn = sqlite3.connect(PUBLIC_DB)
    rows = list(conn.execute(
        "SELECT record_id, kind, capture_id, payload_json FROM validation_records"))
    assert rows
    kinds = {row[1] for row in rows}
    assert "ca1m_heldout_aggregate" in kinds
    assert "ca1m_heldout_capture" in kinds
    for _record_id, _kind, _capture_id, payload in rows:
        data = json.loads(payload)
        assert "png" not in json.dumps(data).lower()
        assert "mp4" not in json.dumps(data).lower()
        if "modelHeight_m" in data:
            assert isinstance(data["modelHeight_m"], (int, float))
            assert isinstance(data["laserReferenceHeight_m"], (int, float))
            assert data["gateResult"] in {"pass", "fail"}
