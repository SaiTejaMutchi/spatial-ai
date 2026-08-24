"""Local results catalog: store completed AI once, never treat a miss as success."""

from __future__ import annotations

import json

from service.results_db import (
    ResultsDB,
    ai_cache_key,
    geometry_result_digest,
    opening_cache_key,
)


def _assessment(status="completed", model="qwen/qwen3.6-27b"):
    return {
        "schemaVersion": "0.1",
        "status": status,
        "model": model,
        "provider": "groq",
        "promptVersion": "spatial_verifier_v0.1",
        "findings": [{"id": "f1", "status": "verified"}],
    }


def test_completed_ai_review_is_stored_once(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    assessment = _assessment()
    db.put_ai("key-1", "scan-a", assessment, [], {"latencyMs": 12})
    db.put_ai("key-1", "scan-b", assessment, [], {"latencyMs": 99})
    row = db.get_completed_ai("key-1")
    assert row is not None
    assert row["assessment"]["findings"] == assessment["findings"]
    assert row["scanId"] == "scan-b"


def test_not_run_ai_review_is_not_reused(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    db.put_ai("key-miss", "scan-a", _assessment(status="not_run"), [], {})
    assert db.get_completed_ai("key-miss") is None


def test_rate_limited_openings_are_not_stored(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    db.put_openings("open-1", "scan-a", [{"id": "opening-001"}], {
        "candidates": 1, "classified": 0, "diagnostics": {"accessFailure": "429 rate limit"},
    })
    assert db.get_openings("open-1") is None


def test_durable_openings_round_trip(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    openings = [{"id": "opening-001", "observationState": "unresolved"}]
    report = {"candidates": 1, "classified": 1, "promotedCount": 0, "diagnostics": {}}
    db.put_openings("open-2", "scan-a", openings, report)
    cached = db.get_openings("open-2")
    assert cached["openings"] == openings
    assert cached["report"]["classified"] == 1


def test_imported_qwen_json_is_stored_and_not_sent_to_groq(tmp_path):
    from service.models import STAGES
    from service.scan_manager import ScanManager

    scan_id = "scan-importtest"
    root = tmp_path / "scans"
    manager = ScanManager(root)
    output = root / scan_id / "output"
    output.mkdir(parents=True)
    model = {
        "rooms": [{"id": "room-001"}],
        "surfaces": [{
            "id": "wall-001", "type": "wall",
            "observationState": "directly_observed",
            "dimensions": {"width_m": 1.0, "height_m": 2.0},
        }],
        "measurements": [{"type": "room_length", "value_m": 4.0}],
        "openings": [],
        "evidence": [{"id": "frame-001", "path": "evidence/a.png"}],
        "aiAssessments": [],
        "provenance": {},
    }
    (output / "spatial_model.json").write_text(json.dumps(model))
    (root / scan_id / "state.json").write_text(json.dumps({
        "scanId": scan_id,
        "status": "complete",
        "label": "import",
        "classification": "final_private_capture",
        "sourcePath": "/tmp",
        "sourceType": "stray_scanner",
        "connector": "Stray",
        "stages": [{"name": name, "state": "complete", "detail": None,
                    "startedAt": None, "finishedAt": None} for name in STAGES],
        "currentStage": "complete",
        "failureClass": None,
        "error": None,
        "createdAt": "2026-08-23T00:00:00+00:00",
        "updatedAt": "2026-08-23T00:00:00+00:00",
        "summary": {"aiStatus": "not_run"},
    }))
    raw = {
        "schemaVersion": "0.1",
        "status": "completed",
        "model": "qwen/qwen3.6-27b",
        "provider": "qwen_offline",
        "promptVersion": "spatial_verifier_v0.1",
        "generatedAt": "2026-08-23T00:00:00+00:00",
        "roomTypeHypothesis": "bedroom",
        "notRunReason": None,
        "findings": [{
            "surfaceId": "wall-001",
            "status": "verified",
            "semanticAgreement": True,
            "reason": "The wall is visible and matches the labelled surface.",
            "evidenceFrameIds": ["frame-001"],
        }],
        "usage": None,
    }
    state = manager.import_ai_review(scan_id, raw)
    assert state["summary"]["aiStatus"] == "completed"
    stored = json.loads((output / "ai_assessment.json").read_text())
    assert stored["assessment"]["status"] == "completed"
    assert stored["assessment"]["findings"][0]["surfaceId"] == "wall-001"
    assert stored["diagnostics"]["importSource"] == "offline_qwen_paste"
    assert stored["diagnostics"].get("cacheHit") is False
    assert stored["assessment"]["provider"] == "qwen_offline"


def test_cache_keys_change_when_measurements_change():
    model_a = {
        "measurements": [{"type": "room_length", "value_m": 7.0}],
        "surfaces": [{"id": "wall-001"}],
        "evidence": [{"id": "frame-001"}],
    }
    model_b = json.loads(json.dumps(model_a))
    model_b["measurements"][0]["value_m"] = 7.1
    assert geometry_result_digest(model_a) != geometry_result_digest(model_b)
    assert ai_cache_key(model_a, "cfg", "model") != ai_cache_key(model_b, "cfg", "model")
    assert opening_cache_key(model_a, "cfg", "model") != opening_cache_key(model_b, "cfg", "model")
