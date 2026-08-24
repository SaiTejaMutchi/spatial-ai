"""The validation runner must never manufacture a result.

The ledger's rule is that a pipeline output is not a validation result unless an
independent expected value exists. These tests are about that rule holding even
when data is missing, which is the case that matters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.validation import run_ledger

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_absent_datasets_are_blocked_not_omitted_and_not_estimated():
    records = run_ledger.blocked_records()
    ids = {r.validation_id for r in records}
    # V02B was unblocked by downloading CA-1M, so it must no longer sit here.
    assert {"V03", "V04", "V07"} <= ids
    assert "V02B" not in ids
    for record in records:
        assert record.verdict == run_ledger.BLOCKED
        assert record.conclusion, "a blocked row must say why"
        assert not record.results, "a blocked row must carry no numbers"


def test_v02b_reports_the_gate_and_never_invents_a_result():
    record = run_ledger.run_v02b()
    if record.verdict == run_ledger.NOT_STARTED:
        assert not record.results
        return
    assert record.verdict == run_ledger.PARTIAL
    assert record.results["roomHeightGate"]["gate"] == "at most 1.5 cm"
    assert record.results["capturesEvaluated"] >= 3
    assert "held out" in record.held_out or "yes" in record.held_out
    assert any("no mobile camera pose" in note for note in record.known_ambiguity), (
        "the record must state where the trajectory came from")
    assert "No geometry parameter was changed" in record.conclusion


def test_a_stray_model_without_a_reference_is_never_accuracy(tmp_path):
    model = tmp_path / "spatial_model.json"
    model.write_text(json.dumps({
        "surfaces": [{"id": "floor-001", "type": "floor", "observationState": "directly_observed"},
                     {"id": "ceiling-001", "type": "ceiling", "observationState": "directly_observed"}],
        "measurements": [{"type": "room_height", "value_m": 2.672}],
    }))
    record = run_ledger.run_v06(model, tmp_path)
    assert record.verdict == run_ledger.PASS
    assert record.reference_source == "none found / none attached"
    assert "NOT COMPARABLE" in record.conclusion
    assert any("unvalidated output" in note for note in record.known_ambiguity)


def test_a_capture_without_a_room_fails_interoperability(tmp_path):
    model = tmp_path / "spatial_model.json"
    model.write_text(json.dumps({"surfaces": [{"id": "wall-001", "type": "wall"}],
                                 "measurements": []}))
    assert run_ledger.run_v06(model, tmp_path).verdict == run_ledger.FAIL


def test_a_missing_model_blocks_v02_rather_than_passing(tmp_path):
    record = run_ledger.run_v02({"47333462": tmp_path / "nope.json"}, tmp_path)
    assert record.verdict == run_ledger.BLOCKED
    assert not record.results


def test_the_summary_never_truncates_mid_word():
    clipped = run_ledger._first_sentence(
        "A sentence that runs on well past any reasonable column width and therefore "
        "has to be shortened somewhere sensible rather than mid word", limit=60)
    assert clipped.endswith("...")
    assert not clipped[:-3].endswith(" ")
    assert " ".join(clipped[:-3].split()) == clipped[:-3]


def test_every_verdict_is_one_the_ledger_allows():
    allowed = {run_ledger.NOT_STARTED, run_ledger.PASS, run_ledger.FAIL,
               run_ledger.PARTIAL, run_ledger.NOT_COMPARABLE, run_ledger.BLOCKED}
    path = (REPO_ROOT / "docs" / "validation"
              / "SPATIAL_AI_VALIDATION_LEDGER.md")
    if not path.exists():
        pytest.skip("Validation ledger doc not present in public repository")
    ledger = path.read_text()
    for verdict in allowed:
        assert f"`{verdict}`" in ledger or verdict in ledger, verdict


def test_the_run_record_carries_the_freeze_state():
    """A result without the configuration that produced it is not evidence."""
    record = run_ledger.RunRecord("VXX", "d", "role", "modality").to_record()
    assert record["pipeline"]["frozen"]["state"] == "PARAMETERS_FROZEN"
    assert record["pipeline"]["frozen"]["geometryConfigHash"]
