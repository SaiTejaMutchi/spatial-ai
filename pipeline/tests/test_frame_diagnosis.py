"""The bounded second opinion on an ambiguous capture frame.

The whole point of this layer is that it cannot do anything. These tests are
about what it is forbidden to produce, not about the quality of its prose.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.ai.frame_diagnosis import DIAGNOSIS_SCHEMA, build_evidence, diagnose_frame

REPO_ROOT = Path(__file__).resolve().parents[2]


def _candidate(axis, plausible, **kwargs):
    base = dict(axis=axis, floorDetected=plausible, ceilingDetected=plausible,
                roomHeightM=2.6 if plausible else None, floorRmsMm=12.0,
                ceilingRmsMm=14.0, horizontalSupportFraction=0.3,
                verticalStructureFraction=0.5, trajectoryVerticalRatio=0.2,
                plausible=plausible)
    base.update(kwargs)
    return SimpleNamespace(**base)


def _resolution(outcome="ambiguous", candidates=None):
    return SimpleNamespace(
        outcome=outcome, axis=None, basis="none", declared_axis="+y",
        declared_verified=False,
        candidates=candidates or [_candidate("+y", True), _candidate("+z", False)])


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def assess(self, system, summary, images, response_schema, model):
        self.calls.append(summary)
        return dict(self.payload)


APPROVED = SimpleNamespace(approved=True, provider="groq", model="test-model")


def test_the_schema_forbids_a_transform_or_a_measurement():
    schema = json.loads(DIAGNOSIS_SCHEMA.read_text())
    assert schema["additionalProperties"] is False
    allowed = set(schema["properties"])
    for forbidden in ("transform", "rotation", "matrix", "roomHeightM", "correction"):
        assert forbidden not in allowed
    assert schema["properties"]["recommendation"]["enum"] == [
        "accept_declared_axis", "accept_single_plausible_candidate",
        "route_to_human_review", "reject_capture"]


def test_it_does_not_run_when_the_frame_was_already_resolved():
    result = diagnose_frame(_resolution(outcome="verified"), "stray_scanner",
                            config=APPROVED, client=_Client({}))
    assert result["status"] == "not_run"


def test_it_does_not_run_without_an_approved_model():
    unapproved = SimpleNamespace(approved=False, provider=None, model=None)
    result = diagnose_frame(_resolution(), "stray_scanner", config=unapproved)
    assert result["status"] == "not_run"
    assert "APPROVED" in result["reason"]


def test_an_axis_the_deterministic_layer_rejected_is_dropped():
    """The model may not introduce an orientation the evidence did not support."""
    client = _Client({"assessment": "a", "reason": "b",
                      "recommendation": "accept_single_plausible_candidate",
                      "recommendedAxis": "+z", "confidence": "high"})
    result = diagnose_frame(_resolution(), "stray_scanner", config=APPROVED, client=client)
    assert result["status"] == "completed"
    assert result["diagnosis"]["recommendedAxis"] is None
    assert result["droppedRecommendedAxis"] == "+z"


def test_a_plausible_axis_survives():
    client = _Client({"assessment": "a", "reason": "b",
                      "recommendation": "accept_declared_axis",
                      "recommendedAxis": "+y", "confidence": "medium"})
    result = diagnose_frame(_resolution(), "stray_scanner", config=APPROVED, client=client)
    assert result["diagnosis"]["recommendedAxis"] == "+y"
    assert result["droppedRecommendedAxis"] is None


def test_it_is_advisory_and_says_so():
    client = _Client({"assessment": "a", "reason": "b",
                      "recommendation": "accept_declared_axis",
                      "recommendedAxis": "+y", "confidence": "high"})
    result = diagnose_frame(_resolution(), "stray_scanner", config=APPROVED, client=client)
    assert result["advisoryOnly"] is True
    assert "human review" in result["note"]


def test_an_off_schema_response_is_discarded_whole():
    client = _Client({"assessment": "a", "recommendation": "do_whatever"})
    result = diagnose_frame(_resolution(), "stray_scanner", config=APPROVED, client=client)
    assert result["status"] == "not_run"
    assert result["diagnosis"] is None


def test_only_scalar_evidence_leaves_the_machine():
    """No pose, no image, no local path is sent to a provider."""
    resolution = _resolution()
    evidence = build_evidence(resolution, "stray_scanner")
    encoded = json.dumps(evidence)
    for leak in ("/Users", "camera_to_world", "depth", ".png", "path"):
        assert leak not in encoded
    assert set(evidence) == {"source", "declaredAxis", "deterministicOutcome",
                             "candidateFrames"}


def test_the_prompt_forbids_inventing_a_transform():
    prompt = (REPO_ROOT / "prompts" / "frame_diagnosis_v0.1.txt").read_text().lower()
    assert "may not propose a rotation" in prompt
    assert "may not propose an axis that is not in the candidate list" in prompt
