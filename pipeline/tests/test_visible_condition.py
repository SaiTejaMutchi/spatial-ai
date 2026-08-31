"""Visible Condition Grounding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from pipeline.ai.make_condition_fixture import write_fixture
from pipeline.ai.verifier import GroqVerifierClient, protected_geometry_digest
from pipeline.ai.visible_condition import (
    attach_recorded_proposal,
    ground_visible_condition,
)
from pipeline.loss_preview.preview import build_loss_preview, register_region_to_surface

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "outputs" / "dev_47333462" / "spatial_model.json"
SCHEMA = REPO_ROOT / "schema" / "visible_condition.schema.json"

requires_generated = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason="run the geometry pipeline first")


class JsonClient(GroqVerifierClient):
    def __init__(self, response: dict):
        super().__init__(post=lambda *a, **k: {}, sleeper=lambda s: None,
                         environ={"GROQ_API_KEY": "gsk_test"})
        self.response = response
        self.calls = []

    def complete_json(self, system, user_text, images, model):
        self.calls.append(user_text)
        return {**self.response, "model": model, "provider": "groq"}


@pytest.fixture
def model():
    return json.loads(MODEL_PATH.read_text())


@pytest.fixture
def wall_and_view(model):
    for view in model["evidence"]:
        for surface_id in view["visibleSurfaceIds"]:
            surface = next(s for s in model["surfaces"] if s["id"] == surface_id)
            if surface["type"] == "wall" and surface["observationState"] == "directly_observed":
                return surface, view
    raise AssertionError("no observed wall with evidence")


@pytest.fixture
def stain_bytes(tmp_path):
    root = write_fixture(tmp_path / "cond")
    return (root / "stained_evidence.png").read_bytes()


def test_schema_rejects_a_smuggled_area():
    schema = json.loads(SCHEMA.read_text())
    bad = {
        "status": "supported", "conditionClass": "staining",
        "surfaceId": "wall-001", "evidenceFrameIds": ["frame-a"],
        "reason": "brown mark",
        "region": {"x0": 0.2, "y0": 0.2, "x1": 0.6, "y1": 0.6},
        "affectedArea_m2": 1.2,
    }
    assert list(Draft202012Validator(schema).iter_errors(bad))


def test_schema_rejects_region_quantity_field():
    schema = json.loads(SCHEMA.read_text())
    bad = {
        "status": "supported", "conditionClass": "staining",
        "surfaceId": "wall-001", "evidenceFrameIds": ["frame-a"],
        "reason": "brown mark",
        "region": {"x0": 0.2, "y0": 0.2, "x1": 0.6, "y1": 0.6, "area_m2": 3},
    }
    assert list(Draft202012Validator(schema).iter_errors(bad))


@requires_generated
def test_supported_region_registers_with_geometry_owned_quantity(
        model, wall_and_view, stain_bytes):
    surface, view = wall_and_view
    before = protected_geometry_digest(model)
    client = JsonClient({
        "status": "supported", "conditionClass": "staining",
        "surfaceId": surface["id"], "evidenceFrameIds": [view["id"]],
        "reason": "a dark stain is visible on this wall",
        "region": {"x0": 0.25, "y0": 0.35, "x1": 0.7, "y1": 0.75},
    })
    result = ground_visible_condition(
        model, stain_bytes, surface, view, client=client)
    assert result.diagnostics["validationResult"] == "accepted"
    assert result.proposal is not None
    assert result.proposal["registration"]["status"] == "registered"
    assert result.proposal["quantity"]["producer"] == "geometry_pipeline"
    assert result.proposal["quantity"]["affectedArea_m2"] > 0
    assert "affectedArea_m2" not in result.record
    assert result.proposal["reviewStatus"] == "human_review_required"
    assert result.diagnostics["geometryMutationCount"] == 0
    assert protected_geometry_digest(model) == before
    assert model["damage"] == []
    assert model["scope"] == []


@requires_generated
def test_no_supported_condition_emits_no_quantity(model, wall_and_view, stain_bytes):
    surface, view = wall_and_view
    client = JsonClient({
        "status": "no_supported_condition", "conditionClass": "none",
        "surfaceId": surface["id"], "evidenceFrameIds": [view["id"]],
        "reason": "the wall is unmarked",
    })
    result = ground_visible_condition(
        model, stain_bytes, surface, view, client=client)
    assert result.proposal is None
    assert result.record["status"] == "no_supported_condition"


@requires_generated
def test_insufficient_evidence_emits_no_quantity(model, wall_and_view, stain_bytes):
    surface, view = wall_and_view
    client = JsonClient({
        "status": "insufficient_evidence", "conditionClass": "none",
        "surfaceId": surface["id"], "evidenceFrameIds": [view["id"]],
        "reason": "the crop is too dark",
    })
    result = ground_visible_condition(
        model, stain_bytes, surface, view, client=client)
    assert result.proposal is None


@requires_generated
def test_surface_id_mismatch_is_not_repaired(model, wall_and_view, stain_bytes):
    surface, view = wall_and_view
    client = JsonClient({
        "status": "supported", "conditionClass": "staining",
        "surfaceId": "wall-does-not-exist", "evidenceFrameIds": [view["id"]],
        "reason": "stain", "region": {"x0": 0.2, "y0": 0.2, "x1": 0.5, "y1": 0.5},
    })
    result = ground_visible_condition(
        model, stain_bytes, surface, view, client=client)
    assert result.proposal is None
    assert result.diagnostics["validationResult"] == "id_rejected"


@requires_generated
def test_metric_field_is_rejected(model, wall_and_view, stain_bytes):
    surface, view = wall_and_view
    client = JsonClient({
        "status": "supported", "conditionClass": "staining",
        "surfaceId": surface["id"], "evidenceFrameIds": [view["id"]],
        "reason": "stain", "region": {"x0": 0.2, "y0": 0.2, "x1": 0.5, "y1": 0.5},
        "affectedArea_m2": 12.0,
    })
    result = ground_visible_condition(
        model, stain_bytes, surface, view, client=client)
    assert result.proposal is None
    assert result.diagnostics["validationResult"] == "metric_rejected"


def test_unresolved_registration_emits_no_quantity():
    surface = {"id": "wall-001", "plane": {"normal": [0.0, 0.0, 0.0], "offset_m": 0.0}}
    registration = register_region_to_surface(
        {"x0": 0.2, "y0": 0.2, "x1": 0.5, "y1": 0.5}, surface,
        __import__("numpy").eye(4),
        {"width": 256, "height": 192, "fx": 200, "fy": 200, "cx": 128, "cy": 96})
    assert registration["status"] == "unresolved"
    assert "affectedArea_m2" not in registration


@requires_generated
def test_full_frame_region_is_inset_until_it_registers(
        model, wall_and_view, stain_bytes):
    surface, view = wall_and_view
    client = JsonClient({
        "status": "supported", "conditionClass": "staining",
        "surfaceId": surface["id"], "evidenceFrameIds": [view["id"]],
        "reason": "stain covers the wall",
        "region": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
    })
    result = ground_visible_condition(
        model, stain_bytes, surface, view, client=client)
    assert result.proposal is not None
    assert result.proposal["registration"]["status"] == "registered"
    assert result.proposal["quantity"]["producer"] == "geometry_pipeline"
    assert result.proposal["affectedRegion"]["geometryInset"] > 0
    assert result.proposal["affectedRegion"]["modelProposedRegion"] == {
        "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0,
    }
    wall_area = surface["dimensions"]["width_m"] * surface["dimensions"]["height_m"]
    assert result.proposal["quantity"]["affectedArea_m2"] <= wall_area * 1.25


@requires_generated
def test_synthetic_fixture_remains_the_first_proposal(model, tmp_path):
    preview = build_loss_preview(model, tmp_path)
    assert preview["label"] == "DEVELOPMENT LOSS FIXTURE"
    assert preview["proposals"][0]["provenance"]["aiProducer"] == (
        "development_fixture_not_a_model")
    assert preview["proposals"][0]["quantity"]["producer"] == "geometry_pipeline"


@requires_generated
def test_recorded_model_region_is_reregistered_by_geometry(
        model, wall_and_view, tmp_path, monkeypatch):
    surface, view = wall_and_view
    recorded = {
        "conditionId": "condition-model-001",
        "surfaceId": surface["id"],
        "conditionClass": "staining",
        "evidenceFrameIds": [view["id"]],
        "normalizedImageRegion": {"x0": 0.3, "y0": 0.4, "x1": 0.6, "y1": 0.7},
        "provenance": {"quantityProducer": "geometry_pipeline",
                       "aiProducer": "qwen/qwen3.6-27b"},
    }
    path = tmp_path / "accepted_proposal.json"
    path.write_text(json.dumps(recorded))
    import pipeline.ai.visible_condition as vc
    monkeypatch.setattr(vc, "RECORDED_PROPOSAL", path)
    preview = {"proposals": []}
    attach_recorded_proposal(preview, model)
    assert preview["modelGeneratedGrounding"] == "completed"
    proposal = preview["proposals"][-1]
    assert proposal["quantity"]["producer"] == "geometry_pipeline"
    assert proposal["affectedRegion"]["producer"] == "ai_visible_condition"
    assert "affectedArea_m2" not in recorded
