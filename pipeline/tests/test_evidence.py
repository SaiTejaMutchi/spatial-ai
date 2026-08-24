"""RGB evidence selection and registration tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.evidence.select import surface_visibility

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "outputs" / "dev_47333462" / "evidence_manifest.json"
MODEL = REPO_ROOT / "outputs" / "dev_47333462" / "spatial_model.json"

requires_generated = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="run the geometry pipeline first")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def model():
    return json.loads(MODEL.read_text())


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def test_a_surface_squarely_in_front_is_fully_visible():
    grid = np.stack(np.meshgrid(np.linspace(-0.4, 0.4, 30),
                                np.linspace(-0.3, 0.3, 30)), axis=-1).reshape(-1, 2)
    points = np.stack([grid[:, 0], grid[:, 1], np.full(len(grid), 2.0)], axis=1)
    visibility = surface_visibility(points, np.eye(4), 200, 200, 128, 96, 256, 192)
    assert visibility == pytest.approx(1.0)


def test_a_surface_behind_the_camera_is_invisible():
    points = np.array([[0.0, 0.0, -2.0], [0.1, 0.1, -3.0]])
    assert surface_visibility(points, np.eye(4), 200, 200, 128, 96, 256, 192) == 0.0


def test_a_surface_outside_the_sensor_is_invisible():
    points = np.array([[50.0, 0.0, 2.0], [60.0, 0.0, 2.0]])
    assert surface_visibility(points, np.eye(4), 200, 200, 128, 96, 256, 192) == 0.0


def test_partial_visibility_is_reported_as_a_fraction():
    points = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0],
                       [50.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    visibility = surface_visibility(points, np.eye(4), 200, 200, 128, 96, 256, 192)
    assert 0.4 < visibility < 0.6


def test_visibility_of_nothing_is_zero():
    assert surface_visibility(np.zeros((0, 3)), np.eye(4), 200, 200, 128, 96, 256, 192) == 0.0


# --------------------------------------------------------------------------
# selection results
# --------------------------------------------------------------------------

@requires_generated
def test_view_count_respects_the_verifier_contract(manifest):
    """The plan specifies 3-8 evidence frames."""
    assert 3 <= len(manifest["views"]) <= 8


@requires_generated
def test_every_view_is_registered_with_full_provenance(manifest):
    for view in manifest["views"]:
        assert view["registration"] == "registered_by_capture_pose"
        assert view["sourceFrame"]
        assert view["timestamp_s"] > 0
        assert np.array(view["cameraToWorld"]).shape == (4, 4)
        assert view["producer"] == "geometry_pipeline"
        assert view["visibleSurfaceIds"]


@requires_generated
def test_every_view_file_exists(manifest):
    for view in manifest["views"]:
        assert (MANIFEST.parent / view["path"]).is_file(), view["path"]


@requires_generated
def test_views_reference_only_real_surfaces(manifest, model):
    surface_ids = {s["id"] for s in model["surfaces"]}
    for view in manifest["views"]:
        for surface_id in view["visibleSurfaceIds"]:
            assert surface_id in surface_ids, surface_id


@requires_generated
def test_recorded_visibility_meets_the_threshold(manifest):
    threshold = manifest["diagnostics"]["minVisibilityThreshold"]
    for view in manifest["views"]:
        for surface_id, fraction in view["surfaceVisibility"].items():
            assert fraction >= threshold, f"{surface_id} at {fraction}"


@requires_generated
def test_selected_views_are_temporally_distinct(manifest):
    """Two consecutive frames are the same photograph twice."""
    times = sorted(v["timestamp_s"] for v in manifest["views"])
    for earlier, later in zip(times, times[1:]):
        assert later - earlier > 1.0, "views are too close together in time"


@requires_generated
def test_structural_surfaces_are_evidence_targets(manifest):
    """Floor and ceiling determine room height; they cannot be unseeable."""
    covered = set(manifest["diagnostics"]["surfacesCovered"])
    assert any(s.startswith("floor") for s in covered)
    assert any(s.startswith("ceiling") for s in covered)


@requires_generated
def test_surfaces_without_evidence_are_listed_with_a_reason(manifest):
    diagnostics = manifest["diagnostics"]
    if diagnostics["surfacesWithoutEvidence"]:
        assert diagnostics["uncoveredReason"]


@requires_generated
def test_inferred_closure_is_never_claimed_as_photographed(manifest, model):
    """A surface with no plane behind it cannot appear in a photograph."""
    inferred = {s["id"] for s in model["surfaces"]
                if s["observationState"] == "inferred"}
    for view in manifest["views"]:
        assert not (set(view["visibleSurfaceIds"]) & inferred)


@requires_generated
def test_manifest_states_its_occlusion_and_privacy_position(manifest):
    diagnostics = manifest["diagnostics"]
    assert "occlusion" in diagnostics["occlusionNote"].lower()
    assert "operator approval" in diagnostics["privacyNote"]


@requires_generated
def test_model_carries_the_same_evidence(manifest, model):
    assert len(model["evidence"]) == len(manifest["views"])
    assert {v["id"] for v in model["evidence"]} == {v["id"] for v in manifest["views"]}
    assert model["provenance"]["evidenceSelection"]["viewsSelected"] == len(manifest["views"])
