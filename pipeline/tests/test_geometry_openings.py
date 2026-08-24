"""Opening detection and conservative recording tests.

The fixtures resolve no opening, which is the correct outcome under the plan's
rules. That makes it essential to prove separately that the detector *does*
fire when the evidence is there — otherwise "found nothing" would be
indistinguishable from "cannot find anything".
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.contracts.validate_model import validate_model
from pipeline.geometry.confidence import load_confidence_rules
from pipeline.geometry.config import load_geometry_config
from pipeline.geometry.openings import _best_empty_rectangle, detect_openings
from pipeline.geometry.planes import Plane

DOOR_WIDTH = 0.90
DOOR_HEIGHT = 2.05
WALL_LENGTH = 4.0
WALL_HEIGHT = 2.5


@pytest.fixture(scope="module")
def config():
    return load_geometry_config()


@pytest.fixture(scope="module")
def rules():
    return load_confidence_rules()


def _wall_with_gap(
    gap_centre_u: float | None = 2.0,
    gap_width: float = DOOR_WIDTH,
    gap_height: float = DOOR_HEIGHT,
    gap_bottom: float = 0.0,
    spacing: float = 0.02,
    beyond_points: int = 0,
    front_points: int = 0,
    seed: int = 11,
):
    """A wall in the plane z = 0 with an optional rectangular hole punched out.

    The wall normal points along +z, so 'beyond the wall' is negative z.
    """
    rng = np.random.default_rng(seed)
    us = np.arange(0.0, WALL_LENGTH + spacing, spacing)
    vs = np.arange(0.0, WALL_HEIGHT + spacing, spacing)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="ij")
    u = grid_u.ravel()
    v = grid_v.ravel()

    if gap_centre_u is not None:
        hole = ((np.abs(u - gap_centre_u) <= gap_width / 2)
                & (v >= gap_bottom) & (v <= gap_bottom + gap_height))
        u, v = u[~hole], v[~hole]

    points = np.stack([u, v, np.zeros(len(u))], axis=1)
    points += rng.normal(scale=0.003, size=points.shape)

    extras = []
    if beyond_points and gap_centre_u is not None:
        extra_u = rng.uniform(gap_centre_u - gap_width / 2, gap_centre_u + gap_width / 2,
                              beyond_points)
        extra_v = rng.uniform(gap_bottom, gap_bottom + gap_height, beyond_points)
        extras.append(np.stack([extra_u, extra_v,
                                rng.uniform(-2.0, -0.5, beyond_points)], axis=1))
    if front_points and gap_centre_u is not None:
        extra_u = rng.uniform(gap_centre_u - gap_width / 2, gap_centre_u + gap_width / 2,
                              front_points)
        extra_v = rng.uniform(gap_bottom, gap_bottom + gap_height, front_points)
        extras.append(np.stack([extra_u, extra_v,
                                rng.uniform(0.5, 2.0, front_points)], axis=1))
    if extras:
        points = np.concatenate([points] + extras)

    plane = Plane(
        plane_id="wall-src-001", kind="wall",
        normal=np.array([0.0, 0.0, 1.0]), offset=0.0,
        inlier_count=len(points), rms_residual_m=0.003, max_residual_m=0.01,
        coverage_fraction=0.9, contributing_frames=30,
        extent={"verticalSpan_m": WALL_HEIGHT},
        centroid=points.mean(axis=0),
        algorithm="synthetic",
    )
    surface = {
        "id": "wall-001", "type": "wall", "observationState": "directly_observed",
        "provenance": {"sourcePlaneId": "wall-src-001"},
    }
    return [surface], {"wall-src-001": plane}, points


# --------------------------------------------------------------------------
# rectangle search
# --------------------------------------------------------------------------

def test_search_prefers_a_door_shape_over_a_wide_thin_strip():
    """A wide unobserved band beats a doorway on raw area; it must not win."""
    grid = np.ones((40, 80), dtype=bool)
    grid[0, :] = False                 # a 1-cell strip across the whole wall
    grid[10:35, 20:38] = False         # a door-shaped hole
    found = _best_empty_rectangle(grid, min_cols=8, max_cols=24, min_rows=8)
    assert found is not None
    r0, c0, r1, c1 = found
    assert (r1 - r0 + 1) >= 20 and 18 <= (c1 - c0 + 1) <= 24


def test_search_returns_nothing_when_no_region_is_large_enough():
    grid = np.ones((20, 20), dtype=bool)
    grid[5, 5] = False
    assert _best_empty_rectangle(grid, min_cols=8, max_cols=20, min_rows=8) is None


def test_a_run_wider_than_the_bound_is_clipped_not_discarded():
    grid = np.ones((30, 100), dtype=bool)
    grid[5:25, 10:90] = False
    found = _best_empty_rectangle(grid, min_cols=8, max_cols=30, min_rows=8)
    assert found is not None
    _, c0, _, c1 = found
    assert (c1 - c0 + 1) == 30


# --------------------------------------------------------------------------
# detection on evidence that genuinely supports an opening
# --------------------------------------------------------------------------

def test_a_real_doorway_is_detected_with_its_measured_extent(config):
    surfaces, planes, points = _wall_with_gap()
    result = detect_openings(surfaces, planes, points, config)
    assert len(result.candidates) == 1, result.diagnostics
    candidate = result.candidates[0]
    assert abs(candidate.width_m - DOOR_WIDTH) < 0.12
    assert abs(candidate.height_m - DOOR_HEIGHT) < 0.12
    assert candidate.surface_id == "wall-001"
    assert candidate.border_support >= config.get("opening_border_support_fraction")


def test_a_detected_doorway_is_still_recorded_unresolved(config):
    """Geometry alone may not resolve an opening, however clean the gap is."""
    surfaces, planes, points = _wall_with_gap()
    result = detect_openings(surfaces, planes, points, config)
    record = result.records[0]
    assert record["observationState"] == "unresolved"
    assert record["dimensions"] is None
    assert record["surfaceId"] == "wall-001"
    assert "RGB or AI corroboration is required" in record["provenance"]["reason"]
    # The measurement exists, but as a diagnostic rather than a dimension.
    assert record["provenance"]["candidateExtent"]["width_m"] > 0
    assert record["provenance"]["whatWouldResolveThis"]
    assert len(record["provenance"]["candidateExtent"]["worldCorners_m"]) == 4


def test_see_through_evidence_is_recorded(config):
    surfaces, planes, points = _wall_with_gap(beyond_points=800, front_points=20)
    result = detect_openings(surfaces, planes, points, config)
    evidence = result.candidates[0].diagnostics
    assert evidence["pointsBeyondPlane"] > evidence["pointsInFrontOfPlane"]
    assert evidence["seeThroughEvidence"] is True


def test_an_occluded_patch_is_distinguished_from_a_hole(config):
    """Returns in front of the wall mean furniture, not an opening."""
    surfaces, planes, points = _wall_with_gap(beyond_points=20, front_points=800)
    result = detect_openings(surfaces, planes, points, config)
    evidence = result.candidates[0].diagnostics
    assert evidence["seeThroughEvidence"] is False


def test_a_gap_at_the_edge_of_the_observed_region_is_rejected(config):
    """No wall to one side means an observation boundary, not an opening."""
    surfaces, planes, points = _wall_with_gap(gap_centre_u=0.30, gap_width=0.8)
    result = detect_openings(surfaces, planes, points, config)
    assert result.candidates == []
    assert any("edge of the observed region" in r["reason"] for r in result.diagnostics["rejected"])


def test_a_wall_with_no_gap_yields_an_explicit_unresolved_record(config):
    surfaces, planes, points = _wall_with_gap(gap_centre_u=None)
    result = detect_openings(surfaces, planes, points, config)
    assert result.candidates == []
    record = result.records[0]
    assert record["id"] == "opening-unresolved-001"
    assert record["observationState"] == "unresolved"
    assert record["dimensions"] is None
    assert record["provenance"]["wallsExamined"] == 1
    assert "recorded as unresolved rather than estimated" in record["provenance"]["reason"]


def test_inferred_walls_are_not_searched(config):
    """A wall that was never observed cannot evidence an opening."""
    surfaces, planes, points = _wall_with_gap()
    surfaces[0]["observationState"] = "inferred"
    result = detect_openings(surfaces, planes, points, config)
    assert result.diagnostics["wallsExamined"] == 0
    assert result.candidates == []


def test_every_rejection_states_its_reason(config):
    surfaces, planes, points = _wall_with_gap(gap_centre_u=0.30, gap_width=0.8)
    for record in detect_openings(surfaces, planes, points, config).diagnostics["rejected"]:
        assert record["surfaceId"]
        assert len(record["reason"]) > 30


def test_detection_is_deterministic(config):
    surfaces, planes, points = _wall_with_gap()
    first = detect_openings(surfaces, planes, points, config).records
    second = detect_openings(surfaces, planes, points, config).records
    assert first == second


def test_config_provenance_is_reported(config):
    surfaces, planes, points = _wall_with_gap()
    provenance = detect_openings(surfaces, planes, points, config).config_provenance
    assert provenance["geometryConfigId"] == "geometry_config_v0.1"
    assert "opening_min_width_m" in provenance["parameters"]


# --------------------------------------------------------------------------
# model-level guarantees
# --------------------------------------------------------------------------

def test_the_model_validator_refuses_a_dimensioned_unresolved_opening(rules):
    """The rule that keeps a candidate from becoming a measurement."""
    import json
    from pathlib import Path

    model = json.loads(
        (Path(__file__).resolve().parents[2] / "outputs" / "dev_47333462"
         / "spatial_model.json").read_text()) if (
        Path(__file__).resolve().parents[2] / "outputs" / "dev_47333462"
        / "spatial_model.json").is_file() else None
    if model is None:
        pytest.skip("no generated model available")
    assert validate_model(model) == []
    model["openings"][0]["dimensions"] = {"width_m": 0.9, "height_m": 2.05}
    assert any("must not carry dimensions" in p for p in validate_model(model))
