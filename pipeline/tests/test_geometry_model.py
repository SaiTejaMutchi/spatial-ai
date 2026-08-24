"""Room envelope, measurements, and canonical JSON tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.connectors.arkitscenes import ARKitScenesConnector
from pipeline.contracts.validate_model import validate_model
from pipeline.geometry.confidence import (
    ConfidenceRulesError,
    load_confidence_rules,
)
from pipeline.geometry.config import load_geometry_config
from pipeline.geometry.envelope import (
    build_envelope,
    minimum_area_rectangle,
    polygon_area,
    select_boundary_walls,
)
from pipeline.geometry.model import build_spatial_model, write_model
from pipeline.geometry.planes import extract_planes
from pipeline.geometry.points import build_point_cloud
from pipeline.geometry.run import run_geometry
from pipeline.tests.test_geometry_planes import (
    ROOM_HEIGHT,
    ROOM_LENGTH,
    ROOM_WIDTH,
    _synthetic_room,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_FIXTURE = REPO_ROOT / "samples/arkitscenes/raw/Training/47333462"

requires_fixture = pytest.mark.skipif(
    not PRIMARY_FIXTURE.is_dir(), reason="ARKitScenes primary fixture is not present")


@pytest.fixture(scope="module")
def config():
    return load_geometry_config()


@pytest.fixture(scope="module")
def rules():
    return load_confidence_rules()


def _floor_points(cloud, planes, config):
    distance = np.abs(cloud.points @ planes.floor.normal - planes.floor.offset)
    return cloud.points[distance <= config.get("plane_inlier_distance_m")]


# --------------------------------------------------------------------------
# plan geometry primitives
# --------------------------------------------------------------------------

def test_polygon_area_matches_a_known_rectangle():
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    assert abs(polygon_area(square) - 12.0) < 1e-9
    assert abs(polygon_area(square[::-1]) - 12.0) < 1e-9, "winding must not matter"


@pytest.mark.parametrize("yaw_deg", [0.0, 17.0, 45.0, 73.0, 120.0])
def test_minimum_area_rectangle_is_orientation_independent(yaw_deg):
    """A rotated room must not report an inflated length and width."""
    base = np.array([[0.0, 0.0], [4.2, 0.0], [4.2, 3.1], [0.0, 3.1]])
    angle = np.deg2rad(yaw_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    length, width, _ = minimum_area_rectangle(base @ rotation.T)
    assert abs(length - 4.2) < 1e-6
    assert abs(width - 3.1) < 1e-6


# --------------------------------------------------------------------------
# confidence rules
# --------------------------------------------------------------------------

def test_confidence_never_claims_calibration(rules):
    assert rules.raw["calibrated"] is False
    label = rules.label("directly_observed", 0.01, 0.9, 50)
    assert label["calibrated"] is False
    assert label["label"] == "high"
    assert label["ruleTriggered"] == "high-direct-strong-support"


@pytest.mark.parametrize("state", ["inferred", "unresolved"])
def test_unsupported_geometry_is_unresolved_not_low(state, rules):
    """Missing evidence is not weak evidence; it must not be labelled."""
    assert rules.label(state, 0.001, 1.0, 999)["label"] == "unresolved"


def test_missing_inputs_cannot_produce_a_label(rules):
    assert rules.label("directly_observed", None, 0.9, 50)["label"] == "unresolved"
    assert rules.label("directly_observed", 0.01, None, 50)["label"] == "unresolved"
    assert rules.label("directly_observed", 0.01, 0.9, None)["label"] == "unresolved"


def test_each_evidence_dimension_can_demote_a_label(rules):
    assert rules.label("directly_observed", 0.03, 0.9, 50)["label"] == "medium"
    assert rules.label("directly_observed", 0.01, 0.3, 50)["label"] == "medium"
    assert rules.label("directly_observed", 0.01, 0.9, 4)["label"] == "medium"
    assert rules.label("directly_observed", 0.2, 0.9, 50)["label"] == "low"
    assert rules.label("directly_observed", 0.01, 0.05, 50)["label"] == "low"


def test_every_label_records_the_inputs_that_caused_it(rules):
    label = rules.label("directly_observed", 0.02, 0.6, 12)
    for key in ("observationState", "rmsResidual_m", "coverageFraction",
                "contributingFrames"):
        assert key in label["inputs"]
    assert label["rulesVersion"] == "confidence_rules_v0.1"


def test_calibrated_rules_are_refused(tmp_path):
    raw = json.loads((REPO_ROOT / "config" / "confidence_rules_v0.1.json").read_text())
    raw["calibrated"] = True
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfidenceRulesError, match="must declare calibrated=false"):
        load_confidence_rules(path)


# --------------------------------------------------------------------------
# envelope on a synthetic room of known size
# --------------------------------------------------------------------------

def test_synthetic_room_measurements_match_the_declared_dimensions(config, rules):
    cloud = _synthetic_room()
    planes = extract_planes(cloud, config)
    envelope = build_envelope(planes, _floor_points(cloud, planes, config), config)

    length, width, _ = minimum_area_rectangle(envelope.footprint)
    assert abs(length - ROOM_LENGTH) < 0.05, f"length {length:.3f} vs {ROOM_LENGTH}"
    assert abs(width - ROOM_WIDTH) < 0.05, f"width {width:.3f} vs {ROOM_WIDTH}"
    assert abs(envelope.height_m - ROOM_HEIGHT) < 0.02
    assert abs(envelope.area_m2 - ROOM_LENGTH * ROOM_WIDTH) < 0.35


def test_a_closed_synthetic_room_needs_no_inferred_closure(config):
    cloud = _synthetic_room()
    planes = extract_planes(cloud, config)
    envelope = build_envelope(planes, _floor_points(cloud, planes, config), config)
    assert envelope.diagnostics["inferredEdgeCount"] == 0
    assert envelope.diagnostics["observedPerimeterFraction"] > 0.99


def test_a_surface_standing_inside_the_room_is_not_a_boundary_wall(config):
    """A wardrobe front cuts part of the floor away; a wall does not."""
    cloud = _synthetic_room()
    planes = extract_planes(cloud, config)
    floor_points = _floor_points(cloud, planes, config)
    kept, excluded = select_boundary_walls(planes, floor_points, config)

    for wall in kept:
        inside = wall.signed_distance(floor_points) >= -config.get(
            "envelope_boundary_tolerance_m")
        assert inside.mean() >= config.get("min_floor_interior_fraction")
    for record in excluded:
        assert record["reason"]


def test_envelope_records_why_each_wall_was_dropped(config):
    cloud = _synthetic_room()
    planes = extract_planes(cloud, config)
    envelope = build_envelope(planes, _floor_points(cloud, planes, config), config)
    for record in envelope.excluded_walls:
        assert record["wallId"]
        assert "reason" in record and record["reason"]


# --------------------------------------------------------------------------
# canonical model
# --------------------------------------------------------------------------

def _synthetic_model(config, rules, tmp_path):
    source = __import__("pipeline.tests.synthetic", fromlist=["synthetic"])
    fixture = source.make_arkitscenes(tmp_path / "src", frames=4, pose_rate=1)
    ARKitScenesConnector(fixture).normalize(tmp_path / "cap")
    from pipeline.contracts.normalized_capture import NormalizedCapture
    capture = NormalizedCapture.read(tmp_path / "cap")

    cloud = _synthetic_room()
    planes = extract_planes(cloud, config)
    envelope = build_envelope(planes, _floor_points(cloud, planes, config), config)
    return build_spatial_model(capture, cloud, envelope, config, rules).model


def test_synthetic_model_passes_schema_and_coherence(config, rules, tmp_path):
    assert validate_model(_synthetic_model(config, rules, tmp_path)) == []


def test_model_keeps_track_b_and_c_arrays_empty(config, rules, tmp_path):
    model = _synthetic_model(config, rules, tmp_path)
    assert model["damage"] == []
    assert model["scope"] == []
    for surface in model["surfaces"]:
        assert surface["damage"] == []


def test_only_geometry_produces_metric_values(config, rules, tmp_path):
    model = _synthetic_model(config, rules, tmp_path)
    assert model["measurements"]
    for measurement in model["measurements"]:
        assert measurement["producer"] == "geometry_pipeline"


def test_every_surface_carries_its_algorithm_and_config_hash(config, rules, tmp_path):
    model = _synthetic_model(config, rules, tmp_path)
    for surface in model["surfaces"]:
        provenance = surface["provenance"]
        assert provenance["algorithm"]
        assert provenance["geometryConfigId"] == "geometry_config_v0.1"
        assert provenance["geometryConfigHash"] == config.sha256


def test_model_records_both_config_hashes(config, rules, tmp_path):
    provenance = _synthetic_model(config, rules, tmp_path)["provenance"]
    assert provenance["geometryConfigHash"] == config.sha256
    assert provenance["confidenceRulesHash"] == rules.sha256
    assert "not calibrated probabilities" in provenance["confidenceStatement"]


def test_reported_area_must_agree_with_the_footprint(config, rules, tmp_path):
    model = _synthetic_model(config, rules, tmp_path)
    for measurement in model["measurements"]:
        if measurement["type"] == "floor_area":
            measurement["value_m"] = measurement["value_m"] + 1.0
    problems = validate_model(model)
    assert any("disagrees with its own footprint" in p for p in problems)


def test_validator_catches_a_dangling_measurement_reference(config, rules, tmp_path):
    model = _synthetic_model(config, rules, tmp_path)
    model["measurements"][0]["entityId"] = "wall-999"
    assert any("resolves to no surface or room" in p for p in validate_model(model))


def test_validator_catches_confidence_asserted_on_inferred_geometry(config, rules, tmp_path):
    model = _synthetic_model(config, rules, tmp_path)
    model["surfaces"][0]["observationState"] = "inferred"
    model["surfaces"][0]["confidence"]["label"] = "high"
    assert any("cannot carry confidence" in p for p in validate_model(model))


def test_validator_refuses_a_non_geometry_producer(config, rules, tmp_path):
    model = _synthetic_model(config, rules, tmp_path)
    model["measurements"][0]["producer"] = "vision_model"
    assert validate_model(model)


def test_validator_refuses_populated_damage_or_scope(config, rules, tmp_path):
    model = _synthetic_model(config, rules, tmp_path)
    model["damage"].append({"damageId": "damage-001"})
    assert any("must remain empty" in p for p in validate_model(model))


# --------------------------------------------------------------------------
# real fixture, end to end
# --------------------------------------------------------------------------

@requires_fixture
def test_real_fixture_produces_a_valid_model(tmp_path):
    ARKitScenesConnector(PRIMARY_FIXTURE, stride=40).normalize(tmp_path / "cap")
    result = run_geometry(tmp_path / "cap", frame_stride=2)
    assert validate_model(result.model) == []

    measurements = {m["type"]: m for m in result.model["measurements"]}
    height = measurements["room_height"]["value_m"]
    area = measurements["floor_area"]["value_m"]
    assert 2.0 < height < 3.5, f"implausible ceiling height {height}"
    assert 4.0 < area < 80.0, f"implausible floor area {area}"
    assert measurements["room_length"]["value_m"] >= measurements["room_width"]["value_m"]


@requires_fixture
def test_unclosed_sides_are_reported_as_inferred(tmp_path):
    """The primary scene sees floor beyond its walls; that must show as inferred."""
    ARKitScenesConnector(PRIMARY_FIXTURE, stride=40).normalize(tmp_path / "cap")
    result = run_geometry(tmp_path / "cap", frame_stride=2)
    inferred = [s for s in result.model["surfaces"]
                if s["observationState"] == "inferred"]
    for surface in inferred:
        assert surface["confidence"]["label"] == "unresolved"
        assert surface["provenance"]["sourcePlaneId"] is None
    fraction = result.envelope.diagnostics["observedPerimeterFraction"]
    assert 0.0 < fraction <= 1.0


@requires_fixture
def test_regeneration_is_deterministic(tmp_path):
    """The frozen configuration is rerun on the final capture; it must not drift."""
    ARKitScenesConnector(PRIMARY_FIXTURE, stride=60).normalize(tmp_path / "cap")
    first = run_geometry(tmp_path / "cap", frame_stride=4).model
    second = run_geometry(tmp_path / "cap", frame_stride=4).model
    for document in (first, second):
        document["provenance"].pop("generatedUtc")
        document.pop("modelId")
    assert first == second


@requires_fixture
def test_model_contains_no_hardcoded_scene_identity(tmp_path):
    """Geometry must not know which scene it processed."""
    import pipeline.geometry as geometry_package

    root = Path(geometry_package.__file__).parent
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "47333462" not in text, f"{path.name} names a fixture scene"
        assert "41418135" not in text, f"{path.name} names a fixture scene"
        assert "lowres_depth" not in text, f"{path.name} names a source-specific path"
