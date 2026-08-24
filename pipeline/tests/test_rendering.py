"""2D plan and 3D model tests.

The requirement that matters most here is that the drawing, the model, and the
JSON cannot disagree. So the tests do not check that the SVG looks reasonable;
they check that every number on it came from a measurement record, and that
every 3D group name is a surface ID.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pipeline.geometry.model import build_spatial_model
from pipeline.rendering.floorplan import render_floorplan
from pipeline.rendering.model_3d import build_model_3d, write_model_3d

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED = REPO_ROOT / "outputs" / "dev_47333462" / "spatial_model.json"

requires_generated = pytest.mark.skipif(
    not GENERATED.is_file(),
    reason="run `python3 -m pipeline.geometry.run` first")


@pytest.fixture(scope="module")
def model():
    return json.loads(GENERATED.read_text())


def _texts(svg: str) -> list[str]:
    root = ET.fromstring(svg)
    return [e.text for e in root.iter() if e.tag.endswith("text") and e.text]


# --------------------------------------------------------------------------
# the plan is valid, self-describing SVG
# --------------------------------------------------------------------------

@requires_generated
def test_floorplan_is_well_formed_svg(model):
    root = ET.fromstring(render_floorplan(model))
    assert root.tag.endswith("svg")
    assert float(root.get("width")) > 0 and float(root.get("height")) > 0


@requires_generated
def test_floorplan_carries_scale_and_a_legend(model):
    texts = _texts(render_floorplan(model))
    assert "SCALE" in texts
    assert "1 m" in texts
    assert "OBSERVATION STATE" in texts
    assert "ROOM" in texts


@requires_generated
def test_development_input_is_badged_as_such(model):
    """A reviewer must never mistake a fixture render for final evidence."""
    assert "DEVELOPMENT FIXTURE" in _texts(render_floorplan(model))


@requires_generated
def test_final_capture_is_badged_differently(model):
    final = json.loads(json.dumps(model))
    final["scan"]["classification"] = "final_private_capture"
    texts = _texts(render_floorplan(final))
    assert "FINAL CAPTURE" in texts
    assert "DEVELOPMENT FIXTURE" not in texts


# --------------------------------------------------------------------------
# labels must match the JSON exactly
# --------------------------------------------------------------------------

@requires_generated
def test_every_room_dimension_on_the_plan_matches_its_measurement(model):
    texts = _texts(render_floorplan(model))
    measurements = {m["type"]: m for m in model["measurements"]}
    for kind in ("room_length", "room_width", "room_height"):
        expected = f"{measurements[kind]['value_m']:.2f} m"
        assert expected in texts, f"{kind} label {expected} missing from the plan"
    area = measurements["floor_area"]
    assert f"{area['value_m']:.2f} m²" in texts


@requires_generated
def test_every_wall_label_matches_its_surface_record(model):
    texts = _texts(render_floorplan(model))
    walls = [s for s in model["surfaces"] if s["type"] == "wall"]
    for surface in walls:
        expected = f"{surface['id']} · {surface['dimensions']['width_m']:.2f} m"
        assert expected in texts, f"missing wall label {expected}"


@requires_generated
def test_the_plan_invents_no_numbers(model):
    """Every metric value drawn must be traceable to the model document."""
    texts = _texts(render_floorplan(model))
    allowed = {f"{m['value_m']:.2f}" for m in model["measurements"]
               if m["value_m"] is not None}
    allowed |= {f"{s['dimensions']['width_m']:.2f}" for s in model["surfaces"]}
    allowed |= {f"{s['dimensions']['height_m']:.2f}" for s in model["surfaces"]}
    allowed |= {"1"}                      # the scale bar
    identifiers = ({s["id"] for s in model["surfaces"]}
                   | {str(model["scan"]["id"]), str(model["modelId"])})
    for text in texts:
        stripped = text
        for identifier in sorted(identifiers, key=len, reverse=True):
            stripped = stripped.replace(identifier, "")
        # Only metric values are checked; identifier suffixes are not numbers
        # the drawing is asserting about the room.
        for number in re.findall(r"\d+\.\d{2}|(?<![\d.])\d+(?![\d.])", stripped):
            assert number in allowed, f"'{number}' in '{text}' is not in the model"


@requires_generated
def test_observation_state_is_visible_on_the_plan(model):
    svg = render_floorplan(model)
    texts = _texts(svg)
    states = {s["observationState"] for s in model["surfaces"] if s["type"] == "wall"}
    for state in states:
        assert state.replace("_", " ") in texts
    if "inferred" in states:
        # Inferred closure must be visually distinct, not merely labelled.
        assert "stroke-dasharray" in svg


@requires_generated
def test_unresolved_openings_are_stated_not_hidden(model):
    texts = _texts(render_floorplan(model))
    unresolved = [o for o in model["openings"] if o["observationState"] == "unresolved"]
    if unresolved:
        assert any("none resolved" in t for t in texts)


def test_a_plan_with_unresolved_height_says_so(model=None):
    document = json.loads(GENERATED.read_text()) if GENERATED.is_file() else None
    if document is None:
        pytest.skip("no generated model")
    for measurement in document["measurements"]:
        if measurement["type"] == "room_height":
            measurement["value_m"] = None
            measurement["confidence"]["label"] = "unresolved"
    assert "unresolved" in _texts(render_floorplan(document))


# --------------------------------------------------------------------------
# the 3D model shares the JSON's identities
# --------------------------------------------------------------------------

@requires_generated
def test_obj_group_names_are_the_surface_ids(model):
    built = build_model_3d(model)
    groups = set(re.findall(r"^g (\S+)$", built.obj, flags=re.MULTILINE))
    expected = {s["id"] for s in model["surfaces"]}
    assert groups == expected, f"group/surface mismatch: {groups ^ expected}"


@requires_generated
def test_obj_faces_reference_real_vertices(model):
    built = build_model_3d(model)
    vertex_count = len(re.findall(r"^v ", built.obj, flags=re.MULTILINE))
    for face in re.findall(r"^f (.+)$", built.obj, flags=re.MULTILINE):
        indices = [int(i) for i in face.split()]
        assert len(indices) >= 3
        for index in indices:
            assert 1 <= index <= vertex_count


@requires_generated
def test_materials_encode_observation_state(model):
    built = build_model_3d(model)
    used = set(re.findall(r"^usemtl (\S+)$", built.obj, flags=re.MULTILINE))
    declared = set(re.findall(r"^newmtl (\S+)$", built.mtl, flags=re.MULTILINE))
    assert used <= declared
    states = {s["observationState"] for s in model["surfaces"]}
    if "inferred" in states and "directly_observed" in states:
        assert {"inferred", "observed"} <= used, (
            "inferred and observed surfaces must not share a material")


@requires_generated
def test_entity_map_covers_every_surface_with_matching_dimensions(model):
    built = build_model_3d(model)
    entities = built.entity_map["surfaces"]
    assert set(entities) == {s["id"] for s in model["surfaces"]}
    for surface in model["surfaces"]:
        entry = entities[surface["id"]]
        assert entry["dimensions"] == surface["dimensions"]
        assert entry["observationState"] == surface["observationState"]
        assert entry["confidenceLabel"] == surface["confidence"]["label"]


@requires_generated
def test_wall_heights_in_3d_match_the_reported_room_height(model):
    built = build_model_3d(model)
    heights = [float(v.split()[2]) for v in
               re.findall(r"^v .+$", built.obj, flags=re.MULTILINE)]
    measurements = {m["type"]: m for m in model["measurements"]}
    expected = measurements["room_height"]["value_m"]
    assert abs(max(heights) - expected) < 1e-6
    assert abs(min(heights)) < 1e-6, "the model must stand on y = 0"


@requires_generated
def test_3d_artifacts_are_written_together(tmp_path, model):
    entity_map = write_model_3d(model, tmp_path)
    for name in ("room_model.obj", "room_model.mtl", "room_model_entity_map.json"):
        assert (tmp_path / name).is_file(), name
    assert "mtllib room_model.mtl" in (tmp_path / "room_model.obj").read_text()
    assert entity_map["surfaceCount"] == len(model["surfaces"])


def test_a_model_without_a_height_emits_floor_only():
    """An unresolved height must not silently extrude a zero-height room."""
    document = json.loads(GENERATED.read_text()) if GENERATED.is_file() else None
    if document is None:
        pytest.skip("no generated model")
    for measurement in document["measurements"]:
        if measurement["type"] == "room_height":
            measurement["value_m"] = None
    built = build_model_3d(document)
    groups = set(re.findall(r"^g (\S+)$", built.obj, flags=re.MULTILINE))
    assert groups == {"floor-001"}
    assert "limitation" in built.entity_map


# --------------------------------------------------------------------------
# both products come from the document, not from a second geometry path
# --------------------------------------------------------------------------

def test_renderers_take_only_the_model_document():
    """Nothing in rendering may reach back into the cloud or the capture."""
    import pipeline.rendering as rendering

    root = Path(rendering.__file__).parent
    for path in root.glob("*.py"):
        text = path.read_text()
        for forbidden in ("build_point_cloud", "extract_planes", "NormalizedCapture",
                          "lowres_depth", "47333462"):
            assert forbidden not in text, f"{path.name} references {forbidden}"


@requires_generated
def test_rendering_is_deterministic(model):
    assert render_floorplan(model) == render_floorplan(model)
    assert build_model_3d(model).obj == build_model_3d(model).obj
