"""Benchmark machinery and public-reference comparison tests.

Two things are being protected here. The arithmetic must be right, because it
is the only place the POC turns geometry into an accuracy claim. And the
development fixtures must stay unmistakably separate from real ground truth,
because the failure mode that would matter most is a fabricated number reaching
a report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.benchmark.compare import ASSIGNMENT_GATES, compare, summarise
from pipeline.benchmark.ground_truth import (
    GroundTruthError,
    load_ground_truth,
)
from pipeline.benchmark.reference import (
    ReferenceError,
    extract_reference_geometry,
    load_reference_config,
    read_pose,
)
from pipeline.benchmark.run_reference import build_report, render_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CSV = REPO_ROOT / "samples/benchmark_fixtures/development_ground_truth.csv"
VISIT_DIR = REPO_ROOT / "samples/arkitscenes/laser_scanner_point_clouds/467138"
MODEL = REPO_ROOT / "outputs/dev_47333462/spatial_model.json"

requires_reference = pytest.mark.skipif(
    not (VISIT_DIR.is_dir() and any(VISIT_DIR.glob("*.ply"))),
    reason="FARO reference scans are not present")
requires_model = pytest.mark.skipif(
    not MODEL.is_file(), reason="run the geometry pipeline first")


# --------------------------------------------------------------------------
# ground-truth parsing — the same path real tape data will take
# --------------------------------------------------------------------------

def test_repeated_readings_reduce_to_their_median():
    truth = load_ground_truth(FIXTURE_CSV)
    length = truth.by_type("room_length")
    assert length.value_m == 4.000
    assert abs(length.spread_m - 0.015) < 1e-9


def test_a_row_with_no_readings_stays_unresolved():
    """A missing measurement must never be filled in from the model."""
    truth = load_ground_truth(FIXTURE_CSV)
    door = truth.by_type("opening_width")
    assert door.value_m is None
    assert door.spread_m is None


def test_a_single_reading_has_no_spread():
    truth = load_ground_truth(FIXTURE_CSV)
    area = truth.by_type("floor_area")
    assert area.value_m == 12.0
    assert area.spread_m is None


def test_development_fixtures_are_flagged_as_such():
    truth = load_ground_truth(FIXTURE_CSV)
    assert truth.contains_development_fixtures is True
    assert all(m.is_development_fixture for m in truth.measurements)


def test_fixture_ids_cannot_collide_with_real_measurements():
    truth = load_ground_truth(FIXTURE_CSV)
    model_ids = {"measurement-room-length", "measurement-room-width",
                 "measurement-room-height", "measurement-floor-area"}
    assert not {m.measurement_id for m in truth.measurements} & model_ids


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(GroundTruthError, match="does not exist"):
        load_ground_truth(tmp_path / "absent.csv")


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("measurement_id,tape_1_m\nx,1.0\n")
    with pytest.raises(GroundTruthError, match="missing required column"):
        load_ground_truth(path)


def test_a_non_numeric_reading_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("measurement_id,type,tape_1_m\nx,room_length,about four\n")
    with pytest.raises(GroundTruthError, match="is not a number"):
        load_ground_truth(path)


def test_a_nonpositive_reading_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("measurement_id,type,tape_1_m\nx,room_length,-2.0\n")
    with pytest.raises(GroundTruthError, match="must be positive"):
        load_ground_truth(path)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("measurement_id,type,tape_1_m\nx,room_length,4.0\nx,room_width,3.0\n")
    with pytest.raises(GroundTruthError, match="repeats measurement_id"):
        load_ground_truth(path)


def test_an_empty_file_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("measurement_id,type,tape_1_m\n")
    with pytest.raises(GroundTruthError, match="no measurements"):
        load_ground_truth(path)


# --------------------------------------------------------------------------
# error arithmetic and the assignment gates
# --------------------------------------------------------------------------

def test_signed_absolute_and_percent_error_are_computed_correctly():
    result = compare("m", "room_length", reference_m=4.0, model_m=4.05)
    assert abs(result.signed_error_m - 0.05) < 1e-9
    assert abs(result.absolute_error_cm - 5.0) < 1e-9
    assert abs(result.percent_error - 1.25) < 1e-9


def test_error_sign_is_preserved():
    assert compare("m", "room_length", 4.0, 3.9).signed_error_m < 0
    assert compare("m", "room_length", 4.0, 4.1).signed_error_m > 0


@pytest.mark.parametrize("kind,reference,model,expected", [
    # 1% or 2 cm, whichever is larger. On 4 m that is 4 cm.
    ("room_length", 4.0, 4.039, "pass"),
    ("room_length", 4.0, 4.041, "fail"),
    # On a short 1 m wall the 2 cm floor applies, not 1%.
    ("wall_length", 1.0, 1.019, "pass"),
    ("wall_length", 1.0, 1.021, "fail"),
    # Ceiling height is an absolute 1.5 cm regardless of size.
    ("room_height", 2.6, 2.614, "pass"),
    ("room_height", 2.6, 2.616, "fail"),
    # Floor area is 2%, relative only.
    ("floor_area", 20.0, 20.39, "pass"),
    ("floor_area", 20.0, 20.41, "fail"),
    ("opening_width", 0.9, 0.919, "pass"),
    ("opening_width", 0.9, 0.921, "fail"),
])
def test_assignment_gates_are_applied_as_written(kind, reference, model, expected):
    assert compare("m", kind, reference, model).result == expected


def test_a_missing_value_is_not_comparable_rather_than_a_pass():
    """A benchmark that drops its gaps is not a benchmark."""
    assert compare("m", "room_height", None, 2.6).result == "not_comparable"
    assert compare("m", "room_height", 2.6, None).result == "not_comparable"
    assert "no reference value" in compare("m", "room_height", None, 2.6).note


def test_every_gate_in_the_table_has_a_description():
    for kind, gate in ASSIGNMENT_GATES.items():
        assert gate["description"], kind
        assert gate["mode"] in {"larger", "relative", "absolute"}


def test_summary_counts_do_not_hide_failures():
    comparisons = [
        compare("a", "room_height", 2.6, 2.7),      # fail
        compare("b", "room_length", 4.0, 4.0),      # pass
        compare("c", "room_width", None, 3.0),      # not comparable
    ]
    summary = summarise(comparisons)
    assert summary == {
        "comparisons": 3, "comparable": 2, "notComparable": 1, "gated": 2,
        "passed": 1, "failed": 1,
        "meanAbsoluteError_cm": 5.0, "maxAbsoluteError_cm": 10.0,
    }


# --------------------------------------------------------------------------
# the independent FARO reference
# --------------------------------------------------------------------------

def test_reference_config_is_versioned_separately_from_geometry():
    """A reference that inherited the pipeline's thresholds would be circular."""
    from pipeline.geometry.config import load_geometry_config

    reference = load_reference_config()
    geometry = load_geometry_config()
    assert reference.config_id != geometry.config_id
    assert reference.sha256 != geometry.sha256
    assert set(reference.raw["parameters"]) & set(geometry.raw["parameters"]) == set()


@requires_reference
def test_scan_poses_are_levelled_and_read_correctly():
    poses = sorted(VISIT_DIR.glob("*_pose.txt"))
    assert poses
    for path in poses:
        matrix = read_pose(path)
        assert matrix.shape == (4, 4)
        # A tripod-levelled scanner preserves the visit frame's vertical axis.
        assert abs(abs(matrix[2, 2]) - 1.0) < 0.01


@requires_reference
def test_reference_extraction_yields_a_plausible_storey_height():
    reference = extract_reference_geometry(VISIT_DIR)
    config = load_reference_config()
    assert config.get("min_storey_height_m") <= reference.separation_m \
        <= config.get("max_storey_height_m")
    assert reference.point_count > 100_000
    assert len(reference.scans_used) == 3
    assert reference.diagnostics["maxScanTiltDeg"] < config.get(
        "level_axis_tolerance_deg")


@requires_reference
def test_reference_extraction_is_deterministic():
    first = extract_reference_geometry(VISIT_DIR)
    second = extract_reference_geometry(VISIT_DIR)
    assert first.separation_m == second.separation_m


def test_a_missing_visit_directory_is_reported(tmp_path):
    with pytest.raises(ReferenceError, match="contains no FARO scans"):
        extract_reference_geometry(tmp_path)


def test_a_scan_without_a_pose_is_refused(tmp_path):
    (tmp_path / "1.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\n"
                                     b"element vertex 0\nend_header\n")
    with pytest.raises(ReferenceError, match="no companion pose file"):
        extract_reference_geometry(tmp_path)


# --------------------------------------------------------------------------
# the report itself
# --------------------------------------------------------------------------

@requires_reference
@requires_model
def test_report_is_labelled_as_a_development_reference():
    report = build_report(json.loads(MODEL.read_text()), VISIT_DIR,
                          "47333462", "PRIMARY_TUNING")
    assert report["label"] == "PUBLIC DEVELOPMENT REFERENCE"
    assert report["isFinalBenchmark"] is False
    assert "not a tape-measure benchmark" in report["statement"]


@requires_reference
@requires_model
def test_report_states_why_horizontal_quantities_are_not_comparable():
    report = build_report(json.loads(MODEL.read_text()), VISIT_DIR,
                          "47333462", "PRIMARY_TUNING")
    non_comparable = [r for r in report["comparisons"] if r["result"] == "not_comparable"]
    assert non_comparable, "the report must not silently omit what it cannot compare"
    for row in non_comparable:
        assert "registration" in row["note"]


@requires_reference
@requires_model
def test_report_reports_a_failing_gate_rather_than_hiding_it():
    report = build_report(json.loads(MODEL.read_text()), VISIT_DIR,
                          "47333462", "PRIMARY_TUNING")
    height = next(r for r in report["comparisons"] if r["type"] == "room_height")
    assert height["result"] in {"pass", "fail"}
    assert height["reference_m"] is not None
    assert height["absoluteError_cm"] is not None
    markdown = render_markdown(report)
    assert f"**{height['result']}**" in markdown


@requires_reference
@requires_model
def test_report_records_the_parameter_set_that_produced_it():
    report = build_report(json.loads(MODEL.read_text()), VISIT_DIR,
                          "47333462", "PRIMARY_TUNING")
    assert report["frozenParameterSet"]["geometryConfigHash"]
    assert report["referenceSource"]["extractionConfigHash"]
    assert report["referenceSource"]["independentOfDeviceUnderTest"] is True
    assert "same reconstruction stack" in report["referenceSource"]["whyNotTheArkitMesh"]


@requires_reference
@requires_model
def test_secondary_scene_is_reported_without_a_reference_claim():
    secondary_path = REPO_ROOT / "outputs/dev_41418135/spatial_model.json"
    if not secondary_path.is_file():
        pytest.skip("secondary model not generated")
    report = build_report(
        json.loads(MODEL.read_text()), VISIT_DIR, "47333462", "PRIMARY_TUNING",
        {"41418135": json.loads(secondary_path.read_text())})
    entry = report["secondaryValidation"]["scenes"]["41418135"]
    assert entry["referenceAvailable"] is False
    assert "never used to change a parameter" in report["secondaryValidation"]["policy"]


@requires_reference
@requires_model
def test_both_scenes_ran_on_the_same_configuration():
    """Anti-overfitting: one config across scenes, no per-scene retuning."""
    secondary_path = REPO_ROOT / "outputs/dev_41418135/spatial_model.json"
    if not secondary_path.is_file():
        pytest.skip("secondary model not generated")
    primary = json.loads(MODEL.read_text())["provenance"]["geometryConfigHash"]
    secondary = json.loads(secondary_path.read_text())["provenance"]["geometryConfigHash"]
    assert primary == secondary
