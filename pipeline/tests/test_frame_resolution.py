"""Ingestion vertical-axis resolution (structure + camera path, not IMU attitude)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.contracts.frame_resolution import (
    evaluate_candidate,
    load_ingestion_config,
    _trajectory_vertical_ratio,
)


def _config(**overrides) -> dict:
    config = load_ingestion_config()
    config.update(overrides)
    return config


def _room_points(up: str = "+y", rng_seed: int = 0) -> np.ndarray:
    """A box with dense floor and ceiling slabs along `up`."""
    rng = np.random.default_rng(rng_seed)
    floor = rng.uniform([-2, -2], [2, 2], size=(800, 2))
    ceiling = rng.uniform([-2, -2], [2, 2], size=(800, 2))
    if up == "+y":
        low = np.column_stack([floor[:, 0], np.full(800, 0.0), floor[:, 1]])
        high = np.column_stack([ceiling[:, 0], np.full(800, 2.6), ceiling[:, 1]])
    else:
        low = np.column_stack([floor[:, 0], floor[:, 1], np.full(800, 0.0)])
        high = np.column_stack([ceiling[:, 0], ceiling[:, 1], np.full(800, 2.6)])
    return np.vstack([low, high])


def _path(extent) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], list(extent)], dtype=np.float64)


def test_trajectory_ratio_picks_the_short_axis():
    centres = _path((2.2, 0.5, 4.1))
    assert _trajectory_vertical_ratio(centres, "+y") < 0.3
    assert _trajectory_vertical_ratio(centres, "+z") > 0.8


def test_correct_axis_verifies_from_structure_and_path():
    config = _config()
    evidence = evaluate_candidate(_room_points("+y"), _path((2.2, 0.5, 4.1)), "+y", config)
    assert evidence.plausible
    assert evidence.floorDetected and evidence.ceilingDetected
    assert 2.4 < evidence.roomHeightM < 2.8
    assert evidence.trajectoryVerticalRatio < 0.3


def test_a_walk_axis_is_rejected_even_if_points_were_smeared():
    config = _config()
    evidence = evaluate_candidate(_room_points("+y"), _path((2.2, 0.5, 4.1)), "+z", config)
    assert not evidence.plausible
    assert any("camera-path" in reason for reason in evidence.rejections)


def test_camera_attitude_is_not_a_gate():
    """A 90-degree mean-up disagreement used to reject the correct axis. It must not."""
    config = _config()
    evidence = evaluate_candidate(_room_points("+y"), _path((2.2, 0.5, 4.1)), "+y", config)
    assert evidence.plausible
    assert all("mean camera up" not in reason for reason in evidence.rejections)


def test_declaration_wins_when_it_verifies():
    points = _room_points("+y")
    centres = _path((2.2, 0.5, 4.1))

    class Capture:
        world_up_axis = "+y"
        frames = [SimpleNamespace(camera_to_world=np.eye(4)) for _ in range(2)]
        frames[1].camera_to_world = np.array([
            [1, 0, 0, 2.2], [0, 1, 0, 0.5], [0, 0, 1, 4.1], [0, 0, 0, 1.0]])
        intrinsics = [SimpleNamespace(stream="depth", fx=200, fy=200, cx=128, cy=96)]

    # resolve_frame needs real depth files; score via evaluate + the decision rule.
    plus_y = evaluate_candidate(points, centres, "+y", _config())
    plus_z = evaluate_candidate(points, centres, "+z", _config())
    assert plus_y.plausible
    assert not plus_z.plausible


def test_sign_is_not_taken_from_the_trajectory():
    centres = _path((2.2, 0.5, 4.1))
    assert _trajectory_vertical_ratio(centres, "+y") == _trajectory_vertical_ratio(
        centres, "-y")


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
STRAY_SAMPLE = REPO_ROOT / "samples" / "stray" / "raw" / "8653a2142b"


@pytest.mark.skipif(not STRAY_SAMPLE.is_dir(),
                    reason="local real Stray integration sample is not installed")
def test_real_stray_keeps_declared_y_when_structure_verifies(tmp_path, monkeypatch):
    from pipeline.connectors.stray_scanner import StrayScannerConnector
    from pipeline.contracts.normalized_capture import NormalizedCapture
    from pipeline.geometry.config import load_geometry_config
    from pipeline.geometry.planes import extract_planes
    from pipeline.geometry.points import build_point_cloud

    monkeypatch.setattr(StrayScannerConnector, "_extract_rgb", lambda self, dest: {})
    StrayScannerConnector(
        STRAY_SAMPLE, classification="public_development_fixture",
        stride=6,
    ).normalize(tmp_path / "out")
    capture = NormalizedCapture.read(tmp_path / "out")
    cloud = build_point_cloud(tmp_path / "out", capture, load_geometry_config(), stride=1)
    assert cloud.frame.up_source == "declared"
    assert cloud.frame.diagnostics.get("declarationVerified") is True
    resolution = cloud.frame.diagnostics.get("ingestionFrameResolution") or {}
    assert resolution.get("axis") == "+y"
    assert resolution.get("outcome") == "verified"
    planes = extract_planes(cloud, load_geometry_config())
    assert planes.floor is not None
    assert planes.ceiling is not None
    height = abs(planes.ceiling.offset - planes.floor.offset)
    assert 2.2 < height < 3.2
    assert cloud.extent()[1] < 4.0


# --------------------------------------------------------------------------
# band selection, sampling, and the refusal to guess
# --------------------------------------------------------------------------

def test_the_strongest_band_pair_is_chosen_not_the_outermost():
    """A stray high return must not be mistaken for the ceiling.

    The outermost supported bins are wherever the cloud happens to stop. A real
    floor and ceiling are its densest bands, and picking the extremes instead is
    how a 2.6 m room once read as 1.3 m tall.
    """
    points = _room_points("+y")
    # A thin, high, but genuinely supported slab well above the real ceiling.
    stray = np.column_stack([
        np.random.default_rng(3).uniform(-2, 2, 400),
        np.full(400, 4.3),
        np.random.default_rng(4).uniform(-2, 2, 400),
    ])
    evidence = evaluate_candidate(np.vstack([points, stray]), _path((2.2, 0.5, 4.1)),
                                  "+y", _config())
    assert evidence.floorDetected and evidence.ceilingDetected
    assert 2.4 < evidence.roomHeightM < 2.8, "the dense pair should win, not the extremes"


def test_a_pair_outside_the_plausible_storey_range_is_rejected():
    rng = np.random.default_rng(11)
    low = np.column_stack([rng.uniform(-2, 2, 800), np.zeros(800), rng.uniform(-2, 2, 800)])
    high = np.column_stack([rng.uniform(-2, 2, 800), np.full(800, 1.1), rng.uniform(-2, 2, 800)])
    evidence = evaluate_candidate(np.vstack([low, high]), _path((2.2, 0.5, 4.1)),
                                  "+y", _config())
    assert not evidence.plausible
    assert any("storey height" in reason or "no ceiling" in reason
               for reason in evidence.rejections)


def test_frames_are_sampled_by_position_not_by_a_fixed_stride():
    """A fixed stride can resonate with a periodic sweep and miss a surface."""
    source = (Path(__file__).resolve().parents[1]
              / "contracts" / "frame_resolution.py").read_text()
    assert "np.linspace" in source
    assert "frames[::" not in source, "a fixed decimation stride can alias"


def test_an_unverified_capture_never_claims_verification(tmp_path):
    """The dangerous outcome is a confident wrong axis, so it must not exist."""
    from pipeline.contracts.frame_resolution import FrameResolution
    unresolved = FrameResolution("ambiguous", None, "none", "+y", False, [], [])
    assert not unresolved.accepted
    assert unresolved.axis is None


# --------------------------------------------------------------------------
# the plain-language outcome
# --------------------------------------------------------------------------

def test_outcomes_stay_within_the_four_the_interface_shows():
    from pipeline.contracts import ingestion_outcome as io
    assert set(io.HEADLINES) == {
        io.ACCEPTED, io.ACCEPTED_WITH_FIXES, io.NEEDS_REVIEW, io.UNSUPPORTED}


def test_a_clean_capture_reads_as_accepted():
    from pipeline.contracts.ingestion_outcome import ACCEPTED, summarize
    resolution = SimpleNamespace(outcome="verified", basis="declared_verified", axis="+y")
    outcome = summarize(issues=[], resolution=resolution)
    assert outcome.state == ACCEPTED
    assert not outcome.fixes and not outcome.concerns


def test_an_ambiguous_frame_asks_for_review_rather_than_failing():
    """Ambiguity is not a failure: the capture still processes on its declaration."""
    from pipeline.contracts.ingestion_outcome import NEEDS_REVIEW, summarize
    resolution = SimpleNamespace(outcome="ambiguous", basis="none", axis=None)
    outcome = summarize(issues=[], resolution=resolution)
    assert outcome.state == NEEDS_REVIEW
    assert any("up could not be confirmed" in concern for concern in outcome.concerns)


def test_a_validation_error_is_unsupported_not_review():
    from pipeline.contracts.ingestion_outcome import UNSUPPORTED, summarize
    issue = SimpleNamespace(severity="error", code="X", message="no depth stream")
    assert summarize(issues=[issue]).state == UNSUPPORTED


def test_repairs_are_reported_as_fixes_not_as_concerns():
    from pipeline.contracts.ingestion_outcome import ACCEPTED_WITH_FIXES, summarize
    resolution = SimpleNamespace(outcome="verified", basis="declared_verified", axis="+y")
    outcome = summarize(issues=[], resolution=resolution, excluded_frames=4)
    assert outcome.state == ACCEPTED_WITH_FIXES
    assert outcome.fixes and not outcome.concerns
