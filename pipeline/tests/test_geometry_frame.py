"""Coordinate, unit, and gravity normalization tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.connectors.arkitscenes import ARKitScenesConnector
from pipeline.contracts.normalized_capture import NormalizedCapture
from pipeline.geometry.config import ConfigError, load_geometry_config
from pipeline.geometry.frame import (
    CANONICAL_UP,
    FrameError,
    angle_between_deg,
    axis_vector,
    estimate_up_from_poses,
    resolve_canonical_frame,
    rotation_taking,
)
from pipeline.geometry.points import build_point_cloud, voxel_downsample
from pipeline.tests import synthetic

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_FIXTURE = REPO_ROOT / "samples/arkitscenes/raw/Training/47333462"

requires_fixture = pytest.mark.skipif(
    not PRIMARY_FIXTURE.is_dir(),
    reason="ARKitScenes primary fixture is not present; see docs/fixture_selection.md")


@pytest.fixture(scope="module")
def config():
    return load_geometry_config()


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def test_every_parameter_documents_its_provenance(config):
    for name, entry in config.raw["parameters"].items():
        assert entry["units"], f"{name} has no units"
        assert len(entry["rationale"]) > 40, f"{name} has no meaningful rationale"
        assert entry["sourceCategory"] in {
            "sensor_characteristic", "published_default", "public_dev_sweep",
            "assignment", "engineering_constraint"}, name
        assert entry["calibrated"] is False, name


def test_undocumented_parameters_are_refused(config):
    with pytest.raises(ConfigError, match="rather than hardcoding it"):
        config.get("wall_fudge_factor")


def test_config_hash_identifies_the_file(config):
    reloaded = load_geometry_config()
    assert reloaded.sha256 == config.sha256
    assert config.provenance("depth_max_m")["geometryConfigHash"] == config.sha256


def test_no_parameter_was_tuned_on_the_secondary_scene(config):
    secondary = config.raw["tuningPolicy"]["secondaryValidationScene"]
    for name, entry in config.raw["parameters"].items():
        assert secondary not in entry["tuningSceneIds"], (
            f"{name} lists the secondary validation scene as a tuning source")


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source", [
    [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, -1, 0], [0, 0, -1],
    [0.3, 0.4, 0.86], [-0.7, 0.1, 0.7],
])
def test_rotation_takes_source_onto_target(source):
    source = np.array(source, dtype=float)
    source /= np.linalg.norm(source)
    rotation = rotation_taking(source, CANONICAL_UP)
    assert np.allclose(rotation @ source, CANONICAL_UP, atol=1e-12)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_rotation_of_an_already_aligned_axis_is_the_identity():
    assert np.allclose(rotation_taking(CANONICAL_UP, CANONICAL_UP), np.eye(3))


def test_antiparallel_rotation_stays_right_handed():
    rotation = rotation_taking(-CANONICAL_UP, CANONICAL_UP)
    assert np.allclose(rotation @ -CANONICAL_UP, CANONICAL_UP, atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_horizontal_frame_is_left_alone():
    """The rotation is minimal: with +z already up, x and y are untouched."""
    rotation = rotation_taking(np.array([0.0, 0.0, 1.0]), CANONICAL_UP)
    assert np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------------
# gravity estimation
# --------------------------------------------------------------------------

def _handheld_poses(up: np.ndarray, count: int = 60, seed: int = 3) -> np.ndarray:
    """Poses for an operator panning around, holding the phone roughly upright."""
    rng = np.random.default_rng(seed)
    up = up / np.linalg.norm(up)
    poses = []
    for i in range(count):
        yaw = 2 * np.pi * i / count
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(up @ helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        east = np.cross(up, helper)
        east /= np.linalg.norm(east)
        north = np.cross(up, east)
        forward = np.cos(yaw) * east + np.sin(yaw) * north
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        # A few degrees of hand tilt, as a real operator would introduce.
        tilt = np.deg2rad(rng.normal(0.0, 4.0))
        down = -up * np.cos(tilt) + forward * np.sin(tilt)
        forward_c = np.cross(right, down)
        rotation = np.stack([right, down, forward_c], axis=1)
        pose = np.eye(4)
        pose[:3, :3] = rotation
        poses.append(pose)
    return np.array(poses)


@pytest.mark.parametrize("up", [[0, 1, 0], [0, 0, 1], [0, -1, 0], [0.1, 0.2, 0.97]])
def test_pose_estimate_recovers_the_up_axis(up):
    up = np.array(up, dtype=float)
    up /= np.linalg.norm(up)
    estimated, diagnostics = estimate_up_from_poses(_handheld_poses(up))
    assert angle_between_deg(estimated, up) < 2.0
    assert diagnostics["poseCount"] == 60


def test_pose_estimate_gets_the_sign_right():
    estimated, _ = estimate_up_from_poses(_handheld_poses(np.array([0.0, 0.0, 1.0])))
    assert estimated[2] > 0


# --------------------------------------------------------------------------
# frame resolution policy
# --------------------------------------------------------------------------

def test_declaration_is_kept_when_poses_agree():
    poses = _handheld_poses(np.array([0.0, 0.0, 1.0]))
    frame = resolve_canonical_frame(poses, "+z", "arkitscenes_world", 10.0, 20)
    assert frame.up_source == "declared"
    assert np.allclose(frame.source_up_axis, [0, 0, 1])
    assert frame.diagnostics["declaredVsEstimatedDeg"] < 2.0


def test_a_wrong_declaration_is_overridden_by_the_poses():
    """A source claiming the wrong axis must not silently tip the room over."""
    poses = _handheld_poses(np.array([0.0, 0.0, 1.0]))
    frame = resolve_canonical_frame(poses, "+y", "some_source", 10.0, 20)
    assert frame.up_source == "pose_estimate"
    assert angle_between_deg(frame.source_up_axis, np.array([0.0, 0.0, 1.0])) < 2.0
    assert "was rejected in favour of the estimate" in frame.diagnostics["note"]


def test_a_structurally_verified_declaration_is_not_overridden():
    """Pose-scatter may disagree; once structure has accepted the declaration, keep it."""
    poses = _handheld_poses(np.array([0.0, 0.0, 1.0]))
    frame = resolve_canonical_frame(
        poses, "+y", "stray_session", 10.0, 20, declaration_verified=True)
    assert frame.up_source == "declared"
    assert np.allclose(frame.source_up_axis, [0.0, 1.0, 0.0])
    assert frame.diagnostics["declarationVerified"] is True
    assert "diagnostic only" in frame.diagnostics["note"]


def test_declaration_is_trusted_when_there_are_too_few_poses():
    poses = _handheld_poses(np.array([0.0, 0.0, 1.0]), count=5)
    frame = resolve_canonical_frame(poses, "+z", "s", 10.0, 20)
    assert frame.up_source == "declared"
    assert frame.diagnostics["poseEstimate"]["skipped"] is True


def test_an_unknown_up_axis_is_not_guessed():
    poses = _handheld_poses(np.array([0.0, 0.0, 1.0]), count=5)
    with pytest.raises(FrameError, match="must not be guessed"):
        resolve_canonical_frame(poses, None, "s", 10.0, 20)


def test_missing_declaration_falls_back_to_the_estimate():
    poses = _handheld_poses(np.array([0.0, 0.0, 1.0]))
    frame = resolve_canonical_frame(poses, None, "s", 10.0, 20)
    assert frame.up_source == "pose_estimate"


def test_an_unrecognised_axis_name_is_rejected():
    with pytest.raises(FrameError, match="not a recognised axis"):
        axis_vector("up")


def test_frame_resolution_needs_poses():
    with pytest.raises(FrameError, match="without any camera poses"):
        resolve_canonical_frame(np.zeros((0, 4, 4)), "+z", "s", 10.0, 20)


def test_canonical_frame_maps_source_up_onto_plus_y():
    poses = _handheld_poses(np.array([0.0, 0.0, 1.0]))
    frame = resolve_canonical_frame(poses, "+z", "arkitscenes_world", 10.0, 20)
    assert np.allclose(frame.apply(np.array([[0.0, 0.0, 2.5]])), [[0.0, 2.5, 0.0]], atol=1e-9)
    assert frame.provenance()["canonicalFrame"]["upAxis"] == "y"
    assert frame.provenance()["floorOriginApplied"] is False


def test_floor_origin_shifts_only_the_vertical_axis():
    poses = _handheld_poses(np.array([0.0, 0.0, 1.0]))
    frame = resolve_canonical_frame(poses, "+z", "s", 10.0, 20)
    lowered = frame.with_floor_origin(1.25)
    moved = lowered.apply(np.array([[1.0, 2.0, 3.0]]))
    original = frame.apply(np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(moved - original, [[0.0, -1.25, 0.0]], atol=1e-12)
    assert lowered.floor_origin_applied is True
    assert np.allclose(lowered.source_to_canonical[:3, :3], frame.source_to_canonical[:3, :3])


def test_poses_transform_with_the_same_frame():
    poses = _handheld_poses(np.array([0.0, 0.0, 1.0]))
    frame = resolve_canonical_frame(poses, "+z", "s", 10.0, 20)
    canonical_pose = frame.apply_pose(poses[0])
    assert np.allclose(canonical_pose[:3, 3], frame.apply(poses[0][:3, 3][None])[0], atol=1e-12)
    assert np.isclose(np.linalg.det(canonical_pose[:3, :3]), 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# downsampling
# --------------------------------------------------------------------------

def test_voxel_downsample_collapses_a_dense_blob_and_keeps_its_centre():
    rng = np.random.default_rng(0)
    # Centred inside a voxel rather than on its corner, so the blob occupies one
    # cell; a corner-centred blob would legitimately straddle eight.
    blob = rng.normal(scale=0.001, size=(5000, 3)) + np.array([1.025, 2.025, 3.025])
    reduced = voxel_downsample(blob, 0.05)
    assert len(reduced) == 1
    assert np.allclose(reduced[0], blob.mean(axis=0), atol=1e-12)


def test_voxel_downsample_bounds_the_point_count_by_the_grid():
    rng = np.random.default_rng(1)
    points = rng.uniform(0.0, 0.1, size=(20000, 3))
    reduced = voxel_downsample(points, 0.05)
    assert len(reduced) <= 3 ** 3
    assert len(reduced) < len(points)


def test_voxel_downsample_keeps_separated_points():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert len(voxel_downsample(points, 0.01)) == 3


def test_voxel_downsample_rejects_a_nonpositive_size():
    with pytest.raises(ValueError, match="must be positive"):
        voxel_downsample(np.zeros((3, 3)), 0.0)


# --------------------------------------------------------------------------
# unprojection
# --------------------------------------------------------------------------

def test_synthetic_depth_lands_where_the_geometry_says_it_should(tmp_path, config):
    """Constant depth in front of an identity camera is a plane at that range.

    The source declares +z up, so canonical +y must carry the range.
    """
    source = synthetic.make_arkitscenes(
        tmp_path / "src", frames=4, pose_rate=1, depth_value_mm=1500)
    ARKitScenesConnector(source).normalize(tmp_path / "out")
    capture = NormalizedCapture.read(tmp_path / "out")
    cloud = build_point_cloud(tmp_path / "out", capture, config)

    assert cloud.frame.up_source == "declared"
    # Camera +z (range) becomes canonical +y.
    assert np.allclose(cloud.points[:, 1], 1.5, atol=1e-9)
    assert len(cloud) > 0


def test_out_of_range_depth_is_dropped_not_clamped(tmp_path, config):
    source = synthetic.make_arkitscenes(
        tmp_path / "src", frames=3, pose_rate=1, depth_value_mm=9000)
    ARKitScenesConnector(source).normalize(tmp_path / "out")
    capture = NormalizedCapture.read(tmp_path / "out")
    with pytest.raises(ValueError, match="no depth pixel survived filtering"):
        build_point_cloud(tmp_path / "out", capture, config)


def test_low_confidence_depth_is_discarded(tmp_path, config):
    source = synthetic.make_arkitscenes(tmp_path / "src", frames=3, pose_rate=1)
    for path in (source / "confidence").glob("*.png"):
        synthetic.write_confidence(path, 16, 12, level=0)
    ARKitScenesConnector(source).normalize(tmp_path / "out")
    capture = NormalizedCapture.read(tmp_path / "out")
    with pytest.raises(ValueError, match="no depth pixel survived filtering"):
        build_point_cloud(tmp_path / "out", capture, config)


def test_filter_losses_are_reported_not_hidden(tmp_path, config):
    source = synthetic.make_arkitscenes(tmp_path / "src", frames=4, pose_rate=1)
    ARKitScenesConnector(source).normalize(tmp_path / "out")
    capture = NormalizedCapture.read(tmp_path / "out")
    cloud = build_point_cloud(tmp_path / "out", capture, config)
    d = cloud.diagnostics
    assert d["pixelsExamined"] > 0
    assert d["retainedAfterNoReturnFilter"] >= d["retainedAfterRangeFilter"]
    assert d["retainedAfterRangeFilter"] >= d["retainedAfterConfidenceFilter"]
    assert 0.0 <= d["retainedFractionOfPixels"] <= 1.0
    assert cloud.config_provenance["geometryConfigId"] == "geometry_config_v0.1"


# --------------------------------------------------------------------------
# real fixture
# --------------------------------------------------------------------------

@requires_fixture
def test_real_fixture_gravity_declaration_survives_an_independent_check(tmp_path, config):
    ARKitScenesConnector(PRIMARY_FIXTURE, stride=30).normalize(tmp_path / "out")
    capture = NormalizedCapture.read(tmp_path / "out")
    cloud = build_point_cloud(tmp_path / "out", capture, config, stride=2)

    assert capture.world_up_axis == "+z"
    assert cloud.frame.up_source == "declared"
    assert cloud.frame.diagnostics["declaredVsEstimatedDeg"] < 5.0

    extent = cloud.extent()
    # A room is much wider than it is tall, and a dwelling's ceiling sits in a
    # narrow band. If the up axis were wrong, the vertical extent would be the
    # room's length instead.
    assert 2.0 < extent[1] < 3.5, f"implausible ceiling height {extent[1]:.2f} m"
    assert extent[0] > extent[1] and extent[2] > extent[1]


@requires_fixture
def test_canonical_points_agree_with_the_arkit_mesh_after_the_same_rotation(tmp_path, config):
    """Rotating the device's own mesh by the same transform must line up."""
    ARKitScenesConnector(PRIMARY_FIXTURE, stride=30).normalize(tmp_path / "out")
    capture = NormalizedCapture.read(tmp_path / "out")
    cloud = build_point_cloud(tmp_path / "out", capture, config, stride=2)

    with (tmp_path / "out" / "mesh.ply").open("rb") as handle:
        header = b""
        while b"end_header" not in header:
            header += handle.readline()
        count = int(next(line for line in header.decode().splitlines()
                         if line.startswith("element vertex")).split()[-1])
        raw = handle.read(count * 16)
    mesh = np.frombuffer(raw, dtype=np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgba", "u1", 4)]), count=count)
    mesh_points = cloud.frame.apply(
        np.stack([mesh["x"], mesh["y"], mesh["z"]], axis=1).astype(np.float64))

    # Floor and ceiling heights are the sharpest shared feature.
    assert abs(cloud.points[:, 1].min() - mesh_points[:, 1].min()) < 0.3
    assert abs(cloud.points[:, 1].max() - mesh_points[:, 1].max()) < 0.5
