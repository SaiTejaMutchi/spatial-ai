"""Structural plane extraction tests.

A synthetic room with dimensions chosen in advance is the honest way to check
a plane fitter: the answer is known before the code runs, so recovering it is
evidence rather than confirmation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.contracts.normalized_capture import NormalizedCapture
from pipeline.geometry.config import load_geometry_config
from pipeline.geometry.planes import (
    coverage_on_plane,
    extract_planes,
    fit_plane_total_least_squares,
)
from pipeline.geometry.points import PointCloud, build_point_cloud
from pipeline.geometry.frame import resolve_canonical_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_FIXTURE = REPO_ROOT / "samples/arkitscenes/raw/Training/47333462"

# Fixed before any code ran; nothing below is permitted to change them.
ROOM_LENGTH = 4.20
ROOM_WIDTH = 3.10
ROOM_HEIGHT = 2.55


@pytest.fixture(scope="module")
def config():
    return load_geometry_config()


def _synthetic_room(
    noise_m: float = 0.004,
    spacing: float = 0.02,
    seed: int = 5,
    yaw_deg: float = 0.0,
) -> PointCloud:
    """A closed box sampled on all six surfaces, with realistic depth noise."""
    rng = np.random.default_rng(seed)
    xs = np.arange(0.0, ROOM_LENGTH + spacing, spacing)
    zs = np.arange(0.0, ROOM_WIDTH + spacing, spacing)
    ys = np.arange(0.0, ROOM_HEIGHT + spacing, spacing)

    patches = []
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    for height in (0.0, ROOM_HEIGHT):
        patches.append(np.stack(
            [gx.ravel(), np.full(gx.size, height), gz.ravel()], axis=1))
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    for offset in (0.0, ROOM_WIDTH):
        patches.append(np.stack(
            [gx.ravel(), gy.ravel(), np.full(gx.size, offset)], axis=1))
    gz, gy = np.meshgrid(zs, ys, indexing="ij")
    for offset in (0.0, ROOM_LENGTH):
        patches.append(np.stack(
            [np.full(gz.size, offset), gy.ravel(), gz.ravel()], axis=1))

    points = np.concatenate(patches)
    points += rng.normal(scale=noise_m, size=points.shape)

    if yaw_deg:
        angle = np.deg2rad(yaw_deg)
        rotation = np.array([[np.cos(angle), 0.0, -np.sin(angle)],
                             [0.0, 1.0, 0.0],
                             [np.sin(angle), 0.0, np.cos(angle)]])
        points = points @ rotation.T

    poses = np.tile(np.eye(4), (40, 1, 1))
    frame = resolve_canonical_frame(poses, "+y", "synthetic", 10.0, 1000)
    return PointCloud(
        points=points,
        frame=frame,
        frame_indices=np.arange(len(points), dtype=np.int32) % 40,
    )


# --------------------------------------------------------------------------
# fitting primitives
# --------------------------------------------------------------------------

def test_total_least_squares_recovers_a_tilted_plane():
    rng = np.random.default_rng(2)
    normal = np.array([0.3, 0.9, -0.32])
    normal /= np.linalg.norm(normal)
    offset = 1.7
    basis = np.linalg.svd(normal[None])[2][1:]
    coords = rng.uniform(-2, 2, size=(4000, 2))
    points = coords @ basis + normal * offset
    points += rng.normal(scale=0.002, size=points.shape)

    fitted_normal, fitted_offset = fit_plane_total_least_squares(points)
    if float(fitted_normal @ normal) < 0:
        fitted_normal, fitted_offset = -fitted_normal, -fitted_offset
    assert np.allclose(fitted_normal, normal, atol=2e-3)
    assert abs(fitted_offset - offset) < 2e-3


def test_plane_fit_needs_three_points():
    with pytest.raises(ValueError, match="at least three points"):
        fit_plane_total_least_squares(np.zeros((2, 3)))


def test_coverage_separates_a_surface_from_a_scatter():
    rng = np.random.default_rng(4)
    grid = np.stack(np.meshgrid(np.arange(0, 2, 0.02), np.arange(0, 2, 0.02),
                                indexing="ij"), axis=-1).reshape(-1, 2)
    dense = np.stack([grid[:, 0], grid[:, 1], np.zeros(len(grid))], axis=1)
    dense_coverage, _ = coverage_on_plane(dense, np.array([0.0, 0.0, 1.0]), 0.1)
    assert dense_coverage > 0.95

    sparse = rng.uniform(0, 2, size=(40, 2))
    scatter = np.stack([sparse[:, 0], sparse[:, 1], np.zeros(len(sparse))], axis=1)
    sparse_coverage, _ = coverage_on_plane(scatter, np.array([0.0, 0.0, 1.0]), 0.1)
    assert sparse_coverage < 0.5


# --------------------------------------------------------------------------
# synthetic room recovery
# --------------------------------------------------------------------------

def test_synthetic_room_floor_and_ceiling_are_recovered(config):
    planes = extract_planes(_synthetic_room(), config)
    assert planes.floor is not None and planes.ceiling is not None
    assert abs(planes.floor.centroid[1] - 0.0) < 0.01
    assert abs(planes.ceiling.centroid[1] - ROOM_HEIGHT) < 0.01
    separation = planes.ceiling.centroid[1] - planes.floor.centroid[1]
    assert abs(separation - ROOM_HEIGHT) < 0.015, f"height error {separation - ROOM_HEIGHT:.4f} m"
    assert planes.floor.rms_residual_m < 0.01
    assert planes.floor.coverage_fraction > 0.9


def test_synthetic_room_recovers_four_wall_directions(config):
    planes = extract_planes(_synthetic_room(), config)
    assert len(planes.walls) >= 4
    strongest = sorted(planes.walls, key=lambda w: -w.inlier_count)[:4]
    for wall in strongest:
        assert abs(float(wall.normal[1])) < 1e-6, "a wall normal must be level"
        assert wall.rms_residual_m < 0.01

    # The four dominant walls must lie on the four true planes.
    expected = [
        (np.array([1.0, 0.0, 0.0]), 0.0), (np.array([1.0, 0.0, 0.0]), ROOM_LENGTH),
        (np.array([0.0, 0.0, 1.0]), 0.0), (np.array([0.0, 0.0, 1.0]), ROOM_WIDTH),
    ]
    for normal, offset in expected:
        matched = [
            w for w in strongest
            if abs(abs(float(w.normal @ normal)) - 1.0) < 0.02
            and abs(abs(float(w.normal @ w.centroid)) - abs(offset)) < 0.03
        ]
        assert matched, f"no wall recovered at offset {offset} along {normal}"


def test_wall_recovery_survives_a_rotated_room(config):
    """Nothing may depend on walls being axis-aligned."""
    planes = extract_planes(_synthetic_room(yaw_deg=37.0), config)
    assert planes.floor is not None
    assert len(planes.walls) >= 4
    normals = np.array([w.normal for w in sorted(
        planes.walls, key=lambda w: -w.inlier_count)[:4]])
    # Two perpendicular directions, whatever the yaw.
    pairings = np.abs(normals @ normals.T)
    off_diagonal = pairings[~np.eye(4, dtype=bool)]
    assert ((off_diagonal < 0.05) | (off_diagonal > 0.95)).all()


def test_wall_normals_point_into_the_room(config):
    planes = extract_planes(_synthetic_room(), config)
    interior = np.array([ROOM_LENGTH / 2, ROOM_HEIGHT / 2, ROOM_WIDTH / 2])
    for wall in sorted(planes.walls, key=lambda w: -w.inlier_count)[:4]:
        assert wall.signed_distance(interior[None])[0] > 0, wall.plane_id


def test_a_room_that_is_too_short_has_its_ceiling_rejected(config):
    """A 0.6 m separation is not a room; reporting it would be worse than nothing."""
    rng = np.random.default_rng(9)
    grid = np.stack(np.meshgrid(np.arange(0, 3, 0.02), np.arange(0, 3, 0.02),
                                indexing="ij"), axis=-1).reshape(-1, 2)
    low = np.stack([grid[:, 0], np.zeros(len(grid)), grid[:, 1]], axis=1)
    high = np.stack([grid[:, 0], np.full(len(grid), 0.6), grid[:, 1]], axis=1)
    points = np.concatenate([low, high]) + rng.normal(scale=0.003, size=(2 * len(grid), 3))
    poses = np.tile(np.eye(4), (30, 1, 1))
    cloud = PointCloud(points=points,
                       frame=resolve_canonical_frame(poses, "+y", "s", 10.0, 1000),
                       frame_indices=np.zeros(len(points), dtype=np.int32))
    planes = extract_planes(cloud, config)
    assert planes.ceiling is None
    assert any("outside the plausible range" in r["reason"] for r in planes.rejected)


def test_rejected_candidates_carry_a_reason(config):
    planes = extract_planes(_synthetic_room(), config)
    for record in planes.rejected:
        assert record["reason"]
        assert record["kind"] in {"floor", "ceiling", "wall", "horizontal"}


def test_plane_records_expose_their_evidence_and_config(config):
    planes = extract_planes(_synthetic_room(), config)
    record = planes.to_record()
    assert record["configProvenance"]["geometryConfigId"] == "geometry_config_v0.1"
    assert record["configProvenance"]["geometryConfigHash"]
    support = record["floor"]["support"]
    for key in ("inlierCount", "contributingFrames", "rmsResidual_m",
                "maxResidual_m", "coverageFraction"):
        assert key in support
    assert record["floor"]["algorithm"]


def test_extraction_is_deterministic(config):
    """The configuration gets frozen and rerun; the same input must not drift."""
    cloud = _synthetic_room()
    first = extract_planes(cloud, config).to_record()
    second = extract_planes(cloud, config).to_record()
    assert first == second


# --------------------------------------------------------------------------
# real fixture
# --------------------------------------------------------------------------

@pytest.mark.skipif(not PRIMARY_FIXTURE.is_dir(), reason="fixture not present")
def test_real_fixture_yields_a_defensible_structure(tmp_path, config):
    from pipeline.connectors.arkitscenes import ARKitScenesConnector

    ARKitScenesConnector(PRIMARY_FIXTURE, stride=40).normalize(tmp_path / "out")
    capture = NormalizedCapture.read(tmp_path / "out")
    cloud = build_point_cloud(tmp_path / "out", capture, config, stride=2)
    planes = extract_planes(cloud, config)

    assert planes.floor is not None, "no floor recovered from the primary fixture"
    assert planes.ceiling is not None
    separation = planes.ceiling.centroid[1] - planes.floor.centroid[1]
    assert config.get("min_room_height_m") <= separation <= config.get("max_room_height_m")
    assert planes.floor.rms_residual_m < 0.05
    assert len(planes.walls) >= 4

    # A rectangular room is dominated by two perpendicular *axes*, each with a
    # facing pair. The two strongest walls are usually that opposing pair, so
    # the structure to check is the axis set, not pairwise perpendicularity.
    dominant = np.array([w.normal for w in
                         sorted(planes.walls, key=lambda w: -w.inlier_count)[:4]])
    pairings = np.abs(dominant @ dominant.T)
    off_diagonal = pairings[~np.eye(len(dominant), dtype=bool)]
    assert ((off_diagonal < 0.2) | (off_diagonal > 0.8)).all(), (
        f"the dominant walls do not form perpendicular axes: {off_diagonal.round(3)}")
    assert (off_diagonal < 0.2).any(), "no perpendicular wall direction was found"
