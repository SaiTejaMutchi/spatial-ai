"""Connector and contract tests.

The happy paths run against the real ARKitScenes fixture where it is present.
Everything else is deliberately broken input, because that is what a real
integration hands you.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.connectors.arkitscenes import ARKitScenesConnector
from pipeline.connectors.base import SourceRejected, match_poses, rodrigues
from pipeline.connectors.detect import detect_source
from pipeline.connectors.stray_scanner import StrayScannerConnector, quaternion_to_matrix
from pipeline.connectors.unity_obj import UnityOBJConnector
from pipeline.contracts.normalized_capture import NormalizedCapture
from pipeline.contracts.validate import validate
from pipeline.tests import synthetic

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_FIXTURE = REPO_ROOT / "samples/arkitscenes/raw/Training/47333462"


def codes(issues) -> set[str]:
    return {i.code for i in issues if i.severity == "error"}


# --------------------------------------------------------------------------
# rotation helpers
# --------------------------------------------------------------------------

def test_rodrigues_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    for _ in range(50):
        axis_angle = rng.normal(size=3) * rng.uniform(0.0, 3.0)
        assert np.allclose(rodrigues(axis_angle), cv2.Rodrigues(axis_angle)[0], atol=1e-12)
    assert np.allclose(rodrigues(np.zeros(3)), np.eye(3))


def test_quaternion_to_matrix_is_a_rotation():
    rng = np.random.default_rng(11)
    for _ in range(50):
        q = rng.normal(size=4)
        rot = quaternion_to_matrix(*q)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-12)


def test_quaternion_rejects_zero_length():
    with pytest.raises(SourceRejected, match="zero-length quaternion"):
        quaternion_to_matrix(0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# timestamp matching
# --------------------------------------------------------------------------

def test_frames_match_poses_by_timestamp_not_index():
    frame_times = np.array([0.00, 0.02, 0.04, 0.06])
    pose_times = np.array([0.00, 0.05])
    index, offset = match_poses(frame_times, pose_times, tolerance_s=0.03)
    # Index matching would pair frame 1 with pose 1; timestamp matching does not.
    assert index.tolist() == [0, 0, 1, 1]
    assert np.allclose(offset, [0.0, 0.02, -0.01, 0.01])


def test_frames_outside_tolerance_are_unmatched():
    index, _ = match_poses(np.array([0.0, 1.0]), np.array([0.0]), tolerance_s=0.05)
    assert index.tolist() == [0, -1]


def test_single_pose_trajectory_still_matches():
    """np.clip collapses when there is one pose; the match must survive it."""
    index, offset = match_poses(np.array([0.0, 1.0]), np.array([0.0]), tolerance_s=0.05)
    assert index.tolist() == [0, -1]
    assert np.allclose(offset, [0.0, 1.0])


def test_empty_trajectory_matches_nothing():
    index, _ = match_poses(np.array([0.0, 1.0]), np.array([]), tolerance_s=0.05)
    assert index.tolist() == [-1, -1]


# --------------------------------------------------------------------------
# ARKitScenes connector — synthetic
# --------------------------------------------------------------------------

def test_arkitscenes_happy_path(tmp_path):
    source = synthetic.make_arkitscenes(tmp_path / "src", frames=6, pose_rate=2)
    capture = ARKitScenesConnector(source).normalize(tmp_path / "out")
    assert capture.pose_convention == "camera_to_world"
    assert capture.depth_scale_m == 0.001
    assert len(capture.frames) == 3          # poses exist for every other frame
    assert len(capture.excluded_frames) == 3
    assert not codes(validate(tmp_path / "out"))


def test_arkitscenes_inverts_the_source_pose_convention(tmp_path):
    # World-to-camera translation -0.1 * i means the camera centre is at +0.1 * i.
    source = synthetic.make_arkitscenes(tmp_path / "src", frames=4, pose_rate=1)
    capture = ARKitScenesConnector(source).normalize(tmp_path / "out")
    centres = np.array([np.array(f.camera_to_world)[:3, 3] for f in capture.frames])
    assert np.allclose(centres[:, 0], [0.0, 0.1, 0.2, 0.3], atol=1e-9)
    assert np.allclose(centres[:, 1:], 0.0, atol=1e-9)


def test_arkitscenes_reports_the_frame_with_no_odometry_row(tmp_path):
    source = synthetic.make_arkitscenes(
        tmp_path / "src", frames=4,
        traj_lines=["1000.00000000 0.0 0.0 0.0 0.0 0.0 0.0"])
    capture = ARKitScenesConnector(source).normalize(tmp_path / "out")
    assert len(capture.frames) == 1
    reasons = [e.reason for e in capture.excluded_frames]
    assert any("has depth data but no matching odometry row" in r for r in reasons)
    assert any("tolerance 50 ms" in r for r in reasons)


def test_arkitscenes_rejects_a_missing_trajectory(tmp_path):
    source = synthetic.make_arkitscenes(tmp_path / "src")
    (source / "lowres_wide.traj").unlink()
    with pytest.raises(SourceRejected, match="not an ARKitScenes raw sequence"):
        ARKitScenesConnector(source).normalize(tmp_path / "out")


def test_arkitscenes_rejects_a_malformed_trajectory_row(tmp_path):
    source = synthetic.make_arkitscenes(
        tmp_path / "src",
        traj_lines=["1000.0 0.0 0.0 0.0 0.0 0.0 0.0", "1000.1 0.0 0.0 0.0"])
    with pytest.raises(SourceRejected, match=r"line 2 has 4 columns"):
        ARKitScenesConnector(source).normalize(tmp_path / "out")


def test_arkitscenes_rejects_an_empty_trajectory(tmp_path):
    source = synthetic.make_arkitscenes(tmp_path / "src", traj_lines=[])
    with pytest.raises(SourceRejected, match="empty"):
        ARKitScenesConnector(source).normalize(tmp_path / "out")


def test_arkitscenes_rejects_missing_depth(tmp_path):
    source = synthetic.make_arkitscenes(tmp_path / "src")
    for path in (source / "lowres_depth").glob("*.png"):
        path.unlink()
    with pytest.raises(SourceRejected, match="no PNG frames"):
        ARKitScenesConnector(source).normalize(tmp_path / "out")


def test_arkitscenes_rejects_missing_intrinsics(tmp_path):
    source = synthetic.make_arkitscenes(tmp_path / "src", with_intrinsics=False)
    with pytest.raises(SourceRejected, match="core modality"):
        ARKitScenesConnector(source).normalize(tmp_path / "out")


def test_arkitscenes_rejects_malformed_intrinsics(tmp_path):
    source = synthetic.make_arkitscenes(tmp_path / "src", intrinsics_text="16 12 200.0")
    with pytest.raises(SourceRejected, match="expected 6"):
        ARKitScenesConnector(source).normalize(tmp_path / "out")


def test_missing_confidence_is_recorded_not_invented(tmp_path):
    source = synthetic.make_arkitscenes(tmp_path / "src", with_confidence=False)
    capture = ARKitScenesConnector(source).normalize(tmp_path / "out")
    assert capture.modalities["confidence"]["available"] is False
    assert all(f.confidence is None for f in capture.frames)
    assert not (tmp_path / "out" / "confidence").exists()
    issues = validate(tmp_path / "out")
    assert not codes(issues)
    assert "MISSING_PREFERRED_MODALITY" in {i.code for i in issues}


def test_rgb_larger_than_depth_rescales_intrinsics(tmp_path):
    source = synthetic.make_arkitscenes(
        tmp_path / "src", depth_size=(16, 12), rgb_size=(64, 48))
    capture = ARKitScenesConnector(source).normalize(tmp_path / "out")
    depth = next(i for i in capture.intrinsics if i.stream == "depth")
    rgb = next(i for i in capture.intrinsics if i.stream == "rgb")
    assert (depth.width, depth.height) == (16, 12)
    assert depth.derivation == "rescaled_from:rgb"
    assert np.isclose(depth.fx, rgb.fx / 4)
    assert np.isclose(depth.cx, rgb.cx / 4)
    assert not codes(validate(tmp_path / "out"))


def test_optional_modalities_absent_are_declared_absent(tmp_path):
    source = synthetic.make_arkitscenes(tmp_path / "src")
    capture = ARKitScenesConnector(source).normalize(tmp_path / "out")
    for modality in ("mesh", "imu", "distortion", "semantics"):
        assert capture.modalities[modality]["available"] is False
        assert capture.modalities[modality]["note"]


def test_stride_selection_is_recorded_not_hidden(tmp_path):
    source = synthetic.make_arkitscenes(tmp_path / "src", frames=8, pose_rate=1)
    capture = ARKitScenesConnector(source, stride=2).normalize(tmp_path / "out")
    assert len(capture.frames) == 4
    assert capture.frame_selection["skipped_by_stride"] == 4
    assert capture.frame_selection["source_frames"] == 8
    # Deliberate subsampling is a count, not four fake failures.
    assert capture.excluded_frames == []


# --------------------------------------------------------------------------
# Stray Scanner connector — synthetic (documented format)
# --------------------------------------------------------------------------

def test_stray_happy_path(tmp_path):
    source = synthetic.make_stray(tmp_path / "src", frames=5)
    capture = StrayScannerConnector(source).normalize(tmp_path / "out")
    assert len(capture.frames) == 5
    assert capture.provenance.classification == "final_private_capture"
    assert capture.frame_selection["pose_convention_verified"] is True
    assert capture.imu == "imu.csv"
    assert not codes(validate(tmp_path / "out"))


def test_stray_accepts_whitespace_in_real_legacy_csv_header(tmp_path):
    source = synthetic.make_stray(tmp_path / "src", frames=3)
    odometry = source / "odometry.csv"
    lines = odometry.read_text().splitlines()
    lines[0] = ", ".join(token.strip() for token in lines[0].split(","))
    odometry.write_text("\n".join(lines) + "\n")

    capture = StrayScannerConnector(source).normalize(tmp_path / "out")

    assert len(capture.frames) == 3
    assert not codes(validate(tmp_path / "out"))


def test_stray_rescales_colour_intrinsics_to_depth(tmp_path):
    source = synthetic.make_stray(tmp_path / "src", depth_size=(16, 12))
    capture = StrayScannerConnector(source).normalize(tmp_path / "out")
    depth = next(i for i in capture.intrinsics if i.stream == "depth")
    assert depth.derivation == "rescaled_from:rgb"
    assert (depth.width, depth.height) == (16, 12)
    # 1400 px focal on a 1920-wide colour frame scales down with the width.
    assert depth.fx < 1400.0


def test_stray_rejects_missing_required_columns(tmp_path):
    source = synthetic.make_stray(
        tmp_path / "src", header="timestamp,frame,x,y,z,qx,qy,qz")
    with pytest.raises(SourceRejected, match=r"missing required column\(s\) \['qw'\]"):
        StrayScannerConnector(source).normalize(tmp_path / "out")


def test_stray_reports_an_odometry_row_with_no_depth_image(tmp_path):
    source = synthetic.make_stray(tmp_path / "src", frames=4)
    (source / "depth" / "000002.png").unlink()
    capture = StrayScannerConnector(source).normalize(tmp_path / "out")
    assert len(capture.frames) == 3
    assert any("depth/000002.png does not exist" in e.reason
               for e in capture.excluded_frames)


def test_stray_reports_a_non_monotonic_timestamp(tmp_path):
    rows = ["1000.000000,000000,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1400.0,1400.0,960.0,720.0,,",
            "999.000000,000001,0.1,0.0,0.0,0.0,0.0,0.0,1.0,1400.0,1400.0,960.0,720.0,,",
            "1000.100000,000002,0.2,0.0,0.0,0.0,0.0,0.0,1.0,1400.0,1400.0,960.0,720.0,,"]
    source = synthetic.make_stray(tmp_path / "src", frames=3, rows=rows)
    capture = StrayScannerConnector(source).normalize(tmp_path / "out")
    assert any("precedes the previous row" in e.reason for e in capture.excluded_frames)
    assert len(capture.frames) == 2


def test_stray_rejects_a_header_only_odometry_file(tmp_path):
    source = synthetic.make_stray(tmp_path / "src", frames=2)
    header = (source / "odometry.csv").read_text().splitlines()[0]
    (source / "odometry.csv").write_text(header + "\n")
    with pytest.raises(SourceRejected, match="header but no pose rows"):
        StrayScannerConnector(source).normalize(tmp_path / "out")


def test_stray_missing_confidence_is_declared(tmp_path):
    source = synthetic.make_stray(tmp_path / "src", with_confidence=False)
    capture = StrayScannerConnector(source).normalize(tmp_path / "out")
    assert capture.modalities["confidence"]["available"] is False
    assert capture.modalities["rgb"]["available"] is False
    assert capture.modalities["rgb"]["note"]


REAL_STRAY_SAMPLE = Path("samples/stray/raw/8653a2142b")


@pytest.mark.skipif(not REAL_STRAY_SAMPLE.is_dir(),
                    reason="local real Stray integration sample is not installed")
def test_real_legacy_stray_sample_reaches_normalized_contract(tmp_path):
    header = (REAL_STRAY_SAMPLE / "odometry.csv").read_text().splitlines()[0]
    assert header == "timestamp, frame, x, y, z, qx, qy, qz, qw"
    capture = StrayScannerConnector(
        REAL_STRAY_SAMPLE,
        classification="public_development_fixture",
        stride=24,
        max_frames=12,
    ).normalize(tmp_path / "real_stray")

    assert len(capture.frames) == 12
    assert capture.modalities["rgb"]["available"] is True
    assert capture.modalities["confidence"]["available"] is True
    assert capture.frame_selection["pose_convention_verified"] is True
    assert not codes(validate(tmp_path / "real_stray"))


# --------------------------------------------------------------------------
# Unity OBJ connector
# --------------------------------------------------------------------------

def test_unity_obj_marks_every_core_modality_unavailable(tmp_path):
    synthetic.make_unity_obj(tmp_path / "src")
    capture = UnityOBJConnector(tmp_path / "src").normalize(tmp_path / "out")
    assert capture.frames == []
    assert capture.modalities["mesh"]["available"] is True
    for modality in ("depth", "intrinsics", "trajectory", "rgb", "confidence"):
        assert capture.modalities[modality]["available"] is False


def test_unity_obj_package_is_refused_for_geometry(tmp_path):
    synthetic.make_unity_obj(tmp_path / "src")
    UnityOBJConnector(tmp_path / "src").normalize(tmp_path / "out")
    failures = codes(validate(tmp_path / "out"))
    assert "MISSING_CORE_MODALITY" in failures
    assert "NO_FRAMES" in failures


def test_unity_obj_rejects_an_empty_export(tmp_path):
    synthetic.make_unity_obj(tmp_path / "src", vertices=0)
    with pytest.raises(SourceRejected, match="declares no vertices"):
        UnityOBJConnector(tmp_path / "src").normalize(tmp_path / "out")


# --------------------------------------------------------------------------
# source detection
# --------------------------------------------------------------------------

def test_detection_picks_the_right_connector(tmp_path):
    arkit = synthetic.make_arkitscenes(tmp_path / "a")
    stray = synthetic.make_stray(tmp_path / "b")
    synthetic.make_unity_obj(tmp_path / "c")
    assert detect_source(arkit) is ARKitScenesConnector
    assert detect_source(stray) is StrayScannerConnector
    assert detect_source(tmp_path / "c") is UnityOBJConnector


def test_detection_names_what_it_saw(tmp_path):
    (tmp_path / "mystery").mkdir()
    (tmp_path / "mystery" / "notes.txt").write_text("hello")
    with pytest.raises(SourceRejected, match="matches no known capture source"):
        detect_source(tmp_path / "mystery")


def test_detection_reports_a_missing_path(tmp_path):
    with pytest.raises(SourceRejected, match="does not exist"):
        detect_source(tmp_path / "absent")


# --------------------------------------------------------------------------
# contract validation of corrupted packages
# --------------------------------------------------------------------------

def _normalized(tmp_path) -> Path:
    source = synthetic.make_arkitscenes(tmp_path / "src", frames=4, pose_rate=1)
    ARKitScenesConnector(source).normalize(tmp_path / "out")
    return tmp_path / "out"


def _rewrite_manifest(root: Path, mutate) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    mutate(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def test_validator_catches_a_missing_contract_file(tmp_path):
    root = _normalized(tmp_path)
    (root / "intrinsics.json").unlink()
    assert "MISSING_CONTRACT_FILE" in codes(validate(root))


def test_validator_catches_a_dangling_depth_reference(tmp_path):
    root = _normalized(tmp_path)
    (root / "depth" / "000001.png").unlink()
    assert "MISSING_DEPTH_FILE" in codes(validate(root))


def test_validator_catches_a_non_orthonormal_pose(tmp_path):
    root = _normalized(tmp_path)

    def mutate(manifest):
        manifest["frames"][1]["camera_to_world"][0][0] = 2.0
    _rewrite_manifest(root, mutate)
    issues = validate(root)
    assert "INVALID_POSE" in codes(issues)
    assert any("Frame 1" in i.message for i in issues)


def test_validator_catches_a_mirrored_pose(tmp_path):
    root = _normalized(tmp_path)

    def mutate(manifest):
        matrix = np.array(manifest["frames"][2]["camera_to_world"])
        matrix[:3, :3] = matrix[:3, :3] @ np.diag([1.0, 1.0, -1.0])
        manifest["frames"][2]["camera_to_world"] = matrix.tolist()
    _rewrite_manifest(root, mutate)
    issues = validate(root)
    assert "INVALID_POSE" in codes(issues)
    assert any("determinant is -1" in i.message for i in issues)


def test_validator_catches_an_implausible_depth_scale(tmp_path):
    root = _normalized(tmp_path)

    def mutate(manifest):
        manifest["depth_scale_m"] = 1.0   # metres-per-unit off by a thousand
    _rewrite_manifest(root, mutate)
    issues = validate(root)
    assert "IMPLAUSIBLE_DEPTH_SCALE" in codes(issues)
    assert any("depth_scale_m=1.0 is likely wrong" in i.message for i in issues)


def test_validator_catches_intrinsics_that_do_not_match_the_images(tmp_path):
    root = _normalized(tmp_path)
    intrinsics = json.loads((root / "intrinsics.json").read_text())
    for stream in intrinsics["streams"]:
        if stream["stream"] == "depth":
            stream["width"], stream["height"] = 640, 480
    (root / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))
    assert "INTRINSICS_RESOLUTION_MISMATCH" in codes(validate(root))


def test_validator_catches_a_principal_point_outside_the_image(tmp_path):
    root = _normalized(tmp_path)
    intrinsics = json.loads((root / "intrinsics.json").read_text())
    intrinsics["streams"][0]["cx"] = 9999.0
    (root / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))
    assert "PRINCIPAL_POINT_OUTSIDE_IMAGE" in codes(validate(root))


def test_validator_catches_an_absurd_focal_length(tmp_path):
    root = _normalized(tmp_path)
    intrinsics = json.loads((root / "intrinsics.json").read_text())
    intrinsics["streams"][0]["fx"] = 0.001
    (root / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))
    assert "IMPLAUSIBLE_INTRINSICS" in codes(validate(root))


def test_validator_catches_non_monotonic_frame_timestamps(tmp_path):
    root = _normalized(tmp_path)

    def mutate(manifest):
        manifest["frames"][2]["timestamp_s"] = 0.0
    _rewrite_manifest(root, mutate)
    issues = validate(root)
    assert "NON_MONOTONIC_TIMESTAMP" in codes(issues)
    assert any("Frame 2" in i.message for i in issues)


def test_validator_catches_an_unknown_intrinsics_stream(tmp_path):
    root = _normalized(tmp_path)

    def mutate(manifest):
        manifest["frames"][0]["intrinsics_stream"] = "thermal"
    _rewrite_manifest(root, mutate)
    assert "UNKNOWN_INTRINSICS_STREAM" in codes(validate(root))


def test_validator_catches_depth_that_is_not_uint16(tmp_path):
    root = _normalized(tmp_path)
    Image.fromarray(np.zeros((12, 16), dtype=np.uint8)).save(root / "depth" / "000000.png")
    assert codes(validate(root)) & {"WRONG_DEPTH_DTYPE", "NO_VALID_DEPTH"}


def test_validator_catches_an_uninverted_trajectory(tmp_path):
    """A world-to-camera trajectory left uninverted inflates the camera path."""
    root = _normalized(tmp_path)

    def mutate(manifest):
        for i, frame in enumerate(manifest["frames"]):
            matrix = np.eye(4)
            matrix[:3, 3] = [i * 300.0, 0.0, 0.0]
            frame["camera_to_world"] = matrix.tolist()
    _rewrite_manifest(root, mutate)
    issues = validate(root)
    assert "IMPLAUSIBLE_CAMERA_PATH" in codes(issues)
    assert any("world-to-camera" in i.message for i in issues)


def test_round_trip_through_disk_preserves_the_contract(tmp_path):
    root = _normalized(tmp_path)
    capture = NormalizedCapture.read(root)
    assert capture.pose_convention == "camera_to_world"
    assert capture.provenance.connector == "ARKitScenesConnector"
    assert len(capture.frames) == 4


# --------------------------------------------------------------------------
# real fixture
# --------------------------------------------------------------------------

requires_fixture = pytest.mark.skipif(
    not PRIMARY_FIXTURE.is_dir(),
    reason="ARKitScenes primary fixture is not present; see docs/fixture_selection.md")


@requires_fixture
def test_real_fixture_normalizes_and_validates(tmp_path):
    capture = ARKitScenesConnector(PRIMARY_FIXTURE, stride=60).normalize(tmp_path / "out")
    assert len(capture.frames) >= 50
    assert capture.modalities["rgb"]["available"] is True
    assert capture.modalities["confidence"]["available"] is True
    assert capture.provenance.classification == "public_development_fixture"
    assert not codes(validate(tmp_path / "out"))


@requires_fixture
def test_real_fixture_back_projects_onto_its_own_arkit_mesh(tmp_path):
    """The strongest available check that the pose convention is right.

    Depth is unprojected with the contract's camera axes and placed with the
    contract's camera_to_world poses. If either were wrong, the resulting cloud
    would not land inside the bounding box of the mesh the same device produced.
    """
    capture = ARKitScenesConnector(PRIMARY_FIXTURE, stride=120).normalize(tmp_path / "out")
    root = tmp_path / "out"
    intr = next(i for i in capture.intrinsics if i.stream == "depth")

    points = []
    for frame in capture.frames:
        depth = np.array(Image.open(root / frame.depth)).astype(np.float64) * capture.depth_scale_m
        confidence = np.array(Image.open(root / frame.confidence))
        rows, cols = np.nonzero((depth > 0.2) & (depth < 5.0) & (confidence >= 2))
        if rows.size == 0:
            continue
        z = depth[rows, cols]
        camera = np.stack([(cols - intr.cx) / intr.fx * z,
                           (rows - intr.cy) / intr.fy * z,
                           z, np.ones_like(z)])
        points.append((np.array(frame.camera_to_world) @ camera)[:3].T)

    cloud = np.concatenate(points)
    assert len(cloud) > 100_000

    with (root / "mesh.ply").open("rb") as handle:
        header = b""
        while b"end_header" not in header:
            header += handle.readline()
        count = int(next(line for line in header.decode().splitlines()
                         if line.startswith("element vertex")).split()[-1])
        raw = handle.read(count * 16)
    mesh = np.frombuffer(raw, dtype=np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgba", "u1", 4)]), count=count)
    mesh_points = np.stack([mesh["x"], mesh["y"], mesh["z"]], axis=1)

    # The cloud must sit inside the mesh's extent with only sensor-noise slack,
    # and share its centroid to within the coverage difference of a sparse
    # frame sample.
    margin = 0.5
    assert (cloud.min(0) >= mesh_points.min(0) - margin).all()
    assert (cloud.max(0) <= mesh_points.max(0) + margin).all()
    assert np.abs(cloud.mean(0) - mesh_points.mean(0)).max() < 1.5
