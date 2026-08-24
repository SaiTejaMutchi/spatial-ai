"""Compare camera-to-world (`R @ p + t`) with the transposed-R alternative.

This settles the Stray pose convention from room structure, not IMU attitude.
It writes evidence only and does not change geometry configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.connectors.stray_scanner import quaternion_to_matrix
from pipeline.geometry.config import load_geometry_config
from pipeline.geometry.frame import CanonicalFrame
from pipeline.geometry.planes import extract_planes
from pipeline.geometry.points import PointCloud, voxel_downsample


def _read_odometry(root: Path) -> list[dict]:
    with (root / "odometry.csv").open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        return list(reader)


def _unproject(root: Path, rows: list[dict], stride: int, use_transpose: bool
               ) -> tuple[np.ndarray, np.ndarray]:
    legacy = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    fx, fy, cx, cy = float(legacy[0, 0]), float(legacy[1, 1]), float(legacy[0, 2]), float(legacy[1, 2])
    chunks: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    for position, row in enumerate(rows[::stride]):
        key = str(row["frame"]).strip()
        path = root / "depth" / f"{key}.png"
        if not path.is_file():
            continue
        raw = np.array(Image.open(path))
        metres = raw.astype(np.float64) * 0.001
        valid = (raw > 0) & (metres >= 0.3) & (metres <= 5.0)
        rows_i, cols_i = np.nonzero(valid)
        if rows_i.size == 0:
            continue
        z = metres[rows_i, cols_i]
        # Depth is 256x192; published intrinsics are RGB-sized. Rescale like the connector.
        depth = Image.open(path)
        dw, dh = depth.size
        rgb_w = int(round(cx * 2)) or 1
        scale_x, scale_y = dw / rgb_w, dh / (int(round(cy * 2)) or 1)
        dfx, dfy, dcx, dcy = fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y
        camera = np.stack([
            (cols_i - dcx) / dfx * z,
            (rows_i - dcy) / dfy * z,
            z,
        ], axis=1)
        rotation = quaternion_to_matrix(
            float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]))
        translation = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        if use_transpose:
            rotation = rotation.T
        chunks.append(camera @ rotation.T + translation)
        centres.append(translation)
    if not chunks:
        raise SystemExit("no depth survived unprojection")
    return np.concatenate(chunks), np.asarray(centres)


def _score(points: np.ndarray, label: str) -> dict:
    config = load_geometry_config()
    reduced = voxel_downsample(points, config.get("voxel_size_m"))
    frame = CanonicalFrame(
        source_to_canonical=np.eye(4),
        source_up_axis=np.array([0.0, 1.0, 0.0]),
        up_source="declared_diagnostic",
        world_frame="arkit_session",
        diagnostics={"diagnosticOnly": True, "construction": label},
    )
    cloud = PointCloud(
        points=reduced, frame=frame,
        frame_indices=np.zeros(len(reduced), dtype=np.int32),
        diagnostics={"diagnosticOnly": True},
        config_provenance=config.provenance("voxel_size_m"),
    )
    planes = extract_planes(cloud, config)
    horizontal = planes.diagnostics.get("horizontal", {})
    floor = planes.floor
    ceiling = planes.ceiling
    height = None
    if floor and ceiling:
        height = abs(float(ceiling.offset - floor.offset))
    return {
        "construction": label,
        "pointCount": int(len(reduced)),
        "extentM": np.ptp(reduced, axis=0).round(6).tolist(),
        "floor": floor is not None,
        "ceiling": ceiling is not None,
        "wallCount": len(planes.walls),
        "floorRmsMm": None if floor is None else round(floor.rms_residual_m * 1000, 3),
        "ceilingRmsMm": None if ceiling is None else round(ceiling.rms_residual_m * 1000, 3),
        "floorSupport": None if floor is None else floor.inlier_count,
        "ceilingSupport": None if ceiling is None else ceiling.inlier_count,
        "roomHeightM": None if height is None else round(height, 4),
        "floorCeilingSeparation_m": horizontal.get("floorCeilingSeparation_m"),
        "winner": floor is not None and ceiling is not None,
    }


def run(root: Path, output: Path, stride: int = 6) -> dict:
    rows = _read_odometry(root)
    r_points, centres = _unproject(root, rows, stride, use_transpose=False)
    rt_points, _ = _unproject(root, rows, stride, use_transpose=True)
    r_score = _score(r_points, "R @ p + t  (camera_to_world)")
    rt_score = _score(rt_points, "R.T @ p + t  (transposed rotation, same t)")
    winner = "R" if r_score["winner"] and not rt_score["winner"] else (
        "R.T" if rt_score["winner"] and not r_score["winner"] else "tie_or_neither")
    result = {
        "sample": root.name,
        "diagnosticOnly": True,
        "stride": stride,
        "cameraCentreExtentM": np.ptp(centres, axis=0).round(6).tolist(),
        "quaternionConversion": "Hamilton, matches StrayVisualizer",
        "R": r_score,
        "RT": rt_score,
        "winner": winner,
        "conclusion": (
            "Poses are camera-to-world: p_world = R @ p_cam + t. "
            "The IMU-attitude probe is confounded by an unknown device-to-camera "
            "offset and is not a gating signal."
            if winner == "R" else
            "Transposed R produced the coherent room; treat recorded R as world-to-camera."
            if winner == "R.T" else
            "Neither construction uniquely produced a floor and ceiling."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pose_convention_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stride", type=int, default=6)
    args = parser.parse_args()
    result = run(args.root, args.output, args.stride)
    print(json.dumps({
        "winner": result["winner"],
        "R": {k: result["R"][k] for k in (
            "floor", "ceiling", "wallCount", "roomHeightM", "floorRmsMm", "ceilingRmsMm")},
        "RT": {k: result["RT"][k] for k in (
            "floor", "ceiling", "wallCount", "roomHeightM", "floorRmsMm", "ceilingRmsMm")},
        "out": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
