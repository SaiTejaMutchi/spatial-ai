"""Readability and coverage checks for an ARKitScenes development fixture.

This proves the raw modalities decode, are metrically
plausible, and have usable pose coverage. It performs no reconstruction,
no coordinate normalization, and no parameter tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

# Documented ARKitScenes raw conventions (see Apple ARKitScenes DATA.md).
DEPTH_UNITS_PER_METRE = 1000.0  # lowres_depth PNGs are uint16 millimetres
CONFIDENCE_LEVELS = {0, 1, 2}   # ARKit ARConfidenceLevel low/medium/high
# Sanity bounds, not tuned thresholds: a handheld indoor room scan.
PLAUSIBLE_DEPTH_M = (0.1, 12.0)
POSE_MATCH_TOLERANCE_S = 0.05   # half of the observed 10 Hz pose interval
# ARKit reports intrinsics per frame and they jitter slightly with autofocus.
# This bound only asserts "one stable camera model", not a calibrated value.
MAX_FOCAL_RELATIVE_SPREAD = 0.02
# A handheld room scan is held at roughly constant height, so vertical travel is
# small by nature. What matters here is that the camera moved enough to give
# parallax, not that it traversed the room's full extent.
MIN_PATH_LENGTH_M = 3.0
MIN_CAMERA_EXTENT_M = 1.0


def frame_timestamp(path: Path) -> float:
    return float(path.stem.split("_", 1)[1])


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle to rotation matrix, matching cv2.Rodrigues."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-12:
        return np.eye(3)
    k = axis_angle / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def camera_positions(traj: np.ndarray) -> np.ndarray:
    """Camera centres in world coordinates.

    ARKitScenes `lowres_wide.traj` columns 2-7 are the WORLD-TO-CAMERA rotation
    (axis-angle) and translation; Apple's own loader inverts them to obtain the
    camera-to-world pose. Reading the raw translations as camera positions is a
    silent frame error, so the inversion happens here once.
    """
    centres = np.empty((len(traj), 3))
    for i, row in enumerate(traj):
        rot = rodrigues(row[1:4])
        centres[i] = -rot.T @ row[4:7]
    return centres


def check(label: str, ok: bool, detail: str, results: list) -> None:
    results.append({"check": label, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")


def validate_scene(scene_dir: Path, samples: int) -> dict:
    print(f"\nscene: {scene_dir.relative_to(REPO_ROOT)}")
    results: list = []

    depth = sorted(scene_dir.glob("lowres_depth/*.png"))
    rgb = sorted(scene_dir.glob("lowres_wide/*.png"))
    conf = sorted(scene_dir.glob("confidence/*.png"))
    intr = sorted(scene_dir.glob("lowres_wide_intrinsics/*.pincam"))
    traj_path = scene_dir / "lowres_wide.traj"

    check("core modalities present",
          bool(depth and intr and traj_path.exists()),
          f"depth={len(depth)} rgb={len(rgb)} confidence={len(conf)} "
          f"intrinsics={len(intr)} trajectory={traj_path.exists()}",
          results)

    if not (depth and intr and traj_path.exists()):
        return {"scene": scene_dir.name, "results": results, "ok": False}

    idx = np.linspace(0, len(depth) - 1, min(samples, len(depth))).astype(int)

    depth_mins, depth_maxs, dtypes, shapes = [], [], set(), set()
    for i in idx:
        arr = np.array(Image.open(depth[i]))
        dtypes.add(str(arr.dtype))
        shapes.add(arr.shape)
        valid = arr[arr > 0]
        if valid.size:
            depth_mins.append(valid.min() / DEPTH_UNITS_PER_METRE)
            depth_maxs.append(valid.max() / DEPTH_UNITS_PER_METRE)

    check("depth decodes as uint16 at one resolution",
          dtypes == {"uint16"} and len(shapes) == 1,
          f"dtypes={sorted(dtypes)} shapes={sorted(shapes)}", results)

    lo, hi = min(depth_mins), max(depth_maxs)
    check("depth is metrically plausible for an indoor room",
          PLAUSIBLE_DEPTH_M[0] <= lo and hi <= PLAUSIBLE_DEPTH_M[1],
          f"observed valid range {lo:.2f}-{hi:.2f} m over {len(idx)} sampled frames", results)

    if conf:
        levels = set()
        for i in idx:
            levels |= set(np.unique(np.array(Image.open(conf[i]))).tolist())
        check("confidence uses documented ARKit levels",
              levels <= CONFIDENCE_LEVELS,
              f"observed levels {sorted(levels)}", results)

    if rgb:
        rgb_shapes = {np.array(Image.open(rgb[i])).shape for i in idx}
        check("RGB decodes at one resolution",
              len(rgb_shapes) == 1, f"shapes={sorted(rgb_shapes)}", results)

    fx_list, wh = [], set()
    for i in idx:
        w, h, fx, fy, cx, cy = (float(v) for v in intr[i].read_text().split())
        wh.add((int(w), int(h)))
        fx_list.append((fx, fy, cx, cy))
    fx = np.array(fx_list)
    width, height = list(wh)[0]
    focal_spread = float(fx[:, 0].std() / fx[:, 0].mean())
    check("intrinsics are one stable camera model inside the image",
          len(wh) == 1
          and focal_spread < MAX_FOCAL_RELATIVE_SPREAD
          and (fx[:, 2] > 0).all() and (fx[:, 2] < width).all()
          and (fx[:, 3] > 0).all() and (fx[:, 3] < height).all(),
          f"size={width}x{height} fx={fx[:,0].mean():.3f} fy={fx[:,1].mean():.3f} "
          f"cx={fx[:,2].mean():.3f} cy={fx[:,3].mean():.3f} "
          f"focal_spread={focal_spread*100:.2f}% (limit {MAX_FOCAL_RELATIVE_SPREAD*100:.0f}%)",
          results)

    rows = [line.split() for line in traj_path.read_text().strip().splitlines()]
    traj = np.array([[float(v) for v in r] for r in rows])
    pose_ts = traj[:, 0]

    check("trajectory is monotonic in time",
          bool((np.diff(pose_ts) > 0).all()),
          f"{len(pose_ts)} poses over {pose_ts.max()-pose_ts.min():.1f} s "
          f"at ~{1/np.median(np.diff(pose_ts)):.1f} Hz", results)

    frame_ts = np.array([frame_timestamp(p) for p in depth])
    ins = np.clip(np.searchsorted(pose_ts, frame_ts), 1, len(pose_ts) - 1)
    gap = np.minimum(np.abs(frame_ts - pose_ts[ins - 1]), np.abs(frame_ts - pose_ts[ins]))
    matched = float((gap <= POSE_MATCH_TOLERANCE_S).mean())
    check("depth frames have nearby poses",
          matched >= 0.95,
          f"{matched*100:.1f}% of {len(frame_ts)} depth frames within "
          f"{POSE_MATCH_TOLERANCE_S*1000:.0f} ms of a pose (max gap {gap.max()*1000:.0f} ms)",
          results)

    centres = camera_positions(traj)
    extent = centres.max(0) - centres.min(0)
    path_length = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum())
    check("camera moved enough to give parallax",
          path_length >= MIN_PATH_LENGTH_M and float(extent.max()) >= MIN_CAMERA_EXTENT_M,
          f"camera-to-world path length {path_length:.2f} m "
          f"(min {MIN_PATH_LENGTH_M}), camera centre extent "
          f"{np.round(extent, 2).tolist()} m", results)

    ok = all(r["status"] == "PASS" for r in results)
    return {
        "scene": scene_dir.name,
        "frames": {"depth": len(depth), "rgb": len(rgb), "confidence": len(conf),
                   "intrinsics": len(intr), "poses": int(len(pose_ts))},
        "duration_s": round(float(frame_ts.max() - frame_ts.min()), 2),
        "depth_range_m": [round(lo, 2), round(hi, 2)],
        "intrinsics": {"width": width, "height": height,
                       "fx_mean": round(float(fx[:, 0].mean()), 3),
                       "fy_mean": round(float(fx[:, 1].mean()), 3),
                       "cx_mean": round(float(fx[:, 2].mean()), 3),
                       "cy_mean": round(float(fx[:, 3].mean()), 3),
                       "focal_relative_spread": round(focal_spread, 5)},
        "pose_match_rate": round(matched, 4),
        "camera_centre_extent_m": [round(float(v), 2) for v in extent],
        "camera_path_length_m": round(path_length, 2),
        "results": results,
        "ok": ok,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", type=Path,
                        help="scene directories; defaults to every fixture in the manifest")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--report", type=Path,
                        default=REPO_ROOT / "samples" / "arkitscenes" / "fixture_validation.json")
    args = parser.parse_args(argv)

    scenes = args.scenes
    if not scenes:
        manifest = json.loads((REPO_ROOT / "samples" / "arkitscenes" / "fixture_manifest.json").read_text())
        scenes = [REPO_ROOT / f["assets"]["scene_dir"]["path"] for f in manifest["fixtures"]]

    reports = [validate_scene(Path(s).resolve(), args.samples) for s in scenes]
    args.report.write_text(json.dumps(reports, indent=2) + "\n")
    ok = all(r["ok"] for r in reports)
    print(f"\nFIXTURE VALIDATION: {'PASS' if ok else 'FAIL'}  -> {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
