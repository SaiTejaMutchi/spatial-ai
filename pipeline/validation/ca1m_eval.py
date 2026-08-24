"""Score a frozen reconstruction against CA-1M's laser-registered ground truth.

CA-1M publishes, per frame, a camera pose registered into the FARO laser
scanner's coordinate frame and a depth image rendered from that laser scan. It
does **not** publish a mobile camera pose: `dataset.py` sets the wide sensor's
`RT` to the identity and keeps everything in camera coordinates. The mobile pose
therefore comes from ARKitScenes' own `lowres_wide.traj`, which is the same
capture's on-device odometry, and CA-1M supplies only ground truth.

That split is the whole point, and this module enforces it structurally:

    reconstruction reads   ARKitScenes RGB, ARKit depth, intrinsics, ARKit poses
    evaluator reads        CA-1M gt/RT.json, gt/depth.png   (after prediction freeze)

Alignment between the two spaces is solved from the *published pose pair* alone
— ARKit pose and laser pose for the same timestamp — never by fitting to the
prediction. Fitting to the prediction would let the reconstruction choose the
frame it is judged in.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MM_TO_M = 1000.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# mobile side - everything the reconstruction was allowed to see
# --------------------------------------------------------------------------

def read_mobile_poses(capture_dir: Path) -> dict[float, np.ndarray]:
    """ARKit camera-to-world poses, keyed by timestamp, from the normalized capture."""
    trajectory = json.loads((capture_dir / "trajectory.json").read_text())
    return {float(pose["timestamp_s"]): np.asarray(pose["camera_to_world"],
                                                   dtype=np.float64)
            for pose in trajectory["poses"]}


# --------------------------------------------------------------------------
# ground-truth side - opened only after the prediction is frozen
# --------------------------------------------------------------------------

@dataclass
class GroundTruthFrame:
    timestamp_s: float
    laser_from_camera: np.ndarray   # gt/RT.json
    depth_path: Path                # gt/depth.png
    intrinsics: np.ndarray          # gt/depth/K.json


def read_ground_truth(root: Path, video_id: str) -> list[GroundTruthFrame]:
    base = root / video_id
    frames: list[GroundTruthFrame] = []
    for gt_dir in sorted(glob.glob(str(base / "*.gt"))):
        stamp = Path(gt_dir).name.split(".")[0]
        gt = Path(gt_dir)
        rt_path, depth_path, k_path = gt / "RT.json", gt / "depth.png", gt / "depth" / "K.json"
        if not (rt_path.is_file() and depth_path.is_file() and k_path.is_file()):
            continue
        frames.append(GroundTruthFrame(
            timestamp_s=float(stamp) / 1e9,
            laser_from_camera=np.asarray(
                json.loads(rt_path.read_text()), dtype=np.float64).reshape(4, 4),
            depth_path=depth_path,
            intrinsics=np.asarray(
                json.loads(k_path.read_text()), dtype=np.float64).reshape(3, 3),
        ))
    return frames


# --------------------------------------------------------------------------
# alignment from the published pose pair only
# --------------------------------------------------------------------------

@dataclass
class Alignment:
    laser_from_arkit: np.ndarray
    pairs_used: int
    rotation_residual_deg: float
    translation_residual_m: float
    inlier_fraction: float = 1.0
    pairs_available: int = 0
    direction_check: dict[str, Any] = field(default_factory=dict)


def _rigid_average(transforms: np.ndarray) -> np.ndarray:
    """Average a set of rigid transforms: SVD on rotations, mean on translations."""
    rotation = transforms[:, :3, :3].sum(axis=0)
    u, _, vt = np.linalg.svd(rotation)
    averaged = u @ vt
    if np.linalg.det(averaged) < 0:
        u[:, -1] *= -1
        averaged = u @ vt
    result = np.eye(4)
    result[:3, :3] = averaged
    result[:3, 3] = transforms[:, :3, 3].mean(axis=0)
    return result


def solve_alignment(
    mobile: dict[float, np.ndarray],
    ground_truth: list[GroundTruthFrame],
    tolerance_s: float = 0.05,
) -> Alignment:
    """T with laser_pose = T @ arkit_pose, solved per matched frame and averaged.

    The spread across frames is the evidence that the direction is right: a
    transposed or inverted convention does not produce a consistent T.
    """
    stamps = np.array(sorted(mobile))
    candidates = []
    for frame in ground_truth:
        index = int(np.argmin(np.abs(stamps - frame.timestamp_s)))
        if abs(stamps[index] - frame.timestamp_s) > tolerance_s:
            continue
        arkit = mobile[float(stamps[index])]
        candidates.append(frame.laser_from_camera @ np.linalg.inv(arkit))
    if len(candidates) < 3:
        raise ValueError(
            f"only {len(candidates)} frames matched between the ARKit trajectory and "
            f"the CA-1M ground truth; alignment cannot be solved")

    stack = np.asarray(candidates)

    # CA-1M orients every frame upright, so the camera roll can change part-way
    # through a capture. Averaging across two device orientations produces a
    # transform that fits neither, so keep the largest mutually consistent set
    # and report how much of the capture it covers.
    rotations = stack[:, :3, :3]
    agreement = np.zeros(len(stack), dtype=int)
    for i, rotation in enumerate(rotations):
        delta = np.einsum("ij,njk->nik", rotation.T, rotations)
        traces = np.trace(delta, axis1=1, axis2=2)
        agreement[i] = int((np.degrees(np.arccos(
            np.clip((traces - 1.0) / 2.0, -1.0, 1.0))) <= 5.0).sum())
    medoid = rotations[int(np.argmax(agreement))]
    delta = np.einsum("ij,njk->nik", medoid.T, rotations)
    traces = np.trace(delta, axis1=1, axis2=2)
    inliers = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))) <= 5.0
    inlier_fraction = float(inliers.mean())
    stack = stack[inliers]
    averaged = _rigid_average(stack)

    inverse_rotation = averaged[:3, :3].T
    angles = []
    for candidate in stack:
        delta = inverse_rotation @ candidate[:3, :3]
        cos = (np.trace(delta) - 1.0) / 2.0
        angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    translations = stack[:, :3, 3] - averaged[:3, 3]

    return Alignment(
        laser_from_arkit=averaged,
        pairs_used=len(stack),
        rotation_residual_deg=float(np.percentile(angles, 90)),
        translation_residual_m=float(np.percentile(np.linalg.norm(translations, axis=1), 90)),
        inlier_fraction=inlier_fraction,
        pairs_available=len(candidates),
    )


# --------------------------------------------------------------------------
# the ground-truth cloud, expressed in the prediction's canonical frame
# --------------------------------------------------------------------------

def ground_truth_cloud(
    frames: list[GroundTruthFrame],
    alignment: Alignment,
    source_to_canonical: np.ndarray,
    max_frames: int = 60,
    pixel_stride: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """FARO depth, unprojected in laser space, carried into canonical space.

    Zero depths are unregistered areas of the laser scan, not zero-range
    returns, so they are dropped and counted rather than treated as data.
    """
    if len(frames) > max_frames:
        picks = np.linspace(0, len(frames) - 1, max_frames)
        frames = [frames[int(round(p))] for p in picks]

    arkit_from_laser = np.linalg.inv(alignment.laser_from_arkit)
    chunks = []
    total = 0
    unregistered = 0
    for frame in frames:
        depth = np.array(Image.open(frame.depth_path))
        depth_m = depth.astype(np.float64) / MM_TO_M
        depth_m = depth_m[::pixel_stride, ::pixel_stride]
        total += depth_m.size
        valid = depth_m > 0
        unregistered += int((~valid).sum())
        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            continue
        z = depth_m[rows, cols]
        fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
        cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]
        camera = np.stack([(cols * pixel_stride - cx) / fx * z,
                           (rows * pixel_stride - cy) / fy * z,
                           z], axis=1)
        laser = camera @ frame.laser_from_camera[:3, :3].T + frame.laser_from_camera[:3, 3]
        arkit = laser @ arkit_from_laser[:3, :3].T + arkit_from_laser[:3, 3]
        chunks.append(arkit @ source_to_canonical[:3, :3].T + source_to_canonical[:3, 3])

    if not chunks:
        raise ValueError("no ground-truth depth survived; nothing to score against")

    points = np.concatenate(chunks)
    coverage = {
        "framesUsed": len(frames),
        "pixelStride": pixel_stride,
        "pixelsConsidered": int(total),
        "pixelsUnregistered": int(unregistered),
        "unregisteredFraction": round(unregistered / max(total, 1), 4),
        "pointsUsed": int(len(points)),
    }
    return points, coverage


def laser_space_storey_height(
    frames: list[GroundTruthFrame],
    max_frames: int = 40,
    pixel_stride: int = 3,
    up_axis: int = 2,
) -> dict[str, Any]:
    """Floor-to-ceiling separation measured in the laser scanner's own frame.

    This deliberately never touches the ARKit alignment. The laser frame is
    gravity-aligned, so a floor and a ceiling are two dense horizontal bands in
    it; fitting a plane to each and measuring between the fitted planes gives a
    reference that carries none of the alignment's rotation error. That matters
    here because the alignment residual is the capture's own odometry drift and
    is far larger than the quantity being measured.
    """
    if len(frames) > max_frames:
        picks = np.linspace(0, len(frames) - 1, max_frames)
        frames = [frames[int(round(p))] for p in picks]

    chunks = []
    for frame in frames:
        depth = np.array(Image.open(frame.depth_path)).astype(np.float64) / MM_TO_M
        depth = depth[::pixel_stride, ::pixel_stride]
        rows, cols = np.nonzero(depth > 0)
        if rows.size == 0:
            continue
        z = depth[rows, cols]
        fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
        cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]
        camera = np.stack([(cols * pixel_stride - cx) / fx * z,
                           (rows * pixel_stride - cy) / fy * z, z], axis=1)
        chunks.append(camera @ frame.laser_from_camera[:3, :3].T
                      + frame.laser_from_camera[:3, 3])
    if not chunks:
        return {"established": False, "reason": "no laser depth survived"}

    points = np.concatenate(chunks)
    heights = points[:, up_axis]
    counts, edges = np.histogram(
        heights, bins=max(int(np.ceil((heights.max() - heights.min()) / 0.02)), 1))
    centres = (edges[:-1] + edges[1:]) / 2.0
    fractions = counts / counts.sum()
    supported = np.nonzero(fractions >= 0.02)[0]
    if supported.size < 2:
        return {"established": False,
                "reason": "the laser cloud shows no two horizontal bands with enough "
                          "support in the laser frame"}

    groups: list[list[int]] = []
    for index in supported:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    bands = [(float(np.average(centres[g], weights=counts[g])), float(fractions[g].sum()))
             for g in groups]
    pairs = [(low, high, lw + hw)
             for i, (low, lw) in enumerate(bands)
             for (high, hw) in bands[i + 1:]
             if 2.0 <= (high - low) <= 4.5]
    if not pairs:
        return {"established": False,
                "reason": "no pair of laser bands lies a plausible storey apart"}

    floor, ceiling, _ = max(pairs, key=lambda item: item[2])

    def fit(centre: float):
        band = points[np.abs(points[:, up_axis] - centre) <= 0.06]
        if len(band) < 500:
            return None
        centroid = band.mean(axis=0)
        _, _, vt = np.linalg.svd(band - centroid, full_matrices=False)
        normal = vt[-1]
        if normal[up_axis] < 0:
            normal = -normal
        return normal, centroid, len(band)

    floor_fit, ceiling_fit = fit(floor), fit(ceiling)
    result = {"established": True,
              "frame": "laser scanner (gravity-aligned, +z up)",
              "alignmentFree": True,
              "floorBandCentre_m": round(floor, 4),
              "ceilingBandCentre_m": round(ceiling, 4)}
    if floor_fit and ceiling_fit:
        floor_normal, floor_centroid, floor_n = floor_fit
        ceiling_normal, ceiling_centroid, ceiling_n = ceiling_fit
        mean_normal = floor_normal + ceiling_normal
        mean_normal /= np.linalg.norm(mean_normal)
        separation = abs(float(mean_normal @ (ceiling_centroid - floor_centroid)))
        result.update({
            "storeyHeight_m": round(separation, 4),
            "floorPlanePoints": floor_n,
            "ceilingPlanePoints": ceiling_n,
            "floorCeilingTilt_deg": round(float(np.degrees(np.arccos(np.clip(
                float(floor_normal @ ceiling_normal), -1.0, 1.0)))), 3),
        })
    else:
        result.update({"storeyHeight_m": round(ceiling - floor, 4),
                       "note": "too few points to fit planes; band centres used"})
    return result


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _distance_stats(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "median_cm": round(float(np.median(absolute)) * 100, 3),
        "mean_cm": round(float(absolute.mean()) * 100, 3),
        "p90_cm": round(float(np.percentile(absolute, 90)) * 100, 3),
        "max_cm": round(float(absolute.max()) * 100, 3),
        "signedMedian_cm": round(float(np.median(values)) * 100, 3),
    }


def _inside_footprint(points: np.ndarray, footprint: list[list[float]],
                      margin_m: float = 0.15) -> np.ndarray:
    """Ray-cast test on the plan polygon, dilated slightly to keep wall skins."""
    polygon = np.asarray(footprint, dtype=np.float64)
    if polygon.ndim != 2 or len(polygon) < 3:
        return np.ones(len(points), dtype=bool)
    centre = polygon.mean(axis=0)
    spread = polygon - centre
    scale = 1.0 + margin_m / max(float(np.abs(spread).max()), 1e-6)
    polygon = centre + spread * scale

    x, z = points[:, 0], points[:, 2]
    inside = np.zeros(len(points), dtype=bool)
    count = len(polygon)
    for i in range(count):
        x0, z0 = polygon[i]
        x1, z1 = polygon[(i + 1) % count]
        straddles = (z0 > z) != (z1 > z)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = (x1 - x0) * (z - z0) / (z1 - z0) + x0
        inside ^= straddles & (x < crossing)
    return inside


def score_surfaces(
    model: dict[str, Any],
    points: np.ndarray,
    band_m: float = 0.25,
    min_support: int = 500,
) -> dict[str, Any]:
    """Distance from laser points to each predicted plane.

    Points are attributed to a surface by proximity band, and the count that
    falls in the band is reported alongside the error. A surface with almost no
    laser support has not been validated, however small its residual looks, so
    support is reported rather than hidden.
    """
    rooms = model.get("rooms") or []
    footprint = rooms[0].get("footprint") if rooms else None
    within_room = (_inside_footprint(points, footprint) if footprint
                   else np.ones(len(points), dtype=bool))

    # Co-planar segments share one plane, so a plane-distance metric describes
    # the plane, not each segment. Say so rather than reporting it twice as if
    # they were independent evidence.
    seen_planes: dict[tuple, str] = {}

    results = []
    for surface in model.get("surfaces", []):
        plane = surface.get("plane") or {}
        normal = np.asarray(plane.get("normal", []), dtype=np.float64)
        offset = plane.get("offset_m")
        entry: dict[str, Any] = {
            "surfaceId": surface["id"],
            "type": surface["type"],
            "observationState": surface.get("observationState"),
        }
        if normal.size != 3 or offset is None or np.linalg.norm(normal) < 1e-9:
            entry.update({
                "scored": False,
                "laserPointsInBand": 0,
                "reason": "this surface carries no fitted plane; it is an inferred "
                          "closure, and there is nothing to measure a distance to",
            })
            results.append(entry)
            continue
        normal = normal / np.linalg.norm(normal)

        key = (tuple(np.round(normal, 4)), round(float(offset), 4))
        duplicate_of = seen_planes.get(key)
        seen_planes.setdefault(key, surface["id"])

        signed_all = points @ normal - float(offset)
        inside = (np.abs(signed_all) <= band_m) & within_room
        signed = signed_all
        support = int(inside.sum())
        if duplicate_of:
            entry["coplanarWith"] = duplicate_of
        entry["laserPointsInBand"] = support
        entry["bandHalfWidth_m"] = band_m
        entry["restrictedToRoomFootprint"] = footprint is not None
        if support >= min_support:
            entry.update(_distance_stats(signed[inside]))
            entry["scored"] = True
        else:
            entry["scored"] = False
            entry["reason"] = (f"only {support} laser points fall within "
                               f"{band_m} m of this plane; too little support to score")
        results.append(entry)
    return {"surfaces": results}


def independent_storey_height(points: np.ndarray, bin_m: float = 0.02,
                              min_fraction: float = 0.02) -> dict[str, Any]:
    """Floor and ceiling fitted from the laser cloud alone, not from the prediction."""
    heights = points[:, 1]
    counts, edges = np.histogram(
        heights, bins=max(int(np.ceil((heights.max() - heights.min()) / bin_m)), 1))
    centres = (edges[:-1] + edges[1:]) / 2.0
    fractions = counts / counts.sum()
    supported = np.nonzero(fractions >= min_fraction)[0]
    if supported.size < 2:
        return {"established": False,
                "reason": "the laser cloud shows no two horizontal bands with enough support"}
    groups: list[list[int]] = []
    for index in supported:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    bands = [(float(np.average(centres[g], weights=counts[g])), float(fractions[g].sum()))
             for g in groups]
    pairs = [(low, high, lw + hw)
             for i, (low, lw) in enumerate(bands)
             for (high, hw) in bands[i + 1:]
             if 2.0 <= (high - low) <= 4.5]
    if not pairs:
        return {"established": False,
                "reason": "no pair of laser bands lies a plausible storey apart"}
    floor, ceiling, _ = max(pairs, key=lambda item: item[2])

    # Band centres are sensitive to any tilt left over from alignment. Fitting a
    # plane to each band and measuring between the fitted planes is invariant
    # under a rigid rotation, so the reference height does not inherit the
    # alignment's rotation error.
    def fit(centre: float) -> tuple[np.ndarray, float, int] | None:
        band = points[np.abs(points[:, 1] - centre) <= 0.06]
        if len(band) < 500:
            return None
        centroid = band.mean(axis=0)
        _, _, vt = np.linalg.svd(band - centroid, full_matrices=False)
        normal = vt[-1]
        if normal[1] < 0:
            normal = -normal
        return normal, float(normal @ centroid), len(band)

    floor_fit, ceiling_fit = fit(floor), fit(ceiling)
    result = {"established": True,
              "method": "band centres, then a least-squares plane per band",
              "floorBandCentre_m": round(floor, 4),
              "ceilingBandCentre_m": round(ceiling, 4),
              "bandCentreSeparation_m": round(ceiling - floor, 4)}
    if floor_fit and ceiling_fit:
        floor_normal, floor_offset, floor_n = floor_fit
        ceiling_normal, ceiling_offset, ceiling_n = ceiling_fit
        mean_normal = floor_normal + ceiling_normal
        mean_normal /= np.linalg.norm(mean_normal)
        separation = abs(float(mean_normal @ (ceiling_normal * ceiling_offset
                                              - floor_normal * floor_offset)))
        result.update({
            "storeyHeight_m": round(separation, 4),
            "floorPlanePoints": floor_n,
            "ceilingPlanePoints": ceiling_n,
            "floorCeilingTilt_deg": round(float(np.degrees(np.arccos(
                np.clip(float(floor_normal @ ceiling_normal), -1.0, 1.0)))), 3),
            "rotationInvariant": True,
        })
    else:
        result["storeyHeight_m"] = round(ceiling - floor, 4)
        result["rotationInvariant"] = False
        result["note"] = "too few points to fit planes; band centres used"
    return result


# --------------------------------------------------------------------------
# one capture, end to end
# --------------------------------------------------------------------------

def evaluate(
    capture_dir: Path,
    model_path: Path,
    ca1m_root: Path,
    video_id: str,
    prediction_freeze: Path | None = None,
) -> dict[str, Any]:
    model = json.loads(model_path.read_text())
    frame = model["coordinateSystem"]["frame"]
    source_to_canonical = np.asarray(frame["sourceToCanonical"], dtype=np.float64)
    if frame.get("floorOriginApplied") or frame.get("roomAxisAlignmentApplied"):
        raise ValueError(
            "this evaluator composes only the source-to-canonical rotation; the model "
            "reports a further origin or axis alignment that would have to be composed too")

    prediction_sha = sha256_file(model_path)
    mobile = read_mobile_poses(capture_dir)

    # Everything below this line reads ground truth.
    gt_read_utc = utc_now()
    ground_truth = read_ground_truth(ca1m_root, video_id)
    if not ground_truth:
        raise ValueError(f"no CA-1M ground-truth frames found for {video_id}")

    alignment = solve_alignment(mobile, ground_truth)
    points, coverage = ground_truth_cloud(ground_truth, alignment, source_to_canonical)

    # Gate item 3: a known camera-frame depth point must land somewhere sensible.
    probe = ground_truth[len(ground_truth) // 2]
    probe_depth = np.array(Image.open(probe.depth_path)).astype(np.float64) / MM_TO_M
    rows, cols = np.nonzero(probe_depth > 0)
    pick = len(rows) // 2
    z = probe_depth[rows[pick], cols[pick]]
    fx, fy = probe.intrinsics[0, 0], probe.intrinsics[1, 1]
    cx, cy = probe.intrinsics[0, 2], probe.intrinsics[1, 2]
    camera_point = np.array([(cols[pick] - cx) / fx * z, (rows[pick] - cy) / fy * z, z])
    laser_point = probe.laser_from_camera[:3, :3] @ camera_point + probe.laser_from_camera[:3, 3]
    canonical_point = (source_to_canonical[:3, :3]
                       @ (np.linalg.inv(alignment.laser_from_arkit)[:3, :3] @ laser_point
                          + np.linalg.inv(alignment.laser_from_arkit)[:3, 3]))
    alignment.direction_check = {
        "cameraFrameDepth_m": round(float(z), 4),
        "cameraFramePoint": [round(float(v), 4) for v in camera_point],
        "laserSpacePoint": [round(float(v), 4) for v in laser_point],
        "canonicalPoint": [round(float(v), 4) for v in canonical_point],
        "heightAboveCanonicalFloor_m": round(float(canonical_point[1]), 4),
    }

    # The reference height is measured in laser space, so it does not inherit the
    # alignment's error. Surface distances cannot avoid the alignment, so they are
    # only reported when it is trustworthy.
    storey = laser_space_storey_height(ground_truth)
    alignment_trustworthy = (alignment.rotation_residual_deg <= 5.0
                             and alignment.inlier_fraction >= 0.6)
    if alignment_trustworthy:
        scored = score_surfaces(model, points)
    else:
        scored = {"surfaces": [], "notComparable": True,
                  "reason": (f"the ARKit-to-laser alignment residual is "
                             f"{alignment.rotation_residual_deg:.1f} deg over "
                             f"{alignment.inlier_fraction:.0%} of matched frames. A "
                             f"surface distance measured through that transform would "
                             f"describe the alignment, not the geometry.")}

    measurements = {m["type"]: m.get("value_m", m.get("value"))
                    for m in model.get("measurements", [])}
    height_error = None
    if storey.get("established") and measurements.get("room_height") is not None:
        height_error = {
            "model_m": measurements["room_height"],
            "laserReference_m": storey["storeyHeight_m"],
            "absoluteError_cm": round(
                abs(measurements["room_height"] - storey["storeyHeight_m"]) * 100, 3),
            "assignmentGate": "at most 1.5 cm",
        }
        height_error["signedError_cm"] = round(
            (measurements["room_height"] - storey["storeyHeight_m"]) * 100, 3)
        height_error["result"] = (
            "pass" if height_error["absoluteError_cm"] <= 1.5 else "fail")

    surfaces = scored["surfaces"]
    unscored = [s for s in surfaces if not s["scored"]]
    height_gate_note = None
    if height_error and storey.get("alignmentFree"):
        height_gate_note = ("Measured in laser space and compared with an internal "
                            "model scalar, so neither side passes through the "
                            "ARKit-to-laser alignment.")
    return {
        "captureId": video_id,
        "heldOut": True,
        "predictionSha256": prediction_sha,
        "predictionFreezeRecord": (json.loads(prediction_freeze.read_text())
                                   if prediction_freeze and prediction_freeze.is_file() else None),
        "groundTruthReadUtc": gt_read_utc,
        "geometryConfigHash": model.get("provenance", {}).get("geometryConfigHash"),
        "surfaceDistancesComparable": alignment_trustworthy,
        "surfaceDistancesNotComparableReason": scored.get("reason"),
        "roomHeightMethodNote": height_gate_note,
        "alignment": {
            "solvedFrom": "published ARKit pose and CA-1M laser pose for the same "
                          "timestamp; never fitted to the prediction",
            "pairsUsed": alignment.pairs_used,
            "pairsAvailable": alignment.pairs_available,
            "inlierFraction": round(alignment.inlier_fraction, 4),
            "rotationResidualP90_deg": round(alignment.rotation_residual_deg, 4),
            "translationResidualP90_m": round(alignment.translation_residual_m, 4),
            "directionCheck": alignment.direction_check,
        },
        "coverage": coverage,
        "storeyHeightFromLaser": storey,
        "alignmentUncertainty": {
            "rotationResidualP90_deg": round(alignment.rotation_residual_deg, 4),
            "impliedHorizontalBandSmear_cm": round(
                float(np.ptp(points[:, [0, 2]], axis=0).max() / 2.0
                      * np.sin(np.radians(alignment.rotation_residual_deg)) * 100), 2),
            "note": "The laser reference is carried into the prediction's frame by a "
                    "transform solved from pose pairs. Its rotation residual tilts the "
                    "cloud, which smears the horizontal floor and ceiling bands by "
                    "roughly the figure above across the room. Errors of that order "
                    "are not separable from alignment.",
        },
        "roomHeightError": height_error,
        "surfaceDistances": surfaces,
        "failures": {
            "surfacesWithoutLaserSupport": len(unscored),
            "surfaceIds": [s["surfaceId"] for s in unscored],
        },
        "measurements_m": measurements,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="normalized capture directory")
    parser.add_argument("model", type=Path, help="frozen spatial_model.json")
    parser.add_argument("ca1m_root", type=Path, help="extracted CA-1M directory")
    parser.add_argument("video_id")
    parser.add_argument("--prediction-freeze", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    result = evaluate(args.capture, args.model, args.ca1m_root, args.video_id,
                      args.prediction_freeze)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"-> {args.output}")
    align = result["alignment"]
    print(f"alignment: {align['pairsUsed']} pose pairs, "
          f"rotation p90 {align['rotationResidualP90_deg']} deg, "
          f"translation p90 {align['translationResidualP90_m']} m")
    print(f"coverage : {result['coverage']['pointsUsed']} laser points, "
          f"{result['coverage']['unregisteredFraction']:.1%} of GT pixels unregistered")
    if result["roomHeightError"]:
        h = result["roomHeightError"]
        print(f"height   : model {h['model_m']} m vs laser {h['laserReference_m']} m "
              f"-> {h['absoluteError_cm']} cm ({h['result']})")
    for surface in result["surfaceDistances"]:
        if surface["scored"]:
            print(f"  {surface['surfaceId']:12s} {surface['type']:8s} "
                  f"median {surface['median_cm']:6.2f} cm  p90 {surface['p90_cm']:6.2f} cm  "
                  f"n={surface['laserPointsInBand']}")
        else:
            print(f"  {surface['surfaceId']:12s} {surface['type']:8s} NOT SCORED "
                  f"({surface['laserPointsInBand']} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
