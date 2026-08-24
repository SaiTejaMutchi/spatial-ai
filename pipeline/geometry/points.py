"""Turn a `normalized_capture` into canonical-frame metric points.

This is the single place where depth pixels become world coordinates. Every
threshold it applies comes from `geometry_config_v0.1.json`, and every filter
reports how much it discarded, so the cost of each choice stays visible instead
of disappearing into a point count.

Nothing here fits a plane or measures a room. It produces the metric substrate
that plane fitting and everything downstream consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..contracts.normalized_capture import NormalizedCapture
from .config import GeometryConfig
from .frame import CanonicalFrame, resolve_canonical_frame

CONFIG_PARAMETERS = (
    "depth_min_m", "depth_max_m", "min_confidence_level", "voxel_size_m",
    "gravity_agreement_tolerance_deg", "min_poses_for_gravity_estimate",
)


@dataclass
class PointCloud:
    """Canonical-frame points with the record of how they were produced."""

    points: np.ndarray                 # (N, 3) metres, canonical frame
    frame: CanonicalFrame
    frame_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    diagnostics: dict[str, Any] = field(default_factory=dict)
    config_provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.points))

    def extent(self) -> np.ndarray:
        if len(self.points) == 0:
            return np.zeros(3)
        return self.points.max(axis=0) - self.points.min(axis=0)


def voxel_downsample(
    points: np.ndarray,
    voxel_size_m: float,
    labels: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Keep one representative per occupied voxel: the mean of its members.

    Averaging rather than picking an arbitrary member keeps sub-voxel accuracy,
    so the grid bounds the point count without biasing a later plane fit toward
    whichever sample happened to be first.

    When `labels` is given (per-point frame indices), each surviving point also
    carries the label of one contributing member, so downstream code can still
    report how many distinct observations support a surface.
    """
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if len(points) == 0:
        return (points, labels) if labels is not None else points
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, first, inverse, counts = np.unique(
        keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    reduced = sums / counts[:, None]
    if labels is None:
        return reduced
    return reduced, np.asarray(labels)[first]


def _load_depth_intrinsics(capture: NormalizedCapture):
    for intrinsics in capture.intrinsics:
        if intrinsics.stream == "depth":
            return intrinsics
    raise ValueError(
        "the capture declares no 'depth' intrinsics stream; depth cannot be unprojected")


def build_point_cloud(
    root: Path,
    capture: NormalizedCapture,
    config: GeometryConfig,
    stride: int = 1,
    max_frames: int | None = None,
) -> PointCloud:
    root = Path(root)
    if not capture.frames:
        raise ValueError("the capture contains no frames; there is nothing to unproject")

    depth_min = config.get("depth_min_m")
    depth_max = config.get("depth_max_m")
    min_confidence = config.get("min_confidence_level")
    voxel_size = config.get("voxel_size_m")
    tolerance_deg = config.get("gravity_agreement_tolerance_deg")
    min_poses = config.get("min_poses_for_gravity_estimate")

    poses = np.array([f.camera_to_world for f in capture.frames], dtype=np.float64)
    declared = capture.world_up_axis
    verified = bool(capture.world_up_axis_verified)
    ingestion_record = None
    from ..contracts.frame_resolution import FrameResolutionError, resolve_frame
    try:
        resolution = resolve_frame(root, capture)
    except FrameResolutionError:
        resolution = None
    else:
        ingestion_record = resolution.to_record()
        if resolution.accepted:
            declared = resolution.axis
            verified = True

    frame = resolve_canonical_frame(
        camera_to_world=poses,
        declared_up_axis=declared,
        world_frame=capture.world_frame,
        tolerance_deg=tolerance_deg,
        min_poses=min_poses,
        declaration_verified=verified,
    )
    if ingestion_record is not None:
        frame.diagnostics["ingestionFrameResolution"] = ingestion_record

    intrinsics = _load_depth_intrinsics(capture)
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy

    selected = capture.frames[::max(1, stride)]
    if max_frames is not None:
        selected = selected[:max_frames]

    confidence_available = capture.modalities.get("confidence", {}).get("available") is True

    total_pixels = 0
    kept_no_return = 0
    kept_range = 0
    kept_confidence = 0
    chunks: list[np.ndarray] = []
    origins: list[np.ndarray] = []

    for record in selected:
        depth_raw = np.array(Image.open(root / record.depth))
        total_pixels += depth_raw.size

        valid = depth_raw > 0
        kept_no_return += int(valid.sum())

        depth_m = depth_raw.astype(np.float64) * capture.depth_scale_m
        valid &= (depth_m >= depth_min) & (depth_m <= depth_max)
        kept_range += int(valid.sum())

        if confidence_available and record.confidence is not None:
            confidence = np.array(Image.open(root / record.confidence))
            valid &= confidence >= min_confidence
        kept_confidence += int(valid.sum())

        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            continue
        z = depth_m[rows, cols]
        camera = np.stack([
            (cols - cx) / fx * z,
            (rows - cy) / fy * z,
            z,
        ], axis=1)
        pose = np.asarray(record.camera_to_world, dtype=np.float64)
        chunks.append(camera @ pose[:3, :3].T + pose[:3, 3])
        origins.append(np.full(rows.size, record.index, dtype=np.int32))

    if not chunks:
        raise ValueError(
            f"no depth pixel survived filtering across {len(selected)} frames "
            f"(range {depth_min}-{depth_max} m, minimum confidence {min_confidence}); "
            f"the capture cannot support geometry")

    source_points = np.concatenate(chunks)
    source_frames = np.concatenate(origins)
    canonical = frame.apply(source_points)
    before_voxel = len(canonical)
    canonical, canonical_frames = voxel_downsample(canonical, voxel_size, source_frames)

    diagnostics = {
        "framesUsed": len(selected),
        "framesAvailable": len(capture.frames),
        "frameStride": stride,
        "pixelsExamined": total_pixels,
        "retainedAfterNoReturnFilter": kept_no_return,
        "retainedAfterRangeFilter": kept_range,
        "retainedAfterConfidenceFilter": kept_confidence,
        "confidenceFilterApplied": confidence_available,
        "retainedFractionOfPixels": round(kept_confidence / max(total_pixels, 1), 6),
        "pointsBeforeVoxelDownsample": before_voxel,
        "pointsAfterVoxelDownsample": int(len(canonical)),
        "contributingFrames": int(len(np.unique(canonical_frames))),
        "canonicalExtentM": [round(float(v), 4)
                             for v in (canonical.max(0) - canonical.min(0))],
        "canonicalBoundsM": {
            "min": [round(float(v), 4) for v in canonical.min(0)],
            "max": [round(float(v), 4) for v in canonical.max(0)],
        },
    }

    return PointCloud(
        points=canonical,
        frame=frame,
        frame_indices=canonical_frames,
        diagnostics=diagnostics,
        config_provenance=config.provenance(*CONFIG_PARAMETERS),
    )
