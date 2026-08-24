"""Select and register the RGB views that will back AI review.

The Spatial AI Verifier is allowed 3-8 evidence frames, and which frames those
are decides what the model can honestly be asked about. A frame is useful here
only if it can be tied to named geometry, so selection is by *measured
visibility*: a surface's own supporting points are projected into each posed
camera, and a frame earns a surface when enough of that surface actually lands
inside the image in front of the lens.

Frames are then chosen greedily for coverage rather than by score alone, so a
small set spans as many distinct surfaces as possible instead of returning
eight views of the same wall.

Every selected view carries its provenance: source frame, timestamp, the pose
that registered it, how far that pose was from the frame in time, and the
surface IDs it sees with the visibility that earned each one. A view that
cannot be tied to geometry is labelled unregistered rather than quietly used.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.normalized_capture import NormalizedCapture
from ..geometry.config import GeometryConfig
from ..geometry.planes import Plane
from ..geometry.points import PointCloud

CONFIG_PARAMETERS = (
    "evidence_min_surface_visibility", "evidence_max_views", "evidence_min_views",
    "plane_inlier_distance_m",
)


@dataclass
class EvidenceView:
    view_id: str
    frame_index: int
    timestamp_s: float
    rgb_path: str
    source_rgb: str
    camera_to_world: list[list[float]]
    camera_to_world_canonical: list[list[float]]
    pose_time_offset_s: float
    visible_surfaces: dict[str, float]
    registration: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "id": self.view_id,
            "type": "rgb_frame",
            "frameIndex": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 6),
            "path": self.rgb_path,
            "sourceFrame": self.source_rgb,
            "cameraToWorld": self.camera_to_world,
            # The canonical-frame pose is what any consumer needs: surfaces,
            # planes, and the point cloud all live in the canonical frame, so a
            # source-frame pose would silently project into the wrong space.
            "cameraToWorldCanonical": self.camera_to_world_canonical,
            "poseTimeOffset_s": round(self.pose_time_offset_s, 6),
            "registration": self.registration,
            "visibleSurfaceIds": sorted(self.visible_surfaces),
            "surfaceVisibility": {k: round(v, 4)
                                  for k, v in sorted(self.visible_surfaces.items())},
            "producer": "geometry_pipeline",
            "diagnostics": self.diagnostics,
        }


def _surface_points(cloud: PointCloud, plane: Plane, tolerance: float) -> np.ndarray:
    mask = np.abs(cloud.points @ plane.normal - plane.offset) <= tolerance
    return cloud.points[mask]


def surface_visibility(
    surface_points: np.ndarray,
    world_to_camera: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    sample: int = 4000,
) -> float:
    """Fraction of a surface's own points that land inside this image.

    Points are taken to the camera frame, dropped if behind the lens, projected
    with the same pinhole model the geometry used, and counted if they fall
    inside the sensor. No occlusion test: a wall behind furniture still counts
    as visible, which is the honest reading for evidence selection because the
    verifier is being asked about occlusion in the first place.
    """
    if len(surface_points) == 0:
        return 0.0
    points = surface_points
    if len(points) > sample:
        step = max(len(points) // sample, 1)
        points = points[::step]

    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    in_front = camera[:, 2] > 1e-6
    if not in_front.any():
        return 0.0
    camera = camera[in_front]
    u = camera[:, 0] / camera[:, 2] * fx + cx
    v = camera[:, 1] / camera[:, 2] * fy + cy
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return float(inside.sum()) / float(len(points))


def select_evidence_views(
    capture_root: Path,
    capture: NormalizedCapture,
    cloud: PointCloud,
    model: dict,
    planes_by_id: dict[str, Plane],
    config: GeometryConfig,
    output_dir: Path,
    frame_stride: int = 1,
) -> tuple[list[EvidenceView], dict]:
    capture_root = Path(capture_root)
    min_visibility = config.get("evidence_min_surface_visibility")
    max_views = int(config.get("evidence_max_views"))
    min_views = int(config.get("evidence_min_views"))
    tolerance = config.get("plane_inlier_distance_m")

    if capture.modalities.get("rgb", {}).get("available") is not True:
        return [], {
            "available": False,
            "reason": ("The capture carries no RGB, so no evidence view can be "
                       "registered and AI review has nothing to look at."),
            "viewsSelected": 0,
        }

    intrinsics = next((i for i in capture.intrinsics if i.stream == "rgb"), None)
    depth_intrinsics = next(i for i in capture.intrinsics if i.stream == "depth")
    if intrinsics is None:
        intrinsics = depth_intrinsics

    # Only surfaces backed by a fitted plane can be seen; inferred closure has
    # no physical surface to appear in a photograph.
    targets: dict[str, np.ndarray] = {}
    for surface in model["surfaces"]:
        plane_id = surface["provenance"].get("sourcePlaneId")
        plane = planes_by_id.get(plane_id) if plane_id else None
        if plane is None:
            continue
        targets[surface["id"]] = _surface_points(cloud, plane, tolerance)

    frames = [f for f in capture.frames if f.rgb is not None][::max(1, frame_stride)]
    scored: list[tuple[dict[str, float], Any]] = []
    for frame in frames:
        pose = np.asarray(frame.camera_to_world, dtype=np.float64)
        world_to_camera = np.linalg.inv(pose)
        # The evidence lives in the canonical frame, so the pose must be too.
        canonical_w2c = world_to_camera @ np.linalg.inv(
            cloud.frame.source_to_canonical)
        visible = {}
        for surface_id, points in targets.items():
            fraction = surface_visibility(
                points, canonical_w2c,
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                intrinsics.width, intrinsics.height)
            if fraction >= min_visibility:
                visible[surface_id] = fraction
        if visible:
            scored.append((visible, frame))

    # Greedy cover: each additional view is the one that adds the most surfaces
    # not already evidenced, so a small set spans the room rather than repeating
    # the best-lit wall.
    chosen: list[tuple[dict[str, float], Any]] = []
    covered: set[str] = set()
    remaining = list(scored)
    while remaining and len(chosen) < max_views:
        def gain(item):
            visible, _ = item
            new = set(visible) - covered
            return (len(new), sum(visible[s] for s in new))
        best = max(remaining, key=gain)
        if not (set(best[0]) - covered):
            break
        chosen.append(best)
        covered |= set(best[0])
        remaining.remove(best)

    # Top up toward the verifier's minimum. Prefer views far in time from those
    # already chosen: two consecutive frames a tenth of a second apart are the
    # same photograph twice, and give the verifier nothing to cross-check.
    if len(chosen) < min_views:
        while remaining and len(chosen) < min_views:
            def distinctness(item):
                _, frame = item
                if not chosen:
                    return (float("inf"), 0.0)
                gap = min(abs(frame.timestamp_s - c[1].timestamp_s) for c in chosen)
                return (gap, sum(item[0].values()))
            best = max(remaining, key=distinctness)
            chosen.append(best)
            remaining.remove(best)
    chosen.sort(key=lambda item: item[1].index)
    to_canonical = cloud.frame.source_to_canonical

    evidence_dir = Path(output_dir) / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    views: list[EvidenceView] = []
    for order, (visible, frame) in enumerate(chosen, start=1):
        name = f"evidence-{order:03d}.png"
        shutil.copyfile(capture_root / frame.rgb, evidence_dir / name)
        views.append(EvidenceView(
            view_id=f"frame-{frame.index:06d}",
            frame_index=frame.index,
            timestamp_s=frame.timestamp_s,
            rgb_path=f"evidence/{name}",
            source_rgb=frame.rgb,
            camera_to_world=frame.camera_to_world,
            camera_to_world_canonical=(
                to_canonical @ np.asarray(frame.camera_to_world, dtype=np.float64)
            ).tolist(),
            pose_time_offset_s=frame.pose_time_offset_s,
            visible_surfaces=visible,
            registration="registered_by_capture_pose",
            diagnostics={
                "poseTimeOffsetMs": round(frame.pose_time_offset_s * 1000, 2),
                "surfacesSeen": len(visible),
                "intrinsicsStream": intrinsics.stream,
            },
        ))

    uncovered = sorted(set(targets) - covered)
    diagnostics = {
        "available": True,
        "framesConsidered": len(frames),
        "framesWithVisibleGeometry": len(scored),
        "viewsSelected": len(views),
        "minVisibilityThreshold": min_visibility,
        "surfacesCovered": sorted(covered),
        "surfacesWithoutEvidence": uncovered,
        "selectionRule": (
            "Greedy set cover over measured surface visibility, capped at "
            f"{max_views} views, then topped up to at least {min_views} by choosing "
            f"views furthest in time from those already selected."),
        "occlusionNote": (
            "Visibility is geometric only. A surface hidden behind furniture still "
            "counts as visible, because judging occlusion is exactly what the "
            "verifier is being asked to do."),
        "privacyNote": (
            "These are public ARKitScenes development frames. A final private "
            "capture requires operator approval before any frame is sent to a "
            "model, and that gate lives in the AI configuration."),
    }
    if uncovered:
        diagnostics["uncoveredReason"] = (
            "These surfaces are inferred closure or were never seen well enough by a "
            "posed RGB frame to meet the visibility threshold. They are listed so AI "
            "review cannot be asked about geometry it has no evidence for.")
    return views, diagnostics
