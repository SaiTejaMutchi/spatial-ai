"""Live-scan Opening Resolver: classify a crop of each geometry gap.

Scene-level recognition of a door in a video is not corroboration. Each
unresolved candidate is shown as a crop of its nominated wall region. Geometry
owns any promoted dimensions. The VLM cannot create, delete, or resize a gap.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..contracts.normalized_capture import NormalizedCapture
from ..evidence.select import surface_visibility
from ..geometry.confidence import load_confidence_rules
from ..geometry.planes import Plane
from ..geometry.points import PointCloud
from .opening_resolver import ResolutionResult, resolve_candidate
from .verifier import AIModelConfig, GroqVerifierClient, load_ai_config

MIN_CROP_PX = 16
CROP_PAD_FRACTION = 0.12


def _intrinsics(capture: NormalizedCapture | None, image_size: tuple[int, int]
                ) -> tuple[float, float, float, float] | None:
    if capture is None:
        return None
    rgb = next((item for item in capture.intrinsics if item.stream == "rgb"), None)
    if rgb is None:
        rgb = next((item for item in capture.intrinsics if item.stream == "depth"), None)
    if rgb is None:
        return None
    width, height = image_size
    scale_x = width / rgb.width if rgb.width else 1.0
    scale_y = height / rgb.height if rgb.height else 1.0
    return rgb.fx * scale_x, rgb.fy * scale_y, rgb.cx * scale_x, rgb.cy * scale_y


def project_corners(
    corners: np.ndarray,
    camera_to_world: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray | None:
    """Project wall-rectangle corners with the same convention as evidence selection."""
    pose = np.asarray(camera_to_world, dtype=np.float64)
    world_to_camera = np.linalg.inv(pose)
    camera = corners @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    if not np.all(camera[:, 2] > 1e-6):
        return None
    u = camera[:, 0] / camera[:, 2] * fx + cx
    v = camera[:, 1] / camera[:, 2] * fy + cy
    return np.stack([u, v], axis=1)


def crop_nominated_region(image: Image.Image, pixels: np.ndarray) -> tuple[bytes, dict]:
    width, height = image.size
    x0 = int(np.floor(pixels[:, 0].min()))
    y0 = int(np.floor(pixels[:, 1].min()))
    x1 = int(np.ceil(pixels[:, 0].max()))
    y1 = int(np.ceil(pixels[:, 1].max()))
    pad_x = max(8, int((x1 - x0) * CROP_PAD_FRACTION))
    pad_y = max(8, int((y1 - y0) * CROP_PAD_FRACTION))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    if (x1 - x0) < MIN_CROP_PX or (y1 - y0) < MIN_CROP_PX:
        raise ValueError("the nominated region does not land in this photograph")
    crop = image.crop((x0, y0, x1, y1))
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    region = {
        "x0": round(x0 / width, 4),
        "y0": round(y0 / height, 4),
        "x1": round(x1 / width, 4),
        "y1": round(y1 / height, 4),
        "pixelBox": [x0, y0, x1, y1],
    }
    return buffer.getvalue(), region


def _view_pose(view: dict) -> np.ndarray | None:
    pose = view.get("cameraToWorldCanonical") or view.get("cameraToWorld")
    if pose is None:
        return None
    return np.asarray(pose, dtype=np.float64)


def _best_registered_view(opening: dict, model: dict) -> dict | None:
    surface_id = opening.get("surfaceId")
    if not surface_id:
        return None
    scored = []
    for view in model.get("evidence") or []:
        if surface_id not in (view.get("visibleSurfaceIds") or []):
            continue
        visibility = (view.get("surfaceVisibility") or {}).get(surface_id, 0.0)
        scored.append((visibility, view))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _search_capture_frame(
    opening: dict,
    capture: NormalizedCapture,
    capture_root: Path,
    cloud: PointCloud,
    plane: Plane,
    config,
) -> dict | None:
    """If the three review stills miss this wall, find any posed frame that sees it."""
    from ..geometry.config import GeometryConfig

    surface_id = opening.get("surfaceId")
    if not surface_id:
        return None
    tolerance = config.get("plane_inlier_distance_m") if isinstance(config, GeometryConfig) \
        else 0.03
    points = cloud.points[np.abs(cloud.points @ plane.normal - plane.offset) <= tolerance]
    if len(points) == 0:
        return None
    rgb = next((item for item in capture.intrinsics if item.stream == "rgb"), None)
    if rgb is None:
        return None
    best = None
    for frame in capture.frames[::6]:
        if frame.rgb is None:
            continue
        pose = np.asarray(frame.camera_to_world, dtype=np.float64)
        world_to_camera = np.linalg.inv(pose)
        canonical_w2c = world_to_camera @ np.linalg.inv(cloud.frame.source_to_canonical)
        fraction = surface_visibility(
            points, canonical_w2c, rgb.fx, rgb.fy, rgb.cx, rgb.cy,
            rgb.width, rgb.height)
        if best is None or fraction > best[0]:
            best = (fraction, frame)
    if best is None or best[0] < 0.05:
        return None
    frame = best[1]
    return {
        "id": f"frame-{frame.index:06d}",
        "path": str(capture_root / frame.rgb),
        "cameraToWorldCanonical": (
            cloud.frame.source_to_canonical
            @ np.asarray(frame.camera_to_world, dtype=np.float64)
        ).tolist(),
        "surfaceVisibility": {surface_id: best[0]},
        "visibleSurfaceIds": [surface_id],
        "_absolutePath": True,
    }


def _insufficient_result(opening: dict, reason: str, code: str) -> ResolutionResult:
    return ResolutionResult(
        resolution={
            "candidateId": opening["id"],
            "surfaceId": opening.get("surfaceId"),
            "semanticClass": "insufficient_evidence",
            "evidenceStatus": "insufficient_evidence",
            "evidenceFrameIds": ["none"],
            "reason": reason,
        },
        promoted=None,
        diagnostics={"validationResult": code, "promoted": False},
    )


def _apply_resolution(
    opening: dict,
    result,
    model: dict,
    crop_name: str | None,
    region: dict | None,
    view_id: str | None,
) -> None:
    provenance = opening.setdefault("provenance", {})
    provenance["semanticHypothesisSource"] = "opening_resolver_v0.1"
    provenance["aiResolution"] = {
        "resolution": result.resolution,
        "promoted": bool(result.promoted),
        "cropPath": f"evidence/{crop_name}" if crop_name else None,
        "evidenceFrameId": view_id,
        "imageRegion": region,
        "diagnostics": {
            key: result.diagnostics.get(key)
            for key in ("validationResult", "promoted", "promotionBlocked",
                        "abstention", "grounding", "promptVersion")
        },
    }
    if result.promoted is None:
        provenance["whatWouldResolveThis"] = (
            "A crop of this nominated gap that supports door, window, or open "
            "passage. A door seen elsewhere in the room is not enough.")
        return

    promoted = result.promoted
    opening["type"] = promoted["type"]
    opening["observationState"] = "directly_observed"
    opening["dimensions"] = promoted["dimensions"]
    opening["producer"] = "geometry_pipeline"
    host = next((surface for surface in model["surfaces"]
                 if surface["id"] == opening["surfaceId"]), None)
    inputs = (host or {}).get("confidence", {}).get("inputs") or {}
    rules = load_confidence_rules()
    opening["confidence"] = rules.label(
        "directly_observed",
        inputs.get("rmsResidual_m"),
        inputs.get("coverageFraction"),
        inputs.get("contributingFrames"),
    )
    provenance["reason"] = (
        "AI classified a crop of this nominated gap as "
        f"{promoted['type']}. Dimensions are the geometry-owned candidate "
        "extent, not a model measurement.")
    provenance["whatWouldResolveThis"] = None


def resolve_scan_openings(
    model: dict,
    output_dir: Path,
    capture: NormalizedCapture | None = None,
    capture_root: Path | None = None,
    cloud: PointCloud | None = None,
    planes_by_id: dict[str, Plane] | None = None,
    config=None,
    client: GroqVerifierClient | None = None,
    ai_config: AIModelConfig | None = None,
) -> dict[str, Any]:
    """Classify each unresolved geometry gap from a nominated crop."""
    output_dir = Path(output_dir)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ai_config = ai_config or load_ai_config()
    report: dict[str, Any] = {
        "candidates": 0,
        "classified": 0,
        "promotedCount": 0,
        "insufficientCount": 0,
        "skipped": [],
        "results": [],
        "diagnostics": {
            "grounding": "nominated_crop_only",
            "note": ("A crop of the geometry gap is classified. The video is not "
                     "watched, and a door elsewhere in the room cannot promote "
                     "this candidate."),
        },
    }

    for opening in model.get("openings") or []:
        if opening.get("observationState") != "unresolved":
            continue
        extent = (opening.get("provenance") or {}).get("candidateExtent") or {}
        corners = extent.get("worldCorners_m")
        if not corners or opening.get("surfaceId") is None:
            report["skipped"].append({
                "openingId": opening.get("id"),
                "reason": "no nominated wall region is stored for this record",
            })
            continue
        report["candidates"] += 1

        view = _best_registered_view(opening, model)
        if view is None and capture is not None and capture_root is not None \
                and cloud is not None and planes_by_id is not None:
            surface = next((item for item in model["surfaces"]
                            if item["id"] == opening["surfaceId"]), None)
            plane_id = (surface or {}).get("provenance", {}).get("sourcePlaneId")
            plane = planes_by_id.get(plane_id) if plane_id else None
            if plane is not None:
                view = _search_capture_frame(
                    opening, capture, Path(capture_root), cloud, plane, config)

        if view is None:
            report["insufficientCount"] += 1
            report["skipped"].append({
                "openingId": opening["id"],
                "reason": "no registered photograph sees this wall",
            })
            _apply_resolution(
                opening,
                ResolutionResult(
                    resolution={
                        "candidateId": opening["id"],
                        "surfaceId": opening["surfaceId"],
                        "semanticClass": "insufficient_evidence",
                        "evidenceStatus": "insufficient_evidence",
                        "evidenceFrameIds": ["none"],
                        "reason": "no registered photograph sees this nominated gap",
                    },
                    promoted=None,
                    diagnostics={"validationResult": "no_view", "promoted": False},
                ),
                model, None, None, None)
            continue

        image_path = Path(view["path"]) if view.get("_absolutePath") \
            else output_dir / view["path"]
        if not image_path.is_file():
            reason = "the registered photograph is missing on disk"
            report["insufficientCount"] += 1
            report["skipped"].append({"openingId": opening["id"], "reason": reason})
            _apply_resolution(opening, _insufficient_result(opening, reason, "missing_image"),
                              model, None, None, view.get("id"))
            continue

        image = Image.open(image_path).convert("RGB")
        intrinsics = _intrinsics(capture, image.size)
        pose = _view_pose(view)
        if intrinsics is None or pose is None:
            reason = "camera pose or intrinsics are missing for this crop"
            report["insufficientCount"] += 1
            report["skipped"].append({"openingId": opening["id"], "reason": reason})
            _apply_resolution(opening, _insufficient_result(opening, reason, "no_camera"),
                              model, None, None, view.get("id"))
            continue
        fx, fy, cx, cy = intrinsics
        pixels = project_corners(np.asarray(corners, dtype=np.float64), pose, fx, fy, cx, cy)
        if pixels is None:
            reason = "the nominated gap projects behind this camera"
            report["insufficientCount"] += 1
            report["skipped"].append({"openingId": opening["id"], "reason": reason})
            _apply_resolution(opening, _insufficient_result(opening, reason, "behind_camera"),
                              model, None, None, view.get("id"))
            continue
        try:
            crop_bytes, region = crop_nominated_region(image, pixels)
        except ValueError as exc:
            reason = str(exc)
            report["insufficientCount"] += 1
            report["skipped"].append({"openingId": opening["id"], "reason": reason})
            _apply_resolution(opening, _insufficient_result(opening, reason, "crop_miss"),
                              model, None, None, view.get("id"))
            continue

        crop_name = f"{opening['id']}-crop.png"
        (evidence_dir / crop_name).write_bytes(crop_bytes)
        candidate = {
            "candidateId": opening["id"],
            "surfaceId": opening["surfaceId"],
            "geometry": {
                "width_m": extent.get("width_m"),
                "height_m": extent.get("height_m"),
                "sillHeight_m": extent.get("sillHeight_m"),
            },
            "imageRegion": region,
        }
        result = resolve_candidate(
            candidate, crop_bytes, config=ai_config, client=client,
            evidence_id=view["id"])
        _apply_resolution(opening, result, model, crop_name, region, view["id"])
        report["classified"] += 1
        if result.promoted:
            report["promotedCount"] += 1
        elif result.resolution.get("semanticClass") == "insufficient_evidence":
            report["insufficientCount"] += 1
        report["results"].append({
            "openingId": opening["id"],
            "surfaceId": opening["surfaceId"],
            "semanticClass": result.resolution.get("semanticClass"),
            "promoted": bool(result.promoted),
            "evidenceFrameId": view["id"],
            "crop": crop_name,
        })

    (output_dir / "opening_resolutions.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report
