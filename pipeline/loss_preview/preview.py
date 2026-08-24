"""Experimental P&C grounding bridge — the contract, not damage detection.

One narrow claim is being demonstrated: a visible observation can be attached to
a *named* surface, and the metric quantity that follows is calculated by
geometry from a registered region — never stated by a model.

The proposal used in development is an unmistakably labelled synthetic fixture.
It has to be: the Spatial AI Verifier is `not_run` without an approved model, so
no real proposal exists, and inventing one that looked real would be the exact
failure this contract is meant to prevent.

What is *not* synthetic is the quantity. The fixture supplies a normalized image
region on a real evidence frame; that region's corners are cast from the real
camera pose through the real intrinsics onto the real fitted surface plane, and
the area is measured there. If any of that fails — the surface has no plane, the
ray runs parallel to it, the region lands behind the camera — registration is
recorded `unresolved` and **no quantity is emitted at all**.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEVELOPMENT_LABEL = "DEVELOPMENT LOSS FIXTURE"

# A fixed region on the evidence frame. Synthetic, and labelled as such
# everywhere it appears; only the geometry it drives is real.
FIXTURE_REGION = {
    "x0": 0.30, "y0": 0.45, "x1": 0.62, "y1": 0.80,
    "note": "Normalized image coordinates. Invented for the fixture.",
}


def _plane_from_surface(surface: dict) -> tuple[np.ndarray, float] | None:
    plane = surface.get("plane") or {}
    normal = np.array(plane.get("normal", [0.0, 0.0, 0.0]), dtype=np.float64)
    if float(np.linalg.norm(normal)) < 1e-9:
        return None
    return normal, float(plane.get("offset_m", 0.0))


def register_region_to_surface(
    region: dict,
    surface: dict,
    camera_to_world: np.ndarray,
    intrinsics: dict,
) -> dict:
    """Cast a normalized image rectangle onto a surface plane and measure it.

    Returns a registration record. On any failure it returns `unresolved` with
    the reason and no area, because a quantity that cannot be grounded is worse
    than no quantity.
    """
    plane = _plane_from_surface(surface)
    if plane is None:
        return {"status": "unresolved",
                "reason": f"surface {surface['id']} carries no fitted plane, so an "
                          f"image region cannot be projected onto it"}
    normal, offset = plane

    width, height = intrinsics["width"], intrinsics["height"]
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    corners_px = [
        (region["x0"] * width, region["y0"] * height),
        (region["x1"] * width, region["y0"] * height),
        (region["x1"] * width, region["y1"] * height),
        (region["x0"] * width, region["y1"] * height),
    ]

    pose = np.asarray(camera_to_world, dtype=np.float64)
    origin = pose[:3, 3]
    rotation = pose[:3, :3]

    denominator_floor = 1e-6
    world_corners = []
    for u, v in corners_px:
        direction = rotation @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        denominator = float(normal @ direction)
        if abs(denominator) < denominator_floor:
            return {"status": "unresolved",
                    "reason": "a region ray runs parallel to the surface plane, so "
                              "it never meets it"}
        distance = (offset - float(normal @ origin)) / denominator
        if distance <= 0:
            return {"status": "unresolved",
                    "reason": "the region projects behind the camera rather than onto "
                              "the surface"}
        world_corners.append(origin + direction * distance)

    points = np.array(world_corners)
    # Area of the planar quadrilateral, from the cross products of its triangles.
    area = 0.5 * (
        float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))
        + float(np.linalg.norm(np.cross(points[2] - points[0], points[3] - points[0])))
    )
    extents = points.max(axis=0) - points.min(axis=0)

    return {
        "status": "registered",
        "method": "pinhole_ray_cast_to_fitted_surface_plane_v0.1",
        "surfaceId": surface["id"],
        "worldCorners_m": [[round(float(v), 6) for v in c] for c in world_corners],
        "affectedArea_m2": round(area, 6),
        "verticalExtent_m": round(float(extents[1]), 6),
        "producer": "geometry_pipeline",
    }


def build_loss_preview(model: dict, output_dir: Path) -> dict:
    """Emit `loss_preview.json` for the development fixture, or say why not."""
    output_dir = Path(output_dir)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    base: dict[str, Any] = {
        "label": DEVELOPMENT_LABEL,
        "isRealDamageEvidence": False,
        "schemaVersion": "0.1",
        "generatedAt": generated,
        "scanId": model["scan"]["id"],
        "statement": (
            "This is a synthetic fixture proving one contract: a named surface, "
            "attached evidence, an AI-shaped proposal, registration, and a metric "
            "quantity that only geometry may produce. It is not damage detection, "
            "not an accuracy claim, and not a repair estimate."),
        "boundary": {
            "aiMayPropose": ["damage class", "surfaceId association",
                             "an image region", "a request for human review"],
            "aiMayNotProduce": ["area", "linear extent", "restoration quantity",
                                "cost", "coverage", "reserve", "settlement"],
            "quantityProducer": "geometry_pipeline",
        },
    }

    # A proposal needs a surface that was actually observed and a frame that
    # actually sees it. Without both, the contract has nothing to demonstrate.
    evidence = model.get("evidence", [])
    candidates = [
        (view, surface_id)
        for view in evidence
        for surface_id in view["visibleSurfaceIds"]
        if any(s["id"] == surface_id and s["type"] == "wall"
               and s["observationState"] == "directly_observed"
               for s in model["surfaces"])
    ]
    if not candidates:
        base.update({
            "status": "not_applicable",
            "statusReason": ("No directly observed wall has a registered evidence "
                             "view, so there is no valid surface-plus-evidence pair "
                             "to demonstrate the contract against."),
            "proposals": [],
        })
        (output_dir / "loss_preview.json").write_text(json.dumps(base, indent=2) + "\n")
        return base

    view, surface_id = candidates[0]
    surface = next(s for s in model["surfaces"] if s["id"] == surface_id)

    intrinsics_source = model["provenance"].get("evidenceSelection", {})
    rgb = model["provenance"].get("capture", {})
    # Intrinsics come from the capture the evidence was selected against; the
    # evidence manifest records which stream was used.
    intrinsics = _intrinsics_for(model, view)

    # The canonical-frame pose, because the surface plane is canonical. Using
    # the source-frame pose would cast the region into a different space.
    pose = np.array(view.get("cameraToWorldCanonical") or view["cameraToWorld"],
                    dtype=np.float64)
    registration = register_region_to_surface(
        FIXTURE_REGION, surface, pose, intrinsics)

    proposal: dict[str, Any] = {
        "damageId": "damage-fixture-001",
        "label": DEVELOPMENT_LABEL,
        "surfaceId": surface_id,
        "status": "proposed_experimental",
        "damageType": "water",
        "severity": "unknown_or_experimental",
        "evidenceFrameIds": [view["id"]],
        "affectedRegion": {
            "method": "registered_visual_region",
            "normalizedImageRegion": FIXTURE_REGION,
        },
        "registration": registration,
        "reviewStatus": "human_review_required",
        "provenance": {
            "aiProducer": "development_fixture_not_a_model",
            "aiProducerNote": (
                "The Spatial AI Verifier is not_run because no model is approved, so "
                "this proposal was written by the fixture, not inferred from the "
                "image. Only the geometry below is real."),
            "quantityProducer": "geometry_pipeline",
            "promptVersion": "loss_proposal_v0.1",
        },
    }

    if registration["status"] == "registered":
        proposal["quantity"] = {
            "affectedArea_m2": registration["affectedArea_m2"],
            "verticalExtent_m": registration["verticalExtent_m"],
            "producer": "geometry_pipeline",
            "method": registration["method"],
            "note": ("Measured by casting the region onto the fitted surface plane "
                     "through the real camera pose and intrinsics. The region is "
                     "synthetic; this measurement of it is not."),
        }
        base["status"] = "development_fixture_registered"
        base["statusReason"] = (
            f"The fixture region registered to {surface_id} and geometry measured "
            f"{registration['affectedArea_m2']:.3f} m² on that surface.")
    else:
        base["status"] = "development_fixture_unresolved"
        base["statusReason"] = (
            f"Registration was unresolved: {registration['reason']}. No quantity is "
            f"emitted, which is the required behaviour.")

    base["proposals"] = [proposal]
    from pipeline.ai.visible_condition import attach_recorded_proposal
    attach_recorded_proposal(base, model)
    (output_dir / "loss_preview.json").write_text(json.dumps(base, indent=2) + "\n")
    model_proposal = next(
        (item for item in base.get("proposals") or []
         if item.get("affectedRegion", {}).get("producer") == "ai_visible_condition"),
        None)
    if model_proposal:
        (output_dir / "visible_condition.json").write_text(json.dumps({
            "label": "DEVELOPMENT CONDITION FIXTURE",
            "isRealDamageEvidence": False,
            "proposal": model_proposal,
            "modelGeneratedGrounding": base.get("modelGeneratedGrounding"),
        }, indent=2) + "\n")
    return base


def _intrinsics_for(model: dict, view: dict) -> dict:
    """Intrinsics for an evidence frame, from the model's own record."""
    recorded = model["provenance"].get("evidenceIntrinsics")
    if recorded:
        return recorded
    # The evidence selector records which stream it used; the capture's
    # intrinsics travel with the normalized package rather than the model, so
    # the model carries a copy for exactly this purpose.
    stream = view.get("diagnostics", {}).get("intrinsicsStream", "rgb")
    fallback = model["provenance"].get("captureIntrinsics", {}).get(stream)
    if fallback:
        return fallback
    raise KeyError(
        "no intrinsics are recorded for the evidence frames; the loss preview "
        "cannot register a region without them")
