"""Emit `spatial_model.json`: the canonical, insurance-facing system of record.

Everything downstream — the 2D plan, the 3D model, the benchmark, the AI
review, the UI — reads this document rather than re-deriving geometry. So the
rules here are strict:

* Geometry owns every metric value. `producer` is always `geometry_pipeline`.
* Nothing is asserted that the evidence does not carry. An unclosed side of the
  room becomes an `inferred` surface whose confidence is `unresolved`, not a
  wall with a plausible-looking length.
* Confidence is whatever the versioned rules return, including `unresolved`.
* `damage[]` and `scope[]` exist and stay empty. They are the attachment points
  for future Track B/C work, not a claim that either is implemented.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.normalized_capture import NormalizedCapture
from .confidence import ConfidenceRules
from .config import GeometryConfig
from .envelope import RoomEnvelope, minimum_area_rectangle, polygon_area
from .planes import Plane
from .points import PointCloud

SCHEMA_VERSION = "0.1"
GEOMETRY_SOURCE = "geometry_pipeline"


def _round_point(point: np.ndarray) -> list[float]:
    return [round(float(v), 6) for v in point]


@dataclass
class ModelBuildResult:
    model: dict
    diagnostics: dict[str, Any]


def _surface_from_plane(
    plane: Plane,
    surface_id: str,
    room_id: str,
    kind: str,
    width_m: float,
    height_m: float,
    rules: ConfidenceRules,
    config: GeometryConfig,
    observation_state: str = "directly_observed",
    extra_provenance: dict | None = None,
) -> dict:
    confidence = rules.label(
        observation_state=observation_state,
        rms_residual_m=plane.rms_residual_m,
        coverage_fraction=plane.coverage_fraction,
        contributing_frames=plane.contributing_frames,
        extra_inputs={"inlierCount": plane.inlier_count,
                      "maxResidual_m": round(plane.max_residual_m, 6)},
    )
    provenance = {
        "geometrySource": GEOMETRY_SOURCE,
        "algorithm": plane.algorithm,
        "geometryConfigId": config.config_id,
        "geometryConfigHash": config.sha256,
        "sourcePlaneId": plane.plane_id,
        "semanticHypothesisSource": None,
        "fitDiagnostics": plane.to_record()["support"],
        "planeDiagnostics": plane.diagnostics,
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    return {
        "id": surface_id,
        "roomId": room_id,
        "type": kind,
        "dimensions": {"width_m": round(float(width_m), 6),
                       "height_m": round(float(height_m), 6)},
        "plane": {"normal": [round(float(v), 9) for v in plane.normal],
                  "offset_m": round(float(plane.offset), 6)},
        "observationState": observation_state,
        "confidence": confidence,
        "provenance": provenance,
        "damage": [],
    }


def _measurement(
    measurement_id: str,
    kind: str,
    value: float | None,
    unit: str,
    entity_id: str | None,
    observation_state: str,
    confidence: dict,
) -> dict:
    return {
        "id": measurement_id,
        "type": kind,
        "entityId": entity_id,
        "value_m": round(float(value), 6) if value is not None else None,
        "unit": unit,
        "producer": GEOMETRY_SOURCE,
        "referenceAvailable": False,
        "observationState": observation_state,
        "confidence": confidence,
    }


def build_spatial_model(
    capture: NormalizedCapture,
    cloud: PointCloud,
    envelope: RoomEnvelope,
    config: GeometryConfig,
    rules: ConfidenceRules,
    scan_id: str | None = None,
    model_id: str | None = None,
    openings: list[dict] | None = None,
    opening_diagnostics: dict | None = None,
) -> ModelBuildResult:
    room_id = "room-001"
    scan_id = scan_id or capture.provenance.source_id
    model_id = model_id or f"model-{uuid.uuid5(uuid.NAMESPACE_URL, scan_id)}"

    height = envelope.height_m
    length, width, orientation = minimum_area_rectangle(envelope.footprint)
    area = polygon_area(envelope.footprint)

    surfaces: list[dict] = []
    measurements: list[dict] = []

    # -- walls, one surface per footprint edge -----------------------------
    wall_height = height if height is not None else 0.0
    for index, edge in enumerate(envelope.edges, start=1):
        surface_id = f"wall-{index:03d}"
        if edge.wall is not None:
            surfaces.append(_surface_from_plane(
                edge.wall, surface_id, room_id, "wall",
                width_m=edge.length,
                height_m=wall_height if wall_height > 0 else edge.wall.extent["verticalSpan_m"],
                rules=rules, config=config,
                observation_state="directly_observed",
                extra_provenance={
                    "footprintEdge": {"start": _round_point(edge.start),
                                      "end": _round_point(edge.end)},
                },
            ))
        else:
            # Closure the walls did not support. It has a length because the
            # floor was observed there, but no plane stands behind it.
            confidence = rules.label("inferred", None, None, None)
            surfaces.append({
                "id": surface_id,
                "roomId": room_id,
                "type": "wall",
                "dimensions": {
                    "width_m": round(float(edge.length), 6),
                    "height_m": round(float(wall_height), 6) if wall_height > 0 else 0.0,
                },
                "plane": {"normal": [0.0, 0.0, 0.0], "offset_m": 0.0},
                "observationState": "inferred",
                "confidence": confidence,
                "provenance": {
                    "geometrySource": GEOMETRY_SOURCE,
                    "algorithm": "floor_extent_closure_v0.1",
                    "geometryConfigId": config.config_id,
                    "geometryConfigHash": config.sha256,
                    "sourcePlaneId": None,
                    "note": "No wall plane bounds the room on this side. The edge "
                            "follows the observed floor extent and is reported as "
                            "inferred closure, not as an observed wall.",
                    "footprintEdge": {"start": _round_point(edge.start),
                                      "end": _round_point(edge.end)},
                },
                "damage": [],
            })
        surface = surfaces[-1]
        measurements.append(_measurement(
            f"measurement-{surface_id}-length", "wall_length",
            surface["dimensions"]["width_m"], "m", surface_id,
            surface["observationState"], surface["confidence"]))

    # -- floor and ceiling --------------------------------------------------
    if envelope.floor is not None:
        surfaces.append(_surface_from_plane(
            envelope.floor, "floor-001", room_id, "floor",
            width_m=length or 1e-6, height_m=width or 1e-6,
            rules=rules, config=config))
    if envelope.ceiling is not None:
        surfaces.append(_surface_from_plane(
            envelope.ceiling, "ceiling-001", room_id, "ceiling",
            width_m=length or 1e-6, height_m=width or 1e-6,
            rules=rules, config=config))

    # -- room-level confidence and measurements ----------------------------
    observed_fraction = envelope.diagnostics["observedPerimeterFraction"]
    room_state = ("directly_observed" if envelope.diagnostics["inferredEdgeCount"] == 0
                  else "partially_observed")
    room_confidence = rules.label(
        observation_state=room_state,
        rms_residual_m=envelope.floor.rms_residual_m if envelope.floor else None,
        coverage_fraction=observed_fraction,
        contributing_frames=envelope.floor.contributing_frames if envelope.floor else None,
        extra_inputs={
            "observedPerimeterFraction": observed_fraction,
            "inferredEdgeCount": envelope.diagnostics["inferredEdgeCount"],
            "wallsSelected": envelope.diagnostics["wallsSelected"],
        },
    )

    if height is None:
        height_confidence = rules.label("unresolved", None, None, None)
        height_state = "unresolved"
    else:
        weaker = min(
            [envelope.floor, envelope.ceiling],
            key=lambda p: (p.coverage_fraction, -p.rms_residual_m))
        height_confidence = rules.label(
            "directly_observed", weaker.rms_residual_m, weaker.coverage_fraction,
            weaker.contributing_frames,
            extra_inputs={"floorHeight_m": round(float(envelope.floor.centroid[1]), 6),
                          "ceilingHeight_m": round(float(envelope.ceiling.centroid[1]), 6),
                          "limitingSurface": weaker.plane_id})
        height_state = "directly_observed"

    measurements.extend([
        _measurement("measurement-room-length", "room_length", length, "m", room_id,
                     room_state, room_confidence),
        _measurement("measurement-room-width", "room_width", width, "m", room_id,
                     room_state, room_confidence),
        _measurement("measurement-room-height", "room_height", height, "m", room_id,
                     height_state, height_confidence),
        _measurement("measurement-floor-area", "floor_area", area, "m2", room_id,
                     room_state, room_confidence),
    ])

    # Openings arrive from opening detection without a confidence label, because only the
    # versioned rules may assign one. Every unresolved opening resolves to
    # `unresolved` here rather than being handed a plausible label.
    opening_records = []
    for record in (openings or []):
        entry = dict(record)
        if entry.get("confidence") is None:
            entry["confidence"] = rules.label(
                entry.get("observationState", "unresolved"), None, None, None)
        opening_records.append(entry)

    model = {
        "schemaVersion": SCHEMA_VERSION,
        "modelId": model_id,
        "scan": {
            "id": scan_id,
            "capturedAt": None,
            "device": capture.provenance.device.get("platform", "unknown"),
            "units": "meters",
            "classification": capture.provenance.classification,
            "sourceType": capture.provenance.source_type,
            "connector": capture.provenance.connector,
        },
        "coordinateSystem": {
            "source": "canonical",
            "upAxis": "y",
            "planAxes": ["x", "z"],
            "handedness": "right",
            "frame": cloud.frame.provenance(),
        },
        "rooms": [{
            "id": room_id,
            "footprint": [_round_point(p) for p in envelope.footprint],
            "footprintEdgeStates": [e.observation_state for e in envelope.edges],
            "observationState": room_state,
            "confidence": room_confidence,
            "orientationRad": round(float(orientation), 6),
        }],
        "surfaces": surfaces,
        "openings": opening_records,
        "measurements": measurements,
        "evidence": [],
        "aiAssessments": [],
        "damage": [],
        "scope": [],
        "provenance": {
            "generatedUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "capture": {
                "contractVersion": capture.contract_version,
                "sourceType": capture.provenance.source_type,
                "sourceId": capture.provenance.source_id,
                "classification": capture.provenance.classification,
                "connector": capture.provenance.connector,
                "connectorVersion": capture.provenance.connector_version,
                "frameCount": len(capture.frames),
                "excludedFrameCount": len(capture.excluded_frames),
            },
            "captureIntrinsics": {
                intrinsics.stream: {
                    "width": intrinsics.width, "height": intrinsics.height,
                    "fx": intrinsics.fx, "fy": intrinsics.fy,
                    "cx": intrinsics.cx, "cy": intrinsics.cy,
                    "derivation": intrinsics.derivation,
                }
                for intrinsics in capture.intrinsics
            },
            "geometryConfigId": config.config_id,
            "geometryConfigHash": config.sha256,
            "confidenceRulesId": rules.rules_id,
            "confidenceRulesHash": rules.sha256,
            "pointCloud": cloud.diagnostics,
            "planeSelection": envelope.diagnostics,
            "excludedWalls": envelope.excluded_walls,
            "openingSearch": opening_diagnostics or {},
            "confidenceStatement": (
                "Confidence labels are evidence-quality heuristics, not calibrated "
                "probabilities or confidence intervals."),
            "trackBCStatement": (
                "damage[] and scope[] are intentionally empty attachment points. "
                "Damage detection and restoration scope are not implemented."),
        },
    }
    diagnostics = {
        "surfaceCount": len(surfaces),
        "openingCount": len(opening_records),
        "unresolvedOpenings": sum(1 for o in opening_records
                                  if o["observationState"] == "unresolved"),
        "measurementCount": len(measurements),
        "inferredSurfaces": sum(1 for s in surfaces
                                if s["observationState"] == "inferred"),
        "confidenceHistogram": _histogram(surfaces),
    }
    return ModelBuildResult(model=model, diagnostics=diagnostics)


def _histogram(surfaces: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for surface in surfaces:
        label = surface["confidence"]["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def write_model(model: dict, path: Path) -> str:
    """Write the model deterministically and return its hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(model, indent=2, sort_keys=False) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
