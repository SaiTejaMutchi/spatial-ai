"""Room envelope: choose the walls that bound the room, then close the polygon.

Plane fitting deliberately returns more wall candidates than a room has walls, because
real rooms contain real planar surfaces — wardrobe fronts, window reveals,
partitions — that sit ten or twenty centimetres off the wall behind them. This
module decides which of those candidates actually *bound* the space.

The test is evidential rather than cosmetic: a boundary wall has essentially
the whole fitted floor on its interior side, while a surface standing inside
the room cuts a visible part of the floor away. Survivors are grouped by
inward direction, one representative per side is kept on evidence strength,
and the room is the intersection of their interior half-planes.

That intersection is convex, which is a real simplification and is recorded as
one. Where the half-planes do not close the room, the polygon is completed
against the floor's own observed bounds and those edges are marked `inferred`
rather than dressed up as observed geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import GeometryConfig
from .planes import Plane, PlaneSet

CONFIG_PARAMETERS = (
    "min_floor_interior_fraction", "envelope_boundary_tolerance_m",
    "envelope_min_edge_length_m", "envelope_parallel_merge_angle_deg",
)


@dataclass
class EnvelopeEdge:
    """One side of the room footprint, in plan coordinates."""

    start: np.ndarray            # (x, z)
    end: np.ndarray              # (x, z)
    wall: Plane | None           # None when the edge is inferred closure
    observation_state: str

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


@dataclass
class RoomEnvelope:
    footprint: np.ndarray                    # (K, 2) ordered plan polygon
    edges: list[EnvelopeEdge]
    floor: Plane | None
    ceiling: Plane | None
    selected_walls: list[Plane]
    excluded_walls: list[dict]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    config_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def area_m2(self) -> float:
        return polygon_area(self.footprint)

    @property
    def height_m(self) -> float | None:
        """Distance between the fitted floor and ceiling planes.

        Taken from the plane offsets rather than from the difference of the two
        point centroids. The centroid is a proxy that happens to sit close to
        the surface; the offset is the surface. On the development fixtures the
        two differ by 2 mm, so this is a correctness choice about which model
        is right, not a change that moves any published number materially.
        """
        if self.floor is None or self.ceiling is None:
            return None
        up = np.array([0.0, 1.0, 0.0])
        floor_h = float(self.floor.offset / (self.floor.normal @ up))
        ceiling_h = float(self.ceiling.offset / (self.ceiling.normal @ up))
        return ceiling_h - floor_h


def polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area, sign-independent."""
    if len(polygon) < 3:
        return 0.0
    x, z = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) / 2.0)


def minimum_area_rectangle(polygon: np.ndarray) -> tuple[float, float, float]:
    """Length, width, and orientation of the tightest enclosing rectangle.

    Room "length" and "width" have to mean something for a room that is not
    axis-aligned, so they are taken from the minimum-area enclosing rectangle
    via rotating calipers over the hull edges. Reporting the bounding box in
    canonical X and Z instead would inflate both numbers for any rotated room.
    """
    if len(polygon) < 3:
        return 0.0, 0.0, 0.0
    hull = _convex_hull(polygon)
    best = None
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        norm = float(np.linalg.norm(edge))
        if norm < 1e-12:
            continue
        axis = edge / norm
        perpendicular = np.array([-axis[1], axis[0]])
        projected_u = hull @ axis
        projected_v = hull @ perpendicular
        extent_u = float(projected_u.max() - projected_u.min())
        extent_v = float(projected_v.max() - projected_v.min())
        area = extent_u * extent_v
        if best is None or area < best[0]:
            best = (area, extent_u, extent_v, float(np.arctan2(axis[1], axis[0])))
    if best is None:
        return 0.0, 0.0, 0.0
    _, extent_u, extent_v, angle = best
    return max(extent_u, extent_v), min(extent_u, extent_v), angle


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone chain hull; deterministic and dependency free."""
    ordered = np.unique(np.round(points, 9), axis=0)
    if len(ordered) <= 2:
        return ordered
    ordered = ordered[np.lexsort((ordered[:, 1], ordered[:, 0]))]

    def half(sequence):
        chain: list[np.ndarray] = []
        for point in sequence:
            while len(chain) >= 2:
                a, b = chain[-2], chain[-1]
                if np.cross(b - a, point - a) <= 0:
                    chain.pop()
                else:
                    break
            chain.append(point)
        return chain

    lower = half(ordered)
    upper = half(ordered[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _plan(normal: np.ndarray) -> np.ndarray:
    """A level wall normal as a 2-vector in plan coordinates."""
    vector = np.array([normal[0], normal[2]], dtype=np.float64)
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-12 else vector


def select_boundary_walls(
    planes: PlaneSet, floor_points: np.ndarray, config: GeometryConfig
) -> tuple[list[Plane], list[dict]]:
    min_fraction = config.get("min_floor_interior_fraction")
    tolerance = config.get("envelope_boundary_tolerance_m")
    merge_angle = config.get("envelope_parallel_merge_angle_deg")

    kept: list[Plane] = []
    excluded: list[dict] = []

    scored: list[tuple[float, float, Plane]] = []
    for wall in planes.walls:
        inside = wall.signed_distance(floor_points) >= -tolerance
        fraction = float(inside.mean()) if len(floor_points) else 0.0
        if fraction < min_fraction:
            excluded.append({
                "wallId": wall.plane_id,
                "reason": f"only {fraction * 100:.1f}% of floor points lie inside this "
                          f"plane, below the {min_fraction * 100:.0f}% required of a "
                          f"boundary wall; it stands within the room rather than "
                          f"bounding it",
                "floorInteriorFraction": round(fraction, 4),
                "inlierCount": wall.inlier_count,
            })
            continue
        scored.append((wall.inlier_count * wall.coverage_fraction, fraction, wall))

    scored.sort(key=lambda item: -item[0])
    for strength, fraction, wall in scored:
        direction = _plan(wall.normal)
        duplicate = None
        for existing in kept:
            angle = float(np.degrees(np.arccos(np.clip(
                float(direction @ _plan(existing.normal)), -1.0, 1.0))))
            if angle <= merge_angle:
                duplicate = existing
                break
        if duplicate is not None:
            excluded.append({
                "wallId": wall.plane_id,
                "reason": f"describes the same side of the room as "
                          f"{duplicate.plane_id}, which carries stronger evidence "
                          f"(support x coverage)",
                "floorInteriorFraction": round(fraction, 4),
                "inlierCount": wall.inlier_count,
            })
            continue
        kept.append(wall)
    return kept, excluded


def _clip_polygon(polygon: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Sutherland-Hodgman clip against the half-plane normal . p >= offset."""
    if len(polygon) == 0:
        return polygon
    result: list[np.ndarray] = []
    for index in range(len(polygon)):
        current = polygon[index]
        following = polygon[(index + 1) % len(polygon)]
        d_current = float(normal @ current) - offset
        d_following = float(normal @ following) - offset
        if d_current >= 0:
            result.append(current)
        if (d_current >= 0) != (d_following >= 0):
            span = d_current - d_following
            if abs(span) > 1e-12:
                result.append(current + (following - current) * (d_current / span))
    return np.array(result) if result else np.zeros((0, 2))


def build_envelope(
    planes: PlaneSet,
    floor_points: np.ndarray,
    config: GeometryConfig,
) -> RoomEnvelope:
    tolerance = config.get("envelope_boundary_tolerance_m")
    min_edge = config.get("envelope_min_edge_length_m")

    floor_plan = floor_points[:, [0, 2]] if len(floor_points) else np.zeros((0, 2))
    selected, excluded = select_boundary_walls(planes, floor_points, config)

    if len(floor_plan) == 0:
        raise ValueError("the floor has no supporting points; no envelope can be built")

    # Start from the floor's own observed extent, so that where walls fail to
    # close the room the closure is at least anchored in observation.
    lower = floor_plan.min(axis=0) - tolerance
    upper = floor_plan.max(axis=0) + tolerance
    polygon = np.array([[lower[0], lower[1]], [upper[0], lower[1]],
                        [upper[0], upper[1]], [lower[0], upper[1]]])
    observed_bounds = polygon.copy()

    for wall in selected:
        clipped = _clip_polygon(polygon, _plan(wall.normal), wall.offset)
        if len(clipped) >= 3 and polygon_area(clipped) > 0:
            polygon = clipped

    polygon = _drop_short_edges(polygon, min_edge)
    if len(polygon) < 3:
        raise ValueError("wall clipping collapsed the footprint; no envelope survives")

    edges = _attribute_edges(polygon, selected, tolerance)

    observed_length = sum(e.length for e in edges if e.observation_state == "directly_observed")
    total_length = sum(e.length for e in edges)

    diagnostics = {
        "wallCandidates": len(planes.walls),
        "wallsSelected": len(selected),
        "wallsExcluded": len(excluded),
        "footprintVertices": int(len(polygon)),
        "observedPerimeterFraction": round(observed_length / total_length, 4)
                                     if total_length > 0 else 0.0,
        "inferredEdgeCount": sum(1 for e in edges if e.observation_state != "directly_observed"),
        "floorObservedBounds": {
            "min": [round(float(v), 4) for v in observed_bounds.min(axis=0)],
            "max": [round(float(v), 4) for v in observed_bounds.max(axis=0)],
        },
        "convexSimplification": (
            "The footprint is the intersection of the selected walls' interior "
            "half-planes and is therefore convex. A non-convex room would be "
            "reported as its convex envelope; this is a stated simplification of "
            "the POC, not a measurement of the room."),
    }

    return RoomEnvelope(
        footprint=polygon,
        edges=edges,
        floor=planes.floor,
        ceiling=planes.ceiling,
        selected_walls=selected,
        excluded_walls=excluded,
        diagnostics=diagnostics,
        config_provenance=config.provenance(*CONFIG_PARAMETERS),
    )


def _drop_short_edges(polygon: np.ndarray, min_edge: float) -> np.ndarray:
    keep: list[np.ndarray] = []
    for index in range(len(polygon)):
        following = polygon[(index + 1) % len(polygon)]
        if float(np.linalg.norm(following - polygon[index])) >= min_edge or not keep:
            keep.append(polygon[index])
    return np.array(keep)


def _attribute_edges(
    polygon: np.ndarray, walls: list[Plane], tolerance: float
) -> list[EnvelopeEdge]:
    """Attribute each polygon edge to the wall that produced it, if any."""
    edges: list[EnvelopeEdge] = []
    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        midpoint = (start + end) / 2.0
        owner = None
        best_gap = tolerance
        for wall in walls:
            direction = _plan(wall.normal)
            gap = abs(float(direction @ midpoint) - wall.offset)
            if gap <= best_gap:
                owner, best_gap = wall, gap
        edges.append(EnvelopeEdge(
            start=start, end=end, wall=owner,
            observation_state="directly_observed" if owner is not None else "inferred",
        ))
    return edges
