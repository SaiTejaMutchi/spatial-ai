"""Opening candidates, recorded honestly rather than resolved optimistically.

The plan is explicit about what this task may not do: *"On raw evidence,
require wall support around a candidate gap plus RGB/AI review; missing points
alone are insufficient"*, and *"Do not infer an opening from missing mesh
alone."*

So this module finds gaps and then refuses to promote them on its own. A
candidate must be an almost-empty region of a wall, wide and tall enough to be
architecture, and surrounded by cells that do carry wall support — a hole in
the middle of a wall, not the ragged edge of the observed region. Candidates
that pass are additionally tested for *see-through* evidence: a genuine opening
has depth returns landing beyond the wall plane, whereas an occluded patch has
returns in front of it.

Even see-through candidates stay `unresolved`. The corroboration the plan
requires is RGB or AI review, ARKitScenes ships no structured opening
semantics, and the Spatial AI Verifier is `not_run` without operator model
approval. Each record therefore names exactly what is missing, and carries its
measured extent in provenance rather than in `dimensions`, so nothing in the
model reads as a measured opening that was never confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import GeometryConfig
from .planes import Plane

CONFIG_PARAMETERS = (
    "opening_grid_cell_m", "opening_min_width_m", "opening_min_height_m",
    "opening_max_width_m", "opening_max_interior_occupancy",
    "opening_border_support_fraction", "opening_seethrough_margin_m",
    "plane_inlier_distance_m",
)

UP = np.array([0.0, 1.0, 0.0])


@dataclass
class OpeningCandidate:
    candidate_id: str
    surface_id: str
    width_m: float
    height_m: float
    sill_height_m: float
    centre: np.ndarray
    interior_occupancy: float
    border_support: float
    seethrough_points: int
    front_points: int
    world_centre: np.ndarray | None = None
    world_corners: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpeningResult:
    candidates: list[OpeningCandidate]
    records: list[dict]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    config_provenance: dict[str, Any] = field(default_factory=dict)


def _wall_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Along-wall and vertical axes for a level wall normal."""
    along = np.cross(UP, normal)
    along /= np.linalg.norm(along)
    return along, UP.copy()


def _best_empty_rectangle(
    occupied: np.ndarray,
    min_cols: int,
    max_cols: int,
    min_rows: int,
) -> tuple[int, int, int, int] | None:
    """Largest all-empty rectangle whose size falls inside the opening bounds.

    Taking the globally largest empty rectangle finds the wrong thing: an
    unobserved strip running the whole width of a wall beats a doorway on area
    every time. So the search is constrained to rectangles that could actually
    be an opening, and a run wider than the bound is clipped to it — any
    contiguous part of an empty run is itself empty.

    Standard maximal-rectangle-under-a-histogram sweep, O(rows * cols), exact
    and deterministic; an approximate search would make the reported extent
    depend on iteration order, which a frozen configuration cannot tolerate.
    """
    rows, cols = occupied.shape
    heights = np.zeros(cols, dtype=np.int64)
    best: tuple[int, int, int, int] | None = None
    best_area = 0

    for row in range(rows):
        heights = np.where(occupied[row], 0, heights + 1)
        stack: list[int] = []
        for col in range(cols + 1):
            current = int(heights[col]) if col < cols else 0
            while stack and int(heights[stack[-1]]) > current:
                height = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                width = col - left
                if height < min_rows or width < min_cols:
                    continue
                clipped = min(width, max_cols)
                area = height * clipped
                if area > best_area:
                    best_area = area
                    # Centre the clipped span inside the empty run, so a wide
                    # gap reports its middle rather than an arbitrary end.
                    offset = left + (width - clipped) // 2
                    best = (row - height + 1, offset, row, offset + clipped - 1)
            stack.append(col)
    return best


def detect_openings(
    surfaces: list[dict],
    planes_by_id: dict[str, Plane],
    points: np.ndarray,
    config: GeometryConfig,
) -> OpeningResult:
    cell = config.get("opening_grid_cell_m")
    min_width = config.get("opening_min_width_m")
    min_height = config.get("opening_min_height_m")
    max_width = config.get("opening_max_width_m")
    max_occupancy = config.get("opening_max_interior_occupancy")
    min_border = config.get("opening_border_support_fraction")
    seethrough_margin = config.get("opening_seethrough_margin_m")
    inlier_distance = config.get("plane_inlier_distance_m")

    candidates: list[OpeningCandidate] = []
    examined = 0
    rejected: list[dict] = []

    for surface in surfaces:
        if surface["type"] != "wall" or surface["observationState"] != "directly_observed":
            continue
        plane = planes_by_id.get(surface["provenance"].get("sourcePlaneId"))
        if plane is None:
            continue
        examined += 1

        normal = plane.normal
        along, vertical = _wall_axes(normal)
        distance = points @ normal - plane.offset
        on_wall = np.abs(distance) <= inlier_distance
        wall_points = points[on_wall]
        if len(wall_points) < 50:
            continue

        u = wall_points @ along
        v = wall_points @ vertical
        u_min, u_max = float(u.min()), float(u.max())
        v_min, v_max = float(v.min()), float(v.max())
        cols = max(int(np.ceil((u_max - u_min) / cell)), 1)
        rows = max(int(np.ceil((v_max - v_min) / cell)), 1)
        if cols < 3 or rows < 3:
            continue

        grid = np.zeros((rows, cols), dtype=bool)
        gi = np.clip(((v - v_min) / cell).astype(np.int64), 0, rows - 1)
        gj = np.clip(((u - u_min) / cell).astype(np.int64), 0, cols - 1)
        grid[gi, gj] = True

        found = _best_empty_rectangle(
            grid,
            min_cols=max(int(np.ceil(min_width / cell)), 1),
            max_cols=max(int(np.floor(max_width / cell)), 1),
            min_rows=max(int(np.ceil(min_height / cell)), 1),
        )
        if found is None:
            rejected.append({
                "surfaceId": surface["id"],
                "reason": f"no empty region on this wall is at least "
                          f"{min_width} m by {min_height} m, so nothing on it could "
                          f"be an opening"})
            continue
        r0, c0, r1, c1 = found
        width = (c1 - c0 + 1) * cell
        height = (r1 - r0 + 1) * cell

        interior = grid[r0:r1 + 1, c0:c1 + 1]
        occupancy = float(interior.mean())
        if occupancy > max_occupancy:
            continue

        # An opening must have wall to its left, to its right, and above it.
        # The strip *below* is deliberately not counted: beneath a doorway there
        # is floor, not wall, and the floor band was excluded from these points
        # back during plane fitting. Including it would penalise every door.
        strips: list[np.ndarray] = []
        if c0 - 1 >= 0:
            strips.append(grid[r0:r1 + 1, c0 - 1])
        if c1 + 1 < cols:
            strips.append(grid[r0:r1 + 1, c1 + 1])
        if r1 + 1 < rows:
            strips.append(grid[r1 + 1, c0:c1 + 1])
        surround = np.concatenate(strips) if strips else np.zeros(0, dtype=bool)
        border_support = float(surround.mean()) if surround.size else 0.0
        if border_support < min_border:
            rejected.append({
                "surfaceId": surface["id"],
                "reason": f"only {border_support * 100:.0f}% of the wall cells to the "
                          f"left, right and above the gap carry support, below the "
                          f"{min_border * 100:.0f}% required; this is the edge of the "
                          f"observed region rather than a hole through the wall",
                "gapWidth_m": round(width, 3), "gapHeight_m": round(height, 3),
                "borderSupportFraction": round(border_support, 4)})
            continue

        # See-through test: a real opening has returns beyond the wall plane.
        centre_u = u_min + (c0 + c1 + 1) / 2 * cell
        centre_v = v_min + (r0 + r1 + 1) / 2 * cell
        projected_u = points @ along
        projected_v = points @ vertical
        within = ((np.abs(projected_u - centre_u) <= width / 2)
                  & (np.abs(projected_v - centre_v) <= height / 2))
        through = int(((points @ normal - plane.offset) < -seethrough_margin)[within].sum())
        front = int(((points @ normal - plane.offset) > seethrough_margin)[within].sum())

        world_centre = (centre_u * along) + (centre_v * vertical) + (plane.offset * normal)
        half_w, half_h = width / 2.0, height / 2.0
        world_corners = np.stack([
            world_centre + (-half_w) * along + (-half_h) * vertical,
            world_centre + (half_w) * along + (-half_h) * vertical,
            world_centre + (half_w) * along + (half_h) * vertical,
            world_centre + (-half_w) * along + (half_h) * vertical,
        ])

        candidates.append(OpeningCandidate(
            candidate_id=f"opening-{len(candidates) + 1:03d}",
            surface_id=surface["id"],
            width_m=width,
            height_m=height,
            sill_height_m=centre_v - height / 2,
            centre=np.array([centre_u, centre_v]),
            interior_occupancy=occupancy,
            border_support=border_support,
            seethrough_points=through,
            front_points=front,
            world_centre=world_centre,
            world_corners=world_corners,
            diagnostics={
                "gridCell_m": cell,
                "gridRows": rows, "gridCols": cols,
                "interiorOccupancy": round(occupancy, 4),
                "borderSupportFraction": round(border_support, 4),
                "pointsBeyondPlane": through,
                "pointsInFrontOfPlane": front,
                "seeThroughEvidence": through > front,
            },
        ))

    records = [_record(c, surfaces) for c in candidates]
    if not records:
        records = [{
            "id": "opening-unresolved-001",
            "surfaceId": None,
            "type": "unresolved",
            "dimensions": None,
            "observationState": "unresolved",
            "confidence": None,      # filled in by the caller's rules
            "provenance": {
                "geometrySource": "geometry_pipeline",
                "algorithm": "wall_occupancy_gap_search_v0.1",
                "reason": (
                    f"No opening was resolved. {examined} directly observed wall "
                    f"surfaces were searched for gaps that are wide and tall enough "
                    f"to be architecture, nearly empty of wall returns, and surrounded "
                    f"by supported wall. Openings are recorded as unresolved rather "
                    f"than estimated."),
                "wallsExamined": examined,
                "rejectedCandidates": rejected,
            },
        }]

    diagnostics = {
        "wallsExamined": examined,
        "candidatesFound": len(candidates),
        "candidatesRejected": len(rejected),
        "rejected": rejected,
        "resolutionPolicy": (
            "Geometric gap evidence alone cannot resolve an opening. The plan "
            "requires wall support around the gap plus RGB or AI corroboration; "
            "ARKitScenes publishes no structured opening semantics and the Spatial "
            "AI Verifier is not_run without operator model approval, so every "
            "candidate remains unresolved and carries no dimensions."),
    }
    return OpeningResult(
        candidates=candidates,
        records=records,
        diagnostics=diagnostics,
        config_provenance=config.provenance(*CONFIG_PARAMETERS),
    )


def _record(candidate: OpeningCandidate, surfaces: list[dict]) -> dict:
    return {
        "id": candidate.candidate_id,
        "surfaceId": candidate.surface_id,
        "type": "unresolved",
        "dimensions": None,
        "observationState": "unresolved",
        "confidence": None,
        "provenance": {
            "geometrySource": "geometry_pipeline",
            "algorithm": "wall_occupancy_gap_search_v0.1",
            "semanticHypothesisSource": None,
            "reason": (
                "A gap in this wall's observed surface meets the geometric tests for "
                "an opening, but geometry alone cannot tell an opening from an "
                "occlusion. RGB or AI corroboration is required and is unavailable: "
                "the source publishes no structured opening semantics and the Spatial "
                "AI Verifier has not run. The measured extent below is a diagnostic, "
                "not a dimension."),
            "candidateExtent": {
                "width_m": round(candidate.width_m, 4),
                "height_m": round(candidate.height_m, 4),
                "sillHeight_m": round(candidate.sill_height_m, 4),
                **({
                    "worldCentre_m": [round(float(v), 6) for v in candidate.world_centre],
                    "worldCorners_m": [
                        [round(float(v), 6) for v in corner]
                        for corner in candidate.world_corners
                    ],
                } if candidate.world_corners is not None
                and candidate.world_centre is not None else {}),
            },
            "evidence": candidate.diagnostics,
            "whatWouldResolveThis": (
                "An operator-approved multimodal review of RGB frames showing this "
                "wall region, or a structured RoomPlan opening hypothesis."),
        },
    }
