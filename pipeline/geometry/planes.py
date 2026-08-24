"""Structural plane extraction: floor, ceiling, and wall candidates.

The cloud arrives gravity-aligned from capture normalization, and that changes what "plane
fitting" has to mean here:

* A **horizontal** plane is then fully determined by one number, its height, so
  floor and ceiling come from peaks in a height histogram rather than from a
  randomised search. The result is exactly reproducible, which matters because
  the configuration behind it gets frozen and rerun on the final capture.
* A **vertical** plane is a line in the floor plan, so walls are found by
  voting over orientation and offset — a Hough transform on the plan
  projection. Also deterministic, and it yields support and coverage as a
  by-product of the vote instead of as an afterthought.

Every candidate that is examined and rejected is retained with the reason, so a
missing wall can be explained rather than merely noticed. No threshold appears
in this file; all of them come from `geometry_config_v0.1.json`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import GeometryConfig
from .points import PointCloud, voxel_downsample

CONFIG_PARAMETERS = (
    "plane_inlier_distance_m", "horizontal_normal_max_deviation_deg",
    "vertical_normal_max_deviation_deg", "height_histogram_bin_m",
    "min_horizontal_support_fraction", "min_room_height_m", "max_room_height_m",
    "min_structural_plan_coverage", "plane_trim_quantile", "plane_trim_iterations",
    "floor_ceiling_exclusion_margin_m", "wall_vote_voxel_size_m",
    "wall_angle_step_deg", "wall_offset_bin_m", "min_wall_height_m",
    "min_wall_length_m", "min_wall_coverage_fraction", "wall_merge_angle_deg",
    "wall_merge_distance_m", "plane_coverage_cell_m",
)

UP = np.array([0.0, 1.0, 0.0])

# Bounds on the search, not on the geometry: a room does not have hundreds of
# walls, and a peak ranked below these has already lost to stronger evidence.
MAX_CANDIDATE_PEAKS = 400
MAX_WALLS = 24
MIN_COARSE_SUPPORT = 30


@dataclass
class Plane:
    """A fitted structural plane and the evidence behind it."""

    plane_id: str
    kind: str                       # "floor" | "ceiling" | "wall"
    normal: np.ndarray              # unit, canonical frame
    offset: float                   # plane is {p : normal . p = offset}
    inlier_count: int
    rms_residual_m: float
    max_residual_m: float
    coverage_fraction: float
    contributing_frames: int
    extent: dict[str, float]
    centroid: np.ndarray
    algorithm: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.normal - self.offset

    def to_record(self) -> dict:
        return {
            "id": self.plane_id,
            "kind": self.kind,
            "normal": [round(float(v), 9) for v in self.normal],
            "offset_m": round(float(self.offset), 6),
            "centroid_m": [round(float(v), 6) for v in self.centroid],
            "algorithm": self.algorithm,
            "support": {
                "inlierCount": int(self.inlier_count),
                "contributingFrames": int(self.contributing_frames),
                "rmsResidual_m": round(float(self.rms_residual_m), 6),
                "maxResidual_m": round(float(self.max_residual_m), 6),
                "coverageFraction": round(float(self.coverage_fraction), 6),
            },
            "extent_m": {k: round(float(v), 6) for k, v in self.extent.items()},
            "diagnostics": self.diagnostics,
        }


@dataclass
class PlaneSet:
    floor: Plane | None
    ceiling: Plane | None
    walls: list[Plane]
    rejected: list[dict]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    config_provenance: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "floor": self.floor.to_record() if self.floor else None,
            "ceiling": self.ceiling.to_record() if self.ceiling else None,
            "walls": [w.to_record() for w in self.walls],
            "rejectedCandidates": self.rejected,
            "diagnostics": self.diagnostics,
            "configProvenance": self.config_provenance,
        }


# --------------------------------------------------------------------------
# shared fitting helpers
# --------------------------------------------------------------------------

def plan_cells(points: np.ndarray, cell_m: float) -> set:
    """Occupied cells of a plan-view grid; the footprint a surface covers."""
    return set(map(tuple, np.floor(points[:, [0, 2]] / cell_m).astype(np.int64)))


def trimmed_plane_fit(
    points: np.ndarray, normal: np.ndarray, offset: float,
    quantile: float, iterations: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Refit a plane after repeatedly discarding its worst-residual tail.

    Clutter that lies inside the inlier band - skirting, cornice, an object
    pressed against the wall - drags a least-squares fit even though it is not
    the surface. Trimming the tail and refitting removes that pull without
    needing to identify what the clutter is.
    """
    retained = points
    for _ in range(int(iterations)):
        residuals = np.abs(retained @ normal - offset)
        if len(retained) < 50:
            break
        cutoff = float(np.quantile(residuals, quantile))
        candidate = retained[residuals <= cutoff]
        if len(candidate) < 50:
            break
        retained = candidate
        normal, offset = fit_plane_total_least_squares(retained)
        if float(normal @ UP) < 0:
            normal, offset = -normal, -offset
    return normal, offset, retained


def fit_plane_total_least_squares(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Best-fit plane through points, minimising perpendicular distance.

    Ordinary least squares would minimise error along one axis and bias a
    near-vertical surface badly, so the normal is taken as the smallest
    principal direction of the centred points instead.
    """
    if len(points) < 3:
        raise ValueError("at least three points are needed to fit a plane")
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, float(normal @ centroid)


def coverage_on_plane(
    points: np.ndarray,
    normal: np.ndarray,
    cell_m: float,
) -> tuple[float, tuple[float, float]]:
    """Occupied fraction of a plane's own bounding rectangle, plus its size.

    A plane bounded by a wide rectangle but supported by a thin scatter of
    points is not a wall. Measuring occupancy separates the two.
    """
    helper = np.array([0.0, 1.0, 0.0])
    if abs(float(normal @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    axis_u = np.cross(normal, helper)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)

    u = points @ axis_u
    v = points @ axis_v
    span_u = float(u.max() - u.min())
    span_v = float(v.max() - v.min())
    if span_u <= 0 or span_v <= 0:
        return 0.0, (span_u, span_v)

    cells_u = max(int(np.ceil(span_u / cell_m)), 1)
    cells_v = max(int(np.ceil(span_v / cell_m)), 1)
    iu = np.clip(((u - u.min()) / cell_m).astype(np.int64), 0, cells_u - 1)
    iv = np.clip(((v - v.min()) / cell_m).astype(np.int64), 0, cells_v - 1)
    occupied = len(np.unique(iu * cells_v + iv))
    return float(occupied) / float(cells_u * cells_v), (span_u, span_v)


def _describe_plane(
    plane_id: str,
    kind: str,
    points: np.ndarray,
    frames: np.ndarray,
    normal: np.ndarray,
    offset: float,
    cell_m: float,
    algorithm: str,
    diagnostics: dict[str, Any],
) -> Plane:
    residuals = np.abs(points @ normal - offset)
    coverage, (span_u, span_v) = coverage_on_plane(points, normal, cell_m)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return Plane(
        plane_id=plane_id,
        kind=kind,
        normal=normal,
        offset=offset,
        inlier_count=int(len(points)),
        rms_residual_m=float(np.sqrt(np.mean(residuals ** 2))),
        max_residual_m=float(residuals.max()),
        coverage_fraction=coverage,
        contributing_frames=int(len(np.unique(frames))),
        extent={
            "inPlaneSpanA_m": span_u,
            "inPlaneSpanB_m": span_v,
            "verticalSpan_m": float(upper[1] - lower[1]),
            "boundsMinY_m": float(lower[1]),
            "boundsMaxY_m": float(upper[1]),
        },
        centroid=points.mean(axis=0),
        algorithm=algorithm,
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------
# horizontal structure
# --------------------------------------------------------------------------

def _horizontal_candidates(heights: np.ndarray, bin_m: float, min_fraction: float):
    edges = np.arange(heights.min(), heights.max() + bin_m, bin_m)
    counts, _ = np.histogram(heights, bins=edges)
    threshold = max(int(min_fraction * len(heights)), 1)
    peaks = []
    for index, count in enumerate(counts):
        if count < threshold:
            continue
        if index > 0 and counts[index - 1] > count:
            continue
        if index + 1 < len(counts) and counts[index + 1] > count:
            continue
        peaks.append((float(edges[index] + bin_m / 2), int(count)))
    return peaks, threshold


def extract_horizontal_planes(
    cloud: PointCloud, config: GeometryConfig, rejected: list[dict]
) -> tuple[Plane | None, Plane | None, dict]:
    points, frames = cloud.points, cloud.frame_indices
    bin_m = config.get("height_histogram_bin_m")
    min_fraction = config.get("min_horizontal_support_fraction")
    inlier_distance = config.get("plane_inlier_distance_m")
    max_tilt = config.get("horizontal_normal_max_deviation_deg")
    cell_m = config.get("plane_coverage_cell_m")
    min_height = config.get("min_room_height_m")
    max_height = config.get("max_room_height_m")
    min_plan_coverage = config.get("min_structural_plan_coverage")
    trim_quantile = config.get("plane_trim_quantile")
    trim_iterations = config.get("plane_trim_iterations")

    peaks, threshold = _horizontal_candidates(points[:, 1], bin_m, min_fraction)
    room_cells = plan_cells(points, cell_m)
    diagnostics = {
        "heightPeakCount": len(peaks),
        "supportThresholdPoints": threshold,
        "roomPlanCells": len(room_cells),
        "minStructuralPlanCoverage": min_plan_coverage,
    }
    if not peaks:
        rejected.append({"kind": "horizontal", "reason":
                         "no height slab reached the minimum support fraction"})
        return None, None, diagnostics

    # Score every candidate before choosing. Taking the lowest and highest peak
    # outright would let a low platform stand in for the floor, or a suspended
    # fixture for the ceiling; a structural surface is distinguished by
    # spanning the room's plan, not by being extreme in height.
    scored: list[dict] = []
    for height, count in peaks:
        mask = np.abs(points[:, 1] - height) <= inlier_distance
        if int(mask.sum()) < 3:
            continue
        slab = points[mask]
        normal, offset = fit_plane_total_least_squares(slab)
        if float(normal @ UP) < 0:
            normal, offset = -normal, -offset
        tilt = float(np.degrees(np.arccos(np.clip(float(normal @ UP), -1.0, 1.0))))
        coverage = len(plan_cells(slab, cell_m)) / max(len(room_cells), 1)
        scored.append({
            "height_m": round(height, 4), "supportCount": int(mask.sum()),
            "tiltDeg": round(tilt, 4), "planCoverage": round(coverage, 4),
            "structural": bool(coverage >= min_plan_coverage and tilt <= max_tilt),
            "mask": mask, "normal": normal, "offset": offset,
        })

    diagnostics["candidates"] = [
        {k: v for k, v in entry.items() if k not in ("mask", "normal", "offset")}
        for entry in scored]

    structural = [entry for entry in scored if entry["structural"]]
    for entry in scored:
        if entry["structural"]:
            continue
        rejected.append({
            "kind": "horizontal",
            "reason": f"horizontal surface at y={entry['height_m']:+.3f} m covers "
                      f"{entry['planCoverage'] * 100:.0f}% of the room plan, below the "
                      f"{min_plan_coverage * 100:.0f}% a structural floor or ceiling "
                      f"spans; treated as furniture or a fixture rather than "
                      f"architecture",
            "supportCount": entry["supportCount"]})

    diagnostics["structuralCandidates"] = len(structural)
    if not structural:
        rejected.append({"kind": "horizontal", "reason":
                         "no horizontal surface spans enough of the room plan to be "
                         "structural; no floor or ceiling was accepted"})
        return None, None, diagnostics

    def build(entry: dict, kind: str, index: int) -> Plane:
        normal, offset, retained = trimmed_plane_fit(
            points[entry["mask"]], entry["normal"], entry["offset"],
            trim_quantile, trim_iterations)
        keep = np.abs(points @ normal - offset) <= inlier_distance
        keep &= entry["mask"]
        selected = points[keep]
        return _describe_plane(
            f"{kind}-{index:03d}", kind, selected, frames[keep], normal, offset, cell_m,
            "gravity_aligned_height_histogram_structural_selection_trimmed_tls_v0.1",
            {"histogramPeakHeight_m": entry["height_m"],
             "tiltFromUpDeg": entry["tiltDeg"],
             "planCoverage": entry["planCoverage"],
             "pointsBeforeTrim": int(entry["mask"].sum()),
             "pointsAfterTrim": int(len(retained))})

    floor = build(structural[0], "floor", 1)
    ceiling = build(structural[-1], "ceiling", 1) if len(structural) > 1 else None

    if floor is not None and ceiling is not None:
        separation = float(ceiling.centroid[1] - floor.centroid[1])
        diagnostics["floorCeilingSeparation_m"] = round(separation, 4)
        if not (min_height <= separation <= max_height):
            rejected.append({
                "kind": "ceiling", "reason":
                f"floor-to-ceiling separation {separation:.3f} m falls outside the "
                f"plausible range {min_height}-{max_height} m; the ceiling candidate "
                f"was discarded rather than reported",
                "supportCount": ceiling.inlier_count})
            ceiling = None
    return floor, ceiling, diagnostics


# --------------------------------------------------------------------------
# vertical structure
# --------------------------------------------------------------------------

def extract_wall_planes(
    cloud: PointCloud,
    config: GeometryConfig,
    floor: Plane | None,
    ceiling: Plane | None,
    rejected: list[dict],
) -> tuple[list[Plane], dict]:
    points, frames = cloud.points, cloud.frame_indices
    margin = config.get("floor_ceiling_exclusion_margin_m")
    vote_voxel = config.get("wall_vote_voxel_size_m")
    angle_step = config.get("wall_angle_step_deg")
    offset_bin = config.get("wall_offset_bin_m")
    inlier_distance = config.get("plane_inlier_distance_m")
    min_wall_height = config.get("min_wall_height_m")
    min_wall_length = config.get("min_wall_length_m")
    min_coverage = config.get("min_wall_coverage_fraction")
    merge_angle = config.get("wall_merge_angle_deg")
    merge_distance = config.get("wall_merge_distance_m")
    cell_m = config.get("plane_coverage_cell_m")
    max_deviation = config.get("vertical_normal_max_deviation_deg")

    mask = np.ones(len(points), dtype=bool)
    if floor is not None:
        mask &= points[:, 1] > floor.centroid[1] + margin
    if ceiling is not None:
        mask &= points[:, 1] < ceiling.centroid[1] - margin
    candidate_points = points[mask]
    candidate_frames = frames[mask]

    diagnostics = {
        "pointsAfterFloorCeilingExclusion": int(len(candidate_points)),
        "pointsExcluded": int(len(points) - len(candidate_points)),
    }
    if len(candidate_points) < 3:
        rejected.append({"kind": "wall", "reason":
                         "no points remain once floor and ceiling bands are excluded"})
        return [], diagnostics

    voted, voted_frames = voxel_downsample(
        candidate_points, vote_voxel, candidate_frames.astype(np.int64))
    diagnostics["voteSamplePoints"] = int(len(voted))

    angles = np.deg2rad(np.arange(0.0, 180.0, angle_step))
    plan = voted[:, [0, 2]]
    best: list[tuple[int, float, float]] = []
    for theta in angles:
        direction = np.array([np.cos(theta), np.sin(theta)])
        rho = plan @ direction
        edges = np.arange(rho.min(), rho.max() + offset_bin, offset_bin)
        if len(edges) < 2:
            continue
        counts, _ = np.histogram(rho, bins=edges)
        for index, count in enumerate(counts):
            if count == 0:
                continue
            if index > 0 and counts[index - 1] > count:
                continue
            if index + 1 < len(counts) and counts[index + 1] > count:
                continue
            best.append((int(count), float(theta),
                         float(edges[index] + offset_bin / 2)))

    best.sort(key=lambda item: -item[0])
    diagnostics["orientationOffsetPeaks"] = len(best)
    # Screening runs on the coarse vote cloud: a wall that cannot be seen at
    # 5 cm is not a wall. Only survivors are refitted at full resolution, which
    # keeps the search linear in the number of accepted surfaces rather than in
    # the number of peaks.
    best = best[:MAX_CANDIDATE_PEAKS]
    diagnostics["candidatePeaksScreened"] = len(best)

    accepted: list[tuple[np.ndarray, float, dict]] = []
    claimed_coarse = np.zeros(len(voted), dtype=bool)

    for count, theta, rho in best:
        if len(accepted) >= MAX_WALLS:
            break
        normal = np.array([np.cos(theta), 0.0, np.sin(theta)])
        fresh = (np.abs(voted @ normal - rho) <= inlier_distance) & ~claimed_coarse
        if int(fresh.sum()) < MIN_COARSE_SUPPORT:
            continue

        selected = voted[fresh]
        fitted_normal, _ = fit_plane_total_least_squares(selected)
        vertical_deviation = float(np.degrees(np.arcsin(
            np.clip(abs(float(fitted_normal @ UP)), -1.0, 1.0))))
        if vertical_deviation > max_deviation:
            rejected.append({
                "kind": "wall", "reason":
                f"candidate at offset {rho:.3f} m tilts {vertical_deviation:.2f} deg "
                f"from vertical, beyond the {max_deviation} deg tolerance",
                "supportCount": int(fresh.sum())})
            continue

        # Re-level the normal so the wall is exactly vertical; a wall that leans
        # a degree is a fit artefact, not architecture.
        levelled = fitted_normal - UP * float(fitted_normal @ UP)
        norm = float(np.linalg.norm(levelled))
        if norm < 1e-9:
            continue
        levelled /= norm
        levelled_offset = float(levelled @ selected.mean(axis=0))

        refined = (np.abs(voted @ levelled - levelled_offset) <= inlier_distance)
        refined &= ~claimed_coarse
        if int(refined.sum()) < MIN_COARSE_SUPPORT:
            continue
        selected = voted[refined]

        vertical_span = float(selected[:, 1].max() - selected[:, 1].min())
        coverage, (span_a, span_b) = coverage_on_plane(selected, levelled, cell_m)
        horizontal_span = min(span_a, span_b) if vertical_span >= max(span_a, span_b) \
            else max(span_a, span_b)
        if abs(span_a - vertical_span) < abs(span_b - vertical_span):
            horizontal_span = span_b
        else:
            horizontal_span = span_a

        if vertical_span < min_wall_height:
            rejected.append({
                "kind": "wall", "reason":
                f"candidate spans only {vertical_span:.2f} m vertically, below the "
                f"{min_wall_height} m minimum for architecture",
                "supportCount": int(refined.sum())})
            continue
        if horizontal_span < min_wall_length:
            rejected.append({
                "kind": "wall", "reason":
                f"candidate runs only {horizontal_span:.2f} m horizontally, below the "
                f"{min_wall_length} m minimum",
                "supportCount": int(refined.sum())})
            continue
        if coverage < min_coverage:
            rejected.append({
                "kind": "wall", "reason":
                f"candidate occupies {coverage * 100:.1f}% of its own bounding "
                f"rectangle, below the {min_coverage * 100:.0f}% minimum; the support "
                f"is scattered rather than a surface",
                "supportCount": int(refined.sum())})
            continue

        # Whether two candidates describe one surface is a question about
        # geometry, not about their offset scalars: those are measured along
        # different normals and are not comparable. Ask instead how far this
        # candidate's own points sit from the plane already accepted.
        centroid = selected.mean(axis=0)
        duplicate = False
        for existing_normal, existing_offset, _ in accepted:
            angle = float(np.degrees(np.arccos(np.clip(
                abs(float(existing_normal @ levelled)), -1.0, 1.0))))
            separation = abs(float(existing_normal @ centroid) - existing_offset)
            if angle <= merge_angle and separation <= merge_distance:
                duplicate = True
                break
        if duplicate:
            continue

        accepted.append((levelled, levelled_offset, {
            "votePeakCount": int(count),
            "voteOrientationDeg": round(float(np.degrees(theta)), 3),
            "voteOffset_m": round(rho, 4),
            "fitTiltFromVerticalDeg": round(vertical_deviation, 4),
            "coarseSupportCount": int(refined.sum()),
        }))
        claimed_coarse |= refined

    # Point every wall normal at the room interior, approximated by the plan
    # centroid of the candidate points. A consistent orientation lets the
    # envelope stage reason about inside and outside without re-deriving it.
    interior = candidate_points.mean(axis=0)
    oriented: list[tuple[np.ndarray, float, dict]] = []
    for normal, offset, notes in accepted:
        if float(normal @ interior) - offset < 0:
            normal, offset = -normal, -offset
        oriented.append((normal, offset, notes))
    accepted = oriented

    # One full-resolution pass, only for surfaces that survived screening.
    walls: list[Plane] = []
    claimed_fine = np.zeros(len(candidate_points), dtype=bool)
    for index, (normal, offset, notes) in enumerate(accepted, start=1):
        fine = (np.abs(candidate_points @ normal - offset) <= inlier_distance)
        fine &= ~claimed_fine
        if int(fine.sum()) < 3:
            continue
        selected = candidate_points[fine]
        coverage, (span_a, span_b) = coverage_on_plane(selected, normal, cell_m)
        vertical_span = float(selected[:, 1].max() - selected[:, 1].min())
        notes = dict(notes)
        notes["horizontalSpan_m"] = round(
            span_b if abs(span_a - vertical_span) < abs(span_b - vertical_span)
            else span_a, 4)
        walls.append(_describe_plane(
            f"wall-{index:03d}", "wall", selected, candidate_frames[fine],
            normal, offset, cell_m,
            "plan_hough_vote_then_levelled_total_least_squares_v0.1", notes))
        claimed_fine |= fine

    walls.sort(key=lambda w: -w.inlier_count)
    for index, wall in enumerate(walls, start=1):
        wall.plane_id = f"wall-{index:03d}"
    diagnostics["wallsAccepted"] = len(walls)
    return walls, diagnostics


def extract_planes(cloud: PointCloud, config: GeometryConfig) -> PlaneSet:
    rejected: list[dict] = []
    floor, ceiling, horizontal_diagnostics = extract_horizontal_planes(
        cloud, config, rejected)
    walls, wall_diagnostics = extract_wall_planes(cloud, config, floor, ceiling, rejected)

    diagnostics = {
        "pointCount": int(len(cloud.points)),
        "horizontal": horizontal_diagnostics,
        "wall": wall_diagnostics,
        "floorFound": floor is not None,
        "ceilingFound": ceiling is not None,
        "wallCount": len(walls),
        "rejectedCandidateCount": len(rejected),
    }
    return PlaneSet(
        floor=floor,
        ceiling=ceiling,
        walls=walls,
        rejected=rejected,
        diagnostics=diagnostics,
        config_provenance=config.provenance(*CONFIG_PARAMETERS),
    )
