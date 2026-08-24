"""Deterministic vertical-axis resolution at the vendor boundary.

A capture arrives with a *declared* gravity axis, because ARKit session frames
are gravity-aligned by construction and the exporter knows it. A declaration is
information, not proof, so it is verified here against what the depth actually
observed: does this orientation produce a floor, a ceiling, a plausible storey
height, and a direction consistent with how the device was held.

The rule this module implements is deliberately boring:

    declared axis exists and verifies            -> accept it
    exactly one candidate verifies               -> accept that one
    several verify, none clearly better          -> ambiguous, hand it on
    none verifies                                -> reject the capture

Verification uses observed structure (floor, ceiling, plausible storey height)
and camera-path extent along the candidate vertical. Mean camera-up versus a
candidate axis is not a gate: device-to-camera IMU offset makes that angle an
unreliable convention signal. Sign comes from the source declaration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ingestion_frame_config_v0.1.json"

CANDIDATE_AXES = ("+y", "-y", "+z", "-z", "+x", "-x")

AXIS_VECTORS = {
    "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
}


class FrameResolutionError(Exception):
    """The vertical axis could not be resolved from this capture."""


def load_ingestion_config(path: Path | None = None) -> dict[str, Any]:
    document = json.loads((path or CONFIG_PATH).read_text())
    return {name: entry["value"] for name, entry in document["parameters"].items()}


@dataclass
class CandidateEvidence:
    """What one candidate up-axis implies about the observed structure."""

    axis: str
    floorDetected: bool = False
    ceilingDetected: bool = False
    floorHeightM: float | None = None
    ceilingHeightM: float | None = None
    roomHeightM: float | None = None
    floorRmsMm: float | None = None
    ceilingRmsMm: float | None = None
    horizontalSupportFraction: float = 0.0
    verticalStructureFraction: float = 0.0
    trajectoryVerticalRatio: float | None = None
    plausible: bool = False
    rejections: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameResolution:
    """The decision, the evidence behind it, and how confident the rule is."""

    outcome: str                 # verified | ambiguous | unsupported
    axis: str | None
    basis: str                   # declared_verified | single_candidate | none
    declared_axis: str | None
    declared_verified: bool
    candidates: list[CandidateEvidence]
    notes: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.outcome == "verified"

    def to_record(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "axis": self.axis,
            "basis": self.basis,
            "declaredAxis": self.declared_axis,
            "declaredVerified": self.declared_verified,
            "candidates": [c.to_record() for c in self.candidates],
            "notes": list(self.notes),
        }


def _rotation_taking(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)
    cross = np.cross(a, b)
    dot = float(a @ b)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        if dot > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(a @ helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        skew = np.array([[0.0, -axis[2], axis[1]],
                         [axis[2], 0.0, -axis[0]],
                         [-axis[1], axis[0], 0.0]])
        return np.eye(3) + 2.0 * (skew @ skew)
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (sine ** 2))


def sample_world_points(
    root: Path,
    capture,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject an evenly spaced subset of depth frames into the source world.

    A subset, because this decides an orientation rather than measures a room;
    evenly spaced and deterministic, so the answer does not depend on ordering.
    Points are built as R @ p_cam + t (camera-to-world), matching the connector.
    """
    frames = capture.frames
    if not frames:
        raise FrameResolutionError("the capture contains no frames; no axis can be verified")

    # Evenly spaced *positions*, not a fixed integer stride. A stride can resonate
    # with a periodic sweep and land on frames that all miss the same surface,
    # which is how a real ceiling went unseen and a room read as 1.3 m tall.
    wanted = int(config["verification_frame_count"])
    if len(frames) <= wanted:
        selected = list(frames)
    else:
        positions = np.linspace(0, len(frames) - 1, wanted)
        selected = [frames[int(round(p))] for p in positions]

    intrinsics = next((i for i in capture.intrinsics if i.stream == "depth"), None)
    if intrinsics is None:
        raise FrameResolutionError("the capture declares no depth intrinsics")

    depth_min = float(config["verification_depth_min_m"])
    depth_max = float(config["verification_depth_max_m"])

    chunks: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for position, record in enumerate(selected):
        raw = np.array(Image.open(root / record.depth))
        metres = raw.astype(np.float64) * capture.depth_scale_m
        valid = (raw > 0) & (metres >= depth_min) & (metres <= depth_max)
        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            continue
        z = metres[rows, cols]
        camera = np.stack([
            (cols - intrinsics.cx) / intrinsics.fx * z,
            (rows - intrinsics.cy) / intrinsics.fy * z,
            z,
        ], axis=1)
        pose = np.asarray(record.camera_to_world, dtype=np.float64)
        chunks.append(camera @ pose[:3, :3].T + pose[:3, 3])
        groups.append(np.full(rows.size, position, dtype=np.int32))

    if not chunks:
        raise FrameResolutionError(
            "no depth return survived range filtering; the capture cannot support "
            "a frame decision")
    return np.concatenate(chunks), np.concatenate(groups)


def _trajectory_vertical_ratio(centres: np.ndarray, axis: str) -> float | None:
    """Camera-path extent along `axis` over the larger horizontal camera-path extent.

    None when the horizontal path is too short to discriminate.
    """
    if len(centres) < 2:
        return None
    extents = np.ptp(np.asarray(centres, dtype=np.float64), axis=0)
    direction = np.abs(AXIS_VECTORS[axis])
    vertical = float(direction @ extents)
    horizontal = max(
        float(extents[i]) for i in range(3) if direction[i] < 0.5)
    if horizontal < 0.4:
        return None
    return vertical / horizontal


def evaluate_candidate(
    points: np.ndarray,
    centres: np.ndarray,
    axis: str,
    config: dict[str, Any],
) -> CandidateEvidence:
    """Score one candidate axis against the observed points and camera path.

    A correct up axis concentrates points into a floor band and a ceiling band
    a storey apart, and the camera path along that axis stays short relative to
    the walk. Sign is not taken from the trajectory; the declaration supplies it.
    Mean camera-up versus the axis is recorded only as a diagnostic.
    """
    evidence = CandidateEvidence(axis=axis)
    direction = AXIS_VECTORS[axis]
    ratio = _trajectory_vertical_ratio(centres, axis)
    evidence.trajectoryVerticalRatio = None if ratio is None else round(ratio, 4)
    max_ratio = float(config["max_trajectory_vertical_ratio"])
    if ratio is not None and ratio > max_ratio:
        evidence.rejections.append(
            f"camera-path extent along this axis is {ratio:.2f}x the larger "
            f"horizontal camera-path extent, above the {max_ratio:.2f} handheld "
            f"bound; this axis is describing the walk, not gravity")

    rotation = _rotation_taking(direction, np.array([0.0, 1.0, 0.0]))
    heights = (points @ rotation.T)[:, 1]

    bin_size = float(config["height_bin_m"])
    low, high = float(heights.min()), float(heights.max())
    if high - low < bin_size:
        evidence.rejections.append("observed points have no vertical extent along this axis")
        return evidence

    counts, edges = np.histogram(heights, bins=max(int(np.ceil((high - low) / bin_size)), 1))
    bin_centres = (edges[:-1] + edges[1:]) / 2.0
    total = float(len(heights))
    fractions = counts / total

    min_support = float(config["min_horizontal_support_fraction"])
    supported = np.nonzero(fractions >= min_support)[0]
    if supported.size == 0:
        evidence.rejections.append(
            f"no height band holds {min_support:.0%} of the points; this axis does not "
            f"concentrate the observation into horizontal structure")
        return evidence

    # Group the supported bins into contiguous bands and take the *strongest*
    # pair a storey apart, not the outermost pair. The extremes are wherever the
    # cloud happens to end, so a partly sampled ceiling or a stray return moves
    # them; the floor and ceiling of a real room are its densest bands.
    groups: list[list[int]] = []
    for index in supported:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])

    bands = []
    for group in groups:
        weight = float(fractions[group].sum())
        centre = float(np.average(bin_centres[group], weights=counts[group]))
        bands.append((centre, weight))

    low_gate = float(config["min_room_height_m"])
    high_gate = float(config["max_room_height_m"])
    pairs = [
        (lower, upper, lw + uw)
        for i, (lower, lw) in enumerate(bands)
        for (upper, uw) in bands[i + 1:]
        if low_gate <= (upper - lower) <= high_gate
    ]

    slab = float(config["plane_slab_m"])

    def band(centre: float) -> np.ndarray:
        return heights[np.abs(heights - centre) <= slab]

    if pairs:
        floor_height, ceiling_height, _ = max(pairs, key=lambda item: item[2])
        floor_points, ceiling_points = band(floor_height), band(ceiling_height)
        evidence.floorDetected = floor_points.size / total >= min_support
        evidence.ceilingDetected = ceiling_points.size / total >= min_support
    else:
        # Nothing a storey apart: report the strongest band so the evidence still
        # says what was seen, and let the plausibility gate below reject it.
        floor_height = max(bands, key=lambda item: item[1])[0]
        ceiling_height = float(bin_centres[int(supported[-1])])
        floor_points, ceiling_points = band(floor_height), band(ceiling_height)
        evidence.floorDetected = floor_points.size / total >= min_support
        evidence.ceilingDetected = (ceiling_points.size / total >= min_support
                                    and abs(ceiling_height - floor_height) > slab)
    evidence.floorHeightM = round(floor_height, 4)
    evidence.ceilingHeightM = round(ceiling_height, 4)
    evidence.horizontalSupportFraction = round(
        float(floor_points.size + ceiling_points.size) / total, 5)

    if evidence.floorDetected:
        evidence.floorRmsMm = round(float(np.sqrt(np.mean(
            (floor_points - floor_points.mean()) ** 2))) * 1000.0, 2)
    if evidence.ceilingDetected:
        evidence.ceilingRmsMm = round(float(np.sqrt(np.mean(
            (ceiling_points - ceiling_points.mean()) ** 2))) * 1000.0, 2)

    between = heights[(heights > floor_height + slab) & (heights < ceiling_height - slab)]
    evidence.verticalStructureFraction = round(float(between.size) / total, 5)

    if not evidence.floorDetected:
        evidence.rejections.append("no floor band")
    if not evidence.ceilingDetected:
        evidence.rejections.append("no ceiling band")

    if evidence.floorDetected and evidence.ceilingDetected:
        height = ceiling_height - floor_height
        evidence.roomHeightM = round(float(height), 4)
        if not (low_gate <= height <= high_gate):
            evidence.rejections.append(
                f"implied storey height {height:.2f} m is outside the plausible "
                f"{low_gate}-{high_gate} m range")

    evidence.plausible = not evidence.rejections
    return evidence


def resolve_frame(
    root: Path,
    capture,
    config: dict[str, Any] | None = None,
    candidates: tuple[str, ...] = CANDIDATE_AXES,
) -> FrameResolution:
    """Decide the vertical axis for a capture, or decline to."""
    config = config or load_ingestion_config()
    points, groups = sample_world_points(Path(root), capture, config)
    centres = np.array([np.asarray(frame.camera_to_world, dtype=np.float64)[:3, 3]
                        for frame in capture.frames])

    evidence = [evaluate_candidate(points, centres, axis, config)
                for axis in candidates]

    def agrees(axis: str) -> bool:
        """Does each half of the sampled frames reach the same verdict alone?

        Sampling a subset of a capture can miss a whole ceiling, and a decision
        that flips with the sample is not one to hand to frozen geometry. Two
        disjoint halves that agree is cheap evidence that it will not flip.
        """
        for half in (groups % 2 == 0, groups % 2 == 1):
            if not half.any():
                return False
            if not evaluate_candidate(points[half], centres, axis, config).plausible:
                return False
        return True
    by_axis = {item.axis: item for item in evidence}
    passing = [item for item in evidence if item.plausible]
    notes: list[str] = []

    declared = (capture.world_up_axis or "").strip().lower() or None

    # 1. A declaration that verifies is the best information available.
    if declared and declared in by_axis and by_axis[declared].plausible:
        if agrees(declared):
            notes.append(
                f"the source declared '{declared}' and the observed structure supports it: "
                f"floor and ceiling {by_axis[declared].roomHeightM} m apart, on both "
                f"halves of the sampled frames independently")
            return FrameResolution("verified", declared, "declared_verified",
                                   declared, True, evidence, notes)
        notes.append(
            f"the source declared '{declared}' and the full sample supports it, but the "
            f"two halves of that sample do not agree; the evidence is too thin to confirm")
        return FrameResolution("ambiguous", None, "none", declared, False, evidence, notes)

    if declared and declared in by_axis and by_axis[declared].rejections:
        notes.append(
            f"the source declared '{declared}' but the observation does not support it: "
            + "; ".join(by_axis[declared].rejections))

    # 2. Exactly one candidate that verifies is still a decision.
    if len(passing) == 1:
        winner = passing[0]
        notes.append(f"'{winner.axis}' is the only orientation that produces a coherent room")
        return FrameResolution("verified", winner.axis, "single_candidate",
                               declared, declared == winner.axis, evidence, notes)

    # 3. Several verify: accept only if one is clearly better supported.
    if len(passing) > 1:
        passing.sort(key=lambda item: item.horizontalSupportFraction, reverse=True)
        best, runner_up = passing[0], passing[1]
        margin = best.horizontalSupportFraction / max(runner_up.horizontalSupportFraction, 1e-9)
        if margin >= float(config["min_support_margin"]):
            notes.append(
                f"'{best.axis}' is supported {margin:.1f}x more strongly than "
                f"'{runner_up.axis}'")
            return FrameResolution("verified", best.axis, "dominant_candidate",
                                   declared, declared == best.axis, evidence, notes)
        notes.append(
            f"'{best.axis}' and '{runner_up.axis}' are both physically plausible and "
            f"within {margin:.2f}x of each other; the observation does not choose")
        return FrameResolution("ambiguous", None, "none", declared, False, evidence, notes)

    # 4. Nothing verifies.
    notes.append("no candidate axis produces a floor, a ceiling and a plausible storey height")
    return FrameResolution("unsupported", None, "none", declared, False, evidence, notes)


def apply_resolution(root: Path, capture, resolution: FrameResolution) -> None:
    """Record the frame decision on the capture and rewrite its manifest.

    The raw bundle is never touched; only the normalized capture carries the
    decision, and it carries the evidence with it so the choice stays auditable
    long after the run.
    """
    root = Path(root)
    if resolution.accepted:
        capture.world_up_axis = resolution.axis
        capture.world_up_axis_verified = True
    else:
        capture.world_up_axis_verified = False

    capture.provenance.notes.append(
        f"Vertical axis {resolution.outcome}"
        + (f" as '{resolution.axis}' ({resolution.basis})." if resolution.axis else ".")
        + " " + " ".join(resolution.notes))

    (root / "frame_resolution.json").write_text(
        json.dumps(resolution.to_record(), indent=2) + "\n")
    capture.write(root)
