"""Independent reference geometry from ARKitScenes FARO laser scans.

This is the only quantitative check in the POC that does not come from the
device under evaluation. The ARKit mesh cannot serve: it is produced by the
same reconstruction stack being measured, so agreeing with it proves nothing.

What can honestly be compared is bounded by what is available. ARKitScenes
publishes no transform between the ARKit session frame and the FARO visit
frame, so horizontal quantities — wall length, floor area — cannot be compared
without first solving registration. The documented fallback applies:
compare **floor-to-ceiling height**, the one room dimension that needs gravity
alignment but no horizontal registration, and state plainly that nothing else
was established.

The scans' own pose files are checked to confirm the visit frame is levelled
before that reasoning is used, rather than assuming it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "reference_extraction_v0.1.json"

# Matches the published ARKitScenes laser-scan PLY layout.
FARO_DTYPE = np.dtype([
    ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
    ("quality", "<f8"), ("radius", "<f8"),
])


class ReferenceError(Exception):
    """The reference asset is missing, unreadable, or cannot support a comparison."""


@dataclass
class ReferenceConfig:
    config_id: str
    raw: dict
    sha256: str

    def get(self, name: str):
        try:
            return self.raw["parameters"][name]["value"]
        except KeyError as exc:
            raise ReferenceError(f"'{name}' is not defined in {self.config_id}") from exc


def load_reference_config(path: Path | None = None) -> ReferenceConfig:
    path = Path(path or DEFAULT_CONFIG)
    if not path.is_file():
        raise ReferenceError(f"reference configuration '{path}' does not exist")
    text = path.read_text()
    raw = json.loads(text)
    return ReferenceConfig(config_id=raw["configId"], raw=raw,
                           sha256=hashlib.sha256(text.encode()).hexdigest())


@dataclass
class ReferenceGeometry:
    floor_height_m: float
    ceiling_height_m: float
    separation_m: float
    scans_used: list[str]
    point_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def read_ply_header(path: Path) -> tuple[int, int]:
    """Vertex count and byte offset of the binary payload."""
    with Path(path).open("rb") as handle:
        header = b""
        while b"end_header" not in header:
            line = handle.readline()
            if not line:
                raise ReferenceError(f"'{path.name}' has no PLY end_header")
            header += line
        offset = handle.tell()
    text = header.decode("ascii", "replace")
    if "binary_little_endian" not in text:
        raise ReferenceError(f"'{path.name}' is not binary little-endian PLY")
    counts = [line for line in text.splitlines() if line.startswith("element vertex")]
    if not counts:
        raise ReferenceError(f"'{path.name}' declares no vertex count")
    return int(counts[0].split()[-1]), offset


def read_pose(path: Path) -> np.ndarray:
    """A scan's 4x4 placement in the visit frame.

    ARKitScenes writes these row-major with translation in the final row, the
    transposed convention, so points transform as `p_visit = p_scan @ M`.
    """
    rows = [[float(v) for v in line.split(",")]
            for line in Path(path).read_text().strip().splitlines()]
    matrix = np.array(rows, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ReferenceError(f"'{path.name}' is {matrix.shape}, expected a 4x4 pose")
    return matrix


def load_visit_points(
    visit_dir: Path, config: ReferenceConfig
) -> tuple[np.ndarray, list[str], dict]:
    visit_dir = Path(visit_dir)
    scans = sorted(visit_dir.glob("*.ply"))
    if not scans:
        raise ReferenceError(f"'{visit_dir}' contains no FARO scans")

    stride = int(config.get("point_stride"))
    tolerance_deg = config.get("level_axis_tolerance_deg")

    chunks: list[np.ndarray] = []
    used: list[str] = []
    level_checks: list[dict] = []

    for scan in scans:
        pose_path = scan.with_name(f"{scan.stem}_pose.txt")
        if not pose_path.is_file():
            raise ReferenceError(f"'{scan.name}' has no companion pose file")
        pose = read_pose(pose_path)

        # A tripod-levelled scanner keeps the visit frame's vertical axis fixed.
        # Verify that rather than assume it, and measure the departure as the
        # angle it actually is, so the height error it would induce is visible.
        vertical_row = pose[2, :3]
        vertical_row = vertical_row / np.linalg.norm(vertical_row)
        tilt_deg = float(np.degrees(np.arccos(np.clip(abs(vertical_row[2]), -1.0, 1.0))))
        level_checks.append({
            "scan": scan.stem,
            "verticalAxis": [round(float(v), 6) for v in vertical_row],
            "tiltFromVisitVerticalDeg": round(tilt_deg, 4),
            "inducedHeightErrorPer2p6m_mm": round(
                2.6 * (1 - np.cos(np.radians(tilt_deg))) * 1000, 3),
            "levelled": tilt_deg <= tolerance_deg,
        })

        count, offset = read_ply_header(scan)
        data = np.memmap(scan, dtype=FARO_DTYPE, mode="r", offset=offset, shape=(count,))
        sampled = data[::stride]
        local = np.stack([sampled["x"], sampled["y"], sampled["z"]], axis=1).astype(np.float64)
        homogeneous = np.concatenate([local, np.ones((len(local), 1))], axis=1)
        chunks.append((homogeneous @ pose)[:, :3])
        used.append(scan.stem)
        del data

    not_level = [c for c in level_checks if not c["levelled"]]
    if not_level:
        worst = max(not_level, key=lambda c: c["tiltFromVisitVerticalDeg"])
        raise ReferenceError(
            f"scan {worst['scan']} tilts {worst['tiltFromVisitVerticalDeg']:.2f} deg "
            f"from the visit frame's vertical, beyond the {tolerance_deg} deg "
            f"tolerance; the frame cannot be treated as levelled and a "
            f"floor-to-ceiling comparison would need full registration")

    points = np.concatenate(chunks)
    diagnostics = {
        "scansUsed": used,
        "pointStride": stride,
        "pointsSampled": int(len(points)),
        "levelCheck": level_checks,
        "visitFrameUpAxis": "+z",
        "maxScanTiltDeg": round(max(c["tiltFromVisitVerticalDeg"]
                                    for c in level_checks), 4),
        "visitFrameUpAxisEvidence": (
            "Every scan pose preserves the visit frame's vertical axis to within the "
            "configured angular tolerance, which is what a tripod-levelled "
            "terrestrial scanner produces. The induced height error is reported "
            "per scan."),
    }
    return points, used, diagnostics


def extract_reference_geometry(
    visit_dir: Path, config: ReferenceConfig | None = None
) -> ReferenceGeometry:
    config = config or load_reference_config()
    points, used, diagnostics = load_visit_points(visit_dir, config)

    bin_m = config.get("height_bin_m")
    min_support = config.get("min_slab_support_fraction")
    min_storey = config.get("min_storey_height_m")
    max_storey = config.get("max_storey_height_m")

    heights = points[:, 2]        # visit frame is +z up, verified above
    edges = np.arange(heights.min(), heights.max() + bin_m, bin_m)
    counts, _ = np.histogram(heights, bins=edges)
    threshold = max(int(min_support * len(heights)), 1)

    peaks: list[tuple[float, int]] = []
    for index, count in enumerate(counts):
        if count < threshold:
            continue
        if index > 0 and counts[index - 1] > count:
            continue
        if index + 1 < len(counts) and counts[index + 1] > count:
            continue
        peaks.append((float(edges[index] + bin_m / 2), int(count)))

    if len(peaks) < 2:
        raise ReferenceError(
            f"only {len(peaks)} horizontal structure candidates were found in the "
            f"reference cloud; a floor-to-ceiling separation cannot be established")

    # The storey is the strongest floor/ceiling pair whose separation is
    # habitable. Picking the extremes would pair a basement slab with a roof.
    best: tuple[int, float, float] | None = None
    for i, (low, low_count) in enumerate(peaks):
        for high, high_count in peaks[i + 1:]:
            separation = high - low
            if not (min_storey <= separation <= max_storey):
                continue
            strength = min(low_count, high_count)
            if best is None or strength > best[0]:
                best = (strength, low, high)
    if best is None:
        raise ReferenceError(
            f"no pair of horizontal structures in the reference cloud is separated by "
            f"{min_storey}-{max_storey} m; no storey height can be extracted")

    # How ambiguous is the reference itself? A visit covers a whole dwelling, and
    # different rooms rarely share a ceiling height to the millimetre. If the
    # plausible pairings span more than the discrepancy under discussion, the
    # reference cannot adjudicate it, and saying so is more useful than a single
    # authoritative-looking number.
    pairings = []
    for i, (low, low_count) in enumerate(sorted(peaks)):
        for high, high_count in sorted(peaks)[i + 1:]:
            separation = high - low
            if min_storey <= separation <= max_storey and min(low_count, high_count) \
                    > threshold * 3:
                pairings.append({"floor_m": round(low, 4),
                                 "ceiling_m": round(high, 4),
                                 "separation_m": round(separation, 4),
                                 "minSupport": int(min(low_count, high_count))})
    spread = ([p["separation_m"] for p in pairings] or [0.0])
    diagnostics["plausiblePairings"] = sorted(
        pairings, key=lambda p: -p["minSupport"])[:10]
    diagnostics["separationRange_m"] = [min(spread), max(spread)]
    diagnostics["separationSpread_cm"] = round((max(spread) - min(spread)) * 100, 2)

    _, floor_height, ceiling_height = best
    diagnostics.update({
        "histogramBin_m": bin_m,
        "supportThresholdPoints": threshold,
        "peakCount": len(peaks),
        "strongestPeaks": [{"height_m": round(h, 4), "count": c}
                           for h, c in sorted(peaks, key=lambda p: -p[1])[:8]],
        "selectedFloor_m": round(floor_height, 4),
        "selectedCeiling_m": round(ceiling_height, 4),
        "referenceConfigId": config.config_id,
        "referenceConfigHash": config.sha256,
    })
    return ReferenceGeometry(
        floor_height_m=floor_height,
        ceiling_height_m=ceiling_height,
        separation_m=ceiling_height - floor_height,
        scans_used=used,
        point_count=int(len(points)),
        diagnostics=diagnostics,
    )
