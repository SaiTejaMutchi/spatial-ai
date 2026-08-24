"""Source-neutral validation of a `normalized_capture` package.

This is the gate that emits `NORMALIZED_CAPTURE_VALID`. It knows nothing about
ARKitScenes or Stray Scanner; it only knows the contract. Every failure names
the specific record at fault so an integration problem points at a frame, not
at a stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .normalized_capture import (
    CORE_MODALITIES,
    MAX_PLAUSIBLE_CAMERA_PATH_M,
    PLAUSIBLE_DEPTH_M,
    PLAUSIBLE_FOCAL_PX,
    NormalizedCapture,
)

ORTHONORMAL_TOLERANCE = 1e-4


@dataclass
class Issue:
    severity: str      # "error" | "warning"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


def _check_rotation(matrix: np.ndarray) -> str | None:
    rot = matrix[:3, :3]
    if not np.isfinite(matrix).all():
        return "contains non-finite values"
    err = float(np.abs(rot @ rot.T - np.eye(3)).max())
    if err > ORTHONORMAL_TOLERANCE:
        return f"rotation is not orthonormal (max |RR^T - I| = {err:.2e})"
    det = float(np.linalg.det(rot))
    if abs(det - 1.0) > ORTHONORMAL_TOLERANCE:
        if abs(det + 1.0) <= ORTHONORMAL_TOLERANCE:
            return "rotation determinant is -1 (left-handed or mirrored frame)"
        return f"rotation determinant is {det:.6f}, expected +1"
    bottom = matrix[3]
    if not np.allclose(bottom, [0, 0, 0, 1], atol=1e-9):
        return f"bottom row is {bottom.tolist()}, expected [0, 0, 0, 1]"
    return None


def validate(root: Path, depth_sample: int = 20) -> list[Issue]:
    root = Path(root)
    issues: list[Issue] = []

    for name in ("manifest.json", "provenance.json", "intrinsics.json", "trajectory.json"):
        if not (root / name).exists():
            issues.append(Issue("error", "MISSING_CONTRACT_FILE", f"{name} is absent"))
    if issues:
        return issues

    try:
        capture = NormalizedCapture.read(root)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        return [Issue("error", "UNREADABLE_CONTRACT", f"{type(exc).__name__}: {exc}")]

    # -- modality availability -------------------------------------------
    for modality in CORE_MODALITIES:
        state = capture.modalities.get(modality, {}).get("available")
        if state is not True:
            issues.append(Issue("error", "MISSING_CORE_MODALITY",
                                f"core modality '{modality}' is not available"))
    for modality in ("rgb", "confidence"):
        if capture.modalities.get(modality, {}).get("available") is not True:
            issues.append(Issue("warning", "MISSING_PREFERRED_MODALITY",
                                f"preferred modality '{modality}' is unavailable; "
                                f"downstream evidence and filtering are degraded"))

    if not capture.frames:
        issues.append(Issue("error", "NO_FRAMES", "the capture contains no usable frames"))
        return issues

    # -- intrinsics -------------------------------------------------------
    streams = {i.stream: i for i in capture.intrinsics}
    for stream, intr in streams.items():
        if not (PLAUSIBLE_FOCAL_PX[0] <= intr.fx <= PLAUSIBLE_FOCAL_PX[1]
                and PLAUSIBLE_FOCAL_PX[0] <= intr.fy <= PLAUSIBLE_FOCAL_PX[1]):
            issues.append(Issue("error", "IMPLAUSIBLE_INTRINSICS",
                                f"stream '{stream}' has focal length "
                                f"fx={intr.fx:.3f} fy={intr.fy:.3f} px outside "
                                f"{PLAUSIBLE_FOCAL_PX}"))
        if not (0 < intr.cx < intr.width and 0 < intr.cy < intr.height):
            issues.append(Issue("error", "PRINCIPAL_POINT_OUTSIDE_IMAGE",
                                f"stream '{stream}' principal point "
                                f"({intr.cx:.3f}, {intr.cy:.3f}) lies outside "
                                f"{intr.width}x{intr.height}"))

    # -- per-frame records ------------------------------------------------
    seen_indices: set[int] = set()
    previous_ts = None
    for frame in capture.frames:
        label = f"Frame {frame.index}"
        if frame.index in seen_indices:
            issues.append(Issue("error", "DUPLICATE_FRAME_INDEX", f"{label} appears twice"))
        seen_indices.add(frame.index)

        depth_path = root / frame.depth
        if not depth_path.exists():
            issues.append(Issue("error", "MISSING_DEPTH_FILE",
                                f"{label} references depth '{frame.depth}' which does not exist"))
        if frame.rgb is not None and not (root / frame.rgb).exists():
            issues.append(Issue("error", "MISSING_RGB_FILE",
                                f"{label} references RGB '{frame.rgb}' which does not exist"))
        if frame.confidence is not None and not (root / frame.confidence).exists():
            issues.append(Issue("error", "MISSING_CONFIDENCE_FILE",
                                f"{label} references confidence '{frame.confidence}' "
                                f"which does not exist"))

        if frame.intrinsics_stream not in streams:
            issues.append(Issue("error", "UNKNOWN_INTRINSICS_STREAM",
                                f"{label} names intrinsics stream "
                                f"'{frame.intrinsics_stream}' with no entry in intrinsics.json"))

        matrix = np.array(frame.camera_to_world, dtype=np.float64)
        if matrix.shape != (4, 4):
            issues.append(Issue("error", "MALFORMED_POSE",
                                f"{label} pose has shape {matrix.shape}, expected (4, 4)"))
        else:
            problem = _check_rotation(matrix)
            if problem:
                issues.append(Issue("error", "INVALID_POSE", f"{label} {problem}"))

        if previous_ts is not None and frame.timestamp_s < previous_ts:
            issues.append(Issue("error", "NON_MONOTONIC_TIMESTAMP",
                                f"{label} timestamp {frame.timestamp_s:.6f} s precedes "
                                f"the previous frame's {previous_ts:.6f} s"))
        previous_ts = frame.timestamp_s

    # -- pixel-level consistency on a sample ------------------------------
    sample = capture.frames
    if len(sample) > depth_sample:
        step = len(sample) / depth_sample
        sample = [capture.frames[int(i * step)] for i in range(depth_sample)]

    depth_shapes: set[tuple[int, int]] = set()
    rgb_shapes: set[tuple[int, int]] = set()
    observed_min, observed_max = np.inf, 0.0
    for frame in sample:
        depth_path = root / frame.depth
        if not depth_path.exists():
            continue
        array = np.array(Image.open(depth_path))
        if array.dtype != np.uint16:
            issues.append(Issue("error", "WRONG_DEPTH_DTYPE",
                                f"Frame {frame.index} depth is {array.dtype}, expected uint16"))
            continue
        depth_shapes.add(array.shape[:2])
        valid = array[array > 0]
        if valid.size:
            observed_min = min(observed_min, valid.min() * capture.depth_scale_m)
            observed_max = max(observed_max, valid.max() * capture.depth_scale_m)
        if frame.rgb is not None and (root / frame.rgb).exists():
            rgb_shapes.add(np.array(Image.open(root / frame.rgb)).shape[:2])

    if len(depth_shapes) > 1:
        issues.append(Issue("error", "INCONSISTENT_DEPTH_RESOLUTION",
                            f"depth frames use multiple resolutions: {sorted(depth_shapes)}"))
    elif depth_shapes:
        height, width = depth_shapes.pop()
        depth_streams = [i for i in capture.intrinsics if i.stream.startswith("depth")]
        for intr in depth_streams:
            if (intr.width, intr.height) != (width, height):
                issues.append(Issue("error", "INTRINSICS_RESOLUTION_MISMATCH",
                                    f"depth images are {width}x{height} but intrinsics "
                                    f"stream '{intr.stream}' declares "
                                    f"{intr.width}x{intr.height}"))

    if rgb_shapes and len(rgb_shapes) > 1:
        issues.append(Issue("error", "INCONSISTENT_RGB_RESOLUTION",
                            f"RGB frames use multiple resolutions: {sorted(rgb_shapes)}"))
    elif rgb_shapes and depth_shapes is not None:
        rgb_hw = next(iter(rgb_shapes))
        if capture.modalities.get("rgb", {}).get("resolution") not in (None, list(rgb_hw[::-1])):
            issues.append(Issue("warning", "RGB_RESOLUTION_UNDECLARED",
                                f"RGB images are {rgb_hw[1]}x{rgb_hw[0]} but the manifest "
                                f"declares {capture.modalities['rgb'].get('resolution')}"))

    if observed_max > 0:
        if not (PLAUSIBLE_DEPTH_M[0] <= observed_min and observed_max <= PLAUSIBLE_DEPTH_M[1]):
            issues.append(Issue("error", "IMPLAUSIBLE_DEPTH_SCALE",
                                f"sampled depth spans {observed_min:.3f}-{observed_max:.3f} m, "
                                f"outside the indoor plausibility range {PLAUSIBLE_DEPTH_M} m; "
                                f"depth_scale_m={capture.depth_scale_m} is likely wrong"))
    else:
        issues.append(Issue("error", "NO_VALID_DEPTH",
                            "every sampled depth frame is entirely zero (no returns)"))

    # -- trajectory-level plausibility ------------------------------------
    centres = np.array([np.array(f.camera_to_world)[:3, 3] for f in capture.frames])
    if np.isfinite(centres).all():
        path = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum())
        if path > MAX_PLAUSIBLE_CAMERA_PATH_M:
            issues.append(Issue("error", "IMPLAUSIBLE_CAMERA_PATH",
                                f"camera path totals {path:.1f} m, beyond the "
                                f"{MAX_PLAUSIBLE_CAMERA_PATH_M} m plausibility bound; "
                                f"poses may be world-to-camera rather than camera-to-world"))
        elif path == 0.0 and len(capture.frames) > 1:
            issues.append(Issue("warning", "STATIONARY_CAMERA",
                                "every pose shares one camera centre; there is no parallax"))

    return issues


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    issues = validate(args.root)
    for issue in issues:
        print(issue)
    errors = [i for i in issues if i.severity == "error"]
    state = "NORMALIZED_CAPTURE_VALID" if not errors else "NORMALIZED_CAPTURE_INVALID"
    print(f"\n{state} ({len(errors)} errors, {len(issues) - len(errors)} warnings)")
    if args.json:
        args.json.write_text(json.dumps({
            "state": state,
            "issues": [{"severity": i.severity, "code": i.code, "message": i.message}
                       for i in issues],
        }, indent=2) + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
