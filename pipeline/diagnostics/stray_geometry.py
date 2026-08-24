"""Compare the active canonical frame with the source-declared Stray +Y-up frame.

This utility writes evidence only. It never mutates normalized input, geometry
configuration, or production reconstruction artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.contracts.normalized_capture import NormalizedCapture
from pipeline.geometry.config import load_geometry_config
from pipeline.geometry.frame import CanonicalFrame
from pipeline.geometry.planes import extract_planes
from pipeline.geometry.points import PointCloud, build_point_cloud


def _plane_summary(planes) -> dict:
    horizontal = planes.diagnostics.get("horizontal", {})
    return {
        "floor": planes.floor.to_record() if planes.floor else None,
        "ceiling": planes.ceiling.to_record() if planes.ceiling else None,
        "wallCount": len(planes.walls),
        "horizontal": horizontal,
        "rejectedHorizontal": [r for r in planes.rejected
                               if r.get("kind") == "horizontal"],
    }


def _write_ply(path: Path, points: np.ndarray, limit: int = 200_000) -> None:
    step = max(int(np.ceil(len(points) / limit)), 1)
    sampled = points[::step]
    with path.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(sampled)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(handle, sampled, fmt="%.6f %.6f %.6f")


def _plot(path: Path, current: np.ndarray, declared: np.ndarray) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1400, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), "Real Stray sample 8653a2142b - frame diagnostic", fill="#111827")

    def panel(points: np.ndarray, axes: tuple[int, int], box, title: str) -> None:
        left, top, right, bottom = box
        draw.rectangle(box, outline="#94a3b8", width=1)
        draw.text((left + 8, top + 7), title, fill="#111827")
        sample = points[::max(int(np.ceil(len(points) / 55_000)), 1)]
        x = sample[:, axes[0]]
        y = sample[:, axes[1]]
        x_min, x_max = np.quantile(x, [0.002, 0.998])
        y_min, y_max = np.quantile(y, [0.002, 0.998])
        px = left + 18 + (x - x_min) / max(x_max - x_min, 1e-9) * (right - left - 36)
        py = bottom - 18 - (y - y_min) / max(y_max - y_min, 1e-9) * (bottom - top - 52)
        height = sample[:, 1]
        h_min, h_max = np.quantile(height, [0.01, 0.99])
        hue = np.clip((height - h_min) / max(h_max - h_min, 1e-9), 0, 1)
        for xx, yy, value in zip(px.astype(int), py.astype(int), hue):
            color = (int(30 + 40 * value), int(80 + 130 * value), int(150 - 80 * value))
            draw.point((int(xx), int(yy)), fill=color)
        draw.text((left + 8, bottom - 16),
                  f"range X {x_min:.2f}..{x_max:.2f} m; vertical {y_min:.2f}..{y_max:.2f} m",
                  fill="#475569")

    panel(current, (0, 1), (25, 55, 690, 500), "Current canonical - elevation X/Y")
    panel(current, (0, 2), (710, 55, 1375, 500), "Current canonical - plan X/Z")
    panel(declared, (0, 1), (25, 530, 690, 975), "Declared Stray +Y up - elevation X/Y")
    panel(declared, (0, 2), (710, 530, 1375, 975), "Declared Stray +Y up - plan X/Z")
    image.save(path)


def run(normalized: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    capture = NormalizedCapture.read(normalized)
    config = load_geometry_config()
    current = build_point_cloud(normalized, capture, config)

    rotation = current.frame.source_to_canonical[:3, :3]
    source_points = current.points @ rotation
    declared_frame = CanonicalFrame(
        source_to_canonical=np.eye(4), source_up_axis=np.array([0.0, 1.0, 0.0]),
        up_source="declared_diagnostic", world_frame=capture.world_frame,
        diagnostics={"diagnosticOnly": True})
    declared = PointCloud(
        points=source_points, frame=declared_frame,
        frame_indices=current.frame_indices.copy(),
        diagnostics={"diagnosticOnly": True},
        config_provenance=current.config_provenance)

    current_planes = extract_planes(current, config)
    declared_planes = extract_planes(declared, config)
    _write_ply(output / "current_canonical_debug.ply", current.points)
    _write_ply(output / "declared_y_up_debug.ply", declared.points)
    _plot(output / "frame_comparison.png", current.points, declared.points)

    centres = np.array([np.array(frame.camera_to_world)[:3, 3]
                        for frame in capture.frames])
    result = {
        "sample": capture.provenance.source_id,
        "diagnosticOnly": True,
        "geometryConfigHash": config.sha256,
        "poseConvention": capture.pose_convention,
        "declaredWorldUp": capture.world_up_axis,
        "declaredWorldUpVerified": capture.world_up_axis_verified,
        "cameraCentreExtentSourceM": np.ptp(centres, axis=0).round(6).tolist(),
        "activeCanonicalFrame": current.frame.provenance(),
        "activeCanonicalBoundsM": {
            "min": current.points.min(0).round(6).tolist(),
            "max": current.points.max(0).round(6).tolist(),
            "extent": current.extent().round(6).tolist(),
        },
        "declaredYUpBoundsM": {
            "min": declared.points.min(0).round(6).tolist(),
            "max": declared.points.max(0).round(6).tolist(),
            "extent": declared.extent().round(6).tolist(),
        },
        "activePlaneResult": _plane_summary(current_planes),
        "declaredYUpPlaneResult": _plane_summary(declared_planes),
    }
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("normalized", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.normalized, args.output)
    print(json.dumps({
        "sample": result["sample"],
        "activeFloor": result["activePlaneResult"]["floor"] is not None,
        "declaredYUpFloor": result["declaredYUpPlaneResult"]["floor"] is not None,
        "artifacts": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
