"""Self-captured real-world ground truth evaluation CLI harness."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

from spatial_ai import Space

REPO_ROOT = Path(__file__).resolve().parents[2]
SELF_CAPTURE_LOG = REPO_ROOT / "pipeline" / "eval" / "results" / "self_captured_runs.json"


def evaluate_self_capture(
    capture_dir: Path | str,
    laser_height_m: float | None = None,
    laser_width_m: float | None = None,
    laser_length_m: float | None = None,
    label: str = "Self-Captured Real Room",
) -> dict[str, Any]:
    """Processes a self-captured capture directory and compares predictions against manual handheld laser measurements."""
    capture_path = Path(capture_dir)
    space = Space.process(capture_path)

    dims = space.dimensions
    pred_height = dims.get("height_m")
    pred_width = dims.get("width_m")
    pred_length = dims.get("length_m")

    deltas = {}
    if laser_height_m is not None and pred_height is not None:
        deltas["height_error_cm"] = round(abs(pred_height - laser_height_m) * 100.0, 3)
    if laser_width_m is not None and pred_width is not None:
        deltas["width_error_cm"] = round(abs(pred_width - laser_width_m) * 100.0, 3)
    if laser_length_m is not None and pred_length is not None:
        deltas["length_error_cm"] = round(abs(pred_length - laser_length_m) * 100.0, 3)

    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": label,
        "capturePath": str(capture_path),
        "spaceId": space.id,
        "predictedDimensions": dims,
        "handheldLaserMeasurements": {
            "height_m": laser_height_m,
            "width_m": laser_width_m,
            "length_m": laser_length_m,
        },
        "errors": deltas,
    }

    # Append to log
    existing = []
    if SELF_CAPTURE_LOG.exists():
        try:
            with open(SELF_CAPTURE_LOG, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.append(record)
    SELF_CAPTURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SELF_CAPTURE_LOG, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a self-captured scan against manual handheld laser measurements.")
    parser.add_argument("--capture", required=True, help="Path to raw capture directory.")
    parser.add_argument("--laser-height", type=float, help="Manual handheld laser room height in meters.")
    parser.add_argument("--laser-width", type=float, help="Manual handheld laser room width in meters.")
    parser.add_argument("--laser-length", type=float, help="Manual handheld laser room length in meters.")
    parser.add_argument("--label", default="Self-Captured Real Room", help="Label/description for this capture.")
    args = parser.parse_args()

    res = evaluate_self_capture(
        capture_dir=args.capture,
        laser_height_m=args.laser_height,
        laser_width_m=args.laser_width,
        laser_length_m=args.laser_length,
        label=args.label,
    )
    print("=== SELF-CAPTURED GROUND TRUTH EVALUATION ===")
    print("Space ID:", res["spaceId"])
    print("Errors:", res["errors"])


if __name__ == "__main__":
    main()
