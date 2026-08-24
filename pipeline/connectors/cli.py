"""Deterministic command: one source directory in, one `normalized_capture` out."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..contracts.frame_resolution import (
    FrameResolutionError, apply_resolution, resolve_frame)
from ..contracts.ingestion_outcome import summarize
from ..contracts.validate import validate
from .base import ConnectorError
from .detect import CONNECTORS, detect_source


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="capture directory or bundle")
    parser.add_argument("destination", type=Path, help="normalized_capture output directory")
    parser.add_argument("--source-type", choices=sorted(CONNECTORS),
                        help="skip detection and force a connector")
    parser.add_argument("--classification", default="public_development_fixture",
                        choices=["public_development_fixture", "final_private_capture",
                                 "baseline_fallback"])
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth frame; recorded in the manifest")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--pose-tolerance-s", type=float, default=0.05)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--no-ai", action="store_true",
                        help="never request an AI diagnosis, even for an "
                             "ambiguous capture")
    parser.add_argument("--skip-frame-resolution", action="store_true",
                        help="normalize without verifying the vertical axis; "
                             "diagnostic use only")
    args = parser.parse_args(argv)

    try:
        connector_cls = (CONNECTORS[args.source_type] if args.source_type
                         else detect_source(args.source))
    except ConnectorError as exc:
        print(f"CONNECTOR_FAILURE: {exc}", file=sys.stderr)
        return 2

    kwargs = {"classification": args.classification,
              "pose_tolerance_s": args.pose_tolerance_s,
              "stride": args.stride,
              "max_frames": args.max_frames}
    supported = connector_cls.accepted_options()
    connector = connector_cls(args.source, **{k: v for k, v in kwargs.items() if k in supported})

    print(f"source_detected: {connector_cls.source_type}")
    try:
        capture = connector.normalize(args.destination)
    except ConnectorError as exc:
        print(f"CONNECTOR_FAILURE: {exc}", file=sys.stderr)
        return 2

    print(f"normalized_capture: {len(capture.frames)} frames, "
          f"{len(capture.excluded_frames)} excluded -> {args.destination}")

    # Frame resolution sits here on purpose: after the vendor layout has been
    # normalized, before the capture is declared valid. A declared gravity axis
    # is verified against observed structure rather than trusted or overridden
    # downstream by a heuristic that cannot see the room.
    resolution = None
    if not args.skip_frame_resolution:
        try:
            resolution = resolve_frame(args.destination, capture)
        except FrameResolutionError as exc:
            print(f"frame_resolution: unavailable ({exc})")
        else:
            apply_resolution(args.destination, capture, resolution)
            print(f"frame_resolution: {resolution.outcome}"
                  + (f" -> {resolution.axis} ({resolution.basis})" if resolution.axis else ""))
            for note in resolution.notes:
                print(f"  {note}")

    # An ambiguous frame is where a bounded second opinion earns its place. It is
    # advisory: the outcome below does not change because of what it says.
    diagnosis = None
    if resolution is not None and resolution.outcome != "verified" and not args.no_ai:
        from ..ai.frame_diagnosis import diagnose_frame
        diagnosis = diagnose_frame(resolution, connector_cls.source_type)
        if diagnosis["status"] == "completed":
            body = diagnosis["diagnosis"]
            print(f"frame_diagnosis (advisory): {body['assessment']} "
                  f"[{body['confidence']}] -> {body['recommendation']}")

    issues = validate(args.destination)
    for issue in issues:
        print(issue)
    errors = [i for i in issues if i.severity == "error"]
    state = "NORMALIZED_CAPTURE_VALID" if not errors else "NORMALIZED_CAPTURE_INVALID"
    print(f"\n{state} ({len(errors)} errors, {len(issues) - len(errors)} warnings)")

    outcome = summarize(issues=issues, resolution=resolution,
                        excluded_frames=len(capture.excluded_frames))
    print(f"\n{outcome.headline}")
    print(f"  {outcome.detail}")
    for fix in outcome.fixes:
        print(f"  fixed: {fix}")
    for concern in outcome.concerns:
        print(f"  review: {concern}")

    if args.report:
        args.report.write_text(json.dumps({
            "source_type": connector_cls.source_type,
            "source": str(args.source),
            "destination": str(args.destination),
            "state": state,
            "frames": len(capture.frames),
            "excluded_frames": len(capture.excluded_frames),
            "issues": [{"severity": i.severity, "code": i.code, "message": i.message}
                       for i in issues],
            "frame_resolution": resolution.to_record() if resolution else None,
            "frame_diagnosis": diagnosis,
            "outcome": outcome.to_record(),
        }, indent=2) + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
