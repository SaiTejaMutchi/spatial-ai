"""Source-neutral geometry pipeline: normalized_capture in, spatial_model.json out.

One deterministic entry point that the local API and the CLI both call,
so there is exactly one geometry path and no chance of the UI and the command
line disagreeing about what the room measures.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.normalized_capture import NormalizedCapture
from ..contracts.validate import validate
from .confidence import ConfidenceRules, load_confidence_rules
from .config import GeometryConfig, load_geometry_config
from .envelope import RoomEnvelope, build_envelope
from .model import ModelBuildResult, build_spatial_model, write_model
from .openings import detect_openings
from .planes import PlaneSet, extract_planes
from .points import PointCloud, build_point_cloud
from ..rendering.floorplan import render_floorplan
from ..rendering.model_3d import write_model_3d
from ..evidence.select import select_evidence_views
from ..ai.verifier import load_ai_config, run_verifier
from ..loss_preview.preview import build_loss_preview


@dataclass
class GeometryResult:
    cloud: PointCloud
    planes: PlaneSet
    envelope: RoomEnvelope
    model: dict
    diagnostics: dict[str, Any]
    capture: NormalizedCapture | None = None
    planes_by_id: dict | None = None


def run_geometry(
    capture_root: Path,
    config: GeometryConfig | None = None,
    rules: ConfidenceRules | None = None,
    frame_stride: int = 4,
    max_frames: int | None = None,
) -> GeometryResult:
    capture_root = Path(capture_root)
    config = config or load_geometry_config()
    rules = rules or load_confidence_rules()

    capture = NormalizedCapture.read(capture_root)
    cloud = build_point_cloud(capture_root, capture, config,
                              stride=frame_stride, max_frames=max_frames)
    planes = extract_planes(cloud, config)

    if planes.floor is None:
        raise ValueError(
            "GEOMETRY_GENERALIZATION_FAILURE: no floor plane could be fitted, so no "
            "room envelope or measurement can be derived from this capture")

    inlier_distance = config.get("plane_inlier_distance_m")
    on_floor = np.abs(
        cloud.points @ planes.floor.normal - planes.floor.offset) <= inlier_distance
    envelope = build_envelope(planes, cloud.points[on_floor], config)

    # Surfaces must exist before openings can be attached to them, so the model
    # is built once to establish stable surface IDs, openings are searched
    # against those surfaces, and the model is rebuilt carrying them.
    provisional = build_spatial_model(
        capture=capture, cloud=cloud, envelope=envelope, config=config, rules=rules)
    # Floor and ceiling are evidence targets too, not only walls; omitting them
    # would leave the verifier unable to be asked about the surfaces that
    # actually determine room height.
    planes_by_id = {w.plane_id: w for w in envelope.selected_walls}
    for structural in (planes.floor, planes.ceiling):
        if structural is not None:
            planes_by_id[structural.plane_id] = structural
    found = detect_openings(
        provisional.model["surfaces"], planes_by_id, cloud.points, config)

    built: ModelBuildResult = build_spatial_model(
        capture=capture, cloud=cloud, envelope=envelope,
        config=config, rules=rules,
        openings=found.records, opening_diagnostics=found.diagnostics)

    diagnostics = {
        "pointCloud": cloud.diagnostics,
        "planes": planes.diagnostics,
        "envelope": envelope.diagnostics,
        "openings": found.diagnostics,
        "model": built.diagnostics,
    }
    return GeometryResult(cloud=cloud, planes=planes, envelope=envelope,
                          model=built.model, diagnostics=diagnostics,
                          capture=capture, planes_by_id=planes_by_id)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="a normalized_capture directory")
    parser.add_argument("output", type=Path, help="directory for generated artifacts")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)

    issues = validate(args.capture)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        for issue in errors:
            print(issue, file=sys.stderr)
        print("\nCONNECTOR_FAILURE: geometry refuses to run on an invalid capture; "
              "NORMALIZED_CAPTURE_VALID was not reached", file=sys.stderr)
        return 2

    try:
        result = run_geometry(args.capture, frame_stride=args.frame_stride,
                              max_frames=args.max_frames)
    except ValueError as exc:
        print(f"GEOMETRY_GENERALIZATION_FAILURE: {exc}", file=sys.stderr)
        return 3

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "spatial_model.json"
    digest = write_model(result.model, model_path)
    (args.output / "geometry_diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2) + "\n")

    # Both visual products are generated from the model document that was just
    # written, never from a parallel geometry path, so they cannot disagree.
    # Evidence views are selected against the finished model, then written back
    # into it, so the document that ships carries its own provenance.
    views, evidence_diagnostics = select_evidence_views(
        capture_root=args.capture, capture=result.capture, cloud=result.cloud,
        model=result.model, planes_by_id=result.planes_by_id,
        config=load_geometry_config(), output_dir=args.output)
    result.model["evidence"] = [v.to_record() for v in views]
    result.model["provenance"]["evidenceSelection"] = evidence_diagnostics
    result.diagnostics["evidence"] = evidence_diagnostics
    from ..ai.opening_pipeline import resolve_scan_openings
    opening_report = resolve_scan_openings(
        result.model, args.output,
        capture=result.capture, capture_root=args.capture,
        cloud=result.cloud, planes_by_id=result.planes_by_id,
        config=load_geometry_config())
    result.model["provenance"]["openingResolution"] = opening_report["diagnostics"]
    result.diagnostics["openingResolution"] = opening_report
    digest = write_model(result.model, model_path)

    (args.output / "floorplan.svg").write_text(render_floorplan(result.model))
    entity_map = write_model_3d(result.model, args.output)
    # AI review runs last and is never on the critical path: geometry and every
    # artifact above are already written by this point.
    verifier = run_verifier(result.model, args.output / "evidence")
    result.model["aiAssessments"] = [verifier.assessment]
    result.model["provenance"]["aiReview"] = verifier.diagnostics
    result.diagnostics["aiReview"] = verifier.diagnostics
    digest = write_model(result.model, model_path)
    (args.output / "ai_assessment.json").write_text(
        json.dumps({"assessment": verifier.assessment,
                    "rejectedFindings": verifier.rejected_findings,
                    "diagnostics": verifier.diagnostics}, indent=2) + "\n")

    preview = build_loss_preview(result.model, args.output)
    result.diagnostics["lossPreview"] = {
        "status": preview["status"], "statusReason": preview["statusReason"]}

    (args.output / "evidence_manifest.json").write_text(json.dumps({
        "scanId": result.model["scan"]["id"],
        "classification": result.model["scan"]["classification"],
        "views": [v.to_record() for v in views],
        "diagnostics": evidence_diagnostics,
    }, indent=2) + "\n")

    measurements = {m["type"]: m for m in result.model["measurements"]}
    print(f"spatial_model.json -> {model_path}")
    print(f"  sha256           {digest}")
    print(f"  surfaces         {len(result.model['surfaces'])} "
          f"({result.diagnostics['model']['inferredSurfaces']} inferred)")
    print(f"  confidence       {result.diagnostics['model']['confidenceHistogram']}")
    for kind in ("room_length", "room_width", "room_height", "floor_area"):
        entry = measurements.get(kind)
        if entry is None:
            continue
        value = entry["value_m"]
        shown = "unresolved" if value is None else f"{value:.3f} {entry['unit']}"
        print(f"  {kind:16s} {shown:>16s}   {entry['confidence']['label']}")
    print(f"  observed perimeter {result.envelope.diagnostics['observedPerimeterFraction']:.0%}")
    openings = result.model["openings"]
    resolved = [o for o in openings if o["observationState"] != "unresolved"]
    print(f"  openings         {len(resolved)} resolved, "
          f"{len(openings) - len(resolved)} unresolved")
    print(f"  evidence         {len(views)} registered views covering "
          f"{len(evidence_diagnostics.get('surfacesCovered', []))} surfaces")
    assessment = verifier.assessment
    print(f"  ai review        {assessment['status']}"
          + (f" — {assessment['notRunReason'][:60]}..."
             if assessment.get("notRunReason") else
             f" — {len(assessment['findings'])} findings"))
    print(f"  loss preview     {preview['status']}")
    print(f"floorplan.svg      -> {args.output / 'floorplan.svg'}")
    print(f"room_model.obj     -> {args.output / 'room_model.obj'} "
          f"({entity_map['surfaceCount']} surfaces, {entity_map['vertexCount']} vertices)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
