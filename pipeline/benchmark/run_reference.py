"""Emit the PUBLIC DEVELOPMENT REFERENCE comparison.

This is a sanity check against independent survey data, not the final
validated benchmark. The two are kept apart on purpose, so every line this
writes says which it is.

What can be compared is limited by what ARKitScenes publishes. There is no
transform between the ARKit session frame and the FARO visit frame, so
horizontal quantities cannot be compared without solving registration first.
Floor-to-ceiling height can, because both frames are gravity-aligned and that
measurement is invariant under any horizontal rigid motion. Everything else is
reported as not comparable, with the reason, rather than omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..geometry.config import load_geometry_config
from .compare import Comparison, compare, summarise
from .reference import ReferenceError, extract_reference_geometry, load_reference_config

LABEL = "PUBLIC DEVELOPMENT REFERENCE"

NOT_COMPARABLE_REASON = (
    "ARKitScenes publishes no transform between the ARKit session frame and the "
    "FARO visit frame, so this quantity cannot be compared without first solving "
    "registration. Only floor-to-ceiling height survives that gap, because it is "
    "invariant under any horizontal rigid motion between two gravity-aligned frames."
)


def build_report(
    model: dict,
    visit_dir: Path,
    scene_id: str,
    scene_role: str,
    secondary_models: dict[str, dict] | None = None,
) -> dict:
    geometry_config = load_geometry_config()
    reference_config = load_reference_config()

    measurements = {m["type"]: m for m in model["measurements"]}
    comparisons: list[Comparison] = []
    reference_error: str | None = None
    reference_diagnostics: dict = {}

    try:
        reference = extract_reference_geometry(visit_dir, reference_config)
        reference_diagnostics = reference.diagnostics
        model_height = measurements.get("room_height", {}).get("value_m")
        comparisons.append(compare(
            "reference-room-height", "room_height",
            reference_m=round(reference.separation_m, 6),
            model_m=model_height,
            note=("Floor-to-ceiling separation. Both frames are gravity-aligned, so "
                  "this comparison needs no horizontal registration."),
            extra={
                "referenceSource": "ARKitScenes FARO terrestrial laser scans",
                "referenceScans": reference.scans_used,
                "referencePointsSampled": reference.point_count,
                "referenceFloor_m": round(reference.floor_height_m, 6),
                "referenceCeiling_m": round(reference.ceiling_height_m, 6),
                "maxScanTiltDeg": reference.diagnostics.get("maxScanTiltDeg"),
            },
        ))
    except ReferenceError as exc:
        reference_error = str(exc)

    for kind in ("room_length", "room_width", "floor_area"):
        entry = measurements.get(kind)
        comparisons.append(compare(
            f"reference-{kind.replace('_', '-')}", kind,
            reference_m=None,
            model_m=entry["value_m"] if entry else None,
            note=NOT_COMPARABLE_REASON,
        ))

    report = {
        "label": LABEL,
        "isFinalBenchmark": False,
        "statement": (
            "This is an independent public sanity check against survey-grade laser "
            "data. It is not a tape-measure benchmark, and it "
            "does not establish accuracy on any real capture."),
        "generatedUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scene": {"id": scene_id, "role": scene_role,
                  "classification": model["scan"]["classification"]},
        "referenceSource": {
            "dataset": "Apple ARKitScenes laser_scanner_point_clouds",
            "visitDirectory": str(visit_dir),
            "independentOfDeviceUnderTest": True,
            "whyNotTheArkitMesh": (
                "The ARKit mesh is produced by the same reconstruction stack being "
                "evaluated, so agreement with it would not evidence accuracy."),
            "extractionProcedure": "docs/fixture_selection.md section 5",
            "extractionConfigId": reference_config.config_id,
            "extractionConfigHash": reference_config.sha256,
        },
        "frozenParameterSet": {
            "geometryConfigId": geometry_config.config_id,
            "geometryConfigHash": geometry_config.sha256,
            "frozen": geometry_config.frozen,
        },
        "comparisons": [c.to_row() for c in comparisons],
        "summary": summarise(comparisons),
        "referenceDiagnostics": reference_diagnostics,
        "limitations": [
            "Only floor-to-ceiling height is comparable; see the note on each "
            "non-comparable row.",
            "The FARO reference covers the whole visit, which may include rooms the "
            "60-second video never entered. Without registration it cannot be "
            "confirmed that the compared storey is the same room the pipeline "
            "reconstructed, so this is a dwelling storey-height check rather than a "
            "same-room comparison.",
            "One scene with one comparable quantity supports no generalization claim.",
            "Confidence labels elsewhere in the model remain uncalibrated heuristics; "
            "nothing here calibrates them.",
        ],
    }
    spread_cm = reference_diagnostics.get("separationSpread_cm")
    if spread_cm:
        report["referenceCorrespondence"] = {
            "established": False,
            "plausiblePairings": reference_diagnostics.get("plausiblePairings", []),
            "separationRange_m": reference_diagnostics.get("separationRange_m"),
            "separationSpread_cm": spread_cm,
            "finding": (
                f"The reference visit supports "
                f"{len(reference_diagnostics.get('plausiblePairings', []))} plausible "
                f"floor-to-ceiling pairings spanning {spread_cm:.1f} cm. Its ceiling "
                f"surfaces extend well beyond the footprint the pipeline "
                f"reconstructed, which is what a whole-dwelling survey looks like. "
                f"The reference is therefore a building storey statistic, not a "
                f"measurement of the captured room."),
            "consequence": (
                "Because the reference's own ambiguity is wider than the discrepancy, "
                "it cannot adjudicate that discrepancy, and no geometry parameter was "
                "tuned toward it. The comparison is retained as a magnitude check: it "
                "shows the pipeline lands within the range of storey heights present "
                "in the building, and nothing stronger."),
        }

    if reference_error:
        report["referenceUnavailable"] = reference_error
        report["limitations"].insert(
            0, f"No reference geometry could be extracted: {reference_error}")

    if secondary_models:
        report["secondaryValidation"] = {
            "policy": ("The secondary scene runs once with the same frozen "
                       "configuration and is never used to change a parameter."),
            "scenes": {
                scene: {
                    "roomHeight_m": {m["type"]: m["value_m"]
                                     for m in document["measurements"]}.get("room_height"),
                    "floorArea_m2": {m["type"]: m["value_m"]
                                     for m in document["measurements"]}.get("floor_area"),
                    "geometryConfigHash": document["provenance"]["geometryConfigHash"],
                    "referenceAvailable": False,
                    "note": ("No FARO reference was retrieved for this visit, by "
                             "design: the plan caps public validation at one "
                             "supported quantitative comparison. This scene "
                             "evidences that the frozen configuration runs unchanged "
                             "on unseen data, and nothing more."),
                }
                for scene, document in secondary_models.items()
            },
        }
    return report


def render_markdown(report: dict) -> str:
    lines = [
        f"# {report['label']}",
        "",
        f"> {report['statement']}",
        "",
        f"- **Scene:** `{report['scene']['id']}` ({report['scene']['role']})",
        f"- **Classification:** `{report['scene']['classification']}`",
        f"- **Reference:** {report['referenceSource']['dataset']}",
        f"- **Geometry config:** `{report['frozenParameterSet']['geometryConfigId']}` "
        f"(`{report['frozenParameterSet']['geometryConfigHash'][:16]}…`, "
        f"frozen={report['frozenParameterSet']['frozen']})",
        f"- **Reference config:** `{report['referenceSource']['extractionConfigId']}` "
        f"(`{report['referenceSource']['extractionConfigHash'][:16]}…`)",
        f"- **Generated:** {report['generatedUtc']}",
        "",
        "## Comparison",
        "",
        "| Measurement | Reference | Model | Signed error | Abs error | % error | "
        "Assignment gate | Result |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["comparisons"]:
        def show(value, suffix="", digits=3):
            return "—" if value is None else f"{value:.{digits}f}{suffix}"
        unit = " m²" if row["type"] == "floor_area" else " m"
        lines.append(
            f"| {row['type']} | {show(row['reference_m'], unit)} | "
            f"{show(row['model_m'], unit)} | {show(row['signedError_m'], unit)} | "
            f"{show(row['absoluteError_cm'], ' cm', 2)} | "
            f"{show(row['percentError'], '%', 2)} | {row['assignmentGate'] or '—'} | "
            f"**{row['result']}** |")

    summary = report["summary"]
    lines += [
        "",
        f"{summary['comparable']} of {summary['comparisons']} quantities were "
        f"comparable; {summary['passed']} passed and {summary['failed']} failed the "
        f"assignment gate.",
        "",
        "## Why most quantities are not comparable",
        "",
        NOT_COMPARABLE_REASON,
        "",
    ]

    correspondence = report.get("referenceCorrespondence")
    if correspondence:
        lines += [
            "## Does the reference describe the same room?",
            "",
            f"**No — correspondence could not be established.** "
            f"{correspondence['finding']}",
            "",
            "| Reference floor | Reference ceiling | Separation | Min support |",
            "|---:|---:|---:|---:|",
        ]
        for pairing in correspondence["plausiblePairings"]:
            lines.append(
                f"| {pairing['floor_m']:.4f} m | {pairing['ceiling_m']:.4f} m | "
                f"**{pairing['separation_m']:.4f} m** | {pairing['minSupport']:,} |")
        lines += ["", correspondence["consequence"], ""]

    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]

    if "secondaryValidation" in report:
        lines += ["", "## Secondary validation scene", "",
                  report["secondaryValidation"]["policy"], ""]
        for scene, entry in report["secondaryValidation"]["scenes"].items():
            height = entry["roomHeight_m"]
            area = entry["floorArea_m2"]
            lines.append(
                f"- `{scene}` — room height "
                f"{'unresolved' if height is None else f'{height:.3f} m'}, floor area "
                f"{'unresolved' if area is None else f'{area:.3f} m²'}, config "
                f"`{entry['geometryConfigHash'][:16]}…`. {entry['note']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("visit_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-role", default="PRIMARY_TUNING")
    parser.add_argument("--secondary", nargs="*", default=[],
                        help="scene_id=path/to/spatial_model.json")
    args = parser.parse_args(argv)

    secondary = {}
    for item in args.secondary:
        scene, _, path = item.partition("=")
        secondary[scene] = json.loads(Path(path).read_text())

    report = build_report(
        json.loads(args.model.read_text()), args.visit_dir,
        args.scene_id, args.scene_role, secondary or None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "development_reference_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (args.output_dir / "development_reference_benchmark.md").write_text(
        render_markdown(report))
    print(f"{LABEL} -> {args.output_dir}")
    for row in report["comparisons"]:
        if row["absoluteError_cm"] is not None:
            print(f"  {row['type']:14s} model {row['model_m']:.4f} vs reference "
                  f"{row['reference_m']:.4f}  error {row['signedError_m']*100:+.2f} cm "
                  f"({row['percentError']:.2f}%)  {row['result']}")
        else:
            print(f"  {row['type']:14s} not comparable")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
