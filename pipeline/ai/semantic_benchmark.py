"""Image-only versus spatial-grounded opening evaluation.

Success is not defined as higher raw classification accuracy. The benchmark
asks whether spatial grounding binds an interpretation to a candidate and
allows geometry-owned quantification.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .make_semantic_fixture import write_fixture
from .opening_resolver import PROMOTABLE, resolve_candidate
from .verifier import GroqVerifierClient, load_ai_config, protected_geometry_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "samples" / "ai_semantic_eval"
IMAGE_ONLY_PROMPT = REPO_ROOT / "prompts" / "opening_image_only_v0.1.txt"
CONCLUSION = (
    "Image-only AI may recognize the same visual object or condition. Spatial grounding "
    "does not claim to make the VLM intrinsically smarter. Its value is that the "
    "interpretation is bound to a stable physical entity and can be deterministically "
    "registered, measured, traced, and audited."
)


def _load_cases(root: Path) -> list[dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    cases = []
    for entry in manifest["cases"]:
        folder = root / entry["id"]
        cases.append({
            "id": entry["id"],
            "folder": folder,
            "candidate": json.loads((folder / "candidate.json").read_text()),
            "labels": json.loads((folder / "labels.json").read_text()),
            "crop": (folder / "crop.png").read_bytes(),
            "context": (folder / "context.png").read_bytes(),
        })
    return cases


def run_benchmark(root: Path | None = None, client: GroqVerifierClient | None = None,
                  spatial_model: dict | None = None) -> dict:
    root = Path(root or FIXTURE)
    if not (root / "manifest.json").is_file():
        write_fixture(root)
    config = load_ai_config()
    client = client or GroqVerifierClient()
    image_only_prompt = IMAGE_ONLY_PROMPT.read_text()
    geometry_before = protected_geometry_digest(spatial_model) if spatial_model else None

    rows = []
    for case in _load_cases(root):
        labels = case["labels"]
        candidate = case["candidate"]
        # Image-only arm: full context, no candidate/surface IDs, no labels.
        try:
            image_only = client.complete_json(
                image_only_prompt,
                "Describe visible doors, windows, or openings in this photograph.",
                [("context", case["context"])],
                config.model or "",
            )
            image_only_error = None
        except Exception as exc:  # noqa: BLE001
            image_only = None
            image_only_error = f"{type(exc).__name__}: {exc}"

        spatial = resolve_candidate(
            candidate, case["crop"], config=config, client=client, evidence_id="crop")

        gt = labels["semanticClass"]
        spatial_class = spatial.resolution.get("semanticClass")
        spatial_correct = spatial_class == gt
        image_only_kinds = []
        if image_only and isinstance(image_only.get("visibleOpenings"), list):
            image_only_kinds = [item.get("kind") for item in image_only["visibleOpenings"]
                                if isinstance(item, dict)]
        promoted = spatial.promoted is not None
        unsupported_promotion = (
            promoted and gt not in PROMOTABLE)
        can_quantify = promoted and spatial.promoted.get("producer") == "geometry_pipeline"
        bound = (
            spatial.resolution.get("candidateId") == candidate["candidateId"]
            and spatial.resolution.get("surfaceId") == candidate["surfaceId"]
            and spatial.diagnostics.get("validationResult") == "accepted"
        )
        rows.append({
            "caseId": case["id"],
            "datasetLabel": "SYNTHETIC_RENDER_SEMANTIC_EVALUATION",
            "candidateId": candidate["candidateId"],
            "surfaceId": candidate["surfaceId"],
            "heldOutLabel": gt,
            "imageOnly": {
                "kinds": image_only_kinds,
                "error": image_only_error,
                "boundToCandidate": False,
                "producedMetricEntity": False,
                "raw": image_only,
            },
            "spatialGrounded": {
                "semanticClass": spatial_class,
                "evidenceStatus": spatial.resolution.get("evidenceStatus"),
                "reason": spatial.resolution.get("reason"),
                "semanticCorrect": spatial_correct,
                "boundToCandidate": bound,
                "promoted": promoted,
                "unsupportedPromotion": unsupported_promotion,
                "geometryOwnedQuantity": can_quantify,
                "promotedRecord": spatial.promoted,
                "diagnostics": spatial.diagnostics,
                "resolution": spatial.resolution,
            },
        })

    geometry_after = protected_geometry_digest(spatial_model) if spatial_model else None
    mutation_count = 0 if geometry_before == geometry_after else 1
    n = len(rows) or 1
    spatial_acc = sum(1 for r in rows if r["spatialGrounded"]["semanticCorrect"]) / n
    bound_rate = sum(1 for r in rows if r["spatialGrounded"]["boundToCandidate"]) / n
    quantify_rate = sum(1 for r in rows if r["spatialGrounded"]["geometryOwnedQuantity"]) / n
    unsupported = sum(1 for r in rows if r["spatialGrounded"]["unsupportedPromotion"])
    abstentions = sum(1 for r in rows
                      if r["spatialGrounded"]["semanticClass"] == "insufficient_evidence")

    report = {
        "label": "SYNTHETIC_RENDER_SEMANTIC_EVALUATION",
        "notOfficialStructured3D": True,
        "generatedUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.model,
        "provider": config.provider,
        "promptHashes": {
            "opening_resolver_v0.1": hashlib.sha256(
                (REPO_ROOT / "prompts/opening_resolver_v0.1.txt").read_bytes()).hexdigest(),
            "opening_image_only_v0.1": hashlib.sha256(
                IMAGE_ONLY_PROMPT.read_bytes()).hexdigest(),
        },
        "interpretationRule": (
            "Success is not defined as Spatial-Grounded achieving higher raw "
            "classification accuracy. Higher raw accuracy is neither assumed nor "
            "required."
        ),
        "conclusion": CONCLUSION,
        "geometryMutationCount": mutation_count,
        "summary": {
            "cases": len(rows),
            "spatialSemanticAccuracy": round(spatial_acc, 4),
            "spatialCandidateBindingRate": round(bound_rate, 4),
            "geometryOwnedQuantityRate": round(quantify_rate, 4),
            "unsupportedPromotionCount": unsupported,
            "insufficientEvidenceCount": abstentions,
            "imageOnlyMetricEntityCount": 0,
        },
        "cases": rows,
    }
    return report


def write_benchmark(report: dict, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = Path(output_dir or REPO_ROOT / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ai_semantic_benchmark.json"
    md_path = output_dir / "ai_semantic_benchmark.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    summary = report["summary"]
    lines = [
        "# AI semantic benchmark",
        "",
        f"**Label:** `{report['label']}`",
        "",
        "This is **not** official Structured3D. Official Structured3D requires a "
        "human-signed terms agreement and was not downloaded.",
        "",
        report["interpretationRule"],
        "",
        f"> {report['conclusion']}",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Cases | {summary['cases']} |",
        f"| Spatial semantic accuracy (tiny, not generalizable) | {summary['spatialSemanticAccuracy']} |",
        f"| Spatial candidate/surface binding rate | {summary['spatialCandidateBindingRate']} |",
        f"| Geometry-owned quantity rate | {summary['geometryOwnedQuantityRate']} |",
        f"| Unsupported promotion count | {summary['unsupportedPromotionCount']} |",
        f"| `insufficient_evidence` count | {summary['insufficientEvidenceCount']} |",
        f"| Image-only metric entity count | {summary['imageOnlyMetricEntityCount']} |",
        f"| Geometry mutation count | {report['geometryMutationCount']} |",
        "",
        "Image-only AI can visually describe or classify an object; spatial "
        "grounding binds that interpretation to a specific metric entity and "
        "enables auditable geometry-owned quantities.",
        "",
    ]
    for row in report["cases"]:
        sg = row["spatialGrounded"]
        lines.append(f"## {row['caseId']}")
        lines.append("")
        lines.append(f"- Held-out label (evaluator only): `{row['heldOutLabel']}`")
        lines.append(f"- Image-only kinds: `{row['imageOnly']['kinds']}` "
                     f"(boundToCandidate={row['imageOnly']['boundToCandidate']})")
        lines.append(f"- Spatial class: `{sg['semanticClass']}` "
                     f"status=`{sg['evidenceStatus']}` bound={sg['boundToCandidate']} "
                     f"promoted={sg['promoted']}")
        lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output")
    args = parser.parse_args()
    model_path = REPO_ROOT / "outputs" / "dev_47333462" / "spatial_model.json"
    model = json.loads(model_path.read_text()) if model_path.is_file() else None
    report = run_benchmark(spatial_model=model)
    paths = write_benchmark(report, args.output)
    print("wrote", paths[0])
    print("wrote", paths[1])
    print("geometryMutationCount", report["geometryMutationCount"])
    print("summary", json.dumps(report["summary"]))
