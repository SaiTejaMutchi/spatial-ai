"""Freeze the AI value layer without restamping the geometry freeze.

This freeze emits `AI_VALUE_LAYER_FROZEN` only after visible-condition grounding has evidenced a real
model-generated region completing surface -> registration -> geometry quantity.
A lone `no_supported_condition` cannot satisfy the development freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_FREEZE = REPO_ROOT / "output" / "config_freeze_manifest.json"
CONDITION_ARTIFACT = REPO_ROOT / "output" / "visible_condition.json"
OUT_PATH = REPO_ROOT / "output" / "ai_value_layer_freeze.json"

PROMPT_FILES = [
    "prompts/spatial_verifier_v0.1.txt",
    "prompts/loss_proposal_v0.1.txt",
    "prompts/opening_resolver_v0.1.txt",
    "prompts/opening_image_only_v0.1.txt",
    "prompts/visible_condition_v0.1.txt",
]
SCHEMA_FILES = [
    "schema/ai_assessment.schema.json",
    "schema/opening_resolution.schema.json",
    "schema/visible_condition.schema.json",
]
AI_CONFIG = "config/ai_model_config.json"
GEOMETRY_CONFIG = "config/geometry_config_v0.1.json"
SEMANTIC_FIXTURE = REPO_ROOT / "samples" / "ai_semantic_eval"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visible_condition_chain_complete() -> tuple[bool, str]:
    if not CONDITION_ARTIFACT.is_file():
        return False, "output/visible_condition.json is missing"
    artifact = json.loads(CONDITION_ARTIFACT.read_text())
    proposal = artifact.get("proposal") or {}
    registration = proposal.get("registration") or {}
    quantity = proposal.get("quantity") or {}
    if artifact.get("modelResponse", {}).get("status") in {
            "no_supported_condition", "insufficient_evidence"} and not proposal:
        return False, "a lone abstention cannot freeze the development AI value layer"
    if registration.get("status") != "registered":
        return False, "visible-condition region did not register"
    if quantity.get("producer") != "geometry_pipeline":
        return False, "visible-condition quantity is not geometry-owned"
    if not quantity.get("affectedArea_m2"):
        return False, "visible-condition step emitted no geometry-owned area"
    return True, (
        f"model region registered to {proposal.get('surfaceId')} with "
        f"{quantity['affectedArea_m2']} m² producer=geometry_pipeline"
    )


def _semantic_fixture_record() -> dict:
    if not (SEMANTIC_FIXTURE / "manifest.json").is_file():
        return {"label": "missing", "note": "semantic evaluation fixture not present"}
    manifest = json.loads((SEMANTIC_FIXTURE / "manifest.json").read_text())
    files = {}
    for path in sorted(SEMANTIC_FIXTURE.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(REPO_ROOT))] = sha256(path)
    return {
        "label": manifest.get("label"),
        "notOfficialStructured3D": True,
        "caseIds": [c["id"] for c in manifest.get("cases", [])],
        "fileHashes": files,
    }


def build_manifest(freeze: bool = True) -> dict:
    geometry_hash = sha256(REPO_ROOT / GEOMETRY_CONFIG)
    recorded = json.loads(GEOMETRY_FREEZE.read_text()) if GEOMETRY_FREEZE.is_file() else {}
    frozen_geometry = (recorded.get("configurations") or {}).get(GEOMETRY_CONFIG)
    chain_ok, chain_detail = _visible_condition_chain_complete()
    live_prompt = None
    if CONDITION_ARTIFACT.is_file():
        live_prompt = (json.loads(CONDITION_ARTIFACT.read_text())
                       .get("diagnostics", {}).get("promptSha256"))
    geometry_untouched = frozen_geometry == geometry_hash
    ai = json.loads((REPO_ROOT / AI_CONFIG).read_text())

    passed = chain_ok and geometry_untouched and bool(ai.get("AI_MODEL_APPROVED"))
    state = "AI_VALUE_LAYER_FROZEN" if (freeze and passed) else "NOT_FROZEN"

    return {
        "manifestVersion": "0.1",
        "state": state,
        "generatedUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statement": (
            "The AI value layer is frozen separately from geometry. Geometry "
            "config remains PARAMETERS_FROZEN. This freeze does not reopen "
            "thresholds, does not claim VLM superiority, and does not complete "
            "final iPhone verification."
        ),
        "provider": ai.get("provider"),
        "model": ai.get("model"),
        "imageLimit": ai.get("imageLimit", 3),
        "imagesPerRequest": ai.get("imagesPerRequest"),
        "fallbackBehaviour": ai.get("fallbackBehaviour"),
        "onAccessFailure": ai.get("onAccessFailure"),
        "aiConfigHash": sha256(REPO_ROOT / AI_CONFIG),
        "promptHashes": {
            relative: sha256(REPO_ROOT / relative)
            for relative in PROMPT_FILES if (REPO_ROOT / relative).is_file()
        },
        "schemaHashes": {
            relative: sha256(REPO_ROOT / relative)
            for relative in SCHEMA_FILES if (REPO_ROOT / relative).is_file()
        },
        "geometryConfigHash": geometry_hash,
        "geometryConfigHashAtT15": frozen_geometry,
        "geometryUntouchedByExtension": geometry_untouched,
        "t14bChain": {"passed": chain_ok, "detail": chain_detail,
                      "livePromptSha256": live_prompt},
        "structured3dFixture": _semantic_fixture_record(),
        "notProvenByThisFreeze": [
            "real iPhone / Stray capture compatibility",
            "measurement accuracy or tape benchmark",
            "generalization beyond the two public scenes and tiny synthetic fixtures",
            "calibrated AI performance",
            "that spatial grounding makes the VLM intrinsically better at classification",
            "Track B damage intelligence or Track C restoration scope",
        ],
        "allChecksPassed": passed,
    }


def emit(path: Path | None = None, freeze: bool = True) -> dict:
    manifest = build_manifest(freeze=freeze)
    if freeze and manifest["state"] != "AI_VALUE_LAYER_FROZEN":
        raise SystemExit(
            "refusing AI_VALUE_LAYER_FROZEN: "
            + json.dumps({
                "t14b": manifest["t14bChain"],
                "geometryUntouched": manifest["geometryUntouchedByExtension"],
            })
        )
    path = Path(path or OUT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="AI value layer freeze")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    manifest = emit(args.out, freeze=True)
    print(json.dumps({
        "state": manifest["state"],
        "geometryUntouchedByExtension": manifest["geometryUntouchedByExtension"],
        "geometryConfigHash": manifest["geometryConfigHash"],
        "t14b": manifest["t14bChain"],
        "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
