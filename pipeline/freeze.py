"""Emit `PARAMETERS_FROZEN` and the configuration freeze manifest.

Freezing is what makes the final Stray run meaningful: the real capture must
traverse the same configuration that was fixed before it existed, so that any
error it reveals is the pipeline's and not a threshold chosen after seeing the
answer.

Freezing is refused unless the anti-overfitting checks pass, because a manifest
that records a hash while the code hides a scene-specific constant would be
worse than no manifest at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILES = [
    "config/geometry_config_v0.1.json",
    "config/confidence_rules_v0.1.json",
    "config/reference_extraction_v0.1.json",
    "config/ai_model_config.json",
    "config/ingestion_frame_config_v0.1.json",
    "schema/spatial_model.schema.json",
    "schema/ai_assessment.schema.json",
    "schema/frame_diagnosis.schema.json",
    "prompts/spatial_verifier_v0.1.txt",
    "prompts/loss_proposal_v0.1.txt",
    "prompts/frame_diagnosis_v0.1.txt",
]

CODE_MODULES = [
    "pipeline/contracts/normalized_capture.py",
    "pipeline/contracts/validate.py",
    "pipeline/contracts/validate_model.py",
    "pipeline/contracts/frame_resolution.py",
    "pipeline/contracts/ingestion_outcome.py",
    "pipeline/connectors/arkitscenes.py",
    "pipeline/connectors/stray_scanner.py",
    "pipeline/connectors/unity_obj.py",
    "pipeline/geometry/frame.py",
    "pipeline/geometry/points.py",
    "pipeline/geometry/planes.py",
    "pipeline/geometry/envelope.py",
    "pipeline/geometry/openings.py",
    "pipeline/geometry/model.py",
    "pipeline/geometry/confidence.py",
    "pipeline/rendering/floorplan.py",
    "pipeline/rendering/model_3d.py",
    "pipeline/benchmark/ground_truth.py",
    "pipeline/benchmark/compare.py",
    "pipeline/benchmark/reference.py",
    "pipeline/evidence/select.py",
    "pipeline/ai/verifier.py",
    "pipeline/ai/frame_diagnosis.py",
    "pipeline/loss_preview/preview.py",
]

# Scene identifiers that must never appear in processing code.
FIXTURE_SCENE_IDS = ("47333462", "41418135", "467138")

# Source-specific vocabulary that must not escape the connector boundary.
SOURCE_SPECIFIC_TOKENS = ("lowres_wide", "lowres_depth", "odometry.csv",
                          "camera_matrix.csv", "3dod_mesh", "rgb.mp4")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anti_overfitting_checks() -> list[dict]:
    """The checks the plan requires before anything may be frozen."""
    results: list[dict] = []

    # 1. No fixture scene ID anywhere in processing code.
    offenders = []
    for relative in CODE_MODULES:
        text = (REPO_ROOT / relative).read_text()
        for scene in FIXTURE_SCENE_IDS:
            if scene in text:
                offenders.append(f"{relative} names scene {scene}")
    results.append({
        "check": "no hardcoded scene identifiers in processing code",
        "passed": not offenders, "detail": offenders or "clean",
    })

    # 2. Source-specific paths exist only inside connectors.
    offenders = []
    for relative in CODE_MODULES:
        if relative.startswith("pipeline/connectors/"):
            continue
        text = (REPO_ROOT / relative).read_text()
        for token in SOURCE_SPECIFIC_TOKENS:
            if token in text:
                offenders.append(f"{relative} references '{token}'")
    results.append({
        "check": "source-specific paths appear only inside connectors",
        "passed": not offenders, "detail": offenders or "clean",
    })

    # 3. Every geometry parameter documents its provenance and is uncalibrated.
    config = json.loads((REPO_ROOT / "config/geometry_config_v0.1.json").read_text())
    offenders = [
        name for name, entry in config["parameters"].items()
        if not entry.get("rationale") or not entry.get("sourceCategory")
        or entry.get("calibrated") is not False
    ]
    results.append({
        "check": "every geometry parameter records rationale, source, and calibrated=false",
        "passed": not offenders, "detail": offenders or
        f"{len(config['parameters'])} parameters documented",
    })

    # 4. Nothing was tuned on the secondary validation scene.
    secondary = config["tuningPolicy"]["secondaryValidationScene"]
    offenders = [name for name, entry in config["parameters"].items()
                 if secondary in entry.get("tuningSceneIds", [])]
    results.append({
        "check": "no parameter was tuned on the secondary validation scene",
        "passed": not offenders,
        "detail": offenders or f"secondary scene {secondary} drove no parameter",
    })

    # 5. One configuration produced every generated model.
    hashes, models = {}, sorted((REPO_ROOT / "outputs").glob("dev_*/spatial_model.json"))
    for path in models:
        document = json.loads(path.read_text())
        hashes[path.parent.name] = document["provenance"]["geometryConfigHash"]
    results.append({
        "check": "all scenes ran on one geometry configuration",
        "passed": len(set(hashes.values())) <= 1,
        "detail": hashes or "no generated models found",
    })

    # 6. AI has not written into geometry, and production arrays stay empty.
    offenders = []
    for path in models:
        document = json.loads(path.read_text())
        if document["damage"] or document["scope"]:
            offenders.append(f"{path.parent.name} has non-empty damage[]/scope[]")
        for measurement in document["measurements"]:
            if measurement["producer"] != "geometry_pipeline":
                offenders.append(
                    f"{path.parent.name} measurement {measurement['id']} was produced "
                    f"by {measurement['producer']}")
    results.append({
        "check": "only geometry produced measurements; damage[] and scope[] are empty",
        "passed": not offenders, "detail": offenders or "clean",
    })

    # 7. No AI model was selected without operator approval.
    ai = json.loads((REPO_ROOT / "config/ai_model_config.json").read_text())
    approved = bool(ai.get("AI_MODEL_APPROVED"))
    selected = bool(ai.get("provider") or ai.get("model"))
    results.append({
        "check": "no provider or model was selected without operator approval",
        "passed": approved or not selected,
        "detail": f"AI_MODEL_APPROVED={approved}, provider={ai.get('provider')}, "
                  f"model={ai.get('model')}",
    })

    return results


def build_manifest(freeze: bool) -> dict:
    checks = anti_overfitting_checks()
    passed = all(check["passed"] for check in checks)

    geometry_config = REPO_ROOT / "config/geometry_config_v0.1.json"
    fixture_spec = REPO_ROOT / "samples/arkitscenes/fixture_spec.json"
    fixtures = json.loads(fixture_spec.read_text()) if fixture_spec.is_file() else {}

    manifest = {
        "manifestVersion": "0.1",
        "state": "PARAMETERS_FROZEN" if (freeze and passed) else "NOT_FROZEN",
        "generatedUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statement": (
            "The final capture must traverse exactly this configuration. Only a "
            "connector format repair, or an independently reproducible source-neutral "
            "bug fix, is permitted afterwards, and each must be logged with its "
            "benchmark-leakage rationale."),
        "antiOverfittingChecks": checks,
        "allChecksPassed": passed,
        "configurations": {
            relative: sha256(REPO_ROOT / relative)
            for relative in CONFIG_FILES if (REPO_ROOT / relative).is_file()
        },
        "codeModules": {
            relative: sha256(REPO_ROOT / relative)
            for relative in CODE_MODULES if (REPO_ROOT / relative).is_file()
        },
        "contractVersions": {
            "normalizedCapture": "normalized_capture/0.2",
            "spatialModelSchema": "0.1",
            "aiAssessmentSchema": "0.1",
            "geometryConfig": "geometry_config_v0.1",
            "confidenceRules": "confidence_rules_v0.1",
            "referenceExtraction": "reference_extraction_v0.1",
            "aiModelConfig": "ai_model_config_v0.1",
            "spatialVerifierPrompt": "spatial_verifier_v0.1",
            "lossProposalPrompt": "loss_proposal_v0.1",
        },
        "fixtures": [
            {"sceneId": entry["scene_id"], "role": entry["role"],
             "visitId": entry.get("visit_id"),
             "tuningPermission": entry.get("tuning_permission")}
            for entry in fixtures.get("fixtures", [])
        ],
        "notProvenByThisFreeze": [
            "Stray Scanner export compatibility.",
            "Measurement accuracy against independent tape references.",
            "Generalization to arbitrary homes.",
            "Calibrated confidence.",
            "Track B damage intelligence or Track C scope generation.",
            "Held-out performance.",
        ],
    }

    config = json.loads(geometry_config.read_text())
    already_frozen = bool(config.get("frozen"))

    # Freezing must be idempotent. Stamping a fresh timestamp on every run would
    # change the config hash and silently invalidate every artifact generated
    # against the previous one — which is exactly the drift a freeze exists to
    # prevent.
    if freeze and passed and not already_frozen:
        config["frozen"] = True
        config["frozenAt"] = manifest["generatedUtc"]
        geometry_config.write_text(json.dumps(config, indent=2) + "\n")
        manifest["configurations"][
            "config/geometry_config_v0.1.json"] = sha256(geometry_config)
        manifest["frozenAt"] = config["frozenAt"]
    elif already_frozen:
        manifest["frozenAt"] = config.get("frozenAt")
        manifest["state"] = "PARAMETERS_FROZEN" if passed else "FROZEN_BUT_CHECKS_FAILING"
        manifest["note"] = (
            "The configuration was already frozen; this run re-verified the checks and "
            "re-recorded hashes without restamping the freeze time.")
    return manifest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true",
                        help="mark the configuration frozen if every check passes")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "output" / "config_freeze_manifest.json")
    args = parser.parse_args(argv)

    manifest = build_manifest(args.freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")

    for check in manifest["antiOverfittingChecks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"[{mark}] {check['check']}")
        if not check["passed"]:
            print(f"       {check['detail']}")
    print(f"\n{manifest['state']} -> {args.output}")
    print(f"  {len(manifest['configurations'])} configurations, "
          f"{len(manifest['codeModules'])} code modules hashed")
    return 0 if manifest["allChecksPassed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
