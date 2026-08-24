"""Packaging and configuration-freeze tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.freeze import (
    CODE_MODULES,
    CONFIG_FILES,
    anti_overfitting_checks,
    build_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "output" / "config_freeze_manifest.json"


def test_every_anti_overfitting_check_passes():
    failures = [c for c in anti_overfitting_checks() if not c["passed"]]
    assert not failures, json.dumps(failures, indent=2)


def test_the_checks_cover_what_the_plan_requires():
    names = " ".join(c["check"] for c in anti_overfitting_checks())
    for requirement in ("hardcoded scene", "source-specific", "secondary validation",
                        "one geometry configuration", "damage[]", "operator approval"):
        assert requirement in names, requirement


def test_freezing_is_idempotent():
    """Restamping the freeze time would invalidate every prior artifact."""
    first = build_manifest(freeze=True)
    second = build_manifest(freeze=True)
    key = "config/geometry_config_v0.1.json"
    assert first["configurations"][key] == second["configurations"][key]
    assert first.get("frozenAt") == second.get("frozenAt")


def test_the_manifest_records_the_configuration_that_is_frozen():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["state"] == "PARAMETERS_FROZEN"
    assert manifest["allChecksPassed"] is True
    for relative in CONFIG_FILES:
        assert relative in manifest["configurations"], relative
    for relative in CODE_MODULES:
        assert relative in manifest["codeModules"], relative


def test_generated_models_reference_the_frozen_configuration():
    manifest = json.loads(MANIFEST.read_text())
    frozen = manifest["configurations"]["config/geometry_config_v0.1.json"]
    models = sorted((REPO_ROOT / "outputs").glob("dev_*/spatial_model.json"))
    if not models:
        models = sorted((REPO_ROOT / "samples" / "public_results").glob("*/output/spatial_model.json"))
    assert models, "no generated models to check"
    for path in models:
        document = json.loads(path.read_text())
        assert document["provenance"]["geometryConfigHash"] == frozen, path.parent.name


def test_the_manifest_states_what_the_freeze_does_not_prove():
    manifest = json.loads(MANIFEST.read_text())
    text = " ".join(manifest["notProvenByThisFreeze"]).lower()
    for claim in ("stray", "accuracy", "generaliz", "calibrated", "track b", "held-out"):
        assert claim in text, claim


def test_fixture_roles_are_recorded_in_the_freeze():
    manifest = json.loads(MANIFEST.read_text())
    roles = {f["role"] for f in manifest["fixtures"]}
    assert {"PRIMARY_TUNING", "SECONDARY_VALIDATION"} <= roles
    secondary = next(f for f in manifest["fixtures"]
                     if f["role"] == "SECONDARY_VALIDATION")
    assert "NO_TUNING" in secondary["tuningPermission"]


def test_the_geometry_config_is_marked_frozen_and_uncalibrated():
    config = json.loads((REPO_ROOT / "config/geometry_config_v0.1.json").read_text())
    assert config["frozen"] is True
    assert config["frozenAt"]
    assert config["calibrated"] is False


# --------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------

def test_one_documented_startup_command_exists():
    script = REPO_ROOT / "run_local.sh"
    assert script.is_file()
    text = script.read_text()
    assert "uvicorn" in text and "service.api:app" in text
    assert "127.0.0.1" in text, "the service must bind to localhost only"


@pytest.mark.parametrize("relative", [
    "config/geometry_config_v0.1.json",
    "config/confidence_rules_v0.1.json",
    "config/ai_model_config.json",
    "prompts/spatial_verifier_v0.1.txt",
    "prompts/loss_proposal_v0.1.txt",
    "output/development_reference_benchmark.json",
    "output/development_reference_benchmark.md",
    "output/config_freeze_manifest.json",
    "schema/spatial_model.schema.json",
    "README.md",
])
def test_every_required_defensibility_artifact_exists(relative):
    path = REPO_ROOT / relative
    assert path.is_file(), relative
    assert path.stat().st_size > 0, relative


def test_the_readme_makes_no_final_accuracy_claim():
    text = (REPO_ROOT / "README.md").read_text().lower()
    assert "not proven" in text
    assert "no accuracy claim" in text or "does not demonstrate accuracy" in text
    assert "dev_complete" in text


def test_the_readme_reports_the_failing_gate():
    """A README that omitted the miss would be the wrong kind of packaging."""
    text = (REPO_ROOT / "README.md").read_text()
    assert "2.46 cm" in text
    assert "misses" in text.lower()
