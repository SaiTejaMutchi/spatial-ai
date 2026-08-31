"""AI value layer freeze."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.ai.value_layer_freeze import (
    GEOMETRY_CONFIG,
    GEOMETRY_FREEZE,
    OUT_PATH,
    build_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_geometry_hash_matches_the_config_freeze():
    manifest = build_manifest(freeze=True)
    recorded = json.loads(GEOMETRY_FREEZE.read_text())
    assert manifest["geometryConfigHash"] == recorded["configurations"][GEOMETRY_CONFIG]
    assert manifest["geometryUntouchedByExtension"] is True


def test_visible_condition_chain_is_required_to_freeze():
    manifest = build_manifest(freeze=True)
    assert manifest["t14bChain"]["passed"] is True
    assert "geometry_pipeline" in manifest["t14bChain"]["detail"]


def test_emitted_manifest_records_the_frozen_ai_layer():
    assert OUT_PATH.is_file()
    manifest = json.loads(OUT_PATH.read_text())
    assert manifest["state"] == "AI_VALUE_LAYER_FROZEN"
    assert manifest["provider"] == "groq"
    assert manifest["model"] == "qwen/qwen3.6-27b"
    assert "prompts/visible_condition_v0.1.txt" in manifest["promptHashes"]
    assert "schema/visible_condition.schema.json" in manifest["schemaHashes"]
    assert manifest["structured3dFixture"]["notOfficialStructured3D"] is True
    text = " ".join(manifest["notProvenByThisFreeze"]).lower()
    assert "classification" in text
    assert "stray" in text
