"""Ground truth must not reach reconstruction.

CA-1M publishes the laser-registered pose and the FARO-rendered depth for the
same frames the pipeline reconstructs from. Those are the answer. If any of them
reached geometry, the benchmark would be measuring the reference against itself.

The ledger names the three assets explicitly, so these tests name them too.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("gt/RT.json", "gt/depth.png", "world.gt/instances.json",
             "RT.json", "instances.json")

# Everything that runs before or during reconstruction.
INFERENCE_PACKAGES = ("pipeline/geometry", "pipeline/connectors", "pipeline/contracts",
                      "pipeline/rendering", "pipeline/evidence", "service")


def _inference_sources() -> list[Path]:
    files: list[Path] = []
    for package in INFERENCE_PACKAGES:
        files.extend(sorted((REPO_ROOT / package).rglob("*.py")))
    return files


def test_no_inference_module_names_a_ground_truth_asset():
    offenders = []
    for path in _inference_sources():
        text = path.read_text()
        for name in FORBIDDEN:
            if name in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} mentions {name}")
    assert not offenders, offenders


def test_no_inference_module_imports_the_ca1m_evaluator():
    """The evaluator reads ground truth, so nothing upstream may import it."""
    offenders = []
    for path in _inference_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if "ca1m_eval" in name or "validation" in name.split("."):
                    offenders.append(f"{path.relative_to(REPO_ROOT)} imports {name}")
    assert not offenders, offenders


def test_the_evaluator_reads_ground_truth_only_after_freezing_the_prediction():
    """Order matters: the prediction hash is taken before any gt file is opened."""
    source = (REPO_ROOT / "pipeline" / "validation" / "ca1m_eval.py").read_text()
    freeze_at = source.index("prediction_sha = sha256_file(model_path)")
    gt_at = source.index("ground_truth = read_ground_truth(")
    assert freeze_at < gt_at
    assert "# Everything below this line reads ground truth." in source


def test_alignment_is_never_fitted_to_the_prediction():
    """Fitting to the prediction would let it choose the frame it is judged in."""
    source = (REPO_ROOT / "pipeline" / "validation" / "ca1m_eval.py").read_text()
    tree = ast.parse(source)
    solve = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "solve_alignment")
    args = {a.arg for a in solve.args.args}
    assert args == {"mobile", "ground_truth", "tolerance_s"}, (
        "solve_alignment must see poses only, never the model", args)


@pytest.mark.skipif(
    not (REPO_ROOT / "validation" / "manifests" / "ca1m_heldout_manifest.json").is_file(),
    reason="the held-out manifest has not been frozen on this machine")
def test_the_manifest_freezes_selection_before_any_error_is_known():
    manifest = json.loads((REPO_ROOT / "validation" / "manifests"
                           / "ca1m_heldout_manifest.json").read_text())
    assert manifest["frozenBeforeAnyErrorWasComputed"] is True
    assert manifest["heldOut"] is True
    assert len(manifest["captures"]) >= 3
    assert manifest["gateCapture"] not in {c["captureId"] for c in manifest["captures"]}
    assert "selectionRule" in manifest and len(manifest["selectionRule"]) > 80
    # The frozen geometry hash must match what the pipeline actually carries.
    frozen = json.loads((REPO_ROOT / "output" / "config_freeze_manifest.json").read_text())
    assert (manifest["geometryConfigHash"]
            == frozen["configurations"]["config/geometry_config_v0.1.json"])
