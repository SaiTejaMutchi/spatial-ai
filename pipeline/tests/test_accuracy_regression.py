"""Regression test suite for Spatial AI accuracy evaluation module."""

from pathlib import Path
import pytest
from pipeline.eval import run_baseline_comparison, run_stratified_evaluation

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_stratified_evaluation_runner():
    """Verifies stratified evaluation runner executes and computes metrics dynamically."""
    res = run_stratified_evaluation()
    assert "overallHeightError_cm" in res
    assert "mean" in res["overallHeightError_cm"]
    assert "stratificationBySizeBucket_cm" in res


def test_baseline_comparison_runner():
    """Verifies baseline comparison runner executes and outputs comparison specifications."""
    res = run_baseline_comparison()
    assert len(res["metrics"]) == 3
    systems = [m["system"] for m in res["metrics"]]
    assert any("Spatial AI" in s for s in systems)
    assert any("RoomPlan" in s for s in systems)
    assert any("RTAB-Map" in s for s in systems)


def test_accuracy_regression_threshold():
    """Asserts that overall room height MAE does not regress beyond 5.0 cm threshold on evaluated scenes."""
    res = run_stratified_evaluation()
    if res["totalScenes"] > 0:
        overall_mae = res["overallHeightError_cm"]["mean"]
        assert overall_mae < 5.0, f"Accuracy regression detected! MAE {overall_mae} cm exceeds 5.0 cm threshold."
