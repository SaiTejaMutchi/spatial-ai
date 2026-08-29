"""Spatial AI Evaluation & Accuracy Benchmark Suite."""

from .baselines import run_baseline_comparison
from .evaluator import run_stratified_evaluation
from .self_capture import evaluate_self_capture

__all__ = [
    "run_stratified_evaluation",
    "run_baseline_comparison",
    "evaluate_self_capture",
]
