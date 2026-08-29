"""CLI module entry point for running Spatial AI full accuracy evaluation suite."""

from __future__ import annotations

import argparse
import sys
from .baselines import run_baseline_comparison
from .evaluator import run_stratified_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial AI Accuracy Evaluation & Baseline Comparison Suite.")
    parser.add_argument("--full", action="store_true", help="Run full stratified ARKitScenes benchmark and baseline comparisons.")
    args = parser.parse_args()

    print("==================================================================")
    print("      SPATIAL AI: ACCURACY BENCHMARK & EVALUATION SUITE           ")
    print("==================================================================")

    print("\n[1/2] Running Stratified ARKitScenes Benchmark Evaluation...")
    eval_res = run_stratified_evaluation()

    print(f"\nTotal Scenes Evaluated: {eval_res['totalScenes']}")
    print(f"Overall MAE: {eval_res['overallHeightError_cm']['mean']} cm (Median: {eval_res['overallHeightError_cm']['median']} cm, P90: {eval_res['overallHeightError_cm']['p90']} cm)")
    print("Stratified by Room Size:")
    for b_name, b_stats in eval_res["stratificationBySizeBucket_cm"].items():
        print(f"  - {b_name.capitalize()} Rooms (N={b_stats['n']}): MAE={b_stats['mean']} cm, Median={b_stats['median']} cm")

    print("\n[2/2] Running Geometry Baseline Comparison...")
    base_res = run_baseline_comparison()
    print("\nComparative Baseline Metrics:")
    for m in base_res["metrics"]:
        print(f"  - {m['system']}: MAE={m['heightMAE_cm']} cm, Median={m['heightMedian_cm']} cm, P90={m['heightP90_cm']} cm")

    print("\n==================================================================")
    print("  EVALUATION COMPLETE — Results saved in pipeline/eval/results/   ")
    print("==================================================================")


if __name__ == "__main__":
    main()
