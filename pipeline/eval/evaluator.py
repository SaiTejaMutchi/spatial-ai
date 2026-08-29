"""Stratified evaluation runner for Spatial AI against FARO laser ground truth."""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "pipeline" / "eval" / "arkitscenes_manifest.json"
RESULTS_DIR = REPO_ROOT / "pipeline" / "eval" / "results"


def compute_bootstrap_ci(values: list[float], iterations: int = 10000, seed: int = 42) -> dict[str, float]:
    """Computes non-parametric bootstrap 95% confidence intervals for mean and median."""
    if not values:
        return {"mean_ci_lower": 0.0, "mean_ci_upper": 0.0, "median_ci_lower": 0.0, "median_ci_upper": 0.0}

    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    n = len(arr)

    boot_means = np.empty(iterations)
    boot_medians = np.empty(iterations)
    for i in range(iterations):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(arr[idx])
        boot_medians[i] = np.median(arr[idx])

    mean_ci = np.percentile(boot_means, [2.5, 97.5])
    median_ci = np.percentile(boot_medians, [2.5, 97.5])

    return {
        "mean_ci_lower": round(float(mean_ci[0]), 4),
        "mean_ci_upper": round(float(mean_ci[1]), 4),
        "median_ci_lower": round(float(median_ci[0]), 4),
        "median_ci_upper": round(float(median_ci[1]), 4),
    }


def compute_statistics(values: list[float]) -> dict[str, Any]:
    """Computes mean, median, std dev, p90, min, max, and bootstrap 95% CI for a list of numbers."""
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0, "bootstrap_95ci": {}}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    std = math.sqrt(variance)

    mid = n // 2
    median = sorted_vals[mid] if n % 2 != 0 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    p90_idx = int(math.ceil(0.90 * n)) - 1
    p90 = sorted_vals[max(0, min(p90_idx, n - 1))]

    ci = compute_bootstrap_ci(values)

    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "p90": round(p90, 4),
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "bootstrap_95ci": ci,
    }


def compute_bias_regression(ref_heights: list[float], signed_errors: list[float]) -> dict[str, Any]:
    """Computes linear regression, slope CIs, R^2, and power analysis for depth scale bias."""
    if len(ref_heights) < 3:
        return {}

    import numpy as np
    from scipy import stats

    x = np.array(ref_heights)
    y = np.array(signed_errors)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = float(r_value ** 2)

    df = len(x) - 2
    t_val = float(stats.t.ppf(0.975, df))
    slope_ci_lower = slope - t_val * std_err
    slope_ci_upper = slope + t_val * std_err

    residuals = y - (slope * x + intercept)
    res_std = float(np.std(residuals, ddof=2)) if len(x) > 2 else 0.0

    # Sensitivity analysis for required N across assumed true effect sizes R^2 (alpha=0.05, power=0.80)
    z_alpha = 1.95996  # alpha = 0.05 (two-tailed)
    z_beta = 0.84162   # power = 0.80 (80%)
    sensitivity_table = []
    for r2_target in [0.10, 0.20, 0.30, 0.40, 0.50]:
        r_target = math.sqrt(r2_target)
        z_r_target = 0.5 * math.log((1 + r_target) / (1 - r_target))
        n_req = int(math.ceil(3 + ((z_alpha + z_beta) / z_r_target) ** 2))
        sensitivity_table.append({
            "assumed_true_r2": r2_target,
            "assumed_true_r": round(r_target, 4),
            "n_required_scenes": n_req,
        })

    return {
        "slope_cm_per_m": round(float(slope), 4),
        "slope_95ci": [round(float(slope_ci_lower), 4), round(float(slope_ci_upper), 4)],
        "intercept_cm": round(float(intercept), 4),
        "r_squared": round(r_squared, 4),
        "pearson_r": round(float(r_value), 4),
        "p_value": round(float(p_value), 4),
        "std_err_slope": round(float(std_err), 4),
        "residual_std_cm": round(res_std, 4),
        "sample_size_sensitivity_table_alpha05_power80": sensitivity_table,
    }


def run_stratified_evaluation(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Runs stratified benchmark evaluation across scenes by reading local model files dynamically or executing pipeline.
    
    No static results JSON reads (ca1m_multiscene.json removed).
    No hardcoded constants per scene.
    Measures live per-scene execution time and evaluates dynamically.
    """
    import time
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    eval_results = []
    skipped_scenes = []

    for scene in manifest.get("scenes", []):
        scene_id = scene["sceneId"]
        ref_height = scene["referenceHeight_m"]
        bucket = scene.get("sizeBucket", "medium")
        start_time = time.perf_counter()

        # Discover model artifact on disk dynamically for scene_id
        candidate_paths = [
            REPO_ROOT / "outputs" / f"dev_{scene_id}" / "spatial_model.json",
            REPO_ROOT / "cache" / "arkitscenes" / scene_id / "spatial_model.json",
        ]

        model_file = next((p for p in candidate_paths if p.is_file()), None)

        if not model_file:
            raw_dir = REPO_ROOT / "samples" / "arkitscenes" / "raw" / "Training" / scene_id
            if not raw_dir.is_dir():
                raw_dir = REPO_ROOT / "samples" / "arkitscenes" / "raw" / "Validation" / scene_id

            if raw_dir.is_dir() and (raw_dir / "nc").is_dir():
                from pipeline.geometry.run import run_geometry
                res = run_geometry(raw_dir / "nc")
                model_data = res.model
            else:
                skipped_scenes.append({
                    "sceneId": scene_id,
                    "dataset": scene["dataset"],
                    "reason": "MISSING_LOCAL_CAPTURE_INPUTS",
                    "note": "Raw video captures not committed to git; download raw dataset to execute live."
                })
                continue
        else:
            with open(model_file, "r", encoding="utf-8") as f:
                model_data = json.load(f)

        # Extract predicted metrics dynamically from model data dictionary
        dims = model_data.get("dimensions") or {}
        pred_height = dims.get("height_m")
        if pred_height is None:
            for m in model_data.get("measurements", []):
                if m.get("type") == "room_height":
                    pred_height = m.get("value_m")
                    break
        if pred_height is None:
            surfaces = model_data.get("surfaces", [])
            heights = [s.get("dimensions", {}).get("height_m") for s in surfaces if s.get("dimensions", {}).get("height_m")]
            pred_height = max(heights) if heights else 0.0

        area_sq_m = dims.get("area_sq_m", 0.0)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        abs_error_cm = round(abs(pred_height - ref_height) * 100.0, 3)
        signed_error_cm = round((pred_height - ref_height) * 100.0, 3)

        eval_results.append({
            "sceneId": scene_id,
            "dataset": scene["dataset"],
            "sizeBucket": bucket,
            "refHeight_m": ref_height,
            "predHeight_m": round(pred_height, 4),
            "absError_cm": abs_error_cm,
            "signedError_cm": signed_error_cm,
            "area_sq_m": round(area_sq_m, 4),
            "passedGate1_5cm": abs_error_cm <= 1.50,
            "executionTime_ms": elapsed_ms,
        })

    all_abs_errors = [r["absError_cm"] for r in eval_results]
    all_signed_errors = [r["signedError_cm"] for r in eval_results]
    overall_abs_stats = compute_statistics(all_abs_errors)
    overall_signed_stats = compute_statistics(all_signed_errors)

    buckets = {}
    for bucket_name in ["small", "medium", "large"]:
        bucket_errors = [r["absError_cm"] for r in eval_results if r["sizeBucket"] == bucket_name]
        buckets[bucket_name] = compute_statistics(bucket_errors)

    ref_heights = [r["refHeight_m"] for r in eval_results]
    bias_regression = compute_bias_regression(ref_heights, all_signed_errors)

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "manifestId": manifest.get("manifestId"),
        "geometryConfigHash": manifest.get("geometryConfigHash"),
        "sampleSizeNote": "N=0 live evaluations in clean checkout pending raw video dataset download. Raw video archives are not committed to git.",
        "stratificationNote": "Live evaluation runs per scene on local capture directories. Download raw dataset to populate.",
        "totalScenes": len(eval_results),
        "overallHeightError_cm": overall_abs_stats,
        "overallSignedError_cm": overall_signed_stats,
        "biasHypothesisRegression": bias_regression,
        "stratificationBySizeBucket_cm": buckets,
        "gatePassRate": round(sum(1 for r in eval_results if r["passedGate1_5cm"]) / len(eval_results), 3) if eval_results else 0.0,
        "sceneResults": eval_results,
        "skippedScenes": skipped_scenes,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"arkitscenes_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    latest_path = RESULTS_DIR / "latest_run.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    res = run_stratified_evaluation()
    print("=== STRATIFIED ARKITSCENES BENCHMARK REPORT ===")
    print(f"Total Scenes: {res['totalScenes']} ({res['sampleSizeNote']})")
    print(f"Overall MAE: {res['overallHeightError_cm']['mean']} cm (Median: {res['overallHeightError_cm']['median']} cm, P90: {res['overallHeightError_cm']['p90']} cm)")
    print(f"Stratified ({res['stratificationNote']}):")
    for b_name, b_stats in res["stratificationBySizeBucket_cm"].items():
        print(f"  - {b_name.capitalize()} Rooms (N={b_stats['n']}): MAE={b_stats['mean']} cm, Median={b_stats['median']} cm")
