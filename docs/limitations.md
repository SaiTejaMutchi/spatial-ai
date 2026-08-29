# Limitations and evidence scope

This document specifies the empirical limits, unvalidated claims, statistical confidence boundaries, and open questions of Spatial AI. It is designed to allow a reviewer or auditor to verify exactly what the current evidence supports and what it explicitly does **not** allow callers to conclude.

---

## 1. README Claim vs. Empirical Evidence Cross-Reference Table

| README Claim | Exact Supporting Evidence | Empirical Gap / Unvalidated Bound | What Current Evidence CANNOT Conclude |
| :--- | :--- | :--- | :--- |
| **"Height Accuracy MAE 2.82 cm (N=10)"** | 10 validation captures recorded in `latest_run.json` (Mean: 2.82 cm, Median: 1.95 cm, Std: 2.32 cm, p90: 5.69 cm, Range: 0.65–8.52 cm). | **95% Bootstrap CI is wide**: `[1.57 cm, 4.38 cm]`. 7 of 10 scenes (70%) fail the 1.50 cm gate (gate pass rate: 3/10 or 30%). Ground truth for scenes 6–10 is carried forward, not locally re-verifiable without re-downloading laser scans. | Cannot conclude accuracy on real-world commercial/residential property claims. Cannot claim sub-2cm accuracy. Earlier N=5 range (1.47–3.06 cm) is superseded by this wider N=10 distribution. |
| **"Systematic Negative Error / Depth Scale Bias Hypothesis"** | Fitted linear regression across N=10 validation scenes yields slope $m = -13.10\text{ cm/m}$, $R^2 = 0.235$, $p = 0.156$. | **Statistically Non-Significant at N=10**: Linear regression slope $p = 0.156 > 0.05$. | Cannot conclude that mobile LiDAR depth scale under-estimation is proven; it remains an unconfirmed hypothesis at $N=10$. |
| **"Deterministic 3D Geometry owns 100% of metric quantities"** | Structural guardrails in `pipeline/geometry/` and schema validation in `schema/spatial_model.schema.json`. | **Requires clean input depth streams**: Distorted or uncalibrated depth sensors will propagate metric errors deterministically. | Cannot guarantee metric accuracy if input sensor poses ($T_{WC}$) or depth images have uncalibrated focal drift. |
| **"AI Verifier cannot mutate geometry metrics"** | Guardrail assertion test in [`pipeline/tests/test_ai_verifier.py#L45`](../pipeline/tests/test_ai_verifier.py#L45) verifying byte-identical `spatial_model.json`. | **Schema-enforced only**: Does not prevent an AI model from hallucinating invalid surface IDs if schema validation is bypassed. | Cannot guarantee semantic accuracy of LLM classifications—only that LLM output cannot write metric numbers into geometry. |
| **"Zero Downloads Required for Quickstart"** | Pre-computed result snapshots bundled in `samples/public_results/`. | **Bundled snapshots are static**: Processing new raw captures requires raw depth streams and local processing. | Cannot process new physical spaces offline without installing pipeline dependencies (`spatial-ai[pipeline]`). |

---

## 2. Statistical Thresholds & Confirmation Requirements

### A. Sensor Scale Bias Hypothesis ($m = -13.10\text{ cm/m}, R^2 = 0.235, p = 0.156$)
- **Current Status**: **Hypothesis (Non-Significant, $p=0.156$)**
- **Observed Sample Fit ($N=10$)**: Slope $m = -13.10\text{ cm/m}$ (95% CI: `[-32.39 cm/m, +6.20 cm/m]`), $R^2 = 0.235$, Pearson $r = -0.4842$, $p = 0.156$.
- **Sample Size Sensitivity Analysis ($\alpha = 0.05, 1-\beta = 0.80$)**:
  - $R^2 = 0.10 \implies N_{\text{required}} = 77$ scenes
  - $R^2 = 0.20 \implies N_{\text{required}} = 37$ scenes
  - $R^2 = 0.30 \implies N_{\text{required}} = 24$ scenes
  - $R^2 = 0.40 \implies N_{\text{required}} = 18$ scenes
  - $R^2 = 0.50 \implies N_{\text{required}} = 14$ scenes
- **Threshold for Confirmation**: If evaluated on $N \ge 24–37$ held-out validation scenes and $p < 0.05$, the depth scale under-estimation hypothesis will move from *preliminary hypothesis* to *confirmed physical effect*.

### B. Accuracy Gate Pass Rate ($1.50\text{ cm}$ Target Gate)
- **Current Status**: **Failed Gate (Pass Rate: 30.0% / 3 of 10 scenes passed)**
- **Observed Aggregate Metrics ($N=10$)**:
  - Mean Absolute Error: $2.82\text{ cm}$ (95% CI: `[1.57 cm, 4.38 cm]`)
  - Median Absolute Error: $1.95\text{ cm}$
  - Standard Deviation: $2.32\text{ cm}$
  - p90 Absolute Error: $5.69\text{ cm}$
  - Min / Max Range: $0.65\text{ cm}$ to $8.52\text{ cm}$
- **Threshold for Gate Satisfaction**: Achieving a $\ge 90\%$ pass rate on the $1.50\text{ cm}$ gate requires applying sensor-calibrated depth scale correction and evaluating across $N \ge 30$ validation scenes.

---

## 3. Explicit Unresolved Open Questions (`docs/checklist.md`)

1. **Unexecuted External Baselines**:
   - External tools (Apple RoomPlan, RTAB-Map, COLMAP, Polycam) are listed in `docs/baselines.md` with unexecuted status due to hardware (iOS LiDAR session requirement) and ROS C++ build prerequisites.
   - **Limitation**: The repository does not claim numerical superiority over Apple RoomPlan or COLMAP.

2. **Occlusion & Clutter Sensitivity**:
   - Scenes with floor area clutter exceeding $>40\%$ (e.g. dense furniture) exhibit higher plane bounding box variance.
   - **Limitation**: RANSAC plane fitting relies on visible planar surface points; heavy occlusions require inferred observation states.

---

## 4. Multi-Scene Laser Ground-Truth Extraction Status & Verification Caveats ($N=10$)

- **Status**: **PARTIALLY RECOVERED ($N=10$ total; Scenes 1–5 fully re-derived, Scenes 6–10 carried forward)**
- **Scenes 1–5 (`groundTruthVerification`: `re-derived-from-source`)**:
  - Ground truth independently re-derived from raw FARO laser `.ply` point cloud files traceable to source (`47333462`, `41418135`, `41418155`, `41418140`, `42444474`).
  - Errors: 3.06 cm, 1.81 cm, 2.07 cm, 1.83 cm, 1.47 cm.
- **Scenes 6–10 (`groundTruthVerification`: `carried-forward-not-reverifiable`)**:
  - Predictions and height errors re-derived from processed spatial models still on disk, matching `latest_run.json` exactly.
  - Ground truth heights carried forward from prior recorded evaluation run because local laser `.ply` files were deleted to free disk space.
  - Visits: `421065` (Scene `42444499`, error 0.91 cm), `421063` (Scene `42444511`, error 2.23 cm), `421061` (Scene `42444514`, error 0.65 cm), `421060` (Scene `42444519`, error 5.69 cm), `421062` (Scene `42444574`, error 8.52 cm).
- **Known Fixable Gap**: Ground truth for scenes 6–10 can **only** be restored to fully-independent, locally re-derivable status by re-downloading the laser point clouds for visits `421065`, `421063`, `421061`, `421060`, `421062` and re-running extraction (`pipeline/eval/extract_gt.py`). This is documented as a known, specific, fixable gap in dataset provenance.
- **Superseded N=5 Distribution**: The earlier 5-scene range (1.47 cm – 3.06 cm, mean 2.05 cm) represented a smaller sample. The wider N=10 distribution (0.65 cm – 8.52 cm, mean 2.82 cm, median 1.95 cm, p90 5.69 cm) supersedes it as the current best estimate of pipeline performance.

---

## 5. Evaluation Execution Model & Generator Provenance

- **Evaluator Execution Mode**: `pipeline/eval/evaluator.py` dynamically parses local spatial models and capture directories, measuring per-scene execution time (`executionTime_ms`). Static JSON file reads (`ca1m_multiscene.json`) and hardcoded prediction constants (`pred_height = 2.427`) have been **completely removed**.
- **Generator Script**: `pipeline/validation/ca1m_benchmark.py` is the official generator script that invokes `pipeline.connectors.cli` + `pipeline.geometry.run` + `pipeline.validation.ca1m_eval` over raw CA-1M capture directories.
- **Handling Missing Inputs**: Captures without local raw video archives on disk are explicitly recorded as skipped with `"reason": "MISSING_LOCAL_CAPTURE_INPUTS"` rather than substituting pre-computed static JSON predictions.
