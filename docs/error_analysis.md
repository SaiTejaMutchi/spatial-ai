# Systematic Error Analysis & Statistical Bias Characterization

## Executive Summary

Evaluation of Spatial AI against terrestrial **FARO laser scanner ground truth** across benchmark captures ($N=10$ live evaluated scenes) reveals:

### N=10 Aggregate Accuracy & Gate Pass Rate
- **Mean Absolute Error (MAE)**: **`2.82 cm`** (2.8237 cm, 95% Bootstrap CI: `[1.57 cm, 4.38 cm]`)
- **Median Absolute Error**: **`1.95 cm`** (1.9465 cm, 95% Bootstrap CI: `[1.36 cm, 3.96 cm]`)
- **Standard Deviation**: **`2.32 cm`** (2.3219 cm)
- **90th Percentile (p90)**: **`5.69 cm`** (5.694 cm)
- **Error Range (Min / Max)**: **`0.65 cm` to `8.52 cm`**
- **Gate Pass Rate ($\le 1.50\text{ cm}$)**: **`3/10` (30.0%)** (Scenes 5, 6, 8 passed)

> [!IMPORTANT]
> **N=10 Supersedes Earlier N=5 Sample**: The earlier five-scene error band of **1.47 cm – 3.06 cm** (mean 2.05 cm) represented a smaller sample size. The wider $N=10$ distribution (**0.65 cm – 8.52 cm**, mean **2.82 cm**, median **1.95 cm**, p90 **5.69 cm**) supersedes it as the current best estimate of system accuracy.

> [!NOTE]
> **Ground Truth Verification Distinction**:
> - **Scenes 1–5**: Ground truth independently re-derived from raw laser `.ply` point cloud files traceable to source (`re-derived-from-source`).
> - **Scenes 6–10**: Predictions and errors re-derived from local spatial models on disk; ground truth carried forward from prior recorded evaluation run because raw laser `.ply` files were removed to free disk space (`carried-forward-not-reverifiable`). Independent status can be restored by re-downloading laser point clouds for visits 421065, 421063, 421061, 421060, 421062 and re-running extraction.

---

## 1. Per-Scene Raw Measurements & Ground Truth Verification Status

| Scene ID | Visit ID | Dataset | FARO Laser Height ($h_{\text{laser}}$) | Reconstructed Height ($h_{\text{pred}}$) | Absolute Error | Signed Error | Status ($\le 1.50\text{ cm}$) | Ground Truth Verification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`47333462`** | `467138` | ARKitScenes | `2.6300 m` | `2.5994 m` | **`3.06 cm`** | `-3.06 cm` | Failed Gate (by 1.56 cm) | Re-derived from source |
| **`41418135`** | `416418` | ARKitScenes | `2.4302 m` | `2.4121 m` | **`1.81 cm`** | `-1.81 cm` | Failed Gate (by 0.31 cm) | Re-derived from source |
| **`41418155`** | `416407` | ARKitScenes | `2.4303 m` | `2.4096 m` | **`2.07 cm`** | `-2.07 cm` | Failed Gate (by 0.57 cm) | Re-derived from source |
| **`41418140`** | `416411` | ARKitScenes | `2.4428 m` | `2.4245 m` | **`1.83 cm`** | `-1.83 cm` | Failed Gate (by 0.33 cm) | Re-derived from source |
| **`42444474`** | `421069` | ARKitScenes | `2.3249 m` | `2.3102 m` | **`1.47 cm`** | `-1.47 cm` | **PASSED GATE** | Re-derived from source |
| **`42444499`** | `421065` | ARKitScenes | `2.2944 m` | `2.2853 m` | **`0.91 cm`** | `-0.91 cm` | **PASSED GATE** | Carried forward (unverifiable locally) |
| **`42444511`** | `421063` | ARKitScenes | `2.3073 m` | `2.3296 m` | **`2.23 cm`** | `+2.23 cm` | Failed Gate (by 0.73 cm) | Carried forward (unverifiable locally) |
| **`42444514`** | `421061` | ARKitScenes | `2.1311 m` | `2.1246 m` | **`0.65 cm`** | `-0.65 cm` | **PASSED GATE** | Carried forward (unverifiable locally) |
| **`42444519`** | `421060` | ARKitScenes | `2.2972 m` | `2.3541 m` | **`5.69 cm`** | `+5.69 cm` | Failed Gate (by 4.19 cm) | Carried forward (unverifiable locally) |
| **`42444574`** | `421062` | ARKitScenes | `2.4578 m` | `2.3726 m` | **`8.52 cm`** | `-8.52 cm` | Failed Gate (by 7.02 cm) | Carried forward (unverifiable locally) |

---

## 2. Statistical Analysis of Error Bias & Scale Hypothesis

### Non-Parametric Bootstrapping ($N=10$, 10,000 Resamples)

| Metric | Point Estimate | Standard Dev ($\sigma$) | Bootstrap 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | `2.82 cm` | `2.32 cm` | **`[1.57 cm, 4.38 cm]`** |
| **Median Absolute Error** | `1.95 cm` | `—` | **`[1.36 cm, 3.96 cm]`** |
| **Mean Signed Error** | `-1.24 cm` | `3.44 cm` | **`[-3.38 cm, +0.91 cm]`** |

### Linear Regression Analysis ($\text{SignedError}_{\text{cm}} \text{ vs } h_{\text{ref\_m}}$)

We model signed height error as a linear function of room reference height:

$$\text{SignedError}_{\text{cm}} = m \cdot h_{\text{ref}} + b$$

- **Fitted Slope ($m$)**: **`-13.10 cm/m`** (95% CI: `[-32.39 cm/m, +6.20 cm/m]`)
- **Intercept ($b$)**: `+29.86 cm`
- **Coefficient of Determination ($R^2$)**: **`0.2345`** ($23.5\%$ of signed error variance explained by height)
- **Pearson Correlation ($r$)**: **`-0.4842`**
- **p-value**: **`0.1561`** ($p > 0.05$, statistically non-significant at $N=10$)
- **Residual Standard Deviation**: `3.36 cm`

### Sample Size Sensitivity Analysis ($N_{\text{required}}$ for Power $= 0.80, \alpha = 0.05$)

Because the linear regression slope is statistically non-significant at $N=10$ ($p = 0.1561$), we report the sample size sensitivity table indicating the required sample size ($N_{\text{required}}$) across assumed true population effect sizes ($R^2$):

| Assumed True Effect Size ($R^2$) | Assumed True Correlation ($|r|$) | Required Sample Size ($N_{\text{required}}$) |
| :---: | :---: | :---: |
| **`0.10`** (Small Effect) | `0.3162` | **`77 scenes`** |
| **`0.20`** (Moderate-Low) | `0.4472` | **`37 scenes`** |
| **`0.30`** (Moderate) | `0.5477` | **`24 scenes`** |
| **`0.40`** (Moderate-High) | `0.6325` | **`18 scenes`** |
| **`0.50`** (Large Effect) | `0.7071` | **`14 scenes`** |

$$\text{Conclusion: At } N=10 \text{, the depth scale bias is statistically non-significant } (p=0.1561)\text{. A sample size of } N \in [24, 77] \text{ scenes is required to confirm or refute the effect at } \alpha=0.05 \text{ with } 80\% \text{ power.}$$

---

## 3. Explicit Failure Modes & Degradation Conditions

1. **Large Spans (> 6.0 m)**: Trajectory drift and optical depth precision decay on room dimensions exceeding 6 meters.
2. **Floor Clutter & Furniture Occlusion**: Heavy furniture coverage reduces directly observed floor points, increasing plane inference uncertainty (e.g. Scene `42444574` with 8.52 cm error).
3. **Glass & Highly Reflective Surfaces**: Transparent or mirrored surfaces cause point cloud sparsity, resulting in inferred rather than directly observed wall bounds.
