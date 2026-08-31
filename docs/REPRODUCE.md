# Reproducing the benchmark and evaluation

This document provides exact instructions to reproduce all evaluation metrics, accuracy figures, statistical confidence intervals, and correctness tests reported in the Spatial AI repository from a clean checkout.

---

## 1. Hardware & System Prerequisites

- **Operating System**: macOS (Apple Silicon M1/M2/M3) or Ubuntu 22.04 LTS
- **Python Version**: Python `3.10`, `3.11`, or `3.12` (Tested on `3.12.2`)
- **Memory**: Minimum `8 GB RAM` (`16 GB RAM` recommended)
- **Disk Space**: `~1.5 GB` for code repository and bundled public result snapshots

---

## 2. Environment Setup & Dependency Pinning

Clone the repository and install exact pinned dependencies from `requirements.lock`:

```bash
git clone https://github.com/SaiTejaMutchi/spatial-ai.git
cd spatial-ai

# Create clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install exact pinned lockfile dependencies
pip install -r requirements.lock
pip install -e .
```

---

## 3. Single-Command Full Evaluation Benchmark Reproduction

To reproduce all accuracy numbers, MAE figures, median error, 90th percentile bounds, bootstrap 95% confidence intervals, linear regression, and power analysis statistics:

```bash
python3 -m pipeline.eval --full
```

### Benchmark Runtime & Hardware Budget
- **Wall-Clock Execution Time**: **`< 1.0 second`** (`~0.39s` on Apple M2 Pro)
- **Primary Artifact Output**: [`pipeline/eval/results/latest_run.json`](../pipeline/eval/results/latest_run.json)

### Expected Output Summary
```text
Total Scenes Evaluated: 6
Overall Height MAE: 2.9413 cm [95% Bootstrap CI: 1.9148 cm to 3.9520 cm]
Median Absolute Error: 2.8770 cm [95% Bootstrap CI: 1.4585 cm to 4.4885 cm]
Mean Signed Error: -2.1080 cm [95% Bootstrap CI: -3.8678 cm to -0.0263 cm]
Depth Scale Bias Linear Regression:
  - Slope: -4.1988 cm/m [95% CI: -10.3956 cm/m, +1.9979 cm/m]
  - R^2: 0.4694 (p = 0.1331)
  - Power Analysis N_required (alpha=0.05, power=0.80): 15 scenes
Gate Pass Rate (<= 1.50 cm): 16.7% (1 of 6 scenes passed)
```

---

## 4. Automated Correctness Test Suite Execution

To execute all 440 automated correctness and boundary guardrail tests:

```bash
python3 -m pytest pipeline/tests/ -q
```

### Test Execution Runtime Budget
- **Wall-Clock Execution Time**: **`~10 minutes`** (`~626s` on single-threaded test runner; `< 30s` with `pytest -n auto`)
- **Test Summary**: `334 passed`, `106 skipped` (106 skips intentional for raw multi-gigabyte video datasets & live API keys—see [`docs/testing.md`](testing.md)).

---

## 5. Explicit Random Seeds & Stochastic Parameters

All random sampling, RANSAC plane fitting, and non-parametric statistical bootstrapping use fixed, explicit random seeds:

| Subsystem / Task | Script / Module Path | Explicit Random Seed | Iterations / Parameters |
| :--- | :--- | :---: | :--- |
| **Statistical Bootstrapping (95% CI)** | [`pipeline/eval/evaluator.py`](../pipeline/eval/evaluator.py#L16) | **`seed = 42`** | `10,000` resamples with replacement |
| **RANSAC 3D Plane Extraction** | [`pipeline/geometry/planes.py`](../pipeline/geometry/planes.py) | **`seed = 42`** | `1,000` iterations, distance threshold = `0.03 m` |
| **2D Hough Wall Line Fitting** | [`pipeline/geometry/planes.py`](../pipeline/geometry/planes.py) | **`seed = 42`** | Angle resolution = $1^\circ$, distance threshold = `0.05 m` |
