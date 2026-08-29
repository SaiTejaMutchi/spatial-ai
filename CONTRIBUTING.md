# Contributing to Spatial AI

Thank you for your interest in contributing to Spatial AI!

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SaiTejaMutchi/spatial-ai.git
   cd spatial-ai
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .[all]
   ```

3. **Run automated test suite**:
   ```bash
   python3 -m pytest pipeline/tests/ -q
   ```

4. **Run accuracy evaluation suite**:
   ```bash
   python3 -m pipeline.eval --full
   ```

---

## Core Invariants

When contributing, ensure your changes preserve Spatial AI's core architectural invariants:

1. **Geometry Owns Metric Truth**: Deterministic geometry in `pipeline/geometry/` owns all length, width, height, and area calculations. AI models interpret semantics; they do not alter dimensions.
2. **Configuration Freeze**: Frozen parameters in `pipeline/geometry/config.py` and `output/config_freeze_manifest.json` must not be tuned to fit benchmark data.
3. **Reproducibility**: All published evaluation numbers must trace to committed JSON result files under `pipeline/eval/results/`.

---

## Submitting Pull Requests

1. Keep pull requests focused on a single issue or feature.
2. Ensure all 330+ correctness tests pass (`python3 -m pytest pipeline/tests/ -q`).
3. Include unit tests for any new connector, adapter, or SDK method.
