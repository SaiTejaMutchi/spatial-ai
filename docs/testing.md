# Testing & Evaluation Guide

Spatial AI maintains a strict separation between **automated correctness tests** and **accuracy evaluation benchmarks**.

---

## 1. Automated Correctness Test Suite

The correctness test suite ([`pipeline/tests/`](../pipeline/tests/)) verifies schema compliance, pipeline execution, connector normalization, and structural guardrails.

```bash
python3 -m pytest pipeline/tests/ -q
```

### Test Skip Breakdown (106 Skipped Tests)

When running on a fresh clone without external dataset downloads or API keys, pytest reports **333 passed, 106 skipped**. The 106 skips are intentional and fall into three categories:

| Skip Reason | Count | Prerequisite to Enable |
| :--- | :---: | :--- |
| **Missing Multi-Gigabyte Raw Video Archives** | ~80 | Download full ARKitScenes / CA-1M raw RGB-D video sequences into `samples/` or `outputs/`. |
| **Missing Multimodal VLM API Keys** | ~20 | Set `GROQ_API_KEY` or `ANTHROPIC_API_KEY` in `.env` to execute live VLM API verifier calls. |
| **Missing Local Model Artifacts** | ~6 | Generated local geometry models in `outputs/dev_47333462/` (run `python3 -m pipeline.geometry.run`). |

---

## 2. Accuracy Evaluation & Regression Suite

Accuracy benchmarks evaluate measurement precision against terrestrial **FARO laser scanner ground truth**.

```bash
python3 -m pipeline.eval --full
```

- Outputs machine-readable evaluation results to [`pipeline/eval/results/latest_run.json`](../pipeline/eval/results/latest_run.json).
- Checks overall height MAE against regression thresholds in [`pipeline/tests/test_accuracy_regression.py`](../pipeline/tests/test_accuracy_regression.py).
