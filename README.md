# Spatial AI

[![PyPI version](https://img.shields.io/badge/pip%20install-spatial--ai-blue.svg)](https://pypi.org/project/spatial-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![MCP Server](https://img.shields.io/badge/MCP-Supported-success.svg)](spatial_ai/adapters/mcp_server.py)
[![Correctness Tests](https://img.shields.io/badge/correctness%20tests-334%20passed-brightgreen.svg)](docs/testing.md)
[![Accuracy MAE](https://img.shields.io/badge/accuracy%20MAE-2.82%20cm-blue.svg)](pipeline/eval/results/latest_run.json)

> **Give AI a memory of the physical world.**  
> Spatial AI turns mobile RGB-D and LiDAR captures into persistent physical entities (`wall-001`, `floor-001`) that AI agents can query, inspect, measure, and reason over.

---

## What Spatial AI Solves

Vision models see images. Text memory systems remember text. Neither naturally understands 3D physical space.

Spatial AI separates measurement from interpretation:

- **Deterministic 3D Geometry**: Gravity-aligned RANSAC plane extraction and Hough wall fitting own all metric quantities.
- **Persistent Entity Identity**: Observations belong to stable physical surfaces (`wall-001`, `floor-001`) across views and scans.
- **Pose-Registered Visual Stills**: Camera pose matrices ($T_{WC}$) link RGB video frames directly to physical entity IDs.
- **Bounded Multimodal AI Verifier**: AI models interpret visual evidence (identifying damage, openings, or room types) but are structurally forbidden from mutating geometry.

---

## Evaluation Summary & Characterized Systematic Error (`DEV_COMPLETE`)

Spatial AI is `DEV_COMPLETE` on public dev data. It is **not proven** on real-world property claims, and **no accuracy claim** is made.

The geometry pipeline is evaluated against terrestrial **FARO laser scanner ground truth** across multi-scene benchmark captures ($N=10$ live evaluated scenes):

| Scene ID | Visit ID | Dataset | Traced FARO Laser GT ($h_{\text{laser}}$) | Reconstructed Height ($h_{\text{pred}}$) | Absolute Error | Signed Error | Gate Status ($\le 1.50\text{ cm}$) | Ground Truth Verification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`47333462`** | `467138` | ARKitScenes | **`2.6300 m`** | **`2.5994 m`** | **`3.06 cm`** | **`-3.06 cm`** | Failed Gate (by 1.56 cm) | Re-derived from source |
| **`41418135`** | `416418` | ARKitScenes | **`2.4302 m`** | **`2.4121 m`** | **`1.81 cm`** | **`-1.81 cm`** | Failed Gate (by 0.31 cm) | Re-derived from source |
| **`41418155`** | `416407` | ARKitScenes | **`2.4303 m`** | **`2.4096 m`** | **`2.07 cm`** | **`-2.07 cm`** | Failed Gate (by 0.57 cm) | Re-derived from source |
| **`41418140`** | `416411` | ARKitScenes | **`2.4428 m`** | **`2.4245 m`** | **`1.83 cm`** | **`-1.83 cm`** | Failed Gate (by 0.33 cm) | Re-derived from source |
| **`42444474`** | `421069` | ARKitScenes | **`2.3249 m`** | **`2.3102 m`** | **`1.47 cm`** | **`-1.47 cm`** | **PASSED GATE** | Re-derived from source |
| **`42444499`** | `421065` | ARKitScenes | **`2.2944 m`** | **`2.2853 m`** | **`0.91 cm`** | **`-0.91 cm`** | **PASSED GATE** | Carried forward (unverifiable locally) |
| **`42444511`** | `421063` | ARKitScenes | **`2.3073 m`** | **`2.3296 m`** | **`2.23 cm`** | **`+2.23 cm`** | Failed Gate (by 0.73 cm) | Carried forward (unverifiable locally) |
| **`42444514`** | `421061` | ARKitScenes | **`2.1311 m`** | **`2.1246 m`** | **`0.65 cm`** | **`-0.65 cm`** | **PASSED GATE** | Carried forward (unverifiable locally) |
| **`42444519`** | `421060` | ARKitScenes | **`2.2972 m`** | **`2.3541 m`** | **`5.69 cm`** | **`+5.69 cm`** | Failed Gate (by 4.19 cm) | Carried forward (unverifiable locally) |
| **`42444574`** | `421062` | ARKitScenes | **`2.4578 m`** | **`2.3726 m`** | **`8.52 cm`** | **`-8.52 cm`** | Failed Gate (by 7.02 cm) | Carried forward (unverifiable locally) |

### Key Findings & Current Best Estimates ($N=10$)
- **Aggregate Accuracy ($N=10$)**: Evaluated across 10 scenes, height measurement achieves a Mean Absolute Error (MAE) of **2.82 cm** (median: **1.95 cm**, std: **2.32 cm**, p90: **5.69 cm**, min/max range: **0.65 cm – 8.52 cm**, 95% Bootstrap CI: `[1.57 cm, 4.38 cm]`). Gate pass rate ($\le 1.50\text{ cm}$) is **3/10 (30.0%)**.
- **Superseding Earlier N=5 Estimate**: The earlier reported five-scene error band of **1.47 cm – 3.06 cm** (mean 2.05 cm) represented a smaller initial sample. The wider $N=10$ distribution supersedes it as the current best estimate of system accuracy.
- **Ground Truth Verification Caveat**: Scenes 1–5 have ground truth independently re-derived from raw laser `.ply` files (`re-derived-from-source`). Scenes 6–10 have predictions re-derived from processed spatial models while ground truth heights are carried forward from prior recorded runs (`carried-forward-not-reverifiable`) due to local laser `.ply` file deletion.
- **Depth Scale Bias Hypothesis**: Fitted linear regression across $N=10$ scenes yields slope $m = -13.10\text{ cm/m}$ ($R^2 = 0.235$, $p = 0.156$). At $N=10$, this depth scale under-estimation trend is statistically non-significant ($p > 0.05$). Sample size sensitivity analysis indicates $N \in [24, 77]$ scenes are required to verify the effect.
- **Detailed Evaluation Reports**: For complete per-scene breakdowns, stratified error tables, failure mode characterization, and baseline specifications, see:
  - [`docs/error_analysis.md`](docs/error_analysis.md) — Detailed error analysis & multi-scene finding breakdown.
  - [`docs/limitations.md`](docs/limitations.md) — Unflinching limitations, empirical bounds, and open questions.
  - [`docs/architecture.md`](docs/architecture.md) — Structural guardrails and module ownership specifications.
  - [`docs/baselines.md`](docs/baselines.md) — External tool prerequisites (Apple RoomPlan, RTAB-Map, COLMAP).
  - [`docs/testing.md`](docs/testing.md) — Correctness suite structure and skip breakdown.

---

## Quickstart (Zero Downloads Required)

Install the Python package:

```bash
pip install spatial-ai
```

Load bundled spatial memory snapshots natively:

```python
from spatial_ai import Space

# Load a processed spatial capture (instant, 0 downloads required)
space = Space.load("samples/public_results/public-stray-8653a2142b/output")

# Inspect room metrics & floor area
print(space.dimensions)
# {'length_m': 5.90773, 'width_m': 4.192505, 'height_m': 2.671629, 'area_sq_m': 21.392849}

# Traverse persistent physical entities
for surface in space.surfaces:
    print(f"{surface.id}: {surface.type}")

# Inspect a specific wall & its registered visual evidence
wall = space.surface("wall-002")
print("Registered evidence stills:", len(wall.evidence))

# Ask AI questions grounded in physical space
assessment = space.ask("Which wall contains the window or opening?")
print(assessment.answer)
print(assessment.entity_ids)
```

---

## AI Agent & Framework Integrations (MCP / LangChain / CrewAI)

Spatial AI includes built-in typed adapters for Model Context Protocol (MCP) servers and AI agent frameworks:

```python
from spatial_ai.adapters import get_space, get_surface, measure, find_evidence
from spatial_ai.adapters.mcp_server import handle_mcp_tool_call

# 1. Query geometry-owned measurements for AI agents
metrics = measure("samples/public_results/public-stray-8653a2142b/output", surface_id="wall-002")
print(metrics)
# {'producer': 'deterministic_3d_geometry', 'surface_id': 'wall-002', 'dimensions': {'width_m': 3.95, 'height_m': 2.67}}

# 2. Invoke via standard Model Context Protocol (MCP) tool call
response = handle_mcp_tool_call("spatial_get_surface", {
    "space_path": "samples/public_results/public-stray-8653a2142b/output",
    "surface_id": "wall-002"
})
print(response["content"][0]["text"])
```

---

## Commands & Verification

- **Accuracy Evaluation CLI**:
  ```bash
  python3 -m pipeline.eval --full
  ```
- **Automated Correctness Tests** (334 passed, 106 skipped — see [`docs/testing.md`](docs/testing.md)):
  ```bash
  python3 -m pytest pipeline/tests/ -q
  ```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
