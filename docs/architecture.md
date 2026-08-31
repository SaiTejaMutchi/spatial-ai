# Spatial AI Architecture Specification

## Core Thesis

> **Sensors observe. Geometry measures. AI interprets. Registration connects.**

Spatial AI enforces a strict separation of concerns between physical measurement and semantic interpretation.

```text
Raw Capture (ARKitScenes / Stray / Unity)
  │
  ▼
pipeline/connectors/         ──────▶ Normalizes sensors to NormalizedCapture contract
  │
  ▼
pipeline/geometry/           ──────▶ Deterministic RANSAC, Hough wall fit & room metrics
  │
  ▼
spatial_model.json           ──────▶ Schema-enforced canonical System of Record (v0.2)
  │
  ├──▶ pipeline/rendering/   ──────▶ 2D SVG metric floorplan & 3D OBJ model
  ├──▶ pipeline/evidence/    ──────▶ Pose matrices (T_WC) register RGB stills to surface IDs
  └──▶ pipeline/ai/          ────── intended ONLY for semantic classification & entity binding
                                    (STRUCTURALLY FORBIDDEN from mutating metrics)
```

---

## Subsystem Boundaries & Ownership

### 1. Ingestion & Connectors Layer (`pipeline/connectors/`)
- **Ownership**: Ingests raw device-specific archives (Stray Scanner `.mp4`/`.csv`, ARKitScenes directories, Unity OBJ bundles).
- **Contract Output**: Validated `NormalizedCapture` directory containing `manifest.json`, intrinsics, depth streams, and pose trajectories ($T_{WC}$).
- **Forbidden Actions**: Performing 3D plane extraction or emitting final spatial entity models.

### 2. Deterministic Geometry Layer (`pipeline/geometry/`)
- **Ownership**: Owns 100% of metric quantities (room height, length, width, floor area, surface bounding boxes).
- **Execution**: Gravity-aligned RANSAC plane fitting, 2D Hough line projection, minimum enclosing bounding boxes, and 2D convex polygon footprint closure.
- **Forbidden Actions**: Depending on external network calls or LLM prompts.

### 3. System of Record (`schema/spatial_model.schema.json`)
- **Ownership**: Canonical `spatial_model.json` structure.
- **Enforcement**: Schema validation ([`schema/spatial_model.schema.json`](../schema/spatial_model.schema.json)) asserting surface entity IDs (`wall-001`), confidence bounds, and observation states (`directly_observed`, `inferred`).

### 4. Registration Layer (`pipeline/evidence/`)
- **Ownership**: Pose matrix transformation ($T_{WC}$) mapping camera frustums to physical surface IDs.
- **Contract Output**: `views[]` array attaching specific RGB frame indices and bounding frustums to surface IDs.

### 5. Multimodal AI Verifier Layer (`pipeline/ai/`)
- **Ownership**: Semantic interpretation (identifying doors, windows, condition stains, or damage).
- **Structural Guardrail**: The AI verifier output schema ([`schema/visible_condition.schema.json`](../schema/visible_condition.schema.json)) **contains zero dimension fields**. The verifier is structurally forbidden from emitting or mutating metric dimensions (`length_m`, `width_m`, `height_m`, `area_sq_m`).

---

## Boundary Enforcement & Guardrail Audit

| Potential Leak | Risk Level | Structural Guardrail Location | Enforcement Mechanism |
| :--- | :---: | :--- | :--- |
| **AI LLM Mutates Room Dimensions** | **Prevented** | [`pipeline/ai/verifier.py`](../pipeline/ai/verifier.py) & [`schema/visible_condition.schema.json`](../schema/visible_condition.schema.json) | VLM JSON schema lacks dimension fields; pipeline rejects extra fields |
| **Geometry Overwritten by AI Candidate** | **Prevented** | [`spatial_ai/space.py`](../spatial_ai/space.py) | `Surface.dimensions` reads directly from geometry-written data dict |
| **Un-grounded Quantities in SDK** | **Prevented** | [`spatial_ai/surface.py`](../spatial_ai/surface.py) | `canonical_dimensions` exposes raw, un-modified geometry record |

---

## Automated Guardrail Test

The guardrail is validated in [`pipeline/tests/test_ai_verifier.py`](../pipeline/tests/test_ai_verifier.py#L45):
If a mocked AI verifier returns a response attempting to alter wall dimensions or room metrics, the pipeline verifies that `spatial_model.json` metric measurements remain **byte-identical** before and after verifier execution.
