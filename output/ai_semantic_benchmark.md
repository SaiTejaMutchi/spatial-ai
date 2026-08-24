# AI semantic benchmark

**Label:** `SYNTHETIC_RENDER_SEMANTIC_EVALUATION`

This is **not** official Structured3D. Official Structured3D requires a human-signed terms agreement and was not downloaded.

Success is not defined as Spatial-Grounded achieving higher raw classification accuracy. Higher raw accuracy is neither assumed nor required.

> Image-only AI may recognize the same visual object or condition. Spatial grounding does not claim to make the VLM intrinsically smarter. Its value is that the interpretation is bound to a stable physical entity and can be deterministically registered, measured, traced, and audited.

| Quantity | Value |
|---|---:|
| Cases | 4 |
| Spatial semantic accuracy (tiny, not generalizable) | 0.5 |
| Spatial candidate/surface binding rate | 1.0 |
| Geometry-owned quantity rate | 0.75 |
| Unsupported promotion count | 1 |
| `insufficient_evidence` count | 1 |
| Image-only metric entity count | 0 |
| Geometry mutation count | 0 |

Image-only AI can visually describe or classify an object; spatial grounding binds that interpretation to a specific metric entity and enables auditable geometry-owned quantities.

## case-door-001

- Held-out label (evaluator only): `door`
- Image-only kinds: `['door']` (boundToCandidate=False)
- Spatial class: `door` status=`supported` bound=True promoted=True

## case-window-001

- Held-out label (evaluator only): `window`
- Image-only kinds: `['window', 'window']` (boundToCandidate=False)
- Spatial class: `window` status=`supported` bound=True promoted=True

## case-occlusion-001

- Held-out label (evaluator only): `occlusion`
- Image-only kinds: `['none']` (boundToCandidate=False)
- Spatial class: `insufficient_evidence` status=`insufficient_evidence` bound=True promoted=False

## case-empty-001

- Held-out label (evaluator only): `scan_gap`
- Image-only kinds: `['uncertain']` (boundToCandidate=False)
- Spatial class: `door` status=`supported` bound=True promoted=True
