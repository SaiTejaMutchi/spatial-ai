# Evidence index

Every claim in this repository resolves to a committed artifact, an enforcing
test, or an explicit statement that the claim is not yet supported. This page
is the index: what a claim rests on, and where to check it.

| Claim | Evidence | Where it's enforced |
| :--- | :--- | :--- |
| Height accuracy, N=10 scenes, MAE 2.82 cm | [`pipeline/eval/results/latest_run.json`](../pipeline/eval/results/latest_run.json), reproduced via [`docs/REPRODUCE.md`](REPRODUCE.md) | [`pipeline/eval/evaluator.py`](../pipeline/eval/evaluator.py) |
| Depth scale bias is a hypothesis, not confirmed (p=0.156 at N=10) | [`docs/error_analysis.md`](error_analysis.md#L24-L35) | Statistical test in the same evaluation run |
| Deterministic geometry owns all metric quantities | [`docs/architecture.md`](architecture.md#L36-L39) | [`pipeline/tests/test_sdk.py`](../pipeline/tests/test_sdk.py), [`pipeline/tests/test_geometry_model.py`](../pipeline/tests/test_geometry_model.py) |
| The AI verifier cannot mutate geometry | [`docs/architecture.md`](architecture.md#L51-L68) | [`pipeline/tests/test_ai_verifier.py:L45`](../pipeline/tests/test_ai_verifier.py#L45), [`schema/visible_condition.schema.json`](../schema/visible_condition.schema.json) |
| 334 passed, 106 skipped correctness tests | [`docs/testing.md`](testing.md#L15-L24) | `python3 -m pytest pipeline/tests/ -q` |
| Baselines (RoomPlan, RTAB-Map, COLMAP, Polycam): only what actually ran is reported | [`docs/baselines.md`](baselines.md#L9-L16) | Unrun baselines are marked unexecuted, not estimated |
| Dataset licensing and provenance (ARKitScenes, CA-1M) | [`docs/data_provenance.md`](data_provenance.md) | SHA256 checksums recorded alongside each source |
| Where the evidence stops | [`docs/limitations.md`](limitations.md) | States exactly what N=10 does and does not support |

Ground rules this repository holds itself to:

- Every published number traces to a committed result file, never to memory or
  a chat transcript.
- A missing or unrun baseline is reported as missing, never filled in with an
  estimate.
- Negative and null results (the 70% gate failure rate, the non-significant
  bias hypothesis) get the same visibility as positive ones.
- An architectural guarantee is only claimed once a test enforces it.
