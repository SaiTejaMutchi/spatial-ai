# Validation summary

Generated 2026-08-23T21:43:49+00:00.

Produced by `python3 -m pipeline.validation.run_ledger`. Every row is either an executed run or a recorded reason it could not run. No row is an estimate.

A pipeline output is not a validation result unless an independent expected value exists. Rows marked `NOT COMPARABLE` have output but no admissible reference, and their numbers must not be quoted as accuracy.

| ID | Dataset | Independent reference | Verdict | What it establishes |
| --- | --- | --- | --- | --- |
| V01 | Locally authored synthetic room fixtures | exact, authored before the run | `PASS` | Regression only |
| V02 | ARKitScenes | ARKitScenes FARO laser_scanner_point_clouds visit 467138 | `PARTIAL` | Development reference only - reference correspondence not established |
| V02B | Apple CA-1M / Cubify Anything (val split, held out) | CA-1M FARO-rendered depth and laser-registered poses | `PARTIAL` | Room height validated on 5 held-out captures against laser-derived reference: median 3.254 cm, worst 4.741 cm, 1 of 5 within the 1.5 cm gate |
| V03 | Redwood Indoor LiDAR-RGBD | High-end laser reference | `BLOCKED` | The dataset is not present on this machine |
| V04 | TUM RGB-D | Trajectory ground truth | `BLOCKED` | The dataset is not present on this machine |
| V05 | ScanNet | Reconstruction reference | `BLOCKED` | Optional, and the ledger says not to spend final-submission time here before V07 exists |
| V06 | Public Stray sample 8653a2142b | none found / none attached | `PASS` | PASS for interoperability: a real Stray export parses, normalizes, resolves its vertical axis, and reconstructs a closed room |
| V07 | Final self-captured iPhone Stray + tape | Independent tape measurements | `BLOCKED` | No self-captured iPhone export and no tape measurements exist yet |
| A01 | Synthetic opening semantic fixture | samples/ai_semantic_eval/manifest.json labels | `PARTIAL` | Image-only AI may recognize the same visual object or condition |
| A02 | ARKitScenes live AI review | none - the frames carry no semantic labels | `NOT COMPARABLE` | The model runs on real frames and produces findings, but no labels exist for them, so nothing is scored |
| A03 | Development condition fixture | samples/ai_condition_eval/labels.json | `PARTIAL` | Grounding mechanics only: the check is that an AI region binds to the correct surface and that any physical quantity stays geometry-produced |

## What is blocked, and why it matters

- **V03 — Redwood Indoor LiDAR-RGBD.** The dataset is not present on this machine.
- **V04 — TUM RGB-D.** The dataset is not present on this machine. This is the validation that would test pose and transform handling directly, which is where the repaired frame defect lived.
- **V05 — ScanNet.** Optional, and the ledger says not to spend final-submission time here before V07 exists.
- **V07 — Final self-captured iPhone Stray + tape.** No self-captured iPhone export and no tape measurements exist yet. This is the only end-to-end Track A accuracy benchmark, and it is P0.

## Claim boundary

- No room-dimension accuracy is validated on any real capture. The only independent reference attached anywhere is the ARKitScenes FARO height on the primary tuning scene, whose mobile-to-reference correspondence is not established, and which fails its gate.
- The Stray sample evidences interoperability only. It has no reference.
- The end-to-end accuracy benchmark (V07) does not exist yet, so no end-to-end accuracy claim is available to make.

