# PUBLIC DEVELOPMENT REFERENCE

> This is an independent public sanity check against survey-grade laser data. It is not a tape-measure benchmark, and it does not establish accuracy on any real capture.

- **Scene:** `47333462` (PRIMARY_TUNING)
- **Classification:** `public_development_fixture`
- **Reference:** Apple ARKitScenes laser_scanner_point_clouds
- **Geometry config:** `geometry_config_v0.1` (`fb7900299311d51c…`, frozen=True)
- **Reference config:** `reference_extraction_v0.1` (`53bc9586f5d978fd…`)
- **Generated:** 2026-08-23T21:43:48+00:00

## Comparison

| Measurement | Reference | Model | Signed error | Abs error | % error | Assignment gate | Result |
|---|---:|---:|---:|---:|---:|---|---|
| room_height | 2.635 m | 2.610 m | -0.025 m | 2.46 cm | 0.93% | at most 1.5 cm | **fail** |
| room_length | — | 5.476 m | — | — | — | at most 1% or 2 cm, whichever is larger | **not_comparable** |
| room_width | — | 3.178 m | — | — | — | at most 1% or 2 cm, whichever is larger | **not_comparable** |
| floor_area | — | 16.002 m² | — | — | — | at most 2% per room | **not_comparable** |

1 of 4 quantities were comparable; 0 passed and 1 failed the assignment gate.

## Why most quantities are not comparable

ARKitScenes publishes no transform between the ARKit session frame and the FARO visit frame, so this quantity cannot be compared without first solving registration. Only floor-to-ceiling height survives that gap, because it is invariant under any horizontal rigid motion between two gravity-aligned frames.

## Does the reference describe the same room?

**No — correspondence could not be established.** The reference visit supports 6 plausible floor-to-ceiling pairings spanning 6.0 cm. Its ceiling surfaces extend well beyond the footprint the pipeline reconstructed, which is what a whole-dwelling survey looks like. The reference is therefore a building storey statistic, not a measurement of the captured room.

| Reference floor | Reference ceiling | Separation | Min support |
|---:|---:|---:|---:|
| -51.1238 m | -48.4888 m | **2.6350 m** | 64,091 |
| -51.1238 m | -48.4688 m | **2.6550 m** | 64,091 |
| -51.1238 m | -48.4538 m | **2.6700 m** | 21,997 |
| -51.0988 m | -48.4888 m | **2.6100 m** | 21,991 |
| -51.0988 m | -48.4688 m | **2.6300 m** | 21,991 |
| -51.0988 m | -48.4538 m | **2.6450 m** | 21,991 |

Because the reference's own ambiguity is wider than the discrepancy, it cannot adjudicate that discrepancy, and no geometry parameter was tuned toward it. The comparison is retained as a magnitude check: it shows the pipeline lands within the range of storey heights present in the building, and nothing stronger.

## Limitations

- Only floor-to-ceiling height is comparable; see the note on each non-comparable row.
- The FARO reference covers the whole visit, which may include rooms the 60-second video never entered. Without registration it cannot be confirmed that the compared storey is the same room the pipeline reconstructed, so this is a dwelling storey-height check rather than a same-room comparison.
- One scene with one comparable quantity supports no generalization claim.
- Confidence labels elsewhere in the model remain uncalibrated heuristics; nothing here calibrates them.

## Secondary validation scene

The secondary scene runs once with the same frozen configuration and is never used to change a parameter.

- `41418135` — room height 2.427 m, floor area 20.473 m², config `fb7900299311d51c…`. No FARO reference was retrieved for this visit, by design: the plan caps public validation at one supported quantitative comparison. This scene evidences that the frozen configuration runs unchanged on unseen data, and nothing more.
