# Real Stray Scanner integration samples

Raw captures live under `samples/stray/raw/` and are intentionally ignored by Git.
They validate the vendor connector but are not final-device evidence and must never be
labelled as a private iPhone acceptance capture.

## `8653a2142b`

- Source: `https://github.com/gisbi-kim/strayscanner-ros2bag-converter`
- Upstream archive member: `sample_data/8653a2142b.zip`
- Local input path: `samples/stray/raw/8653a2142b`
- Classification: public integration sample
- Contents: 412 depth frames, 412 confidence frames, `rgb.mp4`, legacy
  `camera_matrix.csv`, `odometry.csv`, and `imu.csv`
The connector result with `--stride 6` is `NORMALIZED_CAPTURE_VALID`. The first
geometry pass reported `GEOMETRY_GENERALIZATION_FAILURE` because the pose-scatter
heuristic overrode the declared +Y axis (camera looking all around the room makes
right-axis scatter choose a nearly horizontal direction).

Diagnostic `CONNECTOR_FRAME_BUG`, now fixed at ingestion:

- Poses are camera-to-world. `R @ p + t` produces floor + ceiling at 2.68 m;
  `R.T @ p + t` produces neither. Quaternion conversion is Hamilton-correct.
- IMU / mean-camera-up is not a gate (device-to-camera offset).
- Declaration `+Y` is kept when structure verifies (floor, ceiling, plausible
  height, short camera-path extent along that axis).
- With frozen geometry unchanged, the declared +Y frame yields floor, ceiling,
  ~10 walls, room height 2.664 m, floor/ceiling RMS 10.7 / 13.2 mm.

See `outputs/diagnostics/stray_8653a2142b/pose_convention_comparison.json`.

### Production result after the repair

Ingested at connector stride 6 and reconstructed by frozen geometry at stride 1, the
capture now produces a full result rather than failing:

| | |
| --- | --- |
| frame resolution | `verified` as `+y` (`declared_verified`), confirmed on both halves of the sampled frames independently |
| surfaces | 9, of which 2 inferred |
| observed perimeter | 71% |
| room height | 2.672 m |
| room length x width | 5.908 m x 4.193 m |
| floor area | 21.393 m² |
| openings | 0 resolved, 1 unresolved |
| AI review | completed, 9 findings over 3 registered evidence views |

Geometry stride matters here and is set per source in `service/scan_manager.py`: this
export is already connector-strided to 69 frames, and striding again starves the
estimators. The repair is recorded as post-freeze entry R1, with the evidence that the
ARKitScenes development benchmark did not move.

**No accuracy is claimed for this capture.** There is no tape or laser reference for it.
These numbers evidence that a real Stray export reconstructs, not that it measures
correctly.

The real legacy CSV contains spaces after header delimiters. The Stray connector strips
header whitespace at the vendor boundary; a regression test covers that behavior.

## `dinosaur`

- Published reference: `https://github.com/kekeblom/StrayVisualizer`
- Published URL: `https://stray-data.nyc3.digitaloceanspaces.com/datasets/dinosaur.tar.gz`
- Retrieval status on 2026-08-23: HTTP 404. No local copy is present and no test result
  is claimed.

The upstream ROS2 sample repository does not state a redistribution license for the
capture archive. Keep raw sample data local unless redistribution rights are confirmed.
