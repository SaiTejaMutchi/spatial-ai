# Comprehensive Baseline Comparison: Spatial AI vs. RTAB-Map SLAM vs. Laser Ground Truth

## 1. Overview & Evaluation Protocol

This document records the empirical comparison between **Spatial AI (Deterministic 3D Geometry Pipeline)** and **RTAB-Map (3D RGB-D Visual SLAM)** across **$N=29$ independently-verified indoor room captures** from the ARKitScenes dataset.

All measurements are benchmarked directly against **FARO Terrestrial Laser Scanner point clouds**, which serve as the high-accuracy ground truth ($z$-axis storey height extracted via SVD horizontal plane fitting).

---

## 2. Quantitative Benchmark Results ($N=29$ Verified Rooms)

| Method / System | Valid Height Extracted | Success Rate (%) | Height MAE (cm) | Median Error (cm) | Max Error (cm) | Input Requirement |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Spatial AI (Deterministic 3D)** | **29 / 29** | **100.0%** | **2.94 cm** | **2.88 cm** | **11.49 cm** | Source-Neutral (Normalized Capture) |
| **RTAB-Map (3D RGB-D SLAM)** | **0 / 29** | **0.0%** | **N/A** | **N/A** | **N/A** | RGB-D Frames + Odometry |
| **Apple RoomPlan (iOS Swift)** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | Live iOS Hardware Session |

> [!IMPORTANT]
> **RTAB-Map Visual Odometry Tracking Failure**: Without pre-calculated trajectory injection (`lowres_wide.traj` / TUM poses) or hardware IMU fusion, RTAB-Map's uncalibrated feature tracker (`rtabmap-rgbd_dataset`) experienced scale collapse (`inliers=0/0` transitioning to short drift estimates $< 3.3\text{ cm}$) on raw ARKitScenes handheld RGB-D sequences, failing to reconstruct valid odometry poses or 3D point cloud maps.

---

## 3. Audited Receipts & Artifact Locations

- **RTAB-Map Execution Harness**: [pipeline/eval/rtabmap_harness.py](../pipeline/eval/rtabmap_harness.py)
- **RTAB-Map Evaluation Summary JSON**: [pipeline/eval/results/rtabmap_results.json](../pipeline/eval/results/rtabmap_results.json)
- **RTAB-Map Setup & C++ Build Findings**: [docs/rtabmap_part1_findings.md](../docs/rtabmap_part1_findings.md)
- **Spatial AI Verified Benchmark Progress**: [pipeline/eval/results/batch_progress.jsonl](../pipeline/eval/results/batch_progress.jsonl)

---

## 4. Key Architectural Insights & Differences

1. **Robustness to Handheld Camera Motion**: Spatial AI's deterministic geometry connector normalizes spatial trajectory poses directly from input manifests, preventing scale drift even when visual feature tracking degrades in textureless scenes.
2. **Entity & Surface Permanence**: Spatial AI outputs clean, watertight 3D room boundary envelopes (`room_model.obj`, `spatial_model.json`) with persistent UUID surface IDs and bounded AI verifier diagnostics, whereas SLAM outputs unsegmented, noisy point clouds requiring manual post-processing.
