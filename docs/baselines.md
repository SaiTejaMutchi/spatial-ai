# Geometry Baselines and Technical Prerequisites

This document specifies the execution model and offline batch availability for external 3D perception frameworks when evaluated against mobile RGB-D and LiDAR benchmark datasets.

---

## Technical Audit & Offline Execution Capability

| Perception Engine | Primary Model | Offline Batch Capability | Technical Execution Prerequisite |
| :--- | :--- | :---: | :--- |
| **Spatial AI** | Source-neutral deterministic geometry pipeline | **Available** | Runs natively via Python pipeline (`pipeline/geometry/`). |
| **Apple RoomPlan** | iOS Swift `CapturedRoom` API | **Unavailable** | Requires a live ARKit hardware session on an iOS device with LiDAR. |
| **RTAB-Map** | RGB-D Graph-Based SLAM | **Conditional** | Requires a local C++ ROS build environment and sensor-specific trajectory calibration. |
| **COLMAP** | Dense Structure-from-Motion (SfM) | **Conditional** | Computes unscaled relative 3D point clouds requiring external metric scale references. |
| **Polycam / Matterport** | Cloud 3D Reconstruction SaaS | **Unavailable** | Operates as a proprietary cloud service with no public offline batch CLI. |

---

## Methodological Rules

1. **Unexecuted Baselines**: If an external framework cannot be executed offline in the local evaluation environment, its measured accuracy cells are left blank rather than filled with unverified estimates.
2. **Separation of Claims**: Metric measurement accuracy statistics and feature availability matrices are published in separate tables.
