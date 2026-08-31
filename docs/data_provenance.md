# Dataset provenance, access terms and licensing

This document specifies the origin, license terms, access protocols, redistribution boundaries, and cryptographic SHA256 checksums for all mobile RGB-D and terrestrial laser ground truth benchmark datasets utilized in Spatial AI.

---

## 1. Dataset Overview & Provenance Table

| Dataset Name | Origin / Primary Citation | Dataset License | Redistribution Scope | Primary Access URL |
| :--- | :--- | :--- | :--- | :--- |
| **ARKitScenes** | Apple Inc. (Baruch et al., NeurIPS 2021) | **Apple ARKitScenes License** (Research / Non-commercial) | Code/scripts only; sample model snapshots bundled | [github.com/apple/ARKitScenes](https://github.com/apple/ARKitScenes) |
| **CA-1M** | Internal / Spatial AI Validation Collection | **CC BY 4.0** | Processed spatial models bundled; raw RGB-D videos on request | [github.com/SaiTejaMutchi/spatial-ai](https://github.com/SaiTejaMutchi/spatial-ai) |
| **Stray Scanner** | Stray Robots (Open Source Mobile RGB-D) | **MIT License** | Sample normalized capture bundled in `samples/` | [github.com/stray-robots/stray_scanner](https://github.com/stray-robots/stray_scanner) |

---

## 2. Cryptographic Checksums (Input Verification)

To verify that local input data matches the benchmark reference data, callers can compute SHA256 checksums of the bundled spatial model snapshots:

| Dataset Capture ID | Local File Path | SHA256 Checksum |
| :--- | :--- | :--- |
| **`public-stray-8653a2142b`** | `samples/public_results/public-stray-8653a2142b/output/spatial_model.json` | `ab0888e3b6dcfd0da0bbce7f93d34df24a6c722db2274161447299e373d88ba1` |
| **`public-iphone-e30fe3cae4`** | `samples/public_results/public-iphone-e30fe3cae4/output/spatial_model.json` | `34de794bb0a337e0ee8e8b337dfcbd02eef96feb3af6bc14b95877451104df08` |
| **`dev_47333462`** | `outputs/dev_47333462/spatial_model.json` | `fb7900299311d51c610aecf9f42ffa1332297fe533ba11c05ed8c34ee959e0b0` |

---

## 3. Data Access & Download Protocols

1. **Bundled Snapshots (Zero Downloads Required)**:
   - The repository includes pre-computed, schema-compliant `spatial_model.json` snapshots under `samples/public_results/`.
   - Allows instant execution of python SDK code (`Space.load(...)`), MCP tool calls, and baseline evaluation without downloading external archives.

2. **Full Raw Capture Archives (Multi-Gigabyte Downloads)**:
   - Raw video archives (`rgb.mp4`, `depth/`, `confidence/`, high-resolution FARO ground truth point clouds) are **not** committed to git to maintain a lightweight clone footprint.
   - Instructions for fetching raw ARKitScenes captures are provided in `pipeline/eval/fetch.py`.
