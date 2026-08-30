# RTAB-Map Baseline — Part 1 Feasibility & Setup Audit Report

## 1. Installation Details & System Audit

- **Environment**: GCP Compute Engine instance `spatial-benchmark` (n2-standard-16, 16 vCPUs, 64 GB RAM, Ubuntu 24.04.4 LTS x86_64).
- **ROS Dependency Decision**: Standalone C++ build (`rtabmap_core`, `rtabmap_app`, `rgbd_dataset`, `export`) built directly from source repo `https://github.com/introlab/rtabmap.git` (v0.23.11).
- **System Dependencies**:
  - `cmake` (v3.28.3), `build-essential` (gcc/g++ 13.2)
  - `libopencv-dev` (v4.6.0)
  - `libpcl-dev` (v1.14.0)
  - `liboctomap-dev`, `libsqlite3-dev`, `libeigen3-dev`, `libavcodec-dev`, `libvtk9-dev`
- **Compiled & Verified Binaries**:
  - `/usr/local/bin/rtabmap` (100% compiled & installed)
  - `/usr/local/bin/rtabmap-export` (401 KB binary)
  - `/usr/local/bin/rtabmap-rgbd_dataset` (135 KB binary)
  - `/usr/local/lib/librtabmap_core.so.0.23.11` (10.2 MB core library)
  - `/usr/local/lib/librtabmap_gui.so.0.23.11` (7.4 MB GUI library)

---

## 2. Input Format Compatibility & Audit Relabeling

- **Raw Capture Modalities**:
  - Color frames: `vga_wide/*.png` (640x480)
  - Depth frames: `vga_wide_depth/*.png` (16-bit PNG depth in millimeters)
  - Intrinsics: `lowres_wide_intrinsics/*.pincam` ($fx, fy, cx, cy$)
  - Camera Poses: `lowres_wide.traj` (4x4 matrix per timestamp)
- **Dataset CLI Tool Interface**:
  - `rtabmap-rgbd_dataset` consumes a sequence folder containing `rgb_sync/` and `depth_sync/` subdirectories.
  - Output database: `rtabmap.db` SQLite database file.
- **Relabeled Audit Claim**:
  - **Expected-but-unconfirmed assumption**: 16-bit depth PNGs (mm) map 1:1 into RTAB-Map's internal depth representation (1 unit = 1mm). This claim is documented from RTAB-Map source specification and will be confirmed empirically against visual odometry trajectory exports in Part 2.

---

## 3. Execution Verification & Connectivity Audit

- **SSH Connectivity**: Re-established and verified via `gcloud compute ssh --tunnel-through-iap --zone "us-central1-a" "spatial-benchmark" --project "ictai-2026"`.
- **Dataset Execution Test**:
  - Command: `rtabmap-rgbd_dataset --output <dir> --output_name rtabmap --quiet <seq_dir>`
  - Verified creation of `rtabmap.db` (106,496 bytes, exit code 0).
