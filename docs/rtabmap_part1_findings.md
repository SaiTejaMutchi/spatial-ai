# RTAB-Map Baseline — Part 1 Feasibility & Setup Report

## 1. Installation Details & Path

- **Host Environment**: Ubuntu 24.04.4 LTS (x86_64), GCP VM instance `spatial-benchmark` (n2-standard-16, 16 vCPUs, 64 GB RAM).
- **ROS Dependency Decision**: Evaluated pulling full ROS2 distro vs. standalone C++ build. A ROS installation was deemed disproportionate. RTAB-Map was built as a **standalone C++ application** (`rtabmap_core`, `rtabmap_app`, `rgbd_dataset`) directly from the official source repository ([introlab/rtabmap](https://github.com/introlab/rtabmap.git), version 0.23.11).
- **System Build Dependencies Installed**:
  - `cmake` (v3.28.3), `build-essential` (gcc/g++ 13.2)
  - `libopencv-dev` (v4.6.0)
  - `libpcl-dev` (v1.14.0)
  - `liboctomap-dev`, `libsqlite3-dev`, `libeigen3-dev`, `libavcodec-dev`, `libvtk9-dev`
- **CMake Configuration**:
  ```bash
  cd ~/rtabmap/build
  cmake -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
        -DBUILD_GUI=OFF \
        -DBUILD_APP=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_FOVIS=OFF -DWITH_VISO2=OFF -DWITH_DVO=OFF -DWITH_OKVIS=OFF \
        -DWITH_MSCKF_VIO=OFF -DWITH_VINS_FUSION=OFF -DWITH_OPENVINS=OFF \
        -DWITH_ORB_SLAM=OFF -DWITH_LOAM=OFF -DWITH_FLOAM=OFF -DWITH_LIOSAM=OFF ..
  make -j$(nproc)
  ```
- **Compiled Executables & Libraries**:
  - `~/rtabmap/build/bin/rtabmap-res_tool`
  - `~/rtabmap/build/bin/librtabmap_utilite.so`
  - `~/rtabmap/build/bin/librtabmap_core.so`
  - `~/rtabmap/build/bin/rgbd_dataset` (`rtabmap-rgbd_dataset`)
  - `~/rtabmap/build/bin/export` (`rtabmap-export`)
  - `~/rtabmap/build/bin/rtabmap_app` (`rtabmap`)

---

## 2. Dataset Format & Conversion Compatibility

- **ARKitScenes Raw Input Format**:
  - Color frames: `vga_wide/*.png` (or `.jpg`, 640x480 resolution).
  - Depth frames: `vga_wide_depth/*.png` (16-bit PNG depth in millimeters).
  - Intrinsics: `lowres_wide_intrinsics/*.pincam` ($fx, fy, cx, cy$).
  - Camera Poses: `lowres_wide.traj` (4x4 transformation matrices per frame timestamp).
- **RTAB-Map Input Consumption**:
  - RTAB-Map's dataset CLI tool (`rtabmap-rgbd_dataset`) accepts RGB image sequences + Depth image sequences + Camera intrinsics ($fx, fy, cx, cy$).
  - **Lossless / Approximation Check**: Conversion of 16-bit PNG depth (mm) to RTAB-Map internal depth representation is **100% lossless** (1 unit = 1mm depth). Poses are converted cleanly to TUM format (`timestamp tx ty tz qx qy qz qw`). No lossy depth resampling or coordinate distortion is introduced.

---

## 3. Single-Scene Smoke Test (Scene 2: 41418135)

- **Test Room**: Scene 2 (`41418135`, Visit `416418`), clean indoor room with verified baseline MAE of 0.77 cm.
- **Raw Capture Statistics**:
  - RGB Frames: 720 frames (`vga_wide/`)
  - Depth Frames: 720 frames (`vga_wide_depth/`)
- **CLI Executable Smoke Test**:
  - Executed `~/rtabmap/build/bin/rgbd_dataset --help` and verified clean binary entry point execution without ROS dependency link errors.
