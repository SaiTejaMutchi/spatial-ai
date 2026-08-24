"""Tiny synthetic captures for connector and contract tests.

Real fixtures prove the happy path. Deliberately broken synthetic ones prove
the failure paths, which is where an integration actually goes wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def write_depth(path: Path, width: int = 16, height: int = 12, value_mm: int = 1500) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((height, width), value_mm, dtype=np.uint16)).save(path)


def write_rgb(path: Path, width: int = 16, height: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(path)


def write_confidence(path: Path, width: int = 16, height: int = 12, level: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((height, width), level, dtype=np.uint8)).save(path)


def make_arkitscenes(
    root: Path,
    scene_id: str = "12345678",
    frames: int = 6,
    pose_rate: int = 2,
    with_confidence: bool = True,
    with_rgb: bool = True,
    with_intrinsics: bool = True,
    intrinsics_text: str | None = None,
    traj_lines: list[str] | None = None,
    depth_size: tuple[int, int] = (16, 12),
    rgb_size: tuple[int, int] | None = None,
    depth_value_mm: int = 1500,
) -> Path:
    """A minimal but structurally faithful ARKitScenes raw sequence."""
    scene = root / scene_id
    scene.mkdir(parents=True, exist_ok=True)
    rgb_size = rgb_size or depth_size

    timestamps = [1000.0 + i * 0.1 for i in range(frames)]
    for i, ts in enumerate(timestamps):
        name = f"{scene_id}_{ts:.3f}"
        write_depth(scene / "lowres_depth" / f"{name}.png", *depth_size,
                    value_mm=depth_value_mm)
        if with_rgb:
            write_rgb(scene / "lowres_wide" / f"{name}.png", *rgb_size)
        if with_confidence:
            write_confidence(scene / "confidence" / f"{name}.png", *depth_size)
        if with_intrinsics:
            directory = scene / "lowres_wide_intrinsics"
            directory.mkdir(parents=True, exist_ok=True)
            default = (f"{rgb_size[0]} {rgb_size[1]} 200.0 200.0 "
                       f"{rgb_size[0] / 2:.1f} {rgb_size[1] / 2:.1f}")
            (directory / f"{name}.pincam").write_text(intrinsics_text or default)

    if traj_lines is None:
        traj_lines = []
        for i, ts in enumerate(timestamps):
            if i % pose_rate:
                continue
            # Identity rotation, camera translating along +x by 0.1 m per pose.
            # These are WORLD-TO-CAMERA columns, matching the real format.
            traj_lines.append(f"{ts:.8f} 0.0 0.0 0.0 {-0.1 * i:.6f} 0.0 0.0")
    (scene / "lowres_wide.traj").write_text("\n".join(traj_lines) + "\n")
    return scene


def make_stray(
    root: Path,
    name: str = "abcd1234",
    frames: int = 6,
    with_confidence: bool = True,
    with_imu: bool = True,
    header: str | None = None,
    rows: list[str] | None = None,
    depth_size: tuple[int, int] = (16, 12),
) -> Path:
    """A minimal Stray Scanner export, per the app's published format."""
    scan = root / name
    scan.mkdir(parents=True, exist_ok=True)

    default_header = ("timestamp,frame,x,y,z,qx,qy,qz,qw,fx,fy,cx,cy,"
                      "distortion_center_x,distortion_center_y")
    lines = [header if header is not None else default_header]
    if rows is None:
        rows = []
        for i in range(frames):
            rows.append(f"{1000.0 + i * 0.05:.6f},{i:06d},{0.05 * i:.6f},0.0,0.0,"
                        f"0.0,0.0,0.0,1.0,1400.0,1400.0,960.0,720.0,,")
    lines.extend(rows)
    (scan / "odometry.csv").write_text("\n".join(lines) + "\n")
    (scan / "camera_matrix.csv").write_text("1400.0,0.0,960.0\n0.0,1400.0,720.0\n0.0,0.0,1.0\n")

    for i in range(frames):
        write_depth(scan / "depth" / f"{i:06d}.png", *depth_size)
        if with_confidence:
            write_confidence(scan / "confidence" / f"{i:06d}.png", *depth_size)
    if with_imu:
        (scan / "imu.csv").write_text("timestamp,a_x,a_y,a_z,alpha_x,alpha_y,alpha_z\n"
                                      "1000.0,0.0,-9.81,0.0,0.0,0.0,0.0\n")
    return scan


def make_unity_obj(root: Path, name: str = "scan.obj", vertices: int = 4) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    lines = [f"v {i} {i * 0.5} {i * 0.25}" for i in range(vertices)]
    if vertices >= 3:
        lines.append("f 1 2 3")
    path.write_text("\n".join(lines) + "\n")
    return path
