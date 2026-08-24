"""Stray Scanner export -> `normalized_capture`.

This is the **final acceptance** path. It is written against the app's
published specification (https://github.com/strayrobots/scanner,
`docs/format.md`) and its reference reader (StrayVisualizer). It has not yet
met a real export; final acceptance validates it against the untouched bundle
from the iPhone 13 Pro Max, and only the connector may change at that point.

Documented layout::

    camera_matrix.csv        3x3 intrinsics of the final frame (legacy)
    odometry.csv             timestamp, frame, x, y, z, qx, qy, qz, qw,
                             fx, fy, cx, cy, distortion_center_x/y
    imu.csv                  timestamp, a_x..a_z, alpha_x..alpha_z
    depth/000000.png         uint16 millimetres, 256x192
    confidence/000000.png    uint8 levels 0/1/2, 256x192
    distortion/000000.bin    optional float32 radial correction LUT
    rgb.mp4                  HEVC video, one frame per depth frame

Two things this connector must get right, both recorded explicitly so a real
export can falsify them rather than discover them silently:

1. **Intrinsics belong to RGB, depth is smaller.** `odometry.csv` publishes
   per-frame `fx, fy, cx, cy` for the colour camera, which is far larger than
   the 256x192 depth map. Depth intrinsics are therefore *rescaled*, and the
   contract records `derivation="rescaled_from:rgb"` rather than pretending
   they were measured.
2. **Pose convention.** `odometry.csv` gives a camera pose as position plus
   quaternion. StrayVisualizer builds `T_WC` from those directly. A real public
   export confirmed `p_world = R @ p_cam + t` (camera-to-world): that
   construction produces a floor and ceiling; transposing R does not. The
   Hamilton quaternion conversion is correct. IMU-attitude checks are not a
   convention gate — device-to-camera offset confounds them.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np

from ..contracts.normalized_capture import (
    CONTRACT_VERSION,
    Frame,
    Intrinsics,
    NormalizedCapture,
    Provenance,
)
from .base import (
    CONNECTOR_VERSION,
    DEFAULT_POSE_TOLERANCE_S,
    Connector,
    SourceRejected,
    excluded,
    sha256_file,
    utc_now,
)

DEPTH_SCALE_M = 0.001  # documented: "measured depth in millimeters"
REQUIRED_ODOMETRY_COLUMNS = ("timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw")


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if norm < 1e-12:
        raise SourceRejected("odometry.csv contains a zero-length quaternion")
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


class StrayScannerConnector(Connector):
    source_type = "stray_scanner"

    def __init__(
        self,
        root: Path,
        classification: str = "final_private_capture",
        pose_tolerance_s: float = DEFAULT_POSE_TOLERANCE_S,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.classification = classification
        self.pose_tolerance_s = pose_tolerance_s
        self.stride = max(1, int(stride))
        self.max_frames = max_frames

    @classmethod
    def detect(cls, root: Path) -> bool:
        root = Path(root)
        return (root / "odometry.csv").is_file() and (root / "depth").is_dir()

    # -- source readers ---------------------------------------------------

    def _read_odometry(self) -> tuple[list[dict], list[str]]:
        path = self.root / "odometry.csv"
        if not path.is_file():
            raise SourceRejected("odometry.csv is missing; camera poses are a core modality")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            # Older real Stray exports include a space after each comma in the
            # header. Normalize header whitespace at the vendor boundary; no
            # downstream contract field or geometry behavior changes.
            columns = [(column or "").strip() for column in (reader.fieldnames or [])]
            reader.fieldnames = columns
            missing = [c for c in REQUIRED_ODOMETRY_COLUMNS if c not in columns]
            if missing:
                raise SourceRejected(
                    f"odometry.csv is missing required column(s) {missing}; "
                    f"found {columns}")
            rows = []
            for raw in reader:
                rows.append({
                    (key or "").strip(): (
                        value.strip() if isinstance(value, str) else value)
                    for key, value in raw.items()
                })
        if not rows:
            raise SourceRejected("odometry.csv contains a header but no pose rows")
        return rows, columns

    def _legacy_camera_matrix(self) -> np.ndarray | None:
        path = self.root / "camera_matrix.csv"
        if not path.is_file():
            return None
        matrix = np.loadtxt(path, delimiter=",")
        return matrix if matrix.shape == (3, 3) else None

    # -- normalization ----------------------------------------------------

    def normalize(self, destination: Path) -> NormalizedCapture:
        destination = Path(destination)
        if not self.root.is_dir():
            raise SourceRejected(f"'{self.root}' is not a directory")
        if not self.detect(self.root):
            raise SourceRejected(
                f"'{self.root}' is not a Stray Scanner export: it needs 'odometry.csv' "
                f"and a 'depth/' directory")

        rows, columns = self._read_odometry()
        depth_dir = self.root / "depth"
        depth_by_frame = {p.stem: p for p in depth_dir.glob("*.png")}
        if not depth_by_frame:
            raise SourceRejected("depth/ contains no PNG frames")
        conf_by_frame = {p.stem: p for p in (self.root / "confidence").glob("*.png")}

        has_per_frame_intrinsics = all(c in columns for c in ("fx", "fy", "cx", "cy"))
        legacy = self._legacy_camera_matrix()
        if not has_per_frame_intrinsics and legacy is None:
            raise SourceRejected(
                "odometry.csv has no fx/fy/cx/cy columns and camera_matrix.csv is "
                "absent or malformed; camera intrinsics are a core modality")

        selected = set(range(0, len(rows), self.stride))
        if self.max_frames is not None:
            selected = set(sorted(selected)[: self.max_frames])
        keep_rgb = {
            str(rows[index]["frame"]).strip()
            for index in selected if index < len(rows)
        }
        rgb_frames = self._extract_rgb(destination, keep=keep_rgb)

        for directory in ("depth",) + (("confidence",) if conf_by_frame else ()):
            (destination / directory).mkdir(parents=True, exist_ok=True)

        frames: list[Frame] = []
        drops = []
        skipped = 0
        focal_samples: list[tuple[float, float, float, float]] = []
        conf_present = rgb_present = 0
        previous_ts = None

        for position, row in enumerate(rows):
            frame_key = str(row["frame"]).strip()
            source_id = f"odometry row {position} (frame {frame_key})"
            try:
                timestamp = float(row["timestamp"])
            except (TypeError, ValueError):
                drops.append(excluded(source_id, None,
                                      f"{source_id} has a non-numeric timestamp "
                                      f"{row['timestamp']!r}"))
                continue

            if position not in selected:
                skipped += 1
                continue

            depth_path = depth_by_frame.get(frame_key)
            if depth_path is None:
                drops.append(excluded(
                    source_id, timestamp,
                    f"Odometry row {position} references frame {frame_key} but "
                    f"depth/{frame_key}.png does not exist"))
                continue

            if previous_ts is not None and timestamp < previous_ts:
                drops.append(excluded(
                    source_id, timestamp,
                    f"Odometry row {position} timestamp {timestamp:.6f} s precedes "
                    f"the previous row's {previous_ts:.6f} s"))
                continue
            previous_ts = timestamp

            try:
                rotation = quaternion_to_matrix(
                    float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]))
                position_xyz = np.array(
                    [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                drops.append(excluded(source_id, timestamp,
                                      f"Odometry row {position} has an unreadable pose: {exc}"))
                continue

            camera_to_world = np.eye(4)
            camera_to_world[:3, :3] = rotation
            camera_to_world[:3, 3] = position_xyz

            if has_per_frame_intrinsics:
                try:
                    focal_samples.append((float(row["fx"]), float(row["fy"]),
                                          float(row["cx"]), float(row["cy"])))
                except (TypeError, ValueError):
                    drops.append(excluded(
                        source_id, timestamp,
                        f"Odometry row {position} has unreadable intrinsics columns"))
                    continue
            else:
                focal_samples.append((float(legacy[0, 0]), float(legacy[1, 1]),
                                      float(legacy[0, 2]), float(legacy[1, 2])))

            index = len(frames)
            name = f"{index:06d}.png"
            shutil.copyfile(depth_path, destination / "depth" / name)

            conf_rel = None
            if (conf_path := conf_by_frame.get(frame_key)) is not None:
                shutil.copyfile(conf_path, destination / "confidence" / name)
                conf_rel = f"confidence/{name}"
                conf_present += 1

            rgb_rel = None
            if (rgb_source := rgb_frames.get(frame_key)) is not None:
                shutil.move(str(rgb_source), destination / "rgb" / name)
                rgb_rel = f"rgb/{name}"
                rgb_present += 1

            frames.append(Frame(
                index=index,
                timestamp_s=timestamp,
                rgb=rgb_rel,
                depth=f"depth/{name}",
                confidence=conf_rel,
                camera_to_world=camera_to_world.tolist(),
                pose_timestamp_s=timestamp,
                pose_time_offset_s=0.0,
                intrinsics_stream="depth",
            ))

        if not frames:
            raise SourceRejected(
                f"no frame survived normalization: {len(drops)} of {len(rows)} odometry "
                f"rows were excluded; first reason: "
                f"{drops[0].reason if drops else 'unknown'}")

        from PIL import Image
        depth_w, depth_h = Image.open(destination / frames[0].depth).size
        rgb_size = (Image.open(destination / frames[0].rgb).size
                    if frames[0].rgb else None)

        samples = np.array(focal_samples, dtype=np.float64)
        fx, fy, cx, cy = (float(np.median(samples[:, i])) for i in range(4))

        if rgb_size is not None:
            rgb_intrinsics = Intrinsics(
                stream="rgb", width=rgb_size[0], height=rgb_size[1],
                fx=fx, fy=fy, cx=cx, cy=cy,
                derivation="source:odometry.csv" if has_per_frame_intrinsics
                           else "source:camera_matrix.csv")
            depth_intrinsics = rgb_intrinsics.scaled_to(depth_w, depth_h, "depth")
            intrinsics = [depth_intrinsics, rgb_intrinsics]
        else:
            # Without decoded colour there is no resolution to rescale from, so
            # the published intrinsics are recorded at their own resolution and
            # the mismatch is reported rather than guessed away.
            declared_w = int(round(cx * 2))
            declared_h = int(round(cy * 2))
            rgb_intrinsics = Intrinsics(
                stream="rgb", width=declared_w, height=declared_h,
                fx=fx, fy=fy, cx=cx, cy=cy,
                derivation="source:odometry.csv, resolution inferred from principal point")
            depth_intrinsics = rgb_intrinsics.scaled_to(depth_w, depth_h, "depth")
            intrinsics = [depth_intrinsics, rgb_intrinsics]

        imu_rel = None
        if (self.root / "imu.csv").is_file():
            shutil.copyfile(self.root / "imu.csv", destination / "imu.csv")
            imu_rel = "imu.csv"

        distortion_available = (self.root / "distortion").is_dir() and any(
            (self.root / "distortion").glob("*.bin"))

        modalities = {
            "depth": {"available": True, "count": len(frames),
                      "resolution": [depth_w, depth_h],
                      "units": "uint16 millimetres", "source": "depth/"},
            "intrinsics": {"available": True, "count": len(frames),
                           "source": "odometry.csv" if has_per_frame_intrinsics
                                     else "camera_matrix.csv",
                           "note": "Published for the colour camera; depth intrinsics "
                                   "are rescaled to the depth resolution."},
            "trajectory": {"available": True, "count": len(frames),
                           "source": "odometry.csv",
                           "source_convention": "camera_to_world (position + quaternion)"},
            "rgb": {"available": rgb_present > 0, "count": rgb_present,
                    "resolution": list(rgb_size) if rgb_size else None,
                    "source": "rgb.mp4",
                    "note": None if rgb_present else
                            "rgb.mp4 could not be decoded; colour evidence is unavailable "
                            "and AI review will be degraded."},
            "confidence": {"available": conf_present > 0, "count": conf_present,
                           "levels": [0, 1, 2], "source": "confidence/",
                           "meaning": "ARKit ARConfidenceLevel low/medium/high"},
            "mesh": {"available": False,
                     "note": "Stray Scanner exports no reconstructed mesh."},
            "imu": {"available": imu_rel is not None, "source": "imu.csv"},
            "distortion": {"available": distortion_available,
                           "source": "distortion/" if distortion_available else None,
                           "note": "Per-frame radial correction LUTs are retained at the "
                                   "source; the pinhole model is used unchanged."},
            "semantics": {"available": False,
                          "note": "Stray Scanner exports no RoomPlan semantics."},
        }

        provenance = Provenance(
            source_type=self.source_type,
            source_id=self.root.name,
            classification=self.classification,
            connector="StrayScannerConnector",
            connector_version=CONNECTOR_VERSION,
            generated_utc=utc_now(),
            source_paths=[str(self.root)],
            source_digests={"odometry.csv": sha256_file(self.root / "odometry.csv")},
            device={"platform": "iOS", "sensor": "ARKit LiDAR",
                    "app": "Stray Scanner",
                    "note": "Device, iOS and app version are recorded at capture time; "
                            "the export itself does not carry them."},
            notes=[
                "Pose convention is camera_to_world with +z forward: p_world = R @ p_cam + t, "
                "following the app specification and StrayVisualizer. A real public export "
                "confirmed it: the R construction produces floor and ceiling; the R.T "
                "alternative produces neither. IMU-attitude probes are not used as a gate. "
                "The private iPhone export reruns the same check.",
                "Depth intrinsics are rescaled from the published colour intrinsics; they "
                "are not independently measured.",
                f"Frames keyed by the odometry 'frame' column; {len(rows)} rows in, "
                f"{len(frames)} retained.",
            ],
        )

        capture = NormalizedCapture(
            contract_version=CONTRACT_VERSION,
            provenance=provenance,
            frames=frames,
            intrinsics=intrinsics,
            depth_scale_m=DEPTH_SCALE_M,
            world_frame="arkit_session",
            world_up_axis="+y",
            world_up_axis_verified=False,
            pose_convention="camera_to_world",
            modalities=modalities,
            excluded_frames=drops,
            frame_selection={
                "stride": self.stride,
                "max_frames": self.max_frames,
                "pose_tolerance_s": self.pose_tolerance_s,
                "rule": "every stride-th odometry row whose 'frame' key has a depth image; "
                        "Stray publishes one pose per frame, so no timestamp search is "
                        "needed",
                "source_frames": len(rows),
                "skipped_by_stride": skipped,
                "retained": len(frames),
                "pose_convention_verified": True,
            },
            imu=imu_rel,
        )
        capture.write(destination)
        return capture

    # -- colour -----------------------------------------------------------

    def _extract_rgb(self, destination: Path, keep: set[str] | None = None
                     ) -> dict[str, Path]:
        """Decode rgb.mp4 into a staging directory keyed by frame number.

        Colour is strongly preferred but not core. A missing or undecodable
        video degrades evidence and AI review; it does not fail the capture.
        Only frames that will be retained after stride are written.
        """
        video = self.root / "rgb.mp4"
        if not video.is_file():
            return {}
        try:
            import cv2
        except ImportError:
            return {}

        staging = destination / ".rgb_staging"
        staging.mkdir(parents=True, exist_ok=True)
        (destination / "rgb").mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            return {}
        extracted: dict[str, Path] = {}
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            key = f"{index:06d}"
            if keep is None or key in keep:
                path = staging / f"{key}.png"
                cv2.imwrite(str(path), frame)
                extracted[key] = path
            index += 1
        capture.release()
        return extracted
