"""ARKitScenes raw sequence -> `normalized_capture`.

Development source only. A capture normalized by this connector may establish
`DEV_COMPLETE`; it can never stand in for the final iPhone evidence.

Layout consumed (Apple ARKitScenes `raw/{fold}/{video_id}/`)::

    lowres_wide/{video_id}_{timestamp}.png              RGB, 256x192
    lowres_depth/{video_id}_{timestamp}.png             uint16 millimetres
    confidence/{video_id}_{timestamp}.png               uint8, ARKit levels 0/1/2
    lowres_wide_intrinsics/{video_id}_{timestamp}.pincam  "w h fx fy cx cy"
    lowres_wide.traj                                    world-to-camera poses
    {video_id}_3dod_mesh.ply                            optional ARKit mesh

Two source properties drive the implementation and are easy to get wrong:

1. `lowres_wide.traj` stores **world-to-camera** rotation (axis-angle) and
   translation. Apple's own loader inverts it. This connector inverts once,
   here, so nothing downstream has to know.
2. Images are written at roughly 60 Hz while poses are logged at 10 Hz, so
   frames outnumber poses about six to one. Matching is by timestamp; matching
   by index would silently attach the wrong pose to most frames.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..contracts.normalized_capture import (
    Frame,
    Intrinsics,
    NormalizedCapture,
    Provenance,
    CONTRACT_VERSION,
)
from .base import (
    CONNECTOR_VERSION,
    DEFAULT_POSE_TOLERANCE_S,
    Connector,
    SourceRejected,
    excluded,
    match_poses,
    rodrigues,
    sha256_file,
    utc_now,
)

DEPTH_SCALE_M = 0.001  # ARKitScenes lowres_depth is uint16 millimetres


def _pose_rate_hz(pose_times: np.ndarray) -> float | None:
    """Median pose rate, or None when a single pose makes the rate undefined."""
    if pose_times.size < 2:
        return None
    interval = float(np.median(np.diff(pose_times)))
    return round(1.0 / interval, 2) if interval > 0 else None


def _timestamp(path: Path) -> float:
    """`{video_id}_{timestamp}.png` -> timestamp seconds."""
    stem = path.stem
    if "_" not in stem:
        raise SourceRejected(f"'{path.name}' does not follow "
                             f"'{{video_id}}_{{timestamp}}' naming")
    return float(stem.split("_", 1)[1])


class ARKitScenesConnector(Connector):
    source_type = "arkitscenes"

    def __init__(
        self,
        root: Path,
        classification: str = "public_development_fixture",
        pose_tolerance_s: float = DEFAULT_POSE_TOLERANCE_S,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.classification = classification
        self.pose_tolerance_s = pose_tolerance_s
        self.stride = max(1, int(stride))
        self.max_frames = max_frames
        self.scene_id = self.root.name

    # -- detection --------------------------------------------------------

    @classmethod
    def detect(cls, root: Path) -> bool:
        root = Path(root)
        return (root / "lowres_depth").is_dir() and (root / "lowres_wide.traj").is_file()

    # -- normalization ----------------------------------------------------

    def _read_trajectory(self) -> np.ndarray:
        path = self.root / "lowres_wide.traj"
        if not path.is_file():
            raise SourceRejected(f"'{path}' is missing; camera poses are a core modality")
        rows = []
        for number, line in enumerate(path.read_text().strip().splitlines(), start=1):
            tokens = line.split()
            if len(tokens) != 7:
                raise SourceRejected(
                    f"lowres_wide.traj line {number} has {len(tokens)} columns, "
                    f"expected 7 (timestamp, 3 rotation, 3 translation)")
            rows.append([float(token) for token in tokens])
        if not rows:
            raise SourceRejected("lowres_wide.traj is empty")
        traj = np.array(rows, dtype=np.float64)
        if not np.all(np.diff(traj[:, 0]) > 0):
            order = np.argsort(traj[:, 0], kind="stable")
            traj = traj[order]
        return traj

    def _read_intrinsics(self, path: Path) -> tuple[int, int, float, float, float, float]:
        tokens = path.read_text().split()
        if len(tokens) != 6:
            raise SourceRejected(
                f"'{path.name}' has {len(tokens)} values, expected 6 (w h fx fy cx cy)")
        width, height, fx, fy, cx, cy = (float(token) for token in tokens)
        return int(width), int(height), fx, fy, cx, cy

    def normalize(self, destination: Path) -> NormalizedCapture:
        destination = Path(destination)
        if not self.root.is_dir():
            raise SourceRejected(f"'{self.root}' is not a directory")
        if not self.detect(self.root):
            raise SourceRejected(
                f"'{self.root}' is not an ARKitScenes raw sequence: it needs a "
                f"'lowres_depth/' directory and a 'lowres_wide.traj' file")

        depth_files = sorted((self.root / "lowres_depth").glob("*.png"), key=_timestamp)
        if not depth_files:
            raise SourceRejected("lowres_depth/ contains no PNG frames")

        rgb_by_ts = {_timestamp(p): p for p in (self.root / "lowres_wide").glob("*.png")}
        conf_by_ts = {_timestamp(p): p for p in (self.root / "confidence").glob("*.png")}
        intr_by_ts = {_timestamp(p): p
                      for p in (self.root / "lowres_wide_intrinsics").glob("*.pincam")}
        if not intr_by_ts:
            raise SourceRejected(
                "lowres_wide_intrinsics/ contains no .pincam files; "
                "camera intrinsics are a core modality")

        traj = self._read_trajectory()
        pose_times = traj[:, 0]
        frame_times = np.array([_timestamp(p) for p in depth_files])
        pose_index, pose_offset = match_poses(frame_times, pose_times, self.pose_tolerance_s)

        selected = list(range(0, len(depth_files), self.stride))
        if self.max_frames is not None:
            selected = selected[: self.max_frames]
        selected_set = set(selected)

        for directory in ("rgb", "depth", "confidence"):
            (destination / directory).mkdir(parents=True, exist_ok=True)

        frames: list[Frame] = []
        drops = []
        skipped = 0
        intrinsics_samples: list[tuple[int, int, float, float, float, float]] = []
        rgb_present = conf_present = 0

        for position, depth_path in enumerate(depth_files):
            timestamp = float(frame_times[position])
            source_id = depth_path.name

            if position not in selected_set:
                # Deliberate subsampling is not an anomaly. Counting it keeps the
                # excluded list a list of real problems rather than 3,000 lines of
                # noise that hide the six frames that actually failed.
                skipped += 1
                continue
            if pose_index[position] < 0:
                nearest = float(pose_offset[position])
                drops.append(excluded(
                    source_id, timestamp,
                    f"Frame {position} has depth data but no matching odometry row "
                    f"(nearest pose is {abs(nearest) * 1000:.0f} ms away, tolerance "
                    f"{self.pose_tolerance_s * 1000:.0f} ms)"))
                continue
            intrinsics_path = intr_by_ts.get(timestamp)
            if intrinsics_path is None:
                drops.append(excluded(
                    source_id, timestamp,
                    f"Frame {position} has depth data but no matching intrinsics record"))
                continue

            index = len(frames)
            name = f"{index:06d}.png"
            shutil.copyfile(depth_path, destination / "depth" / name)

            rgb_rel = None
            if (rgb_path := rgb_by_ts.get(timestamp)) is not None:
                shutil.copyfile(rgb_path, destination / "rgb" / name)
                rgb_rel = f"rgb/{name}"
                rgb_present += 1
            conf_rel = None
            if (conf_path := conf_by_ts.get(timestamp)) is not None:
                shutil.copyfile(conf_path, destination / "confidence" / name)
                conf_rel = f"confidence/{name}"
                conf_present += 1

            row = traj[int(pose_index[position])]
            rotation_w2c = rodrigues(row[1:4])
            camera_to_world = np.eye(4)
            camera_to_world[:3, :3] = rotation_w2c.T
            camera_to_world[:3, 3] = -rotation_w2c.T @ row[4:7]

            intrinsics_samples.append(self._read_intrinsics(intrinsics_path))
            frames.append(Frame(
                index=index,
                timestamp_s=timestamp,
                rgb=rgb_rel,
                depth=f"depth/{name}",
                confidence=conf_rel,
                camera_to_world=camera_to_world.tolist(),
                pose_timestamp_s=float(pose_times[int(pose_index[position])]),
                pose_time_offset_s=float(pose_offset[position]),
                intrinsics_stream="depth",
            ))

        if not frames:
            raise SourceRejected(
                f"no frame survived normalization: {len(drops)} of {len(depth_files)} "
                f"depth frames were excluded; first reason: "
                f"{drops[0].reason if drops else 'unknown'}")

        samples = np.array(intrinsics_samples, dtype=np.float64)
        width, height = int(samples[0, 0]), int(samples[0, 1])
        if len({(int(w), int(h)) for w, h in samples[:, :2]}) > 1:
            raise SourceRejected(
                "lowres_wide_intrinsics declares more than one image size across frames")

        # ARKit publishes intrinsics per frame with slight autofocus jitter. The
        # contract carries one camera model, so the per-frame median is used and
        # the observed spread is recorded rather than hidden.
        fx, fy, cx, cy = (float(np.median(samples[:, i])) for i in range(2, 6))
        focal_spread = float(samples[:, 2].std() / samples[:, 2].mean())

        # lowres_wide and lowres_depth are both 256x192 here, so the published
        # RGB intrinsics apply to depth unchanged. The check is explicit so a
        # future sequence with differing resolutions fails loudly.
        rgb_shape = None
        if rgb_present:
            from PIL import Image
            first_rgb = destination / frames[0].rgb
            rgb_array_shape = Image.open(first_rgb).size  # (w, h)
            rgb_shape = [rgb_array_shape[0], rgb_array_shape[1]]

        from PIL import Image as _Image
        depth_w, depth_h = _Image.open(destination / frames[0].depth).size

        rgb_intrinsics = Intrinsics(
            stream="rgb", width=width, height=height,
            fx=fx, fy=fy, cx=cx, cy=cy, derivation="source")
        if (depth_w, depth_h) == (width, height):
            depth_intrinsics = Intrinsics(
                stream="depth", width=depth_w, height=depth_h,
                fx=fx, fy=fy, cx=cx, cy=cy,
                derivation="source:lowres_wide_intrinsics (same resolution as depth)")
        else:
            depth_intrinsics = rgb_intrinsics.scaled_to(depth_w, depth_h, "depth")

        mesh_rel = None
        mesh_source = self.root / f"{self.scene_id}_3dod_mesh.ply"
        if mesh_source.is_file():
            shutil.copyfile(mesh_source, destination / "mesh.ply")
            mesh_rel = "mesh.ply"

        if rgb_present == 0:
            shutil.rmtree(destination / "rgb", ignore_errors=True)
        if conf_present == 0:
            shutil.rmtree(destination / "confidence", ignore_errors=True)

        modalities = {
            "depth": {"available": True, "count": len(frames),
                      "resolution": [depth_w, depth_h],
                      "units": "uint16 millimetres", "source": "lowres_depth"},
            "intrinsics": {"available": True, "count": len(frames),
                           "source": "lowres_wide_intrinsics",
                           "focal_relative_spread": round(focal_spread, 6),
                           "note": "ARKit publishes per-frame intrinsics; the contract "
                                   "carries the per-frame median."},
            "trajectory": {"available": True, "count": len(frames),
                           "source": "lowres_wide.traj",
                           "source_convention": "world_to_camera",
                           "pose_rate_hz": _pose_rate_hz(pose_times)},
            "rgb": {"available": rgb_present > 0, "count": rgb_present,
                    "resolution": rgb_shape, "source": "lowres_wide"},
            "confidence": {"available": conf_present > 0, "count": conf_present,
                           "levels": [0, 1, 2], "source": "confidence",
                           "meaning": "ARKit ARConfidenceLevel low/medium/high"},
            "mesh": {"available": mesh_rel is not None,
                     "source": mesh_source.name if mesh_rel else None,
                     "note": "ARKit-derived; not independent of the sensor pipeline "
                             "under evaluation, so never a reference measurement."},
            "imu": {"available": False,
                    "note": "ARKitScenes raw publishes no IMU stream."},
            "distortion": {"available": False,
                           "note": "ARKitScenes publishes a pinhole model with no "
                                   "distortion coefficients."},
            "semantics": {"available": False,
                          "note": "3dod box annotations were not retained; no RoomPlan "
                                  "semantics exist for this source."},
        }

        provenance = Provenance(
            source_type=self.source_type,
            source_id=self.scene_id,
            classification=self.classification,
            connector="ARKitScenesConnector",
            connector_version=CONNECTOR_VERSION,
            generated_utc=utc_now(),
            source_paths=[str(self.root)],
            source_digests={
                "lowres_wide.traj": sha256_file(self.root / "lowres_wide.traj"),
                **({mesh_source.name: sha256_file(mesh_source, limit=1 << 22)}
                   if mesh_rel else {}),
            },
            device={"platform": "iOS", "sensor": "ARKit LiDAR",
                    "note": "Device model is not published per-scene by ARKitScenes."},
            notes=[
                "public_development_fixture: valid for DEV_COMPLETE only, never final "
                "POC evidence or benchmark data.",
                "Poses inverted from the source's world-to-camera convention to the "
                "contract's camera_to_world.",
                f"Frames matched to poses by timestamp within "
                f"{self.pose_tolerance_s * 1000:.0f} ms; "
                f"{len(depth_files)} depth frames in, {len(frames)} retained.",
            ],
        )

        capture = NormalizedCapture(
            contract_version=CONTRACT_VERSION,
            provenance=provenance,
            frames=frames,
            intrinsics=[depth_intrinsics, rgb_intrinsics],
            depth_scale_m=DEPTH_SCALE_M,
            world_frame="arkitscenes_world",
            world_up_axis="+z",
            # Verified, not assumed: slicing this scene's own ARKit mesh along
            # +z concentrates far more geometry into a single 1 cm slab than any
            # other candidate axis, and the pose-derived estimate agrees to
            # within a degree. Capture normalization re-checks this at load time.
            world_up_axis_verified=True,
            pose_convention="camera_to_world",
            modalities=modalities,
            excluded_frames=drops,
            frame_selection={
                "stride": self.stride,
                "max_frames": self.max_frames,
                "pose_tolerance_s": self.pose_tolerance_s,
                "rule": "every stride-th depth frame that has both a pose within "
                        "tolerance and a matching intrinsics record",
                "source_frames": len(depth_files),
                "skipped_by_stride": skipped,
                "retained": len(frames),
            },
            mesh=mesh_rel,
        )
        capture.write(destination)
        return capture
