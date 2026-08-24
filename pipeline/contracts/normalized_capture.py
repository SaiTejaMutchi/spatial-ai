"""The `normalized_capture` contract: the sole source-neutral input to geometry.

Everything downstream of this module reads `normalized_capture` and nothing
else. No ARKitScenes or Stray Scanner filename, directory name, or unit
convention may appear past this boundary.

Conventions fixed here, once, for every source:

* **Depth** — 16-bit PNG, one channel, unsigned millimetres. ``0`` means "no
  return", never "zero distance". ``depth_scale_m`` in the manifest states the
  metres-per-unit factor explicitly rather than leaving it implied.
* **Poses** — ``camera_to_world``, a 4x4 row-major matrix applied to column
  vectors: ``p_world = T @ p_camera``. The camera centre is ``T[:3, 3]``.
  Sources that publish world-to-camera must invert in their connector.
* **Camera axes** — ``+x`` right, ``+y`` down, ``+z`` forward along the optical
  axis (the OpenCV/ARKit image convention), so that ``u = fx * x / z + cx``.
* **World axes** — the source's own metric world frame, unmodified. Gravity
  alignment is the normalization stage's job, not a connector's. The frame is *labelled*, never
  silently rotated. A connector declares `world_up_axis` when the source
  documents one — ARKitScenes republishes its sequences Z-up while ARKit's own
  session frame is Y-up — and marks it unverified when the claim rests on
  documentation rather than on data it has checked.
* **Intrinsics** — pinhole ``fx, fy, cx, cy`` in pixels, recorded per stream
  together with the resolution they belong to. A connector that derives depth
  intrinsics by rescaling RGB intrinsics must say so.

Absent optional modalities are recorded as absent. They are never filled in
with plausible-looking invented values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "normalized_capture/0.2"

CORE_MODALITIES = ("depth", "intrinsics", "trajectory")
PREFERRED_MODALITIES = ("rgb", "confidence")
OPTIONAL_MODALITIES = ("mesh", "imu", "distortion", "semantics")

# Sanity bounds for an indoor handheld room scan. These reject nonsense such as
# centimetre-scaled depth or a unit-less trajectory; they are not tuned values
# and no geometry decision may read them.
PLAUSIBLE_DEPTH_M = (0.05, 20.0)
PLAUSIBLE_FOCAL_PX = (10.0, 20000.0)
MAX_PLAUSIBLE_CAMERA_PATH_M = 500.0


class ContractError(Exception):
    """A validation failure that names the offending record."""


@dataclass
class Intrinsics:
    stream: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"
    derivation: str = "source"  # "source" | "rescaled_from:<stream>"
    distortion: list[float] | None = None

    def scaled_to(self, width: int, height: int, stream: str) -> "Intrinsics":
        sx, sy = width / self.width, height / self.height
        return Intrinsics(
            stream=stream, width=width, height=height,
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            model=self.model, derivation=f"rescaled_from:{self.stream}",
            distortion=self.distortion,
        )


@dataclass
class Frame:
    index: int
    timestamp_s: float
    rgb: str | None
    depth: str
    confidence: str | None
    camera_to_world: list[list[float]]
    pose_timestamp_s: float
    pose_time_offset_s: float
    intrinsics_stream: str


@dataclass
class ExcludedFrame:
    source_id: str
    timestamp_s: float | None
    reason: str


@dataclass
class Provenance:
    source_type: str
    source_id: str
    classification: str          # public_development_fixture | final_private_capture | baseline_fallback
    connector: str
    connector_version: str
    generated_utc: str
    source_paths: list[str] = field(default_factory=list)
    source_digests: dict[str, str] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class NormalizedCapture:
    contract_version: str
    provenance: Provenance
    frames: list[Frame]
    intrinsics: list[Intrinsics]
    depth_scale_m: float
    world_frame: str
    pose_convention: str
    modalities: dict[str, dict[str, Any]]
    world_up_axis: str | None = None
    world_up_axis_verified: bool = False
    excluded_frames: list[ExcludedFrame] = field(default_factory=list)
    frame_selection: dict[str, Any] = field(default_factory=dict)
    mesh: str | None = None
    imu: str | None = None

    # -- persistence -----------------------------------------------------

    def manifest(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "depth_scale_m": self.depth_scale_m,
            "depth_encoding": "uint16 png, 0 = no return",
            "world_frame": self.world_frame,
            "world_up_axis": self.world_up_axis,
            "world_up_axis_verified": self.world_up_axis_verified,
            "pose_convention": self.pose_convention,
            "camera_axes": "+x right, +y down, +z forward",
            "modalities": self.modalities,
            "frame_selection": self.frame_selection,
            "frame_count": len(self.frames),
            "excluded_frame_count": len(self.excluded_frames),
            "mesh": self.mesh,
            "imu": self.imu,
            "frames": [asdict(f) for f in self.frames],
            "excluded_frames": [asdict(f) for f in self.excluded_frames],
        }

    def write(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(json.dumps(self.manifest(), indent=2) + "\n")
        (root / "provenance.json").write_text(json.dumps(asdict(self.provenance), indent=2) + "\n")
        (root / "intrinsics.json").write_text(
            json.dumps({"streams": [asdict(i) for i in self.intrinsics]}, indent=2) + "\n")
        (root / "trajectory.json").write_text(json.dumps({
            "pose_convention": self.pose_convention,
            "world_frame": self.world_frame,
            "world_up_axis": self.world_up_axis,
            "poses": [
                {"index": f.index, "timestamp_s": f.timestamp_s,
                 "camera_to_world": f.camera_to_world,
                 "pose_time_offset_s": f.pose_time_offset_s}
                for f in self.frames
            ],
        }, indent=2) + "\n")
        return root

    @classmethod
    def read(cls, root: Path) -> "NormalizedCapture":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        provenance = json.loads((root / "provenance.json").read_text())
        intrinsics = json.loads((root / "intrinsics.json").read_text())
        return cls(
            contract_version=manifest["contract_version"],
            provenance=Provenance(**provenance),
            frames=[Frame(**f) for f in manifest["frames"]],
            intrinsics=[Intrinsics(**i) for i in intrinsics["streams"]],
            depth_scale_m=manifest["depth_scale_m"],
            world_frame=manifest["world_frame"],
            world_up_axis=manifest.get("world_up_axis"),
            world_up_axis_verified=bool(manifest.get("world_up_axis_verified", False)),
            pose_convention=manifest["pose_convention"],
            modalities=manifest["modalities"],
            excluded_frames=[ExcludedFrame(**e) for e in manifest["excluded_frames"]],
            frame_selection=manifest.get("frame_selection", {}),
            mesh=manifest.get("mesh"),
            imu=manifest.get("imu"),
        )
