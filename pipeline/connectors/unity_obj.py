"""Unity ARKit `scan.obj` export -> `normalized_capture`.

The weakest source and an optional one. A Unity mesh export proves that ARKit
scene reconstruction ran on the device and preserves raw geometry evidence, but
it carries no synchronized RGB, depth, confidence, poses, or intrinsics.

Those absences are recorded as absences. This connector emits a package with
zero frames and every core modality marked unavailable, so the contract
validator refuses it for geometry rather than letting a mesh masquerade as a
full capture. That refusal is the point: the fallback keeps the evidence
without inventing the measurements.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..contracts.normalized_capture import (
    CONTRACT_VERSION,
    NormalizedCapture,
    Provenance,
)
from .base import CONNECTOR_VERSION, Connector, SourceRejected, sha256_file, utc_now

UNAVAILABLE = "Unity OBJ export carries no synchronized {} stream."


class UnityOBJConnector(Connector):
    source_type = "unity_obj"

    def __init__(self, root: Path, classification: str = "baseline_fallback") -> None:
        self.root = Path(root)
        self.classification = classification

    @classmethod
    def detect(cls, root: Path) -> bool:
        root = Path(root)
        if root.is_file():
            return root.suffix.lower() == ".obj"
        return any(root.glob("*.obj"))

    def _source_obj(self) -> Path:
        if self.root.is_file():
            return self.root
        candidates = sorted(self.root.glob("*.obj"))
        if not candidates:
            raise SourceRejected(f"'{self.root}' contains no .obj file")
        preferred = [p for p in candidates if p.name == "scan.obj"]
        return (preferred or candidates)[0]

    def normalize(self, destination: Path) -> NormalizedCapture:
        destination = Path(destination)
        obj = self._source_obj()

        vertices = faces = 0
        with obj.open() as handle:
            for line in handle:
                if line.startswith("v "):
                    vertices += 1
                elif line.startswith("f "):
                    faces += 1
        if vertices == 0:
            raise SourceRejected(
                f"'{obj.name}' declares no vertices; the export is empty and cannot "
                f"serve even as raw geometry evidence")

        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(obj, destination / "mesh.obj")

        modalities = {
            "depth": {"available": False, "note": UNAVAILABLE.format("depth")},
            "intrinsics": {"available": False, "note": UNAVAILABLE.format("intrinsics")},
            "trajectory": {"available": False, "note": UNAVAILABLE.format("camera pose")},
            "rgb": {"available": False, "note": UNAVAILABLE.format("colour")},
            "confidence": {"available": False, "note": UNAVAILABLE.format("depth confidence")},
            "mesh": {"available": True, "format": "obj", "vertices": vertices,
                     "faces": faces, "source": obj.name,
                     "note": "ARKit scene-reconstruction mesh. Raw geometry evidence, "
                             "not a reference measurement."},
            "imu": {"available": False, "note": UNAVAILABLE.format("IMU")},
            "distortion": {"available": False, "note": UNAVAILABLE.format("distortion")},
            "semantics": {"available": False,
                          "note": "The mesh carries no per-surface semantics."},
        }

        provenance = Provenance(
            source_type=self.source_type,
            source_id=obj.stem,
            classification=self.classification,
            connector="UnityOBJConnector",
            connector_version=CONNECTOR_VERSION,
            generated_utc=utc_now(),
            source_paths=[str(obj)],
            source_digests={obj.name: sha256_file(obj, limit=1 << 22)},
            device={"platform": "iOS", "sensor": "ARKit scene reconstruction",
                    "pipeline": "Unity_Meshing"},
            notes=[
                "Mesh-only source. Every core modality is unavailable, so this package "
                "is deliberately invalid for geometry: it preserves evidence without "
                "implying measurements the export cannot support.",
                "Unity, Stray, and RoomPlan captures are separate sessions in different "
                "coordinate frames. Do not fuse this mesh with another source unless an "
                "explicit alignment transform has been computed and validated.",
            ],
        )

        capture = NormalizedCapture(
            contract_version=CONTRACT_VERSION,
            provenance=provenance,
            frames=[],
            intrinsics=[],
            depth_scale_m=0.001,
            world_frame="unity_arkit_session",
            world_up_axis="+y",
            world_up_axis_verified=False,
            pose_convention="not_applicable",
            modalities=modalities,
            excluded_frames=[],
            frame_selection={"rule": "no frames exist in a mesh-only export"},
            mesh="mesh.obj",
        )
        capture.write(destination)
        return capture
