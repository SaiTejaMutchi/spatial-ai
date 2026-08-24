"""Source detection: pick the connector that structurally matches a bundle."""

from __future__ import annotations

from pathlib import Path

from .arkitscenes import ARKitScenesConnector
from .base import SourceRejected
from .stray_scanner import StrayScannerConnector
from .unity_obj import UnityOBJConnector

CONNECTORS = {
    ARKitScenesConnector.source_type: ARKitScenesConnector,
    StrayScannerConnector.source_type: StrayScannerConnector,
    UnityOBJConnector.source_type: UnityOBJConnector,
}

# Most specific first: a Stray bundle and an ARKitScenes sequence both hold
# depth frames, and a Unity export is only a mesh, so it is the last resort.
DETECTION_ORDER = (ARKitScenesConnector, StrayScannerConnector, UnityOBJConnector)


def detect_source(root: Path):
    root = Path(root)
    if not root.exists():
        raise SourceRejected(f"'{root}' does not exist")
    matches = [c for c in DETECTION_ORDER if c.detect(root)]
    if not matches:
        contents = sorted(p.name for p in root.iterdir())[:12] if root.is_dir() else [root.name]
        raise SourceRejected(
            f"'{root}' matches no known capture source. Saw: {contents}. "
            f"Expected an ARKitScenes raw sequence (lowres_depth/ + lowres_wide.traj), "
            f"a Stray Scanner export (rgb.mp4 or rgb/, depth/, odometry.csv, "
            f"camera_matrix.csv), or a Unity OBJ export (scan.obj).")
    return matches[0]
