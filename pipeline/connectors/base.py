"""Shared connector scaffolding.

A connector's entire job is format normalization: decode frames, match them to
poses, preserve metric depth and confidence, convert poses and intrinsics into
the canonical convention, and report precise record-level errors. A connector
never fits a plane, never estimates a room dimension, and never invents a
value to fill a gap.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..contracts.normalized_capture import ExcludedFrame, NormalizedCapture

CONNECTOR_VERSION = "0.1"

# A depth frame is matched to the nearest pose within this window. ARKitScenes
# logs poses at 10 Hz, Stray Scanner at 60 Hz; half of the slower interval is
# the loosest defensible window that still refuses a stale pose.
DEFAULT_POSE_TOLERANCE_S = 0.05


class ConnectorError(Exception):
    """The source could not be read at all."""


@dataclass
class SourceRejected(ConnectorError):
    """The source is structurally wrong for this connector."""
    detail: str

    def __str__(self) -> str:
        return self.detail


def sha256_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    read = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            read += len(block)
            if limit is not None and read >= limit:
                break
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle (radians) to a 3x3 rotation matrix.

    Implemented here rather than pulled from OpenCV so the contract layer keeps
    no computer-vision dependency; `pipeline/tests` checks it against cv2.
    """
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(axis_angle, dtype=np.float64) / theta
    skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def match_poses(
    frame_times: np.ndarray,
    pose_times: np.ndarray,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-pose index and signed time offset for each frame.

    Frames and poses are matched by timestamp, never by index: sources commonly
    log images and odometry at different rates.
    """
    if pose_times.size == 0:
        return np.full(frame_times.shape, -1, dtype=int), np.full(frame_times.shape, np.nan)
    if pose_times.size == 1:
        # np.clip(..., 1, 0) silently collapses when there is a single pose, so
        # this case is handled outright rather than relying on the general path.
        index = np.zeros(frame_times.shape, dtype=int)
        offset = frame_times - pose_times[0]
    else:
        insert = np.clip(np.searchsorted(pose_times, frame_times), 1, pose_times.size - 1)
        left, right = insert - 1, insert
        take_left = (np.abs(frame_times - pose_times[left])
                     <= np.abs(frame_times - pose_times[right]))
        index = np.where(take_left, left, right)
        offset = frame_times - pose_times[index]
    return np.where(np.abs(offset) <= tolerance_s, index, -1), offset


class Connector(ABC):
    """Turns one vendor layout into one `normalized_capture` package."""

    source_type: str = "unknown"

    @classmethod
    def accepted_options(cls) -> set[str]:
        """Constructor options this connector understands.

        The CLI offers a common option set; a connector that cannot honour one
        (a Unity OBJ export has no frames to stride over) simply omits it
        rather than accepting and ignoring it.
        """
        import inspect
        return set(inspect.signature(cls.__init__).parameters) - {"self", "root"}

    @classmethod
    @abstractmethod
    def detect(cls, root: Path) -> bool:
        """True when `root` structurally looks like this source."""

    @abstractmethod
    def normalize(self, destination: Path) -> NormalizedCapture:
        """Emit the contract, or raise `ConnectorError` with a precise reason."""


def excluded(source_id: str, timestamp: float | None, reason: str) -> ExcludedFrame:
    return ExcludedFrame(source_id=source_id, timestamp_s=timestamp, reason=reason)
