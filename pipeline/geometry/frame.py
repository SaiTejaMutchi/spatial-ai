"""Canonical coordinate frame: metres, right-handed, +Y up, X/Z as plan axes.

Sources do not agree on which axis is up. ARKitScenes republishes its raw
sequences in a **Z-up** world frame; ARKit's own session frame, which Stray
Scanner records directly, is **Y-up**. Assuming either one would silently
rotate a room onto its side, so the up axis is resolved rather than assumed:

1. The connector *declares* the source's gravity axis, because ARKit world
   frames are gravity-aligned by construction and the declaration is the most
   accurate information available.
2. That declaration is *cross-checked* against observed structure at ingestion
   (floor, ceiling, plausible storey height, short camera-path extent along the
   candidate vertical). A handheld pose-scatter of camera right-axes is recorded
   as a diagnostic; it is not allowed to override a declaration that structure
   has already verified, because a camera that looks all around the room makes
   that scatter choose a horizontal axis.
3. If the declaration is unverified and disagrees with the pose-derived
   estimate by more than the configured tolerance, the estimate is used instead
   and the substitution is recorded in provenance.

The vertical *origin* is not set here. The floor height is not known until
planes are fitted downstream, so this module produces the rotation and exposes
`with_floor_origin`; the composed transform is what downstream geometry uses.
Room-axis alignment likewise waits for walls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

CANONICAL_UP = np.array([0.0, 1.0, 0.0])

AXIS_VECTORS = {
    "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
}


class FrameError(Exception):
    """The source frame could not be resolved."""


def axis_vector(name: str) -> np.ndarray:
    key = name.strip().lower()
    if key not in AXIS_VECTORS:
        raise FrameError(
            f"'{name}' is not a recognised axis; expected one of {sorted(AXIS_VECTORS)}")
    return AXIS_VECTORS[key].copy()


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def estimate_up_from_poses(camera_to_world: np.ndarray) -> tuple[np.ndarray, dict]:
    """Estimate the world up axis from camera orientations alone.

    The camera's image-right axis stays close to horizontal throughout a
    handheld scan, so gravity is the direction least represented among those
    right-axes: the smallest eigenvector of their scatter matrix. The sign is
    resolved with the mean image-down axis, which points roughly downward.
    """
    rotations = np.asarray(camera_to_world)[:, :3, :3]
    right = rotations @ np.array([1.0, 0.0, 0.0])
    down = rotations @ np.array([0.0, 1.0, 0.0])

    scatter = np.einsum("ni,nj->ij", right, right)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    up = eigenvectors[:, 0]

    mean_down = down.mean(axis=0)
    norm = float(np.linalg.norm(mean_down))
    if norm < 1e-9:
        raise FrameError(
            "camera orientations average to nothing; the up direction cannot be "
            "signed from these poses")
    mean_up = -mean_down / norm
    if float(up @ mean_up) < 0:
        up = -up
    up = up / np.linalg.norm(up)

    return up, {
        "method": "min_scatter_of_camera_right_axes",
        "poseCount": int(len(rotations)),
        "eigenvalueRatio": float(eigenvalues[0] / max(eigenvalues[-1], 1e-12)),
        "agreementWithMeanCameraDownDeg": round(angle_between_deg(up, mean_up), 3),
    }


def rotation_taking(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal rotation carrying `source` onto `target`.

    Minimal on purpose: the horizontal frame is left exactly as the source had
    it, because nothing observed so far justifies choosing different plan axes.
    Room-axis alignment is a later, evidence-backed decision.
    """
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)
    cross = np.cross(a, b)
    dot = float(a @ b)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        if dot > 0:
            return np.eye(3)
        # Antiparallel: rotate a half turn about any axis orthogonal to `a`.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(a @ helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        skew = np.array([[0.0, -axis[2], axis[1]],
                         [axis[2], 0.0, -axis[0]],
                         [-axis[1], axis[0], 0.0]])
        return np.eye(3) + 2.0 * (skew @ skew)
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (sine ** 2))


@dataclass
class CanonicalFrame:
    """The source-to-canonical rigid transform plus how it was decided."""

    source_to_canonical: np.ndarray
    source_up_axis: np.ndarray
    up_source: str                 # "declared" | "pose_estimate"
    world_frame: str
    floor_origin_applied: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return points @ self.source_to_canonical[:3, :3].T + self.source_to_canonical[:3, 3]

    def apply_pose(self, camera_to_world: np.ndarray) -> np.ndarray:
        return self.source_to_canonical @ np.asarray(camera_to_world, dtype=np.float64)

    def with_floor_origin(self, floor_height_m: float) -> "CanonicalFrame":
        """Drop the origin onto the fitted floor, keeping the same rotation."""
        shifted = self.source_to_canonical.copy()
        shifted[1, 3] -= float(floor_height_m)
        diagnostics = dict(self.diagnostics)
        diagnostics["floorOriginOffsetM"] = float(floor_height_m)
        return CanonicalFrame(
            source_to_canonical=shifted,
            source_up_axis=self.source_up_axis,
            up_source=self.up_source,
            world_frame=self.world_frame,
            floor_origin_applied=True,
            diagnostics=diagnostics,
        )

    def provenance(self) -> dict:
        return {
            "sourceWorldFrame": self.world_frame,
            "canonicalFrame": {
                "units": "meters",
                "handedness": "right",
                "upAxis": "y",
                "planAxes": ["x", "z"],
            },
            "sourceUpAxis": [round(float(v), 6) for v in self.source_up_axis],
            "upAxisDeterminedBy": self.up_source,
            "sourceToCanonical": [[round(float(v), 12) for v in row]
                                  for row in self.source_to_canonical],
            "floorOriginApplied": self.floor_origin_applied,
            "roomAxisAlignmentApplied": False,
            "diagnostics": self.diagnostics,
        }


def resolve_canonical_frame(
    camera_to_world: np.ndarray,
    declared_up_axis: str | None,
    world_frame: str,
    tolerance_deg: float,
    min_poses: int,
    declaration_verified: bool = False,
) -> CanonicalFrame:
    poses = np.asarray(camera_to_world, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise FrameError(
            f"expected an (N, 4, 4) array of camera_to_world poses, got shape {poses.shape}")
    if len(poses) == 0:
        raise FrameError("a canonical frame cannot be resolved without any camera poses")

    diagnostics: dict[str, Any] = {"declaredUpAxis": declared_up_axis}

    estimated = None
    if len(poses) >= min_poses:
        estimated, estimate_diagnostics = estimate_up_from_poses(poses)
        diagnostics["poseEstimate"] = estimate_diagnostics
        diagnostics["poseEstimatedUpAxis"] = [round(float(v), 6) for v in estimated]
    else:
        diagnostics["poseEstimate"] = {
            "skipped": True,
            "reason": f"only {len(poses)} poses available; at least {min_poses} are "
                      f"required for an independent cross-check",
        }

    if declared_up_axis is None:
        if estimated is None:
            raise FrameError(
                "the source declares no gravity axis and there are too few poses to "
                "estimate one; the up direction is unknown and must not be guessed")
        up, up_source = estimated, "pose_estimate"
        diagnostics["note"] = ("source declared no gravity axis; the pose-derived "
                               "estimate was used")
    else:
        declared = axis_vector(declared_up_axis)
        if estimated is None:
            up, up_source = declared, "declared"
        elif declaration_verified:
            disagreement = angle_between_deg(declared, estimated)
            diagnostics["declaredVsEstimatedDeg"] = round(disagreement, 3)
            diagnostics["toleranceDeg"] = tolerance_deg
            diagnostics["declarationVerified"] = True
            up, up_source = declared, "declared"
            diagnostics["note"] = (
                f"declared axis '{declared_up_axis}' was kept because ingestion "
                f"structure verification already accepted it. The pose-derived "
                f"estimate differs by {disagreement:.2f} deg and is diagnostic only.")
        else:
            disagreement = angle_between_deg(declared, estimated)
            diagnostics["declaredVsEstimatedDeg"] = round(disagreement, 3)
            diagnostics["toleranceDeg"] = tolerance_deg
            if disagreement <= tolerance_deg:
                up, up_source = declared, "declared"
                diagnostics["note"] = ("declared gravity axis confirmed by the "
                                       "pose-derived estimate")
            else:
                up, up_source = estimated, "pose_estimate"
                diagnostics["note"] = (
                    f"declared axis '{declared_up_axis}' disagrees with the "
                    f"pose-derived estimate by {disagreement:.2f} deg, beyond the "
                    f"{tolerance_deg} deg tolerance; the declaration was rejected in "
                    f"favour of the estimate")

    rotation = rotation_taking(up, CANONICAL_UP)
    transform = np.eye(4)
    transform[:3, :3] = rotation

    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-9:
        raise FrameError(
            f"the computed source-to-canonical rotation has determinant "
            f"{determinant:.9f}; a right-handed frame requires +1")
    diagnostics["rotationDeterminant"] = round(determinant, 12)
    diagnostics["residualUpErrorDeg"] = round(
        angle_between_deg(rotation @ up, CANONICAL_UP), 9)

    return CanonicalFrame(
        source_to_canonical=transform,
        source_up_axis=up,
        up_source=up_source,
        world_frame=world_frame,
        diagnostics=diagnostics,
    )
