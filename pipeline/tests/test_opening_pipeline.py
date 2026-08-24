"""Live opening-resolver wiring: crop the nominated gap, never the whole video."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.ai.opening_pipeline import (
    crop_nominated_region, project_corners, resolve_scan_openings,
)
from pipeline.ai.opening_resolver import resolve_candidate
from pipeline.tests.test_opening_resolver import JsonClient

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_a_front_facing_camera_projects_the_nominated_rectangle():
    corners = np.array([
        [-0.5, -0.5, 2.0],
        [0.5, -0.5, 2.0],
        [0.5, 0.5, 2.0],
        [-0.5, 0.5, 2.0],
    ])
    pose = np.eye(4)
    pixels = project_corners(corners, pose, fx=200, fy=200, cx=100, cy=80)
    assert pixels is not None
    assert pixels[:, 0].min() < pixels[:, 0].max()


def test_a_crop_is_taken_from_the_projected_box(tmp_path):
    image = Image.new("RGB", (200, 160), (200, 190, 180))
    pixels = np.array([[40.0, 30.0], [90.0, 30.0], [90.0, 100.0], [40.0, 100.0]])
    data, region = crop_nominated_region(image, pixels)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert 0 <= region["x0"] < region["x1"] <= 1
    cropped = Image.open(__import__("io").BytesIO(data))
    assert cropped.size[0] >= 16 and cropped.size[1] >= 16


def test_corroboration_promotes_geometry_owned_extent_and_leaves_room_metrics(tmp_path):
    image = Image.new("RGB", (200, 160), (180, 180, 180))
    (tmp_path / "evidence").mkdir()
    image.save(tmp_path / "evidence" / "evidence-001.png")
    model = {
        "surfaces": [{
            "id": "wall-006",
            "confidence": {"inputs": {
                "rmsResidual_m": 0.01, "coverageFraction": 0.9,
                "contributingFrames": 20,
            }},
        }],
        "measurements": [{"id": "measurement-room-length", "type": "room_length",
                          "value_m": 7.39, "producer": "geometry_pipeline"}],
        "openings": [{
            "id": "opening-001",
            "surfaceId": "wall-006",
            "type": "unresolved",
            "dimensions": None,
            "observationState": "unresolved",
            "provenance": {
                "candidateExtent": {
                    "width_m": 0.6, "height_m": 1.45, "sillHeight_m": 0.0,
                    "worldCorners_m": [
                        [-0.3, 0.0, 2.0], [0.3, 0.0, 2.0],
                        [0.3, 1.45, 2.0], [-0.3, 1.45, 2.0],
                    ],
                },
            },
        }],
        "evidence": [{
            "id": "frame-000001",
            "path": "evidence/evidence-001.png",
            "cameraToWorldCanonical": np.eye(4).tolist(),
            "visibleSurfaceIds": ["wall-006"],
            "surfaceVisibility": {"wall-006": 0.8},
        }],
    }

    class Capture:
        intrinsics = [type("I", (), {
            "stream": "rgb", "fx": 200, "fy": 200, "cx": 100, "cy": 80,
            "width": 200, "height": 160,
        })()]

    client = JsonClient({
        "candidateId": "opening-001", "surfaceId": "wall-006",
        "semanticClass": "window", "evidenceStatus": "supported",
        "evidenceFrameIds": ["frame-000001"],
        "reason": "the crop shows a window sash in the nominated gap",
    })
    report = resolve_scan_openings(
        model, tmp_path, capture=Capture(), client=client)
    assert report["promotedCount"] == 1
    opening = model["openings"][0]
    assert opening["type"] == "window"
    assert opening["observationState"] == "directly_observed"
    assert opening["dimensions"]["width_m"] == 0.6
    assert opening["producer"] == "geometry_pipeline"
    assert model["measurements"][0]["value_m"] == 7.39
    assert (tmp_path / "evidence" / "opening-001-crop.png").is_file()
    assert opening["provenance"]["aiResolution"]["cropPath"] == "evidence/opening-001-crop.png"


def test_no_photograph_of_the_wall_does_not_promote(tmp_path):
    model = {
        "surfaces": [{"id": "wall-007", "confidence": {"inputs": {}}}],
        "measurements": [],
        "openings": [{
            "id": "opening-002",
            "surfaceId": "wall-007",
            "type": "unresolved",
            "dimensions": None,
            "observationState": "unresolved",
            "provenance": {
                "candidateExtent": {
                    "width_m": 1.15, "height_m": 1.0, "sillHeight_m": 0.4,
                    "worldCorners_m": [[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]],
                },
            },
        }],
        "evidence": [],
    }
    report = resolve_scan_openings(model, tmp_path, client=JsonClient({
        "candidateId": "opening-002", "surfaceId": "wall-007",
        "semanticClass": "door", "evidenceStatus": "supported",
        "evidenceFrameIds": ["crop"], "reason": "should not be asked",
    }))
    assert report["promotedCount"] == 0
    assert model["openings"][0]["observationState"] == "unresolved"
    assert model["openings"][0]["dimensions"] is None
