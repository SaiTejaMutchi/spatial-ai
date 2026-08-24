"""Tests for spatial_ai Python SDK facade."""

from pathlib import Path
import pytest
from spatial_ai import Space, Surface

PUBLIC_STRAY_SAMPLE = Path("samples/public_results/public-stray-8653a2142b/output")
PUBLIC_IPHONE_SAMPLE = Path("samples/public_results/public-iphone-e30fe3cae4/output")


def test_load_space():
    space = Space.load(PUBLIC_STRAY_SAMPLE)
    assert repr(space).startswith("<Space")
    assert isinstance(space.dimensions, dict)
    assert "length_m" in space.dimensions
    assert "width_m" in space.dimensions
    assert "height_m" in space.dimensions
    assert "area_sq_m" in space.dimensions
    assert space.dimensions["length_m"] > 0


def test_space_surfaces():
    space = Space.load(PUBLIC_STRAY_SAMPLE)
    surfaces = space.surfaces
    assert len(surfaces) > 0
    for s in surfaces:
        assert isinstance(s, Surface)
        assert s.surface_id != ""
        assert s.type in ("wall", "floor", "ceiling")
        assert isinstance(s.measurements, dict)


def test_get_specific_surface():
    space = Space.load(PUBLIC_STRAY_SAMPLE)
    wall = space.surface("wall-002")
    assert wall is not None
    assert wall.surface_id == "wall-002"
    assert wall.type == "wall"
    assert "width_m" in wall.measurements
    assert "height_m" in wall.measurements
    assert repr(wall).startswith("<Surface wall-002")


def test_space_to_dict():
    space = Space.load(PUBLIC_STRAY_SAMPLE)
    model_dict = space.to_dict()
    assert isinstance(model_dict, dict)
    assert "rooms" in model_dict
    assert "surfaces" in model_dict
    assert "measurements" in model_dict


def test_iphone_sample_sdk():
    space = Space.load(PUBLIC_IPHONE_SAMPLE)
    assert space.dimensions["area_sq_m"] > 0
    floor = space.surface("floor-001")
    assert floor is not None
    assert floor.type == "floor"
