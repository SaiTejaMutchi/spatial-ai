"""Executable test for Python code snippets in README.md."""

from pathlib import Path
import pytest
from spatial_ai import Space, StructuredQueryResult, Surface

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = REPO_ROOT / "samples" / "public_results" / "public-stray-8653a2142b" / "output"


def test_readme_quickstart_python_snippet():
    """Executes the Python quickstart code snippet from README.md."""
    space = Space.load(SAMPLE_PATH)

    # Inspect room metrics
    dims = space.dimensions
    assert isinstance(dims, dict)
    assert dims["length_m"] > 0
    assert dims["area_sq_m"] > 0

    # List physical entities
    surfaces = space.surfaces
    assert len(surfaces) > 0
    for surface in surfaces:
        assert surface.surface_id != ""

    # Query a specific wall and its registered visual evidence
    wall = space.surface("wall-002")
    assert wall is not None
    assert isinstance(wall.evidence, list)

    # Query AI
    res = space.ask("Which wall contains the window or opening?")
    assert isinstance(res, StructuredQueryResult)
    assert res.status != ""


def test_readme_python_sdk_reference_snippet():
    """Executes the Python SDK reference snippet from README.md."""
    space = Space.load(SAMPLE_PATH)

    assert space.dimensions is not None
    assert len(space.surfaces) > 0

    wall = space.surface("wall-002")
    assert wall is not None
    assert wall.type == "wall"
    assert "width_m" in wall.measurements
    assert "height_m" in wall.measurements
    assert isinstance(wall.evidence, list)
