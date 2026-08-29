"""Executable test verifying Python code snippets in README.md."""

from pathlib import Path
import pytest
from spatial_ai import Space, StructuredQueryResult, Surface

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = REPO_ROOT / "samples" / "public_results" / "public-stray-8653a2142b" / "output"


def test_readme_quickstart_python_snippet():
    """Executes the Python quickstart code snippet from README.md and asserts metric output agreement."""
    space = Space.load(SAMPLE_PATH)

    # Inspect room metrics
    dims = space.dimensions
    assert isinstance(dims, dict)
    assert round(dims["length_m"], 2) == 5.91
    assert round(dims["width_m"], 2) == 4.19
    assert round(dims["height_m"], 2) == 2.67
    assert round(dims["area_sq_m"], 2) == 21.39

    # List physical entities
    surfaces = space.surfaces
    assert len(surfaces) == 9
    for surface in surfaces:
        assert surface.id != ""

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
    assert "width_m" in wall.dimensions
    assert "height_m" in wall.dimensions
    assert isinstance(wall.evidence, list)
