"""Unit tests for Spatial AI agent adapters and MCP tool handlers."""

import json
from pathlib import Path
import pytest
from spatial_ai.adapters import find_evidence, get_space, get_surface, list_spaces, measure
from spatial_ai.adapters.mcp_server import MCP_TOOLS_MANIFEST, handle_mcp_tool_call

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = REPO_ROOT / "samples" / "public_results" / "public-stray-8653a2142b" / "output"


def test_list_spaces():
    spaces = list_spaces(REPO_ROOT / "samples" / "public_results")
    assert len(spaces) >= 2
    for s in spaces:
        assert "space_id" in s
        assert "area_sq_m" in s


def test_get_space():
    data = get_space(SAMPLE_PATH)
    assert data["space_id"] != ""
    assert "dimensions" in data
    assert len(data["surfaces"]) > 0


def test_get_surface():
    data = get_surface(SAMPLE_PATH, "wall-002")
    assert data["surface_id"] == "wall-002"
    assert data["type"] == "wall"
    assert "dimensions" in data
    assert "observation_state" in data


def test_find_evidence():
    views = find_evidence(SAMPLE_PATH, "wall-002")
    assert isinstance(views, list)


def test_measure():
    m = measure(SAMPLE_PATH)
    assert m["producer"] == "deterministic_3d_geometry"
    assert "room_dimensions" in m

    m_surf = measure(SAMPLE_PATH, "wall-002")
    assert m_surf["producer"] == "deterministic_3d_geometry"
    assert m_surf["surface_id"] == "wall-002"


def test_mcp_tool_dispatcher():
    res = handle_mcp_tool_call("spatial_get_space", {"space_path": str(SAMPLE_PATH)})
    assert "content" in res
    text = res["content"][0]["text"]
    parsed = json.loads(text)
    assert parsed["space_id"] != ""


def test_mcp_manifest_schema_consistency():
    names = [t["name"] for t in MCP_TOOLS_MANIFEST]
    assert "spatial_list_spaces" in names
    assert "spatial_get_space" in names
    assert "spatial_get_surface" in names
    assert "spatial_find_evidence" in names
    assert "spatial_measure" in names
