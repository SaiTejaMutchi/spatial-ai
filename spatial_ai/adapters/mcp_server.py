"""Model Context Protocol (MCP) server implementation for Spatial AI."""

from __future__ import annotations

import json
from typing import Any

from . import find_evidence, get_space, get_surface, list_spaces, measure


def handle_mcp_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatches MCP tool calls to Spatial AI adapter functions."""
    if name == "spatial_list_spaces":
        return {"content": [{"type": "text", "text": json.dumps(list_spaces(arguments.get("catalog_dir", "samples/public_results")))}]}
    elif name == "spatial_get_space":
        return {"content": [{"type": "text", "text": json.dumps(get_space(arguments["space_path"]))}]}
    elif name == "spatial_get_surface":
        return {"content": [{"type": "text", "text": json.dumps(get_surface(arguments["space_path"], arguments["surface_id"]))}]}
    elif name == "spatial_find_evidence":
        return {"content": [{"type": "text", "text": json.dumps(find_evidence(arguments["space_path"], arguments.get("surface_id")))}]}
    elif name == "spatial_measure":
        return {"content": [{"type": "text", "text": json.dumps(measure(arguments["space_path"], arguments.get("surface_id")))}]}
    else:
        raise ValueError(f"Unknown MCP tool name: {name}")


MCP_TOOLS_MANIFEST = [
    {
        "name": "spatial_list_spaces",
        "description": "Lists available processed spatial spaces in the catalog.",
        "inputSchema": {
            "type": "object",
            "properties": {"catalog_dir": {"type": "string", "default": "samples/public_results"}},
        },
    },
    {
        "name": "spatial_get_space",
        "description": "Retrieves space dimensions, surface list, and room metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {"space_path": {"type": "string"}},
            "required": ["space_path"],
        },
    },
    {
        "name": "spatial_get_surface",
        "description": "Retrieves details, dimensions, observation state, and evidence for a surface.",
        "inputSchema": {
            "type": "object",
            "properties": {"space_path": {"type": "string"}, "surface_id": {"type": "string"}},
            "required": ["space_path", "surface_id"],
        },
    },
    {
        "name": "spatial_find_evidence",
        "description": "Retrieves registered camera evidence stills for a space or specific surface.",
        "inputSchema": {
            "type": "object",
            "properties": {"space_path": {"type": "string"}, "surface_id": {"type": "string"}},
            "required": ["space_path"],
        },
    },
    {
        "name": "spatial_measure",
        "description": "Retrieves geometry-owned metric measurements with provenance tags.",
        "inputSchema": {
            "type": "object",
            "properties": {"space_path": {"type": "string"}, "surface_id": {"type": "string"}},
            "required": ["space_path"],
        },
    },
]
