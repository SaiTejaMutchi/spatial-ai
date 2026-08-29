"""Agent & Framework adapters for Spatial AI.

Provides typed tool definitions compatible with Model Context Protocol (MCP), LangChain, and CrewAI.
"""

from pathlib import Path
from typing import Any
from spatial_ai.space import Space


def list_spaces(catalog_dir: Path | str = "samples/public_results") -> list[dict[str, Any]]:
    """Lists available spatial spaces in the catalog.

    Returns:
        List of dictionaries with space_id, path, area_sq_m, and surface_count.
    """
    root = Path(catalog_dir)
    if not root.exists():
        return []

    spaces = []
    for item in sorted(root.iterdir()):
        target = item / "output" if (item / "output" / "spatial_model.json").exists() else item
        if (target / "spatial_model.json").exists():
            try:
                space = Space.load(target)
                spaces.append({
                    "space_id": space.id,
                    "path": str(target),
                    "area_sq_m": space.area,
                    "surface_count": len(space.surfaces),
                })
            except Exception:
                continue
    return spaces


def get_space(space_path: Path | str) -> dict[str, Any]:
    """Retrieves spatial space dimensions, surface summary, and metadata.

    Returns:
        Dictionary containing space dimensions, surfaces, openings, and rooms.
    """
    space = Space.load(space_path)
    return {
        "space_id": space.id,
        "dimensions": space.dimensions,
        "area_sq_m": space.area,
        "surfaces": [{"id": s.id, "type": s.type, "dimensions": s.dimensions} for s in space.surfaces],
        "openings": space.openings,
        "rooms": space.rooms,
    }


def get_surface(space_path: Path | str, surface_id: str) -> dict[str, Any]:
    """Retrieves details, metric dimensions, observation state, and evidence for a specific surface.

    Returns:
        Dictionary containing surface ID, type, dimensions, observation_state, evidence count, and AI findings.
    """
    space = Space.load(space_path)
    surface = space.surface(surface_id)
    if not surface:
        return {"error": f"Surface '{surface_id}' not found in space '{space.id}'"}

    return {
        "space_id": space.id,
        "surface_id": surface.id,
        "type": surface.type,
        "dimensions": surface.dimensions,
        "canonical_dimensions": surface.canonical_dimensions,
        "observation_state": surface.observation_state,
        "confidence": surface.confidence,
        "evidence_count": len(surface.evidence),
        "ai_findings": surface.ai_findings,
    }


def find_evidence(space_path: Path | str, surface_id: str | None = None) -> list[dict[str, Any]]:
    """Retrieves registered RGB camera evidence views, optionally filtered by surface_id.

    Returns:
        List of registered visual evidence frame records.
    """
    space = Space.load(space_path)
    return space.evidence(surface_id=surface_id)


def measure(space_path: Path | str, surface_id: str | None = None) -> dict[str, Any]:
    """Retrieves geometry-owned metric measurements for a space or a surface.

    Returns:
        Dictionary containing deterministic geometry measurements and producer provenance.
    """
    space = Space.load(space_path)
    if surface_id:
        surface = space.surface(surface_id)
        if not surface:
            return {"error": f"Surface '{surface_id}' not found"}
        return {
            "producer": "deterministic_3d_geometry",
            "surface_id": surface.id,
            "dimensions": surface.dimensions,
            "observation_state": surface.observation_state,
        }
    return {
        "producer": "deterministic_3d_geometry",
        "space_id": space.id,
        "room_dimensions": space.dimensions,
        "total_area_sq_m": space.area,
    }


__all__ = [
    "list_spaces",
    "get_space",
    "get_surface",
    "find_evidence",
    "measure",
]
