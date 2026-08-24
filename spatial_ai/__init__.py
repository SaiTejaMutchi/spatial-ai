"""Spatial AI — Persistent Spatial Memory for AI.

Turn mobile RGB-D/LiDAR captures into persistent physical entities AI can query, inspect, and reason over.
"""

from .errors import PipelineExecutionError, SpatialAIError, SpatialModelNotFoundError, SurfaceNotFoundError
from .space import Space, StructuredQueryResult
from .surface import Surface

__all__ = [
    "Space",
    "Surface",
    "StructuredQueryResult",
    "SpatialAIError",
    "SpatialModelNotFoundError",
    "SurfaceNotFoundError",
    "PipelineExecutionError",
]
