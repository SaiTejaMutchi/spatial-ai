"""Custom SDK exceptions for Spatial AI."""

from __future__ import annotations


class SpatialAIError(Exception):
    """Base exception for all Spatial AI SDK errors."""


class SpatialModelNotFoundError(SpatialAIError, FileNotFoundError):
    """Raised when a spatial_model.json file or directory cannot be found."""


class UnsupportedSchemaVersionError(SpatialAIError, ValueError):
    """Raised when an unsupported schemaVersion is encountered in spatial_model.json."""


class SurfaceNotFoundError(SpatialAIError, KeyError):
    """Raised when a requested surface ID does not exist in the spatial model."""


class PipelineExecutionError(SpatialAIError, RuntimeError):
    """Raised when pipeline processing or geometry extraction fails."""
