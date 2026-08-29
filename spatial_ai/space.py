"""Space abstraction for Spatial AI reconstructed environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import (
    PipelineExecutionError,
    SpatialAIError,
    SpatialModelNotFoundError,
    SurfaceNotFoundError,
    UnsupportedSchemaVersionError,
)
from .surface import Surface

SUPPORTED_SCHEMA_VERSIONS = {"v0.1", "v0.2", "0.1", "0.2"}


@dataclass
class StructuredQueryResult:
    """Structured response from querying a spatial model with AI."""
    answer: str
    entity_ids: list[str]
    evidence_ids: list[str]
    status: str
    raw_response: dict[str, Any] | None = None


class Space:
    """Developer-facing spatial memory container for a reconstructed physical space."""

    def __init__(self, model_data: dict[str, Any], result_path: Path | str | None = None) -> None:
        self._model = model_data
        self._path = Path(result_path) if result_path else None

        version = self._model.get("schemaVersion") or self._model.get("schema_version")
        if version and str(version).lower() not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedSchemaVersionError(
                f"Unsupported schemaVersion '{version}'. Supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        surfaces_raw = self._model.get("surfaces", [])
        evidence_raw = self._model.get("evidence", [])
        if not evidence_raw:
            evidence_raw = self._model.get("evidence_manifest", {}).get("views", [])
        ai_raw = self._model.get("aiAssessments", [])
        if not ai_raw:
            ai_raw = self._model.get("ai_assessments", [])

        self._surfaces = [
            Surface(
                data=s,
                evidence_views=evidence_raw,
                ai_assessments=ai_raw,
                path=self._path,
                full_model=self._model,
            )
            for s in surfaces_raw
        ]
        self._surfaces_by_id = {s.id: s for s in self._surfaces}

    @classmethod
    def load(cls, path: Path | str) -> Space:
        """Loads a processed spatial model result directory or JSON file.

        Args:
            path: Path to a result directory containing spatial_model.json or the JSON file itself.
        """
        target = Path(path)
        if target.is_dir():
            json_file = target / "spatial_model.json"
            if not json_file.exists():
                json_file = target / "output" / "spatial_model.json"
        else:
            json_file = target

        if not json_file.exists():
            raise SpatialModelNotFoundError(f"No spatial_model.json found at {path}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(model_data=data, result_path=target)

    @classmethod
    def process(cls, source_dir: Path | str, output_dir: Path | str | None = None) -> Space:
        """Runs the source-neutral ingestion & geometry pipeline on a capture directory.

        Args:
            source_dir: Path to raw (Stray Scanner export, ARKitScenes, etc.) or normalized capture directory.
            output_dir: Optional output directory for saving pipeline artifacts.
        """
        import tempfile
        from pipeline.geometry.run import run_geometry
        from pipeline.geometry.model import write_model
        from pipeline.connectors.detect import detect_source

        capture_path = Path(source_dir).expanduser().resolve()
        if not capture_path.exists():
            raise SpatialModelNotFoundError(f"Source capture path does not exist: {source_dir}")

        out_path = Path(output_dir).expanduser().resolve() if output_dir else None

        try:
            if (capture_path / "manifest.json").exists():
                norm_dir = capture_path
            else:
                temp_dir = tempfile.mkdtemp(prefix="spatial_ai_norm_")
                norm_dir = Path(temp_dir)
                connector_cls = detect_source(capture_path)
                connector = connector_cls(capture_path)
                connector.normalize(norm_dir)

            res = run_geometry(capture_root=norm_dir)

            target_out = out_path or norm_dir
            target_out.mkdir(parents=True, exist_ok=True)
            write_model(res.model, target_out / "spatial_model.json")

            return cls(model_data=res.model, result_path=target_out)
        except Exception as err:
            raise PipelineExecutionError(f"Pipeline execution failed for {source_dir}: {err}") from err

    @classmethod
    def from_capture(cls, source_dir: Path | str, output_dir: Path | str | None = None) -> Space:
        """Alias for process()."""
        return cls.process(source_dir=source_dir, output_dir=output_dir)

    @property
    def id(self) -> str:
        """Model or scan identifier."""
        return self._model.get("modelId") or self._model.get("scan", {}).get("id", "")

    @property
    def rooms(self) -> list[dict[str, Any]]:
        """List of room records in the space."""
        return self._model.get("rooms", [])

    @property
    def surfaces(self) -> list[Surface]:
        """Returns all physical surfaces (walls, floor, ceiling) in the space."""
        return list(self._surfaces)

    @property
    def openings(self) -> list[dict[str, Any]]:
        """List of opening records (doors, windows, openings)."""
        return self._model.get("openings", [])

    def surface(self, surface_id: str) -> Surface | None:
        """Retrieves a physical surface by ID (e.g. 'wall-002', 'floor-001')."""
        return self._surfaces_by_id.get(surface_id)

    @property
    def dimensions(self) -> dict[str, Any]:
        """Returns room-level metric dimensions (length_m, width_m, height_m, area_sq_m)."""
        res = {}
        for m in self._model.get("measurements", []):
            m_type = m.get("type")
            val = m.get("value_m")
            if m_type == "room_length":
                res["length_m"] = val
            elif m_type == "room_width":
                res["width_m"] = val
            elif m_type == "room_height":
                res["height_m"] = val
            elif m_type == "floor_area":
                res["area_sq_m"] = val

        if "length_m" not in res:
            room_dims = self._model.get("room", {}).get("dimensions", {})
            res.update(room_dims)

        return res

    @property
    def area(self) -> float | None:
        """Total room floor area in square meters."""
        return self.dimensions.get("area_sq_m")

    def evidence(self, surface_id: str | None = None) -> list[dict[str, Any]]:
        """Returns registered visual evidence frames, optionally filtered by surface_id."""
        ev = self._model.get("evidence", [])
        if not ev:
            ev = self._model.get("evidence_manifest", {}).get("views", [])

        if surface_id:
            return [
                view for view in ev
                if view.get("surfaceId") == surface_id
                or view.get("surface_id") == surface_id
                or surface_id in view.get("visibleSurfaceIds", [])
                or surface_id in view.get("visible_surface_ids", [])
            ]
        return ev

    @property
    def ai_assessments(self) -> list[dict[str, Any]]:
        """Returns AI assessments attached to the spatial model."""
        ai = self._model.get("aiAssessments", [])
        if not ai:
            ai = self._model.get("ai_assessments", [])
        return ai

    def ask(self, prompt: str) -> StructuredQueryResult:
        """Queries the multimodal AI verifier regarding the spatial model.

        Returns:
            StructuredQueryResult with answer, entity_ids, evidence_ids, and status.
        """
        from pipeline.ai.verifier import load_ai_config, run_verifier
        config = load_ai_config()
        ev_dir = self._path or Path(".")
        try:
            res = run_verifier(
                model=self._model,
                evidence_dir=ev_dir,
                config=config,
            )
            raw = res.assessment
        except Exception as exc:
            raw = {"status": "not_run", "reason": str(exc)}

        status = raw.get("notRunReason") or raw.get("verdict") or "completed"
        findings = raw.get("findings", [])
        entity_ids = [f.get("target_surface_id") or f.get("surfaceId") for f in findings if f.get("target_surface_id") or f.get("surfaceId")]
        evidence_ids = [f.get("evidence_frame_id") for f in findings if f.get("evidence_frame_id")]

        answer_text = raw.get("assessment") or raw.get("summary") or raw.get("notRunReason") or prompt

        return StructuredQueryResult(
            answer=answer_text,
            entity_ids=list(dict.fromkeys(entity_ids)),
            evidence_ids=list(dict.fromkeys(evidence_ids)),
            status=status,
            raw_response=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns the raw canonical spatial_model dictionary."""
        return dict(self._model)

    def __repr__(self) -> str:
        dims = self.dimensions
        length = dims.get("length_m", "?")
        width = dims.get("width_m", "?")
        area = dims.get("area_sq_m", "?")
        return f"<Space dimensions={length}x{width}m ({area} m²), surfaces={len(self._surfaces)}>"
