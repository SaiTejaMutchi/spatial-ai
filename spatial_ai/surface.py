"""Surface abstraction for Spatial AI physical entities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Surface:
    """Represents a persistent physical spatial entity (wall, floor, ceiling)."""

    def __init__(
        self,
        data: dict[str, Any],
        evidence_views: list[dict[str, Any]] | None = None,
        ai_assessments: list[dict[str, Any]] | None = None,
        path: Path | str | None = None,
        full_model: dict[str, Any] | None = None,
    ) -> None:
        self._data = data
        self._evidence = evidence_views or []
        self._ai_assessments = ai_assessments or []
        self._path = Path(path) if path else None
        self._full_model = full_model

    @property
    def id(self) -> str:
        """Stable surface identifier (e.g. 'wall-001', 'floor-001')."""
        return self._data.get("id") or self._data.get("surface_id", "")

    @property
    def surface_id(self) -> str:
        """Alias for id."""
        return self.id

    @property
    def type(self) -> str:
        """Surface classification ('wall', 'floor', 'ceiling')."""
        return self._data.get("type", "unknown")

    @property
    def dimensions(self) -> dict[str, Any]:
        """Metric quantities calculated by deterministic geometry."""
        return dict(self._data.get("dimensions", {}))

    @property
    def canonical_dimensions(self) -> dict[str, Any]:
        """Exact raw measurement dictionary as written in spatial_model.json."""
        return dict(self._data.get("dimensions", {}))

    @property
    def measurements(self) -> dict[str, Any]:
        """Alias for dimensions."""
        return self.dimensions

    @property
    def observation_state(self) -> str:
        """Observation state ('directly_observed', 'partially_observed', 'inferred', 'unresolved')."""
        return self._data.get("observationState", "unresolved")

    @property
    def confidence(self) -> str:
        """Confidence assessment label ('high', 'medium', 'low', 'unresolved')."""
        conf = self._data.get("confidence", {})
        if isinstance(conf, dict):
            return conf.get("label", "unresolved")
        return str(conf)

    @property
    def evidence(self) -> list[dict[str, Any]]:
        """Registered visual evidence stills associated with this surface."""
        return [
            e for e in self._evidence
            if e.get("surfaceId") == self.id
            or e.get("surface_id") == self.id
            or self.id in e.get("visibleSurfaceIds", [])
            or self.id in e.get("visible_surface_ids", [])
        ]

    @property
    def ai_findings(self) -> list[dict[str, Any]]:
        """AI review findings bound to this surface."""
        return [
            a for a in self._ai_assessments
            if a.get("target_surface_id") == self.id
            or a.get("targetSurfaceId") == self.id
            or self.id in a.get("affected_surfaces", [])
            or self.id in a.get("affectedSurfaces", [])
        ]

    @property
    def ai_assessments(self) -> list[dict[str, Any]]:
        """Alias for ai_findings."""
        return self.ai_findings

    def ask(self, prompt: str) -> dict[str, Any]:
        """Queries the multimodal AI verifier specifically regarding this surface."""
        from pipeline.ai.verifier import load_ai_config, run_verifier
        config = load_ai_config()
        ev_dir = self._path or Path(".")
        if self._full_model:
            model = dict(self._full_model)
        else:
            model = {
                "surfaces": [self._data],
                "rooms": [],
                "openings": [],
                "evidence": self.evidence,
            }
        try:
            res = run_verifier(
                model=model,
                evidence_dir=ev_dir,
                config=config,
            )
            return res.assessment
        except Exception as exc:
            return {"status": "not_run", "reason": str(exc)}

    def to_dict(self) -> dict[str, Any]:
        """Returns the raw surface dictionary representation."""
        return dict(self._data)

    def __repr__(self) -> str:
        area = self.dimensions.get("area_sq_m") or self.dimensions.get("width_m", "n/a")
        return f"<Surface {self.id} ({self.type}, area={area} m²)>"
