"""Comprehensive public dataset validation test suite for spatial_ai SDK."""

import json
from pathlib import Path
import pytest
from spatial_ai import Space, Surface, SpatialModelNotFoundError, SurfaceNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_RESULTS_DIR = REPO_ROOT / "samples" / "public_results"


def get_public_space_paths() -> list[Path]:
    """Discovers all published spatial model directories in samples/public_results."""
    if not PUBLIC_RESULTS_DIR.exists():
        return []
    paths = []
    for item in PUBLIC_RESULTS_DIR.iterdir():
        if item.is_dir() and (item / "output" / "spatial_model.json").exists():
            paths.append(item / "output")
        elif item.is_dir() and (item / "spatial_model.json").exists():
            paths.append(item)
    return sorted(paths)


PUBLIC_SPACE_PATHS = get_public_space_paths()


# -----------------------------------------------------------------------------
# Tier 1: Bundled Public Results Validation (Fresh clone, 0 downloads)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("space_path", PUBLIC_SPACE_PATHS, ids=lambda p: p.parent.name if p.name == "output" else p.name)
class TestSDKPublicSpaceValidation:

    def test_space_load_matches_raw_canonical_model(self, space_path: Path):
        """Verifies Space.load() faithfully represents spatial_model.json."""
        space = Space.load(space_path)
        with open(space_path / "spatial_model.json", "r", encoding="utf-8") as f:
            raw_model = json.load(f)

        assert space.id == (raw_model.get("modelId") or raw_model.get("scan", {}).get("id", ""))
        assert len(space.surfaces) == len(raw_model.get("surfaces", []))
        assert space.rooms == raw_model.get("rooms", [])
        assert space.openings == raw_model.get("openings", [])

    def test_space_dimensions_integrity(self, space_path: Path):
        """Verifies space dimensions and floor area calculations."""
        space = Space.load(space_path)
        dims = space.dimensions
        assert isinstance(dims, dict)
        if "area_sq_m" in dims:
            assert space.area == dims["area_sq_m"]
            assert space.area > 0

    def test_surface_lookup_and_properties(self, space_path: Path):
        """Verifies surface lookups and first-class properties."""
        space = Space.load(space_path)
        for surface in space.surfaces:
            assert surface.id != ""
            assert surface.surface_id == surface.id
            assert surface.type in {"wall", "floor", "ceiling", "opening", "unknown"}
            assert isinstance(surface.dimensions, dict)
            assert isinstance(surface.canonical_dimensions, dict)
            assert surface.observation_state in {"directly_observed", "partially_observed", "inferred", "unresolved"}

            # Lookup by ID
            fetched = space.surface(surface.id)
            assert fetched is not None
            assert fetched.id == surface.id

    def test_surface_evidence_binding(self, space_path: Path):
        """Verifies that evidence returned for a surface actually lists that surface ID."""
        space = Space.load(space_path)
        for surface in space.surfaces:
            views = surface.evidence
            for view in views:
                listed_ids = (
                    view.get("visibleSurfaceIds")
                    or view.get("visible_surface_ids")
                    or [view.get("surfaceId"), view.get("surface_id")]
                )
                assert surface.id in listed_ids or view.get("surfaceId") == surface.id

    def test_surface_ai_findings_binding(self, space_path: Path):
        """Verifies AI findings attached to a surface are correctly filtered."""
        space = Space.load(space_path)
        for surface in space.surfaces:
            findings = surface.ai_findings
            for finding in findings:
                target = finding.get("target_surface_id") or finding.get("targetSurfaceId")
                affected = finding.get("affected_surfaces") or finding.get("affectedSurfaces") or []
                assert target == surface.id or surface.id in affected

    def test_nonexistent_surface_returns_none(self, space_path: Path):
        """Verifies requesting an invalid surface ID safely returns None."""
        space = Space.load(space_path)
        assert space.surface("nonexistent-wall-999") is None


# -----------------------------------------------------------------------------
# Tier 2: End-to-End Processing of Raw Public Captures (Stray / ARKit)
# -----------------------------------------------------------------------------

def test_space_process_on_raw_sample_if_present(tmp_path: Path):
    """Executes Space.process() end-to-end if a raw export is available."""
    raw_sample = REPO_ROOT / "samples" / "stray"
    if not (raw_sample / "odometry.csv").exists():
        pytest.skip("No raw Stray capture in samples/stray for Tier 2 test")

    out_dir = tmp_path / "processed_stray"
    space = Space.process(raw_sample, output_dir=out_dir)

    assert space.id != ""
    assert len(space.surfaces) > 0
    assert (out_dir / "spatial_model.json").exists()


def test_space_load_raises_not_found_for_invalid_path():
    """Verifies Space.load() raises SpatialModelNotFoundError for missing paths."""
    with pytest.raises(SpatialModelNotFoundError):
        Space.load("/path/to/definitely_nonexistent_spatial_model_12345")
