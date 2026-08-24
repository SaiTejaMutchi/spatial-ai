"""Opening resolver and image-only vs spatial-grounded benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.ai.make_semantic_fixture import write_fixture
from pipeline.ai.opening_resolver import resolve_candidate
from pipeline.ai.semantic_benchmark import CONCLUSION, run_benchmark
from pipeline.ai.verifier import GroqVerifierClient, load_ai_config, protected_geometry_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "outputs" / "dev_47333462" / "spatial_model.json"


class JsonClient(GroqVerifierClient):
    def __init__(self, spatial_response: dict, image_only_response: dict | None = None):
        super().__init__(post=lambda *a, **k: {}, sleeper=lambda s: None,
                         environ={"GROQ_API_KEY": "gsk_test"})
        self.spatial_response = spatial_response
        self.image_only_response = image_only_response or {
            "visibleOpenings": [{"kind": "door", "description": "a door is visible"}]
        }
        self.spatial_calls = []
        self.image_calls = []

    def complete_json(self, system, user_text, images, model):
        if "Nominated candidate" in user_text:
            self.spatial_calls.append(user_text)
            return {**self.spatial_response, "model": model, "provider": "groq"}
        self.image_calls.append(user_text)
        return {**self.image_only_response, "model": model, "provider": "groq"}


def test_fixture_separates_labels_from_candidates(tmp_path):
    root = write_fixture(tmp_path / "eval")
    candidate = json.loads((root / "case-door-001" / "candidate.json").read_text())
    labels = json.loads((root / "case-door-001" / "labels.json").read_text())
    assert "door" not in json.dumps(candidate)
    assert labels["semanticClass"] == "door"
    assert labels["evaluationOnly"] is True
    assert (root / "case-door-001" / "crop.png").is_file()


def test_schema_rejects_a_smuggled_dimension():
    from jsonschema import Draft202012Validator
    schema = json.loads((REPO_ROOT / "schema/opening_resolution.schema.json").read_text())
    bad = {
        "candidateId": "candidate-001", "surfaceId": "wall-001",
        "semanticClass": "door", "evidenceStatus": "supported",
        "evidenceFrameIds": ["crop"], "reason": "looks like a door",
        "width_m": 0.9,
    }
    errors = list(Draft202012Validator(schema).iter_errors(bad))
    assert errors


def test_corroboration_promotes_with_geometry_owned_dimensions(tmp_path):
    root = write_fixture(tmp_path / "eval")
    candidate = json.loads((root / "case-door-001" / "candidate.json").read_text())
    crop = (root / "case-door-001" / "crop.png").read_bytes()
    client = JsonClient({
        "candidateId": "candidate-001", "surfaceId": "wall-001",
        "semanticClass": "door", "evidenceStatus": "supported",
        "evidenceFrameIds": ["crop"], "reason": "the crop shows a door leaf and handle",
    })
    result = resolve_candidate(candidate, crop, client=client)
    assert result.promoted is not None
    assert result.promoted["producer"] == "geometry_pipeline"
    assert result.promoted["dimensions"]["width_m"] == candidate["geometry"]["width_m"]
    assert "width_m" not in result.resolution


def test_insufficient_evidence_does_not_promote(tmp_path):
    root = write_fixture(tmp_path / "eval")
    candidate = json.loads((root / "case-empty-001" / "candidate.json").read_text())
    crop = (root / "case-empty-001" / "crop.png").read_bytes()
    client = JsonClient({
        "candidateId": "candidate-004", "surfaceId": "wall-004",
        "semanticClass": "insufficient_evidence",
        "evidenceStatus": "insufficient_evidence",
        "evidenceFrameIds": ["crop"], "reason": "the crop is an empty wall",
    })
    result = resolve_candidate(candidate, crop, client=client)
    assert result.promoted is None
    assert result.resolution["semanticClass"] == "insufficient_evidence"


def test_wrong_candidate_id_is_not_repaired_or_promoted(tmp_path):
    root = write_fixture(tmp_path / "eval")
    candidate = json.loads((root / "case-door-001" / "candidate.json").read_text())
    crop = (root / "case-door-001" / "crop.png").read_bytes()
    client = JsonClient({
        "candidateId": "some-other-door", "surfaceId": "wall-001",
        "semanticClass": "door", "evidenceStatus": "supported",
        "evidenceFrameIds": ["crop"],
        "reason": "the room contains a door somewhere",
    })
    result = resolve_candidate(candidate, crop, client=client)
    assert result.promoted is None
    assert result.resolution["semanticClass"] == "insufficient_evidence"


def test_benchmark_image_only_cannot_emit_a_metric_entity(tmp_path):
    root = write_fixture(tmp_path / "eval")
    model = json.loads(MODEL_PATH.read_text()) if MODEL_PATH.is_file() else {
        "surfaces": [], "measurements": [], "rooms": [], "openings": [],
        "coordinateSystem": {}, "damage": [], "scope": [],
    }
    before = protected_geometry_digest(model)
    client = JsonClient({
        "candidateId": "candidate-001", "surfaceId": "wall-001",
        "semanticClass": "door", "evidenceStatus": "supported",
        "evidenceFrameIds": ["crop"], "reason": "crop shows a door",
    })

    def complete_json(system, user_text, images, model_id):
        if "Nominated candidate" in user_text:
            cid = "candidate-001"
            sid = "wall-001"
            if "candidate-002" in user_text:
                cid, sid, klass = "candidate-002", "wall-002", "window"
            elif "candidate-003" in user_text:
                cid, sid, klass = "candidate-003", "wall-003", "occlusion"
            elif "candidate-004" in user_text:
                cid, sid, klass = "candidate-004", "wall-004", "scan_gap"
            else:
                klass = "door"
            status = "supported" if klass in ("door", "window") else "unsupported"
            return {
                "candidateId": cid, "surfaceId": sid, "semanticClass": klass,
                "evidenceStatus": status, "evidenceFrameIds": ["crop"],
                "reason": "nominated region", "model": model_id,
            }
        return {"visibleOpenings": [{"kind": "door", "description": "door in photo"}],
                "model": model_id}

    client.complete_json = complete_json  # type: ignore[method-assign]
    report = run_benchmark(root=root, client=client, spatial_model=model)
    assert report["geometryMutationCount"] == 0
    assert protected_geometry_digest(model) == before
    assert report["summary"]["imageOnlyMetricEntityCount"] == 0
    assert report["summary"]["spatialCandidateBindingRate"] == 1.0
    assert report["conclusion"] == CONCLUSION
    assert "labels.json" not in json.dumps(report["cases"][0]["spatialGrounded"]["resolution"])
    for row in report["cases"]:
        assert row["imageOnly"]["boundToCandidate"] is False
        assert row["imageOnly"]["producedMetricEntity"] is False


def test_benchmark_does_not_send_held_out_labels_to_either_arm(tmp_path):
    root = write_fixture(tmp_path / "eval")
    seen = []

    class Capture(JsonClient):
        def complete_json(self, system, user_text, images, model):
            seen.append(system + user_text)
            return super().complete_json(system, user_text, images, model)

    client = Capture({
        "candidateId": "candidate-001", "surfaceId": "wall-001",
        "semanticClass": "door", "evidenceStatus": "supported",
        "evidenceFrameIds": ["crop"], "reason": "crop shows a door",
    })
    run_benchmark(root=root, client=client, spatial_model={
        "surfaces": [], "measurements": [], "rooms": [], "openings": [],
        "coordinateSystem": {}, "damage": [], "scope": [],
    })
    blob = "\n".join(seen).lower()
    assert "held_out" not in blob
    assert "labelSource" not in blob
    assert "fixture_author" not in blob
