"""Spatial AI Verifier tests.

The verifier's value is entirely in what it refuses to do, so that is what these
tests exercise: it does not run without approval, it does not select a model,
and it cannot move a single number in the geometry.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pipeline.ai.verifier import (
    AIModelConfig,
    VerifierError,
    build_geometry_summary,
    load_ai_config,
    run_verifier,
    validate_assessment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "outputs" / "dev_47333462" / "spatial_model.json"
EVIDENCE_DIR = REPO_ROOT / "outputs" / "dev_47333462" / "evidence"

requires_generated = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason="run the geometry pipeline first")


@pytest.fixture
def model():
    return json.loads(MODEL_PATH.read_text())


@pytest.fixture
def config():
    return load_ai_config()


def _approved(config: AIModelConfig, provider="anthropic", model="test-model"):
    raw = copy.deepcopy(config.raw)
    raw["AI_MODEL_APPROVED"] = True
    raw["provider"], raw["model"] = provider, model
    return AIModelConfig(config_id=config.config_id, approved=True, provider=provider,
                         model=model, raw=raw, sha256=config.sha256)


class StubClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def assess(self, system, summary, images, response_schema, model):
        self.calls.append({"system": system, "summary": summary,
                           "images": images, "model": model})
        return self.response


def _valid_response(surface_id, evidence_id):
    return {
        "schemaVersion": "0.1", "status": "completed", "model": "test-model",
        "provider": "anthropic", "promptVersion": "spatial_verifier_v0.1",
        "generatedAt": "2026-08-23T00:00:00+00:00",
        "roomTypeHypothesis": "bedroom",
        "findings": [{
            "surfaceId": surface_id, "status": "review_recommended",
            "semanticAgreement": True,
            "reason": "Lower portion of the wall is behind a bed.",
            "evidenceFrameIds": [evidence_id],
            "occlusionDescription": "bed", "openingObservation": "cannot_tell",
        }],
    }


# --------------------------------------------------------------------------
# the default state: nothing approved, nothing chosen, nothing run
# --------------------------------------------------------------------------

def _unapproved(config: AIModelConfig):
    raw = copy.deepcopy(config.raw)
    raw["AI_MODEL_APPROVED"] = False
    raw["provider"] = None
    raw["model"] = None
    return AIModelConfig(config_id=config.config_id, approved=False, provider=None,
                         model=None, raw=raw, sha256=config.sha256)


def test_operator_approved_groq_qwen_is_pinned(config):
    assert config.approved is True
    assert config.provider == "groq"
    assert config.model == "qwen/qwen3.6-27b"
    assert config.image_limit == 3
    assert config.raw["approval"]["approvedBy"] == "operator"
    assert "agent did not select" in config.raw["approval"]["statement"].lower() or (
        "did not select" in config.raw["decisionRecord"]["whyThisModel"].lower())


@requires_generated
def test_the_verifier_emits_not_run_without_approval(model, config):
    result = run_verifier(model, EVIDENCE_DIR, config=_unapproved(config))
    assert result.assessment["status"] == "not_run"
    assert "AI_MODEL_APPROVED is false" in result.assessment["notRunReason"]
    assert result.assessment["findings"] == []


@requires_generated
def test_not_run_still_records_the_pinned_configuration(model, config):
    """The boundary must be auditable even when it did not run."""
    result = run_verifier(model, EVIDENCE_DIR, config=_unapproved(config))
    assert result.diagnostics["aiModelConfigHash"] == config.sha256
    assert result.diagnostics["promptVersion"] == "spatial_verifier_v0.1"
    assert result.diagnostics["promptSha256"]


def test_approval_without_a_named_model_is_refused(tmp_path, config):
    raw = copy.deepcopy(config.raw)
    raw["AI_MODEL_APPROVED"] = True
    raw["provider"] = None
    raw["model"] = None
    path = tmp_path / "ai.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(VerifierError, match="refusing to guess"):
        load_ai_config(path)


def test_an_unimplemented_provider_fails_closed(model_path=MODEL_PATH):
    if not model_path.is_file():
        pytest.skip("no generated model")
    model = json.loads(model_path.read_text())
    config = _approved(load_ai_config(), provider="some-other-vendor")
    result = run_verifier(model, EVIDENCE_DIR, config=config)
    assert result.assessment["status"] == "not_run"
    assert "no client implementation" in result.assessment["notRunReason"]


@requires_generated
def test_too_few_evidence_views_prevents_a_run(model, config):
    thin = copy.deepcopy(model)
    thin["evidence"] = thin["evidence"][:1]
    result = run_verifier(thin, EVIDENCE_DIR, config=_approved(config))
    assert result.assessment["status"] == "not_run"
    assert "requires at least" in result.assessment["notRunReason"]


@requires_generated
def test_a_provider_failure_fails_closed_rather_than_raising(model, config):
    class Broken:
        def assess(self, **kwargs):
            raise RuntimeError("connection reset")

    result = run_verifier(model, EVIDENCE_DIR, config=_approved(config),
                          client=Broken())
    assert result.assessment["status"] == "not_run"
    assert "connection reset" in result.assessment["notRunReason"]


def test_a_missing_prompt_is_never_silently_substituted(tmp_path, config):
    if not MODEL_PATH.is_file():
        pytest.skip("no generated model")
    model = json.loads(MODEL_PATH.read_text())
    with pytest.raises(VerifierError, match="no\\s+unversioned fallback prompt"):
        run_verifier(model, EVIDENCE_DIR, config=config,
                     prompt_path=tmp_path / "absent.txt")


# --------------------------------------------------------------------------
# what the model is shown
# --------------------------------------------------------------------------

@requires_generated
def test_the_summary_is_read_only_geometry(model):
    summary = build_geometry_summary(model)
    assert summary["surfaces"]
    for surface in summary["surfaces"]:
        assert set(surface) == {"surfaceId", "type", "observationState",
                                "width_m", "height_m"}
    assert "cannot change any of them" in summary["note"]


@requires_generated
def test_images_are_capped_at_the_contract_limit(model, config):
    stub = StubClient(_valid_response(model["surfaces"][0]["id"],
                                      model["evidence"][0]["id"]))
    run_verifier(model, EVIDENCE_DIR, config=_approved(config), client=stub)
    assert len(stub.calls[0]["images"]) <= config.image_limit


# --------------------------------------------------------------------------
# fail-closed validation of what comes back
# --------------------------------------------------------------------------

@requires_generated
def test_a_valid_response_is_accepted(model, config):
    surface_id = model["surfaces"][0]["id"]
    evidence_id = model["evidence"][0]["id"]
    stub = StubClient(_valid_response(surface_id, evidence_id))
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(config), client=stub)
    assert result.assessment["status"] == "completed"
    assert len(result.assessment["findings"]) == 1
    assert result.rejected_findings == []


@requires_generated
def test_a_finding_naming_an_unknown_surface_is_dropped(model, config):
    response = _valid_response("wall-999", model["evidence"][0]["id"])
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(config),
                          client=StubClient(response))
    assert result.assessment["findings"] == []
    assert "resolves to no surface" in result.rejected_findings[0]["reason"]


@requires_generated
def test_a_finding_citing_an_unsupplied_frame_is_dropped(model, config):
    response = _valid_response(model["surfaces"][0]["id"], "frame-999999")
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(config),
                          client=StubClient(response))
    assert result.assessment["findings"] == []
    assert "never supplied" in result.rejected_findings[0]["reason"]


@requires_generated
def test_a_schema_violating_response_is_rejected_in_full(model, config):
    bad = _valid_response(model["surfaces"][0]["id"], model["evidence"][0]["id"])
    bad["findings"][0]["status"] = "definitely_fine"
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(config),
                          client=StubClient(bad))
    assert result.assessment["status"] == "not_run"
    assert "did not satisfy the assessment schema" in result.assessment["notRunReason"]


@requires_generated
def test_a_response_smuggling_a_measurement_is_rejected(model, config):
    """additionalProperties is false so there is nowhere to put a number."""
    bad = _valid_response(model["surfaces"][0]["id"], model["evidence"][0]["id"])
    bad["findings"][0]["correctedWidth_m"] = 3.9
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(config),
                          client=StubClient(bad))
    assert result.assessment["status"] == "not_run"


@requires_generated
def test_a_finding_without_evidence_is_rejected(model, config):
    bad = _valid_response(model["surfaces"][0]["id"], model["evidence"][0]["id"])
    bad["findings"][0]["evidenceFrameIds"] = []
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(config),
                          client=StubClient(bad))
    assert result.assessment["status"] == "not_run"


# --------------------------------------------------------------------------
# the regression that matters: AI cannot move geometry
# --------------------------------------------------------------------------

@requires_generated
def test_ai_output_cannot_change_any_protected_geometry(model, config):
    """Attaching an assessment must leave every measurement byte-identical."""
    before = copy.deepcopy(model)
    surface_id = model["surfaces"][0]["id"]
    evidence_id = model["evidence"][0]["id"]

    hostile = _valid_response(surface_id, evidence_id)
    hostile["findings"].append({
        "surfaceId": surface_id, "status": "verified", "semanticAgreement": False,
        "reason": "This wall is actually 9 metres long and should be resized.",
        "evidenceFrameIds": [evidence_id],
    })
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(config),
                          client=StubClient(hostile))

    after = copy.deepcopy(model)
    after["aiAssessments"] = [result.assessment]

    for key in ("surfaces", "measurements", "rooms", "openings",
                "coordinateSystem", "damage", "scope"):
        assert after[key] == before[key], f"AI review altered {key}"


@requires_generated
def test_attaching_an_assessment_keeps_the_model_valid(model, config):
    from pipeline.contracts.validate_model import validate_model

    surface_id = model["surfaces"][0]["id"]
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(config),
                          client=StubClient(_valid_response(
                              surface_id, model["evidence"][0]["id"])))
    model["aiAssessments"] = [result.assessment]
    assert validate_model(model) == []


@requires_generated
def test_the_generated_model_assessment_is_auditable(model):
    assert model["aiAssessments"]
    assessment = model["aiAssessments"][0]
    assert assessment["promptVersion"] == "spatial_verifier_v0.1"
    assert assessment["status"] in ("completed", "not_run")
    if assessment["status"] == "completed":
        assert assessment["provider"] == "groq"
        assert "qwen/qwen3.6-27b" in (assessment["model"] or "")
        assert model["provenance"]["aiReview"]["approved"] is True
        assert model["provenance"]["aiReview"]["validationResult"] in (
            "accepted", "schema_rejected", "provider_failure")
    else:
        assert assessment["notRunReason"]


# --------------------------------------------------------------------------
# prompts and configuration are versioned artifacts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["spatial_verifier_v0.1.txt", "loss_proposal_v0.1.txt"])
def test_versioned_prompts_exist_and_forbid_measurement(name):
    text = (REPO_ROOT / "prompts" / name).read_text()
    assert "v0.1" in text
    assert "never" in text.lower()
    lowered = text.lower()
    assert any(phrase in lowered for phrase in
               ("dimension", "square footage", "measurement"))


def test_ai_config_documents_its_decision_record(config):
    record = config.raw["decisionRecord"]
    for key in ("whyThisModel", "alternativesConsidered",
                "whySufficientForTheBoundedTask", "whatItIsNotTrustedToDo",
                "whatPrivateDataCouldBeSent", "whatHappensIfAccessFails"):
        assert record[key], key
    assert config.raw["fallbackBehaviour"] == "not_run"
    assert config.raw["prohibitedUses"]


def test_groq_provider_build_client(config):
    from pipeline.ai.verifier import GroqVerifierClient, build_client
    approved_groq = _approved(config, provider="groq", model="qwen/qwen3.6-27b")
    client = build_client(approved_groq)
    assert isinstance(client, GroqVerifierClient)


def test_load_dotenv_populates_os_environ(tmp_path, monkeypatch):
    from pipeline.ai.verifier import load_dotenv
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=test_groq_key_123\n")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    load_dotenv(env_file)
    import os
    assert os.environ.get("GROQ_API_KEY") == "test_groq_key_123"


def test_groq_verifier_client_without_api_key_raises_error():
    from pipeline.ai.verifier import GroqVerifierClient
    client = GroqVerifierClient(environ={"GROQ_API_KEY": "your_groq_api_key_here"})
    with pytest.raises(VerifierError, match="GROQ_API_KEY is not set or contains default placeholder"):
        client.assess("sys", {}, [], {}, "qwen/qwen3.6-27b")


def test_groq_does_not_read_a_model_override_from_the_environment():
    from pipeline.ai.verifier import GroqVerifierClient
    posts = []

    def post(payload, api_key, timeout_s=90):
        posts.append(payload)
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": json.dumps({
                "schemaVersion": "0.1", "status": "completed",
                "model": payload["model"], "provider": "groq",
                "promptVersion": "spatial_verifier_v0.1",
                "generatedAt": "2026-08-23T00:00:00+00:00",
                "roomTypeHypothesis": None, "findings": [], "usage": None,
            })}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    client = GroqVerifierClient(
        post=post, sleeper=lambda _s: None,
        environ={"GROQ_API_KEY": "gsk_test", "GROQ_MODEL": "some-other-model"},
    )
    parsed = client.assess("sys", {}, [("frame-1", b"abc")], {}, "qwen/qwen3.6-27b")
    assert posts[0]["model"] == "qwen/qwen3.6-27b"
    assert posts[0]["response_format"] == {"type": "json_object"}
    assert posts[0]["reasoning_effort"] == "none"
    assert parsed["model"] == "qwen/qwen3.6-27b"
    assert "some-other-model" not in json.dumps(posts[0])


def test_groq_caps_images_at_three():
    from pipeline.ai.verifier import GROQ_IMAGES_PER_REQUEST, GroqVerifierClient, MAX_IMAGES_PER_REQUEST
    captured = []

    def post(payload, api_key, timeout_s=90):
        captured.append(payload)
        return {
            "model": "qwen/qwen3.6-27b",
            "choices": [{"message": {"content": "{}"}}],
            "usage": {},
        }

    client = GroqVerifierClient(post=post, sleeper=lambda _s: None,
                                environ={"GROQ_API_KEY": "gsk_test"})
    images = [(f"frame-{i}", b"x") for i in range(6)]
    client.assess("sys", {}, images, {"type": "object"}, "qwen/qwen3.6-27b")
    image_parts = [p for p in captured[0]["messages"][1]["content"]
                   if p.get("type") == "image_url"]
    assert MAX_IMAGES_PER_REQUEST == 3
    assert len(image_parts) == GROQ_IMAGES_PER_REQUEST == 1
    assert len(image_parts) <= MAX_IMAGES_PER_REQUEST


def test_groq_retries_429_honouring_retry_after():
    from pipeline.ai.verifier import GroqRateLimitError, GroqVerifierClient
    sleeps, calls = [], []

    def post(payload, api_key, timeout_s=90):
        calls.append(1)
        if len(calls) == 1:
            raise GroqRateLimitError("rate limited", retry_after_s=0.0)
        return {
            "model": "qwen/qwen3.6-27b",
            "choices": [{"message": {"content": "{}"}}],
            "usage": {},
        }

    client = GroqVerifierClient(post=post, sleeper=sleeps.append,
                                environ={"GROQ_API_KEY": "gsk_test"})
    client.assess("sys", {}, [], {}, "qwen/qwen3.6-27b")
    assert sleeps == [0.0]
    assert len(calls) == 2


@requires_generated
def test_groq_timeout_fails_closed_after_bounded_retry(model, config):
    from pipeline.ai.verifier import GroqTimeoutError, GroqVerifierClient
    calls = []

    def post(payload, api_key, timeout_s=90):
        calls.append(timeout_s)
        raise GroqTimeoutError("timed out")

    client = GroqVerifierClient(post=post, sleeper=lambda _s: None,
                                environ={"GROQ_API_KEY": "gsk_test"})
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(
        config, provider="groq", model="qwen/qwen3.6-27b"), client=client)
    assert result.assessment["status"] == "not_run"
    assert "timed out" in result.assessment["notRunReason"]
    assert len(calls) == 2
    assert result.diagnostics["validationResult"] == "provider_failure"


@requires_generated
def test_malformed_groq_json_is_rejected_in_full(model, config):
    from pipeline.ai.verifier import GroqVerifierClient

    def post(payload, api_key, timeout_s=90):
        return {"choices": [{"message": {"content": "definitely not json"}}]}

    client = GroqVerifierClient(post=post, sleeper=lambda _s: None,
                                environ={"GROQ_API_KEY": "gsk_test"})
    result = run_verifier(model, EVIDENCE_DIR, config=_approved(
        config, provider="groq", model="qwen/qwen3.6-27b"), client=client)
    assert result.assessment["status"] == "not_run"
    assert "not JSON" in result.assessment["notRunReason"]
    assert result.diagnostics["validationResult"] == "provider_failure"


@requires_generated
def test_attaching_a_live_shaped_assessment_does_not_move_geometry(model, config):
    from pipeline.ai.verifier import protected_geometry_digest
    before = protected_geometry_digest(model)
    surface_id = model["surfaces"][0]["id"]
    evidence_id = model["evidence"][0]["id"]
    result = run_verifier(
        model, EVIDENCE_DIR,
        config=_approved(config, provider="groq", model="qwen/qwen3.6-27b"),
        client=StubClient(_valid_response(surface_id, evidence_id)),
    )
    after_model = copy.deepcopy(model)
    after_model["aiAssessments"] = [result.assessment]
    assert protected_geometry_digest(after_model) == before
    assert result.diagnostics["geometryDigestBefore"] == before

