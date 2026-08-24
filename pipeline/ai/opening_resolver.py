"""Candidate-grounded Opening Resolver.

The VLM classifies a nominated region. Geometry alone may promote and
quantify. Scene-level recognition of a door elsewhere is not corroboration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .verifier import (
    AIModelConfig, GroqVerifierClient, load_ai_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "prompts" / "opening_resolver_v0.1.txt"
SCHEMA_PATH = REPO_ROOT / "schema" / "opening_resolution.schema.json"

PROMOTABLE = {"door", "window", "open_passage"}


@dataclass
class ResolutionResult:
    resolution: dict
    promoted: dict | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _insufficient(candidate: dict, reason: str, config: AIModelConfig,
                  evidence_ids: list[str]) -> dict:
    return {
        "candidateId": candidate["candidateId"],
        "surfaceId": candidate["surfaceId"],
        "semanticClass": "insufficient_evidence",
        "evidenceStatus": "insufficient_evidence",
        "evidenceFrameIds": evidence_ids or ["crop"],
        "reason": reason,
        "model": config.model,
        "provider": config.provider,
        "usage": None,
    }


def resolve_candidate(
    candidate: dict,
    crop_bytes: bytes,
    config: AIModelConfig | None = None,
    client: GroqVerifierClient | None = None,
    evidence_id: str = "crop",
) -> ResolutionResult:
    config = config or load_ai_config()
    prompt = PROMPT_PATH.read_text()
    schema = json.loads(SCHEMA_PATH.read_text())
    diagnostics = {
        "promptVersion": PROMPT_PATH.stem,
        "promptSha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schemaSha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
        "aiModelConfigHash": config.sha256,
        "grounding": "nominated_crop_only",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    user = (
        "Nominated candidate (geometry already measured the extent; you cannot "
        "change it):\n"
        + json.dumps({
            "candidateId": candidate["candidateId"],
            "surfaceId": candidate["surfaceId"],
            "candidateExtent_diagnostic": candidate.get("geometry"),
            "imageRegion": candidate.get("imageRegion"),
            "allowedEvidenceFrameIds": [evidence_id],
        }, separators=(",", ":"))
        + "\nClassify THIS crop of the nominated region. Return JSON with only "
        "candidateId, surfaceId, semanticClass, evidenceStatus, evidenceFrameIds, "
        "reason. Do not add any dimension field."
    )

    if not config.approved:
        resolution = _insufficient(
            candidate, "AI_MODEL_APPROVED is false", config, [evidence_id])
        diagnostics["validationResult"] = "not_run"
        return ResolutionResult(resolution, None, diagnostics)

    try:
        client = client or GroqVerifierClient()
        raw = client.complete_json(prompt, user, [(evidence_id, crop_bytes)],
                                   config.model or "")
    except Exception as exc:  # noqa: BLE001
        diagnostics["validationResult"] = "provider_failure"
        diagnostics["accessFailure"] = f"{type(exc).__name__}: {exc}"
        resolution = _insufficient(
            candidate, f"model unavailable: {type(exc).__name__}: {exc}",
            config, [evidence_id])
        return ResolutionResult(resolution, None, diagnostics)

    candidate_out = {
        "candidateId": raw.get("candidateId") or candidate["candidateId"],
        "surfaceId": raw.get("surfaceId") or candidate["surfaceId"],
        "semanticClass": raw.get("semanticClass"),
        "evidenceStatus": raw.get("evidenceStatus"),
        "evidenceFrameIds": raw.get("evidenceFrameIds") or [evidence_id],
        "reason": raw.get("reason") or "no reason",
        "model": raw.get("model") or config.model,
        "provider": "groq",
        "usage": raw.get("usage"),
    }

    errors = sorted(Draft202012Validator(schema).iter_errors(candidate_out),
                    key=lambda e: list(e.path))
    if errors:
        diagnostics["validationResult"] = "schema_rejected"
        diagnostics["schemaErrors"] = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in errors[:5])
        resolution = _insufficient(
            candidate,
            f"response rejected: {diagnostics['schemaErrors']}",
            config, [evidence_id])
        return ResolutionResult(resolution, None, diagnostics)

    if candidate_out["candidateId"] != candidate["candidateId"]:
        diagnostics["validationResult"] = "id_rejected"
        resolution = _insufficient(
            candidate, "model returned a different candidateId; not repaired",
            config, [evidence_id])
        return ResolutionResult(resolution, None, diagnostics)
    if candidate_out["surfaceId"] != candidate["surfaceId"]:
        diagnostics["validationResult"] = "id_rejected"
        resolution = _insufficient(
            candidate, "model returned a different surfaceId; not repaired",
            config, [evidence_id])
        return ResolutionResult(resolution, None, diagnostics)

    diagnostics["validationResult"] = "accepted"
    promoted = None
    if (candidate_out["evidenceStatus"] == "supported"
            and candidate_out["semanticClass"] in PROMOTABLE):
        extent = (candidate.get("geometry") or {})
        if all(extent.get(k) is not None for k in ("width_m", "height_m")):
            promoted = {
                "id": candidate["candidateId"],
                "surfaceId": candidate["surfaceId"],
                "type": candidate_out["semanticClass"],
                "observationState": "directly_observed",
                "dimensions": {
                    "width_m": extent["width_m"],
                    "height_m": extent["height_m"],
                    "sillHeight_m": extent.get("sillHeight_m"),
                },
                "producer": "geometry_pipeline",
                "aiSemanticClass": candidate_out["semanticClass"],
                "aiEvidenceStatus": candidate_out["evidenceStatus"],
            }
            diagnostics["promoted"] = True
        else:
            diagnostics["promoted"] = False
            diagnostics["promotionBlocked"] = "candidate has no geometry-owned extent"
    else:
        diagnostics["promoted"] = False
        if candidate_out["semanticClass"] == "insufficient_evidence":
            diagnostics["abstention"] = True

    return ResolutionResult(candidate_out, promoted, diagnostics)


def scene_level_recognition_does_not_promote(resolution: dict) -> bool:
    """A response that never names the nominated candidate cannot promote it."""
    return resolution.get("candidateId") is not None
