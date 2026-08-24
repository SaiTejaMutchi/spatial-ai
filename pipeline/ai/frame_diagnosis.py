"""A bounded second opinion on an ambiguous capture frame.

This runs only when the deterministic layer in
`pipeline.contracts.frame_resolution` has already failed to choose, and it can
only ever produce an opinion. It cannot name an axis the deterministic layer did
not already score as physically plausible, it cannot emit a transform, and its
recommendation does not change what the pipeline does: the outcome stays
`ambiguous` and the capture still goes to a human. The value is the explanation
a reviewer reads, not a decision.

The same operator-approval rule as the spatial verifier applies. Without an
approved model this returns a structured `not_run` and ingestion continues.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .verifier import AIModelConfig, build_client, load_ai_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSIS_SCHEMA = REPO_ROOT / "schema" / "frame_diagnosis.schema.json"
PROMPT = REPO_ROOT / "prompts" / "frame_diagnosis_v0.1.txt"
PROMPT_VERSION = "frame_diagnosis_v0.1"

# The fields a candidate contributes to the model's view. Nothing else is sent:
# no pose, no image, no path on this machine.
EVIDENCE_KEYS = (
    "axis", "floorDetected", "ceilingDetected", "roomHeightM",
    "floorRmsMm", "ceilingRmsMm", "horizontalSupportFraction",
    "verticalStructureFraction", "trajectoryVerticalRatio", "plausible",
)


def build_evidence(resolution, source_type: str) -> dict[str, Any]:
    """The compact object the model is asked to reason about."""
    return {
        "source": source_type,
        "declaredAxis": resolution.declared_axis,
        "deterministicOutcome": resolution.outcome,
        "candidateFrames": [
            {key: getattr(candidate, key) for key in EVIDENCE_KEYS}
            for candidate in resolution.candidates
        ],
    }


def _not_run(reason: str, config: AIModelConfig) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "promptVersion": PROMPT_VERSION,
        "provider": config.provider,
        "model": config.model,
        "diagnosis": None,
    }


def diagnose_frame(
    resolution,
    source_type: str,
    config: AIModelConfig | None = None,
    client=None,
) -> dict[str, Any]:
    """Ask an approved model why the deterministic layer could not choose."""
    config = config or load_ai_config()

    if resolution.outcome == "verified":
        return _not_run("the vertical axis was resolved deterministically; "
                        "no diagnosis was needed", config)
    if not config.approved:
        return _not_run("AI_MODEL_APPROVED is false. No provider or model has been "
                        "approved, so no diagnosis was requested.", config)

    evidence = build_evidence(resolution, source_type)
    schema = json.loads(DIAGNOSIS_SCHEMA.read_text())
    client = client or build_client(config)

    try:
        raw = client.assess(PROMPT.read_text(), evidence, [], schema, config.model)
    except Exception as exc:  # provider, transport, or decoding failure
        return _not_run(f"the diagnosis request failed: {exc}", config)

    # The client folds provider telemetry into the parsed object. Keep it, but
    # out of the way: the schema is closed so the model cannot smuggle fields in.
    telemetry = {key: raw.pop(key) for key in ("usage", "model", "provider")
                 if key in raw}

    errors = sorted(Draft202012Validator(schema).iter_errors(raw), key=lambda e: e.path)
    if errors:
        return _not_run(
            "the response did not satisfy the diagnosis schema and was discarded: "
            + errors[0].message, config)

    # A recommended axis is only meaningful if the deterministic layer already
    # found it physically plausible. Anything else is dropped rather than argued
    # with, because this layer does not get to introduce an orientation.
    plausible = {c.axis for c in resolution.candidates if c.plausible}
    recommended = raw.get("recommendedAxis")
    dropped = None
    if recommended is not None and recommended not in plausible:
        dropped = recommended
        raw["recommendedAxis"] = None

    return {
        "status": "completed",
        "reason": None,
        "promptVersion": PROMPT_VERSION,
        "provider": config.provider,
        "model": config.model,
        "diagnosis": raw,
        "usage": telemetry.get("usage"),
        "advisoryOnly": True,
        "droppedRecommendedAxis": dropped,
        "note": "Advisory only. The capture's outcome is unchanged by this response; "
                "an unresolved frame still goes to human review.",
    }
