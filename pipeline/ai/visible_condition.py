"""Visible Condition Grounding.

The VLM may name a visible condition and propose an image region. Geometry
alone may register that region onto a named surface and emit a quantity.
AI output never contains area, linear extent, or restoration scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from pipeline.loss_preview.preview import register_region_to_surface

from .make_condition_fixture import write_fixture
from .verifier import (
    AIModelConfig, GroqVerifierClient, load_ai_config, protected_geometry_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "prompts" / "visible_condition_v0.1.txt"
SCHEMA_PATH = REPO_ROOT / "schema" / "visible_condition.schema.json"
RECORDED_PROPOSAL = REPO_ROOT / "samples" / "ai_condition_eval" / "accepted_proposal.json"
FIXTURE_DIR = REPO_ROOT / "samples" / "ai_condition_eval"

SUPPORTED_CLASSES = {
    "staining", "discoloration", "cracking", "scorching",
    "moisture_like", "other_visible_anomaly",
}
METRIC_KEYS = {
    "affectedArea_m2", "area_m2", "area", "sqft", "squareFootage",
    "square_footage", "linearFeet", "linear_feet", "quantity",
    "width_m", "height_m", "verticalExtent_m", "repairQuantity",
}


@dataclass
class ConditionResult:
    record: dict
    proposal: dict | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _abstain(status: str, reason: str, config: AIModelConfig,
             surface_id: str, evidence_ids: list[str],
             condition_class: str = "none") -> dict:
    return {
        "status": status,
        "conditionClass": condition_class,
        "surfaceId": surface_id,
        "evidenceFrameIds": evidence_ids,
        "reason": reason,
        "model": config.model,
        "provider": config.provider,
        "usage": None,
    }


def _contains_metric_keys(payload: Any) -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in METRIC_KEYS:
                found.append(key)
            found.extend(_contains_metric_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_contains_metric_keys(item))
    return found


def _normalize_region(region: dict, image_size: tuple[int, int]) -> dict | None:
    if not isinstance(region, dict):
        return None
    if all(k in region for k in ("x0", "y0", "x1", "y1")):
        x0, y0, x1, y1 = (float(region[k]) for k in ("x0", "y0", "x1", "y1"))
    elif isinstance(region.get("polygon"), list) and region["polygon"]:
        xs = [float(p[0]) for p in region["polygon"]]
        ys = [float(p[1]) for p in region["polygon"]]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    else:
        return None
    width, height = image_size
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.0:
        x0, x1 = x0 / width, x1 / width
        y0, y1 = y0 / height, y1 / height
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        return None
    return {"x0": round(x0, 6), "y0": round(y0, 6),
            "x1": round(x1, 6), "y1": round(y1, 6)}


def _intrinsics_for(model: dict, view: dict) -> dict:
    recorded = model["provenance"].get("evidenceIntrinsics")
    if recorded:
        return recorded
    stream = view.get("diagnostics", {}).get("intrinsicsStream", "rgb")
    fallback = model["provenance"].get("captureIntrinsics", {}).get(stream)
    if fallback:
        return fallback
    raise ValueError(
        "no intrinsics are recorded for the evidence frames; the condition "
        "cannot be registered")


def _register_region(region: dict, surface: dict, view: dict, model: dict) -> tuple[dict, dict]:
    """Register a model-proposed region; inset if a corner ray misses the plane.

    A full-image proposal often includes rays that never meet the wall. Geometry
    retries a conservative interior of the *same* proposed region. The VLM still
    proposed the region; it does not produce the quantity.
    """
    pose = np.array(view.get("cameraToWorldCanonical") or view["cameraToWorld"],
                    dtype=np.float64)
    intrinsics = _intrinsics_for(model, view)
    registration = register_region_to_surface(region, surface, pose, intrinsics)
    used = dict(region)
    inset_applied = 0.0
    surface_area = None
    dims = surface.get("dimensions") or {}
    if dims.get("width_m") and dims.get("height_m"):
        surface_area = float(dims["width_m"]) * float(dims["height_m"])

    def _too_large(reg: dict) -> bool:
        if reg.get("status") != "registered" or surface_area is None:
            return False
        return float(reg["affectedArea_m2"]) > surface_area * 1.25

    for inset in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        if registration["status"] == "registered" and not _too_large(registration):
            break
        used = {
            "x0": round(min(0.49, region["x0"] + inset), 6),
            "y0": round(min(0.49, region["y0"] + inset), 6),
            "x1": round(max(0.51, region["x1"] - inset), 6),
            "y1": round(max(0.51, region["y1"] - inset), 6),
        }
        if used["x1"] - used["x0"] < 0.05 or used["y1"] - used["y0"] < 0.05:
            break
        registration = register_region_to_surface(used, surface, pose, intrinsics)
        inset_applied = inset
    if _too_large(registration):
        registration = {
            "status": "unresolved",
            "reason": (
                "the registered area exceeds the named surface, so a grazing-ray "
                "quantity is not emitted"
            ),
        }
    return registration, {"normalizedImageRegion": used, "geometryInset": inset_applied}


def _select_wall_and_view(model: dict) -> tuple[dict, dict]:
    evidence = model.get("evidence") or []
    for view in evidence:
        for surface_id in view.get("visibleSurfaceIds") or []:
            surface = next((s for s in model["surfaces"] if s["id"] == surface_id), None)
            if (surface and surface.get("type") == "wall"
                    and surface.get("observationState") == "directly_observed"):
                return surface, view
    raise ValueError("no directly observed wall has a registered evidence view")


def ground_visible_condition(
    model: dict,
    image_bytes: bytes,
    surface: dict,
    view: dict,
    config: AIModelConfig | None = None,
    client: GroqVerifierClient | None = None,
    image_size: tuple[int, int] = (256, 192),
) -> ConditionResult:
    config = config or load_ai_config()
    prompt = PROMPT_PATH.read_text()
    schema = json.loads(SCHEMA_PATH.read_text())
    surface_id = surface["id"]
    evidence_id = view["id"]
    before = protected_geometry_digest(model)
    diagnostics: dict[str, Any] = {
        "promptVersion": PROMPT_PATH.stem,
        "promptSha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schemaSha256": _sha(SCHEMA_PATH),
        "aiModelConfigHash": config.sha256,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "DEVELOPMENT CONDITION FIXTURE",
        "isRealDamageEvidence": False,
        "geometryDigestBefore": before,
    }

    user = (
        "Named surface and registered evidence (you cannot change geometry):\n"
        + json.dumps({
            "surfaceId": surface_id,
            "allowedEvidenceFrameIds": [evidence_id],
            "imageSizePx": {"width": image_size[0], "height": image_size[1]},
            "regionCoordinates": "normalized 0-1 rectangle x0,y0,x1,y1 of this image",
            "fixtureLabel": (
                "DEVELOPMENT CONDITION FIXTURE of a wall surface. "
                "No furniture is present. Marks on the wall are wall conditions."
            ),
        }, separators=(",", ":"))
        + "\nReturn JSON with only status, conditionClass, surfaceId, "
        "evidenceFrameIds, reason, and region when status is supported. "
        "Do not add any measurement field."
    )

    if not config.approved:
        record = _abstain("insufficient_evidence",
                          "AI_MODEL_APPROVED is false", config,
                          surface_id, [evidence_id])
        diagnostics["validationResult"] = "not_run"
        diagnostics["geometryDigestAfter"] = protected_geometry_digest(model)
        diagnostics["geometryMutationCount"] = 0
        return ConditionResult(record, None, diagnostics)

    try:
        client = client or GroqVerifierClient()
        raw = client.complete_json(prompt, user, [(evidence_id, image_bytes)],
                                   config.model or "")
    except Exception as exc:  # noqa: BLE001
        diagnostics["validationResult"] = "provider_failure"
        diagnostics["accessFailure"] = f"{type(exc).__name__}: {exc}"
        record = _abstain(
            "insufficient_evidence",
            f"model unavailable: {type(exc).__name__}: {exc}",
            config, surface_id, [evidence_id])
        diagnostics["geometryDigestAfter"] = protected_geometry_digest(model)
        diagnostics["geometryMutationCount"] = 0
        return ConditionResult(record, None, diagnostics)

    metric_keys = _contains_metric_keys(raw)
    if metric_keys:
        diagnostics["validationResult"] = "metric_rejected"
        record = _abstain(
            "insufficient_evidence",
            "model emitted a metric field and was rejected: "
            + ", ".join(sorted(set(metric_keys))),
            config, surface_id, [evidence_id])
        diagnostics["geometryDigestAfter"] = protected_geometry_digest(model)
        diagnostics["geometryMutationCount"] = 0
        return ConditionResult(record, None, diagnostics)

    candidate = {
        "status": raw.get("status"),
        "conditionClass": raw.get("conditionClass"),
        "surfaceId": raw.get("surfaceId") or surface_id,
        "evidenceFrameIds": raw.get("evidenceFrameIds") or [evidence_id],
        "reason": raw.get("reason") or "no reason",
        "model": raw.get("model") or config.model,
        "provider": "groq",
        "usage": raw.get("usage"),
    }
    if raw.get("region") is not None:
        normalized = _normalize_region(raw["region"], image_size)
        if normalized is not None:
            candidate["region"] = normalized
        else:
            candidate["region"] = raw["region"]

    errors = sorted(Draft202012Validator(schema).iter_errors(candidate),
                    key=lambda e: list(e.path))
    if errors:
        diagnostics["validationResult"] = "schema_rejected"
        diagnostics["schemaErrors"] = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in errors[:5])
        record = _abstain(
            "insufficient_evidence",
            f"response rejected: {diagnostics['schemaErrors']}",
            config, surface_id, [evidence_id])
        diagnostics["geometryDigestAfter"] = protected_geometry_digest(model)
        diagnostics["geometryMutationCount"] = 0
        return ConditionResult(record, None, diagnostics)

    if candidate["surfaceId"] != surface_id:
        diagnostics["validationResult"] = "id_rejected"
        record = _abstain(
            "insufficient_evidence",
            "model returned a different surfaceId; not repaired",
            config, surface_id, [evidence_id])
        diagnostics["geometryDigestAfter"] = protected_geometry_digest(model)
        diagnostics["geometryMutationCount"] = 0
        return ConditionResult(record, None, diagnostics)
    allowed_ids = {evidence_id}
    if not set(candidate["evidenceFrameIds"]) <= allowed_ids:
        diagnostics["validationResult"] = "id_rejected"
        record = _abstain(
            "insufficient_evidence",
            "model cited an evidence id that was not supplied; not repaired",
            config, surface_id, [evidence_id])
        diagnostics["geometryDigestAfter"] = protected_geometry_digest(model)
        diagnostics["geometryMutationCount"] = 0
        return ConditionResult(record, None, diagnostics)

    diagnostics["validationResult"] = "accepted"
    candidate["model"] = config.model
    candidate["provider"] = config.provider
    proposal = None

    if candidate["status"] == "supported" and candidate["conditionClass"] in SUPPORTED_CLASSES:
        region = _normalize_region(candidate.get("region") or {}, image_size)
        if region is None:
            diagnostics["validationResult"] = "region_rejected"
            candidate["status"] = "insufficient_evidence"
            candidate["reason"] = "supported status without a usable image region"
        else:
            original_region = dict(region)
            registration, region_meta = _register_region(region, surface, view, model)
            region = region_meta["normalizedImageRegion"]
            proposal = {
                "conditionId": "condition-model-001",
                "label": "DEVELOPMENT CONDITION FIXTURE",
                "isRealDamageEvidence": False,
                "surfaceId": surface_id,
                "status": "proposed_experimental",
                "conditionClass": candidate["conditionClass"],
                "evidenceFrameIds": [evidence_id],
                "affectedRegion": {
                    "method": "model_proposed_visual_region",
                    "normalizedImageRegion": region,
                    "modelProposedRegion": original_region,
                    "geometryInset": region_meta["geometryInset"],
                    "producer": "ai_visible_condition",
                },
                "registration": registration,
                "reviewStatus": "human_review_required",
                "provenance": {
                    "aiProducer": config.model,
                    "provider": config.provider,
                    "promptVersion": PROMPT_PATH.stem,
                    "promptSha256": diagnostics["promptSha256"],
                    "quantityProducer": "geometry_pipeline",
                    "aiProducerNote": (
                        "The VLM proposed a visible condition and an image region. "
                        "It did not measure area. Geometry registered the region."),
                },
            }
            if registration["status"] == "registered":
                proposal["quantity"] = {
                    "affectedArea_m2": registration["affectedArea_m2"],
                    "verticalExtent_m": registration["verticalExtent_m"],
                    "producer": "geometry_pipeline",
                    "method": registration["method"],
                    "note": (
                        "Measured by casting the model-proposed region onto the "
                        "fitted surface plane through the real camera pose and "
                        "intrinsics. The VLM did not produce this quantity."),
                }
                diagnostics["chain"] = "completed"
                diagnostics["geometryInset"] = region_meta["geometryInset"]
            else:
                diagnostics["chain"] = "unresolved_registration"
                diagnostics["registrationReason"] = registration.get("reason")

    diagnostics["geometryDigestAfter"] = protected_geometry_digest(model)
    diagnostics["geometryMutationCount"] = int(
        diagnostics["geometryDigestAfter"] != before)
    return ConditionResult(candidate, proposal, diagnostics)


def attach_recorded_proposal(preview: dict, model: dict) -> dict:
    """Re-register a previously accepted model region if IDs still resolve.

    The recorded file stores the model region only. Quantity is always
    recomputed by geometry for the current model.
    """
    if not RECORDED_PROPOSAL.is_file():
        return preview
    recorded = json.loads(RECORDED_PROPOSAL.read_text())
    surface_id = recorded.get("surfaceId")
    evidence_ids = recorded.get("evidenceFrameIds") or []
    region = recorded.get("normalizedImageRegion")
    if not (surface_id and evidence_ids and isinstance(region, dict)):
        return preview
    surface = next((s for s in model.get("surfaces", []) if s["id"] == surface_id), None)
    view = next((v for v in model.get("evidence", []) if v["id"] == evidence_ids[0]), None)
    if surface is None or view is None:
        preview["modelGeneratedGrounding"] = "ids_do_not_resolve"
        return preview
    pose = np.array(view.get("cameraToWorldCanonical") or view["cameraToWorld"],
                    dtype=np.float64)
    registration, region_meta = _register_region(region, surface, view, model)
    proposal = {
        "conditionId": recorded.get("conditionId", "condition-model-001"),
        "label": "DEVELOPMENT CONDITION FIXTURE",
        "isRealDamageEvidence": False,
        "surfaceId": surface_id,
        "status": "proposed_experimental",
        "conditionClass": recorded.get("conditionClass"),
        "evidenceFrameIds": [view["id"]],
        "affectedRegion": {
            "method": "model_proposed_visual_region",
            "normalizedImageRegion": region_meta["normalizedImageRegion"],
            "modelProposedRegion": region,
            "geometryInset": region_meta["geometryInset"],
            "producer": "ai_visible_condition",
        },
        "registration": registration,
        "reviewStatus": "human_review_required",
        "provenance": recorded.get("provenance") or {
            "quantityProducer": "geometry_pipeline",
        },
    }
    if registration["status"] == "registered":
        proposal["quantity"] = {
            "affectedArea_m2": registration["affectedArea_m2"],
            "verticalExtent_m": registration["verticalExtent_m"],
            "producer": "geometry_pipeline",
            "method": registration["method"],
            "note": (
                "Recomputed by geometry from the recorded model-proposed region. "
                "The VLM did not produce this quantity."),
        }
        preview["modelGeneratedGrounding"] = "completed"
    else:
        preview["modelGeneratedGrounding"] = "unresolved_registration"
    preview.setdefault("proposals", []).append(proposal)
    return preview


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    from io import BytesIO
    from PIL import Image
    image = Image.open(BytesIO(image_bytes))
    return image.size


def write_accepted_proposal(result: ConditionResult, path: Path | None = None) -> Path | None:
    if result.proposal is None:
        return None
    if result.proposal.get("registration", {}).get("status") != "registered":
        return None
    if "quantity" not in result.proposal:
        return None
    path = Path(path or RECORDED_PROPOSAL)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conditionId": result.proposal["conditionId"],
        "surfaceId": result.proposal["surfaceId"],
        "conditionClass": result.proposal["conditionClass"],
        "evidenceFrameIds": result.proposal["evidenceFrameIds"],
        "normalizedImageRegion": result.proposal["affectedRegion"]["normalizedImageRegion"],
        "modelProposedRegion": result.proposal["affectedRegion"].get(
            "modelProposedRegion"),
        "provenance": result.proposal["provenance"],
        "note": (
            "Model-proposed region only. Quantity is never stored here; geometry "
            "recomputes it at attach/registration time."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_live(model_path: Path, output_path: Path,
             image_path: Path | None = None) -> dict:
    model = json.loads(Path(model_path).read_text())
    if image_path is None:
        write_fixture()
        image_file = FIXTURE_DIR / "stained_evidence.png"
    else:
        image_file = Path(image_path)
    image_bytes = image_file.read_bytes()
    surface, view = _select_wall_and_view(model)
    result = ground_visible_condition(
        model, image_bytes, surface, view, image_size=_image_size(image_bytes))
    artifact = {
        "label": "DEVELOPMENT CONDITION FIXTURE",
        "isRealDamageEvidence": False,
        "valueClaim": (
            "plain image VLM: possible staining here. spatial pipeline: possible "
            "staining on a named surface + evidence IDs + registered image "
            "polygon + camera pose + surface geometry -> affected area calculated "
            "deterministically by geometry -> auditable human-review record. "
            "The VLM itself did not measure the affected area."
        ),
        "modelResponse": result.record,
        "proposal": result.proposal,
        "diagnostics": result.diagnostics,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    write_accepted_proposal(result)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Visible condition grounding")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()
    artifact = run_live(args.model, args.out, args.image)
    chain = artifact["diagnostics"].get("chain", artifact["modelResponse"]["status"])
    print(json.dumps({
        "status": artifact["modelResponse"]["status"],
        "conditionClass": artifact["modelResponse"].get("conditionClass"),
        "chain": chain,
        "geometryMutationCount": artifact["diagnostics"].get("geometryMutationCount"),
        "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
