"""The Spatial AI Verifier: a bounded, fail-closed integration boundary.

Two properties define this module, and both are enforced in code rather than
promised in prose.

**It never runs without an approved model.** The operator, not the agent, names
the provider and model in `ai_model_config.json`. Without `AI_MODEL_APPROVED`
the verifier returns a structured `not_run` and the pipeline continues.

**It cannot touch geometry.** Responses are schema-validated, then every
surface ID and evidence frame ID is checked against the model that was actually
built. A finding citing anything else is dropped with a reason. The verifier
returns an assessment object; the caller attaches it alongside geometry and a
regression test asserts that attaching it changes no measurement.

The Groq client is the live path for the operator-approved model
`qwen/qwen3.6-27b`. It does not silently substitute another provider or model.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "ai_model_config.json"
ASSESSMENT_SCHEMA = REPO_ROOT / "schema" / "ai_assessment.schema.json"

SCHEMA_VERSION = "0.1"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_IMAGES_PER_REQUEST = 3
GROQ_IMAGES_PER_REQUEST = 1  # Groq on_demand TPM 8000 cannot fit 3 vision images.
REQUEST_TIMEOUT_S = 90.0
MAX_429_ATTEMPTS = 4
MAX_TIMEOUT_ATTEMPTS = 2
MAX_RETRY_AFTER_S = 30.0
PLACEHOLDER_API_KEY = "your_groq_api_key_here"


def load_dotenv(path: Path | None = None) -> None:
    """Load environment variables from a .env file into os.environ if present."""
    env_file = Path(path or REPO_ROOT / ".env")
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


# Fields the verifier may never write. Checked after every run, not trusted.
PROTECTED_MODEL_KEYS = ("surfaces", "measurements", "rooms", "openings",
                        "coordinateSystem", "damage", "scope")


class VerifierError(Exception):
    """The verifier could not be configured or its response was unusable."""


class GroqRateLimitError(VerifierError):
    """HTTP 429. Retry only after honouring the provider Retry-After header."""

    def __init__(self, message: str, retry_after_s: float | None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class GroqTimeoutError(VerifierError):
    """The Groq request exceeded the client timeout."""


@dataclass
class AIModelConfig:
    config_id: str
    approved: bool
    provider: str | None
    model: str | None
    raw: dict
    sha256: str

    @property
    def image_limit(self) -> int:
        return int(self.raw.get("imageLimit", MAX_IMAGES_PER_REQUEST))

    @property
    def minimum_images(self) -> int:
        return int(self.raw.get("minimumImages", 3))


def load_ai_config(path: Path | None = None) -> AIModelConfig:
    path = Path(path or DEFAULT_CONFIG)
    if not path.is_file():
        raise VerifierError(f"AI model configuration '{path}' does not exist")
    text = path.read_text()
    raw = json.loads(text)
    approved = bool(raw.get("AI_MODEL_APPROVED", False))
    provider, model = raw.get("provider"), raw.get("model")
    if approved and not (provider and model):
        raise VerifierError(
            "AI_MODEL_APPROVED is true but provider and model are not both set; "
            "refusing to guess which model was approved")
    return AIModelConfig(
        config_id=raw.get("configId", path.stem),
        approved=approved, provider=provider, model=model,
        raw=raw, sha256=hashlib.sha256(text.encode()).hexdigest())


class MultimodalClient(Protocol):
    """What the verifier needs from a provider. Deliberately tiny."""

    def assess(self, system: str, summary: dict, images: list[tuple[str, bytes]],
               response_schema: dict, model: str) -> dict:
        ...


@dataclass
class VerifierResult:
    assessment: dict
    rejected_findings: list[dict] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def protected_geometry_digest(model: dict) -> str:
    """Stable hash of the fields AI is forbidden to change."""
    payload = {key: model.get(key) for key in PROTECTED_MODEL_KEYS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _not_run(reason: str, config: AIModelConfig, prompt_version: str) -> dict:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": "not_run",
        "notRunReason": reason,
        "model": config.model,
        "provider": config.provider,
        "promptVersion": prompt_version,
        "generatedAt": None,
        "roomTypeHypothesis": None,
        "findings": [],
        "usage": None,
    }


def build_geometry_summary(model: dict) -> dict:
    """The compact, read-only hypothesis the verifier is shown.

    Dimensions are included so a model can notice gross implausibility, and for
    no other reason: it has no authority to revise them, and the schema gives it
    nowhere to try.
    """
    return {
        "roomId": model["rooms"][0]["id"] if model["rooms"] else None,
        "surfaces": [
            {
                "surfaceId": surface["id"],
                "type": surface["type"],
                "observationState": surface["observationState"],
                "width_m": surface["dimensions"]["width_m"],
                "height_m": surface["dimensions"]["height_m"],
            }
            for surface in model["surfaces"]
        ],
        "unresolvedOpenings": [
            {"openingId": opening["id"], "surfaceId": opening["surfaceId"]}
            for opening in model["openings"]
            if opening["observationState"] == "unresolved"
        ],
        "allowedEvidenceFrameIds": [view["id"] for view in model.get("evidence", [])],
        "note": ("Geometry produced every number here. You are reviewing visual "
                 "consistency and cannot change any of them."),
    }


def run_verifier(
    model: dict,
    evidence_dir: Path,
    config: AIModelConfig | None = None,
    client: MultimodalClient | None = None,
    prompt_path: Path | None = None,
) -> VerifierResult:
    config = config or load_ai_config()
    prompt_path = Path(prompt_path or REPO_ROOT / "prompts" / "spatial_verifier_v0.1.txt")
    prompt_version = prompt_path.stem

    if not prompt_path.is_file():
        raise VerifierError(f"prompt '{prompt_path}' does not exist; there is no "
                            f"unversioned fallback prompt")
    system = prompt_path.read_text()

    views = model.get("evidence", [])
    diagnostics: dict[str, Any] = {
        "aiModelConfigId": config.config_id,
        "aiModelConfigHash": config.sha256,
        "promptVersion": prompt_version,
        "promptSha256": hashlib.sha256(system.encode()).hexdigest(),
        "evidenceViewCount": len(views),
        "approved": config.approved,
        "geometryDigestBefore": protected_geometry_digest(model),
    }

    if not config.approved:
        diagnostics["validationResult"] = "not_run"
        return VerifierResult(
            assessment=_not_run(
                "AI_MODEL_APPROVED is false. No provider or model has been approved "
                "by an operator, and the agent must not select one. The integration "
                "boundary, prompts, and response schema are in place and will run "
                "unchanged once approval is recorded.",
                config, prompt_version),
            diagnostics=diagnostics)

    if len(views) < config.minimum_images:
        diagnostics["validationResult"] = "not_run"
        return VerifierResult(
            assessment=_not_run(
                f"only {len(views)} registered evidence views are available; the "
                f"verifier contract requires at least {config.minimum_images}",
                config, prompt_version),
            diagnostics=diagnostics)

    if client is None:
        try:
            client = build_client(config)
        except VerifierError as exc:
            diagnostics["validationResult"] = "provider_failure"
            return VerifierResult(
                assessment=_not_run(str(exc), config, prompt_version),
                diagnostics=diagnostics)

    summary = build_geometry_summary(model)
    image_limit = min(int(config.image_limit), MAX_IMAGES_PER_REQUEST)
    images: list[tuple[str, bytes]] = []
    for view in views[:image_limit]:
        path = Path(evidence_dir).parent / view["path"]
        if not path.is_file():
            continue
        images.append((view["id"], path.read_bytes()))

    diagnostics["evidenceFrameIdsPrepared"] = [view_id for view_id, _ in images]
    diagnostics["imagesPrepared"] = len(images)
    sent = min(len(images), GROQ_IMAGES_PER_REQUEST) if config.provider == "groq" else len(images)
    diagnostics["imagesSent"] = sent
    diagnostics["evidenceFrameIds"] = [view_id for view_id, _ in images[:sent]]
    diagnostics["imageLimit"] = image_limit
    if config.provider == "groq":
        diagnostics["groqImagesPerRequest"] = GROQ_IMAGES_PER_REQUEST

    schema = json.loads(ASSESSMENT_SCHEMA.read_text())
    started = time.perf_counter()
    try:
        raw = client.assess(system=system, summary=summary, images=images,
                            response_schema=schema, model=config.model or "")
    except Exception as exc:  # noqa: BLE001 - any provider failure fails closed
        diagnostics["latencyMs"] = int((time.perf_counter() - started) * 1000)
        diagnostics["validationResult"] = "provider_failure"
        diagnostics["accessFailure"] = f"{type(exc).__name__}: {exc}"
        return VerifierResult(
            assessment=_not_run(
                f"the approved model could not be reached or returned an "
                f"unusable response: {type(exc).__name__}: {exc}",
                config, prompt_version),
            diagnostics=diagnostics)
    diagnostics["latencyMs"] = int((time.perf_counter() - started) * 1000)

    result = validate_assessment(raw, model, config, prompt_version, diagnostics)
    result.diagnostics["geometryDigestAfter"] = protected_geometry_digest(model)
    return result


def validate_assessment(
    raw: dict,
    model: dict,
    config: AIModelConfig,
    prompt_version: str,
    diagnostics: dict | None = None,
) -> VerifierResult:
    """Fail closed: reject the whole response, or drop findings that do not resolve."""
    diagnostics = dict(diagnostics or {})
    schema = json.loads(ASSESSMENT_SCHEMA.read_text())

    candidate = dict(raw)
    candidate["schemaVersion"] = SCHEMA_VERSION
    candidate["promptVersion"] = prompt_version
    candidate["provider"] = config.provider
    candidate.setdefault("model", config.model)
    candidate.setdefault("status", "completed")
    candidate.setdefault("usage", None)
    candidate.setdefault("notRunReason", None)
    candidate.setdefault("roomTypeHypothesis", None)
    candidate.setdefault("findings", [])

    errors = sorted(Draft202012Validator(schema).iter_errors(candidate),
                    key=lambda e: list(e.path))
    if errors:
        detail = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in errors[:5])
        diagnostics["validationResult"] = "schema_rejected"
        diagnostics["schemaErrors"] = detail
        return VerifierResult(
            assessment=_not_run(
                f"the response did not satisfy the assessment schema and was "
                f"rejected in full: {detail}", config, prompt_version),
            diagnostics=diagnostics)

    surface_ids = {s["id"] for s in model["surfaces"]}
    evidence_ids = {v["id"] for v in model.get("evidence", [])}

    kept, rejected = [], []
    for finding in candidate["findings"]:
        if finding["surfaceId"] not in surface_ids:
            rejected.append({"finding": finding, "reason":
                             f"surfaceId '{finding['surfaceId']}' resolves to no "
                             f"surface in this model"})
            continue
        unknown = [f for f in finding["evidenceFrameIds"] if f not in evidence_ids]
        if unknown:
            rejected.append({"finding": finding, "reason":
                             f"evidence frame ids {unknown} were never supplied"})
            continue
        kept.append(finding)

    candidate["findings"] = kept
    candidate["status"] = "completed"
    candidate["generatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    diagnostics.update({
        "validationResult": "accepted",
        "findingsAccepted": len(kept),
        "findingsRejected": len(rejected),
        "exactModel": candidate.get("model"),
        "rejectionPolicy": ("A finding that cites an unknown surface or an unsupplied "
                            "evidence frame is dropped. It is not repaired, and it is "
                            "not allowed through with a warning."),
    })
    return VerifierResult(assessment=candidate, rejected_findings=rejected,
                          diagnostics=diagnostics)


def build_client(config: AIModelConfig) -> MultimodalClient:
    """Construct the client for whichever provider an operator approved."""
    if config.provider == "anthropic":
        return AnthropicVerifierClient()
    if config.provider == "groq":
        return GroqVerifierClient()
    raise VerifierError(
        f"provider '{config.provider}' has no client implementation in this POC; "
        f"only 'anthropic' and 'groq' are implemented")


def _retry_after_seconds(headers: Any) -> float | None:
    raw = None
    if headers is not None:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None or raw == "":
        return None
    try:
        return min(float(raw), MAX_RETRY_AFTER_S)
    except (TypeError, ValueError):
        return None


def groq_chat_completions(payload: dict, api_key: str,
                          timeout_s: float = REQUEST_TIMEOUT_S) -> dict:
    """POST one Groq chat.completions request. No retry here; the client retries."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        GROQ_CHAT_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "spatial-ai/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise GroqTimeoutError(
            f"the Groq request timed out after {timeout_s:.0f}s") from exc
    except socket.timeout as exc:
        raise GroqTimeoutError(
            f"the Groq request timed out after {timeout_s:.0f}s") from exc
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 429:
            raise GroqRateLimitError(
                f"Groq HTTP 429: {err_body[:300]}",
                retry_after_s=_retry_after_seconds(exc.headers),
            ) from exc
        raise VerifierError(f"Groq HTTP {exc.code}: {err_body[:300]}") from exc
    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        if "timed out" in reason.lower():
            raise GroqTimeoutError(
                f"the Groq request timed out after {timeout_s:.0f}s") from exc
        raise VerifierError(f"Groq network error: {type(exc.reason).__name__}") from exc


def _encode_evidence_image(data: bytes) -> tuple[str, str]:
    """Encode one evidence image for Groq under the documented 8k TPM budget.

    The on-disk PNG is left untouched. Only the request copy is downscaled and
    JPEG-compressed. This is an engineering constraint from Groq's on_demand
    TPM limit (8000), not a geometry or model change.
    """
    from io import BytesIO
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return "image/png", base64.b64encode(data).decode("ascii")
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return "image/png", base64.b64encode(data).decode("ascii")
    max_edge = 128
    width, height = image.size
    if max(width, height) > max_edge:
        scale = max_edge / max(width, height)
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BILINEAR,
        )
        quality = 50
    else:
        quality = 85
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "image/jpeg", base64.b64encode(buffer.getvalue()).decode("ascii")


_GROQ_OUTPUT_CONTRACT = (
    "Return a JSON object with only these keys: schemaVersion (\"0.1\"), "
    "status (\"completed\"), model, promptVersion, generatedAt, "
    "roomTypeHypothesis (string or null), findings (array), usage (object or null), "
    "notRunReason (null), provider (\"groq\"). "
    "Each finding has only: surfaceId, status "
    "(verified|review_recommended|occluded|not_visible), semanticAgreement "
    "(boolean), reason, evidenceFrameIds (non-empty array of supplied ids). "
    "Optional finding keys: occlusionDescription, openingObservation "
    "(appears_to_be_an_opening|appears_to_be_unbroken_wall|cannot_tell). "
    "Do not add any other field. Do not state or revise any dimension."
)


class GroqVerifierClient:
    """Concrete Groq client for the operator-approved multimodal model.

    Reads `GROQ_API_KEY` only. The model id is the one passed from the approved
    config; environment variables cannot silently replace it. JSON mode is
    requested; every field is still validated locally. HTTP 429 retries honour
    the provider `Retry-After` header and are bounded.
    """

    def __init__(
        self,
        post: Callable[..., dict] | None = None,
        sleeper: Callable[[float], None] | None = None,
        environ: dict[str, str] | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self._post = post or groq_chat_completions
        self._sleep = sleeper or time.sleep
        self._environ = environ
        self._timeout_s = timeout_s

    def _api_key(self) -> str:
        load_dotenv()
        env = self._environ if self._environ is not None else os.environ
        api_key = env.get("GROQ_API_KEY")
        if not api_key or api_key == PLACEHOLDER_API_KEY:
            raise VerifierError(
                "GROQ_API_KEY is not set or contains default placeholder in .env")
        return api_key

    def _complete(self, system: str, content: list[dict], model: str) -> dict:
        if not model:
            raise VerifierError(
                "no model id was supplied by the approved configuration; "
                "refusing to guess or read GROQ_MODEL")
        api_key = self._api_key()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_completion_tokens": 4096,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "none",
        }
        result = None
        timeout_attempts = 0
        rate_limit_attempts = 0
        backoff_s = 1.0
        while True:
            try:
                result = self._post(payload, api_key, timeout_s=self._timeout_s)
                break
            except GroqRateLimitError as exc:
                rate_limit_attempts += 1
                if rate_limit_attempts >= MAX_429_ATTEMPTS:
                    raise VerifierError(
                        "Groq rate-limited the request and bounded retry was exhausted"
                    ) from exc
                wait_s = (exc.retry_after_s if exc.retry_after_s is not None
                          else backoff_s)
                wait_s = min(max(wait_s, 0.0), MAX_RETRY_AFTER_S)
                self._sleep(wait_s)
                backoff_s = min(backoff_s * 2.0, MAX_RETRY_AFTER_S)
            except GroqTimeoutError as exc:
                timeout_attempts += 1
                if timeout_attempts >= MAX_TIMEOUT_ATTEMPTS:
                    raise VerifierError(
                        f"the Groq request timed out after {self._timeout_s:.0f}s "
                        "and bounded retry was exhausted"
                    ) from exc
        try:
            text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VerifierError(
                "Groq returned a response with no message content") from exc
        if not text or not str(text).strip():
            raise VerifierError("Groq returned empty message content")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VerifierError(
                "Groq returned content that was not JSON and was rejected") from exc
        if not isinstance(parsed, dict):
            raise VerifierError("Groq returned JSON that was not an object")
        usage = result.get("usage") or {}
        parsed["usage"] = {
            "inputTokens": usage.get("prompt_tokens"),
            "outputTokens": usage.get("completion_tokens"),
        }
        parsed["model"] = result.get("model") or model
        parsed["provider"] = "groq"
        return parsed

    def complete_json(self, system: str, user_text: str,
                      images: list[tuple[str, bytes]], model: str) -> dict:
        """One JSON-mode Groq call with at most one image (TPM constraint)."""
        capped = images[:GROQ_IMAGES_PER_REQUEST]
        content: list[dict] = [{"type": "text", "text": user_text}]
        for view_id, data in capped:
            media_type, b64 = _encode_evidence_image(data)
            content.append({"type": "text", "text": f"Image {view_id}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            })
        return self._complete(system, content, model)

    def assess(self, system: str, summary: dict, images: list[tuple[str, bytes]],
               response_schema: dict, model: str) -> dict:
        capped = images[:GROQ_IMAGES_PER_REQUEST]
        content: list[dict] = [{
            "type": "text",
            "text": (
                "Reconstructed room summary:\n"
                + json.dumps(summary, separators=(",", ":"))
                + "\nAllowed evidenceFrameIds: "
                + json.dumps([view_id for view_id, _ in capped])
                + "\n"
                + _GROQ_OUTPUT_CONTRACT
            ),
        }]
        for view_id, data in capped:
            media_type, b64 = _encode_evidence_image(data)
            content.append({"type": "text", "text": f"Evidence frame {view_id}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            })
        return self._complete(system, content, model)


class AnthropicVerifierClient:
    """Concrete client for the Anthropic API, used only when an operator approves it.

    Structured output is requested through `output_config.format` against the
    same schema the response is then validated against, so a malformed response
    is unlikely — and still rejected if it happens.
    """

    def assess(self, system: str, summary: dict, images: list[tuple[str, bytes]],
               response_schema: dict, model: str) -> dict:
        try:
            import anthropic
        except ImportError as exc:
            raise VerifierError(
                "the anthropic SDK is not installed; run `pip install anthropic`"
            ) from exc

        client = anthropic.Anthropic()
        content: list[dict] = [{
            "type": "text",
            "text": ("Reconstructed room summary:\n"
                     + json.dumps(summary, indent=2)),
        }]
        for view_id, data in images:
            content.append({"type": "text", "text": f"Evidence frame {view_id}:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(data).decode("utf-8"),
                },
            })

        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema",
                                      "schema": response_schema}},
            messages=[{"role": "user", "content": content}],
        )
        text = next(block.text for block in response.content if block.type == "text")
        parsed = json.loads(text)
        parsed["usage"] = {
            "inputTokens": response.usage.input_tokens,
            "outputTokens": response.usage.output_tokens,
        }
        parsed["model"] = response.model
        return parsed
