"""Request, status, and artifact contracts for the local service.

Kept deliberately small. The service is orchestration: it holds scan state on
the filesystem and calls pipeline modules. It owns no geometry, no rendering,
no benchmark arithmetic, and no AI logic.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# The named stages the plan requires, in the order they run.
STAGES = (
    "upload_received",
    "source_detected",
    "connector_validation",
    "frame_alignment",
    "normalized_capture",
    "geometry",
    "canonical_model",
    "floorplan",
    "model_3d",
    "benchmark",
    "ai_review",
    "loss_preview",
    "complete",
)


class StageState(str, Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"
    skipped = "skipped"
    not_applicable = "not_applicable"


class FailureClass(str, Enum):
    """The plan's layered failure boundary, reported rather than inferred."""

    connector = "CONNECTOR_FAILURE"
    geometry = "GEOMETRY_GENERALIZATION_FAILURE"
    output = "OUTPUT_UI_FAILURE"


class ScanStatus(str, Enum):
    created = "created"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class CreateScanRequest(BaseModel):
    """Either a fixture already on disk, or an uploaded bundle path."""

    source_path: str = Field(..., description="Capture directory or bundle to ingest")
    label: str | None = None
    classification: str = "public_development_fixture"


class RenameScanRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class StageRecord(BaseModel):
    name: str
    state: StageState = StageState.pending
    detail: str | None = None
    startedAt: str | None = None
    finishedAt: str | None = None


class ArtifactRecord(BaseModel):
    name: str
    available: bool
    status: str            # development-only | final-verified | experimental | pending | unavailable
    path: str | None = None
    bytes: int | None = None
    description: str = ""


class ScanStatusResponse(BaseModel):
    scanId: str
    status: ScanStatus
    sourceType: str | None = None
    connector: str | None = None
    classification: str
    label: str | None = None
    stages: list[StageRecord]
    currentStage: str | None = None
    failureClass: FailureClass | None = None
    error: str | None = None
    createdAt: str
    updatedAt: str
    summary: dict = Field(default_factory=dict)
