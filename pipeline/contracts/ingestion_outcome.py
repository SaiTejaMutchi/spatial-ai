"""One plain sentence about a capture, from everything ingestion learned.

Operators do not want a validator's transcript. They want to know whether the
capture is usable, whether anything had to be repaired to make it so, and
whether a person needs to look. Four outcomes carry that; the transcript stays
underneath for whoever asks.

The mapping is deliberately conservative in one direction: anything that could
not be confirmed reads as needing review rather than as accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ACCEPTED = "accepted"
ACCEPTED_WITH_FIXES = "accepted_with_fixes"
NEEDS_REVIEW = "needs_review"
UNSUPPORTED = "unsupported"

HEADLINES = {
    ACCEPTED: "Capture accepted",
    ACCEPTED_WITH_FIXES: "Capture accepted with normalization fixes",
    NEEDS_REVIEW: "Capture needs review",
    UNSUPPORTED: "Capture unsupported",
}


@dataclass
class IngestionOutcome:
    state: str
    headline: str
    detail: str
    fixes: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "headline": self.headline,
            "detail": self.detail,
            "fixes": list(self.fixes),
            "concerns": list(self.concerns),
        }


def summarize(
    *,
    connector_failed: bool = False,
    connector_message: str | None = None,
    issues: list[Any] | None = None,
    resolution=None,
    excluded_frames: int = 0,
    repairs: list[str] | None = None,
) -> IngestionOutcome:
    """Reduce every ingestion signal to one of four outcomes."""
    if connector_failed:
        return IngestionOutcome(
            UNSUPPORTED, HEADLINES[UNSUPPORTED],
            connector_message or "This capture could not be read.",
            concerns=[connector_message] if connector_message else [])

    issues = issues or []
    errors = [i for i in issues if getattr(i, "severity", "") == "error"]
    warnings = [i for i in issues if getattr(i, "severity", "") == "warning"]

    if errors:
        return IngestionOutcome(
            UNSUPPORTED, HEADLINES[UNSUPPORTED],
            "The capture is missing something measurement depends on.",
            concerns=[i.message for i in errors])

    fixes = list(repairs or [])
    if excluded_frames:
        fixes.append(f"{excluded_frames} frame(s) without a usable pose were left out.")

    concerns = [i.message for i in warnings]

    if resolution is not None:
        if resolution.outcome == "verified":
            if resolution.basis != "declared_verified":
                fixes.append(
                    f"The vertical axis was resolved as '{resolution.axis}' from the "
                    f"observed floor and ceiling, because the source did not declare "
                    f"one that matched.")
        elif resolution.outcome == "ambiguous":
            concerns.append(
                "Which way is up could not be confirmed from this capture. "
                "Processing continues on the source's own declaration.")
        else:
            concerns.append(
                "No orientation of this capture looks like a room, so measurements "
                "from it would not be trustworthy.")

    if concerns:
        return IngestionOutcome(
            NEEDS_REVIEW, HEADLINES[NEEDS_REVIEW],
            "This capture can be processed, but something about it needs a human eye.",
            fixes=fixes, concerns=concerns)

    if fixes:
        return IngestionOutcome(
            ACCEPTED_WITH_FIXES, HEADLINES[ACCEPTED_WITH_FIXES],
            "This capture needed small repairs on the way in. Nothing was guessed.",
            fixes=fixes)

    return IngestionOutcome(
        ACCEPTED, HEADLINES[ACCEPTED],
        "This capture was read cleanly and is ready to measure.")
