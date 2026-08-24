"""Rule-based evidence-quality labels.

Deliberately not a score. The plan forbids weighted formulas without empirical
justification, and there is none here, so the first matching rule in
`confidence_rules_v0.1.json` decides the label and the inputs that triggered it
travel with it. A label with missing inputs is `unresolved`, never a guess.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = REPO_ROOT / "config" / "confidence_rules_v0.1.json"

OBSERVED_STATES = {"directly_observed", "partially_observed"}


class ConfidenceRulesError(Exception):
    """The confidence rules are missing or malformed."""


@dataclass(frozen=True)
class ConfidenceRules:
    rules_id: str
    method: str
    raw: dict
    sha256: str

    def _threshold(self, name: str) -> float:
        try:
            return self.raw["thresholds"][name]["value"]
        except KeyError as exc:
            raise ConfidenceRulesError(
                f"threshold '{name}' is not defined in {self.rules_id}") from exc

    def label(
        self,
        observation_state: str,
        rms_residual_m: float | None,
        coverage_fraction: float | None,
        contributing_frames: int | None,
        extra_inputs: dict[str, Any] | None = None,
    ) -> dict:
        inputs: dict[str, Any] = {
            "observationState": observation_state,
            "rmsResidual_m": rms_residual_m,
            "coverageFraction": coverage_fraction,
            "contributingFrames": contributing_frames,
        }
        if extra_inputs:
            inputs.update(extra_inputs)

        def decided(label: str, rule_id: str) -> dict:
            return {
                "label": label,
                "calibrated": False,
                "method": self.method,
                "rulesVersion": self.rules_id,
                "inputs": inputs,
                "ruleTriggered": rule_id,
            }

        if observation_state == "unresolved":
            return decided("unresolved", "unresolved-no-evidence")
        if observation_state == "inferred":
            return decided("unresolved", "unresolved-inferred-geometry")
        if rms_residual_m is None or coverage_fraction is None or contributing_frames is None:
            return decided("unresolved", "unresolved-no-evidence")
        if observation_state not in OBSERVED_STATES:
            return decided("unresolved", "unresolved-no-evidence")

        if (observation_state == "directly_observed"
                and rms_residual_m <= self._threshold("high_max_rms_residual_m")
                and coverage_fraction >= self._threshold("high_min_coverage_fraction")
                and contributing_frames >= self._threshold("high_min_contributing_frames")):
            return decided("high", "high-direct-strong-support")

        if (rms_residual_m <= self._threshold("medium_max_rms_residual_m")
                and coverage_fraction >= self._threshold("medium_min_coverage_fraction")
                and contributing_frames >= self._threshold("medium_min_contributing_frames")):
            return decided("medium", "medium-direct-partial-support")

        return decided("low", "low-weak-support")


def load_confidence_rules(path: Path | None = None) -> ConfidenceRules:
    path = Path(path or DEFAULT_RULES)
    if not path.is_file():
        raise ConfidenceRulesError(f"confidence rules '{path}' do not exist")
    text = path.read_text()
    raw = json.loads(text)
    if raw.get("calibrated") is not False:
        raise ConfidenceRulesError(
            "confidence rules must declare calibrated=false; these labels are "
            "evidence-quality heuristics and must never imply calibration")
    for field in ("rulesId", "method", "rules", "thresholds"):
        if field not in raw:
            raise ConfidenceRulesError(f"'{path.name}' is missing required field '{field}'")
    return ConfidenceRules(
        rules_id=raw["rulesId"],
        method=raw["method"],
        raw=raw,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
