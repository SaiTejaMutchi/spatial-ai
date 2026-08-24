"""Model-versus-reference error arithmetic and the assignment's Track A gates.

Nothing here may change a model value. The comparison reads two numbers and
reports the difference; if either is missing the row says so rather than
quietly dropping out of the table, because a benchmark that silently omits its
failures is not a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Straight from the assignment. Recorded as data so the report can cite the
# gate alongside each result instead of asserting a verdict in prose.
ASSIGNMENT_GATES: dict[str, dict[str, Any]] = {
    "wall_length": {"description": "at most 1% or 2 cm, whichever is larger",
                    "relative": 0.01, "absolute_m": 0.02, "mode": "larger"},
    "room_length": {"description": "at most 1% or 2 cm, whichever is larger",
                    "relative": 0.01, "absolute_m": 0.02, "mode": "larger"},
    "room_width": {"description": "at most 1% or 2 cm, whichever is larger",
                   "relative": 0.01, "absolute_m": 0.02, "mode": "larger"},
    "room_height": {"description": "at most 1.5 cm",
                    "relative": None, "absolute_m": 0.015, "mode": "absolute"},
    "floor_area": {"description": "at most 2% per room",
                   "relative": 0.02, "absolute_m": None, "mode": "relative"},
    "opening_width": {"description": "at most 2 cm",
                      "relative": None, "absolute_m": 0.02, "mode": "absolute"},
    "opening_height": {"description": "at most 2 cm",
                       "relative": None, "absolute_m": 0.02, "mode": "absolute"},
}


@dataclass
class Comparison:
    measurement_id: str
    type: str
    reference_m: float | None
    model_m: float | None
    signed_error_m: float | None = None
    absolute_error_cm: float | None = None
    percent_error: float | None = None
    gate: str | None = None
    result: str = "not_comparable"
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "measurementId": self.measurement_id,
            "type": self.type,
            "reference_m": self.reference_m,
            "model_m": self.model_m,
            "signedError_m": self.signed_error_m,
            "absoluteError_cm": self.absolute_error_cm,
            "percentError": self.percent_error,
            "assignmentGate": self.gate,
            "result": self.result,
            "note": self.note,
            **({"extra": self.extra} if self.extra else {}),
        }


def _tolerance(kind: str, reference: float) -> float | None:
    gate = ASSIGNMENT_GATES.get(kind)
    if gate is None:
        return None
    relative = gate["relative"] * reference if gate["relative"] is not None else None
    absolute = gate["absolute_m"]
    if gate["mode"] == "larger":
        return max(relative or 0.0, absolute or 0.0)
    if gate["mode"] == "relative":
        return relative
    return absolute


def compare(
    measurement_id: str,
    kind: str,
    reference_m: float | None,
    model_m: float | None,
    note: str = "",
    extra: dict | None = None,
) -> Comparison:
    result = Comparison(measurement_id=measurement_id, type=kind,
                        reference_m=reference_m, model_m=model_m,
                        note=note, extra=extra or {})
    gate = ASSIGNMENT_GATES.get(kind)
    result.gate = gate["description"] if gate else None

    if reference_m is None or model_m is None:
        missing = "reference" if reference_m is None else "model"
        result.result = "not_comparable"
        result.note = (result.note or
                       f"no {missing} value is available for this measurement")
        return result

    signed = model_m - reference_m
    result.signed_error_m = round(signed, 6)
    result.absolute_error_cm = round(abs(signed) * 100.0, 4)
    result.percent_error = (round(abs(signed) / reference_m * 100.0, 4)
                            if reference_m else None)

    tolerance = _tolerance(kind, reference_m)
    if tolerance is None:
        result.result = "no_gate_defined"
    else:
        result.result = "pass" if abs(signed) <= tolerance else "fail"
        result.extra["toleranceApplied_m"] = round(tolerance, 6)
    return result


def summarise(comparisons: list[Comparison]) -> dict:
    comparable = [c for c in comparisons if c.absolute_error_cm is not None]
    gated = [c for c in comparable if c.result in {"pass", "fail"}]
    return {
        "comparisons": len(comparisons),
        "comparable": len(comparable),
        "notComparable": len(comparisons) - len(comparable),
        "gated": len(gated),
        "passed": sum(1 for c in gated if c.result == "pass"),
        "failed": sum(1 for c in gated if c.result == "fail"),
        "meanAbsoluteError_cm": (round(sum(c.absolute_error_cm for c in comparable)
                                       / len(comparable), 4) if comparable else None),
        "maxAbsoluteError_cm": (round(max(c.absolute_error_cm for c in comparable), 4)
                                if comparable else None),
    }
