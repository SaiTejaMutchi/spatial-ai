"""Independent reference measurements, read through one path.

The same parser handles the labelled development fixtures used now and the real
tape records supplied after `DEV_COMPLETE`, so the ground-truth machinery has
already been exercised by the time real values exist.

Two properties matter more than convenience here. Reference values are
read-only: nothing in this module can write back toward geometry. And a row
with no readings stays unresolved — a missing measurement is reported as
missing, never filled in from the model it is supposed to be checking.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

DEVELOPMENT_MARKER = "DEVELOPMENT FIXTURE"
REQUIRED_COLUMNS = ("measurement_id", "type")
READING_COLUMNS = ("tape_1_m", "tape_2_m", "tape_3_m")


class GroundTruthError(Exception):
    """The reference file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class ReferenceMeasurement:
    measurement_id: str
    type: str
    readings: tuple[float, ...]
    instrument: str
    operator: str
    timestamp: str
    notes: str
    is_development_fixture: bool

    @property
    def value_m(self) -> float | None:
        """Median of the repeated readings, or None when there are none."""
        return float(median(self.readings)) if self.readings else None

    @property
    def spread_m(self) -> float | None:
        if len(self.readings) < 2:
            return None
        return float(max(self.readings) - min(self.readings))


@dataclass(frozen=True)
class GroundTruthSet:
    measurements: tuple[ReferenceMeasurement, ...]
    source_path: str
    contains_development_fixtures: bool

    def by_type(self, kind: str) -> ReferenceMeasurement | None:
        for measurement in self.measurements:
            if measurement.type == kind:
                return measurement
        return None


def load_ground_truth(path: Path) -> GroundTruthSet:
    path = Path(path)
    if not path.is_file():
        raise GroundTruthError(f"ground-truth file '{path}' does not exist")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise GroundTruthError(
                f"'{path.name}' is missing required column(s) {missing}; found {columns}")
        rows = list(reader)

    if not rows:
        raise GroundTruthError(f"'{path.name}' contains a header but no measurements")

    measurements: list[ReferenceMeasurement] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        identifier = (row.get("measurement_id") or "").strip()
        if not identifier:
            raise GroundTruthError(f"'{path.name}' line {index} has no measurement_id")
        if identifier in seen:
            raise GroundTruthError(
                f"'{path.name}' line {index} repeats measurement_id '{identifier}'")
        seen.add(identifier)

        readings: list[float] = []
        for column in READING_COLUMNS:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError as exc:
                raise GroundTruthError(
                    f"'{path.name}' line {index} column {column} is not a number: "
                    f"{raw!r}") from exc
            if value <= 0:
                raise GroundTruthError(
                    f"'{path.name}' line {index} column {column} is {value}; a "
                    f"physical measurement must be positive")
            readings.append(value)

        instrument = (row.get("instrument") or "").strip()
        measurements.append(ReferenceMeasurement(
            measurement_id=identifier,
            type=(row.get("type") or "").strip(),
            readings=tuple(readings),
            instrument=instrument,
            operator=(row.get("operator") or "").strip(),
            timestamp=(row.get("timestamp") or "").strip(),
            notes=(row.get("notes") or "").strip(),
            is_development_fixture=(DEVELOPMENT_MARKER in instrument
                                    or identifier.startswith("FIXTURE-")),
        ))

    return GroundTruthSet(
        measurements=tuple(measurements),
        source_path=str(path),
        contains_development_fixtures=any(m.is_development_fixture
                                          for m in measurements),
    )
