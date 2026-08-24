"""Validate `spatial_model.json` against schema 0.1 and its own internal rules.

Schema conformance is necessary but not sufficient. A document can satisfy the
schema and still be incoherent — a measurement naming a surface that does not
exist, a floor area that disagrees with its own footprint, a confidence label
asserted where the evidence is absent. Those are checked here too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schema" / "spatial_model.schema.json"

AREA_TOLERANCE_M2 = 1e-4


def _shoelace(polygon: list[list[float]]) -> float:
    total = 0.0
    for index in range(len(polygon)):
        x1, z1 = polygon[index]
        x2, z2 = polygon[(index + 1) % len(polygon)]
        total += x1 * z2 - x2 * z1
    return abs(total) / 2.0


def validate_model(model: dict, schema_path: Path | None = None) -> list[str]:
    schema = json.loads(Path(schema_path or SCHEMA_PATH).read_text())
    validator = Draft202012Validator(schema)
    problems = [
        f"schema: {'/'.join(str(p) for p in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(model), key=lambda e: list(e.path))
    ]

    surface_ids = {s["id"] for s in model["surfaces"]}
    room_ids = {r["id"] for r in model["rooms"]}

    if len(surface_ids) != len(model["surfaces"]):
        problems.append("surfaces: ids are not unique")
    if len(room_ids) != len(model["rooms"]):
        problems.append("rooms: ids are not unique")

    for surface in model["surfaces"]:
        if surface["roomId"] not in room_ids:
            problems.append(
                f"surfaces/{surface['id']}: roomId '{surface['roomId']}' resolves to no room")
        for key, value in surface["dimensions"].items():
            if value is not None and (value < 0 or value != value):
                problems.append(
                    f"surfaces/{surface['id']}: dimension {key} is {value}")

    measurement_ids = set()
    for measurement in model["measurements"]:
        if measurement["id"] in measurement_ids:
            problems.append(f"measurements: duplicate id '{measurement['id']}'")
        measurement_ids.add(measurement["id"])
        entity = measurement.get("entityId")
        if entity is not None and entity not in surface_ids | room_ids:
            problems.append(
                f"measurements/{measurement['id']}: entityId '{entity}' resolves to "
                f"no surface or room")
        if measurement["producer"] != "geometry_pipeline":
            problems.append(
                f"measurements/{measurement['id']}: producer is "
                f"'{measurement['producer']}'; only geometry may produce metric values")
        value = measurement["value_m"]
        if value is not None and (value < 0 or value != value):
            problems.append(f"measurements/{measurement['id']}: value_m is {value}")
        if value is None and measurement["confidence"]["label"] != "unresolved":
            problems.append(
                f"measurements/{measurement['id']}: has no value but claims confidence "
                f"'{measurement['confidence']['label']}'")

    for opening in model["openings"]:
        if opening["surfaceId"] is not None and opening["surfaceId"] not in surface_ids:
            problems.append(
                f"openings/{opening['id']}: surfaceId '{opening['surfaceId']}' "
                f"resolves to no surface")
        dimensions = opening.get("dimensions")
        if dimensions and opening["observationState"] == "unresolved":
            problems.append(
                f"openings/{opening['id']}: unresolved openings must not carry "
                f"dimensions")

    for room in model["rooms"]:
        states = room.get("footprintEdgeStates")
        if states is not None and len(states) != len(room["footprint"]):
            problems.append(
                f"rooms/{room['id']}: {len(states)} edge states for "
                f"{len(room['footprint'])} footprint vertices")
        area_measurements = [m for m in model["measurements"]
                             if m["type"] == "floor_area" and m["entityId"] == room["id"]]
        for measurement in area_measurements:
            if measurement["value_m"] is None:
                continue
            expected = _shoelace(room["footprint"])
            if abs(expected - measurement["value_m"]) > AREA_TOLERANCE_M2:
                problems.append(
                    f"rooms/{room['id']}: reported floor area "
                    f"{measurement['value_m']:.6f} m2 disagrees with its own footprint "
                    f"({expected:.6f} m2)")

    for entity in list(model["surfaces"]) + list(model["rooms"]) + list(model["measurements"]):
        confidence = entity.get("confidence")
        if not confidence:
            continue
        if confidence["calibrated"] is not False:
            problems.append(f"{entity['id']}: confidence claims to be calibrated")
        if entity.get("observationState") in {"inferred", "unresolved"} \
                and confidence["label"] != "unresolved":
            problems.append(
                f"{entity['id']}: observation state '{entity['observationState']}' "
                f"cannot carry confidence '{confidence['label']}'")

    if model["damage"] or model["scope"]:
        problems.append(
            "damage[] and scope[] must remain empty; neither track is implemented")

    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args(argv)
    problems = validate_model(json.loads(args.model.read_text()))
    for problem in problems:
        print(f"[ERROR] {problem}")
    state = "SPATIAL_MODEL_VALID" if not problems else "SPATIAL_MODEL_INVALID"
    print(f"\n{state} ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
