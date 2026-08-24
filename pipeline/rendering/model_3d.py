"""Semantic 3D model, built from `spatial_model.json` and nothing else.

Format is OBJ with a companion MTL and an entity map. That choice is the plan's:
GLB conversion is an explicit hard scope cut, and Hour 4:30 asks for a canonical
surface model or a labelled raw evidence model plus surface-ID mapping. OBJ
opens in Preview, Quick Look, Blender, and MeshLab without a toolchain, which is
what "opens reliably during the defense" actually requires.

Each surface becomes its own named OBJ group carrying the same stable ID the
JSON and the floor plan use, so selecting `wall-003` means the same thing in all
three. Materials encode observation state, so an inferred closure cannot be
mistaken for an observed wall in the viewer any more than it can on the drawing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

STATE_MATERIAL = {
    "directly_observed": ("observed", (0.42, 0.52, 0.60)),
    "partially_observed": ("partially_observed", (0.36, 0.60, 0.72)),
    "inferred": ("inferred", (0.78, 0.52, 0.22)),
    "unresolved": ("unresolved", (0.62, 0.64, 0.67)),
}


@dataclass
class Model3D:
    obj: str
    mtl: str
    entity_map: dict


def _material_library() -> str:
    lines = ["# Materials encode observation state, not appearance.",
             "# An inferred closure must not look like an observed wall.", ""]
    for _, (name, (r, g, b)) in STATE_MATERIAL.items():
        lines.extend([
            f"newmtl {name}",
            f"Kd {r:.4f} {g:.4f} {b:.4f}",
            f"Ka {r * 0.3:.4f} {g * 0.3:.4f} {b * 0.3:.4f}",
            "Ks 0.0500 0.0500 0.0500",
            "Ns 12.0",
            "d 1.0",
            "",
        ])
    return "\n".join(lines)


def build_model_3d(model: dict, mtl_filename: str = "room_model.mtl") -> Model3D:
    room = model["rooms"][0]
    footprint = [(float(x), float(z)) for x, z in room["footprint"]]
    states = room.get("footprintEdgeStates") or ["directly_observed"] * len(footprint)

    measurements = {m["type"]: m for m in model["measurements"]}
    height_measurement = measurements.get("room_height")
    height = height_measurement["value_m"] if height_measurement and \
        height_measurement["value_m"] is not None else None

    surfaces = {s["id"]: s for s in model["surfaces"]}
    wall_surfaces = [s for s in model["surfaces"] if s["type"] == "wall"]

    floor_surface = surfaces.get("floor-001")
    ceiling_surface = surfaces.get("ceiling-001")

    # The canonical frame puts the floor at the fitted floor plane; the model is
    # emitted with the floor at y = 0 so the artifact stands on the viewer's
    # ground plane, and the applied offset is recorded in the entity map.
    floor_y = 0.0
    ceiling_y = height if height is not None else 0.0

    vertices: list[tuple[float, float, float]] = []
    lines: list[str] = [
        "# Canonical semantic room model",
        f"# Generated from spatial_model.json (modelId {model['modelId']})",
        f"# Classification: {model['scan']['classification']}",
        "# Units: metres. Right-handed, +Y up, X/Z plan axes.",
        "# Group names are the same stable surface IDs used by the JSON and the plan.",
        f"mtllib {mtl_filename}",
        "",
    ]
    entity_map: dict = {
        "modelId": model["modelId"],
        "scanId": model["scan"]["id"],
        "classification": model["scan"]["classification"],
        "units": "meters",
        "upAxis": "y",
        "floorOffsetApplied_m": 0.0,
        "note": ("Every OBJ group name is a surface ID from spatial_model.json. "
                 "Materials encode observation state."),
        "surfaces": {},
    }

    def add_vertex(x: float, y: float, z: float) -> int:
        vertices.append((x, y, z))
        return len(vertices)

    def emit(surface: dict, face_indices: list[int]) -> None:
        state = surface["observationState"]
        material = STATE_MATERIAL.get(state, STATE_MATERIAL["unresolved"])[0]
        lines.append(f"g {surface['id']}")
        lines.append(f"usemtl {material}")
        lines.append("f " + " ".join(str(i) for i in face_indices))
        lines.append("")
        entity_map["surfaces"][surface["id"]] = {
            "objGroup": surface["id"],
            "type": surface["type"],
            "observationState": state,
            "material": material,
            "confidenceLabel": surface["confidence"]["label"],
            "dimensions": surface["dimensions"],
        }

    # Walls, one quad per footprint edge, in the same order as the plan.
    if height is not None:
        for index in range(len(footprint)):
            if index >= len(wall_surfaces):
                break
            start = footprint[index]
            end = footprint[(index + 1) % len(footprint)]
            surface = wall_surfaces[index]
            a = add_vertex(start[0], floor_y, start[1])
            b = add_vertex(end[0], floor_y, end[1])
            c = add_vertex(end[0], ceiling_y, end[1])
            d = add_vertex(start[0], ceiling_y, start[1])
            emit(surface, [a, b, c, d])

    if floor_surface is not None:
        indices = [add_vertex(x, floor_y, z) for x, z in footprint]
        emit(floor_surface, indices)

    if ceiling_surface is not None and height is not None:
        indices = [add_vertex(x, ceiling_y, z) for x, z in reversed(footprint)]
        emit(ceiling_surface, indices)

    header_end = lines.index("") + 1
    vertex_lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    obj = "\n".join(lines[:header_end] + vertex_lines + [""] + lines[header_end:])

    entity_map["vertexCount"] = len(vertices)
    entity_map["surfaceCount"] = len(entity_map["surfaces"])
    if height is None:
        entity_map["limitation"] = (
            "Room height is unresolved, so no wall or ceiling geometry could be "
            "extruded. The artifact contains the floor polygon only.")
    return Model3D(obj=obj, mtl=_material_library(), entity_map=entity_map)


def write_model_3d(model: dict, output_dir: Path, stem: str = "room_model") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    built = build_model_3d(model, mtl_filename=f"{stem}.mtl")
    (output_dir / f"{stem}.obj").write_text(built.obj)
    (output_dir / f"{stem}.mtl").write_text(built.mtl)
    (output_dir / f"{stem}_entity_map.json").write_text(
        json.dumps(built.entity_map, indent=2) + "\n")
    return built.entity_map


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    entity_map = write_model_3d(json.loads(args.model.read_text()), args.output_dir)
    print(f"room_model.obj -> {args.output_dir} "
          f"({entity_map['surfaceCount']} surfaces, {entity_map['vertexCount']} vertices)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
