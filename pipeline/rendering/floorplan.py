"""Dimensioned 2D floor plan, rendered from `spatial_model.json` and nothing else.

The plan requires that the drawing and the JSON cannot disagree, so this module
takes the model document as its only input. It never touches the point cloud,
the plane fits, or the capture. If a number appears on the drawing, it was read
from a measurement record, and a test asserts exactly that.

Observation state is visible rather than implied: an observed wall is a solid
line, an inferred closure is dashed and labelled, and the legend says which is
which. A reviewer should not have to open the JSON to find out which parts of
the outline were actually seen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.sax.saxutils import escape

MARGIN_PX = 96
LABEL_OFFSET_PX = 16
SCALE_TARGET_PX = 900
# ROOM + observation legend + 1 m scale bar. A tall narrow room otherwise
# clips those columns (viewBox followed the footprint, not the footer).
FOOTER_MIN_WIDTH_PX = MARGIN_PX + 430 + 300 + 160

STATE_STYLE = {
    "directly_observed": {"stroke": "#1c3f5e", "width": 6.0, "dash": None,
                          "label": "Directly observed"},
    "partially_observed": {"stroke": "#3d7ea6", "width": 5.0, "dash": "18 8",
                           "label": "Partially observed"},
    "inferred": {"stroke": "#b8762e", "width": 4.0, "dash": "14 10",
                 "label": "Inferred closure"},
    "unresolved": {"stroke": "#9aa3ab", "width": 3.0, "dash": "4 10",
                   "label": "Unresolved"},
}

CLASSIFICATION_BADGE = {
    "public_development_fixture": ("DEVELOPMENT FIXTURE", "#b8762e"),
    "final_private_capture": ("FINAL CAPTURE", "#1c3f5e"),
    "baseline_fallback": ("BASELINE FALLBACK", "#9aa3ab"),
}


def _measurements(model: dict) -> dict:
    return {m["type"]: m for m in model["measurements"]}


def _format(measurement: dict | None) -> str:
    """Render exactly the stored value, or say it is unresolved."""
    if measurement is None or measurement["value_m"] is None:
        return "unresolved"
    unit = "m²" if measurement["unit"] == "m2" else "m"
    return f"{measurement['value_m']:.2f} {unit}"


def render_floorplan(model: dict) -> str:
    room = model["rooms"][0]
    footprint = [(float(x), float(z)) for x, z in room["footprint"]]
    states = room.get("footprintEdgeStates") or ["directly_observed"] * len(footprint)
    measurements = _measurements(model)

    xs = [p[0] for p in footprint]
    zs = [p[1] for p in footprint]
    span_x = max(xs) - min(xs)
    span_z = max(zs) - min(zs)
    scale = SCALE_TARGET_PX / max(span_x, span_z, 1e-6)

    width_px = max(span_x * scale + 2 * MARGIN_PX, FOOTER_MIN_WIDTH_PX)
    height_px = span_z * scale + 2 * MARGIN_PX + 150

    def project(point: tuple[float, float]) -> tuple[float, float]:
        # Viewed from above along -Y: +x to the right, +z up the page.
        return (MARGIN_PX + (point[0] - min(xs)) * scale,
                MARGIN_PX + (max(zs) - point[1]) * scale)

    wall_surfaces = [s for s in model["surfaces"] if s["type"] == "wall"]
    parts: list[str] = []

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px:.0f}" '
        f'height="{height_px:.0f}" viewBox="0 0 {width_px:.0f} {height_px:.0f}" '
        f'font-family="ui-sans-serif, -apple-system, Helvetica, Arial, sans-serif">')
    parts.append(f'<rect width="{width_px:.0f}" height="{height_px:.0f}" fill="#fbfaf8"/>')

    # Floor fill, so the enclosed area reads as a room rather than an outline.
    polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in (project(p) for p in footprint))
    parts.append(f'<polygon points="{polygon}" fill="#eef2f5" stroke="none"/>')

    # Walls, styled by what was actually observed.
    for index in range(len(footprint)):
        start = project(footprint[index])
        end = project(footprint[(index + 1) % len(footprint)])
        state = states[index] if index < len(states) else "inferred"
        style = STATE_STYLE.get(state, STATE_STYLE["unresolved"])
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        parts.append(
            f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" '
            f'y2="{end[1]:.2f}" stroke="{style["stroke"]}" '
            f'stroke-width="{style["width"]}" stroke-linecap="round"{dash}/>')

        surface = wall_surfaces[index] if index < len(wall_surfaces) else None
        if surface is None:
            continue
        mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_px = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        normal_x, normal_y = -dy / length_px, dx / length_px
        label_x = mid_x + normal_x * LABEL_OFFSET_PX
        label_y = mid_y + normal_y * LABEL_OFFSET_PX
        angle = 0.0 if abs(dx) < 1e-9 and abs(dy) < 1e-9 else \
            __import__("math").degrees(__import__("math").atan2(dy, dx))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        parts.append(
            f'<g transform="translate({label_x:.2f},{label_y:.2f}) rotate({angle:.2f})">'
            f'<text text-anchor="middle" font-size="17" fill="#22303c">'
            f'{escape(surface["id"])} · {surface["dimensions"]["width_m"]:.2f} m</text>'
            f'<text y="18" text-anchor="middle" font-size="13" fill="#6b7780">'
            f'{escape(surface["observationState"].replace("_", " "))}</text></g>')

    # Openings. None resolved is a fact worth stating on the drawing.
    resolved_openings = [o for o in model["openings"]
                         if o["observationState"] != "unresolved" and o.get("dimensions")]
    for opening in resolved_openings:
        parts.append(
            f'<!-- opening {escape(opening["id"])} on {escape(str(opening["surfaceId"]))} -->')

    base_y = MARGIN_PX + span_z * scale + 56
    parts.append(_dimension_block(model, measurements, MARGIN_PX, base_y,
                                  len(resolved_openings), len(model["openings"])))
    parts.append(_legend(MARGIN_PX + 430, base_y, states))
    parts.append(_scale_bar(MARGIN_PX + 430 + 300, base_y, scale))
    parts.append(_title(model, MARGIN_PX, 44))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _title(model: dict, x: float, y: float) -> str:
    classification = model["scan"]["classification"]
    badge_text, badge_colour = CLASSIFICATION_BADGE.get(
        classification, (classification.upper(), "#9aa3ab"))
    scan_id = escape(str(model["scan"]["id"]))
    return (
        f'<text x="{x}" y="{y}" font-size="24" font-weight="600" fill="#12212e">'
        f'Floor plan · {scan_id}</text>'
        f'<g transform="translate({x + 330},{y - 20})">'
        f'<rect width="250" height="26" rx="13" fill="{badge_colour}"/>'
        f'<text x="125" y="18" text-anchor="middle" font-size="13" font-weight="700" '
        f'fill="#ffffff" letter-spacing="1">{escape(badge_text)}</text></g>')


def _dimension_block(
    model: dict, measurements: dict, x: float, y: float,
    resolved_openings: int, total_openings: int,
) -> str:
    rows = [
        ("Length", _format(measurements.get("room_length"))),
        ("Width", _format(measurements.get("room_width"))),
        ("Height", _format(measurements.get("room_height"))),
        ("Floor area", _format(measurements.get("floor_area"))),
    ]
    if resolved_openings:
        rows.append(("Openings", f"{resolved_openings} resolved"))
    else:
        rows.append(("Openings", f"none resolved ({total_openings} unresolved)"))

    parts = [f'<text x="{x}" y="{y}" font-size="15" font-weight="700" '
             f'fill="#12212e" letter-spacing="1">ROOM</text>']
    for index, (label, value) in enumerate(rows):
        row_y = y + 26 + index * 22
        parts.append(
            f'<text x="{x}" y="{row_y}" font-size="15" fill="#6b7780">{label}</text>'
            f'<text x="{x + 150}" y="{row_y}" font-size="15" font-weight="600" '
            f'fill="#12212e">{escape(value)}</text>')
    return "".join(parts)


def _legend(x: float, y: float, states: list[str]) -> str:
    present = [s for s in STATE_STYLE if s in set(states)]
    parts = [f'<text x="{x}" y="{y}" font-size="15" font-weight="700" '
             f'fill="#12212e" letter-spacing="1">OBSERVATION STATE</text>']
    for index, state in enumerate(present):
        style = STATE_STYLE[state]
        row_y = y + 26 + index * 22
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        parts.append(
            f'<line x1="{x}" y1="{row_y - 5}" x2="{x + 44}" y2="{row_y - 5}" '
            f'stroke="{style["stroke"]}" stroke-width="{style["width"]}" '
            f'stroke-linecap="round"{dash}/>'
            f'<text x="{x + 58}" y="{row_y}" font-size="15" fill="#22303c">'
            f'{style["label"]}</text>')
    return "".join(parts)


def _scale_bar(x: float, y: float, scale: float) -> str:
    metres = 1.0
    length = metres * scale
    return (
        f'<text x="{x}" y="{y}" font-size="15" font-weight="700" fill="#12212e" '
        f'letter-spacing="1">SCALE</text>'
        f'<line x1="{x}" y1="{y + 26}" x2="{x + length:.2f}" y2="{y + 26}" '
        f'stroke="#12212e" stroke-width="3"/>'
        f'<line x1="{x}" y1="{y + 20}" x2="{x}" y2="{y + 32}" stroke="#12212e" '
        f'stroke-width="3"/>'
        f'<line x1="{x + length:.2f}" y1="{y + 20}" x2="{x + length:.2f}" '
        f'y2="{y + 32}" stroke="#12212e" stroke-width="3"/>'
        f'<text x="{x + length / 2:.2f}" y="{y + 50}" text-anchor="middle" '
        f'font-size="14" fill="#6b7780">{metres:.0f} m</text>')


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    import json
    svg = render_floorplan(json.loads(args.model.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(f"floorplan -> {args.output} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
