"""Write the tiny synthetic-render semantic evaluation fixture.

Official Structured3D requires a human-signed terms agreement and is not
downloaded here. This local fixture follows the same evaluation protocol and
is labelled SYNTHETIC_RENDER_SEMANTIC_EVALUATION.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "samples" / "ai_semantic_eval"

# Pixel boxes are the nominated regions. Semantic labels live only in labels.json.
CASES = [
    {
        "id": "case-door-001",
        "candidateId": "candidate-001",
        "surfaceId": "wall-001",
        "kind_draw": "door",
        "box": (90, 40, 150, 180),
        "width_m": 0.90, "height_m": 2.10, "sillHeight_m": 0.0,
        "label": "door",
    },
    {
        "id": "case-window-001",
        "candidateId": "candidate-002",
        "surfaceId": "wall-002",
        "kind_draw": "window",
        "box": (70, 50, 170, 120),
        "width_m": 1.20, "height_m": 1.40, "sillHeight_m": 0.90,
        "label": "window",
    },
    {
        "id": "case-occlusion-001",
        "candidateId": "candidate-003",
        "surfaceId": "wall-003",
        "kind_draw": "occlusion",
        "box": (80, 70, 180, 170),
        "width_m": 1.10, "height_m": 1.20, "sillHeight_m": 0.40,
        "label": "occlusion",
    },
    {
        "id": "case-empty-001",
        "candidateId": "candidate-004",
        "surfaceId": "wall-004",
        "kind_draw": "empty",
        "box": (100, 50, 160, 150),
        "width_m": 0.80, "height_m": 1.50, "sillHeight_m": 0.50,
        "label": "scan_gap",
    },
]


def _wall(size=(256, 192), floor_y=180) -> Image.Image:
    image = Image.new("RGB", size, (210, 205, 198))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, floor_y, size[0], size[1]), fill=(120, 110, 100))
    return image


def _draw_feature(image: Image.Image, kind: str, box: tuple[int, int, int, int],
                  offset: int = 0) -> Image.Image:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    x0, x1 = x0 + offset, x1 + offset
    if kind == "door":
        draw.rectangle((x0, y0, x1, y1), fill=(92, 58, 32), outline=(40, 24, 12), width=2)
        draw.ellipse((x1 - 14, (y0 + y1) // 2 - 4, x1 - 6, (y0 + y1) // 2 + 4), fill=(200, 180, 80))
    elif kind == "window":
        draw.rectangle((x0, y0, x1, y1), fill=(160, 200, 220), outline=(80, 90, 100), width=3)
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        draw.line((mx, y0, mx, y1), fill=(80, 90, 100), width=2)
        draw.line((x0, my, x1, my), fill=(80, 90, 100), width=2)
    elif kind == "occlusion":
        draw.rectangle((x0, y0, x1, y1), fill=(205, 205, 200), outline=(180, 180, 175), width=1)
        draw.polygon([(x0 + 10, y1), (x0 + 40, y0 + 20), (x1 - 10, y1)], fill=(70, 70, 75))
        draw.rectangle((x0 + 20, y0 + 40, x1 - 20, y1), fill=(45, 40, 38))
    else:
        draw.rectangle((x0, y0, x1, y1), outline=(180, 170, 160), width=1)
    return image


def _crop(image: Image.Image, box: tuple[int, int, int, int], pad: int = 8) -> Image.Image:
    x0, y0, x1, y1 = box
    return image.crop((max(0, x0 - pad), max(0, y0 - pad),
                       min(image.width, x1 + pad), min(image.height, y1 + pad)))


def write_fixture(root: Path | None = None) -> Path:
    root = Path(root or OUT)
    root.mkdir(parents=True, exist_ok=True)
    index = []
    for case in CASES:
        folder = root / case["id"]
        folder.mkdir(parents=True, exist_ok=True)
        context = _draw_feature(_wall(), case["kind_draw"], case["box"])
        confirm = _draw_feature(_wall(), case["kind_draw"], case["box"], offset=6)
        crop = _crop(context, case["box"])
        context.save(folder / "context.png")
        confirm.save(folder / "confirm.png")
        crop.save(folder / "crop.png")
        w, h = context.size
        x0, y0, x1, y1 = case["box"]
        candidate = {
            "candidateId": case["candidateId"],
            "surfaceId": case["surfaceId"],
            "geometry": {
                "width_m": case["width_m"],
                "height_m": case["height_m"],
                "sillHeight_m": case["sillHeight_m"],
                "producer": "geometry_pipeline",
            },
            "imageRegion": {
                "x0": round(x0 / w, 4), "y0": round(y0 / h, 4),
                "x1": round(x1 / w, 4), "y1": round(y1 / h, 4),
            },
            "images": {
                "context": "context.png",
                "crop": "crop.png",
                "confirm": "confirm.png",
            },
        }
        labels = {
            "semanticClass": case["label"],
            "labelSource": "fixture_author_held_out",
            "evaluationOnly": True,
            "datasetLabel": "SYNTHETIC_RENDER_SEMANTIC_EVALUATION",
        }
        (folder / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
        (folder / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
        index.append({"id": case["id"], "candidateId": case["candidateId"],
                      "surfaceId": case["surfaceId"]})
    manifest = {
        "label": "SYNTHETIC_RENDER_SEMANTIC_EVALUATION",
        "notOfficialStructured3D": True,
        "reason": (
            "Official Structured3D download requires a human-signed terms "
            "agreement. This fixture is a tiny locally authored synthetic render "
            "used only to exercise the semantic evaluation protocol."),
        "cases": index,
        "privilegedGroundTruth": "samples/ai_semantic_eval/*/labels.json is evaluator-only",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


if __name__ == "__main__":
    path = write_fixture()
    print(f"wrote {path}")
