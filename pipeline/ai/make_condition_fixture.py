"""Write a labelled development condition fixture.

Public ARKitScenes rooms typically show no insurer-relevant visible condition.
This fixture overlays a high-contrast stain on a copy of a real registered
evidence frame so a model can propose a region. The overlay is synthetic and
labelled; the camera pose, intrinsics, and surface plane used later are real.

The region consumed by registration must still come from the live model.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "samples" / "ai_condition_eval"
LABEL = "DEVELOPMENT CONDITION FIXTURE"


def _find_evidence_png() -> Path | None:
    matches = sorted((REPO_ROOT / "outputs").glob("dev_*/evidence/evidence-001.png"))
    return matches[0] if matches else None


def _stain(image: Image.Image) -> Image.Image:
    """Paint a large dark stain that remains obvious after 128px JPEG encode."""
    image = image.convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    x0, y0 = int(width * 0.22), int(height * 0.28)
    x1, y1 = int(width * 0.78), int(height * 0.78)
    draw.ellipse((x0, y0, x1, y1), fill=(70, 32, 12, 220))
    # Drips keep the mark readable after heavy downscale.
    drip_w = max(8, (x1 - x0) // 8)
    for i, frac in enumerate((0.35, 0.50, 0.65)):
        cx = int(x0 + (x1 - x0) * frac)
        draw.ellipse(
            (cx - drip_w, y1 - 8, cx + drip_w, min(height - 4, y1 + height // 6 + i * 4)),
            fill=(55, 24, 8, 200),
        )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.2))
    stained = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return stained


def _cartoon_stain(size=(128, 96)) -> Image.Image:
    """Ceiling-originating yellow-brown water stain on a wall. Not furniture."""
    image = Image.new("RGB", size, (214, 206, 192))
    draw = ImageDraw.Draw(image)
    floor_y = int(size[1] * 0.90)
    draw.rectangle((0, floor_y, size[0], size[1]), fill=(122, 112, 100))
    # Spreads down from the top edge, the way a water stain does.
    outer = [
        (18, 0), (110, 0), (102, 28), (88, 48), (70, 58),
        (52, 50), (36, 32), (24, 14),
    ]
    draw.polygon(outer, fill=(176, 142, 72))
    inner = [
        (40, 0), (86, 0), (80, 22), (64, 36), (48, 24),
    ]
    draw.polygon(inner, fill=(148, 108, 48))
    for x, bot, width in ((42, 70, 3), (64, 76, 4), (84, 64, 3)):
        draw.rectangle((x, 48, x + width, bot), fill=(160, 122, 58))
    return image


def write_fixture(root: Path | None = None, evidence_png: Path | None = None) -> Path:
    root = Path(root or OUT)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(evidence_png) if evidence_png else _find_evidence_png()
    if source and source.is_file():
        overlaid = _stain(Image.open(source).convert("RGB"))
        overlaid.save(root / "stained_evidence_overlay.png")
        source_note = (
            "cartoon 128px development fixture plus an overlay copy of a real "
            "registered evidence frame. The live model is sent the cartoon, which "
            "survives Groq's 128px JPEG TPM encode; registration still uses the "
            "real evidence pose and surface plane."
        )
    else:
        source_note = "cartoon 128px development fixture; no real evidence PNG was available"

    stained = _cartoon_stain()
    image_path = root / "stained_evidence.png"
    stained.save(image_path)

    record = {
        "label": LABEL,
        "isRealDamageEvidence": False,
        "evaluationOnly": True,
        "purpose": (
            "Development Visible Condition Grounding. The stain is a labelled "
            "synthetic fixture. A live model must still propose the region; "
            "geometry registers that region onto a real surface using the real "
            "evidence pose."
        ),
        "sourceNote": source_note,
        "image": "stained_evidence.png",
        "intendedConditionClass": "staining",
    }
    (root / "labels.json").write_text(json.dumps(record, indent=2) + "\n")
    (root / "README.md").write_text(
        "# Development condition fixture\n\n"
        "This is a **DEVELOPMENT CONDITION FIXTURE**. It is not real damage "
        "evidence. A synthetic stain is overlaid so the pipeline can demonstrate "
        "`surfaceId -> AI region -> registration -> geometry-owned quantity` "
        "when public ARKitScenes frames contain no supported condition.\n"
    )
    return root


if __name__ == "__main__":
    path = write_fixture()
    print(f"wrote {path / 'stained_evidence.png'}")
