"""Seeded random sampler for candidate ARKitScenes scenes with FARO laser ground truth."""

from __future__ import annotations

import json
import random
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "samples" / "arkitscenes" / "laser_scanner_point_clouds_mapping.csv"
MANIFEST_PATH = REPO_ROOT / "pipeline" / "eval" / "arkitscenes_manifest.json"


def select_seeded_scenes(sample_size: int = 30, seed: int = 42) -> list[str]:
    """Selects a reproducible seeded random sample from the 1,004 unique laser-mapped visits."""
    df = pd.read_csv(CSV_PATH)
    unique_visits = sorted(list(df["visit_id"].unique()))

    rng = random.Random(seed)
    sampled_visits = rng.sample(unique_visits, sample_size)
    return [str(v) for v in sampled_visits]


if __name__ == "__main__":
    sampled = select_seeded_scenes(30, seed=42)
    print("=== SEEDED RANDOM SAMPLE SELECTION (seed=42) ===")
    print(f"Sampled {len(sampled)} visit IDs from 1,004 candidates:")
    print(sampled[:10], "... (and 20 more)")
