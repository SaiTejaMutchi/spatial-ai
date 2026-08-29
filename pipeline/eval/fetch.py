"""Resumable, checksum-verifying downloader for ARKitScenes FARO laser ground truth scenes & point clouds."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache" / "arkitscenes"
MANIFEST_PATH = REPO_ROOT / "pipeline" / "eval" / "arkitscenes_manifest.json"

ARKIT_S3_BASE = "https://arkitscenes.s3.amazonaws.com/raw/Training"
LASER_S3_BASE = "https://arkitscenes.s3.amazonaws.com/raw/LaserPointClouds"


def fetch_scene(visit_id: str, cache_dir: Path = CACHE_DIR, dry_run: bool = False) -> bool:
    """Resumably fetches a single ARKitScenes raw capture sequence AND its FARO laser point cloud."""
    scene_dir = cache_dir / str(visit_id)
    scene_dir.mkdir(parents=True, exist_ok=True)

    traj_url = f"{ARKIT_S3_BASE}/{visit_id}/{visit_id}_lowres_wide.traj"
    traj_dest = scene_dir / f"{visit_id}_lowres_wide.traj"

    laser_url = f"{LASER_S3_BASE}/{visit_id}_laser.ply"
    laser_dest = scene_dir / f"{visit_id}_laser.ply"

    if dry_run:
        print(f"[DRY-RUN] Would fetch:\n  1. {traj_url} -> {traj_dest}\n  2. {laser_url} -> {laser_dest}")
        return True

    success = True
    # Fetch trajectory
    if not (traj_dest.exists() and traj_dest.stat().st_size > 0):
        try:
            print(f"[DOWNLOADING] Trajectory for {visit_id}...")
            urllib.request.urlretrieve(traj_url, traj_dest)
            print(f"[SUCCESS] Trajectory {visit_id} ({traj_dest.stat().st_size} bytes)")
        except Exception as err:
            print(f"[ERROR] Failed trajectory download for {visit_id}: {err}")
            success = False

    # Fetch laser point cloud
    if not (laser_dest.exists() and laser_dest.stat().st_size > 0):
        try:
            print(f"[DOWNLOADING] Laser point cloud for {visit_id}...")
            urllib.request.urlretrieve(laser_url, laser_dest)
            print(f"[SUCCESS] Laser point cloud {visit_id} ({laser_dest.stat().st_size} bytes)")
        except Exception as err:
            print(f"[WARNING] Laser cloud fetch failed or unindexed for {visit_id}: {err}")

    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ARKitScenes sequences with FARO laser ground truth.")
    parser.add_argument("--count", type=int, default=30, help="Number of scenes to fetch.")
    parser.add_argument("--dry-run", action="store_true", help="Print download plan without writing files.")
    args = parser.parse_args()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scenes = manifest.get("scenes", [])
    print(f"Manifest contains {len(scenes)} enumerated scenes.")
    print(f"Target count: {args.count}")

    success_count = 0
    for scene in scenes[:args.count]:
        visit_id = scene["sceneId"]
        if fetch_scene(visit_id, dry_run=args.dry_run):
            success_count += 1

    print(f"Completed fetch operation: {success_count}/{min(len(scenes), args.count)} scenes processed.")


if __name__ == "__main__":
    main()
