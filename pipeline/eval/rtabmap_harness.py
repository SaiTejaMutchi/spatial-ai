"""RTAB-Map Baseline Evaluation Harness.

Runs RTAB-Map RGB-D SLAM against ARKitScenes raw captures, exports the 3D point cloud,
fits floor and ceiling planes using the repo's SVD plane-fitting algorithm (extract_ground_truth_from_ply),
and computes height error metrics against FARO laser ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.eval.extract_gt import extract_ground_truth_from_ply

MANIFEST_PATH = REPO_ROOT / "pipeline" / "eval" / "arkitscenes_manifest.json"
RESULTS_DIR = REPO_ROOT / "pipeline" / "eval" / "results"


def run_rtabmap_on_scene(
    scene_id: str,
    visit_id: str,
    laser_gt_m: float,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    bin_dir: str = "/usr/local/bin",
) -> dict:
    """Runs RTAB-Map on a single ARKitScenes capture, exports PLY point cloud, and measures height error."""
    start_time = time.time()
    
    # Locate executables
    dataset_bin = os.path.join(bin_dir, "rtabmap-rgbd_dataset")
    if not os.path.exists(dataset_bin):
        dataset_bin = "rtabmap-rgbd_dataset"
        
    export_bin = os.path.join(bin_dir, "rtabmap-export")
    if not os.path.exists(export_bin):
        export_bin = "rtabmap-export"

    # Set output paths
    out_base = output_dir or (REPO_ROOT / "output" / f"rtabmap_{scene_id}")
    out_base.mkdir(parents=True, exist_ok=True)
    db_path = out_base / "rtabmap.db"
    ply_path = out_base / "cloud.ply"

    # Search raw capture directory if not specified
    if not raw_dir or not raw_dir.is_dir():
        candidates = [
            REPO_ROOT / "samples" / "arkitscenes" / "raw" / "Training" / scene_id,
            REPO_ROOT / "samples" / "arkitscenes" / "raw" / "Validation" / scene_id,
            REPO_ROOT / "outputs" / f"dev_{scene_id}",
            REPO_ROOT / ".batch_work" / scene_id,
        ]
        raw_dir = next((p for p in candidates if p.is_dir()), None)

    if not raw_dir:
        return {
            "scene_id": scene_id,
            "visit_id": visit_id,
            "laser_gt_m": laser_gt_m,
            "status": "missing_input_data",
            "note": f"Raw capture directory for scene {scene_id} not found on disk.",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # Setup sequence folder for rtabmap-rgbd_dataset
    seq_dir = out_base / "sequence"
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)

    rgb_src = raw_dir / "vga_wide"
    depth_src = raw_dir / "vga_wide_depth"

    if not (rgb_src.is_dir() and depth_src.is_dir()):
        return {
            "scene_id": scene_id,
            "visit_id": visit_id,
            "laser_gt_m": laser_gt_m,
            "status": "invalid_sequence_structure",
            "note": f"Missing vga_wide or vga_wide_depth subdirectories in {raw_dir}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    rgb_sync = seq_dir / "rgb_sync"
    depth_sync = seq_dir / "depth_sync"
    try:
        os.symlink(rgb_src, rgb_sync)
        os.symlink(depth_src, depth_sync)
    except Exception:
        shutil.copytree(rgb_src, rgb_sync)
        shutil.copytree(depth_src, depth_sync)

    # 1. Execute rtabmap-rgbd_dataset
    cmd_dataset = [
        dataset_bin,
        "--output", str(out_base),
        "--output_name", "rtabmap",
        "--quiet",
        str(seq_dir),
    ]

    proc_ds = subprocess.run(cmd_dataset, capture_output=True, text=True)
    
    db_size = db_path.stat().st_size if db_path.exists() else 0
    if proc_ds.returncode != 0 or db_size == 0:
        return {
            "scene_id": scene_id,
            "visit_id": visit_id,
            "laser_gt_m": laser_gt_m,
            "status": "rtabmap_execution_failed",
            "exit_code": proc_ds.returncode,
            "db_size_bytes": db_size,
            "stderr": proc_ds.stderr[-500:] if proc_ds.stderr else "",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # 2. Export 3D point cloud PLY
    cmd_export = [export_bin, "--ply", "--output", str(ply_path), str(db_path)]
    proc_exp = subprocess.run(cmd_export, capture_output=True, text=True)

    ply_size = ply_path.stat().st_size if ply_path.exists() else 0
    if proc_exp.returncode != 0 or ply_size == 0:
        return {
            "scene_id": scene_id,
            "visit_id": visit_id,
            "laser_gt_m": laser_gt_m,
            "status": "export_ply_failed",
            "exit_code": proc_exp.returncode,
            "ply_size_bytes": ply_size,
            "stderr": proc_exp.stderr[-500:] if proc_exp.stderr else "",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # 3. Fit horizontal planes using SVD algorithm
    try:
        gt_info = extract_ground_truth_from_ply(ply_path, strict_completeness=False)
        pred_height_m = gt_info["extractedHeight_m"]
        abs_error_cm = round(abs(pred_height_m - laser_gt_m) * 100.0, 2)
        signed_error_cm = round((pred_height_m - laser_gt_m) * 100.0, 2)
        
        elapsed_sec = round(time.time() - start_time, 2)
        
        return {
            "scene_id": scene_id,
            "visit_id": visit_id,
            "laser_gt_m": laser_gt_m,
            "pred_height_m": pred_height_m,
            "abs_error_cm": abs_error_cm,
            "signed_error_cm": signed_error_cm,
            "status": "success",
            "point_count": gt_info["pointCountLoaded"],
            "floor_plane_z": gt_info["floorPlaneZ"],
            "ceiling_plane_z": gt_info["ceilingPlaneZ"],
            "execution_time_sec": elapsed_sec,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    except Exception as exc:
        return {
            "scene_id": scene_id,
            "visit_id": visit_id,
            "laser_gt_m": laser_gt_m,
            "status": "plane_fitting_failed",
            "reason": str(exc),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="RTAB-Map Baseline Evaluation Harness.")
    parser.add_argument("--scene", help="Single scene ID to process.")
    parser.add_argument("--all", action="store_true", help="Process all enumerated manifest scenes.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Results output directory.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scenes = manifest.get("scenes", [])
    if args.scene:
        scenes = [s for s in scenes if s["sceneId"] == args.scene]

    results = []
    print(f"Starting RTAB-Map Baseline Evaluation across {len(scenes)} scene(s)...")

    for idx, sc in enumerate(scenes, start=1):
        scene_id = sc["sceneId"]
        visit_id = sc["visitId"]
        laser_gt = sc["referenceHeight_m"]

        print(f"[{idx}/{len(scenes)}] Evaluating Scene {scene_id} (Visit {visit_id}, GT={laser_gt:.4f}m)...")
        res = run_rtabmap_on_scene(scene_id, visit_id, laser_gt)
        results.append(res)

        res_path = args.output_dir / f"rtabmap_scene_{scene_id}.json"
        res_path.write_text(json.dumps(res, indent=2) + "\n")
        print(f"   Status: {res['status']}, Pred Height: {res.get('pred_height_m', 'N/A')}, Error: {res.get('abs_error_cm', 'N/A')} cm")

    summary_path = args.output_dir / "rtabmap_results_summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"Saved complete RTAB-Map evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
