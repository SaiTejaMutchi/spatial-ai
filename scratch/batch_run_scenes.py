import os
import sys
import json
import csv
import urllib.request
import zipfile
import shutil
import subprocess
import time
import math
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.eval.extract_gt import extract_ground_truth_from_ply
from pipeline.eval.evaluator import compute_statistics, compute_bias_regression
from pipeline.eval.baselines import run_baseline_comparison

# Initial clean N=10 benchmark baseline results
INITIAL_10_SCENES = [
    {"scene_num": 1, "scene_id": "47333462", "visit_id": "467138", "fold": "Training", "laser_gt_m": 2.6300, "pred_height_m": 2.6105, "abs_error_cm": 1.95, "signed_error_cm": -1.95, "gate_passed": False},
    {"scene_num": 2, "scene_id": "41418135", "visit_id": "416418", "fold": "Training", "laser_gt_m": 2.4302, "pred_height_m": 2.4155, "abs_error_cm": 1.47, "signed_error_cm": -1.47, "gate_passed": True},
    {"scene_num": 3, "scene_id": "41418155", "visit_id": "416407", "fold": "Training", "laser_gt_m": 2.4303, "pred_height_m": 2.4042, "abs_error_cm": 2.61, "signed_error_cm": -2.61, "gate_passed": False},
    {"scene_num": 4, "scene_id": "41418140", "visit_id": "416411", "fold": "Training", "laser_gt_m": 2.4428, "pred_height_m": 2.4122, "abs_error_cm": 3.06, "signed_error_cm": -3.06, "gate_passed": False},
    {"scene_num": 5, "scene_id": "42444474", "visit_id": "421069", "fold": "Training", "laser_gt_m": 2.3249, "pred_height_m": 2.3102, "abs_error_cm": 1.47, "signed_error_cm": -1.47, "gate_passed": True},
    {"scene_num": 6, "scene_id": "42444499", "visit_id": "421065", "fold": "Training", "laser_gt_m": 2.2944, "pred_height_m": 2.2853, "abs_error_cm": 0.91, "signed_error_cm": -0.91, "gate_passed": True},
    {"scene_num": 7, "scene_id": "42444511", "visit_id": "421063", "fold": "Training", "laser_gt_m": 2.3073, "pred_height_m": 2.3296, "abs_error_cm": 2.23, "signed_error_cm": 2.23, "gate_passed": False},
    {"scene_num": 8, "scene_id": "42444514", "visit_id": "421061", "fold": "Training", "laser_gt_m": 2.1311, "pred_height_m": 2.1246, "abs_error_cm": 0.66, "signed_error_cm": -0.66, "gate_passed": True},
    {"scene_num": 9, "scene_id": "4244519", "visit_id": "421060", "fold": "Training", "laser_gt_m": 2.2972, "pred_height_m": 2.3541, "abs_error_cm": 5.69, "signed_error_cm": 5.69, "gate_passed": False},
    {"scene_num": 10, "scene_id": "42444574", "visit_id": "421062", "fold": "Training", "laser_gt_m": 2.4578, "pred_height_m": 2.3726, "abs_error_cm": 8.52, "signed_error_cm": -8.52, "gate_passed": False},
]

def update_stage_file(scene_id: str, stage: str):
    try:
        Path("/tmp/current_batch_scene").write_text(scene_id)
        Path("/tmp/current_batch_stage").write_text(stage)
        Path("/tmp/current_batch_pid").write_text(str(os.getpid()))
    except Exception:
        pass

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg, flush=True)
    try:
        with open(Path.home() / "batch_nanny.log", "a") as f:
            f.write(full_msg + "\n")
    except Exception:
        pass

def log_progress(record: dict):
    progress_file = REPO_ROOT / "batch_progress.jsonl"
    with open(progress_file, "a") as f:
        f.write(json.dumps(record) + "\n")

def get_cdn_content_length(url: str) -> int:
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get('Content-Length'))

def get_candidate_scenes(initial_visit_ids: set):
    visit_to_scans = {}
    mapping_csv = REPO_ROOT / "samples" / "arkitscenes" / "laser_scanner_point_clouds_mapping.csv"
    if mapping_csv.is_file():
        with open(mapping_csv) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    scan_id, visit_id = row[0], row[1]
                    if visit_id not in visit_to_scans:
                        visit_to_scans[visit_id] = []
                    visit_to_scans[visit_id].append(scan_id)

    raw_csv = REPO_ROOT / "samples" / "arkitscenes" / "raw_metadata.csv"
    candidates = []
    seen_visits = set(initial_visit_ids)
    if raw_csv.is_file():
        with open(raw_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('has_laser_scanner_point_clouds') == 'True':
                    vid = row['video_id']
                    v_id = row['visit_id']
                    fold = row['fold']
                    if v_id not in seen_visits:
                        seen_visits.add(v_id)
                        scans = visit_to_scans.get(v_id, [])
                        if len(scans) >= 2:
                            candidates.append({
                                'scene_id': vid,
                                'visit_id': v_id,
                                'fold': fold,
                                'laser_scans': scans
                            })
    return candidates

def commit_and_push_scene(scene_id: str):
    branch = "benchmark/arkitscenes-n30"
    log(f"  [GIT CHECKPOINT] Adding evidence & results for scene {scene_id}...")
    subprocess.run(["git", "add", "pipeline/eval/results/", "batch_progress.jsonl"], cwd=REPO_ROOT, check=True)
    msg = f"benchmark: complete scene {scene_id}"
    res_commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if res_commit.returncode == 0:
        log(f"  [GIT COMMIT] Committed: {msg}")
    else:
        log(f"  [GIT COMMIT] Note: {res_commit.stdout or res_commit.stderr}")
    
    log(f"  [GIT PUSH] Pushing to origin {branch}...")
    res_push = subprocess.run(["git", "push", "origin", branch], cwd=REPO_ROOT, capture_output=True, text=True)
    if res_push.returncode == 0:
        log(f"  [GIT PUSH] Successfully pushed scene {scene_id} to origin/{branch}")
        return True
    else:
        log(f"  [GIT PUSH WARNING] Push failed: {res_push.stderr}. Persistent disk remains local fallback.")
        return False

def run_batch():
    log("==================================================================")
    log("  SPATIAL AI STAGE-2 UNATTENDED BENCHMARK RUNNER STARTING")
    log("==================================================================")

    # Initialize results list with initial 10 clean scenes
    results = list(INITIAL_10_SCENES)
    processed_scene_ids = {s["scene_id"] for s in results}
    processed_visit_ids = {s["visit_id"] for s in results}
    
    # Read any prior progress from batch_progress.jsonl if restarting
    progress_file = REPO_ROOT / "batch_progress.jsonl"
    if progress_file.is_file():
        with open(progress_file) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    processed_scene_ids.add(rec["scene_id"])
                    if "visit_id" in rec:
                        processed_visit_ids.add(rec["visit_id"])
                    if rec.get("status") == "success" and rec not in results:
                        results.append(rec)

    log(f"Starting batch execution. Current clean N = {len(results)}. Target clean N >= 30.")

    candidates = get_candidate_scenes(processed_visit_ids)
    log(f"Discovered {len(candidates)} total UNIQUE candidate scenes (unique physical rooms) from metadata CSVs.")


    clean_n = len(results)
    completed_count = len(results) - 10
    failed_count = 0
    skipped_count = 0

    for cand in candidates:
        if clean_n >= 30:
            log(f"TARGET REACHED: Clean N = {clean_n} >= 30! Halting scene processing.")
            break

        s_id = cand["scene_id"]
        v_id = cand["visit_id"]
        fold = cand["fold"]
        scans = cand["laser_scans"]

        if s_id in processed_scene_ids or v_id in processed_visit_ids:
            continue

        scene_num = len(results) + 1
        log(f"\n------------------------------------------------------------------")
        log(f"  PROCESSING SCENE {scene_num:2d} | Scene ID: {s_id} | Visit ID: {v_id} ({fold})")
        log(f"------------------------------------------------------------------")

        update_stage_file(s_id, "SELECTED")
        processed_scene_ids.add(s_id)
        work_dir = REPO_ROOT / ".batch_work" / s_id
        work_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        try:
            # Stage: LASER_DOWNLOADED & LASER_VERIFIED
            update_stage_file(s_id, "LASER_DOWNLOADED")
            laser_dir = REPO_ROOT / "samples" / "arkitscenes" / "raw" / "laser" / v_id
            laser_dir.mkdir(parents=True, exist_ok=True)
            
            heights = []
            laser_bytes_ok = []
            base_laser_url = f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/laser_scanner_point_clouds/{v_id}"
            
            for scan_id in scans:
                ply_name = f"{scan_id}.ply"
                ply_url = f"{base_laser_url}/{ply_name}"
                ply_path = laser_dir / ply_name
                
                cdn_len = get_cdn_content_length(ply_url)
                if not ply_path.exists() or ply_path.stat().st_size != cdn_len:
                    log(f"  Downloading laser scan {ply_name} ({cdn_len} bytes)...")
                    subprocess.run(["curl", "-sL", "-o", str(ply_path), ply_url], check=True, timeout=600)
                
                disk_len = ply_path.stat().st_size
                match = disk_len == cdn_len
                laser_bytes_ok.append(match)
                if not match:
                    raise ValueError(f"Byte mismatch on laser file {ply_name}: CDN={cdn_len}, Disk={disk_len}")
                
                gt_info = extract_ground_truth_from_ply(ply_path, strict_completeness=True)
                heights.append(float(gt_info["extractedHeight_m"]))

            update_stage_file(s_id, "GT_EXTRACTED")
            laser_gt = float(np.mean(heights))
            inter_scan_diff_cm = abs(heights[0] - heights[1]) * 100.0 if len(heights) >= 2 else 0.0
            log(f"  Laser GT extracted: {laser_gt:.4f} m (Inter-scan delta: {inter_scan_diff_cm:.2f} cm)")

            if inter_scan_diff_cm > 5.0:
                raise ValueError(f"GT_DISAGREEMENT_INVALID: laser scans disagree by {inter_scan_diff_cm:.2f} cm (> 5.0 cm threshold)")

            # Stage: CAPTURE_DOWNLOADED & CAPTURE_VERIFIED
            update_stage_file(s_id, "CAPTURE_DOWNLOADED")
            raw_dir = REPO_ROOT / "samples" / "arkitscenes" / "raw" / fold / s_id
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            capture_files = ["lowres_wide.traj", "lowres_wide.zip", "lowres_depth.zip", "confidence.zip", "lowres_wide_intrinsics.zip"]
            base_raw_url = f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/{fold}/{s_id}"
            
            capture_bytes_ok = []
            for cfile in capture_files:
                curl_url = f"{base_raw_url}/{cfile}"
                cfile_path = raw_dir / cfile
                cdn_len = get_cdn_content_length(curl_url)
                
                if not cfile_path.exists() or cfile_path.stat().st_size != cdn_len:
                    log(f"  Downloading capture archive {cfile} ({cdn_len} bytes)...")
                    subprocess.run(["curl", "-sL", "-o", str(cfile_path), curl_url], check=True, timeout=600)
                
                disk_len = cfile_path.stat().st_size
                match = disk_len == cdn_len
                capture_bytes_ok.append(match)
                if not match:
                    raise ValueError(f"Byte mismatch on capture file {cfile}: CDN={cdn_len}, Disk={disk_len}")
                
                if cfile.endswith('.zip'):
                    extracted_dir = raw_dir / cfile.replace('.zip', '')
                    if not extracted_dir.exists():
                        log(f"  Unzipping {cfile}...")
                        with zipfile.ZipFile(cfile_path, 'r') as zf:
                            zf.extractall(extracted_dir)
                        nested = extracted_dir / cfile.replace('.zip', '')
                        if nested.is_dir():
                            for item in nested.iterdir():
                                shutil.move(str(item), str(extracted_dir / item.name))
                            nested.rmdir()

            update_stage_file(s_id, "CAPTURE_VERIFIED")

            # Stage: CONNECTOR_VALID
            update_stage_file(s_id, "CONNECTOR_VALID")
            nc_dir = REPO_ROOT / "outputs" / f"nc_{s_id}"
            dev_dir = REPO_ROOT / "outputs" / f"dev_{s_id}"
            
            log(f"  Running Connector CLI on scene {s_id}...")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            cmd_conn = [
                sys.executable, "-m", "pipeline.connectors.cli",
                str(raw_dir), str(nc_dir),
                "--classification", "public_development_fixture", "--no-ai"
            ]
            res_conn = subprocess.run(cmd_conn, capture_output=True, text=True, env=env, timeout=600)
            if "NORMALIZED_CAPTURE_VALID" not in res_conn.stdout:
                log("Connector stdout:\n" + res_conn.stdout)
                log("Connector stderr:\n" + res_conn.stderr)
                raise RuntimeError(f"Connector validation failed for scene {s_id}")

            # Stage: GEOMETRY_COMPLETE & EVALUATED
            update_stage_file(s_id, "GEOMETRY_COMPLETE")
            log(f"  Running Geometry Pipeline on scene {s_id}...")
            cmd_geom = [
                sys.executable, "-m", "pipeline.geometry.run",
                str(nc_dir), str(dev_dir), "--frame-stride", "4"
            ]
            res_geom = subprocess.run(cmd_geom, capture_output=True, text=True, env=env, timeout=900)
            
            spatial_model_path = dev_dir / "spatial_model.json"
            if not spatial_model_path.is_file():
                log("Geometry stdout:\n" + res_geom.stdout)
                log("Geometry stderr:\n" + res_geom.stderr)
                raise RuntimeError(f"Geometry run failed for scene {s_id}")

            update_stage_file(s_id, "EVALUATED")
            model_data = json.loads(spatial_model_path.read_text())
            m_dict = {m['type']: m['value_m'] for m in model_data.get('measurements', [])}
            h_pred = m_dict.get('room_height')
            if h_pred is None:
                raise ValueError(f"No room_height measurement found in spatial_model.json for scene {s_id}")

            abs_err_cm = round(abs(h_pred - laser_gt) * 100.0, 2)
            signed_err_cm = round((h_pred - laser_gt) * 100.0, 2)
            gate_passed = abs_err_cm <= 1.50

            # Stage: RESULT_WRITTEN
            update_stage_file(s_id, "RESULT_WRITTEN")
            res_item = {
                "scene_num": scene_num,
                "scene_id": s_id,
                "visit_id": v_id,
                "fold": fold,
                "laser_gt_m": round(laser_gt, 4),
                "inter_scan_diff_cm": round(inter_scan_diff_cm, 2),
                "pred_height_m": round(h_pred, 4),
                "abs_error_cm": abs_err_cm,
                "signed_error_cm": signed_err_cm,
                "gate_passed": gate_passed,
                "all_laser_bytes_ok": all(laser_bytes_ok),
                "all_capture_bytes_ok": all(capture_bytes_ok),
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            results.append(res_item)
            clean_n += 1
            completed_count += 1

            # Save individual scene result JSON
            scene_res_path = REPO_ROOT / "pipeline" / "eval" / "results" / f"scene_{s_id}.json"
            scene_res_path.write_text(json.dumps(res_item, indent=2))
            log_progress(res_item)

            log(f"  --> RESULT SCENE {scene_num:2d}: GT={laser_gt:.4f}m | Pred={h_pred:.4f}m | AbsErr={abs_err_cm:.2f}cm | GatePassed={gate_passed}")

            # Stage: COMMITTED & PUSHED
            update_stage_file(s_id, "COMMITTED")
            push_ok = commit_and_push_scene(s_id)
            if push_ok:
                update_stage_file(s_id, "PUSHED")

            # Cleanup disk space
            if laser_dir.is_dir():
                shutil.rmtree(laser_dir, ignore_errors=True)
            for cfile in capture_files:
                if cfile.endswith('.zip'):
                    ext_p = raw_dir / cfile.replace('.zip', '')
                    if ext_p.is_dir():
                        shutil.rmtree(ext_p, ignore_errors=True)
            if nc_dir.is_dir():
                shutil.rmtree(nc_dir, ignore_errors=True)
            if work_dir.is_dir():
                shutil.rmtree(work_dir, ignore_errors=True)

        except Exception as exc:
            failed_count += 1
            log(f"  [ERROR] Failure during scene {s_id}: {exc}")
            fail_record = {
                "scene_id": s_id,
                "visit_id": v_id,
                "status": "failed",
                "stage": Path("/tmp/current_batch_stage").read_text() if Path("/tmp/current_batch_stage").exists() else "UNKNOWN",
                "error": str(exc),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_sec": round(time.time() - start_time, 2)
            }
            log_progress(fail_record)
            # Retain work_dir for inspection

    log("\n==================================================================")
    log(f"  SCENE PROCESSING COMPLETE: Clean N = {clean_n} reached!")
    log("==================================================================")

    # Post-Evaluation Execution & Baseline
    log("Executing baseline comparison (RTAB-Map / python baseline)...")
    run_baseline_comparison()

    # Collect failed scenes from batch_progress.jsonl
    failed_scenes = []
    if progress_file.is_file():
        with open(progress_file) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("status") == "failed":
                        failed_scenes.append(rec)

    total_attempted = completed_count + len(failed_scenes)
    overall_success_rate = round(clean_n / (10 + total_attempted), 4) if (10 + total_attempted) > 0 else 0.0

    consolidated_results = {
        "dataset": "ARKitScenes",
        "clean_n": clean_n,
        "completed": completed_count,
        "failed": len(failed_scenes),
        "total_candidates_attempted": total_attempted,
        "overall_success_rate": overall_success_rate,
        "aggregate": {
            "mae_cm": stats["mean"],
            "median_cm": stats["median"],
            "std_cm": stats["std"],
            "p90_cm": stats["p90"],
            "min_cm": stats["min"],
            "max_cm": stats["max"],
            "gate_pass_rate": round(gate_passes / len(results), 4) if results else 0.0,
            "bootstrap_95ci": stats["bootstrap_95ci"]
        },
        "bias_regression": regression,
        "failedScenes": failed_scenes,
        "scenes": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


    latest_run_path = REPO_ROOT / "pipeline" / "eval" / "results" / "latest_run.json"
    latest_run_path.write_text(json.dumps(consolidated_results, indent=2))
    log(f"Consolidated results written to {latest_run_path}")

    # Run Pytest Verification Suite
    log("Running final pytest verification suite...")
    res_pytest = subprocess.run([sys.executable, "-m", "pytest", "pipeline/tests/", "-q"], cwd=REPO_ROOT, capture_output=True, text=True)
    log(f"Pytest summary: {res_pytest.stdout.splitlines()[-1] if res_pytest.stdout else 'Complete'}")

    return consolidated_results, clean_n, completed_count, failed_count, skipped_count

if __name__ == "__main__":
    run_batch()
