import os, sys, glob, subprocess, time, json, shutil
import numpy as np

def run_real_rtab_scene():
    scene_id = '42444733'
    raw_dir = os.path.expanduser(f'~/spatial-ai/samples/arkitscenes/raw/Training/{scene_id}')
    rgb_dir = os.path.join(raw_dir, 'lowres_wide')
    depth_dir = os.path.join(raw_dir, 'lowres_depth')
    
    seq_dir = os.path.expanduser('~/spatial-ai/output/rtabmap_scene_sequence')
    if os.path.exists(seq_dir):
        shutil.rmtree(seq_dir)
    rgb_sync = os.path.join(seq_dir, 'rgb_sync')
    depth_sync = os.path.join(seq_dir, 'depth_sync')
    os.makedirs(rgb_sync, exist_ok=True)
    os.makedirs(depth_sync, exist_ok=True)
    
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))[:200]
    depth_files = set(os.path.basename(f) for f in glob.glob(os.path.join(depth_dir, '*.png')))
    
    matched = 0
    for rgb_f in rgb_files:
        fname = os.path.basename(rgb_f)
        if fname in depth_files:
            ts_name = f'{matched+1:06d}.png'
            os.symlink(rgb_f, os.path.join(rgb_sync, ts_name))
            os.symlink(os.path.join(depth_dir, fname), os.path.join(depth_sync, ts_name))
            matched += 1
            
    print(f'Matched {matched} RGB-D frame pairs')
    
    out_dir = os.path.expanduser('~/spatial-ai/output/rtabmap_scene_test')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dataset_bin = os.path.expanduser('~/rtabmap/build/bin/rtabmap-rgbd_dataset')
    export_bin = os.path.expanduser('~/rtabmap/build/bin/rtabmap-export')
    
    cmd = [
        dataset_bin,
        '--output', out_dir,
        '--output_name', 'rtabmap',
        '--quiet',
        seq_dir
    ]
    
    print('\nExecuting Command:', ' '.join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    print(f'Exit Code: {proc.returncode}')
    print(f'Execution Time: {elapsed:.2f} seconds')

    db_path = os.path.join(out_dir, 'rtabmap.db')
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    print(f'rtabmap.db size: {db_size} bytes ({db_size / (1024*1024):.2f} MB)')

    ply_path = os.path.join(out_dir, 'cloud.ply')
    if os.path.exists(db_path) and db_size > 0:
        exp_cmd = [export_bin, '--ply', '--output', ply_path, db_path]
        print('Exporting 3D PLY Cloud:', ' '.join(exp_cmd))
        exp_proc = subprocess.run(exp_cmd, capture_output=True, text=True)
        print('Export Exit Code:', exp_proc.returncode)

    extracted_height = None
    if os.path.exists(ply_path) and os.path.getsize(ply_path) > 0:
        pts = []
        with open(ply_path, 'r', errors='ignore') as f:
            header = True
            for line in f:
                if header:
                    if line.strip() == 'end_header':
                        header = False
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        pass
        pts = np.array(pts)
        print(f'Loaded {len(pts)} points from PLY')
        if len(pts) > 100:
            z_vals = pts[:, 2]
            floor_z = float(np.percentile(z_vals, 2))
            ceil_z = float(np.percentile(z_vals, 98))
            extracted_height = abs(ceil_z - floor_z)
            print(f'Extracted Height: {extracted_height:.4f} meters ({extracted_height*100:.2f} cm)')

if __name__ == '__main__':
    run_real_rtab_scene()
