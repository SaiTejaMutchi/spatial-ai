import os, sys, glob, subprocess, time, json, shutil, struct
import numpy as np

def run_full_scene():
    scene_id = '42444733'
    raw_dir = os.path.expanduser(f'~/spatial-ai/samples/arkitscenes/raw/Training/{scene_id}')
    rgb_dir = os.path.join(raw_dir, 'lowres_wide')
    depth_dir = os.path.join(raw_dir, 'lowres_depth')
    
    seq_dir = os.path.expanduser('~/spatial-ai/output/rtabmap_full_sequence')
    if os.path.exists(seq_dir):
        shutil.rmtree(seq_dir)
    rgb_sync = os.path.join(seq_dir, 'rgb_sync')
    depth_sync = os.path.join(seq_dir, 'depth_sync')
    os.makedirs(rgb_sync, exist_ok=True)
    os.makedirs(depth_sync, exist_ok=True)
    
    # Calibration YAML matching 256x192
    calib_yaml = os.path.join(seq_dir, 'rtabmap_calib.yaml')
    with open(calib_yaml, 'w') as f:
        f.write('''%YAML:1.0
image_width: 256
image_height: 192
camera_name: rtabmap_calib
camera_matrix:
   rows: 3
   cols: 3
   data: [ 213.413, 0., 128.751, 0., 213.413, 95.7609, 0., 0., 1. ]
distortion_coefficients:
   rows: 1
   cols: 5
   data: [ 0., 0., 0., 0., 0. ]
distortion_model: plumb_bob
rectification_matrix:
   rows: 3
   cols: 3
   data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
projection_matrix:
   rows: 3
   cols: 4
   data: [ 213.413, 0., 128.751, 0., 0., 213.413, 95.7609, 0., 0., 0., 1., 0. ]
''')

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    depth_files = set(os.path.basename(f) for f in glob.glob(os.path.join(depth_dir, '*.png')))
    
    matched = 0
    for rgb_f in rgb_files:
        fname = os.path.basename(rgb_f)
        if fname in depth_files:
            ts_name = f'{matched+1:06d}.png'
            os.symlink(rgb_f, os.path.join(rgb_sync, ts_name))
            os.symlink(os.path.join(depth_dir, fname), os.path.join(depth_sync, ts_name))
            matched += 1
            
    print(f'Matched {matched} total RGB-D frame pairs across full scene {scene_id}')
    
    out_dir = os.path.expanduser('~/spatial-ai/output/rtabmap_full_test')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dataset_bin = os.path.expanduser('~/rtabmap/build/bin/rtabmap-rgbd_dataset')
    cmd = [
        dataset_bin,
        '--output', out_dir,
        '--output_name', 'rtabmap',
        '--quiet',
        seq_dir
    ]
    
    print('\nExecuting RTAB-Map dataset command across all frames...')
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    print(f'Exit Code: {proc.returncode}')
    print(f'Execution Time: {elapsed:.2f} seconds')

    db_path = os.path.join(out_dir, 'rtabmap.db')
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    print(f'rtabmap.db size: {db_size} bytes ({db_size / (1024*1024):.2f} MB)')

    # Export PLY
    export_bin = os.path.expanduser('~/rtabmap/build/bin/rtabmap-export')
    exp_cmd = [export_bin, '--ply', '--voxel', '0', '--output', 'full_cloud', db_path]
    print('\nExporting full point cloud PLY...')
    exp_proc = subprocess.run(exp_cmd, cwd=out_dir, capture_output=True, text=True)
    print('Export Exit Code:', exp_proc.returncode)

    ply_path = os.path.join(out_dir, 'full_cloud_cloud.ply')
    if os.path.exists(ply_path):
        with open(ply_path, 'rb') as f:
            num_vertices = 0
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                if line == 'end_header':
                    break

            record_struct = struct.Struct('<ffffffBBBf')
            pts = []
            for _ in range(num_vertices):
                data = f.read(record_struct.size)
                if len(data) < record_struct.size:
                    break
                vals = record_struct.unpack(data)
                pts.append([vals[0], vals[1], vals[2]])

        pts = np.array(pts)
        print(f'\n=== FULL TRAJECTORY POINT CLOUD ANALYSIS ({len(pts)} points) ===')
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        print(f'X bounds: min={x.min():.4f}m, max={x.max():.4f}m (Span: {x.max()-x.min():.4f}m / {(x.max()-x.min())*100:.2f}cm)')
        print(f'Y bounds: min={y.min():.4f}m, max={y.max():.4f}m (Span: {y.max()-y.min():.4f}m / {(y.max()-y.min())*100:.2f}cm)')
        print(f'Z bounds: min={z.min():.4f}m, max={z.max():.4f}m (Span: {z.max()-z.min():.4f}m / {(z.max()-z.min())*100:.2f}cm)')

        percentiles = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]
        print('\n--- Z-AXIS PERCENTILES (meters) ---')
        for p in percentiles:
            val = np.percentile(z, p)
            print(f' {p:3d}th percentile: {val:8.4f} m ({val*100:7.2f} cm)')

        hist, bin_edges = np.histogram(z, bins=10)
        print('\n--- Z-AXIS HISTOGRAM (10 Bins) ---')
        for i in range(len(hist)):
            print(f' [{bin_edges[i]:7.3f}m to {bin_edges[i+1]:7.3f}m]: {hist[i]:6d} points ({(hist[i]/len(z))*100:5.1f}%)')

        p2 = np.percentile(z, 2)
        p98 = np.percentile(z, 98)
        h_p2_p98 = abs(p98 - p2)

        p5 = np.percentile(z, 5)
        p95 = np.percentile(z, 95)
        h_p5_p95 = abs(p95 - p5)

        print(f'\nExtracted Height (P2 to P98): {h_p2_p98:.4f} m ({h_p2_p98*100:.2f} cm)')
        print(f'Extracted Height (P5 to P95): {h_p5_p95:.4f} m ({h_p5_p95*100:.2f} cm)')

if __name__ == '__main__':
    run_full_scene()
