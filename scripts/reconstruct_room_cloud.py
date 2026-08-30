import os, sys, glob, time
import numpy as np
from PIL import Image

def euler_to_rot(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def ransac_plane(pts, max_iter=200, thresh=0.03):
    best_plane = None
    best_inliers = []
    num_pts = len(pts)
    if num_pts < 3:
        return None, []
        
    for _ in range(max_iter):
        idx = np.random.choice(num_pts, 3, replace=False)
        p1, p2, p3 = pts[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)
        
        dists = np.abs(np.dot(pts, normal) + d)
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)
            
    return best_plane, best_inliers

def reconstruct_full_room():
    scene_id = '42444733'
    raw_dir = os.path.expanduser(f'~/spatial-ai/samples/arkitscenes/raw/Training/{scene_id}')
    rgb_dir = os.path.join(raw_dir, 'lowres_wide')
    depth_dir = os.path.join(raw_dir, 'lowres_depth')
    traj_file = os.path.join(raw_dir, 'lowres_wide.traj')
    
    fx, fy = 213.413, 213.413
    cx, cy = 128.751, 95.7609
    
    traj_poses = []
    with open(traj_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                rx, ry, rz = float(parts[4]), float(parts[5]), float(parts[6])
                R = euler_to_rot(rx, ry, rz)
                t = np.array([tx, ty, tz])
                traj_poses.append((ts, R, t))

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    depth_files = set(os.path.basename(f) for f in glob.glob(os.path.join(depth_dir, '*.png')))
    
    rgb_map = {}
    for f in rgb_files:
        fname = os.path.basename(f)
        if fname in depth_files:
            ts_str = fname.replace(f'{scene_id}_', '').replace('.png', '')
            try:
                rgb_map[float(ts_str)] = f
            except ValueError:
                pass

    rgb_ts_list = np.array(sorted(rgb_map.keys()))
    h, w = 192, 256
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    all_world_pts = []
    
    for ts, R, t in traj_poses[::2]:
        idx = np.argmin(np.abs(rgb_ts_list - ts))
        closest_rgb_ts = rgb_ts_list[idx]
        if abs(closest_rgb_ts - ts) > 0.05:
            continue
            
        rgb_path = rgb_map[closest_rgb_ts]
        fname = os.path.basename(rgb_path)
        depth_path = os.path.join(depth_dir, fname)
        
        d_img = Image.open(depth_path)
        d_arr = np.array(d_img).astype(np.float32) / 1000.0
        
        valid_mask = (d_arr > 0.3) & (d_arr < 4.0)
        sub_mask = valid_mask & ((u_grid % 4 == 0) & (v_grid % 4 == 0))
        
        z_c = d_arr[sub_mask]
        u_c = u_grid[sub_mask]
        v_c = v_grid[sub_mask]
        
        x_c = (u_c - cx) * z_c / fx
        y_c = (v_c - cy) * z_c / fy
        
        pts_cam = np.vstack((x_c, y_c, z_c)).T
        pts_world = pts_cam.dot(R.T) + t
        all_world_pts.append(pts_world)
        
    all_world_pts = np.vstack(all_world_pts)
    print(f'=== REAL 6DOF TRAJECTORY RECONSTRUCTED ROOM CLOUD ({len(all_world_pts)} points) ===')
    x, y, z = all_world_pts[:, 0], all_world_pts[:, 1], all_world_pts[:, 2]

    print(f'X bounds: min={x.min():.4f}m, max={x.max():.4f}m (Span: {x.max()-x.min():.4f}m / {(x.max()-x.min())*100:.2f}cm)')
    print(f'Y bounds: min={y.min():.4f}m, max={y.max():.4f}m (Span: {y.max()-y.min():.4f}m / {(y.max()-y.min())*100:.2f}cm)')
    print(f'Z bounds: min={z.min():.4f}m, max={z.max():.4f}m (Span: {z.max()-z.min():.4f}m / {(z.max()-z.min())*100:.2f}cm)')

    # PCA Alignment to align gravity / principal axes
    cov = np.cov(all_world_pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    aligned_pts = all_world_pts.dot(eigvecs)
    ax, ay, az = aligned_pts[:, 0], aligned_pts[:, 1], aligned_pts[:, 2]

    print('\n=== PCA-ALIGNED SPATIAL EXTENTS ===')
    print(f'PCA Axis 0 (Minor): min={ax.min():.4f}m, max={ax.max():.4f}m (Span: {ax.max()-ax.min():.4f}m)')
    print(f'PCA Axis 1 (Medium): min={ay.min():.4f}m, max={ay.max():.4f}m (Span: {ay.max()-ay.min():.4f}m)')
    print(f'PCA Axis 2 (Major): min={az.min():.4f}m, max={az.max():.4f}m (Span: {az.max()-az.min():.4f}m)')

    # Floor / Ceiling RANSAC plane fitting on minor axis
    floor_pts_candidates = aligned_pts[ax < np.percentile(ax, 15)]
    ceil_pts_candidates = aligned_pts[ax > np.percentile(ax, 85)]

    f_plane, f_inliers = ransac_plane(floor_pts_candidates, max_iter=300, thresh=0.03)
    c_plane, c_inliers = ransac_plane(ceil_pts_candidates, max_iter=300, thresh=0.03)

    if f_plane and c_plane:
        f_norm, f_d = f_plane
        c_norm, c_d = c_plane
        # Distance between parallel planes: |d2 - d1| / ||normal||
        dist = abs(c_d - f_d) if np.dot(f_norm, c_norm) > 0 else abs(c_d + f_d)
        print(f'\n--- RANSAC PLANE FITTING RESULT ---')
        print(f'Floor plane inliers: {len(f_inliers)} points')
        print(f'Ceiling plane inliers: {len(c_inliers)} points')
        print(f'Perpendicular Floor-to-Ceiling Distance: {dist:.4f} meters ({dist*100:.2f} cm)')

if __name__ == '__main__':
    reconstruct_full_room()
