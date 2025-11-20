#!/usr/bin/env python3
"""
Complete reconstruction pipeline with depth estimation:
1. SfM: Estimate camera poses from RGB images
2. Dense stereo: Estimate depth maps between image pairs
3. Depth fusion: Merge depth maps into 3D point cloud
4. Mesh generation
"""

import os
import cv2
import numpy as np
import pycolmap
from pathlib import Path
import shutil
import open3d as o3d

USE_STANFORD = True

if USE_STANFORD:
    DATA_DIR = "./stanford_extracted/seq-01"
    OUTPUT_DIR = "./sfm_dense_pipeline_stanford"
    IMAGE_PATTERN = "frame-*.color.png"
    FRAME_SKIP = 5 
    MAX_FRAMES = 20 
else:
    DATA_DIR = "./images2"
    OUTPUT_DIR = "./sfm_dense_pipeline_iphone"
    IMAGE_PATTERN = "*.jpeg"
    FRAME_SKIP = 1
    MAX_FRAMES = 100


shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_FOLDER = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

print("="*70)
print("FULL SfM + DENSE DEPTH ESTIMATION PIPELINE")
print("="*70)
print(f"Dataset: {'Stanford RGBD' if USE_STANFORD else 'iPhone Images'}")
print(f"Source: {DATA_DIR}")

image_files = sorted(Path(DATA_DIR).glob(IMAGE_PATTERN))
selected_frames = image_files[::FRAME_SKIP][:MAX_FRAMES]

print(f"Found {len(image_files)} total images")
print(f"Selected {len(selected_frames)} images")

for i, src_path in enumerate(selected_frames):
    dst_path = os.path.join(IMAGE_FOLDER, f"image_{i:04d}{src_path.suffix}")
    shutil.copy(src_path, dst_path)

print(f"✓ Copied {len(selected_frames)} images")

print(f"\nConfiguration:")
print(f"  Frame skip: {FRAME_SKIP}")
print(f"  Max frames: {MAX_FRAMES}")
print(f"  Using: {len(selected_frames)} images")

# STEP 1: SfM - Estimate camera poses
print("\n" + "="*70)
print("STEP 1: STRUCTURE FROM MOTION (SfM)")
print("="*70)

COLMAP_DIR = os.path.join(OUTPUT_DIR, "colmap")
os.makedirs(COLMAP_DIR, exist_ok=True)

database_path = os.path.join(COLMAP_DIR, "database.db")
sparse_path = os.path.join(COLMAP_DIR, "sparse")
os.makedirs(sparse_path, exist_ok=True)

print("1a. Feature extraction...")
pycolmap.extract_features(
    database_path=database_path,
    image_path=IMAGE_FOLDER
)

print("1b. Feature matching...")
pycolmap.match_exhaustive(database_path=database_path)

print("1c. Incremental mapping...")
options = pycolmap.IncrementalPipelineOptions()
maps = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=IMAGE_FOLDER,
    output_path=sparse_path,
    options=options
)

if len(maps) == 0:
    print("✗ SfM failed - no reconstruction!")
    exit(1)

reconstruction = maps[0]
print(f"\n✓ SfM complete:")
print(f"  - {reconstruction.num_reg_images()} registered images")
print(f"  - {len(reconstruction.points3D):,} sparse 3D points")

sparse_ply = os.path.join(OUTPUT_DIR, "sparse_points.ply")
reconstruction.export_PLY(sparse_ply)
print(f"  - Saved: {sparse_ply}")

# STEP 2: DENSE DEPTH ESTIMATION
print("\n" + "="*70)
print("STEP 2: DENSE DEPTH ESTIMATION (STEREO MATCHING)")
print("="*70)

registered_images = {}
for img_id, img in reconstruction.images.items():
    img_path = os.path.join(IMAGE_FOLDER, img.name)
    registered_images[img.name] = {
        'path': img_path,
        'image': cv2.imread(img_path),
        'pose': img.cam_from_world(),
        'camera': reconstruction.cameras[img.camera_id]
    }

print(f"Processing {len(registered_images)} registered images")

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=256,
    blockSize=3,
    P1=8 * 3 * 3**2,
    P2=32 * 3 * 3**2,
    disp12MaxDiff=2,
    uniquenessRatio=5,
    speckleWindowSize=150,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

image_names = sorted(registered_images.keys())
all_depth_points = []
all_depth_colors = []

for i in range(len(image_names) - 1):
    img1_name = image_names[i]
    img2_name = image_names[i + 1]
    
    print(f"\n2.{i+1}. Processing pair: {img1_name} <-> {img2_name}")
    
    img1_data = registered_images[img1_name]
    img2_data = registered_images[img2_name]
    
    img1 = img1_data['image']
    img2 = img2_data['image']
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    scale = 0.5
    gray1_small = cv2.resize(gray1, None, fx=scale, fy=scale)
    gray2_small = cv2.resize(gray2, None, fx=scale, fy=scale)
    
    print("   Computing disparity map...")
    disparity_small = stereo.compute(gray1_small, gray2_small).astype(np.float32) / 16.0
    
    disparity = cv2.resize(disparity_small, (gray1.shape[1], gray1.shape[0])) / scale
    
    valid = (disparity > 2.0) & (disparity < 200)
    
    num_valid = valid.sum()
    print(f"   Valid disparities: {num_valid:,} / {valid.size:,}")
    print(f"   Disparity range: [{disparity[valid].min():.1f}, {disparity[valid].max():.1f}]")
    
    if num_valid < 10000:
        print(f"   ⚠ Too few valid disparities, skipping this pair")
        continue
    
    pose1 = img1_data['pose']
    pose2 = img2_data['pose']
    
    R1 = pose1.rotation.matrix()
    t1 = pose1.translation
    cam1_world = -R1.T @ t1
    
    R2 = pose2.rotation.matrix()
    t2 = pose2.translation
    cam2_world = -R2.T @ t2
    
    baseline = np.linalg.norm(cam2_world - cam1_world)
    
    camera = img1_data['camera']
    fx = camera.params[0]
    
    depth = np.zeros_like(disparity)
    depth[valid] = (baseline * fx) / (disparity[valid] + 1e-6)
    
    depth = np.clip(depth, 0.5, 20.0)
    
    print(f"   Baseline: {baseline:.3f}")
    print(f"   Depth range: [{depth[valid].min():.3f}, {depth[valid].max():.3f}]")
    print(f"   Valid pixels: {valid.sum():,} / {valid.size:,}")
    
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    cx = camera.params[2] if len(camera.params) > 2 else w / 2
    cy = camera.params[3] if len(camera.params) > 3 else h / 2
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fx
    z = depth
    
    points_cam = np.stack([x[valid], y[valid], z[valid]], axis=1)
    
    points_cam_h = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)
    
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R1
    T_cam_world[:3, 3] = t1
    
    T_world_cam = np.linalg.inv(T_cam_world)
    
    points_world_h = (T_world_cam @ points_cam_h.T).T
    points_world = points_world_h[:, :3]
    
    colors = img1[valid].astype(float)[:, [2, 1, 0]] / 255.0 
    
    all_depth_points.append(points_world)
    all_depth_colors.append(colors)
    
    print(f"   Generated {len(points_world):,} 3D points")

# STEP 3: FUSE DEPTH MAPS
print("\n" + "="*70)
print("STEP 3: FUSING DEPTH MAPS")
print("="*70)

dense_points = np.vstack(all_depth_points)
dense_colors = np.vstack(all_depth_colors)

print(f"Total dense points: {len(dense_points):,}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(dense_points)
pcd.colors = o3d.utility.Vector3dVector(dense_colors)

print("Downsampling point cloud...")
pcd_down = pcd.voxel_down_sample(voxel_size=0.02)
print(f"Downsampled to {len(pcd_down.points):,} points")

dense_ply = os.path.join(OUTPUT_DIR, "dense_points.ply")
o3d.io.write_point_cloud(dense_ply, pcd_down)
print(f"✓ Saved dense point cloud: {dense_ply}")

# STEP 4: MESH GENERATION
print("\n" + "="*70)
print("STEP 4: MESH GENERATION")
print("="*70)

print("Estimating normals...")
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)

print("Creating mesh (Ball Pivoting)...")
try:
    distances = pcd_down.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_down,
        o3d.utility.DoubleVector([radius, radius * 2])
    )
    
    mesh_path = os.path.join(OUTPUT_DIR, "mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"✓ Saved mesh: {mesh_path}")
    print(f"  - {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
except Exception as e:
    print(f"✗ Mesh creation failed: {e}")

print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)
print(f"Results in: {OUTPUT_DIR}")
print(f"  - Sparse SfM: sparse_points.ply ({len(reconstruction.points3D):,} points)")
print(f"  - Dense reconstruction: dense_points.ply ({len(pcd_down.points):,} points)")
print(f"  - Mesh: mesh.ply")

# Visualize
print("\nVisualizing dense point cloud...")
o3d.visualization.draw_geometries([pcd_down], window_name="Dense Reconstruction")
