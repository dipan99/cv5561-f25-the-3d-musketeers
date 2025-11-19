#!/usr/bin/env python3
"""
RGBD-based 3D reconstruction using depth maps and known poses
Much more efficient and accurate than SfM!
"""

import os
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "./stanford_extracted/seq-01"
OUTPUT_DIR = "./rgbd_reconstruction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use every Nth frame to save memory
FRAME_SKIP = 20  # Use every 20th frame
MAX_FRAMES = 50  # Maximum number of frames to use

# Camera intrinsics (typical values for RGB-D cameras, adjust if you have actual values)
# These are approximate - adjust based on your dataset
WIDTH = 640
HEIGHT = 480
FX = 525.0  # focal length x
FY = 525.0  # focal length y
CX = 319.5  # principal point x
CY = 239.5  # principal point y

print("="*70)
print("RGBD 3D RECONSTRUCTION")
print("="*70)
print(f"Data directory: {DATA_DIR}")
print(f"Using every {FRAME_SKIP}th frame")
print(f"Maximum frames: {MAX_FRAMES}")

# Get all color images
color_files = sorted(Path(DATA_DIR).glob("frame-*.color.png"))
print(f"\nFound {len(color_files)} color images")

# Select subset of frames
selected_frames = color_files[::FRAME_SKIP][:MAX_FRAMES]
print(f"Selected {len(selected_frames)} frames for reconstruction")

def load_pose(pose_file):
    """Load 4x4 camera pose matrix"""
    return np.loadtxt(pose_file)

def depth_to_pointcloud(color_img, depth_img, pose, fx, fy, cx, cy):
    """Convert RGBD image to 3D point cloud in world coordinates"""
    h, w = depth_img.shape
    
    # Create coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert depth to meters (assuming depth is in mm)
    depth_m = depth_img.astype(float) / 1000.0
    
    # Filter out invalid depths
    valid = (depth_m > 0) & (depth_m < 5.0)  # Keep depths between 0-5m
    
    # Back-project to 3D camera coordinates
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    
    # Stack into Nx3 array
    points_cam = np.stack([x[valid], y[valid], z[valid]], axis=1)
    colors = color_img[valid] / 255.0  # Normalize to [0, 1]
    
    # Transform to world coordinates
    # pose is camera-to-world transform
    points_cam_h = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)
    points_world_h = (pose @ points_cam_h.T).T
    points_world = points_world_h[:, :3]
    
    return points_world, colors

print("\n" + "="*70)
print("PROCESSING FRAMES")
print("="*70)

# Accumulate all points
all_points = []
all_colors = []

for i, color_file in enumerate(selected_frames):
    frame_num = color_file.stem.split('.')[0]
    depth_file = color_file.parent / f"{frame_num}.depth.png"
    pose_file = color_file.parent / f"{frame_num}.pose.txt"
    
    if not depth_file.exists() or not pose_file.exists():
        print(f"⚠ Skipping {frame_num}: missing depth or pose")
        continue
    
    # Load data
    color = cv2.imread(str(color_file))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
    pose = load_pose(pose_file)
    
    # Get actual image size
    h, w = depth.shape
    if i == 0:
        # Update camera intrinsics based on actual image size
        CX = w / 2.0
        CY = h / 2.0
        FX = w * 0.8  # Rough estimate
        FY = h * 0.8
        print(f"Image size: {w}x{h}")
        print(f"Using intrinsics: fx={FX:.1f}, fy={FY:.1f}, cx={CX:.1f}, cy={CY:.1f}")
    
    # Convert to point cloud
    points, colors = depth_to_pointcloud(color, depth, pose, FX, FY, CX, CY)
    
    all_points.append(points)
    all_colors.append(colors)
    
    print(f"Frame {i+1}/{len(selected_frames)}: {frame_num} - {len(points):,} points")

# Combine all points
print("\n" + "="*70)
print("COMBINING POINT CLOUDS")
print("="*70)

points_combined = np.vstack(all_points)
colors_combined = np.vstack(all_colors)

print(f"Total points: {len(points_combined):,}")
print(f"Point cloud bounds:")
print(f"  X: [{points_combined[:, 0].min():.2f}, {points_combined[:, 0].max():.2f}]")
print(f"  Y: [{points_combined[:, 1].min():.2f}, {points_combined[:, 1].max():.2f}]")
print(f"  Z: [{points_combined[:, 2].min():.2f}, {points_combined[:, 2].max():.2f}]")

# Create Open3D point cloud
print("\n" + "="*70)
print("CREATING OPEN3D POINT CLOUD")
print("="*70)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_combined)
pcd.colors = o3d.utility.Vector3dVector(colors_combined)

# Downsample for visualization (optional)
print("Downsampling point cloud...")
pcd_down = pcd.voxel_down_sample(voxel_size=0.02)  # 2cm voxels
print(f"Downsampled to {len(pcd_down.points):,} points")

# Save point cloud
ply_path = os.path.join(OUTPUT_DIR, "reconstruction.ply")
o3d.io.write_point_cloud(ply_path, pcd_down)
print(f"\n✓ Saved point cloud to {ply_path}")

# Optional: Estimate normals and create mesh
print("\n" + "="*70)
print("ESTIMATING NORMALS")
print("="*70)

pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
print("✓ Normals estimated")

# Try Poisson surface reconstruction
print("\n" + "="*70)
print("CREATING MESH (POISSON)")
print("="*70)

try:
    print("Attempting Poisson reconstruction (depth=8)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_down, depth=8, width=0, scale=1.1, linear_fit=False
    )
    print(f"✓ Created mesh with {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    
    # Remove low-density vertices (noise)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"✓ Cleaned mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    
    # Save mesh
    mesh_path = os.path.join(OUTPUT_DIR, "mesh_poisson.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"✓ Saved Poisson mesh to {mesh_path}")
except Exception as e:
    print(f"✗ Poisson mesh creation failed: {e}")
    mesh = None

# Try alternative: Ball Pivoting Algorithm
print("\n" + "="*70)
print("CREATING MESH (BALL PIVOTING)")
print("="*70)

try:
    print("Estimating point cloud radius...")
    distances = pcd_down.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    
    print(f"Using ball radii: [{radius:.4f}, {radius*2:.4f}]")
    
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_down,
        o3d.utility.DoubleVector([radius, radius * 2])
    )
    print(f"✓ Created BPA mesh with {len(bpa_mesh.vertices):,} vertices, {len(bpa_mesh.triangles):,} triangles")
    
    # Save BPA mesh
    bpa_path = os.path.join(OUTPUT_DIR, "mesh_bpa.ply")
    o3d.io.write_triangle_mesh(bpa_path, bpa_mesh)
    print(f"✓ Saved BPA mesh to {bpa_path}")
except Exception as e:
    print(f"✗ BPA mesh creation failed: {e}")
    bpa_mesh = None

print("\n" + "="*70)
print("VISUALIZING")
print("="*70)

# Create visualization
print("Creating visualization...")
o3d.visualization.draw_geometries(
    [pcd_down],
    window_name="RGBD Reconstruction",
    width=1024,
    height=768
)

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"Results saved to: {OUTPUT_DIR}")
print(f"  - Point cloud: reconstruction.ply")
if mesh is not None:
    print(f"  - Poisson mesh: mesh_poisson.ply")
if bpa_mesh is not None:
    print(f"  - BPA mesh: mesh_bpa.ply")
