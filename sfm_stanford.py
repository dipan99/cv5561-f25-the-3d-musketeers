#!/usr/bin/env python3
"""
SfM reconstruction using only RGB images from Stanford dataset
Ignoring depth maps and poses - estimating everything from scratch
"""

import os
import cv2
import numpy as np
import pycolmap
from pathlib import Path
import shutil

# Configuration
DATA_DIR = "./stanford_extracted/seq-01"
OUTPUT_DIR = "./sfm_stanford"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extract only color images to a separate folder
IMAGE_FOLDER = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

FRAME_SKIP = 20  # Use every 20th frame
MAX_FRAMES = 30  # Use 30 frames

print("="*70)
print("SfM RECONSTRUCTION (RGB ONLY)")
print("="*70)
print(f"Source: {DATA_DIR}")
print(f"Frame skip: {FRAME_SKIP}")
print(f"Max frames: {MAX_FRAMES}")

# Get color images
color_files = sorted(Path(DATA_DIR).glob("frame-*.color.png"))
selected_frames = color_files[::FRAME_SKIP][:MAX_FRAMES]

print(f"\nFound {len(color_files)} total frames")
print(f"Selected {len(selected_frames)} frames for SfM")

# Copy selected images to working directory
print("\n" + "="*70)
print("COPYING IMAGES")
print("="*70)

for i, src_path in enumerate(selected_frames):
    frame_num = src_path.stem.split('.')[0]
    dst_path = os.path.join(IMAGE_FOLDER, f"frame_{i:04d}.png")
    shutil.copy(src_path, dst_path)
    print(f"Copied {frame_num} -> frame_{i:04d}.png")

print(f"\n✓ Copied {len(selected_frames)} images to {IMAGE_FOLDER}")

# Setup COLMAP paths
COLMAP_DIR = os.path.join(OUTPUT_DIR, "colmap")
os.makedirs(COLMAP_DIR, exist_ok=True)

database_path = os.path.join(COLMAP_DIR, "database.db")
sparse_path = os.path.join(COLMAP_DIR, "sparse")
os.makedirs(sparse_path, exist_ok=True)

print("\n" + "="*70)
print("STEP 1: FEATURE EXTRACTION")
print("="*70)

pycolmap.extract_features(
    database_path=database_path,
    image_path=IMAGE_FOLDER,
    sift_options={
        "max_num_features": 8192,
    }
)

print("✓ Feature extraction complete")

print("\n" + "="*70)
print("STEP 2: FEATURE MATCHING")
print("="*70)

pycolmap.match_exhaustive(
    database_path=database_path,
    sift_options={
        "max_num_matches": 32768,
    }
)

print("✓ Feature matching complete")

print("\n" + "="*70)
print("STEP 3: INCREMENTAL MAPPING")
print("="*70)

options = pycolmap.IncrementalPipelineOptions()
options.min_num_matches = 15
options.init_min_num_inliers = 50
options.abs_pose_min_num_inliers = 15

maps = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=IMAGE_FOLDER,
    output_path=sparse_path,
    options=options
)

print(f"\n✓ Created {len(maps)} reconstruction(s)")

if len(maps) == 0:
    print("\n✗ No reconstruction created!")
    exit(1)

# Get the best reconstruction
reconstruction = maps[0]

print(f"\nReconstruction stats:")
print(f"  Registered images: {reconstruction.num_reg_images()} / {len(selected_frames)}")
print(f"  3D points: {len(reconstruction.points3D):,}")
print(f"  Cameras: {len(reconstruction.cameras)}")

print("\nRegistered images:")
for img_id, img in reconstruction.images.items():
    print(f"  - {img.name}")

print("\n" + "="*70)
print("STEP 4: EXPORT")
print("="*70)

# Export point cloud
ply_path = os.path.join(OUTPUT_DIR, "sparse_points.ply")
reconstruction.export_PLY(ply_path)
print(f"✓ Saved sparse point cloud to {ply_path}")

# Save reconstruction info
info_path = os.path.join(OUTPUT_DIR, "reconstruction_info.txt")
with open(info_path, 'w') as f:
    f.write(f"SfM Reconstruction Results\n")
    f.write(f"="*50 + "\n\n")
    f.write(f"Total frames used: {len(selected_frames)}\n")
    f.write(f"Registered images: {reconstruction.num_reg_images()}\n")
    f.write(f"3D points: {len(reconstruction.points3D):,}\n")
    f.write(f"Cameras: {len(reconstruction.cameras)}\n\n")
    f.write(f"Registered images:\n")
    for img_id, img in reconstruction.images.items():
        f.write(f"  - {img.name}\n")

print(f"✓ Saved reconstruction info to {info_path}")

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"Results saved to: {OUTPUT_DIR}")
print(f"  - Sparse point cloud: sparse_points.ply")
print(f"  - Info: reconstruction_info.txt")
print(f"  - COLMAP data: {COLMAP_DIR}")
print(f"\nVisualize with:")
print(f"  python visualize_reconstruction.py")
print(f"  (Update SPARSE_DIR to '{sparse_path}/0')")
