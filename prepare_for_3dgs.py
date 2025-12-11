#!/usr/bin/env python3
"""
SfM Pipeline for 3D Gaussian Splatting (Mac - CPU)
Prepares camera poses and sparse points for 3DGS training on GPU machine
"""

import os
import shutil
import numpy as np
import pycolmap
from pathlib import Path

USE_STANFORD = True

if USE_STANFORD:
    DATA_DIR = "./stanford_extracted/seq-01"
    OUTPUT_DIR = "./sfm_for_3dgs_stanford"
    IMAGE_PATTERN = "frame-*.color.png"
    FRAME_SKIP = 10
    MAX_FRAMES = 100
else:
    DATA_DIR = "./images2"
    OUTPUT_DIR = "./sfm_for_3dgs_custom"
    IMAGE_PATTERN = "*.jpeg"
    FRAME_SKIP = 1
    MAX_FRAMES = 100

print("="*70)
print("SfM PIPELINE FOR 3D GAUSSIAN SPLATTING")
print("="*70)
print(f"Dataset: {'Stanford RGBD' if USE_STANFORD else 'Custom Images'}")
print(f"Source: {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_FOLDER = os.path.join(OUTPUT_DIR, "images")
COLMAP_DIR = os.path.join(OUTPUT_DIR, "sparse", "0")
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(COLMAP_DIR, exist_ok=True)

os.makedirs(COLMAP_DIR, exist_ok=True)

# STEP 1: PREPARE IMAGES
print("="*70)
print("STEP 1: PREPARING IMAGES")
print("="*70)

source_images = sorted(Path(DATA_DIR).glob(IMAGE_PATTERN))
selected_images = source_images[::FRAME_SKIP][:MAX_FRAMES]

print(f"Found {len(source_images)} total images")
print(f"Selected {len(selected_images)} images (skip={FRAME_SKIP}, max={MAX_FRAMES})")

if len(selected_images) < 10:
    print("⚠ Warning: Less than 10 images - 3DGS works best with 50+ images")

print("\nCopying images...")
for i, src_path in enumerate(selected_images):
    dst_name = f"frame_{i:04d}{src_path.suffix}"
    dst_path = os.path.join(IMAGE_FOLDER, dst_name)
    shutil.copy(src_path, dst_path)
    if i < 5 or i >= len(selected_images) - 2:
        print(f"  {src_path.name} -> {dst_name}")
    elif i == 5:
        print(f"  ... ({len(selected_images) - 7} more) ...")

print(f"\n✓ Copied {len(selected_images)} images to {IMAGE_FOLDER}")

print(f"\n✓ Copied {len(selected_images)} images to {IMAGE_FOLDER}")

# STEP 2: STRUCTURE FROM MOTION
print("\n" + "="*70)
print("STEP 2: RUNNING STRUCTURE FROM MOTION")
print("="*70)

database_path = os.path.join(OUTPUT_DIR, "database.db")

print("\n2a. Feature Extraction...")
pycolmap.extract_features(
    database_path=database_path,
    image_path=IMAGE_FOLDER,
    sift_options={
        "max_num_features": 8192,
    }
)
print("    ✓ Features extracted")

print("\n2b. Feature Matching...")
pycolmap.match_exhaustive(
    database_path=database_path,
    sift_options={
        "max_num_matches": 32768,
    }
)
print("    ✓ Features matched")

print("\n2c. Incremental Mapping (SfM)...")
options = pycolmap.IncrementalPipelineOptions()
options.min_num_matches = 15
options.init_min_num_inliers = 50
options.abs_pose_min_num_inliers = 15
options.ba_refine_focal_length = True
options.ba_refine_principal_point = True
options.ba_refine_extra_params = True

maps = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=IMAGE_FOLDER,
    output_path=os.path.join(OUTPUT_DIR, "sparse"),
    options=options
)

if len(maps) == 0:
    print("\n✗ ERROR: SfM failed - no reconstruction created!")
    print("  Possible reasons:")
    print("  - Images don't have enough overlap")
    print("  - Too few feature matches")
    print("  - Scene is too repetitive/textureless")
    exit(1)

reconstruction = maps[0]

print(f"\n    ✓ SfM complete!")
print(f"      - Registered images: {reconstruction.num_reg_images()} / {len(selected_images)}")
print(f"      - 3D points: {len(reconstruction.points3D):,}")
print(f"      - Cameras: {len(reconstruction.cameras)}")

if reconstruction.num_reg_images() < len(selected_images) * 0.5:
    print(f"\n    ⚠ Warning: Only {reconstruction.num_reg_images()}/{len(selected_images)} images registered")
    print("      Consider retaking images with more overlap")

# STEP 3: EXPORT RESULTS
print("\n" + "="*70)
print("STEP 3: EXPORTING FOR 3D GAUSSIAN SPLATTING")
print("="*70)

required_files = ['cameras.bin', 'images.bin', 'points3D.bin']
missing_files = []

for fname in required_files:
    fpath = os.path.join(COLMAP_DIR, fname)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  ✓ {fname} ({size_mb:.2f} MB)")
    else:
        missing_files.append(fname)
        print(f"  ✗ {fname} (MISSING)")

if missing_files:
    print(f"\n✗ ERROR: Missing required files: {missing_files}")
    exit(1)

ply_path = os.path.join(OUTPUT_DIR, "sparse_points.ply")
reconstruction.export_PLY(ply_path)
print(f"\n  ✓ Sparse point cloud: sparse_points.ply ({len(reconstruction.points3D):,} points)")

info_path = os.path.join(OUTPUT_DIR, "README.txt")
with open(info_path, 'w') as f:
    f.write("SfM OUTPUT FOR 3D GAUSSIAN SPLATTING\n")
    f.write("="*70 + "\n\n")
    f.write(f"Dataset: {'Stanford RGBD' if USE_STANFORD else 'Custom Images'}\n")
    f.write(f"Total images processed: {len(selected_images)}\n")
    f.write(f"Registered images: {reconstruction.num_reg_images()}\n")
    f.write(f"Sparse 3D points: {len(reconstruction.points3D):,}\n")
    f.write(f"Cameras: {len(reconstruction.cameras)}\n\n")
    
    f.write("DIRECTORY STRUCTURE:\n")
    f.write("-"*70 + "\n")
    f.write("images/           - Input images for training\n")
    f.write("sparse/0/         - COLMAP reconstruction (cameras, poses, points)\n")
    f.write("  ├─ cameras.bin  - Camera intrinsics\n")
    f.write("  ├─ images.bin   - Camera poses\n")
    f.write("  └─ points3D.bin - Sparse 3D points\n")
    f.write("database.db       - COLMAP feature database\n")
    f.write("sparse_points.ply - Visualization of sparse points\n\n")
    
    f.write("NEXT STEPS (on GPU machine):\n")
    f.write("-"*70 + "\n")
    f.write("1. Transfer this entire folder to your MSI (Windows/Linux)\n\n")
    f.write("2. Install 3D Gaussian Splatting:\n")
    f.write("   git clone https://github.com/graphdeco-inria/gaussian-splatting.git\n")
    f.write("   cd gaussian-splatting\n")
    f.write("   conda env create -f environment.yml\n")
    f.write("   conda activate gaussian_splatting\n\n")
    f.write("3. Train 3DGS:\n")
    f.write(f"   python train.py -s /path/to/{OUTPUT_DIR} -m output/model\n\n")
    f.write("4. Render novel views:\n")
    f.write("   python render.py -m output/model\n\n")
    f.write("Alternative: Use Nerfstudio (easier installation):\n")
    f.write("   pip install nerfstudio\n")
    f.write(f"   ns-train splatfacto --data /path/to/{OUTPUT_DIR}\n\n")

print(f"  ✓ Instructions: README.txt")

# STEP 4: SUMMARY
print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nContents:")
print(f"  - {reconstruction.num_reg_images()} images in images/")
print(f"  - COLMAP reconstruction in sparse/0/")
print(f"  - {len(reconstruction.points3D):,} sparse points")
print(f"\nNext steps:")
print(f"  1. Transfer '{OUTPUT_DIR}' folder to your MSI")
print(f"  2. Install 3D Gaussian Splatting (see README.txt)")
print(f"  3. Train on GPU: ~10-30 minutes")
print(f"  4. Render novel views!")
print("\nRegistered images:")
for img_id, img in list(reconstruction.images.items())[:10]:
    print(f"  - {img.name}")
if reconstruction.num_reg_images() > 10:
    print(f"  ... and {reconstruction.num_reg_images() - 10} more")

print("\n" + "="*70)
print("Ready for 3DGS training on GPU!")
print("="*70)
