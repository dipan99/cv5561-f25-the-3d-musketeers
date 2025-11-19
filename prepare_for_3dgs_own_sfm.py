#!/usr/bin/env python3
"""
Custom SfM Pipeline for 3D Gaussian Splatting (Mac - CPU)
Implements SfM from scratch without COLMAP, then exports for 3DGS
"""

import os
import shutil
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import struct

USE_STANFORD = True

if USE_STANFORD:
    DATA_DIR = "./stanford_extracted/seq-01"
    OUTPUT_DIR = "./sfm_for_3dgs_custom_sfm"
    IMAGE_PATTERN = "frame-*.color.png"
    FRAME_SKIP = 10
    MAX_FRAMES = 50
else:
    DATA_DIR = "./images2"
    OUTPUT_DIR = "./sfm_for_3dgs_custom_sfm"
    IMAGE_PATTERN = "*.jpeg"
    FRAME_SKIP = 1
    MAX_FRAMES = 20

class CustomSfM:
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.images = {}
        self.features = {}
        self.matches = {}
        self.cameras = {}
        self.poses = {}
        self.points3D = []
        self.point_colors = []
        self.registered_images = set()
        
    def load_images(self, image_files):
        print("Loading images...")
        for i, img_file in enumerate(image_files):
            img = cv2.imread(str(img_file))
            self.images[img_file.name] = img
            if i < 3:
                print(f"  Loaded {img_file.name}: {img.shape[1]}x{img.shape[0]}")
        print(f"  ... ({len(image_files)} total)")
        
    def detect_features(self):
        print("\nDetecting SIFT features...")
        sift = cv2.SIFT_create(nfeatures=8192)
        
        for img_name, img in self.images.items():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(gray, None)
            self.features[img_name] = (kp, desc)
            print(f"  {img_name}: {len(kp)} features")
            
    def match_features(self):
        print("\nMatching features...")
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        image_names = list(self.images.keys())
        count = 0
        
        for i in range(len(image_names)):
            for j in range(i + 1, len(image_names)):
                img1_name = image_names[i]
                img2_name = image_names[j]
                
                kp1, desc1 = self.features[img1_name]
                kp2, desc2 = self.features[img2_name]
                
                matches = bf.knnMatch(desc1, desc2, k=2)
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) >= 50:
                    self.matches[(img1_name, img2_name)] = good_matches
                    count += 1
                    if count <= 5:
                        print(f"  {img1_name} <-> {img2_name}: {len(good_matches)} matches")
        
        print(f"  ... ({len(self.matches)} pairs with >=50 matches)")
        
    def estimate_camera_matrix(self, img_name):
        img = self.images[img_name]
        h, w = img.shape[:2]
        
        focal_length = max(w, h) * 0.8
        cx = w / 2.0
        cy = h / 2.0
        
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        
        return K
    
    def bootstrap_reconstruction(self):
        print("\nBootstrapping reconstruction...")
        
        best_pair = max(self.matches.items(), key=lambda x: len(x[1]))
        (img1_name, img2_name), matches = best_pair
        
        print(f"  Initial pair: {img1_name} <-> {img2_name} ({len(matches)} matches)")
        
        kp1, _ = self.features[img1_name]
        kp2, _ = self.features[img2_name]
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        K1 = self.estimate_camera_matrix(img1_name)
        K2 = self.estimate_camera_matrix(img2_name)
        
        self.cameras[img1_name] = K1
        self.cameras[img2_name] = K2
        
        E, mask = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K1, mask=mask)
        
        self.poses[img1_name] = (np.eye(3), np.zeros(3))
        self.registered_images.add(img1_name)
        
        self.poses[img2_name] = (R, t.flatten())
        self.registered_images.add(img2_name)
        
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]
        
        P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K2 @ np.hstack([R, t])
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        valid = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 100)
        self.points3D = points_3d[valid].tolist()
        
        img1 = self.images[img1_name]
        for pt in pts1_inliers[valid]:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                color = img1[y, x][[2, 1, 0]]
                self.point_colors.append(color.tolist())
            else:
                self.point_colors.append([128, 128, 128])
        
        print(f"  Triangulated {len(self.points3D)} points")
        
    def add_image(self, img_name):
        print(f"\nAdding image: {img_name}")
        
        kp_new, desc_new = self.features[img_name]
        K_new = self.estimate_camera_matrix(img_name)
        self.cameras[img_name] = K_new
        
        registered = list(self.registered_images)[0]
        
        pair_key = None
        if (registered, img_name) in self.matches:
            pair_key = (registered, img_name)
        elif (img_name, registered) in self.matches:
            pair_key = (img_name, registered)
        
        if pair_key is None:
            print(f"  No matches with registered images, skipping")
            return False
        
        # For simplicity, estimate pose from essential matrix like bootstrap
        # In real SfM, you'd use PnP with existing 3D points
        kp_reg, _ = self.features[registered]
        matches = self.matches[pair_key]
        
        if pair_key[0] == img_name:
            pts_new = np.float32([kp_new[m.queryIdx].pt for m in matches])
            pts_reg = np.float32([kp_reg[m.trainIdx].pt for m in matches])
        else:
            pts_new = np.float32([kp_new[m.trainIdx].pt for m in matches])
            pts_reg = np.float32([kp_reg[m.queryIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts_reg, pts_new, K_new, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, pts_reg, pts_new, K_new, mask=mask)
        
        self.poses[img_name] = (R, t.flatten())
        self.registered_images.add(img_name)
        
        print(f"  Registered successfully")
        return True
        
    def run_reconstruction(self):
        print("\n" + "="*70)
        print("RUNNING CUSTOM SfM RECONSTRUCTION")
        print("="*70)
        
        self.bootstrap_reconstruction()
        
        unregistered = set(self.images.keys()) - self.registered_images
        
        for img_name in sorted(unregistered):
            self.add_image(img_name)
        
        print(f"\n✓ Reconstruction complete:")
        print(f"  - Registered images: {len(self.registered_images)} / {len(self.images)}")
        print(f"  - 3D points: {len(self.points3D)}")
        
    def export_colmap_format(self, output_dir):
        print("\nExporting to COLMAP format...")
        
        sparse_dir = os.path.join(output_dir, "sparse", "0")
        os.makedirs(sparse_dir, exist_ok=True)
        
        cameras_file = os.path.join(sparse_dir, "cameras.bin")
        with open(cameras_file, 'wb') as f:
            f.write(struct.pack('Q', len(self.cameras)))
            
            for camera_id, (img_name, K) in enumerate(self.cameras.items(), 1):
                f.write(struct.pack('i', camera_id))
                f.write(struct.pack('i', 1))
                
                img = self.images[img_name]
                f.write(struct.pack('Q', img.shape[1]))
                f.write(struct.pack('Q', img.shape[0]))
                
                f.write(struct.pack('d', K[0, 0]))
                f.write(struct.pack('d', K[0, 2]))
                f.write(struct.pack('d', K[1, 2]))
        
        print(f"  ✓ cameras.bin")
        
        images_file = os.path.join(sparse_dir, "images.bin")
        with open(images_file, 'wb') as f:
            f.write(struct.pack('Q', len(self.registered_images)))
            
            for image_id, img_name in enumerate(sorted(self.registered_images), 1):
                R, t = self.poses[img_name]
                
                qvec = Rotation.from_matrix(R).as_quat()
                qvec = np.roll(qvec, 1)
                
                f.write(struct.pack('i', image_id))
                f.write(struct.pack('dddd', *qvec))
                f.write(struct.pack('ddd', *t))
                f.write(struct.pack('i', 1))
                
                name_bytes = img_name.encode('utf-8')
                f.write(struct.pack('Q', len(name_bytes)))
                f.write(name_bytes)
                
                f.write(struct.pack('Q', 0))
        
        print(f"  ✓ images.bin")
        
        points_file = os.path.join(sparse_dir, "points3D.bin")
        with open(points_file, 'wb') as f:
            f.write(struct.pack('Q', len(self.points3D)))
            
            for point_id, (pt3d, color) in enumerate(zip(self.points3D, self.point_colors), 1):
                f.write(struct.pack('Q', point_id))
                f.write(struct.pack('ddd', *pt3d))
                f.write(struct.pack('BBB', *color))
                f.write(struct.pack('d', 1.0))
                f.write(struct.pack('Q', 0))
        
        print(f"  ✓ points3D.bin")

# MAIN PIPELINE
print("="*70)
# MAIN PIPELINE
print("="*70)
print("CUSTOM SfM PIPELINE FOR 3D GAUSSIAN SPLATTING")
print("="*70)
print(f"Dataset: {'Stanford RGBD' if USE_STANFORD else 'Custom Images'}")
print(f"Source: {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_FOLDER = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

IMAGE_FOLDER = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# STEP 1: PREPARING IMAGES
print("="*70)
print("STEP 1: PREPARING IMAGES")
print("="*70)

source_images = sorted(Path(DATA_DIR).glob(IMAGE_PATTERN))
selected_images = source_images[::FRAME_SKIP][:MAX_FRAMES]

print(f"Selected {len(selected_images)} images")

for i, src_path in enumerate(selected_images):
    dst_name = f"frame_{i:04d}{src_path.suffix}"
    dst_path = os.path.join(IMAGE_FOLDER, dst_name)
    shutil.copy(src_path, dst_path)

    shutil.copy(src_path, dst_path)

# STEP 2: RUN CUSTOM SfM
sfm = CustomSfM(IMAGE_FOLDER)

image_files = sorted(Path(IMAGE_FOLDER).glob(f"*{selected_images[0].suffix}"))
sfm.load_images(image_files)
sfm.detect_features()
sfm.match_features()

if len(sfm.matches) == 0:
    print("\n✗ ERROR: No image pairs with sufficient matches!")
    exit(1)

sfm.run_reconstruction()

# STEP 3: EXPORT RESULTS
sfm.export_colmap_format(OUTPUT_DIR)

ply_path = os.path.join(OUTPUT_DIR, "sparse_points.ply")
with open(ply_path, 'w') as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(sfm.points3D)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    
    for pt, color in zip(sfm.points3D, sfm.point_colors):
        f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]}\n")

print(f"\n✓ Sparse point cloud: sparse_points.ply")

readme_path = os.path.join(OUTPUT_DIR, "README.txt")
with open(readme_path, 'w') as f:
    f.write("CUSTOM SfM OUTPUT FOR 3D GAUSSIAN SPLATTING\n")
    f.write("="*70 + "\n\n")
    f.write("This reconstruction was created using CUSTOM SfM implementation\n")
    f.write("(not COLMAP), but exported in COLMAP format for 3DGS compatibility.\n\n")
    f.write(f"Registered images: {len(sfm.registered_images)} / {len(selected_images)}\n")
    f.write(f"Sparse 3D points: {len(sfm.points3D)}\n\n")
    f.write("Transfer to GPU machine and train with:\n")
    f.write(f"  python train.py -s /path/to/{OUTPUT_DIR} -m output/model\n")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nOutput: {OUTPUT_DIR}")
print(f"  - {len(sfm.registered_images)} registered images")
print(f"  - {len(sfm.points3D)} sparse points")
print(f"\nReady for 3DGS training on GPU!")
print("="*70)
