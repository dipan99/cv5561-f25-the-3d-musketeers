import os
import cv2
import numpy as np


def load_step1_data(npz_path="./step1_features.npz"):
    """Load features and matches from Step 1"""
    print(f"[load] Loading Step 1 data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    image_paths = data['image_paths']
    
    # Load keypoints and descriptors (stored separately as keypoints_0, descriptors_0, etc.)
    num_images = int(data['num_images'])
    keypoints = [data[f'keypoints_{i}'] for i in range(num_images)]
    descriptors = [data[f'descriptors_{i}'] for i in range(num_images)]
    
    pair_matches_raw = data['pair_matches'].item()
    
    # Convert pair_matches back to usable format
    pair_matches = {}
    for (i, j), match_data in pair_matches_raw.items():
        # match_data is list of (queryIdx, trainIdx, distance)
        matches = [cv2.DMatch(int(q), int(t), float(d)) 
                   for q, t, d in match_data]
        pair_matches[(i, j)] = matches
    
    print(f"[load] Loaded {len(image_paths)} images")
    print(f"[load] Loaded {len(pair_matches)} image pairs")
    
    return image_paths, keypoints, descriptors, pair_matches


def load_images_from_paths(image_paths):
    """Load images from saved paths"""
    images = [cv2.imread(str(p)) for p in image_paths]
    for i, img in enumerate(images):
        if img is None:
            raise ValueError(f"Failed to load image: {image_paths[i]}")
    return images


def estimate_camera_matrix(img_shape):
    """Estimate camera intrinsic matrix (simplified approach)"""
    h, w = img_shape[:2]
    # Assume focal length ~ image width
    focal_length = w
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return K


def compute_essential_matrix(pts1, pts2, K):
    """
    Compute Essential Matrix E from matched 2D points
    
    Steps:
    1. Take 2D coordinates of matched keypoints
    2. Normalize them using K (camera intrinsic matrix)
    3. Estimate E with RANSAC
    
    Args:
        pts1: 2D points in image 1 (Nx2)
        pts2: 2D points in image 2 (Nx2)
        K: Camera intrinsic matrix (3x3)
    
    Returns:
        E: Essential matrix (3x3)
        mask: Inlier mask (N,) - 1 for inliers, 0 for outliers
        pts1_normalized: Normalized points from image 1
        pts2_normalized: Normalized points from image 2
    """
    print(f"\n[Essential Matrix] Computing from {len(pts1)} point correspondences")
    
    # Step 1: Normalize points using K^-1
    print("[Essential Matrix] Normalizing points with K^-1...")
    K_inv = np.linalg.inv(K)
    
    # Convert to homogeneous coordinates
    pts1_homogeneous = np.column_stack([pts1, np.ones(len(pts1))])
    pts2_homogeneous = np.column_stack([pts2, np.ones(len(pts2))])
    
    # Normalize: x_norm = K^-1 * [u, v, 1]^T
    pts1_normalized = (K_inv @ pts1_homogeneous.T).T[:, :2]
    pts2_normalized = (K_inv @ pts2_homogeneous.T).T[:, :2]
    
    print(f"[Essential Matrix] Sample normalized point 1: {pts1_normalized[0]}")
    print(f"[Essential Matrix] Sample normalized point 2: {pts2_normalized[0]}")
    
    # Step 2: Estimate Essential matrix with RANSAC
    print("[Essential Matrix] Estimating E with RANSAC...")
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, 
        method=cv2.RANSAC, 
        prob=0.999,  # Confidence level
        threshold=1.0  # Reprojection error threshold in pixels
    )
    
    # Count inliers
    num_inliers = np.sum(mask)
    num_outliers = len(mask) - num_inliers
    inlier_ratio = num_inliers / len(mask) * 100
    
    print(f"\n[Essential Matrix] Results:")
    print(f"  Total correspondences: {len(mask)}")
    print(f"  Inliers: {num_inliers} ({inlier_ratio:.1f}%)")
    print(f"  Outliers: {num_outliers} ({100-inlier_ratio:.1f}%)")
    print(f"\n[Essential Matrix] E =")
    print(E)
    
    # Verify Essential matrix properties
    # E should have rank 2 and satisfy: det(E) ≈ 0
    det_E = np.linalg.det(E)
    print(f"\n[Essential Matrix] Properties:")
    print(f"  det(E) = {det_E:.6e} (should be ~0)")
    
    # SVD of E should have singular values [σ, σ, 0]
    U, S, Vt = np.linalg.svd(E)
    print(f"  Singular values: {S}")
    print(f"  (should be approximately [σ, σ, 0])")
    
    return E, mask, pts1_normalized, pts2_normalized


def bootstrap_reconstruction(kp1_pts, kp2_pts, matches, K):
    """
    Bootstrap 3D reconstruction from seed pair using Essential matrix
    
    Args:
        kp1_pts: Keypoint coordinates from image 1 (Nx2 array)
        kp2_pts: Keypoint coordinates from image 2 (Nx2 array)
        matches: List of cv2.DMatch objects
        K: Camera intrinsic matrix
    
    Returns:
        E: Essential matrix
        R: Rotation matrix
        t: Translation vector
        points_3d: Initial 3D points
        matched_pts1: 2D points in image 1 (inliers only)
        matched_pts2: 2D points in image 2 (inliers only)
        mask: Inlier mask
    """
    # Extract matched keypoints
    pts1 = np.float32([kp1_pts[m.queryIdx] for m in matches])
    pts2 = np.float32([kp2_pts[m.trainIdx] for m in matches])
    
    print(f"[bootstrap] Using {len(matches)} matches for reconstruction")
    
    # ═══════════════════════════════════════════════════════════
    # STEP 3: Compute Essential Matrix E
    # ═══════════════════════════════════════════════════════════
    E, mask, pts1_norm, pts2_norm = compute_essential_matrix(pts1, pts2, K)
    
    # Filter inliers based on RANSAC mask
    inlier_mask = mask.ravel() == 1
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]
    
    print(f"\n[bootstrap] After RANSAC filtering: {len(pts1_inliers)} inliers")
    
    # Recover pose (R, t) from Essential matrix
    print("\n[bootstrap] Recovering camera pose from E...")
    _, R, t, pose_mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
    
    # Further filter points that are in front of both cameras
    pose_inliers = pose_mask.ravel() > 0
    pts1_final = pts1_inliers[pose_inliers]
    pts2_final = pts2_inliers[pose_inliers]
    
    print(f"[bootstrap] {len(pts1_final)} points in front of both cameras")
    
    # Triangulate 3D points
    # Camera 1 at origin: P1 = K[I|0]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    # Camera 2: P2 = K[R|t]
    P2 = K @ np.hstack([R, t])
    
    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1_final.T, pts2_final.T)
    points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous to 3D
    points_3d = points_3d.T
    
    print(f"[bootstrap] Triangulated {len(points_3d)} 3D points")
    print(f"\n[bootstrap] Final camera pose:")
    print(f"  Rotation matrix R:\n{R}")
    print(f"  Translation vector t:\n{t.T}")
    
    return E, R, t, points_3d, pts1_final, pts2_final, mask


def visualize_3d_points(points_3d, save_path=None):
    """
    Simple visualization of 3D points statistics
    For actual 3D visualization, you'd need matplotlib or Open3D
    """
    print("\n[3D Points Statistics]")
    print(f"Number of points: {len(points_3d)}")
    print(f"X range: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
    print(f"Y range: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
    print(f"Z range: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")
    print(f"Mean position: {points_3d.mean(axis=0)}")
    
    if save_path:
        # Save point cloud to simple text format
        np.savetxt(save_path, points_3d, fmt='%.6f', 
                   header='X Y Z', comments='')
        print(f"[save] 3D points saved to {save_path}")


def main():
    # Configuration - Load available pairs first
    print(f"[main] Bootstrapping 3D reconstruction...")
    
    # Load Step 1 results
    image_paths, keypoints, descriptors, pair_matches = load_step1_data()
    
    # Dynamically select the best matching pair (first available)
    if not pair_matches:
        raise ValueError("No matching pairs found in Step 1 results!")
    
    # Select pair with most matches
    seed_pair = max(pair_matches.keys(), key=lambda k: len(pair_matches[k]))
    print(f"[main] Selected seed pair {seed_pair} with {len(pair_matches[seed_pair])} matches")
    
    # Check if seed pair exists in matches
    if seed_pair not in pair_matches:
        raise ValueError(f"Seed pair {seed_pair} not found in Step 1 results!")
    
    matches = pair_matches[seed_pair]
    print(f"[main] Seed pair {seed_pair} has {len(matches)} matches")
    
    # Load only the seed pair images
    idx1, idx2 = seed_pair
    img1 = cv2.imread(str(image_paths[idx1]))
    img2 = cv2.imread(str(image_paths[idx2]))
    
    if img1 is None or img2 is None:
        raise ValueError(f"Failed to load seed pair images")
    
    print(f"[main] Image shape: {img1.shape}")
    
    # Get keypoints for seed pair
    kp1_pts = keypoints[idx1]
    kp2_pts = keypoints[idx2]
    
    print(f"[main] Image {idx1}: {len(kp1_pts)} keypoints")
    print(f"[main] Image {idx2}: {len(kp2_pts)} keypoints")
    
    # Estimate camera matrix
    print("[main] Estimating camera matrix...")
    K = estimate_camera_matrix(img1.shape)
    print(f"[main] Camera matrix K:\n{K}")
    
    # Bootstrap reconstruction
    print("[main] Bootstrapping 3D reconstruction...")
    E, R, t, points_3d, pts1, pts2, inlier_mask = bootstrap_reconstruction(
        kp1_pts, kp2_pts, matches, K
    )
    
    # Visualize/save results
    visualize_3d_points(points_3d, save_path="./initial_point_cloud.txt")
    
    # Save camera poses and Essential matrix
    print("\n[main] Saving camera poses and Essential matrix...")
    np.savez("./bootstrap_data.npz",
             K=K,
             E=E,
             R=R,
             t=t,
             points_3d=points_3d,
             inlier_mask=inlier_mask,
             seed_pair=seed_pair,
             image_paths=image_paths)
    print("[main] Bootstrap data saved to bootstrap_data.npz")
    
    print("\n" + "="*60)
    print("STEP 3 COMPLETE: Essential Matrix Computation")
    print("="*60)
    print(f"✓ Essential matrix E computed with RANSAC")
    print(f"✓ Inliers identified: {np.sum(inlier_mask)} / {len(inlier_mask)}")
    print(f"✓ Camera pose recovered (R, t)")
    print(f"✓ Initial 3D points triangulated: {len(points_3d)}")
    print("="*60)
    
    print("\n[main] Bootstrap complete!")
    print(f"[main] Initial reconstruction: {len(points_3d)} 3D points")


if __name__ == "__main__":
    main()
