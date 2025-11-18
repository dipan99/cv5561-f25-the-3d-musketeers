import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_bootstrap_data(npz_path="./bootstrap_data.npz"):
    """Load bootstrap data from Step 2"""
    print(f"[load] Loading bootstrap data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    K = data['K']
    E = data['E']
    seed_pair = tuple(data['seed_pair'])
    image_paths = data['image_paths']
    inlier_mask = data['inlier_mask']
    
    print(f"[load] Loaded Essential matrix and camera intrinsics")
    print(f"[load] Seed pair: {seed_pair}")
    
    return K, E, seed_pair, image_paths, inlier_mask


def load_seed_pair_data(image_paths, seed_pair):
    """Load keypoints and matches for seed pair from Step 1"""
    print(f"\n[load] Loading Step 1 features...")
    step1_data = np.load("./step1_features.npz", allow_pickle=True)
    
    num_images = int(step1_data['num_images'])
    pair_matches_raw = step1_data['pair_matches'].item()
    
    idx1, idx2 = seed_pair
    
    # Get keypoints for seed pair (stored separately as keypoints_0, keypoints_1, etc.)
    kp1_pts = step1_data[f'keypoints_{idx1}']
    kp2_pts = step1_data[f'keypoints_{idx2}']
    
    # Get matches for seed pair
    matches = [cv2.DMatch(int(q), int(t), float(d)) 
               for q, t, d in pair_matches_raw[seed_pair]]
    
    # Extract matched points
    pts1 = np.float32([kp1_pts[m.queryIdx] for m in matches])
    pts2 = np.float32([kp2_pts[m.trainIdx] for m in matches])
    
    print(f"[load] Image {idx1}: {len(kp1_pts)} keypoints")
    print(f"[load] Image {idx2}: {len(kp2_pts)} keypoints")
    print(f"[load] Matches: {len(matches)}")
    
    return pts1, pts2, matches


def decompose_essential_matrix(E):
    """
    Decompose Essential matrix into 4 possible (R, t) combinations
    
    Using SVD: E = U * diag(1, 1, 0) * V^T
    Then recover R and t using the formula
    
    Returns:
        List of 4 tuples (R, t)
    """
    print("\n" + "="*60)
    print("STEP 4: Recovering Relative Pose (R, t)")
    print("="*60)
    
    print("\n[decompose] Decomposing Essential matrix E...")
    
    # SVD of E
    U, S, Vt = np.linalg.svd(E)
    
    print(f"[decompose] Singular values: {S}")
    
    # Ensure proper rotation matrices (det = +1)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    
    # W matrix for rotation
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # Four possible solutions
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2].reshape(3, 1)
    t2 = -U[:, 2].reshape(3, 1)
    
    candidates = [
        (R1, t1),
        (R1, t2),
        (R2, t1),
        (R2, t2)
    ]
    
    print(f"[decompose] Generated 4 candidate (R, t) combinations")
    
    return candidates


def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate 3D points from two views
    
    Args:
        P1: Projection matrix for camera 1 (3x4)
        P2: Projection matrix for camera 2 (3x4)
        pts1: 2D points in image 1 (Nx2)
        pts2: 2D points in image 2 (Nx2)
    
    Returns:
        points_3d: 3D points (Nx3)
    """
    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert from homogeneous to 3D: (x, y, z, w) -> (x/w, y/w, z/w)
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T
    
    return points_3d


def check_cheirality(R, t, K, pts1, pts2):
    """
    Check cheirality condition: points should have positive depth in both cameras
    
    Args:
        R, t: Camera pose
        K: Camera intrinsic matrix
        pts1, pts2: 2D point correspondences
    
    Returns:
        num_positive: Number of points with positive depth in both cameras
        points_3d: Triangulated 3D points
    """
    # Camera 0 at origin: P1 = K[I|0]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    # Camera 1: P2 = K[R|t]
    P2 = K @ np.hstack([R, t])
    
    # Triangulate
    points_3d = triangulate_points(P1, P2, pts1, pts2)
    
    # Check depth in camera 1 (just Z coordinate)
    depths_cam1 = points_3d[:, 2]
    
    # Check depth in camera 2: transform points to camera 2 frame
    points_cam2 = (R @ points_3d.T + t).T
    depths_cam2 = points_cam2[:, 2]
    
    # Count points with positive depth in both cameras
    positive_mask = (depths_cam1 > 0) & (depths_cam2 > 0)
    num_positive = np.sum(positive_mask)
    
    return num_positive, points_3d, positive_mask


def recover_pose_from_essential(E, K, pts1, pts2):
    """
    Recover the correct (R, t) from Essential matrix by testing all 4 candidates
    
    Returns:
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        points_3d: Initial triangulated 3D points
        positive_mask: Mask of points with positive depth
    """
    print("\n[recover_pose] Testing 4 candidate (R, t) solutions...")
    
    # Get 4 candidates
    candidates = decompose_essential_matrix(E)
    
    best_num_positive = 0
    best_R = None
    best_t = None
    best_points_3d = None
    best_mask = None
    
    # Test each candidate
    for i, (R, t) in enumerate(candidates):
        num_positive, points_3d, mask = check_cheirality(R, t, K, pts1, pts2)
        
        print(f"[recover_pose] Candidate {i+1}: {num_positive}/{len(pts1)} points with positive depth")
        
        if num_positive > best_num_positive:
            best_num_positive = num_positive
            best_R = R
            best_t = t
            best_points_3d = points_3d
            best_mask = mask
    
    print(f"\n[recover_pose] ✓ Best candidate: {best_num_positive}/{len(pts1)} points with positive depth")
    print(f"[recover_pose] Rotation matrix R:")
    print(best_R)
    print(f"[recover_pose] Translation vector t:")
    print(best_t.T)
    
    return best_R, best_t, best_points_3d, best_mask


def filter_3d_points(points_3d, mask, min_depth=0.1, max_depth=100.0):
    """
    Filter 3D points based on quality criteria
    
    Removes points with:
    - Negative or tiny depth
    - Absurd coordinates (too far away)
    
    Args:
        points_3d: 3D points (Nx3)
        mask: Initial validity mask from cheirality
        min_depth: Minimum depth threshold
        max_depth: Maximum depth threshold
    
    Returns:
        filtered_points: Clean 3D points
        filter_mask: Combined mask of valid points
    """
    print("\n" + "="*60)
    print("STEP 5: Triangulating Clean Initial 3D Point Cloud")
    print("="*60)
    
    print(f"\n[filter] Initial points: {len(points_3d)}")
    print(f"[filter] Points passing cheirality: {np.sum(mask)}")
    
    # Start with cheirality mask
    filter_mask = mask.copy()
    
    # Filter by depth
    depths = points_3d[:, 2]
    depth_mask = (depths > min_depth) & (depths < max_depth)
    filter_mask = filter_mask & depth_mask
    
    print(f"[filter] After depth filter ({min_depth} < z < {max_depth}): {np.sum(filter_mask)}")
    
    # Filter by distance from origin (absurd coordinates)
    distances = np.linalg.norm(points_3d, axis=1)
    distance_mask = distances < max_depth
    filter_mask = filter_mask & distance_mask
    
    print(f"[filter] After distance filter (dist < {max_depth}): {np.sum(filter_mask)}")
    
    # Apply filter
    filtered_points = points_3d[filter_mask]
    
    print(f"\n[filter] ✓ Final clean point cloud: {len(filtered_points)} points")
    
    return filtered_points, filter_mask


def visualize_3d_point_cloud(points_3d, save_path=None):
    """
    Visualize 3D point cloud with matplotlib
    """
    print("\n[visualize] Creating 3D point cloud visualization...")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               c=points_3d[:, 2], cmap='viridis', marker='.', s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Initial 3D Point Cloud ({len(points_3d)} points)')
    
    # Equal aspect ratio
    max_range = np.array([points_3d[:, 0].max()-points_3d[:, 0].min(),
                          points_3d[:, 1].max()-points_3d[:, 1].min(),
                          points_3d[:, 2].max()-points_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max()+points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max()+points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max()+points_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[visualize] Saved visualization to {save_path}")
    
    plt.show()
    print("[visualize] Close the plot window to continue...")


def print_point_cloud_stats(points_3d):
    """Print statistics about the 3D point cloud"""
    print("\n" + "="*60)
    print("3D POINT CLOUD STATISTICS")
    print("="*60)
    print(f"Shape: {points_3d.shape}")
    print(f"Number of points: {len(points_3d)}")
    print(f"\nCoordinate ranges:")
    print(f"  X: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
    print(f"  Y: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
    print(f"  Z: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")
    print(f"\nMean position: [{points_3d[:, 0].mean():.2f}, {points_3d[:, 1].mean():.2f}, {points_3d[:, 2].mean():.2f}]")
    print(f"Std deviation: [{points_3d[:, 0].std():.2f}, {points_3d[:, 1].std():.2f}, {points_3d[:, 2].std():.2f}]")
    print("="*60)


def main():
    # Load data from Step 2
    K, E, seed_pair, image_paths, inlier_mask = load_bootstrap_data()
    
    # Load seed pair features from Step 1
    pts1, pts2, matches = load_seed_pair_data(image_paths, seed_pair)
    
    # Apply inlier mask from RANSAC
    inlier_indices = inlier_mask.ravel() == 1
    pts1_inliers = pts1[inlier_indices]
    pts2_inliers = pts2[inlier_indices]
    
    print(f"\n[main] Using {len(pts1_inliers)} inlier correspondences")
    
    # ═══════════════════════════════════════════════════════════
    # STEP 4: Recover Relative Pose (R, t)
    # ═══════════════════════════════════════════════════════════
    R, t, points_3d_initial, cheirality_mask = recover_pose_from_essential(
        E, K, pts1_inliers, pts2_inliers
    )
    
    # ═══════════════════════════════════════════════════════════
    # STEP 5: Triangulate Clean Initial 3D Point Cloud
    # ═══════════════════════════════════════════════════════════
    points_3d, final_mask = filter_3d_points(
        points_3d_initial, cheirality_mask,
        min_depth=0.1, max_depth=50.0
    )
    
    # Get corresponding 2D observations
    # Map back to original inlier indices
    final_inlier_indices = np.where(inlier_indices)[0][final_mask]
    pts1_final = pts1[final_inlier_indices]
    pts2_final = pts2[final_inlier_indices]
    
    # Store observations: which 2D points correspond to each 3D point
    obs_2d = {
        'image_0': pts1_final,  # 2D points in image 3 (camera 0)
        'image_1': pts2_final,  # 2D points in image 4 (camera 1)
        'image_indices': seed_pair
    }
    
    # Print statistics
    print_point_cloud_stats(points_3d)
    
    # Save results
    print("\n[main] Saving reconstruction data...")
    np.savez("./reconstruction_initial.npz",
             # Camera poses
             R0=np.eye(3),  # Camera 0 (image 3) at origin
             t0=np.zeros((3, 1)),
             R1=R,  # Camera 1 (image 4)
             t1=t,
             K=K,
             # 3D points
             points_3d=points_3d,
             # 2D observations
             obs_2d_img0=pts1_final,
             obs_2d_img1=pts2_final,
             # Metadata
             seed_pair=seed_pair,
             image_paths=image_paths)
    
    # Save point cloud as text file
    np.savetxt("./point_cloud_initial.txt", points_3d, 
               fmt='%.6f', header='X Y Z', comments='')
    
    print("[main] Saved to:")
    print("  - reconstruction_initial.npz (all data)")
    print("  - point_cloud_initial.txt (3D points)")
    
    # Visualize
    print("\n[main] Visualizing 3D point cloud...")
    visualize_3d_point_cloud(points_3d, save_path="./point_cloud_initial.png")
    
    print("\n" + "="*60)
    print("STEPS 4 & 5 COMPLETE!")
    print("="*60)
    print(f"✓ Camera 0 (Image {seed_pair[0]}): R = I, t = 0")
    print(f"✓ Camera 1 (Image {seed_pair[1]}): R and t recovered")
    print(f"✓ Initial 3D point cloud: {len(points_3d)} points")
    print(f"✓ 2D observations stored for both cameras")
    print("\nYou can now:")
    print("  1. Inspect points_3d.shape")
    print("  2. View the 3D plot (point_cloud_initial.png)")
    print("  3. Verify the structure resembles your room")
    print("="*60)


if __name__ == "__main__":
    main()
