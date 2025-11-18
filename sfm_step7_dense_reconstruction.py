import cv2
import numpy as np
import pickle
from sfm_step4_reconstruction_state import ReconstructionState
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_images(image_paths):
    """Load images from paths"""
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        images.append(img)
    return images


def rectify_stereo_pair(img1, img2, K, R1, t1, R2, t2):
    """
    Rectify a stereo pair for dense matching
    
    Args:
        img1, img2: Images
        K: Camera intrinsic matrix
        R1, t1: Pose of camera 1
        R2, t2: Pose of camera 2
    
    Returns:
        img1_rect, img2_rect: Rectified images
        Q: Disparity-to-depth mapping matrix
        roi1, roi2: Valid regions of interest
    """
    h, w = img1.shape[:2]
    
    # Compute relative pose from camera 1 to camera 2
    # R_rel = R2 @ R1^T
    # t_rel = t2 - R_rel @ t1
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    
    # Stereo rectification
    R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, None,  # Camera 1 intrinsics, no distortion
        K, None,  # Camera 2 intrinsics, no distortion
        (w, h),
        R_rel,
        t_rel,
        alpha=0,  # 0 = crop to valid pixels only
        newImageSize=(w, h)
    )
    
    # Compute rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K, None, R1_rect, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, None, R2_rect, P2, (w, h), cv2.CV_32FC1)
    
    # Remap images
    img1_rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    return img1_rect, img2_rect, Q, roi1, roi2


def compute_dense_disparity(img1_rect, img2_rect, max_disparity=256):
    """
    Compute dense disparity map using Semi-Global Block Matching
    
    Args:
        img1_rect, img2_rect: Rectified stereo pair
        max_disparity: Maximum disparity to search
    
    Returns:
        disparity: Disparity map (normalized)
    """
    print(f"  [Dense] Computing disparity map...")
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)
    
    # Create StereoSGBM matcher with more lenient parameters
    window_size = 3
    min_disp = 0
    num_disp = max_disparity - min_disp
    
    # Ensure num_disp is divisible by 16
    num_disp = ((num_disp + 15) // 16) * 16
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=2,
        uniquenessRatio=5,  # More lenient
        speckleWindowSize=50,  # Smaller to keep more
        speckleRange=16,  # More lenient
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity
    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    
    # Filter invalid disparities more leniently
    disparity[disparity < 0] = 0
    
    valid_pixels = np.sum(disparity > 0)
    print(f"  [Dense] Disparity range: [{disparity[disparity > 0].min():.1f}, {disparity.max():.1f}]")
    print(f"  [Dense] Valid disparity pixels: {valid_pixels}/{disparity.size} ({valid_pixels/disparity.size*100:.1f}%)")
    
    return disparity


def disparity_to_point_cloud(disparity, img, Q, roi):
    """
    Convert disparity map to 3D point cloud
    
    Args:
        disparity: Disparity map
        img: Original color image
        Q: Disparity-to-depth reprojection matrix
        roi: Region of interest (x, y, w, h)
    
    Returns:
        points_3d: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors
    """
    print(f"  [Dense] Converting disparity to 3D points...")
    
    # Reproject to 3D
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    
    # Get colors from image
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop to ROI if specified
    if roi is not None and roi[2] > 0 and roi[3] > 0:
        x, y, w, h = roi
        points_3d = points_3d[y:y+h, x:x+w]
        colors = colors[y:y+h, x:x+w]
        disparity_roi = disparity[y:y+h, x:x+w]
    else:
        disparity_roi = disparity
    
    # Create mask for valid points (disparity > 1 to avoid infinity)
    mask = disparity_roi > 1.0
    
    print(f"  [Dense] Valid disparity pixels: {np.sum(mask)}/{mask.size}")
    
    # Filter points
    points_3d = points_3d[mask]
    colors = colors[mask]
    
    if len(points_3d) == 0:
        print(f"  [Dense] No points after disparity filter")
        return points_3d, colors
    
    print(f"  [Dense] Point cloud Z range before filter: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")
    print(f"  [Dense] Point cloud magnitude range: [{np.linalg.norm(points_3d, axis=1).min():.2f}, {np.linalg.norm(points_3d, axis=1).max():.2f}]")
    
    # More lenient depth filtering
    # Remove points with negative or very large Z (bad triangulation)
    z_valid = (points_3d[:, 2] > -100) & (points_3d[:, 2] < 100)
    
    # Remove points that are extremely far from origin (outliers)
    distance = np.linalg.norm(points_3d, axis=1)
    dist_valid = distance < 100
    
    # Combine filters
    valid = z_valid & dist_valid
    
    points_3d = points_3d[valid]
    colors = colors[valid]
    
    print(f"  [Dense] Generated {len(points_3d)} dense 3D points after filtering")
    
    return points_3d, colors


def transform_points_to_world(points_cam, R, t):
    """
    Transform points from camera coordinates to world coordinates
    
    Args:
        points_cam: (N, 3) points in camera frame
        R: Camera rotation matrix
        t: Camera translation vector
    
    Returns:
        points_world: (N, 3) points in world frame
    """
    # World point = R^T @ (cam_point - t)
    points_world = (R.T @ (points_cam.T - t)).T
    return points_world


def dense_reconstruct_from_pair(img1, img2, K, R1, t1, R2, t2, pair_name=""):
    """
    Perform dense reconstruction from a stereo pair
    
    Returns:
        points_3d: Dense 3D points in world coordinates
        colors: RGB colors for points
    """
    print(f"\n[Dense Pair {pair_name}] Processing stereo pair...")
    
    # Rectify stereo pair
    print(f"  [Dense] Rectifying stereo pair...")
    img1_rect, img2_rect, Q, roi1, roi2 = rectify_stereo_pair(
        img1, img2, K, R1, t1, R2, t2
    )
    
    # Compute disparity
    disparity = compute_dense_disparity(img1_rect, img2_rect, max_disparity=96)
    
    # Convert to 3D points in camera 1 frame
    points_cam, colors = disparity_to_point_cloud(disparity, img1_rect, Q, roi1)
    
    if len(points_cam) == 0:
        print(f"  [Dense] Warning: No valid points generated")
        return np.array([]), np.array([])
    
    # Transform to world coordinates using camera 1 pose
    points_world = transform_points_to_world(points_cam, R1, t1)
    
    print(f"  [Dense] ✓ Generated {len(points_world)} points from pair {pair_name}")
    
    return points_world, colors


def filter_outlier_points(points, colors, nb_neighbors=20, std_ratio=2.0):
    """
    Statistical outlier removal
    
    Args:
        points: (N, 3) point cloud
        colors: (N, 3) colors
        nb_neighbors: Number of neighbors to consider
        std_ratio: Standard deviation ratio threshold
    
    Returns:
        filtered_points, filtered_colors
    """
    print(f"\n[Filter] Removing outliers from {len(points)} points...")
    
    if len(points) < nb_neighbors:
        return points, colors
    
    # For each point, compute distance to k nearest neighbors
    from scipy.spatial import cKDTree
    
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=nb_neighbors+1)
    
    # Compute mean distance to neighbors (excluding self)
    mean_distances = distances[:, 1:].mean(axis=1)
    
    # Compute global statistics
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    
    # Filter points
    threshold = global_mean + std_ratio * global_std
    mask = mean_distances < threshold
    
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    print(f"[Filter] Kept {len(filtered_points)}/{len(points)} points ({len(filtered_points)/len(points)*100:.1f}%)")
    
    return filtered_points, filtered_colors


def downsample_point_cloud(points, colors, voxel_size=0.05):
    """
    Downsample point cloud using voxel grid
    
    Args:
        points: (N, 3) points
        colors: (N, 3) colors
        voxel_size: Voxel size for downsampling
    
    Returns:
        downsampled_points, downsampled_colors
    """
    print(f"\n[Downsample] Downsampling {len(points)} points with voxel size {voxel_size}...")
    
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Create unique voxel keys
    voxel_keys = [tuple(idx) for idx in voxel_indices]
    
    # Group points by voxel
    from collections import defaultdict
    voxel_dict = defaultdict(list)
    for i, key in enumerate(voxel_keys):
        voxel_dict[key].append(i)
    
    # Average points and colors in each voxel
    downsampled_points = []
    downsampled_colors = []
    
    for indices in voxel_dict.values():
        downsampled_points.append(points[indices].mean(axis=0))
        downsampled_colors.append(colors[indices].mean(axis=0))
    
    downsampled_points = np.array(downsampled_points)
    downsampled_colors = np.array(downsampled_colors)
    
    print(f"[Downsample] Reduced to {len(downsampled_points)} points")
    
    return downsampled_points, downsampled_colors


def dense_reconstruction(state, images):
    """
    Perform dense reconstruction using all registered camera pairs
    
    Args:
        state: ReconstructionState
        images: List of all images
    
    Returns:
        dense_points: Dense 3D point cloud
        dense_colors: Colors for each point
    """
    print("\n" + "="*70)
    print("STEP 7: DENSE MULTI-VIEW STEREO RECONSTRUCTION")
    print("="*70)
    
    registered_cameras = sorted(state.get_registered_cameras())
    
    print(f"\n[Dense] Processing {len(registered_cameras)} cameras")
    print(f"[Dense] Will compute {len(registered_cameras)-1} stereo pairs")
    
    all_points = []
    all_colors = []
    
    # Process consecutive camera pairs
    for i in range(len(registered_cameras) - 1):
        cam_idx1 = registered_cameras[i]
        cam_idx2 = registered_cameras[i + 1]
        
        cam1 = state.get_camera(cam_idx1)
        cam2 = state.get_camera(cam_idx2)
        
        img1 = images[cam_idx1]
        img2 = images[cam_idx2]
        
        # Dense reconstruction for this pair
        points, colors = dense_reconstruct_from_pair(
            img1, img2,
            state.K,
            cam1['R'], cam1['t'],
            cam2['R'], cam2['t'],
            pair_name=f"{cam_idx1}-{cam_idx2}"
        )
        
        if len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
    
    if not all_points:
        print("\n[Dense] ERROR: No dense points generated!")
        return None, None
    
    # Combine all points
    print(f"\n[Dense] Combining point clouds...")
    dense_points = np.vstack(all_points)
    dense_colors = np.vstack(all_colors)
    
    print(f"[Dense] Total raw points: {len(dense_points)}")
    
    # Filter outliers
    dense_points, dense_colors = filter_outlier_points(
        dense_points, dense_colors, nb_neighbors=20, std_ratio=2.0
    )
    
    # Downsample
    dense_points, dense_colors = downsample_point_cloud(
        dense_points, dense_colors, voxel_size=0.02
    )
    
    print(f"\n[Dense] ✓ Final dense point cloud: {len(dense_points)} points")
    
    return dense_points, dense_colors


def export_colored_ply(points, colors, filepath):
    """
    Export colored point cloud to PLY format
    
    Args:
        points: (N, 3) 3D points
        colors: (N, 3) RGB colors (0-255)
        filepath: Output file path
    """
    print(f"\n[Export] Saving dense point cloud to {filepath}...")
    
    # Ensure colors are in 0-255 range
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    
    with open(filepath, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices with colors
        for point, color in zip(points, colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
    
    print(f"[Export] ✓ Saved {len(points)} colored points")


def visualize_dense_cloud(points, colors, save_path=None):
    """
    Visualize dense point cloud with colors
    """
    print("\n[Visualize] Creating dense point cloud visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors for matplotlib (0-1 range)
    if colors.max() > 1.0:
        colors_norm = colors / 255.0
    else:
        colors_norm = colors
    
    # Subsample for visualization if too many points
    if len(points) > 100000:
        indices = np.random.choice(len(points), 100000, replace=False)
        points_viz = points[indices]
        colors_viz = colors_norm[indices]
        print(f"[Visualize] Subsampling to 100k points for visualization")
    else:
        points_viz = points
        colors_viz = colors_norm
    
    ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2],
               c=colors_viz, marker='.', s=1, alpha=0.8)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title(f'Dense 3D Reconstruction ({len(points)} points)', 
                 fontsize=16, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualize] Saved visualization to {save_path}")
    
    plt.show()
    print("[Visualize] Close the plot window to continue...")


def main():
    print("="*70)
    print("DENSE RECONSTRUCTION: Multi-View Stereo")
    print("="*70)
    
    # Load reconstruction state
    print("\n[main] Loading reconstruction state...")
    state = ReconstructionState.load("./reconstruction_final.pkl")
    
    # Load images
    print("\n[main] Loading images...")
    images = load_images(state.image_paths)
    print(f"[main] Loaded {len(images)} images")
    
    # Perform dense reconstruction
    dense_points, dense_colors = dense_reconstruction(state, images)
    
    if dense_points is None:
        print("\n[main] Dense reconstruction failed!")
        return
    
    # Save dense point cloud
    print("\n[main] Saving dense reconstruction...")
    np.savez("./dense_reconstruction.npz",
             points=dense_points,
             colors=dense_colors)
    
    # Export to PLY
    export_colored_ply(dense_points, dense_colors, "./dense_pointcloud.ply")
    
    # Visualize
    print("\n[main] Visualizing dense point cloud...")
    visualize_dense_cloud(dense_points, dense_colors, 
                         save_path="./dense_reconstruction.png")
    
    print("\n" + "="*70)
    print("DENSE RECONSTRUCTION COMPLETE!")
    print("="*70)
    print(f"\n✓ Dense point cloud: {len(dense_points)} points")
    print(f"\n📊 Generated files:")
    print("  • dense_reconstruction.npz - Dense point cloud data")
    print("  • dense_pointcloud.ply - Colored PLY (open in MeshLab/CloudCompare)")
    print("  • dense_reconstruction.png - Visualization")
    print("\n💡 Next: Open dense_pointcloud.ply in MeshLab or CloudCompare")
    print("         to see your full 3D room reconstruction!")
    print("="*70)


if __name__ == "__main__":
    main()
