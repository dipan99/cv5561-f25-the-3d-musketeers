import cv2
import numpy as np
import pickle
from sfm_step4_reconstruction_state import ReconstructionState


def load_step1_features():
    """Load all features and matches from Step 1"""
    print("[load] Loading Step 1 features...")
    data = np.load("./step1_features.npz", allow_pickle=True)
    
    image_paths = data['image_paths']
    
    # Load keypoints and descriptors (stored separately as keypoints_0, descriptors_0, etc.)
    num_images = int(data['num_images'])
    keypoints = [data[f'keypoints_{i}'] for i in range(num_images)]
    descriptors = [data[f'descriptors_{i}'] for i in range(num_images)]
    
    pair_matches_raw = data['pair_matches'].item()
    
    # Convert matches back to cv2.DMatch format
    pair_matches = {}
    for (i, j), match_data in pair_matches_raw.items():
        matches = [cv2.DMatch(int(q), int(t), float(d)) 
                   for q, t, d in match_data]
        pair_matches[(i, j)] = matches
    
    return image_paths, keypoints, descriptors, pair_matches


def find_2d_3d_correspondences(state, new_img_idx, keypoints, descriptors, pair_matches):
    """
    Find 2D-3D correspondences between new image and existing 3D points
    
    For each registered camera:
    - Get matches between new image and registered image
    - For each match, check if the point in registered image corresponds to a 3D point
    - Build list of (2D point in new image -> 3D point)
    
    Args:
        state: ReconstructionState
        new_img_idx: Index of new image to register
        keypoints: All keypoints from Step 1
        descriptors: All descriptors from Step 1
        pair_matches: All matches from Step 1
    
    Returns:
        points_2d: 2D points in new image (Nx2)
        points_3d: Corresponding 3D points (Nx3)
        point_3d_indices: Indices of 3D points in state.points_3d
    """
    registered_cameras = state.get_registered_cameras()
    
    # Map: keypoint index in registered image -> 3D point index
    # We need to build this from the tracks
    kp_to_3d = {}  # {(img_idx, kp_idx): point_3d_idx}
    
    for point_3d_idx, track in enumerate(state.tracks):
        for img_idx in track:
            # Find which keypoint this corresponds to
            # The track stores the 2D coordinate, we need to find the keypoint index
            # For now, we'll store by coordinate and match later
            kp_to_3d[(img_idx, tuple(track[img_idx]))] = point_3d_idx
    
    correspondences = []  # List of (2d_point, 3d_point, 3d_idx)
    
    print(f"\n[2D-3D] Finding correspondences for image {new_img_idx}...")
    
    for reg_img_idx in registered_cameras:
        # Get matches between new image and registered image
        if new_img_idx < reg_img_idx:
            pair_key = (new_img_idx, reg_img_idx)
            matches = pair_matches.get(pair_key, [])
            is_flipped = False
        else:
            pair_key = (reg_img_idx, new_img_idx)
            matches = pair_matches.get(pair_key, [])
            is_flipped = True
        
        if not matches:
            continue
        
        print(f"  Checking {len(matches)} matches with registered image {reg_img_idx}")
        
        # For each match, check if registered image point corresponds to 3D point
        for match in matches:
            if is_flipped:
                # Match is (reg_img, new_img)
                reg_kp_idx = match.queryIdx
                new_kp_idx = match.trainIdx
            else:
                # Match is (new_img, reg_img)
                new_kp_idx = match.queryIdx
                reg_kp_idx = match.trainIdx
            
            # Get 2D point in new image
            pt_2d_new = keypoints[new_img_idx][new_kp_idx]
            
            # Get 2D point in registered image
            pt_2d_reg = keypoints[reg_img_idx][reg_kp_idx]
            
            # Check if this registered image point corresponds to a 3D point
            # We need to find it in the tracks
            found_3d = False
            for point_3d_idx, track in enumerate(state.tracks):
                if reg_img_idx in track:
                    # Check if the 2D point matches (within small tolerance)
                    tracked_pt = track[reg_img_idx]
                    if np.linalg.norm(tracked_pt - pt_2d_reg) < 1.0:  # 1 pixel tolerance
                        # Found a correspondence!
                        correspondences.append((pt_2d_new, state.points_3d[point_3d_idx], point_3d_idx))
                        found_3d = True
                        break
    
    if not correspondences:
        print(f"  [2D-3D] No correspondences found!")
        return None, None, None
    
    # Convert to arrays
    points_2d = np.array([c[0] for c in correspondences], dtype=np.float32)
    points_3d = np.array([c[1] for c in correspondences], dtype=np.float32)
    point_3d_indices = [c[2] for c in correspondences]
    
    print(f"  [2D-3D] Found {len(correspondences)} 2D-3D correspondences")
    
    return points_2d, points_3d, point_3d_indices


def estimate_camera_pose_pnp(points_2d, points_3d, K):
    """
    Estimate camera pose using PnP with RANSAC
    
    Args:
        points_2d: 2D points in image (Nx2)
        points_3d: Corresponding 3D points (Nx3)
        K: Camera intrinsic matrix (3x3)
    
    Returns:
        success: Whether PnP succeeded
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        inliers: Boolean mask of inlier correspondences
    """
    print(f"\n[PnP] Estimating camera pose from {len(points_2d)} correspondences...")
    
    # Use solvePnPRansac for robustness
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        K,
        None,  # No distortion
        iterationsCount=1000,
        reprojectionError=8.0,  # Pixels
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success or inliers is None:
        print("[PnP] Failed to estimate pose!")
        return False, None, None, None
    
    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec
    
    num_inliers = len(inliers)
    inlier_ratio = num_inliers / len(points_2d) * 100
    
    print(f"[PnP] Success! {num_inliers}/{len(points_2d)} inliers ({inlier_ratio:.1f}%)")
    
    # Create boolean mask
    inlier_mask = np.zeros(len(points_2d), dtype=bool)
    inlier_mask[inliers.ravel()] = True
    
    return True, R, t, inlier_mask


def compute_reprojection_error(points_3d, points_2d, R, t, K):
    """
    Compute reprojection error for verification
    
    Args:
        points_3d: 3D points (Nx3)
        points_2d: Observed 2D points (Nx2)
        R, t: Camera pose
        K: Camera intrinsic matrix
    
    Returns:
        mean_error: Mean reprojection error in pixels
        errors: Per-point errors
    """
    # Project 3D points to 2D
    P = K @ np.hstack([R, t])
    points_3d_hom = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    projected = (P @ points_3d_hom.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    
    # Compute errors
    errors = np.linalg.norm(projected - points_2d, axis=1)
    mean_error = np.mean(errors)
    
    return mean_error, errors


def triangulate_new_points(state, new_img_idx, keypoints, pair_matches):
    """
    Triangulate new 3D points between newly registered camera and existing cameras
    
    Args:
        state: ReconstructionState
        new_img_idx: Index of newly registered camera
        keypoints: All keypoints from Step 1
        pair_matches: All matches from Step 1
    
    Returns:
        new_points_3d: List of new 3D points
        new_tracks: List of tracks for new points
    """
    print(f"\n[Triangulate] Finding new points for camera {new_img_idx}...")
    
    new_points_3d = []
    new_tracks = []
    
    registered_cameras = [cam for cam in state.get_registered_cameras() if cam != new_img_idx]
    
    # Get projection matrix for new camera
    P_new = state.get_projection_matrix(new_img_idx)
    
    for reg_img_idx in registered_cameras:
        # Get matches between new image and this registered image
        if new_img_idx < reg_img_idx:
            pair_key = (new_img_idx, reg_img_idx)
            matches = pair_matches.get(pair_key, [])
            is_flipped = False
        else:
            pair_key = (reg_img_idx, new_img_idx)
            matches = pair_matches.get(pair_key, [])
            is_flipped = True
        
        if not matches:
            continue
        
        # Get projection matrix for registered camera
        P_reg = state.get_projection_matrix(reg_img_idx)
        
        # For each match, check if it's already a 3D point
        for match in matches:
            if is_flipped:
                reg_kp_idx = match.queryIdx
                new_kp_idx = match.trainIdx
            else:
                new_kp_idx = match.queryIdx
                reg_kp_idx = match.trainIdx
            
            # Get 2D points
            pt_2d_new = keypoints[new_img_idx][new_kp_idx]
            pt_2d_reg = keypoints[reg_img_idx][reg_kp_idx]
            
            # Check if this correspondence already exists as a 3D point
            already_exists = False
            for track in state.tracks:
                if reg_img_idx in track:
                    tracked_pt = track[reg_img_idx]
                    if np.linalg.norm(tracked_pt - pt_2d_reg) < 1.0:
                        already_exists = True
                        # Update track with new observation
                        if new_img_idx not in track:
                            track[new_img_idx] = pt_2d_new
                        break
            
            if already_exists:
                continue
            
            # Triangulate new point
            pts1 = pt_2d_new.reshape(2, 1)
            pts2 = pt_2d_reg.reshape(2, 1)
            
            point_4d = cv2.triangulatePoints(P_new, P_reg, pts1, pts2)
            point_3d = point_4d[:3] / point_4d[3]
            point_3d = point_3d.ravel()
            
            # Filter: check depth and reasonable coordinates
            if point_3d[2] > 0 and point_3d[2] < 50 and np.linalg.norm(point_3d) < 50:
                new_points_3d.append(point_3d)
                new_tracks.append({
                    new_img_idx: pt_2d_new,
                    reg_img_idx: pt_2d_reg
                })
    
    print(f"[Triangulate] Created {len(new_points_3d)} new 3D points")
    
    return new_points_3d, new_tracks


def select_next_image(state, keypoints, pair_matches):
    """
    Select the next best image to register
    
    Strategy: Choose image with most 2D-3D correspondences to existing reconstruction
    
    Returns:
        best_img_idx: Index of best image to add next, or None if no good candidates
    """
    print("\n[Select] Choosing next image to register...")
    
    registered = set(state.get_registered_cameras())
    unregistered = [i for i in range(state.num_images) if i not in registered]
    
    if not unregistered:
        print("[Select] All images are registered!")
        return None
    
    best_img_idx = None
    best_count = 0
    
    for img_idx in unregistered:
        # Count potential 2D-3D correspondences
        count = 0
        
        for reg_img_idx in registered:
            if img_idx < reg_img_idx:
                pair_key = (img_idx, reg_img_idx)
            else:
                pair_key = (reg_img_idx, img_idx)
            
            matches = pair_matches.get(pair_key, [])
            count += len(matches)
        
        print(f"  Image {img_idx}: ~{count} potential correspondences")
        
        if count > best_count:
            best_count = count
            best_img_idx = img_idx
    
    print(f"[Select] Best candidate: Image {best_img_idx} with ~{best_count} correspondences")
    
    return best_img_idx


def register_camera(state, new_img_idx, keypoints, descriptors, pair_matches):
    """
    Register a new camera using PnP
    
    Returns:
        success: Whether registration succeeded
    """
    print("\n" + "="*60)
    print(f"REGISTERING CAMERA {new_img_idx}")
    print("="*60)
    
    # Step 1: Find 2D-3D correspondences
    points_2d, points_3d, point_3d_indices = find_2d_3d_correspondences(
        state, new_img_idx, keypoints, descriptors, pair_matches
    )
    
    if points_2d is None or len(points_2d) < 6:
        print(f"[Register] Not enough correspondences (need at least 6)")
        return False
    
    # Step 2: Estimate camera pose with PnP
    success, R, t, inliers = estimate_camera_pose_pnp(points_2d, points_3d, state.K)
    
    if not success:
        return False
    
    # Step 3: Verify pose with reprojection error
    points_2d_inliers = points_2d[inliers]
    points_3d_inliers = points_3d[inliers]
    
    mean_error, errors = compute_reprojection_error(
        points_3d_inliers, points_2d_inliers, R, t, state.K
    )
    
    print(f"[Register] Mean reprojection error: {mean_error:.2f} pixels")
    
    # Check if error is acceptable
    if mean_error > 10.0:
        print(f"[Register] Reprojection error too high, rejecting!")
        return False
    
    # Step 4: Add camera to state
    state.add_camera(new_img_idx, R, t, registered=True)
    
    # Step 5: Triangulate new points
    new_points, new_tracks = triangulate_new_points(
        state, new_img_idx, keypoints, pair_matches
    )
    
    if new_points:
        state.add_points_batch(np.array(new_points), new_tracks)
    
    print(f"\n[Register] ✓ Camera {new_img_idx} successfully registered!")
    print(f"  Total cameras: {len(state.get_registered_cameras())}/{state.num_images}")
    print(f"  Total 3D points: {len(state.points_3d)}")
    
    return True


def incremental_reconstruction(state, keypoints, descriptors, pair_matches):
    """
    Main incremental reconstruction loop
    
    Iteratively adds cameras until all are registered
    """
    print("\n" + "="*60)
    print("STEP 7: INCREMENTAL CAMERA REGISTRATION")
    print("="*60)
    
    max_iterations = state.num_images
    iteration = 0
    
    while iteration < max_iterations:
        # Select next image to register
        next_img_idx = select_next_image(state, keypoints, pair_matches)
        
        if next_img_idx is None:
            print("\n[Incremental] All images registered!")
            break
        
        # Try to register it
        success = register_camera(state, next_img_idx, keypoints, descriptors, pair_matches)
        
        if not success:
            print(f"\n[Incremental] Failed to register image {next_img_idx}, skipping...")
            # Remove from consideration (would need more sophisticated handling in production)
            
        iteration += 1
    
    print("\n" + "="*60)
    print("INCREMENTAL RECONSTRUCTION COMPLETE")
    print("="*60)
    state.summary()


def main():
    # Load reconstruction state from Step 6
    print("[main] Loading reconstruction state...")
    state = ReconstructionState.load("./reconstruction_state.pkl")
    
    # Load features from Step 1
    image_paths, keypoints, descriptors, pair_matches = load_step1_features()
    
    # Run incremental reconstruction
    incremental_reconstruction(state, keypoints, descriptors, pair_matches)
    
    # Save final state
    print("\n[main] Saving final reconstruction state...")
    state.save("./reconstruction_final.pkl")
    
    # Save point cloud
    if state.points_3d:
        points_array = np.array(state.points_3d)
        np.savetxt("./point_cloud_final.txt", points_array, 
                   fmt='%.6f', header='X Y Z', comments='')
        print("[main] Saved final point cloud to point_cloud_final.txt")
    
    print("\n" + "="*60)
    print("STRUCTURE FROM MOTION COMPLETE!")
    print("="*60)
    print(f"✓ {len(state.get_registered_cameras())} cameras registered")
    print(f"✓ {len(state.points_3d)} 3D points reconstructed")
    print("="*60)


if __name__ == "__main__":
    main()
