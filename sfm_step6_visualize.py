import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
from sfm_step4_reconstruction_state import ReconstructionState


def draw_camera(ax, R, t, K, scale=0.5, color='blue', label=None):
    """
    Draw a camera frustum in 3D
    
    Args:
        ax: matplotlib 3D axis
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        K: Camera intrinsic matrix
        scale: Size of camera frustum
        color: Color of camera
        label: Label for legend
    """
    # Camera center in world coordinates: C = -R^T * t
    C = -R.T @ t
    C = C.ravel()
    
    # Define image plane corners in camera coordinates
    w, h = 640, 480  # Assume standard image size
    corners_image = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T
    
    # Unproject to normalized camera coordinates
    K_inv = np.linalg.inv(K)
    corners_cam = K_inv @ corners_image
    corners_cam = corners_cam * scale  # Scale for visualization
    
    # Transform to world coordinates
    corners_world = R.T @ (corners_cam - t)
    corners_world = corners_world.T
    
    # Draw camera center
    ax.scatter([C[0]], [C[1]], [C[2]], c=color, s=100, marker='o', 
               label=label, edgecolors='black', linewidths=1)
    
    # Draw frustum edges from camera center to corners
    for corner in corners_world:
        ax.plot([C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]], 
                c=color, linewidth=1, alpha=0.6)
    
    # Draw image plane (connect corners)
    for i in range(4):
        next_i = (i + 1) % 4
        ax.plot([corners_world[i, 0], corners_world[next_i, 0]],
                [corners_world[i, 1], corners_world[next_i, 1]],
                [corners_world[i, 2], corners_world[next_i, 2]],
                c=color, linewidth=2, alpha=0.8)
    
    # Draw camera orientation (principal axis)
    principal_axis = R.T @ np.array([[0], [0], [scale * 1.5]]) - R.T @ t
    principal_axis = principal_axis.ravel()
    ax.plot([C[0], principal_axis[0]], [C[1], principal_axis[1]], [C[2], principal_axis[2]],
            c=color, linewidth=2, alpha=0.9)
    
    return C


def visualize_reconstruction(state, save_path=None, show_cameras=True, show_points=True):
    """
    Comprehensive 3D visualization of SfM reconstruction
    
    Args:
        state: ReconstructionState object
        save_path: Path to save figure (optional)
        show_cameras: Whether to show camera frustums
        show_points: Whether to show 3D points
    """
    print("\n[Visualize] Creating 3D reconstruction visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get registered cameras
    registered_cameras = sorted(state.get_registered_cameras())
    
    # Define colors for cameras
    colors = plt.cm.tab10(np.linspace(0, 1, len(registered_cameras)))
    
    camera_centers = []
    
    # Draw cameras
    if show_cameras and registered_cameras:
        print(f"[Visualize] Drawing {len(registered_cameras)} cameras...")
        
        for idx, cam_idx in enumerate(registered_cameras):
            cam = state.get_camera(cam_idx)
            R = cam['R']
            t = cam['t']
            
            color = colors[idx]
            label = f'Camera {cam_idx}'
            
            C = draw_camera(ax, R, t, state.K, scale=1.0, color=color, label=label)
            camera_centers.append(C)
    
    # Draw 3D points
    if show_points and state.points_3d:
        print(f"[Visualize] Drawing {len(state.points_3d)} 3D points...")
        
        points_array = np.array(state.points_3d)
        
        # Color points by track length (number of cameras observing each point)
        track_lengths = [len(track) for track in state.tracks]
        
        scatter = ax.scatter(points_array[:, 0], 
                           points_array[:, 1], 
                           points_array[:, 2],
                           c=track_lengths,
                           cmap='viridis',
                           marker='.',
                           s=1,
                           alpha=0.5,
                           label='3D Points')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Track Length (# observations)', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title('3D Reconstruction: Room Structure from Motion', fontsize=16, fontweight='bold')
    
    # Set equal aspect ratio
    if state.points_3d:
        points_array = np.array(state.points_3d)
        
        if camera_centers:
            # Include camera centers in bounds
            all_points = np.vstack([points_array, np.array(camera_centers)])
        else:
            all_points = points_array
        
        max_range = np.array([
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualize] Saved visualization to {save_path}")
    
    plt.show()
    print("[Visualize] Close the plot window to continue...")


def export_to_ply(state, filepath):
    """
    Export point cloud to PLY format for viewing in external tools
    (MeshLab, CloudCompare, etc.)
    
    Args:
        state: ReconstructionState object
        filepath: Output PLY file path
    """
    print(f"\n[Export] Exporting point cloud to PLY format...")
    
    points_array = np.array(state.points_3d)
    
    # Color points by track length
    track_lengths = np.array([len(track) for track in state.tracks])
    
    # Normalize track lengths to 0-255 for color
    max_track = track_lengths.max()
    colors = (track_lengths / max_track * 255).astype(np.uint8)
    
    # Create RGB colors (using a simple gradient: blue to red)
    r = colors
    g = 255 - colors
    b = np.zeros_like(colors)
    
    with open(filepath, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_array)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices
        for i, point in enumerate(points_array):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r[i]} {g[i]} {b[i]}\n")
    
    print(f"[Export] Saved {len(points_array)} points to {filepath}")
    print(f"[Export] Open with MeshLab, CloudCompare, or similar viewer")


def create_camera_trajectory_plot(state, save_path=None):
    """
    Create a 2D plot showing camera positions (bird's eye view)
    
    Args:
        state: ReconstructionState object
        save_path: Path to save figure (optional)
    """
    print("\n[Visualize] Creating camera trajectory plot...")
    
    registered_cameras = sorted(state.get_registered_cameras())
    
    if not registered_cameras:
        print("[Visualize] No registered cameras to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Compute camera centers
    camera_centers = []
    camera_indices = []
    
    for cam_idx in registered_cameras:
        cam = state.get_camera(cam_idx)
        R = cam['R']
        t = cam['t']
        C = -R.T @ t
        camera_centers.append(C.ravel())
        camera_indices.append(cam_idx)
    
    camera_centers = np.array(camera_centers)
    
    # Plot 1: Top view (X-Z plane)
    ax1.plot(camera_centers[:, 0], camera_centers[:, 2], 'b-o', linewidth=2, markersize=8)
    for i, idx in enumerate(camera_indices):
        ax1.annotate(f'{idx}', (camera_centers[i, 0], camera_centers[i, 2]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Z (meters)', fontsize=12)
    ax1.set_title('Top View (Bird\'s Eye)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Side view (Y-Z plane)
    ax2.plot(camera_centers[:, 1], camera_centers[:, 2], 'r-o', linewidth=2, markersize=8)
    for i, idx in enumerate(camera_indices):
        ax2.annotate(f'{idx}', (camera_centers[i, 1], camera_centers[i, 2]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Y (meters)', fontsize=12)
    ax2.set_ylabel('Z (meters)', fontsize=12)
    ax2.set_title('Side View', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualize] Saved trajectory plot to {save_path}")
    
    plt.show()
    print("[Visualize] Close the plot window to continue...")


def print_reconstruction_statistics(state):
    """
    Print detailed statistics about the reconstruction
    """
    print("\n" + "="*70)
    print("DETAILED RECONSTRUCTION STATISTICS")
    print("="*70)
    
    registered_cameras = state.get_registered_cameras()
    points_array = np.array(state.points_3d) if state.points_3d else None
    
    print(f"\n📷 CAMERAS:")
    print(f"  Total images: {state.num_images}")
    print(f"  Registered: {len(registered_cameras)}")
    print(f"  Registration rate: {len(registered_cameras)/state.num_images*100:.1f}%")
    print(f"  Registered camera indices: {sorted(registered_cameras)}")
    
    if points_array is not None:
        print(f"\n🎯 3D POINTS:")
        print(f"  Total points: {len(points_array)}")
        print(f"  Coordinate ranges:")
        print(f"    X: [{points_array[:, 0].min():.2f}, {points_array[:, 0].max():.2f}] meters")
        print(f"    Y: [{points_array[:, 1].min():.2f}, {points_array[:, 1].max():.2f}] meters")
        print(f"    Z: [{points_array[:, 2].min():.2f}, {points_array[:, 2].max():.2f}] meters")
        print(f"  Mean position: [{points_array[:, 0].mean():.2f}, {points_array[:, 1].mean():.2f}, {points_array[:, 2].mean():.2f}]")
        
        # Point density
        volume = (points_array[:, 0].max() - points_array[:, 0].min()) * \
                 (points_array[:, 1].max() - points_array[:, 1].min()) * \
                 (points_array[:, 2].max() - points_array[:, 2].min())
        density = len(points_array) / volume if volume > 0 else 0
        print(f"  Point density: {density:.2f} points/m³")
    
    if state.tracks:
        track_lengths = [len(track) for track in state.tracks]
        print(f"\n🔗 TRACKS (Point Observations):")
        print(f"  Min observations per point: {min(track_lengths)}")
        print(f"  Max observations per point: {max(track_lengths)}")
        print(f"  Mean observations per point: {np.mean(track_lengths):.2f}")
        print(f"  Median observations per point: {np.median(track_lengths):.1f}")
        
        # Distribution
        for i in range(2, max(track_lengths) + 1):
            count = sum(1 for t in track_lengths if t == i)
            if count > 0:
                print(f"    {i}-view points: {count} ({count/len(track_lengths)*100:.1f}%)")
    
    # Camera spacing
    if len(registered_cameras) > 1:
        camera_centers = []
        for cam_idx in registered_cameras:
            cam = state.get_camera(cam_idx)
            C = -cam['R'].T @ cam['t']
            camera_centers.append(C.ravel())
        
        camera_centers = np.array(camera_centers)
        distances = []
        for i in range(len(camera_centers) - 1):
            dist = np.linalg.norm(camera_centers[i+1] - camera_centers[i])
            distances.append(dist)
        
        print(f"\n📏 CAMERA SPACING:")
        print(f"  Min distance: {min(distances):.2f} meters")
        print(f"  Max distance: {max(distances):.2f} meters")
        print(f"  Mean distance: {np.mean(distances):.2f} meters")
    
    print("="*70)


def main():
    print("="*70)
    print("STEP 8: VISUALIZATION OF 3D RECONSTRUCTION")
    print("="*70)
    
    # Load final reconstruction state
    print("\n[main] Loading final reconstruction state...")
    state = ReconstructionState.load("./reconstruction_final.pkl")
    
    # Print detailed statistics
    print_reconstruction_statistics(state)
    
    # Create main 3D visualization
    print("\n[main] Creating 3D visualization...")
    visualize_reconstruction(
        state, 
        save_path="./reconstruction_3d.png",
        show_cameras=True,
        show_points=True
    )
    
    # Create camera trajectory plot
    print("\n[main] Creating camera trajectory plot...")
    create_camera_trajectory_plot(
        state,
        save_path="./camera_trajectory.png"
    )
    
    # Export to PLY format
    print("\n[main] Exporting to PLY format...")
    export_to_ply(state, "./reconstruction_final.ply")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print("\n📊 Generated files:")
    print("  • reconstruction_3d.png - Full 3D visualization")
    print("  • camera_trajectory.png - Camera positions (top & side view)")
    print("  • reconstruction_final.ply - Point cloud (open with MeshLab/CloudCompare)")
    print("\n💡 Tips:")
    print("  • Check if cameras form a reasonable trajectory around the room")
    print("  • Verify point cloud resembles room structure")
    print("  • Points with more observations (darker/redder) are more reliable")
    print("="*70)


if __name__ == "__main__":
    main()
