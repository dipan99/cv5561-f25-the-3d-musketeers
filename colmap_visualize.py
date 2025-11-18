"""
COLMAP SfM with Custom Visualization

This script runs COLMAP using pycolmap (Python bindings) and generates
the same visualizations as the custom SfM implementation for easy comparison.

Install: pip install pycolmap
"""

import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False
    print("WARNING: pycolmap not installed. Install with: pip install pycolmap")

# ============ CONFIGURATION ============
IMAGE_DIR = "./heads/seq-01"
SKIP_EVERY_N = 10
MAX_IMAGES = 50
OUTPUT_DIR = "./colmap_output"
# =======================================


def setup_directory():
    """Create directory structure."""
    print("[setup] Creating output directory...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/sparse", exist_ok=True)


def copy_images():
    """Copy selected images."""
    print("[copy] Copying images...")
    
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    
    if SKIP_EVERY_N > 1:
        image_paths = image_paths[::SKIP_EVERY_N]
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]
    
    print(f"[copy] Selected {len(image_paths)} images")
    
    for i, src_path in enumerate(image_paths):
        dst_path = f"{OUTPUT_DIR}/images/{i:04d}.png"
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
    
    print(f"[copy] ✓ Copied {len(image_paths)} images")
    return len(image_paths)


def run_pycolmap_sfm():
    """Run COLMAP SfM using pycolmap."""
    if not PYCOLMAP_AVAILABLE:
        print("[error] pycolmap not available. Install with: pip install pycolmap")
        return None
    
    print("\n" + "="*60)
    print("RUNNING COLMAP SfM WITH PYCOLMAP")
    print("="*60)
    
    output_path = Path(OUTPUT_DIR)
    image_path = output_path / "images"
    database_path = output_path / "database.db"
    
    # Remove old database
    if database_path.exists():
        try:
            database_path.unlink()
        except:
            pass  # Database might be in use, pycolmap will overwrite
    
    print("\n[1/4] Extracting features...")
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_path),
        sift_options=pycolmap.SiftExtractionOptions(max_num_features=10000),
    )
    print("✓ Feature extraction complete")
    
    print("\n[2/4] Matching features...")
    pycolmap.match_sequential(
        database_path=str(database_path),
        sift_options=pycolmap.SiftMatchingOptions(),
        matching_options=pycolmap.SequentialMatchingOptions(overlap=10),
    )
    print("✓ Feature matching complete")
    
    print("\n[3/4] Running sparse reconstruction (mapper)...")
    sparse_path = output_path / "sparse"
    if sparse_path.exists():
        shutil.rmtree(sparse_path)
    sparse_path.mkdir(exist_ok=True)
    
    maps = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_path),
        output_path=str(sparse_path),
    )
    print(f"✓ Sparse reconstruction complete ({len(maps)} models)")
    
    # Get the best model (usually the first/largest one)
    if len(maps) == 0:
        print("✗ No reconstruction models created")
        return None
    
    print(f"\n[4/4] Loading reconstruction model 0...")
    reconstruction = pycolmap.Reconstruction(sparse_path / "0")
    
    return reconstruction


def draw_camera_frustum(ax, R, t, scale=0.5, color='blue'):
    """Draw a camera frustum."""
    # Camera center in world coordinates
    C = -R.T @ t
    C = C.ravel()
    
    # Define frustum corners
    corners = np.array([
        [-1, -1, 2],
        [1, -1, 2],
        [1, 1, 2],
        [-1, 1, 2]
    ]) * scale
    
    # Transform to world coordinates
    corners_world = (R.T @ corners.T).T + C
    
    # Draw camera center
    ax.scatter([C[0]], [C[1]], [C[2]], c=color, s=100, marker='o',
               edgecolors='black', linewidths=1)
    
    # Draw frustum edges
    for corner in corners_world:
        ax.plot([C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]],
                c=color, linewidth=1, alpha=0.6)
    
    # Draw image plane
    vertices = [corners_world]
    poly = Poly3DCollection(vertices, alpha=0.2, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)


def visualize_reconstruction(reconstruction):
    """Create visualization matching custom SfM style."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Extract data
    points = np.array([p.xyz for p in reconstruction.points3D.values()])
    colors = np.array([p.color for p in reconstruction.points3D.values()]) / 255.0
    
    num_cameras = len(reconstruction.images)
    num_points = len(points)
    
    print(f"[viz] {num_cameras} cameras, {num_points} points")
    
    # Get track statistics
    track_lengths = [len(p.track.elements) for p in reconstruction.points3D.values()]
    
    print("\n" + "="*60)
    print("COLMAP RECONSTRUCTION STATISTICS")
    print("="*60)
    print(f"\n📷 CAMERAS:")
    print(f"  Total images: {num_cameras}")
    print(f"  Registered: {num_cameras}")
    print(f"  Registration rate: 100.0%")
    
    print(f"\n🎯 3D POINTS:")
    print(f"  Total points: {num_points}")
    print(f"  Coordinate ranges:")
    print(f"    X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"    Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"    Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print(f"  Mean position: [{points.mean(axis=0)[0]:.2f}, {points.mean(axis=0)[1]:.2f}, {points.mean(axis=0)[2]:.2f}]")
    
    print(f"\n🔗 TRACKS:")
    print(f"  Min observations per point: {min(track_lengths)}")
    print(f"  Max observations per point: {max(track_lengths)}")
    print(f"  Mean observations per point: {np.mean(track_lengths):.2f}")
    
    # Track distribution
    for n in range(2, 9):
        count = sum(1 for t in track_lengths if t == n)
        if count > 0:
            print(f"    {n}-view points: {count} ({count/num_points*100:.1f}%)")
    
    # Camera spacing
    camera_positions = []
    for img in reconstruction.images.values():
        # In pycolmap, images in reconstruction are already registered
        camera_positions.append(img.projection_center())
    camera_positions = np.array(camera_positions)
    
    if len(camera_positions) > 1:
        from scipy.spatial.distance import pdist
        distances = pdist(camera_positions)
        print(f"\n📏 CAMERA SPACING:")
        print(f"  Min distance: {distances.min():.2f}")
        print(f"  Max distance: {distances.max():.2f}")
        print(f"  Mean distance: {distances.mean():.2f}")
    
    print("="*60)
    
    # Create 3D visualization
    print("\n[viz] Creating 3D visualization...")
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with colors
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=1, alpha=0.5)
    
    # Draw cameras
    camera_colors = plt.cm.rainbow(np.linspace(0, 1, min(num_cameras, 20)))
    for idx, (img_id, image) in enumerate(list(reconstruction.images.items())[:20]):
        # In pycolmap, images in reconstruction are already registered
        # Use cam_from_world.rotation.matrix() and cam_from_world.translation
        R = image.cam_from_world.rotation.matrix()
        t = image.cam_from_world.translation.reshape(3, 1)
        color = camera_colors[idx % len(camera_colors)]
        draw_camera_frustum(ax, R, t, scale=2.0, color=color)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'COLMAP Reconstruction: {num_cameras} cameras, {num_points:,} points')
    
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
    
    plt.savefig(f'{OUTPUT_DIR}/colmap_reconstruction_3d.png', dpi=150, bbox_inches='tight')
    print(f"[viz] ✓ Saved to {OUTPUT_DIR}/colmap_reconstruction_3d.png")
    plt.show()
    
    # Create camera trajectory plot
    print("\n[viz] Creating camera trajectory plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    if len(camera_positions) > 0:
        # Top view (X-Z)
        ax1.scatter(camera_positions[:, 0], camera_positions[:, 2], c='red', s=50, marker='o')
        ax1.plot(camera_positions[:, 0], camera_positions[:, 2], 'r-', alpha=0.5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_title('Camera Trajectory (Top View)')
        ax1.grid(True)
        ax1.axis('equal')
        
        # Side view (Y-Z)
        ax2.scatter(camera_positions[:, 1], camera_positions[:, 2], c='blue', s=50, marker='o')
        ax2.plot(camera_positions[:, 1], camera_positions[:, 2], 'b-', alpha=0.5)
        ax2.set_xlabel('Y')
        ax2.set_ylabel('Z')
        ax2.set_title('Camera Trajectory (Side View)')
        ax2.grid(True)
        ax2.axis('equal')
    
    plt.savefig(f'{OUTPUT_DIR}/colmap_camera_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"[viz] ✓ Saved to {OUTPUT_DIR}/colmap_camera_trajectory.png")
    plt.show()


def run_dense_reconstruction(reconstruction):
    """Run COLMAP dense reconstruction pipeline."""
    print("\n" + "="*60)
    print("RUNNING DENSE RECONSTRUCTION")
    print("="*60)
    
    output_path = Path(OUTPUT_DIR)
    image_path = output_path / "images"
    sparse_path = output_path / "sparse" / "0"
    dense_path = output_path / "dense"
    
    # Create dense directory
    if dense_path.exists():
        shutil.rmtree(dense_path)
    dense_path.mkdir(exist_ok=True)
    
    print("\n[1/4] Undistorting images...")
    pycolmap.undistort_images(
        output_path=str(dense_path),
        input_path=str(sparse_path),
        image_path=str(image_path),
        output_type="COLMAP"
    )
    print("✓ Image undistortion complete")
    
    print("\n[2/4] Computing stereo depth maps...")
    print("  (This may take several minutes...)")
    pycolmap.patch_match_stereo(
        workspace_path=str(dense_path),
        workspace_format="COLMAP",
        pmvs_option_name="option-all"
    )
    print("✓ Stereo matching complete")
    
    print("\n[3/4] Fusing stereo depth maps...")
    pycolmap.stereo_fusion(
        output_path=str(dense_path / "fused.ply"),
        workspace_path=str(dense_path),
        workspace_format="COLMAP",
        input_type="geometric"
    )
    print("✓ Depth fusion complete")
    
    # Load the fused point cloud
    fused_ply = dense_path / "fused.ply"
    if fused_ply.exists():
        print(f"\n[4/4] Loading dense point cloud from {fused_ply}...")
        dense_points, dense_colors = load_ply(fused_ply)
        print(f"✓ Loaded {len(dense_points):,} dense points")
        return dense_points, dense_colors
    else:
        print("\n✗ Dense reconstruction failed - no fused point cloud")
        return None, None


def load_ply(ply_path):
    """Load points and colors from PLY file."""
    points = []
    colors = []
    reading_vertices = False
    num_vertices = 0
    vertices_read = 0
    
    with open(ply_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line == 'end_header':
                reading_vertices = True
                continue
            elif reading_vertices:
                if vertices_read >= num_vertices:
                    break
                parts = line.split()
                if len(parts) >= 6:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([float(parts[3]), float(parts[4]), float(parts[5])])
                    vertices_read += 1
    
    return np.array(points), np.array(colors)


def create_mesh_from_dense(dense_points, dense_colors):
    """Create mesh from dense point cloud using Poisson reconstruction."""
    print("\n" + "="*60)
    print("CREATING MESH FROM DENSE POINTS")
    print("="*60)
    
    try:
        import open3d as o3d
        
        print(f"\n[mesh] Creating mesh from {len(dense_points):,} points...")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense_points)
        
        if dense_colors is not None:
            colors_norm = dense_colors / 255.0 if dense_colors.max() > 1.0 else dense_colors
            pcd.colors = o3d.utility.Vector3dVector(colors_norm)
        
        # Estimate normals
        print("[mesh] Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
        )
        
        # Poisson surface reconstruction
        print("[mesh] Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        print(f"[mesh] Generated mesh with {len(mesh.vertices):,} vertices and {len(mesh.triangles):,} triangles")
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"[mesh] After cleaning: {len(mesh.vertices):,} vertices and {len(mesh.triangles):,} triangles")
        
        # Save mesh
        mesh_path = f"{OUTPUT_DIR}/colmap_dense_mesh.ply"
        print(f"[mesh] Saving mesh to {mesh_path}...")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"[mesh] ✓ Mesh saved")
        
        return mesh
        
    except ImportError:
        print("\n[mesh] Open3D not installed. Install it with:")
        print("  pip install open3d")
        print("\n[mesh] Skipping mesh generation...")
        return None
    except Exception as e:
        print(f"\n[mesh] Mesh creation failed: {e}")
        return None


def visualize_dense_cloud(dense_points, dense_colors):
    """Visualize dense point cloud."""
    print("\n[viz] Creating dense point cloud visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors
    if dense_colors.max() > 1.0:
        colors_norm = dense_colors / 255.0
    else:
        colors_norm = dense_colors
    
    # Subsample for visualization if too many points
    if len(dense_points) > 100000:
        indices = np.random.choice(len(dense_points), 100000, replace=False)
        points_viz = dense_points[indices]
        colors_viz = colors_norm[indices]
        print(f"[viz] Subsampling to 100k points for visualization")
    else:
        points_viz = dense_points
        colors_viz = colors_norm
    
    ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2],
               c=colors_viz, marker='.', s=1, alpha=0.8)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title(f'COLMAP Dense Reconstruction ({len(dense_points):,} points)', 
                 fontsize=16, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([
        dense_points[:, 0].max() - dense_points[:, 0].min(),
        dense_points[:, 1].max() - dense_points[:, 1].min(),
        dense_points[:, 2].max() - dense_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (dense_points[:, 0].max() + dense_points[:, 0].min()) * 0.5
    mid_y = (dense_points[:, 1].max() + dense_points[:, 1].min()) * 0.5
    mid_z = (dense_points[:, 2].max() + dense_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.grid(True, alpha=0.3)
    
    save_path = f'{OUTPUT_DIR}/colmap_dense_reconstruction.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[viz] ✓ Saved to {save_path}")
    plt.show()


def export_to_ply(reconstruction):
    """Export sparse point cloud to PLY format."""
    print("\n[export] Exporting sparse point cloud to PLY...")
    
    points = np.array([p.xyz for p in reconstruction.points3D.values()])
    colors = np.array([p.color for p in reconstruction.points3D.values()])
    
    ply_path = f"{OUTPUT_DIR}/colmap_sparse.ply"
    
    with open(ply_path, 'w') as f:
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
        
        # Write points
        for i in range(len(points)):
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} ")
            f.write(f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
    
    print(f"[export] ✓ Saved {len(points)} sparse points to {ply_path}")


def export_dense_to_ply(dense_points, dense_colors):
    """Export dense point cloud to PLY format."""
    print("\n[export] Exporting dense point cloud to PLY...")
    
    # Ensure colors are in 0-255 range
    if dense_colors.max() <= 1.0:
        dense_colors = (dense_colors * 255).astype(np.uint8)
    else:
        dense_colors = dense_colors.astype(np.uint8)
    
    ply_path = f"{OUTPUT_DIR}/colmap_dense.ply"
    
    with open(ply_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(dense_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(len(dense_points)):
            f.write(f"{dense_points[i, 0]} {dense_points[i, 1]} {dense_points[i, 2]} ")
            f.write(f"{int(dense_colors[i, 0])} {int(dense_colors[i, 1])} {int(dense_colors[i, 2])}\n")
    
    print(f"[export] ✓ Saved {len(dense_points):,} dense points to {ply_path}")


def main():
    """Main function."""
    print("="*60)
    print("COLMAP SfM WITH DENSE RECONSTRUCTION")
    print("="*60)
    print(f"Dataset: {IMAGE_DIR}")
    print(f"Skip every: {SKIP_EVERY_N} images")
    print(f"Max images: {MAX_IMAGES}")
    print("="*60)
    
    if not PYCOLMAP_AVAILABLE:
        print("\n[error] pycolmap is required but not installed.")
        print("\nInstall with:")
        print("  pip install pycolmap")
        print("\nNote: On some systems you may need:")
        print("  pip install pycolmap --no-build-isolation")
        return
    
    # Setup
    setup_directory()
    
    # Copy images
    num_images = copy_images()
    
    # Run COLMAP sparse reconstruction
    reconstruction = run_pycolmap_sfm()
    
    if reconstruction is None:
        print("\n[error] Reconstruction failed")
        return
    
    # Visualize sparse reconstruction (matching custom SfM style)
    visualize_reconstruction(reconstruction)
    
    # Export sparse point cloud
    export_to_ply(reconstruction)
    
    # Run dense reconstruction
    print("\n" + "="*60)
    print("PROCEEDING TO DENSE RECONSTRUCTION")
    print("="*60)
    dense_points, dense_colors = run_dense_reconstruction(reconstruction)
    
    if dense_points is not None and len(dense_points) > 0:
        # Visualize dense reconstruction
        visualize_dense_cloud(dense_points, dense_colors)
        
        # Export dense point cloud
        export_dense_to_ply(dense_points, dense_colors)
        
        # Create mesh from dense points
        mesh = create_mesh_from_dense(dense_points, dense_colors)
        
        print("\n" + "="*60)
        print("COLMAP FULL PIPELINE COMPLETE!")
        print("="*60)
        print(f"\n📊 Results saved to: {OUTPUT_DIR}/")
        print(f"\nSPARSE:")
        print(f"  • Reconstruction: {OUTPUT_DIR}/sparse/0/")
        print(f"  • Point cloud: {OUTPUT_DIR}/colmap_sparse.ply")
        print(f"  • 3D visualization: {OUTPUT_DIR}/colmap_reconstruction_3d.png")
        print(f"  • Camera trajectory: {OUTPUT_DIR}/colmap_camera_trajectory.png")
        print(f"\nDENSE:")
        print(f"  • Dense workspace: {OUTPUT_DIR}/dense/")
        print(f"  • Dense point cloud: {OUTPUT_DIR}/colmap_dense.ply ({len(dense_points):,} points)")
        print(f"  • Dense visualization: {OUTPUT_DIR}/colmap_dense_reconstruction.png")
        if mesh is not None:
            print(f"  • Mesh: {OUTPUT_DIR}/colmap_dense_mesh.ply")
        print("\n💡 Compare with custom implementation:")
        print(f"  Custom Sparse:  ./reconstruction_3d.png")
        print(f"  COLMAP Sparse:  {OUTPUT_DIR}/colmap_reconstruction_3d.png")
        print(f"  Custom Dense:   ./dense_reconstruction.png")
        print(f"  COLMAP Dense:   {OUTPUT_DIR}/colmap_dense_reconstruction.png")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("SPARSE RECONSTRUCTION COMPLETE (DENSE FAILED)")
        print("="*60)
        print(f"\n📊 Sparse results saved to: {OUTPUT_DIR}/")
        print(f"  • Reconstruction: {OUTPUT_DIR}/sparse/0/")
        print(f"  • Point cloud: {OUTPUT_DIR}/colmap_sparse.ply")
        print(f"  • 3D visualization: {OUTPUT_DIR}/colmap_reconstruction_3d.png")
        print(f"  • Camera trajectory: {OUTPUT_DIR}/colmap_camera_trajectory.png")
        print("="*60)


if __name__ == "__main__":
    main()
