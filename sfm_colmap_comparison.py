"""
SfM using COLMAP library for comparison with custom implementation.

This script uses pycolmap (Python bindings for COLMAP) to perform
Structure from Motion on the same dataset and compare results.

Install: pip install pycolmap
"""

import os
import glob
import shutil
import subprocess
import numpy as np
import cv2
from pathlib import Path

# ============ CONFIGURATION ============
IMAGE_DIR = "./heads/seq-01"
SKIP_EVERY_N = 10
MAX_IMAGES = 50
OUTPUT_DIR = "./colmap_output"
# =======================================


def setup_colmap_directory():
    """Create directory structure for COLMAP."""
    print("[setup] Creating COLMAP directory structure...")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/sparse", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/dense", exist_ok=True)
    
    print(f"[setup] Output directory: {OUTPUT_DIR}")


def copy_images_for_colmap():
    """Copy selected images to COLMAP directory."""
    print("[copy] Copying images for COLMAP...")
    
    # Get all images
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    
    # Apply skip and max
    if SKIP_EVERY_N > 1:
        image_paths = image_paths[::SKIP_EVERY_N]
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]
    
    print(f"[copy] Selected {len(image_paths)} images")
    
    # Copy images with simpler names
    for i, src_path in enumerate(image_paths):
        dst_path = f"{OUTPUT_DIR}/images/{i:04d}.png"
        shutil.copy2(src_path, dst_path)
        if i % 10 == 0:
            print(f"[copy] Copied {i+1}/{len(image_paths)} images...")
    
    print(f"[copy] ✓ Copied {len(image_paths)} images to {OUTPUT_DIR}/images/")
    return len(image_paths)


def run_colmap_feature_extraction():
    """Run COLMAP feature extraction."""
    print("\n" + "="*60)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*60)
    
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", f"{OUTPUT_DIR}/database.db",
        "--image_path", f"{OUTPUT_DIR}/images",
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", "10000",
    ]
    
    print(f"[colmap] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[colmap] ERROR in feature extraction:")
        print(result.stderr)
        return False
    
    print("[colmap] ✓ Feature extraction complete")
    return True

def run_colmap_feature_matching():
    """Run COLMAP feature matching."""
    print("\n" + "="*60)
    print("STEP 2: FEATURE MATCHING")
    print("="*60)

    cmd = [
        "colmap", "sequential_matcher",
        "--database_path", f"{OUTPUT_DIR}/database.db",
        "--SequentialMatching.overlap", "10",
    ]
    
    print(f"[colmap] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[colmap] ERROR in feature matching:")
        print(result.stderr)
        return False
    
    print("[colmap] ✓ Feature matching complete")
    return True


def run_colmap_mapper():
    """Run COLMAP sparse reconstruction."""
    print("\n" + "="*60)
    print("STEP 3: SPARSE RECONSTRUCTION (MAPPER)")
    print("="*60)
    
    cmd = [
        "colmap", "mapper",
        "--database_path", f"{OUTPUT_DIR}/database.db",
        "--image_path", f"{OUTPUT_DIR}/images",
        "--output_path", f"{OUTPUT_DIR}/sparse",
    ]
    
    print(f"[colmap] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[colmap] ERROR in mapper:")
        print(result.stderr)
        return False
    
    print("[colmap] ✓ Sparse reconstruction complete")
    return True


def run_colmap_dense_reconstruction():
    """Run COLMAP dense reconstruction (optional, can be slow)."""
    print("\n" + "="*60)
    print("STEP 4: DENSE RECONSTRUCTION")
    print("="*60)
    print("[colmap] WARNING: Dense reconstruction can be very slow!")
    print("[colmap] Skipping dense reconstruction for faster comparison...")
    print("[colmap] To enable, uncomment the dense reconstruction code.")
    
    # Uncomment below to run dense reconstruction
    """
    # Undistort images
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", f"{OUTPUT_DIR}/images",
        "--input_path", f"{OUTPUT_DIR}/sparse/0",
        "--output_path", f"{OUTPUT_DIR}/dense",
        "--output_type", "COLMAP",
    ]
    subprocess.run(cmd, capture_output=True)
    
    # Patch match stereo
    cmd = [
        "colmap", "patch_match_stereo",
        "--workspace_path", f"{OUTPUT_DIR}/dense",
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ]
    subprocess.run(cmd, capture_output=True)
    
    # Stereo fusion
    cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", f"{OUTPUT_DIR}/dense",
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", f"{OUTPUT_DIR}/dense/fused.ply",
    ]
    subprocess.run(cmd, capture_output=True)
    """
    
    return True


def parse_colmap_results():
    """Parse COLMAP reconstruction results."""
    print("\n" + "="*60)
    print("PARSING COLMAP RESULTS")
    print("="*60)
    
    try:
        import pycolmap
        
        reconstruction = pycolmap.Reconstruction(f"{OUTPUT_DIR}/sparse/0")
        
        print(f"\n[results] COLMAP Reconstruction Summary:")
        print(f"  • Registered images: {len(reconstruction.images)}")
        print(f"  • Total cameras: {len(reconstruction.cameras)}")
        print(f"  • 3D points: {len(reconstruction.points3D)}")
        
        # Get point statistics
        if len(reconstruction.points3D) > 0:
            points = np.array([p.xyz for p in reconstruction.points3D.values()])
            print(f"\n[results] 3D Point Cloud Statistics:")
            print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
            print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
            print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
            print(f"  Mean position: [{points.mean(axis=0)[0]:.2f}, {points.mean(axis=0)[1]:.2f}, {points.mean(axis=0)[2]:.2f}]")
            
            # Calculate track lengths
            track_lengths = [len(p.track.elements) for p in reconstruction.points3D.values()]
            print(f"\n[results] Track Statistics:")
            print(f"  Min observations per point: {min(track_lengths)}")
            print(f"  Max observations per point: {max(track_lengths)}")
            print(f"  Avg observations per point: {np.mean(track_lengths):.2f}")
        
        # Camera poses
        print(f"\n[results] Camera Information:")
        for img_id, image in list(reconstruction.images.items())[:3]:
            print(f"  Camera {img_id}:")
            print(f"    Position: {image.projection_center()}")
            print(f"    Registered: {image.registered}")
        
        if len(reconstruction.images) > 3:
            print(f"  ... and {len(reconstruction.images) - 3} more cameras")
        
        # Export to PLY
        ply_path = f"{OUTPUT_DIR}/colmap_sparse.ply"
        print(f"\n[export] Exporting point cloud to {ply_path}...")
        export_colmap_to_ply(reconstruction, ply_path)
        
        return reconstruction
        
    except ImportError:
        print("[results] ERROR: pycolmap not installed")
        print("[results] Install with: pip install pycolmap")
        print("[results] Falling back to manual file parsing...")
        return parse_colmap_files_manually()
    except Exception as e:
        print(f"[results] ERROR parsing COLMAP results: {e}")
        return None


def parse_colmap_files_manually():
    """Manually parse COLMAP text files if pycolmap not available."""
    points_file = f"{OUTPUT_DIR}/sparse/0/points3D.txt"
    cameras_file = f"{OUTPUT_DIR}/sparse/0/cameras.txt"
    images_file = f"{OUTPUT_DIR}/sparse/0/images.txt"
    
    # Count cameras
    num_cameras = 0
    if os.path.exists(cameras_file):
        with open(cameras_file, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    num_cameras += 1
    
    # Count images
    num_images = 0
    if os.path.exists(images_file):
        with open(images_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    num_images += 1
    num_images = num_images // 2  # Each image has 2 lines
    
    # Parse 3D points
    points = []
    if os.path.exists(points_file):
        with open(points_file, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        points.append([x, y, z])
    
    points = np.array(points)
    
    print(f"\n[results] COLMAP Reconstruction Summary (manual parsing):")
    print(f"  • Cameras: {num_cameras}")
    print(f"  • Registered images: {num_images}")
    print(f"  • 3D points: {len(points)}")
    
    if len(points) > 0:
        print(f"\n[results] 3D Point Cloud Statistics:")
        print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    return None


def export_colmap_to_ply(reconstruction, output_path):
    """Export COLMAP point cloud to PLY format."""
    points = np.array([p.xyz for p in reconstruction.points3D.values()])
    colors = np.array([p.color for p in reconstruction.points3D.values()])
    
    with open(output_path, 'w') as f:
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
    
    print(f"[export] ✓ Exported {len(points)} points to {output_path}")


def create_mesh_from_colmap(reconstruction):
    """Create 3D mesh from COLMAP point cloud using Poisson reconstruction."""
    print("\n" + "="*60)
    print("CREATING 3D MESH FROM COLMAP POINT CLOUD")
    print("="*60)
    
    try:
        import open3d as o3d
        
        points = np.array([p.xyz for p in reconstruction.points3D.values()])
        colors = np.array([p.color for p in reconstruction.points3D.values()])
        
        print(f"\n[mesh] Creating mesh from {len(points):,} points using Open3D...")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Normalize colors to 0-1
        colors_norm = colors / 255.0 if colors.max() > 1.0 else colors
        pcd.colors = o3d.utility.Vector3dVector(colors_norm)
        
        # Estimate normals
        print("[mesh] Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
        )
        
        # Poisson surface reconstruction
        print("[mesh] Running Poisson surface reconstruction...")
        print("[mesh] (This may take a few minutes...)")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        print(f"[mesh] Generated mesh with {len(mesh.vertices):,} vertices and {len(mesh.triangles):,} triangles")
        
        # Remove low density vertices
        print("[mesh] Cleaning mesh...")
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"[mesh] After cleaning: {len(mesh.vertices):,} vertices and {len(mesh.triangles):,} triangles")
        
        # Save mesh
        mesh_path = f"{OUTPUT_DIR}/colmap_mesh.ply"
        print(f"[mesh] Saving mesh to {mesh_path}...")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"[mesh] ✓ Mesh saved to {mesh_path}")
        
        # Visualize mesh
        print("\n[mesh] Opening mesh visualization...")
        print("[mesh] Close the visualization window to continue...")
        o3d.visualization.draw_geometries(
            [mesh],
            mesh_show_back_face=True,
            window_name="COLMAP 3D Mesh Reconstruction"
        )
        
        return mesh
        
    except ImportError:
        print("\n[mesh] ✗ Open3D not installed. Cannot create mesh.")
        print("[mesh] Install it with:")
        print("  pip install open3d")
        print("\n[mesh] Skipping mesh generation...")
        return None
    except Exception as e:
        print(f"\n[mesh] ✗ Mesh creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_custom_implementation():
    """Compare COLMAP results with custom implementation."""
    print("\n" + "="*60)
    print("COMPARISON: COLMAP vs CUSTOM IMPLEMENTATION")
    print("="*60)
    
    # Load custom implementation results
    try:
        custom_data = np.load("./step1_features.npz", allow_pickle=True)
        num_custom_images = custom_data['num_images']
        
        # Load final reconstruction
        if os.path.exists("./reconstruction_final.pkl"):
            import pickle
            with open("./reconstruction_final.pkl", 'rb') as f:
                custom_state = pickle.load(f)
            
            num_custom_cameras = len([c for c in custom_state.cameras if c.registered])
            num_custom_points = len(custom_state.points_3d)
            
            print(f"\n[comparison] Custom Implementation:")
            print(f"  • Total images: {num_custom_images}")
            print(f"  • Registered cameras: {num_custom_cameras}")
            print(f"  • 3D points: {num_custom_points}")
        else:
            print("[comparison] Custom implementation results not found")
            print("[comparison] Run 'make all' first to generate custom results")
    
    except Exception as e:
        print(f"[comparison] Could not load custom results: {e}")

def check_colmap_installed():
    """Check if COLMAP is installed."""
    try:
        result = subprocess.run(["colmap", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[check] ✓ COLMAP is installed: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        print("[check] ✗ COLMAP not found")
        print("\n[install] To install COLMAP:")
        print("  macOS:   brew install colmap")
        print("  Ubuntu:  sudo apt install colmap")
        print("  Windows: Download from https://github.com/colmap/colmap/releases")
        print("\nAlternatively, use pycolmap (Python-only):")
        print("  pip install pycolmap")
        return False

def main():
    """Main function to run COLMAP SfM pipeline."""
    print("="*60)
    print("COLMAP SfM COMPARISON")
    print("="*60)
    print(f"Dataset: {IMAGE_DIR}")
    print(f"Skip every: {SKIP_EVERY_N} images")
    print(f"Max images: {MAX_IMAGES}")
    print("="*60)
    
    # Check if COLMAP is installed
    if not check_colmap_installed():
        print("\n[error] COLMAP not installed. Exiting.")
        return
    
    # Setup
    setup_colmap_directory()
    
    # Copy images
    num_images = copy_images_for_colmap()
    
    # Run COLMAP pipeline
    if not run_colmap_feature_extraction():
        return
    
    if not run_colmap_feature_matching():
        return
    
    if not run_colmap_mapper():
        return
    
    # Optional: dense reconstruction
    run_colmap_dense_reconstruction()
    
    # Parse and display results
    reconstruction = parse_colmap_results()
    
    # Create 3D mesh from COLMAP point cloud
    if reconstruction is not None:
        mesh = create_mesh_from_colmap(reconstruction)
    
    # Compare with custom implementation
    compare_with_custom_implementation()
    
    print("\n" + "="*60)
    print("COLMAP PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📊 Results saved to: {OUTPUT_DIR}/")
    print(f"  • Database: {OUTPUT_DIR}/database.db")
    print(f"  • Sparse reconstruction: {OUTPUT_DIR}/sparse/0/")
    print(f"  • Sparse point cloud (PLY): {OUTPUT_DIR}/colmap_sparse.ply")
    if reconstruction is not None:
        print(f"  • 3D Mesh (PLY): {OUTPUT_DIR}/colmap_mesh.ply")
    print("\n💡 Compare the results:")
    print(f"  COLMAP Point Cloud:  {OUTPUT_DIR}/colmap_sparse.ply")
    print(f"  Custom Point Cloud:  ./reconstruction_final.ply")
    if reconstruction is not None:
        print(f"  COLMAP Mesh:         {OUTPUT_DIR}/colmap_mesh.ply")
        print(f"  Custom Mesh:         ./reconstruction_mesh.ply")
    print("\n💡 Open .ply files with MeshLab, CloudCompare, or Blender")
    print("="*60)



if __name__ == "__main__":
    main()
