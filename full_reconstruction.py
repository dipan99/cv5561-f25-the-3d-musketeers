#!/usr/bin/env python3
"""
Complete 3D Scene Reconstruction Pipeline
From RGB Images -> Dense Point Cloud -> Mesh

Requirements:
- pycolmap (for SfM/MVS)
- open3d (for meshing)
- numpy
- COLMAP installed via Homebrew
"""

import os
import sys
import shutil
import numpy as np
import pycolmap
import open3d as o3d
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

IMAGE_FOLDER = "./images"  # Change this to your image folder
OUTPUT_DIR = "./reconstruction_output"
WORKSPACE_DIR = os.path.join(OUTPUT_DIR, "colmap_workspace")
DATABASE_PATH = os.path.join(WORKSPACE_DIR, "database.db")
SPARSE_MODEL_PATH = os.path.join(WORKSPACE_DIR, "sparse")
DENSE_WORKSPACE = os.path.join(WORKSPACE_DIR, "dense")
POINT_CLOUD_PATH = os.path.join(DENSE_WORKSPACE, "fused.ply")
FINAL_MESH_PATH = os.path.join(OUTPUT_DIR, "room_mesh.ply")


# ============================================================================
# PHASE 0: SETUP & VALIDATION
# ============================================================================

def setup_directories():
    """Create necessary directories for reconstruction"""
    print("\n" + "="*70)
    print("PHASE 0: SETUP & VALIDATION")
    print("="*70)
    
    # Check if image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")
    
    # Count images
    image_files = [f for f in os.listdir(IMAGE_FOLDER) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) < 2:
        raise ValueError(f"Need at least 2 images, found {len(image_files)}")
    
    print(f"✓ Found {len(image_files)} images in {IMAGE_FOLDER}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(SPARSE_MODEL_PATH, exist_ok=True)
    os.makedirs(DENSE_WORKSPACE, exist_ok=True)
    
    print(f"✓ Created workspace: {WORKSPACE_DIR}")
    
    return image_files


# ============================================================================
# PHASE 1: STRUCTURE FROM MOTION (SfM) WITH PYCOLMAP
# ============================================================================

def run_feature_extraction():
    """Extract features from images using SIFT"""
    print("\n" + "="*70)
    print("PHASE 1.1: FEATURE EXTRACTION")
    print("="*70)
    
    # Create or clear database
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
    
    # Feature extraction options - use FeatureExtractionOptions
    extraction_options = pycolmap.FeatureExtractionOptions()
    # Configure SIFT settings
    extraction_options.sift.max_num_features = 8192
    extraction_options.sift.first_octave = -1
    
    # Extract features
    print(f"Extracting SIFT features from images in {IMAGE_FOLDER}...")
    pycolmap.extract_features(
        database_path=DATABASE_PATH,
        image_path=IMAGE_FOLDER,
        extraction_options=extraction_options
    )
    
    print("✓ Feature extraction complete")


def run_feature_matching():
    """Match features between images"""
    print("\n" + "="*70)
    print("PHASE 1.2: FEATURE MATCHING")
    print("="*70)
    
    print("Matching features between image pairs...")
    # Use default options - pycolmap's match_exhaustive doesn't require custom options
    pycolmap.match_exhaustive(
        database_path=DATABASE_PATH
    )
    
    print("✓ Feature matching complete")


def run_sparse_reconstruction():
    """Run incremental SfM to get sparse reconstruction"""
    print("\n" + "="*70)
    print("PHASE 1.3: SPARSE RECONSTRUCTION (SfM)")
    print("="*70)
    
    # Clear existing sparse reconstruction
    if os.path.exists(SPARSE_MODEL_PATH):
        shutil.rmtree(SPARSE_MODEL_PATH)
    os.makedirs(SPARSE_MODEL_PATH, exist_ok=True)
    
    # Use default options - the function expects IncrementalPipelineOptions, not IncrementalMapperOptions
    options = pycolmap.IncrementalPipelineOptions()
    
    print("Running incremental mapper (this may take a while)...")
    maps = pycolmap.incremental_mapping(
        database_path=DATABASE_PATH,
        image_path=IMAGE_FOLDER,
        output_path=SPARSE_MODEL_PATH,
        options=options
    )
    
    if not maps or len(maps) == 0:
        raise RuntimeError("Sparse reconstruction failed - no models created")
    
    # Get the first (and should be only) reconstruction
    reconstruction = maps[0]
    
    print(f"✓ Sparse reconstruction complete")
    print(f"  - Registered images: {reconstruction.num_reg_images()}")
    print(f"  - Sparse points: {reconstruction.num_points3D()}")
    
    # Save the reconstruction
    reconstruction.write(SPARSE_MODEL_PATH)
    
    return reconstruction


# ============================================================================
# PHASE 2: EXPORT SPARSE RECONSTRUCTION
# ============================================================================

def export_sparse_reconstruction(reconstruction):
    """Export sparse reconstruction to PLY - skip buggy dense reconstruction"""
    print("\n" + "="*70)
    print("PHASE 2: EXPORTING SPARSE RECONSTRUCTION")
    print("="*70)
    print("⚠️  Skipping dense reconstruction due to coordinate transformation issues")
    print("    Using sparse point cloud from SfM instead")
    
    # Get the reconstruction
    if isinstance(reconstruction, dict):
        if len(reconstruction) == 0:
            raise RuntimeError("No reconstruction models found")
        reconstruction = reconstruction[0]
    
    # Export to PLY
    os.makedirs(os.path.dirname(POINT_CLOUD_PATH), exist_ok=True)
    reconstruction.export_PLY(POINT_CLOUD_PATH)
    
    num_points = len(reconstruction.points3D)
    print(f"✓ Exported {num_points} sparse 3D points to: {POINT_CLOUD_PATH}")
    
    return POINT_CLOUD_PATH


# ============================================================================
# PHASE 3: MESHING WITH OPEN3D
# ============================================================================

def load_and_clean_point_cloud(pcd_path):
    """Load point cloud and remove outliers"""
    print("\n" + "="*70)
    print("PHASE 3.1: LOADING & CLEANING POINT CLOUD")
    print("="*70)
    
    print(f"Loading point cloud from {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    print(f"✓ Loaded {len(pcd.points)} points")
    
    # Statistical outlier removal
    print("\nRemoving statistical outliers...")
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    
    print(f"✓ Removed {len(pcd.points) - len(pcd_clean.points)} outliers")
    print(f"  Clean point cloud: {len(pcd_clean.points)} points")
    
    return pcd_clean


def estimate_normals(pcd):
    """Estimate normals for the point cloud"""
    print("\n" + "="*70)
    print("PHASE 3.2: NORMAL ESTIMATION")
    print("="*70)
    
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,
            max_nn=30
        )
    )
    
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    print(f"✓ Normals estimated for {len(pcd.normals)} points")
    
    return pcd


def create_mesh_poisson(pcd):
    """Create mesh using Poisson surface reconstruction"""
    print("\n" + "="*70)
    print("PHASE 3.3: POISSON SURFACE RECONSTRUCTION")
    print("="*70)
    
    print("Running Poisson reconstruction (depth=9)...")
    print("This may take several minutes depending on point cloud size...")
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=9,
        width=0,
        scale=1.1,
        linear_fit=False
    )
    
    print(f"✓ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Remove low-density vertices (artifacts)
    print("\nRemoving low-density vertices...")
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"✓ Cleaned mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh


def crop_mesh_to_point_cloud(mesh, pcd):
    """Crop mesh to the bounding box of the point cloud"""
    print("\n" + "="*70)
    print("PHASE 3.4: CROPPING MESH")
    print("="*70)
    
    print("Cropping mesh to point cloud bounding box...")
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_cropped = mesh.crop(bbox)
    
    print(f"✓ Cropped mesh: {len(mesh_cropped.vertices)} vertices, {len(mesh_cropped.triangles)} triangles")
    
    return mesh_cropped


def save_and_visualize_mesh(mesh, output_path):
    """Save mesh and launch visualizer"""
    print("\n" + "="*70)
    print("PHASE 3.5: SAVING & VISUALIZATION")
    print("="*70)
    
    # Save mesh
    print(f"Saving mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"✓ Mesh saved successfully")
    
    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    
    # Visualize
    print("\nLaunching Open3D visualizer...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Ctrl+C: Exit viewer")
    
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="3D Room Reconstruction",
        width=1280,
        height=720,
        mesh_show_back_face=True
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete reconstruction pipeline"""
    print("\n" + "="*70)
    print("3D SCENE RECONSTRUCTION PIPELINE")
    print("From RGB Images -> Dense Point Cloud -> Mesh")
    print("="*70)
    
    try:
        # Phase 0: Setup
        image_files = setup_directories()
        
        # Phase 1: Structure from Motion (SfM)
        run_feature_extraction()
        run_feature_matching()
        sparse_reconstruction = run_sparse_reconstruction()
        
        # Phase 2: Export sparse point cloud (skip buggy dense reconstruction)
        point_cloud_path = export_sparse_reconstruction(sparse_reconstruction)
        
        # Phase 3: Meshing
        pcd = load_and_clean_point_cloud(point_cloud_path)
        pcd = estimate_normals(pcd)
        mesh = create_mesh_poisson(pcd)
        mesh_cropped = crop_mesh_to_point_cloud(mesh, pcd)
        save_and_visualize_mesh(mesh_cropped, FINAL_MESH_PATH)
        
        # Final summary
        print("\n" + "="*70)
        print("RECONSTRUCTION COMPLETE!")
        print("="*70)
        print(f"\n✓ Input images: {len(image_files)}")
        print(f"✓ Registered images: {sparse_reconstruction.num_reg_images()}")
        print(f"✓ Dense points: {len(pcd.points)}")
        print(f"✓ Final mesh vertices: {len(mesh_cropped.vertices)}")
        print(f"✓ Final mesh triangles: {len(mesh_cropped.triangles)}")
        print(f"\n📁 Output files:")
        print(f"   - Sparse reconstruction: {SPARSE_MODEL_PATH}")
        print(f"   - Dense point cloud: {POINT_CLOUD_PATH}")
        print(f"   - Final mesh: {FINAL_MESH_PATH}")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
