"""
Create 3D mesh from COLMAP sparse reconstruction results.
"""

import numpy as np
import os

OUTPUT_DIR = "./colmap_output"


def create_mesh_from_colmap():
    """Create 3D mesh from COLMAP point cloud using Poisson reconstruction."""
    print("="*60)
    print("CREATING 3D MESH FROM COLMAP POINT CLOUD")
    print("="*60)
    
    try:
        import pycolmap
        import open3d as o3d
        
        # Load COLMAP reconstruction
        print(f"\n[load] Loading COLMAP reconstruction from {OUTPUT_DIR}/sparse/0/...")
        reconstruction = pycolmap.Reconstruction(f"{OUTPUT_DIR}/sparse/0")
        
        points = np.array([p.xyz for p in reconstruction.points3D.values()])
        colors = np.array([p.color for p in reconstruction.points3D.values()])
        
        print(f"[load] ✓ Loaded {len(points):,} points from COLMAP reconstruction")
        
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
        print(f"\n[save] Saving mesh to {mesh_path}...")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"[save] ✓ Mesh saved to {mesh_path}")
        
        # Visualize mesh
        print("\n[viz] Opening mesh visualization...")
        print("[viz] Close the visualization window to continue...")
        o3d.visualization.draw_geometries(
            [mesh],
            mesh_show_back_face=True,
            window_name="COLMAP 3D Mesh Reconstruction",
            width=1280,
            height=720
        )
        
        print("\n" + "="*60)
        print("MESH CREATION COMPLETE!")
        print("="*60)
        print(f"\n✓ COLMAP mesh saved to: {mesh_path}")
        print(f"\n💡 Compare meshes:")
        print(f"  COLMAP Mesh:  {mesh_path}")
        print(f"  Custom Mesh:  ./reconstruction_mesh.ply")
        print(f"\n💡 Open with: MeshLab, CloudCompare, or Blender")
        print("="*60)
        
        return mesh
        
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install pycolmap open3d")
        return None
    except Exception as e:
        print(f"\n✗ Mesh creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if not os.path.exists(f"{OUTPUT_DIR}/sparse/0"):
        print(f"✗ ERROR: COLMAP reconstruction not found at {OUTPUT_DIR}/sparse/0/")
        print("\nRun one of these first:")
        print("  python3 colmap_visualize.py")
        print("  python3 sfm_colmap_comparison.py")
    else:
        create_mesh_from_colmap()
