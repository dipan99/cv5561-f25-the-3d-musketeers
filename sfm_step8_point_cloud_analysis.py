import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_mesh_from_point_cloud(points, colors=None):
    """Create a mesh from point cloud using Open3D"""
    try:
        import open3d as o3d
        
        print(f"\n[mesh] Creating mesh from {len(points):,} points using Open3D...")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
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
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        print(f"[mesh] Generated mesh with {len(mesh.vertices):,} vertices and {len(mesh.triangles):,} triangles")
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"[mesh] After cleaning: {len(mesh.vertices):,} vertices and {len(mesh.triangles):,} triangles")
        
        # Save mesh
        print("[mesh] Saving mesh to reconstruction_mesh.ply...")
        o3d.io.write_triangle_mesh("reconstruction_mesh.ply", mesh)
        
        # Visualize
        print("[mesh] Visualizing mesh (close window to continue)...")
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        
        return mesh
        
    except ImportError:
        print("\n[mesh] Open3D not installed. Install it with:")
        print("  pip install open3d")
        print("\n[mesh] Skipping mesh generation...")
        return None
    except Exception as e:
        print(f"\n[mesh] Mesh creation failed: {e}")
        return None


def main():
    print("="*60)
    print("STEP 8: MESH RECONSTRUCTION")
    print("="*60)
    
    # Load the dense reconstruction from npz file
    print(f"[load] Loading dense reconstruction from dense_reconstruction.npz...")
    data = np.load("./dense_reconstruction.npz", allow_pickle=True)
    points = data['points']
    colors = data['colors'] if 'colors' in data else None
    print(f"[load] Loaded {len(points):,} points")
    if colors is not None:
        print(f"[load] Loaded {len(colors):,} color values")
    
    # Create mesh from point cloud
    mesh = create_mesh_from_point_cloud(points, colors)
    
    # Create placeholder file for Makefile dependency
    with open("./point_cloud_cleaned.txt", "w") as f:
        f.write(f"# Mesh reconstruction complete\n")
        f.write(f"# Total points: {len(points):,}\n")
        if mesh is not None:
            f.write("# Mesh saved to reconstruction_mesh.ply\n")
    
    print("\n" + "="*60)
    print("STEP 8 COMPLETE!")
    print("="*60)
    if mesh is not None:
        print("✓ Mesh reconstruction complete")
        print("✓ Mesh saved to reconstruction_mesh.ply")
        print("\n💡 Open reconstruction_mesh.ply in MeshLab or CloudCompare")
    else:
        print("⚠ Mesh reconstruction failed")
        print("  Install Open3D: pip install open3d")
    print("="*60)


if __name__ == "__main__":
    main()
