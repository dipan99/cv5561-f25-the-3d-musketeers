"""
Comparison tool to analyze and compare different SfM implementations.

Compares:
1. Custom implementation (your pipeline)
2. OpenCV SfM (if available)
3. COLMAP (if available)
"""

import os
import numpy as np
import pickle
from pathlib import Path


def load_custom_results():
    """Load results from custom implementation."""
    print("="*60)
    print("LOADING CUSTOM IMPLEMENTATION RESULTS")
    print("="*60)
    
    results = {}
    
    try:
        # Load feature data
        if os.path.exists("./step1_features.npz"):
            features = np.load("./step1_features.npz", allow_pickle=True)
            results['num_images'] = int(features['num_images'])
            results['pair_matches'] = features['pair_matches'].item()
            print(f"✓ Features: {results['num_images']} images")
        
        # Load final reconstruction
        if os.path.exists("./reconstruction_final.pkl"):
            with open("./reconstruction_final.pkl", 'rb') as f:
                state = pickle.load(f)
            
            # Handle both dict and object formats
            if hasattr(state, 'cameras'):
                results['cameras'] = state.cameras
                results['points_3d'] = np.array(state.points_3d)
                results['K'] = state.K
            else:
                results['cameras'] = state['cameras']
                results['points_3d'] = np.array(state['points_3d'])
                results['K'] = state['K']
            
            num_registered = len([c for c in results['cameras'].values() if c['registered']]) if isinstance(results['cameras'], dict) else len([c for c in results['cameras'] if c.get('registered', False)])
            print(f"✓ Reconstruction: {num_registered} cameras, {len(results['points_3d'])} points")
        
        # Load dense reconstruction
        if os.path.exists("./dense_reconstruction.npz"):
            dense = np.load("./dense_reconstruction.npz", allow_pickle=True)
            results['dense_points'] = dense['points']
            results['dense_colors'] = dense['colors']
            print(f"✓ Dense reconstruction: {len(results['dense_points'])} points")
        
        return results
    
    except Exception as e:
        print(f"✗ Error loading custom results: {e}")
        return None


def load_opencv_results():
    """Load results from OpenCV SfM."""
    print("\n" + "="*60)
    print("LOADING OPENCV SFM RESULTS")
    print("="*60)
    
    ply_path = "./opencv_sfm_output/opencv_reconstruction.ply"
    
    if not os.path.exists(ply_path):
        print("✗ OpenCV results not found")
        print("  Run: python3 sfm_opencv_comparison.py")
        return None
    
    try:
        # Parse PLY file
        points = []
        with open(ply_path, 'r') as f:
            in_header = True
            for line in f:
                if line.startswith('end_header'):
                    in_header = False
                    continue
                if not in_header:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        points = np.array(points)
        print(f"✓ Loaded {len(points)} points from OpenCV reconstruction")
        
        return {'points_3d': points}
    
    except Exception as e:
        print(f"✗ Error loading OpenCV results: {e}")
        return None


def load_colmap_results():
    """Load results from COLMAP."""
    print("\n" + "="*60)
    print("LOADING COLMAP RESULTS")
    print("="*60)
    
    points_file = "./colmap_output/sparse/0/points3D.txt"
    
    if not os.path.exists(points_file):
        print("✗ COLMAP results not found")
        print("  Run: python3 sfm_colmap_comparison.py")
        return None
    
    try:
        points = []
        with open(points_file, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        points.append([x, y, z])
        
        points = np.array(points)
        print(f"✓ Loaded {len(points)} points from COLMAP reconstruction")
        
        return {'points_3d': points}
    
    except Exception as e:
        print(f"✗ Error loading COLMAP results: {e}")
        return None


def analyze_point_cloud(points, name="Point Cloud"):
    """Analyze and print statistics for a point cloud."""
    if len(points) == 0:
        print(f"  {name}: No points")
        return
    
    print(f"\n{name} Statistics:")
    print(f"  Number of points: {len(points):,}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print(f"  Mean position: [{points.mean(axis=0)[0]:.2f}, {points.mean(axis=0)[1]:.2f}, {points.mean(axis=0)[2]:.2f}]")
    print(f"  Std deviation: [{points.std(axis=0)[0]:.2f}, {points.std(axis=0)[1]:.2f}, {points.std(axis=0)[2]:.2f}]")
    
    # Calculate bounding box volume
    ranges = points.max(axis=0) - points.min(axis=0)
    volume = ranges[0] * ranges[1] * ranges[2]
    print(f"  Bounding box volume: {volume:.2f} cubic units")
    print(f"  Point density: {len(points)/volume:.2f} points/cubic unit")


def compare_results(custom, opencv, colmap):
    """Compare results from all implementations."""
    print("\n" + "="*70)
    print("DETAILED COMPARISON")
    print("="*70)
    
    # Comparison table
    print("\n" + "-"*70)
    print(f"{'Metric':<30} {'Custom':<15} {'OpenCV':<15} {'COLMAP':<15}")
    print("-"*70)
    
    # Cameras
    if custom and 'cameras' in custom:
        if isinstance(custom['cameras'], dict):
            custom_cams = len([c for c in custom['cameras'].values() if c.get('registered', False)])
        else:
            custom_cams = len([c for c in custom['cameras'] if c.get('registered', False)])
    else:
        custom_cams = "N/A"
    print(f"{'Registered Cameras':<30} {str(custom_cams):<15} {'2':<15} {'N/A':<15}")
    
    # Sparse points
    custom_sparse = len(custom['points_3d']) if custom and 'points_3d' in custom else "N/A"
    opencv_sparse = len(opencv['points_3d']) if opencv and 'points_3d' in opencv else "N/A"
    colmap_sparse = len(colmap['points_3d']) if colmap and 'points_3d' in colmap else "N/A"
    print(f"{'Sparse 3D Points':<30} {str(custom_sparse):<15} {str(opencv_sparse):<15} {str(colmap_sparse):<15}")
    
    # Dense points
    if custom and 'dense_points' in custom:
        custom_dense = len(custom['dense_points'])
    else:
        custom_dense = "N/A"
    print(f"{'Dense 3D Points':<30} {str(custom_dense):<15} {'N/A':<15} {'N/A':<15}")
    
    print("-"*70)
    
    # Detailed analysis for each
    if custom and 'points_3d' in custom:
        analyze_point_cloud(custom['points_3d'], "Custom Implementation (Sparse)")
    
    if custom and 'dense_points' in custom:
        analyze_point_cloud(custom['dense_points'], "Custom Implementation (Dense)")
    
    if opencv and 'points_3d' in opencv:
        analyze_point_cloud(opencv['points_3d'], "OpenCV SfM")
    
    if colmap and 'points_3d' in colmap:
        analyze_point_cloud(colmap['points_3d'], "COLMAP")
    
    # Recommendations
    print("\n" + "="*70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*70)
    
    if custom and 'points_3d' in custom:
        sparse_points = len(custom['points_3d'])
        if sparse_points < 1000:
            print("⚠ Low number of sparse points detected")
            print("  Recommendations:")
            print("  - Try MATCH_ALL_PAIRS = True in sfm_step1_features.py")
            print("  - Reduce MATCH_THRESHOLD (currently 50)")
            print("  - Increase nfeatures in SIFT (currently 10000)")
        elif sparse_points > 10000:
            print("✓ Good number of sparse points!")
        else:
            print("✓ Reasonable number of sparse points")
        
        if 'dense_points' in custom:
            dense_points = len(custom['dense_points'])
            if dense_points < sparse_points * 10:
                print("⚠ Dense reconstruction may need tuning")
                print("  Consider adjusting StereoSGBM parameters")
            else:
                print(f"✓ Good dense reconstruction ({dense_points/sparse_points:.1f}x sparse)")
    
    # Quality metrics
    print("\n" + "="*70)
    print("QUALITY METRICS")
    print("="*70)
    
    if custom and 'cameras' in custom:
        if isinstance(custom['cameras'], dict):
            registered = [c for c in custom['cameras'].values() if c.get('registered', False)]
            total = len(custom['cameras'])
        else:
            registered = [c for c in custom['cameras'] if c.get('registered', False)]
            total = len(custom['cameras'])
        registration_rate = len(registered) / total * 100
        
        print(f"Camera Registration Rate: {registration_rate:.1f}% ({len(registered)}/{total})")
        
        if registration_rate < 50:
            print("  ⚠ Low registration rate - many cameras failed to register")
        elif registration_rate < 80:
            print("  ⚙ Moderate registration rate - some cameras failed")
        else:
            print("  ✓ Good registration rate!")
    
    if custom and 'points_3d' in custom:
        # Estimate track quality (if available in future)
        print(f"\nPoint Cloud Coverage:")
        points = custom['points_3d']
        ranges = points.max(axis=0) - points.min(axis=0)
        coverage = ranges[0] * ranges[1] * ranges[2]
        print(f"  Spatial coverage: {coverage:.2f} cubic units")


def export_comparison_report(custom, opencv, colmap):
    """Export comparison results to a text file."""
    report_path = "./comparison_report.txt"
    
    print(f"\n" + "="*70)
    print(f"Exporting comparison report to: {report_path}")
    print("="*70)
    
    with open(report_path, 'w') as f:
        f.write("SfM IMPLEMENTATION COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        
        if custom:
            f.write("CUSTOM IMPLEMENTATION:\n")
            if 'cameras' in custom:
                if isinstance(custom['cameras'], dict):
                    reg = len([c for c in custom['cameras'].values() if c.get('registered', False)])
                else:
                    reg = len([c for c in custom['cameras'] if c.get('registered', False)])
                f.write(f"  Cameras: {reg}\n")
            if 'points_3d' in custom:
                f.write(f"  Sparse points: {len(custom['points_3d'])}\n")
            if 'dense_points' in custom:
                f.write(f"  Dense points: {len(custom['dense_points'])}\n")
            f.write("\n")
        
        if opencv:
            f.write("OPENCV SFM:\n")
            if 'points_3d' in opencv:
                f.write(f"  Points: {len(opencv['points_3d'])}\n")
            f.write("\n")
        
        if colmap:
            f.write("COLMAP:\n")
            if 'points_3d' in colmap:
                f.write(f"  Points: {len(colmap['points_3d'])}\n")
            f.write("\n")
    
    print(f"✓ Report saved to {report_path}")


def main():
    """Main comparison function."""
    print("\n" + "="*70)
    print("SfM IMPLEMENTATION COMPARISON TOOL")
    print("="*70)
    
    # Load all results
    custom = load_custom_results()
    opencv = load_opencv_results()
    colmap = load_colmap_results()
    
    # Compare
    compare_results(custom, opencv, colmap)
    
    # Export report
    export_comparison_report(custom, opencv, colmap)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\n💡 Files to visualize in MeshLab/CloudCompare:")
    print("  • Custom (sparse): ./reconstruction_final.ply")
    print("  • Custom (dense):  ./dense_pointcloud.ply")
    print("  • OpenCV:          ./opencv_sfm_output/opencv_reconstruction.ply")
    print("  • COLMAP:          ./colmap_output/colmap_sparse.ply")
    print("\n📊 Comparison report: ./comparison_report.txt")
    print("="*70)


if __name__ == "__main__":
    main()
