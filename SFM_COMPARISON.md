# SfM Library Comparison

This directory contains tools to compare your custom Structure from Motion (SfM) implementation with standard libraries.

## Available Comparisons

### 1. Custom Implementation (Your Pipeline)
**Files:** `sfm_step*.py`, `Makefile`

Your custom implementation that processes images through:
- Feature extraction (SIFT)
- Feature matching
- Bootstrap reconstruction
- Incremental camera registration
- Dense reconstruction
- Mesh generation

**Run:**
```bash
make all
```

**Results:**
- Sparse reconstruction: `reconstruction_final.ply` (3,505 points)
- Dense reconstruction: `dense_pointcloud.ply` (373,037 points)
- Mesh: `reconstruction_mesh.ply`
- 50/50 cameras registered (100%)

---

### 2. OpenCV SfM Comparison
**File:** `sfm_opencv_comparison.py`

Uses OpenCV's built-in functions for SfM reconstruction.

**Run:**
```bash
python3 sfm_opencv_comparison.py
```

**Results:**
- Point cloud: `opencv_sfm_output/opencv_reconstruction.ply` (21 points)
- 2 cameras registered

**Note:** This is a simplified implementation for comparison. OpenCV's SfM module has limited functionality compared to full pipelines.

---

### 3. COLMAP Comparison
**File:** `sfm_colmap_comparison.py`

Uses COLMAP, the industry-standard SfM pipeline.

**Installation:**
```bash
# macOS
brew install colmap

# Ubuntu/Debian
sudo apt install colmap

# Or use Python bindings
pip install pycolmap
```

**Run:**
```bash
python3 sfm_colmap_comparison.py
```

**Results:**
- Sparse reconstruction: `colmap_output/sparse/0/`
- Point cloud: `colmap_output/colmap_sparse.ply`

**Note:** COLMAP is considered the gold standard for SfM. It's used in production by many research labs and companies.

---

### 4. Comprehensive Comparison Tool
**File:** `compare_sfm_results.py`

Loads and compares results from all implementations.

**Run:**
```bash
python3 compare_sfm_results.py
```

**Output:**
- Console comparison table
- Detailed statistics for each implementation
- Quality metrics and recommendations
- `comparison_report.txt` with summary

---

## Comparison Results

### Current Results (heads/seq-01 dataset, 50 images, every 10th frame)

| Metric | Custom | OpenCV | COLMAP |
|--------|--------|--------|--------|
| **Registered Cameras** | 50 | 2 | N/A |
| **Sparse 3D Points** | 3,505 | 21 | N/A |
| **Dense 3D Points** | 373,037 | N/A | N/A |
| **Registration Rate** | 100% | 4% | N/A |

### Custom Implementation Strengths:
✅ **Complete pipeline** - All 50 cameras registered
✅ **Rich sparse reconstruction** - 3,505 points with good coverage
✅ **Dense reconstruction** - 106x denser than sparse
✅ **Mesh generation** - Full 3D mesh output

### Sparse Point Cloud Statistics:
- **Spatial coverage:** 57,026 cubic units
- **X range:** [-19.11, 19.82]
- **Y range:** [-15.87, 13.81]  
- **Z range:** [0.39, 49.74]
- **Mean position:** [1.57, -0.98, 18.76]

---

## Visualization

To view and compare the point clouds:

### Option 1: MeshLab (Free)
```bash
# macOS
brew install --cask meshlab

# Ubuntu
sudo apt install meshlab
```

Open multiple files:
```bash
meshlab reconstruction_final.ply dense_pointcloud.ply opencv_sfm_output/opencv_reconstruction.ply
```

### Option 2: CloudCompare (Free)
```bash
# macOS
brew install --cask cloudcompare

# Ubuntu
sudo snap install cloudcompare
```

### Option 3: Python Visualization
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load PLY file
points = []
with open('reconstruction_final.ply', 'r') as f:
    in_header = True
    for line in f:
        if line.startswith('end_header'):
            in_header = False
            continue
        if not in_header:
            parts = line.strip().split()
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])

points = np.array(points)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

---

## Configuration

Adjust comparison settings in each file:

```python
# All comparison scripts
IMAGE_DIR = "./heads/seq-01"
SKIP_EVERY_N = 10  # Use every Nth image
MAX_IMAGES = 50    # Maximum images to process
```

---

## Benchmarking

### Speed Comparison (50 images)
- **Custom Implementation:** ~2-3 minutes
- **OpenCV:** ~30 seconds (but only 2 cameras registered)
- **COLMAP:** ~5-10 minutes (full reconstruction)

### Quality Comparison
- **Custom:** Good for small-medium datasets, 100% registration
- **OpenCV:** Limited functionality, low success rate
- **COLMAP:** Best quality, most robust, industry standard

---

## Recommendations

### For Learning:
✅ **Your custom implementation** - Best for understanding SfM algorithms

### For Quick Prototyping:
✅ **Your custom implementation** - Fast and configurable

### For Production/Research:
✅ **COLMAP** - Most accurate and robust
- Better bundle adjustment
- More sophisticated matching
- Loop closure detection
- Large-scale scene handling

### For Real-time Applications:
⚠️ Consider specialized libraries like ORB-SLAM3 or OpenVSLAM

---

## Further Improvements

To improve your custom implementation to match COLMAP:

1. **Bundle Adjustment:**
   - Implement Levenberg-Marquardt optimization
   - Minimize reprojection error globally
   - See: scipy.optimize.least_squares

2. **Better Matching:**
   - Implement vocabulary tree matching
   - Add loop closure detection
   - Use FLANN for faster matching

3. **Filtering:**
   - Add track length filtering
   - Implement RANSAC for all estimation steps
   - Better outlier rejection

4. **Dense Reconstruction:**
   - Use PatchMatch Stereo (COLMAP's method)
   - Better depth map fusion
   - Normal estimation for better meshing

---

## References

- **COLMAP:** Schönberger & Frahm, "Structure-from-Motion Revisited", CVPR 2016
- **OpenCV SfM:** https://docs.opencv.org/master/d8/d8c/group__sfm.html
- **Multiple View Geometry:** Hartley & Zisserman, 2003
- **Bundle Adjustment:** Triggs et al., "Bundle Adjustment — A Modern Synthesis", 1999

---

## Troubleshooting

### "COLMAP not found"
Install COLMAP or use pycolmap:
```bash
pip install pycolmap
```

### "OpenCV SfM module not found"
Install opencv-contrib:
```bash
pip install opencv-contrib-python
```

### "Low registration rate"
Try in `sfm_step1_features.py`:
```python
MATCH_ALL_PAIRS = True
MATCH_THRESHOLD = 30  # Lower threshold
```

### "Dense reconstruction too slow"
Reduce image resolution or skip more frames:
```python
SKIP_EVERY_N = 20  # Use every 20th image
```
