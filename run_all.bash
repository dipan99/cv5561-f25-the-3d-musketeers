#!/bin/bash
# Run all SfM pipeline steps sequentially with error checking

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "Starting SfM Pipeline"
echo "========================================"

# Step 1: Feature extraction
echo ""
echo "[Step 1/7] Extracting features..."
python3 sfm_step1_features.py
if [ $? -ne 0 ]; then
    echo "ERROR: Feature extraction failed"
    exit 1
fi
echo "✓ Step 1 complete"

# Step 2: Bootstrap
echo ""
echo "[Step 2/7] Bootstrapping reconstruction..."
python3 sfm_step2_bootstrap.py
if [ $? -ne 0 ]; then
    echo "ERROR: Bootstrap failed"
    exit 1
fi
echo "✓ Step 2 complete"

# Step 3: Pose estimation and triangulation
echo ""
echo "[Step 3/7] Pose estimation and triangulation..."
python3 sfm_step3_pose_triangulation.py
if [ $? -ne 0 ]; then
    echo "ERROR: Pose triangulation failed"
    exit 1
fi
echo "✓ Step 3 complete"

# Step 4: Update reconstruction state
echo ""
echo "[Step 4/7] Updating reconstruction state..."
python3 sfm_step4_reconstruction_state.py
if [ $? -ne 0 ]; then
    echo "ERROR: Reconstruction state update failed"
    exit 1
fi
echo "✓ Step 4 complete"

# Step 5: Incremental reconstruction
echo ""
echo "[Step 5/7] Running incremental reconstruction..."
python3 sfm_step5_incremental.py
if [ $? -ne 0 ]; then
    echo "ERROR: Incremental reconstruction failed"
    exit 1
fi
echo "✓ Step 5 complete"

# Step 6: Visualization
echo ""
echo "[Step 6/7] Generating visualization..."
python3 sfm_step6_visualize.py
if [ $? -ne 0 ]; then
    echo "ERROR: Visualization failed"
    exit 1
fi
echo "✓ Step 6 complete"

# Step 7: Dense reconstruction
echo ""
echo "[Step 7/7] Running dense reconstruction..."
python3 sfm_step7_dense_reconstruction.py
if [ $? -ne 0 ]; then
    echo "ERROR: Dense reconstruction failed"
    exit 1
fi
echo "✓ Step 7 complete"

echo ""
echo "========================================"
echo "SfM Pipeline Complete!"
echo "========================================"



