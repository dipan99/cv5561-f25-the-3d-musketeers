.PHONY: all clean features bootstrap pose_triangulation reconstruction incremental visualize dense help run

# Python interpreter
PYTHON := python3

# Generated files
FEATURES := step1_features.npz
BOOTSTRAP := bootstrap_data.npz
RECONSTRUCTION := reconstruction_initial.npz
POINT_CLOUD := point_cloud_initial.txt
FINAL_POINT_CLOUD := point_cloud_final.txt
DENSE := dense_reconstruction.npz

# Default target
all: run

# Help message
help:
	@echo "SfM Pipeline Makefile"
	@echo "====================="
	@echo ""
	@echo "Targets:"
	@echo "  make run          - Run the complete SfM pipeline"
	@echo "  make features     - Step 1: Extract features and match"
	@echo "  make bootstrap    - Step 2: Bootstrap reconstruction"
	@echo "  make pose         - Step 3: Pose estimation and triangulation"
	@echo "  make reconstruction - Step 4: Update reconstruction state"
	@echo "  make incremental  - Step 5: Incremental reconstruction"
	@echo "  make visualize    - Step 6: Visualize point cloud"
	@echo "  make dense        - Step 7: Dense reconstruction"
	@echo "  make clean        - Remove all generated files"
	@echo "  make help         - Show this help message"
	@echo ""

# Run complete pipeline
run: dense
	@echo ""
	@echo "========================================"
	@echo "SfM Pipeline Complete!"
	@echo "========================================"

# Step 1: Feature extraction and matching
features: $(FEATURES)

$(FEATURES):
	@echo ""
	@echo "[Step 1/7] Extracting features..."
	$(PYTHON) sfm_step1_features.py
	@echo "✓ Step 1 complete"

# Step 2: Bootstrap reconstruction
bootstrap: $(BOOTSTRAP)

$(BOOTSTRAP): $(FEATURES)
	@echo ""
	@echo "[Step 2/7] Bootstrapping reconstruction..."
	$(PYTHON) sfm_step2_bootstrap.py
	@echo "✓ Step 2 complete"

# Step 3: Pose estimation and triangulation
pose: $(RECONSTRUCTION)

$(RECONSTRUCTION): $(BOOTSTRAP)
	@echo ""
	@echo "[Step 3/7] Pose estimation and triangulation..."
	$(PYTHON) sfm_step3_pose_triangulation.py
	@echo "✓ Step 3 complete"

# Step 4: Update reconstruction state
reconstruction: $(POINT_CLOUD)

$(POINT_CLOUD): $(RECONSTRUCTION)
	@echo ""
	@echo "[Step 4/7] Updating reconstruction state..."
	$(PYTHON) sfm_step4_reconstruction_state.py
	@echo "✓ Step 4 complete"

# Step 5: Incremental reconstruction
incremental: $(FINAL_POINT_CLOUD)

$(FINAL_POINT_CLOUD): $(POINT_CLOUD)
	@echo ""
	@echo "[Step 5/7] Running incremental reconstruction..."
	$(PYTHON) sfm_step5_incremental.py
	@echo "✓ Step 5 complete"

# Step 6: Visualization
visualize: $(FINAL_POINT_CLOUD)
	@echo ""
	@echo "[Step 6/7] Generating visualization..."
	$(PYTHON) sfm_step6_visualize.py
	@echo "✓ Step 6 complete"

# Step 7: Dense reconstruction
dense: $(DENSE)

$(DENSE): $(FINAL_POINT_CLOUD)
	@echo ""
	@echo "[Step 7/7] Running dense reconstruction..."
	$(PYTHON) sfm_step7_dense_reconstruction.py
	@echo "✓ Step 7 complete"

# Clean up generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f *.npz *.txt
	@rm -f *.png *.jpg
	@rm -f *.txt *.pkl *.ply
	@echo "✓ Clean complete"

# Clean and rerun
rebuild: clean all