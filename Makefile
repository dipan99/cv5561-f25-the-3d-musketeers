.PHONY: all clean features bootstrap pose_triangulation reconstruction incremental visualize dense help run install shell update check run-3dgs run-custom-3dgs

# Python interpreter
PYTHON := pipenv run python

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
	@echo "Setup:"
	@echo "  make install      - Install dependencies with Pipenv"
	@echo "  make shell        - Activate Pipenv shell"
	@echo "  make update       - Update all dependencies"
	@echo "  make check        - Check for security vulnerabilities"
	@echo ""
	@echo "3DGS Preparation:"
	@echo "  make run-3dgs     - Run SfM preparation (pycolmap)"
	@echo "  make run-custom-3dgs - Run custom SfM implementation"
	@echo ""
	@echo "Original Pipeline:"
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

# Setup commands
install:
	@echo "Installing dependencies with Pipenv..."
	pipenv install
	@echo "Done! Run 'make shell' to activate the environment."

shell:
	pipenv shell

update:
	@echo "Updating dependencies..."
	pipenv update

check:
	pipenv check

# 3DGS preparation commands
run-3dgs:
	@echo "Running SfM preparation with pycolmap..."
	$(PYTHON) prepare_for_3dgs.py

run-custom-3dgs:
	@echo "Running custom SfM implementation..."
	$(PYTHON) prepare_for_3dgs_own_sfm.py

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
	@rm -f *.npz
	@rm -f *.pkl *.ply
	@rm -rf __pycache__
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf sfm_for_3dgs_*
	@rm -rf reconstruction_output
	@rm -f database.db
	@echo "✓ Clean complete"

# Clean and rerun
rebuild: clean all