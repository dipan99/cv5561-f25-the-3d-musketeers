# Project Setup Guide

This project uses **Pipenv** for dependency management to ensure consistent environments across different machines.

## Prerequisites

- Python 3.12
- Homebrew (macOS) or appropriate package manager for your OS
- COLMAP (for SfM processing)

## Quick Start

### 1. Install Pipenv

```bash
# macOS/Linux
pip install --user pipenv

# Or using Homebrew on macOS
brew install pipenv
```

### 2. Install COLMAP

```bash
# macOS
brew install colmap

# Linux (Ubuntu/Debian)
sudo apt-get install colmap

# Windows
# Download from https://github.com/colmap/colmap/releases
```

### 3. Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/dipan99/cv5561-f25-the-3d-musketeers.git
cd cv5561-f25-the-3d-musketeers

# Install dependencies with Pipenv
pipenv install

# Activate the virtual environment
pipenv shell
```

## Usage

### Running Scripts

Once you're in the Pipenv shell, you can run any script:

```bash
# Using pycolmap for SfM preparation
python prepare_for_3dgs.py

# Using custom SfM implementation
python prepare_for_3dgs_own_sfm.py

# Full pipeline with depth estimation
python full_pipeline_depth_estimation.py

# RGBD reconstruction (if you have depth maps)
python rgbd_reconstruction.py
```

### Running Without Entering Shell

You can also run scripts without activating the shell:

```bash
pipenv run python prepare_for_3dgs.py
```

## Development

### Adding New Dependencies

```bash
# Add a production dependency
pipenv install package-name

# Add a development dependency
pipenv install --dev package-name

# Install from a specific version
pipenv install package-name~=1.2.0
```

### Updating Dependencies

```bash
# Update all packages
pipenv update

# Update a specific package
pipenv update package-name
```

### Checking for Security Vulnerabilities

```bash
pipenv check
```

## Project Structure

```
.
├── Pipfile                          # Dependency declarations
├── Pipfile.lock                     # Locked dependency versions (auto-generated)
├── prepare_for_3dgs.py              # SfM prep using pycolmap
├── prepare_for_3dgs_own_sfm.py      # Custom SfM implementation
├── full_pipeline_depth_estimation.py # SfM + dense stereo
├── rgbd_reconstruction.py           # RGBD fusion pipeline
└── README.md                        # Project documentation
```

## Configuration

### Dataset Selection

Both main scripts have a configuration section at the top:

```python
USE_STANFORD = True  # Set to False for custom images

if USE_STANFORD:
    DATA_DIR = "./stanford_extracted/seq-01"
    # ...
else:
    DATA_DIR = "./images2"
    # ...
```

### Parameters

Adjust these in the scripts as needed:
- `FRAME_SKIP`: Sample every Nth frame
- `MAX_FRAMES`: Maximum number of frames to process
- `OUTPUT_DIR`: Where to save results

## Troubleshooting

### Pipenv Not Found

If `pipenv` command is not found after installation:

```bash
# Add to your PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"
source ~/.zshrc  # or ~/.bashrc
```

### Python Version Mismatch

If you don't have Python 3.12:

```bash
# macOS
brew install python@3.12

# Then specify the Python version
pipenv --python 3.12
pipenv install
```

### COLMAP Not Found

Make sure COLMAP is installed and in your PATH:

```bash
which colmap
colmap -h
```

### Memory Issues

For large datasets, reduce `MAX_FRAMES` or `FRAME_SKIP` values in the scripts.

## Outputs

### SfM Preparation Scripts

Both `prepare_for_3dgs.py` and `prepare_for_3dgs_own_sfm.py` generate:

```
sfm_for_3dgs_stanford/
├── images/              # Input images
├── sparse/0/            # COLMAP reconstruction
│   ├── cameras.bin     # Camera intrinsics
│   ├── images.bin      # Camera poses
│   └── points3D.bin    # Sparse 3D points
├── sparse_points.ply   # Visualization
└── README.txt          # Instructions for 3DGS
```

Transfer this entire folder to a GPU machine for 3D Gaussian Splatting training.

## Next Steps (GPU Machine)

### Install 3D Gaussian Splatting

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
conda env create -f environment.yml
conda activate gaussian_splatting
```

### Train

```bash
python train.py -s /path/to/sfm_for_3dgs_stanford -m output/model
```

### Render

```bash
python render.py -m output/model
```

## Additional Resources

- [Pipenv Documentation](https://pipenv.pypa.io/)
- [COLMAP Documentation](https://colmap.github.io/)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Nerfstudio](https://docs.nerf.studio/)

## License

See LICENSE file for details.

## Contributors

The 3D Musketeers Team
