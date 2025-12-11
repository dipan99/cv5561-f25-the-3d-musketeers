# 3D Room Reconstruction using Structure from Motion and Gaussian Splatting

## Team

- **Dipan Bag** — [bag00003@umn.edu](mailto:bag00003@umn.edu)  
  *Lead: Structure from Motion Pipeline*

- **Cheston Opsasnick** — [opsas002@umn.edu](mailto:opsas002@umn.edu)  
  *Lead: 3D Gaussian Splatting Integration*

- **Lulin Liu** — [liu02721@umn.edu](mailto:liu02721@umn.edu)  
  *Lead: Evaluation and Analysis*

## Project Overview

Build an end-to-end 3D reconstruction pipeline from 2D images to photorealistic 3D scenes, comparing classical geometry-based (custom SfM) versus learned prior-based (VGGT) approaches.

## Quick Start

### Google Colab

1. **Open the notebook**: [`SfM_3DGS_Final_Notebook.ipynb`](./SfM_3DGS_Final_Notebook.ipynb)
2. **Upload to Google Colab**: File → Upload notebook
3. **Connect to GPU runtime**: Runtime → Change runtime type → GPU (T4 or A100)
4. **Run all cells sequentially**

The notebook includes:
- Dataset download and preparation
- Custom GPU-accelerated SfM pipeline
- 3D Gaussian Splatting training
- Evaluation and visualization


### Pipeline Workflows

**Structure from Motion Pipeline:**

![SfM Pipeline](/sfm_workflow.jpg)

**3D Gaussian Splatting Process:**

![3DGS Process](/3dgs_workflow.jpg)



## Pipeline Components

### Stage 1: Structure from Motion
- **Feature Detection**: SIFT (8,192 keypoints/image)
- **Feature Matching**: Lowe's ratio test (threshold: 0.7)
- **Initialization**: Essential Matrix bootstrap from best image pair
- **Incremental Registration**: PnP + RANSAC
- **Triangulation**: Multi-view 3D point reconstruction
- **Bundle Adjustment**: GPU-accelerated (PyTorch/CUDA with Adam optimizer)
- **Export**: COLMAP-compatible format

### Stage 2: 3D Gaussian Splatting
- **Initialization**: Sparse point cloud from SfM
- **Training**: 30,000 iterations
- **Loss Function**: ℒ = 0.8·L1 + 0.2·D-SSIM
- **Adaptive Densification**: Clone/split Gaussians in under-reconstructed regions

### Stage 3: Evaluation
- **Metrics**: PSNR, SSIM, LPIPS on held-out test views
- **Visualization**: Interactive 3D rendering

## Results

### Quantitative Comparison

| Method | Dataset | SSIM ↑ | PSNR ↑ | LPIPS ↓ | SfM Init Time | Training Time |
|--------|---------|--------|--------|---------|---------------|---------------|
| **Our Custom SfM + 3DGS** | Microsoft RGB-D Indoor & Our Own | 0.5124 | 19.0457 | 0.7407 | 15 mins | 30 mins (30k iter) |
| **VGGT (518×518) + 3DGS** | Microsoft RGB-D Indoor & Our Own | 0.7513 | 23.6469 | 0.17524 | 11s (all included)<br>0.2s (infer only) | 2 mins (2k iter) |

### Key Findings

1. **Learned priors outperform classical SfM**: VGGT achieves 24% higher PSNR (23.6 vs 19.0) with 50× faster initialization (11s vs 15min)
2. **GPU acceleration is essential**: PyTorch-based bundle adjustment makes optimization practical for large-scale scenes
3. **Format precision matters**: Correct COLMAP binary export (null-terminated strings) is critical for 3DGS integration
4. **Bundle adjustment is non-optional**: Skipping BA resulted in blurry reconstructions with significant drift

## Configuration

Key parameters in the SfM pipeline (edit in notebook or script):
```python
DATA_DIR = "./raw_1fps/raw_1fps/frames"
OUTPUT_DIR = "./sfm_for_3dgs_custom_sfm"
IMAGE_PATTERN = "frame_*.png"
FRAME_SKIP = 1
MAX_FRAMES = 25
USE_GPU = True  # Enable GPU-accelerated bundle adjustment
```

## Citations

### Structure from Motion
```bibtex
@inproceedings{snavely2006photo,
  title={Photo tourism: exploring photo collections in 3D},
  author={Snavely, Noah and Seitz, Steven M and Szeliski, Richard},
  booktitle={ACM SIGGRAPH 2006 Papers},
  pages={835--846},
  year={2006}
}

@book{hartley2003multiple,
  title={Multiple view geometry in computer vision},
  author={Hartley, Richard and Zisserman, Andrew},
  year={2003},
  publisher={Cambridge university press}
}

@inproceedings{triggs1999bundle,
  title={Bundle adjustment—a modern synthesis},
  author={Triggs, Bill and McLauchlan, Philip F and Hartley, Richard I and Fitzgibbon, Andrew W},
  booktitle={International workshop on vision algorithms},
  pages={298--372},
  year={1999},
  organization={Springer}
}
```

### 3D Gaussian Splatting
```bibtex
@article{kerbl20233d,
  title={3d gaussian splatting for real-time radiance field rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023}
}
```

### Distributed 3DGS (Grendel)
```bibtex
@inproceedings{zhao2025grendel,
  title={On Scaling Up 3D Gaussian Splatting Training},
  author={Zhao, Hexu and Weng, Haoyang and Lu, Daohan and Li, Ang and Li, Jinyang and Panda, Aurojit and Xie, Saining},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

### VGGT (Comparison Baseline)
```bibtex
@inproceedings{wang2024vggt,
  title={VGGT: Efficient Gaussian Geometry from 3D Transformers},
  author={Wang, Peng and others},
  booktitle={CVPR},
  year={2024}
}
```

## Datasets

- **Microsoft RGB-D Indoor Dataset**: Standard benchmark for indoor scene reconstruction
- **Custom Captures**: Self-captured workspace scenes

## Known Limitations

1. **Scale Ambiguity**: Monocular reconstruction has arbitrary scale (relative to first camera pair baseline)
2. **Texture-less Regions**: SIFT struggles with plain walls and uniform surfaces
3. **Limited Coverage**: Quality improves with more diverse viewpoints
4. **Computational Cost**: Bundle adjustment is expensive without GPU acceleration

## Future Work

- Integrate learned depth priors (e.g., VGGT) for initialization
- Implement multi-GPU distributed training
- Add real-time reconstruction capability
- Support depth sensor input for scale recovery

## Acknowledgments

Based on the classical SfM pipeline (Snavely et al., 2006) and 3D Gaussian Splatting (Kerbl et al., 2023). GPU acceleration inspired by distributed training techniques from Grendel (Zhao et al., 2025).

## License

This project is for educational purposes (CSCI 5561 - Computer Vision). See individual component licenses:
- 3D Gaussian Splatting: [Inria License](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)