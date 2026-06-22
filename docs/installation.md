# Installation

## Requirements

- Python 3.11+
- Conda (recommended) or virtualenv
- CUDA 13.2 for RTX 5070 Ti / sm_120 (or CUDA 12.1+ for older GPUs)

## Conda environment

**CPU only:**
```bash
conda create -n rseg python=3.11 -y
conda activate rseg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scikit-learn scipy omegaconf pillow tqdm torchgeo
```

**GPU — CUDA 13.2 (RTX 5070 Ti, sm_120):**
```bash
conda create -n rseg python=3.11 -y
conda activate rseg
pip install torch==2.12.1+cu132 torchvision --index-url https://download.pytorch.org/whl/cu132
pip install numpy scikit-learn scipy omegaconf pillow tqdm torchgeo rasterio
```

**GPU — CUDA 12.1 (RTX 3090/4090 and older):**
```bash
conda create -n rseg python=3.11 -y
conda activate rseg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn scipy omegaconf pillow tqdm torchgeo rasterio
```

Or use the Makefile shortcuts:
```bash
make env        # CPU
make env-gpu    # GPU (CUDA 13.2)
```

## Sentinel-2 support

`rasterio` is required only for Sentinel-2 GeoTIFF loading:
```bash
pip install rasterio
```

## DINOv2 weights

Weights are downloaded automatically from Meta AI on first run and cached in:
```
~/.cache/torch/hub/checkpoints/
```

| Backbone       | Size   |
|----------------|--------|
| dinov2_vits14  | 84 MB  |
| dinov2_vitb14  | 330 MB |
| dinov2_vitl14  | 1.1 GB |

## Verifying the installation

```bash
make test   # smoke test — no internet, no real data, < 2 min
```
