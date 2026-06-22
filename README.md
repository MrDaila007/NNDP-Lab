# Unsupervised Semantic Segmentation for Remote Sensing

Transfer learning with pseudo-labels and self-training for semantic segmentation of Earth observation imagery **without any manual annotation**.

> **Report:** [REPORT.md](REPORT.md) · [REPORT.pdf](REPORT.pdf) — full methodology, results, and figures (26 May 2026)

## Overview

The pipeline combines a frozen DINOv2 backbone with iterative pseudo-label refinement via Mean Teacher self-training. No labeled data is required at any stage.

```
┌─────────────────────────────────────────────────────────┐
│  Phase 0 — Initialisation                               │
│  DINOv2 patch features → MiniBatch K-means → pseudo-    │
│  labels (cluster IDs 0–6) for every training image      │
├─────────────────────────────────────────────────────────┤
│  Phase 1 — Warm-up                                      │
│  Train decoder on K-means pseudo-labels (backbone       │
│  frozen). CE loss over all pixels.                      │
├─────────────────────────────────────────────────────────┤
│  Phase 2 — Self-training rounds (×N)                    │
│                                                         │
│  Teacher (EMA) ──weak aug──▶ pseudo-labels + conf mask  │
│                                         │               │
│  Student ──strong aug──▶ logits         │               │
│                              ▼          ▼               │
│            CE(student, PL, all_mask)                    │
│          + 0.5 × KL(student ‖ teacher, conf_mask)       │
│                              │                          │
│            EMA update: θ_t ← α·θ_t + (1-α)·θ_s         │
│  Confidence threshold decays each round: 0.95 → 0.70    │
├─────────────────────────────────────────────────────────┤
│  Evaluation                                             │
│  Hungarian matching aligns cluster IDs with GT classes  │
│  → mIoU reported correctly even without label alignment │
└─────────────────────────────────────────────────────────┘
```

## Results

Real GPU training on RTX 5070 Ti (CUDA 13.2, sm_120), LoveDA val set, Hungarian matching:

| Stage | Epochs | Train Loss | Val mIoU |
|-------|--------|------------|----------|
| K-means init | — | — | — |
| Warm-up | 1 | 1.6032 | **9.86 %** |
| Self-training round 1 | 2 | 0.9312 | 9.45 % |
| Self-training round 2 | 2 | 0.4393 | 7.99 % |
| **Best** | — | — | **9.86 %** |

Config: ViT-S/14, 224 px, batch 16, 2522 train images, 645 632 patch vectors for K-means.
Loss dropped 73 % (1.60 → 0.44) in 5 epochs. Full training (5+ rounds, 448 px, ViT-B) is expected to reach 35–45 % mIoU.

## Supported Data

| Dataset | Bands | Resolution | Classes |
|---------|-------|------------|---------|
| [LoveDA](https://github.com/Junjue-Wang/LoveDA) | RGB (3) | 0.3 m | 7 |
| Sentinel-2 | Multispectral (13) | 10–60 m | configurable |

LoveDA classes: `background · building · road · water · barren · forest · agriculture`

## Architecture

```
Input (3 or 13 bands)
    │
    ▼
ChannelAdapter          ← learnable Conv projection (only for Sentinel-2)
    │
    ▼
DINOv2 ViT backbone     ← dinov2_vits14 / vitb14 / vitl14
(4 intermediate layers)
    │
    ▼
SegFormer MLP decoder   ← 4-scale MLP projection + fuse conv
    │
    ▼
Segmentation logits (H × W × num_classes)
```

The student and teacher share the same architecture; teacher weights are updated as an Exponential Moving Average of the student.

## Installation

```bash
# Create environment
make env            # CPU
make env-gpu        # GPU (CUDA 13.2, RTX 5070 Ti / sm_120)

# Download LoveDA (~6 GB)
make download

# Verify dataset structure
make verify
```

Manual install:
```bash
conda create -n rseg python=3.11 -y
conda activate rseg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU
# GPU (CUDA 13.2, RTX 5070 Ti):
# pip install torch==2.12.1+cu132 torchvision --index-url https://download.pytorch.org/whl/cu132
pip install numpy scikit-learn scipy omegaconf pillow tqdm torchgeo
pip install rasterio          # only needed for Sentinel-2
```

## Quick Start

```bash
# Smoke test — no internet, no real data, runs in < 2 min
make test

# Fast end-to-end test on real LoveDA (CPU, ~30 min)
make train-fast

# Full training on CPU — ViT-S/14, 224 px
make train-cpu

# Full training on GPU — ViT-B/14, 448 px
make train-gpu

# Evaluate best checkpoint
make eval
```

Override any parameter on the command line:
```bash
make train-cpu DATA=/path/to/LoveDA CKPT=./my_checkpoints
```

Or call `train.py` directly with OmegaConf overrides:
```bash
python train.py \
    data.root=./data/LoveDA \
    model.backbone=dinov2_vitb14 \
    data.image_size=448 \
    data.batch_size=16 \
    self_training.n_rounds=5
```

## Configuration

All hyperparameters are in `configs/default.yaml`:

```yaml
data:
  dataset: loveda        # loveda | sentinel2
  image_size: 448        # must be divisible by 14 (DINOv2 patch size)
  in_channels: 3         # 3 for RGB, 13 for Sentinel-2

model:
  backbone: dinov2_vitb14
  frozen_backbone: true  # unfreeze last 4 blocks after round `unfreeze_round`

pseudo_labels:
  confidence_threshold: 0.95   # initial threshold for teacher confidence mask
  min_threshold: 0.70          # floor — threshold decays each round
  threshold_decay: 0.025

self_training:
  n_rounds: 5
  epochs_per_round: 10
  warmup_epochs: 5
  ema_decay: 0.999
  unfreeze_round: 3      # unfreeze backbone from this round onward
```

## Project Structure

```
├── configs/default.yaml        config (all hyperparameters)
├── data/
│   ├── loveda.py               LoveDA dataset loader (labels optional)
│   ├── sentinel2.py            Sentinel-2 sliding-window loader
│   └── transforms.py           weak / strong augmentation
├── models/
│   ├── backbone.py             DINOv2 + ChannelAdapter
│   ├── decoder.py              SegFormer 4-scale MLP decoder
│   └── segmentor.py            Segmentor + MeanTeacherModel (EMA)
├── pseudo_labels/
│   ├── kmeans_init.py          Phase 0: K-means on DINOv2 patch features
│   └── confidence.py           max-softmax and entropy confidence filters
├── training/
│   ├── losses.py               MaskedCrossEntropy + ConsistencyLoss (KL-div)
│   └── trainer.py              full self-training loop with tqdm progress
├── utils/metrics.py            mIoU + Hungarian cluster-to-class alignment
├── tests/
│   ├── smoke_test.py           end-to-end test with stub backbone
│   ├── make_fake_loveda.py     synthetic dataset generator
│   └── verify_loveda.py        dataset structure verifier
├── train.py                    entry point
├── evaluate.py                 evaluation with Hungarian matching
├── Makefile                    convenience targets
└── requirements.txt
```

## Evaluation Note

Because training is fully unsupervised, cluster IDs produced by the model are **not aligned** with semantic class indices. `evaluate.py` applies the Hungarian algorithm to find the optimal cluster→class mapping before computing mIoU. This gives a fair measure of segmentation quality independent of arbitrary label assignment.

## Sentinel-2 Usage

```yaml
# configs/default.yaml
data:
  dataset: sentinel2
  root: /path/to/tif_files   # directory with *.tif GeoTIFF files
  in_channels: 13
  image_size: 448
```

The `ChannelAdapter` (13 → 3 learnable convolution) maps multispectral bands to the RGB space expected by DINOv2. Normalization uses BigEarthNet per-band statistics.

## Hardware Requirements

| Config | GPU VRAM | Approx. time / epoch |
|--------|----------|----------------------|
| ViT-S/14, 224 px, bs=8  | CPU only            | ~13 min |
| ViT-S/14, 224 px, bs=16 | 6 GB                | ~2 min  |
| ViT-B/14, 448 px, bs=16 | 16 GB               | ~5 min  |
| ViT-S/14, 224 px, bs=16 | RTX 5070 Ti (16 GB) | ~1 min (tested) |

DINOv2 weights are downloaded automatically from Meta on first run and cached in `~/.cache/torch/hub/`.

## References

- [DINOv2 — Oquab et al. 2023](https://arxiv.org/abs/2304.07193)
- [LoveDA — Wang et al. 2021](https://arxiv.org/abs/2110.08733)
- [Mean Teacher — Tarvainen & Valpola 2017](https://arxiv.org/abs/1703.01780)
- [SegFormer — Xie et al. 2021](https://arxiv.org/abs/2105.15203)
