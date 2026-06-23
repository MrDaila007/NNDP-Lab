# Unsupervised Semantic Segmentation for Remote Sensing

Transfer learning with pseudo-labels and self-training for semantic segmentation of Earth observation imagery **without any manual annotation**.

> **Report:** [docs/report.md](docs/report.md) · [REPORT.pdf](REPORT.pdf) — full methodology, results, and figures (26 May 2026)

## Documentation

| Page | Description |
|------|-------------|
| [Installation](docs/installation.md) | Environment setup, CUDA versions |
| [Quick Start](docs/quickstart.md) | Download data, train, evaluate |
| [Architecture](docs/architecture.md) | DINOv2 + SegFormer + Mean Teacher |
| [Training](docs/training.md) | Pipeline, config reference, hardware |
| [Evaluation](docs/evaluation.md) | Hungarian matching, metrics |
| [Datasets](docs/datasets.md) | LoveDA and Sentinel-2 |

## Results

Real GPU training on RTX 5070 Ti (CUDA 13.2), LoveDA val set, Hungarian matching:

**Full training** (ViT-B/14, 448 px, RTX 5070 Ti, 2 h 18 min):

| Stage | Epochs | Train Loss | Val mIoU | Confident px |
|-------|--------|------------|----------|--------------|
| Warm-up | 5 | 1.60 → 1.50 | **21.38 %** | — |
| Round 1 (θ=0.95) | 10 | 1.13 → 1.02 | 14.05 % | 0.0 % |
| Round 2 (θ=0.93) | 10 | 0.74 → 0.70 | 9.25 % | 0.0 % |
| Round 3 (θ=0.90) | 10 | 0.50 → 0.40 | 7.83 % | 12.6 % |
| Round 4 (θ=0.88) | 10 | 0.28 → 0.21 | 8.44 % | 59.3 % |
| Round 5 (θ=0.85) | 10 | 0.16 → 0.15 | 8.56 % | 82.9 % |
| **Best** | — | — | **21.38 %** | — |

**Quick test** (ViT-S/14, 224 px, 5 epochs): best mIoU **9.86 %**.

## Quick Start

```bash
make env-gpu    # create conda env (CUDA 13.2)
make download   # download LoveDA (~6 GB)
make train-gpu  # ViT-B/14, 448 px, full training
make eval       # evaluate with Hungarian matching
```

## Architecture

```
Input → ChannelAdapter → DINOv2 ViT (frozen) → SegFormer MLP Decoder → Logits
                                ↕ EMA
                         Mean Teacher
```

Backbone: `dinov2_vits14` / `vitb14` / `vitl14` · Decoder: 4-scale MLP head · Teacher: EMA α=0.999

## Supported Datasets

| Dataset | Bands | Resolution | Classes |
|---------|-------|------------|---------|
| [LoveDA](https://github.com/Junjue-Wang/LoveDA) | RGB (3) | 0.3 m/px | 7 |
| Sentinel-2 | Multispectral (13) | 10–60 m | configurable |

## References

- [DINOv2 — Oquab et al. 2023](https://arxiv.org/abs/2304.07193)
- [LoveDA — Wang et al. 2021](https://arxiv.org/abs/2110.08733)
- [Mean Teacher — Tarvainen & Valpola 2017](https://arxiv.org/abs/1703.01780)
- [SegFormer — Xie et al. 2021](https://arxiv.org/abs/2105.15203)
