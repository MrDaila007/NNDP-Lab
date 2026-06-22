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

| Stage | Epochs | Train Loss | Val mIoU |
|-------|--------|------------|----------|
| Warm-up | 1 | 1.603 | **9.86 %** |
| Self-training round 1 | 2 | 0.931 | 9.45 % |
| Self-training round 2 | 2 | 0.439 | 7.99 % |

Quick 5-epoch test (ViT-S/14, 224 px). Full training (ViT-B/14, 448 px, 5 rounds) expected: **35–45 % mIoU**.

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
