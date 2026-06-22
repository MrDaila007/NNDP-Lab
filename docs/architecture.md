# Architecture

## Overview

The model is a **student/teacher pair** sharing the same architecture: a frozen DINOv2 ViT backbone feeding a lightweight SegFormer-style MLP decoder. The teacher's weights are an Exponential Moving Average (EMA) of the student's, providing stable pseudo-label targets.

```
Input image (3 × H × W)
        │
        ▼
ChannelAdapter          ← 1×1 Conv projection (Sentinel-2 only: 13 → 3 channels)
        │
        ▼
DINOv2 ViT Backbone     ← dinov2_vits14 / vitb14 / vitl14
(4 intermediate layers, each H/14 × W/14 × D)
        │
        ▼
SegFormer MLP Decoder   ← 4-scale MLP projection + fuse conv
        │
        ▼
Segmentation logits (num_classes × H × W)
```

## Backbone — DINOv2

| Variant        | Dim D | Params | Notes |
|----------------|-------|--------|-------|
| dinov2_vits14  | 384   | 21 M   | fast, ~6 GB VRAM at 448 px |
| dinov2_vitb14  | 768   | 86 M   | **recommended**, ~10 GB VRAM at 448 px |
| dinov2_vitl14  | 1024  | 307 M  | best quality, needs ≥20 GB VRAM |

Patch size is fixed at **14×14 px**, so input resolution must be divisible by 14.
Features from 4 intermediate transformer layers are extracted and passed to the decoder.

Backbone is **frozen** during warm-up and early self-training rounds. It is unfrozen (last 4 blocks) from `self_training.unfreeze_round` onward with a lower LR (`optimizer.backbone_lr`).

## ChannelAdapter

A single learnable 1×1 convolution that projects arbitrary-channel inputs (e.g. 13-band Sentinel-2) to the 3-channel RGB space expected by DINOv2. Only used when `data.in_channels ≠ 3`.

## Decoder — SegFormer MLP Head

Simplified version of the MiX-Transformer segmentation head:

1. Four independent MLP layers project each backbone scale to `decoder_dim` (default 256).
2. All feature maps are bilinearly upsampled to the largest scale resolution.
3. Concatenated and fused by a 1×1 Conv → BN → ReLU.
4. Final 1×1 Conv outputs `num_classes` channels.

The decoder is always trained; backbone is conditionally unfrozen.

## Mean Teacher (EMA)

After each student optimizer step:

```
θ_teacher ← α · θ_teacher + (1 − α) · θ_student
```

Default `α = 0.999`. The teacher never receives gradient updates directly — its stability comes solely from the EMA. This makes teacher predictions smoother and more reliable than the student's, which is why teacher outputs are used as pseudo-label targets.
