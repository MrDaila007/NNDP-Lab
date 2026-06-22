# Training Pipeline

## Three-phase pipeline

### Phase 0 — K-means initialisation

The frozen backbone processes all training images and extracts patch-level features from the last transformer layer (dim D = 384 for ViT-S, 768 for ViT-B). All patch vectors are collected and fed to **MiniBatch K-means** with `k = num_classes` clusters.

Each patch is assigned a cluster ID — this becomes the initial pseudo-label map at resolution `H/14 × W/14`.

### Phase 1 — Warm-up

The decoder is trained on K-means pseudo-labels with the backbone frozen. Standard cross-entropy over all pixels:

```
L = CE(student_logits, kmeans_pseudo_labels)
```

Duration: `self_training.warmup_epochs` epochs (default 5).

### Phase 2 — Self-training rounds

Repeated `self_training.n_rounds` times (default 5):

**a. Update pseudo-labels.** The teacher processes all training images (no augmentation) and saves new pseudo-labels for pixels where `max(softmax) > θ`.

**b. Train student.** For each batch:
- Teacher receives **weakly augmented** images (random horizontal flip) → soft pseudo-labels + confidence mask.
- Student receives **strongly augmented** images (random resized crop + colour jitter + Gaussian blur).

Loss:
```
L = CE(student_logits, pseudo_labels, all_pixels)
  + 0.5 × KL(student_logits ‖ teacher_logits, confident_pixels)
```

**c. EMA teacher update** after every step:
```
θ_teacher ← α · θ_teacher + (1−α) · θ_student
```

**d. Threshold decay.** After each round:
```
θ ← max(θ − threshold_decay, min_threshold)
```

## Configuration reference

All hyperparameters live in `configs/default.yaml`:

```yaml
data:
  dataset: loveda          # loveda | sentinel2
  root: ./data/LoveDA
  num_classes: 7
  image_size: 448          # divisible by 14
  in_channels: 3           # 3 for RGB, 13 for Sentinel-2
  num_workers: 4
  batch_size: 8

model:
  backbone: dinov2_vitb14  # vits14 | vitb14 | vitl14
  decoder_dim: 256
  frozen_backbone: true

pseudo_labels:
  confidence_threshold: 0.95   # initial teacher threshold
  min_threshold: 0.70          # floor after decay
  threshold_decay: 0.025       # reduction per round

self_training:
  n_rounds: 5
  epochs_per_round: 10
  warmup_epochs: 5
  ema_decay: 0.999
  unfreeze_round: 3            # unfreeze backbone from this round

optimizer:
  lr: 1e-4                     # decoder learning rate
  backbone_lr: 1e-5            # backbone LR when unfrozen
  weight_decay: 1e-4

augmentation:
  weak:
    flip_prob: 0.5
  strong:
    color_jitter: 0.5
    scale_range: [0.5, 1.0]
    blur_prob: 0.5
```

## Hardware requirements

| Config | GPU VRAM | Time / epoch |
|--------|----------|-------------|
| ViT-S/14, 224 px, bs=8  | CPU only            | ~13 min     |
| ViT-S/14, 224 px, bs=16 | 6 GB                | ~2 min      |
| ViT-B/14, 448 px, bs=16 | 16 GB               | ~2–3 min    |
| ViT-S/14, 224 px, bs=16 | RTX 5070 Ti (16 GB) | ~1 min (measured) |

## Monitoring

Training logs are written to `logs/train.log`. Key lines:

```
INFO  Phase 1: K-means pseudo-label initialisation
INFO    2,582,528 patch vectors of dim 768
INFO  === Self-training round 1/5 ===
INFO    Confident pixels: 45/126543872 (36.2%)
INFO    epoch 3/10 loss=0.312
INFO  Validation mIoU (hungarian): 0.2841
INFO  Training complete. Best mIoU: 0.3156
```

Best checkpoint is saved automatically to `checkpoints/best_model.pth`.
