# Evaluation

## The alignment problem

Because training is fully unsupervised, the model's cluster IDs are **arbitrary** — cluster 3 might correspond to "forest" or "road" depending on initialisation. Comparing cluster IDs directly to ground-truth class indices would give meaningless results.

## Hungarian algorithm

`evaluate.py` solves this with the **Hungarian algorithm** (linear assignment):

1. Run the trained model on the validation set → predicted cluster maps.
2. Build a `num_classes × num_classes` cost matrix where entry `(i, j)` is the IoU between predicted cluster `i` and ground-truth class `j`.
3. Find the permutation that maximises total IoU (scipy `linear_sum_assignment`).
4. Remap predicted cluster IDs using this permutation, then compute per-class IoU and mIoU.

This gives a **fair measure of segmentation quality** independent of how clusters were numbered.

## Running evaluation

```bash
make eval
# or
python evaluate.py \
    data.root=./data/LoveDA \
    data.image_size=448 \
    model.backbone=dinov2_vitb14
```

Loads `checkpoints/best_model.pth` by default.

## Metrics

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union across all classes (primary metric) |
| **per-class IoU** | IoU for each of the 7 LoveDA classes |

## Results (RTX 5070 Ti, ViT-S/14, 224 px, 5 epochs)

| Stage | Epochs | Loss | Val mIoU |
|-------|--------|------|----------|
| Warm-up | 1 | 1.603 | **9.86 %** |
| Round 1 | 2 | 0.931 | 9.45 % |
| Round 2 | 2 | 0.439 | 7.99 % |

These are results from a quick 5-epoch test run. Full training (ViT-B/14, 448 px, 5 rounds × 10 epochs) is expected to reach **35–45 % mIoU**.

## Known limitations

- **Confidence collapse at high θ**: with `confidence_threshold=0.70` in early rounds, the teacher may produce 0% confident pixels, disabling the consistency loss. Use `confidence_threshold=0.95` with `threshold_decay=0.025` (default) to start strict and relax gradually.
- **Rare classes**: *Barren* (4.7% of pixels) consistently achieves the lowest IoU due to class imbalance.
- **Low resolution**: at 224 px, thin structures (roads) are blurred. Use 448 px for best road/building separation.
