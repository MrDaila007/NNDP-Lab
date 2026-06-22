# Quick Start

## 1. Set up environment

```bash
make env        # CPU
# or
make env-gpu    # GPU (CUDA 13.2)
```

See [installation.md](installation.md) for manual setup.

## 2. Download LoveDA dataset

```bash
make download   # ~6 GB, downloads via torchgeo
make verify     # check dataset structure
```

Data is saved to `./data/LoveDA/`.

## 3. Run smoke test (no data needed)

```bash
make test       # stub backbone, synthetic data, < 2 min
```

## 4. Train

| Target | Config | Est. time |
|--------|--------|-----------|
| `make train-fast` | ViT-S/14, 224 px, 1 warm-up + 1 round | ~15 min CPU |
| `make train-cpu`  | ViT-S/14, 224 px, full (default config) | ~3–4 h CPU |
| `make train-gpu`  | ViT-B/14, 448 px, full (default config) | ~1.5–2 h GPU |

```bash
make train-gpu
```

## 5. Evaluate

```bash
make eval
```

Prints per-class IoU and mIoU after Hungarian cluster→class alignment.

## Custom parameters

Override any config value on the command line:

```bash
python train.py \
    data.root=./data/LoveDA \
    model.backbone=dinov2_vitb14 \
    data.image_size=448 \
    data.batch_size=16 \
    self_training.n_rounds=5 \
    self_training.epochs_per_round=10
```

Or via Makefile overrides:
```bash
make train-gpu DATA=/mnt/data/LoveDA CKPT=./my_checkpoints
```

## Google Colab

Open `NNDP_Lab_Colab.ipynb` for an interactive notebook with training, evaluation, and visualization.
