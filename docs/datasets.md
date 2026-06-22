# Datasets

## LoveDA

**Land-cOVEr Domain Adaptive semantic segmentation** — high-resolution RGB remote sensing imagery from Google Earth over urban and rural areas in China.

| Parameter | Value |
|-----------|-------|
| Source | Google Earth (Nanjing, Changsha, Wuhan) |
| Spatial resolution | 0.3 m/px |
| Channels | RGB (3) |
| Image size | 1024 × 1024 px |
| Classes | 7 |
| Train images | 2 522 (Urban: 1 156, Rural: 1 366) |
| Val images | 1 669 (Urban: 677, Rural: 992) |
| Test images | 1 796 (no labels) |

### Classes

| ID | Class | Urban share | Rural share | Overall |
|----|-------|-------------|-------------|---------|
| 0 | Background | 38.3 % | 31.9 % | 34.7 % |
| 1 | Building | 17.2 % | 3.0 % | 9.2 % |
| 2 | Road | 8.3 % | 2.2 % | 4.9 % |
| 3 | Water | 6.4 % | 9.7 % | 8.2 % |
| 4 | Barren | 6.5 % | 3.3 % | 4.7 % |
| 5 | Forest | 8.6 % | 14.9 % | 12.2 % |
| 6 | Agriculture | 14.7 % | 34.9 % | 26.1 % |

Background dominates (34.7%); Barren is the rarest class (4.7% — 7× less than Background).

### Download

```bash
make download   # uses torchgeo, ~6 GB
```

Data is saved to `./data/LoveDA/` with the following structure:
```
data/LoveDA/
├── Train/
│   ├── Urban/images_png/   ← RGB PNGs
│   ├── Urban/masks_png/    ← label masks (1-indexed, 1–7)
│   ├── Rural/images_png/
│   └── Rural/masks_png/
└── Val/
    ├── Urban/...
    └── Rural/...
```

Masks use **1-based indexing** (values 1–7). The loader subtracts 1 to produce 0-based indices (0–6).

### Usage note

Labels are loaded **only for evaluation** (`return_labels=True`). During training the dataset is used in unsupervised mode (`return_labels=False`) — no mask is read, only images.

---

## Sentinel-2

Multispectral GeoTIFF imagery with 13 bands at 10–60 m resolution. Loaded via a sliding-window approach using `rasterio`.

### Configuration

```yaml
data:
  dataset: sentinel2
  root: /path/to/tif_files   # directory containing *.tif files
  in_channels: 13
  image_size: 448
```

### ChannelAdapter

A learnable 1×1 convolution (`models/backbone.py`) maps 13 input channels to the 3-channel RGB space expected by DINOv2. Normalisation uses **BigEarthNet** per-band statistics.

### Requirements

```bash
pip install rasterio
```
