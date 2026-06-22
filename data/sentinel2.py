"""Sentinel-2 sliding-window patch dataset (unsupervised, no GT labels)."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# BigEarthNet per-band statistics (13 bands, indices 0–12)
_S2_MEAN = [1353.0, 1116.8, 1041.4,  945.1, 1199.5, 2002.2,
            2376.3, 2303.0, 732.7,    12.1, 1818.7, 1116.3, 2602.7]
_S2_STD  = [245.7,  333.1,  395.4,  593.8,  566.4,  861.0,
            1086.9, 1117.4, 416.0,    4.1,  1002.4,  761.3, 1231.8]


class Sentinel2Dataset(Dataset):
    """Sliding-window patches from Sentinel-2 GeoTIFF files.

    Requires `rasterio`. Each .tif in `root` is tiled into non-overlapping
    (or overlapping with stride) patches of size `patch_size × patch_size`.

    Args:
        root:       directory containing *.tif files
        patch_size: spatial size of each output patch (must be divisible by 14)
        stride:     step between patch origins (patch_size//2 for 50% overlap)
    """

    def __init__(self, root: str, patch_size: int = 448, stride: int = 224):
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for Sentinel2Dataset. "
                              "Install with: pip install rasterio")

        self.patch_size = patch_size
        self.stride     = stride
        self.mean       = torch.tensor(_S2_MEAN, dtype=torch.float32).view(-1, 1, 1)
        self.std        = torch.tensor(_S2_STD,  dtype=torch.float32).view(-1, 1, 1)

        tif_files = sorted(Path(root).glob('*.tif'))
        if not tif_files:
            raise FileNotFoundError(f"No .tif files found in {root}")

        self.patches: List[Tuple[Path, int, int]] = []
        for path in tif_files:
            import rasterio as rio
            with rio.open(path) as src:
                H, W = src.height, src.width
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    self.patches.append((path, y, x))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        import rasterio as rio
        path, y, x = self.patches[idx]
        p = self.patch_size
        with rio.open(path) as src:
            window = rio.windows.Window(x, y, p, p)
            data   = src.read(window=window).astype(np.float32)  # (13, p, p)

        tensor = torch.from_numpy(data)
        tensor = (tensor - self.mean) / self.std
        return tensor, torch.tensor(idx, dtype=torch.long)
