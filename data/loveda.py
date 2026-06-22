"""LoveDA dataset loader — supports unlabeled (train) and labeled (val) modes."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ImageNet statistics used by DINOv2
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

DOMAINS = ['Urban', 'Rural']


class LoveDADataset(Dataset):
    """LoveDA semantic segmentation dataset.

    Directory layout (as produced by torchgeo or make_fake_loveda.py):
        root/{split}/{domain}/images_png/*.png
        root/{split}/{domain}/masks_png/*.png   (only needed when return_labels=True)

    Args:
        root:          path to dataset root
        split:         'train' or 'val'
        domain:        'Urban', 'Rural', or 'all'
        image_size:    resize shorter edge then centre-crop to this size
        return_labels: if True yields (image, mask); else yields (image, idx)
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        domain: str = 'all',
        image_size: int = 448,
        return_labels: bool = False,
    ):
        self.root          = Path(root)
        self.split         = split
        self.image_size    = image_size
        self.return_labels = return_labels

        domains = DOMAINS if domain == 'all' else [domain]
        self.samples: List[Tuple[Path, Optional[Path]]] = []

        # torchgeo downloads into Train/Val (capitalised); fall back to train/val
        split_dir = split.capitalize() if (self.root / split.capitalize()).exists() else split

        for d in domains:
            img_dir  = self.root / split_dir / d / 'images_png'
            mask_dir = self.root / split_dir / d / 'masks_png'
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob('*.png')):
                mask_path = mask_dir / img_path.name
                self.samples.append((img_path, mask_path if mask_path.exists() else None))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root}/{split}/ "
                f"(domains={domains}). Run 'make download' first."
            )

        self.img_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)   # (3, H, W) float32

        if self.return_labels:
            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for {img_path}")
            mask = np.array(Image.open(mask_path).resize(
                (self.image_size, self.image_size), Image.NEAREST
            ), dtype=np.int64)
            # LoveDA masks use 1-based class indices; clamp background (0) to 0
            mask = np.clip(mask - 1, 0, 6)
            return image, torch.from_numpy(mask)

        return image, torch.tensor(idx, dtype=torch.long)
