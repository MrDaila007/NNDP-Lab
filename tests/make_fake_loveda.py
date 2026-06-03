"""Generate a tiny synthetic LoveDA dataset for smoke testing."""

import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image


NUM_CLASSES = 7
IMG_SIZE    = 112   # small enough for CPU smoke test (must be divisible by 14)
N_TRAIN     = 12
N_VAL       = 6


def make_segmentation_image(size: int) -> np.ndarray:
    """RGB image with blocky colour regions."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    n_blocks = random.randint(3, 8)
    for _ in range(n_blocks):
        h = random.randint(size // 6, size // 2)
        w = random.randint(size // 6, size // 2)
        y = random.randint(0, size - h)
        x = random.randint(0, size - w)
        colour = [random.randint(50, 255) for _ in range(3)]
        img[y:y+h, x:x+w] = colour
    return img


def make_mask(size: int) -> np.ndarray:
    """Mask with pixel values 1–7 (LoveDA convention)."""
    mask = np.ones((size, size), dtype=np.uint8)
    n_regions = random.randint(3, 8)
    for _ in range(n_regions):
        h = random.randint(size // 6, size // 2)
        w = random.randint(size // 6, size // 2)
        y = random.randint(0, size - h)
        x = random.randint(0, size - w)
        mask[y:y+h, x:x+w] = random.randint(1, NUM_CLASSES)
    return mask


def generate(root: str, n_train: int, n_val: int, size: int):
    root = Path(root)
    for split, n, domains in [
        ('train', n_train, ['Urban', 'Rural']),
        ('val',   n_val,   ['Urban', 'Rural']),
    ]:
        for domain in domains:
            img_dir  = root / split / domain / 'images_png'
            mask_dir = root / split / domain / 'masks_png'
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n // len(domains)):
                fname = f'{split}_{domain}_{i:04d}.png'
                Image.fromarray(make_segmentation_image(size)).save(img_dir / fname)
                Image.fromarray(make_mask(size)).save(mask_dir / fname)
    print(f"Synthetic LoveDA written to {root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',    default='./data/LoveDA_fake')
    parser.add_argument('--n_train', type=int, default=N_TRAIN)
    parser.add_argument('--n_val',   type=int, default=N_VAL)
    parser.add_argument('--size',    type=int, default=IMG_SIZE)
    args = parser.parse_args()
    generate(args.root, args.n_train, args.n_val, args.size)
