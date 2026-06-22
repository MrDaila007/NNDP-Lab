"""Weak and strong augmentations for Mean Teacher self-training."""

import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class WeakAugmentation:
    """Random horizontal flip only — preserves spatial structure for teacher."""

    def __init__(self, size: int, flip_prob: float = 0.5):
        self.size      = size
        self.flip_prob = flip_prob

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
        return image


class StrongAugmentation:
    """Colour jitter + random resized crop + Gaussian blur."""

    def __init__(
        self,
        size: int,
        scale_range: Tuple[float, float] = (0.7, 1.0),
        color_jitter: float = 0.3,
        blur_prob: float = 0.3,
    ):
        self.size         = size
        self.scale_range  = scale_range
        self.color_jitter = color_jitter
        self.blur_prob    = blur_prob

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, H, W = image.shape

        # Random resized crop
        scale  = random.uniform(*self.scale_range)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        top    = random.randint(0, H - crop_h)
        left   = random.randint(0, W - crop_w)
        image  = image[:, top:top + crop_h, left:left + crop_w]
        image  = F.interpolate(
            image.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0)

        # Colour jitter (brightness / contrast / saturation)
        j = self.color_jitter
        image = TF.adjust_brightness(image, 1.0 + random.uniform(-j, j))
        image = TF.adjust_contrast(image,   1.0 + random.uniform(-j, j))
        image = TF.adjust_saturation(image, 1.0 + random.uniform(-j, j))

        # Random horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)

        # Gaussian blur
        if random.random() < self.blur_prob:
            k = random.choice([3, 5])
            image = TF.gaussian_blur(image, kernel_size=k, sigma=random.uniform(0.1, 2.0))

        return image
