"""Entry point: unsupervised self-training for remote-sensing segmentation."""

import logging
import os
import sys
import torch

# Force unbuffered stdout/stderr so tqdm and logging appear immediately
os.environ.setdefault('PYTHONUNBUFFERED', '1')
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from pathlib import Path
from omegaconf import OmegaConf

from data.loveda import LoveDADataset
from data.sentinel2 import Sentinel2Dataset
from data.transforms import WeakAugmentation, StrongAugmentation
from training.trainer import SelfTrainer

Path('./logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('./logs/train.log', mode='w'),
    ],
)
logger = logging.getLogger(__name__)


def build_datasets(cfg):
    if cfg.data.dataset == 'loveda':
        train_ds = LoveDADataset(
            root=cfg.data.root,
            split='train',
            domain='all',
            image_size=cfg.data.image_size,
            return_labels=False,
        )
        val_ds = LoveDADataset(
            root=cfg.data.root,
            split='val',
            domain='all',
            image_size=cfg.data.image_size,
            return_labels=True,
        )
    elif cfg.data.dataset == 'sentinel2':
        train_ds = Sentinel2Dataset(
            root=cfg.data.root,
            patch_size=cfg.data.image_size,
            stride=cfg.data.image_size // 2,
        )
        val_ds = None   # no GT labels available
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")
    return train_ds, val_ds


def main():
    # Allow CLI overrides: python train.py data.batch_size=4 model.backbone=dinov2_vits14
    cfg = OmegaConf.merge(
        OmegaConf.load('configs/default.yaml'),
        OmegaConf.from_cli(),
    )
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    train_ds, val_ds = build_datasets(cfg)
    logger.info(f"Train: {len(train_ds)} samples  |  "
                f"Val: {len(val_ds) if val_ds else 0} samples")

    aug_cfg = cfg.augmentation
    weak_aug = WeakAugmentation(
        size=cfg.data.image_size,
        flip_prob=aug_cfg.weak.flip_prob,
    )
    strong_aug = StrongAugmentation(
        size=cfg.data.image_size,
        scale_range=tuple(aug_cfg.strong.scale_range),
        color_jitter=aug_cfg.strong.color_jitter,
        blur_prob=aug_cfg.strong.blur_prob,
    )

    trainer = SelfTrainer(cfg, train_ds, val_ds, device)
    trainer.train(weak_aug=weak_aug, strong_aug=strong_aug)


if __name__ == '__main__':
    main()
