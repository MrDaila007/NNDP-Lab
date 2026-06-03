"""End-to-end smoke test.

Uses a tiny stub backbone (random Conv features) instead of DINOv2
so no internet download is required and the test runs on CPU in < 2 min.

Run from project root:
    conda run -n rseg python tests/smoke_test.py
"""

import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Stub backbone: replaces DINOv2, outputs same shape contract
# ------------------------------------------------------------------
EMBED_DIM = 64   # tiny for speed

class StubBackbone(nn.Module):
    """Mimics DINOv2Backbone interface without downloading anything."""

    def __init__(self, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.channel_adapter = nn.Identity()
        # 4 conv layers, each downsampling by 14 in total
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=14, stride=14, padding=0),
            nn.GELU(),
        )

    def forward(self, x):
        feat = self.encoder(x)          # (B, D, H/14, W/14)
        return [feat, feat, feat, feat] # 4 identical "scales"

    def unfreeze_last_n_blocks(self, n=4):
        pass   # no-op for stub


# Monkey-patch backbone module before any trainer import
import models.backbone as _bb_module

class _PatchedDINO(_bb_module.DINOv2Backbone):
    def __init__(self, model_name='dinov2_vitb14', in_channels=3, frozen=True):
        nn.Module.__init__(self)
        self.embed_dim = EMBED_DIM
        self.channel_adapter = (
            _bb_module.ChannelAdapter(in_channels)
            if in_channels != 3 else nn.Identity()
        )
        self._stub = StubBackbone(3, EMBED_DIM)

    def forward(self, x):
        x = self.channel_adapter(x)
        return self._stub(x)

    def unfreeze_last_n_blocks(self, n=4):
        pass

_bb_module.DINOv2Backbone = _PatchedDINO

# Now import the rest
from models.decoder import SegFormerDecoder
from models.segmentor import MeanTeacherModel, Segmentor
from data.loveda import LoveDADataset
from data.transforms import WeakAugmentation, StrongAugmentation
from pseudo_labels.kmeans_init import kmeans_pseudo_labels
from pseudo_labels.confidence import confidence_filter
from training.losses import MaskedCrossEntropy, ConsistencyLoss
from training.trainer import SelfTrainer
from utils.metrics import SegmentationMetrics
from tests.make_fake_loveda import generate as make_data

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
IMG_SIZE = 112   # 8 × 14 — divisible by DINOv2 patch size

CFG_YAML = f"""
data:
  dataset: loveda
  root: PLACEHOLDER
  num_classes: 7
  image_size: {IMG_SIZE}
  in_channels: 3
  num_workers: 0
  batch_size: 4
model:
  backbone: dinov2_vitb14
  decoder_dim: 32
  frozen_backbone: true
pseudo_labels:
  confidence_threshold: 0.50
  min_threshold: 0.30
  threshold_decay: 0.05
self_training:
  n_rounds: 2
  epochs_per_round: 2
  warmup_epochs: 2
  ema_decay: 0.99
  unfreeze_round: 99
optimizer:
  lr: 1e-3
  backbone_lr: 1e-4
  weight_decay: 1e-4
augmentation:
  weak:
    flip_prob: 0.5
  strong:
    color_jitter: 0.3
    scale_range: [0.7, 1.0]
    blur_prob: 0.3
logging:
  log_dir: PLACEHOLDER/logs
  save_dir: PLACEHOLDER/ckpts
"""


def run():
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')
    log = logging.getLogger('smoke')

    tmp = tempfile.mkdtemp(prefix='rseg_smoke_')
    data_root = Path(tmp) / 'LoveDA'
    log.info(f"Temp dir: {tmp}")

    try:
        # ---- 1. Create synthetic data ----
        log.info("Generating synthetic LoveDA data …")
        make_data(str(data_root), n_train=12, n_val=6, size=IMG_SIZE)

        # ---- 2. Build config ----
        cfg = OmegaConf.create(
            CFG_YAML
            .replace('PLACEHOLDER/logs', str(Path(tmp) / 'logs'))
            .replace('PLACEHOLDER/ckpts', str(Path(tmp) / 'ckpts'))
            .replace('PLACEHOLDER', str(data_root))
        )
        cfg.data.root = str(data_root)

        device = torch.device('cpu')

        # ---- 3. Datasets ----
        train_ds = LoveDADataset(str(data_root), split='train', domain='all',
                                 image_size=IMG_SIZE, return_labels=False)
        val_ds   = LoveDADataset(str(data_root), split='val',   domain='all',
                                 image_size=IMG_SIZE, return_labels=True)
        log.info(f"train={len(train_ds)}  val={len(val_ds)}")

        # ---- 4. Transforms ----
        weak_aug   = WeakAugmentation(IMG_SIZE)
        strong_aug = StrongAugmentation(IMG_SIZE, scale_range=(0.7, 1.0))

        # ---- 5. Full self-training pipeline ----
        trainer = SelfTrainer(cfg, train_ds, val_ds, device)
        best_miou = trainer.train(weak_aug=weak_aug, strong_aug=strong_aug)

        # ---- 6. Verify checkpoint exists ----
        ckpt = Path(tmp) / 'ckpts' / 'best_model.pth'
        assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
        log.info(f"Checkpoint saved: {ckpt}")

        # ---- 7. Quick unit checks ----
        # Losses
        ce   = MaskedCrossEntropy(7)
        cons = ConsistencyLoss()
        logits = torch.randn(2, 7, IMG_SIZE, IMG_SIZE)
        pl     = torch.randint(0, 7, (2, IMG_SIZE, IMG_SIZE))
        mask   = torch.ones(2, IMG_SIZE, IMG_SIZE, dtype=torch.bool)
        ce_val   = ce(logits, pl, mask)
        cons_val = cons(logits, logits.detach(), mask)
        assert ce_val.item() > 0,   "CE loss must be > 0"
        assert cons_val.item() >= 0, "Consistency loss must be >= 0"

        # Confidence filter
        pred, cmask = confidence_filter(logits, threshold=0.5)
        assert pred.shape  == (2, IMG_SIZE, IMG_SIZE)
        assert cmask.shape == (2, IMG_SIZE, IMG_SIZE)
        assert cmask.dtype == torch.bool

        # Metrics + Hungarian
        metrics = SegmentationMetrics(7)
        metrics.update(torch.randint(0, 7, (4, IMG_SIZE, IMG_SIZE)),
                       torch.randint(0, 7, (4, IMG_SIZE, IMG_SIZE)))
        miou_h = metrics.compute_miou(hungarian=True)
        miou   = metrics.compute_miou(hungarian=False)
        assert 0.0 <= miou_h <= 1.0
        assert 0.0 <= miou   <= 1.0

        log.info("=" * 55)
        log.info(f"  SMOKE TEST PASSED")
        log.info(f"  Best mIoU (hungarian): {best_miou:.4f}")
        log.info(f"  CE loss sample:        {ce_val.item():.4f}")
        log.info(f"  Consistency loss:      {cons_val.item():.4f}")
        log.info("=" * 55)
        return True

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
