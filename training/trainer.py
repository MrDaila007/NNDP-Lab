"""Self-training loop with Mean Teacher for unsupervised segmentation."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.segmentor import MeanTeacherModel
from pseudo_labels.kmeans_init import kmeans_pseudo_labels
from pseudo_labels.confidence import confidence_filter
from training.losses import MaskedCrossEntropy, ConsistencyLoss
from utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


class SelfTrainer:
    """Orchestrates K-means init → warm-up → iterative self-training.

    Args:
        cfg:           OmegaConf config object (see configs/default.yaml)
        train_dataset: unlabeled dataset (returns image, idx)
        val_dataset:   dataset with GT labels (returns image, mask); may be None
        device:        torch device
    """

    def __init__(self, cfg, train_dataset, val_dataset, device):
        self.cfg = cfg
        self.train_ds = train_dataset
        self.val_ds   = val_dataset
        self.device   = device

        self.model = MeanTeacherModel(
            backbone_name=cfg.model.backbone,
            in_channels=cfg.data.in_channels,
            decoder_dim=cfg.model.decoder_dim,
            num_classes=cfg.data.num_classes,
            frozen_backbone=cfg.model.frozen_backbone,
            ema_decay=cfg.self_training.ema_decay,
        ).to(device)

        self.ce_loss   = MaskedCrossEntropy(cfg.data.num_classes)
        self.cons_loss = ConsistencyLoss()
        self.metrics   = SegmentationMetrics(cfg.data.num_classes)

        self.pseudo_labels: dict = {}   # {image_idx: (Hp, Wp) int64 array}

        save_dir = Path(cfg.logging.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = save_dir / 'best_model.pth'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_optimizer(self):
        cfg = self.cfg
        return torch.optim.AdamW([
            {'params': self.model.student.decoder.parameters(),
             'lr': cfg.optimizer.lr},
            {'params': list(self.model.student.backbone.parameters()),
             'lr': cfg.optimizer.backbone_lr},
        ], weight_decay=cfg.optimizer.weight_decay)

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            drop_last=shuffle,
        )

    def _pseudo_tensor(self, pl_map: np.ndarray, target_hw: tuple) -> torch.Tensor:
        """Resize feature-resolution pseudo-label map to image resolution."""
        t = torch.from_numpy(pl_map).float().unsqueeze(0).unsqueeze(0)  # (1,1,Hp,Wp)
        t = F.interpolate(t, size=target_hw, mode='nearest').squeeze().long()
        return t

    # ------------------------------------------------------------------
    # Phase 1: K-means initialisation
    # ------------------------------------------------------------------

    def initialize_pseudo_labels(self):
        logger.info("=== Phase 1: K-means pseudo-label initialisation ===")
        pl_map, _ = kmeans_pseudo_labels(
            backbone=self.model.student.backbone,
            dataset=self.train_ds,
            num_clusters=self.cfg.data.num_classes,
            device=self.device,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
        )
        self.pseudo_labels = pl_map

    # ------------------------------------------------------------------
    # Phase 2: Update pseudo-labels via teacher
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_pseudo_labels(self, threshold: float):
        logger.info(f"  Updating pseudo-labels (threshold={threshold:.3f}) …")
        self.model.teacher.eval()
        loader = self._make_loader(self.train_ds, shuffle=False)
        new_pl, total, confident = {}, 0, 0

        for images, indices in tqdm(loader, desc='update PL', leave=False):
            images = images.to(self.device)
            logits = self.model.forward_teacher(images)
            pseudo, mask = confidence_filter(logits, threshold)
            for b, idx in enumerate(indices.tolist()):
                new_pl[idx] = pseudo[b].cpu().numpy().astype(np.int64)
                total     += mask[b].numel()
                confident += mask[b].sum().item()

        self.pseudo_labels = new_pl
        logger.info(f"  Confident pixels: {confident}/{total} "
                    f"({100*confident/total:.1f}%)")

    # ------------------------------------------------------------------
    # Phase 3: One epoch of student training
    # ------------------------------------------------------------------

    def train_epoch(self, optimizer, weak_aug, strong_aug, threshold: float) -> float:
        self.model.student.train()
        self.model.teacher.eval()
        loader = self._make_loader(self.train_ds, shuffle=True)
        total_loss, n = 0.0, 0

        for images, indices in tqdm(loader, desc='train', leave=False):
            images = images.to(self.device)
            B, _, H, W = images.shape

            # Collect pseudo-labels for the batch
            pls = [self.pseudo_labels.get(idx.item()) for idx in indices]
            if any(pl is None for pl in pls):
                continue
            pl_batch = torch.stack(
                [self._pseudo_tensor(pl, (H, W)) for pl in pls]
            ).to(self.device)              # (B, H, W) long

            # Teacher on weakly augmented images → confidence mask for consistency
            with torch.no_grad():
                weak_imgs = torch.stack([weak_aug(images[b]) for b in range(B)])
                t_logits  = self.model.forward_teacher(weak_imgs)
                t_pseudo, conf_mask = confidence_filter(t_logits, threshold)
                # Refine stored pseudo-labels with confident teacher predictions
                pl_batch[conf_mask] = t_pseudo[conf_mask]

            # Student on strongly augmented images
            strong_imgs = torch.stack([strong_aug(images[b]) for b in range(B)])
            s_logits = self.model.forward_student(strong_imgs)

            # Align spatial sizes
            s_logits_full = F.interpolate(s_logits, (H, W), mode='bilinear', align_corners=False)
            t_logits_full = F.interpolate(t_logits, (H, W), mode='bilinear', align_corners=False)

            # CE on ALL stored pseudo-labels; consistency only on confident pixels
            all_mask = torch.ones_like(conf_mask)
            loss = (
                self.ce_loss(s_logits_full, pl_batch, all_mask)
                + 0.5 * self.cons_loss(s_logits_full, t_logits_full, conf_mask)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.student.parameters(), 1.0)
            optimizer.step()
            self.model.update_teacher()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_ds is None:
            return 0.0
        self.model.teacher.eval()
        loader = self._make_loader(self.val_ds, shuffle=False)
        self.metrics.reset()

        for images, masks in tqdm(loader, desc='eval', leave=False):
            images = images.to(self.device)
            logits = self.model.forward_teacher(images)
            preds  = logits.argmax(dim=1).cpu()
            self.metrics.update(preds, masks)

        # Use Hungarian matching because cluster IDs may not align with GT
        miou = self.metrics.compute_miou(hungarian=True)
        logger.info(f"  Validation mIoU (hungarian): {miou:.4f}")
        return miou

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self, weak_aug, strong_aug):
        cfg = self.cfg

        n_rounds   = cfg.self_training.n_rounds
        n_warmup   = cfg.self_training.warmup_epochs
        n_epr      = cfg.self_training.epochs_per_round
        total_phases = 1 + n_rounds          # kmeans + rounds

        # Top-level phases bar
        phase_bar = tqdm(
            total=total_phases,
            desc='Pipeline',
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        )

        # ── Phase 0: K-means initialisation ──────────────────────────
        phase_bar.set_description('Phase 0/? | K-means init')
        self.initialize_pseudo_labels()
        phase_bar.update(1)

        optimizer = self._make_optimizer()
        best_miou = 0.0
        threshold = cfg.pseudo_labels.confidence_threshold

        # ── Warm-up ───────────────────────────────────────────────────
        phase_bar.set_description(f'Warm-up ({n_warmup} epochs)')
        warmup_bar = tqdm(
            range(n_warmup),
            desc='  warm-up',
            position=1,
            leave=False,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for ep in warmup_bar:
            loss = self.train_epoch(optimizer, weak_aug, strong_aug, threshold)
            warmup_bar.set_postfix(loss=f'{loss:.4f}')
            logger.info(f"  [warm-up {ep+1}/{n_warmup}] loss={loss:.4f}")
        warmup_bar.close()

        miou = self.evaluate()
        if miou > best_miou:
            best_miou = miou
            torch.save(self.model.state_dict(), self.save_path)

        # ── Self-training rounds ──────────────────────────────────────
        for rnd in range(n_rounds):
            phase_bar.set_description(
                f'Round {rnd+1}/{n_rounds} | thr={threshold:.3f} | best={best_miou:.4f}'
            )
            logger.info(f"=== Self-training round {rnd+1}/{n_rounds} ===")

            threshold = max(
                cfg.pseudo_labels.min_threshold,
                threshold - cfg.pseudo_labels.threshold_decay,
            )

            if rnd + 1 == cfg.self_training.unfreeze_round and cfg.model.frozen_backbone:
                logger.info("  Unfreezing last 4 backbone blocks")
                self.model.unfreeze_backbone(last_n_blocks=4)
                optimizer = self._make_optimizer()

            self.update_pseudo_labels(threshold)

            epoch_bar = tqdm(
                range(n_epr),
                desc=f'  round {rnd+1}',
                position=1,
                leave=False,
                file=sys.stdout,
                dynamic_ncols=True,
            )
            for ep in epoch_bar:
                loss = self.train_epoch(optimizer, weak_aug, strong_aug, threshold)
                epoch_bar.set_postfix(loss=f'{loss:.4f}')
                logger.info(f"  epoch {ep+1}/{n_epr} loss={loss:.4f}")
            epoch_bar.close()

            miou = self.evaluate()
            phase_bar.update(1)

            if miou > best_miou:
                best_miou = miou
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f"  ✓ New best mIoU: {best_miou:.4f}")
                phase_bar.set_description(
                    f'Round {rnd+1}/{n_rounds} ✓ best={best_miou:.4f}'
                )

        phase_bar.close()
        logger.info(f"Training complete. Best mIoU: {best_miou:.4f}")
        return best_miou
