"""Evaluate a trained teacher model on LoveDA validation split."""

import logging
import sys
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loveda import LoveDADataset, CLASSES
from models.segmentor import Segmentor
from utils.metrics import SegmentationMetrics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def main():
    cfg = OmegaConf.merge(
        OmegaConf.load('configs/default.yaml'),
        OmegaConf.from_cli(),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_ds = LoveDADataset(
        root=cfg.data.root,
        split='val',
        domain='all',
        image_size=cfg.data.image_size,
        return_labels=True,
    )

    model = Segmentor(
        backbone_name=cfg.model.backbone,
        in_channels=cfg.data.in_channels,
        decoder_dim=cfg.model.decoder_dim,
        num_classes=cfg.data.num_classes,
        frozen_backbone=False,
    ).to(device)

    ckpt = torch.load('checkpoints/best_model.pth', map_location=device)
    # Extract teacher sub-model weights from MeanTeacherModel checkpoint
    teacher_state = {
        k[len('teacher.'):]: v
        for k, v in ckpt.items()
        if k.startswith('teacher.')
    }
    model.load_state_dict(teacher_state)
    model.eval()
    logger.info("Checkpoint loaded.")

    metrics = SegmentationMetrics(cfg.data.num_classes)
    loader  = DataLoader(val_ds, batch_size=cfg.data.batch_size,
                         shuffle=False, num_workers=cfg.data.num_workers)

    with torch.no_grad():
        for images, masks in tqdm(loader):
            logits = model(images.to(device))
            preds  = logits.argmax(dim=1).cpu()
            metrics.update(preds, masks)

    # Hungarian matching aligns cluster IDs with GT semantic classes
    miou_h = metrics.compute_miou(hungarian=True)
    miou   = metrics.compute_miou(hungarian=False)
    acc    = metrics.compute_pixel_accuracy()
    iou_pc = metrics.compute_per_class_iou(hungarian=True)

    logger.info(f"\nmIoU (hungarian): {miou_h:.4f}   mIoU (direct): {miou:.4f}   "
                f"Pixel-Acc: {acc:.4f}\n")
    for cls, iou in zip(CLASSES, iou_pc):
        logger.info(f"  {cls:15s}  IoU={iou:.4f}")


if __name__ == '__main__':
    main()
