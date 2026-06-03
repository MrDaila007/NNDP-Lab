"""K-means initialisation of pseudo-labels using frozen DINOv2 features."""

import logging
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_patch_features(backbone, dataset, batch_size: int, num_workers: int, device):
    """Extract last-scale DINOv2 patch features for every image in `dataset`.

    Returns:
        features:     (N_patches_total, D) float32 array
        spatial_info: list of (image_idx, patch_row, patch_col, Hp, Wp)
    """
    backbone.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    all_feats, all_info = [], []

    for images, indices in tqdm(loader, desc='  extracting features',
                                 file=sys.stdout, dynamic_ncols=True):
        images = images.to(device)
        feat_maps = backbone(images)   # 4 × (B, D, Hp, Wp)
        feat = feat_maps[-1]           # use finest scale
        B, D, Hp, Wp = feat.shape

        flat = feat.permute(0, 2, 3, 1).reshape(-1, D).cpu().numpy()
        all_feats.append(flat)

        for b in range(B):
            img_idx = indices[b].item()
            for h in range(Hp):
                for w in range(Wp):
                    all_info.append((img_idx, h, w, Hp, Wp))

    return np.concatenate(all_feats, axis=0), all_info


def kmeans_pseudo_labels(backbone, dataset, num_clusters: int, device,
                          batch_size: int = 8, num_workers: int = 4):
    """Run MiniBatch K-means on DINOv2 features and return per-image label maps.

    Returns:
        pseudo_label_map: dict {image_idx: np.ndarray(Hp, Wp) of dtype int64}
        kmeans:           fitted MiniBatchKMeans object
    """
    logger.info("Extracting patch features for K-means initialisation …")
    features, spatial_info = extract_patch_features(
        backbone, dataset, batch_size, num_workers, device
    )
    logger.info(f"  {len(features):,} patch vectors of dim {features.shape[1]}")

    logger.info(f"Running MiniBatch K-means with k={num_clusters} …")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=42,
        batch_size=min(8192, len(features)),
        n_init=10,
        max_iter=300,
        verbose=0,
    )
    labels = kmeans.fit_predict(features)
    logger.info("  K-means done.")

    # Reconstruct per-image spatial label maps
    pseudo_label_map = {}
    for cluster_id, (img_idx, h, w, Hp, Wp) in zip(labels, spatial_info):
        if img_idx not in pseudo_label_map:
            pseudo_label_map[img_idx] = np.zeros((Hp, Wp), dtype=np.int64)
        pseudo_label_map[img_idx][h, w] = int(cluster_id)

    logger.info(f"  Pseudo-label maps generated for {len(pseudo_label_map)} images.")
    return pseudo_label_map, kmeans
