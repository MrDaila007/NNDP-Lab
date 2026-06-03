import numpy as np
from scipy.optimize import linear_sum_assignment


class SegmentationMetrics:
    """Accumulates predictions and computes mIoU / pixel-accuracy.

    When `hungarian=True` the best cluster→class assignment is found
    automatically (needed when pseudo-label cluster IDs don't match GT
    class indices, i.e. fully unsupervised evaluation).
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.K = num_classes
        self.ignore = ignore_index
        self._cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self._cm.fill(0)

    def update(self, preds, targets):
        """
        preds:   (B, H, W) int64 tensor or array  — predicted class / cluster IDs
        targets: (B, H, W) int64 tensor or array  — ground-truth class indices
        """
        p = np.asarray(preds).ravel()
        t = np.asarray(targets).ravel()
        mask = (t >= 0) & (t < self.K) & (t != self.ignore)
        p, t = p[mask], t[mask]
        valid_p = (p >= 0) & (p < self.K)
        p, t = p[valid_p], t[valid_p]
        np.add.at(self._cm, (t, p), 1)

    def _remap(self) -> np.ndarray:
        """Hungarian matching: find optimal cluster→class permutation."""
        cost = -self._cm          # maximise overlap = minimise -overlap
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = np.arange(self.K)
        for r, c in zip(row_ind, col_ind):
            mapping[c] = r        # cluster c maps to GT class r
        remapped = np.zeros_like(self._cm)
        for c in range(self.K):
            remapped[:, mapping[c]] += self._cm[:, c]
        return remapped

    def compute_miou(self, hungarian: bool = False) -> float:
        cm = self._remap() if hungarian else self._cm
        inter = np.diag(cm)
        union = cm.sum(1) + cm.sum(0) - inter
        iou = np.where(union > 0, inter / union, 0.0)
        return float(iou.mean())

    def compute_per_class_iou(self, hungarian: bool = False) -> np.ndarray:
        cm = self._remap() if hungarian else self._cm
        inter = np.diag(cm)
        union = cm.sum(1) + cm.sum(0) - inter
        return np.where(union > 0, inter / union, 0.0)

    def compute_pixel_accuracy(self) -> float:
        return float(np.diag(self._cm).sum() / (self._cm.sum() + 1e-8))
