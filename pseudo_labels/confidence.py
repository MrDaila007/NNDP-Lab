import torch
import torch.nn.functional as F


def confidence_filter(logits: torch.Tensor, threshold: float):
    """Return pseudo-labels and a boolean mask of confident pixels.

    Args:
        logits:    (B, C, H, W) raw model output
        threshold: minimum max-softmax probability to be considered confident

    Returns:
        pseudo_labels:  (B, H, W) long — argmax prediction
        confident_mask: (B, H, W) bool — True where max_prob >= threshold
    """
    probs = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = probs.max(dim=1)
    return pseudo_labels, max_probs >= threshold


def entropy_filter(logits: torch.Tensor, max_entropy_ratio: float = 0.5):
    """Confidence filter based on normalised Shannon entropy (alternative).

    Lower entropy → more confident prediction.

    Args:
        logits:            (B, C, H, W)
        max_entropy_ratio: keep pixels with H/H_max <= this value

    Returns:
        pseudo_labels:  (B, H, W) long
        confident_mask: (B, H, W) bool
    """
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * (probs.clamp(min=1e-8)).log()).sum(dim=1)   # (B, H, W)
    max_h = torch.log(torch.tensor(logits.shape[1], dtype=torch.float))
    confident_mask = entropy / max_h <= max_entropy_ratio
    return probs.argmax(dim=1), confident_mask
