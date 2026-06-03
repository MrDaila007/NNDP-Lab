import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = 255


class MaskedCrossEntropy(nn.Module):
    """Cross-entropy loss computed only over confident pixels."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction='mean')
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,          # (B, C, H, W)
        pseudo_labels: torch.Tensor,   # (B, H, W) long
        confident_mask: torch.Tensor,  # (B, H, W) bool
    ) -> torch.Tensor:
        targets = pseudo_labels.clone()
        targets[~confident_mask] = IGNORE
        if (targets != IGNORE).sum() == 0:
            return logits.sum() * 0.0  # keep graph alive, loss = 0
        return self.ce(logits, targets)


class ConsistencyLoss(nn.Module):
    """KL-divergence between student and (detached) teacher distributions."""

    def forward(
        self,
        student_logits: torch.Tensor,  # (B, C, H, W)
        teacher_logits: torch.Tensor,  # (B, C, H, W)
        confident_mask: torch.Tensor,  # (B, H, W) bool
    ) -> torch.Tensor:
        s_log_p = F.log_softmax(student_logits, dim=1)
        t_p     = F.softmax(teacher_logits.detach(), dim=1)
        kl = F.kl_div(s_log_p, t_p, reduction='none').sum(dim=1)  # (B, H, W)
        if confident_mask.sum() == 0:
            return kl.mean() * 0.0
        return kl[confident_mask].mean()
