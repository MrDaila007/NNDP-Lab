import copy
import torch
import torch.nn as nn

from models.backbone import DINOv2Backbone
from models.decoder import SegFormerDecoder


class Segmentor(nn.Module):
    """DINOv2 backbone + SegFormer decoder."""

    def __init__(
        self,
        backbone_name: str = 'dinov2_vitb14',
        in_channels: int = 3,
        decoder_dim: int = 256,
        num_classes: int = 7,
        frozen_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = DINOv2Backbone(backbone_name, in_channels, frozen=frozen_backbone)
        self.decoder  = SegFormerDecoder(self.backbone.embed_dim, decoder_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.decoder(features, target_size=x.shape[2:])


class MeanTeacherModel(nn.Module):
    """Student–Teacher wrapper with Exponential Moving Average update.

    Only the student receives gradients; teacher weights are updated as:
        θ_t ← α·θ_t + (1−α)·θ_s
    """

    def __init__(
        self,
        backbone_name: str = 'dinov2_vitb14',
        in_channels: int = 3,
        decoder_dim: int = 256,
        num_classes: int = 7,
        frozen_backbone: bool = True,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.student = Segmentor(backbone_name, in_channels, decoder_dim,
                                 num_classes, frozen_backbone)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self):
        α = self.ema_decay
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(α).add_(sp.data, alpha=1.0 - α)

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher(x)

    def unfreeze_backbone(self, last_n_blocks: int = 4):
        for model in (self.student, self.teacher):
            model.backbone.unfreeze_last_n_blocks(last_n_blocks)
