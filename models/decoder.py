import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, H*W, C)
        x = self.norm(self.linear(x))
        return x.transpose(1, 2).reshape(B, -1, H, W)


class SegFormerDecoder(nn.Module):
    """Lightweight 4-scale MLP decoder (SegFormer-style).

    Args:
        embed_dim:    backbone embedding dimension
        decoder_dim:  internal feature width
        num_classes:  number of output segmentation classes
    """

    def __init__(
        self,
        embed_dim: int = 768,
        decoder_dim: int = 256,
        num_classes: int = 7,
    ):
        super().__init__()
        self.projs = nn.ModuleList([
            _MLPProj(embed_dim, decoder_dim) for _ in range(4)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )
        self.seg_head = nn.Conv2d(decoder_dim, num_classes, 1)

    def forward(self, features: list, target_size: tuple) -> torch.Tensor:
        """
        Args:
            features:    list of 4 tensors (B, embed_dim, Hp, Wp) — all same spatial size
            target_size: (H, W) of the original image for final upsample
        """
        Hp, Wp = features[-1].shape[2], features[-1].shape[3]

        projected = []
        for feat, proj in zip(features, self.projs):
            f = proj(feat)
            if f.shape[2:] != (Hp, Wp):
                f = F.interpolate(f, (Hp, Wp), mode='bilinear', align_corners=False)
            projected.append(f)

        x = self.fuse(torch.cat(projected, dim=1))  # (B, decoder_dim, Hp, Wp)
        x = self.seg_head(x)                         # (B, num_classes, Hp, Wp)
        return F.interpolate(x, target_size, mode='bilinear', align_corners=False)
