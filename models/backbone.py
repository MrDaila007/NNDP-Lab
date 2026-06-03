import torch
import torch.nn as nn

# Intermediate layer indices (0-based) for 4-scale feature extraction
_INTERMEDIATE = {
    'dinov2_vits14': [2, 5, 8, 11],
    'dinov2_vitb14': [2, 5, 8, 11],
    'dinov2_vitl14': [5, 11, 17, 23],
    'dinov2_vitg14': [9, 19, 29, 39],
}
_EMBED_DIM = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class ChannelAdapter(nn.Module):
    """Learnable projection from N spectral bands → 3 channels."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, 1, bias=False),
        )
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        return self.proj(x)


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT backbone returning 4-scale spatial feature maps.

    For multispectral input (e.g. Sentinel-2) a learnable ChannelAdapter
    projects from `in_channels` → 3 before the ViT patch embedding.

    Output: list of 4 tensors, each ``(B, embed_dim, H/14, W/14)``.
    """

    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        in_channels: int = 3,
        frozen: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = _EMBED_DIM[model_name]
        self._intermediate = _INTERMEDIATE[model_name]

        self.dino = torch.hub.load('facebookresearch/dinov2', model_name,
                                   skip_validation=True)

        self.channel_adapter = (
            ChannelAdapter(in_channels) if in_channels != 3 else nn.Identity()
        )

        if frozen:
            for p in self.dino.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        x = self.channel_adapter(x)
        # get_intermediate_layers with reshape=True → (B, D, H/14, W/14)
        features = self.dino.get_intermediate_layers(
            x,
            n=self._intermediate,
            reshape=True,
        )
        return list(features)   # 4 × (B, embed_dim, Hp, Wp)

    def unfreeze_last_n_blocks(self, n: int = 4):
        blocks = self.dino.blocks
        for blk in blocks[-n:]:
            blk.requires_grad_(True)
        # Also unfreeze norm
        if hasattr(self.dino, 'norm'):
            self.dino.norm.requires_grad_(True)
