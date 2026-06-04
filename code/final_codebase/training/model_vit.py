from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from .utils import l2_normalize


class SiameseViT(nn.Module):
    """ViT-B/16 siamese encoder.

    ViT is useful as an alternative backbone when comparing whether the store
    geometry is backbone-specific or stable across model families.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        backbone = vit_b_16(weights=weights)
        feature_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return l2_normalize(self.embedding_head(features))

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor):
        return self.encode(image_a), self.encode(image_b)

