from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .utils import l2_normalize


class SiameseConvNeXt(nn.Module):
    """ConvNeXt-Tiny siamese encoder for duplicate detection.

    The same ConvNeXt backbone is shared for both images. The output embedding
    is L2-normalized so nearest-bank retrieval can use Euclidean distance.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(1)
        return l2_normalize(self.embedding_head(features))

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor):
        emb_a = self.encode(image_a)
        emb_b = self.encode(image_b)
        return emb_a, emb_b

