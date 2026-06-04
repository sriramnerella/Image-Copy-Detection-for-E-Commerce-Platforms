from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .utils import l2_normalize


class SiameseResNet(nn.Module):
    """ResNet-50 siamese encoder.

    This file is separate from ConvNeXt/ViT so experiments can switch models
    without changing training, store, or evaluation logic.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding_head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return l2_normalize(self.embedding_head(features))

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor):
        return self.encode(image_a), self.encode(image_b)

