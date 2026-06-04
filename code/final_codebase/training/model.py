from __future__ import annotations

import torch
from torch import nn

from .model_convnext import SiameseConvNeXt
from .model_resnet import SiameseResNet
from .model_vit import SiameseViT


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor, weights=None):
        distances = torch.norm(emb_a - emb_b, dim=1)
        pos = labels * distances.pow(2)
        neg = (1.0 - labels) * torch.relu(self.margin - distances).pow(2)
        loss = pos + neg
        if weights is not None:
            loss = loss * weights
        return loss.mean(), distances


def build_model(name: str, embedding_dim: int = 512, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name in {"convnext", "convnext_tiny"}:
        return SiameseConvNeXt(embedding_dim=embedding_dim, pretrained=pretrained)
    if name in {"resnet", "resnet50"}:
        return SiameseResNet(embedding_dim=embedding_dim, pretrained=pretrained)
    if name in {"vit", "vit_b_16"}:
        return SiameseViT(embedding_dim=embedding_dim, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {name}")
