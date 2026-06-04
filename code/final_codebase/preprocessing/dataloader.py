from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .attacks import AttackChooser, apply_attack
from .utils import load_rgb


def image_transform(image_size: int = 224):
    """Default ImageNet normalization used by ConvNeXt, ResNet, and ViT."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@dataclass
class BatchConfig:
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True


class SiamesePairDataset(Dataset):
    """Loads pair manifest rows for siamese training.

    Hard positives receive strict-panel attacks in RAM. Soft positives receive
    softer attacks in RAM. Negatives keep original images because their job is
    to push known and unknown neighborhoods apart.
    """

    def __init__(
        self,
        manifest: list[dict],
        strict_attacks: tuple[str, ...],
        soft_attacks: tuple[str, ...],
        image_size: int = 224,
        epoch: int = 1,
    ):
        self.manifest = manifest
        self.transform = image_transform(image_size)
        self.strict_chooser = AttackChooser(strict_attacks)
        self.soft_chooser = AttackChooser(soft_attacks)
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest[index]
        image_a = load_rgb(row["path_1"])
        image_b = load_rgb(row["path_2"])

        pair_type = row.get("pair_type", "")
        if pair_type == "hard_pos":
            attack = self.strict_chooser.choose(row["path_2"], self.epoch, int(row.get("times_trained", 0)))
            image_b = apply_attack(image_b, attack)
        elif pair_type == "soft_pos":
            attack = self.soft_chooser.choose(row["path_2"], self.epoch, int(row.get("times_trained", 0)))
            image_b = apply_attack(image_b, attack)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return self.transform(image_a), self.transform(image_b), label, index


class QueryImageDataset(Dataset):
    """Loads single images for store embedding and per-attack evaluation."""

    def __init__(self, rows: list[dict], attack: str = "original", image_size: int = 224):
        self.rows = rows
        self.attack = attack
        self.transform = image_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = apply_attack(load_rgb(row["path"]), self.attack)
        return self.transform(image), index


def make_pair_loader(
    manifest: list[dict],
    strict_attacks: tuple[str, ...],
    soft_attacks: tuple[str, ...],
    config: BatchConfig,
    epoch: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = SiamesePairDataset(manifest, strict_attacks, soft_attacks, config.image_size, epoch)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )


def make_query_loader(rows: list[dict], attack: str, config: BatchConfig) -> DataLoader:
    dataset = QueryImageDataset(rows, attack=attack, image_size=config.image_size)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

