"""Compatibility aliases for older imports.

New code should import from ``duplicate_bank.dataloader``.
"""

from .dataloader import BatchConfig, QueryImageDataset, SiamesePairDataset, image_transform

default_transform = image_transform
PairDataset = SiamesePairDataset
ImageRowsDataset = QueryImageDataset

__all__ = [
    "BatchConfig",
    "ImageRowsDataset",
    "PairDataset",
    "QueryImageDataset",
    "SiamesePairDataset",
    "default_transform",
    "image_transform",
]
