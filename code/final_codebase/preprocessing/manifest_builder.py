from __future__ import annotations

from pathlib import Path

from .manifest import build_pairs, collect_images, split_known_unknown, write_dataset_files


def build_ucid_manifest(image_root: str | Path, output_dir: str | Path, known_count: int = 3500) -> None:
    """Build a UCID-style manifest.

    UCID is small, so the default pair count is 25k:
    10k hard positives, 5k soft positives, 6.25k hard negatives, 3.75k soft negatives.
    """

    rows = split_known_unknown(collect_images(image_root), known_count=known_count, seed=20260519)
    pairs = build_pairs(
        rows,
        total_pairs=25000,
        hard_pos=10000,
        soft_pos=5000,
        hard_neg=6250,
        soft_neg=3750,
        seed=20260519,
    )
    write_dataset_files(rows, pairs, output_dir, "ucid")


def build_amazon50k_manifest(image_root: str | Path, output_dir: str | Path, known_count: int = 35000) -> None:
    """Build the Amazon-50k manifest.

    Amazon plan:
    35k known, 15k unknown, 125k training pairs.
    """

    rows = split_known_unknown(collect_images(image_root), known_count=known_count, seed=20260519)
    pairs = build_pairs(
        rows,
        total_pairs=125000,
        hard_pos=50000,
        soft_pos=25000,
        hard_neg=31250,
        soft_neg=18750,
        seed=20260519,
    )
    write_dataset_files(rows, pairs, output_dir, "amazon50k")

