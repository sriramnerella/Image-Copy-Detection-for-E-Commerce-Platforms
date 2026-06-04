from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


STRICT_ATTACKS = [
    "rotate_m10",
    "rotate40",
    "crop40",
    "heavy_bright",
    "flip_h",
    "flip_v",
    "resize_compress",
    "watermark_asset",
    "blur",
    "bg_color_change",
]

SOFT_ATTACKS = [
    "rotate_m5",
    "rotate5",
    "rotate15",
    "crop5",
    "bright_light",
    "bright",
    "resize",
    "blur",
    "watermark_asset",
    "contrast",
]

ALL_EVAL_ATTACKS = [
    "original",
    "rotate_m5",
    "rotate_m10",
    "rotate5",
    "rotate15",
    "rotate30",
    "rotate40",
    "crop5",
    "crop10",
    "crop20",
    "crop40",
    "bright_light",
    "bright",
    "heavy_bright",
    "flip_h",
    "flip_v",
    "flip",
    "resize_compress",
    "resize",
    "watermark_text",
    "watermark_asset",
    "mix_rotate10_bright",
    "mix_crop20_watermark",
    "bg_color_change",
    "blur",
    "contrast",
]


@dataclass(frozen=True)
class AttackPolicy:
    """Attack groups used by training, strict harvest, and reporting."""

    strict_panel: tuple[str, ...] = ("original", *STRICT_ATTACKS)
    strict_attacks: tuple[str, ...] = tuple(STRICT_ATTACKS)
    soft_attacks: tuple[str, ...] = tuple(SOFT_ATTACKS)
    eval_attacks: tuple[str, ...] = tuple(ALL_EVAL_ATTACKS)


@dataclass
class StoreConfig:
    store_json: Path
    active_pool_json: Path
    source_pool_json: Path
    known_active: int = 1000
    unknown_active: int = 250
    harvest_min_passes: int = 11
    harvest_min_streak: int = 1
    self_threshold: float = 0.34
    self_percentile: float = 0.98
    known_margin: float = 0.0
    unknown_margin: float = 0.0


@dataclass
class EvalConfig:
    checkpoint: Path
    store_json: Path
    output_json: Path
    query_known: int = 100
    query_unknown: int = 100
    batch_size: int = 64
    distance_chunk: int = 256
    seed: int = 20260519
    attacks: tuple[str, ...] = tuple(ALL_EVAL_ATTACKS)


@dataclass
class TrainConfig:
    manifest_json: Path
    output_dir: Path
    store: StoreConfig
    attack_policy: AttackPolicy = field(default_factory=AttackPolicy)
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-6
    num_workers: int = 4
    max_batches_per_epoch: int = 0
    embedding_dim: int = 512
    margin: float = 1.0
    pair_loss_weight: float = 0.25
    strict_consistency_weight: float = 0.80
    strict_bank_known_weight: float = 0.20
    strict_bank_unknown_weight: float = 0.25
    unknown_push_weight: float = 0.0
    eval_every: int = 1
    prune_accepted_only: bool = True
    harvest_remove_train_pairs: bool = True
    harvest_prune_min_trainings: int = 3
    harvest_prune_pos_max_dist: float = 4.0
    harvest_prune_neg_min_dist: float = 14.0
    device: str = "cuda"
