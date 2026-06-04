from __future__ import annotations

from pathlib import Path

import torch

from .store import EmbeddingStore, remove_accepted_low_loss_pairs
from .utils import read_json, write_json


def load_store(path: str | Path) -> EmbeddingStore:
    return EmbeddingStore.load(path)


def save_store(path: str | Path, store: EmbeddingStore, epoch: int | None = None) -> None:
    store.save(path, epoch=epoch)


def build_known_bank(store: EmbeddingStore, device: str = "cpu") -> torch.Tensor | None:
    return store.known_bank_tensor(device)


def store_counts(path: str | Path) -> dict:
    store = EmbeddingStore.load(path)
    return {
        "accepted_known": len(store.accepted_known),
        "accepted_unknown": len(store.accepted_unknown),
        "processed_paths": len(store.processed_paths),
    }


def snapshot_store(store_json: str | Path, epoch: int) -> Path:
    """Save epoch-specific store snapshot for later rollback/eval."""

    store_json = Path(store_json)
    snapshot = store_json.with_name(f"{store_json.stem}_epoch-{epoch}.json")
    data = read_json(store_json)
    write_json(snapshot, data)
    return snapshot


def prune_manifest_using_store_acceptance(
    manifest_json: str | Path,
    accepted_paths: list[str],
    known_paths: set[str],
    min_trainings: int = 3,
    pos_max_dist: float = 4.0,
    neg_min_dist: float = 14.0,
) -> dict:
    """Remove only safe pairs attached to accepted store images."""

    manifest = read_json(manifest_json)
    manifest, removed = remove_accepted_low_loss_pairs(
        manifest,
        accepted_paths,
        known_paths,
        min_trainings=min_trainings,
        pos_max_dist=pos_max_dist,
        neg_min_dist=neg_min_dist,
    )
    write_json(manifest_json, manifest)
    return {"pairs_pruned": removed, "pairs_remaining": len(manifest)}

