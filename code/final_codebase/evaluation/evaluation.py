from __future__ import annotations

import random
from pathlib import Path

import torch

from .config import ALL_EVAL_ATTACKS
from .dataloader import BatchConfig, make_query_loader
from .model import build_model
from .utils import read_json, write_json


@torch.no_grad()
def embed_query_rows(model, rows: list[dict], attack: str, batch_config: BatchConfig, device: str) -> torch.Tensor:
    loader = make_query_loader(rows, attack, batch_config)
    vectors = []
    model.eval()
    for images, _ in loader:
        vectors.append(model.encode(images.to(device)).cpu())
    if not vectors:
        return torch.empty(0, 512)
    return torch.cat(vectors, dim=0)


def choose_best_threshold(known_distances: torch.Tensor, unknown_distances: torch.Tensor) -> dict:
    """Find the best single threshold for one attack report."""

    if len(known_distances) == 0 or len(unknown_distances) == 0:
        return {"known": 0.0, "unknown": 0.0, "total": 0.0, "threshold": 0.0}

    candidates = torch.unique(torch.cat([known_distances, unknown_distances])).sort().values
    best = {"known": 0.0, "unknown": 0.0, "total": -1.0, "threshold": 0.0}
    for threshold in candidates:
        known = (known_distances <= threshold).float().mean().item() * 100.0
        unknown = (unknown_distances > threshold).float().mean().item() * 100.0
        total = (known + unknown) / 2.0
        if total > best["total"]:
            best = {
                "known": round(known, 2),
                "unknown": round(unknown, 2),
                "total": round(total, 2),
                "threshold": round(float(threshold), 6),
            }
    return best


def evaluate_store_26_attacks(
    checkpoint: str | Path,
    store_json: str | Path,
    output_json: str | Path,
    model_name: str = "convnext",
    query_known: int = 100,
    query_unknown: int = 100,
    batch_size: int = 64,
    seed: int = 20260519,
    device: str = "cuda",
) -> dict:
    """Evaluate 100 known + 100 unknown store queries against known bank.

    Known queries are expected to be duplicates, so distance should be below
    threshold. Unknown queries are expected to be non-duplicates, so distance
    should be above threshold.
    """

    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    store = read_json(store_json)
    known_rows = list(store.get("accepted_known", []))
    unknown_rows = list(store.get("accepted_unknown", []))
    if not known_rows:
        raise RuntimeError("accepted_known is empty; cannot build known bank.")

    rng = random.Random(seed)
    rng.shuffle(known_rows)
    rng.shuffle(unknown_rows)
    query_known_rows = known_rows[: min(query_known, len(known_rows))]
    query_unknown_rows = unknown_rows[: min(query_unknown, len(unknown_rows))]
    known_bank = torch.tensor([row["embedding"] for row in known_rows], dtype=torch.float32)

    model = build_model(model_name, pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    batch_config = BatchConfig(batch_size=batch_size, num_workers=0)

    attack_results = {}
    for attack in ALL_EVAL_ATTACKS:
        known_emb = embed_query_rows(model, query_known_rows, attack, batch_config, device)
        unknown_emb = embed_query_rows(model, query_unknown_rows, attack, batch_config, device)
        known_dist = torch.cdist(known_emb, known_bank).min(dim=1).values if len(query_known_rows) else torch.empty(0)
        unknown_dist = torch.cdist(unknown_emb, known_bank).min(dim=1).values if len(query_unknown_rows) else torch.empty(0)
        attack_results[attack] = choose_best_threshold(known_dist, unknown_dist)

    payload = {
        "checkpoint": str(checkpoint),
        "store_json": str(store_json),
        "model_name": model_name,
        "store_known": len(known_rows),
        "store_unknown": len(unknown_rows),
        "query_known": len(query_known_rows),
        "query_unknown": len(query_unknown_rows),
        "attacks": attack_results,
    }
    write_json(output_json, payload)
    return payload

