from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .utils import write_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: str | Path) -> list[dict]:
    """Collect images as rows with path/category.

    Category is the parent folder name. For flat datasets, every image belongs
    to category ``default``.
    """

    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        category = path.parent.name if path.parent != root else "default"
        rows.append({"path": str(path), "category": category})
    return rows


def split_known_unknown(rows: list[dict], known_count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    known_paths = {row["path"] for row in rows[:known_count]}
    for row in rows:
        row["known"] = row["path"] in known_paths
    return rows


def _group_by_category(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("category", "default")].append(row)
    return groups


def build_pairs(
    rows: list[dict],
    total_pairs: int,
    hard_pos: int,
    soft_pos: int,
    hard_neg: int,
    soft_neg: int,
    seed: int = 20260519,
) -> list[dict]:
    """Build pair manifest with the same high-level composition used in runs."""

    if total_pairs != hard_pos + soft_pos + hard_neg + soft_neg:
        raise ValueError("pair counts must sum to total_pairs")

    rng = random.Random(seed)
    known = [row for row in rows if row.get("known")]
    unknown = [row for row in rows if not row.get("known")]
    known_by_cat = _group_by_category(known)
    multi_cats = [cat for cat, items in known_by_cat.items() if len(items) >= 2]
    pairs = []

    def add_pair(path_1, path_2, label, pair_type, priority):
        pairs.append(
            {
                "path_1": path_1,
                "path_2": path_2,
                "label": int(label),
                "pair_type": pair_type,
                "priority": float(priority),
                "times_trained": 0,
                "last_distance": 999.0,
            }
        )

    for _ in range(hard_pos):
        cat = rng.choice(multi_cats)
        a, b = rng.sample(known_by_cat[cat], 2)
        add_pair(a["path"], b["path"], 1, "hard_pos", 0.80)

    for _ in range(soft_pos):
        row = rng.choice(known)
        add_pair(row["path"], row["path"], 1, "soft_pos", 0.60)

    for _ in range(hard_neg):
        a = rng.choice(known)
        b = rng.choice(unknown)
        add_pair(a["path"], b["path"], 0, "hard_neg", 0.80)

    for _ in range(soft_neg):
        a = rng.choice(known)
        b = rng.choice(unknown)
        add_pair(a["path"], b["path"], 0, "soft_neg", 0.60)

    rng.shuffle(pairs)
    for idx, row in enumerate(pairs):
        row["pair_id"] = f"pair_{idx:08d}"
    return pairs


def write_dataset_files(rows: list[dict], pairs: list[dict], output_dir: str | Path, prefix: str) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / f"source_pool_{prefix}.json", rows)
    write_json(output_dir / f"training_pairs_{prefix}.json", pairs)
    known_categories = sorted({row["category"] for row in rows if row.get("known")})
    write_json(output_dir / "known_categories.json", known_categories)
