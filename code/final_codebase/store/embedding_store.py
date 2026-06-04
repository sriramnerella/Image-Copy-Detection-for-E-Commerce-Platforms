from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .utils import read_json, write_json


@dataclass
class EmbeddingStore:
    accepted_known: list[dict]
    accepted_unknown: list[dict]
    processed_paths: set[str]
    candidate_known: dict[str, int]
    candidate_unknown: dict[str, int]

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingStore":
        data = read_json(
            path,
            {
                "accepted_known": [],
                "accepted_unknown": [],
                "processed_paths": [],
                "candidate_known": {},
                "candidate_unknown": {},
            },
        )
        return cls(
            accepted_known=list(data.get("accepted_known", [])),
            accepted_unknown=list(data.get("accepted_unknown", [])),
            processed_paths=set(data.get("processed_paths", [])),
            candidate_known=dict(data.get("candidate_known", {})),
            candidate_unknown=dict(data.get("candidate_unknown", {})),
        )

    def save(self, path: str | Path, epoch: int | None = None) -> None:
        data = {
            "accepted_known": self.accepted_known,
            "accepted_unknown": self.accepted_unknown,
            "processed_paths": sorted(self.processed_paths),
            "candidate_known": self.candidate_known,
            "candidate_unknown": self.candidate_unknown,
            "last_epoch": epoch,
            "meta": {
                "accepted_known_total": len(self.accepted_known),
                "accepted_unknown_total": len(self.accepted_unknown),
                "processed_total": len(self.processed_paths),
            },
        }
        write_json(path, data)

    def known_bank_tensor(self, device: str) -> torch.Tensor | None:
        if not self.accepted_known:
            return None
        bank = torch.tensor([row["embedding"] for row in self.accepted_known], dtype=torch.float32)
        return bank.to(device)

    def append(self, row: dict, embedding: torch.Tensor, epoch: int, known: bool, panel_passes: int, panel_attacks: list[str]) -> None:
        out = {
            "epoch": epoch,
            "path": row["path"],
            "category": row.get("category", ""),
            "known": bool(known),
            "embedding": [round(float(x), 6) for x in embedding.detach().cpu().tolist()],
            "panel_passes": panel_passes,
            "panel_pass_attacks": panel_attacks,
        }
        if known:
            self.accepted_known.append(out)
        else:
            self.accepted_unknown.append(out)
        self.processed_paths.add(row["path"])


def remove_accepted_low_loss_pairs(
    manifest: list[dict],
    accepted_paths: list[str],
    known_paths: set[str],
    min_trainings: int = 3,
    pos_max_dist: float = 4.0,
    neg_min_dist: float = 14.0,
) -> tuple[list[dict], int]:
    """Strict accepted-only pruning.

    A pair can be removed only if it touches a path that was just accepted into
    the store and the pair itself is already well learned.
    """

    accepted = set(accepted_paths)
    kept = []
    removed = 0
    for pair in manifest:
        p1 = pair.get("path_1", "")
        p2 = pair.get("path_2", "")
        if p1 not in accepted and p2 not in accepted:
            kept.append(pair)
            continue
        if int(pair.get("times_trained", 0)) < min_trainings:
            kept.append(pair)
            continue
        distance = float(pair.get("last_distance", pair.get("priority", 999.0)))
        label = float(pair.get("label", 0.0))
        k1 = p1 in known_paths
        k2 = p2 in known_paths
        remove_pair = False
        if label > 0.5 and k1 and k2 and distance <= pos_max_dist:
            remove_pair = True
        elif label < 0.5 and (k1 != k2) and distance >= neg_min_dist:
            remove_pair = True
        if remove_pair:
            removed += 1
        else:
            kept.append(pair)
    return kept, removed
