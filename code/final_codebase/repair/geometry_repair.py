from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attacks import AttackChooser, apply_attack
from .config import STRICT_ATTACKS
from .dataloader import QueryImageDataset
from .model import build_model
from .utils import load_rgb, read_json


def repair_store_geometry(
    checkpoint: str | Path,
    store_json: str | Path,
    output_checkpoint: str | Path,
    model_name: str = "convnext",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-6,
    strict_weight: float = 1.0,
    device: str = "cuda",
) -> Path:
    """Repair an existing store without adding new images.

    Goal:
    - keep original store images stable
    - pull strict attack variants close to their own original embedding
    - avoid aggressive geometry movement
    """

    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    store = read_json(store_json)
    rows = list(store.get("accepted_known", [])) + list(store.get("accepted_unknown", []))
    if not rows:
        raise RuntimeError("store has no accepted rows to repair")

    model = build_model(model_name, pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    chooser = AttackChooser(tuple(STRICT_ATTACKS))
    dataset = QueryImageDataset(rows, attack="original")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for originals, indices in loader:
            originals = originals.to(device)
            original_emb = model.encode(originals)
            attack_tensors = []
            for local_i, row_idx in enumerate(indices.tolist()):
                row = rows[row_idx]
                attack = chooser.choose(row["path"], epoch, local_i)
                attacked = dataset.transform(apply_attack(load_rgb(row["path"]), attack))
                attack_tensors.append(attacked)
            attacks = torch.stack(attack_tensors, dim=0).to(device)
            attack_emb = model.encode(attacks)
            loss = strict_weight * F.smooth_l1_loss(attack_emb, original_emb.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        print(f"[Repair] epoch {epoch} loss {total / max(1, batches):.6f}")

    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "repair_epochs": epochs}, output_checkpoint)
    return output_checkpoint

