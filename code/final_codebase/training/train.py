from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attacks import AttackChooser, apply_attack
from .config import TrainConfig
from .data import ImageRowsDataset, PairDataset, default_transform
from .eval import best_threshold
from .model import ContrastiveLoss, SiameseConvNeXt
from .store import EmbeddingStore, remove_accepted_low_loss_pairs
from .utils import load_rgb, read_json, set_seed, write_json


def _known_path_set(source_rows: list[dict]) -> set[str]:
    return {row["path"] for row in source_rows if row.get("known")}


@torch.no_grad()
def _embed_active(model, rows, attack, batch_size, device):
    dataset = ImageRowsDataset(rows, attack=attack)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for images, _ in loader:
        chunks.append(model.encode(images.to(device)).cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 512)


def _ensure_active_pool(config: TrainConfig):
    active_path = config.store.active_pool_json
    if active_path.exists():
        return read_json(active_path)
    source = read_json(config.store.source_pool_json)
    store = EmbeddingStore.load(config.store.store_json)
    known = [r for r in source if r.get("known") and r["path"] not in store.processed_paths]
    unknown = [r for r in source if not r.get("known") and r["path"] not in store.processed_paths]
    active = known[: config.store.known_active] + unknown[: config.store.unknown_active]
    write_json(active_path, active)
    return active


def _rebuild_active_pool(config: TrainConfig, active_rows: list[dict], accepted_paths: set[str]) -> None:
    source = read_json(config.store.source_pool_json)
    store = EmbeddingStore.load(config.store.store_json)
    kept = [r for r in active_rows if r["path"] not in accepted_paths]
    known = [r for r in kept if r.get("known")]
    unknown = [r for r in kept if not r.get("known")]
    active_paths = {r["path"] for r in kept}
    source_known = [r for r in source if r.get("known") and r["path"] not in store.processed_paths and r["path"] not in active_paths]
    source_unknown = [r for r in source if not r.get("known") and r["path"] not in store.processed_paths and r["path"] not in active_paths]
    known = (known + source_known)[: config.store.known_active]
    unknown = (unknown + source_unknown)[: config.store.unknown_active]
    write_json(config.store.active_pool_json, known + unknown)


def _harvest(model, config: TrainConfig, epoch: int, device: str) -> dict:
    active = _ensure_active_pool(config)
    store = EmbeddingStore.load(config.store.store_json)
    known_active = [row for row in active if row.get("known")]
    unknown_active = [row for row in active if not row.get("known")]
    bank = store.known_bank_tensor(device)

    if bank is None:
        bank_rows = known_active
        bank_emb = _embed_active(model, bank_rows, "original", 64, device).to(device)
    else:
        bank_emb = bank

    known_orig = _embed_active(model, known_active, "original", 64, device)
    unknown_orig = _embed_active(model, unknown_active, "original", 64, device)
    known_base = torch.cdist(known_orig, bank_emb.cpu()).min(dim=1).values if len(known_active) else torch.empty(0)
    unknown_base = torch.cdist(unknown_orig, bank_emb.cpu()).min(dim=1).values if len(unknown_active) else torch.empty(0)

    dup_thr = known_base.quantile(0.60).item() if len(known_base) else 0.0
    not_thr = dup_thr
    accepted_paths = []

    def accept_rows(rows, orig_emb, base_dists, known):
        added = 0
        for idx, row in enumerate(rows):
            if row["path"] in store.processed_paths:
                continue
            orig_pass = base_dists[idx].item() <= dup_thr if known else base_dists[idx].item() >= not_thr
            pass_attacks = ["original"] if orig_pass else []
            for attack in config.attack_policy.strict_attacks:
                attack_emb = _embed_active(model, [row], attack, 1, device)[0]
                self_dist = torch.norm(attack_emb - orig_emb[idx]).item()
                if self_dist <= config.store.self_threshold:
                    pass_attacks.append(attack)
            if orig_pass and len(pass_attacks) >= config.store.harvest_min_passes:
                store.append(row, orig_emb[idx], epoch, known, len(pass_attacks), pass_attacks)
                accepted_paths.append(row["path"])
                added += 1
        return added

    added_known = accept_rows(known_active, known_orig, known_base, True)
    added_unknown = accept_rows(unknown_active, unknown_orig, unknown_base, False)
    store.save(config.store.store_json, epoch)
    _rebuild_active_pool(config, active, set(accepted_paths))
    return {
        "accepted_known": added_known,
        "accepted_unknown": added_unknown,
        "accepted_known_total": len(store.accepted_known),
        "accepted_unknown_total": len(store.accepted_unknown),
        "processed_now_paths": accepted_paths,
        "known_query_total": len(known_active),
        "unknown_query_total": len(unknown_active),
    }


def train(config: TrainConfig, seed: int = 20260519) -> None:
    set_seed(seed)
    device = config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(config.manifest_json)
    source_rows = read_json(config.store.source_pool_json)
    known_paths = _known_path_set(source_rows)

    model = SiameseConvNeXt(embedding_dim=config.embedding_dim).to(device)
    loss_fn = ContrastiveLoss(config.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    transform = default_transform()
    strict_chooser = AttackChooser(config.attack_policy.strict_attacks)

    for epoch in range(1, config.epochs + 1):
        dataset = PairDataset(manifest, config.attack_policy.strict_attacks, config.attack_policy.soft_attacks)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=False)
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch_idx, (img_a, img_b, labels, indices, _, _) in enumerate(loader):
            if config.max_batches_per_epoch and batch_idx >= config.max_batches_per_epoch:
                break
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)
            emb_a, emb_b = model(img_a, img_b)
            pair_loss, distances = loss_fn(emb_a, emb_b, labels)

            strict_loss = torch.tensor(0.0, device=device)
            for local_i, manifest_idx in enumerate(indices.tolist()):
                row = manifest[manifest_idx]
                attack = strict_chooser.choose(row["path_1"], epoch, local_i)
                attacked = transform(apply_attack(load_rgb(row["path_1"]), attack)).unsqueeze(0).to(device)
                attacked_emb = model.encode(attacked)
                strict_loss = strict_loss + F.smooth_l1_loss(attacked_emb, emb_a[local_i : local_i + 1].detach())
            strict_loss = strict_loss / max(1, len(indices))

            loss = config.pair_loss_weight * pair_loss + config.strict_consistency_weight * strict_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            batches += 1
            with torch.no_grad():
                for local_i, manifest_idx in enumerate(indices.tolist()):
                    row = manifest[manifest_idx]
                    row["last_distance"] = round(float(distances[local_i].detach().cpu()), 6)
                    row["priority"] = row["last_distance"]
                    row["times_trained"] = int(row.get("times_trained", 0)) + 1

        ckpt = config.output_dir / f"checkpoint_epoch-{epoch}.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt)
        write_json(config.manifest_json, manifest)
        print(f"[Epoch] {epoch} loss {epoch_loss / max(1, batches):.4f} | pairs {len(manifest)} | checkpoint {ckpt}")

        if epoch % config.eval_every == 0:
            stats = _harvest(model, config, epoch, device)
            print(
                f"[Store] epoch {epoch} added_known {stats['accepted_known']} | "
                f"added_unknown {stats['accepted_unknown']} | "
                f"store_known_total {stats['accepted_known_total']} | "
                f"store_unknown_total {stats['accepted_unknown_total']}"
            )
            if config.harvest_remove_train_pairs:
                manifest, removed = remove_accepted_low_loss_pairs(
                    manifest,
                    stats["processed_now_paths"],
                    known_paths,
                    min_trainings=config.harvest_prune_min_trainings,
                    pos_max_dist=config.harvest_prune_pos_max_dist,
                    neg_min_dist=config.harvest_prune_neg_min_dist,
                )
                write_json(config.manifest_json, manifest)
                print(f"[HarvestPrune] pairs_pruned {removed} | pairs_remaining {len(manifest)}")
