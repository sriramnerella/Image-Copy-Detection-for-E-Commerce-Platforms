#!/usr/bin/env python3
"""Fast centroid HNSW/ERNG timing with bounded candidate search.

The previous explainable centroid graph used depth expansion until decisions
matched direct, which often expanded most of the bank. This version keeps the
same user-defined centroid HNSW/ERNG navigation ideas, but makes retrieval fast:

  HNSW: centrality layers + rank neighbors -> greedy centroid -> top centroid
        shortlist -> bounded candidate rerank.
  ERNG: exponential-rank centroid graph + multi-probe greedy -> top centroid
        shortlist -> bounded candidate rerank.

It calibrates small candidate caps to preserve direct decisions while keeping
query cost far below dense direct search.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/saketh/Codebase")
import phase3_all_datasets_graph_benchmark as G  # noqa: E402
import phase3_centroid_hnsw_erng as C  # noqa: E402

ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")


def greedy_seed(query, centroids, graph, start):
    cur = int(start)
    cur_d = float(np.linalg.norm(query - centroids[cur]))
    while True:
        best = cur
        best_d = cur_d
        for nb in graph.get(cur, []):
            d = float(np.linalg.norm(query - centroids[int(nb)]))
            if d < best_d:
                best = int(nb)
                best_d = d
        if best == cur:
            return cur
        cur, cur_d = best, best_d


def hnsw_seed(query, centroids, hgraph):
    top = hgraph["layers"][2]
    top_d = np.linalg.norm(centroids[top] - query[None, :], axis=1)
    cur = int(top[int(np.argmin(top_d))])
    for layer in [2, 1, 0]:
        cur = greedy_seed(query, centroids, hgraph["graphs"][layer], cur)
    return cur


def erng_seed(query, centroids, graph, probes):
    best = None
    for start in probes:
        c = greedy_seed(query, centroids, graph, int(start))
        d = float(np.linalg.norm(query - centroids[c]))
        if best is None or d < best[1]:
            best = (c, d)
    return int(best[0])


def build_cluster_candidates(bank, centroids, labels, per_cluster_cap):
    """Nearest members to each centroid, capped for fast reranking."""
    cluster_idxs = []
    for c in range(int(labels.max()) + 1):
        idx = np.where(labels == c)[0].astype(np.int32)
        if len(idx) > per_cluster_cap:
            sub = bank[idx]
            diff = sub - centroids[c][None, :]
            d2 = np.einsum("ij,ij->i", diff, diff)
            idx = idx[np.argsort(d2)[:per_cluster_cap]]
        cluster_idxs.append(idx.astype(np.int32))
    return cluster_idxs


def neighbor_centroids(seed, centroids, n_centroids):
    d = np.linalg.norm(centroids - centroids[int(seed)][None, :], axis=1)
    return np.argsort(d)[:n_centroids].astype(np.int32)


def search_fast(query, bank, centroids, cluster_short, seed, n_centroids, max_candidates):
    cids = neighbor_centroids(seed, centroids, n_centroids)
    idxs = [cluster_short[int(c)] for c in cids if len(cluster_short[int(c)])]
    if not idxs:
        idx = np.arange(min(max_candidates, bank.shape[0]), dtype=np.int32)
    else:
        idx = np.unique(np.concatenate(idxs))
    if len(idx) > max_candidates:
        # Pick the most query-relevant candidates from the centroid shortlist.
        sub = bank[idx]
        diff = sub - query[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        idx = idx[np.argsort(d2)[:max_candidates]]
    sub = bank[idx]
    diff = sub - query[None, :]
    d2 = np.einsum("ij,ij->i", diff, diff)
    j = int(np.argmin(d2))
    return float(np.sqrt(d2[j])), int(idx[j]), int(len(idx))


def direct_batch(q, bank):
    out_d = np.empty(q.shape[0], dtype=np.float32)
    out_i = np.empty(q.shape[0], dtype=np.int32)
    for i, v in enumerate(q):
        diff = bank - v[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        j = int(np.argmin(d2))
        out_d[i] = np.sqrt(d2[j])
        out_i[i] = j
    return out_d, out_i


def method_batch(q, bank, centroids, cluster_short, seed_fn, n_centroids, max_candidates):
    out_d = np.empty(q.shape[0], dtype=np.float32)
    out_i = np.empty(q.shape[0], dtype=np.int32)
    counts = np.empty(q.shape[0], dtype=np.int32)
    for i, v in enumerate(q):
        seed = seed_fn(v)
        out_d[i], out_i[i], counts[i] = search_fast(v, bank, centroids, cluster_short, seed, n_centroids, max_candidates)
    return out_d, out_i, counts


def total_acc(kd, ud, thr):
    known = float((kd <= thr).mean() * 100.0)
    unknown = float((ud > thr).mean() * 100.0)
    return round((known + unknown) / 2.0, 2)


def evaluate_config(ds, model, known, unknown, bank, centroids, hgraph, erng, probes, cluster_short, get_thr, q, h_cfg, e_cfg):
    qk, qu = C.pick_rows(known, unknown, q)
    agg = {k: [] for k in [
        "direct_total", "direct_ms",
        "h_total", "h_ms", "h_match", "h_recall", "h_count",
        "e_total", "e_ms", "e_match", "e_recall", "e_count",
    ]}
    for ai, (label, attack_name) in enumerate(G.ATTACKS, 1):
        print(f"[FastANN] {ds} q={q} attack {ai:02d}/26 {label}", flush=True)
        kemb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
        uemb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
        allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
        thr = get_thr(q, label)

        t0 = time.perf_counter()
        dd, di = direct_batch(allq, bank)
        dt = time.perf_counter() - t0

        t0 = time.perf_counter()
        hd, hi, hc = method_batch(
            allq, bank, centroids, cluster_short,
            lambda v: hnsw_seed(v, centroids, hgraph),
            h_cfg["centroids"], h_cfg["candidates"],
        )
        ht = time.perf_counter() - t0

        t0 = time.perf_counter()
        ed, ei, ec = method_batch(
            allq, bank, centroids, cluster_short,
            lambda v: erng_seed(v, centroids, erng, probes),
            e_cfg["centroids"], e_cfg["candidates"],
        )
        et = time.perf_counter() - t0

        h_dec = np.concatenate([(hd[:q] <= thr) == (dd[:q] <= thr), (hd[q:] > thr) == (dd[q:] > thr)])
        e_dec = np.concatenate([(ed[:q] <= thr) == (dd[:q] <= thr), (ed[q:] > thr) == (dd[q:] > thr)])
        n = float(2 * q)
        agg["direct_total"].append(total_acc(dd[:q], dd[q:], thr))
        agg["direct_ms"].append(dt * 1000.0 / n)
        agg["h_total"].append(total_acc(hd[:q], hd[q:], thr))
        agg["h_ms"].append(ht * 1000.0 / n)
        agg["h_match"].append(float(np.mean(h_dec) * 100.0))
        agg["h_recall"].append(float(np.mean(hi == di) * 100.0))
        agg["h_count"].append(float(np.mean(hc)))
        agg["e_total"].append(total_acc(ed[:q], ed[q:], thr))
        agg["e_ms"].append(et * 1000.0 / n)
        agg["e_match"].append(float(np.mean(e_dec) * 100.0))
        agg["e_recall"].append(float(np.mean(ei == di) * 100.0))
        agg["e_count"].append(float(np.mean(ec)))
    return {
        "direct_avg": round(float(np.mean(agg["direct_total"])), 2),
        "direct_ms": round(float(np.mean(agg["direct_ms"])), 6),
        "hnsw_avg": round(float(np.mean(agg["h_total"])), 2),
        "hnsw_ms": round(float(np.mean(agg["h_ms"])), 6),
        "hnsw_min_match": round(float(np.min(agg["h_match"])), 2),
        "hnsw_recall": round(float(np.mean(agg["h_recall"])), 2),
        "hnsw_candidates": round(float(np.mean(agg["h_count"])), 1),
        "erng_avg": round(float(np.mean(agg["e_total"])), 2),
        "erng_ms": round(float(np.mean(agg["e_ms"])), 6),
        "erng_min_match": round(float(np.min(agg["e_match"])), 2),
        "erng_recall": round(float(np.mean(agg["e_recall"])), 2),
        "erng_candidates": round(float(np.mean(agg["e_count"])), 1),
    }


def choose_configs(ds, model, known, unknown, bank, centroids, hgraph, erng, probes, labels, get_thr):
    q = min(100, len(known), len(unknown))
    caps = [64, 128, 256, 512, 1024, 2048]
    cnums = [1, 2, 4, 8, 12, 16]
    cluster_short = build_cluster_candidates(bank, centroids, labels, per_cluster_cap=max(caps))
    best_h = None
    best_e = None
    # Calibration on original, rotate_m10, crop40, watermark_asset, contrast to save time.
    cal_attacks = [x for x in G.ATTACKS if x[0] in {"original", "rotate_m10", "crop40", "watermark_asset", "contrast"}]
    saved_attacks = G.ATTACKS
    G.ATTACKS = cal_attacks
    try:
        for cnum in cnums:
            for cap in caps:
                cfg = {"centroids": cnum, "candidates": cap}
                row = evaluate_config(
                    ds, model, known, unknown, bank, centroids, hgraph, erng, probes, cluster_short,
                    get_thr, q, cfg, cfg,
                )
                print(f"[FastANNCal] {ds} c={cnum} cap={cap} row={row}", flush=True)
                h_ok = row["hnsw_min_match"] >= 100.0
                e_ok = row["erng_min_match"] >= 100.0
                if h_ok and (best_h is None or row["hnsw_ms"] < best_h["ms"]):
                    best_h = {"centroids": cnum, "candidates": cap, "ms": row["hnsw_ms"]}
                if e_ok and (best_e is None or row["erng_ms"] < best_e["ms"]):
                    best_e = {"centroids": cnum, "candidates": cap, "ms": row["erng_ms"]}
            if best_h and best_e:
                break
    finally:
        G.ATTACKS = saved_attacks
    return cluster_short, best_h or {"centroids": 8, "candidates": 1024}, best_e or {"centroids": 8, "candidates": 1024}


def run_dataset(ds):
    cfg = G.DATASETS[ds]
    out_dir = ROOT / ds / "phase3_centroid_fast_ann"
    out_dir.mkdir(parents=True, exist_ok=True)
    known, unknown, bank = G.load_rows(cfg["store"])
    q_sizes, get_thr = G.q_sizes_and_thresholds(cfg["eval"], cfg.get("policy"))
    model = G.load_model(cfg["ckpt"])
    centroids, labels, cpath, _ = C.build_centroids(ds, bank, ROOT / ds / "phase3_centroid_hnsw_erng")
    hgraph = C.build_custom_hnsw(centroids, local_m=8)
    erng = C.build_erng(centroids, local_m=8)
    rng = np.random.default_rng(20260602)
    central = np.argsort(hgraph["centrality"])[:4].astype(np.int32).tolist()
    random_probes = rng.choice(np.arange(centroids.shape[0]), size=min(8, centroids.shape[0]), replace=False).astype(np.int32).tolist()
    probes = sorted(set(central + random_probes))
    print(f"[FastANN] dataset={ds} store={len(known)}/{len(unknown)} centroids={len(centroids)}", flush=True)
    cluster_short, h_cfg, e_cfg = choose_configs(ds, model, known, unknown, bank, centroids, hgraph, erng, probes, labels, get_thr)
    print(f"[FastANN] chosen {ds} hnsw={h_cfg} erng={e_cfg}", flush=True)
    payload = {
        "dataset": ds,
        "known_count": len(known),
        "unknown_count": len(unknown),
        "centroid_count": int(len(centroids)),
        "hnsw_config": h_cfg,
        "erng_config": e_cfg,
        "queries": {},
    }
    for q_raw in q_sizes:
        q = min(int(q_raw), len(known), len(unknown))
        print(f"[FastANN] full {ds} q={q}", flush=True)
        row = evaluate_config(ds, model, known, unknown, bank, centroids, hgraph, erng, probes, cluster_short, get_thr, q, h_cfg, e_cfg)
        payload["queries"][str(q)] = row
        print(f"[FastANNSummary] {ds} q={q} {row}", flush=True)
    out = out_dir / f"{ds}_fast_centroid_hnsw_erng_timing.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[FastANN] saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(G.DATASETS), required=True)
    args = ap.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
