#!/usr/bin/env python3
"""Exact centroid branch-and-bound timing for HNSW/ERNG routed search.

This is the efficient correction for the centroid graph implementation:
graph routing provides an initial candidate, then centroid radii provide a
mathematical lower bound to skip clusters that cannot beat the current best.

For a centroid C with radius R and query q:
  lower_bound(cluster) = max(0, ||q-C|| - R)
If lower_bound >= current_best_distance, no point in that cluster can be closer.

This preserves exact nearest-neighbor recall/decision while avoiding dense search
when the cluster bounds are useful.
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


def greedy(query, centroids, graph, start):
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
        cur = greedy(query, centroids, hgraph["graphs"][layer], cur)
    return cur


def erng_seed(query, centroids, graph, probes):
    best = None
    for start in probes:
        c = greedy(query, centroids, graph, int(start))
        d = float(np.linalg.norm(query - centroids[c]))
        if best is None or d < best[1]:
            best = (c, d)
    return int(best[0])


def cluster_data(bank, centroids, labels):
    clusters = []
    radii = []
    for c in range(int(labels.max()) + 1):
        idx = np.where(labels == c)[0].astype(np.int32)
        clusters.append(idx)
        if len(idx):
            diff = bank[idx] - centroids[c][None, :]
            d2 = np.einsum("ij,ij->i", diff, diff)
            radii.append(float(np.sqrt(d2.max())))
        else:
            radii.append(0.0)
    return clusters, np.array(radii, dtype=np.float32)


def search_cluster(query, bank, idx):
    if len(idx) == 0:
        return float("inf"), -1
    sub = bank[idx]
    diff = sub - query[None, :]
    d2 = np.einsum("ij,ij->i", diff, diff)
    j = int(np.argmin(d2))
    return float(np.sqrt(d2[j])), int(idx[j])


def direct_batch(q, bank):
    d = np.empty(q.shape[0], dtype=np.float32)
    idx = np.empty(q.shape[0], dtype=np.int32)
    for i, v in enumerate(q):
        d[i], idx[i] = search_cluster(v, bank, np.arange(bank.shape[0], dtype=np.int32))
    return d, idx


def exact_bb_one(query, bank, centroids, clusters, radii, seed):
    # Seed best from routed cluster first.
    best_d, best_i = search_cluster(query, bank, clusters[int(seed)])
    cdist = np.linalg.norm(centroids - query[None, :], axis=1)
    lower = np.maximum(0.0, cdist - radii)
    order = np.argsort(lower)
    searched = 0
    for c in order:
        c = int(c)
        if lower[c] >= best_d:
            break
        if c == int(seed):
            searched += len(clusters[c])
            continue
        d, i = search_cluster(query, bank, clusters[c])
        searched += len(clusters[c])
        if d < best_d:
            best_d, best_i = d, i
    return best_d, best_i, searched


def method_batch(q, bank, centroids, clusters, radii, seed_fn):
    d = np.empty(q.shape[0], dtype=np.float32)
    idx = np.empty(q.shape[0], dtype=np.int32)
    seen = np.empty(q.shape[0], dtype=np.int32)
    for i, v in enumerate(q):
        d[i], idx[i], seen[i] = exact_bb_one(v, bank, centroids, clusters, radii, seed_fn(v))
    return d, idx, seen


def total_acc(kd, ud, thr):
    known = float((kd <= thr).mean() * 100.0)
    unknown = float((ud > thr).mean() * 100.0)
    return round((known + unknown) / 2.0, 2)


def run_dataset(ds):
    cfg = G.DATASETS[ds]
    out_dir = ROOT / ds / "phase3_centroid_branchbound"
    out_dir.mkdir(parents=True, exist_ok=True)
    known, unknown, bank = G.load_rows(cfg["store"])
    q_sizes, get_thr = G.q_sizes_and_thresholds(cfg["eval"], cfg.get("policy"))
    model = G.load_model(cfg["ckpt"])
    centroids, labels, cpath, _ = C.build_centroids(ds, bank, ROOT / ds / "phase3_centroid_hnsw_erng")
    hgraph = C.build_custom_hnsw(centroids, local_m=8)
    erng = C.build_erng(centroids, local_m=8)
    clusters, radii = cluster_data(bank, centroids, labels)
    rng = np.random.default_rng(20260602)
    central = np.argsort(hgraph["centrality"])[:4].astype(np.int32).tolist()
    random_probes = rng.choice(np.arange(centroids.shape[0]), size=min(8, centroids.shape[0]), replace=False).astype(np.int32).tolist()
    probes = sorted(set(central + random_probes))
    payload = {
        "dataset": ds,
        "known_count": len(known),
        "unknown_count": len(unknown),
        "centroid_count": int(len(centroids)),
        "queries": {},
    }
    print(f"[BB] dataset={ds} store={len(known)}/{len(unknown)} centroids={len(centroids)}", flush=True)
    for q_raw in q_sizes:
        q = min(int(q_raw), len(known), len(unknown))
        qk, qu = C.pick_rows(known, unknown, q)
        agg = {k: [] for k in ["direct_total","direct_ms","h_total","h_ms","h_match","h_recall","h_seen","e_total","e_ms","e_match","e_recall","e_seen"]}
        print(f"[BB] {ds} q={q}", flush=True)
        for ai, (label, attack_name) in enumerate(G.ATTACKS, 1):
            print(f"[BB] {ds} q={q} attack {ai:02d}/26 {label}", flush=True)
            kemb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
            uemb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
            allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
            thr = get_thr(q, label)
            t0 = time.perf_counter(); dd, di = direct_batch(allq, bank); dt = time.perf_counter() - t0
            t0 = time.perf_counter(); hd, hi, hs = method_batch(allq, bank, centroids, clusters, radii, lambda v: hnsw_seed(v, centroids, hgraph)); ht = time.perf_counter() - t0
            t0 = time.perf_counter(); ed, ei, es = method_batch(allq, bank, centroids, clusters, radii, lambda v: erng_seed(v, centroids, erng, probes)); et = time.perf_counter() - t0
            n = float(2*q)
            hdec = np.concatenate([(hd[:q] <= thr) == (dd[:q] <= thr), (hd[q:] > thr) == (dd[q:] > thr)])
            edec = np.concatenate([(ed[:q] <= thr) == (dd[:q] <= thr), (ed[q:] > thr) == (dd[q:] > thr)])
            agg["direct_total"].append(total_acc(dd[:q], dd[q:], thr)); agg["direct_ms"].append(dt*1000/n)
            agg["h_total"].append(total_acc(hd[:q], hd[q:], thr)); agg["h_ms"].append(ht*1000/n); agg["h_match"].append(float(np.mean(hdec)*100)); agg["h_recall"].append(float(np.mean(hi==di)*100)); agg["h_seen"].append(float(np.mean(hs)))
            agg["e_total"].append(total_acc(ed[:q], ed[q:], thr)); agg["e_ms"].append(et*1000/n); agg["e_match"].append(float(np.mean(edec)*100)); agg["e_recall"].append(float(np.mean(ei==di)*100)); agg["e_seen"].append(float(np.mean(es)))
        row = {
            "direct_avg": round(float(np.mean(agg["direct_total"])), 2),
            "direct_ms": round(float(np.mean(agg["direct_ms"])), 6),
            "hnsw_avg": round(float(np.mean(agg["h_total"])), 2),
            "hnsw_ms": round(float(np.mean(agg["h_ms"])), 6),
            "hnsw_min_match": round(float(np.min(agg["h_match"])), 2),
            "hnsw_recall": round(float(np.mean(agg["h_recall"])), 2),
            "hnsw_seen": round(float(np.mean(agg["h_seen"])), 1),
            "erng_avg": round(float(np.mean(agg["e_total"])), 2),
            "erng_ms": round(float(np.mean(agg["e_ms"])), 6),
            "erng_min_match": round(float(np.min(agg["e_match"])), 2),
            "erng_recall": round(float(np.mean(agg["e_recall"])), 2),
            "erng_seen": round(float(np.mean(agg["e_seen"])), 1),
        }
        payload["queries"][str(q)] = row
        print(f"[BBSummary] {ds} q={q} {row}", flush=True)
    out = out_dir / f"{ds}_centroid_branchbound_timing.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[BB] saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(G.DATASETS), required=True)
    args = ap.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
