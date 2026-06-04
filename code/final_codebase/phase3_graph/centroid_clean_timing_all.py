#!/usr/bin/env python3
"""Clean timing pass for centroid HNSW and ERNG.

This reuses the already-built centroid graph JSON/depths, but times:
  1. direct dense search
  2. HNSW centroid search
  3. ERNG centroid search
separately. It deliberately avoids path tracing and avoids the old combined
HNSW+ERNG timer.
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
KNOWN_DEPTHS = {
    "cifar": {"h_depth": 4, "e_depth": 3},
    "flickr": {"h_depth": 3, "e_depth": 3},
    "ucid": {"h_depth": 2, "e_depth": 2},
    "amazon": {"h_depth": 2, "e_depth": 3},
}


def greedy_no_path(query, centroids, graph, start):
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
        cur = best
        cur_d = best_d


def hnsw_seed(query, centroids, hgraph):
    top = hgraph["layers"][2]
    top_d = np.linalg.norm(centroids[top] - query[None, :], axis=1)
    cur = int(top[int(np.argmin(top_d))])
    for layer in [2, 1, 0]:
        cur = greedy_no_path(query, centroids, hgraph["graphs"][layer], cur)
    return cur


def erng_seed(query, centroids, graph, probes):
    best_c = None
    best_d = None
    for start in probes:
        c = greedy_no_path(query, centroids, graph, int(start))
        d = float(np.linalg.norm(query - centroids[c]))
        if best_d is None or d < best_d:
            best_c = c
            best_d = d
    return int(best_c)


def make_candidate_cache(clusters, base_graph, depth):
    cache = {}
    for seed in sorted(base_graph.keys()):
        ids = C.expand_centroids([seed], base_graph, depth)
        idxs = []
        for c in ids:
            if 0 <= int(c) < len(clusters):
                idxs.append(clusters[int(c)])
        if idxs:
            idx = np.unique(np.concatenate(idxs).astype(np.int32))
        else:
            idx = np.arange(sum(len(x) for x in clusters), dtype=np.int32)
        cache[int(seed)] = idx
    return cache


def search_cached(query, bank, idx):
    sub = bank[idx]
    diff = sub - query[None, :]
    d2 = np.einsum("ij,ij->i", diff, diff)
    j = int(np.argmin(d2))
    return float(np.sqrt(d2[j])), int(idx[j]), int(len(idx))


def graph_batch_one(q, bank, cache, seed_fn):
    dists = np.empty(q.shape[0], dtype=np.float32)
    idxs = np.empty(q.shape[0], dtype=np.int32)
    counts = np.empty(q.shape[0], dtype=np.int32)
    for i, v in enumerate(q):
        seed = seed_fn(v)
        dists[i], idxs[i], n = search_cached(v, bank, cache[int(seed)])
        counts[i] = n
    return dists, idxs, counts


def total_acc(kd, ud, thr):
    known = float((kd <= thr).mean() * 100.0)
    unknown = float((ud > thr).mean() * 100.0)
    return round((known + unknown) / 2.0, 2)


def run_dataset(ds):
    cfg = G.DATASETS[ds]
    out_dir = ROOT / ds / "phase3_centroid_hnsw_erng"
    old_path = out_dir / f"{ds}_centroid_hnsw_erng_allq.json"
    old = json.load(open(old_path)) if old_path.exists() else {}
    h_depth = int(KNOWN_DEPTHS.get(ds, {}).get("h_depth", old.get("calibration", {}).get("h_depth", 99)))
    e_depth = int(KNOWN_DEPTHS.get(ds, {}).get("e_depth", old.get("calibration", {}).get("e_depth", 99)))

    known, unknown, bank = G.load_rows(cfg["store"])
    q_sizes, get_thr = G.q_sizes_and_thresholds(cfg["eval"], cfg.get("policy"))
    model = G.load_model(cfg["ckpt"])
    centroids, labels, cpath, _ = C.build_centroids(ds, bank, out_dir)
    clusters = C.cluster_members(labels)
    hgraph = C.build_custom_hnsw(centroids, local_m=8)
    erng = C.build_erng(centroids, local_m=8)
    h_cache = make_candidate_cache(clusters, hgraph["graphs"][0], h_depth)
    e_cache = make_candidate_cache(clusters, erng, e_depth)
    probes = old.get("probes")
    if not probes:
        rng = np.random.default_rng(20260602)
        central = np.argsort(hgraph["centrality"])[:4].astype(np.int32).tolist()
        random_probes = rng.choice(np.arange(centroids.shape[0]), size=min(8, centroids.shape[0]), replace=False).astype(np.int32).tolist()
        probes = sorted(set(central + random_probes))

    print(f"[CleanTiming] dataset={ds} store={len(known)}/{len(unknown)} centroids={len(centroids)} h_depth={h_depth} e_depth={e_depth}", flush=True)
    result = {
        "dataset": ds,
        "source": str(old_path),
        "known_count": len(known),
        "unknown_count": len(unknown),
        "centroid_count": int(len(centroids)),
        "h_depth": h_depth,
        "e_depth": e_depth,
        "probes": probes,
        "queries": {},
    }

    for q_raw in q_sizes:
        q = min(int(q_raw), len(known), len(unknown))
        qk, qu = C.pick_rows(known, unknown, q)
        sums = {
            "direct_total": [],
            "direct_ms": [],
            "hnsw_total": [],
            "hnsw_ms": [],
            "hnsw_match": [],
            "hnsw_recall": [],
            "hnsw_candidates": [],
            "erng_total": [],
            "erng_ms": [],
            "erng_match": [],
            "erng_recall": [],
            "erng_candidates": [],
        }
        print(f"[CleanTiming] {ds} q={q}", flush=True)
        for ai, (label, attack_name) in enumerate(G.ATTACKS, 1):
            print(f"[CleanTiming] {ds} q={q} attack {ai:02d}/26 {label}", flush=True)
            kemb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
            uemb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
            allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
            thr = get_thr(q, label)

            t0 = time.perf_counter()
            dd, di = C.direct_batch(allq, bank)
            direct_elapsed = time.perf_counter() - t0

            t0 = time.perf_counter()
            hd, hi, hc = graph_batch_one(
                allq,
                bank,
                h_cache,
                lambda v: hnsw_seed(v, centroids, hgraph),
            )
            h_elapsed = time.perf_counter() - t0

            t0 = time.perf_counter()
            ed, ei, ec = graph_batch_one(
                allq,
                bank,
                e_cache,
                lambda v: erng_seed(v, centroids, erng, probes),
            )
            e_elapsed = time.perf_counter() - t0

            h_dec = np.concatenate([(hd[:q] <= thr) == (dd[:q] <= thr), (hd[q:] > thr) == (dd[q:] > thr)])
            e_dec = np.concatenate([(ed[:q] <= thr) == (dd[:q] <= thr), (ed[q:] > thr) == (dd[q:] > thr)])

            n = float(2 * q)
            sums["direct_total"].append(total_acc(dd[:q], dd[q:], thr))
            sums["direct_ms"].append(direct_elapsed * 1000.0 / n)
            sums["hnsw_total"].append(total_acc(hd[:q], hd[q:], thr))
            sums["hnsw_ms"].append(h_elapsed * 1000.0 / n)
            sums["hnsw_match"].append(float(np.mean(h_dec) * 100.0))
            sums["hnsw_recall"].append(float(np.mean(hi == di) * 100.0))
            sums["hnsw_candidates"].append(float(np.mean(hc)))
            sums["erng_total"].append(total_acc(ed[:q], ed[q:], thr))
            sums["erng_ms"].append(e_elapsed * 1000.0 / n)
            sums["erng_match"].append(float(np.mean(e_dec) * 100.0))
            sums["erng_recall"].append(float(np.mean(ei == di) * 100.0))
            sums["erng_candidates"].append(float(np.mean(ec)))

        row = {
            "direct_avg": round(float(np.mean(sums["direct_total"])), 2),
            "direct_ms": round(float(np.mean(sums["direct_ms"])), 6),
            "hnsw_avg": round(float(np.mean(sums["hnsw_total"])), 2),
            "hnsw_ms": round(float(np.mean(sums["hnsw_ms"])), 6),
            "hnsw_min_match": round(float(np.min(sums["hnsw_match"])), 2),
            "hnsw_recall": round(float(np.mean(sums["hnsw_recall"])), 2),
            "hnsw_candidates": round(float(np.mean(sums["hnsw_candidates"])), 1),
            "erng_avg": round(float(np.mean(sums["erng_total"])), 2),
            "erng_ms": round(float(np.mean(sums["erng_ms"])), 6),
            "erng_min_match": round(float(np.min(sums["erng_match"])), 2),
            "erng_recall": round(float(np.mean(sums["erng_recall"])), 2),
            "erng_candidates": round(float(np.mean(sums["erng_candidates"])), 1),
        }
        result["queries"][str(q)] = row
        print(f"[CleanTimingSummary] {ds} q={q} {row}", flush=True)

    out = out_dir / f"{ds}_centroid_hnsw_erng_clean_timing.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[CleanTiming] saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(G.DATASETS), required=True)
    args = ap.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
