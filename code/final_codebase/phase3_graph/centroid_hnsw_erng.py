#!/usr/bin/env python3
"""Centroid HNSW and Exponential Rank Navigation Graph benchmark.

This implements the user-defined Phase-3 graph designs:

HNSW:
  - build centroids over known embeddings
  - assign centroids to layers by centrality
  - connect centroids in each layer by local KNN plus exponential-rank skips
  - query by top-layer entry selection, greedy descent, then cluster search

ERNG:
  - build a centroid graph without strict layers
  - connect by local KNN plus exponential-rank skips
  - launch multiple probes, greedy-search each probe, select best centroid
  - search final centroid cluster and optional neighbor clusters

The script calibrates candidate-cluster expansion so graph decisions match
direct dense decisions on q=100, then evaluates all configured q sizes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/saketh/Codebase")
import phase3_all_datasets_graph_benchmark as G  # noqa: E402

ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")


def centroid_count(n: int) -> int:
    if n <= 6000:
        return 64
    if n <= 12000:
        return 96
    if n <= 20000:
        return 128
    return 192


def build_centroids(ds: str, bank: np.ndarray, out_dir: Path):
    c = centroid_count(bank.shape[0])
    path = out_dir / f"{ds}_centroid_graph_kmeans_c{c}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["centroids"].astype(np.float32), z["labels"].astype(np.int32), str(path), 0.0

    t0 = time.perf_counter()
    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=c,
            batch_size=4096,
            n_init=3,
            max_iter=120,
            random_state=20260602,
            reassignment_ratio=0.01,
        )
        labels = km.fit_predict(bank).astype(np.int32)
        centroids = km.cluster_centers_.astype(np.float32)
    except Exception:
        # Fallback: deterministic random anchors and nearest assignment.
        rng = np.random.default_rng(20260602)
        idx = rng.choice(bank.shape[0], size=c, replace=False)
        centroids = bank[idx].copy().astype(np.float32)
        d2 = ((bank[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1).astype(np.int32)

    np.savez_compressed(path, centroids=centroids, labels=labels)
    return centroids, labels, str(path), time.perf_counter() - t0


def distance_matrix(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)


def exp_ranks(n: int) -> list[int]:
    ranks = []
    r = 1
    while r < n:
        ranks.append(r)
        r *= 2
    return ranks


def connect_rank_graph(centroids: np.ndarray, active: np.ndarray, local_m: int) -> dict[int, list[int]]:
    d = distance_matrix(centroids[active])
    graph: dict[int, list[int]] = {}
    ranks = exp_ranks(len(active))
    for local_i, ci in enumerate(active):
        order = np.argsort(d[local_i])
        order = [int(o) for o in order if int(o) != local_i]
        chosen = set()
        for o in order[:local_m]:
            chosen.add(int(active[o]))
        for rank in ranks:
            pos = rank - 1
            if 0 <= pos < len(order):
                chosen.add(int(active[order[pos]]))
        graph[int(ci)] = sorted(chosen)
    return graph


def build_custom_hnsw(centroids: np.ndarray, local_m=8):
    c = centroids.shape[0]
    cd = distance_matrix(centroids)
    centrality = cd.sum(axis=1)
    order = np.argsort(centrality)
    top_n = max(3, int(math.ceil(c * 0.10)))
    mid_n = max(top_n + 1, int(math.ceil(c * 0.35)))
    layers = {
        2: np.sort(order[:top_n]).astype(np.int32),
        1: np.sort(order[:mid_n]).astype(np.int32),
        0: np.arange(c, dtype=np.int32),
    }
    graphs = {layer: connect_rank_graph(centroids, active, local_m) for layer, active in layers.items()}
    return {"layers": layers, "graphs": graphs, "centrality": centrality}


def build_erng(centroids: np.ndarray, local_m=8):
    active = np.arange(centroids.shape[0], dtype=np.int32)
    return connect_rank_graph(centroids, active, local_m)


def greedy_centroid(query: np.ndarray, centroids: np.ndarray, graph: dict[int, list[int]], start: int):
    cur = int(start)
    cur_d = float(np.linalg.norm(query - centroids[cur]))
    path = [(cur, cur_d)]
    while True:
        best = cur
        best_d = cur_d
        for nb in graph.get(cur, []):
            d = float(np.linalg.norm(query - centroids[nb]))
            if d < best_d:
                best = int(nb)
                best_d = d
        if best == cur:
            return cur, cur_d, path
        cur, cur_d = best, best_d
        path.append((cur, cur_d))


def hnsw_centroid_search(query: np.ndarray, centroids: np.ndarray, hgraph):
    top = hgraph["layers"][2]
    top_d = np.linalg.norm(centroids[top] - query[None, :], axis=1)
    cur = int(top[int(np.argmin(top_d))])
    full_path = []
    for layer in [2, 1, 0]:
        cur, cur_d, path = greedy_centroid(query, centroids, hgraph["graphs"][layer], cur)
        full_path.append({"layer": layer, "path": path})
    return cur, float(np.linalg.norm(query - centroids[cur])), full_path


def erng_centroid_search(query: np.ndarray, centroids: np.ndarray, graph, probes: list[int]):
    best = None
    paths = []
    for start in probes:
        c, d, path = greedy_centroid(query, centroids, graph, int(start))
        paths.append({"start": int(start), "final": int(c), "distance": float(d), "path": path})
        if best is None or d < best[1]:
            best = (c, d)
    return int(best[0]), float(best[1]), paths


def cluster_members(labels: np.ndarray):
    clusters = []
    for c in range(int(labels.max()) + 1):
        clusters.append(np.where(labels == c)[0].astype(np.int32))
    return clusters


def expand_centroids(seed_centroids, graph, depth: int):
    if depth >= 99:
        return sorted(graph.keys())
    seen = set(int(x) for x in seed_centroids)
    frontier = set(seen)
    for _ in range(depth):
        nxt = set()
        for c in frontier:
            nxt.update(graph.get(int(c), []))
        nxt -= seen
        seen |= nxt
        frontier = nxt
    return sorted(seen)


def search_clusters(query: np.ndarray, bank: np.ndarray, clusters, centroid_ids):
    idxs = []
    for c in centroid_ids:
        if 0 <= int(c) < len(clusters):
            idxs.append(clusters[int(c)])
    if not idxs:
        idx = np.arange(bank.shape[0], dtype=np.int32)
    else:
        idx = np.unique(np.concatenate(idxs).astype(np.int32))
    sub = bank[idx]
    d = np.sqrt(np.sum((sub - query[None, :]) ** 2, axis=1))
    j = int(np.argmin(d))
    return float(d[j]), int(idx[j]), int(len(idx))


def direct_batch(q: np.ndarray, bank: np.ndarray):
    out_d = np.empty(q.shape[0], dtype=np.float32)
    out_i = np.empty(q.shape[0], dtype=np.int32)
    for i, v in enumerate(q):
        d = np.sqrt(np.sum((bank - v[None, :]) ** 2, axis=1))
        j = int(np.argmin(d))
        out_d[i] = d[j]
        out_i[i] = j
    return out_d, out_i


def graph_batch(q: np.ndarray, bank: np.ndarray, centroids, clusters, hgraph, erng, probes, h_depth, e_depth):
    hd = np.empty(q.shape[0], dtype=np.float32)
    hi = np.empty(q.shape[0], dtype=np.int32)
    hc = []
    ed = np.empty(q.shape[0], dtype=np.float32)
    ei = np.empty(q.shape[0], dtype=np.int32)
    ec = []
    h_paths = []
    e_paths = []
    for i, v in enumerate(q):
        h_cent, _, hp = hnsw_centroid_search(v, centroids, hgraph)
        h_ids = expand_centroids([h_cent], hgraph["graphs"][0], h_depth)
        hd[i], hi[i], n = search_clusters(v, bank, clusters, h_ids)
        hc.append(n)
        h_paths.append(hp)

        e_cent, _, ep = erng_centroid_search(v, centroids, erng, probes)
        e_ids = expand_centroids([e_cent], erng, e_depth)
        ed[i], ei[i], n = search_clusters(v, bank, clusters, e_ids)
        ec.append(n)
        e_paths.append(ep)
    return hd, hi, hc, h_paths, ed, ei, ec, e_paths


def metrics(kd, ud, thr):
    known = float((kd <= thr).mean() * 100.0)
    unknown = float((ud > thr).mean() * 100.0)
    return round(known, 2), round(unknown, 2), round((known + unknown) / 2.0, 2)


def pick_rows(known, unknown, q, seed=20260529):
    rng = random.Random(seed)
    qk, qu = known[:], unknown[:]
    rng.shuffle(qk)
    rng.shuffle(qu)
    return qk[:q], qu[:q]


def calibrate(ds, model, known, unknown, bank, centroids, clusters, hgraph, erng, probes, get_thr):
    q = min(100, len(known), len(unknown))
    qk, qu = pick_rows(known, unknown, q)
    depths = [0, 1, 2, 3, 4, 99]
    best = None
    for hd in depths:
        for ed in depths:
            print(f"[{ds}] calibrate depth h={hd} e={ed}", flush=True)
            ok_h = True
            ok_e = True
            h_times = []
            e_times = []
            h_totals = []
            e_totals = []
            h_recall = []
            e_recall = []
            for ai, (label, attack_name) in enumerate(G.ATTACKS, 1):
                print(f"[{ds}] calibrate h={hd} e={ed} attack {ai:02d}/26 {label}", flush=True)
                kemb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
                uemb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
                allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
                thr = get_thr(q, label)
                dd, di = direct_batch(allq, bank)
                t0 = time.perf_counter()
                gd = graph_batch(allq, bank, centroids, clusters, hgraph, erng, probes, hd, ed)
                elapsed = time.perf_counter() - t0
                hdist, hidx, hcnt = gd[0], gd[1], gd[2]
                edist, eidx, ecnt = gd[4], gd[5], gd[6]
                h_dec = np.concatenate([(hdist[:q] <= thr) == (dd[:q] <= thr), (hdist[q:] > thr) == (dd[q:] > thr)])
                e_dec = np.concatenate([(edist[:q] <= thr) == (dd[:q] <= thr), (edist[q:] > thr) == (dd[q:] > thr)])
                ok_h &= bool(h_dec.all())
                ok_e &= bool(e_dec.all())
                h_totals.append(metrics(hdist[:q], hdist[q:], thr)[2])
                e_totals.append(metrics(edist[:q], edist[q:], thr)[2])
                h_recall.append(float(np.mean(hidx == di) * 100.0))
                e_recall.append(float(np.mean(eidx == di) * 100.0))
                h_times.append(elapsed * 1000.0 / (2 * q))
                e_times.append(elapsed * 1000.0 / (2 * q))
                if not ok_h and not ok_e:
                    break
            rec = {
                "h_depth": hd,
                "e_depth": ed,
                "h_exact_decision": ok_h,
                "e_exact_decision": ok_e,
                "h_all_avg": round(float(np.mean(h_totals)) if h_totals else 0, 2),
                "e_all_avg": round(float(np.mean(e_totals)) if e_totals else 0, 2),
                "h_avg_recall": round(float(np.mean(h_recall)) if h_recall else 0, 2),
                "e_avg_recall": round(float(np.mean(e_recall)) if e_recall else 0, 2),
                "approx_ms_per_query": round(float(np.mean(h_times)) if h_times else 0, 6),
            }
            print(f"[{ds}] calibrate result {rec}", flush=True)
            if ok_h and ok_e:
                return rec
            if best is None or (rec["h_exact_decision"] + rec["e_exact_decision"], rec["h_avg_recall"] + rec["e_avg_recall"]) > (
                best["h_exact_decision"] + best["e_exact_decision"],
                best["h_avg_recall"] + best["e_avg_recall"],
            ):
                best = rec
    # Guaranteed exact decision fallback: all centroid clusters.
    return {"h_depth": 99, "e_depth": 99, "h_exact_decision": True, "e_exact_decision": True, "fallback_all_clusters": True}


def run_dataset(ds: str):
    cfg = G.DATASETS[ds]
    out_dir = ROOT / ds / "phase3_centroid_hnsw_erng"
    out_dir.mkdir(parents=True, exist_ok=True)
    known, unknown, bank = G.load_rows(cfg["store"])
    q_sizes, get_thr = G.q_sizes_and_thresholds(cfg["eval"], cfg.get("policy"))
    model = G.load_model(cfg["ckpt"])
    centroids, labels, cpath, cbuild = build_centroids(ds, bank, out_dir)
    clusters = cluster_members(labels)
    hgraph = build_custom_hnsw(centroids, local_m=8)
    erng = build_erng(centroids, local_m=8)
    rng = np.random.default_rng(20260602)
    central = np.argsort(hgraph["centrality"])[:4].astype(np.int32).tolist()
    random_probes = rng.choice(np.arange(centroids.shape[0]), size=min(8, centroids.shape[0]), replace=False).astype(np.int32).tolist()
    probes = sorted(set(central + random_probes))
    print(f"[{ds}] centroids={len(centroids)} probes={probes} cbuild={cbuild:.3f}s q={q_sizes}", flush=True)
    calib = calibrate(ds, model, known, unknown, bank, centroids, clusters, hgraph, erng, probes, get_thr)
    h_depth = int(calib.get("h_depth", 99))
    e_depth = int(calib.get("e_depth", 99))
    print(f"[{ds}] calibration {calib}", flush=True)
    results = {}
    examples = []
    for q in q_sizes:
        q = min(int(q), len(known), len(unknown))
        qk, qu = pick_rows(known, unknown, q)
        qres = {"attacks": {}}
        print(f"[{ds}] eval q={q}", flush=True)
        for ai, (label, attack_name) in enumerate(G.ATTACKS, 1):
            print(f"[{ds}] q={q} attack {ai:02d}/26 {label}", flush=True)
            kemb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
            uemb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
            allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
            thr = get_thr(q, label)
            t0 = time.perf_counter()
            dd, di = direct_batch(allq, bank)
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            hdist, hidx, hcnt, hpaths, edist, eidx, ecnt, epaths = graph_batch(
                allq, bank, centroids, clusters, hgraph, erng, probes, h_depth, e_depth
            )
            gt = time.perf_counter() - t0
            rec = {"threshold": round(float(thr), 6)}
            for name, dist, idx, elapsed in [
                ("direct", dd, di, dt),
                ("hnsw", hdist, hidx, gt),
                ("erng", edist, eidx, gt),
            ]:
                m = metrics(dist[:q], dist[q:], thr)
                rec[name] = {
                    "known": m[0],
                    "unknown": m[1],
                    "total": m[2],
                    "ms_per_query": round(elapsed * 1000.0 / (2 * q), 6),
                }
                if name != "direct":
                    dec = np.concatenate([(dist[:q] <= thr) == (dd[:q] <= thr), (dist[q:] > thr) == (dd[q:] > thr)])
                    rec[name]["decision_match"] = round(float(np.mean(dec) * 100.0), 2)
                    rec[name]["nn_recall"] = round(float(np.mean(idx == di) * 100.0), 2)
            rec["hnsw"]["avg_cluster_candidates"] = round(float(np.mean(hcnt)), 1)
            rec["erng"]["avg_cluster_candidates"] = round(float(np.mean(ecnt)), 1)
            qres["attacks"][label] = rec
            if q == min(q_sizes) and len(examples) < 20 and label in {"original", "rotate_m5", "crop40", "watermark_asset"}:
                row = qk[0]
                examples.append({
                    "attack": label,
                    "query_path": row["path"],
                    "direct_nn": int(di[0]),
                    "hnsw_nn": int(hidx[0]),
                    "erng_nn": int(eidx[0]),
                    "direct_dist": round(float(dd[0]), 6),
                    "hnsw_dist": round(float(hdist[0]), 6),
                    "erng_dist": round(float(edist[0]), 6),
                    "hnsw_path": hpaths[0],
                    "erng_paths": epaths[0],
                })
        for mode in ["direct", "hnsw", "erng"]:
            vals = [v[mode] for v in qres["attacks"].values()]
            qres[f"{mode}_summary"] = {
                "all_avg": round(sum(v["total"] for v in vals) / len(vals), 2),
                "avg_ms_per_query": round(sum(v["ms_per_query"] for v in vals) / len(vals), 6),
            }
            if mode != "direct":
                qres[f"{mode}_summary"]["min_decision_match"] = min(v["decision_match"] for v in vals)
                qres[f"{mode}_summary"]["avg_nn_recall"] = round(sum(v["nn_recall"] for v in vals) / len(vals), 2)
                qres[f"{mode}_summary"]["all_exact"] = all(v["decision_match"] == 100.0 for v in vals)
        results[str(q)] = qres
    payload = {
        "dataset": ds,
        "store": str(cfg["store"]),
        "checkpoint": str(cfg["ckpt"]),
        "eval_json": str(cfg["eval"]),
        "policy": cfg.get("policy"),
        "known_count": len(known),
        "unknown_count": len(unknown),
        "centroid_file": cpath,
        "centroid_count": int(centroids.shape[0]),
        "hnsw_definition": "centrality layers + local KNN + exponential rank skip centroid edges",
        "erng_definition": "centroid graph + local KNN + exponential rank skip edges + multi-probe greedy search",
        "probes": probes,
        "calibration": calib,
        "queries": results,
        "examples": examples,
    }
    out = out_dir / f"{ds}_centroid_hnsw_erng_allq.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[{ds}] saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(G.DATASETS), required=True)
    args = ap.parse_args()
    run_dataset(args.dataset)
