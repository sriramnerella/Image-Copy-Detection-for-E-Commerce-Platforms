#!/usr/bin/env python3
"""Auto-tune centroid count for exact HNSW/ERNG branch-bound search."""
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
GRID = {
    "ucid": [96, 128, 192, 256, 384, 512, 768],
    "flickr": [128, 192, 256, 384, 512, 768, 1024],
    "cifar": [192, 256, 384, 512, 768, 1024, 1536],
    "amazon": [256, 384, 512, 768, 1024, 1536, 2048],
}
CAL_ATTACKS = {"original", "rotate_m10", "crop40", "watermark_asset", "contrast"}


def build_centroids(ds, bank, c, out_dir):
    c = min(int(c), bank.shape[0])
    path = out_dir / f"{ds}_autotune_centroids_c{c}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["centroids"].astype(np.float32), z["labels"].astype(np.int32), str(path)
    print(f"[AutoBB] building centroids ds={ds} c={c}", flush=True)
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=c,
            batch_size=4096,
            n_init=1,
            max_iter=45,
            random_state=20260603 + c,
            reassignment_ratio=0.005,
        )
        labels = km.fit_predict(bank).astype(np.int32)
        centroids = km.cluster_centers_.astype(np.float32)
    except Exception:
        rng = np.random.default_rng(20260603 + c)
        idx = rng.choice(bank.shape[0], size=c, replace=False)
        centroids = bank[idx].copy().astype(np.float32)
        labels = np.empty(bank.shape[0], dtype=np.int32)
        for s in range(0, bank.shape[0], 2048):
            x = bank[s:s+2048]
            d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels[s:s+len(x)] = np.argmin(d2, axis=1).astype(np.int32)
    np.savez_compressed(path, centroids=centroids, labels=labels)
    return centroids, labels, str(path)


def greedy(query, centroids, graph, start):
    cur = int(start); cur_d = float(np.linalg.norm(query - centroids[cur]))
    while True:
        best = cur; best_d = cur_d
        for nb in graph.get(cur, []):
            d = float(np.linalg.norm(query - centroids[int(nb)]))
            if d < best_d:
                best = int(nb); best_d = d
        if best == cur:
            return cur
        cur, cur_d = best, best_d


def hseed(query, centroids, hgraph):
    top = hgraph["layers"][2]
    cur = int(top[int(np.argmin(np.linalg.norm(centroids[top] - query[None, :], axis=1)))])
    for layer in [2, 1, 0]:
        cur = greedy(query, centroids, hgraph["graphs"][layer], cur)
    return cur


def eseed(query, centroids, graph, probes):
    best = None
    for start in probes:
        c = greedy(query, centroids, graph, int(start))
        d = float(np.linalg.norm(query - centroids[c]))
        if best is None or d < best[1]:
            best = (c, d)
    return int(best[0])


def prep_clusters(bank, centroids, labels):
    clusters, radii = [], []
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


def search_idx(query, bank, idx):
    if len(idx) == 0: return float("inf"), -1
    sub = bank[idx]
    diff = sub - query[None, :]
    d2 = np.einsum("ij,ij->i", diff, diff)
    j = int(np.argmin(d2))
    return float(np.sqrt(d2[j])), int(idx[j])


def direct_batch(q, bank):
    idx = np.arange(bank.shape[0], dtype=np.int32)
    d = np.empty(q.shape[0], dtype=np.float32); nn = np.empty(q.shape[0], dtype=np.int32)
    for i, v in enumerate(q):
        d[i], nn[i] = search_idx(v, bank, idx)
    return d, nn


def bb_one(query, bank, centroids, clusters, radii, seed, init_k):
    cdist = np.linalg.norm(centroids - query[None, :], axis=1)
    lower = np.maximum(0.0, cdist - radii)
    best_d = float("inf"); best_i = -1; seen = 0; done = set()
    init = list(np.argsort(cdist)[:init_k])
    if seed not in init: init.append(int(seed))
    for c in init:
        c = int(c); d, i = search_idx(query, bank, clusters[c]); seen += len(clusters[c]); done.add(c)
        if d < best_d: best_d, best_i = d, i
    for c in np.argsort(lower):
        c = int(c)
        if lower[c] >= best_d: break
        if c in done: continue
        d, i = search_idx(query, bank, clusters[c]); seen += len(clusters[c])
        if d < best_d: best_d, best_i = d, i
    return best_d, best_i, seen


def method_batch(q, bank, centroids, clusters, radii, seed_fn, init_k):
    d = np.empty(q.shape[0], dtype=np.float32); nn = np.empty(q.shape[0], dtype=np.int32); seen = np.empty(q.shape[0], dtype=np.int32)
    for i, v in enumerate(q):
        d[i], nn[i], seen[i] = bb_one(v, bank, centroids, clusters, radii, seed_fn(v), init_k)
    return d, nn, seen


def total_acc(kd, ud, thr):
    return round((float((kd <= thr).mean()*100) + float((ud > thr).mean()*100))/2, 2)


def eval_rows(ds, model, known, unknown, bank, centroids, labels, get_thr, q, attacks):
    hgraph = C.build_custom_hnsw(centroids, local_m=12)
    erng = C.build_erng(centroids, local_m=12)
    clusters, radii = prep_clusters(bank, centroids, labels)
    rng = np.random.default_rng(20260603)
    central = np.argsort(hgraph["centrality"])[:4].astype(np.int32).tolist()
    random_probes = rng.choice(np.arange(centroids.shape[0]), size=min(4, centroids.shape[0]), replace=False).astype(np.int32).tolist()
    probes = sorted(set(central + random_probes))
    init_k = 2
    qk, qu = C.pick_rows(known, unknown, q)
    agg = {k: [] for k in ["direct_total","direct_ms","h_total","h_ms","h_match","h_recall","h_seen","e_total","e_ms","e_match","e_recall","e_seen"]}
    for ai, (label, attack_name) in enumerate(attacks, 1):
        print(f"[AutoBB] {ds} q={q} attack {ai:02d}/{len(attacks)} {label}", flush=True)
        kemb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
        uemb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
        allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
        thr = get_thr(q, label)
        t0=time.perf_counter(); dd,di=direct_batch(allq,bank); dt=time.perf_counter()-t0
        t0=time.perf_counter(); hd,hi,hs=method_batch(allq,bank,centroids,clusters,radii,lambda v: hseed(v,centroids,hgraph),init_k); ht=time.perf_counter()-t0
        t0=time.perf_counter(); ed,ei,es=method_batch(allq,bank,centroids,clusters,radii,lambda v: eseed(v,centroids,erng,probes),init_k); et=time.perf_counter()-t0
        n=float(2*q)
        hdec=np.concatenate([(hd[:q]<=thr)==(dd[:q]<=thr),(hd[q:]>thr)==(dd[q:]>thr)])
        edec=np.concatenate([(ed[:q]<=thr)==(dd[:q]<=thr),(ed[q:]>thr)==(dd[q:]>thr)])
        agg["direct_total"].append(total_acc(dd[:q],dd[q:],thr)); agg["direct_ms"].append(dt*1000/n)
        agg["h_total"].append(total_acc(hd[:q],hd[q:],thr)); agg["h_ms"].append(ht*1000/n); agg["h_match"].append(float(np.mean(hdec)*100)); agg["h_recall"].append(float(np.mean(hi==di)*100)); agg["h_seen"].append(float(np.mean(hs)))
        agg["e_total"].append(total_acc(ed[:q],ed[q:],thr)); agg["e_ms"].append(et*1000/n); agg["e_match"].append(float(np.mean(edec)*100)); agg["e_recall"].append(float(np.mean(ei==di)*100)); agg["e_seen"].append(float(np.mean(es)))
    return {
        "direct_avg": round(float(np.mean(agg["direct_total"])),2),
        "direct_ms": round(float(np.mean(agg["direct_ms"])),6),
        "hnsw_avg": round(float(np.mean(agg["h_total"])),2),
        "hnsw_ms": round(float(np.mean(agg["h_ms"])),6),
        "hnsw_min_match": round(float(np.min(agg["h_match"])),2),
        "hnsw_recall": round(float(np.mean(agg["h_recall"])),2),
        "hnsw_seen": round(float(np.mean(agg["h_seen"])),1),
        "erng_avg": round(float(np.mean(agg["e_total"])),2),
        "erng_ms": round(float(np.mean(agg["e_ms"])),6),
        "erng_min_match": round(float(np.min(agg["e_match"])),2),
        "erng_recall": round(float(np.mean(agg["e_recall"])),2),
        "erng_seen": round(float(np.mean(agg["e_seen"])),1),
    }


def run_dataset(ds):
    cfg = G.DATASETS[ds]
    out_dir = ROOT / ds / "phase3_centroid_branchbound_autotune"
    out_dir.mkdir(parents=True, exist_ok=True)
    known, unknown, bank = G.load_rows(cfg["store"])
    q_sizes, get_thr = G.q_sizes_and_thresholds(cfg["eval"], cfg.get("policy"))
    model = G.load_model(cfg["ckpt"])
    cal_attacks = [x for x in G.ATTACKS if x[0] in CAL_ATTACKS]
    q_cal = min(100, len(known), len(unknown))
    tried = []
    best = None
    for c in GRID[ds]:
        centroids, labels, cpath = build_centroids(ds, bank, c, out_dir)
        row = eval_rows(ds, model, known, unknown, bank, centroids, labels, get_thr, q_cal, cal_attacks)
        row["centroids"] = int(len(centroids)); row["centroid_file"] = cpath
        tried.append(row)
        valid = row["hnsw_min_match"] == 100.0 and row["erng_min_match"] == 100.0 and row["hnsw_recall"] == 100.0 and row["erng_recall"] == 100.0
        score = row["hnsw_ms"] + row["erng_ms"]
        print(f"[AutoBBCal] {ds} c={c} valid={valid} score={score:.6f} row={row}", flush=True)
        if valid and (best is None or score < best["score"]):
            best = {"score": score, "row": row}
    if best is None:
        # Fall back to best exact-decision candidate, then recall.
        best_row = sorted(tried, key=lambda r: (-(r["hnsw_min_match"]+r["erng_min_match"]), -(r["hnsw_recall"]+r["erng_recall"]), r["hnsw_ms"]+r["erng_ms"]))[0]
    else:
        best_row = best["row"]
    best_c = int(best_row["centroids"])
    centroids, labels, cpath = build_centroids(ds, bank, best_c, out_dir)
    payload = {"dataset": ds, "known_count": len(known), "unknown_count": len(unknown), "selected_centroids": best_c, "calibration": tried, "queries": {}}
    print(f"[AutoBB] selected ds={ds} centroids={best_c}", flush=True)
    for q_raw in q_sizes:
        q = min(int(q_raw), len(known), len(unknown))
        row = eval_rows(ds, model, known, unknown, bank, centroids, labels, get_thr, q, G.ATTACKS)
        payload["queries"][str(q)] = row
        print(f"[AutoBBSummary] {ds} q={q} {row}", flush=True)
    out = out_dir / f"{ds}_centroid_branchbound_autotune_timing.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[AutoBB] saved {out}", flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(G.DATASETS), required=True)
    args=ap.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
