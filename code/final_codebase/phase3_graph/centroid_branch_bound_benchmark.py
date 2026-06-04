#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/saketh/Codebase")
import phase3_all_datasets_graph_benchmark as G  # noqa: E402
import phase3_centroid_hnsw_erng as C  # noqa: E402

ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
LIB = Path("/home/saketh/Codebase/libcentroid_bb_kernel.so")
CPP = Path("/home/saketh/Codebase/centroid_bb_kernel.cpp")
CENTROID_GRID = {
    "ucid": [64, 96, 128, 192, 256, 384, 512, 768],
    "flickr": [96, 128, 192, 256, 384, 512, 768, 1024],
    "cifar": [128, 192, 256, 384, 512, 768, 1024, 1536],
    "amazon": [192, 256, 384, 512, 768, 1024, 1536, 2048],
}
CAL_ATTACKS = None


def ensure_lib():
    if LIB.exists() and LIB.stat().st_mtime >= CPP.stat().st_mtime:
        return
    cmd = [
        "g++", "-O3", "-march=native", "-shared", "-std=c++17", "-fPIC",
        str(CPP), "-o", str(LIB)
    ]
    subprocess.check_call(cmd)


def load_lib():
    ensure_lib()
    lib = ctypes.cdll.LoadLibrary(str(LIB))
    fptr = ctypes.POINTER(ctypes.c_float)
    iptr = ctypes.POINTER(ctypes.c_int)
    lib.direct_l2_search.argtypes = [fptr, ctypes.c_int, fptr, ctypes.c_int, ctypes.c_int, fptr, iptr]
    lib.centroid_branch_bound_search.argtypes = [
        fptr, ctypes.c_int, fptr, ctypes.c_int, ctypes.c_int,
        fptr, ctypes.c_int, fptr, iptr, iptr, ctypes.c_int, fptr, iptr, iptr
    ]
    lib.centroid_threshold_decision_search.argtypes = [
        fptr, ctypes.c_int, fptr, ctypes.c_int, ctypes.c_int,
        fptr, ctypes.c_int, fptr, iptr, iptr, ctypes.c_float,
        ctypes.c_int, fptr, iptr, iptr, iptr
    ]
    lib.centroid_threshold_decision_search_seeded.argtypes = [
        fptr, ctypes.c_int, fptr, ctypes.c_int, ctypes.c_int,
        fptr, ctypes.c_int, fptr, iptr, iptr, iptr,
        ctypes.c_float, ctypes.c_int, fptr, iptr, iptr, iptr
    ]
    return lib


def build_centroids(ds, bank, out_dir, c):
    c = min(int(c), bank.shape[0])
    path = out_dir / f"{ds}_cpp_centroids_c{c}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["centroids"].astype(np.float32), z["labels"].astype(np.int32), str(path)
    print(f"[CppBB] build centroids {ds} c={c}", flush=True)
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=c, batch_size=4096, n_init=1, max_iter=60,
            random_state=20260603 + c, reassignment_ratio=0.005
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


def cluster_flat(bank, centroids, labels):
    order = np.argsort(labels, kind="stable").astype(np.int32)
    counts = np.bincount(labels, minlength=centroids.shape[0]).astype(np.int32)
    offsets = np.zeros(centroids.shape[0] + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    radii = np.zeros(centroids.shape[0], dtype=np.float32)
    for c in range(centroids.shape[0]):
        idx = order[offsets[c]:offsets[c+1]]
        if len(idx):
            diff = bank[idx] - centroids[c][None, :]
            d2 = np.einsum("ij,ij->i", diff, diff)
            radii[c] = np.sqrt(d2.max()).astype(np.float32)
    return np.ascontiguousarray(order), np.ascontiguousarray(offsets), np.ascontiguousarray(radii)


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


def make_graphs(centroids):
    hgraph = C.build_custom_hnsw(centroids, local_m=12)
    erng = C.build_erng(centroids, local_m=12)
    rng = np.random.default_rng(20260603)
    central = np.argsort(hgraph["centrality"])[:4].astype(np.int32).tolist()
    random_probes = rng.choice(np.arange(centroids.shape[0]), size=min(4, centroids.shape[0]), replace=False).astype(np.int32).tolist()
    probes = sorted(set(central + random_probes))
    return hgraph, erng, probes


def hnsw_seeds(q, centroids, hgraph):
    seeds = np.empty(q.shape[0], dtype=np.int32)
    top = hgraph["layers"][2]
    for i, v in enumerate(q):
        top_d = np.linalg.norm(centroids[top] - v[None, :], axis=1)
        cur = int(top[int(np.argmin(top_d))])
        for layer in [2, 1, 0]:
            cur = greedy_seed(v, centroids, hgraph["graphs"][layer], cur)
        seeds[i] = cur
    return seeds


def erng_seeds(q, centroids, erng, probes):
    seeds = np.empty(q.shape[0], dtype=np.int32)
    for i, v in enumerate(q):
        best = None
        for start in probes:
            c = greedy_seed(v, centroids, erng, int(start))
            d = float(np.linalg.norm(v - centroids[c]))
            if best is None or d < best[1]:
                best = (c, d)
        seeds[i] = int(best[0])
    return seeds


def call_direct(lib, q, bank):
    q = np.ascontiguousarray(q.astype(np.float32))
    bank = np.ascontiguousarray(bank.astype(np.float32))
    out_d2 = np.empty(q.shape[0], dtype=np.float32)
    out_i = np.empty(q.shape[0], dtype=np.int32)
    t0 = time.perf_counter()
    lib.direct_l2_search(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), q.shape[0],
        bank.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), bank.shape[0], bank.shape[1],
        out_d2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return out_d2, out_i, time.perf_counter() - t0


def call_bb(lib, q, bank, centroids, radii, cidx, offsets, init_k):
    q = np.ascontiguousarray(q.astype(np.float32))
    out_d2 = np.empty(q.shape[0], dtype=np.float32)
    out_i = np.empty(q.shape[0], dtype=np.int32)
    out_seen = np.empty(q.shape[0], dtype=np.int32)
    t0 = time.perf_counter()
    lib.centroid_branch_bound_search(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), q.shape[0],
        bank.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), bank.shape[0], bank.shape[1],
        centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), centroids.shape[0],
        radii.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        init_k,
        out_d2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_seen.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return out_d2, out_i, out_seen, time.perf_counter() - t0


def call_threshold_decision(lib, q, bank, centroids, radii, cidx, offsets, threshold, init_k):
    q = np.ascontiguousarray(q.astype(np.float32))
    out_d2 = np.empty(q.shape[0], dtype=np.float32)
    out_i = np.empty(q.shape[0], dtype=np.int32)
    out_seen = np.empty(q.shape[0], dtype=np.int32)
    out_dup = np.empty(q.shape[0], dtype=np.int32)
    t0 = time.perf_counter()
    lib.centroid_threshold_decision_search(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), q.shape[0],
        bank.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), bank.shape[0], bank.shape[1],
        centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), centroids.shape[0],
        radii.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_float(float(threshold) * float(threshold)),
        init_k,
        out_d2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_seen.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_dup.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return out_d2, out_i, out_seen, out_dup, time.perf_counter() - t0


def call_threshold_decision_seeded(lib, q, bank, centroids, radii, cidx, offsets, seeds, threshold, init_k):
    q = np.ascontiguousarray(q.astype(np.float32))
    seeds = np.ascontiguousarray(seeds.astype(np.int32))
    out_d2 = np.empty(q.shape[0], dtype=np.float32)
    out_i = np.empty(q.shape[0], dtype=np.int32)
    out_seen = np.empty(q.shape[0], dtype=np.int32)
    out_dup = np.empty(q.shape[0], dtype=np.int32)
    t0 = time.perf_counter()
    lib.centroid_threshold_decision_search_seeded(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), q.shape[0],
        bank.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), bank.shape[0], bank.shape[1],
        centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), centroids.shape[0],
        radii.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_float(float(threshold) * float(threshold)),
        init_k,
        out_d2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_seen.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_dup.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return out_d2, out_i, out_seen, out_dup, time.perf_counter() - t0


def total_acc(kd2, ud2, thr):
    t2 = float(thr) * float(thr)
    known = float((kd2 <= t2).mean() * 100.0)
    unknown = float((ud2 > t2).mean() * 100.0)
    return round((known + unknown) / 2.0, 2)


def eval_with_centroids(ds, lib, model, known, unknown, bank, get_thr, centroids, labels, q, attacks):
    centroids = np.ascontiguousarray(centroids.astype(np.float32))
    cidx, offsets, radii = cluster_flat(bank, centroids, labels)
    hgraph, erng, probes = make_graphs(centroids)
    qk, qu = C.pick_rows(known, unknown, q)
    agg = {k: [] for k in [
        "direct_total","direct_ms",
        "h_total","h_ms","h_match","h_recall","h_seen",
        "e_total","e_ms","e_match","e_recall","e_seen",
    ]}
    print(f"[CppBB] {ds} q={q} c={centroids.shape[0]} attacks={len(attacks)}", flush=True)
    for ai, (label, attack_name) in enumerate(attacks, 1):
        print(f"[CppBB] {ds} q={q} c={centroids.shape[0]} attack {ai:02d}/{len(attacks)} {label}", flush=True)
        kemb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
        uemb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
        allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0).astype(np.float32))
        thr = get_thr(q, label)
        dd2, di, dt = call_direct(lib, allq, bank)
        hseeds = hnsw_seeds(allq, centroids, hgraph)
        eseeds = erng_seeds(allq, centroids, erng, probes)
        hd2, hi, hseen, hdup, ht = call_threshold_decision_seeded(lib, allq, bank, centroids, radii, cidx, offsets, hseeds, thr, init_k=2)
        ed2, ei, eseen, edup, et = call_threshold_decision_seeded(lib, allq, bank, centroids, radii, cidx, offsets, eseeds, thr, init_k=2)
        n = float(2*q)
        direct_dup = dd2 <= (thr * thr)
        agg["direct_total"].append(total_acc(dd2[:q], dd2[q:], thr))
        agg["direct_ms"].append(dt*1000.0/n)

        hgraph_dup = hdup.astype(bool)
        hdec = hgraph_dup == direct_dup
        hknown = float(hgraph_dup[:q].mean() * 100.0)
        hunknown = float((~hgraph_dup[q:]).mean() * 100.0)
        agg["h_total"].append(round((hknown + hunknown) / 2.0, 2))
        agg["h_ms"].append(ht*1000.0/n)
        agg["h_match"].append(float(np.mean(hdec)*100.0))
        agg["h_recall"].append(float(np.mean(hi == di)*100.0))
        agg["h_seen"].append(float(np.mean(hseen)))

        egraph_dup = edup.astype(bool)
        edec = egraph_dup == direct_dup
        eknown = float(egraph_dup[:q].mean() * 100.0)
        eunknown = float((~egraph_dup[q:]).mean() * 100.0)
        agg["e_total"].append(round((eknown + eunknown) / 2.0, 2))
        agg["e_ms"].append(et*1000.0/n)
        agg["e_match"].append(float(np.mean(edec)*100.0))
        agg["e_recall"].append(float(np.mean(ei == di)*100.0))
        agg["e_seen"].append(float(np.mean(eseen)))
    return {
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


def run_dataset(ds):
    lib = load_lib()
    cfg = G.DATASETS[ds]
    out_dir = ROOT / ds / "phase3_cpp_centroid_branchbound"
    out_dir.mkdir(parents=True, exist_ok=True)
    known, unknown, bank = G.load_rows(cfg["store"])
    bank = np.ascontiguousarray(bank.astype(np.float32))
    q_sizes, get_thr = G.q_sizes_and_thresholds(cfg["eval"], cfg.get("policy"))
    model = G.load_model(cfg["ckpt"])
    cal_attacks = G.ATTACKS if CAL_ATTACKS is None else [x for x in G.ATTACKS if x[0] in CAL_ATTACKS]
    q_cal = min(100, len(known), len(unknown))
    calibration = []
    best = None
    for c in CENTROID_GRID[ds]:
        centroids, labels, cpath = build_centroids(ds, bank, out_dir, c)
        row = eval_with_centroids(ds, lib, model, known, unknown, bank, get_thr, centroids, labels, q_cal, cal_attacks)
        row["centroids"] = int(centroids.shape[0])
        row["centroid_file"] = cpath
        # This phase is a duplicate/not-duplicate accelerator. A centroid point is
        # valid when both graph routes reproduce the direct threshold decision.
        # Exact NN recall is reported separately because threshold proof may stop
        # before visiting the true nearest item while preserving the same decision.
        valid = row["hnsw_min_match"] == 100.0 and row["erng_min_match"] == 100.0
        row["valid"] = valid
        row["selection_score_ms"] = round(float(max(row["hnsw_ms"], row["erng_ms"])), 6)
        calibration.append(row)
        print(f"[CppBBCal] {ds} c={c} valid={valid} row={row}", flush=True)
        if valid and (best is None or row["selection_score_ms"] < best["selection_score_ms"]):
            best = row
    if best is None:
        best = sorted(
            calibration,
            key=lambda r: (
                -min(r["hnsw_min_match"], r["erng_min_match"]),
                r["selection_score_ms"],
                -min(r["hnsw_recall"], r["erng_recall"]),
            ),
        )[0]
    selected_c = int(best["centroids"])
    centroids, labels, cpath = build_centroids(ds, bank, out_dir, selected_c)
    payload = {
        "dataset": ds,
        "known_count": len(known),
        "unknown_count": len(unknown),
        "selected_centroid_count": selected_c,
        "centroid_file": cpath,
        "method": "C++ exact centroid branch-bound selected by C++ centroid-count calibration",
        "calibration": calibration,
        "queries": {},
    }
    print(f"[CppBB] selected ds={ds} c={selected_c}", flush=True)
    for q_raw in q_sizes:
        q = min(int(q_raw), len(known), len(unknown))
        row = eval_with_centroids(ds, lib, model, known, unknown, bank, get_thr, centroids, labels, q, G.ATTACKS)
        payload["queries"][str(q)] = row
        print(f"[CppBBSummary] {ds} q={q} {row}", flush=True)
    out = out_dir / f"{ds}_cpp_centroid_branchbound_timing.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[CppBB] saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(G.DATASETS), required=True)
    args = ap.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
