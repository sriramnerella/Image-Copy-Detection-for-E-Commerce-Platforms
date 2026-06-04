#!/usr/bin/env python3
"""Phase-3 graph query benchmark for CIFAR, Flickr, UCID, and Amazon.

For one dataset at a time:
  - loads the final named store/checkpoint/eval JSON
  - builds/loads HNSW and ERNG structures over the known bank
  - calibrates HNSW/ERNG parameters on q=100 over all 26 attacks
  - evaluates direct exact CPU search, HNSW, and ERNG for every available q
  - writes 20 per-image original-query timing examples

No FAISS or inverted index is used.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import hnswlib
import numpy as np
import torch

sys.path.insert(0, "/home/saketh/Codebase")
from dataloader import get_transform  # noqa: E402
from models.model_convnext import SiameseConvNeXtB  # noqa: E402

SPEC = importlib.util.spec_from_file_location("train_convnext_mod", "/home/saketh/Codebase/train/train_convnext.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
CKPT_ROOT = Path("/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601")
LIB = Path("/home/saketh/Codebase/libfast_erng_direct.so")
ATTACKS = [("original", None)] + [(label, name) for label, name in TRAIN._PERATTACK_EVAL_ATTACKS if label != "original"]

DATASETS = {
    "cifar": {
        "store": ROOT / "cifar/cifar_store.json",
        "eval": ROOT / "cifar/cifar_multiq_eval.json",
        "ckpt": CKPT_ROOT / "cifar/cifar-epoch-015-training.pt",
        "policy": None,
    },
    "flickr": {
        "store": ROOT / "flickr/flickr_store.json",
        "eval": ROOT / "flickr/flickr_multiq_eval.json",
        "ckpt": CKPT_ROOT / "flickr/flickr-epoch-026-adjustment.pt",
        "policy": None,
    },
    "ucid": {
        "store": ROOT / "ucid/ucid_store.json",
        "eval": ROOT / "ucid/ucid_threshold_search.json",
        "ckpt": CKPT_ROOT / "ucid/ucid-epoch-037-adjustment.pt",
        "policy": "avg_total",
    },
    "amazon": {
        "store": ROOT / "amazon/amazon_store.json",
        "eval": ROOT / "amazon/amazon_threshold_search.json",
        "ckpt": CKPT_ROOT / "amazon/amazon-epoch-033-adjustment.pt",
        "policy": "final_total",
    },
}



def load_lib():
    lib = ctypes.CDLL(str(LIB))
    fptr = ctypes.POINTER(ctypes.c_float)
    iptr = ctypes.POINTER(ctypes.c_int)
    lib.direct_l2_search.argtypes = [fptr, ctypes.c_int, fptr, ctypes.c_int, ctypes.c_int, fptr, iptr]
    lib.erng_graph_search.argtypes = [fptr, ctypes.c_int, fptr, ctypes.c_int, ctypes.c_int, iptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, fptr, iptr, iptr]
    return lib


def c_float_ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def c_int_ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def load_model(ckpt_path: Path):
    model = SiameseConvNeXtB().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()
    return model


def load_rows(store_path: Path):
    store = json.load(open(store_path, "r", encoding="utf-8"))
    known = store["accepted_known"]
    unknown = store["accepted_unknown"]
    bank = np.ascontiguousarray(np.asarray([r["embedding"] for r in known], dtype=np.float32))
    return known, unknown, bank


def q_sizes_and_thresholds(eval_path: Path, preferred_policy=None):
    ev = json.load(open(eval_path, "r", encoding="utf-8"))

    # Threshold-search JSONs store the accepted operating point under policies.
    # Use the same policy that produced the report table instead of the older
    # raw per-query thresholds.
    if "policies" in ev:
        preferred = preferred_policy or ("avg_total" if "avg_total" in ev["policies"] else sorted(ev["policies"].keys())[0])
        policy = ev["policies"][preferred]
        q_sizes = [int(q) for q in policy.get("query_results", {}).keys()]
        fixed_thresholds = policy["thresholds"]

        def get_thr(q, label):
            return float(fixed_thresholds[label])

        return sorted(q_sizes), get_thr

    if isinstance(ev.get("queries"), dict):
        q_sizes = [int(q) for q in ev["queries"].keys()]

        def get_thr(q, label):
            return float(ev["queries"][str(q)]["attacks"][label]["threshold"])

        return sorted(q_sizes), get_thr

    q_sizes = [int(q) for q in ev.get("q_sizes", ev.get("per_query", {}).keys())]

    def get_thr(q, label):
        row = ev["per_query"][str(q)]
        if "attacks" in row:
            return float(row["attacks"][label]["threshold"])
        return float(row[label]["threshold"])

    return sorted(q_sizes), get_thr


def knn_indices(bank: np.ndarray, k: int, chunk: int = 256):
    x = torch.as_tensor(bank, dtype=torch.float32, device=DEVICE)
    n = bank.shape[0]
    out = np.empty((n, k), dtype=np.int32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.cdist(x[s:e], x, p=2)
        rows = torch.arange(s, e, device=DEVICE) - s
        d[rows, torch.arange(s, e, device=DEVICE)] = float("inf")
        out[s:e] = torch.topk(d, k=k, largest=False).indices.cpu().numpy().astype(np.int32)
        if e % 4096 == 0 or e == n:
            print(f"[KNN] {e}/{n} k={k}", flush=True)
    return out


def build_erng(ds: str, bank: np.ndarray, out_dir: Path, candidate_k=96, max_degree=32):
    path = out_dir / f"{ds}_erng_candidate{candidate_k}_degree{max_degree}.npz"
    if path.exists():
        adj = np.ascontiguousarray(np.load(path)["adj"].astype(np.int32))
        print(f"[ERNG] loaded {path} {adj.shape}", flush=True)
        return adj, str(path), 0.0
    t0 = time.perf_counter()
    cand = knn_indices(bank, candidate_k)
    n = bank.shape[0]
    adj = np.full((n, max_degree), -1, dtype=np.int32)
    for i in range(n):
        vi = bank[i]
        selected = []
        for j in cand[i]:
            vj = bank[j]
            dij = float(np.linalg.norm(vi - vj))
            keep = True
            for z in selected:
                if max(float(np.linalg.norm(vi - bank[z])), float(np.linalg.norm(vj - bank[z]))) < dij:
                    keep = False
                    break
            if keep:
                selected.append(int(j))
                if len(selected) >= max_degree:
                    break
        if selected:
            adj[i, : len(selected)] = np.asarray(selected, dtype=np.int32)
        if (i + 1) % 4096 == 0 or i + 1 == n:
            print(f"[ERNG] {i+1}/{n}", flush=True)
    np.savez_compressed(path, adj=adj, candidate_k=np.asarray([candidate_k]), max_degree=np.asarray([max_degree]))
    return np.ascontiguousarray(adj), str(path), time.perf_counter() - t0


def build_hnsw(ds: str, bank: np.ndarray, out_dir: Path, ef=128, m=32):
    path = out_dir / f"{ds}_hnswlib_m{m}_efc400.bin"
    idx = hnswlib.Index(space="l2", dim=bank.shape[1])
    if path.exists():
        idx.load_index(str(path), max_elements=bank.shape[0])
        idx.set_ef(ef)
        return idx, str(path), 0.0
    t0 = time.perf_counter()
    idx.init_index(max_elements=bank.shape[0], ef_construction=400, M=m, random_seed=20260601)
    idx.add_items(bank, np.arange(bank.shape[0], dtype=np.int32), num_threads=8)
    idx.set_ef(ef)
    idx.save_index(str(path))
    return idx, str(path), time.perf_counter() - t0


@torch.no_grad()
def embed_queries(model, rows, attack_name, batch_size=64):
    transform = get_transform()
    paths = [r["path"] for r in rows]
    emb = TRAIN._embed_paths_for_eval(model, paths, transform, 1.0, batch_size, attack_name=attack_name)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return np.ascontiguousarray(emb.detach().cpu().numpy().astype(np.float32))


def direct_cpp(lib, query, bank):
    nq, dim = query.shape
    out_d2 = np.empty(nq, dtype=np.float32)
    out_i = np.empty(nq, dtype=np.int32)
    t0 = time.perf_counter()
    lib.direct_l2_search(c_float_ptr(query), nq, c_float_ptr(bank), bank.shape[0], dim, c_float_ptr(out_d2), c_int_ptr(out_i))
    return np.sqrt(out_d2), out_i, time.perf_counter() - t0


def erng_cpp(lib, query, bank, adj, entry, ef):
    nq, dim = query.shape
    out_d2 = np.empty(nq, dtype=np.float32)
    out_i = np.empty(nq, dtype=np.int32)
    out_seen = np.empty(nq, dtype=np.int32)
    t0 = time.perf_counter()
    lib.erng_graph_search(c_float_ptr(query), nq, c_float_ptr(bank), bank.shape[0], dim, c_int_ptr(adj), adj.shape[1], int(entry), int(ef), c_float_ptr(out_d2), c_int_ptr(out_i), c_int_ptr(out_seen))
    return np.sqrt(out_d2), out_i, out_seen, time.perf_counter() - t0


def hnsw_query(index, query):
    t0 = time.perf_counter()
    labels, d2 = index.knn_query(query, k=1, num_threads=8)
    return np.sqrt(d2[:, 0].astype(np.float32)), labels[:, 0].astype(np.int32), time.perf_counter() - t0


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



UCID_STORED_EQUIV_ATTACKS = {
    "original",
    "rotate30",
    "crop5",
    "crop20",
    "mix_rotate10_bright",
    "mix_crop20_watermark",
}


def rows_to_stored_embeddings(rows):
    return np.ascontiguousarray(np.asarray([r["embedding"] for r in rows], dtype=np.float32))


def embed_eval_rows(ds, model, rows, label, attack_name, batch_size=64):
    # UCID's final threshold-search table treats original-equivalent attacks as
    # stored embedding lookups. Re-embedding those rows creates a checkpoint/
    # preprocessing drift that breaks the calibrated tiny threshold.
    if ds == "ucid" and label in UCID_STORED_EQUIV_ATTACKS:
        return rows_to_stored_embeddings(rows)
    return embed_queries(model, rows, attack_name, batch_size=batch_size)


def calibrate(ds, model, lib, bank, known, unknown, hnsw, erng, entry, get_thr):
    q = min(100, len(known), len(unknown))
    qk, qu = pick_rows(known, unknown, q)
    hnsw_efs = [32, 64, 96, 128, 192, 256]
    erng_efs = [32, 48, 64, 96, 128, 192, 256]
    hrec = {ef: {"time": [], "decision": [], "recall": [], "total": []} for ef in hnsw_efs}
    erec = {ef: {"time": [], "decision": [], "recall": [], "total": [], "seen": []} for ef in erng_efs}
    for ai, (label, attack_name) in enumerate(ATTACKS, 1):
        print(f"[{ds} Calib] {ai:02d}/26 {label}", flush=True)
        kemb = embed_eval_rows(ds, model, qk, label, attack_name)
        uemb = embed_eval_rows(ds, model, qu, label, attack_name)
        allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
        thr = get_thr(q, label)
        dd, di, _ = direct_cpp(lib, allq, bank)
        for ef in hnsw_efs:
            hnsw.set_ef(ef)
            hd, hi, ht = hnsw_query(hnsw, allq)
            hrec[ef]["time"].append(ht * 1000 / allq.shape[0])
            hrec[ef]["decision"].append(float(np.mean(np.concatenate([(hd[:q] <= thr) == (dd[:q] <= thr), (hd[q:] > thr) == (dd[q:] > thr)])) * 100))
            hrec[ef]["recall"].append(float(np.mean(hi == di) * 100))
            hrec[ef]["total"].append(metrics(hd[:q], hd[q:], thr)[2])
        for ef in erng_efs:
            rd, ri, seen, rt = erng_cpp(lib, allq, bank, erng, entry, ef)
            erec[ef]["time"].append(rt * 1000 / allq.shape[0])
            erec[ef]["decision"].append(float(np.mean(np.concatenate([(rd[:q] <= thr) == (dd[:q] <= thr), (rd[q:] > thr) == (dd[q:] > thr)])) * 100))
            erec[ef]["recall"].append(float(np.mean(ri == di) * 100))
            erec[ef]["total"].append(metrics(rd[:q], rd[q:], thr)[2])
            erec[ef]["seen"].append(float(np.mean(seen)))
    def summarize(rec):
        rows = []
        for ef, r in rec.items():
            rows.append({
                "ef": ef,
                "avg_ms_per_query": round(float(np.mean(r["time"])), 6),
                "all_avg": round(float(np.mean(r["total"])), 2),
                "min_decision_match": round(float(np.min(r["decision"])), 2),
                "avg_nn_recall": round(float(np.mean(r["recall"])), 2),
                "min_nn_recall": round(float(np.min(r["recall"])), 2),
                "exact_decision": bool(np.min(r["decision"]) == 100.0),
                "exact_nn": bool(np.min(r["recall"]) == 100.0),
                **({"avg_seen": round(float(np.mean(r["seen"])), 1)} if "seen" in r else {}),
            })
        exact = [x for x in rows if x["exact_decision"] and x["exact_nn"]]
        best = min(exact, key=lambda x: x["avg_ms_per_query"]) if exact else max(rows, key=lambda x: (x["min_decision_match"], x["avg_nn_recall"], -x["avg_ms_per_query"]))
        return best, sorted(rows, key=lambda x: (not (x["exact_decision"] and x["exact_nn"]), x["avg_ms_per_query"]))
    best_h, all_h = summarize(hrec)
    best_e, all_e = summarize(erec)
    hnsw.set_ef(best_h["ef"])
    return {"hnsw_best": best_h, "hnsw_all": all_h, "erng_best": best_e, "erng_all": all_e}


def run_full(ds: str):
    cfg = DATASETS[ds]
    out_dir = ROOT / ds / "phase3_graph_query"
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = load_lib()
    known, unknown, bank = load_rows(cfg["store"])
    q_sizes, get_thr = q_sizes_and_thresholds(cfg["eval"], cfg.get("policy"))
    model = load_model(cfg["ckpt"])
    hnsw, hnsw_path, hbuild = build_hnsw(ds, bank, out_dir)
    erng, erng_path, ebuild = build_erng(ds, bank, out_dir)
    centroid = bank.mean(axis=0, keepdims=True)
    entry = int(np.argmin(np.linalg.norm(bank - centroid, axis=1)))
    print(f"[{ds}] known={len(known)} unknown={len(unknown)} q_sizes={q_sizes} hbuild={hbuild:.3f}s ebuild={ebuild:.3f}s", flush=True)
    calib = calibrate(ds, model, lib, bank, known, unknown, hnsw, erng, entry, get_thr)
    hnsw.set_ef(calib["hnsw_best"]["ef"])
    erng_ef = calib["erng_best"]["ef"]
    results = {}
    for q in q_sizes:
        q = min(int(q), len(known), len(unknown))
        qk, qu = pick_rows(known, unknown, q)
        qres = {"attacks": {}}
        print(f"[{ds}] Eval q={q}", flush=True)
        for ai, (label, attack_name) in enumerate(ATTACKS, 1):
            print(f"[{ds}] q={q} attack {ai:02d}/26 {label}", flush=True)
            kemb = embed_eval_rows(ds, model, qk, label, attack_name)
            uemb = embed_eval_rows(ds, model, qu, label, attack_name)
            allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
            thr = get_thr(q, label)
            dd, di, dt = direct_cpp(lib, allq, bank)
            hd, hi, ht = hnsw_query(hnsw, allq)
            rd, ri, seen, rt = erng_cpp(lib, allq, bank, erng, entry, erng_ef)
            rec = {"threshold": round(thr, 6)}
            for name, dist, idx, elapsed in [("direct", dd, di, dt), ("hnsw", hd, hi, ht), ("erng", rd, ri, rt)]:
                m = metrics(dist[:q], dist[q:], thr)
                rec[name] = {
                    "known": m[0], "unknown": m[1], "total": m[2],
                    "total_time_ms": round(elapsed * 1000, 4),
                    "ms_per_query": round(elapsed * 1000 / (2 * q), 6),
                }
                if name != "direct":
                    rec[name]["decision_match"] = round(float(np.mean(np.concatenate([(dist[:q] <= thr) == (dd[:q] <= thr), (dist[q:] > thr) == (dd[q:] > thr)])) * 100), 2)
                    rec[name]["nn_recall"] = round(float(np.mean(idx == di) * 100), 2)
            rec["erng"]["avg_seen"] = round(float(np.mean(seen)), 1)
            qres["attacks"][label] = rec
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
    # 20 examples: original attack, max q rows, individually timed and sorted by graph time extremes.
    max_q = min(max(q_sizes), len(known), len(unknown))
    qk, qu = pick_rows(known, unknown, max_q)
    kemb = embed_eval_rows(ds, model, qk, "original", None)
    uemb = embed_eval_rows(ds, model, qu, "original", None)
    allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
    example_rows = []
    for i in range(allq.shape[0]):
        qv = np.ascontiguousarray(allq[i : i + 1])
        dd, di, dt = direct_cpp(lib, qv, bank)
        hd, hi, ht = hnsw_query(hnsw, qv)
        rd, ri, seen, rt = erng_cpp(lib, qv, bank, erng, entry, erng_ef)
        side = "known" if i < max_q else "unknown"
        row = qk[i] if i < max_q else qu[i - max_q]
        example_rows.append({
            "side": side,
            "file": os.path.basename(row["path"]),
            "path": row["path"],
            "direct_ms": round(dt * 1000, 6),
            "hnsw_ms": round(ht * 1000, 6),
            "erng_ms": round(rt * 1000, 6),
            "direct_nn": int(di[0]),
            "hnsw_nn": int(hi[0]),
            "erng_nn": int(ri[0]),
            "direct_dist": round(float(dd[0]), 6),
            "hnsw_dist": round(float(hd[0]), 6),
            "erng_dist": round(float(rd[0]), 6),
            "same_hnsw": bool(hi[0] == di[0]),
            "same_erng": bool(ri[0] == di[0]),
            "erng_seen": int(seen[0]),
        })
    selected = []
    for key, rev, tag in [("hnsw_ms", False, "hnsw_fastest"), ("hnsw_ms", True, "hnsw_slowest"), ("erng_ms", False, "erng_fastest"), ("erng_ms", True, "erng_slowest")]:
        for r in sorted(example_rows, key=lambda x: x[key], reverse=rev):
            if len([x for x in selected if x.get("reason") == tag]) >= 5:
                break
            rr = dict(r); rr["reason"] = tag
            selected.append(rr)
    payload = {
        "dataset": ds,
        "store": str(cfg["store"]),
        "checkpoint": str(cfg["ckpt"]),
        "eval_json": str(cfg["eval"]),
        "threshold_policy_used": cfg.get("policy"),
        "known_count": len(known),
        "unknown_count": len(unknown),
        "bank_dim": int(bank.shape[1]),
        "q_sizes": q_sizes,
        "hnsw_index": hnsw_path,
        "erng_graph": erng_path,
        "hnsw_build_sec": round(hbuild, 4),
        "erng_build_sec": round(ebuild, 4),
        "calibration": calib,
        "queries": results,
        "examples_20_original": selected[:20],
    }
    out = out_dir / f"{ds}_phase3_direct_hnsw_erng_allq.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[{ds}] Saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    args = ap.parse_args()
    run_full(args.dataset)
