#!/usr/bin/env python3
"""Robust Phase-3 example timing report.

This script intentionally reports repeated median/p95 timings for individual
examples, not one-shot timings. It reuses the already-built HNSW and ERNG graph
files and the final named stores/checkpoints.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

import phase3_all_datasets_graph_benchmark as G

ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
ATTACK_PICK = [
    ("original", None),
    ("rotate_m5", "rotate_m5"),
    ("rotate40", "rotate40"),
    ("crop40", "crop40"),
    ("bright", "bright"),
    ("heavy_bright", "heavy_bright"),
    ("flip_h", "flip_h"),
    ("watermark_asset", "watermark_asset"),
    ("blur", "blur"),
    ("contrast", "contrast"),
]


def median_p95(values):
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0, 0.0, 0.0
    p95_idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
    return statistics.median(vals), vals[p95_idx], max(vals)


def timed_one(lib, hnsw, bank, erng, entry, erng_ef, query, repeats=17, warmup=5):
    q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
    for _ in range(warmup):
        G.direct_cpp(lib, q, bank)
        G.hnsw_query(hnsw, q)
        G.erng_cpp(lib, q, bank, erng, entry, erng_ef)
    direct, hnsw_t, erng_t = [], [], []
    di = hi = ri = None
    dd = hd = rd = None
    seen = None
    for _ in range(repeats):
        dd, di, dt = G.direct_cpp(lib, q, bank)
        hd, hi, ht = G.hnsw_query(hnsw, q)
        rd, ri, seen, rt = G.erng_cpp(lib, q, bank, erng, entry, erng_ef)
        direct.append(dt * 1000.0)
        hnsw_t.append(ht * 1000.0)
        erng_t.append(rt * 1000.0)
    dm, dp95, dmax = median_p95(direct)
    hm, hp95, hmax = median_p95(hnsw_t)
    em, ep95, emax = median_p95(erng_t)
    return {
        "direct_median_ms": round(dm, 6),
        "direct_p95_ms": round(dp95, 6),
        "direct_max_ms": round(dmax, 6),
        "hnsw_median_ms": round(hm, 6),
        "hnsw_p95_ms": round(hp95, 6),
        "hnsw_max_ms": round(hmax, 6),
        "erng_median_ms": round(em, 6),
        "erng_p95_ms": round(ep95, 6),
        "erng_max_ms": round(emax, 6),
        "direct_nn": int(di[0]),
        "hnsw_nn": int(hi[0]),
        "erng_nn": int(ri[0]),
        "same_hnsw": bool(hi[0] == di[0]),
        "same_erng": bool(ri[0] == di[0]),
        "direct_dist": round(float(dd[0]), 6),
        "hnsw_dist": round(float(hd[0]), 6),
        "erng_dist": round(float(rd[0]), 6),
        "erng_seen": int(seen[0]),
    }


def choose_rows(known, unknown):
    # Five known and five unknown rows, spread across the currently trusted store.
    rows = []
    for side, src in [("known", known), ("unknown", unknown)]:
        n = len(src)
        idxs = [0, max(0, n // 7), max(0, n // 3), max(0, (2 * n) // 3), n - 1]
        for i in idxs:
            rows.append((side, src[i]))
    return rows


def run_dataset(ds):
    cfg = G.DATASETS[ds]
    out_dir = ROOT / ds / "phase3_graph_query"
    allq_path = out_dir / f"{ds}_phase3_direct_hnsw_erng_allq.json"
    allq = json.load(open(allq_path, "r", encoding="utf-8"))
    known, unknown, bank = G.load_rows(cfg["store"])
    model = G.load_model(cfg["ckpt"])
    lib = G.load_lib()
    hnsw, _, _ = G.build_hnsw(ds, bank, out_dir)
    erng, _, _ = G.build_erng(ds, bank, out_dir)
    hnsw.set_ef(allq["calibration"]["hnsw_best"]["ef"])
    erng_ef = allq["calibration"]["erng_best"]["ef"]
    centroid = bank.mean(axis=0, keepdims=True)
    entry = int(np.argmin(np.linalg.norm(bank - centroid, axis=1)))

    rows = choose_rows(known, unknown)
    examples = []
    for attack_label, attack_name in ATTACK_PICK:
        print(f"[{ds}] robust examples attack {attack_label}", flush=True)
        emb = G.embed_queries(model, [r for _, r in rows], attack_name, batch_size=32)
        for (side, row), qv in zip(rows, emb):
            rec = timed_one(lib, hnsw, bank, erng, entry, erng_ef, qv)
            rec.update({
                "dataset": ds,
                "side": side,
                "attack": attack_label,
                "file": os.path.basename(row["path"]),
                "path": row["path"],
            })
            examples.append(rec)

    # Sort by graph speed and pick 10 broad examples plus 3 report cases.
    for e in examples:
        e["graph_combo_median_ms"] = round(e["hnsw_median_ms"] + e["erng_median_ms"], 6)
    sorted_rows = sorted(examples, key=lambda x: x["graph_combo_median_ms"])
    best = sorted_rows[0]
    avg = sorted_rows[len(sorted_rows) // 2]
    worst = sorted_rows[-1]
    selected10 = []
    seen = set()
    for candidate in sorted_rows[:: max(1, len(sorted_rows) // 10)]:
        key = (candidate["side"], candidate["attack"], candidate["path"])
        if key not in seen:
            selected10.append(candidate)
            seen.add(key)
        if len(selected10) >= 10:
            break
    payload = {
        "dataset": ds,
        "store": str(cfg["store"]),
        "checkpoint": str(cfg["ckpt"]),
        "eval_json": str(cfg["eval"]),
        "known_count": len(known),
        "unknown_count": len(unknown),
        "hnsw_ef": allq["calibration"]["hnsw_best"]["ef"],
        "erng_ef": allq["calibration"]["erng_best"]["ef"],
        "timing_note": "Per-image timings are repeated median/p95 over warm runs, not single raw samples.",
        "best_avg_worst": {"best": best, "avg": avg, "worst": worst},
        "examples10": selected10,
    }
    out = out_dir / f"{ds}_phase3_robust_examples_original_and_attacks.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[{ds}] wrote {out}", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(G.DATASETS), required=True)
    args = ap.parse_args()
    run_dataset(args.dataset)
