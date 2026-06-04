#!/usr/bin/env python3
"""Tune ERNG query parameters against direct and HNSW references."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import cifar_phase3_fast_benchmark as B

OUT_DIR = Path("/home/saketh/json/final_named_artifacts_20260601/cifar/phase3_graph_query")
OUT_JSON = OUT_DIR / "cifar_q100_erng_tuning_best.json"


def pick_entries(bank: np.ndarray, counts=(1, 4, 8, 16)):
    centroid = bank.mean(axis=0, keepdims=True)
    d = np.linalg.norm(bank - centroid, axis=1)
    center = int(np.argmin(d))
    far = np.argsort(d)[::-1]
    out = {}
    for c in counts:
        entries = [center]
        for idx in far:
            idx = int(idx)
            if idx not in entries:
                entries.append(idx)
            if len(entries) >= c:
                break
        out[c] = entries
    return out


def erng_multi(lib, query, bank, adj, entries, ef):
    best_d = None
    best_i = None
    best_seen = None
    total_time = 0.0
    total_seen = 0
    for entry in entries:
        d, i, seen, elapsed = B.erng_cpp(lib, query, bank, adj, entry, ef)
        total_time += elapsed
        total_seen += seen
        if best_d is None:
            best_d, best_i, best_seen = d, i, seen
        else:
            mask = d < best_d
            best_d[mask] = d[mask]
            best_i[mask] = i[mask]
            best_seen[mask] = seen[mask]
    return best_d, best_i, total_seen, total_time


def main():
    lib = B.load_lib()
    known, unknown, bank = B.load_data()
    erng = np.ascontiguousarray(np.load(B.ERNG_NPZ)["adj"].astype(np.int32))
    hnsw, _, _ = B.build_or_load_hnsw(bank, ef_search=128)
    thresholds = json.load(open(B.EVAL, "r", encoding="utf-8"))["queries"]["100"]["attacks"]
    import random

    rng = random.Random(20260529)
    qk, qu = known[:], unknown[:]
    rng.shuffle(qk)
    rng.shuffle(qu)
    qk, qu = qk[:100], qu[:100]
    model = B.load_model()
    entry_sets = pick_entries(bank, counts=(1, 2, 4, 8))
    ef_values = [32, 48, 64, 96, 128, 192, 256]

    variants = {(c, ef): {"times": [], "decision": [], "recall": [], "totals": [], "seen": []} for c in entry_sets for ef in ef_values}
    direct_times = []
    hnsw_times = []
    direct_totals = []
    hnsw_totals = []
    examples = None

    for ai, (label, attack_name) in enumerate(B.ATTACKS, 1):
        print(f"[SweepAttack] {ai:02d}/{len(B.ATTACKS)} {label}", flush=True)
        kemb = B.embed_queries(model, qk, attack_name)
        uemb = B.embed_queries(model, qu, attack_name)
        allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
        thr = float(thresholds[label]["threshold"])
        dd, di, dt = B.direct_cpp(lib, allq, bank)
        hd, hi, ht = B.hnsw_query(hnsw, allq)
        direct_times.append(dt * 1000 / allq.shape[0])
        hnsw_times.append(ht * 1000 / allq.shape[0])
        direct_totals.append(B.metrics(dd[:100], dd[100:], thr)[2])
        hnsw_totals.append(B.metrics(hd[:100], hd[100:], thr)[2])

        for (entry_count, ef), rec in variants.items():
            rd, ri, seen, rt = erng_multi(lib, allq, bank, erng, entry_sets[entry_count], ef)
            dec = float(np.mean(np.concatenate([(rd[:100] <= thr) == (dd[:100] <= thr), (rd[100:] > thr) == (dd[100:] > thr)])) * 100.0)
            recall = float(np.mean(ri == di) * 100.0)
            rec["times"].append(rt * 1000 / allq.shape[0])
            rec["decision"].append(dec)
            rec["recall"].append(recall)
            rec["totals"].append(B.metrics(rd[:100], rd[100:], thr)[2])
            rec["seen"].append(float(np.mean(seen)))

        if label == "original":
            examples = {"threshold": round(thr, 6), "rows": []}
            # Fill with best later.
            for side, rows, off in [("known", qk, 0), ("unknown", qu, 100)]:
                for i, row in enumerate(rows[:5]):
                    examples["rows"].append({"side": side, "path": row["path"], "file": os.path.basename(row["path"]), "idx": off + i})

    summary = []
    for (entry_count, ef), rec in variants.items():
        summary.append(
            {
                "entry_count": entry_count,
                "ef": ef,
                "all_avg": round(float(np.mean(rec["totals"])), 2),
                "avg_ms_per_query": round(float(np.mean(rec["times"])), 6),
                "min_decision_match": round(float(np.min(rec["decision"])), 2),
                "avg_nn_recall": round(float(np.mean(rec["recall"])), 2),
                "min_nn_recall": round(float(np.min(rec["recall"])), 2),
                "avg_seen": round(float(np.mean(rec["seen"])), 1),
                "exact_decision": bool(np.min(rec["decision"]) == 100.0),
                "exact_nn": bool(np.min(rec["recall"]) == 100.0),
            }
        )
    exact = [s for s in summary if s["exact_decision"] and s["exact_nn"]]
    best = min(exact, key=lambda x: x["avg_ms_per_query"]) if exact else max(summary, key=lambda x: (x["min_decision_match"], x["avg_nn_recall"], -x["avg_ms_per_query"]))

    # Per-image timings for the best ERNG variant.
    best_entries = entry_sets[best["entry_count"]]
    model = B.load_model()
    kemb = B.embed_queries(model, qk, None)
    uemb = B.embed_queries(model, qu, None)
    allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0))
    dd, di, _ = B.direct_cpp(lib, allq, bank)
    hd, hi, _ = B.hnsw_query(hnsw, allq)
    rd, ri, seen, _ = erng_multi(lib, allq, bank, erng, best_entries, best["ef"])
    for r in examples["rows"]:
        j = r.pop("idx")
        q = np.ascontiguousarray(allq[j : j + 1])
        _, _, dt = B.direct_cpp(lib, q, bank)
        _, _, ht = B.hnsw_query(hnsw, q)
        _, _, _, rt = erng_multi(lib, q, bank, erng, best_entries, best["ef"])
        r.update(
            {
                "direct_ms": round(dt * 1000, 6),
                "hnsw_ms": round(ht * 1000, 6),
                "erng_ms": round(rt * 1000, 6),
                "direct_nn": int(di[j]),
                "hnsw_nn": int(hi[j]),
                "erng_nn": int(ri[j]),
                "direct_dist": round(float(dd[j]), 6),
                "hnsw_dist": round(float(hd[j]), 6),
                "erng_dist": round(float(rd[j]), 6),
                "same_hnsw": bool(hi[j] == di[j]),
                "same_erng": bool(ri[j] == di[j]),
                "erng_seen": int(seen[j]),
            }
        )

    payload = {
        "dataset": "cifar",
        "q": 100,
        "direct_summary": {"all_avg": round(float(np.mean(direct_totals)), 2), "avg_ms_per_query": round(float(np.mean(direct_times)), 6)},
        "hnsw_summary": {"all_avg": round(float(np.mean(hnsw_totals)), 2), "avg_ms_per_query": round(float(np.mean(hnsw_times)), 6), "ef": 128},
        "best_erng": best,
        "all_erng_variants": sorted(summary, key=lambda x: (not (x["exact_decision"] and x["exact_nn"]), x["avg_ms_per_query"])),
        "examples": examples,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[Saved] {OUT_JSON}", flush=True)
    print("[Direct]", payload["direct_summary"], flush=True)
    print("[HNSW]", payload["hnsw_summary"], flush=True)
    print("[BestERNG]", payload["best_erng"], flush=True)


if __name__ == "__main__":
    main()
