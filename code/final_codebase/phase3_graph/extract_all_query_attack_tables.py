#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/saketh/Codebase')
import phase3_cpp_centroid_bb_benchmark as B
import phase3_all_datasets_graph_benchmark as G

ROOT = Path('/home/saketh/json/final_named_artifacts_20260601')
FALLBACK_C = {'cifar': 512, 'flickr': 192, 'ucid': 128, 'amazon': 512}


def selected_centroids(ds):
    p = ROOT / ds / 'phase3_cpp_centroid_branchbound' / f'{ds}_cpp_centroid_branchbound_timing.json'
    if p.exists():
        try:
            return int(json.load(open(p, 'r', encoding='utf-8'))['selected_centroid_count'])
        except Exception:
            pass
    return FALLBACK_C[ds]


def side_acc_from_d2(d2, q, thr):
    t2 = float(thr) * float(thr)
    known = float((d2[:q] <= t2).mean() * 100.0)
    unknown = float((d2[q:] > t2).mean() * 100.0)
    return known, unknown, (known + unknown) / 2.0


def run(ds):
    lib = B.load_lib()
    cfg = G.DATASETS[ds]
    out_dir = ROOT / ds / 'phase3_cpp_centroid_branchbound'
    out_dir.mkdir(parents=True, exist_ok=True)
    known, unknown, bank = G.load_rows(cfg['store'])
    bank = np.ascontiguousarray(bank.astype(np.float32))
    q_sizes, get_thr = G.q_sizes_and_thresholds(cfg['eval'])
    model = G.load_model(cfg['ckpt'])
    c = selected_centroids(ds)
    centroids, labels, cpath = B.build_centroids(ds, bank, out_dir, c)
    centroids = np.ascontiguousarray(centroids.astype(np.float32))
    cidx, offsets, radii = B.cluster_flat(bank, centroids, labels)
    hgraph, erng, probes = B.make_graphs(centroids)
    payload = {
        'dataset': ds,
        'selected_centroids': c,
        'centroid_file': str(cpath),
        'known_count': len(known),
        'unknown_count': len(unknown),
        'queries': {},
    }
    print(f'[PerAttackAllQ] {ds} start c={c} q_sizes={q_sizes}', flush=True)
    for q_raw in q_sizes:
        q = min(int(q_raw), len(known), len(unknown))
        qk, qu = B.C.pick_rows(known, unknown, q)
        rows = []
        print(f'[PerAttackAllQ] {ds} q={q} attacks={len(G.ATTACKS)}', flush=True)
        for ai, (label, attack_name) in enumerate(G.ATTACKS, 1):
            print(f'[PerAttackAllQ] {ds} q={q} attack {ai:02d}/26 {label}', flush=True)
            kemb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
            uemb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
            allq = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0).astype(np.float32))
            thr = get_thr(q, label)
            dd2, di, dt = B.call_direct(lib, allq, bank)
            hseeds = B.hnsw_seeds(allq, centroids, hgraph)
            eseeds = B.erng_seeds(allq, centroids, erng, probes)
            hd2, hi, hseen, hdup, ht = B.call_threshold_decision_seeded(
                lib, allq, bank, centroids, radii, cidx, offsets, hseeds, thr, init_k=2
            )
            ed2, ei, eseen, edup, et = B.call_threshold_decision_seeded(
                lib, allq, bank, centroids, radii, cidx, offsets, eseeds, thr, init_k=2
            )
            n = float(2 * q)
            direct_dup = dd2 <= thr * thr
            dk, du, dtot = side_acc_from_d2(dd2, q, thr)
            hbool = hdup.astype(bool)
            ebool = edup.astype(bool)
            hk = float(hbool[:q].mean() * 100.0)
            hu = float((~hbool[q:]).mean() * 100.0)
            ek = float(ebool[:q].mean() * 100.0)
            eu = float((~ebool[q:]).mean() * 100.0)
            rows.append({
                'attack': label,
                'threshold': round(float(thr), 6),
                'direct_known': round(dk, 2),
                'direct_unknown': round(du, 2),
                'direct_total': round(dtot, 2),
                'hnsw_known': round(hk, 2),
                'hnsw_unknown': round(hu, 2),
                'hnsw_total': round((hk + hu) / 2.0, 2),
                'hnsw_match': round(float((hbool == direct_dup).mean() * 100.0), 2),
                'erng_known': round(ek, 2),
                'erng_unknown': round(eu, 2),
                'erng_total': round((ek + eu) / 2.0, 2),
                'erng_match': round(float((ebool == direct_dup).mean() * 100.0), 2),
                'direct_ms': round(dt * 1000.0 / n, 6),
                'hnsw_ms': round(ht * 1000.0 / n, 6),
                'erng_ms': round(et * 1000.0 / n, 6),
                'hnsw_speedup': round(dt / max(ht, 1e-12), 2),
                'erng_speedup': round(dt / max(et, 1e-12), 2),
                'hnsw_seen': round(float(np.mean(hseen)), 1),
                'erng_seen': round(float(np.mean(eseen)), 1),
            })
        payload['queries'][str(q)] = {'rows': rows}
        out = out_dir / f'{ds}_allq_per_attack_direct_hnsw_erng.json'
        out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'[PerAttackAllQ] {ds} q={q} saved_partial {out}', flush=True)
    out = out_dir / f'{ds}_allq_per_attack_direct_hnsw_erng.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'[PerAttackAllQ] {ds} done saved {out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(G.DATASETS))
    args = ap.parse_args()
    run(args.dataset)
