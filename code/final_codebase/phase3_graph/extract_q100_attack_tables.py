#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/saketh/Codebase')
import phase3_cpp_centroid_bb_benchmark as B
import phase3_all_datasets_graph_benchmark as G

ROOT=Path('/home/saketh/json/final_named_artifacts_20260601')
SELECT={'cifar':512,'flickr':192,'ucid':128,'amazon':512}

def per_attack(ds, q=100):
    lib=B.load_lib(); cfg=G.DATASETS[ds]
    known, unknown, bank = G.load_rows(cfg['store'])
    bank=np.ascontiguousarray(bank.astype(np.float32))
    q_sizes, get_thr = G.q_sizes_and_thresholds(cfg['eval'])
    model=G.load_model(cfg['ckpt'])
    out_dir=ROOT/ds/'phase3_cpp_centroid_branchbound'
    c=SELECT[ds]
    centroids, labels, cpath = B.build_centroids(ds, bank, out_dir, c)
    centroids=np.ascontiguousarray(centroids.astype(np.float32))
    cidx, offsets, radii = B.cluster_flat(bank, centroids, labels)
    hgraph, erng, probes = B.make_graphs(centroids)
    q=min(q, len(known), len(unknown))
    qk, qu = B.C.pick_rows(known, unknown, q)
    rows=[]
    print(f'[PerAttackCompare] {ds} q={q} c={c} attacks={len(G.ATTACKS)}', flush=True)
    for ai,(label,attack_name) in enumerate(G.ATTACKS,1):
        print(f'[PerAttackCompare] {ds} attack {ai:02d}/26 {label}', flush=True)
        kemb=G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
        uemb=G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
        allq=np.ascontiguousarray(np.concatenate([kemb,uemb],axis=0).astype(np.float32))
        thr=get_thr(q,label)
        dd2,di,dt=B.call_direct(lib, allq, bank)
        hseeds=B.hnsw_seeds(allq, centroids, hgraph)
        eseeds=B.erng_seeds(allq, centroids, erng, probes)
        hd2,hi,hseen,hdup,ht=B.call_threshold_decision_seeded(lib, allq, bank, centroids, radii, cidx, offsets, hseeds, thr, init_k=2)
        ed2,ei,eseen,edup,et=B.call_threshold_decision_seeded(lib, allq, bank, centroids, radii, cidx, offsets, eseeds, thr, init_k=2)
        n=float(2*q); direct_dup=dd2 <= thr*thr
        def side_acc(d2):
            known_acc=float((d2[:q] <= thr*thr).mean()*100.0)
            unknown_acc=float((d2[q:] > thr*thr).mean()*100.0)
            return known_acc, unknown_acc, (known_acc+unknown_acc)/2
        dk,du,dtot=side_acc(dd2)
        hbool=hdup.astype(bool); ebool=edup.astype(bool)
        hk=float(hbool[:q].mean()*100.0); hu=float((~hbool[q:]).mean()*100.0); htot=(hk+hu)/2
        ek=float(ebool[:q].mean()*100.0); eu=float((~ebool[q:]).mean()*100.0); etot=(ek+eu)/2
        rows.append({
            'attack':label,'threshold':round(float(thr),6),
            'direct_total':round(dtot,2),'hnsw_total':round(htot,2),'erng_total':round(etot,2),
            'hnsw_match':round(float((hbool==direct_dup).mean()*100.0),2),
            'erng_match':round(float((ebool==direct_dup).mean()*100.0),2),
            'direct_ms':round(dt*1000.0/n,6),'hnsw_ms':round(ht*1000.0/n,6),'erng_ms':round(et*1000.0/n,6),
            'hnsw_seen':round(float(np.mean(hseen)),1),'erng_seen':round(float(np.mean(eseen)),1)
        })
    out=out_dir/f'{ds}_q{q}_per_attack_direct_hnsw_erng.json'
    out.write_text(json.dumps({'dataset':ds,'q':q,'centroids':c,'rows':rows},indent=2))
    print('[PerAttackCompare] saved', out, flush=True)

if __name__=='__main__':
    for ds in sys.argv[1:]: per_attack(ds,100)
