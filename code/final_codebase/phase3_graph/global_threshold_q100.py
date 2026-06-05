#!/usr/bin/env python3
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/saketh/Codebase')
import phase3_all_datasets_graph_benchmark as G
import phase3_cpp_centroid_bb_benchmark as B

ROOT=Path('/home/saketh/json/final_named_artifacts_20260601')
OUT=ROOT/'phase3_global_threshold_q100'
OUT.mkdir(parents=True, exist_ok=True)
ATTACKS=G.ATTACKS
Q=100

def pick_global_threshold(kd2_list, ud2_list):
    kd2=np.concatenate(kd2_list)
    ud2=np.concatenate(ud2_list)
    d=np.sqrt(np.concatenate([kd2,ud2]))
    # Include small epsilon around sorted observed distances.
    cand=np.unique(np.round(d, 6))
    if len(cand)>4000:
        # Dense enough quantiles for speed; q100 has 5200 total distances.
        cand=np.unique(np.quantile(cand, np.linspace(0,1,4000)))
    best=None
    rows=[]
    for t in cand:
        t2=float(t*t)
        known=float((kd2<=t2).mean()*100.0)
        unknown=float((ud2>t2).mean()*100.0)
        total=(known+unknown)/2.0
        rec=(total, known, unknown, float(t))
        rows.append(rec)
        if best is None or total>best[0] or (abs(total-best[0])<1e-9 and min(known,unknown)>min(best[1],best[2])):
            best=rec
    best_total,bk,bu,bt=best
    # Buffered operating point: largest threshold on the near-best plateau.
    plateau=[r for r in rows if r[3]>=bt and r[0]>=best_total-0.10]
    buf=max(plateau, key=lambda r:r[3]) if plateau else best
    return {
        'best_threshold': round(bt,6), 'best_total': round(best_total,2), 'best_known': round(bk,2), 'best_unknown': round(bu,2),
        'buffer_threshold': round(buf[3],6), 'buffer_total': round(buf[0],2), 'buffer_known': round(buf[1],2), 'buffer_unknown': round(buf[2],2),
        'plateau_tolerance': 0.10
    }

def acc_from_dup(dup, q):
    known=float(dup[:q].mean()*100.0)
    unknown=float((~dup[q:]).mean()*100.0)
    return known, unknown, (known+unknown)/2.0

def run(ds):
    cfg=G.DATASETS[ds]
    timing=ROOT/ds/'phase3_cpp_centroid_branchbound'/f'{ds}_cpp_centroid_branchbound_timing.json'
    selected=json.load(open(timing))['selected_centroid_count']
    known,unknown,bank=G.load_rows(cfg['store'])
    bank=np.ascontiguousarray(bank.astype(np.float32))
    model=G.load_model(cfg['ckpt'])
    lib=B.load_lib()
    out_dir=ROOT/ds/'phase3_cpp_centroid_branchbound'
    centroids,labels,cpath=B.build_centroids(ds,bank,out_dir,int(selected))
    centroids=np.ascontiguousarray(centroids.astype(np.float32))
    cidx,offsets,radii=B.cluster_flat(bank,centroids,labels)
    hgraph,erng,probes=B.make_graphs(centroids)
    qk,qu=B.C.pick_rows(known,unknown,Q)
    direct_kd2=[]; direct_ud2=[]; cached=[]
    print(f'[{ds}] q100 global-threshold distance pass attacks={len(ATTACKS)} centroids={selected}', flush=True)
    for ai,(label,attack_name) in enumerate(ATTACKS,1):
        print(f'[{ds}] direct distance {ai:02d}/{len(ATTACKS)} {label}', flush=True)
        kemb=G.embed_eval_rows(ds,model,qk,label,attack_name,batch_size=64)
        uemb=G.embed_eval_rows(ds,model,qu,label,attack_name,batch_size=64)
        allq=np.ascontiguousarray(np.concatenate([kemb,uemb],axis=0).astype(np.float32))
        dd2,di,dt=B.call_direct(lib,allq,bank)
        direct_kd2.append(dd2[:Q]); direct_ud2.append(dd2[Q:])
        cached.append((label,attack_name,allq,dd2,dt))
    thr_info=pick_global_threshold(direct_kd2,direct_ud2)
    thr=float(thr_info['buffer_threshold'])
    print(f'[{ds}] global threshold selected {thr_info}', flush=True)
    rows=[]
    agg={k:[] for k in ['direct_total','direct_known','direct_unknown','direct_ms','h_total','h_known','h_unknown','h_ms','h_match','h_seen','e_total','e_known','e_unknown','e_ms','e_match','e_seen']}
    for ai,(label,attack_name,allq,dd2,dt) in enumerate(cached,1):
        print(f'[{ds}] graph eval {ai:02d}/{len(cached)} {label}', flush=True)
        direct_dup=dd2 <= thr*thr
        dk,du,dtot=acc_from_dup(direct_dup,Q)
        hseeds=B.hnsw_seeds(allq,centroids,hgraph)
        eseeds=B.erng_seeds(allq,centroids,erng,probes)
        hd2,hi,hseen,hdup,ht=B.call_threshold_decision_seeded(lib,allq,bank,centroids,radii,cidx,offsets,hseeds,thr,init_k=2)
        ed2,ei,eseen,edup,et=B.call_threshold_decision_seeded(lib,allq,bank,centroids,radii,cidx,offsets,eseeds,thr,init_k=2)
        hdup=hdup.astype(bool); edup=edup.astype(bool)
        hk,hu,htot=acc_from_dup(hdup,Q)
        ek,eu,etot=acc_from_dup(edup,Q)
        row={
            'attack':label,'threshold':round(thr,6),
            'direct_known':round(dk,2),'direct_unknown':round(du,2),'direct_total':round(dtot,2),'direct_ms':round(dt*1000/(2*Q),6),
            'hnsw_known':round(hk,2),'hnsw_unknown':round(hu,2),'hnsw_total':round(htot,2),'hnsw_match':round(float((hdup==direct_dup).mean()*100.0),2),'hnsw_ms':round(ht*1000/(2*Q),6),'hnsw_seen':round(float(hseen.mean()),2),
            'erng_known':round(ek,2),'erng_unknown':round(eu,2),'erng_total':round(etot,2),'erng_match':round(float((edup==direct_dup).mean()*100.0),2),'erng_ms':round(et*1000/(2*Q),6),'erng_seen':round(float(eseen.mean()),2),
        }
        rows.append(row)
        for k in agg:
            key={'h_total':'hnsw_total','h_known':'hnsw_known','h_unknown':'hnsw_unknown','h_ms':'hnsw_ms','h_match':'hnsw_match','h_seen':'hnsw_seen','e_total':'erng_total','e_known':'erng_known','e_unknown':'erng_unknown','e_ms':'erng_ms','e_match':'erng_match','e_seen':'erng_seen'}.get(k,k)
            agg[k].append(row[key])
    summary={k:round(float(np.mean(v)),6) for k,v in agg.items()}
    payload={'dataset':ds,'q':Q,'mode':'single_global_threshold_unknown_attack_type','known_count':len(known),'unknown_count':len(unknown),'selected_centroids':int(selected),'centroid_file':str(cpath),'probes':[f'C{int(p):04d}' for p in probes],'threshold_selection':thr_info,'summary':summary,'rows':rows}
    out=OUT/f'{ds}_q100_global_threshold_direct_hnsw_erng.json'
    out.write_text(json.dumps(payload,indent=2))
    print('[saved]',out,flush=True)
    print('[summary]',ds,summary,flush=True)

if __name__=='__main__':
    for ds in sys.argv[1:]: run(ds)
