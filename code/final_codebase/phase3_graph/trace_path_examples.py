#!/usr/bin/env python3
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/saketh/Codebase')
import phase3_cpp_centroid_bb_benchmark as B
import phase3_all_datasets_graph_benchmark as G

ROOT=Path('/home/saketh/json/final_named_artifacts_20260601')
ATTACK_EXAMPLES=[('original',None),('rotate40','rotate40'),('crop40','crop40'),('bright','bright'),('watermark_asset','watermark_asset'),('contrast','contrast')]
TRACE_PICK=[('original',None),('rotate40','rotate40'),('crop40','crop40')]

def row_name(r):
    return Path(r.get('path', r.get('image_path','unknown'))).name

def dist(a,b):
    return float(np.linalg.norm(a-b))

def make_named_centroids(centroids):
    return [f'C{i:04d}' for i in range(len(centroids))]

def greedy_trace(v, centroids, nbrs, start):
    names=make_named_centroids(centroids)
    cur=int(start); curd=dist(v, centroids[cur]); path=[{'centroid':names[cur],'idx':cur,'distance':round(curd,6),'reason':'entry'}]
    seen=set([cur])
    while True:
        candidates=[int(x) for x in nbrs[cur] if int(x)>=0]
        best=cur; bestd=curd
        expanded=[]
        for nb in candidates:
            d=dist(v, centroids[nb])
            expanded.append({'centroid':names[nb],'idx':nb,'distance':round(d,6)})
            if d < bestd:
                best=nb; bestd=d
        expanded=sorted(expanded,key=lambda x:x['distance'])[:10]
        path[-1]['expanded_best10']=expanded
        if best==cur or best in seen:
            path[-1]['reason']='stop_local_minimum'
            break
        cur=best; curd=bestd; seen.add(cur)
        path.append({'centroid':names[cur],'idx':cur,'distance':round(curd,6),'reason':'move_closer'})
        if len(path)>64:
            path[-1]['reason']='stop_path_limit'; break
    return path, cur

def hnsw_trace(v, centroids, hgraph):
    names=make_named_centroids(centroids)
    top=hgraph['layers'][2]
    d=np.linalg.norm(centroids[top]-v.reshape(1,-1),axis=1)
    cur=int(top[int(np.argmin(d))])
    layers=[]
    for layer_id in [2,1,0]:
        nbrs=hgraph['graphs'][layer_id]
        path, cur=greedy_trace(v, centroids, nbrs, cur)
        layers.append({'layer':layer_id,'path':path,'exit_centroid':names[cur],'exit_idx':cur})
    return {'entry_layer':2,'layers':layers,'final_centroid':names[cur],'final_idx':cur}

def erng_trace(v, centroids, erng, probes):
    traces=[]
    best=None
    for p in probes:
        path, final=greedy_trace(v, centroids, erng, int(p))
        fd=dist(v, centroids[final])
        rec={'probe_entry':f'C{int(p):04d}','entry_idx':int(p),'path':path,'final_centroid':f'C{final:04d}','final_idx':int(final),'final_distance':round(fd,6)}
        traces.append(rec)
        if best is None or fd < best['final_distance']:
            best=rec
    return {'probes':traces,'selected_probe_final':best}

def one_query(lib, bank, centroids, radii, cidx, offsets, qv, thr, seed):
    q=np.ascontiguousarray(qv.reshape(1,-1).astype(np.float32))
    seeds=np.asarray([seed],dtype=np.int32)
    t0=time.perf_counter(); dd2,di,dt=B.call_direct(lib,q,bank); t1=time.perf_counter()
    gd2,gi,gseen,gdup,gt=B.call_threshold_decision_seeded(lib,q,bank,centroids,radii,cidx,offsets,seeds,thr,init_k=2); t2=time.perf_counter()
    return {'direct_idx':int(di[0]),'direct_dist':round(float(dd2[0]**0.5),6),'direct_duplicate':bool(dd2[0]<=thr*thr),'direct_ms_total':round((t1-t0)*1000,6),'graph_idx':int(gi[0]),'graph_dist':round(float(gd2[0]**0.5),6),'graph_duplicate':bool(gdup[0]),'graph_seen':int(gseen[0]),'graph_ms_total':round((t2-t1)*1000,6)}

def run(ds):
    cfg=G.DATASETS[ds]
    timing=ROOT/ds/'phase3_cpp_centroid_branchbound'/f'{ds}_cpp_centroid_branchbound_timing.json'
    selected=json.load(open(timing))['selected_centroid_count'] if timing.exists() else {'amazon':512}.get(ds,512)
    known,unknown,bank=G.load_rows(cfg['store']); bank=np.ascontiguousarray(bank.astype(np.float32))
    q_sizes,get_thr=G.q_sizes_and_thresholds(cfg['eval']); model=G.load_model(cfg['ckpt']); lib=B.load_lib()
    out_dir=ROOT/ds/'phase3_cpp_centroid_branchbound'; centroids,labels,cpath=B.build_centroids(ds,bank,out_dir,int(selected))
    centroids=np.ascontiguousarray(centroids.astype(np.float32)); cidx,offsets,radii=B.cluster_flat(bank,centroids,labels)
    hgraph,erng,probes=B.make_graphs(centroids)
    pool=[]
    base_rows=[('known',known[0]),('known',known[len(known)//3]),('known',known[-1]),('unknown',unknown[0]),('unknown',unknown[len(unknown)//2]),('unknown',unknown[-1])]
    for attack,label_name in ATTACK_EXAMPLES:
        rows=[r for _,r in base_rows]
        embs=G.embed_eval_rows(ds,model,rows,attack,label_name,batch_size=16)
        for (side,row),qv in zip(base_rows,embs):
            qv=np.ascontiguousarray(qv.astype(np.float32)); thr=get_thr(min(100,len(known),len(unknown)),attack)
            ht=hnsw_trace(qv,centroids,hgraph); et=erng_trace(qv,centroids,erng,probes)
            hq=one_query(lib,bank,centroids,radii,cidx,offsets,qv,thr,ht['final_idx'])
            eq=one_query(lib,bank,centroids,radii,cidx,offsets,qv,thr,et['selected_probe_final']['final_idx'])
            pool.append({'dataset':ds,'side':side,'attack':attack,'file':row_name(row),'path':row.get('path',''),'threshold':round(float(thr),6),'hnsw':hq,'erng':eq,'hnsw_speedup':round(hq['direct_ms_total']/max(hq['graph_ms_total'],1e-9),2),'erng_speedup':round(eq['direct_ms_total']/max(eq['graph_ms_total'],1e-9),2)})
    pool=sorted(pool,key=lambda x:x['hnsw']['graph_ms_total']+x['erng']['graph_ms_total'])
    examples10=[]
    for i in np.linspace(0,len(pool)-1,10,dtype=int): examples10.append(pool[int(i)])
    trace_examples=[]
    for attack,label_name in TRACE_PICK:
        side,row='known',known[len(trace_examples)%len(known)]
        qv=G.embed_eval_rows(ds,model,[row],attack,label_name,batch_size=1)[0].astype(np.float32)
        thr=get_thr(min(100,len(known),len(unknown)),attack)
        ht=hnsw_trace(qv,centroids,hgraph); et=erng_trace(qv,centroids,erng,probes)
        hq=one_query(lib,bank,centroids,radii,cidx,offsets,qv,thr,ht['final_idx'])
        eq=one_query(lib,bank,centroids,radii,cidx,offsets,qv,thr,et['selected_probe_final']['final_idx'])
        trace_examples.append({'dataset':ds,'attack':attack,'file':row_name(row),'path':row.get('path',''),'threshold':round(float(thr),6),'hnsw_trace':ht,'erng_trace':et,'hnsw_decision':hq,'erng_decision':eq})
    payload={'dataset':ds,'selected_centroids':int(selected),'centroid_file':str(cpath),'known_count':len(known),'unknown_count':len(unknown),'examples10':examples10,'best_avg_worst':{'best':pool[0],'avg':pool[len(pool)//2],'worst':pool[-1]},'trace_examples3':trace_examples}
    out=out_dir/f'{ds}_graph_path_examples_hnsw_erng.json'; out.write_text(json.dumps(payload,indent=2))
    print('[TraceExamples] saved',out,flush=True)

if __name__=='__main__':
    for ds in sys.argv[1:]: run(ds)
