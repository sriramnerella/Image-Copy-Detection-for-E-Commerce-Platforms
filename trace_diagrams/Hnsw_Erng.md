# HNSW And ERNG Architecture README

This document explains how the Phase-3 graph-query layer was built from the final embedding stores, including the numerical settings used for each dataset.

The goal of Phase-3 is not to retrain the embedding model. The goal is to make query search faster while preserving the same duplicate / non-duplicate decision as direct dense search.

## Input To Phase-3

Each dataset already has a trained Siamese ConvNeXt embedding store.

```text
image
   -> ConvNeXt Siamese encoder
   -> 384-dimensional embedding vector
   -> stored in known-bank / unknown-bank JSON store
```

For graph search, only the known-bank embeddings are used as the searchable duplicate bank.

```text
query embedding
   -> graph routes to centroid neighborhood
   -> candidate known embeddings are checked
   -> nearest distance is compared with calibrated threshold
```

Final decision:

```text
distance <= threshold  => duplicate
distance > threshold   => non-duplicate
```

## Dataset Store Sizes

| Dataset | Known Embeddings | Unknown Embeddings | Search Bank Used By Graph |
|---|---:|---:|---:|
| CIFAR | 17,000 | 9,000 | 17,000 known |
| Flickr | 10,000 | 5,000 | 10,000 known |
| UCID | 4,500 | 850 | 4,500 known |
| Amazon | 35,000 | 15,000 | 35,000 known |

## Centroid Construction

For each dataset, the known-bank embeddings are compressed into centroids.

Implementation:

```text
MiniBatchKMeans over known-bank embeddings
```

Fallback if KMeans fails:

```text
random known-bank embeddings are selected as centroids
then every known embedding is assigned to its nearest centroid
```

Each centroid stores:

```text
centroid vector
assigned known embedding indices
cluster radius
cluster offset range
```

The cluster radius is the maximum distance from the centroid to any assigned known embedding:

```text
radius(Ci) = max distance(Ci, assigned_embedding_j)
```

This allows branch-bound search:

```text
if centroid_distance - radius is already worse than current best,
that centroid cluster can be skipped.
```

## Centroid Sweep

Centroid counts were swept per dataset, then the final count was selected based on matching direct-search decisions while reducing query time.

| Dataset | Swept Centroid Counts | Final Selected Centroids |
|---|---|---:|
| CIFAR | 128, 192, 256, 384, 512, 768, 1024, 1536 | 512 |
| Flickr | 96, 128, 192, 256, 384, 512, 768, 1024 | 192 |
| UCID | 64, 96, 128, 192, 256, 384, 512, 768 | 128 |
| Amazon | 192, 256, 384, 512, 768, 1024, 1536, 2048 | 512 |

Sweep artifacts are stored in:

```text
artifacts/centroid_sweep/cifar
artifacts/centroid_sweep/flickr
artifacts/centroid_sweep/ucid
artifacts/centroid_sweep/amazon
```

Final selected centroid files are stored in:

```text
artifacts/centroids/cifar
artifacts/centroids/flickr
artifacts/centroids/ucid
artifacts/centroids/amazon
```

## HNSW Architecture

HNSW means:

```text
Hierarchical Navigable Small World graph
```

In this project, HNSW is built over centroids, not over every image embedding directly.

### HNSW Layer Construction

Centroid centrality is computed first:

```text
centrality(Ci) = sum distance(Ci, Cj) for all centroids Cj
```

More central centroids are placed in higher layers.

Layer rule:

```text
Layer 2: top 10% most central centroids
Layer 1: top 35% most central centroids
Layer 0: all centroids
```

Every centroid in a higher layer also exists in lower layers conceptually.

### HNSW Layer Sizes

| Dataset | Total Centroids | Layer 2 Top Layer | Layer 1 Middle Layer | Layer 0 Bottom Layer |
|---|---:|---:|---:|---:|
| CIFAR | 512 | 52 | 180 | 512 |
| Flickr | 192 | 20 | 68 | 192 |
| UCID | 128 | 13 | 45 | 128 |
| Amazon | 512 | 52 | 180 | 512 |

### HNSW Edge Construction

Within each layer, each active centroid gets:

```text
local KNN edges
exponential-rank skip edges
```

Final local setting used in the C++ branch-bound version:

```text
local_m = 12
```

For each centroid:

```text
1. Sort other active centroids by distance.
2. Connect to the nearest 12 local neighbors.
3. Add exponential rank neighbors at ranks:
   1, 2, 4, 8, 16, 32, ...
4. Remove duplicate edges.
```

The exponential ranks give long-range shortcuts. The local KNN edges give dense neighborhood movement.

### HNSW Query

```text
query embedding
   |
   v
choose nearest centroid in Layer 2
   |
   v
greedy search inside Layer 2
   |
   v
drop to Layer 1 and continue greedy search
   |
   v
drop to Layer 0 and continue greedy search
   |
   v
final centroid seed
   |
   v
branch-bound candidate scan around centroid cluster
   |
   v
nearest known embedding
   |
   v
threshold decision
```

HNSW is therefore a layered route to a good candidate region.

## ERNG Architecture

ERNG means:

```text
Exponential Rank Navigation Graph
```

ERNG is a flatter graph than HNSW. It does not use strict layers. It uses multiple probes and exponential-rank links.

### ERNG Edge Construction

ERNG uses all centroids as active nodes:

```text
active = all centroids
```

For each centroid:

```text
1. Sort all other centroids by distance.
2. Connect to nearest 12 local neighbors.
3. Add exponential rank skip links:
   1, 2, 4, 8, 16, 32, ...
4. Remove duplicate edges.
```

Final local setting:

```text
local_m = 12
```

### ERNG Probe Construction

The final ERNG search uses 8 probes per dataset.

Probe construction:

```text
4 most central centroids
+ 4 deterministic random centroids
= up to 8 unique probe starts
```

Random seed:

```text
20260603
```

### ERNG Probe IDs Used

| Dataset | Number Of ERNG Probes | Probe Centroids |
|---|---:|---|
| CIFAR | 8 | C0005, C0036, C0129, C0193, C0253, C0280, C0297, C0310 |
| Flickr | 8 | C0013, C0026, C0097, C0104, C0110, C0115, C0123, C0140 |
| UCID | 8 | C0009, C0031, C0044, C0069, C0073, C0076, C0085, C0111 |
| Amazon | 8 | C0036, C0130, C0191, C0280, C0297, C0310, C0401, C0445 |

### ERNG Query

```text
query embedding
   |
   v
launch 8 probes from selected centroids
   |
   v
each probe greedily follows local + exponential-rank edges
   |
   v
each probe stops at a local centroid minimum
   |
   v
select the probe whose final centroid is closest to the query
   |
   v
branch-bound candidate scan around selected centroid cluster
   |
   v
nearest known embedding
   |
   v
threshold decision
```

ERNG is therefore a multi-start graph route to a good candidate region.

## Branch-Bound Candidate Search

After HNSW or ERNG produces a centroid seed, the C++ branch-bound kernel searches candidate clusters efficiently.

The important data structures are:

```text
centroids: centroid vectors
labels: centroid assignment for each known embedding
cidx: known-bank indices sorted by centroid label
offsets: start/end range per centroid cluster
radii: maximum radius per centroid cluster
```

The C++ search compares the query to centroid clusters and skips clusters that cannot improve the current nearest result.

This is why graph search is faster than dense search:

```text
direct dense search:
    compare query with every known embedding

HNSW / ERNG:
    route to useful centroid region
    scan much smaller candidate subset
    apply same duplicate threshold
```

## Final Numerical Results For Selected Centroids

These numbers are from the final selected centroid timing JSON files.

| Dataset | Direct Avg | Direct ms/query | HNSW Avg | HNSW ms/query | HNSW Match | HNSW Recall | HNSW Seen | ERNG Avg | ERNG ms/query | ERNG Match | ERNG Recall | ERNG Seen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR | 91.17 | 8.463841 | 91.17 | 0.719120 | 100.00 | 79.38 | 709.30 | 91.17 | 0.712509 | 100.00 | 79.21 | 706.80 |
| Flickr | 88.12 | 4.900420 | 88.12 | 0.201396 | 100.00 | 91.67 | 205.70 | 88.12 | 0.201636 | 100.00 | 91.65 | 205.40 |
| UCID | 83.81 | 2.315464 | 83.81 | 0.201394 | 100.00 | 89.08 | 242.90 | 83.81 | 0.200719 | 100.00 | 89.08 | 242.90 |
| Amazon | 81.96 | 40.965028 | 81.96 | 1.276281 | 100.00 | 70.81 | 573.40 | 81.96 | 1.249923 | 100.00 | 70.81 | 572.20 |

Interpretation:

```text
Match = duplicate / non-duplicate decision match versus direct dense search.
Recall = exact nearest-neighbor identity recall versus direct dense search.
Seen = average candidate embeddings checked after graph routing.
```

The project objective is duplicate decision equivalence, so `Match = 100%` is the key correctness metric.

## Match vs Recall

Match and recall are not the same.

```text
Decision match:
    graph and direct both say duplicate / non-duplicate.

Nearest-neighbor recall:
    graph returns the exact same nearest image index as direct.
```

Example:

```text
Direct nearest distance = 0.42
Graph nearest distance  = 0.45
Threshold               = 0.60

Both are <= threshold, so both decide duplicate.
Decision match = correct.
Exact NN recall may be different if the nearest index is not identical.
```

## Exact Original vs Duplicate Neighborhood

Two duplicate cases are valid:

### Exact Original Match

```text
attack query
   -> centroid route
   -> exact original known-bank image
   -> distance <= threshold
   -> duplicate
```

### Duplicate Neighborhood Match

```text
attack query
   -> centroid route
   -> nearby known-bank image
   -> distance <= threshold
   -> duplicate
```

Both are duplicate decisions. For identity-level explanation, use exact-original traces. For duplicate classification, either case is valid if the threshold decision matches direct search.

## Important Files

Main graph code:

```text
code/final_codebase/phase3_graph/centroid_hnsw_erng.py
code/final_codebase/phase3_graph/centroid_branch_bound_benchmark.py
code/final_codebase/phase3_graph/centroid_branch_bound_kernel.cpp
```

Centroid sweep / autotune code:

```text
code/final_codebase/phase3_graph/erng_parameter_sweep.py
code/final_codebase/phase3_graph/centroid_branch_bound_autotune.py
code/final_codebase/phase3_graph/centroid_branch_bound_timing.py
code/final_codebase/phase3_graph/centroid_branch_bound_tuned.py
code/final_codebase/phase3_graph/centroid_clean_timing_all.py
code/final_codebase/phase3_graph/fast_centroid_ann.py
```

Final selected centroid timing JSONs:

```text
artifacts/json_results/cifar/cifar_cpp_centroid_branchbound_timing.json
artifacts/json_results/flickr/flickr_cpp_centroid_branchbound_timing.json
artifacts/json_results/ucid/ucid_cpp_centroid_branchbound_timing.json
artifacts/json_results/amazon/amazon_cpp_centroid_branchbound_timing.json
```

Trace examples:

```text
trace_diagrams/phase3_labeled_trace_diagrams_with_matches.md
trace_diagrams/phase3_attack_variant_trace_diagrams.md
```

Exact original attack traces:

```text
artifacts/json_results/cifar/cifar_exact_original_attack_traces.json
artifacts/json_results/flickr/flickr_exact_original_attack_traces.json
artifacts/json_results/ucid/ucid_exact_original_attack_traces.json
artifacts/json_results/amazon/amazon_exact_original_attack_traces.json
```
