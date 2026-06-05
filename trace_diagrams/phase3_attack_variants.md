# Attack Variant Reaches Original / Duplicate Neighborhood Diagrams

This is the corrected version in the same style as the original-image trace file. Each example starts from an attack variant and ends at either the exact original known-bank image or the duplicate neighborhood that produced the same duplicate decision as direct dense search.

Important: CIFAR, Flickr, and UCID include exact original-index attack matches. Amazon saved examples are duplicate-correct but land on nearby duplicate-neighborhood indices, so they are marked as duplicate-neighborhood matches instead of falsely calling them exact original matches.

## CIFAR

### Attack Variant Example 1: `bird_train_36832_org.png` attack `bright`

Reasoning: the attacked query is routed into the `bird_pool region`. The final graph distance is compared with the calibrated `bright` threshold `0.762759`. Exact original-index match: `True`.

#### HNSW
```text
QUERY ATTACK VARIANT: bird_train_36832_org.png + bright

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: bird_pool region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: bird_pool region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: bird_pool region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: bird_pool region
        | MATCHED KNOWN IMAGE: bird_train_36832_org.png (exact same original index 16999)
        | graph distance: 0.700132  threshold: 0.762759
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 10.311178 ms | HNSW time: 0.366937 ms | checked: 28
```

#### ERNG
```text
QUERY ATTACK VARIANT: bird_train_36832_org.png + bright

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters bird_pool region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks bird_pool region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in bird_pool region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: bird_pool region
        | MATCHED KNOWN IMAGE: bird_train_36832_org.png (exact same original index 16999)
        | graph distance: 0.700132  threshold: 0.762759
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 10.387564 ms | ERNG time: 0.353309 ms | checked: 28
```

### Attack Variant Example 2: `bird_train_36832_org.png` attack `contrast`

Reasoning: the attacked query is routed into the `bird_pool region`. The final graph distance is compared with the calibrated `contrast` threshold `0.639837`. Exact original-index match: `True`.

#### HNSW
```text
QUERY ATTACK VARIANT: bird_train_36832_org.png + contrast

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: bird_pool region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: bird_pool region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: bird_pool region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: bird_pool region
        | MATCHED KNOWN IMAGE: bird_train_36832_org.png (exact same original index 16999)
        | graph distance: 0.362703  threshold: 0.639837
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 10.384661 ms | HNSW time: 0.380107 ms | checked: 28
```

#### ERNG
```text
QUERY ATTACK VARIANT: bird_train_36832_org.png + contrast

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters bird_pool region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks bird_pool region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in bird_pool region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: bird_pool region
        | MATCHED KNOWN IMAGE: bird_train_36832_org.png (exact same original index 16999)
        | graph distance: 0.362703  threshold: 0.639837
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 10.440252 ms | ERNG time: 0.367029 ms | checked: 28
```

### Attack Variant Example 3: `bird_train_36832_org.png` attack `crop40`

Reasoning: the attacked query is routed into the `bird_pool region`. The final graph distance is compared with the calibrated `crop40` threshold `0.798677`. Exact original-index match: `True`.

#### HNSW
```text
QUERY ATTACK VARIANT: bird_train_36832_org.png + crop40

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: bird_pool region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: bird_pool region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: bird_pool region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: bird_pool region
        | MATCHED KNOWN IMAGE: bird_train_36832_org.png (exact same original index 15755)
        | graph distance: 0.627144  threshold: 0.798677
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 11.946399 ms | HNSW time: 0.494367 ms | checked: 10
```

#### ERNG
```text
QUERY ATTACK VARIANT: bird_train_36832_org.png + crop40

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters bird_pool region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks bird_pool region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in bird_pool region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: bird_pool region
        | MATCHED KNOWN IMAGE: bird_train_36832_org.png (exact same original index 15755)
        | graph distance: 0.627144  threshold: 0.798677
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 11.632583 ms | ERNG time: 0.52571 ms | checked: 10
```

## FLICKR

### Attack Variant Example 1: `2575233295.jpg` attack `bright`

Reasoning: the attacked query is routed into the `Flickr natural-image region`. The final graph distance is compared with the calibrated `bright` threshold `0.522851`. Exact original-index match: `True`.

#### HNSW
```text
QUERY ATTACK VARIANT: 2575233295.jpg + bright

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: Flickr natural-image region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: Flickr natural-image region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: Flickr natural-image region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: Flickr natural-image region
        | MATCHED KNOWN IMAGE: 2575233295.jpg (exact same original index 3333)
        | graph distance: 0.383788  threshold: 0.522851
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 6.196034 ms | HNSW time: 0.197979 ms | checked: 39
```

#### ERNG
```text
QUERY ATTACK VARIANT: 2575233295.jpg + bright

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters Flickr natural-image region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks Flickr natural-image region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in Flickr natural-image region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: Flickr natural-image region
        | MATCHED KNOWN IMAGE: 2575233295.jpg (exact same original index 3333)
        | graph distance: 0.383788  threshold: 0.522851
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 6.209304 ms | ERNG time: 0.183923 ms | checked: 39
```

### Attack Variant Example 2: `3750418259.jpg` attack `bright`

Reasoning: the attacked query is routed into the `Flickr natural-image region`. The final graph distance is compared with the calibrated `bright` threshold `0.522851`. Exact original-index match: `True`.

#### HNSW
```text
QUERY ATTACK VARIANT: 3750418259.jpg + bright

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: Flickr natural-image region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: Flickr natural-image region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: Flickr natural-image region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: Flickr natural-image region
        | MATCHED KNOWN IMAGE: 3750418259.jpg (exact same original index 3633)
        | graph distance: 0.468832  threshold: 0.522851
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 6.095898 ms | HNSW time: 0.176906 ms | checked: 70
```

#### ERNG
```text
QUERY ATTACK VARIANT: 3750418259.jpg + bright

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters Flickr natural-image region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks Flickr natural-image region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in Flickr natural-image region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: Flickr natural-image region
        | MATCHED KNOWN IMAGE: 3750418259.jpg (exact same original index 3633)
        | graph distance: 0.468832  threshold: 0.522851
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 6.113799 ms | ERNG time: 0.185327 ms | checked: 70
```

### Attack Variant Example 3: `2494088238.jpg` attack `rotate40`

Reasoning: the attacked query is routed into the `Flickr natural-image region`. The final graph distance is compared with the calibrated `rotate40` threshold `0.50314`. Exact original-index match: `True`.

#### HNSW
```text
QUERY ATTACK VARIANT: 2494088238.jpg + rotate40

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: Flickr natural-image region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: Flickr natural-image region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: Flickr natural-image region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: Flickr natural-image region
        | MATCHED KNOWN IMAGE: 2494088238.jpg (exact same original index 0)
        | graph distance: 0.451386  threshold: 0.50314
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 6.904714 ms | HNSW time: 0.262709 ms | checked: 79
```

#### ERNG
```text
QUERY ATTACK VARIANT: 2494088238.jpg + rotate40

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters Flickr natural-image region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks Flickr natural-image region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in Flickr natural-image region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: Flickr natural-image region
        | MATCHED KNOWN IMAGE: 2494088238.jpg (exact same original index 0)
        | graph distance: 0.451386  threshold: 0.50314
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 7.149734 ms | ERNG time: 0.319408 ms | checked: 79
```

## UCID

### Attack Variant Example 1: `542_aug_1.jpg` attack `rotate40`

Reasoning: the attacked query is routed into the `UCID scene region`. The final graph distance is compared with the calibrated `rotate40` threshold `0.727884`. Exact original-index match: `True`.

#### HNSW
```text
QUERY ATTACK VARIANT: 542_aug_1.jpg + rotate40

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: UCID scene region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: UCID scene region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: UCID scene region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: UCID scene region
        | MATCHED KNOWN IMAGE: 542_aug_1.jpg (exact same original index 1500)
        | graph distance: 0.479585  threshold: 0.727884
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 2.907867 ms | HNSW time: 0.137984 ms | checked: 32
```

#### ERNG
```text
QUERY ATTACK VARIANT: 542_aug_1.jpg + rotate40

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters UCID scene region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks UCID scene region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in UCID scene region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: UCID scene region
        | MATCHED KNOWN IMAGE: 542_aug_1.jpg (exact same original index 1500)
        | graph distance: 0.479585  threshold: 0.727884
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 2.897093 ms | ERNG time: 0.132318 ms | checked: 32
```

### Attack Variant Example 2: `830_aug_1.jpg` attack `rotate40`

Reasoning: the attacked query is routed into the `UCID scene region`. The final graph distance is compared with the calibrated `rotate40` threshold `0.727884`. Exact original-index match: `True`.

#### HNSW
```text
QUERY ATTACK VARIANT: 830_aug_1.jpg + rotate40

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: UCID scene region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: UCID scene region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: UCID scene region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: UCID scene region
        | MATCHED KNOWN IMAGE: 830_aug_1.jpg (exact same original index 3451)
        | graph distance: 0.705829  threshold: 0.727884
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 2.90489 ms | HNSW time: 0.146092 ms | checked: 26
```

#### ERNG
```text
QUERY ATTACK VARIANT: 830_aug_1.jpg + rotate40

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters UCID scene region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks UCID scene region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in UCID scene region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: UCID scene region
        | MATCHED KNOWN IMAGE: 830_aug_1.jpg (exact same original index 3451)
        | graph distance: 0.705829  threshold: 0.727884
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 2.899269 ms | ERNG time: 0.138603 ms | checked: 26
```

### Attack Variant Example 3: `542_aug_1.jpg` attack `watermark_asset`

Reasoning: the attacked query is routed into the `UCID scene region`. The final graph distance is compared with the calibrated `watermark_asset` threshold `0.660665`. Exact original-index match: `True`.

#### HNSW
```text
QUERY ATTACK VARIANT: 542_aug_1.jpg + watermark_asset

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: UCID scene region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: UCID scene region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: UCID scene region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: UCID scene region
        | MATCHED KNOWN IMAGE: 542_aug_1.jpg (exact same original index 1500)
        | graph distance: 0.474477  threshold: 0.660665
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 3.467777 ms | HNSW time: 0.256855 ms | checked: 32
```

#### ERNG
```text
QUERY ATTACK VARIANT: 542_aug_1.jpg + watermark_asset

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters UCID scene region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks UCID scene region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in UCID scene region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: UCID scene region
        | MATCHED KNOWN IMAGE: 542_aug_1.jpg (exact same original index 1500)
        | graph distance: 0.474477  threshold: 0.660665
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 3.458141 ms | ERNG time: 0.270031 ms | checked: 32
```

## AMAZON

### Attack Variant Example 1: `headboards_56_org.jpg` attack `crop40`

Reasoning: the attacked query is routed into the `headboards / headboards_56 product region`. The final graph distance is compared with the calibrated `crop40` threshold `1.003517`. Exact original-index match: `False`.

#### HNSW
```text
QUERY ATTACK VARIANT: headboards_56_org.jpg + crop40

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: headboards / headboards_56 product region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: headboards / headboards_56 product region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: headboards / headboards_56 product region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: headboards / headboards_56 product region
        | MATCHED DUPLICATE NEIGHBOR: known index 11158 (direct original index 24525)
        | graph distance: 0.910958  threshold: 1.003517
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 56.519996 ms | HNSW time: 0.733813 ms | checked: 101
```

#### ERNG
```text
QUERY ATTACK VARIANT: headboards_56_org.jpg + crop40

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters headboards / headboards_56 product region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks headboards / headboards_56 product region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in headboards / headboards_56 product region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: headboards / headboards_56 product region
        | MATCHED DUPLICATE NEIGHBOR: known index 11158 (direct original index 24525)
        | graph distance: 0.910958  threshold: 1.003517
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 56.120395 ms | ERNG time: 0.724469 ms | checked: 101
```

### Attack Variant Example 2: `portable_cd_players_37_org.jpg` attack `crop40`

Reasoning: the attacked query is routed into the `portable_cd_players / portable_cd_players_37 product region`. The final graph distance is compared with the calibrated `crop40` threshold `1.003517`. Exact original-index match: `False`.

#### HNSW
```text
QUERY ATTACK VARIANT: portable_cd_players_37_org.jpg + crop40

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: portable_cd_players / portable_cd_players_37 product region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: portable_cd_players / portable_cd_players_37 product region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: portable_cd_players / portable_cd_players_37 product region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: portable_cd_players / portable_cd_players_37 product region
        | MATCHED DUPLICATE NEIGHBOR: known index 40 (direct original index 8143)
        | graph distance: 0.621152  threshold: 1.003517
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 62.516486 ms | HNSW time: 0.89058 ms | checked: 72
```

#### ERNG
```text
QUERY ATTACK VARIANT: portable_cd_players_37_org.jpg + crop40

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters portable_cd_players / portable_cd_players_37 product region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks portable_cd_players / portable_cd_players_37 product region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in portable_cd_players / portable_cd_players_37 product region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: portable_cd_players / portable_cd_players_37 product region
        | MATCHED DUPLICATE NEIGHBOR: known index 40 (direct original index 8143)
        | graph distance: 0.621152  threshold: 1.003517
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 59.926745 ms | ERNG time: 0.701889 ms | checked: 72
```

### Attack Variant Example 3: `rear_facing_mirrors_42_org.jpg` attack `crop40`

Reasoning: the attacked query is routed into the `rear_facing_mirrors / rear_facing_mirrors_42 product region`. The final graph distance is compared with the calibrated `crop40` threshold `1.003517`. Exact original-index match: `False`.

#### HNSW
```text
QUERY ATTACK VARIANT: rear_facing_mirrors_42_org.jpg + crop40

HNSW layered route
        +--------------------------------------------------+
        | Layer 2 broad shortcut: rear_facing_mirrors / rear_facing_mirrors_42 product region
        | coarse route toward visual family
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 1 medium neighborhood: rear_facing_mirrors / rear_facing_mirrors_42 product region
        | closer centroid neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Layer 0 dense local search: rear_facing_mirrors / rear_facing_mirrors_42 product region
        | candidate duplicate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: rear_facing_mirrors / rear_facing_mirrors_42 product region
        | MATCHED DUPLICATE NEIGHBOR: known index 1269 (direct original index 22129)
        | graph distance: 0.842299  threshold: 1.003517
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | HNSW duplicate: True
Direct time: 57.168417 ms | HNSW time: 0.707576 ms | checked: 105
```

#### ERNG
```text
QUERY ATTACK VARIANT: rear_facing_mirrors_42_org.jpg + crop40

ERNG multi-probe route
        +--------------------------------------------------+
        | Probe A enters rear_facing_mirrors / rear_facing_mirrors_42 product region
        | local KNN + exponential-rank skip checks
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Probe B cross-checks rear_facing_mirrors / rear_facing_mirrors_42 product region
        | independent greedy path
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | Selected best probe in rear_facing_mirrors / rear_facing_mirrors_42 product region
        | final candidate neighborhood
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: rear_facing_mirrors / rear_facing_mirrors_42 product region
        | MATCHED DUPLICATE NEIGHBOR: known index 1269 (direct original index 22129)
        | graph distance: 0.842299  threshold: 1.003517
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct duplicate: True | ERNG duplicate: True
Direct time: 61.355487 ms | ERNG time: 0.700524 ms | checked: 105
```
