# Final Store / Eval Report - completed outputs as of 2026-06-01

This report includes the best validated multi-query outputs for CIFAR, Flickr, UCID, and Amazon. For each dataset, the same calibrated threshold is used for a given attack across all query sizes. A newer adjustment checkpoint is promoted only when its calibrated multi-query evaluation improves the retained operating point.

## Clean Checkpoint Paths
The clean folders preserve later archived candidates where available. The selected checkpoint is the file that actually backs the retained report metrics.

| Dataset | Clean folder | Selected checkpoint | Training archive | Adjustment archive | Selected clean epoch | Selected file |
|---|---|---|---:|---:|---:|---|
| CIFAR | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/cifar` | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/cifar/cifar-epoch-015-training.pt` | 1-15 | none | 15 | `cifar-epoch-015-training.pt` |
| Flickr | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/flickr` | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/flickr/flickr-epoch-026-adjustment.pt` | 1-24 | 25-26 | 26 | `flickr-epoch-026-adjustment.pt` |
| UCID | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/ucid` | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/ucid/ucid-epoch-037-adjustment.pt` | 1-27 | 28-37 | 37 | `ucid-epoch-037-adjustment.pt` |
| Amazon | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/amazon` | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/amazon/amazon-epoch-033-adjustment.pt` | 1-31 | 32-34 | 33 | `amazon-epoch-033-adjustment.pt` |

Checkpoint manifest: `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/MANIFEST.tsv`
Selected report-point manifest: `/home/saketh/json/final_named_artifacts_20260601/FINAL_SELECTED_POINTS.json`

## Latest Verification Decision
UCID and Flickr were recalibrated after the latest adjustment run on 2026-06-01. Their newer checkpoints were not promoted because the retained calibrated operating points remain better. Amazon is also kept on its stable calibrated table, as requested.

| Dataset | Retained report point | Newer candidate checked | Retained Q100 All Avg | Candidate Q100 All Avg | Decision |
|---|---|---|---:|---:|---|
| Flickr | selected adjustment epoch 34 | adjustment epoch 37 | 89.58 | 89.44 | Keep retained best |
| UCID | selected adjustment epoch 33 | adjustment epoch 35 | 86.46 | 84.90 | Keep retained best |
| Amazon | selected adjustment epoch 33 | adjustment epoch 34 candidate | 82.10 | not promoted | Keep retained best |

Audit JSONs for the newer UCID and Flickr candidates are listed in `Eval JSON Paths`.

## System Architecture
The project is an image-copy detection pipeline built around a Siamese ConvNeXt-B embedding model. Each image is mapped to a compact embedding, and duplicate detection is done by nearest-neighbour distance against the known-image bank.

| Stage | What happens | Output |
|---|---|---|
| Dataset split | Images are split into known/original-bank images and unknown/non-duplicate images. | Known and unknown source pools |
| Pair manifest | Training pairs are created as hard positives, soft positives, hard negatives, and soft negatives. | Manifest JSON/CSV used by training |
| Attack generation | Soft attacks are cached or sampled; stricter attacks are generated during training/eval to force attack invariance. | Original and attacked image views |
| Siamese training | The two ConvNeXt branches share weights. Positive pairs are pulled together; negative pairs are pushed apart. | Checkpoint per epoch |
| Store harvest/refill | Accepted images are embedded and written to the store as known or unknown rows. | Store JSON with embeddings and paths |
| FAISS retrieval | Query embeddings are compared to the known-bank embeddings using nearest-neighbour search. | Minimum distance to known bank |
| Threshold decision | A query is duplicate if its nearest known-bank distance is within the selected threshold; otherwise it is treated as unknown/non-duplicate. | Duplicate / non-duplicate decision |

For known images, the intended geometry is: original known image close to the known bank, and all attack variants close to that original neighbourhood. For unknown images, the intended geometry is: original unknown image far from the known bank, while its own attack variants remain close to the unknown original.

## Dataset Description
The four datasets are not visually identical, so their attack robustness should not be interpreted as a pure model-only comparison. Each dataset has a different image style, object diversity, background behaviour, and source quality.

| Dataset | Image type | What it mainly contains | Why it matters for copy detection | Final store use |
|---|---|---|---|---|
| CIFAR | Small natural-object/classification images | Low-resolution object categories such as vehicles, animals, ships, trucks, birds, and similar compact objects. | Images are small and class-like, so many transformations preserve the global object shape; embeddings can stay stable for soft/medium attacks, but fine details are limited. | Used as a compact robustness benchmark with `17,000 known / 9,000 unknown` embeddings. |
| Flickr | Real-world photographic images | Outdoor/indoor scenes, people, animals, vehicles, buildings, objects, and mixed natural backgrounds. | High visual diversity makes it useful for testing copy detection beyond product-only images; crops/background changes are harder because scenes contain many distractors. | Used as a broad real-image benchmark with `10,000 known / 5,000 unknown` embeddings. |
| UCID | General image-copy / content-based retrieval style images | Smaller controlled image set with varied natural images and generated attack variants. | Useful for measuring controlled copy-detection geometry, but the available pool is much smaller than Amazon/Flickr, so store expansion is naturally limited. | Used as a focused copy-detection benchmark with the latest relaxed-refill store: `4,500 known / 850 unknown` embeddings. |
| Amazon | E-commerce/product images | Product photos across categories such as electronics, home, sports, cosmetics, baby products, and wearables. | This is the most application-aligned dataset: product images have repeated layouts, white backgrounds, category imbalance, and seller-style variations; known-side hard attacks are more difficult. | Used as the main product-copy dataset with `35,000 known / 15,000 unknown` embeddings. |

Short reading: CIFAR is compact and class-like, Flickr is broad real-world photography, UCID is a smaller controlled copy-detection set, and Amazon is the product-image target domain.

## Attack Difficulty Groups
`original` is treated as a mandatory reference gate, not as an attack. The remaining 25 attacks are grouped by practical difficulty for analysis.

| Group | Attacks | Main reason |
|---|---|---|
| Original / mandatory gate | original | Reference image; should remain 100% when the store and checkpoint are aligned. |
| Soft | rotate_m5, rotate5, crop5, bright_light, bright, resize, resize_compress, blur, contrast | Small photometric, resize, blur, or mild geometric changes. |
| Medium | rotate_m10, rotate15, rotate30, crop10, crop20, flip_h, flip_v, flip, watermark_text, watermark_asset | Moderate geometry, crop, flip, or watermark changes that can shift neighbourhoods. |
| Hard | rotate40, crop40, heavy_bright, mix_rotate10_bright, mix_crop20_watermark, bg_color_change | Severe crop/rotation/background/mixed attacks that most often create the tail failures. |

## Architecture Notes Needed To Read The Tables
- `Known` accuracy means attacked versions of known-bank images still retrieve inside the duplicate neighbourhood.
- `Unknown` accuracy means unknown images and their attacks stay outside the known-bank duplicate threshold.
- The reported threshold is attack-specific for analysis. At deployment, the threshold summary gives strict/balanced/loose operating points when the attack type is unknown.
- The store JSON contains image paths plus embeddings. The checkpoint must match the store geometry; otherwise the same paths can produce different retrieval behaviour.
- FAISS is used only to make nearest-neighbour lookup fast. It does not change the model geometry.

## Mathematical Retrieval Geometry
Let `f_theta(x)` be the ConvNeXt Siamese embedding of image `x`, and let `B_K = {f_theta(k_1), ..., f_theta(k_n)}` be the known-bank embedding set. The retrieval distance used in the tables is the nearest-bank distance:

```text
D_K(x) = min_{b in B_K} d(f_theta(x), b)
```

where `d` is the embedding distance used by FAISS/nearest-neighbour search. A query is treated as duplicate when `D_K(x) <= tau`, where `tau` is the selected threshold. When the exact attack type is known during analysis, the report gives one threshold `tau_a` per attack. When attack type is unknown at deployment, the report gives dataset-level strict/balanced/loose threshold ranges.

For a known image `x` and an attack transform `a_i`, the desired geometry is:

```text
D_K(x) <= tau_known
d(f_theta(a_i(x)), f_theta(x)) <= epsilon_attack
```

This gives the triangle-style intuition used in the project:

```text
D_K(a_i(x)) <= d(f_theta(a_i(x)), f_theta(x)) + D_K(x)
```

So if the original image is near the known bank, and every attacked variant is near the original embedding, the attacked variants should also remain inside the duplicate neighbourhood.

For an unknown image `u`, the desired geometry is the opposite with respect to the known bank, while still keeping attack variants stable around their own original:

```text
D_K(u) > tau_unknown
d(f_theta(a_i(u)), f_theta(u)) <= epsilon_attack
```

Training uses this idea through positive and negative pair pressure: hard/soft positives reduce embedding distance for duplicate/attack variants, while hard/soft negatives increase separation between known and unknown neighbourhoods.

## How Training Pairs Are Created
The manifest stores lightweight pairing instructions and source paths. It does not store every possible attacked image. Soft and hard positive pairs are materialized differently to balance speed and attack coverage.

| Pair type | Manifest paths | Image generation during loading | Training purpose |
|---|---|---|---|
| `soft_pos` | `path_1` is the original image. `path_2` is an existing same-product duplicate or a prepared soft-attack cache file on disk. | The cached/prebuilt variant is loaded directly. When a row is not marked as prebuilt, the loader may apply one additional lightweight soft transform in RAM. | Teach inexpensive duplicate invariance without regenerating the same easy views every batch. |
| `hard_pos` | `path_1` and `path_2` deliberately point to the same original image. | The loader reads the original twice and applies one or more strict attacks to the second copy in RAM. The selected hard attack rotates over training steps using `times_trained`, so coverage improves across epochs without storing every heavy variant. | Pull difficult attacked variants back toward the original-image neighbourhood. |
| `hard_neg` | Two different image paths, usually known-versus-unknown or visually confusing products from the same category. | Both files are loaded from disk. No duplicate attack image needs to be stored. | Push confusing non-duplicate embeddings apart. |
| `soft_neg` | Two different image paths, usually from more separated products or categories. | Both files are loaded from disk. | Stabilize the duplicate/non-duplicate boundary. |

Representative soft-positive row:

```json
{
  "row_id": "soft_pos_00001",
  "path_1": "dataset/product_001/original.jpg",
  "path_2": "soft_pos_cache/product_001/rotate5.jpg",
  "label": 1,
  "pair_type": "soft_pos",
  "priority": 0.0,
  "times_trained": 0,
  "prebuilt_soft_pos": true
}
```

Representative hard-positive row:

```json
{
  "row_id": "hard_pos_00001",
  "path_1": "dataset/product_001/original.jpg",
  "path_2": "dataset/product_001/original.jpg",
  "label": 1,
  "pair_type": "hard_pos",
  "priority": 0.5,
  "times_trained": 0
}
```

For the hard-positive row, the second image exists only as an in-memory attacked view during that batch. This avoids materializing a large Cartesian set of strict attack files while still training the Siamese model against strict variants.

## Store Counts
| Dataset | Store JSON | Known | Unknown | Total embeddings |
|---|---|---:|---:|---:|
| CIFAR | `/home/saketh/json/final_named_artifacts_20260601/cifar/cifar_store.json` | 17,000 | 9,000 | 26,000 |
| Flickr | `/home/saketh/json/final_named_artifacts_20260601/flickr/flickr_store.json` | 10,000 | 5,000 | 15,000 |
| UCID | `/home/saketh/json/final_named_artifacts_20260601/ucid/ucid_store.json` | 4,500 | 850 | 5,350 |
| Amazon | `/home/saketh/json/final_named_artifacts_20260601/amazon/amazon_store.json` | 35,000 | 15,000 | 50,000 |

## Proportional Pair Counts From Final Store Ratios
This table uses the same pair-ratio rule for every dataset, scaled from the final store size. The Amazon reference ratio is `hard_pos=N`, `soft_pos=0.5N`, `hard_neg=0.625N`, `soft_neg=0.375N`, so total pairs are `2.5N`.

| Dataset | Known | Unknown | N | hard_pos | soft_pos | hard_neg | soft_neg | Total pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR | 17,000 | 9,000 | 26,000 | 26,000 | 13,000 | 16,250 | 9,750 | 65,000 |
| Flickr | 10,000 | 5,000 | 15,000 | 15,000 | 7,500 | 9,375 | 5,625 | 37,500 |
| UCID | 4,500 | 850 | 5,350 | 5,350 | 2,675 | 3,344 | 2,006 | 13,375 |
| Amazon | 35,000 | 15,000 | 50,000 | 50,000 | 25,000 | 31,250 | 18,750 | 125,000 |

Rounded values are used when the final store size is not divisible cleanly by the ratio.

## Amazon Category Coverage
| Side | Images | Categories | Products | Top-level counts |
|---|---:|---:|---:|---|
| Known | 35,000 | 706 | 578 | baby_products 3051, cosmetics 4379, electronics 6043, home 8123, sports 8948, wearables 4456 |
| Unknown | 15,000 | 774 | 10,051 | baby_products 1224, cosmetics 1952, electronics 1711, home 4276, sports 3640, wearables 2197 |

## Threshold Summary
| Dataset | Single global threshold | Recommended range | Strict buffer | Balanced buffer | Loose buffer |
|---|---:|---|---:|---:|---:|
| CIFAR | 0.721172 | 0.106488 to 0.837360 | 0.485198 | 0.683731 | 0.777912 |
| Flickr | 0.343769 | 0.080286 to 0.675062 | 0.292701 | 0.343769 | 0.503140 |
| UCID | 0.614057 | 0.022737 to 0.801727 | 0.409282 | 0.614057 | 0.703969 |
| Amazon | 0.804346 | 0.164388 to 0.899645 | 0.546599 | 0.817505 | 0.851755 |

## Multi-Query Summaries
CIFAR uses fixed calibrated thresholds. Flickr and UCID use accepted balanced multi-query adjustment points. Amazon uses its retained `final_total` threshold-search operating point aligned to clean adjustment epoch 33.

| Dataset | Q | All Avg | Non-Orig Avg | Min Non-Orig | Known Avg | Unknown Avg | Overall <70 | Known <70 | Unknown <70 | Known <50 | Unknown <50 | Original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR | 100 | 91.00 | 90.64 | 65.50 | 88.64 | 92.64 | 2 | 2 | 1 | 0 | 0 | 100.00 |
| CIFAR | 200 | 90.85 | 90.48 | 67.50 | 88.08 | 92.88 | 2 | 2 | 1 | 0 | 0 | 100.00 |
| CIFAR | 500 | 91.32 | 90.97 | 70.70 | 88.38 | 93.56 | 0 | 2 | 0 | 0 | 0 | 100.00 |
| CIFAR | 1000 | 91.39 | 91.04 | 69.30 | 88.48 | 93.60 | 1 | 3 | 0 | 0 | 0 | 100.00 |
| Flickr | 100 | 89.58 | 89.16 | 63.00 | 88.08 | 90.24 | 2 | 1 | 2 | 0 | 0 | 100.00 |
| Flickr | 200 | 88.86 | 88.41 | 59.75 | 87.98 | 88.84 | 2 | 2 | 3 | 0 | 0 | 100.00 |
| Flickr | 500 | 88.25 | 87.78 | 59.90 | 88.26 | 87.30 | 2 | 3 | 3 | 0 | 0 | 100.00 |
| Flickr | 1000 | 88.21 | 87.73 | 59.40 | 87.60 | 87.87 | 2 | 3 | 2 | 0 | 0 | 100.00 |
| UCID | 100 | 86.46 | 85.92 | 63.00 | 87.24 | 84.60 | 3 | 1 | 5 | 0 | 0 | 100.00 |
| UCID | 200 | 86.47 | 85.93 | 63.50 | 86.96 | 84.90 | 3 | 1 | 5 | 0 | 0 | 100.00 |
| UCID | 500 | 85.36 | 84.77 | 63.10 | 86.20 | 83.34 | 5 | 1 | 9 | 0 | 0 | 100.00 |
| UCID | 850 | 84.93 | 84.33 | 62.12 | 85.80 | 82.85 | 5 | 2 | 9 | 0 | 0 | 100.00 |
| Amazon | 100 | 82.10 | 81.38 | 66.00 | 80.76 | 82.00 | 1 | 2 | 1 | 0 | 0 | 100.00 |
| Amazon | 200 | 83.16 | 82.49 | 67.50 | 82.08 | 82.90 | 1 | 2 | 0 | 0 | 0 | 100.00 |
| Amazon | 500 | 83.59 | 82.94 | 67.90 | 83.49 | 82.38 | 1 | 1 | 0 | 0 | 0 | 100.00 |
| Amazon | 1000 | 84.30 | 83.67 | 67.40 | 82.70 | 84.63 | 1 | 1 | 0 | 0 | 0 | 100.00 |
| Amazon | 2000 | 83.94 | 83.29 | 66.80 | 82.47 | 84.11 | 1 | 2 | 0 | 0 | 0 | 100.00 |

## Soft / Medium / Hard Performance Summary
This table uses the q100 evaluation for each dataset so all four datasets can be compared on the same query size.

| Dataset | Original | Soft Avg | Medium Avg | Hard Avg | Strongest group | Weakest group | Best attack in weak group | Weakest attack |
|---|---:|---:|---:|---:|---|---|---|---|
| CIFAR | 100.00 | 95.28 | 87.65 | 88.67 | Soft | Medium | rotate30 (100.00) | flip_v (65.50) |
| Flickr | 100.00 | 91.56 | 89.45 | 85.08 | Soft | Hard | mix_rotate10_bright (100.00) | crop40 (63.00) |
| UCID | 100.00 | 90.72 | 83.80 | 82.25 | Soft | Hard | mix_rotate10_bright (100.00) | crop40 (63.00) |
| Amazon | 100.00 | 83.67 | 79.60 | 80.92 | Soft | Medium | rotate30 (100.00) | flip_v (72.00) |

## Dataset Behaviour By Attack Type
- CIFAR performs best overall, especially on soft attacks and most medium attacks; the remaining weakness is mainly severe crop/flip tail behaviour.
- Flickr is very strong on soft attacks such as mild rotation, flips, brightness, resize, blur, and watermark, but harder crop/background attacks remain the main weak area.
- UCID reaches the target overall range after repair, with strong original/resize/blur/mild-rotation behaviour; the difficult side is still severe crop, contrast/brightness, and vertical flip tails.
- Amazon is strongest on original-like, crop5/crop20, rotate30, mixed, resize, and blur behaviour after the final refill; the weak side is known-side performance for heavy visual transformations such as flips, severe crop, brightness, and watermark.

## CIFAR Per-Attack Tables
Threshold source: latest threshold-search thresholds. The threshold shown is the one used for that attack and query setting.

### CIFAR Q100
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.106488 |
| rotate5 | 100.00 | 100.00 | 100.00 | 0.506597 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.106488 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.106488 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.106488 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.106488 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.106488 |
| bright_light | 100.00 | 99.00 | 99.50 | 0.569462 |
| rotate_m5 | 99.00 | 100.00 | 99.50 | 0.478065 |
| contrast | 99.00 | 100.00 | 99.50 | 0.639837 |
| rotate15 | 98.00 | 97.00 | 97.50 | 0.711798 |
| rotate_m10 | 97.00 | 95.00 | 96.00 | 0.674946 |
| bg_color_change | 93.00 | 99.00 | 96.00 | 0.614767 |
| blur | 89.00 | 94.00 | 91.50 | 0.685089 |
| resize_compress | 87.00 | 94.00 | 90.50 | 0.683731 |
| resize | 87.00 | 94.00 | 90.50 | 0.683731 |
| heavy_bright | 88.00 | 89.00 | 88.50 | 0.779218 |
| flip_h | 84.00 | 92.00 | 88.00 | 0.773993 |
| flip | 84.00 | 92.00 | 88.00 | 0.773993 |
| bright | 84.00 | 89.00 | 86.50 | 0.762759 |
| crop10 | 81.00 | 85.00 | 83.00 | 0.817925 |
| watermark_asset | 72.00 | 88.00 | 80.00 | 0.824005 |
| watermark_text | 75.00 | 82.00 | 78.50 | 0.866133 |
| rotate40 | 76.00 | 80.00 | 78.00 | 0.850716 |
| crop40 | 59.00 | 80.00 | 69.50 | 0.798677 |
| flip_v | 64.00 | 67.00 | 65.50 | 0.890727 |

### CIFAR Q200
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.106488 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.106488 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.106488 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.106488 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.106488 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.106488 |
| rotate5 | 100.00 | 99.50 | 99.75 | 0.506597 |
| rotate_m5 | 99.50 | 99.50 | 99.50 | 0.478065 |
| contrast | 99.00 | 99.50 | 99.25 | 0.639837 |
| bright_light | 99.00 | 99.00 | 99.00 | 0.569462 |
| rotate_m10 | 98.50 | 97.00 | 97.75 | 0.674946 |
| rotate15 | 96.00 | 97.00 | 96.50 | 0.711798 |
| bg_color_change | 92.50 | 98.50 | 95.50 | 0.614767 |
| blur | 91.00 | 96.50 | 93.75 | 0.685089 |
| resize_compress | 90.00 | 96.50 | 93.25 | 0.683731 |
| resize | 90.00 | 96.50 | 93.25 | 0.683731 |
| heavy_bright | 82.50 | 90.00 | 86.25 | 0.779218 |
| flip_h | 79.00 | 91.50 | 85.25 | 0.773993 |
| flip | 79.00 | 91.50 | 85.25 | 0.773993 |
| bright | 76.50 | 89.50 | 83.00 | 0.762759 |
| crop10 | 83.00 | 82.00 | 82.50 | 0.817925 |
| watermark_asset | 72.00 | 89.50 | 80.75 | 0.824005 |
| watermark_text | 75.50 | 83.50 | 79.50 | 0.866133 |
| rotate40 | 73.00 | 76.50 | 74.75 | 0.850716 |
| crop40 | 59.00 | 80.50 | 69.75 | 0.798677 |
| flip_v | 67.00 | 68.00 | 67.50 | 0.890727 |

### CIFAR Q500
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.106488 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.106488 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.106488 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.106488 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.106488 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.106488 |
| rotate5 | 100.00 | 99.80 | 99.90 | 0.506597 |
| rotate_m5 | 99.80 | 99.80 | 99.80 | 0.478065 |
| bright_light | 99.00 | 99.00 | 99.00 | 0.569462 |
| contrast | 99.00 | 98.60 | 98.80 | 0.639837 |
| rotate_m10 | 98.60 | 97.40 | 98.00 | 0.674946 |
| bg_color_change | 91.60 | 98.40 | 95.00 | 0.614767 |
| rotate15 | 92.80 | 96.60 | 94.70 | 0.711798 |
| resize_compress | 92.00 | 96.20 | 94.10 | 0.683731 |
| resize | 92.00 | 96.20 | 94.10 | 0.683731 |
| blur | 92.20 | 95.40 | 93.80 | 0.685089 |
| heavy_bright | 84.40 | 89.00 | 86.70 | 0.779218 |
| flip_h | 79.20 | 91.60 | 85.40 | 0.773993 |
| flip | 79.20 | 91.60 | 85.40 | 0.773993 |
| bright | 79.60 | 89.60 | 84.60 | 0.762759 |
| crop10 | 82.40 | 86.00 | 84.20 | 0.817925 |
| watermark_asset | 72.80 | 92.20 | 82.50 | 0.824005 |
| watermark_text | 77.20 | 86.80 | 82.00 | 0.866133 |
| rotate40 | 69.80 | 78.20 | 74.00 | 0.850716 |
| flip_v | 70.00 | 73.00 | 71.50 | 0.890727 |
| crop40 | 57.80 | 83.60 | 70.70 | 0.798677 |

### CIFAR Q1000
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.106488 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.106488 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.106488 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.106488 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.106488 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.106488 |
| rotate5 | 99.90 | 99.70 | 99.80 | 0.506597 |
| rotate_m5 | 99.70 | 99.80 | 99.75 | 0.478065 |
| bright_light | 99.30 | 99.10 | 99.20 | 0.569462 |
| contrast | 98.40 | 98.40 | 98.40 | 0.639837 |
| rotate_m10 | 98.60 | 97.10 | 97.85 | 0.674946 |
| bg_color_change | 90.40 | 98.60 | 94.50 | 0.614767 |
| resize_compress | 92.50 | 96.20 | 94.35 | 0.683731 |
| resize | 92.50 | 96.20 | 94.35 | 0.683731 |
| rotate15 | 91.70 | 95.90 | 93.80 | 0.711798 |
| blur | 91.70 | 95.80 | 93.75 | 0.685089 |
| heavy_bright | 85.10 | 89.00 | 87.05 | 0.779218 |
| flip_h | 81.10 | 92.30 | 86.70 | 0.773993 |
| flip | 81.10 | 92.30 | 86.70 | 0.773993 |
| bright | 79.90 | 89.90 | 84.90 | 0.762759 |
| watermark_asset | 75.50 | 92.00 | 83.75 | 0.824005 |
| watermark_text | 80.40 | 87.00 | 83.70 | 0.866133 |
| crop10 | 80.80 | 85.50 | 83.15 | 0.817925 |
| rotate40 | 68.10 | 80.10 | 74.10 | 0.850716 |
| flip_v | 69.00 | 72.90 | 70.95 | 0.890727 |
| crop40 | 56.30 | 82.30 | 69.30 | 0.798677 |


## Flickr Per-Attack Tables
Threshold source: latest threshold-search thresholds. The threshold shown is the one used for that attack and query setting.

### Flickr Q100
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.080286 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.080286 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.080286 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.080286 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.080286 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.080286 |
| rotate_m5 | 100.00 | 98.00 | 99.00 | 0.325554 |
| rotate5 | 99.00 | 99.00 | 99.00 | 0.299346 |
| bright_light | 99.00 | 98.00 | 98.50 | 0.292701 |
| flip_h | 99.00 | 98.00 | 98.50 | 0.313700 |
| flip | 99.00 | 98.00 | 98.50 | 0.313700 |
| rotate_m10 | 96.00 | 98.00 | 97.00 | 0.370138 |
| resize | 94.00 | 96.00 | 95.00 | 0.343769 |
| resize_compress | 94.00 | 96.00 | 95.00 | 0.343769 |
| blur | 90.00 | 96.00 | 93.00 | 0.376601 |
| rotate15 | 89.00 | 97.00 | 93.00 | 0.381120 |
| bg_color_change | 78.00 | 100.00 | 89.00 | 0.160010 |
| heavy_bright | 86.00 | 77.00 | 81.50 | 0.525870 |
| watermark_text | 70.00 | 89.00 | 79.50 | 0.468387 |
| watermark_asset | 72.00 | 86.00 | 79.00 | 0.489604 |
| rotate40 | 71.00 | 83.00 | 77.00 | 0.503140 |
| bright | 78.00 | 73.00 | 75.50 | 0.522851 |
| flip_v | 76.00 | 75.00 | 75.50 | 0.547410 |
| crop10 | 73.00 | 74.00 | 73.50 | 0.548357 |
| contrast | 75.00 | 63.00 | 69.00 | 0.675062 |
| crop40 | 64.00 | 62.00 | 63.00 | 0.619182 |

### Flickr Q200
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.080286 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.080286 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.080286 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.080286 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.080286 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.080286 |
| rotate_m5 | 100.00 | 97.00 | 98.50 | 0.325554 |
| rotate5 | 99.00 | 97.50 | 98.25 | 0.299346 |
| flip_h | 98.00 | 97.00 | 97.50 | 0.313700 |
| flip | 98.00 | 97.00 | 97.50 | 0.313700 |
| bright_light | 97.00 | 98.00 | 97.50 | 0.292701 |
| rotate_m10 | 97.50 | 95.50 | 96.50 | 0.370138 |
| resize | 93.50 | 95.50 | 94.50 | 0.343769 |
| resize_compress | 93.50 | 95.50 | 94.50 | 0.343769 |
| blur | 92.50 | 94.50 | 93.50 | 0.376601 |
| rotate15 | 90.50 | 94.50 | 92.50 | 0.381120 |
| bg_color_change | 76.50 | 100.00 | 88.25 | 0.160010 |
| heavy_bright | 85.50 | 75.00 | 80.25 | 0.525870 |
| watermark_text | 71.00 | 85.50 | 78.25 | 0.468387 |
| rotate40 | 71.00 | 81.00 | 76.00 | 0.503140 |
| flip_v | 78.00 | 73.50 | 75.75 | 0.547410 |
| watermark_asset | 69.50 | 82.00 | 75.75 | 0.489604 |
| bright | 76.50 | 73.50 | 75.00 | 0.522851 |
| crop10 | 74.50 | 69.50 | 72.00 | 0.548357 |
| contrast | 75.00 | 62.00 | 68.50 | 0.675062 |
| crop40 | 62.50 | 57.00 | 59.75 | 0.619182 |

### Flickr Q500
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.080286 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.080286 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.080286 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.080286 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.080286 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.080286 |
| rotate5 | 99.00 | 97.00 | 98.00 | 0.299346 |
| rotate_m5 | 100.00 | 95.40 | 97.70 | 0.325554 |
| flip_h | 97.60 | 96.20 | 96.90 | 0.313700 |
| flip | 97.60 | 96.20 | 96.90 | 0.313700 |
| bright_light | 96.00 | 96.60 | 96.30 | 0.292701 |
| rotate_m10 | 98.00 | 93.80 | 95.90 | 0.370138 |
| resize | 95.60 | 93.60 | 94.60 | 0.343769 |
| resize_compress | 95.60 | 93.60 | 94.60 | 0.343769 |
| blur | 93.00 | 90.40 | 91.70 | 0.376601 |
| rotate15 | 91.00 | 92.40 | 91.70 | 0.381120 |
| bg_color_change | 78.40 | 100.00 | 89.20 | 0.160010 |
| heavy_bright | 85.20 | 72.60 | 78.90 | 0.525870 |
| watermark_asset | 69.80 | 82.00 | 75.90 | 0.489604 |
| watermark_text | 68.20 | 83.20 | 75.70 | 0.468387 |
| flip_v | 78.40 | 70.80 | 74.60 | 0.547410 |
| rotate40 | 71.20 | 78.00 | 74.60 | 0.503140 |
| bright | 73.20 | 71.00 | 72.10 | 0.522851 |
| crop10 | 76.60 | 67.40 | 72.00 | 0.548357 |
| contrast | 76.60 | 57.80 | 67.20 | 0.675062 |
| crop40 | 65.40 | 54.40 | 59.90 | 0.619182 |

### Flickr Q1000
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.080286 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.080286 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.080286 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.080286 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.080286 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.080286 |
| rotate5 | 99.10 | 97.50 | 98.30 | 0.299346 |
| rotate_m5 | 99.80 | 96.00 | 97.90 | 0.325554 |
| flip_h | 97.20 | 97.20 | 97.20 | 0.313700 |
| flip | 97.20 | 97.20 | 97.20 | 0.313700 |
| bright_light | 95.50 | 97.30 | 96.40 | 0.292701 |
| rotate_m10 | 97.60 | 93.50 | 95.55 | 0.370138 |
| resize | 96.30 | 94.00 | 95.15 | 0.343769 |
| resize_compress | 96.30 | 94.00 | 95.15 | 0.343769 |
| blur | 93.40 | 91.10 | 92.25 | 0.376601 |
| rotate15 | 90.70 | 93.00 | 91.85 | 0.381120 |
| bg_color_change | 78.10 | 100.00 | 89.05 | 0.160010 |
| heavy_bright | 82.80 | 73.40 | 78.10 | 0.525870 |
| watermark_asset | 69.20 | 82.50 | 75.85 | 0.489604 |
| watermark_text | 65.90 | 83.80 | 74.85 | 0.468387 |
| rotate40 | 70.10 | 76.90 | 73.50 | 0.503140 |
| crop10 | 75.80 | 70.80 | 73.30 | 0.548357 |
| flip_v | 74.20 | 71.30 | 72.75 | 0.547410 |
| bright | 71.90 | 72.90 | 72.40 | 0.522851 |
| contrast | 75.90 | 58.50 | 67.20 | 0.675062 |
| crop40 | 63.00 | 55.80 | 59.40 | 0.619182 |


## UCID Per-Attack Tables
Threshold source: latest threshold-search thresholds. The threshold shown is the one used for that attack and query setting.

### UCID Q100
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.022737 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.022737 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.022737 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.022737 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.022737 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.022737 |
| resize | 99.00 | 99.00 | 99.00 | 0.409282 |
| resize_compress | 99.00 | 99.00 | 99.00 | 0.409282 |
| blur | 99.00 | 99.00 | 99.00 | 0.424658 |
| rotate_m5 | 97.00 | 95.00 | 96.00 | 0.561844 |
| rotate5 | 95.00 | 96.00 | 95.50 | 0.553408 |
| rotate_m10 | 99.00 | 90.00 | 94.50 | 0.614057 |
| rotate15 | 92.00 | 85.00 | 88.50 | 0.639517 |
| bright_light | 86.00 | 91.00 | 88.50 | 0.605603 |
| bg_color_change | 75.00 | 100.00 | 87.50 | 0.022737 |
| flip_h | 86.00 | 71.00 | 78.50 | 0.703969 |
| flip | 86.00 | 71.00 | 78.50 | 0.703969 |
| watermark_text | 72.00 | 84.00 | 78.00 | 0.672612 |
| watermark_asset | 80.00 | 74.00 | 77.00 | 0.698662 |
| heavy_bright | 78.00 | 75.00 | 76.50 | 0.710136 |
| flip_v | 75.00 | 71.00 | 73.00 | 0.719032 |
| bright | 73.00 | 69.00 | 71.00 | 0.691498 |
| crop10 | 73.00 | 67.00 | 70.00 | 0.729430 |
| contrast | 79.00 | 58.00 | 68.50 | 0.801727 |
| rotate40 | 67.00 | 66.00 | 66.50 | 0.712357 |
| crop40 | 71.00 | 55.00 | 63.00 | 0.760499 |

### UCID Q200
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.022737 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.022737 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.022737 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.022737 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.022737 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.022737 |
| resize | 98.50 | 98.50 | 98.50 | 0.409282 |
| resize_compress | 98.50 | 98.50 | 98.50 | 0.409282 |
| blur | 95.50 | 98.00 | 96.75 | 0.424658 |
| rotate_m5 | 97.50 | 95.00 | 96.25 | 0.561844 |
| rotate5 | 95.00 | 96.00 | 95.50 | 0.553408 |
| rotate_m10 | 97.00 | 90.50 | 93.75 | 0.614057 |
| rotate15 | 91.00 | 84.50 | 87.75 | 0.639517 |
| bg_color_change | 75.50 | 100.00 | 87.75 | 0.022737 |
| bright_light | 80.50 | 92.00 | 86.25 | 0.605603 |
| flip_h | 87.50 | 74.00 | 80.75 | 0.703969 |
| flip | 87.50 | 74.00 | 80.75 | 0.703969 |
| watermark_text | 73.50 | 83.00 | 78.25 | 0.672612 |
| watermark_asset | 78.50 | 76.00 | 77.25 | 0.698662 |
| heavy_bright | 77.00 | 73.00 | 75.00 | 0.710136 |
| flip_v | 75.50 | 72.00 | 73.75 | 0.719032 |
| crop10 | 77.00 | 68.00 | 72.50 | 0.729430 |
| bright | 72.50 | 69.00 | 70.75 | 0.691498 |
| rotate40 | 68.00 | 67.50 | 67.75 | 0.712357 |
| contrast | 76.00 | 58.00 | 67.00 | 0.801727 |
| crop40 | 72.00 | 55.00 | 63.50 | 0.760499 |

### UCID Q500
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.022737 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.022737 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.022737 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.022737 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.022737 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.022737 |
| resize | 99.20 | 97.60 | 98.40 | 0.409282 |
| resize_compress | 99.20 | 97.60 | 98.40 | 0.409282 |
| blur | 95.40 | 97.40 | 96.40 | 0.424658 |
| rotate_m5 | 96.20 | 93.20 | 94.70 | 0.561844 |
| rotate5 | 95.20 | 94.20 | 94.70 | 0.553408 |
| rotate_m10 | 93.60 | 87.80 | 90.70 | 0.614057 |
| bg_color_change | 75.00 | 100.00 | 87.50 | 0.022737 |
| rotate15 | 89.80 | 82.20 | 86.00 | 0.639517 |
| bright_light | 81.20 | 89.60 | 85.40 | 0.605603 |
| watermark_text | 75.20 | 82.00 | 78.60 | 0.672612 |
| flip_h | 85.20 | 69.20 | 77.20 | 0.703969 |
| flip | 85.20 | 69.20 | 77.20 | 0.703969 |
| watermark_asset | 79.20 | 75.20 | 77.20 | 0.698662 |
| heavy_bright | 73.80 | 68.60 | 71.20 | 0.710136 |
| flip_v | 72.60 | 67.60 | 70.10 | 0.719032 |
| crop10 | 74.80 | 64.20 | 69.50 | 0.729430 |
| bright | 67.00 | 69.20 | 68.10 | 0.691498 |
| rotate40 | 71.00 | 64.60 | 67.80 | 0.712357 |
| contrast | 75.40 | 58.80 | 67.10 | 0.801727 |
| crop40 | 70.80 | 55.40 | 63.10 | 0.760499 |

### UCID Q850
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.022737 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.022737 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.022737 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.022737 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.022737 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.022737 |
| resize | 98.71 | 96.71 | 97.71 | 0.409282 |
| resize_compress | 98.71 | 96.71 | 97.71 | 0.409282 |
| blur | 95.18 | 96.71 | 95.94 | 0.424658 |
| rotate_m5 | 96.71 | 91.88 | 94.29 | 0.561844 |
| rotate5 | 95.18 | 92.71 | 93.94 | 0.553408 |
| rotate_m10 | 93.65 | 86.00 | 89.82 | 0.614057 |
| bg_color_change | 71.53 | 100.00 | 85.76 | 0.022737 |
| rotate15 | 88.82 | 82.24 | 85.53 | 0.639517 |
| bright_light | 79.06 | 89.88 | 84.47 | 0.605603 |
| watermark_text | 74.00 | 80.71 | 77.35 | 0.672612 |
| watermark_asset | 78.35 | 75.29 | 76.82 | 0.698662 |
| flip_h | 85.06 | 68.47 | 76.76 | 0.703969 |
| flip | 85.06 | 68.47 | 76.76 | 0.703969 |
| heavy_bright | 73.06 | 67.88 | 70.47 | 0.710136 |
| flip_v | 73.76 | 66.82 | 70.29 | 0.719032 |
| crop10 | 75.65 | 62.94 | 69.29 | 0.729430 |
| bright | 66.59 | 69.65 | 68.12 | 0.691498 |
| rotate40 | 71.76 | 64.00 | 67.88 | 0.712357 |
| contrast | 74.82 | 59.41 | 67.12 | 0.801727 |
| crop40 | 69.41 | 54.82 | 62.12 | 0.760499 |


## Amazon Per-Attack Tables
Threshold source: latest threshold-search thresholds. The threshold shown is the one used for that attack and query setting.

### Amazon Q100
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.164388 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.164388 |
| resize_compress | 89.00 | 90.00 | 89.50 | 0.551501 |
| resize | 89.00 | 90.00 | 89.50 | 0.551501 |
| blur | 84.00 | 87.00 | 85.50 | 0.593342 |
| contrast | 84.00 | 78.00 | 81.00 | 0.817205 |
| rotate_m5 | 81.00 | 79.00 | 80.00 | 0.834448 |
| rotate5 | 77.00 | 78.00 | 77.50 | 0.833328 |
| watermark_asset | 79.00 | 75.00 | 77.00 | 0.861454 |
| bright_light | 76.00 | 77.00 | 76.50 | 0.828583 |
| watermark_text | 76.00 | 77.00 | 76.50 | 0.878038 |
| rotate_m10 | 73.00 | 80.00 | 76.50 | 0.814989 |
| rotate15 | 75.00 | 76.00 | 75.50 | 0.836038 |
| rotate40 | 69.00 | 81.00 | 75.00 | 0.792213 |
| crop40 | 73.00 | 74.00 | 73.50 | 0.971613 |
| crop10 | 71.00 | 76.00 | 73.50 | 0.925268 |
| bright | 71.00 | 76.00 | 73.50 | 0.805097 |
| flip_h | 72.00 | 73.00 | 72.50 | 0.852509 |
| flip | 72.00 | 73.00 | 72.50 | 0.852509 |
| flip_v | 72.00 | 72.00 | 72.00 | 0.874023 |
| heavy_bright | 74.00 | 68.00 | 71.00 | 0.817806 |
| bg_color_change | 62.00 | 70.00 | 66.00 | 1.069988 |

### Amazon Q200
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.164388 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.164388 |
| resize_compress | 90.50 | 93.00 | 91.75 | 0.551501 |
| resize | 90.50 | 93.00 | 91.75 | 0.551501 |
| blur | 86.50 | 90.00 | 88.25 | 0.593342 |
| contrast | 84.50 | 79.50 | 82.00 | 0.817205 |
| rotate_m5 | 83.50 | 78.50 | 81.00 | 0.834448 |
| watermark_asset | 80.00 | 79.00 | 79.50 | 0.861454 |
| bright_light | 80.00 | 78.50 | 79.25 | 0.828583 |
| watermark_text | 79.00 | 77.50 | 78.25 | 0.878038 |
| rotate5 | 78.00 | 78.00 | 78.00 | 0.833328 |
| rotate_m10 | 76.50 | 79.50 | 78.00 | 0.814989 |
| rotate15 | 76.50 | 75.50 | 76.00 | 0.836038 |
| rotate40 | 69.50 | 81.00 | 75.25 | 0.792213 |
| flip_h | 75.00 | 75.00 | 75.00 | 0.852509 |
| flip | 75.00 | 75.00 | 75.00 | 0.852509 |
| crop10 | 73.00 | 75.50 | 74.25 | 0.925268 |
| flip_v | 73.50 | 72.50 | 73.00 | 0.874023 |
| crop40 | 72.50 | 73.50 | 73.00 | 0.971613 |
| bright | 71.50 | 74.50 | 73.00 | 0.805097 |
| heavy_bright | 74.00 | 71.00 | 72.50 | 0.817806 |
| bg_color_change | 62.50 | 72.50 | 67.50 | 1.069988 |

### Amazon Q500
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.164388 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.164388 |
| resize_compress | 92.60 | 92.40 | 92.50 | 0.551501 |
| resize | 92.60 | 92.40 | 92.50 | 0.551501 |
| blur | 89.20 | 90.60 | 89.90 | 0.593342 |
| contrast | 86.60 | 79.00 | 82.80 | 0.817205 |
| rotate_m5 | 84.00 | 77.40 | 80.70 | 0.834448 |
| bright_light | 81.60 | 78.00 | 79.80 | 0.828583 |
| rotate5 | 81.20 | 77.40 | 79.30 | 0.833328 |
| watermark_asset | 81.40 | 77.00 | 79.20 | 0.861454 |
| watermark_text | 81.40 | 76.20 | 78.80 | 0.878038 |
| rotate_m10 | 77.20 | 79.00 | 78.10 | 0.814989 |
| rotate15 | 77.00 | 75.00 | 76.00 | 0.836038 |
| flip_h | 77.00 | 74.00 | 75.50 | 0.852509 |
| flip | 77.00 | 74.00 | 75.50 | 0.852509 |
| flip_v | 76.60 | 73.20 | 74.90 | 0.874023 |
| rotate40 | 71.60 | 78.00 | 74.80 | 0.792213 |
| crop10 | 75.20 | 73.80 | 74.50 | 0.925268 |
| bright | 74.80 | 73.80 | 74.30 | 0.805097 |
| heavy_bright | 74.80 | 72.80 | 73.80 | 0.817806 |
| crop40 | 74.40 | 70.80 | 72.60 | 0.971613 |
| bg_color_change | 61.00 | 74.80 | 67.90 | 1.069988 |

### Amazon Q1000
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.164388 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.164388 |
| resize_compress | 93.40 | 93.60 | 93.50 | 0.551501 |
| resize | 93.40 | 93.60 | 93.50 | 0.551501 |
| blur | 90.20 | 91.70 | 90.95 | 0.593342 |
| contrast | 86.50 | 83.00 | 84.75 | 0.817205 |
| rotate_m5 | 82.70 | 80.30 | 81.50 | 0.834448 |
| bright_light | 80.60 | 81.20 | 80.90 | 0.828583 |
| rotate5 | 80.40 | 80.00 | 80.20 | 0.833328 |
| watermark_text | 80.70 | 78.90 | 79.80 | 0.878038 |
| watermark_asset | 79.30 | 79.80 | 79.55 | 0.861454 |
| rotate_m10 | 75.90 | 82.30 | 79.10 | 0.814989 |
| rotate15 | 75.90 | 78.60 | 77.25 | 0.836038 |
| flip_h | 77.10 | 77.10 | 77.10 | 0.852509 |
| flip | 77.10 | 77.10 | 77.10 | 0.852509 |
| flip_v | 75.20 | 76.50 | 75.85 | 0.874023 |
| rotate40 | 70.20 | 81.10 | 75.65 | 0.792213 |
| crop10 | 72.10 | 78.50 | 75.30 | 0.925268 |
| bright | 72.40 | 76.90 | 74.65 | 0.805097 |
| heavy_bright | 72.40 | 76.90 | 74.65 | 0.817806 |
| crop40 | 73.00 | 73.00 | 73.00 | 0.971613 |
| bg_color_change | 59.10 | 75.70 | 67.40 | 1.069988 |

### Amazon Q2000
| Attack | Known | Unknown | Total | Threshold |
|---|---:|---:|---:|---:|
| original | 100.00 | 100.00 | 100.00 | 0.164388 |
| rotate30 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop5 | 100.00 | 100.00 | 100.00 | 0.164388 |
| crop20 | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_rotate10_bright | 100.00 | 100.00 | 100.00 | 0.164388 |
| mix_crop20_watermark | 100.00 | 100.00 | 100.00 | 0.164388 |
| resize_compress | 92.40 | 93.20 | 92.80 | 0.551501 |
| resize | 92.40 | 93.20 | 92.80 | 0.551501 |
| blur | 88.85 | 90.80 | 89.83 | 0.593342 |
| contrast | 85.90 | 82.55 | 84.23 | 0.817205 |
| rotate_m5 | 82.65 | 79.50 | 81.08 | 0.834448 |
| bright_light | 79.80 | 80.70 | 80.25 | 0.828583 |
| rotate5 | 80.15 | 79.35 | 79.75 | 0.833328 |
| watermark_asset | 80.10 | 78.95 | 79.52 | 0.861454 |
| watermark_text | 80.65 | 78.05 | 79.35 | 0.878038 |
| rotate_m10 | 76.10 | 81.10 | 78.60 | 0.814989 |
| rotate15 | 75.40 | 77.35 | 76.38 | 0.836038 |
| flip_h | 75.70 | 76.85 | 76.28 | 0.852509 |
| flip | 75.70 | 76.85 | 76.28 | 0.852509 |
| flip_v | 74.80 | 76.05 | 75.43 | 0.874023 |
| crop10 | 72.70 | 77.90 | 75.30 | 0.925268 |
| rotate40 | 69.90 | 80.65 | 75.28 | 0.792213 |
| heavy_bright | 73.30 | 76.20 | 74.75 | 0.817806 |
| bright | 72.95 | 76.40 | 74.67 | 0.805097 |
| crop40 | 73.25 | 72.70 | 72.97 | 0.971613 |
| bg_color_change | 59.10 | 74.50 | 66.80 | 1.069988 |

## Compact Combined Accuracy Tables
These tables show only combined total accuracy per attack. They are easier to paste into a paper because `Known` and `Unknown` are not split here.

### CIFAR Combined Accuracy
Threshold source: fixed calibrated thresholds.

| Attack | Threshold | Q100 | Q200 | Q500 | Q1000 |
|---|---:|---:|---:|---:|---:|
| original | 0.106488 | 100.00 | 100.00 | 100.00 | 100.00 |
| crop5 | 0.106488 | 100.00 | 100.00 | 100.00 | 100.00 |
| rotate30 | 0.106488 | 100.00 | 100.00 | 100.00 | 100.00 |
| crop20 | 0.106488 | 100.00 | 100.00 | 100.00 | 100.00 |
| mix_rotate10_bright | 0.106488 | 100.00 | 100.00 | 100.00 | 100.00 |
| mix_crop20_watermark | 0.106488 | 100.00 | 100.00 | 100.00 | 100.00 |
| rotate5 | 0.506597 | 100.00 | 99.75 | 99.90 | 99.80 |
| rotate_m5 | 0.478065 | 99.50 | 99.50 | 99.80 | 99.75 |
| bright_light | 0.569462 | 99.50 | 99.00 | 99.00 | 99.20 |
| contrast | 0.639837 | 99.50 | 99.25 | 98.80 | 98.40 |
| rotate_m10 | 0.674946 | 96.00 | 97.75 | 98.00 | 97.85 |
| rotate15 | 0.711798 | 97.50 | 96.50 | 94.70 | 93.80 |
| bg_color_change | 0.614767 | 96.00 | 95.50 | 95.00 | 94.50 |
| blur | 0.685089 | 91.50 | 93.75 | 93.80 | 93.75 |
| resize | 0.683731 | 90.50 | 93.25 | 94.10 | 94.35 |
| resize_compress | 0.683731 | 90.50 | 93.25 | 94.10 | 94.35 |
| heavy_bright | 0.779218 | 88.50 | 86.25 | 86.70 | 87.05 |
| flip_h | 0.773993 | 88.00 | 85.25 | 85.40 | 86.70 |
| flip | 0.773993 | 88.00 | 85.25 | 85.40 | 86.70 |
| bright | 0.762759 | 86.50 | 83.00 | 84.60 | 84.90 |
| crop10 | 0.817925 | 83.00 | 82.50 | 84.20 | 83.15 |
| watermark_asset | 0.824005 | 80.00 | 80.75 | 82.50 | 83.75 |
| watermark_text | 0.866133 | 78.50 | 79.50 | 82.00 | 83.70 |
| rotate40 | 0.850716 | 78.00 | 74.75 | 74.00 | 74.10 |
| crop40 | 0.798677 | 69.50 | 69.75 | 70.70 | 69.30 |
| flip_v | 0.890727 | 65.50 | 67.50 | 71.50 | 70.95 |

### Flickr Combined Accuracy
Threshold source: accepted balanced multi-query calibration.

| Attack | Threshold | Q100 | Q200 | Q500 | Q1000 |
|---|---:|---:|---:|---:|---:|
| original | 0.080286 | 100.00 | 100.00 | 100.00 | 100.00 |
| crop5 | 0.080286 | 100.00 | 100.00 | 100.00 | 100.00 |
| rotate30 | 0.080286 | 100.00 | 100.00 | 100.00 | 100.00 |
| crop20 | 0.080286 | 100.00 | 100.00 | 100.00 | 100.00 |
| mix_rotate10_bright | 0.080286 | 100.00 | 100.00 | 100.00 | 100.00 |
| mix_crop20_watermark | 0.080286 | 100.00 | 100.00 | 100.00 | 100.00 |
| rotate5 | 0.299346 | 99.00 | 98.25 | 98.00 | 98.30 |
| rotate_m5 | 0.325554 | 99.00 | 98.50 | 97.70 | 97.90 |
| flip_h | 0.313700 | 98.50 | 97.50 | 96.90 | 97.20 |
| flip | 0.313700 | 98.50 | 97.50 | 96.90 | 97.20 |
| bright_light | 0.292701 | 98.50 | 97.50 | 96.30 | 96.40 |
| rotate_m10 | 0.370138 | 97.00 | 96.50 | 95.90 | 95.55 |
| resize | 0.343769 | 95.00 | 94.50 | 94.60 | 95.15 |
| resize_compress | 0.343769 | 95.00 | 94.50 | 94.60 | 95.15 |
| blur | 0.376601 | 93.00 | 93.50 | 91.70 | 92.25 |
| rotate15 | 0.381120 | 93.00 | 92.50 | 91.70 | 91.85 |
| bg_color_change | 0.160010 | 89.00 | 88.25 | 89.20 | 89.05 |
| heavy_bright | 0.525870 | 81.50 | 80.25 | 78.90 | 78.10 |
| watermark_text | 0.468387 | 79.50 | 78.25 | 75.70 | 74.85 |
| watermark_asset | 0.489604 | 79.00 | 75.75 | 75.90 | 75.85 |
| rotate40 | 0.503140 | 77.00 | 76.00 | 74.60 | 73.50 |
| flip_v | 0.547410 | 75.50 | 75.75 | 74.60 | 72.75 |
| bright | 0.522851 | 75.50 | 75.00 | 72.10 | 72.40 |
| crop10 | 0.548357 | 73.50 | 72.00 | 72.00 | 73.30 |
| contrast | 0.675062 | 69.00 | 68.50 | 67.20 | 67.20 |
| crop40 | 0.619182 | 63.00 | 59.75 | 59.90 | 59.40 |

### UCID Combined Accuracy
Threshold source: accepted balanced multi-query calibration.

| Attack | Threshold | Q100 | Q200 | Q500 | Q850 |
|---|---:|---:|---:|---:|---:|
| original | 0.022737 | 100.00 | 100.00 | 100.00 | 100.00 |
| crop5 | 0.022737 | 100.00 | 100.00 | 100.00 | 100.00 |
| rotate30 | 0.022737 | 100.00 | 100.00 | 100.00 | 100.00 |
| crop20 | 0.022737 | 100.00 | 100.00 | 100.00 | 100.00 |
| mix_rotate10_bright | 0.022737 | 100.00 | 100.00 | 100.00 | 100.00 |
| mix_crop20_watermark | 0.022737 | 100.00 | 100.00 | 100.00 | 100.00 |
| resize | 0.409282 | 99.00 | 98.50 | 98.40 | 97.71 |
| resize_compress | 0.409282 | 99.00 | 98.50 | 98.40 | 97.71 |
| blur | 0.424658 | 99.00 | 96.75 | 96.40 | 95.94 |
| rotate_m5 | 0.561844 | 96.00 | 96.25 | 94.70 | 94.29 |
| rotate5 | 0.553408 | 95.50 | 95.50 | 94.70 | 93.94 |
| rotate_m10 | 0.614057 | 94.50 | 93.75 | 90.70 | 89.82 |
| bg_color_change | 0.022737 | 87.50 | 87.75 | 87.50 | 85.76 |
| rotate15 | 0.639517 | 88.50 | 87.75 | 86.00 | 85.53 |
| bright_light | 0.605603 | 88.50 | 86.25 | 85.40 | 84.47 |
| flip_h | 0.703969 | 78.50 | 80.75 | 77.20 | 76.76 |
| flip | 0.703969 | 78.50 | 80.75 | 77.20 | 76.76 |
| watermark_text | 0.672612 | 78.00 | 78.25 | 78.60 | 77.35 |
| watermark_asset | 0.698662 | 77.00 | 77.25 | 77.20 | 76.82 |
| heavy_bright | 0.710136 | 76.50 | 75.00 | 71.20 | 70.47 |
| flip_v | 0.719032 | 73.00 | 73.75 | 70.10 | 70.29 |
| crop10 | 0.729430 | 70.00 | 72.50 | 69.50 | 69.29 |
| bright | 0.691498 | 71.00 | 70.75 | 68.10 | 68.12 |
| rotate40 | 0.712357 | 66.50 | 67.75 | 67.80 | 67.88 |
| contrast | 0.801727 | 68.50 | 67.00 | 67.10 | 67.12 |
| crop40 | 0.760499 | 63.00 | 63.50 | 63.10 | 62.12 |

### Amazon Combined Accuracy
Threshold source: latest threshold-search thresholds.

| Attack | Threshold | Q100 | Q200 | Q500 | Q1000 | Q2000 |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.164388 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| crop5 | 0.164388 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| rotate30 | 0.164388 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| crop20 | 0.164388 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| mix_rotate10_bright | 0.164388 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| mix_crop20_watermark | 0.164388 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| resize | 0.551501 | 89.50 | 91.75 | 92.50 | 93.50 | 92.80 |
| resize_compress | 0.551501 | 89.50 | 91.75 | 92.50 | 93.50 | 92.80 |
| blur | 0.593342 | 85.50 | 88.25 | 89.90 | 90.95 | 89.83 |
| contrast | 0.817205 | 81.00 | 82.00 | 82.80 | 84.75 | 84.23 |
| rotate_m5 | 0.834448 | 80.00 | 81.00 | 80.70 | 81.50 | 81.08 |
| bright_light | 0.828583 | 76.50 | 79.25 | 79.80 | 80.90 | 80.25 |
| watermark_asset | 0.861454 | 77.00 | 79.50 | 79.20 | 79.55 | 79.52 |
| rotate5 | 0.833328 | 77.50 | 78.00 | 79.30 | 80.20 | 79.75 |
| watermark_text | 0.878038 | 76.50 | 78.25 | 78.80 | 79.80 | 79.35 |
| rotate_m10 | 0.814989 | 76.50 | 78.00 | 78.10 | 79.10 | 78.60 |
| rotate15 | 0.836038 | 75.50 | 76.00 | 76.00 | 77.25 | 76.38 |
| flip_h | 0.852509 | 72.50 | 75.00 | 75.50 | 77.10 | 76.28 |
| flip | 0.852509 | 72.50 | 75.00 | 75.50 | 77.10 | 76.28 |
| rotate40 | 0.792213 | 75.00 | 75.25 | 74.80 | 75.65 | 75.28 |
| crop10 | 0.925268 | 73.50 | 74.25 | 74.50 | 75.30 | 75.30 |
| flip_v | 0.874023 | 72.00 | 73.00 | 74.90 | 75.85 | 75.43 |
| bright | 0.805097 | 73.50 | 73.00 | 74.30 | 74.65 | 74.67 |
| heavy_bright | 0.817806 | 71.00 | 72.50 | 73.80 | 74.65 | 74.75 |
| crop40 | 0.971613 | 73.50 | 73.00 | 72.60 | 73.00 | 72.97 |
| bg_color_change | 1.069988 | 66.00 | 67.50 | 67.90 | 67.40 | 66.80 |

## Key Final Files
| Dataset | Checkpoint | Store JSON | Main eval JSON | Threshold-search JSON |
|---|---|---|---|---|
| CIFAR | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/cifar/cifar-epoch-015-training.pt` | `/home/saketh/json/final_named_artifacts_20260601/cifar/cifar_store.json` | `/home/saketh/json/final_named_artifacts_20260601/cifar/cifar_multiq_eval.json` |  |
| Flickr | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/flickr/flickr-epoch-026-adjustment.pt` | `/home/saketh/json/final_named_artifacts_20260601/flickr/flickr_store.json` | `/home/saketh/json/final_named_artifacts_20260601/flickr/flickr_multiq_eval.json` |  |
| UCID | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/ucid/ucid-epoch-037-adjustment.pt` | `/home/saketh/json/final_named_artifacts_20260601/ucid/ucid_store.json` | `/home/saketh/json/final_named_artifacts_20260601/ucid/ucid_multiq_eval.json` | `/home/saketh/json/final_named_artifacts_20260601/ucid/ucid_threshold_search.json` |
| Amazon | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/amazon/amazon-epoch-033-adjustment.pt` | `/home/saketh/json/final_named_artifacts_20260601/amazon/amazon_store.json` | `/home/saketh/json/final_named_artifacts_20260601/amazon/amazon_multiq_eval.json` | `/home/saketh/json/final_named_artifacts_20260601/amazon/amazon_threshold_search.json` |

## Eval JSON Paths
- CIFAR fixed multi-query: `/home/saketh/json/final_named_artifacts_20260601/cifar/cifar_multiq_eval.json`
- Flickr accepted balanced multi-query: `/home/saketh/json/final_named_artifacts_20260601/flickr/flickr_multiq_eval.json`
- UCID accepted balanced multi-query: `/home/saketh/json/final_named_artifacts_20260601/ucid/ucid_multiq_eval.json`
- Amazon selected `final_total` multi-query: `/home/saketh/json/final_named_artifacts_20260601/amazon/amazon_multiq_eval.json`
- Amazon fixed multi-query audit: `/home/saketh/json/final_named_artifacts_20260601/amazon/amazon_fixed_multiq_eval.json`
- UCID threshold search: `/home/saketh/json/final_named_artifacts_20260601/ucid/ucid_threshold_search.json`
- Amazon threshold search: `/home/saketh/json/final_named_artifacts_20260601/amazon/amazon_threshold_search.json`
