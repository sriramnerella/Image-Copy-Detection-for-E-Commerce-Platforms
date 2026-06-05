Below are **generalized architecture diagrams** you can use in the report/presentation. These are not tied to one image; they explain how the two search structures work.

# HNSW: Hierarchical Navigable Small World

HNSW behaves like a **layered shortcut graph**. It often looks tree-like during a single query because the query follows one greedy path downward, but structurally each layer is still a graph.

```text
QUERY IMAGE
attack/original variant
        |
        v
ConvNeXt embedding vector q
        |
        v

====================================================================
                        HNSW SEARCH STRUCTURE
====================================================================

                         LAYER 3
                 very sparse global shortcuts

               +-------------------------+
               | C12: broad visual hub   |
               | e.g., mixed object zone |
               +-------------------------+
                  /          |          \
                 /           |           \
                v            v            v
        +------------+  +------------+  +------------+
        | C03        |  | C44        |  | C87        |
        | animal hub |  | vehicle hub|  | product hub|
        +------------+  +------------+  +------------+

Query compares with top-layer candidates.
Chooses centroid closest to q.
Example route chooses C44.


                         LAYER 2
                 coarse category neighborhood

                         +------------+
                         | C44        |
                         | vehicle hub|
                         +------------+
                           /    |    \
                          /     |     \
                         v      v      v
              +-------------+ +-------------+ +-------------+
              | C18         | | C51         | | C72         |
              | airplane    | | ship/truck  | | bird/sky    |
              | region      | | region      | | bridge      |
              +-------------+ +-------------+ +-------------+

Query checks neighbors of C44.
C18 is closest, so route moves to C18.


                         LAYER 1
                  medium visual neighborhood

                         +-------------+
                         | C18         |
                         | airplane    |
                         | region      |
                         +-------------+
                           /    |    \
                          /     |     \
                         v      v      v
              +-------------+ +-------------+ +-------------+
              | C09         | | C27         | | C38         |
              | airplane    | | airplane    | | ship-like   |
              | side views  | | front views | | bridge zone |
              +-------------+ +-------------+ +-------------+

Query checks local neighbors.
C27 is closest, so route moves to C27.


                         LAYER 0
                    dense local centroid graph

                         +-------------+
                         | C27         |
                         | airplane    |
                         | front views |
                         +-------------+
                         /   |    |    \
                        /    |    |     \
                       v     v    v      v
        +----------------+ +----------------+ +----------------+
        | K0000          | | K0342          | | K1290          |
        | airplane_orig  | | airplane_aug   | | airplane_dup   |
        +----------------+ +----------------+ +----------------+

Final candidate search happens inside / near this centroid region.
Then distance(q, candidate) is compared to threshold.
```

## HNSW Query Flow

```text
q
|
v
enter top sparse layer
|
v
greedy move to closest neighbor
|
v
when no closer neighbor exists, drop down one layer
|
v
repeat until bottom layer
|
v
search final local candidate neighborhood
|
v
distance <= threshold ? duplicate : not duplicate
```

## HNSW Intuition

```text
Top layers      = fast long-range navigation
Middle layers   = category-level routing
Bottom layer    = dense local duplicate neighborhood
```

It avoids scanning the full known bank:

```text
Direct:
q -> compare all known embeddings -> decision

HNSW:
q -> few top centroids -> few middle centroids -> local cluster -> decision
```

---

# ERNG: Exponential Rank Navigation Graph

ERNG is a **flat multi-probe graph**. It does not use strict layers. Each centroid connects to:

```text
local KNN neighbors
+
exponential-rank skip neighbors: rank 1, 2, 4, 8, 16, ...
```

This gives it both local precision and long jumps.

```text
QUERY IMAGE
attack/original variant
        |
        v
ConvNeXt embedding vector q
        |
        v

====================================================================
              ERNG: EXPONENTIAL RANK NAVIGATION GRAPH
====================================================================

                    multiple probe entries

             Probe A             Probe B             Probe C
           random/central       random/central       random/central
                |                    |                    |
                v                    v                    v

          +-----------+        +-----------+        +-----------+
          | C05       |        | C44       |        | C91       |
          | broad hub |        | product   |        | animal    |
          +-----------+        +-----------+        +-----------+
             / | \                / | \                / | \
            /  |  \              /  |  \              /  |  \
           v   v   v            v   v   v            v   v   v

       local edges + exponential skip edges from every centroid

          C05 -------------------------------> C72
           | \                                long skip, rank 16
           |  \-----> C33
           |          medium skip, rank 4
           v
          C12
       local neighbor

          C44 -------------------------------> C18
           | \                                long skip
           |  \-----> C27
           |          medium skip
           v
          C51
       local neighbor

          C91 -------------------------------> C07
           | \                                long skip
           |  \-----> C38
           v
          C83


Each probe greedily walks to a closer centroid.
```

## ERNG Probe Search

```text
Probe A path:

C05
 |
 | checks local neighbors and exponential skips
 v
C33
 |
 | closer to q
 v
C18
 |
 | closer to q
 v
C27
 |
 | no neighbor closer
 v
stop at C27


Probe B path:

C44
 |
 v
C51
 |
 v
C27
 |
 v
stop at C27


Probe C path:

C91
 |
 v
C38
 |
 v
C72
 |
 v
stop at C72
```

Then ERNG selects the best final centroid among probes:

```text
Probe A final: C27, distance 0.31
Probe B final: C27, distance 0.31
Probe C final: C72, distance 0.58

Selected final centroid = C27
```

Then candidate search happens inside that centroid region:

```text
C27 local region

+---------------------+
| airplane_orig       |
+---------------------+
| airplane_rotated    |
+---------------------+
| airplane_cropped    |
+---------------------+

distance(q, candidate) <= threshold
        |
        v
duplicate detected
```

## ERNG General Diagram

```text
                         ERNG FLAT GRAPH

             exponential skip edges are long arrows
             local KNN edges are short arrows

          +------+       +------+       +------+
          | C01  |------>| C02  |------>| C03  |
          +------+       +------+       +------+
             |              |              |
             |              |              |
             v              v              v
          +------+       +------+       +------+
          | C10  |------>| C11  |------>| C12  |
          +------+       +------+       +------+
             |  \           |  \           |  \
             |   \          |   \          |   \
             |    \         |    \         |    \
             v     v        v     v        v     v
          +------+ +------+ +------+ +------+ +------+
          | C20  | | C32  | | C21  | | C40  | | C55  |
          +------+ +------+ +------+ +------+ +------+

          short edge: nearest neighbor
          medium edge: rank 4 / rank 8
          long edge: rank 16 / rank 32
```

## ERNG Query Flow

```text
q
|
v
start R probes from several centroids
|
v
each probe greedily checks local + exponential-rank neighbors
|
v
each probe stops at a local minimum
|
v
choose best final centroid among probes
|
v
search candidate images in that centroid region
|
v
distance <= threshold ? duplicate : not duplicate
```

---

# Side-by-Side Difference

```text
HNSW
----
Layered structure.

q
|
v
top sparse graph
|
v
middle graph
|
v
bottom dense graph
|
v
candidate neighborhood
|
v
decision


ERNG
----
Flat multi-probe graph.

q
|------------------|
|        |         |
v        v         v
probe A  probe B   probe C
|        |         |
v        v         v
greedy walks with local + exponential skip edges
|        |         |
v        v         v
final A  final B   final C
        |
        v
choose best final centroid
        |
        v
candidate neighborhood
        |
        v
decision
```

# Why They Can Reach The Same Decision

Both use the same final rule:

```text
distance(query, matched known image) <= calibrated threshold
```

So if HNSW and ERNG route into a region where they find a candidate with the same duplicate/not-duplicate outcome as direct search, the accuracy remains the same.

```text
Direct:
scan all known images

HNSW / ERNG:
route to small candidate region
scan only that small region
```

That is why speed improves while duplicate decision stays same.


Yes. For the actual **query/evaluation runs**, the attack variants were applied properly for all Q sizes and all datasets.

Important distinction:

- The **evaluation/query code** used the project’s real attack pipeline during embedding/eval.
- The issue was only in the **separate visual example image export folder**, where some attacks were not visibly generated because the export helper had fall-through aliases.
- I fixed that visual export folder separately.

For the real all-Q tables:

```text
cifar_allq_per_attack_direct_hnsw_erng.json
flickr_allq_per_attack_direct_hnsw_erng.json
ucid_allq_per_attack_direct_hnsw_erng.json
amazon_allq_per_attack_direct_hnsw_erng.json
```

each Q uses all 26 attacks:

```text
original
rotate_m5
rotate_m10
rotate5
rotate15
rotate30
rotate40
crop5
crop10
crop20
crop40
bright_light
bright
heavy_bright
flip_h
flip_v
flip
resize_compress
resize
watermark_text
watermark_asset
mix_rotate10_bright
mix_crop20_watermark
bg_color_change
blur
contrast
```

And the graph decision comparison is:

```text
Direct query with attack variant
HNSW query with same attack variant
ERNG query with same attack variant
```

So yes: for all Q sizes, for each dataset, the attack variants were applied during querying/evaluation.


Confirmed: for the actual querying/evaluation tables, there is no evidence of that visual-export mistake.

The real all-Q evaluation used the evaluation pipeline and produced valid 26-attack rows for every dataset/Q. HNSW and ERNG were compared against Direct on those same attacked query embeddings, and the duplicate/not-duplicate match is reported as `100.00`.

The mistake was only in the separate exported image-example folder, and that has now been regenerated and validated.


Yes. These are the cases we actually observed from the final traces/results.

**Case 1: Original Query Reaches Exact Original**
Example from CIFAR:

```text
QUERY:
airplane_train_29944_org.png + original

HNSW route:
C0048 airplane region
   -> C0378 airplane region
   -> C0271 airplane region

Final centroid:
C0271 / airplane_pool region

Final match:
airplane_train_29944_org.png

Distance:
0.000006 <= threshold 0.106488

Decision:
DUPLICATE
```

Meaning:

```text
original image -> airplane centroid neighborhood -> same original image -> duplicate
```

**Case 2: Weak Attack Reaches Exact Original**
Example from CIFAR:

```text
QUERY:
airplane_train_29944_org.png + rotate_m5

Final centroid region:
airplane_pool region

Final match:
airplane_train_29944_org.png

Decision:
DUPLICATE
```

Meaning:

```text
rotated airplane -> airplane centroid neighborhood -> exact original airplane -> duplicate
```

This is the cleanest duplicate case. The attack variant still lands close enough to its original embedding.

**Case 3: Attack Reaches Duplicate Neighborhood, Not Exact Original**
Example from Amazon earlier saved trace:

```text
QUERY:
portable_cd_players_37_org.jpg + rotate40

Direct nearest original index:
32231

Graph nearest index:
2486

Graph distance:
0.754589 <= threshold 0.847949

Decision:
DUPLICATE
```

Meaning:

```text
rotated product image
   -> electronics/product centroid region
   -> not the exact original index
   -> but still a close duplicate-neighborhood image
   -> duplicate
```

So here the graph decision is correct, but it does not literally end at the exact original file.

**Case 4: Unknown Query Routes Somewhere But Stays Far**
Example from Flickr:

```text
QUERY:
7719481340.jpg + bright

Final region:
Flickr natural-image centroid region

Nearest known distance:
above threshold

Decision:
NON-DUPLICATE
```

Meaning:

```text
unknown bright image
   -> nearest available centroid region
   -> nearest known image is still too far
   -> not duplicate
```

Important point: unknown images do not avoid centroids. Every query reaches some centroid. The difference is distance after reaching that region.

**Final Rule**
```text
Query image
   -> HNSW / ERNG centroid route
   -> final candidate neighborhood
   -> nearest stored embedding
   -> compare distance with attack threshold
```

Then:

```text
distance <= threshold  => duplicate
distance > threshold   => non-duplicate
```

So your understanding is right with one correction:

```text
Unknown does not stay far from all centroids.
Unknown stays far from known embeddings after routing to the nearest centroid region.
```



Yes. These are the same four observed cases, now with the updated exact names.

**Case 1: Original Query Reaches Exact Original**

Example from CIFAR:

```text
QUERY:
airplane_train_29944_org.png + original

HNSW route:
C0048 / airplane_pool region
   -> C0378 / airplane_pool region
   -> C0271 / airplane_pool region

Final centroid:
C0271 / airplane_pool region

Final match:
airplane_train_29944_org.png

Distance:
0.000006 <= threshold 0.106488

Decision:
DUPLICATE
```

Meaning:

```text
original image
   -> airplane centroid neighborhood
   -> same original image
   -> duplicate
```

**Case 2: Weak Attack Reaches Exact Original**

Example from Amazon:

```text
QUERY:
creams_lotions_32_org.jpg + rotate_m5

Final centroid region:
cosmetics / creams_lotions product region

Final match:
creams_lotions_32_org.jpg

Distance:
within threshold

Decision:
DUPLICATE
```

Meaning:

```text
rotated product image
   -> creams_lotions centroid neighborhood
   -> exact original product image
   -> duplicate
```

This is the cleanest attack-duplicate case. The attacked variant still lands close enough to its original embedding.

**Case 3: Attack Reaches Duplicate Neighborhood, Not Exact Original**

Example from Amazon earlier saved trace:

```text
QUERY:
portable_cd_players_37_org.jpg + rotate40

Direct nearest index:
32231

Direct nearest image:
speaker_systems_86_org.jpg

Graph nearest index:
2486

Graph nearest image:
skateboarding-68_org.jpg

Graph distance:
0.754589 <= threshold 0.847949

Decision:
DUPLICATE
```

Meaning:

```text
rotated product image
   -> graph routes to a close candidate neighborhood
   -> does not end at the exact original file
   -> nearest graph image is still inside duplicate threshold
   -> duplicate
```

So here the graph decision is correct, but it does not literally end at the exact original file.

**Case 4: Unknown Query Routes Somewhere But Stays Far**

Example from Flickr:

```text
QUERY:
7719481340.jpg + bright

Final region:
Flickr natural-image centroid region

Nearest known distance:
above threshold

Decision:
NON-DUPLICATE
```

Meaning:

```text
unknown bright image
   -> nearest available centroid region
   -> nearest known image is still too far
   -> not duplicate
```

Important point: unknown images do not avoid centroids. Every query reaches some centroid. The difference is distance after reaching that region.

**Final Rule**

```text
Query image
   -> HNSW / ERNG centroid route
   -> final candidate neighborhood
   -> nearest stored embedding
   -> compare distance with attack threshold
```

Then:

```text
distance <= threshold  => duplicate
distance > threshold   => non-duplicate
```

So the correction is:

```text
Unknown does not stay far from all centroids.
Unknown stays far from known embeddings after routing to the nearest centroid region.
```


Yes. That can happen, and it is not automatically wrong.

In that Amazon example:

```text
QUERY:
portable_cd_players_37_org.jpg + rotate40

Graph nearest image:
skateboarding-68_org.jpg

Distance:
0.754589 <= threshold 0.847949

Decision:
DUPLICATE
```

The confusing part is that the filenames/categories look unrelated:

```text
portable_cd_players  vs  skateboarding
```

But the model/graph is not using the category name. It is using the learned embedding geometry. Two different product images can land in the same or nearby centroid region if their visual embedding features are similar.

Possible reasons:

```text
1. Shape similarity
   Both may have round/compact object structure.

2. Color/background similarity
   Amazon product images often have white backgrounds, centered objects, similar lighting.

3. Texture/layout similarity
   The model may focus on object silhouette, edges, shadows, or packaging layout rather than category label.

4. Rotate40 attack distortion
   A strong rotation can move the portable CD player embedding away from its exact original and closer to another visually similar product cluster.

5. Centroid is coarse
   A centroid is not one image identity. It is a region containing many embeddings. Different product categories can share one centroid if they are visually close in embedding space.
```

So the correct interpretation is:

```text
portable_cd_players_37 + rotate40
      |
      v
embedding moved by attack
      |
      v
nearest graph neighborhood happens to contain skateboarding-68
      |
      v
distance is still below rotate40 threshold
      |
      v
duplicate decision
```

But this also tells us something important:

```text
This is a duplicate-decision success, not an identity-match success.
```

For presentation, you should separate these two:

```text
Exact original retrieval:
query reaches the same original image.

Duplicate classification:
query reaches any close known-bank neighbor under threshold.
```

The project’s final evaluation is based on duplicate / non-duplicate decision, so this example is counted as correct. But if your goal is strict “same original identity retrieval,” then this specific example is weaker and should not be used as the clean diagram example.

Use this clean one instead for explaining exact identity:

```text
creams_lotions_32_org.jpg + rotate_m5
      |
      v
creams_lotions centroid region
      |
      v
MATCHED KNOWN IMAGE: creams_lotions_32_org.jpg
      |
      v
DUPLICATE DETECTED
```

That one is easier and safer to explain.