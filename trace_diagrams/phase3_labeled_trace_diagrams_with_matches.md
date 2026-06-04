# Labeled HNSW / ERNG Trace Diagrams With Final Matched Images

Each centroid is named by the nearest real stored image assigned to that centroid. Each route now ends with the matched known-bank image file, then the duplicate / non-duplicate decision.

## CIFAR

Store: `17000` known / `9000` unknown. Selected centroids: `512`.

### Duplicate Examples

#### Duplicate Example 1: `airplane_train_29944_org.png` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: airplane_train_29944_org.png + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / nearest airplane_train_11280_org.png
        | query-centroid distance: 1.292476
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / nearest airplane_train_11280_org.png
        | query-centroid distance: 1.292476
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0378 / airplane_pool region / nearest airplane_train_49535_org.png
        | query-centroid distance: 0.90509
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0378 / airplane_pool region / nearest airplane_train_49535_org.png
        | query-centroid distance: 0.90509
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        | query-centroid distance: 0.211204
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_29944_org.png
        | graph distance: 6e-06  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_29944_org.png
Direct duplicate: True | HNSW duplicate: True
Direct time: 7.837308 ms | HNSW time: 0.377745 ms | checked: 234
```

##### ERNG

```text
QUERY: airplane_train_29944_org.png + original

ERNG selected multi-probe path
Probe entry: C0005 / bird_pool region / nearest bird_train_44340_org.png

        +--------------------------------------------------+
        | C0005 / bird_pool region / nearest bird_train_44340_org.png
        | query-centroid distance: 2.217242
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0332 / airplane_pool region / nearest airplane_train_08444_org.png
        | query-centroid distance: 1.921669
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0378 / airplane_pool region / nearest airplane_train_49535_org.png
        | query-centroid distance: 0.90509
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        | query-centroid distance: 0.211204
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_29944_org.png
        | graph distance: 6e-06  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_29944_org.png
Direct duplicate: True | ERNG duplicate: True
Direct time: 7.780285 ms | ERNG time: 0.367723 ms | checked: 234
```

#### Duplicate Example 2: `airplane_train_07520_org.png` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: airplane_train_07520_org.png + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / nearest airplane_train_11280_org.png
        | query-centroid distance: 1.137142
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / nearest airplane_train_11280_org.png
        | query-centroid distance: 1.137142
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0378 / airplane_pool region / nearest airplane_train_49535_org.png
        | query-centroid distance: 0.731144
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0378 / airplane_pool region / nearest airplane_train_49535_org.png
        | query-centroid distance: 0.731144
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        | query-centroid distance: 0.340388
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_07520_org.png
        | graph distance: 6e-06  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_07520_org.png
Direct duplicate: True | HNSW duplicate: True
Direct time: 7.844295 ms | HNSW time: 0.407938 ms | checked: 234
```

##### ERNG

```text
QUERY: airplane_train_07520_org.png + original

ERNG selected multi-probe path
Probe entry: C0005 / bird_pool region / nearest bird_train_44340_org.png

        +--------------------------------------------------+
        | C0005 / bird_pool region / nearest bird_train_44340_org.png
        | query-centroid distance: 2.092684
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0332 / airplane_pool region / nearest airplane_train_08444_org.png
        | query-centroid distance: 1.753736
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0378 / airplane_pool region / nearest airplane_train_49535_org.png
        | query-centroid distance: 0.731144
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        | query-centroid distance: 0.340388
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_07520_org.png
        | graph distance: 6e-06  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_07520_org.png
Direct duplicate: True | ERNG duplicate: True
Direct time: 7.754371 ms | ERNG time: 0.437823 ms | checked: 234
```

#### Duplicate Example 3: `ship_train_00062_org.png` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: ship_train_00062_org.png + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / nearest airplane_train_11280_org.png
        | query-centroid distance: 1.253564
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / nearest airplane_train_11280_org.png
        | query-centroid distance: 1.253564
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0257 / airplane_pool region / nearest airplane_train_03859_org.png
        | query-centroid distance: 0.899232
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0257 / airplane_pool region / nearest airplane_train_03859_org.png
        | query-centroid distance: 0.899232
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        | query-centroid distance: 0.349789
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: ship_train_00062_org.png
        | graph distance: 6e-06  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: ship_train_00062_org.png
Direct duplicate: True | HNSW duplicate: True
Direct time: 10.636981 ms | HNSW time: 0.541382 ms | checked: 234
```

##### ERNG

```text
QUERY: ship_train_00062_org.png + original

ERNG selected multi-probe path
Probe entry: C0005 / bird_pool region / nearest bird_train_44340_org.png

        +--------------------------------------------------+
        | C0005 / bird_pool region / nearest bird_train_44340_org.png
        | query-centroid distance: 2.227633
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0420 / deer_pool region / nearest deer_train_48760_org.png
        | query-centroid distance: 1.941317
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0116 / airplane_pool region / nearest airplane_train_33146_org.png
        | query-centroid distance: 1.123702
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0140 / bird_pool region / nearest bird_train_10480_org.png
        | query-centroid distance: 0.602796
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        | query-centroid distance: 0.349789
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0271 / airplane_pool region / nearest airplane_train_21305_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: ship_train_00062_org.png
        | graph distance: 6e-06  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: ship_train_00062_org.png
Direct duplicate: True | ERNG duplicate: True
Direct time: 10.390464 ms | ERNG time: 0.577749 ms | checked: 234
```

### Non-Duplicate Examples

#### Non-Duplicate Example 1: `cat_train_00021_org.png` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: cat_train_00021_org.png + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0337 / ship_pool region / nearest ship_train_46975_org.png
        | query-centroid distance: 1.603852
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0337 / ship_pool region / nearest ship_train_46975_org.png
        | query-centroid distance: 1.603852
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0300 / frog_pool region / nearest frog_train_23795_org.png
        | query-centroid distance: 1.379437
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0300 / frog_pool region / nearest frog_train_23795_org.png
        | query-centroid distance: 1.379437
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0008 / deer_pool region / nearest deer_train_12611_org.png
        | query-centroid distance: 0.562831
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0008 / deer_pool region / nearest deer_train_12611_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_12079_org.png
        | graph distance: 0.208815  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: airplane_train_12079_org.png
Direct duplicate: False | HNSW duplicate: False
Direct time: 10.36786 ms | HNSW time: 0.872916 ms | checked: 248
```

##### ERNG

```text
QUERY: cat_train_00021_org.png + original

ERNG selected multi-probe path
Probe entry: C0310 / deer_pool region / nearest deer_train_00712_org.png

        +--------------------------------------------------+
        | C0310 / deer_pool region / nearest deer_train_00712_org.png
        | query-centroid distance: 3.448983
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0132 / frog_pool region / nearest frog_train_46193_org.png
        | query-centroid distance: 2.132477
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0382 / bird_pool region / nearest bird_train_12043_org.png
        | query-centroid distance: 1.202349
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0008 / deer_pool region / nearest deer_train_12611_org.png
        | query-centroid distance: 0.562831
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0008 / deer_pool region / nearest deer_train_12611_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_12079_org.png
        | graph distance: 0.208815  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: airplane_train_12079_org.png
Direct duplicate: False | ERNG duplicate: False
Direct time: 10.347566 ms | ERNG time: 0.807619 ms | checked: 248
```

#### Non-Duplicate Example 2: `automobile_train_08085_org.png` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: automobile_train_08085_org.png + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0098 / bird_pool region / nearest bird_train_36186_org.png
        | query-centroid distance: 2.106381
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0098 / bird_pool region / nearest bird_train_36186_org.png
        | query-centroid distance: 2.106381
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0098 / bird_pool region / nearest bird_train_36186_org.png
        | query-centroid distance: 2.106381
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0108 / ship_pool region / nearest ship_train_48044_org.png
        | query-centroid distance: 1.517454
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0108 / ship_pool region / nearest ship_train_48044_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_17762_org.png
        | graph distance: 1.119368  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: airplane_train_17762_org.png
Direct duplicate: False | HNSW duplicate: False
Direct time: 10.627412 ms | HNSW time: 0.476978 ms | checked: 23
```

##### ERNG

```text
QUERY: automobile_train_08085_org.png + original

ERNG selected multi-probe path
Probe entry: C0310 / deer_pool region / nearest deer_train_00712_org.png

        +--------------------------------------------------+
        | C0310 / deer_pool region / nearest deer_train_00712_org.png
        | query-centroid distance: 3.656726
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0438 / airplane_pool region / nearest airplane_train_42797_org.png
        | query-centroid distance: 2.308471
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0108 / ship_pool region / nearest ship_train_48044_org.png
        | query-centroid distance: 1.517454
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0108 / ship_pool region / nearest ship_train_48044_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_17762_org.png
        | graph distance: 1.119368  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: airplane_train_17762_org.png
Direct duplicate: False | ERNG duplicate: False
Direct time: 10.337671 ms | ERNG time: 0.631567 ms | checked: 23
```

#### Non-Duplicate Example 3: `automobile_train_38333_org.png` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: automobile_train_38333_org.png + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0132 / frog_pool region / nearest frog_train_46193_org.png
        | query-centroid distance: 2.222028
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0132 / frog_pool region / nearest frog_train_46193_org.png
        | query-centroid distance: 2.222028
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0208 / deer_pool region / nearest deer_train_33004_org.png
        | query-centroid distance: 1.547636
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0208 / deer_pool region / nearest deer_train_33004_org.png
        | query-centroid distance: 1.547636
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0437 / frog_pool region / nearest frog_train_28860_org.png
        | query-centroid distance: 1.393069
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0437 / frog_pool region / nearest frog_train_28860_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: frog_train_28644_org.png
        | graph distance: 1.254079  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: bird_train_30672_org.png
Direct duplicate: False | HNSW duplicate: False
Direct time: 10.379433 ms | HNSW time: 0.729789 ms | checked: 14
```

##### ERNG

```text
QUERY: automobile_train_38333_org.png + original

ERNG selected multi-probe path
Probe entry: C0280 / frog_pool region / nearest frog_train_44915_org.png

        +--------------------------------------------------+
        | C0280 / frog_pool region / nearest frog_train_44915_org.png
        | query-centroid distance: 3.138988
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0043 / frog_pool region / nearest frog_train_02439_org.png
        | query-centroid distance: 2.685875
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0190 / deer_pool region / nearest deer_train_21254_org.png
        | query-centroid distance: 2.203371
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0208 / deer_pool region / nearest deer_train_33004_org.png
        | query-centroid distance: 1.547636
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0437 / frog_pool region / nearest frog_train_28860_org.png
        | query-centroid distance: 1.393069
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0437 / frog_pool region / nearest frog_train_28860_org.png
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: frog_train_28644_org.png
        | graph distance: 1.254079  threshold: 0.106488
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: bird_train_30672_org.png
Direct duplicate: False | ERNG duplicate: False
Direct time: 10.482196 ms | ERNG time: 0.53062 ms | checked: 14
```

## Flickr

Store: `10000` known / `5000` unknown. Selected centroids: `192`.

### Duplicate Examples

#### Duplicate Example 1: `2494088238.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 2494088238.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0083 / flickr30k_images region / nearest 3397228832.jpg
        | query-centroid distance: 3.234508
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0083 / flickr30k_images region / nearest 3397228832.jpg
        | query-centroid distance: 3.234508
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0058 / flickr30k_images region / nearest 328916930.jpg
        | query-centroid distance: 1.985492
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0058 / flickr30k_images region / nearest 328916930.jpg
        | query-centroid distance: 1.985492
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0092 / flickr30k_images region / nearest 4725965235.jpg
        | query-centroid distance: 0.763657
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0092 / flickr30k_images region / nearest 4725965235.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2494088238.jpg
        | graph distance: 0.008563  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2494088238.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 6.351392 ms | HNSW time: 0.283932 ms | checked: 79
```

##### ERNG

```text
QUERY: 2494088238.jpg + original

ERNG selected multi-probe path
Probe entry: C0140 / flickr30k_images region / nearest 6166659366.jpg

        +--------------------------------------------------+
        | C0140 / flickr30k_images region / nearest 6166659366.jpg
        | query-centroid distance: 5.56117
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0150 / flickr30k_images region / nearest 4813957025.jpg
        | query-centroid distance: 4.214097
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0098 / flickr30k_images region / nearest 2538642969.jpg
        | query-centroid distance: 1.910858
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0092 / flickr30k_images region / nearest 4725965235.jpg
        | query-centroid distance: 0.763657
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0092 / flickr30k_images region / nearest 4725965235.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2494088238.jpg
        | graph distance: 0.008563  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2494088238.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 6.324182 ms | ERNG time: 0.298006 ms | checked: 79
```

#### Duplicate Example 2: `2796544901.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 2796544901.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0119 / flickr30k_images region / nearest 3265578645.jpg
        | query-centroid distance: 1.112328
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0119 / flickr30k_images region / nearest 3265578645.jpg
        | query-centroid distance: 1.112328
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0119 / flickr30k_images region / nearest 3265578645.jpg
        | query-centroid distance: 1.112328
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0119 / flickr30k_images region / nearest 3265578645.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2796544901.jpg
        | graph distance: 0.007072  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2796544901.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 6.314159 ms | HNSW time: 0.315144 ms | checked: 44
```

##### ERNG

```text
QUERY: 2796544901.jpg + original

ERNG selected multi-probe path
Probe entry: C0140 / flickr30k_images region / nearest 6166659366.jpg

        +--------------------------------------------------+
        | C0140 / flickr30k_images region / nearest 6166659366.jpg
        | query-centroid distance: 2.24086
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0119 / flickr30k_images region / nearest 3265578645.jpg
        | query-centroid distance: 1.112328
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0119 / flickr30k_images region / nearest 3265578645.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2796544901.jpg
        | graph distance: 0.007072  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2796544901.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 6.291937 ms | ERNG time: 0.296767 ms | checked: 44
```

#### Duplicate Example 3: `1718184338.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 1718184338.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0140 / flickr30k_images region / nearest 6166659366.jpg
        | query-centroid distance: 1.331541
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0140 / flickr30k_images region / nearest 6166659366.jpg
        | query-centroid distance: 1.331541
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0140 / flickr30k_images region / nearest 6166659366.jpg
        | query-centroid distance: 1.331541
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0140 / flickr30k_images region / nearest 6166659366.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 1718184338.jpg
        | graph distance: 0.007817  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 1718184338.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 6.475939 ms | HNSW time: 0.321439 ms | checked: 67
```

##### ERNG

```text
QUERY: 1718184338.jpg + original

ERNG selected multi-probe path
Probe entry: C0013 / flickr30k_images region / nearest 4507548183.jpg

        +--------------------------------------------------+
        | C0013 / flickr30k_images region / nearest 4507548183.jpg
        | query-centroid distance: 4.336624
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0097 / flickr30k_images region / nearest 2230983294.jpg
        | query-centroid distance: 2.263212
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0140 / flickr30k_images region / nearest 6166659366.jpg
        | query-centroid distance: 1.331541
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0140 / flickr30k_images region / nearest 6166659366.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 1718184338.jpg
        | graph distance: 0.007817  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 1718184338.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 6.30106 ms | ERNG time: 0.308539 ms | checked: 67
```

### Non-Duplicate Examples

#### Non-Duplicate Example 1: `3750418259.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 3750418259.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0093 / flickr30k_images region / nearest 1884727806.jpg
        | query-centroid distance: 3.433873
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0093 / flickr30k_images region / nearest 1884727806.jpg
        | query-centroid distance: 3.433873
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0160 / flickr30k_images region / nearest 4435343922.jpg
        | query-centroid distance: 2.193813
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0160 / flickr30k_images region / nearest 4435343922.jpg
        | query-centroid distance: 2.193813
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0158 / flickr30k_images region / nearest 138705546.jpg
        | query-centroid distance: 1.019342
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0158 / flickr30k_images region / nearest 138705546.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2935703360.jpg
        | graph distance: 0.472409  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 2935703360.jpg
Direct duplicate: False | HNSW duplicate: False
Direct time: 4.613996 ms | HNSW time: 0.316045 ms | checked: 169
```

##### ERNG

```text
QUERY: 3750418259.jpg + original

ERNG selected multi-probe path
Probe entry: C0013 / flickr30k_images region / nearest 4507548183.jpg

        +--------------------------------------------------+
        | C0013 / flickr30k_images region / nearest 4507548183.jpg
        | query-centroid distance: 5.810879
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0020 / flickr30k_images region / nearest 2918653119.jpg
        | query-centroid distance: 3.907043
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0160 / flickr30k_images region / nearest 4435343922.jpg
        | query-centroid distance: 2.193813
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0158 / flickr30k_images region / nearest 138705546.jpg
        | query-centroid distance: 1.019342
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0158 / flickr30k_images region / nearest 138705546.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2935703360.jpg
        | graph distance: 0.472409  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 2935703360.jpg
Direct duplicate: False | ERNG duplicate: False
Direct time: 4.582475 ms | ERNG time: 0.291818 ms | checked: 169
```

#### Non-Duplicate Example 2: `6168223686.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 6168223686.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0093 / flickr30k_images region / nearest 1884727806.jpg
        | query-centroid distance: 3.73734
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0093 / flickr30k_images region / nearest 1884727806.jpg
        | query-centroid distance: 3.73734
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0155 / flickr30k_images region / nearest 2508249781.jpg
        | query-centroid distance: 2.312972
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0155 / flickr30k_images region / nearest 2508249781.jpg
        | query-centroid distance: 2.312972
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0028 / flickr30k_images region / nearest 533854547.jpg
        | query-centroid distance: 1.436738
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0028 / flickr30k_images region / nearest 533854547.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 33064663.jpg
        | graph distance: 0.779644  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 33064663.jpg
Direct duplicate: False | HNSW duplicate: False
Direct time: 4.747041 ms | HNSW time: 0.277681 ms | checked: 128
```

##### ERNG

```text
QUERY: 6168223686.jpg + original

ERNG selected multi-probe path
Probe entry: C0013 / flickr30k_images region / nearest 4507548183.jpg

        +--------------------------------------------------+
        | C0013 / flickr30k_images region / nearest 4507548183.jpg
        | query-centroid distance: 6.532086
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0126 / flickr30k_images region / nearest 2506113060.jpg
        | query-centroid distance: 4.577941
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0042 / flickr30k_images region / nearest 4938457809.jpg
        | query-centroid distance: 2.507472
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0072 / flickr30k_images region / nearest 2787868417.jpg
        | query-centroid distance: 1.5114
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0028 / flickr30k_images region / nearest 533854547.jpg
        | query-centroid distance: 1.436738
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0028 / flickr30k_images region / nearest 533854547.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 33064663.jpg
        | graph distance: 0.779644  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 33064663.jpg
Direct duplicate: False | ERNG duplicate: False
Direct time: 4.578137 ms | ERNG time: 0.271616 ms | checked: 128
```

#### Non-Duplicate Example 3: `58046927.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 58046927.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0119 / flickr30k_images region / nearest 3265578645.jpg
        | query-centroid distance: 1.367488
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0119 / flickr30k_images region / nearest 3265578645.jpg
        | query-centroid distance: 1.367488
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0081 / flickr30k_images region / nearest 1147391743.jpg
        | query-centroid distance: 1.239944
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0081 / flickr30k_images region / nearest 1147391743.jpg
        | query-centroid distance: 1.239944
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0159 / flickr30k_images region / nearest 7634501754.jpg
        | query-centroid distance: 1.165591
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0159 / flickr30k_images region / nearest 7634501754.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 3041170372.jpg
        | graph distance: 0.406281  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 3041170372.jpg
Direct duplicate: False | HNSW duplicate: False
Direct time: 4.569572 ms | HNSW time: 0.578268 ms | checked: 609
```

##### ERNG

```text
QUERY: 58046927.jpg + original

ERNG selected multi-probe path
Probe entry: C0013 / flickr30k_images region / nearest 4507548183.jpg

        +--------------------------------------------------+
        | C0013 / flickr30k_images region / nearest 4507548183.jpg
        | query-centroid distance: 3.042758
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0081 / flickr30k_images region / nearest 1147391743.jpg
        | query-centroid distance: 1.239944
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0159 / flickr30k_images region / nearest 7634501754.jpg
        | query-centroid distance: 1.165591
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0159 / flickr30k_images region / nearest 7634501754.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 3041170372.jpg
        | graph distance: 0.406281  threshold: 0.080286
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 3041170372.jpg
Direct duplicate: False | ERNG duplicate: False
Direct time: 4.571275 ms | ERNG time: 0.729122 ms | checked: 609
```

## UCID

Store: `4500` known / `850` unknown. Selected centroids: `128`.

### Duplicate Examples

#### Duplicate Example 1: `60_aug_3.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 60_aug_3.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0043 / test region / nearest 848_orig.jpg
        | query-centroid distance: 13.19521
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0043 / test region / nearest 848_orig.jpg
        | query-centroid distance: 13.19521
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0066 / train region / nearest 640_aug_1.jpg
        | query-centroid distance: 11.411723
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0003 / train region / nearest 826_aug_2.jpg
        | query-centroid distance: 10.596786
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0003 / train region / nearest 826_aug_2.jpg
        | query-centroid distance: 10.596786
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0122 / train region / nearest 1172_aug_1.jpg
        | query-centroid distance: 9.47544
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0008 / train region / nearest 1316_orig.jpg
        | query-centroid distance: 7.536637
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0030 / train region / nearest 655_aug_1.jpg
        | query-centroid distance: 5.887594
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0125 / train region / nearest 989_aug_2.jpg
        | query-centroid distance: 3.475959
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0101 / train region / nearest 987_aug_2.jpg
        | query-centroid distance: 0.555841
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0101 / train region / nearest 987_aug_2.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 60_aug_3.jpg
        | graph distance: 0.0  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 60_aug_3.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 2.126641 ms | HNSW time: 0.155219 ms | checked: 37
```

##### ERNG

```text
QUERY: 60_aug_3.jpg + original

ERNG selected multi-probe path
Probe entry: C0009 / train region / nearest 126_aug_2.jpg

        +--------------------------------------------------+
        | C0009 / train region / nearest 126_aug_2.jpg
        | query-centroid distance: 19.696182
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0116 / train region / nearest 1150_aug_1.jpg
        | query-centroid distance: 14.974404
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0097 / train region / nearest 362_aug_1.jpg
        | query-centroid distance: 12.22041
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0062 / train region / nearest 459_aug_3.jpg
        | query-centroid distance: 10.025852
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0075 / train region / nearest 1036_aug_3.jpg
        | query-centroid distance: 7.981226
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0014 / train region / nearest 1277_orig.jpg
        | query-centroid distance: 0.972889
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0101 / train region / nearest 987_aug_2.jpg
        | query-centroid distance: 0.555841
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0101 / train region / nearest 987_aug_2.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 60_aug_3.jpg
        | graph distance: 0.0  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 60_aug_3.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 2.057576 ms | ERNG time: 0.150755 ms | checked: 37
```

#### Duplicate Example 2: `995_orig.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 995_orig.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0126 / test region / nearest 1111_aug_2.jpg
        | query-centroid distance: 3.605433
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0126 / test region / nearest 1111_aug_2.jpg
        | query-centroid distance: 3.605433
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0001 / train region / nearest 1239_aug_3.jpg
        | query-centroid distance: 2.564591
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0109 / train region / nearest 1291_aug_2.jpg
        | query-centroid distance: 1.938089
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0109 / train region / nearest 1291_aug_2.jpg
        | query-centroid distance: 1.938089
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0036 / train region / nearest 225_aug_1.jpg
        | query-centroid distance: 1.234467
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0035 / train region / nearest 995_aug_2.jpg
        | query-centroid distance: 0.833692
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0084 / train region / nearest 78_aug_1.jpg
        | query-centroid distance: 0.765735
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0084 / train region / nearest 78_aug_1.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 995_orig.jpg
        | graph distance: 0.0  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 995_orig.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 2.103087 ms | HNSW time: 0.171249 ms | checked: 86
```

##### ERNG

```text
QUERY: 995_orig.jpg + original

ERNG selected multi-probe path
Probe entry: C0111 / train region / nearest 1104_aug_3.jpg

        +--------------------------------------------------+
        | C0111 / train region / nearest 1104_aug_3.jpg
        | query-centroid distance: 5.03934
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0015 / train region / nearest 474_aug_1.jpg
        | query-centroid distance: 1.859166
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0091 / train region / nearest 663_orig.jpg
        | query-centroid distance: 0.956368
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0084 / train region / nearest 78_aug_1.jpg
        | query-centroid distance: 0.765735
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0084 / train region / nearest 78_aug_1.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 995_orig.jpg
        | graph distance: 0.0  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 995_orig.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 2.017107 ms | ERNG time: 0.14831 ms | checked: 86
```

#### Duplicate Example 3: `1132_aug_2.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 1132_aug_2.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0126 / test region / nearest 1111_aug_2.jpg
        | query-centroid distance: 3.25608
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0126 / test region / nearest 1111_aug_2.jpg
        | query-centroid distance: 3.25608
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0001 / train region / nearest 1239_aug_3.jpg
        | query-centroid distance: 2.081143
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0067 / test region / nearest 981_aug_2.jpg
        | query-centroid distance: 1.588857
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0067 / test region / nearest 981_aug_2.jpg
        | query-centroid distance: 1.588857
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0073 / train region / nearest 983_aug_2.jpg
        | query-centroid distance: 0.953223
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0120 / train region / nearest 698_aug_2.jpg
        | query-centroid distance: 0.621333
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0120 / train region / nearest 698_aug_2.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 1132_aug_2.jpg
        | graph distance: 0.0  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 1132_aug_2.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 2.064904 ms | HNSW time: 0.152285 ms | checked: 66
```

##### ERNG

```text
QUERY: 1132_aug_2.jpg + original

ERNG selected multi-probe path
Probe entry: C0111 / train region / nearest 1104_aug_3.jpg

        +--------------------------------------------------+
        | C0111 / train region / nearest 1104_aug_3.jpg
        | query-centroid distance: 4.625782
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0015 / train region / nearest 474_aug_1.jpg
        | query-centroid distance: 1.048709
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0120 / train region / nearest 698_aug_2.jpg
        | query-centroid distance: 0.621333
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0120 / train region / nearest 698_aug_2.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 1132_aug_2.jpg
        | graph distance: 0.0  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 1132_aug_2.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 2.040072 ms | ERNG time: 0.143844 ms | checked: 66
```

### Non-Duplicate Examples

#### Non-Duplicate Example 1: `1068_aug_2.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 1068_aug_2.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0043 / test region / nearest 848_orig.jpg
        | query-centroid distance: 17.030003
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0043 / test region / nearest 848_orig.jpg
        | query-centroid distance: 17.030003
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0066 / train region / nearest 640_aug_1.jpg
        | query-centroid distance: 15.315225
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0003 / train region / nearest 826_aug_2.jpg
        | query-centroid distance: 14.364097
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0003 / train region / nearest 826_aug_2.jpg
        | query-centroid distance: 14.364097
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0060 / train region / nearest 800_orig.jpg
        | query-centroid distance: 13.319462
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0072 / train region / nearest 1314_aug_3.jpg
        | query-centroid distance: 9.304431
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0105 / train region / nearest 43_aug_3.jpg
        | query-centroid distance: 6.916686
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0101 / train region / nearest 987_aug_2.jpg
        | query-centroid distance: 4.398706
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0118 / train region / nearest 878_orig.jpg
        | query-centroid distance: 0.58428
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0118 / train region / nearest 878_orig.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 877_orig.jpg
        | graph distance: 0.37807  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 877_orig.jpg
Direct duplicate: False | HNSW duplicate: False
Direct time: 2.026672 ms | HNSW time: 0.165726 ms | checked: 43
```

##### ERNG

```text
QUERY: 1068_aug_2.jpg + original

ERNG selected multi-probe path
Probe entry: C0111 / train region / nearest 1104_aug_3.jpg

        +--------------------------------------------------+
        | C0111 / train region / nearest 1104_aug_3.jpg
        | query-centroid distance: 19.289465
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0070 / test region / nearest 97_aug_3.jpg
        | query-centroid distance: 17.034739
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0087 / train region / nearest 288_aug_3.jpg
        | query-centroid distance: 11.18525
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0105 / train region / nearest 43_aug_3.jpg
        | query-centroid distance: 6.916686
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0101 / train region / nearest 987_aug_2.jpg
        | query-centroid distance: 4.398706
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0118 / train region / nearest 878_orig.jpg
        | query-centroid distance: 0.58428
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0118 / train region / nearest 878_orig.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 877_orig.jpg
        | graph distance: 0.37807  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 877_orig.jpg
Direct duplicate: False | ERNG duplicate: False
Direct time: 1.999942 ms | ERNG time: 0.150936 ms | checked: 43
```

#### Non-Duplicate Example 2: `353_aug_1.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 353_aug_1.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0126 / test region / nearest 1111_aug_2.jpg
        | query-centroid distance: 3.845366
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0126 / test region / nearest 1111_aug_2.jpg
        | query-centroid distance: 3.845366
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0001 / train region / nearest 1239_aug_3.jpg
        | query-centroid distance: 2.713559
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0067 / test region / nearest 981_aug_2.jpg
        | query-centroid distance: 2.239038
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0067 / test region / nearest 981_aug_2.jpg
        | query-centroid distance: 2.239038
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0073 / train region / nearest 983_aug_2.jpg
        | query-centroid distance: 1.404755
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0004 / train region / nearest 701_aug_1.jpg
        | query-centroid distance: 0.768398
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0004 / train region / nearest 701_aug_1.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 387_aug_1.jpg
        | graph distance: 0.632362  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 387_aug_1.jpg
Direct duplicate: False | HNSW duplicate: False
Direct time: 2.066018 ms | HNSW time: 0.299704 ms | checked: 305
```

##### ERNG

```text
QUERY: 353_aug_1.jpg + original

ERNG selected multi-probe path
Probe entry: C0009 / train region / nearest 126_aug_2.jpg

        +--------------------------------------------------+
        | C0009 / train region / nearest 126_aug_2.jpg
        | query-centroid distance: 1.1394
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0004 / train region / nearest 701_aug_1.jpg
        | query-centroid distance: 0.768398
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0004 / train region / nearest 701_aug_1.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 387_aug_1.jpg
        | graph distance: 0.632362  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 387_aug_1.jpg
Direct duplicate: False | ERNG duplicate: False
Direct time: 2.040968 ms | ERNG time: 0.290643 ms | checked: 305
```

#### Non-Duplicate Example 3: `863_orig.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: 863_orig.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0043 / test region / nearest 848_orig.jpg
        | query-centroid distance: 4.954541
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0043 / test region / nearest 848_orig.jpg
        | query-centroid distance: 4.954541
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0090 / train region / nearest 1193_aug_1.jpg
        | query-centroid distance: 3.504538
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0003 / train region / nearest 826_aug_2.jpg
        | query-centroid distance: 2.285389
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0003 / train region / nearest 826_aug_2.jpg
        | query-centroid distance: 2.285389
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0060 / train region / nearest 800_orig.jpg
        | query-centroid distance: 1.332481
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0037 / train region / nearest 51_aug_2.jpg
        | query-centroid distance: 0.945099
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0037 / train region / nearest 51_aug_2.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 233_aug_3.jpg
        | graph distance: 0.885057  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 233_aug_3.jpg
Direct duplicate: False | HNSW duplicate: False
Direct time: 2.027222 ms | HNSW time: 0.202515 ms | checked: 129
```

##### ERNG

```text
QUERY: 863_orig.jpg + original

ERNG selected multi-probe path
Probe entry: C0009 / train region / nearest 126_aug_2.jpg

        +--------------------------------------------------+
        | C0009 / train region / nearest 126_aug_2.jpg
        | query-centroid distance: 11.452824
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0116 / train region / nearest 1150_aug_1.jpg
        | query-centroid distance: 6.555267
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0097 / train region / nearest 362_aug_1.jpg
        | query-centroid distance: 4.122964
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0003 / train region / nearest 826_aug_2.jpg
        | query-centroid distance: 2.285389
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0060 / train region / nearest 800_orig.jpg
        | query-centroid distance: 1.332481
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0037 / train region / nearest 51_aug_2.jpg
        | query-centroid distance: 0.945099
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0037 / train region / nearest 51_aug_2.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 233_aug_3.jpg
        | graph distance: 0.885057  threshold: 0.034055
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: 233_aug_3.jpg
Direct duplicate: False | ERNG duplicate: False
Direct time: 2.008886 ms | ERNG time: 0.188205 ms | checked: 129
```

## Amazon

Store: `35000` known / `15000` unknown. Selected centroids: `512`.

### Duplicate Examples

#### Duplicate Example 1: `creams_lotions_32_org.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: creams_lotions_32_org.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0401 / dinnerware_serveware_5 region / nearest dinnerware_serveware_5_org.jpg
        | query-centroid distance: 0.460149
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0401 / dinnerware_serveware_5 region / nearest dinnerware_serveware_5_org.jpg
        | query-centroid distance: 0.460149
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0401 / dinnerware_serveware_5 region / nearest dinnerware_serveware_5_org.jpg
        | query-centroid distance: 0.460149
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0401 / dinnerware_serveware_5 region / nearest dinnerware_serveware_5_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: creams_lotions_32_org.jpg
        | graph distance: 1.1e-05  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: creams_lotions_32_org.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 35.320412 ms | HNSW time: 0.675421 ms | checked: 148
```

##### ERNG

```text
QUERY: creams_lotions_32_org.jpg + original

ERNG selected multi-probe path
Probe entry: C0036 / special_occasion-18 region / nearest special_occasion-18_org.jpg

        +--------------------------------------------------+
        | C0036 / special_occasion-18 region / nearest special_occasion-18_org.jpg
        | query-centroid distance: 20.32082
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0130 / beer_glasses_59 region / nearest beer_glasses_59_org.jpg
        | query-centroid distance: 0.903925
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0401 / dinnerware_serveware_5 region / nearest dinnerware_serveware_5_org.jpg
        | query-centroid distance: 0.460149
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0401 / dinnerware_serveware_5 region / nearest dinnerware_serveware_5_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: creams_lotions_32_org.jpg
        | graph distance: 1.1e-05  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: creams_lotions_32_org.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 35.262131 ms | ERNG time: 0.632782 ms | checked: 148
```

#### Duplicate Example 2: `beverage_warmers_83_org.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: beverage_warmers_83_org.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 11.979316
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 11.979316
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0505 / curling_tongs_80 region / nearest curling_tongs_80_org.jpg
        | query-centroid distance: 9.802219
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0262 / tactical_vests_6 region / nearest tactical_vests_6_org.jpg
        | query-centroid distance: 8.698644
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0086 / putties_25 region / nearest putties_25_org.jpg
        | query-centroid distance: 7.909788
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0022 / cartridge_razors_60 region / nearest cartridge_razors_60_org.jpg
        | query-centroid distance: 7.854097
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0022 / cartridge_razors_60 region / nearest cartridge_razors_60_org.jpg
        | query-centroid distance: 7.854097
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0239 / cd_players_75 region / nearest cd_players_75_org.jpg
        | query-centroid distance: 2.098774
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0200 / clock_radios_41 region / nearest clock_radios_41_org.jpg
        | query-centroid distance: 0.529602
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0200 / clock_radios_41 region / nearest clock_radios_41_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: beverage_warmers_83_org.jpg
        | graph distance: 1.1e-05  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: beverage_warmers_83_org.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 35.264842 ms | HNSW time: 0.670463 ms | checked: 143
```

##### ERNG

```text
QUERY: beverage_warmers_83_org.jpg + original

ERNG selected multi-probe path
Probe entry: C0445 / pillow_inserts_70 region / nearest pillow_inserts_70_org.jpg

        +--------------------------------------------------+
        | C0445 / pillow_inserts_70 region / nearest pillow_inserts_70_org.jpg
        | query-centroid distance: 14.817552
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0496 / stroller_connectors_100 region / nearest stroller_connectors_100_org.jpg
        | query-centroid distance: 5.090735
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0042 / tandem_60 region / nearest tandem_60_org.jpg
        | query-centroid distance: 1.772001
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0065 / camera_cases_4 region / nearest camera_cases_4_org.jpg
        | query-centroid distance: 0.734115
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0200 / clock_radios_41 region / nearest clock_radios_41_org.jpg
        | query-centroid distance: 0.529602
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0200 / clock_radios_41 region / nearest clock_radios_41_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: beverage_warmers_83_org.jpg
        | graph distance: 1.1e-05  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: beverage_warmers_83_org.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 35.298132 ms | ERNG time: 0.65084 ms | checked: 143
```

#### Duplicate Example 3: `towel_racks_68_org.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: towel_racks_68_org.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 0.785464
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 0.785464
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 0.785464
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: towel_racks_68_org.jpg
        | graph distance: 1.1e-05  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: towel_racks_68_org.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 35.277882 ms | HNSW time: 0.683051 ms | checked: 145
```

##### ERNG

```text
QUERY: towel_racks_68_org.jpg + original

ERNG selected multi-probe path
Probe entry: C0445 / pillow_inserts_70 region / nearest pillow_inserts_70_org.jpg

        +--------------------------------------------------+
        | C0445 / pillow_inserts_70 region / nearest pillow_inserts_70_org.jpg
        | query-centroid distance: 2.991219
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0300 / nasal_aspirators_7 region / nearest nasal_aspirators_7_org.jpg
        | query-centroid distance: 1.004247
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 0.785464
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: towel_racks_68_org.jpg
        | graph distance: 1.1e-05  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: towel_racks_68_org.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 35.264834 ms | ERNG time: 0.66325 ms | checked: 145
```

### Non-Duplicate Examples

#### Non-Duplicate Example 1: `headboards_56_org.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: headboards_56_org.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 2.671242
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 2.671242
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0000 / thermometers_85 region / nearest thermometers_85_org.jpg
        | query-centroid distance: 1.122182
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0000 / thermometers_85 region / nearest thermometers_85_org.jpg
        | query-centroid distance: 1.122182
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0000 / thermometers_85 region / nearest thermometers_85_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: soap_dishes_9_org.jpg
        | graph distance: 0.953429  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: soap_dishes_9_org.jpg
Direct duplicate: False | HNSW duplicate: False
Direct time: 35.610893 ms | HNSW time: 2.516496 ms | checked: 1697
```

##### ERNG

```text
QUERY: headboards_56_org.jpg + original

ERNG selected multi-probe path
Probe entry: C0445 / pillow_inserts_70 region / nearest pillow_inserts_70_org.jpg

        +--------------------------------------------------+
        | C0445 / pillow_inserts_70 region / nearest pillow_inserts_70_org.jpg
        | query-centroid distance: 5.44958
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0300 / nasal_aspirators_7 region / nearest nasal_aspirators_7_org.jpg
        | query-centroid distance: 2.17865
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0317 / locking_carabiners_2 region / nearest locking_carabiners_2_org.jpg
        | query-centroid distance: 1.418989
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0000 / thermometers_85 region / nearest thermometers_85_org.jpg
        | query-centroid distance: 1.122182
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0000 / thermometers_85 region / nearest thermometers_85_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: soap_dishes_9_org.jpg
        | graph distance: 0.953429  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: soap_dishes_9_org.jpg
Direct duplicate: False | ERNG duplicate: False
Direct time: 35.54753 ms | ERNG time: 2.50218 ms | checked: 1697
```

#### Non-Duplicate Example 2: `dslr_cameras_35_org.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: dslr_cameras_35_org.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 9.330336
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 9.330336
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0505 / curling_tongs_80 region / nearest curling_tongs_80_org.jpg
        | query-centroid distance: 7.088582
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0262 / tactical_vests_6 region / nearest tactical_vests_6_org.jpg
        | query-centroid distance: 5.948635
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0086 / putties_25 region / nearest putties_25_org.jpg
        | query-centroid distance: 5.128182
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0086 / putties_25 region / nearest putties_25_org.jpg
        | query-centroid distance: 5.128182
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0196 / neckties-67 region / nearest neckties-67_org.jpg
        | query-centroid distance: 1.51562
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0220 / monitor_arms_48 region / nearest monitor_arms_48_org.jpg
        | query-centroid distance: 0.852708
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0484 / mirrorless_cameras_4 region / nearest mirrorless_cameras_4_org.jpg
        | query-centroid distance: 0.623094
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0484 / mirrorless_cameras_4 region / nearest mirrorless_cameras_4_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: calf_socks-98_org.jpg
        | graph distance: 0.57905  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: calf_socks-98_org.jpg
Direct duplicate: False | HNSW duplicate: False
Direct time: 35.583705 ms | HNSW time: 2.496651 ms | checked: 1683
```

##### ERNG

```text
QUERY: dslr_cameras_35_org.jpg + original

ERNG selected multi-probe path
Probe entry: C0036 / special_occasion-18 region / nearest special_occasion-18_org.jpg

        +--------------------------------------------------+
        | C0036 / special_occasion-18 region / nearest special_occasion-18_org.jpg
        | query-centroid distance: 33.486759
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0130 / beer_glasses_59 region / nearest beer_glasses_59_org.jpg
        | query-centroid distance: 12.591318
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0293 / cables_cords_28 region / nearest cables_cords_28_org.jpg
        | query-centroid distance: 2.375255
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0258 / on_dash_cameras_52 region / nearest on_dash_cameras_52_org.jpg
        | query-centroid distance: 1.03266
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0484 / mirrorless_cameras_4 region / nearest mirrorless_cameras_4_org.jpg
        | query-centroid distance: 0.623094
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0484 / mirrorless_cameras_4 region / nearest mirrorless_cameras_4_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: calf_socks-98_org.jpg
        | graph distance: 0.57905  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: calf_socks-98_org.jpg
Direct duplicate: False | ERNG duplicate: False
Direct time: 35.566241 ms | ERNG time: 2.497635 ms | checked: 1683
```

#### Non-Duplicate Example 3: `candlestick_holders_86_org.jpg` attack `original`

Reasoning: route through centroid regions, search only the final candidate neighborhood, compare the matched image distance to the same calibrated threshold used by direct search.

##### HNSW

```text
QUERY: candlestick_holders_86_org.jpg + original

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 5.375618
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0309 / pillow_inserts_73 region / nearest pillow_inserts_73_org.jpg
        | query-centroid distance: 5.375618
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0000 / thermometers_85 region / nearest thermometers_85_org.jpg
        | query-centroid distance: 3.446152
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0336 / cord_management_36 region / nearest cord_management_36_org.jpg
        | query-centroid distance: 2.294097
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0022 / cartridge_razors_60 region / nearest cartridge_razors_60_org.jpg
        | query-centroid distance: 1.394034
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0086 / putties_25 region / nearest putties_25_org.jpg
        | query-centroid distance: 1.259755
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0086 / putties_25 region / nearest putties_25_org.jpg
        | query-centroid distance: 1.259755
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0010 / beard_mustache_trimmers_86 region / nearest beard_mustache_trimmers_86_org.jpg
        | query-centroid distance: 0.700052
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0010 / beard_mustache_trimmers_86 region / nearest beard_mustache_trimmers_86_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: flatware_32_org.jpg
        | graph distance: 0.708279  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: flatware_32_org.jpg
Direct duplicate: False | HNSW duplicate: False
Direct time: 39.145526 ms | HNSW time: 4.615253 ms | checked: 2797
```

##### ERNG

```text
QUERY: candlestick_holders_86_org.jpg + original

ERNG selected multi-probe path
Probe entry: C0445 / pillow_inserts_70 region / nearest pillow_inserts_70_org.jpg

        +--------------------------------------------------+
        | C0445 / pillow_inserts_70 region / nearest pillow_inserts_70_org.jpg
        | query-centroid distance: 8.28382
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0496 / stroller_connectors_100 region / nearest stroller_connectors_100_org.jpg
        | query-centroid distance: 2.558155
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0338 / seating_55 region / nearest seating_55_org.jpg
        | query-centroid distance: 0.848313
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0010 / beard_mustache_trimmers_86 region / nearest beard_mustache_trimmers_86_org.jpg
        | query-centroid distance: 0.700052
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0010 / beard_mustache_trimmers_86 region / nearest beard_mustache_trimmers_86_org.jpg
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: flatware_32_org.jpg
        | graph distance: 0.708279  threshold: 0.164388
        +--------------------------------------------------+
                         |
                         v
        NOT DUPLICATE

Direct matched image: flatware_32_org.jpg
Direct duplicate: False | ERNG duplicate: False
Direct time: 39.458028 ms | ERNG time: 4.610536 ms | checked: 2797
```

