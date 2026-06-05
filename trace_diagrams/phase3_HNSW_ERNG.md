# Attack Variant HNSW / ERNG Trace Diagrams With Exact Original Matches

Each example starts from an attacked image variant and reaches the exact original known-bank image. The graph index equals the direct dense-search original index for both HNSW and ERNG.

## CIFAR

### Attack Variant Example 1: `airplane_train_29944_org.png` attack `rotate_m5`

Reasoning: the `rotate_m5` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.478065`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: airplane_train_29944_org.png + rotate_m5

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / distance 1.292476
        | reason: stop_local_minimum
        | checked nearest centroids: C0506:1.474391, C0347:1.549571, C0472:1.631429, C0426:1.738202, C0498:1.764246
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / distance 1.292476
        | reason: entry
        | checked nearest centroids: C0378:0.90509, C0257:0.928262, C0383:0.968163, C0116:1.136034, C0311:1.197868
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0378 / airplane_pool region / distance 0.90509
        | reason: stop_local_minimum
        | checked nearest centroids: C0257:0.928262, C0112:0.933024, C0383:0.968163, C0135:1.005082, C0078:1.012728
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0378 / airplane_pool region / distance 0.90509
        | reason: entry
        | checked nearest centroids: C0271:0.211204, C0481:0.240685, C0025:0.33601, C0055:0.513776, C0184:0.587031
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0271 / airplane_pool region / distance 0.211204
        | reason: stop_local_minimum
        | checked nearest centroids: C0481:0.240685, C0025:0.33601, C0376:0.364744, C0055:0.513776, C0101:0.52283
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0271 / airplane_pool region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_29944_org.png
        | exact original index: 0
        | graph distance: 6e-06  threshold: 0.478065
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_29944_org.png
Direct duplicate: True | HNSW duplicate: True
Direct time: 26.84121 ms | HNSW time: 0.728473 ms | checked: 234
```

#### ERNG
```text
QUERY ATTACK VARIANT: airplane_train_29944_org.png + rotate_m5

ERNG multi-probe exponential-rank graph

Probe 1: start C0005
      |-- C0005 / airplane_pool region / distance 2.217242 [entry]
      |      checked: C0332:1.921669, C0420:1.962756, C0296:2.096125, C0411:2.116433
      |-- C0332 / airplane_pool region / distance 1.921669 [move_closer]
      |      checked: C0378:0.90509, C0383:0.968163, C0197:1.255217, C0048:1.292476
      |-- C0378 / airplane_pool region / distance 0.90509 [move_closer]
      |      checked: C0271:0.211204, C0481:0.240685, C0025:0.33601, C0055:0.513776
      |-- C0271 / airplane_pool region / distance 0.211204 [stop_local_minimum]
      |      checked: C0481:0.240685, C0025:0.33601, C0376:0.364744, C0055:0.513776
      `-- probe final: C0271 / distance 0.211204

Probe 2: start C0036
      |-- C0036 / airplane_pool region / distance 2.423698 [entry]
      |      checked: C0461:1.806209, C0491:2.006856, C0096:2.038674, C0005:2.217242
      |-- C0461 / airplane_pool region / distance 1.806209 [move_closer]
      |      checked: C0112:0.933024, C0452:1.30102, C0491:2.006856, C0021:2.027081
      |-- C0112 / airplane_pool region / distance 0.933024 [move_closer]
      |      checked: C0271:0.211204, C0481:0.240685, C0025:0.33601, C0376:0.364744
      |-- C0271 / airplane_pool region / distance 0.211204 [stop_local_minimum]
      |      checked: C0481:0.240685, C0025:0.33601, C0376:0.364744, C0055:0.513776
      `-- probe final: C0271 / distance 0.211204

Probe 3: start C0129
      |-- C0129 / airplane_pool region / distance 2.555749 [entry]
      |      checked: C0070:1.909995, C0057:2.16182, C0237:2.289533, C0253:2.379463
      |-- C0070 / airplane_pool region / distance 1.909995 [move_closer]
      |      checked: C0378:0.90509, C0346:1.362568, C0023:1.763492, C0206:1.790025
      |-- C0378 / airplane_pool region / distance 0.90509 [move_closer]
      |      checked: C0271:0.211204, C0481:0.240685, C0025:0.33601, C0055:0.513776
      |-- C0271 / airplane_pool region / distance 0.211204 [stop_local_minimum]
      |      checked: C0481:0.240685, C0025:0.33601, C0376:0.364744, C0055:0.513776
      `-- probe final: C0271 / distance 0.211204

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0271 / airplane_pool region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_29944_org.png
        | exact original index: 0
        | graph distance: 6e-06  threshold: 0.478065
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_29944_org.png
Direct duplicate: True | ERNG duplicate: True
Direct time: 12.25148 ms | ERNG time: 0.567761 ms | checked: 234
```

### Attack Variant Example 2: `airplane_train_29944_org.png` attack `rotate_m10`

Reasoning: the `rotate_m10` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.674946`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: airplane_train_29944_org.png + rotate_m10

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / distance 1.292476
        | reason: stop_local_minimum
        | checked nearest centroids: C0506:1.474391, C0347:1.549571, C0472:1.631429, C0426:1.738202, C0498:1.764246
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / distance 1.292476
        | reason: entry
        | checked nearest centroids: C0378:0.90509, C0257:0.928262, C0383:0.968163, C0116:1.136034, C0311:1.197868
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0378 / airplane_pool region / distance 0.90509
        | reason: stop_local_minimum
        | checked nearest centroids: C0257:0.928262, C0112:0.933024, C0383:0.968163, C0135:1.005082, C0078:1.012728
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0378 / airplane_pool region / distance 0.90509
        | reason: entry
        | checked nearest centroids: C0271:0.211204, C0481:0.240685, C0025:0.33601, C0055:0.513776, C0184:0.587031
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0271 / airplane_pool region / distance 0.211204
        | reason: stop_local_minimum
        | checked nearest centroids: C0481:0.240685, C0025:0.33601, C0376:0.364744, C0055:0.513776, C0101:0.52283
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0271 / airplane_pool region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_29944_org.png
        | exact original index: 0
        | graph distance: 6e-06  threshold: 0.674946
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_29944_org.png
Direct duplicate: True | HNSW duplicate: True
Direct time: 15.597964 ms | HNSW time: 0.618844 ms | checked: 234
```

#### ERNG
```text
QUERY ATTACK VARIANT: airplane_train_29944_org.png + rotate_m10

ERNG multi-probe exponential-rank graph

Probe 1: start C0005
      |-- C0005 / airplane_pool region / distance 2.217242 [entry]
      |      checked: C0332:1.921669, C0420:1.962756, C0296:2.096125, C0411:2.116433
      |-- C0332 / airplane_pool region / distance 1.921669 [move_closer]
      |      checked: C0378:0.90509, C0383:0.968163, C0197:1.255217, C0048:1.292476
      |-- C0378 / airplane_pool region / distance 0.90509 [move_closer]
      |      checked: C0271:0.211204, C0481:0.240685, C0025:0.33601, C0055:0.513776
      |-- C0271 / airplane_pool region / distance 0.211204 [stop_local_minimum]
      |      checked: C0481:0.240685, C0025:0.33601, C0376:0.364744, C0055:0.513776
      `-- probe final: C0271 / distance 0.211204

Probe 2: start C0036
      |-- C0036 / airplane_pool region / distance 2.423698 [entry]
      |      checked: C0461:1.806209, C0491:2.006856, C0096:2.038674, C0005:2.217242
      |-- C0461 / airplane_pool region / distance 1.806209 [move_closer]
      |      checked: C0112:0.933024, C0452:1.30102, C0491:2.006856, C0021:2.027081
      |-- C0112 / airplane_pool region / distance 0.933024 [move_closer]
      |      checked: C0271:0.211204, C0481:0.240685, C0025:0.33601, C0376:0.364744
      |-- C0271 / airplane_pool region / distance 0.211204 [stop_local_minimum]
      |      checked: C0481:0.240685, C0025:0.33601, C0376:0.364744, C0055:0.513776
      `-- probe final: C0271 / distance 0.211204

Probe 3: start C0129
      |-- C0129 / airplane_pool region / distance 2.555749 [entry]
      |      checked: C0070:1.909995, C0057:2.16182, C0237:2.289533, C0253:2.379463
      |-- C0070 / airplane_pool region / distance 1.909995 [move_closer]
      |      checked: C0378:0.90509, C0346:1.362568, C0023:1.763492, C0206:1.790025
      |-- C0378 / airplane_pool region / distance 0.90509 [move_closer]
      |      checked: C0271:0.211204, C0481:0.240685, C0025:0.33601, C0055:0.513776
      |-- C0271 / airplane_pool region / distance 0.211204 [stop_local_minimum]
      |      checked: C0481:0.240685, C0025:0.33601, C0376:0.364744, C0055:0.513776
      `-- probe final: C0271 / distance 0.211204

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0271 / airplane_pool region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_29944_org.png
        | exact original index: 0
        | graph distance: 6e-06  threshold: 0.674946
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_29944_org.png
Direct duplicate: True | ERNG duplicate: True
Direct time: 11.630163 ms | ERNG time: 0.527061 ms | checked: 234
```

### Attack Variant Example 3: `airplane_train_29944_org.png` attack `rotate5`

Reasoning: the `rotate5` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.506597`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: airplane_train_29944_org.png + rotate5

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / distance 1.292826
        | reason: stop_local_minimum
        | checked nearest centroids: C0506:1.479373, C0347:1.55892, C0472:1.643821, C0426:1.743973, C0498:1.764529
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0048 / airplane_pool region / distance 1.292826
        | reason: entry
        | checked nearest centroids: C0378:0.914193, C0257:0.92765, C0383:0.974101, C0116:1.134519, C0311:1.19609
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0378 / airplane_pool region / distance 0.914193
        | reason: stop_local_minimum
        | checked nearest centroids: C0257:0.92765, C0112:0.948466, C0383:0.974101, C0135:1.005269, C0078:1.01608
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0378 / airplane_pool region / distance 0.914193
        | reason: entry
        | checked nearest centroids: C0271:0.201866, C0481:0.267147, C0025:0.336648, C0055:0.536395, C0184:0.577184
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0271 / airplane_pool region / distance 0.201866
        | reason: stop_local_minimum
        | checked nearest centroids: C0481:0.267147, C0025:0.336648, C0376:0.373514, C0101:0.51595, C0055:0.536395
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0271 / airplane_pool region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_29944_org.png
        | exact original index: 0
        | graph distance: 0.030251  threshold: 0.506597
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_29944_org.png
Direct duplicate: True | HNSW duplicate: True
Direct time: 19.441669 ms | HNSW time: 0.625753 ms | checked: 234
```

#### ERNG
```text
QUERY ATTACK VARIANT: airplane_train_29944_org.png + rotate5

ERNG multi-probe exponential-rank graph

Probe 1: start C0005
      |-- C0005 / airplane_pool region / distance 2.228008 [entry]
      |      checked: C0332:1.931744, C0420:1.973046, C0296:2.106636, C0411:2.128153
      |-- C0332 / airplane_pool region / distance 1.931744 [move_closer]
      |      checked: C0378:0.914193, C0383:0.974101, C0197:1.258898, C0048:1.292826
      |-- C0378 / airplane_pool region / distance 0.914193 [move_closer]
      |      checked: C0271:0.201866, C0481:0.267147, C0025:0.336648, C0055:0.536395
      |-- C0271 / airplane_pool region / distance 0.201866 [stop_local_minimum]
      |      checked: C0481:0.267147, C0025:0.336648, C0376:0.373514, C0101:0.51595
      `-- probe final: C0271 / distance 0.201866

Probe 2: start C0036
      |-- C0036 / airplane_pool region / distance 2.435746 [entry]
      |      checked: C0461:1.818303, C0491:2.017196, C0096:2.044679, C0005:2.228008
      |-- C0461 / airplane_pool region / distance 1.818303 [move_closer]
      |      checked: C0112:0.948466, C0452:1.315288, C0491:2.017196, C0021:2.041135
      |-- C0112 / airplane_pool region / distance 0.948466 [move_closer]
      |      checked: C0271:0.201866, C0481:0.267147, C0025:0.336648, C0376:0.373514
      |-- C0271 / airplane_pool region / distance 0.201866 [stop_local_minimum]
      |      checked: C0481:0.267147, C0025:0.336648, C0376:0.373514, C0101:0.51595
      `-- probe final: C0271 / distance 0.201866

Probe 3: start C0129
      |-- C0129 / airplane_pool region / distance 2.564318 [entry]
      |      checked: C0070:1.916999, C0057:2.167959, C0237:2.295953, C0253:2.383817
      |-- C0070 / airplane_pool region / distance 1.916999 [move_closer]
      |      checked: C0378:0.914193, C0346:1.356631, C0023:1.762426, C0206:1.799698
      |-- C0378 / airplane_pool region / distance 0.914193 [move_closer]
      |      checked: C0271:0.201866, C0481:0.267147, C0025:0.336648, C0055:0.536395
      |-- C0271 / airplane_pool region / distance 0.201866 [stop_local_minimum]
      |      checked: C0481:0.267147, C0025:0.336648, C0376:0.373514, C0101:0.51595
      `-- probe final: C0271 / distance 0.201866

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0271 / airplane_pool region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: airplane_train_29944_org.png
        | exact original index: 0
        | graph distance: 0.030251  threshold: 0.506597
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: airplane_train_29944_org.png
Direct duplicate: True | ERNG duplicate: True
Direct time: 11.174601 ms | ERNG time: 0.513152 ms | checked: 234
```

## FLICKR

### Attack Variant Example 1: `2494088238.jpg` attack `rotate_m5`

Reasoning: the `rotate_m5` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.325554`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: 2494088238.jpg + rotate_m5

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0083 / flickr natural-image region / distance 3.234507
        | reason: stop_local_minimum
        | checked nearest centroids: C0139:3.406322, C0145:4.045293, C0150:4.214097, C0079:4.288479, C0026:4.713554
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0083 / flickr natural-image region / distance 3.234507
        | reason: entry
        | checked nearest centroids: C0058:1.985491, C0147:2.410418, C0004:2.591338, C0168:3.215424, C0139:3.406322
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0058 / flickr natural-image region / distance 1.985491
        | reason: stop_local_minimum
        | checked nearest centroids: C0147:2.410418, C0004:2.591338, C0168:3.215424, C0108:3.231726, C0083:3.234507
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0058 / flickr natural-image region / distance 1.985491
        | reason: entry
        | checked nearest centroids: C0092:0.763657, C0144:1.214988, C0010:1.637241, C0084:1.773111, C0147:2.410418
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0092 / flickr natural-image region / distance 0.763657
        | reason: stop_local_minimum
        | checked nearest centroids: C0144:1.214988, C0010:1.637241, C0048:1.663613, C0084:1.773111, C0098:1.910858
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0092 / flickr natural-image region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2494088238.jpg
        | exact original index: 0
        | graph distance: 0.008563  threshold: 0.325554
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2494088238.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 11.327733 ms | HNSW time: 0.293192 ms | checked: 79
```

#### ERNG
```text
QUERY ATTACK VARIANT: 2494088238.jpg + rotate_m5

ERNG multi-probe exponential-rank graph

Probe 1: start C0013
      |-- C0013 / flickr natural-image region / distance 8.622779 [entry]
      |      checked: C0097:6.008104, C0177:6.195439, C0037:6.487239, C0020:7.133799
      |-- C0097 / flickr natural-image region / distance 6.008104 [move_closer]
      |      checked: C0143:3.706248, C0026:4.713554, C0121:5.560345, C0140:5.56117
      |-- C0143 / flickr natural-image region / distance 3.706248 [move_closer]
      |      checked: C0058:1.985491, C0113:2.589946, C0132:2.92874, C0139:3.406322
      |-- C0058 / flickr natural-image region / distance 1.985491 [move_closer]
      |      checked: C0092:0.763657, C0144:1.214988, C0010:1.637241, C0084:1.773111
      |-- C0092 / flickr natural-image region / distance 0.763657 [stop_local_minimum]
      |      checked: C0144:1.214988, C0010:1.637241, C0048:1.663613, C0084:1.773111
      `-- probe final: C0092 / distance 0.763657

Probe 2: start C0026
      |-- C0026 / flickr natural-image region / distance 4.713554 [entry]
      |      checked: C0004:2.591338, C0083:3.234507, C0139:3.406322, C0145:4.045293
      |-- C0004 / flickr natural-image region / distance 2.591338 [move_closer]
      |      checked: C0144:1.214988, C0048:1.663613, C0063:2.213134, C0147:2.410418
      |-- C0144 / flickr natural-image region / distance 1.214988 [move_closer]
      |      checked: C0092:0.763657, C0010:1.637241, C0048:1.663613, C0084:1.773111
      |-- C0092 / flickr natural-image region / distance 0.763657 [stop_local_minimum]
      |      checked: C0144:1.214988, C0010:1.637241, C0048:1.663613, C0084:1.773111
      `-- probe final: C0092 / distance 0.763657

Probe 3: start C0097
      |-- C0097 / flickr natural-image region / distance 6.008104 [entry]
      |      checked: C0143:3.706248, C0026:4.713554, C0121:5.560345, C0140:5.56117
      |-- C0143 / flickr natural-image region / distance 3.706248 [move_closer]
      |      checked: C0058:1.985491, C0113:2.589946, C0132:2.92874, C0139:3.406322
      |-- C0058 / flickr natural-image region / distance 1.985491 [move_closer]
      |      checked: C0092:0.763657, C0144:1.214988, C0010:1.637241, C0084:1.773111
      |-- C0092 / flickr natural-image region / distance 0.763657 [stop_local_minimum]
      |      checked: C0144:1.214988, C0010:1.637241, C0048:1.663613, C0084:1.773111
      `-- probe final: C0092 / distance 0.763657

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0092 / flickr natural-image region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2494088238.jpg
        | exact original index: 0
        | graph distance: 0.008563  threshold: 0.325554
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2494088238.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 6.506595 ms | ERNG time: 0.252422 ms | checked: 79
```

### Attack Variant Example 2: `2494088238.jpg` attack `rotate_m10`

Reasoning: the `rotate_m10` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.370138`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: 2494088238.jpg + rotate_m10

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0083 / flickr natural-image region / distance 3.234507
        | reason: stop_local_minimum
        | checked nearest centroids: C0139:3.406322, C0145:4.045293, C0150:4.214097, C0079:4.288479, C0026:4.713554
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0083 / flickr natural-image region / distance 3.234507
        | reason: entry
        | checked nearest centroids: C0058:1.985491, C0147:2.410418, C0004:2.591338, C0168:3.215424, C0139:3.406322
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0058 / flickr natural-image region / distance 1.985491
        | reason: stop_local_minimum
        | checked nearest centroids: C0147:2.410418, C0004:2.591338, C0168:3.215424, C0108:3.231726, C0083:3.234507
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0058 / flickr natural-image region / distance 1.985491
        | reason: entry
        | checked nearest centroids: C0092:0.763657, C0144:1.214988, C0010:1.637241, C0084:1.773111, C0147:2.410418
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0092 / flickr natural-image region / distance 0.763657
        | reason: stop_local_minimum
        | checked nearest centroids: C0144:1.214988, C0010:1.637241, C0048:1.663613, C0084:1.773111, C0098:1.910858
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0092 / flickr natural-image region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2494088238.jpg
        | exact original index: 0
        | graph distance: 0.008563  threshold: 0.370138
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2494088238.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 10.699457 ms | HNSW time: 0.22396 ms | checked: 79
```

#### ERNG
```text
QUERY ATTACK VARIANT: 2494088238.jpg + rotate_m10

ERNG multi-probe exponential-rank graph

Probe 1: start C0013
      |-- C0013 / flickr natural-image region / distance 8.622779 [entry]
      |      checked: C0097:6.008104, C0177:6.195439, C0037:6.487239, C0020:7.133799
      |-- C0097 / flickr natural-image region / distance 6.008104 [move_closer]
      |      checked: C0143:3.706248, C0026:4.713554, C0121:5.560345, C0140:5.56117
      |-- C0143 / flickr natural-image region / distance 3.706248 [move_closer]
      |      checked: C0058:1.985491, C0113:2.589946, C0132:2.92874, C0139:3.406322
      |-- C0058 / flickr natural-image region / distance 1.985491 [move_closer]
      |      checked: C0092:0.763657, C0144:1.214988, C0010:1.637241, C0084:1.773111
      |-- C0092 / flickr natural-image region / distance 0.763657 [stop_local_minimum]
      |      checked: C0144:1.214988, C0010:1.637241, C0048:1.663613, C0084:1.773111
      `-- probe final: C0092 / distance 0.763657

Probe 2: start C0026
      |-- C0026 / flickr natural-image region / distance 4.713554 [entry]
      |      checked: C0004:2.591338, C0083:3.234507, C0139:3.406322, C0145:4.045293
      |-- C0004 / flickr natural-image region / distance 2.591338 [move_closer]
      |      checked: C0144:1.214988, C0048:1.663613, C0063:2.213134, C0147:2.410418
      |-- C0144 / flickr natural-image region / distance 1.214988 [move_closer]
      |      checked: C0092:0.763657, C0010:1.637241, C0048:1.663613, C0084:1.773111
      |-- C0092 / flickr natural-image region / distance 0.763657 [stop_local_minimum]
      |      checked: C0144:1.214988, C0010:1.637241, C0048:1.663613, C0084:1.773111
      `-- probe final: C0092 / distance 0.763657

Probe 3: start C0097
      |-- C0097 / flickr natural-image region / distance 6.008104 [entry]
      |      checked: C0143:3.706248, C0026:4.713554, C0121:5.560345, C0140:5.56117
      |-- C0143 / flickr natural-image region / distance 3.706248 [move_closer]
      |      checked: C0058:1.985491, C0113:2.589946, C0132:2.92874, C0139:3.406322
      |-- C0058 / flickr natural-image region / distance 1.985491 [move_closer]
      |      checked: C0092:0.763657, C0144:1.214988, C0010:1.637241, C0084:1.773111
      |-- C0092 / flickr natural-image region / distance 0.763657 [stop_local_minimum]
      |      checked: C0144:1.214988, C0010:1.637241, C0048:1.663613, C0084:1.773111
      `-- probe final: C0092 / distance 0.763657

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0092 / flickr natural-image region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2494088238.jpg
        | exact original index: 0
        | graph distance: 0.008563  threshold: 0.370138
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2494088238.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 6.353029 ms | ERNG time: 0.18418 ms | checked: 79
```

### Attack Variant Example 3: `2494088238.jpg` attack `rotate5`

Reasoning: the `rotate5` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.299346`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: 2494088238.jpg + rotate5

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0083 / flickr natural-image region / distance 3.204459
        | reason: stop_local_minimum
        | checked nearest centroids: C0139:3.372777, C0145:4.031436, C0150:4.172382, C0079:4.225019, C0026:4.674204
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0083 / flickr natural-image region / distance 3.204459
        | reason: entry
        | checked nearest centroids: C0058:2.020765, C0147:2.368314, C0004:2.525585, C0168:3.196131, C0139:3.372777
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0058 / flickr natural-image region / distance 2.020765
        | reason: stop_local_minimum
        | checked nearest centroids: C0147:2.368314, C0004:2.525585, C0168:3.196131, C0083:3.204459, C0108:3.252221
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0058 / flickr natural-image region / distance 2.020765
        | reason: entry
        | checked nearest centroids: C0092:0.776704, C0144:1.175128, C0010:1.6125, C0084:1.778579, C0147:2.368314
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0092 / flickr natural-image region / distance 0.776704
        | reason: stop_local_minimum
        | checked nearest centroids: C0144:1.175128, C0048:1.565417, C0010:1.6125, C0084:1.778579, C0098:1.929207
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0092 / flickr natural-image region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2494088238.jpg
        | exact original index: 0
        | graph distance: 0.10399  threshold: 0.299346
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2494088238.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 11.144343 ms | HNSW time: 0.282241 ms | checked: 79
```

#### ERNG
```text
QUERY ATTACK VARIANT: 2494088238.jpg + rotate5

ERNG multi-probe exponential-rank graph

Probe 1: start C0013
      |-- C0013 / flickr natural-image region / distance 8.607704 [entry]
      |      checked: C0097:5.96576, C0177:6.204725, C0037:6.476044, C0020:7.146208
      |-- C0097 / flickr natural-image region / distance 5.96576 [move_closer]
      |      checked: C0143:3.711715, C0026:4.674204, C0121:5.513837, C0140:5.529514
      |-- C0143 / flickr natural-image region / distance 3.711715 [move_closer]
      |      checked: C0058:2.020765, C0113:2.615315, C0132:2.877496, C0139:3.372777
      |-- C0058 / flickr natural-image region / distance 2.020765 [move_closer]
      |      checked: C0092:0.776704, C0144:1.175128, C0010:1.6125, C0084:1.778579
      |-- C0092 / flickr natural-image region / distance 0.776704 [stop_local_minimum]
      |      checked: C0144:1.175128, C0048:1.565417, C0010:1.6125, C0084:1.778579
      `-- probe final: C0092 / distance 0.776704

Probe 2: start C0026
      |-- C0026 / flickr natural-image region / distance 4.674204 [entry]
      |      checked: C0004:2.525585, C0083:3.204459, C0139:3.372777, C0145:4.031436
      |-- C0004 / flickr natural-image region / distance 2.525585 [move_closer]
      |      checked: C0144:1.175128, C0048:1.565417, C0063:2.135701, C0147:2.368314
      |-- C0144 / flickr natural-image region / distance 1.175128 [move_closer]
      |      checked: C0092:0.776704, C0048:1.565417, C0010:1.6125, C0084:1.778579
      |-- C0092 / flickr natural-image region / distance 0.776704 [stop_local_minimum]
      |      checked: C0144:1.175128, C0048:1.565417, C0010:1.6125, C0084:1.778579
      `-- probe final: C0092 / distance 0.776704

Probe 3: start C0097
      |-- C0097 / flickr natural-image region / distance 5.96576 [entry]
      |      checked: C0143:3.711715, C0026:4.674204, C0121:5.513837, C0140:5.529514
      |-- C0143 / flickr natural-image region / distance 3.711715 [move_closer]
      |      checked: C0058:2.020765, C0113:2.615315, C0132:2.877496, C0139:3.372777
      |-- C0058 / flickr natural-image region / distance 2.020765 [move_closer]
      |      checked: C0092:0.776704, C0144:1.175128, C0010:1.6125, C0084:1.778579
      |-- C0092 / flickr natural-image region / distance 0.776704 [stop_local_minimum]
      |      checked: C0144:1.175128, C0048:1.565417, C0010:1.6125, C0084:1.778579
      `-- probe final: C0092 / distance 0.776704

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0092 / flickr natural-image region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 2494088238.jpg
        | exact original index: 0
        | graph distance: 0.10399  threshold: 0.299346
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 2494088238.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 6.749199 ms | ERNG time: 0.41106 ms | checked: 79
```

## UCID

### Attack Variant Example 1: `60_aug_3.jpg` attack `rotate_m5`

Reasoning: the `rotate_m5` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.547669`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: 60_aug_3.jpg + rotate_m5

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0043 / ucid scene region / distance 13.168694
        | reason: stop_local_minimum
        | checked nearest centroids: C0044:14.124324, C0110:14.161181, C0005:14.339123, C0098:14.454134, C0085:14.74296
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0043 / ucid scene region / distance 13.168694
        | reason: entry
        | checked nearest centroids: C0066:11.391785, C0090:11.614844, C0034:11.989266, C0026:12.062479, C0024:13.04974
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0066 / ucid scene region / distance 11.391785
        | reason: move_closer
        | checked nearest centroids: C0003:10.566667, C0095:11.174939, C0090:11.614844, C0069:11.831499, C0034:11.989266
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0003 / ucid scene region / distance 10.566667
        | reason: stop_local_minimum
        | checked nearest centroids: C0095:11.174939, C0066:11.391785, C0090:11.614844, C0069:11.831499, C0034:11.989266
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0003 / ucid scene region / distance 10.566667
        | reason: entry
        | checked nearest centroids: C0122:9.449332, C0060:9.604359, C0103:9.952814, C0062:10.003445, C0052:10.426368
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0122 / ucid scene region / distance 9.449332
        | reason: move_closer
        | checked nearest centroids: C0008:7.516869, C0075:7.959181, C0037:8.116124, C0064:8.155673, C0020:8.690041
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0008 / ucid scene region / distance 7.516869
        | reason: move_closer
        | checked nearest centroids: C0030:5.868537, C0106:6.341735, C0021:6.567077, C0058:6.953505, C0087:7.334362
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0030 / ucid scene region / distance 5.868537
        | reason: move_closer
        | checked nearest centroids: C0125:3.440338, C0013:4.532736, C0071:4.55437, C0061:5.471447, C0072:5.807949
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0125 / ucid scene region / distance 3.440338
        | reason: move_closer
        | checked nearest centroids: C0101:0.545404, C0042:1.398839, C0028:1.651695, C0104:2.03558, C0000:2.368189
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0101 / ucid scene region / distance 0.545404
        | reason: stop_local_minimum
        | checked nearest centroids: C0014:0.991709, C0074:1.367468, C0042:1.398839, C0028:1.651695, C0056:1.766366
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0101 / ucid scene region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 60_aug_3.jpg
        | exact original index: 0
        | graph distance: 0.083898  threshold: 0.547669
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 60_aug_3.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 6.785963 ms | HNSW time: 0.344926 ms | checked: 37
```

#### ERNG
```text
QUERY ATTACK VARIANT: 60_aug_3.jpg + rotate_m5

ERNG multi-probe exponential-rank graph

Probe 1: start C0009
      |-- C0009 / ucid scene region / distance 19.667091 [entry]
      |      checked: C0116:14.940717, C0049:18.594194, C0019:18.983662, C0002:18.984549
      |-- C0116 / ucid scene region / distance 14.940717 [move_closer]
      |      checked: C0097:12.195786, C0043:13.168694, C0119:13.567439, C0044:14.124324
      |-- C0097 / ucid scene region / distance 12.195786 [move_closer]
      |      checked: C0062:10.003445, C0003:10.566667, C0095:11.174939, C0066:11.391785
      |-- C0062 / ucid scene region / distance 10.003445 [move_closer]
      |      checked: C0075:7.959181, C0020:8.690041, C0016:9.094838, C0122:9.449332
      |-- C0075 / ucid scene region / distance 7.959181 [move_closer]
      |      checked: C0014:0.991709, C0030:5.868537, C0081:6.157603, C0106:6.341735
      |-- C0014 / ucid scene region / distance 0.991709 [move_closer]
      |      checked: C0101:0.545404, C0074:1.367468, C0042:1.398839, C0028:1.651695
      |-- C0101 / ucid scene region / distance 0.545404 [stop_local_minimum]
      |      checked: C0014:0.991709, C0074:1.367468, C0042:1.398839, C0028:1.651695
      `-- probe final: C0101 / distance 0.545404

Probe 2: start C0031
      |-- C0031 / ucid scene region / distance 15.066999 [entry]
      |      checked: C0052:10.426368, C0026:12.062479, C0044:14.124324, C0110:14.161181
      |-- C0052 / ucid scene region / distance 10.426368 [move_closer]
      |      checked: C0105:3.400798, C0106:6.341735, C0020:8.690041, C0122:9.449332
      |-- C0105 / ucid scene region / distance 3.400798 [move_closer]
      |      checked: C0101:0.545404, C0042:1.398839, C0028:1.651695, C0104:2.03558
      |-- C0101 / ucid scene region / distance 0.545404 [stop_local_minimum]
      |      checked: C0014:0.991709, C0074:1.367468, C0042:1.398839, C0028:1.651695
      `-- probe final: C0101 / distance 0.545404

Probe 3: start C0044
      |-- C0044 / ucid scene region / distance 14.124324 [entry]
      |      checked: C0020:8.690041, C0024:13.04974, C0053:13.089617, C0043:13.168694
      |-- C0020 / ucid scene region / distance 8.690041 [move_closer]
      |      checked: C0087:7.334362, C0008:7.516869, C0124:7.571045, C0075:7.959181
      |-- C0087 / ucid scene region / distance 7.334362 [move_closer]
      |      checked: C0105:3.400798, C0061:5.471447, C0072:5.807949, C0030:5.868537
      |-- C0105 / ucid scene region / distance 3.400798 [move_closer]
      |      checked: C0101:0.545404, C0042:1.398839, C0028:1.651695, C0104:2.03558
      |-- C0101 / ucid scene region / distance 0.545404 [stop_local_minimum]
      |      checked: C0014:0.991709, C0074:1.367468, C0042:1.398839, C0028:1.651695
      `-- probe final: C0101 / distance 0.545404

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0101 / ucid scene region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 60_aug_3.jpg
        | exact original index: 0
        | graph distance: 0.083898  threshold: 0.547669
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 60_aug_3.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 3.526716 ms | ERNG time: 0.283197 ms | checked: 37
```

### Attack Variant Example 2: `60_aug_3.jpg` attack `rotate_m10`

Reasoning: the `rotate_m10` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.593123`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: 60_aug_3.jpg + rotate_m10

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0043 / ucid scene region / distance 13.168694
        | reason: stop_local_minimum
        | checked nearest centroids: C0044:14.124324, C0110:14.161181, C0005:14.339123, C0098:14.454134, C0085:14.74296
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0043 / ucid scene region / distance 13.168694
        | reason: entry
        | checked nearest centroids: C0066:11.391785, C0090:11.614844, C0034:11.989266, C0026:12.062479, C0024:13.04974
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0066 / ucid scene region / distance 11.391785
        | reason: move_closer
        | checked nearest centroids: C0003:10.566667, C0095:11.174939, C0090:11.614844, C0069:11.831499, C0034:11.989266
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0003 / ucid scene region / distance 10.566667
        | reason: stop_local_minimum
        | checked nearest centroids: C0095:11.174939, C0066:11.391785, C0090:11.614844, C0069:11.831499, C0034:11.989266
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0003 / ucid scene region / distance 10.566667
        | reason: entry
        | checked nearest centroids: C0122:9.449332, C0060:9.604359, C0103:9.952814, C0062:10.003445, C0052:10.426368
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0122 / ucid scene region / distance 9.449332
        | reason: move_closer
        | checked nearest centroids: C0008:7.516869, C0075:7.959181, C0037:8.116124, C0064:8.155673, C0020:8.690041
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0008 / ucid scene region / distance 7.516869
        | reason: move_closer
        | checked nearest centroids: C0030:5.868537, C0106:6.341735, C0021:6.567077, C0058:6.953505, C0087:7.334362
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0030 / ucid scene region / distance 5.868537
        | reason: move_closer
        | checked nearest centroids: C0125:3.440338, C0013:4.532736, C0071:4.55437, C0061:5.471447, C0072:5.807949
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0125 / ucid scene region / distance 3.440338
        | reason: move_closer
        | checked nearest centroids: C0101:0.545404, C0042:1.398839, C0028:1.651695, C0104:2.03558, C0000:2.368189
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0101 / ucid scene region / distance 0.545404
        | reason: stop_local_minimum
        | checked nearest centroids: C0014:0.991709, C0074:1.367468, C0042:1.398839, C0028:1.651695, C0056:1.766366
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0101 / ucid scene region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 60_aug_3.jpg
        | exact original index: 0
        | graph distance: 0.083898  threshold: 0.593123
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 60_aug_3.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 3.262395 ms | HNSW time: 0.230968 ms | checked: 37
```

#### ERNG
```text
QUERY ATTACK VARIANT: 60_aug_3.jpg + rotate_m10

ERNG multi-probe exponential-rank graph

Probe 1: start C0009
      |-- C0009 / ucid scene region / distance 19.667091 [entry]
      |      checked: C0116:14.940717, C0049:18.594194, C0019:18.983662, C0002:18.984549
      |-- C0116 / ucid scene region / distance 14.940717 [move_closer]
      |      checked: C0097:12.195786, C0043:13.168694, C0119:13.567439, C0044:14.124324
      |-- C0097 / ucid scene region / distance 12.195786 [move_closer]
      |      checked: C0062:10.003445, C0003:10.566667, C0095:11.174939, C0066:11.391785
      |-- C0062 / ucid scene region / distance 10.003445 [move_closer]
      |      checked: C0075:7.959181, C0020:8.690041, C0016:9.094838, C0122:9.449332
      |-- C0075 / ucid scene region / distance 7.959181 [move_closer]
      |      checked: C0014:0.991709, C0030:5.868537, C0081:6.157603, C0106:6.341735
      |-- C0014 / ucid scene region / distance 0.991709 [move_closer]
      |      checked: C0101:0.545404, C0074:1.367468, C0042:1.398839, C0028:1.651695
      |-- C0101 / ucid scene region / distance 0.545404 [stop_local_minimum]
      |      checked: C0014:0.991709, C0074:1.367468, C0042:1.398839, C0028:1.651695
      `-- probe final: C0101 / distance 0.545404

Probe 2: start C0031
      |-- C0031 / ucid scene region / distance 15.066999 [entry]
      |      checked: C0052:10.426368, C0026:12.062479, C0044:14.124324, C0110:14.161181
      |-- C0052 / ucid scene region / distance 10.426368 [move_closer]
      |      checked: C0105:3.400798, C0106:6.341735, C0020:8.690041, C0122:9.449332
      |-- C0105 / ucid scene region / distance 3.400798 [move_closer]
      |      checked: C0101:0.545404, C0042:1.398839, C0028:1.651695, C0104:2.03558
      |-- C0101 / ucid scene region / distance 0.545404 [stop_local_minimum]
      |      checked: C0014:0.991709, C0074:1.367468, C0042:1.398839, C0028:1.651695
      `-- probe final: C0101 / distance 0.545404

Probe 3: start C0044
      |-- C0044 / ucid scene region / distance 14.124324 [entry]
      |      checked: C0020:8.690041, C0024:13.04974, C0053:13.089617, C0043:13.168694
      |-- C0020 / ucid scene region / distance 8.690041 [move_closer]
      |      checked: C0087:7.334362, C0008:7.516869, C0124:7.571045, C0075:7.959181
      |-- C0087 / ucid scene region / distance 7.334362 [move_closer]
      |      checked: C0105:3.400798, C0061:5.471447, C0072:5.807949, C0030:5.868537
      |-- C0105 / ucid scene region / distance 3.400798 [move_closer]
      |      checked: C0101:0.545404, C0042:1.398839, C0028:1.651695, C0104:2.03558
      |-- C0101 / ucid scene region / distance 0.545404 [stop_local_minimum]
      |      checked: C0014:0.991709, C0074:1.367468, C0042:1.398839, C0028:1.651695
      `-- probe final: C0101 / distance 0.545404

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0101 / ucid scene region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 60_aug_3.jpg
        | exact original index: 0
        | graph distance: 0.083898  threshold: 0.593123
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 60_aug_3.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 3.259887 ms | ERNG time: 0.191296 ms | checked: 37
```

### Attack Variant Example 3: `60_aug_3.jpg` attack `rotate5`

Reasoning: the `rotate5` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.568879`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: 60_aug_3.jpg + rotate5

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0043 / ucid scene region / distance 13.178079
        | reason: stop_local_minimum
        | checked nearest centroids: C0044:14.132773, C0110:14.163018, C0005:14.349863, C0098:14.468727, C0085:14.751662
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0043 / ucid scene region / distance 13.178079
        | reason: entry
        | checked nearest centroids: C0066:11.407456, C0090:11.624999, C0034:11.993272, C0026:12.071597, C0024:13.054691
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0066 / ucid scene region / distance 11.407456
        | reason: move_closer
        | checked nearest centroids: C0003:10.570126, C0095:11.181806, C0090:11.624999, C0069:11.8318, C0034:11.993272
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0003 / ucid scene region / distance 10.570126
        | reason: stop_local_minimum
        | checked nearest centroids: C0095:11.181806, C0066:11.407456, C0090:11.624999, C0069:11.8318, C0034:11.993272
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0003 / ucid scene region / distance 10.570126
        | reason: entry
        | checked nearest centroids: C0122:9.458828, C0060:9.601945, C0103:9.961019, C0062:10.016636, C0052:10.427211
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0122 / ucid scene region / distance 9.458828
        | reason: move_closer
        | checked nearest centroids: C0008:7.533944, C0075:7.971166, C0037:8.114096, C0064:8.164117, C0020:8.698948
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0008 / ucid scene region / distance 7.533944
        | reason: move_closer
        | checked nearest centroids: C0030:5.884866, C0106:6.359777, C0021:6.570617, C0058:6.947521, C0087:7.338997
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0030 / ucid scene region / distance 5.884866
        | reason: move_closer
        | checked nearest centroids: C0125:3.434746, C0013:4.517236, C0071:4.565736, C0061:5.468366, C0072:5.790061
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0125 / ucid scene region / distance 3.434746
        | reason: move_closer
        | checked nearest centroids: C0101:0.534045, C0042:1.4077, C0028:1.593994, C0104:2.029949, C0000:2.328766
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0101 / ucid scene region / distance 0.534045
        | reason: stop_local_minimum
        | checked nearest centroids: C0014:0.950838, C0074:1.280506, C0042:1.4077, C0028:1.593994, C0056:1.695768
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0101 / ucid scene region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 60_aug_3.jpg
        | exact original index: 0
        | graph distance: 0.206645  threshold: 0.568879
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 60_aug_3.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 3.419169 ms | HNSW time: 0.246839 ms | checked: 37
```

#### ERNG
```text
QUERY ATTACK VARIANT: 60_aug_3.jpg + rotate5

ERNG multi-probe exponential-rank graph

Probe 1: start C0009
      |-- C0009 / ucid scene region / distance 19.677547 [entry]
      |      checked: C0116:14.941721, C0049:18.606871, C0002:18.993832, C0019:18.995859
      |-- C0116 / ucid scene region / distance 14.941721 [move_closer]
      |      checked: C0097:12.206278, C0043:13.178079, C0119:13.570727, C0044:14.132773
      |-- C0097 / ucid scene region / distance 12.206278 [move_closer]
      |      checked: C0062:10.016636, C0003:10.570126, C0095:11.181806, C0066:11.407456
      |-- C0062 / ucid scene region / distance 10.016636 [move_closer]
      |      checked: C0075:7.971166, C0020:8.698948, C0016:9.111478, C0122:9.458828
      |-- C0075 / ucid scene region / distance 7.971166 [move_closer]
      |      checked: C0014:0.950838, C0030:5.884866, C0081:6.164608, C0106:6.359777
      |-- C0014 / ucid scene region / distance 0.950838 [move_closer]
      |      checked: C0101:0.534045, C0074:1.280506, C0042:1.4077, C0028:1.593994
      |-- C0101 / ucid scene region / distance 0.534045 [stop_local_minimum]
      |      checked: C0014:0.950838, C0074:1.280506, C0042:1.4077, C0028:1.593994
      `-- probe final: C0101 / distance 0.534045

Probe 2: start C0031
      |-- C0031 / ucid scene region / distance 15.076795 [entry]
      |      checked: C0052:10.427211, C0026:12.071597, C0044:14.132773, C0110:14.163018
      |-- C0052 / ucid scene region / distance 10.427211 [move_closer]
      |      checked: C0105:3.373335, C0106:6.359777, C0020:8.698948, C0122:9.458828
      |-- C0105 / ucid scene region / distance 3.373335 [move_closer]
      |      checked: C0101:0.534045, C0042:1.4077, C0028:1.593994, C0104:2.029949
      |-- C0101 / ucid scene region / distance 0.534045 [stop_local_minimum]
      |      checked: C0014:0.950838, C0074:1.280506, C0042:1.4077, C0028:1.593994
      `-- probe final: C0101 / distance 0.534045

Probe 3: start C0044
      |-- C0044 / ucid scene region / distance 14.132773 [entry]
      |      checked: C0020:8.698948, C0024:13.054691, C0053:13.103786, C0043:13.178079
      |-- C0020 / ucid scene region / distance 8.698948 [move_closer]
      |      checked: C0087:7.338997, C0008:7.533944, C0124:7.588544, C0075:7.971166
      |-- C0087 / ucid scene region / distance 7.338997 [move_closer]
      |      checked: C0105:3.373335, C0061:5.468366, C0072:5.790061, C0030:5.884866
      |-- C0105 / ucid scene region / distance 3.373335 [move_closer]
      |      checked: C0101:0.534045, C0042:1.4077, C0028:1.593994, C0104:2.029949
      |-- C0101 / ucid scene region / distance 0.534045 [stop_local_minimum]
      |      checked: C0014:0.950838, C0074:1.280506, C0042:1.4077, C0028:1.593994
      `-- probe final: C0101 / distance 0.534045

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0101 / ucid scene region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: 60_aug_3.jpg
        | exact original index: 0
        | graph distance: 0.206645  threshold: 0.568879
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: 60_aug_3.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 3.133582 ms | ERNG time: 0.211143 ms | checked: 37
```

## AMAZON

### Attack Variant Example 1: `creams_lotions_32_org.jpg` attack `rotate_m5`

Reasoning: the `rotate_m5` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.890214`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: creams_lotions_32_org.jpg + rotate_m5

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015
        | reason: stop_local_minimum
        | checked nearest centroids: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926, C0407:0.920295
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015
        | reason: stop_local_minimum
        | checked nearest centroids: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926, C0407:0.920295
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015
        | reason: stop_local_minimum
        | checked nearest centroids: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926, C0407:0.920295
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0401 / creams_lotions / creams_lotions_32 product region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: creams_lotions_32_org.jpg
        | exact original index: 0
        | graph distance: 5e-06  threshold: 0.890214
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: creams_lotions_32_org.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 70.308804 ms | HNSW time: 1.054339 ms | checked: 148
```

#### ERNG
```text
QUERY ATTACK VARIANT: creams_lotions_32_org.jpg + rotate_m5

ERNG multi-probe exponential-rank graph

Probe 1: start C0036
      |-- C0036 / creams_lotions / creams_lotions_32 product region / distance 20.32082 [entry]
      |      checked: C0130:0.903926, C0392:11.535015, C0391:18.075657, C0012:18.805935
      |-- C0130 / creams_lotions / creams_lotions_32 product region / distance 0.903926 [move_closer]
      |      checked: C0401:0.46015, C0045:0.606521, C0176:0.88946, C0103:1.038411
      |-- C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015 [stop_local_minimum]
      |      checked: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926
      `-- probe final: C0401 / distance 0.46015

Probe 2: start C0130
      |-- C0130 / creams_lotions / creams_lotions_32 product region / distance 0.903926 [entry]
      |      checked: C0401:0.46015, C0045:0.606521, C0176:0.88946, C0103:1.038411
      |-- C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015 [stop_local_minimum]
      |      checked: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926
      `-- probe final: C0401 / distance 0.46015

Probe 3: start C0191
      |-- C0191 / creams_lotions / creams_lotions_32 product region / distance 1.172441 [entry]
      |      checked: C0401:0.46015, C0045:0.606521, C0312:0.885446, C0130:0.903926
      |-- C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015 [stop_local_minimum]
      |      checked: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926
      `-- probe final: C0401 / distance 0.46015

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0401 / creams_lotions / creams_lotions_32 product region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: creams_lotions_32_org.jpg
        | exact original index: 0
        | graph distance: 5e-06  threshold: 0.890214
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: creams_lotions_32_org.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 72.826222 ms | ERNG time: 0.930092 ms | checked: 148
```

### Attack Variant Example 2: `creams_lotions_32_org.jpg` attack `rotate_m10`

Reasoning: the `rotate_m10` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.836095`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: creams_lotions_32_org.jpg + rotate_m10

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015
        | reason: stop_local_minimum
        | checked nearest centroids: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926, C0407:0.920295
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015
        | reason: stop_local_minimum
        | checked nearest centroids: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926, C0407:0.920295
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015
        | reason: stop_local_minimum
        | checked nearest centroids: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926, C0407:0.920295
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0401 / creams_lotions / creams_lotions_32 product region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: creams_lotions_32_org.jpg
        | exact original index: 0
        | graph distance: 5e-06  threshold: 0.836095
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: creams_lotions_32_org.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 58.453986 ms | HNSW time: 0.758595 ms | checked: 148
```

#### ERNG
```text
QUERY ATTACK VARIANT: creams_lotions_32_org.jpg + rotate_m10

ERNG multi-probe exponential-rank graph

Probe 1: start C0036
      |-- C0036 / creams_lotions / creams_lotions_32 product region / distance 20.32082 [entry]
      |      checked: C0130:0.903926, C0392:11.535015, C0391:18.075657, C0012:18.805935
      |-- C0130 / creams_lotions / creams_lotions_32 product region / distance 0.903926 [move_closer]
      |      checked: C0401:0.46015, C0045:0.606521, C0176:0.88946, C0103:1.038411
      |-- C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015 [stop_local_minimum]
      |      checked: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926
      `-- probe final: C0401 / distance 0.46015

Probe 2: start C0130
      |-- C0130 / creams_lotions / creams_lotions_32 product region / distance 0.903926 [entry]
      |      checked: C0401:0.46015, C0045:0.606521, C0176:0.88946, C0103:1.038411
      |-- C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015 [stop_local_minimum]
      |      checked: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926
      `-- probe final: C0401 / distance 0.46015

Probe 3: start C0191
      |-- C0191 / creams_lotions / creams_lotions_32 product region / distance 1.172441 [entry]
      |      checked: C0401:0.46015, C0045:0.606521, C0312:0.885446, C0130:0.903926
      |-- C0401 / creams_lotions / creams_lotions_32 product region / distance 0.46015 [stop_local_minimum]
      |      checked: C0045:0.606521, C0312:0.885446, C0176:0.88946, C0130:0.903926
      `-- probe final: C0401 / distance 0.46015

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0401 / creams_lotions / creams_lotions_32 product region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: creams_lotions_32_org.jpg
        | exact original index: 0
        | graph distance: 5e-06  threshold: 0.836095
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: creams_lotions_32_org.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 57.06613 ms | ERNG time: 0.711995 ms | checked: 148
```

### Attack Variant Example 3: `sneakers-40_org.jpg` attack `rotate5`

Reasoning: the `rotate5` variant is routed through centroid neighborhoods, then the final candidate set is checked using the same calibrated threshold `0.815317`. The final graph match is the original image itself.

#### HNSW
```text
QUERY ATTACK VARIANT: sneakers-40_org.jpg + rotate5

HNSW Layer 2 - broad shortcut layer

        +--------------------------------------------------+
        | C0309 / sneakers / sneakers-40 product region / distance 4.424276
        | reason: stop_local_minimum
        | checked nearest centroids: C0254:4.46072, C0019:4.604105, C0483:4.906303, C0494:5.129951, C0343:5.180554
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 1 - medium visual-neighborhood layer

        +--------------------------------------------------+
        | C0309 / sneakers / sneakers-40 product region / distance 4.424276
        | reason: entry
        | checked nearest centroids: C0000:2.464501, C0138:3.362898, C0063:3.448399, C0379:3.499936, C0505:3.752417
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0000 / sneakers / sneakers-40 product region / distance 2.464501
        | reason: move_closer
        | checked nearest centroids: C0336:1.11354, C0051:1.509693, C0090:1.606623, C0356:1.698634, C0410:1.941929
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0336 / sneakers / sneakers-40 product region / distance 1.11354
        | reason: move_closer
        | checked nearest centroids: C0345:0.631772, C0022:0.77393, C0456:0.855068, C0347:0.920245, C0459:1.148409
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | C0345 / sneakers / sneakers-40 product region / distance 0.631772
        | reason: stop_local_minimum
        | checked nearest centroids: C0022:0.77393, C0456:0.855068, C0347:0.920245, C0336:1.11354, C0459:1.148409
        +--------------------------------------------------+
                         |
                         v

HNSW Layer 0 - dense local centroid layer

        +--------------------------------------------------+
        | C0345 / sneakers / sneakers-40 product region / distance 0.631772
        | reason: stop_local_minimum
        | checked nearest centroids: C0040:0.683761, C0022:0.77393, C0456:0.855068, C0347:0.920245, C0064:0.95737
        +--------------------------------------------------+
                         |
                         v

        +--------------------------------------------------+
        | FINAL CENTROID REGION: C0345 / sneakers / sneakers-40 product region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: sneakers-40_org.jpg
        | exact original index: 175
        | graph distance: 0.546553  threshold: 0.815317
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: sneakers-40_org.jpg
Direct duplicate: True | HNSW duplicate: True
Direct time: 64.703477 ms | HNSW time: 0.852277 ms | checked: 146
```

#### ERNG
```text
QUERY ATTACK VARIANT: sneakers-40_org.jpg + rotate5

ERNG multi-probe exponential-rank graph

Probe 1: start C0036
      |-- C0036 / sneakers / sneakers-40 product region / distance 28.444408 [entry]
      |      checked: C0130:7.630435, C0392:19.60854, C0391:26.173988, C0012:26.888096
      |-- C0130 / sneakers / sneakers-40 product region / distance 7.630435 [move_closer]
      |      checked: C0293:3.524018, C0510:5.218757, C0218:6.267121, C0256:6.591033
      |-- C0293 / sneakers / sneakers-40 product region / distance 3.524018 [move_closer]
      |      checked: C0351:2.111485, C0417:2.76878, C0107:2.912606, C0006:3.063865
      |-- C0351 / sneakers / sneakers-40 product region / distance 2.111485 [move_closer]
      |      checked: C0241:1.053876, C0010:1.577152, C0365:1.604618, C0105:1.805119
      |-- C0241 / sneakers / sneakers-40 product region / distance 1.053876 [move_closer]
      |      checked: C0345:0.631772, C0022:0.77393, C0347:0.920245, C0064:0.95737
      |-- C0345 / sneakers / sneakers-40 product region / distance 0.631772 [stop_local_minimum]
      |      checked: C0040:0.683761, C0022:0.77393, C0456:0.855068, C0347:0.920245
      `-- probe final: C0345 / distance 0.631772

Probe 2: start C0130
      |-- C0130 / sneakers / sneakers-40 product region / distance 7.630435 [entry]
      |      checked: C0293:3.524018, C0510:5.218757, C0218:6.267121, C0256:6.591033
      |-- C0293 / sneakers / sneakers-40 product region / distance 3.524018 [move_closer]
      |      checked: C0351:2.111485, C0417:2.76878, C0107:2.912606, C0006:3.063865
      |-- C0351 / sneakers / sneakers-40 product region / distance 2.111485 [move_closer]
      |      checked: C0241:1.053876, C0010:1.577152, C0365:1.604618, C0105:1.805119
      |-- C0241 / sneakers / sneakers-40 product region / distance 1.053876 [move_closer]
      |      checked: C0345:0.631772, C0022:0.77393, C0347:0.920245, C0064:0.95737
      |-- C0345 / sneakers / sneakers-40 product region / distance 0.631772 [stop_local_minimum]
      |      checked: C0040:0.683761, C0022:0.77393, C0456:0.855068, C0347:0.920245
      `-- probe final: C0345 / distance 0.631772

Probe 3: start C0191
      |-- C0191 / sneakers / sneakers-40 product region / distance 7.684644 [entry]
      |      checked: C0025:4.633135, C0326:4.950647, C0256:6.591033, C0066:6.632754
      |-- C0025 / sneakers / sneakers-40 product region / distance 4.633135 [move_closer]
      |      checked: C0011:2.060893, C0286:2.411325, C0029:2.778284, C0053:3.187413
      |-- C0011 / sneakers / sneakers-40 product region / distance 2.060893 [move_closer]
      |      checked: C0283:1.279454, C0354:1.316592, C0051:1.509693, C0356:1.698634
      |-- C0283 / sneakers / sneakers-40 product region / distance 1.279454 [move_closer]
      |      checked: C0040:0.683761, C0347:0.920245, C0241:1.053876, C0336:1.11354
      |-- C0040 / sneakers / sneakers-40 product region / distance 0.683761 [move_closer]
      |      checked: C0345:0.631772, C0022:0.77393, C0456:0.855068, C0347:0.920245
      |-- C0345 / sneakers / sneakers-40 product region / distance 0.631772 [stop_local_minimum]
      |      checked: C0040:0.683761, C0022:0.77393, C0456:0.855068, C0347:0.920245
      `-- probe final: C0345 / distance 0.631772

        +--------------------------------------------------+
        | SELECTED FINAL CENTROID: C0345 / sneakers / sneakers-40 product region
        +--------------------------------------------------+
                         |
                         v
        +--------------------------------------------------+
        | MATCHED KNOWN IMAGE: sneakers-40_org.jpg
        | exact original index: 175
        | graph distance: 0.546553  threshold: 0.815317
        +--------------------------------------------------+
                         |
                         v
        DUPLICATE DETECTED

Direct matched image: sneakers-40_org.jpg
Direct duplicate: True | ERNG duplicate: True
Direct time: 59.166281 ms | ERNG time: 0.70151 ms | checked: 146
```
