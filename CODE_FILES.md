# Final Important Code Files

All paths below are local paths inside the final submission package.

## Core Training Files

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase\training\config.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\training\model.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\training\model_convnext.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\training\model_resnet.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\training\model_vit.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\training\train.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\training\run_train_model.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\training\utils.py
```

## Preprocessing / Attacks / Manifest Files

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase\preprocessing\attacks.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\preprocessing\build_manifest.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\preprocessing\dataloader.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\preprocessing\dataset.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\preprocessing\manifest.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\preprocessing\manifest_builder.py
```

## Store / Embedding Bank Files

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase\store\embedding_store.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\store\store_manager.py
```

## Evaluation / Threshold Calibration Files

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase\evaluation\evaluation.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\evaluation\eval_store.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\evaluation\run_eval_store.py
```

## Repair / Geometry Adjustment Files

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase\repair\geometry_repair.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\repair\run_repair_geometry.py
```

## Phase-3 Graph Query / HNSW / ERNG Files

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\all_dataset_graph_benchmark.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\centroid_branch_bound_benchmark.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\centroid_branch_bound_kernel.cpp
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\centroid_hnsw_erng.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\extract_all_query_attack_tables.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\extract_q100_attack_tables.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\fast_erng_direct_kernel.cpp
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\robust_examples_report.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\run_all_query_attack_tables.sh
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\run_graph_benchmark.sh
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\trace_path_examples.py
```

## Example / Report Files

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase\examples\example_config.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\reports\final_project_report.md
C:\Users\saket\Test\final_submission_package\code\final_codebase\reports\phase3_report.md
C:\Users\saket\Test\final_submission_package\code\final_codebase\reports\phase3_trace_diagrams.md
```

## Key Final Reports And Artifacts

```text
C:\Users\saket\Test\final_submission_package\PATHS.md
C:\Users\saket\Test\final_submission_package\reports\final_project_report.md
C:\Users\saket\Test\final_submission_package\reports\phase3_report.md
C:\Users\saket\Test\final_submission_package\trace_diagrams\phase3_labeled_trace_diagrams_with_matches.md
C:\Users\saket\Test\final_submission_package\artifacts\json_results
C:\Users\saket\Test\final_submission_package\artifacts\centroids
C:\Users\saket\Test\final_submission_package\attack_image_examples
```

## Architecture Confirmation

The Phase-3 graph-query files implement the final HNSW and ERNG query layer over the trained embedding stores.

HNSW:

```text
Hierarchical Navigable Small World graph:
centroid hierarchy + centrality/shortcut layers + local KNN links + exponential-rank skip links.
```

ERNG:

```text
Exponential Rank Navigation Graph:
flat multi-probe centroid graph + local KNN links + exponential-rank skip links.
```

The final comparison tables use the same duplicate / non-duplicate decision as direct dense search. The graph methods are considered correct when their duplicate decision matches direct search under the calibrated threshold policy.

## Phase-3 Centroid Sweep / Autotune Files

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\erng_parameter_sweep.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\centroid_branch_bound_autotune.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\centroid_branch_bound_timing.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\centroid_branch_bound_tuned.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\centroid_clean_timing_all.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\fast_centroid_ann.py
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\run_centroid_autotune.sh
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\run_centroid_branch_bound.sh
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\run_centroid_tuned_branch_bound.sh
C:\Users\saket\Test\final_submission_package\code\final_codebase\phase3_graph\run_fast_ann.sh
```
