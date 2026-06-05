# Final Code And Key Files Index

This is the only path/code index kept in the final submission package. `PATHS.md` was removed as requested.

Root package:

```text
final_submission_package
```

## Training

```text
code\final_codebase\training\config.py
code\final_codebase\training\model.py
code\final_codebase\training\model_convnext.py
code\final_codebase\training\model_resnet.py
code\final_codebase\training\model_vit.py
code\final_codebase\training\run_train_model.py
code\final_codebase\training\train.py
code\final_codebase\training\utils.py
```

## Preprocessing / Attacks / Manifest

```text
code\final_codebase\preprocessing\attacks.py
code\final_codebase\preprocessing\build_manifest.py
code\final_codebase\preprocessing\dataloader.py
code\final_codebase\preprocessing\dataset.py
code\final_codebase\preprocessing\manifest.py
code\final_codebase\preprocessing\manifest_builder.py
```

## Embedding Store

```text
code\final_codebase\store\embedding_store.py
code\final_codebase\store\store_manager.py
```

## Evaluation / Threshold Calibration

```text
code\final_codebase\evaluation\eval_store.py
code\final_codebase\evaluation\evaluation.py
code\final_codebase\evaluation\run_eval_store.py
```

## Repair / Geometry Adjustment

```text
code\final_codebase\repair\geometry_repair.py
code\final_codebase\repair\run_repair_geometry.py
```

## Phase-3 HNSW / ERNG / Graph Query

```text
code\final_codebase\phase3_graph\all_dataset_graph_benchmark.py
code\final_codebase\phase3_graph\centroid_branch_bound_autotune.py
code\final_codebase\phase3_graph\centroid_branch_bound_benchmark.py
code\final_codebase\phase3_graph\centroid_branch_bound_kernel.cpp
code\final_codebase\phase3_graph\centroid_branch_bound_timing.py
code\final_codebase\phase3_graph\centroid_branch_bound_tuned.py
code\final_codebase\phase3_graph\centroid_clean_timing_all.py
code\final_codebase\phase3_graph\centroid_hnsw_erng.py
code\final_codebase\phase3_graph\erng_parameter_sweep.py
code\final_codebase\phase3_graph\extract_all_query_attack_tables.py
code\final_codebase\phase3_graph\extract_q100_attack_tables.py
code\final_codebase\phase3_graph\fast_centroid_ann.py
code\final_codebase\phase3_graph\fast_erng_direct_kernel.cpp
code\final_codebase\phase3_graph\global_threshold_q100.py
code\final_codebase\phase3_graph\robust_examples_report.py
code\final_codebase\phase3_graph\run_all_query_attack_tables.sh
code\final_codebase\phase3_graph\run_centroid_autotune.sh
code\final_codebase\phase3_graph\run_centroid_branch_bound.sh
code\final_codebase\phase3_graph\run_centroid_tuned_branch_bound.sh
code\final_codebase\phase3_graph\run_fast_ann.sh
code\final_codebase\phase3_graph\run_graph_benchmark.sh
code\final_codebase\phase3_graph\trace_path_examples.py
```

## Example Configuration

```text
code\final_codebase\examples\example_config.py
```

## Codebase Embedded Reports

```text
code\final_codebase\reports\final_project_report.md
code\final_codebase\reports\phase3_report.md
code\final_codebase\reports\phase3_trace_diagrams.md
```

## Main Reports

```text
reports\Complete_Report.md
reports\Phase 3 Report From Artifacts.md
reports\Phase 3 Report.md
```

## Trace / Architecture Diagrams

```text
trace_diagrams\Hnsw_Erng.md
trace_diagrams\phase3_attack_variants.md
trace_diagrams\phase3_HNSW_ERNG.md
trace_diagrams\Trace.md
```

## Artifact Folders

```text
artifacts\centroids\amazon
artifacts\centroids\cifar
artifacts\centroids\flickr
artifacts\centroids\ucid
artifacts\centroid_sweep\amazon
artifacts\centroid_sweep\cifar
artifacts\centroid_sweep\flickr
artifacts\centroid_sweep\ucid
artifacts\json_results\amazon
artifacts\json_results\cifar
artifacts\json_results\flickr
artifacts\json_results\ucid
artifacts\logs\amazon
artifacts\logs\cifar
artifacts\logs\flickr
artifacts\logs\ucid
attack_image_examples\amazon
attack_image_examples\cifar
attack_image_examples\flickr
attack_image_examples\ucid
```

## Notes

- `README.md` is the GitHub entry README.
- `CODE_FILES.md` is the only retained index file.
- `PATHS.md` was removed from the final package.
- Phase-3 graph files include HNSW, ERNG, centroid branch-bound search, centroid sweep/autotune, q100 global-threshold evaluator, per-attack table extraction, and trace generation support.
