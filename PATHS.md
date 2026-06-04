# Final Package Path Index

Root:

```text
C:\Users\saket\Test\final_submission_package
```

This is the final local package for the project. It contains the clean renamed code, final reports, Phase-3 HNSW/ERNG artifacts, centroid files, JSON result files, trace diagrams with matched images, and corrected attack image examples.

## Main Folders

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase
C:\Users\saket\Test\final_submission_package\artifacts\centroids
C:\Users\saket\Test\final_submission_package\artifacts\json_results
C:\Users\saket\Test\final_submission_package\artifacts\logs
C:\Users\saket\Test\final_submission_package\reports
C:\Users\saket\Test\final_submission_package\trace_diagrams
C:\Users\saket\Test\final_submission_package\attack_image_examples
```

## Dataset Artifact Layout

Centroids:

```text
C:\Users\saket\Test\final_submission_package\artifacts\centroids\cifar
C:\Users\saket\Test\final_submission_package\artifacts\centroids\flickr
C:\Users\saket\Test\final_submission_package\artifacts\centroids\ucid
C:\Users\saket\Test\final_submission_package\artifacts\centroids\amazon
```

JSON results:

```text
C:\Users\saket\Test\final_submission_package\artifacts\json_results\cifar
C:\Users\saket\Test\final_submission_package\artifacts\json_results\flickr
C:\Users\saket\Test\final_submission_package\artifacts\json_results\ucid
C:\Users\saket\Test\final_submission_package\artifacts\json_results\amazon
C:\Users\saket\Test\final_submission_package\artifacts\json_results\common
```

Logs:

```text
C:\Users\saket\Test\final_submission_package\artifacts\logs\cifar
C:\Users\saket\Test\final_submission_package\artifacts\logs\flickr
C:\Users\saket\Test\final_submission_package\artifacts\logs\ucid
C:\Users\saket\Test\final_submission_package\artifacts\logs\amazon
```

## Reports

```text
C:\Users\saket\Test\final_submission_package\reports\final_project_report.md
C:\Users\saket\Test\final_submission_package\reports\phase3_report.md
C:\Users\saket\Test\final_submission_package\reports\phase3_report_from_artifacts.md
```

Trace diagrams:

```text
C:\Users\saket\Test\final_submission_package\trace_diagrams\phase3_trace_diagrams.md
C:\Users\saket\Test\final_submission_package\trace_diagrams\phase3_labeled_trace_diagrams_with_matches.md
```

## Corrected Attack Image Examples

Root:

```text
C:\Users\saket\Test\final_submission_package\attack_image_examples
```

Structure:

```text
attack_image_examples\cifar\10 image folders x 26 images
attack_image_examples\flickr\10 image folders x 26 images
attack_image_examples\ucid\10 image folders x 26 images
attack_image_examples\amazon\10 image folders x 26 images
```

## Final Clean Codebase

```text
C:\Users\saket\Test\final_submission_package\code\final_codebase
```

See:

```text
C:\Users\saket\Test\final_submission_package\CODE_FILES.md
```

## Centroid Sweep / Autotune Artifacts

Centroid sweep artifacts are separated from the final selected centroids here:

```text
C:\Users\saket\Test\final_submission_package\artifacts\centroid_sweep\cifar
C:\Users\saket\Test\final_submission_package\artifacts\centroid_sweep\flickr
C:\Users\saket\Test\final_submission_package\artifacts\centroid_sweep\ucid
C:\Users\saket\Test\final_submission_package\artifacts\centroid_sweep\amazon
```

Each dataset folder may contain:

```text
phase3_centroid_branchbound
phase3_centroid_branchbound_autotune
phase3_centroid_branchbound_tuned
phase3_graph_query
```

These include centroid-count sweep `.npz` files, timing JSON files, tuned run outputs, and sweep logs used before selecting the final Phase-3 graph settings.
