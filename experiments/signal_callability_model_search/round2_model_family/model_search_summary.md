# Call-Role Model Search

| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |
|---|---:|---:|---:|---:|---:|
| search_audio_logreg | 0.479 | 0.489 | 0.475 | 0.323 | 0.456 |
| search_audio_rf | 0.509 | 0.485 | 0.501 | 0.316 | 0.479 |
| search_audio_gradient_boosting | 0.502 | 0.480 | 0.497 | 0.315 | 0.475 |
| search_audio_linear_svc | 0.478 | 0.479 | 0.471 | 0.313 | 0.457 |
| search_audio_extra_trees | 0.504 | 0.476 | 0.501 | 0.312 | 0.430 |
| search_audio_rf_shallow | 0.489 | 0.470 | 0.478 | 0.299 | 0.455 |
| search_audio_rf_leaf1 | 0.496 | 0.466 | 0.500 | 0.307 | 0.419 |
| search_audio_rbf_svc | 0.466 | 0.459 | 0.465 | 0.299 | 0.475 |
| search_audio_rf_deep | 0.484 | 0.443 | 0.489 | 0.291 | 0.421 |

Best by macro-F1: `search_audio_logreg` (accuracy=0.479, macro-F1=0.489).
