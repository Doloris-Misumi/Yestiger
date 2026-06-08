# Call-Role Model Search

| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |
|---|---:|---:|---:|---:|---:|
| search_audio_vote_rf_logreg_gb | 0.526 | 0.513 | 0.516 | 0.338 | 0.474 |
| search_audio_vote_rf2_logreg1 | 0.509 | 0.505 | 0.501 | 0.333 | 0.482 |
| search_audio_vote_rf_logreg | 0.493 | 0.496 | 0.487 | 0.329 | 0.479 |
| search_audio_vote_rf1_logreg2 | 0.489 | 0.495 | 0.485 | 0.330 | 0.456 |
| search_audio_logreg | 0.479 | 0.489 | 0.475 | 0.323 | 0.456 |
| search_audio_rf | 0.509 | 0.485 | 0.501 | 0.316 | 0.479 |

Best by macro-F1: `search_audio_vote_rf_logreg_gb` (accuracy=0.526, macro-F1=0.513).
