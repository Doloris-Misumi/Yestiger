# Signal Callability Experiment: nijuu_no_niji

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.619 | 0.339 | 97 |
| callability_rule | 0.536 | 0.376 | 97 |

## Target Role Distribution

- `keepspace`: 9
- `rhythmcall`: 36
- `mix`: 44
- `underground_gei`: 8

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.615 | 0.571 | 0.593 |
| fused_novelty_topk | 0.500 | 0.500 | 0.500 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 1.000 | 0.421 | 0.593 | 19 | 8 |
| manual_fine | fused_novelty_topk | 0.421 | 0.421 | 0.421 | 19 | 19 |
| manual_coarse | allin1_structure | 0.875 | 0.583 | 0.700 | 12 | 8 |
| manual_coarse | fused_novelty_topk | 0.417 | 0.417 | 0.417 | 12 | 12 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.625 | 0.258 | 0.500 |
| callability_rule | 0.531 | 0.231 | 0.500 |
| loso_structure_rf | 0.604 | 0.417 | 0.552 |
| loso_audio_rf | 0.708 | 0.534 | 0.462 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.719 | 0.546 | 0.571 |
