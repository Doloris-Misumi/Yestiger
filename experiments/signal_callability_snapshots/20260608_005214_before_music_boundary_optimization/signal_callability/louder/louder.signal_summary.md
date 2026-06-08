# Signal Callability Experiment: louder

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.227 | 0.153 | 88 |
| callability_rule | 0.330 | 0.228 | 88 |

## Target Role Distribution

- `keepspace`: 21
- `rhythmcall`: 23
- `mix`: 36
- `underground_gei`: 8

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.600 | 0.136 | 0.222 |
| fused_novelty_topk | 0.409 | 0.409 | 0.409 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.875 | 0.467 | 0.609 | 15 | 8 |
| manual_fine | fused_novelty_topk | 0.333 | 0.333 | 0.333 | 15 | 15 |
| manual_coarse | allin1_structure | 0.875 | 0.700 | 0.778 | 10 | 8 |
| manual_coarse | fused_novelty_topk | 0.400 | 0.400 | 0.400 | 10 | 10 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.222 | 0.081 | 0.077 |
| callability_rule | 0.328 | 0.136 | 0.211 |
| loso_structure_rf | 0.431 | 0.281 | 0.250 |
| loso_audio_rf | 0.465 | 0.318 | 0.526 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.465 | 0.318 | 0.571 |
| loso_audio_rf_boundary_topk_struct_heavy_len | 0.465 | 0.318 | 0.526 |
