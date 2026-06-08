# Signal Callability Experiment: xiuwaxiuwa

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.500 | 0.393 | 114 |
| callability_rule | 0.342 | 0.271 | 114 |

## Target Role Distribution

- `keepspace`: 30
- `rhythmcall`: 29
- `mix`: 55
- `underground_gei`: 0

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.750 | 0.257 | 0.383 |
| fused_novelty_topk | 0.457 | 0.457 | 0.457 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 1.000 | 0.450 | 0.621 | 20 | 9 |
| manual_fine | fused_novelty_topk | 0.450 | 0.450 | 0.450 | 20 | 20 |
| manual_coarse | allin1_structure | 0.889 | 0.667 | 0.762 | 12 | 9 |
| manual_coarse | fused_novelty_topk | 0.333 | 0.333 | 0.333 | 12 | 12 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.503 | 0.287 | 0.244 |
| callability_rule | 0.371 | 0.180 | 0.320 |
| loso_structure_rf | 0.240 | 0.115 | 0.471 |
| loso_audio_rf | 0.463 | 0.197 | 0.646 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.533 | 0.247 | 0.492 |
| loso_audio_rf_boundary_topk_struct_heavy_len | 0.486 | 0.215 | 0.644 |
