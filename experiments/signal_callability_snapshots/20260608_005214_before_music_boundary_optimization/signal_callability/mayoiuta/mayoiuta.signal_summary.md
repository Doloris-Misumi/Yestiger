# Signal Callability Experiment: mayoiuta

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.310 | 0.251 | 84 |
| callability_rule | 0.155 | 0.104 | 84 |

## Target Role Distribution

- `keepspace`: 12
- `rhythmcall`: 13
- `mix`: 43
- `underground_gei`: 16

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.636 | 0.467 | 0.538 |
| fused_novelty_topk | 0.267 | 0.267 | 0.267 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.778 | 0.438 | 0.560 | 16 | 9 |
| manual_fine | fused_novelty_topk | 0.250 | 0.250 | 0.250 | 16 | 16 |
| manual_coarse | allin1_structure | 0.778 | 0.778 | 0.778 | 9 | 9 |
| manual_coarse | fused_novelty_topk | 0.111 | 0.111 | 0.111 | 9 | 9 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.307 | 0.146 | 0.200 |
| callability_rule | 0.149 | 0.048 | 0.100 |
| loso_structure_rf | 0.558 | 0.394 | 0.514 |
| loso_audio_rf | 0.586 | 0.435 | 0.432 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.474 | 0.340 | 0.368 |
| loso_audio_rf_boundary_topk_struct_heavy_len | 0.681 | 0.505 | 0.438 |
