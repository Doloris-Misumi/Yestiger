# Signal Callability Experiment: starttruedreams

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.500 | 0.268 | 90 |
| callability_rule | 0.333 | 0.144 | 90 |

## Target Role Distribution

- `keepspace`: 8
- `rhythmcall`: 35
- `mix`: 39
- `underground_gei`: 8

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.857 | 0.300 | 0.444 |
| fused_novelty_topk | 0.400 | 0.400 | 0.400 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 1.000 | 0.444 | 0.615 | 18 | 8 |
| manual_fine | fused_novelty_topk | 0.333 | 0.333 | 0.333 | 18 | 18 |
| manual_coarse | allin1_structure | 0.875 | 0.538 | 0.667 | 13 | 8 |
| manual_coarse | fused_novelty_topk | 0.385 | 0.385 | 0.385 | 13 | 13 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.500 | 0.182 | 0.333 |
| callability_rule | 0.333 | 0.092 | 0.414 |
| loso_structure_rf | 0.539 | 0.355 | 0.606 |
| loso_audio_rf | 0.478 | 0.295 | 0.571 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.522 | 0.339 | 0.703 |
| loso_audio_rf_boundary_topk_struct_heavy_len | 0.467 | 0.287 | 0.564 |
