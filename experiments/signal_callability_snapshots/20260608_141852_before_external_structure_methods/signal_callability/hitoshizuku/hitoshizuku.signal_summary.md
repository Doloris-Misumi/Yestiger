# Signal Callability Experiment: hitoshizuku

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.361 | 0.251 | 83 |
| callability_rule | 0.349 | 0.267 | 83 |

## Target Role Distribution

- `keepspace`: 8
- `rhythmcall`: 28
- `mix`: 38
- `underground_gei`: 9

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 1.000 | 0.059 | 0.111 |
| fused_novelty_topk | 0.294 | 0.294 | 0.294 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 1.000 | 0.385 | 0.556 | 13 | 5 |
| manual_fine | fused_novelty_topk | 0.385 | 0.385 | 0.385 | 13 | 13 |
| manual_coarse | allin1_structure | 1.000 | 0.556 | 0.714 | 9 | 5 |
| manual_coarse | fused_novelty_topk | 0.333 | 0.333 | 0.333 | 9 | 9 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.363 | 0.160 | 0.111 |
| callability_rule | 0.350 | 0.147 | 0.250 |
| loso_structure_rf | 0.453 | 0.332 | 0.364 |
| loso_audio_rf | 0.385 | 0.309 | 0.490 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.417 | 0.305 | 0.450 |
| loso_audio_rf_boundary_topk_struct_heavy_len | 0.248 | 0.211 | 0.588 |
