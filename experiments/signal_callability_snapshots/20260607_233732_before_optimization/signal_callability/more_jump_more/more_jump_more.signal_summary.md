# Signal Callability Experiment: more_jump_more

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.325 | 0.123 | 77 |
| callability_rule | 0.364 | 0.295 | 77 |

## Target Role Distribution

- `keepspace`: 3
- `rhythmcall`: 25
- `mix`: 45
- `underground_gei`: 4

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 1.000 | 0.316 | 0.480 |
| fused_novelty_topk | 0.474 | 0.474 | 0.474 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.600 | 0.136 | 0.222 | 22 | 5 |
| manual_fine | fused_novelty_topk | 0.591 | 0.591 | 0.591 | 22 | 22 |
| manual_coarse | allin1_structure | 0.600 | 0.231 | 0.333 | 13 | 5 |
| manual_coarse | fused_novelty_topk | 0.462 | 0.462 | 0.462 | 13 | 13 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.326 | 0.082 | 0.000 |
| callability_rule | 0.359 | 0.150 | 0.483 |
| loso_structure_rf | 0.491 | 0.314 | 0.600 |
| loso_audio_rf | 0.452 | 0.220 | 0.513 |
