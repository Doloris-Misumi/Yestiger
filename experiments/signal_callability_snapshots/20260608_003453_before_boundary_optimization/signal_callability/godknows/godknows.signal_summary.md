# Signal Callability Experiment: godknows

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.526 | 0.304 | 175 |
| callability_rule | 0.480 | 0.292 | 175 |

## Target Role Distribution

- `keepspace`: 27
- `rhythmcall`: 46
- `mix`: 86
- `underground_gei`: 16

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.500 | 0.400 | 0.444 |
| fused_novelty_topk | 0.280 | 0.280 | 0.280 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 1.000 | 0.571 | 0.727 | 14 | 8 |
| manual_fine | fused_novelty_topk | 0.071 | 0.071 | 0.071 | 14 | 14 |
| manual_coarse | allin1_structure | 1.000 | 0.800 | 0.889 | 10 | 8 |
| manual_coarse | fused_novelty_topk | 0.100 | 0.100 | 0.100 | 10 | 10 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.528 | 0.222 | 0.200 |
| callability_rule | 0.478 | 0.196 | 0.364 |
| loso_structure_rf | 0.507 | 0.456 | 0.455 |
| loso_audio_rf | 0.754 | 0.520 | 0.600 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.685 | 0.513 | 0.457 |
