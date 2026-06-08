# Signal Callability Experiment: teardrops

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.552 | 0.354 | 87 |
| callability_rule | 0.402 | 0.257 | 87 |

## Target Role Distribution

- `keepspace`: 17
- `rhythmcall`: 35
- `mix`: 27
- `underground_gei`: 8

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.833 | 0.357 | 0.500 |
| fused_novelty_topk | 0.500 | 0.500 | 0.500 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.875 | 0.389 | 0.538 | 18 | 8 |
| manual_fine | fused_novelty_topk | 0.278 | 0.278 | 0.278 | 18 | 18 |
| manual_coarse | allin1_structure | 0.875 | 0.538 | 0.667 | 13 | 8 |
| manual_coarse | fused_novelty_topk | 0.231 | 0.231 | 0.231 | 13 | 13 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.556 | 0.250 | 0.188 |
| callability_rule | 0.398 | 0.151 | 0.590 |
| loso_structure_rf | 0.558 | 0.344 | 0.520 |
| loso_audio_rf | 0.326 | 0.191 | 0.500 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.349 | 0.228 | 0.520 |
| loso_audio_rf_boundary_topk_struct_heavy_len | 0.256 | 0.128 | 0.522 |
