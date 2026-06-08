# Signal Callability Experiment: kizunamusic

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.333 | 0.329 | 126 |
| callability_rule | 0.325 | 0.264 | 126 |

## Target Role Distribution

- `keepspace`: 6
- `rhythmcall`: 25
- `mix`: 55
- `underground_gei`: 40

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.583 | 0.500 | 0.538 |
| fused_novelty_topk | 0.500 | 0.500 | 0.500 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.900 | 0.529 | 0.667 | 17 | 10 |
| manual_fine | fused_novelty_topk | 0.235 | 0.235 | 0.235 | 17 | 17 |
| manual_coarse | allin1_structure | 0.900 | 0.750 | 0.818 | 12 | 10 |
| manual_coarse | fused_novelty_topk | 0.167 | 0.167 | 0.167 | 12 | 12 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.334 | 0.201 | 0.222 |
| callability_rule | 0.326 | 0.150 | 0.541 |
| loso_structure_rf | 0.733 | 0.496 | 0.467 |
| loso_audio_rf | 0.789 | 0.643 | 0.514 |
