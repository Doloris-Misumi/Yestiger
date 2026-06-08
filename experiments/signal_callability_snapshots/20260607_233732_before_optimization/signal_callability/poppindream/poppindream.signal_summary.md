# Signal Callability Experiment: poppindream

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.354 | 0.350 | 113 |
| callability_rule | 0.292 | 0.223 | 113 |

## Target Role Distribution

- `keepspace`: 6
- `rhythmcall`: 36
- `mix`: 55
- `underground_gei`: 16

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.444 | 0.222 | 0.296 |
| fused_novelty_topk | 0.278 | 0.278 | 0.278 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.625 | 0.250 | 0.357 | 20 | 8 |
| manual_fine | fused_novelty_topk | 0.200 | 0.200 | 0.200 | 20 | 20 |
| manual_coarse | allin1_structure | 0.500 | 0.333 | 0.400 | 12 | 8 |
| manual_coarse | fused_novelty_topk | 0.333 | 0.333 | 0.333 | 12 | 12 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.357 | 0.256 | 0.250 |
| callability_rule | 0.288 | 0.102 | 0.345 |
| loso_structure_rf | 0.565 | 0.419 | 0.176 |
| loso_audio_rf | 0.503 | 0.403 | 0.341 |
