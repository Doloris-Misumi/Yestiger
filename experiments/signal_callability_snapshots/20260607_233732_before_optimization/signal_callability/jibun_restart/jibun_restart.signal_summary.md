# Signal Callability Experiment: jibun_restart

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.324 | 0.192 | 225 |
| callability_rule | 0.227 | 0.168 | 225 |

## Target Role Distribution

- `keepspace`: 88
- `rhythmcall`: 86
- `mix`: 19
- `underground_gei`: 32

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.846 | 0.220 | 0.349 |
| fused_novelty_topk | 0.420 | 0.420 | 0.420 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.889 | 0.500 | 0.640 | 16 | 9 |
| manual_fine | fused_novelty_topk | 0.188 | 0.188 | 0.188 | 16 | 16 |
| manual_coarse | allin1_structure | 0.889 | 0.727 | 0.800 | 11 | 9 |
| manual_coarse | fused_novelty_topk | 0.091 | 0.091 | 0.091 | 11 | 11 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.295 | 0.105 | 0.182 |
| callability_rule | 0.232 | 0.096 | 0.496 |
| loso_structure_rf | 0.306 | 0.194 | 0.269 |
| loso_audio_rf | 0.346 | 0.225 | 0.466 |
