# Signal Callability Experiment: dododo

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.378 | 0.284 | 111 |
| callability_rule | 0.369 | 0.274 | 111 |

## Target Role Distribution

- `keepspace`: 10
- `rhythmcall`: 34
- `mix`: 51
- `underground_gei`: 16

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.545 | 0.240 | 0.333 |
| fused_novelty_topk | 0.520 | 0.520 | 0.520 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.857 | 0.316 | 0.462 | 19 | 7 |
| manual_fine | fused_novelty_topk | 0.316 | 0.316 | 0.316 | 19 | 19 |
| manual_coarse | allin1_structure | 0.714 | 0.455 | 0.556 | 11 | 7 |
| manual_coarse | fused_novelty_topk | 0.273 | 0.273 | 0.273 | 11 | 11 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.378 | 0.178 | 0.074 |
| callability_rule | 0.361 | 0.136 | 0.558 |
| loso_structure_rf | 0.507 | 0.342 | 0.356 |
| loso_audio_rf | 0.397 | 0.271 | 0.385 |
