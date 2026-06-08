# Signal Callability Experiment: ishizue_no_hanakanmuri

## Metrics

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.145 | 0.148 | 138 |
| callability_rule | 0.167 | 0.164 | 138 |

## Target Role Distribution

- `keepspace`: 27
- `rhythmcall`: 8
- `mix`: 59
- `underground_gei`: 44

## Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.429 | 0.214 | 0.286 |
| fused_novelty_topk | 0.214 | 0.214 | 0.214 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 1.000 | 0.353 | 0.522 | 17 | 6 |
| manual_fine | fused_novelty_topk | 0.294 | 0.294 | 0.294 | 17 | 17 |
| manual_coarse | allin1_structure | 1.000 | 0.500 | 0.667 | 12 | 6 |
| manual_coarse | fused_novelty_topk | 0.333 | 0.333 | 0.333 | 12 | 12 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.143 | 0.070 | 0.222 |
| callability_rule | 0.157 | 0.073 | 0.133 |
| loso_structure_rf | 0.229 | 0.095 | 0.375 |
| loso_audio_rf | 0.268 | 0.155 | 0.241 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.363 | 0.243 | 0.316 |
