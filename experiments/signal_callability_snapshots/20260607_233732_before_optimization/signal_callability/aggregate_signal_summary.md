# Signal Callability Experiment: Aggregate Results

| Song | Bars | Structure Acc | Structure Macro-F1 | Callability Acc | Callability Macro-F1 |
|---|---:|---:|---:|---:|---:|
| dododo | 111 | 0.378 | 0.284 | 0.369 | 0.274 |
| godknows | 175 | 0.526 | 0.304 | 0.480 | 0.292 |
| hitoshizuku | 83 | 0.361 | 0.251 | 0.349 | 0.267 |
| ishizue_no_hanakanmuri | 138 | 0.145 | 0.148 | 0.167 | 0.164 |
| jibun_restart | 225 | 0.324 | 0.192 | 0.227 | 0.168 |
| kizunamusic | 126 | 0.333 | 0.329 | 0.325 | 0.264 |
| louder | 88 | 0.227 | 0.153 | 0.330 | 0.228 |
| mayoiuta | 84 | 0.310 | 0.251 | 0.155 | 0.104 |
| more_jump_more | 77 | 0.325 | 0.123 | 0.364 | 0.295 |
| nijuu_no_niji | 97 | 0.619 | 0.339 | 0.536 | 0.376 |
| poppindream | 113 | 0.354 | 0.350 | 0.292 | 0.223 |
| starttruedreams | 90 | 0.500 | 0.268 | 0.333 | 0.144 |
| teardrops | 87 | 0.552 | 0.354 | 0.402 | 0.257 |
| xiuwaxiuwa | 114 | 0.500 | 0.393 | 0.342 | 0.271 |

## Overall

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.386 | 0.254 | 1608 |
| callability_rule | 0.328 | 0.229 | 1608 |
| loso_structure_rf | 0.470 | 0.457 | 1608 |
| loso_audio_rf | 0.509 | 0.485 | 1608 |

## Call-Role Boundary Detection

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.655 | 0.288 | 0.400 |
| fused_novelty_topk | 0.405 | 0.405 | 0.405 |

## Music Segment Boundary Comparison

Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.889 | 0.393 | 0.545 | 244 | 108 |
| manual_fine | fused_novelty_topk | 0.320 | 0.320 | 0.320 | 244 | 244 |
| manual_coarse | allin1_structure | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | fused_novelty_topk | 0.289 | 0.289 | 0.289 | 159 | 159 |

## Merged Span Comparison

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.387 | 0.157 | 0.199 |
| callability_rule | 0.331 | 0.129 | 0.412 |
| loso_structure_rf | 0.481 | 0.303 | 0.418 |
| loso_audio_rf | 0.501 | 0.316 | 0.479 |
