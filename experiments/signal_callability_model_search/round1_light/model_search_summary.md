# Call-Role Model Search

| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |
|---|---:|---:|---:|---:|---:|
| search_audio_rf_context | 0.503 | 0.480 | 0.500 | 0.316 | 0.456 |
| search_audio_rf_context_viterbi_tw050_bb055 | 0.504 | 0.465 | 0.501 | 0.304 | 0.300 |
| search_audio_et_context | 0.488 | 0.458 | 0.491 | 0.302 | 0.417 |
| search_audio_rf_context_viterbi_tw080_bb055 | 0.486 | 0.441 | 0.486 | 0.289 | 0.263 |
| search_audio_et_context_viterbi_tw050_bb055 | 0.463 | 0.417 | 0.462 | 0.264 | 0.270 |

Best by macro-F1: `search_audio_rf_context` (accuracy=0.503, macro-F1=0.480).
