# Boundary Decoder Search

| Method | Base | Decoder | Time-W Acc | Macro IoU | Boundary P | Boundary R | Boundary F1 | Pred Bnd |
|---|---|---|---:|---:|---:|---:|---:|---:|
| loso_audio_rf__topk_struct_heavy_len_scale1p1 | loso_audio_rf | topk | 0.497 | 0.311 | 0.495 | 0.509 | 0.502 | 325 |
| loso_audio_rf__topk_struct_novelty_len_scale1p1 | loso_audio_rf | topk | 0.493 | 0.307 | 0.492 | 0.503 | 0.498 | 323 |
| loso_audio_rf__topk_struct_novelty_len80_scale1p02 | loso_audio_rf | topk | 0.502 | 0.312 | 0.505 | 0.491 | 0.498 | 307 |
| loso_audio_rf__topk_novelty_len_scale1p1 | loso_audio_rf | topk | 0.490 | 0.305 | 0.491 | 0.503 | 0.497 | 324 |
| loso_audio_rf__topk_struct_heavy_len_scale1p02 | loso_audio_rf | topk | 0.503 | 0.316 | 0.503 | 0.491 | 0.497 | 308 |
| loso_audio_rf__topk_struct_novelty_len_scale1p02 | loso_audio_rf | topk | 0.494 | 0.308 | 0.503 | 0.491 | 0.497 | 308 |
| loso_audio_rf__topk_struct_novelty_len50_scale1p14 | loso_audio_rf | topk | 0.490 | 0.306 | 0.483 | 0.509 | 0.496 | 333 |
| loso_audio_rf__topk_struct_heavy_len_scale1p14 | loso_audio_rf | topk | 0.488 | 0.305 | 0.483 | 0.509 | 0.496 | 333 |
| loso_audio_rf__topk_struct_novelty_len80_scale1p06 | loso_audio_rf | topk | 0.491 | 0.303 | 0.495 | 0.497 | 0.496 | 317 |
| loso_audio_rf__topk_struct_novelty_len50_scale1p18 | loso_audio_rf | topk | 0.489 | 0.305 | 0.474 | 0.519 | 0.495 | 346 |
| loso_audio_rf__topk_novelty_len_scale1p14 | loso_audio_rf | topk | 0.490 | 0.308 | 0.481 | 0.509 | 0.495 | 335 |
| loso_audio_rf__topk_struct_novelty_len50_scale1p02 | loso_audio_rf | topk | 0.500 | 0.313 | 0.502 | 0.487 | 0.494 | 307 |
| loso_audio_rf__topk_struct_novelty_len_scale1p14 | loso_audio_rf | topk | 0.488 | 0.304 | 0.482 | 0.506 | 0.494 | 332 |
| loso_audio_rf__learned_rf_scale1p1 | loso_audio_rf | learned_topk | 0.487 | 0.306 | 0.488 | 0.500 | 0.494 | 324 |
| loso_audio_rf__topk_call_window_scale1p06 | loso_audio_rf | topk | 0.495 | 0.310 | 0.491 | 0.497 | 0.494 | 320 |
| loso_audio_rf__topk_struct_novelty_len_scale1p06 | loso_audio_rf | topk | 0.488 | 0.303 | 0.494 | 0.494 | 0.494 | 316 |
| loso_audio_rf__topk_novelty_len_scale1p02 | loso_audio_rf | topk | 0.496 | 0.310 | 0.500 | 0.487 | 0.494 | 308 |
| loso_audio_rf__topk_novelty_len_scale1p18 | loso_audio_rf | topk | 0.489 | 0.307 | 0.472 | 0.516 | 0.493 | 345 |
| loso_audio_rf__topk_struct_novelty_len_scale1p18 | loso_audio_rf | topk | 0.486 | 0.304 | 0.472 | 0.516 | 0.493 | 345 |
| loso_audio_rf__topk_struct_heavy_len_scale1p18 | loso_audio_rf | topk | 0.486 | 0.304 | 0.472 | 0.516 | 0.493 | 345 |
| loso_audio_rf__topk_call_window_scale1p14 | loso_audio_rf | topk | 0.488 | 0.306 | 0.480 | 0.506 | 0.493 | 333 |
| loso_audio_rf__topk_call_window_scale1p1 | loso_audio_rf | topk | 0.489 | 0.306 | 0.486 | 0.500 | 0.493 | 325 |
| loso_audio_rf__topk_struct_heavy_len_scale1p06 | loso_audio_rf | topk | 0.503 | 0.316 | 0.492 | 0.494 | 0.493 | 317 |
| loso_audio_rf__topk_call_window_scale1p02 | loso_audio_rf | topk | 0.497 | 0.311 | 0.498 | 0.487 | 0.493 | 309 |
| loso_audio_rf__topk_struct_novelty_len50_scale1p1 | loso_audio_rf | topk | 0.499 | 0.312 | 0.488 | 0.497 | 0.492 | 322 |
| loso_audio_rf__topk_struct_novelty_len50_scale1p06 | loso_audio_rf | topk | 0.494 | 0.308 | 0.492 | 0.491 | 0.491 | 315 |
| loso_audio_rf__topk_novelty_len_scale1p06 | loso_audio_rf | topk | 0.488 | 0.304 | 0.492 | 0.491 | 0.491 | 315 |
| loso_audio_rf__topk_call_window_scale1p18 | loso_audio_rf | topk | 0.488 | 0.305 | 0.471 | 0.513 | 0.491 | 344 |
| loso_audio_rf__topk_struct_novelty_len80_scale1p14 | loso_audio_rf | topk | 0.490 | 0.305 | 0.477 | 0.503 | 0.490 | 333 |
| loso_audio_rf__topk_struct_novelty_len80_scale1p18 | loso_audio_rf | topk | 0.490 | 0.307 | 0.468 | 0.513 | 0.489 | 346 |
| loso_audio_rf__learned_rf_scale1p14 | loso_audio_rf | learned_topk | 0.498 | 0.313 | 0.475 | 0.503 | 0.488 | 335 |
| loso_audio_rf__topk_struct_novelty_len80_scale1p1 | loso_audio_rf | topk | 0.489 | 0.302 | 0.483 | 0.494 | 0.488 | 323 |
| loso_audio_rf__learned_gb_scale1p1 | loso_audio_rf | learned_topk | 0.482 | 0.302 | 0.480 | 0.491 | 0.485 | 323 |
| loso_audio_rf__learned_rf_scale1p06 | loso_audio_rf | learned_topk | 0.480 | 0.298 | 0.486 | 0.484 | 0.485 | 315 |
| loso_audio_rf__learned_gb_scale1p06 | loso_audio_rf | learned_topk | 0.480 | 0.298 | 0.484 | 0.484 | 0.484 | 316 |
| loso_audio_rf__learned_gb_scale1p14 | loso_audio_rf | learned_topk | 0.483 | 0.301 | 0.468 | 0.494 | 0.481 | 333 |
| loso_audio_rf__raw | loso_audio_rf | raw | 0.501 | 0.316 | 0.431 | 0.538 | 0.479 | 394 |
| loso_audio_rf__learned_gb_scale1p0 | loso_audio_rf | learned_topk | 0.487 | 0.303 | 0.490 | 0.462 | 0.476 | 298 |
| loso_audio_rf__learned_rf_scale1p0 | loso_audio_rf | learned_topk | 0.482 | 0.300 | 0.490 | 0.462 | 0.476 | 298 |
| loso_audio_rf__learned_logreg_scale1p06 | loso_audio_rf | learned_topk | 0.493 | 0.310 | 0.467 | 0.475 | 0.471 | 321 |
| loso_audio_rf__learned_logreg_scale1p0 | loso_audio_rf | learned_topk | 0.499 | 0.315 | 0.480 | 0.459 | 0.469 | 302 |
| loso_audio_rf__learned_logreg_scale1p1 | loso_audio_rf | learned_topk | 0.498 | 0.315 | 0.460 | 0.478 | 0.469 | 328 |
| loso_audio_rf__learned_logreg_scale1p14 | loso_audio_rf | learned_topk | 0.488 | 0.307 | 0.454 | 0.484 | 0.469 | 337 |

Best by boundary F1: `loso_audio_rf__topk_struct_heavy_len_scale1p1` (P=0.495, R=0.509, F1=0.502).
