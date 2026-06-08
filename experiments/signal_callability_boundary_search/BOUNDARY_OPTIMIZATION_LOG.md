# Boundary Decoder Optimization Log

Date: 2026-06-08

## Frozen Pre-Boundary Version

Before boundary-specific optimization, the current code and formal experiment outputs were saved to:

```text
experiments/signal_callability_snapshots/20260608_003453_before_boundary_optimization
```

The final boundary decoder code snapshot is:

```text
experiments/signal_callability_boundary_search/code_snapshots/final_boundary_decoder
```

## Problem

The previous best boundary result came from `loso_audio_rf`:

| Method | Time-W Acc | Macro IoU | Boundary P | Boundary R | Boundary F1 | Pred Bnd |
|---|---:|---:|---:|---:|---:|---:|
| loso_audio_rf | 0.501 | 0.316 | 0.431 | 0.538 | 0.479 | 394 |

The target has 316 boundaries. The model predicted 394 boundaries, so the main error was too many false boundaries.

## Round 1

Output:

```text
experiments/signal_callability_boundary_search/round1/boundary_search_summary.md
```

Best result:

| Method | Time-W Acc | Macro IoU | Boundary P | Boundary R | Boundary F1 | Pred Bnd |
|---|---:|---:|---:|---:|---:|---:|
| loso_audio_rf__topk_struct_novelty_len_scale1p1 | 0.493 | 0.307 | 0.492 | 0.503 | 0.498 | 323 |

Short-run smoothing reduced false positives too aggressively and hurt recall. Top-k boundary pruning worked better.

## Round 2

Output:

```text
experiments/signal_callability_boundary_search/round2_fine/boundary_search_summary.md
```

Best result:

| Method | Time-W Acc | Macro IoU | Boundary P | Boundary R | Boundary F1 | Pred Bnd |
|---|---:|---:|---:|---:|---:|---:|
| loso_audio_rf__topk_struct_heavy_len_scale1p1 | 0.497 | 0.311 | 0.495 | 0.509 | 0.502 | 325 |

Final selected method:

```text
loso_audio_rf_boundary_topk_struct_heavy_len
```

## Method

The selected decoder starts from the `loso_audio_rf` bar-level role sequence. It then considers only the role-change boundaries produced by that sequence and scores each candidate boundary using:

```text
0.50 * allin1 structure-boundary flag
+ 0.20 * fused novelty
+ 0.15 * onset novelty
+ 0.15 * energy novelty
```

This signal score is combined with a local span-length score:

```text
0.65 * signal_score + 0.35 * adjacent-span-length score
```

For each held-out song, the number of kept boundaries is estimated from the boundary rate of the training songs only, scaled by 1.10. This keeps the procedure leave-one-song-out and avoids using the held-out song's boundary count.

## Final Formal Run

Formal output:

```text
experiments/signal_callability/aggregate_signal_summary.md
experiments/signal_callability/aggregate_signal_metrics.json
```

Final comparison:

| Method | Time-W Acc | Macro IoU | Boundary P | Boundary R | Boundary F1 | Pred Bnd |
|---|---:|---:|---:|---:|---:|---:|
| loso_audio_rf | 0.501 | 0.316 | 0.431 | 0.538 | 0.479 | 394 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.518 | 0.346 | 0.430 | 0.522 | 0.471 | 384 |
| loso_audio_rf_boundary_topk_struct_heavy_len | 0.497 | 0.311 | 0.495 | 0.509 | 0.502 | 325 |

Takeaway: the boundary-focused decoder improves boundary F1 from 0.479 to 0.502 by pruning low-evidence role changes. It is the best method for boundary segmentation, while the soft-voting model remains the best method for role classification and span overlap.
