# Signal Callability Model Optimization Log

Date: 2026-06-07 / 2026-06-08

## Frozen Baseline

Before optimization, the current script and experiment outputs were copied to:

```text
experiments/signal_callability_snapshots/20260607_233732_before_optimization
```

Baseline best supervised result:

| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |
|---|---:|---:|---:|---:|---:|
| loso_audio_rf | 0.509 | 0.485 | 0.501 | 0.316 | 0.479 |

Final selected code snapshot:

```text
experiments/signal_callability_model_search/code_snapshots/final_selected
```

Each search round also stores its candidate configuration inside `model_search_metrics.json`.

## Round 1: Context and Viterbi

Output:

```text
experiments/signal_callability_model_search/round1_light/model_search_summary.md
```

Best result:

| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |
|---|---:|---:|---:|---:|---:|
| search_audio_rf_context | 0.503 | 0.480 | 0.500 | 0.316 | 0.456 |

Interpretation: local context features and Viterbi smoothing did not improve this dataset. Viterbi reduced excessive switches, but it also hurt role-level macro-F1 and boundary recall.

## Round 2: Model Family Search

Output:

```text
experiments/signal_callability_model_search/round2_model_family/model_search_summary.md
```

Best result:

| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |
|---|---:|---:|---:|---:|---:|
| search_audio_logreg | 0.479 | 0.489 | 0.475 | 0.323 | 0.456 |

Interpretation: Logistic Regression improved macro-F1 by improving minority-role balance, but it reduced overall accuracy and span boundary quality.

## Round 3: Soft Voting

Output:

```text
experiments/signal_callability_model_search/round3_ensemble/model_search_summary.md
```

Best result:

| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |
|---|---:|---:|---:|---:|---:|
| search_audio_vote_rf_logreg_gb | 0.526 | 0.513 | 0.516 | 0.338 | 0.474 |

Interpretation: RF, Logistic Regression, and Gradient Boosting have complementary errors. Soft voting improved both accuracy and macro-F1 over the previous RF baseline.

## Round 4: Voting Weights

Output:

```text
experiments/signal_callability_model_search/round4_vote_weights_small/model_search_summary.md
```

Best result:

| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |
|---|---:|---:|---:|---:|---:|
| search_audio_vote_rf1_logreg1_gb1 | 0.527 | 0.519 | 0.518 | 0.346 | 0.471 |

Final selected method:

```text
loso_audio_vote_rf1_logreg1_gb1
```

This method uses equal-weight soft voting over Random Forest, Logistic Regression, and Gradient Boosting. Each leave-one-song-out fold trains the three models only on the other songs.

## Final Formal Run

Output:

```text
experiments/signal_callability/aggregate_signal_summary.md
experiments/signal_callability/aggregate_signal_metrics.json
```

Final comparison:

| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |
|---|---:|---:|---:|---:|---:|
| loso_structure_rf | 0.470 | 0.457 | 0.481 | 0.303 | 0.418 |
| loso_audio_rf | 0.509 | 0.485 | 0.501 | 0.316 | 0.479 |
| loso_audio_vote_rf1_logreg1_gb1 | 0.527 | 0.519 | 0.518 | 0.346 | 0.471 |

Takeaway: the selected soft-voting model improves role prediction and span overlap, while the old audio RF still has a slightly higher span boundary F1. A future boundary-specific decoder should target that remaining weakness.
