# Music Structure Boundary Optimization Log

Date: 2026-06-08

## Frozen Pre-Optimization Version

Before this music-boundary-specific search, the current code and outputs were saved to:

```text
experiments/signal_callability_snapshots/20260608_005214_before_music_boundary_optimization
```

Final search code snapshot:

```text
experiments/signal_callability_music_boundary_search/code_snapshots/final_music_boundary_search
```

## Task

This experiment is different from call-role boundary detection. It compares music-section boundaries against two human targets:

- `manual_fine`: all human music-section boundaries.
- `manual_coarse`: fine human labels collapsed into an allin1-like coarse label set.

## Baseline Table

| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | true | 0.889 | 0.393 | 0.545 | 244 | 108 |
| manual_fine | fused_novelty_topk | false | 0.320 | 0.320 | 0.320 | 244 | 244 |
| manual_coarse | allin1_structure | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | fused_novelty_topk | false | 0.289 | 0.289 | 0.289 | 159 | 159 |

## Search Results

Search outputs:

```text
experiments/signal_callability_music_boundary_search/round1/music_boundary_search_summary.md
experiments/signal_callability_music_boundary_search/round2_allin1_plus/music_boundary_search_summary.md
experiments/signal_callability_music_boundary_search/round3_allin1_plus_smallscale/music_boundary_search_summary.md
experiments/signal_callability_music_boundary_search/round4_coarse_tiny_scale/music_boundary_search_summary.md
```

### Fine Target

Best result:

| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|---:|
| manual_fine | allin1_plus_learned_rf_scale0p65 | true | 0.774 | 0.504 | 0.610 | 244 | 159 |

Best pure-audio result:

| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|---:|
| manual_fine | audio_learned_rf_scale1p15 | false | 0.431 | 0.500 | 0.463 | 244 | 283 |

Interpretation: fine-grained human boundaries can be improved by starting from allin1 and adding a small number of learned signal-supported boundaries. Pure audio improves over the old `fused_novelty_topk` baseline but does not approach allin1.

### Coarse Target

Best result remains allin1:

| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|---:|
| manual_coarse | allin1_structure | true | 0.843 | 0.572 | 0.682 | 159 | 108 |

Closest searched variants:

| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|---:|
| manual_coarse | allin1_plus_learned_rf_scale0p71 | true | 0.829 | 0.579 | 0.681 | 159 | 111 |
| manual_coarse | allin1_plus_learned_gb_scale0p71 | true | 0.829 | 0.579 | 0.681 | 159 | 111 |
| manual_coarse | audio_learned_gb_scale1p15 | false | 0.378 | 0.440 | 0.407 | 159 | 185 |

Interpretation: allin1 is already a strong coarse music-structure boundary detector. Adding signal-derived boundaries improves recall slightly, but the added false positives cancel the gain. Under the current data and feature set, the honest conclusion is that our signal features improve fine-boundary refinement, not coarse allin1 segmentation.

## Presentation Takeaway

Use this wording:

```text
For coarse music-section boundaries, allin1 remains the strongest baseline.
For fine human annotations, a hybrid allin1 + signal model improves F1 from 0.545 to 0.610.
Pure audio features alone improve over raw novelty but are not enough to replace allin1 structure detection.
```
