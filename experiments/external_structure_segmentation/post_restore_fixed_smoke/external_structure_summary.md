# External Music Structure Segmentation

This experiment compares non-allin1 music-structure boundary detectors on the same bar grid used by the signal-callability experiments.

## Baselines From Previous Run

| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | true | 0.889 | 0.393 | 0.545 | 244 | 108 |
| manual_fine | fused_novelty_topk | false | 0.320 | 0.320 | 0.320 | 244 | 244 |
| manual_coarse | allin1_structure | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | fused_novelty_topk | false | 0.289 | 0.289 | 0.289 | 159 | 159 |

## New Methods

| Group | Target | Method | Uses manual training | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 01_cbm_block_dp | manual_coarse | cbm_dp_full_fixed | false | 0.283 | 0.616 | 0.388 | 159 | 346 |
| 01_cbm_block_dp | manual_coarse | cbm_dp_chroma_fixed | false | 0.272 | 0.535 | 0.361 | 159 | 312 |
| 02_classic_msaf_like | manual_coarse | agglomerative_full_loso_count | true | 0.362 | 0.365 | 0.364 | 159 | 160 |
| 02_classic_msaf_like | manual_coarse | foote_checkerboard_full_loso_count | true | 0.356 | 0.358 | 0.357 | 159 | 160 |
| 02_classic_msaf_like | manual_coarse | spectral_full_loso_count | true | 0.338 | 0.340 | 0.339 | 159 | 160 |
| 01_cbm_block_dp | manual_fine | cbm_dp_full_fixed | false | 0.379 | 0.537 | 0.444 | 244 | 346 |
| 01_cbm_block_dp | manual_fine | cbm_dp_chroma_fixed | false | 0.362 | 0.463 | 0.406 | 244 | 312 |
| 02_classic_msaf_like | manual_fine | foote_checkerboard_full_loso_count | true | 0.463 | 0.467 | 0.465 | 244 | 246 |
| 02_classic_msaf_like | manual_fine | agglomerative_full_loso_count | true | 0.451 | 0.455 | 0.453 | 244 | 246 |
| 02_classic_msaf_like | manual_fine | spectral_full_loso_count | true | 0.407 | 0.410 | 0.408 | 244 | 246 |

## Best Per Target

- `manual_fine` best new method: `foote_checkerboard_full_loso_count` (02_classic_msaf_like), F1=0.465.
- `manual_fine` best fixed unsupervised method: `cbm_dp_full_fixed`, F1=0.444.
- `manual_coarse` best new method: `cbm_dp_full_fixed` (01_cbm_block_dp), F1=0.388.
- `manual_coarse` best fixed unsupervised method: `cbm_dp_full_fixed`, F1=0.388.
