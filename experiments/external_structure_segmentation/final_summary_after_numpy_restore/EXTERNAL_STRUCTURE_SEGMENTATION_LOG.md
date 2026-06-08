# External Structure Segmentation Final Summary

| Target | Group | Method | Uses allin1 | Uses manual training | Precision | Recall | F1 | Target Bnd | Pred Bnd | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| manual_coarse | 00_previous_baseline | allin1_structure | true | false | 0.843 | 0.572 | 0.682 | 159 | 108 | experiments/signal_callability |
| manual_coarse | 02_classic_msaf_like | foote_checkerboard_loso_tuned | false | true | 0.494 | 0.497 | 0.495 | 159 | 160 | experiments\external_structure_segmentation\round2_classic_fast |
| manual_coarse | 01_cbm_block_dp | cbm_dp_plus_novelty_fill_loso_tuned | false | true | 0.390 | 0.434 | 0.411 | 159 | 177 | experiments\external_structure_segmentation\round3_cbm_fill_fast |
| manual_coarse | 02_classic_msaf_like | agglomerative_loso_tuned | false | true | 0.354 | 0.465 | 0.402 | 159 | 209 | experiments\external_structure_segmentation\round2_classic_fast |
| manual_coarse | 01_cbm_block_dp | cbm_dp_full_fixed | false | false | 0.283 | 0.616 | 0.388 | 159 | 346 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 02_classic_msaf_like | agglomerative_full_loso_count | false | true | 0.362 | 0.365 | 0.364 | 159 | 160 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 01_cbm_block_dp | cbm_dp_chroma_fixed | false | false | 0.272 | 0.535 | 0.361 | 159 | 312 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 02_classic_msaf_like | foote_checkerboard_full_loso_count | false | true | 0.356 | 0.358 | 0.357 | 159 | 160 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 01_cbm_block_dp | cbm_dp_loso_tuned | false | true | 0.308 | 0.384 | 0.342 | 159 | 198 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 02_classic_msaf_like | spectral_full_loso_count | false | true | 0.338 | 0.340 | 0.339 | 159 | 160 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 00_previous_baseline | fused_novelty_topk | false | false | 0.289 | 0.289 | 0.289 | 159 | 159 | experiments/signal_callability |
| manual_fine | 00_previous_baseline | allin1_structure | true | false | 0.889 | 0.393 | 0.545 | 244 | 108 | experiments/signal_callability |
| manual_fine | 02_classic_msaf_like | foote_checkerboard_loso_tuned | false | true | 0.468 | 0.508 | 0.487 | 244 | 265 | experiments\external_structure_segmentation\round2_classic_fast |
| manual_fine | 02_classic_msaf_like | foote_checkerboard_full_loso_count | false | true | 0.463 | 0.467 | 0.465 | 244 | 246 | experiments\external_structure_segmentation\round1 |
| manual_fine | 02_classic_msaf_like | agglomerative_full_loso_count | false | true | 0.451 | 0.455 | 0.453 | 244 | 246 | experiments\external_structure_segmentation\round1 |
| manual_fine | 01_cbm_block_dp | cbm_dp_loso_tuned | false | true | 0.387 | 0.541 | 0.451 | 244 | 341 | experiments\external_structure_segmentation\round1 |
| manual_fine | 01_cbm_block_dp | cbm_dp_full_fixed | false | false | 0.379 | 0.537 | 0.444 | 244 | 346 | experiments\external_structure_segmentation\round1 |
| manual_fine | 02_classic_msaf_like | agglomerative_loso_tuned | false | true | 0.394 | 0.508 | 0.444 | 244 | 315 | experiments\external_structure_segmentation\round2_classic_fast |
| manual_fine | 01_cbm_block_dp | cbm_dp_plus_novelty_fill_loso_tuned | false | true | 0.430 | 0.430 | 0.430 | 244 | 244 | experiments\external_structure_segmentation\round3_cbm_fill_fast |
| manual_fine | 02_classic_msaf_like | spectral_full_loso_count | false | true | 0.407 | 0.410 | 0.408 | 244 | 246 | experiments\external_structure_segmentation\round1 |
| manual_fine | 01_cbm_block_dp | cbm_dp_chroma_fixed | false | false | 0.362 | 0.463 | 0.406 | 244 | 312 | experiments\external_structure_segmentation\round1 |
| manual_fine | 00_previous_baseline | fused_novelty_topk | false | false | 0.320 | 0.320 | 0.320 | 244 | 244 | experiments/signal_callability |

## Takeaways

- `manual_fine` overall best: `allin1_structure` F1=0.545; best non-allin1: `foote_checkerboard_loso_tuned` F1=0.487.
- `manual_coarse` overall best: `allin1_structure` F1=0.682; best non-allin1: `foote_checkerboard_loso_tuned` F1=0.495.

## Code And Result Locations

- Main implementation: `scripts/external_structure_segmentation.py`.
- Aggregation script: `scripts/aggregate_external_structure_results.py`.
- Each round contains a `code_snapshot/` directory and per-method `methods/<group>/<method>/<target>/` metrics/predictions.
