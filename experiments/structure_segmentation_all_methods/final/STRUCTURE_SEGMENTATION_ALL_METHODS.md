# Structure Segmentation All Methods Summary

| Target | Group | Method | Uses allin1 | Uses manual training | Precision | Recall | F1 | Target Bnd | Pred Bnd | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| manual_coarse | 00_previous_baseline | allin1_structure | true | false | 0.843 | 0.572 | 0.682 | 159 | 108 | experiments\signal_callability |
| manual_coarse | 02_classic_msaf_like | foote_checkerboard_loso_tuned | false | true | 0.494 | 0.497 | 0.495 | 159 | 160 | experiments\external_structure_segmentation\round2_classic_fast |
| manual_coarse | 04_mert_embedding | mert95m_contextual_foote_loso_tuned | false | true | 0.453 | 0.491 | 0.471 | 159 | 172 | experiments\mert_structure_segmentation\round1_tuned |
| manual_coarse | 01_cbm_block_dp | cbm_dp_plus_novelty_fill_loso_tuned | false | true | 0.390 | 0.434 | 0.411 | 159 | 177 | experiments\external_structure_segmentation\round3_cbm_fill_fast |
| manual_coarse | 02_classic_msaf_like | agglomerative_loso_tuned | false | true | 0.354 | 0.465 | 0.402 | 159 | 209 | experiments\external_structure_segmentation\round2_classic_fast |
| manual_coarse | 01_cbm_block_dp | cbm_dp_full_fixed | false | false | 0.283 | 0.616 | 0.388 | 159 | 346 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 02_classic_msaf_like | agglomerative_full_loso_count | false | true | 0.362 | 0.365 | 0.364 | 159 | 160 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 01_cbm_block_dp | cbm_dp_chroma_fixed | false | false | 0.272 | 0.535 | 0.361 | 159 | 312 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 02_classic_msaf_like | foote_checkerboard_full_loso_count | false | true | 0.356 | 0.358 | 0.357 | 159 | 160 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 01_cbm_block_dp | cbm_dp_loso_tuned | false | true | 0.308 | 0.384 | 0.342 | 159 | 198 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 02_classic_msaf_like | spectral_full_loso_count | false | true | 0.338 | 0.340 | 0.339 | 159 | 160 | experiments\external_structure_segmentation\round1 |
| manual_coarse | 00_previous_baseline | fused_novelty_topk | false | false | 0.289 | 0.289 | 0.289 | 159 | 159 | experiments\signal_callability |
| manual_coarse | 03_official_msaf_modules | official_msaf_foote_bar_features_loso_tuned | false | true | 0.511 | 0.151 | 0.233 | 159 | 47 | experiments\official_msaf_segmentation\round1_foote_tuned |
| manual_fine | 05_hybrid_allin1_plus_signal | allin1_plus_learned_rf_scale0p65 | true | true | 0.774 | 0.504 | 0.610 | 244 | 159 | experiments\signal_callability_music_boundary_search\round3_allin1_plus_smallscale |
| manual_fine | 00_previous_baseline | allin1_structure | true | false | 0.889 | 0.393 | 0.545 | 244 | 108 | experiments\signal_callability |
| manual_fine | 02_classic_msaf_like | foote_checkerboard_loso_tuned | false | true | 0.468 | 0.508 | 0.487 | 244 | 265 | experiments\external_structure_segmentation\round2_classic_fast |
| manual_fine | 04_mert_embedding | mert95m_contextual_foote_loso_tuned | false | true | 0.467 | 0.496 | 0.481 | 244 | 259 | experiments\mert_structure_segmentation\round1_tuned |
| manual_fine | 02_classic_msaf_like | foote_checkerboard_full_loso_count | false | true | 0.463 | 0.467 | 0.465 | 244 | 246 | experiments\external_structure_segmentation\round1 |
| manual_fine | 02_classic_msaf_like | agglomerative_full_loso_count | false | true | 0.451 | 0.455 | 0.453 | 244 | 246 | experiments\external_structure_segmentation\round1 |
| manual_fine | 01_cbm_block_dp | cbm_dp_loso_tuned | false | true | 0.387 | 0.541 | 0.451 | 244 | 341 | experiments\external_structure_segmentation\round1 |
| manual_fine | 01_cbm_block_dp | cbm_dp_full_fixed | false | false | 0.379 | 0.537 | 0.444 | 244 | 346 | experiments\external_structure_segmentation\round1 |
| manual_fine | 02_classic_msaf_like | agglomerative_loso_tuned | false | true | 0.394 | 0.508 | 0.444 | 244 | 315 | experiments\external_structure_segmentation\round2_classic_fast |
| manual_fine | 01_cbm_block_dp | cbm_dp_plus_novelty_fill_loso_tuned | false | true | 0.430 | 0.430 | 0.430 | 244 | 244 | experiments\external_structure_segmentation\round3_cbm_fill_fast |
| manual_fine | 02_classic_msaf_like | spectral_full_loso_count | false | true | 0.407 | 0.410 | 0.408 | 244 | 246 | experiments\external_structure_segmentation\round1 |
| manual_fine | 01_cbm_block_dp | cbm_dp_chroma_fixed | false | false | 0.362 | 0.463 | 0.406 | 244 | 312 | experiments\external_structure_segmentation\round1 |
| manual_fine | 00_previous_baseline | fused_novelty_topk | false | false | 0.320 | 0.320 | 0.320 | 244 | 244 | experiments\signal_callability |
| manual_fine | 03_official_msaf_modules | official_msaf_foote_bar_features_loso_tuned | false | true | 0.432 | 0.078 | 0.132 | 244 | 44 | experiments\official_msaf_segmentation\round1_foote_tuned |

## Main Comparisons

- `manual_fine` overall best: `allin1_plus_learned_rf_scale0p65` F1=0.610; best non-allin1: `foote_checkerboard_loso_tuned` F1=0.487.
- `manual_coarse` overall best: `allin1_structure` F1=0.682; best non-allin1: `foote_checkerboard_loso_tuned` F1=0.495.

## Method Notes

- `official_msaf_foote_bar_features_loso_tuned` uses the official MSAF Foote Segmenter module on existing bar-level features. Full `msaf.process` timed out on full MP3 files, and CNMF import failed because `cvxopt` could not load a DLL.
- `mert95m_contextual_foote_loso_tuned` uses MERT-v1-95M weights loaded as a standard Wav2Vec2Model with exact state-dict match. No Hugging Face remote code was executed.
- MERT embeddings were cached per song under `experiments/mert_structure_segmentation/cache/`.
