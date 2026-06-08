# Music Boundary Detector Search

| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|---:|
| manual_coarse | allin1_structure | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | hybrid_learned_rf_scale1p15 | true | 0.497 | 0.579 | 0.535 | 159 | 185 |
| manual_coarse | hybrid_learned_rf_scale1p0 | true | 0.531 | 0.535 | 0.533 | 159 | 160 |
| manual_coarse | hybrid_learned_gb_scale1p0 | true | 0.525 | 0.528 | 0.527 | 159 | 160 |
| manual_coarse | hybrid_learned_rf_scale0p8 | true | 0.557 | 0.459 | 0.503 | 159 | 131 |
| manual_coarse | hybrid_learned_gb_scale0p8 | true | 0.557 | 0.459 | 0.503 | 159 | 131 |
| manual_coarse | hybrid_learned_logreg_scale1p15 | true | 0.438 | 0.509 | 0.471 | 159 | 185 |
| manual_coarse | hybrid_learned_logreg_scale1p0 | true | 0.444 | 0.447 | 0.445 | 159 | 160 |
| manual_coarse | hybrid_learned_logreg_scale0p8 | true | 0.489 | 0.403 | 0.441 | 159 | 131 |
| manual_coarse | audio_learned_gb_scale1p15 | false | 0.378 | 0.440 | 0.407 | 159 | 185 |
| manual_coarse | audio_learned_gb_scale1p0 | false | 0.394 | 0.396 | 0.395 | 159 | 160 |
| manual_coarse | audio_learned_rf_scale1p15 | false | 0.362 | 0.421 | 0.390 | 159 | 185 |
| manual_coarse | audio_learned_rf_scale1p0 | false | 0.381 | 0.384 | 0.382 | 159 | 160 |
| manual_coarse | audio_learned_gb_scale0p8 | false | 0.420 | 0.346 | 0.379 | 159 | 131 |
| manual_coarse | audio_learned_rf_scale0p8 | false | 0.397 | 0.327 | 0.359 | 159 | 131 |
| manual_coarse | audio_learned_logreg_scale1p15 | false | 0.303 | 0.352 | 0.326 | 159 | 185 |
| manual_coarse | audio_learned_logreg_scale1p0 | false | 0.312 | 0.314 | 0.313 | 159 | 160 |
| manual_coarse | audio_formula_audio_scale1p2 | false | 0.266 | 0.321 | 0.291 | 159 | 192 |
| manual_coarse | audio_formula_rhythmic_scale1p2 | false | 0.266 | 0.321 | 0.291 | 159 | 192 |
| manual_coarse | audio_formula_contrast_scale1p2 | false | 0.260 | 0.314 | 0.285 | 159 | 192 |
| manual_coarse | audio_formula_audio_scale1p0 | false | 0.275 | 0.277 | 0.276 | 159 | 160 |
| manual_coarse | audio_formula_rhythmic_scale1p0 | false | 0.275 | 0.277 | 0.276 | 159 | 160 |
| manual_coarse | audio_formula_contrast_scale1p0 | false | 0.263 | 0.264 | 0.263 | 159 | 160 |
| manual_coarse | audio_formula_audio_scale0p8 | false | 0.290 | 0.239 | 0.262 | 159 | 131 |
| manual_coarse | audio_learned_logreg_scale0p8 | false | 0.290 | 0.239 | 0.262 | 159 | 131 |
| manual_coarse | audio_formula_rhythmic_scale0p8 | false | 0.282 | 0.233 | 0.255 | 159 | 131 |
| manual_coarse | audio_formula_contrast_scale0p8 | false | 0.260 | 0.214 | 0.234 | 159 | 131 |
| manual_fine | hybrid_learned_rf_scale1p15 | true | 0.512 | 0.594 | 0.550 | 244 | 283 |
| manual_fine | allin1_structure | true | 0.889 | 0.393 | 0.545 | 244 | 108 |
| manual_fine | hybrid_learned_rf_scale1p0 | true | 0.537 | 0.541 | 0.539 | 244 | 246 |
| manual_fine | hybrid_learned_rf_scale0p8 | true | 0.598 | 0.488 | 0.537 | 244 | 199 |
| manual_fine | hybrid_learned_gb_scale1p15 | true | 0.498 | 0.578 | 0.535 | 244 | 283 |
| manual_fine | hybrid_learned_gb_scale1p0 | true | 0.524 | 0.529 | 0.527 | 244 | 246 |
| manual_fine | hybrid_learned_gb_scale0p8 | true | 0.583 | 0.475 | 0.524 | 244 | 199 |
| manual_fine | hybrid_learned_logreg_scale1p0 | true | 0.512 | 0.516 | 0.514 | 244 | 246 |
| manual_fine | hybrid_learned_logreg_scale1p15 | true | 0.470 | 0.545 | 0.505 | 244 | 283 |
| manual_fine | hybrid_learned_logreg_scale0p8 | true | 0.558 | 0.455 | 0.501 | 244 | 199 |
| manual_fine | audio_learned_rf_scale1p15 | false | 0.431 | 0.500 | 0.463 | 244 | 283 |
| manual_fine | audio_learned_rf_scale1p0 | false | 0.459 | 0.463 | 0.461 | 244 | 246 |
| manual_fine | audio_learned_rf_scale0p8 | false | 0.508 | 0.414 | 0.456 | 244 | 199 |
| manual_fine | audio_learned_gb_scale1p15 | false | 0.420 | 0.488 | 0.452 | 244 | 283 |
| manual_fine | audio_learned_gb_scale1p0 | false | 0.435 | 0.439 | 0.437 | 244 | 246 |
| manual_fine | audio_learned_gb_scale0p8 | false | 0.482 | 0.393 | 0.433 | 244 | 199 |
| manual_fine | audio_learned_logreg_scale1p15 | false | 0.375 | 0.434 | 0.402 | 244 | 283 |
| manual_fine | audio_learned_logreg_scale1p0 | false | 0.386 | 0.389 | 0.388 | 244 | 246 |
| manual_fine | audio_learned_logreg_scale0p8 | false | 0.407 | 0.332 | 0.366 | 244 | 199 |
| manual_fine | audio_formula_rhythmic_scale1p2 | false | 0.311 | 0.377 | 0.341 | 244 | 296 |
| manual_fine | audio_formula_rhythmic_scale1p0 | false | 0.329 | 0.332 | 0.331 | 244 | 246 |
| manual_fine | audio_formula_audio_scale1p2 | false | 0.294 | 0.357 | 0.322 | 244 | 296 |
| manual_fine | audio_formula_audio_scale1p0 | false | 0.305 | 0.307 | 0.306 | 244 | 246 |
| manual_fine | audio_formula_contrast_scale1p2 | false | 0.277 | 0.336 | 0.304 | 244 | 296 |
| manual_fine | audio_formula_contrast_scale1p0 | false | 0.301 | 0.303 | 0.302 | 244 | 246 |
| manual_fine | audio_formula_contrast_scale0p8 | false | 0.332 | 0.270 | 0.298 | 244 | 199 |
| manual_fine | audio_formula_audio_scale0p8 | false | 0.322 | 0.262 | 0.289 | 244 | 199 |
| manual_fine | audio_formula_rhythmic_scale0p8 | false | 0.317 | 0.258 | 0.284 | 244 | 199 |

Best `manual_fine`: `hybrid_learned_rf_scale1p15` F1=0.550.
Best pure-audio `manual_fine`: `audio_learned_rf_scale1p15` F1=0.463.

Best `manual_coarse`: `allin1_structure` F1=0.682.
Best pure-audio `manual_coarse`: `audio_learned_gb_scale1p15` F1=0.407.
