# Music Boundary Detector Search

| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|---:|
| manual_coarse | allin1_structure | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_learned_gb_scale1p0 | true | 0.650 | 0.654 | 0.652 | 159 | 160 |
| manual_coarse | allin1_plus_learned_gb_scale1p15 | true | 0.607 | 0.698 | 0.649 | 159 | 183 |
| manual_coarse | allin1_plus_rhythmic_scale1p0 | true | 0.644 | 0.648 | 0.646 | 159 | 160 |
| manual_coarse | allin1_plus_learned_rf_scale1p0 | true | 0.644 | 0.648 | 0.646 | 159 | 160 |
| manual_coarse | allin1_plus_audio_scale1p0 | true | 0.625 | 0.629 | 0.627 | 159 | 160 |
| manual_coarse | allin1_plus_learned_logreg_scale1p0 | true | 0.625 | 0.629 | 0.627 | 159 | 160 |
| manual_coarse | allin1_plus_learned_rf_scale1p15 | true | 0.585 | 0.673 | 0.626 | 159 | 183 |
| manual_coarse | allin1_plus_contrast_scale1p0 | true | 0.619 | 0.623 | 0.621 | 159 | 160 |
| manual_coarse | allin1_plus_rhythmic_scale1p15 | true | 0.579 | 0.667 | 0.620 | 159 | 183 |
| manual_coarse | allin1_plus_learned_gb_scale1p3 | true | 0.549 | 0.704 | 0.617 | 159 | 204 |
| manual_coarse | allin1_plus_audio_scale1p15 | true | 0.574 | 0.660 | 0.614 | 159 | 183 |
| manual_coarse | allin1_plus_contrast_scale1p15 | true | 0.563 | 0.648 | 0.602 | 159 | 183 |
| manual_coarse | allin1_plus_learned_rf_scale1p3 | true | 0.534 | 0.686 | 0.601 | 159 | 204 |
| manual_coarse | allin1_plus_audio_scale1p3 | true | 0.529 | 0.679 | 0.595 | 159 | 204 |
| manual_coarse | allin1_plus_rhythmic_scale1p3 | true | 0.529 | 0.679 | 0.595 | 159 | 204 |
| manual_coarse | allin1_plus_contrast_scale1p3 | true | 0.529 | 0.679 | 0.595 | 159 | 204 |
| manual_coarse | allin1_plus_learned_gb_scale1p5 | true | 0.492 | 0.742 | 0.591 | 159 | 240 |
| manual_coarse | allin1_plus_learned_logreg_scale1p15 | true | 0.552 | 0.635 | 0.591 | 159 | 183 |
| manual_coarse | allin1_plus_learned_logreg_scale1p3 | true | 0.525 | 0.673 | 0.590 | 159 | 204 |
| manual_coarse | allin1_plus_learned_logreg_scale1p5 | true | 0.487 | 0.736 | 0.586 | 159 | 240 |
| manual_coarse | allin1_plus_learned_rf_scale1p5 | true | 0.479 | 0.723 | 0.576 | 159 | 240 |
| manual_coarse | allin1_plus_audio_scale1p5 | true | 0.475 | 0.717 | 0.571 | 159 | 240 |
| manual_coarse | allin1_plus_learned_gb_scale1p8 | true | 0.437 | 0.786 | 0.562 | 159 | 286 |
| manual_coarse | allin1_plus_rhythmic_scale1p5 | true | 0.467 | 0.704 | 0.561 | 159 | 240 |
| manual_coarse | allin1_plus_contrast_scale1p5 | true | 0.463 | 0.698 | 0.556 | 159 | 240 |
| manual_coarse | allin1_plus_learned_logreg_scale1p8 | true | 0.430 | 0.774 | 0.553 | 159 | 286 |
| manual_coarse | allin1_plus_learned_rf_scale1p8 | true | 0.427 | 0.767 | 0.548 | 159 | 286 |
| manual_coarse | allin1_plus_audio_scale1p8 | true | 0.406 | 0.730 | 0.521 | 159 | 286 |
| manual_coarse | allin1_plus_rhythmic_scale1p8 | true | 0.406 | 0.730 | 0.521 | 159 | 286 |
| manual_coarse | allin1_plus_contrast_scale1p8 | true | 0.399 | 0.717 | 0.512 | 159 | 286 |
| manual_fine | allin1_plus_learned_rf_scale1p0 | true | 0.594 | 0.594 | 0.594 | 244 | 244 |
| manual_fine | allin1_plus_learned_logreg_scale1p0 | true | 0.590 | 0.590 | 0.590 | 244 | 244 |
| manual_fine | allin1_plus_learned_gb_scale1p0 | true | 0.586 | 0.586 | 0.586 | 244 | 244 |
| manual_fine | allin1_plus_learned_logreg_scale1p15 | true | 0.543 | 0.627 | 0.582 | 244 | 282 |
| manual_fine | allin1_plus_learned_gb_scale1p15 | true | 0.543 | 0.627 | 0.582 | 244 | 282 |
| manual_fine | allin1_plus_learned_rf_scale1p15 | true | 0.532 | 0.615 | 0.570 | 244 | 282 |
| manual_fine | allin1_plus_learned_gb_scale1p5 | true | 0.469 | 0.705 | 0.563 | 244 | 367 |
| manual_fine | allin1_plus_learned_gb_scale1p3 | true | 0.497 | 0.648 | 0.562 | 244 | 318 |
| manual_fine | allin1_plus_learned_rf_scale1p3 | true | 0.494 | 0.643 | 0.559 | 244 | 318 |
| manual_fine | allin1_plus_learned_logreg_scale1p3 | true | 0.491 | 0.639 | 0.555 | 244 | 318 |
| manual_fine | allin1_plus_learned_rf_scale1p5 | true | 0.455 | 0.684 | 0.547 | 244 | 367 |
| manual_fine | allin1_structure | true | 0.889 | 0.393 | 0.545 | 244 | 108 |
| manual_fine | allin1_plus_learned_logreg_scale1p5 | true | 0.452 | 0.680 | 0.543 | 244 | 367 |
| manual_fine | allin1_plus_audio_scale1p0 | true | 0.541 | 0.541 | 0.541 | 244 | 244 |
| manual_fine | allin1_plus_learned_gb_scale1p8 | true | 0.420 | 0.758 | 0.541 | 244 | 440 |
| manual_fine | allin1_plus_learned_rf_scale1p8 | true | 0.418 | 0.754 | 0.538 | 244 | 440 |
| manual_fine | allin1_plus_learned_logreg_scale1p8 | true | 0.411 | 0.742 | 0.529 | 244 | 440 |
| manual_fine | allin1_plus_contrast_scale1p0 | true | 0.525 | 0.525 | 0.525 | 244 | 244 |
| manual_fine | allin1_plus_rhythmic_scale1p0 | true | 0.520 | 0.520 | 0.520 | 244 | 244 |
| manual_fine | allin1_plus_audio_scale1p15 | true | 0.475 | 0.549 | 0.510 | 244 | 282 |
| manual_fine | allin1_plus_rhythmic_scale1p15 | true | 0.475 | 0.549 | 0.510 | 244 | 282 |
| manual_fine | allin1_plus_contrast_scale1p15 | true | 0.472 | 0.545 | 0.506 | 244 | 282 |
| manual_fine | allin1_plus_rhythmic_scale1p3 | true | 0.443 | 0.578 | 0.502 | 244 | 318 |
| manual_fine | allin1_plus_rhythmic_scale1p5 | true | 0.414 | 0.623 | 0.498 | 244 | 367 |
| manual_fine | allin1_plus_audio_scale1p3 | true | 0.437 | 0.570 | 0.495 | 244 | 318 |
| manual_fine | allin1_plus_contrast_scale1p3 | true | 0.428 | 0.557 | 0.484 | 244 | 318 |
| manual_fine | allin1_plus_contrast_scale1p5 | true | 0.392 | 0.590 | 0.471 | 244 | 367 |
| manual_fine | allin1_plus_audio_scale1p5 | true | 0.390 | 0.586 | 0.468 | 244 | 367 |
| manual_fine | allin1_plus_rhythmic_scale1p8 | true | 0.364 | 0.656 | 0.468 | 244 | 440 |
| manual_fine | allin1_plus_audio_scale1p8 | true | 0.355 | 0.639 | 0.456 | 244 | 440 |
| manual_fine | allin1_plus_contrast_scale1p8 | true | 0.352 | 0.635 | 0.453 | 244 | 440 |

Best `manual_fine`: `allin1_plus_learned_rf_scale1p0` F1=0.594.

Best `manual_coarse`: `allin1_structure` F1=0.682.
