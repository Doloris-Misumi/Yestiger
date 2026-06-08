# Music Boundary Detector Search

| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|---:|
| manual_coarse | allin1_structure | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_audio_scale0p67 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_audio_scale0p69 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_rhythmic_scale0p67 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_rhythmic_scale0p69 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_contrast_scale0p67 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_contrast_scale0p69 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_learned_logreg_scale0p67 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_learned_logreg_scale0p69 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_learned_rf_scale0p67 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_learned_rf_scale0p69 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_learned_gb_scale0p67 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_learned_gb_scale0p69 | true | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | allin1_plus_learned_rf_scale0p71 | true | 0.829 | 0.579 | 0.681 | 159 | 111 |
| manual_coarse | allin1_plus_learned_gb_scale0p71 | true | 0.829 | 0.579 | 0.681 | 159 | 111 |
| manual_coarse | allin1_plus_learned_rf_scale0p73 | true | 0.797 | 0.591 | 0.679 | 159 | 118 |
| manual_coarse | allin1_plus_rhythmic_scale0p75 | true | 0.790 | 0.591 | 0.676 | 159 | 119 |
| manual_coarse | allin1_plus_learned_rf_scale0p75 | true | 0.790 | 0.591 | 0.676 | 159 | 119 |
| manual_coarse | allin1_plus_learned_gb_scale0p75 | true | 0.790 | 0.591 | 0.676 | 159 | 119 |
| manual_coarse | allin1_plus_audio_scale0p71 | true | 0.820 | 0.572 | 0.674 | 159 | 111 |
| manual_coarse | allin1_plus_rhythmic_scale0p71 | true | 0.820 | 0.572 | 0.674 | 159 | 111 |
| manual_coarse | allin1_plus_contrast_scale0p71 | true | 0.820 | 0.572 | 0.674 | 159 | 111 |
| manual_coarse | allin1_plus_learned_logreg_scale0p71 | true | 0.820 | 0.572 | 0.674 | 159 | 111 |
| manual_coarse | allin1_plus_rhythmic_scale0p73 | true | 0.788 | 0.585 | 0.671 | 159 | 118 |
| manual_coarse | allin1_plus_learned_gb_scale0p73 | true | 0.788 | 0.585 | 0.671 | 159 | 118 |
| manual_coarse | allin1_plus_learned_logreg_scale0p75 | true | 0.782 | 0.585 | 0.669 | 159 | 119 |
| manual_coarse | allin1_plus_learned_logreg_scale0p73 | true | 0.780 | 0.579 | 0.664 | 159 | 118 |
| manual_coarse | allin1_plus_audio_scale0p73 | true | 0.771 | 0.572 | 0.657 | 159 | 118 |
| manual_coarse | allin1_plus_contrast_scale0p73 | true | 0.771 | 0.572 | 0.657 | 159 | 118 |
| manual_coarse | allin1_plus_audio_scale0p75 | true | 0.765 | 0.572 | 0.655 | 159 | 119 |
| manual_coarse | allin1_plus_contrast_scale0p75 | true | 0.765 | 0.572 | 0.655 | 159 | 119 |

Best `manual_coarse`: `allin1_structure` F1=0.682.
