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
| 01_cbm_block_dp | manual_coarse | cbm_dp_plus_novelty_fill_loso_tuned | true | 0.390 | 0.434 | 0.411 | 159 | 177 |
| 01_cbm_block_dp | manual_fine | cbm_dp_plus_novelty_fill_loso_tuned | true | 0.430 | 0.430 | 0.430 | 244 | 244 |

## Best Per Target

- `manual_fine` best new method: `cbm_dp_plus_novelty_fill_loso_tuned` (01_cbm_block_dp), F1=0.430.
- `manual_coarse` best new method: `cbm_dp_plus_novelty_fill_loso_tuned` (01_cbm_block_dp), F1=0.411.
