# Method Notes

## 01_cbm_block_dp

- Inspired by CBM / autosimilarity segmentation: build a bar-level self-similarity matrix, score candidate blocks by within-block homogeneity, and solve the best whole-song segmentation by dynamic programming.
- `cbm_dp_*_fixed` uses fixed hyperparameters and no manual labels.
- `cbm_dp_loso_tuned` selects CBM hyperparameters on the other songs only.
- `cbm_dp_plus_novelty_fill_loso_tuned` first runs CBM-DP, then adjusts the number of boundaries with a Foote-style novelty curve; this is a CBM + local novelty hybrid.

## 02_classic_msaf_like

- `foote_checkerboard_*` is a checkerboard novelty detector on the self-similarity matrix.
- `agglomerative_*` is a temporally constrained agglomerative clustering baseline, similar in spirit to classic MSAF segmentation recipes.
- `spectral_*` is an optional spectral-clustering segmentation baseline on the self-similarity matrix.

## 03_modern_embedding_interface

- MERT/BEATs/OpenBEATs are stronger modern feature extractors, but they are not direct structure segmenters. The intended extension is: extract bar-level embeddings, replace `bar_feature_matrix`, then reuse CBM-DP/Foote/spectral decoding.
- This run does not download large pretrained models; it keeps the interface and method note separate from the measured results.

## External Package Check

```json
{
  "barmuscomp": {
    "installed": true,
    "origin": "D:\\yetiger\\.venv\\lib\\site-packages\\barmuscomp\\__init__.py"
  },
  "as_seg": {
    "installed": true,
    "origin": "D:\\yetiger\\.venv\\lib\\site-packages\\as_seg\\__init__.py",
    "importable": false,
    "note": "SyntaxError on Python 3.9: invalid syntax (CBM_algorithm.py, line 319)"
  },
  "msaf": {
    "installed": false,
    "origin": null
  },
  "transformers": {
    "installed": false,
    "origin": null
  }
}
```
