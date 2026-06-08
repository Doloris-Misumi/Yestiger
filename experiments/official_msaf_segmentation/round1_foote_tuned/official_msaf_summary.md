# Official MSAF Module Segmentation

MSAF's full `msaf.process` path timed out on full MP3 files in this environment, and CNMF cannot import because cvxopt fails to load a DLL. This run uses official MSAF Foote/SF Segmenter modules on the existing bar-level features.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_coarse | official_msaf_foote_bar_features_loso_tuned | 0.511 | 0.151 | 0.233 | 159 | 47 |
| manual_fine | official_msaf_foote_bar_features_loso_tuned | 0.432 | 0.078 | 0.132 | 244 | 44 |
