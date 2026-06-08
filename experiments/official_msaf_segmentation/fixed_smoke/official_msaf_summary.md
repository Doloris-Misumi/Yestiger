# Official MSAF Module Segmentation

MSAF's full `msaf.process` path timed out on full MP3 files in this environment, and CNMF cannot import because cvxopt fails to load a DLL. This run uses official MSAF Foote/SF Segmenter modules on the existing bar-level features.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_coarse | official_msaf_foote_fixed_full_bar_features | 0.200 | 0.006 | 0.012 | 159 | 5 |
| manual_fine | official_msaf_foote_fixed_full_bar_features | 0.200 | 0.004 | 0.008 | 244 | 5 |
