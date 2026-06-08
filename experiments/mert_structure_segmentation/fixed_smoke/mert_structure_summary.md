# MERT Embedding Structure Segmentation

MERT-v1-95M was loaded from local Hugging Face weights as a standard Wav2Vec2Model with exact state-dict match. No remote repository code was executed.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_coarse | mert95m_contextual_foote_fixed | 0.369 | 0.371 | 0.370 | 159 | 160 |
| manual_fine | mert95m_contextual_foote_fixed | 0.427 | 0.430 | 0.429 | 244 | 246 |
