# MERT Embedding Structure Segmentation

MERT-v1-95M was loaded from local Hugging Face weights as a standard Wav2Vec2Model with exact state-dict match. No remote repository code was executed.

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_coarse | mert95m_contextual_foote_loso_tuned | 0.453 | 0.491 | 0.471 | 159 | 172 |
| manual_fine | mert95m_contextual_foote_loso_tuned | 0.467 | 0.496 | 0.481 | 244 | 259 |
