# Signal Callability Experiments

This folder stores reproducible signal-processing experiments for the final project.

## Experiment 1: Beat-Synchronous Signal Features and Callability

Date: 2026-06-07

Script:

```powershell
.\.venv\Scripts\python.exe scripts\callability_signal_experiment.py --all --sr 11025 --hop-length 512
```

Runtime environment used during local runs:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLCONFIGDIR='D:\yetiger\runtime_cache\matplotlib'
$env:NUMBA_CACHE_DIR='D:\yetiger\runtime_cache\numba'
```

### Goal

Convert the current annotation-centric YesTiger data into a signal-processing experiment:

```text
audio + downbeat grid
-> bar-level signal features
-> callability curves
-> call-role prediction experiments
```

### Features

Frame-level features:

- RMS energy
- onset strength
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flatness
- 13-dimensional MFCC
- 12-dimensional chroma

Bar-level derived features:

- energy
- onset
- vocal-density proxy
- beat stability
- timbre novelty
- harmony novelty
- energy novelty
- onset novelty
- fused novelty
- callability scores for `keepspace`, `rhythmcall`, `mix`, and `underground_gei`

The current vocal-density estimate is a lightweight spectral proxy. It is not source-separated vocal energy. The optional next step is to replace or compare it with Demucs-based vocal energy ratio.

### Outputs

Aggregate:

- `aggregate_signal_metrics.json`
- `aggregate_signal_summary.md`
- `aggregate_callability_confusion.png`
- `loso_predictions.jsonl`
- `loso_audio_rf_confusion.png`

Per song:

- `<song_id>.signal_bars.jsonl`
- `<song_id>.signal_metrics.json`
- `<song_id>.signal_summary.md`
- `<song_id>.merged.target_manual_grid.call_spans.json`
- `<song_id>.merged.target_manual_grid.callbook.md`
- `<song_id>.merged.<method>.call_spans.json`
- `<song_id>.merged.<method>.callbook.md`
- `<song_id>.callability_curves.png`
- `<song_id>.self_similarity.png`
- `<song_id>.self_similarity.npy`
- `<song_id>.callability_confusion.png`

Recommended presentation figures:

- `poppindream/poppindream.callability_curves.png`
- `poppindream/poppindream.self_similarity.png`
- `loso_audio_rf_confusion.png`

### Results

Dataset: 14 annotated songs, 1608 bars.

| Method | Accuracy | Macro-F1 | Bars |
|---|---:|---:|---:|
| structure_baseline | 0.386 | 0.254 | 1608 |
| callability_rule | 0.328 | 0.229 | 1608 |
| loso_structure_rf | 0.470 | 0.457 | 1608 |
| loso_audio_rf | 0.509 | 0.485 | 1608 |

Boundary detection against human call-span role changes, with a tolerance of +/-1 bar. These are call-role change boundaries, not native allin1 call-role labels:

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure_boundary | 0.655 | 0.288 | 0.400 |
| fused_novelty_topk | 0.405 | 0.405 | 0.405 |

Music-section boundary comparison. `manual_fine` keeps all human music-section boundaries. `manual_coarse` first collapses fine-grained human labels into an allin1-like coarse label set, which is fairer because the manual annotations contain more section types than allin1:

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual_fine | allin1_structure | 0.889 | 0.393 | 0.545 | 244 | 108 |
| manual_fine | fused_novelty_topk | 0.320 | 0.320 | 0.320 | 244 | 244 |
| manual_coarse | allin1_structure | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual_coarse | fused_novelty_topk | 0.289 | 0.289 | 0.289 | 159 | 159 |

Merged-span comparison after converting bar-level role predictions into continuous call spans:

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure_baseline | 0.387 | 0.157 | 0.199 |
| callability_rule | 0.331 | 0.129 | 0.412 |
| loso_structure_rf | 0.481 | 0.303 | 0.418 |
| loso_audio_rf | 0.501 | 0.316 | 0.479 |

### Interpretation

The hand-written callability rule is useful as an interpretable curve, but it is not yet a strong classifier. The more important comparison is the leave-one-song-out Random Forest experiment:

- structure-only RF: macro-F1 = 0.457
- structure + audio-signal RF: macro-F1 = 0.485

This shows that the added signal-processing features provide measurable downstream information beyond the coarse allin1 structure labels. The improvement is modest, but it is enough for the Week 14/Wednesday presentation as a first controlled experiment.

For call-role boundary detection, fused novelty has slightly higher overall F1 than structure boundaries, but the gain is small. This suggests that raw novelty alone is not sufficient for robust call-window segmentation; it should be combined with vocal density, repeated-section context, and role-aware decoding.

For music-section boundary detection, allin1 is still a strong coarse structural baseline. Its F1 improves from 0.545 against the fine manual target to 0.682 against the coarse target, which confirms that a direct comparison against all fine-grained human labels would be unfair. The signal novelty baseline is weaker for this native segmentation task, so the current result should not be presented as "novelty replaces allin1." A more accurate conclusion is that allin1 provides useful coarse structure, while the added signal-processing features improve downstream call-role and merged call-span prediction.

For merged spans, the audio-feature LOSO model performs best among the current methods. This is the most direct comparison against the human call-span annotations after post-processing the bar-level predictions into continuous intervals.

Important wording: allin1 does not provide native `call_role`. The `structure_baseline` entries in the role-prediction tables are derived baselines: either a hand-written structure-to-role heuristic or a leave-one-song-out model using only structure-related features.

### Data Leakage Note

This experiment does not use the previous `models/tiny_pipeline/*.pt` models trained on the earlier 8-song dataset. The only supervised classifier added here is a leave-one-song-out Random Forest:

- for each test song, the model is trained on the other songs only;
- the held-out song's human labels are used only for evaluation;
- annotation files are read to map human call spans to the bar grid and to provide ground-truth labels, not as input prediction features.

### Next Steps

1. Add normalized vocal-density variants and compare them.
2. Create slide-ready visualizations with fewer curves and clearer labels.
3. Try Demucs-based vocal ratio only if time permits.
4. Add a role-aware boundary decoder that combines novelty with callability scores.
