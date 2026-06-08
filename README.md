# YesTiger

YesTiger is a local idol/anison live callbook and music-structure annotation workspace. The current focus is building a small training pipeline that learns from human-approved song annotations:

```text
audio/allin1 struct + human annotation
-> bar-level training rows
-> MusicSegmenter
-> CallSlotter
-> ActionRanker
-> predicted segments + call_spans + action candidates
```

YesTiger 是一个本地偶像 / Anisong 现场 callbook 与歌曲结构标注工作区。当前重点是把人工确认过的歌曲标注转成一个小规模训练 pipeline：

```text
音频 / allin1 结构 + 人工 annotation
-> bar-level 训练样本
-> 音乐结构模型 MusicSegmenter
-> call 区间模型 CallSlotter
-> action 排序模型 ActionRanker
-> 预测 segments + call_spans + action candidates
```

## Current Data / 当前数据

- Annotated songs / 已标注歌曲：8
- Bar-level rows / 小节级样本：911
- Action ranking pairs / action 排序样本：4712
- Positive action span groups / 正样本 action span 组：209

Main annotation format:

主要标注格式：

- `annotations/*/*.annotation.json`
- `segments`: music structure only, such as `intro`, `verse`, `pre_chorus`, `chorus`, `solo`, `outro`
- `call_spans`: call layer only, such as `keepspace`, `rhythmcall`, `mix`, `underground_gei`
- `knowledge/call_mix_library.json`: action knowledge library used by the ranker

## Pipeline / 训练流程

### New song prediction / 新歌模型流预测

The preferred path is now the model pipeline:

现在推荐的新歌流程是模型流：

```text
mp3 -> allin1 struct -> tiny pipeline prediction -> predicted annotation + callbook
```

Run allin1 first:

先跑 allin1：

```powershell
.\run_allin1.bat .\songs\NewSong.mp3 -o .\struct --no-multiprocess -d cpu
```

Then run the tiny model pipeline:

再跑小模型 pipeline：

```powershell
.\.venv\Scripts\python.exe scripts\predict_tiny_pipeline.py `
  --struct .\struct\NewSong.json `
  --audio .\songs\NewSong.mp3 `
  --song-id new_song `
  --title "New Song"
```

Outputs:

输出：

```text
outputs\model_pipeline\new_song\new_song.model.annotation.json
outputs\model_pipeline\new_song\new_song.model.callbook.json
outputs\model_pipeline\new_song\new_song.model.callbook.md
outputs\model_pipeline\new_song\new_song.model.bars.jsonl
```

The predicted annotation is a machine draft. Review it manually before moving it into `annotations/<song_id>/<song_id>.annotation.json`.

预测 annotation 是机器草稿。正式放入 `annotations/<song_id>/<song_id>.annotation.json` 前需要人工检查。

### Legacy rule pipeline / 旧规则流

The old rule-based flow is packaged behind one wrapper and kept only as a fallback/reference path:

旧规则流已经收进一个 wrapper，只作为 fallback / reference 保留：

```powershell
.\.venv\Scripts\python.exe scripts\run_legacy_rule_pipeline.py `
  --audio .\songs\NewSong.mp3 `
  --struct .\struct\NewSong.json
```

If `--struct` is omitted, the wrapper will run allin1 first.

如果省略 `--struct`，wrapper 会先跑 allin1。

### 1. Validate annotations / 验证标注

```powershell
python scripts\validate_annotation.py
```

The validator checks schema, known action IDs, action/category consistency, duration constraints, and rough alignment against allin1 struct files.

验证器会检查 schema、action id 是否存在、action/category 是否匹配、长度约束，以及和 allin1 struct 的粗略对齐。

### 2. Build training data / 构建训练数据

```powershell
.\.venv\Scripts\python.exe scripts\build_pipeline_dataset.py --all
```

Outputs:

输出：

- `datasets/pipeline/bar_rows.jsonl`: one row per bar/timestep
- `datasets/pipeline/action_pairs.jsonl`: one candidate action pair per call span/action candidate
- `datasets/pipeline/manifest.json`: song list, label vocabs, counts
- `datasets/pipeline/songs/*.sequence.json`: per-song sequence view

The bar rows contain both music targets and call targets:

bar row 同时包含音乐结构目标和 call 目标：

```json
{
  "song_id": "mayoiuta",
  "bar_index": 7,
  "start": 16.33,
  "end": 18.85,
  "features": {
    "relative_pos": 0.077254,
    "allin1_struct_label": "verse"
  },
  "target": {
    "music_label": "verse",
    "boundary": 1,
    "call_role": "rhythmcall",
    "call_boundary": 1,
    "recommended_actions": ["clap"]
  }
}
```

### 3. Train tiny models / 训练小模型

```powershell
.\.venv\Scripts\python.exe scripts\train_tiny_pipeline.py --epochs 220 --action-epochs 280
```

Outputs:

输出：

- `models/tiny_pipeline/segmenter.pt`
- `models/tiny_pipeline/call_slotter.pt`
- `models/tiny_pipeline/action_ranker.pt`
- `models/tiny_pipeline/metadata.json`
- `models/tiny_pipeline/metrics.json`
- `models/tiny_pipeline/train_predictions.json`

## Model Design / 模型设计

### MusicSegmenter / 音乐结构模型

Tiny BiLSTM sequence tagger.

小型 BiLSTM 时序标注模型。

Input:

输入：

- bar position
- bar duration ratio
- observed/extrapolated downbeat flags
- allin1 struct label

Output:

输出：

- `music_label`
- segment boundary at bar start

### CallSlotter / Call 区间模型

Tiny BiLSTM sequence tagger.

小型 BiLSTM 时序标注模型。

Input:

输入：

- bar features
- music label, from oracle annotations during training or predicted labels during pipeline prediction

Output:

输出：

- `call_role`: `keepspace`, `rhythmcall`, `mix`, `underground_gei`
- call boundary at bar start

### ActionRanker / Action 排序模型

Small MLP binary ranker.

小型 MLP 二分类排序器。

It does not blindly classify across every action. Instead, the knowledge library provides candidates by `call_role`, and the ranker scores each candidate using:

它不是从所有 action 中硬分类，而是先由知识库按 `call_role` 给出候选，再让排序器给候选打分。特征包括：

- duration in bars
- music label
- allin1 struct label
- action category, intensity, risk
- action min/max bar requirements
- context overlap

## Current Tiny Training Result / 当前小训练结果

This is a training-set sanity check on only 8 songs. It is intentionally allowed to overfit; do not read these numbers as generalization quality yet.

这是只在 8 首歌训练集上的 sanity check，允许过拟合。下面数字不能当作泛化能力，只说明 pipeline 已经能跑通并学到现有标注。

From `models/tiny_pipeline/metrics.json`:

来自 `models/tiny_pipeline/metrics.json`：

- MusicSegmenter train label accuracy: `0.9978`
- MusicSegmenter train boundary F1: `1.0000`
- CallSlotter train label accuracy with predicted segments: `0.9978`
- CallSlotter train boundary F1 with predicted segments: `0.9436`
- ActionRanker train top-1 hit rate: `0.7895`
- ActionRanker train top-3 hit rate: `0.9856`

Example prediction for `mayoiuta` is in:

`mayoiuta` 的训练集预测样例在：

```text
models/tiny_pipeline/train_predictions.json
```

## Recommended Next Steps / 建议下一步

1. Add more annotated songs, then rebuild data and retrain.
2. Add real allin1 embeddings/activations into `bar_rows.jsonl`.
3. Add a held-out song split once there are 20+ songs.
4. Add post-processing rules for minimum segment length and preferred downbeat snapping.
5. Use `ActionRanker` as a candidate scorer, while continuing to keep action duration/category rules in `knowledge/call_mix_library.json`.

1. 继续增加已标注歌曲，然后重建数据、重新训练。
2. 把真实 allin1 embeddings / activations 接入 `bar_rows.jsonl`。
3. 到 20 首以上后加入按歌曲划分的验证集。
4. 增加最短段落长度、优先贴近 downbeat 的后处理规则。
5. `ActionRanker` 只作为候选打分器，action 的长度和分类规则继续保留在 `knowledge/call_mix_library.json`。

## Notes / 说明

- Train/test split must be by song, not by random bars.
- Current generated data treats uncovered call regions as `keepspace`.
- Struct files are produced by allin1 and stored in `struct/*.json`.
- The root pipeline is local-first and currently does not require network access.

- 训练 / 测试必须按歌曲切分，不能随机按 bar 切分。
- 当前数据构建会把没有 call_span 覆盖的区域视为 `keepspace`。
- `struct/*.json` 来自 allin1 输出。
- 当前 pipeline 是本地优先，不需要联网。
