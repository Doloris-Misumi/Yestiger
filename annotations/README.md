# Annotations / 标注说明

This folder stores human-approved song annotations for YesTiger.

本目录保存 YesTiger 的人工确认歌曲标注。

## Current Schema / 当前 Schema

Current annotation files use `annotation_version: 0.2.0`.

当前标注文件使用 `annotation_version: 0.2.0`。

```json
{
  "annotation_version": "0.2.0",
  "song": {
    "song_id": "mayoiuta",
    "title": "迷星叫",
    "artist": "MyGO!!!!!",
    "franchise": "bang_dream",
    "audio_path": "songs/迷星叫.mp3",
    "bpm": 95,
    "call_bpm": 95,
    "call_bar_multiplier": 1.0,
    "meter": "4/4"
  },
  "segments": [],
  "call_spans": []
}
```

## `segments` / 音乐结构层

`segments` should describe music structure only.

`segments` 只描述音乐结构。

```json
{
  "start": 36.53,
  "end": 41.58,
  "music_label": "pre_chorus",
  "notes": ""
}
```

Supported labels:

支持的标签：

```text
intro
verse
pre_chorus
pre_chorus_build
chorus
post_chorus
interlude
instrumental_break
bridge
solo
outro
end
chant
```

## `call_spans` / Call 层

`call_spans` should describe call/mix behavior only.

`call_spans` 只描述 call / mix 行为。

```json
{
  "start": 73.18,
  "end": 83.27,
  "call_role": "mix",
  "recommended_actions": ["bandor_mix"],
  "notes": ""
}
```

Supported roles:

支持的角色：

```text
keepspace
rhythmcall
mix
underground_gei
```

`recommended_actions` must use IDs from:

`recommended_actions` 必须来自：

```text
knowledge/call_mix_library.json
```

## Validation / 验证

Run:

运行：

```powershell
python scripts\validate_annotation.py
```

The validator checks:

验证器会检查：

- timeline order
- music labels
- call roles
- known action IDs
- action category vs. call role
- rough duration and downbeat alignment

- 时间顺序
- 音乐结构标签
- call 角色
- action ID 是否存在
- action 分类是否匹配 call role
- 粗略长度和 downbeat 对齐

## Dataset Export / 数据集导出

For training data:

导出训练数据：

```powershell
.\.venv\Scripts\python.exe scripts\build_pipeline_dataset.py --all
```

Output:

输出：

```text
datasets/pipeline/bar_rows.jsonl
datasets/pipeline/action_pairs.jsonl
datasets/pipeline/manifest.json
datasets/pipeline/songs/*.sequence.json
```
