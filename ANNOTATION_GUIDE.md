# YesTiger 标注指南

欢迎参与 YesTiger 数据集标注！本指南将帮助你快速上手，为偶像/动漫歌曲演唱会标注 **应援歌单（Callbook）**。

---

## 1. 项目简介

**YesTiger** 是一个自动化应援歌单生成系统。通过分析音频信号和标注数据，训练模型来预测一首歌应该在哪些位置进行什么类型的观众互动（MIX、CALL、地下艺等）。

你的工作是：**为新歌标注音乐结构段落和观众互动动作**，帮助扩充训练数据集。

---

## 2. 文件结构

```
yetiger/
├── songs/                  # 原始音频 (.mp3)
├── struct/                 # allin1 自动分析的结构文件 (.json)
├── demix/                  # 音源分离结果（自动生成）
├── annotations/            # 标注文件（你主要编辑的目录）
│   ├── template.annotation.json   # 标注模板
│   ├── godknows/                  # 已标注歌曲示例
│   │   └── godknows.annotation.json
│   ├── dokimekiexperience/
│   │   └── dokimekiexperience.annotation.json
│   └── ...
├── knowledge/
│   ├── call_mix_library.json      # 动作知识库（所有可选动作）
│   └── call_mix_library.md        # 动作知识库（可读版）
├── scripts/
│   └── predict_tiny_pipeline.py   # 模型预测脚本
└── outputs/
    └── model_pipeline/            # 模型预测输出
```

---

## 3. 准备工作

### 3.1 环境要求

- **音频文件**：放在 `songs/` 目录下，MP3 格式
- **文本编辑器**：VS Code（推荐）或任何 JSON 编辑器
- **Python 环境**（可选，如需本地运行预测）：见 `README.md`

### 3.2 了解歌曲

在标注前，请先完整听一遍歌曲，注意：
- 歌曲的结构（前奏→主歌→副歌→间奏…）
- 哪里有明显的器乐段落
- 观众通常会在哪里互动

---

## 4. 标注流程

### 步骤 1：获取 struct 文件

struct 文件由 `allin1` 工具自动生成，包含拍点、重拍和初步的段落切分。如果你已有 struct 文件，跳到步骤 2。

如果你需要自己生成 struct，联系项目管理员，或运行：

```bash
allin1 songs/你的歌曲.mp3 -o struct --no-multiprocess -d cpu
```

输出文件会保存在 `struct/你的歌曲.json`。

### 步骤 2：修正 struct 中的音乐段落标签

打开 `struct/你的歌曲.json`，找到 `segments` 数组，修正每个段落的 `label` 字段。

**可用的音乐标签（music_label）：**

| 标签 | 含义 | 典型特征 |
|---|---|---|
| `intro` | 前奏 | 歌曲开头，通常无人声或少量人声 |
| `verse` | 主歌 | 叙事段落，人声较密集 |
| `pre_chorus` | 预副歌 | 能量上升，推向副歌 |
| `pre_chorus_build` | 预副歌推进 | 预副歌末尾，能量急剧上升 |
| `chorus` | 副歌 | 高潮段落，旋律最抓耳 |
| `post_chorus` | 副歌后段 | 副歌后的回落段落 |
| `instrumental_break` | 器乐间奏 | 纯器乐段落 |
| `interlude` | 插曲段落 | 较短的过渡段落 |
| `bridge` | 桥段 | 结构变化段落，常在第二段副歌后 |
| `solo` | 独奏 | 吉他/键盘等乐器独奏 |
| `chant` | 呼喊段落 | 适合观众呼喊的段落 |
| `outro` | 尾奏 | 歌曲结尾 |
| `end` | 结束 | 最后收尾（通常静音或淡出） |

**示例：**

```json
{
  "start": 29.2,
  "end": 39.86,
  "label": "verse"
}
```

> **合并相邻同标签段落**：如果相邻两个 segment 标签相同，建议手动合并。例如 0-4s 和 4-8s 都是 `intro`，合并为 0-8s。

### 步骤 3：同步到 annotation 文件

联系项目管理员运行同步脚本，或手动创建 annotation 文件。

annotation 文件格式见第 5 节。annotation 中的 `segments` 应与 struct 保持一致。

### 步骤 4：标注 call_spans（核心工作）

在 annotation 文件中，编辑 `call_spans` 数组。每个 call_span 描述一个时间段内的观众互动。

**call_span 字段：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `start` | float | 开始时间（秒） |
| `end` | float | 结束时间（秒） |
| `call_role` | string | 互动类型（见下方） |
| `recommended_actions` | string[] | 推荐的具体动作 |
| `notes` | string | 备注（可选） |

**互动类型（call_role）：**

| call_role | 含义 | 说明 |
|---|---|---|
| `keepspace` | 保持安静 | 不应互动，留给观众欣赏 |
| `rhythmcall` | 节奏 CALL | 跟节奏喊口号、拍手等 |
| `mix` | MIX | 日式应援 MIX |
| `underground_gei` | 地下艺 | 地下偶像风格的动作/呼喊 |

**可用动作（recommended_actions）：**

完整动作列表参见 `knowledge/call_mix_library.md`。以下是常用动作速查：

| 动作 ID | 名称 | 类型 | 适用场景 |
|---|---|---|---|
| `standard_mix` | Standard MIX | mix | 前奏、间奏、副歌后 |
| `japanese_mix` | Japanese MIX | mix | 第二前奏、器乐间奏 |
| `ppph` | PPPH | rhythmcall | 主歌、预副歌 |
| `clap` | Clap | rhythmcall | 任何适合拍手的段落 |
| `hai_hai` | Hai Hai | rhythmcall | 副歌、间奏 |
| `name_call` | Name Call | rhythmcall | solo 空档、短句间 |
| `fufu_call` | Fu Fu Fu Call | rhythmcall | 预副歌推进 |
| `ietora` | Ietora | mix | 预副歌推进、副歌入口 |
| `haiseno_activation` | Haiseno Activation | mix | 1-2 小节的过渡 |
| `tiger_fire_activation` | Tiger Fire | mix | 副歌后、前奏返回 |
| `aiai_mix` | Aiai MIX | mix | 2-4 小节高能空档 |
| `bismarck_mix` | Bismarck MIX | mix | 6-7 个八拍空档 |

**示例：**

```json
{
  "start": 27.82,
  "end": 47.02,
  "call_role": "rhythmcall",
  "recommended_actions": ["clap"],
  "notes": "主歌节奏稳定，适合拍手"
}
```

### 步骤 5：验证

完成标注后，检查以下事项：
- [ ] `segments` 从 0.0 秒开始，到歌曲结束为止，无间断、无重叠
- [ ] 每个 segment 的 `music_label` 是有效标签
- [ ] `call_spans` 覆盖了整个歌曲（含 `keepspace` 类型）
- [ ] `call_spans` 无时间重叠
- [ ] `recommended_actions` 中的动作 ID 存在于知识库中

---

## 5. Annotation 文件格式

完整格式参考（以 godknows 为例）：

```json
{
  "annotation_version": "0.2.0",
  "song": {
    "song_id": "godknows",
    "title": "God knows...",
    "artist": "Aya Hirano",
    "franchise": "suzumiya_haruhi",
    "audio_path": "songs/Godknows.mp3",
    "bpm": 150,
    "call_bpm": 75,
    "call_bar_multiplier": 0.5,
    "meter": "4/4"
  },
  "segments": [
    {
      "start": 0.0,
      "end": 27.82,
      "music_label": "intro",
      "notes": ""
    }
  ],
  "call_spans": [
    {
      "start": 0.0,
      "end": 2.33,
      "call_role": "keepspace",
      "recommended_actions": [],
      "notes": ""
    },
    {
      "start": 2.33,
      "end": 27.82,
      "call_role": "mix",
      "recommended_actions": ["lin_xiu_mix"],
      "notes": ""
    }
  ]
}
```

### song 字段说明

| 字段 | 说明 |
|---|---|
| `song_id` | 歌曲唯一 ID，用小写英文和下划线 |
| `title` | 歌曲标题 |
| `artist` | 艺术家 |
| `franchise` | 所属作品/企划 |
| `bpm` | 音乐 BPM（曲速） |
| `call_bpm` | 应援 BPM（通常 = bpm，快歌可能 = bpm/2） |
| `call_bar_multiplier` | 应援小节倍率（call_bpm / bpm） |
| `meter` | 拍号，通常是 4/4 |

---

## 6. 标注 Tips

1. **先粗后细**：先划分大的音乐段落，再逐个段标注 call_spans
2. **参考已标注歌曲**：`annotations/godknows/` 是完整标注的好例子
3. **keepspace 很重要**：不是所有地方都需要互动，人声密集处通常保持安静
4. **动作不要太密**：一般每个 call_span 推荐 1-3 个候选动作即可
5. **时长约束**：动作有最短/最长小节数限制，参考 `call_mix_library.json` 中的 `allowed_bars`
6. **不确定就留空**：如果不确定用什么动作，可以只标 `call_role`，`recommended_actions` 留空

---

## 7. 常见问题

**Q: `call_bpm` 和 `bpm` 有什么区别？**
A: 对于快歌（BPM > 120），应援通常用半速打拍，此时 `call_bpm = bpm / 2`，`call_bar_multiplier = 0.5`。

**Q: 动作的最小/最大时长怎么算？**
A: 以 `call_bar_multiplier` 为基准。例如 `call_bar_multiplier=0.5` 时，1 个 call_bar = 2 个 music_bar。动作的 `allowed_bars` 以 call_bar 为单位。

**Q: 可以用 `inst` 代替 `instrumental_break` 吗？**
A: 可以，两个标签等价，保持一致即可。

---

## 8. 提交标注

完成后，将 annotation 文件提交到 GitHub：

```bash
git add annotations/你的歌曲/
git commit -m "annotations: add 你的歌曲"
git push
```

或者直接将 `.annotation.json` 文件发给项目管理员。

---

> 有任何问题，请通过 GitHub Issues 或项目群组联系。
> 感谢你的贡献！🎉
