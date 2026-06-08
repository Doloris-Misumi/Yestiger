# 周三汇报材料草稿

## 一句话主线

我们把“现场什么时候适合观众 call”转化成一个音频信号处理问题：基于小节级能量、起音、频谱变化、人声密度代理、新颖度和重复结构，估计每个小节适合观众参与的程度。

## 推荐 6 页结构

### 1. 问题背景

普通说法：

> 在现场音乐里，观众会拍手、短喊、回应节奏，或者在重要人声处保持安静。错误的 call 会遮盖人声或破坏节奏，所以我们希望从音频中自动判断“哪里适合观众参与”。

不要先讲 YesTiger。先讲普通观众互动，再说这个项目以偶像 / Anisong 为案例。

### 2. 任务定义

输入：

```text
song audio + beat/downbeat grid
```

输出：

```text
bar-level call role:
keepspace / rhythmcall / mix / underground_gei
```

中间表示：

```text
callability curve
```

解释：

> callability 是每个小节对不同观众行为的适合程度，不是歌词生成，也不是简单分类标签。

### 3. 信号处理方法

核心流程：

```text
STFT / spectral features
-> beat-synchronous pooling
-> novelty + vocal-density proxy + beat stability
-> callability scores
-> call role prediction
```

可讲特征：

- RMS energy: 现场能量
- onset strength: 鼓点和节奏清晰度
- spectral centroid/bandwidth/rolloff/flatness: 音色和频谱状态
- MFCC: 音色表征
- chroma: 和声/音高类别
- fused novelty: 段落变化
- vocal-density proxy: 人声密度代理
- self-similarity matrix: 重复副歌/重复段落

### 4. Case Study: Poppin'Dream!

用图：

```text
experiments/signal_callability/poppindream/poppindream.callability_curves.png
```

讲法：

> 上半部分是音频信号曲线：能量、起音、人声密度代理和 fused novelty。背景色表示人工标注的 call role。下半部分是四类 callability score。可以看到不同段落的音频信号状态和人工 call 行为之间有一定对应关系。

注意：

> 这张图展示的是“可解释证据”，不是最终最优分类器。

### 5. 重复结构分析

用图：

```text
experiments/signal_callability/poppindream/poppindream.self_similarity.png
```

讲法：

> self-similarity matrix 用小节级音频特征计算。对角线附近是局部连续性，非对角线的亮块表示不同位置的音乐内容相似。重复副歌或相似段落可以支持 call pattern 的复用。

### 6. 小实验结果

用表：

```text
experiments/signal_callability/aggregate_signal_summary.md
```

重点数字：

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| structure-only RF | 0.470 | 0.457 |
| structure + audio RF | 0.509 | 0.485 |
| RF + LogReg + GB soft voting | 0.527 | 0.519 |

讲法：

> 在 leave-one-song-out 设置下，加入音频信号特征后，macro-F1 从 0.457 提升到 0.485；进一步使用 RF、Logistic Regression 和 Gradient Boosting 的等权 soft voting 后，macro-F1 提升到 0.519。这说明音频特征确实提供了粗粒度结构标签之外的信息，而且不同模型捕捉到的错误模式有互补性。

合并成连续 call spans 后：

| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |
|---|---:|---:|---:|
| structure-to-role heuristic | 0.387 | 0.157 | 0.199 |
| callability rule | 0.331 | 0.129 | 0.412 |
| LOSO structure RF | 0.481 | 0.303 | 0.418 |
| LOSO audio RF | 0.501 | 0.316 | 0.479 |
| LOSO audio soft voting | 0.518 | 0.346 | 0.471 |

讲法：

> 我们不只在小节级分类，也把连续相同 role 的小节合并成 call spans，再和人工 call spans 比较。优化后的 soft-voting 模型在 time-weighted role accuracy 和 macro role IoU 上最高；旧的 audio RF 在 span boundary F1 上略高一点，说明 ensemble 更会判断 role，但边界还可以继续用专门的 boundary decoder 改进。

合并后的 Poppin'Dream 输出：

```text
experiments/signal_callability/poppindream/poppindream.merged.loso_audio_rf.callbook.md
experiments/signal_callability/poppindream/poppindream.merged.loso_audio_rf.call_spans.json
experiments/signal_callability/poppindream/poppindream.merged.loso_audio_vote_rf1_logreg1_gb1.callbook.md
experiments/signal_callability/poppindream/poppindream.merged.loso_audio_vote_rf1_logreg1_gb1.call_spans.json
```

注意：

> allin1 本身只给音乐结构分段，不给 `call_role`。所以这里的 `structure-to-role heuristic` 不是 allin1 原生输出，而是把 allin1 的结构标签映射到 call role 的一个可解释 baseline；`LOSO structure RF` 也是只用结构相关特征训练出来的 role baseline。

call-role 边界检测结果：

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| structure boundary | 0.655 | 0.288 | 0.400 |
| fused novelty top-k | 0.405 | 0.405 | 0.405 |

讲法：

> 这张表比较的是“预测边界是否接近人工 call role 变化点”，不是比较段落标签。structure boundary 精度高但召回低，说明 allin1 的大段落变化比较可靠，但会漏掉人工 call 标注中的细变化；novelty 召回更均衡，但误检也更多。

音乐段落边界的公平对比：

| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |
|---|---|---:|---:|---:|---:|---:|
| manual fine | allin1 structure | 0.889 | 0.393 | 0.545 | 244 | 108 |
| manual fine | fused novelty top-k | 0.320 | 0.320 | 0.320 | 244 | 244 |
| manual coarse | allin1 structure | 0.843 | 0.572 | 0.682 | 159 | 108 |
| manual coarse | fused novelty top-k | 0.289 | 0.289 | 0.289 | 159 | 159 |

讲法：

> 因为我的人工标注比 allin1 多了很多细段落类型，所以直接拿全部人工边界比会惩罚 allin1。这里做了两套目标：`manual fine` 保留所有人工段落边界；`manual coarse` 先把 verse/pre-chorus/chant、chorus/post-chorus、inst/interlude 等细标签合并成 allin1 更接近的粗类别，再做边界比较。结果说明 allin1 在粗音乐段落上仍然是强 baseline，而音频 novelty 单独替代不了 allin1；本项目的增益主要体现在用音频信号改进 call role / call span 预测。

## 当前代码和结果

核心脚本：

```text
scripts/callability_signal_experiment.py
```

复现实验：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLCONFIGDIR='D:\yetiger\runtime_cache\matplotlib'
$env:NUMBA_CACHE_DIR='D:\yetiger\runtime_cache\numba'
.\.venv\Scripts\python.exe scripts\callability_signal_experiment.py --all --sr 11025 --hop-length 512
```

实验日志：

```text
experiments/signal_callability/EXPERIMENT_LOG.md
```

## 老师可能会问的问题

### Q1: 这和文本标注有什么区别？

回答：

> 文本标注只是 ground truth。我们的输入特征来自音频信号，包括 STFT 频谱特征、onset、MFCC、chroma、novelty 和 self-similarity。实验目标是验证这些音频信号能否预测人工标注的 call role。

### Q2: vocal density 是怎么来的？

回答：

> 当前版本是轻量级频谱代理，不是源分离。它结合能量、起音强度、频谱平坦度和频谱重心中频性。后续可以用 Demucs 分离 vocal stem 后计算 vocal energy ratio。

### Q3: 为什么不用深度大模型？

回答：

> 本课程重点是高级信号处理，所以我们先使用可解释的音频特征和小模型。这样可以清楚分析每类信号对任务的贡献。

### Q4: 结果为什么提升不大？

回答：

> 数据只有 14 首歌，且 call 行为包含文化和人工偏好，不完全由音频决定。当前结果证明音频特征有增益；后续会改进 vocal density、重复结构建模和 role-aware boundary decoding。

### Q5: 有没有用之前 8 首歌训练的小模型？会不会数据泄露？

回答：

> 这轮信号处理实验没有使用之前的 `models/tiny_pipeline/*.pt`。量化模型是 leave-one-song-out Random Forest：每次留一首歌测试，只用其他歌曲训练。人工 annotation 只用于生成 ground truth 和评估，不作为测试歌的输入预测特征。

### Q6: allin1 没有 call_role，为什么还能做 role 对比？

回答：

> 是的，allin1 没有 call_role。因此我们没有把 allin1 当作直接的 call_role 预测器。role 对比里的 structure baseline 是两种派生 baseline：一种是人工写的 structure-to-role heuristic，另一种是只用 allin1 结构相关特征训练的 LOSO structure RF。真正和 allin1 原生能力可比的是音乐段落边界表，其中我们还做了 fine/coarse 两套人工目标来保证公平。

### Q7: 这次算法迭代具体改进了什么？

回答：

> 我们先保存了旧代码和旧结果，然后只在现有小节级音频特征上做模型搜索。上下文特征和 Viterbi 平滑没有稳定提升；Logistic Regression 改善了类别均衡；最后 RF、Logistic Regression 和 Gradient Boosting 的等权 soft voting 表现最好。所有模型仍然是 leave-one-song-out：测试歌曲不参与训练。
