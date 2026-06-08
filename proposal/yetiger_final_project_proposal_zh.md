# 基于音频的偶像/Anisong 现场应援时机检测

## 摘要

在许多流行音乐现场中，观众并不只是被动聆听，而是会在特定时间点进行拍手、齐声短喊、节奏回应，或在重要人声段落保持安静。本文把这类与音乐同步的观众行为统称为 **call**。对于偶像、动画歌曲和部分 J-pop 演出，提前规划这些 call 的时间点可以帮助观众更整齐地参与现场；但如果 call 插入在错误位置，例如人声密集处或节奏不稳定处，也会破坏音乐表达。因此，“什么时候适合 call”本质上可以被看作一个音频信号分析问题。

本项目计划研究一种基于音频的 call 时机检测方法。我们会从歌曲波形中提取节拍、能量、起音强度、音色变化、和声变化、重复段落以及人声密度等信号，并将它们聚合到小节级别，估计每个小节对不同观众行为的适合程度。我们将这种小节级适合度称为 **callability**。实验将使用我们已构建的小规模歌曲标注数据集，其中包含音频、自动分析出的 beat/downbeat，以及人工标注的音乐段落和 call 区间。预期结果是一个可复现的信号处理 pipeline，用于检测适合观众参与的音乐窗口，并为后续 callbook 生成提供可解释的音频依据。

**关键词：** 音乐信息检索，音频信号处理，节拍同步分析，音乐结构分析，人声密度，callbook 生成。

## 1. 研究动机

现场音乐中的观众互动需要和音乐结构严格对齐。以最简单的拍手为例，如果节拍不稳定，观众很难整齐参与；如果主唱正在演唱密集歌词，过强的呼喊会遮盖人声；如果歌曲即将进入副歌，段落边界附近则可能是更适合集体回应或情绪推进的位置。因此，合适的 call 时机通常由多个音频信号共同决定，而不是只由 BPM 或歌曲标题决定。

目前大多数自动音乐分析任务更关注“这段是 verse 还是 chorus”这一类结构标签。然而，对于现场观众参与来说，仅知道音乐段落还不够。我们还需要判断当前小节是否有稳定的节拍、是否有足够清晰的起音、能量是否正在上升、人声是否过密，以及当前段落是否和之前的副歌或间奏重复。这些问题都可以用高级信号处理课程中的核心技术来建模，例如短时傅里叶分析、频谱特征、起音检测、节拍同步特征聚合、新颖度曲线和自相似矩阵。

我们已经在前期工作中搭建了一个小型数据基础：若干首偶像 / Anisong 歌曲的音频文件、自动提取的 beat/downbeat 和粗粒度段落，以及人工标注的音乐段落和 call 区间。这个已有数据集为本项目提供了 ground truth，使我们可以把问题从“手工写 callbook”推进到“从音频信号中自动检测适合 call 的时间窗口”。因此，本期末项目的重点不是生成文本式应援词，而是构建并评估一个面向现场观众互动的音频信号处理方法。

## 2. 研究问题

给定一首歌的波形 $x(t)$ 和 beat/downbeat 网格，本项目的目标是为每个小节 $b$ 估计一个向量：

```text
c_b = [
  c_b^space,
  c_b^rhythm,
  c_b^mix,
  c_b^gei
]
```

这四个维度分别对应：保持安静、轻量节奏 call、MIX，以及高强度 underground-gei 风格动作。和标准音乐结构分析不同，这里的输出不只是 verse 或 chorus 这样的结构标签，而是一个面向行动的音频可供性：当前音乐信号是否提供了适合某种观众行为的窗口。

本项目的核心假设是：相比只使用音乐结构标签的 baseline，加入人声感知和节拍同步的信号特征，可以提升 call role 预测效果。第二个假设是：受 downbeat 约束的新颖度分析，比逐帧阈值判断更能产生可解释的 call window。

## 3. 方法设计

方法分为四个阶段。

### 3.1 节拍同步特征提取

首先通过短时频谱分析，把音频转成帧级特征。候选特征包括 RMS 能量、onset strength、spectral centroid、spectral bandwidth、spectral rolloff、MFCC、chroma 或基于 CQT 的和声特征，以及类似 tempogram 的节奏稳定性特征。

我们不会直接在帧级别做最终判断，而是根据 downbeat 网格，把这些特征聚合到 beat 或 bar 区间：

```text
f_b = Pool{ phi(x_t): t in [tau_b, tau_{b+1}) }
```

其中，`phi(.)` 表示帧级音频描述符，`tau_b` 表示小节边界。

### 3.2 人声感知的 Callability

现场应援中有一个关键规则：人声密集的地方不应该被激进的 call 覆盖。因此，我们会估计一个人声密度信号。可行的基础版本会使用项目里已经有的频谱和 onset 线索；如果时间允许，会使用 Demucs 分离出 vocal stem，再计算 vocal energy ratio：

```text
v_b = E_vocal(b) / (E_mix(b) + epsilon)
```

高人声密度会抑制 MIX callability，但仍然可能允许轻量 rhythm call 或 sing-along 支持。

### 3.3 新颖度与重复段落线索

音乐结构将通过音色、和声和能量的新颖度曲线进行建模。同时，我们会基于小节级特征计算 self-similarity matrix，用于识别重复副歌或器乐段落。

这会提供两类有用信号：

- boundary strength：表示可能适合开始 MIX 的段落边界；
- repetition similarity：支持在相似段落之间复用 call pattern。

### 3.4 Call Role 解码

小节级 callability 向量会被解码成 YesTiger 现有的 call role：

```text
y_b in { keepspace, rhythmcall, mix, underground_gei }
```

我们会比较两种解码方式：一种是基于规则的 decoder，另一种是在 YesTiger 标注上训练的小型监督模型。监督模型会保持轻量，以保证项目贡献仍然是可解释的信号处理，而不是不可解释的大模型。

## 4. 数据集与 Baseline

实验将使用本地 YesTiger 数据集。在 proposal 阶段，工作区中包含 15 个音频文件、15 个 allin1 结构文件，以及 14 个已人工标注的 annotation 文件。每个 annotation 都包含音乐段落和 call span。目前训练 manifest 只使用了 8 首歌；在项目过程中，我们会重建数据集，把新增标注歌曲也纳入训练和评估。

Baseline 包括：

- 只使用 allin1 粗粒度结构特征；
- 当前 YesTiger tiny pipeline：使用相对位置、小节长度、downbeat 标记和 allin1 段落标签；
- 当前 secondary audio analysis heuristic：基于 novelty、timbre、energy 和 onset 变化。

## 5. 评估计划

我们会同时评估信号层效果和下游行为效果。

### 5.1 边界评估

预测出的结构边界或 call-window 边界，会与人工 annotation 边界进行比较。指标包括 precision、recall 和 F1。容忍窗口可以设置为一个 downbeat 或一秒。

### 5.2 Call Role 评估

每个小节都会根据人工 call span 分配 ground-truth call role。我们会报告四个 call role 的 accuracy 和 macro-F1。消融实验会分别移除人声密度、新颖度、重复段落线索或 downbeat 约束，以衡量它们各自的贡献。

### 5.3 Case Study

最终报告会包含对 **Poppin'Dream!** 的详细案例分析。我们会可视化 energy、vocal density、novelty、callability 和被选中的 call window，然后把生成草稿与人工 annotation 进行比较。

## 6. 预期贡献

本项目的预期贡献包括：

1. 提出一个新的任务定义：面向 live callbook 生成的节拍同步 callability 估计；
2. 提出一个人声感知的信号处理 pipeline，用于检测适合观众 call 的窗口；
3. 建立一个使用人工 callbook annotation 作为 ground truth 的评估协议；
4. 在 YesTiger 中集成展示：把音频证据转化为可解释的 callbook 草稿。

## 7. 可行性与时间安排

如果控制好范围，这个项目大约一周内可完成。音频文件、annotation、验证脚本和 tiny learning pipeline 已经存在。主要实现工作是：添加小节级音频特征、计算 callability curve、重建数据集、重新训练小模型，并运行消融实验。

计划时间表如下：

- 第 1-2 天：实现小节级音频特征提取和人声密度估计；
- 第 3-4 天：把 callability 特征加入数据集，并训练/评估 call-role 模型；
- 第 5 天：运行消融实验并生成可视化；
- 第 6-7 天：准备 IEEE 风格论文和 10 分钟展示。

基于 Demucs 的 vocal stem 实验会作为加分项。如果它太耗时，最终项目会使用频谱代理特征来估计人声密度。

## 8. 作者贡献声明

在 proposal 阶段，组员和具体职责还未最终确定。在最终论文中，本节会明确说明每位作者的贡献，包括音频特征提取、模型实现、annotation 复核、实验、写作和展示准备。

## 致谢

本项目基于已有 YesTiger 工作区，包括人工复核过的 annotation、allin1 结构输出，以及一个本地小型训练 pipeline。

## 参考文献

[1] J. Foote, "Automatic audio segmentation using a measure of audio novelty," Proc. IEEE International Conference on Multimedia and Expo, 2000.

[2] G. Peeters, "Self-similarity-based and novelty-based loss for music structure analysis," Proc. International Society for Music Information Retrieval Conference, 2023.

[3] J. Serra, M. Muller, P. Grosche, and J. L. Arcos, "Unsupervised detection of music boundaries by time series structure features," Proc. AAAI Conference on Artificial Intelligence, 2012.

[4] Y. Li et al., "MERT: Acoustic music understanding model with large-scale self-supervised training," arXiv:2306.00107, 2023.

[5] S. Rouard, F. Massa, and A. Defossez, "Hybrid transformers for music source separation," Proc. IEEE International Conference on Acoustics, Speech and Signal Processing, 2023.
