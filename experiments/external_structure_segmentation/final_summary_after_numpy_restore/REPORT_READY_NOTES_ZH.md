# 歌曲结构划分实验汇报备注

## 实验目的

这一组实验只讨论“歌曲结构边界划分”，不讨论 call role 分类。所有方法都在同一个小节网格上输出边界，再分别和两种人工标注比较：

- `manual_fine`：人工细粒度段落边界。
- `manual_coarse`：把人工细粒度标签折叠到 allin1 类似的粗标签后再比较。

## 方法口径

| 方法组 | 方法 | 核心思想 | 是否用 allin1 | 是否用人工训练歌调参 |
|---|---|---|---:|---:|
| previous baseline | `allin1_structure` | allin1 直接给出的结构段落 | 是 | 否 |
| previous baseline | `fused_novelty_topk` | 旧的原始 novelty top-k | 否 | 否 |
| CBM-style | `cbm_dp_*` | 小节级自相似矩阵 + 段内一致性评分 + 动态规划找全局最优分段 | 否 | fixed 版否，LOSO 版是 |
| MSAF-like classic | `foote_checkerboard_*` | checkerboard novelty：在自相似矩阵上找边界前后的局部突变 | 否 | 是 |
| MSAF-like classic | `agglomerative_*` | 对小节音频特征做时序约束聚类，聚类标签变化处作为边界 | 否 | 是 |
| official MSAF module | `official_msaf_foote_*` | 调用官方 MSAF Foote Segmenter 模块，但输入换成已有 bar-level 特征 | 否 | 是 |
| MERT embedding | `mert95m_contextual_foote_*` | 用 MERT-v1-95M 提取小节级上下文 embedding，再做自相似 novelty 解码 | 否 | 是 |
| hybrid | `allin1_plus_learned_rf_*` | 以 allin1 边界为基础，用学习到的信号边界补充 fine boundary | 是 | 是 |

## 最终对比表

| Target | Method | Precision | Recall | F1 | 说明 |
|---|---|---:|---:|---:|---|
| manual fine | fused novelty top-k | 0.320 | 0.320 | 0.320 | 旧纯音频 novelty baseline |
| manual fine | CBM-DP fixed full | 0.379 | 0.537 | 0.444 | 不用人工调参，纯自相似动态规划 |
| manual fine | official MSAF Foote | 0.432 | 0.078 | 0.132 | 官方 peak picking 太保守，只预测 44 个边界 |
| manual fine | MERT-v1-95M + Foote | 0.467 | 0.496 | 0.481 | 真正跑了 MERT embedding，略低于本地 Foote |
| manual fine | Foote checkerboard LOSO | 0.468 | 0.508 | 0.487 | 最强非 allin1 方法 |
| manual fine | allin1 structure | 0.889 | 0.393 | 0.545 | 强 baseline，精度高但召回低 |
| manual fine | allin1 + learned signal RF | 0.774 | 0.504 | 0.610 | 最强 overall，但使用 allin1 |
| manual coarse | fused novelty top-k | 0.289 | 0.289 | 0.289 | 旧纯音频 novelty baseline |
| manual coarse | CBM-DP fixed full | 0.283 | 0.616 | 0.388 | 召回高，但误检多 |
| manual coarse | official MSAF Foote | 0.511 | 0.151 | 0.233 | 精度尚可，但召回太低 |
| manual coarse | MERT-v1-95M + Foote | 0.453 | 0.491 | 0.471 | 比 CBM 强，但低于本地 Foote |
| manual coarse | Foote checkerboard LOSO | 0.494 | 0.497 | 0.495 | 最强非 allin1 方法 |
| manual coarse | allin1 structure | 0.843 | 0.572 | 0.682 | 粗结构上仍然最强 |

## MSAF 和 MERT 的真实情况

官方 MSAF 做了，但不是完整 `msaf.process` 全流程。完整流程在当前环境里有两个问题：第一，`cnmf` 依赖的 `cvxopt` DLL 加载失败；第二，`msaf.process` 在完整 MP3 上特征提取超时。因此最终采用的是“官方 MSAF Foote Segmenter 模块 + 已有小节级特征”的方式。这个结果可以诚实地叫 official MSAF module experiment，但不要说成完整官方 MSAF pipeline。

MERT 也做了。`m-a-p/MERT-v1-95M` 要求 `trust_remote_code=True`，这会执行 Hugging Face 仓库代码，所以没有采用。我们下载了权重数据，并把它作为标准 `Wav2Vec2Model` 加载，state dict 完全匹配，没有执行远程代码。随后用 20 秒分块提取全 14 首歌的小节级 contextual embedding，并缓存到 `experiments/mert_structure_segmentation/cache/`。

## 可以直接在汇报里讲的话

旧的 `fused novelty top-k` 只是把音频变化最大的点取出来，效果较弱。新的实验把结构划分建模为“小节级自相似矩阵上的边界检测”：CBM 追求段内一致性，Foote checkerboard 追求边界前后局部差异，MERT 则用预训练音频表示替代手工特征。

结果显示，不使用 allin1 时，最强方法是 Foote checkerboard LOSO：fine F1 从 0.320 提高到 0.487，coarse F1 从 0.289 提高到 0.495。MERT embedding 也接近这一水平，fine F1=0.481，说明预训练音频表示确实有帮助，但在当前小数据和解码器下没有超过精心调过的经典自相似方法。

最稳妥的结论是：信号处理方法可以显著增强纯音频边界检测；粗结构上 allin1 仍然最强；当 allin1 与学习到的信号边界结合时，fine boundary 的总体 F1 可以进一步提高到 0.610。

## 文件位置

- 综合总表：`experiments/structure_segmentation_all_methods/final/STRUCTURE_SEGMENTATION_ALL_METHODS.md`
- MERT 结果：`experiments/mert_structure_segmentation/round1_tuned/mert_structure_summary.md`
- 官方 MSAF 结果：`experiments/official_msaf_segmentation/round1_foote_tuned/official_msaf_summary.md`
- CBM/MSAF-like 结果：`experiments/external_structure_segmentation/final_summary_after_numpy_restore/EXTERNAL_STRUCTURE_SEGMENTATION_LOG.md`
- 主代码：`scripts/external_structure_segmentation.py`
- 官方 MSAF 代码：`scripts/official_msaf_segmentation.py`
- MERT 代码：`scripts/mert_embedding_segmentation.py`
- 综合汇总代码：`scripts/aggregate_structure_segmentation_all.py`

## 环境注意事项

本轮安装了 `msaf` 和 `transformers`，并确认 `numpy=1.23.5`、`madmom`、`torch`、`torchaudio`、`transformers` 都可以正常 import。MERT 实验使用 `torchaudio.load` 读取 MP3。

复测时发现当前环境下 `librosa.load` 读取 MP3 会卡住，但 `librosa` import 本身正常，且已有 bar-level 特征和本轮新增结果不受影响。如果后续要重新从原始音频提取特征，建议把音频读取部分改成 `torchaudio.load + torchaudio.functional.resample`。
