# 外部歌曲结构划分实验汇报备注

## 实验目的

这一组实验只讨论“歌曲结构边界划分”，不讨论 call role 分类。所有方法都在同一个小节网格上输出边界，再分别和两种人工标注比较：

- `manual_fine`：人工细粒度段落边界。
- `manual_coarse`：把人工细粒度标签折叠到 allin1 类似的粗标签后再比较。

## 方法区分

| 方法组 | 方法 | 核心思想 | 是否用 allin1 | 是否用人工训练歌调参 |
|---|---|---|---:|---:|
| previous baseline | `allin1_structure` | allin1 直接给出的结构段落 | 是 | 否 |
| previous baseline | `fused_novelty_topk` | 旧的原始 novelty top-k | 否 | 否 |
| CBM-style | `cbm_dp_*` | 小节级自相似矩阵 + 段内一致性评分 + 动态规划找全局最优分段 | 否 | fixed 版否，LOSO 版是 |
| Classic/MSAF-like | `foote_checkerboard_*` | 在自相似矩阵对角线附近用 checkerboard kernel 找局部结构突变 | 否 | 是 |
| Classic/MSAF-like | `agglomerative_*` | 对小节音频特征做时序约束聚类，聚类标签变化处作为边界 | 否 | 是 |
| Modern embedding interface | MERT/BEATs/OpenBEATs interface | 用预训练音频模型替代 MFCC/chroma 等手工特征，再接同样的边界解码器 | 否 | 本次未下载大模型 |

## 最终对比表

| Target | Method | Precision | Recall | F1 | 说明 |
|---|---|---:|---:|---:|---|
| manual fine | allin1 structure | 0.889 | 0.393 | 0.545 | 强 baseline，精度高但召回低 |
| manual fine | fused novelty top-k | 0.320 | 0.320 | 0.320 | 旧纯音频 novelty baseline |
| manual fine | CBM-DP fixed full | 0.379 | 0.537 | 0.444 | 不用人工调参，纯自相似动态规划 |
| manual fine | Foote checkerboard LOSO tuned | 0.468 | 0.508 | 0.487 | 本轮最强非 allin1 开源经典算法路线 |
| manual fine | allin1 + learned signal RF | 0.774 | 0.504 | 0.610 | 之前实验的混合路线，不属于“除了 allin1” |
| manual coarse | allin1 structure | 0.843 | 0.572 | 0.682 | 粗结构上仍然最强 |
| manual coarse | fused novelty top-k | 0.289 | 0.289 | 0.289 | 旧纯音频 novelty baseline |
| manual coarse | CBM-DP fixed full | 0.283 | 0.616 | 0.388 | 召回高，但误检多 |
| manual coarse | Foote checkerboard LOSO tuned | 0.494 | 0.497 | 0.495 | 最强非 allin1 方法，但仍低于 allin1 |
| manual coarse | allin1 + learned signal RF/GB | 0.829 | 0.579 | 0.681 | 之前实验的混合路线，接近但没有超过 allin1 |

## 可以直接在汇报里讲的话

本项目原先的 `fused novelty top-k` 只是把音频变化最大的点取出来，效果比较弱。新的实验把歌曲结构划分明确建模为“小节级自相似矩阵上的边界检测”：CBM-style 方法追求段内一致性，Foote checkerboard 方法追求边界前后局部对比。结果显示，Foote checkerboard LOSO 调参在不使用 allin1 的前提下，把 fine target 的 F1 从 0.320 提高到 0.487，把 coarse target 的 F1 从 0.289 提高到 0.495。

但粗结构上 allin1 仍然明显更强，说明当前手工音频特征还不足以完全替代专门的歌曲结构模型。比较合理的结论不是“我们打败了 allin1”，而是：信号处理方法可以显著增强纯音频边界检测，并且和 allin1 结合后可以提高细粒度人工边界的召回与总体 F1。

## 文件位置

- 总表：`experiments/external_structure_segmentation/final_summary_after_numpy_restore/EXTERNAL_STRUCTURE_SEGMENTATION_LOG.md`
- 主实现：`scripts/external_structure_segmentation.py`
- 汇总脚本：`scripts/aggregate_external_structure_results.py`
- 每轮代码快照：各 round 目录下的 `code_snapshot/`
- 每个方法预测：各 round 目录下的 `methods/<group>/<method>/<target>/predictions.jsonl`
