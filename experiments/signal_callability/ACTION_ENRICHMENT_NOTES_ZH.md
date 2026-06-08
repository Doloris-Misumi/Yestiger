# 动作补全模块说明

## 目标

此前的最佳 callbook 版本主要输出每个时间段的 `call_role`，例如 `keepspace`、`rhythmcall`、`mix`、`underground_gei`。这一步在不改变原有分段和角色预测结果的前提下，为每个 call 段补充更具体的 `recommended_actions`，使输出从“这一段适合什么类型的 call”进一步变成“这一段可以具体做什么动作”。

## 当前实现

代码位置：

- `scripts/enrich_callbook_actions.py`

输入：

- 现有最佳方法输出的 `*.merged.loso_audio_vote_rf1_logreg1_gb1.call_spans.json`
- 其他歌曲人工标注中的 `recommended_actions`
- `knowledge/call_mix_library.json` 中的动作知识库

输出：

- `*.action_call_spans.json`
- `*.action_callbook.md`

## 方法逻辑

1. 保留原有时间边界和 `call_role` 预测结果。
2. 对每个待生成的歌曲，读取除本歌曲之外的人工标注动作样本，形成留一首歌验证式的动作原型库。
3. 按照 `call_role`、歌曲结构上下文、段落长度、bar 数、上下文标签等信息，为当前段匹配最接近的动作样本。
4. 如果标注样本不足，则使用 `knowledge/call_mix_library.json` 中的动作知识库作为后备。
5. 对中高风险动作添加风险标记，例如 `[medium]`、`[high]`，方便后续人工审阅。

## 策略区分

当前脚本保留两种动作选择策略：

- `frequency`：旧版策略。把相似训练样本的分数累加，因此训练集中出现频次高的动作会明显占优。这个策略容易导致 mix 段过度选择 `ietora`。
- `balanced`：新版策略。以最相似样本的匹配质量为主，频次只作为很小的辅助项；同时对高风险 mix 动作降权，并在同一 mix 段内尽量选择不同动作家族。
- `barfit`：长度约束策略。候选动作评分沿用 `balanced` 的思想，但动作排布时进一步读取知识库中的 `preferred_bars`、`allowed_bars`、`min_bars` 和 `max_bars`，用动态规划选择一组动作来覆盖当前 call 段的小节数。每个动作会落到真实 bar 边界上，并在输出中显示 `(N bars)`。如果不能合理填满，剩余部分会标为 `Keep Space / Unassigned Gap`，而不是把某个动作硬拉长。

输出文件名用不同后缀区分：

- 旧频次版：`*.action_callbook.md`
- 均衡候选版：`*.balanced_action_callbook.md`
- 长度约束版：`*.barfit_action_callbook.md`

## 当前评分机制

动作候选总分由两部分组成：

1. 标注原型匹配分：比较当前段与其他歌曲标注样本在 `call_role`、音乐段落标签、粗粒度结构、allin1 结构、位置、段落长度等维度上的相似度。`balanced`/`barfit` 不再简单累计出现频次，而是主要看最相似样本的匹配质量。
2. 知识库适配分：检查动作类别是否匹配当前 `call_role`，上下文标签是否匹配动作的 `best_context`，当前段小节数是否满足动作的长度要求，并对中高风险动作扣分。

在 `barfit` 中，排序后的候选动作还会进入长度排布阶段。该阶段优先选择能满足动作规定小节长度的组合，并输出 `duration_fit`。如果动作长度命中 `allowed_bars`，`duration_fit=1.0`；如果只是允许延展但偏离 `preferred_bars`，会在表格中显示例如 `fit=0.67`，提示需要人工审阅。

如果某个预测出来的 call 段太短，没有任何候选动作能满足知识库中的长度约束，`barfit` 不会再把最高分动作硬塞进去，而是输出 `Keep Space / Too Short for Action`。例如 1 bar 的 `underground_gei` 段不会被填入地下艺动作，因为地下艺动作通常至少需要 4/8/16 bars。当前全量 `barfit_action` 输出已经通过长度一致性检查：实际动作没有违反 `allowed_bars`、`min_bars`、`max_bars` 或 `bar_multiple` 的情况。

`barfit` 还加入了 song-level 的长 MIX 去重规则：如果一个 MIX 动作在当前歌曲中已经以 3 bars 或更长的形式使用过，后续段落会尽量改用其他长 MIX 或留空，不再重复同一个长 MIX。1-2 bars 的短 MIX、`ietora` 类短触发和 activation 类动作仍允许重复。当前全量 `barfit_action` 输出中，不可重复长 MIX 的重复次数为 0。

## 数据泄露控制

当前模块没有使用之前的小模型 `ActionRanker`。

对某一首歌生成动作时，该歌曲自己标注文件里的 `recommended_actions` 不会被用于动作选择。例如生成 `poppindream` 时，训练动作样本会排除 `poppindream.annotation.json`，只使用其他歌曲的动作标注和通用知识库。

## 已生成结果

已为 14 首歌批量生成动作版 callbook。以 `poppindream` 为例：

- `experiments/signal_callability/poppindream/poppindream.merged.loso_audio_vote_rf1_logreg1_gb1.action_call_spans.json`
- `experiments/signal_callability/poppindream/poppindream.merged.loso_audio_vote_rf1_logreg1_gb1.action_callbook.md`
- `experiments/signal_callability/poppindream/poppindream.merged.loso_audio_vote_rf1_logreg1_gb1.balanced_action_call_spans.json`
- `experiments/signal_callability/poppindream/poppindream.merged.loso_audio_vote_rf1_logreg1_gb1.balanced_action_callbook.md`
- `experiments/signal_callability/poppindream/poppindream.merged.loso_audio_vote_rf1_logreg1_gb1.barfit_action_call_spans.json`
- `experiments/signal_callability/poppindream/poppindream.merged.loso_audio_vote_rf1_logreg1_gb1.barfit_action_callbook.md`

## 注意事项

这个模块目前是“动作建议器”，不是动作真值预测器。它已经能把 role-level callbook 扩展为 action-level callbook，但类似 `ietora`、地下艺等高风险动作仍需要人工确认是否适合具体场景。报告中可以把它作为“从结构分段到可执行 callbook 的后处理模块”，展示系统完整性。
