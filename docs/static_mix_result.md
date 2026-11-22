# 单图混入训练（static-mix）— 结果

实验：lrdisc 时序方案 + ThyroidXL 单图（repeat 模式：单帧伪视频，历史=当前帧副本）。
训练集 21726 帧（视频 12185 + 单图 9541，43.9%），val/test 纯视频不变，从 baseline_full 初始化。
训练完成 **18/20 epoch** 后按用户指示停止（中途曾挂起/检查点续训一次，epoch 1–9 与 10–18 两段）。
black/jitter/mixed 变体未跑。

> **勘误（2026-11-23）**：本文档早期版本误将 last.pt 记为 ep9/12.2k 步。实际续训在被停止前
> 已完成到 ep18（last.pt epoch=17 0-indexed，results.csv 18 行，mtime 2026-11-22 23:28）。
> last 评测结果 .665 属于 **ep18（24.4k 步）**。"步数更少仍更强"仅对 **best ep8（10.9k 步）** 成立。
> 相关提交 d85fb85/12a41fe 的说明中的 epoch 数以此处为准。

## 视频 test（2610 帧）流式评测 — 全部权重

| 模型 | 权重 | 步数 | mAP50 | mAP50-95 | P | R | F1 | FPS |
|---|---|---|---|---|---|---|---|---|
| **xlrep（混单图）** | **last ep18** | **24.4k** | **.665** | **.345** | **.778** | .603 | **.679** | 34.8 |
| xlrep | best ep8 | 10.9k | .652 | .340 | .775 | .575 | .660 | 33.5 |
| main lrdisc | last ep20 | 15.2k | .638 | .329 | .725 | **.609** | .662 | **35.8** |
| main lrdisc | best ep6 | 4.6k | .625 | .325 | .686 | .605 | .643 | 29.3* |
| baseline | ep20† | 15.2k | .609 | .317 | .721 | .565 | .633 | 35.5 |

\* best ep6 的 FPS 测于持久性门向量化之前；向量化后同架构 35.8（数学等价，精度不变）。
† baseline 实为 30ep 训练（baseline_full），此处列其权威流式结果作参照。

## 结论

**混入单图（repeat 模式）提升视频检测性能**：
xlrep last（ep18）vs 此前最佳交付 main lrdisc ep20 last——
- mAP50 **+2.7pp**（.665 vs .638），mAP50-95 **+1.6pp**（.345 vs .329）
- precision **+5.3pp**（.778 vs .725），recall 基本持平（.603 vs .609，−0.6pp）
- F1 **+1.7pp**（.679 vs .662）；FPS 均实时
- vs baseline：mAP50 +5.6pp、R +3.8pp、F1 +4.6pp

归因与证据强度：
1. **数据收益的直接证据来自 best ep8**：仅 10.9k 步（lrdisc ep20 的 72%）test mAP50 已达 .652 > .638，
   排除了"多算收益"解释。但 best 经 539 帧小验证集选点，有选点假象风险（aux 实验教训）。
2. **last ep18 无选点假象**（固定 epoch），确认最终质量；但其步数为 lrdisc ep20 的 1.6 倍，
   单独不能排除算力贡献。两条证据合起来支持"单图混入有净收益"。
3. **干净的等步数 last-vs-last 对比缺失**：无 ep11–12（≈15.2k 步）中间检查点。
   如需补：重训时设 `save_period` 存中间检查点，或续训至 20ep 做同 epoch 对比。
4. best ep8 的 recall 低谷（.575）到 ep18 恢复（.603）而 precision 保持——早期"以召回换精度"
   是训练中期现象，非最终权衡。
5. **val_sub 选点在混合训练下不可靠**：ep8 后 val fitness 不再刷新 best，但 test 持续改善到 ep18
   （ep18 val_sub mAP50 .9007 却未成新 best）——小验证集与 test 排序不一致的又一实例。
6. 去噪预处理已另行否定（见 xl_noise_analysis.md）：散斑是检测信号，混入用原图。

未竟事项（按用户指示停止）：20ep 跑满、black/jitter/mixed 历史合成变体、等步数对照。
评测产物：`runs/stream_test_temporal_lrdisc_xlrep_{best,last}.json`、
`runs/stream_test_temporal_lrdisc_best.json`；既有参考 `runs/stream_test_temporal_lrdisc_last.json`（ep20）。
交付候选权重：`runs/temporal_lrdisc_xlrep/weights/last.pt`（ep18）。
续训命令（如需跑满 20ep）：`python scripts/train_temporal.py --resume runs/temporal_lrdisc_xlrep/weights/last.pt`。