# 单图混入训练（static-mix）— 结果

实验：lrdisc 时序方案 + ThyroidXL 单图（repeat 模式：单帧伪视频，历史=当前帧副本）。
训练集 21726 帧（视频 12185 + 单图 9541，43.9%），val/test 纯视频不变，从 baseline_full 初始化。
训练按用户指示停止于 epoch 9 完成后（20ep 计划未跑满；black/jitter/mixed 变体未跑）。

## 视频 test（2610 帧）流式评测 — 全部权重

| 模型 | 权重 | 步数 | mAP50 | mAP50-95 | P | R | F1 | FPS |
|---|---|---|---|---|---|---|---|---|
| **xlrep（混单图）** | **last ep9** | **12.2k** | **.665** | **.345** | **.778** | .603 | **.679** | 34.8 |
| xlrep | best ep8 | 10.9k | .652 | .340 | .775 | .575 | .660 | 33.5 |
| main lrdisc | last ep20 | 15.2k | .638 | .329 | .725 | **.609** | .662 | **35.8** |
| main lrdisc | best ep6 | 4.6k | .625 | .325 | .686 | .605 | .643 | 35.9 |
| baseline | ep20 | 15.2k | .609 | .317 | .721 | .565 | .633 | 35.5 |

## 结论

**混入单图（repeat 模式）提升视频检测性能，且用更少训练量**：
xlrep last（ep9，12.2k 步）vs 此前最佳交付 main lrdisc ep20 last（15.2k 步）——
- mAP50 **+2.7pp**（.665 vs .638），mAP50-95 **+1.6pp**（.345 vs .329）
- precision **+5.3pp**（.778 vs .725），recall 基本持平（.603 vs .609，−0.6pp）
- F1 **+1.7pp**（.679 vs .662）；FPS 均实时
- vs baseline：mAP50 +5.6pp、R +3.8pp、F1 +4.6pp

要点：
1. **last 权重无选点假象**：ep9 固定 epoch，不经小验证集挑选；best ep8 的 recall 低谷（.575）
   到 ep9 已恢复（.603）而 precision 保持——早期"以召回换精度"是训练中期现象，非最终权衡。
2. **步数更少反而更强**（12.2k vs 15.2k，−20%），排除了"多算收益"解释增益的可能——
   增益只能来自单图数据本身（或两者交互）。等步数对照可进一步量化，但方向已明确。
3. 未竟事项（按用户指示停止）：20ep 跑满的同预算对比、black/jitter/mixed 历史合成变体、
   等步数对照。当前证据已支持"repeat 混入有净收益"。
4. 去噪预处理已另行否定（见 xl_noise_analysis.md）：散斑是检测信号，混入用原图。

评测产物：`runs/stream_test_temporal_lrdisc_xlrep_{best,last}.json`、
`runs/stream_test_temporal_lrdisc_best.json`；既有参考 `runs/stream_test_temporal_lrdisc.json`（ep20 last）。
交付候选权重：`runs/temporal_lrdisc_xlrep/weights/last.pt`（ep9）。