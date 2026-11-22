# 单图混入训练（static-mix）— 中期结果

实验：lrdisc 时序方案 + ThyroidXL 单图（repeat 模式：单帧伪视频，历史=当前帧副本）。
训练集 21726 帧（视频 12185 + 单图 9541，43.9%），val/test 纯视频不变，从 baseline_full 初始化。
**训练在 epoch 9 完成后被用户暂停，后按指示停止于 epoch 10 中途**；以下为 best 权重中期对比。

## 视频 test（2610 帧）流式评测 — best vs best（用户指定对比）

| 模型 | 选点 | mAP50 | mAP50-95 | P | R | F1 | FPS |
|---|---|---|---|---|---|---|---|
| **xlrep best** | ep8（混合集） | **.652** | **.340** | **.775** | .575 | **.660** | 33.5 |
| main lrdisc best | ep6（纯视频） | .625 | .325 | .686 | **.605** | .643 | **35.9** |

参考（既有权威对比，last.pt 同 epoch20）：
| 参考 | mAP50 | mAP50-95 | P | R | F1 |
|---|---|---|---|---|---|
| baseline ep20 | .609 | .317 | .721 | .565 | .633 |
| main lrdisc ep20 (last) | .638 | .329 | .725 | .609 | .662 |

## 解读

- **xlrep best（仅 8 epoch、约 10.9k 步）mAP50 已超过 main lrdisc ep20 last（15.2k 步）**：
  .652 vs .638（+1.4pp），mAP50-95 .340 vs .329，precision .775 vs .725。
  更少步数拿到更高 mAP —— 单图混入的**积极信号**。
- 代价模式与 aux 监督类似：**precision↑（+8.9pp vs lrdisc best）换 recall↓（−3.1pp）**；
  F1 .660 vs .643（+1.7pp），vs lrdisc ep20 last 的 .662 基本持平。
- **不可作为最终结论**的三个原因：
  1. best 由 539 帧小验证集选点（aux 实验已证明有选点假象风险）；
  2. 两模型选点 epoch/步数不同，非等预算；
  3. xlrep 训练未完成（8/20），后期 precision/recall 权衡可能变化。
- FPS 33.5 vs 35.9：均实时；差异在测量噪声范围（此前同架构多次测得 35±1）。

## 待完成（用户决定）
- 续训 xlrep 至 20ep（`python scripts/train_temporal.py --resume runs/temporal_lrdisc_xlrep/weights/last.pt`，
  last.pt=ep9，约 1.5h）→ 与 lrdisc ep20 last 同预算对比；
- 实验 B（black 黑帧模式）20ep；
- 如有增益：等步数对照（~11ep）区分数据收益 vs 算力收益。

评测产物：`runs/stream_test_temporal_lrdisc_xlrep_best.json`、`runs/stream_test_temporal_lrdisc_best.json`。