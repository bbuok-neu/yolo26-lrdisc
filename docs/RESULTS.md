> 历史自动汇总（lite 时代数字，原 GBK 编码已转 UTF-8）。权威结果见 [EXPERIMENTS.md](EXPERIMENTS.md)；scripts/make_report.py 会重写本文件。

# 结果汇总（自动生成）

## 训练期验证（val_sub, 539 帧）

| 模型 | best mAP50 | best mAP50-95 |
|---|---|---|
| YOLO26n 基线 | 91.94 | 62.07 |
| ThyroidTempoNet | 88.02 | 57.66 |

## 流式评测（完整 test, 2610 帧, 按视频因果顺序）

| 模型 | mAP50 | mAP50-95 | Precision | Recall | F1 | FP/img@R90 | FPS |
|---|---|---|---|---|---|---|---|
| YOLO26n 基线 | 60.88 | 31.66 | 72.11 | 56.45 | 63.33 | – | 35.53 |
| ThyroidTempoNet | 61.71 | 32.68 | 68.32 | 59.55 | 63.63 | – | 31.93 |

**流式 mAP50 提升：+0.84 pp；mAP50-95 提升：+1.02 pp。**
