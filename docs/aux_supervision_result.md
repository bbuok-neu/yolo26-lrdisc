# 因果辅助监督（Auto-DFLLs 借鉴）实验结果 — aux-supervision 分支

配置：lrdisc + 因果辅助监督 w=0.5（每个历史帧用"仅更早帧"上下文算检测损失，颈部特征 detached，
梯度只过 temporal+Detect；训练期开销、推理 0 成本）。完整数据 20ep、batch16、从 baseline_full 初始化。
对照：main 分支 lrdisc（无辅助监督），同预算。

## 完整 test（2610 帧）流式评测，同 epoch20 (last.pt)
| 配置 | mAP50 | mAP50-95 | P | R | F1 | FPS |
|---|---|---|---|---|---|---|
| baseline | .609 | .317 | .721 | .565 | .633 | 35.5 |
| main lrdisc (last) | **.638** | **.329** | **.725** | **.609** | **.662** | **35.8** |
| aux lrdisc+因果辅助 (last) | .605 | .321 | .794 | .548 | .649 | 35.5 |

best-val checkpoint 对比（val_sub 选点）：lrdisc best(ep19) .625/.686/.605/.643@29.3；aux best(ep3) .591/.752/.547/.634@31.6。

## 结论
- 因果辅助监督 w=0.5 **不是净收益**：precision↑(.794) 但 recall↓(−6.1pp)、mAP50↓(−3.2pp)、F1↓(−1.3pp)
  → 以召回换精度，偏离"precision≈基线且 recall↑"目标。
- val_sub(539帧) 上 aux 同 epoch 略优(+1~2pp) 是**小验证集选点假象**；full test 揭示真实权衡。
  aux val 早峰(ep3)过拟合，best checkpoint 偏早进一步拉低 test。
- **main lrdisc(epoch20/last) 为最佳交付**：precision≈基线、recall +4.4pp、mAP50 +2.9pp、F1 +2.9pp、FPS≈基线(实时不劣化)。
- 图像级时间一致性/负上下文：按要求暂弃（另有 onset 冷启动缺陷与温和门控净收益≈0 的实测，见 borrow_autodflls.md）。
- 若仍想借辅助监督：试更低 w(0.1–0.2) 或更长训练+按 full-test 选点；当前 w=0.5 证据不支持。

## 复现
git checkout aux-supervision
python scripts/train_temporal.py --data data/thyroid_yolo/thyroid_full.yaml --weights runs/baseline_full/weights/best.pt \
    --epochs 20 --batch 16 --name temporal_lrdisc_aux --levels lrdisc --strides 3,10,25 --aux 0.5
python scripts/stream_eval.py --weights runs/temporal_lrdisc_aux/weights/last.pt --temporal --split test