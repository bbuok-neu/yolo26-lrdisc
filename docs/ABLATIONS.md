# 消融与诊断计划

主对比（相同训练量：1/5 子集 2578 帧、50 epochs、batch16、imgsz640、seed0）：
- **baseline_sub**：YOLO26n
- **temporal_v1**：YOLO26n + 因果时序颈部（T=4, hid=8, P3=TAda / P4=TAda+GRU+queue / P5=+EMA）

## 消融（按优先级，时间允许时跑）
1. **T 窗口**：T=2 / T=4（默认）/ T=6 —— 时序窗口长度收益
2. **模块组件**：`--levels tada_only`（仅 TAda）/ `--levels p5only`（仅 P5 融合）/ `--levels noema`（去 EMA）/ full
3. **历史采样**：history_window=1（仅相邻帧）vs 50（随机历史，默认）
4. **隐状态宽度**：hid=4 / 8 / 16

## 诊断（证明"真时序"，TemporalLens 式）
`scripts/diagnose.py`：normal / shuffle（乱序历史）/ freeze（冻结缓存）/ zero（清零历史）
- 真时序模型：扰动后 mAP 下降；单帧模型：不受影响
- 预期 temporal_v1 在 shuffle/freeze 下 mAP50 明显下降

## 评测协议
- 训练期：val_sub（539 帧）选 best.pt
- 权威：`stream_eval.py` 完整 test（2610 帧，按视频因果顺序，滚动缓存），mAP50/50-95/P/R/F1/FP@R90/FPS
- 基线与时序模型用**同一**流式 harness， apples-to-apples
- FPS 在 GPU 空闲时测（batch1, 640, FP16）

## 定性
`scripts/visualize.py`：同一测试视频 GT | baseline | ThyroidTempoNet 三联图 + 置信度曲线