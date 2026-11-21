# ThyroidTempoNet — 实时甲状腺超声视频结节检测

以 **YOLO26n**（本地 ultralytics 8.4.175，`C:\Users\bbuok\udisk-code\ultralytics`）为基线，
融合 5 篇时序建模医学视频检测论文的核心机制，改造出**严格因果、实时**的甲状腺结节视频检测模型，
并在相同训练量下与基线对比。

## 参考论文（`C:\Users\bbuok\thyroid-od`）
- **TSdetector** (MIA 2025)：TAda 时序自适应卷积 + ConvLSTM 队列 + 渐进队列整合（YOLOX 系，严格因果）
- **UltraDet** (MICCAI 2023)：滚动特征缓存 + 逆光流负时序上下文（因果，实时）
- **STNet** (MICCAI 2023)：时空可变形注意力（非因果，需 CUDA 算子）
- **MWNet** (MIA 2026)：小波卷积 + 长短期记忆库（detached 历史特征，甲状腺视频验证）
- **CL-MC** (MIDL 2026)：检测器在环的记忆重置状态机（免训练）

精读笔记见 [docs/paper_notes.md](docs/paper_notes.md)。

## 方法（ThyroidTempoNet）
在 YOLO26 头部三个输出 P3/P4/P5（层 16/19/22）与 `Detect` 之间插入**因果时序融合模块** `TemporalFusion`：
- **TAda-lite**：通道池化 → 因果时序路由 MLP → 逐帧通道校准 α（零初始化 α=1）→ 深度可分离 3×3 → 因果时序均值平滑
- **ConvGRU 队列**：3×3 门控、8 通道隐状态，沿时间窗因果推进（替代 TSdetector 较重的 ConvLSTM+5×5 Refine）
- **渐进队列整合**：TSdetector 式 1×1 降维残差累积
- **EMA 记忆指针**（P5）：EMA-SAM 式置信度门控 EMA，形成稳定病灶原型
- **零初始化残差输出**：初始等价于预训练 YOLO26，时序修正逐步学习

**实时关键**：推理用 UltraDet 式**滚动特征缓存**——历史帧不重复过 backbone（TSdetector 原实现 4× 重算）。
**训练关键**：MWNet 式 **detached 历史特征**（历史帧 no_grad 提特征，当前帧带梯度），6GB 显存可训。
历史采样遵循 TSdetector：从最多 50 帧前随机取 3 帧，最近一帧强制 t-1。

## 数据
`dataset.hdf5`（192 段视频、17412 帧、802×1054 灰度 + 结节掩码）→ YOLO 格式：
- PNG 命名 `视频名_帧号(4位).png`，标签为归一化 `cls xc yc w h`（单类 nodule）
- **按视频切分** train/val/test = 106/43/43 段（12185/2617/2610 帧），无帧泄漏
- 快速迭代用 **1/5 训练子集**（24 段、2578 帧，硬链接），val/test 保持完整

## 复现
```bash
# 1. 数据转换
python scripts/convert_hdf5_to_yolo.py
# 2. 1/5 训练子集
python scripts/make_subset.py
# 3. 基线
python scripts/train_baseline.py --data data/thyroid_yolo/thyroid_sub.yaml --name baseline_sub
# 4. 时序模型（相同训练量）
python scripts/train_temporal.py --data data/thyroid_yolo/thyroid_sub.yaml --name temporal_v1
# 5. 流式（因果、按视频顺序）评测 + FPS
python scripts/stream_eval.py --weights runs/baseline_sub/weights/best.pt --split test
python scripts/stream_eval.py --weights runs/temporal_v1/weights/best.pt --temporal --split test
```

环境：conda `vocod-yolo`（torch 2.7.0+cu128），GPU RTX 3060 Laptop 6GB。

## 目录
- `thyroid_temporal/`：时序模块、模型、clip 数据集、增强、trainer
- `scripts/`：转换/子集/训练/评测入口
- `data/thyroid_yolo/`：YOLO 数据集 + 子集
- `runs/`：训练输出与指标
- `docs/`：论文笔记、日志、报告