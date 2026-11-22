# 甲状腺超声视频实时检测：YOLO26 基线 vs 时序改进模型 对比报告

## 1. 任务与数据
- `dataset.hdf5`：192 段甲状腺超声视频、17412 帧、802×1054 灰度、单结节掩码。
- YOLO 格式：PNG `视频名_帧号(4位).png`，单类 nodule，掩码连通域→bbox。**按视频切分** train106/val43/test43 段（12185/2617/2610 帧），无帧泄漏。
- 血管**无标注**（是假阳性来源），故血管判别只能靠时序动态隐式学习。

## 2. 模型族（均在 YOLO26n 颈部 P3/P4/P5 与 Detect 之间插因果时序模块，推理滚动缓存保实时）
- **baseline**：原版 YOLO26n，完整数据 30ep。
- **lite**：因果时序均值去噪 + 零初始化小残差（参数 +0.2M）。从基线权重微调 15ep。
- **lrdisc（长程判别）**：稀疏跨步历史 [t-3,t-10,t-25] + 运动容忍长程持久性门（局部相关）+ 多尺度帧差运动分支（参数 +0.22M）。从基线权重微调 20ep。针对"血管 vs 结节时序动态差异"。
- **full（复杂）**：TAda-lite+ConvGRU+渐进队列+P5 EMA（+1M）。30ep。**过拟合，低于基线**（教训）。

## 3. 权威结果（完整 test 2610 帧，流式因果，同一 harness；百分数）
| 模型 | mAP50 | mAP50-95 | Precision | Recall | F1 | FPS |
|---|---|---|---|---|---|---|
| YOLO26n 基线 | 60.88 | 31.66 | **72.11** | 56.45 | 63.33 | **35.5** |
| lite（实时推荐） | 61.71 | **32.68** | 68.32 | 59.55 | 63.63 | 31.9 |
| **lrdisc（精度推荐）** | **62.48** | 32.48 | 68.58 | **60.51** | **64.30** | 29.3 |
| full（复杂，过拟合） | 58.26 | 29.10 | 64.63 | 50.52 | 56.71 | 26.5 |

- **lrdisc 最佳 mAP50(+1.6pp)/Recall(+4.1pp)/F1(+1.0pp)**；lite 最佳 mAP50-95(+1.0pp) 且实时(32fps)。
- **Precision（血管假阳性）所有时序模型均未超基线**——血管判别假设在本数据上未被当前机制证实（见 §5）。
- lrdisc 推理 **29.3fps**（向量化 unfold 局部相关后；原 Python 位移循环 launch-bound 仅 15fps），lite 32fps，均实时级。

### 时序诊断（TemporalLens）
- lite：zero −38.6pp（强依赖历史去噪）、freeze −2.8pp、shuffle≈0。
- lrdisc：freeze −4.15pp（用长程更新上下文）、zero≈0（帧差分支兜底）、shuffle≈0（去噪/持久性而非运动顺序）。
- 两者均**真正利用时序**，非仅当前帧+更多参数。

## 4. 调研支撑（顶刊顶会，docs/research_longrange_temporal.md）
Auto-DFLLs(npj Digit Med 2026) 因果滑窗+光流+时序注意力压瞬态FP；STNMamba(TMM 2026) 帧差运动判别；TransVOD(TPAMI 2022) 时序query融合降漏检误检；MWNet(MedIA 2025) 长短期记忆；稀疏跨步关键帧库(Δt≈5–15)；一致性校验头压血管横截面。lrdisc 即这些机制的轻量因果落地。

## 5. 诚实结论与局限
1. 时序建模在本数据带来**modest 但一致的 mAP/Recall 增益**（lrdisc mAP50 +1.6pp、Recall +4.1pp），并证实真时序 usage。
2. **血管 precision 未提升**：可能因 (a) 血管横截面在探头停留期间也持久，持久性门难分离；(b) 无对齐的局部相关对探头大幅移动鲁棒性不足；(c) 无血管标注，隐式学习信号弱。文献中有效压血管者（Auto-DFLLs）用了**光流对齐**，本文免光流轻量版判别力有限。
3. **复杂时序过拟合**（full 低于基线）→ 容量须匹配数据量；轻量(lite/lrdisc)更稳健。
4. **大幅超越需更大/更难数据**（服务器大数据重训）或**显式血管标注/光流对齐/可变形对齐**。本数据结节大、基线强(mAP50 0.61)，时序天花板有限。

## 6. 推荐与复现
- 实时场景：`--levels lite`（32fps，mAP50-95 最佳）。
- 精度优先：`--levels lrdisc --strides 3,10,25`（mAP50/Recall/F1 最佳，29.3fps）。

### PR 工作点：precision≈基线时 recall 超越基线（scripts/pr_analysis.py）
P/R 随置信度阈值变化；报表 max-F1 点不可直接比。锁定 precision≥基线水平(72%)时：
| 模型 | R@P≥70% | R@P≥72% | R@P≥75% |
|---|---|---|---|
| 基线 | 57.68 | 56.45 | 53.66 |
| lite | 58.25 | 56.19 | 54.19 |
| **lrdisc** | **59.29(+1.61)** | **57.18(+0.73)** | **55.30(+1.65)** |
→ **lrdisc 可在 precision 不输基线(≥72%)下 recall 超越基线**（调高置信度阈值即可，无需重训），源于其更高 AP/PR 曲线。
原持久性门用 Python 循环遍历 9 位移×3 历史×3 层（≈324 kernel launch/帧）→ **launch-bound** 15fps。
改为 `F.unfold` 一次性向量化全部位移（几个 kernel）→ 29.3fps，且数学与原训练完全等价（pool=1）→ **精度不变、无需重训**。
可选 pool=2（半分辨率相关）探针 31.3fps，但需重训保持一致；pool=1 向量化已达标 ≥25fps。
- 服务器大数据：用完整数据重训 lrdisc/lite，预期增益放大；可加光流/可变形对齐与血管标注进一步压 FP。
- 命令见 README/AGENTS。图表 [comparison.png](comparison.png)、定性 [qualitative/](qualitative/)。

## 7. 后续更新（本表数字的时效性说明）

- §3 表为**早期选点**结果。lrdisc 的**最终权威交付为 ep20 last**（同预算流式评测）：
  mAP50 **.6377** / mAP50-95 .3287 / P **.7247** / R **.6090** / F1 **.6618** / 35.8fps——
  即 precision 追平基线（.721）的同时 recall +4.5pp、mAP50 +2.9pp，§5 的"precision 未超基线"
  结论对 ep20 last 已不成立（best ep6 选点偏早所致）。
- 因果辅助监督（aux-supervision 分支，w=0.5）为负收益（mAP50 .605 < .638），未采纳。
- **static-mix 分支**：混入 ThyroidXL 单图后 xlrep ep18 达 mAP50 **.6654** / F1 .6789（当前全项目最佳），
  见 docs/static_mix_result.md 与 docs/EXPERIMENTS.md（权威台账）。
- 接手请先读根目录 AGENTS.md。