# 医学视频检测/分割论文精读笔记（面向 YOLO26 时序改造）

> 依据 5 篇 PDF 原文 + 本地官方代码核对（TSdetector/UltraDet/STNet/MWNet 仓库均在 `C:\Users\bbuok\thyroid-od\`）。
> 目标平台：ultralytics YOLO26（NMS-free one-to-one 头，reg_max=1 无 DFL，Detect 接收 P3/P4/P5），RTX 3060 laptop 6GB，实时。
> 注意：5 篇中没有一篇是"甲状腺视频检测"——TSdetector=结肠镜息肉，UltraDet/STNet=乳腺超声视频（CVA-BUS），MWNet=甲状腺超声视频分割（域最接近），CL-MC=喉镜声门定位。结论迁移时需留意域差。

## 0. 对先前 survey 的勘误表

| Survey 说法 | 核对结果 |
|---|---|
| TSdetector = YOLOX + TAdaConv + R_CLSTM，窗口 [t-3,t-2,t-1,t] | 论文名为 GT-Conv / HQIM / PAC；TAdaConv、R_CLSTM 是代码实现名。窗口长 4 属实，但**训练时 3 个过去帧从当前帧前 ≤50 帧内随机抽**（最后一帧强制为 t-1），并非固定相邻。另有第三个模块 PAC（后处理置信度聚类校正），survey 漏掉，消融贡献 +1.8 AP |
| TSdetector 严格因果 | 对当前帧成立（当前帧是窗口末位）；R_CLSTM 状态**每窗口清零**，窗口间无记忆；官方代码每窗口重提 4 帧特征（无缓存），28.29 FPS 是 3090 上 YOLOX-X 的数字 |
| UltraDet = NTCA 光流 warp 过去 p5 特征 | 特征层是 **res4（C4，stride 16，ResNet34 半通道）**，不是 p5；flow 由**冻结 FlowNetS（加载 DFF 权重）**预测，warp 整张 F_τ 后乘 DFF scale map，再用当前帧 box 做 RoIAlign（论文表述为 IOF Align 变形网格，二者等价）。且论文明确消融：**RoI 级聚合 > 特征级（FGFA 式）**。基础检测器还含 RDN 式 Temporal Relation head，不只 NTCA |
| STNet 非因果、需 CUDA op | 属实（输入含 k+1 帧；MSDeformAttn CUDA op）。补充关键提速设计：encoder feature shuffle，一次前向预测 3 帧 |
| MWNet ConvLSTM + 记忆库 | 代码中 TemporalFusion 的门全是 **1×1 conv**（非标准 3×3 ConvLSTM），cell 更新额外过一个 1×1 conv；记忆库仅作用于**最深层**，t_long=5 / t_short=2，余弦相似度压缩；编码器小波是**固定 db1/Haar 基**（ECCV24 WTConv），自适应提升小波只在解码器 HFF |
| CL-MC 检测器闭环校正 SAM2，免训练 | 属实。补充：硬重置仅在 State 3（检测器高置信且与 tracker IoU 不一致）触发，其余状态软更新；置信度用滑窗 min-max 归一化对齐 |

---

## 1. TSdetector（MIA 2025）：GT-Conv + HQIM + PAC，YOLOX-X 基座

### 1.1 时序机制
三个模块插入位置：GT-Conv（代码 `TAda`）作用于 backbone 输出 dark3/dark4/dark5（进 PAFPN 之前，三个尺度都做）；HQIM（`R_CLSTM_5` + 渐进队列）作用于 PAFPN 三个输出 P3_out/P4_out/P5_out（进 head 之前）；PAC 是 NMS 替代后处理。窗口 T=4（代码硬编码 `x.size(0)//4`）。

**GT-Conv 数学**（论文 Eq.1-5）：`W̄_t = W_t·α_t^w, b̄_t = b_t·α_t^b`；`α = 1 + F(Ŝ_t)`，F 为时间核 [3,1,1] 的 Conv3d；`Ŝ_t = BN(ReLU(F(GAP(X̂))) + F(SAP(X̂)))`。初始 α=1（零初始化），保证加载预训练权重起步。

**TAda block 精确结构**（`nets/tadaconv.py`，逐行核对）：
- `b = TAdaConv2d(kernel=[1,3,3], pad=[0,1,1], bias=False, cal_dim="cin")`：基权重逐帧乘 α（按输入通道校准），用 groups=B*T 的分组 conv2d 一次算完 4 帧。
- `b_rf = RouteFuncMLP(c_in, ratio=4, kernels=[3,3])`：`avgpool=AdaptiveAvgPool3d((None,1,1))`（逐帧 GAP，保留 T）+ `globalpool=AdaptiveAvgPool3d(1)`（全局，经 1×1×1 Conv3d `g` 广播相加）→ `a`=Conv3d [3,1,1] C→C/4 → BN3d → ReLU → `b`=Conv3d [3,1,1] C/4→C（**权重零初始化**）→ +1 得 α。
- 主路：`x = b(x, b_rf(x))` → `b_bn(x) + b_avgpool_bn(b_avgpool(x))`，其中 `b_avgpool=AvgPool3d([3,1,1], stride=1, pad=[1,0,0])`（时间邻域平均旁路），`b_avgpool_bn` 的 weight/bias **全零初始化**（起步等价普通 2D conv）→ ReLU。
- 侧路：`x_1 = conv1(x)`（1×1 BaseConv，逐帧）；输出 `conv3(cat(x_1, x))`，conv3 为 1×1 BaseConv 2C→C。即"校准特征 + 时间 AvgPool 旁路 + 当前帧 1×1 侧路 concat"。

**R_CLSTM_5 精确结构**（`nets/yolo.py`，C=该层通道数，hidden=8）：
- 门：`F_t/I_t/C_t = Conv3x3(C+8→8)` + sigmoid/sigmoid/tanh；`Q_t = Conv3x3(C+8→C)` + sigmoid。输入 `cat(x_t, h_{t-1})`。
- 更新：`c_t = F_t·c_{t-1} + I_t·C_t`；`(h_t, out_t) = Refine(cat(c_t, Q_t))`。
- `Refine = R_3`：两层 5×5 conv（C+8→C+8，默认无 BN）+ReLU → `convh` 3×3（C+8→8）出新 h；`conv2` 5×5（C+8→C）出该帧输出。**h/c 每窗口零初始化**，顺序 t-3→t 迭代 4 步。
- 注意：输出门作用在 Refine 之前（Q_t 参与 Refine 输入），与论文 Eq.17 `ĥ=F(o·C)` 写法不完全一致，以代码为准。

**渐进队列**（每尺度 6 个 `reduce*_*` = 1×1 BaseConv C→C/2，带 BN+SiLU）：
```
m0 = cat(r0(h_{t-3}), r1(h_{t-2})) + h_{t-2}
m1 = cat(r2(m0),      r3(h_{t-1})) + h_{t-1}
m2 = cat(r4(m1),      r5(h_t))     + h_t      # 残差挂在被融合帧上
```
只有 m2（当前帧融合特征）送 head；PAFPN 内部自底向上路径仍用未融合的逐帧特征。

**PAC**（后处理）：候选框按 IoU>θ 建邻接图；低分邻居集 L（IoU>δ 且分数更低）给正增强 `E = Q/(Q+1)·(1−P(a_i))·max P(a_j)`；高分邻居抑制 `S = P(a_i)·IoU(a_i, a_j*)`；`P̂ = P + E − S`（论文 Eq.21 首项写成 P(a_j)，疑笔误）。δ≈0.8 最优，敏感性 <0.6 AP。

### 1.2 训练配方
COCO 预训练微调 50 epochs；RTX 3090，batch=2；SGD lr=0.001×bs/64（线性缩放），cosine + 5 epoch warmup，wd=5e-4，momentum=0.937；输入 640×640；Mosaic+Mixup（对窗口内每帧独立做，沿用 YOLOX）。窗口 4 帧：`framerange=50` 内随机抽 3 个过去帧（不放回、排序、末位强制 t-1；不足则放回抽样）。代码默认 freeze 20 epochs（bs16）后解冻 bs4。损失 = YOLOX 原损失，只监督当前帧，**无任何时序一致性损失**。

### 1.3 关键结果（SUN 结肠镜视频，按 case 7:1:2 划分）
- YOLOX-X 基线 → TSdetector：AP50:75 52.4→56.4，AP50 93.7→95.3，AP75 54.5→61.2，Recall 90.4→92.8，F1 91.0→92.1。
- 速度/参数：33.61→28.29 FPS（3090），92.07M→98.61M。GT-Conv 单独 −1.46 FPS/+2.04M；HQIM −4.04 FPS/+4.50M。
- 消融（AP50:75）：+GT-Conv 54.5（+2.1）；+HQIM 54.4（+2.0）；+PAC 54.2（+1.8）；GT+HQ 55.4；三件套 56.4。AP75 提升更大（PAC +2.7）。
- 阴性视频 FP 率：17.23%→10.21%（时序上下文压假阳）。
- **模块可移植性实证**：换基座 YOLOv8 54.7→57.5、FCOS 47.5→51.2、RDN 43.7→45.2 —— 三模块与检测器解耦，这是移植 YOLO26 的最直接依据。

### 1.4 移植到 YOLO26
- TAda block 可整体搬到 neck 输出→Detect 之间（输入改为滚动缓存的 T 帧 neck 特征，无需动 backbone）。α 零初始化 + AvgPool 旁路零初始化 BN 保证从预训练权重平滑起步，对 ultralytics 微调极友好。
- R_CLSTM 的 5×5 Refine 在 C=256 时很贵（这是 HQIM −4 FPS 的主因），建议换 1×1/3×3 门的轻量 ConvGRU；渐进队列本身极便宜（6 个 1×1），可保留。
- PAC 与 one-to-one 头**不兼容**（每目标只出 1 框，无候选簇可聚类）——放弃，或用 CL-MC 式输出级状态机替代其职能。
- 官方代码每窗口重提 4 帧 backbone（4× 计算）；移植时必须改为逐帧特征缓存，这是最大的免费提速点。

---

## 2. UltraDet（MICCAI 2023）：负时序上下文抑制假阳，乳腺超声实时检测

### 2.1 时序机制
BasicDet = ResNet34（`HALF_CHANNEL=True`）+ RPN（16 proposals）+ RDN 式 Temporal Relation head（L=2 层，d_model=256），特征层 **res4（stride 16）**，res5 block 移入 RoI head。输入当前帧 + T=15 个过去帧。

**IOF Align**（论文 Eq.3 + `context_head.py`）：
- `O_{t→τ} = FlowNetS(I_t, I_τ)`，**冻结**（加载 DFF 权重，`flownet_method="DFF"`），pooling stride=4 出 1/4 分辨率 flow，再双线性缩放到特征分辨率。
- 实现：`grid_sample(F_τ, flow_grid)` warp 整张过去特征，再乘 FlowNet 输出的 `scale_map`（DFF 置信加权）；随后用**当前帧的 box B_t** 在 warp 后特征上 RoIAlign + 相对位置编码（`cpb_mlp`: Linear(2→d)→ReLU→Linear(d→d)，log 尺度相对坐标）+ 空间平均 → C_{t,τ}。数学上等价于论文的"对 RoI 规则网格施加逆光流偏移再采样"。**关键 trick：warp 特征而非 box，当前帧 proposal 可直接复用**。
- 上下文帧 T_ctxt=2，推理时 `CONTEXT_STEP_LEN=8`（每 8 帧采一帧入 deque，覆盖约 16 帧跨度）；所有帧的 proposal 共享同一组上下文帧。

**Temporal Aggregation**（Eq.4，`TemporalContextAggregation`）：`Q^l = Q^{l-1} + MHA(Q^{l-1}, C_t, C_t)` + Dropout + LayerNorm。**Q=当前 proposal 特征（1, B*N, D），K=V=该 proposal 在 2 个上下文帧的特征（T_ctxt, B*N, D）**——注意力只在同一 proposal 的时间维上做，不跨 proposal；跨 proposal 交互交给 Relation 层（Eq.5，全体 16 帧 × 16 proposals）。消融替代：时间平均 + concat + Linear(2D→D)。
- 负上下文证据机制：不做显式"阴性判别"，而是让 RoI 级注意力看到过去同位置的非病灶外观（NTC），由分类头学着压低分数。放射科医生判定 50–70% 的 FP 可被 NTC 抑制。

### 2.2 训练配方
8×RTX 3090；10k iters，AdamW lr=2e-4，steps(6000,8000)，warmup 1000；`SEGS_PER_BATCH=8`（每 GPU 1 段）；每段 16 帧，`TRAIN_FRAME_SAMPLER INTERVAL=30`（帧间隔 ≤30 随机）；固定面积 900000（≈0.9MP），scale jitter 0.3–1.5；训练时上下文帧从 16 帧内随机选 2（可含"未来"帧，推理才严格因果）。损失 = L_reg + L_cls（Faster R-CNN 式），当前帧 + **全部过去帧加辅助损失**。推理缓存过去帧中间结果（proposal/特征/multi_features 的 history buffer）。

### 2.3 关键结果（CVA-BUS 186 视频，重标注高质量框，4-fold）
- UltraDet：Pr90 90.8%（比最强对手 +5.4），FP90 5.7 seq/min（约减半），AP50 91.6，R@16 93.8，30.4 FPS（TensorRT 35.2）。BasicDet：Pr90 83.6 / FP90 13.4 / AP50 88.7 / 42.2 FPS。
- 消融（Pr90 / FP90）：无 NTCA 83.6/13.4；只 TempAgg（普通 RoIAlign 无 flow）84.3/12.5；只 IOF Align（时间平均池化）88.7/9.0；两者 90.8/5.7 —— **光流对齐贡献最大，注意力聚合其次**。
- 聚合层级：RoI 级 90.8 > 特征级（FGFA warp）84.6 > 双级 88.7（双级反而无益）。
- 指标设计值得照搬：Pr80/Pr90（高召回下精度）、FP80/FP90（FP 序列数/分钟）、AP50（超声边界模糊，AP75 无临床意义）、R@16。

### 2.4 移植到 YOLO26
- NTCA 是 RoI 级模块，one-to-one 稠密头没有 RoI —— 不能直接搬。两条路：(a) 输出级近似：对最终 top-1 框，在缓存的过去 P4/P5 特征上按对齐偏移采样小邻域特征，接一个小 MLP/attention 复核置信度（单结节场景开销极小）；(b) 放弃显式 NTC，靠 neck 级时序融合隐式实现。
- FlowNetS 在 3060 上跑 1/4 分辨率光流约 10ms+ 量级，实时预算吃不下；探头运动以全局平移/缩放为主，可用 1/8 灰度图相位相关或 ECC 估全局仿射（CPU 即可），把过去 P5 特征整数/亚像素平移后再融合，作为廉价对齐消融项。
- "过去帧辅助损失"与"推理特征缓存"两个工程点可直接照搬。

---

## 3. STNet（MICCAI 2023）：时空可变形注意力，乳腺视频（非因果）

### 3.1 时序机制
Deformable DETR 基座。输入 6 帧 = 3 相邻帧（k−1,k,k+1）+ 3 全视频随机参考帧 → **非因果**。
STDA（Eq.1）：`STDA(z_q,p_q,{F_t}) = Σ_m W_m Σ_{t=1..T} Σ_{l=1..L} Σ_{k=1..K} A_{tlqk}·F_t^l(φ_l(p_q)+Δp_{tlqk})`；偏移 Δp 与权重 A 均由 query 经线性层预测，A 在 T×L×K 上 softmax 归一。相对标准 deformable attention 只是把采样域扩到"帧×尺度×点"。
- ST-Encoder：6 层 STDA+FFN，query=每个多尺度像素（局部时空融合发生在每一层，批评 CVA-Net 只在 encoder 前做一次全局 self-attention）。
- ST-Decoder：6 层 self-attn + STDA cross-attn + FFN，N 个可学习 query。
- **encoder feature shuffle**（推理提速核心）：backbone+encoder 只跑一次，交换 3 个相邻帧在 encoder 特征中的顺序，decoder 跑 3 次 → 一次前向出 3 帧预测。3090 上 21.84 FPS（CVA-Net 12.17）。

### 3.2 训练配方
ResNet-50 ImageNet 预训练；两阶段：Adam 8 epochs → SGD 20 epochs，lr=5e-5，wd=1e-4，batch=1，单卡 A100；多尺度短边 480–800、max 1333（测试 512），随机翻转/裁剪/缩放。损失沿用 Deformable DETR（匈牙利匹配 + focal cls + L1 + GIoU），无时序一致性损失。

### 3.3 关键结果（BLUVD-186 = CVA-BUS，CVA-Net 划分，测试 38 视频；含良/恶性两类）
AP 40.0 / AP50 70.3 / AP75 43.3（CVA-Net 36.1/65.1/38.5）。消融（AP）：单帧基线 30.2；多帧全局 MHSA 融合 35.1；只 ST-Encoder 34.9；只 ST-Decoder 35.8；全量 40.0 —— **局部采样式时空融合显著优于全局注意力融合，且 encoder/decoder 两端都要**。注意与 UltraDet 数字不可比（划分、类别数、标注版本都不同）。

### 3.4 移植到 YOLO26
- 整体不可搬：非因果 + MSDeformAttn CUDA op + DETR 框架。可取的结论：① 局部时空融合 > 全局注意力（支持选 TAda/ConvGRU 而非 cross-frame transformer）；② "共享编码、按帧重排解码"思想 ≈ 逐帧特征缓存 + 窗口复用；③ 随机参考帧提供长程上下文（与 MWNet 记忆库互补）。若想要可变形采样，可用 `grid_sample` 实现 4 点/层的双线性版本（无自定义 CUDA），但 6GB 上优先级低。

---

## 4. MWNet（MIA 2026）：小波卷积 + 长短期记忆库，甲状腺超声视频分割

### 4.1 时序机制与架构
Encoder = WTConvNeXt（ConvNeXt-T 式 4 stage，通道 [96,192,384,768]）：每 stage 若干 WTConv block + 一个 TemporalFusion；最深层后接 LSMemory；Decoder = 逐级 HFF。
**WTConv2d**（`MWConv.py`，即 ECCV24 Finder et al. 方案）：固定 db1/Haar 小波滤波器（`requires_grad=False`）；`base_conv` = 5×5 depthwise；每级小波分解出 LL/LH/HL/HH，4 子带拼一起做 5×5 depthwise（groups=4C），`_ScaleModule` 初始 0.1；LL 递归分解（论文：直到最深层分辨率的 1/4），IWT 逐级重建：`X_LL2^{k-1} = IWT(X_LL2^k, X_H1^k) + X_LL1^{k-1}`，输出 `IWT(...) + DWConv(X)`。作用：小核获得大感受野 + 高频细节保留，省显存。
**TemporalFusion**（代码核对，非标准 ConvLSTM）：f/i/o/c_hat 全为 **1×1 Conv(2C→C)+BN**（sigmoid×3 + tanh）；`c = Conv1x1_tanh(cat(f*c, i*c_hat))`；`h = o * leaky_relu(c)`。每段（10 帧）零状态起步，输出各帧 h。
**LSMemory（长短期记忆库，仅最深层 C=768）**：
- 读（Eq.6）：LN → 双路 CrossAttention（8 头，正弦位置编码加在 Q 和 K/V 上）：`f_l = Softmax(Q_t K_l^T/√C) V_l`，Q=当前帧像素（HW 个 token），K/V=记忆库所有帧像素在空间维拼接（t_l·HW 个 token）；短期同理 → `conv1x1(cat(f_l,f_s)) + x` → LN → FFN（1×1 C→4C，GELU，1×1 4C→C）+ 残差。
- 写/压缩（`update_memory`，论文只给文字，规则以代码为准）：记忆条目 `clone().detach()`（**梯度不回传历史**）。短期库恒为最近 2 帧（t_s=2）。长期库追加新帧，超过 t_l=5 时压缩到 5：各帧特征 flatten 成 C×HW 向量 → L2 归一化 → 两两余弦相似度矩阵（掩掉上三角）→ 取 top-(n−5) 相似对 → **删除每对中较晚的帧**（保留早期信息）。视频切换（帧号不连续）→ 双库重置。
**HFF（解码器）**：`Y = AWLF(UP(pw(Z_{l+1}))) + AWHF(pw(X_l))`；`Z_l = pw(AWLF(Y)) + pw(AWHF(Y))`。AWT 用提升方案：按奇偶拆分，`H = x_o − ConvP(x_e)`，`L = x_e + ConvU(H)`（P/U 为可学习 1D conv），行列各做一次成 2D；SE 通道注意力对 4 子带重加权（高通偏 HF 子带、低通偏 LF），IAWT 重建。作用：抑制斑点噪声 LF 化 + 锐化边界 HF。

### 4.2 训练配方
AdamW lr=1e-4，PolyLR + LinearLR warmup + layer-wise lr decay；batch=2，180k iters；输入 512×512；视频随机起点裁成 10 帧段（**段内所有帧同一套增广**：blur/flip/color jitter）；损失 = Dice + BCE（逐帧，无时序一致性损失）；WTConv backbone 先 ImageNet-1k 预训练 300 epochs；单卡 RTX 4090；72.99M 参数，28 FPS（bs=1）。公开 config `frames=3`（与论文 10 帧段不一致，复现时注意）。

### 4.3 关键结果
- ThyroidNodule（64 视频 5611 帧，40/10/14 划分）：DSC 88.03 / IoU 78.61（LGR-Net 85.58/74.79，MemSAM 85.48/74.64）。VTUS（100 视频 9342 帧）：87.68/78.06。TG3K：87.90/78.41。CAMUS：94.34/89.29。
- 消融（ThyroidNodule DSC）：ConvNeXt 基线 82.63 → +WTConv 84.46（+1.83）→ +Memory（TemporalFusion+记忆库）87.21（**+2.75，最大头**）→ +HFF 88.03（+0.82）。VTUS 上 Memory +3.34。小结节/长视频收益最大；大目标（腺体/心脏）收益小。
- 记忆库长度扫描：先升后降（冗余有害）；段长 10–60 精度基本不变（记忆机制对段长鲁棒）。
- 可视化：无记忆库时相邻帧边界抖动明显，有记忆库时序一致性显著改善。

### 4.4 移植到 YOLO26
- 记忆库思想最值得搬：放在 P5（20×20=400 token，7 条记忆=2800 key，d=256 的 cross-attn 毫秒级）。t_l=5/t_s=2 + 余弦压缩 + detach 写入 + 场景切换重置，全套规则可直接复用，解决"探头移开再移回"的重识别。
- TemporalFusion 的 1×1 门 ConvLSTM 比 TSdetector 的 5×5 R_CLSTM 便宜一个量级，是轻量时序融合的好模板（但每段清零 = 无跨段记忆，靠记忆库补）。
- WTConv 换 backbone 会丢 YOLO 预训练权重，不建议；HFF/自适应小波可作为 neck 融合处的可选抗斑点噪声模块（有 FreqFusion 等更轻替代），列为二期消融。

---

## 5. CL-MC（arXiv 2026.02）：检测器闭环校正 SAM2 记忆，声门定位（免训练）

### 5.1 机制与决策规则
YOLO12-m 检测器 + SAM2.1-Large tracker，无微调。初始化：首个 `s_y > τ_init=0.75` 的帧用检测框 prompt SAM2。
**置信度对齐**（Eq.1）：`s'_s = (s_s − min(H_t)) / (max(H_t) − min(H_t) + ε)`，H_t = 过去 T_win=30 帧 tracker 分数滑窗（20–40 均稳定）。`v_iou = IoU(b_y, b_s)`。
**状态机**（Algorithm 1，按序判定）：
1. Agreement：`v_iou > τ_iou=0.5` → `α = s_y/(s_y+s'_s)`，`b_final = α·b_y + (1−α)·b_s`；软更新记忆。
2. YOLO Lost：`s_y < τ_lost=0.1` → 用 `b_s`；软更新。
3. Drift：`s_y > τ_drift=0.5`（0.7 略优）且 `v_iou < τ_iou` → 用 `b_y`，**硬重置** `M_t ← Encode(I_t, b_final)`（整库覆盖）。
4. 其余冲突 → 用 `b_s`；软更新 `M_t ← Update(M_{t-1}, Encode(I_t, b_final))`。
即：**硬重置触发条件 = 检测器高置信 + 与 tracker 空间不一致**；其余一律软更新。

### 5.2 训练配方（仅检测器）
YOLO12-m，COCO 初始化，Laryngoscope8（2497 图）+ YouTube 标注 583 图，微调 200 epochs。评测：Harborview 急诊插管 24 视频 8931 帧（297s），逐帧标注。闭环部分零训练。

### 5.3 关键结果
mAP50 84.32（YOLO12 82.41），Miss Rate 6.85%（8.81%），AUC 76.52（74.57）；mAP50:95 50.95≈51.05（框贴合度略降，换稳定性）。裸 SAM2 只有 70.47 / Miss 22.24%。消融：开环覆盖记忆 82.88/Miss 8.63；固定 0.5 融合 84.15/7.01；去掉滑窗归一化 83.85/7.02 —— **闭环重置与置信度对齐各贡献约 1 个点 Miss**。drift 帧上 YOLO 平均 IoU 0.588 vs SAM2 0.459（漂移主因是 tracker）。数据越少闭环收益越大（70% 数据时 Miss −2.9）。超参极稳（AUC 76.5±0.16），无需精调。

### 5.4 移植到 YOLO26
SAM2-L 在 6GB 上不可能实时，整套 tracker 不搬。可免训练搬的是**输出级状态机**：单结节场景维护一条 track；一致时按置信度加权 EMA 平滑框；检测丢失 ≤N 帧用预测/保持框"惯性滑行"（降置信输出或不输出）；高置信检测与 track 冲突 → 立即采信检测并**重置时序模块状态**（ConvGRU hidden、记忆库、分数滑窗）。这正是 PAC 在 one-to-one 头上的职能替代，也是时序模块漂移的保险丝。

---

## 6. 对 YOLO26 改造的设计建议

### 6.1 总体判断
- 五篇共识：**因果 + 特征缓存 + 局部时序融合**是实时医学视频检测的正确形态（TSdetector/UltraDet 实证）；全局 cross-frame attention（STNet 基线、CVA-Net）贵且弱。
- 收益排序（各自消融）：对齐/校准类（UltraDet IOF +4.4 Pr90；TSdetector GT-Conv +2.1 AP）≥ 记忆/融合类（MWNet Memory +2.75 DSC；TSdetector HQIM +2.0）> 输出级校正（TSdetector PAC +1.8；CL-MC Miss −2）。甲状腺单结节 + 探头运动场景，预期 FP 抑制（UltraDet 路线）与丢帧找回（MWNet 路线）是主要收益来源。
- 所有论文都**没有时序一致性损失**——只做特征级融合 + 逐帧检测损失。这既是安全默认，也是可发论文的空白点。

### 6.2 推荐架构（YOLO26n/s，640 输入）
```
frame_t → backbone+neck（每帧只跑一次）→ P3/P4/P5 入滚动缓存（T=4）
  ├─ Module A: TAda-lite @ P4,P5（可选 P3）
  │    RouteFuncMLP(ratio=4, kernels=[3,3]) → α=1+zero-init Conv3d[3,1,1]
  │    校准 conv3x3 + AvgPool3d[3,1,1] 旁路(zero-init BN) + 当前帧 1x1 侧路 concat
  ├─ Module B: ConvGRU(hidden=8~16, 门 3x3 或 1x1) @ P5（可选 P4）
  │    + TSdetector 渐进队列（1x1 C→C/2 ×6，残差挂被融合帧）→ 融合 P5
  │    推理时状态跨帧携带（改掉每窗口清零），场景切换时重置
  ├─ Module C（二期）: MWNet 式记忆库 @ P5：t_l=5, t_s=2，
  │    余弦压缩删晚帧，detach 写入，cross-attn d=256/8头
  └─ Detect(P3, P4_fused, P5_fused) → one-to-one 输出
       └─ Module D: CL-MC-lite 状态机（免训练）：滑窗分数归一化 +
            一致→置信度加权 EMA；丢失≤5帧→滑行；高置信冲突→采信检测+重置 B/C 状态
```
- 通道宽度：按 YOLO26n neck 实际宽度（约 P3=64/P4=128/P5=256）；GRU hidden 8 起步（TSdetector 用 8 已够）；记忆库投影到 128–256。
- 窗口 T=4（t-3..t，与 TSdetector 一致）；训练采样照搬 TSdetector：过去 3 帧从 ≤30–50 帧内随机抽、末位强制 t-1，天然抗探头快慢速差异；推理用连续帧（或 stride 2 扩时域覆盖）。
- 对齐：不搬 FlowNetS。先不做显式对齐（相邻帧位移小，TAda+GRU 可隐式吸收）；二期加 1/8 分辨率相位相关/ECC 全局仿射 → 平移缓存 P5，作为消融项。
- 增广：窗口内 4 帧**同一套**几何/色彩增广（MWNet 做法；TSdetector 逐帧独立 Mosaic 会破坏时序对应，建议窗口内禁用或共享 Mosaic 参数）。

### 6.3 6GB 显存与实时预算
- 训练：过去帧 backbone/neck 用 `no_grad` 提特征并 detach（MWNet 记忆 detach 同理），只对当前帧路径反传 → bs 8–16 @640 AMP 可容；若效果不足再消融"全窗口反传 bs2–4"。注意 BN：缓存特征跨 step 复用需冻结或换 GN/冻结 BN 统计。
- 推理：逐帧缓存替代 TSdetector 的 4 帧重提（省 ~75% backbone 计算）。TAda-lite+GRU+队列在 n 宽度下增量 <1M 参数、预计 <10% 延迟；记忆库 cross-attn（400×2800×256）约 1–2ms。目标 3060 上 ≥30 FPS。
- 不要引入：MSDeformAttn CUDA op（STNet）、FlowNetS（UltraDet）、SAM2（CL-MC）、WTConv 换 backbone（丢预训练）。

### 6.4 消融清单（按优先级）
1. baseline YOLO26n vs +A vs +A+B vs +B（分离校准与记忆贡献，对照 TSdetector Table 5）。
2. T ∈ {2,4,6}；GRU hidden ∈ {8,16}；作用层 P5 vs P4+P5 vs 三层全做。
3. 训练采样：相邻帧 vs 随机 ≤50 帧（TSdetector 配方）；窗口内共享增广 on/off。
4. 过去帧辅助损失（UltraDet 式，特征已算好，近乎免费）on/off；可选时序一致性正则（相邻窗口融合特征 L2 / 框抖动惩罚）——文献空白，小权重试。
5. 全局仿射对齐 on/off；记忆库 on/off（重点看探头移开-移回序列）；状态机 on/off（重点看 FP 序列数与抖动）。
6. detach 缓存训练 vs 全窗口反传（精度-显存权衡）。
- 评测指标：AP50 为主（超声边界模糊，学 UltraDet 不追 AP75）；**FP seq/min @ recall 0.9**（UltraDet 协议，临床最相关）；相邻帧框中心抖动/IoU 方差（时序稳定性）；阴性视频 FP 率（TSdetector 协议）；3060 实测 FPS 与峰值显存。

### 6.5 风险与备注
- one-to-one 头 + reg_max=1：所有融合放在 Detect 之前不影响匹配分配；PAC 类候选框聚类彻底不适用。
- TSdetector 的 R_CLSTM 输出门位置论文与代码不一致、PAC Eq.21 疑似笔误、MWNet 公开 config frames=3 与论文 10 不符——复现对照时以代码为准。
- 域差：息肉/乳腺/声门的运动幅度与形变都大于甲状腺扫查；甲状腺帧间变化小（MWNet 在 TG3K 上视频方法收益微弱），预期时序模块在小结节、低对比度、探头快速移动片段收益最大，静态大结节片段收益有限——消融时按结节大小/运动幅度分层看指标。