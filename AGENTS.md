# AGENTS.md — 接手指南（任何 agent 从这里开始）

甲状腺超声视频实时结节检测研究项目。基线 YOLO26n，改造方向：严格因果时序建模 + 单图数据混入。
本文件是交接入口；完整实验台账见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 1. 环境（本机，Windows）

- **Python**：`C:\Users\bbuok\.conda\envs\vocod-yolo\python.exe`（conda env `vocod-yolo`，torch 2.7.0+cu128）。
  所有脚本从**项目根目录**用该解释器运行（脚本内部已把本地 ultralytics 与项目根插入 sys.path）。
- **ultralytics**：本地仓库 `C:\Users\bbuok\udisk-code\ultralytics`（8.4.175，**未修改的上游代码**，
  不要 pip 安装覆盖）。预训练权重在其 `pt/yolo26n.pt`。
- **GPU**：RTX 3060 Laptop **6GB** —— batch16/imgsz640/T4 是显存上限配置（峰值 ~3.3GB）；
  不要同时跑两个训练。推理评测（~0.6GB）可与训练共存但会拖慢双方。
- **AMP 检查离线化**：GitHub 下载被墙。`weights/yolo26n.pt`（项目内，gitignored）是预放的
  AMP 检查资产（check_amp 读 WEIGHTS_DIR），**勿删 weights/ 目录**，否则 AMP 检查尝试联网。
- **RAM 16GB**：dataloader workers=2；trainer 已跳过训练末尾 full-test final_eval（会 OOM），
  权威评测一律用 `scripts/stream_eval.py` 单独跑。
- `thyroid_temporal/robust.py`：Windows 下 results.csv 瞬态文件锁的重试补丁，训练脚本自动安装。

## 2. 分支地图

| 分支 | 内容 | 状态 |
|---|---|---|
| `main` | 基线 + lite/lrdisc 时序模型与全部诊断 | lrdisc ep20 为该分支最佳交付 |
| `aux-supervision` | main + 因果辅助监督（Auto-DFLLs 借鉴，--aux） | **负结果**：w=0.5 以召回换精度，mAP 净降；见该分支 docs/aux_supervision_result.md |
| `static-mix` | main + ThyroidXL 单图混入（--static-mode）+ 去噪研究 + 检查点续训 | **正结果**：xlrepeat ep18 mAP50 .665 > lrdisc ep20 .638；见 docs/static_mix_result.md |

模型架构代码（temporal_modules.py）三分支一致；static-mix 对训练循环的改动在 repeat 模式下行为等价于 main
（差异仅数据 + 默认关闭的开关 + 续训支持）。本文件与 EXPERIMENTS.md 在 main/static-mix 同步维护；
分支特有文档（static_mix_result.md、xl_noise_analysis.md 仅在 static-mix；aux_supervision_result.md
仅在 aux-supervision）需切到对应分支阅读。

## 3. 数据（data/ 与 runs/ 均 gitignored，本机存在；换机需重建）

### 视频数据（主数据）
- 源：`C:\Users\bbuok\Documents\ChatGPT\甲状腺大模型\公开数据集\dataset.hdf5`
  （192 段视频、17412 帧、802×1054 灰度 + 结节掩码；血管无标注，是主要 FP 来源）。
- 重建：`python scripts/convert_hdf5_to_yolo.py` → `data/thyroid_yolo/`
  （PNG `视频名_帧号4位.png`，单类 nodule；**按视频切分** train106/val43/test43 段 = 12185/2617/2610 帧）。
- 子集：`python scripts/make_subset.py` → train_sub/val_sub（1/5，硬链接）。
- yaml：thyroid_full.yaml（train 全量 + val_sub）、thyroid_fast.yaml（train_sub + val_sub）、thyroid.yaml。

### ThyroidXL 单图数据（static-mix 分支引入）
- 源：`C:\Users\bbuok\Documents\ChatGPT\甲状腺大模型\公开数据集\ThyroidXL`
  （MICCAI 2025 workshop，DOI 10.1007/978-3-032-05182-0_60；4093 患者 11545 图，
  训练 3354 患者 9541 图 / 测试 739 患者 2094 图，患者级划分无重叠；README 的 3275 训练患者数有误）。
- 原标注：YOLO bbox + COCO JSON + 分割掩码；2 类（0 良性 lanhtinh / 1 恶性 actinh），每图恰 1 框。
  图像为裁剪后超声区域（主流 727×532），**非连续视频帧**（每患者 1–10 张独立关键帧，已量化核实）。
- 重建：`python scripts/convert_thyroidxl.py` → 每图一个单帧伪视频 `xl*****_0000.png`，
  两类合并为 nodule；`images/train_mix` = 视频 train + XL（21726 帧，43.9% 单图，硬链接）；
  `thyroid_full_xl.yaml`；XL test → `images/test_xl`（静态评测用）。
- **噪声结论（重要，勿重复踩坑）**：XL 的"严重噪声"是空间相关散斑纹理（var∝mean^1.67，lag-1 ρ=0.88），
  非椒盐/JPEG/周期干扰。**8 种去噪全部损害检测**（零样本探针 mAP50 .596→.214 随强度单调下降）；
  散斑携带检测信号，**混入训练用原图**。详见 static-mix 分支 docs/xl_noise_analysis.md。

## 4. 复现命令（均从项目根、vocod-yolo 解释器）

```bash
# 基线（全量 30ep）
python scripts/train_baseline.py --data data/thyroid_yolo/thyroid_full.yaml --epochs 30 --name baseline_full
# lrdisc 时序（main 最佳交付配方：全量 20ep，从基线初始化）
python scripts/train_temporal.py --data data/thyroid_yolo/thyroid_full.yaml --weights runs/baseline_full/weights/best.pt --epochs 20 --batch 16 --name temporal_lrdisc --levels lrdisc --strides 3,10,25
# 单图混入（static-mix 配方）
python scripts/convert_thyroidxl.py
python scripts/train_temporal.py --data data/thyroid_yolo/thyroid_full_xl.yaml --weights runs/baseline_full/weights/best.pt --epochs 20 --batch 16 --name temporal_lrdisc_xlrep --levels lrdisc --strides 3,10,25 --static-mode repeat
# 检查点续训（epoch/优化器/EMA 全恢复；tcfg 从 run 目录 tcfg.json 重建）
python scripts/train_temporal.py --resume runs/<name>/weights/last.pt
# 权威评测（流式因果、按视频顺序、滚动缓存；时序模型加 --temporal）
python scripts/stream_eval.py --weights runs/<name>/weights/last.pt --temporal --split test
# 时序真实性诊断 / PR 工作点 / 定性
python scripts/diagnose.py --weights runs/<name>/weights/last.pt
python scripts/pr_analysis.py
python scripts/visualize.py
```

## 5. 已知陷阱（全部踩过，勿重复）

1. **权重加载顺序**：必须先以原始 Detect 索引（23）加载预训练权重，再插时序层（Detect 移到 26）。
   build_temporal_model 已按此实现；自建模型时注意。
2. **续训头宽**：本地 ultralytics `BaseModel.load` 会按**源模型**宽度重建分类分支（c3=80 继承自
   COCO 头，而非 nc=1 推出的 64）。续训必须走 load() 路径（已实现 + 逐张量校验），strict load 会炸。
3. **续训数据覆盖**：ultralytics check_resume 会用传入的 data 覆盖检查点 data。续训时**只传
   resume + 运行时参数**（已实现）；曾因此静默训到纯视频集（762 vs 1358 迭代/epoch 可辨别）。
4. **val 必须 nms=False**：YOLO26 end2end 一对一推理，否则 val mAP=0（训练脚本已设）。
5. **val_sub（539 帧）选点不可靠**：aux 与 xlrep 实验均出现 val 排序与 full-test 不一致；
   权威结论一律看 stream_eval full test；best.pt 仅作参考。
6. **历史帧 BN 用 eval 模式**：_forward_clip 中历史 no_grad + eval（running stats），否则训练/推理
   特征失配使时序融合有害（已诊断修复，勿改回）。
7. **FPS 测量**：GPU 空闲时测（batch1/640/FP16）；与训练共存时测得的 FPS 偏低无效。
   lrdisc 持久性门已 F.unfold 向量化（数学等价，29.3→35.8fps，无需重训）。
8. **RESULTS.md 是历史自动产物**（lite 时代数字，曾为 GBK 乱码已转 UTF-8）；
   `scripts/make_report.py` 会重新生成它。**权威数字以 docs/EXPERIMENTS.md 为准**。
9. **runs/ 与 data/ 不入 git**：换机后权重/结果 JSON 不存在，按 §4 重建；关键数字已固化在文档。
10. 中断路径：训练可安全 kill（last.pt 每 epoch 落盘）；续训见 §4。挂起进程方式已弃用（占显存）。

## 6. 当前状态与开放问题（2026-11-23）

- **最佳交付**：`runs/temporal_lrdisc_xlrep/weights/last.pt`（ep18，test mAP50 .665 / F1 .679 / 34.8fps）。
  训练停在 18/20（用户指示），续训命令见 §4。
- **已确认**：单图混入（repeat）有净收益（best ep8 以 72% 步数超 lrdisc ep20 + last ep18 无选点确认）；
  去噪有害；aux 监督 w=0.5 负收益；复杂 full 模型过拟合。
- **开放方向**（按价值排序）：
  1. 等步数对照（xlrep ~ep11–12 ≈ 15.2k 步 vs lrdisc ep20）——需重训并存中间检查点（save_period）；
  2. black/jitter/mixed 历史合成变体（代码已就绪，--static-mode 切换即可）；
  3. 混入比例扫描（当前 43.9% 单图；--frac 控制 XL 用量）；
  4. 噪声增强（对视频帧加散斑样噪声）替代去噪的鲁棒性方向（用户暂缓）；
  5. 血管 FP：需光流对齐/显式血管标注（REPORT.md §5 分析）；
  6. 更大服务器数据重训（REPORT.md 预期增益放大）。
- 阅读顺序建议：本文件 → EXPERIMENTS.md（台账）→ REPORT.md（时序方法结论）→
  static-mix 分支的 static_mix_result.md + xl_noise_analysis.md（单图混入）→ paper_notes.md（文献机制）。