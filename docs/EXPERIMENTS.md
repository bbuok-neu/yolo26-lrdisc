# 实验台账（EXPERIMENTS.md）— 权威数字与运行注册表

本文件固化所有关键结果（runs/ 与 data/ 不入 git，换机后此表是唯一权威来源）。
评测协议：`scripts/stream_eval.py`，完整 test 2610 帧，按视频因果顺序，滚动特征缓存，
FP16/batch1/640，ultralytics ap_per_class；FPS 只在 GPU 空闲时有效。
接手入口见根目录 [AGENTS.md](../AGENTS.md)。

## 1. 权威结果总表（视频 test，2610 帧）

| 运行/权重 | 分支代码 | 训练数据 | 预算 | mAP50 | mAP50-95 | P | R | F1 | FPS | 结果 JSON（runs/） |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_full（30ep 末） | main | 视频全量 | 30ep | .6088 | .3166 | .7211 | .5645 | .6333 | 35.5 | stream_test_baseline_full |
| temporal_v1 best（TAda+GRU+queue+EMA, T4） | main | 1/5 子集 | 50ep | .3267 | .1466 | .4378 | .3328 | .3782 | 25.3 | stream_test_temporal_v1 |
| baseline_sub（50ep） | main | 1/5 子集 | 50ep | .3695 | .1782 | .4540 | .4117 | .4318 | 34.8 | stream_test_baseline_sub |
| temporal_full best（复杂时序） | main | 视频全量 | 30ep | .5826 | .2910 | .6463 | .5052 | .5671 | 26.5 | stream_test_temporal_full |
| temporal_lite best | main | 视频全量 | 15ep | .6171 | .3268 | .6832 | .5955 | .6363 | 31.9 | stream_test_temporal_lite |
| temporal_lrdisc best（ep6） | main | 视频全量 | 20ep | .6249 | .3248 | .6855 | .6051 | .6428 | 35.9* | stream_test_temporal_lrdisc_best |
| **temporal_lrdisc last（ep20）** | main | 视频全量 | 20ep | .6377 | .3287 | .7247 | .6090 | .6618 | 35.8 | stream_test_temporal_lrdisc_last |
| temporal_lrdisc_aux best（ep3, w=0.5） | aux-supervision | 视频全量 | 20ep | .5906 | .3127 | .7522 | .5473 | .6336 | 31.6 | stream_test_temporal_lrdisc_aux |
| temporal_lrdisc_aux last（ep20, w=0.5） | aux-supervision | 视频全量 | 20ep | .6054 | .3210 | .7936 | .5484 | .6486 | 35.5 | stream_test_temporal_lrdisc_aux_last |
| xlrep best（ep8, repeat 混单图） | static-mix | 视频+XL 21726 帧 | 20ep 计划 | .6524 | .3401 | .7748 | .5745 | .6598 | 33.5 | stream_test_temporal_lrdisc_xlrep_best |
| **xlrep last（ep18, repeat 混单图）** | static-mix | 视频+XL 21726 帧 | 停在 18/20 | **.6654** | **.3453** | **.7776** | .6025 | **.6789** | 34.8 | stream_test_temporal_lrdisc_xlrep_last |
| 探针：video test + gauss1 去噪 | baseline_full 权重 | —（评测变换） | — | .5859 | .3029 | .7285 | .5406 | .6206 | (无效†) | stream_test_gauss1_weights |

\* lrdisc best 的早期测量为 29.3fps（持久性门向量化前）；向量化数学等价，重测 35.9。
stream_test_temporal_lrdisc / _fast 与 _best 是同一权重的重复测量。
† 与训练共存时测得，无效。
子集运行（v1/baseline_sub）与全量运行**不可直接比**（训练量差 5 倍），仅作历史迭代记录。

**当前最佳交付：xlrep last（ep18）**。步数说明：xlrep 每 epoch 1358 步（21726/16），
lrdisc/baseline 每 epoch 762 步（12185/16）。xlrep best ep8=10.9k 步 < lrdisc ep20=15.2k 步；
xlrep last ep18=24.4k 步 > 15.2k 步。等步数 last-vs-last 对照缺失（无中间检查点）。

## 2. 运行注册表（runs/ 目录，本机）

| 目录 | 内容 | 初始化 | 状态 |
|---|---|---|---|
| baseline_yolo26n | 最早基线冒烟 | COCO yolo26n | 历史 |
| baseline_sub / baseline_full | YOLO26n 基线（子集 50ep / 全量 30ep） | COCO yolo26n | 完成；full 为后续时序实验的初始化源 |
| baseline_{sub,full}_{val,test} | 基线标准 val/test 输出 | — | 历史 |
| temporal_smoke / smoke2 / temporal_smoke_xl | 冒烟（smoke_xl 为 static-mix 混合集 1ep 验证） | baseline_full | 验证用 |
| temporal_v1 | 复杂时序首版（子集 50ep） | COCO yolo26n | 完成（子集，弱） |
| temporal_full | 复杂时序全量 30ep | COCO yolo26n | 完成，**过拟合低于基线** |
| temporal_lite / temporal_lite_smoke | 轻量因果均值去噪残差 | baseline_full best | 完成 15ep |
| temporal_lrdisc | 长程判别（strides 3,10,25, shift2） | baseline_full best | 完成 20ep，main 最佳 |
| temporal_lrdisc_aux | +因果辅助监督 w=0.5（aux-supervision 分支） | baseline_full best | 完成 20ep，**负收益** |
| temporal_lrdisc_xlrep | +ThyroidXL 单图 repeat 混入（static-mix 分支） | baseline_full best | **18/20ep 停止**（用户指示）；last.pt=ep17(0-indexed)；tcfg.json 在目录内；可 --resume 续训 |
| diagnose_temporal*.json | TemporalLens 诊断（normal/shuffle/freeze/zero） | — | lite: zero −38.6pp；lrdisc: freeze −4.15pp（真时序证据） |
| pr_analysis.json | PR 工作点（R@P≥70/72/75%） | — | lrdisc 在 P≥72% 下 R 超基线 +0.73pp |
| mitigated_gate.json / borrow_experiment.json | Auto-DFLLs 温和门控/借鉴探针 | — | 净收益≈0，见 docs/borrow_autodflls.md |
| stream_test_*.json | §1 各评测原始输出 | — | — |

训练日志：docs/*_train.log、docs/temporal_lrdisc_xlrep_resume.log（续训段）。
xlrep 的 results.csv 含 18 行（1–9 原始段 + 10–18 续训段）。

## 3. 静态图（ThyroidXL test_xl，2094 图）零样本探针

baseline_full 权重，同 harness（每图独立"视频"）；去噪研究用，详见 docs/xl_noise_analysis.md：

| 测试集版本 | mAP50 | mAP50-95 | P | R | F1 |
|---|---|---|---|---|---|
| original | .5956 | .2599 | .7085 | .5640 | .6281 |
| gauss1 | .5038 | .2308 | .6079 | .4709 | .5307 |
| nlm8 | .3792 | .1848 | .4911 | .3997 | .4407 |
| bilateral | .3312 | .1637 | .4278 | .3792 | .4021 |
| nlm14 | .2777 | .1423 | .3858 | .3147 | .3466 |
| lee_fit | .2141 | .1103 | .3301 | .2674 | .2955 |

结论：去噪单调损害检测；散斑是信号。噪声表征：var∝mean^1.67、lag-1 ρ=0.88、无椒盐/JPEG/周期分量。

## 4. 关键教训（跨实验）

1. 容量匹配数据量：full（+1M 参数）过拟合低于基线；lite/lrdisc（+0.2M）稳健增益。
2. 小验证集（val_sub 539 帧）选点两次误导（aux best ep3、xlrep best ep8 后 test 仍升）；
   权威结论看 full-test 流式评测，best.pt 仅参考。
3. 历史帧 BN 必须 eval 模式（训练/推理特征一致性），否则时序融合有害。
4. 血管 precision 未被当前免光流机制压住（REPORT.md §5）；需对齐/标注。
5. 单图混入 repeat 模式：precision 大涨、recall 中期下探后恢复；净收益为正（static_mix_result.md）。
6. 图像质量指标（CNR/SSIM）与检测性能负相关（去噪研究）——医学超声预处理以任务指标为准。

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| ../AGENTS.md | 接手入口：环境/分支/数据/复现/陷阱/状态 |
| ../README.md | 项目概览与快速复现 |
| REPORT.md | 时序方法对比报告（baseline/lite/lrdisc/full + 诊断 + PR 工作点 + 局限） |
| ABLATIONS.md | 消融计划与评测协议 |
| paper_notes.md | 5 篇时序论文精读（TSdetector/UltraDet/STNet/MWNet/CL-MC） |
| research_longrange_temporal.md / research_vessel_nodule.md | 文献调研 |
| borrow_autodflls.md | Auto-DFLLs 借鉴探针（门控/一致性，净收益≈0） |
| aux_supervision_result.md | 因果辅助监督负结果（**仅 aux-supervision 分支**） |
| static_mix_result.md | 单图混入结果（含 ep18 勘误；**仅 static-mix 分支**） |
| xl_noise_analysis.md | 噪声表征 + 8 方案去噪对比 + 检测探针（**仅 static-mix 分支**） |
| RESULTS.md | 历史自动汇总（lite 时代，GBK 已转 UTF-8；make_report.py 会重写它，勿当权威） |
| qualitative/ / samples/ / *.png | 定性对比与架构图 |