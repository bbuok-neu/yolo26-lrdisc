"""Auto-DFLLs architecture diagram for algorithm engineers (self-contained, standard terminology)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
    if any(f.lower() in x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

EDGE = "#334155"
C_IN = "#dbeafe"; C_BB = "#e0e7ff"; C_RPN = "#fef3c7"; C_LES = "#dcfce7"; C_IMG = "#fce7f3"; C_HEAD = "#d1fae5"; C_SIDE = "#f1f5f9"


def box(ax, x, y, w, h, lines, fc, fs=9.5, bold=False, ec=EDGE, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#0f172a", linespacing=1.35)


def arr(ax, x1, y1, x2, y2, color=EDGE, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))


fig = plt.figure(figsize=(17, 13))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(33, 98.5, "Auto-DFLLs 架构（算法工程师版 · 自包含）", ha="center", fontsize=17, fontweight="bold")
ax.text(33, 96, "两阶段检测器 + 双层时序融合：proposal 级 attention 增强真目标，图像级光流一致性抑制瞬态 FP",
        ha="center", fontsize=10.5, color="#475569")

box(ax, 4, 90, 58, 4.5, [r"输入 clip $\{I_{t-T},...,I_t\}$ (T+1 帧, 因果滑窗; 推理流式同窗口)"], C_IN, fs=10)
box(ax, 4, 80, 58, 7, [
    r"逐帧 ResNet34(ImageNet) $\rightarrow$ 层级特征 $\{C_l\}$",
    r"BiFPN 双向跨尺度加权融合 $\rightarrow$ 金字塔 $\{P_l\}$ (小病灶细节+上下文)"], C_BB)
arr(ax, 33, 90, 33, 87)
box(ax, 4, 72, 58, 5, [r"RPN on $\{P_l\}$ $\rightarrow$ 每帧候选 $\{b_i^\tau\},\ \tau\in[t-T,t]$ (多尺度)"], C_RPN)
arr(ax, 33, 80, 33, 77)
box(ax, 4, 56, 58, 13, [
    "病灶级时空融合 (proposal-level)",
    r"(a) 自适应多尺度 RoI: $f_i^l$=RoIAlign$(P_l,b_i)$",
    r"     $w_l=\sigma(W g(f_i))$;  $f_i=\|_l\, w_l f_i^l$ (自选可信尺度)",
    r"(b) 时序 relation: $Q=\{f_i^\tau\}$;  m 轮 $f_i\leftarrow$Relation$(f_i,Q,B)$",
    r"     $B$=位置编码; 当前 attend 同帧(空间)+过去T帧(时间)"], C_LES, fs=9.3)
arr(ax, 33, 72, 33, 69)
box(ax, 4, 41, 58, 12, [
    "图像级时空融合 (image-level, 背景感知时间校验)",
    r"(a) 预训练光流 $F_{t\rightarrow t-k}$; warp $b_i^t$ 到过去帧坐标",
    r"(b) 一致性/负时序上下文: warp 区域在过去帧有对应→保留",
    r"     无对应(瞬态)→抑制 → specificity↑ 且 sensitivity 不变"], C_IMG, fs=9.3)
arr(ax, 33, 56, 33, 53)
box(ax, 4, 31, 58, 7, [
    r"检测头 cls(CE)+reg(smoothL1) on 增强当前 proposal",
    r"$\rightarrow$ 最终框;  ~30.4 fps (2×4090)"], C_HEAD, bold=True)
arr(ax, 33, 41, 33, 38)
box(ax, 14, 24, 38, 4.5, ["输出: 当前帧病灶框 (实时提示)"], C_HEAD, fs=10, bold=True)
arr(ax, 33, 31, 33, 28.5)

box(ax, 66, 80, 32, 14, [
    "训练", r"$L=L_{cur}+\sum_k L_{t-k}$ (辅助监督)",
    "CE(cls)+smoothL1(reg); 每帧因果", "Adam lr2e-4, bs16, 5 epochs",
    "aug: flip0.5 + 亮度/对比度0.5"], C_SIDE, fs=9)
box(ax, 66, 62, 32, 14, [
    "推理 / 实时", "流式 T+1 窗口 (同训练)", "历史特征/光流缓存复用",
    "每帧仅当前帧过 backbone", "因果; 30.4fps (2×4090)"], C_SIDE, fs=9)
box(ax, 66, 44, 32, 14, [
    "指标 (临床导向)", "AP50: IoU0.5 的 AP", "Pr70: recall70% 时 precision",
    "FP70: FP合并成段, 段/分钟", ".7772 / .7967 / 3.4688"], C_SIDE, fs=9)
box(ax, 66, 24, 32, 16, [
    "设计要点 / 局限", "真=空间连续+时间持久; FP=瞬态",
    "proposal attn ≈ SELSA/TransVOD", "光流一致性 ≈ UltraDet 负上下文",
    "两阶段 RPN, 非单阶段 YOLO", "T/光流网未详; 代码不公开",
    "持续存在的血管横截面仍难"], C_SIDE, fs=8.8)

box(ax, 4, 4, 94, 16, [
    "关键公式速览",
    r"尺度加权: $w_l=\sigma(W g(f_i))$,  $f_i=\|_l\, w_l\,$RoIAlign$(P_l,b_i)$   (自选可信尺度, 利好小病灶)",
    r"时序 relation: $f_i^{(j+1)}$=Relation$(f_i^{(j)},Q,B)$,  Q=全窗口 proposal 特征, B=位置编码   (语义+位置跨帧互补)",
    r"光流校验: $\hat b_i^{t-k}$=warp$(b_i^t,F_{t\rightarrow t-k})$;  过去帧无对应 → 瞬态 FP 抑制   (负时序上下文)",
    r"因果辅助损失: $L_{total}=L(I_t)+\sum_{k=1}^{T}L(I_{t-k})$,  每帧仅用 $\leq$ 自身时刻信息"], C_SIDE, fs=9)

fig.savefig("docs/autodflls_architecture.png", dpi=150, bbox_inches="tight")
print("saved technical docs/autodflls_architecture.png")