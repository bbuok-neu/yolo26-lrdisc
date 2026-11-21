"""Draw YOLO26n vs ThyroidTempoNet-lite architecture comparison + lite module detail."""
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

C_BACK = "#dbeafe"; C_NECK = "#e0e7ff"; C_DET = "#fef3c7"; C_LITE = "#dcfce7"
C_HIST = "#fce7f3"; C_EDGE = "#334155"; C_HL = "#16a34a"


def box(ax, x, y, w, h, text, fc, ec=C_EDGE, fs=10, bold=False, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#0f172a", linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=C_EDGE, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))


fig = plt.figure(figsize=(15, 11))
fig.suptitle("YOLO26n 原版  vs  ThyroidTempoNet-lite（改动仅在颈部输出与 Detect 之间）",
             fontsize=16, fontweight="bold", y=0.985)

# ---- Panel A: original ----
axA = fig.add_axes([0.03, 0.70, 0.94, 0.22]); axA.set_xlim(0, 100); axA.set_ylim(0, 22); axA.axis("off")
axA.text(0, 21, "① 原版 YOLO26n（单帧，逐帧独立检测）", fontsize=12, fontweight="bold", color="#1e3a8a")
box(axA, 1, 7, 12, 6, "输入帧\n(3×640×640)", C_BACK)
box(axA, 17, 7, 16, 6, "Backbone\n层 0–10\nConv/C3k2/SPPF/C2PSA", C_BACK)
box(axA, 37, 7, 16, 6, "PAN Neck\n层 11–22", C_NECK)
box(axA, 58, 13, 12, 4.5, "P3 层16 (64ch)", C_NECK, fs=9)
box(axA, 58, 7.5, 12, 4.5, "P4 层19 (128ch)", C_NECK, fs=9)
box(axA, 58, 2, 12, 4.5, "P5 层22 (256ch)", C_NECK, fs=9)
box(axA, 76, 7, 13, 6, "Detect 层23\nNMS-free\none2one", C_DET, bold=True)
box(axA, 92, 7, 7, 6, "检测框\n(300,6)", "#f1f5f9", fs=9)
arrow(axA, 13, 10, 17, 10); arrow(axA, 33, 10, 37, 10)
arrow(axA, 53, 11, 58, 15); arrow(axA, 53, 10, 58, 9.7); arrow(axA, 53, 9, 58, 4.2)
arrow(axA, 70, 15, 76, 11); arrow(axA, 70, 9.7, 76, 10); arrow(axA, 70, 4.2, 76, 9)
arrow(axA, 89, 10, 92, 10)

# ---- Panel B: lite ----
axB = fig.add_axes([0.03, 0.40, 0.94, 0.26]); axB.set_xlim(0, 100); axB.set_ylim(0, 26); axB.axis("off")
axB.text(0, 25, "② ThyroidTempoNet-lite（因果时序，仅插入 3 个绿色模块，其余完全相同）",
         fontsize=12, fontweight="bold", color=C_HL)
box(axB, 1, 9, 12, 6, "视频帧 t\n(+历史 t-1..t-3)", C_BACK)
box(axB, 17, 9, 16, 6, "Backbone\n层 0–10\n【不变】", C_BACK)
box(axB, 37, 9, 16, 6, "PAN Neck\n层 11–22\n【不变】", C_NECK)
box(axB, 57, 16, 11, 4.2, "P3 层16", C_NECK, fs=9)
box(axB, 57, 10.5, 11, 4.2, "P4 层19", C_NECK, fs=9)
box(axB, 57, 5, 11, 4.2, "P5 层22", C_NECK, fs=9)
box(axB, 71, 16, 12, 4.2, "★Lite融合\n层23", C_LITE, ec=C_HL, fs=9, bold=True, lw=2)
box(axB, 71, 10.5, 12, 4.2, "★Lite融合\n层24", C_LITE, ec=C_HL, fs=9, bold=True, lw=2)
box(axB, 71, 5, 12, 4.2, "★Lite融合\n层25", C_LITE, ec=C_HL, fs=9, bold=True, lw=2)
box(axB, 86, 9, 12, 6, "Detect 层26\n【不变】", C_DET, bold=True)
arrow(axB, 13, 12, 17, 12); arrow(axB, 33, 12, 37, 12)
arrow(axB, 53, 13, 57, 18); arrow(axB, 53, 12, 57, 12.6); arrow(axB, 53, 11, 57, 7.1)
arrow(axB, 68, 18, 71, 18); arrow(axB, 68, 12.6, 71, 12.6); arrow(axB, 68, 7.1, 71, 7.1)
arrow(axB, 83, 18, 86, 13); arrow(axB, 83, 12.6, 86, 12); arrow(axB, 83, 7.1, 86, 11)
# history cache
box(axB, 71, 0.3, 12, 3.2, "滚动特征缓存\ndeque(T-1)", C_HIST, fs=8)
arrow(axB, 77, 3.5, 77, 5, color="#db2777", lw=1.3)
axB.text(84.5, 1.8, "历史颈部特征\n(不重算backbone)", fontsize=8, color="#db2777", va="center")

# ---- Panel C: lite module detail ----
axC = fig.add_axes([0.03, 0.04, 0.94, 0.32]); axC.set_xlim(0, 100); axC.set_ylim(0, 30); axC.axis("off")
axC.text(0, 29, "③ Lite 时序融合模块内部（每个颈部 level 一个，严格因果，零初始化残差）",
         fontsize=12, fontweight="bold", color=C_HL)
box(axC, 1, 18, 16, 7, "历史颈部特征\n[t-3, t-2, t-1]\n(缓存/eval-no_grad)", C_HIST, fs=9)
box(axC, 1, 8, 16, 6, "当前帧颈部特征\nx_t (带梯度)", C_NECK, fs=9)
box(axC, 24, 12, 16, 8, "拼接时间窗\nwin=[hist, x_t]\n(B,T,C,H,W)", "#f1f5f9", fs=9)
box(axC, 46, 12, 16, 8, "因果时序均值\nmean = win.mean(T)\n(去噪，只用≤t)", C_LITE, ec=C_HL, fs=9, bold=True, lw=2)
box(axC, 68, 20, 20, 6, "残差 Conv1×1\nConvBNAct([mean, x_t])\nconv.weight 零初始化", C_LITE, fs=8.5)
box(axC, 68, 9, 20, 6, "逐元素相加\nout = mean + 残差", "#fef9c3", fs=9, bold=True)
box(axC, 92, 9, 7, 6, "融合\n特征\n→Detect", C_DET, fs=8.5)
arrow(axC, 17, 21.5, 24, 17); arrow(axC, 17, 11, 24, 14)
arrow(axC, 40, 16, 46, 16)
arrow(axC, 62, 18, 68, 22); arrow(axC, 62, 14, 68, 12)
arrow(axC, 78, 20, 78, 15, color=C_HL)
arrow(axC, 62, 16, 68, 12, color="#94a3b8", lw=1.2)
arrow(axC, 88, 12, 92, 12)
axC.text(46, 4.5, "初始：残差=0 → out=mean（即时序均值去噪，已优于单帧基线）；训练学习小残差修正。",
         fontsize=9.5, color="#334155")
axC.text(46, 1.5, "新增参数仅 ~0.2M（+8%）：P3=8.3k / P4=33k / P5=131.6k。Backbone/Neck/Detect/损失全部不变。",
         fontsize=9.5, color=C_HL, fontweight="bold")

fig.savefig("docs/architecture_lite.png", dpi=150, bbox_inches="tight")
print("saved docs/architecture_lite.png")