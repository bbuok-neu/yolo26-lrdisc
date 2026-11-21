"""Aggregate baseline + temporal metrics into docs/RESULTS.md and a comparison plot."""

import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "runs"


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def best_val_from_csv(run_dir):
    csvp = Path(run_dir) / "results.csv"
    if not csvp.exists():
        return None
    rows = list(csv.DictReader(open(csvp)))
    if not rows:
        return None
    key = next((k for k in rows[0] if "mAP50(B)" in k or k.strip() == "metrics/mAP50(B)"), None)
    if key is None:
        key = next((k for k in rows[0] if "mAP50" in k and "95" not in k), None)
    if key is None:
        return None
    best = max(rows, key=lambda r: float(r[key] or 0))
    k95 = next((k for k in rows[0] if "mAP50-95" in k), None)
    return {"epoch": best.get("epoch"), "mAP50": float(best[key] or 0),
            "mAP50_95": float(best[k95] or 0) if k95 else None}


def fmt(x, pct=True):
    if x is None:
        return "–"
    return f"{x*100:.2f}" if pct else f"{x:.2f}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="baseline_full")
    ap.add_argument("--temp", default="temporal_full")
    ap.add_argument("--suffix", default="full")
    args = ap.parse_args()
    base_final = load_json(RUNS / args.base / "final_metrics.json") or {}
    base_stream = load_json(RUNS / f"stream_test_{args.base}.json")
    temp_stream = load_json(RUNS / f"stream_test_{args.temp}.json")
    temp_val = best_val_from_csv(RUNS / args.temp)
    base_val = best_val_from_csv(RUNS / args.base)

    lines = ["# 结果汇总（自动生成）\n", "## 训练期验证（val_sub, 539 帧）\n",
             "| 模型 | best mAP50 | best mAP50-95 |", "|---|---|---|"]
    lines.append(f"| YOLO26n 基线 | {fmt(base_val['mAP50'] if base_val else None)} | {fmt(base_val['mAP50_95'] if base_val else None)} |")
    lines.append(f"| ThyroidTempoNet | {fmt(temp_val['mAP50'] if temp_val else None)} | {fmt(temp_val['mAP50_95'] if temp_val else None)} |")

    lines += ["\n## 流式评测（完整 test, 2610 帧, 按视频因果顺序）\n",
              "| 模型 | mAP50 | mAP50-95 | Precision | Recall | F1 | FP/img@R90 | FPS |",
              "|---|---|---|---|---|---|---|---|"]
    for name, m in [("YOLO26n 基线", base_stream), ("ThyroidTempoNet", temp_stream)]:
        if m:
            lines.append(f"| {name} | {fmt(m['mAP50'])} | {fmt(m['mAP50_95'])} | {fmt(m['precision'])} | "
                         f"{fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m.get('FP_per_image_at_recall90'), False)} | "
                         f"{fmt(m.get('FPS'), False)} |")
        else:
            lines.append(f"| {name} | – | – | – | – | – | – | – |")

    if base_stream and temp_stream:
        d50 = (temp_stream['mAP50'] - base_stream['mAP50']) * 100
        d95 = (temp_stream['mAP50_95'] - base_stream['mAP50_95']) * 100
        lines.append(f"\n**流式 mAP50 提升：{d50:+.2f} pp；mAP50-95 提升：{d95:+.2f} pp。**")

    (PROJECT / "docs" / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        if base_stream and temp_stream:
            metrics = ["mAP50", "mAP50_95", "recall", "f1"]
            b = [base_stream.get(k, 0) * 100 for k in metrics]
            t = [temp_stream.get(k, 0) * 100 for k in metrics]
            x = np.arange(len(metrics)); w = 0.35
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x - w/2, b, w, label="YOLO26n baseline", color="#888888")
            ax.bar(x + w/2, t, w, label="ThyroidTempoNet", color="#2a7de1")
            ax.set_xticks(x); ax.set_xticklabels(["mAP50", "mAP50-95", "Recall", "F1"])
            ax.set_ylabel("%"); ax.set_title("Streaming test (full, causal video order)")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            for i, (bv, tv) in enumerate(zip(b, t)):
                ax.text(i - w/2, bv + 0.5, f"{bv:.1f}", ha="center", fontsize=8)
                ax.text(i + w/2, tv + 0.5, f"{tv:.1f}", ha="center", fontsize=8)
            fig.tight_layout(); fig.savefig(PROJECT / "docs" / "comparison.png", dpi=150)
            print("saved docs/comparison.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()