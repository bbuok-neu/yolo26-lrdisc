"""PR operating-point analysis: recall achievable at precision >= target (e.g. baseline's precision).

Answers: can a temporal model beat baseline RECALL while keeping PRECISION ~ baseline?
P/R are threshold-dependent; a model with higher AP yields higher recall at the same precision.
"""
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, r"C:\Users\bbuok\udisk-code\ultralytics")
sys.path.insert(0, str(PROJECT))
from scripts.stream_eval import load_model, evaluate

MODELS = [
    ("baseline", "runs/baseline_full/weights/best.pt", False),
    ("lite", "runs/temporal_lite/weights/best.pt", True),
    ("lrdisc", "runs/temporal_lrdisc/weights/best.pt", True),
]
TARGETS = [0.70, 0.72, 0.75]


def main():
    dev = torch.device("cuda:0")
    img = PROJECT / "data/thyroid_yolo/images/test"
    lbl = PROJECT / "data/thyroid_yolo/labels/test"
    rows = {}
    for name, w, temporal in MODELS:
        m = load_model(w, temporal, dev)
        met, res = evaluate(m, img, lbl, dev, half=True, measure_fps=False, return_curves=True)
        p_curve, r_curve, x = res[7][0], res[8][0], res[10]
        rec_at = {}
        for t in TARGETS:
            mask = p_curve >= t
            rec_at[t] = float(r_curve[mask].max()) if mask.any() else 0.0
        rows[name] = {"maxF1_P": met["precision"], "maxF1_R": met["recall"],
                      "mAP50": met["mAP50"], "rec_at": rec_at}
        del m
        torch.cuda.empty_cache()
    print(f"{'model':10s} {'mAP50':>7s} {'maxF1 P/R':>14s} " + " ".join(f"R@P>={t:.2f}" for t in TARGETS))
    for name, r in rows.items():
        print(f"{name:10s} {r['mAP50']*100:7.2f} {r['maxF1_P']*100:6.2f}/{r['maxF1_R']*100:6.2f} "
              + " ".join(f"{r['rec_at'][t]*100:9.2f}" for t in TARGETS))
    b = rows["baseline"]
    print("\nRecall gain over baseline at matched precision:")
    for name in ["lite", "lrdisc"]:
        for t in TARGETS:
            d = (rows[name]["rec_at"][t] - b["rec_at"][t]) * 100
            print(f"  {name} @P>={t:.2f}: {d:+.2f} pp")
    import json
    (PROJECT / "runs/pr_analysis.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()