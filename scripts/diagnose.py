"""TemporalLens-style diagnostic: prove the model genuinely uses history frames.

Compares streaming mAP under: normal / shuffled history / frozen cache / zeroed history.
A model that truly uses temporal context drops under perturbations; a single-frame model is unaffected.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = r"C:\Users\bbuok\udisk-code\ultralytics"
sys.path.insert(0, REPO)
sys.path.insert(0, str(PROJECT))

from scripts.stream_eval import load_model, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(PROJECT / "runs/temporal_v1/weights/best.pt"))
    ap.add_argument("--temporal", action="store_true", default=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-frames", type=int, default=800)
    ap.add_argument("--out", default=str(PROJECT / "runs/diagnose_temporal.json"))
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_dir = PROJECT / "data/thyroid_yolo/images" / args.split
    label_dir = PROJECT / "data/thyroid_yolo/labels" / args.split
    results = {}
    for mode in [None, "shuffle", "freeze", "zero"]:
        model = load_model(args.weights, args.temporal, device)
        m = evaluate(model, img_dir, label_dir, device, half=True,
                     max_frames=args.max_frames, measure_fps=False, perturb=mode)
        results[mode or "normal"] = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "recall": m["recall"]}
        print(mode or "normal", json.dumps(results[mode or "normal"]))
        del model
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, indent=1))
    base = results["normal"]["mAP50"]
    print("\nmAP50 drop vs normal:")
    for k, v in results.items():
        print(f"  {k}: {(v['mAP50']-base)*100:+.2f} pp")
    print("saved", args.out)


if __name__ == "__main__":
    main()