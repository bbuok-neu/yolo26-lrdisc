"""Zero-shot detection probe: does denoising ThyroidXL stills improve (or hurt) detection by a
video-trained model? Uses the authoritative streaming harness on test_xl and its denoised
variants. No FPS measurement (GPU may be shared with a training run).

Usage: python scripts/probe_denoise.py --weights runs/baseline_full/weights/best.pt [--max-frames 2094]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = r"C:\Users\bbuok\udisk-code\ultralytics"
sys.path.insert(0, REPO)
sys.path.insert(0, str(PROJECT))

import torch  # noqa: E402

from scripts.stream_eval import evaluate, load_model  # noqa: E402

BASE = PROJECT / "data" / "thyroid_yolo"
VARIANTS = ["test_xl", "test_xl_gauss1", "test_xl_bilateral", "test_xl_nlm8", "test_xl_nlm14", "test_xl_lee_fit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(PROJECT / "runs" / "baseline_full" / "weights" / "best.pt"))
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, args.temporal, device)
    results = {}
    for v in VARIANTS:
        img_dir = BASE / "images" / v
        if not img_dir.exists():
            print(f"skip {v} (missing)")
            continue
        m = evaluate(model, img_dir, BASE / "labels" / v, device, half=True,
                     measure_fps=False, max_frames=args.max_frames)
        results[v] = m
        print(v, json.dumps({k: round(m[k], 4) for k in ("mAP50", "mAP50_95", "precision", "recall", "f1")}))
    out = PROJECT / "docs" / "xl_noise" / f"probe_{args.tag}.json"
    out.write_text(json.dumps(results, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()