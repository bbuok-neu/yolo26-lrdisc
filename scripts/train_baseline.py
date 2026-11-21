"""Train baseline YOLO26 (local repo) on the thyroid YOLO dataset.

Usage:
    python scripts/train_baseline.py --epochs 50 --batch 16 --name baseline_sub
    python scripts/train_baseline.py --resume runs/baseline_sub/weights/last.pt
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = r"C:\Users\bbuok\udisk-code\ultralytics"
sys.path.insert(0, REPO)
sys.path.insert(0, str(PROJECT))

from thyroid_temporal import robust  # noqa: E402
robust.install()

from ultralytics import YOLO  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(Path(REPO) / "pt" / "yolo26n.pt"))
    p.add_argument("--data", default=str(PROJECT / "data" / "thyroid_yolo" / "thyroid_fast.yaml"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--name", default="baseline_sub")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", default="0")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True, workers=args.workers)
        run_dir = Path(args.resume).parent.parent
    else:
        model = YOLO(args.weights)
        model.train(
            data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
            workers=args.workers, seed=args.seed, deterministic=True,
            project=str(PROJECT / "runs"), name=args.name, exist_ok=True, device=args.device,
            amp=True, cache=False, val=True, plots=True, mosaic=args.mosaic,
            patience=args.patience, optimizer="auto", cos_lr=False, close_mosaic=10,
        )
        run_dir = PROJECT / "runs" / args.name

    out = {}
    for split in ("val", "test"):
        try:
            m = model.val(data=args.data, split=split, imgsz=args.imgsz, batch=args.batch,
                          device=args.device, workers=2, plots=False,
                          project=str(PROJECT / "runs"), name=f"{args.name}_{split}", exist_ok=True)
            out[split] = {
                "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                "precision": float(m.box.mp), "recall": float(m.box.mr),
                "f1": float(m.box.f1[0]) if len(m.box.f1) else None,
            }
        except Exception as e:
            out[split] = {"error": str(e)}
        with open(run_dir / "final_metrics.json", "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()