"""Train ThyroidTempoNet (YOLO26 + causal temporal neck) with the same budget as the baseline.

Usage:
    python scripts/train_temporal.py --epochs 50 --batch 16 --name temporal_v1 --T 4 --hid 8
"""

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = r"C:\Users\bbuok\udisk-code\ultralytics"
sys.path.insert(0, REPO)
sys.path.insert(0, str(PROJECT))

from thyroid_temporal import robust  # noqa: E402
robust.install()

from thyroid_temporal.model import default_level_cfg  # noqa: E402
from thyroid_temporal.trainer import TemporalTrainer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(Path(REPO) / "pt" / "yolo26n.pt"))
    p.add_argument("--data", default=str(PROJECT / "data" / "thyroid_yolo" / "thyroid.yaml"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--name", default="temporal_v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--hid", type=int, default=8)
    p.add_argument("--history-window", type=int, default=50)
    p.add_argument("--device", default="0")
    p.add_argument("--levels", default="full", choices=["full", "p5only", "tada_only", "noema", "lite", "lrdisc"])
    p.add_argument("--strides", default="3,10,25", help="sparse far-history strides for lrdisc")
    args = p.parse_args()

    strides = tuple(int(s) for s in args.strides.split(","))
    level_cfg = default_level_cfg(args.T, args.hid)
    if args.levels == "lrdisc":
        for k in level_cfg:
            level_cfg[k] = dict(lrdisc=True, strides=strides, shift=2)
    elif args.levels == "lite":
        for k in level_cfg:
            level_cfg[k] = dict(T=args.T, hid=args.hid, lite=True)
    elif args.levels == "p5only":
        level_cfg[16] = dict(T=args.T, hid=args.hid, tada=False, gru=False, queue=False, ema=False)
        level_cfg[19] = dict(T=args.T, hid=args.hid, tada=False, gru=False, queue=False, ema=False)
    elif args.levels == "tada_only":
        for k in level_cfg:
            level_cfg[k] = dict(T=args.T, hid=args.hid, tada=True, gru=False, queue=False, ema=False)
    elif args.levels == "noema":
        level_cfg[22]["ema"] = False
    # identity-passthrough levels still need a module; disable all flags -> out conv has 0 inputs?
    # ensure at least tada so the module is well-formed; for p5only we instead skip fusion on P3/P4
    tcfg = dict(T=args.T, hid=args.hid, history_window=args.history_window, level_cfg=level_cfg,
                levels=args.levels, strides=strides if args.levels == "lrdisc" else None)

    overrides = dict(
        model=args.weights, data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        workers=args.workers, seed=args.seed, deterministic=True, project=str(PROJECT / "runs"),
        name=args.name, exist_ok=True, device=args.device, amp=True, cache=False, val=True,
        plots=False, mosaic=0.0, patience=args.epochs, optimizer="auto", cos_lr=False, close_mosaic=0,
        nms=False,  # force end2end one-to-one inference in validation (else val mAP=0)
    )
    trainer = TemporalTrainer(overrides=overrides, tcfg=tcfg)
    trainer.train()
    print("TRAIN DONE", str(PROJECT / "runs" / args.name))


if __name__ == "__main__":
    main()