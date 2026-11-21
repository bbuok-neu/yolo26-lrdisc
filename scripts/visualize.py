"""Side-by-side streaming detections (baseline vs temporal) on a test video + confidence curves."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = r"C:\Users\bbuok\udisk-code\ultralytics"
sys.path.insert(0, REPO)
sys.path.insert(0, str(PROJECT))

from thyroid_temporal.augment import letterbox  # noqa: E402
from scripts.stream_eval import load_model, group_videos, read_gt  # noqa: E402


@torch.no_grad()
def run_video(model, frames, device, imgsz=640, half=True):
    model.eval()
    if half:
        model = model.half()
    reset = getattr(model, "reset_state", None)
    if reset:
        reset()
    outs = []
    for p in frames:
        im0 = cv2.imread(str(p), cv2.IMREAD_COLOR)
        H, W = im0.shape[:2]
        im, ratio, pad = letterbox(im0, imgsz)
        x = torch.from_numpy(im.transpose(2, 0, 1)[None]).to(device)
        x = (x.half() if half else x.float()) / 255
        o = model(x)
        y = (o[0] if isinstance(o, (tuple, list)) else o)[0].float().cpu().numpy()
        y = y[y[:, 4] > 0.25]
        from ultralytics.utils.ops import scale_boxes
        if len(y):
            y[:, :4] = scale_boxes((imgsz, imgsz), torch.from_numpy(y[:, :4]).float(), (H, W), (ratio, pad)).numpy()
        outs.append((im0, y))
    return outs


def draw(im, y, color, label):
    im = im.copy()
    for x1, y1, x2, y2, s, c in y:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(im, f"{label} {s:.2f}", (int(x1), max(20, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(PROJECT / "runs/baseline_sub/weights/best.pt"))
    ap.add_argument("--temporal", default=str(PROJECT / "runs/temporal_v1/weights/best.pt"))
    ap.add_argument("--video", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=str(PROJECT / "docs/qualitative"))
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    vids = group_videos(PROJECT / "data/thyroid_yolo/images" / args.split)
    video = args.video or sorted(vids, key=lambda v: -len(vids[v]))[0]
    frames = vids[video]
    bm = load_model(args.baseline, False, device)
    tm = load_model(args.temporal, True, device)
    bo = run_video(bm, frames, device)
    to = run_video(tm, frames, device)

    idxs = np.linspace(0, len(frames) - 1, args.n).astype(int)
    rows = []
    for i in idxs:
        im0, by = bo[i]
        _, ty = to[i]
        gt, _ = read_gt(PROJECT / "data/thyroid_yolo/labels" / args.split / (frames[i].stem + ".txt"),
                        im0.shape[1], im0.shape[0])
        gtim = im0.copy()
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(gtim, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
        bvis = draw(im0, by, (0, 200, 0), "base")
        tvis = draw(im0, ty, (255, 128, 0), "tempo")
        row = np.concatenate([gtim, bvis, tvis], axis=1)
        row = cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2))
        cv2.putText(row, f"{frames[i].stem}  GT | baseline | ThyroidTempoNet", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out / f"{video}_compare.jpg"), grid)

    # confidence curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bc = [float(y[:, 4].max()) if len(y) else 0.0 for _, y in bo]
    tc = [float(y[:, 4].max()) if len(y) else 0.0 for _, y in to]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(bc, label="baseline", color="#888888")
    ax.plot(tc, label="ThyroidTempoNet", color="#2a7de1")
    ax.set_xlabel("frame"); ax.set_ylabel("max confidence"); ax.set_title(f"video {video}")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out / f"{video}_conf.png", dpi=150)
    print("saved", out / f"{video}_compare.jpg", "and conf curve")


if __name__ == "__main__":
    main()