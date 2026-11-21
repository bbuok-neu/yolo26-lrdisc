"""Streaming (causal, video-ordered) evaluation — the authoritative comparison protocol.

Processes each video frame-by-frame with a rolling feature cache (temporal model) or statelessly
(baseline), scales boxes back to native resolution, and computes COCO-style mAP with the SAME
ultralytics ap_per_class used by the standard validator. Also reports FPS and FP@recall0.90.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = r"C:\Users\bbuok\udisk-code\ultralytics"
sys.path.insert(0, REPO)
sys.path.insert(0, str(PROJECT))

from ultralytics.utils.metrics import ap_per_class, bbox_iou  # noqa: E402
from ultralytics.utils.ops import scale_boxes  # noqa: E402

from thyroid_temporal.augment import letterbox  # noqa: E402

IOU_THRS = np.linspace(0.5, 0.95, 10)


def load_model(weights, temporal, device):
    if temporal:
        import thyroid_temporal  # noqa: F401  (register classes for unpickling)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        model = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        model = model.float().to(device).eval()
    else:
        from ultralytics import YOLO
        model = YOLO(weights).model.to(device).eval()
    if hasattr(model, "end2end"):
        model.end2end = True
    return model


def match_tp(pred, conf, gt, iou_thrs=IOU_THRS):
    N = len(pred)
    tp = np.zeros((N, len(iou_thrs)), dtype=bool)
    M = len(gt)
    if N == 0 or M == 0:
        return tp
    iou = bbox_iou(torch.from_numpy(pred).float()[:, None], torch.from_numpy(gt).float()[None],
                   xywh=False).numpy().reshape(N, M)  # (N,M) pairwise
    order = np.argsort(-conf)
    for ti, thr in enumerate(iou_thrs):
        used = np.zeros(M, dtype=bool)
        for pi in order:
            cand = np.where((iou[pi] >= thr) & ~used)[0]
            if len(cand):
                best = cand[np.argmax(iou[pi, cand])]
                used[best] = True
                tp[pi, ti] = True
    return tp


def group_videos(img_dir):
    vids = {}
    for p in sorted(Path(img_dir).glob("*.png")):
        vids.setdefault(p.stem.split("_")[0], []).append(p)
    for v in vids:
        vids[v].sort(key=lambda p: int(p.stem.split("_")[1]))
    return vids


def read_gt(label_path, W, H):
    if not label_path.exists():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    arr = np.loadtxt(label_path, ndmin=2, dtype=np.float32).reshape(-1, 5)
    if arr.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    xc, yc, bw, bh = arr[:, 1] * W, arr[:, 2] * H, arr[:, 3] * W, arr[:, 4] * H
    xyxy = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1)
    return xyxy, arr[:, 0]


@torch.no_grad()
def evaluate(model, img_dir, label_dir, device, imgsz=640, half=True, conf_thres=0.001,
             max_frames=None, measure_fps=True, perturb=None, return_curves=False):
    model.eval()
    if half:
        model = model.half()
    for m in getattr(model, "temporal_layers", []):
        m.perturb = perturb
    vids = group_videos(img_dir)
    all_tp, all_conf, all_pcls, all_tcls = [], [], [], []
    n_gt_total = 0
    times = []
    fp_at_recall = {}
    reset = getattr(model, "reset_state", None)
    done = 0
    for v, frames in vids.items():
        if reset:
            reset()
        for fi, p in enumerate(frames):
            if max_frames and done >= max_frames:
                break
            im0 = cv2.imread(str(p), cv2.IMREAD_COLOR)
            H, W = im0.shape[:2]
            im, ratio, pad = letterbox(im0, imgsz)
            x = torch.from_numpy(im.transpose(2, 0, 1)[None]).to(device)
            x = (x.half() if half else x.float()) / 255
            if measure_fps and done >= 10:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            out = model(x)
            if measure_fps and done >= 10:
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            y = out[0] if isinstance(out, (tuple, list)) else out
            y = y[0].float().cpu().numpy()  # (300,6) xyxy(640) score cls
            keep = y[:, 4] > conf_thres
            y = y[keep]
            if len(y):
                boxes = scale_boxes((imgsz, imgsz), torch.from_numpy(y[:, :4]).float(), (H, W), (ratio, pad)).numpy()
                conf = y[:, 4]
                pcls = y[:, 5]
            else:
                boxes = np.zeros((0, 4), np.float32); conf = np.zeros(0); pcls = np.zeros(0)
            gt, tcls = read_gt(label_dir / (p.stem + ".txt"), W, H)
            n_gt_total += len(gt)
            tp = match_tp(boxes, conf, gt)
            all_tp.append(tp); all_conf.append(conf); all_pcls.append(pcls); all_tcls.append(tcls)
            done += 1
        if max_frames and done >= max_frames:
            break

    tp = np.concatenate(all_tp, 0) if all_tp else np.zeros((0, 10))
    conf = np.concatenate(all_conf, 0)
    pcls = np.concatenate(all_pcls, 0)
    tcls = np.concatenate(all_tcls, 0)
    res = ap_per_class(tp, conf, pcls, tcls, plot=False, names={0: "nodule"})
    p, r, f1, ap = res[2], res[3], res[4], res[5]
    mAP50 = float(ap[:, 0].mean()) if ap.size else 0.0
    mAP5095 = float(ap.mean()) if ap.size else 0.0
    # FP per image at recall>=0.90 (IoU0.5), using conf sorted
    metrics = {
        "mAP50": mAP50, "mAP50_95": mAP5095,
        "precision": float(p.mean()) if p.size else 0.0,
        "recall": float(r.mean()) if r.size else 0.0,
        "f1": float(f1.mean()) if f1.size else 0.0,
        "n_frames": done, "n_gt": int(n_gt_total), "n_pred": int(len(conf)),
    }
    if len(conf):
        order = np.argsort(-conf)
        tp0 = tp[order, 0]
        tpc = tp0.cumsum(0)
        fpc = (~tp0).cumsum(0)
        rec = tpc / max(n_gt_total, 1)
        idx = np.where(rec >= 0.90)[0]
        if len(idx):
            k = idx[0]
            metrics["FP_per_image_at_recall90"] = float(fpc[k] / done)
            metrics["conf_at_recall90"] = float(conf[order][k])
    if times:
        ms = float(np.mean(times) * 1000)
        metrics["inference_ms"] = ms
        metrics["FPS"] = 1000.0 / ms
    return (metrics, res) if return_curves else metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=str(PROJECT / "data" / "thyroid_yolo"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, args.temporal, device)
    img_dir = Path(args.data) / "images" / args.split
    label_dir = Path(args.data) / "labels" / args.split
    m = evaluate(model, img_dir, label_dir, device, args.imgsz, half=not args.no_half,
                 max_frames=args.max_frames)
    print(json.dumps(m, indent=1))
    out = Path(args.out) if args.out else PROJECT / "runs" / f"stream_{args.split}_{Path(args.weights).parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(m, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()