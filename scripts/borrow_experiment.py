"""Cheap inference-only borrowings from Auto-DFLLs: negative-temporal-context gate + vesselness gate.

No retraining. Measures precision/recall/mAP gain vs FPS cost on baseline & lrdisc.
- persist gate: detection-level temporal consistency (negative temporal context). A current detection
  matched (IoU/center within motion tolerance) in far-past cached frames is persistent -> keep;
  transient (no past correspondence) -> down-weight. Approximates Auto-DFLLs image-level consistency
  WITHOUT optical flow (uses cached detections + motion-tolerant matching).
- vessel gate: downsampled Frangi vesselness (dark tubular) -> suppress vessel-like elongated FP.
"""
import sys
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, r"C:\Users\bbuok\udisk-code\ultralytics")
sys.path.insert(0, str(PROJECT))
from scripts.stream_eval import load_model, match_tp, read_gt, group_videos
from thyroid_temporal.augment import letterbox
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.ops import scale_boxes

STRIDES = (3, 10, 25)


def vesselness_map(gray, pool=4):
    """Downsampled multi-scale Frangi vesselness for DARK tubular structures, upsampled to full res."""
    small = cv2.resize(gray, (gray.shape[1] // pool, gray.shape[0] // pool), interpolation=cv2.INTER_AREA)
    img = small.astype(np.float32) / 255.0
    best = np.zeros_like(img)
    for sigma in (1.0, 2.0, 3.0):
        s = ndimage.gaussian_filter(img, sigma)
        Iyy, Ixx = np.gradient(np.gradient(s, axis=0), axis=0)
        Ixy = np.gradient(np.gradient(s, axis=0), axis=1)
        Iyy = ndimage.gaussian_filter(Iyy, sigma)
        Ixx = ndimage.gaussian_filter(Ixx, sigma)
        Ixy = ndimage.gaussian_filter(Ixy, sigma)
        tmp = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2)
        l1 = 0.5 * (Ixx + Iyy + tmp)
        l2 = 0.5 * (Ixx + Iyy - tmp)
        # order by |.|: la small, lb large
        la = np.where(np.abs(l1) <= np.abs(l2), l1, l2)
        lb = np.where(np.abs(l1) <= np.abs(l2), l2, l1)
        Rb = np.abs(la) / (np.abs(lb) + 1e-6)
        S = np.sqrt(la ** 2 + lb ** 2)
        v = np.exp(-Rb ** 2 / (2 * 0.5 ** 2)) * (1 - np.exp(-S ** 2 / (2 * 0.15 ** 2)))
        v = np.where(lb > 0, v, 0.0)  # dark ridge (anechoic tube)
        best = np.maximum(best, v)
    best = best / (best.max() + 1e-6)
    return cv2.resize(best, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def box_mean(m, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(m.shape[1], x2), min(m.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(m[y1:y2, x1:x2].mean())


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-6)


@torch.no_grad()
def run(model, frames, label_dir, device, gate, half=True, imgsz=640):
    model.eval()
    if half:
        model = model.half()
    reset = getattr(model, "reset_state", None)
    hist = deque(maxlen=max(STRIDES))
    TP, CONF, PCLS, TCLS = [], [], [], []
    ngt = 0
    times = []
    for fi, p in enumerate(frames):
        im0 = cv2.imread(str(p)); H, W = im0.shape[:2]
        gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        im, ratio, pad = letterbox(im0, imgsz)
        x = torch.from_numpy(im.transpose(2, 0, 1)[None]).to(device)
        x = (x.half() if half else x.float()) / 255
        if fi >= 10:
            torch.cuda.synchronize(); t0 = __import__("time").perf_counter()
        o = model(x)
        y = (o[0] if isinstance(o, (tuple, list)) else o)[0].float().cpu().numpy()
        y = y[y[:, 4] > 0.001]
        if len(y):
            boxes = scale_boxes((imgsz, imgsz), torch.from_numpy(y[:, :4]).float(), (H, W), (ratio, pad)).numpy()
            conf, pcls = y[:, 4].copy(), y[:, 5]
        else:
            boxes = np.zeros((0, 4)); conf = np.zeros(0); pcls = np.zeros(0)
        # gates (inference-only)
        if gate in ("persist", "both") and len(boxes):
            pers = np.zeros(len(boxes))
            for s in STRIDES:
                if len(hist) >= s:
                    pb = hist[-s]
                    if len(pb):
                        m = iou_mat(boxes, pb)
                        pers += (m.max(1) >= 0.3).astype(float)
            pers /= len(STRIDES)
            conf = conf * (0.35 + 0.65 * pers)  # transient (pers=0) heavily down-weighted
        if gate in ("vessel", "both") and len(boxes):
            vm = vesselness_map(gray)
            v = np.array([box_mean(vm, b) for b in boxes])
            conf = conf * (1 - 0.6 * v)
        hist.append(boxes.copy())
        if fi >= 10:
            torch.cuda.synchronize(); times.append(__import__("time").perf_counter() - t0)
        gt, tcls = read_gt(label_dir / (p.stem + ".txt"), W, H)
        ngt += len(gt)
        TP.append(match_tp(boxes, conf, gt)); CONF.append(conf); PCLS.append(pcls); TCLS.append(tcls)
    tp = np.concatenate(TP, 0); cf = np.concatenate(CONF); pc = np.concatenate(PCLS); tc = np.concatenate(TCLS)
    res = ap_per_class(tp, cf, pc, tc, names={0: "nodule"})
    out = {"mAP50": float(res[5][:, 0].mean()), "mAP50_95": float(res[5].mean()),
           "P": float(res[2].mean()), "R": float(res[3].mean()), "F1": float(res[4].mean())}
    if times:
        out["FPS"] = 1000.0 / (float(np.mean(times)) * 1000)
    return out


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--out", default=str(PROJECT / "runs/borrow_experiment.json"))
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    vids = group_videos(PROJECT / "data/thyroid_yolo/images/test")
    frames = []
    for v, fs in vids.items():
        frames.extend(fs)
        if len(frames) >= args.max_frames:
            break
    frames = frames[:args.max_frames]
    lbl = PROJECT / "data/thyroid_yolo/labels/test"
    results = {}
    for name, w, temporal in [("baseline", "runs/baseline_full/weights/best.pt", False),
                              ("lrdisc", "runs/temporal_lrdisc/weights/best.pt", True)]:
        gates = ["none", "persist"]  # Auto-DFLLs-only: image-level temporal consistency / negative context
        for g in gates:
            m = load_model(w, temporal, dev)
            r = run(m, frames, lbl, dev, g)
            results[f"{name}/{g}"] = r
            print(f"{name}/{g}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
            del m; torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()