"""Test mitigated temporal-consistency gate: confidence exemption + short window + grace period.

Variants:
  none      : no gate
  naive     : strides(3,10,25), score*(0.35+0.65*pers), no exemption/grace  (suppresses onset)
  mitigated : strides(1,2) short window; grace=first G frames no gate; conf>=EXEMPT no gate;
              else score*(0.35+0.65*pers_short)
Reports aggregate (P/R/F1/mAP50/FPS) + onset(first K frames) recall, per video reset.
"""
import sys
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, r"C:\Users\bbuok\udisk-code\ultralytics")
sys.path.insert(0, str(PROJECT))
from scripts.stream_eval import load_model, match_tp, read_gt, group_videos
from scripts.borrow_experiment import iou_mat
from thyroid_temporal.augment import letterbox
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.ops import scale_boxes

NAIVE_STRIDES = (3, 10, 25)
MIT_STRIDES = (1, 2)
GRACE = 3
EXEMPT = 0.5
K_ONSET = 5
CAP_PER_VIDEO = 30


def pers_score(boxes, hist, strides):
    pers = np.zeros(len(boxes))
    used = 0
    for s in strides:
        if len(hist) >= s:
            pb = hist[-s]
            if len(pb):
                pers += (iou_mat(boxes, pb).max(1) >= 0.3).astype(float)
                used += 1
    return pers / max(used, 1)


@torch.no_grad()
def evaluate(model, vids, label_dir, device, gate, half=True, imgsz=640):
    model.eval()
    if half:
        model = model.half()
    reset = getattr(model, "reset_state", None)
    TP, CONF, PCLS, TCLS = [], [], [], []
    onset_tp = 0.0
    onset_gt = 0
    times = []
    for v, frames in vids.items():
        if reset:
            reset()
        hist = deque(maxlen=max(NAIVE_STRIDES))
        for fi, p in enumerate(frames[:CAP_PER_VIDEO]):
            im0 = cv2.imread(str(p)); H, W = im0.shape[:2]
            im, ratio, pad = letterbox(im0, imgsz)
            x = torch.from_numpy(im.transpose(2, 0, 1)[None]).to(device)
            x = (x.half() if half else x.float()) / 255
            torch.cuda.synchronize(); t0 = time.perf_counter()
            o = model(x)
            y = (o[0] if isinstance(o, (tuple, list)) else o)[0].float().cpu().numpy()
            y = y[y[:, 4] > 0.001]
            if len(y):
                boxes = scale_boxes((imgsz, imgsz), torch.from_numpy(y[:, :4]).float(), (H, W), (ratio, pad)).numpy()
                conf, pcls = y[:, 4].copy(), y[:, 5]
            else:
                boxes = np.zeros((0, 4)); conf = np.zeros(0); pcls = np.zeros(0)
            if len(boxes):
                if gate == "naive":
                    conf = conf * (0.35 + 0.65 * pers_score(boxes, hist, NAIVE_STRIDES))
                elif gate == "mitigated":
                    exempt = (conf >= EXEMPT) | (fi < GRACE)
                    ps = pers_score(boxes, hist, MIT_STRIDES)
                    conf = np.where(exempt, conf, conf * (0.35 + 0.65 * ps))
            hist.append(boxes.copy())
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            gt, tcls = read_gt(label_dir / (p.stem + ".txt"), W, H)
            TP.append(match_tp(boxes, conf, gt)); CONF.append(conf); PCLS.append(pcls); TCLS.append(tcls)
            if fi < K_ONSET:
                onset_gt += len(gt)
                if len(gt) and len(boxes):
                    m = iou_mat(boxes, gt)
                    onset_tp += ((m >= 0.5) & (conf[:, None] >= 0.25)).any(0).sum()
    tp = np.concatenate(TP, 0); cf = np.concatenate(CONF); pc = np.concatenate(PCLS); tc = np.concatenate(TCLS)
    res = ap_per_class(tp, cf, pc, tc, names={0: "nodule"})
    out = {"mAP50": float(res[5][:, 0].mean()), "P": float(res[2].mean()), "R": float(res[3].mean()),
           "F1": float(res[4].mean()), "onsetR": onset_tp / max(onset_gt, 1),
           "FPS": 1000.0 / (float(np.mean(times)) * 1000)}
    return out


def main():
    import json
    dev = torch.device("cuda:0")
    vids = group_videos(PROJECT / "data/thyroid_yolo/images/test")
    lbl = PROJECT / "data/thyroid_yolo/labels/test"
    results = {}
    for name, w, temporal in [("baseline", "runs/baseline_full/weights/best.pt", False),
                              ("lrdisc", "runs/temporal_lrdisc/weights/best.pt", True)]:
        for gate in ["none", "naive", "mitigated"]:
            m = load_model(w, temporal, dev)
            r = evaluate(m, vids, lbl, dev, gate)
            results[f"{name}/{gate}"] = r
            print(f"{name}/{gate}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
            del m; torch.cuda.empty_cache()
    (PROJECT / "runs/mitigated_gate.json").write_text(json.dumps(results, indent=1))
    print("saved runs/mitigated_gate.json")


if __name__ == "__main__":
    main()