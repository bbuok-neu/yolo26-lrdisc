"""Onset (cold-start) analysis: does the temporal-consistency gate suppress newly-appearing nodules?

Evaluates recall on the FIRST K frames of each video (cache empty / nodule just appeared),
with vs without the persist gate, for baseline & lrdisc.
"""
import sys
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, r"C:\Users\bbuok\udisk-code\ultralytics")
sys.path.insert(0, str(PROJECT))
from scripts.stream_eval import load_model, match_tp, read_gt, group_videos
from scripts.borrow_experiment import iou_mat, STRIDES
from thyroid_temporal.augment import letterbox
from ultralytics.utils.ops import scale_boxes

K = 5


@torch.no_grad()
def onset_recall(model, vids, label_dir, device, gate, half=True, imgsz=640, k=K):
    model.eval()
    if half:
        model = model.half()
    reset = getattr(model, "reset_state", None)
    tp_sum = 0.0
    gt_sum = 0
    for v, frames in vids.items():
        if reset:
            reset()
        hist = deque(maxlen=max(STRIDES))
        for p in frames[:k]:
            im0 = cv2.imread(str(p)); H, W = im0.shape[:2]
            im, ratio, pad = letterbox(im0, imgsz)
            x = torch.from_numpy(im.transpose(2, 0, 1)[None]).to(device)
            x = (x.half() if half else x.float()) / 255
            o = model(x)
            y = (o[0] if isinstance(o, (tuple, list)) else o)[0].float().cpu().numpy()
            y = y[y[:, 4] > 0.001]
            if len(y):
                boxes = scale_boxes((imgsz, imgsz), torch.from_numpy(y[:, :4]).float(), (H, W), (ratio, pad)).numpy()
                conf = y[:, 4].copy()
            else:
                boxes = np.zeros((0, 4)); conf = np.zeros(0)
            if gate == "persist" and len(boxes):
                pers = np.zeros(len(boxes))
                for s in STRIDES:
                    if len(hist) >= s:
                        pb = hist[-s]
                        if len(pb):
                            pers += (iou_mat(boxes, pb).max(1) >= 0.3).astype(float)
                pers /= len(STRIDES)
                conf = conf * (0.35 + 0.65 * pers)
            hist.append(boxes.copy())
            gt, tcls = read_gt(label_dir / (p.stem + ".txt"), W, H)
            gt_sum += len(gt)
            if len(gt):
                tp = match_tp(boxes, conf, gt)
                # count a gt matched if any pred with conf>=0.25 matches (operating threshold)
                m = iou_mat(boxes, gt)
                matched = ((m >= 0.5) & (conf[:, None] >= 0.25)).any(0)
                tp_sum += matched.sum()
    return tp_sum / max(gt_sum, 1), gt_sum


def main():
    dev = torch.device("cuda:0")
    vids = group_videos(PROJECT / "data/thyroid_yolo/images/test")
    lbl = PROJECT / "data/thyroid_yolo/labels/test"
    for name, w, temporal in [("baseline", "runs/baseline_full/weights/best.pt", False),
                              ("lrdisc", "runs/temporal_lrdisc/weights/best.pt", True)]:
        for gate in ["none", "persist"]:
            m = load_model(w, temporal, dev)
            r, n = onset_recall(m, vids, lbl, dev, gate)
            print(f"{name}/{gate}: onset(first {K} frames) recall={r:.3f} (gt={n})", flush=True)
            del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()