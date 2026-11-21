"""Shared-parameter clip augmentation (same geometric/photometric transform for all frames in a clip).

Mimics ultralytics RandomPerspective + HSV distributions but WITHOUT mosaic/mixup, which would break
temporal coherence. History frames get the identical transform so motion stays consistent.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np


def letterbox(im, new=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (new - nw) / 2, (new - nh) / 2
    if (w, h) != (nw, nh):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


def augment_hsv(im, hgain=0.015, sgain=0.7, vgain=0.4):
    if im.dtype != np.uint8:
        return
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)


def _perspective_matrix(new, translate=0.1, scale=0.5):
    s = random.uniform(1 - scale, 1 + scale)
    tx = random.uniform(-translate, translate) * new
    ty = random.uniform(-translate, translate) * new
    cx = cy = new / 2
    M = np.eye(3, dtype=np.float32)
    M[0, 0] = M[1, 1] = s
    M[0, 2] = tx + (1 - s) * cx
    M[1, 2] = ty + (1 - s) * cy
    return M


def _apply_boxes(boxes_xyxy, M, new, flip):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, np.zeros(0, dtype=bool)
    n = len(boxes_xyxy)
    x0, y0, x1, y1 = boxes_xyxy.T
    corners = np.stack(
        [np.stack([x0, y0], 1), np.stack([x0, y1], 1),
         np.stack([x1, y0], 1), np.stack([x1, y1], 1)], axis=1
    ).reshape(n * 4, 2)
    hom = np.concatenate([corners, np.ones((n * 4, 1), dtype=np.float32)], axis=1)
    trans = (M @ hom.T).T[:, :2].reshape(n, 4, 2)
    xyxy = np.concatenate([trans.min(1), trans.max(1)], axis=1)
    if flip:
        xyxy[:, [0, 2]] = new - xyxy[:, [2, 0]]
    xyxy = np.clip(xyxy, 0, new)
    wh = xyxy[:, 2:] - xyxy[:, :2]
    area = wh[:, 0] * wh[:, 1]
    keep = (wh[:, 0] > 2) & (wh[:, 1] > 2) & (area > 4)
    return xyxy[keep], keep


def augment_clip(imgs, boxes_xywh_norm, new=640):
    """imgs: list of T BGR images (native res). boxes_xywh_norm: (n,5) cls+xywh normalized for target.
    Returns augmented imgs (list), target boxes (n,5) cls+xywh normalized in `new` space, ratio_pad."""
    lb = [letterbox(im, new) for im in imgs]
    imgs = [x[0] for x in lb]
    ratio_pad = (lb[0][1], lb[0][2])
    M = _perspective_matrix(new)
    flip = random.random() < 0.5
    imgs = [cv2.warpAffine(im, M[:2], (new, new), borderValue=(114, 114, 114)) for im in imgs]
    if flip:
        imgs = [im[:, ::-1].copy() for im in imgs]
    if len(boxes_xywh_norm):
        cls = boxes_xywh_norm[:, 0:1]
        xywh = boxes_xywh_norm[:, 1:5].copy()
        dw, dh = ratio_pad[1]
        xc = xywh[:, 0] * (new - 2 * dw) + dw
        yc = xywh[:, 1] * (new - 2 * dh) + dh
        bw = xywh[:, 2] * (new - 2 * dw)
        bh = xywh[:, 3] * (new - 2 * dh)
        xyxy = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)
        xyxy, keep = _apply_boxes(xyxy, M, new, flip)
        cls = cls[keep]
        xc = (xyxy[:, 0] + xyxy[:, 2]) / 2 / new
        yc = (xyxy[:, 1] + xyxy[:, 3]) / 2 / new
        bw = (xyxy[:, 2] - xyxy[:, 0]) / new
        bh = (xyxy[:, 3] - xyxy[:, 1]) / new
        boxes = np.concatenate([cls, np.stack([xc, yc, bw, bh], 1)], 1).astype(np.float32)
    else:
        boxes = np.zeros((0, 5), dtype=np.float32)
    for im in imgs:
        augment_hsv(im)
    return imgs, boxes, ratio_pad


def letterbox_clip(imgs, boxes_xywh_norm, new=640):
    lb = [letterbox(im, new) for im in imgs]
    imgs = [x[0] for x in lb]
    ratio_pad = (lb[0][1], lb[0][2])
    if len(boxes_xywh_norm):
        cls = boxes_xywh_norm[:, 0:1]
        xywh = boxes_xywh_norm[:, 1:5]
        r = ratio_pad[0][0]
        dw, dh = ratio_pad[1]
        xc = (xywh[:, 0] * (new - 2 * dw) + dw) / new
        yc = (xywh[:, 1] * (new - 2 * dh) + dh) / new
        bw = xywh[:, 2] * (new - 2 * dw) / new
        bh = xywh[:, 3] * (new - 2 * dh) / new
        boxes = np.concatenate([cls, np.stack([xc, yc, bw, bh], 1)], 1).astype(np.float32)
    else:
        boxes = np.zeros((0, 5), dtype=np.float32)
    return imgs, boxes, ratio_pad