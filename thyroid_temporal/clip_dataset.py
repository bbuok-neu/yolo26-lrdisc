"""Clip dataset: groups consecutive video frames, supervises only the current (last) frame.

Train/val split is at video level (done in conversion). History sampling follows TSdetector: 3 past
frames randomly drawn from up to `history_window` frames back, the most recent forced to t-1.
Validation uses deterministic consecutive history.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.dataset import YOLODataset

from .augment import augment_clip, letterbox_clip


class ClipDataset(Dataset):
    def __init__(self, img_dir, data=None, imgsz=640, T=4, augment=False,
                 history_window=50, deterministic_hist=False, stride=32, strides=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(str(self.img_dir).replace("images", "labels"))
        self.imgsz, self.T = imgsz, T
        self.augment = augment
        self.history_window = history_window
        self.deterministic_hist = deterministic_hist
        self.stride = stride
        self.strides = tuple(strides) if strides else None
        if self.strides:
            self.T = len(self.strides) + 1

        files = sorted(self.img_dir.glob("*.png"))
        self.videos = {}
        for p in files:
            video = p.stem.split("_")[0]
            self.videos.setdefault(video, []).append(p)
        for v in self.videos:
            self.videos[v].sort(key=lambda p: int(p.stem.split("_")[1]))
        self.samples = [(v, j) for v, frames in self.videos.items() for j in range(len(frames))]

        self.labels = {}
        for p in files:
            lp = self.label_dir / (p.stem + ".txt")
            if lp.exists():
                arr = np.loadtxt(lp, ndmin=2, dtype=np.float32)
                self.labels[p] = arr.reshape(-1, 5) if arr.size else np.zeros((0, 5), np.float32)
            else:
                self.labels[p] = np.zeros((0, 5), np.float32)

    def __len__(self):
        return len(self.samples)

    def _sample_hist(self, j, n):
        T1 = self.T - 1
        if self.strides:
            # sparse strided far-history (matches inference rolling buffer), clamp at video start
            return [max(0, j - s) for s in self.strides]
        if j == 0:
            return [0] * T1
        if self.deterministic_hist:
            return [max(0, j - (T1 - k)) for k in range(T1)]
        lo = max(0, j - self.history_window)
        pool = list(range(lo, j))
        others_pool = pool[:-1]
        need = T1 - 1
        if len(others_pool) >= need:
            others = sorted(random.sample(others_pool, need))
        else:
            others = others_pool[:]
            while len(others) < need:
                others.insert(0, others[0] if others else lo)
        return others + [j - 1]

    def __getitem__(self, i):
        video, j = self.samples[i]
        frames = self.videos[video]
        idxs = self._sample_hist(j, len(frames)) + [j]
        imgs = [cv2.imread(str(frames[k]), cv2.IMREAD_COLOR) for k in idxs]
        for k, im in zip(idxs, imgs):
            if im is None:
                raise FileNotFoundError(f"failed to read {frames[k]} (video {video}, idx {k})")
        ori_shape = imgs[-1].shape[:2]
        boxes = self.labels[frames[j]]
        if self.augment:
            imgs, boxes, ratio_pad = augment_clip(imgs, boxes, self.imgsz)
        else:
            imgs, boxes, ratio_pad = letterbox_clip(imgs, boxes, self.imgsz)
        arr = np.stack([im.transpose(2, 0, 1) for im in imgs], axis=0)  # (T,3,H,W) BGR
        img = torch.from_numpy(np.ascontiguousarray(arr))
        n = len(boxes)
        return {
            "img": img,
            "cls": torch.from_numpy(boxes[:, 0:1]) if n else torch.zeros((0, 1)),
            "bboxes": torch.from_numpy(boxes[:, 1:5]) if n else torch.zeros((0, 4)),
            "batch_idx": torch.zeros(n),
            "im_file": str(frames[j]),
            "ori_shape": ori_shape,
            "ratio_pad": ratio_pad,
        }

    @staticmethod
    def collate_fn(batch):
        return YOLODataset.collate_fn(batch)