"""Custom trainer wiring the clip dataset + temporal model into the ultralytics training loop."""

from __future__ import annotations

from copy import copy

import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import unwrap_model

from .clip_dataset import ClipDataset
from .model import build_temporal_model


class TemporalValidator(DetectionValidator):
    """DetectionValidator that handles 5D clip batches (B,T,3,H,W).

    The base _prepare_batch reads imgsz = batch['img'].shape[2:], which for a 5D clip tensor
    yields (3,H,W) and corrupts GT box scaling -> mAP=0. Use shape[-2:] instead.
    """

    def _prepare_batch(self, si, batch):
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[-2:]  # (H,W) regardless of 4D/5D
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
        return {
            "cls": cls, "bboxes": bbox, "ori_shape": ori_shape, "imgsz": imgsz,
            "ratio_pad": ratio_pad, "im_file": batch["im_file"][si],
        }


class TemporalTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, tcfg=None, _callbacks=None):
        self.tcfg = tcfg or {}
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = build_temporal_model(
            weights, nc=self.data["nc"], tcfg=self.tcfg, verbose=verbose,
            ch=self.data.get("channels", 3),
        )
        model.args = self.args
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        return ClipDataset(
            img_path,
            data=self.data,
            imgsz=self.args.imgsz,
            T=self.tcfg.get("T", 4),
            augment=(mode == "train"),
            history_window=self.tcfg.get("history_window", 50),
            deterministic_hist=True,  # consecutive history matches streaming inference exactly
            strides=self.tcfg.get("strides"),
            static_mode=self.tcfg.get("static_mode", "repeat"),
        )

    def preprocess_batch(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = batch["img"].float() / 255  # (B,T,3,H,W)
        return batch

    def _build_train_pipeline(self):
        # Skip DetectionTrainer's dataset introspection (_check_max_det / auto_batch expect
        # YOLODataset.labels as a list of dicts); ClipDataset.labels is a path->array dict.
        super(DetectionTrainer, self)._build_train_pipeline()
        unwrap_model(self.model).set_head_attr(max_det=self.args.max_det)

    def get_validator(self):
        return TemporalValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def final_eval(self):
        # Skip ultralytics' end-of-training full-test validation (dataloader worker OOM on 16GB RAM).
        # Authoritative test evaluation is done separately via scripts/stream_eval.py.
        pass