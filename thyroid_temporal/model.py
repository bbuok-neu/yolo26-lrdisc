"""Build YOLO26 + causal temporal neck (ThyroidTempoNet) and a clip-aware DetectionModel subclass."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel

from .temporal_modules import TemporalFusion, LongRangeDisc

REPO = Path(r"C:\Users\bbuok\udisk-code\ultralytics")
YOLO26_YAML = REPO / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"
FEAT_LAYERS = (16, 19, 22)  # YOLO26 head P3/P4/P5 outputs feeding Detect


def default_level_cfg(T=4, hid=8):
    """Per-level module config: light on P3, full on P5 (paper_notes.md recommendation)."""
    return {
        16: dict(T=T, hid=hid, tada=True, gru=False, queue=False, ema=False),
        19: dict(T=T, hid=hid, tada=True, gru=True, queue=True, ema=False),
        22: dict(T=T, hid=hid, tada=True, gru=True, queue=True, ema=True),
    }


class TemporalDetectionModel(DetectionModel):
    def __init__(self, cfg=str(YOLO26_YAML), ch=3, nc=1, tcfg=None, verbose=True, insert_temporal=True):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.tcfg = tcfg or {}
        self.temporal_layers = []
        if insert_temporal:
            self._insert_temporal()

    def _insert_temporal(self):
        T = self.tcfg.get("T", 4)
        hid = self.tcfg.get("hid", 8)
        level_cfg = self.tcfg.get("level_cfg", default_level_cfg(T, hid))
        seq = list(self.model)
        detect = seq[-1]
        chs = [detect.cv2[i][0].conv.in_channels for i in range(detect.nl)]
        base = seq[:-1]  # layers 0..22
        idx = len(base)
        new_layers, detect_f = [], []
        for k, src in enumerate(FEAT_LAYERS):
            cfg = level_cfg[src]
            if cfg.get("lrdisc"):
                tf = LongRangeDisc(chs[k], strides=cfg.get("strides", (3, 10, 25)),
                                   shift=cfg.get("shift", 1))
                tf.f = src
                tf.i = idx
                new_layers.append(tf)
                detect_f.append(idx)
                idx += 1
            elif any(cfg.get(f) for f in ("tada", "gru", "queue", "ema", "lite")):
                tf = TemporalFusion(chs[k], **cfg)
                tf.f = src
                tf.i = idx
                new_layers.append(tf)
                detect_f.append(idx)
                idx += 1
            else:
                detect_f.append(src)  # passthrough: Detect reads the original feature directly
        detect.f = detect_f
        detect.i = idx
        self.model = nn.Sequential(*(base + new_layers + [detect]))
        self.save = sorted(
            {j for m in self.model for j in ([m.f] if isinstance(m.f, int) else m.f) if j != -1}
        )
        self.temporal_layers = new_layers

    def _extract_features(self, x):
        y = []
        stop = self.temporal_layers[0].i if self.temporal_layers else self.model[-1].i
        for m in self.model:
            if m.i >= stop:
                break
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return {src: y[src] for src in FEAT_LAYERS}

    def _set_ctx(self, ctx):
        for m in self.temporal_layers:
            m.ctx = None if ctx is None else ctx[m.f]

    def _forward_clip(self, x):
        B, T = x.shape[:2]
        hist = x[:, :-1].reshape(B * (T - 1), *x.shape[2:])
        # Extract history features in EVAL mode (running BN stats) so they match streaming
        # inference exactly. Training-mode batch-stat BN on the history pass creates a
        # train/inference feature mismatch that makes the learned temporal fusion harmful
        # (diagnosed: frozen cache >> updating cache). Current pass stays in train mode.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            feats = self._extract_features(hist)
        if was_training:
            self.train()
        ctx = {src: feats[src].view(B, T - 1, *feats[src].shape[1:]) for src in FEAT_LAYERS}
        self._set_ctx(ctx)
        try:
            return self._predict_once(x[:, -1])
        finally:
            self._set_ctx(None)

    def predict(self, x, profile=False, batch=None, augment=False, embed=None):
        if isinstance(x, torch.Tensor) and x.ndim == 5:
            return self._forward_clip(x)
        return self._predict_once(x)

    def reset_state(self):
        for m in self.temporal_layers:
            m.reset_state()


def build_temporal_model(weights=None, nc=1, tcfg=None, verbose=True, ch=3):
    if isinstance(weights, TemporalDetectionModel):
        # Resume path: checkpoint already holds the full temporal architecture (Detect moved to
        # index 26). Rebuilding the base and re-inserting would misalign Detect weight transfer.
        # Build the identical architecture and use BaseModel.load, which rebuilds the cls branch
        # at the source width (c3 depends on the pretrained head, not nc) before transferring.
        model = TemporalDetectionModel(cfg=str(YOLO26_YAML), ch=ch, nc=nc, tcfg=tcfg, verbose=verbose,
                                       insert_temporal=True)
        model.load(weights, verbose=verbose)
        wsd, msd = weights.float().state_dict(), model.state_dict()
        bad = [k for k in wsd if k not in msd or msd[k].shape != wsd[k].shape
               or (wsd[k].is_floating_point() and not torch.equal(msd[k], wsd[k]))]
        if bad:
            raise RuntimeError(f"resume weight transfer incomplete (tcfg differs from checkpoint?): {bad[:5]}")
        model.end2end = True
        return model
    # Build base first, load pretrained weights while Detect is still at its original index (23),
    # THEN insert temporal layers (the Detect object is moved, preserving its loaded weights).
    model = TemporalDetectionModel(cfg=str(YOLO26_YAML), ch=ch, nc=nc, tcfg=tcfg, verbose=verbose,
                                   insert_temporal=False)
    if weights is not None:
        if isinstance(weights, (str, Path)):
            weights = torch.load(weights, map_location="cpu", weights_only=False)
        model.load(weights, verbose=verbose)
    model._insert_temporal()
    model.end2end = True  # NMS-free one-to-one inference (yaml flag isn't a state_dict attr)
    return model