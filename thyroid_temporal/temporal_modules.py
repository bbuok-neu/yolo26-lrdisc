"""Causal temporal fusion modules for YOLO26 thyroid video detection.

Design references (see docs/paper_notes.md):
  - TSdetector (MIA 2025): TAda temporal-adaptive conv calibration + ConvLSTM queue + progressive
    hierarchical queue integration. We use a lightweight TAda-lite + ConvGRU (2 gates) + progressive
    queue, dropping the expensive 5x5 Refine and the NMS-dependent PAC post-processing.
  - UltraDet (MICCAI 2023): rolling per-frame feature cache so history frames are NOT re-run through
    the backbone at inference (the main real-time win over TSdetector's 4x recompute).
  - EMA-SAM (2025): confidence-gated EMA memory pointer for a stable lesion prototype.
  - MWNet (MIA 2026): detached history features (no grad to past frames) -> fits 6GB GPU.

All modules are strictly causal (only frames <= t) and start as an identity residual (zero-init output
conv) so a pretrained YOLO26 is preserved at initialization and temporal corrections are learned.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def causal_temporal_mean(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Causal moving average over the time dim (dim=1) of (B,T,C,H,W), left-padded."""
    T = x.shape[1]
    cs = x.cumsum(dim=1)
    zero = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    cs = torch.cat([zero, cs], dim=1)  # cs[:, t] = sum of frames [0, t)
    idx = torch.arange(T, device=x.device)
    lo = (idx - k).clamp(min=0)
    sums = cs[:, idx + 1] - cs[:, lo]
    counts = (idx + 1 - lo).clamp(min=1).to(x.dtype).view(1, T, 1, 1, 1)
    return sums / counts


class ConvGRUCell(nn.Module):
    """ConvGRU with 3x3 gates, small hidden state (TSdetector R_CLSTM uses 8 channels)."""

    def __init__(self, c, hid):
        super().__init__()
        self.hid = hid
        self.gate = nn.Conv2d(c + hid, 2 * hid, 3, padding=1)
        self.cand = nn.Conv2d(c + hid, hid, 3, padding=1)

    def forward(self, x, h):
        z, r = self.gate(torch.cat([x, h], 1)).chunk(2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class TemporalFusion(nn.Module):
    """Fuses current-frame neck features with a causal window of history features.

    Training: history supplied via ``ctx`` (B,T-1,C,H,W), detached (MWNet-style).
    Streaming inference: history kept in an internal rolling deque (UltraDet-style cache).
    Output = current + zero-init residual fusion, so init == identity (pretrained YOLO26 preserved).
    """

    def __init__(self, c, T=4, hid=8, tada=True, gru=True, queue=True, ema=False, route_ratio=4, lite=False):
        super().__init__()
        self.c, self.T, self.hid = c, T, hid
        self.lite = lite
        if lite:
            # Lightweight denoising: out = causal_mean(window) + zero-init residual.
            # Starts at the pure temporal-mean output (already > baseline), few params -> low overfit.
            tada = gru = queue = ema = False
            self.out = ConvBNAct(2 * c, c, 1, act=False)
            nn.init.zeros_(self.out.conv.weight)
            self.hist = deque(maxlen=T - 1)
            self.ema_state = None
            self.ctx = None
            self.perturb = None
            return
        self.use_tada, self.use_gru, self.use_queue, self.use_ema = tada, gru, queue, ema
        ch = []

        if tada:
            rc = max(c // route_ratio, 8)
            # causal temporal route MLP over channel-pooled window -> per-frame calibration alpha
            self.route_t = nn.Conv1d(c, rc, 3, padding=0)
            self.route_c = nn.Conv1d(rc, c, 1)
            nn.init.zeros_(self.route_c.weight)
            nn.init.zeros_(self.route_c.bias)  # alpha = 1 + tanh(0) = 1 at init
            self.dw = ConvBNAct(c, c, 3, g=c)
            self.lateral = ConvBNAct(c, c, 1)
            ch.append(c)  # x_sm last
        if gru:
            self.gru = ConvGRUCell(c, hid)
            self.q = nn.Conv2d(hid, c, 1)
            ch.append(c)  # gru_q
        if queue:
            self.reduces = nn.ModuleList([ConvBNAct(c, c // 2, 1) for _ in range(2 * (T - 1))])
            ch.append(c)  # progressive h2
        if ema:
            self.ema_gate = nn.Conv1d(c, c, 1)
            self.ema_proj = ConvBNAct(c, c, 1)
            self.ema_m = 0.9
            ch.append(c)

        self.out = ConvBNAct(sum(ch), c, 1, act=False)
        # Zero-init ONLY the conv weight (keep bn.weight=1): fused=0 at init (identity residual),
        # but gradients still flow to conv.weight so the temporal branch can learn. Zeroing bn.weight
        # too would create a dead-gradient loop (only bn.bias learns -> constant channel offset).
        nn.init.zeros_(self.out.conv.weight)

        # streaming state
        self.hist = deque(maxlen=T - 1)
        self.ema_state = None
        self.ctx = None  # training context (B,T-1,C,H,W)
        self.perturb = None  # TemporalLens diagnostic: None | 'shuffle' | 'freeze' | 'zero'

    def reset_state(self):
        self.hist.clear()
        self.ema_state = None

    def _build_window(self, x):
        B = x.shape[0]
        if self.ctx is not None:
            hist = self.ctx
            if hist.shape[0] != B:
                raise RuntimeError("ctx batch mismatch")
            return torch.cat([hist, x.unsqueeze(1)], dim=1), True
        # streaming: pad history by repeating earliest available frame (causal, no future)
        if len(self.hist) == 0 or self.hist[0].shape[0] != B or self.hist[0].shape[-2:] != x.shape[-2:]:
            self.reset_state()
        frames = list(self.hist)
        if self.perturb == "shuffle":
            import random
            random.shuffle(frames)
        while len(frames) < self.T - 1:
            frames.insert(0, frames[0] if frames else x.detach())
        hist = torch.stack(frames, dim=1)
        if self.perturb == "zero":
            hist = torch.zeros_like(hist)
        return torch.cat([hist, x.unsqueeze(1)], dim=1), False

    def forward(self, x):
        win, training_ctx = self._build_window(x)  # (B,T,C,H,W)
        B, T, C, H, W = win.shape
        if self.lite:
            mean = win.mean(dim=1)  # causal mean of last T frames incl current (denoised)
            out = mean + self.out(torch.cat([mean, x], dim=1))  # zero-init residual -> starts at mean
            if not training_ctx and self.perturb != "freeze":
                self.hist.append(x.detach())
            return out
        feats = []
        x_cal = win
        if self.use_tada:
            pooled = win.mean(dim=(-1, -2)).transpose(1, 2)  # (B,C,T)
            r = F.pad(pooled, (2, 0))  # causal left pad for k=3
            r = self.route_c(F.silu(self.route_t(r)))  # (B,C,T)
            alpha = 1.0 + torch.tanh(r).transpose(1, 2)  # (B,T,C)
            x_cal = win * alpha[..., None, None]
            dw = self.dw(x_cal.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)
            x_sm = causal_temporal_mean(dw, 3)
            feats.append(x_sm[:, -1])
        if self.use_gru:
            h = x.new_zeros(B, self.hid, H, W)
            src = x_sm if self.use_tada else x_cal
            for t in range(T):
                h = self.gru(src[:, t], h)
            gru_q = self.q(h)
            feats.append(gru_q)
        if self.use_queue:
            acc = x_cal[:, 0]
            for t in range(1, T - 1):
                a = self.reduces[2 * (t - 1)](acc)
                b = self.reduces[2 * (t - 1) + 1](x_cal[:, t])
                acc = torch.cat([a, b], dim=1) + x_cal[:, t]
            last = gru_q if self.use_gru else x_cal[:, -1]
            a = self.reduces[2 * (T - 2)](acc)
            b = self.reduces[2 * (T - 2) + 1](last)
            h2 = torch.cat([a, b], dim=1) + last
            feats.append(h2)
        if self.use_ema:
            g = torch.sigmoid(self.ema_gate(x_cal[:, -1].mean(dim=(-1, -2)).unsqueeze(-1))).squeeze(-1)
            g = g[..., None, None]
            if training_ctx:
                ema = x.new_zeros(B, C, H, W)
                for t in range(T):
                    gt = torch.sigmoid(
                        self.ema_gate(x_cal[:, t].mean(dim=(-1, -2)).unsqueeze(-1))
                    ).squeeze(-1)[..., None, None]
                    ema = self.ema_m * ema + (1 - self.ema_m) * gt * x_cal[:, t]
            else:
                if self.ema_state is None or self.ema_state.shape[0] != B:
                    self.ema_state = x.new_zeros(B, C, H, W)
                self.ema_state = self.ema_m * self.ema_state + (1 - self.ema_m) * g * x_cal[:, -1]
                ema = self.ema_state
            feats.append(self.ema_proj(ema))

        fused = self.out(torch.cat(feats, dim=1))
        out = x + fused

        if not training_ctx and self.perturb != "freeze":
            self.hist.append(x.detach())
        return out


def _shift(t, dy, dx):
    """Spatially shift a (B,C,H,W) tensor by (dy,dx) with zero padding (motion tolerance)."""
    B, C, H, W = t.shape
    p = F.pad(t, (abs(dx), abs(dx), abs(dy), abs(dy)))
    y0 = abs(dy) + dy
    x0 = abs(dx) + dx
    return p[:, :, y0:y0 + H, x0:x0 + W]


class LongRangeDisc(nn.Module):
    """Long-range DISCRIMINATIVE temporal fusion (vessel vs nodule), not just denoising.

    Motivation (top-venue survey, docs/research_longrange_temporal.md): under probe sweeping a solid
    nodule is a persistent stable mass, while a vessel cross-section collapses/elongates/pulsates.
    At ~30fps adjacent frames barely change, so history is sampled at SPARSE STRIDES (far frames).

    Two parameter-light branches with strong inductive bias (low overfit):
      * persistence gate: motion-tolerant local correlation of current neck feature vs each far-history
        feature (max over small shifts). Stable (nodule) -> high correlation -> gate open;
        changing (vessel) -> low correlation -> gate closed.
      * change/motion branch: multi-scale frame differences (x_t - h_k), depthwise conv (STNMamba-style,
        no optical flow) -> lets the net suppress dynamic vessel-like regions.
    Output = x_t * gate + zero-init residual(change, x_t). Causal; rolling strided buffer at inference.
    """

    def __init__(self, c, strides=(3, 10, 25), shift=2, use_change=True, use_persist=True, pool=4):
        super().__init__()
        self.c = c
        self.strides = tuple(strides)
        self.K = len(strides)
        self.shift = shift
        self.pool = pool  # persistence correlation at 1/pool resolution (speed); gate upsampled back
        self.use_change = use_change
        self.use_persist = use_persist
        self.maxlen = max(strides)
        self.buf = deque(maxlen=self.maxlen)
        self.ctx = None
        self.perturb = None
        if use_persist:
            self.pgate = nn.Conv2d(1, c, 1)
            nn.init.zeros_(self.pgate.weight)
            nn.init.constant_(self.pgate.bias, 2.0)  # sigmoid(2)~0.88 gate ~open at init
        if use_change:
            self.change = ConvBNAct(c * self.K, c, 3, g=c)  # depthwise, light
        n_in = c * (1 + (1 if use_change else 0))
        self.out = ConvBNAct(n_in, c, 1, act=False)
        nn.init.zeros_(self.out.conv.weight)

    def reset_state(self):
        self.buf.clear()

    def _history(self, x):
        if self.ctx is not None:
            return self.ctx, True
        B = x.shape[0]
        if len(self.buf) == 0 or self.buf[0].shape[0] != B or self.buf[0].shape[-2:] != x.shape[-2:]:
            self.reset_state()
        feats = []
        for s in self.strides:
            if len(self.buf) >= s:
                feats.append(self.buf[-s])
            else:
                feats.append(self.buf[0] if self.buf else x.detach())
        return torch.stack(feats, dim=1), False

    def forward(self, x):
        Hst, training_ctx = self._history(x)  # (B,K,C,H,W)
        B, C, H, W = x.shape
        gate = 1.0
        if self.use_persist:
            # Persistence map computed WITHOUT grad (inputs detached): avoids storing activations
            # for every shift x history x level (the grad-through version blew past 6GB VRAM).
            # The gate still learns via pgate, and x*gate still passes grad to x.
            with torch.no_grad():
                pool = getattr(self, "pool", 1)  # old ckpts predate pool attr -> full-res (== trained)
                xd = x.detach()
                Hs = Hst
                B, K, C, H, W = Hs.shape
                if pool > 1:
                    xd = F.avg_pool2d(xd, pool)
                    Hs = F.avg_pool2d(Hs.reshape(B * K, C, H, W), pool)
                h, w = xd.shape[-2:]
                xn = F.normalize(xd, dim=1)
                hn = F.normalize(Hs.reshape(B * K, C, h, w), dim=1)
                # Vectorized motion-tolerant local correlation: unfold gives all 9 shifts at once
                # (a Python loop over shifts was launch-bound: ~324 kernel launches/frame -> 15fps).
                patches = F.unfold(hn, 3, padding=1).reshape(B * K, C, 9, h, w)
                xn_r = xn.unsqueeze(1).expand(B, K, C, h, w).reshape(B * K, C, 1, h, w)
                corr = (xn_r * patches).sum(1)  # (B*K,9,h,w)
                best = corr.max(1, keepdim=True).values  # (B*K,1,h,w)
                P = best.reshape(B, K, 1, h, w).mean(1)  # (B,1,h,w)
                if pool > 1:
                    P = F.interpolate(P, size=(H, W), mode="bilinear", align_corners=False)
                if self.perturb == "zero":
                    P = torch.zeros_like(P)
                elif self.perturb == "shuffle":
                    P = P.flip(-1)
            gate = torch.sigmoid(self.pgate(P))
        feats = [x]
        if self.use_change:
            D = torch.cat([x - Hst[:, k] for k in range(self.K)], dim=1)
            feats.append(self.change(D))
        out = x * gate + self.out(torch.cat(feats, dim=1))
        if not training_ctx and self.perturb != "freeze":
            self.buf.append(x.detach())
        return out