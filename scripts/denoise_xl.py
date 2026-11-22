"""Compare denoising schemes for ThyroidXL stills (noise characterized as spatially-correlated,
signal-dependent granular speckle; no impulse/JPEG/periodic components).

Methods: median3 (impulse control), gauss1, bilateral, NLM (h=8/14), Lee 7x7 (speckle MMSE),
anisotropic diffusion (Perona-Malik). Metrics use ThyroidXL masks:
- flat-patch HF sigma (noise level), tissue CV (speckle contrast)
- CNR nodule-vs-surround, nodule mean shift (contrast/mean preservation)
- edge gradient retention + gradient correlation on the mask boundary band (edge preservation)
- SSIM vs original (structural fidelity)

Usage:
  python scripts/denoise_xl.py compare [--n 120]        # metrics + visual grid on train sample
  python scripts/denoise_xl.py apply --method lee7 --split test   # materialize denoised variant
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
BASE = PROJECT / "data" / "thyroid_yolo"
XL_SRC = Path(r"C:\Users\bbuok\Documents\ChatGPT\甲状腺大模型\公开数据集\ThyroidXL")
OUT = PROJECT / "docs" / "xl_noise"

from scripts.analyze_xl_noise import flat_patch_stats  # noqa: E402


def imread_gray(path):
    """cv2.imread fails on non-ASCII (Chinese) paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


def lee_filter(img, size=7):
    """Lee MMSE speckle filter: y = mu + k*(x-mu), k = max(var - sigma_n^2, 0)/var.
    sigma_n^2 estimated as the mean local variance of the flattest quartile (multiplicative
    noise: scale with local mean^2 via CV of flat patches)."""
    g = img.astype(np.float64)
    mu = cv2.boxFilter(g, -1, (size, size))
    mu2 = cv2.boxFilter(g * g, -1, (size, size))
    var = np.maximum(mu2 - mu * mu, 0)
    # noise CV from flat regions: use global robust estimate
    cv_flat = 0.12  # measured flat-patch sigma/mean ~ 0.10-0.15 for XL; conservative
    noise_var = (cv_flat * mu) ** 2
    k = np.clip(var - noise_var, 0, None) / np.maximum(var, 1e-9)
    return np.clip(mu + k * (g - mu), 0, 255).astype(np.uint8)


def aniso_diffusion(img, iters=15, kappa=20.0, lam=0.15):
    g = img.astype(np.float64)
    for _ in range(iters):
        dn = np.roll(g, -1, 0) - g
        ds = np.roll(g, 1, 0) - g
        de = np.roll(g, -1, 1) - g
        dw = np.roll(g, 1, 1) - g
        cn = np.exp(-(dn / kappa) ** 2)
        cs = np.exp(-(ds / kappa) ** 2)
        ce = np.exp(-(de / kappa) ** 2)
        cw = np.exp(-(dw / kappa) ** 2)
        g += lam * (cn * dn + cs * ds + ce * de + cw * dw)
    return np.clip(g, 0, 255).astype(np.uint8)


def lee_fit(img, size=7):
    """Generalized Lee with an empirical signal-dependent noise model var_noise(mu) interpolated
    from measured flat-patch statistics (XL noise is not purely multiplicative: CV falls with
    brightness, so constant-CV Lee under-smooths dark regions)."""
    rep = json.loads((OUT / "noise_report.json").read_text())["thyroidxl"]
    pairs = [(m, v) for m, v in zip(rep["mean_bins"], rep["var_bins"]) if m and v]
    ms = np.array([p[0] for p in pairs])
    vs = np.maximum.accumulate([p[1] for p in pairs])  # enforce monotone var(mu)
    g = img.astype(np.float64)
    mu = cv2.boxFilter(g, -1, (size, size))
    var = np.maximum(cv2.boxFilter(g * g, -1, (size, size)) - mu * mu, 0)
    noise_var = np.interp(mu, ms, vs)
    k = np.clip(var - noise_var, 0, None) / np.maximum(var, 1e-9)
    return np.clip(mu + k * (g - mu), 0, 255).astype(np.uint8)


METHODS = {
    "median3": lambda im: cv2.medianBlur(im, 3),
    "gauss1": lambda im: cv2.GaussianBlur(im, (0, 0), 1.0),
    "bilateral": lambda im: cv2.bilateralFilter(im, 7, 25, 5),
    "nlm8": lambda im: cv2.fastNlMeansDenoising(im, None, h=8, templateWindowSize=7, searchWindowSize=21),
    "nlm14": lambda im: cv2.fastNlMeansDenoising(im, None, h=14, templateWindowSize=7, searchWindowSize=21),
    "lee7": lee_filter,
    "lee_fit": lee_fit,
    "aniso": aniso_diffusion,
}


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = k @ k.T
    f = lambda x: cv2.filter2D(x, -1, win)[5:-5, 5:-5]
    ma, mb = f(a), f(b)
    va, vb = f(a * a) - ma * ma, f(b * b) - mb * mb
    cov = f(a * b) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def grad_mag(g):
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def image_metrics(orig, den, mask):
    m, v, s = flat_patch_stats(den)
    hf = float(s.mean())
    g = den.astype(np.float64)
    # tissue CV
    h, w = g.shape
    ph, pw = h // 16, w // 16
    p = g[: ph * 16, : pw * 16].reshape(ph, 16, pw, 16).transpose(0, 2, 1, 3).reshape(-1, 16, 16)
    pm, pv = p.mean((1, 2)), p.var((1, 2))
    sel = (pm > 60) & (pm < 200)
    tcv = float((np.sqrt(pv[sel]) / pm[sel]).mean()) if sel.any() else float("nan")

    interior = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    ring = (cv2.dilate(mask, np.ones((31, 31), np.uint8)) > 0) & (mask == 0)
    go = orig.astype(np.float64)
    ring &= go > 30
    out = {"hf_sigma": hf, "tissue_cv": tcv, "ssim": ssim(orig, den)}
    if interior.sum() > 50 and ring.sum() > 50:
        mi, si = g[interior].mean(), g[interior].std()
        mb, sb = g[ring].mean(), g[ring].std()
        out["cnr"] = float(abs(mi - mb) / np.sqrt((si ** 2 + sb ** 2) / 2 + 1e-9))
        out["nodule_mean_shift"] = float(mi - go[interior].mean())
        band = (cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0) & (cv2.erode(mask, np.ones((7, 7), np.uint8)) == 0)
        g0, g1 = grad_mag(go), grad_mag(g)
        out["edge_grad_retention"] = float(g1[band].mean() / max(g0[band].mean(), 1e-9))
        a, b = g0[band], g1[band]
        out["edge_grad_corr"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
    return out


def load_manifest(split):
    rows = list(csv.DictReader(open(BASE / f"manifest_{split}.csv")))
    return rows


def compare(n):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_manifest("train_xl")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), min(n, len(rows)), replace=False)
    agg = {m: [] for m in ["original"] + list(METHODS)}
    grid_examples = []
    for c, i in enumerate(pick):
        r = rows[i]
        orig = cv2.imread(str(BASE / "images" / "train_xl" / (r["pseudo"] + ".png")), cv2.IMREAD_GRAYSCALE)
        mask = imread_gray(XL_SRC / "train" / "masks" / r["orig"])
        if mask is None or mask.shape != orig.shape:
            continue
        for name, fn in [("original", lambda x: x)] + list(METHODS.items()):
            den = fn(orig)
            agg[name].append(image_metrics(orig, den, mask))
        if c < 3:
            ys, xs = np.where(mask > 0)
            cy, cx = int(ys.mean()), int(xs.mean())
            h, w = orig.shape
            y0, x0 = np.clip(cy - 96, 0, h - 192), np.clip(cx - 96, 0, w - 192)
            grid_examples.append((r["pseudo"], orig, mask, (y0, x0)))
        if (c + 1) % 30 == 0:
            print(f"{c + 1}/{len(pick)}")

    summary = {}
    keys = ["hf_sigma", "tissue_cv", "cnr", "nodule_mean_shift", "edge_grad_retention", "edge_grad_corr", "ssim"]
    for name, recs in agg.items():
        summary[name] = {k: float(np.nanmean([r.get(k, np.nan) for r in recs])) for k in keys}
    (OUT / "denoise_compare.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    # visual grid: rows = examples, cols = original + methods, 2x zoom crops
    names = ["original"] + list(METHODS)
    cell = 384
    pad = 28
    grid = np.full((len(grid_examples) * (cell + pad), len(names) * cell, 3), 255, np.uint8)
    for ri, (pseudo, orig, mask, (y0, x0)) in enumerate(grid_examples):
        for ci, name in enumerate(names):
            den = orig if name == "original" else METHODS[name](orig)
            crop = den[y0:y0 + 192, x0:x0 + 192]
            crop = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_NEAREST)
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            # nodule boundary overlay (thin green) for reference
            mb = (cv2.dilate(mask, np.ones((3, 3), np.uint8)) - cv2.erode(mask, np.ones((3, 3), np.uint8)))
            mbc = mb[y0:y0 + 192, x0:x0 + 192]
            mbc = cv2.resize(mbc, (cell, cell), interpolation=cv2.INTER_NEAREST) > 0
            crop[mbc] = (0, 200, 0)
            y = ri * (cell + pad) + pad
            grid[y:y + cell, ci * cell:(ci + 1) * cell] = crop
            if ri == 0:
                cv2.putText(grid, name, (ci * cell + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(OUT / "denoise_grid.png"), grid)
    print("grid saved:", OUT / "denoise_grid.png")


def apply(method, split):
    """Materialize a denoised variant of images/<split> as images/<split>_<method> (+label links)."""
    fn = METHODS[method]
    src_img = BASE / "images" / split
    dst_img = BASE / "images" / f"{split}_{method}"
    dst_lbl = BASE / "labels" / f"{split}_{method}"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    files = sorted(src_img.glob("*.png"))
    for k, p in enumerate(files):
        out = dst_img / p.name
        if not out.exists():
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out), fn(im))
        lp = BASE / "labels" / split / (p.stem + ".txt")
        dl = dst_lbl / lp.name
        if lp.exists() and not dl.exists():
            try:
                os.link(lp, dl)
            except OSError:
                shutil.copy2(lp, dl)
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{len(files)}")
    print(f"done: {dst_img} ({len(files)} images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "apply"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--method", default="lee7")
    ap.add_argument("--split", default="test_xl")
    args = ap.parse_args()
    if args.cmd == "compare":
        compare(args.n)
    else:
        apply(args.method, args.split)


if __name__ == "__main__":
    main()