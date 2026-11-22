"""Characterize ThyroidXL image noise vs the project's video frames.

Analyses (all on grayscale):
1. Local variance vs local mean on flat 16x16 patches -> noise model:
   var ~ const (additive), var ~ mean (Poisson), var ~ mean^2 (multiplicative/speckle).
2. Flat-patch high-frequency sigma (Immerkaer Laplacian estimator).
3. Mean radial power spectrum -> periodic interference peaks / scan-line anisotropy.
4. JPEG 8x8 blockiness score (PNG container may hide JPEG history).
5. Saves zoomed crops for visual inspection.

Usage: python scripts/analyze_xl_noise.py [--n 200]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "data" / "thyroid_yolo"
OUT = PROJECT / "docs" / "xl_noise"

LAP = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def flat_patch_stats(img, patch=16, flat_frac=0.25):
    """Return (means, vars, hf_sigmas) over the flattest fraction of patches."""
    g = img.astype(np.float64)
    h, w = g.shape
    ph, pw = h // patch, w // patch
    g = g[: ph * patch, : pw * patch].reshape(ph, patch, pw, patch).transpose(0, 2, 1, 3).reshape(-1, patch, patch)
    mean = g.mean(axis=(1, 2))
    var = g.var(axis=(1, 2))
    # gradient energy proxy for structure: variance of first differences
    gx = np.abs(np.diff(g, axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(g, axis=1)).mean(axis=(1, 2))
    grad = gx + gy
    # flatness: low gradient relative to patch mean (exclude near-black border patches)
    ok = mean > 20
    rel = grad / np.maximum(mean, 1)
    thr = np.quantile(rel[ok], flat_frac)
    sel = ok & (rel <= thr)
    m, v = mean[sel], var[sel]
    # Immerkaer sigma: sqrt(pi/2) * mean(|LAP conv response|) / 6, per patch
    resp = np.zeros((len(g), patch - 2, patch - 2))
    for di in range(3):
        for dj in range(3):
            resp += LAP[di, dj] * g[:, di:di + patch - 2, dj:dj + patch - 2]
    sigma = np.sqrt(np.pi / 2) * np.abs(resp).mean(axis=(1, 2)) / 6
    return mean[sel], var[sel], sigma[sel]


def radial_spectrum(imgs, size=256):
    acc = np.zeros((size, size))
    for im in imgs:
        r = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)
        r = (r - r.mean()) * np.hanning(size)[:, None] * np.hanning(size)[None, :]
        acc += np.abs(np.fft.fftshift(np.fft.fft2(r))) ** 2
    acc /= len(imgs)
    cy = cx = size // 2
    y, x = np.indices((size, size))
    rad = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    prof = np.bincount(rad.ravel(), acc.ravel()) / np.maximum(np.bincount(rad.ravel()), 1)
    return np.log10(prof + 1e-12), np.log10(acc + 1e-12)


def blockiness(img):
    g = img.astype(np.float64)
    dx = np.abs(np.diff(g, axis=1))
    cols = np.arange(dx.shape[1])
    on = dx[:, (cols + 1) % 8 == 0].mean()   # differences crossing 8-px boundaries
    off = dx[:, (cols + 1) % 8 != 0].mean()
    return float(on / max(off, 1e-9))


def fit_var_model(means, vars):
    ok = (means > 10) & (vars > 1e-6)
    lm, lv = np.log(means[ok]), np.log(vars[ok])
    b, a = np.polyfit(lm, lv, 1)  # log var = b*log mean + a
    return float(b), float(np.exp(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    report = {}
    samples = {}
    for name, d in (("thyroidxl", BASE / "images" / "train_xl"),
                    ("video", BASE / "images" / "train")):
        files = sorted(d.glob("*.png"))
        pick = rng.choice(len(files), min(args.n, len(files)), replace=False)
        imgs = [cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE) for i in pick]
        samples[name] = [files[i] for i in pick[:6]]
        M, V, S, B = [], [], [], []
        for im in imgs:
            m, v, s = flat_patch_stats(im)
            M.append(m); V.append(v); S.append(s)
            B.append(blockiness(im))
        M, V, S = np.concatenate(M), np.concatenate(V), np.concatenate(S)
        b, a = fit_var_model(M, V)
        prof, spec = radial_spectrum(imgs[:60])
        # periodic peakiness: max deviation of profile from smooth trend in mid/high freq
        from scipy.ndimage import uniform_filter1d
        trend = uniform_filter1d(prof, 9)
        resid = prof - trend
        hi = resid[20:120]
        report[name] = {
            "n_images": len(imgs),
            "flat_patch_hf_sigma_mean": float(S.mean()),
            "flat_patch_hf_sigma_p50": float(np.median(S)),
            "var_vs_mean_exponent_b": b,
            "var_vs_mean_coeff_a": a,
            "blockiness_ratio": float(np.mean(B)),
            "spectrum_peak_max_resid": float(hi.max()),
            "spectrum_resid_p99": float(np.quantile(hi, 0.99)),
            "mean_size": np.median([im.shape for im in imgs], axis=0).astype(int).tolist(),
        }
        np.save(OUT / f"spectrum_{name}.npy", spec)
        np.save(OUT / f"radial_{name}.npy", prof)
        # bin scatter for plotting
        bins = np.linspace(0, 255, 26)
        idx = np.digitize(M, bins)
        bm = [float(M[idx == i].mean()) if (idx == i).any() else None for i in range(1, 26)]
        bv = [float(V[idx == i].mean()) if (idx == i).any() else None for i in range(1, 26)]
        report[name]["mean_bins"] = bm
        report[name]["var_bins"] = bv

    # zoomed crops for visual inspection (2x nearest to expose pixel-level noise)
    for name, paths in samples.items():
        for k, p in enumerate(paths[:4]):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            h, w = im.shape
            crop = im[h // 2 - 96:h // 2 + 96, w // 2 - 96:w // 2 + 96]
            cv2.imwrite(str(OUT / f"crop_{name}_{k}.png"),
                        cv2.resize(crop, (384, 384), interpolation=cv2.INTER_NEAREST))

    (OUT / "noise_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()