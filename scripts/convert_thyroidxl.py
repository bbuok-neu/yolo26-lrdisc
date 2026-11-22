"""Convert ThyroidXL static images into single-frame pseudo-videos and merge them into the
video training set (hardlinks, no extra disk).

Each still becomes its own pseudo-video `xl<NNNNN>_0000.png`, so the lrdisc strided history
sampler (clamped at video start) degenerates to repeating the current frame; ClipDataset's
`static_mode` can instead synthesize black/jittered history for length-1 videos.

Both ThyroidXL classes (0=benign, 1=malignant) map to the single project class 0=nodule.
Val/test video splits are untouched; ThyroidXL test is converted separately as `test_xl`
for optional static-image evaluation.

Usage:
    python scripts/convert_thyroidxl.py [--src <ThyroidXL root>] [--frac 1.0]
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "data" / "thyroid_yolo"
DEFAULT_SRC = Path(r"C:\Users\bbuok\Documents\ChatGPT\甲状腺大模型\公开数据集\ThyroidXL")


def link(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_split(src, split, dst_name, frac=1.0):
    img_out = BASE / "images" / dst_name
    lbl_out = BASE / "labels" / dst_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    files = sorted((src / split / "images").glob("*.png"))
    if frac < 1.0:
        import random

        random.Random(0).shuffle(files)
        files = sorted(files[: int(len(files) * frac)])
    rows = []
    for n, p in enumerate(files):
        lp = src / split / "labels" / (p.stem + ".txt")
        lines = [ln.split() for ln in lp.read_text().strip().splitlines() if ln.strip()]
        assert lines, f"ThyroidXL image without bbox: {p}"
        stem = f"xl{n:05d}_0000"
        link(p, img_out / (stem + ".png"))
        # remap every class to 0 (nodule); keep normalized xywh as-is
        lbl_out.joinpath(stem + ".txt").write_text(
            "".join("0 " + " ".join(ln[1:5]) + "\n" for ln in lines)
        )
        rows.append({"pseudo": stem, "orig": p.name, "patient": p.stem.split("_")[0],
                     "orig_class": lines[0][0], "n_boxes": len(lines)})
    with open(BASE / f"manifest_{dst_name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pseudo", "orig", "patient", "orig_class", "n_boxes"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def build_mix():
    """train_mix = hardlinks of images/train (video frames) + images/train_xl (stills)."""
    n = 0
    for sub, ext in (("images", ".png"), ("labels", ".txt")):
        d = BASE / sub / "train_mix"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            old.unlink()
        for src_dir in ("train", "train_xl"):
            for p in (BASE / sub / src_dir).glob("*" + ext):
                link(p, d / p.name)
                n += 1
    yaml_path = BASE / "thyroid_full_xl.yaml"
    yaml_path.write_text(
        f"path: {BASE.as_posix()}\ntrain: images/train_mix\nval: images/val_sub\ntest: images/test\n"
        "names:\n  0: nodule\n"
    )
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of ThyroidXL train to use")
    args = ap.parse_args()
    src = Path(args.src)

    n_train = convert_split(src, "train", "train_xl", args.frac)
    n_test = convert_split(src, "test", "test_xl")
    yaml_path = build_mix()
    n_video = len(list((BASE / "images" / "train").glob("*.png")))
    n_mix = len(list((BASE / "images" / "train_mix").glob("*.png")))
    assert n_mix == n_video + n_train, (n_mix, n_video, n_train)
    print(f"video train frames: {n_video}")
    print(f"ThyroidXL train stills: {n_train} (frac={args.frac})")
    print(f"ThyroidXL test stills: {n_test} -> images/test_xl")
    print(f"train_mix total: {n_mix} ({n_train / n_mix:.1%} stills)")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()