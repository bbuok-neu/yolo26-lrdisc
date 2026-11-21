"""Create a ~1/5 video-level training subset (hardlinks, no extra disk) for fast iteration.

Val/test stay full and untouched (held-out videos), so final numbers remain comparable.
"""

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "data" / "thyroid_yolo"


def main(frac=5, seed=0):
    split = json.loads((BASE / "split_videos.json").read_text())
    train_videos = split["train"]
    counts = defaultdict(int)
    with open(BASE / "manifest.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                counts[row["video"]] += 1
    total = sum(counts[v] for v in train_videos)
    target = total / frac

    rng = np.random.default_rng(seed)
    order = list(train_videos)
    rng.shuffle(order)
    chosen, acc = [], 0
    for v in order:
        chosen.append(v)
        acc += counts[v]
        if acc >= target:
            break
    chosen = sorted(chosen)

    for sub in ("images", "labels"):
        d = BASE / sub / "train_sub"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            old.unlink()
    n = 0
    for v in chosen:
        for img in (BASE / "images" / "train").glob(f"{v}_*.png"):
            for sub, ext in (("images", ".png"), ("labels", ".txt")):
                src = BASE / sub / "train" / (img.stem + ext)
                dst = BASE / sub / "train_sub" / (img.stem + ext)
                if src.exists():
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
            n += 1

    yaml_path = BASE / "thyroid_sub.yaml"
    yaml_path.write_text(
        f"path: {BASE.as_posix()}\ntrain: images/train_sub\nval: images/val\ntest: images/test\n"
        "names:\n  0: nodule\n"
    )

    # 1/5 val subset for fast in-training model selection (final eval stays on full test)
    val_videos = split["val"]
    vcounts = defaultdict(int)
    with open(BASE / "manifest.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                vcounts[row["video"]] += 1
    vtotal = sum(vcounts[v] for v in val_videos)
    rngv = np.random.default_rng(seed)
    vorder = list(val_videos)
    rngv.shuffle(vorder)
    vchosen, vacc = [], 0
    for v in vorder:
        vchosen.append(v)
        vacc += vcounts[v]
        if vacc >= vtotal / frac:
            break
    vchosen = sorted(vchosen)
    for sub in ("images", "labels"):
        d = BASE / sub / "val_sub"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            old.unlink()
    vn = 0
    for v in vchosen:
        for img in (BASE / "images" / "val").glob(f"{v}_*.png"):
            for sub, ext in (("images", ".png"), ("labels", ".txt")):
                src = BASE / sub / "val" / (img.stem + ext)
                dst = BASE / sub / "val_sub" / (img.stem + ext)
                if src.exists():
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
            vn += 1
    fast_yaml = BASE / "thyroid_fast.yaml"
    fast_yaml.write_text(
        f"path: {BASE.as_posix()}\ntrain: images/train_sub\nval: images/val_sub\ntest: images/test\n"
        "names:\n  0: nodule\n"
    )

    info = {"chosen_videos": chosen, "n_videos": len(chosen), "n_frames": n,
            "val_videos": vchosen, "val_frames": vn,
            "full_train_frames": total, "target_frac": frac,
            "yaml": str(yaml_path), "fast_yaml": str(fast_yaml)}
    (BASE / "subset_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()