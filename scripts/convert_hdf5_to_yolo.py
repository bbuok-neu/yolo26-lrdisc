"""Convert thyroid ultrasound video HDF5 (image/mask/annot_id/frame_num) to YOLO detection format.

Output layout (under --out):
    images/{train,val,test}/<video>_<frame:04d>.png   (grayscale PNG, native resolution)
    labels/{train,val,test}/<video>_<frame:04d>.txt   (YOLO: cls xc yc w h, normalized; empty file = background)
    thyroid.yaml, split_videos.json, manifest.csv, conversion_stats.json

Split is done at VIDEO level (no frame leakage across splits), greedily balanced by frame count.
Boxes come from connected components of mask > --thresh, filtered by --min-area pixels.
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy import ndimage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", default=r"C:\Users\bbuok\Documents\ChatGPT\甲状腺大模型\公开数据集\dataset.hdf5")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "thyroid_yolo"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    p.add_argument("--thresh", type=int, default=127, help="mask binarization threshold (mask is 0..255)")
    p.add_argument("--min-area", type=int, default=25, help="drop components smaller than this many pixels")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def split_videos(video_frames: dict, ratios, seed):
    """Greedy balanced assignment of whole videos to train/val/test by frame count."""
    rng = np.random.default_rng(seed)
    videos = sorted(video_frames)
    rng.shuffle(videos)
    videos.sort(key=lambda v: -video_frames[v])  # largest first, random tie order
    total = sum(video_frames.values())
    targets = [r * total for r in ratios]
    counts = [0, 0, 0]
    assign = {}
    for v in videos:
        # split with largest remaining deficit relative to target
        deficits = [targets[i] - counts[i] for i in range(3)]
        i = int(np.argmax(deficits))
        assign[v] = ["train", "val", "test"][i]
        counts[i] += video_frames[v]
    return assign, counts


def mask_to_boxes(mask, thresh, min_area, w, h):
    binary = mask > thresh
    if not binary.any():
        return []
    lab, n = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    boxes = []
    slices = ndimage.find_objects(lab)
    areas = np.bincount(lab.ravel())
    for i, sl in enumerate(slices, start=1):
        if sl is None or areas[i] < min_area:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        xc = (x0 + x1) / 2 / w
        yc = (y0 + y1) / 2 / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h
        boxes.append((xc, yc, bw, bh, int(areas[i])))
    return boxes


def main():
    args = parse_args()
    out = Path(args.out)
    for sp in ("train", "val", "test"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    (out.parents[1] / "docs" / "samples").mkdir(parents=True, exist_ok=True)

    f = h5py.File(args.hdf5, "r")
    aids = f["annot_id"][:].astype(str)
    fns = f["frame_num"][:].astype(int)
    n = len(aids)
    videos = [a.rstrip("_") for a in aids]

    video_frames = defaultdict(int)
    for v in videos:
        video_frames[v] += 1
    assign, split_counts = split_videos(video_frames, args.ratios, args.seed)
    print(f"frames={n} videos={len(video_frames)} split frame counts={split_counts}", flush=True)

    # sanity: frames unique per video?
    seen = set()
    dup = 0
    for v, fr in zip(videos, fns):
        key = (v, int(fr))
        if key in seen:
            dup += 1
        seen.add(key)
    if dup:
        print(f"WARNING: {dup} duplicate (video,frame) pairs", flush=True)

    img_ds, msk_ds = f["image"], f["mask"]
    H, W = img_ds.shape[1], img_ds.shape[2]

    stats = {
        "n_frames": n, "n_videos": len(video_frames), "image_hw": [H, W],
        "thresh": args.thresh, "min_area": args.min_area, "seed": args.seed,
        "split_videos": {sp: 0 for sp in ("train", "val", "test")},
        "split_frames": {sp: 0 for sp in ("train", "val", "test")},
        "split_boxes": {sp: 0 for sp in ("train", "val", "test")},
        "empty_frames": 0, "multi_box_frames": 0, "dropped_small_components": 0,
        "rel_box_areas": [],
    }
    for v, sp in assign.items():
        stats["split_videos"][sp] += 1

    manifest_rows = []
    sample_idx = set(np.linspace(0, n - 1, 6).astype(int).tolist())
    t0 = time.time()

    def write_png(path, img):
        cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for i in range(n):
            v, fr = videos[i], int(fns[i])
            sp = assign[v]
            stem = f"{v}_{fr:04d}"
            img = img_ds[i]
            msk = msk_ds[i]
            boxes = mask_to_boxes(msk, args.thresh, args.min_area, W, H)

            png_path = out / "images" / sp / f"{stem}.png"
            futures.append(pool.submit(write_png, png_path, img))
            lbl_path = out / "labels" / sp / f"{stem}.txt"
            with open(lbl_path, "w") as lf:
                for xc, yc, bw, bh, _ in boxes:
                    lf.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            stats["split_frames"][sp] += 1
            stats["split_boxes"][sp] += len(boxes)
            if not boxes:
                stats["empty_frames"] += 1
            if len(boxes) > 1:
                stats["multi_box_frames"] += 1
            binary = msk > args.thresh
            if binary.any():
                lab, nc = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
                if nc > len(boxes):
                    stats["dropped_small_components"] += nc - len(boxes)
            for xc, yc, bw, bh, area in boxes:
                stats["rel_box_areas"].append(bw * bh)
            manifest_rows.append([v, fr, sp, stem, len(boxes), sum(b[4] for b in boxes)])

            if i in sample_idx:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for xc, yc, bw, bh, _ in boxes:
                    x0 = int((xc - bw / 2) * W); y0 = int((yc - bh / 2) * H)
                    x1 = int((xc + bw / 2) * W); y1 = int((yc + bh / 2) * H)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 3)
                cv2.imwrite(str(out.parents[1] / "docs" / "samples" / f"{stem}_vis.jpg"), vis)

            if (i + 1) % 1000 == 0:
                el = time.time() - t0
                print(f"{i+1}/{n} frames  {el:.0f}s  ({(i+1)/el:.1f} fps)", flush=True)
        for fu in futures:
            fu.result()

    areas = np.array(stats["rel_box_areas"])
    stats["rel_box_area_percentiles"] = {
        str(q): float(np.percentile(areas, q)) for q in (1, 5, 25, 50, 75, 95, 99)
    } if len(areas) else {}
    stats["rel_box_areas"] = None  # don't dump 17k floats
    stats["elapsed_sec"] = round(time.time() - t0, 1)

    with open(out / "thyroid.yaml", "w") as yf:
        yf.write(
            f"path: {out.as_posix()}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: nodule\n"
        )
    with open(out / "split_videos.json", "w") as sf:
        json.dump({sp: sorted(v for v, s in assign.items() if s == sp) for sp in ("train", "val", "test")}, sf, indent=1)
    with open(out / "manifest.csv", "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["video", "frame", "split", "stem", "n_boxes", "total_area_px"])
        w.writerows(manifest_rows)
    with open(out / "conversion_stats.json", "w") as sf:
        json.dump(stats, sf, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "rel_box_areas"}, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()