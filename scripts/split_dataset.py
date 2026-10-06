"""Split a single-folder dataset into train/val/test folders (stratified, optionally group-aware).

Why group-aware? If several photos come from the same leaf/plant, a random split puts
near-duplicates in both train and test and inflates the results. Provide a regex that
extracts a group id from the file name and all images of a group stay together.

Examples:
    python scripts/split_dataset.py --src raw_data --dst data/split
    python scripts/split_dataset.py --src raw_data --dst data/split --group_regex "^(leaf\\d+)_"
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def carve(indices, labels, groups, frac, seed):
    n_splits = max(2, round(1 / frac))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, held = next(iter(skf.split(indices, labels, groups)))
    return indices[rest], indices[held]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--group_regex", default=None)
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    classes = sorted(d.name for d in src.iterdir() if d.is_dir())
    files, labels, groups = [], [], []
    for ci, c in enumerate(classes):
        for f in sorted((src / c).rglob("*")):
            if f.suffix.lower() not in EXTS:
                continue
            key = str(f)
            if args.group_regex:
                m = re.search(args.group_regex, f.name)
                if m:
                    key = m.group(1) if m.groups() else m.group(0)
            files.append(f)
            labels.append(ci)
            groups.append(f"{c}/{key}")
    labels, groups = np.array(labels), np.array(groups)
    idx = np.arange(len(files))
    print(f"{len(files)} images, {len(classes)} classes, {len(set(groups))} groups")

    trainval, test = carve(idx, labels, groups, args.test_size, args.seed)
    rel_val = args.val_size / (1 - args.test_size)
    tr_sub, va_sub = carve(np.arange(len(trainval)), labels[trainval], groups[trainval],
                           rel_val, args.seed)
    train, val = trainval[tr_sub], trainval[va_sub]

    for name, ids in [("train", train), ("val", val), ("test", test)]:
        for i in ids:
            target = dst / name / classes[labels[i]] / files[i].name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target = target.with_name(f"{target.stem}_{i}{target.suffix}")
            if args.mode == "copy":
                shutil.copy2(files[i], target)
            else:
                target.symlink_to(files[i].resolve())
        print(f"{name:5s}: {len(ids)} images")
    print(f"Done -> {dst}")


if __name__ == "__main__":
    main()
