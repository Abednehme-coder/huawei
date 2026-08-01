#!/usr/bin/env python3
"""
Compute per-channel mean/std (in [0,1] scale) over a flowpic image dataset's
train/ split, for use as --norm-mean/--norm-std in train_resnext_flowpic_v1.sh
instead of ImageNet's stats.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True, type=Path, help="Dataset root containing train/")
    args = parser.parse_args()

    train_root = args.images_root / "train"
    paths = [
        p
        for cls_dir in train_root.iterdir()
        if cls_dir.is_dir()
        for p in cls_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        raise SystemExit(f"No images found under {train_root}")

    sum_ = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
        sum_ += arr.sum(axis=(0, 1))
        sum_sq += (arr**2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]

    mean = sum_ / n_pixels
    var = sum_sq / n_pixels - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))

    print(f"n_images={len(paths)}")
    print(f"mean = {mean.round(4).tolist()}")
    print(f"std  = {std.round(4).tolist()}")
    print()
    print(f'NORM_MEAN="{mean[0]:.4f} {mean[1]:.4f} {mean[2]:.4f}"')
    print(f'NORM_STD="{std[0]:.4f} {std[1]:.4f} {std[2]:.4f}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
