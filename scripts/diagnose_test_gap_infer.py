#!/usr/bin/env python3
"""Run the flowpic_0p3_validated_v1_continued checkpoint on the test images
for the two flagged pcap stems (SAT-01-12-2018_0617 / _0619) and dump
per-image predictions with bucket IDs, so they can be cross-referenced
against the official CICFlowMeter Syn.csv labels.

Scratch/one-off script for the 2026-08-05 CSV cross-check (see
HANDOVER_2026-08-05.md section 3 / WORK_LOG.md). Not part of the regular
eval suite.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "notebook"))

import mindspore as ms
from mindspore import Tensor
import train_resnext as tr

DATA_ROOT = _ROOT / "dataset" / "images_flowpic_0p3_validated_v1"
CKPT = _ROOT / "model" / "flowpic_0p3_validated_v1_continued" / "resnext50_32x4d_best.ckpt"
NORM_MEAN = [0.0071, 0.0003, 0.0004]
NORM_STD = [0.0592, 0.0157, 0.0145]
IMG_SIZE = 224
RESIZE_SIZE = 256
STEMS = ["SAT-01-12-2018_0617", "SAT-01-12-2018_0619"]
OUT_CSV = _ROOT / "dataset" / "CIC_official_labels" / "flagged_windows_predictions.csv"

BUCKET_RE = re.compile(r"_b(\d+)\.png$")


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    # resize shorter-matching then center-crop, mirroring make_transforms()
    img = img.resize((RESIZE_SIZE, RESIZE_SIZE), Image.BILINEAR)
    left = (RESIZE_SIZE - IMG_SIZE) // 2
    top = (RESIZE_SIZE - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std = np.array(NORM_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr


def main():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU", device_id=0)

    train_dir = DATA_ROOT / "train"
    class_indexing = tr.build_class_indexing(train_dir)
    class_names = sorted(class_indexing.keys(), key=lambda n: class_indexing[n])
    print(f"class_indexing: {class_indexing}")

    net = tr.create_model("resnext50_32x4d", pretrained=False, num_classes=len(class_indexing))
    tr.load_ckpt(net, str(CKPT), strict=True)
    net.set_train(False)

    rows = []
    for true_label in ["normal", "ddos"]:
        class_dir = DATA_ROOT / "test" / true_label
        for stem in STEMS:
            paths = sorted(class_dir.glob(f"{stem}_b*.png"))
            for p in paths:
                m = BUCKET_RE.search(p.name)
                bucket_id = m.group(1) if m else ""
                rows.append({"path": str(p), "stem": stem, "bucket_id": bucket_id, "true_label": true_label})

    print(f"Total flagged-stem test images: {len(rows)}")

    batch_size = 64
    preds = []
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        batch = np.stack([preprocess(Path(r["path"])) for r in batch_rows], axis=0)
        logits = net(Tensor(batch, ms.float32))
        pred_idx = logits.asnumpy().argmax(axis=1)
        preds.extend(pred_idx.tolist())
        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch_rows)}/{len(rows)}", flush=True)

    for r, p in zip(rows, preds):
        r["pred_label"] = class_names[p]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "stem", "bucket_id", "true_label", "pred_label"])
        writer.writeheader()
        writer.writerows(rows)

    n_fp = sum(1 for r in rows if r["true_label"] == "normal" and r["pred_label"] == "ddos")
    n_normal = sum(1 for r in rows if r["true_label"] == "normal")
    print(f"False positives (true=normal, pred=ddos) in flagged stems: {n_fp}/{n_normal}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
