#!/usr/bin/env python3
"""Load a checkpoint and report confusion matrix + per-class P/R/F1 on val or test.

Uses only the primary 0.3s focal-v2 checkpoint unless HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT=1
(see eval_primary_ckpt.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Repo layout: scripts/ vs notebook/train_resnext.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "notebook") not in sys.path:
    sys.path.insert(0, str(_ROOT / "notebook"))

from eval_primary_ckpt import primary_ckpt_path, require_primary_ckpt_for_eval

import mindspore as ms
from mindspore import ops

import train_resnext as tr


def confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.flat, y_pred.flat):
        cm[int(t), int(p)] += 1
    out = []
    eps = 1e-12
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        out.append((prec, rec, f1))
    return cm, out


def main():
    ap = argparse.ArgumentParser(description="Detailed eval: confusion matrix + per-class metrics.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=_ROOT / "dataset" / "images_per_second_window_0p3",
        help="Must contain train/ for class order (ImageFolder). Val/test read from here unless --eval-root is set.",
    )
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="If set, read val/ or test/ from this directory instead of data-root (matches split-train / per-second training).",
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=primary_ckpt_path(),
        help="Must be the 0.3s focal-v2 best checkpoint (see scripts/eval_primary_ckpt.py).",
    )
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--model-name", default="resnext50_32x4d")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--resize-size", type=int, default=256)
    ap.add_argument(
        "--norm-mean",
        type=float,
        nargs=3,
        default=None,
        help="Per-channel normalization mean (default: ImageNet mean). Must match the value "
        "the checkpoint was TRAINED with, not just the data-root -- e.g. flowpic checkpoints "
        "use dataset-computed stats, not ImageNet.",
    )
    ap.add_argument(
        "--norm-std",
        type=float,
        nargs=3,
        default=None,
        help="Per-channel normalization std (default: ImageNet std; see --norm-mean).",
    )
    ap.add_argument("--device-target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--mode", choices=["GRAPH", "PYNATIVE"], default="GRAPH")
    args = ap.parse_args()
    require_primary_ckpt_for_eval(args.ckpt)

    mode = ms.GRAPH_MODE if args.mode == "GRAPH" else ms.PYNATIVE_MODE
    ms.set_context(mode=mode, device_target=args.device_target, device_id=args.device_id)

    train_dir = args.data_root / "train"
    class_indexing = tr.build_class_indexing(train_dir)
    class_names = sorted(class_indexing.keys(), key=lambda n: class_indexing[n])
    num_classes = len(class_indexing)
    resize = None if args.resize_size == 0 else int(args.resize_size)

    net = tr.create_model(
        args.model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    tr.load_ckpt(net, str(args.ckpt), strict=True)
    net.set_train(False)

    image_root = args.eval_root if args.eval_root is not None else args.data_root
    eval_ds = tr.make_dataset(
        str(image_root),
        args.split,
        args.batch_size,
        False,
        class_indexing,
        args.img_size,
        resize,
        args.num_workers,
        False,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )

    argmax = ops.Argmax(axis=1, output_type=ms.int32)
    preds_list = []
    labels_list = []
    for batch in eval_ds.create_dict_iterator(num_epochs=1):
        imgs = batch["image"]
        lab = batch["label"]
        logits = net(imgs)
        pred = argmax(logits)
        preds_list.append(pred.asnumpy())
        labels_list.append(lab.asnumpy())

    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0)

    cm, per_class = confusion_and_metrics(y_true, y_pred, num_classes)
    acc = float((y_true == y_pred).mean())

    print(f"Checkpoint: {args.ckpt}")
    print(f"Class index from: {train_dir}")
    print(f"Images from: {image_root / args.split}")
    print(f"Split: {args.split}  |  samples: {len(y_true)}")
    print(f"Overall accuracy: {acc:.6f}")
    print()
    print("Confusion matrix [rows=true, cols=pred]:")
    header = "".join(f"{n:>12}" for n in class_names)
    print(f"{'':12}{header}")
    for i, name in enumerate(class_names):
        row = "".join(f"{cm[i, j]:12d}" for j in range(num_classes))
        print(f"{name:12}{row}")
    print()
    print(f"{'class':12} {'precision':>12} {'recall':>12} {'f1':>12}")
    macro_f1 = 0.0
    for i, name in enumerate(class_names):
        p, r, f = per_class[i]
        macro_f1 += f
        print(f"{name:12} {p:12.6f} {r:12.6f} {f:12.6f}")
    macro_f1 /= num_classes
    print(f"{'macro F1':12} {'':12} {'':12} {macro_f1:12.6f}")
    supports = cm.sum(axis=1).astype(np.float64)
    n = float(supports.sum()) + 1e-12
    weighted_f1 = float(sum((supports[i] / n) * per_class[i][2] for i in range(num_classes)))
    print(f"{'weighted F1':12} {'':12} {'':12} {weighted_f1:12.6f}")
    if num_classes == 2:
        rec0, rec1 = per_class[0][1], per_class[1][1]
        bal_acc = 0.5 * (rec0 + rec1)
        print(f"{'balanced acc':12} {'':12} {'':12} {bal_acc:12.6f}")

    # Majority baseline: always predict most frequent class in this split
    majority = int(np.bincount(y_true.astype(np.int64)).argmax())
    baseline_acc = float((y_true == majority).mean())
    print()
    print(f"Majority-class baseline accuracy (always predict '{class_names[majority]}'): {baseline_acc:.6f}")


if __name__ == "__main__":
    main()
