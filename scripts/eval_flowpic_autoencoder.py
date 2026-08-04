#!/usr/bin/env python3
"""
Threshold calibration + test evaluation for the unsupervised flowpic
autoencoder baseline (scripts/train_flowpic_autoencoder.py).

The only place labels touch this pipeline at all: picking a reconstruction-
error threshold on val (best-F1 sweep) and reporting metrics. The model
itself never saw a ddos label during training.

Usage: scripts/eval_flowpic_autoencoder.py [--ckpt ...] [--data-root ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowpic_autoencoder_model import ConvAutoencoder, make_image_only_dataset  # noqa: E402


def per_image_recon_errors(net, data_root: str, split: str, class_name: str, batch_size: int = 64):
    dataset, paths = make_image_only_dataset(
        data_root, split, class_name, batch_size, shuffle=False, drop_remainder=False
    )
    errors = []
    for (image,) in dataset.create_tuple_iterator():
        recon = net(image)
        diff = (recon - image) ** 2
        # per-image MSE: mean over C,H,W (axes 1,2,3)
        per_image = ms.ops.mean(diff, axis=(1, 2, 3))
        errors.append(per_image.asnumpy())
    return np.concatenate(errors) if errors else np.array([]), paths


def collect_scores(net, data_root: str, split: str, batch_size: int):
    ddos_err, ddos_paths = per_image_recon_errors(net, data_root, split, "ddos", batch_size)
    normal_err, normal_paths = per_image_recon_errors(net, data_root, split, "normal", batch_size)
    scores = np.concatenate([ddos_err, normal_err])
    labels = np.concatenate([np.ones(len(ddos_err), dtype=np.int64), np.zeros(len(normal_err), dtype=np.int64)])
    return scores, labels


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(scores)
    candidates = np.unique(scores)
    best_t, best_f1, best_stats = None, -1.0, None
    for t in candidates:
        pred = (scores >= t).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t, best_stats = f1, t, (p, r, f1)
    return float(best_t), best_stats


def report_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    acc = float((pred == labels).mean())
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    roc_auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    pr_auc = float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision_ddos": float(p),
        "recall_ddos": float(r),
        "f1_ddos": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_ddos": int(labels.sum()),
        "n_normal": int((labels == 0).sum()),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(project_root / "dataset" / "images_flowpic_0p3_validated_v1"))
    ap.add_argument("--ckpt", default=str(project_root / "model" / "flowpic_autoencoder_v1" / "autoencoder_best.ckpt"))
    ap.add_argument("--output-dir", default=str(project_root / "model" / "flowpic_autoencoder_v1"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device-target", default="GPU")
    args = ap.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)

    net = ConvAutoencoder()
    params = load_checkpoint(args.ckpt)
    load_param_into_net(net, params)
    net.set_train(False)

    t0 = time.time()
    val_scores, val_labels = collect_scores(net, args.data_root, "val", args.batch_size)
    print(f"val scored: {len(val_labels)} images ({int(val_labels.sum())} ddos) in {time.time()-t0:.1f}s")

    threshold, val_stats = best_f1_threshold(val_scores, val_labels)
    val_report = report_at_threshold(val_scores, val_labels, threshold)
    print(f"chosen threshold (best F1 on val): {threshold:.6f}")
    print("val @ threshold:", json.dumps(val_report, indent=2))

    t1 = time.time()
    test_scores, test_labels = collect_scores(net, args.data_root, "test", args.batch_size)
    print(f"test scored: {len(test_labels)} images ({int(test_labels.sum())} ddos) in {time.time()-t1:.1f}s")

    test_report = report_at_threshold(test_scores, test_labels, threshold)
    print("\n=== TEST SET RESULT (threshold calibrated on val only) ===")
    print(json.dumps(test_report, indent=2))

    out = {
        "model": "flowpic_autoencoder_v1 (unsupervised, trained on normal-only windows)",
        "checkpoint": args.ckpt,
        "threshold_source": "val, best-F1 sweep",
        "val": val_report,
        "test": test_report,
    }
    out_path = Path(args.output_dir) / "eval_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
