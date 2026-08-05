#!/usr/bin/env python3
"""
Threshold calibration + test evaluation for the Deep SVDD unsupervised
flowpic baseline (scripts/train_deep_svdd.py). Mirrors
eval_flowpic_autoencoder.py's val-threshold-sweep + test-report structure
exactly, for direct comparability between the two unsupervised baselines --
only the scoring function differs (squared distance to a learned center,
not reconstruction MSE).

The only place labels touch this pipeline at all: picking a distance
threshold on val (best-F1 sweep) and reporting metrics. The encoder itself
never saw a ddos label during training.

Usage: scripts/eval_deep_svdd.py [--ckpt ...] [--center ...] [--data-root ...]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowpic_autoencoder_model import make_image_only_dataset  # noqa: E402
from deep_svdd_model import SVDDEncoder, per_image_distances  # noqa: E402
from eval_flowpic_autoencoder import best_f1_threshold, report_at_threshold  # noqa: E402


def collect_scores(net, center: np.ndarray, data_root: str, split: str, batch_size: int):
    ddos_ds, ddos_paths = make_image_only_dataset(data_root, split, "ddos", batch_size, shuffle=False, drop_remainder=False)
    ddos_dist = per_image_distances(net, center, ddos_ds)
    normal_ds, normal_paths = make_image_only_dataset(data_root, split, "normal", batch_size, shuffle=False, drop_remainder=False)
    normal_dist = per_image_distances(net, center, normal_ds)
    scores = np.concatenate([ddos_dist, normal_dist])
    labels = np.concatenate([np.ones(len(ddos_dist), dtype=np.int64), np.zeros(len(normal_dist), dtype=np.int64)])
    return scores, labels


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(project_root / "dataset" / "images_flowpic_0p3_validated_v1"))
    ap.add_argument("--ckpt", default=str(project_root / "model" / "flowpic_deep_svdd_v1" / "svdd_encoder_best.ckpt"))
    ap.add_argument("--center", default=str(project_root / "model" / "flowpic_deep_svdd_v1" / "center.json"))
    ap.add_argument("--output-dir", default=str(project_root / "model" / "flowpic_deep_svdd_v1"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device-target", default="GPU")
    args = ap.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)

    with open(args.center) as f:
        center_data = json.load(f)
    center = np.array(center_data["center"], dtype=np.float32)
    embed_dim = int(center_data["embed_dim"])

    net = SVDDEncoder(embed_dim)
    params = load_checkpoint(args.ckpt)
    load_param_into_net(net, params)
    net.set_train(False)

    t0 = time.time()
    val_scores, val_labels = collect_scores(net, center, args.data_root, "val", args.batch_size)
    print(f"val scored: {len(val_labels)} images ({int(val_labels.sum())} ddos) in {time.time()-t0:.1f}s")

    threshold, val_stats = best_f1_threshold(val_scores, val_labels)
    val_report = report_at_threshold(val_scores, val_labels, threshold)
    print(f"chosen threshold (best F1 on val): {threshold:.6f}")
    print("val @ threshold:", json.dumps(val_report, indent=2))

    t1 = time.time()
    test_scores, test_labels = collect_scores(net, center, args.data_root, "test", args.batch_size)
    print(f"test scored: {len(test_labels)} images ({int(test_labels.sum())} ddos) in {time.time()-t1:.1f}s")

    test_report = report_at_threshold(test_scores, test_labels, threshold)
    print("\n=== TEST SET RESULT (threshold calibrated on val only) ===")
    print(json.dumps(test_report, indent=2))

    out = {
        "model": "flowpic_deep_svdd_v1 (unsupervised, trained on normal-only windows)",
        "checkpoint": args.ckpt,
        "center": args.center,
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
