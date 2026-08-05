#!/usr/bin/env python3
"""
Train the Deep SVDD unsupervised flowpic baseline (see deep_svdd_model.py
for the method and why it's a second unsupervised approach alongside the
reconstruction-MSE autoencoder in train_flowpic_autoencoder.py).

Trains ONLY on normal-labeled train/ windows -- no ddos label is ever seen
during training (labels only touch eval_deep_svdd.py's threshold sweep).

Two phases, both early-stopped on normal-only val/:
  1. Pretrain encoder+decoder as a reconstruction autoencoder (val recon MSE).
  2. Fix center c from the pretrained encoder, fine-tune the encoder alone
     to minimize mean squared distance to c (val mean squared distance).

Usage: scripts/train_deep_svdd.py [--data-root ...] [--pretrain-epochs 8]
       [--svdd-epochs 25] ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowpic_autoencoder_model import make_image_only_dataset  # noqa: E402
from deep_svdd_model import (  # noqa: E402
    PretrainAutoencoder,
    SVDDLossCell,
    compute_center,
    embedding_std,
    per_image_distances,
)


class ReconLossCell(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.mse = nn.MSELoss()

    def construct(self, image):
        recon = self.net(image)
        return self.mse(recon, image)


def evaluate_recon_loss(net, dataset) -> float:
    net.set_train(False)
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for (image,) in dataset.create_tuple_iterator():
        recon = net(image)
        loss = mse(recon, image)
        total += float(loss.asnumpy())
        n += 1
    return total / max(n, 1)


def evaluate_svdd_loss(encoder, center: np.ndarray, val_ds) -> float:
    dists = per_image_distances(encoder, center, val_ds)
    return float(dists.mean()) if dists.size else float("nan")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(project_root / "dataset" / "images_flowpic_0p3_validated_v1"))
    ap.add_argument("--output-dir", default=str(project_root / "model" / "flowpic_deep_svdd_v1"))
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--pretrain-epochs", type=int, default=15)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--pretrain-early-stop", type=int, default=6)
    ap.add_argument("--svdd-epochs", type=int, default=30)
    ap.add_argument("--svdd-lr", type=float, default=1e-4)
    ap.add_argument("--svdd-weight-decay", type=float, default=1e-4)
    ap.add_argument("--svdd-early-stop", type=int, default=8)
    ap.add_argument("--center-eps", type=float, default=0.1)
    ap.add_argument(
        "--min-embed-std",
        type=float,
        default=0.05,
        help="Collapse guard: val_dist2 trivially minimizes toward 0 as the encoder "
        "collapses toward a constant embedding, so it can't be used alone to pick or "
        "stop on the best checkpoint. A checkpoint is only accepted as 'best' if "
        "val_embed_std stays at or above this floor; training stops immediately "
        "(not just no-improve-patience) the first epoch it drops below, since "
        "collapse does not recover.",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device-target", default="GPU")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)
    ms.set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training.log")
    logf = open(log_path, "a")

    def log(line: str):
        print(line)
        logf.write(line + "\n")
        logf.flush()

    train_ds, train_paths = make_image_only_dataset(
        args.data_root, "train", "normal", args.batch_size, shuffle=True
    )
    val_ds, val_paths = make_image_only_dataset(
        args.data_root, "val", "normal", args.batch_size, shuffle=False, drop_remainder=False
    )
    # Unshuffled copy of train for center computation (order doesn't matter,
    # but drop_remainder=False so every training image contributes).
    train_ds_ordered, _ = make_image_only_dataset(
        args.data_root, "train", "normal", args.batch_size, shuffle=False, drop_remainder=False
    )
    log(f"train normal images: {len(train_paths)} | val normal images: {len(val_paths)}")

    # ---- Phase 1: autoencoder pretraining ----
    ae = PretrainAutoencoder(args.embed_dim)
    n_params = sum(p.size for p in ae.trainable_params())
    log(f"model params (pretrain AE): {n_params:,} | embed_dim={args.embed_dim}")

    pretrain_step = nn.TrainOneStepCell(
        ReconLossCell(ae), nn.Adam(ae.trainable_params(), learning_rate=args.pretrain_lr)
    )
    pretrain_step.set_train()

    pretrain_ckpt = os.path.join(args.output_dir, "pretrain_ae_best.ckpt")
    best_val, best_epoch, no_improve = float("inf"), -1, 0
    log("\n=== Phase 1: autoencoder pretraining ===")
    for epoch in range(1, args.pretrain_epochs + 1):
        t0 = time.time()
        total, n = 0.0, 0
        for (image,) in train_ds.create_tuple_iterator():
            loss = pretrain_step(image)
            total += float(loss.asnumpy())
            n += 1
        train_loss = total / max(n, 1)
        val_loss = evaluate_recon_loss(ae, val_ds)
        dt = time.time() - t0
        log(f"pretrain epoch {epoch}/{args.pretrain_epochs} train_mse={train_loss:.6f} val_mse={val_loss:.6f} time={dt:.1f}s")

        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            ms.save_checkpoint(ae, pretrain_ckpt)
        else:
            no_improve += 1
            if args.pretrain_early_stop > 0 and no_improve >= args.pretrain_early_stop:
                log(f"pretrain early stopping at epoch {epoch} (best={best_epoch}, best_val_mse={best_val:.6f})")
                break

    # Reload best pretrain epoch's weights (in-memory `ae` may hold a later,
    # worse epoch if early stopping triggered after the best one).
    params = ms.load_checkpoint(pretrain_ckpt)
    ms.load_param_into_net(ae, params)
    encoder = ae.encoder
    log(f"Loaded best pretrain checkpoint: {pretrain_ckpt} (epoch {best_epoch}, val_mse={best_val:.6f})")

    # ---- Phase 2: SVDD fine-tuning ----
    log("\n=== Phase 2: SVDD center init + fine-tuning ===")
    center = compute_center(encoder, train_ds_ordered, eps=args.center_eps)
    center_path = os.path.join(args.output_dir, "center.json")
    with open(center_path, "w") as f:
        json.dump({"center": center.tolist(), "embed_dim": args.embed_dim}, f)
    log(f"Center computed from {len(train_paths)} normal train images, eps-clamp={args.center_eps} -> {center_path}")

    svdd_loss_cell = SVDDLossCell(encoder, center)
    svdd_step = nn.TrainOneStepCell(
        svdd_loss_cell,
        nn.Adam(encoder.trainable_params(), learning_rate=args.svdd_lr, weight_decay=args.svdd_weight_decay),
    )
    svdd_step.set_train()

    svdd_ckpt = os.path.join(args.output_dir, "svdd_encoder_best.ckpt")
    best_val, best_epoch, no_improve = float("inf"), -1, 0
    for epoch in range(1, args.svdd_epochs + 1):
        t0 = time.time()
        total, n = 0.0, 0
        for (image,) in train_ds.create_tuple_iterator():
            loss = svdd_step(image)
            total += float(loss.asnumpy())
            n += 1
        train_loss = total / max(n, 1)
        val_loss = evaluate_svdd_loss(encoder, center, val_ds)
        std = embedding_std(encoder, val_ds)
        dt = time.time() - t0
        collapsed = std < args.min_embed_std
        collapse_flag = f" [COLLAPSE: embed_std < {args.min_embed_std}]" if collapsed else ""
        log(
            f"svdd epoch {epoch}/{args.svdd_epochs} train_dist2={train_loss:.6f} "
            f"val_dist2={val_loss:.6f} val_embed_std={std:.6f} time={dt:.1f}s{collapse_flag}"
        )

        if collapsed:
            log(
                f"svdd stopping at epoch {epoch}: embed_std={std:.6f} fell below "
                f"--min-embed-std={args.min_embed_std} -- collapse does not recover, "
                f"not treating this or later epochs as candidates."
            )
            break

        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            ms.save_checkpoint(encoder, svdd_ckpt)
        else:
            no_improve += 1
            if args.svdd_early_stop > 0 and no_improve >= args.svdd_early_stop:
                log(f"svdd early stopping at epoch {epoch} (best={best_epoch}, best_val_dist2={best_val:.6f})")
                break

    if best_epoch == -1:
        log(
            "\nNo SVDD checkpoint met the embed_std floor -- every epoch collapsed "
            "before a usable one was found. Lower --svdd-lr or raise "
            "--svdd-weight-decay further and retry."
        )
        return 1

    log(f"\nBest SVDD checkpoint: {svdd_ckpt} (epoch {best_epoch}, val_dist2={best_val:.6f})")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
