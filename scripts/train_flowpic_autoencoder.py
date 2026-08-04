#!/usr/bin/env python3
"""
Train the unsupervised flowpic autoencoder baseline.

Trains ONLY on normal-labeled train/ windows -- no ddos label is ever seen
during training, which is the entire point of the supervised-vs-unsupervised
comparison (scripts/train_resnext_flowpic_v1.sh is the supervised side,
trained on both classes with full labels).

Early stopping / checkpoint selection uses reconstruction loss on the
normal-labeled portion of val/ -- still unsupervised (no ddos labels used),
just held-out normal traffic to detect overfitting.

Usage: scripts/train_flowpic_autoencoder.py [--data-root ...] [--epochs 50] ...
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowpic_autoencoder_model import ConvAutoencoder, make_image_only_dataset  # noqa: E402


class ReconLossCell(nn.Cell):
    def __init__(self, net: nn.Cell):
        super().__init__()
        self.net = net
        self.mse = nn.MSELoss()

    def construct(self, image):
        recon = self.net(image)
        return self.mse(recon, image)


def evaluate_recon_loss(net: nn.Cell, dataset) -> float:
    net.set_train(False)
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for (image,) in dataset.create_tuple_iterator():
        recon = net(image)
        total += float(mse(recon, image).asnumpy())
        n += 1
    net.set_train(True)
    return total / max(n, 1)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(project_root / "dataset" / "images_flowpic_0p3_validated_v1"))
    ap.add_argument("--output-dir", default=str(project_root / "model" / "flowpic_autoencoder_v1"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--early-stop", type=int, default=8)
    ap.add_argument("--device-target", default="GPU")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)
    ms.set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, train_paths = make_image_only_dataset(
        args.data_root, "train", "normal", args.batch_size, shuffle=True
    )
    val_ds, val_paths = make_image_only_dataset(
        args.data_root, "val", "normal", args.batch_size, shuffle=False, drop_remainder=False
    )
    print(f"train normal images: {len(train_paths)} | val normal images: {len(val_paths)}")
    print(f"steps/epoch: {train_ds.get_dataset_size()}")

    net = ConvAutoencoder()
    n_params = sum(p.size for p in net.trainable_params())
    print(f"model params: {n_params:,}")

    loss_cell = ReconLossCell(net)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
    train_step = nn.TrainOneStepCell(loss_cell, optimizer)
    train_step.set_train()

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    best_ckpt = os.path.join(args.output_dir, "autoencoder_best.ckpt")

    log_path = os.path.join(args.output_dir, "training.log")
    with open(log_path, "a") as logf:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            total, n = 0.0, 0
            for (image,) in train_ds.create_tuple_iterator():
                loss = train_step(image)
                total += float(loss.asnumpy())
                n += 1
            train_loss = total / max(n, 1)
            val_loss = evaluate_recon_loss(net, val_ds)
            dt = time.time() - t0

            line = (
                f"epoch {epoch}/{args.epochs} train_mse={train_loss:.6f} "
                f"val_mse={val_loss:.6f} time={dt:.1f}s"
            )
            print(line)
            logf.write(line + "\n")
            logf.flush()

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
                ms.save_checkpoint(net, best_ckpt)
            else:
                no_improve += 1
                if args.early_stop > 0 and no_improve >= args.early_stop:
                    msg = f"early stopping at epoch {epoch} (best={best_epoch}, best_val_mse={best_val:.6f})"
                    print(msg)
                    logf.write(msg + "\n")
                    break

    print(f"\nBest checkpoint: {best_ckpt} (epoch {best_epoch}, val_mse={best_val:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
