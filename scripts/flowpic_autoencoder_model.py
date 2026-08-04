"""Shared conv-autoencoder architecture + data helpers for the unsupervised
flowpic baseline (scripts/train_flowpic_autoencoder.py,
scripts/eval_flowpic_autoencoder.py).

Unsupervised by construction: trained ONLY on normal-labeled 0.3s flowpic
windows (never shown a ddos-labeled image during training), reconstructing
its own input. At eval time, per-image reconstruction MSE is used as an
anomaly score -- a well-reconstructed image looks like the normal traffic
distribution it was trained on; a poorly-reconstructed one doesn't.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

IMG_SIZE = 64


class ConvAutoencoder(nn.Cell):
    """64x64x3 -> 4x4x256 bottleneck -> 64x64x3. Sigmoid output: trained
    against Rescale(1/255) input, no ImageNet-style mean/std normalization,
    so reconstruction target and output share the same [0,1] range."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.SequentialCell([
            nn.Conv2d(3, 32, 3, stride=2, pad_mode="same", has_bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, pad_mode="same", has_bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, pad_mode="same", has_bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, pad_mode="same", has_bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ])
        self.decoder = nn.SequentialCell([
            nn.Conv2dTranspose(256, 128, 3, stride=2, pad_mode="same", has_bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2dTranspose(128, 64, 3, stride=2, pad_mode="same", has_bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2dTranspose(64, 32, 3, stride=2, pad_mode="same", has_bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2dTranspose(32, 3, 3, stride=2, pad_mode="same", has_bias=True),
            nn.Sigmoid(),
        ])

    def construct(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def _file_list(data_root: str, split: str, class_name: str) -> list[str]:
    d = Path(data_root) / split / class_name
    if not d.is_dir():
        raise FileNotFoundError(d)
    return sorted(str(p) for p in d.glob("*.png"))


class _PathListSource:
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            return (np.frombuffer(f.read(), dtype=np.uint8),)

    def __len__(self):
        return len(self.paths)


def make_image_only_dataset(
    data_root: str,
    split: str,
    class_name: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    drop_remainder: bool = True,
):
    """Loads images from a single class subdir only, no labels -- used both
    for unsupervised training (normal only) and for per-class reconstruction
    scoring at eval time (call once per class, concat scores after)."""
    paths = _file_list(data_root, split, class_name)
    if not paths:
        raise RuntimeError(f"No images found under {data_root}/{split}/{class_name}")
    source = _PathListSource(paths)
    dataset = ds.GeneratorDataset(source, column_names=["image"], shuffle=shuffle)
    ops_list = [
        vision.Decode(),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.HWC2CHW(),
    ]
    dataset = dataset.map(operations=ops_list, input_columns="image", num_parallel_workers=num_workers)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset, paths
