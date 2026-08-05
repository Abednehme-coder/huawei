"""Deep SVDD (Ruff et al., "Deep One-Class Classification", ICML 2018)
unsupervised baseline for the flowpic 0.3s images -- a second unsupervised
approach alongside scripts/flowpic_autoencoder_model.py's reconstruction-MSE
autoencoder.

Why a second unsupervised baseline: the reconstruction-MSE autoencoder's
core assumption (anomalies reconstruct worse than normal traffic) doesn't
hold on this dataset -- ddos-labeled flowpic windows are ~3.7x *sparser*
than normal ones (a SYN flood in a 0.3s window is a uniform, concentrated
pattern; normal traffic is busier), so the autoencoder -- which converges to
reconstructing sparse/background patterns well -- scores ddos as *more*
normal, not less (val ROC AUC 0.288, see model/flowpic_autoencoder_v1/).

Deep SVDD scores anomalies differently: it maps normal training data into a
tight hypersphere in embedding space (minimizing mean squared distance to a
fixed center c) rather than requiring pixel-level reconstruction fidelity.
Whether ddos windows are sparser or busier than normal, what matters is
whether their *learned embedding* falls inside or outside the hypersphere
carved out by normal traffic -- a different, less pixel-density-coupled
notion of "anomaly" than reconstruction error.

Two known collapse modes this implementation guards against per the paper:
  1. Trivial "map everything to c" solution -- prevented structurally by
     making the SVDD encoder bias-free (no conv/dense bias terms, no
     BatchNorm affine gamma/beta) so a constant output isn't representable
     regardless of input.
  2. A center with an all/mostly-zero coordinate -- prevented by clamping
     any |c_j| below `eps` away from zero (sign-preserving) before it's used
     as the fixed target, per the paper's initialization procedure.

Two-phase training (see scripts/train_deep_svdd.py):
  1. Pretrain the bias-free encoder as part of a reconstruction autoencoder
     on normal-only train windows (standard Deep SVDD initialization --
     gives the encoder a non-trivial starting point instead of random init).
  2. Fix center c = mean embedding of the pretrained encoder over the full
     normal train set (with the eps-clamp above), then fine-tune the SAME
     encoder to minimize mean squared distance to c.

At eval time (scripts/eval_deep_svdd.py), anomaly score = ||phi(x) - c||^2;
higher = farther from the normal-traffic hypersphere = more anomalous.
"""
from __future__ import annotations

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

IMG_SIZE = 64
DEFAULT_EMBED_DIM = 32


class SVDDEncoder(nn.Cell):
    """64x64x3 -> 4x4x256 conv stack -> flatten -> bias-free Dense bottleneck.

    Same channel progression as flowpic_autoencoder_model.ConvAutoencoder's
    encoder for a like-for-like architectural comparison, but with every
    bias term removed (has_bias=False on conv/dense, affine=False on
    BatchNorm) -- required so the network can't trivially collapse to a
    constant embedding regardless of input (see module docstring)."""

    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.SequentialCell([
            nn.Conv2d(3, 32, 3, stride=2, pad_mode="same", has_bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, stride=2, pad_mode="same", has_bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, stride=2, pad_mode="same", has_bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, stride=2, pad_mode="same", has_bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.LeakyReLU(0.1),
        ])
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(256 * 4 * 4, embed_dim, has_bias=False)

    def construct(self, x):
        z = self.conv(x)
        z = self.flatten(z)
        return self.fc(z)


class SVDDDecoder(nn.Cell):
    """Mirrors SVDDEncoder, bias-free, for autoencoder pretraining only --
    discarded after phase 1, never used for SVDD scoring."""

    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        self.fc = nn.Dense(embed_dim, 256 * 4 * 4, has_bias=False)
        self.reshape = ops.Reshape()
        self.deconv = nn.SequentialCell([
            nn.Conv2dTranspose(256, 128, 3, stride=2, pad_mode="same", has_bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.1),
            nn.Conv2dTranspose(128, 64, 3, stride=2, pad_mode="same", has_bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.1),
            nn.Conv2dTranspose(64, 32, 3, stride=2, pad_mode="same", has_bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(0.1),
            nn.Conv2dTranspose(32, 3, 3, stride=2, pad_mode="same", has_bias=False),
            nn.Sigmoid(),
        ])

    def construct(self, z):
        x = self.fc(z)
        x = self.reshape(x, (-1, 256, 4, 4))
        return self.deconv(x)


class PretrainAutoencoder(nn.Cell):
    """Phase-1-only wrapper: real submodule `self.encoder` is the exact
    SVDDEncoder instance handed off to phase 2 (by reference, in-process --
    see train_deep_svdd.py), not a separately-constructed lookalike."""

    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        self.encoder = SVDDEncoder(embed_dim)
        self.decoder = SVDDDecoder(embed_dim)

    def construct(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def compute_center(encoder: nn.Cell, dataset, eps: float = 0.1) -> np.ndarray:
    """Mean embedding over `dataset` under the current (pretrained) encoder,
    with any near-zero coordinate pushed to +-eps (sign-preserving; zero
    itself pushed positive) -- per Ruff et al., prevents the trivial
    "collapse toward a zero coordinate of c" solution during phase 2."""
    encoder.set_train(False)
    total = None
    n = 0
    for (image,) in dataset.create_tuple_iterator():
        z = encoder(image).asnumpy()
        total = z.sum(axis=0) if total is None else total + z.sum(axis=0)
        n += z.shape[0]
    c = total / max(n, 1)
    small = np.abs(c) < eps
    c[small & (c >= 0)] = eps
    c[small & (c < 0)] = -eps
    return c.astype(np.float32)


class SVDDLossCell(nn.Cell):
    """Mean squared distance from each embedding to the fixed center c.
    Weight decay (L2 on encoder params) is applied via the optimizer, not
    here, matching the reconstruction autoencoder's convention."""

    def __init__(self, encoder: nn.Cell, center: np.ndarray):
        super().__init__()
        self.encoder = encoder
        self.center = ms.Tensor(center, dtype=ms.float32)

    def construct(self, image):
        z = self.encoder(image)
        diff = z - self.center
        dist2 = ops.reduce_sum(diff * diff, 1)
        return ops.reduce_mean(dist2)


def per_image_distances(encoder: nn.Cell, center: np.ndarray, dataset) -> np.ndarray:
    """Squared distance to center per image -- the anomaly score. Higher =
    farther from the normal-traffic hypersphere = more anomalous."""
    encoder.set_train(False)
    center_t = ms.Tensor(center, dtype=ms.float32)
    out = []
    for (image,) in dataset.create_tuple_iterator():
        z = encoder(image)
        diff = z - center_t
        dist2 = ops.reduce_sum(diff * diff, 1)
        out.append(dist2.asnumpy())
    return np.concatenate(out) if out else np.array([])


def embedding_std(encoder: nn.Cell, dataset) -> float:
    """Mean per-dimension std of embeddings over `dataset` -- a collapse
    diagnostic (not a stopping criterion). If this drops toward ~0 the
    encoder has mapped everything to (near) the same point, which trivially
    minimizes the SVDD loss without learning anything -- print a warning in
    the training loop if seen, don't silently trust a low val loss alone."""
    encoder.set_train(False)
    embeds = []
    for (image,) in dataset.create_tuple_iterator():
        embeds.append(encoder(image).asnumpy())
    if not embeds:
        return float("nan")
    return float(np.concatenate(embeds, axis=0).std(axis=0).mean())
