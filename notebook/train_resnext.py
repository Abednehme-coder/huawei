import argparse
import math
import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import mindspore as ms
from mindspore import nn, ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.train import Model
try:
    from mindspore.train.metrics import Metric
except Exception:  # pragma: no cover
    try:
        from mindspore.train.metrics.metric import Metric  # type: ignore
    except Exception:  # pragma: no cover
        Metric = object  # type: ignore
from mindspore.train.callback import (
    Callback,
    LossMonitor,
    TimeMonitor,
    ModelCheckpoint,
    CheckpointConfig,
)
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

# MindSpore 1.8 lacks nn.SiLU; add a minimal implementation for mindcv.
if not hasattr(nn, "SiLU"):
    class SiLU(nn.Cell):
        def construct(self, x):
            return ops.Sigmoid()(x) * x
    nn.SiLU = SiLU  # type: ignore

from mindcv.models import create_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "dataset").is_dir() or (parent / "model").is_dir() or (parent / "notebook").is_dir():
            return parent
    return start


def default_data_root(project_root: Path) -> str:
    """Prefer 0.3s window images, then legacy per-packet images, else dataset/."""
    w03 = project_root / "dataset" / "images_per_second_window_0p3"
    if (w03 / "train").is_dir():
        return str(w03)
    images = project_root / "dataset" / "images"
    if (images / "train").is_dir():
        return str(images)
    return str(project_root / "dataset")


def build_class_indexing(train_dir: Path) -> dict:
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise FileNotFoundError(f"No class subfolders found in: {train_dir}")
    return {name: idx for idx, name in enumerate(class_names)}


def count_images_per_class(train_dir: Path, class_indexing: dict) -> np.ndarray:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    counts = np.zeros(len(class_indexing), dtype=np.int64)
    for class_name, class_idx in class_indexing.items():
        class_path = train_dir / class_name
        if not class_path.is_dir():
            continue
        counts[class_idx] = sum(
            1
            for p in class_path.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        )
    return counts


def compute_class_weights(class_counts: np.ndarray, minority_boost: float = 1.0) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=np.float32)
    counts = np.maximum(counts, 1.0)
    total = float(np.sum(counts))
    weights = total / (len(counts) * counts)
    if minority_boost and minority_boost != 1.0:
        mi = int(np.argmin(counts))
        weights[mi] *= float(minority_boost)
    weights = weights / float(np.mean(weights))
    return weights.astype(np.float32)


def build_learning_rate_schedule(
    schedule: str,
    max_lr: float,
    min_lr: float,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> Union[float, list]:
    """Per-step LR list for cosine (+ optional linear warmup), or scalar if disabled."""
    if schedule != "cosine" or steps_per_epoch <= 0:
        return max_lr
    total_step = int(steps_per_epoch * epochs)
    if total_step <= 0:
        return max_lr
    warmup_steps = int(warmup_epochs * steps_per_epoch) if warmup_epochs > 0 else 0
    warmup_steps = min(max(warmup_steps, 0), total_step)
    if warmup_steps == 0:
        return nn.cosine_decay_lr(min_lr, max_lr, total_step, steps_per_epoch, epochs)
    decay_steps = max(total_step - warmup_steps, 1)
    lrs: list[float] = []
    for i in range(total_step):
        if i < warmup_steps:
            lrs.append(max_lr * float(i + 1) / float(max(warmup_steps, 1)))
        else:
            j = i - warmup_steps
            progress = j / float(max(decay_steps - 1, 1))
            c = 0.5 * (1.0 + math.cos(math.pi * progress))
            lrs.append(min_lr + (max_lr - min_lr) * c)
    return lrs


def make_transforms(
    split: str,
    img_size: int,
    resize_size: Optional[int],
    augment: bool,
    crop_scale_min: float = 0.7,
    crop_scale_max: float = 1.0,
    norm_mean: Optional[list] = None,
    norm_std: Optional[list] = None,
):
    def to_rgb_np(img):
        # img: HWC numpy array
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img

    ops_list = [
        vision.Decode(),
        to_rgb_np,  # ensure 3 channels
    ]

    if split == "train" and augment:
        if hasattr(vision, "RandomResizedCrop"):
            ops_list.append(
                vision.RandomResizedCrop(img_size, scale=(crop_scale_min, crop_scale_max))
            )
        else:
            ops_list.append(vision.Resize((img_size, img_size)))
        if hasattr(vision, "RandomHorizontalFlip"):
            ops_list.append(vision.RandomHorizontalFlip(prob=0.5))
    else:
        if resize_size is not None and resize_size != img_size:
            ops_list.append(vision.Resize((resize_size, resize_size)))
            if hasattr(vision, "CenterCrop"):
                ops_list.append(vision.CenterCrop(img_size))
            else:
                ops_list.append(vision.Resize((img_size, img_size)))
        else:
            ops_list += [vision.Resize((img_size, img_size))]

    # IMPORTANT: rescale to [0,1] before normalization.
    mean = norm_mean if norm_mean is not None else IMAGENET_MEAN
    std = norm_std if norm_std is not None else IMAGENET_STD
    ops_list += [
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=mean, std=std, is_hwc=True),
        vision.HWC2CHW(),
    ]
    return ops_list


def make_dataset(
    data_root: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    class_indexing: Optional[dict],
    img_size: int,
    resize_size: Optional[int],
    num_workers: int,
    augment: bool,
    crop_scale_min: float = 0.7,
    crop_scale_max: float = 1.0,
    norm_mean: Optional[list] = None,
    norm_std: Optional[list] = None,
):
    split_dir = Path(data_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    try:
        dataset = ds.ImageFolderDataset(str(split_dir), shuffle=shuffle, class_indexing=class_indexing)
    except TypeError:
        dataset = ds.ImageFolderDataset(str(split_dir), shuffle=shuffle)

    dataset = dataset.map(
        operations=make_transforms(
            split, img_size, resize_size, augment, crop_scale_min, crop_scale_max, norm_mean, norm_std
        ),
        input_columns="image",
        num_parallel_workers=num_workers,
    )
    dataset = dataset.map(
        operations=transforms.TypeCast(ms.int32),
        input_columns="label",
        num_parallel_workers=num_workers,
    )
    dataset = dataset.batch(batch_size, drop_remainder=(split == "train"))
    return dataset

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self.correct = 0
        self.total = 0

    def update(self, *inputs):
        y_pred, y_true = inputs
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        if hasattr(y_pred, "asnumpy"):
            y_pred = y_pred.asnumpy()
        if hasattr(y_true, "asnumpy"):
            y_true = y_true.asnumpy()
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        self.correct += int(np.sum(y_pred == y_true))
        self.total += int(y_true.size)

    def eval(self):
        return float(self.correct / self.total) if self.total else 0.0


class F1Score(Metric):
    def __init__(self, num_classes: int, average: str = "macro", eps: float = 1e-12):
        super().__init__()
        self.num_classes = int(num_classes)
        self.average = average
        self.eps = float(eps)
        self.clear()

    def clear(self):
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, *inputs):
        y_pred, y_true = inputs
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        if hasattr(y_pred, "asnumpy"):
            y_pred = y_pred.asnumpy()
        if hasattr(y_true, "asnumpy"):
            y_true = y_true.asnumpy()

        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        for c in range(self.num_classes):
            pred_c = y_pred == c
            true_c = y_true == c
            self.tp[c] += int(np.sum(pred_c & true_c))
            self.fp[c] += int(np.sum(pred_c & ~true_c))
            self.fn[c] += int(np.sum(~pred_c & true_c))

    def eval(self):
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)

        if self.average == "macro":
            return float(np.mean(f1))
        if self.average == "weighted":
            support = self.tp + self.fn
            total = float(np.sum(support))
            if total == 0.0:
                return 0.0
            return float(np.sum(f1 * (support / total)))

        raise ValueError(f"Unsupported average: {self.average}")


class F1SingleClass(Metric):
    """Macro F1 restricted to one class index (e.g. 0 = ddos when classes sorted as ddos, normal)."""

    def __init__(self, num_classes: int, class_index: int, eps: float = 1e-12):
        super().__init__()
        self.num_classes = int(num_classes)
        self.class_index = int(class_index)
        self.eps = float(eps)
        self.clear()

    def clear(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, *inputs):
        y_pred, y_true = inputs
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        if hasattr(y_pred, "asnumpy"):
            y_pred = y_pred.asnumpy()
        if hasattr(y_true, "asnumpy"):
            y_true = y_true.asnumpy()
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        c = self.class_index
        pred_c = y_pred == c
        true_c = y_true == c
        self.tp += int(np.sum(pred_c & true_c))
        self.fp += int(np.sum(pred_c & ~true_c))
        self.fn += int(np.sum(~pred_c & true_c))

    def eval(self):
        prec = self.tp / (self.tp + self.fp + self.eps)
        rec = self.tp / (self.tp + self.fn + self.eps)
        return float(2 * prec * rec / (prec + rec + self.eps))


class WeightedCrossEntropyLoss(nn.Cell):
    def __init__(self, class_weights: Optional[ms.Tensor]):
        super().__init__()
        self.class_weights = class_weights
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.cast = ops.Cast()
        self.gather = ops.Gather()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.eps = 1e-12

    def construct(self, logits, labels):
        labels = self.cast(labels, ms.int32)
        per_example = self.ce(logits, labels)
        if self.class_weights is None:
            return self.reduce_mean(per_example)
        w = self.gather(self.class_weights, labels, 0)
        weighted = per_example * w
        return self.reduce_sum(weighted) / (self.reduce_sum(w) + self.eps)


class FocalLoss(nn.Cell):
    """Focal modulator (1-p_t)^gamma * CE; optional per-class alpha from class_weights."""
    def __init__(self, gamma: float, class_weights: Optional[ms.Tensor]):
        super().__init__()
        self.gamma = float(gamma)
        self.class_weights = class_weights
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.cast = ops.Cast()
        self.gather = ops.Gather()
        self.reduce_mean = ops.ReduceMean()
        self.eps = 1e-7

    def construct(self, logits, labels):
        labels = self.cast(labels, ms.int32)
        ce = self.ce(logits, labels)
        pt = ops.exp(-ce)
        mod = ops.pow(1.0 - pt + self.eps, self.gamma)
        loss = mod * ce
        if self.class_weights is not None:
            alpha = self.gather(self.class_weights, labels, 0)
            loss = loss * alpha
        return self.reduce_mean(loss)


def load_ckpt(net, ckpt_path: str, strict: bool):
    params = load_checkpoint(ckpt_path)
    if strict:
        load_param_into_net(net, params)
        print(f"Loaded checkpoint (strict): {ckpt_path}")
        return

    net_params = net.parameters_dict()
    filtered = {}
    skipped = 0

    def candidate_names(param_name: str):
        yield param_name
        while "." in param_name:
            param_name = param_name.split(".", 1)[1]
            yield param_name

    for name, value in params.items():
        loaded = False
        for cand in candidate_names(name):
            if cand in net_params and getattr(net_params[cand], "shape", None) == getattr(value, "shape", None):
                filtered[cand] = value
                loaded = True
                break
        if not loaded:
            skipped += 1
    load_param_into_net(net, filtered)
    print(f"Loaded checkpoint (non-strict): {ckpt_path} (loaded={len(filtered)}, skipped={skipped})")


class EvalAndSaveBest(Callback):
    def __init__(
        self,
        model: Model,
        network,
        eval_dataset,
        metric_name: str,
        ckpt_dir: str,
        prefix: str,
        patience: Optional[int],
        dataset_sink_mode: bool,
    ):
        super().__init__()
        self.model = model
        self.network = network
        self.eval_dataset = eval_dataset
        self.metric_name = metric_name
        self.ckpt_dir = ckpt_dir
        self.prefix = prefix
        self.patience = patience
        self.dataset_sink_mode = dataset_sink_mode
        self.best_score = -1.0
        self.wait = 0
        self.best_ckpt_path = os.path.join(self.ckpt_dir, f"{self.prefix}_best.ckpt")

    def epoch_end(self, run_context):
        results = self.model.eval(self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode)
        score = results.get(self.metric_name)
        if score is None:
            print(f"[val] {results}")
            return

        score = float(score)
        improved = score > self.best_score
        print(f"[val] {results} | best_{self.metric_name}={self.best_score:.6f}")

        if improved:
            self.best_score = score
            self.wait = 0
            os.makedirs(self.ckpt_dir, exist_ok=True)
            save_checkpoint(self.network, self.best_ckpt_path)
            print(f"Saved best checkpoint: {self.best_ckpt_path}")
            return

        if self.patience is None:
            return

        self.wait += 1
        if self.wait >= self.patience:
            print(f"Early stopping (patience={self.patience})")
            run_context.request_stop()


def build_model(
    model_name: str,
    num_classes: int,
    learning_rate: Union[float, list],
    weight_decay: float,
    class_weights: Optional[ms.Tensor],
    ckpt_path: Optional[str],
    strict_load: bool,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    ddos_class_index: int = 0,
):
    net = create_model(
        model_name=model_name,
        pretrained=False,  # offline
        num_classes=num_classes
    )

    if ckpt_path:
        load_ckpt(net, ckpt_path, strict=strict_load)

    if loss_type == "focal":
        loss = FocalLoss(focal_gamma, class_weights)
    elif class_weights is not None:
        loss = WeightedCrossEntropyLoss(class_weights)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    if hasattr(nn, "AdamWeightDecay"):
        optimizer = nn.AdamWeightDecay(
            net.trainable_params(), learning_rate=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)

    model = Model(
        net,
        loss_fn=loss,
        optimizer=optimizer,
        metrics={
            "acc": Accuracy(),
            "f1": F1Score(num_classes=num_classes, average="macro"),
            "f1_ddos": F1SingleClass(num_classes, ddos_class_index),
        },
    )
    return model, net


def parse_args():
    project_root = find_project_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Train/Eval ResNeXt50 on packet images.")
    parser.add_argument(
        "--data-root",
        default=default_data_root(project_root),
        help="Dataset root with train/ (required). Default: dataset/images_per_second_window_0p3 if present, else dataset/images.",
    )
    parser.add_argument(
        "--val-data-root",
        default=None,
        help="Optional root containing val/ (ImageFolder). Default: same as --data-root.",
    )
    parser.add_argument(
        "--test-data-root",
        default=None,
        help="Optional root containing test/ (ImageFolder). Default: same as --data-root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "model"),
        help="Directory for checkpoints (ModelCheckpoint + *_best.ckpt). Use a new path per experiment.",
    )
    parser.add_argument("--model-name", default="resnext50_32x4d")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak LR (cosine max) or constant LR when --lr-schedule none.")
    parser.add_argument(
        "--lr-schedule",
        choices=["none", "cosine"],
        default="cosine",
        help="Training LR schedule (ignored for --eval-only).",
    )
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Cosine floor (per-step schedule tail).")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="Linear warmup epochs before cosine (only with --lr-schedule cosine).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=None, help="Auto-inferred from train/ if omitted.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256, help="Eval resize before center crop (set to 0 to disable).")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--crop-scale-min",
        type=float,
        default=0.7,
        help="RandomResizedCrop min scale (train augment only).",
    )
    parser.add_argument(
        "--crop-scale-max",
        type=float,
        default=1.0,
        help="RandomResizedCrop max scale (train augment only).",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time augmentation.")
    parser.add_argument(
        "--norm-mean",
        type=float,
        nargs=3,
        default=None,
        help="Per-channel normalization mean (default: ImageNet mean, for non-natural-image inputs "
        "trained from scratch pass dataset-computed stats instead).",
    )
    parser.add_argument(
        "--norm-std",
        type=float,
        nargs=3,
        default=None,
        help="Per-channel normalization std (default: ImageNet std; see --norm-mean).",
    )
    parser.add_argument("--no-class-weights", action="store_true", help="Disable auto class-weighting.")
    parser.add_argument(
        "--minority-boost",
        type=float,
        default=1.0,
        help="Extra multiplier on the minority class weight (when class weights are enabled).",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "focal"],
        default="ce",
        help="ce=weighted cross-entropy (if class weights); focal=Focal loss (needs class weights for alpha).",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (only --loss focal).")
    parser.add_argument(
        "--ddos-class-name",
        default="ddos",
        help="Folder name for attack class; used to set f1_ddos index (default: ddos).",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["f1", "f1_ddos"],
        default="f1",
        help="Validation metric for best ckpt + early stopping: f1=macro F1, f1_ddos=F1 on the attack class only.",
    )
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint to load before train/eval.")
    parser.add_argument("--strict-load", action="store_true", help="Strict checkpoint loading (will error on shape mismatch).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate.")
    parser.add_argument("--mode", choices=["GRAPH", "PYNATIVE"], default="GRAPH")
    parser.add_argument(
        "--device-target",
        default="CPU",
        choices=["CPU", "GPU", "Ascend"],
        help="MindSpore device target. Use GPU/Ascend only if your environment supports it.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-sink", action="store_true", help="Enable dataset sink mode (often faster on Ascend).")
    parser.add_argument(
        "--early-stop",
        type=int,
        default=5,
        help="Early stop patience on val using --early-stop-metric (0 disables).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = ms.GRAPH_MODE if args.mode == "GRAPH" else ms.PYNATIVE_MODE
    ms.set_context(mode=mode, device_target=args.device_target, device_id=args.device_id)
    ms.set_seed(args.seed)
    if hasattr(ds, "config") and hasattr(ds.config, "set_seed"):
        ds.config.set_seed(args.seed)

    train_dir = Path(args.data_root) / "train"
    val_root = args.val_data_root or args.data_root
    test_root = args.test_data_root or args.data_root
    class_indexing = build_class_indexing(train_dir)
    num_classes = len(class_indexing) if args.num_classes is None else int(args.num_classes)
    if num_classes != len(class_indexing):
        print(f"WARNING: --num-classes={num_classes} but found {len(class_indexing)} classes in {train_dir}")

    print(
        f"Paths: train={args.data_root}/train | val={val_root}/val | test={test_root}/test | "
        f"checkpoints={args.output_dir}"
    )

    resize_size = None if args.resize_size == 0 else int(args.resize_size)
    augment = not args.no_augment

    train_ds = make_dataset(
        args.data_root,
        "train",
        args.batch_size,
        True,
        class_indexing,
        args.img_size,
        resize_size,
        args.num_workers,
        augment,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    val_ds = make_dataset(
        val_root,
        "val",
        args.batch_size,
        False,
        class_indexing,
        args.img_size,
        resize_size,
        args.num_workers,
        False,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    test_ds = make_dataset(
        test_root,
        "test",
        args.batch_size,
        False,
        class_indexing,
        args.img_size,
        resize_size,
        args.num_workers,
        False,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )

    steps_per_epoch = train_ds.get_dataset_size()
    if args.eval_only or args.lr_schedule == "none":
        learning_rate: Union[float, list] = args.lr
    else:
        learning_rate = build_learning_rate_schedule(
            args.lr_schedule,
            args.lr,
            args.min_lr,
            int(steps_per_epoch) if steps_per_epoch else 0,
            args.epochs,
            args.warmup_epochs,
        )
    if isinstance(learning_rate, list):
        print(
            f"LR schedule: cosine, steps={len(learning_rate)}, "
            f"first={learning_rate[0]:.2e} last={learning_rate[-1]:.2e}"
        )
    else:
        print(f"LR: constant {learning_rate}")

    class_weights = None
    if not args.no_class_weights:
        class_counts = count_images_per_class(train_dir, class_indexing)
        class_weights = ms.Tensor(
            compute_class_weights(class_counts, minority_boost=args.minority_boost),
            dtype=ms.float32,
        )
        print(
            f"Class counts: {class_counts.tolist()} | class weights: {class_weights.asnumpy().tolist()} "
            f"(minority_boost={args.minority_boost})"
        )

    if args.loss == "focal" and class_weights is None:
        raise ValueError("--loss focal requires class weights; remove --no-class-weights or use --loss ce.")

    if args.ddos_class_name in class_indexing:
        ddos_class_index = int(class_indexing[args.ddos_class_name])
    else:
        print(
            f"WARNING: --ddos-class-name={args.ddos_class_name!r} not in {list(class_indexing.keys())}; "
            f"using index 0 for f1_ddos."
        )
        ddos_class_index = 0

    model, net = build_model(
        model_name=args.model_name,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        ckpt_path=args.ckpt,
        strict_load=args.strict_load,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        ddos_class_index=ddos_class_index,
    )

    if not args.eval_only:
        print(
            f"\nStarting training: loss={args.loss}"
            + (f" (gamma={args.focal_gamma})" if args.loss == "focal" else "")
            + f" | early_stop_metric={args.early_stop_metric}\n"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        if steps_per_epoch == 0:
            raise RuntimeError("Training dataset is empty. Check your dataset/train folder.")

        ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
        ckpt_cb = ModelCheckpoint(prefix=args.model_name, directory=args.output_dir, config=ckpt_config)
        eval_cb = EvalAndSaveBest(
            model=model,
            network=net,
            eval_dataset=val_ds,
            metric_name=args.early_stop_metric,
            ckpt_dir=args.output_dir,
            prefix=args.model_name,
            patience=(None if args.early_stop <= 0 else int(args.early_stop)),
            dataset_sink_mode=args.dataset_sink,
        )

        model.train(
            args.epochs,
            train_ds,
            callbacks=[LossMonitor(), TimeMonitor(), ckpt_cb, eval_cb],
            dataset_sink_mode=args.dataset_sink,
        )

    print("\nEvaluating on test set...\n")
    best_ckpt_path = os.path.join(args.output_dir, f"{args.model_name}_best.ckpt")
    if os.path.isfile(best_ckpt_path):
        load_ckpt(net, best_ckpt_path, strict=True)
    metrics = model.eval(test_ds, dataset_sink_mode=args.dataset_sink)
    print(metrics)


if __name__ == "__main__":
    main()
