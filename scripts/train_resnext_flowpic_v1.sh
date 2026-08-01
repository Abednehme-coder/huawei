#!/usr/bin/env bash
# Train on FlowPic-style 3-channel packet-histogram images (see
# scripts/pcap_to_flowpic_images.py), on the same cluster-validated labels as
# train_resnext_per_second_0p3_validated_v1.sh. Same architecture/loss/LR as
# the other 0.3s runs -- only the input representation differs, isolating the
# encoding as the sole experimental variable in the comparison.
#
# --no-augment: RandomResizedCrop/RandomHorizontalFlip assume natural-photo
# invariances (crop-scale, left-right symmetry) that don't hold for a
# size-axis x time-axis histogram -- cropping deletes a real packet-size
# range, flipping reverses time order.
# --norm-mean/--norm-std: dataset-computed per-channel stats (not ImageNet's),
# since these channels have very different sparsity (SYN-only vs all-packets)
# and the model is trained from scratch anyway. Computed via
# scripts/compute_flowpic_norm_stats.py once images exist; override via env
# if you recompute them.
#
# Checkpoints: model/flowpic_0p3_validated_v1/ (override with OUTPUT_DIR=...)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_PY="${PROJECT_ROOT}/notebook/train_resnext.py"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model/flowpic_0p3_validated_v1}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/images_flowpic_0p3_validated_v1}"

# Placeholder until scripts/compute_flowpic_norm_stats.py runs on the real
# generated dataset -- override via env (NORM_MEAN="r g b" NORM_STD="r g b").
NORM_MEAN="${NORM_MEAN:-0.5 0.5 0.5}"
NORM_STD="${NORM_STD:-0.25 0.25 0.25}"

mkdir -p "${OUTPUT_DIR}"

exec "${VENV_PY}" "${TRAIN_PY}" --device-target GPU \
  --data-root "${DATA_ROOT}" \
  --val-data-root "${DATA_ROOT}" \
  --test-data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --loss focal \
  --focal-gamma 2.0 \
  --minority-boost 8.0 \
  --early-stop-metric f1_ddos \
  --early-stop 10 \
  --epochs 40 \
  --batch-size 32 \
  --lr 5e-4 \
  --lr-schedule cosine \
  --min-lr 1e-6 \
  --warmup-epochs 1 \
  --no-augment \
  --norm-mean ${NORM_MEAN} \
  --norm-std ${NORM_STD} \
  "$@"
