#!/usr/bin/env bash
# Train on 0.3s time-window PNGs (byte-dump encoding, unchanged), but on
# cluster-validated labels (dataset/window_labels_validated_v1.json) instead
# of the raw schedule+SYN-count heuristic. Copied from
# train_resnext_per_second_0p3.sh with only OUTPUT_DIR/DATA_ROOT changed, so
# labeling is the only experimental variable vs. the existing primary run.
# Checkpoints: model/per_second_0p3_validated_v1/ (override with OUTPUT_DIR=...)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_PY="${PROJECT_ROOT}/notebook/train_resnext.py"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model/per_second_0p3_validated_v1}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/images_per_second_window_0p3_validated_v1}"

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
  "$@"
