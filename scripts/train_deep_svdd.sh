#!/usr/bin/env bash
# Train + eval the Deep SVDD unsupervised flowpic baseline (see
# scripts/deep_svdd_model.py for the method). Second unsupervised approach
# alongside scripts/train_flowpic_autoencoder.py's reconstruction-MSE
# autoencoder, added because that autoencoder's core assumption doesn't
# hold on this dataset (ddos windows are sparser, not busier, than normal --
# see WORK_LOG.md 2026-08-04).
#
# Sets CUDA_HOME/LD_LIBRARY_PATH itself -- calling train_deep_svdd.py or
# eval_deep_svdd.py directly without this wrapper repeats a bug hit earlier
# today with the reconstruction autoencoder (MindSpore silently can't find
# GPU libs and falls back/errors without these set).
#
# Checkpoints + logs: model/flowpic_deep_svdd_v1/ (override with OUTPUT_DIR=...)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model/flowpic_deep_svdd_v1}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/images_flowpic_0p3_validated_v1}"

mkdir -p "${OUTPUT_DIR}"

"${VENV_PY}" "${SCRIPT_DIR}/train_deep_svdd.py" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device-target GPU

"${VENV_PY}" "${SCRIPT_DIR}/eval_deep_svdd.py" \
  --data-root "${DATA_ROOT}" \
  --ckpt "${OUTPUT_DIR}/svdd_encoder_best.ckpt" \
  --center "${OUTPUT_DIR}/center.json" \
  --output-dir "${OUTPUT_DIR}" \
  --device-target GPU
