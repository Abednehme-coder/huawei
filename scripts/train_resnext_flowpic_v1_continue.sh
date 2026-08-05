#!/usr/bin/env bash
# Continue training the flowpic ResNeXt-50 classifier from the epoch-22
# checkpoint (model/flowpic_0p3_validated_v1/resnext50_32x4d_best.ckpt),
# which is where the original 40-epoch run (scripts/train_resnext_flowpic_v1.sh)
# got stuck at after the 4th 2026-08-04 power loss.
#
# NOT a true resume -- train_resnext.py has no optimizer/LR-schedule state
# checkpointing, only weight loading (--ckpt loads weights, strict, via
# load_ckpt()). This starts a FRESH cosine schedule from those weights
# instead of continuing epoch 23's schedule position. To avoid disrupting
# already-good weights with the original run's peak LR (5e-4 -- plausibly
# a contributor to the val f1_ddos volatility seen late in that run, e.g.
# epoch 20 crashing to 0.059 before recovering by epoch 22), this uses a
# notably lower peak LR (1.5e-4) over a shorter schedule (18 epochs).
#
# Writes to a NEW output dir so the epoch-22 checkpoint this is continuing
# from is never overwritten -- compare both afterward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_PY="${PROJECT_ROOT}/notebook/train_resnext.py"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model/flowpic_0p3_validated_v1_continued}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/images_flowpic_0p3_validated_v1}"
INIT_CKPT="${INIT_CKPT:-${PROJECT_ROOT}/model/flowpic_0p3_validated_v1/resnext50_32x4d_best.ckpt}"

# Same dataset-computed stats as the original run -- see
# scripts/train_resnext_flowpic_v1.sh for how these were computed.
NORM_MEAN="${NORM_MEAN:-0.0071 0.0003 0.0004}"
NORM_STD="${NORM_STD:-0.0592 0.0157 0.0145}"

if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "Missing init checkpoint: ${INIT_CKPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

exec "${VENV_PY}" "${TRAIN_PY}" --device-target GPU \
  --data-root "${DATA_ROOT}" \
  --val-data-root "${DATA_ROOT}" \
  --test-data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --ckpt "${INIT_CKPT}" \
  --strict-load \
  --loss focal \
  --focal-gamma 2.0 \
  --minority-boost 8.0 \
  --early-stop-metric f1_ddos \
  --early-stop 8 \
  --epochs 18 \
  --batch-size 32 \
  --lr 1.5e-4 \
  --lr-schedule cosine \
  --min-lr 1e-6 \
  --warmup-epochs 1 \
  --no-augment \
  --norm-mean ${NORM_MEAN} \
  --norm-std ${NORM_STD} \
  "$@"
