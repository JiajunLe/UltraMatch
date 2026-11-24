#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_PATH="${LOG_PATH:-logs/scannet.log}"

data_cfg_path="configs/data/scannet_test_1500.py"
main_cfg_path="configs/ultramatch/indoor/ultramatch_indoor.py"
ckpt_path="${CKPT_PATH:-weights/outdoor.ckpt}"

mkdir -p "$(dirname "${LOG_PATH}")"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u ./test.py \
    "${data_cfg_path}" \
    "${main_cfg_path}" \
    --gpus=1 \
    --num_nodes=1 \
    --accelerator=ddp \
    --batch_size=1 \
    --num_workers="${NUM_WORKERS}" \
    --ckpt_path="${ckpt_path}" \
    --profiler_name=inference \
    "$@" \
    2>&1 | tee "${LOG_PATH}"
