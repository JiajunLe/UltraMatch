#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=, read -r -a gpu_list <<< "${GPU_IDS}"
N_GPUS="${N_GPUS:-${#gpu_list[@]}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
EXP_NAME="${EXP_NAME:-ultramatch}"
RECOVERY_STEPS="${RECOVERY_STEPS:-1150}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:--1}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:--1}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
DISABLE_CKPT="${DISABLE_CKPT:-false}"
SAVE_TOP_K="${SAVE_TOP_K:--1}"
LOG_PATH="${LOG_PATH:-logs/${EXP_NAME}.train.log}"

data_cfg_path="configs/data/megadepth_trainval_832.py"
main_cfg_path="configs/ultramatch/outdoor/ultramatch_train.py"

mkdir -p "$(dirname "${LOG_PATH}")"

CMD=(
    "${PYTHON_BIN}" -u ./train.py
    "${data_cfg_path}"
    "${main_cfg_path}"
    --exp_name="${EXP_NAME}"
    --gpus="${N_GPUS}"
    --num_nodes=1
    --accelerator=ddp
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --pin_memory=true
    --check_val_every_n_epoch=1
    --log_every_n_steps=100
    --num_sanity_val_steps=0
    --benchmark=true
    --max_epochs="${MAX_EPOCHS}"
    --limit_train_batches="${LIMIT_TRAIN_BATCHES}"
    --limit_val_batches="${LIMIT_VAL_BATCHES}"
    --split_data_idx=0
    --recovery_ckpt_every_n_train_steps="${RECOVERY_STEPS}"
    --save_top_k="${SAVE_TOP_K}"
)

if [[ -n "${RESUME_CKPT_PATH}" ]]; then
    CMD+=(--resume_ckpt_path="${RESUME_CKPT_PATH}")
fi
if [[ "${DISABLE_CKPT}" == "true" ]]; then
    CMD+=(--disable_ckpt)
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
