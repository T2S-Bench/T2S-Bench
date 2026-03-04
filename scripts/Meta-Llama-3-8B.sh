#!/bin/bash

MODEL="meta-llama/Meta-Llama-3-8B" ## CHANGE THIS
MODEL_SAFE="${MODEL##*/}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# MR
TEST_DIR_MR="${REPO_ROOT}/data/T2S-Bench-MR"
OUT_DIR_MR="${REPO_ROOT}/results/MR"
mkdir -p "${OUT_DIR_MR}"

python evaluate_model.py \
    --data_dir           "${TEST_DIR_MR}" \
    --model_type         hf \
    --model_name_or_path "${MODEL}" \
    --output             "${OUT_DIR_MR}/${MODEL_SAFE}.json" \
    --num_workers        8 \

# E2E
TEST_DIR_E2E="${REPO_ROOT}/data/T2S-Bench-E2E"
OUT_DIR_E2E="${REPO_ROOT}/results/E2E"
mkdir -p "${OUT_DIR_E2E}"

python evaluate_structure.py \
    --data_dir           "${TEST_DIR_E2E}" \
    --model_type         hf \
    --model_name_or_path "${MODEL}" \
    --output             "${OUT_DIR_E2E}/${MODEL_SAFE}.json" \
    --num_workers        8 \
    --resume \