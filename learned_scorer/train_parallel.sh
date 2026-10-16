#!/bin/bash
# 8-GPU parallel training launcher
# Each GPU processes a shard of the data, then we merge weights

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_BASE="${OUTPUT_BASE:-${WORK_DIR}/learned_scorer/checkpoints/llama8b}"
NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-200}"
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))  # 25

mkdir -p "${OUTPUT_BASE}"

echo "=== Launching ${NUM_GPUS} parallel training jobs ==="
echo "Model: ${MODEL}"
echo "Samples per GPU: ${SAMPLES_PER_GPU}"

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_DIR="${OUTPUT_BASE}/shard_${GPU_ID}"
    mkdir -p "${SHARD_DIR}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${WORK_DIR}/learned_scorer/train.py \
        --model_name "${MODEL}" \
        --dataset cnn_dailymail \
        --num_samples ${SAMPLES_PER_GPU} \
        --max_prompt_len 4096 \
        --collect_steps 128 \
        --collect_interval 1 \
        --scorer_dim 64 \
        --lr 1e-3 \
        --batch_size 64 \
        --train_steps_per_sample 4 \
        --ranking_weight 0.1 \
        --seed $((42 + GPU_ID * 1000)) \
        --output_dir "${SHARD_DIR}" \
        > "${SHARD_DIR}/train.log" 2>&1 &

    PIDS+=($!)
    echo "  GPU ${GPU_ID}: PID $! -> ${SHARD_DIR}"
done

echo "=== Waiting for all jobs to finish ==="
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait ${PIDS[$i]}; then
        echo "  GPU ${i}: done"
    else
        echo "  GPU ${i}: FAILED (exit code $?)"
        FAILED=1
    fi
done

if [ ${FAILED} -eq 1 ]; then
    echo "Some jobs failed. Check logs in ${OUTPUT_BASE}/shard_*/train.log"
    exit 1
fi

echo "=== Merging shards ==="
${PYTHON} ${WORK_DIR}/learned_scorer/merge_shards.py \
    --shard_dir "${OUTPUT_BASE}" \
    --num_shards ${NUM_GPUS} \
    --output "${OUTPUT_BASE}/scorer_final.pt"

echo "=== Done! Final model: ${OUTPUT_BASE}/scorer_final.pt ==="
