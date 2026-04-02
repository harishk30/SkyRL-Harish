#!/bin/bash
# ============================================================================
# Citation Prediction V3 SFT warm-start on ai-lab.
#
# This job:
# 1. Prepares an exact single-turn Qwen-format SFT dataset from Claude rollouts.
# 2. Writes a filtered RL train parquet with those prompts removed.
# 3. Runs policy-only SFT on Qwen3-4B.
#
# Usage:
#   sbatch train_citation_prediction_sft_ailab.slurm
#   MAX_PROMPTS=128 EPOCHS=1 sbatch train_citation_prediction_sft_ailab.slurm
# ============================================================================

#SBATCH --job-name=cit-v3-sft
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/cit-v3-sft_%A_%a.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/cit-v3-sft_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --partition=ailab

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /home/hk4638/SkyRL/skyrl-train

SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
LOG_DIR="${SCRATCH_BASE}/logs"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction_v3/full"
PREP_BASE="${SCRATCH_BASE}/data/citation_prediction_v3/sft_claude_high_recall"
MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
TRAJECTORY_FILE="/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/high_recall_max4_sft_beneficial.json"

MAX_PROMPTS=${MAX_PROMPTS:-318}
SORT_BY=${SORT_BY:-recall_desc}
PROMPT_SOURCE=${PROMPT_SOURCE:-rl_train}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-32768}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-65536}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-2}
MICRO_TRAIN_BATCH_SIZE_PER_GPU=${MICRO_TRAIN_BATCH_SIZE_PER_GPU:-1}
if [ -n "${LR_LIST:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LR_ARRAY=(${LR_LIST})
    LR=${LR_ARRAY[$SLURM_ARRAY_TASK_ID]}
else
    LR=${LR:-5e-6}
fi
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-50}
LR_TAG=$(printf '%s' "${LR}" | sed 's/+//g; s/\./p/g; s/-/m/g')
RUN_TAG=${RUN_TAG:-"p${MAX_PROMPTS}_e${EPOCHS}_lr${LR_TAG}"}

PREP_DIR="${PREP_BASE}/${RUN_TAG}"
SFT_OUTPUT_DIR="${SCRATCH_BASE}/checkpoints/citation_prediction_v3_sft/${RUN_TAG}"

mkdir -p "${LOG_DIR}" "${PREP_DIR}" "${SFT_OUTPUT_DIR}"

echo "=== Citation Prediction V3 SFT ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "MAX_PROMPTS=${MAX_PROMPTS}"
echo "MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-none}"
echo "EPOCHS=${EPOCHS}"
echo "MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS}"
echo "MAX_BATCH_SIZE=${MAX_BATCH_SIZE}"
echo "LR=${LR}"
echo "PREP_DIR=${PREP_DIR}"
echo "SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR}"

export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda/pkgs"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"
export UV_CACHE_DIR="${SCRATCH_BASE}/uv_cache"
export UV_LINK_MODE=copy
export HF_HOME="${SCRATCH_BASE}/huggingface"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_BASE}/huggingface/hub"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/huggingface/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
export VIRTUAL_ENV="/home/hk4638/SkyRL/skyrl-train/.venv"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKYRL_DISABLE_RAY_RUNTIME_ENV=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK

PREP_ARGS=(
    python
    /home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/harish_setup/prepare_high_recall_sft_data.py
    --trajectories "${TRAJECTORY_FILE}"
    --rl-train-parquet "${DATA_DIR}/train.parquet"
    --rl-data-dir "${DATA_DIR}"
    --tokenizer-path "${MODEL_PATH}"
    --output-dir "${PREP_DIR}"
    --max-prompts "${MAX_PROMPTS}"
    --sort-by "${SORT_BY}"
    --prompt-source "${PROMPT_SOURCE}"
)

if [ -n "${MAX_SEQUENCE_LENGTH}" ]; then
    PREP_ARGS+=(--max-sequence-length "${MAX_SEQUENCE_LENGTH}")
fi

echo "=== Preparing SFT data ==="
uv run --active --no-sync "${PREP_ARGS[@]}"

TRAIN_ARGS=(
    python
    /home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/harish_setup/train_citation_prediction_sft.py
    --dataset "${PREP_DIR}/sft_data/train.jsonl"
    --model-path "${MODEL_PATH}"
    --output-dir "${SFT_OUTPUT_DIR}"
    --epochs "${EPOCHS}"
    --max-batch-tokens "${MAX_BATCH_TOKENS}"
    --max-batch-size "${MAX_BATCH_SIZE}"
    --micro-train-batch-size-per-gpu "${MICRO_TRAIN_BATCH_SIZE_PER_GPU}"
    --learning-rate "${LR}"
    --save-every-steps "${SAVE_EVERY_STEPS}"
)

if [ -n "${MAX_STEPS}" ]; then
    TRAIN_ARGS+=(--max-steps "${MAX_STEPS}")
fi

if [ -n "${MAX_SEQUENCE_LENGTH}" ]; then
    TRAIN_ARGS+=(--max-sequence-length "${MAX_SEQUENCE_LENGTH}")
fi

echo "=== Starting SFT training ==="
uv run --active --no-sync --extra vllm "${TRAIN_ARGS[@]}"

echo "=== Done ==="
echo "End time: $(date)"
