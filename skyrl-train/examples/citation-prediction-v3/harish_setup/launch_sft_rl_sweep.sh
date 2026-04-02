#!/bin/bash
# Launch SFT-init GRPO sweep: 6 configs, each chained 5 jobs.
# All use auto LR via sqrt(B/1024) scaling, no KL.
#
# Usage:
#   bash harish_setup/launch_sft_rl_sweep.sh [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/train_citation_prediction_4b_ailab.slurm"

SFT_MODEL="/scratch/gpfs/ZHUANGL/hk4638/checkpoints/citation_prediction_v3_sft/p318_e1_lr5em6/step_143/policy"
SFT_DATA_DIR="/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/sft_filtered"
CHAIN_LENGTH=5
CKPT_INTERVAL=5
RUN_PREFIX="cit-v3-sft5e6-rl"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Sweep grid: (BP, N) — all auto LR, no KL
CONFIGS=(
    "128 10"
    "128 20"
    "128 40"
    "256 10"
    "256 20"
    "512 10"
)

echo "SFT-init GRPO sweep (lr=5e-6 checkpoint)"
echo "  Model: ${SFT_MODEL}"
echo "  Chain length: ${CHAIN_LENGTH}"
echo "  Checkpoint interval: ${CKPT_INTERVAL}"
echo "  Configs: ${#CONFIGS[@]}"
echo ""

for config in "${CONFIGS[@]}"; do
    read -r BP N <<< "$config"

    EXPORT_VARS="ALL,SWEEP_BP=${BP},SWEEP_N=${N},SWEEP_CKPT_INTERVAL=${CKPT_INTERVAL},SWEEP_MODEL_PATH=${SFT_MODEL},SWEEP_RUN_PREFIX=${RUN_PREFIX},SWEEP_DATA_DIR=${SFT_DATA_DIR}"

    PREV_JOB_ID=""
    for i in $(seq 1 ${CHAIN_LENGTH}); do
        if [ -z "${PREV_JOB_ID}" ]; then
            CMD="sbatch --export=${EXPORT_VARS} ${SLURM_SCRIPT}"
        else
            CMD="sbatch --dependency=afterany:${PREV_JOB_ID} --export=${EXPORT_VARS} ${SLURM_SCRIPT}"
        fi

        if [ "$DRY_RUN" = true ]; then
            if [ $i -eq 1 ]; then
                echo "[dry-run] bp${BP}_n${N}: ${CMD}"
                echo "          ... + $((CHAIN_LENGTH-1)) chained jobs"
            fi
            PREV_JOB_ID="FAKE"
        else
            OUTPUT=$(eval "${CMD}")
            PREV_JOB_ID=$(echo "${OUTPUT}" | grep -oP '\d+')
            if [ $i -eq 1 ]; then
                FIRST_JOB="${PREV_JOB_ID}"
            fi
            if [ $i -eq ${CHAIN_LENGTH} ]; then
                echo "bp${BP}_n${N}: jobs ${FIRST_JOB}→${PREV_JOB_ID} (${CHAIN_LENGTH} chained)"
            fi
        fi
    done
done

echo ""
echo "Done. All configs submitted with ${CHAIN_LENGTH}-job chains."
