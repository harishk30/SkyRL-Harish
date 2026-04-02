#!/bin/bash
# Launch a chain of 5 sequential GRPO jobs for bp256_n10.
# Each job resumes from the previous checkpoint via resume_mode=latest.
#
# Usage:
#   bash harish_setup/launch_chained_bp256_n10.sh [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/train_citation_prediction_4b_ailab.slurm"

NUM_JOBS=5
CKPT_INTERVAL=5

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

EXPORT_VARS="ALL,SWEEP_BP=256,SWEEP_N=10,SWEEP_CKPT_INTERVAL=${CKPT_INTERVAL}"

PREV_JOB_ID=""
for i in $(seq 1 ${NUM_JOBS}); do
    if [ -z "${PREV_JOB_ID}" ]; then
        CMD="sbatch --export=${EXPORT_VARS} ${SLURM_SCRIPT}"
    else
        CMD="sbatch --dependency=afterany:${PREV_JOB_ID} --export=${EXPORT_VARS} ${SLURM_SCRIPT}"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] Job ${i}/${NUM_JOBS}: ${CMD}"
        PREV_JOB_ID="FAKE_${i}"
    else
        OUTPUT=$(eval "${CMD}")
        PREV_JOB_ID=$(echo "${OUTPUT}" | grep -oP '\d+')
        echo "Job ${i}/${NUM_JOBS}: submitted ${PREV_JOB_ID} ${CMD}"
    fi
done

echo ""
echo "Chain of ${NUM_JOBS} jobs submitted."
echo "Run name: cit-v3-4b-h200-bp256_n10_lr2e-06_nokl"
echo "Checkpoint interval: every ${CKPT_INTERVAL} steps"
echo "Expected ~18 steps/job × ${NUM_JOBS} jobs = ~90 total steps (~5+ epochs)"
