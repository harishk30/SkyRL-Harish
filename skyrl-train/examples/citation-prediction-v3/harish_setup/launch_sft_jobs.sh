#!/bin/bash
# ============================================================================
# Launch parallel SFT trajectory generation jobs on gpu-test.
#
# Submits NUM_CHUNKS jobs, each processing a different slice of the
# sampled queries. All chunks use the same seed and sample_frac, so
# the full sampled set is deterministic and partitioned evenly.
#
# Usage:
#   bash launch_sft_jobs.sh                    # 5 chunks, defaults
#   bash launch_sft_jobs.sh --chunks 20        # 20 chunks
#   bash launch_sft_jobs.sh --dry-run          # show commands only
#   bash launch_sft_jobs.sh --chunks 3 --samples 5  # quick test
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/sft_trajectories_job.slurm"

# Defaults
NUM_CHUNKS=120
START_CHUNK=0
END_CHUNK=10
NUM_SAMPLES=20
SAMPLE_FRAC="0.5"
SEED=42
DRY_RUN=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --chunks)     NUM_CHUNKS="$2"; shift 2 ;;
        --start)      START_CHUNK="$2"; shift 2 ;;
        --end)        END_CHUNK="$2"; shift 2 ;;
        --samples)    NUM_SAMPLES="$2"; shift 2 ;;
        --frac)       SAMPLE_FRAC="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Launching SFT trajectory jobs ==="
echo "Total chunks: ${NUM_CHUNKS}, submitting [${START_CHUNK}, ${END_CHUNK})"
echo "Samples per query: ${NUM_SAMPLES}"
echo "Sample fraction: ${SAMPLE_FRAC}"
echo "Seed: ${SEED}"
echo ""

JOB_IDS=()
for i in $(seq ${START_CHUNK} $((END_CHUNK - 1))); do
    CMD="sbatch --export=ALL,CHUNK_ID=${i},NUM_CHUNKS=${NUM_CHUNKS},NUM_SAMPLES=${NUM_SAMPLES},SAMPLE_FRAC=${SAMPLE_FRAC},SEED=${SEED} ${SLURM_SCRIPT}"

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] ${CMD}"
    else
        OUTPUT=$(${CMD})
        JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+')
        JOB_IDS+=("$JOB_ID")
        echo "Chunk ${i}/${NUM_CHUNKS}: ${OUTPUT}"
    fi
done

if [ "$DRY_RUN" = false ] && [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "=== All jobs submitted ==="
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor: squeue -u \$USER -n sft-traj-v3"
    echo "Merge after completion:"
    echo "  python ${SCRIPT_DIR}/../training/merge_sft_chunks.py \\"
    echo "    --input_dir /scratch/gpfs/ZHUANGL/hk4638/logs/sft_trajectories_v3/ \\"
    echo "    --output /scratch/gpfs/ZHUANGL/hk4638/logs/sft_trajectories_v3/sft_merged.json"
fi
