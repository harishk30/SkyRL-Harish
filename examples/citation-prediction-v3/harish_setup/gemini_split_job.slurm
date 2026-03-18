#!/bin/bash
# ============================================================================
# Gemini subsection splitting (CPU-only, needs internet for Gemini API).
#
# Supports two modes:
#   MODE=batch-submit  Upload JSONL and create Gemini Batch API job (~30s)
#   MODE=batch-collect Download batch results and process into final output
#   MODE=streaming     Original streaming mode with async calls (legacy)
#
# Env vars:
#   MODE             batch-submit | batch-collect | streaming (default: batch-submit)
#   GEMINI_MODEL     Gemini model name (default: gemini-3-flash-preview)
#   SPLIT            Only process this split (default: all)
#   JOB_NAME         Batch job name for collect mode (from submit output)
#   CONCURRENCY      Max concurrent requests for streaming mode (default: 3)
#
# Usage:
#   sbatch gemini_split_job.slurm                              # batch submit all
#   MODE=batch-collect JOB_NAME=batches/XXX sbatch gemini_split_job.slurm
#   MODE=streaming CONCURRENCY=3 sbatch gemini_split_job.slurm # legacy streaming
# ============================================================================

#SBATCH --job-name=gemini-split-v3
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/gemini-split-v3_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/gemini-split-v3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

set -e
export PYTHONUNBUFFERED=1

cd /home/hk4638/SkyRL

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
OUTPUT_DIR="${SCRATCH_BASE}/data/citation_prediction_v3"
LOG_DIR="${SCRATCH_BASE}/logs"
V3_DIR="/home/hk4638/SkyRL/examples/citation-prediction-v3"

MODE=${MODE:-"batch-submit"}
GEMINI_MODEL=${GEMINI_MODEL:-"gemini-3-flash-preview"}
SPLIT=${SPLIT:-""}
JOB_NAME=${JOB_NAME:-""}
CONCURRENCY=${CONCURRENCY:-3}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== Gemini Subsection Splitting (V3) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Mode: ${MODE}"
echo "Model: ${GEMINI_MODEL}"

export UV_CACHE_DIR="${SCRATCH_BASE}/uv_cache"
export UV_LINK_MODE=copy
export VIRTUAL_ENV="/home/hk4638/SkyRL/.venv"
export PATH="$HOME/.local/bin:$PATH"

# CRITICAL: Load proxy for Gemini API access
module load proxy/default

# Load secrets (GEMINI_API_KEY)
ENV_FILE="/home/hk4638/SkyRL/examples/search/harish_setup/.env"
if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment from ${ENV_FILE}"
    set -a && source "${ENV_FILE}" && set +a
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set"
    exit 1
fi

# ============================================================================
# INPUT VALIDATION
# ============================================================================
INPUT_FILE="${OUTPUT_DIR}/filtered_papers.json"
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: filtered_papers.json not found at ${INPUT_FILE}"
    echo "Run Steps 1-2 on the login node first."
    exit 1
fi

SPLIT_ARG=""
if [ -n "$SPLIT" ]; then
    SPLIT_ARG="--split ${SPLIT}"
fi

# ============================================================================
# RUN
# ============================================================================
if [ "$MODE" = "batch-submit" ]; then
    echo "=== Submitting Gemini Batch Job ==="
    /home/hk4638/.local/bin/uv run --active --frozen \
        python "${V3_DIR}/data_v3/gemini_batch_split.py" submit \
            --input "${INPUT_FILE}" \
            --model "${GEMINI_MODEL}" \
            ${SPLIT_ARG}

elif [ "$MODE" = "batch-collect" ]; then
    if [ -z "$JOB_NAME" ]; then
        echo "ERROR: Set JOB_NAME env var (from batch-submit output)"
        exit 1
    fi
    echo "=== Collecting Batch Results ==="
    echo "Job name: ${JOB_NAME}"
    /home/hk4638/.local/bin/uv run --active --frozen \
        python "${V3_DIR}/data_v3/gemini_batch_split.py" collect \
            --job-name "${JOB_NAME}" \
            --input "${INPUT_FILE}" \
            --output "${OUTPUT_DIR}/gemini_subsections_all.json"

elif [ "$MODE" = "streaming" ]; then
    echo "=== Running Streaming Gemini Splitting ==="
    echo "Concurrency: ${CONCURRENCY}"
    /home/hk4638/.local/bin/uv run --active --frozen \
        python "${V3_DIR}/data_v3/gemini_split_subsections.py" \
            --input "${INPUT_FILE}" \
            --output "${OUTPUT_DIR}/gemini_subsections_all.json" \
            --checkpoint "${OUTPUT_DIR}/gemini_subsections_all.checkpoint.jsonl" \
            --concurrency "${CONCURRENCY}" \
            --model "${GEMINI_MODEL}" \
            ${SPLIT_ARG}
else
    echo "ERROR: Unknown MODE=${MODE}. Use batch-submit, batch-collect, or streaming."
    exit 1
fi

EXIT_CODE=$?

echo "=== Done ==="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
