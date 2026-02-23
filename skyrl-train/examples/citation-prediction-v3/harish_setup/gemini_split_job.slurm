#!/bin/bash
# ============================================================================
# Gemini subsection splitting (CPU-only, needs internet for Gemini API).
#
# Runs gemini_split_subsections.py on a compute node with proxy enabled.
# No GPUs needed — just Gemini API calls.
#
# Env vars (optional):
#   MAX_EXAMPLES     Number of papers to process (default: 100, 0=all)
#   CONCURRENCY      Max concurrent Gemini requests (default: 10)
#   GEMINI_MODEL     Gemini model name (default: gemini-2.5-flash-preview-05-20)
#   SEED             Random seed for sampling (default: 42)
#   SPLIT            Only process this split (default: all)
#
# Usage:
#   sbatch gemini_split_job.slurm                    # 100 papers (test)
#   MAX_EXAMPLES=0 sbatch gemini_split_job.slurm     # all papers
#   MAX_EXAMPLES=500 sbatch gemini_split_job.slurm   # 500 papers
# ============================================================================

#SBATCH --job-name=gemini-split-v3
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/gemini-split-v3_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/gemini-split-v3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

set -e
export PYTHONUNBUFFERED=1

cd /home/hk4638/SkyRL/skyrl-train

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
OUTPUT_DIR="${SCRATCH_BASE}/data/citation_prediction_v3"
LOG_DIR="${SCRATCH_BASE}/logs"
V3_DIR="/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3"

MAX_EXAMPLES=${MAX_EXAMPLES:-100}
CONCURRENCY=${CONCURRENCY:-10}
GEMINI_MODEL=${GEMINI_MODEL:-"gemini-3-pro-preview"}
SEED=${SEED:-42}
SPLIT=${SPLIT:-""}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== Gemini Subsection Splitting (V3) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Config: max_examples=${MAX_EXAMPLES}, concurrency=${CONCURRENCY}"
echo "Config: model=${GEMINI_MODEL}, seed=${SEED}, split=${SPLIT:-all}"

export UV_CACHE_DIR="${SCRATCH_BASE}/uv_cache"
export UV_LINK_MODE=copy
export VIRTUAL_ENV="/home/hk4638/SkyRL/skyrl-train/.venv"
export PATH="$HOME/.local/bin:$PATH"

# CRITICAL: Load proxy for Gemini API access
module load proxy/default

# Load secrets (GEMINI_API_KEY)
ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env"
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

# ============================================================================
# SAMPLE IF NEEDED
# ============================================================================
if [ "$MAX_EXAMPLES" -gt 0 ]; then
    SAMPLE_FILE="${OUTPUT_DIR}/filtered_papers_sample${MAX_EXAMPLES}_seed${SEED}.json"
    if [ ! -f "${SAMPLE_FILE}" ]; then
        echo "Sampling ${MAX_EXAMPLES} papers (seed=${SEED})..."
        /home/hk4638/.local/bin/uv run --active --frozen python -c "
import json, random
random.seed(${SEED})
with open('${INPUT_FILE}') as f:
    papers = json.load(f)
sample = random.sample(papers, min(${MAX_EXAMPLES}, len(papers)))
with open('${SAMPLE_FILE}', 'w') as f:
    json.dump(sample, f, indent=2)
print(f'Sampled {len(sample)} papers from {len(papers)} total')
"
    else
        echo "Sample file already exists: ${SAMPLE_FILE}"
    fi
    GEMINI_INPUT="${SAMPLE_FILE}"
    OUTPUT_SUFFIX="_sample${MAX_EXAMPLES}_seed${SEED}"
else
    GEMINI_INPUT="${INPUT_FILE}"
    OUTPUT_SUFFIX=""
fi

GEMINI_OUTPUT="${OUTPUT_DIR}/gemini_subsections${OUTPUT_SUFFIX}.json"
CHECKPOINT="${OUTPUT_DIR}/gemini_subsections${OUTPUT_SUFFIX}.checkpoint.jsonl"

echo "Input: ${GEMINI_INPUT}"
echo "Output: ${GEMINI_OUTPUT}"
echo "Checkpoint: ${CHECKPOINT}"

# ============================================================================
# RUN GEMINI SPLITTING
# ============================================================================
echo "=== Running Gemini subsection splitting ==="

SPLIT_ARG=""
if [ -n "$SPLIT" ]; then
    SPLIT_ARG="--split ${SPLIT}"
fi

/home/hk4638/.local/bin/uv run --active --frozen \
    python "${V3_DIR}/data_v3/gemini_split_subsections.py" \
        --input "${GEMINI_INPUT}" \
        --output "${GEMINI_OUTPUT}" \
        --checkpoint "${CHECKPOINT}" \
        --concurrency "${CONCURRENCY}" \
        --model "${GEMINI_MODEL}" \
        ${SPLIT_ARG}

EXIT_CODE=$?

echo "=== Gemini splitting complete ==="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "Output: ${GEMINI_OUTPUT}"

exit ${EXIT_CODE}
