#!/bin/bash
# ============================================================================
# Generate SFT trajectories using Gemini with retriever-in-the-loop.
#
# 1-node job: retriever (background) + generate_sft_trajectories.py
# No vLLM needed — Gemini is called via API.
#
# Usage:
#   sbatch sft_trajectories_job.slurm
# ============================================================================

#SBATCH --job-name=sft-traj
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/sft-traj_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/sft-traj_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short

set -e
export PYTHONUNBUFFERED=1

cd /home/hk4638/SkyRL/skyrl-train

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction"
LOG_DIR="${SCRATCH_BASE}/logs"
V2_DIR="/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v2"

MAX_EXAMPLES=${MAX_EXAMPLES:-50}
NUM_SAMPLES=${NUM_SAMPLES:-5}
MAX_TURNS=${MAX_TURNS:-4}
TOPK=${TOPK:-5}
RECALL_THRESHOLD=${RECALL_THRESHOLD:-"0.3"}
GEMINI_MODEL=${GEMINI_MODEL:-"gemini-3-pro-preview"}
TEMPERATURE=${TEMPERATURE:-"1.0"}
MAX_TOKENS=${MAX_TOKENS:-1024}
SPLIT=${SPLIT:-"train"}

OUTPUT_DIR="${SCRATCH_BASE}/logs/sft_trajectories"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

CORPUS_FILE="${SCRATCH_BASE}/logs/tree_build/subsection_corpus.json"
OUTPUT_FILE="${OUTPUT_DIR}/sft_trajectories_test.json"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== SFT Trajectory Generation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Config: max_examples=${MAX_EXAMPLES}, num_samples=${NUM_SAMPLES}"
echo "Config: m=${MAX_TURNS}, n=${TOPK}, model=${GEMINI_MODEL}"
echo "Output: ${OUTPUT_FILE}"

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

module load proxy/default
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

# Load secrets
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
# RETRIEVER CONFIGURATION
# ============================================================================
CORPUS_DATA="${DATA_DIR}/arxiv_wikiformat_with_ids.jsonl"
RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
INDEX_FILE="${DATA_DIR}/qwen3_4b_embed/qwen3_Flat.index"

if [ ! -f "${INDEX_FILE}" ]; then
    echo "ERROR: FAISS index not found at ${INDEX_FILE}"
    exit 1
fi

if [ ! -f "${CORPUS_FILE}" ]; then
    echo "ERROR: Subsection corpus not found at ${CORPUS_FILE}"
    exit 1
fi

# ============================================================================
# START RETRIEVER (background on same node)
# ============================================================================
echo "=== Starting retriever ==="

source /usr/licensed/anaconda3/2024.10/etc/profile.d/conda.sh
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"
conda activate retriever

python /home/hk4638/SkyRL/skyrl-train/examples/citation_prediction/retriever/retrieval_server.py \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_DATA}" \
    --topk "${TOPK}" \
    --retriever_name qwen3 \
    --retriever_model "${RETRIEVER_MODEL_PATH}" \
    --faiss_gpu \
    > "${LOG_DIR}/retriever_sft_${SLURM_JOB_ID}.log" 2>&1 &

RETRIEVER_PID=$!

RETRIEVER_URL="http://localhost:8000/retrieve"
echo "Waiting for retriever at ${RETRIEVER_URL}..."

MAX_RETRIES=60
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -X POST "${RETRIEVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "topk": 1}' > /dev/null 2>&1; then
        echo "Retriever ready!"
        break
    fi
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: Retriever failed to start!"
    cat "${LOG_DIR}/retriever_sft_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# RUN SFT TRAJECTORY GENERATION
# ============================================================================
echo "=== Running SFT trajectory generation ==="

# Switch back to the main venv for the generation script
conda deactivate

/home/hk4638/.local/bin/uv run --active --frozen \
    python "${V2_DIR}/data_tree/generate_sft_trajectories.py" \
        --corpus "${CORPUS_FILE}" \
        --search_url "${RETRIEVER_URL}" \
        --output "${OUTPUT_FILE}" \
        --split "${SPLIT}" \
        --max_examples "${MAX_EXAMPLES}" \
        --num_samples "${NUM_SAMPLES}" \
        --max_turns "${MAX_TURNS}" \
        --topk "${TOPK}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --recall_threshold "${RECALL_THRESHOLD}" \
        --gemini_model "${GEMINI_MODEL}"

GEN_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
kill ${RETRIEVER_PID} 2>/dev/null || true

echo "=== SFT trajectory generation complete ==="
echo "End time: $(date)"
echo "Exit code: ${GEN_EXIT_CODE}"
echo "Output: ${OUTPUT_FILE}"

exit ${GEN_EXIT_CODE}
