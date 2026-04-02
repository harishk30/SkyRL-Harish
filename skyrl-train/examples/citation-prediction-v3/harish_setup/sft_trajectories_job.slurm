#!/bin/bash
# ============================================================================
# Generate SFT trajectories using Gemini with retriever-in-the-loop (V3).
#
# Pass@20 rejection sampling on a chunk of queries. Submit multiple jobs
# to cover the full dataset in parallel.
#
# 1-node gpu-test job (<1 hour): retriever (background) + Gemini API calls.
#
# Usage:
#   # Single chunk (for testing):
#   sbatch sft_trajectories_job.slurm
#
#   # Parallel chunks via launcher:
#   bash launch_sft_jobs.sh
#
#   # Override:
#   CHUNK_ID=2 NUM_CHUNKS=20 sbatch sft_trajectories_job.slurm
# ============================================================================

#SBATCH --job-name=sft-traj-v3
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/sft-traj-v3_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/sft-traj-v3_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-test

set -e
export PYTHONUNBUFFERED=1

cd /home/hk4638/SkyRL/skyrl-train

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction"
LOG_DIR="${SCRATCH_BASE}/logs"
V3_DIR="/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3"

# Chunking
CHUNK_ID=${CHUNK_ID:-0}
NUM_CHUNKS=${NUM_CHUNKS:-1}

# Pass@k config
SAMPLE_FRAC=${SAMPLE_FRAC:-"0.5"}
NUM_SAMPLES=${NUM_SAMPLES:-20}
SEED=${SEED:-42}

# Env-matching config
MAX_TURNS=${MAX_TURNS:-4}
TOPK=${TOPK:-10}
MAX_TOKENS=${MAX_TOKENS:-4096}
RECALL_THRESHOLD=${RECALL_THRESHOLD:-"0.0"}
PROMPT_VERSION=${PROMPT_VERSION:-"v3"}
SPLIT=${SPLIT:-"train"}

# Gemini config
GEMINI_MODEL=${GEMINI_MODEL:-"gemini-3-flash-preview"}
# Rate limits split for 3 concurrent gpu-test jobs (QOS max)
RPM=${RPM:-300}
RPD=${RPD:-3000}

OUTPUT_DIR="${SCRATCH_BASE}/logs/sft_trajectories_v3"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

CORPUS_FILE="${SCRATCH_BASE}/data/citation_prediction_v3/subsection_corpus_v3_all.json"

OUTPUT_FILE="${OUTPUT_DIR}/sft_chunk${CHUNK_ID}_of_${NUM_CHUNKS}.json"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== V3 SFT Trajectory Generation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Chunk: ${CHUNK_ID}/${NUM_CHUNKS}"
echo "Config: sample_frac=${SAMPLE_FRAC}, num_samples=${NUM_SAMPLES}, seed=${SEED}"
echo "Config: max_turns=${MAX_TURNS}, topk=${TOPK}, model=${GEMINI_MODEL}"
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
export http_proxy=http://172.28.15.7:8080
export https_proxy=http://172.28.15.7:8080
export HTTP_PROXY=http://172.28.15.7:8080
export HTTPS_PROXY=http://172.28.15.7:8080
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

# Load secrets
ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a && source "${ENV_FILE}" && set +a
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set"
    exit 1
fi

# ============================================================================
# RETRIEVER
# ============================================================================
CORPUS_DATA="${DATA_DIR}/arxiv_wikiformat_with_ids.jsonl"
RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
INDEX_FILE="${DATA_DIR}/qwen3_4b_embed/qwen3_Flat.index"

if [ ! -f "${INDEX_FILE}" ]; then
    echo "ERROR: FAISS index not found at ${INDEX_FILE}"
    exit 1
fi
if [ ! -f "${CORPUS_FILE}" ]; then
    echo "ERROR: Corpus not found at ${CORPUS_FILE}"
    exit 1
fi

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
    > "${LOG_DIR}/retriever_sft_v3_${SLURM_JOB_ID}.log" 2>&1 &

RETRIEVER_PID=$!
RETRIEVER_URL="http://localhost:8000/retrieve"

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
    tail -50 "${LOG_DIR}/retriever_sft_v3_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# RUN GENERATION
# ============================================================================
echo "=== Running SFT trajectory generation (chunk ${CHUNK_ID}/${NUM_CHUNKS}) ==="

conda deactivate

/home/hk4638/.local/bin/uv run --active --no-sync \
    python "${V3_DIR}/training/generate_sft_trajectories_v3.py" \
        --corpus "${CORPUS_FILE}" \
        --search_url "${RETRIEVER_URL}" \
        --output "${OUTPUT_FILE}" \
        --split "${SPLIT}" \
        --sample_frac "${SAMPLE_FRAC}" \
        --seed "${SEED}" \
        --num_samples "${NUM_SAMPLES}" \
        --max_turns "${MAX_TURNS}" \
        --topk "${TOPK}" \
        --max_tokens "${MAX_TOKENS}" \
        --recall_threshold "${RECALL_THRESHOLD}" \
        --gemini_model "${GEMINI_MODEL}" \
        --prompt_version "${PROMPT_VERSION}" \
        --chunk_id "${CHUNK_ID}" \
        --num_chunks "${NUM_CHUNKS}" \
        --rpm "${RPM}" \
        --rpd "${RPD}"

GEN_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
kill ${RETRIEVER_PID} 2>/dev/null || true

echo "=== Done (chunk ${CHUNK_ID}/${NUM_CHUNKS}) ==="
echo "End time: $(date)"
echo "Exit code: ${GEN_EXIT_CODE}"
echo "Output: ${OUTPUT_FILE}"

exit ${GEN_EXIT_CODE}
