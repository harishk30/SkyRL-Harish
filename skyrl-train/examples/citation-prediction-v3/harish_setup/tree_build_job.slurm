#!/bin/bash
# ============================================================================
# Adaptive tree decomposition of V3 subsections.
#
# 2-node architecture:
#   Node 0: Retriever (4 GPUs for FAISS sharding)
#   Node 1: vLLM inference (4 GPUs) + runs adaptive_decompose.py
#
# Uses the v2 adaptive_decompose.py directly (v3 corpus is schema-compatible).
#
# Env vars (optional):
#   MAX_EXAMPLES     Number of subsections to process (default: 50)
#   NUM_SAMPLES      Rollouts per node eval (k; default: 8)
#   MAX_TURNS        Search turns per rollout (m; default: 4)
#   TOPK             Retriever results per search (n; default: 5)
#   RECALL_THRESHOLD Tractable threshold (default: 0.3)
#   MIN_CITATIONS    Stop splitting below this (default: 3)
#   MAX_DEPTH        Max tree depth (default: 3)
#   SPLIT            Data split to process (default: train)
#
# Prerequisites:
#   - GEMINI_API_KEY in .env file or environment
#   - subsection_corpus_v3.json built on login node
#
# Usage:
#   sbatch tree_build_job.slurm
#   MAX_EXAMPLES=10 sbatch tree_build_job.slurm   # small test
# ============================================================================

#SBATCH --job-name=tree-build-v3
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/tree-build-v3_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/tree-build-v3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-short

set -e

cd /home/hk4638/SkyRL/skyrl-train

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction"
LOG_DIR="${SCRATCH_BASE}/logs"
# Uses v2 adaptive_decompose.py (v3 corpus is schema-compatible)
V2_DIR="/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v2"

# Optional env vars with defaults
MAX_EXAMPLES=${MAX_EXAMPLES:-50}
NUM_SAMPLES=${NUM_SAMPLES:-8}
MAX_TURNS=${MAX_TURNS:-4}
TOPK=${TOPK:-5}
RECALL_THRESHOLD=${RECALL_THRESHOLD:-"0.3"}
MIN_CITATIONS=${MIN_CITATIONS:-2}
MAX_DEPTH=${MAX_DEPTH:-3}
SPLIT=${SPLIT:-"train"}
SEED=${SEED:-42}
TEMPERATURE=${TEMPERATURE:-"1.0"}
GEMINI_MODEL=${GEMINI_MODEL:-"gemini-3-flash-preview"}
MAX_HEADING_REFINE=${MAX_HEADING_REFINE:-1}
PROMPT_IDS_FILE=${PROMPT_IDS_FILE:-""}

# Output paths
OUTPUT_DIR="${SCRATCH_BASE}/logs/tree_build_v3"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

CORPUS_FILE="${OUTPUT_DIR}/subsection_corpus_v3.json"
TREES_OUTPUT="${OUTPUT_DIR}/trees.json"
CHECKPOINT_FILE="${OUTPUT_DIR}/trees_checkpoint.jsonl"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== V3 Tree Build: Adaptive Decomposition ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo "Config: max_examples=${MAX_EXAMPLES}, k=${NUM_SAMPLES}, m=${MAX_TURNS}, n=${TOPK}"
echo "Config: threshold=${RECALL_THRESHOLD}, min_cites=${MIN_CITATIONS}, max_depth=${MAX_DEPTH}"
echo "Config: split=${SPLIT}, gemini=${GEMINI_MODEL}"
echo "Output: ${OUTPUT_DIR}"

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

MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

module load proxy/default

# Load secrets (GEMINI_API_KEY, WANDB_API_KEY)
ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env"
if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment from ${ENV_FILE}"
    set -a && source "${ENV_FILE}" && set +a
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set. Add it to .env file."
    exit 1
fi

# Get node list
NODELIST=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
RETRIEVER_NODE=${NODELIST[0]}
VLLM_NODE=${NODELIST[1]}

echo "Retriever node: ${RETRIEVER_NODE}"
echo "vLLM node: ${VLLM_NODE}"

export NO_PROXY="localhost,127.0.0.1,${RETRIEVER_NODE},${VLLM_NODE}"
export no_proxy="$NO_PROXY"

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

echo "Index: ${INDEX_FILE}"
echo "Corpus: ${CORPUS_DATA}"

# ============================================================================
# CHECK SUBSECTION CORPUS
# ============================================================================
if [ ! -f "${CORPUS_FILE}" ]; then
    echo "ERROR: V3 subsection corpus not found at ${CORPUS_FILE}"
    echo "Build it on login node using the v3 data pipeline first."
    exit 1
fi
echo "Subsection corpus: ${CORPUS_FILE}"

# ============================================================================
# START RETRIEVER
# ============================================================================
echo "=== Starting retriever on ${RETRIEVER_NODE} ==="

srun --nodes=1 --ntasks=1 --nodelist=${RETRIEVER_NODE} \
    /bin/bash -c "
        source /usr/licensed/anaconda3/2024.10/etc/profile.d/conda.sh
        export CONDA_ENVS_PATH=${SCRATCH_BASE}/conda/envs
        conda activate retriever
        export HF_HOME=${SCRATCH_BASE}/huggingface
        export TRANSFORMERS_CACHE=${SCRATCH_BASE}/huggingface/transformers
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

        echo 'Starting retriever on \$(hostname) with 4 GPUs...'
        python /home/hk4638/SkyRL/skyrl-train/examples/citation_prediction/retriever/retrieval_server.py \
            --index_path ${INDEX_FILE} \
            --corpus_path ${CORPUS_DATA} \
            --topk ${TOPK} \
            --retriever_name qwen3 \
            --retriever_model ${RETRIEVER_MODEL_PATH} \
            --faiss_gpu
    " > "${LOG_DIR}/retriever_tree_v3_${SLURM_JOB_ID}.log" 2>&1 &

RETRIEVER_PID=$!

RETRIEVER_URL="http://${RETRIEVER_NODE}:8000/retrieve"
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
    cat "${LOG_DIR}/retriever_tree_v3_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# RUN ADAPTIVE DECOMPOSE ON VLLM NODE
# ============================================================================
echo "=== Running adaptive decompose on ${VLLM_NODE} ==="

srun --nodes=1 --ntasks=1 --nodelist=${VLLM_NODE} \
    /bin/bash -c "
        export PATH=/home/hk4638/.local/bin:\$PATH
        export VIRTUAL_ENV=/home/hk4638/SkyRL/skyrl-train/.venv
        export UV_CACHE_DIR=${SCRATCH_BASE}/uv_cache
        export UV_LINK_MODE=copy
        export HF_HOME=${SCRATCH_BASE}/huggingface
        export HF_DATASETS_CACHE=${SCRATCH_BASE}/huggingface/datasets
        export HUGGINGFACE_HUB_CACHE=${SCRATCH_BASE}/huggingface/hub
        export TRANSFORMERS_CACHE=${SCRATCH_BASE}/huggingface/transformers
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export GEMINI_API_KEY=${GEMINI_API_KEY}
        export http_proxy=${http_proxy}
        export https_proxy=${https_proxy}
        export NO_PROXY='${NO_PROXY}'
        export no_proxy='${NO_PROXY}'

        cd ${V2_DIR}/data_tree

        /home/hk4638/.local/bin/uv run --active --frozen --extra vllm \
            python adaptive_decompose.py \
                --corpus ${CORPUS_FILE} \
                --search_url ${RETRIEVER_URL} \
                --model ${MODEL_PATH} \
                --output ${TREES_OUTPUT} \
                --checkpoint ${CHECKPOINT_FILE} \
                --split ${SPLIT} \
                --max_examples ${MAX_EXAMPLES} \
                --num_samples ${NUM_SAMPLES} \
                --max_turns ${MAX_TURNS} \
                --topk ${TOPK} \
                --temperature ${TEMPERATURE} \
                --seed ${SEED} \
                --min_citations ${MIN_CITATIONS} \
                --max_depth ${MAX_DEPTH} \
                --recall_threshold ${RECALL_THRESHOLD} \
                --gemini_model ${GEMINI_MODEL} \
                --max_heading_refine ${MAX_HEADING_REFINE} \
                ${PROMPT_IDS_FILE:+--prompt_ids_file ${PROMPT_IDS_FILE}} \
                --tensor_parallel_size 4 \
                --gpu_memory_utilization 0.9
    "

DECOMPOSE_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
kill ${RETRIEVER_PID} 2>/dev/null || true

echo "=== Tree build complete ==="
echo "End time: $(date)"
echo "Exit code: ${DECOMPOSE_EXIT_CODE}"
echo "Trees: ${TREES_OUTPUT}"
echo "Checkpoint: ${CHECKPOINT_FILE}"

exit ${DECOMPOSE_EXIT_CODE}
