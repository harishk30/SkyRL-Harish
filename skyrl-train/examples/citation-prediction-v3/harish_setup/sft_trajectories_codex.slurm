#!/bin/bash
# ============================================================================
# Generate SFT trajectories using Codex (OpenAI terminal agent) with
# retriever-in-the-loop.
#
# Architecture:
#   - Retriever (background): FAISS + Qwen3-Embedding on GPU
#   - MCP search server: STDIO server spawned per Codex invocation
#   - Codex exec: non-interactive agent with --json event capture
#   - Orchestrator: Python async loop over dataset queries
#
# Usage:
#   sbatch sft_trajectories_codex.slurm
#
# Override defaults with env vars:
#   MAX_EXAMPLES=100 NUM_SAMPLES=3 CODEX_MODEL=o3 sbatch ...
# ============================================================================

#SBATCH --job-name=sft-codex-v3
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/sft-codex-v3_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/sft-codex-v3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
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
SCRIPT_DIR="/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/training"

MAX_EXAMPLES=${MAX_EXAMPLES:-50}
NUM_SAMPLES=${NUM_SAMPLES:-3}
MAX_TURNS=${MAX_TURNS:-6}
TOPK=${TOPK:-5}
RECALL_THRESHOLD=${RECALL_THRESHOLD:-"0.3"}
CODEX_MODEL=${CODEX_MODEL:-"o3"}
CONCURRENCY=${CONCURRENCY:-1}
TIMEOUT=${TIMEOUT:-300}
SPLIT=${SPLIT:-"train"}

OUTPUT_DIR="${SCRATCH_BASE}/logs/sft_codex_v3"
DEBUG_DIR="${OUTPUT_DIR}/debug"
CORPUS_FILE="${SCRATCH_BASE}/data/citation_prediction_v3/subsection_corpus_v3.json"
OUTPUT_FILE="${OUTPUT_DIR}/sft_trajectories_codex.json"

mkdir -p "${OUTPUT_DIR}" "${DEBUG_DIR}" "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== Codex SFT Trajectory Generation (V3) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Config: max_examples=${MAX_EXAMPLES}, num_samples=${NUM_SAMPLES}"
echo "Config: max_turns=${MAX_TURNS}, topk=${TOPK}, model=${CODEX_MODEL}"
echo "Config: concurrency=${CONCURRENCY}, timeout=${TIMEOUT}s"
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

# Internet access for Codex (OpenAI API) and retriever model
module load proxy/default
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

# Load secrets
ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env"
if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment from ${ENV_FILE}"
    set -a && source "${ENV_FILE}" && set +a
fi

# Codex needs CODEX_API_KEY or OPENAI_API_KEY
if [ -z "${CODEX_API_KEY}" ] && [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: Set CODEX_API_KEY or OPENAI_API_KEY in ${ENV_FILE}"
    exit 1
fi

# ============================================================================
# NODE.JS + CODEX CLI
# ============================================================================
echo "=== Setting up Codex CLI ==="

# Load Node.js (Della module or local install)
if command -v module &>/dev/null; then
    module load anaconda3/2024.10 2>/dev/null || true
    # Try loading nodejs module if available
    module load nodejs 2>/dev/null || true
fi

# Check if codex is installed
if ! command -v codex &>/dev/null; then
    echo "codex CLI not found, installing via npm..."
    npm install -g @openai/codex 2>&1 || {
        echo "ERROR: Failed to install codex. Install manually: npm install -g @openai/codex"
        exit 1
    }
fi
echo "Codex CLI: $(which codex)"

# ============================================================================
# MCP SEARCH SERVER SETUP
# ============================================================================
echo "=== Setting up MCP search server ==="

# Create a venv for the MCP server if needed
MCP_VENV="${SCRATCH_BASE}/conda/envs/mcp_server_venv"
MCP_PYTHON="${MCP_VENV}/bin/python"

if [ ! -f "${MCP_PYTHON}" ]; then
    echo "Creating MCP server venv at ${MCP_VENV}..."
    /home/hk4638/.local/bin/uv venv "${MCP_VENV}" --python 3.11
    /home/hk4638/.local/bin/uv pip install --python "${MCP_PYTHON}" mcp requests
fi

# Write Codex config with MCP search server
CODEX_CONFIG_DIR="${HOME}/.codex"
mkdir -p "${CODEX_CONFIG_DIR}"

RETRIEVER_URL="http://localhost:8000/retrieve"

cat > "${CODEX_CONFIG_DIR}/config.toml" << TOML
# Auto-generated by sft_trajectories_codex.slurm (job ${SLURM_JOB_ID})
# MCP search server for arxiv retrieval

[mcp_servers.arxiv_search]
command = "${MCP_PYTHON}"
args = ["${SCRIPT_DIR}/mcp_search_server.py"]
enabled_tools = ["search"]
tool_timeout_sec = 45
startup_timeout_sec = 15

[mcp_servers.arxiv_search.env]
RETRIEVER_URL = "${RETRIEVER_URL}"
TOPK = "${TOPK}"
MAX_SEARCHES = "${MAX_TURNS}"
TOML

echo "Codex config written to ${CODEX_CONFIG_DIR}/config.toml"
cat "${CODEX_CONFIG_DIR}/config.toml"

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
    echo "ERROR: V3 subsection corpus not found at ${CORPUS_FILE}"
    echo "Trying tree_build_v3 location..."
    CORPUS_FILE="${SCRATCH_BASE}/logs/tree_build_v3/subsection_corpus_v3.json"
    if [ ! -f "${CORPUS_FILE}" ]; then
        echo "ERROR: Corpus not found at fallback location either"
        exit 1
    fi
fi
echo "Corpus: ${CORPUS_FILE}"

# ============================================================================
# START RETRIEVER (background)
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
    > "${LOG_DIR}/retriever_codex_v3_${SLURM_JOB_ID}.log" 2>&1 &

RETRIEVER_PID=$!

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
    cat "${LOG_DIR}/retriever_codex_v3_${SLURM_JOB_ID}.log" | tail -50
    exit 1
fi

# ============================================================================
# RUN CODEX SFT TRAJECTORY GENERATION
# ============================================================================
echo "=== Running Codex SFT trajectory generation ==="

# Switch to training venv for the orchestrator
conda deactivate

/home/hk4638/.local/bin/uv run --active --frozen \
    python "${SCRIPT_DIR}/generate_sft_trajectories_codex.py" \
        --corpus "${CORPUS_FILE}" \
        --output "${OUTPUT_FILE}" \
        --split "${SPLIT}" \
        --max_examples "${MAX_EXAMPLES}" \
        --num_samples "${NUM_SAMPLES}" \
        --max_turns "${MAX_TURNS}" \
        --recall_threshold "${RECALL_THRESHOLD}" \
        --codex_model "${CODEX_MODEL}" \
        --concurrency "${CONCURRENCY}" \
        --timeout "${TIMEOUT}" \
        --debug_dir "${DEBUG_DIR}"

GEN_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
kill ${RETRIEVER_PID} 2>/dev/null || true

echo "=== Codex SFT trajectory generation complete ==="
echo "End time: $(date)"
echo "Exit code: ${GEN_EXIT_CODE}"
echo "Output: ${OUTPUT_FILE}"
echo "Debug events: ${DEBUG_DIR}"

exit ${GEN_EXIT_CODE}
