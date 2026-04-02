#!/bin/bash
# ============================================================================
# Standalone SFT trajectory generation for L40 cluster.
#
# Single script: starts retriever (1 GPU) + runs generation (CPU).
# Designed for clusters with infinite runtime and no queue time.
# Handles the full dataset in one run, respecting Gemini API rate limits.
#
# L40 (40GB) is enough for:
#   - Qwen3-Embedding-4B (~8GB) + FAISS index (~4GB) = ~12GB on 1 GPU
#   - Generation script uses CPU only (Gemini API calls)
#
# Usage:
#   # Set paths first (see CONFIGURATION section), then:
#   bash run_sft_l40.sh
#
#   # Or override:
#   GEMINI_API_KEY=... SAMPLE_FRAC=0.25 bash run_sft_l40.sh
# ============================================================================

set -e
export PYTHONUNBUFFERED=1

# ============================================================================
# CONFIGURATION — EDIT THESE FOR YOUR L40 CLUSTER
# ============================================================================

# Base directory where data + code live
DATA_BASE="${DATA_BASE:-/n/fs/vision-mix/hk4638/retriever_embeddings}"
CODE_DIR="${CODE_DIR:-${DATA_BASE}/code}"

# Data files
CORPUS_FILE="${DATA_BASE}/subsection_corpus_v3_all.json"
ARXIV_CORPUS="${DATA_BASE}/arxiv_wikiformat_with_ids.jsonl"
FAISS_INDEX="${DATA_BASE}/qwen3_Flat.index"
EMBEDDING_MODEL="${DATA_BASE}/Qwen3-Embedding-4B"

# Output
OUTPUT_DIR="${DATA_BASE}/sft_trajectories_v3"
OUTPUT_FILE="${OUTPUT_DIR}/sft_trajectories_full.json"

# Gemini config
# Load .env if present
if [ -f "${CODE_DIR}/.env" ]; then
    set -a && source "${CODE_DIR}/.env" && set +a
fi
GEMINI_API_KEY="${GEMINI_API_KEY:?Set GEMINI_API_KEY or copy .env to ${CODE_DIR}/}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash-preview}"

# Generation config (matching training: m=4, n=10)
SAMPLE_FRAC="${SAMPLE_FRAC:-0.5}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
SEED="${SEED:-42}"
MAX_TURNS="${MAX_TURNS:-4}"
TOPK="${TOPK:-10}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PROMPT_VERSION="${PROMPT_VERSION:-v3}"

# GPU for retriever (single L40)
RETRIEVER_GPU="${RETRIEVER_GPU:-0}"

# Python — set to your venv/conda python with google-genai, requests, faiss, torch
PYTHON="${PYTHON:-python3}"

# ============================================================================
# SETUP
# ============================================================================
echo "=== SFT Trajectory Generation (L40 standalone) ==="
echo "Start time: $(date)"
echo "Config: sample_frac=${SAMPLE_FRAC}, num_samples=${NUM_SAMPLES}"
echo "Config: max_turns=${MAX_TURNS}, topk=${TOPK}, model=${GEMINI_MODEL}"
echo "Output: ${OUTPUT_FILE}"

mkdir -p "${OUTPUT_DIR}"

# Validate files exist
for f in "${CORPUS_FILE}" "${ARXIV_CORPUS}" "${FAISS_INDEX}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

if [ ! -d "${EMBEDDING_MODEL}" ]; then
    echo "ERROR: Embedding model dir not found: ${EMBEDDING_MODEL}"
    exit 1
fi

# ============================================================================
# START RETRIEVER (background, 1 GPU)
# ============================================================================
echo "=== Starting retriever on GPU ${RETRIEVER_GPU} ==="

CUDA_VISIBLE_DEVICES=${RETRIEVER_GPU} ${PYTHON} \
    "${CODE_DIR}/retriever/retrieval_server.py" \
    --index_path "${FAISS_INDEX}" \
    --corpus_path "${ARXIV_CORPUS}" \
    --topk "${TOPK}" \
    --retriever_name qwen3 \
    --retriever_model "${EMBEDDING_MODEL}" \
    --faiss_gpu \
    > "${OUTPUT_DIR}/retriever.log" 2>&1 &

RETRIEVER_PID=$!
RETRIEVER_URL="http://localhost:8000/retrieve"

echo "Waiting for retriever (PID ${RETRIEVER_PID})..."
MAX_RETRIES=120
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -X POST "${RETRIEVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "topk": 1}' > /dev/null 2>&1; then
        echo "Retriever ready!"
        break
    fi
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: Retriever failed to start. Check ${OUTPUT_DIR}/retriever.log"
    kill ${RETRIEVER_PID} 2>/dev/null || true
    exit 1
fi

# ============================================================================
# RUN GENERATION (CPU, full dataset, rate-limited)
# ============================================================================
echo "=== Running SFT trajectory generation ==="
echo "This will take several days due to Gemini API daily rate limits (10K RPD)."
echo "Progress is saved incrementally every 5 queries."
echo ""

export GEMINI_API_KEY

${PYTHON} "${CODE_DIR}/training/generate_sft_trajectories_v3.py" \
    --corpus "${CORPUS_FILE}" \
    --search_url "${RETRIEVER_URL}" \
    --output "${OUTPUT_FILE}" \
    --split train \
    --sample_frac "${SAMPLE_FRAC}" \
    --seed "${SEED}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_turns "${MAX_TURNS}" \
    --topk "${TOPK}" \
    --max_tokens "${MAX_TOKENS}" \
    --gemini_model "${GEMINI_MODEL}" \
    --prompt_version "${PROMPT_VERSION}"

GEN_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
kill ${RETRIEVER_PID} 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "End time: $(date)"
echo "Exit code: ${GEN_EXIT_CODE}"
echo "Output: ${OUTPUT_FILE}"

exit ${GEN_EXIT_CODE}
