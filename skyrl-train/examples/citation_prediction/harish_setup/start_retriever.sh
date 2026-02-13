#!/bin/bash
# Start the FAISS retrieval server
# Prerequisites: Run setup_scratch.sh and setup_retriever.sh first

set -x

SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction"

# Set HuggingFace cache
export HF_HOME="${SCRATCH_BASE}/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/huggingface/transformers"

# Embedding model toggle: qwen3_4b (default for citation prediction) or qwen3_06b
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_4b"}

if [ "$EMBEDDING_MODEL" = "qwen3_4b" ]; then
    MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
    INDEX_FILE="${DATA_DIR}/qwen3_4b_embed/qwen3_Flat.index"
    CORPUS_PATH="${DATA_DIR}/arxiv_wikiformat_with_ids.jsonl"
    RETRIEVER_NAME=qwen3
elif [ "$EMBEDDING_MODEL" = "qwen3_06b" ]; then
    MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    INDEX_FILE="${DATA_DIR}/qwen3_06b_embed/qwen3_Flat.index"
    CORPUS_PATH="${DATA_DIR}/arxiv_wikiformat_with_ids.jsonl"
    RETRIEVER_NAME=qwen3
fi

# Activate retriever environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retriever

echo "Starting retrieval server..."
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Index: ${INDEX_FILE}"
echo "Corpus: ${CORPUS_PATH}"
echo ""
echo "Server will be available at: http://127.0.0.1:8000/retrieve"
echo "Logs redirected to retrieval_server.log"
echo ""

cd /home/hk4638/SkyRL/skyrl-train

python examples/citation_prediction/retriever/retrieval_server.py \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_PATH}" \
  --topk 3 \
  --retriever_name $RETRIEVER_NAME \
  --retriever_model "${MODEL_PATH}" \
  --faiss_gpu \
  > retrieval_server.log 2>&1

echo "Server stopped. Check retrieval_server.log for details."
