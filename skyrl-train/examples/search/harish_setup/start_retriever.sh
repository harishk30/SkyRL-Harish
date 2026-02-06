#!/bin/bash
# Start the FAISS retrieval server
# Prerequisites: Run setup_scratch.sh and setup_retriever.sh first

set -x

SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/searchR1"

# Set HuggingFace cache
export HF_HOME="${SCRATCH_BASE}/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/huggingface/transformers"

# Activate retriever environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retriever

echo "Starting retrieval server..."
echo "Index: ${DATA_DIR}/e5_Flat.index"
echo "Corpus: ${DATA_DIR}/wiki-18.jsonl"
echo ""
echo "Server will be available at: http://127.0.0.1:8000/retrieve"
echo "Logs redirected to retrieval_server.log"
echo ""

cd /home/hk4638/SkyRL/skyrl-train

python examples/search/retriever/retrieval_server.py \
  --index_path "${DATA_DIR}/e5_Flat.index" \
  --corpus_path "${DATA_DIR}/wiki-18.jsonl" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu \
  > retrieval_server.log 2>&1

echo "Server stopped. Check retrieval_server.log for details."
