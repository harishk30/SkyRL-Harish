#!/bin/bash
#SBATCH --job-name=qwen3_4b_arxiv
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/qwen3_4b_arxiv_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/qwen3_4b_arxiv_%j.err

# ============================================================
# Build Qwen3-Embedding-4B FAISS index over arxiv corpus
# (with arxiv IDs prepended to contents)
#
# Prerequisites:
#   1. Run build_arxiv_corpus.py to create the ID-augmented corpus
#   2. Download Qwen3-Embedding-4B model (see Step 2 in CLAUDE.md)
#
# Usage:
#   sbatch examples/citation_prediction/data/build_4b_index.sbatch
# ============================================================

set -e

SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"

# Initialize conda and activate retriever env
source $(conda info --base)/etc/profile.d/conda.sh
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"
conda activate retriever

# CD to repo root
cd /home/hk4638/SkyRL/skyrl-train

# Environment setup
export HF_HOME="${SCRATCH_BASE}/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model path
QWEN3_4B_MODEL_PATH=$(find ${HF_HOME}/hub/models--Qwen--Qwen3-Embedding-4B/snapshots -maxdepth 1 -mindepth 1 -type d | head -1)
if [ -z "$QWEN3_4B_MODEL_PATH" ]; then
    echo "ERROR: Qwen3-Embedding-4B model not found. Download it first from the login node:"
    echo "  export HF_HOME=${SCRATCH_BASE}/huggingface"
    echo '  python -c "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3-Embedding-4B\")"'
    exit 1
fi

# Data paths (use absolute scratch paths, not symlinks)
CORPUS_PATH="${SCRATCH_BASE}/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl"
OUTPUT_DIR="${SCRATCH_BASE}/data/citation_prediction/qwen3_4b_embed"

if [ ! -f "$CORPUS_PATH" ]; then
    echo "ERROR: Corpus not found at $CORPUS_PATH"
    echo "Run build_arxiv_corpus.py first."
    exit 1
fi

echo "Building Qwen3-4B FAISS index for arxiv corpus..."
echo "  Model:  Qwen3-Embedding-4B ($QWEN3_4B_MODEL_PATH)"
echo "  Corpus: $CORPUS_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs:   $(nvidia-smi -L | wc -l)"
echo "  Max Length: 256"
echo "  Batch Size: 32 (per-GPU, scaled by DataParallel)"

mkdir -p "$OUTPUT_DIR"

python /home/hk4638/scratch/shared/retriever/index_builder.py \
    --retrieval_method qwen3 \
    --model_path "$QWEN3_4B_MODEL_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --save_dir "$OUTPUT_DIR" \
    --use_fp16 \
    --max_length 256 \
    --batch_size 32 \
    --pooling_method last_token \
    --faiss_type Flat \
    --save_embedding \
    --faiss_gpu

echo "Done! Index saved to: $OUTPUT_DIR/qwen3_Flat.index"
