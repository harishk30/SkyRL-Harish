#!/bin/bash
# ============================================================================
# Quick test: Retriever with FAISS sharding across 4 GPUs
# ============================================================================

#SBATCH --job-name=test-retriever
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/test-retriever_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/test-retriever_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-test

set -e

SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction"

export HF_HOME="${SCRATCH_BASE}/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/huggingface/transformers"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"

echo "=== Testing retriever with 4-GPU FAISS sharding ==="
echo "Node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

source $(conda info --base)/etc/profile.d/conda.sh
conda activate retriever

echo ""
echo "=== Starting retriever server ==="
echo "Index: ${DATA_DIR}/e5_Flat.index"
echo "Index size: $(ls -lh ${DATA_DIR}/e5_Flat.index | awk '{print $5}')"
echo "Expected: ~15GB per GPU after FP16 + sharding across 4 GPUs"

# Don't set CUDA_VISIBLE_DEVICES - let FAISS shard across all 4 GPUs
python /home/hk4638/SkyRL/skyrl-train/examples/citation_prediction/retriever/retrieval_server.py \
    --index_path ${DATA_DIR}/e5_Flat.index \
    --corpus_path ${DATA_DIR}/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu &

SERVER_PID=$!

# Wait for server to start (loading + sharding takes time)
echo "Waiting for server to start (loading 61GB index + sharding)..."
sleep 120

# Test a query
echo ""
echo "=== Testing retrieval ==="
RESULT=$(curl -s -X POST "http://localhost:8000/retrieve" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the capital of France?", "topk": 3}')

if [ -n "$RESULT" ]; then
    echo "SUCCESS! Retriever is working."
    echo "Sample result: ${RESULT:0:500}..."
else
    echo "FAILED: No response from retriever"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Show GPU memory usage after loading
echo ""
echo "=== GPU Memory Usage (should be ~15GB each) ==="
nvidia-smi

# Benchmark latency
echo ""
echo "=== Benchmarking retrieval latency ==="
for i in 1 2 3 4 5; do
    START=$(date +%s.%N)
    curl -s -X POST "http://localhost:8000/retrieve" \
        -H "Content-Type: application/json" \
        -d '{"query": "Who invented the telephone?", "topk": 3}' > /dev/null
    END=$(date +%s.%N)
    echo "Query $i: $(echo "$END - $START" | bc) seconds"
done

kill $SERVER_PID 2>/dev/null || true
echo "=== Test complete ==="
