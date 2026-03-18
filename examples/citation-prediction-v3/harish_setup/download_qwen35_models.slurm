#!/bin/bash
#SBATCH --job-name=download-qwen35
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/download_qwen35_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/download_qwen35_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu

set -e

module load proxy/default

export HF_HOME=/scratch/gpfs/ZHUANGL/hk4638/huggingface
export HF_HUB_OFFLINE=0
export VIRTUAL_ENV=/home/hk4638/SkyRL/.venv
export UV_CACHE_DIR=/scratch/gpfs/ZHUANGL/hk4638/uv_cache
export UV_LINK_MODE=copy
export PATH=/home/hk4638/.local/bin:$PATH

echo "=== Downloading Qwen3.5 models ==="
echo "Start time: $(date)"

echo ""
echo "--- Qwen/Qwen3.5-4B (~8GB) ---"
uv run --active --frozen huggingface-cli download Qwen/Qwen3.5-4B
echo "Snapshot:"
ls -la /scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/

echo ""
echo "--- Qwen/Qwen3.5-9B (~18GB) ---"
uv run --active --frozen huggingface-cli download Qwen/Qwen3.5-9B
echo "Snapshot:"
ls -la /scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/

echo ""
echo "--- Qwen/Qwen3.5-35B-A3B (checking if already cached) ---"
EXISTING_35B="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/b1fc3d59ae0ab1e4279e04a8dd0fc4dc361fc2b6"
if [ -d "$EXISTING_35B" ]; then
    echo "Already cached at: $EXISTING_35B"
else
    echo "Not cached, downloading..."
    uv run --active --frozen huggingface-cli download Qwen/Qwen3.5-35B-A3B
    echo "Snapshot:"
    ls -la /scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/
fi

echo ""
echo "=== All downloads complete ==="
echo "End time: $(date)"
echo ""
echo "Record snapshot hashes for eval scripts:"
echo "  Qwen3.5-4B:      $(ls /scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/ 2>/dev/null || echo 'NOT FOUND')"
echo "  Qwen3.5-9B:      $(ls /scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/ 2>/dev/null || echo 'NOT FOUND')"
echo "  Qwen3.5-35B-A3B: $(ls /scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ 2>/dev/null || echo 'NOT FOUND')"
