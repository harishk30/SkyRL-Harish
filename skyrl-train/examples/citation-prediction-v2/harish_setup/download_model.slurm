#!/bin/bash
#SBATCH --job-name=download-model
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/download_model_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/download_model_%j.err
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
export VIRTUAL_ENV=/home/hk4638/SkyRL/skyrl-train/.venv
export UV_CACHE_DIR=/scratch/gpfs/ZHUANGL/hk4638/uv_cache
export UV_LINK_MODE=copy
export PATH=/home/hk4638/.local/bin:$PATH

echo "Downloading Qwen/Qwen3-4B-Thinking-2507..."
uv run --active --frozen huggingface-cli download Qwen/Qwen3-4B-Thinking-2507

echo "Download complete. Snapshot path:"
ls -la /scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/
