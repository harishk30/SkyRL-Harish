#!/bin/bash
# Setup script for Citation Prediction data download on scratch filesystem
# Target: /scratch/gpfs/ZHUANGL/hk4638
#
# This script:
# 1. Sets up cache directories for conda and uv on scratch
# 2. Downloads the Citation Prediction dataset
# 3. Downloads and reconstructs the FAISS index
# 4. Provides instructions for retriever environment setup

set -e  # Exit on error
set -x  # Print commands

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

#######################################
# CONFIGURATION
#######################################
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction"
CONDA_CACHE="${SCRATCH_BASE}/conda"
UV_CACHE="${SCRATCH_BASE}/uv_cache"
HF_CACHE="${SCRATCH_BASE}/huggingface"

#######################################
# Step 1: Create directory structure
#######################################
echo "=== Creating directory structure ==="
mkdir -p "${DATA_DIR}"
mkdir -p "${CONDA_CACHE}"
mkdir -p "${UV_CACHE}"
mkdir -p "${HF_CACHE}"

#######################################
# Step 2: Set environment variables
#######################################
echo "=== Setting environment variables ==="

# These should also be added to your ~/.bashrc or job scripts
export CONDA_PKGS_DIRS="${CONDA_CACHE}/pkgs"
export CONDA_ENVS_PATH="${CONDA_CACHE}/envs"
export UV_CACHE_DIR="${UV_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"

# Print for verification
echo "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
echo "CONDA_ENVS_PATH=${CONDA_ENVS_PATH}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "HF_HOME=${HF_HOME}"

#######################################
# Step 3: Download Citation Prediction dataset
#######################################
echo "=== Downloading Citation Prediction dataset ==="
cd /home/hk4638/SkyRL/skyrl-train

# Download and process the dataset
uv run --isolated examples/citation_prediction/citation_prediction_dataset.py --local_dir "${DATA_DIR}"

echo "Dataset downloaded to: ${DATA_DIR}"
ls -lh "${DATA_DIR}"/short/*.parquet "${DATA_DIR}"/extended/*.parquet 2>/dev/null || ls -lh "${DATA_DIR}"/*.parquet 2>/dev/null || echo "No parquets found yet"

#######################################
# Step 4: Download FAISS index
#######################################
echo "=== Downloading FAISS index (this may take a while) ==="

# Download index parts and corpus
uv run --isolated examples/citation_prediction/citation_prediction_download.py --local_dir "${DATA_DIR}"

# Reconstruct the full index from parts
echo "=== Reconstructing FAISS index ==="
cat "${DATA_DIR}"/part_* > "${DATA_DIR}/e5_Flat.index"

# Remove parts to save space (optional)
# rm "${DATA_DIR}"/part_*

# Decompress corpus
echo "=== Decompressing Wikipedia corpus ==="
gzip -d "${DATA_DIR}/wiki-18.jsonl.gz"

echo "=== Download complete ==="
echo "Data directory contents:"
ls -lh "${DATA_DIR}"

#######################################
# Step 5: Print environment setup for .bashrc
#######################################
echo ""
echo "=============================================="
echo "Add the following to your ~/.bashrc or job scripts:"
echo "=============================================="
cat << 'EOF'

# SkyRL Citation Prediction cache configuration
export SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda/pkgs"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"
export UV_CACHE_DIR="${SCRATCH_BASE}/uv_cache"
export HF_HOME="${SCRATCH_BASE}/huggingface"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_BASE}/huggingface/hub"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/huggingface/transformers"

# Ray configuration for uv
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

EOF

echo ""
echo "=============================================="
echo "Next steps:"
echo "=============================================="
echo "1. Add the above exports to your ~/.bashrc"
echo "2. Create the retriever conda environment (see setup_retriever.sh)"
echo "3. Update run_citation_prediction.sh to use DATA_DIR=${DATA_DIR}"
echo "=============================================="
