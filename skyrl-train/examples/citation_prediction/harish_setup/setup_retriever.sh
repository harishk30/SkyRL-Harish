#!/bin/bash
# Setup script for the retriever conda environment on scratch filesystem
# Run this AFTER setup_scratch.sh
#
# This creates a separate conda environment for the FAISS retrieval server
# (Required because faiss-gpu conflicts with training dependencies)

set -e  # Exit on error
set -x  # Print commands

#######################################
# CONFIGURATION
#######################################
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
CONDA_CACHE="${SCRATCH_BASE}/conda"

# Set conda paths
export CONDA_PKGS_DIRS="${CONDA_CACHE}/pkgs"
export CONDA_ENVS_PATH="${CONDA_CACHE}/envs"

#######################################
# Step 1: Create retriever environment
#######################################
echo "=== Creating retriever conda environment ==="

# Create environment (will be stored in CONDA_ENVS_PATH)
conda create -n retriever python=3.10 -y

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retriever

#######################################
# Step 2: Install dependencies
#######################################
echo "=== Installing dependencies ==="

# Install numpy first to prevent version conflicts
conda install numpy==1.26.4 -y

# Install PyTorch with CUDA 12.4 support (good for A100s)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install transformers datasets pyserini huggingface_hub

# Install faiss-gpu (must use conda, not pip)
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

# Install API framework
pip install uvicorn fastapi

#######################################
# Step 3: Verify installation
#######################################
echo "=== Verifying installation ==="
python -c "import faiss; print(f'FAISS version: {faiss.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=============================================="
echo "Retriever environment setup complete!"
echo "=============================================="
echo ""
echo "Environment location: ${CONDA_ENVS_PATH}/retriever"
echo ""
echo "To activate: conda activate retriever"
echo "To start server: bash examples/citation_prediction/retriever/retrieval_launch.sh"
echo "=============================================="
