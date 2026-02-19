# Harish's Citation Prediction Setup for Della Cluster

Scripts for training Citation Prediction on Princeton's Della cluster with data on `/scratch/gpfs/ZHUANGL/hk4638`.

## Quick Start

### Initial Setup (Already Done)
```bash
# Step 1: Download all data
bash examples/citation_prediction/harish_setup/setup_scratch.sh

# Step 2: Create retriever conda environment
bash examples/citation_prediction/harish_setup/setup_retriever.sh
```

### Run Training on Della
```bash
# Step 1: Add environment variables to ~/.bashrc
cat >> ~/.bashrc << 'EOF'
export SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"
export UV_CACHE_DIR="${SCRATCH_BASE}/uv_cache"
export HF_HOME="${SCRATCH_BASE}/huggingface"
export PATH="$HOME/.local/bin:$PATH"
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
EOF
source ~/.bashrc

# Step 2: Set WandB API key
export WANDB_API_KEY=your_key_here

# Step 3: Submit job from della-gpu login node
cd /home/hk4638/SkyRL/skyrl-train
sbatch examples/citation_prediction/harish_setup/train_citation_prediction.slurm

# Step 4: Monitor job
squeue -u $USER
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction_<jobid>.out
```

## What Gets Downloaded

| Item | Size | Location |
|------|------|----------|
| Training data | ~2 GB | `data/citation_prediction/train.parquet` |
| Validation data | ~100 MB | `data/citation_prediction/validation.parquet` |
| FAISS index | ~60 GB | `data/citation_prediction/e5_Flat.index` |
| Wikipedia corpus | ~70 GB | `data/citation_prediction/wiki-18.jsonl` |

**Total: ~135 GB**

## Directory Structure After Setup

```
/scratch/gpfs/ZHUANGL/hk4638/
├── data/citation_prediction/  # All Citation Prediction data
├── conda/envs/retriever/      # Retriever conda environment
├── uv_cache/                  # UV package cache
└── huggingface/               # HuggingFace model cache
```

## Files

| Script | Purpose |
|--------|---------|
| `setup_scratch.sh` | Downloads dataset and FAISS index |
| `setup_retriever.sh` | Creates conda env with faiss-gpu |
| `start_retriever.sh` | Launches the FAISS retrieval server |
