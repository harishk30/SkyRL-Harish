# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the `examples/search/harish_setup` directory for training **Search-R1** on Princeton's Della cluster. Search-R1 is a multi-turn search-augmented reasoning setup that trains LLMs to learn when/how to search and synthesize results using GRPO.

## Current Setup Status

**Location**: All data and caches are on scratch filesystem at `/scratch/gpfs/ZHUANGL/hk4638`

| Component | Status | Location |
|-----------|--------|----------|
| Training data | ✅ Ready | `/scratch/gpfs/ZHUANGL/hk4638/data/searchR1/train.parquet` (339MB) |
| Validation data | ✅ Ready | `/scratch/gpfs/ZHUANGL/hk4638/data/searchR1/validation.parquet` (69MB) |
| FAISS index | ✅ Ready | `/scratch/gpfs/ZHUANGL/hk4638/data/searchR1/e5_Flat.index` (61GB) |
| Wikipedia corpus | ✅ Ready | `/scratch/gpfs/ZHUANGL/hk4638/data/searchR1/wiki-18.jsonl` (14GB) |
| Retriever conda env | ✅ Ready | `/scratch/gpfs/ZHUANGL/hk4638/conda/envs/retriever` |
| Training venv | ✅ Ready | `/scratch/gpfs/ZHUANGL/hk4638/venvs/skyrl-train` (symlinked from `.venv`) |
| Qwen model | ✅ Cached | `/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/` |
| e5 encoder | ✅ Cached | `/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/` |
| uv | ✅ Installed | `~/.local/bin/uv` |
| WandB API key | ✅ Configured | `.env` file |

## Files in harish_setup/

| File | Purpose |
|------|---------|
| `train_search.slurm` | **Main SLURM script** - 24-hour training (3 nodes × 4 GPUs) |
| `smoke_test.slurm` | **Quick test** - 1-hour validation of full pipeline |
| `test_retriever.slurm` | Test retriever only (4 GPUs for FAISS sharding) |
| `.env` | WandB API key (sourced by SLURM script) |
| `.gitignore` | Prevents .env from being committed |
| `CLAUDE.md` | This file |

## Critical: Della Cluster Constraints

### 1. Compute Nodes Have No General Internet Access

Compute nodes can only reach a **pre-approved list of APIs** via proxy. This is the most common source of failures.

**Approved services** (via `module load proxy/default`):
- `api.wandb.ai` ✅ (WandB logging works)
- `download.pytorch.org` ✅
- `api.openai.com`, `api.anthropic.com` ✅

**NOT approved**:
- `pypi.org` ❌ (package downloads fail)
- `huggingface.co` ❌ (model downloads fail)
- General internet ❌

### 2. Pre-download Everything from Login Node

```bash
# Run these commands from the LOGIN NODE before submitting jobs

cd /home/hk4638/SkyRL/skyrl-train
export HF_HOME=/scratch/gpfs/ZHUANGL/hk4638/huggingface
export UV_CACHE_DIR=/scratch/gpfs/ZHUANGL/hk4638/uv_cache
export UV_LINK_MODE=copy

# Install ALL dependencies including vllm extras
uv sync --frozen --extra vllm

# Cache the training model
uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
"

# Cache the retriever encoder (use retriever conda env)
conda activate retriever
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('intfloat/e5-base-v2')
AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
"
```

### 3. Proxy Module for WandB

Add this to SLURM scripts for WandB access:
```bash
module load proxy/default
```
**Important**: Only load this on compute nodes, not login nodes.

### 4. Home Directory Quota (~50GB)

The `.venv` is too large for home. It's symlinked to scratch:
```
/home/hk4638/SkyRL/skyrl-train/.venv -> /scratch/gpfs/ZHUANGL/hk4638/venvs/skyrl-train
```

### 5. Ray Must Run via uv (with --active flag)

Ray is installed in the uv-managed venv, not system-wide. Always use:
```bash
export VIRTUAL_ENV="/home/hk4638/SkyRL/skyrl-train/.venv"
uv run --active --frozen ray start ...
uv run --active --frozen ray stop
```

**Critical**: The `--active` flag tells uv to use the existing `VIRTUAL_ENV` instead of creating a new virtual environment. Without this flag, uv may try to create a temporary environment, which causes Ray to package the working directory and try to build packages on workers.

**Never** call `ray` directly - it won't be found on compute nodes.

### 6. HuggingFace Offline Mode and Local Model Paths

Compute nodes cannot reach huggingface.co. Even with models cached, HF libraries may try to fetch metadata or check for updates. **Force offline mode** and **use local paths**:

```bash
# Force offline mode - prevents any HuggingFace HTTP requests
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Use local cached model path instead of HuggingFace model name
MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

# In training command, use the local path:
trainer.policy.model.path="${MODEL_PATH}"
```

**How to find cached model path**:
```bash
ls /scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--<org>--<model>/snapshots/
# Use the hash directory path (not the model name)
```

See [HuggingFace Offline Mode docs](https://huggingface.co/docs/transformers/installation#offline-mode) for details.

### 7. Disable Ray's UV Runtime Env Auto-Detection

Ray 2.51+ automatically detects when you run scripts via `uv run` and:
1. Sets `py_executable` to `uv run` for workers
2. Packages `working_dir` (including pyproject.toml) to workers
3. Workers try to build the package, requiring PyPI access

**The fix** is to disable this auto-detection with:
```bash
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK
```

**Important**: Do NOT set `SKYRL_DISABLE_RAY_RUNTIME_ENV=1` - we need runtime_env to pass `HF_HOME` and other environment variables to Ray workers.

### 8. Multi-Node Ray: Set RAY_ADDRESS

When running a multi-node Ray cluster (started via srun on head/worker nodes), the training command must know where the cluster is. Otherwise it starts a **new local Ray instance** with only the local node's GPUs.

**The fix**: Export `RAY_ADDRESS` before running the training command:
```bash
# After starting Ray head and worker via srun
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 --nodelist=${HEAD_NODE} hostname -i)
RAY_PORT=6379

# Start Ray head on HEAD_NODE via srun...
# Start Ray worker on WORKER_NODE via srun...

# CRITICAL: Set RAY_ADDRESS before training
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
uv run --active --frozen --extra vllm -m skyrl_train.entrypoints.main_base ...
```

**Symptom**: If you see `Started a local Ray instance` in logs followed by `Failed to create placement group with 8 bundles`, the training started a local Ray instead of connecting to the cluster.

See [GitHub issue #59639](https://github.com/ray-project/ray/issues/59639) for details.

### 9. BASH_SOURCE Doesn't Work in SLURM

SLURM copies scripts to `/var/spool/slurmd/`, breaking `BASH_SOURCE[0]` path resolution. **Hardcode paths** for .env files:
```bash
# BAD - doesn't work in SLURM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GOOD - hardcode the path
ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env"
```

### 10. FAISS Index Requires Multi-GPU Sharding

The 61GB FAISS index is too large for a single GPU (even 80GB A100). The retriever uses `faiss.index_cpu_to_all_gpus()` with `co.shard = True` to distribute across GPUs.

- 61GB index → ~30GB in FP16 → ~8GB per GPU with 4 GPUs
- **Do NOT set CUDA_VISIBLE_DEVICES** - let FAISS see all GPUs
- Retriever needs **4 GPUs** for sharding

### 11. No `trainer.max_steps` Config Option

The trainer uses `epochs` to control training duration, not `max_steps`. Don't try to set `trainer.max_steps` - it doesn't exist and will cause a Hydra error.

### 12. Do NOT Use `--isolated` with uv run

```bash
# BAD - --isolated creates a temp env, Ray packages working dir, workers fail to build
uv run --isolated --frozen --extra vllm -m skyrl_train.entrypoints.main_base ...

# GOOD - use the existing venv
uv run --frozen --extra vllm -m skyrl_train.entrypoints.main_base ...
```

The `--isolated` flag creates a temporary environment. Ray then detects the working directory and packages it for workers. When workers try to build the package, they need `setuptools` and `wheel` from PyPI, which fails without internet access.

### 13. Proxy Bypass for Internal Cluster Traffic (NO_PROXY)

**CRITICAL**: The proxy module intercepts ALL HTTP traffic, including internal cluster requests to the retriever. This causes `403 Forbidden` errors when the training tries to call the search API.

**The fix**: Set `NO_PROXY` to bypass the proxy for internal hostnames. Must use **exact hostnames** (wildcards like `della-l*` don't work with Python's requests library):

```bash
# After determining node names from SLURM
NODELIST=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
RETRIEVER_NODE=${NODELIST[0]}
HEAD_NODE=${NODELIST[1]}
WORKER_NODE=${NODELIST[2]}

# Bypass proxy for internal cluster traffic
export NO_PROXY="localhost,127.0.0.1,${RETRIEVER_NODE},${HEAD_NODE},${WORKER_NODE}"
export no_proxy="$NO_PROXY"
```

**Symptom**: If you see `403 Client Error: Forbidden for url: http://della-XXXX:8000/retrieve` in logs, the proxy is intercepting internal traffic.

### 14. Valid ResumeMode Values

The `trainer.resume_mode` config only accepts these values:
- `none` - Start fresh, don't resume from checkpoint
- `latest` - Resume from the latest checkpoint in `ckpt_path`
- `from_path` - Resume from a specific checkpoint path

**Common mistake**: Using `disabled` which is NOT valid and causes:
```
ValueError: 'disabled' is not a valid ResumeMode
```

### 15. Retriever Node Must Join Ray Cluster

The training script runs on the retriever node (Node 0), but it needs to be part of the Ray cluster to submit tasks. Join with 0 GPUs since the retriever uses all GPUs on that node:

```bash
# After starting Ray head and worker on other nodes
uv run --active --frozen ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=0
```

## How to Run Training

### 1. Pre-flight Check (from login node)
```bash
cd /home/hk4638/SkyRL/skyrl-train
export UV_CACHE_DIR=/scratch/gpfs/ZHUANGL/hk4638/uv_cache
export UV_LINK_MODE=copy

# Ensure all deps are installed (including vllm)
uv sync --frozen --extra vllm
```

### 2. Run Smoke Test First
```bash
sbatch examples/search/harish_setup/smoke_test.slurm
```
This runs a quick 1-hour test with minimal batch size to verify:
- Retriever starts with 4-GPU FAISS sharding
- Ray cluster forms across 2 training nodes
- vLLM inference engines initialize
- Training loop runs

### 3. Submit Full Training
```bash
sbatch examples/search/harish_setup/train_search.slurm
```

### Monitor Job
```bash
# Check job status
squeue -u $USER

# Watch training logs
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/search-r1_<jobid>.out

# Watch retriever logs
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/retriever_<jobid>.log
```

### Cancel Job
```bash
scancel <jobid>
```

## SLURM Configuration

### Resource Allocation (Simple 3-Node Job)
```
┌─────────────────────────────────────────────────────────────────────┐
│ 3 nodes × 4 GPUs = 12 GPUs                                          │
├─────────────────────────────────────────────────────────────────────┤
│ Node 0: Retriever (4 GPUs for FAISS sharding)                      │
│ Node 1: Training head (4 GPUs)                                     │
│ Node 2: Training worker (4 GPUs)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Time: 24 hours (full) / 1 hour (smoke test)                        │
│ Partition: gpu-short (full) / gpu-test (smoke test)                │
└─────────────────────────────────────────────────────────────────────┘
```

**Note**: We use a simple 3-node job (not hetjob) since all nodes need the same resources (4 GPUs each).

### Execution Flow
```
1. Setup environment variables
         ↓
2. Load proxy module (for WandB access)
         ↓
3. Get node list, set NO_PROXY for internal traffic
         ↓
4. Source .env file (hardcoded path)
         ↓
5. Start retriever on Node 0 (4 GPUs for FAISS sharding)
         ↓
6. Wait for retriever health check (up to 10 min)
         ↓
7. Start Ray head on Node 1 (via uv run, with sleep infinity)
         ↓
8. Start Ray worker on Node 2 (via uv run, with sleep infinity)
         ↓
9. Join Ray cluster from Node 0 with 0 GPUs
         ↓
10. Set RAY_ADDRESS, run training with uv run --extra vllm
         ↓
11. Cleanup (ray stop, kill background processes)
```

## Training Configuration

### Key Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `policy_num_gpus_per_node` | 4 | Split across 2 nodes |
| `policy_num_nodes` | 2 | Total 8 GPUs for training |
| `num_inference_engines` | 4 | vLLM engines |
| `inference_engine_tensor_parallel_size` | 2 | 2 GPUs per engine |
| `train_batch_size` | 512 | Prompts per step |
| `n_samples_per_prompt` | 5 | Rollouts for GRPO |
| `max_turns` | 4 | Multi-turn search |
| `max_generate_length` | 500 | Tokens per turn |

### Algorithm
- **GRPO** with advantage estimation
- **TIS** (Truncated Importance Sampling) with cap=2.0
- **KL loss** coefficient=0.001
- Learning rate: 1e-6

## Troubleshooting

### "No such file or directory: ray"
Ray must be run via `uv run`:
```bash
uv run --frozen ray start --head ...
```

### "Name or service not known" / DNS errors
Compute nodes can't reach the internet. Either:
- Pre-download from login node (for packages/models)
- Use `module load proxy/default` (only works for approved APIs like WandB)

### "403 Forbidden" errors from retriever / search API
The error looks like:
```
API Request Error: 403 Client Error: Forbidden for url: http://della-XXXX:8000/retrieve
```

**Root cause**: The proxy module intercepts internal HTTP traffic between cluster nodes. The proxy doesn't recognize internal hostnames and returns 403.

**Fix**: Set `NO_PROXY` with exact hostnames (not wildcards) AFTER determining node names:
```bash
export NO_PROXY="localhost,127.0.0.1,${RETRIEVER_NODE},${HEAD_NODE},${WORKER_NODE}"
export no_proxy="$NO_PROXY"
```

**Important**: Python's `requests` library doesn't support wildcards like `della-l*`. Use exact hostnames.

### "'disabled' is not a valid ResumeMode"
The error looks like:
```
ValueError: 'disabled' is not a valid ResumeMode
```

**Fix**: Use valid values: `none`, `latest`, or `from_path` (not `disabled`).

### HuggingFace ReadTimeoutError or LocalEntryNotFoundError
The error looks like:
```
ReadTimeoutError... huggingface.co... Read timed out
LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled
```

**Root cause**: HuggingFace libraries try to reach huggingface.co even with cached models.

**Fix**: Use offline mode and local paths:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/<hash>"
# Use MODEL_PATH instead of "Qwen/Qwen2.5-3B-Instruct"
```

### Ray workers failing with PyPI download errors
The error looks like:
```
Failed to resolve requirements from `build-system.requires`
Failed to fetch: `https://pypi.org/simple/setuptools/`
```

**Root cause**: Ray 2.51+ auto-detects `uv run` and packages the working_dir to workers, who then try to build skyrl-train (requiring PyPI).

**Fix**: Disable the auto-detection:
```bash
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK
```

**Important**: Do NOT set `SKYRL_DISABLE_RAY_RUNTIME_ENV=1` - we need runtime_env to pass `HF_HOME` to workers.

Also avoid `--isolated` with uv run.

### "Failed to create placement group with 8 bundles" / "Started a local Ray instance"
The error looks like:
```
Started a local Ray instance.
...
RuntimeError: Failed to create placement group with 8 bundles (requiring 8.0 GPUs...)
```

**Root cause**: Training started its own local Ray instance instead of connecting to the multi-node cluster started via srun. This happens when `RAY_ADDRESS` is not set.

**Fix**: Export `RAY_ADDRESS` before running the training command:
```bash
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
uv run --active --frozen --extra vllm -m skyrl_train.entrypoints.main_base ...
```

### "Can't find node_ip_address.json" when connecting to Ray cluster
The error looks like:
```
ValueError: Can't find a `node_ip_address.json` file from /tmp/ray/session_... for 60 seconds.
A ray instance hasn't started. Did you do `ray start` or `ray.init` on this host?
```

**Root cause**: The training command is running on a node that's NOT part of the Ray cluster. When `ray.init()` tries to connect to a remote cluster (via `RAY_ADDRESS`), the local node must also be part of that cluster. In SLURM multi-node jobs, the main script runs on the first node by default.

**Example**: If Ray head is on Node 1 and Ray worker is on Node 2, but the training command runs on Node 0 (retriever node), the training will fail because Node 0 is not a Ray cluster member.

**Fix**: Have the node running the training script join the Ray cluster (with 0 GPUs if GPUs are used by other processes like the retriever):
```bash
# Start Ray as daemons (no --block) so srun returns immediately
srun --nodelist=${HEAD_NODE} bash -c "uv run --active --frozen ray start --head --port=6379 --num-gpus=4"
srun --nodelist=${WORKER_NODE} bash -c "uv run --active --frozen ray start --address=${HEAD_NODE_IP}:6379 --num-gpus=4"

# Join from current node with 0 GPUs (retriever uses the GPUs)
uv run --active --frozen ray start --address=${HEAD_NODE_IP}:6379 --num-gpus=0

# Now training can run locally and connect to the cluster
export RAY_ADDRESS="${HEAD_NODE_IP}:6379"
uv run --active --frozen --extra vllm -m skyrl_train.entrypoints.main_base ...
```

### "Requested nodes are busy" / SLURM step creation fails
The error looks like:
```
srun: Job step creation temporarily disabled, retrying (Requested nodes are busy)
srun: error: Unable to create step for job: Job/step already completing or completed
```

**Root cause**: Using `ray start --block` with `srun &` keeps the srun process running and holding all node resources. When you try to run another `srun` on the same node, there are no resources left.

**Fix**: Don't use `--block` with `ray start`. Start Ray as daemons instead:
```bash
# BAD - holds resources, blocks other srun commands on this node
srun --nodelist=${HEAD_NODE} bash -c "ray start --head --block" &

# GOOD - starts daemon and returns immediately
srun --nodelist=${HEAD_NODE} bash -c "ray start --head"
```

### Ray worker registration timeout
You may see:
```
worker_pool.cc:589: Some workers ... have not registered within the timeout
```
This is usually a **symptom** of the runtime env build hanging (PyPI blocked).
Increasing the timeout (e.g., `export RAY_worker_register_timeout_seconds=600`) can delay the error,
but does **not** fix the underlying packaging issue.

### FAISS "cudaMalloc error out of memory"
Request 4 GPUs for retriever and don't set `CUDA_VISIBLE_DEVICES`:
```bash
#SBATCH --gres=gpu:4
# Don't set CUDA_VISIBLE_DEVICES - let FAISS shard across all GPUs
```

### WANDB_API_KEY not found
The .env path resolution fails in SLURM. Hardcode the path:
```bash
ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env"
source "${ENV_FILE}"
```

### "Key 'max_steps' is not in struct"
The `trainer.max_steps` config option doesn't exist. Use `trainer.epochs` instead.

### Disk quota exceeded
Ensure these point to scratch:
- `.venv` symlinked to scratch
- `UV_CACHE_DIR=/scratch/gpfs/ZHUANGL/hk4638/uv_cache`
- `HF_HOME=/scratch/gpfs/ZHUANGL/hk4638/huggingface`

### Job pending with "(Resources)"
Cluster is busy. Use `gpu-test` partition for faster scheduling during testing.

## Architecture

### Training Flow
```
Dataset (parquet) → Generator (vLLM) → SearchEnv → Rewards → GRPO Training
                         ↓
                   Multi-turn loop (max 4 turns):
                   1. Model generates <search>query</search>
                   2. HTTP call to retriever → top-3 documents
                   3. Model continues with <information>...</information>
                   4. Model outputs <answer>...</answer>
                   5. Reward = Exact Match score (0 or 1)
```

### Key Components
| Component | Location | Purpose |
|-----------|----------|---------|
| **Trainer** | `skyrl_train/trainer.py` | Main loop, GRPO, weight sync |
| **Generator** | `skyrl_train/generators/skyrl_gym_generator.py` | Multi-turn rollouts |
| **SearchEnv** | `skyrl-gym/skyrl_gym/envs/search/env.py` | Parses actions, rewards |
| **SearchTool** | `skyrl-gym/skyrl_gym/tools/search.py` | HTTP client for FAISS |
| **Retriever** | `examples/search/retriever/retrieval_server.py` | FAISS + FastAPI |

## Output Locations

| Output | Location |
|--------|----------|
| Job stdout | `/scratch/gpfs/ZHUANGL/hk4638/logs/search-r1_<jobid>.out` |
| Job stderr | `/scratch/gpfs/ZHUANGL/hk4638/logs/search-r1_<jobid>.err` |
| Retriever log | `/scratch/gpfs/ZHUANGL/hk4638/logs/retriever_<jobid>.log` |
| Checkpoints | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/<run_name>/` |
| Model exports | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/<run_name>/exports/` |
| WandB | https://wandb.ai/*/skyrl-search |
