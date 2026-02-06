#!/bin/bash
# ============================================================================
# Search-R1 Training Script
#
# 3 nodes × 4 GPUs = 12 GPUs:
#   Node 0: Retriever (4 GPUs for FAISS sharding)
#   Node 1: Training head (4 GPUs)
#   Node 2: Training worker (4 GPUs)
# ============================================================================

#SBATCH --job-name=search-r1
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/search-r1_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/search-r1_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-short

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/searchR1"
CKPT_DIR="${SCRATCH_BASE}/checkpoints"
LOG_DIR="${SCRATCH_BASE}/logs"

RUN_NAME="search-r1-$(date +%Y%m%d-%H%M%S)"

mkdir -p "${CKPT_DIR}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== Setting up environment ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"

export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda/pkgs"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"
export UV_CACHE_DIR="${SCRATCH_BASE}/uv_cache"
export UV_LINK_MODE=copy
export HF_HOME="${SCRATCH_BASE}/huggingface"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_BASE}/huggingface/hub"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/huggingface/transformers"
# Force offline mode - skip HTTP requests to check for model updates
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"

# Set VIRTUAL_ENV explicitly so --active flag works
export VIRTUAL_ENV="/home/hk4638/SkyRL/skyrl-train/.venv"

# Local path to cached model (avoids HuggingFace network requests)
MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

# Load proxy for WandB access (api.wandb.ai is on the approved list)
# Note: PyPI is NOT on the list, so all packages must be pre-installed
module load proxy/default

# CRITICAL: Disable Ray's automatic uv runtime_env detection
# Without this, Ray detects uv run, packages the working_dir, and tries to build
# skyrl-train on workers - which fails because workers can't reach PyPI
# See: https://github.com/ray-project/ray/issues/59639
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Also unset the hook to prevent automatic runtime_env creation
unset RAY_RUNTIME_ENV_HOOK
# NOTE: Do NOT set SKYRL_DISABLE_RAY_RUNTIME_ENV=1 - we need runtime_env to pass
# HF_HOME and other env vars to Ray workers

# Load secrets (hardcoded path - BASH_SOURCE doesn't work reliably in SLURM)
ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env"
if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment from ${ENV_FILE}"
    set -a && source "${ENV_FILE}" && set +a
else
    echo "WARNING: .env file not found at ${ENV_FILE}"
    echo "WandB logging may not work without WANDB_API_KEY"
fi

# Get node list
NODELIST=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
RETRIEVER_NODE=${NODELIST[0]}
HEAD_NODE=${NODELIST[1]}
WORKER_NODE=${NODELIST[2]}

echo "Retriever node: ${RETRIEVER_NODE}"
echo "Training head: ${HEAD_NODE}"
echo "Training worker: ${WORKER_NODE}"

# Bypass proxy for internal cluster traffic (must use exact hostnames, not wildcards)
export NO_PROXY="localhost,127.0.0.1,${RETRIEVER_NODE},${HEAD_NODE},${WORKER_NODE}"
export no_proxy="$NO_PROXY"
echo "NO_PROXY set to: $NO_PROXY"

# ============================================================================
# START RETRIEVER SERVER
# ============================================================================
echo "=== Starting retriever server on ${RETRIEVER_NODE} ==="

srun --nodes=1 --ntasks=1 --nodelist=${RETRIEVER_NODE} \
    bash -c "
        source \$(conda info --base)/etc/profile.d/conda.sh
        export CONDA_ENVS_PATH=${SCRATCH_BASE}/conda/envs
        conda activate retriever
        export HF_HOME=${SCRATCH_BASE}/huggingface
        export TRANSFORMERS_CACHE=${SCRATCH_BASE}/huggingface/transformers
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

        echo 'Starting retriever on \$(hostname) with 4 GPUs for FAISS sharding...'
        python /home/hk4638/SkyRL/skyrl-train/examples/search/retriever/retrieval_server.py \
            --index_path ${DATA_DIR}/e5_Flat.index \
            --corpus_path ${DATA_DIR}/wiki-18.jsonl \
            --topk 3 \
            --retriever_name e5 \
            --retriever_model intfloat/e5-base-v2 \
            --faiss_gpu
    " > "${LOG_DIR}/retriever_${SLURM_JOB_ID}.log" 2>&1 &

RETRIEVER_PID=$!
echo "Retriever PID: ${RETRIEVER_PID}"

# Wait for retriever to be ready
RETRIEVER_URL="http://${RETRIEVER_NODE}:8000/retrieve"
echo "Waiting for retriever at ${RETRIEVER_URL}..."

MAX_RETRIES=60
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -X POST "${RETRIEVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "topk": 1}' > /dev/null 2>&1; then
        echo "Retriever is ready!"
        break
    fi
    echo "Waiting for retriever... (attempt $((RETRY_COUNT+1))/${MAX_RETRIES})"
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: Retriever failed to start!"
    echo "Check log: ${LOG_DIR}/retriever_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# START RAY CLUSTER
# ============================================================================
echo "=== Starting Ray cluster ==="

HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 --nodelist=${HEAD_NODE} hostname -i)
RAY_PORT=6379

echo "Starting Ray head on ${HEAD_NODE} (${HEAD_NODE_IP}:${RAY_PORT})"

cd /home/hk4638/SkyRL/skyrl-train

# Start Ray head - use sleep infinity to keep srun alive (not --block which holds resources)
srun --nodes=1 --ntasks=1 --nodelist=${HEAD_NODE} \
    bash -c "cd /home/hk4638/SkyRL/skyrl-train && \
        export UV_CACHE_DIR=${SCRATCH_BASE}/uv_cache && \
        export UV_LINK_MODE=copy && \
        export VIRTUAL_ENV=/home/hk4638/SkyRL/skyrl-train/.venv && \
        export PATH=\$HOME/.local/bin:\$PATH && \
        export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 && \
        uv run --active --frozen ray start --head --port=${RAY_PORT} --num-gpus=4 && \
        sleep infinity" &
RAY_HEAD_PID=$!
sleep 15

echo "Starting Ray worker on ${WORKER_NODE}"
srun --nodes=1 --ntasks=1 --nodelist=${WORKER_NODE} \
    bash -c "cd /home/hk4638/SkyRL/skyrl-train && \
        export UV_CACHE_DIR=${SCRATCH_BASE}/uv_cache && \
        export UV_LINK_MODE=copy && \
        export VIRTUAL_ENV=/home/hk4638/SkyRL/skyrl-train/.venv && \
        export PATH=\$HOME/.local/bin:\$PATH && \
        export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 && \
        uv run --active --frozen ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=4 && \
        sleep infinity" &
RAY_WORKER_PID=$!
sleep 15

# Join Ray cluster from retriever node with 0 GPUs (retriever uses the GPUs)
echo "Joining Ray cluster from ${RETRIEVER_NODE} with 0 GPUs"
uv run --active --frozen ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=0
sleep 5

echo "Ray cluster started on head=${HEAD_NODE_IP}:${RAY_PORT}"

# Verify cluster status
echo "Verifying Ray cluster status..."
uv run --active --frozen ray status

# ============================================================================
# RUN TRAINING
# ============================================================================
echo "=== Starting training ==="
echo "Run name: ${RUN_NAME}"
echo "Data dir: ${DATA_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}/${RUN_NAME}"
echo "Retriever URL: ${RETRIEVER_URL}"

# CRITICAL: Set RAY_ADDRESS so training connects to existing cluster instead of starting local
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
echo "Connecting to Ray cluster at RAY_ADDRESS=${RAY_ADDRESS}"

# Run training with 8 GPUs (2 nodes × 4 GPUs)
# NOTE: Use --active to tell uv to use the existing VIRTUAL_ENV instead of creating new venv
uv run --active --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=0.5 \
    trainer.policy.optimizer_config.num_warmup_steps=94 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.use_tis=true \
    trainer.algorithm.tis_imp_ratio_cap=2.0 \
    trainer.policy.model.path="${MODEL_PATH}" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.policy_num_nodes=2 \
    trainer.placement.ref_num_gpus_per_node=4 \
    trainer.placement.ref_num_nodes=2 \
    generator.num_inference_engines=4 \
    generator.inference_engine_tensor_parallel_size=2 \
    generator.backend=vllm \
    generator.run_engines_locally=true \
    generator.weight_sync_backend=nccl \
    generator.gpu_memory_utilization=0.5 \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=512 \
    trainer.policy_mini_batch_size=256 \
    trainer.micro_forward_batch_size_per_gpu=4 \
    trainer.micro_train_batch_size_per_gpu=4 \
    trainer.max_prompt_length=2048 \
    generator.max_input_length=4096 \
    generator.sampling_params.max_generate_length=500 \
    generator.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=5 \
    generator.max_turns=4 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>", "</answer>"]' \
    environment.env_class="search" \
    environment.skyrl_gym.max_env_workers=16 \
    environment.skyrl_gym.search.log_requests=false \
    environment.skyrl_gym.search.search_url="${RETRIEVER_URL}" \
    environment.skyrl_gym.search.topk=3 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl-search" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=20 \
    trainer.hf_save_interval=100 \
    trainer.max_ckpts_to_keep=5 \
    trainer.resume_mode=latest \
    trainer.ckpt_path="${CKPT_DIR}/${RUN_NAME}" \
    trainer.eval_batch_size=256 \
    trainer.eval_before_train=false \
    generator.eval_sampling_params.temperature=0 \
    generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
    trainer.export_path="${CKPT_DIR}/${RUN_NAME}/exports" \
    trainer.eval_interval=50

TRAINING_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="

# Stop Ray on local node
uv run --active --frozen ray stop || true

# Kill background srun processes (this also stops Ray on those nodes)
kill ${RAY_HEAD_PID} 2>/dev/null || true
kill ${RAY_WORKER_PID} 2>/dev/null || true
kill ${RETRIEVER_PID} 2>/dev/null || true

# Wait for background processes to terminate
wait ${RAY_HEAD_PID} 2>/dev/null || true
wait ${RAY_WORKER_PID} 2>/dev/null || true

echo "=== Job complete ==="
echo "End time: $(date)"
echo "Training exit code: ${TRAINING_EXIT_CODE}"

exit ${TRAINING_EXIT_CODE}
