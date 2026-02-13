#!/bin/bash
# ============================================================================
# Smoke Test - Verify full pipeline works before 24-hour run
#
# Usage:
#   sbatch smoke_test.slurm
#   PROMPT_STYLE=extended sbatch smoke_test.slurm
#
# 3 nodes × 4 GPUs = 12 GPUs:
#   Node 0: Retriever (4 GPUs for FAISS sharding)
#   Node 1: Training head (4 GPUs)
#   Node 2: Training worker (4 GPUs)
# ============================================================================

#SBATCH --job-name=smoke-test
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/smoke-test_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/smoke-test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-test

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction"
CKPT_DIR="${SCRATCH_BASE}/checkpoints"
LOG_DIR="${SCRATCH_BASE}/logs"
PROMPT_STYLE=${PROMPT_STYLE:-"short"}   # "short" or "extended"

RUN_NAME="smoke-test-${PROMPT_STYLE}-$(date +%Y%m%d-%H%M%S)"

mkdir -p "${CKPT_DIR}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== SMOKE TEST: Verifying full pipeline ==="
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
MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

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
ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/citation_prediction/harish_setup/.env"
if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment from ${ENV_FILE}"
    set -a && source "${ENV_FILE}" && set +a
else
    echo "WARNING: .env file not found at ${ENV_FILE}"
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
# RETRIEVER CONFIGURATION
# ============================================================================
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_4b"}

if [ "$EMBEDDING_MODEL" = "qwen3_4b" ]; then
    RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
    INDEX_FILE="${DATA_DIR}/qwen3_4b_embed/qwen3_Flat.index"
    CORPUS_FILE="${DATA_DIR}/arxiv_wikiformat_with_ids.jsonl"
    RETRIEVER_NAME=qwen3
elif [ "$EMBEDDING_MODEL" = "qwen3_06b" ]; then
    RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    INDEX_FILE="${DATA_DIR}/qwen3_06b_embed/qwen3_Flat.index"
    CORPUS_FILE="${DATA_DIR}/arxiv_wikiformat_with_ids.jsonl"
    RETRIEVER_NAME=qwen3
fi

# ============================================================================
# START RETRIEVER
# ============================================================================
echo "=== Starting retriever on ${RETRIEVER_NODE} ==="
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Index: ${INDEX_FILE}"
echo "Corpus: ${CORPUS_FILE}"

srun --nodes=1 --ntasks=1 --nodelist=${RETRIEVER_NODE} \
    bash -c "
        source \$(conda info --base)/etc/profile.d/conda.sh
        export CONDA_ENVS_PATH=${SCRATCH_BASE}/conda/envs
        conda activate retriever
        export HF_HOME=${SCRATCH_BASE}/huggingface
        export TRANSFORMERS_CACHE=${SCRATCH_BASE}/huggingface/transformers
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

        python /home/hk4638/SkyRL/skyrl-train/examples/citation_prediction/retriever/retrieval_server.py \
            --index_path ${INDEX_FILE} \
            --corpus_path ${CORPUS_FILE} \
            --topk 3 \
            --retriever_name ${RETRIEVER_NAME} \
            --retriever_model ${RETRIEVER_MODEL_PATH} \
            --faiss_gpu
    " > "${LOG_DIR}/retriever_${SLURM_JOB_ID}.log" 2>&1 &

RETRIEVER_PID=$!

RETRIEVER_URL="http://${RETRIEVER_NODE}:8000/retrieve"
echo "Waiting for retriever at ${RETRIEVER_URL}..."

MAX_RETRIES=60
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -X POST "${RETRIEVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "topk": 1}' > /dev/null 2>&1; then
        echo "Retriever ready!"
        break
    fi
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: Retriever failed!"
    cat "${LOG_DIR}/retriever_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# START RAY CLUSTER
# ============================================================================
echo "=== Starting Ray cluster ==="

HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 --nodelist=${HEAD_NODE} hostname -i)
RAY_PORT=6379

cd /home/hk4638/SkyRL/skyrl-train

# Start Ray daemons and keep srun alive with "sleep infinity"
# Without keeping srun alive, SLURM kills the Ray daemon when the step completes
# Run in background (&) so we can continue with other setup

echo "Starting Ray head on ${HEAD_NODE} (${HEAD_NODE_IP}:${RAY_PORT})"
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

# Join the Ray cluster from the current node (retriever node) with 0 GPUs
# This allows the training script to run here and connect to the cluster
echo "Joining Ray cluster from current node (${RETRIEVER_NODE}) with 0 GPUs"
uv run --active --frozen ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=0
sleep 5

echo "Ray cluster started on head=${HEAD_NODE_IP}:${RAY_PORT}"

# Verify cluster has all nodes
echo "Verifying Ray cluster status..."
uv run --active --frozen ray status

# ============================================================================
# RUN MINIMAL TRAINING
# ============================================================================
echo "=== Running minimal training test ==="
echo "Prompt style: ${PROMPT_STYLE}"

# Training runs on the current node (retriever node) which has joined the Ray cluster
# RAY_ADDRESS tells ray.init() to connect to the existing cluster
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
echo "Running training with RAY_ADDRESS=${RAY_ADDRESS}"

# Minimal config: small batch, 1 epoch
uv run --active --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/${PROMPT_STYLE}/train.parquet']" \
    data.val_data="['${DATA_DIR}/${PROMPT_STYLE}/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=0.5 \
    trainer.policy.optimizer_config.num_warmup_steps=1 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.policy.model.path="${MODEL_PATH}" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.use_liger_kernel=true \
    trainer.policy.sequence_parallel_size=2 \
    trainer.ref.use_liger_kernel=true \
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
    trainer.train_batch_size=16 \
    trainer.policy_mini_batch_size=16 \
    trainer.micro_forward_batch_size_per_gpu=2 \
    trainer.micro_train_batch_size_per_gpu=2 \
    trainer.max_prompt_length=2048 \
    generator.max_input_length=8192 \
    generator.sampling_params.max_generate_length=500 \
    generator.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=2 \
    generator.max_turns=2 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>"]' \
    environment.env_class="citation_prediction" \
    environment.skyrl_gym.max_env_workers=4 \
    environment.skyrl_gym.citation_prediction.log_requests=false \
    environment.skyrl_gym.citation_prediction.search_url="${RETRIEVER_URL}" \
    environment.skyrl_gym.citation_prediction.topk=3 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl-citation-prediction" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=1000 \
    trainer.max_ckpts_to_keep=1 \
    trainer.resume_mode=none \
    trainer.ckpt_path="${CKPT_DIR}/${RUN_NAME}" \
    trainer.eval_batch_size=16 \
    trainer.eval_before_train=false \
    trainer.eval_interval=1000

TRAINING_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
# Stop Ray on local node
uv run --active --frozen ray stop || true
# Kill the background srun processes (this also stops Ray on those nodes)
kill ${RAY_HEAD_PID} 2>/dev/null || true
kill ${RAY_WORKER_PID} 2>/dev/null || true
kill ${RETRIEVER_PID} 2>/dev/null || true
# Wait for background processes to terminate
wait ${RAY_HEAD_PID} 2>/dev/null || true
wait ${RAY_WORKER_PID} 2>/dev/null || true

echo "=== SMOKE TEST COMPLETE ==="
echo "Exit code: ${TRAINING_EXIT_CODE}"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Ready for full training run."
else
    echo "FAILED! Check logs before running full job."
fi

exit ${TRAINING_EXIT_CODE}
