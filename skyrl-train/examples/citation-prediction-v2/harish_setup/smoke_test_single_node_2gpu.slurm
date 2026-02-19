#!/bin/bash
# ============================================================================
# Single-node 2-GPU Smoke Test - Minimal training step
# - Retriever runs on CPU (no FAISS GPU sharding)
# - Ray runs on one node
# ============================================================================

#SBATCH --job-name=smoke-1node-2gpu
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/smoke-1node-2gpu_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/smoke-1node-2gpu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
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

RUN_NAME="smoke-1node-2gpu-${PROMPT_STYLE}-$(date +%Y%m%d-%H%M%S)"

mkdir -p "${CKPT_DIR}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== SMOKE TEST (1 node, 2 GPUs): Minimal training ==="
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

# Load proxy for WandB access (api.wandb.ai is on the approved list)
# Note: PyPI is NOT on the list, so all packages must be pre-installed
module load proxy/default

# CRITICAL: Disable Ray's automatic uv runtime_env detection
# This prevents Ray from packaging working_dir and trying to build on workers
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK
# NOTE: Do NOT set SKYRL_DISABLE_RAY_RUNTIME_ENV=1 - we need runtime_env to pass
# HF_HOME and other env vars to Ray workers

echo "VIRTUAL_ENV=${VIRTUAL_ENV}"
echo "RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV}"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"

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

echo "Node: ${RETRIEVER_NODE}"

# ============================================================================
# START RETRIEVER (CPU)
# ============================================================================
echo "=== Starting retriever (CPU) ==="

# For single-node jobs, run directly without srun to avoid resource contention
(
    source $(conda info --base)/etc/profile.d/conda.sh
    export CONDA_ENVS_PATH=${SCRATCH_BASE}/conda/envs
    conda activate retriever
    export HF_HOME=${SCRATCH_BASE}/huggingface
    export TRANSFORMERS_CACHE=${SCRATCH_BASE}/huggingface/transformers
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    python /home/hk4638/SkyRL/skyrl-train/examples/citation_prediction/retriever/retrieval_server.py \
        --index_path ${DATA_DIR}/e5_Flat.index \
        --corpus_path ${DATA_DIR}/wiki-18.jsonl \
        --topk 3 \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2
) > "${LOG_DIR}/retriever_${SLURM_JOB_ID}.log" 2>&1 &

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
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: Retriever failed!"
    cat "${LOG_DIR}/retriever_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# START RAY (single node)
# ============================================================================
echo "=== Starting Ray head ==="

HEAD_NODE_IP=$(hostname -i)
RAY_PORT=6379

cd /home/hk4638/SkyRL/skyrl-train

# For single-node jobs, run directly without srun
uv run --active --frozen ray start --head --port=${RAY_PORT} --num-gpus=2 --block &
sleep 10

echo "Ray head started at ${HEAD_NODE_IP}:${RAY_PORT}"

# ============================================================================
# RUN MINIMAL TRAINING
# ============================================================================
echo "=== Running minimal training test ==="

# Use --active to tell uv to use the existing VIRTUAL_ENV
uv run --active --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/${PROMPT_STYLE}/train.parquet']" \
    data.val_data="['${DATA_DIR}/${PROMPT_STYLE}/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.max_grad_norm=0.5 \
    trainer.policy.optimizer_config.num_warmup_steps=1 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.policy.model.path="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.placement.policy_num_gpus_per_node=2 \
    trainer.placement.policy_num_nodes=1 \
    trainer.placement.ref_num_gpus_per_node=2 \
    trainer.placement.ref_num_nodes=1 \
    generator.num_inference_engines=2 \
    generator.inference_engine_tensor_parallel_size=1 \
    generator.backend=vllm \
    generator.run_engines_locally=true \
    generator.weight_sync_backend=nccl \
    generator.gpu_memory_utilization=0.4 \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=2 \
    trainer.policy_mini_batch_size=2 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=1024 \
    generator.max_input_length=4096 \
    generator.sampling_params.max_generate_length=128 \
    generator.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=1 \
    generator.max_turns=2 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>", "</answer>"]' \
    environment.env_class="search" \
    environment.skyrl_gym.max_env_workers=2 \
    environment.skyrl_gym.search.log_requests=false \
    environment.skyrl_gym.search.search_url="${RETRIEVER_URL}" \
    environment.skyrl_gym.search.topk=3 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl-citation-prediction" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=1000 \
    trainer.max_ckpts_to_keep=1 \
    trainer.resume_mode=disabled \
    trainer.ckpt_path="${CKPT_DIR}/${RUN_NAME}" \
    trainer.eval_batch_size=2 \
    trainer.eval_before_train=false \
    trainer.eval_interval=1000

TRAINING_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
uv run --active --frozen ray stop
kill ${RETRIEVER_PID} 2>/dev/null || true

echo "=== SMOKE TEST COMPLETE ==="
echo "Exit code: ${TRAINING_EXIT_CODE}"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Ready for larger runs."
else
    echo "FAILED! Check logs before running larger jobs."
fi

exit ${TRAINING_EXIT_CODE}
