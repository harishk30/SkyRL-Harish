#!/bin/bash
# ============================================================================
# Single-node 2-GPU Smoke Test - Citation Prediction V3
#
# Minimal test to verify the pipeline works on a single node.
# Retriever runs on CPU (no FAISS GPU sharding).
# Useful for quick iteration when 3-node jobs are slow to schedule.
#
# 1 node, 2 GPUs:
#   - Retriever on CPU (background process)
#   - Ray head + training on 2 GPUs
#
# Usage:
#   sbatch smoke_test_single_node_2gpu.slurm
# ============================================================================

#SBATCH --job-name=cit-v3-smoke-1n
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/cit-v3-smoke-1n_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/cit-v3-smoke-1n_%j.err
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
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction_v3/full"
CKPT_DIR="${SCRATCH_BASE}/checkpoints"
LOG_DIR="${SCRATCH_BASE}/logs"

RUN_NAME="cit-v3-smoke-1n-$(date +%Y%m%d-%H%M%S)"

mkdir -p "${CKPT_DIR}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== SMOKE TEST (1 node, 2 GPUs): Citation Prediction V3 ==="
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"

export VIRTUAL_ENV="/home/hk4638/SkyRL/.venv"

# Local path to cached model
MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

module load proxy/default

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK

echo "VIRTUAL_ENV=${VIRTUAL_ENV}"
echo "RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV}"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"

# Load secrets
ENV_FILE="/home/hk4638/SkyRL/examples/search/harish_setup/.env"
if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment from ${ENV_FILE}"
    set -a && source "${ENV_FILE}" && set +a
else
    echo "WARNING: .env file not found at ${ENV_FILE}"
fi

# Get node
NODELIST=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
RETRIEVER_NODE=${NODELIST[0]}

echo "Node: ${RETRIEVER_NODE}"

# ============================================================================
# RETRIEVER CONFIGURATION
# ============================================================================
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_4b"}
CORPUS_FILE="${SCRATCH_BASE}/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl"

if [ "$EMBEDDING_MODEL" = "qwen3_4b" ]; then
    RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
    INDEX_FILE="${SCRATCH_BASE}/data/citation_prediction/qwen3_4b_embed/qwen3_Flat.index"
    RETRIEVER_NAME=qwen3
elif [ "$EMBEDDING_MODEL" = "qwen3_06b" ]; then
    RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    INDEX_FILE="/home/hk4638/scratch/shared/data/qwen3_06_embed/qwen3_Flat.index"
    RETRIEVER_NAME=qwen3
fi

# ============================================================================
# START RETRIEVER (CPU - no FAISS GPU sharding)
# ============================================================================
echo "=== Starting retriever (CPU) ==="

(
    source /usr/licensed/anaconda3/2024.10/etc/profile.d/conda.sh
    export CONDA_ENVS_PATH=${SCRATCH_BASE}/conda/envs
    conda activate retriever
    export HF_HOME=${SCRATCH_BASE}/huggingface
    export TRANSFORMERS_CACHE=${SCRATCH_BASE}/huggingface/transformers
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    python /home/hk4638/SkyRL/examples/citation_prediction/retriever/retrieval_server.py \
        --index_path ${INDEX_FILE} \
        --corpus_path ${CORPUS_FILE} \
        --topk 3 \
        --retriever_name ${RETRIEVER_NAME} \
        --retriever_model ${RETRIEVER_MODEL_PATH}
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

cd /home/hk4638/SkyRL

uv run --active --frozen ray start --head --port=${RAY_PORT} --num-gpus=2 --block &
sleep 10

echo "Ray head started at ${HEAD_NODE_IP}:${RAY_PORT}"

# ============================================================================
# RUN MINIMAL TRAINING
# ============================================================================
echo "=== Running minimal training test ==="

uv run --active --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
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
    trainer.ref.use_liger_kernel=true \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.placement.policy_num_gpus_per_node=2 \
    trainer.placement.policy_num_nodes=1 \
    trainer.placement.ref_num_gpus_per_node=2 \
    trainer.placement.ref_num_nodes=1 \
    generator.inference_engine.num_engines=2 \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.4 \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=2 \
    trainer.policy_mini_batch_size=2 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=4096 \
    generator.max_input_length=32768 \
    generator.sampling_params.max_generate_length=512 \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=1 \
    generator.max_turns=6 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>", "</done>"]' \
    environment.env_class="citation_prediction_v2" \
    environment.skyrl_gym.max_env_workers=2 \
    environment.skyrl_gym.citation_prediction_v2.log_requests=false \
    environment.skyrl_gym.citation_prediction_v2.search_url="${RETRIEVER_URL}" \
    environment.skyrl_gym.citation_prediction_v2.topk=5 \
    environment.skyrl_gym.citation_prediction_v2.max_predictions_ratio=2.0 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl-citation-prediction-v3" \
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
uv run --active --frozen ray stop || true
kill ${RETRIEVER_PID} 2>/dev/null || true

echo "=== SMOKE TEST COMPLETE ==="
echo "Exit code: ${TRAINING_EXIT_CODE}"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Ready for larger runs."
else
    echo "FAILED! Check logs before running larger jobs."
fi

exit ${TRAINING_EXIT_CODE}
