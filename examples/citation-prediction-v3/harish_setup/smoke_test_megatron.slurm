#!/bin/bash
# ============================================================================
# Smoke Test - Citation Prediction V3 (Megatron Backend)
#
# Verifies the full 5-node pipeline works with Megatron strategy.
# Uses minimal batch sizes and 1 epoch to complete quickly.
#
# 5 nodes x 4 GPUs = 20 GPUs:
#   Node 0: Retriever (4 GPUs for FAISS sharding)
#   Nodes 1-4: Training (16 GPUs)
#
# Megatron parallelism (16 training GPUs):
#   TP=4, PP=2, EP=8, ETP=1 → DP=2 (matches reference script)
#
# Usage:
#   sbatch smoke_test_megatron.slurm
# ============================================================================

#SBATCH --job-name=cit-v3-mcore-smoke
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/cit-v3-mcore-smoke_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/cit-v3-mcore-smoke_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-test

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction_v3/full"
CKPT_DIR="${SCRATCH_BASE}/checkpoints"
LOG_DIR="${SCRATCH_BASE}/logs"

RUN_NAME="cit-v3-mcore-smoke-$(date +%Y%m%d-%H%M%S)"

mkdir -p "${CKPT_DIR}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

# ============================================================================
# MEGATRON PARALLELISM CONFIG
# ============================================================================
MEGATRON_TP=4
MEGATRON_PP=2
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

RECOMPUTE_GRANULARITY="full"
RECOMPUTE_METHOD="uniform"
RECOMPUTE_NUM_LAYERS=1

NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TP=2

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== SMOKE TEST: Citation Prediction V3 (Megatron, 4-node) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo "Parallelism: TP=${MEGATRON_TP}, PP=${MEGATRON_PP}, EP=${MEGATRON_EP}, ETP=${MEGATRON_ETP}"

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
MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

module load proxy/default

# Force working proxy IP (172.28.15.2 is broken, 172.28.15.7 works)
export http_proxy=http://172.28.15.7:8080
export https_proxy=http://172.28.15.7:8080
export HTTP_PROXY=http://172.28.15.7:8080
export HTTPS_PROXY=http://172.28.15.7:8080

# Increase WandB init timeout (cluster proxy can be slow)
export WANDB_INIT_TIMEOUT=300

# Disable Ray's automatic uv runtime_env detection
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK

# Load secrets
ENV_FILE="/home/hk4638/SkyRL/examples/search/harish_setup/.env"
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
WORKER_NODE_2=${NODELIST[3]}
WORKER_NODE_3=${NODELIST[4]}

echo "Retriever node: ${RETRIEVER_NODE}"
echo "Training head: ${HEAD_NODE}"
echo "Training worker 1: ${WORKER_NODE}"
echo "Training worker 2: ${WORKER_NODE_2}"
echo "Training worker 3: ${WORKER_NODE_3}"

# Bypass proxy for internal cluster traffic
export NO_PROXY="localhost,127.0.0.1,${RETRIEVER_NODE},${HEAD_NODE},${WORKER_NODE},${WORKER_NODE_2},${WORKER_NODE_3}"
export no_proxy="$NO_PROXY"
echo "NO_PROXY set to: $NO_PROXY"

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
# START RETRIEVER
# ============================================================================
echo "=== Starting retriever on ${RETRIEVER_NODE} ==="

srun --nodes=1 --ntasks=1 --nodelist=${RETRIEVER_NODE} \
    /bin/bash -c "
        source /usr/licensed/anaconda3/2024.10/etc/profile.d/conda.sh
        export CONDA_ENVS_PATH=${SCRATCH_BASE}/conda/envs
        conda activate retriever
        export HF_HOME=${SCRATCH_BASE}/huggingface
        export TRANSFORMERS_CACHE=${SCRATCH_BASE}/huggingface/transformers
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

        echo 'Starting retriever on \$(hostname) with 4 GPUs for FAISS sharding...'
        python /home/hk4638/SkyRL/examples/citation_prediction/retriever/retrieval_server.py \
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

cd /home/hk4638/SkyRL

echo "Starting Ray head on ${HEAD_NODE} (${HEAD_NODE_IP}:${RAY_PORT})"
srun --nodes=1 --ntasks=1 --nodelist=${HEAD_NODE} \
    bash -c "cd /home/hk4638/SkyRL && \
        export UV_CACHE_DIR=${SCRATCH_BASE}/uv_cache && \
        export UV_LINK_MODE=copy && \
        export VIRTUAL_ENV=/home/hk4638/SkyRL/.venv && \
        export PATH=\$HOME/.local/bin:\$PATH && \
        export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 && \
        export http_proxy=${http_proxy} && \
        export https_proxy=${https_proxy} && \
        export HTTP_PROXY=${HTTP_PROXY} && \
        export HTTPS_PROXY=${HTTPS_PROXY} && \
        export NO_PROXY=${NO_PROXY} && \
        export no_proxy=${no_proxy} && \
        export WANDB_API_KEY=${WANDB_API_KEY} && \
        export WANDB_INIT_TIMEOUT=300 && \
        uv run --active --no-sync ray start --head --port=${RAY_PORT} --num-gpus=4 && \
        sleep infinity" &
RAY_HEAD_PID=$!
sleep 15

echo "Starting Ray worker on ${WORKER_NODE}"
srun --nodes=1 --ntasks=1 --nodelist=${WORKER_NODE} \
    bash -c "cd /home/hk4638/SkyRL && \
        export UV_CACHE_DIR=${SCRATCH_BASE}/uv_cache && \
        export UV_LINK_MODE=copy && \
        export VIRTUAL_ENV=/home/hk4638/SkyRL/.venv && \
        export PATH=\$HOME/.local/bin:\$PATH && \
        export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 && \
        export http_proxy=${http_proxy} && \
        export https_proxy=${https_proxy} && \
        export HTTP_PROXY=${HTTP_PROXY} && \
        export HTTPS_PROXY=${HTTPS_PROXY} && \
        export NO_PROXY=${NO_PROXY} && \
        export no_proxy=${no_proxy} && \
        export WANDB_API_KEY=${WANDB_API_KEY} && \
        export WANDB_INIT_TIMEOUT=300 && \
        uv run --active --no-sync ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=4 && \
        sleep infinity" &
RAY_WORKER_PID=$!
sleep 15

echo "Starting Ray worker 2 on ${WORKER_NODE_2}"
srun --nodes=1 --ntasks=1 --nodelist=${WORKER_NODE_2} \
    bash -c "cd /home/hk4638/SkyRL && \
        export UV_CACHE_DIR=${SCRATCH_BASE}/uv_cache && \
        export UV_LINK_MODE=copy && \
        export VIRTUAL_ENV=/home/hk4638/SkyRL/.venv && \
        export PATH=\$HOME/.local/bin:\$PATH && \
        export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 && \
        export http_proxy=${http_proxy} && \
        export https_proxy=${https_proxy} && \
        export HTTP_PROXY=${HTTP_PROXY} && \
        export HTTPS_PROXY=${HTTPS_PROXY} && \
        export NO_PROXY=${NO_PROXY} && \
        export no_proxy=${no_proxy} && \
        export WANDB_API_KEY=${WANDB_API_KEY} && \
        export WANDB_INIT_TIMEOUT=300 && \
        uv run --active --no-sync ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=4 && \
        sleep infinity" &
RAY_WORKER_2_PID=$!
sleep 15

echo "Starting Ray worker 3 on ${WORKER_NODE_3}"
srun --nodes=1 --ntasks=1 --nodelist=${WORKER_NODE_3} \
    bash -c "cd /home/hk4638/SkyRL && \
        export UV_CACHE_DIR=${SCRATCH_BASE}/uv_cache && \
        export UV_LINK_MODE=copy && \
        export VIRTUAL_ENV=/home/hk4638/SkyRL/.venv && \
        export PATH=\$HOME/.local/bin:\$PATH && \
        export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 && \
        export http_proxy=${http_proxy} && \
        export https_proxy=${https_proxy} && \
        export HTTP_PROXY=${HTTP_PROXY} && \
        export HTTPS_PROXY=${HTTPS_PROXY} && \
        export NO_PROXY=${NO_PROXY} && \
        export no_proxy=${no_proxy} && \
        export WANDB_API_KEY=${WANDB_API_KEY} && \
        export WANDB_INIT_TIMEOUT=300 && \
        uv run --active --no-sync ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=4 && \
        sleep infinity" &
RAY_WORKER_3_PID=$!
sleep 15

# Join Ray cluster from retriever node with 0 GPUs
echo "Joining Ray cluster from ${RETRIEVER_NODE} with 0 GPUs"
uv run --active --no-sync ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=0
sleep 5

echo "Ray cluster started on head=${HEAD_NODE_IP}:${RAY_PORT}"

echo "Verifying Ray cluster status..."
uv run --active --no-sync ray status

# ============================================================================
# RUN MINIMAL TRAINING (MEGATRON)
# ============================================================================
echo "=== Running minimal Megatron training test ==="

export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
echo "Running training with RAY_ADDRESS=${RAY_ADDRESS}"

# Minimal config: small batches, 1 epoch
uv run --active --no-sync --extra megatron -m skyrl.train.entrypoints.main_base \
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
    trainer.strategy=megatron \
    trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
    trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
    trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
    trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
    trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
    trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
    trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
    trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
    trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
    trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
    trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=$RECOMPUTE_GRANULARITY \
    trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=$RECOMPUTE_METHOD \
    trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=$RECOMPUTE_NUM_LAYERS \
    trainer.use_sample_packing=true \
    trainer.flash_attn=true \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.policy_num_nodes=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    trainer.placement.ref_num_nodes=4 \
    generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
    generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.6 \
    trainer.epochs=1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=12 \
    trainer.policy_mini_batch_size=12 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=4096 \
    generator.max_input_length=32768 \
    generator.sampling_params.max_generate_length=1024 \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=2 \
    generator.max_turns=6 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>", "</done>"]' \
    environment.env_class="citation_prediction_v2" \
    environment.skyrl_gym.max_env_workers=4 \
    environment.skyrl_gym.citation_prediction_v2.log_requests=false \
    environment.skyrl_gym.citation_prediction_v2.search_url="${RETRIEVER_URL}" \
    environment.skyrl_gym.citation_prediction_v2.topk=5 \
    environment.skyrl_gym.citation_prediction_v2.max_predictions_ratio=2.0 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl-citation-prediction-v3" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=1000 \
    trainer.max_ckpts_to_keep=1 \
    trainer.resume_mode=none \
    trainer.ckpt_path="${CKPT_DIR}/${RUN_NAME}" \
    trainer.eval_batch_size=12 \
    trainer.eval_before_train=false \
    trainer.eval_interval=1000

TRAINING_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
uv run --active --no-sync ray stop || true
kill ${RAY_HEAD_PID} 2>/dev/null || true
kill ${RAY_WORKER_PID} 2>/dev/null || true
kill ${RAY_WORKER_2_PID} 2>/dev/null || true
kill ${RAY_WORKER_3_PID} 2>/dev/null || true
kill ${RETRIEVER_PID} 2>/dev/null || true
wait ${RAY_HEAD_PID} 2>/dev/null || true
wait ${RAY_WORKER_PID} 2>/dev/null || true
wait ${RAY_WORKER_2_PID} 2>/dev/null || true
wait ${RAY_WORKER_3_PID} 2>/dev/null || true

echo "=== MEGATRON SMOKE TEST COMPLETE ==="
echo "Exit code: ${TRAINING_EXIT_CODE}"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Ready for full Megatron training run."
else
    echo "FAILED! Check logs before running full job."
fi

exit ${TRAINING_EXIT_CODE}
