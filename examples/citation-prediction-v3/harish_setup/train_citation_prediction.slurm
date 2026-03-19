#!/bin/bash
# ============================================================================
# Citation Prediction V3 Training Script (recall-based)
#
# Usage:
#   sbatch train_citation_prediction.slurm
#
# 4 nodes × 4 GPUs = 16 GPUs:
#   Node 0: Retriever (4 GPUs for FAISS sharding)
#   Node 1: Training head (4 GPUs)
#   Node 2: Training worker 1 (4 GPUs)
#   Node 3: Training worker 2 (4 GPUs)
# ============================================================================

#SBATCH --job-name=citation-prediction-v3
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-short

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction_v3/full"
CKPT_DIR="${SCRATCH_BASE}/checkpoints"
LOG_DIR="${SCRATCH_BASE}/logs"

# Sweep hyperparameters (override via sbatch --export)
SWEEP_LR=${SWEEP_LR:-"1.0e-6"}
SWEEP_KL=${SWEEP_KL:-"true"}
SWEEP_CLIP=${SWEEP_CLIP:-"0.2"}

# Build descriptive run name
KL_TAG=$( [ "$SWEEP_KL" = "true" ] && echo "kl" || echo "nokl" )
RUN_NAME="cit-v3-lr${SWEEP_LR}_${KL_TAG}_clip${SWEEP_CLIP}-$(date +%Y%m%d-%H%M%S)"

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
export VIRTUAL_ENV="/home/hk4638/SkyRL/.venv"

# Local path to cached model (avoids HuggingFace network requests)
MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

# Load proxy for WandB access (api.wandb.ai is on the approved list)
module load proxy/default

# Force working proxy IP (172.28.15.2 is broken, 172.28.15.7 works)
export http_proxy=http://172.28.15.7:8080
export https_proxy=http://172.28.15.7:8080
export HTTP_PROXY=http://172.28.15.7:8080
export HTTPS_PROXY=http://172.28.15.7:8080

# Increase WandB init timeout (cluster proxy can be slow)
export WANDB_INIT_TIMEOUT=300

# CRITICAL: Disable Ray's automatic uv runtime_env detection
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK

# Reduce GPU memory fragmentation (helps MoE aux loss large tensor allocations)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load secrets (shared .env with WANDB_API_KEY + GEMINI_API_KEY)
ENV_FILE="/home/hk4638/SkyRL/examples/search/harish_setup/.env"
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
WORKER_NODE_2=${NODELIST[3]}

echo "Retriever node: ${RETRIEVER_NODE}"
echo "Training head: ${HEAD_NODE}"
echo "Training worker 1: ${WORKER_NODE}"
echo "Training worker 2: ${WORKER_NODE_2}"

# Bypass proxy for internal cluster traffic
export NO_PROXY="localhost,127.0.0.1,${RETRIEVER_NODE},${HEAD_NODE},${WORKER_NODE},${WORKER_NODE_2}"
export no_proxy="$NO_PROXY"
echo "NO_PROXY set to: $NO_PROXY"

# ============================================================================
# RETRIEVER CONFIGURATION
# ============================================================================
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_4b"}
# Corpus and index are shared across v1/v2/v3
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
# START RETRIEVER SERVER
# ============================================================================
echo "=== Starting retriever server on ${RETRIEVER_NODE} ==="
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Index: ${INDEX_FILE}"
echo "Corpus: ${CORPUS_FILE}"

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

cd /home/hk4638/SkyRL

# Start Ray head
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

# Join Ray cluster from retriever node with 0 GPUs
echo "Joining Ray cluster from ${RETRIEVER_NODE} with 0 GPUs"
uv run --active --no-sync ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=0
sleep 5

echo "Ray cluster started on head=${HEAD_NODE_IP}:${RAY_PORT}"

# Verify cluster status
echo "Verifying Ray cluster status..."
uv run --active --no-sync ray status

# ============================================================================
# RUN TRAINING
# ============================================================================
echo "=== Starting training ==="
echo "Run name: ${RUN_NAME}"
echo "Data dir: ${DATA_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}/${RUN_NAME}"
echo "Retriever URL: ${RETRIEVER_URL}"

# CRITICAL: Set RAY_ADDRESS so training connects to existing cluster
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
echo "Connecting to Ray cluster at RAY_ADDRESS=${RAY_ADDRESS}"

# Run training with 12 GPUs (3 nodes × 4 GPUs)
uv run --active --no-sync --extra fsdp -m skyrl.train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.optimizer_config.lr=${SWEEP_LR} \
    trainer.policy.optimizer_config.max_grad_norm=0.5 \
    trainer.policy.optimizer_config.num_warmup_steps=31 \
    trainer.algorithm.use_kl_loss=${SWEEP_KL} \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.eps_clip_high=${SWEEP_CLIP} \
    trainer.algorithm.use_tis=true \
    trainer.algorithm.tis_imp_ratio_cap=2.0 \
    trainer.policy.model.path="${MODEL_PATH}" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.policy.use_liger_kernel=true \
    trainer.ref.use_liger_kernel=true \
    trainer.policy.sequence_parallel_size=1 \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.gradient_checkpointing_use_reentrant=false \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.policy_num_nodes=3 \
    trainer.placement.ref_num_gpus_per_node=4 \
    trainer.placement.ref_num_nodes=3 \
    generator.inference_engine.num_engines=6 \
    generator.inference_engine.tensor_parallel_size=2 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.9 \
    trainer.epochs=5 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=128 \
    trainer.policy_mini_batch_size=128 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=4096 \
    generator.max_input_length=32768 \
    generator.sampling_params.max_generate_length=1024 \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=10 \
    generator.max_turns=6 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>", "</done>"]' \
    environment.env_class="citation_prediction_v2" \
    environment.skyrl_gym.max_env_workers=16 \
    environment.skyrl_gym.citation_prediction_v2.log_requests=false \
    environment.skyrl_gym.citation_prediction_v2.search_url="${RETRIEVER_URL}" \
    environment.skyrl_gym.citation_prediction_v2.topk=5 \
    environment.skyrl_gym.citation_prediction_v2.max_predictions_ratio=2.0 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl-citation-prediction-v3" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=20 \
    trainer.hf_save_interval=100 \
    trainer.max_ckpts_to_keep=5 \
    trainer.resume_mode=latest \
    trainer.ckpt_path="${CKPT_DIR}/${RUN_NAME}" \
    trainer.eval_batch_size=256 \
    trainer.eval_before_train=false \
    generator.eval_sampling_params.temperature=0 \
    generator.eval_sampling_params.stop='["</search>", "</done>"]' \
    trainer.export_path="${CKPT_DIR}/${RUN_NAME}/exports" \
    trainer.eval_interval=50

TRAINING_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="

# Stop Ray on local node
uv run --active --no-sync ray stop || true

# Kill background srun processes
kill ${RAY_HEAD_PID} 2>/dev/null || true
kill ${RAY_WORKER_PID} 2>/dev/null || true
kill ${RAY_WORKER_2_PID} 2>/dev/null || true
kill ${RETRIEVER_PID} 2>/dev/null || true

wait ${RAY_HEAD_PID} 2>/dev/null || true
wait ${RAY_WORKER_PID} 2>/dev/null || true
wait ${RAY_WORKER_2_PID} 2>/dev/null || true

echo "=== Job complete ==="
echo "End time: $(date)"
echo "Training exit code: ${TRAINING_EXIT_CODE}"

exit ${TRAINING_EXIT_CODE}
