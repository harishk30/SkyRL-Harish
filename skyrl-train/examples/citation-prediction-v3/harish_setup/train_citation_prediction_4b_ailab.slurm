#!/bin/bash
# ============================================================================
# Citation Prediction V3 Training Script - Qwen3-4B (H200 ai-lab)
#
# Usage (single run):
#   sbatch --export=ALL,SWEEP_BP=128,SWEEP_N=10 train_citation_prediction_4b_ailab.slurm
#
# Usage (job array, 1 at a time):
#   sbatch --export=ALL,SWEEP_BP=128,SWEEP_N_LIST="20 40" --array=0-1%1 train_citation_prediction_4b_ailab.slurm
#
# See launch_ailab_sweep.sh for the full sweep.
#
# Single node, 2 H200 GPUs (141 GB each):
#   GPU 0: Shared — retriever + training
#   GPU 1: Training only
# LR auto-computed via sqrt(Bp) scaling: lr = 1e-6 * sqrt(Bp / 128)
# ============================================================================

#SBATCH --job-name=cit-pred-v3-4b
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_%A_%a.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=17:00:00
#SBATCH --partition=ailab

set -e
MAIN_PID=$$

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SWEEP_DATA_DIR:-${SCRATCH_BASE}/data/citation_prediction_v3/full}"
CKPT_DIR="${SCRATCH_BASE}/checkpoints"
LOG_DIR="${SCRATCH_BASE}/logs"

# Handle job array: pick n from SWEEP_N_LIST if set, else use SWEEP_N
if [ -n "${SWEEP_N_LIST}" ] && [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    N_ARRAY=(${SWEEP_N_LIST})
    SWEEP_N=${N_ARRAY[$SLURM_ARRAY_TASK_ID]}
    echo "Job array task ${SLURM_ARRAY_TASK_ID}: picked n=${SWEEP_N} from list [${SWEEP_N_LIST}]"
else
    SWEEP_N=${SWEEP_N:-"10"}
fi

SWEEP_BP=${SWEEP_BP:-"128"}
SWEEP_KL=${SWEEP_KL:-"false"}
SWEEP_CLIP=${SWEEP_CLIP:-"0.2"}
SWEEP_CKPT_INTERVAL=${SWEEP_CKPT_INTERVAL:-"20"}

# LR: use override if set, otherwise auto-compute via IsoCompute sqrt scaling
B=$((SWEEP_BP * SWEEP_N))
if [ -n "${SWEEP_LR_OVERRIDE}" ]; then
    SWEEP_LR="${SWEEP_LR_OVERRIDE}"
    echo "Using override LR: ${SWEEP_LR} for Bp=${SWEEP_BP}, n=${SWEEP_N}, B=${B}"
else
    SWEEP_LR=$(python3 -c "import math; print(f'{1e-6 * math.sqrt(${B} / 1024):.6e}')")
    echo "Auto-computed LR: ${SWEEP_LR} for B=${B} (Bp=${SWEEP_BP} x n=${SWEEP_N})"
fi

# Deterministic run name (enables checkpoint resume if resubmitted)
KL_TAG=$( [ "$SWEEP_KL" = "true" ] && echo "kl" || echo "nokl" )
LR_SHORT=$(python3 -c "print(f'{float(\"${SWEEP_LR}\"):.0e}')")
RUN_PREFIX="${SWEEP_RUN_PREFIX:-cit-v3-4b-h200}"
RUN_NAME="${RUN_PREFIX}-bp${SWEEP_BP}_n${SWEEP_N}_lr${LR_SHORT}_${KL_TAG}"

# Cap concurrent environment work as n grows to avoid overloading the shared
# local retriever on GPU 0 and triggering retriever 500 retry storms.
if [ -n "${MAX_ENV_WORKERS_OVERRIDE}" ]; then
    MAX_ENV_WORKERS="${MAX_ENV_WORKERS_OVERRIDE}"
elif [ "${SWEEP_N}" -ge 40 ]; then
    MAX_ENV_WORKERS=4
elif [ "${SWEEP_N}" -ge 20 ]; then
    MAX_ENV_WORKERS=8
else
    MAX_ENV_WORKERS=16
fi

mkdir -p "${CKPT_DIR}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

# Pick localhost ports dynamically so multiple jobs can share a node safely.
pick_free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

RETRIEVER_PORT=$(pick_free_port)
RAY_PORT=$(pick_free_port)
while [ "${RAY_PORT}" = "${RETRIEVER_PORT}" ]; do
    RAY_PORT=$(pick_free_port)
done

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== Setting up environment ==="
echo "Job ID: ${SLURM_JOB_ID} (array: ${SLURM_ARRAY_TASK_ID:-none})"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo "Config: Bp=${SWEEP_BP}, n=${SWEEP_N}, B=${B}, LR=${SWEEP_LR}, KL=${SWEEP_KL}"
echo "Env workers: ${MAX_ENV_WORKERS}"
echo "Ports: retriever=${RETRIEVER_PORT}, ray=${RAY_PORT}"

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
export VIRTUAL_ENV="/home/hk4638/SkyRL/skyrl-train/.venv"

MODEL_PATH="${SWEEP_MODEL_PATH:-/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"

module load proxy/default
export http_proxy=http://172.28.15.7:8080
export https_proxy=http://172.28.15.7:8080
export HTTP_PROXY=http://172.28.15.7:8080
export HTTPS_PROXY=http://172.28.15.7:8080
export WANDB_INIT_TIMEOUT=300

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK

ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env"
if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment from ${ENV_FILE}"
    set -a && source "${ENV_FILE}" && set +a
else
    echo "WARNING: .env file not found at ${ENV_FILE}"
fi

THIS_NODE=$(hostname)
export NO_PROXY="localhost,127.0.0.1,${THIS_NODE}"
export no_proxy="$NO_PROXY"

# ============================================================================
# RETRIEVER (GPU 0, shared with training)
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

echo "=== Starting retriever on GPU 0 ==="
(
    source /usr/licensed/anaconda3/2024.10/etc/profile.d/conda.sh
    export CONDA_ENVS_PATH=${SCRATCH_BASE}/conda/envs
    conda activate retriever
    export HF_HOME=${SCRATCH_BASE}/huggingface
    export TRANSFORMERS_CACHE=${SCRATCH_BASE}/huggingface/transformers
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export CUDA_VISIBLE_DEVICES=0

    python /home/hk4638/SkyRL/skyrl-train/examples/citation_prediction/retriever/retrieval_server.py \
        --index_path ${INDEX_FILE} \
        --corpus_path ${CORPUS_FILE} \
        --topk 3 \
        --retriever_name ${RETRIEVER_NAME} \
        --retriever_model ${RETRIEVER_MODEL_PATH} \
        --host 127.0.0.1 \
        --port ${RETRIEVER_PORT} \
) > "${LOG_DIR}/retriever_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log" 2>&1 &

RETRIEVER_PID=$!

# Fail fast if the retriever crashes after startup. Otherwise training can loop
# indefinitely retrying localhost:${RETRIEVER_PORT} while making no progress.
(
    while kill -0 "${RETRIEVER_PID}" 2>/dev/null; do
        sleep 15
    done
    echo "ERROR: Retriever process ${RETRIEVER_PID} died; terminating job."
    kill -TERM "${MAIN_PID}" 2>/dev/null || true
) &
RETRIEVER_MONITOR_PID=$!

RETRIEVER_URL="http://127.0.0.1:${RETRIEVER_PORT}/retrieve"
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
    exit 1
fi

# ============================================================================
# RAY CLUSTER (GPUs 0-1, single node)
# ============================================================================
echo "=== Starting Ray cluster on GPUs 0-1 ==="
cd /home/hk4638/SkyRL/skyrl-train

export CUDA_VISIBLE_DEVICES=0,1
uv run --active --no-sync ray start --head --port="${RAY_PORT}" --num-gpus=2
sleep 5

echo "Ray cluster started (single node, 2 GPUs)"
uv run --active --no-sync ray status --address="127.0.0.1:${RAY_PORT}"

# ============================================================================
# TRAINING
# ============================================================================
echo "=== Starting training ==="
echo "Run name: ${RUN_NAME}"
echo "Checkpoint dir: ${CKPT_DIR}/${RUN_NAME}"

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

uv run --active --no-sync --extra vllm -m skyrl_train.entrypoints.main_base \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.optimizer_config.lr=${SWEEP_LR} \
    trainer.policy.optimizer_config.max_grad_norm=0.5 \
    trainer.policy.optimizer_config.num_warmup_steps=10 \
    trainer.algorithm.use_kl_loss=${SWEEP_KL} \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.eps_clip_high=${SWEEP_CLIP} \
    trainer.algorithm.use_tis=true \
    trainer.algorithm.tis_imp_ratio_cap=2.0 \
    trainer.policy.model.path="${MODEL_PATH}" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    +trainer.policy.use_liger_kernel=true \
    trainer.policy.sequence_parallel_size=1 \
    +trainer.ref.use_liger_kernel=true \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.ref.fsdp_config.cpu_offload=false \
    trainer.gradient_checkpointing_use_reentrant=false \
    trainer.placement.policy_num_gpus_per_node=2 \
    trainer.placement.policy_num_nodes=1 \
    trainer.placement.ref_num_gpus_per_node=2 \
    trainer.placement.ref_num_nodes=1 \
    generator.num_inference_engines=2 \
    generator.inference_engine_tensor_parallel_size=1 \
    generator.backend=vllm \
    generator.run_engines_locally=true \
    generator.weight_sync_backend=nccl \
    generator.gpu_memory_utilization=0.8 \
    trainer.epochs=50 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=${SWEEP_BP} \
    trainer.policy_mini_batch_size=128 \
    trainer.micro_forward_batch_size_per_gpu=2 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=4096 \
    generator.max_input_length=32768 \
    generator.sampling_params.max_generate_length=4096 \
    generator.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=${SWEEP_N} \
    generator.max_turns=4 \
    generator.sampling_params.temperature=0.6 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.stop='["</search>", "</done>"]' \
    environment.env_class="citation_prediction_v2" \
    environment.skyrl_gym.max_env_workers=${MAX_ENV_WORKERS} \
    environment.skyrl_gym.citation_prediction_v2.log_requests=false \
    environment.skyrl_gym.citation_prediction_v2.search_url="${RETRIEVER_URL}" \
    environment.skyrl_gym.citation_prediction_v2.topk=10 \
    environment.skyrl_gym.citation_prediction_v2.max_predictions_ratio=2.0 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl-citation-prediction-v3" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_interval=${SWEEP_CKPT_INTERVAL} \
    trainer.hf_save_interval=100 \
    trainer.max_ckpts_to_keep=5 \
    trainer.resume_mode=latest \
    trainer.ckpt_path="${CKPT_DIR}/${RUN_NAME}" \
    trainer.eval_batch_size=256 \
    trainer.eval_before_train=true \
    generator.eval_sampling_params.temperature=0 \
    generator.eval_sampling_params.stop='["</search>", "</done>"]' \
    trainer.export_path="${CKPT_DIR}/${RUN_NAME}/exports" \
    trainer.eval_interval=10

TRAINING_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
uv run --active --no-sync ray stop || true
kill ${RETRIEVER_PID} 2>/dev/null || true
kill ${RETRIEVER_MONITOR_PID} 2>/dev/null || true

echo "=== Job complete ==="
echo "End time: $(date)"
echo "Training exit code: ${TRAINING_EXIT_CODE}"
exit ${TRAINING_EXIT_CODE}
