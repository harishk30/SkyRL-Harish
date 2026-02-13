#!/bin/bash
# ============================================================================
# Evaluate base model on citation prediction using SkyRL's eval pipeline
#
# Uses main_generate (eval-only entrypoint) which runs the same multi-turn
# rollouts as training but without GRPO updates. Reports avg_score, pass@1.
#
# 2 nodes × 4 GPUs:
#   Node 0: Retriever (4 GPUs for FAISS sharding)
#   Node 1: vLLM inference (2 engines × TP=2)
#
# Env vars (all optional):
#   EVAL_SPLIT       train | validation | test  (default: train)
#   PROMPT_STYLE     short | extended           (default: short)
#   EMBEDDING_MODEL  qwen3_4b | qwen3_06b      (default: qwen3_4b)
#
# Usage — run all 4 combos on train to find best pre-RL config:
#   PROMPT_STYLE=short    EMBEDDING_MODEL=qwen3_4b  sbatch eval_base_model.slurm
#   PROMPT_STYLE=short    EMBEDDING_MODEL=qwen3_06b sbatch eval_base_model.slurm
#   PROMPT_STYLE=extended EMBEDDING_MODEL=qwen3_4b  sbatch eval_base_model.slurm
#   PROMPT_STYLE=extended EMBEDDING_MODEL=qwen3_06b sbatch eval_base_model.slurm
#
# Or quick validation check:
#   EVAL_SPLIT=validation sbatch eval_base_model.slurm
# ============================================================================

#SBATCH --job-name=eval-citation
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/eval-citation_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/eval-citation_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction"
LOG_DIR="${SCRATCH_BASE}/logs"
EVAL_SPLIT=${EVAL_SPLIT:-"train"}              # train, validation, or test
PROMPT_STYLE=${PROMPT_STYLE:-"short"}          # "short" or "extended"
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_4b"} # "qwen3_4b" or "qwen3_06b"

RUN_NAME="eval-${PROMPT_STYLE}-${EMBEDDING_MODEL}-${EVAL_SPLIT}-$(date +%Y%m%d-%H%M%S)"

mkdir -p "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== SkyRL eval on citation prediction ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo "Config: prompt=${PROMPT_STYLE}, embed=${EMBEDDING_MODEL}, split=${EVAL_SPLIT}"

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

MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

module load proxy/default

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK

# Load secrets for WandB
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
EVAL_NODE=${NODELIST[1]}

echo "Retriever node: ${RETRIEVER_NODE}"
echo "Eval node: ${EVAL_NODE}"

export NO_PROXY="localhost,127.0.0.1,${RETRIEVER_NODE},${EVAL_NODE}"
export no_proxy="$NO_PROXY"
echo "NO_PROXY set to: $NO_PROXY"

# ============================================================================
# RETRIEVER CONFIGURATION
# ============================================================================
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
else
    echo "ERROR: Unknown EMBEDDING_MODEL=${EMBEDDING_MODEL}. Use qwen3_4b or qwen3_06b."
    exit 1
fi

# Verify index exists before proceeding
if [ ! -f "${INDEX_FILE}" ]; then
    echo "ERROR: FAISS index not found at ${INDEX_FILE}"
    echo "Build it first with: sbatch examples/citation_prediction/data/build_${EMBEDDING_MODEL/qwen3_/}b_index.sbatch"
    exit 1
fi

echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Index: ${INDEX_FILE}"
echo "Corpus: ${CORPUS_FILE}"

# ============================================================================
# START RETRIEVER
# ============================================================================
echo "=== Starting retriever on ${RETRIEVER_NODE} ==="

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
    echo "ERROR: Retriever failed to start!"
    cat "${LOG_DIR}/retriever_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# START RAY ON EVAL NODE
# ============================================================================
echo "=== Starting Ray on ${EVAL_NODE} ==="

EVAL_NODE_IP=$(srun --nodes=1 --ntasks=1 --nodelist=${EVAL_NODE} hostname -i)
RAY_PORT=6379

cd /home/hk4638/SkyRL/skyrl-train

# Start Ray head on eval node
srun --nodes=1 --ntasks=1 --nodelist=${EVAL_NODE} \
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

# Join Ray cluster from retriever node with 0 GPUs
echo "Joining Ray cluster from ${RETRIEVER_NODE} with 0 GPUs"
uv run --active --frozen ray start --address=${EVAL_NODE_IP}:${RAY_PORT} --num-gpus=0
sleep 5

echo "Ray cluster started on head=${EVAL_NODE_IP}:${RAY_PORT}"
uv run --active --frozen ray status

# ============================================================================
# RUN SKYRL EVAL
# ============================================================================
echo "=== Running SkyRL evaluation ==="
echo "Model: ${MODEL_PATH}"
echo "Prompt style: ${PROMPT_STYLE}"
echo "Data: ${DATA_DIR}/${PROMPT_STYLE}/${EVAL_SPLIT}.parquet"
echo "Retriever: ${RETRIEVER_URL}"

export RAY_ADDRESS="${EVAL_NODE_IP}:${RAY_PORT}"
echo "Connecting to Ray cluster at RAY_ADDRESS=${RAY_ADDRESS}"

# Eval-only entrypoint: same generator/env config as training, no GRPO updates
# - colocate_all=false: no training workers to share GPUs with
# - gpu_memory_utilization=0.9: vLLM gets full GPU (no FSDP competition)
# - eval_sampling_params.temperature=0: greedy decoding for deterministic eval
uv run --active --frozen --extra vllm -m skyrl_train.entrypoints.main_generate \
    data.val_data="['${DATA_DIR}/${PROMPT_STYLE}/${EVAL_SPLIT}.parquet']" \
    trainer.policy.model.path="${MODEL_PATH}" \
    trainer.policy.use_liger_kernel=true \
    trainer.policy.sequence_parallel_size=2 \
    trainer.ref.use_liger_kernel=true \
    trainer.strategy=fsdp2 \
    trainer.placement.colocate_all=false \
    generator.num_inference_engines=2 \
    generator.inference_engine_tensor_parallel_size=2 \
    generator.backend=vllm \
    generator.run_engines_locally=true \
    generator.gpu_memory_utilization=0.9 \
    generator.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.max_turns=4 \
    generator.max_input_length=8192 \
    generator.sampling_params.max_generate_length=500 \
    generator.sampling_params.stop='["</search>"]' \
    generator.eval_sampling_params.temperature=0 \
    generator.eval_sampling_params.stop='["</search>"]' \
    generator.eval_n_samples_per_prompt=1 \
    trainer.eval_interval=1 \
    trainer.eval_batch_size=256 \
    trainer.max_prompt_length=2048 \
    environment.env_class="citation_prediction" \
    environment.skyrl_gym.max_env_workers=16 \
    environment.skyrl_gym.citation_prediction.log_requests=false \
    environment.skyrl_gym.citation_prediction.search_url="${RETRIEVER_URL}" \
    environment.skyrl_gym.citation_prediction.topk=3 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl-citation-prediction" \
    trainer.run_name="${RUN_NAME}" \
    trainer.export_path="${LOG_DIR}/eval_${SLURM_JOB_ID}"

EVAL_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
uv run --active --frozen ray stop || true
kill ${RAY_HEAD_PID} 2>/dev/null || true
kill ${RETRIEVER_PID} 2>/dev/null || true
wait ${RAY_HEAD_PID} 2>/dev/null || true

echo "=== Eval complete ==="
echo "End time: $(date)"
echo "Exit code: ${EVAL_EXIT_CODE}"
echo "Results logged to WandB project: skyrl-citation-prediction, run: ${RUN_NAME}"
echo "Trajectories dumped to: ${LOG_DIR}/eval_${SLURM_JOB_ID}/dumped_evals/"

exit ${EVAL_EXIT_CODE}
