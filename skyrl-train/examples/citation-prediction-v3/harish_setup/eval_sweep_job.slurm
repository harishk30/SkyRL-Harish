#!/bin/bash
# ============================================================================
# Single sweep eval job using SkyRL's main_generate pipeline (V3).
#
# Evaluates one (m, n) hyperparameter combo with pass@k sampling.
#
# Node 0: Retriever (4 GPUs for FAISS sharding) + joins Ray + runs main_generate
# Node 1: Ray head with vLLM inference engines (4 GPUs)
#
# Env vars (required):
#   MAX_TURNS        Number of search turns (m)
#   TOPK             Number of search results per turn (n)
#
# Env vars (optional):
#   NUM_SAMPLES      Number of stochastic samples per prompt (k; default: 20)
#   TEMPERATURE      Sampling temperature (default: 1.0)
#   SEED             Random seed for prompt sampling (default: 42)
#   EMBEDDING_MODEL  qwen3_4b | qwen3_06b (default: qwen3_4b)
#   MAX_EXAMPLES     Number of prompts to evaluate (default: 100)
#   PROMPT_VARIANT   with_intro | no_intro (default: with_intro)
#
# Usage:
#   MAX_TURNS=4 TOPK=3 sbatch eval_sweep_job.slurm
#   MAX_TURNS=2 TOPK=5 NUM_SAMPLES=10 sbatch eval_sweep_job.slurm
#   PROMPT_VARIANT=no_intro MAX_TURNS=4 TOPK=5 sbatch eval_sweep_job.slurm
# ============================================================================

#SBATCH --job-name=sweep-eval-v3
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/sweep-eval-v3_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/sweep-eval-v3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hk4638@princeton.edu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-test

set -e

cd /home/hk4638/SkyRL/skyrl-train

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction_v3"
LOG_DIR="${SCRATCH_BASE}/logs"

# Required env vars
if [ -z "$MAX_TURNS" ] || [ -z "$TOPK" ]; then
    echo "ERROR: MAX_TURNS and TOPK env vars are required"
    echo "Usage: MAX_TURNS=4 TOPK=3 sbatch eval_sweep_job.slurm"
    exit 1
fi

# Optional env vars with defaults
NUM_SAMPLES=${NUM_SAMPLES:-20}
TEMPERATURE=${TEMPERATURE:-"1.0"}
SEED=${SEED:-42}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_4b"}
MAX_EXAMPLES=${MAX_EXAMPLES:-100}
MODEL_NAME=${MODEL_NAME:-"qwen3_4b"}
PROMPT_VARIANT=${PROMPT_VARIANT:-"with_intro"}

# Output directory
if [ "$MODEL_NAME" = "qwen3_4b_thinking" ]; then
    DIR_PREFIX="v3-thinking"
elif [ "$MODEL_NAME" = "qwen3_4b_yarn" ]; then
    DIR_PREFIX="v3-yarn"
elif [ "$MODEL_NAME" = "qwen3_30b_a3b" ]; then
    DIR_PREFIX="v3-30b"
else
    DIR_PREFIX="v3"
fi
# Append prompt variant to dir prefix if not default
if [ "$PROMPT_VARIANT" != "with_intro" ]; then
    DIR_PREFIX="${DIR_PREFIX}-${PROMPT_VARIANT}"
fi
PART_SUFFIX=""
if [ -n "$PART" ]; then
    PART_SUFFIX="_part${PART}"
fi
SWEEP_DIR="${LOG_DIR}/sweep/${DIR_PREFIX}_m${MAX_TURNS}_n${TOPK}_k${NUM_SAMPLES}${PART_SUFFIX}"
RUN_NAME="sweep-${DIR_PREFIX}-m${MAX_TURNS}-n${TOPK}-k${NUM_SAMPLES}-seed${SEED}${PART_SUFFIX}"

mkdir -p "${SWEEP_DIR}" "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== Sweep eval (V3): m=${MAX_TURNS}, n=${TOPK}, k=${NUM_SAMPLES} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo "Config: temp=${TEMPERATURE}, seed=${SEED}, embed=${EMBEDDING_MODEL}"
echo "Output: ${SWEEP_DIR}"

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

# Model selection
if [ "$MODEL_NAME" = "qwen3_4b_thinking" ]; then
    MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea"
    MAX_INPUT_LENGTH=163840
    MAX_GEN_LENGTH=8192
    ROPE_SCALING="+generator.engine_init_kwargs.max_model_len=180224"
elif [ "$MODEL_NAME" = "qwen3_4b_yarn" ]; then
    MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B-YaRN"
    MAX_INPUT_LENGTH=131072
    MAX_GEN_LENGTH=1024
    ROPE_SCALING="+generator.engine_init_kwargs.max_model_len=131072"
elif [ "$MODEL_NAME" = "qwen3_30b_a3b" ]; then
    MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    MAX_INPUT_LENGTH=32000
    MAX_GEN_LENGTH=1024
    ROPE_SCALING=""
else
    MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
    MAX_INPUT_LENGTH=32000
    MAX_GEN_LENGTH=1024
    ROPE_SCALING=""
fi

module load proxy/default

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK

# Load secrets
ENV_FILE="/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env"
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
# DATA FILE — pre-sampled parquet created by sweep_eval.sh
# ============================================================================
PART=${PART:-""}
VARIANT_DIR="${DATA_DIR}/${PROMPT_VARIANT}"
if [ -n "$PART" ]; then
    DATA_FILE="${VARIANT_DIR}/train_sample${MAX_EXAMPLES}_seed${SEED}_part${PART}.parquet"
else
    DATA_FILE="${VARIANT_DIR}/train_sample${MAX_EXAMPLES}_seed${SEED}.parquet"
fi
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Sampled parquet not found at ${DATA_FILE}"
    echo "Run sweep_eval.sh first to create it."
    exit 1
fi
echo "Data: ${DATA_FILE}"

# ============================================================================
# RETRIEVER CONFIGURATION
# ============================================================================
CORPUS_FILE="${SCRATCH_BASE}/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl"

if [ "$EMBEDDING_MODEL" = "qwen3_4b" ]; then
    RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
    INDEX_FILE="${SCRATCH_BASE}/data/citation_prediction/qwen3_4b_embed/qwen3_Flat.index"
    RETRIEVER_NAME=qwen3
elif [ "$EMBEDDING_MODEL" = "qwen3_06b" ]; then
    RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    INDEX_FILE="/home/hk4638/scratch/shared/data/qwen3_06_embed/qwen3_Flat.index"
    RETRIEVER_NAME=qwen3
else
    echo "ERROR: Unknown EMBEDDING_MODEL=${EMBEDDING_MODEL}. Use qwen3_4b or qwen3_06b."
    exit 1
fi

if [ ! -f "${INDEX_FILE}" ]; then
    echo "ERROR: FAISS index not found at ${INDEX_FILE}"
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
    /bin/bash -c "
        source /usr/licensed/anaconda3/2024.10/etc/profile.d/conda.sh
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
            --topk ${TOPK} \
            --retriever_name ${RETRIEVER_NAME} \
            --retriever_model ${RETRIEVER_MODEL_PATH} \
            --faiss_gpu
    " > "${LOG_DIR}/retriever_sweep_v3_${SLURM_JOB_ID}.log" 2>&1 &

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
    cat "${LOG_DIR}/retriever_sweep_v3_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# START RAY + RUN EVAL ON EVAL NODE
# ============================================================================
echo "=== Starting Ray + eval on ${EVAL_NODE} ==="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_FILE}"
echo "Retriever: ${RETRIEVER_URL}"
echo "Params: m=${MAX_TURNS}, n=${TOPK}, k=${NUM_SAMPLES}, temp=${TEMPERATURE}"

RAY_PORT=6379

# For single-turn (m=1), SkyRL requires max_input_length == max_prompt_length
if [ "${MAX_TURNS}" -eq 1 ]; then
    MAX_PROMPT_LENGTH=30000
else
    MAX_PROMPT_LENGTH=4096
fi

srun --nodes=1 --ntasks=1 --nodelist=${EVAL_NODE} \
    /bin/bash -c "cd /home/hk4638/SkyRL/skyrl-train && \
        export HOME=/home/hk4638 && \
        export UV_CACHE_DIR=${SCRATCH_BASE}/uv_cache && \
        export UV_LINK_MODE=copy && \
        export VIRTUAL_ENV=/home/hk4638/SkyRL/skyrl-train/.venv && \
        export PATH=/home/hk4638/.local/bin:\$PATH && \
        export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 && \
        export NO_PROXY=localhost,127.0.0.1,${RETRIEVER_NODE},${EVAL_NODE} && \
        export no_proxy=\$NO_PROXY && \
        export WANDB_API_KEY=${WANDB_API_KEY} && \
        export http_proxy=${http_proxy} && \
        export https_proxy=${https_proxy} && \
        export HTTP_PROXY=${HTTP_PROXY} && \
        export HTTPS_PROXY=${HTTPS_PROXY} && \
        export HF_HOME=${HF_HOME} && \
        export HF_HUB_OFFLINE=1 && \
        export TRANSFORMERS_OFFLINE=1 && \
        export TRANSFORMERS_CACHE=${HF_HOME} && \
        echo 'Starting Ray head...' && \
        uv run --active --frozen ray start --head --port=${RAY_PORT} --num-gpus=4 && \
        sleep 3 && \
        export RAY_ADDRESS=127.0.0.1:${RAY_PORT} && \
        echo 'Ray started, launching eval...' && \
        uv run --active --frozen --extra vllm -m skyrl_train.entrypoints.main_generate \
    data.val_data=\"['${DATA_FILE}']\" \
    trainer.policy.model.path=\"${MODEL_PATH}\" \
    +trainer.policy.use_liger_kernel=true \
    +trainer.ref.use_liger_kernel=true \
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
    generator.max_turns=${MAX_TURNS} \
    generator.max_input_length=${MAX_INPUT_LENGTH} \
    generator.sampling_params.max_generate_length=${MAX_GEN_LENGTH} \
    generator.sampling_params.stop='[\"</search>\", \"</done>\"]' \
    generator.eval_sampling_params.temperature=${TEMPERATURE} \
    generator.eval_sampling_params.stop='[\"</search>\", \"</done>\"]' \
    generator.eval_n_samples_per_prompt=${NUM_SAMPLES} \
    trainer.eval_interval=1 \
    trainer.eval_batch_size=256 \
    trainer.max_prompt_length=${MAX_PROMPT_LENGTH} \
    environment.env_class=\"citation_prediction_v2\" \
    environment.skyrl_gym.max_env_workers=16 \
    environment.skyrl_gym.citation_prediction_v2.log_requests=false \
    environment.skyrl_gym.citation_prediction_v2.search_url=\"${RETRIEVER_URL}\" \
    environment.skyrl_gym.citation_prediction_v2.topk=${TOPK} \
    environment.skyrl_gym.citation_prediction_v2.max_predictions_ratio=2.0 \
    trainer.logger=\"wandb\" \
    trainer.project_name=\"skyrl-citation-prediction-v3\" \
    trainer.run_name=\"${RUN_NAME}\" \
    trainer.export_path=\"${SWEEP_DIR}\" \
    ${ROPE_SCALING}"

EVAL_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
uv run --active --frozen ray stop || true
srun --nodes=1 --ntasks=1 --nodelist=${EVAL_NODE} \
    /bin/bash -c "export VIRTUAL_ENV=/home/hk4638/SkyRL/skyrl-train/.venv && \
        export PATH=/home/hk4638/.local/bin:\$PATH && \
        uv run --active --frozen ray stop" 2>/dev/null || true
kill ${RETRIEVER_PID} 2>/dev/null || true

echo "=== Sweep eval complete ==="
echo "End time: $(date)"
echo "Exit code: ${EVAL_EXIT_CODE}"
echo "Results logged to WandB project: skyrl-citation-prediction-v3, run: ${RUN_NAME}"
echo "Trajectories dumped to: ${SWEEP_DIR}/dumped_evals/"

exit ${EVAL_EXIT_CODE}
