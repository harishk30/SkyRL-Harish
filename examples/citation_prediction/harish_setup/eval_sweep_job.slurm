#!/bin/bash
# ============================================================================
# Single sweep eval job using SkyRL's main_generate pipeline.
#
# Evaluates one (m, n) hyperparameter combo with pass@k sampling.
# Same architecture as eval_base_model.slurm but parameterized for sweeps.
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
#   PROMPT_STYLE     short | extended (default: short)
#   TEMPERATURE      Sampling temperature (default: 1.0)
#   SEED             Random seed for prompt sampling (default: 42)
#   EMBEDDING_MODEL  qwen3_4b | qwen3_06b (default: qwen3_4b)
#   MAX_EXAMPLES     Number of prompts to evaluate (default: 100)
#
# Usage:
#   MAX_TURNS=4 TOPK=3 sbatch eval_sweep_job.slurm
#   MAX_TURNS=2 TOPK=5 NUM_SAMPLES=10 sbatch eval_sweep_job.slurm
# ============================================================================

#SBATCH --job-name=sweep-eval
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/sweep-eval_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/sweep-eval_%j.err
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

# Set working directory explicitly (srun inherits cwd; if sbatch was submitted
# from a local-only path like /tmp, remote nodes can't resolve it)
cd /home/hk4638/SkyRL

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction"
LOG_DIR="${SCRATCH_BASE}/logs"

# Required env vars
if [ -z "$MAX_TURNS" ] || [ -z "$TOPK" ]; then
    echo "ERROR: MAX_TURNS and TOPK env vars are required"
    echo "Usage: MAX_TURNS=4 TOPK=3 sbatch eval_sweep_job.slurm"
    exit 1
fi

# Optional env vars with defaults
NUM_SAMPLES=${NUM_SAMPLES:-20}
PROMPT_STYLE=${PROMPT_STYLE:-"short"}
TEMPERATURE=${TEMPERATURE:-"1.0"}
SEED=${SEED:-42}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_4b"}
MAX_EXAMPLES=${MAX_EXAMPLES:-100}

# Output directory — SkyRL writes dumped_evals/ inside this path
SWEEP_DIR="${LOG_DIR}/sweep/${PROMPT_STYLE}_m${MAX_TURNS}_n${TOPK}_k${NUM_SAMPLES}"
RUN_NAME="sweep-${PROMPT_STYLE}-m${MAX_TURNS}-n${TOPK}-k${NUM_SAMPLES}-seed${SEED}"

mkdir -p "${SWEEP_DIR}" "${LOG_DIR}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=== Sweep eval: m=${MAX_TURNS}, n=${TOPK}, k=${NUM_SAMPLES} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Start time: $(date)"
echo "Config: prompt=${PROMPT_STYLE}, temp=${TEMPERATURE}, seed=${SEED}, embed=${EMBEDDING_MODEL}"
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

export VIRTUAL_ENV="/home/hk4638/SkyRL/.venv"

MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

module load proxy/default

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK

# Load secrets for WandB
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
EVAL_NODE=${NODELIST[1]}

echo "Retriever node: ${RETRIEVER_NODE}"
echo "Eval node: ${EVAL_NODE}"

export NO_PROXY="localhost,127.0.0.1,${RETRIEVER_NODE},${EVAL_NODE}"
export no_proxy="$NO_PROXY"
echo "NO_PROXY set to: $NO_PROXY"

# ============================================================================
# DATA FILE — pre-sampled parquet created by sweep_eval.sh
# ============================================================================
DATA_FILE="${DATA_DIR}/${PROMPT_STYLE}/train_sample${MAX_EXAMPLES}_seed${SEED}.parquet"
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Sampled parquet not found at ${DATA_FILE}"
    echo "Run sweep_eval.sh first to create it."
    exit 1
fi
echo "Data: ${DATA_FILE}"

# ============================================================================
# RETRIEVER CONFIGURATION
# ============================================================================
CORPUS_FILE="${DATA_DIR}/arxiv_wikiformat_with_ids.jsonl"

if [ "$EMBEDDING_MODEL" = "qwen3_4b" ]; then
    RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
    INDEX_FILE="${DATA_DIR}/qwen3_4b_embed/qwen3_Flat.index"
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
        python /home/hk4638/SkyRL/examples/citation_prediction/retriever/retrieval_server.py \
            --index_path ${INDEX_FILE} \
            --corpus_path ${CORPUS_FILE} \
            --topk ${TOPK} \
            --retriever_name ${RETRIEVER_NAME} \
            --retriever_model ${RETRIEVER_MODEL_PATH} \
            --faiss_gpu
    " > "${LOG_DIR}/retriever_sweep_${SLURM_JOB_ID}.log" 2>&1 &

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
    cat "${LOG_DIR}/retriever_sweep_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# START RAY ON EVAL NODE
# ============================================================================
echo "=== Starting Ray on ${EVAL_NODE} ==="

EVAL_NODE_IP=$(srun --nodes=1 --ntasks=1 --nodelist=${EVAL_NODE} /usr/bin/hostname -i)
RAY_PORT=6379

# Start Ray head on eval node (daemon mode — no sleep infinity)
srun --nodes=1 --ntasks=1 --nodelist=${EVAL_NODE} \
    /bin/bash -c "cd /home/hk4638/SkyRL && \
        export UV_CACHE_DIR=${SCRATCH_BASE}/uv_cache && \
        export UV_LINK_MODE=copy && \
        export VIRTUAL_ENV=/home/hk4638/SkyRL/.venv && \
        export PATH=/home/hk4638/.local/bin:\$PATH && \
        export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 && \
        uv run --active --frozen ray start --head --port=${RAY_PORT} --num-gpus=4"
sleep 5

# Join Ray cluster from retriever node with 0 GPUs
echo "Joining Ray cluster from ${RETRIEVER_NODE} with 0 GPUs"
uv run --active --frozen ray start --address=${EVAL_NODE_IP}:${RAY_PORT} --num-gpus=0
sleep 5

echo "Ray cluster started on head=${EVAL_NODE_IP}:${RAY_PORT}"
uv run --active --frozen ray status

# ============================================================================
# RUN SKYRL EVAL
# ============================================================================
echo "=== Running SkyRL sweep evaluation ==="
echo "Model: ${MODEL_PATH}"
echo "Prompt style: ${PROMPT_STYLE}"
echo "Data: ${DATA_FILE}"
echo "Retriever: ${RETRIEVER_URL}"
echo "Params: m=${MAX_TURNS}, n=${TOPK}, k=${NUM_SAMPLES}, temp=${TEMPERATURE}"

export RAY_ADDRESS="${EVAL_NODE_IP}:${RAY_PORT}"
echo "Connecting to Ray cluster at RAY_ADDRESS=${RAY_ADDRESS}"

# SkyRL eval with sweep parameters
# Same config as eval_base_model.slurm but with parameterized hyperparameters:
#   - generator.max_turns = MAX_TURNS (sweep over m)
#   - environment.skyrl_gym.citation_prediction.topk = TOPK (sweep over n)
#   - generator.eval_n_samples_per_prompt = NUM_SAMPLES (pass@k)
#   - generator.eval_sampling_params.temperature = TEMPERATURE (stochastic sampling)

# For single-turn (m=1), SkyRL requires max_input_length == max_prompt_length
if [ "${MAX_TURNS}" -eq 1 ]; then
    MAX_PROMPT_LENGTH=30000
else
    MAX_PROMPT_LENGTH=2048
fi

uv run --active --frozen --extra fsdp -m skyrl.train.entrypoints.main_generate \
    data.val_data="['${DATA_FILE}']" \
    trainer.policy.model.path="${MODEL_PATH}" \
    trainer.strategy=fsdp2 \
    trainer.policy.use_liger_kernel=true \
    trainer.ref.use_liger_kernel=true \
    trainer.placement.colocate_all=false \
    generator.inference_engine.num_engines=2 \
    generator.inference_engine.tensor_parallel_size=2 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.gpu_memory_utilization=0.9 \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.max_turns=${MAX_TURNS} \
    generator.max_input_length=30000 \
    generator.sampling_params.max_generate_length=1024 \
    generator.sampling_params.stop='["</search>", "</answer>"]' \
    generator.eval_sampling_params.temperature=${TEMPERATURE} \
    generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
    generator.eval_n_samples_per_prompt=${NUM_SAMPLES} \
    trainer.eval_interval=1 \
    trainer.eval_batch_size=256 \
    trainer.max_prompt_length=${MAX_PROMPT_LENGTH} \
    environment.env_class="citation_prediction" \
    environment.skyrl_gym.max_env_workers=16 \
    environment.skyrl_gym.citation_prediction.log_requests=false \
    environment.skyrl_gym.citation_prediction.search_url="${RETRIEVER_URL}" \
    environment.skyrl_gym.citation_prediction.topk=${TOPK} \
    trainer.logger="wandb" \
    trainer.project_name="skyrl-citation-prediction" \
    trainer.run_name="${RUN_NAME}" \
    trainer.export_path="${SWEEP_DIR}"

EVAL_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
# Stop Ray on local node
uv run --active --frozen ray stop || true
# Stop Ray on eval node
srun --nodes=1 --ntasks=1 --nodelist=${EVAL_NODE} \
    /bin/bash -c "export VIRTUAL_ENV=/home/hk4638/SkyRL/.venv && \
        export PATH=/home/hk4638/.local/bin:\$PATH && \
        uv run --active --frozen ray stop" 2>/dev/null || true
kill ${RETRIEVER_PID} 2>/dev/null || true

echo "=== Sweep eval complete ==="
echo "End time: $(date)"
echo "Exit code: ${EVAL_EXIT_CODE}"
echo "Results logged to WandB project: skyrl-citation-prediction, run: ${RUN_NAME}"
echo "Trajectories dumped to: ${SWEEP_DIR}/dumped_evals/"

exit ${EVAL_EXIT_CODE}
