#!/bin/bash
# ============================================================================
# Standalone eval job for Qwen3.5 models on citation prediction.
#
# No SkyRL/Ray dependency — uses vLLM offline LLM class directly.
#
# Node 0: Retriever (4 GPUs for FAISS sharding)
# Node 1: vLLM eval (standalone_eval.py with new venv)
#
# Env vars (required):
#   MAX_TURNS        Number of search turns (m)
#   TOPK             Number of search results per turn (n)
#
# Env vars (optional):
#   NUM_SAMPLES      Samples per prompt (default: 20)
#   TEMPERATURE      Sampling temperature (default: 1.0)
#   SEED             Random seed for prompt sampling (default: 42)
#   MODEL_NAME       qwen3_5_35b_a3b | qwen3_5_9b | qwen3_5_4b (default: qwen3_5_9b)
#   EMBEDDING_MODEL  qwen3_4b | qwen3_06b (default: qwen3_4b)
#   MAX_EXAMPLES     Prompts per part (default: 50)
#   PART             Part index (default: "")
#
# Usage:
#   MAX_TURNS=4 TOPK=5 MODEL_NAME=qwen3_5_9b sbatch eval_qwen35_job.slurm
# ============================================================================

#SBATCH --job-name=eval-qwen35
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/eval-qwen35_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/eval-qwen35_%j.err
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

cd /home/hk4638/SkyRL

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRATCH_BASE="/scratch/gpfs/ZHUANGL/hk4638"
DATA_DIR="${SCRATCH_BASE}/data/citation_prediction_v3"
LOG_DIR="${SCRATCH_BASE}/logs"
SCRIPT_DIR="/home/hk4638/SkyRL/examples/citation-prediction-v3/harish_setup"

# Required env vars
if [ -z "$MAX_TURNS" ] || [ -z "$TOPK" ]; then
    echo "ERROR: MAX_TURNS and TOPK env vars are required"
    exit 1
fi

# Optional env vars with defaults
NUM_SAMPLES=${NUM_SAMPLES:-20}
TEMPERATURE=${TEMPERATURE:-"1.0"}
SEED=${SEED:-42}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_4b"}
MAX_EXAMPLES=${MAX_EXAMPLES:-50}
MODEL_NAME=${MODEL_NAME:-"qwen3_5_9b"}

# ============================================================================
# MODEL SELECTION
# ============================================================================
# NOTE: After running download_qwen35_models.slurm, update these snapshot hashes
# with the actual values from the download log.

if [ "$MODEL_NAME" = "qwen3_5_35b_a3b" ]; then
    MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/b1fc3d59ae0ab1e4279e04a8dd0fc4dc361fc2b6"
    TP=2
    MAX_MODEL_LEN=65536
    MAX_GEN_LENGTH=4096
    DIR_PREFIX="qwen35-35b"
elif [ "$MODEL_NAME" = "qwen3_5_9b" ]; then
    # TODO: Update snapshot hash after download
    MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/PLACEHOLDER"
    TP=2
    MAX_MODEL_LEN=65536
    MAX_GEN_LENGTH=4096
    DIR_PREFIX="qwen35-9b"
elif [ "$MODEL_NAME" = "qwen3_5_4b" ]; then
    MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    TP=1
    MAX_MODEL_LEN=65536
    MAX_GEN_LENGTH=4096
    DIR_PREFIX="qwen35-4b"
else
    echo "ERROR: Unknown MODEL_NAME=${MODEL_NAME}. Use qwen3_5_35b_a3b, qwen3_5_9b, or qwen3_5_4b."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    echo "Run download_qwen35_models.slurm first, then update snapshot hashes in this script."
    exit 1
fi

# Output directory
PART=${PART:-""}
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
echo "=== Qwen3.5 Standalone Eval: m=${MAX_TURNS}, n=${TOPK}, k=${NUM_SAMPLES} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Model: ${MODEL_NAME} (TP=${TP})"
echo "Model path: ${MODEL_PATH}"
echo "Start time: $(date)"
echo "Output: ${SWEEP_DIR}"

export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda/pkgs"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"
export HF_HOME="${SCRATCH_BASE}/huggingface"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_BASE}/huggingface/hub"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/huggingface/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load proxy/default
# Force working proxy IP
export http_proxy=http://172.28.15.7:8080
export https_proxy=http://172.28.15.7:8080
export HTTP_PROXY=http://172.28.15.7:8080
export HTTPS_PROXY=http://172.28.15.7:8080

# Load secrets
ENV_FILE="/home/hk4638/SkyRL/examples/search/harish_setup/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a && source "${ENV_FILE}" && set +a
fi

# Get node list
NODELIST=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
RETRIEVER_NODE=${NODELIST[0]}
EVAL_NODE=${NODELIST[1]}

echo "Retriever node: ${RETRIEVER_NODE}"
echo "Eval node: ${EVAL_NODE}"

export NO_PROXY="localhost,127.0.0.1,${RETRIEVER_NODE},${EVAL_NODE}"
export no_proxy="$NO_PROXY"

# ============================================================================
# DATA FILE
# ============================================================================
VARIANT_DIR="${DATA_DIR}/full"
if [ -n "$PART" ]; then
    DATA_FILE="${VARIANT_DIR}/train_sample${MAX_EXAMPLES}_seed${SEED}_part${PART}.parquet"
else
    DATA_FILE="${VARIANT_DIR}/train_sample${MAX_EXAMPLES}_seed${SEED}.parquet"
fi
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Sampled parquet not found at ${DATA_FILE}"
    echo "Run sweep_eval_qwen35.sh first to create it."
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
    echo "ERROR: Unknown EMBEDDING_MODEL=${EMBEDDING_MODEL}."
    exit 1
fi

if [ ! -f "${INDEX_FILE}" ]; then
    echo "ERROR: FAISS index not found at ${INDEX_FILE}"
    exit 1
fi

echo "Embedding: ${EMBEDDING_MODEL}"
echo "Index: ${INDEX_FILE}"

# ============================================================================
# START RETRIEVER (Node 0)
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

        echo 'Starting retriever on \$(hostname) with 4 GPUs...'
        python /home/hk4638/SkyRL/examples/citation_prediction/retriever/retrieval_server.py \
            --index_path ${INDEX_FILE} \
            --corpus_path ${CORPUS_FILE} \
            --topk ${TOPK} \
            --retriever_name ${RETRIEVER_NAME} \
            --retriever_model ${RETRIEVER_MODEL_PATH} \
            --faiss_gpu
    " > "${LOG_DIR}/retriever_qwen35_${SLURM_JOB_ID}.log" 2>&1 &

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
    cat "${LOG_DIR}/retriever_qwen35_${SLURM_JOB_ID}.log"
    exit 1
fi

# ============================================================================
# RUN EVAL (Node 1) — standalone, no Ray
# ============================================================================
echo "=== Starting eval on ${EVAL_NODE} ==="
echo "Model: ${MODEL_PATH} (TP=${TP})"
echo "Retriever: ${RETRIEVER_URL}"
echo "Params: m=${MAX_TURNS}, n=${TOPK}, k=${NUM_SAMPLES}, temp=${TEMPERATURE}"

EVAL_VENV="/home/hk4638/SkyRL/.venv-eval-qwen35"

srun --nodes=1 --ntasks=1 --nodelist=${EVAL_NODE} \
    /bin/bash -c "
        export HOME=/home/hk4638
        export NO_PROXY=localhost,127.0.0.1,${RETRIEVER_NODE},${EVAL_NODE}
        export no_proxy=\$NO_PROXY
        export WANDB_API_KEY=${WANDB_API_KEY}
        export http_proxy=${http_proxy}
        export https_proxy=${https_proxy}
        export HTTP_PROXY=${HTTP_PROXY}
        export HTTPS_PROXY=${HTTPS_PROXY}
        export HF_HOME=${HF_HOME}
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export TRANSFORMERS_CACHE=${HF_HOME}

        echo 'Activating eval venv at ${EVAL_VENV}...'
        source ${EVAL_VENV}/bin/activate

        echo 'Python: \$(which python)'
        echo 'vLLM version: \$(python -c \"import vllm; print(vllm.__version__)\")'

        python ${SCRIPT_DIR}/standalone_eval.py \
            --model_path ${MODEL_PATH} \
            --data_path ${DATA_FILE} \
            --retriever_url ${RETRIEVER_URL} \
            --max_turns ${MAX_TURNS} \
            --topk ${TOPK} \
            --n_samples ${NUM_SAMPLES} \
            --temperature ${TEMPERATURE} \
            --max_gen_length ${MAX_GEN_LENGTH} \
            --max_model_len ${MAX_MODEL_LEN} \
            --tp ${TP} \
            --output_dir ${SWEEP_DIR} \
            --wandb_project skyrl-citation-prediction-v3 \
            --run_name ${RUN_NAME}
    "

EVAL_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================
echo "=== Cleaning up ==="
kill ${RETRIEVER_PID} 2>/dev/null || true

echo "=== Eval complete ==="
echo "End time: $(date)"
echo "Exit code: ${EVAL_EXIT_CODE}"
echo "Results: ${SWEEP_DIR}/"

exit ${EVAL_EXIT_CODE}
