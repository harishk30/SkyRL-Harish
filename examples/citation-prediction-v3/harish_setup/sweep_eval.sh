#!/bin/bash
# ============================================================================
# Launch hyperparameter sweep for V3 base model evaluation.
#
# Submits TWO SLURM jobs per (m, n) combination (part0 + part1), each
# evaluating half the prompts. Each job is 2-node:
# node 0 = retriever (4 GPUs) + Ray worker + runs main_generate
# node 1 = Ray head with vLLM inference (4 GPUs)
#
# Usage:
#   bash sweep_eval.sh                    # Submit all jobs
#   bash sweep_eval.sh --dry-run          # Print commands without submitting
#   bash sweep_eval.sh --k 10             # Override num_samples
#   bash sweep_eval.sh --examples 100     # Total prompts (split into 2 parts)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="${SCRIPT_DIR}/eval_sweep_job.slurm"

# ============================================================================
# Defaults
# ============================================================================
NUM_SAMPLES=20
TEMPERATURE="1.0"
SEED=42
EMBEDDING_MODEL="qwen3_4b"
MODEL_NAME="qwen3_4b"
MAX_EXAMPLES=50  # per part (total = 2 * MAX_EXAMPLES)
NUM_PARTS=2
DRY_RUN=false

# Grid values
M_VALUES=(4 6 8 10 15)
N_VALUES=(3 5 10 20)

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --k)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --temp)
            TEMPERATURE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --embed)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --m-values)
            IFS=',' read -ra M_VALUES <<< "$2"
            shift 2
            ;;
        --n-values)
            IFS=',' read -ra N_VALUES <<< "$2"
            shift 2
            ;;
        --parts)
            NUM_PARTS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash sweep_eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Print sbatch commands without submitting"
            echo "  --k NUM            Number of samples per prompt (default: 20)"
            echo "  --temp FLOAT       Sampling temperature (default: 1.0)"
            echo "  --seed INT         Random seed (default: 42)"
            echo "  --embed MODEL      Embedding model: qwen3_4b (default) or qwen3_06b"
            echo "  --model MODEL      LLM model: qwen3_4b (default) or qwen3_4b_thinking"
            echo "  --examples NUM     Number of prompts per part (default: 50, total = 2*N)"
            echo "  --parts NUM        Number of parts to split into (default: 2)"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set token budget based on model
if [ "$MODEL_NAME" = "qwen3_4b_thinking" ]; then
    GEN_PER_TURN=8192
    TOKEN_BUDGET=155648  # 163840 - 4096 (prompt) - 4096 (margin)
elif [ "$MODEL_NAME" = "qwen3_4b_yarn" ]; then
    GEN_PER_TURN=1024
    TOKEN_BUDGET=126976  # 131072 - 4096 (prompt)
elif [ "$MODEL_NAME" = "qwen3_5_35b_a3b" ]; then
    GEN_PER_TURN=4096
    TOKEN_BUDGET=61440  # 65536 - 4096 (prompt). Thinking content stays in context (single-turn mode)
else
    GEN_PER_TURN=1024
    TOKEN_BUDGET=27904  # 32000 - 4096 (prompt)
fi

# ============================================================================
# Create sampled + split parquets (SkyRL evaluates all rows, so we pre-sample)
# ============================================================================
DATA_DIR="/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/full"
TOTAL_EXAMPLES=$((MAX_EXAMPLES * NUM_PARTS))

# Check if all part files exist
ALL_PARTS_EXIST=true
for ((p=0; p<NUM_PARTS; p++)); do
    PART_FILE="${DATA_DIR}/train_sample${MAX_EXAMPLES}_seed${SEED}_part${p}.parquet"
    if [ ! -f "${PART_FILE}" ]; then
        ALL_PARTS_EXIST=false
        break
    fi
done

if [ "$ALL_PARTS_EXIST" = false ]; then
    echo "Creating ${NUM_PARTS} part files (${MAX_EXAMPLES} examples each, ${TOTAL_EXAMPLES} total)..."
    cd /home/hk4638/SkyRL
    export VIRTUAL_ENV="/home/hk4638/SkyRL/.venv"
    export UV_CACHE_DIR="/scratch/gpfs/ZHUANGL/hk4638/uv_cache"
    export UV_LINK_MODE=copy
    uv run --active --frozen python -c "
import pandas as pd
import numpy as np
df = pd.read_parquet('${DATA_DIR}/train.parquet')
total = min(${TOTAL_EXAMPLES}, len(df))
sample = df.sample(n=total, random_state=${SEED})
parts = np.array_split(sample, ${NUM_PARTS})
for i, part in enumerate(parts):
    path = '${DATA_DIR}/train_sample${MAX_EXAMPLES}_seed${SEED}_part' + str(i) + '.parquet'
    part.to_parquet(path, index=False)
    print(f'  Part {i}: {len(part)} examples -> {path}')
print(f'Total: {total} examples from {len(df)} sampled, split into ${NUM_PARTS} parts')
"
    echo ""
else
    echo "Part files already exist:"
    for ((p=0; p<NUM_PARTS; p++)); do
        echo "  ${DATA_DIR}/train_sample${MAX_EXAMPLES}_seed${SEED}_part${p}.parquet"
    done
    echo ""
fi

# ============================================================================
# Submit jobs
# ============================================================================
echo "============================================"
echo "V3 Sweep Configuration"
echo "============================================"
echo "Model:        ${MODEL_NAME}"
echo "Num samples:  ${NUM_SAMPLES}"
echo "Temperature:  ${TEMPERATURE}"
echo "Seed:         ${SEED}"
echo "Embedding:    ${EMBEDDING_MODEL}"
echo "Examples:     ${MAX_EXAMPLES} per part x ${NUM_PARTS} parts = ${TOTAL_EXAMPLES} total"
echo "M values:     ${M_VALUES[*]}"
echo "N values:     ${N_VALUES[*]}"
echo "Token budget: ${TOKEN_BUDGET} (gen/turn=${GEN_PER_TURN})"
echo "Dry run:      ${DRY_RUN}"
echo "============================================"
echo ""

SUBMITTED=0
SKIPPED=0
JOB_IDS=()

for m in "${M_VALUES[@]}"; do
    for n in "${N_VALUES[@]}"; do
        # Check token budget
        tokens_per_turn=$((GEN_PER_TURN + 300 * n))
        total_tokens=$((m * tokens_per_turn))

        if [ $total_tokens -gt $TOKEN_BUDGET ]; then
            echo "SKIP: m=${m}, n=${n} (est. ${total_tokens} tokens > ${TOKEN_BUDGET} budget)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Submit one job per part
        for ((p=0; p<NUM_PARTS; p++)); do
            echo -n "m=${m}, n=${n}, part${p} (est. ${total_tokens} tokens): "

            EXPORT_VARS="MAX_TURNS=${m},TOPK=${n},NUM_SAMPLES=${NUM_SAMPLES},TEMPERATURE=${TEMPERATURE},SEED=${SEED},EMBEDDING_MODEL=${EMBEDDING_MODEL},MAX_EXAMPLES=${MAX_EXAMPLES},MODEL_NAME=${MODEL_NAME},PART=${p}"

            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] sbatch --export=${EXPORT_VARS} ${SLURM_TEMPLATE}"
            else
                JOB_OUTPUT=$(sbatch --export="${EXPORT_VARS}" "${SLURM_TEMPLATE}" 2>&1)
                JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
                JOB_IDS+=("${JOB_ID}")
                echo "submitted (job ${JOB_ID})"
            fi

            SUBMITTED=$((SUBMITTED + 1))
        done
    done
done

echo ""
echo "============================================"
echo "Summary"
echo "============================================"
echo "Submitted: ${SUBMITTED} jobs (${NUM_PARTS} parts per combo)"
echo "Skipped:   ${SKIPPED} combos (over token budget)"

if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  watch -n 10 'squeue -u \$USER | grep sweep'"
    echo ""
    echo "Results will be written to:"
    echo "  /scratch/gpfs/ZHUANGL/hk4638/logs/sweep/v3_m{m}_n{n}_k{k}_part{p}/dumped_evals/"
fi
