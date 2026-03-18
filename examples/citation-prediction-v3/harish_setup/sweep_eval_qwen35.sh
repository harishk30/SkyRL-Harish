#!/bin/bash
# ============================================================================
# Launch eval sweep for Qwen3.5 models on citation prediction.
#
# Submits TWO SLURM jobs per (m, n) combo (part0 + part1).
# Uses standalone_eval.py (no SkyRL/Ray) with .venv-eval-qwen35.
#
# Usage:
#   bash sweep_eval_qwen35.sh --model qwen3_5_9b
#   bash sweep_eval_qwen35.sh --model qwen3_5_4b
#   bash sweep_eval_qwen35.sh --model qwen3_5_35b_a3b
#   bash sweep_eval_qwen35.sh --model qwen3_5_9b --dry-run
#   bash sweep_eval_qwen35.sh --model qwen3_5_9b --m-values 4,6 --n-values 3,5
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="${SCRIPT_DIR}/eval_qwen35_job.slurm"

# ============================================================================
# Defaults
# ============================================================================
NUM_SAMPLES=20
TEMPERATURE="1.0"
SEED=42
EMBEDDING_MODEL="qwen3_4b"
MODEL_NAME=""
MAX_EXAMPLES=50  # per part (total = 2 * MAX_EXAMPLES)
NUM_PARTS=2
DRY_RUN=false

# Grid values (no m=15 — exceeds 65K budget with any n)
M_VALUES=(4 6 8 10)
N_VALUES=(3 5 10 20)

# Token budget for Qwen3.5 (all models use 65K context)
GEN_PER_TURN=4096
TOKEN_BUDGET=61440  # 65536 - 4096 (prompt)

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
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
            echo "Usage: bash sweep_eval_qwen35.sh --model MODEL [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --model MODEL      qwen3_5_9b | qwen3_5_4b | qwen3_5_35b_a3b"
            echo ""
            echo "Options:"
            echo "  --dry-run          Print sbatch commands without submitting"
            echo "  --k NUM            Samples per prompt (default: 20)"
            echo "  --temp FLOAT       Sampling temperature (default: 1.0)"
            echo "  --seed INT         Random seed (default: 42)"
            echo "  --embed MODEL      Embedding model (default: qwen3_4b)"
            echo "  --examples NUM     Prompts per part (default: 50, total = 2*N)"
            echo "  --m-values M1,M2   Override M grid (default: 4,6,8,10)"
            echo "  --n-values N1,N2   Override N grid (default: 3,5,10,20)"
            echo "  --parts NUM        Number of parts (default: 2)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: --model is required. Use qwen3_5_9b, qwen3_5_4b, or qwen3_5_35b_a3b."
    exit 1
fi

# Validate model name
case "$MODEL_NAME" in
    qwen3_5_9b|qwen3_5_4b|qwen3_5_35b_a3b) ;;
    *)
        echo "ERROR: Unknown model '${MODEL_NAME}'. Use qwen3_5_9b, qwen3_5_4b, or qwen3_5_35b_a3b."
        exit 1
        ;;
esac

# ============================================================================
# Create sampled + split parquets
# ============================================================================
DATA_DIR="/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/full"
TOTAL_EXAMPLES=$((MAX_EXAMPLES * NUM_PARTS))

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
    ~/.local/bin/uv run --active --frozen python -c "
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
echo "Qwen3.5 Eval Sweep Configuration"
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
    echo "  watch -n 10 'squeue -u \$USER | grep eval-qwen35'"
    echo ""
    echo "Results will be written to:"
    echo "  /scratch/gpfs/ZHUANGL/hk4638/logs/sweep/<model>_m{m}_n{n}_k{k}_part{p}/"
fi
