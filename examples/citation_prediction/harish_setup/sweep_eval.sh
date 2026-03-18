#!/bin/bash
# ============================================================================
# Launch hyperparameter sweep for base model evaluation.
#
# Submits one SLURM job per (m, n) combination. Each job is 2-node:
# node 0 = retriever (4 GPUs) + Ray worker + runs main_generate
# node 1 = Ray head with vLLM inference (4 GPUs)
#
# Phase 1 (default): Sweep m x n with prompt=short, k=20, temp=1.0
# Phase 2 (later):   Roll back k on best (m, n)
# Phase 3 (later):   Try different prompts on best (m, n, k)
#
# Usage:
#   bash sweep_eval.sh                    # Submit all phase 1 jobs
#   bash sweep_eval.sh --dry-run          # Print commands without submitting
#   bash sweep_eval.sh --prompt extended  # Use extended prompt
#   bash sweep_eval.sh --k 10            # Override num_samples
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="${SCRIPT_DIR}/eval_sweep_job.slurm"

# ============================================================================
# Defaults
# ============================================================================
PROMPT_STYLE="short"
NUM_SAMPLES=20
TEMPERATURE="1.0"
SEED=42
EMBEDDING_MODEL="qwen3_4b"
MAX_EXAMPLES=100
DRY_RUN=false

# Grid values
M_VALUES=(2 4 6 8)
N_VALUES=(1 3 5)

# Token budget: m * (1024 + 600*n) + 2048 < 30000
# => m * (1024 + 600*n) < 27952
TOKEN_BUDGET=27952

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --prompt)
            PROMPT_STYLE="$2"
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
        -h|--help)
            echo "Usage: bash sweep_eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Print sbatch commands without submitting"
            echo "  --prompt STYLE     Prompt style: short (default) or extended"
            echo "  --k NUM            Number of samples per prompt (default: 20)"
            echo "  --temp FLOAT       Sampling temperature (default: 1.0)"
            echo "  --seed INT         Random seed (default: 42)"
            echo "  --embed MODEL      Embedding model: qwen3_4b (default) or qwen3_06b"
            echo "  --examples NUM     Number of prompts to evaluate (default: 100)"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Create sampled parquet (SkyRL evaluates all rows, so we pre-sample)
# ============================================================================
DATA_DIR="/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction"
SAMPLE_FILE="${DATA_DIR}/${PROMPT_STYLE}/train_sample${MAX_EXAMPLES}_seed${SEED}.parquet"

if [ ! -f "${SAMPLE_FILE}" ]; then
    echo "Creating sampled parquet: ${SAMPLE_FILE}"
    cd /home/hk4638/SkyRL
    export VIRTUAL_ENV="/home/hk4638/SkyRL/.venv"
    export UV_CACHE_DIR="/scratch/gpfs/ZHUANGL/hk4638/uv_cache"
    export UV_LINK_MODE=copy
    uv run --active --frozen python -c "
import pandas as pd
df = pd.read_parquet('${DATA_DIR}/${PROMPT_STYLE}/train.parquet')
sample = df.sample(n=min(${MAX_EXAMPLES}, len(df)), random_state=${SEED})
sample.to_parquet('${SAMPLE_FILE}')
print(f'Created: {len(sample)} examples sampled from {len(df)} total')
"
    echo ""
else
    echo "Sampled parquet already exists: ${SAMPLE_FILE}"
    echo ""
fi

# ============================================================================
# Submit jobs
# ============================================================================
echo "============================================"
echo "Sweep Configuration"
echo "============================================"
echo "Prompt:       ${PROMPT_STYLE}"
echo "Num samples:  ${NUM_SAMPLES}"
echo "Temperature:  ${TEMPERATURE}"
echo "Seed:         ${SEED}"
echo "Embedding:    ${EMBEDDING_MODEL}"
echo "Max examples: ${MAX_EXAMPLES}"
echo "M values:     ${M_VALUES[*]}"
echo "N values:     ${N_VALUES[*]}"
echo "Dry run:      ${DRY_RUN}"
echo "============================================"
echo ""

SUBMITTED=0
SKIPPED=0
JOB_IDS=()

for m in "${M_VALUES[@]}"; do
    for n in "${N_VALUES[@]}"; do
        # Check token budget
        tokens_per_turn=$((1024 + 600 * n))
        total_tokens=$((m * tokens_per_turn))

        if [ $total_tokens -gt $TOKEN_BUDGET ]; then
            echo "SKIP: m=${m}, n=${n} (est. ${total_tokens} tokens > ${TOKEN_BUDGET} budget)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo -n "m=${m}, n=${n} (est. ${total_tokens} tokens): "

        EXPORT_VARS="MAX_TURNS=${m},TOPK=${n},NUM_SAMPLES=${NUM_SAMPLES},PROMPT_STYLE=${PROMPT_STYLE},TEMPERATURE=${TEMPERATURE},SEED=${SEED},EMBEDDING_MODEL=${EMBEDDING_MODEL},MAX_EXAMPLES=${MAX_EXAMPLES}"

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

echo ""
echo "============================================"
echo "Summary"
echo "============================================"
echo "Submitted: ${SUBMITTED} jobs"
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
    echo "  /scratch/gpfs/ZHUANGL/hk4638/logs/sweep/{prompt}_m{m}_n{n}_k{k}/dumped_evals/"
fi
