#!/bin/bash
# ============================================================================
# Launch IsoCompute-guided sweep on AI Lab H200s
#
# 3 job arrays + 1 standalone job:
#
# Array 1 (Bp=128, LR=1e-6 fixed): n=10, n=20, n=40
# Array 2 (Bp=256, LR=sqrt scaled): n=10, n=20
# Array 3 (Bp=512, LR=sqrt scaled): n=10
# Standalone: Bp=128, n=10, LR=2e-6 (manual override)
#
# Usage:
#   bash launch_ailab_sweep.sh          # submit all
#   bash launch_ailab_sweep.sh --dry-run # show commands without submitting
# ============================================================================

set -e
cd "$(dirname "$0")"

SCRIPT="train_citation_prediction_4b_ailab.slurm"
DRY_RUN=false
[ "$1" = "--dry-run" ] && DRY_RUN=true

submit() {
    local desc="$1"
    shift
    echo "=== ${desc} ==="
    echo "  sbatch $@"
    if [ "$DRY_RUN" = "false" ]; then
        sbatch "$@"
    fi
    echo ""
}

# Array 1: Bp=128, n={10, 20, 40}, LR=1e-6 (fixed, not auto-scaled)
submit "Bp=128, LR=1e-6: n=10, n=20, n=40" \
    --export=ALL,SWEEP_BP=128,SWEEP_N_LIST="10 20 40",SWEEP_LR_OVERRIDE=1.0e-6 \
    --array=0-2%1 \
    ${SCRIPT}

# Array 2: Bp=256, n={10, 20}, LR=auto (sqrt scaled)
submit "Bp=256, LR=auto: n=10, n=20" \
    --export=ALL,SWEEP_BP=256,SWEEP_N_LIST="10 20" \
    --array=0-1%1 \
    ${SCRIPT}

# Array 3: Bp=512, n=10, LR=auto (sqrt scaled)
submit "Bp=512, LR=auto: n=10" \
    --export=ALL,SWEEP_BP=512,SWEEP_N=10 \
    ${SCRIPT}

# Standalone: Bp=128, n=10, LR=2e-6 (manual override for LR comparison)
submit "Bp=128, n=10, LR=2e-6 (standalone)" \
    --export=ALL,SWEEP_BP=128,SWEEP_N=10,SWEEP_LR_OVERRIDE=2.0e-6 \
    ${SCRIPT}

echo "Done. Monitor with: squeue -u \$USER -p ailab"
