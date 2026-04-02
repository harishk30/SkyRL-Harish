# Citation Prediction V3 Run Summary (Apr 1, 2026)

This is a point-in-time summary of the current GRPO training jobs, recent job history, the SFT warm-start models, and the `m=4, n=10` filtered eval jobs.

## 1. Current GRPO Training Jobs

These are the AI Lab H200 jobs currently active from `harish_setup/train_citation_prediction_4b_ailab.slurm`.

### Running now

| Slurm job | Status | Config | Run name | Checkpoint dir | Log |
|---|---|---|---|---|---|
| `6310252_0` | RUNNING | `Bp=128, n=20, LR=1e-6, KL=false` | `cit-v3-4b-h200-bp128_n20_lr1e-06_nokl` | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/cit-v3-4b-h200-bp128_n20_lr1e-06_nokl` | `/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_6310252_0.out` |
| `6310253` | RUNNING | `Bp=256, n=20, LR=2.236068e-6, KL=false` | `cit-v3-4b-h200-bp256_n20_lr2e-06_nokl` | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/cit-v3-4b-h200-bp256_n20_lr2e-06_nokl` | `/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_6310253_4294967294.out` |
| `6310254` | RUNNING | `Bp=128, n=10, LR=2e-6, KL=false` | `cit-v3-4b-h200-bp128_n10_lr2e-06_nokl` | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/cit-v3-4b-h200-bp128_n10_lr2e-06_nokl` | `/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_6310254_4294967294.out` |

### Submitted but not started

| Slurm job | Status | Expected config | Log |
|---|---|---|---|
| `6310252_1` | PENDING | `Bp=128, n=40, LR=1e-6, KL=false` | `/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_6310252_1.out` |

This comes from:

```bash
sbatch --export=ALL,SWEEP_BP=128,SWEEP_N_LIST="20 40",SWEEP_LR_OVERRIDE=1.0e-6 --array=0-1%1 harish_setup/train_citation_prediction_4b_ailab.slurm
```

### Sweep intent

From `harish_setup/launch_ailab_sweep.sh`, the intended AI Lab sweep is:

- `Bp=128`, fixed `LR=1e-6`, `n in {10, 20, 40}`
- `Bp=256`, auto LR, `n in {10, 20}`
- `Bp=512`, auto LR, `n=10`
- standalone LR comparison: `Bp=128, n=10, LR=2e-6`

As of this snapshot, the jobs actually running or queued are:

- `Bp=128, n=20, LR=1e-6`
- `Bp=128, n=40, LR=1e-6` pending
- `Bp=256, n=20, LR=2.236068e-6`
- `Bp=128, n=10, LR=2e-6`

## 2. Recent GRPO Training History

Recent `cit-pred-v3-4b` job history from Slurm since Mar 28:

### Early short failures

- `6105673_*`, `6105674_*`, `6105675`, `6105676` on Mar 28: failed after `~2 min` to `~4 hr`
- `6205250_*`, `6205251_0`, `6205252`, `6205253` on Mar 29: failed or cancelled quickly

### Longer runs that reached the walltime

- `6211972_0`: `TIMEOUT` after `17:05:18`
- `6211976_0`: `TIMEOUT` after `17:05:17`
- `6211979`: `TIMEOUT` after `17:05:05`

### Interrupted/resubmitted runs

- `6211972_1`, `6211976_1`: cancelled on Mar 31
- `6211978`: cancelled before start

The current active block is:

- `6310252_0`, `6310253`, `6310254` running
- `6310252_1` pending

## 3. SFT Warm-Start Models

The successful SFT jobs came from `harish_setup/train_citation_prediction_sft_ailab.slurm` on Apr 1.

### Successful SFT jobs

| Slurm job | LR | Prep dir | Checkpoint root | Final checkpoint used for eval |
|---|---|---|---|---|
| `6309994_0` | `2e-6` | `/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/sft_claude_high_recall/p318_e1_lr2em6` | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/citation_prediction_v3_sft/p318_e1_lr2em6` | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/citation_prediction_v3_sft/p318_e1_lr2em6/step_143/policy` |
| `6309994_1` | `3e-6` | `/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/sft_claude_high_recall/p318_e1_lr3em6` | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/citation_prediction_v3_sft/p318_e1_lr3em6` | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/citation_prediction_v3_sft/p318_e1_lr3em6/step_143/policy` |
| `6309994_2` | `5e-6` | `/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/sft_claude_high_recall/p318_e1_lr5em6` | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/citation_prediction_v3_sft/p318_e1_lr5em6` | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/citation_prediction_v3_sft/p318_e1_lr5em6/step_143/policy` |

### Shared SFT data stats

All three successful SFT jobs used the same dataset prep settings:

- `MAX_PROMPTS=318`
- `EPOCHS=1`
- `MAX_SEQUENCE_LENGTH=32768`
- `prepared_sft_count=286`
- `dropped_for_length_count=32`
- `filtered_rl_train_count=2569`
- `original_rl_train_count=2855`

### Earlier failed SFT attempts

- `6206263_0` on Mar 29 failed quickly
- `6211977_0`, `6211977_1`, `6211977_2` on Mar 31 failed quickly

## 4. Filtered Eval Jobs (`m=4, n=10`)

These are the `sft_filtered` eval jobs using `harish_setup/eval_sweep_job.slurm` with:

- `MAX_TURNS=4`
- `TOPK=10`
- `NUM_SAMPLES=20`
- `PROMPT_VARIANT=sft_filtered`

### Completed jobs

| Slurm job | Model | Part | Output dir | Avg score | Pass@20 |
|---|---|---|---|---|---|
| `6349893` | `lr=2e-6` | `part0` | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr2em6-sft_filtered_m4_n10_k20_part0` | `0.075758` | `0.58` |
| `6349898` | `lr=3e-6` | `part1` | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr3em6-sft_filtered_m4_n10_k20_part1` | `0.054569` | `0.48` |
| `6349897` | `lr=5e-6` | `part0` | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr5em6-sft_filtered_m4_n10_k20_part0` | `0.073317` | `0.60` |
| `6349896` | `lr=5e-6` | `part1` | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr5em6-sft_filtered_m4_n10_k20_part1` | `0.051700` | `0.46` |

Current combined result for the only fully finished LR (`5e-6`):

- `avg_score = 0.062508`
- `pass@20 = 0.53`

### Pending jobs

| Slurm job | Model | Part | Current scheduled start | Expected output dir |
|---|---|---|---|---|
| `6349903` | `lr=3e-6` | `part0` | `2026-04-01 17:21 EDT` | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr3em6-sft_filtered_m4_n10_k20_part0` |
| `6349904` | `lr=2e-6` | `part1` | `2026-04-01 17:48 EDT` | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr2em6-sft_filtered_m4_n10_k20_part1` |

These are backfill estimates and may move.

## 5. How To Check Jobs

### Live queue status

```bash
squeue -u "$USER" -o "%.18i %.20j %.10T %.10P %.20S %.20M %.30R"
```

For just the jobs we care about:

```bash
squeue -u "$USER" | rg "cit-pred-v3-4b|cit-v3-sft|sweep-eval-v3"
```

### Recent accounting history

```bash
sacct -u "$USER" \
  --format=JobID,JobName%30,Partition,State,Start,End,Elapsed \
  -S 2026-03-28 | rg "cit-pred-v3-4b|cit-v3-sft|sweep-eval-v3"
```

### Inspect a specific Slurm job definition

This is useful for seeing the exact `sbatch --export=...` line:

```bash
scontrol show jobid -d 6310253
scontrol show jobid -d 6349903
```

### Tail training logs

```bash
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_6310252_0.out
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_6310253_4294967294.out
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_6310254_4294967294.out
```

### Tail SFT logs

```bash
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/cit-v3-sft_6309994_0.out
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/cit-v3-sft_6309994_1.out
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/cit-v3-sft_6309994_2.out
```

### Check filtered eval metrics

Each completed eval writes:

- `dumped_evals/eval_only/aggregated_results.jsonl`
- `dumped_evals/eval_only/citation_prediction_v3_iclr.jsonl`

Quick metric check:

```bash
sed -n '1p' /scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr5em6-sft_filtered_m4_n10_k20_part0/dumped_evals/eval_only/aggregated_results.jsonl
sed -n '1p' /scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr5em6-sft_filtered_m4_n10_k20_part1/dumped_evals/eval_only/aggregated_results.jsonl
```

Count eval examples:

```bash
wc -l /scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr5em6-sft_filtered_m4_n10_k20_part0/dumped_evals/eval_only/citation_prediction_v3_iclr.jsonl
```

### Open the eval viewer

```bash
conda activate viewer
python /home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/harish_setup/base_eval_viewer.py \
  --evals-dir /scratch/gpfs/ZHUANGL/hk4638/logs/sweep/
```

## 6. Most Important Paths

### Training

- GRPO script: `/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/harish_setup/train_citation_prediction_4b_ailab.slurm`
- sweep launcher: `/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/harish_setup/launch_ailab_sweep.sh`
- logs: `/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3-4b_*`
- checkpoints: `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/cit-v3-4b-h200-*`

### SFT

- SFT script: `/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/harish_setup/train_citation_prediction_sft_ailab.slurm`
- prep dirs: `/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/sft_claude_high_recall/p318_e1_lr*`
- checkpoints: `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/citation_prediction_v3_sft/p318_e1_lr*/step_143/policy`

### Filtered evals

- eval script: `/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/harish_setup/eval_sweep_job.slurm`
- output root: `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/`
- filtered runs:
  - `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr2em6-sft_filtered_m4_n10_k20_part0`
  - `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr3em6-sft_filtered_m4_n10_k20_part1`
  - `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr5em6-sft_filtered_m4_n10_k20_part0`
  - `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/sft-lr5em6-sft_filtered_m4_n10_k20_part1`
