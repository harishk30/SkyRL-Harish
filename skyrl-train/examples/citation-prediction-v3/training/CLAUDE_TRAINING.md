# Citation Prediction V3 — Training Analysis & Guidance

## 1. 4B GRPO Sweep Results (Mar 9-17, 2026)

### What We Ran
- **Model**: Qwen3-4B (dense), 3 nodes x 4 GPUs (8 training GPUs), FSDP2
- **Sweep**: LR ∈ {1e-6, 5e-7} × KL ∈ {on, off} × clip ∈ {0.2, 0.28} = 8 configs
- **Key params**: `train_batch_size=128`, `n_samples_per_prompt=10`, `max_turns=6`
- **Duration**: 12hr per job (gpu-short), completed ~22/23 steps (~1 epoch)
- **WandB**: `harishkk30-princeton-university/skyrl-citation-prediction-v3`
- **Jobs**: 5294529-5294536 (7 completed, 1 liger failure)

### Result: Zero Improvement
All 7 completed configs showed identical flat reward curves — pure batch sampling noise, no upward trend. The model stayed at base model performance throughout.

| Metric | Range | Meaning |
|--------|-------|---------|
| avg_raw_reward | 0.05-0.08 | Only 6% of target citations found |
| pass@10 | 0.40-0.55 | ~50% of prompt groups have ALL-ZERO rewards |
| policy_entropy | ~0.35 (flat) | No policy change |
| grad_norm | 0.06-0.11 | Negligible gradients |
| clip_ratio | 0.0 | Clipping threshold never hit |
| is_ratio (mean) | ~1.0 | Policy ≈ reference model |
| KL divergence | ~0.0009 | Essentially zero |

No config differentiation — all hyperparameter combos performed identically because the gradient signal is too weak for differences to matter.

### Timing (per step)
- generate: ~6 min (128 prompts × 10 samples, multi-turn)
- fwd_logprobs: ~6-11 min
- policy_train: ~18-22 min
- **Total: ~33 min/step**, 1 epoch (23 steps) ≈ 12.7 hours

## 2. Root Cause Analysis

### Primary Cause: 50% of prompt groups produce zero gradients
With `n_samples_per_prompt=10` and `pass@10 ≈ 0.45`, roughly **half of all prompt groups have all-zero rewards** across all 10 samples. GRPO computes advantages per-group: if all rewards in a group are 0, advantages are all 0, gradients are 0. This wastes 50% of compute on zero-signal training.

### Contributing Factor: Credit assignment over long sequences
Average sequence length is ~8000 tokens across 4-5 search turns. Reward is only given at episode end. Per-token gradient signal is extremely dilute. Notably, **failed samples are longer** (8300-9700 tokens) than successful ones (6200-7500 tokens) — the model spends more tokens on bad trajectories.

### Contributing Factor: Over-citation behavior
The model predicts **~22 citations** on average while ground truth is **~4**. The environment gives an explicit citation budget (2× GT), and the reward function has a hard spam penalty (`max_predictions_ratio=2.0` → budget ≈ 8). Samples exceeding this get reward = 0, even if they found correct citations. This is correct behavior by the reward function — the model should learn to stay under budget — but it further reduces the pool of non-zero reward samples.

### Why All Hyperparameter Configs Are Identical
The gradient signal is so weak (50% zero, rest very noisy) that the LR, KL, and clip parameters make no measurable difference. The learning rate doesn't matter when there's no gradient to scale.

## 3. Solution: Dynamic Sampling (Built Into SkyRL)

### What It Is
SkyRL already implements dynamic sampling with two strategies:

**`filter` (DAPO-style)** — `trainer.algorithm.dynamic_sampling.type="filter"`:
1. Generate a batch of prompts with n_samples each
2. Compute rewards, filter to keep only prompt groups with reward variance > 0
3. If not enough "good" prompts to fill `train_batch_size`, generate another batch and accumulate
4. Repeat until we have enough, then proceed to training

This ensures **100% of prompt groups in the training batch have gradient signal**, vs ~45% without it.

**`replace` (POLARIS-style)** — `trainer.algorithm.dynamic_sampling.type="replace"`:
- Instead of resampling, duplicate good prompt groups to replace bad ones
- Simpler, no extra generation, but may introduce bias

### Expected Impact
- With pass@10 ≈ 0.45, ~45% of prompt groups have variance > 0
- To fill batch of 128 prompts: need ~128/0.45 ≈ 284 prompts generated ≈ 2.2 batches
- Extra generation time: ~7 min per step (~2 extra generation rounds × ~6 min each, overlapping)
- But backward pass is now on 128 **all-useful** prompts (2× effective signal)
- Net: ~20-35% slower per step, but **2× gradient signal density**

### Config Changes
```yaml
trainer.algorithm.dynamic_sampling.type="filter"
trainer.algorithm.dynamic_sampling.max_sample_batches=5  # safety limit
```

### Reference
This approach matches the paper's methodology:
> "We create an initial training set, D', by filtering out problems where the model either always fails or always succeeds, as these will offer no learning signal for RL training."

SkyRL's `filter` strategy does exactly this at the batch level.

## 4. Other Potential Improvements (Not Yet Implemented)

### 4a. Reduce max_turns (credit assignment)
Reduce from 6 to 3-4 turns. Successful samples average ~4.5 turns, so 3-4 should capture most value while halving sequence length (better credit assignment, faster generation).

### 4b. Increase n_samples (more contrast)
Go from 10 to 20 samples per prompt, with `train_batch_size=64` to keep total sequences constant. pass@20 ≈ 0.70, meaning 70% of groups have variance (vs 45% with n=10). Combined with dynamic sampling, this means fewer resampling rounds needed.

### 4c. Curriculum learning (data-level)
Pre-evaluate all 2855 training prompts with base model (pass@20). Sort by difficulty; train on easiest 30-50% first (prompts with 2-3 GT citations). This is the paper's "data refresh" concept — continuously monitor and refresh the training set as the model improves.

### 4d. SFT warmstart
Generate successful trajectories using Gemini or a stronger model. SFT the 4B model on those, then GRPO from the SFT checkpoint. Standard DeepSeek-R1 approach. This gives the model a much stronger starting point so GRPO has more reward signal to work with.

## 5. What To Watch For (Diagnostic Guide)

### Key Metrics Dashboard

| Metric | Failed Runs (baseline) | Healthy Target | Red Flag |
|--------|----------------------|----------------|----------|
| `reward/avg_raw_reward` | 0.06 flat | Upward trend | — |
| `policy/policy_entropy` | 0.35 flat | Drifting 0.35 → 0.25 | Collapse to <0.10 |
| `policy/policy_kl` (KL runs) | 0.0009 | Growing to 0.01-0.5 | Explosion >10 |
| `loss/avg_raw_advantages` (signed) | ±0.02 | ±0.05-0.10 | Still ±0.02 |
| `loss/avg_raw_advantages_abs` | 0.28-0.43 | Same or higher | Drop to <0.10 |
| `policy/grad_norm` | 0.06-0.10 | 0.2-1.0 | Spikes >5.0 |
| `policy/loss_metrics/clip_ratio` | 0.0 | 0.01-0.10 | >0.30 |
| `policy/loss_metrics/is_ratio_max` | 10-19 (dormant) | <10 | Consistently >50 |
| `rollout_train_logprobs_abs_diff_mean` | 0.018 → 0.016 (declining) | Stable or increasing | — |

### Metric-by-Metric Interpretation

**Entropy** (`policy/policy_entropy`):
Dead flat at ~0.35 for all 22 steps across all configs — the policy distribution did not change. In healthy RL, entropy should gradually decrease (model becoming more confident) or briefly increase then decrease (explore-then-exploit). Flat = not learning. Collapse below 0.10 = model became deterministic too fast, kills exploration, usually unrecoverable without reverting to a checkpoint.

**KL Divergence** (`policy/policy_kl`):
Was 0.0009 the entire training — the policy never diverged from the reference. Healthy RLHF sees KL grow from 0 to 1-5. If KL starts growing to 0.01+, the model is learning. In no-KL runs, watch for rapid divergence — KL > 10 means the policy has drifted dangerously far.

**Advantages**:
- `avg_raw_advantages` (signed mean): ±0.02 in failed runs. This is ~0 **by construction** — GRPO subtracts the group mean, so advantages within each group sum to ~0. This metric is not very diagnostic on its own.
- `avg_raw_advantages_abs`: 0.28-0.43 — the actual per-sample signal within each prompt group. Successful samples get large positive advantages (~+3.0), failed ones get small negatives (~-0.33). This signal exists but is spread across ~8,000 tokens per trajectory, diluting the per-token gradient. Should stay similar or increase with dynamic sampling.

**Importance Sampling Ratios**:
- `is_ratio_mean` ≈ 1.0, `is_ratio_std` ≈ 0.20 = policy identical to reference. Learning → std should grow above 0.25+.
- `is_ratio_max` spiked to 10-19 on tokens with near-zero advantages (dormant, no harm yet). **Latent instability risk** — if those tokens get non-zero advantages, could cause sudden large updates. Consistently >50 = unstable.
- `clip_ratio` = 0.0 — PPO clipping never activated. Healthy is 0.01-0.10; >0.30 means too much gradient is being clipped away.

**Gradient Norm**: 0.06-0.10, roughly 50x smaller than typical (0.5-5.0 for 4B). Should increase to 0.2+ if dynamic sampling works. Spikes >5.0 = instability.

**Logprob Drift** (`rollout_train_logprobs_abs_diff_mean`): Was ~0.018 and declining — model becoming MORE static. Should be stable or slightly increasing if learning. Sharp increase >0.1 = policy changing too fast, on-policy assumption breaking.

**Learning Rate**: Linear warmup spans the full epoch. At step 22, LR was only ~68% of target. Don't over-interpret the first 10 steps.

### Decision Framework After Dynamic Sampling Runs

**If reward trends upward AND entropy/KL start moving:**
Dynamic sampling worked. Continue training longer (24hr).

**If grad_norm increases (0.2+) BUT reward stays flat:**
Gradient signal exists but doesn't translate to better generation. Likely credit assignment — try reducing `max_turns` from 6 to 3.

**If grad_norm stays at 0.06-0.10 and reward stays flat:**
Dynamic sampling didn't provide enough signal. Move to:
1. Curriculum learning (train on easy prompts first)
2. Increase `n_samples` to 20 with `train_batch_size=64`
3. SFT warmstart

**If entropy collapses (<0.10) or KL explodes (>10):**
Instability. Revert to last stable checkpoint, lower LR, enable KL penalty.

## 6. Training Plan

### Phase 1: Dynamic Sampling (immediate)
Full 8-config sweep with `dynamic_sampling.type="filter"` on Qwen3-4B. Jobs 5767991-5767998, 12hr gpu-short.

### Phase 2: Evaluate & Iterate
After Phase 1 results:
- If **reward increases**: continue training longer (24hr), pick best config
- If **still flat**: combine dynamic sampling with reduced turns or increased n_samples
- If **still flat after combinations**: SFT warmstart or curriculum learning

## 7. Key File References

| File | Purpose |
|------|---------|
| `skyrl_train/utils/trainer_utils.py:307` | `handle_dynamic_sampling()` — filter/replace logic |
| `skyrl_train/utils/trainer_utils.py:441` | `handle_filter_sampling()` — DAPO-style accumulation |
| `skyrl_train/utils/trainer_utils.py:568` | `zero_variance_filter()` — per-batch filtering |
| `skyrl_train/trainer.py:229-235` | Dynamic sampling integration in training loop |
| `skyrl_train/config/config.py:237` | `DynamicSamplingConfig` |
| `skyrl_train/config/ppo_base_config.yaml:170` | Default dynamic sampling config |
| `skyrl_gym/envs/citation_prediction_v2/utils.py:39` | `compute_recall_reward()` with spam penalty |
| `skyrl_gym/envs/citation_prediction_v2/env.py:113` | Reward computation (recall at episode end) |
| `examples/citation-prediction-v3/harish_setup/train_citation_prediction_4b.slurm` | 4B training script |
