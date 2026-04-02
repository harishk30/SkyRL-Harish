# Citation Prediction GRPO Training — Deep Dive (Mar 20, 2026)

## 1. RESULTS

### 1.1 Experiment Setup

**Model:** Qwen3-4B (dense), FSDP2, 3 nodes × 4 GPUs (8 training GPUs, 80GB A100s)

**Sweep:** LR ∈ {1e-6, 5e-7} × KL ∈ {on, off} × clip ∈ {0.2, 0.28} = 8 configs

**Key training params:**
- `train_batch_size=128`, `n_samples_per_prompt=10`, `max_turns=6`
- `micro_forward_batch_size_per_gpu=2`, `micro_train_batch_size_per_gpu=2`
- Recall-based reward with spam penalty (`max_predictions_ratio=2.0`)
- Linear LR warmup over first epoch

**Duration:** 12hr per job (gpu-short), completed 11-22 / 23 steps (~1 epoch)

**WandB:** `harishkk30-princeton-university/skyrl-citation-prediction-v3`

### 1.2 Summary: Zero Improvement

All 7 completed configs produced identical flat curves — pure batch sampling noise, no upward trend. The model stayed at base model performance throughout.

| Config | Final Reward | Pass@10 | Entropy | Grad Norm | Steps |
|--------|-------------|---------|---------|-----------|-------|
| lr=1e-6, noKL, clip=0.2 | 0.063 | 0.445 | 0.352 | 0.060 | 22 |
| lr=1e-6, noKL, clip=0.28 | 0.057 | 0.414 | 0.352 | 0.081 | 22 |
| lr=1e-6, KL, clip=0.28 | 0.059 | 0.445 | 0.350 | 0.111 | 18 |
| lr=5e-7, noKL, clip=0.2 | 0.054 | 0.398 | 0.355 | 0.062 | 22 |
| lr=5e-7, noKL, clip=0.28 | 0.061 | 0.398 | 0.357 | 0.063 | 22 |
| lr=5e-7, KL, clip=0.2 | 0.067 | 0.484 | 0.352 | 0.069 | 11 |
| lr=5e-7, KL, clip=0.28 | 0.051 | 0.445 | 0.352 | 0.063 | 18 |
| **Base model (no training)** | **~0.06** | **~0.44** | **—** | **—** | **—** |

No config differentiation — LR, KL, and clip made zero difference.

### 1.3 Step-by-Step Trajectories

**Reward over time** (best run: lr=1e-6, noKL, clip=0.2):
```
Step:    0     2     4     6     8    10    12    14    16    18    20    22
Reward: .054  .067  .053  .065  .064  .050  .051  .067  .073  .080  .086  .063
Pass10: .461  .539  .453  .516  .492  .422  .461  .531  .523  .477  .508  .445
```
Reward oscillates 0.050-0.086 with no trend. Pass@10 oscillates 0.42-0.55. This is batch sampling variance, not learning.

**Entropy over time** (all runs superimposed):
```
Step:    0     5    10    15    20    22
Run1:  .354  .350  .349  .350  .347  .352    (lr=1e-6, noKL, clip=0.2)
Run2:  .356  .356  .352  .349  .346  .352    (lr=1e-6, noKL, clip=0.28)
Run3:  .356  .347  .354  .348  .347  .350    (lr=1e-6, KL, clip=0.28)
```
Dead flat at 0.35 ± 0.005. No policy change in any config.

**Gradient norm over time** (all runs):
```
Typical range: 0.059 - 0.096
Occasional spikes: 0.13 (run3 step 13), 0.18 (run2 step 9)
Expected healthy range: 0.5 - 5.0
```
Gradients are 10-50× smaller than expected for a 4B model doing RL.

### 1.4 Training Diagnostics Summary

| Metric | Observed | Expected (healthy) | Interpretation |
|--------|----------|-------------------|----------------|
| `policy_entropy` | 0.35 flat | Drifting down | Policy not changing |
| `policy_kl` (KL runs) | 0.0009 | 0.01 - 1.0 | Policy ≈ reference model |
| `grad_norm` | 0.06-0.10 | 0.5-5.0 | ~50× too small |
| `clip_ratio` | 0.0 | 0.01-0.10 | PPO clipping never activates |
| `is_ratio_mean` | 0.9998 | < 0.99 | No policy shift |
| `is_ratio_std` | 0.20 constant | Growing | Static |
| `is_ratio_max` | 10-19 sporadic | < 10 | Latent instability |
| `avg_raw_advantages` (signed) | ±0.02 | ±0.02 | ~0 by construction (GRPO mean-subtraction) |
| `avg_raw_advantages_abs` | 0.28-0.43 | 0.28-0.43 | Per-sample signal exists |
| `logprobs_abs_diff_mean` | 0.018 → 0.016 | Stable/increasing | Model becoming MORE static |
| `policy_lr` at step 22 | 68% of target | 100% | Still in warmup |

### 1.5 Model Behavior

| Metric | Value | Meaning |
|--------|-------|---------|
| avg num_predicted | ~22 | Model over-cites (budget is ~8) |
| avg num_ground_truth | ~4 | Target citations per prompt |
| avg num_correct | 0.23 per sample | Model finds <1 correct citation per attempt |
| avg num_turns | 4.5 / 6 | Uses most of its turns |
| answered rate | ~96% | Almost always completes |
| avg tokens (all) | ~8,000 | Very long sequences |
| avg tokens (zero reward) | 8,300-9,700 | Failed samples are LONGER |
| avg tokens (non-zero reward) | 6,200-7,500 | Successful samples are shorter |

The model is actively using the search tools and completing episodes. It's not failing to act — it's failing to find the right papers and ignoring the citation budget.

### 1.6 Timing

```
Per step (~33 min):
  generate:       ~340s  (128 prompts × 10 samples, multi-turn with retrieval)
  fwd_logprobs:   ~500s  (reference + policy logprob computation)
  policy_train:   ~1150s (backward pass + optimizer, 8 GPUs)
  overhead:       ~40s   (checkpointing, metrics, sync)

1 epoch = 23 steps ≈ 12.7 hours
5 epochs ≈ 63 hours (infeasible on 12-24hr partitions)
LR warmup spans entire first epoch
```

---

## 2. FAILURE MODES

### FM1: Zero-Variance Prompt Groups (PRIMARY)

**What:** With `n_samples=10` and base model `pass@10 ≈ 0.45`, roughly 55% of prompt groups have all-zero rewards across all 10 samples. GRPO computes per-group advantages: all-zero rewards → all-zero advantages → zero gradients. These groups consume generation, forward pass, AND backward pass compute while contributing nothing to learning.

**Magnitude:** ~55% of total training compute is wasted on zero-gradient prompt groups. Only ~45% of the batch produces useful gradients.

**Evidence:**
- `pass@10 ≈ 0.45` means 55% of groups never see a successful sample
- From the base model eval sweep: 56% of prompts got 0/20 correct, 27% got 1-5/20, only 17% got 6+/20
- The reward distribution is heavily bimodal: most prompts are either impossible or moderately solvable for the base model

**Fix applied:** Dynamic sampling (`trainer.algorithm.dynamic_sampling.type="filter"`) — filters out zero-variance groups and resamples until the batch is full of groups with reward contrast. Jobs 5767991-5767998 submitted.

### FM2: Credit Assignment Over Long Sequences (PRIMARY)

**What:** Average sequence length is ~8,000 tokens across 4-5 search turns. GRPO applies the per-sample advantage to ALL tokens in the trajectory equally. But the tokens that actually matter for reward — the search queries and the citation decisions — are a small fraction of the total. The gradient signal per meaningful token is severely diluted.

**Magnitude:** If the "meaningful" tokens (search queries, citation text) are ~500 out of 8,000 total tokens, the effective signal per meaningful token is ~16× weaker than if we only applied gradients to those tokens.

**Evidence:**
- Failed samples average 8,300-9,700 tokens; successful ones 6,200-7,500
- The model goes down longer rabbit holes on failures — more tokens spent on bad search queries and irrelevant results
- `grad_norm` is 0.06-0.10, roughly 50× smaller than expected
- The GRPO tricks literature (Dr. GRPO) identifies sequence-level length bias: "GRPO continues to increase response length after rewards begin to plateau" and "output lengths become noticeably longer for incorrect responses"

**Potential fixes:**
- Reduce `max_turns` from 6 to 3 (halves sequence length)
- Token-level loss normalization (DAPO approach): average loss across all tokens, not per-sample then per-batch — prevents long sequences from having muted per-token impact
- Dr. GRPO's fix: normalize by `MAX_TOKENS` constant instead of actual sequence length

### FM3: LR Warmup Consumes the Entire Epoch (CONTRIBUTING)

**What:** The linear warmup schedule spans the full first epoch (23 steps). At step 22 (the last step we completed), LR was only at 68% of the target. The first ~10 steps had less than 50% of the target LR.

**Magnitude:** Effective training only occurs in the second half of the epoch. The first 10 steps (~5.5 hours) produce negligible parameter updates regardless of gradient quality.

**Evidence:**
- `policy_lr` at step 0: 0.0
- `policy_lr` at step 10: 3.2e-7 (32% of target 1e-6)
- `policy_lr` at step 22: 6.8e-7 (68% of target)
- `logprobs_abs_diff_mean` declining from 0.018 to 0.016 — model becoming MORE static over time, not less

**Potential fix:** Shorter warmup (e.g., 10% of first epoch instead of 100%). This is a standard practice — most RL training uses short warmups or no warmup at all.

### FM4: Insufficient Training Duration (CONTRIBUTING)

**What:** We only completed ~96% of 1 epoch. Combined with the LR warmup consuming that epoch, the model received effectively ~0.5 epochs of meaningful training. Even with perfect gradient signal, this may not be enough to move a 4B parameter model.

**Evidence:**
- 22 steps completed out of 23 (one epoch)
- 5 configured epochs would need ~63 hours (infeasible on 12-24hr partitions)
- Policy metrics (KL, entropy, is_ratio) show zero change — consistent with insufficient updates

**Note:** This is a contributing factor, not the root cause. If gradient signal were strong, even a few steps should show SOME movement in entropy/KL. The fact that we see nothing suggests the signal quality issues (FM1, FM2) are more fundamental.

### FM5: Batch Size Interaction with Reward Sparsity (CONTRIBUTING)

**What:** The GRPO tricks literature emphasizes that batch size is critical: "Using a small batch size in GRPO is one of the most common mistakes in RL training." The recommended baseline is 512 total sequences (64 prompts × 8 samples). We use 1,280 (128 × 10), which should be sufficient. However, after zero-variance filtering, our effective batch is only ~576 (128 × 0.45 × 10), and the effective unique prompts is ~58. This is borderline.

**Evidence:**
- 128 prompts per batch, but only ~58 have reward variance
- With dynamic sampling, we'll have 128 prompts all with variance — a stronger effective batch

### FM6: Over-Citation Triggers Spam Penalty (CONTRIBUTING)

**What:** The 4B model during training predicts ~22 citations on average vs a budget of ~8 (2× the ~4 ground truth). The environment explicitly shows the budget at each turn (`"Citations so far: X/8 max"`), but the model consistently over-cites. Samples exceeding the budget get reward = 0.

**Evidence (from 4B training WandB):**
- `num_predicted ≈ 22`, `num_ground_truth ≈ 4`, `budget ≈ 8`
- `num_correct ≈ 0.23` per sample — even when citing 22 papers, fewer than 1 is correct
- Spam penalty zeros out samples that might have found a correct citation but also over-cited

**Note:** We don't have saved 4B training trajectories to inspect the exact behavior. The over-citation could be exacerbated by temperature=1.0 sampling (vs greedy eval), or the 4B model may simply be weaker at following the budget instruction.

### FM7: IS Ratio Spikes (LATENT RISK)

**What:** `is_ratio_max` occasionally spikes to 10-19, meaning some tokens have importance ratios far outside the clipping bounds [0.8, 1.2]. Currently these occur on tokens with near-zero advantages, so they don't affect the gradient. But they represent a latent instability risk.

**Why:** The GRPO tricks literature identifies the "engine gap" — separate inference engines (vLLM for rollout) vs training engines (FSDP for update) can produce different token probabilities. This makes training implicitly off-policy. SkyRL's TIS (Truncated Importance Sampling) mechanism should catch extreme ratios, but `tis_token_clip_high_ratio ≈ 0.00001` suggests it's barely activating.

**Risk:** If learning kicks in and those tokens start getting non-zero advantages, the extreme ratios could cause sudden large gradient updates → instability.

---

## 3. RECOMMENDATIONS

### Tier 0: Already Submitted — Dynamic Sampling

**Change:** `trainer.algorithm.dynamic_sampling.type="filter"`, `max_sample_batches=5`

**What it does:** Filters out zero-variance prompt groups, resamples until the batch is full of groups with reward contrast. Directly addresses FM1 (zero-variance groups).

**Expected impact:**
- 100% of training groups will have gradient signal (vs ~45% before)
- ~2-3 generation rounds per step → ~30% slower per step
- Net: fewer steps per epoch, but every step has 2× useful signal density

**Jobs:** 5767991-5767998, full 8-config sweep. First results expected ~10 PM tonight (Mar 20).

**How we'll know if it works:** `grad_norm` should increase from 0.06 → 0.2+, `entropy` should start drifting (not flat), `reward` should trend upward. If these metrics don't move after 15+ steps at meaningful LR, dynamic sampling alone isn't enough.

### Tier 1: Config-Level Changes (hours, no code changes)

**1a. Shorten LR warmup**
- Current: warmup over full first epoch (23 steps)
- Proposed: warmup over 3-5 steps (10-20% of epoch)
- Rationale: Standard RL practice. Current warmup wastes the first ~10 steps of training. Addresses FM3.
- SkyRL config: check `trainer.warmup_steps` or similar

**1b. Reduce max_turns (6 → 3-4)**
- Rationale: Successful samples average 4.5 turns. Reducing to 3-4 cuts avg sequence length from ~8,000 to ~4,000-5,000 tokens. Directly addresses FM2 (credit assignment). Also halves generation time.
- Trade-off: Model has fewer chances to refine searches. But with 3 turns × 5 results each = 15 documents seen, which should be sufficient.

**1c. Increase n_samples (10 → 20), halve batch (128 → 64)**
- Same total sequences (1,280), but pass@20 ≈ 0.70 → 70% of groups have variance (vs 45% with n=10)
- Combined with dynamic sampling, fewer resampling rounds needed
- Trade-off: Fewer unique prompts per batch (64 vs 128). Literature suggests this is fine if prompts have good variance.

### Tier 2: Algorithmic Improvements (days, may need code changes)

**2a. Token-level loss aggregation (DAPO)**
- Current: GRPO averages loss within each sample, then across samples. Long sequences have diluted per-token impact.
- Proposed: Average loss across ALL tokens in the batch directly. Each token gets equal weight.
- Addresses FM2. This is a DAPO contribution that directly targets the length bias problem.
- Check if SkyRL already supports this via config.

**2b. Remove advantage std normalization (Dr. GRPO)**
- Current GRPO advantage: `(reward_i - mean) / std`
- Dr. GRPO advantage: `reward_i - mean` (drop the std denominator)
- Rationale: The std denominator amplifies advantages for groups with very low variance (near-uniform rewards), creating noisy gradients. Without it, advantages are more stable.
- This is a small code change in the advantage computation.

**2c. Overlong response penalty/masking (DAPO)**
- Truncated samples (hit max generation length) get confusing gradients — the model is penalized for running out of tokens, not for bad reasoning.
- Option A: Mask truncated samples from loss entirely
- Option B: Soft penalty for responses approaching max length
- Evidence: `response_length` metric shows some samples hitting 31,000+ tokens (max). These are getting full negative advantages despite potentially being on the right track.

### Tier 3: Data-Level Improvements (days-week)

**3a. Curriculum learning / difficulty filtering**
- Pre-evaluate all 2,855 training prompts with base model (e.g., pass@20)
- Filter to prompts of moderate difficulty (pass@20 between 0.1-0.8)
- Remove prompts where model always fails (no signal) or always succeeds (no contrast)
- This is the approach from the literature: "filter out problems where the model either always fails or always succeeds"
- Can periodically refresh the training set as the model improves (data refresh pipeline)

**3b. SFT warmstart**
- Generate successful trajectories using Gemini (or another strong model) on the citation prediction task
- SFT the 4B model on those trajectories
- Then switch to GRPO from the SFT checkpoint
- Rationale: The base 4B model has pass@1 ≈ 5-10%. After SFT on successful trajectories, pass@1 might be 20-30%, giving GRPO much more signal to work with.
- Standard DeepSeek-R1 approach: SFT before RL.

### Priority Order

```
Already running:  Dynamic sampling (Tier 0) — results tonight
If still flat:    1b (reduce turns) + 1a (fix warmup) — submit tomorrow
If still flat:    1c (increase n_samples) + 2a (token-level loss) — requires investigation
Parallel track:   3a (curriculum) — can start data evaluation now
Last resort:      3b (SFT warmstart) — most effort, most reliable
```
