# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is `examples/citation-prediction-v3/harish_setup` for training **Citation Prediction V3** on Princeton's Della cluster using **Qwen3.5-35B-A3B** (MoE, 35B total / 3B active params) with GRPO.

V3 uses LLM-based (Gemini) subsection splitting to include ALL papers (not just those with explicit subsection headings), generating rich descriptive queries that capture author intent.

### V3 vs V2

| Aspect | V2 | V3 |
|--------|----|----|
| Paper selection | Only papers with explicit subsection headings in RW | ALL papers with Related Work sections |
| Subsection splitting | Regex-based on numbered headings (e.g. "2.1 FOO") | Gemini LLM splits by citation clusters |
| Query/heading | Raw heading text ("2.1 Transfer Learning") | Rich author-intent query ("The author is looking to find...") |
| Scale | ~1,580 subsections from ~2K papers | ~10K-20K subsections from ~5K papers |
| Coverage filter | None (just needs headings) | >=70% of citations must be in arxiv corpus |
| Downstream env | `citation_prediction_v2` | `citation_prediction_v2` (same, drop-in compatible) |

### Key Design Decisions
- **env_class stays `citation_prediction_v2`** — the gym environment, stop strings, and reward computation are unchanged
- **V3 corpus is schema-compatible with V2** — tree decomposition, SFT trajectory generation, and all viewers work unchanged
- **No local .env** — all scripts source `/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env` which has both `WANDB_API_KEY` and `GEMINI_API_KEY`
- **Viewers are symlinked from v2** — `dataset_viewer.py`, `base_eval_viewer.py`, `trajectory_viewer.py` work with v3 data (same parquet schema)

## Model: Qwen3.5-35B-A3B (current)

- **Architecture**: Gated DeltaNet + sparse MoE hybrid. 35B total params, 3B active per token, 256 experts (vs 128 in Qwen3-30B-A3B), top-k=8
- **Model class**: `Qwen3_5MoeForConditionalGeneration` (new `qwen3_5_moe` architecture, NOT compatible with standard transformers)
- **Native context**: 262,144 tokens
- **Path**: `/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/b1fc3d59ae0ab1e4279e04a8dd0fc4dc361fc2b6`
- **Requires 80GB A100 GPUs** (`#SBATCH --constraint=gpu80`)
- **Thinking by default**: Model generates `<think>...</think>` blocks before responding. Cannot be disabled without quality degradation.

### Qwen3.5 Recommended Sampling Parameters
- **Thinking mode (general)**: `temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5`
- **Non-thinking mode**: `temperature=0.7, top_p=0.8, top_k=20, min_p=0`
- `presence_penalty=1.5` is important for reducing repetitive patterns during long-form generation

### Qwen3.5 Thinking Content in Multi-Turn History
The Qwen3.5 chat template automatically strips `<think>...</think>` from **historical** assistant messages (before the last user query) and only preserves thinking for the **latest** assistant response. However, this only works with `use_conversation_multi_turn=true` (proper multi-turn message format). Our current eval config uses `use_conversation_multi_turn=false` (single assistant message), so thinking from all turns persists in context. This is acceptable for now because:
1. The model can reference prior reasoning (helpful for iterative search)
2. Token budget filtering prevents context overflow
3. Switching to true multi-turn + thinking stripping requires Path A (retokenize each turn with custom template), which is a bigger change

**To actually strip thinking from history**, you'd need BOTH `use_conversation_multi_turn=true` AND a `custom_chat_template` (Path A in SkyRL). Path B (`use_conversation_multi_turn=true` without custom template) is token-in-token-out and does NOT retokenize, so thinking tokens from prior turns remain in the accumulated `input_ids`.

### Qwen3.5 Dependency Requirements

The model requires bleeding-edge dependencies not in the standard pyproject.toml:

| Package | Required Version | Standard Version | Why |
|---------|-----------------|-----------------|-----|
| transformers | 5.3.0.dev0 (git main) | 4.57.x | `qwen3_5_moe` architecture not in stable releases |
| vLLM | 0.15.1 (latest stable) | 0.13.x | Qwen3.5 model support + restructured openai API |
| torch | 2.9.1+cu128 | varies | Must be CUDA build — vLLM install pulls in CPU-only torch |
| ray | 2.51.1 (pinned) | 2.54.0 (from vLLM) | 2.54 removed `ray.experimental.collective.util` needed by SkyRL |
| huggingface_hub | 1.4.1+ | older | `is_offline_mode` import needed by transformers 5.x |

**Installation commands** (from login node, no proxy needed):
```bash
cd /home/hk4638/SkyRL/skyrl-train
export VIRTUAL_ENV=".venv"

# vLLM 0.15.1 stable (install first — may downgrade transformers, pull CPU torch, and upgrade Ray)
~/.local/bin/uv pip install vllm==0.15.1 --torch-backend=auto

# torch with CUDA (--reinstall needed because uv sees the CPU version as satisfying the requirement)
~/.local/bin/uv pip install --reinstall torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Ray downgrade (vLLM pulls in 2.54 which removes ray.experimental.collective.util needed by SkyRL)
~/.local/bin/uv pip install ray==2.51.1

# transformers from git main (--no-deps to prevent pulling back down to stable)
~/.local/bin/uv pip install --no-deps 'transformers[serving] @ git+https://github.com/huggingface/transformers.git@main'

# huggingface_hub upgrade
~/.local/bin/uv pip install -U huggingface_hub
```

**These must be re-applied if the venv is recreated or `uv sync` is run, as they override pinned versions.**

### Critical: vLLM 0.15+ Import Path Changes

vLLM 0.15+ restructured the openai entrypoints. SkyRL's `vllm_engine.py` imports have been patched:

| Old Import (vLLM ≤0.13) | New Import (vLLM 0.15+) |
|--------------------------|------------------------|
| `vllm.entrypoints.openai.serving_chat` | `vllm.entrypoints.openai.chat_completion.serving` |
| `vllm.entrypoints.openai.serving_completion` | `vllm.entrypoints.openai.completion.serving` |
| `vllm.entrypoints.openai.serving_models` | `vllm.entrypoints.openai.models.serving` |
| `vllm.entrypoints.openai.protocol` (ChatCompletionRequest, etc.) | Split: `chat_completion.protocol`, `completion.protocol`, `engine.protocol` |

**Do NOT use vLLM nightly (0.16.0rc2)** — it has the same restructuring plus additional breaking changes that cause `libtorch_cuda.so` load failures.

### Critical: `uv run --no-sync` (NOT `--frozen`)

SLURM scripts must use `uv run --active --no-sync` instead of `uv run --active --frozen`. The `--frozen` flag prevents lockfile updates but still **syncs the venv to the lockfile**, which reverts our manually installed packages (transformers, vLLM, torch) back to their pinned versions. `--no-sync` skips the sync entirely.

## Previous Model: Qwen3-30B-A3B (deprecated)

- **Path**: `/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Still cached but no longer used for sweep/training. All sweep configs now target `qwen3_5_35b_a3b`.

## Temporal Filtering (Data Leakage Prevention)

The retriever now supports date-based filtering to prevent the model from seeing papers published after the query paper's submission date. This prevents data leakage where future papers (e.g., 2024 papers for an ICLR 2022 query) appear in search results.

### How It Works
- ArXiv IDs encode dates: `YYMM.NNNNN` (e.g., `2103.12345` = March 2021)
- ICLR year maps to cutoff: `ICLR 20XX → cutoff YYMM = (XX-1)*100 + 10` (October of year before)
- Formula in code: `iclr_year_to_cutoff(year) = (year - 2001) * 100 + 10`

| ICLR Year | Submission Deadline | Cutoff (YYMM) |
|-----------|-------------------|---------------|
| 2020 | ~Sep 2019 | 1910 |
| 2021 | ~Oct 2020 | 2010 |
| 2022 | ~Oct 2021 | 2110 |
| 2023 | ~Sep 2022 | 2210 |
| 2024 | ~Sep 2023 | 2310 |
| 2025 | ~Oct 2024 | 2410 |

### Files Modified (4 files, all backward-compatible)
1. **Retriever server** (`examples/citation_prediction/retriever/retrieval_server.py`): Added `arxiv_id_to_yymm()`, date index in `DenseRetriever.__init__`, `max_date` param in `_search` (over-retrieves 3x then filters), `max_date` in `QueryRequest`
2. **Search client** (`skyrl_gym/tools/search.py`): Added `max_date: Optional[int]` to `call_search_api`, passed in payload
3. **Environment** (`skyrl_gym/envs/citation_prediction_v2/env.py`): Extracts `self.max_date` from `extras`, passes to `call_search_api`
4. **Parquet generation** (`data_v3/generate_v3_parquet.py`): Added `--metadata_csv` arg, computes `max_date` per paper from ICLR year, adds as top-level parquet column (flows into `env_extras` automatically via `dataset.py`)

When `max_date=None` (old parquets or missing data), no filtering is applied — fully backward-compatible.

## Critical: dataset.py Chat Template Fix

**Problem**: Qwen3.5's chat template strictly validates message structure. The `_prompt_token_count` filter in `dataset.py` was passing raw JSON strings to `apply_chat_template`, causing `jinja2.exceptions.TemplateError: No user query found in messages.`

**Fix** (already applied in `skyrl_train/dataset/dataset.py`):
```python
def _prompt_token_count(doc):
    messages = doc[prompt_key]
    if isinstance(messages, str):
        messages = json.loads(messages)
    return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
```

This handles the fact that citation prediction parquets store prompts as JSON strings, not native list-of-dicts.

## Critical: Qwen3 MoE + FSDP2 + Gradient Checkpointing Bug

**Problem**: Qwen3 MoE models crash with `CheckpointError: A different number of tensors was saved during the original forward and recomputation` on the second training step. This is a [known issue](https://github.com/volcengine/verl/issues/3258) — the MoE expert routing in transformers only loops over "hit" experts, which varies between forward passes, causing non-deterministic tensor counts under `torch.utils.checkpoint`.

**Fix**: Patch the installed transformers to loop over ALL experts instead of only hit experts:

File: `.venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py`

```python
# BEFORE (line ~249):
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
for expert_idx in expert_hit:

# AFTER:
# expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
for expert_idx in range(self.num_experts):
```

This makes the forward deterministic (always iterates all experts). Inactive experts get empty inputs so overhead is minimal. **This patch must be re-applied if transformers is reinstalled or updated.**

**Note**: For Qwen3.5-35B-A3B (256 experts), the equivalent file may be at `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`. Verify the path after installing transformers from git main. The same pattern applies — loop over `range(self.num_experts)` instead of `expert_hit`.

## Critical: Della Proxy Configuration

**Problem**: `module load proxy/default` sets `http_proxy=http://della-proxy:8080`. The DNS name `della-proxy` resolves to TWO IPs: `172.28.15.2` (BROKEN) and `172.28.15.7` (WORKING). If the broken IP is tried first, WandB and any internet access times out.

**Fix**: After `module load proxy/default`, override with the working IP:
```bash
export http_proxy=http://172.28.15.7:8080
export https_proxy=http://172.28.15.7:8080
export HTTP_PROXY=http://172.28.15.7:8080
export HTTPS_PROXY=http://172.28.15.7:8080
```

**Note**: Proxy is NOT needed on the login node. If you get 403 Forbidden errors when downloading models/packages from the login node, `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`.

Proxy vars must also be explicitly forwarded into Ray head/worker srun subshells and to the Ray runtime env (handled by `prepare_runtime_environment()` in `skyrl_train/utils/utils.py`).

## Critical: Liger Kernel

- **Version**: 0.7.0 (installed in .venv, not in pyproject.toml — manually installed)
- SkyRL uses `AutoLigerKernelForCausalLM` in `model_wrapper.py` which auto-detects model type
- Liger Kernel is a performance optimization (fused Triton kernels) — training works without it but uses more memory/time
- Enabled via `+trainer.policy.use_liger_kernel=true` and `+trainer.ref.use_liger_kernel=true` (the `+` prefix is needed because the YAML config doesn't declare these keys — OmegaConf struct mode blocks undeclared keys)
- The MoE gradient checkpointing bug affects training with OR without Liger — it's a transformers issue, not Liger

## Eval Sweep Configuration

### 2-Node Architecture (per job)
- **Node 0**: Retriever (4 GPUs for FAISS sharding) + joins Ray cluster
- **Node 1**: Ray head with vLLM inference (4 GPUs, TP=2, 2 engines)

### Sweep Grid
- **M values** (search turns): 4, 6, 8, 10, 15
- **N values** (results per turn): 3, 5, 10, 20
- **Parts**: 2 per combo (50 prompts each, 100 total)
- Combos exceeding token budget are automatically skipped

### Model-Specific Configs in eval_sweep_job.slurm

| Config | qwen3_4b | qwen3_5_35b_a3b |
|--------|----------|-----------------|
| MAX_INPUT_LENGTH | 32000 | 65536 |
| MAX_GEN_LENGTH | 1024 | 4096 (for thinking) |
| ROPE_SCALING | none | max_model_len=65536 |
| DIR_PREFIX | v3 | v3-35b |

### Token Budget Calculation (sweep_eval.sh)
```
tokens_per_turn = GEN_PER_TURN + 300 * n
total_tokens = m * tokens_per_turn
```

| Model | GEN_PER_TURN | TOKEN_BUDGET | Notes |
|-------|-------------|-------------|-------|
| qwen3_4b | 1024 | 27904 | 32K context - 4K prompt |
| qwen3_4b_thinking | 8192 | 155648 | 164K context - margins |
| qwen3_5_35b_a3b | 4096 | 61440 | 65K context - 4K prompt |

### SLURM Queue Limit
The `gpu-test` partition has a **25-job per-user submit limit**. With 2 parts per combo, plan accordingly. If submitting 13 combos (26 jobs), the last one will be rejected — submit it after a job completes.

## Training Configuration (FSDP2)

### 4-Node Architecture (FSDP2)
- **Node 0** (retriever node): FAISS retriever server on port 8000, joins Ray cluster with 0 GPUs
- **Nodes 1-3** (training nodes): Ray head + 2 workers, 4 GPUs each = 12 training GPUs
- FSDP2 strategy with Liger kernel, ref model CPU offload

### Key Parameters (Full Training)

| Parameter | Value |
|-----------|-------|
| `train_batch_size` | 128 |
| `policy_mini_batch_size` | 128 |
| `micro_train_batch_size_per_gpu` | 2 |
| `micro_forward_batch_size_per_gpu` | 2 |
| `n_samples_per_prompt` | 10 |
| `epochs` | 5 |
| `max_turns` | 6 |
| `gpu_memory_utilization` | 0.9 |
| `num_inference_engines` | 6 |
| `inference_engine_tensor_parallel_size` | 2 |
| `time` | 24 hours |
| `partition` | gpu-short |

### GRPO Hyperparameter Sweep (Feb 28, 2026)
8 configs submitted on `gpu-short`, all FSDP2 with Qwen3-30B-A3B:

| # | LR | KL | eps_clip_high | Job ID |
|---|-----|-----|-------------|--------|
| 1 | 1e-6 | on | 0.2 | 5101939 |
| 2 | 1e-6 | on | 0.28 | 5175196 |
| 3 | 1e-6 | off | 0.2 | 5175198 |
| 4 | 1e-6 | off | 0.28 | 5175199 |
| 5 | 5e-7 | on | 0.2 | 5175201 |
| 6 | 5e-7 | on | 0.28 | 5175203 |
| 7 | 5e-7 | off | 0.2 | 5175204 |
| 8 | 5e-7 | off | 0.28 | 5175206 |

Script is parameterized via `SWEEP_LR`, `SWEEP_KL`, `SWEEP_CLIP` env vars (defaults to config #1).
Submit with: `sbatch --export=ALL,SWEEP_LR=5.0e-7,SWEEP_KL=false,SWEEP_CLIP=0.28 train_citation_prediction.slurm`

### Batch Size Divisibility (FSDP2, 12 training GPUs)
- DP = 12 (FSDP2 shards across all GPUs)
- `policy_mini_batch_size_per_gpu = mini_batch * n_samples / DP`
- Must be divisible by `micro_train_batch_size_per_gpu`
- Example: 128 * 10 / 12 = 106.67 — works because FSDP2 DP = sequence_parallel_size = 1, so effective DP = 12

## Megatron Backend (Experimental)

### Status: Works but slower than FSDP2 on 4-GPU/node clusters

Megatron-Core was tested as an alternative to FSDP2. Key findings:
- **12 GPUs (3×4) NOT enough** — OOMs on gradient buffer allocation regardless of TP/EP config
- **16 GPUs (4×4) works** with TP=4, PP=2, EP=8 (matching reference script)
- **Step latency**: ~353s (Megatron) vs ~250s (FSDP2) — FSDP is faster
- **Why slower**: PP=2 adds pipeline bubble overhead, EP=8 adds all-to-all communication, micro_batch forced to 1

### Megatron Config (5-node, 16 training GPUs)
```
TP=4, PP=2, EP=8, ETP=1 → DP=2
gpu_memory_utilization=0.6 (NOT 0.9 — must leave room for activations)
micro_train_batch_size_per_gpu=1
NUM_INFERENCE_ENGINES=8, INFERENCE_ENGINE_TP=2
```

### Why FSDP2 > Megatron for our setup
- FSDP2 shards everything across all 12 GPUs (1/12 per GPU at rest), communicates via all-gather/reduce-scatter
- Megatron keeps full layer shards on each GPU (TP=4 means 1/4 per GPU permanently), needs PP/EP to fit
- On 8-GPU/node clusters, Megatron wins (TP all-reduce on NVLink). On 4-GPU/node, the advantage is minimal
- FSDP2 also has simpler weight sync to vLLM (no TP/PP resharding needed)

### Megatron Batch Size Divisibility (16 training GPUs)
- `policy_dp_size = world_size / (TP × PP × CP)` — **EP is NOT included** in DP calculation
- With TP=4, PP=2: DP = 16/8 = 2
- `mini_batch_per_gpu = mini_batch * n_samples / DP`

### Files
- Smoke test: `harish_setup/smoke_test_megatron.slurm`
- Full training: `harish_setup/train_citation_prediction_megatron.slurm` (needs updating to match smoke test config)
- Reference: `examples/megatron/run_megatron_qwen3-30b-a3b.sh`

### Memory Layout (colocate_all=true, per GPU)
- vLLM: 28.5 GiB loaded → sleeps to ~1.2 GiB
- FSDP Policy (no cpu_offload): ~28 GiB active during training
- FSDP Ref (cpu_offload=true): params on CPU, GPU usage only during forward
- KV cache: ~37 GiB available after model load
- Total: fits in 80GB A100 with micro_batch=1

## Della Cluster Constraints

1. **No internet on compute nodes** — pre-download everything from login node
2. **Proxy**: Use forced IP `172.28.15.7:8080` on compute nodes (see proxy section above). NOT needed on login node.
3. **Set `NO_PROXY`** for internal cluster traffic (exact hostnames, not wildcards)
4. **Use `uv run --active --frozen`** for Ray and training commands
5. **Set `HF_HUB_OFFLINE=1`** and use local model paths
6. **Set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`** to prevent Ray from packaging working_dir
7. **Use absolute paths** in srun subshells (minimal PATH on compute nodes)
8. **Use `source /usr/licensed/anaconda3/2024.10/etc/profile.d/conda.sh`** for conda init
9. **`--constraint=gpu80`** required for 35B model
10. **`--mem=256G`** sufficient for eval sweep (colocate_all=false, 2 nodes). Training with colocate_all=true may need `--mem=512G`.
11. **SLURM copies scripts at submission** — editing .slurm after submission does NOT affect pending jobs
12. **srun subshells don't inherit proxy vars** — must explicitly export them
13. **`gpu-test` partition**: max 25 concurrent submitted jobs per user, 1-hour time limit

## V3 Data Pipeline (5 Steps, run on login node)

```
Step 1: extract_all_related_work.py  →  all_papers_rw.json
Step 2: filter_by_coverage.py        →  filtered_papers.json + coverage_distribution.png
Step 3: gemini_split_subsections.py  →  gemini_subsections.json  (needs GEMINI_API_KEY)
Step 4: build_v3_corpus.py           →  subsection_corpus_v3.json
Step 5: generate_v3_parquet.py       →  train.parquet, validation.parquet, test.parquet
```

### Running the Pipeline

```bash
cd /home/hk4638/SkyRL/skyrl-train
export VIRTUAL_ENV="/home/hk4638/SkyRL/skyrl-train/.venv"
export UV_CACHE_DIR="/scratch/gpfs/ZHUANGL/hk4638/uv_cache"
export UV_LINK_MODE=copy

METADATA=/home/hk4638/scratch/shared/data/massive_metadata.csv
SPLIT_DIR=/home/hk4638/scratch/shared/data/iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6
CORPUS=/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl
OUTPUT_DIR=/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3

# Step 1: Extract Related Work from ALL papers (~10 min)
uv run --active --frozen python examples/citation-prediction-v3/data_v3/extract_all_related_work.py \
    --metadata_csv $METADATA --split_dir $SPLIT_DIR --output $OUTPUT_DIR/all_papers_rw.json

# Step 2: Filter by corpus coverage (~15 min, reads full corpus)
uv run --active --frozen python examples/citation-prediction-v3/data_v3/filter_by_coverage.py \
    --input $OUTPUT_DIR/all_papers_rw.json --arxiv_corpus $CORPUS \
    --output $OUTPUT_DIR/filtered_papers.json --plot $OUTPUT_DIR/coverage_distribution.png

# Step 3: Gemini splitting (~30 min for ~5K papers, needs internet + GEMINI_API_KEY)
uv run --active --frozen python examples/citation-prediction-v3/data_v3/gemini_split_subsections.py \
    --input $OUTPUT_DIR/filtered_papers.json --metadata_csv $METADATA \
    --output $OUTPUT_DIR/gemini_subsections.json --concurrency 10

# Step 4: Build corpus with cited paper metadata (~15 min)
uv run --active --frozen python examples/citation-prediction-v3/data_v3/build_v3_corpus.py \
    --input $OUTPUT_DIR/gemini_subsections.json --arxiv_corpus $CORPUS \
    --output $OUTPUT_DIR/subsection_corpus_v3.json

# Step 5: Generate parquets (now with temporal filtering metadata)
uv run --active --frozen python examples/citation-prediction-v3/data_v3/generate_v3_parquet.py \
    --input $OUTPUT_DIR/subsection_corpus_v3.json \
    --output_dir $OUTPUT_DIR/full \
    --prompt_version v1 \
    --metadata_csv $METADATA
```

## Current Setup Status

| Component | Status | Location |
|-----------|--------|----------|
| V3 training data (with max_date) | ✅ Ready | `.../citation_prediction_v3/full/train.parquet` (2855 examples) |
| V3 validation data | ✅ Ready | `.../citation_prediction_v3/full/validation.parquet` |
| Arxiv corpus (with IDs) | ✅ Ready | `.../citation_prediction/arxiv_wikiformat_with_ids.jsonl` |
| FAISS index (4B) | ✅ Ready | `.../citation_prediction/qwen3_4b_embed/qwen3_Flat.index` |
| Qwen3.5-35B-A3B | ✅ Cached | `.../huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/...` |
| Qwen3-30B-A3B (deprecated) | ✅ Cached | `.../huggingface/hub/models--Qwen--Qwen3-30B-A3B/...` |
| Qwen3-Embedding-4B | ✅ Cached | `.../huggingface/hub/models--Qwen--Qwen3-Embedding-4B/` |
| Retriever conda env | ✅ Ready | `.../conda/envs/retriever` |
| Training venv | ✅ Ready | `/home/hk4638/SkyRL/skyrl-train/.venv` |
| torch (CUDA) | ✅ 2.9.1+cu128 | Reinstalled with `--index-url .../cu128` |
| transformers (git main) | ✅ 5.3.0.dev0 | Installed with `--no-deps` |
| vLLM (stable) | ✅ 0.15.1 | Latest stable with Qwen3.5 support |
| huggingface_hub | ✅ 1.4.1 | Upgraded for transformers compat |
| vLLM import patches | ✅ Applied | `vllm_engine.py` updated for 0.15+ API |
| Liger Kernel | ✅ v0.7.0 | Installed in .venv (not in pyproject.toml) |
| Gradient checkpoint patch | ⚠️ Must verify | Check `qwen3_5_moe` model file for Qwen3.5 |
| dataset.py chat template fix | ✅ Applied | JSON parsing in filter lambda |
| Temporal filtering | ✅ Deployed | retriever + search client + env + parquets |
| API keys | ✅ In `.env` | `examples/search/harish_setup/.env` |

## Files in harish_setup/

| File | Purpose |
|------|---------|
| `smoke_test.slurm` | Quick 1-hour test (3 nodes, gpu-test partition) |
| `smoke_test_single_node_2gpu.slurm` | Minimal single-node smoke test |
| `train_citation_prediction.slurm` | **Main SLURM script** - 24-hour training (3 nodes x 4 GPUs) |
| `eval_sweep_job.slurm` | Parameterized sweep eval (2 nodes, 1 hour). Supports qwen3_4b, qwen3_4b_thinking, qwen3_4b_yarn, qwen3_30b_a3b, qwen3_5_35b_a3b |
| `sweep_eval.sh` | Sweep orchestrator — submits 2 jobs per (m, n) combo (part0 + part1) |
| `tree_build_job.slurm` | Tree decomposition of v3 subsections (2 nodes, 8 hours) |
| `sft_trajectories_job.slurm` | SFT trajectory generation with Gemini (1 node, 4 hours) |
| `start_retriever.sh` | Start retriever locally (symlink to v2) |
| `download_model.slurm` | Download model weights (symlink to v2) |
| `dataset_viewer.py` | Gradio dataset inspector (symlink to v2) |
| `base_eval_viewer.py` | Gradio sweep/eval viewer (symlink to v2) |
| `trajectory_viewer.py` | Gradio training trajectory viewer (symlink to v2) |
| `CLAUDE.md` | This file |

## Files in data_v3/

| File | Purpose |
|------|---------|
| `extract_all_related_work.py` | Step 1: Extract RW from ALL papers (flat + structured) |
| `filter_by_coverage.py` | Step 2: Filter by arxiv corpus coverage + distribution plots |
| `gemini_split_subsections.py` | Step 3: Gemini LLM splits + rich query generation |
| `build_v3_corpus.py` | Step 4: Assemble v2-compatible corpus with `rich_query` field |
| `generate_v3_parquet.py` | Step 5: SkyRL parquet output (with `--metadata_csv` for temporal filtering) |
| `gemini_batch_split.py` | Batch variant of step 3 for large-scale processing |

## Architecture

```
V3 Data Pipeline → V2-compatible parquet (with max_date) → Generator (vLLM) → CitationPredictionV2Env → GRPO
                                                                  ↓
                                                            Multi-turn loop:
                                                            1. Model reasons in <think>...</think>
                                                            2. Model generates <search>query</search>
                                                            3. HTTP call to retriever → top-k arxiv docs
                                                               (filtered by max_date for temporal consistency)
                                                            4. Model cites with <citation>ID</citation>
                                                            5. When done → <done></done>
                                                            6. Reward = recall (found citations / total targets)
```

## Output Locations

| Output | Location |
|--------|----------|
| V3 data pipeline outputs | `/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/` |
| Job logs | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep-eval-v3_<jobid>.{out,err}` |
| Sweep results (4B base) | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/v3_m{m}_n{n}_k{k}_part{p}/` |
| Sweep results (35B) | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/v3-35b_m{m}_n{n}_k{k}_part{p}/` |
| Tree decomposition | `/scratch/gpfs/ZHUANGL/hk4638/logs/tree_build_v3/` |
| SFT trajectories | `/scratch/gpfs/ZHUANGL/hk4638/logs/sft_trajectories_v3/` |
| Checkpoints | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/cit-v3-smoke-*/` |
| WandB | https://wandb.ai/harishkk30-princeton-university/skyrl-citation-prediction-v3 |

## Debugging History (Feb 2026)

Issues encountered and resolved while setting up GRPO training:

1. **WandB ReadTimeout** — Della proxy has 2 IPs, one broken. Fixed by forcing `172.28.15.7:8080`
2. **OmegaConf `use_liger_kernel` key error** — YAML doesn't declare it. Fixed with `+` prefix
3. **GPU OOM on 40GB A100s** — 30B model needs 80GB. Fixed with `--constraint=gpu80`
4. **System RAM OOM (256G)** — colocate_all=true needs more RAM for ref cpu_offload. Fixed with `--mem=512G`
5. **GPU OOM during training step** — MoE aux loss allocates huge expert_mask tensor. Fixed by reducing `micro_train_batch_size_per_gpu` from 2 to 1
6. **CheckpointError on step 2** — MoE expert routing is non-deterministic across forward passes. Fixed by patching transformers to loop over all experts (see bug section above)
7. **Qwen3.5 chat template error** — `jinja2.exceptions.TemplateError: No user query found in messages.` during dataset filtering. Caused by passing JSON strings (not parsed dicts) to `apply_chat_template`. Fixed in `dataset.py` by adding `json.loads()` before template application.
8. **transformers 4.57.x doesn't support Qwen3.5** — `qwen3_5_moe` architecture not recognized. Fixed by installing transformers from git main (5.3.0.dev0).
9. **vLLM 0.13.x doesn't support Qwen3.5** — Fixed by installing vLLM nightly from `wheels.vllm.ai`.
10. **vLLM nightly downgrades transformers** — Installing vLLM nightly pulls in stable transformers, overwriting the git main version. Fixed by reinstalling transformers with `--no-deps` after vLLM.
11. **huggingface_hub `is_offline_mode` ImportError** — transformers 5.x requires newer huggingface_hub. Fixed with `uv pip install -U huggingface_hub`.
12. **Proxy 403 on login node** — Login node doesn't need proxy. `unset http_proxy https_proxy` fixes downloads.
13. **gpu-test 25-job limit** — QOS enforces max 25 submitted jobs. Plan sweep partitioning accordingly.
14. **`uv run --frozen` reverts packages** — `--frozen` still syncs venv to lockfile, overwriting manually installed transformers/vLLM/torch. Fixed by switching to `--no-sync` in all SLURM scripts.
15. **vLLM nightly installs CPU-only torch** — `vllm` pip install pulls in `torch+cpu`. Must reinstall torch with `--reinstall --index-url .../cu128` after vLLM install.
16. **vLLM 0.15+ restructured openai API** — `serving_chat`, `serving_completion`, `serving_models`, `protocol` modules moved to subdirectories. Fixed imports in `skyrl_train/inference_engines/vllm/vllm_engine.py`.
17. **vLLM 0.16.0 nightly incompatible** — Same API restructuring as 0.15 plus `libtorch_cuda.so` load failures. Downgraded to 0.15.1 stable.
18. **Ray 2.54.0 broke SkyRL** — vLLM 0.15.1 pulled in Ray 2.54.0 which removed `ray.experimental.collective.util`. SkyRL's `inference_engines/utils.py` imports `get_address_and_port` from there. Fixed by pinning Ray to 2.51.1 (lockfile version).
19. **Megatron OOM with TP=4, PP=1, EP=1 (12 GPUs)** — 62.86 GiB model + 54.74 GiB gradient buffer exceeded 80GB A100. Not enough GPUs to shard the 128 experts. Fixed by scaling to 16 GPUs with EP=4+.
20. **Megatron OOM with gpu_memory_utilization=0.9** — Colocated vLLM reserved too much GPU memory, leaving no room for training activations during forward_backward. Fixed by reducing `gpu_memory_utilization` to 0.6.
21. **Megatron OOM with micro_batch=4** — Even with TP=4, PP=2, EP=8 (matching reference), micro_batch=4 allocated 17.56 GiB for vocab logits. Reference script uses GSM8K (short sequences); citation prediction has much longer sequences. Fixed by reducing `micro_train_batch_size_per_gpu` to 1.
22. **Megatron DP excludes EP** — `policy_dp_size = world_size / (TP × PP × CP)` — EP is NOT part of the denominator. With TP=4, PP=2 on 16 GPUs, DP=2 (not DP=1). Batch sizes must be divisible accordingly.
23. **Megatron slower than FSDP2 on 4-GPU/node** — ~353s/step (16 GPUs, Megatron) vs ~250s/step (12 GPUs, FSDP2). PP=2 pipeline bubble, EP=8 all-to-all overhead, and micro_batch=1 underutilization all contribute. FSDP2 is the better strategy for Della's 4-GPU/node topology.
