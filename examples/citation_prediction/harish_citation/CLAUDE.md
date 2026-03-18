# CLAUDE.md — Masked Citation Prediction

This file provides context for Claude Code when working on the citation prediction project.

## Project Overview

**Goal**: Train an LLM to identify masked citations in research paper Related Works sections using multi-turn retrieval-augmented search over an arxiv database.

**How it differs from Search-R1**: In Search-R1, the model searches for information and then outputs `<answer>text</answer>`, with reward = exact match against a ground truth string. In citation prediction, the environment automatically checks whether the correct paper (by arxiv ID) appears in the retriever's top-k results. There are no `<answer>` tags — the reward comes entirely from whether the retriever returned the correct paper.

**Branch**: `harish/citation-prediction-setup` (off `fork/main`)

## Current State of Assets (as of 2025-02-10)

### What Exists

| Asset | Path | Size | Status |
|-------|------|------|--------|
| Training data (short prompt) | `.../citation_prediction/short/train.parquet` | — | Ready (5,876 examples) |
| Training data (extended prompt) | `.../citation_prediction/extended/train.parquet` | — | Ready (5,876 examples) |
| Validation data (short) | `.../citation_prediction/short/validation.parquet` | — | Ready (335 examples) |
| Validation data (extended) | `.../citation_prediction/extended/validation.parquet` | — | Ready (335 examples) |
| Test data (short) | `.../citation_prediction/short/test.parquet` | — | Ready (648 examples) |
| Test data (extended) | `.../citation_prediction/extended/test.parquet` | — | Ready (648 examples) |
| Source arxiv corpus | `/home/hk4638/scratch/shared/data/arxiv_wikiformat.jsonl` | 3.4 GB (2,922,211 lines) | Ready |
| ID-augmented arxiv corpus | `/home/hk4638/scratch/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl` | ~3.4 GB | Ready |
| Qwen3-0.6B FAISS index (over arxiv) | `/home/hk4638/scratch/shared/data/qwen3_06_embed/qwen3_Flat.index` | 12 GB | Ready (1024-dim, built by sk7524) |
| Qwen3-0.6B embeddings memmap | `/home/hk4638/scratch/shared/data/qwen3_06_embed/emb_qwen3.memmap` | 12 GB | Ready |
| Shared retriever code | `/home/hk4638/scratch/shared/retriever/index_builder.py` | 13 KB | Ready (multi-GPU DataParallel, last_token pooling) |
| 0.6B SLURM template | `/home/hk4638/scratch/shared/retriever/build_qwen3_wiki_and_arxiv.sbatch` | 2.2 KB | Reference for 4B job |
| Shared retriever server | `/home/hk4638/scratch/shared/retriever/retrieval_server.py` | 24 KB | Ready |
| Qwen3-Embedding-4B model | `/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/` | — | Cached |
| Qwen3-4B (training model) | `/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c` | — | Cached |
| Qwen2.5-3B-Instruct (old) | `/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/` | — | Cached (superseded by Qwen3-4B) |
| liger-kernel v0.6.5 | Installed in `.venv` | — | Ready (fused RoPE, SwiGLU, RMSNorm, CrossEntropy for Qwen3) |

**Note**: Parquet paths are under `/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction/`. All scripts accept `PROMPT_STYLE=short|extended` (default: `short`) to select the prompt variant. The two prompt styles differ only in the system message (short = minimal instructions, extended = detailed strategy guidance with tips).

### What Does NOT Exist Yet

| Asset | Blocker |
|-------|---------|
| Qwen3-4B FAISS index | Submit `build_4b_index.sbatch` (depends on ID-augmented corpus + 4B model, both ready) |

## Implementation Steps — Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Build corpus with arxiv IDs in `contents` | **Done** — `build_arxiv_corpus.py` created + corpus written |
| 2 | Download Qwen3-Embedding-4B model | **Done** — cached at `/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/` |
| 3 | Build Qwen3-4B FAISS index | **Script ready** — `build_4b_index.sbatch` created; needs `sbatch` submission |
| 4 | Update retriever launch scripts (0.6B/4B toggle) | **Done** — `EMBEDDING_MODEL` env var added to `retrieval_launch.sh` and `start_retriever.sh` |
| 5 | Create `CitationPredictionEnv` | **Done** — `skyrl-gym/skyrl_gym/envs/citation_prediction/{env.py,utils.py,__init__.py}` created + registered |
| 6 | Dataset script with search-oriented prompt | **Done** — `build_citation_dataset.py` has `<search>`/`<think>` prompt + `env_class="citation_prediction"` |
| 7 | Update training config | **Done** — `run_citation_prediction.sh` uses `citation_prediction` env, stop=`["</search>"]` |
| 8 | Update SLURM scripts with embedding toggle | **Done** — `train_citation_prediction.slurm` has `EMBEDDING_MODEL` toggle + `citation_prediction` config |
| — | Hydra default config | **Done** — `citation_prediction` section added to `skyrl_train/config/skyrl_gym_config/default.yaml` |
| — | Base model eval script | **Done** — `eval_base_model.slurm` uses `main_generate` entrypoint; accepts `PROMPT_STYLE`, `EVAL_SPLIT`, `EMBEDDING_MODEL` |
| — | Switch to Qwen3-4B | **Done** — All scripts updated from Qwen2.5-3B-Instruct to Qwen3-4B; model cached |
| — | Enable Liger Kernel | **Done** — `use_liger_kernel` added to PolicyConfig/RefConfig, wired through fsdp_worker, `liger-kernel` installed |
| — | Enable sequence parallelism | **Done** — `sequence_parallel_size=2` set in all training scripts (Ulysses backend) |
| — | Two prompt styles (short/extended) | **Done** — `build_citation_dataset.py` supports `--prompt_style`, parquets in `short/` and `extended/` subdirs |
| — | Fix citation counting | **Done** — Handles both parenthetical `(Author et al., 2019)` and in-text `Author et al. (2019)` citations |
| — | PROMPT_STYLE env var | **Done** — All SLURM/shell scripts accept `PROMPT_STYLE=short\|extended` (default: `short`) |

### Remaining Action: Build the 4B FAISS Index

```bash
cd /home/hk4638/SkyRL/skyrl-train
sbatch examples/citation_prediction/data/build_4b_index.sbatch
```

Output: `/home/hk4638/scratch/data/citation_prediction/qwen3_Flat.index`

Until the 4B index is built, you can test with the 0.6B index by setting `EMBEDDING_MODEL=qwen3_06b`. Note: the 0.6B index was built against the original corpus (no `[arxiv:ID]` prefixes in `contents`), so `CitationPredictionEnv` won't be able to match IDs from the retriever text. To test the full pipeline with 0.6B, you'd need to rebuild the 0.6B index against the ID-augmented corpus.

## Embedding Model Toggle Design (0.6B vs 4B)

Both Qwen3-Embedding-0.6B and Qwen3-Embedding-4B use:
- **Pooling**: `last_token`
- **Retrieval method name**: `qwen3` (same FAISS naming convention: `qwen3_Flat.index`)
- **Max length**: 256

The difference is only in model size, embedding dimension, and paths.

### How the toggle works

The retriever launch scripts (`start_retriever.sh`, `retrieval_launch.sh`) and the SLURM training script accept an `EMBEDDING_MODEL` env var:

```bash
# In start_retriever.sh (default: qwen3_4b for citation prediction):
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_4b"}

# In retrieval_launch.sh (default: qwen3_06b):
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_06b"}

if [ "$EMBEDDING_MODEL" = "qwen3_4b" ]; then
    MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
    INDEX_FILE="${DATA_DIR}/qwen3_4b_embed/qwen3_Flat.index"
    CORPUS_PATH="${DATA_DIR}/arxiv_wikiformat_with_ids.jsonl"
    RETRIEVER_NAME=qwen3
elif [ "$EMBEDDING_MODEL" = "qwen3_06b" ]; then
    MODEL_PATH="Qwen/Qwen3-Embedding-0.6B"
    INDEX_FILE="${SCRATCH_BASE}/shared/data/qwen3_06_embed/qwen3_Flat.index"
    CORPUS_PATH="${SCRATCH_BASE}/shared/data/arxiv_wikiformat.jsonl"
    RETRIEVER_NAME=qwen3
fi
```

**Important**: The 0.6B index was built against the *original* corpus (without IDs in `contents`). The 4B index will be built against the *ID-augmented* corpus. This means:
- **4B retriever**: Returns `Doc N: [arxiv:XXXX.XXXXX] Title...` — IDs are visible in text for matching
- **0.6B retriever**: Returns `Doc N: Title...` — IDs are NOT in the text

For the `CitationPredictionEnv` to match IDs, we'll need to either:
1. Rebuild the 0.6B index against the ID-augmented corpus too (preferred for consistency), OR
2. Have the env extract IDs from the raw document metadata (the `id` JSON field survives in the retriever's document response)

Decision: Go with approach 1 if we end up using 0.6B for training. For now, the 4B path is the priority.

### Corpus files

| Corpus | Path | Has IDs in `contents`? |
|--------|------|------------------------|
| Original | `/home/hk4638/scratch/shared/data/arxiv_wikiformat.jsonl` | No (`contents` has title+authors+abstract+DOI) |
| ID-augmented | `/home/hk4638/scratch/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl` | Yes (`[arxiv:XXXX.XXXXX] ` prepended) |

### Index files

| Model | Index dir | Corpus used | Status |
|-------|-----------|-------------|--------|
| Qwen3-0.6B | `/home/hk4638/scratch/shared/data/qwen3_06_embed/` | Original (no IDs) | Ready |
| Qwen3-4B | `/home/hk4638/scratch/data/citation_prediction/` | ID-augmented | **Not built yet** — submit `build_4b_index.sbatch` |

## Architecture

```
Dataset (parquet)  →  Generator (vLLM)  →  CitationPredictionEnv  →  Rewards  →  GRPO Training
                            ↓
                      Multi-turn loop:
                      1. Model reasons in <think>...</think>
                      2. Model generates <search>query</search>
                      3. Retriever returns top-k arxiv papers
                      4. Env checks if ground-truth arxiv ID is in results
                      5a. Found → reward=1.0, done=True
                      5b. Not found → observation with results, continue
                      6. Repeat until found or max_turns reached (reward=0.0)
```

### Context Levels

| Context Level | Prompt contains | Notes |
|---------------|----------------|-------|
| `sentence` | Only the sentence with `[MASKED]` | Harder task, less signal for the model |
| `paragraph` | The masked sentence + the preceding paragraph as a `Context:` block | Easier task, more topical clues |

The `build_citation_dataset.py` script accepts `--context_level` (`sentence` or `paragraph`).

### Prompt Styles

| Style | Description |
|-------|-------------|
| `short` | Minimal instructions: reason in `<think>`, search with `<search>`, refine queries |
| `extended` | Detailed strategy guidance: numbered steps, tips for extracting clues, analysis advice |

Both prompts instruct the model to use `<think>` and `<search>` tags. The extended prompt adds explicit search strategy tips (extract technical terms, consider subfield, look at surrounding context). Set via `--prompt_style` in `build_citation_dataset.py` or `PROMPT_STYLE` env var in SLURM scripts.

### Citation Counting

The dataset builder handles two citation formats:
- **Parenthetical**: `(Author et al., 2019)` or `(Author & Other, 2019; Third, 2020)` — matched by `CITE_GROUP_RE`
- **In-text**: `Author et al. (2019)` or `Author and Other (2019)` — matched by `INLINE_CITE_RE`

The `find_all_citations()` function finds both types while avoiding double-counting (inline citations that overlap with parenthetical spans are skipped). Only sentences with exactly one citation (across both types) are eligible for masking.

### Reward Mechanism

- **Search-R1**: `compute_score(chat_history, ground_truth)` → extracts `<answer>` text → exact match
- **Citation prediction**: `check_citation_match(retriever_result_ids, ground_truth_arxiv_id)` → set membership check
- No answer extraction needed. No `<answer>` tags.
- Stop tokens: `["</search>"]` only
- Regex: `\[arxiv:(\d{4}\.\d{4,5})\]` applied to retriever output text

### CitationPredictionEnv Design

`CitationPredictionEnv(BaseTextEnv)` in `skyrl-gym/skyrl_gym/envs/citation_prediction/env.py`:

- **Constructor**: Extracts `ground_truth_id` from `extras["reward_spec"]["ground_truth"]["target"]`, creates `SearchToolGroup` from `env_config`, tracks `found_correct_paper = False`
- **`_parse_action(action)`**: Regex `<search>(.*?)</search>` to extract query
- **`_execute_tool()`**: Calls `SearchToolGroup.search()`, then `extract_arxiv_ids()` on the output to check for the target ID; sets `found_correct_paper = True` on match; wraps output in `<information>` tags
- **`_is_done(action)`**: `found_correct_paper OR turns >= max_turns`
- **`_get_reward(done)`**: 1.0 if found, 0.0 otherwise (only on done)
- **`_validate_action(action)`**: Asserts `</search>` is the last string if present (stop string check)
- **`get_metrics()`**: Returns `{"found_paper": 0/1, "num_turns": N}`
- **Reuses**: `SearchToolGroup` from `skyrl-gym/skyrl_gym/tools/search.py` (no changes needed)

## Key Files Reference

### This project

| File | Role | Status |
|------|------|--------|
| `examples/citation_prediction/harish_citation/CLAUDE.md` | This file — project state and plan | — |
| `examples/citation_prediction/data/build_arxiv_corpus.py` | Prepend IDs to corpus | Done |
| `examples/citation_prediction/data/build_4b_index.sbatch` | SLURM job for 4B FAISS index | Ready to submit |
| `examples/citation_prediction/data/build_citation_dataset.py` | ICLR CSV → masked-citation parquets (supports `--prompt_style`) | Done |
| `examples/citation_prediction/data/update_parquet_prompts.py` | Switch prompt style in existing parquets without re-running CSV pipeline | Done |
| `examples/citation_prediction/data/view_dataset.py` | Inspect parquet contents | — |
| `examples/citation_prediction/eval_base_model.py` | Standalone vLLM eval — test prompts/embeddings before RL (legacy) | Done |

### Environment (created)

| File | Role |
|------|------|
| `skyrl-gym/skyrl_gym/envs/citation_prediction/env.py` | `CitationPredictionEnv(BaseTextEnv)` with ID-matching reward |
| `skyrl-gym/skyrl_gym/envs/citation_prediction/utils.py` | `extract_arxiv_ids()`, `check_citation_match()` |
| `skyrl-gym/skyrl_gym/envs/citation_prediction/__init__.py` | Package init |
| `skyrl-gym/skyrl_gym/envs/__init__.py` | Registration: `id="citation_prediction"` entry added |

### Config (updated)

| File | Role |
|------|------|
| `skyrl_train/config/skyrl_gym_config/default.yaml` | Added `citation_prediction` section (same fields as `search`) |
| `examples/citation_prediction/run_citation_prediction.sh` | Uses `env_class="citation_prediction"`, `PROMPT_STYLE` toggle |
| `examples/citation_prediction/run_citation_prediction_conversation_format.sh` | Same but with `use_conversation_multi_turn=true`, `PROMPT_STYLE` toggle |

### Retriever / training (updated)

| File | Role |
|------|------|
| `examples/citation_prediction/retriever/retrieval_launch.sh` | Generic retriever launch with `EMBEDDING_MODEL` toggle |
| `examples/citation_prediction/retriever/retrieval_server.py` | Local copy of retriever server |
| `examples/citation_prediction/harish_setup/start_retriever.sh` | Della-specific retriever with `EMBEDDING_MODEL` toggle (default: `qwen3_4b`) |
| `examples/citation_prediction/harish_setup/train_citation_prediction.slurm` | Main SLURM script with `EMBEDDING_MODEL` + `PROMPT_STYLE` toggles |
| `examples/citation_prediction/harish_setup/eval_base_model.slurm` | 2-node eval: `PROMPT_STYLE`, `EVAL_SPLIT`, `EMBEDDING_MODEL` params |
| `examples/citation_prediction/harish_setup/smoke_test.slurm` | Quick 1-hour validation test (2 nodes × 4 GPUs) |
| `examples/citation_prediction/harish_setup/smoke_test_single_node_2gpu.slurm` | Minimal test (1 node, 2 GPUs, CPU retriever) |
| `examples/citation_prediction/harish_setup/resume_training.slurm` | Resume training from checkpoint |

### Shared infrastructure (don't modify)

| File | Role |
|------|------|
| `/home/hk4638/scratch/shared/retriever/index_builder.py` | Multi-GPU DataParallel index builder (last_token pooling, memmap + FAISS) |
| `/home/hk4638/scratch/shared/retriever/retrieval_server.py` | FastAPI retriever server (`POST /retrieve`) |
| `/home/hk4638/scratch/shared/retriever/build_qwen3_wiki_and_arxiv.sbatch` | 0.6B SLURM template (reference) |

### Reference implementations (for context only)

| File | Role |
|------|------|
| `skyrl-gym/skyrl_gym/envs/search/env.py` | SearchEnv — the original search env that CitationPredictionEnv is modeled after |
| `skyrl-gym/skyrl_gym/envs/search/utils.py` | `compute_score`, `extract_solution` — exact-match reward (not used by citation prediction) |
| `skyrl-gym/skyrl_gym/envs/base_text_env.py` | Base class inherited by CitationPredictionEnv |
| `skyrl-gym/skyrl_gym/tools/search.py` | `SearchToolGroup` + `_passages2string()` — reused as-is by CitationPredictionEnv |

## Retriever API Details

The retriever server exposes `POST /retrieve`:

**Request**:
```json
{"query": "attention mechanism transformers", "topk": 3, "return_scores": true}
```

**Response**:
```json
{
  "result": [[
    {"document": {"title": "...", "text": "...", "contents": "[arxiv:1706.03762] ..."}, "score": 0.95}
  ]]
}
```

`_passages2string()` in `skyrl-gym/skyrl_gym/tools/search.py` formats results as:
```
Doc 1: [arxiv:1706.03762] "Attention Is All You Need" Authors:... Abstract:...
Doc 2: [arxiv:2301.12345] ...
```

The `[arxiv:XXXX.XXXXX]` prefix is what `CitationPredictionEnv` regex-parses to check for matches.

## Della SLURM Header Conventions

When writing SLURM scripts for this cluster, follow these conventions (learned from `train_search.slurm`):

- **No `--account`**: Do not specify `--account`. The default account is used automatically. Specifying one (e.g., `--account=zhuangl`) causes `invalid account` errors.
- **`--partition=gpu-short`**: Use `gpu-short` for GPU jobs. `gpu` and `pli` are not directly usable by this user.
- **`--gres=gpu:N`**: Use `--gres=gpu:4`, NOT `--gpus=4`. The latter is not recognized on this cluster.
- **`--ntasks-per-node=1`**: Preferred over `--ntasks=1` for multi-node jobs.
- **Log paths**: Use absolute paths on scratch (e.g., `/scratch/gpfs/ZHUANGL/hk4638/logs/`) rather than relative `logs/` dirs.

Example header that works:
```bash
#SBATCH --job-name=my_job
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/my_job_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/my_job_%j.err
```

## Config Key Mapping

```bash
# Search-R1:
environment.env_class="search"
environment.skyrl_gym.search.search_url="http://127.0.0.1:8000/retrieve"
generator.sampling_params.stop='["</search>", "</answer>"]'

# Citation Prediction:
environment.env_class="citation_prediction"
environment.skyrl_gym.citation_prediction.search_url="http://127.0.0.1:8000/retrieve"
environment.skyrl_gym.citation_prediction.topk=3
generator.sampling_params.stop='["</search>"]'

# Data paths (PROMPT_STYLE=short|extended):
data.train_data="['${DATA_DIR}/${PROMPT_STYLE}/train.parquet']"
data.val_data="['${DATA_DIR}/${PROMPT_STYLE}/validation.parquet']"

# Training params:
trainer.epochs=3                              # ~36 steps (5876 examples / 512 batch)
trainer.policy.optimizer_config.num_warmup_steps=5
generator.max_input_length=8192               # room for 4 turns of gen + search results

# Optimizations:
trainer.policy.use_liger_kernel=true
trainer.ref.use_liger_kernel=true
trainer.policy.sequence_parallel_size=2
```

## Training Model

**Current**: `Qwen/Qwen3-4B` (Qwen3 with built-in thinking mode, 4B params)
**Previous**: `Qwen/Qwen2.5-3B-Instruct` (superseded)

Cached at: `/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`

Qwen3 models have built-in `<think>` token support (no separate Instruct variant needed). All Qwen3 models are instruction-tuned by default.

## Training Optimizations

### Liger Kernel

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) provides fused Triton kernels for transformer training, reducing memory and improving throughput.

**What it fuses for Qwen3** (`apply_liger_kernel_to_qwen3`):
- RoPE (rotary position embeddings)
- SwiGLU activation
- RMSNorm
- CrossEntropyLoss / FusedLinearCrossEntropy

**Config**:
```bash
trainer.policy.use_liger_kernel=true   # policy model
trainer.ref.use_liger_kernel=true      # reference model
```

**How it works**: SkyRL's `HFModelWrapper` uses `AutoLigerKernelForCausalLM.from_pretrained()` instead of `AutoModelForCausalLM.from_pretrained()`, which auto-detects the model type and applies the right patches.

**Code path**: `config.py:PolicyConfig.use_liger_kernel` → `fsdp_worker.py:init_model()` → `model_wrapper.py:HFModelWrapper(use_liger_kernel=True)` → `liger_kernel.transformers.AutoLigerKernelForCausalLM`

### Sequence Parallelism (Ulysses)

Splits long sequences across GPUs within a node for reduced memory per GPU.

**Config**:
```bash
trainer.policy.sequence_parallel_size=2    # 2-way SP within each 4-GPU node
trainer.sequence_parallel_backend=ulysses  # default, uses all-to-all
```

With 4 GPUs/node: SP=2 creates 2 SP groups of 2 GPUs each. Each SP group processes half the sequence length, then all-to-all gathers for attention.

## Base Model Evaluation (Pre-RL)

Uses SkyRL's built-in eval-only entrypoint (`main_generate`) which runs the same multi-turn rollouts as training (same generator, same `CitationPredictionEnv`, same reward) but without GRPO updates. Reports `eval/all/avg_score` and `eval/all/pass_at_1` to WandB.

### How it works

`main_generate` (`skyrl_train.entrypoints.main_generate`) creates an `EvalOnlyEntrypoint` that:
1. Loads the model into vLLM inference engines (no FSDP workers)
2. Runs multi-turn rollouts through `CitationPredictionEnv` on the validation set
3. Computes rewards (1.0 if target arxiv ID found, 0.0 otherwise)
4. Reports `avg_score`, `pass_at_1`, `mean_positive_reward` to WandB
5. Dumps full trajectories to `export_path/dumped_evals/eval_only/`

### Usage

```bash
# Default: validation set, short prompt, qwen3_4b embeddings, greedy decoding
sbatch examples/citation_prediction/harish_setup/eval_base_model.slurm

# Use extended prompt
PROMPT_STYLE=extended sbatch examples/citation_prediction/harish_setup/eval_base_model.slurm

# Use 0.6B embeddings
EMBEDDING_MODEL=qwen3_06b sbatch examples/citation_prediction/harish_setup/eval_base_model.slurm

# Eval on test set
EVAL_SPLIT=test sbatch examples/citation_prediction/harish_setup/eval_base_model.slurm
```

### Key differences from training SLURM script

| Aspect | Training | Eval |
|--------|----------|------|
| Entrypoint | `main_base` | `main_generate` |
| Nodes | 3 (retriever + 2 training) | 2 (retriever + inference) |
| GPU usage | FSDP + vLLM share GPUs | vLLM only |
| `colocate_all` | `true` | `false` |
| `gpu_memory_utilization` | `0.5` | `0.9` |
| Sampling temp | `1.0` (exploration) | `0.0` (greedy) |
| Inference engines | 4 × TP=2 | 2 × TP=2 |

### Output

- **WandB**: Logged to `skyrl-citation-prediction` project with run name `eval-citation-<timestamp>`
- **Trajectories**: Dumped to `${LOG_DIR}/eval_${SLURM_JOB_ID}/dumped_evals/eval_only/`
- **Metrics**: `eval/all/avg_score` (mean reward), `eval/all/pass_at_1` (fraction with reward > 0)

### Legacy standalone script

`eval_base_model.py` still exists for quick ad-hoc testing (direct vLLM, no Ray/SkyRL overhead). The SLURM script now uses `main_generate` instead.
