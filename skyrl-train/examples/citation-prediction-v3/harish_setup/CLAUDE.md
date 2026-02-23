# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is `examples/citation-prediction-v3/harish_setup` for training **Citation Prediction V3** on Princeton's Della cluster. V3 uses LLM-based (Gemini) subsection splitting to include ALL papers (not just those with explicit subsection headings), generating rich descriptive queries that capture author intent.

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

# Step 5: Generate parquets
uv run --active --frozen python examples/citation-prediction-v3/data_v3/generate_v3_parquet.py \
    --input $OUTPUT_DIR/subsection_corpus_v3.json --output_dir $OUTPUT_DIR
```

## Current Setup Status

**Location**: All data and caches are on scratch filesystem at `/scratch/gpfs/ZHUANGL/hk4638`

| Component | Status | Location |
|-----------|--------|----------|
| V3 training data | 🔨 Build with pipeline above | `.../citation_prediction_v3/train.parquet` |
| V3 validation data | 🔨 Build with pipeline above | `.../citation_prediction_v3/validation.parquet` |
| V3 test data | 🔨 Build with pipeline above | `.../citation_prediction_v3/test.parquet` |
| Arxiv corpus (with IDs) | ✅ Ready (shared with v1/v2) | `.../citation_prediction/arxiv_wikiformat_with_ids.jsonl` |
| Qwen3-4B FAISS index | ✅ Ready (shared with v2) | `.../citation_prediction/qwen3_4b_embed/qwen3_Flat.index` |
| Retriever conda env | ✅ Ready (shared with v2) | `.../conda/envs/retriever` |
| Training venv | ✅ Ready (shared with v2) | `/home/hk4638/SkyRL/skyrl-train/.venv` |
| Qwen3-4B (training model) | ✅ Cached (shared with v2) | `.../huggingface/hub/models--Qwen--Qwen3-4B/...` |
| Qwen3-Embedding-4B | ✅ Cached (shared with v2) | `.../huggingface/hub/models--Qwen--Qwen3-Embedding-4B/` |
| API keys | ✅ In `examples/search/harish_setup/.env` | WANDB_API_KEY + GEMINI_API_KEY |

## Files in harish_setup/

| File | Purpose |
|------|---------|
| `train_citation_prediction.slurm` | **Main SLURM script** - 24-hour training (3 nodes × 4 GPUs) |
| `eval_sweep_job.slurm` | Parameterized sweep eval (2 nodes, 1 hour) |
| `sweep_eval.sh` | Sweep orchestrator — submits one job per (m, n) combo |
| `tree_build_job.slurm` | Tree decomposition of v3 subsections (2 nodes, 8 hours) |
| `sft_trajectories_job.slurm` | SFT trajectory generation with Gemini (1 node, 4 hours) |
| `start_retriever.sh` | Start retriever locally (symlink to v2) |
| `download_model.slurm` | Download model weights (symlink to v2) |
| `dataset_viewer.py` | Gradio dataset inspector (symlink to v2) |
| `base_eval_viewer.py` | Gradio sweep/eval viewer (symlink to v2) |
| `trajectory_viewer.py` | Gradio training trajectory viewer (symlink to v2) |
| `.env` | Not present — uses `examples/search/harish_setup/.env` |
| `.gitignore` | Prevents .env from being committed |
| `CLAUDE.md` | This file |
| `README.md` | Quick start guide |

## Files in data_v3/

| File | Purpose |
|------|---------|
| `extract_all_related_work.py` | Step 1: Extract RW from ALL papers (flat + structured) |
| `filter_by_coverage.py` | Step 2: Filter by arxiv corpus coverage + distribution plots |
| `gemini_split_subsections.py` | Step 3: Gemini LLM splits + rich query generation |
| `build_v3_corpus.py` | Step 4: Assemble v2-compatible corpus with `rich_query` field |
| `generate_v3_parquet.py` | Step 5: SkyRL parquet output |

## Critical: Della Cluster Constraints

Same constraints as v2 — see `examples/citation-prediction-v2/harish_setup/CLAUDE.md` for full details. Key reminders:

1. **No internet on compute nodes** — pre-download everything from login node
2. **Use `module load proxy/default`** for WandB/Gemini API access
3. **Set `NO_PROXY`** for internal cluster traffic (exact hostnames, not wildcards)
4. **Use `uv run --active --frozen`** for Ray and training commands
5. **Set `HF_HUB_OFFLINE=1`** and use local model paths
6. **Set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`** to prevent Ray from packaging working_dir
7. **Use absolute paths** in srun subshells (minimal PATH on compute nodes)
8. **Use `source /usr/licensed/anaconda3/2024.10/etc/profile.d/conda.sh`** for conda init

## How to Run Training

### 1. Build V3 Dataset (login node)
Follow the pipeline commands above.

### 2. Run Sweep Eval (to find optimal m, n)
```bash
cd /home/hk4638/SkyRL/skyrl-train
bash examples/citation-prediction-v3/harish_setup/sweep_eval.sh --examples 50 --dry-run
# Review, then remove --dry-run to submit
bash examples/citation-prediction-v3/harish_setup/sweep_eval.sh --examples 50
```

### 3. View Sweep Results
```bash
conda activate viewer
python examples/citation-prediction-v3/harish_setup/base_eval_viewer.py \
    --evals-dir /scratch/gpfs/ZHUANGL/hk4638/logs/sweep/
```

### 4. Submit Full Training
```bash
sbatch examples/citation-prediction-v3/harish_setup/train_citation_prediction.slurm
```

### 5. Monitor
```bash
squeue -u $USER
tail -f /scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3_<jobid>.out
```

## Training Configuration

Same as v2 — see v2 CLAUDE.md for full parameter table. Key differences:
- `data.train_data` points to v3 parquets at `.../citation_prediction_v3/train.parquet`
- `trainer.project_name` = `skyrl-citation-prediction-v3`
- `data_source` in parquets = `citation_prediction_v3_iclr` (for WandB filtering)

## Architecture

Same as v2. The only change is at the data layer:

```
V3 Data Pipeline → V2-compatible parquet → Generator (vLLM) → CitationPredictionV2Env → GRPO
                                                   ↓
                                             Multi-turn loop:
                                             1. Model reasons in <think>...</think>
                                             2. Model generates <search>query</search>
                                             3. HTTP call to retriever → top-k arxiv docs
                                             4. Model cites with <citation>ID</citation>
                                             5. When done → <done></done>
                                             6. Reward = recall (found citations / total targets)
```

## Output Locations

| Output | Location |
|--------|----------|
| V3 data pipeline outputs | `/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/` |
| Job stdout | `/scratch/gpfs/ZHUANGL/hk4638/logs/citation-prediction-v3_<jobid>.out` |
| Sweep results | `/scratch/gpfs/ZHUANGL/hk4638/logs/sweep/v3_m{m}_n{n}_k{k}/` |
| Tree decomposition | `/scratch/gpfs/ZHUANGL/hk4638/logs/tree_build_v3/` |
| SFT trajectories | `/scratch/gpfs/ZHUANGL/hk4638/logs/sft_trajectories_v3/` |
| Checkpoints | `/scratch/gpfs/ZHUANGL/hk4638/checkpoints/citation-prediction-v3-*/` |
| WandB | https://wandb.ai/*/skyrl-citation-prediction-v3 |
