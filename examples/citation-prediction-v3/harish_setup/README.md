# Citation Prediction V3: LLM-Based Subsection Splitting

V3 improves on V2 by using Gemini to split Related Work sections into logical subsections for ALL papers (not just those with explicit subsection headings). This produces richer, more descriptive queries and scales from ~1.6K to ~10K-20K training examples.

## What's New in V3

- **All papers included**: V2 only processed papers with numbered subsections (e.g., "2.1 Transfer Learning"). V3 includes flat/unstructured Related Work sections too.
- **Coverage filtering**: Only keeps papers where >=70% of citations are findable in the arxiv corpus.
- **Gemini splitting**: Single Gemini call per paper partitions citations into thematic subsets and generates rich descriptive queries.
- **Rich queries**: Instead of a bare heading, the model gets "The author is looking to find prior normalization techniques for deep networks. What papers should they cite?"

## Prerequisites

All shared infrastructure from V2 must be set up:
- Retriever conda environment at `/scratch/gpfs/ZHUANGL/hk4638/conda/envs/retriever`
- FAISS index at `/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction/qwen3_4b_embed/qwen3_Flat.index`
- Arxiv corpus at `/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl`
- API keys in `/home/hk4638/SkyRL/skyrl-train/examples/search/harish_setup/.env`

## Quick Start

### 1. Build Dataset (login node, ~1 hour total)

```bash
cd /home/hk4638/SkyRL/skyrl-train
export VIRTUAL_ENV="/home/hk4638/SkyRL/skyrl-train/.venv"
export UV_CACHE_DIR="/scratch/gpfs/ZHUANGL/hk4638/uv_cache"
export UV_LINK_MODE=copy

METADATA=/home/hk4638/scratch/shared/data/massive_metadata.csv
SPLIT_DIR=/home/hk4638/scratch/shared/data/iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6
CORPUS=/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl
OUT=/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3

# Steps 1-5 (see CLAUDE.md for full commands)
uv run --active --frozen python examples/citation-prediction-v3/data_v3/extract_all_related_work.py \
    --metadata_csv $METADATA --split_dir $SPLIT_DIR --output $OUT/all_papers_rw.json

uv run --active --frozen python examples/citation-prediction-v3/data_v3/filter_by_coverage.py \
    --input $OUT/all_papers_rw.json --arxiv_corpus $CORPUS \
    --output $OUT/filtered_papers.json --plot $OUT/coverage_distribution.png

uv run --active --frozen python examples/citation-prediction-v3/data_v3/gemini_split_subsections.py \
    --input $OUT/filtered_papers.json --metadata_csv $METADATA \
    --output $OUT/gemini_subsections.json

uv run --active --frozen python examples/citation-prediction-v3/data_v3/build_v3_corpus.py \
    --input $OUT/gemini_subsections.json --arxiv_corpus $CORPUS \
    --output $OUT/subsection_corpus_v3.json

uv run --active --frozen python examples/citation-prediction-v3/data_v3/generate_v3_parquet.py \
    --input $OUT/subsection_corpus_v3.json --output_dir $OUT
```

### 2. Run Sweep Eval
```bash
bash examples/citation-prediction-v3/harish_setup/sweep_eval.sh --examples 50
```

### 3. Train
```bash
sbatch examples/citation-prediction-v3/harish_setup/train_citation_prediction.slurm
```

### 4. Optional: Tree Decomposition
```bash
# Copy v3 corpus to tree_build_v3 output dir first
cp $OUT/subsection_corpus_v3.json /scratch/gpfs/ZHUANGL/hk4638/logs/tree_build_v3/
sbatch examples/citation-prediction-v3/harish_setup/tree_build_job.slurm
```

## File Structure

```
examples/citation-prediction-v3/
├── data_v3/
│   ├── extract_all_related_work.py     # Step 1: Extract RW from ALL papers
│   ├── filter_by_coverage.py           # Step 2: Filter + distribution plots
│   ├── gemini_split_subsections.py     # Step 3: LLM split + rich queries
│   ├── build_v3_corpus.py              # Step 4: Assemble corpus
│   └── generate_v3_parquet.py          # Step 5: SkyRL parquet
└── harish_setup/
    ├── train_citation_prediction.slurm # 3-node GRPO training
    ├── eval_sweep_job.slurm            # 2-node sweep eval
    ├── sweep_eval.sh                   # Sweep orchestrator
    ├── tree_build_job.slurm            # Tree decomposition
    ├── sft_trajectories_job.slurm      # SFT trajectory generation
    ├── start_retriever.sh              # → symlink to v2
    ├── download_model.slurm            # → symlink to v2
    ├── dataset_viewer.py               # → symlink to v2
    ├── base_eval_viewer.py             # → symlink to v2
    ├── trajectory_viewer.py            # → symlink to v2
    ├── CLAUDE.md                       # Detailed documentation
    └── README.md                       # This file
```

## Downstream Compatibility

V3 uses `env_class: citation_prediction_v2` — no changes needed to:
- `skyrl_gym/envs/citation_prediction_v2/env.py` (reward, actions)
- `adaptive_decompose.py` (tree decomposition)
- `generate_leaf_dataset.py` (leaf parquets)
- `generate_sft_trajectories.py` (SFT data)
- All SLURM training/eval scripts (same config keys)
- All viewers (same parquet schema)
