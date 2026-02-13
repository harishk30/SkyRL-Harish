# Masked Citation Prediction

Train an LLM to identify masked citations in research paper Related Works sections using multi-turn search over an arxiv paper index.

## Problem Description

Given a sentence from a research paper's Related Works section with **exactly one** citation masked (replaced with `[MASKED]`), the model uses `<search>query</search>` to query an arxiv paper index. The environment checks if the correct paper (by arxiv ID) appears in the top-k search results.

- **Match found** → reward = 1.0, episode ends
- **No match** → model is told the paper wasn't found, can refine the query and search again
- **Max turns reached with no match** → reward = 0.0, episode ends

No `<answer>` tags are needed — the reward comes entirely from whether the retriever returns the correct paper.

### How it differs from Search-R1

| Aspect | Search-R1 | Citation Prediction |
|--------|-----------|---------------------|
| Task | Open-domain QA | Identify masked citation |
| Reward trigger | Model outputs `<answer>text</answer>` | Correct paper appears in search results |
| Reward function | Exact match on answer text | Arxiv ID match in retriever results |
| Stop tokens | `</search>`, `</answer>` | `</search>` only |
| Corpus | Wikipedia (wiki-18.jsonl) | Arxiv papers (~1.7M) |
| Index | e5-base-v2 embeddings | Qwen-3 4B embeddings (TBD) |

### Context levels

The amount of surrounding text given with the masked citation is configurable via a **context level** parameter in the data pipeline (`create_dataset.py`). This controls task difficulty — more context makes it easier to identify the cited paper.

| Context Level | What the model sees | Difficulty |
|---------------|---------------------|------------|
| `sentence` | Only the sentence containing `[MASKED]` | Harder — less context to infer the paper |
| `paragraph` | The sentence + the preceding paragraph | Easier — surrounding discussion provides topical clues |

The context level is set at dataset creation time (Workstream 1, Step 4) and stored in the parquet. Different training runs can use different context levels by pointing to different parquet files.

### Example prompts

**Context level: `sentence`**
```
You are given a sentence from a research paper's Related Works section with one citation masked.
Your task is to find the correct cited paper by searching an arxiv database.
Use <search>query</search> to search. Results appear between <information> and </information>.
If the correct paper is found in the results, you succeed. Otherwise, you can refine your
search query and try again.

Sentence: "Recent work by [MASKED] demonstrated that large language models can perform
in-context learning without any parameter updates."
```

**Context level: `paragraph`**
```
You are given a passage from a research paper's Related Works section with one citation masked.
Your task is to find the correct cited paper by searching an arxiv database.
Use <search>query</search> to search. Results appear between <information> and </information>.
If the correct paper is found in the results, you succeed. Otherwise, you can refine your
search query and try again.

Context: "In-context learning has emerged as a powerful paradigm for adapting large language
models to downstream tasks. Several studies have explored why this capability arises in
sufficiently large models and how it relates to implicit gradient descent."

Sentence: "Recent work by [MASKED] demonstrated that large language models can perform
in-context learning without any parameter updates."
```

## Architecture Overview

```
Dataset (parquet)  →  Generator (vLLM)  →  CitationPredictionEnv  →  Rewards  →  GRPO Training
                            ↓
                      Multi-turn loop (max N turns):
                      1. Model generates <search>query</search>
                      2. HTTP call to retriever → top-k arxiv papers
                      3. Env checks if ground-truth arxiv ID is in results
                      4a. If found: reward=1.0, done
                      4b. If not: "Paper not found, try again" observation
                      5. Repeat until found or max_turns reached
```

## Workstreams

### Workstream 1: Data Processing Pipeline

Transform raw ICLR papers into training/validation parquet files with masked citation sentences.

**Scripts** (to be created in `../data/`):

| Script | Purpose | Status |
|--------|---------|--------|
| `extract_related_works.py` | Extract Related Works sections from ICLR papers | TODO |
| `parse_citations.py` | Split into sentences, identify citations, resolve to arxiv IDs | TODO |
| `filter_citations.py` | Remove sentences whose cited arxiv ID isn't in the arxiv database | TODO |
| `create_dataset.py` | Mask citations, build SkyRL-format parquet files | TODO |

**Pipeline**:
1. **Extract**: Parse ICLR papers (LaTeX/JSON/PDF — format TBD) to find Related Works sections
2. **Parse**: Split text into sentences, identify citation markers (`\cite{...}`, `[Author, Year]`, etc.), resolve to arxiv IDs via bibliography
3. **Filter**: Keep only sentences with exactly 1 citation whose arxiv ID exists in the Kaggle arxiv dataset
4. **Create**: Replace citation with `[MASKED]`, select context level (`sentence` or `paragraph`), format as SkyRL training parquet with `env_class: "citation_prediction"`

**Data format** (parquet row):
```python
{
    "data_source": "citation_prediction_iclr",
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<task instructions + masked sentence>"}
    ],
    "ability": "citation_prediction",
    "env_class": "citation_prediction",
    "reward_spec": {"ground_truth": {"target": "2301.12345"}},
    "extra_info": {
        "index": 0,
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "citation_prediction": {
                "create_kwargs": {
                    "ground_truth": {"target": "2301.12345"},
                    "question": "<masked sentence>",
                    "data_source": "citation_prediction_iclr"
                }
            }
        }
    }
}
```

### Workstream 2: Arxiv Embedding Index Construction

Build a FAISS index over arxiv paper abstracts for the retrieval server.

| Step | Description | Status |
|------|-------------|--------|
| Download arxiv metadata | Kaggle dataset (~1.7M papers, ~1.68GB JSON) | TODO |
| Build corpus JSONL | `{"contents": "Title: ... Abstract: ...", "id": "arxiv_id", ...}` | TODO |
| Embed with Qwen-3 0.6B | Pre-existing index available as starting point | Partial |
| Embed with Qwen-3 4B | Build FAISS Flat index (~1.7M vectors) | TODO |
| Update retriever config | Point to new corpus, index, and embedding model | TODO |

**Scripts**:
- `../data/download_arxiv_metadata.py` — Download and process Kaggle arxiv metadata
- `../data/build_faiss_index.py` — Encode abstracts with Qwen-3 4B, build FAISS index
- `../harish_setup/build_index.slurm` — SLURM job for GPU-intensive embedding

**Corpus format** (`arxiv_corpus.jsonl`):
```json
{"contents": "Title: Attention Is All You Need. Abstract: The dominant sequence...", "id": "1706.03762", "title": "Attention Is All You Need", "authors": "Vaswani et al.", "categories": "cs.CL"}
```

The `contents` field is what gets indexed and returned by the retriever. Metadata (title, authors, arxiv_id) helps the model identify papers.

### Workstream 3: Citation Prediction Environment

Create a new `citation_prediction` environment in skyrl-gym.

**Files**:
- `skyrl-gym/skyrl_gym/envs/citation_prediction/__init__.py`
- `skyrl-gym/skyrl_gym/envs/citation_prediction/env.py` — `CitationPredictionEnv(BaseTextEnv)`
- `skyrl-gym/skyrl_gym/envs/citation_prediction/utils.py` — ID extraction and matching helpers
- `skyrl-gym/skyrl_gym/envs/__init__.py` — Add registration

**Key design decisions**:
- Reuses `SearchToolGroup` (retriever API is identical)
- `step()` checks retriever results for ground-truth arxiv ID match after every search
- No `<answer>` tag parsing — reward is purely from retriever result matching
- Stop tokens: only `["</search>"]`
- `_is_done()`: True when correct paper found OR max_turns reached
- `_get_reward()`: 1.0 if found, 0.0 otherwise

### Workstream 4: Training Configuration

Update training scripts for the new environment and dataset.

| File | Changes | Status |
|------|---------|--------|
| `citation_prediction_dataset.py` | Rewrite `process_single_row` for citation format | TODO |
| `run_citation_prediction.sh` | Change env_class, stop tokens, config keys | TODO |
| `harish_setup/*.slurm` | Update data paths, model paths | TODO |

## Data Requirements

| Data | Source | Size | Status |
|------|--------|------|--------|
| ICLR paper corpus | TBD (LaTeX, PDF, or structured JSON) | TBD | Not available |
| Arxiv metadata | [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data) | ~1.68GB JSON, ~1.7M papers | Not downloaded |
| Qwen-3 0.6B index | Pre-built | TBD | Partial |
| Qwen-3 4B index | To be built | TBD | Not started |

## Implementation Order

1. **Workstream 3** (Environment) — small, foundational, unblocks testing with mock data
2. **Workstream 1** (Data pipeline) — depends on ICLR paper format
3. **Workstream 2** (Index) — once arxiv metadata is downloaded
4. **Workstream 4** (Training config) — once data format is settled

## References

- **Cluster setup (Della)**: [`../harish_setup/README.md`](../harish_setup/README.md) — SLURM scripts, scratch paths, retriever setup
- **Cluster constraints**: [`../harish_setup/CLAUDE.md`](../harish_setup/CLAUDE.md) — Offline mode, proxy, FAISS sharding, etc.
- **Base Search-R1 framework**: [`../README.md`](../README.md) — Dataset preparation, retriever environment, training launch
- **SearchEnv reference**: `skyrl-gym/skyrl_gym/envs/search/env.py` — Environment to model `CitationPredictionEnv` after
