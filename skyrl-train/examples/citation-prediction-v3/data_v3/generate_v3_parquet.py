"""Step 5: Generate SkyRL parquet from v3 corpus.

Converts the v3 subsection corpus into SkyRL-compatible parquet format.
Supports prompt versions (v1/v2) and intro ablation.

Uses env_class: citation_prediction_v2 — no env changes needed.

Usage:
    python generate_v3_parquet.py \
        --input subsection_corpus_v3.json \
        --output_dir /path/to/v3_parquets/

    # v2 prompt (improved):
    python generate_v3_parquet.py \
        --input subsection_corpus_v3.json \
        --output_dir /path/to/v3_parquets_v2/ \
        --prompt_version v2
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# v1 system prompt (condensed original)
SYSTEM_PROMPT_V1 = (
    "You are a research assistant helping an author find papers to cite in their Related Work section. "
    "You have access to a semantic search engine over arxiv. "
    "Results are formatted as: [arxiv:ID] \"Title\" Authors... Abstract...\n\n"
    "Use <think>...</think> to reason, <search>query</search> to search (use descriptive keywords, "
    "NOT author names or IDs), and <citation>arxiv_id</citation> to cite. "
    "Cite papers as you find them — all <citation> tags count toward your predictions. "
    "Write <done></done> when finished.\n\n"
    "Only cite IDs from search results. Stay within the citation budget shown in the prompt — "
    "exceeding it results in zero reward.\n\n"
    "Example:\n"
    "<think>Search for transfer learning methods.</think>\n"
    "<search>transfer learning pretrained language models</search>\n"
    "[Results in <information> tags]\n"
    "<think>Doc 3 matches. Need domain adaptation papers too.</think>\n"
    "<citation>XXXX.XXXXX</citation>\n"
    "<search>domain adaptation distribution shift</search>\n"
    "[Results]\n"
    "<citation>YYYY.YYYYY</citation>\n"
    "<done></done>"
)

# v2 system prompt: no budget (moved to user prompt), no-hallucinate instruction,
# example shows post-search reasoning about whether results match
SYSTEM_PROMPT_V2 = (
    "You are a research assistant helping an author find papers to cite in their Related Work section. "
    "You have access to a semantic search engine over arxiv. "
    "Results are formatted as: [arxiv:ID] \"Title\" Authors... Abstract...\n\n"
    "Use <think>...</think> to reason, <search>query</search> to search (use descriptive keywords, "
    "NOT author names or IDs), and <citation>arxiv_id</citation> to cite. "
    "Write <done></done> when finished.\n\n"
    "Important:\n"
    "- ONLY cite arxiv IDs that appear in search results. Never fabricate or guess IDs.\n"
    "- Do NOT speculate about specific arxiv IDs or paper metadata in your thinking — "
    "only reference papers after you have seen them in search results.\n"
    "- After each search, carefully check which results actually match the subtopic "
    "before citing. Not every result is relevant.\n\n"
    "Example:\n"
    "<think>Search for transfer learning methods.</think>\n"
    "<search>transfer learning pretrained language models</search>\n"
    "[Results in <information> tags]\n"
    "<think>Doc 3 is about pre-trained models for NLP — matches. "
    "Doc 7 is about vision transformers — not relevant. "
    "Still need domain adaptation papers.</think>\n"
    "<citation>XXXX.XXXXX</citation>\n"
    "<search>domain adaptation distribution shift</search>\n"
    "[Results]\n"
    "<think>Doc 1 addresses domain shift in NLP — relevant.</think>\n"
    "<citation>YYYY.YYYYY</citation>\n"
    "<done></done>"
)

# v3 system prompt: shorter think blocks, inter-turn reasoning, cite-as-you-go,
# no hallucinating IDs in thinking
SYSTEM_PROMPT_V3 = (
    "You are a research assistant helping an author find papers to cite in their Related Work section. "
    "You have access to a semantic search engine over arxiv. "
    "Results are formatted as: [arxiv:ID] \"Title\" Authors... Abstract...\n\n"
    "Use <think>...</think> to reason, <search>query</search> to search (use descriptive keywords, "
    "NOT author names or IDs), and <citation>arxiv_id</citation> to cite. "
    "Write <done></done> when finished.\n\n"
    "Guidelines:\n"
    "- Think briefly before searching. Do NOT try to recall specific arxiv IDs or paper details from memory — use search to find them.\n"
    "- After each search, reason about which results match the subtopic before continuing. "
    "If a result looks like a strong match for the Related Work, cite it right away before your next search.\n"
    "- Only cite IDs that appeared in search results. Stay within the citation budget shown in the prompt.\n\n"
    "Example:\n"
    "<think>I need papers on transfer learning for NLP.</think>\n"
    "<search>transfer learning pretrained language models</search>\n"
    "[Results in <information> tags]\n"
    "<think>Doc 3 is about pre-trained models for NLP — good match.</think>\n"
    "<citation>XXXX.XXXXX</citation>\n"
    "<search>domain adaptation distribution shift</search>\n"
    "[Results]\n"
    "<think>Doc 1 addresses domain shift — relevant.</think>\n"
    "<citation>YYYY.YYYYY</citation>\n"
    "<done></done>"
)

SYSTEM_PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2, "v3": SYSTEM_PROMPT_V3}


def format_user_prompt(
    title: str,
    abstract: str,
    introduction: str,
    rich_query: str,
    num_citations: int,
    include_intro: bool = True,
) -> str:
    """Build user prompt with the rich query as a natural question."""
    budget = num_citations * 2
    parts = [
        f"Paper title: {title}",
        f"\nAbstract:\n{abstract}",
    ]
    if include_intro:
        parts.append(f"\nIntroduction:\n{introduction}")
    parts.extend([
        f"\n{rich_query}",
        f"\nCitation budget: at most {budget} citations (you may cite fewer — only cite papers you are confident about). "
        "Exceeding this budget results in zero reward.",
        "\nIdentify all papers that would be cited in this Related Work subtopic. "
        "Search for them and report their arxiv IDs using <citation> tags.",
    ])
    return "\n".join(parts)


def build_skyrl_row(entry: dict, idx: int, include_intro: bool = True,
                    prompt_version: str = "v1") -> dict:
    """Convert a corpus entry into a SkyRL-format row."""
    num_citations = len(entry["citation_ids"])
    user_content = format_user_prompt(
        entry["title"],
        entry["abstract"],
        entry["introduction"],
        entry["rich_query"],
        num_citations,
        include_intro=include_intro,
    )

    return {
        "data_source": "citation_prediction_v3_iclr",
        "prompt": json.dumps([
            {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
            {"role": "user", "content": user_content},
        ]),
        "ability": "citation_prediction_v2",
        "env_class": "citation_prediction_v2",
        "reward_spec": json.dumps({
            "ground_truth": {"targets": entry["citation_ids"]}
        }),
        "extra_info": json.dumps({
            "index": idx,
            "need_tools_kwargs": True,
            "question": entry["rich_query"],
            "split": entry["split"],
            "tools_kwargs": {
                "citation_prediction_v2": {
                    "create_kwargs": {
                        "ground_truth": {"targets": entry["citation_ids"]},
                        "question": entry["rich_query"],
                        "data_source": "citation_prediction_v3_iclr",
                    }
                }
            },
        }),
        "metadata": json.dumps({
            "paper_id": entry["paper_id"],
            "subsection_heading": entry["subsection_heading"],
            "rich_query": entry["rich_query"],
            "num_citations": num_citations,
            "target_ids": entry["citation_ids"],
        }),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SkyRL parquet from v3 corpus (Step 5 of v3 pipeline)."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to subsection_corpus_v3.json from Step 4")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--min_citations", type=int, default=2,
                        help="Minimum citations per subsection (default: 2)")
    parser.add_argument("--no_intro", action="store_true",
                        help="Omit introduction from user prompt (ablation)")
    parser.add_argument("--prompt_version", type=str, default="v1",
                        choices=["v1", "v2", "v3"],
                        help="Prompt version: v1 (original condensed), v2 (no-hallucinate), or v3 (inter-turn reasoning)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_intro = not args.no_intro
    prompt_version = args.prompt_version
    print(f"Prompt version: {prompt_version}, include_intro: {include_intro}")

    # Load corpus
    print(f"Loading corpus from {args.input}...")
    with open(args.input) as f:
        entries = json.load(f)
    print(f"  Loaded {len(entries)} entries")

    # Filter by min citations
    entries = [e for e in entries if len(e["citation_ids"]) >= args.min_citations]
    print(f"  After min_citations={args.min_citations} filter: {len(entries)}")

    # Group by split
    by_split: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_split[entry["split"]].append(entry)

    # Build rows and write parquet per split
    print("\nWriting parquet files...")
    for split_name, split_entries in by_split.items():
        rows = [build_skyrl_row(e, i, include_intro=include_intro, prompt_version=prompt_version) for i, e in enumerate(split_entries)]

        df = pd.DataFrame(rows)
        out_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {split_name}: {len(rows)} examples -> {out_path}")

    # Summary
    total = sum(len(v) for v in by_split.values())
    print(f"\n=== Summary ===")
    print(f"Total examples: {total}")
    for split_name in ["train", "validation", "test"]:
        count = len(by_split.get(split_name, []))
        print(f"  {split_name}: {count}")

    # Citation count distribution
    if entries:
        counts = [len(e["citation_ids"]) for e in entries]
        print(f"\nCitation count distribution:")
        print(f"  Min: {min(counts)}, Max: {max(counts)}, Mean: {sum(counts)/len(counts):.1f}")
        dist = Counter(counts)
        for k in sorted(dist.keys())[:15]:
            print(f"  {k} citations: {dist[k]} examples")

    # Show example prompt
    if entries:
        example = entries[0]
        example_prompt = format_user_prompt(
            example["title"], example["abstract"], example["introduction"],
            example["rich_query"], len(example["citation_ids"]),
            include_intro=include_intro,
        )
        sys_words = len(SYSTEM_PROMPTS[prompt_version].split())
        user_words = len(example_prompt.split())
        print(f"\nPrompt sizes (example): system={sys_words} words, user={user_words} words")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
