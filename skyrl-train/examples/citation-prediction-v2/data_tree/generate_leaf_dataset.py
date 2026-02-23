#!/usr/bin/env python3
"""Generate SkyRL v2-compatible parquet from tree leaf nodes.

Reads trees.json, extracts all leaf nodes, and creates a parquet file
with the same schema as build_citation_dataset_v2.py output. The resulting
dataset can be used directly with the citation_prediction_v2 environment
for GRPO training.

Usage:
    python generate_leaf_dataset.py \
        --trees trees.json \
        --output leaves.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from tree_utils import load_trees

# Same system prompt as v2
SYSTEM_PROMPT = (
    "You are a research assistant. Given a paper's title, abstract, introduction, "
    "and a Related Work subsection heading, your job is to identify ALL papers "
    "cited in that subsection by their arxiv IDs.\n\n"
    "You have access to a semantic search engine over an arxiv paper database. "
    "Each result is formatted as: [arxiv:ID] \"Title\" followed by authors and abstract. "
    "The search engine matches by meaning, so use descriptive topic keywords — "
    "do NOT search for author names, citation strings, or arxiv IDs.\n\n"
    "You must reason inside <think> and </think> tags. "
    "To search, write <search> query </search> and results appear in "
    "<information> tags. You MUST search at least once before citing.\n\n"
    "When you find a relevant paper in the search results, cite it immediately with "
    "<citation>arxiv_id</citation>. You can include multiple IDs in one tag: "
    "<citation>XXXX.XXXXX, YYYY.YYYYY</citation>. Cite as you go — every <citation> tag "
    "across the entire trajectory counts toward your final predictions. Do not wait "
    "until the end to cite.\n\n"
    "You have a citation budget (shown in the prompt). You may cite fewer papers than "
    "the budget — only cite papers you are confident about. However, exceeding the "
    "budget results in zero reward.\n\n"
    "When you have found all cited papers, write <done></done> to finish.\n\n"
    "Important guidelines:\n"
    "- Put ONLY search keywords inside <search> tags. Do all reasoning inside <think> tags.\n"
    "- ONLY cite arxiv IDs that appear in search results. Never guess or make up IDs.\n"
    "- The introduction mentions author names (e.g., \"Smith et al.\") — these are not "
    "searchable, but the topics and methods they describe ARE. Use the introduction to "
    "identify what topics to search for.\n"
    "- Be persistent: search from multiple angles using different keywords. Rephrase queries, "
    "try specific method names, dataset names, or task descriptions. Use all your turns.\n"
    "- Always close your tags. Every <think> needs </think>, every <search> needs </search>, "
    "every <citation> needs </citation>. Do not leave stray or unclosed tags.\n\n"
    "Example workflow:\n"
    "<think>The subsection is about \"Transfer Learning\". I should search for key topics.</think>\n"
    "<search>transfer learning pretrained language models fine-tuning</search>\n"
    "[Results appear in <information> tags: [arxiv:ID] \"Title\" Authors... Abstract...]\n"
    "<think>Doc 3 looks like a paper on pre-trained models for NLP. I'll cite it. "
    "But I still need to find papers on domain adaptation mentioned in the introduction.</think>\n"
    "<citation>XXXX.XXXXX</citation>\n"
    "<search>domain adaptation neural networks distribution shift</search>\n"
    "[More results appear...]\n"
    "<think>Doc 1 and Doc 4 are both relevant. I haven't found anything about "
    "multi-task learning yet, which the subsection heading also relates to.</think>\n"
    "<citation>YYYY.YYYYY, ZZZZ.ZZZZZ</citation>\n"
    "<search>multi-task learning shared representations</search>\n"
    "[More results...]\n"
    "<think>I've covered the main topics now. Let me finish.</think>\n"
    "<done></done>"
)


def format_user_prompt(
    title: str, abstract: str, introduction: str, heading: str, num_citations: int,
) -> str:
    budget = num_citations * 2
    parts = [
        f"Paper title: {title}",
        f"\nAbstract:\n{abstract}",
        f"\nIntroduction:\n{introduction}",
        f'\nRelated Work subtopic: "{heading}"',
        f"\nCitation budget: at most {budget} citations (you may cite fewer — only cite papers you are confident about). "
        "Exceeding this budget results in zero reward.",
        "\nIdentify all papers that would be cited in this Related Work subtopic. "
        "Search for them and report their arxiv IDs using <citation> tags.",
    ]
    return "\n".join(parts)


def build_skyrl_row(tree, leaf, idx: int) -> dict:
    """Convert a leaf node into a SkyRL v2-format row."""
    user_content = format_user_prompt(
        tree.title, tree.abstract, tree.introduction,
        leaf.heading, len(leaf.citation_ids),
    )

    return {
        "data_source": "citation_prediction_v2_iclr_leaves",
        "prompt": json.dumps([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]),
        "ability": "citation_prediction_v2",
        "env_class": "citation_prediction_v2",
        "reward_spec": json.dumps({
            "ground_truth": {"targets": leaf.citation_ids}
        }),
        "extra_info": json.dumps({
            "index": idx,
            "need_tools_kwargs": True,
            "question": leaf.heading,
            "split": "train",
            "tools_kwargs": {
                "citation_prediction_v2": {
                    "create_kwargs": {
                        "ground_truth": {"targets": leaf.citation_ids},
                        "question": leaf.heading,
                        "data_source": "citation_prediction_v2_iclr_leaves",
                    }
                }
            },
        }),
        "metadata": json.dumps({
            "paper_id": tree.paper_id,
            "subsection_heading": tree.root_heading,
            "leaf_node_id": leaf.node_id,
            "leaf_heading": leaf.heading,
            "tree_depth": leaf.depth,
            "num_citations": len(leaf.citation_ids),
            "target_ids": leaf.citation_ids,
            "best_of_k_recall": leaf.best_of_k_recall,
            "mean_recall": leaf.mean_recall,
        }),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SkyRL v2-compatible parquet from tree leaves."
    )
    parser.add_argument("--trees", type=str, required=True,
                        help="Path to trees.json")
    parser.add_argument("--output", type=str, required=True,
                        help="Output parquet file path")
    parser.add_argument("--min_citations", type=int, default=1,
                        help="Minimum citations per leaf (default: 1)")
    parser.add_argument("--min_recall", type=float, default=None,
                        help="Only include leaves with best-of-k recall >= this")
    parser.add_argument("--max_recall", type=float, default=None,
                        help="Only include leaves with best-of-k recall <= this")
    args = parser.parse_args()

    # Load trees
    print(f"Loading trees from {args.trees}...")
    trees = load_trees(args.trees)
    print(f"Loaded {len(trees)} trees")

    # Extract leaves
    rows = []
    skipped_citations = 0
    skipped_recall_low = 0
    skipped_recall_high = 0

    for tree in trees:
        for leaf in tree.get_leaves():
            if len(leaf.citation_ids) < args.min_citations:
                skipped_citations += 1
                continue
            if args.min_recall is not None and leaf.best_of_k_recall is not None:
                if leaf.best_of_k_recall < args.min_recall:
                    skipped_recall_low += 1
                    continue
            if args.max_recall is not None and leaf.best_of_k_recall is not None:
                if leaf.best_of_k_recall > args.max_recall:
                    skipped_recall_high += 1
                    continue
            rows.append(build_skyrl_row(tree, leaf, len(rows)))

    if not rows:
        print("ERROR: No leaf nodes found matching criteria")
        return

    # Write parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Trees: {len(trees)}")
    print(f"Total leaves: {sum(len(t.get_leaves()) for t in trees)}")
    print(f"Skipped (< {args.min_citations} citations): {skipped_citations}")
    if args.min_recall is not None:
        print(f"Skipped (recall < {args.min_recall}): {skipped_recall_low}")
    if args.max_recall is not None:
        print(f"Skipped (recall > {args.max_recall}): {skipped_recall_high}")
    print(f"Output rows: {len(rows)}")
    print(f"Output: {output_path}")

    # Citation distribution
    cite_counts = [json.loads(r["metadata"])["num_citations"] for r in rows]
    if cite_counts:
        print(f"\nCitation count distribution:")
        print(f"  Min: {min(cite_counts)}, Max: {max(cite_counts)}, "
              f"Mean: {sum(cite_counts)/len(cite_counts):.1f}")

    # Recall distribution
    recall_vals = [json.loads(r["metadata"])["best_of_k_recall"]
                   for r in rows if json.loads(r["metadata"])["best_of_k_recall"] is not None]
    if recall_vals:
        zero_pct = sum(1 for r in recall_vals if r == 0.0) / len(recall_vals) * 100
        print(f"\nRecall distribution:")
        print(f"  Min: {min(recall_vals):.2f}, Max: {max(recall_vals):.2f}, "
              f"Mean: {sum(recall_vals)/len(recall_vals):.2f}")
        print(f"  Zero recall: {zero_pct:.0f}%")


if __name__ == "__main__":
    main()
