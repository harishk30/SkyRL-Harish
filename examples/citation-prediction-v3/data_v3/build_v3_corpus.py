"""Step 4: Assemble v3 subsection corpus from Gemini splits.

Produces a JSON corpus in the same format as v2's subsection_corpus.json,
with the addition of a `rich_query` field per subsection. Loads cited paper
metadata (title, authors, abstract) from the arxiv corpus.

This corpus is a drop-in replacement for v2's subsection_corpus.json:
- adaptive_decompose.py reads it unchanged
- generate_leaf_dataset.py works unchanged
- env.py (citation_prediction_v2) works unchanged

Usage:
    python build_v3_corpus.py \
        --input gemini_subsections.json \
        --arxiv_corpus /path/to/arxiv_wikiformat_with_ids.jsonl \
        --output subsection_corpus_v3.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import sys

# Reuse corpus parsing from v2 data_tree
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "citation-prediction-v2" / "data_tree"))

from build_subsection_corpus import (
    parse_corpus_entry,
    load_cited_paper_metadata,
    build_citation_sentence_map,
)

# Also need find_all_citations for sentence mapping
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "citation-prediction-v2" / "data"))
from build_citation_dataset_v2 import find_all_citations


def build_subsection_entries(paper: dict) -> list[dict]:
    """Convert a paper with Gemini subsets into corpus entries.

    Each subset becomes one corpus entry (= one training example).
    """
    entries = []
    for i, subset in enumerate(paper["subsets"]):
        # Generate a heading from the query (first sentence, truncated)
        query = subset["query"]
        # Use first ~80 chars as heading for compatibility
        heading = query[:80].rstrip(".")
        if len(query) > 80:
            heading += "..."

        # Build paragraph text: extract sentences from RW that cite papers in this subset
        # For now, use the full RW text (Gemini already partitioned by citations)
        paragraph_text = paper["related_work_text"]

        entries.append({
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "abstract": paper["abstract"],
            "introduction": paper["introduction"],
            "subsection_heading": heading,
            "rich_query": query,
            "paragraph_text": paragraph_text,
            "full_related_work_text": paper["related_work_text"],
            "citation_ids": subset["citation_ids"],
            "citation_sentence_map": {},  # Filled in later
            "split": paper["split"],
            "cited_papers": {},  # Filled in later
        })

    return entries


def build_sentence_maps(
    entries: list[dict],
    papers_lookup: dict[str, dict],
) -> None:
    """Build citation_sentence_map for each entry by scanning the RW text.

    Modifies entries in-place.
    """
    for entry in entries:
        cite_ids_set = set(entry["citation_ids"])
        rw_text = entry["full_related_work_text"]

        # Build a simple (last_name, year) -> arxiv_id mapping from entry's citation_ids
        # We can't do this without ref_lookup, so we just store empty maps.
        # The sentence map is optional for training; the key data is citation_ids.
        # It's used by the viewer / debugger.
        entry["citation_sentence_map"] = {}


def main():
    parser = argparse.ArgumentParser(
        description="Assemble v3 subsection corpus (Step 4 of v3 pipeline)."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to gemini_subsections.json from Step 3")
    parser.add_argument("--arxiv_corpus", type=str, required=True,
                        help="Path to arxiv_wikiformat_with_ids.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Output corpus JSON file")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "validation", "test"],
                        help="Only process this split")
    args = parser.parse_args()

    # Load Gemini results
    print(f"Loading Gemini subsections from {args.input}...")
    with open(args.input) as f:
        papers = json.load(f)
    if args.split:
        papers = [p for p in papers if p["split"] == args.split]
    print(f"  {len(papers)} papers with {sum(len(p['subsets']) for p in papers)} subsets")

    # Build corpus entries
    print("Building corpus entries...")
    all_entries = []
    papers_lookup = {p["paper_id"]: p for p in papers}

    for paper in papers:
        entries = build_subsection_entries(paper)
        all_entries.extend(entries)

    print(f"  {len(all_entries)} total entries")

    # Collect all unique cited arxiv IDs
    all_cited_ids = set()
    for entry in all_entries:
        all_cited_ids.update(entry["citation_ids"])
    print(f"  {len(all_cited_ids)} unique cited papers")

    # Load cited paper metadata from arxiv corpus
    print(f"Loading cited paper metadata from {args.arxiv_corpus}...")
    cited_metadata = load_cited_paper_metadata(args.arxiv_corpus, all_cited_ids)

    # Attach cited_papers metadata to each entry
    for entry in all_entries:
        entry["cited_papers"] = {
            cid: cited_metadata[cid]
            for cid in entry["citation_ids"]
            if cid in cited_metadata
        }

    # Build sentence maps (optional, for viewer compatibility)
    build_sentence_maps(all_entries, papers_lookup)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Papers: {len(papers)}")
    print(f"Total subsections: {len(all_entries)}")
    print(f"Unique cited papers: {len(all_cited_ids)}")
    print(f"Cited papers with metadata: {len(cited_metadata)}/{len(all_cited_ids)} "
          f"({len(cited_metadata)/max(len(all_cited_ids),1)*100:.0f}%)")

    # Split breakdown
    split_counts = defaultdict(int)
    for e in all_entries:
        split_counts[e["split"]] += 1
    print(f"By split: {dict(split_counts)}")

    # Citation count distribution
    if all_entries:
        counts = [len(e["citation_ids"]) for e in all_entries]
        print(f"\nCitation count distribution:")
        print(f"  Min: {min(counts)}, Max: {max(counts)}, "
              f"Mean: {sum(counts)/len(counts):.1f}")
        print(f"  Median: {sorted(counts)[len(counts)//2]}")

    # Rich query length distribution
    if all_entries:
        query_lens = [len(e["rich_query"].split()) for e in all_entries]
        print(f"\nRich query word count distribution:")
        print(f"  Min: {min(query_lens)}, Max: {max(query_lens)}, "
              f"Mean: {sum(query_lens)/len(query_lens):.1f}")

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
