"""Step 1: Extract Related Work text from ALL papers (including flat/unstructured).

Unlike v2 which only processes papers with explicit subsection headings in
Related Work, this extracts the full RW text from every paper regardless of
structure. Also counts total vs resolvable citations for coverage filtering.

Usage:
    python extract_all_related_work.py \
        --metadata_csv /path/to/massive_metadata.csv \
        --split_dir /path/to/iclr_..._v6 \
        --output all_papers_rw.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import sys

import pandas as pd

# Add v2 data/ directory to path for reusable functions
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "citation-prediction-v2" / "data"))

from build_citation_dataset_v2 import (
    build_ref_lookup,
    find_all_citations,
    parse_single_citation,
    extract_introduction,
    _extract_abstract,
    load_split_ids,
    _is_heading,
    _get_section_num,
    _is_subsection_of,
    _TOP_SECTION_RE,
    _SUBSECTION_RE,
    CITE_GROUP_RE,
    INLINE_CITE_RE,
)


# ---------------------------------------------------------------------------
# Expanded Related Work heading detection
# ---------------------------------------------------------------------------

_RW_KEYWORDS = [
    "related work",
    "related works",
    "prior work",
    "previous work",
    "literature review",
    "background and related",
]


def _is_related_work_heading(text: str) -> bool:
    """Check if heading text refers to a Related Work section."""
    text_lower = text.lower().strip()
    # Strip leading section number (e.g. "2 RELATED WORK" -> "related work")
    stripped = re.sub(r"^\d+[\.\d]*\s+", "", text_lower)
    return any(kw in stripped for kw in _RW_KEYWORDS)


# ---------------------------------------------------------------------------
# Full RW extraction (works for both structured and flat)
# ---------------------------------------------------------------------------

def extract_related_work_full(content_list: list[dict]) -> dict | None:
    """Extract the full Related Work section text, preserving any subsection headings.

    Unlike v2's extract_related_work_subsections(), this does NOT return None
    for flat/unstructured RW sections. It collects all paragraphs between the
    RW heading and the next top-level section.

    Returns:
        {
            "text": str,  # Full RW text with ### subsection markers
            "existing_subsection_headings": list[str],  # e.g. ["2.1 Transfer Learning"]
        }
        or None if no Related Work section found.
    """
    # Find the Related Work heading
    rw_idx = None
    rw_num = None
    for i, item in enumerate(content_list):
        if not _is_heading(item):
            continue
        text = item.get("text", "").strip()
        if _is_related_work_heading(text):
            rw_idx = i
            rw_num = _get_section_num(text)
            break

    if rw_idx is None:
        return None

    # Collect all content until next top-level section
    parts = []
    subsection_headings = []

    for j in range(rw_idx + 1, len(content_list)):
        item = content_list[j]

        if _is_heading(item):
            heading_text = item.get("text", "").strip()

            # Check if this is a subsection of the Related Work section
            if rw_num and _is_subsection_of(heading_text, rw_num):
                parts.append(f"### {heading_text}")
                subsection_headings.append(heading_text)
                continue

            # Not a subsection — new top-level section, stop
            break

        if item.get("type") == "text":
            text = item.get("text", "").strip()
            if text:
                parts.append(text)

    if not parts:
        return None

    return {
        "text": "\n\n".join(parts),
        "existing_subsection_headings": subsection_headings,
    }


# ---------------------------------------------------------------------------
# Citation counting
# ---------------------------------------------------------------------------

def count_citations_in_text(
    text: str,
    ref_lookup: dict[tuple[str, str], str],
) -> dict:
    """Count total citations and resolvable citations in text.

    Returns:
        {
            "total_citation_strings": list[str],  # all "(Author, Year)" strings found
            "resolvable_citations": list[tuple[str, str]],  # (last_name, year) pairs resolved
            "all_citation_ids": list[str],  # unique resolved arxiv IDs
        }
    """
    all_citation_strings = []
    all_name_year_pairs = set()  # unique (last_name, year) pairs parsed
    seen_ids = set()
    all_ids = []
    citation_display_map = {}  # arxiv_id -> "LastName, Year" display string

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        citations = find_all_citations(sentence)
        for cite in citations:
            all_citation_strings.append(cite["original"])
            for last_name, year in cite["parsed"]:
                all_name_year_pairs.add((last_name, year))
                arxiv_id = ref_lookup.get((last_name, year))
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    all_ids.append(arxiv_id)
                    citation_display_map[arxiv_id] = f"{last_name}, {year}"

    return {
        "total_citation_strings": all_citation_strings,
        "total_unique_citations": len(all_name_year_pairs),  # all unique (name, year) pairs
        "all_citation_ids": all_ids,  # only those resolved to arxiv IDs
        "citation_display_map": citation_display_map,  # arxiv_id -> display string
    }


# ---------------------------------------------------------------------------
# Paper processing
# ---------------------------------------------------------------------------

def process_paper(
    submission_id: str,
    title: str,
    abstract: str | None,
    content_list: list[dict],
    split_name: str,
    stats: dict,
) -> dict | None:
    """Process a single paper and return its RW extraction result."""
    ref_lookup = build_ref_lookup(content_list)
    if not ref_lookup:
        stats["no_arxiv_refs"] += 1
        return None

    if not title:
        stats["no_title"] += 1
        return None

    introduction = extract_introduction(content_list)
    if not introduction:
        stats["no_introduction"] += 1
        return None

    rw_result = extract_related_work_full(content_list)
    if rw_result is None:
        stats["no_related_work"] += 1
        return None

    # Count citations in the RW text
    cite_info = count_citations_in_text(rw_result["text"], ref_lookup)

    if len(cite_info["all_citation_ids"]) < 1:
        stats["no_resolvable_citations"] += 1
        return None

    stats["success"] += 1

    return {
        "paper_id": submission_id,
        "title": title,
        "abstract": abstract or "",
        "introduction": introduction,
        "related_work_text": rw_result["text"],
        "existing_subsection_headings": rw_result["existing_subsection_headings"],
        "total_citations_in_text": len(cite_info["total_citation_strings"]),
        "citation_strings": cite_info["total_citation_strings"],
        "total_unique_citations": cite_info["total_unique_citations"],
        "resolvable_citations": len(cite_info["all_citation_ids"]),
        "all_citation_ids": cite_info["all_citation_ids"],
        "citation_display_map": cite_info["citation_display_map"],
        "split": split_name,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Related Work from ALL papers (Step 1 of v3 pipeline)."
    )
    parser.add_argument(
        "--metadata_csv", type=str, required=True,
        help="Path to massive_metadata.csv",
    )
    parser.add_argument(
        "--split_dir", type=str, required=True,
        help="Prefix of split directories (appends _train, _validation, _test)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON file path (all_papers_rw.json)",
    )
    args = parser.parse_args()

    # Load split IDs and abstracts
    print("Loading split submission IDs and abstracts...")
    split_mapping, abstracts = load_split_ids(args.split_dir)
    split_counts = defaultdict(int)
    for s in split_mapping.values():
        split_counts[s] += 1
    print(f"  Loaded {len(split_mapping)} submission IDs: {dict(split_counts)}")
    print(f"  Loaded {len(abstracts)} abstracts ({len(abstracts)/len(split_mapping)*100:.1f}% coverage)")

    # Process papers
    print("Processing papers from CSV...")
    all_results = []
    stats = {
        "papers_processed": 0,
        "no_arxiv_refs": 0,
        "no_title": 0,
        "no_introduction": 0,
        "no_related_work": 0,
        "no_resolvable_citations": 0,
        "success": 0,
    }

    for chunk in pd.read_csv(args.metadata_csv, chunksize=500):
        for _, row in chunk.iterrows():
            sid = str(row.get("submission_id", ""))
            if sid not in split_mapping:
                continue

            split_name = split_mapping[sid]
            stats["papers_processed"] += 1

            title = str(row.get("title", "")).strip()

            try:
                content_list = json.loads(row["content_list_json"])
            except (json.JSONDecodeError, TypeError):
                continue

            abstract = abstracts.get(sid)
            result = process_paper(sid, title, abstract, content_list, split_name, stats)
            if result is not None:
                all_results.append(result)

        if stats["papers_processed"] % 2000 == 0 and stats["papers_processed"] > 0:
            print(f"  Processed {stats['papers_processed']} papers, {len(all_results)} with RW so far...")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Papers in splits: {len(split_mapping)}")
    print(f"Papers processed (found in CSV): {stats['papers_processed']}")
    print(f"  No arxiv refs: {stats['no_arxiv_refs']}")
    print(f"  No title: {stats['no_title']}")
    print(f"  No introduction: {stats['no_introduction']}")
    print(f"  No Related Work section: {stats['no_related_work']}")
    print(f"  No resolvable citations: {stats['no_resolvable_citations']}")
    print(f"  Success (have RW + citations): {stats['success']}")

    # Split breakdown
    split_counts_result = defaultdict(int)
    for r in all_results:
        split_counts_result[r["split"]] += 1
    print(f"\nBy split: {dict(split_counts_result)}")

    # Has-subsections breakdown
    has_sub = sum(1 for r in all_results if r["existing_subsection_headings"])
    flat = sum(1 for r in all_results if not r["existing_subsection_headings"])
    print(f"Papers with subsection headings: {has_sub}")
    print(f"Papers with flat RW (no subsections): {flat}")

    # Citation count distribution
    if all_results:
        counts = [r["resolvable_citations"] for r in all_results]
        print(f"\nResolvable citation count distribution:")
        print(f"  Min: {min(counts)}, Max: {max(counts)}, Mean: {sum(counts)/len(counts):.1f}")
        print(f"  Median: {sorted(counts)[len(counts)//2]}")

    print(f"\nOutput: {output_path} ({len(all_results)} papers)")


if __name__ == "__main__":
    main()
