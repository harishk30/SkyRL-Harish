"""Extract subsection corpus with raw paragraph text, citation-sentence mappings,
full Related Work section text, and cited paper metadata from the arxiv corpus.

Reuses parsing logic from build_citation_dataset_v2.py. Additionally stores:
  - Raw paragraph text per subsection
  - Per-citation sentence mapping: {arxiv_id: [sentence_text_1, ...]}
  - Full Related Work section text (all subsections concatenated with headings)
  - Cited paper metadata: {arxiv_id: {title, authors, abstract}} from arxiv corpus

Output: subsection_corpus.json — one entry per subsection.

Usage:
    python build_subsection_corpus.py \
        --metadata_csv /path/to/massive_metadata.csv \
        --split_dir /path/to/iclr_..._v6 \
        --arxiv_corpus /path/to/arxiv_wikiformat_with_ids.jsonl \
        --output subsection_corpus.json \
        --split train
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import sys

import pandas as pd

# Add the data/ directory to path so we can import from build_citation_dataset_v2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from build_citation_dataset_v2 import (
    build_ref_lookup,
    find_all_citations,
    collect_subsection_citations,
    extract_related_work_subsections,
    extract_introduction,
    _extract_abstract,
    load_split_ids,
)


# ---------------------------------------------------------------------------
# Arxiv corpus parsing
# ---------------------------------------------------------------------------

# Parse: [arxiv:ID] "Title"\nAuthors:...\n\nAbstract:...
_CORPUS_TITLE_RE = re.compile(r'^\[arxiv:[^\]]+\]\s*"(.+?)"', re.DOTALL)
_CORPUS_AUTHORS_RE = re.compile(r'\nAuthors?:\s*(.+?)(?:\n\n|\nAbstract:)', re.DOTALL)
_CORPUS_ABSTRACT_RE = re.compile(r'\nAbstract:\s*(.+?)(?:\n\nDOI:|\Z)', re.DOTALL)


def parse_corpus_entry(contents: str) -> dict:
    """Parse title, authors, abstract from a corpus entry's contents field."""
    result = {}

    title_match = _CORPUS_TITLE_RE.search(contents)
    if title_match:
        result["title"] = title_match.group(1).strip()

    authors_match = _CORPUS_AUTHORS_RE.search(contents)
    if authors_match:
        result["authors"] = authors_match.group(1).strip()

    abstract_match = _CORPUS_ABSTRACT_RE.search(contents)
    if abstract_match:
        result["abstract"] = abstract_match.group(1).strip()

    return result


def load_cited_paper_metadata(
    corpus_path: str, needed_ids: set[str],
) -> dict[str, dict]:
    """Scan arxiv corpus JSONL and extract metadata for needed paper IDs.

    Returns {arxiv_id: {"title": ..., "authors": ..., "abstract": ...}}.
    """
    metadata = {}
    found = 0
    total = len(needed_ids)

    print(f"  Scanning corpus for {total} cited paper IDs...")
    with open(corpus_path) as f:
        for line_num, line in enumerate(f):
            if found >= total:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = entry.get("id", "")
            if doc_id in needed_ids:
                parsed = parse_corpus_entry(entry.get("contents", ""))
                if parsed:
                    metadata[doc_id] = parsed
                    found += 1

            if (line_num + 1) % 500_000 == 0:
                print(f"    Scanned {line_num + 1} lines, found {found}/{total}")

    print(f"  Found metadata for {found}/{total} cited papers")
    return metadata


# ---------------------------------------------------------------------------
# Full Related Work text extraction
# ---------------------------------------------------------------------------

def extract_full_related_work_text(subsections: list[dict]) -> str:
    """Build full Related Work section text from subsections with headings."""
    parts = []
    for sub in subsections:
        parts.append(f"### {sub['heading']}")
        parts.append("\n\n".join(sub["paragraphs"]))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Citation-sentence mapping
# ---------------------------------------------------------------------------

def build_citation_sentence_map(
    paragraphs: list[str],
    ref_lookup: dict[tuple[str, str], str],
) -> dict[str, list[str]]:
    """Map each cited arxiv ID to the sentences that cite it.

    Returns {arxiv_id: [sentence_1, sentence_2, ...]}.
    """
    cite_map: dict[str, list[str]] = defaultdict(list)

    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for sentence in sentences:
            citations = find_all_citations(sentence)
            for cite in citations:
                for last_name, year in cite["parsed"]:
                    arxiv_id = ref_lookup.get((last_name, year))
                    if arxiv_id:
                        if sentence not in cite_map[arxiv_id]:
                            cite_map[arxiv_id].append(sentence)

    return dict(cite_map)


# ---------------------------------------------------------------------------
# Paper processing
# ---------------------------------------------------------------------------

def process_paper_corpus(
    submission_id: str,
    title: str,
    abstract: str | None,
    content_list: list[dict],
    split_name: str,
    stats: dict,
) -> list[dict]:
    """Process a single paper and return subsection corpus entries."""
    ref_lookup = build_ref_lookup(content_list)
    if not ref_lookup:
        stats["no_arxiv_refs"] += 1
        return []

    if not title:
        stats["no_title"] += 1
        return []

    introduction = extract_introduction(content_list)
    if not introduction:
        stats["no_introduction"] += 1
        return []

    subsections = extract_related_work_subsections(content_list)
    if subsections is None:
        stats["no_subsections"] += 1
        return []

    # Build full Related Work section text (shared across all subsections of this paper)
    full_rw_text = extract_full_related_work_text(subsections)

    entries = []
    for sub in subsections:
        citation_ids = collect_subsection_citations(sub["paragraphs"], ref_lookup)

        # Require at least 2 resolvable citations (same as v2 dataset)
        if len(citation_ids) < 2:
            stats["too_few_citations"] += 1
            continue

        paragraph_text = "\n\n".join(sub["paragraphs"])
        citation_sentence_map = build_citation_sentence_map(
            sub["paragraphs"], ref_lookup
        )

        entries.append({
            "paper_id": submission_id,
            "title": title,
            "abstract": abstract or "",
            "introduction": introduction,
            "subsection_heading": sub["heading"],
            "paragraph_text": paragraph_text,
            "full_related_work_text": full_rw_text,
            "citation_ids": citation_ids,
            "citation_sentence_map": citation_sentence_map,
            "split": split_name,
        })

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract subsection corpus for tree decomposition."
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
        "--arxiv_corpus", type=str, required=True,
        help="Path to arxiv_wikiformat_with_ids.jsonl",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        choices=["train", "validation", "test"],
        help="Only process this split (default: all splits)",
    )
    args = parser.parse_args()

    # Step 1: Load split IDs and abstracts
    print("Loading split submission IDs and abstracts...")
    split_mapping, abstracts = load_split_ids(args.split_dir)
    split_counts = defaultdict(int)
    for s in split_mapping.values():
        split_counts[s] += 1
    print(f"  Loaded {len(split_mapping)} submission IDs: {dict(split_counts)}")

    # Step 2: Process papers to get subsection entries
    print("Processing papers from CSV...")
    all_entries = []
    stats = {
        "papers_processed": 0,
        "no_arxiv_refs": 0,
        "no_title": 0,
        "no_introduction": 0,
        "no_subsections": 0,
        "too_few_citations": 0,
    }

    for chunk in pd.read_csv(args.metadata_csv, chunksize=500):
        for _, row in chunk.iterrows():
            sid = str(row.get("submission_id", ""))
            if sid not in split_mapping:
                continue

            split_name = split_mapping[sid]
            if args.split and split_name != args.split:
                continue

            stats["papers_processed"] += 1

            title = str(row.get("title", "")).strip()

            try:
                content_list = json.loads(row["content_list_json"])
            except (json.JSONDecodeError, TypeError):
                continue

            abstract = abstracts.get(sid)
            entries = process_paper_corpus(
                sid, title, abstract, content_list, split_name, stats
            )
            all_entries.extend(entries)

        if stats["papers_processed"] % 1000 == 0 and stats["papers_processed"] > 0:
            print(f"  Processed {stats['papers_processed']} papers, "
                  f"{len(all_entries)} subsections so far...")

    # Step 3: Collect all unique cited arxiv IDs across all entries
    all_cited_ids = set()
    for entry in all_entries:
        all_cited_ids.update(entry["citation_ids"])
    print(f"\nTotal unique cited papers: {len(all_cited_ids)}")

    # Step 4: Load cited paper metadata from arxiv corpus
    print(f"Loading cited paper metadata from {args.arxiv_corpus}...")
    cited_papers = load_cited_paper_metadata(args.arxiv_corpus, all_cited_ids)

    # Step 5: Attach cited_papers metadata to each entry
    # Store as a per-entry dict containing only the papers cited in that subsection
    for entry in all_entries:
        entry["cited_papers"] = {
            cid: cited_papers[cid]
            for cid in entry["citation_ids"]
            if cid in cited_papers
        }

    # Step 6: Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Papers processed: {stats['papers_processed']}")
    print(f"Papers with no arxiv refs: {stats['no_arxiv_refs']}")
    print(f"Papers with no title: {stats['no_title']}")
    print(f"Papers with no introduction: {stats['no_introduction']}")
    print(f"Papers with flat Related Work: {stats['no_subsections']}")
    print(f"Subsections with < 2 citations: {stats['too_few_citations']}")
    print(f"Total subsections: {len(all_entries)}")
    print(f"Cited papers with metadata: {len(cited_papers)}/{len(all_cited_ids)} "
          f"({len(cited_papers)/len(all_cited_ids)*100:.0f}%)" if all_cited_ids else "")
    print(f"Output: {output_path}")

    # Citation count distribution
    if all_entries:
        counts = [len(e["citation_ids"]) for e in all_entries]
        print(f"\nCitation count distribution:")
        print(f"  Min: {min(counts)}, Max: {max(counts)}, Mean: {sum(counts)/len(counts):.1f}")


if __name__ == "__main__":
    main()
