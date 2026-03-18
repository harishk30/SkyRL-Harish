"""Step 2: Filter papers by arxiv corpus coverage.

For each paper, computes what fraction of its resolvable citations are
actually present in the arxiv corpus. Keeps papers with coverage >= threshold
and at least min_citations resolvable citations.

Generates distribution plots for analysis.

Usage:
    python filter_by_coverage.py \
        --input all_papers_rw.json \
        --arxiv_corpus /path/to/arxiv_wikiformat_with_ids.jsonl \
        --output filtered_papers.json \
        --plot coverage_distribution.png \
        --coverage_threshold 0.7 \
        --min_citations 2
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_corpus_ids(corpus_path: str) -> set[str]:
    """Load all arxiv IDs from the corpus JSONL (just the 'id' field)."""
    ids = set()
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
                doc_id = entry.get("id", "")
                if doc_id:
                    ids.add(doc_id)
            except json.JSONDecodeError:
                continue
            if (i + 1) % 500_000 == 0:
                print(f"  Loaded {i + 1} corpus entries, {len(ids)} unique IDs...")
    print(f"  Total corpus IDs: {len(ids)}")
    return ids


def compute_coverage(paper: dict, corpus_ids: set[str]) -> dict:
    """Compute corpus coverage for a paper and add coverage fields.

    Coverage = (resolved arxiv IDs in corpus) / (total unique citation pairs parsed).
    This measures what fraction of ALL citations in the RW text we can actually
    use for training (i.e., can resolve to an arxiv ID in our corpus).
    """
    citation_ids = paper["all_citation_ids"]
    in_corpus = [cid for cid in citation_ids if cid in corpus_ids]

    # Denominator: total unique (last_name, year) pairs parsed from RW text.
    # This includes citations to non-arxiv papers (books, conference papers
    # without arxiv preprints, etc.) that can never be resolved.
    total_unique = paper.get("total_unique_citations", 0)
    if total_unique == 0:
        # Fallback for old-format data: use resolvable_citations
        total_unique = paper.get("resolvable_citations", 0)

    coverage = len(in_corpus) / total_unique if total_unique > 0 else 0.0

    return {
        **paper,
        "citations_in_corpus": in_corpus,
        "num_in_corpus": len(in_corpus),
        "total_unique_citations": total_unique,
        "coverage": coverage,
    }


def make_plots(papers: list[dict], output_path: str, thresholds: list[float]):
    """Generate coverage distribution plots."""
    coverages = [p["coverage"] for p in papers]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Histogram: # papers at each coverage bin
    ax = axes[0]
    ax.hist(coverages, bins=np.arange(0, 1.05, 0.1), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Coverage (fraction of citations in corpus)")
    ax.set_ylabel("Number of papers")
    ax.set_title(f"Coverage Distribution (N={len(papers)})")
    ax.axvline(0.7, color="red", linestyle="--", label="0.7 threshold")
    ax.legend()

    # 2. CDF curve
    ax = axes[1]
    sorted_cov = np.sort(coverages)
    cdf = np.arange(1, len(sorted_cov) + 1) / len(sorted_cov)
    ax.plot(sorted_cov, cdf, linewidth=2)
    ax.set_xlabel("Coverage threshold")
    ax.set_ylabel("Fraction of papers at or below threshold")
    ax.set_title("Coverage CDF")
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 3. Bar chart: # papers passing at different thresholds
    ax = axes[2]
    counts_at_thresh = []
    for t in thresholds:
        count = sum(1 for c in coverages if c >= t)
        counts_at_thresh.append(count)
    bars = ax.bar([f"{t:.1f}" for t in thresholds], counts_at_thresh,
                  edgecolor="black", alpha=0.7)
    ax.set_xlabel("Coverage threshold")
    ax.set_ylabel("Number of papers passing")
    ax.set_title("Papers Passing Coverage Threshold")
    for bar, count in zip(bars, counts_at_thresh):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Filter papers by arxiv corpus coverage (Step 2 of v3 pipeline)."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to all_papers_rw.json from Step 1")
    parser.add_argument("--arxiv_corpus", type=str, required=True,
                        help="Path to arxiv_wikiformat_with_ids.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Output filtered_papers.json")
    parser.add_argument("--plot", type=str, default=None,
                        help="Output plot path (coverage_distribution.png)")
    parser.add_argument("--coverage_threshold", type=float, default=0.7,
                        help="Minimum coverage to keep a paper (default: 0.7)")
    parser.add_argument("--min_citations", type=int, default=2,
                        help="Minimum resolvable citations (default: 2)")
    args = parser.parse_args()

    # Load papers from Step 1
    print(f"Loading papers from {args.input}...")
    with open(args.input) as f:
        papers = json.load(f)
    print(f"  Loaded {len(papers)} papers")

    # Load corpus IDs
    print(f"Loading corpus IDs from {args.arxiv_corpus}...")
    corpus_ids = load_corpus_ids(args.arxiv_corpus)

    # Compute coverage for each paper
    print("Computing coverage...")
    papers_with_coverage = [compute_coverage(p, corpus_ids) for p in papers]

    # Generate plots (before filtering, to show full distribution)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if args.plot:
        print("Generating coverage distribution plots...")
        make_plots(papers_with_coverage, args.plot, thresholds)

    # Print threshold analysis (coverage)
    print("\n=== Coverage Threshold Analysis ===")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        passing = [p for p in papers_with_coverage
                   if p["coverage"] >= t and p["num_in_corpus"] >= args.min_citations]
        total_cites = sum(p["num_in_corpus"] for p in passing)
        print(f"  coverage >= {t:.1f}: {len(passing):>5} papers, "
              f"{total_cites:>6} total in-corpus cites")

    # Print min-citations analysis (more relevant for training quality)
    print("\n=== Min In-Corpus Citations Analysis (no coverage filter) ===")
    for mc in [2, 3, 4, 5, 8, 10]:
        passing = [p for p in papers_with_coverage if p["num_in_corpus"] >= mc]
        total_cites = sum(p["num_in_corpus"] for p in passing)
        mean_cov = sum(p["coverage"] for p in passing) / len(passing) if passing else 0
        print(f"  in_corpus >= {mc:>2}: {len(passing):>5} papers, "
              f"{total_cites:>6} total cites, mean coverage {mean_cov:.2f}")

    # Filter — use num_in_corpus (citations actually in the arxiv corpus)
    # rather than resolvable_citations (which is just "resolved to arxiv ID")
    filtered = [
        p for p in papers_with_coverage
        if p["coverage"] >= args.coverage_threshold
        and p["num_in_corpus"] >= args.min_citations
    ]

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n=== Filter Summary ===")
    print(f"Input papers: {len(papers)}")
    print(f"Coverage threshold: {args.coverage_threshold}")
    print(f"Min citations: {args.min_citations}")
    print(f"Papers passing filter: {len(filtered)}")

    # Split breakdown
    split_counts = defaultdict(int)
    for p in filtered:
        split_counts[p["split"]] += 1
    print(f"By split: {dict(split_counts)}")

    # Has-subsections breakdown
    has_sub = sum(1 for p in filtered if p.get("existing_subsection_headings"))
    flat = sum(1 for p in filtered if not p.get("existing_subsection_headings"))
    print(f"With subsection headings: {has_sub}")
    print(f"Flat RW (no subsections): {flat}")

    # Citation stats
    if filtered:
        in_corpus_counts = [p["num_in_corpus"] for p in filtered]
        print(f"\nIn-corpus citation counts:")
        print(f"  Min: {min(in_corpus_counts)}, Max: {max(in_corpus_counts)}, "
              f"Mean: {sum(in_corpus_counts)/len(in_corpus_counts):.1f}")

    print(f"\nOutput: {output_path} ({len(filtered)} papers)")


if __name__ == "__main__":
    main()
