"""Step 3: Use Gemini to split Related Work into logical subsections with rich queries.

For each paper, makes a single Gemini call that:
1. Partitions the set of citations into thematic subsets
2. Generates a rich descriptive query per subset from the author's perspective

Uses async batching (~10 concurrent) with JSONL checkpoint/resume.

Usage:
    python gemini_split_subsections.py \
        --input filtered_papers.json \
        --output gemini_subsections.json \
        --checkpoint gemini_checkpoint.jsonl \
        --concurrency 10
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# Load .env — try search harish_setup (has real keys), then local
for _env_candidate in [
    Path(__file__).resolve().parent.parent.parent.parent / "search" / "harish_setup" / ".env",
    Path(__file__).resolve().parent.parent / "harish_setup" / ".env",
]:
    if _env_candidate.exists():
        load_dotenv(_env_candidate)
        break

from google import genai
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Response schema (enforced by Gemini structured output)
# ---------------------------------------------------------------------------

class CitationSubset(BaseModel):
    citation_strings: list[str]
    query: str

class SplitResponse(BaseModel):
    subsets: list[CitationSubset]


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

GEMINI_PROMPT_TEMPLATE = """\
You are an expert at analyzing academic Related Work sections.

=== PAPER ===
Title: {title}
Abstract: {abstract}

=== INTRODUCTION ===
{introduction}

=== RELATED WORK SECTION ===
{related_work_text}

=== CITATIONS FOUND IN RELATED WORK ===
{citation_list}

{subsection_hint}
=== TASK ===
Partition the citations above into thematic subsets based on why the author \
cites them together. Then, for each subset, generate a search query from the \
author's perspective.

The query (2-4 sentences) should:
- Describe what the author is LOOKING FOR — their need, not the papers themselves
- Be just specific enough that the task is tractable, but not so descriptive \
that it gives away the answers
- Frame it as: "The author is looking to find [intent]. What papers should they cite?"
- Reference the author's own work context (what they're building/proposing)
- Do NOT include author names, years, arxiv IDs, or paper titles

If the Related Work section has explicit subsection headings, use them as a guide \
for your partitioning — but you may split further or merge across headings if the \
citation groupings warrant it.

Rules:
- Every citation must belong to exactly one subset
- Each subset must contain at least 2 citations
- Prefer 2-6 subsets (merge small groups if needed)
- If all citations serve one coherent purpose, return 1 subset

Return ONLY valid JSON:
{{"subsets": [
  {{
    "citation_strings": ["Smith et al., 2019", "Jones and Lee, 2020"],
    "query": "The author is looking to find..."
  }},
  ...
]}}"""


def build_citation_list_text(paper: dict) -> str:
    """Build a numbered list of citation strings for the prompt.

    Only includes citations that resolved to arxiv IDs AND are in the corpus
    (i.e., citations_in_corpus from Step 2). Uses citation_display_map to
    convert arxiv IDs to readable "LastName, Year" strings.
    """
    display_map = paper.get("citation_display_map", {})
    # Use citations_in_corpus (from Step 2) if available, else all_citation_ids
    corpus_ids = paper.get("citations_in_corpus", paper.get("all_citation_ids", []))
    display_strings = []
    for cid in corpus_ids:
        display = display_map.get(cid)
        if display:
            display_strings.append(display)
    return "\n".join(f"{i+1}. {ds}" for i, ds in enumerate(display_strings))


def build_subsection_hint(paper: dict) -> str:
    """Build hint about existing subsection headings if present."""
    headings = paper.get("existing_subsection_headings", [])
    if not headings:
        return ""
    heading_list = "\n".join(f"- {h}" for h in headings)
    return f"=== EXISTING SUBSECTION HEADINGS (for guidance) ===\n{heading_list}\n\n"


def build_prompt(paper: dict) -> str:
    """Build the full Gemini prompt for a paper."""
    return GEMINI_PROMPT_TEMPLATE.format(
        title=paper["title"],
        abstract=paper["abstract"],
        introduction=paper["introduction"],
        related_work_text=paper["related_work_text"],
        citation_list=build_citation_list_text(paper),
        subsection_hint=build_subsection_hint(paper),
    )


# ---------------------------------------------------------------------------
# Citation string -> arxiv ID mapping
# ---------------------------------------------------------------------------

# Reuse v2 parsing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "citation-prediction-v2" / "data"))
from build_citation_dataset_v2 import parse_single_citation




# ---------------------------------------------------------------------------
# Gemini API calls
# ---------------------------------------------------------------------------

async def call_gemini(
    client: genai.Client,
    paper: dict,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
) -> dict | None:
    """Call Gemini API for a single paper. Returns parsed JSON or None on failure."""
    prompt = build_prompt(paper)

    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SplitResponse,
                    temperature=0.3,
                    max_output_tokens=8192,
                ),
            )
            text = response.text.strip()
            result = SplitResponse.model_validate_json(text)
            return result.model_dump()

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  WARNING [{paper['paper_id']}] attempt {attempt+1}: Parse/validation error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                wait = 2 ** (attempt + 2)
                print(f"  Rate limited [{paper['paper_id']}], waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  ERROR [{paper['paper_id']}] attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# Post-processing: map citation strings back to arxiv IDs
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> set[str]:
    """Load already-processed paper IDs from checkpoint JSONL."""
    done = set()
    path = Path(checkpoint_path)
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                done.add(entry["paper_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def load_checkpoint_results(checkpoint_path: str) -> list[dict]:
    """Load all results from checkpoint JSONL."""
    results = []
    path = Path(checkpoint_path)
    if not path.exists():
        return results
    with open(path) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


async def process_batch(
    client: genai.Client,
    papers: list[dict],
    checkpoint_path: str,
    model: str,
    concurrency: int,
) -> list[dict]:
    """Process papers with async concurrency, writing checkpoint as we go."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    checkpoint_f = open(checkpoint_path, "a")
    completed = 0
    total = len(papers)

    async def process_one(paper: dict):
        nonlocal completed
        async with semaphore:
            result = await call_gemini(client, paper, model=model)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  Progress: {completed}/{total}")

            if result is None:
                entry = {
                    "paper_id": paper["paper_id"],
                    "status": "error",
                    "subsets": [],
                }
            else:
                entry = {
                    "paper_id": paper["paper_id"],
                    "status": "ok",
                    "subsets": result["subsets"],
                }

            # Write to checkpoint
            checkpoint_f.write(json.dumps(entry) + "\n")
            checkpoint_f.flush()
            results.append(entry)

    tasks = [process_one(p) for p in papers]
    await asyncio.gather(*tasks)

    checkpoint_f.close()
    return results


def resolve_subsets_to_ids(
    gemini_result: dict,
    paper: dict,
) -> list[dict]:
    """Resolve Gemini's citation strings to arxiv IDs using the display map.

    Since we only gave Gemini resolvable citations (as "LastName, Year" strings),
    we can map back directly using the inverse of citation_display_map.

    Returns list of subsets with resolved citation_ids.
    Drops subsets where < 2 citations resolved.
    """
    # Build reverse map: display_string -> arxiv_id
    display_map = paper.get("citation_display_map", {})
    display_to_id = {v: k for k, v in display_map.items()}

    # Also build normalized versions for fuzzy matching
    display_to_id_norm = {}
    for display, arxiv_id in display_to_id.items():
        # Normalize: strip, lowercase
        display_to_id_norm[display.strip().lower()] = arxiv_id

    corpus_ids = set(paper.get("citations_in_corpus", paper.get("all_citation_ids", [])))

    resolved_subsets = []

    for subset in gemini_result.get("subsets", []):
        subset_ids = []
        seen = set()

        for cite_str in subset["citation_strings"]:
            cite_norm = cite_str.strip().lower()
            arxiv_id = display_to_id_norm.get(cite_norm)
            if not arxiv_id:
                # Try parsing as "LastName, Year" and matching
                parsed = parse_single_citation(cite_str)
                if parsed:
                    last_name, year = parsed
                    key = f"{last_name}, {year}".lower()
                    arxiv_id = display_to_id_norm.get(key)

            if arxiv_id and arxiv_id in corpus_ids and arxiv_id not in seen:
                seen.add(arxiv_id)
                subset_ids.append(arxiv_id)

        if len(subset_ids) >= 2:
            resolved_subsets.append({
                "citation_strings": subset["citation_strings"],
                "query": subset["query"],
                "citation_ids": subset_ids,
            })

    return resolved_subsets


# ---------------------------------------------------------------------------
# Post-processing: rebuild ref_lookups and resolve
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gemini-based subsection splitting (Step 3 of v3 pipeline)."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to filtered_papers.json from Step 2")
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="(Deprecated, no longer needed)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output gemini_subsections.json")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="JSONL checkpoint file for resume (default: output.checkpoint.jsonl)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent Gemini requests (default: 10)")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview",
                        help="Gemini model name")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "validation", "test"],
                        help="Only process this split")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or (args.output + ".checkpoint.jsonl")

    # Load papers
    print(f"Loading papers from {args.input}...")
    with open(args.input) as f:
        papers = json.load(f)
    if args.split:
        papers = [p for p in papers if p["split"] == args.split]
    print(f"  {len(papers)} papers to process")

    # Load checkpoint
    done_ids = load_checkpoint(checkpoint_path)
    checkpoint_results = load_checkpoint_results(checkpoint_path)
    print(f"  {len(done_ids)} already completed in checkpoint")

    # Filter to remaining papers
    remaining = [p for p in papers if p["paper_id"] not in done_ids]
    print(f"  {len(remaining)} remaining to process")

    # Build paper_id -> paper lookup
    paper_lookup = {p["paper_id"]: p for p in papers}

    # Run Gemini calls
    if remaining:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
            sys.exit(1)

        client = genai.Client(api_key=api_key)

        print(f"\nCalling Gemini ({args.model}) with concurrency={args.concurrency}...")
        start = time.time()
        new_results = asyncio.run(
            process_batch(client, remaining, checkpoint_path, args.model, args.concurrency)
        )
        elapsed = time.time() - start
        print(f"  Completed {len(new_results)} calls in {elapsed:.0f}s "
              f"({len(new_results)/elapsed:.1f} calls/s)")

        checkpoint_results.extend(new_results)

    # Resolve Gemini's citation strings -> arxiv IDs using the display map
    # (no ref_lookup rebuild needed — we gave Gemini only resolvable citations)
    print("\nResolving citation strings to arxiv IDs...")
    final_results = []
    stats = defaultdict(int)

    for entry in checkpoint_results:
        paper_id = entry["paper_id"]
        paper = paper_lookup.get(paper_id)
        if paper is None:
            stats["paper_not_found"] += 1
            continue

        if entry["status"] != "ok" or not entry.get("subsets"):
            stats["gemini_error"] += 1
            continue

        resolved = resolve_subsets_to_ids(entry, paper)

        if not resolved:
            stats["no_valid_subsets"] += 1
            continue

        stats["success"] += 1
        final_results.append({
            "paper_id": paper_id,
            "title": paper["title"],
            "abstract": paper["abstract"],
            "introduction": paper["introduction"],
            "related_work_text": paper["related_work_text"],
            "existing_subsection_headings": paper.get("existing_subsection_headings", []),
            "all_citation_ids": paper["all_citation_ids"],
            "citation_display_map": paper.get("citation_display_map", {}),
            "split": paper["split"],
            "subsets": resolved,
        })

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # Summary
    total_subsets = sum(len(r["subsets"]) for r in final_results)
    total_citations = sum(
        len(s["citation_ids"])
        for r in final_results
        for s in r["subsets"]
    )
    print(f"\n=== Summary ===")
    print(f"Papers processed: {len(checkpoint_results)}")
    print(f"  Gemini errors: {stats.get('gemini_error', 0)}")
    print(f"  No valid subsets (< 2 resolved citations): {stats.get('no_valid_subsets', 0)}")
    print(f"  Success: {stats.get('success', 0)}")
    print(f"\nFinal papers: {len(final_results)}")
    print(f"Total subsets: {total_subsets}")
    print(f"Total citations across subsets: {total_citations}")
    if total_subsets > 0:
        print(f"Avg citations per subset: {total_citations/total_subsets:.1f}")
        print(f"Avg subsets per paper: {total_subsets/len(final_results):.1f}")

    # Subset size distribution
    subset_sizes = [len(s["citation_ids"]) for r in final_results for s in r["subsets"]]
    if subset_sizes:
        from collections import Counter
        dist = Counter(subset_sizes)
        print(f"\nSubset size distribution:")
        for k in sorted(dist.keys())[:15]:
            print(f"  {k} citations: {dist[k]} subsets")

    # Split breakdown
    split_counts = defaultdict(int)
    for r in final_results:
        split_counts[r["split"]] += 1
    print(f"\nBy split: {dict(split_counts)}")

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
