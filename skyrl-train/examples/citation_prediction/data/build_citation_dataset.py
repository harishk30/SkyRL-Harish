"""
Build masked citation prediction dataset from ICLR metadata CSV.

Transforms massive_metadata.csv into SkyRL-format parquet files where each
example is a sentence from a Related Work section with exactly one citation
masked, and the target is the arxiv ID of the cited paper.

Usage:
    python build_citation_dataset.py \
        --metadata_csv /path/to/massive_metadata.csv \
        --split_dir /path/to/iclr_..._v6 \
        --output_dir /path/to/output/ \
        --context_level sentence
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SHORT = (
    "You are a research assistant. Given a sentence from a Related Work section "
    "of an academic paper with one citation replaced by [MASKED], your job is to "
    "craft targeted search queries that return the cited paper in the top results.\n\n"
    "You must reason inside <think> and </think> every time you get new information. "
    "To search, write <search> query </search> and results appear between "
    "<information> and </information>. You can search multiple times. Use the "
    "returned information to refine your queries if the cited paper hasn't appeared yet."
)

SYSTEM_PROMPT_EXTENDED = (
    "You are a research assistant specializing in academic literature. Your task is to "
    "identify a masked citation in a sentence from the Related Work section of an "
    "academic paper.\n\n"
    "The sentence contains [MASKED] where a citation (e.g., \"Author et al., 2020\") "
    "has been removed. Your goal is to craft targeted search queries that return the "
    "cited paper in the top results.\n\n"
    "## Instructions\n\n"
    "1. **Reason first**: Before each search, think carefully inside <think> and "
    "</think> tags about what the sentence is describing, what field it relates to, "
    "and what specific paper would be cited here.\n\n"
    "2. **Search to find the paper**: Write <search> your query </search> to search "
    "over a corpus of arxiv papers. The search engine will return the top matching "
    "papers between <information> and </information> tags. Your query should be "
    "specific enough that the cited paper appears in these results.\n\n"
    "3. **Refine based on results**: If the cited paper hasn't appeared yet, use the "
    "returned information to refine your search. The results give you clues about "
    "related work in the area — use these to narrow down your query. You can search "
    "multiple times. Use context clues from the sentence — author names mentioned "
    "nearby, specific methods or datasets referenced, the year and venue — to "
    "construct more targeted queries.\n\n"
    "4. **Analyze results carefully**: When you receive search results, look for papers "
    "that match the description in the sentence. Consider the paper's topic, methodology, "
    "and how it relates to the surrounding context.\n\n"
    "## Tips\n"
    "- Extract key technical terms, method names, or dataset names from the sentence.\n"
    "- If the sentence mentions specific results or contributions, search for those.\n"
    "- Consider what subfield of ML/AI the sentence is discussing.\n"
    "- Look at surrounding context for additional clues about the cited work.\n"
    "- Make your queries specific — a good query directly describes the paper you're "
    "looking for, not just the general topic."
)

SYSTEM_PROMPTS = {
    "short": SYSTEM_PROMPT_SHORT,
    "extended": SYSTEM_PROMPT_EXTENDED,
}

# Keep backward-compatible default
SYSTEM_PROMPT = SYSTEM_PROMPT_SHORT

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches arxiv IDs like arXiv:1606.06565 or arXiv preprint arXiv:1606.06565
ARXIV_RE = re.compile(
    r"arXiv[:\s]*(?:preprint\s+)?(?:arXiv[:\s]*)?(\d{4}\.\d{4,5})", re.IGNORECASE
)

# Parenthetical citation group: (Author et al., 2019; Other, 2020a)
# Requires alphabetic content inside parens (author names), so bare (2017) won't match.
CITE_GROUP_RE = re.compile(r"\(([^)]*[a-zA-Z][^)]*(?:19|20)\d{2}[a-z]?[^)]*)\)")

# In-text citation: Author (2019), Author et al. (2019), Author and Other (2019)
INLINE_CITE_RE = re.compile(
    r"([A-Z][a-zA-Z\-']+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z\-']+)?)"
    r"\s*\(((?:19|20)\d{2}[a-z]?)\)"
)

# Matches year (optionally with letter) at the end of a single citation
YEAR_RE = re.compile(r"((?:19|20)\d{2}[a-z]?)\s*$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_split_ids(split_dir: str) -> dict[str, str]:
    """Return {submission_id: split_name} from the three split JSON files."""
    mapping: dict[str, str] = {}
    base = Path(split_dir)
    for split_name, suffix in [
        ("train", "_train"),
        ("validation", "_validation"),
        ("test", "_test"),
    ]:
        path = base.parent / (base.name + suffix) / "data.json"
        with open(path) as f:
            items = json.load(f)
        for item in items:
            meta = item.get("_metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            sid = meta.get("submission_id")
            if sid:
                mapping[sid] = split_name
    return mapping


def build_ref_lookup(content_list: list[dict]) -> dict[tuple[str, str], str]:
    """Build (last_name, year) -> arxiv_id from reference list items."""
    lookup: dict[tuple[str, str], str] = {}
    for item in content_list:
        if item.get("sub_type") != "ref_text":
            continue
        for ref_text in item.get("list_items", []):
            arxiv_match = ARXIV_RE.search(ref_text)
            if not arxiv_match:
                continue
            arxiv_id = arxiv_match.group(1)

            # Extract first author last name: text before first comma, take last word
            comma_pos = ref_text.find(",")
            if comma_pos == -1:
                continue
            author_part = ref_text[:comma_pos].strip()
            words = author_part.split()
            if not words:
                continue
            last_name = words[-1].lower().rstrip(".")

            # Extract year: 4-digit year optionally followed by a letter
            year_matches = re.findall(r"\b((?:19|20)\d{2}[a-z]?)\b", ref_text)
            if not year_matches:
                continue
            # Use the last year match (typically the publication year, not a year in the title)
            year = year_matches[-1]

            lookup[(last_name, year)] = arxiv_id
    return lookup


def extract_related_work(content_list: list[dict]) -> list[list[str]] | None:
    """Extract Related Work section as a list of paragraphs.

    Returns None if no Related Work section is found.
    Each paragraph is the full text of a text item.
    Returns list of paragraphs (list[str]) grouped by section.
    """
    start_idx = None
    for i, item in enumerate(content_list):
        if item.get("text_level") == 1:
            heading = item.get("text", "").lower()
            if "related" in heading or "prior work" in heading:
                start_idx = i + 1
                break

    if start_idx is None:
        return None

    paragraphs: list[str] = []
    for j in range(start_idx, len(content_list)):
        item = content_list[j]
        # Stop at next level-1 heading
        if item.get("text_level") == 1:
            break
        if item.get("type") == "text" and not item.get("text_level"):
            text = item.get("text", "").strip()
            if text:
                paragraphs.append(text)

    return paragraphs if paragraphs else None


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def parse_single_citation(cite_str: str) -> tuple[str, str] | None:
    """Parse a single citation string like 'Author et al., 2019a'.

    Returns (last_name, year) or None.
    """
    cite_str = cite_str.strip()
    year_match = YEAR_RE.search(cite_str)
    if not year_match:
        return None
    year = year_match.group(1)

    # Get the author part (before the year)
    author_part = cite_str[: year_match.start()].strip().rstrip(",").strip()

    # Remove "et al." suffix
    author_part = re.sub(r"\s+et\s+al\.?$", "", author_part, flags=re.IGNORECASE)

    # If there's an "&" or "and", take the first author
    for sep in ["&", " and "]:
        if sep in author_part:
            author_part = author_part.split(sep)[0].strip().rstrip(",").strip()
            break

    # Last name is the last word of the remaining author part
    words = author_part.split()
    if not words:
        return None
    last_name = words[-1].lower().rstrip(".")
    return (last_name, year)


def find_all_citations(sentence: str) -> list[dict]:
    """Find all citations (parenthetical and in-text) in a sentence.

    Returns a list of dicts, each with:
        type: "parenthetical" or "inline"
        span: (start, end) — the region to replace with [MASKED]
        parsed: list of (last_name, year) tuples for resolved citations
        original: the original citation text
        count: number of individual citations in this group
    """
    citations = []
    paren_spans = []

    # 1. Parenthetical citations: (Author et al., 2019) or (A; B)
    for match in CITE_GROUP_RE.finditer(sentence):
        group_text = match.group(1)
        individuals = [c.strip() for c in group_text.split(";")]
        parsed = []
        for ind in individuals:
            p = parse_single_citation(ind)
            if p:
                parsed.append(p)
        citations.append({
            "type": "parenthetical",
            "span": (match.start(), match.end()),
            "parsed": parsed,
            "original": group_text,
            "count": len(individuals),
        })
        paren_spans.append((match.start(), match.end()))

    # 2. In-text citations: Author (2019), Author et al. (2019)
    for match in INLINE_CITE_RE.finditer(sentence):
        m_start, m_end = match.start(), match.end()
        # Skip if overlapping with a parenthetical citation
        if any(ps <= m_start < pe or ps < m_end <= pe for ps, pe in paren_spans):
            continue

        author_part = match.group(1)
        year = match.group(2)

        # Extract last name from author part
        clean = re.sub(r"\s+et\s+al\.?$", "", author_part, flags=re.IGNORECASE)
        for sep in ["&", " and "]:
            if sep in clean:
                clean = clean.split(sep)[0].strip()
                break
        words = clean.split()
        if not words:
            continue
        last_name = words[-1].lower().rstrip(".")

        citations.append({
            "type": "inline",
            "span": (m_start, m_end),
            "parsed": [(last_name, year)],
            "original": match.group(0),
            "count": 1,
        })

    return citations


def format_user_prompt(
    masked_sentence: str, context: str | None, context_level: str
) -> str:
    if context_level == "paragraph" and context:
        return f'Context: "{context}"\n\nSentence: "{masked_sentence}"'
    return f'Sentence: "{masked_sentence}"'


def process_paper(
    submission_id: str,
    content_list: list[dict],
    split_name: str,
    context_level: str,
    stats: dict,
) -> list[dict]:
    """Process a single paper and return a list of example dicts."""
    ref_lookup = build_ref_lookup(content_list)
    if not ref_lookup:
        stats["no_arxiv_refs"] += 1
        return []

    paragraphs = extract_related_work(content_list)
    if paragraphs is None:
        stats["no_related_work"] += 1
        return []

    examples = []
    for para_idx, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)
        preceding_paragraph = paragraphs[para_idx - 1] if para_idx > 0 else ""

        for sentence in sentences:
            # Find all citations (both parenthetical and in-text)
            citations = find_all_citations(sentence)
            if not citations:
                continue

            # Total individual citations across all groups
            total_count = sum(c["count"] for c in citations)
            if total_count != 1:
                continue

            # Exactly one citation — get its info
            cite = citations[0]
            if not cite["parsed"]:
                continue

            last_name, year = cite["parsed"][0]
            arxiv_id = ref_lookup.get((last_name, year))
            if arxiv_id is None:
                continue

            # Mask the citation span
            start, end = cite["span"]
            masked_sentence = sentence[:start] + "[MASKED]" + sentence[end:]

            context = preceding_paragraph if context_level == "paragraph" else None

            examples.append(
                {
                    "masked_sentence": masked_sentence,
                    "arxiv_id": arxiv_id,
                    "paper_id": submission_id,
                    "split": split_name,
                    "context": context or "",
                    "original_citation": cite["original"],
                }
            )

    return examples


def build_skyrl_row(
    example: dict, idx: int, context_level: str, prompt_style: str = "short"
) -> dict:
    """Convert an example into a SkyRL-format row."""
    masked_sentence = example["masked_sentence"]
    arxiv_id = example["arxiv_id"]
    context = example["context"] if context_level == "paragraph" else None

    user_content = format_user_prompt(masked_sentence, context, context_level)
    system_prompt = SYSTEM_PROMPTS[prompt_style]

    return {
        "data_source": "citation_prediction_iclr",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "ability": "citation_prediction",
        "env_class": "citation_prediction",
        "reward_spec": json.dumps({"ground_truth": {"target": arxiv_id}}),
        "extra_info": json.dumps(
            {
                "index": idx,
                "need_tools_kwargs": True,
                "question": masked_sentence,
                "split": example["split"],
                "tools_kwargs": {
                    "citation_prediction": {
                        "create_kwargs": {
                            "ground_truth": {"target": arxiv_id},
                            "question": masked_sentence,
                            "data_source": "citation_prediction_iclr",
                        }
                    }
                },
            }
        ),
        "metadata": json.dumps(
            {
                "paper_id": example["paper_id"],
                "original_citation": example["original_citation"],
                "context_level": context_level,
            }
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build masked citation prediction dataset from ICLR metadata."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Path to massive_metadata.csv",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        required=True,
        help="Prefix of split directories (appends _train, _validation, _test)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--context_level",
        type=str,
        default="sentence",
        choices=["sentence", "paragraph"],
        help="Context level: sentence (default) or paragraph",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="short",
        choices=list(SYSTEM_PROMPTS.keys()),
        help="System prompt style: 'short' (minimal) or 'extended' (with strategy guidance)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load split submission_ids
    print("Loading split submission IDs...")
    split_mapping = load_split_ids(args.split_dir)
    split_counts = defaultdict(int)
    for s in split_mapping.values():
        split_counts[s] += 1
    print(f"  Loaded {len(split_mapping)} submission IDs: {dict(split_counts)}")

    print(f"Prompt style: {args.prompt_style}")

    # Step 2-3: Stream CSV and process papers
    print("Processing papers from CSV...")
    all_examples: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    stats = {
        "papers_processed": 0,
        "no_related_work": 0,
        "no_arxiv_refs": 0,
    }

    for chunk in pd.read_csv(args.metadata_csv, chunksize=500):
        for _, row in chunk.iterrows():
            sid = str(row.get("submission_id", ""))
            if sid not in split_mapping:
                continue

            split_name = split_mapping[sid]
            stats["papers_processed"] += 1

            try:
                content_list = json.loads(row["content_list_json"])
            except (json.JSONDecodeError, TypeError):
                continue

            examples = process_paper(
                sid, content_list, split_name, args.context_level, stats
            )
            all_examples[split_name].extend(examples)

        # Progress
        if stats["papers_processed"] % 1000 == 0 and stats["papers_processed"] > 0:
            total = sum(len(v) for v in all_examples.values())
            print(f"  Processed {stats['papers_processed']} papers, {total} examples so far...")

    # Step 4-5: Build SkyRL rows and write parquet
    print("\nWriting parquet files...")
    for split_name, examples in all_examples.items():
        if not examples:
            print(f"  {split_name}: 0 examples (skipped)")
            continue

        rows = []
        for idx, ex in enumerate(examples):
            rows.append(build_skyrl_row(ex, idx, args.context_level, args.prompt_style))

        # Convert prompt list to JSON string for parquet storage
        for row in rows:
            row["prompt"] = json.dumps(row["prompt"])

        df = pd.DataFrame(rows)
        out_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {split_name}: {len(rows)} examples -> {out_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"Papers processed: {stats['papers_processed']}")
    print(f"Papers with no Related Work section: {stats['no_related_work']}")
    print(f"Papers with no arxiv refs: {stats['no_arxiv_refs']}")
    for split_name in ["train", "validation", "test"]:
        print(f"{split_name}: {len(all_examples[split_name])} examples")
    total = sum(len(v) for v in all_examples.values())
    print(f"Total: {total} examples")


if __name__ == "__main__":
    main()
