"""
Build Related Work subsection citation recall dataset (v2).

Given a paper's title, abstract, introduction, and a Related Work subsection
heading, the model must predict ALL citations in that subsection. Papers without
subsection structure in Related Work are skipped entirely.

Each subsection with >=2 resolvable arxiv citations becomes one training example.

Usage:
    python build_citation_dataset_v2.py \
        --metadata_csv /path/to/massive_metadata.csv \
        --split_dir /path/to/iclr_..._v6 \
        --output_dir /path/to/output/ \
        --prompt_style short

Data format notes:
    - content_list_json uses text_level=1 for ALL headings (sections AND subsections)
    - Section depth is encoded in the text: "2 RELATED WORK" (top-level), "2.1 FOO" (subsection)
    - Title comes from CSV `title` column, not content_list
    - Abstract is not present in content_list (stripped during PDF parsing)
    - Abstract is extracted from the split JSON files (conversations[human] field)
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

SYSTEM_PROMPT_SHORT_V2 = (
    "You are a research assistant specializing in machine learning and AI literature. "
    "Given a paper's title, abstract, introduction, and a Related Work subsection heading, "
    "identify ALL papers cited in that subsection by searching an arxiv paper database.\n\n"
    "The search engine is semantic — it matches by meaning, not exact strings. "
    "Each result is formatted as: [arxiv:ID] \"Title\" followed by authors and abstract.\n\n"
    "IMPORTANT search guidelines:\n"
    "- Search using descriptive topic keywords (e.g., \"graph neural network node classification\").\n"
    "- Do NOT search for author names (e.g., \"Li et al. 2018\") or arxiv IDs — the engine cannot match these.\n"
    "- Use the subsection heading and paper context to form targeted queries.\n"
    "- Use specific method names, model names, dataset names, or task descriptions.\n"
    "- Each query MUST be substantially different from previous ones.\n\n"
    "You must reason inside <think> and </think> tags. To search, write "
    "<search> query </search> and results appear in <information> tags. "
    "You MUST search at least once before citing.\n\n"
    "When you find a relevant paper, record its arxiv ID with <citation>arxiv_id</citation>. "
    "You can include multiple IDs: <citation>XXXX.XXXXX, YYYY.YYYYY</citation>.\n\n"
    "You have a citation budget (shown in the prompt). You may cite fewer papers than "
    "the budget — only cite papers you are confident about. However, exceeding the "
    "budget results in zero reward.\n\n"
    "When you have found all cited papers, write <done></done> to finish.\n\n"
    "Example workflow:\n"
    "<think>The subsection is about \"Transfer Learning\". I should search for key topics.</think>\n"
    "<search>transfer learning pretrained language models fine-tuning</search>\n"
    "[Results appear in <information> tags with format: [arxiv:ID] \"Title\" ...]\n"
    "<think>Doc 3 [arxiv:1810.04805] is BERT which is relevant to transfer learning.</think>\n"
    "<citation>1810.04805</citation>\n"
    "<search>domain adaptation neural networks</search>\n"
    "..."
)

SYSTEM_PROMPTS = {
    "short": SYSTEM_PROMPT_SHORT,
    "short_v2": SYSTEM_PROMPT_SHORT_V2,
}

# ---------------------------------------------------------------------------
# Regex patterns (reused from v1)
# ---------------------------------------------------------------------------

ARXIV_RE = re.compile(
    r"arXiv[:\s]*(?:preprint\s+)?(?:arXiv[:\s]*)?(\d{4}\.\d{4,5})", re.IGNORECASE
)

# Parenthetical citation group: (Author et al., 2019; Other, 2020a)
CITE_GROUP_RE = re.compile(r"\(([^)]*[a-zA-Z][^)]*(?:19|20)\d{2}[a-z]?[^)]*)\)")

# In-text citation: Author (2019), Author et al. (2019)
INLINE_CITE_RE = re.compile(
    r"([A-Z][a-zA-Z\-']+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z\-']+)?)"
    r"\s*\(((?:19|20)\d{2}[a-z]?)\)"
)

YEAR_RE = re.compile(r"((?:19|20)\d{2}[a-z]?)\s*$")


# ---------------------------------------------------------------------------
# Helpers (reused from v1)
# ---------------------------------------------------------------------------

def _extract_abstract(conversations: list[dict]) -> str | None:
    """Extract abstract from the human conversation turn in split JSON data.

    The paper text is in the 'human' turn and contains a '# Abstract' section
    followed by the abstract text until the next heading.
    """
    for conv in conversations:
        if conv.get("from") != "human":
            continue
        text = conv.get("value", "")
        # Find "# Abstract" heading (case-insensitive)
        match = re.search(r"^#\s+Abstract\s*$", text, re.MULTILINE | re.IGNORECASE)
        if not match:
            continue
        # Extract text from after the heading to the next heading or end
        rest = text[match.end():]
        # Stop at the next markdown heading
        next_heading = re.search(r"^#\s+", rest, re.MULTILINE)
        if next_heading:
            abstract = rest[:next_heading.start()].strip()
        else:
            abstract = rest.strip()
        if abstract:
            return abstract
    return None


def load_split_ids(split_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({submission_id: split_name}, {submission_id: abstract}) from the three split JSON files."""
    mapping: dict[str, str] = {}
    abstracts: dict[str, str] = {}
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
                abstract = _extract_abstract(item.get("conversations", []))
                if abstract:
                    abstracts[sid] = abstract
    return mapping, abstracts


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

            comma_pos = ref_text.find(",")
            if comma_pos == -1:
                continue
            author_part = ref_text[:comma_pos].strip()
            words = author_part.split()
            if not words:
                continue
            last_name = words[-1].lower().rstrip(".")

            year_matches = re.findall(r"\b((?:19|20)\d{2}[a-z]?)\b", ref_text)
            if not year_matches:
                continue
            year = year_matches[-1]

            lookup[(last_name, year)] = arxiv_id
    return lookup


def parse_single_citation(cite_str: str) -> tuple[str, str] | None:
    """Parse a single citation string like 'Author et al., 2019a'.
    Returns (last_name, year) or None.
    """
    cite_str = cite_str.strip()
    year_match = YEAR_RE.search(cite_str)
    if not year_match:
        return None
    year = year_match.group(1)

    author_part = cite_str[: year_match.start()].strip().rstrip(",").strip()
    author_part = re.sub(r"\s+et\s+al\.?$", "", author_part, flags=re.IGNORECASE)

    for sep in ["&", " and "]:
        if sep in author_part:
            author_part = author_part.split(sep)[0].strip().rstrip(",").strip()
            break

    words = author_part.split()
    if not words:
        return None
    last_name = words[-1].lower().rstrip(".")
    return (last_name, year)


def find_all_citations(sentence: str) -> list[dict]:
    """Find all citations (parenthetical and in-text) in a sentence."""
    citations = []
    paren_spans = []

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
            "parsed": parsed,
            "original": group_text,
        })
        paren_spans.append((match.start(), match.end()))

    for match in INLINE_CITE_RE.finditer(sentence):
        m_start, m_end = match.start(), match.end()
        if any(ps <= m_start < pe or ps < m_end <= pe for ps, pe in paren_spans):
            continue

        author_part = match.group(1)
        year = match.group(2)

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
            "parsed": [(last_name, year)],
            "original": match.group(0),
        })

    return citations


# ---------------------------------------------------------------------------
# V2-specific extraction functions
#
# content_list uses text_level=1 for ALL headings. Section depth is encoded
# in the heading text itself: "2 RELATED WORK" (top-level) vs "2.1 FOO"
# (subsection). We parse the section number prefix to distinguish levels.
# ---------------------------------------------------------------------------

# Matches a top-level section number: "2 RELATED WORK" -> "2"
_TOP_SECTION_RE = re.compile(r"^(\d+)\s+\S")
# Matches a subsection number: "2.1 FOO" -> "2.1"
_SUBSECTION_RE = re.compile(r"^(\d+\.\d+)")


def _is_heading(item: dict) -> bool:
    """Check if a content_list item is a section/subsection heading."""
    return item.get("text_level") == 1


def _get_section_num(heading_text: str) -> str | None:
    """Extract the top-level section number (e.g., '2') from heading text."""
    m = _TOP_SECTION_RE.match(heading_text.strip())
    return m.group(1) if m else None


def _is_subsection_of(heading_text: str, parent_num: str) -> bool:
    """Check if heading_text is a subsection of the given parent section number."""
    m = _SUBSECTION_RE.match(heading_text.strip())
    if not m:
        return False
    return m.group(1).startswith(parent_num + ".")


def extract_introduction(content_list: list[dict]) -> str | None:
    """Extract introduction section text.

    Finds heading matching "introduction" or starting with "1 " and collects
    all non-heading text items until the next top-level section heading.
    """
    start_idx = None
    intro_num = None
    for i, item in enumerate(content_list):
        if not _is_heading(item):
            continue
        text = item.get("text", "").strip()
        text_lower = text.lower()
        if "introduction" in text_lower or text_lower.startswith("1 "):
            start_idx = i + 1
            intro_num = _get_section_num(text)
            break

    if start_idx is None:
        return None

    paragraphs = []
    for j in range(start_idx, len(content_list)):
        item = content_list[j]
        if _is_heading(item):
            heading_text = item.get("text", "").strip()
            # If it's a subsection of intro (e.g., "1.1 Motivation"), include its body
            if intro_num and _is_subsection_of(heading_text, intro_num):
                continue
            # Otherwise it's a new top-level section — stop
            break
        if item.get("type") == "text":
            text = item.get("text", "").strip()
            if text:
                paragraphs.append(text)

    return "\n\n".join(paragraphs) if paragraphs else None


def extract_related_work_subsections(content_list: list[dict]) -> list[dict] | None:
    """Extract Related Work subsections with their headings and paragraphs.

    Since text_level=1 is used for ALL headings, we distinguish section depth
    by parsing the section number from the heading text:
    - "2 RELATED WORK" -> top-level section (number "2")
    - "2.1 TRANSFER LEARNING" -> subsection of section 2

    Returns None if:
    - No Related Work section found
    - Related Work has no subsection headings (flat structure)

    Returns list of {heading: str, paragraphs: list[str]} for each subsection.
    """
    # Find the Related Work heading
    rw_idx = None
    rw_num = None
    for i, item in enumerate(content_list):
        if not _is_heading(item):
            continue
        text = item.get("text", "").strip()
        text_lower = text.lower()
        if "related" in text_lower or "prior work" in text_lower:
            rw_idx = i
            rw_num = _get_section_num(text)
            break

    if rw_idx is None:
        return None

    # Collect all items from after the RW heading until the next top-level section
    rw_items = []  # list of (item, is_subsection_heading)
    has_subsections = False

    for j in range(rw_idx + 1, len(content_list)):
        item = content_list[j]

        if _is_heading(item):
            heading_text = item.get("text", "").strip()

            # Check if this is a subsection of the Related Work section
            if rw_num and _is_subsection_of(heading_text, rw_num):
                rw_items.append((item, True))
                has_subsections = True
                continue

            # Not a subsection — it's a new top-level section, stop
            break

        rw_items.append((item, False))

    if not has_subsections:
        return None  # Flat Related Work — skip paper

    # Group content by subsection headings
    subsections = []
    current_heading = None
    current_paragraphs = []

    for item, is_sub_heading in rw_items:
        if is_sub_heading:
            # Save previous subsection
            if current_heading is not None and current_paragraphs:
                subsections.append({
                    "heading": current_heading,
                    "paragraphs": current_paragraphs,
                })
            current_heading = item.get("text", "").strip()
            current_paragraphs = []
        elif item.get("type") == "text":
            text = item.get("text", "").strip()
            if text and current_heading is not None:
                current_paragraphs.append(text)

    # Don't forget the last subsection
    if current_heading is not None and current_paragraphs:
        subsections.append({
            "heading": current_heading,
            "paragraphs": current_paragraphs,
        })

    return subsections if subsections else None


def collect_subsection_citations(
    paragraphs: list[str], ref_lookup: dict[tuple[str, str], str]
) -> list[str]:
    """Extract all unique arxiv IDs cited in the given paragraphs.

    Finds all (last_name, year) citations, resolves them via ref_lookup,
    and returns deduplicated arxiv IDs.
    """
    seen_ids = set()
    result = []

    for para in paragraphs:
        # Split into sentences for finer-grained citation parsing
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for sentence in sentences:
            citations = find_all_citations(sentence)
            for cite in citations:
                for last_name, year in cite["parsed"]:
                    arxiv_id = ref_lookup.get((last_name, year))
                    if arxiv_id and arxiv_id not in seen_ids:
                        seen_ids.add(arxiv_id)
                        result.append(arxiv_id)

    return result


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def format_user_prompt(
    title: str, abstract: str | None, introduction: str, subsection_heading: str,
    num_citations: int = 0,
) -> str:
    budget = num_citations * 2
    parts = []
    parts.append(f"Paper title: {title}")
    if abstract:
        parts.append(f"\nAbstract:\n{abstract}")
    parts.append(f"\nIntroduction:\n{introduction}")
    parts.append(f"\nRelated Work subsection heading: \"{subsection_heading}\"")
    parts.append(
        f"\nCitation budget: at most {budget} citations (you may cite fewer — only cite papers you are confident about). "
        "Exceeding this budget results in zero reward."
    )
    parts.append(
        "\nIdentify all papers that would be cited in this Related Work subsection. "
        "Search for them and report their arxiv IDs using <citation> tags."
    )
    return "\n".join(parts)


def process_paper(
    submission_id: str,
    title: str,
    abstract: str | None,
    content_list: list[dict],
    split_name: str,
    stats: dict,
) -> list[dict]:
    """Process a single paper and return a list of subsection example dicts."""
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

    examples = []
    for sub in subsections:
        arxiv_ids = collect_subsection_citations(sub["paragraphs"], ref_lookup)

        # Require at least 2 resolvable citations
        if len(arxiv_ids) < 2:
            stats["too_few_citations"] += 1
            continue

        examples.append({
            "title": title,
            "abstract": abstract,
            "introduction": introduction,
            "subsection_heading": sub["heading"],
            "target_ids": arxiv_ids,
            "num_citations": len(arxiv_ids),
            "paper_id": submission_id,
            "split": split_name,
        })

    return examples


def build_skyrl_row(example: dict, idx: int, prompt_style: str = "short") -> dict:
    """Convert an example into a SkyRL-format row."""
    user_content = format_user_prompt(
        example["title"],
        example.get("abstract"),
        example["introduction"],
        example["subsection_heading"],
        num_citations=example["num_citations"],
    )
    system_prompt = SYSTEM_PROMPTS[prompt_style]

    return {
        "data_source": "citation_prediction_v2_iclr",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "ability": "citation_prediction_v2",
        "env_class": "citation_prediction_v2",
        "reward_spec": json.dumps({
            "ground_truth": {"targets": example["target_ids"]}
        }),
        "extra_info": json.dumps({
            "index": idx,
            "need_tools_kwargs": True,
            "question": example["subsection_heading"],
            "split": example["split"],
            "tools_kwargs": {
                "citation_prediction_v2": {
                    "create_kwargs": {
                        "ground_truth": {"targets": example["target_ids"]},
                        "question": example["subsection_heading"],
                        "data_source": "citation_prediction_v2_iclr",
                    }
                }
            },
        }),
        "metadata": json.dumps({
            "paper_id": example["paper_id"],
            "subsection_heading": example["subsection_heading"],
            "num_citations": example["num_citations"],
            "target_ids": example["target_ids"],
        }),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Related Work subsection citation recall dataset (v2)."
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
        "--output_dir", type=str, required=True,
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--prompt_style", type=str, default="short",
        choices=list(SYSTEM_PROMPTS.keys()),
        help="System prompt style",
    )
    parser.add_argument(
        "--min_citations", type=int, default=2,
        help="Minimum resolvable citations per subsection (default: 2)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load split submission_ids and abstracts
    print("Loading split submission IDs and abstracts...")
    split_mapping, abstracts = load_split_ids(args.split_dir)
    split_counts = defaultdict(int)
    for s in split_mapping.values():
        split_counts[s] += 1
    print(f"  Loaded {len(split_mapping)} submission IDs: {dict(split_counts)}")
    print(f"  Loaded {len(abstracts)} abstracts ({len(abstracts)/len(split_mapping)*100:.1f}% coverage)")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Min citations per subsection: {args.min_citations}")

    # Step 2: Process papers
    print("Processing papers from CSV...")
    all_examples: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
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
            stats["papers_processed"] += 1

            title = str(row.get("title", "")).strip()

            try:
                content_list = json.loads(row["content_list_json"])
            except (json.JSONDecodeError, TypeError):
                continue

            abstract = abstracts.get(sid)
            examples = process_paper(sid, title, abstract, content_list, split_name, stats)
            all_examples[split_name].extend(examples)

        if stats["papers_processed"] % 1000 == 0 and stats["papers_processed"] > 0:
            total = sum(len(v) for v in all_examples.values())
            print(f"  Processed {stats['papers_processed']} papers, {total} examples so far...")

    # Step 3: Build SkyRL rows and write parquet
    print("\nWriting parquet files...")
    for split_name, examples in all_examples.items():
        if not examples:
            print(f"  {split_name}: 0 examples (skipped)")
            continue

        rows = []
        for idx, ex in enumerate(examples):
            rows.append(build_skyrl_row(ex, idx, args.prompt_style))

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
    print(f"Papers with no arxiv refs: {stats['no_arxiv_refs']}")
    print(f"Papers with no title: {stats['no_title']}")
    print(f"Papers with no introduction: {stats['no_introduction']}")
    print(f"Papers with flat Related Work (no subsections): {stats['no_subsections']}")
    print(f"Subsections with < {args.min_citations} citations: {stats['too_few_citations']}")
    for split_name in ["train", "validation", "test"]:
        print(f"{split_name}: {len(all_examples[split_name])} examples")
    total = sum(len(v) for v in all_examples.values())
    print(f"Total: {total} examples")

    # Citation count distribution
    all_ex = [ex for exs in all_examples.values() for ex in exs]
    if all_ex:
        counts = [ex["num_citations"] for ex in all_ex]
        print(f"\nCitation count distribution:")
        print(f"  Min: {min(counts)}, Max: {max(counts)}, Mean: {sum(counts)/len(counts):.1f}")
        from collections import Counter
        dist = Counter(counts)
        for k in sorted(dist.keys())[:15]:
            print(f"  {k} citations: {dist[k]} examples")


if __name__ == "__main__":
    main()
