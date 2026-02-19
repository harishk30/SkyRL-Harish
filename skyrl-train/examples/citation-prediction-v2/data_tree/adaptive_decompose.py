#!/usr/bin/env python3
"""Adaptive hierarchical decomposition of Related Work subsections.

Evaluates each subsection via multi-turn vLLM rollouts, then uses Gemini
to split intractable subsections into finer-grained subtopics. Recurses
until all leaves are either tractable (recall >= threshold) or unsplittable.

Requires:
  - A running retriever server (FAISS + embedding model)
  - A running vLLM server OR local vLLM model
  - GEMINI_API_KEY env var for Gemini API access

Usage:
    python adaptive_decompose.py \
        --corpus subsection_corpus.json \
        --search_url http://retriever-host:8000/retrieve \
        --model /path/to/Qwen3-4B \
        --output trees.json \
        --checkpoint trees_checkpoint.jsonl \
        --split train \
        --max_examples 50
"""

import argparse
import copy
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from tree_utils import (
    SubsectionTree,
    TreeNode,
    make_root_node,
    make_child_node,
    save_trees,
    save_checkpoint,
    get_checkpoint_keys,
    load_checkpoint,
)

# ---------------------------------------------------------------------------
# System prompt (same as v2 eval)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DecomposeConfig:
    min_citations: int = 2          # stop splitting below this
    max_depth: int = 3              # max tree depth
    recall_threshold: float = 0.3   # "tractable" if best-of-k recall >= this
    num_samples: int = 8            # k rollouts per node eval
    max_turns: int = 4              # m (search turns per rollout)
    topk: int = 5                   # n (retriever results per search)
    max_tokens: int = 1024          # max tokens per generation turn
    temperature: float = 1.0        # sampling temperature for rollouts
    gemini_model: str = "gemini-3-flash-preview"
    seed: int = 42
    max_heading_refine: int = 1     # max heading refinement attempts per node


# ---------------------------------------------------------------------------
# Arxiv ID utilities (from eval_base_model.py)
# ---------------------------------------------------------------------------

CITATION_TAG_RE = re.compile(r"<citation>(.*?)</citation>", re.DOTALL)


def normalize_arxiv_id(s: str) -> str:
    s = s.strip()
    match = re.search(r"(\d{4}\.\d{4,5})", s)
    if match:
        return match.group(1)
    s = s.strip("[]")
    s = re.sub(r"^[A-Za-z-]+[:/]\s*", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s.strip()


def extract_all_citations(text: str) -> set[str]:
    ids = set()
    for match in CITATION_TAG_RE.finditer(text):
        raw = match.group(1)
        for part in raw.split(","):
            part = part.strip()
            if part:
                normalized = normalize_arxiv_id(part)
                if normalized and re.match(r"\d{4}\.\d{4,5}$", normalized):
                    ids.add(normalized)
    return ids


def compute_recall(messages: list[dict], targets: list[str]) -> float:
    """Compute recall from a completed trajectory.

    Returns 0.0 if predictions exceed 2x budget (spam penalty).
    """
    full_text = "".join(m["content"] for m in messages)
    predicted = extract_all_citations(full_text)
    gt_set = {normalize_arxiv_id(t) for t in targets}
    gt_set.discard("")

    if not gt_set:
        return 0.0

    # Spam penalty
    if len(predicted) > 2.0 * len(gt_set):
        return 0.0

    correct = predicted & gt_set
    return len(correct) / len(gt_set)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def call_retriever(search_url: str, query: str, topk: int = 5, timeout: int = 30) -> str:
    try:
        resp = requests.post(
            search_url,
            json={"query": query, "topk": topk, "return_scores": True},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_results = data.get("result", [])
        if raw_results:
            pretty_parts = []
            for retrieval in raw_results:
                formatted = ""
                for idx, doc_item in enumerate(retrieval):
                    content = doc_item["document"]["contents"].strip()
                    formatted += f"Doc {idx+1}: {content}\n"
                pretty_parts.append(formatted)
            final_result = "\n---\n".join(pretty_parts)
            return json.dumps({"result": final_result})
        return json.dumps({"result": "No search results found."})
    except Exception as e:
        return json.dumps({"result": f"Search error: {e}"})


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

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


def build_messages(tree: SubsectionTree, node: TreeNode) -> list[dict]:
    """Build the initial messages for evaluating a node."""
    user_content = format_user_prompt(
        tree.title, tree.abstract, tree.introduction,
        node.heading, len(node.citation_ids),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Multi-turn evaluation (adapted from eval_base_model.py)
# ---------------------------------------------------------------------------

def run_batched_rollouts(
    examples: list[dict],
    llm: LLM,
    tokenizer,
    config: DecomposeConfig,
    search_url: str,
) -> list[list[float]]:
    """Run k rollouts for each example, return per-example list of recall values.

    All k rollouts for all examples run in a single batch (k * n states),
    so the GPU processes everything in parallel per turn.

    Each example is {"messages": [...], "targets": [...]}.
    Returns list of lists: recalls[i][j] = recall for example i, sample j.
    """
    n = len(examples)
    k = config.num_samples

    # Create all k*n states upfront: state index = example_idx * k + sample_idx
    states = []
    seed_per_state = []
    example_idx_per_state = []
    for ex_idx, ex in enumerate(examples):
        for sample_idx in range(k):
            states.append({
                "messages": copy.deepcopy(ex["messages"]),
                "targets": ex["targets"],
                "done": False,
                "turns": 0,
            })
            seed_per_state.append(config.seed + sample_idx)
            example_idx_per_state.append(ex_idx)

    total_states = len(states)
    active_indices = list(range(total_states))

    for turn in range(config.max_turns):
        if not active_indices:
            break

        # Build prompts and per-prompt SamplingParams for all active states
        prompts = []
        params_list = []
        for idx in active_indices:
            text = tokenizer.apply_chat_template(
                states[idx]["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)
            params_list.append(SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stop=["</search>", "</done>"],
                include_stop_str_in_output=True,
                seed=seed_per_state[idx],
            ))

        # Single batched vLLM call with per-prompt seeds
        outputs = llm.generate(prompts, params_list)

        # Process all responses and do retriever calls
        still_active = []
        for i, idx in enumerate(active_indices):
            st = states[idx]
            response = outputs[i].outputs[0].text
            st["turns"] += 1
            st["messages"].append({"role": "assistant", "content": response})

            if "<done>" in response:
                st["done"] = True
                continue

            match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
            if match is None:
                still_active.append(idx)
                continue

            query = match.group(1).strip()
            tool_output = call_retriever(search_url, query, config.topk)
            observation = "\n<information>" + tool_output + "</information>\n"

            remaining = config.max_turns - st["turns"]
            if remaining > 1:
                observation += f"\n\n{remaining} turns remaining."
            elif remaining == 1:
                observation += "\n\nThis is your last turn. Cite remaining papers and write <done></done>."

            st["messages"].append({"role": "user", "content": observation})
            still_active.append(idx)

        active_indices = still_active

    # Compute recall and reshape: recalls[example_idx][sample_idx]
    all_recalls = [[] for _ in range(n)]
    for state_idx, st in enumerate(states):
        ex_idx = example_idx_per_state[state_idx]
        recall = compute_recall(st["messages"], st["targets"])
        all_recalls[ex_idx].append(recall)

    return all_recalls


def evaluate_nodes(
    nodes: list[TreeNode],
    tree: SubsectionTree,
    llm: LLM,
    tokenizer,
    config: DecomposeConfig,
    search_url: str,
) -> None:
    """Evaluate a batch of nodes and populate their recall fields."""
    if not nodes:
        return

    examples = []
    for node in nodes:
        messages = build_messages(tree, node)
        examples.append({
            "messages": messages,
            "targets": node.citation_ids,
        })

    all_recalls = run_batched_rollouts(examples, llm, tokenizer, config, search_url)

    for i, node in enumerate(nodes):
        recalls = all_recalls[i]
        node.best_of_k_recall = max(recalls) if recalls else 0.0
        node.mean_recall = sum(recalls) / len(recalls) if recalls else 0.0


# ---------------------------------------------------------------------------
# Gemini splitting
# ---------------------------------------------------------------------------

def _format_tree_view(node: TreeNode, tree: SubsectionTree) -> str:
    """Format the current decomposition tree as an indented view.

    Marks the node being split with ← SPLITTING THIS.
    Shows headings and citation counts for all nodes.
    """
    root = tree.get_root()

    def _render(nid: str, depth: int) -> list[str]:
        n = tree.get_node(nid)
        indent = "  " * depth
        marker = " ← SPLITTING THIS" if n.node_id == node.node_id else ""
        status = ""
        if n.is_leaf and n.node_id != node.node_id:
            if n.best_of_k_recall is not None:
                status = f" (leaf, recall={n.best_of_k_recall:.2f})"
            else:
                status = " (leaf)"
        line = f"{indent}- \"{n.heading}\" [{len(n.citation_ids)} citations]{status}{marker}"
        lines = [line]
        for cid in n.children:
            lines.extend(_render(cid, depth + 1))
        return lines

    return "\n".join(_render(root.node_id, 0))


def _format_cited_papers_block(citation_ids: list[str], cited_papers: dict) -> str:
    """Format cited paper metadata for the Gemini prompt."""
    lines = []
    for cid in citation_ids:
        meta = cited_papers.get(cid, {})
        title = meta.get("title", "Unknown")
        authors = meta.get("authors", "Unknown")
        abstract = meta.get("abstract", "")
        lines.append(f"  [{cid}] \"{title}\" by {authors}\n    Abstract: {abstract}")
    return "\n".join(lines)


def _format_citation_sentences(citation_ids: list[str], cite_map: dict) -> str:
    """Format citation-sentence mappings for the Gemini prompt."""
    lines = []
    for cid in citation_ids:
        sentences = cite_map.get(cid, [])
        if sentences:
            lines.append(f"  [{cid}] cited in: {' | '.join(sentences)}")
    return "\n".join(lines)


def gemini_split(
    node: TreeNode,
    tree: SubsectionTree,
    gemini_client,
    gemini_model: str,
    max_retries: int = 3,
) -> list[dict] | None:
    """Call Gemini to split a node into 2-4 subtopics.

    Provides rich context: source paper info, full Related Work section,
    cited paper metadata, and citation-sentence mappings.

    Returns list of {"heading": str, "citation_ids": [str]} or None on failure.
    """
    # Build context blocks
    cited_block = _format_cited_papers_block(node.citation_ids, tree.cited_papers)
    cite_sentences = _format_citation_sentences(
        node.citation_ids, tree.citation_sentence_map
    )
    tree_view = _format_tree_view(node, tree)

    prompt = f"""You are helping decompose a Related Work section into finer-grained subtopics
for a citation prediction task. Given the context below, split the citations in
the CURRENT SUBTOPIC into 2-4 coherent thematic groups.

=== SOURCE PAPER ===
Title: {tree.title}
Abstract: {tree.abstract}

Introduction:
{tree.introduction}

=== FULL RELATED WORK SECTION ===
{tree.full_related_work_text}

=== CURRENT DECOMPOSITION TREE ===
{tree_view}

=== CURRENT SUBTOPIC TO SPLIT ===
Heading: {node.heading}
Citations to split: {json.dumps(node.citation_ids)}

=== CITED PAPER DETAILS ===
{cited_block}

=== WHERE EACH PAPER IS CITED ===
{cite_sentences}

=== TASK ===
Group the {len(node.citation_ids)} citations above into 2-4 thematic subtopics.
Each citation must appear in exactly one subtopic. Every citation must be assigned.
Create subtopics that are distinct from the existing sibling nodes shown in the tree above.

Return ONLY valid JSON:
{{"subtopics": [{{"heading": "descriptive subtopic name", "citation_ids": ["arxiv_id_1", ...]}}]}}

Rules:
- Each subtopic must have at least 1 citation
- All {len(node.citation_ids)} citations must appear in exactly one subtopic
- Headings should be descriptive (e.g., "Attention mechanisms for sequence modeling")
- 2-4 subtopics total
- Do NOT add any citations not in the list above
- Group by thematic similarity based on the paper content and citing context"""

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )
            result = json.loads(response.text)
            subtopics = result.get("subtopics", [])

            # Validate
            if len(subtopics) < 2:
                print(f"    Gemini returned {len(subtopics)} subtopics, need >= 2")
                return None

            # Check all citations assigned
            assigned = set()
            for st in subtopics:
                for cid in st.get("citation_ids", []):
                    assigned.add(cid)

            original = set(node.citation_ids)
            missing = original - assigned
            extra = assigned - original

            if extra:
                # Remove hallucinated citations
                for st in subtopics:
                    st["citation_ids"] = [c for c in st["citation_ids"] if c in original]

            if missing:
                # Assign missing citations to the first subtopic
                subtopics[0]["citation_ids"].extend(list(missing))
                print(f"    Reassigned {len(missing)} missing citations to first subtopic")

            # Remove empty subtopics
            subtopics = [st for st in subtopics if st.get("citation_ids")]
            if len(subtopics) < 2:
                return None

            return subtopics

        except Exception as e:
            print(f"    Gemini attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# Heading refinement
# ---------------------------------------------------------------------------

def gemini_refine_heading(
    node: TreeNode,
    tree: SubsectionTree,
    gemini_client,
    gemini_model: str,
    previous_headings: list[str],
    max_retries: int = 3,
) -> str | None:
    """Call Gemini to generate a better heading for a node that the model struggles with.

    The refined heading should help the search model find the cited papers by
    being more descriptive of the specific methods/topics covered.

    Returns a new heading string, or None on failure.
    """
    cited_block = _format_cited_papers_block(node.citation_ids, tree.cited_papers)
    cite_sentences = _format_citation_sentences(
        node.citation_ids, tree.citation_sentence_map
    )
    tree_view = _format_tree_view(node, tree)

    tried_block = ""
    if previous_headings:
        tried_block = f"""
=== PREVIOUSLY TRIED HEADINGS (did not improve recall) ===
{chr(10).join(f'- "{h}"' for h in previous_headings)}
"""

    prompt = f"""You are helping improve a Related Work subtopic heading for a citation prediction task.
A search model is given the heading and must find the cited papers using a semantic search engine.
The current heading is too vague — the model fails to find the papers. Generate a BETTER heading
that is more descriptive and would help a search model identify the right topics to search for.

=== SOURCE PAPER ===
Title: {tree.title}
Abstract: {tree.abstract}

Introduction:
{tree.introduction}

=== FULL RELATED WORK SECTION ===
{tree.full_related_work_text}

=== CURRENT DECOMPOSITION TREE ===
{tree_view}

=== CURRENT HEADING ===
"{node.heading}"

=== PAPERS TO FIND ===
{cited_block}

=== WHERE EACH PAPER IS CITED ===
{cite_sentences}
{tried_block}
=== TASK ===
Generate a single, concise Related Work subsection heading (like "Attention Mechanisms for Sequence Modeling"
or "Dense Retrieval Methods for Open-Domain QA") that captures the specific methods, tasks, or concepts
covered by these {len(node.citation_ids)} papers. Keep it short (3-10 words) but more specific than the current heading.
{"Make sure your new heading is DIFFERENT from all previously tried headings listed above." if previous_headings else ""}

Return ONLY valid JSON:
{{"heading": "your improved heading here"}}"""

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )
            result = json.loads(response.text)
            heading = result.get("heading", "").strip()
            if heading and heading != node.heading:
                return heading
            return None
        except Exception as e:
            print(f"    Gemini refine attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


def try_heading_refinement(
    node: TreeNode,
    tree: SubsectionTree,
    llm: LLM,
    tokenizer,
    config: DecomposeConfig,
    search_url: str,
    gemini_client,
) -> None:
    """Try to improve a node's recall by refining its heading via Gemini.

    Creates a single child node with a refined heading (same citations).
    Iterates up to max_heading_refine times, keeping the best refinement.
    """
    indent = "  " * node.depth
    best_node = node
    best_recall = node.best_of_k_recall
    previous_headings = [node.heading]

    for attempt in range(config.max_heading_refine):
        print(f"{indent}  Heading refinement {attempt + 1}/{config.max_heading_refine}...")
        new_heading = gemini_refine_heading(
            best_node, tree, gemini_client, config.gemini_model, previous_headings,
        )
        if not new_heading:
            print(f"{indent}    Gemini couldn't generate a new heading")
            break

        previous_headings.append(new_heading)
        print(f"{indent}    Trying: \"{new_heading[:80]}...\"" if len(new_heading) > 80
              else f"{indent}    Trying: \"{new_heading}\"")

        # Create a candidate child node with refined heading
        child = make_child_node(
            parent=node,
            child_idx=0,
            heading=new_heading,
            citation_ids=node.citation_ids,
            paragraph_text=node.paragraph_text,
        )
        tree.add_node(child)

        # Evaluate
        evaluate_nodes([child], tree, llm, tokenizer, config, search_url)
        print(f"{indent}    Recall: best={child.best_of_k_recall:.2f}, mean={child.mean_recall:.2f}")

        if child.best_of_k_recall > best_recall:
            # Improvement — keep this child, update tracking
            best_recall = child.best_of_k_recall
            best_node = child
            node.is_leaf = False
            node.children = [child.node_id]
            child.is_leaf = True
            print(f"{indent}    Improved! ({node.best_of_k_recall:.2f} -> {best_recall:.2f})")

            if best_recall >= config.recall_threshold:
                print(f"{indent}    Reached threshold {config.recall_threshold}")
                break

            # For next iteration, refine from the current best child
            # Future children will be children of this child
            node = child
        else:
            # No improvement — remove candidate and try again with a different heading
            tree.remove_node(child.node_id)
            print(f"{indent}    No improvement, trying another heading...")


# ---------------------------------------------------------------------------
# Core recursive decomposition
# ---------------------------------------------------------------------------

def process_node(
    node: TreeNode,
    tree: SubsectionTree,
    llm: LLM,
    tokenizer,
    config: DecomposeConfig,
    search_url: str,
    gemini_client,
) -> None:
    """Recursively evaluate and split a node."""
    indent = "  " * node.depth
    print(f"{indent}Processing node '{node.heading}' "
          f"(depth={node.depth}, citations={len(node.citation_ids)})")

    # 1. Evaluate this node if not already evaluated
    if node.best_of_k_recall is None:
        evaluate_nodes([node], tree, llm, tokenizer, config, search_url)
    print(f"{indent}  Recall: best={node.best_of_k_recall:.2f}, mean={node.mean_recall:.2f}")

    # 2. Check if already tractable
    if node.best_of_k_recall >= config.recall_threshold:
        node.is_leaf = True
        print(f"{indent}  -> Leaf (tractable: recall >= {config.recall_threshold})")
        return

    # 3. Check if unsplittable — fall back to heading refinement
    if len(node.citation_ids) < config.min_citations:
        print(f"{indent}  Too few citations ({len(node.citation_ids)}) to split, trying heading refinement...")
        try_heading_refinement(node, tree, llm, tokenizer, config, search_url, gemini_client)
        return

    if node.depth >= config.max_depth:
        print(f"{indent}  Max depth {config.max_depth}, trying heading refinement...")
        try_heading_refinement(node, tree, llm, tokenizer, config, search_url, gemini_client)
        return

    # 4. Call Gemini to split
    print(f"{indent}  Splitting via Gemini...")
    subtopics = gemini_split(node, tree, gemini_client, config.gemini_model)
    if not subtopics or len(subtopics) < 2:
        print(f"{indent}  Gemini couldn't split, trying heading refinement...")
        try_heading_refinement(node, tree, llm, tokenizer, config, search_url, gemini_client)
        return

    # 5. Create children — pass parent's paragraph text so deeper splits
    #    still have full context for Gemini to work with
    children = []
    for i, st in enumerate(subtopics):
        child = make_child_node(
            parent=node,
            child_idx=i,
            heading=st["heading"],
            citation_ids=st["citation_ids"],
            paragraph_text=node.paragraph_text,
        )
        children.append(child)
        tree.add_node(child)

    node.children = [c.node_id for c in children]

    # 6. Evaluate all children in a batch
    print(f"{indent}  Evaluating {len(children)} children...")
    evaluate_nodes(children, tree, llm, tokenizer, config, search_url)

    for child in children:
        print(f"{indent}    Child '{child.heading}': "
              f"citations={len(child.citation_ids)}, "
              f"recall={child.best_of_k_recall:.2f}")

    # 7. Check improvement: if mean child recall doesn't beat parent, revert and try heading refinement
    mean_child_recall = sum(c.best_of_k_recall for c in children) / len(children)
    if mean_child_recall <= node.best_of_k_recall:
        print(f"{indent}  Reverting split: mean child recall {mean_child_recall:.2f} "
              f"<= parent {node.best_of_k_recall:.2f}")
        for child in children:
            tree.remove_node(child.node_id)
        node.children = []
        node.split_reverted = True
        # Fall back to heading refinement
        print(f"{indent}  Trying heading refinement instead...")
        try_heading_refinement(node, tree, llm, tokenizer, config, search_url, gemini_client)
        return

    print(f"{indent}  Split accepted: mean child recall {mean_child_recall:.2f} "
          f"> parent {node.best_of_k_recall:.2f}")

    # 8. Commit split, recurse on children that need it
    node.is_leaf = False
    for child in children:
        if (child.best_of_k_recall < config.recall_threshold
                and len(child.citation_ids) >= config.min_citations
                and child.depth < config.max_depth):
            process_node(child, tree, llm, tokenizer, config, search_url, gemini_client)
        else:
            child.is_leaf = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive hierarchical decomposition of Related Work subsections."
    )
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to subsection_corpus.json")
    parser.add_argument("--search_url", type=str, required=True,
                        help="Retriever URL (e.g., http://host:8000/retrieve)")
    parser.add_argument("--model", type=str, required=True,
                        help="vLLM model path")
    parser.add_argument("--output", type=str, default="trees.json",
                        help="Output JSON file for completed trees")
    parser.add_argument("--checkpoint", type=str, default="trees_checkpoint.jsonl",
                        help="JSONL checkpoint file (append-only)")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "validation", "test"],
                        help="Only process this split")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of subsections to process")
    parser.add_argument("--prompt_ids_file", type=str, default=None,
                        help="JSON file with list of {paper_id, subsection_heading} to filter")
    # Eval params
    parser.add_argument("--num_samples", type=int, default=8,
                        help="k rollouts per node eval")
    parser.add_argument("--max_turns", type=int, default=4,
                        help="m search turns per rollout")
    parser.add_argument("--topk", type=int, default=5,
                        help="n retriever results per search")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    # Decomposition params
    parser.add_argument("--min_citations", type=int, default=2,
                        help="Stop splitting below this many citations")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Max tree depth")
    parser.add_argument("--recall_threshold", type=float, default=0.3,
                        help="Tractable if best-of-k recall >= this")
    parser.add_argument("--gemini_model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model for splitting")
    parser.add_argument("--max_heading_refine", type=int, default=1,
                        help="Max heading refinement attempts per node")
    # vLLM params
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Validate Gemini API key and create client
    # ---------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY env var not set")
        sys.exit(1)

    from google import genai
    gemini_client = genai.Client(api_key=api_key)

    # ---------------------------------------------------------------
    # Build config
    # ---------------------------------------------------------------
    config = DecomposeConfig(
        min_citations=args.min_citations,
        max_depth=args.max_depth,
        recall_threshold=args.recall_threshold,
        num_samples=args.num_samples,
        max_turns=args.max_turns,
        topk=args.topk,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gemini_model=args.gemini_model,
        seed=args.seed,
        max_heading_refine=args.max_heading_refine,
    )

    # ---------------------------------------------------------------
    # Load corpus
    # ---------------------------------------------------------------
    print(f"Loading corpus from {args.corpus}...")
    with open(args.corpus) as f:
        corpus = json.load(f)

    if args.split:
        corpus = [e for e in corpus if e["split"] == args.split]

    if args.prompt_ids_file:
        with open(args.prompt_ids_file) as f:
            prompt_ids = json.load(f)
        prompt_set = {(p["paper_id"], p["subsection_heading"]) for p in prompt_ids}
        corpus = [e for e in corpus if (e["paper_id"], e["subsection_heading"]) in prompt_set]
        print(f"Filtered to {len(corpus)} subsections matching prompt_ids_file")

    if args.max_examples and args.max_examples < len(corpus):
        corpus = corpus[:args.max_examples]

    print(f"Processing {len(corpus)} subsections")

    # ---------------------------------------------------------------
    # Load checkpoint (skip already-processed subsections)
    # ---------------------------------------------------------------
    done_keys = get_checkpoint_keys(args.checkpoint)
    if done_keys:
        print(f"Resuming: {len(done_keys)} subsections already in checkpoint")

    # ---------------------------------------------------------------
    # Load vLLM model
    # ---------------------------------------------------------------
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
    )

    # ---------------------------------------------------------------
    # Verify retriever
    # ---------------------------------------------------------------
    print(f"Checking retriever at {args.search_url}...")
    try:
        resp = requests.post(
            args.search_url,
            json={"query": "test", "topk": 1},
            timeout=10,
        )
        resp.raise_for_status()
        print("Retriever ready!")
    except Exception as e:
        print(f"ERROR: Cannot reach retriever: {e}")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Process each subsection
    # ---------------------------------------------------------------
    all_trees = []
    total_start = time.time()

    for i, entry in enumerate(corpus):
        key = (entry["paper_id"], entry["subsection_heading"])
        if key in done_keys:
            continue

        print(f"\n{'='*60}")
        print(f"Subsection {i+1}/{len(corpus)}: {entry['subsection_heading'][:60]}")
        print(f"Paper: {entry['paper_id']}, Citations: {len(entry['citation_ids'])}")
        print(f"{'='*60}")

        # Build tree with root node
        sub_idx = i
        root_id, root_node = make_root_node(
            paper_id=entry["paper_id"],
            subsection_idx=sub_idx,
            heading=entry["subsection_heading"],
            citation_ids=entry["citation_ids"],
            paragraph_text=entry["paragraph_text"],
        )

        tree = SubsectionTree(
            paper_id=entry["paper_id"],
            title=entry["title"],
            abstract=entry["abstract"],
            introduction=entry["introduction"],
            root_heading=entry["subsection_heading"],
            root_id=root_id,
            full_related_work_text=entry.get("full_related_work_text", ""),
            citation_sentence_map=entry.get("citation_sentence_map", {}),
            cited_papers=entry.get("cited_papers", {}),
        )
        tree.add_node(root_node)

        # Run adaptive decomposition
        process_node(root_node, tree, llm, tokenizer, config, args.search_url, gemini_client)

        # Validate
        issues = tree.validate()
        if issues:
            print(f"  WARNING: Tree validation issues: {issues}")

        # Print summary
        leaves = tree.get_leaves()
        leaf_recalls = [l.best_of_k_recall for l in leaves if l.best_of_k_recall is not None]
        tractable = sum(1 for r in leaf_recalls if r >= config.recall_threshold)
        print(f"  Tree summary: {len(tree.nodes)} nodes, {len(leaves)} leaves, "
              f"{tractable}/{len(leaves)} tractable")
        if leaf_recalls:
            print(f"  Leaf recalls: min={min(leaf_recalls):.2f}, "
                  f"max={max(leaf_recalls):.2f}, "
                  f"mean={sum(leaf_recalls)/len(leaf_recalls):.2f}")

        # Save checkpoint
        save_checkpoint(tree, args.checkpoint)
        all_trees.append(tree)

    # ---------------------------------------------------------------
    # Load any previously checkpointed trees and merge
    # ---------------------------------------------------------------
    if done_keys:
        prev_trees = load_checkpoint(args.checkpoint)
        # Deduplicate: prev_trees might include what we just appended
        seen = set()
        merged = []
        for t in prev_trees:
            k = (t.paper_id, t.root_heading)
            if k not in seen:
                seen.add(k)
                merged.append(t)
        all_trees = merged

    # ---------------------------------------------------------------
    # Save final output
    # ---------------------------------------------------------------
    save_trees(all_trees, args.output)

    elapsed = time.time() - total_start

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    total_nodes = sum(len(t.nodes) for t in all_trees)
    total_leaves = sum(len(t.get_leaves()) for t in all_trees)
    all_leaf_recalls = []
    for t in all_trees:
        for leaf in t.get_leaves():
            if leaf.best_of_k_recall is not None:
                all_leaf_recalls.append(leaf.best_of_k_recall)

    tractable_count = sum(1 for r in all_leaf_recalls if r >= config.recall_threshold)
    zero_count = sum(1 for r in all_leaf_recalls if r == 0.0)

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Trees: {len(all_trees)}")
    print(f"Total nodes: {total_nodes}")
    print(f"Total leaves: {total_leaves}")
    if all_leaf_recalls:
        print(f"Leaf recall: mean={sum(all_leaf_recalls)/len(all_leaf_recalls):.2f}, "
              f"min={min(all_leaf_recalls):.2f}, max={max(all_leaf_recalls):.2f}")
        print(f"Tractable (>= {config.recall_threshold}): "
              f"{tractable_count}/{total_leaves} ({tractable_count/total_leaves*100:.0f}%)")
        print(f"Zero recall: {zero_count}/{total_leaves} ({zero_count/total_leaves*100:.0f}%)")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
