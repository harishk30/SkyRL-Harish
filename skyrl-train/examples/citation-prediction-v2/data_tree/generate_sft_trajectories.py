#!/usr/bin/env python3
"""Generate SFT training trajectories using Gemini with retriever-in-the-loop.

Runs Gemini through the same multi-turn search pipeline as vLLM evaluation:
same system prompt, same retriever, same tag format. Gemini does NOT see the
ground-truth answers — it must search authentically.

Output trajectories are in Qwen3 chat format (system/user/assistant turns)
for direct use in SFT fine-tuning.

Usage:
    python generate_sft_trajectories.py \
        --corpus subsection_corpus.json \
        --search_url http://retriever-host:8000/retrieve \
        --output sft_trajectories.json \
        --num_samples 5 \
        --max_examples 5 \
        --gemini_model gemini-3.1-pro-preview
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Reuse system prompt and utilities from adaptive_decompose
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

# Gemini-specific prompt: addresses failure modes observed in testing
GEMINI_SYSTEM_PROMPT = (
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
    "When you find a relevant paper in the search results, cite it IMMEDIATELY "
    "OUTSIDE of <think> tags using <citation>arxiv_id</citation>. "
    "You can include multiple IDs: <citation>XXXX.XXXXX, YYYY.YYYYY</citation>.\n\n"
    "CRITICAL RULES:\n"
    "1. NEVER put <citation> tags inside <think> blocks. Citations inside <think> "
    "are IGNORED. Always close </think> first, then write <citation>.\n"
    "2. Cite IMMEDIATELY after each search when you find relevant papers. Do NOT "
    "accumulate papers to cite later — you may run out of turns.\n"
    "3. Every <citation> tag across the entire conversation counts. Do not duplicate "
    "IDs you already cited.\n"
    "4. ONLY cite arxiv IDs that appear in [arxiv:ID] format in search results. "
    "Never guess or fabricate IDs.\n"
    "5. The subsection heading hints at the MAIN topic, but the actual citations may "
    "also cover related methods, datasets, or techniques mentioned in the introduction. "
    "Read the introduction carefully for clues about what else might be cited.\n\n"
    "You have a citation budget (shown in the prompt). Exceeding the budget results in "
    "zero reward. Track your citations carefully.\n\n"
    "When you have found all cited papers, write <done></done> to finish.\n\n"
    "Additional search tips:\n"
    "- The introduction mentions author names (e.g., \"Smith et al.\") — these are not "
    "searchable, but the topics and methods they describe ARE.\n"
    "- Be persistent: search from multiple angles. Rephrase queries, try specific "
    "method names, dataset names, or task descriptions.\n"
    "- Think about what SPECIFIC papers the authors would cite — not just famous papers "
    "in the area, but the particular ones referenced by the author names in the intro.\n"
    "- Keep your reasoning concise to save token budget for more search turns.\n\n"
    "Example workflow:\n"
    "<think>The subsection is about \"Transfer Learning\". The intro mentions Smith et al. "
    "on domain adaptation and Jones et al. on pre-training. Let me search.</think>\n"
    "<search>transfer learning pretrained language models fine-tuning</search>\n"
    "[Results appear in <information> tags]\n"
    "<think>Doc 3 matches the pre-training work mentioned in the intro.</think>\n"
    "<citation>XXXX.XXXXX</citation>\n"
    "<search>domain adaptation neural networks distribution shift</search>\n"
    "[More results...]\n"
    "<think>Doc 1 and Doc 4 are relevant.</think>\n"
    "<citation>YYYY.YYYYY, ZZZZ.ZZZZZ</citation>\n"
    "<think>I've covered the main topics. Finishing.</think>\n"
    "<done></done>"
)

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
    # Strip <think>...</think> blocks to avoid counting citations from reasoning
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    ids = set()
    for match in CITATION_TAG_RE.finditer(cleaned):
        raw = match.group(1)
        for part in raw.split(","):
            part = part.strip()
            if part:
                normalized = normalize_arxiv_id(part)
                if normalized and re.match(r"\d{4}\.\d{4,5}$", normalized):
                    ids.add(normalized)
    return ids


def compute_recall(messages: list[dict], targets: list[str]) -> float:
    full_text = "".join(m["content"] for m in messages)
    predicted = extract_all_citations(full_text)
    gt_set = {normalize_arxiv_id(t) for t in targets}
    gt_set.discard("")
    if not gt_set:
        return 0.0
    if len(predicted) > 2.0 * len(gt_set):
        return 0.0
    correct = predicted & gt_set
    return len(correct) / len(gt_set)


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


# ---------------------------------------------------------------------------
# Gemini multi-turn rollout
# ---------------------------------------------------------------------------

def run_gemini_rollout(
    messages: list[dict],
    targets: list[str],
    gemini_client,
    gemini_model: str,
    search_url: str,
    max_turns: int = 4,
    topk: int = 5,
    max_tokens: int = 1024,
    temperature: float = 1.0,
) -> dict:
    """Run a single multi-turn rollout using Gemini with retriever-in-the-loop.

    Returns dict with messages, recall, num_turns, and success flag.
    """
    # Convert our messages to Gemini format
    # Gemini uses "user" and "model" roles, and system instruction separately
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]

    # Build Gemini conversation history
    gemini_history = []
    current_messages = list(messages)  # our tracking copy

    # First user message
    gemini_contents = [{"role": "user", "parts": [{"text": user_content}]}]

    for turn in range(max_turns):
        # Call Gemini with retry on 500 errors
        assistant_text = None
        for attempt in range(3):
            try:
                response = gemini_client.models.generate_content(
                    model=gemini_model,
                    contents=gemini_contents,
                    config={
                        "system_instruction": system_content,
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                        "stop_sequences": ["</search>", "</done>"],
                    },
                )
                assistant_text = response.text or ""
                break
            except Exception as e:
                print(f"    Gemini API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1s, 2s backoff
        if assistant_text is None:
            print(f"    All retries failed, ending rollout")
            break

        # Re-append stop string if truncated (Gemini strips stop sequences)
        if "<search>" in assistant_text and "</search>" not in assistant_text:
            assistant_text += "</search>"
        if "<done>" in assistant_text and "</done>" not in assistant_text:
            assistant_text += "</done>"

        current_messages.append({"role": "assistant", "content": assistant_text})
        gemini_contents.append({"role": "model", "parts": [{"text": assistant_text}]})

        # Check if done
        if "<done>" in assistant_text:
            break

        # Extract search query
        match = re.search(r"<search>(.*?)</search>", assistant_text, re.DOTALL)
        if match is None:
            # No search tag — let it continue
            continue

        query = match.group(1).strip()
        tool_output = call_retriever(search_url, query, topk)
        observation = "\n<information>" + tool_output + "</information>\n"

        remaining = max_turns - (turn + 1)
        if remaining > 1:
            observation += f"\n\n{remaining} turns remaining."
        elif remaining == 1:
            observation += "\n\nThis is your last turn. Cite remaining papers and write <done></done>."

        current_messages.append({"role": "user", "content": observation})
        gemini_contents.append({"role": "user", "parts": [{"text": observation}]})

    recall = compute_recall(current_messages, targets)
    return {
        "messages": current_messages,
        "recall": recall,
        "num_turns": turn + 1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories with Gemini")
    parser.add_argument("--corpus", required=True, help="Subsection corpus JSON")
    parser.add_argument("--search_url", required=True, help="Retriever URL")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Rollouts per subsection (keep best)")
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--recall_threshold", type=float, default=0.3,
                        help="Minimum recall to keep a trajectory")
    parser.add_argument("--gemini_model", type=str, default="gemini-3.1-pro-preview")
    parser.add_argument("--prompt_ids_file", type=str, default=None,
                        help="JSON file with list of {paper_id, subsection_heading} to filter")
    args = parser.parse_args()

    # Validate Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY env var not set")
        sys.exit(1)

    from google import genai
    gemini_client = genai.Client(api_key=api_key)

    # Load corpus
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

    print(f"Processing {len(corpus)} subsections, {args.num_samples} rollout(s) each")
    print(f"Config: m={args.max_turns}, n={args.topk}, model={args.gemini_model}")
    print(f"Recall threshold: {args.recall_threshold}")

    # Check retriever
    print(f"Checking retriever at {args.search_url}...")
    test_result = call_retriever(args.search_url, "test query", topk=1)
    if "error" in test_result.lower():
        print(f"WARNING: Retriever may not be ready: {test_result}")
    else:
        print("Retriever ready!")

    results = []
    total_kept = 0
    total_attempted = 0

    for idx, entry in enumerate(corpus):
        paper_id = entry["paper_id"]
        heading = entry["subsection_heading"]
        citation_ids = entry["citation_ids"]
        num_cites = len(citation_ids)

        print(f"\n{'='*60}")
        print(f"Subsection {idx+1}/{len(corpus)}: {heading}")
        print(f"Paper: {paper_id}, Citations: {num_cites}")
        print(f"{'='*60}")

        # Build initial messages
        user_content = format_user_prompt(
            entry["title"], entry["abstract"], entry["introduction"],
            heading, num_cites,
        )
        initial_messages = [
            {"role": "system", "content": GEMINI_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        best_result = None
        best_recall = -1

        for sample_idx in range(args.num_samples):
            total_attempted += 1
            print(f"  Rollout {sample_idx+1}/{args.num_samples}...")

            result = run_gemini_rollout(
                messages=initial_messages,
                targets=citation_ids,
                gemini_client=gemini_client,
                gemini_model=args.gemini_model,
                search_url=args.search_url,
                max_turns=args.max_turns,
                topk=args.topk,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            # Debug: show citations extracted and targets
            full_text = "".join(m["content"] for m in result["messages"])
            predicted = extract_all_citations(full_text)
            gt_set = {normalize_arxiv_id(t) for t in citation_ids}
            print(f"    Recall: {result['recall']:.2f}, Turns: {result['num_turns']}")
            print(f"    Predicted: {predicted}")
            print(f"    Targets:   {gt_set}")
            if sample_idx == 0 and idx < 3:
                # Print full trajectory for first rollout of first 3 subsections only
                print(f"    --- Full trajectory ---")
                for m in result["messages"][2:]:
                    role = m["role"]
                    content = m["content"]
                    print(f"    [{role}]: {content}")
                print(f"    --- End trajectory ---")

            if result["recall"] > best_recall:
                best_recall = result["recall"]
                best_result = result

            # Rate limit
            time.sleep(1)

        if best_result and best_recall >= args.recall_threshold:
            total_kept += 1
            results.append({
                "paper_id": paper_id,
                "subsection_heading": heading,
                "citation_ids": citation_ids,
                "recall": best_recall,
                "num_turns": best_result["num_turns"],
                "messages": best_result["messages"],
            })
            print(f"  KEPT (recall={best_recall:.2f})")
        else:
            print(f"  DISCARDED (best recall={best_recall:.2f} < {args.recall_threshold})")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Attempted: {total_attempted} rollouts across {len(corpus)} subsections")
    print(f"  Kept: {total_kept}/{len(corpus)} subsections ({100*total_kept/max(len(corpus),1):.0f}%)")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
