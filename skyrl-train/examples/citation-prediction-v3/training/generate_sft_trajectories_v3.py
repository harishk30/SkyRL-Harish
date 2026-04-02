#!/usr/bin/env python3
"""Generate SFT trajectories using Gemini with retriever-in-the-loop.

Exactly replicates the training environment trajectory format:
- System/user prompts matching v3 parquet generation
- Observation format matching CitationPredictionV2Env (env.py)
- Turn counter with citation count matching env.py
- Stop sequences: </search>, </done>
- max_output_tokens=1024 matching generator.sampling_params.max_generate_length

Pass@k rejection sampling: runs N rollouts per query, keeps ALL trajectories
with recall > 0 (not just the best one).

Usage:
    python generate_sft_trajectories_v3.py \
        --corpus /path/to/subsection_corpus_v3.json \
        --search_url http://localhost:8000/retrieve \
        --output sft_trajectories.json \
        --num_samples 20 \
        --sample_frac 0.5 \
        --max_turns 6 \
        --topk 5 \
        --prompt_version v3
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from collections import deque

import requests

# Must import before use in rollout
from google.genai import types as genai_types


# ---------------------------------------------------------------------------
# Rate limiter for Gemini API (1K RPM, 10K RPD for Flash)
# ---------------------------------------------------------------------------

class GeminiRateLimiter:
    """Token-bucket style rate limiter respecting RPM and RPD limits."""

    def __init__(self, rpm: int = 900, rpd: int = 9500):
        # Use 90% of limits to leave headroom
        self.rpm = rpm
        self.rpd = rpd
        self._minute_window: deque[float] = deque()
        self._day_start: float = time.time()
        self._day_count: int = 0

    def wait(self) -> None:
        now = time.time()

        # Reset daily counter if >24h elapsed
        if now - self._day_start > 86400:
            self._day_start = now
            self._day_count = 0

        # Check daily limit
        if self._day_count >= self.rpd:
            sleep_until = self._day_start + 86400
            wait_secs = max(0, sleep_until - now)
            print(f"    [rate-limit] Daily limit ({self.rpd}) hit, sleeping {wait_secs/3600:.1f}h",
                  file=sys.stderr)
            time.sleep(wait_secs + 1)
            self._day_start = time.time()
            self._day_count = 0

        # Prune minute window
        while self._minute_window and self._minute_window[0] < now - 60:
            self._minute_window.popleft()

        # Check per-minute limit
        if len(self._minute_window) >= self.rpm:
            wait_secs = 60 - (now - self._minute_window[0]) + 0.5
            if wait_secs > 0:
                print(f"    [rate-limit] RPM limit, sleeping {wait_secs:.1f}s",
                      file=sys.stderr)
                time.sleep(wait_secs)

        self._minute_window.append(time.time())
        self._day_count += 1

    @property
    def stats(self) -> str:
        return f"RPM={len(self._minute_window)}/{self.rpm}, RPD={self._day_count}/{self.rpd}"

# ---------------------------------------------------------------------------
# System prompts — identical to generate_v3_parquet.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V1 = (
    "You are a research assistant helping an author find papers to cite in their Related Work section. "
    "You have access to a semantic search engine over arxiv. "
    "Results are formatted as: [arxiv:ID] \"Title\" Authors... Abstract...\n\n"
    "Use <think>...</think> to reason, <search>query</search> to search (use descriptive keywords, "
    "NOT author names or IDs), and <citation>arxiv_id</citation> to cite. "
    "Cite papers as you find them — all <citation> tags count toward your predictions. "
    "Write <done></done> when finished.\n\n"
    "Only cite IDs from search results. Stay within the citation budget shown in the prompt — "
    "exceeding it results in zero reward.\n\n"
    "Example:\n"
    "<think>Search for transfer learning methods.</think>\n"
    "<search>transfer learning pretrained language models</search>\n"
    "[Results in <information> tags]\n"
    "<think>Doc 3 matches. Need domain adaptation papers too.</think>\n"
    "<citation>XXXX.XXXXX</citation>\n"
    "<search>domain adaptation distribution shift</search>\n"
    "[Results]\n"
    "<citation>YYYY.YYYYY</citation>\n"
    "<done></done>"
)

SYSTEM_PROMPT_V2 = (
    "You are a research assistant helping an author find papers to cite in their Related Work section. "
    "You have access to a semantic search engine over arxiv. "
    "Results are formatted as: [arxiv:ID] \"Title\" Authors... Abstract...\n\n"
    "Use <think>...</think> to reason, <search>query</search> to search (use descriptive keywords, "
    "NOT author names or IDs), and <citation>arxiv_id</citation> to cite. "
    "Write <done></done> when finished.\n\n"
    "Important:\n"
    "- ONLY cite arxiv IDs that appear in search results. Never fabricate or guess IDs.\n"
    "- Do NOT speculate about specific arxiv IDs or paper metadata in your thinking — "
    "only reference papers after you have seen them in search results.\n"
    "- After each search, carefully check which results actually match the subtopic "
    "before citing. Not every result is relevant.\n\n"
    "Example:\n"
    "<think>Search for transfer learning methods.</think>\n"
    "<search>transfer learning pretrained language models</search>\n"
    "[Results in <information> tags]\n"
    "<think>Doc 3 is about pre-trained models for NLP — matches. "
    "Doc 7 is about vision transformers — not relevant. "
    "Still need domain adaptation papers.</think>\n"
    "<citation>XXXX.XXXXX</citation>\n"
    "<search>domain adaptation distribution shift</search>\n"
    "[Results]\n"
    "<think>Doc 1 addresses domain shift in NLP — relevant.</think>\n"
    "<citation>YYYY.YYYYY</citation>\n"
    "<done></done>"
)

SYSTEM_PROMPT_V3 = (
    "You are a research assistant helping an author find papers to cite in their Related Work section. "
    "You have access to a semantic search engine over arxiv. "
    "Results are formatted as: [arxiv:ID] \"Title\" Authors... Abstract...\n\n"
    "Use <think>...</think> to reason, <search>query</search> to search (use descriptive keywords, "
    "NOT author names or IDs), and <citation>arxiv_id</citation> to cite. "
    "Write <done></done> when finished.\n\n"
    "Guidelines:\n"
    "- Think briefly before searching. Do NOT try to recall specific arxiv IDs or paper details from memory — use search to find them.\n"
    "- After each search, reason about which results match the subtopic before continuing. "
    "If a result looks like a strong match for the Related Work, cite it right away before your next search.\n"
    "- Only cite IDs that appeared in search results. Stay within the citation budget shown in the prompt.\n\n"
    "Example:\n"
    "<think>I need papers on transfer learning for NLP.</think>\n"
    "<search>transfer learning pretrained language models</search>\n"
    "[Results in <information> tags]\n"
    "<think>Doc 3 is about pre-trained models for NLP — good match.</think>\n"
    "<citation>XXXX.XXXXX</citation>\n"
    "<search>domain adaptation distribution shift</search>\n"
    "[Results]\n"
    "<think>Doc 1 addresses domain shift — relevant.</think>\n"
    "<citation>YYYY.YYYYY</citation>\n"
    "<done></done>"
)

SYSTEM_PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2, "v3": SYSTEM_PROMPT_V3}

# ---------------------------------------------------------------------------
# Citation extraction — identical to env utils.py
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
    """Extract arxiv IDs from <citation> tags (matching env utils.py)."""
    ids: set[str] = set()
    for match in CITATION_TAG_RE.finditer(text):
        raw = match.group(1)
        for part in raw.split(","):
            part = part.strip()
            if part:
                normalized = normalize_arxiv_id(part)
                if normalized and re.match(r"\d{4}\.\d{4,5}$", normalized):
                    ids.add(normalized)
    return ids


def compute_recall(messages: list[dict], targets: list[str], max_ratio: float = 2.0) -> float:
    """Compute recall reward matching env compute_recall_reward."""
    full_text = "".join(m["content"] for m in messages)
    predicted = extract_all_citations(full_text)
    gt_set = {normalize_arxiv_id(t) for t in targets}
    gt_set.discard("")
    if not gt_set:
        return 0.0
    if len(predicted) > max_ratio * len(gt_set):
        return 0.0
    correct = predicted & gt_set
    return len(correct) / len(gt_set)


# ---------------------------------------------------------------------------
# User prompt — identical to generate_v3_parquet.py format_user_prompt
# ---------------------------------------------------------------------------

def format_user_prompt(
    title: str,
    abstract: str,
    introduction: str,
    rich_query: str,
    num_citations: int,
) -> str:
    budget = num_citations * 2
    parts = [
        f"Paper title: {title}",
        f"\nAbstract:\n{abstract}",
        f"\nIntroduction:\n{introduction}",
        f"\n{rich_query}",
        f"\nCitation budget: at most {budget} citations (you may cite fewer — only cite papers you are confident about). "
        "Exceeding this budget results in zero reward.",
        "\nIdentify all papers that would be cited in this Related Work subtopic. "
        "Search for them and report their arxiv IDs using <citation> tags.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Retriever call — matches env _do_search output format exactly
# ---------------------------------------------------------------------------

def call_retriever(search_url: str, query: str, topk: int = 5, timeout: int = 30) -> str:
    """Call retriever and return observation string matching env._do_search format.

    Returns the EXACT string that CitationPredictionV2Env._do_search produces:
        \\n<information>{"result": "Doc 1: ...\\nDoc 2: ...\\n"}</information>\\n
    """
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
        if not raw_results:
            return "\n<information>" + json.dumps({"result": "No search results found."}) + "</information>\n"

        # Format matching env._do_search exactly
        pretty_parts = []
        for retrieval in raw_results:
            formatted = ""
            for idx, doc_item in enumerate(retrieval):
                content = doc_item["document"]["contents"].strip()
                formatted += f"Doc {idx+1}: {content}\n"
            pretty_parts.append(formatted)
        final_result = "\n---\n".join(pretty_parts)
        return "\n<information>" + json.dumps({"result": final_result}) + "</information>\n"

    except Exception as e:
        return "\n<information>" + json.dumps({"result": f"Search error: {e}"}) + "</information>\n"


# ---------------------------------------------------------------------------
# Turn counter — matches env.py step() lines 166-172 exactly
# ---------------------------------------------------------------------------

def format_turn_counter(
    remaining: int,
    num_cited: int,
    citation_budget: int,
) -> str:
    """Produce the turn counter string matching env.py exactly."""
    if remaining > 1:
        return (
            f"\n\n{remaining} turns remaining. "
            f"Citations so far: {num_cited}/{citation_budget} max. "
            "You may cite fewer than the max — only cite papers you are confident belong in this subsection."
        )
    elif remaining == 1:
        return (
            f"\n\nThis is your last turn. "
            f"Citations so far: {num_cited}/{citation_budget} max. "
            "Cite any remaining papers and write <done></done>."
        )
    return ""


# ---------------------------------------------------------------------------
# Gemini multi-turn rollout
# ---------------------------------------------------------------------------

def run_gemini_rollout(
    system_prompt: str,
    user_prompt: str,
    targets: list[str],
    citation_budget: int,
    gemini_client,
    gemini_model: str,
    search_url: str,
    max_turns: int = 6,
    topk: int = 5,
    max_tokens: int = 1024,
    rate_limiter: GeminiRateLimiter | None = None,
) -> dict:
    """Run a single multi-turn rollout replicating the training env exactly.

    The loop mirrors CitationPredictionV2Env.step():
    1. Model generates with stop_sequences=["</search>", "</done>"]
    2. Re-append stripped stop sequence (Gemini strips it)
    3. If <done>: done
    4. If <search>: call retriever, format observation matching env._do_search
    5. Append turn counter matching env.step() format
    6. Increment turn count; if turns >= max_turns: done
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    gemini_contents = [{"role": "user", "parts": [{"text": user_prompt}]}]

    turns = 0

    for _ in range(max_turns):
        # Call Gemini with internal thinking enabled
        if rate_limiter:
            rate_limiter.wait()
        response = None
        for attempt in range(5):
            try:
                response = gemini_client.models.generate_content(
                    model=gemini_model,
                    contents=gemini_contents,
                    config={
                        "system_instruction": system_prompt,
                        "max_output_tokens": max_tokens,
                        "stop_sequences": ["</search>", "</done>"],
                        "thinking_config": genai_types.ThinkingConfig(
                            thinking_budget=-1,      # AUTO: model decides
                            include_thoughts=True,   # expose internal reasoning
                        ),
                    },
                )
                break
            except Exception as e:
                err_str = str(e)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                wait = (30 * (attempt + 1)) if is_429 else (2 ** attempt)
                print(f"    Gemini error (attempt {attempt+1}/5): {e}", file=sys.stderr)
                if attempt < 4:
                    print(f"    Retrying in {wait}s...", file=sys.stderr)
                    time.sleep(wait)
        if response is None:
            print("    All retries failed, ending rollout", file=sys.stderr)
            break

        # Extract internal thoughts and visible output separately
        parts = response.candidates[0].content.parts
        think_text = ""
        output_text = ""
        for p in parts:
            if getattr(p, "thought", False):
                think_text += (p.text or "")
            else:
                output_text += (p.text or "")
        think_text = think_text.strip()
        output_text = output_text.strip()

        # Inject internal thoughts as <think> tags if the visible output
        # doesn't already have them (model may write its own <think> blocks
        # since the system prompt mentions them — if so, use those as-is)
        if think_text and "<think>" not in output_text:
            assistant_text = f"<think>\n{think_text}\n</think>\n{output_text}"
        else:
            assistant_text = output_text

        # Re-append stop string (Gemini strips it)
        if "<search>" in assistant_text and "</search>" not in assistant_text:
            assistant_text += "</search>"
        if "<done>" in assistant_text and "</done>" not in assistant_text:
            assistant_text += "</done>"

        messages.append({"role": "assistant", "content": assistant_text})
        gemini_contents.append({"role": "model", "parts": [{"text": assistant_text}]})

        # Increment turn count (matching env: self.turns += 1 at top of step)
        turns += 1

        # Check done (matching env: _is_done checks <done> or turns >= max_turns)
        if "<done>" in assistant_text:
            break
        if turns >= max_turns:
            break

        # Parse search query
        match = re.search(r"<search>(.*?)</search>", assistant_text, re.DOTALL)
        if match is None:
            continue

        query = match.group(1).strip()

        # Call retriever — format matches env._do_search exactly
        observation = call_retriever(search_url, query, topk)

        # Count citations so far (matching env: across full chat_history)
        chat_str = "".join(m["content"] for m in messages)
        num_cited = len(extract_all_citations(chat_str))

        # Turn counter (matching env: remaining = max_turns - turns)
        remaining = max_turns - turns
        turn_msg = format_turn_counter(remaining, num_cited, citation_budget)

        # Append turn counter to observation (matching env: new_obs["content"] += turn_msg)
        observation += turn_msg

        messages.append({"role": "user", "content": observation})
        gemini_contents.append({"role": "user", "parts": [{"text": observation}]})

    recall = compute_recall(messages, targets)
    return {
        "messages": messages,
        "recall": recall,
        "num_turns": turns,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT trajectories with Gemini (env-matched format)"
    )
    parser.add_argument("--corpus", required=True,
                        help="Subsection corpus JSON (subsection_corpus_v3.json)")
    parser.add_argument("--search_url", required=True, help="Retriever URL")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample_frac", type=float, default=0.5,
                        help="Fraction of training queries to sample (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Rollouts per query (pass@k)")
    parser.add_argument("--max_turns", type=int, default=4,
                        help="Max search turns (matching training config)")
    parser.add_argument("--topk", type=int, default=10,
                        help="Results per search (matching training config)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max output tokens per generation for the oracle")
    parser.add_argument("--recall_threshold", type=float, default=0.0,
                        help="Min recall to keep (0 = any correct citation)")
    parser.add_argument("--gemini_model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--prompt_version", type=str, default="v3",
                        choices=["v1", "v2", "v3"],
                        help="System prompt version (must match parquet)")
    parser.add_argument("--prompt_ids_file", type=str, default=None,
                        help="JSON list of {paper_id, subsection_heading} to filter")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Hard cap on number of queries (applied after sampling)")
    parser.add_argument("--chunk_id", type=int, default=None,
                        help="Chunk index (0-based) for parallel execution")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Total number of chunks for parallel execution")
    parser.add_argument("--rpm", type=int, default=900,
                        help="Gemini RPM limit (split across concurrent jobs)")
    parser.add_argument("--rpd", type=int, default=9500,
                        help="Gemini RPD limit (split across concurrent jobs)")
    args = parser.parse_args()

    # Validate
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from google import genai
    gemini_client = genai.Client(api_key=api_key)
    rate_limiter = GeminiRateLimiter(rpm=args.rpm, rpd=args.rpd)

    system_prompt = SYSTEM_PROMPTS[args.prompt_version]

    # Load corpus
    print(f"Loading corpus from {args.corpus}...", file=sys.stderr)
    with open(args.corpus) as f:
        corpus = json.load(f)

    if args.split:
        corpus = [e for e in corpus if e.get("split") == args.split]
    print(f"  {len(corpus)} entries in split={args.split}", file=sys.stderr)

    if args.prompt_ids_file:
        with open(args.prompt_ids_file) as f:
            prompt_ids = json.load(f)
        prompt_set = {(p["paper_id"], p["subsection_heading"]) for p in prompt_ids}
        corpus = [e for e in corpus if (e["paper_id"], e["subsection_heading"]) in prompt_set]
        print(f"  Filtered to {len(corpus)} via prompt_ids_file", file=sys.stderr)

    # Sample fraction of training queries
    if args.sample_frac < 1.0:
        random.seed(args.seed)
        n_sample = max(1, int(len(corpus) * args.sample_frac))
        corpus = random.sample(corpus, n_sample)
        print(f"  Sampled {len(corpus)} entries ({args.sample_frac*100:.0f}%)", file=sys.stderr)

    if args.max_examples and args.max_examples < len(corpus):
        corpus = corpus[:args.max_examples]

    # Chunk splitting for parallel gpu-test jobs
    if args.chunk_id is not None and args.num_chunks is not None:
        assert 0 <= args.chunk_id < args.num_chunks, (
            f"chunk_id={args.chunk_id} out of range [0, {args.num_chunks})")
        chunk_size = len(corpus) // args.num_chunks
        remainder = len(corpus) % args.num_chunks
        # Distribute remainder across first chunks
        start = args.chunk_id * chunk_size + min(args.chunk_id, remainder)
        end = start + chunk_size + (1 if args.chunk_id < remainder else 0)
        corpus = corpus[start:end]
        print(f"  Chunk {args.chunk_id}/{args.num_chunks}: queries [{start}, {end}) = {len(corpus)} entries",
              file=sys.stderr)

    print(f"Processing {len(corpus)} queries, {args.num_samples} rollout(s) each", file=sys.stderr)
    print(f"Config: max_turns={args.max_turns}, topk={args.topk}, "
          f"max_tokens={args.max_tokens}, model={args.gemini_model}", file=sys.stderr)
    print(f"Prompt version: {args.prompt_version}", file=sys.stderr)
    print(f"Rejection threshold: recall > {args.recall_threshold}", file=sys.stderr)

    # Check retriever
    print(f"Checking retriever at {args.search_url}...", file=sys.stderr)
    test_result = call_retriever(args.search_url, "test query", topk=1)
    if "error" in test_result.lower():
        print(f"WARNING: Retriever may not be ready: {test_result}", file=sys.stderr)
    else:
        print("Retriever ready!", file=sys.stderr)

    results: list[dict] = []
    total_kept = 0
    total_attempted = 0

    for idx, entry in enumerate(corpus):
        paper_id = entry["paper_id"]
        heading = entry.get("subsection_heading", "")
        rich_query = entry.get("rich_query", heading)
        citation_ids = entry["citation_ids"]
        num_cites = len(citation_ids)
        citation_budget = num_cites * 2

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[{idx+1}/{len(corpus)}] {heading}", file=sys.stderr)
        print(f"Paper: {paper_id}, Citations: {num_cites}, Budget: {citation_budget}", file=sys.stderr)

        user_prompt = format_user_prompt(
            entry["title"], entry["abstract"], entry["introduction"],
            rich_query, num_cites,
        )

        kept_this_query = 0
        for sample_idx in range(args.num_samples):
            total_attempted += 1

            result = run_gemini_rollout(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                targets=citation_ids,
                citation_budget=citation_budget,
                gemini_client=gemini_client,
                gemini_model=args.gemini_model,
                search_url=args.search_url,
                max_turns=args.max_turns,
                topk=args.topk,
                max_tokens=args.max_tokens,
                rate_limiter=rate_limiter,
            )

            recall = result["recall"]

            # Debug output for first few
            if sample_idx < 3 or recall > 0:
                full_text = "".join(m["content"] for m in result["messages"])
                predicted = extract_all_citations(full_text)
                gt_set = {normalize_arxiv_id(t) for t in citation_ids}
                print(f"  s{sample_idx+1}: recall={recall:.2f}, turns={result['num_turns']}  "
                      f"pred={predicted}  gt={gt_set}", file=sys.stderr)

            # Rejection sampling: keep if recall > threshold
            if recall > args.recall_threshold:
                total_kept += 1
                kept_this_query += 1
                results.append({
                    "paper_id": paper_id,
                    "subsection_heading": heading,
                    "rich_query": rich_query,
                    "citation_ids": citation_ids,
                    "recall": recall,
                    "num_turns": result["num_turns"],
                    "messages": result["messages"],
                })

        if (idx + 1) % 10 == 0:
            print(f"  [{rate_limiter.stats}]", file=sys.stderr)

        print(f"  Kept {kept_this_query}/{args.num_samples} rollouts", file=sys.stderr)

        # Incremental save every 5 queries (don't lose progress)
        if (idx + 1) % 5 == 0 and results:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [saved {len(results)} trajectories to {args.output}]", file=sys.stderr)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    n_queries = len(corpus)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Summary:", file=sys.stderr)
    print(f"  Queries: {n_queries}", file=sys.stderr)
    print(f"  Total rollouts: {total_attempted}", file=sys.stderr)
    print(f"  Kept trajectories: {total_kept} "
          f"({100*total_kept/max(total_attempted,1):.0f}% of rollouts)", file=sys.stderr)
    print(f"  Queries with >=1 success: "
          f"{sum(1 for e in set(r['paper_id'] for r in results))}", file=sys.stderr)
    print(f"  Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
