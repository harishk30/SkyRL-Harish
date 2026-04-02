#!/usr/bin/env python3
"""Generate SFT training trajectories using Codex with retriever-in-the-loop.

Runs OpenAI Codex (terminal agent) through the same search pipeline as vLLM
evaluation. Codex uses an MCP search tool to query the arxiv retriever. The
JSON event stream is captured and converted to SkyRL training format.

Prerequisites:
    1. Codex CLI installed:  npm install -g @openai/codex
    2. CODEX_API_KEY or OPENAI_API_KEY set
    3. MCP search server configured in ~/.codex/config.toml
    4. Retriever server running at the configured URL

Usage:
    python generate_sft_trajectories_codex.py \
        --corpus subsection_corpus_v3.json \
        --output sft_trajectories.json \
        --max_examples 50 \
        --num_samples 5 \
        --max_turns 4 \
        --codex_model o3

Output format is identical to generate_sft_trajectories.py (Gemini version):
    [{"paper_id", "subsection_heading", "citation_ids", "recall",
      "num_turns", "messages": [{"role","content"}, ...]}, ...]
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Citation extraction (reused from v2 generate_sft_trajectories.py)
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
    """Extract arxiv IDs from <citation> tags, ignoring those inside <think>."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    ids: set[str] = set()
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


# ---------------------------------------------------------------------------
# Codex JSON event parsing
# ---------------------------------------------------------------------------

def parse_codex_events(stdout: str) -> list[dict]:
    """Parse newline-delimited JSON from codex exec --json."""
    events: list[dict] = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def events_to_messages(
    events: list[dict],
    system_prompt: str,
    user_prompt: str,
    max_turns: int,
) -> list[dict]:
    """Convert Codex JSON events to SkyRL-compatible message trajectory.

    Mapping:
        reasoning       → <think>text</think> in assistant turn
        agentMessage    → assistant turn text (may contain <citation>, <done>)
        mcpToolCall     → <search>query</search> (assistant) +
                          <information>result</information> (user)
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    assistant_parts: list[str] = []
    search_count = 0

    for event in events:
        if event.get("type") != "item.completed":
            continue

        item = event.get("item", {})
        item_type = item.get("type", "")

        # --- Internal reasoning (CoT) ---
        if item_type == "reasoning":
            text = item.get("summary") or item.get("content") or ""
            if isinstance(text, dict):
                text = text.get("text", str(text))
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            text = str(text).strip()
            if text:
                assistant_parts.append(f"<think>{text}</think>")

        # --- Agent spoken text ---
        elif item_type == "agentMessage":
            text = item.get("text", "").strip()
            if text:
                assistant_parts.append(text)

        # --- MCP tool call (search) ---
        elif item_type == "mcpToolCall":
            tool_name = item.get("tool", "")
            if tool_name != "search":
                continue

            # Extract query from arguments
            args = item.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"query": args}
            query = args.get("query", str(args))

            # Add search tag to assistant buffer
            assistant_parts.append(f"<search>{query}</search>")

            # Flush assistant buffer as one turn
            if assistant_parts:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(assistant_parts),
                })
                assistant_parts = []

            # Add search result as user observation
            result = item.get("result", "No results returned.")
            if isinstance(result, dict):
                result = json.dumps(result)
            search_count += 1
            remaining = max_turns - search_count
            observation = f"\n<information>{result}</information>\n"
            if remaining > 1:
                observation += f"\n{remaining} searches remaining."
            elif remaining == 1:
                observation += (
                    "\nThis is your last search. "
                    "Cite remaining papers and write <done></done>."
                )
            elif remaining <= 0:
                observation += (
                    "\nNo searches remaining. "
                    "Cite your findings and write <done></done>."
                )
            messages.append({"role": "user", "content": observation})

    # Flush remaining assistant buffer
    if assistant_parts:
        messages.append({
            "role": "assistant",
            "content": "\n".join(assistant_parts),
        })

    return messages


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research assistant. Given a paper's title, abstract, introduction, \
and a Related Work subsection heading, your job is to identify ALL papers \
cited in that subsection by their arxiv IDs.

You have ONE tool available: `search`, which queries an arxiv paper database \
by semantic keywords. Each result shows [arxiv:ID] followed by title, \
authors, and abstract. The search engine matches by meaning, so use \
descriptive topic keywords — do NOT search for author names or arxiv IDs.

WORKFLOW:
1. Reason about what papers the subsection likely cites.
2. Use the `search` tool with descriptive keywords.
3. When you find a relevant paper in the results, cite it IMMEDIATELY with \
<citation>arxiv_id</citation>. You can cite multiple IDs: \
<citation>XXXX.XXXXX, YYYY.YYYYY</citation>.
4. Search from multiple angles — rephrase queries, try method names, \
dataset names, task descriptions. Use all your turns.
5. When you have found all cited papers, write <done></done>.

CRITICAL RULES:
- ONLY cite arxiv IDs that appear as [arxiv:ID] in search results. \
Never guess or fabricate IDs.
- Cite IMMEDIATELY after each search when you find relevant papers. \
Do not accumulate papers to cite later.
- You have a citation budget (shown below). Exceeding it gives zero reward.
- Keep reasoning concise to maximize the number of searches.
- Do NOT write code, create files, or run shell commands. ONLY use search.
- The introduction mentions author names (e.g. "Smith et al.") — these are \
not searchable, but the topics and methods they describe ARE. Use the \
introduction to identify what topics to search for.\
"""


def format_user_prompt(
    title: str,
    abstract: str,
    introduction: str,
    heading: str,
    num_citations: int,
) -> str:
    budget = num_citations * 2
    return "\n".join([
        f"Paper title: {title}",
        f"\nAbstract:\n{abstract}",
        f"\nIntroduction:\n{introduction}",
        f'\nRelated Work subtopic: "{heading}"',
        f"\nCitation budget: at most {budget} citations (you may cite fewer — "
        "only cite papers you are confident about). "
        "Exceeding this budget results in zero reward.",
        "\nIdentify all papers that would be cited in this Related Work "
        "subtopic. Search for them and report their arxiv IDs using "
        "<citation> tags.",
    ])


# ---------------------------------------------------------------------------
# Codex invocation
# ---------------------------------------------------------------------------

async def run_codex_rollout(
    prompt: str,
    targets: list[str],
    max_turns: int,
    codex_model: str,
    timeout: int = 300,
    debug_dir: str | None = None,
    debug_id: str = "",
) -> dict:
    """Run a single Codex exec rollout. Returns trajectory + recall."""

    cmd = [
        "codex", "exec",
        "--json",
        "--full-auto",
        "--sandbox", "read-only",
        "--skip-git-repo-check",
    ]
    if codex_model:
        cmd.extend(["--model", codex_model])
    cmd.append("-")  # read prompt from stdin

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        return {"messages": [], "recall": 0.0, "num_turns": 0, "error": "timeout"}
    except FileNotFoundError:
        return {
            "messages": [], "recall": 0.0, "num_turns": 0,
            "error": "codex CLI not found — install with: npm install -g @openai/codex",
        }
    except Exception as e:
        return {"messages": [], "recall": 0.0, "num_turns": 0, "error": str(e)}

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    # Save raw output for debugging
    if debug_dir and debug_id:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        (debug_path / f"{debug_id}_stdout.jsonl").write_text(stdout)
        (debug_path / f"{debug_id}_stderr.txt").write_text(stderr)

    if proc.returncode != 0:
        return {
            "messages": [], "recall": 0.0, "num_turns": 0,
            "error": f"exit {proc.returncode}: {stderr[:500]}",
        }

    # Parse events
    events = parse_codex_events(stdout)

    # Build user prompt (we need it for the message list, extract from the
    # combined prompt we sent — it's after the system instructions)
    # Since we combine system+user into one Codex prompt, we store them
    # separately in the trajectory
    messages = events_to_messages(events, SYSTEM_PROMPT, "", max_turns)

    # Count search turns
    num_turns = sum(
        1 for e in events
        if e.get("type") == "item.completed"
        and e.get("item", {}).get("type") == "mcpToolCall"
        and e.get("item", {}).get("tool") == "search"
    )

    recall = compute_recall(messages, targets)

    return {
        "messages": messages,
        "recall": recall,
        "num_turns": num_turns,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def process_entry(
    idx: int,
    total: int,
    entry: dict,
    num_samples: int,
    max_turns: int,
    codex_model: str,
    timeout: int,
    recall_threshold: float,
    semaphore: asyncio.Semaphore,
    debug_dir: str | None,
) -> dict | None:
    """Process a single corpus entry with multiple rollouts."""

    paper_id = entry["paper_id"]
    heading = entry.get("rich_query", entry.get("subsection_heading", ""))
    display_heading = entry.get("subsection_heading", heading)
    citation_ids = entry["citation_ids"]
    num_cites = len(citation_ids)

    print(f"\n{'='*60}", file=sys.stderr)
    print(
        f"[{idx+1}/{total}] {display_heading} "
        f"(paper={paper_id}, cites={num_cites})",
        file=sys.stderr,
    )

    user_prompt = format_user_prompt(
        entry["title"],
        entry["abstract"],
        entry["introduction"],
        heading,
        num_cites,
    )

    # Codex gets system + user as a single prompt (it has its own system prompt)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{'='*40}\n\n{user_prompt}"

    best_result = None
    best_recall = -1.0

    for sample_idx in range(num_samples):
        async with semaphore:
            debug_id = f"{paper_id}_{idx}_s{sample_idx}" if debug_dir else ""
            result = await run_codex_rollout(
                prompt=full_prompt,
                targets=citation_ids,
                max_turns=max_turns,
                codex_model=codex_model,
                timeout=timeout,
                debug_dir=debug_dir,
                debug_id=debug_id,
            )

        if result.get("error"):
            print(
                f"  rollout {sample_idx+1}/{num_samples}: "
                f"ERROR — {result['error']}",
                file=sys.stderr,
            )
            continue

        # Show predictions vs targets
        full_text = "".join(m["content"] for m in result["messages"])
        predicted = extract_all_citations(full_text)
        gt_set = {normalize_arxiv_id(t) for t in citation_ids}
        print(
            f"  rollout {sample_idx+1}/{num_samples}: "
            f"recall={result['recall']:.2f}, turns={result['num_turns']}  "
            f"predicted={predicted}  targets={gt_set}",
            file=sys.stderr,
        )

        if result["recall"] > best_recall:
            best_recall = result["recall"]
            best_result = result

    if best_result and best_recall >= recall_threshold:
        # Patch the user prompt into the messages (slot 1)
        if len(best_result["messages"]) >= 2:
            best_result["messages"][1]["content"] = user_prompt

        print(f"  KEPT (recall={best_recall:.2f})", file=sys.stderr)
        return {
            "paper_id": paper_id,
            "subsection_heading": display_heading,
            "citation_ids": citation_ids,
            "recall": best_recall,
            "num_turns": best_result["num_turns"],
            "messages": best_result["messages"],
        }
    else:
        print(
            f"  DISCARDED (best recall={best_recall:.2f} "
            f"< {recall_threshold})",
            file=sys.stderr,
        )
        return None


async def async_main(args: argparse.Namespace) -> None:
    # Load corpus
    print(f"Loading corpus from {args.corpus}...", file=sys.stderr)
    with open(args.corpus) as f:
        corpus = json.load(f)

    if args.split:
        corpus = [e for e in corpus if e.get("split") == args.split]

    if args.prompt_ids_file:
        with open(args.prompt_ids_file) as f:
            prompt_ids = json.load(f)
        prompt_set = {(p["paper_id"], p["subsection_heading"]) for p in prompt_ids}
        corpus = [
            e for e in corpus
            if (e["paper_id"], e["subsection_heading"]) in prompt_set
        ]
        print(
            f"Filtered to {len(corpus)} entries via prompt_ids_file",
            file=sys.stderr,
        )

    if args.max_examples and args.max_examples < len(corpus):
        corpus = corpus[: args.max_examples]

    print(
        f"Processing {len(corpus)} entries, "
        f"{args.num_samples} rollout(s) each",
        file=sys.stderr,
    )
    print(
        f"Config: model={args.codex_model}, max_turns={args.max_turns}, "
        f"concurrency={args.concurrency}, timeout={args.timeout}s",
        file=sys.stderr,
    )
    print(f"Recall threshold: {args.recall_threshold}", file=sys.stderr)

    semaphore = asyncio.Semaphore(args.concurrency)

    results: list[dict] = []
    for idx, entry in enumerate(corpus):
        out = await process_entry(
            idx=idx,
            total=len(corpus),
            entry=entry,
            num_samples=args.num_samples,
            max_turns=args.max_turns,
            codex_model=args.codex_model,
            timeout=args.timeout,
            recall_threshold=args.recall_threshold,
            semaphore=semaphore,
            debug_dir=args.debug_dir,
        )
        if out is not None:
            results.append(out)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total = len(corpus)
    kept = len(results)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Summary:", file=sys.stderr)
    print(f"  Kept: {kept}/{total} ({100*kept/max(total,1):.0f}%)", file=sys.stderr)
    print(f"  Output: {args.output}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SFT trajectories with Codex",
    )
    parser.add_argument("--corpus", required=True, help="Subsection corpus JSON")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Rollouts per entry (keep best)")
    parser.add_argument("--max_turns", type=int, default=4,
                        help="Max search turns (also set MAX_SEARCHES in MCP env)")
    parser.add_argument("--recall_threshold", type=float, default=0.3)
    parser.add_argument("--codex_model", type=str, default="o3",
                        help="Model for Codex (o3, gpt-4o, etc.)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Max concurrent Codex invocations")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per rollout in seconds")
    parser.add_argument("--prompt_ids_file", type=str, default=None,
                        help="JSON list of {paper_id, subsection_heading} to filter")
    parser.add_argument("--debug_dir", type=str, default=None,
                        help="Directory to save raw Codex events for debugging")
    args = parser.parse_args()

    # Validate API key
    if not os.environ.get("CODEX_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: Set CODEX_API_KEY or OPENAI_API_KEY environment variable",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check codex CLI
    import shutil
    if not shutil.which("codex"):
        print(
            "ERROR: codex CLI not found. Install with: npm install -g @openai/codex",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
