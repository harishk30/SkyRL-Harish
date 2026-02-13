#!/usr/bin/env python3
"""Evaluate a base model on the citation prediction dataset without RL.

Runs multi-turn inference: model generates <search>query</search>,
retriever returns results, we check if the target arxiv ID appeared.

Use this to iterate on prompt templates and embedding models before training.

Requires:
  - A retriever server running (see start_retriever.sh)
  - A GPU for vLLM inference

Usage:
    # Quick test on 20 validation examples
    python examples/citation_prediction/eval_base_model.py \
        --model Qwen/Qwen3-4B \
        --data /path/to/validation.parquet \
        --search_url http://localhost:8000/retrieve \
        --max_examples 20

    # Override system prompt
    python examples/citation_prediction/eval_base_model.py \
        --model Qwen/Qwen3-4B \
        --data /path/to/validation.parquet \
        --system_prompt "You are a research assistant. ..."

    # Save full trajectories for analysis
    python examples/citation_prediction/eval_base_model.py \
        --model Qwen/Qwen3-4B \
        --data /path/to/validation.parquet \
        --output results.json

On Della (interactive GPU session):
    salloc --partition=gpu-test --gres=gpu:1 --time=1:00:00 --mem=64G
    # (in another terminal/job, start the retriever)
    cd /home/hk4638/SkyRL/skyrl-train
    export HF_HOME=/scratch/gpfs/ZHUANGL/hk4638/huggingface
    uv run --active --frozen python examples/citation_prediction/eval_base_model.py \
        --model /scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
        --data /scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction/validation.parquet \
        --search_url http://<retriever-node>:8000/retrieve \
        --max_examples 50
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ARXIV_ID_RE = re.compile(r"\[arxiv:(\d{4}\.\d{4,5})\]")


def extract_arxiv_ids(text: str) -> set:
    """Extract arxiv IDs like [arxiv:1706.03762] from text."""
    return set(ARXIV_ID_RE.findall(text))


def call_retriever(search_url: str, query: str, topk: int = 3, timeout: int = 30) -> str:
    """Call the retriever API. Returns the same JSON string format as SearchToolGroup.search().

    Output format: '{"result": "Doc 1: [arxiv:1706.03762] ...\\nDoc 2: ..."}'
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
        if raw_results:
            # Match _passages2string format from skyrl_gym/tools/search.py
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


def load_examples(data_path: str, system_prompt_override: str = None, max_examples: int = None):
    """Load parquet and parse into example dicts."""
    df = pd.read_parquet(data_path)
    if max_examples:
        df = df.head(max_examples)

    examples = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        if system_prompt_override and prompt and prompt[0]["role"] == "system":
            prompt[0]["content"] = system_prompt_override

        reward_spec = row["reward_spec"]
        if isinstance(reward_spec, str):
            reward_spec = json.loads(reward_spec)
        target = reward_spec["ground_truth"]["target"]

        examples.append({
            "messages": list(prompt),
            "target": target,
            "found": False,
            "turns": 0,
            "queries": [],
        })
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base model on citation prediction dataset"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HuggingFace name")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to parquet file (e.g. validation.parquet)")
    parser.add_argument("--search_url", type=str,
                        default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples (for quick testing)")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Max tokens per generation turn")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Override system prompt (direct text)")
    parser.add_argument("--system_prompt_file", type=str, default=None,
                        help="Override system prompt (read from file)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output", type=str, default=None,
                        help="Save detailed results as JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example results")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Resolve system prompt override
    # ----------------------------------------------------------------
    system_prompt_override = None
    if args.system_prompt_file:
        system_prompt_override = Path(args.system_prompt_file).read_text().strip()
    elif args.system_prompt:
        system_prompt_override = args.system_prompt

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model with vLLM: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["</search>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    # ----------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------
    print(f"Loading data: {args.data}")
    examples = load_examples(args.data, system_prompt_override, args.max_examples)
    print(f"Loaded {len(examples)} examples")

    print(f"\nSystem prompt:\n  {examples[0]['messages'][0]['content'][:200]}...\n")

    # ----------------------------------------------------------------
    # Verify retriever
    # ----------------------------------------------------------------
    print(f"Checking retriever at {args.search_url}...")
    try:
        resp = requests.post(
            args.search_url,
            json={"query": "test", "topk": 1},
            timeout=5,
        )
        resp.raise_for_status()
        print("Retriever is ready!\n")
    except Exception as e:
        print(f"ERROR: Cannot reach retriever at {args.search_url}: {e}")
        print("Start the retriever first (see start_retriever.sh)")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Multi-turn evaluation
    # ----------------------------------------------------------------
    active_indices = list(range(len(examples)))
    start_time = time.time()

    for turn in range(args.max_turns):
        if not active_indices:
            break

        n_active = len(active_indices)
        print(f"--- Turn {turn + 1}/{args.max_turns} ({n_active} active) ---")

        # 1. Format prompts for all active examples
        prompts = []
        for idx in active_indices:
            text = tokenizer.apply_chat_template(
                examples[idx]["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)

        # 2. Batch generate with vLLM
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        gen_time = time.time() - t0
        print(f"  Generation: {n_active} responses in {gen_time:.1f}s")

        # 3. Process results: parse queries, call retriever, check IDs
        t0 = time.time()
        still_active = []
        found_this_turn = 0
        no_search_count = 0

        for i, idx in enumerate(active_indices):
            ex = examples[idx]
            response = outputs[i].outputs[0].text
            ex["turns"] += 1
            ex["messages"].append({"role": "assistant", "content": response})

            # Parse <search>query</search>
            match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
            if match is None:
                no_search_count += 1
                # Model didn't search — give it another turn
                still_active.append(idx)
                continue

            query = match.group(1).strip()
            ex["queries"].append(query)

            # Call retriever
            tool_output = call_retriever(args.search_url, query, args.topk)
            observation = "\n<information>" + tool_output + "</information>\n"
            ex["messages"].append({"role": "user", "content": observation})

            # Check for target arxiv ID
            retrieved_ids = extract_arxiv_ids(tool_output)
            if ex["target"] in retrieved_ids:
                ex["found"] = True
                found_this_turn += 1
                if args.verbose:
                    print(f"    [FOUND] idx={idx} target={ex['target']} query='{query}'")
            else:
                still_active.append(idx)
                if args.verbose:
                    print(f"    [MISS]  idx={idx} target={ex['target']} query='{query}' got={retrieved_ids or '{}'}")

        retriever_time = time.time() - t0
        total_found = sum(1 for ex in examples if ex["found"])
        active_indices = still_active
        print(f"  Retriever: {n_active - no_search_count} calls in {retriever_time:.1f}s")
        print(f"  Found this turn: {found_this_turn} | Total: {total_found}/{len(examples)} "
              f"({total_found/len(examples)*100:.1f}%) | No search tag: {no_search_count}")

    elapsed = time.time() - start_time

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    total = len(examples)
    found = sum(1 for ex in examples if ex["found"])
    found_turns = [ex["turns"] for ex in examples if ex["found"]]
    avg_turns_found = sum(found_turns) / len(found_turns) if found_turns else 0
    avg_turns_all = sum(ex["turns"] for ex in examples) / total

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Examples:           {total}")
    print(f"Papers found:       {found}/{total} ({found / total * 100:.1f}%)")
    print(f"Avg turns (found):  {avg_turns_found:.2f}")
    print(f"Avg turns (all):    {avg_turns_all:.2f}")
    print(f"Total time:         {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"Model:              {args.model}")
    print(f"Retriever:          {args.search_url}")
    print(f"Top-k:              {args.topk}")
    print(f"Max turns:          {args.max_turns}")
    print(f"Temperature:        {args.temperature}")

    # Per-turn breakdown
    turn_counts = [0] * args.max_turns
    for ex in examples:
        if ex["found"]:
            turn_counts[ex["turns"] - 1] += 1
    print(f"\nPer-turn breakdown (found):")
    for t, count in enumerate(turn_counts):
        if count > 0:
            print(f"  Turn {t + 1}: {count} ({count / total * 100:.1f}%)")
    not_found = total - found
    if not_found > 0:
        print(f"  Not found: {not_found} ({not_found / total * 100:.1f}%)")

    # ----------------------------------------------------------------
    # Save detailed results
    # ----------------------------------------------------------------
    if args.output:
        results = []
        for i, ex in enumerate(examples):
            entry = {
                "index": i,
                "target": ex["target"],
                "found": ex["found"],
                "turns": ex["turns"],
                "queries": ex["queries"],
            }
            # Include full trajectory for inspection
            entry["messages"] = ex["messages"]
            results.append(entry)

        output_data = {
            "config": {
                "model": args.model,
                "data": args.data,
                "search_url": args.search_url,
                "topk": args.topk,
                "max_turns": args.max_turns,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "system_prompt_override": system_prompt_override,
            },
            "summary": {
                "total": total,
                "found": found,
                "found_rate": found / total,
                "avg_turns_found": avg_turns_found,
                "avg_turns_all": avg_turns_all,
                "elapsed_seconds": elapsed,
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
