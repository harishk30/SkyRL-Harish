#!/usr/bin/env python3
"""Evaluate a base model on the citation prediction dataset without RL.

Runs multi-turn inference: model generates <search>query</search>,
retriever returns results, we check if the target arxiv ID appeared.

Supports pass@k evaluation: generate multiple stochastic samples per prompt
and compute the unbiased pass@k estimator for k=1..num_samples.

Use this to iterate on prompt templates and embedding models before training.

Requires:
  - A retriever server running (see start_retriever.sh)
  - A GPU for vLLM inference

Usage:
    # Quick greedy test on 20 validation examples
    python examples/citation_prediction/eval_base_model.py \
        --model Qwen/Qwen3-4B \
        --data /path/to/validation.parquet \
        --search_url http://localhost:8000/retrieve \
        --max_examples 20 --temperature 0

    # Pass@k=20 evaluation on 100 random training examples
    python examples/citation_prediction/eval_base_model.py \
        --model Qwen/Qwen3-4B \
        --data /path/to/train.parquet \
        --search_url http://localhost:8000/retrieve \
        --max_examples 100 --random_sample \
        --num_samples 20 --temperature 1.0 \
        --output results.json
"""

import argparse
import copy
import json
import re
import sys
import time
from math import comb
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


def load_examples(data_path: str, system_prompt_override: str = None,
                  max_examples: int = None, random_sample: bool = False,
                  seed: int = 42):
    """Load parquet and parse into example dicts."""
    df = pd.read_parquet(data_path)
    if max_examples and max_examples < len(df):
        if random_sample:
            df = df.sample(n=max_examples, random_state=seed)
        else:
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
        })
    return examples


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    n = total samples, c = number of correct samples, k = k value.
    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def run_single_sample(example, llm, tokenizer, sampling_params, args):
    """Run one multi-turn trajectory for a single example.

    Returns dict with {found, turns, queries, messages}.
    """
    messages = copy.deepcopy(example["messages"])
    target = example["target"]
    found = False
    turns = 0
    queries = []

    for turn in range(args.max_turns):
        if found:
            break

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        outputs = llm.generate([text], sampling_params)
        response = outputs[0].outputs[0].text
        turns += 1
        messages.append({"role": "assistant", "content": response})

        match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
        if match is None:
            continue

        query = match.group(1).strip()
        queries.append(query)

        tool_output = call_retriever(args.search_url, query, args.topk)
        observation = "\n<information>" + tool_output + "</information>\n"
        messages.append({"role": "user", "content": observation})

        retrieved_ids = extract_arxiv_ids(tool_output)
        if target in retrieved_ids:
            found = True

    return {
        "found": found,
        "turns": turns,
        "queries": queries,
        "messages": messages,
    }


def run_batched_single_sample(examples, llm, tokenizer, sampling_params, args):
    """Run one multi-turn trajectory for all examples using batched vLLM generation.

    Returns list of dicts with {found, turns, queries, messages}.
    """
    n = len(examples)
    # Initialize per-example state
    states = []
    for ex in examples:
        states.append({
            "messages": copy.deepcopy(ex["messages"]),
            "target": ex["target"],
            "found": False,
            "turns": 0,
            "queries": [],
        })

    active_indices = list(range(n))

    for turn in range(args.max_turns):
        if not active_indices:
            break

        n_active = len(active_indices)

        # 1. Format prompts for all active examples
        prompts = []
        for idx in active_indices:
            text = tokenizer.apply_chat_template(
                states[idx]["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)

        # 2. Batch generate with vLLM
        outputs = llm.generate(prompts, sampling_params)

        # 3. Process results
        still_active = []
        for i, idx in enumerate(active_indices):
            st = states[idx]
            response = outputs[i].outputs[0].text
            st["turns"] += 1
            st["messages"].append({"role": "assistant", "content": response})

            match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
            if match is None:
                still_active.append(idx)
                continue

            query = match.group(1).strip()
            st["queries"].append(query)

            tool_output = call_retriever(args.search_url, query, args.topk)
            observation = "\n<information>" + tool_output + "</information>\n"
            st["messages"].append({"role": "user", "content": observation})

            retrieved_ids = extract_arxiv_ids(tool_output)
            if st["target"] in retrieved_ids:
                st["found"] = True
            else:
                still_active.append(idx)

        active_indices = still_active

    return [
        {
            "found": st["found"],
            "turns": st["turns"],
            "queries": st["queries"],
            "messages": st["messages"],
        }
        for st in states
    ]


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
    parser.add_argument("--topk", type=int, default=3,
                        help="Number of search results per query (n)")
    parser.add_argument("--max_turns", type=int, default=4,
                        help="Max search turns per trajectory (m)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples (for quick testing)")
    parser.add_argument("--random_sample", action="store_true",
                        help="Randomly sample examples instead of taking head")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of stochastic samples per prompt for pass@k")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Max tokens per generation turn")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (0 = greedy, 1.0 for pass@k)")
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
    examples = load_examples(
        args.data, system_prompt_override, args.max_examples,
        args.random_sample, args.seed,
    )
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
    # Multi-sample evaluation
    # ----------------------------------------------------------------
    num_samples = args.num_samples
    total_examples = len(examples)
    start_time = time.time()

    # per_example_samples[i] = list of sample results for example i
    per_example_samples = [[] for _ in range(total_examples)]

    print(f"Running {num_samples} sample(s) per prompt, {total_examples} prompts")
    print(f"Temperature: {args.temperature}, Max turns: {args.max_turns}, Top-k: {args.topk}")

    for sample_idx in range(num_samples):
        sample_start = time.time()

        # Use different seed per sample for stochastic diversity
        sample_seed = args.seed + sample_idx
        sample_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["</search>"],
            include_stop_str_in_output=True,
            seed=sample_seed,
        )

        # Use batched generation for efficiency
        sample_results = run_batched_single_sample(
            examples, llm, tokenizer, sample_params, args,
        )

        for i, result in enumerate(sample_results):
            per_example_samples[i].append(result)

        sample_found = sum(1 for r in sample_results if r["found"])
        sample_time = time.time() - sample_start
        print(f"  Sample {sample_idx + 1}/{num_samples}: "
              f"{sample_found}/{total_examples} found ({sample_found/total_examples*100:.1f}%) "
              f"in {sample_time:.1f}s")

    elapsed = time.time() - start_time

    # ----------------------------------------------------------------
    # Compute pass@k
    # ----------------------------------------------------------------
    k_values = sorted(set(
        k for k in [1, 2, 3, 5, 10, 15, 20] if k <= num_samples
    ))

    # Per-example: count how many samples found the target
    per_example_correct = []
    for i in range(total_examples):
        c = sum(1 for s in per_example_samples[i] if s["found"])
        per_example_correct.append(c)

    # Compute pass@k for each k
    pass_at_k_results = {}
    for k in k_values:
        scores = [pass_at_k(num_samples, c, k) for c in per_example_correct]
        pass_at_k_results[k] = sum(scores) / len(scores)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Examples:           {total_examples}")
    print(f"Samples/prompt:     {num_samples}")
    print(f"Temperature:        {args.temperature}")
    print(f"Max turns (m):      {args.max_turns}")
    print(f"Top-k (n):          {args.topk}")
    print(f"Total time:         {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print(f"\nPass@k results:")
    for k, rate in sorted(pass_at_k_results.items()):
        print(f"  pass@{k:>2d}: {rate*100:.1f}%")

    # Per-turn breakdown (across all samples)
    all_sample_turns_found = []
    for i in range(total_examples):
        for s in per_example_samples[i]:
            if s["found"]:
                all_sample_turns_found.append(s["turns"])

    if all_sample_turns_found:
        avg_turns_found = sum(all_sample_turns_found) / len(all_sample_turns_found)
        print(f"\nAvg turns (when found): {avg_turns_found:.2f}")

        turn_counts = {}
        for t in all_sample_turns_found:
            turn_counts[t] = turn_counts.get(t, 0) + 1
        print(f"Per-turn breakdown (found, across all samples):")
        for t in sorted(turn_counts):
            print(f"  Turn {t}: {turn_counts[t]}")

    # ----------------------------------------------------------------
    # Save detailed results
    # ----------------------------------------------------------------
    if args.output:
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        for i in range(total_examples):
            samples_data = []
            for s in per_example_samples[i]:
                samples_data.append({
                    "found": s["found"],
                    "turns": s["turns"],
                    "queries": s["queries"],
                    "messages": s["messages"],
                })

            results.append({
                "index": i,
                "target": examples[i]["target"],
                "num_correct": per_example_correct[i],
                "samples": samples_data,
            })

        output_data = {
            "config": {
                "model": args.model,
                "data": args.data,
                "search_url": args.search_url,
                "topk": args.topk,
                "max_turns": args.max_turns,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "num_samples": num_samples,
                "seed": args.seed,
                "random_sample": args.random_sample,
                "system_prompt_override": system_prompt_override,
            },
            "summary": {
                "total": total_examples,
                "pass_at_k": {str(k): v for k, v in pass_at_k_results.items()},
                "avg_turns_found": (
                    sum(all_sample_turns_found) / len(all_sample_turns_found)
                    if all_sample_turns_found else 0
                ),
                "elapsed_seconds": elapsed,
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
