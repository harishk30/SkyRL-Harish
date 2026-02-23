#!/usr/bin/env python3
"""Evaluate a base model on the citation prediction v2 dataset without RL.

Runs multi-turn inference: model generates <search>query</search>,
retriever returns results, model accumulates <citation> tags throughout.

Metrics: recall, precision, num_correct per subsection.

Supports multi-sample evaluation: generate multiple stochastic samples per
prompt and report mean/max recall across samples.

Requires:
  - A retriever server running (see start_retriever.sh)
  - A GPU for vLLM inference

Usage:
    # Quick greedy test on 20 examples
    python eval_base_model.py \
        --model Qwen/Qwen3-4B \
        --data /path/to/validation.parquet \
        --search_url http://localhost:8000/retrieve \
        --max_examples 20 --temperature 0

    # Multi-sample evaluation on 100 random training examples
    python eval_base_model.py \
        --model Qwen/Qwen3-4B \
        --data /path/to/train.parquet \
        --search_url http://localhost:8000/retrieve \
        --max_examples 100 --random_sample \
        --num_samples 10 --temperature 1.0 \
        --output results.json
"""

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Arxiv ID extraction (from retriever results and <citation> tags)
# ---------------------------------------------------------------------------

ARXIV_ID_RE = re.compile(r"\[arxiv:(\d{4}\.\d{4,5})\]")
CITATION_TAG_RE = re.compile(r"<citation>(.*?)</citation>", re.DOTALL)


def normalize_arxiv_id(s: str) -> str:
    """Normalize an arxiv ID to YYMM.NNNNN format."""
    s = s.strip()
    match = re.search(r"(\d{4}\.\d{4,5})", s)
    if match:
        return match.group(1)
    s = s.strip("[]")
    s = re.sub(r"^[A-Za-z-]+[:/]\s*", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s.strip()


def extract_all_citations(text: str) -> set[str]:
    """Extract all arxiv IDs from <citation> tags in the trajectory."""
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


def compute_recall_metrics(
    messages: list[dict],
    targets: list[str],
    max_predictions_ratio: float = 2.0,
) -> dict:
    """Compute recall/precision metrics from a completed trajectory."""
    full_text = "".join(m["content"] for m in messages)
    predicted = extract_all_citations(full_text)
    gt_set = {normalize_arxiv_id(t) for t in targets}
    gt_set.discard("")

    correct = predicted & gt_set
    num_gt = len(gt_set)
    num_pred = len(predicted)
    num_correct = len(correct)

    # Spam penalty
    if num_pred > max_predictions_ratio * num_gt:
        recall = 0.0
    else:
        recall = num_correct / num_gt if num_gt else 0.0

    precision = num_correct / num_pred if num_pred else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "num_predicted": num_pred,
        "num_correct": num_correct,
        "num_ground_truth": num_gt,
        "predicted_ids": sorted(predicted),
        "correct_ids": sorted(correct),
    }


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def call_retriever(search_url: str, query: str, topk: int = 3, timeout: int = 30) -> str:
    """Call the retriever API. Returns JSON string matching env format."""
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
# Data loading
# ---------------------------------------------------------------------------


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
        targets = reward_spec["ground_truth"]["targets"]  # list of arxiv IDs

        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        examples.append({
            "messages": list(prompt),
            "targets": targets,
            "subsection_heading": metadata.get("subsection_heading", ""),
            "num_ground_truth": len(targets),
        })
    return examples


# ---------------------------------------------------------------------------
# Batched multi-turn inference
# ---------------------------------------------------------------------------


def run_batched_single_sample(examples, llm, tokenizer, sampling_params, args):
    """Run one multi-turn trajectory for all examples using batched vLLM generation.

    Each trajectory runs until max_turns or the model outputs <done>.
    Returns list of dicts with {recall, precision, num_predicted, num_correct,
    num_ground_truth, turns, queries, messages}.
    """
    n = len(examples)
    states = []
    for ex in examples:
        states.append({
            "messages": copy.deepcopy(ex["messages"]),
            "targets": ex["targets"],
            "done": False,
            "turns": 0,
            "queries": [],
        })

    active_indices = list(range(n))

    for turn in range(args.max_turns):
        if not active_indices:
            break

        # 1. Format prompts for active examples
        prompts = []
        for idx in active_indices:
            text = tokenizer.apply_chat_template(
                states[idx]["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)

        # 2. Batch generate
        outputs = llm.generate(prompts, sampling_params)

        # 3. Process results
        still_active = []
        for i, idx in enumerate(active_indices):
            st = states[idx]
            response = outputs[i].outputs[0].text
            st["turns"] += 1
            st["messages"].append({"role": "assistant", "content": response})

            # Check if model signaled done
            if "<done>" in response:
                st["done"] = True
                continue

            # Check for search query
            match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
            if match is None:
                # No search and no done — model generated thinking/citations only
                still_active.append(idx)
                continue

            query = match.group(1).strip()
            st["queries"].append(query)

            # Call retriever
            tool_output = call_retriever(args.search_url, query, args.topk)
            observation = "\n<information>" + tool_output + "</information>\n"

            # Add turn counter
            remaining = args.max_turns - st["turns"]
            if remaining > 1:
                observation += f"\n\n{remaining} turns remaining."
            elif remaining == 1:
                observation += "\n\nThis is your last turn. Cite remaining papers and write <done></done>."

            st["messages"].append({"role": "user", "content": observation})
            still_active.append(idx)

        active_indices = still_active

    # Compute metrics for all examples
    results = []
    for st in states:
        metrics = compute_recall_metrics(st["messages"], st["targets"])
        results.append({
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "num_predicted": metrics["num_predicted"],
            "num_correct": metrics["num_correct"],
            "num_ground_truth": metrics["num_ground_truth"],
            "predicted_ids": metrics["predicted_ids"],
            "correct_ids": metrics["correct_ids"],
            "done": st["done"],
            "turns": st["turns"],
            "queries": st["queries"],
            "messages": st["messages"],
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base model on citation prediction v2 (recall)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HuggingFace name")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to parquet file (e.g. validation.parquet)")
    parser.add_argument("--search_url", type=str,
                        default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3,
                        help="Number of search results per query (n)")
    parser.add_argument("--max_turns", type=int, default=15,
                        help="Max search turns per trajectory (m)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples")
    parser.add_argument("--random_sample", action="store_true",
                        help="Randomly sample examples instead of taking head")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of stochastic samples per prompt")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens per generation turn")
    parser.add_argument("--temperature", type=float, default=1.0,
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

    # Print ground truth distribution
    gt_counts = [ex["num_ground_truth"] for ex in examples]
    print(f"Ground truth citations per example: min={min(gt_counts)}, "
          f"max={max(gt_counts)}, mean={sum(gt_counts)/len(gt_counts):.1f}")

    # ----------------------------------------------------------------
    # Verify retriever
    # ----------------------------------------------------------------
    print(f"\nChecking retriever at {args.search_url}...")
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

        sample_seed = args.seed + sample_idx
        sample_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["</search>", "</done>"],
            include_stop_str_in_output=True,
            seed=sample_seed,
        )

        sample_results = run_batched_single_sample(
            examples, llm, tokenizer, sample_params, args,
        )

        for i, result in enumerate(sample_results):
            per_example_samples[i].append(result)

        # Per-sample summary
        recalls = [r["recall"] for r in sample_results]
        mean_recall = sum(recalls) / len(recalls)
        nonzero_recall = sum(1 for r in recalls if r > 0)
        sample_time = time.time() - sample_start
        print(f"  Sample {sample_idx + 1}/{num_samples}: "
              f"mean_recall={mean_recall*100:.1f}%, "
              f"nonzero={nonzero_recall}/{total_examples} "
              f"({nonzero_recall/total_examples*100:.1f}%) "
              f"in {sample_time:.1f}s")

    elapsed = time.time() - start_time

    # ----------------------------------------------------------------
    # Aggregate metrics
    # ----------------------------------------------------------------

    # Per-example: best recall across samples, mean recall
    per_example_best_recall = []
    per_example_mean_recall = []
    per_example_best_precision = []
    for i in range(total_examples):
        recalls = [s["recall"] for s in per_example_samples[i]]
        precisions = [s["precision"] for s in per_example_samples[i]]
        per_example_best_recall.append(max(recalls))
        per_example_mean_recall.append(sum(recalls) / len(recalls))
        per_example_best_precision.append(max(precisions))

    mean_best_recall = sum(per_example_best_recall) / total_examples
    mean_mean_recall = sum(per_example_mean_recall) / total_examples
    mean_best_precision = sum(per_example_best_precision) / total_examples

    # Fraction of examples where at least one sample got nonzero recall
    any_correct = sum(1 for r in per_example_best_recall if r > 0)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Examples:                  {total_examples}")
    print(f"Samples/prompt:            {num_samples}")
    print(f"Temperature:               {args.temperature}")
    print(f"Max turns (m):             {args.max_turns}")
    print(f"Top-k (n):                 {args.topk}")
    print(f"Total time:                {elapsed:.1f}s")
    print(f"{'=' * 60}")

    print(f"\nRecall metrics:")
    print(f"  Mean recall (all samples):     {mean_mean_recall*100:.1f}%")
    print(f"  Mean best recall (per example):{mean_best_recall*100:.1f}%")
    print(f"  Any correct (>0 recall):       {any_correct}/{total_examples} "
          f"({any_correct/total_examples*100:.1f}%)")
    print(f"  Mean best precision:           {mean_best_precision*100:.1f}%")

    # Recall distribution (best per example)
    print(f"\nRecall distribution (best per example):")
    bins = [(0, 0), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.01)]
    labels = ["0%", "1-25%", "25-50%", "50-75%", "75-99%", "100%"]
    for (lo, hi), label in zip(bins, labels):
        count = sum(1 for r in per_example_best_recall if lo <= r < hi)
        if label == "100%":
            count = sum(1 for r in per_example_best_recall if r >= 1.0)
        print(f"  {label:>8s}: {count:>4d} ({count/total_examples*100:.1f}%)")

    # Turn usage
    all_turns = [s["turns"] for samples in per_example_samples for s in samples]
    avg_turns = sum(all_turns) / len(all_turns) if all_turns else 0
    print(f"\nAvg turns: {avg_turns:.1f}")

    # Verbose per-example output
    if args.verbose:
        print(f"\nPer-example results (best sample):")
        for i in range(total_examples):
            best_idx = max(range(num_samples),
                           key=lambda j: per_example_samples[i][j]["recall"])
            best = per_example_samples[i][best_idx]
            print(f"  [{i:>3d}] heading='{examples[i]['subsection_heading'][:50]}' "
                  f"gt={best['num_ground_truth']} "
                  f"pred={best['num_predicted']} "
                  f"correct={best['num_correct']} "
                  f"recall={best['recall']*100:.0f}%")

    # ----------------------------------------------------------------
    # Save detailed results
    # ----------------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        for i in range(total_examples):
            samples_data = []
            for s in per_example_samples[i]:
                samples_data.append({
                    "recall": s["recall"],
                    "precision": s["precision"],
                    "num_predicted": s["num_predicted"],
                    "num_correct": s["num_correct"],
                    "num_ground_truth": s["num_ground_truth"],
                    "predicted_ids": s["predicted_ids"],
                    "correct_ids": s["correct_ids"],
                    "done": s["done"],
                    "turns": s["turns"],
                    "queries": s["queries"],
                    "messages": s["messages"],
                })

            results.append({
                "index": i,
                "targets": examples[i]["targets"],
                "subsection_heading": examples[i]["subsection_heading"],
                "num_ground_truth": examples[i]["num_ground_truth"],
                "best_recall": per_example_best_recall[i],
                "mean_recall": per_example_mean_recall[i],
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
                "mean_recall_all": mean_mean_recall,
                "mean_best_recall": mean_best_recall,
                "mean_best_precision": mean_best_precision,
                "any_correct_fraction": any_correct / total_examples,
                "avg_turns": avg_turns,
                "elapsed_seconds": elapsed,
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
