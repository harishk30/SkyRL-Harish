"""Standalone eval script for citation prediction using vLLM offline LLM class.

No SkyRL or skyrl-gym dependencies — only vLLM + basic Python packages.
Replicates the multi-turn eval loop from CitationPredictionV2Env.

Key optimization: all n_samples are batched per turn, so vLLM processes
them in parallel instead of sequentially.

Usage:
    python standalone_eval.py \
        --model_path /path/to/model \
        --data_path /path/to/train_sample50_seed42_part0.parquet \
        --retriever_url http://<node>:8000/retrieve \
        --max_turns 6 --topk 5 --n_samples 20 \
        --temperature 1.0 --max_gen_length 4096 \
        --max_model_len 65536 --tp 2 \
        --output_dir /path/to/results
"""

import argparse
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Citation extraction & reward (from skyrl_gym/envs/citation_prediction_v2/utils.py)
# ============================================================================

def normalize_arxiv_id(s: str) -> str:
    s = s.strip()
    match = re.search(r"(\d{4}\.\d{4,5})", s)
    if match:
        return match.group(1)
    s = s.strip("[]")
    s = re.sub(r"^[A-Za-z-]+[:/]\s*", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s.strip()


def extract_all_citations(text: str) -> set:
    ids = set()
    for match in re.finditer(r"<citation>(.*?)</citation>", text, re.DOTALL):
        raw = match.group(1)
        for part in raw.split(","):
            part = part.strip()
            if part:
                normalized = normalize_arxiv_id(part)
                if normalized:
                    ids.add(normalized)
    return ids


def compute_recall_reward(text: str, ground_truth_ids: list, max_ratio: float = 2.0) -> float:
    predicted = extract_all_citations(text)
    if not ground_truth_ids:
        return 0.0
    gt_set = {normalize_arxiv_id(gid) for gid in ground_truth_ids}
    if len(predicted) > max_ratio * len(gt_set):
        return 0.0
    correct = predicted & gt_set
    return len(correct) / len(gt_set)


# ============================================================================
# Search API (from skyrl_gym/tools/search.py, simplified)
# ============================================================================

MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1


def call_search_api(url: str, query: str, topk: int = 3, max_date: int = None,
                    timeout: int = 30, session: requests.Session = None) -> tuple:
    """Returns (response_json, error_msg)."""
    payload = {"query": query, "topk": topk, "return_scores": True}
    if max_date is not None:
        payload["max_date"] = max_date
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    if session is None:
        session = requests.Session()

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"Server Error ({response.status_code}) on attempt {attempt + 1}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json(), None
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection Error: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
        except requests.exceptions.Timeout as e:
            last_error = f"Timeout Error: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
        except requests.exceptions.RequestException as e:
            last_error = f"Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            last_error = f"JSON Decode Error: {e}"
            break

    return None, last_error


def format_search_results(api_response: dict) -> str:
    """Format API response into <information> block for the model."""
    if not api_response:
        return "\n<information>" + json.dumps({"result": "Search request failed."}) + "</information>\n"

    raw_results = api_response.get("result", [])
    if not raw_results:
        return "\n<information>" + json.dumps({"result": "No search results found."}) + "</information>\n"

    pretty_parts = []
    for retrieval in raw_results:
        formatted = ""
        for idx, doc_item in enumerate(retrieval):
            content = doc_item["document"]["contents"].strip()
            formatted += f"Doc {idx+1}: {content}\n"
        pretty_parts.append(formatted)
    final_result = "\n---\n".join(pretty_parts)

    return "\n<information>" + json.dumps({"result": final_result}) + "</information>\n"


def do_search(query: str, retriever_url: str, topk: int, max_date: int,
              session: requests.Session) -> str:
    """Execute a single search and return formatted observation."""
    try:
        api_response, error_msg = call_search_api(
            retriever_url, query, topk=topk, max_date=max_date, session=session,
        )
        if error_msg or not api_response:
            return "\n<information>" + json.dumps({"result": f"Search error: {error_msg}"}) + "</information>\n"
        return format_search_results(api_response)
    except Exception as e:
        return "\n<information>" + json.dumps({"result": f"Search error: {e}"}) + "</information>\n"


# ============================================================================
# Batched multi-turn eval loop
# ============================================================================

def evaluate_prompt_batched(llm, sampling_params, messages: list, ground_truth_ids: list,
                            n_samples: int, max_turns: int, topk: int, retriever_url: str,
                            max_date: int = None, session: requests.Session = None,
                            search_workers: int = 8) -> dict:
    """Run n_samples episodes in parallel using batched vLLM generation.

    All samples advance turn-by-turn together. Finished samples are excluded
    from subsequent generation batches.
    """
    citation_budget = len(ground_truth_ids) * 2

    # Initialize n_samples copies of the chat history
    histories = [[dict(m) for m in messages] for _ in range(n_samples)]
    done = [False] * n_samples
    final_turns = [0] * n_samples

    for turn in range(1, max_turns + 1):
        # Collect active (not done) sample indices
        active = [i for i in range(n_samples) if not done[i]]
        if not active:
            break

        # Batch generate for all active samples
        batch_messages = [histories[i] for i in active]
        outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)

        # Collect search queries for parallel execution
        search_tasks = []  # list of (active_idx, query_str)

        for batch_idx, sample_idx in enumerate(active):
            response_text = outputs[batch_idx].outputs[0].text
            histories[sample_idx].append({"role": "assistant", "content": response_text})
            final_turns[sample_idx] = turn

            if "<done>" in response_text or turn >= max_turns:
                done[sample_idx] = True
            else:
                search_match = re.search(r"<search>(.*?)</search>", response_text, re.DOTALL)
                if search_match:
                    search_tasks.append((sample_idx, search_match.group(1).strip()))
                else:
                    search_tasks.append((sample_idx, None))

        # Execute all searches in parallel
        search_results = {}
        if search_tasks:
            queries_to_run = [(idx, q) for idx, q in search_tasks if q is not None]
            if queries_to_run:
                with ThreadPoolExecutor(max_workers=search_workers) as executor:
                    futures = {
                        executor.submit(do_search, q, retriever_url, topk, max_date, session): idx
                        for idx, q in queries_to_run
                    }
                    for future in futures:
                        idx = futures[future]
                        try:
                            search_results[idx] = future.result(timeout=60)
                        except Exception as e:
                            search_results[idx] = "\n<information>" + json.dumps({"result": f"Search error: {e}"}) + "</information>\n"

        # Append observations and turn messages for still-active samples
        for sample_idx, query in search_tasks:
            if done[sample_idx]:
                continue

            observation = search_results.get(sample_idx)

            # Count citations so far
            full_text = "".join(m["content"] for m in histories[sample_idx])
            num_cited = len(extract_all_citations(full_text))

            remaining = max_turns - turn
            if remaining > 1:
                turn_msg = f"\n\n{remaining} turns remaining. Citations so far: {num_cited}/{citation_budget} max. You may cite fewer than the max — only cite papers you are confident belong in this subsection."
            elif remaining == 1:
                turn_msg = f"\n\nThis is your last turn. Citations so far: {num_cited}/{citation_budget} max. Cite any remaining papers and write <done></done>."
            else:
                turn_msg = ""

            if observation:
                user_content = observation
                if turn_msg:
                    user_content += turn_msg
            else:
                user_content = turn_msg.strip() if turn_msg else None

            if user_content:
                histories[sample_idx].append({"role": "user", "content": user_content})

    # Compute rewards and metrics for each sample
    results = []
    gt_set = {normalize_arxiv_id(gid) for gid in ground_truth_ids}
    for i in range(n_samples):
        full_text = "".join(m["content"] for m in histories[i])
        reward = compute_recall_reward(full_text, ground_truth_ids, max_ratio=2.0)
        predicted = extract_all_citations(full_text)
        correct = predicted & gt_set
        recall = len(correct) / len(gt_set) if gt_set else 0.0
        precision = len(correct) / len(predicted) if predicted else 0.0
        answered = "<done>" in full_text

        results.append({
            "reward": reward,
            "recall": recall,
            "precision": precision,
            "num_predicted": len(predicted),
            "num_ground_truth": len(gt_set),
            "num_correct": len(correct),
            "answered": int(answered),
            "num_turns": final_turns[i],
            "chat_history": histories[i],
        })

    rewards = [r["reward"] for r in results]
    pass_at_1 = float(np.mean([r > 0 for r in rewards]))
    pass_at_k = 1.0 if any(r > 0 for r in rewards) else 0.0

    return {
        "n_samples": n_samples,
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)),
        "best_reward": float(max(rewards)),
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "mean_recall": float(np.mean([r["recall"] for r in results])),
        "mean_precision": float(np.mean([r["precision"] for r in results])),
        "mean_num_predicted": float(np.mean([r["num_predicted"] for r in results])),
        "mean_num_turns": float(np.mean([r["num_turns"] for r in results])),
        "mean_answered": float(np.mean([r["answered"] for r in results])),
        "num_ground_truth": results[0]["num_ground_truth"],
        "per_sample": results,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone citation prediction eval with vLLM")
    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=65536, help="Max model context length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to parquet file")

    # Retriever
    parser.add_argument("--retriever_url", type=str, required=True, help="Retriever API URL")

    # Eval params
    parser.add_argument("--max_turns", type=int, default=6, help="Max search turns")
    parser.add_argument("--topk", type=int, default=5, help="Search results per query")
    parser.add_argument("--n_samples", type=int, default=20, help="Samples per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--presence_penalty", type=float, default=1.5, help="Presence penalty")
    parser.add_argument("--max_gen_length", type=int, default=4096, help="Max tokens per generation")
    parser.add_argument("--max_predictions_ratio", type=float, default=2.0, help="Spam penalty ratio")

    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for results")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Load model
    # ========================================================================
    logger.info(f"Loading model from {args.model_path} (TP={args.tp}, max_model_len={args.max_model_len})")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=False,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_gen_length,
        stop=["</search>", "</done>"],
        include_stop_str_in_output=True,
    )

    logger.info(f"Model loaded. Sampling params: {sampling_params}")

    # ========================================================================
    # Load data
    # ========================================================================
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    logger.info(f"Loaded {len(df)} prompts")

    # ========================================================================
    # Init WandB
    # ========================================================================
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            run_name = args.run_name or f"standalone-m{args.max_turns}-n{args.topk}-k{args.n_samples}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
            )
            logger.info(f"WandB initialized: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # ========================================================================
    # Run eval
    # ========================================================================
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    all_results = []
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # Parse prompt (stored as JSON string)
        messages = row["prompt"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        # Parse reward spec
        reward_spec = row.get("reward_spec", "{}")
        if isinstance(reward_spec, str):
            reward_spec = json.loads(reward_spec)
        ground_truth_ids = reward_spec.get("ground_truth", {}).get("targets", [])

        # Get max_date for temporal filtering
        max_date = row.get("max_date", None)
        if pd.isna(max_date) if isinstance(max_date, float) else max_date is None:
            max_date = None
        else:
            max_date = int(max_date)

        # Parse metadata for logging
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        logger.info(f"Prompt {idx}: {metadata.get('paper_id', '?')} / "
                     f"{metadata.get('subsection_heading', '?')[:50]} "
                     f"({len(ground_truth_ids)} targets, max_date={max_date})")

        prompt_result = evaluate_prompt_batched(
            llm, sampling_params, messages, ground_truth_ids,
            n_samples=args.n_samples, max_turns=args.max_turns,
            topk=args.topk, retriever_url=args.retriever_url,
            max_date=max_date, session=session,
        )

        prompt_result["prompt_idx"] = int(idx)
        prompt_result["paper_id"] = metadata.get("paper_id", "")
        prompt_result["subsection_heading"] = metadata.get("subsection_heading", "")
        prompt_result["max_date"] = max_date

        all_results.append(prompt_result)

        logger.info(f"  -> mean_reward={prompt_result['mean_reward']:.3f}, "
                     f"pass@1={prompt_result['pass_at_1']:.3f}, "
                     f"pass@k={prompt_result['pass_at_k']:.3f}, "
                     f"mean_recall={prompt_result['mean_recall']:.3f}")

        # Log to WandB
        if wandb_run:
            try:
                wandb.log({
                    "prompt_idx": int(idx),
                    "mean_reward": prompt_result["mean_reward"],
                    "best_reward": prompt_result["best_reward"],
                    "pass_at_1": prompt_result["pass_at_1"],
                    "pass_at_k": prompt_result["pass_at_k"],
                    "mean_recall": prompt_result["mean_recall"],
                    "mean_precision": prompt_result["mean_precision"],
                    "mean_num_turns": prompt_result["mean_num_turns"],
                    "num_ground_truth": prompt_result["num_ground_truth"],
                })
            except Exception as e:
                logger.warning(f"WandB log failed: {e}")

    elapsed = time.time() - start_time

    # ========================================================================
    # Aggregate results
    # ========================================================================
    n_prompts = len(all_results)
    agg = {
        "n_prompts": n_prompts,
        "n_samples": args.n_samples,
        "max_turns": args.max_turns,
        "topk": args.topk,
        "temperature": args.temperature,
        "model_path": args.model_path,
        "elapsed_seconds": elapsed,
        "mean_reward": float(np.mean([r["mean_reward"] for r in all_results])),
        "mean_best_reward": float(np.mean([r["best_reward"] for r in all_results])),
        "pass_at_1": float(np.mean([r["pass_at_1"] for r in all_results])),
        "pass_at_k": float(np.mean([r["pass_at_k"] for r in all_results])),
        "mean_recall": float(np.mean([r["mean_recall"] for r in all_results])),
        "mean_precision": float(np.mean([r["mean_precision"] for r in all_results])),
        "mean_num_turns": float(np.mean([r["mean_num_turns"] for r in all_results])),
        "mean_answered": float(np.mean([r["mean_answered"] for r in all_results])),
    }

    logger.info("=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)
    for k, v in agg.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # ========================================================================
    # Save results
    # ========================================================================
    # Save aggregate
    agg_path = output_dir / "aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info(f"Aggregate results saved to {agg_path}")

    # Save per-prompt results (without chat histories for space)
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "prompt_idx": r["prompt_idx"],
            "paper_id": r["paper_id"],
            "subsection_heading": r["subsection_heading"],
            "num_ground_truth": r["num_ground_truth"],
            "mean_reward": r["mean_reward"],
            "best_reward": r["best_reward"],
            "pass_at_1": r["pass_at_1"],
            "pass_at_k": r["pass_at_k"],
            "mean_recall": r["mean_recall"],
            "mean_precision": r["mean_precision"],
            "mean_num_turns": r["mean_num_turns"],
            "mean_answered": r["mean_answered"],
            "rewards": r["rewards"],
        })
    summary_path = output_dir / "per_prompt_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info(f"Per-prompt results saved to {summary_path}")

    # Save trajectories (chat histories for first 3 samples per prompt)
    trajectories_path = output_dir / "trajectories.json"
    save_trajs = []
    for r in all_results:
        entry = {
            "prompt_idx": r["prompt_idx"],
            "paper_id": r["paper_id"],
            "samples": [],
        }
        for i, s in enumerate(r["per_sample"]):
            se = {
                "reward": s["reward"],
                "recall": s["recall"],
                "num_predicted": s["num_predicted"],
                "num_correct": s["num_correct"],
                "num_turns": s["num_turns"],
                "answered": s["answered"],
            }
            if i < 3:
                se["chat_history"] = s["chat_history"]
            entry["samples"].append(se)
        save_trajs.append(entry)
    with open(trajectories_path, "w") as f:
        json.dump(save_trajs, f)
    logger.info(f"Trajectories saved to {trajectories_path}")

    # Final WandB summary
    if wandb_run:
        try:
            wandb.summary.update(agg)
            wandb.finish()
        except Exception as e:
            logger.warning(f"WandB finish failed: {e}")

    logger.info(f"Done! Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
