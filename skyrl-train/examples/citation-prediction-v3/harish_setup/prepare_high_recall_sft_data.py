#!/usr/bin/env python3
"""Prepare Claude citation trajectories for SFT and filtered RL training.

This script does three things:
1. Selects a subset of high-recall Claude trajectories.
2. Rebuilds each SFT sample in the exact single-turn token format used by the
   current Qwen citation training path (`generator.use_conversation_multi_turn=false`).
3. Writes a filtered RL dataset with the selected prompts removed from train.

The prompt prefix is taken from the current RL parquet by default, so the SFT
warm-start sees the same initial prompt formatting as the model used for RL.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


PromptKey = tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path(
            "/home/hk4638/SkyRL/skyrl-train/examples/citation-prediction-v3/high_recall_max4_sft_beneficial.json"
        ),
        help="Claude trajectory JSON file.",
    )
    parser.add_argument(
        "--rl-train-parquet",
        type=Path,
        default=Path("/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/full/train.parquet"),
        help="Current RL train parquet used for citation training.",
    )
    parser.add_argument(
        "--rl-data-dir",
        type=Path,
        default=Path("/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v3/full"),
        help="Directory containing RL train/validation/test parquet files.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        help="Tokenizer/model path used to build exact prompt/response token ids.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for prepared SFT data and filtered RL data.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional limit on number of prompts to keep.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["input_order", "recall_desc"],
        default="recall_desc",
        help="How to rank trajectories before applying --max-prompts.",
    )
    parser.add_argument(
        "--prompt-source",
        choices=["rl_train", "trajectory"],
        default="rl_train",
        help="Where to take the initial system+user prompt from.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="If set, drop SFT examples whose exact tokenized sequence length exceeds this cap.",
    )
    return parser.parse_args()


def prompt_key_from_traj(entry: dict[str, Any]) -> PromptKey:
    return entry["paper_id"], entry["subsection_heading"]


def prompt_key_from_metadata(metadata_json: str) -> PromptKey:
    metadata = json.loads(metadata_json)
    return metadata["paper_id"], metadata["subsection_heading"]


def load_rl_train_lookup(rl_train_parquet: Path) -> tuple[pd.DataFrame, dict[PromptKey, dict[str, Any]]]:
    train_df = pd.read_parquet(rl_train_parquet)
    lookup: dict[PromptKey, dict[str, Any]] = {}
    for row in train_df.to_dict(orient="records"):
        key = prompt_key_from_metadata(row["metadata"])
        if key in lookup:
            raise ValueError(f"Duplicate prompt key in RL parquet: {key}")
        lookup[key] = row
    return train_df, lookup


def select_trajectories(entries: list[dict[str, Any]], sort_by: str, max_prompts: int | None) -> list[dict[str, Any]]:
    if sort_by == "recall_desc":
        entries = sorted(
            entries,
            key=lambda x: (
                -float(x["recall"]),
                int(x["num_turns"]),
                -int(x["num_predicted"]),
                x["paper_id"],
                x["subsection_heading"],
            ),
        )
    elif sort_by != "input_order":
        raise ValueError(f"Unsupported sort mode: {sort_by}")

    if max_prompts is not None:
        entries = entries[:max_prompts]
    return entries


def canonical_prompt_messages(
    trajectory: dict[str, Any], rl_row: dict[str, Any], prompt_source: str
) -> tuple[list[dict[str, str]], dict[str, int]]:
    trajectory_prompt = trajectory["messages"][:2]
    rl_prompt = json.loads(rl_row["prompt"])

    mismatch_counts = {
        "system_mismatch": int(trajectory_prompt[0] != rl_prompt[0]),
        "user_mismatch": int(trajectory_prompt[1] != rl_prompt[1]),
    }
    if prompt_source == "trajectory":
        return trajectory_prompt, mismatch_counts
    if prompt_source == "rl_train":
        return rl_prompt, mismatch_counts
    raise ValueError(f"Unsupported prompt source: {prompt_source}")


def build_sft_record(
    trajectory: dict[str, Any],
    rl_row: dict[str, Any],
    tokenizer,
    prompt_source: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    prompt_messages, mismatch_counts = canonical_prompt_messages(trajectory, rl_row, prompt_source)
    prompt_token_ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=True)

    response_ids: list[int] = []
    loss_mask: list[int] = []
    assistant_chars = 0
    observation_chars = 0

    for message in trajectory["messages"][2:]:
        token_ids = tokenizer.encode(message["content"], add_special_tokens=False)
        if message["role"] == "assistant":
            if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
            response_ids.extend(token_ids)
            loss_mask.extend([1] * len(token_ids))
            assistant_chars += len(message["content"])
        else:
            response_ids.extend(token_ids)
            loss_mask.extend([0] * len(token_ids))
            observation_chars += len(message["content"])

    if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
        response_ids.append(tokenizer.eos_token_id)
        loss_mask.append(1)

    metadata = json.loads(rl_row["metadata"])
    reward_spec = json.loads(rl_row["reward_spec"])
    extra_info = json.loads(rl_row["extra_info"])
    sequence_length = len(prompt_token_ids) + len(response_ids)

    return (
        {
            "paper_id": trajectory["paper_id"],
            "subsection_heading": trajectory["subsection_heading"],
            "rich_query": trajectory["rich_query"],
            "recall": trajectory["recall"],
            "num_turns": trajectory["num_turns"],
            "citation_ids": trajectory["citation_ids"],
            "predicted_ids": trajectory["predicted_ids"],
            "passed_threshold": trajectory["passed_threshold"],
            "raw_log_path": trajectory["raw_log_path"],
            "prompt_messages": prompt_messages,
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "loss_mask": loss_mask,
            "prompt_length": len(prompt_token_ids),
            "response_length": len(response_ids),
            "sequence_length": sequence_length,
            "supervised_tokens": int(sum(loss_mask)),
            "assistant_message_count": sum(1 for m in trajectory["messages"][2:] if m["role"] == "assistant"),
            "observation_message_count": sum(1 for m in trajectory["messages"][2:] if m["role"] != "assistant"),
            "assistant_chars": assistant_chars,
            "observation_chars": observation_chars,
            "prompt_source": prompt_source,
            "rl_prompt_metadata": metadata,
            "rl_reward_spec": reward_spec,
            "rl_extra_info": extra_info,
        },
        mismatch_counts,
    )


def copy_rl_eval_files(src_dir: Path, dst_dir: Path) -> None:
    for split in ("validation.parquet", "test.parquet"):
        src = src_dir / split
        if src.exists():
            shutil.copy2(src, dst_dir / split)


def summarize_lengths(records: list[dict[str, Any]]) -> dict[str, Any]:
    seq_lens = sorted(record["sequence_length"] for record in records)
    resp_lens = sorted(record["response_length"] for record in records)
    sup_lens = sorted(record["supervised_tokens"] for record in records)

    def quantile(values: list[int], frac: float) -> int:
        return values[min(len(values) - 1, int(len(values) * frac))]

    return {
        "count": len(records),
        "sequence_length": {
            "min": seq_lens[0],
            "p50": quantile(seq_lens, 0.50),
            "p90": quantile(seq_lens, 0.90),
            "max": seq_lens[-1],
            "mean": mean(seq_lens),
        },
        "response_length": {
            "min": resp_lens[0],
            "p50": quantile(resp_lens, 0.50),
            "p90": quantile(resp_lens, 0.90),
            "max": resp_lens[-1],
            "mean": mean(resp_lens),
        },
        "supervised_tokens": {
            "min": sup_lens[0],
            "p50": quantile(sup_lens, 0.50),
            "p90": quantile(sup_lens, 0.90),
            "max": sup_lens[-1],
            "mean": mean(sup_lens),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_dir = args.output_dir / "sft_data"
    rl_dir = args.output_dir / "rl_data"
    sft_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)

    with args.trajectories.open() as f:
        trajectories: list[dict[str, Any]] = json.load(f)

    trajectories = select_trajectories(trajectories, sort_by=args.sort_by, max_prompts=args.max_prompts)

    rl_train_df, rl_lookup = load_rl_train_lookup(args.rl_train_parquet)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    prepared_records: list[dict[str, Any]] = []
    selected_keys: list[PromptKey] = []
    dropped_too_long: list[dict[str, Any]] = []
    mismatch_totals = {"system_mismatch": 0, "user_mismatch": 0}

    for trajectory in trajectories:
        key = prompt_key_from_traj(trajectory)
        if key not in rl_lookup:
            raise KeyError(f"Trajectory prompt not found in RL train parquet: {key}")

        record, mismatch_counts = build_sft_record(
            trajectory=trajectory,
            rl_row=rl_lookup[key],
            tokenizer=tokenizer,
            prompt_source=args.prompt_source,
        )
        mismatch_totals["system_mismatch"] += mismatch_counts["system_mismatch"]
        mismatch_totals["user_mismatch"] += mismatch_counts["user_mismatch"]

        if args.max_sequence_length is not None and record["sequence_length"] > args.max_sequence_length:
            dropped_too_long.append(
                {
                    "paper_id": record["paper_id"],
                    "subsection_heading": record["subsection_heading"],
                    "sequence_length": record["sequence_length"],
                    "recall": record["recall"],
                }
            )
            continue

        prepared_records.append(record)
        selected_keys.append(key)

    if not prepared_records:
        raise ValueError("No SFT records remain after filtering.")

    sft_path = sft_dir / "train.jsonl"
    with sft_path.open("w", encoding="utf-8") as f:
        for record in prepared_records:
            f.write(json.dumps(record))
            f.write("\n")

    selected_key_set = set(selected_keys)
    keep_mask = [prompt_key_from_metadata(meta) not in selected_key_set for meta in rl_train_df["metadata"]]
    filtered_train_df = rl_train_df.loc[keep_mask].reset_index(drop=True)
    filtered_train_path = rl_dir / "train.parquet"
    filtered_train_df.to_parquet(filtered_train_path, index=False)
    copy_rl_eval_files(args.rl_data_dir, rl_dir)

    manifest = [
        {
            "paper_id": record["paper_id"],
            "subsection_heading": record["subsection_heading"],
            "recall": record["recall"],
            "sequence_length": record["sequence_length"],
            "supervised_tokens": record["supervised_tokens"],
        }
        for record in prepared_records
    ]
    manifest_path = args.output_dir / "selected_prompts.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "input_trajectory_count": len(trajectories),
        "prepared_sft_count": len(prepared_records),
        "dropped_for_length_count": len(dropped_too_long),
        "filtered_rl_train_count": len(filtered_train_df),
        "original_rl_train_count": len(rl_train_df),
        "prompt_source": args.prompt_source,
        "sort_by": args.sort_by,
        "max_prompts": args.max_prompts,
        "max_sequence_length": args.max_sequence_length,
        "prompt_mismatch_counts": mismatch_totals,
        "length_stats": summarize_lengths(prepared_records),
        "paths": {
            "sft_train_jsonl": str(sft_path),
            "selected_prompts_json": str(manifest_path),
            "filtered_rl_train_parquet": str(filtered_train_path),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "dropped_too_long.json").open("w", encoding="utf-8") as f:
        json.dump(dropped_too_long, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
