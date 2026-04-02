#!/usr/bin/env python3
"""Citation-prediction SFT trainer using SkyRL policy workers only."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

import hydra
import numpy as np
import ray
import torch
from loguru import logger
from omegaconf import OmegaConf, open_dict

from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.utils import initialize_ray, validate_cfg
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl_train.workers.worker_dispatch import WorkerDispatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared tokenized SFT dataset JSONL.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        help="HF model/tokenizer path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported checkpoints.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-batch-tokens", type=int, default=24000)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--micro-train-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--save-every-steps", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--dp-size", type=int, default=None, help="Override data-parallel size used for batching.")
    parser.add_argument("--dry-run", action="store_true", help="Load data, build batches, print stats, then exit.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    if not examples:
        raise ValueError(f"No SFT examples found in {path}")
    return examples


def build_batches(
    examples: list[dict[str, Any]],
    max_batch_tokens: int,
    max_batch_size: int,
    dp_size: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    if max_batch_size % dp_size != 0:
        raise ValueError(f"max_batch_size must be divisible by dp_size, got {max_batch_size} and {dp_size}")

    ordered = sorted(examples, key=lambda x: x["sequence_length"])
    batches: list[list[dict[str, Any]]] = []
    dropped: list[dict[str, Any]] = []

    for i in range(0, len(ordered), max_batch_size):
        batch = ordered[i : i + max_batch_size]
        if len(batch) < max_batch_size:
            dropped.extend(batch)
            break
        batches.append(batch)

    if batches:
        max_observed = max(max(ex["sequence_length"] for ex in batch) * len(batch) for batch in batches)
        if max_observed > max_batch_tokens:
            logger.warning(
                f"Observed batch token cost {max_observed} exceeds max_batch_tokens={max_batch_tokens}. "
                "This script currently treats max_batch_tokens as a diagnostic, not a hard cap, when dp_size > 1."
            )
    return batches, dropped


def collate_sft_batch(examples: list[dict[str, Any]], pad_token_id: int) -> TrainingInputBatch:
    max_seq_len = max(example["sequence_length"] for example in examples)
    max_resp_len = max(example["response_length"] for example in examples)

    sequences = []
    attention_masks = []
    loss_masks = []

    for example in examples:
        full_sequence = example["prompt_token_ids"] + example["response_ids"]
        pad_len = max_seq_len - len(full_sequence)
        response_pad = max_resp_len - len(example["response_ids"])

        sequences.append([pad_token_id] * pad_len + full_sequence)
        attention_masks.append([0] * pad_len + [1] * len(full_sequence))
        loss_masks.append([0] * response_pad + example["loss_mask"])

    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor(sequences, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "loss_mask": torch.tensor(loss_masks, dtype=torch.long),
        }
    )
    batch.metadata = {"response_length": max_resp_len}
    return batch


def get_sft_config(args: argparse.Namespace, num_gpus: int, num_training_steps: int):
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = args.model_path
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.logger = "console"
    cfg.trainer.use_sample_packing = False
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.gradient_checkpointing_use_reentrant = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.trainer.policy.sequence_parallel_size = 1
    cfg.trainer.micro_train_batch_size_per_gpu = args.micro_train_batch_size_per_gpu
    cfg.trainer.train_batch_size = args.max_batch_size
    cfg.trainer.policy_mini_batch_size = args.max_batch_size
    cfg.trainer.policy.optimizer_config.lr = args.learning_rate
    cfg.trainer.policy.optimizer_config.weight_decay = args.weight_decay
    cfg.trainer.policy.optimizer_config.max_grad_norm = args.max_grad_norm
    cfg.trainer.policy.optimizer_config.num_warmup_steps = min(args.warmup_steps, num_training_steps)
    cfg.trainer.export_path = str(args.output_dir)

    # The FSDP policy worker now unconditionally reads this field. Force-add it
    # here because older base configs may not define it under struct mode.
    # Keep the schema default (`False`) unless SFT explicitly opts in later.
    with open_dict(cfg.trainer.policy):
        cfg.trainer.policy.use_liger_kernel = False
    if OmegaConf.select(cfg, "trainer.policy.fsdp_config.cpu_offload") is not None:
        cfg.trainer.policy.fsdp_config.cpu_offload = False

    validate_cfg(cfg)
    return cfg


def save_model(dispatch: WorkerDispatch, tokenizer, output_dir: Path, step: int) -> None:
    export_dir = output_dir / f"step_{step}"
    export_dir.mkdir(parents=True, exist_ok=True)
    dispatch.save_hf_model("policy", str(export_dir / "policy"), tokenizer)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    examples = load_examples(args.dataset)
    if args.max_sequence_length is not None:
        before = len(examples)
        examples = [example for example in examples if example["sequence_length"] <= args.max_sequence_length]
        logger.info(f"Filtered max_sequence_length={args.max_sequence_length}: kept {len(examples)}/{before}")
        if not examples:
            raise ValueError("All examples were filtered out by --max-sequence-length")

    num_gpus = args.dp_size if args.dp_size is not None else torch.cuda.device_count()
    if num_gpus <= 0:
        num_gpus = 1

    batches = build_batches(
        examples=examples,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_size=args.max_batch_size,
        dp_size=num_gpus,
    )
    batches, dropped_for_divisibility = batches
    total_steps = len(batches) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    seq_lens = [example["sequence_length"] for example in examples]
    logger.info(
        "Loaded {} examples | seq len min={} p50={} max={} | {} batches/epoch | dropped_for_divisibility={}".format(
            len(examples),
            min(seq_lens),
            sorted(seq_lens)[len(seq_lens) // 2],
            max(seq_lens),
            len(batches),
            len(dropped_for_divisibility),
        )
    )

    if args.dry_run:
        batch_token_costs = [max(ex["sequence_length"] for ex in batch) * len(batch) for batch in batches]
        print(
            json.dumps(
                {
                    "examples": len(examples),
                    "batches_per_epoch": len(batches),
                    "max_batch_tokens_observed": max(batch_token_costs),
                    "mean_batch_tokens_observed": mean(batch_token_costs),
                    "max_batch_size_observed": max(len(batch) for batch in batches),
                    "dp_size": num_gpus,
                    "dropped_for_divisibility": len(dropped_for_divisibility),
                },
                indent=2,
            )
        )
        return

    if num_gpus <= 0:
        raise RuntimeError("No CUDA devices visible for SFT training.")

    cfg = get_sft_config(args=args, num_gpus=num_gpus, num_training_steps=total_steps)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.environ.setdefault("SKYRL_DISABLE_RAY_RUNTIME_ENV", "1")
    initialize_ray(cfg)

    pg = placement_group([{"GPU": num_gpus, "CPU": num_gpus}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=60)

    actor_group = PPORayActorGroup(
        cfg,
        num_nodes=1,
        num_gpus_per_node=num_gpus,
        ray_actor_type=PolicyWorker,
        pg=pg,
        num_gpus_per_actor=1,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
    )
    ray.get(actor_group.async_init_model(args.model_path, num_training_steps=total_steps))
    ray.get(actor_group.async_run_ray_method("pass_through", "_set_pad_token_id", tokenizer.pad_token_id))

    dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)

    global_step = 0
    train_start = time.time()
    recent_losses: list[float] = []

    try:
        for epoch in range(args.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
            for batch_examples in batches:
                if args.max_steps is not None and global_step >= args.max_steps:
                    break

                batch = collate_sft_batch(batch_examples, tokenizer.pad_token_id)
                metrics = dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")
                grad_norm = dispatch.optim_step("policy")

                global_step += 1
                loss_value = float(metrics["loss"])
                recent_losses.append(loss_value)
                if len(recent_losses) > args.log_every:
                    recent_losses.pop(0)

                if global_step == 1 or global_step % args.log_every == 0:
                    logger.info(
                        "step={} loss={:.4f} avg_recent_loss={:.4f} grad_norm={:.4f} batch_size={} max_seq={} max_resp={}".format(
                            global_step,
                            loss_value,
                            mean(recent_losses),
                            float(grad_norm) if grad_norm is not None else float("nan"),
                            len(batch_examples),
                            max(ex["sequence_length"] for ex in batch_examples),
                            max(ex["response_length"] for ex in batch_examples),
                        )
                    )

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    save_model(dispatch, tokenizer, args.output_dir, global_step)

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        save_model(dispatch, tokenizer, args.output_dir, global_step)
        summary = {
            "global_step": global_step,
            "epochs_requested": args.epochs,
            "examples": len(examples),
            "batches_per_epoch": len(batches),
            "wall_time_sec": time.time() - train_start,
            "dataset": str(args.dataset),
            "model_path": args.model_path,
        }
        with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Training complete: {json.dumps(summary, indent=2)}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
