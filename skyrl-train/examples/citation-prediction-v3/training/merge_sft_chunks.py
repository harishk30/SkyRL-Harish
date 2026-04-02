#!/usr/bin/env python3
"""Merge chunked SFT trajectory outputs into a single file.

Usage:
    python merge_sft_chunks.py \
        --input_dir /path/to/sft_trajectories_v3/ \
        --output /path/to/sft_merged.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge SFT trajectory chunks")
    parser.add_argument("--input_dir", required=True, help="Directory with chunk files")
    parser.add_argument("--output", required=True, help="Merged output JSON")
    parser.add_argument("--pattern", default="sft_chunk*_of_*.json",
                        help="Glob pattern for chunk files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    chunk_files = sorted(input_dir.glob(args.pattern))

    if not chunk_files:
        print(f"No files matching {args.pattern} in {input_dir}")
        return

    merged: list[dict] = []
    for f in chunk_files:
        with open(f) as fh:
            data = json.load(fh)
        print(f"  {f.name}: {len(data)} trajectories")
        merged.extend(data)

    # Stats
    paper_ids = set(r["paper_id"] for r in merged)
    recalls = [r["recall"] for r in merged]
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged {len(merged)} trajectories from {len(chunk_files)} chunks")
    print(f"Unique queries: {len(paper_ids)}")
    print(f"Avg recall: {avg_recall:.3f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
