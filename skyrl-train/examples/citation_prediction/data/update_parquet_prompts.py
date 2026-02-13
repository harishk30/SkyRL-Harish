"""
Update the system prompt in existing citation prediction parquet files.

Use this to switch between prompt styles without re-running the full
build_citation_dataset.py pipeline (which requires the metadata CSV).

Usage:
    python update_parquet_prompts.py \
        --input_dir /path/to/existing/parquets \
        --output_dir /path/to/output \
        --prompt_style extended
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from build_citation_dataset import SYSTEM_PROMPTS


def update_parquet(input_path: Path, output_path: Path, prompt_style: str) -> int:
    """Update system prompts in a parquet file. Returns number of rows."""
    df = pd.read_parquet(input_path)
    system_prompt = SYSTEM_PROMPTS[prompt_style]

    def update_prompt(prompt_json: str) -> str:
        messages = json.loads(prompt_json)
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] = system_prompt
                break
        return json.dumps(messages)

    df["prompt"] = df["prompt"].apply(update_prompt)
    df.to_parquet(output_path, index=False)
    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Update system prompts in existing citation prediction parquets."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory with existing parquets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory (can be same as input to overwrite)",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        required=True,
        choices=list(SYSTEM_PROMPTS.keys()),
        help="Target prompt style",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        input_path = input_dir / f"{split}.parquet"
        if not input_path.exists():
            print(f"  {split}: not found, skipping")
            continue
        output_path = output_dir / f"{split}.parquet"
        n = update_parquet(input_path, output_path, args.prompt_style)
        print(f"  {split}: {n} rows -> {output_path}")

    print(f"\nDone. Prompt style: {args.prompt_style}")


if __name__ == "__main__":
    main()
