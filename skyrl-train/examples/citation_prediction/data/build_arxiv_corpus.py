#!/usr/bin/env python3
"""Build arxiv corpus with IDs prepended to the contents field.

Reads the original arxiv_wikiformat.jsonl and prepends [arxiv:<id>] to each
line's `contents` field so that the arxiv ID is visible in retriever output text
(used by _passages2string → "Doc N: [arxiv:XXXX.XXXXX] Title...").

Row ordering is preserved, so FAISS positional indexing stays aligned.

Usage:
    python examples/citation_prediction/data/build_arxiv_corpus.py \
        --input /home/hk4638/scratch/shared/data/arxiv_wikiformat.jsonl \
        --output /home/hk4638/scratch/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Prepend arxiv IDs to corpus contents field.")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/hk4638/scratch/shared/data/arxiv_wikiformat.jsonl",
        help="Path to original arxiv_wikiformat.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/hk4638/scratch/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl",
        help="Path to output JSONL with IDs in contents",
    )
    args = parser.parse_args()

    count = 0
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            arxiv_id = obj["id"]
            obj["contents"] = f"[arxiv:{arxiv_id}] {obj['contents']}"
            fout.write(json.dumps(obj) + "\n")
            count += 1
            if count % 500_000 == 0:
                print(f"Processed {count:,} lines...", flush=True)

    print(f"Done. Wrote {count:,} lines to {args.output}")


if __name__ == "__main__":
    main()
