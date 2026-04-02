"""Gemini Batch API version of subsection splitting.

Three modes:
  submit  — build JSONL, upload, create batch job
  poll    — check batch job status
  collect — download results, resolve citations, write output

Usage:
    # Submit batch job
    python gemini_batch_split.py submit \
        --input filtered_papers.json \
        --model gemini-3-flash-preview

    # Check status (prints job name on submit)
    python gemini_batch_split.py poll --job-name batches/XXXX

    # Collect results
    python gemini_batch_split.py collect \
        --job-name batches/XXXX \
        --input filtered_papers.json \
        --output gemini_subsections_all.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# Load .env
for _env_candidate in [
    Path(__file__).resolve().parent.parent.parent.parent / "search" / "harish_setup" / ".env",
    Path(__file__).resolve().parent.parent / "harish_setup" / ".env",
]:
    if _env_candidate.exists():
        load_dotenv(_env_candidate)
        break

from google import genai
from google.genai import types

# Reuse prompt building from the main script
from gemini_split_subsections import (
    SplitResponse,
    build_prompt,
    resolve_subsets_to_ids,
)


def cmd_submit(args):
    """Build JSONL requests file, upload, and create batch job."""
    # Load papers
    print(f"Loading papers from {args.input}...")
    with open(args.input) as f:
        papers = json.load(f)
    if args.split:
        papers = [p for p in papers if p["split"] == args.split]
    print(f"  {len(papers)} papers to process")

    # Build JSONL request file
    jsonl_path = args.input.replace(".json", "_batch_requests.jsonl")
    print(f"Building batch request file: {jsonl_path}")

    # Inline JSON schema (no $ref) for Gemini Batch API compatibility
    schema = {
        "type": "OBJECT",
        "properties": {
            "subsets": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "citation_strings": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        },
                        "query": {"type": "STRING"},
                    },
                    "required": ["citation_strings", "query"],
                },
            }
        },
        "required": ["subsets"],
    }

    with open(jsonl_path, "w") as f:
        for paper in papers:
            prompt = build_prompt(paper)
            request_line = {
                "key": paper["paper_id"],
                "request": {
                    "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                    "generation_config": {
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                        "temperature": 0.3,
                        "max_output_tokens": 8192,
                    },
                },
            }
            f.write(json.dumps(request_line) + "\n")

    file_size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)
    print(f"  Written {len(papers)} requests ({file_size_mb:.1f} MB)")

    # Upload file
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("Uploading batch request file...")
    uploaded_file = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=f"citation-split-{len(papers)}papers",
            mime_type="jsonl",
        ),
    )
    print(f"  Uploaded: {uploaded_file.name}")

    # Create batch job
    print(f"Creating batch job (model={args.model})...")
    batch_job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={"display_name": f"citation-split-{len(papers)}papers"},
    )
    print(f"\n=== Batch Job Created ===")
    print(f"Job name: {batch_job.name}")
    print(f"State: {batch_job.state.name}")
    print(f"\nTo check status:")
    print(f"  python gemini_batch_split.py poll --job-name {batch_job.name}")
    print(f"\nTo collect results when done:")
    print(f"  python gemini_batch_split.py collect --job-name {batch_job.name} --input {args.input} --output OUTPUT.json")

    # Save job name for convenience
    job_info_path = args.input.replace(".json", "_batch_job.json")
    with open(job_info_path, "w") as f:
        json.dump({"job_name": batch_job.name, "model": args.model, "num_papers": len(papers)}, f, indent=2)
    print(f"\nJob info saved to: {job_info_path}")


def cmd_poll(args):
    """Poll batch job status."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    batch_job = client.batches.get(name=args.job_name)
    print(f"Job: {args.job_name}")
    print(f"State: {batch_job.state.name}")

    if args.wait:
        while batch_job.state.name not in completed_states:
            print(f"  {time.strftime('%H:%M:%S')} - {batch_job.state.name}")
            time.sleep(30)
            batch_job = client.batches.get(name=args.job_name)

        print(f"\nFinal state: {batch_job.state.name}")
        if batch_job.state.name == "JOB_STATE_FAILED":
            print(f"Error: {batch_job.error}")


def cmd_collect(args):
    """Download batch results and process into final output."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    # Check job status
    batch_job = client.batches.get(name=args.job_name)
    print(f"Job: {args.job_name}")
    print(f"State: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"ERROR: Job not succeeded (state={batch_job.state.name})")
        sys.exit(1)

    # Download result file
    print("Downloading results...")
    result_file_name = batch_job.dest.file_name
    file_data = client.files.download(file=result_file_name)
    result_text = file_data.decode("utf-8")

    # Parse results
    results_by_key = {}
    errors = 0
    for line in result_text.strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        key = entry.get("key", "")
        response = entry.get("response")
        if response and response.get("candidates"):
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            try:
                parsed = json.loads(text)
                results_by_key[key] = {"status": "ok", "subsets": parsed.get("subsets", [])}
            except json.JSONDecodeError:
                results_by_key[key] = {"status": "error", "subsets": []}
                errors += 1
        else:
            results_by_key[key] = {"status": "error", "subsets": []}
            errors += 1

    print(f"  Parsed {len(results_by_key)} results ({errors} errors)")

    # Load papers for resolution
    print(f"Loading papers from {args.input}...")
    with open(args.input) as f:
        papers = json.load(f)
    paper_lookup = {p["paper_id"]: p for p in papers}

    # Resolve citation strings -> arxiv IDs
    print("Resolving citation strings to arxiv IDs...")
    final_results = []
    stats = defaultdict(int)

    for paper_id, result in results_by_key.items():
        paper = paper_lookup.get(paper_id)
        if paper is None:
            stats["paper_not_found"] += 1
            continue

        if result["status"] != "ok" or not result.get("subsets"):
            stats["gemini_error"] += 1
            continue

        resolved = resolve_subsets_to_ids(result, paper)

        if not resolved:
            stats["no_valid_subsets"] += 1
            continue

        stats["success"] += 1
        final_results.append({
            "paper_id": paper_id,
            "title": paper["title"],
            "abstract": paper["abstract"],
            "introduction": paper["introduction"],
            "related_work_text": paper["related_work_text"],
            "existing_subsection_headings": paper.get("existing_subsection_headings", []),
            "all_citation_ids": paper["all_citation_ids"],
            "citation_display_map": paper.get("citation_display_map", {}),
            "split": paper["split"],
            "subsets": resolved,
        })

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # Summary
    total_subsets = sum(len(r["subsets"]) for r in final_results)
    total_citations = sum(
        len(s["citation_ids"])
        for r in final_results
        for s in r["subsets"]
    )
    print(f"\n=== Summary ===")
    print(f"Results from batch: {len(results_by_key)}")
    print(f"  Gemini errors: {stats.get('gemini_error', 0)}")
    print(f"  No valid subsets: {stats.get('no_valid_subsets', 0)}")
    print(f"  Success: {stats.get('success', 0)}")
    print(f"\nFinal papers: {len(final_results)}")
    print(f"Total subsets: {total_subsets}")
    print(f"Total citations: {total_citations}")
    if total_subsets > 0:
        print(f"Avg citations/subset: {total_citations/total_subsets:.1f}")
        print(f"Avg subsets/paper: {total_subsets/len(final_results):.1f}")

    # Split breakdown
    split_counts = defaultdict(int)
    for r in final_results:
        split_counts[r["split"]] += 1
    print(f"\nBy split: {dict(split_counts)}")
    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Gemini Batch API subsection splitting")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # submit
    p_submit = subparsers.add_parser("submit", help="Build JSONL, upload, create batch job")
    p_submit.add_argument("--input", required=True, help="Path to filtered_papers.json")
    p_submit.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model")
    p_submit.add_argument("--split", default=None, choices=["train", "validation", "test"])

    # poll
    p_poll = subparsers.add_parser("poll", help="Check batch job status")
    p_poll.add_argument("--job-name", required=True, help="Batch job name from submit")
    p_poll.add_argument("--wait", action="store_true", help="Poll until completion")

    # collect
    p_collect = subparsers.add_parser("collect", help="Download results and process")
    p_collect.add_argument("--job-name", required=True, help="Batch job name")
    p_collect.add_argument("--input", required=True, help="Path to filtered_papers.json")
    p_collect.add_argument("--output", required=True, help="Output gemini_subsections.json")

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "poll":
        cmd_poll(args)
    elif args.command == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    main()
