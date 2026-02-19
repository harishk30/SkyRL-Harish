#!/usr/bin/env python3
"""Gradio viewer for the citation prediction v2 dataset.

Displays the prompt structure (title, intro, subsection heading) and
ground-truth citation targets for each example.

Usage:
    conda activate viewer
    GRADIO_TEMP_DIR=/scratch/gpfs/ZHUANGL/hk4638/gradio_tmp \
        python dataset_viewer.py --data /path/to/train.parquet
"""

import argparse
import json
from pathlib import Path

import gradio as gr
import pandas as pd


def load_dataset(parquet_path: str) -> list[dict]:
    """Load parquet and parse JSON fields into structured dicts."""
    df = pd.read_parquet(parquet_path)
    examples = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        reward_spec = row.get("reward_spec", "{}")
        if isinstance(reward_spec, str):
            reward_spec = json.loads(reward_spec)

        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # Extract parts from the user message
        user_msg = ""
        system_msg = ""
        for msg in prompt:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_msg = msg["content"]

        targets = reward_spec.get("ground_truth", {}).get("targets", [])

        # Parse user prompt into components
        title = ""
        abstract = ""
        introduction = ""
        import re
        title_match = re.search(r"Paper title:\s*(.*?)(?:\n\nAbstract:|\n\nIntroduction:)", user_msg, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()

        abstract_match = re.search(r"Abstract:\n(.*?)(?:\n\nIntroduction:)", user_msg, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()

        intro_match = re.search(r"Introduction:\n(.*?)(?:\n\nRelated Work subsection heading:)", user_msg, re.DOTALL)
        if intro_match:
            introduction = intro_match.group(1).strip()

        examples.append({
            "index": len(examples),
            "system_prompt": system_msg,
            "user_prompt": user_msg,
            "title": title,
            "abstract": abstract,
            "introduction": introduction,
            "subsection_heading": metadata.get("subsection_heading", ""),
            "paper_id": metadata.get("paper_id", ""),
            "num_citations": metadata.get("num_citations", len(targets)),
            "target_ids": targets,
        })
    return examples


def build_viewer(examples: list[dict], parquet_path: str):
    """Build and return the Gradio interface."""

    # Summary stats
    num_examples = len(examples)
    citation_counts = [ex["num_citations"] for ex in examples]
    avg_citations = sum(citation_counts) / len(citation_counts) if citation_counts else 0

    # Distribution
    from collections import Counter
    dist = Counter(citation_counts)
    dist_text = "\n".join(
        f"  {k} citations: {dist[k]} examples"
        for k in sorted(dist.keys())
    )

    with gr.Blocks(title="Citation Prediction v2 Dataset Viewer") as demo:
        gr.Markdown(f"# Citation Prediction v2 Dataset Viewer")
        gr.Markdown(
            f"**File**: `{parquet_path}`\n\n"
            f"**Total examples**: {num_examples} | "
            f"**Mean citations/example**: {avg_citations:.1f} | "
            f"**Range**: {min(citation_counts)}-{max(citation_counts)}"
        )

        with gr.Tabs():
            # ---- Tab 1: Browse examples ----
            with gr.TabItem("Browse Examples"):
                with gr.Row():
                    example_slider = gr.Slider(
                        minimum=0, maximum=num_examples - 1,
                        step=1, value=0, label="Example Index"
                    )
                    filter_min = gr.Number(value=2, label="Min Citations", precision=0)
                    filter_max = gr.Number(value=25, label="Max Citations", precision=0)

                with gr.Row():
                    paper_id_box = gr.Textbox(label="Paper ID", interactive=False)
                    heading_box = gr.Textbox(label="Subsection Heading", interactive=False)
                    num_citations_box = gr.Textbox(label="# Ground Truth Citations", interactive=False)

                targets_box = gr.Textbox(
                    label="Ground Truth Arxiv IDs",
                    lines=3, interactive=False
                )

                title_box = gr.Textbox(
                    label="Title",
                    lines=2, interactive=False
                )

                abstract_box = gr.Textbox(
                    label="Abstract",
                    lines=6, interactive=False
                )

                introduction_box = gr.Textbox(
                    label="Introduction",
                    lines=12, interactive=False
                )

                system_prompt_box = gr.Textbox(
                    label="System Prompt",
                    lines=8, interactive=False
                )

                all_outputs = [paper_id_box, heading_box, num_citations_box,
                               targets_box, title_box, abstract_box,
                               introduction_box, system_prompt_box]

                def show_example(idx, min_c, max_c):
                    # Filter examples
                    filtered = [
                        ex for ex in examples
                        if min_c <= ex["num_citations"] <= max_c
                    ]
                    if not filtered:
                        return ("", "", "", "", "", "", "", "No examples match filter")

                    idx = int(idx) % len(filtered)
                    ex = filtered[idx]

                    return (
                        ex["paper_id"],
                        ex["subsection_heading"],
                        str(ex["num_citations"]),
                        ", ".join(ex["target_ids"]),
                        ex["title"],
                        ex["abstract"],
                        ex["introduction"],
                        ex["system_prompt"],
                    )

                for inp in [example_slider, filter_min, filter_max]:
                    inp.change(
                        show_example,
                        inputs=[example_slider, filter_min, filter_max],
                        outputs=all_outputs,
                    )

                # Initial load
                demo.load(
                    show_example,
                    inputs=[example_slider, filter_min, filter_max],
                    outputs=all_outputs,
                )

            # ---- Tab 2: Distribution ----
            with gr.TabItem("Statistics"):
                gr.Markdown("### Citation Count Distribution")
                gr.Textbox(
                    value=dist_text,
                    label="Citations per subsection",
                    lines=20, interactive=False
                )

                # Heading word frequency
                from collections import Counter as C2
                heading_words = []
                for ex in examples:
                    heading_words.extend(ex["subsection_heading"].lower().split())
                word_freq = C2(heading_words).most_common(30)
                freq_text = "\n".join(f"  {w}: {c}" for w, c in word_freq)
                gr.Markdown("### Top 30 Heading Words")
                gr.Textbox(
                    value=freq_text,
                    label="Word frequency in subsection headings",
                    lines=15, interactive=False
                )

                # Sample headings
                sample_headings = sorted(set(
                    ex["subsection_heading"] for ex in examples
                ))[:50]
                gr.Markdown("### Sample Subsection Headings (first 50 unique)")
                gr.Textbox(
                    value="\n".join(sample_headings),
                    label="Headings",
                    lines=20, interactive=False
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio viewer for v2 dataset")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to parquet file")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    print(f"Loading dataset: {args.data}")
    examples = load_dataset(args.data)
    print(f"Loaded {len(examples)} examples")

    demo = build_viewer(examples, args.data)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
