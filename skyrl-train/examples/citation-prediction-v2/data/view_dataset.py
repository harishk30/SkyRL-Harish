"""
Gradio interface for browsing the masked citation prediction dataset.

Supports viewing both short and extended prompt styles side-by-side.

Usage:
    conda activate /home/hk4638/scratch/conda/envs/gradio_viewer
    python examples/citation_prediction/data/view_dataset.py \
        --data_dir /home/hk4638/scratch/data/citation_prediction/
"""

import argparse
import json
from pathlib import Path

import gradio as gr
import pandas as pd


def load_dataset(data_dir: str) -> pd.DataFrame:
    """Load all split parquets and unpack JSON fields into flat columns."""
    frames = []
    for split in ["train", "validation", "test"]:
        path = Path(data_dir) / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["split"] = split
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    records = []
    for _, row in raw.iterrows():
        extra = json.loads(row["extra_info"])
        meta = json.loads(row["metadata"])
        reward = json.loads(row["reward_spec"])

        masked = extra["question"]
        citation = meta["original_citation"]
        arxiv_id = reward["ground_truth"]["target"]
        paper_id = meta["paper_id"]
        split = extra["split"]

        original = masked.replace("[MASKED]", f"({citation})")

        # Extract system prompt from the prompt messages
        messages = json.loads(row["prompt"])
        system_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break

        records.append(
            {
                "split": split,
                "paper_id": paper_id,
                "original_sentence": original,
                "masked_sentence": masked,
                "gold_citation": citation,
                "arxiv_id": arxiv_id,
                "system_prompt": system_prompt,
            }
        )

    return pd.DataFrame(records)


def build_app(datasets: dict[str, pd.DataFrame]) -> gr.Blocks:
    prompt_styles = list(datasets.keys())
    default_style = prompt_styles[0]
    df0 = datasets[default_style]
    splits = sorted(df0["split"].unique().tolist())
    split_counts = {s: len(df0[df0["split"] == s]) for s in splits}

    with gr.Blocks(title="Citation Prediction Dataset Viewer") as app:
        gr.Markdown("# Masked Citation Prediction Dataset Viewer")
        gr.Markdown(
            f"**Total examples**: {len(df0)} &nbsp;|&nbsp; "
            + " &nbsp;|&nbsp; ".join(f"**{s}**: {split_counts[s]}" for s in splits)
        )

        with gr.Row():
            style_dd = gr.Dropdown(
                choices=prompt_styles,
                value=default_style,
                label="Prompt Style",
            )
            split_dd = gr.Dropdown(
                choices=splits, value=splits[0], label="Split"
            )
            idx_slider = gr.Slider(
                minimum=0,
                maximum=split_counts[splits[0]] - 1,
                step=1,
                value=0,
                label="Example index",
            )

        with gr.Row():
            paper_id_box = gr.Textbox(label="Paper ID", interactive=False)
            citation_box = gr.Textbox(label="Gold Citation", interactive=False)
            arxiv_box = gr.Textbox(label="Arxiv ID", interactive=False)

        original_box = gr.Textbox(
            label="Original Sentence", lines=4, interactive=False
        )
        masked_box = gr.Textbox(
            label="Masked Sentence", lines=4, interactive=False
        )
        system_box = gr.Textbox(
            label="System Prompt", lines=10, interactive=False
        )

        state = gr.State(value=None)

        def get_row_outputs(row):
            return (
                row["paper_id"],
                row["gold_citation"],
                row["arxiv_id"],
                row["original_sentence"],
                row["masked_sentence"],
                row["system_prompt"],
            )

        def update_view(style, split, idx_ignored=0):
            df = datasets[style]
            subset = df[df["split"] == split].reset_index(drop=True)
            n = len(subset)
            row = subset.iloc[0]
            return (
                gr.update(maximum=n - 1, value=0),
                *get_row_outputs(row),
                subset,
            )

        def update_index(idx, style, split, subset):
            df = datasets[style]
            if subset is None:
                subset = df[df["split"] == split].reset_index(drop=True)
            idx = int(idx)
            if idx >= len(subset):
                idx = len(subset) - 1
            row = subset.iloc[idx]
            return get_row_outputs(row)

        outputs = [
            idx_slider,
            paper_id_box,
            citation_box,
            arxiv_box,
            original_box,
            masked_box,
            system_box,
            state,
        ]

        detail_outputs = [
            paper_id_box,
            citation_box,
            arxiv_box,
            original_box,
            masked_box,
            system_box,
        ]

        style_dd.change(
            fn=update_view,
            inputs=[style_dd, split_dd],
            outputs=outputs,
        )

        split_dd.change(
            fn=update_view,
            inputs=[style_dd, split_dd],
            outputs=outputs,
        )

        idx_slider.change(
            fn=update_index,
            inputs=[idx_slider, style_dd, split_dd, state],
            outputs=detail_outputs,
        )

        app.load(
            fn=update_view,
            inputs=[style_dd, split_dd],
            outputs=outputs,
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="View citation prediction dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/hk4638/scratch/data/citation_prediction/",
        help="Root directory containing prompt-style subdirectories",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    datasets = {}
    for style_dir in sorted(data_root.iterdir()):
        if style_dir.is_dir() and (style_dir / "train.parquet").exists():
            name = style_dir.name
            print(f"Loading {name}...")
            datasets[name] = load_dataset(str(style_dir))
            print(f"  {name}: {len(datasets[name])} examples")

    if not datasets:
        print("No datasets found! Check --data_dir.")
        return

    app = build_app(datasets)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
