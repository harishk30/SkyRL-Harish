"""
Gradio Trajectory Viewer for Citation Prediction eval runs.

Usage:
    conda activate viewer   # or any env with gradio
    python trajectory_viewer.py [--checkpoints-dir /path/to/checkpoints] [--port 7860]
"""

import argparse
import html
import json
import os
import re
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Default checkpoint root (override with --checkpoints-dir)
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINTS_DIR = "/scratch/gpfs/ZHUANGL/hk4638/checkpoints"

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_runs(checkpoints_dir: str) -> list[str]:
    """Return run names that have an exports/dumped_evals directory."""
    base = Path(checkpoints_dir)
    if not base.is_dir():
        return []
    runs = []
    for entry in sorted(base.iterdir()):
        evals_dir = entry / "exports" / "dumped_evals"
        if evals_dir.is_dir():
            runs.append(entry.name)
    return runs


def discover_steps(checkpoints_dir: str, run_name: str) -> list[str]:
    """Return sorted step folder names for a given run."""
    evals_dir = Path(checkpoints_dir) / run_name / "exports" / "dumped_evals"
    if not evals_dir.is_dir():
        return []
    steps = []
    for entry in sorted(evals_dir.iterdir()):
        if entry.is_dir() and entry.name.endswith("_evals"):
            steps.append(entry.name)
    # Sort by step number
    def step_num(name: str) -> int:
        m = re.search(r"step_(\d+)", name)
        return int(m.group(1)) if m else 0
    steps.sort(key=step_num)
    return steps


def step_label(name: str) -> str:
    m = re.search(r"step_(\d+)", name)
    return f"Step {m.group(1)}" if m else name


def discover_datasets(checkpoints_dir: str, run_name: str, step: str) -> list[str]:
    """Return dataset JSONL file stems (excluding aggregated_results)."""
    step_dir = Path(checkpoints_dir) / run_name / "exports" / "dumped_evals" / step
    if not step_dir.is_dir():
        return []
    datasets = []
    for f in sorted(step_dir.iterdir()):
        if f.suffix == ".jsonl" and f.stem != "aggregated_results":
            datasets.append(f.stem)
    return datasets


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(checkpoints_dir: str, run_name: str, step: str, dataset: str) -> list[dict]:
    """Load all records from a single dataset JSONL file."""
    path = Path(checkpoints_dir) / run_name / "exports" / "dumped_evals" / step / f"{dataset}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_aggregated(checkpoints_dir: str, run_name: str, step: str) -> dict:
    """Load aggregated_results.jsonl (single JSON object) for a step."""
    path = Path(checkpoints_dir) / run_name / "exports" / "dumped_evals" / step / "aggregated_results.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def record_is_correct(record: dict) -> bool:
    """Check whether a record's score indicates a correct answer."""
    score = record.get("score", [])
    if isinstance(score, list):
        return any(s > 0 for s in score)
    return score > 0


def get_score_value(record: dict) -> float:
    """Get the scalar score value from a record."""
    score = record.get("score", [])
    if isinstance(score, list):
        return max(score) if score else 0.0
    return float(score)


def get_question(record: dict) -> str:
    return record.get("env_extras", {}).get("extra_info", {}).get("question", "N/A")


def get_targets(record: dict) -> list:
    return record.get("env_extras", {}).get("reward_spec", {}).get("ground_truth", {}).get("target", [])


# ---------------------------------------------------------------------------
# Trajectory parsing & HTML rendering
# ---------------------------------------------------------------------------

TAG_PATTERN = re.compile(
    r"<(think|search|information|answer)>(.*?)</\1>",
    re.DOTALL,
)


def parse_trajectory(response: str) -> list[dict]:
    """Parse the output_response into a list of segments.

    Each segment is {type: "think"|"search"|"information"|"answer"|"text", content: str}.
    """
    segments: list[dict] = []
    last_end = 0
    for m in TAG_PATTERN.finditer(response):
        # Any text between tags
        if m.start() > last_end:
            text = response[last_end:m.start()].strip()
            if text:
                segments.append({"type": "text", "content": text})
        segments.append({"type": m.group(1), "content": m.group(2).strip()})
        last_end = m.end()
    # Trailing text
    if last_end < len(response):
        text = response[last_end:].strip()
        # Remove trailing special tokens
        text = re.sub(r"<\|im_end\|>.*", "", text).strip()
        if text:
            segments.append({"type": "text", "content": text})
    return segments


def render_trajectory_html(record: dict, label: str = "") -> str:
    """Render a single trajectory record as styled HTML."""
    question = get_question(record)
    targets = get_targets(record)
    correct = record_is_correct(record)
    score_val = get_score_value(record)
    stop_reason = record.get("stop_reason", "N/A")
    data_source = record.get("data_source", "N/A")
    response = record.get("output_response", "")

    segments = parse_trajectory(response)

    # Build HTML
    parts: list[str] = []
    parts.append(CSS)

    # Header
    correctness_class = "correct" if correct else "incorrect"
    correctness_label = "Correct" if correct else "Incorrect"
    label_html = f"<div class='tv-step-label'>{html.escape(label)}</div>" if label else ""
    parts.append(f"""
    <div class="tv-header">
        {label_html}
        <div class="tv-question"><strong>Question:</strong> {html.escape(question)}</div>
        <div class="tv-meta">
            <span class="tv-badge tv-badge-{correctness_class}">{correctness_label} (score: {score_val:.2f})</span>
            <span class="tv-badge tv-badge-neutral">Stop: {html.escape(stop_reason)}</span>
            <span class="tv-badge tv-badge-neutral">Dataset: {html.escape(data_source)}</span>
        </div>
        <div class="tv-ground-truth"><strong>Ground Truth:</strong> {html.escape(', '.join(str(t) for t in targets))}</div>
    </div>
    """)

    # Trajectory segments
    parts.append('<div class="tv-trajectory">')

    turn_num = 0
    for seg in segments:
        content_escaped = html.escape(seg["content"])
        if seg["type"] == "think":
            turn_num += 1
            parts.append(f"""
            <details class="tv-segment tv-think">
                <summary>Thinking (Turn {turn_num})</summary>
                <pre>{content_escaped}</pre>
            </details>
            """)
        elif seg["type"] == "search":
            parts.append(f"""
            <div class="tv-segment tv-search">
                <div class="tv-segment-label">Search Query</div>
                <div class="tv-segment-content">{content_escaped}</div>
            </div>
            """)
        elif seg["type"] == "information":
            parts.append(f"""
            <details class="tv-segment tv-information">
                <summary>Search Results</summary>
                <pre>{content_escaped}</pre>
            </details>
            """)
        elif seg["type"] == "answer":
            answer_class = "tv-answer-correct" if correct else "tv-answer-incorrect"
            parts.append(f"""
            <div class="tv-segment tv-answer {answer_class}">
                <div class="tv-segment-label">Final Answer</div>
                <div class="tv-segment-content">{content_escaped}</div>
            </div>
            """)
        else:
            # Plain text between tags
            if content_escaped.strip():
                parts.append(f"""
                <div class="tv-segment tv-text">
                    <pre>{content_escaped}</pre>
                </div>
                """)

    parts.append("</div>")  # tv-trajectory

    return "\n".join(parts)


CSS = """
<style>
/* Force dark text everywhere to avoid invisible text in dark Gradio themes */
.tv-header, .tv-trajectory, .tv-segment, .tv-segment pre,
.tv-segment-content, .tv-segment-label, .tv-question, .tv-ground-truth,
.tv-header strong, .tv-step-label {
    color: #212529 !important;
}
.tv-header {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}
.tv-step-label {
    font-size: 1.2em;
    font-weight: 700;
    color: #0d6efd !important;
    margin-bottom: 8px;
}
.tv-question {
    font-size: 1.1em;
    margin-bottom: 8px;
}
.tv-ground-truth {
    margin-top: 8px;
    color: #495057 !important;
}
.tv-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.tv-badge {
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
}
.tv-badge-correct { background: #d4edda; color: #155724 !important; }
.tv-badge-incorrect { background: #f8d7da; color: #721c24 !important; }
.tv-badge-neutral { background: #e2e3e5; color: #383d41 !important; }

.tv-trajectory {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.tv-segment {
    border-radius: 6px;
    padding: 10px 14px;
    border-left: 4px solid;
}
.tv-segment-label {
    font-weight: 600;
    font-size: 0.85em;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.tv-segment-content {
    font-size: 1em;
}
.tv-segment pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 0;
    font-size: 0.9em;
    font-family: inherit;
}

/* Think */
.tv-think {
    background: #f1f3f5;
    border-left-color: #868e96;
}
.tv-think summary {
    cursor: pointer;
    font-weight: 600;
    color: #495057 !important;
}

/* Search query */
.tv-search {
    background: #d0ebff;
    border-left-color: #228be6;
}
.tv-search .tv-segment-label { color: #1864ab !important; }

/* Information / search results */
.tv-information {
    background: #d3f9d8;
    border-left-color: #40c057;
}
.tv-information summary {
    cursor: pointer;
    font-weight: 600;
    color: #2b8a3e !important;
}
.tv-information pre {
    max-height: 300px;
    overflow-y: auto;
}

/* Answer */
.tv-answer {
    padding: 14px;
    font-size: 1.05em;
}
.tv-answer .tv-segment-label { font-size: 0.9em; }
.tv-answer-correct {
    background: #d4edda;
    border-left-color: #28a745;
}
.tv-answer-correct .tv-segment-label { color: #155724 !important; }
.tv-answer-incorrect {
    background: #f8d7da;
    border-left-color: #dc3545;
}
.tv-answer-incorrect .tv-segment-label { color: #721c24 !important; }

/* Plain text */
.tv-text {
    background: #fff;
    border-left-color: #ced4da;
    color: #6c757d !important;
}

/* Comparison layout */
.tv-compare-container {
    display: flex;
    gap: 16px;
}
.tv-compare-side {
    flex: 1;
    min-width: 0;
}

/* Metrics table */
.metrics-table { border-collapse: collapse; width: 100%; }
.metrics-table th, .metrics-table td {
    border: 1px solid #dee2e6;
    padding: 8px 12px;
    text-align: center;
    color: #212529 !important;
}
.metrics-table th { background: #f8f9fa; font-weight: 600; }
.metrics-table tr:nth-child(even) { background: #f8f9fa; }
.metrics-table td.score-high { color: #155724 !important; font-weight: 600; }
.metrics-table td.score-low { color: #721c24 !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Aggregated metrics table
# ---------------------------------------------------------------------------

def build_metrics_table(checkpoints_dir: str, run_name: str) -> str:
    """Build an HTML table of per-dataset avg_score across all steps."""
    steps = discover_steps(checkpoints_dir, run_name)
    if not steps:
        return "<p>No evaluation steps found.</p>"

    # Gather data: {dataset: {step: score}}
    all_datasets: set[str] = set()
    step_data: dict[str, dict[str, float]] = {}

    for step in steps:
        agg = load_aggregated(checkpoints_dir, run_name, step)
        step_data[step] = {}
        for key, value in agg.items():
            if "/avg_score" in key:
                # key looks like "eval/searchR1_nq/avg_score"
                parts = key.split("/")
                if len(parts) >= 3:
                    ds = parts[1]
                    all_datasets.add(ds)
                    step_data[step][ds] = value

    datasets = sorted(all_datasets)
    if not datasets:
        return "<p>No aggregated metrics found.</p>"

    rows: list[str] = []
    rows.append(CSS)
    rows.append('<table class="metrics-table">')
    rows.append("<thead><tr><th>Dataset</th>")
    for step in steps:
        rows.append(f"<th>{step_label(step)}</th>")
    rows.append("</tr></thead>")

    rows.append("<tbody>")
    for ds in datasets:
        rows.append(f"<tr><td><strong>{html.escape(ds)}</strong></td>")
        for step in steps:
            score = step_data.get(step, {}).get(ds)
            if score is not None:
                css_class = "score-high" if score >= 0.5 else "score-low"
                rows.append(f'<td class="{css_class}">{score:.4f}</td>')
            else:
                rows.append("<td>-</td>")
        rows.append("</tr>")

    # Average row
    rows.append('<tr><td><strong>Average</strong></td>')
    for step in steps:
        scores = [v for v in step_data.get(step, {}).values()]
        if scores:
            avg = sum(scores) / len(scores)
            css_class = "score-high" if avg >= 0.5 else "score-low"
            rows.append(f'<td class="{css_class}"><strong>{avg:.4f}</strong></td>')
        else:
            rows.append("<td>-</td>")
    rows.append("</tr>")

    rows.append("</tbody></table>")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_app(checkpoints_dir: str) -> gr.Blocks:
    # State: cached records for current selection (browser tab)
    cached_records: list[dict] = []
    filtered_indices: list[int] = []

    # State: cached records for comparison tab
    compare_records_a: list[dict] = []
    compare_records_b: list[dict] = []
    compare_filtered: list[int] = []

    def update_steps(run_name):
        steps = discover_steps(checkpoints_dir, run_name)
        default = steps[0] if steps else None
        return gr.update(choices=steps, value=default)

    def update_datasets(run_name, step):
        if not run_name or not step:
            return gr.update(choices=[], value=None)
        datasets = discover_datasets(checkpoints_dir, run_name, step)
        default = datasets[0] if datasets else None
        return gr.update(choices=datasets, value=default)

    # ---- Browser tab callbacks ----

    def load_and_filter(run_name, step, dataset, score_filter):
        nonlocal cached_records, filtered_indices
        if not run_name or not step or not dataset:
            cached_records = []
            filtered_indices = []
            return "0 samples", 0, gr.update(maximum=0, value=0), "<p>No data loaded.</p>"

        cached_records = load_dataset(checkpoints_dir, run_name, step, dataset)

        if score_filter == "Correct":
            filtered_indices = [i for i, r in enumerate(cached_records) if record_is_correct(r)]
        elif score_filter == "Incorrect":
            filtered_indices = [i for i, r in enumerate(cached_records) if not record_is_correct(r)]
        else:
            filtered_indices = list(range(len(cached_records)))

        total = len(filtered_indices)
        if total == 0:
            return f"0 / {len(cached_records)} samples", 0, gr.update(maximum=0, value=0), "<p>No matching samples.</p>"

        record = cached_records[filtered_indices[0]]
        html_content = render_trajectory_html(record)
        return (
            f"{total} / {len(cached_records)} samples",
            0,
            gr.update(maximum=total - 1, value=0),
            html_content,
        )

    def navigate(index):
        if not filtered_indices:
            return "<p>No data loaded.</p>"
        idx = max(0, min(int(index), len(filtered_indices) - 1))
        record = cached_records[filtered_indices[idx]]
        return render_trajectory_html(record)

    def go_prev(current_idx):
        new_idx = max(0, int(current_idx) - 1)
        return new_idx, navigate(new_idx)

    def go_next(current_idx):
        max_idx = len(filtered_indices) - 1 if filtered_indices else 0
        new_idx = min(max_idx, int(current_idx) + 1)
        return new_idx, navigate(new_idx)

    # ---- Compare tab callbacks ----

    def compare_update_steps(run_name):
        steps = discover_steps(checkpoints_dir, run_name)
        default_a = steps[0] if len(steps) > 0 else None
        default_b = steps[-1] if len(steps) > 0 else None
        ds_choices = []
        if run_name and default_a:
            ds_choices = discover_datasets(checkpoints_dir, run_name, default_a)
        ds_default = ds_choices[0] if ds_choices else None
        return (
            gr.update(choices=steps, value=default_a),
            gr.update(choices=steps, value=default_b),
            gr.update(choices=ds_choices, value=ds_default),
        )

    def compare_update_datasets(run_name, step_a):
        if not run_name or not step_a:
            return gr.update(choices=[], value=None)
        datasets = discover_datasets(checkpoints_dir, run_name, step_a)
        default = datasets[0] if datasets else None
        return gr.update(choices=datasets, value=default)

    def compare_load(run_name, step_a, step_b, dataset, score_filter):
        nonlocal compare_records_a, compare_records_b, compare_filtered
        if not all([run_name, step_a, step_b, dataset]):
            compare_records_a = []
            compare_records_b = []
            compare_filtered = []
            return "0 samples", 0, gr.update(maximum=0, value=0), "<p>Select all fields.</p>"

        compare_records_a = load_dataset(checkpoints_dir, run_name, step_a, dataset)
        compare_records_b = load_dataset(checkpoints_dir, run_name, step_b, dataset)

        # Use the shorter list length to stay in bounds
        n = min(len(compare_records_a), len(compare_records_b))

        if score_filter == "Correct in both":
            compare_filtered = [i for i in range(n) if record_is_correct(compare_records_a[i]) and record_is_correct(compare_records_b[i])]
        elif score_filter == "Incorrect in both":
            compare_filtered = [i for i in range(n) if not record_is_correct(compare_records_a[i]) and not record_is_correct(compare_records_b[i])]
        elif score_filter == "Fixed (A wrong, B correct)":
            compare_filtered = [i for i in range(n) if not record_is_correct(compare_records_a[i]) and record_is_correct(compare_records_b[i])]
        elif score_filter == "Regressed (A correct, B wrong)":
            compare_filtered = [i for i in range(n) if record_is_correct(compare_records_a[i]) and not record_is_correct(compare_records_b[i])]
        else:
            compare_filtered = list(range(n))

        total = len(compare_filtered)
        if total == 0:
            return f"0 / {n} samples", 0, gr.update(maximum=0, value=0), "<p>No matching samples.</p>"

        html_content = render_comparison(compare_records_a, compare_records_b, compare_filtered, 0, step_a, step_b)
        return (
            f"{total} / {n} samples",
            0,
            gr.update(maximum=total - 1, value=0),
            html_content,
        )

    def render_comparison(recs_a, recs_b, filtered, idx, step_a, step_b):
        idx = max(0, min(int(idx), len(filtered) - 1))
        i = filtered[idx]
        html_a = render_trajectory_html(recs_a[i], label=step_label(step_a))
        html_b = render_trajectory_html(recs_b[i], label=step_label(step_b))
        return f'{CSS}<div class="tv-compare-container"><div class="tv-compare-side">{html_a}</div><div class="tv-compare-side">{html_b}</div></div>'

    def compare_navigate(index, step_a, step_b):
        if not compare_filtered:
            return "<p>No data loaded.</p>"
        return render_comparison(compare_records_a, compare_records_b, compare_filtered, index, step_a, step_b)

    def compare_prev(current_idx, step_a, step_b):
        new_idx = max(0, int(current_idx) - 1)
        return new_idx, compare_navigate(new_idx, step_a, step_b)

    def compare_next(current_idx, step_a, step_b):
        max_idx = len(compare_filtered) - 1 if compare_filtered else 0
        new_idx = min(max_idx, int(current_idx) + 1)
        return new_idx, compare_navigate(new_idx, step_a, step_b)

    # ---- Metrics tab callbacks ----

    def update_metrics(run_name):
        if not run_name:
            return "<p>Select a run.</p>"
        return build_metrics_table(checkpoints_dir, run_name)

    # --- Build UI ---
    runs = discover_runs(checkpoints_dir)
    default_run = runs[0] if runs else None
    default_steps = discover_steps(checkpoints_dir, default_run) if default_run else []
    default_step = default_steps[0] if default_steps else None
    default_step_last = default_steps[-1] if default_steps else None
    default_datasets = discover_datasets(checkpoints_dir, default_run, default_step) if default_run and default_step else []
    default_dataset = default_datasets[0] if default_datasets else None

    with gr.Blocks(title="Citation Prediction Trajectory Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Citation Prediction Eval Trajectory Viewer")

        with gr.Tabs():
            # ---- Tab 1: Trajectory Browser ----
            with gr.Tab("Browse"):
                with gr.Row():
                    run_dd = gr.Dropdown(choices=runs, value=default_run, label="Run", scale=2)
                    step_dd = gr.Dropdown(choices=default_steps, value=default_step, label="Eval Step", scale=1)
                    dataset_dd = gr.Dropdown(choices=default_datasets, value=default_dataset, label="Dataset", scale=1)
                    score_dd = gr.Dropdown(choices=["All", "Correct", "Incorrect"], value="All", label="Score Filter", scale=1)

                with gr.Row():
                    sample_count = gr.Textbox(label="Matching", interactive=False, scale=1)
                    prev_btn = gr.Button("< Prev", scale=1)
                    idx_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Sample Index", scale=3)
                    next_btn = gr.Button("Next >", scale=1)

                current_idx = gr.State(0)
                trajectory_html = gr.HTML(label="Trajectory")

                # Wiring
                run_dd.change(fn=update_steps, inputs=[run_dd], outputs=[step_dd])
                step_dd.change(fn=update_datasets, inputs=[run_dd, step_dd], outputs=[dataset_dd])
                dataset_dd.change(fn=load_and_filter, inputs=[run_dd, step_dd, dataset_dd, score_dd], outputs=[sample_count, current_idx, idx_slider, trajectory_html])
                score_dd.change(fn=load_and_filter, inputs=[run_dd, step_dd, dataset_dd, score_dd], outputs=[sample_count, current_idx, idx_slider, trajectory_html])
                prev_btn.click(fn=go_prev, inputs=[current_idx], outputs=[current_idx, trajectory_html])
                next_btn.click(fn=go_next, inputs=[current_idx], outputs=[current_idx, trajectory_html])
                idx_slider.release(fn=lambda idx: (idx, navigate(idx)), inputs=[idx_slider], outputs=[current_idx, trajectory_html])
                app.load(fn=load_and_filter, inputs=[run_dd, step_dd, dataset_dd, score_dd], outputs=[sample_count, current_idx, idx_slider, trajectory_html])

            # ---- Tab 2: Compare Steps ----
            with gr.Tab("Compare Steps"):
                with gr.Row():
                    cmp_run_dd = gr.Dropdown(choices=runs, value=default_run, label="Run", scale=2)
                    cmp_step_a = gr.Dropdown(choices=default_steps, value=default_step, label="Step A (left)", scale=1)
                    cmp_step_b = gr.Dropdown(choices=default_steps, value=default_step_last, label="Step B (right)", scale=1)
                with gr.Row():
                    cmp_dataset_dd = gr.Dropdown(choices=default_datasets, value=default_dataset, label="Dataset", scale=2)
                    cmp_score_dd = gr.Dropdown(
                        choices=["All", "Correct in both", "Incorrect in both", "Fixed (A wrong, B correct)", "Regressed (A correct, B wrong)"],
                        value="All",
                        label="Score Filter",
                        scale=2,
                    )

                with gr.Row():
                    cmp_count = gr.Textbox(label="Matching", interactive=False, scale=1)
                    cmp_prev_btn = gr.Button("< Prev", scale=1)
                    cmp_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Sample Index", scale=3)
                    cmp_next_btn = gr.Button("Next >", scale=1)

                cmp_idx = gr.State(0)
                cmp_html = gr.HTML(label="Comparison")

                # Wiring
                cmp_run_dd.change(fn=compare_update_steps, inputs=[cmp_run_dd], outputs=[cmp_step_a, cmp_step_b, cmp_dataset_dd])
                cmp_step_a.change(fn=compare_update_datasets, inputs=[cmp_run_dd, cmp_step_a], outputs=[cmp_dataset_dd])
                for trigger in [cmp_step_a, cmp_step_b, cmp_dataset_dd, cmp_score_dd]:
                    trigger.change(fn=compare_load, inputs=[cmp_run_dd, cmp_step_a, cmp_step_b, cmp_dataset_dd, cmp_score_dd], outputs=[cmp_count, cmp_idx, cmp_slider, cmp_html])
                cmp_prev_btn.click(fn=compare_prev, inputs=[cmp_idx, cmp_step_a, cmp_step_b], outputs=[cmp_idx, cmp_html])
                cmp_next_btn.click(fn=compare_next, inputs=[cmp_idx, cmp_step_a, cmp_step_b], outputs=[cmp_idx, cmp_html])
                cmp_slider.release(fn=lambda idx, sa, sb: (idx, compare_navigate(idx, sa, sb)), inputs=[cmp_slider, cmp_step_a, cmp_step_b], outputs=[cmp_idx, cmp_html])

            # ---- Tab 3: Aggregated Metrics ----
            with gr.Tab("Aggregated Metrics"):
                metrics_run_dd = gr.Dropdown(choices=runs, value=default_run, label="Run")
                metrics_html = gr.HTML(label="Metrics Table")

                metrics_run_dd.change(fn=update_metrics, inputs=[metrics_run_dd], outputs=[metrics_html])
                app.load(fn=update_metrics, inputs=[metrics_run_dd], outputs=[metrics_html])

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Citation Prediction Trajectory Viewer")
    parser.add_argument(
        "--checkpoints-dir",
        default=DEFAULT_CHECKPOINTS_DIR,
        help="Root directory containing run checkpoints",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    app = create_app(args.checkpoints_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
