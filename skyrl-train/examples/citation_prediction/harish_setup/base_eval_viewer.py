"""
Gradio viewer for base-model evaluation trajectories.

Auto-discovers eval runs from the logs directory, parses SLURM output
to label each run (prompt style + embedding model), and provides:
  - Browse: view individual trajectories with formatted tags
  - Compare: side-by-side comparison of two setups
  - Summary: accuracy/score table across all setups

Supports live updates — click Refresh to reload JSONL as evals run.

Usage:
    python base_eval_viewer.py [--evals-dir /path/to/logs] [--port 7861] [--share]
"""

import argparse
import html
import json
import re
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EVALS_DIR = "/scratch/gpfs/ZHUANGL/hk4638/logs"

# ---------------------------------------------------------------------------
# Trajectory parsing & rendering (from trajectory_viewer.py)
# ---------------------------------------------------------------------------

TAG_PATTERN = re.compile(
    r"<(think|search|information|answer)>(.*?)</\1>",
    re.DOTALL,
)

CSS = """
<style>
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
.tv-question { font-size: 1.1em; margin-bottom: 8px; }
.tv-ground-truth { margin-top: 8px; color: #495057 !important; }
.tv-meta { display: flex; gap: 8px; flex-wrap: wrap; }
.tv-badge {
    padding: 2px 10px; border-radius: 12px;
    font-size: 0.85em; font-weight: 600;
}
.tv-badge-correct { background: #d4edda; color: #155724 !important; }
.tv-badge-incorrect { background: #f8d7da; color: #721c24 !important; }
.tv-badge-neutral { background: #e2e3e5; color: #383d41 !important; }

.tv-trajectory { display: flex; flex-direction: column; gap: 8px; }
.tv-segment { border-radius: 6px; padding: 10px 14px; border-left: 4px solid; }
.tv-segment-label {
    font-weight: 600; font-size: 0.85em; margin-bottom: 4px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.tv-segment-content { font-size: 1em; }
.tv-segment pre {
    white-space: pre-wrap; word-wrap: break-word;
    margin: 0; font-size: 0.9em; font-family: inherit;
}

.tv-think { background: #f1f3f5; border-left-color: #868e96; }
.tv-think summary { cursor: pointer; font-weight: 600; color: #495057 !important; }
.tv-search { background: #d0ebff; border-left-color: #228be6; }
.tv-search .tv-segment-label { color: #1864ab !important; }
.tv-information { background: #d3f9d8; border-left-color: #40c057; }
.tv-information summary { cursor: pointer; font-weight: 600; color: #2b8a3e !important; }
.tv-information pre { max-height: 300px; overflow-y: auto; }
.tv-answer { padding: 14px; font-size: 1.05em; }
.tv-answer .tv-segment-label { font-size: 0.9em; }
.tv-answer-correct { background: #d4edda; border-left-color: #28a745; }
.tv-answer-correct .tv-segment-label { color: #155724 !important; }
.tv-answer-incorrect { background: #f8d7da; border-left-color: #dc3545; }
.tv-answer-incorrect .tv-segment-label { color: #721c24 !important; }
.tv-text { background: #fff; border-left-color: #ced4da; color: #6c757d !important; }

.tv-compare-container { display: flex; gap: 16px; }
.tv-compare-side { flex: 1; min-width: 0; }

.metrics-table { border-collapse: collapse; width: 100%; }
.metrics-table th, .metrics-table td {
    border: 1px solid #dee2e6; padding: 8px 12px;
    text-align: center; color: #212529 !important;
}
.metrics-table th { background: #f8f9fa; font-weight: 600; }
.metrics-table tr:nth-child(even) { background: #f8f9fa; }
.metrics-table td.score-high { color: #155724 !important; font-weight: 600; }
.metrics-table td.score-low { color: #721c24 !important; }
</style>
"""


def parse_trajectory(response: str) -> list[dict]:
    """Parse output_response into segments: think/search/information/answer/text."""
    segments: list[dict] = []
    last_end = 0
    for m in TAG_PATTERN.finditer(response):
        if m.start() > last_end:
            text = response[last_end : m.start()].strip()
            if text:
                segments.append({"type": "text", "content": text})
        segments.append({"type": m.group(1), "content": m.group(2).strip()})
        last_end = m.end()
    if last_end < len(response):
        text = response[last_end:].strip()
        text = re.sub(r"<\|im_end\|>.*", "", text).strip()
        if text:
            segments.append({"type": "text", "content": text})
    return segments


def record_is_correct(record: dict) -> bool:
    score = record.get("score", 0)
    if isinstance(score, list):
        return any(s > 0 for s in score)
    return score > 0


def get_score_value(record: dict) -> float:
    score = record.get("score", 0)
    if isinstance(score, list):
        return max(score) if score else 0.0
    return float(score)


def get_question(record: dict) -> str:
    extras = record.get("env_extras", {})
    extra_info = extras.get("extra_info", {})
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    return extra_info.get("question", "N/A")


def get_targets(record: dict) -> list:
    extras = record.get("env_extras", {})
    reward_spec = extras.get("reward_spec", {})
    if isinstance(reward_spec, str):
        reward_spec = json.loads(reward_spec)
    target = reward_spec.get("ground_truth", {}).get("target", [])
    if isinstance(target, str):
        return [target]
    return target


def render_trajectory_html(record: dict, label: str = "") -> str:
    """Render a single trajectory record as styled HTML."""
    question = get_question(record)
    targets = get_targets(record)
    correct = record_is_correct(record)
    score_val = get_score_value(record)
    stop_reason = record.get("stop_reason", "N/A")
    response = record.get("output_response", "")

    segments = parse_trajectory(response)

    parts: list[str] = [CSS]

    correctness_class = "correct" if correct else "incorrect"
    correctness_label = "Correct" if correct else "Incorrect"
    label_html = f"<div class='tv-step-label'>{html.escape(str(label))}</div>" if label else ""
    parts.append(f"""
    <div class="tv-header">
        {label_html}
        <div class="tv-question"><strong>Question:</strong> {html.escape(question)}</div>
        <div class="tv-meta">
            <span class="tv-badge tv-badge-{correctness_class}">{correctness_label} (score: {score_val:.2f})</span>
            <span class="tv-badge tv-badge-neutral">Stop: {html.escape(str(stop_reason))}</span>
        </div>
        <div class="tv-ground-truth"><strong>Ground Truth:</strong> {html.escape(', '.join(str(t) for t in targets))}</div>
    </div>
    """)

    parts.append('<div class="tv-trajectory">')
    turn_num = 0
    for seg in segments:
        content_escaped = html.escape(seg["content"])
        if seg["type"] == "think":
            turn_num += 1
            parts.append(f"""
            <details class="tv-segment tv-think" open>
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
            if content_escaped.strip():
                parts.append(f"""
                <div class="tv-segment tv-text">
                    <pre>{content_escaped}</pre>
                </div>
                """)

    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_eval_runs(evals_dir: str) -> dict[str, Path]:
    """Auto-discover eval runs. Returns {label: jsonl_dir_path}."""
    base = Path(evals_dir)
    if not base.is_dir():
        return {}

    runs: dict[str, Path] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("eval_"):
            continue
        eval_only_dir = entry / "dumped_evals" / "eval_only"
        if not eval_only_dir.is_dir():
            continue
        # Check for JSONL files
        jsonl_files = list(eval_only_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue

        # Try to get label from SLURM .out file
        job_id = entry.name.replace("eval_", "")
        label = _parse_slurm_label(base, job_id)
        if not label:
            label = entry.name

        runs[label] = eval_only_dir

    return runs


def _parse_slurm_label(logs_dir: Path, job_id: str) -> str | None:
    """Parse SLURM .out file to extract prompt/embed/split config."""
    out_file = logs_dir / f"eval-citation_{job_id}.out"
    if not out_file.exists():
        # Try other patterns
        for pattern in [f"eval-citation_{job_id}.out", f"*_{job_id}.out"]:
            matches = list(logs_dir.glob(pattern))
            if matches:
                out_file = matches[0]
                break
        else:
            return None

    try:
        with open(out_file) as f:
            for line in f:
                m = re.search(
                    r"Config:\s*prompt=(\w+),\s*embed=(\w+),\s*split=(\w+)", line
                )
                if m:
                    prompt, embed, split = m.group(1), m.group(2), m.group(3)
                    return f"{prompt} + {embed} ({split})"
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(jsonl_dir: Path) -> list[dict]:
    """Load all records from JSONL files in a directory."""
    records = []
    for f in sorted(jsonl_dir.glob("*.jsonl")):
        if f.stem == "aggregated_results":
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_aggregated(jsonl_dir: Path) -> dict:
    """Load aggregated_results.jsonl."""
    path = jsonl_dir / "aggregated_results.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_app(evals_dir: str) -> gr.Blocks:
    # Mutable state
    discovered: dict[str, Path] = {}
    cached_records: dict[str, list[dict]] = {}
    browse_filtered: list[int] = []
    compare_filtered: list[int] = []

    def refresh_discovery():
        """Re-scan evals directory."""
        nonlocal discovered, cached_records
        discovered.clear()
        cached_records.clear()
        discovered.update(discover_eval_runs(evals_dir))
        for label, path in discovered.items():
            cached_records[label] = load_records(path)
        return list(discovered.keys())

    def get_records(label: str) -> list[dict]:
        if label not in cached_records:
            if label in discovered:
                cached_records[label] = load_records(discovered[label])
            else:
                return []
        return cached_records[label]

    # ---- Browse callbacks ----

    def on_refresh_browse():
        setups = refresh_discovery()
        if not setups:
            return (
                gr.update(choices=[], value=None),
                "No eval runs found",
                gr.update(maximum=0, value=0),
                "<p>No eval runs found. Click Refresh after evals start producing output.</p>",
            )
        default = setups[0]
        return (
            gr.update(choices=setups, value=default),
            *_load_browse(default, "All"),
        )

    def _load_browse(setup, score_filter):
        nonlocal browse_filtered
        if not setup:
            browse_filtered = []
            return "0 samples", gr.update(maximum=0, value=0), "<p>Select a setup.</p>"

        records = get_records(setup)
        if not records:
            browse_filtered = []
            return "0 samples", gr.update(maximum=0, value=0), "<p>No records yet. Click Refresh.</p>"

        if score_filter == "Correct":
            browse_filtered = [i for i, r in enumerate(records) if record_is_correct(r)]
        elif score_filter == "Incorrect":
            browse_filtered = [i for i, r in enumerate(records) if not record_is_correct(r)]
        else:
            browse_filtered = list(range(len(records)))

        n_correct = sum(1 for r in records if record_is_correct(r))
        accuracy = n_correct / len(records) * 100 if records else 0
        total = len(browse_filtered)

        stats = f"{total} / {len(records)} samples | Accuracy: {accuracy:.1f}%"
        if total == 0:
            return stats, gr.update(maximum=0, value=0), "<p>No matching samples.</p>"

        record = records[browse_filtered[0]]
        return stats, gr.update(maximum=total - 1, value=0), render_trajectory_html(record)

    def on_browse_change(setup, score_filter):
        return _load_browse(setup, score_filter)

    def on_browse_navigate(idx, setup):
        if not browse_filtered or not setup:
            return "<p>No data.</p>"
        records = get_records(setup)
        idx = max(0, min(int(idx), len(browse_filtered) - 1))
        return render_trajectory_html(records[browse_filtered[idx]])

    def on_browse_prev(idx, setup):
        new_idx = max(0, int(idx) - 1)
        return new_idx, on_browse_navigate(new_idx, setup)

    def on_browse_next(idx, setup):
        max_idx = len(browse_filtered) - 1 if browse_filtered else 0
        new_idx = min(max_idx, int(idx) + 1)
        return new_idx, on_browse_navigate(new_idx, setup)

    # ---- Compare callbacks ----

    def on_refresh_compare():
        setups = refresh_discovery()
        if len(setups) < 2:
            return (
                gr.update(choices=setups, value=setups[0] if setups else None),
                gr.update(choices=setups, value=setups[0] if setups else None),
                "Need at least 2 setups",
                gr.update(maximum=0, value=0),
                "<p>Need at least 2 eval setups to compare.</p>",
            )
        return (
            gr.update(choices=setups, value=setups[0]),
            gr.update(choices=setups, value=setups[1] if len(setups) > 1 else setups[0]),
            *_load_compare(setups[0], setups[1] if len(setups) > 1 else setups[0], "All"),
        )

    def _load_compare(setup_a, setup_b, score_filter):
        nonlocal compare_filtered
        if not setup_a or not setup_b:
            compare_filtered = []
            return "0 samples", gr.update(maximum=0, value=0), "<p>Select both setups.</p>"

        recs_a = get_records(setup_a)
        recs_b = get_records(setup_b)
        n = min(len(recs_a), len(recs_b))
        if n == 0:
            compare_filtered = []
            return "0 samples", gr.update(maximum=0, value=0), "<p>No records yet.</p>"

        if score_filter == "Correct in both":
            compare_filtered = [i for i in range(n) if record_is_correct(recs_a[i]) and record_is_correct(recs_b[i])]
        elif score_filter == "Incorrect in both":
            compare_filtered = [i for i in range(n) if not record_is_correct(recs_a[i]) and not record_is_correct(recs_b[i])]
        elif score_filter == "A correct, B wrong":
            compare_filtered = [i for i in range(n) if record_is_correct(recs_a[i]) and not record_is_correct(recs_b[i])]
        elif score_filter == "A wrong, B correct":
            compare_filtered = [i for i in range(n) if not record_is_correct(recs_a[i]) and record_is_correct(recs_b[i])]
        else:
            compare_filtered = list(range(n))

        total = len(compare_filtered)
        stats = f"{total} / {n} samples"
        if total == 0:
            return stats, gr.update(maximum=0, value=0), "<p>No matching samples.</p>"

        html_content = _render_compare_pair(recs_a, recs_b, compare_filtered[0], setup_a, setup_b)
        return stats, gr.update(maximum=total - 1, value=0), html_content

    def _render_compare_pair(recs_a, recs_b, raw_idx, label_a, label_b):
        html_a = render_trajectory_html(recs_a[raw_idx], label=label_a)
        html_b = render_trajectory_html(recs_b[raw_idx], label=label_b)
        return f'{CSS}<div class="tv-compare-container"><div class="tv-compare-side">{html_a}</div><div class="tv-compare-side">{html_b}</div></div>'

    def on_compare_change(setup_a, setup_b, score_filter):
        return _load_compare(setup_a, setup_b, score_filter)

    def on_compare_navigate(idx, setup_a, setup_b):
        if not compare_filtered or not setup_a or not setup_b:
            return "<p>No data.</p>"
        idx = max(0, min(int(idx), len(compare_filtered) - 1))
        recs_a = get_records(setup_a)
        recs_b = get_records(setup_b)
        return _render_compare_pair(recs_a, recs_b, compare_filtered[idx], setup_a, setup_b)

    def on_compare_prev(idx, setup_a, setup_b):
        new_idx = max(0, int(idx) - 1)
        return new_idx, on_compare_navigate(new_idx, setup_a, setup_b)

    def on_compare_next(idx, setup_a, setup_b):
        max_idx = len(compare_filtered) - 1 if compare_filtered else 0
        new_idx = min(max_idx, int(idx) + 1)
        return new_idx, on_compare_navigate(new_idx, setup_a, setup_b)

    # ---- Summary callback ----

    def build_summary():
        refresh_discovery()
        if not discovered:
            return "<p>No eval runs found. Click Refresh after evals start.</p>"

        rows: list[str] = [CSS, '<table class="metrics-table">']
        rows.append("<thead><tr><th>Setup</th><th># Samples</th><th>Accuracy (%)</th><th>Avg Score</th></tr></thead>")
        rows.append("<tbody>")

        for label in sorted(discovered.keys()):
            records = get_records(label)
            n = len(records)
            if n == 0:
                rows.append(f"<tr><td><strong>{html.escape(label)}</strong></td><td>0</td><td>-</td><td>-</td></tr>")
                continue

            n_correct = sum(1 for r in records if record_is_correct(r))
            accuracy = n_correct / n * 100
            avg_score = sum(get_score_value(r) for r in records) / n

            acc_class = "score-high" if accuracy >= 50 else "score-low"
            score_class = "score-high" if avg_score >= 0.5 else "score-low"
            rows.append(
                f'<tr><td><strong>{html.escape(label)}</strong></td>'
                f"<td>{n}</td>"
                f'<td class="{acc_class}">{accuracy:.1f}</td>'
                f'<td class="{score_class}">{avg_score:.4f}</td></tr>'
            )

        rows.append("</tbody></table>")
        return "\n".join(rows)

    # ---- Initial discovery ----
    initial_setups = refresh_discovery()
    default_setup = initial_setups[0] if initial_setups else None

    # ---- Build UI ----
    with gr.Blocks(title="Base Model Eval Viewer") as app:
        gr.Markdown("# Base Model Eval Viewer — Citation Prediction")
        gr.Markdown("Browse and compare base model evaluation trajectories across prompt styles and embedding models.")

        with gr.Tabs():
            # ---- Tab 1: Browse ----
            with gr.Tab("Browse"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="secondary", scale=1)
                    setup_dd = gr.Dropdown(choices=initial_setups, value=default_setup, label="Setup", scale=2)
                    score_dd = gr.Dropdown(choices=["All", "Correct", "Incorrect"], value="All", label="Filter", scale=1)

                with gr.Row():
                    stats_box = gr.Textbox(label="Stats", interactive=False, scale=1)
                    prev_btn = gr.Button("< Prev", scale=1)
                    idx_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Sample", scale=3)
                    next_btn = gr.Button("Next >", scale=1)

                browse_idx = gr.State(0)
                browse_html = gr.HTML(label="Trajectory")

                # Wire browse
                browse_outputs = [stats_box, idx_slider, browse_html]

                refresh_btn.click(
                    fn=on_refresh_browse,
                    outputs=[setup_dd, *browse_outputs],
                )
                setup_dd.change(fn=on_browse_change, inputs=[setup_dd, score_dd], outputs=browse_outputs)
                score_dd.change(fn=on_browse_change, inputs=[setup_dd, score_dd], outputs=browse_outputs)
                prev_btn.click(fn=on_browse_prev, inputs=[browse_idx, setup_dd], outputs=[browse_idx, browse_html])
                next_btn.click(fn=on_browse_next, inputs=[browse_idx, setup_dd], outputs=[browse_idx, browse_html])
                idx_slider.release(
                    fn=lambda idx, setup: (idx, on_browse_navigate(idx, setup)),
                    inputs=[idx_slider, setup_dd],
                    outputs=[browse_idx, browse_html],
                )

                # Initial load
                app.load(fn=on_browse_change, inputs=[setup_dd, score_dd], outputs=browse_outputs)

            # ---- Tab 2: Compare ----
            with gr.Tab("Compare Setups"):
                with gr.Row():
                    cmp_refresh = gr.Button("Refresh", variant="secondary", scale=1)
                    cmp_setup_a = gr.Dropdown(
                        choices=initial_setups,
                        value=initial_setups[0] if initial_setups else None,
                        label="Setup A (left)",
                        scale=2,
                    )
                    cmp_setup_b = gr.Dropdown(
                        choices=initial_setups,
                        value=initial_setups[1] if len(initial_setups) > 1 else (initial_setups[0] if initial_setups else None),
                        label="Setup B (right)",
                        scale=2,
                    )
                with gr.Row():
                    cmp_score_dd = gr.Dropdown(
                        choices=["All", "Correct in both", "Incorrect in both", "A correct, B wrong", "A wrong, B correct"],
                        value="All",
                        label="Filter",
                        scale=2,
                    )

                with gr.Row():
                    cmp_stats = gr.Textbox(label="Stats", interactive=False, scale=1)
                    cmp_prev = gr.Button("< Prev", scale=1)
                    cmp_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Sample", scale=3)
                    cmp_next = gr.Button("Next >", scale=1)

                cmp_idx = gr.State(0)
                cmp_html = gr.HTML(label="Comparison")

                cmp_outputs = [cmp_stats, cmp_slider, cmp_html]

                cmp_refresh.click(
                    fn=on_refresh_compare,
                    outputs=[cmp_setup_a, cmp_setup_b, *cmp_outputs],
                )
                for trigger in [cmp_setup_a, cmp_setup_b, cmp_score_dd]:
                    trigger.change(
                        fn=on_compare_change,
                        inputs=[cmp_setup_a, cmp_setup_b, cmp_score_dd],
                        outputs=cmp_outputs,
                    )
                cmp_prev.click(
                    fn=on_compare_prev,
                    inputs=[cmp_idx, cmp_setup_a, cmp_setup_b],
                    outputs=[cmp_idx, cmp_html],
                )
                cmp_next.click(
                    fn=on_compare_next,
                    inputs=[cmp_idx, cmp_setup_a, cmp_setup_b],
                    outputs=[cmp_idx, cmp_html],
                )
                cmp_slider.release(
                    fn=lambda idx, a, b: (idx, on_compare_navigate(idx, a, b)),
                    inputs=[cmp_slider, cmp_setup_a, cmp_setup_b],
                    outputs=[cmp_idx, cmp_html],
                )

            # ---- Tab 3: Summary ----
            with gr.Tab("Summary"):
                summary_refresh = gr.Button("Refresh", variant="secondary")
                summary_html = gr.HTML(label="Summary Table")

                summary_refresh.click(fn=build_summary, outputs=[summary_html])
                app.load(fn=build_summary, outputs=[summary_html])

    return app


def main():
    parser = argparse.ArgumentParser(description="Base Model Eval Viewer")
    parser.add_argument(
        "--evals-dir",
        default=DEFAULT_EVALS_DIR,
        help="Directory containing eval_* output directories",
    )
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app = create_app(args.evals_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
