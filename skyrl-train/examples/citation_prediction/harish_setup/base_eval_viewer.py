"""
Gradio viewer for base-model evaluation trajectories and sweep results.

Reads SkyRL's dumped_evals JSONL format from sweep runs. Provides:
  - Sweep Results: heatmap of pass@k across (m, n) hyperparameters
  - Pass@k Curves: plot pass@k vs k for selected setups
  - Browse: view individual trajectories with hyperparameter filtering
  - Summary: accuracy/score table across all setups

Usage:
    python base_eval_viewer.py [--evals-dir /path/to/logs] [--port 7861] [--share]
"""

import argparse
import html
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import gradio as gr

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EVALS_DIR = "/scratch/gpfs/ZHUANGL/hk4638/logs"

# Standard k values for pass@k computation
K_VALUES = [1, 2, 3, 5, 10, 15, 20]

# ---------------------------------------------------------------------------
# Pass@k computation
# ---------------------------------------------------------------------------

def _pass_at_k_single(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator for a single prompt.

    n: total samples, c: number correct, k: the k in pass@k.
    Returns 1 - C(n-c, k) / C(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k_values(prompts: list[dict], max_k: int = 20) -> dict[int, float]:
    """Compute pass@k for every k from 1 to max_k, averaged across prompts."""
    if not prompts:
        return {}
    result = {}
    for k in range(1, max_k + 1):
        total = 0.0
        valid = 0
        for p in prompts:
            n = len(p["samples"])
            if k > n:
                continue
            c = p["num_correct"]
            total += _pass_at_k_single(n, c, k)
            valid += 1
        if valid > 0:
            result[k] = total / valid
    return result


# ---------------------------------------------------------------------------
# Trajectory parsing & rendering
# ---------------------------------------------------------------------------

ACTION_PATTERN = re.compile(
    r"<(search|information|answer)>(.*?)</\1>",
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
.tv-badge-info { background: #cce5ff; color: #004085 !important; }

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

.metrics-table { border-collapse: collapse; width: 100%; }
.metrics-table th, .metrics-table td, .metrics-table td strong {
    border: 1px solid #dee2e6; padding: 8px 12px;
    text-align: center; color: #212529 !important; background: #fff !important;
}
.metrics-table th { background: #f8f9fa !important; font-weight: 600; color: #212529 !important; }
.metrics-table tr:nth-child(even) td { background: #f8f9fa !important; }
.metrics-table td.score-high { color: #155724 !important; background: #d4edda !important; font-weight: 600; }
.metrics-table td.score-low { color: #721c24 !important; background: #f8d7da !important; }

.heatmap-table { border-collapse: collapse; width: auto; margin: 20px auto; }
.heatmap-table th, .heatmap-table td, .heatmap-table td strong {
    border: 1px solid #dee2e6; padding: 12px 20px;
    text-align: center; color: #212529 !important;
    font-size: 1.0em;
}
.heatmap-table th { background: #f8f9fa !important; font-weight: 700; color: #212529 !important; }
.heatmap-table td { font-weight: 600; min-width: 80px; color: #212529 !important; }
</style>
"""


def parse_trajectory(response: str) -> list[dict]:
    """Parse output_response into segments (think/search/information/answer/text).

    Action tags (search/information/answer) are matched first as reliable
    boundaries — they never nest inside each other.  The model sometimes
    leaves <think> open across search/information turns, so we handle think
    markers separately within the gaps between action tags.
    """
    segments: list[dict] = []
    last_end = 0

    for m in ACTION_PATTERN.finditer(response):
        if m.start() > last_end:
            _split_think_text(segments, response[last_end : m.start()])
        segments.append({"type": m.group(1), "content": m.group(2).strip()})
        last_end = m.end()

    if last_end < len(response):
        _split_think_text(segments, response[last_end:])

    return segments


def _split_think_text(segments: list[dict], text: str) -> None:
    """Split a gap between action tags on <think>/</think> markers."""
    text = re.sub(r"<\|im_end\|>.*", "", text).strip()
    if not text:
        return

    parts = re.split(r"(</?think>)", text)
    in_think = False
    for part in parts:
        if part == "<think>":
            in_think = True
            continue
        if part == "</think>":
            in_think = False
            continue
        part = part.strip()
        if not part:
            continue
        segments.append({"type": "think" if in_think else "text", "content": part})


def extract_question_from_prompt(input_prompt: str) -> str:
    """Extract the user question from a decoded SkyRL prompt."""
    # Qwen3 chat format: <|im_start|>user\n{question}<|im_end|>
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", input_prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: last 300 chars
    return input_prompt[-300:]


def extract_target(env_extras: dict) -> str:
    """Extract target arxiv ID from env_extras."""
    try:
        return env_extras.get("reward_spec", {}).get("ground_truth", {}).get("target", "unknown")
    except (AttributeError, TypeError):
        return "unknown"


def score_is_positive(score) -> bool:
    """Check if a score indicates success (handles scalar and list rewards)."""
    if isinstance(score, list):
        return any(s > 0 for s in score)
    return score > 0


def render_trajectory_html(input_prompt: str, output_response: str,
                           target: str, found: bool,
                           label: str = "", extra_badges: list = None) -> str:
    """Render a SkyRL trajectory as styled HTML."""
    question = extract_question_from_prompt(input_prompt)
    segments = parse_trajectory(output_response)

    parts: list[str] = [CSS]

    correctness_class = "correct" if found else "incorrect"
    correctness_label = "Found" if found else "Not Found"
    label_html = f"<div class='tv-step-label'>{html.escape(str(label))}</div>" if label else ""

    badge_html = f'<span class="tv-badge tv-badge-{correctness_class}">{correctness_label}</span>'
    if extra_badges:
        for badge in extra_badges:
            badge_html += f' <span class="tv-badge tv-badge-info">{html.escape(badge)}</span>'

    parts.append(f"""
    <div class="tv-header">
        {label_html}
        <div class="tv-question"><strong>Question:</strong> {html.escape(question)}</div>
        <div class="tv-meta">{badge_html}</div>
        <div class="tv-ground-truth"><strong>Target:</strong> {html.escape(str(target))}</div>
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
            answer_class = "tv-answer-correct" if found else "tv-answer-incorrect"
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
# Sweep discovery & data loading (SkyRL JSONL format)
# ---------------------------------------------------------------------------

# Directory name pattern: {prompt}_m{m}_n{n}_k{k}
SWEEP_DIR_RE = re.compile(r"^(\w+)_m(\d+)_n(\d+)_k(\d+)$")


def discover_sweep_runs(evals_dir: str) -> list[dict]:
    """Discover sweep results from the sweep/ subdirectory.

    Looks for SkyRL's dumped_evals/eval_only/ directories with JSONL files.
    Returns list of dicts with keys: label, prompt, m, n, k, path, dir_name.
    """
    sweep_dir = Path(evals_dir) / "sweep"
    if not sweep_dir.is_dir():
        return []

    runs = []
    for entry in sorted(sweep_dir.iterdir()):
        if not entry.is_dir():
            continue

        # SkyRL outputs to {export_path}/dumped_evals/eval_only/
        eval_dir = entry / "dumped_evals" / "eval_only"
        if not eval_dir.is_dir():
            continue

        # Check for JSONL data files (not just aggregated_results)
        jsonl_files = [f for f in eval_dir.glob("*.jsonl") if f.stem != "aggregated_results"]
        if not jsonl_files:
            continue

        # Parse directory name for hyperparameters
        match = SWEEP_DIR_RE.match(entry.name)
        if not match:
            continue

        prompt = match.group(1)
        m = int(match.group(2))
        n = int(match.group(3))
        k = int(match.group(4))

        label = f"{prompt} m={m} n={n} k={k}"
        runs.append({
            "label": label,
            "prompt": prompt,
            "m": m,
            "n": n,
            "k": k,
            "path": eval_dir,
            "dir_name": entry.name,
        })

    return runs


def load_sweep_data(eval_dir: Path) -> dict | None:
    """Load trajectories from SkyRL JSONL files, group by prompt, compute pass@k.

    Returns dict with keys: prompts, pass_at_k, aggregated, n_prompts, n_samples.
    """
    # Read all trajectory JSONL files
    trajectories = []
    for f in sorted(eval_dir.glob("*.jsonl")):
        if f.stem == "aggregated_results":
            continue
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        trajectories.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            continue

    if not trajectories:
        return None

    # Read aggregated results (single JSON object)
    aggregated = {}
    agg_file = eval_dir / "aggregated_results.jsonl"
    if agg_file.exists():
        try:
            with open(agg_file) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        aggregated = json.loads(line)
                        break
        except (json.JSONDecodeError, OSError):
            pass

    # Group trajectories by input_prompt (N samples per prompt share the same prompt)
    prompt_groups = OrderedDict()
    for traj in trajectories:
        key = traj.get("input_prompt", "")
        if key not in prompt_groups:
            prompt_groups[key] = {
                "input_prompt": key,
                "target": extract_target(traj.get("env_extras", {})),
                "samples": [],
            }
        prompt_groups[key]["samples"].append({
            "output_response": traj.get("output_response", ""),
            "score": traj.get("score", 0),
            "stop_reason": traj.get("stop_reason"),
        })

    prompts = list(prompt_groups.values())
    for p in prompts:
        p["num_correct"] = sum(1 for s in p["samples"] if score_is_positive(s["score"]))

    # Compute pass@k for standard k values
    max_k = max((len(p["samples"]) for p in prompts), default=1)
    pass_at_k = compute_pass_at_k_values(prompts, max_k)

    return {
        "prompts": prompts,
        "pass_at_k": pass_at_k,
        "aggregated": aggregated,
        "n_prompts": len(prompts),
        "n_samples": len(prompts[0]["samples"]) if prompts else 0,
    }


# ---------------------------------------------------------------------------
# Legacy eval run discovery (for backward compat with old eval runs)
# ---------------------------------------------------------------------------

def discover_eval_runs(evals_dir: str) -> dict[str, Path]:
    """Auto-discover legacy eval runs (non-sweep). Returns {label: jsonl_dir_path}."""
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
        jsonl_files = list(eval_only_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue

        job_id = entry.name.replace("eval_", "")
        label = _parse_slurm_label(base, job_id)
        if not label:
            label = entry.name

        runs[label] = eval_only_dir

    return runs


def _parse_slurm_label(logs_dir: Path, job_id: str) -> str | None:
    out_file = logs_dir / f"eval-citation_{job_id}.out"
    if not out_file.exists():
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


def load_legacy_records(jsonl_dir: Path) -> list[dict]:
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


def record_is_correct(record: dict) -> bool:
    return score_is_positive(record.get("score", 0))


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def _heatmap_color(value: float) -> str:
    """Return a CSS background color for a 0-1 value (green=high, red=low)."""
    if value >= 0.7:
        return "#c6efce"
    elif value >= 0.5:
        return "#e2efda"
    elif value >= 0.3:
        return "#fff2cc"
    elif value >= 0.1:
        return "#fce4d6"
    else:
        return "#ffc7ce"


def build_heatmap_html(sweep_runs: list[dict], sweep_data_cache: dict,
                       prompt_filter: str, k_display: int) -> str:
    """Build an HTML heatmap table: rows=m, cols=n, cells=pass@k."""
    filtered = [r for r in sweep_runs if r["prompt"] == prompt_filter]
    if not filtered:
        return "<p>No sweep results found for this prompt style.</p>"

    m_values = sorted(set(r["m"] for r in filtered))
    n_values = sorted(set(r["n"] for r in filtered))

    # Build lookup: (m, n) -> pass@k_display
    lookup = {}
    for r in filtered:
        data = sweep_data_cache.get(r["dir_name"])
        if data is None:
            continue
        pak = data.get("pass_at_k", {})
        value = pak.get(k_display)
        if value is not None:
            lookup[(r["m"], r["n"])] = value

    if not lookup:
        return "<p>No pass@k data found. Results may still be running.</p>"

    rows = [CSS, f'<h3 style="text-align:center;color:#212529!important;">Pass@{k_display} — Prompt: {html.escape(prompt_filter)}</h3>']
    rows.append('<table class="heatmap-table">')

    rows.append('<thead><tr><th>m \\ n</th>')
    for n in n_values:
        rows.append(f'<th>n={n}</th>')
    rows.append('</tr></thead>')

    rows.append('<tbody>')
    for m in m_values:
        rows.append(f'<tr><th>m={m}</th>')
        for n in n_values:
            val = lookup.get((m, n))
            if val is not None:
                color = _heatmap_color(val)
                rows.append(f'<td style="background:{color}">{val*100:.1f}%</td>')
            else:
                rows.append('<td style="background:#f0f0f0">-</td>')
        rows.append('</tr>')
    rows.append('</tbody></table>')

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Pass@k curve plotting
# ---------------------------------------------------------------------------

def build_pass_at_k_plot(sweep_runs: list[dict], sweep_data_cache: dict,
                         selected_labels: list[str]) -> str | None:
    """Generate a pass@k curve plot. Returns image path or None."""
    if not HAS_MATPLOTLIB or not selected_labels:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for label in selected_labels:
        run = next((r for r in sweep_runs if r["label"] == label), None)
        if run is None:
            continue

        data = sweep_data_cache.get(run["dir_name"])
        if data is None:
            continue

        pak = data.get("pass_at_k", {})
        if not pak:
            continue

        ks = sorted(pak.keys())
        values = [pak[k] * 100 for k in ks]
        ax.plot(ks, values, marker='o', linewidth=2, label=f"m={run['m']} n={run['n']}")

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Pass@k (%)", fontsize=12)
    ax.set_title("Pass@k Curves", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return tmp.name


def build_pass_at_k_table(sweep_runs: list[dict], sweep_data_cache: dict,
                          selected_labels: list[str]) -> str:
    """Build an HTML table of pass@k values for selected setups."""
    if not selected_labels:
        return "<p>Select setups to compare.</p>"

    k_cols = [1, 2, 3, 5, 10, 15, 20]

    rows = [CSS, '<table class="metrics-table">']
    rows.append('<thead><tr><th>Setup</th>')
    for k in k_cols:
        rows.append(f'<th>pass@{k}</th>')
    rows.append('<th>Avg Score</th>')
    rows.append('</tr></thead>')

    rows.append('<tbody>')
    for label in selected_labels:
        run = next((r for r in sweep_runs if r["label"] == label), None)
        if run is None:
            continue

        data = sweep_data_cache.get(run["dir_name"])
        if data is None:
            continue

        pak = data.get("pass_at_k", {})
        avg_score = data.get("aggregated", {}).get("eval/all/avg_score", 0)

        rows.append(f'<tr><td><strong>m={run["m"]} n={run["n"]}</strong></td>')
        for k in k_cols:
            val = pak.get(k)
            if val is not None:
                css_class = "score-high" if val >= 0.5 else "score-low"
                rows.append(f'<td class="{css_class}">{val*100:.1f}%</td>')
            else:
                rows.append('<td>-</td>')
        rows.append(f'<td>{avg_score:.3f}</td>')
        rows.append('</tr>')

    rows.append('</tbody></table>')
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_app(evals_dir: str) -> gr.Blocks:
    # Mutable state
    sweep_runs: list[dict] = []
    sweep_data_cache: dict[str, dict] = {}  # dir_name -> loaded data
    legacy_discovered: dict[str, Path] = {}
    legacy_records: dict[str, list[dict]] = {}
    browse_filtered: list[int] = []

    def refresh_sweep():
        nonlocal sweep_runs, sweep_data_cache
        sweep_runs.clear()
        sweep_data_cache.clear()
        sweep_runs.extend(discover_sweep_runs(evals_dir))
        for r in sweep_runs:
            data = load_sweep_data(r["path"])
            if data:
                sweep_data_cache[r["dir_name"]] = data

    def refresh_legacy():
        nonlocal legacy_discovered, legacy_records
        legacy_discovered.clear()
        legacy_records.clear()
        legacy_discovered.update(discover_eval_runs(evals_dir))
        for label, path in legacy_discovered.items():
            legacy_records[label] = load_legacy_records(path)

    def get_prompt_styles():
        return sorted(set(r["prompt"] for r in sweep_runs)) or ["short"]

    def get_sweep_labels(prompt_filter=None):
        if prompt_filter:
            return [r["label"] for r in sweep_runs if r["prompt"] == prompt_filter]
        return [r["label"] for r in sweep_runs]

    def get_k_values():
        all_ks = set()
        for data in sweep_data_cache.values():
            all_ks.update(data.get("pass_at_k", {}).keys())
        return sorted(all_ks) or [1, 5, 10, 20]

    # ---- Initial load ----
    refresh_sweep()
    refresh_legacy()

    initial_prompts = get_prompt_styles()
    initial_labels = get_sweep_labels()
    initial_ks = get_k_values()

    # ==== BUILD UI ====
    with gr.Blocks(title="Base Model Eval Viewer") as app:
        gr.Markdown("# Base Model Eval Viewer — Citation Prediction")
        gr.Markdown("Sweep over hyperparameters (m=turns, n=topk) with pass@k evaluation.")

        with gr.Tabs():
            # ================================================================
            # Tab 1: Sweep Results (Heatmap)
            # ================================================================
            with gr.Tab("Sweep Results"):
                with gr.Row():
                    sweep_refresh = gr.Button("Refresh", variant="secondary", scale=1)
                    sweep_prompt_dd = gr.Dropdown(
                        choices=initial_prompts,
                        value=initial_prompts[0] if initial_prompts else None,
                        label="Prompt Style",
                        scale=2,
                    )
                    sweep_k_dd = gr.Dropdown(
                        choices=[str(k) for k in initial_ks],
                        value=str(initial_ks[-1]) if initial_ks else "20",
                        label="Display pass@k",
                        scale=1,
                    )

                sweep_heatmap = gr.HTML(label="Heatmap")

                def on_sweep_refresh():
                    refresh_sweep()
                    prompts = get_prompt_styles()
                    ks = get_k_values()
                    default_prompt = prompts[0] if prompts else None
                    default_k = str(ks[-1]) if ks else "20"
                    hm = build_heatmap_html(sweep_runs, sweep_data_cache,
                                            default_prompt or "short",
                                            int(default_k))
                    return (
                        gr.update(choices=prompts, value=default_prompt),
                        gr.update(choices=[str(k) for k in ks], value=default_k),
                        hm,
                    )

                def on_sweep_change(prompt, k_str):
                    if not prompt or not k_str:
                        return "<p>Select prompt style and k value.</p>"
                    return build_heatmap_html(sweep_runs, sweep_data_cache,
                                              prompt, int(k_str))

                sweep_refresh.click(
                    fn=on_sweep_refresh,
                    outputs=[sweep_prompt_dd, sweep_k_dd, sweep_heatmap],
                )
                sweep_prompt_dd.change(
                    fn=on_sweep_change,
                    inputs=[sweep_prompt_dd, sweep_k_dd],
                    outputs=[sweep_heatmap],
                )
                sweep_k_dd.change(
                    fn=on_sweep_change,
                    inputs=[sweep_prompt_dd, sweep_k_dd],
                    outputs=[sweep_heatmap],
                )
                app.load(
                    fn=on_sweep_change,
                    inputs=[sweep_prompt_dd, sweep_k_dd],
                    outputs=[sweep_heatmap],
                )

            # ================================================================
            # Tab 2: Pass@k Curves
            # ================================================================
            with gr.Tab("Pass@k Curves"):
                with gr.Row():
                    curve_refresh = gr.Button("Refresh", variant="secondary", scale=1)
                    curve_prompt_dd = gr.Dropdown(
                        choices=initial_prompts,
                        value=initial_prompts[0] if initial_prompts else None,
                        label="Prompt Style",
                        scale=1,
                    )
                    curve_select = gr.CheckboxGroup(
                        choices=initial_labels,
                        value=initial_labels[:5],
                        label="Select setups to compare",
                    )

                curve_plot = gr.Image(label="Pass@k Curves", type="filepath")
                curve_table = gr.HTML(label="Pass@k Values")

                def on_curve_refresh():
                    refresh_sweep()
                    prompts = get_prompt_styles()
                    default_prompt = prompts[0] if prompts else None
                    labels = get_sweep_labels(default_prompt)
                    selected = labels[:5]
                    plot_path = build_pass_at_k_plot(sweep_runs, sweep_data_cache, selected)
                    table_html = build_pass_at_k_table(sweep_runs, sweep_data_cache, selected)
                    return (
                        gr.update(choices=prompts, value=default_prompt),
                        gr.update(choices=labels, value=selected),
                        plot_path,
                        table_html,
                    )

                def on_curve_prompt_change(prompt):
                    labels = get_sweep_labels(prompt)
                    return gr.update(choices=labels, value=labels[:5])

                def on_curve_update(selected):
                    if not selected:
                        return None, "<p>Select setups to compare.</p>"
                    plot_path = build_pass_at_k_plot(sweep_runs, sweep_data_cache, selected)
                    table_html = build_pass_at_k_table(sweep_runs, sweep_data_cache, selected)
                    return plot_path, table_html

                curve_refresh.click(
                    fn=on_curve_refresh,
                    outputs=[curve_prompt_dd, curve_select, curve_plot, curve_table],
                )
                curve_prompt_dd.change(
                    fn=on_curve_prompt_change,
                    inputs=[curve_prompt_dd],
                    outputs=[curve_select],
                )
                curve_select.change(
                    fn=on_curve_update,
                    inputs=[curve_select],
                    outputs=[curve_plot, curve_table],
                )
                app.load(
                    fn=on_curve_update,
                    inputs=[curve_select],
                    outputs=[curve_plot, curve_table],
                )

            # ================================================================
            # Tab 3: Browse Trajectories
            # ================================================================
            with gr.Tab("Browse"):
                with gr.Row():
                    browse_refresh = gr.Button("Refresh", variant="secondary", scale=1)
                    browse_setup_dd = gr.Dropdown(
                        choices=initial_labels,
                        value=initial_labels[0] if initial_labels else None,
                        label="Setup",
                        scale=2,
                    )
                    browse_filter_dd = gr.Dropdown(
                        choices=["All", "Found", "Not Found"],
                        value="All",
                        label="Filter",
                        scale=1,
                    )
                    browse_sample_dd = gr.Dropdown(
                        choices=["Sample 0"],
                        value="Sample 0",
                        label="Sample",
                        scale=1,
                    )

                with gr.Row():
                    browse_stats = gr.Textbox(label="Stats", interactive=False, scale=1)
                    browse_prev = gr.Button("< Prev", scale=1)
                    browse_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0,
                                              label="Example", scale=3)
                    browse_next = gr.Button("Next >", scale=1)

                browse_idx_state = gr.State(0)
                browse_html = gr.HTML(label="Trajectory")

                def _get_browse_data(setup_label):
                    """Get the sweep run and loaded data for a setup label."""
                    for r in sweep_runs:
                        if r["label"] == setup_label:
                            return r, sweep_data_cache.get(r["dir_name"])
                    return None, None

                def on_browse_refresh():
                    refresh_sweep()
                    labels = get_sweep_labels()
                    default = labels[0] if labels else None
                    if default:
                        return (
                            gr.update(choices=labels, value=default),
                            *_load_browse(default, "All"),
                        )
                    return (
                        gr.update(choices=[], value=None),
                        "No data",
                        gr.update(maximum=0, value=0),
                        gr.update(choices=["Sample 0"], value="Sample 0"),
                        "<p>No sweep results found.</p>",
                    )

                def _load_browse(setup, score_filter):
                    nonlocal browse_filtered
                    if not setup:
                        browse_filtered = []
                        return ("0 prompts",
                                gr.update(maximum=0, value=0),
                                gr.update(choices=["Sample 0"], value="Sample 0"),
                                "<p>Select a setup.</p>")

                    run, data = _get_browse_data(setup)
                    if data is None:
                        browse_filtered = []
                        return ("0 prompts",
                                gr.update(maximum=0, value=0),
                                gr.update(choices=["Sample 0"], value="Sample 0"),
                                "<p>No data for this setup.</p>")

                    prompts = data.get("prompts", [])
                    n_samples = data.get("n_samples", 1)

                    # Build sample dropdown
                    sample_choices = [f"Sample {i}" for i in range(n_samples)]
                    sample_value = sample_choices[0] if sample_choices else "Sample 0"

                    # Filter by correctness
                    if score_filter == "Found":
                        browse_filtered = [i for i, p in enumerate(prompts) if p["num_correct"] > 0]
                    elif score_filter == "Not Found":
                        browse_filtered = [i for i, p in enumerate(prompts) if p["num_correct"] == 0]
                    else:
                        browse_filtered = list(range(len(prompts)))

                    total = len(browse_filtered)
                    n_found = sum(1 for p in prompts if p["num_correct"] > 0)
                    accuracy = n_found / len(prompts) * 100 if prompts else 0

                    stats = f"{total} / {len(prompts)} prompts | {n_found} with at least 1 correct ({accuracy:.1f}%)"
                    if total == 0:
                        return (stats,
                                gr.update(maximum=0, value=0),
                                gr.update(choices=sample_choices, value=sample_value),
                                "<p>No matching examples.</p>")

                    html_content = _render_browse_example(data, run, browse_filtered[0], 0)
                    return (stats,
                            gr.update(maximum=total - 1, value=0),
                            gr.update(choices=sample_choices, value=sample_value),
                            html_content)

                def _render_browse_example(data, run, prompt_idx, sample_idx):
                    prompts = data.get("prompts", [])
                    if prompt_idx >= len(prompts):
                        return "<p>Index out of range.</p>"

                    prompt_data = prompts[prompt_idx]
                    samples = prompt_data.get("samples", [])
                    sample_idx = max(0, min(sample_idx, len(samples) - 1))

                    if not samples:
                        return "<p>No sample data available.</p>"

                    sample = samples[sample_idx]
                    found = score_is_positive(sample["score"])

                    # Count search turns from the output
                    n_searches = len(re.findall(r"<search>", sample["output_response"]))

                    badges = [
                        f"m={run['m']}" if run else "",
                        f"n={run['n']}" if run else "",
                        f"sample {sample_idx}/{len(samples)}",
                        f"{prompt_data['num_correct']}/{len(samples)} correct",
                        f"{n_searches} searches",
                    ]
                    badges = [b for b in badges if b]

                    return render_trajectory_html(
                        input_prompt=prompt_data["input_prompt"],
                        output_response=sample["output_response"],
                        target=prompt_data["target"],
                        found=found,
                        label=f"Prompt {prompt_idx}",
                        extra_badges=badges,
                    )

                def on_browse_change(setup, score_filter):
                    return _load_browse(setup, score_filter)

                def on_browse_navigate(idx, setup, sample_str):
                    if not browse_filtered or not setup:
                        return "<p>No data.</p>"
                    idx = max(0, min(int(idx), len(browse_filtered) - 1))
                    sample_idx = int(re.search(r"\d+", sample_str or "0").group()) if sample_str else 0
                    run, data = _get_browse_data(setup)
                    if data is None:
                        return "<p>No data.</p>"
                    return _render_browse_example(data, run, browse_filtered[idx], sample_idx)

                def on_browse_prev(idx, setup, sample_str):
                    new_idx = max(0, int(idx) - 1)
                    return new_idx, on_browse_navigate(new_idx, setup, sample_str)

                def on_browse_next(idx, setup, sample_str):
                    max_idx = len(browse_filtered) - 1 if browse_filtered else 0
                    new_idx = min(max_idx, int(idx) + 1)
                    return new_idx, on_browse_navigate(new_idx, setup, sample_str)

                browse_outputs = [browse_stats, browse_slider, browse_sample_dd, browse_html]

                browse_refresh.click(
                    fn=on_browse_refresh,
                    outputs=[browse_setup_dd, *browse_outputs],
                )
                browse_setup_dd.change(
                    fn=on_browse_change,
                    inputs=[browse_setup_dd, browse_filter_dd],
                    outputs=browse_outputs,
                )
                browse_filter_dd.change(
                    fn=on_browse_change,
                    inputs=[browse_setup_dd, browse_filter_dd],
                    outputs=browse_outputs,
                )
                browse_prev.click(
                    fn=on_browse_prev,
                    inputs=[browse_idx_state, browse_setup_dd, browse_sample_dd],
                    outputs=[browse_idx_state, browse_html],
                )
                browse_next.click(
                    fn=on_browse_next,
                    inputs=[browse_idx_state, browse_setup_dd, browse_sample_dd],
                    outputs=[browse_idx_state, browse_html],
                )
                browse_slider.release(
                    fn=lambda idx, setup, sample: (idx, on_browse_navigate(idx, setup, sample)),
                    inputs=[browse_slider, browse_setup_dd, browse_sample_dd],
                    outputs=[browse_idx_state, browse_html],
                )
                browse_sample_dd.change(
                    fn=lambda setup, score_f, sample, idx: on_browse_navigate(idx, setup, sample),
                    inputs=[browse_setup_dd, browse_filter_dd, browse_sample_dd, browse_idx_state],
                    outputs=[browse_html],
                )

            # ================================================================
            # Tab 4: Summary Table
            # ================================================================
            with gr.Tab("Summary"):
                summary_refresh = gr.Button("Refresh", variant="secondary")
                summary_html = gr.HTML(label="Summary Table")

                def build_summary():
                    refresh_sweep()
                    refresh_legacy()

                    rows = [CSS, '<table class="metrics-table">']
                    rows.append("<thead><tr>")
                    rows.append("<th>Setup</th><th>Prompt</th><th>m</th><th>n</th>")
                    rows.append("<th>pass@1</th><th>pass@5</th><th>pass@10</th><th>pass@20</th>")
                    rows.append("<th>Avg Score</th><th># Prompts</th>")
                    rows.append("</tr></thead><tbody>")

                    # Sweep results
                    for r in sorted(sweep_runs, key=lambda x: (x["prompt"], x["m"], x["n"])):
                        data = sweep_data_cache.get(r["dir_name"])
                        if data is None:
                            continue

                        pak = data.get("pass_at_k", {})
                        avg_score = data.get("aggregated", {}).get("eval/all/avg_score", 0)
                        n_prompts = data.get("n_prompts", 0)

                        rows.append(f'<tr><td><strong>{html.escape(r["label"])}</strong></td>')
                        rows.append(f'<td>{html.escape(r["prompt"])}</td>')
                        rows.append(f'<td>{r["m"]}</td><td>{r["n"]}</td>')

                        for k in [1, 5, 10, 20]:
                            val = pak.get(k)
                            if val is not None:
                                css_class = "score-high" if val >= 0.5 else "score-low"
                                rows.append(f'<td class="{css_class}">{val*100:.1f}%</td>')
                            else:
                                rows.append('<td>-</td>')

                        rows.append(f'<td>{avg_score:.3f}</td>')
                        rows.append(f'<td>{n_prompts}</td></tr>')

                    # Legacy results
                    if legacy_discovered:
                        rows.append('<tr><td colspan="10" style="background:#e9ecef;font-weight:700;">Legacy Eval Runs</td></tr>')
                        for label in sorted(legacy_discovered.keys()):
                            records = legacy_records.get(label, [])
                            n = len(records)
                            if n == 0:
                                continue
                            n_correct = sum(1 for r in records if record_is_correct(r))
                            accuracy = n_correct / n * 100

                            rows.append(f'<tr><td><strong>{html.escape(label)}</strong></td>')
                            rows.append('<td>-</td><td>-</td><td>-</td>')
                            acc_class = "score-high" if accuracy >= 50 else "score-low"
                            rows.append(f'<td class="{acc_class}">{accuracy:.1f}%</td>')
                            rows.append('<td>-</td><td>-</td><td>-</td>')
                            rows.append(f'<td>-</td><td>{n}</td></tr>')

                    rows.append("</tbody></table>")

                    if not sweep_runs and not legacy_discovered:
                        return "<p>No eval results found. Run sweep_eval.sh and click Refresh.</p>"

                    return "\n".join(rows)

                summary_refresh.click(fn=build_summary, outputs=[summary_html])
                app.load(fn=build_summary, outputs=[summary_html])

    return app


def main():
    parser = argparse.ArgumentParser(description="Base Model Eval Viewer")
    parser.add_argument(
        "--evals-dir",
        default=DEFAULT_EVALS_DIR,
        help="Directory containing eval_* and sweep/ output directories",
    )
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app = create_app(args.evals_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
