"""
Gradio viewer for citation prediction v2 evaluation trajectories.

Reads SkyRL's dumped_evals JSONL format. Provides:
  - Sweep Results: heatmap of mean recall / best-of-k recall across (m, n)
  - Best-of-k Curves: plot best recall vs k for selected setups
  - Browse: view individual trajectories with citation highlighting
  - Summary: recall metrics table across all setups

Usage:
    python base_eval_viewer.py [--evals-dir /path/to/logs] [--port 7861] [--share]
"""

import argparse
import html
import json
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

# ---------------------------------------------------------------------------
# Citation extraction & recall computation
# ---------------------------------------------------------------------------

CITATION_TAG_RE = re.compile(r"<citation>(.*?)</citation>", re.DOTALL)
ARXIV_ID_RE = re.compile(r"\d{4}\.\d{4,5}")


def normalize_arxiv_id(s: str) -> str:
    """Normalize an arxiv ID to YYMM.NNNNN format."""
    m = re.search(r"(\d{4}\.\d{4,5})", s.strip())
    return m.group(1) if m else ""


def extract_all_citations(text: str) -> set[str]:
    """Extract all arxiv IDs from <citation> tags in trajectory text."""
    ids = set()
    for match in CITATION_TAG_RE.finditer(text):
        raw = match.group(1)
        for part in raw.split(","):
            nid = normalize_arxiv_id(part)
            if nid:
                ids.add(nid)
    return ids


def compute_recall(predicted: set[str], targets: list[str],
                   max_ratio: float = 2.0) -> dict:
    """Compute recall with spam penalty."""
    gt_set = {normalize_arxiv_id(t) for t in targets}
    gt_set.discard("")
    correct = predicted & gt_set
    n_gt = len(gt_set)
    n_pred = len(predicted)
    n_correct = len(correct)

    if n_pred > max_ratio * n_gt:
        recall = 0.0
    else:
        recall = n_correct / n_gt if n_gt else 0.0

    precision = n_correct / n_pred if n_pred else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "n_predicted": n_pred,
        "n_correct": n_correct,
        "n_ground_truth": n_gt,
        "correct_ids": sorted(correct),
        "predicted_ids": sorted(predicted),
    }


# ---------------------------------------------------------------------------
# Best-of-k recall computation
# ---------------------------------------------------------------------------

def compute_best_of_k_recall(prompts: list[dict], max_k: int = 20) -> dict[int, float]:
    """For each k, average across prompts: best recall among k random samples.

    Samples are already in random order (stochastic rollouts), so best-of-k
    is simply the max recall among the first k samples for each prompt,
    averaged across prompts.
    """
    if not prompts:
        return {}
    result = {}
    for k in range(1, max_k + 1):
        total = 0.0
        valid = 0
        for p in prompts:
            recalls = [s["recall"] for s in p["samples"]]
            if k > len(recalls):
                continue
            best_k = max(recalls[:k])
            total += best_k
            valid += 1
        if valid > 0:
            result[k] = total / valid
    return result


def compute_any_nonzero_at_k(prompts: list[dict], max_k: int = 20) -> dict[int, float]:
    """Fraction of prompts where at least 1 of k samples has recall > 0."""
    if not prompts:
        return {}
    result = {}
    for k in range(1, max_k + 1):
        total = 0.0
        valid = 0
        for p in prompts:
            recalls = [s["recall"] for s in p["samples"]]
            if k > len(recalls):
                continue
            any_pos = any(r > 0 for r in recalls[:k])
            total += float(any_pos)
            valid += 1
        if valid > 0:
            result[k] = total / valid
    return result


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def get_final_score(score) -> float:
    """Extract final scalar reward from SkyRL score (may be per-token list)."""
    if isinstance(score, list):
        return score[-1] if score else 0.0
    return float(score) if score else 0.0


def extract_targets(env_extras: dict) -> list[str]:
    """Extract target arxiv IDs list from env_extras."""
    try:
        rs = env_extras.get("reward_spec", {})
        if isinstance(rs, str):
            rs = json.loads(rs)
        return rs.get("ground_truth", {}).get("targets", [])
    except (AttributeError, TypeError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# Trajectory parsing & rendering
# ---------------------------------------------------------------------------

ACTION_PATTERN = re.compile(
    r"<(search|information|citation|done)>(.*?)</\1>",
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
.tv-meta { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
.tv-badge {
    padding: 2px 10px; border-radius: 12px;
    font-size: 0.85em; font-weight: 600;
}
.tv-badge-good { background: #d4edda; color: #155724 !important; }
.tv-badge-partial { background: #fff3cd; color: #856404 !important; }
.tv-badge-bad { background: #f8d7da; color: #721c24 !important; }
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
.tv-citation { background: #e8d5f5; border-left-color: #7c3aed; }
.tv-citation .tv-segment-label { color: #5b21b6 !important; }
.tv-citation-id-correct { color: #155724 !important; font-weight: 700; background: #d4edda; padding: 1px 6px; border-radius: 4px; }
.tv-citation-id-incorrect { color: #721c24 !important; font-weight: 700; background: #f8d7da; padding: 1px 6px; border-radius: 4px; }
.tv-citation-id-unknown { color: #856404 !important; font-weight: 700; background: #fff3cd; padding: 1px 6px; border-radius: 4px; }
.tv-done { background: #e2e3e5; border-left-color: #383d41; padding: 14px; }
.tv-done .tv-segment-label { color: #383d41 !important; }
.tv-text { background: #fff; border-left-color: #ced4da; color: #6c757d !important; }

.tv-targets { margin-top: 8px; }
.tv-target-found { color: #155724 !important; font-weight: 600; }
.tv-target-missed { color: #721c24 !important; }

.metrics-table { border-collapse: collapse; width: 100%; }
.metrics-table th, .metrics-table td, .metrics-table td strong {
    border: 1px solid #dee2e6; padding: 8px 12px;
    text-align: center; color: #212529 !important; background: #fff !important;
}
.metrics-table th { background: #f8f9fa !important; font-weight: 600; color: #212529 !important; }
.metrics-table tr:nth-child(even) td { background: #f8f9fa !important; }
.metrics-table td.score-high { color: #155724 !important; background: #d4edda !important; font-weight: 600; }
.metrics-table td.score-mid { color: #856404 !important; background: #fff3cd !important; font-weight: 600; }
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
    """Parse output_response into segments (think/search/information/citation/done/text)."""
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


def extract_question_from_prompt(input_prompt: str) -> tuple[str, str, str, str]:
    """Extract summary, full user prompt, system prompt, and query from a decoded SkyRL prompt.

    Returns (summary_line, full_user_prompt, system_prompt, query).
    """
    # Extract system prompt
    sys_match = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", input_prompt, re.DOTALL)
    system_prompt = sys_match.group(1).strip() if sys_match else ""

    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", input_prompt, re.DOTALL)
    if match:
        text = match.group(1).strip()
        # Extract compact summary
        heading_match = re.search(r'Related Work subsection heading:\s*"(.+?)"', text)
        title_match = re.search(r"Paper title:\s*(.+?)(?:\n|$)", text)
        if heading_match and title_match:
            summary = f"{title_match.group(1).strip()} -> [{heading_match.group(1).strip()}]"
        else:
            summary = text[:300]

        # Extract the rich query (last paragraph before "Citation budget:")
        query = ""
        query_match = re.search(
            r".*\n\n(.+?)\n\nCitation budget:",
            text, re.DOTALL,
        )
        if query_match:
            query = query_match.group(1).strip()

        return summary, text, system_prompt, query
    return input_prompt[-300:], input_prompt, system_prompt, ""


def render_citation_html(raw_content: str, gt_set: set[str]) -> str:
    """Render citation tag content with correct/incorrect highlighting."""
    parts = []
    for chunk in raw_content.split(","):
        chunk = chunk.strip()
        nid = normalize_arxiv_id(chunk)
        if nid and nid in gt_set:
            parts.append(f'<span class="tv-citation-id-correct">{html.escape(chunk)}</span>')
        elif nid:
            parts.append(f'<span class="tv-citation-id-incorrect">{html.escape(chunk)}</span>')
        elif chunk:
            parts.append(f'<span class="tv-citation-id-unknown">{html.escape(chunk)}</span>')
    return ", ".join(parts)


def render_trajectory_html(input_prompt: str, output_response: str,
                           targets: list[str], recall: float,
                           metrics: dict = None,
                           label: str = "", extra_badges: list = None) -> str:
    """Render a v2 trajectory as styled HTML with citation highlighting."""
    question, full_prompt, system_prompt, query = extract_question_from_prompt(input_prompt)
    segments = parse_trajectory(output_response)
    gt_set = {normalize_arxiv_id(t) for t in targets}
    gt_set.discard("")

    # Predicted IDs from trajectory
    predicted = extract_all_citations(output_response)

    parts: list[str] = [CSS]

    # Recall badge
    if recall >= 0.5:
        badge_class = "good"
    elif recall > 0:
        badge_class = "partial"
    else:
        badge_class = "bad"
    recall_label = f"Recall: {recall*100:.0f}%"
    label_html = f"<div class='tv-step-label'>{html.escape(str(label))}</div>" if label else ""

    badge_html = f'<span class="tv-badge tv-badge-{badge_class}">{recall_label}</span>'
    if metrics:
        badge_html += f' <span class="tv-badge tv-badge-info">{metrics["n_correct"]}/{metrics["n_ground_truth"]} correct</span>'
        badge_html += f' <span class="tv-badge tv-badge-info">{metrics["n_predicted"]} predicted</span>'
    if extra_badges:
        for badge in extra_badges:
            badge_html += f' <span class="tv-badge tv-badge-info">{html.escape(badge)}</span>'

    # Ground truth display with found/missed highlighting
    targets_html = []
    for t in targets:
        nid = normalize_arxiv_id(t)
        if nid in predicted:
            targets_html.append(f'<span class="tv-target-found">{html.escape(nid)}</span>')
        else:
            targets_html.append(f'<span class="tv-target-missed">{html.escape(nid)}</span>')

    # Query display (prominent at top)
    query_html = ""
    if query:
        query_html = f'''<div style="background:#e7f5ff;border:1px solid #74c0fc;border-radius:8px;padding:12px 16px;margin-bottom:12px;">
            <div style="font-weight:700;color:#1864ab!important;font-size:0.85em;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Query</div>
            <div style="color:#212529!important;font-size:1.05em;">{html.escape(query)}</div>
        </div>'''

    parts.append(f"""
    <div class="tv-header">
        {label_html}
        {query_html}
        <div class="tv-meta">{badge_html}</div>
        <div class="tv-targets"><strong>Targets ({len(gt_set)}):</strong> {", ".join(targets_html)}</div>
        {"" if not system_prompt else '''<details style="margin-top:10px;">
            <summary style="cursor:pointer;font-weight:600;color:#495057!important;">System Prompt (click to expand)</summary>
            <pre style="white-space:pre-wrap;word-wrap:break-word;font-size:0.85em;margin-top:8px;max-height:500px;overflow-y:auto;background:#f1f3f5;padding:10px;border-radius:6px;color:#212529!important;">''' + html.escape(system_prompt) + '''</pre>
        </details>'''}
        <details style="margin-top:10px;">
            <summary style="cursor:pointer;font-weight:600;color:#495057!important;">Full Prompt (click to expand)</summary>
            <pre style="white-space:pre-wrap;word-wrap:break-word;font-size:0.85em;margin-top:8px;max-height:500px;overflow-y:auto;background:#f1f3f5;padding:10px;border-radius:6px;color:#212529!important;">{html.escape(full_prompt)}</pre>
        </details>
    </div>
    """)

    parts.append('<div class="tv-trajectory">')
    turn_num = 0
    for seg in segments:
        if seg["type"] == "think":
            turn_num += 1
            content_escaped = html.escape(seg["content"])
            parts.append(f"""
            <details class="tv-segment tv-think">
                <summary>Thinking (Turn {turn_num})</summary>
                <pre>{content_escaped}</pre>
            </details>
            """)
        elif seg["type"] == "search":
            content_escaped = html.escape(seg["content"])
            parts.append(f"""
            <div class="tv-segment tv-search">
                <div class="tv-segment-label">Search Query</div>
                <div class="tv-segment-content">{content_escaped}</div>
            </div>
            """)
        elif seg["type"] == "information":
            content_escaped = html.escape(seg["content"])
            parts.append(f"""
            <details class="tv-segment tv-information">
                <summary>Search Results</summary>
                <pre>{content_escaped}</pre>
            </details>
            """)
        elif seg["type"] == "citation":
            citation_html = render_citation_html(seg["content"], gt_set)
            parts.append(f"""
            <div class="tv-segment tv-citation">
                <div class="tv-segment-label">Citation</div>
                <div class="tv-segment-content">{citation_html}</div>
            </div>
            """)
        elif seg["type"] == "done":
            parts.append(f"""
            <div class="tv-segment tv-done">
                <div class="tv-segment-label">Done</div>
            </div>
            """)
        else:
            content_escaped = html.escape(seg["content"])
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

SWEEP_DIR_RE = re.compile(r"^([\w-]+)_m(\d+)_n(\d+)_k(\d+)(?:_part\d+)?$")


def discover_sweep_runs(evals_dir: str) -> list[dict]:
    """Discover sweep results from the sweep/ subdirectory.

    Merges _part0/_part1/... directories into a single logical run by
    collecting all their eval_dir paths together.
    """
    sweep_dir = Path(evals_dir) / "sweep"
    if not sweep_dir.is_dir():
        return []

    # Group directories by config key (without _partN suffix)
    config_parts: dict[str, list[Path]] = OrderedDict()
    config_meta: dict[str, dict] = {}

    for entry in sorted(sweep_dir.iterdir()):
        if not entry.is_dir():
            continue

        eval_dir = entry / "dumped_evals" / "eval_only"
        if not eval_dir.is_dir():
            continue

        jsonl_files = [f for f in eval_dir.glob("*.jsonl") if f.stem != "aggregated_results"]
        if not jsonl_files:
            continue

        match = SWEEP_DIR_RE.match(entry.name)
        if not match:
            continue

        prompt = match.group(1)
        m = int(match.group(2))
        n = int(match.group(3))
        k = int(match.group(4))

        # Strip _partN to get config key
        config_key = f"{prompt}_m{m}_n{n}_k{k}"
        if config_key not in config_parts:
            config_parts[config_key] = []
            config_meta[config_key] = {"prompt": prompt, "m": m, "n": n, "k": k}
        config_parts[config_key].append(eval_dir)

    runs = []
    for config_key, eval_dirs in config_parts.items():
        meta = config_meta[config_key]
        label = f"{meta['prompt']} m={meta['m']} n={meta['n']} k={meta['k']}"
        runs.append({
            "label": label,
            "prompt": meta["prompt"],
            "m": meta["m"],
            "n": meta["n"],
            "k": meta["k"],
            "path": eval_dirs,  # list of eval_dir paths (merged parts)
            "dir_name": config_key,
        })

    return runs


def load_sweep_data(eval_dirs: list[Path] | Path) -> dict | None:
    """Load trajectories, group by prompt, compute recall metrics.

    eval_dirs can be a single Path or a list of Paths (merged parts).
    """
    if isinstance(eval_dirs, Path):
        eval_dirs = [eval_dirs]

    trajectories = []
    for eval_dir in eval_dirs:
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

    # Read aggregated results (average across parts)
    aggregated = {}
    agg_results = []
    for eval_dir in eval_dirs:
        agg_file = eval_dir / "aggregated_results.jsonl"
        if agg_file.exists():
            try:
                with open(agg_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            agg_results.append(json.loads(line))
                            break
            except (json.JSONDecodeError, OSError):
                pass
    if agg_results:
        # Average numeric values across parts
        all_keys = set()
        for ar in agg_results:
            all_keys.update(ar.keys())
        for key in all_keys:
            vals = [ar[key] for ar in agg_results if key in ar and isinstance(ar[key], (int, float))]
            if vals:
                aggregated[key] = sum(vals) / len(vals)

    # Group trajectories by input_prompt
    prompt_groups = OrderedDict()
    for traj in trajectories:
        key = traj.get("input_prompt", "")
        if key not in prompt_groups:
            targets = extract_targets(traj.get("env_extras", {}))
            prompt_groups[key] = {
                "input_prompt": key,
                "targets": targets,
                "samples": [],
            }

        output = traj.get("output_response", "")
        score = get_final_score(traj.get("score", 0))
        predicted = extract_all_citations(output)
        targets = prompt_groups[key]["targets"]
        metrics = compute_recall(predicted, targets)

        prompt_groups[key]["samples"].append({
            "output_response": output,
            "score": score,
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "n_predicted": metrics["n_predicted"],
            "n_correct": metrics["n_correct"],
            "n_ground_truth": metrics["n_ground_truth"],
            "predicted_ids": metrics["predicted_ids"],
            "correct_ids": metrics["correct_ids"],
            "stop_reason": traj.get("stop_reason"),
        })

    prompts = list(prompt_groups.values())

    # Per-prompt aggregated stats
    for p in prompts:
        recalls = [s["recall"] for s in p["samples"]]
        p["best_recall"] = max(recalls) if recalls else 0.0
        p["mean_recall"] = sum(recalls) / len(recalls) if recalls else 0.0
        p["any_nonzero"] = any(r > 0 for r in recalls)
        p["num_nonzero"] = sum(1 for r in recalls if r > 0)

    # Compute best-of-k recall and any-nonzero@k
    max_k = max((len(p["samples"]) for p in prompts), default=1)
    best_of_k = compute_best_of_k_recall(prompts, max_k)
    any_nonzero_at_k = compute_any_nonzero_at_k(prompts, max_k)

    # Overall stats
    mean_recall_1 = sum(p["samples"][0]["recall"] for p in prompts) / len(prompts) if prompts else 0
    mean_best_recall = sum(p["best_recall"] for p in prompts) / len(prompts) if prompts else 0
    frac_any_nonzero = sum(1 for p in prompts if p["any_nonzero"]) / len(prompts) if prompts else 0
    frac_all_zero = sum(1 for p in prompts if not p["any_nonzero"]) / len(prompts) if prompts else 0

    return {
        "prompts": prompts,
        "best_of_k": best_of_k,
        "any_nonzero_at_k": any_nonzero_at_k,
        "aggregated": aggregated,
        "n_prompts": len(prompts),
        "n_samples": len(prompts[0]["samples"]) if prompts else 0,
        "mean_recall_1": mean_recall_1,
        "mean_best_recall": mean_best_recall,
        "frac_any_nonzero": frac_any_nonzero,
        "frac_all_zero": frac_all_zero,
    }


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def _heatmap_color(value: float) -> str:
    if value >= 0.5:
        return "#c6efce"
    elif value >= 0.3:
        return "#e2efda"
    elif value >= 0.15:
        return "#fff2cc"
    elif value >= 0.05:
        return "#fce4d6"
    else:
        return "#ffc7ce"


def build_heatmap_html(sweep_runs: list[dict], sweep_data_cache: dict,
                       prompt_filter: str, metric: str) -> str:
    """Build heatmap: rows=m, cols=n, cells=selected metric."""
    filtered = [r for r in sweep_runs if r["prompt"] == prompt_filter]
    if not filtered:
        return "<p>No sweep results found for this prompt style.</p>"

    m_values = sorted(set(r["m"] for r in filtered))
    n_values = sorted(set(r["n"] for r in filtered))

    lookup = {}
    for r in filtered:
        data = sweep_data_cache.get(r["dir_name"])
        if data is None:
            continue
        if metric == "mean_best_recall":
            lookup[(r["m"], r["n"])] = data["mean_best_recall"]
        elif metric == "mean_recall_1":
            lookup[(r["m"], r["n"])] = data["mean_recall_1"]
        elif metric == "frac_any_nonzero":
            lookup[(r["m"], r["n"])] = data["frac_any_nonzero"]

    if not lookup:
        return "<p>No data found. Results may still be running.</p>"

    metric_labels = {
        "mean_best_recall": "Mean Best Recall (best of k samples)",
        "mean_recall_1": "Mean Recall@1",
        "frac_any_nonzero": "Fraction with Any Nonzero Recall",
    }

    rows = [CSS, f'<h3 style="text-align:center;color:#212529!important;">{metric_labels.get(metric, metric)} — Prompt: {html.escape(prompt_filter)}</h3>']
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
# Best-of-k curve plotting
# ---------------------------------------------------------------------------

def build_best_of_k_plot(sweep_runs: list[dict], sweep_data_cache: dict,
                         selected_labels: list[str], plot_type: str = "best_of_k") -> str | None:
    """Generate best-of-k recall or any-nonzero@k curve plot."""
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

        curve = data.get(plot_type, {})
        if not curve:
            continue

        ks = sorted(curve.keys())
        values = [curve[k] * 100 for k in ks]
        ax.plot(ks, values, marker='o', linewidth=2, label=f"m={run['m']} n={run['n']}")

    ylabel = "Best-of-k Recall (%)" if plot_type == "best_of_k" else "Any Nonzero@k (%)"
    title = "Best-of-k Recall Curves" if plot_type == "best_of_k" else "Any Nonzero@k Curves"

    ax.set_xlabel("k (number of samples)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return tmp.name


def build_recall_distribution_plot(sweep_runs: list[dict], sweep_data_cache: dict,
                                   selected_labels: list[str]) -> str | None:
    """Generate a Best-of-20 recall distribution histogram for selected setups."""
    if not HAS_MATPLOTLIB or not selected_labels:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    bins = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bin_labels = ["0%", "(0,10]", "(10,20]", "(20,30]", "(30,40]",
                  "(40,50]", "(50,60]", "(60,70]", "(70,80]", "(80,90]", "(90,100]"]

    import numpy as np
    n_setups = len(selected_labels)
    width = 0.8 / n_setups
    x = np.arange(len(bin_labels))

    for i, label in enumerate(selected_labels):
        run = next((r for r in sweep_runs if r["label"] == label), None)
        if run is None:
            continue
        data = sweep_data_cache.get(run["dir_name"])
        if data is None:
            continue

        best_recalls = [p["best_recall"] for p in data["prompts"]]
        counts, _ = np.histogram(best_recalls, bins=bins)
        fracs = counts / len(best_recalls) * 100

        offset = (i - n_setups / 2 + 0.5) * width
        bars = ax.bar(x + offset, fracs, width, label=label, alpha=0.8)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(int(count)), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Best-of-20 Recall", fontsize=12)
    ax.set_ylabel("% of Prompts", fontsize=12)
    ax.set_title("Best-of-20 Recall Distribution by Prompt", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return tmp.name


def build_metrics_table(sweep_runs: list[dict], sweep_data_cache: dict,
                        selected_labels: list[str]) -> str:
    """Build HTML table of recall metrics for selected setups."""
    if not selected_labels:
        return "<p>Select setups to compare.</p>"

    k_cols = [1, 2, 5, 10, 15, 20]

    rows = [CSS, '<table class="metrics-table">']
    rows.append('<thead><tr><th>Setup</th>')
    for k in k_cols:
        rows.append(f'<th>best@{k}</th>')
    rows.append('<th>any>0@20</th><th>Avg Score</th>')
    rows.append('</tr></thead>')

    rows.append('<tbody>')
    for label in selected_labels:
        run = next((r for r in sweep_runs if r["label"] == label), None)
        if run is None:
            continue
        data = sweep_data_cache.get(run["dir_name"])
        if data is None:
            continue

        bok = data.get("best_of_k", {})
        anz = data.get("any_nonzero_at_k", {})
        avg_score = data.get("aggregated", {}).get("eval/all/avg_score", data.get("mean_recall_1", 0))

        rows.append(f'<tr><td><strong>m={run["m"]} n={run["n"]}</strong></td>')
        for k in k_cols:
            val = bok.get(k)
            if val is not None:
                if val >= 0.3:
                    css_class = "score-high"
                elif val >= 0.1:
                    css_class = "score-mid"
                else:
                    css_class = "score-low"
                rows.append(f'<td class="{css_class}">{val*100:.1f}%</td>')
            else:
                rows.append('<td>-</td>')

        anz_20 = anz.get(20)
        if anz_20 is not None:
            rows.append(f'<td>{anz_20*100:.1f}%</td>')
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
    sweep_runs: list[dict] = []
    sweep_data_cache: dict[str, dict] = {}
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

    def get_prompt_styles():
        return sorted(set(r["prompt"] for r in sweep_runs)) or ["short"]

    def get_sweep_labels(prompt_filter=None):
        if prompt_filter:
            return [r["label"] for r in sweep_runs if r["prompt"] == prompt_filter]
        return [r["label"] for r in sweep_runs]

    # Initial load
    refresh_sweep()
    initial_prompts = get_prompt_styles()
    initial_labels = get_sweep_labels()

    METRIC_CHOICES = ["mean_best_recall", "mean_recall_1", "frac_any_nonzero"]

    with gr.Blocks(title="Citation Prediction v3 — Eval Viewer") as app:
        gr.Markdown("# Citation Prediction v3 — Eval Viewer")
        gr.Markdown("Recall-based evaluation: predict all citations in a Related Work subsection.")

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
                        label="Prompt Style", scale=2,
                    )
                    sweep_metric_dd = gr.Dropdown(
                        choices=METRIC_CHOICES,
                        value="mean_best_recall",
                        label="Metric", scale=2,
                    )

                sweep_heatmap = gr.HTML(label="Heatmap")

                def on_sweep_refresh():
                    refresh_sweep()
                    prompts = get_prompt_styles()
                    default_prompt = prompts[0] if prompts else None
                    hm = build_heatmap_html(sweep_runs, sweep_data_cache,
                                            default_prompt or "short", "mean_best_recall")
                    return (
                        gr.update(choices=prompts, value=default_prompt),
                        hm,
                    )

                def on_sweep_change(prompt, metric):
                    if not prompt or not metric:
                        return "<p>Select prompt style and metric.</p>"
                    return build_heatmap_html(sweep_runs, sweep_data_cache, prompt, metric)

                sweep_refresh.click(fn=on_sweep_refresh, outputs=[sweep_prompt_dd, sweep_heatmap])
                sweep_prompt_dd.change(fn=on_sweep_change, inputs=[sweep_prompt_dd, sweep_metric_dd], outputs=[sweep_heatmap])
                sweep_metric_dd.change(fn=on_sweep_change, inputs=[sweep_prompt_dd, sweep_metric_dd], outputs=[sweep_heatmap])
                app.load(fn=on_sweep_change, inputs=[sweep_prompt_dd, sweep_metric_dd], outputs=[sweep_heatmap])

            # ================================================================
            # Tab 2: Best-of-k Curves
            # ================================================================
            with gr.Tab("Best-of-k Curves"):
                with gr.Row():
                    curve_refresh = gr.Button("Refresh", variant="secondary", scale=1)
                    curve_type_dd = gr.Dropdown(
                        choices=["best_of_k", "any_nonzero_at_k"],
                        value="best_of_k",
                        label="Curve Type", scale=1,
                    )
                    curve_select = gr.CheckboxGroup(
                        choices=initial_labels,
                        value=initial_labels[:5],
                        label="Select setups to compare",
                    )

                curve_plot = gr.Image(label="Curves", type="filepath")
                curve_table = gr.HTML(label="Metrics Table")

                def on_curve_refresh():
                    refresh_sweep()
                    labels = get_sweep_labels()
                    selected = labels[:5]
                    plot_path = build_best_of_k_plot(sweep_runs, sweep_data_cache, selected)
                    table_html = build_metrics_table(sweep_runs, sweep_data_cache, selected)
                    return (
                        gr.update(choices=labels, value=selected),
                        plot_path,
                        table_html,
                    )

                def on_curve_update(selected, curve_type):
                    if not selected:
                        return None, "<p>Select setups to compare.</p>"
                    plot_path = build_best_of_k_plot(sweep_runs, sweep_data_cache, selected, curve_type)
                    table_html = build_metrics_table(sweep_runs, sweep_data_cache, selected)
                    return plot_path, table_html

                curve_refresh.click(fn=on_curve_refresh, outputs=[curve_select, curve_plot, curve_table])
                curve_select.change(fn=on_curve_update, inputs=[curve_select, curve_type_dd], outputs=[curve_plot, curve_table])
                curve_type_dd.change(fn=on_curve_update, inputs=[curve_select, curve_type_dd], outputs=[curve_plot, curve_table])
                app.load(fn=on_curve_update, inputs=[curve_select, curve_type_dd], outputs=[curve_plot, curve_table])

            # ================================================================
            # Tab 3: Recall Distribution
            # ================================================================
            with gr.Tab("Recall Distribution"):
                with gr.Row():
                    dist_refresh = gr.Button("Refresh", variant="secondary", scale=1)
                    dist_select = gr.CheckboxGroup(
                        choices=initial_labels,
                        value=initial_labels[:4],
                        label="Select setups to compare",
                    )

                dist_plot = gr.Image(label="Best-of-20 Recall Distribution", type="filepath")

                def on_dist_refresh():
                    refresh_sweep()
                    labels = get_sweep_labels()
                    selected = labels[:4]
                    plot_path = build_recall_distribution_plot(sweep_runs, sweep_data_cache, selected)
                    return gr.update(choices=labels, value=selected), plot_path

                def on_dist_update(selected):
                    if not selected:
                        return None
                    return build_recall_distribution_plot(sweep_runs, sweep_data_cache, selected)

                dist_refresh.click(fn=on_dist_refresh, outputs=[dist_select, dist_plot])
                dist_select.change(fn=on_dist_update, inputs=[dist_select], outputs=[dist_plot])
                app.load(fn=on_dist_update, inputs=[dist_select], outputs=[dist_plot])

            # ================================================================
            # Tab 4: Browse Trajectories
            # ================================================================
            with gr.Tab("Browse"):
                with gr.Row():
                    browse_refresh = gr.Button("Refresh", variant="secondary", scale=1)
                    browse_setup_dd = gr.Dropdown(
                        choices=initial_labels,
                        value=initial_labels[0] if initial_labels else None,
                        label="Setup", scale=2,
                    )
                    browse_filter_dd = gr.Dropdown(
                        choices=["All", "Any Recall > 0", "All Recall = 0", "Best Recall > 50%"],
                        value="All",
                        label="Filter", scale=1,
                    )
                    browse_sample_dd = gr.Dropdown(
                        choices=["Best Sample", "Sample 0"],
                        value="Best Sample",
                        label="Sample", scale=1,
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
                        gr.update(choices=["Best Sample", "Sample 0"], value="Best Sample"),
                        "<p>No sweep results found.</p>",
                    )

                def _load_browse(setup, score_filter):
                    nonlocal browse_filtered
                    if not setup:
                        browse_filtered = []
                        return ("0 prompts",
                                gr.update(maximum=0, value=0),
                                gr.update(choices=["Best Sample"], value="Best Sample"),
                                "<p>Select a setup.</p>")

                    run, data = _get_browse_data(setup)
                    if data is None:
                        browse_filtered = []
                        return ("0 prompts",
                                gr.update(maximum=0, value=0),
                                gr.update(choices=["Best Sample"], value="Best Sample"),
                                "<p>No data for this setup.</p>")

                    prompts = data.get("prompts", [])
                    n_samples = data.get("n_samples", 1)

                    sample_choices = ["Best Sample"] + [f"Sample {i}" for i in range(n_samples)]
                    sample_value = sample_choices[0]

                    # Filter
                    if score_filter == "Any Recall > 0":
                        browse_filtered = [i for i, p in enumerate(prompts) if p["any_nonzero"]]
                    elif score_filter == "All Recall = 0":
                        browse_filtered = [i for i, p in enumerate(prompts) if not p["any_nonzero"]]
                    elif score_filter == "Best Recall > 50%":
                        browse_filtered = [i for i, p in enumerate(prompts) if p["best_recall"] > 0.5]
                    else:
                        browse_filtered = list(range(len(prompts)))

                    total = len(browse_filtered)
                    n_any = sum(1 for p in prompts if p["any_nonzero"])
                    mean_best = sum(p["best_recall"] for p in prompts) / len(prompts) * 100 if prompts else 0

                    stats = (f"{total} / {len(prompts)} prompts | "
                             f"{n_any} with recall>0 ({n_any/len(prompts)*100:.0f}%) | "
                             f"mean best recall: {mean_best:.1f}%")
                    if total == 0:
                        return (stats,
                                gr.update(maximum=0, value=0),
                                gr.update(choices=sample_choices, value=sample_value),
                                "<p>No matching examples.</p>")

                    html_content = _render_browse_example(data, run, browse_filtered[0], "Best Sample")
                    return (stats,
                            gr.update(maximum=total - 1, value=0),
                            gr.update(choices=sample_choices, value=sample_value),
                            html_content)

                def _render_browse_example(data, run, prompt_idx, sample_str):
                    prompts = data.get("prompts", [])
                    if prompt_idx >= len(prompts):
                        return "<p>Index out of range.</p>"

                    prompt_data = prompts[prompt_idx]
                    samples = prompt_data.get("samples", [])
                    if not samples:
                        return "<p>No sample data available.</p>"

                    if sample_str == "Best Sample":
                        sample_idx = max(range(len(samples)), key=lambda i: samples[i]["recall"])
                    else:
                        sample_idx = int(re.search(r"\d+", sample_str or "0").group()) if sample_str else 0
                    sample_idx = max(0, min(sample_idx, len(samples) - 1))

                    sample = samples[sample_idx]
                    n_searches = len(re.findall(r"<search>", sample["output_response"]))

                    badges = [
                        f"m={run['m']}" if run else "",
                        f"n={run['n']}" if run else "",
                        f"sample {sample_idx}/{len(samples)}",
                        f"{prompt_data['num_nonzero']}/{len(samples)} nonzero",
                        f"{n_searches} searches",
                        f"best recall: {prompt_data['best_recall']*100:.0f}%",
                    ]
                    badges = [b for b in badges if b]

                    metrics = {
                        "n_correct": sample["n_correct"],
                        "n_ground_truth": sample["n_ground_truth"],
                        "n_predicted": sample["n_predicted"],
                    }

                    return render_trajectory_html(
                        input_prompt=prompt_data["input_prompt"],
                        output_response=sample["output_response"],
                        targets=prompt_data["targets"],
                        recall=sample["recall"],
                        metrics=metrics,
                        label=f"Prompt {prompt_idx}",
                        extra_badges=badges,
                    )

                def on_browse_change(setup, score_filter):
                    return _load_browse(setup, score_filter)

                def on_browse_navigate(idx, setup, sample_str):
                    if not browse_filtered or not setup:
                        return "<p>No data.</p>"
                    idx = max(0, min(int(idx), len(browse_filtered) - 1))
                    run, data = _get_browse_data(setup)
                    if data is None:
                        return "<p>No data.</p>"
                    return _render_browse_example(data, run, browse_filtered[idx], sample_str)

                def on_browse_prev(idx, setup, sample_str):
                    new_idx = max(0, int(idx) - 1)
                    return new_idx, on_browse_navigate(new_idx, setup, sample_str)

                def on_browse_next(idx, setup, sample_str):
                    max_idx = len(browse_filtered) - 1 if browse_filtered else 0
                    new_idx = min(max_idx, int(idx) + 1)
                    return new_idx, on_browse_navigate(new_idx, setup, sample_str)

                browse_outputs = [browse_stats, browse_slider, browse_sample_dd, browse_html]

                browse_refresh.click(fn=on_browse_refresh, outputs=[browse_setup_dd, *browse_outputs])
                browse_setup_dd.change(fn=on_browse_change, inputs=[browse_setup_dd, browse_filter_dd], outputs=browse_outputs)
                browse_filter_dd.change(fn=on_browse_change, inputs=[browse_setup_dd, browse_filter_dd], outputs=browse_outputs)
                browse_prev.click(fn=on_browse_prev, inputs=[browse_idx_state, browse_setup_dd, browse_sample_dd], outputs=[browse_idx_state, browse_html])
                browse_next.click(fn=on_browse_next, inputs=[browse_idx_state, browse_setup_dd, browse_sample_dd], outputs=[browse_idx_state, browse_html])
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
            # Tab 5: Summary Table
            # ================================================================
            with gr.Tab("Summary"):
                summary_refresh = gr.Button("Refresh", variant="secondary")
                summary_html = gr.HTML(label="Summary Table")

                def build_summary():
                    refresh_sweep()

                    rows = [CSS, '<table class="metrics-table">']
                    rows.append("<thead><tr>")
                    rows.append("<th>Setup</th><th>m</th><th>n</th>")
                    rows.append("<th>Recall@1</th><th>Best@5</th><th>Best@10</th><th>Best@20</th>")
                    rows.append("<th>Any>0@20</th><th>Dead%</th><th># Prompts</th>")
                    rows.append("</tr></thead><tbody>")

                    for r in sorted(sweep_runs, key=lambda x: (x["prompt"], x["m"], x["n"])):
                        data = sweep_data_cache.get(r["dir_name"])
                        if data is None:
                            continue

                        bok = data.get("best_of_k", {})
                        anz = data.get("any_nonzero_at_k", {})

                        rows.append(f'<tr><td><strong>{html.escape(r["label"])}</strong></td>')
                        rows.append(f'<td>{r["m"]}</td><td>{r["n"]}</td>')

                        # Recall@1
                        r1 = data.get("mean_recall_1", 0)
                        cls = "score-high" if r1 >= 0.15 else "score-mid" if r1 >= 0.05 else "score-low"
                        rows.append(f'<td class="{cls}">{r1*100:.1f}%</td>')

                        for k in [5, 10, 20]:
                            val = bok.get(k)
                            if val is not None:
                                cls = "score-high" if val >= 0.3 else "score-mid" if val >= 0.1 else "score-low"
                                rows.append(f'<td class="{cls}">{val*100:.1f}%</td>')
                            else:
                                rows.append('<td>-</td>')

                        anz_20 = anz.get(20, 0)
                        rows.append(f'<td>{anz_20*100:.1f}%</td>')
                        rows.append(f'<td>{data["frac_all_zero"]*100:.0f}%</td>')
                        rows.append(f'<td>{data["n_prompts"]}</td></tr>')

                    rows.append("</tbody></table>")

                    if not sweep_runs:
                        return "<p>No eval results found. Submit a sweep job and click Refresh.</p>"

                    return "\n".join(rows)

                summary_refresh.click(fn=build_summary, outputs=[summary_html])
                app.load(fn=build_summary, outputs=[summary_html])

    return app


def main():
    parser = argparse.ArgumentParser(description="Citation Prediction v3 — Eval Viewer")
    parser.add_argument(
        "--evals-dir",
        default=DEFAULT_EVALS_DIR,
        help="Directory containing sweep/ output directories",
    )
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app = create_app(args.evals_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
