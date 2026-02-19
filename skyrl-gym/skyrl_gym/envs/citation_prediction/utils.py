"""Citation prediction utilities — arxiv ID scoring."""

import re


def normalize_arxiv_id(s: str) -> str:
    """Normalize an arxiv ID by extracting the core YYMM.NNNNN pattern.

    Handles any prefix (arXiv:, cs:, [arxiv:...], URLs) and version suffixes.

    Examples:
        "1909.03093"              -> "1909.03093"
        "arXiv:1909.03093"       -> "1909.03093"
        "[arxiv:1909.03093]"     -> "1909.03093"
        "cs:1909.03093v2"        -> "1909.03093"
        "hep-th/1909.03093v1"   -> "1909.03093"
    """
    s = s.strip()
    # Try to extract the core arxiv ID pattern: YYMM.NNNNN (4 digits, dot, 4-5 digits)
    match = re.search(r"(\d{4}\.\d{4,5})", s)
    if match:
        return match.group(1)
    # Fallback: strip common prefixes/suffixes for older-format IDs
    s = s.strip("[]")
    s = re.sub(r"^[A-Za-z-]+[:/]\s*", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s.strip()


def extract_answer(solution_str: str) -> str | None:
    """Extract the last <answer>...</answer> content from a string."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def compute_arxiv_score(solution_str: str, ground_truth: dict) -> float:
    """Score an arxiv ID answer against ground truth.

    Args:
        solution_str: Full chat history concatenated as a string.
        ground_truth: Dict with "target" key containing the ground-truth arxiv ID.

    Returns:
        1.0 if the extracted answer matches the target after normalization, 0.0 otherwise.
    """
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0

    predicted = normalize_arxiv_id(answer)
    target = normalize_arxiv_id(ground_truth["target"])

    return 1.0 if predicted == target else 0.0
