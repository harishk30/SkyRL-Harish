import re

ARXIV_ID_RE = re.compile(r"\[arxiv:(\d{4}\.\d{4,5})\]")


def extract_arxiv_ids(text: str) -> set[str]:
    """Extract arxiv IDs from retriever output text.

    Matches patterns like [arxiv:1706.03762] that are prepended to corpus
    documents by build_arxiv_corpus.py.
    """
    return set(ARXIV_ID_RE.findall(text))


def check_citation_match(retrieved_ids: set[str], target_id: str) -> bool:
    """Check if the target arxiv ID appears in the retrieved set."""
    return target_id in retrieved_ids
