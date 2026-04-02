#!/usr/bin/env python3
"""MCP STDIO server: exposes arxiv search as a single tool for Codex.

Wraps the retriever HTTP API as a `search(query)` MCP tool. Each Codex
invocation starts a fresh server process, so the search counter resets
per query.

Configure in ~/.codex/config.toml:

    [mcp_servers.arxiv_search]
    command = "/path/to/python"
    args = ["/path/to/mcp_search_server.py"]
    enabled_tools = ["search"]
    tool_timeout_sec = 45
    startup_timeout_sec = 15

    [mcp_servers.arxiv_search.env]
    RETRIEVER_URL = "http://localhost:8000/retrieve"
    TOPK = "5"
    MAX_SEARCHES = "15"
"""

import json
import os
import sys

import requests
from mcp.server.fastmcp import FastMCP

RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://localhost:8000/retrieve")
TOPK = int(os.environ.get("TOPK", "5"))
MAX_SEARCHES = int(os.environ.get("MAX_SEARCHES", "15"))

_search_count = 0

mcp = FastMCP("arxiv-search")


@mcp.tool()
def search(query: str) -> str:
    """Search arxiv papers by semantic query.

    Returns top-k papers from an arxiv database. Each result includes:
    - [arxiv:ID] prefix with the paper's arxiv identifier
    - Paper title in quotes, followed by authors and abstract

    Use descriptive topic keywords (method names, task descriptions,
    dataset names). Do NOT search for author names or arxiv IDs.
    """
    global _search_count
    _search_count += 1

    if _search_count > MAX_SEARCHES:
        return (
            f"Search limit reached ({MAX_SEARCHES} searches used). "
            "You must finalize your citations and write <done></done> now."
        )

    remaining = MAX_SEARCHES - _search_count

    try:
        resp = requests.post(
            RETRIEVER_URL,
            json={"query": query, "topk": TOPK, "return_scores": True},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        raw_results = data.get("result", [])
        if not raw_results:
            return f"No results found. Try different search terms. ({remaining} searches remaining)"

        parts: list[str] = []
        for retrieval in raw_results:
            for idx, doc_item in enumerate(retrieval):
                content = doc_item["document"]["contents"].strip()
                parts.append(f"Doc {idx + 1}: {content}")

        result = "\n".join(parts)
        result += f"\n\n{remaining} search(es) remaining."
        return result

    except requests.Timeout:
        _search_count -= 1  # don't penalize for timeout
        return "Search timed out. Try again."
    except Exception as e:
        _search_count -= 1
        return f"Search error: {e}. Try again."


if __name__ == "__main__":
    mcp.run()
