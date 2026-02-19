"""Tree data structures for hierarchical subsection decomposition.

Each subsection becomes a tree where the root is the original subsection
and children are finer-grained subtopics produced by Gemini splitting.
Leaf nodes become tractable training tasks for GRPO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TreeNode:
    """A node in the subsection decomposition tree."""
    node_id: str                          # "paperID_sub1_child0_child1"
    heading: str                          # subtopic description
    parent_id: Optional[str]
    children: list[str] = field(default_factory=list)  # child node IDs
    citation_ids: list[str] = field(default_factory=list)  # arxiv IDs at this node
    paragraph_text: str = ""              # text content
    depth: int = 0
    is_leaf: bool = True
    # Eval results (populated during adaptive loop)
    best_of_k_recall: Optional[float] = None
    mean_recall: Optional[float] = None
    # Metadata from splitting
    split_reverted: bool = False          # True if split was tried but reverted


@dataclass
class SubsectionTree:
    """A decomposition tree for one Related Work subsection."""
    paper_id: str
    title: str
    abstract: str
    introduction: str
    root_heading: str
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    root_id: str = ""
    # Full Related Work section text (all subsections with headings)
    full_related_work_text: str = ""
    # {arxiv_id: [sentence_1, ...]} — per-citation sentence context
    citation_sentence_map: dict[str, list[str]] = field(default_factory=dict)
    # {arxiv_id: {title, authors, abstract}} — from arxiv corpus
    cited_papers: dict[str, dict] = field(default_factory=dict)

    def get_root(self) -> TreeNode:
        return self.nodes[self.root_id]

    def get_leaves(self) -> list[TreeNode]:
        """Return all leaf nodes in the tree."""
        return [n for n in self.nodes.values() if n.is_leaf]

    def get_node(self, node_id: str) -> TreeNode:
        return self.nodes[node_id]

    def add_node(self, node: TreeNode) -> None:
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node and clean up parent's children list."""
        node = self.nodes.get(node_id)
        if node is None:
            return
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            parent.children = [c for c in parent.children if c != node_id]
        del self.nodes[node_id]

    def total_citations(self) -> int:
        """Count total citations across all leaf nodes."""
        return sum(len(n.citation_ids) for n in self.get_leaves())

    def validate(self) -> list[str]:
        """Validate tree integrity. Returns list of issues found."""
        issues = []

        # Check root exists
        if self.root_id not in self.nodes:
            issues.append(f"Root node '{self.root_id}' not found in nodes")
            return issues

        # Check all children exist
        for node in self.nodes.values():
            for child_id in node.children:
                if child_id not in self.nodes:
                    issues.append(f"Node '{node.node_id}' references missing child '{child_id}'")

        # Check parent references
        for node in self.nodes.values():
            if node.parent_id and node.parent_id not in self.nodes:
                issues.append(f"Node '{node.node_id}' references missing parent '{node.parent_id}'")

        # Check leaf citations cover root citations
        root = self.nodes[self.root_id]
        root_cites = set(root.citation_ids)
        leaf_cites = set()
        for leaf in self.get_leaves():
            leaf_cites.update(leaf.citation_ids)
        missing = root_cites - leaf_cites
        if missing:
            issues.append(f"Citations not covered by leaves: {missing}")

        extra = leaf_cites - root_cites
        if extra:
            issues.append(f"Leaf citations not in root: {extra}")

        # Check is_leaf consistency
        for node in self.nodes.values():
            if node.is_leaf and node.children:
                issues.append(f"Node '{node.node_id}' is marked as leaf but has children")
            if not node.is_leaf and not node.children:
                issues.append(f"Node '{node.node_id}' is not leaf but has no children")

        return issues

    def to_dict(self) -> dict:
        """Serialize tree to a dict."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "introduction": self.introduction,
            "root_heading": self.root_heading,
            "root_id": self.root_id,
            "full_related_work_text": self.full_related_work_text,
            "citation_sentence_map": self.citation_sentence_map,
            "cited_papers": self.cited_papers,
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> SubsectionTree:
        """Deserialize tree from a dict."""
        tree = cls(
            paper_id=d["paper_id"],
            title=d["title"],
            abstract=d["abstract"],
            introduction=d["introduction"],
            root_heading=d["root_heading"],
            root_id=d["root_id"],
            full_related_work_text=d.get("full_related_work_text", ""),
            citation_sentence_map=d.get("citation_sentence_map", {}),
            cited_papers=d.get("cited_papers", {}),
        )
        for nid, node_dict in d["nodes"].items():
            tree.nodes[nid] = TreeNode(**node_dict)
        return tree

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> SubsectionTree:
        return cls.from_dict(json.loads(s))


def make_root_node(
    paper_id: str,
    subsection_idx: int,
    heading: str,
    citation_ids: list[str],
    paragraph_text: str,
) -> tuple[str, TreeNode]:
    """Create a root node for a subsection tree.

    Returns (node_id, TreeNode).
    """
    node_id = f"{paper_id}_sub{subsection_idx}"
    node = TreeNode(
        node_id=node_id,
        heading=heading,
        parent_id=None,
        citation_ids=list(citation_ids),
        paragraph_text=paragraph_text,
        depth=0,
        is_leaf=True,
    )
    return node_id, node


def make_child_node(
    parent: TreeNode,
    child_idx: int,
    heading: str,
    citation_ids: list[str],
    paragraph_text: str = "",
) -> TreeNode:
    """Create a child node under the given parent."""
    node_id = f"{parent.node_id}_c{child_idx}"
    return TreeNode(
        node_id=node_id,
        heading=heading,
        parent_id=parent.node_id,
        citation_ids=list(citation_ids),
        paragraph_text=paragraph_text,
        depth=parent.depth + 1,
        is_leaf=True,
    )


def save_trees(trees: list[SubsectionTree], path: str) -> None:
    """Save a list of trees to a JSON file."""
    with open(path, "w") as f:
        json.dump([t.to_dict() for t in trees], f, ensure_ascii=False, indent=2)


def load_trees(path: str) -> list[SubsectionTree]:
    """Load a list of trees from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [SubsectionTree.from_dict(d) for d in data]


def save_checkpoint(tree: SubsectionTree, path: str) -> None:
    """Append a single tree to a JSONL checkpoint file."""
    with open(path, "a") as f:
        f.write(tree.to_json() + "\n")


def load_checkpoint(path: str) -> list[SubsectionTree]:
    """Load trees from a JSONL checkpoint file."""
    trees = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(SubsectionTree.from_json(line))
    return trees


def get_checkpoint_keys(path: str) -> set[str]:
    """Get set of (paper_id, root_heading) keys already in checkpoint."""
    keys = set()
    try:
        trees = load_checkpoint(path)
        for tree in trees:
            keys.add((tree.paper_id, tree.root_heading))
    except FileNotFoundError:
        pass
    return keys
