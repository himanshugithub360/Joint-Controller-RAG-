"""The decision tree for one question (plan Stage 6; first used by the Stage 4 pilot).

A node is the evidence after a sequence of RETRIEVE(K) choices and is named by its path of
K values: "" is the root (no retrieval), then "3", "5", "3-5", ... down to depth 3. With
K in {3, 5} that is 1 + 2 + 4 + 8 = 15 nodes.

How a node's children get their passages:
  root      searches the question itself (its top 10 come from the Stage 2 cache);
  deeper    the helper writes one sub-query from the node's evidence; both children share
            that sub-query and its top 10.
Passages already in the node's evidence are dropped from the top 10, then the list is cut
to K and appended to the evidence (so a node's evidence never repeats a passage).
"""

from __future__ import annotations


def child_path(path: str, k: int) -> str:
    return f"{path}-{k}" if path else str(int(k))


def depth(path: str) -> int:
    return path.count("-") + 1 if path else 0


def parent(path: str) -> str | None:
    if not path:
        return None
    return path.rsplit("-", 1)[0] if "-" in path else ""


def paths(k_choices, max_depth: int = 3) -> list[str]:
    """Every node path, level by level: ["", "3", "5", "3-3", "3-5", ...]."""
    out, level = [""], [""]
    for _ in range(max_depth):
        level = [child_path(p, k) for p in level for k in k_choices]
        out += level
    return out


def extend(evidence: list[int], hits: list[int], k: int) -> list[int]:
    """Evidence of a child: the parent's evidence plus the first k new passages from hits."""
    seen = set(evidence)
    new = [d for d in hits if d not in seen][:k]
    return list(evidence) + new


def node_id(path: str) -> str:
    """Path as written into job ids and file names ("root" for the root)."""
    return path or "root"


__all__ = ["child_path", "depth", "parent", "paths", "extend", "node_id"]
