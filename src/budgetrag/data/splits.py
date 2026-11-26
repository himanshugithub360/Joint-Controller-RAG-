"""Deterministic, disjoint split sampling and the frozen ID list files.

Train-side splits (train_pool, val, live_ft) are consecutive slices of ONE seeded
permutation of each source's train set, so they can never overlap. Test splits are a
seeded sample of each dataset's test split (or all of it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TRAIN_SIDE = ("train_pool", "val", "live_ft")


def sample_train_side(ids: list[str], sizes: dict[str, int], seed: int) -> dict[str, list[str]]:
    """Slice one permutation of `ids` into train_pool, val and live_ft, in that order."""
    need = sum(sizes[name] for name in TRAIN_SIDE)
    if need > len(ids):
        raise ValueError(f"need {need} questions but the source has only {len(ids)}")
    order = np.random.default_rng(seed).permutation(len(ids))
    out, start = {}, 0
    for name in TRAIN_SIDE:
        n = sizes[name]
        out[name] = [ids[i] for i in order[start : start + n]]
        start += n
    return out


def sample_test(ids: list[str], n: int | None, seed: int) -> list[str]:
    """Seeded sample of n ids, kept in file order; all ids when n is None or too large."""
    if n is None or n >= len(ids):
        return list(ids)
    keep = np.sort(np.random.default_rng(seed).choice(len(ids), size=n, replace=False))
    return [ids[i] for i in keep]


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def read_ids(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_all_splits(splits_dir: Path) -> dict[str, list[str]]:
    """{split name: ids} for every <name>.ids.txt in splits_dir."""
    return {p.name.removesuffix(".ids.txt"): read_ids(p) for p in sorted(splits_dir.glob("*.ids.txt"))}
