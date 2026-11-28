"""Stage 4/6 tree helpers and the Stage 4 job builder."""

from __future__ import annotations

import gzip
import json

import pytest

from budgetrag.cache import tree
from budgetrag.config import load_config, repo_path


def test_paths_count_and_order():
    ps = tree.paths([3, 5])
    assert len(ps) == 15 and ps[:3] == ["", "3", "5"] and ps[3:7] == ["3-3", "3-5", "5-3", "5-5"]
    assert tree.paths([3]) == ["", "3", "3-3", "3-3-3"]
    assert [tree.depth(p) for p in ps] == [0] + [1] * 2 + [2] * 4 + [3] * 8


def test_parent_and_child():
    assert tree.parent("") is None and tree.parent("3") == "" and tree.parent("3-5-3") == "3-5"
    assert all(tree.parent(tree.child_path(p, k)) == p for p in tree.paths([3, 5], 2) for k in (3, 5))
    assert tree.node_id("") == "root" and tree.node_id("5-3") == "5-3"


def test_extend_skips_seen_passages():
    assert tree.extend([], [9, 8, 7, 6], 3) == [9, 8, 7]
    assert tree.extend([9, 8, 7], [8, 1, 9, 2, 3, 4], 3) == [9, 8, 7, 1, 2, 3]
    assert tree.extend([1], [1, 2], 5) == [1, 2]                 # fewer new passages than K


def test_depth1_is_question_topk():
    hits = list(range(10))
    assert tree.extend([], hits, 3) == hits[:3] and tree.extend([], hits, 5) == hits[:5]


CFG = load_config()
R1 = repo_path(CFG.llm.stage4.jobs_dir) / "stage4_r1.jsonl.gz"


@pytest.mark.data
@pytest.mark.skipif(not R1.exists(), reason="run scripts/s04_pilot.py round 1 first")
def test_round1_jobs():
    with gzip.open(R1, "rt", encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f]
    meta = json.loads(R1.with_name("stage4_r1.meta.json").read_text(encoding="utf-8"))
    assert meta["run_tiers"] == [meta["helper"]] and meta["models"][meta["helper"]] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    val = [j for j in jobs if j["group"] == "val"]
    pil = [j for j in jobs if j["group"] == "pilot"]
    n_val, n_pil = len({j["qid"] for j in val}), len({j["qid"] for j in pil})
    assert n_pil == 2 * CFG.llm.stage4.pilot_per_source
    assert len(val) == n_val * 3 and len(pil) == n_pil * 5   # tentatives (root + depth 1) + depth-1 sub-queries
    assert all(len(j["doc_ids"]) == int(j["path"]) for j in jobs if j["path"])
    assert all(j["prompt"].endswith("<|im_start|>assistant\n") for j in jobs)
