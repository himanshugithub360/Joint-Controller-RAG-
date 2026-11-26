"""Stage 1 unit tests: split sampling logic (synthetic) and the frozen splits (real data)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from budgetrag.config import load_config, repo_path
from budgetrag.data import flashrag
from budgetrag.data.splits import read_all_splits, sample_test, sample_train_side

SIZES = {"train_pool": 50, "val": 10, "live_ft": 20}


def test_train_side_disjoint_and_sized():
    ids = [f"x-{i}" for i in range(100)]
    parts = sample_train_side(ids, SIZES, seed=42)
    assert {k: len(v) for k, v in parts.items()} == SIZES
    flat = [i for v in parts.values() for i in v]
    assert len(flat) == len(set(flat))


def test_sampling_is_deterministic():
    ids = [f"x-{i}" for i in range(100)]
    assert sample_train_side(ids, SIZES, 42) == sample_train_side(ids, SIZES, 42)
    assert sample_test(ids, 30, 42) == sample_test(ids, 30, 42)
    assert sample_train_side(ids, SIZES, 42) != sample_train_side(ids, SIZES, 43)


def test_too_few_questions_raises():
    with pytest.raises(ValueError):
        sample_train_side([f"x-{i}" for i in range(10)], SIZES, 42)


def test_sample_test_all_when_none_or_large():
    ids = [f"x-{i}" for i in range(5)]
    assert sample_test(ids, None, 42) == ids
    assert sample_test(ids, 99, 42) == ids


def test_flashrag_reader(tmp_path: Path):
    d = tmp_path / "musique"
    d.mkdir()
    rec = {"id": "dev_0", "question": " Q? ", "golden_answers": ["A"],
           "metadata": {"question_decomposition": [{}, {}, {}]}}
    (d / "dev.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (q,) = flashrag.load_questions("musique", "dev", tmp_path)
    assert (q.qid, q.question, q.answers, q.hops) == ("musique-dev_0", "Q?", ["A"], 3)


# ---- tests on the real frozen splits (skipped until Stage 1 has run) -------------------

CFG = load_config()
SPLITS_DIR = repo_path(CFG.paths.splits)
QUESTIONS = repo_path(CFG.paths.questions)
needs_data = pytest.mark.skipif(not QUESTIONS.exists(), reason="run scripts/s01_make_splits.py first")


@pytest.fixture(scope="module")
def splits():
    return read_all_splits(SPLITS_DIR)


@pytest.mark.data
@needs_data
def test_no_id_in_two_splits(splits):
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        assert len(ids) == len(set(ids)), f"duplicate ids inside {name}"
        for qid in ids:
            assert qid not in seen, f"{qid} in both {seen[qid]} and {name}"
            seen[qid] = name


@pytest.mark.data
@needs_data
def test_every_split_id_exists_in_raw_files(splits):
    df = pd.read_parquet(QUESTIONS)
    assert set(df.qid) == {q for ids in splits.values() for q in ids}


@pytest.mark.data
@needs_data
def test_answers_non_empty():
    df = pd.read_parquet(QUESTIONS)
    assert df["answers"].map(len).min() > 0


@pytest.mark.data
@needs_data
def test_split_sizes_match_profile(splits):
    marker = (SPLITS_DIR / "PROFILE").read_text().strip()
    cfg = load_config([f"profile={marker}"])
    n_src = len(cfg.flashrag.train_sources)
    for name, size in cfg.splits.per_source.items():
        assert len(splits[name]) == size * n_src
