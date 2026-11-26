"""Stage 2 unit tests: chunked flat search vs FAISS, memo, K-slicing, corpus access.

FAISS runs in a subprocess: on Windows the faiss-cpu and torch wheels ship different
OpenMP runtimes that abort when both start in one process (OMP Error #15). Project code
never imports faiss; it is only the reference here.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from budgetrag.config import load_config, repo_path
from budgetrag.retrieval.corpus import Corpus, prepare
from budgetrag.retrieval.flat_index import FlatIndexFile
from budgetrag.retrieval.memo import RetrievalMemo

FAISS_SCRIPT = """
import sys, numpy as np, faiss
d = np.load(sys.argv[1])
index = faiss.IndexFlatIP(d["xb"].shape[1]); index.add(d["xb"])
faiss.write_index(index, sys.argv[2])
s, i = faiss.read_index(sys.argv[2]).search(d["xq"], 10)
np.savez(sys.argv[3], s=s, i=i)
"""


def faiss_reference(tmp_path: Path, xb: np.ndarray, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Write tmp_path/whole.index with FAISS and return FAISS's own top-10 for xq."""
    np.savez(tmp_path / "in.npz", xb=xb, xq=xq)
    subprocess.run([sys.executable, "-c", FAISS_SCRIPT, str(tmp_path / "in.npz"),
                    str(tmp_path / "whole.index"), str(tmp_path / "ref.npz")], check=True)
    ref = np.load(tmp_path / "ref.npz")
    return ref["s"], ref["i"]


def write_split_index(tmp_path: Path, n: int = 5000, d: int = 32, cut: int = 100_003,
                      xq: np.ndarray | None = None):
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((n, d)).astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    if xq is None:
        xq = np.random.default_rng(1).standard_normal((40, d)).astype(np.float32)
    ref = faiss_reference(tmp_path, xb, xq)
    whole = tmp_path / "whole.index"
    raw = whole.read_bytes()
    parts = [tmp_path / "part_aa", tmp_path / "part_ab"]
    parts[0].write_bytes(raw[:cut])   # cut falls in the middle of a vector, like Search-R1's parts
    parts[1].write_bytes(raw[cut:])
    return xb, xq, parts, ref


def test_header_parsed(tmp_path):
    xb, _, parts, _ = write_split_index(tmp_path)
    idx = FlatIndexFile(parts)
    assert (idx.ntotal, idx.d) == xb.shape
    np.testing.assert_array_equal(idx.rows(0, 10), xb[:10])
    np.testing.assert_array_equal(idx.rows(3200, 3300), xb[3200:3300])  # spans the part boundary
    idx.close()


@pytest.mark.parametrize("chunk_rows,query_block", [(700, 7), (5000, 2048), (1, 50)])
def test_chunked_search_equals_faiss(tmp_path, chunk_rows, query_block):
    _, xq, parts, (ref_s, ref_i) = write_split_index(tmp_path)
    idx = FlatIndexFile(parts)
    s, i = idx.search(xq, 10, chunk_rows=chunk_rows, query_block=query_block)
    idx.close()
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_allclose(s, ref_s, rtol=1e-5, atol=1e-5)


def test_k_slicing_matches_smaller_search(tmp_path):
    xq = np.random.default_rng(2).standard_normal((100, 32)).astype(np.float32)
    _, _, parts, _ = write_split_index(tmp_path, xq=xq)
    idx = FlatIndexFile(parts)
    _, i10 = idx.search(xq, 10, chunk_rows=999)
    for k in (3, 5):
        _, ik = idx.search(xq, k, chunk_rows=999)
        np.testing.assert_array_equal(i10[:, :k], ik)
    idx.close()


def test_memo_roundtrip_and_slicing(tmp_path):
    memo = RetrievalMemo(tmp_path / "m.sqlite")
    memo.put_many("e5", 10, {"q1": (list(range(10)), [1.0 - j / 10 for j in range(10)])})
    assert memo.get_many("e5", ["q1", "q2"], 10)["q1"][0] == list(range(10))
    assert memo.get_many("e5", ["q1"], 3)["q1"][0] == [0, 1, 2]
    assert "q2" not in memo.get_many("e5", ["q2"], 3)
    assert memo.get_many("bm25", ["q1"], 3) == {}
    assert memo.export(tmp_path / "m.parquet") == 1
    memo.close()


def test_corpus_offsets(tmp_path):
    recs = [{"id": str(i), "contents": f'"Title {i}"\nText number {i}.'} for i in range(50)]
    gz = tmp_path / "c.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    n = prepare(gz, tmp_path / "c.jsonl", tmp_path / "c.npy", progress_every=0)
    assert n == 50
    c = Corpus(tmp_path / "c.jsonl", tmp_path / "c.npy")
    got = c.get([42, 0, 7])
    c.close()
    assert [g["title"] for g in got] == ['"Title 42"', '"Title 0"', '"Title 7"']
    assert got[0]["text"] == "Text number 42."


def test_corpus_from_tar_gz_like_search_r1(tmp_path):
    """Search-R1's wiki-18.jsonl.gz is a gzipped tar; ids must still equal line numbers."""
    import io
    import tarfile
    payload = "".join(json.dumps({"id": str(i), "contents": f'"T{i}"\nx{i}'}) + "\n" for i in range(30)).encode()
    gz = tmp_path / "wiki.jsonl.gz"
    with tarfile.open(gz, "w:gz") as tar:
        info = tarfile.TarInfo("data00/wiki_dump.jsonl")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    assert prepare(gz, tmp_path / "w.jsonl", tmp_path / "w.npy", progress_every=0, check_every=1) == 30
    c = Corpus(tmp_path / "w.jsonl", tmp_path / "w.npy")
    assert [g["title"] for g in c.get([0, 29])] == ['"T0"', '"T29"']
    c.close()


def test_corpus_rejects_misnumbered_ids(tmp_path):
    gz = tmp_path / "bad.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": "5", "contents": '"A"\nb'}) + "\n")
    with pytest.raises(ValueError):
        prepare(gz, tmp_path / "b.jsonl", tmp_path / "b.npy", progress_every=0, check_every=1)


# ---- on the real cache (skipped until Stage 2 has run) ---------------------------------

CFG = load_config()
CACHE = repo_path(CFG.paths.retrieval_cache)
needs_cache = pytest.mark.skipif(not CACHE.exists(), reason="run scripts/s02_build_retrieval_cache.py first")


@pytest.mark.data
@needs_cache
def test_every_question_has_10_passages():
    cache = pd.read_parquet(CACHE)
    q = pd.read_parquet(repo_path(CFG.paths.questions))
    have = set(cache[cache.retriever == "e5"]["query"])
    assert set(q["question"]) <= have
    assert cache["doc_ids"].map(len).min() == 10
    passages = pd.read_parquet(repo_path(CFG.paths.passages))
    assert {d for ids in cache["doc_ids"] for d in ids} <= set(passages["doc_id"])


@pytest.mark.data
@needs_cache
def test_memo_returns_identical_ids_twice():
    from budgetrag.retrieval.client import E5Retriever
    r = E5Retriever([], repo_path(CFG.paths.retrieval_memo))  # memo hits never touch the index
    qs = pd.read_parquet(repo_path(CFG.paths.questions))["question"].sample(100, random_state=0).tolist()
    assert r.retrieve(qs, k=10) == r.retrieve(qs, k=10)
    assert [h[:3] for h in r.retrieve(qs, k=10)] == r.retrieve(qs, k=3)
    r.close()
