"""Stage 2: E5 retrieval cache for every question in every split.

Needs Search-R1's corpus and index in data/wiki18/ first:
    python third_party/Search-R1/scripts/download.py --save_path data/wiki18

    python scripts/s02_build_retrieval_cache.py

Steps
  1. Decompress the corpus once and build the passage offset table.
  2. Retrieve the top 10 E5 passages for every split question plus a sanity query, in one
     exact pass over the 60 GB index (CPU).
  3. Write data/retrieval_cache.parquet (query -> 10 ids + scores) and data/passages.parquet
     (doc_id -> title, text) for every passage retrieved.
  4. Write reports/STAGE_2_REPORT.md: timing, sanity check, answer recall@k per dataset.
"""

from __future__ import annotations

import re
import string
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budgetrag.config import load_config, repo_path, save_run_config  # noqa: E402
from budgetrag.retrieval import corpus as corpus_mod  # noqa: E402
from budgetrag.retrieval.client import E5Retriever  # noqa: E402

SANITY = "Who wrote Hamlet?"


def normalize(s: str) -> str:
    """SQuAD-style normalisation (lower, drop punctuation and articles, squeeze spaces)."""
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def has_answer(passages: list[dict], answers: list[str]) -> bool:
    text = " " + normalize(" ".join(p["title"] + " " + p["text"] for p in passages)) + " "
    return any(f" {normalize(a)} " in text for a in answers if normalize(a))


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    wiki = repo_path(cfg.paths.wiki18)
    parts = [wiki / p for p in cfg.paths.index_parts]
    missing = [p for p in parts + [repo_path(cfg.paths.corpus_gz)] if not p.exists()]
    if missing and not repo_path(cfg.paths.corpus_jsonl).exists():
        sys.exit("missing " + ", ".join(map(str, missing)) + "; run Search-R1's download.py first")

    print("1/4 preparing corpus")
    t = time.time()
    n_docs = corpus_mod.prepare(repo_path(cfg.paths.corpus_gz), repo_path(cfg.paths.corpus_jsonl),
                                repo_path(cfg.paths.corpus_offsets))
    print(f"    {n_docs:,} passages ({time.time() - t:.0f} s)")
    corpus = corpus_mod.Corpus(repo_path(cfg.paths.corpus_jsonl), repo_path(cfg.paths.corpus_offsets))

    retriever = E5Retriever(parts, repo_path(cfg.paths.retrieval_memo), cfg.models.retriever,
                            store_k=cfg.retrieval.topk)
    if retriever.index.ntotal != n_docs:
        sys.exit(f"index has {retriever.index.ntotal:,} vectors but the corpus has {n_docs:,} passages")

    questions = pd.read_parquet(repo_path(cfg.paths.questions))
    queries = [SANITY] + questions["question"].tolist()

    print(f"2/4 retrieving top-{cfg.retrieval.topk} for {len(set(queries)):,} queries")
    t = time.time()
    last = [0.0]

    def progress(done: int, total: int) -> None:
        if time.time() - last[0] > 60 or done == total:
            last[0] = time.time()
            print(f"    {done / total:6.1%} of index scanned, {time.time() - t:.0f} s", flush=True)

    hits = retriever.retrieve(queries, k=cfg.retrieval.topk, progress=progress)
    search_s = time.time() - t

    print("3/4 writing cache and passages")
    n_cache = retriever.memo.export(repo_path(cfg.paths.retrieval_cache))
    doc_ids = sorted({d for h in hits for d, _ in h})
    passages = pd.DataFrame(corpus.get(doc_ids))
    passages.to_parquet(repo_path(cfg.paths.passages), index=False)
    by_id = passages.set_index("doc_id").to_dict("index")

    print("4/4 report")
    sanity_top = by_id[hits[0][0][0]]
    sanity_ok = "shakespeare" in (sanity_top["title"] + sanity_top["text"]).lower()
    rows = []
    for (_, q), h in zip(questions.iterrows(), hits[1:]):
        ps = [by_id[d] for d, _ in h]
        rows.append({"split": q.split, "dataset": q.dataset,
                     **{f"recall@{k}": has_answer(ps[:k], list(q.answers)) for k in (1, 3, 5, 10)}})
    rec = pd.DataFrame(rows)
    test = rec[rec.split.str.startswith("test_")].groupby("dataset")[[c for c in rec if c.startswith("recall")]].mean()
    train = rec[~rec.split.str.startswith("test_")].groupby("dataset")[[c for c in rec if c.startswith("recall")]].mean()

    def table(df: pd.DataFrame) -> list[str]:
        cols = list(df.columns)
        out = ["| Dataset | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        out += [f"| {ds} | " + " | ".join(f"{v:.1%}" for v in r) + " |" for ds, r in df.iterrows()]
        return out

    lines = [
        f"# Stage 2 report ({cfg.profile_name} profile)", "",
        f"Retriever: {cfg.models.retriever}, Search-R1's flat E5 index over {n_docs:,} Wikipedia-2018 "
        "passages, exact inner-product search on CPU (index streamed from disk).", "",
        f"- Queries retrieved: {len(set(queries)):,} (all split questions + 1 sanity query); "
        f"cache rows: {n_cache:,}; distinct passages: {len(doc_ids):,}.",
        f"- Time for encoding + one full index pass: {search_s / 60:.1f} min "
        f"({search_s / max(1, len(set(queries))) * 1000:.0f} ms per query amortised).",
        f"- Sanity: top passage for \"{SANITY}\" is \"{sanity_top['title']}\" — "
        f"{'mentions Shakespeare (pass)' if sanity_ok else 'does NOT mention Shakespeare (FAIL)'}.", "",
        "Answer recall@k (a gold answer string appears in the top-k passages), test samples:", "",
        *table(test), "", "Training-side splits (train_pool, val, live_ft):", "", *table(train), "",
        "BM25 (Search-R1's Lucene index, needed only for the robustness experiment E6) is not built "
        "yet: Pyserini needs Java 21; planned as a Kaggle CPU job.",
    ]
    report = repo_path(cfg.paths.reports) / "STAGE_2_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_run_config(cfg, repo_path(cfg.paths.reports) / "stage2")
    print("\n".join(lines))
    retriever.close()
    corpus.close()
    if not sanity_ok:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
