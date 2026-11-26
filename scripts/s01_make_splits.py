"""Stage 1: download the FlashRAG questions, freeze the split ID lists, build questions.parquet.

    python scripts/s01_make_splits.py                  # lite profile (default)
    python scripts/s01_make_splits.py profile=full     # original plan's sizes

Outputs
  data/raw/<dataset>/<split>.jsonl      downloaded files (git-ignored)
  data/splits/<name>.ids.txt            frozen ID lists (committed to git)
  data/questions.parquet                qid, dataset, split, source_split, question,
                                        answers, n_tokens, hops   (git-ignored)
  reports/STAGE_1_REPORT.md             counts per split and dataset

If data/splits/ already holds ID lists for the same profile they are reused, never
re-sampled, so every later stage sees the same questions. Pass --resample to overwrite.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budgetrag.config import load_config, repo_path, save_run_config  # noqa: E402
from budgetrag.data import flashrag  # noqa: E402
from budgetrag.data.splits import (  # noqa: E402
    TRAIN_SIDE, read_all_splits, sample_test, sample_train_side, write_ids,
)


def main(argv: list[str]) -> None:
    resample = "--resample" in argv
    cfg = load_config([a for a in argv if a != "--resample"])
    raw, splits_dir = repo_path(cfg.paths.raw), repo_path(cfg.paths.splits)
    seed = cfg.seeds.data
    test_split = dict(cfg.flashrag.test_split)

    # 1. Download what this profile needs.
    files = [(ds, "train") for ds in cfg.flashrag.train_sources] + list(test_split.items())
    for ds, split in files:
        p = flashrag.download(ds, split, raw, cfg.flashrag.hf_repo)
        print(f"have {p.relative_to(repo_path('.'))} ({p.stat().st_size / 2**20:.1f} MB)")

    pools = {(ds, split): flashrag.load_questions(ds, split, raw) for ds, split in files}

    # 2. Freeze ID lists (or reuse the committed ones).
    marker = splits_dir / "PROFILE"
    existing = read_all_splits(splits_dir) if splits_dir.exists() else {}
    if existing and not resample:
        found = marker.read_text().strip() if marker.exists() else "?"
        if found != cfg.profile_name:
            sys.exit(f"data/splits holds the '{found}' profile; pass --resample to replace it "
                     f"with '{cfg.profile_name}'")
        splits = existing
        print(f"reusing frozen splits in {splits_dir}")
    else:
        for old in splits_dir.glob("*.ids.txt"):
            old.unlink()
        splits = {name: [] for name in TRAIN_SIDE}
        for ds in cfg.flashrag.train_sources:
            ids = [q.qid for q in pools[(ds, "train")]]
            for name, part in sample_train_side(ids, dict(cfg.splits.per_source), seed).items():
                splits[name] += part
        for ds, split in test_split.items():
            ids = [q.qid for q in pools[(ds, split)]]
            splits[f"test_{ds}"] = sample_test(ids, cfg.splits.test_per_dataset, seed)
        for name, ids in splits.items():
            write_ids(splits_dir / f"{name}.ids.txt", ids)
        marker.write_text(cfg.profile_name + "\n")
        print(f"wrote {len(splits)} ID lists to {splits_dir}")

    # 3. One table for every question any split uses.
    by_qid = {q.qid: q for qs in pools.values() for q in qs}
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    rows = []
    for name, ids in splits.items():
        for qid in ids:
            q = by_qid[qid]
            rows.append(dict(qid=qid, dataset=q.dataset, split=name, source_split=q.source_split,
                             question=q.question, answers=q.answers,
                             n_tokens=len(tok(q.question)["input_ids"]), hops=q.hops))
    df = pd.DataFrame(rows)
    df["hops"] = df["hops"].astype("Int64")
    out = repo_path(cfg.paths.questions)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    save_run_config(cfg, repo_path(cfg.paths.reports) / "stage1")

    # 4. Report.
    counts = Counter(zip(df["split"], df["dataset"]))
    lines = [f"# Stage 1 report ({cfg.profile_name} profile)", "",
             f"Seed {seed}. {len(df):,} questions in {len(splits)} splits; "
             f"{df['qid'].nunique():,} unique ids.", "",
             "| Split | Dataset | Questions | Mean tokens |", "|---|---|---|---|"]
    for (split, ds), n in sorted(counts.items()):
        mean_tok = df[(df.split == split) & (df.dataset == ds)]["n_tokens"].mean()
        lines.append(f"| {split} | {ds} | {n:,} | {mean_tok:.1f} |")
    full_sizes = {ds: len(pools[(ds, s)]) for ds, s in test_split.items()}
    lines += ["", "Full test-split sizes in the downloaded files: "
              + ", ".join(f"{ds} {n:,}" for ds, n in full_sizes.items()) + "."]
    hop = df[df.dataset == "musique"]["hops"].value_counts().sort_index()
    lines += ["", "MuSiQue hops in the test sample: "
              + ", ".join(f"{int(h)}-hop {n}" for h, n in hop.items()) + "."]
    report = repo_path(cfg.paths.reports) / "STAGE_1_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
