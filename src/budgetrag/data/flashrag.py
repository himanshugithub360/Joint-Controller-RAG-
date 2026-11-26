"""Download and read the FlashRAG benchmark files (the datasets Search-R1 and AutoSearch use).

Each FlashRAG JSONL record has: id, question, golden_answers (list), optional metadata.
FlashRAG ids repeat across datasets ("train_0" exists in every file), so the project-wide
question id is "<dataset>-<flashrag id>", e.g. "nq-train_0".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download

HF_REPO = "RUC-NLPIR/FlashRAG_datasets"


@dataclass(frozen=True)
class Question:
    qid: str
    dataset: str
    source_split: str
    question: str
    answers: list[str]
    hops: int | None


def make_qid(dataset: str, flashrag_id: str) -> str:
    return f"{dataset}-{flashrag_id}"


def download(dataset: str, split: str, raw_dir: Path, repo: str = HF_REPO) -> Path:
    """Fetch <dataset>/<split>.jsonl into raw_dir (skipped if already there)."""
    target = raw_dir / dataset / f"{split}.jsonl"
    if target.exists():
        return target
    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=f"{dataset}/{split}.jsonl",
        local_dir=raw_dir,
    )
    return Path(path)


def musique_hops(record: dict) -> int | None:
    decomposition = (record.get("metadata") or {}).get("question_decomposition")
    return len(decomposition) if decomposition else None


def iter_questions(dataset: str, split: str, raw_dir: Path) -> Iterator[Question]:
    path = raw_dir / dataset / f"{split}.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            yield Question(
                qid=make_qid(dataset, rec["id"]),
                dataset=dataset,
                source_split=split,
                question=rec["question"].strip(),
                answers=[str(a) for a in rec.get("golden_answers") or []],
                hops=musique_hops(rec) if dataset == "musique" else None,
            )


def load_questions(dataset: str, split: str, raw_dir: Path) -> list[Question]:
    return list(iter_questions(dataset, split, raw_dir))
