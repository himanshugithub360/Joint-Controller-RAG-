"""Stage 3, workstation side: render the validation prompts that Kaggle will run.

    python scripts/s03_build_prompts.py

For every validation question (500 in the lite profile) it writes four calls, each a fully
rendered Qwen chat prompt:
  answer     question + top-3 E5 passages -> short answer        (configs/prompts/v1/answer.txt)
  closed     question only -> short answer                        (answer_nodocs.txt)
  tentative  AutoSearch-style <answer> self-answer on top-3        (tentative.txt)
  subquery   one search query for the missing fact, given top-3    (subquery.txt)
A seeded sample of `llm.stage3.determinism_n` answer calls is marked to be generated twice.

Writes kaggle/inputs/stage3_jobs.jsonl.gz and stage3_jobs.meta.json (committed, so the Kaggle
notebook reads them from the cloned repo; Kaggle needs no Hydra or local data).
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf  # noqa: E402

from budgetrag.config import git_hash, load_config, repo_path  # noqa: E402
from budgetrag.llm.prompts import fill, load_template, render_chat  # noqa: E402


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    st = cfg.llm.stage3
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)  # Qwen2.5 sizes share tokenizer and chat template

    q = pd.read_parquet(repo_path(cfg.paths.questions))
    val = q[q.split == "val"].sort_values("qid").reset_index(drop=True)
    cache = pd.read_parquet(repo_path(cfg.paths.retrieval_cache))
    cache = cache[cache.retriever == cfg.retrieval.primary].set_index("query")
    passages = pd.read_parquet(repo_path(cfg.paths.passages)).set_index("doc_id")

    templates = {kind: load_template(c.prompt, cfg.llm.prompt_version) for kind, c in cfg.llm.calls.items()}
    rng = np.random.default_rng(cfg.seeds.data)
    repeat = set(rng.choice(val.qid.to_numpy(), size=min(st.determinism_n, len(val)), replace=False))

    jobs = []
    for row in val.itertuples():
        ids = [int(i) for i in cache.loc[row.question, "doc_ids"][: st.k]]
        docs = [{"title": passages.at[i, "title"], "text": passages.at[i, "text"]} for i in ids]
        for kind, call in cfg.llm.calls.items():
            text = fill(templates[kind], row.question, None if kind == "closed" else docs)
            jobs.append({"job_id": f"{row.qid}|{kind}", "qid": row.qid, "kind": kind,
                         "max_tokens": int(call.max_tokens), "stop": list(call.stop),
                         "repeat": kind == "answer" and row.qid in repeat,
                         "doc_ids": [] if kind == "closed" else ids,
                         "prompt": render_chat(tok, text)})

    out = repo_path(st.jobs)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs)
    meta = {
        "profile": cfg.profile_name, "prompt_version": cfg.llm.prompt_version, "git_hash": git_hash(),
        "tokenizer": cfg.tokenizer, "k": int(st.k), "n_questions": len(val), "n_jobs": len(jobs),
        "n_repeat": sum(j["repeat"] for j in jobs),
        "models": OmegaConf.to_container(cfg.models.tiers),
        "tensor_parallel": OmegaConf.to_container(cfg.models.tensor_parallel),
        "dtype": cfg.models.dtype, "max_model_len": int(cfg.models.max_model_len), "seed": int(cfg.llm.seed),
        "gpu_memory_utilization": float(cfg.llm.gpu_memory_utilization), "batch_size": int(cfg.llm.batch_size),
    }
    out.with_name("stage3_jobs.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    n_tok = [len(tok(j["prompt"]).input_ids) for j in jobs]
    print(f"{len(jobs)} jobs for {len(val)} questions ({meta['n_repeat']} repeated) -> {out} "
          f"({out.stat().st_size / 1e6:.2f} MB); prompt tokens mean {np.mean(n_tok):.0f}, max {max(n_tok)}")


if __name__ == "__main__":
    main(sys.argv[1:])
