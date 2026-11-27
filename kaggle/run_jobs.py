"""Kaggle side, any stage from 4 on: run the jobs of one tier from a job file (one process per tier).

    /tmp/vllm-env/bin/python kaggle/run_jobs.py --jobs kaggle/inputs/stage4_r1.jsonl.gz --tier S

A job file is gzipped JSONL, one rendered prompt per line:
    {job_id, qid, tier, kind, max_tokens, stop, prompt, ...}
next to <name>.meta.json (models per tier, tensor parallel, dtype, seed, batch size).
Jobs of the chosen tier run grouped by decoding settings, in file order (the builder sorts
them so prompts of one question sit together and share vLLM's prefix cache).

Writes <out>/<tier>.jsonl (one row per job: job_id, qid, tier, kind, text, n_in, n_out,
mean_logprob, finish_reason, gpu_s) and <out>/<tier>.meta.json (timing, versions).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from budgetrag.llm.client import LLMClient  # noqa: E402


def meta_path(jobs_path: Path) -> Path:
    return jobs_path.with_name(jobs_path.name.replace(".jsonl.gz", ".meta.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N jobs of this tier only (quick check)")
    args = ap.parse_args()

    jobs_path = Path(args.jobs)
    meta = json.loads(meta_path(jobs_path).read_text(encoding="utf-8"))
    with gzip.open(jobs_path, "rt", encoding="utf-8") as f:
        jobs = [j for j in map(json.loads, f) if j["tier"] == args.tier]
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print(f"no jobs for tier {args.tier} in {jobs_path}")
        return

    client = LLMClient(meta["models"], meta["tensor_parallel"], dtype=meta["dtype"],
                       max_model_len=meta["max_model_len"], seed=meta["seed"],
                       gpu_memory_utilization=meta["gpu_memory_utilization"], batch_size=meta["batch_size"])
    client.load(args.tier)
    print(f"tier {args.tier} ({meta['models'][args.tier]}) loaded in {client.load_s[args.tier]:.1f} s", flush=True)

    groups = defaultdict(list)
    for j in jobs:
        groups[(j["max_tokens"], tuple(j["stop"]))].append(j)
    rows, batches = [], []
    t_all = time.time()
    for (max_tokens, stop), js in groups.items():
        t0 = time.time()
        res = client.generate(args.tier, [j["prompt"] for j in js], max_tokens, list(stop))
        for j, g in zip(js, res.gens):
            rows.append({"job_id": j["job_id"], "qid": j["qid"], "tier": args.tier, "kind": j["kind"], **asdict(g)})
        batches.extend({"max_tokens": max_tokens, "stop": list(stop), **asdict(b)} for b in res.batches)
        print(f"  {len(js)} prompts (max_tokens {max_tokens}, stop {list(stop)!r}) in {time.time() - t0:.1f} s",
              flush=True)
    total = time.time() - t_all

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.tier}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    import torch
    import vllm
    info = {"tier": args.tier, "model": meta["models"][args.tier], "tp": meta["tensor_parallel"][args.tier],
            "jobs": jobs_path.name, "load_s": client.load_s[args.tier], "generate_s": total, "n_rows": len(rows),
            "batches": batches, "vllm": vllm.__version__, "torch": torch.__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    (out_dir / f"{args.tier}.meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"tier {args.tier}: {len(rows)} rows in {total:.1f} s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
