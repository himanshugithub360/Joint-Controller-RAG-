"""Stage 3, Kaggle side: run every Stage 3 job on one tier (one process per tier).

    /tmp/vllm-env/bin/python kaggle/stage3_generate.py --tier S
    /tmp/vllm-env/bin/python kaggle/stage3_generate.py --tier L      # 7B, both GPUs

Reads kaggle/inputs/stage3_jobs.jsonl.gz (+ .meta.json for model names and settings), runs
the jobs grouped by decoding settings through budgetrag.llm.client, then runs the jobs marked
`repeat` a second time for the determinism check. Writes <out>/<tier>.jsonl (one row per job
and pass) and <out>/<tier>.meta.json (load time, per-batch timing, versions).
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["S", "M", "L"])
    ap.add_argument("--jobs", default=str(REPO / "kaggle/inputs/stage3_jobs.jsonl.gz"))
    ap.add_argument("--out-dir", default="/kaggle/working/stage3_out")
    ap.add_argument("--limit", type=int, default=0, help="first N jobs only (quick check)")
    args = ap.parse_args()

    jobs_path = Path(args.jobs)
    meta = json.loads(jobs_path.with_name("stage3_jobs.meta.json").read_text(encoding="utf-8"))
    with gzip.open(jobs_path, "rt", encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f]
    if args.limit:
        jobs = jobs[: args.limit]

    client = LLMClient(meta["models"], meta["tensor_parallel"], dtype=meta["dtype"],
                       max_model_len=meta["max_model_len"], seed=meta["seed"],
                       gpu_memory_utilization=meta["gpu_memory_utilization"], batch_size=meta["batch_size"])
    client.load(args.tier)
    print(f"tier {args.tier} ({meta['models'][args.tier]}) loaded in {client.load_s[args.tier]:.1f} s", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, batches = [], []

    def run(pass_no: int, subset: list[dict]) -> None:
        groups = defaultdict(list)
        for j in subset:
            groups[(j["max_tokens"], tuple(j["stop"]))].append(j)
        for (max_tokens, stop), js in groups.items():
            t0 = time.time()
            res = client.generate(args.tier, [j["prompt"] for j in js], max_tokens, list(stop))
            for j, g in zip(js, res.gens):
                rows.append({"job_id": j["job_id"], "qid": j["qid"], "kind": j["kind"], "tier": args.tier,
                             "pass": pass_no, **asdict(g)})
            batches.extend({"pass": pass_no, "max_tokens": max_tokens, "stop": list(stop), **asdict(b)}
                           for b in res.batches)
            print(f"  pass {pass_no}: {len(js)} prompts (max_tokens {max_tokens}, stop {list(stop)!r}) "
                  f"in {time.time() - t0:.1f} s", flush=True)

    t_all = time.time()
    run(1, jobs)
    run(2, [j for j in jobs if j["repeat"]])
    total = time.time() - t_all

    with open(out_dir / f"{args.tier}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    import torch
    import vllm
    info = {"tier": args.tier, "model": meta["models"][args.tier], "tp": meta["tensor_parallel"][args.tier],
            "load_s": client.load_s[args.tier], "generate_s": total, "n_rows": len(rows),
            "jobs_meta": meta, "batches": batches, "vllm": vllm.__version__, "torch": torch.__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    (out_dir / f"{args.tier}.meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"tier {args.tier}: {len(rows)} rows in {total:.1f} s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
