"""Stage 0 smoke test for one model tier on Kaggle (run once per tier, each in its own process).

    python kaggle/smoke_tier.py --model Qwen/Qwen2.5-1.5B-Instruct
    python kaggle/smoke_tier.py --model Qwen/Qwen2.5-7B-Instruct --tp 2

Loads the model with vLLM (float16, seed 0, greedy), asks the plan's smoke question twice to
check determinism, and appends one JSON line to smoke_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time

from vllm import LLM, SamplingParams

QUESTION = "What is the capital of France?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", default="smoke_results.jsonl")
    args = ap.parse_args()

    t0 = time.time()
    llm = LLM(model=args.model, dtype="float16", tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, seed=0, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)
    load_s = time.time() - t0

    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION + " Answer with the city name only."}],
        tokenize=False, add_generation_prompt=True)
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, logprobs=1)

    t1 = time.time()
    outs = llm.generate([prompt, prompt], params)
    gen_s = time.time() - t1
    texts = [o.outputs[0].text.strip() for o in outs]

    result = {
        "model": args.model, "tp": args.tp, "load_s": round(load_s, 1), "gen_s": round(gen_s, 2),
        "answer": texts[0], "deterministic": texts[0] == texts[1],
        "correct": "paris" in texts[0].lower(),
        "n_in": len(outs[0].prompt_token_ids), "n_out": len(outs[0].outputs[0].token_ids),
    }
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    if not (result["correct"] and result["deterministic"]):
        raise SystemExit("smoke test failed")


if __name__ == "__main__":
    main()
