"""Thin wrapper around vLLM's offline engine: generate(tier, prompts, ...) (plan Stage 3, step 2).

Runs on Kaggle (vLLM needs compute capability >= 7.0). Deviation from the plan, which starts
one OpenAI-compatible server per tier: two T4s cannot hold the three tiers at once (the 7B
tier needs both GPUs), so tiers run one at a time, each in its own process, through vLLM's
in-process engine. Same engine, same greedy decoding, exact token counts, no HTTP layer.

Every call is greedy (temperature 0, top-p 1). Per prompt it returns the text, input and
output token counts, the mean log-probability of the generated tokens and the finish reason.
Per batch it logs wall-clock seconds and GPU-seconds (wall x GPUs used); each prompt gets a
share of its batch's GPU-seconds in proportion to its input + output tokens.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass
class Generation:
    text: str
    n_in: int
    n_out: int
    mean_logprob: float | None
    finish_reason: str | None
    gpu_s: float = 0.0


@dataclass
class BatchStat:
    tier: str
    n_prompts: int
    n_in: int
    n_out: int
    wall_s: float
    gpu_s: float


@dataclass
class Result:
    gens: list[Generation] = field(default_factory=list)
    batches: list[BatchStat] = field(default_factory=list)


def mean_logprob(token_ids, logprobs) -> float | None:
    """Mean log-probability of the sampled tokens from vLLM's per-step logprob dicts."""
    if not token_ids or not logprobs:
        return None
    vals = [step[t].logprob for t, step in zip(token_ids, logprobs) if step and t in step]
    return sum(vals) / len(vals) if vals else None


class LLMClient:
    def __init__(self, tiers: dict[str, str], tensor_parallel: dict[str, int], dtype: str = "float16",
                 max_model_len: int = 4096, seed: int = 0, gpu_memory_utilization: float = 0.90,
                 batch_size: int = 256):
        self.tiers = dict(tiers)
        self.tp = dict(tensor_parallel)
        self.dtype, self.max_model_len, self.seed = dtype, max_model_len, seed
        self.gpu_memory_utilization, self.batch_size = gpu_memory_utilization, batch_size
        self.tier: str | None = None
        self.llm = None
        self.load_s: dict[str, float] = {}

    def load(self, tier: str):
        """Load one tier. One tier per process: vLLM does not reliably free a tensor-parallel
        engine inside a running process, so a second tier is refused rather than half-loaded."""
        if self.tier == tier:
            return self.llm
        if self.tier is not None:
            raise RuntimeError(f"tier {self.tier} already loaded; start a new process for tier {tier}")
        from vllm import LLM  # imported here: the workstation has no vLLM
        t0 = time.time()
        self.llm = LLM(model=self.tiers[tier], dtype=self.dtype, tensor_parallel_size=self.tp.get(tier, 1),
                       max_model_len=self.max_model_len, seed=self.seed,
                       gpu_memory_utilization=self.gpu_memory_utilization, enable_prefix_caching=True)
        self.load_s[tier] = time.time() - t0
        self.tier = tier
        return self.llm

    def generate(self, tier: str, prompts: list[str], max_tokens: int, stop: list[str] | None = None,
                 logprobs: bool = True) -> Result:
        from vllm import SamplingParams
        llm = self.load(tier)
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=stop or None,
                                logprobs=1 if logprobs else None, seed=self.seed)
        n_gpu = self.tp.get(tier, 1)
        res = Result()
        for s in range(0, len(prompts), self.batch_size):
            batch = prompts[s:s + self.batch_size]
            t0 = time.time()
            outs = llm.generate(batch, params, use_tqdm=False)
            wall = time.time() - t0
            gens = []
            for o in outs:
                c = o.outputs[0]
                gens.append(Generation(text=c.text, n_in=len(o.prompt_token_ids), n_out=len(c.token_ids),
                                       mean_logprob=mean_logprob(c.token_ids, c.logprobs) if logprobs else None,
                                       finish_reason=c.finish_reason))
            tokens = sum(g.n_in + g.n_out for g in gens) or 1
            for g in gens:
                g.gpu_s = wall * n_gpu * (g.n_in + g.n_out) / tokens
            res.gens.extend(gens)
            res.batches.append(BatchStat(tier, len(batch), sum(g.n_in for g in gens),
                                         sum(g.n_out for g in gens), wall, wall * n_gpu))
        return res


__all__ = ["LLMClient", "Generation", "BatchStat", "Result", "mean_logprob", "asdict"]
