"""Stage 3 unit tests: answer extraction, reused EM/F1, prompts, the vLLM wrapper (with a fake
vLLM, since the workstation cannot run it), and the scoring script on fake Kaggle output."""

from __future__ import annotations

import gzip
import json
import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest

from budgetrag.config import load_config, repo_path
from budgetrag.eval import metrics
from budgetrag.eval.answers import extract_answer, score
from budgetrag.llm.client import LLMClient, mean_logprob
from budgetrag.llm.prompts import fill, format_passages, load_template

# ---- answer extraction: 20 hand-written edge cases (plan Stage 3, step 8) ---------------

CASES = [
    ("", ""),
    (None, ""),
    ("   \n\n  ", ""),
    ("Paris", "Paris"),
    ("Paris.", "Paris"),
    ("  Paris  \n", "Paris"),
    ("The answer is Paris.", "Paris"),
    ("Answer: Paris", "Paris"),
    ("answer - Paris", "Paris"),
    ("Final answer: 1984", "1984"),
    ('"Paris"', "Paris"),
    ("'Paris'", "Paris"),
    ("“Paris”", "Paris"),
    ("**Paris**", "Paris"),
    ("Paris\nParis is the capital of France.", "Paris"),
    ("<answer> Beijing </answer>", "Beijing"),
    ("<think>it is in China</think>\n<answer> Beijing", "Beijing"),
    ("Richard Bright, Al Pacino", "Richard Bright, Al Pacino"),
    ("yes!", "yes"),
    ("U.S.A.", "U.S.A"),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_extract_answer(raw, expected):
    assert extract_answer(raw) == expected


def test_twenty_cases():
    assert len(CASES) == 20


# ---- reused EM / F1 ---------------------------------------------------------------------

def test_metrics_reused_behaviour():
    assert metrics.normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert metrics.exact_match("the Eiffel tower", ["Eiffel Tower", "x"]) == 1.0
    assert metrics.exact_match("Eiffel", ["Eiffel Tower"]) == 0.0
    assert metrics.f1("Eiffel", ["Eiffel Tower"]) == pytest.approx(2 / 3)
    assert metrics.f1("Eiffel Tower", ["nope", "Eiffel Tower"]) == 1.0
    assert metrics.f1("no", ["yes"]) == 0.0          # FlashRAG's yes/no rule
    assert metrics.f1("", ["Paris"]) == 0.0
    assert "flashrag" not in sys.modules            # the stub package is removed again


def test_score_combines_extraction_and_metrics():
    s = score("The answer is Richard Bright.", ["Richard Bright"])
    assert s == {"pred": "Richard Bright", "em": 1.0, "f1": 1.0}


# ---- prompts ----------------------------------------------------------------------------

def test_passage_format_matches_search_r1():
    # Search-R1's _passages2string on contents '"T"\ntext' gives 'Doc i(Title: "T") text\n'
    docs = [{"title": '"Hamlet"', "text": "Hamlet is a tragedy."}, {"title": '"B"', "text": "b {x}"}]
    assert format_passages(docs) == 'Doc 1(Title: "Hamlet") Hamlet is a tragedy.\nDoc 2(Title: "B") b {x}\n'


def test_templates_load_without_comments_and_fill():
    for name in ["answer.txt", "answer_nodocs.txt", "tentative.txt", "subquery.txt"]:
        t = load_template(name)
        assert not t.startswith("#") and "{question}" in t
    t = load_template("answer.txt")
    out = fill(t, " Who wrote Hamlet? ", [{"title": '"Hamlet"', "text": "has {braces}"}])
    assert "Question: Who wrote Hamlet?\nAnswer:" in out and "has {braces}" in out and "{passages}" not in out
    assert "<answer>" in load_template("tentative.txt")


# ---- client, with a fake vLLM -----------------------------------------------------------

class FakeLLM:
    def __init__(self, **kw):
        self.kw = kw

    def generate(self, prompts, params, use_tqdm=False):
        outs = []
        for p in prompts:
            ids = [1, 2]
            lps = [{1: SimpleNamespace(logprob=-0.5)}, {2: SimpleNamespace(logprob=-1.5)}]
            outs.append(SimpleNamespace(prompt_token_ids=list(range(len(p))),
                                        outputs=[SimpleNamespace(text=p.upper(), token_ids=ids, logprobs=lps,
                                                                 finish_reason="stop")]))
        return outs


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    mod.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod


def test_client_generate_batches_and_shares_gpu_seconds(fake_vllm):
    c = LLMClient({"S": "m-s", "L": "m-l"}, {"S": 1, "L": 2}, batch_size=2)
    res = c.generate("L", ["aa", "bbbb", "c"], max_tokens=8, stop=["\n"])
    assert [g.text for g in res.gens] == ["AA", "BBBB", "C"]
    assert [g.n_in for g in res.gens] == [2, 4, 1] and all(g.n_out == 2 for g in res.gens)
    assert all(g.mean_logprob == -1.0 for g in res.gens)
    assert [b.n_prompts for b in res.batches] == [2, 1]
    assert all(b.gpu_s == pytest.approx(2 * b.wall_s) for b in res.batches)      # tp = 2
    first = res.batches[0]
    assert res.gens[0].gpu_s + res.gens[1].gpu_s == pytest.approx(first.gpu_s)
    assert c.llm.kw["tensor_parallel_size"] == 2 and c.llm.kw["enable_prefix_caching"]
    with pytest.raises(RuntimeError):
        c.load("S")                                                            # one tier per process


def test_mean_logprob_edge_cases():
    assert mean_logprob([], None) is None
    assert mean_logprob([5], [{5: SimpleNamespace(logprob=-2.0)}]) == -2.0


# ---- the committed Kaggle job file ------------------------------------------------------

CFG = load_config()
JOBS = repo_path(CFG.llm.stage3.jobs)


@pytest.mark.data
@pytest.mark.skipif(not JOBS.exists(), reason="run scripts/s03_build_prompts.py first")
def test_jobs_file_covers_validation():
    with gzip.open(JOBS, "rt", encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f]
    meta = json.loads(JOBS.with_name("stage3_jobs.meta.json").read_text(encoding="utf-8"))
    q = pd.read_parquet(repo_path(CFG.paths.questions))
    val = set(q[q.split == "val"].qid)
    assert {j["qid"] for j in jobs} == val
    assert len(jobs) == 4 * len(val) == meta["n_jobs"]
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    assert sum(j["repeat"] for j in jobs) == CFG.llm.stage3.determinism_n
    assert all(j["prompt"].endswith("<|im_start|>assistant\n") for j in jobs)
    assert all(len(j["doc_ids"]) == (0 if j["kind"] == "closed" else CFG.llm.stage3.k) for j in jobs)


# ---- the scoring script on fake Kaggle output -------------------------------------------

@pytest.mark.data
@pytest.mark.skipif(not JOBS.exists(), reason="run scripts/s03_build_prompts.py first")
def test_score_script_on_fake_outputs(tmp_path):
    sys.path.insert(0, str(repo_path("scripts")))
    import s03_score
    with gzip.open(JOBS, "rt", encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f]
    q = pd.read_parquet(repo_path(CFG.paths.questions)).set_index("qid")
    rows = []
    for p, subset in [(1, jobs), (2, [j for j in jobs if j["repeat"]])]:
        for j in subset:
            gold = q.at[j["qid"], "answers"][0]
            text = f"<answer> {gold} </answer>" if j["kind"] == "tentative" else f"{gold}."
            rows.append({"job_id": j["job_id"], "qid": j["qid"], "kind": j["kind"], "tier": "S", "pass": p,
                         "text": text, "n_in": 100, "n_out": 3, "mean_logprob": -0.1, "finish_reason": "stop",
                         "gpu_s": 0.01})
    pd.DataFrame(rows).to_json(tmp_path / "S.jsonl", orient="records", lines=True)
    with pytest.raises(SystemExit):        # M and L missing -> not done
        s03_score.main([f"llm.stage3.outputs={tmp_path.as_posix()}", f"paths.reports={tmp_path.as_posix()}"])
    report = (tmp_path / "STAGE_3_REPORT.md").read_text(encoding="utf-8")
    assert "missing tiers: M, L" in report
    scored = pd.read_parquet(tmp_path / "scored.parquet")
    assert scored[scored.kind != "subquery"].em.mean() == 1.0
    assert "| S | 100.0% of 200 | 100.0% |" in report
