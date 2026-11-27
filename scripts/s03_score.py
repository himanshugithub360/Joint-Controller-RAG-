"""Stage 3, workstation side: score the Kaggle outputs and write the Stage 3 report.

    python scripts/s03_score.py            # reads data/stage3/{S,M,L}.jsonl (+ .meta.json)

Put the files from the Kaggle notebook's output (stage3_out/) into data/stage3/ first.

Checks the plan's "done when": every tier answered every validation question with top-3
passages, with input and output token counts on every call. Also reports EM/F1 per tier,
call kind and dataset (scored with the reused Search-R1 / FlashRAG functions), the
determinism rate (repeated prompts), empty and truncated outputs, and timing.
Writes data/stage3/scored.parquet and reports/STAGE_3_REPORT.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budgetrag.config import load_config, repo_path, save_run_config  # noqa: E402
from budgetrag.eval.answers import extract_answer, extract_query  # noqa: E402
from budgetrag.eval import metrics  # noqa: E402

TIERS = ["S", "M", "L"]
SCORED_KINDS = ["answer", "closed", "tentative"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main(argv: list[str]) -> None:
    cfg = load_config(argv)
    out_dir = repo_path(cfg.llm.stage3.outputs)
    jobs_meta = json.loads(repo_path(cfg.llm.stage3.jobs).with_name("stage3_jobs.meta.json").read_text(encoding="utf-8"))
    q = pd.read_parquet(repo_path(cfg.paths.questions)).set_index("qid")

    frames, metas, missing = [], {}, []
    for t in TIERS:
        f = out_dir / f"{t}.jsonl"
        if not f.exists():
            missing.append(t)
            continue
        frames.append(pd.read_json(f, lines=True, dtype={"text": str}))
        mf = out_dir / f"{t}.meta.json"
        metas[t] = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    if not frames:
        raise SystemExit(f"no Kaggle outputs in {out_dir}: copy S.jsonl, M.jsonl, L.jsonl there first")
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("")
    df["dataset"] = df.qid.map(q["dataset"])
    golds = df.qid.map(q["answers"]).map(list)

    df["pred"] = [extract_query(t) if k == "subquery" else extract_answer(t) for t, k in zip(df.text, df.kind)]
    scored = df.kind.isin(SCORED_KINDS)
    df["em"] = [metrics.exact_match(p, g) if s else None for p, g, s in zip(df.pred, golds, scored)]
    df["f1"] = [metrics.f1(p, g) if s else None for p, g, s in zip(df.pred, golds, scored)]
    df["is_empty"] = df.pred.str.len() == 0
    df["truncated"] = df.finish_reason == "length"
    df.to_parquet(out_dir / "scored.parquet", index=False)

    p1 = df[df["pass"] == 1]
    n_q = jobs_meta["n_questions"]
    lines = [f"# Stage 3 report ({jobs_meta['profile']} profile)", "",
             f"Prompts {jobs_meta['prompt_version']} (configs/prompts/), greedy decoding, top-{jobs_meta['k']} E5 passages, "
             f"{n_q} validation questions, {jobs_meta['n_jobs']} calls per tier "
             f"(answer, closed-book, tentative, sub-query) plus {jobs_meta['n_repeat']} repeated answer calls. "
             "Generated on Kaggle (2x T4, vLLM, float16); scored on the workstation with Search-R1's EM and "
             "FlashRAG's token F1.", ""]

    # done-when check
    lines += ["## Done-when check", "", "| Tier | Model | Calls (pass 1) | Answer calls | Token counts on every call | Load (s) | Generate (s) | GPU-s |",
              "|---|---|---|---|---|---|---|---|"]
    ok = not missing
    for t in TIERS:
        if t in missing:
            lines.append(f"| {t} | {jobs_meta['models'][t]} | missing | | | | | |")
            continue
        d = p1[p1.tier == t]
        n_ans = int((d.kind == "answer").sum())
        tok_ok = bool((d.n_in > 0).all() and d.n_out.notna().all())
        full = len(d) == jobs_meta["n_jobs"] and n_ans == n_q
        ok &= full and tok_ok
        m = metas.get(t, {})
        lines.append(f"| {t} | {jobs_meta['models'][t]} | {len(d)} / {jobs_meta['n_jobs']} | {n_ans} / {n_q} | "
                     f"{'yes' if tok_ok else 'NO'} | {m.get('load_s', float('nan')):.0f} | "
                     f"{m.get('generate_s', float('nan')):.0f} | {df[df.tier == t].gpu_s.sum():.0f} |")
    lines += ["", f"Result: **{'pass' if ok else 'not done'}**" + (f" (missing tiers: {', '.join(missing)})" if missing else ""), ""]

    # quality
    lines += ["## Answer quality (pass 1)", "", "F1 / EM in %, max over gold answers.", "",
              "| Tier | Call | NQ F1 | NQ EM | HotpotQA F1 | HotpotQA EM | All F1 | All EM |", "|---|---|---|---|---|---|---|---|"]
    for t in TIERS:
        for k in SCORED_KINDS:
            d = p1[(p1.tier == t) & (p1.kind == k)]
            if d.empty:
                continue
            cells = []
            for ds in ["nq", "hotpotqa"]:
                e = d[d.dataset == ds]
                cells += [pct(e.f1.mean()), pct(e.em.mean())]
            cells += [pct(d.f1.mean()), pct(d.em.mean())]
            lines.append(f"| {t} | {k} | " + " | ".join(cells) + " |")
    ans = p1[p1.kind == "answer"].groupby("tier").f1.mean()
    if len(ans) > 1:
        gaps = [f"{b} - {a}: {100 * (ans[b] - ans[a]):+.1f}" for a, b in zip(TIERS, TIERS[1:]) if a in ans and b in ans]
        lines += ["", "F1 gaps with top-3 passages (preview of the Stage 4 gate, which needs >= 3 points each): "
                  + "; ".join(gaps) + "."]
    lines.append("")

    # output hygiene
    lines += ["## Output checks (pass 1)", "",
              "| Tier | Call | Empty % | Hit max_tokens % | Mean input tokens | Mean output tokens | Mean log-prob |",
              "|---|---|---|---|---|---|---|"]
    for t in TIERS:
        for k in ["answer", "closed", "tentative", "subquery"]:
            d = p1[(p1.tier == t) & (p1.kind == k)]
            if d.empty:
                continue
            lines.append(f"| {t} | {k} | {pct(d.is_empty.mean())} | {pct(d.truncated.mean())} | {d.n_in.mean():.0f} | "
                         f"{d.n_out.mean():.1f} | {d.mean_logprob.mean():.3f} |")
    tent = p1[p1.kind == "tentative"]
    if not tent.empty:
        tag = tent.groupby("tier").text.apply(lambda s: s.str.contains("<answer>").mean())
        lines += ["", "Tentative calls that used the <answer> tag: "
                  + ", ".join(f"{t} {pct(v)}%" for t, v in tag.items()) + "."]
    lines.append("")

    # determinism
    lines += ["## Determinism", "", f"{jobs_meta['n_repeat']} answer prompts generated twice in the same process "
              "(second pass after all other calls, so the prefix cache is warm and batches differ).", "",
              "| Tier | Identical text | Identical scored answer |", "|---|---|---|"]
    for t in TIERS:
        d = df[(df.tier == t) & (df.kind == "answer")]
        if d.empty:
            continue
        a = d[d["pass"] == 1].set_index("job_id")
        b = d[d["pass"] == 2].set_index("job_id")
        common = b.index.intersection(a.index)
        if len(common):
            same_t = (a.loc[common, "text"] == b.loc[common, "text"]).mean()
            same_p = (a.loc[common, "pred"] == b.loc[common, "pred"]).mean()
            lines.append(f"| {t} | {pct(same_t)}% of {len(common)} | {pct(same_p)}% |")
    lines.append("")

    # determinism failures, verbatim
    for t in TIERS:
        d = df[(df.tier == t) & (df.kind == "answer")]
        a = d[d["pass"] == 1].set_index("job_id")
        b = d[d["pass"] == 2].set_index("job_id")
        for j in b.index.intersection(a.index):
            if a.at[j, "text"] != b.at[j, "text"]:
                lines.append(f"- Tier {t}, `{j}`: {a.at[j, 'text']!r} then {b.at[j, 'text']!r}.")
    lines.append("")

    # empty tentative answers: abstentions or broken output?
    tent_empty = tent[tent.is_empty] if not tent.empty else tent
    if not tent_empty.empty:
        abstain = tent_empty.text.str.strip().str.fullmatch(r"<answer>\s*").mean()
        lines += ["Empty tentative answers: " + ", ".join(f"{t} {n}" for t, n in tent_empty.tier.value_counts().sort_index().items())
                  + f"; {pct(abstain)}% of them are the model opening <answer> and closing it at once (an abstention, "
                  "not a parsing failure).", ""]

    # sub-queries
    subq = p1[p1.kind == "subquery"]
    if not subq.empty:
        restates = subq.groupby("tier").apply(
            lambda x: (x.pred.map(metrics.normalize_answer) == x.qid.map(q["question"]).map(metrics.normalize_answer)).mean())
        lines += ["## Sub-queries", "",
                  "Share of sub-queries that only repeat the question (equal after SQuAD normalisation): "
                  + ", ".join(f"{t} {pct(v)}%" for t, v in restates.items()) + ".", ""]
        for t in TIERS:
            ex = subq[(subq.tier == t) & (subq.dataset == "hotpotqa")].head(5)
            if ex.empty:
                continue
            lines += [f"Tier {t}, first five HotpotQA questions:", ""]
            lines += [f"- {q.at[r.qid, 'question']} → `{r.pred}`" for r in ex.itertuples()]
            lines.append("")

    report = repo_path(cfg.paths.reports) / "STAGE_3_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    save_run_config(cfg, repo_path(cfg.paths.reports) / "stage3")
    print("\n".join(lines))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
