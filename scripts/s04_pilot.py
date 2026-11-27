"""Stage 4: pilot run and the tier-gap gate (workstation side).

Multi-step retrieval alternates between the LLMs (Kaggle) and E5 search (this workstation,
where the 60 GB index lives), so the stage runs as three Kaggle rounds:

    python scripts/s04_pilot.py round 1     # jobs for Kaggle round 1, then run it on Kaggle
    #   unzip stage4_r1.zip into data/stage4/r1/
    python scripts/s04_pilot.py round 2     # retrieves round 1's sub-queries, writes round 2 jobs
    #   unzip stage4_r2.zip into data/stage4/r2/
    python scripts/s04_pilot.py round 3     # retrieves round 2's sub-queries, writes round 3 jobs
    #   unzip stage4_r3.zip into data/stage4/r3/
    python scripts/s04_pilot.py report      # scores everything, writes reports/STAGE_4_REPORT.md

Two question groups go through the same tree code (budgetrag.cache.tree):
  val    all validation questions, K = 3 only: nodes root, 3, 3-3, 3-3-3. Gives the 9
         configurations: no retrieval (root), top-3 single step (3), fixed 3-step (3-3-3).
  pilot  a seeded sample of the training pool with the full Stage 6 tree (K in {3, 5},
         15 nodes): measures calls, tokens and GPU-seconds per question for the cost
         projection, and whether K = 5 changes answers.
Helper calls (tentative answers, sub-queries) use the helper tier; every node is answered by
every tier, plus the gate fallback tiers in `llm.stage4.try_fallback`.

Round 1 asks the helper about the depth 0 and 1 nodes, round 2 about depth 2, and round 3
about depth 3; round 3 also has every tier answer every node.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf  # noqa: E402

from budgetrag.cache import tree  # noqa: E402
from budgetrag.config import git_hash, load_config, repo_path, save_run_config  # noqa: E402
from budgetrag.eval import metrics  # noqa: E402
from budgetrag.eval.answers import extract_answer, extract_query  # noqa: E402
from budgetrag.llm.prompts import fill, load_template, render_chat  # noqa: E402

NODE_COLS = ["qid", "group", "path", "depth", "doc_ids", "subquery", "subquery_raw", "search_query"]


# ---- setup ----------------------------------------------------------------------------

def tiers(cfg) -> dict:
    """Answer tiers for this stage: the profile's tiers plus the fallback candidates."""
    st = cfg.llm.stage4
    models = dict(OmegaConf.to_container(cfg.models.tiers))
    tp = dict(OmegaConf.to_container(cfg.models.tensor_parallel))
    for t in st.try_fallback:
        label = cfg.models.fallback_label[t]
        models[label] = cfg.models.fallback[t]
        tp[label] = tp[t]
    order = [cfg.models.fallback_label[t] for t in st.try_fallback if t == "S"] + list(cfg.models.tiers) + \
            [cfg.models.fallback_label[t] for t in st.try_fallback if t == "L"]
    return {"models": models, "tp": tp, "order": order, "helper": str(cfg.models.helper)}


def select_questions(cfg) -> pd.DataFrame:
    st = cfg.llm.stage4
    q = pd.read_parquet(repo_path(cfg.paths.questions))
    val = q[q.split == "val"].assign(group="val")
    pool = q[q.split == st.pilot_split]
    rng = np.random.default_rng(cfg.seeds.data)
    pilot = [g.loc[rng.choice(g.index.to_numpy(), size=min(st.pilot_per_source, len(g)), replace=False)]
             for _, g in pool.groupby("dataset", sort=True)]
    pilot = pd.concat(pilot).assign(group="pilot")
    return pd.concat([val, pilot]).sort_values(["group", "qid"]).reset_index(drop=True)


def k_choices(cfg, group: str) -> list[int]:
    return list(cfg.llm.stage4.val_k) if group == "val" else list(cfg.retrieval.K_choices)


class Store:
    """Passage text by doc id: data/passages.parquet first, the wiki-18 corpus for the rest
    (new passages are appended to passages.parquet so later stages find them)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.path = repo_path(cfg.paths.passages)
        self.df = pd.read_parquet(self.path).set_index("doc_id")

    def ensure(self, doc_ids) -> None:
        missing = sorted(set(int(d) for d in doc_ids) - set(self.df.index))
        if not missing:
            return
        from budgetrag.retrieval.corpus import Corpus
        c = Corpus(repo_path(self.cfg.paths.corpus_jsonl), repo_path(self.cfg.paths.corpus_offsets))
        new = pd.DataFrame(c.get(missing)).set_index("doc_id")
        c.close()
        self.df = pd.concat([self.df, new])
        tmp = self.path.with_suffix(".tmp")
        self.df.reset_index().to_parquet(tmp, index=False)
        tmp.replace(self.path)
        print(f"    {len(missing)} new passages added to {self.path.name}")

    def docs(self, ids) -> list[dict]:
        return [{"title": self.df.at[int(i), "title"], "text": self.df.at[int(i), "text"]} for i in ids]


# ---- jobs -----------------------------------------------------------------------------

def out_dir(cfg) -> Path:
    return repo_path(cfg.llm.stage4.outputs)


def jobs_file(cfg, r: int) -> Path:
    return repo_path(cfg.llm.stage4.jobs_dir) / f"stage4_r{r}.jsonl.gz"


def make_job(tok, templates, calls, row, node, kind, tier, store) -> dict:
    ids = list(node["doc_ids"])
    if kind == "answer" and not ids:
        tmpl, call = templates["closed"], calls["closed"]          # root: closed-book answer prompt
    else:
        tmpl, call = templates[kind], calls[kind]
    text = fill(tmpl, row.question, store.docs(ids) if ids else None)
    return {"job_id": f"{row.qid}|{tree.node_id(node['path'])}|{kind}|{tier}", "qid": row.qid,
            "group": row.group, "path": node["path"], "kind": kind, "tier": tier,
            "max_tokens": int(call.max_tokens), "stop": list(call.stop), "doc_ids": ids,
            "prompt": render_chat(tok, text)}


def write_jobs(cfg, r: int, jobs: list[dict], tr: dict, run_tiers: list[str]) -> None:
    order = {t: i for i, t in enumerate(tr["order"])}
    kinds = {"tentative": 0, "subquery": 1, "answer": 2}
    jobs.sort(key=lambda j: (order[j["tier"]], kinds[j["kind"]], j["qid"], j["path"]))
    path = jobs_file(cfg, r)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs)
    counts = pd.Series([j["tier"] for j in jobs]).value_counts().to_dict()
    meta = {"stage": 4, "round": r, "profile": cfg.profile_name, "prompt_version": cfg.llm.prompt_version,
            "git_hash": git_hash(), "tokenizer": cfg.tokenizer, "n_jobs": len(jobs), "jobs_per_tier": counts,
            "run_tiers": [t for t in tr["order"] if t in run_tiers and t in counts],
            "models": tr["models"], "tensor_parallel": tr["tp"], "helper": tr["helper"],
            "dtype": cfg.models.dtype, "max_model_len": int(cfg.models.max_model_len), "seed": int(cfg.llm.seed),
            "gpu_memory_utilization": float(cfg.llm.gpu_memory_utilization), "batch_size": int(cfg.llm.batch_size)}
    path.with_name(f"stage4_r{r}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"round {r}: {len(jobs)} jobs {counts} -> {path} ({path.stat().st_size / 1e6:.2f} MB)")


def read_round(cfg, r: int, tiers_needed: list[str]) -> pd.DataFrame:
    d = out_dir(cfg) / f"r{r}"
    frames = []
    for t in tiers_needed:
        f = d / f"{t}.jsonl"
        if not f.exists():
            raise SystemExit(f"missing {f}: unzip stage4_r{r}.zip from Kaggle into {d}")
        frames.append(pd.read_json(f, lines=True, dtype={"text": str}))
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("")
    return df


def load_jobs(cfg, r: int) -> pd.DataFrame:
    with gzip.open(jobs_file(cfg, r), "rt", encoding="utf-8") as f:
        rows = [{k: v for k, v in json.loads(line).items() if k != "prompt"} for line in f]
    return pd.DataFrame(rows)


# ---- rounds ---------------------------------------------------------------------------

def nodes_path(cfg) -> Path:
    return out_dir(cfg) / "nodes.parquet"


def save_nodes(cfg, nodes: list[dict]) -> None:
    df = pd.DataFrame(nodes, columns=NODE_COLS)
    df["subquery"] = df["subquery"].astype(object)
    out = nodes_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def round_1(cfg, tok, templates, calls, tr, store) -> None:
    qs = select_questions(cfg)
    cache = pd.read_parquet(repo_path(cfg.paths.retrieval_cache))
    cache = cache[cache.retriever == cfg.retrieval.primary].set_index("query")
    nodes, jobs = [], []
    for row in qs.itertuples():
        hits = [int(i) for i in cache.loc[row.question, "doc_ids"]]
        level = [{"path": "", "doc_ids": []}]
        level += [{"path": tree.child_path("", k), "doc_ids": tree.extend([], hits, k)} for k in k_choices(cfg, row.group)]
        store.ensure(d for n in level for d in n["doc_ids"])
        for n in level:
            n.update(qid=row.qid, group=row.group, depth=tree.depth(n["path"]), subquery=None, subquery_raw=None,
                     search_query=row.question if n["path"] else None)
            nodes.append(n)
            jobs.append(make_job(tok, templates, calls, row, n, "tentative", tr["helper"], store))
            if 0 < n["depth"] < cfg.llm.stage4.max_depth:
                jobs.append(make_job(tok, templates, calls, row, n, "subquery", tr["helper"], store))
    save_nodes(cfg, nodes)
    write_jobs(cfg, 1, jobs, tr, [tr["helper"]])


def grow(cfg, r: int, tr, store) -> tuple[pd.DataFrame, list[dict], dict]:
    """Read round r's sub-queries for the depth-r nodes, retrieve them, add the depth r+1 nodes."""
    nodes = pd.read_parquet(nodes_path(cfg))
    if (nodes.depth > r).any():
        print(f"    depth {r + 1} nodes already built; rebuilding them")
        nodes = nodes[nodes.depth <= r].copy()
    out = read_round(cfg, r, [tr["helper"]])
    sq = out[out.kind == "subquery"].copy()
    sq["path"] = sq.job_id.str.split("|").str[1]
    sq = sq.set_index(["qid", "path"])
    q = pd.read_parquet(repo_path(cfg.paths.questions)).set_index("qid")

    nodes = nodes.to_dict("records")
    frontier = [n for n in nodes if n["depth"] == r]
    for n in frontier:
        raw = sq.at[(n["qid"], n["path"]), "text"]
        query = extract_query(raw)
        n["subquery_raw"], n["subquery"] = raw, query
        n["search_query"] = query or q.at[n["qid"], "question"]     # empty sub-query: search the question
    from budgetrag.retrieval.client import E5Retriever
    wiki = repo_path(cfg.paths.wiki18)
    retr = E5Retriever([wiki / p for p in cfg.paths.index_parts], repo_path(cfg.paths.retrieval_memo),
                       cfg.models.retriever, store_k=cfg.retrieval.topk)
    queries = [n["search_query"] for n in frontier]
    t0 = time.time()
    last = [0.0]

    def progress(done, total):
        if time.time() - last[0] > 120 or done == total:
            last[0] = time.time()
            print(f"    {done / total:6.1%} of index scanned, {time.time() - t0:.0f} s", flush=True)

    n_new = len(set(queries)) - len(retr.memo.get_many(retr.name, queries, cfg.retrieval.topk))
    hits = retr.retrieve(queries, k=cfg.retrieval.topk, progress=progress)
    timing = {"round": r, "queries": len(queries), "distinct": len(set(queries)), "new": n_new,
              "seconds": time.time() - t0}
    retr.close()
    print(f"    retrieved {len(queries)} sub-queries ({n_new} new) in {timing['seconds']:.0f} s")

    new_nodes = []
    for n, h in zip(frontier, hits):
        ids = [d for d, _ in h]
        for k in k_choices(cfg, n["group"]):
            c = {"qid": n["qid"], "group": n["group"], "path": tree.child_path(n["path"], k), "depth": r + 1,
                 "doc_ids": tree.extend(list(n["doc_ids"]), ids, k), "subquery": None, "subquery_raw": None,
                 "search_query": n["search_query"]}
            new_nodes.append(c)
    store.ensure(d for c in new_nodes for d in c["doc_ids"])
    nodes += new_nodes
    save_nodes(cfg, nodes)
    log = out_dir(cfg) / "retrieval_log.json"
    hist = [x for x in (json.loads(log.read_text()) if log.exists() else []) if x["round"] != r]
    log.write_text(json.dumps(hist + [timing], indent=2))
    return pd.DataFrame(nodes), new_nodes, timing


def round_n(cfg, r: int, tok, templates, calls, tr, store) -> None:
    _, new_nodes, _ = grow(cfg, r - 1, tr, store)
    q = pd.read_parquet(repo_path(cfg.paths.questions)).set_index("qid")
    rows = {qid: q.loc[qid] for qid in {n["qid"] for n in new_nodes}}
    jobs = []
    for n in new_nodes:
        row = rows[n["qid"]]
        row = pd.Series({"qid": n["qid"], "group": n["group"], "question": row["question"]})
        jobs.append(make_job(tok, templates, calls, row, n, "tentative", tr["helper"], store))
        if n["depth"] < cfg.llm.stage4.max_depth:
            jobs.append(make_job(tok, templates, calls, row, n, "subquery", tr["helper"], store))
    run = [tr["helper"]]
    if r == 3:                                   # last round: every tier answers every node
        allnodes = pd.read_parquet(nodes_path(cfg))
        for n in allnodes.to_dict("records"):
            row = pd.Series({"qid": n["qid"], "group": n["group"], "question": q.at[n["qid"], "question"]})
            for t in tr["order"]:
                jobs.append(make_job(tok, templates, calls, row, n, "answer", t, store))
        run = tr["order"]
    write_jobs(cfg, r, jobs, tr, run)


# ---- report ---------------------------------------------------------------------------

def pct(x) -> str:
    return "" if x is None or pd.isna(x) else f"{100 * x:.1f}"


def report(cfg) -> None:
    st = cfg.llm.stage4
    tr = tiers(cfg)
    q = pd.read_parquet(repo_path(cfg.paths.questions)).set_index("qid")
    nodes = pd.read_parquet(nodes_path(cfg))

    frames, missing, metas = [], [], {}
    for r in (1, 2, 3):
        jm = json.loads(jobs_file(cfg, r).with_name(f"stage4_r{r}.meta.json").read_text(encoding="utf-8"))
        jobs = load_jobs(cfg, r)
        for t in jm["run_tiers"]:
            f = out_dir(cfg) / f"r{r}" / f"{t}.jsonl"
            if not f.exists():
                missing.append(f"r{r}/{t}")
                continue
            o = pd.read_json(f, lines=True, dtype={"text": str})
            mf = f.with_suffix(".meta.json")
            metas[(r, t)] = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
            o = o.drop(columns=["qid", "tier", "kind"]).merge(jobs, on="job_id", how="right")
            o = o[o.tier == t]
            o["round"] = r
            frames.append(o)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("")
    df["complete"] = df.n_in.notna()
    df["dataset"] = df.qid.map(q["dataset"])
    golds = df.qid.map(q["answers"]).map(list)
    df["pred"] = [extract_query(t) if k == "subquery" else extract_answer(t) for t, k in zip(df.text, df.kind)]
    scored = df.kind != "subquery"
    df["em"] = [metrics.exact_match(p, g) if s else np.nan for p, g, s in zip(df.pred, golds, scored)]
    df["f1"] = [metrics.f1(p, g) if s else np.nan for p, g, s in zip(df.pred, golds, scored)]
    df.to_parquet(out_dir(cfg) / "pilot.parquet", index=False)

    ok_rows = bool(df.complete.all()) and not missing
    ans = df[(df.kind == "answer") & df.complete]
    val = ans[ans.group == "val"]
    lines = [f"# Stage 4 report ({cfg.profile_name} profile)", "",
             f"Pilot and tier-gap gate. Prompts {cfg.llm.prompt_version}, greedy decoding, E5 top-10 per search, "
             f"helper `{tr['helper']}` ({tr['models'][tr['helper']]}) writes sub-queries and tentative answers. "
             f"Three Kaggle rounds (2x T4, vLLM, float16) with E5 retrieval of new sub-queries on the workstation "
             "between them.", "",
             "Tiers: " + ", ".join(f"{t} = {tr['models'][t]}" for t in tr["order"]) +
             (". " + ", ".join(f"{cfg.models.fallback_label[t]} is the gate fallback for {t}" for t in st.try_fallback) + "."
              if st.try_fallback else "."), "",
             f"Completeness: {int(df.complete.sum())} / {len(df)} calls returned" +
             (f"; missing outputs: {', '.join(missing)}" if missing else "") + ".", ""]

    # 9 configurations
    configs = {"no retrieval": "", "top-3, one step": "3", "fixed 3-step (3-3-3)": "3-3-3"}
    lines += ["## The 9 configurations (validation)", "",
              f"{val.qid.nunique()} validation questions. F1 / EM in %. "
              "No retrieval uses the closed-book prompt; the 3-step path searches the question, then two "
              "helper-written sub-queries, adding 3 new passages each time (9 passages in the end).", "",
              "| Tier | Config | NQ F1 | HotpotQA F1 | All F1 | All EM |", "|---|---|---|---|---|---|"]
    f1 = {}
    for t in tr["order"]:
        for name, path in configs.items():
            d = val[(val.tier == t) & (val.path == path)]
            if d.empty:
                continue
            f1[(t, path)] = 100 * d.f1.mean()
            f1[(t, path, "hotpotqa")] = 100 * d[d.dataset == "hotpotqa"].f1.mean()
            lines.append(f"| {t} | {name} | {pct(d[d.dataset == 'nq'].f1.mean())} | "
                         f"{pct(d[d.dataset == 'hotpotqa'].f1.mean())} | {pct(d.f1.mean())} | {pct(d.em.mean())} |")
    lines.append("")

    # gate
    gmin = float(st.gate_min_gap)
    g = lambda a, b: f1.get((b, "3"), np.nan) - f1.get((a, "3"), np.nan)
    base = list(cfg.models.tiers)
    lines += ["## Tier-gap gate", "",
              f"Rule (fixed in the plan before any run): with top-3 retrieval on the merged validation set, "
              f"F1(M) - F1(S) >= {gmin:g} and F1(L) - F1(M) >= {gmin:g} points. If M - S is too small, S is "
              f"replaced by {cfg.models.fallback.S}; if L - M is too small, L by {cfg.models.fallback.L}. "
              "Stage 3 had already shown M - S = +2.2 on these questions, so the S fallback was run in the "
              "same rounds instead of a fourth one.", ""]
    final = {t: tr["models"][t] for t in base}
    attempts = [("S, M, L", base)]
    gap_ms, gap_lm = g("S", "M"), g("M", "L")
    gate_ok = gap_ms >= gmin and gap_lm >= gmin
    lines.append(f"- Attempt 1, S/M/L = {', '.join(tr['models'][t].split('/')[-1] for t in base)}: "
                 f"M - S = {gap_ms:+.1f}, L - M = {gap_lm:+.1f} -> **{'pass' if gate_ok else 'fail'}**.")
    if not gate_ok:
        s_t, l_t = "S", "L"
        if gap_ms < gmin and "S" in st.try_fallback:
            s_t = cfg.models.fallback_label.S
        if gap_lm < gmin and "L" in st.try_fallback:
            l_t = cfg.models.fallback_label.L
        need = [x for x, gap in (("S", gap_ms), ("L", gap_lm)) if gap < gmin and x not in st.try_fallback]
        if need:
            lines.append(f"- The fallback for {', '.join(need)} was not run: add it to `llm.stage4.try_fallback` "
                         "and repeat round 3.")
        else:
            gap_ms2, gap_lm2 = g(s_t, "M"), g("M", l_t)
            gate_ok = gap_ms2 >= gmin and gap_lm2 >= gmin
            final = {"S": tr["models"][s_t], "M": tr["models"]["M"], "L": tr["models"][l_t]}
            lines.append(f"- Attempt 2, S/M/L = {', '.join(v.split('/')[-1] for v in final.values())}: "
                         f"M - S = {gap_ms2:+.1f}, L - M = {gap_lm2:+.1f} -> **{'pass' if gate_ok else 'fail'}**.")
    lines += ["", f"Final tiers: " + ", ".join(f"{k} = {v}" for k, v in final.items()) +
              f"; the helper stays {tr['models'][tr['helper']]}.", ""]
    final_labels = {k: next(t for t in tr["order"] if tr["models"][t] == v) for k, v in final.items()}

    # retrieval helps on HotpotQA
    helps = []
    for k, t in final_labels.items():
        closed = f1.get((t, "", "hotpotqa"), np.nan)
        one, three = f1.get((t, "3", "hotpotqa"), np.nan), f1.get((t, "3-3-3", "hotpotqa"), np.nan)
        helps.append(one > closed and three > closed)
        lines.append(f"- HotpotQA, {k} ({t}): no retrieval {closed:.1f}, top-3 {one:.1f}, 3-step {three:.1f} "
                     f"-> retrieval {'helps' if helps[-1] else 'does NOT help'}.")
    retrieval_ok = all(helps)
    lines += ["", f"Retrieval check (F1 with retrieval > without on HotpotQA, every final tier): "
              f"**{'pass' if retrieval_ok else 'fail'}**.", ""]

    # consistency with Stage 3
    s3 = repo_path(cfg.llm.stage3.outputs) / "scored.parquet"
    if s3.exists():
        s3 = pd.read_parquet(s3)
        s3 = s3[(s3.kind == "answer") & (s3["pass"] == 1)].set_index(["tier", "qid"]).text
        v1 = val[val.path == "3"].set_index(["tier", "qid"]).text
        common = v1.index.intersection(s3.index)
        if len(common):
            same = (v1.loc[common] == s3.loc[common]).mean()
            lines += [f"Same top-3 answer prompts as Stage 3: {pct(same)}% of {len(common)} outputs are identical "
                      "(a separate Kaggle session, other batch neighbours).", ""]

    # K set
    pil = ans[ans.group == "pilot"]
    lines += ["## K set", ""]
    rows = []
    for t in tr["order"]:
        d = pil[pil.tier == t].set_index(["qid", "path"])
        pairs = []
        for (qid, path) in d.index:
            if path.endswith("3") and (path == "3" or path.endswith("-3")):
                sib = path[:-1] + "5"
                if (qid, sib) in d.index:
                    pairs.append((d.at[(qid, path), "pred"], d.at[(qid, sib), "pred"],
                                  d.at[(qid, path), "f1"], d.at[(qid, sib), "f1"]))
        if pairs:
            p = pd.DataFrame(pairs, columns=["a3", "a5", "f3", "f5"])
            changed = (p.a3.map(metrics.normalize_answer) != p.a5.map(metrics.normalize_answer)).mean()
            rows.append((t, len(p), changed, 100 * (p.f5 - p.f3).mean()))
    if rows:
        lines += ["Sibling nodes that differ only in the last K (3 vs 5), training-pool pilot:", "",
                  "| Tier | Pairs | Answer changes | Mean F1(K=5) - F1(K=3) |", "|---|---|---|---|"]
        lines += [f"| {t} | {n} | {pct(c)}% | {d:+.1f} |" for t, n, c, d in rows]
        min_change = min(c for t, _, c, _ in rows if t in final_labels.values())
        kset = list(cfg.retrieval.K_choices) if min_change >= st.k_change_min else [3, 10]
        lines += ["", f"Rule: keep {{3, 5}} unless K = 5 changes the answer on fewer than {pct(st.k_change_min)}% "
                  f"of pairs for some final tier. Lowest change rate: {pct(min_change)}% -> K set "
                  f"**{{{', '.join(map(str, kset))}}}**.", ""]

    # cost pilot
    pq = df[(df.group == "pilot") & df.complete]
    nq = pq.qid.nunique()
    lines += ["## Cost pilot", "",
              f"Full Stage 6 tree (K in {{3, 5}}, 15 nodes) on {nq} training-pool questions. Per question:", "",
              "| Model | Role | Calls | Input tokens | Output tokens | GPU-s | Wall-s |", "|---|---|---|---|---|---|---|"]
    per_q = {}
    for t in tr["order"]:
        for role, sel in (("helper", pq.kind != "answer"), ("answers", pq.kind == "answer")):
            d = pq[(pq.tier == t) & sel]
            if d.empty:
                continue
            tp = tr["tp"][t]
            per_q[(t, role)] = {"calls": len(d) / nq, "n_in": d.n_in.sum() / nq, "n_out": d.n_out.sum() / nq,
                                "gpu_s": d.gpu_s.sum() / nq, "wall_s": d.gpu_s.sum() / tp / nq}
            v = per_q[(t, role)]
            lines.append(f"| {t} ({tr['models'][t].split('/')[-1]}) | {role} | {v['calls']:.1f} | {v['n_in']:.0f} | "
                         f"{v['n_out']:.0f} | {v['gpu_s']:.2f} | {v['wall_s']:.2f} |")
    used = [(tr["helper"], "helper")] + [(t, "answers") for t in final_labels.values()]
    gpu_q = sum(per_q.get(k, {}).get("gpu_s", 0) for k in used)
    wall_q = sum(per_q.get(k, {}).get("wall_s", 0) for k in used)
    n_cache = {"train_pool": int((q.split == "train_pool").sum()), "val": int((q.split == "val").sum()),
               "test": int(q.split.str.startswith("test_").sum())}
    n_tot = sum(n_cache.values())
    loads = sum(m.get("load_s", 0) for (r, t), m in metas.items())
    gpu_h, wall_h = gpu_q * n_tot / 3600, wall_q * n_tot / 3600
    setup_h = 3 * 0.25 + loads / 3600                     # 3 rounds: ~15 min venv + downloads each, plus model loads
    total_h = wall_h + setup_h
    fits = total_h <= float(st.kaggle_quota_h)
    log = out_dir(cfg) / "retrieval_log.json"
    rlog = json.loads(log.read_text()) if log.exists() else []
    lines += ["", f"Final tiers plus helper: {gpu_q:.2f} GPU-s and {wall_q:.2f} wall-s of generation per question.", "",
              f"Projected outcome cache ({cfg.profile_name}: {n_cache['train_pool']:,} train + {n_cache['val']:,} "
              f"validation + {n_cache['test']:,} test = {n_tot:,} questions): **{gpu_h:.1f} GPU-hours** "
              f"({wall_h:.1f} h of Kaggle session time for generation, plus about {setup_h:.1f} h for three rounds "
              f"of setup and model loading = **{total_h:.1f} h**) against a quota of {st.kaggle_quota_h} h per week "
              f"-> {'fits' if fits else 'does NOT fit'} in one week.", ""]
    if rlog:
        lines += ["Workstation retrieval between rounds: " + "; ".join(
            f"after round {x['round']}: {x['new']} new queries in {x['seconds'] / 60:.1f} min" for x in rlog) +
            f". The full cache adds about {6 * n_tot:,} sub-query searches in two passes (one per depth).", ""]

    # done when
    done = ok_rows and gate_ok and retrieval_ok and fits
    lines += ["## Done-when check", "",
              f"- All calls returned: {'yes' if ok_rows else 'no'}",
              f"- Tier-gap gate passes (after the planned fallback): {'yes' if gate_ok else 'no'}",
              f"- Retrieval helps on HotpotQA for every final tier: {'yes' if retrieval_ok else 'no'}",
              f"- Projected cache build fits the Kaggle quota: {'yes' if fits else 'no'}", "",
              f"Result: **{'pass' if done else 'not done'}**", ""]

    # sub-query notes
    sq = nodes[nodes.subquery.notna()]
    if not sq.empty:
        restate = (sq.subquery.map(metrics.normalize_answer) == sq.qid.map(q["question"]).map(metrics.normalize_answer))
        empty = sq.subquery.str.len() == 0
        lines += ["## Sub-queries", "",
                  f"{len(sq)} helper sub-queries: {pct(restate.mean())}% repeat the question, {pct(empty.mean())}% empty "
                  "(those search the question instead).", ""]

    rep = repo_path(cfg.paths.reports) / "STAGE_4_REPORT.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    (out_dir(cfg) / "decision.json").write_text(json.dumps(
        {"final_tiers": final, "gate_pass": bool(gate_ok), "retrieval_pass": bool(retrieval_ok),
         "projected_gpu_h": gpu_h, "projected_session_h": total_h, "done": bool(done)}, indent=2))
    save_run_config(cfg, repo_path(cfg.paths.reports) / "stage4")
    print("\n".join(lines))
    if not done:
        raise SystemExit(1)


# ---- entry ----------------------------------------------------------------------------

def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("round", "report"):
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    r = int(rest.pop(0)) if cmd == "round" else 0
    cfg = load_config(rest)
    if cmd == "report":
        return report(cfg)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    templates = {kind: load_template(c.prompt, cfg.llm.prompt_version) for kind, c in cfg.llm.calls.items()}
    calls = cfg.llm.calls
    tr = tiers(cfg)
    store = Store(cfg)
    if r == 1:
        round_1(cfg, tok, templates, calls, tr, store)
    elif r in (2, 3):
        round_n(cfg, r, tok, templates, calls, tr, store)
    else:
        raise SystemExit("round must be 1, 2 or 3")


if __name__ == "__main__":
    main(sys.argv[1:])
