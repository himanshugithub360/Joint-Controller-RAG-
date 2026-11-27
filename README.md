# Joint-Controller-RAG

Budget-constrained joint retrieval and LLM routing: a small learned controller decides, for
each question and a cost budget B₀, whether to **RETRIEVE(K)** more passages or
**ANSWER(M)** with a small, medium or large LLM. It is trained as a constrained POMDP
(behaviour cloning, then PPO-Lagrangian with an action mask) on an offline outcome cache,
then fine-tuned and evaluated live. M.Tech thesis code.

## Where things run

This project runs on one workstation plus free Kaggle GPUs.

| Machine | Hardware | Runs |
|---|---|---|
| Workstation (Windows) | Quadro P4000 8 GB, Xeon 16 threads, 64 GB RAM, ~1.8 TB free | Data, BM25, features, simulator, controller training, evaluation, statistics |
| Kaggle notebook | 2× T4 16 GB, ~30 GPU-hours/week | All LLM calls (vLLM): pilots, outcome cache, live fine-tuning and evaluation |

vLLM needs compute capability ≥ 7.0, so the P4000 (6.1) cannot serve models. Kaggle's
**T4 x2** accelerator (7.5) can; its P100 option cannot.

### Scale profiles

`configs/profile/lite.yaml` (default) fits the hardware above; `full.yaml` keeps the
original plan for a later run on bigger GPUs.

| | lite | full |
|---|---|---|
| Model tiers S / M / L | Qwen2.5 1.5B / 3B / 7B Instruct (float16) | Qwen2.5 3B / 7B / 32B Instruct (bfloat16) |
| Training pool | 5,000 (2,500 NQ + 2,500 HotpotQA) | 20,000 |
| Validation / live fine-tuning | 500 / 500 | 1,000 / 2,000 |
| Test | 500 per dataset (Bamboogle 125) ≈ 3,125 | all test splits ≈ 51,700 |
| Primary retriever | E5 over 21M passages, exact search on CPU (Search-R1's index) | E5 + FAISS over 21M passages |

Lite results are not directly comparable with Search-R1's published tables; the full
profile is kept so they can be.

## Setup

Local (Anaconda is at `D:\anaconda3` on the workstation):

```powershell
conda env create -f envs/local.yml
conda activate budgetrag
pip install -e .
python scripts/s00_check_env.py --label local
python scripts/s00_fetch_third_party.py
pytest
```

Kaggle: upload `kaggle/stage0_smoke.ipynb`, set Accelerator to **GPU T4 x2** and Internet
on, add a `GITHUB_TOKEN` secret if this repo is private, then run all cells.

## Layout

```
configs/              Hydra config, scale profiles, versioned prompt templates (configs/prompts/v1)
envs/                 local conda env, Kaggle pip requirements
src/budgetrag/        thesis code (data, retrieval, llm, eval now; cost, cache, env, features, controller next)
scripts/sNN_*.py      one entry script per stage
kaggle/               notebooks and scripts that run on Kaggle GPUs
tests/                pytest unit tests
data/splits/          frozen split ID lists (committed); everything else in data/ is rebuilt
reports/              STAGE_N reports (thesis appendix material)
third_party/          reused repos at pinned commits (fetched, not committed; see THIRD_PARTY.md)
```

## Stages

| Stage | What | Status |
|---|---|---|
| 0 | Environments, pinned third-party code, smoke tests | Done: local and Kaggle pass, all three tiers answer (`reports/STAGE_0_REPORT.md`) |
| 1 | Datasets and frozen splits | Done, lite profile: 9,125 questions (`reports/STAGE_1_REPORT.md`) |
| 2 | Retrieval (BM25, E5) and retrieval cache | E5 done: top-10 for all 9,125 questions (`reports/STAGE_2_REPORT.md`); BM25 pending (only for E6) |
| 3 | Model serving on Kaggle, prompts, EM/F1 | Done: all three tiers answer the 500 validation questions; top-3 answer F1 1.5B 44.8 / 3B 47.0 / 7B 50.6; 99.5–100% deterministic (`reports/STAGE_3_REPORT.md`) |
| 4 | Pilot and tier-gap gate | |
| 5 | Cost model and budgets | |
| 6 | Outcome cache | |
| 7 | Simulator, oracle, faithfulness gate | |
| 8–12 | Features, controller, behaviour cloning, PPO-Lagrangian, live fine-tuning | |
| 13–16 | Baselines, evaluation, statistics, release | |

Conventions: seed 42 for data sampling, 0–4 for training; questions are keyed
`<dataset>-<flashrag id>` (e.g. `nq-train_0`) and read only through `data/splits/`;
each stage writes `reports/STAGE_N_REPORT.md`.
