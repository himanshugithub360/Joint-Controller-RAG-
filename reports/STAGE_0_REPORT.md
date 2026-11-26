# Stage 0 report: environments and smoke tests

All Stage 0 smoke tests pass on both machines (2026-11-27).

| Machine | Details | Result |
|---|---|---|
| Workstation | Quadro P4000 8 GB (compute 6.1), 16 threads, 64 GB RAM; conda env `budgetrag`, Python 3.10 | 15/15 packages import; pytest passes ([stage0_env_local.md](stage0_env_local.md)) |
| Kaggle | 2x Tesla T4 16 GB (compute 7.5), 4 CPUs, 31 GB RAM; vLLM env Python 3.12 | 5/5 packages import ([stage0_env_kaggle.md](stage0_env_kaggle.md)) |

Model smoke test on Kaggle ("What is the capital of France?", greedy, asked twice;
raw rows in [stage0_kaggle_smoke.jsonl](stage0_kaggle_smoke.jsonl)):

| Tier | Model | GPUs | Load (s) | Answer | Correct | Deterministic |
|---|---|---|---|---|---|---|
| S | Qwen2.5-1.5B-Instruct | 1 | 49.5 | Paris | yes | yes |
| M | Qwen2.5-3B-Instruct | 1 | 65.9 | Paris | yes | yes |
| L | Qwen2.5-7B-Instruct | 2 (tensor parallel) | 191.8 | Paris | yes | yes |

Problems met and fixes:

- The P4000 (compute 6.1) cannot run vLLM (needs 7.0+), so all LLM calls run on Kaggle.
- Kaggle's image is Python 3.13; vLLM 0.6.6 needs xformers wheels that stop at 3.12. vLLM
  is installed in a separate Python 3.12 env made with uv (`/tmp/vllm-env`).
- In that env, pyarrow >= 18 needs NumPy 2 but vLLM pins NumPy 1.26: pyarrow pinned to 17.0.0.
- OmniSafe 0.5.0 pins pandas 2.0.3 exactly: the local env uses that version.
- On Windows, the faiss-cpu and torch wheels ship different OpenMP runtimes and abort when
  both run in one process. Project code does not use faiss (exact search is done in torch);
  tests call faiss only in a subprocess.
- Reused repositories are pinned in THIRD_PARTY.md and fetched by a script rather than added
  as git submodules.
