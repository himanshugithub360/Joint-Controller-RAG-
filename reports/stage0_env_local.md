# Stage 0 environment: local

Recorded 2026-11-26 20:47 UTC.

| Item | Value |
|---|---|
| OS | Windows-10-10.0.26200-SP0 |
| CPU | Intel64 Family 6 Model 85 Stepping 4, GenuineIntel, 16 logical cores |
| RAM | 63.7 GB |
| Disk free at repo | 271 GB of 459 GB |
| Python | 3.10.22 (D:\anaconda3\envs\budgetrag\python.exe) |
| PyTorch | torch 2.5.1+cpu: CUDA not available (CPU build or no usable GPU) |
| GPU 0 | Quadro P4000, 8192 MiB, driver 571.59, compute capability 6.1 |

## Import smoke test

| Module | Version | OK |
|---|---|---|
| torch | 2.5.1+cpu | yes |
| transformers | 4.46.3 | yes |
| datasets | 3.1.0 | yes |
| faiss | 1.9.0 | yes |
| bm25s | 0.2.5 | yes |
| gymnasium | 0.28.1 | yes |
| omnisafe | 0.5.0 | yes |
| pandas | 2.0.3 | yes |
| pyarrow | 18.1.0 | yes |
| hydra | 1.3.2 | yes |
| scipy | 1.14.1 | yes |
| sklearn | 1.5.2 | yes |
| matplotlib | 3.9.3 | yes |
| pytest | 8.3.4 | yes |
| budgetrag | 0.1.0 | yes |

Note: Quadro P4000 has compute capability 6.1; vLLM needs 7.0 or newer, so LLM serving runs on Kaggle T4s, not here.
