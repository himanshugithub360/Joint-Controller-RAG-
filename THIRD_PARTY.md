# Third-party code

The thesis reuses these repositories rather than rebuilding them. Each is pinned to one
commit (the default branch's head on 2026-11-27). `python scripts/s00_fetch_third_party.py`
fetches them into `third_party/<name>/` (git-ignored). Reused code is not edited in place.

| Name | Repository | Pinned commit | Licence | Used for |
|---|---|---|---|---|
| Search-R1 | [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) | `598e61bd1d36895726d28a8d06b3a15bed19f5d3` | Apache-2.0 | Retrieval server, corpus and E5 index download, EM/F1, baseline checkpoints |
| FlashRAG | [RUC-NLPIR/FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | `7926ac3c3c5ce0781f53d60ebac48fc511abed12` | MIT | Benchmark datasets, BM25 index builder, evaluation code |
| AutoSearch | [bofusun/AutoSearch](https://github.com/bofusun/AutoSearch) | `75ce1c5aba097ded06318a2b1347f02175452eec` | Apache-2.0 | Intermediate-answer prompt, earliest-correct-depth label, baseline |
| Adaptive-RAG | [starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) | `0c88670af8707667eb5c1163151bb5ce61b14acb` | Apache-2.0 | Query-complexity classifier, baseline |
| RouteLLM | [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) | `0b64fdafe049e596a3f5657c219329f24af24198` | Apache-2.0 | Model-routing baselines |
| Router-R1 | [ulab-uiuc/Router-R1](https://github.com/ulab-uiuc/Router-R1) | `801a240e37701577907c32de27a548af4e6c4430` | Apache-2.0 | Multi-round routing baseline |

Python packages reused without copying code: [OmniSafe](https://github.com/PKU-Alignment/omnisafe)
0.5.0 (Apache-2.0) for PPO-Lagrangian, vLLM for serving on Kaggle, bm25s for CPU BM25,
FAISS for dense search. Versions are pinned in `envs/`.
