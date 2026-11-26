# Stage 2 report (lite profile)

Retriever: intfloat/e5-base-v2, Search-R1's flat E5 index over 21,015,324 Wikipedia-2018 passages, exact inner-product search on CPU (index streamed from disk).

- Queries retrieved: 9,120 (all split questions + 1 sanity query); cache rows: 9,120; distinct passages: 84,404.
- Time for encoding + one full index pass: 38.5 min (254 ms per query amortised).
- Sanity: top passage for "Who wrote Hamlet?" is ""Hamlet (band)"" — mentions Shakespeare (pass).
  Top 5 by score: "Hamlet (band)" 0.854, "Hamlet (band)" 0.852, Hamlet (the play: "a tragedy
  written by William Shakespeare") 0.849, "Critical approaches to Hamlet" 0.846, Hamlet 0.839.
  The answer passage is at rank 3; the two band passages above it are a known dense-retrieval
  failure on very short queries.

Answer recall@k (a gold answer string appears in the top-k passages), test samples:

| Dataset | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|
| 2wikimultihopqa | 19.0% | 31.0% | 38.6% | 45.8% |
| bamboogle | 9.6% | 20.8% | 22.4% | 27.2% |
| hotpotqa | 26.8% | 38.0% | 44.0% | 51.2% |
| musique | 7.6% | 14.6% | 18.2% | 26.4% |
| nq | 56.0% | 71.8% | 78.0% | 84.4% |
| popqa | 45.6% | 60.6% | 66.2% | 72.2% |
| triviaqa | 57.6% | 71.2% | 76.4% | 81.6% |

Training-side splits (train_pool, val, live_ft):

| Dataset | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|
| hotpotqa | 36.6% | 50.0% | 56.3% | 62.5% |
| nq | 54.6% | 71.8% | 77.7% | 83.0% |

BM25 (Search-R1's Lucene index, needed only for the robustness experiment E6) is not built yet: Pyserini needs Java 21; planned as a Kaggle CPU job.
