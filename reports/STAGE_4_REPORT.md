# Stage 4 report (lite profile)

Pilot and tier-gap gate. Prompts v1, greedy decoding, E5 top-10 per search, helper `S` (Qwen/Qwen2.5-1.5B-Instruct) writes sub-queries and tentative answers. Three Kaggle rounds (2x T4, vLLM, float16) with E5 retrieval of new sub-queries on the workstation between them.

Tiers: XS = Qwen/Qwen2.5-0.5B-Instruct, S = Qwen/Qwen2.5-1.5B-Instruct, M = Qwen/Qwen2.5-3B-Instruct, L = Qwen/Qwen2.5-7B-Instruct. XS is the gate fallback for S.

Completeness: 51500 / 51500 calls returned.

## The 9 configurations (validation)

500 validation questions. F1 / EM in %. No retrieval uses the closed-book prompt; the 3-step path searches the question, then two helper-written sub-queries, adding 3 new passages each time (9 passages in the end).

| Tier | Config | NQ F1 | HotpotQA F1 | All F1 | All EM |
|---|---|---|---|---|---|
| XS | no retrieval | 7.1 | 12.8 | 10.0 | 6.2 |
| XS | top-3, one step | 38.7 | 32.9 | 35.8 | 29.0 |
| XS | fixed 3-step (3-3-3) | 38.9 | 27.5 | 33.2 | 24.8 |
| S | no retrieval | 13.9 | 23.5 | 18.7 | 12.8 |
| S | top-3, one step | 49.3 | 40.8 | 45.0 | 35.4 |
| S | fixed 3-step (3-3-3) | 49.2 | 47.0 | 48.1 | 38.4 |
| M | no retrieval | 17.6 | 28.3 | 23.0 | 17.0 |
| M | top-3, one step | 50.7 | 43.6 | 47.2 | 38.6 |
| M | fixed 3-step (3-3-3) | 51.1 | 50.5 | 50.8 | 40.8 |
| L | no retrieval | 24.3 | 33.1 | 28.7 | 21.4 |
| L | top-3, one step | 52.4 | 48.8 | 50.6 | 41.4 |
| L | fixed 3-step (3-3-3) | 54.6 | 54.2 | 54.4 | 44.8 |

## Tier-gap gate

Rule (fixed in the plan before any run): with top-3 retrieval on the merged validation set, F1(M) - F1(S) >= 3 and F1(L) - F1(M) >= 3 points. If M - S is too small, S is replaced by Qwen/Qwen2.5-0.5B-Instruct; if L - M is too small, L by Qwen/Qwen2.5-14B-Instruct. Stage 3 had already shown M - S = +2.2 on these questions, so the S fallback was run in the same rounds instead of a fourth one.

- Attempt 1, S/M/L = Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct: M - S = +2.1, L - M = +3.5 -> **fail**.
- Attempt 2, S/M/L = Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct: M - S = +11.4, L - M = +3.5 -> **pass**.

Final tiers: S = Qwen/Qwen2.5-0.5B-Instruct, M = Qwen/Qwen2.5-3B-Instruct, L = Qwen/Qwen2.5-7B-Instruct; the helper stays Qwen/Qwen2.5-1.5B-Instruct.

- HotpotQA, S (XS): no retrieval 12.8, top-3 32.9, 3-step 27.5 -> retrieval helps.
- HotpotQA, M (M): no retrieval 28.3, top-3 43.6, 3-step 50.5 -> retrieval helps.
- HotpotQA, L (L): no retrieval 33.1, top-3 48.8, 3-step 54.2 -> retrieval helps.

Retrieval check (F1 with retrieval > without on HotpotQA, every final tier): **pass**.

Same top-3 answer prompts as Stage 3: 99.6% of 1500 outputs are identical (a separate Kaggle session, other batch neighbours).

## K set

Sibling nodes that differ only in the last K (3 vs 5), training-pool pilot:

| Tier | Pairs | Answer changes | Mean F1(K=5) - F1(K=3) |
|---|---|---|---|
| XS | 3500 | 29.6% | -0.7 |
| S | 3500 | 19.1% | +0.4 |
| M | 3500 | 19.9% | +0.2 |
| L | 3500 | 15.5% | +0.4 |

Rule: keep {3, 5} unless K = 5 changes the answer on fewer than 1.0% of pairs for some final tier. Lowest change rate: 15.5% -> K set **{3, 5}**.

## Cost pilot

Full Stage 6 tree (K in {3, 5}, 15 nodes) on 500 training-pool questions. Per question:

| Model | Role | Calls | Input tokens | Output tokens | GPU-s | Wall-s |
|---|---|---|---|---|---|---|
| XS (Qwen2.5-0.5B-Instruct) | answers | 15.0 | 21296 | 123 | 0.94 | 0.94 |
| S (Qwen2.5-1.5B-Instruct) | helper | 21.0 | 28501 | 309 | 3.96 | 3.96 |
| S (Qwen2.5-1.5B-Instruct) | answers | 15.0 | 21296 | 85 | 2.42 | 2.42 |
| M (Qwen2.5-3B-Instruct) | answers | 15.0 | 21296 | 83 | 4.32 | 4.32 |
| L (Qwen2.5-7B-Instruct) | answers | 15.0 | 21296 | 78 | 10.86 | 5.43 |

Final tiers plus helper: 20.08 GPU-s of generation per question; 14.65 s of session time if every model ran alone, 11.84 s with the two-GPU schedule of `kaggle/run_stage4.sh` (one-GPU models side by side, one per T4).

Projected outcome cache (lite: 5,000 train + 500 validation + 3,125 test = 8,625 questions): **48.1 GPU-hours**. Kaggle session time: 28.4 h of generation with the two-GPU schedule (35.1 h one model at a time), plus about 0.9 h for three rounds of setup and model loading = **29.2 h** against a quota of 30 h per week -> fits in one week's quota.

The margin is thin: the setup allowance is a rough 15 min per round, and the one-week bar is this report's own reading of "fits the available GPU time" (the cache may also be built over two weeks). Rounds 1 and 2 run only the helper and leave one T4 idle; splitting the helper's jobs over both GPUs would cut session time further if needed.

Workstation retrieval between rounds: after round 1: 1147 new queries in 9.0 min; after round 2: 834 new queries in 7.9 min. The full cache adds about 51,750 sub-query searches in two passes (one per depth).

## Done-when check

- All calls returned: yes
- Tier-gap gate passes (after the planned fallback): yes
- Retrieval helps on HotpotQA for every final tier: yes
- Projected cache build fits the Kaggle quota: yes

Result: **pass**

## Sub-queries

4000 helper sub-queries: 22.4% repeat the question, 0.0% empty (those search the question instead).
