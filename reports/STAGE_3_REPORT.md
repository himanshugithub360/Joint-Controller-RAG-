# Stage 3 report (lite profile)

Prompts v1 (configs/prompts/), greedy decoding, top-3 E5 passages, 500 validation questions, 2000 calls per tier (answer, closed-book, tentative, sub-query) plus 200 repeated answer calls. Generated on Kaggle (2x T4, vLLM, float16); scored on the workstation with Search-R1's EM and FlashRAG's token F1.

## Done-when check

| Tier | Model | Calls (pass 1) | Answer calls | Token counts on every call | Load (s) | Generate (s) | GPU-s |
|---|---|---|---|---|---|---|---|
| S | Qwen/Qwen2.5-1.5B-Instruct | 2000 / 2000 | 500 / 500 | yes | 55 | 165 | 165 |
| M | Qwen/Qwen2.5-3B-Instruct | 2000 / 2000 | 500 / 500 | yes | 71 | 334 | 334 |
| L | Qwen/Qwen2.5-7B-Instruct | 2000 / 2000 | 500 / 500 | yes | 200 | 388 | 775 |

Result: **pass**

## Answer quality (pass 1)

F1 / EM in %, max over gold answers.

| Tier | Call | NQ F1 | NQ EM | HotpotQA F1 | HotpotQA EM | All F1 | All EM |
|---|---|---|---|---|---|---|---|
| S | answer | 48.8 | 38.4 | 40.8 | 32.0 | 44.8 | 35.2 |
| S | closed | 13.9 | 8.4 | 23.5 | 17.2 | 18.7 | 12.8 |
| S | tentative | 45.2 | 34.0 | 36.6 | 28.0 | 40.9 | 31.0 |
| M | answer | 50.5 | 40.4 | 43.6 | 36.8 | 47.0 | 38.6 |
| M | closed | 17.6 | 10.4 | 28.3 | 23.6 | 23.0 | 17.0 |
| M | tentative | 49.3 | 39.6 | 40.2 | 33.6 | 44.8 | 36.6 |
| L | answer | 52.4 | 41.2 | 48.8 | 41.6 | 50.6 | 41.4 |
| L | closed | 24.3 | 15.6 | 33.1 | 27.2 | 28.7 | 21.4 |
| L | tentative | 51.4 | 40.8 | 46.4 | 38.0 | 48.9 | 39.4 |

F1 gaps with top-3 passages (preview of the Stage 4 gate, which needs >= 3 points each): M - S: +2.2; L - M: +3.6.

## Output checks (pass 1)

| Tier | Call | Empty % | Hit max_tokens % | Mean input tokens | Mean output tokens | Mean log-prob |
|---|---|---|---|---|---|---|
| S | answer | 0.0 | 0.2 | 559 | 5.3 | -0.103 |
| S | closed | 0.0 | 0.0 | 76 | 5.0 | -0.516 |
| S | tentative | 0.0 | 7.2 | 576 | 11.0 | -0.164 |
| S | subquery | 0.0 | 1.4 | 572 | 19.2 | -0.167 |
| M | answer | 0.0 | 0.4 | 559 | 5.3 | -0.053 |
| M | closed | 0.0 | 0.0 | 76 | 4.8 | -0.346 |
| M | tentative | 0.0 | 1.4 | 576 | 10.4 | -0.043 |
| M | subquery | 0.0 | 0.6 | 572 | 18.4 | -0.105 |
| L | answer | 0.0 | 0.2 | 559 | 4.9 | -0.053 |
| L | closed | 0.0 | 0.4 | 76 | 4.8 | -0.275 |
| L | tentative | 2.0 | 1.6 | 576 | 10.1 | -0.035 |
| L | subquery | 0.0 | 0.2 | 572 | 13.3 | -0.105 |

Tentative calls that used the <answer> tag: L 98.6%, M 100.0%, S 34.6%.

## Determinism

200 answer prompts generated twice in the same process (second pass after all other calls, so the prefix cache is warm and batches differ).

| Tier | Identical text | Identical scored answer |
|---|---|---|
| S | 99.5% of 200 | 99.5% |
| M | 100.0% of 200 | 100.0% |
| L | 100.0% of 200 | 100.0% |

- Tier S, `nq-train_74405|answer`: 'Carli Lloyd' then 'Geoff Hurst'.

Empty tentative answers: L 10; 100.0% of them are the model opening <answer> and closing it at once (an abstention, not a parsing failure).

## Sub-queries

Share of sub-queries that only repeat the question (equal after SQuAD normalisation): L 3.0%, M 18.4%, S 21.2%.

Tier S, first five HotpotQA questions:

- Where does the team Rick Bennett coaches play? → `Where does the team Rick Bennett coaches play in the American ice hockey league?`
- The 601 is equipped with which American all-weather, carrier-capable tactical airborne early warning (AEW) aircraft? → `Which American all-weather, carrier-capable tactical airborne early warning (AEW) aircraft is the 601 equipped with?`
- Which episode of this season of "The Simpsons"' that had two hold-over episodes from the previous season was written by Ken Keeler, Dan Greaney, and David S. Cohen? → `Which episode of season 5 of "The Simpsons" had two hold-over episodes from the previous season and was written by Ken Keeler, Dan Greaney, and David S. Cohen?`
- The hotel and casino in Winchester, Nevada where the 37th Daytime Emmy Awards were held, has how many hotel rooms? → `What is the number of hotel rooms at the Winchester, Nevada hotel and casino where the 37th Daytime Emmy Awards were held?`
- Birkelandodden is the northern point of Akseloya, that was named for a Norwegian scientist who was nominated for the Nobel prize how many times? → `How many times was Kristian Birkeland nominated for the Nobel Prize?`

Tier M, first five HotpotQA questions:

- Where does the team Rick Bennett coaches play? → `Where does the Union Dutchmen ice hockey team play?`
- The 601 is equipped with which American all-weather, carrier-capable tactical airborne early warning (AEW) aircraft? → `What is the Northrop Grumman E-2 Hawkeye and how many versions have been produced?`
- Which episode of this season of "The Simpsons"' that had two hold-over episodes from the previous season was written by Ken Keeler, Dan Greaney, and David S. Cohen? → `Which episode of season 5 of The Simpsons was written by Ken Keeler, Dan Greaney, and David S. Cohen?`
- The hotel and casino in Winchester, Nevada where the 37th Daytime Emmy Awards were held, has how many hotel rooms? → `How many hotel rooms does the Westgate Las Vegas Resort & Casino have?`
- Birkelandodden is the northern point of Akseloya, that was named for a Norwegian scientist who was nominated for the Nobel prize how many times? → `How many times was Kristian Birkeland nominated for the Nobel Prize?`

Tier L, first five HotpotQA questions:

- Where does the team Rick Bennett coaches play? → `Union College hockey team league`
- The 601 is equipped with which American all-weather, carrier-capable tactical airborne early warning (AEW) aircraft? → `Northrop Grumman E-2 Hawkeye 601 aircraft`
- Which episode of this season of "The Simpsons"' that had two hold-over episodes from the previous season was written by Ken Keeler, Dan Greaney, and David S. Cohen? → `"The Simpsons" season 5 episodes written by Ken Keeler Dan Greaney David S Cohen`
- The hotel and casino in Winchester, Nevada where the 37th Daytime Emmy Awards were held, has how many hotel rooms? → `Number of hotel rooms at Westgate Las Vegas Resort & Casino`
- Birkelandodden is the northern point of Akseloya, that was named for a Norwegian scientist who was nominated for the Nobel prize how many times? → `Kristian_Birkeland_nomination_for_Nobel_Prize`
