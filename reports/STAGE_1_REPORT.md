# Stage 1 report (lite profile)

Seed 42. 9,125 questions in 10 splits; 9,125 unique ids.

| Split | Dataset | Questions | Mean tokens |
|---|---|---|---|
| live_ft | hotpotqa | 250 | 25.0 |
| live_ft | nq | 250 | 10.6 |
| test_2wikimultihopqa | 2wikimultihopqa | 500 | 17.0 |
| test_bamboogle | bamboogle | 125 | 14.0 |
| test_hotpotqa | hotpotqa | 500 | 20.7 |
| test_musique | musique | 500 | 21.2 |
| test_nq | nq | 500 | 10.5 |
| test_popqa | popqa | 500 | 8.6 |
| test_triviaqa | triviaqa | 500 | 19.3 |
| train_pool | hotpotqa | 2,500 | 24.0 |
| train_pool | nq | 2,500 | 10.4 |
| val | hotpotqa | 250 | 23.7 |
| val | nq | 250 | 10.5 |

Full test-split sizes in the downloaded files: nq 3,610, hotpotqa 7,405, triviaqa 11,313, popqa 14,267, 2wikimultihopqa 12,576, musique 2,417, bamboogle 125.

MuSiQue hops in the test sample: 2-hop 274, 3-hop 156, 4-hop 70.
