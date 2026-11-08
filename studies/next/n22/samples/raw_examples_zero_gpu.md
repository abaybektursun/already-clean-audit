# N22 raw examples, zero-GPU part

The first ten freeze-half items of `studies/next/n22/items.json` (sha256 in `n22_score.ITEMS_SHA`), as the model sees them. Every model output is PENDING: no bank holds a forward of these prompts. The L1 job writes them first, to `samples/raw_examples_t15.jsonl`, before any statistic.

The prompt (BOS, then three solved lines, then the query), item 0:

```
<|endoftext|>(3 + 5) * 4 + 2 = 34
(12 + 7) * 3 + 9 = 66
(25 + 16) * 2 + 8 = 90
(2 + 20) * 4 + 62 = |
```

| item | query | s1 | s2 | y (token id) | v | f1 = v c + d | f2 = v + d | partner's f1, f2 | slot-1 counterfactual | slot-2 counterfactual |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `(2 + 20) * 4 + 62 =` | 22 | 88 | 150 (3965) | 81 | 386 | 143 | 388, 145 | `(8 + 73) * 4 + 62 =` | `(19 + 8) * 3 + 62 =` |
| 2 | `(12 + 1) * 4 + 64 =` | 13 | 52 | 116 (8027) | 86 | 408 | 150 | 776, 174 | `(27 + 59) * 4 + 64 =` | `(11 + 32) * 2 + 64 =` |
| 4 | `(2 + 12) * 8 + 88 =` | 14 | 112 | 200 (1049) | 76 | 696 | 164 | 655, 123 | `(30 + 46) * 8 + 88 =` | `(2 + 17) * 4 + 88 =` |
| 6 | `(10 + 39) * 8 + 47 =` | 49 | 392 | 439 (20963) | 95 | 807 | 142 | 598, 123 | `(58 + 37) * 8 + 47 =` | `(15 + 4) * 5 + 47 =` |
| 8 | `(27 + 9) * 6 + 28 =` | 36 | 216 | 244 (13719) | 51 | 334 | 79 | 301, 148 | `(36 + 15) * 6 + 28 =` | `(2 + 15) * 3 + 28 =` |
| 10 | `(40 + 18) * 4 + 97 =` | 58 | 232 | 329 (18196) | 86 | 441 | 183 | 495, 151 | `(19 + 67) * 4 + 97 =` | `(23 + 20) * 2 + 97 =` |
| 12 | `(3 + 11) * 5 + 65 =` | 14 | 70 | 135 (8878) | 80 | 465 | 145 | 401, 161 | `(60 + 20) * 5 + 65 =` | `(18 + 22) * 2 + 65 =` |
| 14 | `(25 + 16) * 4 + 81 =` | 41 | 164 | 245 (13078) | 63 | 333 | 144 | 548, 107 | `(52 + 11) * 4 + 81 =` | `(7 + 2) * 7 + 81 =` |
| 16 | `(34 + 13) * 8 + 44 =` | 47 | 376 | 420 (12819) | 75 | 644 | 119 | 440, 140 | `(46 + 29) * 8 + 44 =` | `(4 + 21) * 3 + 44 =` |
| 18 | `(2 + 12) * 5 + 65 =` | 14 | 70 | 135 (8878) | 52 | 325 | 117 | 504, 140 | `(44 + 8) * 5 + 65 =` | `(2 + 11) * 4 + 65 =` |

What they show at zero GPU: every number is one token; the query occupies positions 49-62 of a 63-token prompt (BOS at 0); the operands sit at 50, 53, 57, 60 and the answer is predicted at 62, the token `Ġ` after `Ġ=`. For every item, f1(v) and f2(v) differ from each other, from v and from every number in the item, so a raised target cannot be the injected token or a copied operand. The counterfactual of slot 1 keeps c and d and its answer is f1(v); that of slot 2 keeps d and its answer is f2(v).

PENDING, written by the job: the model's top-1 and probability, p(y), the lens top-10 at the bands, and the outputs after LENS_U, LENS_M, DOM_M, ANS_U, ANS_M and their random floors for v index 0.
