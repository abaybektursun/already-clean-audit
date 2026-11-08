# N22 raw examples at T_15 (L1 `l1-01`), freeze half

Filled from `banks/l1-01/n22/t15/{accuracy.json, clean_lp_numbers.npy, readout.npz}` by `samples/raw_examples_l1.py`. The job wrote no `raw_examples_t15.jsonl`: 2 freeze and 5 test items were answered, under the 50 per half the edits need, so the lens top-10 at the bands and the outputs after the edits were not produced. The first ten freeze items are all wrong; the first answered freeze item (714) is added. Ranks are in the full vocabulary of 100,278 (1 = top); blocks are source blocks 0–29.

| item | query | s1 | s2 | y | model top-1 (p) | p(y) | rank of y among 1,000 numbers | correct |
|---|---|---|---|---|---|---|---|---|
| 0 | `(2 + 20) * 4 + 62 =` | 22 | 88 | 150 | 124 (0.055) | 0.0144 | 23 | no |
| 2 | `(12 + 1) * 4 + 64 =` | 13 | 52 | 116 | 80 (0.1232) | 0.00437 | 34 | no |
| 4 | `(2 + 12) * 8 + 88 =` | 14 | 112 | 200 | 88 (0.0688) | 0.00981 | 27 | no |
| 6 | `(10 + 39) * 8 + 47 =` | 49 | 392 | 439 | 426 (0.0107) | 0.00246 | 133 | no |
| 8 | `(27 + 9) * 6 + 28 =` | 36 | 216 | 244 | 180 (0.098) | 0.002 | 77 | no |
| 10 | `(40 + 18) * 4 + 97 =` | 58 | 232 | 329 | 172 (0.0237) | 0.00059 | 279 | no |
| 12 | `(3 + 11) * 5 + 65 =` | 14 | 70 | 135 | 110 (0.0563) | 0.02015 | 13 | no |
| 14 | `(25 + 16) * 4 + 81 =` | 41 | 164 | 245 | 180 (0.1088) | 0.0008 | 161 | no |
| 16 | `(34 + 13) * 8 + 44 =` | 47 | 376 | 420 | 376 (0.0177) | 0.00669 | 47 | no |
| 18 | `(2 + 12) * 5 + 65 =` | 14 | 70 | 135 | 85 (0.0721) | 0.01343 | 20 | no |
| 714 | `(5 + 45) * 2 + 36 =` | 50 | 100 | 136 | 136 (0.0644) | 0.06438 | 1 | yes |

Best (smallest) rank of s1, s2 and y over blocks 0–29, with the block, under the full lens and the logit lens:

| item | full s1 | full s2 | full y | logit s1 | logit s2 | logit y |
|---|---|---|---|---|---|---|
| 0 | 29 (block 25) | 17 (block 25) | 43 (block 29) | 12 (block 21) | 8 (block 24) | 38 (block 29) |
| 2 | 192 (block 28) | 51 (block 26) | 49 (block 29) | 6 (block 19) | 33 (block 26) | 53 (block 29) |
| 4 | 7 (block 22) | 10 (block 29) | 63 (block 29) | 6 (block 22) | 13 (block 29) | 63 (block 29) |
| 6 | 270 (block 21) | 5 (block 29) | 147 (block 27) | 14 (block 21) | 23 (block 29) | 206 (block 29) |
| 8 | 35 (block 26) | 7 (block 26) | 96 (block 29) | 2 (block 21) | 4 (block 26) | 139 (block 29) |
| 10 | 11 (block 24) | 43 (block 29) | 321 (block 29) | 3 (block 21) | 38 (block 29) | 295 (block 29) |
| 12 | 234 (block 29) | 6 (block 27) | 28 (block 29) | 60 (block 22) | 1 (block 27) | 14 (block 29) |
| 14 | 75 (block 26) | 14 (block 29) | 116 (block 27) | 16 (block 22) | 12 (block 29) | 202 (block 28) |
| 16 | 737 (block 29) | 22 (block 28) | 36 (block 29) | 121 (block 20) | 4 (block 26) | 45 (block 29) |
| 18 | 147 (block 28) | 6 (block 27) | 20 (block 25) | 30 (block 22) | 5 (block 27) | 51 (block 29) |
| 714 | 86 (block 26) | 40 (block 29) | 1 (block 29) | 104 (block 26) | 33 (block 29) | 1 (block 27) |
