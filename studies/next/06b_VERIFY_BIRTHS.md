# 06b. Independent check of 06_CRITIC Findings 0.1 to 0.3: D1 births against random regroupings

2026-11-05. Recomputed from the banked files with NumPy on the laptop. The reviewer's scripts were not read.
Of `06_CRITIC.md`, only lines 55 to 160 were read.

**Answer.** The finding holds in part. Claims (b) and (c) reproduce. 150 of the 1,238 late-born jobs have
p < 0.05 against random same-size groups (review: 148). Under random regrouping b is 0.530 [0.497, 0.563]
(review: 0.530 [0.491, 0.568]), and the real b = 0.476 is below all 300 draws. Claim (a) reproduces as a
number but not as a conclusion. The total-variation distance is 0.0406 (review: 0.040). But one random draw
differs from the other 299 by 0.0158 on average and by at most 0.0280. The real histogram is therefore
outside every random draw (p = 0.0033). It is close to the random-group histogram, not the same as it.

Abbreviations used below:
- `BANK` = `/Users/abaybektursun/projects/interp_research_banks/neurons`
- `OUT` = `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/verify_births/out`
- `SCR` = `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/verify_births`

## Summary against the review

| quantity | review (06_CRITIC.md) | this check | file |
|---|---|---|---|
| birth mismatches against `q2_clusters.jsonl` | 0 of 2,454 | 0 of 2,454 (birth_pct, birth_raw, main, late flags, status, B4 match) | `OUT/s01_recompute.txt` |
| recomputed b | 0.476 | 0.475610 on 574 poles, bootstrap [0.4355, 0.5174] | `OUT/s01_recompute.txt` |
| TV distance, real against random groups (16 points) | 0.040 | 0.0406 (set A), 0.0404 (set B) | `OUT/s03_histogram_tv.txt` |
| TV, one random draw against the other 299 | not reported | mean 0.0158, 95th pct 0.0225, max 0.0280 | `OUT/s03_histogram_tv.txt` |
| late-born before B3 and B4, real against random | 1,535 against 1,282 [1,238, 1,329] | 1,535 against 1,282.1 [1,239, 1,326] | `OUT/s03_histogram_tv.txt` |
| non-main births at T_14 and T_15 | 268 against 205 | 268 against 204.4 [191, 219] | `OUT/s03_histogram_tv.txt` |
| late-born jobs with p < 0.05 | 148 of 1,238 (0.120) | 150 of 1,238 (0.121); 152 against an independent set of 300 draws | `OUT/s04_per_cluster_p.txt` |
| count expected by chance | "at most 0.05, about 62" | 28.3 with the same selection as the real jobs (B3, B4); 20.0 with the percentile reading only | `OUT/s04_per_cluster_p.txt` |
| share born later than the median random group | 0.519 | 0.532; chance level with the same selection 0.432 [0.386, 0.473] | `OUT/s04_per_cluster_p.txt` |
| b under random regrouping | 0.530 [0.491, 0.568] | 0.530 [0.497, 0.563] (set A), 0.529 [0.495, 0.566] (set B) | `OUT/s05_b_random.txt` |
| registered permutation reference for b | mean 0.414, 97.5th 0.449 | mean 0.4141, 97.5th 0.4495 (repo `birth_order_test`, 10,000 permutations, seed 0), identical to `BANK/w5/verdict/metrics.json` | `OUT/s05_b_random.txt` |

## 1. The rule, as read from the code

From `infra/pg/neurons_verdict.py`. The pole-direction activation is
`signed_magnitude(a, s) = np.maximum(s * a, 0.0)`, with s = +1 on the + pole and -1 on the - pole. `cluster_table`
takes, for each sample-0 cluster, `med = np.median(mag[:, ranks], axis=1)` over its T_15 top-50 ranks at each of
the 16 points. `presence` returns `alive & (med >= q999)` (the percentile reading) and
`alive & (med >= 0.3 * max_abs)` (the raw reading), where `alive = med > 0`. `q999` and `max_abs` are the pole's
own 99.9th percentile and maximum of a+ at that point. `birth` returns the first t with
`p[t] and (t == len(p) - 1 or p[t + 1])`, so a cluster present only at T_15 is born at T_15. Echo clusters take
no part. `main` is `max(jobs, key=lambda r: (r["peak_end"], -r["c"]))`, where `peak_end = sub[-1].max()` is the
cluster's largest a+ at T_15. `late_flags` marks a cluster late if its birth is greater than the smallest birth
among the pole's non-echo clusters. The status `late_primary` (the 1,238 jobs) requires all of: late by the
percentile reading; birth not T_1; late by the raw reading too (B3); and `matched`, meaning one sample-1 cluster
holds more than half of its exemplars (the code's B4 exclusion). `birth_order_test` computes b over poles with
at least two born non-echo clusters and a born main. b is the share of those poles where
`b[mi] == b.min()`, so ties count as earliest. Its registered reference replaces main's birth with a uniform
draw from the pole's births. The judge flag is `flagged = sum(tests[key]["random"]) <= 1`: at most one of the
four random-decoy judgments is correct.

Facts about the data that the later steps rely on:
- **Every cluster is born.** All 2,454 non-echo clusters are present at T_15, because they are T_15 top-50
  positions. `never_born` is 0 (`OUT/s01_recompute.txt`), and the same holds for every random group (asserted
  in `SCR/s02_random_groups.py`). The 16-point histogram is therefore complete.
- **The clusters partition the top 50.** In every sample of `BANK/w4/o7_s_v2/explain/clusters.jsonl`
  (1,200 of 1,200), they cover ranks 0 to 49 exactly once. So "the non-echo top-50 positions" are the 50
  ranks minus the echo cluster's ranks.
- **The banked inputs line up.** Exemplar row i at T_15 equals line i of T_15 `samples/contexts.jsonl` (topk `a`,
  maximum difference 0), and the pole order is the same at all 16 points (`OUT/s00_load.txt`).

## 2. Step 1: recomputed births against the banked births

Two implementations were run on the same inputs: one written here, and the repo's own `cluster_table`. They
were compared with all 2,454 non-echo rows of `BANK/w5/verdict/samples/q2_clusters.jsonl`
(`OUT/s01_recompute.txt`).
- **Mismatches.** There were 0 for birth_pct, birth_raw, is_main, late_pct, late_raw, status and matched.
  The maximum difference in `med` was 0.0.
- **Histogram and status counts.** The recomputed birth histogram (main and non-main) equals
  `metrics.json` `q2.births`. The status counts equal `q2.status_counts`: 919 earliest, 244
  readings_disagree, 53 unmatched_in_sample_1, 1,238 late_primary.
- **b.** It is 0.475610 on 574 poles from both implementations, with bootstrap [0.4355, 0.5174]. This equals
  `q2.birth_order`.
- **Broken-input check.** The comparison does report errors when the rule is wrong. Dropping the t+1
  requirement gives 155 birth mismatches. Reading q999 one point late gives 1,469.
- **Identity check of the random-group code.** Feeding the real grouping through the random-group code in
  place of a shuffle gives 0 mismatches on every field (`OUT/s02_identity.txt`).

## 3. Step 2: birth histogram, real clusters against random groups

**Method.** For each pole, the non-echo positions were shuffled and cut into groups of the real non-echo
cluster sizes, in the real clusters' order. Each group got the rule of section 1. There are 300 draws per pole.
Set A (seed 20261105) is the reference. Set B (seed 20261106) is an independent second set, used for the noise
floors.

**Table.** Counts of 2,454 clusters. Random columns are means over the 300 draws of set A
(`OUT/s03_histogram_tv.txt`).

| point | step | real | real share | random | random share | main real | main random | non-main real | non-main random |
|---|---|---|---|---|---|---|---|---|---|
| T_0 | 0 | 0 | 0.000 | 0.0 | 0.000 | 0 | 0.0 | 0 | 0.0 |
| T_1 | 1,000 | 1 | 0.000 | 0.1 | 0.000 | 1 | 0.0 | 0 | 0.1 |
| T_2 | 2,000 | 3 | 0.001 | 4.2 | 0.002 | 1 | 0.5 | 2 | 3.7 |
| T_3 | 3,000 | 14 | 0.006 | 7.9 | 0.003 | 5 | 1.8 | 9 | 6.1 |
| T_4 | 5,000 | 30 | 0.012 | 24.4 | 0.010 | 8 | 5.7 | 22 | 18.7 |
| T_5 | 8,000 | 49 | 0.020 | 48.0 | 0.020 | 19 | 15.0 | 30 | 33.0 |
| T_6 | 12,000 | 62 | 0.025 | 55.0 | 0.022 | 17 | 15.8 | 45 | 39.2 |
| T_7 | 20,000 | 97 | 0.040 | 108.3 | 0.044 | 34 | 31.7 | 63 | 76.6 |
| T_8 | 35,000 | 160 | 0.065 | 158.7 | 0.065 | 42 | 45.4 | 118 | 113.3 |
| T_9 | 60,000 | 176 | 0.072 | 189.7 | 0.077 | 50 | 51.6 | 126 | 138.2 |
| T_10 | 100,000 | 238 | 0.097 | 263.8 | 0.107 | 74 | 72.0 | 164 | 191.8 |
| T_11 | 180,000 | 346 | 0.141 | 389.9 | 0.159 | 89 | 94.8 | 257 | 295.1 |
| T_12 | 320,000 | 467 | 0.190 | 457.5 | 0.186 | 97 | 104.7 | 370 | 352.9 |
| T_13 | 560,000 | 489 | 0.199 | 492.7 | 0.201 | 109 | 111.8 | 380 | 380.9 |
| T_14 | 1,000,000 | 286 | 0.117 | 241.8 | 0.099 | 47 | 47.0 | 239 | 194.8 |
| T_15 | 1,413,814 | 36 | 0.015 | 12.0 | 0.005 | 7 | 2.4 | 29 | 9.6 |

The shares agree with the review's table (T_4 to T_15) to within 0.001 at every point it lists.

**Total-variation distance** (TV, in units of share of clusters; `OUT/s03_histogram_tv.txt`):

| comparison | TV |
|---|---|
| real against set A pooled (300 draws) | 0.0406 |
| real against set B pooled | 0.0404 |
| one draw of A against the other 299 of A (noise floor for the line above) | mean 0.0158, median 0.0158, 95th pct 0.0225, max 0.0280 |
| one draw of A against one draw of B (two single draws) | mean 0.0220, 95th pct 0.0322 |
| real against one single draw of A | mean 0.0442, min 0.0310 |
| set A pooled against set B pooled | 0.0010 |

**Reading.**
- **The real histogram differs from random beyond draw-to-draw noise.** TV(real, pooled) = 0.0406 exceeds
  all 300 leave-one-out values. That gives p = 1/301 = 0.0033, the smallest p 300 draws can give.
- **The shape is close.** A TV of 0.041 means 4.1% of the birth mass, about 100 of 2,454 clusters, would
  have to move to match.
- **Where the difference lies.** Real clusters have extra births at T_14 (286 against 241.8) and T_15 (36
  against 12.0), and fewer at T_10 to T_11 (584 against 653.7).
- **The late-born counts point the same way.** By the percentile reading, 1,535 real clusters are late-born
  against 1,282.1 [1,239, 1,326] random groups.
- **The gap widens after the filters.** With B3 added (and T_1 excluded) it is 1,291 against 939.4
  [887, 988]. With B4 added (late_primary) it is 1,238 against 471.6 [437, 505]. B4 requires a
  majority match to a sample-1 cluster, and random groups rarely meet that. The late_primary gap therefore
  measures how well the two clusterer samples agree, not when the clusters were born.

## 4. Step 3: per-job p against same-size random groups from the same pole

**Method.** In each draw, the group in a job's slot is a uniform random subset of that job's size from its
pole's non-echo positions. The one-sided p is `(1 + #{draws with random birth >= real birth}) / 301`. With 300
draws, p < 0.05 is the same as "born later than more than 95% of the random groups". Ties count against the job.

**Chance level.** The jobs were selected for being late, so 0.05 × n is not the chance level. The
selection-matched chance level was computed by taking each draw d of set A in turn as a pseudo-real
grouping, applying the same selection to its groups, and scoring each selected group against the other 299
draws.

All numbers are from `OUT/s04_per_cluster_p.txt`:

| set | n | p < 0.05 | expected at chance |
|---|---|---|---|
| late_primary jobs (B3, B4) | 1,238 | 150 (0.121) | 28.3 (pass share 0.0229 among 471.6 selected per draw; set B as pseudo-real: 30.8) |
| same, against set B | 1,238 | 152 | |
| same, against sets A and B together (600 draws) | 1,238 | 152 | |
| all late-born by the percentile reading | 1,535 | 163 | 0.0161 pass share × 1,535 = 24.8 (set A leave-one-out) |
| review's chance level | 1,238 | 148 | 61.9 (0.05 × n) |

Tie sensitivity: with mid-p (ties count half), 210 jobs pass, against a chance level of 66.7 under the same
selection.

Passing jobs by birth point, as [late_primary jobs born there, passing]:

| point | T_4 | T_5 | T_6 | T_7 | T_8 | T_9 | T_10 | T_11 | T_12 | T_13 | T_14 | T_15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [n, pass] | [2, 0] | [10, 0] | [14, 1] | [27, 0] | [67, 7] | [76, 3] | [114, 9] | [164, 19] | [247, 30] | [271, 27] | [212, 33] | [34, 21] |

**Reading.**
- **The count reproduces.** 150 against the review's 148.
- **The chance level is lower than the review states.** It is 28, not 62, because births fall on 16
  discrete points and ties count against the job. The excess is therefore about 122 jobs, 5.3 times chance.
- **The failing jobs are not all dated where a random group would be.** All late_primary jobs together are
  later than their random median in 0.532 of cases, against a chance level of 0.432 [0.386, 0.473]. The
  1,088 failing jobs alone are later in 0.468 of cases, against 0.418 [0.374, 0.460]. A failed per-job test is
  low power, not evidence that a job has no timing of its own.

**Banked examples** (`OUT/s07_examples.txt`):
- **Passing.** `L00N2968-:c1` "identifier subword fragments (Hexadecimal, nummer, trie, logo)", size 17,
  born T_14. Its 300 random groups are born at T_13 (298) and T_14 (2), so p = 0.0100. Judge random-decoy
  accuracy 1.0, not flagged.
- **Failing.** `L00N181+:c0` "heart symbol in usernames" (this pole's main), size 7, born T_14. Its random
  groups are born at T_13 (182) and T_14 (118), so p = 0.3953. Judge accuracy 1.0, not flagged.

## 5. Step 4: b under random regrouping, beside the registered reference

All numbers are from `OUT/s05_b_random.txt` and `OUT/s05b_b_main_slot_fixed.txt`. The real b is 0.4756 on
574 poles.

| reference | mean | 2.5th pct | 97.5th pct | draws at or below 0.4756 |
|---|---|---|---|---|
| registered: births permuted within pole (10,000, seed 0) | 0.4141 | not reported | 0.4495 | permutation p 0.0005, real above |
| random regrouping, set A (300) | 0.5303 | 0.4973 | 0.5627 | 0 of 300 |
| random regrouping, set B (300) | 0.5291 | 0.4948 | 0.5662 | 3 of 300 |
| extra: main's size and its top position fixed, rest shuffled (300) | 0.5306 | 0.4930 | 0.5679 | 1 of 300 |
| extra: main's slot size fixed, top position not forced (300) | 0.5120 | 0.4739 | 0.5462 | 10 of 300 |

Over sets A and B, the one-sided p that the real b is at or below random is 4/601 = 0.0067.

**What each reference controls for.**
- **The registered permutation reference** holds each pole's set of births fixed and asks whether main's
  birth is a typical draw from it. It treats main as an arbitrary cluster. It does not account for how
  main is chosen: main holds the pole's largest T_15 activation (574 of 574 poles), and it is usually large
  (in 0.784 of poles it is at or above its pole's median cluster size).
- **The random-regrouping reference** holds the positions, their activations, the cluster sizes and the
  main-selection rule fixed. It randomizes only which positions form a job. It asks whether real job
  membership makes main earliest beyond what the selection rule gives.

**Reading.**
- **The review's claim holds.** Against random regrouping, the real b falls below the 2.5th percentile in
  both sets.
- **Most of the gap is main's size.** The two extra variants split the gap between the permutation mean
  (0.414) and the random-regrouping mean (0.530). Holding only main's slot size fixed gives 0.512. Also
  giving that slot the top position adds 0.019, to 0.531.
- **No reference that holds main's size fixed has the real main earliest more often than a random group.**
  Against the size-only variant the real b is at its lower edge (10 of 300 draws at or below it).
- **The review's stated cause is not what the data show.** The review says main "has the highest median". At
  T_15 that is true in 259 of 574 poles (0.451; `OUT/s08_main_median.txt`). The selection effect is real, but
  that sentence does not describe what produces it.

## 6. Step 5: the judge's flag on passing and failing jobs

The judge flag (`flagged` in `q2_clusters.jsonl`, at most one of four random-decoy judgments correct) is
present for all 1,238 late_primary jobs. 125 are flagged, the same as `metrics.json` `q2.flags.n_flagged`.
`a_ctrl` = 0.9242 is from `metrics.json` `q2.p_u`. All numbers are from `OUT/s06_judge_by_pass.txt`.

| group | n | flagged | flag rate (Wilson 95%) | mean random-decoy accuracy | p_u point | mean hard-decoy accuracy |
|---|---|---|---|---|---|---|
| p < 0.05 (pass) | 150 | 20 | 0.133 [0.088, 0.197] | 0.7817 | 0.336 | 0.6667 |
| p >= 0.05 (fail) | 1,088 | 105 | 0.097 [0.080, 0.115] | 0.8157 | 0.256 | 0.7013 |

The two groups do not differ detectably: Fisher exact on the flag gives odds ratio 1.44, p = 0.19, and
Mann-Whitney on the random-decoy accuracy gives p = 0.18. The direction is that passing jobs are flagged
slightly more often. The test has little room: the passing group's flag-rate interval spans 0.088 to 0.197.
The jobs that are dated beyond chance are therefore not more often judged related to their pole's earlier
record.

## 7. Where my reading of the rule differs from the review's description

1. **Presence.** The code also requires `med > 0`. The review omits this. It changes no cluster here.
2. **Birth at T_15.** The review's sentence "present at t and at t+1" leaves out the boundary case. At T_15
   there is no next point, so presence alone gives birth = T_15. 36 real clusters are born there. The
   review's histogram includes T_15 births, so its scripts apply the case.
3. **"Late-born jobs" (the 1,238).** These have status `late_primary`. That requires late by the percentile
   reading, birth not T_1, late by the raw reading (B3), and a majority match to a sample-1 cluster (B4).
   Two consequences:
   - **B4 differs between spec and code.** In the spec, B4 is the pole-level check `|m - m'| <= 1`. The code's
     B4 exclusion is the per-cluster majority match. The 1,238 use the per-cluster match.
   - **The review's chance level skips this selection.** It applies 0.05 × n to jobs that were selected for
     being late. With the selection applied, the chance level is 28, not 62 (section 4).
4. **Main.** The spec text says "largest max|a| at T_15". The code uses the largest a+ in the pole's direction
   (`peak_end`), and breaks ties by the lowest cluster index. The review's wording ("holds the pole's largest
   T_15 activation") matches the code. Its added sentence that main "has the highest median" holds in only
   0.451 of poles (section 5).
5. **b.** It counts ties as earliest. It is taken over poles with at least two born non-echo clusters and a born
   main. Because every cluster is born, those are the 574 poles with m >= 2. The review's description agrees.
6. **The total-variation distance.** The review reports it without a noise floor. The number reproduces
   (0.0406). Against draw-to-draw noise (max 0.0280 over 300 draws) it shows that the real and random
   histograms differ. The review's word "reproduces" leaves that out.

## 8. What this means for the claims (within this check's scope)

- **b against "main is oldest".** The closed reading, b = 0.476 above the permutation 97.5th percentile
  0.449, does not survive a reference that holds the main-selection rule fixed. Against random regrouping
  the real b is below the 2.5th percentile (0.497 in set A, 0.495 in set B). The review's call to retag
  this claim is supported.
- **Birth histogram as a count of job-formation events.** The real histogram is within TV 0.041 of what
  random groups of the same sizes give. So random groups reproduce most of its shape. The rest (excess
  births at T_14 and T_15) is beyond chance.
- **Per-job dating.** About 150 late-born jobs (150 to 152 depending on the reference set) are born later than
  95% of their random same-size groups, against about 28 at chance. The chance level is lower than the review
  states, so the evidence for job-specific timing is stronger than the review states.
- **Judge flag.** Being dated beyond chance does not predict the judge's flag (p = 0.19).

Not checked here: Finding 0.3's eta-squared analysis, the top-50 turnover (`top50_out.txt`), Finding 0.4, and the
review's decorrelation account of why the regrouping matches.

## 9. Scripts and outputs

Each script runs as `python3 -I <script>` from `SCR`, in this order. Each reads only `BANK` and the outputs of
the earlier scripts.

| script | what it does | outputs |
|---|---|---|
| `SCR/s00_load.py` | reads the 16 `exemplar_a.npz`, the 16 `contexts.jsonl`, `clusters.jsonl` and `q2_clusters.jsonl`; checks row order | `OUT/inputs.npz`, `OUT/clusters.json`, `OUT/q2_banked.json`, `OUT/s00_load.txt` |
| `SCR/s01_recompute.py` | step 1, with an independent implementation and the repo's `cluster_table` / `birth_order_test`, plus the broken-input check | `OUT/s01_recompute.json`, `OUT/s01_recompute.txt` |
| `SCR/s02_random_groups.py` | identity check; random regroupings, sets A and B, 300 draws each | `OUT/random_groups.npz`, `OUT/s02_identity.txt` |
| `SCR/s03_histogram_tv.py` | step 2: histogram, TV, noise floors, late-born counts | `OUT/s03_histogram_tv.json`, `OUT/s03_histogram_tv.txt` |
| `SCR/s04_per_cluster_p.py` | step 3: per-job p and the selection-matched chance level | `OUT/s04_per_cluster_p.json` (p for every cluster), `OUT/s04_per_cluster_p.txt` |
| `SCR/s05_b_random.py` | step 4: b under random regrouping and the registered permutation reference | `OUT/s05_b_random.json`, `OUT/s05_b_random.txt` |
| `SCR/s05b_b_main_slot_fixed.py` | step 4, the two extra variants (main's size fixed, with and without the top position) | `OUT/s05b_b_main_slot_fixed.json`, `OUT/s05b_b_main_slot_fixed.txt` |
| `SCR/s06_judge_by_pass.py` | step 5: the judge flag on passing and failing jobs | `OUT/s06_judge_by_pass.json`, `OUT/s06_judge_by_pass.txt` |
| `SCR/s07_examples.py` | one passing and one failing banked example | `OUT/s07_examples.txt` |
| `SCR/s08_main_median.py` | the check of the review's "highest median" sentence | `OUT/s08_main_median.txt` |

The inputs, all read-only, are:
- `BANK/w3/grid/1/out/t00_step0` to `t06_step12000` and `t15_step1413814`, plus
  `BANK/w3/grid/1b/out/t07_step20000` to `t14_step1000000`: `poles/exemplar_a.npz` and `samples/contexts.jsonl` in each.
- `BANK/w4/o7_s_v2/explain/clusters.jsonl`
- `BANK/w5/verdict/samples/q2_clusters.jsonl`
- `BANK/w5/verdict/metrics.json`
- `/Users/abaybektursun/projects/interp_research/infra/pg/neurons_verdict.py`, imported for the cross-check only.

The scratchpad is session-specific. Copy `SCR` into the repo before it is cleared if these outputs are to be
cited later.
