# 10 — Third recount of the already-clean birth dates and of b

Written 2026-11-05 by an Opus worker. CPU only, banked arrays, zero spend. Scripts: `studies/next/10_recount2/`
(`01_load.py` … `06_examples.py`, shared rule in `common.py`). Outputs: `studies/next/10_recount2/out/`; the two
files over 1 MB are in `/Users/abaybektursun/projects/interp_research_banks/neurons/next/recount2/` with pointers in
`out/arrays.pointer.json` and `out/births.pointer.json`. Every path below without a directory is under `out/`.

**Answer: no. The banked b does not exceed its chance reference. b = 0.476 [0.436, 0.516] (273 of 574 poles whose
main cluster is earliest-born, ties included; `04_b.json`). Random groups of the same pole's exemplars, with the
real clusters' sizes and with main's group holding the top exemplar, give b = 0.530 [0.495, 0.568] (`05_chance.json`
→ `summary.R1t`). The banked b lies below that interval, not inside it.** The paired gap, b minus the per-pole chance
probability, is −0.054 [−0.092, −0.017] over the same 574 poles (same file).

The banked dates themselves are the registered rule: 2,454 of 2,454 non-echo clusters get the same birth date, the
same raw-ratio date, the same main flag and the same per-checkpoint medians (largest difference 0.0) in my recount
(`03_compare.json`). The dates are right; the comparison set the study used for b was not a chance reference for
cluster size and rank.

## 1. What I read

- `CLAUDE.md`, `studies/next/README.md`, `studies/already-clean/SPEC_D1.md`, `studies/already-clean/README.md`,
  `studies/already-clean/archive/DATA_CONTRACT.md`, and `studies/already-clean/JOURNAL.md` (grep for the dating
  rule and b; the entry of 2026-10-21T17:36:22Z states the coded formulas).
- Banked data: `w3/grid/{1,1b}/out/tNN_stepN/poles/exemplar_a.npz` and `samples/contexts.jsonl` (16 points),
  `w4/o7_s_v2/explain/clusters.jsonl`, `w5/verdict/samples/q2_clusters.jsonl`, `w5/verdict/metrics.json`,
  `w3/o7/draw/README.md`. Input SHA-256 values are in `arrays.pointer.json`.
- Not opened before section 11: everything on the blindness list. No dating code was read.

## 2. The rule as registered (quotes from `SPEC_D1.md`)

Clusters, lines 274–278: "At `T_15`, Agent 1 (clustering role, a fourth prompt) groups the pole's top-50 contexts
from the 20M-token corpus into `m` clusters … Contexts in the double-negative regime (`g < 0, u < 0`) are the other
pole's echo … the clusterer puts them in one cluster flagged `echo`, which is reported but excluded from `m`, from the
parent test and from every count … `E_c` is the exemplar set of cluster `c` (its contexts' `(doc_id, position)`)."

Presence and birth, lines 280–286: "At checkpoint `t`, cluster `c` is present if the median exemplar in `E_c` has
`a⁺_t ≥ q_{99.9}(a⁺_t)`, where `a⁺_t = max(±a_t, 0)` is the activation in the pole's own direction … and the
percentile is the pole's own 99.9th percentile of that quantity at `t` (within-checkpoint normalization …).
`birth(c)` is the first `t` at which `c` is present and stays present at the next grid point; a cluster first present
at `T_15` has `birth = T_15` (no next point exists; it is late-born with `t_before = T_14`; amended 2026-10-19)."

Raw-ratio reading, lines 286–287: "median exemplar `median_{E_c} a⁺_t ≥ 0.3·max|a_t|`, with the same a⁺ = max(s·a, 0)
as the percentile reading and median > 0".

Main and late-born, lines 291–292: "`main` is the cluster with the largest `max|a|` at `T_15`. A cluster is
**late-born** if `birth(c) > min_c' birth(c')` for that pole."

Birth order, lines 308–309: "Per pole with `m ≥ 2`, whether `main` is the earliest-born cluster. [Comparison set:]
birth ranks permuted within the pole."

The quantity, lines 506–510: "`b` = fraction of poles with `m ≥ 2` whose `main` is earliest-born. [Comparison set:]
10,000 permutations of birth ranks within each pole. Supports the 'main job is the oldest' form of H_RG … if `b`
exceeds the 97.5th percentile of [that set]".

Interval, line 533: "`p_u`, `b`, `v`: bootstrap over poles, 10,000 resamples, seed logged."

Grid, lines 399–416: T_0 = step 0 (random init), T_1 1,000, T_2 2,000, T_3 3,000, T_4 5,000, T_5 8,000, T_6 12,000,
T_7 20,000, T_8 35,000, T_9 60,000, T_10 100,000, T_11 180,000, T_12 320,000, T_13 560,000, T_14 1,000,000,
T_15 1,413,814. Dates below are these indices.

Data contract: the percentile available per pole and checkpoint is `q999`, "upper edge of the bin holding that
quantile of |a| over the pole's tokens (resolution 5.9%)" (`DATA_CONTRACT.md` line 79–80); `exemplar_a.npz` holds
`a`, `g`, `u` [P, 50] at the T_15 top-50 positions of every pole, in T_15 rank order (lines 89–92, 194–202); echo
clusters are "excluded from m, main, births, the parent test and every count" (lines 172–173).

## 3. Ambiguities and the choice made

1. **Which 99.9th percentile.** The spec names the 99.9th percentile of a⁺_t. The only per-checkpoint percentile in
   the banks is `q999` in each checkpoint's `contexts.jsonl`: a histogram bin upper edge (10^(k/40) spacing; checked,
   `01_load_checks.json` → `q999_off_bin_edge_max` 1e-6) of |a| over the pole's own-sign tokens. A percentile of a⁺
   over all tokens would include the other sign's zeros and be lower. I used the banked `q999`; no other choice is
   computable from the banks.
2. **Median of an even-size set.** I used the mean of the two middle values (numpy median). The lower middle value
   changes 205 of 2,454 dates and the upper middle value 218 (`02_counts.json`); b becomes 0.476 and 0.453
   (`04_b.json` → `b_median_convention_sensitivity`). Only the mean-of-two convention matches the banked dates
   (2,249 and 2,236 matches for the others, `03_compare.json`).
3. **Two-point rule and T_15.** Birth = the first t ≤ T_14 with presence at t and t+1; if none, T_15 when present
   at T_15; else never born. Under a literal reading of "first present at T_15", 9 clusters dated T_15 that had an
   isolated earlier presence would be never born instead (`02_counts.json` →
   `sample0_birth15_with_earlier_isolated_presence`). No cluster is never born under my reading.
4. **Median > 0.** Not in the percentile sentence; the journal (2026-10-21T17:36:22Z) says the code requires it. It
   changes nothing: the smallest `q999` on the grid is 0.0376 (`01_load_checks.json`).
5. **Main.** The non-echo cluster with the largest |a| at T_15 over its exemplars, computed from the arrays. It holds
   the pole's top non-echo rank in all 574 poles (asserted in `05_chance.py`); in 598 of 600 poles that is rank 0
   (`02_counts.json` → `sample0_main_min_rank_hist`).
6. **Which clustering sample.** The files number samples 0 and 1; the B4 text calls the primary "sample 1". I used
   sample 0 as the primary (600 poles, 2,454 non-echo clusters, 11 echo). Sample 1 is dated too (2,025 non-echo
   clusters, `02_counts.json`) and not used for b.
7. **Ties in b.** Main counts as earliest-born when no cluster of the pole is born strictly before it, the complement
   of "late-born" at line 292. The strict reading (main alone at the earliest date) is reported beside it.
8. **Which clusters enter b.** All non-echo clusters of sample 0. The B3 and B4 filters (lines 500, and the B4 rule in
   `DATA_CONTRACT.md` line 185) are written for `p_u`, not for b.
9. **Row alignment.** Row i of `exemplar_a.npz` is line i of the same directory's `contexts.jsonl`. Checked: at T_15
   the exemplar a equals the top-50 a exactly; at T_1 to T_14 every T_15 position that is also in that checkpoint's
   own top-50 has the same a (2 to 12,822 overlaps, largest difference 0.0). At T_0 no position overlaps, so the row
   order there is assumed (`01_load_checks.json`).
10. **b interval.** Percentile bootstrap over poles, 10,000 resamples, seed 1105.

## 4. Recount and comparison with the banked dates

| quantity (file) | value |
|---|---|
| poles (sample 0), non-echo clusters, echo clusters (`02_counts.json`) | 600, 2,454, 11 |
| poles with m ≥ 2 (`02_counts.json`) | 574 |
| banked non-echo rows in `q2_clusters.jsonl` (`03_compare.json`) | 2,454, same keys |
| exemplar ranks equal | 2,454 of 2,454 |
| per-checkpoint median, largest absolute difference | 0.0 |
| presence vectors equal, percentile and raw-ratio | 2,454 and 2,454 |
| birth date equal, percentile reading | 2,454 of 2,454 |
| birth date equal, raw-ratio reading (6 banked never born by raw ratio, matched) | 2,454 of 2,454 |
| main flag equal | 2,454 of 2,454 |
| mismatches | none |

Percentile birth dates over the 2,454 clusters (`02_counts.json`): T_1 1, T_2 3, T_3 14, T_4 30, T_5 49, T_6 62,
T_7 97, T_8 160, T_9 176, T_10 238, T_11 346, T_12 467, T_13 489, T_14 286, T_15 36.

## 5. Failing inputs for the comparison

Each planted input goes through the same dating and the same comparison (`03_compare.json` → `failing_inputs`).
A comparison that always agreed would show 2,454 on every row.

| input | exact matches with the banked date (of 2,454) |
|---|---|
| true input, same routine | 2,454 |
| (a) the 16 checkpoints in a shuffled order (activations and thresholds moved together) | 0 |
| (b) pole rows shifted by two (each pole read as the next neuron's same-sign pole) | 298 |
| (c) each cluster's ranks replaced by a random same-size draw from its pole's non-echo ranks | 991 |
| (d) every checkpoint's threshold replaced by the T_15 threshold | 106 |
| (e) the median replaced by the maximum over the exemplar set | 550 |
| (f) the next-point requirement dropped (birth = first presence) | 2,299 |

Input (c) already says something about the dates: a random group of the cluster's size from the same pole gets the
identical date in 991 of 2,454 cases (40%).

## 6. b and the spec's permutation band

`04_b.json`: b = 0.4756 (273 of 574), percentile bootstrap interval [0.4355, 0.5157]. Strict reading (main alone
at the earliest date): 0.2195 [0.1864, 0.2544]; 147 of the 273 mains share the earliest date with another cluster.
The spec's permutation band, recomputed: mean 0.4143, 97.5th percentile 0.4495, one-sided p 0.0003. These reproduce
`w5/verdict/metrics.json` → `q2/birth_order` (b 0.4756, interval [0.4355, 0.5174], band mean 0.4141, 97.5th 0.4495);
the upper interval end differs by one pole (1/574) because the bootstrap seed differs.

## 7. The chance reference

Design, written into the docstring of `05_chance.py` before it ran. For each pole with m ≥ 2, the cluster labels of
its non-echo T_15 exemplar ranks are shuffled 2,000 times (seed 20261105); each random group keeps its real
cluster's size and is dated by the same rule from the same arrays and thresholds. Main of a random partition is
chosen by the same rule (largest |a| at T_15).

- **R1t, primary (size and top rank):** the pole's top non-echo rank keeps main's label, so main's replacement holds
  the top exemplar, which is what makes a cluster main.
- **R1 (size only):** all labels shuffled; main is whichever group receives the top rank.
- **R2 (size, top rank and rank composition):** as R1t, but labels move only within rank bins of five (0–4, …,
  45–49), so every group keeps its count of exemplars per bin.

Reference value = mean of the 2,000 replicate b's (each replicate re-dates one random partition of all 574 poles);
interval = their 2.5th and 97.5th percentiles. Paired gap = mean over poles of (real indicator − the pole's
probability under the reference), percentile bootstrap over poles, 10,000 resamples.

All values are fractions of the 574 poles (`05_chance.json` → `summary.<ref>`):

| quantity | real clusters | R1t | R1 | R2 |
|---|---|---|---|---|
| b, main earliest-born, ties included (the registered b) | 0.476 | 0.530 [0.495, 0.568] | 0.530 [0.495, 0.570] | 0.538 [0.505, 0.571] |
| share of replicates with b ≥ 0.476 | — | 0.9985 | 0.999 | 1.000 |
| paired gap, real − reference | — | −0.054 [−0.092, −0.017] | −0.054 [−0.092, −0.017] | −0.062 [−0.095, −0.029] |
| b strict, main alone at the earliest date (exploratory) | 0.220 | 0.123 [0.099, 0.150] | 0.121 [0.096, 0.146] | 0.157 [0.132, 0.181] |
| paired gap, strict | — | +0.096 [+0.065, +0.128] | +0.099 [+0.067, +0.131] | +0.063 [+0.035, +0.091] |
| share of the pole's other clusters born after main, ties as one half (exploratory, tie-neutral) | 0.560 | 0.519 [0.495, 0.544] | 0.518 [0.496, 0.539] | 0.543 [0.523, 0.563] |
| paired gap, tie-neutral | — | +0.041 [+0.015, +0.067] | +0.042 [+0.016, +0.068] | +0.017 [−0.005, +0.039] |
| calibration input: one extra random partition used as the clusters, b | — | 0.531 | 0.556 | 0.542 |

Why the permutation band sits lower than the chance reference. The band reassigns the real dates within a pole, so
it keeps the real clusters' ties. Random groups of one pole's exemplars tie much more often, because their medians
all follow the same pool: under R1t main shares the earliest date with another group in 0.530 − 0.123 = 0.407 of
poles; the real main does so in 147/574 = 0.256 (`04_b.json`, `05_chance.json`). The registered b counts a tie as
"earliest", so a chance process with the real sizes and the real main-selection rule already reaches 0.530.

The two exploratory rows split that result. With ties removed, the real main is alone at the earliest date more
often than chance (0.220 against 0.123). With ties as one half, the real main is born before 0.560 of its pole's
other clusters against 0.519 at chance (size and top rank); when rank composition is also matched (R2) the gap is
+0.017 [−0.005, +0.039] and does not exclude zero. Neither row is the registered quantity.

## 8. Per-cluster result

Threshold stated before the run (docstring of `05_chance.py`): one-sided 0.05 in each direction. For each cluster of
the 574 poles, p_late = P(a random group of its size, from its pole, is born at the same checkpoint or later) and
p_early = P(born at the same checkpoint or earlier), from that cluster's column of the R1t partitions. Never born
counts as latest. The count expected at chance is the sum over clusters of P(p ≤ 0.05) when the date itself is drawn
from the reference; it is below 0.05 × n because the dates are discrete and tie. The 26 clusters of the m = 1 poles
cannot differ from a random group of their own size and are not tested.

R1t (`05_chance.json` → `summary.R1t`; per cluster in `05_per_cluster_R1t.csv`):

| clusters | n | later than chance (p_late ≤ 0.05) | expected at chance | earlier than chance (p_early ≤ 0.05) | expected at chance | calibration input, later / earlier |
|---|---|---|---|---|---|---|
| all | 2,428 | 153 | 22.4 | 160 | 29.6 | 26 / 29 |
| main | 574 | 21 | 4.3 | 46 | 5.0 | 8 / 5 |
| non-main | 1,854 | 132 | 18.2 | 114 | 24.6 | 18 / 24 |

The per-cluster chance level is 22.4 / 2,428 = 0.0092 (later) and 29.6 / 2,428 = 0.0122 (earlier); the simple
0.05 × 2,428 = 121.4 would overstate it. Positive controls (non-main clusters replaced by the group of their size made
of the pole's latest-dated, or earliest-dated, single exemplars, top rank excluded): 1,172 of 1,854 later than chance
and 1,404 of 1,854 earlier than chance (`summary.R1t.ctrl_late`, `ctrl_early`). The test is calibrated (26 and 29
against 22.4 and 29.6) and has power.

Under R2 (rank composition matched): 77 later (15.6 expected) and 90 earlier (20.0 expected) of 2,428
(`summary.R2.all`). Under R1: 163 (22.7) and 168 (30.1).

How much of the date is set by size and rank (`06_predictability.json`): 2,115 of 2,428 clusters (0.871) are inside
the chance band on both sides; 1,102 (0.454) are born exactly at the median date of their random groups; Pearson
correlation between a cluster's date and its reference mean date 0.797 (Spearman 0.817); mean absolute difference
0.98 grid points, against a standard deviation of the real dates of 2.51 grid points.

So 313 of 2,428 clusters (0.129) are dated beyond chance in one direction or the other, against 52.0 expected at
chance (R1t): about one cluster in ten carries date information beyond its size and top rank, in roughly equal numbers
later (153) and earlier (160), and the mains among them lean early (46 earlier, 21 later). With rank composition also
matched (R2) the count is 167 against 35.6 expected, about one cluster in eighteen.

## 9. Three raw examples

Activations are a⁺_t = max(s·a_t, 0) in raw MLP activation units; q999 is the banked threshold in the same units.
All three from `06_examples.json`; positions are (document, position in document, corpus token index) at T_15. The
reference histograms are a fresh draw of 2,000 groups (seed 7), so their p differs slightly from the main run.

**(1) Dated later than chance.** `L10N9794+` cluster 3, "access / CRUD in data-rights and privacy text", non-main,
size 4 of a pole with m = 4 (other clusters: main c0 size 19 born T_9; c1 size 15 born T_6; c2 size 12 born T_10).
Exemplars: rank 24 " access" (doc 1318, pos 1447, tok 1495023; "…such as the right to"), rank 27 " Access" (doc 1822,
pos 425, tok 2214621), rank 35 " access" (doc 9012, pos 176, tok 12758136; "You can exercise your rights:"), rank 36
" CRUD" (doc 6327, pos 66, tok 9765520; a code comment "Contains all methods for performing").

| t | a⁺ of ranks 24, 27, 35, 36 | median | q999 | present |
|---|---|---|---|---|
| T_8 | 0, 0, 0, 16.40 | 0 | 18.84 | no |
| T_9 | 1.22, 0.237, 0.569, 10.71 | 0.895 | 4.217 | no |
| T_10 | 0.479, 0.388, 0.331, 3.456 | 0.434 | 1.334 | no |
| T_11 | 0.977, 0.375, 0.490, 4.261 | 0.733 | 0.841 | no |
| T_12 | 0.988, 0.708, 0.644, 3.459 | 0.848 | 0.668 | yes |
| T_13 | 1.384, 1.010, 0.657, 3.250 | 1.197 | 0.562 | yes |
| T_14 | 3.029, 1.757, 1.960, 2.552 | 2.256 | 0.398 | yes |
| T_15 | 1.660, 1.634, 1.524, 1.497 | 1.579 | 0.282 | yes |

Birth T_12 (step 320,000) in the recount and in the bank. Random size-4 groups of the pole: T_5 97, T_6 136, T_7 288,
T_8 357, T_9 913, T_10 202, T_11 6, T_12 1 of 2,000; p_late 0.0005 (main run 0 of 2,000).

**(2) At chance.** `L09N1264-` cluster 3, "Mid-sentence function/noun slot (you, fact, deal)", non-main, size 5 of a
pole with m = 6 (main c0 size 12 born T_13). Exemplars: rank 10 " fact" (doc 12919, pos 65289, tok 19369897),
rank 15 " you" (doc 11607, pos 3357, tok 16541246), rank 27 " you" (doc 5809, pos 2181, tok 7757602), rank 29 " deal"
(doc 5930, pos 32738, tok 8167198), rank 42 " you" (doc 5809, pos 1979, tok 7757400).

| t | a⁺ of ranks 10, 15, 27, 29, 42 | median | q999 | present |
|---|---|---|---|---|
| T_12 | 0, 0.093, 0.034, 0.089, 0.018 | 0.034 | 0.316 | no |
| T_13 | 0.120, 0.154, 0.133, 0.222, 0.118 | 0.133 | 0.282 | no |
| T_14 | 0.450, 0.392, 0.154, 0.329, 0.220 | 0.329 | 0.200 | yes |
| T_15 | 0.479, 0.455, 0.416, 0.414, 0.399 | 0.416 | 0.150 | yes |

At T_3 rank 10 has a⁺ 17.53 against q999 167.9 and the median is 0.150; no earlier checkpoint is close. Birth T_14
in the recount and in the bank. Random size-5 groups: T_12 236, T_13 734, T_14 922, T_15 108 of 2,000; p_late 0.515,
p_early 0.946 (main run 0.503 and 0.939).

**(3) No recount mismatch exists; the closest case is a date that moves with the median convention.** `L00N8952-`
cluster 5, "Downloads and ESL tokens", non-main, size 4 of a pole with m = 7. Exemplars: rank 14 " Downloads" (doc
12046, pos 298, tok 17017719), rank 15 " ESL" (doc 7877, pos 1897, tok 11351414), rank 16 " ESL" (doc 820, pos 438,
tok 936154), rank 22 " Downloads" (doc 4826, pos 397, tok 6595776).

| t | a⁺ of ranks 14, 15, 16, 22 | mean-of-two median | lower middle value | q999 | present (mean-of-two) |
|---|---|---|---|---|---|
| T_12 | 0.139, 0.145, 0.038, 0.063 | 0.101 | 0.063 | 0.211 | no |
| T_13 | 0.494, 0.148, 0.069, 0.342 | 0.245 | 0.148 | 0.200 | yes |
| T_14 | 0.330, 0.224, 0.191, 0.289 | 0.257 | 0.224 | 0.119 | yes |
| T_15 | 0.196, 0.194, 0.193, 0.185 | 0.194 | 0.193 | 0.056 | yes |

Birth T_13 under the mean of the two middle values (recount and bank agree), T_14 under the lower middle value.
Random size-4 groups: T_12 727, T_13 1,184, T_14 89 of 2,000; at chance either way.

## 10. Retag recommendation

The D1 claim should read: "Birth order does not support 'the main job is the oldest': the main cluster is earliest-born
(ties included) in 0.476 [0.436, 0.516] of 574 poles, which is above the within-pole permutation band (97.5th
percentile 0.449) but below the 0.530 [0.495, 0.568] given by random same-size groups of the same pole's exemplars
with main's group holding the top exemplar."

## 11. Comparison with the two earlier computations

Sections 1–10 were written to disk before `06b_VERIFY_BIRTHS.md` and `06_CRITIC.md` Part 0 (lines 55–214) were opened.
No number above was changed after reading them. The earlier computations count beyond-chance clusters over the
1,238 `late_primary` clusters or the 1,535 clusters late-born by the percentile reading; mine counted over all 2,428.
For a like-for-like row I wrote one script after reading, `10_recount2/07_like_for_like.py` (output
`07_like_for_like.json`): it re-slices the per-cluster test onto their subsets with the size-only design (R1, fresh
seed 7001) and changes nothing in sections 1–10.

| quantity | this recount | 06b verifier | 06 critic, Part 0 | agree? | reason for a disagreement |
|---|---|---|---|---|---|
| non-echo clusters, sample 0; poles with m ≥ 2 | 2,454; 574 (`02_counts.json`) | 2,454; 574 | 2,454 | agree | — |
| mismatches with `q2_clusters.jsonl` | 0 on birth_pct, birth_raw, main, medians, presence (`03_compare.json`) | 0, also on late flags, status, B4 match | 0 | agree | — |
| failing input: next-point requirement dropped | 155 mismatches (2,299 of 2,454 match) | 155 | not reported | agree | — |
| b, ties included | 0.4756 (273/574), [0.4355, 0.5157] (`04_b.json`) | 0.475610, [0.4355, 0.5174] | 0.476 [0.436, 0.517], the banked value | agree | upper end differs by one pole (1/574): bootstrap seed |
| spec's permutation band, mean and 97.5th percentile | 0.4143, 0.4495 | 0.4141, 0.4495 | 0.414, 0.449 | agree | — |
| chance reference for b, sizes fixed, main = group holding the top rank | R1: 0.530 [0.495, 0.570], 2,000 replicates (`05_chance.json`) | set A 0.5303 [0.4973, 0.5627]; set B 0.5291 [0.4948, 0.5662]; 300 draws each | 0.530 [0.491, 0.568] | agree | interval ends differ with the number of draws |
| chance reference, main's size and top rank fixed | R1t: 0.530 [0.495, 0.568] (primary) | 0.5306 [0.4930, 0.5679] | not computed | agree | — |
| does the banked b lie inside the reference? | no, below: 3 of 2,000 R1t replicates (2 of 2,000 R1) are below 0.476; paired gap −0.054 [−0.092, −0.017] | no, below: 0 of 300 (A) and 3 of 300 (B) at or below; p = 4/601 | no, below the 2.5th percentile | agree | — |
| clusters later than chance, `late_primary` (1,238) | 150 under R1, both p conventions (`07_like_for_like.json`); 142 under R1t | 150 (152 against set B) | 148 | agree | — |
| clusters later than chance, late-born by percentile (1,535) | 162 under R1 (`07_like_for_like.json`) | 163 | not reported | agree | draw noise |
| clusters later than chance, all 2,428 | 153 under R1t, 163 under R1 (`05_chance.json`); 162 under R1 with seed 7001 | not reported | not reported | — | every cluster dated later than chance is late-born, so this equals the 1,535 row |
| count expected at chance for that count | all 2,428, no selection: 22.4 (R1t), 22.7 (R1); late-born by percentile, selection-matched: 27.1 (`07_like_for_like.json`) | 28.3 for `late_primary` with its selection; 24.8 for late-born by percentile | "at most 0.05, about 62" | agree with the verifier; disagree with the critic | dates fall on 16 points and tie, so P(p ≤ 0.05) is far below 0.05; selecting late-born groups raises it to about 0.018. The verifier's 24.8 against my 27.1: leave-one-out over 300 draws against scoring on the full column of 2,000 |
| per-cluster chance level | 0.0092 later, 0.0122 earlier, all clusters, R1t; 0.0176 among late-born random groups, R1 | 0.0161 (late-born by percentile); 0.0229 (`late_primary` selection) | 0.05 | agree with the verifier; disagree with the critic | as above |
| clusters earlier than chance | 160 of 2,428, 29.6 expected; mains 46 of 574, 5.0 expected | not reported | not reported | — | — |
| b strict, and the tie-neutral share (exploratory) | strict 0.220 against 0.123 [0.099, 0.150]; tie-neutral 0.560 against 0.519 [0.495, 0.544] (R1t), 0.543 [0.523, 0.563] (R2) | not reported | not reported | — | — |
| why the reference sits above the permutation band | random groups share the earliest date with main in 0.407 of poles, real clusters in 0.256; requiring main's group to hold the top rank moves the reference by less than 0.001 (R1 0.5300, R1t 0.5300) | mostly main's size: a slot of main's size gives 0.512; giving it the top position adds 0.019 | main "has the highest median" | disagree on the explanation; agree on every number both could check | the verifier found the critic's sentence true in 0.451 of poles. The verifier's 0.512 variant changes two things at once (main's size fixed, and real dates replaced by random-group dates, which brings the extra ties), so it does not separate size from ties; nothing in this recount separates them either |

No bug was found in this recount after reading the two reports. The verifier's histogram total-variation distance,
the judge-flag split and the critic's eta-squared, turnover, scale and lithium findings were not recomputed here.
