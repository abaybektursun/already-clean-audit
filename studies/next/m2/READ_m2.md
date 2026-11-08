# M2 (J9), independent read: does the verdict reproduce?

Reader: a fresh Opus agent on branch `m2-read` (from `m2` at `540ea0f54`), 2026-11-06. Numpy on banked files and the
job's S3 outputs on the laptop; no GPU, no model run. Scripts and outputs in `studies/next/m2/read/`. Phase 1 (my own
numbers) was committed as `ce7683df8` before I opened `verdict_m2.json`, `thresholds.json`, the verdict and label
code of `infra/pg/m2.py`, the M2 journal or spec §12.

## Answer

**The headline reproduces exactly: H_B SUPPORTED, 42/71 = 0.592, Wilson 95 % [0.475, 0.698], against a bar of
0.0122** (`read/r2_from_ranks.json` → `test.pre_B`; `read/r7_floor.json` → `bar`). Every label matches:
`H_B SUPPORTED; H_sel_B BELOW_SELECTION; H_A SUPPORTED; H_AB LOSES`, and H_sel_A AT_SELECTION (`read/r8_compare.json`).
The job's ranks are right where I could recompute them. Score A: all 757 unit-role records, from the banked top-50
arrays, ranks equal 757 of 757 (`read/r3_A.json`). Score B: 155 records on 25 (layer, point) pairs, from the job's
vectors and the banked histograms, ranks equal 155 of 155, largest |B − B_job| 1.1e-16 (`read/r4_B.json`).

**The twin finding.** BELOW_SELECTION reproduces (hit_real − hit_twin −0.171 [−0.286, −0.057], `read/r2_from_ranks.json`
→ `test.twin_B`). The gap is not in the carrier's own percentile. It is in the other poles' percentiles on the
carrier's positions. It sits in the births whose exemplars repeat one or a few current tokens (" CHARSET", " Copyright").
On births whose exemplars carry varied tokens, carriers and twins hit at the same rate: 57 and 58 of 71, difference
−0.014 [−0.113, +0.085] over both halves (`read/r5b_token_share.json`). The twin construction almost never makes a
repeated-token group. So the twins' difference from real births can by itself produce BELOW_SELECTION. Lateness does
not produce it. This split is exploratory and was chosen after I saw a binned table (§5).

## 1. Blindness record

What I saw before the phase-1 commit that a blind reader should not have seen:
- The brief named 153 births, the halves 82 and 71, twins 79 and 70, "2,454 of 2,454", and asked whether the twins'
  difference "could by itself produce BELOW_SELECTION".
- `git log` on `m2`, run to find the base commit, printed the subjects of `540ea0f54` and `8fb431807`. They name the
  headline labels and numbers (42/71 = 0.592 [0.475, 0.698], bar 0.0122, twins 53/70, −0.171 [−0.286, −0.057],
  13/71 = 0.183, −0.408 [−0.521, −0.296], construction control 71/71, DISCOVERY B 39/82, A 7/82, twins 58/79).
- Spec sections read besides the listed ones: §1.2 (decision table), §1.4 (prior), §1.5, §2.4–2.5, §4 (gates), §9.4
  (S3 layout). Not read before the commit: §0 except §0.5, §10, §12, the journal, `verdict_m2.json`,
  `thresholds.json`, any function of `infra/pg/m2.py` or `m2_score.py`.

So my phase-1 code was written knowing the target numbers. What makes it independent is that it computes from
different inputs than the verdict does: A from the S3 top-50 arrays, B from the vectors and the S3 histograms, the
dates from `arrays.npz`, the matched sets from the S3 `max_abs`. The verdict computes from the job's records only.

## 2. Field by field: `verdict_m2.json` (test) and `thresholds.json` → `discovery` (freeze) against mine

Source of my numbers: `read/r2_from_ranks.json` (from the job's records), `read/r7_floor.json`, `read/r6_gates.json`;
the comparison is `read/r8_compare.json`. Hits are counts; intervals are Wilson 95 % or bootstrap 95 %.

| field | job | reader | largest difference |
|---|---|---|---|
| test pre_B (H_B) | 42/71, 0.5915 [0.4754, 0.6983] | the same | 9.1e-10 (the job's Wilson z is 1.959964, mine 1.95996398) |
| test pre_A (H_A) | 13/71, 0.1831 [0.1102, 0.2885] | the same | 9.3e-10 |
| test pre_ratio | 41/71 | 41/71 | 9.0e-10 |
| test pre_maxabs, density, colnorm | 0/71 each | 0/71 each | 7.7e-10 |
| test birth_B, birth_A (the ceiling) | 65/71 [0.828, 0.961]; 31/71 [0.327, 0.552] | the same | 8.9e-10 |
| test end_B, end_A (construction control) | 71/71, 71/71 | 71/71, 71/71 | 7.7e-10 |
| test end_opposite_B, end_opposite_A | 0/71, 0/71 | 0/71, 0/71 (every rank 22,016) | 7.7e-10 |
| test pb_pre_B, pb_pre_A (twins) | 53/70, 14/70 | 53/70, 14/70 | 9.5e-10 |
| test pair_real_pb_B (H_sel_B) | −0.1714 [−0.2857, −0.0571] | the same | 0 |
| test pair_real_pb_A (H_sel_A) | −0.0286 [−0.1571, +0.1000] | the same | 0 |
| test pair_A_B (H_AB) | −0.4085 [−0.5211, −0.2958] | the same | 0 |
| test floor_B, floor_A: 97.5th percentile | 0.0, 0.0 | 0.0, 0.0 | 0 (replicate mean differs by 7e-5 under B; see §7) |
| freeze pre_B, pre_A, pre_ratio | 39/82, 7/82, 38/82 | the same | 8.2e-10 |
| freeze birth_B, birth_A | 71/82, 34/82 | the same | 8.4e-10 |
| freeze end_B, end_A; opposite | 81/82, 82/82; 0, 0 | the same | 6.8e-10 |
| freeze pb_pre_B, pb_pre_A | 58/79, 19/79 | the same | 8.9e-10 |
| freeze pair_real_pb_B, pair_real_pb_A | −0.2532 [−0.3797, −0.1266]; −0.1646 [−0.2658, −0.0633] | the same | 0 |
| freeze pair_A_B | −0.3902 [−0.5000, −0.2683] | −0.3902 [−0.5000, −0.2805] | **0.0122** in the upper bound: the bootstrap resamples depend on the order of births. With births sorted by id, as `m2.usable` sorts them, I get −0.2683 exactly |
| freeze floor_B, floor_A: 97.5th percentile | 0.0122, 0.0122 (1/82) | the same | 0 |
| bar_B, bar_A | 0.012195 | 0.012195 | 0 |
| gates | pass, no problems, complete | 52 gates, all pass, every failing input fails, complete, nothing parked | none |
| cross-device G-EX points over 0.038718 | T_13, T_11, T_10, T_9, T_7 | the same | none |
| degenerate poles | T_10 L24N9826+, T_8 L07N640− | the same; L24N9826+ found again in my own read of the T_10 layer-24 histogram (`read/r4_B.json`) | none |
| sha256 of `ranks_freeze.jsonl`, of `metrics.json` | `864fd0dd…`, `172639f8…` | the same files, and equal to the S3 copies | none |

## 3. What I recomputed beyond the job's records

- **Births and dates** (`read/r1_births.json`). The dating rule (first t with median a⁺ ≥ q999 at t and t+1; T_15 if
  present only there) on `arrays.npz` gives the banked date for 2,454 of 2,454 sample-0 non-echo clusters. p_late ≤ 0.05
  gives exactly the 153 births of `births.json`; their k, ranks, positions, rows and halves all agree. 140 carriers,
  1,234 positions; the halves share no neuron. Every input's sha256 equals its registered value.
- **Twins** (`read/r1_births.json`, `read/r9_twin_draws.json`). 149 twins (79 and 70), 149 of 149 with the registered
  properties: same layer, another neuron, same size, positions equal the twin pole's T_15 top-50 at its ranks, non-echo
  ranks only, dated k by my own dating code. Four births have no twin (L03N7746+:c0, L11N10245−:c1, L20N8272−:c4,
  L21N7258+:c3). In phase 1 my guess at the draw order reproduced 33 of 153 twins (a scratch run, not banked). After reading `m2.pseudo_birth`
  (candidates sorted by id, 2,000 draws per candidate taken at once), my re-run reproduces 153 of 153, draw counts
  included.
- **Score A, all records** (`read/r3_A.json`). The `tok` member of 157 banked `topk/layer_LL.npz` files, read by HTTP
  range requests. 757 of 757 ranks, values and top-10 values equal the job's. Matched sets: 153 of 153 equal the job's
  list in the same order, and their A ranks equal. Construction control: A = |E_c| for 153 of 153 carriers at T_15.
  Opposite-pole ranks 302 of 302 equal. Failing input of my instrument: the neuron axis shifted by one gives the first
  carrier A = 0 instead of 3.
- **Score B, a sample** (`read/r4_B.json`). 25 (layer, point) pairs: the 20 pairs of the raw examples, plus T_5, T_8,
  T_10 and two T_15 pairs. 155 records: 45 births at the pre role (26 freeze, 19 test; 21 hits), 44 twins, 31 birth-role
  and 35 end-role records. My B (numpy median of a⁺ from the vectors; the 400-bin histogram, log-linear share in v's
  bin; B = 0 for v = 0 or an empty histogram) equals the job's on every record: ranks 155 of 155, top-10 names 155 of
  155, matched B ranks 45 of 45, ratio ranks 155 of 155. Failing input: the vectors' neuron axis shifted by one moves
  L01N369+:c0 from rank 3 to rank 2,135.
- **The carriers' own activations against the bank** (`read/r4_B.json` → `own_a_vs_bank_maxabs_by_t`, per-record
  `own_a_vs_bank_p99_d`). At T_15 and T_5 (L40S banks) the vectors equal the bank bit for bit. At the Blackwell-banked
  points the largest per-record p99 of |new − bank| / (0.05 + |bank|) is 0.088 (T_11, T_7). The largest absolute
  difference is 18.0, at T_7, where activations are of order 10²–10³. This agrees with the reported cross-device G-EX
  (T_7 p99 0.0864).
- **Floors** (`read/r7_floor.json`). From my own matched A ranks and the job's matched B ranks: freeze 1/82 under A and
  B, test 0 under both. Three different ways to draw one matched pole per birth give the same 97.5th percentiles.
  Bar = max(0.00045, 0.0122, 0) = 0.0122 under A and B. Most matched poles have v = 0 on E_c (B = 0, rank 22,016), so
  the floor stays near chance. Matched hits in total: freeze 3 under A and 6 under B, test 1 and 1.

## 4. Gates (`read/r6_gates.json`, from `banks/score-06/m2/metrics.json`)

52 gates. Each one passed and has a failing input that failed. `complete` true, nothing parked.
- K0: 63 entries on 10 poles, p99 0 (bit-identical); failing input (T_14's bank) p99 3.794 against 0.021159.
- I-ID: 5,415,936 activations of 66 units (33 births and their 33 twins), p99 0, 0 carrier flips under B; failing input
  p99 3.381.
- G-TOPK per point, p99 against its tolerance: T_15 0, T_14 0.0142, T_13 0.0132, T_12 0.0158, T_11 0.0130, T_10 0.0115,
  T_9 0.0098, T_8 0.0113, T_7 0.0068 (0.044273); T_6 0 and T_5 0 (0.021159; 4 entries each). Failing input p99 0.998 to
  1.403.
- G-EX gated (same device): T_15 2,295 entries, T_6 5, T_5 10, all p99 0; failing input p99 3.787, 1.211, 0.988.
- G-EX reported (cross-device, A3), p99 against 0.038718: over at T_13 0.0519 (11 of 591 read entries above), T_11
  0.0640 (8 of 538), T_10 0.0430 (4 of 378), T_9 0.0981 (8 of 240), T_7 0.0864 (4 of 114); under at T_14 0.0289,
  T_12 0.0325, T_8 0.0270. Each failing input p99 is 1.04 to 14.7.
- G0: gate max ratio 0.00335 to 0.00607; failing input u × 1.1, minimum over layers 0.0943 to 0.0980, fails at all 11
  points. The activation swap is reported and stays under 0.01 at T_6 (0.00970) and T_5 (0.00729), as A5 predicted.
- Degenerate poles: L24N9826+ at T_10 and L07N640− at T_8 only.

## 5. The twins (exploratory; `read/r5_twins.json`, `read/r5b_token_share.json`)

The spec (§3.3) names two ways a twin differs from its birth: its group is not a coherent cluster, and its date is
typical for its pole rather than late. A hit under B needs the carrier's own percentile v to beat every other pole's
own percentile on the same positions. So the paired gap can come from the unit's own side or from the competitors'
side. I split the two with the job's records. "Bar" below means the 10th-best B among the layer's poles on those
positions, taken from the record's top-10 list. Own B ≥ bar reproduces every real hit (41 of 41) and every twin hit
(53 of 53) on the test half.

**Own side: no difference.** At T_(k−1) the carrier's own B and the twin pole's own B have the same distribution: test
medians 0.99836 and 0.99782, twin above real in 33 of 70 pairs (pooled 0.99838 and 0.99816, 75 of 149). Swap the own
B and keep the real birth's bar: twins would hit 38 times where carriers hit 41; difference +0.043 [−0.043, +0.129]
(pooled +0.027 [−0.034, +0.087]).

**Competitor side: where the gap is.** The bar is higher on the real birth's positions: test medians 0.99663 against
0.98552, higher in 53 of 70 pairs. Put the twin's own B against the real birth's bar: 38 hits, not 53. Put the
carrier's own B against the twin's bar: 49 hits, not 41; difference −0.114 [−0.214, −0.029] (pooled −0.195
[−0.268, −0.128]).

**Lateness does not explain it.** My lateness proxy is the share of 2,000 uniform same-size groups of the pole's
non-echo ranks dated at k or later, from `arrays.npz`; it is not R1t. Real births sit at median 0.013 (90th percentile
0.041), twins at 0.215 (test 0.391). The 36 twins as late as the real median (proxy ≤ 0.013) hit 29 of 36 = 0.81
[0.65, 0.90]; their own births hit 15 of 36. The other 113 twins hit 82 times. Late twins lead their layers as often
as other twins. Nor is the twins' own ratio v / q999 higher: test medians 0.839 real and 0.754 twin; pooled 0.836 and
0.812.

**Repeated tokens explain it, in this split.** Take the share of distinct current tokens among a unit's exemplar
positions, from the grid's banked T_15 `samples/contexts.jsonl` (sha256 `a4d46cd8…`; positions checked against
`births.json`). It is 1/size when all positions carry one token. Real births: median 0.67, and 78 of 149 below 0.67.
Twins: median 1.0, and 16 of 149 below 0.67. Paired real − twin under B, split by the real birth's share:

| births | half | n | carrier hits | twin hits | difference [95 % bootstrap] | bar, real / twin (median) | own B, real / twin (median) |
|---|---|---|---|---|---|---|---|
| share ≥ 0.67 (varied tokens) | test | 31 | 25 | 27 | −0.065 [−0.194, +0.065] | 0.9907 / 0.9801 | 0.99821 / 0.99701 |
| share < 0.67 (repeated tokens) | test | 39 | 16 | 26 | −0.256 [−0.436, −0.077] | 0.99948 / 0.99566 | 0.99836 / 0.99834 |
| share ≥ 0.67 | freeze | 40 | 32 | 31 | +0.025 [−0.125, +0.175] | 0.9903 / 0.9805 | 0.99844 / 0.99853 |
| share < 0.67 | freeze | 39 | 6 | 27 | −0.538 [−0.692, −0.385] | 0.99961 / 0.99086 | 0.99815 / 0.99816 |
| share ≥ 0.67 | both | 71 | 57 | 58 | −0.014 [−0.113, +0.085] | 0.9907 / 0.9801 | 0.99838 / 0.99814 |
| share < 0.67 | both | 78 | 22 | 53 | −0.397 [−0.513, −0.282] | 0.99957 / 0.99442 | 0.99836 / 0.99825 |

On repeated-token positions other poles of the layer already sit near their own top percentile. The test half's
CHARSET birth shows it in one pair (`read/r5_twins.json` rows). L02N4649+:c3 has 1 distinct token among 7 positions.
Its own B is 0.99910 against a bar of 0.99998, so it ranks 597. Its twin L02N5291+ has 3 distinct tokens among 7. Its
own B is 0.99901, nearly the same, against a bar of 0.99889, so it ranks 7.

What this does and does not show:
- The 0.67 cut is a bin edge of a table I looked at before making the paired split. The freeze and test halves point
  the same way, but the split was not frozen. It is a hypothesis for a registered test, not a result.
- Token repetition is not the same thing as the spec's "coherent". A second coherence measure, the W4 judge's accuracy
  on the cluster's label (`judgments.jsonl`, 139 births), goes the other way: real hit rate 26/67 at judge accuracy ≤ 0.67
  and 45/72 above it. I have no explanation for that.
- Top-50 membership does not show the competitor effect. The other poles holding exemplar positions in their
  T_(k−1) top-50 are about as common for twins as for carriers. Test medians, carriers then twins: 1 and 2 poles,
  0.24 and 0.27 slots per position, 0.22 and 0.20 of positions held. The difference appears in the percentile bar, not in top-50 membership.
- Reading. BELOW_SELECTION is the frozen label and stands. Its decision-table reading ("other poles held the job's
  contexts first", §1.2) fits the data. But the twin does not separate a job-specific effect from an effect of repeated
  tokens, which the twin construction (random subsets of another pole's top-50) does not match. A twin drawn to match
  the birth's number of distinct current tokens would draw that boundary.

## 6. The raw examples

All 20 job examples decode as recorded. For 189 of 189 exemplars, the marked token and the position equal the banked
T_15 contexts (`read/r10_examples_decode.json`). The listed poles' "top3" contexts are their own top-50 at T_(k−1)
(`m2_score.example` reads the point's topk). They are not the exemplar positions. The quoted texts are from
`banks/score-06/m2/samples/raw_examples_{test,freeze}.jsonl`.

- **Test, B right: L22N7483−:c5** ("Miscellaneous prose and punctuation tokens … in web text and R code", 7 exemplars,
  pre T_12). The exemplars read as the name says: `pewinternet[[.org]]`, `pretty interesting[[.]]`, `geom_bar(stat[[=']]
  count'`, `posted a[[ rise]]`, `[[ etc]]`. There is no single job; the distinct-token share is 0.86. Under B the carrier
  is first (0.99884). The next nine poles (0.9982 to 0.9801) have their own top contexts on other topics (SQL commas,
  market reports, SEO), as a score that only says "high own percentile on these positions" allows. Under A every pole
  of layer 22 scores 0: no exemplar position is in any top-50 at T_12. Matched pole L22N3516+ (`Ladies and gentlemen[[,]]`,
  `Couch potatoes[[,]]`, comma-in-prose contexts) has v = 0 on these positions, rank 22,016. This hit is a carrier that
  leads because nothing else of its layer responds strongly to a mixed set. It is not a named job.
- **Test, B wrong: L02N4649+:c3** ("SQL dump CHARSET after DEFAULT", pre T_14). The exemplars read as the job: seven
  `ENGINE=… DEFAULT[[ CHARSET]]=…` lines. Carrier B 0.99910, rank 597. The top ten sit at 0.999984 to 0.999996. Three of
  them have a CHARSET line among their own top-3 contexts at T_14 (L02N5687−, L02N6933−, L02N9326+). Under A, L02N9310+
  holds 4 of the 7 positions in its top-50 and three others hold 3 each; the carrier holds 0. Other poles of layer 2
  held this job one point before the birth, as the score claims. Matched pole L02N9894− (`[[ Objective]] Raids`,
  `T[[onia]] Cochrane`, name and word pieces) reads generic, rank 4,150.
- **Freeze, B right: L16N1206+:c1** ("Source code and SQL, mid-expression tokens", pre T_13). The exemplars are mixed
  code tokens (`[[ identical]]`, `[[uation]]`, `[[,]]`, `[[_l]]abs`, `[[)]]`); distinct-token share 1.0. Carrier first,
  B 0.99838; the next poles fall to 0.973–0.934, a wide gap; their top contexts are unrelated prose. Under A the carrier
  holds 1 position and ties six others. Matched pole L16N4452+ (`F[[rowning]]`, `f[[rown]]`, ` stared`) has v = 0, rank
  22,016.
- **Freeze, B wrong: L10N9052+:c2** ("NPR 'Transcript provided by NPR, Copyright'", pre T_13). Five `Transcript
  provided by NPR,[[ Copyright]] NPR.` lines. Carrier B 0.585, rank 7,175. L10N3510+ has B 0.9999993 and holds all 5
  positions in its top-50 at T_13 (A = 5). Its own top-2 contexts are other tokens (`Turns[[ On]]`, `[[ delimiter]]`).
  Matched pole L10N4199+ (`textual[[ content]]`, `video and[[ text]] chat`) reads generic, rank 4,320.

My reading agrees with the journal's 07:53Z and 07:57Z entries where we read the same births (NPR and CHARSET). Both
B misses are single-token template jobs. Both B hits are mixed-token groups. That is the pattern of §5.

## 7. Code against the spec (read after `ce7683df8`)

None of these changes a label.
1. **Bootstrap order.** `m2.usable` sorts births by id, and `m2.paired` draws `rng.integers` per resample. The interval
   is therefore a function of the birth order. Births sorted by id reproduce every interval exactly. The freeze H_AB
   upper bound moves by 0.012 between two orders. That is the Monte Carlo error of the third digit. No label is near
   its boundary (test H_sel_B upper −0.057, H_AB upper −0.296).
2. **Floor draws.** `m2.floor_reps` uses a fresh `default_rng(0)` per score and half, over births sorted by id. The
   97.5th percentiles are the same under every draw I tried. The replicate means differ from mine by up to 7e-5 (the
   draws land on different rows).
3. **Wilson z** is 1.959964, not 1.95996398. Effect at most 1e-9. The bounds are clipped to [0, 1].
4. **Twin candidates** (`m2.pseudo_birth`) are the sample-0 clustered poles of the layer (600 of the 640 grid poles;
   e.g. L00N450± is excluded), sorted by id. All 2,000 draws of a candidate are made before any is dated. §3.3 says
   "another grid pole" and "up to 2,000 draws each". The code's reading follows the docstring. The difference matters
   only for reproducing the seeds.
5. **I-ID** scores all 66 pre-role units at T_13 (33 births and 33 twins) and checks flips on all 66. §3.2 names "the
   33 births with k = 14". The code checks more than the spec. Its failing input uses the 33 births' T_14 vectors.
6. **B below the histogram's first edge.** `m2.percentile` clips the bin index and the in-bin share, so 0 < v < 1e-6
   gives B = 0. §0.5 says B = 0 for v = 0. It is the same in practice: my searchsorted version agrees on 155 of 155.
7. **Matched ranks** are also written for the ratio score. That is an extra field and does not change any rule.
8. **`m2.gates_ok`** checks that each gate present passed and each failing input failed, and that `complete` is true.
   It does not check that the expected set of gates is present. It relies on the job's own asserts and on `complete`.
   In this run every gate has a failing input (`read/r6_gates.json`).
9. **ABOVE_CEILING, construction control, NOT_RUN and the bar** are computed as §5 states. The construction control
   compares the point value with 0.90. ABOVE_CEILING compares the pre-role Wilson lower bound with the birth-role
   upper bound.

## 8. What I could not check

- The forward pass itself. I took the vectors as given. Their correctness rests on K0, G-EX, G-TOPK and I-ID
  (bit-identical on the L40S points) and on my own comparison of the carriers' activations with the bank (§3).
- B for the 108 births outside my sample. Their B ranks are the job's numbers. The same code reproduced 155 of 155
  records on 25 pairs. The pairs cover every scored point except T_6, which holds one birth-role record.
- G-TOPK and I-ID per entry. The job banks only aggregates. The vectors hold all 11,008 neurons, but only at the
  exemplar positions.
- The top-3 contexts of the listed poles in the raw examples. I checked only the carriers' exemplars against the
  banked contexts.
- §7 step 5 asks for the recount "on an instance". The brief kept me on the laptop with no model run. Nothing here
  needed a forward pass.
- The twin analysis (§5) is exploratory and post hoc. The judge-accuracy proxy contradicts the token reading, and I
  have no explanation.

## Files

`studies/next/m2/READ_m2.md` (this file) and `studies/next/m2/read/`:
- `r1_births.py`/`.json`: dates, births, split, twins' properties.
- `r2_from_ranks.py`/`.json`: every quantity from the job's records; labels by §5.
- `r3_A.py`/`.json`: A from the S3 top-50 arrays; matched sets; competition counts.
- `r4_B.py`/`.json`: B from the vectors and the S3 histograms on 25 pairs.
- `r5_twins.py`/`.json`, `r5b_token_share.py`/`.json`: the twin split.
- `r6_gates.py`/`.json`: the gates.
- `r7_floor.py`/`.json`: floors and the bar.
- `s3range.py`: range reads of single .npz members, read-only.
- After the phase-1 commit: `r8_compare.py`/`.json` (field by field), `r9_twin_draws.py`/`.json` (seeds reproduced),
  `r10_examples_decode.py`/`.json` (raw-example decodes).
