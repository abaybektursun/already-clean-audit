# N1 independent read of the test half (2026-11-07)

**Answer.** The verdict's label stands under my recount: **NEITHER: in BELOW_SELECTION (NOT_ABOVE_FLOOR), out
AT_SELECTION (NOT_ABOVE_FLOOR).** The number that settles it is the input-side paired real − dt on e over the window:
**−0.432 [−0.817, −0.059]**, 67 births, percentile bootstrap, 10,000 resamples, seed 0 (`read_n1/verdict_recount.json`).
The upper bound is below zero under all seven interval methods and seeds I tried. The registered verdict code, run once
into the scratchpad, gives the same headline and the same 180 numbers to 1.1e-16 (`read_n1/compare.json`).

The label has little margin, and it does not mean the carriers change less than ordinary neurons:
- Margin. Removing the single most negative birth leaves the upper bound at −0.0024. Removing the two most negative
  gives −0.302 [−0.649, +0.041], which reads AT (`read_n1/fragility.json`).
- The freeze half read AT with the same sign (−0.217 [−0.638, 0.204]). The halves do not differ: test − freeze −0.215,
  z −0.74, p 0.46. If the true mean were the freeze estimate, the test half reads BELOW with probability 0.19–0.20.
  The two halves pooled: −0.320 [−0.604, −0.035].
- In the test half, carriers sit at the layer's mean e_in (carrier − layer mean +0.11 [−0.15, 0.36]). They also sit at the mean of their
  activation-matched neurons (+0.23 [−0.04, 0.51]). The twins sit above the layer mean (dt +0.52 [0.26, 0.78], pb +0.61
  [0.38, 0.83]). The gap comes from twins changing more than an average neuron, not from carriers changing less
  (`read_n1/baseline.json`; exploratory, not registered).
- Carriers are not matched to their dt twins on activation size. At T_(k−1) their max |activation| is 1.39 times the
  twin's (log difference 0.329 [0.202, 0.468], Wilcoxon p 3e-5). Adjusting the paired gap for it gives −0.20 [−0.59,
  0.21]. Within each kind, though, e_in does not track log max |activation| (slopes −0.03, −0.01, −0.06 per unit log,
  all p > 0.27). I find no direct evidence that activation size produces the gap.
- In natural units the gap is small. One z unit is 2–6 % of r in brackets 4–15. Over the window, the carriers' input rows
  changed 1.6 % less per bracket than their twins' (log r, [−3.2 %, +0.03 %], no reference subtracted).

Worktree `interp_research-n1read`, branch `n1read`, from tag `n1-freeze-2` (commit `6709f754e`). CPU only. No model run.
I read no file outside the worktree except the S3 objects named below. Scripts and outputs are in `studies/next/n1/read_n1/`.

---

## 1. Inputs and their provenance

S3 `s3://pg-lineage-769063704696/results/next_n1/weights-03/out/weights/n1/`, read 2026-11-07, sha256 (`read_n1/gates.json`):

| file | sha256 | banked copy in the worktree |
|---|---|---|
| `stats.npz` (342,541,143 bytes) | `36b5ce46…7a33`, as registered in the brief | not banked |
| `metrics.json` | `d801d09a…9b56` | equal; equals `thresholds.json` → `sha256_job_metrics` |
| `n1_freeze.jsonl` | `f00f65d7…b10e` | equal; equals `thresholds.json` → `sha256_n1_freeze` |
| `samples/raw_examples_freeze.jsonl` | `515fae90…6884` | equal |
| `n1_test.jsonl` | `e9de79c5…e595` | not banked |
| `samples/raw_examples_test.jsonl` | `53137c08…1321` | not banked |

`thresholds.json` sha256 `fad4fe71…a966`. `units.json` `ada35f38…`, `kb_bank.json` `401c6134…`: the registered values.

## 2. Gates (`read_n1/gates.py` → `gates.json`)

`metrics.json`: `complete` true, 49 gates, all pass. Every gate has a failing input, and all 49 failing inputs fail.

| gate | value | pass rule | failing input's value |
|---|---|---|---|
| G_SAME | torch 2.12.0 | equals the registered 2.12.0 | version "0": does not match |
| INPUTS | no mismatch | sha256 of the three inputs as registered | an all-zero sha256: does not match |
| G_REV ×16 | the 16 commits equal `m2.POINTS` (16 of 16; branch names equal too) | commit = registered revision | an all-zero revision: does not match |
| G_SHAPE ×16 | 32 layers, no bad layer, bf16 | [N,H], [N,H], [H,N] | the down tensor transposed: does not match |
| PC ×14 (brackets 2–15) | planted share 1.0 on both sides at every bracket | ≥ 0.9 on both sides | planted size 0: share 0.038–0.068 |
| I_ID | norms bitwise equal on the second read; T_15 − T_15 = 0 for every neuron | both true | T_14 against T_15 is all zero: false |

The failing inputs of G_SAME, INPUTS, G_REV and G_SHAPE are fixed comparisons inside the job. They fail by
construction. The dry run is where these gates were shown to stop the job; I did not rerun it. The failing inputs of
PC, I_ID and K-B come from the data.

**K-B (reported, not gated).** The job reports max_rel 2.0e-8 to 1.49e-7 at the 11 points t05–t15 (t15 over 352,464
columns), against the tolerance 1.3e-4. Its failing input (our T_(t−1) against the bank's T_t) reads max_rel 0.062–0.793.
My recount from `stats.npz` (`read_n1/recount.json` → `K_B`) covers all 775 banked values and N6's 352,256 T_15 values:
largest relative difference 1.49e-7. Three column norms written out:

| layer, neuron, t | kb_bank.json | stats.npz | relative |
|---|---|---|---|
| L3 N9245, T_5 | 1.6101737022 | 1.6101737263 | 1.5e-8 |
| L1 N8419, T_10 | 2.2990562916 | 2.2990561370 | 6.7e-8 |
| L0 N181, T_15 | 0.9653901458 | 0.9653901833 | 3.9e-8 |

The failing input, taken column by column, has its smallest relative difference at t11: 0.00056. That is 4.3 times the
tolerance. It is smaller than the offline gate's 0.00155, which was taken over bank-to-bank adjacent pairs only. Every
column still fails.

**Internal consistency of stats.npz.** For every vector, ‖a − b‖² = ‖a‖² + ‖b‖² − 2a·b holds to 5.0e-16 relative
(norms, diff and dot are separate arrays). With the norms shifted by one bracket, the median relative error is 0.10. So
the bracket indexing of `diff` and `dot` is right. Smallest change norm 0.032; no zero change.

## 3. The statistic (`read_n1/recount.py` → `recount.json`, `units_recount.json`)

My code computes r, z and e from `stats.npz` and imports nothing from `infra.pg`. Medians use an explicit sort. In:
r = √(Δgate² + Δup²) / √(gate² + up²) at T_(j−1). Out: Δdown / down. z: log r minus the layer's median over 11,008
neurons, divided by 1.4826 × MAD, per bracket and layer. e: mean z over brackets k−2, k−1, k minus mean z over
brackets 2 … k−3. Ceiling: mean over the window of the layer's 99th percentile of z.

Compared with the job's records for all 442 units (234 freeze, 208 test), sides in, out, gate, up:

| quantity | largest absolute difference |
|---|---|
| e over the window | 8.9e-15 |
| e_k | 2.5e-14 (out, L29N38+:c1) |
| e per bracket k−2 … k+1 | 2.9e-14 |
| r, cos | 0 (bitwise) |
| z | 5.0e-6 (the records round z to 5 decimals) |
| matched neurons' e | 2.0e-14 |
| ceiling | 6.2e-15 |

Ten test units are written out in `recount.json` → `ten_units`: five real (L00N3387+:c2, L03N9245+:c5, L09N10681+:c6,
L20N5759+:c2, L25N2013+:c0) and five dt (the twins of L00N3387+:c2, L03N9245+:c5, L08N10397-:c1, L20N5759+:c2,
L24N8193-:c1). Each lists its window z values. Difference from the record on e_in and e_out: 0.0 for all ten. The
largest e difference over all units is 2.5e-14.

My recount of the freeze half reproduces `thresholds.json` (floors, ceilings, DISCOVERY summary): 38 numbers, largest
difference 7.8e-16 (`read_n1/freeze_check.log`).

## 4. The verdict quantities, test half (`read_n1/verdict_recount.py` → `verdict_recount.json`)

Paired by birth, real − dt (67 pairs; 4 of the 71 carriers have no dt twin). Percentile bootstrap, 10,000, seed 0,
unless named.

**Input side, e over the window: −0.4317.** Real above its twin in 23 births, below in 44.

| method | 95 % interval | label |
|---|---|---|
| percentile, seed 0 (registered) | [−0.8167, −0.0590] | BELOW |
| percentile, seed 1 | [−0.8095, −0.0526] | BELOW |
| percentile, seed 2 | [−0.8081, −0.0623] | BELOW |
| percentile, 100,000 resamples in one draw, seed 0 | [−0.8121, −0.0562] | BELOW |
| basic, seed 0 | [−0.8044, −0.0467] | BELOW |
| BCa, seed 0 (z0 0.012, acceleration −0.003) | [−0.8135, −0.0561] | BELOW |
| Student t | [−0.8186, −0.0448] | BELOW |
| cluster bootstrap by carrier neuron (58 clusters) | [−0.823, −0.059] | BELOW |

Sign test p 0.014; Wilcoxon p 0.023; median −0.410 [−0.951, −0.216]; 10 % trimmed mean −0.428.

**Output side: −0.2581.** Real above in 26, below in 41. Percentile seed 0 [−0.6545, 0.1302]; seed 1 [−0.6592, 0.1330];
seed 2 [−0.6544, 0.1277]; 100,000 [−0.6528, 0.1310]; basic [−0.6464, 0.1383]; BCa [−0.6589, 0.1276]; t [−0.6603,
0.1441]; all AT. Sign test p 0.086; Wilcoxon p 0.165.

The label does not depend on the method.

**Reported quantities.**

| quantity | in | out |
|---|---|---|
| real − pb (70 pairs) | −0.500 [−0.839, −0.180]; 29 above, 41 below | −0.209 [−0.509, 0.086]; 31 / 39 |
| gate rows alone, real − dt | −0.387 [−0.721, −0.056]; 27 / 40 | — |
| up rows alone, real − dt | −0.374 [−0.761, −0.0014]; 23 / 44 | — |
| e_k (birth bracket alone), real − dt | −0.488 [−0.955, −0.028]; 27 / 40 | −0.410 [−0.905, 0.064]; 27 / 40 |
| e_k, real − pb | −0.643 [−1.066, −0.238] | −0.366 [−0.707, −0.026] |
| real e_k | −0.412 [−0.728, −0.111] | −0.148 [−0.436, 0.145] |
| real mean e (71) | −0.297 [−0.550, −0.059] | −0.039 [−0.276, 0.202] |
| dt mean e (67) | +0.112 [−0.158, 0.377] | +0.193 [−0.085, 0.484] |
| pb mean e (70) | +0.201 [−0.016, 0.414] | +0.175 [−0.025, 0.367] |
| real gate, up | gate −0.122 [−0.332, 0.082]; up −0.344 [−0.613, −0.095] | — |

The registered code reports real − pb for gate −0.331 [−0.626, −0.044] and up −0.511 [−0.848, −0.187]. I did not
recompute those two.

**Profile by bracket, real e (registered readout)**, offsets k−2, k−1, k, k+1 (n 71, 71, 71, 64):
- in: −0.183 [−0.442, 0.070]; −0.296 [−0.589, −0.026]; −0.412 [−0.728, −0.111]; −0.589 [−1.052, −0.178].
- out: +0.093 [−0.141, 0.329]; −0.062 [−0.329, 0.205]; −0.148 [−0.436, 0.145]; −0.240 [−0.541, 0.070].

**Paired profile real − dt (not registered)** (n 67, 67, 67, 62):
- in: −0.235 [−0.626, 0.163]; −0.571 [−0.994, −0.153]; −0.488 [−0.955, −0.028]; −0.632 [−1.251, −0.029].
- out: −0.039 [−0.427, 0.335]; −0.325 [−0.754, 0.095]; −0.410 [−0.905, 0.064]; −0.558 [−1.068, −0.056].

The input-side gap is still there at k+1, after the birth bracket. It is not confined to the rise window.

**Floors and ceilings.**

| | in | out |
|---|---|---|
| matched floor, test (1,000 replicates, seed 0, 99 neurons per birth) | mean −0.542, q025 −1.081, q975 −0.090 | mean −0.242, q025 −0.558, q975 0.067 |
| matched floor, freeze (thresholds.json) | q975 −0.0534 | q975 0.1468 |
| bar = max(freeze, test) q975 | −0.0534 | 0.1468 |
| real mean's lower bound | −0.550 | −0.276 |
| floor label | NOT_ABOVE_FLOOR | NOT_ABOVE_FLOOR |
| ceiling, test (median over real births) | 2.402 | 2.026 |
| real lower bound above the ceiling? | no | no |

## 5. The §5 label rules applied by hand

1. The freeze records' sha256 equals `thresholds.json` (`f00f65d7…`). 2. Gates pass and `complete` is true. 3. 71 real
test units ≥ 50. 4. sel_in: hi −0.0590 < 0 → BELOW_SELECTION. sel_out: lo −0.6545 < 0 < hi 0.1302 → AT_SELECTION.
Both have 67 ≥ 50 pairs. floor_in and floor_out are NOT_ABOVE_FLOOR. No real lower bound exceeds its ceiling. 5. No
side is NOT_RUN or ABOVE_CEILING. 6. Neither side is ABOVE. Not both AT. So the hypothesis is NEITHER, not the H_0
form.

**My label: NEITHER: in BELOW_SELECTION (NOT_ABOVE_FLOOR), out AT_SELECTION (NOT_ABOVE_FLOOR).** In §1.3's terms:
- H_rows is refuted (in BELOW).
- H_col is refuted (out AT).
- H_together is refuted.
- H_sel is refuted ("any ABOVE or BELOW").

The outcome is the one the §1.4 prior gave 0.1: "a BELOW label on either side".

## 6. The registered verdict code

`python3 -I` with the worktree on `sys.path`, running `infra.pg.n1` as `__main__` with `verdict --bank <scratchpad>/bank
--thresholds studies/next/n1/thresholds.json --out <scratchpad>/verdict/verdict_n1.json`. The bank directory holds the S3
copies of `metrics.json`, `n1_freeze.jsonl` and `n1_test.jsonl`. Output sha256 `71240078…56ad`. A copy is at
`read_n1/verdict_n1_registered_code_rerun.json`.

Printed headline: `NEITHER: in BELOW_SELECTION (NOT_ABOVE_FLOOR), out AT_SELECTION (NOT_ABOVE_FLOOR)`. Gates pass,
thresholds sha256 `fad4fe71…`. Compared with my recount: 180 numbers, largest difference 1.1e-16. These cover every
paired diff, lo, hi and count for dt (4 sides, window and e_k) and for pb (in and out); every group mean and interval;
every profile; both floors, both ceilings and both bars. The labels are equal.

## 7. Confounds

**(a) e against the birth bracket k.** Twins carry their birth's k in 67 of 67 dt pairs and 70 of 70 pb pairs
(`read_n1/matching.json`). So the paired gap cannot come from a k difference between carriers and twins.

Within each group, e_in does not depend on k. Test half slopes per unit k: real −0.006 (se 0.063, p 0.93), dt −0.090
(se 0.072, p 0.22). Both halves pooled: real −0.023 (p 0.67, n 153), dt −0.068 (p 0.29, n 140).

The paired gap against k: test slope +0.086 (se 0.105, p 0.41). By k bin (test): k 6–11 −0.44 (n 24), k 12 −0.71 (13),
k 13 −0.88 (12), k 14–15 +0.08 (18). In the freeze half the bins run +0.39, −0.77, +0.31, −0.39. There is no consistent
k pattern across halves.

A later birth has a longer reference (k−4 brackets). That does not show up as a trend in e.

**(b) Matching of carriers and twins.** dt twins match their carrier on layer, k, group size and number of distinct
tokens in 67 of 67 pairs. No twin is its carrier's neuron. pb twins match on layer, k and size; on distinct tokens in
only 14 of 70, by design.

Activation, at T_(k−1), test half, paired (N12's `ranks_test.jsonl`, role pre; `onsets.json`):
- Cluster median activation over the pole's own threshold: carriers 0.821, twins 0.786 (medians). Paired difference
  +0.019 [−0.059, 0.098]. Matched, as the dating rule makes them.
- max |activation|: carriers larger by 0.329 in log [0.202, 0.468] (×1.39), Wilcoxon p 3e-5. Freeze half: 0.246
  [0.121, 0.372]. Not matched. Unpaired, the two distributions overlap (KS p 0.86); the difference is within pairs.
- The ratio at T_k: carriers higher by 0.30 [0.10, 0.52].
- Weight norms at T_(k−1): carriers' input rows 2.4 % [0.4 %, 4.5 %] longer, output column 2.0 % [0.5 %, 3.5 %] longer.

Does activation size explain the gap? Regressing the paired e_in gap on the log max-activation difference: intercept
−0.20 [−0.59, 0.21], slope −0.70 [−1.43, 0.06] (`fragility.json`). Pairs whose activation sizes are close (|Δ log| at or
below the median, 34 pairs) show −0.436 [−0.857, +0.001]. Far pairs show −0.427. Within each kind, e_in has no slope on
log max-activation: test real −0.026 (p 0.64), dt −0.009 (p 0.88), pb −0.056 (p 0.27). The mismatch is real. Adjusting
for it removes the label, but only by widening the interval. The data do not show that activation size produces the
gap.

The q999 threshold itself is in J1's bank arrays, outside the worktree. I did not read it (§9).

**(c) The two halves.**

| side | freeze | test | test − freeze | z, p | P(test BELOW \| freeze estimate) | pooled, 140 pairs |
|---|---|---|---|---|---|---|
| in | −0.217 (se 0.217) | −0.432 (se 0.194) | −0.215 | −0.74, 0.46 | 0.200 normal; 0.186 resampling freeze pairs | −0.320 [−0.604, −0.035] BELOW |
| out | −0.189 (se 0.228) | −0.258 (se 0.201) | −0.069 | −0.23, 0.82 | 0.154; 0.087 | −0.222 [−0.535, 0.060] AT |

The test label is about a one-in-five outcome if the freeze estimate were true. The halves agree within their
intervals.

The carriers' mean e_in is stable across halves: −0.337 freeze, −0.297 test. The twins' means move: dt −0.093 → +0.112,
pb −0.422 → +0.201. The test half reads BELOW mainly because its twins are higher.

**(d) Room.**
- ABOVE. At the test SE (0.194 in, 0.201 out), ABOVE needs a paired mean of about +0.38 (in) or +0.39 (out). That is a
  shift of +0.82 from the observed −0.43. A paired mean of +0.38 is a quarter of the per-birth sd (1.59): reachable.
- Matched floor. The real lower bound must exceed −0.053 (in) or +0.147 (out). With half-widths of about 0.245, the real
  mean would need about +0.19 (in) or +0.39 (out). The test half's dt (+0.11) and pb (+0.20) means on the input side
  lie in that range, so the floor could have been passed.
- Ceiling. 2.40 (in) and 2.03 (out) against real lower bounds of −0.55 and −0.28. The single-unit maxima are 1.61 (in)
  and 3.48 (out); one of 71 carriers is above its own ceiling on the output side. The ceiling was far from firing.

**(e) Other findings.**
1. *Single births.* No single drop flips the percentile label (upper bound range −0.121 to −0.0024). Three drops flip
   the t interval. Dropping the two most negative pairs gives +0.041 (AT). The three most negative pairs: L25N2013+:c0
   (−4.71, k 12), L10N5687+:c7 (−4.57, k 12), L03N9245+:c5 (−4.18, k 8).
2. *Window and reference.* The input-side gap −0.43 is the window part, real − dt mean z −0.27 [−0.65, 0.10], minus the
   reference part, +0.16 [−0.09, 0.40]. Carriers' input rows changed slightly more than their twins' in the reference
   brackets and slightly less in the window. Neither part is below zero alone.
3. *Against the whole layer and the matched neurons* (`baseline.json`, exploratory). Over all 11,008 neurons of a layer
   with the same k, the mean e_in is −0.41 and the median +0.02. The distribution has a long negative tail. Against the
   layer mean:
   - test: carriers +0.11 [−0.15, 0.36], dt +0.52 [0.26, 0.78], pb +0.61 [0.38, 0.83];
   - freeze: carriers −0.01, dt +0.25 [−0.11, 0.57], pb −0.10.

   Carrier minus the mean of its matched neurons: test +0.23 [−0.04, 0.51], freeze +0.07 [−0.21, 0.37]. The BELOW label
   is relative to dated twins only. Against the layer and the activation-matched neurons, the carriers are not below.
4. *Units.* One z unit is 1.4826 × MAD of log r: 0.022–0.058 in log r for the input side in brackets 4–15 (0.033–0.083
   for the output side). In raw log r over the window, the paired gap is −1.6 % [−3.2 %, +0.03 %] (in) and −1.5 %
   [−3.7 %, +0.6 %] (out) (`scale.json`). The median r per bracket is 0.47–0.77 in brackets 4–15. Neurons of a layer
   turn at nearly the same relative rate.
5. *Dependence.* The 67 pairs use 58 carrier neurons and 60 twin neurons. 16 twins' neurons are carriers of other test
   births. 26 neurons occur more than once among the pairs. The cluster bootstrap by carrier gives the same interval
   ([−0.823, −0.059]). It does not remove the dependence through shared twin neurons.
6. *Code.* The `infra/pg/n1.py` docstring says the reference is brackets 2 … j−2. The code (`reference`) and the spec
   both use 2 … k−3. The docstring is stale; no number depends on it.

## 8. Raw examples (`read_n1/examples.py` → `examples.json`)

Of the ten test example births, 9 have real e_in below their dt twin's (the label's direction) and 1 above. In the test
half overall it is 44 of 67. Every number in `raw_examples_test.jsonl` equals my recount from `stats.npz` (largest
difference 2.2e-16). The exemplars are the first ones in N12's banked raw examples of the same births
(`studies/next/n12/banks/score-02/n12/samples/raw_examples_test.jsonl`).

**With the label: L22N7483-:c5** (layer 22, k 13, window brackets 11–13, size 7, 6 distinct tokens).
- Carrier: e_in −2.367, e_out −0.960, e_k in −2.411. r_in in the window 0.718, 0.698, 0.653; cos_in at k 0.764.
- dt twin L22N7806-: e_in +0.475, e_out +0.920. r_in 0.768, 0.789, 0.741.
- pb twin L22N4270-: e_in +0.316, e_out −0.440.
- Real − dt: in −2.842, out −1.880. A raw r difference of 6–12 % per bracket becomes about 3 z units.
- Carrier exemplar: "… Pew's social media use data [http://www.pewinternet [[.org]] /data-trend/…" and "… thought this
  was pretty interesting [[.]] I ran a batch analysis …".
- Twin exemplar: "… it will be combating the [[ Taliban]] ; it will be population security."
- Record against recount: 0.0.

**Against the label: L03N9631+:c3** (layer 3, k 13, size 10, 8 distinct tokens).
- Carrier: e_in +0.464, e_out +0.036, e_k in +0.566. r_in 0.765, 0.814, 0.789.
- dt twin L03N10451-: e_in +0.423, e_out −0.016. r_in 0.763, 0.808, 0.764.
- pb twin L03N1487+: e_in −0.021, e_out +0.557.
- Real − dt: in +0.042, out +0.052. This is the only example against the label, and by a small amount.
- Carrier exemplar: "… Spence's An [[ecd]] otes, only tantalise our curiosity …" and "… he could have few rivals as an
  [[ ump]] ire."
- Twin exemplar: "Express- [[Checkout]] as PDF".
- Record against recount: 0.0.

## 9. What I could not verify

1. `stats.npz` against the checkpoints themselves. No weights were downloaded (CPU only, no model). The checks are the
   law of cosines across its three arrays and K-B against 775 + 352,256 banked fp32 column norms. The input-row norms
   have no bank; only their internal consistency is checked.
2. PC, I_ID and G_REV are read from `metrics.json`. I did not rerun them. The revisions match `m2.POINTS`, but I did not
   check them against the HF hub.
3. The fixed-comparison failing inputs (G_SAME, INPUTS, G_REV, G_SHAPE). Their real demonstration is the dry run, which
   I did not rerun.
4. The carriers' and twins' q999 activation thresholds. They live in J1's bank arrays outside the worktree. I used max
   |activation| at T_(k−1) and the threshold ratio from `onsets.json` instead.
5. Real − pb for gate and up alone: quoted from the registered code's output, not recomputed.
6. The exemplar texts come from N12's bank. I did not check them against the corpus.

## 10. Files

`studies/next/n1/read_n1/`:
- `recount.py` (+ `recount.json`, `units_recount.json`, `recount.log`)
- `gates.py` (+ `gates.json`, `gates.log`)
- `verdict_recount.py` (+ `verdict_recount.json`, `verdict_recount.log`)
- `fragility.py` (+ `fragility.json`, `fragility.log`)
- `baseline.py` (+ `baseline.json`, `baseline.log`)
- `scale.py` (+ `scale.json`, `scale.log`)
- `matching.py` (+ `matching.json`, `matching.log`)
- `examples.py` (+ `examples.json`, `examples.log`)
- `compare.py` (+ `compare.json`, `compare.log`)
- `verdict_n1_registered_code_rerun.json`
- `freeze_check.log`
