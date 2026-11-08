# N6 (J11): edit the young. Pre-registration

**Answer.** Two launches on one L40S answer the card's question, projected at 3.34–7.95 GPU-h in all ($7.48–17.81 at $2.24 per
hour; `studies/next/n6/projection.json`). The matching the card asks for is done and shown from the banked arrays: 74 pairs in one
layer each, the two members at least three grid points apart in main-cluster birth, matched on ln max|a| (caliper 0.10) and ln
density (caliper 0.50). On these pairs the critic's statistic, with each pair as its own stratum, is ρ(age, ln max|a|) = +0.027
(bound 0.10; unmatched within-layer value −0.592) and the standardized mean difference of ln max|a| is +0.005 (bound 0.10)
(`studies/next/n6/pairs.json` → `balance`). The column norms and the activation energies on the eval text are not banked; L1 (0.29–0.67
GPU-h) measures them with the known numbers and the ten raw examples, and the final pairs are matched on them offline before L2
(3.05–7.28 GPU-h) spends. Every corpus loss is read from an fp32 forward, so the rounding floor is measured, not assumed. The weight-free
dry run printed `N6_DRYRUN_OK` (78 checks, 0 failed; 83 after amendment A1, 99 after A2 and A3, 117 after A4, 136 after A5, §14), and every gate whose quantity the banks hold passes on its pass side and fails
on its registered failing input (`N6_OFFLINE_GATE.md`). After amendment A4 (1,024 eval windows, the positive control in 8 layers) L2 is
projected at 35.20 GPU-h center and 44.01 high from L1-02's measured rate (`studies/next/n6/projection_L1.json`; §14 A4). After
amendment A5 L2 runs as two launches: L2a on the freeze half with every gate (19.05 / 23.82 GPU-h), whose records decide freeze 2 and
the power gate, and L2b on the test half (18.24 / 22.80 GPU-h), launched only if the power gate passes (§14 A5).

Written 2026-11-06 by an Opus worker in worktree `interp_research-n6`, branch `n6` (from `clean-neurons` at `3b6e1d34d`). No
model was run, no weights were downloaded, nothing was launched. Network reads: S3 read-only (six small JSON files and one zip
directory read by byte range, §0.2); three web searches and one full-text read (§1.5).

---

## 0. Status and provenance

| field | value |
|---|---|
| status | **CLOSED 2026-11-08: DOES NOT COUNT (G-PC failed and its failing input passed; G-POWER failed, no test label).** Verdict `studies/next/n6/verdict_n6.json` (sha256 `b774450623aac2caab55a8fb482dbce9fc04a09e3ea11a9931d2586a41c33599`): `H_age DOES NOT COUNT; H_adj DOES NOT COUNT; H_eff DOES NOT COUNT`, each realized as UNDECIDED_NO_POWER; L2b not launched (the head's decision, 2026-11-08 15:15Z). History: FROZEN at freeze 1. The head's review (2026-11-06 15:10Z) accepted the four design calls as written; nothing in the draft changed. No GPU number exists at the tag. L1-01 stopped at G-POLE and does not count; amendment A1 (§14) and freeze 1a followed. L1-02 completed; amendments A2 (G-ZN and G-CACHE tolerances), A3 (the stage-2 split) and A4 (G-RES at the factor 50; 1,024 eval windows, G-POWER at freeze 2, the positive control in 8 layers) are written (§14). Amendment A5 (L2 as two launches, L2a on the freeze half, L2b on the test half after freeze 2 with G-POWER passed) is written (§14). L2a-01 completed on 2026-11-08 (113 of 113 edits, nothing parked); G-PC failed (col > rand in 2 of 8 layers, its failing input passed, 6 of 8); freeze 2 is tagged with G-POWER failed (ratio 1.186, bar 0.5); L2b is not launched. |
| freeze 1 | annotated tag `n6-freeze-1` (tag object `9f9c3b040`) on `743d22324` (`743d223244a2ca93cdd13faaa5495b9356fef2ce`), 2026-11-06T15:13:49Z. Before the tag the dry run was rerun on that commit: `N6_DRYRUN_OK`, 78 checks, 0 failed, every CHECK line equal to the draft's run, `dryrun_summary.json` equal (`studies/next/n6/offline_gate/dryrun_freeze1.log`); `gate_sides.py` printed `GATE_SIDES_OK`, output equal to the committed `gate_sides.json`. |
| freeze 1a | annotated tag `n6-freeze-1a` (tag object `ba2674cc5`) on `e5c5f21f2` (`e5c5f21f23bebd49bbb94a14db50b3b03e0ce0cd`), 2026-11-06T15:48:33Z, after amendment A1 (§14: G-POLE's tolerance 0.05; `g_pole.npz`; peak memory per phase; the stage-2 file `pairs_L2.json`). Dry run on that tree: `N6_DRYRUN_OK`, 83 checks, 0 failed (`studies/next/n6/offline_gate/dryrun_freeze1a.log`); `gate_sides.py` `GATE_SIDES_OK`. `INPUT_SHA["L1"]` and `next_n6_L1.json` unchanged. Gates L1-02. |
| freeze 1b | annotated tag `n6-freeze-1b` (tag object `4cc2251e0`) on `eba7035cb` (`eba7035cbfd874c433274f09b720ea0b4ff6989a`), 2026-11-06T17:28:17Z, after amendments A2, A3 and A4 (§14). `INPUT_SHA["L2"]`: `pairs_L2.json` `7e3a99480b3a2b939ca6eb0a8ae6fe43ee0e8bad22bee302c1a37ca1e4e1755b`, `e1_known.json` `79da230c…`; `next_n6_L2.json` `--budget-min 3089`; hard kill 3,433 min; L2 projected at 35.20 / 44.01 GPU-h (`projection_L1.json`). Before the tag the dry run was rerun on that commit: `N6_DRYRUN_OK`, 117 checks, CHECK lines and `dryrun_summary.json` equal to `offline_gate/dryrun_A4.log`'s run. Gates L2. |
| freeze 1c | annotated tag `n6-freeze-1c` (tag object `843f66877`) on `b94dbd8e1` (`b94dbd8e1d405af3c5e8760696774e4ba9cf3989`), 2026-11-06T17:54:17Z, after amendment A5 (§14: L2 as two launches; `--half freeze` L2a, `--half test` L2b with `--against` L2a's `metrics.json`). `INPUT_SHA["L2"]` unchanged. `next_n6_L2a.json` `--budget-min 1672`, hard kill 1,858 min, L2a projected at 19.054 / 23.818 GPU-h; `next_n6_L2b.json` `--budget-min 1601`, hard kill 1,779 min, L2b at 18.239 / 22.799 GPU-h (`projection_L1.json`). Before the tag the dry run was rerun on that commit: `N6_DRYRUN_OK`, 136 checks, CHECK lines and `dryrun_summary.json` equal to `offline_gate/dryrun_A5.log`'s run; `gate_sides.py` output equal. Gates L2a; L2b also needs `n6-freeze-2` with G-POWER passed. |
| freeze 1 (as planned) | annotated tag `n6-freeze-1` on the commit that holds this spec, `infra/pg/n6.py`, `infra/aws_pg/jobs/n6_edit.py`, `infra/aws_pg/jobs/n6_dryrun.py`, `infra/aws_pg/steps/next_n6_L1.json`, `studies/next/n6/pairs.json` (sha256 registered in `n6_edit.INPUT_SHA["L1"]`), `studies/next/n6/inputs/e1_known.json` (same), the offline-gate files. Gates L1. |
| freeze 1b (planned) | after L1 and the head's read of the ten raw examples: `studies/next/n6/pairs_L2.json` from `n6 final` on L1's banked column norms and energies, its sha256 in `INPUT_SHA["L2"]`, `infra/aws_pg/steps/next_n6_L2.json` with the budget from L1's measured rate. Gates L2. Prepared with A2, A3 and A4: `pairs_L2.json` sha256 `7e3a9948…` in `INPUT_SHA["L2"]` (24 test, 21 freeze; 1,024 eval windows; was `2d206ffc…` with 96), `next_n6_L2.json` `--budget-min 3089`, hard kill 3,433 min (`projection_L1.json`); dry run `N6_DRYRUN_OK` 117 checks (`offline_gate/dryrun_A4.log`). |
| freeze 2 | annotated tag `n6-freeze-2` (tag object `adb92679a`) on `037948fac` (`037948fac25ad7b5bd168e28caf5b2a329056bfd`), 2026-11-08T15:08:50Z, from L2a-01 alone (`studies/next/n6/banks/L2a-01/`, `i-00e9ff97327126cd4`, `PG_GIT_SHA` `d5554e2ad921`; §6.2, §14 A5): `studies/next/n6/thresholds.json` sha256 `7dbdd22881089bc29f92a062c68c86cb5e86bcb68ae0d71ef1b6d1b9b59a8c95`. F_eps 3.365e-9, τ_eps 3.365e-8, τ_s 1.5096e-6; G-RES passes (3.365e-9 ≤ 4.876e-9); **G-POWER fails** (τ_s 1.5096e-6 against 0.5 × median \|D_col\| = 6.363e-7; ratio 1.186). L2a's gates: every gate passes except **G-PC** (col > rand in 2 of 8; the swapped labels in 6 of 8, so its failing input passed); G-DOSE 8 of 8. Under §5 the verdict reads UNDECIDED_NO_POWER for H_age, H_adj and H_eff with no test pair read, and DOES NOT COUNT because G-PC failed; the verdict and L2b are the head's call. DISCOVERY numbers and the test-half prediction in `studies/next/n6/JOURNAL.md` (2026-11-08T15:06Z–15:08Z). |
| parents | `studies/next/09_PLAN.md` card J11; `studies/next/06_CRITIC.md` N6 (lines 642–684); `studies/next/03_IDEAS.md` N6 (line 116); `studies/next/05_SCORING.md` line 133; `studies/next/10_RECOUNT2.md` (J1); `studies/next/12_SHAPING_NOTE.md` §4 item 4 |
| method | repo `CLAUDE.md`; the `research-spec` skill; the house style of `M1_SPEC.md`, `M2_SPEC.md`, `N12_SPEC.md`; M2's amendment A5 for the offline gate |
| vocabulary | `studies/next/README.md`. A run whose gate fails is labelled **DOES NOT COUNT**. |

### 0.1 The brief's requirements and where each sits

| requirement (head's brief, 2026-11-06) | where |
|---|---|
| at equal edit size and matched energy, does an early-born pole cost more corpus loss than a late-born one | §1 |
| pairs in one layer matched on max\|a\|, density and column norm; \|ρ(age, ln max\|a\|)\| below a frozen value, shown from the banks | §2.3; `pairs.json`; §9.3 |
| edits of equal Frobenius norm, realized after rounding, asserted within a tolerance | §2.2 (G-NORM, G-EPS) |
| positive control: each layer's highest-energy pole beats the random-direction edit | §3.3 (G-PC) |
| random-direction edit of the same norm as the floor | §2.2, §3.4 |
| efficacy readout on the pole's own exemplars | §2.2, §5 (H_eff) |
| known number: E1's banked zero-replacement value reproduced first | §3.1 (K1) |
| the identity pair and a zero-norm edit | §3.2 (G-ID, G-ZN) |
| hypotheses with refuting observations; H_0 | §1.3 |
| thresholds frozen on control data; freeze half and test half | §6 |
| floors, ceiling, does-not-count branches | §5, §6 |
| the ten raw examples first, at zero GPU where the banks allow | §10; `studies/next/n6/samples/` |
| lessons (1)–(5) of the brief | §2.2 (fp32, the floor), §2.2 (the written direction), §8 (no Blackwell point), §9.3 (power at T_15), §2.2 (realized norm) |
| budget, run counts, park order, stop rule | §9.6 |
| the three dimensions | §11 |

### 0.2 Facts read for this draft

| fact | value | source |
|---|---|---|
| model, point | `allenai/Olmo-3-1025-7B`, `stage1-step1413814` (T_15), snapshot `373bad25…`; 32 layers, width 11,008, hidden 4,096, vocabulary 100,278; post-feedforward RMSNorm | `studies/next/m2/offline_gate/hf/config_stage1-step1413814.json`; S3 T_15 grid `manifest.json` |
| dated poles | 600 (sample 0), 16–20 per layer; main-cluster birth T_1…T_15; dates equal to the banked ones on 2,454 of 2,454 clusters | `studies/next/10_recount2/out/births_all.csv`, `02_counts.json`, `03_compare.json` |
| the random-group reference | R1t mean date per cluster; real date against it Pearson 0.797 | `05_per_cluster_R1t.csv`, `06_predictability.json` |
| age against T_15 statistics | within-layer Spearman with ln max\|a\| −0.592, ln density +0.502 (the critic's); recomputed here −0.5915, +0.5018 | `06_CRITIC.md` N6; dry run A1, A2 |
| adjusted age (age − R1t reference) against ln max\|a\| | within-layer Spearman +0.00003 (574 poles with a reference), against age +0.452 | `pairs.json` → `adjusted_age` |
| T_15 corpus NLL, first 120,000 targets | 2.675372037760417 in the grid harvest and in the 50M harvest (two runs, equal to the last digit); T_14 2.714761865234375 | S3 `w3/grid/1/out/t15_step1413814/controls.json`, `w1/o7_t15/1/out/o7_t15/controls.json`, `w3/grid/1b/out/t14_step1000000/controls.json` |
| E1's zero-replacement value | planted unit PL05_00, side J, the own set's 50 upstream neurons written to 0 at every window position: job_fall 0.6742178218025651 over 12 contexts; the cross arm −0.0023430985169776305 | `studies/instruments/results/e1/1/t15/samples/e1.jsonl` (sha256 `3967eda5…`); `studies/next/n6/inputs/e1_known.json` |
| E1's run | L40S, torch 2.12.0+cu130, transformers 5.9.0, bf16, git `f676b7a6767c`; 28.7M tokens forwarded in 1,958 s after load | `studies/instruments/results/e1/1/t15/metrics.json` |
| degenerate poles at T_15 | none in the 32 layer files (every pole has tokens of its own sign) | `studies/next/m2/offline_gate/bank_scan.json` → `points.15` |
| bf16 rounding noise of a window-summed loss | median \|ΔL\| 0.65 nats against a first-order 0.022; grows with the number of later loss terms | `studies/next/n12/READ_n12.md` §6 |
| corpus bands | band 3 = OLMo-3 tokens 20,570,743 – 50,540,791 (documents 14,106–34,789), ordered by directory; the grid harvest scored windows 0–10,043 | S3 corpus `manifest.json` → `bands`; T_15 `manifest.json` → `n_windows` |
| download, load | 14.61 GB in 157 s on E1's run; load 125 s | E1 `metrics.json` → `download`, `load_s` |

### 0.3 Facts to verify at run time

| # | fact | verified by | rule if it differs |
|---|---|---|---|
| 1 | the AMI has torch 2.12.0+cu130 and transformers 5.9.0; the GPU is an L40S | G-SAME | the job stops in seconds |
| 2 | the bf16 forward reproduces E1's value and the harvest's NLL | K1, K2 | the job stops |
| 3 | the model in fp32 (29.2 GB) and two kept residuals (6.4 GB) fit in 46 GB with batches of 4 windows (A4: at L2 the kept residual of 1,024 windows, 34.4 GB, is in host memory and the GPU holds two residuals of at most 96 windows; the job checks host memory at start) | L1 (peak 36.201 GiB); L2's `measured.peak_mem_gib`, `host_mem` | out of memory is not caught; precision and settings are not changed to fit; the shortfall goes to the head as a lane decision (the head's note of 16:00Z, journal) |
| 4 | the fp32 rate is 25–60 TFLOP/s | L1 `measured.sec_per_edit_layer_unit` | L2's budget is set from it at freeze 1b |
| 5 | the fp32 forward is bitwise repeatable on the instance (G-ID, one path twice); G-ZN and G-CACHE compare two paths and read the tolerances of amendment A2 | G-ID, G-CACHE, G-ZN | the labels do not count |
| 6 | the bf16 rounding probe moves the corpus loss by more than 1e-6 nats per token | G-EPS's failing input | the labels do not count (the floor has no measured failing side) |

### 0.4 Already-done audit

| cell | prior measurement | path | what it gives N6 |
|---|---|---|---|
| the birth dates and their reference | J1: dates reproduce; b below the random-group reference; 153 later than chance | `studies/next/10_RECOUNT2.md` | the age (main-cluster birth) and the adjusted age |
| age against activation statistics | ρ −0.592 (ln max\|a\|), +0.502 (ln density) on 600 poles | `06_CRITIC.md` N6, `crit06/age_energy_out.txt` (scratchpad) | the reason for matching; reproduced by the dry run |
| removal cost and energy | wdd-descendants e733: a Fisher price that is "nearly the activation energy" (0.94–0.97 over 4,096 rows), Pythia-410m | `04_LITERATURE.md` N6 (REPO-DOC) | energy is the matched quantity |
| single-neuron loss collateral | NEURONOFF: Arm A cancelled (the positive control failed on 37–42 % of poles); under the cancelled label median D −0.0001 nats (Q4B, n 26) | `studies/neuronoff/results/RESULTS.md` | single-pole effects sit near the noise of a bf16 per-prompt readout; N6 reads corpus loss in fp32 over 196,608 targets |
| the zero-replacement operator | E1: zero written into a set's down-projection inputs at every window position, BOS left; the zero arm's two-neuron chance level 0.0074 | `studies/instruments/results/e1/verdict_e1.json` → `reported.A1.zero__e` | the operator N6 restricts to one pole's sign; K1 |
| bf16 loss resolution | N12 G-VAL: a window-summed bf16 loss cannot resolve a 0.02-nat effect | `studies/next/n12/READ_n12.md` §6, `READ_head.md` addendum | fp32 forward; the rounding probe |
| the carrier and competition | M2, N12: a pole's late cluster is often contested by other poles; the carrier wins by its own rise | `m2/READ_head.md`, `n12/READ_head.md` | no direct bearing on collateral; stated as context only |

Nothing in the repository has measured a corpus-loss change of a single-pole edit on O7 at T_15.

### 0.5 Objects, defined once

- **Pole** (l, n, s): sign s of neuron n in layer l; its activation a_n(x) = silu(g_n(x)) u_n(x) at token x; the pole fires at x when
  s a_n(x) > 0.
- **Age** of a dated pole: the birth index k of its main cluster (T_k, k = 1…15) under the D1 percentile reading, from J1's recount
  (`births_all.csv`, sample 0). Main = the non-echo cluster with the largest max\|a\| at T_15. Early-born = smaller k.
- **Adjusted age**: age minus the R1t reference mean date of the main cluster (random same-size groups of the pole's exemplars holding
  the top exemplar, J1 §7); 0 for the 26 one-cluster poles. It removes the part of the date that the pole's activation profile sets.
- **Column** w = W_down[:, n] (fp32 from the bf16 checkpoint, exact). ‖w‖ its Euclidean norm.
- **Edit** of pole (l, n, s) with vector δ ∈ R^4096: at every eval position x ≥ 1 (BOS left, as E1 leaves it) where s a_n(x) > 0, the
  down projection's output gains a_n(x) δ; nowhere else. This is a rank-one change of column n applied on the pole's own tokens. Its
  edit size is c = ‖δ‖ (the Frobenius norm of the rank-one change). A whole-column weight edit would also change the neuron's other
  pole, whose age differs; restricting the change to the pole's sign keeps the edited object equal to the dated object.
  - **col**: δ = −(c/‖w‖) w, the written direction (12_SHAPING_NOTE §4 item 4). κ = c/‖w‖ is the fraction of the pole's write removed.
    At κ = 1 it equals E1's zero replacement restricted to the pole's sign (dry run, max logit difference 4.8e-6 in fp32).
  - **rand**: δ = c r, r a unit vector drawn by a seed from the edit's name.
  - **col_half** (positive control only): δ = −0.5 w.
  - **probe**: rand's δ added and then subtracted in the same hook; zero effect in exact arithmetic, so its measured loss change is the
    rounding floor.
  - **zero**: δ = 0.
- **Pair edit size**: c = min(‖w_early‖, ‖w_late‖), the same for all four edits of a pair. The smaller-norm member is removed fully
  (κ = 1), the other by κ = c/‖w‖ < 1.
- **E⁺** of a pole: Σ_x (s a_n(x))₊² over the eval positions x ≥ 1, fp32. Under an equal-size edit the energy added to the down
  projection's output is c² E⁺, so E⁺ is the energy the card asks to match.
- **Eval set**: W harvest-aligned windows of 2,048 corpus tokens wholly in band 3, spread evenly over it (window indices 10,045 +
  ⌊i · 14,633 / W⌋, i = 0 … W − 1, ascending), each with BOS first. L1: W = 96, 196,608 targets. L2 (amendment A4, §14): W = 1,024,
  2,097,152 targets, windows 10,045 to 24,663 (`n6.eval_windows(1024)`; sha256 of the JSON index list
  `5822191728d49aba45817f8479058c266bcaaa057dc1037fa0d5509507d1b223`); 32 of L1's 96 windows are among them. Band 3 is outside the 20M
  corpus on which the ages, max\|a\| and density were measured.
- **Corpus loss change** ΔL(edit) = (1/(2,048 W)) Σ [ℓ_edit(x) − ℓ_base(x)] over the eval targets, ℓ the next-token cross-entropy, fp32 forward, fp32 logits and
  loss, nats per token. Per-window values are banked for the window bootstrap.
- **Eval exemplars** of a pole: its 50 eval positions (1 ≤ j ≤ 2,047) with the largest s a_n.
- **Efficacy** KL_eff: mean over the 50 eval exemplars of KL(p_base ‖ p_edit) of the next-token distribution at the exemplar, nats.
- **Pair contrast**: D_col = ΔL_col(early) − ΔL_col(late); D_rand the same for rand; Q = ΔL_col / KL_eff per pole.
- **Positive-control pole** of layer l: the pole with the largest E⁺ among the layer's 22,016, chosen in base pass 1 before any edit.
  Amendment A4: the positive-control edits run in the 8 layers 0, 4, …, 28; the poles of layers 0 and 31 also carry two rounding probes.

---

## 1. Question, decision, hypotheses

### 1.1 The question

At equal edit size and matched activation energy, does an edit along an early-born pole's written direction cost more corpus loss
than the same-size edit along a late-born pole's, within one layer?

### 1.2 The decision this run makes

| outcome | what is done next |
|---|---|
| H_age SUPPORTED (the cost is in the written direction) | age enters the editor's rule (§11, practical); N28 (forgetting from weights) gets a unit-level predictor to test; the birth dates gain a use beyond description |
| SUPPORTED_UNSPECIFIC | the early poles' tokens are where any perturbation costs more; the follow-up is about token placement, not readers; the age rule is a proxy for it |
| FLAT (with power) | age carries no information about collateral beyond max\|a\|, density, norm and energy; N6 closes and the "readers accumulate with age" premise is set aside for editing |
| REVERSED | late-born poles cost more; the premise is wrong in sign and the editor's rule reads "edit the old" |
| UNDECIDED / no power | a larger eval set or more pairs (a second sample of dated poles) is the only next step; no claim |

2026-11-08: the realized outcome is UNDECIDED / no power (G-POWER failed at freeze 2), with every label DOES NOT COUNT because G-PC
failed and its failing input passed (`verdict_n6.json`). The registered next step is the row above (a larger eval set or more pairs). The
head is not taking it now: it has lower leverage than the rank test of the RL delta.

### 1.3 Hypotheses and the observation that refutes each

| hypothesis | claim | refuting observation (test half) |
|---|---|---|
| **H_age** (the card) | the early-born member's column edit costs more corpus loss: p_col > 0.5 | the Wilson 95% interval of p_col lies inside [0.30, 0.70] with half the test pairs resolved (FLAT), or below 0.5 (REVERSED) |
| **H_0** (the boring one) | age carries no information beyond the matched quantities: p_col = 0.5 | the interval of p_col lies wholly above or wholly below 0.5 |
| **H_dir** (the written direction) | the early-born cost is specific to the direction the pole writes | p_col − p_rand has its bootstrap interval at or below 0 while p_col's lower bound is above 0.5 (label SUPPORTED_UNSPECIFIC) |
| **H_less** (the young costs less because it does less) | the early pole's edit also does more on its own exemplars, so its collateral per unit efficacy is not larger | refuted if H_eff is SUPPORTED: the early member's Q = ΔL_col / KL_eff is larger in a share of pairs whose lower bound exceeds 0.5 |
| **H_adj** (age beyond the reference) | the same as H_age on the pairs whose adjusted ages differ in the same direction by at least 1 grid point | as H_age, on those pairs |
| **energy** (the critic's confound) | a residual energy difference orders the cost | the balance gates (§4) bound it before L2; D_rand measures it on the same tokens |

p_col is the share of test pairs with D_col > τ_eps, ties (|D_col| ≤ τ_eps) counted one half. H_age and H_0 predict opposite outcomes on
this one readout.

### 1.4 The author's prior, logged before any N6 number exists (2026-11-06)

H_age SUPPORTED 0.25; SUPPORTED_UNSPECIFIC 0.15; FLAT 0.25; REVERSED 0.05; UNDECIDED (including no power) 0.30. H_eff SUPPORTED
given H_age SUPPORTED: 0.35. Reasons: the energy confound is matched away, and the literature gives no measured age effect at a fixed
checkpoint; the 35 test pairs give a Wilson half-width of about 0.16, so only p_col ≥ 0.69 reads SUPPORTED.

### 1.5 Literature (query terms recorded)

Searched 2026-11-06 under the field's names for the object (MLP neuron, down-projection column), the method (ablation, rank-one
edit, matched random direction) and the effect (collateral, perplexity, by training time):

1. `ablation cost of neurons by time of emergence during pretraining "early-learned" features more important language model checkpoints`
2. `single neuron ablation loss increase predicted by activation energy versus training step neuron formed OLMo Pythia "ablation" "checkpoints" 2026`
3. `"rank-one" edit down projection column equal Frobenius norm random direction control collateral perplexity MLP neuron editing`
4. The seven queries of `04_LITERATURE.md` N6 (verdict OPEN) are not repeated.

Read in full for the specific claim: "Negative pre-activations differentiate syntax"
(https://arxiv.org/abs/2509.24198, HTML v1): the checkpoint ablations follow one cohort selected at the final checkpoint and relate the
cohort's mean Wasserstein distance to grammatical error ("as Wasserstein neurons' distributions become more non-Gaussian, ablation
yields larger grammatical damage", §6); its controls are matched by perplexity impact and rank, not by activation statistics, energy
or norms; it assigns no per-neuron age. Seen in search results only, not read, no claim taken: arXiv 2311.00863 (contextual n-grams in
Pythia), 2508.00903 (universal neurons in GPT-2), 2509.17196 (concept evolution in Pythia). Verdict unchanged: OPEN. No work found
conditions a unit's edit or ablation cost on that unit's training age at a fixed checkpoint, matched on activation statistics.

---

## 2. The measurement and the instrument

### 2.1 In plain words

For each matched pair, edit the early pole and, separately, the late pole by the same amount along the direction each writes, and
measure how much the mean next-token loss on 2,097,152 held-out corpus tokens (1,024 windows, amendment A4) rises, in nats per token.
Do the same with a random direction of the same size. Read, over the test pairs (24 after amendment A3), how often the early pole's
edit costs more.

### 2.2 The instrument, exactly

- **Forward.** The checkpoint loaded as E1 and the harvest loaded it (`neurons_harvest.load_model`, bf16, sdpa); the bf16 phase runs
  K1, K2, G-POLE and the bf16 rounding probes; then `model.float()`, `torch.set_float32_matmul_precision("highest")`, TF32 off (G-FP32).
  Every corpus loss after that is an fp32 forward with fp32 logits and fp32 cross-entropy. Lesson (1): the bf16 window-summed loss is
  rounding noise at the 0.5-nat level (N12); the fp32 forward's floor is measured by the probes (G-EPS) and must lie below 1/50 of the
  10th percentile of \|ΔL_col\| over the freeze half (G-RES; 1/100 at freeze 1, amendment A4).
- **The progression** (`n6_edit.Job.progression`). The eval windows' residual h_l (fp32) is kept in host memory: 3.2 GB for L1's 96
  windows, 34.4 GB for L2's 1,024 (amendment A4). At layer l the windows go to the GPU in chunks of 96 (`--chunk 96`); for each chunk
  h_(l+1) = layer_l(h_l) is computed once; each edit of layer l runs layer l with the hook on `layers[l].mlp.down_proj`, then layers
  l+1…31, the final norm and the head, from the chunk's h_l; then the chunk's h_(l+1) replaces its h_l in host memory. Every window is
  computed in the same batches of 4 as without chunks, so a record does not depend on the chunk size (dry run B A4). The pieces are called as `Olmo3Model.forward` calls them (the masks by `config.layer_types[i]`, the rotary
  embedding from the residual); from layer 0 this equals the model's forward bit for bit (dry run), and at the end the base losses
  recomputed from h_32 must equal base pass 1 bitwise (G-CACHE).
- **The hook** (`n6_edit.Edit`): forward hook on the down projection; out + 1[s a_n > 0, position ≥ 1] a_n δ, the addition in fp32. It
  records Σ coef² (the pole's E⁺ as it sees it) and Σ ‖added‖².
- **Realized size** (lesson 5). ‖δ‖ is asserted within 1e-6 of c when the edit is built; the hook's E⁺ must equal base pass 1's E⁺ of
  the pole within 1e-4 (G-NORM: the change landed on the pole's own neuron, sign and tokens); the realized change of the residual
  after layer l, R = ‖h_(l+1),edit − h_(l+1),base‖_F, is banked per edit. In fp32 a scaled change does not round back (N13 A30 is a
  bf16 effect); stochastic rounding is not needed.
- **Read on the input side separately** (lesson 2). The edit acts on the output column only; which tokens are the pole's is read from
  its own activation (the input side), which the edit does not change at layer l. G-POLE checks the pole's identity on the input side
  against the bank.
- **Efficacy.** KL_eff at the pole's 50 eval exemplars, from the base log-probabilities banked in base pass 2 and the edited logits of
  the same forward; Δℓ at those exemplars banked beside it.
- **Per edit record** (`edits.jsonl`): edit, layer, role, kind, pole, n, sign, c, ‖w‖, κ, ‖δ‖, ΔL, the per-window ΔL (96 in L1, 1,024 in L2), R, E⁺ (hook and
  pass 1), KL_eff, Δℓ at the exemplars, the five tokens with the largest \|Δℓ\| (window, target position, Δℓ, base loss), bitwise
  equality with base, seconds.

### 2.3 The pairing (frozen; `n6.build_pairs`, `n6.final_pairs`)

- **Candidates**: two dated poles of the same layer and different neurons whose main-cluster births differ by at least 3 grid points.
- **Stage 1 (now, banks only).** Calipers on \|Δ ln max\|a\|\| from {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} and \|Δ ln density\| from
  {0.1, …, 0.5}; greedy matching by d = Σ \|Δx\| / caliper_x, each pole in at most one pair; the grid point with the most pairs whose
  matched set is balanced: \|ρ_pair(age, x)\| ≤ 0.10 and \|SMD_x\| ≤ 0.10 for x = ln max\|a\|, ln density; ties to the smaller calipers.
  Chosen: 0.10 and 0.50, **74 pairs** (39 freeze, 35 test; 29 layers). These are L1's pairs.
- **Stage 2 (after L1, before L2).** The stage-1 calipers stay; calipers on \|Δ ln ‖w‖\| from {0.05, 0.10, 0.15, 0.20, 0.30} and on
  \|Δ ln E⁺\| from {0.25, 0.50, 0.75, 1.00} (L1's arrays) are added and chosen by the same rule; balance also needs \|SMD\| ≤ 0.10 on
  ln ‖w‖ and ln E⁺. All candidates are matched again. If no grid point is balanced, or fewer than 40 pairs result, **L2 does not
  launch** (§9.5). On a planted set of L1 arrays (a simulation, not data: column norms lognormal with sd 0.15, E⁺ a function of the
  bank statistics with noise sd 0.3) stage 2 kept 67 pairs (dry run A13); a planted age-dependent column norm left no balanced point
  and stopped (A14).
- **ρ_pair** is the critic's within-layer Spearman with each pair as its stratum: ranks within the pair, centred, pooled. It equals
  1 − 2f, f the share of pairs whose earlier member has the larger value. The bound 0.10 means age explains at most 1 % of the
  within-pair rank variance of ln max\|a\| (ρ² ≤ 0.01). **SMD** = mean over pairs of (early − late) / the within-layer sd over all 600
  dated poles.
- **The pooled within-layer Spearman** on the 148 matched poles is −0.261 (reported, not gated): pairs at a high max\|a\| have both
  members earlier than pairs at a low one, and the paired contrast does not use variation between pairs.
- **The split**: freeze when the early member's neuron index is even.

| balance of the stage-1 set (`pairs.json` → `balance`) | ln max\|a\| | ln density |
|---|---|---|
| ρ_pair (bound 0.10) | +0.027 | +0.027 |
| SMD (bound 0.10) | +0.005 | −0.004 |
| mean \|Δ\| within pairs | 0.041 | 0.108 |
| within-layer Spearman, 600 unmatched poles | −0.592 | +0.502 |
| failing input: 176 unmatched pairs (same layer, \|Δ age\| ≥ 3, no caliper), ρ_pair and SMD | −0.739, +1.488 (fails) | +0.659, −1.135 (fails) |

Ages in the stage-1 set: the early member is 4.62 grid points earlier on average (3 to 11; from `pairs.json` → `pairs`); its adjusted
age is 2.09 points lower (`balance.adj_difference`); 47 of 74 pairs (23 of 35 test) are concordant on the adjusted age by at least 1 point.

### 2.4 Settings identical to the banked runs

Model loader, dtype (bf16), sdpa, BOS 100257, harvest window 2,048, 120,000-target NLL rule, windows in corpus order: as the harvest
(K2). E1's readout (`reuse_ablate.readout`, length 256, batch 32, `ablation_dict(..., "zero")`): unchanged (K1). The fp32 phase has
no banked counterpart; K3 compares it with K2 on the same targets.

### 2.5 What runs at zero GPU

The ages, the matching quantities from the bank, the pairs, the balance, the split, the eval windows, the edit plan, the projection,
the zero-GPU half of the raw examples, both sides of the bank-held gates. Not at zero GPU: the column norms, E⁺, every loss, every edit.
They are L1, the first GPU step.

---

## 3. Controls and references

### 3.1 The known numbers

- **K1** (the card's): E1's banked zero-replacement value. The planted unit PL05_00 (layer 5, neurons 3025 and 5757, λ 7.9326), side J,
  zero written into the own set's 50 upstream neurons at every window position, BOS left; job_fall = mean over the 12 contexts of
  (a_before − a_after)/a_before. Pass: \|value − 0.6742178218025651\| ≤ 1e-3. Failing input: the banked cross-arm value −0.0023431 as the
  value (677 times the tolerance away). Same device class, torch and transformers as E1's run (G-SAME), so bit-identity is expected.
- **K2**: corpus NLL on the first 120,000 targets, bf16, the harvest's rule: \|value − 2.675372037760417\| ≤ 1e-3 (two banked harvests
  give this value to the last digit). Failing input: T_14's 2.714761865234375 (39 times the tolerance away).
- **K3**: the same targets in fp32: \|K3 − K2\| ≤ 0.02. Failing input: the fp32 NLL with targets one position off (input j predicting
  j + 2), which must lie more than 0.02 away (tiny world: 7.98 against 2.22).

### 3.2 The identity pair and the zero-norm edit

- **G-ID**: base pass 1 and base pass 2 (the same weights scored twice, the same code path) give bitwise equal per-token losses.
  Failing input: the first column edit's losses compared the same way (must differ).
- **G-ZN** (amended by A2, §14: \|ΔL\| ≤ 1.221e-7 nats per token, the two-path tolerance; bitwise at freeze 1): the zero-norm edit (δ = 0, the hook installed and run) gives losses bitwise equal to base, ΔL exactly 0. Failing input: the
  first column edit (must differ).
- **G-CACHE**: the progression's final residual gives the base losses bitwise. Failing input (dry run): a forward that skips layer 0.
  Amended by A2 (§14): \|mean per-token difference\| ≤ 1.221e-7 and largest per-token difference ≤ 1e-5 nats; failing input in the job:
  each window scored from another window's final residual.

### 3.3 The positive control

G-PC (amendment A4, §14): in the 8 positive-control layers 0, 4, …, 28, the layer's highest-E⁺ pole, col at κ = 1 against rand at the
same c: ΔL_col > ΔL_rand + τ_eps in at least 6 of 8 (P(X ≥ 6 \| n = 8, p = 0.5) = 37/256 = 0.145). G-DOSE: col at κ = 1 against
col_half: the full edit costs more in at least 6 of 8. Failing inputs: the labels swapped (the count of the reverse ordering must stay
below 6). At freeze 1: all 32 layers, 24 of 32 (0.0035). The PC is also the ceiling of H_age (§5).

### 3.4 The floors

- **Rounding floor** F_eps = max \|ΔL_probe\| over four fp32 probes (two pair poles, the positive controls of layers 0 and 31). Bound
  1e-6 nats per token (G-EPS). Failing input: the same two pair-pole probes in the bf16 model before the cast must exceed 1e-6. The tie
  margin is τ_eps = 10 F_eps.
- **G-RES** (at freeze 2): F_eps ≤ min(1e-6, 0.02 × the 10th percentile of \|ΔL_col\| over the freeze half's pair edits): the factor
  50, set by amendment A4 after L1-02's preview (0.01, the factor 100, at freeze 1).
- **G-POWER** (at freeze 2, amendment A4): τ_s ≤ 0.5 × the median \|D_col\| over the freeze pairs (§6.2); if not, UNDECIDED_NO_POWER
  before the test half is read (§5).
- **The matched floor of the contrast**: the random-direction edit of the same pole, same c, same tokens. p_rand is the age statistic
  computed on it. Under H_0 and balance both p_col and p_rand sit at 0.5; a p_rand away from 0.5 is a residual imbalance in where and how
  strongly the poles fire.
- **The exact floor of p_col**: 0.5, from the within-pair exchangeability of the early and late labels under H_0.

### 3.5 Monitoring level and the read/edit rule

The edit is along the written direction (the down-projection column); detection of the pole's tokens is by its own activation. This
follows `12_SHAPING_NOTE.md` §4 item 4, whose mapping to MLP neurons is ours, not the paper's.

---

## 4. Gates

| gate | where | passes when | registered failing input | identity |
|---|---|---|---|---|
| G-SAME | job, first | torch 2.12.0+cu130, transformers 5.9.0, NVIDIA L40S | each version string with a prefix | — |
| G-REV | job | snapshot 373bad25… | forty zeros | — |
| K1 | job, bf16 | §3.1 | the cross arm's banked value | — |
| K2 | job, bf16 | §3.1 | T_14's banked value | — |
| G-POLE | job, bf16 | every edited dated pole's first 10 banked T_15 exemplar activations recomputed (full harvest-window prefixes, a = act(g) u in fp32) within 5e-2 of max\|a\| (amendment A1, §14; 1e-2 at freeze 1); g, u and a per exemplar written to `g_pole.npz` | neuron n + 1 at the same positions, same forward (L1-01: smallest 0.923); offline the T_14 values (min 0.150, 3.0 times the tolerance 0.05, over 197 poles) | — |
| G-FP32 | job | every parameter fp32, precision "highest", TF32 off | — (dry run plants a bf16 parameter) | — |
| K3 | job, fp32 | §3.1 | targets one position off | — |
| G-BAL (bank) | offline, freeze 1 | §2.3 | 176 unmatched pairs | — |
| G-BAL (stage 2) | offline, freeze 1b | §2.3 on four quantities, ≥ 40 pairs | a planted age-dependent column norm (A14) | — |
| G-INST | job, L2a and L2b | this instance's ‖w‖ and E⁺ of every paired pole equal L1's (the values in `pairs_L2.json`) within 1e-5; E⁺ from a pass over L1's 96 windows (`inst_windows`, amendment A4) | each pole against its partner's values (dry run A4: E⁺ read on L2's eval set stops the job) | — |
| G-ID, G-ZN, G-CACHE | job | §3.2 | §3.2 | the identity pair itself |
| G-NORM | job, every edit | ‖δ‖ within 1e-6 of c; the hook's E⁺ within 1e-4 of pass 1's | the neighbouring neuron's E⁺ (dry run: 71 against 849) | the zero edit (E⁺ read, nothing added) |
| G-EPS | job | §3.4 | the bf16 probes | the probe is itself an identity in exact arithmetic |
| G-PC, G-DOSE | job, L2a (A5) | §3.3 | labels swapped | — |
| G-SAMPLES | job, L1 and L2a | the ten raw examples written | — | — |
| L2b identity (A5) | job L2b, before its first gate; `n6.read_banks` | L2a's `metrics.json` is a complete freeze-half launch with nothing parked and the same eval windows, eval tokens and pairs content (sha256) | (dry run: an L2b whose eval set differs stops before G-SAME; the verdict given such a bank stops; an L2b against an L2a that parked stops) | — |
| G-RES | freeze 2 | §3.4 (factor 50, A4) | (dry run: F_eps 5e-5 fails; A4: a floor at 0.05 × q10 fails, 0.015 × q10 passes) | — |
| G-POWER | freeze 2 (A4), from L2a | §6.2: τ_s ≤ 0.5 × median \|D_col\| over the freeze pairs | (dry run: sd at 0.6 × median \|D_col\| fails; 0.4 passes) | — |
| G-PARK | verdict | parked test pairs ≤ 0.2 of the test half | (dry run: 8 of 35 fails) | — |

A gate the job runs before the edits stops the job when it fails or when its failing input passes (`Job.gate`). Gates evaluated after
the edits (G-ID's failing side, G-ZN, G-CACHE, G-EPS, G-NORM, G-PC, G-DOSE) are written to `metrics.json` and read by the verdict: any
failure, or a failing input that passed, makes every label DOES NOT COUNT, with the realized label kept beside it. G-POWER failing
(amendment A4) makes every label UNDECIDED_NO_POWER without reading the test half (§5). After amendment A5 the verdict reads the gates of
both launches, L2b's under the prefix `L2b:`; L2b re-runs every gate that does not depend on the half (G-SAME, G-REV, K1, K2, G-POLE,
G-FP32, K3, G-INST, G-ID, G-ZN, G-CACHE, G-EPS on its two pair probes, G-NORM); G-PC, G-DOSE and G-SAMPLES are L2a's.

---

## 5. Labels and the verdict, as the script runs it (`n6.verdict`)

First (amendment A4): if `thresholds.json` → `G_POWER` failed, the script writes H_age, H_adj and H_eff as UNDECIDED_NO_POWER (DOES NOT
COUNT, with UNDECIDED_NO_POWER kept as the realized label, if a gate also failed), reads no test pair and stops; no test label is issued.
Otherwise, on the test half, with τ_eps and τ_s from `thresholds.json`, ceiling = the upper Wilson bound of the share of the 8 PC layers
(amendment A4; 32 at freeze 1) in which col beat rand:

1. p_col = (wins + ½ ties)/n over test pairs, D_col against τ_eps; Wilson 95% [lo, hi]. p_rand likewise on D_rand. Δp = p_col − p_rand
   with a percentile bootstrap over pairs (10,000, seed 0). R = share of test pairs with \|D_col\| > 2 τ_s.
2. **H_age**: lo > ceiling → ABOVE_CEILING (an age ordering more reliable than the instrument's ordering of a known pair of edits is
   read as a defect until inspected); else lo > 0.5 and lo(Δp) > 0 → SUPPORTED; lo > 0.5 and lo(Δp) ≤ 0 → SUPPORTED_UNSPECIFIC;
   hi < 0.5 → REVERSED; lo ≥ 0.30 and hi ≤ 0.70 → FLAT if R ≥ 0.5, else UNDECIDED_NO_POWER; otherwise UNDECIDED.
3. **H_adj**: the same rule on the concordant test pairs (adjusted age at least 1 point lower for the early member), without the Δp
   condition.
4. **H_eff**: the same rule on Q_early − Q_late (ties at 0; pairs with KL_eff = 0 left out and counted); R is not applied.
5. **Practical**: the cost saved by editing the later-born member of each pair, against a random pick (mean D_col / 2) and against the
   energy rule (edit the member with the smaller E⁺), means over test pairs with bootstrap intervals.
6. Any failed gate, failing input that passed (in L2a or L2b, amendment A5), G-RES failed, or parked test pairs above 0.2 of the
   half: every label DOES NOT COUNT.
7. Headline: `H_age <label>; H_adj <label>; H_eff <label>`.

Every branch was planted in the dry run (C checks): SUPPORTED, SUPPORTED_UNSPECIFIC, REVERSED, FLAT, UNDECIDED_NO_POWER, UNDECIDED,
ABOVE_CEILING, DOES NOT COUNT from a failed gate and from a failing input that passed, G-RES, parking; after A4, G-POWER on both
sides (0.4 passes and the test half is read; 0.6 gives UNDECIDED_NO_POWER with no test pair read, also with every test record removed).

---

## 6. The split, freeze 2 and room

### 6.1 The split

Stage 2 by amendment A3 (§14): the halves are swapped (test when the early member's neuron index is even) and the four pairs whose
outcome L1 banked stay in the freeze half: 24 test pairs, 21 freeze pairs. As registered at freeze 1:
Freeze half: pairs whose early member has an even neuron index (39 in stage 1). Test half: odd (35). The same rule on the stage-2
pairs. The verdict script reads only test pairs; `n6.freeze` reads only freeze pairs (dry run C: a planted test-half effect does not move
the freeze median).

### 6.2 Freeze 2 (`n6.freeze`, from the freeze half and the gates only)

Amendment A5: freeze 2 reads L2a's bank alone (`n6 freeze --bank studies/next/n6/banks/L2a-01`); L2b does not exist yet.

- τ_eps = 10 F_eps (control data: the probes).
- τ_s = the median over freeze pairs of the window-bootstrap sd of D_col (1,000 resamples of the eval windows; 1,024 at L2, amendment
  A4): the eval-set sampling noise of one pair's contrast. Used in R (the power of FLAT) and in G-POWER.
- **G-POWER** (amendment A4): τ_s ≤ 0.5 × the median over the freeze pairs of \|D_col\|. If not, the verdict declares
  UNDECIDED_NO_POWER before the test half is read, and no test label is issued (§5). Planted in the dry run: 0.4 passes, 0.6 fails. On
  L1-02's five pairs at 96 windows the ratio is 1.42 (fails); scaled by √(96/1,024) with \|D_col\| held at its 96-window value, 0.43
  (`studies/next/n6/diag/L1-02/a4_preview.json`).
- G-RES's bound from the freeze half's \|ΔL_col\|.
- DISCOVERY numbers written beside them (p_col, p_rand, R, median D_col on the freeze half) and the prediction for the test half,
  copied into `studies/next/n6/JOURNAL.md` before the verdict runs.

### 6.3 Room (each rule can fire at the realized n)

At 35 test pairs: p_col = 0.5 gives Wilson [0.343, 0.657], inside the FLAT band; 24 wins lift the lower bound above 0.5 (dry run C).
REVERSED needs at most 11 wins. The PC rule (24 of 32) can fail (planted 16 of 32 fails); after A4, 6 of 8 (planted 6 passes, 5
fails). G-RES can fail (planted; after A4 a floor at 0.05 × q10 fails). G-POWER can fire both ways (planted 0.4 and 0.6; the L1 preview
reads 1.42 at 96 windows and 0.43 projected at 1,024). With 8 PC layers the ceiling is 0.929, 0.978 or 1.0 at 6, 7 or 8 of 8; at 24
test pairs the largest lower bound is 0.862 (24 wins), so ABOVE_CEILING cannot fire while G-PC passes, as at 24 of 32 (0.867). The stage-2 rule
can stop L2 (planted). At the A3 split (24 test pairs, §14) Wilson at 0.5 is [0.314, 0.686], so FLAT can fire only at exactly 12 wins
(ties as halves); SUPPORTED needs 17, REVERSED at most 7; H_adj has 16 concordant test pairs (FLAT cannot fire; SUPPORTED needs 12,
REVERSED at most 4). If stage 2 leaves fewer than 35 test pairs the room is recomputed by `n6.freeze` and written at freeze 2; the
FLAT band does not change.

---

## 7. Unblinding order

1. L1 complete; the head reads `samples/raw_examples.jsonl` (§10) and the L1 gates; `studies/next/n6/JOURNAL.md` records the answers to
   the three sample questions (input as specified; output in the expected format; the scored quantity what the model produced).
2. `n6 final` → `pairs_L2.json`; freeze 1b.
3. L2a complete (amendment A5: the freeze half, the positive control, every gate); open its freeze pairs' raw rows and the PC rows;
   `n6 freeze --bank studies/next/n6/banks/L2a-01`; freeze 2 (`thresholds.json`, tag `n6-freeze-2`). If G-POWER failed, L2b is not
   launched: `n6 verdict --bank studies/next/n6/banks/L2a-01` writes UNDECIDED_NO_POWER for H_age, H_adj and H_eff, no test pair is
   read, and the study ends at L2a's cost.
4. G-POWER passed: the prediction for the test half in the journal; L2b launched (the test half; the same pairs file and eval set,
   checked against L2a's `metrics.json`); L2b complete; `n6 verdict --bank studies/next/n6/banks/L2a-01 --bank
   studies/next/n6/banks/L2b-01` once; fill §12.
5. A fresh Opus recount with its own code from both launches' `edits.jsonl`, and the head's read.

---

## 8. Threats to validity

| threat | effect | mitigation |
|---|---|---|
| age is collinear with activation statistics (ρ −0.592) | the age contrast measures energy | matched pairs (§2.3); E⁺ matched at stage 2; D_rand on the same tokens |
| the birth rule dates heavy-tailed poles early by construction | age is partly the activation profile | H_adj on the adjusted age (uncorrelated with ln max\|a\|: +0.00003) |
| column norms differ within a pair | unequal fractions removed | equal c; ‖w‖ matched at stage 2; κ banked per edit |
| a whole-column edit would touch the other pole | the dated object is not the edited one | the edit is restricted to the pole's sign (§0.5) |
| bf16 rounding noise | ΔL is noise (N12) | fp32 forward; G-EPS with a bf16 failing side; G-RES |
| eval-set sampling | D's sign varies with the windows | 1,024 windows across band 3 at L2 (A4; 96 in L1); τ_s; G-POWER; R for FLAT; bootstrap over pairs |
| the eval text overlaps the exemplars | collateral read on the pole's own contexts | band 3 is outside the 20M corpus of the dates and statistics |
| few pairs (35 test) | wide intervals | stated; UNDECIDED is a registered outcome; the room table |
| the positive-control pole is a massive-activation neuron | col and rand both catastrophic; G-PC fails without a defect | 6 of 8 leaves room for two such layers (A4; 24 of 32 left eight); ΔL banked per layer |
| cross-device banks | gate on a residual of hardware | no gate reads a Blackwell-harvested point (T_7–T_14): the ages are inputs, G-POLE and K2 read T_15 (L40S) |
| one checkpoint, one model, one sample of dated poles | scope | stated in §11; no claim beyond O7 T_15 |
| the PC is not matched to the pairs | its ordering is easier than the pairs' | it is the instrument's ceiling, not a floor of the contrast |
| two launches (A5) | L2b runs on another instance; τ_eps, τ_s and the ceiling come from L2a | each pair's contrast lies within one launch; L2b re-runs K1, K2, K3, G-POLE, G-SAME at the same tolerances, asserts L2a's eval set and pairs content (sha256), and repeats L2a's two pair probes and zero edit (their values in both launches' G-EPS and G-ZN records) |

---

## 9. Runbook

### 9.1 Checklist before L1

`N6_DRYRUN_OK`; `gate_sides.py` prints `GATE_SIDES_OK`; the head's review; commit; tag `n6-freeze-1`; `./infra/aws_pg/pg ps` and
`audit` show no stray instance; `aws s3 ls s3://pg-lineage-769063704696/results/next_n6/ --profile fuelos --region us-east-1` lists
nothing.

### 9.2 Code (NEW or REUSED, checked against the tree at `3b6e1d34d`)

| file / function | status | what |
|---|---|---|
| `infra/pg/n6.py` | NEW | dated poles, matching, balance, split, eval windows, plan, projection, freeze, verdict; CLI `pairs`, `final`, `e1`, `projection`, `freeze`, `verdict` |
| `infra/aws_pg/jobs/n6_edit.py` | NEW | the job (`Model`, `Edit`, `Job`, `pc_gates`) |
| `infra/aws_pg/jobs/n6_dryrun.py` | NEW | the offline gate's dry run |
| `infra/aws_pg/steps/next_n6_L1.json`, `next_n6_L2.json` | NEW | step lists for `jobs.pipeline` (amendment A5: `next_n6_L2a.json`, `next_n6_L2b.json` replace `next_n6_L2.json`) |
| `infra/pg/reuse_ablate.readout`, `ablation_dict`, `replaced` | REUSED unchanged | K1 and G-POLE |
| `infra/pg/unite_route.planted_unit`, `pack`, `upto_layer` | REUSED unchanged | K1, G-POLE |
| `infra/pg/neurons_harvest.load_model`, `text_tower`, `decoder` | REUSED unchanged | load, tower, decoding |
| `infra/aws_pg/jobs/neurons_harvest.download`, `fetch`, `fetch_corpus`; `infra/pg/neurons_corpus.load_tok` | REUSED unchanged | download, inputs, corpus |
| `transformers.masking_utils.create_causal_mask`, `create_sliding_window_causal_mask` (5.9.0) | REUSED | the progression's masks, as `Olmo3Model.forward` builds them |

### 9.3 The offline gate

`N6_OFFLINE_GATE.md`. Run on the laptop:
```
PYTHONPATH=. python3 -m infra.pg.n6 pairs --out studies/next/n6/pairs.json
PYTHONPATH=. python3 -m infra.pg.n6 e1 --out studies/next/n6/inputs/e1_known.json
PYTHONPATH=. python3 -m infra.pg.n6 projection --pairs studies/next/n6/pairs.json --out studies/next/n6/projection.json
PYTHONPATH=. python3 studies/next/n6/offline_gate/gate_sides.py --out studies/next/n6/offline_gate/gate_sides.json
PYTHONPATH=. <venv>/bin/python -m infra.aws_pg.jobs.n6_dryrun --out <scratch>       # torch 2.12.0, transformers 5.9.0, CPU
```

### 9.4 S3 layout

```
results/next_n6/inputs/{pairs.json, e1_known.json}                     (L1; sha256 in n6_edit.INPUT_SHA["L1"])
results/next_n6/inputs_L2/{pairs_L2.json, e1_known.json}               (L2; sha256 set at freeze 1b)
results/next_n6/<attempt>/steps.json  STATUS  job.log  heartbeat.jsonl
results/next_n6/<attempt>/out/edit/n6/{metrics.json, edits.jsonl, colnorms.npy, energies.npy, g_pole.npz}
results/next_n6/<attempt>/out/edit/n6/samples/raw_examples.jsonl
```
Attempts: `L1-01`, `L2-01`; a relaunch takes the next number, the old prefix kept. `L1-01` stopped at G-POLE and does not count
(amendment A1); L1's attempt on `n6-freeze-1a` is `L1-02`. Amendment A5: L2's attempts are `L2a-01` (the freeze half) and `L2b-01` (the
test half); both read `inputs_L2/`; L2b reads `results/next_n6/L2a-01/out/edit/n6/metrics.json` (`--against`). A relaunch of L2a takes
`L2a-02`, and L2b's `--against` names it (a dated record).

### 9.5 Sessions

```
B=s3://pg-lineage-769063704696
```
**S1. Freeze 1 (laptop).** §9.1. Upload:
```
aws s3 cp studies/next/n6/pairs.json $B/results/next_n6/inputs/pairs.json --profile fuelos --region us-east-1
aws s3 cp studies/next/n6/inputs/e1_known.json $B/results/next_n6/inputs/e1_known.json --profile fuelos --region us-east-1
aws s3 cp infra/aws_pg/steps/next_n6_L1.json $B/results/next_n6/L1-01/steps.json --profile fuelos --region us-east-1
```
**S2. L1 (0.29–0.67 GPU-h; job stop rule at 0.9 × 47 min; hard kill 53 min).** Attempt `L1-02` on `n6-freeze-1a` (A1); the
steps file is uploaded to `$B/results/next_n6/L1-02/steps.json` as in S1.
```
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_n6/L1-02/steps.json" \
    --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_n6/L1-02 --budget-min 53 --json --detach
./infra/aws_pg/pg watch results/next_n6/L1-02
```
The first gates to read are G-SAME, K1, K2. Then the ten raw examples, the bf16 probe values, `measured.sec_per_edit_layer_unit`.
**S3. Freeze 1b (laptop).** Pull `colnorms.npy`, `energies.npy`, `g_pole.npz`, `metrics.json`, `edits.jsonl`, `samples/` to
`studies/next/n6/banks/L1-02/`. Read the raw examples (§7 step 1). `PYTHONPATH=. python3 -m infra.pg.n6 final --pairs
studies/next/n6/pairs.json --l1 studies/next/n6/banks/L1-02 --out studies/next/n6/pairs_L2.json` (the stage-2 file is
`pairs_L2.json`, the name the L2 job fetches from `inputs_L2/`; A1), then `PYTHONPATH=. python3 studies/next/n6/stage2_check.py
studies/next/n6/banks/L1-02 studies/next/n6/pairs_L2.json studies/next/n6/stage2_check.json`. If it stops (no balanced point,
or fewer than 40 pairs): **L2 does not launch**; the record goes to the journal with the banked arrays, and the head decides. Else
register the sha256 in `INPUT_SHA["L2"]`, recompute `projection.json` at L1's measured rate (center: the measured rate; high: 1.25 ×
it), write the L2 budget into `next_n6_L2.json` (job `--budget-min` = 0.9 × the hard kill, hard kill = 1.3 × the high projection),
rerun the dry run, commit, tag `n6-freeze-1b`, upload to `inputs_L2/`.
**S4. L2a** (amendment A5: the freeze half and every gate; `next_n6_L2a.json`, attempt `L2a-01`; job stop rule `--budget-min 1672`,
hard kill 1,858 min; the exact commands in `studies/next/n6/STATE.md`). Pull `metrics.json`, `edits.jsonl`, `colnorms.npy`,
`energies.npy`, `g_pole.npz`, `samples/` to `studies/next/n6/banks/L2a-01/`. **S5. Freeze 2** (§6.2, §7 step 3): `PYTHONPATH=. python3
-m infra.pg.n6 freeze --bank studies/next/n6/banks/L2a-01 --pairs studies/next/n6/pairs_L2.json --out studies/next/n6/thresholds.json`;
the DISCOVERY numbers and the prediction in the journal; commit; tag `n6-freeze-2`. If G-POWER failed: the verdict on L2a alone
(UNDECIDED_NO_POWER) and stop. **S6. L2b**, only after `n6-freeze-2` with G-POWER passed (`next_n6_L2b.json`, attempt `L2b-01`,
`--against` L2a-01's `metrics.json`; job stop rule `--budget-min 1601`, hard kill 1,779 min); pull to `banks/L2b-01/`. **S7. Unblind**
(§7 step 4): `PYTHONPATH=. python3 -m infra.pg.n6 verdict --bank studies/next/n6/banks/L2a-01 --bank studies/next/n6/banks/L2b-01
--pairs studies/next/n6/pairs_L2.json --thresholds studies/next/n6/thresholds.json --out studies/next/n6/verdict_n6.json`.
**S8. The recount and the head's read.** (Before A5, at A4: one launch `L2-01`, job stop rule 3,089 min, hard kill 3,433 min.)

### 9.6 Budget, stop rule, park order

From FLOP counts (`n6.projection`): per token per layer 2(4·4096² + 3·4096·11,008) for the matmuls plus 4·1,024·4,096 for attention at
a mean context of 1,024, the head 2·4,096·100,278; an edit at layer l costs (32 − l) layers plus the head over W × 2,049 positions (196,704 for L1's 96 windows; 2,098,176 for L2's 1,024,
amendment A4); fixed
cost (download 14.6 GB, load, cast, records) 420 / 600 / 900 s; the bf16 terms (K2, G-POLE on 197 poles × 10 prefixes,
the bf16 base pass and two probes) at four times the fp32 rate; K3 and its failing input in fp32. Rates: 60 / 40 / 25 TFLOP/s sustained fp32 (no TF32)
on one L40S; no fp32 rate is banked, L1 measures it.

| launch | contents | runs | GPU-h low / center / high |
|---|---|---|---|
| L1 | K1, K2, G-POLE (197 poles), bf16 probes, cast, K3, two base passes over 96 windows, column norms and E⁺ of every pole, 10 raw column edits, 2 probes, 1 zero edit | 13 edits | 0.29 / 0.43 / 0.67 |
| L2 | the same preamble; 96 positive-control edits (32 layers × col, rand, col_half), 4 probes, 1 zero edit, 296 pair edits (74 pairs × 2 members × col, rand) | 397 edits | 3.05 / 4.56 / 7.28 |
| total | | 2 launches | **3.34 / 4.99 / 7.95** ($7.48 / $11.18 / $17.81 at $2.24 per hour; on g6e.4xlarge at $3.00424: $10.03 / $14.99 / $23.88) |
| L2 after A4 (L1-02's rate) | the same preamble with 1,024 eval windows; G-INST's pass over L1's 96 windows; 24 positive-control edits (8 layers × col, rand, col_half), 4 probes, 1 zero edit, 180 pair edits (45 pairs × 2 members × col, rand) | 209 edits | — / 35.20 / 44.01 ($78.86 / $98.57 at $2.24 per hour; `projection_L1.json`) |
| L2a after A5 (L1-02's rate) | the same preamble; 24 positive-control edits, 2 PC probes, 84 freeze-pair edits (21 pairs × 2 members × col, rand), 2 pair probes, 1 zero edit | 113 edits | — / 19.05 / 23.82 ($42.68 / $53.35) |
| L2b after A5 (L1-02's rate) | the same preamble; 96 test-pair edits (24 pairs × 2 members × col, rand), the same 2 pair probes and zero edit | 99 edits | — / 18.24 / 22.80 ($40.86 / $51.07) |
| L2a + L2b | A4's one launch, one more setup (1.50 GPU-h center), the 3 edits run in both (0.59 GPU-h center) | 212 edits | — / 37.29 / 46.62 ($83.54 / $104.42; L2a alone if G-POWER fails) |

The card's range is 4–8 GPU-h. **Hard kills**: L1 53 min (1.3 × 0.67 h); L2 568 min on today's projection, reset at freeze 1b from
L1's rate: 436 min at A3 (96 windows), 3,433 min after A4 (1.3 × 44.01 h; job stop rule 3,089 min); after A5, L2a 1,858 min
(1.3 × 23.82 h; job stop rule 1,672 min) and L2b 1,779 min (1.3 × 22.80 h; job stop rule 1,601 min). **Stop rule, in the job**: before each layer the job projects the remaining edits from its measured seconds per edit-layer
unit; if the elapsed time plus the projection exceeds 0.9 × `--budget-min`, every remaining edit is PARKED, the job writes its records
and ends normally (planted in the dry run). **Park order**: the progression runs layer 0 to 31, so parking removes the highest layers
first. If parked test pairs exceed 0.2 of the half, the verdict is DOES NOT COUNT (G-PARK); a second launch for the parked layers
(starting the progression at the first parked layer) is the head's decision. After A5, an L2a that parks any edit has no G-PC record
and L2b refuses it (`Job.same_comparison`); a relaunch is the head's decision. Before L2, if the measured rate makes the high projection
exceed 12 GPU-h, the head chooses among: drop col_half (−32 edits, −8 %), or 64 eval windows instead of 96 (−33 %), recorded as a
dated amendment before freeze 1b. After A4 the high projection is 44.01 GPU-h, above 12; A4 is the head's dated amendment, and its cost
is stated in §14 A4 for the launch decision.

### 9.7 Artifacts

`studies/next/n6/`: `pairs.json`, `pairs_L2.json`, `projection.json`, `inputs/e1_known.json`, `offline_gate/`, `samples/`,
`banks/<attempt>/`, `thresholds.json`, `verdict_n6.json`, `JOURNAL.md`, `figures/fig_n6.png` (D_col by pair with the rand floor and the
PC per layer drawn), `READ_n6.md` (the recount).

---

## 10. The ten raw examples

The five freeze pairs `n6.raw_example_pairs` picks (the first freeze pair of the layers nearest 0, 8, 15, 22, 30 that hold one), both
members' column edits: pole, partner, Frobenius norm c, corpus loss change in nats per token, the five tokens whose loss changed most
with 32 tokens of context, and (L2) the random-direction twin's ΔL. Zero-GPU part: `studies/next/n6/samples/raw_examples_zero_gpu.{md,json}`
(pole, partner, layer, ages, adjusted age, max\|a\|, density, the three strongest T_15 contexts). The GPU part is the first GPU step
(L1, `samples/raw_examples.jsonl`).

What the zero-GPU part shows. The pairs are L03N6372− / L03N7574− (layer 3, ages T_5 / T_12), L08N9190− / L08N5063− (T_8 / T_13),
L15N10390− / L15N6531− (T_6 / T_10), L22N7990− / L22N10231+ (T_9 / T_12), L30N6280+ / L30N7717− (T_10 / T_13). Within each pair
max\|a\| agrees within 10 % (0.254 / 0.254; 1.108 / 1.071; 3.371 / 3.560; 5.616 / 5.536; 18.86 / 17.23) and density within a factor 1.65, as the calipers require. The contexts are
what a reader would expect of poles with different jobs at matched strength: L03N6372− fires on "PROC**EDURE**" in SQL `CREATE
PROCEDURE` (three of three top contexts; this is the neuron the critic calls the lithium neuron), its partner L03N7574− on unrelated
mid-word pieces (" unab|ated", "O|XY", "O|xy"); L15N10390− on the subject noun after "The" in business news (" consortium", " scheme",
" owner"), its partner on sentence punctuation. The edit size, ΔL and the five tokens are filled from L1-02 in
`studies/next/n6/samples/raw_examples_L1.{json,md}` (ΔL of the ten column edits −4.87e-6 to +9.76e-6 nats per token).

---

## 11. The three dimensions

- **Predictive**: a pre-registered paired sign test frozen before any edit's loss exists, with its floor (0.5 and p_rand), ceiling (the
  PC ordering) and room shown; the prediction for the test half is written after the freeze half. A FLAT or REVERSED result counts the
  same as SUPPORTED.
- **Novel**: no work found conditions edit collateral on a unit's training age at a fixed checkpoint with energy-matched pairs (§1.5;
  `04_LITERATURE.md` N6 OPEN). Novel if any label other than UNDECIDED lands; a FLAT result is new as a measured negative.
- **Practical**: defined here as a measured win against a baseline: the corpus loss saved by the rule "edit the later-born of two
  matched candidates" against a random pick and against the energy rule, in nats per token with intervals (§5 item 5). A win counts only
  if its lower bound against the energy rule is above 0. Without it the practical dimension is empty, and the spec says so.

---

## 12. Prediction grid (filled at unblinding, whatever the result)

| quantity | prediction | realized | label |
|---|---|---|---|
| K1 job_fall | 0.6742 (bit-identical) | | |
| K2, K3 | 2.67537; K3 − K2 within ±0.005 | | |
| F_eps fp32; bf16 probe | ≤ 1e-8; 1e-5 – 1e-3 | | |
| G-PC col > rand | 28–32 of 32 | | |
| stage-2 pairs | 50–74 | | |
| median \|ΔL_col\| of a pair edit | 1e-5 – 1e-3 nats per token | | |
| p_col (test) | 0.60 [0.43, 0.75] | | |
| p_rand (test) | 0.52 | | |
| H_age | prior §1.4 | | |
| H_adj | as H_age, wider | | |
| H_eff | FLAT or UNDECIDED | | |
| practical gain vs energy rule | interval containing 0 | | |
| GPU-h | 3.34–7.95 | | |

---

## 13. References

- `studies/next/09_PLAN.md` J11; `studies/next/06_CRITIC.md` N6; `studies/next/03_IDEAS.md` N6; `studies/next/05_SCORING.md` line 133;
  `studies/next/10_RECOUNT2.md`; `studies/next/12_SHAPING_NOTE.md` §4; `studies/next/04_LITERATURE.md` N6.
- `studies/instruments/SPEC_instruments.md` §3; `studies/instruments/results/e1/verdict_e1.json`; `.../1/t15/samples/e1.jsonl`.
- `studies/next/n12/READ_n12.md` §6 and `READ_head.md` addendum; `studies/next/m2/READ_head.md`; `studies/next/07_PICKS/M2_OFFLINE_GATE.md`
  amendment A5.
- arXiv 2509.24198, read in full for the claim in §1.5.

## 14. Amendments

### A1 (2026-11-06, the head's decision at 15:40Z; before freeze 1a): G-POLE's tolerance 0.05

**G-POLE's tolerance goes from 1e-2 to 5e-2 on the same statistic; nothing else in the gate changes.** The statistic stays the
maximum over the poles of the maximum over the first 10 banked T_15 exemplars of \|a_new − a_bank\| / max\|a_bank\|. The failing
input (neuron n + 1 at the same positions, read in the same forward) and the stop on failure are unchanged. Code:
`n6_edit.POLE_TOL`.

- **Cause.** L1-01 (`i-075cff985f3d39f3e`) stopped at G-POLE: the worst pole was 0.01506 (L08N3846+) against 0.01, and 6 of 197
  poles were above 0.01. K1 and K2 had passed (`studies/next/n6/diag/L1-01/cause.md`, `metrics.json` → `gates`).
- **Reasoning.** The statistic compares two computation paths in bf16. The bank came from full 2,049-token windows in batches
  of 8 (the harvest). The recompute runs right-padded prefixes cut at each exemplar, sorted by position, in batches of 8
  (`reuse_ablate.readout`). Different matrix shapes can select GPU kernels that round bf16 differently, so the pass side carries
  kernel rounding of a few bf16 steps (one step is 0.39–0.78 % of a value). Measured in L1-01: 0 to 0.0151 over 197 poles. The
  failing input sits at 0.92 or more. At 0.05 the pass side has room of 3.3 times (0.05 / 0.01506) and the failing side 18 times
  (0.9234 / 0.05); the offline failing input, the T_14 values at the same positions (smallest 0.150), is 3.0 times above. The
  gate's purpose, that the pole edited is the pole the bank describes, is served at that bar.
- **Said plainly.** The bar was set after seeing the pass-side values of one run (L1-01). It was not frozen from a control pair
  measured before the run.
- **Evidence: all 197 values of L1-01** in `studies/next/n6/diag/L1-01/A1_g_pole_197.tsv` (pole, layer, value, failing input,
  pair, member, half; copied at full precision from `diag/L1-01/metrics.json` → `gates.G_POLE.per_pole`, file sha256
  `830928793e55c051532d99bfff38543a2281ea89521a320ce1aabebb658a7aeb`). Their distribution:

  | value | poles |
  |---|---|
  | exactly 0 | 111 |
  | (0, 1e-6] | 45 |
  | (1e-6, 1e-4] | 0 |
  | (1e-4, 1e-3] | 1 |
  | (1e-3, 5e-3] | 15 |
  | (5e-3, 1e-2] | 19 |
  | (1e-2, 5e-2] | 6: L08N3846+ 0.01506, L31N3149− 0.01197, L18N2982+ 0.01152, L25N10599− 0.01100, L26N10050− 0.01029, L15N6531− 0.01026 |
  | above 5e-2 | 0 |
  | median; 90th percentile; maximum | 0.0; 0.00592; 0.01506 |
  | failing input (neuron n + 1): smallest; median | 0.9234; 1.0015 |

- **What the job now records** (the head's item 2). `out/edit/n6/g_pole.npz`, written before the gate reads it: per pole
  (sorted by name) the layer, the neuron, neuron n + 1, the 10 bank positions, the bank's a, and the recomputed g, u and a of the
  neuron (column 0) and of neuron n + 1 (column 1). The job's input holds the bank's a only; the harvest's bf16 g and u at the same
  positions are in the T_15 `contexts.jsonl` and are joined offline. `metrics.json` → `measured.peak_mem_gib`: the peak of
  `torch.cuda.max_memory_allocated` in GiB for each phase (load_K1, K2, G_POLE, bf16_probes, cast, K3, pass1, pass2,
  progression), the counter reset after each.
- **The rule for every later gate of this kind** (written into `N6_OFFLINE_GATE.md` §6 as the gap this run found): a gate that
  compares two computation paths needs its tolerance frozen from a control pair measured on those two paths, or it is reported
  rather than gated. G-POLE at 0.05 does not meet the rule; it is gated at 0.05 by the head's decision, on the reasoning above.
- **Dry run.** `N6_DRYRUN_OK`, 83 checks, 0 failed (78 before; five added: `g_pole.npz` holds what it should and its statistic
  equals `per_pole`; the peak memory is recorded per phase; a planted G-POLE value of 0.03 passes; a planted 0.06 stops the job,
  and its value is 0.06). Log `studies/next/n6/offline_gate/dryrun_freeze1a.log`.
- **The stage-2 file name** is `studies/next/n6/pairs_L2.json`, the name the L2 job fetches from `inputs_L2/` (§9.4, §9.5);
  `stage2_check.py`'s default changed from `pairs_final.json`.
- **Unchanged.** `pairs.json` and `e1_known.json` and their sha256 in `INPUT_SHA["L1"]`; `next_n6_L1.json` (job stop rule 47 min);
  the hard kill 53 min; every other gate, threshold and label. The next L1 attempt is `L1-02` on `n6-freeze-1a`.

### A2 (2026-11-06, the head's decision at 16:30Z; before freeze 1b): G-ZN and G-CACHE get tolerances frozen from L1-02

**G-ZN passes when every zero-norm edit's \|ΔL\| ≤ 1.221e-7 nats per token. G-CACHE passes when the mean per-token loss difference
is at most 1.221e-7 in absolute value and the largest per-token difference is at most 1e-5 nats.** 1.221e-7 is the registered tie
margin τ_eps at L1-02 (10 × F_eps, F_eps = 1.2210622426951182e-8, `banks/L1-02/metrics.json` → `gates.G_EPS.value`). Code:
`n6_edit.TAU_EPS_L1`, `CACHE_MAX_TOL`, `zn_gate`, `cache_gate`. G-ID is unchanged (one path scored twice, bitwise; 0.0 at L1-02).

- **Cause.** L1-02 failed both gates at their bitwise bar (`studies/next/n6/diag/L1-02/record.md`). Both compare two computation
  paths: the progression applies the final norm and the head to one window at a time (`Model.window_logits`, input [1, 2,049, 4,096]);
  base pass 1 is the model's own forward, which applies them to a batch of 4 windows ([4, 2,049, 4,096]). They were registered
  bitwise without a control pair on those two paths: the gap the two-path rule names (`N6_OFFLINE_GATE.md` §6).
- **Reasoning.** L1-02's zero edit is the control on these two paths: δ = 0, the hook installed, the progression's path read against
  base pass 1. Its values set the bars. G-ZN's bar is 4,000 times the observed value; G-CACHE's largest-difference bar is 10 times the
  observed value; the mean bar is the tie margin the verdict already uses, so an offset under it cannot move a counted sign.
- **Said plainly.** The tolerances were set after seeing L1-02's values. They were not frozen from a control pair measured before the
  run.
- **Evidence (L1-02).** G_ZN `value_max_abs` 3.015144210694043e-11 (P002_late_zero, L03N7574−). G_CACHE `value_max_abs`
  9.5367431640625e-07 (2^-20, one fp32 step of a per-token loss between 8 and 16 nats). The zero edit's five largest per-token
  differences are 9.537e-7 each, at base losses 8.7–13.2 (windows 93 and 91; `edits.jsonl` → `top5`), equal to G_CACHE's largest:
  the zero edit and G-CACHE read the same path. G-CACHE's mean was not recorded in L1-02; the zero edit's ΔL on the same path is
  −3.0e-11. G-ZN's failing input, the first column edit (P002_early_col), had \|ΔL\| 4.43e-6, 36 times the new bar. G-ID 0.0.
- **Failing inputs.** G-ZN: the first column edit's ΔL must exceed the bar (unchanged in kind). G-CACHE: each window scored from the
  next window's final residual, computed in the job beside the pass side; it must exceed a bar.
- **What the job now records.** G_CACHE: `value_mean`, `value_max_abs`, `value_mean_abs`, `bitwise_equal`, the failing input's mean
  and largest difference. G_ZN: `n_bitwise_equal_base`, `failing_input.value`. The head's input shapes seen in each phase
  (`measured.head_input_shapes`, from a forward pre-hook on `lm_head`), and in both gate records the shapes of the two paths.
- **Dry run.** `N6_DRYRUN_OK`, 99 checks, 0 failed (`studies/next/n6/offline_gate/dryrun_A2A3.log`); the 83 earlier CHECK lines are
  unchanged. Added for A2: a planted zero edit at 5e-8 passes and at 5e-7 fails; a planted largest per-token difference of 5e-6 passes
  and of 5e-5 fails (the failing input fails in each); a uniform offset of 2e-7 fails on the mean condition; a first column edit planted
  at 5e-8 makes the failing side pass, so it registers; the shapes recorded on the CPU are [2, 2,049, 32] for base pass 1 (the dry
  run's batch 2) and [1, 2,049, 32] for the progression.

### A3 (2026-11-06, the head's decision at 16:30Z; before freeze 1b): the stage-2 split swapped; pairs with an L1 outcome stay in the freeze half

**At stage 2 the test half is the pairs whose early member has an even neuron index, the freeze half those with an odd one, and a pair
holding a pole that L1 edited stays in the freeze half. On the 45 stage-2 pairs this gives 24 test pairs and 21 freeze pairs.** Code:
`n6.half_of`, `n6.final_pairs(..., l1_edited)`; `n6 final` reads the edited poles from L1's `edits.jsonl`. Stage 1 (`pairs.json`) is
unchanged.

- **Cause.** Under §6.1 stage 2 gave 28 freeze pairs and 17 test pairs. At 17 test pairs FLAT cannot fire: Wilson at p = 0.5 is
  [0.285, 0.715], outside the band [0.30, 0.70]. The head's decision: before any outcome of these pairs exists, the test half becomes
  the 28-pair half; the pairs L1's edits touched stay in the freeze half.
- **What L1 touched.** L1-02 edited ten poles: the column edits of five stage-1 freeze pairs, and the probes and the zero edit on two
  of those poles. Four of the 45 stage-2 pairs are the same two poles as an L1 pair, with the same c (the smaller of the same two column
  norms), so both members' ΔL_col and the pair's D_col were banked by L1-02: P003 (L03N6372− / L03N7574−; stage-1 P002), P008
  (L22N7990− / L22N10231+; P020), P025 (L15N10390− / L15N6531−; P039), P039 (L30N6280+ / L30N7717−; P069). Each has an even early
  neuron, so the swap alone would have put all four in the test half; they stay in the freeze half. The fifth L1 pair (L08N9190− /
  L08N5063−, stage-1 P037) is not a stage-2 pair. No other stage-2 pair holds an L1-edited pole, so no edit outcome of the 24 test
  pairs exists.
- **The realized split is 24 / 21, not 28 / 17.** The 28-pair half less the four pairs above.
- **Room** (`n6.wilson`, the rules of §5):

  | test pairs | Wilson at p = 0.5 | FLAT fires at (wins, ties as halves) | SUPPORTED needs | REVERSED at most |
  |---|---|---|---|---|
  | 24 (A3, realized) | [0.314, 0.686] | 12.0 only | 17 | 7 |
  | 28 (the head's count) | [0.326, 0.674] | 13.5 to 14.5 | 20 (19.5 with a tie) | 8 (8.5 with a tie) |
  | 17 (§6.1) | [0.285, 0.715] | never | 13 | 4 |

  H_adj reads the 16 concordant test pairs: FLAT cannot fire; SUPPORTED needs 12; REVERSED at most 4. FLAT also needs R ≥ 0.5.
- **The freeze half's uses, from the code.** `n6.freeze` reads only freeze pairs: τ_eps comes from the gates (the fp32 probes), not
  from a half; τ_s is the median over the 21 freeze pairs of `n6.boot_sd_win`; G-RES's q10 is over their 42 column edits; the discovery
  numbers. The positive control does not depend on the halves: `n6_edit.pc_gates` counts the 32 positive-control layers (each layer's
  highest-E⁺ pole, not a pair pole); `n6.freeze` does not read it; `n6.verdict` reads `gates.G_PC` for the ceiling. The freeze half
  also supplies the two probe poles (`n6.probes`: L01N8975−, L30N9731+) and the ten raw examples (`n6.raw_example_pairs`: P005, P035,
  P025, P008, P015; the freeze half spans 15 layers).
- **The file.** `pairs_L2.json` regenerated by `n6 final` (sha256 `2d206ffcce0acccf08d26da71192d5958896a0d8a17ce7e976a5a6e9ad8da496`;
  under §6.1 it was `466278270944e73f831f6bee6fbdb23f0e7503e7d7cda2ae7266c81f04603849`). The pairs, calipers and balance are unchanged;
  only `half`, `rule.split`, `n_freeze`, `n_test` and the new `l1_edited` differ. `stage2_check.json` and `projection_L1.json` were rerun
  on it; the projection is unchanged (281 edits, 4.466 / 5.582 GPU-h, hard kill 436 min, job `--budget-min` 392).
- **Dry run.** Added for A3: `pairs_L2.json` rebuilt from L1-02's banks equals the committed file (floats within 1e-6: written under
  numpy 2.4.3, the dry-run venv has 1.26.4); 24 test, 21 freeze, the four pairs in the freeze half and every other pair placed by its
  parity; probes and raw examples drawn from the freeze half; `INPUT_SHA["L2"]` equals the committed files; a planted L1-edited pole
  moves its pair, and only it, to the freeze half; the room at 24, 28 and 17 test pairs as tabled, with `n6.freeze` computing τ_s and
  q10 from 21, 17 and 28 freeze pairs.

### A4 (2026-11-06, the head's decision at 17:00Z; before freeze 1b): G-RES at the factor 50; the design changed for power

**Part 1: G-RES passes when F_eps ≤ min(1e-6, q10 / 50); it was q10 / 100. Part 2: L2 reads 1,024 eval windows instead of 96; a power
gate, G-POWER, is registered at freeze 2 (τ_s ≤ 0.5 × the median \|D_col\| over the freeze pairs, else UNDECIDED_NO_POWER before the test
half is read); the positive control runs in 8 layers (0, 4, …, 28), and G-PC and G-DOSE need 6 of 8.** L2 has not run, so no outcome of
any test pair exists. Code: `n6.EPS_REL` 0.02, `W_EVAL_L2`, `eval_set`, `pc_layers`, `PC_MIN`, `POWER_MAX`, `freeze` → `G_POWER`,
`no_power`, `verdict`; `n6_edit.CHUNK`, `progression`, `edit_acc`, `run_edit`, `edit_record`, `pc_gates`, `host_mem_available`, the
G-INST pass; `INPUT_SHA["L2"]`.

**Part 1, G-RES.**

- **Cause.** On L1-02's ten column edits q10 = 1.0753e-6 and F_eps = 1.2211e-8 (P002_late_probe), so q10 / F_eps = 88.06 where the rule
  asked 100; the preview failed by 1.136 (`studies/next/n6/diag/L1-02/g_res_preview.json`). It is not a path effect: the probe runs the
  zero edit's path, whose ΔL is −3.0e-11, 0.25 % of F_eps. The edits are small: median \|ΔL_col\| 2.237e-6 nats per token, against the
  1e-5 to 1e-3 predicted in §12.
- **Reasoning.** The gate's purpose is that the edits' effects sit far above the fp32 rounding floor. At 88 times that is met. The factor
  becomes 50.
- **Said plainly.** The factor was set after seeing the preview. It was not frozen from control data before L1-02.
- **Evidence.** At the factor 50 the preview's bound is 2.1505e-8 and F_eps / bound is 0.568: it passes
  (`studies/next/n6/diag/L1-02/a4_preview.json`, from `diag/a4_preview.py`). `diag/g_res_preview.py` now fixes its factor at freeze 1's
  0.01 (`EPS_REL_FREEZE1`); rerun, it writes the committed JSON byte for byte.
- **Failing inputs.** F_eps 5e-5 still fails (the absolute bound). Added: a floor at 0.05 × q10 fails; 0.015 × q10 passes (it would
  fail at the factor 100).

**Part 2, power.**

- **Cause.** The window-bootstrap sd of D_col on the five L1 pairs is 3.47e-6 to 1.156e-5 (median 6.63e-6) against \|D_col\| of 2.15e-6
  to 5.99e-6 (`g_res_preview.json` → `eval_set_noise`). No pair's sign is resolved at 2 sd, and 9 of the 10 single edits lie within one sd
  of zero. At 96 windows the sign test would read noise.
- **(a) The eval set.** L2 reads the registered rule of §0.5 at W = 1,024: `n6.eval_windows(1024)`, windows 10,045 + ⌊i · 14,633 /
  1,024⌋ for i = 0 … 1,023, that is 10,045 to 24,663, ascending, distinct, wholly in band 3, BOS first; 2,097,152 targets. sha256 of the
  JSON list: `5822191728d49aba45817f8479058c266bcaaa057dc1037fa0d5509507d1b223`. 32 of L1's 96 windows are among them. The corpus file is
  the same (`tokens.npy` sha256 `2aca80ca6537f46b8f60a550b4a3f2f13be2d18c5402be8f1af5c00caa86f531`, which the job checks against the
  corpus manifest); the job also writes the sha256 of the window list and of the eval token array (int64, 1,024 × 2,048, without BOS)
  to `metrics.json` (`eval_windows_sha256`, `eval_tokens_sha256`). For independent windows the sampling sd falls by √(1,024 / 96) = 3.27.
  - The reading of "the first 1,024 windows of the registered order": the registered order is the rule's ascending list at W = 1,024.
    The other reading, the first 1,024 consecutive band-3 windows (10,045 to 11,068), holds 2.1M of band 3's 30M tokens and about 3.5 of
    its 50 source directories (band 3 is ordered by directory, `neurons_corpus`). It would not be the text the 96 windows sample, so it
    was not taken.
  - `pairs_L2.json` carries the list (`eval_windows`, `w_eval` 1,024) and G-INST's windows (`inst_windows`, L1's 96). Nothing else in the
    file changed: the pairs, the halves and the balance are A3's. sha256 `7e3a99480b3a2b939ca6eb0a8ae6fe43ee0e8bad22bee302c1a37ca1e4e1755b`
    (A3's: `2d206ffc…`). `stage2_check.json` rerun: only its record of this sha256 changed.
  - **G-INST follows.** L1's E⁺ is a sum over its 96 windows. Over 1,024 windows E⁺ differs, and the registered comparison (rel 1e-5)
    would fail without a defect. The L2 job reads G-INST's E⁺ in its own pass over L1's 96 windows (`inst_windows`, the same batches as
    L1), at the cost of one 96-window pass (113 s). Its failing input is unchanged; a dry-run failing input is added: G-INST read on L2's
    eval set stops the job.
- **(b) G-POWER**, at freeze 2 (`n6.freeze` → `thresholds.json` → `G_POWER`): τ_s, the median over the freeze pairs of the window-bootstrap
  sd of D_col, must be at most 0.5 × the median \|D_col\| over the freeze pairs. If not, `n6.verdict` reads it before anything else and
  writes UNDECIDED_NO_POWER for H_age, H_adj and H_eff; no test pair is read and no test label is issued (with a failed gate as well, DOES
  NOT COUNT, UNDECIDED_NO_POWER kept as the realized label).
  - Room. On the five L1 pairs with freeze's seeds: τ_s 6.356e-6, median \|D_col\| 4.474e-6, ratio 1.42, so it fails at 96 windows.
    Scaled by √(96 / 1,024) with \|D_col\| held at its 96-window value: 0.435, so it passes at 1,024 (`a4_preview.json`). The ratio reaches
    0.5 at 774 windows under the same assumption.
  - Said plainly. Holding \|D_col\| is optimistic: at 96 windows each \|D_col\| carries the same window noise. If the true contrasts are
    smaller than the five 96-window values, the ratio at 1,024 windows is higher and G-POWER fails.
- **(c) The roster.** The positive control runs in the 8 layers 0, 4, …, 28 instead of 32: 24 edits instead of 96. G-PC needs at least 6
  of 8 (the chance of 6 or more of 8 under a fair coin is 37/256 = 0.145; 24 of 32 had 0.0035). G-DOSE reads the same 8 layers' edits, so
  it needs 6 of 8 too. The ceiling of §5 is the upper Wilson bound over the 8 layers. The probes are unchanged: two pair poles and the
  positive-control poles of layers 0 and 31 (layer 31's pole is still chosen in base pass 1; no positive-control edit runs there). The
  random-direction edits stay for every pair in both halves (H_dir needs them); the raw-example edits stay. L2: 209 edits (24 positive
  control, 2 positive-control probes, 180 pair edits, 2 pair probes, 1 zero edit); A3's roster was 281.
- **Memory; the kept residual in chunks.** At 1,024 windows h_l is 34.4 GB (32.02 GiB). h_l and h_(l+1) on the GPU would be 64 GiB beside
  the 27.2 GiB model, so the residual does not fit. The job keeps h_l of every window in host memory (fp32). At each layer the windows go
  to the GPU in chunks of 96 (10 chunks of 96 and one of 64): the chunk's h_(l+1) is computed once, every edit of the layer runs on the
  chunk, and h_(l+1) is copied back over h_l. Each window is computed in the same batches of 4 as without chunks, so the records do not
  depend on the chunk size (dry run: one chunk and two chunks give equal records, field by field). A record is written when its layer's
  chunks are all done; the stop rule still parks whole layers.
  - GPU peak per phase, projected (GiB): every phase but the progression runs L1-02's batch shapes, so L1-02's measured values hold:
    load_K1 13.94, K2 18.195, G_POLE 15.384, bf16_probes 16.664, cast 27.96, K3 31.023, pass1 31.024, inst 31.024 (new: pass 1's forward on
    96 windows), pass2 31.024. The progression holds h_l and h_(l+1) of at most 96 windows, as L1-02 did, without L1-02's concatenation
    copy: at most 36.201 (38.9 GB). The L40S has 48 GB (44.7 GiB). No step is projected above it, so there is no lane decision on memory.
  - Host memory, projected: the residual 32.02 GiB, the exemplar log-probabilities 1.85 GiB (99 edited poles × 50 × 100,278 × 4 bytes),
    L1-02's largest RSS 2.41 GiB: 36.3 GiB of the g6e.2xlarge's 64 GiB (128 on g6e.4xlarge). The job asserts at start that MemAvailable is
    at least 40.0 GiB (the residual plus 8 GiB) and records it (`host_mem`). Precision and the batch are unchanged.
- **Projection** (`studies/next/n6/projection_L1.json`, `projection_l1.py`). L1-02's edit rate is 29.01 TFLOP/s. L2's edits: 121,345 s
  (FLOPs at that rate; the job's 3.20 s per edit-layer unit scaled by 1,024 / 96 gives 136,268 s, since its unit time also holds the layer
  advance). L1's window-dependent seconds outside the edits, 402 s (base pass 1 measured, base pass 2 taken equal, the three bf16 probe
  passes, the layer advance, G-CACHE), scaled by 10.67: 4,290 s. Fixed: 604 s. Added: the G-INST pass 113 s; the host-device copies 384 s
  (2,303 GB at an assumed 6 GB/s, not measured). **Center 126,736 s = 35.20 GPU-h; high (1.25 ×) 44.01 GPU-h; $78.86 and $98.57 at $2.24
  per hour; hard kill ⌈1.3 × 44.01 h⌉ = 3,433 min; job stop rule ⌊0.9 × 3,433⌋ = 3,089 min** (`next_n6_L2.json`). A3's projection at 96
  windows was 4.466 / 5.582 GPU-h, 436 / 392 min.
  - Said plainly. The trim pays for 72 of 281 edits (1,296 of 5,287 edit-layer units, 24.5 % of the roster at 96 windows). The eval set
    multiplies every edit's cost by 10.67. The net is 7.9 times A3's center. The §9.6 rule fires (high above 12 GPU-h); A4 is the head's
    decision under it, and the cost is for the launch decision. The same model at other sizes, with A4's roster (center / high GPU-h; the
    preview's G-POWER ratio with \|D_col\| held): 256 windows 8.95 / 11.19 (0.87); 512 17.70 / 22.13 (0.615); 774 26.66 / 33.32 (0.500);
    1,024 35.20 / 44.01 (0.435) (`projection_L1.json` → `L2.by_window_count`).
- **Dry run.** `N6_DRYRUN_OK`, 117 checks, 0 failed (`studies/next/n6/offline_gate/dryrun_A4.log`). Added (18): the eval set and its
  sha256, G-INST's windows; the 209-edit plan; 37/256; one chunk against two chunks, field by field; an 8-window L2 eval set where G-INST
  reads L1's 4 windows and passes; G-INST read on L2's eval set stops the job; G-PC planted 6 of 8 passes and 5 fails (G-DOSE the same; the
  swapped labels fail each time); G-RES planted 0.015 × q10 passes and 0.05 × q10 fails; G-POWER planted 0.4 passes and the test half is
  read, 0.6 fails with UNDECIDED_NO_POWER and no test pair read, the same with every test record removed, and DOES NOT COUNT with a failed
  gate. Changed (7 of the 99 earlier lines): A17 (the new sha256), the peak-memory phases (with `inst`), G-PC computed on the
  positive-control layers, the tiny plan's edit count (59, was 71), the stop rule's counts, the planted G-PC line's wording, the G-RES
  line's wording. The other 92 are identical. `gate_sides.py` prints `GATE_SIDES_OK`, output equal.
- **Unchanged.** The pairs, the halves, the calipers and the balance; `e1_known.json`; L1's inputs; every other gate, threshold and label;
  the tie margin; the FLAT band and R.

### A5 (2026-11-06, the head's decision at 17:35Z; before freeze 1c): L2 runs as two launches, the freeze half first

**L2 is split by half into two launches. L2a reads the freeze half and runs every gate; its records decide freeze 2: τ_eps, τ_s,
G-POWER, G-RES and the positive control. L2b reads the test half and is launched only after freeze 2 is tagged with G-POWER passed. If
G-POWER fails, the study ends as UNDECIDED_NO_POWER at L2a's cost.** No L2 launch has run, so no outcome of any pair exists. Code:
`n6.plan(half=)`, `n6.SAME_KEYS`, `n6.read_banks`, the `freeze` and `verdict` CLI (`--bank`); `n6_edit` `--half`, `--against`,
`Job.same_comparison`, `half` and `pairs_sha256` in `metrics.json`; `projection_l1.py` (`L2a`, `L2b`, `A5_sum`).

- **Reason.** G-POWER is a check on the freeze half made before the test half is read, so it is paid for first. At 1,024 windows one L2
  launch projects to 35.20 / 44.01 GPU-h, and the power gate may still fail: the preview's 0.435 holds \|D_col\| at its 96-window size
  (A4). Split, the check costs L2a's 19.05 / 23.82 GPU-h.
- **The mechanism: one argument, one pairs file.** The job takes `--half freeze` (L2a) or `--half test` (L2b) at stage L2; stage L2
  without `--half` stops, and so does stage L1 with it. `n6.plan(pairs, half=...)` selects the roster from the same `pairs_L2.json`
  (sha256 `7e3a9948…`; `INPUT_SHA["L2"]` unchanged). Not taken: two pairs files (two new sha256 to register, the eval set written twice).
  - L2a, 113 edits: the positive control in the 8 layers (24), its two probes (layers 0 and 31), the 21 freeze pairs' col and rand (84),
    the two pair probes and the zero edit (both on freeze poles, as `n6.probes` chooses them). The raw-example edits are freeze-pair col
    edits, so they are in L2a. Gates: all of L2's (G-SAME, G-REV, K1, K2, G-POLE, G-FP32, K3, G-INST, G-ID, G-ZN, G-CACHE, G-EPS on four
    probes, G-NORM, G-PC, G-DOSE, G-SAMPLES).
  - L2b, 99 edits: the 24 test pairs' col and rand (96) and the same two pair probes and zero edit (3). Gates: the 13 that do not depend
    on the half; G-EPS reads its two pair probes; no G-PC, G-DOSE or G-SAMPLES.
  - The three edits in both launches. The head listed G-EPS and G-ZN among L2b's gates; G-EPS reads fp32 probe edits and G-ZN the zero
    edit, so L2b runs them. They are L2a's own edits, so their values in the two launches can be compared directly (both launches' G-EPS
    and G-ZN records). They cost 2,129 s, 0.591 GPU-h center (`P005_late_probe` and `P005_late_zero` at layer 1, `P015_late_probe` at
    layer 30). Dropping them would drop G-EPS and G-ZN from L2b; that is the head's call and is not taken here.
- **One comparison.** Each launch records `half` and `pairs_sha256` (the sha256 of the pairs content as canonical JSON) beside A4's
  `eval_windows_sha256` and `eval_tokens_sha256`. Before its first gate L2b reads L2a's `metrics.json` (`--against
  s3://pg-lineage-769063704696/results/next_n6/L2a-01/out/edit/n6/metrics.json`) and stops unless it is stage L2, half freeze, complete
  with nothing parked, and equal on `eval_windows_sha256`, `eval_tokens_sha256`, `pairs_sha256` and `w_eval` (`Job.same_comparison`;
  L2b records them under `against`). An L2a that parked any edit has no G-PC record, so the verdict would read DOES NOT COUNT; L2b
  refuses it so that the money is not spent.
- **The bank.** Two directories, `studies/next/n6/banks/L2a-01/` and `banks/L2b-01/`, each as the job writes it. The record formats are
  unchanged (dry run: every L2a and L2b record has the keys and the values of the one-launch record of that edit). `n6 freeze --bank
  banks/L2a-01` reads L2a alone; given L2b's bank it stops. `n6 verdict --bank banks/L2a-01 --bank banks/L2b-01` reads both through
  `n6.read_banks`: it asserts the halves, stage L2 and the four keys equal; it takes every record of L2a and the test-pair records of
  L2b (L2b's probe and zero records are read by its own G-EPS and G-ZN); it reads L2a's gates under their names and L2b's under
  `L2b:<name>`, so a failed gate in either launch makes every label DOES NOT COUNT. With G-POWER failed the verdict takes L2a's bank
  alone (given L2b's it stops); with G-POWER passed it needs both (L2a alone stops).
- **Projection** (`studies/next/n6/projection_L1.json` → `L2a`, `L2b`, `A5_sum`, from `projection_l1.py`: A4's model at L1-02's rate,
  29.01 TFLOP/s). Each launch pays the whole setup: the fixed part 604 s, the window part 4,290 s, the G-INST pass 113 s, the residual
  copies 384 s; 5,391 s (1.498 GPU-h). A4's `L2` block is unchanged by the rerun.

  | launch | edits | edit s | center GPU-h | high GPU-h | $ center / high at $2.24 | hard kill (1.3 × high) | job stop rule (0.9 × hard kill) |
  |---|---|---|---|---|---|---|---|
  | L2a (freeze) | 113 | 63,205 | 19.054 | 23.818 | 42.68 / 53.35 | 1,858 min | 1,672 min |
  | L2b (test) | 99 | 60,269 | 18.239 | 22.799 | 40.86 / 51.07 | 1,779 min | 1,601 min |
  | L2a + L2b | 212 | 123,474 | 37.293 | 46.617 | 83.54 / 104.42 | — | — |
  | A4, one launch | 209 | 121,345 | 35.204 | 44.005 | 78.86 / 98.57 | 3,433 min | 3,089 min |

  The sum is A4's one launch plus one more setup (1.498 GPU-h) plus the three edits in both launches (0.591 GPU-h): 35.204 + 1.498 +
  0.591 = 37.293 GPU-h center (the script asserts it). If G-POWER fails, the study spends L2a only.
- **Manifests.** `infra/aws_pg/steps/next_n6_L2a.json` (`--stage L2 --half freeze`, `--budget-min 1672`) and `next_n6_L2b.json`
  (`--stage L2 --half test --against …/L2a-01/out/edit/n6/metrics.json`, `--budget-min 1601`). A4's `next_n6_L2.json` is removed (it
  stays at `n6-freeze-1b`). A relaunch of L2a (`L2a-02`) changes L2b's `--against`, recorded with a date.
- **Said plainly.** L2b's budget is projected from L1-02's rate, as the head asked; L2a measures the rate at 1,024 windows
  (`measured.sec_per_edit_layer_unit`) before L2b is launched. L2b's launch condition (freeze 2 tagged, G-POWER passed) is a runbook
  step, not a check in the job: the job reads L2a's record, not `thresholds.json`.
- **Dry run.** `N6_DRYRUN_OK`, 136 checks, 0 failed (`studies/next/n6/offline_gate/dryrun_A5.log`); the 117 earlier CHECK lines are
  identical. Added (19): A22, the split of the real plan (113 + 99; the 3 edits in both; together the 209-edit plan in its order); in the
  tiny world an L2a and then an L2b, each record equal field by field to the one-launch record of that edit; L2b's 13 gates on both sides;
  L2a's gates, G-PC, G-DOSE, G-SAMPLES and raw examples equal to the one launch's; the sha256 record; freeze 2 through `n6 freeze --bank
  L2a` equal to freeze on the one launch; the verdict through `n6 verdict --bank L2a --bank L2b` (G-POWER planted to pass) equal to the
  one launch's in every field, all 6 test pairs read; a planted failure of L2b's K3 gives DOES NOT COUNT; G-POWER failed, the verdict on
  L2a alone is UNDECIDED_NO_POWER with no test pair read; the two manifests pass the job's argument checks, their `--budget-min` equal
  to the projection's job stop rules. Planted, each stops: an L2b whose eval set differs from L2a's (before G-SAME;
  no gate or record written); the verdict given an L2b bank with another eval set; an L2b against an L2a that parked; `n6 freeze` given
  L2b's bank; the verdict with G-POWER passed and L2a alone; the job at stage L2 without `--half`; `--half test` without `--against`.
  `gate_sides.py` prints `GATE_SIDES_OK`, output equal. pyflakes clean on the changed files.
- **Unchanged.** The pairs, the halves, the eval set, `pairs_L2.json`, `INPUT_SHA["L2"]`, `e1_known.json`; every gate, threshold,
  tolerance and label; the edit record; G-POWER's rule and its place before the test half.
