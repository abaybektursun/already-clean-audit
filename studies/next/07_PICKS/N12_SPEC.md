# N12: competition at birth, with M2's registered follow-up (the distinct-token twin). Pre-registration

**Answer.** One launch on one L40S answers both questions, projected at 1.64–2.33 GPU-h ($3.67–5.22 on g6e.2xlarge;
`studies/next/n12/projection.json`), hard kill 182 min (1.3 × the high projection). The competitor set of a birth is
B's top 20 at T_(k−1), and it needs **no new forward**: M2's job banked every neuron's activation at every exemplar
position at T_(k−1) and T_k (S3 `results/next_m2/score-06/out/score/m2/vectors/`), and B for any pole follows from
those vectors and the banked histograms (the ten raw examples recompute M2's top-10 lists exactly, 10 of 10 at both
roles). The launch is needed for three things the banks do not hold: the gradient of the loss with respect to each
pole's activation at T_(k−1) (the critic's instrument) and its validation against zero ablation, the twins' T_k
activations (M2 forwarded twins only at T_(k−1) and T_15), and the new distinct-token twins at every role. The ten
raw examples at zero GPU (freeze half) show the pattern the run is built to test: in 8 of 10 births the carrier's own
rise alone would have put it at rank 1–3 of 22,016 at T_k, while its competitors' moves alone would have left it at
or below its T_(k−1) rank (§10). That is a freeze-half look, not a result; the twins decide what part of it the
selection rule gives. The weight-free dry run printed `N12_DRYRUN_OK` (94 checks, 0 failed), and every gate whose
quantity the banks hold passed on its pass side and failed on its registered failing input at every point
(`N12_OFFLINE_GATE.md`); the gradient, the ablation and their identity have no bank and were shown on a tiny model only.

Written 2026-11-06 by an Opus worker in worktree `interp_research-bio`, branch `bio` (from `clean-neurons` at
`258c95815`). No model was run, no weights were downloaded, nothing was launched. Network reads: S3, read-only (M2's
vectors of score-04, score-05 and score-06, small files; the histogram members of 20 stats files by byte-range
reads; listings of `results/next_m2/`); four web searches (§1.5).

---

## 0. Status and provenance

| field | value |
|---|---|
| status | Reviewed by the head 2026-11-06: one decision before freeze 1 (G-VAL's bar ρ ≥ 0.7, §3.5); everything else accepted as written. Amended once after score-01 (A1, §14). The weight-free dry run printed `N12_DRYRUN_OK`, 94 checks, 0 failed, before and after the decision (`studies/next/n12/offline_gate/dryrun.log`; record `N12_OFFLINE_GATE.md`). The freeze-half raw examples (§10) were read before this draft; nothing of the test half's N12 quantities was computed. |
| freeze 1 | annotated tag `n12-freeze-1` (tag object `8a7dc51a5`) on commit `f60249949607b2e48182b8a6fe2bccbceb61ab45`, 2026-11-06T09:49:08Z. It holds this spec; `infra/pg/n12.py`; `infra/aws_pg/jobs/n12_score.py`; `infra/aws_pg/jobs/n12_dryrun.py`; `infra/aws_pg/steps/next_n12_score-01.json`; `studies/next/n12/units.json` (sha256 `c223dfee…`, registered as `n12_score.UNITS_SHA`); `studies/next/n12/offline_gate/gate_sides.py`; `studies/next/n12/samples/raw_examples.py`. The dry run printed `N12_DRYRUN_OK` (94 checks, 0 failed) on that code. |
| score-01 | `results/next_n12/score-01` on `n12-freeze-1`, 2026-11-06 09:53–10:34Z: every gate passed at T_14, T_15, T_13 until I-GRAD at T_13 (Spearman 0.999642, 4 of 98 u_S changed; rule 0). **DOES NOT COUNT.** `studies/next/n12/diag/score-01/`. |
| amendment A1 | the head's decision 2026-11-06 10:45Z: the gradient pass with deterministic algorithms and the sdpa math kernel (with bf16 reductions, the executor's addition for memory); I-GRAD's rule unchanged; flip records; the u_S resolution report; projection 1.83–2.42 GPU-h, hard kill 189 min. §14 A1. |
| freeze 1a | annotated tag `n12-freeze-1a` (tag object `936ce18ed`) on commit `8bde645942f2b2ab3b153fd15e73f418c3d368eb`, 2026-11-06T11:29:07Z. It holds A1 (§14) on top of freeze 1: `n12_score.py` (`grad_mode`, the I-GRAD records, the memory record), `n12.py` (`u_resolution`, `order_ranks`, the projection options), `n12_dryrun.py` (6 new checks), `infra/aws_pg/steps/next_n12_score-02.json`, `studies/next/n12/projection_a1.json`, `offline_gate/a1_memory.{py,json}`. The dry run printed `N12_DRYRUN_OK`, 100 checks, 0 failed, on `1791c3cfa`, whose code the tagged commit holds unchanged (`offline_gate/dryrun_a1.log`). |
| score-02 | `results/next_n12/score-02` on `n12-freeze-1a`, `i-0057dc2e305a57a3b` (g6e.2xlarge), 2026-11-06 11:36:05Z–13:25:52Z: complete, 73 gates pass with every failing input failing, I-GRAD bit-identical, nothing parked; 1.824–1.829 GPU-h. Banks `studies/next/n12/banks/score-02/`. |
| freeze 2 | annotated tag `n12-freeze-2` (tag object `328078daf`) on commit `709464e3aab480b73516c5b055f2c572c077b53e`, 2026-11-06T13:28:55Z: `studies/next/n12/thresholds.json` (sha256 `85726798…`), from the freeze half only. G-VAL failed on the freeze half (ρ 0.0996). |
| verdict | run once 2026-11-06T13:29:28Z: `studies/next/n12/verdict_n12.json`. Headline `H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal DOES NOT COUNT; H_bridge DOES NOT COUNT; H_dt SUPPORTED` (§12). Not yet read by a different model (§7 step 5). |
| parents | `studies/next/09_PLAN.md` §3 card J12; `studies/next/07_PICKS/M2_SPEC.md` §1.2 (row "H_B SUPPORTED, H_sel_B BELOW_SELECTION": N12 moves up), §3.3, §3.8; `studies/next/m2/READ_head.md` (with the addendum); `studies/next/m2/READ_m2.md` §5; `studies/next/06_CRITIC.md` N2, N12, Part 6; `studies/next/03_IDEAS.md` N12; `studies/next/12_SHAPING_NOTE.md` §4 items 3–5 |
| method | repo `CLAUDE.md`; the `research-spec` skill; the format of `M2_SPEC.md` and `M1_SPEC.md`; M2's amendment A5 for the offline gate |
| vocabulary | `studies/next/README.md`. A run whose gate fails is labelled **DOES NOT COUNT**. |
| scope of the J12 card | N12 and M2's follow-up only. **N3 is out of scope**: the unite Arm 1 review is not closed (`studies/unite/results/RESULTS.md` line 3, "Not closed: awaiting review by a different model"; `studies/unite/results/reads/REVIEW_2026-11-01.md` §7, "Result status: not closed; this review does not close it"). N1 and N2 are not in this brief; N2's instrument (the gradient per unit activation validated against ablation) is built here and can be reused. |

### 0.1 The brief's requirements and where each sits

| requirement (head's brief, 2026-11-06) | where |
|---|---|
| N12: does the carrier win by the larger update toward the contexts, or because the leaders move away | §1.1–§1.3 |
| competitor set: B's top 20 at T_(k−1); does 20 need the forward again | §0.5, §2.5: no |
| per competitor: the change of its own B and of its rank between T_(k−1) and T_k | §2.1, the records (§9.4) |
| the critic's instrument: gradient per unit activation on E_c at T_(k−1), validated against ablation at T_(k−1) | §2.2, §3.5 (G-VAL) |
| carrier against competitors on these quantities | §1.3 (H_cf, H_signal), §2.1 |
| loss-matched contexts | §2.2 (the specific pull S = P − P′) |
| what a gradient at T_(k−1) can and cannot say about the realized update; what is measured instead | §2.3 |
| thresholds frozen on the freeze half (even neuron), test on the odd half | §6 |
| floors from matched random poles and from the twins; ceiling; does-not-count branches | §3.3, §3.4, §3.6, §5 |
| known number: M2's banked B values for 10 carriers at T_(k−1) | §3.1 (K0) |
| identity pair | §3.2 (I-ID, I-GRAD, the decomposition's identity) |
| M2's follow-up: twins matched on the number of distinct tokens, scored under B at T_(k−1) as M2's twins; hypothesis, refutation, prior | §1.3 (H_dt), §1.4, §3.3 |
| ten raw examples first, at zero GPU where possible, with a matched random-pole reading | §10; `studies/next/n12/samples/raw_examples_zero_gpu.{md,json}` |
| budget from M2's measured rates; hard kill 1.3 × projection; park order | §9.6 |
| code, a weight-free dry run with both sides of every gate on planted inputs and on the banked files at all 11 points | §9.2, §9.3; `N12_OFFLINE_GATE.md` |
| the three dimensions | §11 |

### 0.2 Facts read for this draft

| fact | value | source |
|---|---|---|
| model, points, banks | `allenai/Olmo-3-1025-7B`; T_5 … T_15 = steps 8k … 1,413,814; revisions and devices as in `m2.POINTS` | `M2_SPEC.md` §0.2 |
| the births | 153 (82 freeze, 71 test), split by the carrier's neuron parity; M2's twins for 149 (79, 70) | `studies/next/m2/births.json` (sha256 `fdca794b…`) |
| M2's result on the test half | carrier in its layer's top 10 under B at T_(k−1): 42 of 71 (0.592); twins 53 of 70; real − twin −0.171 [−0.286, −0.057]; at T_k 65 of 71 | `studies/next/m2/verdict_m2.json` |
| where the twin gap sits | the competitors' side: the 10th-best B on the carrier's positions is higher for real births (test medians 0.99663 against 0.98552); on repeated-token births (distinct-token share < 0.67) real − twin −0.397 [−0.513, −0.282] (both halves), on varied-token births −0.014 [−0.113, +0.085]; the split was chosen after the data | `studies/next/m2/READ_m2.md` §5; `m2/read/r5b_token_share.json` |
| M2's banked vectors | 757 files, 252,621,094 bytes; `a` [m, 11,008] fp32 at every exemplar position, per (unit, role); real units at pre, birth, end; twins at pre, end | `aws s3 ls --recursive --summarize` 2026-11-06; `infra/aws_pg/jobs/m2_score.py` step 6 |
| earlier L40S runs of the same windows | score-04 (T_10–T_15) and score-05 (T_7–T_15) vectors on S3 | `aws s3 ls` 2026-11-06 |
| M2's measured rates (score-06, L40S) | 0.2156 s per window forward in batches of 8 (9,526 tokens per second); fixed cost per point 61–130 s; weight download 14.61 GB in 45–82 s (178–321 MB/s); job time 2,214 s | `studies/next/m2/banks/score-06/m2/metrics.json` → `measured`, `points` |
| optimizer state | not public for stage 1 (verified only at `stage3/step11921`); no Adam update can be computed | `studies/next/00_ASSETS.md` §6 item 8 |
| unite Arm 1 | review not closed (see §0 scope) | `studies/unite/results/RESULTS.md`; `studies/unite/results/reads/REVIEW_2026-11-01.md` §7 |

### 0.3 Facts to verify at run time

| # | fact | verified by | rule if it differs |
|---|---|---|---|
| 1 | the AMI has torch 2.12.0+cu130, transformers 5.9.0 | G-SAME | the job stops in seconds |
| 2 | the L40S forward reproduces M2's banked B for 10 carriers at T_14 and M2's vectors at every point | K0, G-M2V | the job stops |
| 3 | the gradient pass is deterministic on the L40S | I-GRAD at T_13 | the job stops (score-01: not with the default kernels; A1, §14) |
| 4 | an ablation batch's clean row equals the forward's loss of that window | G-ABL-ID at every point with ablations | the job stops |
| 5 | the backward of one 2,049-token window fits in 48 GB from the lowest needed layer | the first point | the job stops (out of memory is not caught) |
| 6 | the gradient rate (0.54 s per window assumed, 1.0 s in the high case) | `measured.per_grad_s` at T_14 | the stop rule (§9.6) |

### 0.4 Already-done audit

| cell | prior measurement | path | what it gives N12 |
|---|---|---|---|
| naming the carrier before birth | M2: H_B SUPPORTED 0.592; H_sel_B BELOW_SELECTION | `studies/next/m2/verdict_m2.json` | the competitor set's definition (B at T_(k−1)) and the reason N12 moved up |
| the twin gap | on the competitors' side; repeated-token births; lateness does not explain it; giving a carrier its twin's competitors raises its hits from 41 to 49 | `m2/READ_m2.md` §5, `m2/read/r5_twins.json` | the prior for part B; the reason for the distinct-token twin |
| the carriers' own percentiles at T_(k−1) | as high as the twins' (test medians 0.99836 against 0.99783) | `m2/READ_head.md`; `READ_m2.md` §5 | the carrier is not behind on its own contexts; the contest is with other poles |
| the forward reproduces the bank | K0 and I-ID bit-identical in every M2 attempt on the L40S; score-05 against score-06 vectors exact at T_7–T_15 | `M2_SPEC.md` §12; `n12/offline_gate/gate_sides.json` → `G_M2V` | the known number has room to pass |
| the cross-device residual | Blackwell-banked histograms (T_7–T_14) against L40S forwards: G-EX over the A2 tolerance at T_13, T_11, T_10, T_9, T_7 on 1–2 % of entries | `verdict_m2.json` → `cross_device_gex` | enters every pole's B alike; a stated limitation (§8) |
| gradient supply between competing neuron sets | Pythia-410M, first ~165 steps: winners receive 2.25–2.73 times the loser's gradient supply, and "effective updates and weight magnitudes show no corresponding asymmetry" (Li and Li, arXiv 2609.01170, read in full by the `04_LITERATURE.md` author, TEXT) | `studies/next/04_LITERATURE.md` N12 | the warning that a gradient is not an update; the reason the realized change is measured |
| a future neuron's gradient share before recruitment | wdd e589 (Pythia-410m): class positions supply 0.215 of the recruit's gradient, matched non-recruits 0.086, random 0.039 (REPO-DOC) | `04_LITERATURE.md` N12 | a prior for the specific pull |
| unite Arm 1 upstream overlap | open; N3 out of scope | §0 | — |

The freeze-half look made before this spec: the ten raw examples (§10), M2's freeze-half example births. Nothing of
the test half was computed: no test-half competitor set, no test-half T_k value of any competitor, no twin's T_k
value, no gradient.

### 0.5 Objects, defined once

- **Births, halves, E_c, a⁺, B, the layer rank, a hit**: M2's (`M2_SPEC.md` §0.5). B_q(t) is pole q's own percentile
  at point t of its median a⁺ over the unit's positions. The layer rank of q is 1 + the number of the layer's other
  22,015 poles whose B is at least q's (ties count against q). A hit is a rank of 10 or better.
- **Units.** Per birth, three units share its layer, size and birth point k:
  - **real**: the carrier on E_c;
  - **pb**: M2's pseudo-birth twin (a random same-size group of another grid pole's own T_15 top-50, dated k by the same
    rule);
  - **dt**: the distinct-token twin, new here (§3.3): as pb, and in addition with exactly the birth's number of distinct
    current tokens. 140 of 153 births have one (73 freeze, 67 test; `units.json` → `dt_missing` lists the 13).
  The unit's **pole** is the carrier (real) or the twin's pole. Roles: **pre** = T_(k−1), **birth** = T_k; dt also
  **end** = T_15 (its construction control, as M2's twins).
- **d**, the number of distinct current tokens among a unit's positions: the distinct decoded `token` strings of the
  grid's banked T_15 `samples/contexts.jsonl` at those positions (the reader's measure, `READ_m2.md` §5).
- **Competitor set C**: the 20 poles of the layer with the highest B at T_(k−1) on the unit's positions, B > 0, the
  unit pole's neuron (both signs) excluded (`n12.competitors`). A frozen rule on T_(k−1) data only, so the set is
  fixed before the birth, as the critic asks (`06_CRITIC.md` N12 fix and Part 6: "run only as the prediction in M2").
  Computed by the job from its own forward; for real and pb units it equals the set computed from M2's banked vectors
  (G-M2V makes them the same arrays).
- **Leaders**: the members of C whose B at T_(k−1) exceeds the unit pole's.
- **Counterfactual ranks** (`n12.cf_ranks`), all layer ranks under B on the unit's positions:
  r0 at T_(k−1); r1 at T_k; **r_c** = the pole's rank with its own T_k value among the others' T_(k−1) values (only
  the pole moved); **r_o** = its rank with its own T_(k−1) value among the others' T_k values (only the others moved).
  **F** = log10 r_o − log10 r_c. F > 0: the pole's own move buys more rank than the others' move does.
- **Drop** of the competitors: the median over C of log10 rank_q(T_k) − log10 rank_q(T_(k−1)) (> 0: the leaders lose
  rank). Every competitor starts at a layer rank of 1–21, for real units and twins alike.
- **G_n(i)** = ∂L/∂h_n(i): the gradient of L, the summed next-token cross-entropy of the window (input index j
  predicts input j + 1, j = 0 … 2,047), with respect to h_n(i), the down-projection input of neuron n at input index i,
  at T_(k−1). The exemplar at the window's last index (i = 2,048) has no downstream loss and is left out.
- **Pull** of pole q = (n, sign) at i: s_q(i) = −sign_q · G_n(i) · q999_q(T_(k−1)), sign +1 for the + pole and −1 for
  the − pole. Units: nats per own-q999 of activation. s > 0: a step of gradient descent on this window would raise the
  pole's activation in its own direction at i. q999 puts poles of different scales in one unit and does not use the
  pole's activation at i, which is selected high for the carrier (the critic's objection to activation × gradient).
- **P_q** = mean over the unit's positions of s_q; **P′_q** = the same over the loss-matched positions; **S_q = P_q −
  P′_q**, the specific pull.
- **Loss-matched positions** (`n12.loss_matched`): for each exemplar index i, the 16 indices j of the same window,
  1 ≤ j ≤ 2,047, whose next-token loss ℓ(j) at T_(k−1) is nearest ℓ(i), none within 8 tokens of any unit position of
  that window; ties by the smaller j.
- **u**: the share of C at or above the unit pole under a score (0: above every competitor; 0.5 is the share expected
  if the pole were one more draw from C). u_S, u_P, u_P′ under S, P, P′; u_gain under the realized gain
  log10 r(T_(k−1)) − log10 r(T_k).
- **Ablation**: the neuron's down-projection input set to 0 at the unit's positions of a window; ΔL = L(ablated) −
  L(clean), summed over the unit's windows. Its first-order prediction is ΔL̂ = Σ_i −h_n(i) G_n(i).

---

## 1. Question, decision, hypotheses

### 1.1 The question

One checkpoint before a cluster's birth the carrier is often not the leader on the cluster's contexts: other poles of
its layer lead (M2), and they lead most on repeated-token births. One checkpoint later the carrier leads in 65 of 71
test births. Does the carrier win because its own response to those contexts rises past the leaders (the competition
decided by what the training signal did to the carrier), or because the leaders at T_(k−1) move away? And did the
training signal at T_(k−1) already favour the carrier: was its pull toward those contexts, per unit of its own
activation and beyond loss-matched contexts, larger than its competitors'?

### 1.2 The decision this run makes

| outcome (test half) | reading | next |
|---|---|---|
| H_cf OWN_RISE, H_away_sel_pb AT or BELOW | The carrier wins by its own rise; the leaders fall no more than any top-20 selected the same way. | The weight change behind the rise is the next object (N1, with the onsets located first, `06_CRITIC.md` Part 6). |
| H_cf OWN_RISE, H_away_sel_pb ABOVE | Both: the carrier's rise decides the rank, and the leaders of real births also move away more than selection gives. | A follow-up on where the leaders go (their own top-50 at T_k). |
| H_cf OTHERS_FALL | The leaders' fall is what lets the carrier in. | With ABOVE: the leaders move away specifically at real births; the "competition" is a release. With AT: the fall is what selection of the top 20 gives. |
| H_cf MIXED | Neither move alone accounts for more of the rank. | Report both components with their intervals; no mechanism claim. |
| H_signal SUPPORTED (and G-VAL passes) | At T_(k−1) the training signal on these contexts already pulls the carrier more than its competitors. With H_bridge FORETELLS: the pull at T_(k−1) foretells who gains rank. | A prediction at T_(k−2) and the question whether the pull beats B as a forecast (`06_CRITIC.md` M2's gradient score). |
| H_signal REFUTED | The carrier's pull at T_(k−1) is not above its competitors' beyond the matched floor: the signal at that point does not single out the winner. | The winner is decided between the points, not visible in one gradient; N1 with finer checkpoints. |
| H_signal DOES NOT COUNT (G-VAL fails) | The first-order instrument does not track ablation in this model at these points. | Report the realized labels only; the gradient readout needs a different instrument (finite differences or ablation per pole). |
| H_dt SUPPORTED | Token repetition explains M2's real − twin gap: twins matched on distinct tokens hit as rarely as the carriers. | M2's BELOW_SELECTION reads as "repeated-token contexts are contested", not as something about jobs. |
| H_dt REFUTED | The gap persists with matched tokens: real births are more contested than matched twins. | Coherence or lateness beyond token repetition; the next control matches lateness. |
| DOES NOT COUNT | No claim; the failed gate and its value are reported. | fix, fresh attempt prefix |

### 1.3 Hypotheses and the observation that refutes each

On the test half's units at the pre and birth roles, labels as `n12.verdict` computes them (§5).

| id | hypothesis | predicts | refuted by |
|---|---|---|---|
| **H_rise** (the head's "decided by the training signal", realized form) | The carrier wins by its own rise toward the contexts. | **H_cf = OWN_RISE**: mean F over real units, bootstrap 95 % interval above 0. | the interval of mean F below 0 (OTHERS_FALL). |
| **H_away** (the head's "the leaders move away") | The carrier wins because the leaders move away, more than a selected top 20 would. | H_cf = OTHERS_FALL, and **H_away_sel_pb = ABOVE_SELECTION**: the paired drop real − pb has its interval above 0. | H_cf = OWN_RISE, or H_away_sel_pb AT or BELOW (the leaders of real births fall no more than the twins' leaders). |
| **H_signal** (the signal at T_(k−1)) | At T_(k−1) the carrier's specific pull on E_c exceeds its competitors'. | mean u_S over real units with its upper bound below the bar (§5 step 6; at most 0.5). | mean u_S at or above the bar (REFUTED). DOES NOT COUNT if G-VAL fails on either half. |
| **H_sel_signal_pb** (the boundary of H_signal) | The carrier's advantage in pull is larger than a twin pole's over its own competitors. | paired u_pb − u_real, interval above 0 (ABOVE_SELECTION). | interval containing 0 (AT) or below 0. |
| **H_bridge** (does the gradient foretell the realized move) | Competitors with more specific pull at T_(k−1) lose less rank by T_k. | mean over real units of Spearman(S_q, gain_q) over C, interval above 0 (FORETELLS). | interval containing 0 (UNDECIDED) or below 0 (CONTRARY). |
| **H_dt** (M2's follow-up) | Token repetition produces M2's real − twin gap. | **H_sel_dt** (paired hit_real − hit_dt under B at T_(k−1)) AT or ABOVE, and **H_dt_pb** (paired hit_dt − hit_pb) LOWER. | H_sel_dt = BELOW_SELECTION: twins matched on distinct tokens still hit more often than the carriers. |
| **H_0** (the boring one) | Everything N12 measures is what the selection gives: the dating rule makes the pole rise, the top-20 selection makes the leaders regress. | every real − twin contrast AT_SELECTION. | any real − twin contrast ABOVE or BELOW. |

What these labels can and cannot show. H_cf is a within-unit comparison of two counterfactual ranks; it needs no
twin, and its sign is not forced by the dating rule: the rule makes the carrier's median cross its own q999 at T_k
(B about 0.999), while the leaders of a repeated-token birth sit at B 0.9999 and above (bar median 0.99957, §0.2), so
the carrier's own rise alone need not pass them. The size of F does depend on how far below the top the carrier starts.
That is why the twin contrast is made on the drop of the competitors, whose starting ranks are 1–21 for every unit,
and not on F. Room: on planted outcomes of the test half's size every label reaches each branch (`n12.room_check`,
dry run SCORE).

### 1.4 The author's prior, logged before any test-half N12 quantity exists (2026-11-06)

| quantity | prior | basis |
|---|---|---|
| H_cf | OWN_RISE 0.75, MIXED 0.20, OTHERS_FALL 0.05 | freeze-half raw examples: F > 0 in 8 of 10, F = 0 in the 2 births at rank 1 throughout (§10) |
| H_away_sel_pb | AT 0.45, ABOVE 0.35, BELOW 0.20 | the leaders of the ten examples drop 0.27–2.34 decades (median over C); a selected top 20 regresses; no twin value read |
| H_signal | SUPPORTED 0.35, UNDECIDED 0.35, REFUTED 0.30 | wdd e589's 0.215 against 0.086 is a within-pole share at 410M; the grid points are 100k–440k steps apart |
| G-VAL passes on both halves | 0.6 | single-neuron zero ablation in bf16 on 7B; the fp64 tiny-model check gives Spearman 0.990 (dry run INSTR); the RelP warning for MLP outputs (Pearson 0.006 in GPT-2 Large, `06_CRITIC.md` N2) |
| H_bridge | FORETELLS 0.30, UNDECIDED 0.55, CONTRARY 0.15 | 2609.01170: a gradient advantage did not reach the updates |
| H_dt | SUPPORTED 0.50, UNDECIDED 0.30, REFUTED 0.20 | the reader's post-hoc split, logged as the prior: varied-token births real − twin −0.014 [−0.113, +0.085]; repeated-token −0.397 [−0.513, −0.282] (`m2/read/r5b_token_share.json`, both halves, chosen after the data) |
| K0 bit-identical | 0.9 | 10 of 10 carriers bit-identical between score-05 and score-06 (gate sides) |
| I-GRAD bit-identical | 0.6; within its rule 0.95 | the sdpa backward on the L40S is not shown deterministic anywhere in the repo. score-01: not bit-identical, outside the rule (4 of 98). A1's prior is in §14 |

### 1.5 Literature (query terms recorded)

Searches of 2026-11-06 (WebSearch): (1) `which neuron wins a feature during training competition between neurons
gradient language model checkpoints`; (2) `"gradient starvation" neural networks feature competition Pezeshki`;
(3) `attribution patching MLP neuron zero ablation agreement first-order approximation AtP* failure modes`;
(4) `neurons "compete" for features during training "winner-take-all" emergence monosemantic neuron training dynamics
checkpoints 2025 2026 arXiv`. Earlier record: `studies/next/04_LITERATURE.md` N12 (six queries, the arXiv API).

- Closest: Li and Li, arXiv 2609.01170 (Pythia-410M, first ~165 steps; gradient supply to winning and losing task
  neuron sets; the asymmetry did not reach the updates). Read in full by the `04_LITERATURE.md` author; its numbers are
  quoted from that record, not re-read here.
- Gradient starvation, Pezeshki et al., arXiv 2011.09468: the name; a theory of feature imbalance under cross-entropy.
  Search summary only; no number is cited.
- AtP*, Kramár et al., arXiv 2403.00745: first-order attribution has false negatives from attention saturation and from
  cancellation; search summary only. It is the reason G-VAL compares the first-order prediction with real ablation.
- Not found in four searches: the gradient to every pole of a layer on one job's contexts, before a dated birth, at 7B,
  with the realized rank change of every competitor and a selection-matched control. Four searches are not a hole in
  the literature.

---

## 2. The measurement and the instrument

### 2.1 In plain words

- **Items.** M2's 153 births (82 freeze, 71 test); per birth the real unit and its twins (pb 149, dt 140).
- **Model.** `allenai/Olmo-3-1025-7B` at the 11 stage-1 branches T_5 … T_15 (`m2.POINTS`), on one L40S.
- **Quantities, per unit.**
  - Realized, from the forward at T_(k−1) and T_k: the pole's B and layer rank at both points; each competitor's B
    and layer rank at both points (ΔB and Δrank per competitor); r0, r1, r_c, r_o, F; the drop of the competitors; the
    pole's u_gain.
  - Signal at T_(k−1), from the gradient pass: P, P′, S for every pole of the layer; u_S, u_P, u_P′; Spearman of S
    against the realized gain over C; the mean |ℓ(j) − ℓ(i)| of the loss match.
  - Validation at T_(k−1), real units only: the measured ablation ΔL and its first-order prediction ΔL̂ for the carrier
    and its top 3 competitors under B.
  - Part B at T_(k−1): the dt pole's layer rank under B (a hit or not), exactly as M2 scored its twins.
- **Pairing.** Real and its twins pair by birth. Every pole of a unit is read from the same forward and the same
  backward.
- **Summary.** Means over units with a percentile bootstrap 95 % interval over units (10,000 resamples, seed 0);
  paired differences by birth (`m2.paired`, the same bootstrap); Wilson intervals for hit shares.

### 2.2 The instrument, exactly

`infra/aws_pg/jobs/n12_score.py`, one launch. Per point t, in the order T_14, T_15, T_13, T_12, …, T_5:

1. topk/ and stats/ of the needed layers from S3 `results/neurons/w3/grid/<dir>/` (`m2_score.layer_files`).
2. The branch download, G-REV, G-CFG, `neurons_harvest.load_model` (bf16, sdpa), every parameter frozen, G0 (u × 1.1
   its failing input; the activation swap reported, M2's A5).
3. At T_14 first: K0 (§3.1).
4. The forward: every window holding a position of a (unit, role) at t, BOS + 2,048 corpus tokens, batches of exactly
   8 (the last padded with its own last window), all logits. a = act(g)·u in fp32 for every neuron of each needed layer
   at each needed position (`neurons_harvest.MLPTap`), and each window's summed loss in fp32. Then G-M2V, G-EX
   (gated at T_5, T_6, T_15, reported at T_7–T_14, M2's A3) and G-TOPK.
5. Per (unit, role): `m2.scores` and `m2.record` (an M2-format record, so part B is scored exactly as M2's twins), the
   arrays (B [11,008, 2], a) to `arrays/tNN/`. At pre: C.
6. The gradient pass at pre: each window holding a pre-role position, alone (batch 1), all logits, L = summed
   cross-entropy; the down-projection input of the window's lowest needed layer becomes a detached leaf that requires a
   gradient, higher needed layers keep theirs; one backward. G and h at each exemplar index, the mean G over its 16
   loss-matched indices, and ℓ (`grad_capture`). P and P′ to `arrays/tNN/<unit>_pull.npz` with the gradient rows.
   A1 (§14): this pass runs with deterministic algorithms and the sdpa math kernel (bf16 reductions).
7. Ablation at real pre units: per window of the unit, one row per pole (the carrier and its top 3 competitors under
   B), the neuron's down-projection input zeroed at the unit's indices in that window; one clean row per unit; batches
   of 8 rows (`ablate`). ΔL against the forward's loss of the same window; ΔL̂ from step 6.
8. At T_13: I-ID and I-GRAD (§3.2).

After the last point, one record per unit with both roles (`n12.unit_record`) to `n12_<half>.jsonl`, and the raw
examples. `metrics.json` carries gates, counts and timings, never a rank, a hit or a label.

### 2.3 What a gradient at T_(k−1) can and cannot say

- **It can say** how the loss on these few windows, under the weights of T_(k−1), changes per unit of each pole's
  activation at the job's positions, relative to positions of the same windows with the same loss. All poles of a layer
  share the same error vector at a position, so the comparison among them is not moved by the error level (the
  critic's "the error falls for everyone"): only the alignment of each pole's output path with that error and its
  scale enter. Its validity as a local derivative is checked against finite differences offline and against zero
  ablation on the instance (G-VAL).
- **It cannot say** what the weights did between T_(k−1) and T_k. That interval is 4,000 to 440,000 optimizer steps
  (T_5 → T_6 to T_14 → T_15), of other data, under AdamW: each parameter's step is the gradient's running mean divided
  by its running root mean square, with weight decay, and the weights move far along the way, so the gradient at
  T_(k−1) is not the gradient along the path. The stage-1 optimizer state is not public (`00_ASSETS.md` §6 item 8), so
  no Adam update is computed and none is assumed. In Pythia-410M a 2.25–2.73-fold gradient advantage did not reach the
  updates (2609.01170).
- **Measured instead of assumed.** The realized change on the job's contexts: every competitor's ΔB and Δrank, the
  carrier's, and their counterfactual decomposition (H_cf, H_away_sel). Whether the signal at T_(k−1) foretells it is
  itself a test (H_bridge). The realized weight change of rows and columns is N1's object and is not measured here.

### 2.4 Settings identical to the banked harvest

As `M2_SPEC.md` §2.4 for the forward (weights, corpus, windows, batch 8, bf16, sdpa, a in fp32 from the bf16
projections, versions, one L40S). New here: all logits (the loss), the batch-1 gradient pass and the batch-8 ablation
rows. The forward's activations do not depend on the logits kept. The gradient pass's own h and G are used only for
the pull and ΔL̂; every B comes from the batch-8 forward.

### 2.5 Does it run at zero GPU on banked arrays?

- **The competitor set and the realized part for real units: yes.** M2's vectors hold every neuron at E_c at T_(k−1)
  and T_k; with the banked histograms they give every pole's B. The ten raw examples do this (§10). The launch
  recomputes it from its own forward so that one forward serves every number, and G-M2V shows the two agree.
- **The twins' T_k: no.** M2 forwarded twins at T_(k−1) and T_15 only.
- **The dt twins: no** (new positions).
- **The gradient and the ablation: no.**

---

## 3. Controls and references

Rule 4 of "Instruments and controls": fit on both sides (K0 at T_14 and G-M2V at every point, against M2's banked
values; G-VAL against ablation), a positive control that fired (the movement control and G-VAL; the dt construction
control at T_15), and the identity pair (I-ID, I-GRAD, the decomposition's identity).

### 3.1 The known number K0

At T_14, before any other forward: the windows of the first 10 real carriers (by unit id) whose pre point is T_14
(L00N9356−:c8, L02N1367−:c7, L02N4649+:c3, L02N7380+:c2, L03N6372+:c10, L03N7746+:c0, L04N7919−:c0, L05N8714−:c5,
L06N2185+:c5, L06N9844+:c1; `gate_sides.json` → `K0.units`) run first. Pass: each carrier's B recomputed from the new
forward equals M2's banked pre-role B (`ranks_*.jsonl` → `value.B`) within 1e-6, and the carriers' activations at their
positions equal M2's banked vectors within the same-device tolerance of M2's A1 statistic (p99 of
|new − bank| / (0.05 + |bank|) ≤ 0.021159). Failing input: the same carriers' banked birth-role B (T_15); offline its
smallest |ΔB| is 8.995e-04, 900 times the tolerance. Pass side offline: score-05 and score-06 give bit-identical
activations for all 10 carriers, so their B is identical. If K0 fails, the job stops.

**G-M2V** repeats this at every point for every (unit, role) M2 banked (real pre and birth, pb pre, pb at T_15): all
11,008 neurons, the same statistic and tolerance; failing input: M2's vectors with the neuron axis rolled by one.
Offline: score-05 (and score-04) against score-06 exact at T_7–T_15; the failing input's p99 1.2–3,062 against
0.021159. T_5 and T_6 have no second run; the pass side there is assumed (same-device bit-identity, shown at 9 points).

### 3.2 The identity pair

- **I-ID** (as M2): at T_13 the pre-role windows run a second time in reversed order; the activations agree within the
  same-device tolerance and no unit pole changes its top-10 membership under B. Failing input: M2's birth-role vectors
  at T_14 of the same real units (offline p99 3.381).
- **I-GRAD**: at T_13 the gradient pass runs a second time in reversed order. Pass: Spearman between the two passes'
  G over every captured entry ≥ 0.999 and no unit's u_S changes. Failing input: the second pass with the neuron axis
  rolled by one. No gradient was ever banked, so both sides are shown only in the dry run (tiny world: exact; the rolled
  axis −0.006; a planted noisy second pass stops the job). A1 (§14): both passes deterministic; the job also records
  the largest |ΔG| and |ΔS| and every unit whose u_S or S order changed; a planted second pass with one moved entry
  (Spearman ≥ 0.999) must fail the gate.
- **The decomposition's identity**: every record carries r_c, r_o and F computed with T_k replaced by T_(k−1); they must
  be r0, r0 and 0 (freeze refuses and the verdict reads DOES NOT COUNT otherwise). Shown on the ten raw examples' real
  arrays and in the dry run.

### 3.3 The structured controls: the twins (the boundary of every claim), and part B

- **pb, M2's twin** (`M2_SPEC.md` §3.3): same layer, size, birth point and selection operator; not a coherent cluster;
  typically dated for its pole. N12 scores it at T_(k−1) and T_k with its own C, pull and counterfactual ranks.
- **dt, the distinct-token twin** (M2's registered follow-up, `n12.dt_twin`): a group of another grid pole's own T_15
  top-50 non-echo ranks in the birth's layer, of the birth's size, with **exactly the birth's d**, dated k by the same
  rule. Candidates as M2's twin (the sample-0 clustered grid poles of the layer, another neuron, sorted, then
  `default_rng([20261106, layer, neuron, sign, c])`); per candidate 2,000 draws, each choosing d distinct tokens
  uniformly, one position of each, and the rest uniformly from the chosen tokens' other positions; the first draw dated
  k is taken. Every dt unit was re-dated, checked for layer, neuron, size, d and non-echo ranks in the dry run (REAL,
  140 of 140). Distinct-token shares below 0.67: real 81 of 153, dt 70 of 140, pb 16 of 149 (the reader's 78 and 16 of
  the 149 births with a twin reproduce).
- **Part B's measurement** (exactly M2's twin scoring): the dt pole's layer rank under B at T_(k−1) on its positions, a
  hit at rank ≤ 10; paired by birth with the carrier (H_sel_dt) and with pb (H_dt_pb). The bar on the dt positions (the
  10th-best other B) is reported beside the real and pb bars, the reader's mechanism. dt's construction control: at
  T_15 the dt pole's positions are its own top-50, so its precision at 10 under B must be at least 0.90 and its
  opposite pole's below (offline, M2's twins under the same operator: 79 of 79 and 0 of 79, freeze half). Below it,
  H_dt DOES NOT COUNT.
- **What a dt twin still differs in.** It is not an LLM-labelled cluster, and its date is typical for its pole while
  real births are later than chance (`M2_SPEC.md` §3.3). Matching d does not match lateness; H_dt REFUTED would point
  there.

### 3.4 The floor for the signal: matched random poles

For each real unit, M2's 100 poles of the layer nearest the carrier in log max_abs at T_(k−1) (the carrier's neuron
excluded, `m2.matched`). Each matched pole's u under S against the unit's C. One replicate draws one matched pole per
unit and takes the mean u; over 1,000 replicates (seed 0) the floor is the 2.5th percentile (lower is better). The bar
for H_signal is min(0.5, the freeze floor, the test floor). It answers whether any active pole of the layer, placed
among these competitors, looks as favoured by the signal as the carrier.

The floor for the realized readouts is the twins: a top 20 selected by the same operator on positions with no
contested job regresses by the same mechanism, and their poles rise by the same dating rule. Matched random poles have
B = 0 on E_c almost always (`READ_m2.md` §3, floors) and carry no realized information here.

### 3.5 The instrument's validation G-VAL (the positive control of the gradient readouts)

Over the (real unit, pole) pairs of a half (the carrier and its top 3 competitors: up to 328 freeze and 284 test pairs),
the Spearman ρ between ΔL̂ and the measured ΔL. Pass: ρ ≥ 0.7 (registered a priori; head's decision, 2026-11-06, before
freeze 1: raised from the draft's 0.5) and its bootstrap lower bound over units above the 97.5th percentile of ρ under
1,000 shuffles of ΔL̂. Failing input: ΔL̂ with its sign flipped. Ceiling: ρ > 0.999 reads DOES NOT COUNT until inspected
(the ablation would equal its own prediction). It must pass on the freeze half before freeze 2 records it and on the
test half at the verdict; otherwise every gradient label DOES NOT COUNT, is reported as exploratory, and the realized
labels stand. The head's reason for 0.7: a first-order instrument whose rank agreement with the ablation it predicts is
below 0.7 explains too little of the realized ordering to carry H_signal or H_bridge. The slope of ΔL on ΔL̂ through
the origin is reported. Offline: fp64 tiny model, autograd
against central finite differences (largest relative error 1.3e-03; the neighbouring neuron 0.26 at least) and zero
ablation against ΔL̂ (Spearman 0.990 on 24 pairs).

**G-ABL-ID**, the ablation harness's identity: each unit's clean row (nothing zeroed) must equal the forward's loss of
that window within 1e-3 nats; failing input: another window's loss. Dry run only (no ablation banked).

### 3.6 The ceilings, and the floor and does-not-count branches of the realized labels

- **Of the signal (H_signal)**: u_gain, the carrier's place among its competitors under the realized gain itself. The
  pull at T_(k−1) cannot foretell the realized move better than the move. If the upper bound of mean u_S lies below the
  lower bound of mean u_gain, the label is ABOVE_CEILING and the headline reads DOES NOT COUNT until inspected.
- **Of the realized decomposition (H_cf, H_away_sel)**: H_cf is a sign contrast within units; its zero is fixed by
  the identity (F = 0 when nothing moves), its reference is the twins' F (reported beside it), and it has no ceiling
  beyond the total change. Its does-not-count branches: the movement control (mean log10 r0 − log10 r1 over real units
  with its lower bound above 0 on each half, so there is a win to decompose; failing input the identity, 0; offline on
  the freeze half from M2's ranks 1.056 [0.839, 1.285]) and the identity on every record. H_away_sel's floor is the
  twins' drop by construction (it is a paired difference against it); its ceiling is the competitors' full fall
  (at most log10 22,016 = 4.34 decades for a competitor at rank 1), which no difference can exceed.

### 3.7 The loss-matched contexts (the critic's comparison)

The critic asks for the competitors' share on the job's contexts against loss-matched non-job contexts at the same
point. P is the pull on E_c, P′ the same pole's pull on loss-matched positions of the same windows, S their difference.
u_P and u_P′ place the carrier among its competitors on each set; if the carrier leads on both, its advantage is
general to these windows, and S removes it. The loss match is reported per unit (the mean |ℓ(j) − ℓ(i)|).

### 3.8 Monitoring level and the read/edit rule (`12_SHAPING_NOTE.md` §4 items 4–5)

Every quantity reads one pole at a fixed identity in one layer (the L1 type). B reads the input side (a from the gate
and up rows). The pull reads the output side's path to the loss (G = ∂L/∂h runs through the down column and the layers
above); the two are kept as separate quantities. Nothing is edited along a direction; the ablation zeroes the neuron's
down-projection input, the read side, at single positions, and is used only to validate the gradient. A label that
says "no pole leads" says nothing about a job carried by a direction spread over neurons.

### 3.9 The matched random-pole reading (`12_SHAPING_NOTE.md` §4 item 3)

Beside each raw example: the first matched pole's own top-3 contexts at T_(k−1) and its rank under B (from M2's banked
raw examples at zero GPU; written again by the job).

---

## 4. Gates

Every instance gate is asserted in the job: the step exits non-zero, the pipeline aborts, STATUS shows `exit≠0`. Each
gate's failing-input value is written to `out/score/n12/metrics.json`; the verdict reads every gate again.

| gate | where | pass | registered failing input (shown failing offline or in the dry run) | if it fails |
|---|---|---|---|---|
| G-SAME | job start | torch 2.12.0+cu130, transformers 5.9.0 | transformers 0.0.0 | stop |
| INPUTS | job start | `units.json`, M2's two ranks files and the 16 exemplar files have their registered sha256 | a zero sha256 | stop |
| G-CORPUS | corpus fetch | tokens sha256 `2aca80ca…` | a wrong sha256 | stop |
| G-REV, G-CFG | each point | as M2 | a wrong revision; model_type renamed (in-job), width changed (dry run) | stop |
| G0 | each point | as M2's A5 | u × 1.1 | stop |
| K0 | T_14, first | §3.1 | M2's birth-role B; M2's B moved by 1e-3 (dry run) | stop |
| G-M2V | each point | §3.1 | the rolled neuron axis; M2's vectors × 1.1 (dry run) | stop |
| G-EX | each point | gated at T_5, T_6, T_15; reported at T_7–T_14 (M2's A3) | the neighbouring point's bank; T_13's bank in T_14's place (dry run) | stop where gated |
| G-TOPK | each point | as M2 | the neuron axis shifted by one | stop |
| G-ABL-ID | each point with ablations | §3.5 | another window's loss; the clean row ablating the whole MLP (dry run) | stop |
| I-ID | T_13 | §3.2 | M2's birth-role vectors at T_14; the second forward × 1.1 (dry run) | stop |
| I-GRAD | T_13 | §3.2 | the rolled neuron axis; a noisy second pass (dry run) | stop |
| G-VAL | freeze, verdict, per half | §3.5: ρ ≥ 0.7 (head's decision, 2026-11-06, before freeze 1), lower bound above the shuffle's 97.5th percentile, ceiling 0.999 | the sign-flipped prediction; an unrelated prediction (dry run) | gradient labels DOES NOT COUNT (reported as exploratory) |
| usable units | freeze, verdict | ≥ 50 real units per half with both roles | 30 test units (dry run); the tiny job's 20 (dry run) | DOES NOT COUNT (missing); twins below 50: their labels NOT_RUN |
| movement | freeze, verdict | §3.6 | the identity | DOES NOT COUNT |
| the decomposition's identity | freeze, verdict | §3.2 | a planted broken record (dry run) | DOES NOT COUNT |
| dt construction control | verdict | §3.3 | the opposite pole | H_dt DOES NOT COUNT |
| room | freeze | every label branch reachable at n = 71 | (asserted) | no freeze |
| freeze records unchanged | verdict | sha256 of `n12_freeze.jsonl` = the frozen one | one byte appended (dry run) | the verdict refuses |

---

## 5. Labels and the verdict, as the script runs it

`python -m infra.pg.n12 verdict --banks studies/next/n12/banks --thresholds studies/next/n12/thresholds.json --out studies/next/n12/verdict_n12.json`

1. The frozen freeze records' sha256 must match.
2. Gates: every instance gate passed, every failing input failed, the job complete; else **DOES NOT COUNT: gates**.
3. Real test units with both roles ≥ 50, else **DOES NOT COUNT: missing**.
4. The movement control's lower bound above 0, and the decomposition's identity on every record; else **DOES NOT COUNT**.
5. **H_cf**: mean F over real units: OWN_RISE if the interval's lower bound > 0; OTHERS_FALL if its upper bound < 0;
   MIXED otherwise.
6. **H_away_sel_pb, H_away_sel_dt**: paired drop real − twin over births with that twin: ABOVE_SELECTION (lower bound
   > 0), AT_SELECTION, BELOW_SELECTION (upper bound < 0); NOT_RUN below 50 pairs.
7. **Gradient labels**, only if G-VAL passed on both halves with its failing input failing and neither above its
   ceiling; otherwise each reads DOES NOT COUNT with the reason:
   - **H_signal**: bar = min(0.5, freeze floor q025, test floor q025); SUPPORTED if the upper bound of mean u_S < bar;
     REFUTED if mean u_S ≥ bar; UNDECIDED otherwise; ABOVE_CEILING (§3.6) replaces it.
   - **H_sel_signal_pb, H_sel_signal_dt**: paired u_twin − u_real: ABOVE, AT, BELOW_SELECTION; NOT_RUN below 50.
   - **H_bridge**: mean Spearman(S, gain) over real units with a defined value: FORETELLS (lower bound > 0), UNDECIDED,
     CONTRARY (upper bound < 0).
8. **Part B**: H_sel_dt (paired hit_real − hit_dt) and H_dt_pb (paired hit_dt − hit_pb: HIGHER, SAME, LOWER), NOT_RUN
   below 50 dt pairs; **H_dt** SUPPORTED if H_sel_dt is AT or ABOVE and H_dt_pb is LOWER, REFUTED if H_sel_dt is BELOW,
   UNDECIDED otherwise; DOES NOT COUNT if the dt construction control fails.
9. Reported, no label: F for pb and dt; the two components log10 r0 − log10 r_c and log10 r0 − log10 r_o; the hit
   shares of r0, r1, r_c, r_o per kind; u_P, u_P′; Spearman(S, B at T_(k−1)) (whether the bridge could ride on the
   leaders' starting height); the median bars per kind; the loss match; the cross-device G-EX line and the degenerate
   poles (M2's A3 and A4 lines).
10. Headline: DOES NOT COUNT (reasons), or `H_cf …; H_away_sel_pb …; H_signal …; H_bridge …; H_dt …`. Every value is
    written whatever the headline.

The dry run produced every branch on synthetic records (`N12_OFFLINE_GATE.md`, VERDICT).

---

## 6. The split, freeze 2 and room

**The split**: M2's, fixed before any birth was read: the carrier's neuron even → freeze (82 births), odd → test (71).
A birth's twins follow it. **What was read before this spec**: the ten freeze-half raw examples (§10) and, from M2's
banks, the freeze half's starting ranks (real: 21 of 82 at rank 1, 43 above rank 10; pb: 40 of 79 at rank 1).

**Freeze 2** (`n12.freeze`) refuses unless the gates pass, the freeze half has ≥ 50 real units, the movement control
fires, the decomposition's identity holds and the room check passes. It writes `thresholds.json` with the freeze
floor of u (H_signal's bar), the freeze half's G-VAL, the sha256 of `n12_freeze.jsonl` and of the job's
`metrics.json`, and the freeze half's full summary labelled DISCOVERY.

**What is frozen on control data, and what is registered a priori.** Frozen on the freeze half: the matched floor of
u (its 2.5th percentile enters the bar), G-VAL's freeze-half pass. Frozen on banked control data before this spec:
K0's tolerance on B (1e-6, with the failing input 900 times above it and two L40S runs bit-identical), the same-device
tolerance of G-M2V, G-EX, I-ID (M2's A1, frozen on the T_14 device pair). Registered a priori, with the reason: G-VAL's
ρ ≥ 0.7 (head's decision, 2026-11-06, before freeze 1; the draft had 0.5) and ceiling 0.999 (no ablation of this model
exists to freeze them on; the shuffle floor is the data-driven part); I-GRAD's 0.999; G-ABL-ID's 1e-3 nats (expected 0
exactly); every paired label's 0.

**Room.** On planted outcomes at n = 71 every label reaches each of its branches (`n12.room_check`). With 71 units a
mean u resolves about ±0.03 at a per-unit spread of 0.13, and a paired hit difference about ±0.11 (M2's intervals), so
H_dt can separate the reader's repeated-token gap (−0.4) from 0.

---

## 7. Unblinding order

1. Freeze 1: `N12_DRYRUN_OK`; the head reviews; commit; tag `n12-freeze-1`.
2. L1 (`score-01`), the one launch. During the run only `n12/metrics.json` is read. After A1 (§14): `score-02` on
   `n12-freeze-1a`; banks under `studies/next/n12/banks/score-02/`.
3. Pull `n12/metrics.json`, `n12/n12_freeze.jsonl` and `n12/samples/raw_examples_freeze.jsonl`. Open the freeze
   examples (§10) and answer the sample questions in `studies/next/n12/JOURNAL.md`. Run `n12 freeze`, commit
   `thresholds.json`, tag `n12-freeze-2`.
4. Pull `n12/n12_test.jsonl` and `n12/samples/raw_examples_test.jsonl`. Run the verdict once. A rule found wrong
   afterwards gets a dated exploratory deviation; the frozen rule still decides the headline.
5. A fresh Opus agent, blind to this code, recomputes the headline numbers from `n12/arrays/` and the S3 grid arrays;
   the head's read is the different-model read. The result document comes after both.

---

## 8. Threats to validity

| # | threat | effect | mitigation |
|---|---|---|---|
| 1 | the carrier is chosen as the pole holding E_c at T_15, and dated by its own crossing at T_k | its rise is selected | the twins carry the same selection and the same dating; the twin contrast is on the competitors' drop, not on F |
| 2 | the competitors are selected as the top 20 at T_(k−1) | regression: they lose rank by selection alone | the twins' competitors are selected the same way and start at the same ranks 1–21 |
| 3 | F depends on the carrier's starting rank | real and twin F are not comparable | F is used within the unit only (H_cf); the twin contrast uses the drop |
| 4 | a gradient is not an Adam update; the interval is 4k–440k steps | the pull may not foretell the move | §2.3; H_bridge measures it; no update is assumed |
| 5 | first-order attribution can miss cancellations and saturation | the pull misreads | G-VAL against real zero ablation on both halves; the fp64 finite-difference check |
| 6 | the pull at one checkpoint, one window per position | noisy | the loss-matched subtraction; 20 competitors per unit; intervals over units |
| 7 | B at T_7–T_14 mixes Blackwell-banked histograms and L40S activations | percentiles shift slightly | the same for every pole of a layer (M2's A3 limitation); G-EX reported |
| 8 | B saturates near 1 (ties at 1.0 at T_k) | ties | ties count against the pole, as M2 |
| 9 | the dt twin is matched on d by the decoded token string | two token ids with one string merge | the reader's measure; rare; stated |
| 10 | the dt twin is not matched on lateness | H_dt REFUTED is ambiguous between coherence and lateness | stated (§3.3) |
| 11 | 153 births on 140 carriers | births of one pole are not independent | stated; the halves share no pole |
| 12 | one model, one training run | no replicate | stated |
| 13 | the gradient pass runs at batch 1 | its kernels differ from the batch-8 forward | its h and G are used only for the pull and ΔL̂; every B comes from the batch-8 forward |

---

## 9. Runbook

### 9.1 Checklist before the launch

- Read `.claude/skills/aws-pg/SKILL.md` and the docstrings of `infra/aws_pg/jobs/pipeline.py` and `n12_score.py`.
  Profile `fuelos`, region `us-east-1`, on every call.
- Worktree `../interp_research-bio`, branch `bio`; it needs its own `.pgvenv` for `./infra/aws_pg/pg`
  (`09_PLAN.md` J0). Commit named files only.
- Liveness: an instance id from `./infra/aws_pg/pg ps`, or `updated_utc` of `out/score/n12/metrics.json`. Never
  poll `job.log`.
- Cost and incidents: `infra/aws_pg/operations/2026-11-<dd>_next_n12.md`; science: `studies/next/n12/JOURNAL.md`.

### 9.2 Code (NEW or REUSED, each checked against the tree at `258c95815`)

| file | status | role |
|---|---|---|
| `infra/pg/n12.py` | NEW | units (dt twins), plan, competitors, counterfactual ranks, pull, loss match, records, labels, freeze, verdict, projection |
| `infra/aws_pg/jobs/n12_score.py` | NEW | the job |
| `infra/aws_pg/jobs/n12_dryrun.py` | NEW | the offline gate |
| `infra/aws_pg/steps/next_n12_score-01.json` | NEW | the step list |
| `studies/next/n12/offline_gate/gate_sides.py` | NEW | both sides of every gate from the banks |
| `studies/next/n12/samples/raw_examples.py` | NEW | the ten raw examples at zero GPU |
| `infra/pg/m2.py` (`scores`, `record`, `percentile`, `median_a_plus`, `ranks`, `matched`, `births_matrix`, `load_bank`, `nonecho_ranks`, `values_gate`, `TOL`, `tol_for`, `tol_ex`, `ex_is_gate`, `paired`, `wilson`, `gates_ok`, `cross_device_gex`, `degenerate_counts`) | REUSED, unchanged | M2's scores and gates |
| `infra/aws_pg/jobs/m2_score.py` (`Run`, `download`, `layer_files`, `windows_tensor`, `a_matrix`, `ex_entries`, `ex_gate`, `topk_gate`, `decode_ctx`, `VERSIONS`, `TOKENS_SHA`) | REUSED, unchanged | M2's job pieces |
| `infra/aws_pg/jobs/m2_dryrun.py` (`make_corpus`, `tiny_cfg`, `tiny_model`, `make_remote`, `fake_hub`, `tiny_births`, `launch`, `last_error`) | REUSED, unchanged | the tiny world |
| `infra/pg/neurons_harvest.py`, `infra/aws_pg/jobs/neurons_harvest.py`, `infra/pg/neurons_corpus.py`, `infra/pg/m1.py`, `infra/pg/diff_harvest.py`, `infra/aws_pg/jobs/pipeline.py` | REUSED, unchanged | as M2 |
| `studies/next/m2/read/s3range.py` | REUSED, unchanged | byte-range reads of single npz members |

### 9.3 The offline gate

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd /Users/abaybektursun/projects/interp_research-bio
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n12 units --out studies/next/n12/units.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n12 project --units studies/next/n12/units.json > studies/next/n12/projection.json
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n12/samples/raw_examples.py --out studies/next/n12/samples --cache $SP/n12/stats_cache
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n12/offline_gate/gate_sides.py --out studies/next/n12/offline_gate/gate_sides.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.n12_dryrun --bank /Users/abaybektursun/projects/interp_research_banks/neurons \
    --hf $SP/m2/hf --sides studies/next/n12/offline_gate/gate_sides.json \
    --examples studies/next/n12/samples/raw_examples_zero_gpu.json --out $SP/n12/dry
```
It must end with `N12_DRYRUN_OK`. The record is `N12_OFFLINE_GATE.md`.

### 9.4 S3 layout

```
results/next_n12/inputs/{units.json, m2_ranks_freeze.jsonl, m2_ranks_test.jsonl}
results/next_m2/inputs/exemplar_a_tNN.npz                                  (M2's, registered in m2.EXEMPLAR_SHA)
results/next_m2/score-06/out/score/m2/vectors/                             (M2's, read by K0, G-M2V, I-ID)
results/next_n12/score-NN/steps.json  STATUS  job.log  heartbeat.jsonl
results/next_n12/score-NN/out/score/n12/{metrics.json, ranks_freeze.jsonl, ranks_test.jsonl, n12_freeze.jsonl, n12_test.jsonl}
results/next_n12/score-NN/out/score/n12/samples/{raw_examples_freeze.jsonl, raw_examples_test.jsonl}
results/next_n12/score-NN/out/score/n12/arrays/tNN/<unit>_<role>.npz      (B, a; about 0.3 GB)
results/next_n12/score-NN/out/score/n12/arrays/tNN/<unit>_pull.npz        (P, P', G, H, Glm; about 0.6 GB)
```

### 9.5 Sessions

```
B=s3://pg-lineage-769063704696
```

**S1. Freeze 1 and pre-launch (laptop).** §9.3 to `N12_DRYRUN_OK`; the head's review; commit; tag `n12-freeze-1`;
record tag, SHA, UTC in §0. `./infra/aws_pg/pg ps` and `audit`: no stray instance. `aws s3 ls $B/results/next_n12/
--profile fuelos --region us-east-1` lists nothing. Upload:
```
aws s3 cp studies/next/n12/units.json $B/results/next_n12/inputs/units.json --profile fuelos --region us-east-1
for h in freeze test; do aws s3 cp studies/next/m2/banks/score-06/m2/ranks_$h.jsonl \
  $B/results/next_n12/inputs/m2_ranks_$h.jsonl --profile fuelos --region us-east-1; done
```

**S2. L1, the one launch (`score-01`; 1.64–2.33 GPU-h; job stop rule at 0.9 × 168 min; hard kill 182 min).**
```
aws s3 cp infra/aws_pg/steps/next_n12_score-01.json $B/results/next_n12/score-01/steps.json --profile fuelos --region us-east-1
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_n12/score-01/steps.json" \
    --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_n12/score-01 --budget-min 182 --json
./infra/aws_pg/pg watch results/next_n12/score-01
```
The capacity rule of M2 (every 180 s for six tries, then every 600 s until two hours after the first). The first gate
to read is K0; then `measured.per_grad_s` and `projection`. **score-02 (A1, §14)**: the same commands with `score-02`,
the step file `infra/aws_pg/steps/next_n12_score-02.json` (job `--budget-min 175`) and `--budget-min 189`; the capacity
rule of the head's J12 brief (every 180 s for six tries, then every 600 s with no time limit).

**S3. Freeze 2 (laptop, freeze half only).** Pull `metrics.json`, `n12_freeze.jsonl`, `samples/raw_examples_freeze.jsonl`
to `studies/next/n12/banks/n12/`. Open the freeze examples: for one real unit scored right and one wrong under H_signal
(u_S ≤ 0.25 and ≥ 0.75), check that the exemplars decode to the cluster's name, that the top-3 competitors' contexts
are what their B claims, that the loss-matched positions' losses are near the exemplar's and their texts are not the
job's, and that the ablation ΔL has the sign of ΔL̂ for the carrier. Answer in `studies/next/n12/JOURNAL.md`. Then
`PYTHONPATH=. python3 -m infra.pg.n12 freeze --banks studies/next/n12/banks --out studies/next/n12/thresholds.json`,
commit, tag `n12-freeze-2`.

**S4. Unblind (laptop).** Pull `n12_test.jsonl` and the test examples; run the verdict once; fill §12.

**S5. The independent recount** (a fresh Opus agent, its own code, reading `arrays/` and the grid arrays) and the
head's read.

### 9.6 Budget, stop rule, park order

| launch | contents | GPU-h, center | GPU-h, high |
|---|---|---|---|
| L1 `score-01` | boot, corpus, M2's vectors, 11 points: 6,208 window forwards, 2,801 gradient windows, 4,105 ablation rows, the identity re-runs at T_13, the final records | 1.64 | 2.33 |
| L1 `score-02` (A1, §14) | the same, the gradient pass deterministic on the math kernel; rates from score-01's measurements and the A1 estimate (`projection_a1.json`) | 1.83 | 2.42 |

From M2's measured rates (`n12.project`; `studies/next/n12/projection.json`): forward rows at 0.2156 s (center) and
0.25 s (high) per window; the gradient pass at 0.54 s per window (center: 2.5 × the forward rate, an estimate from the
forward-plus-backward cost from the lowest needed layer at batch 1) and 1.0 s (high); fixed cost per point 110 s and
160 s (M2 measured 61–130 s with the download); boot 164 s, corpus 60 s, M2's vectors 60 s, records 120 s. The
gradient rate is the only unmeasured one. At $2.24 per hour: $3.67–5.22; on g6e.4xlarge ($3.00424): $4.93–7.00.

**Hard kill: 182 min** (1.3 × 2.33 h), the launcher's `--budget-min`; for `score-02` **189 min** (1.3 × 2.42 h), the
job's stop rule at 0.9 × 175 min (A1, §14). **Stop rule, in the job**: after the first point
(T_14) it measures its fixed cost, its seconds per forward or ablation row and per gradient window; before each later
point it projects the rest; if the elapsed time plus that projection exceeds 0.9 × 168 min (151 min), this point and
every later one are PARKED and the job ends normally. Parking was planted in the dry run.

**Park order.** T_14 runs first (K0), then T_15, then T_13 down to T_5. A birth needs both T_(k−1) and T_k, so
parking from the bottom removes the fewest births: k = 6 (1 birth), k = 8 (6), k = 9 (4). The identity pair (T_13)
comes third. If parking leaves fewer than 50 real test units, the verdict is DOES NOT COUNT (missing) and a second
launch for the parked points is the head's decision.

### 9.7 Artifacts

- On S3 (§9.4): gates with their failing-input values, the M2-format records, the N12 records, the arrays, the
  examples.
- On the laptop: `studies/next/n12/{units.json, projection.json, samples/, offline_gate/, banks/, thresholds.json,
  verdict_n12.json, JOURNAL.md}`.
- Operations: `infra/aws_pg/operations/2026-11-<dd>_next_n12.md`.

---

## 10. The ten raw examples

**Zero-GPU part, written now** (`studies/next/n12/samples/raw_examples_zero_gpu.{md,json}`, by
`studies/next/n12/samples/raw_examples.py`). The ten births are M2's freeze-half example births (selected by M2's
frozen rule before N12 existed). Each record shows the 20 competitors at T_(k−1) with their B and layer rank at
T_(k−1) and at T_k, the carrier's r0, r1, r_c, r_o and F, the decoded exemplars, the own top-3 contexts of the top three
competitors and of the first matched random pole (from M2's banked raw examples), M2's twin's rank, and the dt twin.

| birth | k | size, d | r0 → r1 | r_c | r_o | F | carrier B T_(k−1) → T_k | leaders of 20 at T_(k−1) → T_k |
|---|---|---|---|---|---|---|---|---|
| L03N7574+ c6 (code/text subword tokens) | 12 | 5, 5 | 1,668 → 1 | 1 | 1,831 | +3.26 | 0.8776 → 0.9999 | 20 → 0 |
| L03N7746+ c0 (C corp) | 15 | 5, 2 | 193 → 1 | 1 | 316 | +2.50 | 0.9984 → 1.0 | 20 → 0 |
| L06N9844+ c1 (IRC logs) | 15 | 7, 2 | 80 → 1 | 1 | 86 | +1.93 | 0.9967 → 1.0 | 20 → 0 |
| L16N1206+ c1 (code mid-expression) | 14 | 9, 9 | 1 → 1 | 1 | 1 | 0 | 0.9984 → 1.0000 | 0 → 0 |
| L20N5782+ c1 (responsibility words) | 12 | 15, 7 | 3 → 1 | 1 | 4 | +0.60 | 0.9989 → 1.0000 | 2 → 0 |
| L07N8340+ c4 (rare-name fragments) | 14 | 5, 4 | 265 → 1 | 2 | 258 | +2.11 | 0.9807 → 0.9999 | 20 → 0 |
| L10N9052+ c2 (NPR "Copyright") | 14 | 5, 1 | 7,175 → 52 | 59 | 7,175 | +2.08 | 0.5849 → 0.9997 | 20 → 6 |
| L10N9794+ c2 (get/delete verbs) | 10 | 12, 9 | 1 → 1 | 1 | 1 | 0 | 0.9986 → 0.9999 | 0 → 0 |
| L19N7374− c1 (web boilerplate) | 13 | 10, 4 | 31 → 3 | 2 | 55 | +1.44 | 0.9986 → 1.0000 | 20 → 1 |
| L24N5066− c1 (company-name commas) | 12 | 15, 3 | 19 → 4 | 3 | 37 | +1.09 | 0.9989 → 0.9999 | 18 → 3 |

What they show:
- In every birth where the carrier starts below rank 1, its own T_k value placed among the others' T_(k−1) values
  ranks 1–3 (59 for NPR, whose T_k rank is 52), while the others' T_k values around the carrier's T_(k−1) value leave
  it at or below its starting rank (r_o ≥ r0 in 7 of 8). Read plainly: the carrier rises past leaders that were still
  there; the leaders' own moves alone would not have let it in.
- The leaders do lose rank: the median competitor drops 0.27–2.34 decades of layer rank (for example L10N3510+, the NPR
  leader that held all five exemplar positions in its top-50, goes from rank 1 to 3,154). A top 20 selected at T_(k−1)
  regresses; whether it regresses more at real births than at twins is H_away_sel, which needs the twins' T_k forward.
- The two births at rank 1 throughout (L16N1206+ c1, L10N9794+ c2) are varied-token births (d = 9 of 9 and 9 of 12);
  the contested ones include the repeated-token births (NPR d = 1, C corp d = 2, IRC d = 2).
- The first matched random pole of each birth ranks 346–22,016 under B (5 of 10 at 22,016, B = 0) and its own top
  contexts read as unrelated token families. The top three competitors' own top contexts at T_(k−1) are mostly other
  topics; three carry the job's token or a near one in other documents (C corp: L03N5771− on "C[[orpor]]ate"; the
  get/delete verbs: L10N5132− on "[[ GET]]_ALL_BRANDS"; the cable-glands commas: L24N234+ on "Cable"). Their high B on
  E_c is a response to these positions, not their own best contexts.
- PENDING (the job writes them): the pull, the ablation, the twins' T_k values, the dt twin's rank.

**GPU part, written by the job**: `n12/samples/raw_examples_freeze.jsonl` (the same ten births: the full N12 record of
the real, pb and dt units; the decoded exemplars; the top three competitors' and the matched pole's top-3 contexts at
T_(k−1); for three exemplars, two loss-matched positions with their losses) and `raw_examples_test.jsonl` (M2's ten
test example births, drawn without reading their data), opened at unblinding. Every record in `n12_*.jsonl` carries
its 20 competitors with their B, ranks and pull, so any instance can be shown.

---

## 11. The three dimensions

- **Predictive.** H_signal and H_bridge are predictions made at T_(k−1) about the realized order at T_k, scored once on
  the test half under rules frozen on the other. H_cf is a measurement, not a prediction. H_dt is a registered test of
  a post-hoc finding.
- **Novel.** The gradient per unit activation to every pole of a 7B layer on a dated job's contexts, validated against
  ablation, beside the realized rank change of each competitor and a selection-matched control, was not found in the
  searches (§1.5). The closest work measures gradient supply to neuron sets in the first 165 steps of Pythia-410M.
- **Practical.** Small and stated as such. If H_signal is SUPPORTED with G-VAL passing, a gradient at one checkpoint
  names which of the leading poles will hold a job, a cheaper forecast than a later checkpoint; the measured win would
  have to be shown against B (M2's 0.59) in a follow-up. If REFUTED, the same says one checkpoint's gradient is not a
  usable forecast at this spacing.

---

## 12. Prediction grid (filled at unblinding, whatever the result)

Filled 2026-11-06 after the verdict's one run, from `studies/next/n12/verdict_n12.json` (labels and test-half values)
and the two files it reads: `thresholds.json` (freeze half, frozen at `n12-freeze-2`) and the job's `metrics.json`
(gates). Headline: `H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal DOES NOT COUNT; H_bridge DOES NOT COUNT;
H_dt SUPPORTED`. H_0 (every real − twin contrast at selection) is refuted for the pb twin (the drop, BELOW; the hits,
−0.171) and holds for the dt twin on both labelled contrasts (AT, AT). u_S resolution (A1): noise 0 (I-GRAD's
largest |ΔS|), 0 of 82 freeze and 0 of 71 test real units within it; smallest gaps 4.5e-06 and 2.9e-06, medians
4.9e-04 and 6.2e-04; at score-01's nondeterministic noise the limit is stated from its count only (4 of 98).

| item | rule or prior | observed | verdict |
|---|---|---|---|
| K0 | §3.1 (prior bit-identical 0.9) | max \|ΔB\| 0.0 over 10 carriers, 39 activations exact; failing input 8.995e-04 (`metrics.json` → `gates.K0`) | pass, bit-identical |
| G-M2V at 11 points | §3.1 | p99 0, exact share 1.000 at every point; failing input p99 6.93–2,851 (`metrics.json`) | pass at 11 of 11 |
| G-EX, G-TOPK at 11 points | M2's A3 | G-EX gated and exact at T_15, T_6, T_5; reported at T_7–T_14, over the 0.038718 tolerance at T_13, T_11, T_10, T_9, T_8, T_7 (`verdict_n12.json` → `cross_device_gex`); G-TOPK p99 0–0.0153, failing inputs 1.02–1.80 | G-EX pass where gated, reported elsewhere; G-TOPK pass at 11 of 11 |
| G-ABL-ID | §3.5 | 0.0 nats at the 9 points with ablations (153 clean rows); failing input 2.17–1,054 nats | pass |
| I-ID, I-GRAD | §3.2 (I-GRAD bit-identical 0.6; A1's prior 0.9) | score-01 (default kernels): I-GRAD 4 of 98 u_S changed, DOES NOT COUNT. score-02 (A1): I-ID exact (8,035,840; failing input 3.381); I-GRAD exact share 1.000 of 7,419,392, 0 u_S changes, \|ΔS\| 0.0; failing input ρ 0.000393 | pass, both bit-identical (score-02) |
| G-VAL, freeze and test | ρ ≥ 0.7 (head's decision, 2026-11-06, before freeze 1; the prior pass 0.6 of §1.4 was logged for the draft's ρ ≥ 0.5) | freeze ρ 0.0996 [−0.0276, 0.2146], 328 pairs, shuffle q975 0.1103, slope 0.489 (`thresholds.json`); test ρ 0.0425 [−0.0594, 0.1411], 284 pairs, shuffle q975 0.1159, slope 0.326 (`verdict_n12.json` → `test.validation`); sign-flipped inputs fail | **fails on both halves**: every gradient label DOES NOT COUNT |
| movement control, both halves | lower bound > 0 | freeze 1.056 [0.840, 1.286]; test 0.854 [0.645, 1.074] decades | pass |
| floor of u, both halves | §3.4 | q025 freeze 0.4457, test 0.4225; the bar would be 0.4225 | frozen; not used (G-VAL) |
| H_cf, test | OWN_RISE 0.75 | mean F 0.951 [0.738, 1.180], 71 units; components: own move 0.879 [0.667, 1.106], others' move −0.072 [−0.117, −0.029] decades; F of pb 0.765, dt 0.942 | **OWN_RISE** |
| H_away_sel_pb, test | AT 0.45, ABOVE 0.35, BELOW 0.20 | drop real 0.939, pb 1.164 decades; paired real − pb −0.226 [−0.408, −0.056], 70 pairs | **BELOW_SELECTION** |
| H_away_sel_dt, test | (reported with its label) | drop dt 1.062; paired real − dt −0.113 [−0.295, +0.065], 67 pairs | AT_SELECTION |
| H_signal, test | SUPPORTED 0.35, UNDECIDED 0.35, REFUTED 0.30 | exploratory: mean u_S 0.527 [0.451, 0.600] against the bar 0.4225; u_gain 0.063 [0.041, 0.088]; u_P 0.497, u_P′ 0.456 | **DOES NOT COUNT** (G-VAL) |
| H_sel_signal_pb, test | (no prior) | exploratory: paired u_pb − u_real −0.039 [−0.134, +0.057]; dt −0.020 [−0.124, +0.084] | DOES NOT COUNT (G-VAL) |
| H_bridge, test | FORETELLS 0.30 | exploratory: mean Spearman(S, gain) −0.013 [−0.072, +0.046]; Spearman(S, B at T_(k−1)) 0.009 | DOES NOT COUNT (G-VAL) |
| H_dt, test (H_sel_dt, H_dt_pb) | SUPPORTED 0.50, UNDECIDED 0.30, REFUTED 0.20 | hits at T_(k−1): real 42 of 71, dt 41 of 67, pb 53 of 70; real − dt −0.030 [−0.149, +0.090] (AT); dt − pb −0.134 [−0.254, −0.015] (LOWER); real − pb −0.171 [−0.286, −0.057]; median bar real 0.9961, dt 0.9897, pb 0.9836 | **SUPPORTED** |
| dt construction control | ≥ 0.90 | 66 of 67 dt poles in their layer's top 10 at T_15 (0.985 [0.920, 0.997]); opposite pole 0 | pass |
| GPU-h | 1.64–2.33; kill 182 min (score-01); A1: 1.83–2.42, kill 189 min (score-02) | score-01 0.659–0.691 h; score-02 1.824–1.829 h (job 6,454.5 s; gradient 0.564 s per window, peak 36.04 GB); N12 2.483–2.520 h, $5.56–5.64 (operations note) | within the projection |

---

## 13. References

`06_CRITIC.md` N2, N12, M2, Part 6; `03_IDEAS.md` N12; `05_SCORING.md` (N12 row: 720; M2 adopted as the surviving
form of N12); `09_PLAN.md` J12; `M2_SPEC.md`; `m2/READ_head.md`; `m2/READ_m2.md`; `04_LITERATURE.md` N12 (2609.01170
and wdd e589 numbers, at that record's read levels); `12_SHAPING_NOTE.md` §4. No other citation carries a number here.

---

## 14. Amendments

### A1 (2026-11-06): a deterministic backward for the gradient pass

**Decision.** The head's, 2026-11-06 10:45Z, after `score-01` stopped at I-GRAD and before any record of either half
was opened. Written and frozen as `n12-freeze-1a` (§0) before `score-02`.

**Cause.** `score-01` (`results/next_n12/score-01`, on `n12-freeze-1`) passed every gate at T_14, T_15 and T_13 and
stopped at I-GRAD at T_13: Spearman of the two gradient passes 0.999642 (bar 0.999, passes), exact share 0.2129 of
7,419,392 entries, and 4 of 98 pre-role units' u_S changed (the rule allows 0). I-ID at the same point was exact, so
the batch-1 backward on the L40S is not bit-deterministic and the forward is
(`studies/next/n12/diag/score-01/metrics.json` → `gates.I_GRAD`, `gates.I_ID`; `diag/score-01/cause.md`). The run
DOES NOT COUNT.

**What changes** (`infra/aws_pg/jobs/n12_score.py`):
1. The gradient pass (§2.2 step 6) and its I-GRAD re-run (§3.2) run inside `grad_mode()`, entered at the start of each
   pass and left at its end: `torch.use_deterministic_algorithms(True)`; scaled-dot-product attention on the math
   kernel only (`torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`); and the math kernel's reductions kept in bf16
   (`torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`). The last setting is the executor's, not the
   head's; the reason is the memory estimate below. `grad_mode` asserts that none of the three is on outside it.
2. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set by `n12_score.main` before CUDA starts (the job asserts
   `torch.cuda.is_initialized()` is false at that point); `grad_mode` refuses without it.
3. Unchanged and outside `grad_mode`, with deterministic mode off and the harvest's sdpa kernel: the batch-8 forward
   (§2.2 step 4), G0, K0, G-M2V, G-EX, G-TOPK, the ablation rows (step 7), I-ID. Their equality gates are as
   registered.
4. I-GRAD's rule is unchanged: Spearman ≥ 0.999 over every captured entry and no change of u_S; failing input the
   neuron axis rolled by one. Recorded beside it in `metrics.json`: the largest |ΔG| and |G|, the largest |ΔS| over
   every unit's carrier and 20 competitors (the noise level of item "Resolution" below), and the count of changed units
   per half. Every unit whose u_S or S order (the carrier and its competitors ranked by S) differs between the passes
   is written to `n12/igrad_changed_<half>.jsonl` with its unit id, half, kind, both u_S values and both orders. The
   values go to these files, not to `metrics.json`, so that the only file read during the run carries no u_S; the test
   half's file is opened with the test records, after freeze 2.
5. The peak GPU memory of each point's gradient pass is written (`points.<t>.grad_max_mem_gb`), and the settings
   (`grad_pass_settings`).

**Determinism facts and their sources.** Torch is pinned to the harvest's 2.12.0+cu130 (G-SAME). Read for this
amendment: the docstring of `torch.use_deterministic_algorithms` in the CPU wheel torch 2.12.0, and the source at tag
`v2.12.0` of github.com/pytorch/pytorch (fetched 2026-11-06).
- The fused attention backwards are nondeterministic by default. `aten/src/ATen/native/transformers/cuda/attention_backward.cu`
  lines 120–129: the flash backward is called with `deterministic = false` unless deterministic mode is on (its warning
  reads "Flash Attention defaults to a non-deterministic algorithm"); lines 898–908: the memory-efficient backward may
  split the keys (`num_splits_key > 1`), which deterministic mode forces to 1; lines 210–214: the same warning for cuDNN
  attention, which `sdp_utils.cpp` (`check_cudnn_deterministic`, line 725) does not select in deterministic mode.
- Which kernel ran in score-01. On compute capability 8.9 the default order is flash, memory-efficient, math, cuDNN
  (`sdp_utils.cpp` lines 74–115: cuDNN goes first only on major 9 or 10). transformers 5.9 calls sdpa on these windows
  with no mask and `is_causal=True` (measured on the CPU, `offline_gate/a1_memory.json` → `sdpa_call_forms`), which
  flash accepts. So score-01's backward was most likely flash's nondeterministic one. Not verified on the L40S.
- The math kernel (`aten/src/ATen/native/transformers/attention.cpp` lines 826–913) is a composite of matmul, add,
  `_safe_softmax` and matmul, so its backward is autograd over those operations. It upcasts bf16 q, k and v to fp32
  unless `allow_fp16_bf16_reduction_math_sdp` is set (lines 842–859).
- `torch.use_deterministic_algorithms` (docstring): operations with a deterministic alternative switch to it (on this
  path: `torch.gather` on a CUDA tensor that requires grad, `index_select`, `index_add`, `scatter_add_`); operations
  without one raise `RuntimeError`; new tensors are filled (NaN) while `fill_uninitialized_memory` is on, a time cost.
  Among the listed raising operations only `torch.nn.NLLLoss` is near this path, and its check sits in
  `aten/src/ATen/native/cuda/NLLLoss2d.cu` lines 241–245 (the 2-D loss with a reduction); the job's cross-entropy is
  1-D with `reduction="none"`, and `aten/src/ATen/native/cuda/Loss.cu` has no such check. An operation that raises
  would stop the job at T_14's gradient pass.
- cuBLAS (`aten/src/ATen/cuda/CublasHandlePool.cpp` lines 158–199, 204–226): on compute capability 8.9 torch's default
  cuBLAS workspace is `:4096:2:16:8` (8.125 MiB); `:4096:8` gives 32 MiB, and the cuBLASLt workspace follows it
  (unified by default). Torch 2.5 refused cuBLAS calls in deterministic mode unless the variable was `:4096:8` or
  `:16:8` (`aten/src/ATen/Context.cpp` lines 172–195 at tag `v2.5.0`); at `v2.12.0` that check is gone from
  `Context.cpp`, and the handle pool gives each (handle, stream) pair its own workspace (`CublasHandlePool.cpp` lines
  276–301). So in 2.12 the only effect of the setting found in the source is the workspace size; it is set because the
  head's decision names it, and `grad_mode` checks it. The setting is process-wide, so the forward's matrix products
  also run with the larger workspace, which the banked harvest did not use. If cuBLAS then chooses another algorithm for some shape, the forward's bits change. K0 at T_14 (B within 1e-6
  and activations within the same-device tolerance, against M2's banks) is the first gate after the change and stops
  the job within about 4 minutes if so.

**Cost** (`studies/next/n12/offline_gate/a1_memory.py`, `a1_memory.json`: OLMo-3-7B's layer at one eighth of its
width, random bf16 weights, one 2,049-token window, CPU; every saved tensor scales with the width, so a full-width layer
saves 8 times as much).
- Memory saved for the backward per layer in the graph: 0.388 GB with the default kernel, 0.959 GB with the math
  kernel's default fp32 upcast (a 2,049 × 2,049 fp32 attention matrix for each of 32 heads, 0.537 GB, beside fp32
  q, k, v), 0.640 GB with bf16 reductions. A window whose lowest needed layer is 0 (64 of the 2,801 gradient windows,
  at T_14 and T_13) has 31 layers in the graph. With the 14.6 GB of weights and the logits' fp32 log-softmax and
  gradient, the estimated peak is 28.7 GB (default; score-01 fitted), **46.4 GB (math, fp32 upcast)** and **36.5 GB
  (math, bf16 reductions)**, before the transient of one attention layer (about 1–2 GB). The L40S reports 46,068 MiB
  (48.3 GB) less the CUDA context. The fp32 upcast leaves about 1 GB, so the job would likely stop out of memory at
  T_14; the bf16 reductions leave about 11 GB. This is why the executor added the third setting.
- Precision of the bf16 reductions, on the same CPU model: the gradient at layer 0 differs from the fused kernel's by
  a relative L2 of 0.0098 (2 layers) and 0.0124 (3 layers), against 0.0088 and 0.0112 for the fp32 upcast; the two
  math variants differ from each other by 0.0097 and 0.0123 (`a1_memory.json` → `grad`). At bf16 model precision the
  three kernels give gradients about 1 % apart, and neither math variant is closer to the fused one.
- Time, estimated (not measured): the math kernel moves the attention matrix through memory several times per layer
  in the forward (32 layers) and in the backward (31 − lowest layer, 16.7 on average), about 4–5 ms per layer at the
  L40S's 864 GB/s in bf16, plus the deterministic fill of new tensors. Estimate: +0.2–0.3 s per window over score-01's
  measured 0.371 s (T_14) and 0.385 s (T_13), so 0.65 s per window at the center and 1.1 s in the high case. The job
  measures it at T_14 (`measured.per_grad_s`), and the stop rule uses the measured value.
- Projection (`n12 project --per-window 0.2367 0.25 --grad-window 0.65 1.1 --fixed 116.5 160`, from score-01's
  measured 0.2367 s per forward or ablation row and 105.7–116.5 s fixed cost per point;
  `studies/next/n12/projection_a1.json`): **1.83–2.42 GPU-h, $4.10–5.42 on g6e.2xlarge. Hard kill 189 min** (1.3 ×
  2.42 h). The job's stop rule uses `--budget-min 175` (stop at 0.9 × 175 = 157.5 min; the step file
  `infra/aws_pg/steps/next_n12_score-02.json`), 14 minutes below the hard kill as in score-01 (168 against 182).

**Resolution of u_S.** In score-01's two nondeterministic passes 4 of 98 units' u_S changed. The instrument does not
resolve a difference in S between a carrier and a competitor that is smaller than the backward's noise. That noise
cannot be measured from score-01: the job banks the first pass's G only (the second pass of I-GRAD is never written),
so no |ΔS| between score-01's passes exists in any file, for either half. Nothing of score-01 was opened beyond its
gates and diagnostics (`metrics.json`, `STATUS`, `pipeline_status.json`, `heartbeat.jsonl`, `dmon.log`, `job.log`,
copied once after the end): no array, no ranks file, no record. The noise level is therefore score-02's own identity
pair, I-GRAD's largest |ΔS|, which reads 0 if the deterministic backward works. Freeze 2 (freeze half) and the verdict
(both halves) report `u_S_resolution`: the number of real units whose carrier's S lies within that noise of its
nearest competitor's (`n12.u_resolution`). With a noise of 0 this counts exact ties only, and the limit at the
nondeterministic noise level is stated from score-01's count alone (4 of 98, size not recorded). It is reported, not a
gate.

**The offline gate.** The gap: the dry run's backward runs on the CPU, where it is exact, and its only planted second
pass was noisy (Spearman 0.696, 29 flips); no planted input had a nearly exact second pass, so the zero-flip clause
was never shown against one. Planted now (`n12_dryrun.one_entry`): in the tiny world, the reversed second pass with
one gradient entry moved by 1e6 at the carrier neuron of a T_13 unit. It must keep Spearman ≥ 0.999, change u_S only
in units whose rows read that entry (and at least the chosen one), and fail I-GRAD. Also new: `grad_mode` on and off,
and its refusal without `CUBLAS_WORKSPACE_CONFIG`; the dispatched attention operations of the gradient pass (the math
kernel's `_safe_softmax`, no fused kernel; without `grad_mode` the fused kernel, the failing input); the I-GRAD records
in the tiny job; the resolution count on planted gaps and in the verdict. Record: `N12_OFFLINE_GATE.md`, section
"Amendment A1".

**Prior, logged before score-02** (executor, 2026-11-06): I-GRAD bit-identical with A1 0.9; K0 still bit-identical
with the 32 MiB cuBLAS workspace 0.85; the gradient pass fits in memory 0.95; measured gradient rate within 0.5–1.1 s
per window 0.8.
