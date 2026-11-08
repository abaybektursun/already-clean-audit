# N2V: N2's instrument re-validated in fp32 (J12, N12's registered follow-up). Pre-registration

**Answer.** One launch on one L40S answers the question. Under the coordinator's decision of 2026-11-07 (§0.6 item 8:
parts core, fd and jvp, so the own-next-token-loss readout has its prediction) it is projected at 6.53–9.78 GPU-h
($14.63–21.91 on g6e.2xlarge; `studies/next/n2v/projection.json`), hard kill 763 min (1.3 × the high projection), job
budget 700 min. Freeze 1 had registered parts core and fd only (3.90–5.10 GPU-h, hard kill 398 min;
`studies/next/n2v/projection_core_fd.json`). The full launch is 11–16 times the 0.3–0.6
GPU-h the brief expected, for a measured reason: an fp32 forward row on the L40S costs 1.18 s (N6 L1-02: 113.4 s for 96
windows at batch 4) against 0.229 s in bf16 (N12 score-02), and the card's three readouts need 13,818 fp32 rows
(987 unit windows × 10 plain rows, plus 3,948 forward-mode rows at about twice the cost). The decisive readout alone
(the zero ablation and the floor probe, no finite difference, no own-loss readout) is 2.61–3.56 GPU-h; with the finite
difference 3.90–5.10 (§9.6). N12's gradient pass in fp32 does not fit the L40S: the fp32 math-kernel backward needs
67.2 GB for a unit in layer 0 and fits under 46.3 GB for only 48 of the 153 real units (layers 20 and up), by N12 A1's
saved-tensor method, which predicted N12's bf16 pass at 36.92 GB against the 36.04 GB the instance measured
(`studies/next/n2v/offline_gate/mem_fp32.json`). Under the user's rule that pass would run on a Vast H100. This spec
instead computes the fp32 first-order prediction by forward-mode differentiation, which stores no backward graph and
fits the L40S (about 38 GB estimated, measured by the job at its first point); it equals reverse-mode Σ −h·G to 3e-6
relative on a tiny fp32 Olmo 3 (dry run INSTR). The decision itself does not need any new gradient: it compares N12's
banked bf16 prediction with the fp32 measurement, and if that passes, N12's gradient labels are re-read from N12's
banked records under N12's frozen code with no gradient run (every one of the 442 records carries its pull;
`studies/next/n2v/offline_gate/gate_sides.json` → `reread_inputs`). On the banked files every gate the banks hold
fails on its registered failing input (N12's bf16 measurement fails G-VAL with N12's ρ 0.0996 and 0.0425; the bf16
noise as floor leaves G-VAL no room, median ρ 0.063 and 0.077) and passes on a planted fp32 measurement; the weight-free
dry run printed `N2V_DRYRUN_OK` (72 checks, 0 failed).

Written 2026-11-06 by an Opus worker in worktree `interp_research-bio2`, branch `bio2` (from `clean-neurons` at
`ae8dc637f`). No model was run, no weights were downloaded, nothing was launched. Network reads: S3 read-only (the ten
example units' pull arrays of N12 score-02, 12 MB; listings of `results/next_n12/`, `results/next_n2v/`); four web
searches (§1.5).

---

## 0. Status and provenance

| field | value |
|---|---|
| status | ACCEPTED by the head (2026-11-06 19:25Z) with the decisions of §0.6, applied 2026-11-07 by the second J12 executor; the coordinator's decision of 2026-11-07 (§0.6 item 8) adds the forward-mode part and orders the launch. Dry run on the freeze-1a code: `N2V_DRYRUN_OK` (`studies/next/n2v/offline_gate/dryrun_f1a.log`; record `N2V_OFFLINE_GATE.md`, section "Freeze 1a"). Nothing of N2V's own measurement exists. N12's test half was unblinded on 2026-11-06 and its exploratory gradient labels are printed in `n12/READ_n12.md` §5; §1.2 says what that means for the re-reading. |
| freeze 1 | an annotated tag `n2v-freeze-1` on the commit holding this spec, `infra/pg/n2v.py`, `infra/aws_pg/jobs/n2v_score.py`, `infra/aws_pg/jobs/n2v_dryrun.py`, `infra/aws_pg/steps/next_n2v_score-01.json` (parts core,fd; job budget 360 min), `studies/next/n2v/offline_gate/{gate_sides.py, mem_fp32.py}`, `studies/next/n2v/samples/raw_examples.py`, `studies/next/n2v/projection_core_fd.json`. **Tagged** `n2v-freeze-1` 2026-11-07T15:30:31Z on commit `f977f93a3` (tag object `76840b83d`). Inputs uploaded to `results/next_n2v/inputs/` and read back 15:30:48Z: 3 of 3 sha256 equal to the registry (`c223dfee…`, `30f31cb1…`, `877926da…`) |
| freeze 1a | the coordinator's decision of 2026-11-07 (§0.6 item 8): `--parts core,fd,jvp --budget-min 700` in `infra/aws_pg/steps/next_n2v_score-01.json`, hard kill 763 min; the dry run's check names updated; **Tagged** `n2v-freeze-1a` (annotated, tag object `baa19e63f`) on commit `cd628c8a8`, 2026-11-07T16:06:06Z |
| freeze 1b | A1 (§14, 2026-11-07): score-01 stopped at `t14_G_CAUSAL` (428 of 963 rows bitwise before i0) because the fp32 loss read each row's logits in place, at a 16-byte alignment that alternates with the batch slot's parity; `ce_pos` now copies each row first; G-CAUSAL records per-row differences; report `slot_control`; I-FWD and I-JVP record their largest difference; manifest `infra/aws_pg/steps/next_n2v_score-02.json` (same parts and budgets); dry run `N2V_DRYRUN_OK`, 83 checks, 0 failed (`studies/next/n2v/offline_gate/dryrun_a1.log`); **Tagged** `n2v-freeze-1b` (annotated, tag object `4accda02c`) on commit `d0745eb81`, 2026-11-07T18:48:14Z; the launch is `score-02`. No rule changed |
| freeze 2 | `studies/next/n2v/thresholds.json` from the freeze half only (`n2v freeze`; A2, §14: freeze reads the freeze file only; sha256 `cdd2ed26…`, 2026-11-08T00:52:42Z, score-02); **Tagged** `n2v-freeze-2` (annotated, tag object `94de754c6`) on commit `0a3742e43`, 2026-11-08T00:53:08Z. The verdict ran once at 00:53:22Z (`studies/next/n2v/verdict_n2v.json`) |
| parents | `studies/next/09_PLAN.md` §3 card J12; `studies/next/07_PICKS/N12_SPEC.md` §0.5, §1, §2, §3.5, §14 (A1); `studies/next/n12/READ_head.md` and its addendum (the registered follow-up); `studies/next/n12/READ_n12.md` §6; `studies/next/06_CRITIC.md` N2 (validate first-order scores against ablation); `studies/next/03_IDEAS.md` N2; N6's §2 on branch `n6` (the fp32 forward and its rounding-floor probes); `studies/next/12_SHAPING_NOTE.md` §4 items 3–6 |
| method | repo `CLAUDE.md`; the `research-spec` skill; N12's spec format; M2 A5 for the offline gate (both sides of every gate on the banked files) |
| vocabulary | `studies/next/README.md`; N12's §0.5. A run whose gate fails is labelled **DOES NOT COUNT** |

### 0.1 The brief's requirements and where each sits

| requirement | where |
|---|---|
| does −h·G of a single-neuron ablation track the measured ablation with the loss read in fp32; second readout the exemplar's own next-token loss | §1.1, §1.3, §2.1 |
| the same (unit, pole) pairs (carrier + top 3 at the pre point; 328 freeze, 284 test), the same windows | §0.5, §2.2; `gate_sides.json` → `plan` |
| forward and ablation rows in fp32 | §2.2 |
| the gradient pass under N12's deterministic settings; if the fp32 backward does not fit the L40S, the numbers and a Vast H100 lane, no precision compromise | §2.3 (numbers; the forward-mode instrument on the L40S; the H100 lane as the head's alternative) |
| identity pairs: the fp32 forward twice (bitwise or a probe-measured floor), the gradient repeat | §3.2 (I-FWD bitwise, I-JVP bitwise, G-CAUSAL; the probe floor in §3.3) |
| G-VAL as N12 registered it, at freeze on the freeze pairs and at the verdict on the test pairs | §3.4, §5 |
| the finite difference at 10 % as the second validation | §2.1, §3.4 (G-FD) |
| the decision table; whether N12's labels need a new gradient run (from the records) | §1.2; no new run (`gate_sides.json` → `reread_inputs`) |
| if it fails: retire, and cost ablation per pole | §1.2, §9.6 |
| budget from N12's measured rates | §9.6 (N12's bf16 rates and N6's fp32 rate, both measured on the L40S) |
| standing rules: floor, ceiling and does-not-count per pass condition; failing inputs with power shown on the banks; two-path tolerances from control pairs; fp32 with the identity's floor; deterministic forward-mode pass; cross-device points; read with one direction, edit along the other; the nonlinear check beside a failing linear readout; the reader's blindness | §3, §4, §7 |

### 0.2 Facts read for this draft

| fact | value | source |
|---|---|---|
| the pairs | 153 real units (82 freeze, 71 test) at 9 pre points (T_5, T_7–T_14); 4 poles each (N12's `abl` order: carrier, then the top 3 competitors under B); 987 unit windows (windows holding a valid exemplar index 1 ≤ i ≤ 2,047); 328 and 284 pairs | `studies/next/n12/banks/score-02/n12/n12_{freeze,test}.jsonl` (sha256 `30f31cb1…`, `877926da…`); `gate_sides.json` → `plan` |
| N12's banked numbers per pair | `abl[i] = [pole, Σ −h G (bf16 forward, bf16 math-kernel backward), measured window ΔL (bf16)]`; median \|pred\| 0.021 (freeze) and 0.022 (test) nats, 10th percentile 0.0017 and 0.0022 | records; `gate_sides.json` → `halves.*.abs_pred_quantiles_10_50_90` |
| G-VAL on N12's bf16 measurement | ρ 0.0996 (freeze), 0.0425 (test); fails; reproduced by N2V's code from the records | `gate_sides.json` → `halves.*.G_VAL_on_N12_bf16_measurement` |
| fp32 rate on the L40S | 113.4 s for one pass over 96 windows at batch 4, fp32 weights, precision "highest", TF32 off: 1.18 s per window; peak 31.0 GiB for that forward | `n6:studies/next/n6/banks/L1-02/metrics.json` → `measured.pass_s`, `measured.peak_mem_gib.K3` |
| fp32 rounding floor of a probe (N6) | 1.22e-8 and 5.65e-9 nats per token (two probes, 196,608 targets); the same probes in bf16 3.9e-5 and 5.8e-6 | same file → `gates.G_EPS` |
| bf16 rates (N12) | 0.2293 s per forward or ablation row (batch 8); fixed 56–416 s per point with the download; gradient 0.5832 s per window | `studies/next/n12/banks/score-02/n12/metrics.json` → `measured`, `points` |
| memory | N12's bf16 gradient pass measured 36.04 GB; N2V's estimate of it 36.92 GB; fp32 math-kernel backward 1.145 GB per layer, 31.69 GB fixed, 67.2 GB at layer 0, 48 of 153 real units fit | `mem_fp32.json` |
| the re-reading's inputs | 442 of 442 records with `pull`; 153 real with `u_matched`; `thresholds.json` → `floor_u_freeze.q025` 0.4457 | `gate_sides.json` → `reread_inputs` |
| N12's K0 units | L00N9356−:c8 … L06N9844+:c1 (10) | N12 `metrics.json` → `gates.K0.units` |

### 0.3 Facts to verify at run time

| # | fact | verified by | rule if it differs |
|---|---|---|---|
| 1 | torch 2.12.0+cu130, transformers 5.9.0, an L40S | G-SAME | stop |
| 2 | the instance reproduces N12's bf16 ablation rows | K-A at T_14 (bitwise expected) | stop |
| 3 | `model.float()` leaves every parameter fp32, precision "highest", TF32 off | G-FP32 at every point | stop |
| 4 | a row's loss does not depend on its batch companions (same shape, same kernel) | G-CAUSAL and I-FWD, bitwise | stop |
| 5 | the forward-mode rows are deterministic | I-JVP at T_13, bitwise | stop |
| 6 | the forward-mode rows fit the L40S at batch 2 | the first point (T_14); `points.14.jvp_max_mem_gb` | out of memory stops the job at T_14 (about 30 minutes in, after its plain rows); the head decides batch 1 or the H100 lane |
| 7 | the rates (1.18 s per plain row, 2× for a forward-mode row) | `measured` after T_14 | the stop rule (§9.6) |

### 0.4 Already-done audit

| cell | prior measurement | path | what it gives N2V |
|---|---|---|---|
| first-order prediction against zero ablation, bf16, these pairs | N12: G-VAL ρ 0.0996 and 0.0425; median \|ΔL\| 0.65 nats against \|ΔL̂\| 0.022 | `studies/next/n12/verdict_n12.json`, `thresholds.json` | the failing side of every N2V gate, banked |
| why it failed | the reader: \|ΔL\| unrelated to the size of the prediction (Spearman 0.010) or of h (−0.018), growing with the later loss terms (0.241); a correct ΔL̂ plus that noise gives ρ 0.054; "would pass in an fp32 or fp64 forward"; "would also pass on … the exemplar's own next-token loss" (reasoning, not measured) | `studies/next/n12/READ_n12.md` §6 | the hypothesis this card tests |
| the instrument's arithmetic | I-GRAD bit-identical (score-02); P = f(G, q999) to 2e-16; bf16 gradients about 1 % from fp32 in relative L2 on a CPU stand-in | `READ_n12.md` §6; N12 §14 A1 | the banked G is reproducible; its precision against fp32 is reported here (§3.4, `bf16_vs_jvp_prediction`) |
| fp32 losses on this model | N6 L1-02: fp32 forward, G-ID bitwise, probe floor 1.2e-8 per token; K3 2.675162 | `n6:…/banks/L1-02/metrics.json` | the rate, the memory, the expectation of bitwise identity |
| the tiny-model checks | N12 dry run INSTR: fp64 autograd against central differences (1.3e-3); zero ablation against −h G (Spearman 0.990, 24 pairs) | `N12_OFFLINE_GATE.md` | the forward-mode instrument is checked against the same reverse mode (§3.4) |

### 0.5 Objects, defined once

- **Units, pairs, windows.** N12's real units at T_(k−1); pole q ∈ {carrier, competitor 1, 2, 3} in N12's `abl` order; a
  unit window is a window of 2,049 tokens (BOS + 2,048) holding at least one of the unit's exemplar positions at input
  index 1 ≤ i ≤ 2,047 (`n2v.unit_windows`, the same entries N12 zeroed).
- **h** the down-projection input of pole q's neuron at those indices; **ℓ(j)** the next-token cross-entropy of input j
  (predicting input j + 1), j = 0 … 2,047, fp32.
- **Rows** (one fp32 forward each): clean; zero q (h set to 0); fd q (h × 0.9); probe (the carrier's output change
  v = −h W_down[:, n] added to the down projection's output and subtracted again: zero in exact arithmetic); jvp q (the
  forward-mode derivative of every ℓ(j) along the tangent −h at the entries).
- **Readouts per pair**, summed over the unit's windows, in fp64 over fp32 per-position differences:
  ΔL = Σ_j [ℓ_zero(j) − ℓ_clean(j)] (window); ΔL_own = the same over the unit's indices only (the exemplar's own
  next-token loss); ΔL_fd (and ΔL_fd_own) from the fd row; P = the unit's probe change (window and own).
- **Predictions.** pred_bf16 = N12's banked Σ −h·G (the instrument whose labels would be re-read); pred_jvp = the fp32
  forward-mode prediction of ΔL; pred_jvp_own of ΔL_own; the finite difference's prediction is 0.1 × pred_bf16.
- **i0** of a unit window: its smallest zeroed index. Rows differ from their clean row only at j ≥ i0 (causal attention).

### 0.6 The head's decisions before freeze 1 (2026-11-06 19:25Z), applied 2026-11-07

1. **Forward-mode differentiation in fp32** is the registered way to compute an fp32 first-order prediction (§2.3), in
   place of the reverse-mode pass on a Vast H100. Under item 2 this launch computes none.
2. **Parts: core and the 10 % finite difference** (`--parts core,fd`). The launch reads the window readout (N12's banked
   bf16 prediction against the fp32 zero ablation), G-FD, the probe floor and G-ROOM, K-A, G-FP32, G-CAUSAL and I-FWD.
   It runs no forward-mode row, so I-JVP (§3.2, §4), G-VAL window_jvp (§3.4), `bf16_vs_jvp_prediction` and §0.3 rows 5–6
   do not apply. Projection (`studies/next/n2v/projection_core_fd.json`, `n2v project --parts core,fd`): 9,870 fp32 plain
   rows (987 unit windows × 10), 3.90–5.10 GPU-h, $8.74–11.42 on g6e.2xlarge, hard kill 398 min (1.3 × 5.10 h); the
   job's stop rule parks at 0.9 × 360 min = 324 min, above the high projection's 306 min.
3. **The own-next-token-loss readout does not ride on these rows.** Its measurement does: ΔL_own and ΔL_fd_own are sums
   of the zero and fd rows over the unit's own indices, and every pair record carries them. Its registered prediction,
   pred_jvp_own, comes only from the forward-mode rows (the same rows as pred_jvp, at no extra row); N12 banked the
   gradient of the window loss, not of the own loss. So G-VAL own reads NOT_RUN, and ΔL_own and ΔL_fd_own are reported
   without a label. Adding the own readout means adding the jvp part: 6.53–9.78 GPU-h in all (2.63 center and 4.68 high
   more), hard kill 763 min, and the forward-mode memory risk of §0.3 row 6.
4. **The inspection of N12's 0.999 ceiling** as written in §3.4.
5. **K-A** as written in §3.1: bitwise expected, pass when every pair is within 1e-6 nats.
6. **G-ROOM** as written in §3.3.
7. The decision table (§1.2) is unchanged; own modifies the reading, not the branch, so NOT_RUN changes no branch. Under
   these parts branches A and B are reached on synthetic banks in the dry run (VERDICT `core,fd`), C and D as before.
8. **The coordinator's decision of 2026-11-07, which replaces items 2 and 3: parts core, fd and jvp**
   (`--parts core,fd,jvp`), so G-VAL own has its prediction pred_jvp_own; the full 6.53–9.78 GPU-h is approved. The launch
   runs everything the draft registered: 9,870 fp32 plain rows and 3,948 forward-mode rows; I-JVP at T_13, G-VAL own and
   window_jvp, `bf16_vs_jvp_prediction`, and §0.3 rows 5–6 (the forward-mode rows' determinism and memory at batch 2,
   measured at T_14) apply again. Allowance 1.3 × 9.78 h = 763 min (the hard kill, `pg run --budget-min 763`); the job's
   own budget 700 min (`--budget-min 700` in the manifest), whose stop rule parks at 0.9 × 700 min = 630 min, above the
   high projection's 587 min, and leaves 63 min for start and upload. Launch under the capacity rule as the coordinator
   gave it: `--type g6e.2xlarge`, every 180 s for six tries, then every 600 s; VcpuLimitExceeded and
   RequestLimitExceeded retried; any other error stops and is reported.

---

## 1. Question, decision, hypotheses

### 1.1 The question

N12's first-order instrument failed its validation: zeroing a pole at its exemplar positions changed the bf16
window-summed loss by 0.65 nats at the median, 27 times the prediction, with chance-level sign agreement. The reader
showed the change behaves as bf16 rounding noise of a loss summed over 2,048 terms. Read in fp32, does the first-order
prediction −h·G track the measured single-neuron zero ablation, on the window sum and on the exemplar's own next-token
loss? And does a 10 % reduction of h, a perturbation where first order should hold, agree with a tenth of the
prediction?

### 1.2 The decision this run makes

| outcome (both halves) | reading | next |
|---|---|---|
| **A**: G-VAL window PASS (N12's banked bf16 prediction against the fp32 window ΔL) | N12's instrument tracks zero ablation once the loss is resolved | `n2v.reread`: N12's gradient labels (H_signal, H_sel_signal_pb, H_sel_signal_dt, H_bridge) from N12's banked records under N12's frozen code, the measured column replaced by N2V's fp32 one. No gradient run: every record carries its pull. Status: a dated re-reading, not a confirmation, because N12's test half was read on 2026-11-06 and the exploratory labels are already printed (`READ_n12.md` §5: H_signal REFUTED at u_S 0.527 against the bar 0.4225, the two selection contrasts AT, H_bridge UNDECIDED). Branch A changes their status from DOES NOT COUNT to counted; it cannot change their values. |
| **B**: window FAIL, G-FD PASS | the gradient is right for a small change; −h·G does not stand in for removing the neuron at its strongest positions | N12's gradient labels stay DOES NOT COUNT (N12's frozen rule names zero ablation). Removal claims need ablation per pole: 14–17 GPU-h for real units (carrier + 20 competitors at E_c and at the loss-matched positions, fp32), 42–51 with both twins, plus 32–38 for the matched-random floor (§9.6) |
| **C**: window FAIL, G-FD FAIL or NOT_RUN | the first-order instrument does not track ablation in fp32 in this model; retired | as B, the same costs |
| **D**: DOES NOT COUNT (no room, a failing input that passed, the ceiling not cleared, a gate) | no claim about the instrument | the failed gate and its value; fp64 rows or a different floor is the head's call |

The own-loss readout (G-VAL own) modifies the reading, not the branch: window FAIL with own PASS says the first-order
term is right at the exemplar and fails through the later positions the window sum adds up; own FAIL with window PASS
says the window sum is carried by later positions.

### 1.3 Hypotheses and the observation that refutes each

| id | hypothesis | predicts | refuted by |
|---|---|---|---|
| **H_res** (the reader's) | N12's failure was the resolution of a bf16 window-summed loss | G-VAL window PASS on both halves in fp32 (ρ ≥ 0.7, lower bound above the shuffle) | G-VAL window FAIL with G-ROOM passing (the fp32 floor resolved the effects and the rank agreement is still below 0.7) |
| **H_nonlin** | zero ablation at a neuron's strongest positions is outside first order; the gradient itself is right | window FAIL, G-FD PASS (branch B) | G-FD FAIL (a tenth of the change disagrees too), or window PASS |
| **H_grad** | the banked gradient is wrong for this model (not a precision matter) | window FAIL and G-FD FAIL (branch C); pred_bf16 far from pred_jvp | G-FD PASS, or bf16-vs-JVP Spearman above 0.99 with G-VAL window_jvp PASS |
| **H_later** | first order holds at the exemplar; the window sum adds effects that pass through later positions | G-VAL own PASS, window FAIL | own FAIL, or window PASS |
| **H_floor** (the boring one) | fp32 does not resolve these effects either | G-ROOM FAIL: the probe floor is above about 0.017 nats per unit (window) or 0.0017 (fd) | G-ROOM PASS on both halves |

Room: on the banked predictions, G-ROOM passes for a uniform floor up to 0.0169 (freeze) and 0.0174 (test) nats per
unit on the window readout, 0.0017 on the finite difference (`gate_sides.json` → `largest_floor_with_room_*`). N6's
fp32 probe floor, 1.2e-8 nats per token, sums to about 1e-4 over a few windows of 2,048 tokens, two orders of magnitude
below the window bound and one below the finite difference's. Both branches of G-VAL are reachable: a planted fp32
measurement passes (ρ 0.9998), N12's bf16 one fails.

### 1.4 The author's prior, logged before any N2V number exists (2026-11-06)

| quantity | prior | basis |
|---|---|---|
| G-ROOM passes, window | 0.9 | the bound 0.017 nats against an expected floor near 1e-4 |
| G-ROOM passes, fd | 0.75 | the bound 0.0017 is closer to the expected floor |
| G-VAL window PASS on both halves (H_res) | 0.6 | the tiny model's strongest neuron: zero ablation 0.134 against a prediction of 0.119 (12 %); attribution patching against activation patching gave R² 0.27 in one edge-level study (search summary, §1.5) |
| G-VAL window above 0.999 (the ceiling) | 0.1 | the zero ablation's second-order term is several percent |
| G-FD PASS | 0.8 | the tiny model: 2.5 % between the 10 % difference and a tenth of the prediction |
| G-VAL own PASS | 0.7 | the reader's reasoning |
| K-A bitwise | 0.85; within 1e-6 0.9 | N12's G-ABL-ID was 0.0 with different batch companions |
| G-CAUSAL and I-FWD bitwise | 0.9 | same shapes and kernels; N6 G-ID 0.0 in fp32 |
| the forward-mode rows fit at batch 2 | 0.85 | estimate in §2.3 |

### 1.5 Literature (query terms recorded)

Searches of 2026-11-06 (WebSearch): (1) `attribution patching validation against activation patching single neuron
ablation numerical precision bfloat16 noise`; (2) `first-order Taylor approximation zero ablation MLP neuron agreement
"attribution patching" finite difference fp32 language model`. Earlier records: N12 §1.5 (four queries).

- AtP*, Kramár et al., arXiv 2403.00745: the first-order expansion around clean activations; its framing excludes
  interventions to values that do not occur, such as zero or mean ablation; false negatives from saturated attention
  and cancellation. Search summary only; no number cited. It is why G-FD exists beside zero ablation.
- "When Attribution Patching Lies: Diagnosis and a Second-Order Correction", arXiv 2606.09899: the approximation error
  splits into a dominant second-order (Hessian) term and a smaller cubic remainder. Search summary only; not read in
  full; no number cited. It predicts H_nonlin's pattern (the zero ablation deviates, a small step does not).
- An edge-level comparison of attribution and activation patching (arXiv 2310.10348) reports R² 0.27 between the two;
  search summary, edges not neurons, quoted only as the basis of a prior.
- Not found in two searches: any measurement of how floating-point precision of the forward limits the validation of a
  first-order neuron attribution against ablation. Two searches are not a hole in the literature.

---

## 2. The measurement and the instrument

### 2.1 In plain words

- **Items.** N12's 612 ablated pairs (328 freeze, 284 test), at their pre points, the same windows and indices.
- **Model.** `allenai/Olmo-3-1025-7B` at T_5 and T_7–T_14 (`m2.POINTS`), loaded as N12 loaded it (bf16, sdpa), then cast
  to fp32 for every N2V row (`model.float()`, precision "highest", TF32 off).
- **Quantities per pair**: ΔL, ΔL_own, ΔL_fd, ΔL_fd_own, pred_jvp, pred_jvp_own, the unit's probe P and P_own, Σ|h|,
  the reachable loss terms; and N12's pred_bf16 and bf16 ΔL.
- **Pairing.** Every readout of a pair comes from rows of the same windows on the same weights; prediction and
  measurement pair by (unit, pole).
- **Summary.** N12's rule (`n12.validation`, unchanged): Spearman ρ over the half's pairs; a percentile bootstrap over
  units (1,000 resamples, seed 0, as N12's code ran it; N12's spec text said 10,000, `READ_n12.md` §9 item 1); the 97.5th
  percentile of ρ under 1,000 shuffles of the prediction; the slope through the origin reported.

### 2.2 The instrument, exactly

`infra/aws_pg/jobs/n2v_score.py`, one launch, points in the order T_14, T_13, T_12, T_11, T_10, T_9, T_8, T_7, T_5:

1. Download, G-REV, G-CFG, `neurons_harvest.load_model` (bf16, sdpa), every parameter frozen.
2. At T_14 only, in bf16, N12's own functions: K-A (§3.1), then the same ten units' probe rows in bf16 (reported).
3. `model.float()`, G-FP32.
4. Plain rows, batches of 4, default kernels, deterministic mode off: per unit window clean, zero q (4), fd q (4),
   probe; every row's ℓ(j) [2,048] fp32 (`plain_rows`). G-CAUSAL (§3.2). h at the indices captured from the clean row.
5. Forward-mode rows (`jvp_rows`), batches of 2, under `jvp_mode()`: `torch.use_deterministic_algorithms(True)` and
   the sdpa math kernel (N12 A1's settings; in fp32 there is no reduction setting to choose); the tangent −h at
   (indices, neuron) made dual at the down projection's input (`torch.autograd.forward_ad.make_dual`); the tangent of
   every ℓ(j) read from the dual cross-entropy.
6. At T_13: I-FWD and I-JVP (§3.2).
7. Per pair `n2v.pair_values`; every row's ℓ(j) and every tangent to `losses/tNN.npz` (about 113 MB), so any pair
   recounts from the banked rows.

After the last point: `n2v_<half>.jsonl` (one line per pair) and `samples/raw_examples_<half>.jsonl`; `metrics.json`
carries gates, counts and timings, never a Spearman, a room value or a label.

### 2.3 The lane: the fp32 backward does not fit the L40S; the forward-mode prediction does

- **Numbers** (`studies/next/n2v/offline_gate/mem_fp32.py`, N12 A1's method: one eighth of a layer's width on the CPU,
  saved tensors counted, × 8): the fp32 model with the math kernel saves 1.145 GB per layer for a 2,049-token window
  (bf16 with bf16 reductions: 0.640); with 29.19 GB of fp32 weights and the fp32 logits, log-softmax and gradient the
  peak is 67.2 GB for a unit in layer 0 and 68.3 GB at most over the 153 real units. The L40S reports 48.3 GB. 48 of the
  153 real units (layers 20–31) would fit with 2 GB to spare. The method's control: it puts N12's bf16 pass at 36.92 GB;
  the instance measured 36.04.
- **Under the user's rule** this reverse-mode pass would run on a Vast H100 (80 GB: 68.3 GB plus about 2 GB of transients
  fits). That lane needs a harvest-stack image on Vast (none exists; the journal of 10:00 CST), a Vast top-up, and it
  would put the own-loss comparison across two GPU models unless the plain rows moved there too.
- **What this spec registers instead.** The first-order prediction of a pair is a directional derivative: Σ_i −h(i) G(i)
  is the derivative of the loss along δh = −h. Forward-mode differentiation computes it in one forward pass that carries a
  tangent beside every value and stores no graph, so its memory is about twice a forward's transients, not a backward's
  saved tensors: fp32 weights 27.2 GiB, logits and their tangents at batch 2 3.1 GiB, one row's dual log-softmax 1.5 GiB,
  one layer's dual math-kernel attention about 4 GiB, about 36 GiB (38 GB) in all, against N6's measured 31.0 GiB for a
  plain fp32 forward at batch 4. It is not a precision compromise: fp32 throughout. On a tiny fp32 Olmo 3 it equals
  reverse-mode Σ −h·G to 3e-6 relative (dry run INSTR: 0.1190132 against 0.1190135), and in fp64 to 4e-10 (§9.3 note).
  The cost: one forward-mode row per pole (4 per window) at about twice a plain row, against one backward per window in
  reverse mode; 2.59 GPU-h center for the 3,948 rows.
- **What it gives up.** It gives the prediction for the four ablated poles, not G for every neuron of the layer, so it
  cannot recompute N12's pull. It does not need to: the re-reading uses N12's banked pull, and the forward-mode
  prediction serves the own-loss readout and the bf16-against-fp32 comparison of the prediction itself.
- **The head's alternative** (not registered; an amendment would register it): the reverse-mode fp32 pass on a Vast
  H100, window and own losses, 987 backward windows, plus the plain rows on the same card to keep one GPU model per
  comparison. Its rates on an H100 in fp32 are not measured anywhere in the repo.

### 2.4 What runs at zero GPU

The plan, the pairs, every gate's failing side and the G-ROOM bound on N12's banked records, the re-reading on N12's real
records (planted fp32 measurement: the labels; N12's bf16 measurement: DOES NOT COUNT), the ten examples' banked part,
the memory estimate. Not at zero GPU: every fp32 loss and every forward-mode prediction.

---

## 3. Controls and references

Rule 4 of "Instruments and controls": fit on both sides (K-A against N12's banked dL; G-VAL against the measurement), a
positive control that fired (G-FD on the tiny model; the planted pass of every gate offline), the identity pair (I-FWD,
I-JVP; G-CAUSAL is an identity in exact arithmetic before i0).

### 3.1 The known number K-A

At T_14, before the fp32 cast: N12's ablation rows of its K0 units (the first ten real units by id whose pre point is
T_14, the ten N12's K0 ran on) run again through N12's own `n12_score.ablate` (batches of 8 rows) and the windows'
losses through `n12_score.forward_loss` (batches of 8), harvest kernels, deterministic mode off, as in N12 score-02.
Pass: every one of the 40 pairs' ΔL within 1e-6 nats of N12's banked `abl[i][2]` (bitwise expected: the same device
type, code and shapes; N12's G-ABL-ID gave 0.0 at 153 clean rows whose batch companions differed). Failing input: the
banked ΔL with the poles rolled inside each unit; offline its smallest gap is 0.0166 nats, 16,600 times the tolerance
(`gate_sides.json` → `K_A`). Pass side offline: not banked (no second bf16 run of these rows exists); in the dry run's
tiny world 40 of 40 pairs are bitwise. If K-A fails, the job stops: the instance does not reproduce N12's rows.

### 3.2 The identity pairs and the causal identity

- **G-CAUSAL** at every point: every changed row (zero, fd, probe) equals its unit window's clean row bitwise at every
  j < i0. Causal attention guarantees it in exact arithmetic; bitwise needs the row's value not to depend on its batch
  companions (same shape, same kernel), which N12's G-ABL-ID showed in bf16 and N6's G-ID in fp32. Failing input: the
  zero rows compared through j = i0, which the change reaches; it must differ (dry run: a hook planted one index early
  stops the job at `t14_G_CAUSAL`, 162 of 288 rows bitwise).
- **I-FWD** at T_13: the clean rows again, in reversed order, in batches of clean rows only (other companions than in
  the main pass); bitwise. Failing input: each clean row against its window's first zero row. A planted second pass
  1e-3 off stops the job (dry run).
- **I-JVP** at T_13: the first 16 unit windows' forward-mode rows of poles 0 and 1 again, reversed; bitwise (deterministic
  mode). Failing input: pole 1's tangent against pole 0's. A planted second pass × 1.001 stops the job (dry run).
- Tolerances: bitwise, from control pairs on the same path (N6 G-ID 0.0, fp32, L40S; N12 G-ABL-ID 0.0 and I-GRAD exact
  under deterministic mode). No comparison here crosses two paths: the forward-mode and plain rows are never compared
  value for value; their primal outputs are not used.

### 3.3 The floor: the probe row (G-EPS → G-ROOM)

Per unit window, the probe row adds the carrier's own output change v = −h W_down[:, n] to the down projection's output
at the indices and subtracts it again in fp32: zero in exact arithmetic, a rounding-sized change in fp32, propagated
through every later position like the ablation's own rounding. Its loss change per unit (window and own) is the floor of
this measurement path at these positions (N6's probe, at the ablation's own size). On the tiny model it is 9.5e-7 nats
(one fp32 step of one loss) against a zero ablation of 0.134 (dry run INSTR).

**G-ROOM** (per readout, per half, at freeze and at the verdict): a perfect instrument (measurement = prediction) plus
each unit's own probe change with a random sign per pair; over 1,000 replicates (seed 0) the median Spearman must be at
least 0.9. Failing input with power: N12's banked bf16 measured change of the carrier as each unit's floor, median ρ
0.063 (freeze) and 0.077 (test), both below 0.9 (`gate_sides.json`). Pass side: a planted floor of 1e-4 nats, median ρ
0.99998 (window) and 0.999 (fd). If G-ROOM fails, the readout DOES NOT COUNT (no room), whatever ρ is. The bar 0.9 is a
property of the design (room well above the 0.7 pass bar), registered a priori; the floor itself is measured.

### 3.4 The validations: G-VAL (window, own, window_jvp) and G-FD

All four run N12's `n12.validation` unchanged: pass when ρ ≥ 0.7 and the bootstrap lower bound over units exceeds the
shuffle's 97.5th percentile; failing input the prediction with its sign flipped; ceiling ρ > 0.999 reads "DOES NOT COUNT
until inspected" (N12's rule).

| readout | prediction | measurement | role |
|---|---|---|---|
| **window** (primary) | pred_bf16 (N12's banked) | ΔL (fp32) | decides the branch; on A, the re-reading |
| **fd** (the second validation; `12_SHAPING_NOTE.md` §4 item 6, the nonlinear check beside the linear readout) | 0.1 × pred_bf16 | ΔL_fd | separates "the gradient is wrong" from "zero ablation is outside first order" |
| **own** (the second readout) | pred_jvp_own (fp32) | ΔL_own | localizes a failure to the exemplar or to later positions |
| **window_jvp** (reported) | pred_jvp | ΔL | the same as window with the fp32 prediction; with `bf16_vs_jvp_prediction` it says whether the bf16 gradient's precision matters |

**The inspection of N12's ceiling**, frozen now, before any fp32 number exists. In fp32 a near-linear effect could
rank-agree above 0.999 with no defect (a planted 5 % deviation gives ρ 0.9998). The ceiling's purpose is a measurement
that is a copy of its prediction. The inspection clears it when no pair's measurement equals its prediction and the
median |meas − pred| / |pred| is at least 1e-3 (the bf16 prediction's own error is about 1 %, N12 A1); offline a planted
5 % deviation clears (median 0.035) and a copy does not (`gate_sides.json`). The reader's recount of ΔL from the banked
per-row losses is its third part (§7 step 5). On branch A with a cleared ceiling, `n2v.reread` lifts N12's ceiling for
that call only (N12's code reads `n12.VAL_RHO_CEIL` at call time) and restores it. G-FD's own ceiling is the copy test
alone: every pair's |ΔL_fd − 0.1 pred| / |0.1 pred| below 1e-6 reads DOES NOT COUNT until inspected; a finite difference
that agrees closely with a tenth of the prediction is the expected result, not a defect.

**Floor, ceiling and does-not-count of each pass condition**: floor = the shuffle's 97.5th percentile (the bound must
clear it) and G-ROOM (the measured floor must leave room); ceiling = ρ > 0.999 with the inspection (window, own,
window_jvp) or the copy test (fd); does-not-count = no room, a failing input that passed, the ceiling not cleared, a job
gate failed, fewer than 50 units in the half.

### 3.5 Read with one direction, edit along the other (`12_SHAPING_NOTE.md` §4 item 4)

The zero and fd rows edit the read side (the neuron's down-projection input, its activation) at the exemplar
positions; the probe acts on the written side (the column's output) and is an identity. This card validates an
instrument; it makes no claim about a direction.

### 3.6 Monitoring level

Every readout is the L1 type: one pole, a fixed identity, one layer. A failing instrument says nothing about a job
carried by a direction spread over neurons.

---

## 4. Gates

Every instance gate is asserted in the job (non-zero exit, STATUS `exit≠0`); its failing-input value is written to
`out/score/n2v/metrics.json`; the verdict reads every gate again (`m2.gates_ok`).

| gate | where | pass | registered failing input; shown failing where | if it fails |
|---|---|---|---|---|
| G-SAME | start | torch 2.12.0+cu130, transformers 5.9.0, NVIDIA L40S | transformers 0.0.0 (in job); dry run | stop |
| INPUTS | start | `units.json`, N12's two record files at their sha256 | a zero sha256; dry run | stop |
| G-CORPUS | start | tokens sha256 `2aca80ca…` | a zero sha256 | stop |
| G-REV, G-CFG | each point | as N12 | a zero revision; model_type renamed | stop |
| K-A | T_14, bf16 | §3.1 | rolled poles: offline 0.0166 nats against 1e-6; dry run: N12's dL + 1e-3 stops the job | stop |
| G-FP32 | each point | every parameter fp32, "highest", TF32 off | the dtypes before the cast (bf16); dry run: precision "high" stops the job | stop |
| G-CAUSAL | each point | §3.2 | through i0; dry run: a hook one index early stops the job | stop |
| I-FWD, I-JVP | T_13 | §3.2 | §3.2; dry run planted second passes stop the job | stop |
| G-ROOM (window, fd, own, window_jvp) | freeze, verdict, per half | §3.3 | N12's bf16 measured change as floor (offline, both halves) | that readout DOES NOT COUNT; freeze refuses if the failing input passes on the freeze half |
| G-VAL window, own, window_jvp | freeze, verdict, per half | §3.4 | the sign flip (offline on planted, both halves); N12's bf16 measurement (offline, real, fails) | the readout's label FAIL |
| G-FD | freeze, verdict, per half | §3.4 | the sign flip; a copy above its ceiling (offline) | FAIL |
| usable | freeze, verdict | ≥ 50 units per half | (asserted by freeze and checked by the verdict; not planted in the dry run) | DOES NOT COUNT |
| freeze records unchanged | verdict | sha256 of `n2v_freeze.jsonl` | one byte appended (dry run) | the verdict refuses |

---

## 5. Labels and the verdict, as the script runs it

`python -m infra.pg.n2v verdict --bank studies/next/n2v/banks/<attempt> --thresholds studies/next/n2v/thresholds.json --out studies/next/n2v/verdict_n2v.json`

1. The freeze records' sha256 must match `thresholds.json`.
2. Gates (`m2.gates_ok`): every instance gate passed, every failing input failed, the job complete; else DOES NOT COUNT.
3. ≥ 50 test units; G-ROOM's failing input fails on the test half; else DOES NOT COUNT.
4. Per readout and half (`n2v.g_label`): NOT_RUN (part not run), DOES NOT COUNT (no room; failing input passed; above
   the ceiling and not cleared), PASS, FAIL. Over the halves (`n2v.both`): PASS only if both pass; a does-not-count half
   decides; FAIL otherwise.
5. Branch (`n2v.decision`): A if window PASS; else B if fd PASS; else C; D if window does not count.
6. On A: `n2v.reread` (N12's code, N12's records and thresholds, the fp32 window measurement); its headline and the four
   gradient labels are written.
7. Headline: `G-VAL window …; G-FD …; G-VAL own …; branch …[; N12 re-read: …]`. Every value is written whatever the headline.

The dry run produced branches A, B, C and D, the copy case, the core-only case and both refusals on synthetic banks built
over N12's real records (`N2V_OFFLINE_GATE.md`, VERDICT).

---

## 6. The split, freeze 2 and room

**The split**: N12's (M2's): the carrier's neuron even → freeze (82 units, 328 pairs), odd → test (71, 284). **What was
read before this spec**: N12's banked numbers of both halves (they are N12's unblinded result) and, for the ten freeze
examples, N12's pull arrays. Nothing of N2V exists.

**Freeze 2** (`n2v.freeze`) refuses unless the gates pass, the freeze half has ≥ 50 units and G-ROOM's failing input fails
on it. It writes the freeze half's G-ROOM and G-VAL for every readout, their labels, the inspection where the ceiling
fired, and the sha256 of `n2v_freeze.jsonl` and of the job's `metrics.json`.

**What is frozen on control data, what is registered a priori.** On banked control data: K-A's 1e-6 (N12's G-ABL-ID 0.0
and the bitwise K0 of M2 and N12), the bitwise identities (N6 G-ID, N12 I-GRAD). On the freeze half: the freeze half's
validations, which enter the both-halves rule. A priori, with reasons: N12's bar 0.7 and ceiling 0.999 (the brief: as N12
registered them); G-ROOM's 0.9 (room above the bar); the inspection's 1e-3; G-FD's 1e-6 copy test.

**Room** (§1.3): every gate reaches both sides on the banked predictions; every branch of the decision table is reached
in the dry run.

---

## 7. Unblinding order

1. Freeze 1 after the head's review: `N2V_DRYRUN_OK` on the frozen commit; tag `n2v-freeze-1`.
2. The one launch, `score-01`. During the run only `n2v/metrics.json` is read.
3. Pull `metrics.json`, `n2v_freeze.jsonl`, `samples/raw_examples_freeze.jsonl`. Open the freeze examples and answer in
   `studies/next/n2v/JOURNAL.md`: the decoded exemplars are N12's; the rows' losses before i0 equal the clean row's; the
   carrier's five largest per-position changes sit at or after the zeroed indices; the probe is at the fp32 floor; one pair
   with |ΔL − pred| small and one with it large, with their tokens. Then `n2v freeze`, commit, tag `n2v-freeze-2`.
4. Pull `n2v_test.jsonl` and the test examples; run the verdict once.
5. A fresh Opus reader, in a worktree made from a commit whose subject names no label (the M2 and N12 readers' blindness
   broke on `git log`), recomputes every pair's ΔL from `losses/tNN.npz`, the four validations and G-ROOM with its own
   code; on branch A it reruns the re-reading. The head's read is the different-model read. The result document follows.

---

## 8. Threats to validity

| # | threat | effect | mitigation |
|---|---|---|---|
| 1 | the probe floor is a tiny perturbation; an ablation's own rounding could be larger | G-ROOM passes while the ablation is noisier | the sign agreement and the slope are reported; pairs with \|pred\| < 1e-4 show the measured noise directly; G-CAUSAL bounds where differences can sit |
| 2 | pred_bf16 used h from the bf16 forward; the fp32 rows zero the fp32 h | a 1 % mismatch | `bf16_vs_jvp_prediction` reported; window_jvp is the fp32-only readout |
| 3 | zero ablation at the strongest positions is far from first order | window FAIL with a correct gradient | G-FD (branch B) |
| 4 | the re-read labels are already known | a re-reading cannot be a test | stated in §1.2: status only |
| 5 | N12's ceiling fires on a legitimately near-linear effect | a good instrument reads DOES NOT COUNT | the inspection frozen in §3.4 |
| 6 | the forward-mode prediction is a second computation path for the same quantity | a defect in one path | the tiny model: forward and reverse mode agree to 3e-6; window and window_jvp both reported |
| 7 | one model, one run, 153 units on 140 carriers | no replicate | stated |
| 8 | memory of the forward-mode rows not measured on the L40S | out of memory at T_14 | it fails about 30 minutes in (after T_14's plain rows); the head decides batch 1 or the H100 lane |

---

## 9. Runbook

### 9.1 Checklist before the launch

- Read `.claude/skills/aws-pg/SKILL.md` and the docstrings of `infra/aws_pg/jobs/pipeline.py` and `n2v_score.py`.
  Profile `fuelos`, region `us-east-1`, on every call. The worktree needs its own `.pgvenv` for `./infra/aws_pg/pg`.
- Liveness: an instance id from `./infra/aws_pg/pg ps`, or `updated_utc` of `out/score/n2v/metrics.json`. Never poll `job.log`.
- Cost and incidents: `infra/aws_pg/operations/2026-11-<dd>_next_n2v.md`; science: `studies/next/n2v/JOURNAL.md`.

### 9.2 Code (NEW or REUSED, each checked against the tree at `ae8dc637f`)

| file | status | role |
|---|---|---|
| `infra/pg/n2v.py` | NEW | plan, pair values, G-CAUSAL helper, validations (through `n12.validation`), G-ROOM, the inspection, labels, decision, freeze, verdict, the N12 re-reading (through `n12.verdict`), projection |
| `infra/aws_pg/jobs/n2v_score.py` | NEW | the job (`plain_rows`, `jvp_rows`, `jvp_mode`, `run`) |
| `infra/aws_pg/jobs/n2v_dryrun.py` | NEW | the offline gate |
| `infra/aws_pg/steps/next_n2v_score-01.json` | NEW | the step list |
| `studies/next/n2v/offline_gate/gate_sides.py`, `mem_fp32.py` | NEW | both sides on the banks; the memory estimate |
| `studies/next/n2v/samples/raw_examples.py` | NEW | the ten raw examples at zero GPU |
| `infra/pg/n12.py` (`validation`, `verdict`, `spearman`, `N_ABL`, `N_MIN`, `VAL_RHO_MIN`, `VAL_RHO_CEIL`) | REUSED, unchanged | N12's rule and N12's verdict |
| `infra/aws_pg/jobs/n12_score.py` (`ablate`, `forward_loss`), `n12_dryrun.py` (`job`, the tiny world) | REUSED, unchanged | K-A on N12's code; the dry run's world |
| `infra/pg/m2.py`, `infra/aws_pg/jobs/m2_score.py`, `m2_dryrun.py`, `infra/pg/neurons_harvest.py`, `infra/aws_pg/jobs/neurons_harvest.py`, `infra/pg/neurons_corpus.py`, `infra/pg/m1.py` | REUSED, unchanged | loaders, gates, the corpus, the stand-in hub |

### 9.3 The offline gate

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd /Users/abaybektursun/projects/interp_research-bio2
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n2v/offline_gate/gate_sides.py --out studies/next/n2v/offline_gate/gate_sides.json
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n2v/offline_gate/mem_fp32.py --hf $SP/m2/hf --out studies/next/n2v/offline_gate/mem_fp32.json
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n2v/samples/raw_examples.py --cache $SP/bio2/n12pull --out studies/next/n2v/samples
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n2v project > studies/next/n2v/projection.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.n2v_dryrun --hf $SP/m2/hf \
    --sides studies/next/n2v/offline_gate/gate_sides.json --out $SP/bio2/n2v_dry
```
It must end with `N2V_DRYRUN_OK`. The record is `N2V_OFFLINE_GATE.md`. Note on the fp64 check: on the tiny model in
fp64 the forward-mode prediction equals reverse mode to 4e-10 relative (`$SP/bio2/jvp_test.py`, a scratch check);
OLMo-3's norm layers compute in fp32 internally, so an "fp64" model is not fp64 throughout, which is why the dry run's
checks are in fp32.

### 9.4 S3 layout

```
results/next_n2v/inputs/{units.json, n12_freeze.jsonl, n12_test.jsonl}
results/next_n2v/score-NN/steps.json  STATUS  job.log  heartbeat.jsonl
results/next_n2v/score-NN/out/score/n2v/{metrics.json, n2v_freeze.jsonl, n2v_test.jsonl}
results/next_n2v/score-NN/out/score/n2v/samples/{raw_examples_freeze.jsonl, raw_examples_test.jsonl}
results/next_n2v/score-NN/out/score/n2v/losses/tNN.npz              (every row's l(j), every tangent; about 113 MB)
```
`results/next_n2v/` listed empty on 2026-11-06.

### 9.5 Sessions

```
B=s3://pg-lineage-769063704696
```
**S1. Freeze 1 (laptop).** §9.3 to `N2V_DRYRUN_OK`; the head's review; commit; tag `n2v-freeze-1`; record it in §0.
`./infra/aws_pg/pg ps` and `audit`: no stray instance. Upload and read back:
```
aws s3 cp studies/next/n12/units.json $B/results/next_n2v/inputs/units.json --profile fuelos --region us-east-1
for h in freeze test; do aws s3 cp studies/next/n12/banks/score-02/n12/n12_$h.jsonl \
  $B/results/next_n2v/inputs/n12_$h.jsonl --profile fuelos --region us-east-1; done
```
(the two record files are git LFS objects; `git lfs pull` them first and check their sha256 against §0.2).

**S2. The launch (`score-01`; parts core,fd,jvp (§0.6 item 8); 6.53–9.78 GPU-h; job stop rule at 0.9 × 700 min; hard kill 763 min).**
```
aws s3 cp infra/aws_pg/steps/next_n2v_score-01.json $B/results/next_n2v/score-01/steps.json --profile fuelos --region us-east-1
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_n2v/score-01/steps.json" \
    --no-weights --type g6e.2xlarge --s3-prefix results/next_n2v/score-01 --budget-min 763 --json --detach
./infra/aws_pg/pg watch results/next_n2v/score-01
```
The capacity rule as the coordinator gave it on 2026-11-07 (`--type g6e.2xlarge` only; every 180 s for six tries, then
every 600 s; VcpuLimitExceeded and RequestLimitExceeded retried; any other error stops and is reported). The first gates
to read: K-A, then `t14_G_FP32`, `t14_G_CAUSAL`, `points.14.jvp_max_mem_gb`, `measured`. The exact commands are in `studies/next/n2v/STATE.md`.

**S3. Freeze 2 (laptop, freeze half).** §7 step 3.

**S4. Unblind (laptop).** §7 step 4; fill §12.

**S5. The reader and the head** (§7 step 5).

### 9.6 Budget, stop rule, park order

| launch | contents | GPU-h center | GPU-h high | hard kill |
|---|---|---|---|---|
| `score-01`, parts core,fd,jvp (**registered**: the draft; the coordinator's decision of 2026-11-07, §0.6 item 8) | 9 points; 9,870 plain rows (987 unit windows × 10), 3,948 forward-mode rows, K-A (227 bf16 rows: 134 ablation rows, 31 window losses, 62 probe rows), the identities at T_13 | 6.53 | 9.78 | 763 min |
| parts core,fd (freeze 1, the head's choice of 2026-11-06; replaced; own and window_jvp NOT_RUN) | 5,922 + 3,948 plain rows | 3.90 | 5.10 | 398 min |
| parts core (fd, own, window_jvp NOT_RUN; branch B unreachable) | 5,922 plain rows | 2.61 | 3.56 | 278 min |

Rates (`n2v.RATES`): a plain fp32 row 1.18 s center (N6 L1-02 measured) and 1.40 s high; a forward-mode row 2× a plain row
center, 3× high (estimated: one extra matmul per matmul; measured by the job at T_14); bf16 rows 0.2293 and 0.25 s (N12
measured); 200 s and 420 s fixed per point (N12 measured 56–416 s with the download; the fp32 cast added); start 344–450 s.
At $2.24 per hour: $14.63–21.91; on g6e.4xlarge ($3.00424): $19.62–29.38.

**Stop rule, in the job**: after T_14 the job measures its fixed cost and its seconds per plain and per forward-mode row;
before each later point it projects the rest; if elapsed time plus that exceeds 0.9 × 700 min, this point and every later
one are PARKED and the job ends normally. **Park order**: T_14 (K-A), T_13 (the identities), T_12, T_11, T_10, T_9, T_8,
T_7, T_5; parking from the end removes the fewest units (T_5 1, T_7 6, T_8 4, T_9 8). Fewer than 50 units in a half reads
DOES NOT COUNT (missing); a second launch for parked points is the head's decision.

**Ablation per pole, if the instrument is retired (branches B, C)**, from the same rates: real units, the carrier and its
20 competitors zeroed at E_c and at the 16 loss-matched positions per exemplar, fp32: 41,454 rows, 14.1–17.2 GPU-h; with
both twins (3,029 unit windows): 127,218 rows, 42–51 GPU-h; the matched-random floor (100 matched poles per real unit at
E_c) 98,700 rows, 32–38 GPU-h more. N12's loss-matched indices are not banked (only their mean G), so they would be
recomputed from fp32 per-position losses.

### 9.7 Artifacts

- On S3 (§9.4). On the laptop: `studies/next/n2v/{projection.json, samples/, offline_gate/, banks/, thresholds.json,
  verdict_n2v.json, JOURNAL.md}`. Operations: `infra/aws_pg/operations/2026-11-<dd>_next_n2v.md`.

---

## 10. The ten raw examples

**Zero-GPU part, written now** (`studies/next/n2v/samples/raw_examples_zero_gpu.{md,json}`): N12's ten freeze-half example
births, their 40 pairs with N12's banked bf16 prediction and measurement, Σ|h| (from N12's pull arrays), the windows and
reachable loss terms, the finite difference's predicted size, one exemplar and the first matched random pole's top
context (§4 item 3 of the shaping note; texts holding a word this program does not print are skipped and the index given).

What they show: over the 40 pairs the bf16 measurement is 0.88 nats at the median against a prediction of 0.026; sign
agreement 0.62; the ratio runs from −724 to 8,280 (L10N9052+, the NPR "Copyright" carrier: prediction −0.00014 nats,
measurement −1.13). The measurement does not follow Σ|h|: L24N5066−'s competitors remove Σ|h| 279–404 in one window and
move the loss 0.23–0.39 nats, L07N8340+'s remove 0.9–2.2 and move it 2.1–2.7. The finite difference must resolve a tenth of
the prediction, 0.0026 nats at the median and 1.4e-5 at the smallest (the NPR carrier); G-ROOM decides whether the fp32
floor allows that. The matched random poles' top contexts are unrelated token families (motivation, Atomic Robo, NASA),
as in N12. PENDING (the job writes them): every fp32 value.

**GPU part, written by the job**: `n2v/samples/raw_examples_{freeze,test}.jsonl`: the same births' four pairs with every
readout, the decoded exemplars, the carrier's five largest per-position changes with the token each predicts; opened at
§7 steps 3 and 4.

---

## 11. The three dimensions

- **Predictive.** G-VAL on the test half is a registered prediction of N12's reader (H_res), scored once under rules frozen
  before the data. Branch A turns N12's already-printed exploratory gradient labels into counted ones; that is not a new
  prediction.
- **Novel.** Small. Validating attribution against ablation is standard; measuring that the bf16 resolution of a summed loss
  hides a first-order effect at 7B, and validating it in fp32 with a finite difference and a forward-mode prediction, was
  not found in the searches. Its value is as an instrument check for this program.
- **Practical.** If the instrument passes, one backward per window ranks every pole of a layer by its first-order effect,
  replacing 14–51 GPU-h of per-pole ablation (§9.6); if it fails, that cost is the measured price of the competitor
  comparison.

---

## 12. Prediction grid (filled at unblinding, whatever the result)

Filled 2026-11-08 from `studies/next/n2v/verdict_n2v.json` (labels, test values, the re-reading), `studies/next/n2v/thresholds.json` (freeze values) and `banks/score-02/n2v/metrics.json` (gates, memory).

| item | rule or prior | observed | verdict |
|---|---|---|---|
| K-A | §3.1 (bitwise 0.85) | 40 of 40 pairs exact, max abs diff 0.0; rolled poles fail (min 0.0166) | as predicted |
| G-FP32, G-CAUSAL, I-FWD, I-JVP | §3.2 (0.9) | G-FP32 9 of 9; G-CAUSAL 9 of 9 points, 8,883 changed rows all bitwise before i0 (score-02, after A1; score-01 failed it, 428 of 963); I-FWD 187 of 187; I-JVP 32 of 32; every failing input fails | as predicted on score-02; G-CAUSAL took A1 |
| forward-mode memory at T_14 | ≤ 45 GiB (0.85) | 36.62 GB (`torch.cuda.max_memory_allocated`, 1e9 bytes), the same at every point | as predicted |
| G-ROOM window, fd, own (freeze; test) | §3.3 (0.9; 0.75) | window 0.99989; 0.99990; fd 0.99435; 0.99571; own 1.00000; 1.00000; failing input (N12's bf16 measurement as floor) 0.064; 0.076 | pass |
| G-VAL window (freeze; test) | ρ ≥ 0.7 etc. (0.6) | ρ 0.971 [0.939, 0.993], shuffle q975 0.104; ρ 0.960 [0.915, 0.990], shuffle q975 0.118; slope 1.110; 0.837 | PASS |
| G-FD (freeze; test) | (0.8) | ρ 0.978 [0.952, 0.993], shuffle q975 0.107; ρ 0.987 [0.971, 0.995], shuffle q975 0.116 | PASS |
| G-VAL own (freeze; test) | (0.7) | ρ 0.985 [0.971, 0.995], shuffle q975 0.106; ρ 0.986 [0.969, 0.996], shuffle q975 0.118 | PASS |
| window_jvp; bf16 against fp32 prediction | reported | window_jvp ρ 0.979 [0.948, 0.996], shuffle q975 0.111; ρ 0.967 [0.923, 0.994], shuffle q975 0.119 (PASS); bf16 against forward-mode prediction Spearman 0.987; 0.995, median relative difference 0.038; 0.026 | reported |
| branch | §1.2 | A: N12's banked bf16 first-order prediction tracks single-neuron zero ablation read in fp32 | A |
| N12 re-reading (branch A only) | known values: H_signal REFUTED, sel AT/AT, H_bridge UNDECIDED | `H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal REFUTED; H_bridge UNDECIDED; H_dt SUPPORTED`; H_signal REFUTED (0.527 [0.451, 0.600], bar 0.423); sel pb AT_SELECTION, dt AT_SELECTION; H_bridge UNDECIDED; N12's G-VAL with the fp32 measurement ρ 0.971 (freeze), 0.960 (test); ceiling not lifted | the known values; status: a dated re-reading, not a prediction |
| GPU-h | 6.53–9.78; kill 763 min (parts core,fd,jvp) | score-02 at most 5.93 GPU-h ($13.28); with score-01 (at most 0.66, $1.47) at most 6.59 GPU-h, $14.75 | under the center |

---

## 13. References

`06_CRITIC.md` N2, N12; `03_IDEAS.md` N2; `N12_SPEC.md`; `n12/READ_head.md`; `n12/READ_n12.md`; N6's spec §2 and its
L1-02 bank (branch `n6`); `12_SHAPING_NOTE.md` §4. AtP* (2403.00745), 2606.09899 and 2310.10348 at search-summary level
only; no number of theirs enters a rule.

---

## 14. Amendments

### A1 (2026-11-07): G-CAUSAL failed on the loss's memory alignment; score-01 does not count

**What happened.** score-01 (`i-03754787362f997e8`, g6e.2xlarge, launched 2026-11-07T16:06:49Z on `n2v-freeze-1a`)
passed G_SAME, INPUTS, G_CORPUS, t14_G_REV, t14_G_CFG, K_A (40 of 40 pairs exact) and t14_G_FP32, then stopped at
`t14_G_CAUSAL`: 428 of 963 changed rows equal their clean row bitwise before i0; the failing input failed as registered
(2 zero rows equal through i0). STATUS `exit=1` at 16:44:52Z. The job recorded only the count
(`studies/next/n2v/diag/score-01/metrics.json`). Nothing was unblinded.

**The cause, read from the code.** The 428 are exactly the rows whose batch slot has the same parity as their clean
row's slot. Each unit window adds 10 plain rows (clean, zero 0–3, fd 0–3, probe) to one list cut into batches of 4, so
every clean row sits in slot 0 or 2, and zero 1, zero 3, fd 1 and fd 3 always sit an even number of slots from it:
4 × 107 unit windows = 428 at T_14; the other five kinds, 535 rows, sit an odd number of slots away
(`studies/next/n2v/diag/score-01/slot_count.py` rebuilds the job's row order from the real plan →
`slot_count.json`). The one step of the forward whose result can depend on that parity is the loss. `ce_pos` computed
`F.cross_entropy(logits[b, :-1].float(), …)`. In fp32 `.float()` returns the same memory, so the log-softmax read a row
view of the [4, 2,049, 100,278] logits. One position's logits are 100,278 × 4 = 401,112 bytes, 8 bytes past a multiple
of 16, so position j of slot b starts on a 16-byte boundary when b · 2,049 + j is even and 8 bytes off it when it is
odd. PyTorch's CUDA log-softmax reads a row with 16-byte vector loads and handles a misaligned start by shifting which
thread sums which element, so the sum of exponentials is added in another order and rounds differently. Every other
row in the forward (4,096, 11,008 or 128 fp32 values) is a multiple of 16 bytes. In bf16 (N12's job, and K-A here)
`.float()` copies the row into a fresh buffer, where position j's alignment depends on j alone; that is why N12's
G-ABL-ID and this run's K-A were bitwise and the fp32 rows were not. The same cause would have failed I-FWD (clean rows
re-run in reversed order change slots) and I-JVP (reversal in batches of 2 swaps slots 0 and 1) at T_13. Not verified:
the CUDA kernel's order dependence was not run (the laptop has no GPU); the count and the alignment arithmetic are
checked. In this design the rows at an even slot distance are also exactly the rows of poles 1 and 3, so slot parity and
pole parity are the same set; the slot control below separates them on the instance.

**The change (code only; no rule).**
1. `ce_pos` copies each row's logits into a buffer of their own before the cross-entropy
   (`logits[b, :-1].to(torch.float32, copy=True)`), so bitwise equality before i0 is expected again. In bf16 this is the
   same single copy `.float()` made before.
2. G-CAUSAL records, per changed row, the largest |ℓ_changed(j) − ℓ_clean(j)| over j < i0, the number of positions that
   differ, and the batch and slot of the row and of its clean row; over the rows, the tolerance (0), the largest
   difference, the count over the tolerance, that count by slot parity, and the ten worst rows. The pass rule (every
   changed row bitwise equal to its clean row before i0) and the failing input (the zero rows through i0 must differ)
   are unchanged.
3. A report, `slot_control`, at the first point (T_14), before the plain rows: one unit window's clean row in all four
   slots of one batch; per slot, the largest difference against slot 0 of the logits, of the loss through the copy and
   of the loss through the row view (score-01's path), and the view's start offset mod 16 bytes. Predicted on the L40S:
   logits 0 and copy 0 at every slot; view 0 at slot 2 and above 0 at slots 1 and 3; offsets 0, 8, 0, 8. A view at 0
   in every slot refutes the diagnosis; G-CAUSAL's per-row record is then what is read.
4. I-FWD and I-JVP also record their largest difference (rules unchanged).

**Dry run.** 2026-11-07T18:42:47Z–18:46:56Z (`$SP/bio2/n2v_dry_a1`): `N2V_DRYRUN_OK`, 83 checks, 0 failed
(`studies/next/n2v/offline_gate/dryrun_a1.log`, `dryrun_a1_summary.json`). Seven checks are new. INSTR, on the tiny
Olmo 3 with the real 100,278-token vocabulary: the row view's start sits at 0, 8, 0, 8 bytes mod 16 in slots 0–3 and
every copy at 0; the copy's loss equals the view's bitwise on the CPU; `slot_control` runs. JOB: G-CAUSAL's per-row
record at 3 points with 0 rows over the tolerance; `slot_control` reported; I-FWD and I-JVP record 0. FAIL: the planted
early hook still stops the job at `t14_G_CAUSAL` (162 of 288 rows bitwise), and its record shows 126 rows over the
tolerance, largest 0.0812, the worst a zero row in batch 28, slot 1. The manifest check reads score-02's file. Every
other CHECK line equals freeze 1a's, values included (the tiny world's ρ, the dL recount), apart from the
loading-speed text. CPU kernels do not reproduce the L40S's alignment-dependent order, so the dry run cannot show the
fix working on the GPU; `slot_control` is that check, on the instance.

**score-01 DOES NOT COUNT.** The relaunch is `score-02` on tag `n2v-freeze-1b`, manifest
`infra/aws_pg/steps/next_n2v_score-02.json` (the same parts core,fd,jvp and job budget 700 min as score-01's), hard
kill 763 min, fresh prefix `results/next_n2v/score-02/`. The projection is unchanged (`n2v project` regenerates
`studies/next/n2v/projection.json` byte-equal; 6.53–9.78 GPU-h): the slot control is one batch of 4 rows, and the copy
is one 0.82 GB device copy per row. The copy adds one row's logits and tangent, about 1.6 GB, to the forward-mode rows'
peak; `points.14.jvp_max_mem_gb` measures it. Not changed and noted: score-01's T_14 download ran at 15.6 MB/s (935 s)
against the projection's 200–420 s fixed cost per point; at that rate the job's own stop rule may park the last points
(T_5, T_7 first).

§7 and §9.5 read `score-02` wherever they name `score-01`.

### A2 (2026-11-08): freeze 2 reads the freeze half only

**What happened.** score-02 ended with STATUS `exit=0` and every gate passing. At §7 step 3, with only the freeze files
banked, `n2v freeze` stopped with `FileNotFoundError` on `n2v_test.jsonl`: `n2v.read_bank` loaded both halves, and
`n2v.freeze` used only the freeze half. Loading the test file before freeze 2 is tagged breaks §7's order, so it was
not pulled. The dry run never showed this: its synthetic banks always held both files.

**The change (analysis code only; no rule, no job code).** `read_bank(bank, halves)`; `freeze` reads `("freeze",)`;
the verdict still reads both halves. A new dry-run check: freeze on a bank without the test file gives the same
thresholds as freeze on the full bank. Dry run on the A2 code, 2026-11-08T00:47:43Z–00:52:14Z (`$SP/bio2/n2v_dry_a2`):
`N2V_DRYRUN_OK`, 84 checks, 0 failed (`studies/next/n2v/offline_gate/dryrun_a2.log`, `dryrun_a2_summary.json`); every
other CHECK line equals A1's run. The test records stayed on S3, unopened, until `n2v-freeze-2` was tagged.
