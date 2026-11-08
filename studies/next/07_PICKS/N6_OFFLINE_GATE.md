# N6 offline gate (J11, edit the young)

**Answer.** The offline gate is green. The weight-free dry run printed `N6_DRYRUN_OK`, 78 checks, 0 failed
(`studies/next/n6/offline_gate/dryrun.log`); after amendment A1 (§6), 83 checks, 0 failed (`dryrun_freeze1a.log`); after
amendments A2 and A3 (§7), 99 checks, 0 failed (`dryrun_A2A3.log`); after amendment A4 (§8), 117 checks, 0 failed (`dryrun_A4.log`);
after amendment A5 (§9), 136 checks, 0 failed (`dryrun_A5.log`, `dryrun_summary.json`). Every gate whose quantity the banks hold was evaluated on both sides
at T_15, the one point N6 edits, and passed on its pass side and failed on its registered failing input
(`studies/next/n6/offline_gate/gate_sides.json`, `GATE_SIDES_OK`). The gates the banks cannot hold (every forward-dependent gate) were
shown on both sides in a tiny trained OLMo 3 on the CPU; that checks the code path, not the science.

Written 2026-11-06 by the N6 spec writer, worktree `interp_research-n6`, branch `n6`. Method: M2's amendment A5 (both sides of every
gate at every point where it is applied, from the banked files, before a freeze); N12's `gate_sides.py` as the pattern; M2's
`bank_scan.json` reused for the degenerate poles at T_15.

## 1. What was run

```
PYTHONPATH=. python3 -m infra.pg.n6 pairs --out studies/next/n6/pairs.json
PYTHONPATH=. python3 -m infra.pg.n6 e1 --out studies/next/n6/inputs/e1_known.json
PYTHONPATH=. python3 -m infra.pg.n6 projection --pairs studies/next/n6/pairs.json --out studies/next/n6/projection.json
PYTHONPATH=. python3 studies/next/n6/offline_gate/gate_sides.py --out studies/next/n6/offline_gate/gate_sides.json   # log gate_sides.log
PYTHONPATH=. python3 studies/next/n6/samples/raw_examples.py
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.n6_dryrun --out $SP/n6/dry/final3                       # log dryrun.log
```
`$SP` is the session scratchpad; the venv has torch 2.12.0, transformers 5.9.0, numpy 1.26.4 (the AMI's torch and transformers).
pyflakes is clean on `n6.py`, `n6_edit.py`, `n6_dryrun.py`, `gate_sides.py`, `raw_examples.py`. The tiny run's `metrics.json` (L1 and
L2) and raw examples are copied to `offline_gate/dryrun_tiny_{L1,L2}_metrics.json` and `dryrun_tiny_raw_examples.jsonl`.

## 2. Both sides of the bank-held gates, at T_15

| gate | pass side (banks) | registered failing input (banks) | power |
|---|---|---|---|
| G-BAL, stage 1 | 74 pairs: ρ_pair(age, ln max\|a\|) +0.027, SMD +0.005; ρ_pair(age, ln density) +0.027, SMD −0.004 (bounds 0.10) | 176 unmatched pairs: ρ_pair −0.739, SMD +1.488; +0.659, −1.135 | fails by 7× the bound |
| K1 (E1's zero value) | banked job_fall 0.6742178218025651 equals the mean of the 12 banked per-context falls exactly; E1 ran on an L40S with torch 2.12.0+cu130, transformers 5.9.0 | the cross arm's banked −0.0023431 | 677× the tolerance 1e-3 |
| K2 (corpus NLL, 120,000 targets) | two independent T_15 harvests (grid `w3`, 50M `w1`) both 2.675372037760417 | T_14's 2.714761865234375 | 39× the tolerance 1e-3 |
| G-REV | banked T_15 manifest revision = 373bad25… | forty zeros | fails |
| G-SAME | the T_15 harvest and E1's run: torch 2.12.0+cu130, transformers 5.9.0, NVIDIA L40S = the registered versions | a prefixed version string | fails |
| G-POLE | the banked T_15 exemplar activations in `arrays.npz` equal those of `contexts.jsonl` for all 197 graph poles (difference 0.0). This compared the bank with itself, not the job's recompute path with the bank (§6) | the T_14 values at the same positions: smallest max relative difference 0.150, median 0.662 (tolerance 0.05 by amendment A1, 0.01 at freeze 1; no pole at or below either) | 3.0× at the worst pole (15× at freeze 1) |
| degenerate poles | none at T_15 in any of the 32 layer files (`bank_scan.json` → `points.15`); the scan's own control equal 640 of 640 | — | — |
| eval windows | 96 windows, indices 10,045–24,525, wholly in band 3, outside the grid's 10,044 | — | — |
| eval windows, L2 (A4) | 1,024 windows, indices 10,045–24,663 (`n6.eval_windows(1024)`, list sha256 `58221917…`), wholly in band 3, outside the grid's 10,044 (dry run A19) | — | — |

No gate reads a Blackwell-harvested point (T_7–T_14). The ages come from J1's recount of all 16 points, as inputs; they are not gated.

## 3. The dry run (78 checks at freeze 1; five more after amendment A1, §6)

**Part A, the real banks (15 checks).** The critic's within-layer Spearman reproduced (−0.5915 against −0.592; +0.5018 against +0.502);
`pairs.json` rebuilt equal to the committed file; the balance gate on both sides; every pair in one layer, different neurons,
\|Δ age\| ≥ 3, inside the calipers, each pole once; the L1 inputs' sha256 equal to `n6_edit.INPUT_SHA["L1"]`; Wilson against known
values; the eval windows; the pair-stratified Spearman equals 1 − 2f. Stage 2 on **planted** L1 arrays (a simulation: column norms
lognormal sd 0.15, E⁺ from the bank statistics with noise sd 0.3) kept 67 pairs, balanced on all four quantities (SMD ln ‖w‖ −0.079,
ln E⁺ +0.016); a planted age-dependent column norm left no balanced point and stopped with "L2 does not launch".

**Part B, the tiny world (46 checks).** A 6-layer OLMo 3 (hidden 32, vocabulary 256, 2,049-token windows, sliding window 64 so that
the layer types matter), trained 500 Adam steps (final loss 1.93) so that its losses carry signal. L1 on the stage-1 pairs, then the
stage-2 pairs from L1's arrays, then L2:

| check | result |
|---|---|
| G-SAME, G-REV, K1, K2, G-POLE, G-FP32, K3, G-INST, G-ID, G-CACHE, G-EPS, G-ZN, G-NORM, G-SAMPLES | pass side passes, failing input fails, each |
| K2 against a second implementation (one window at a time) | equal (2.22086) |
| G-EPS | fp32 probes at most 1.9e-8 nats per token; bf16 probes −1.5e-4 and 4.7e-6 (both sides of the 1e-6 bound) |
| K3's failing input (targets one position off) | 7.98 against 2.22 |
| the progression from layer 0 against the model's forward | bitwise equal; skipping layer 0 differs by 16.2 in a logit |
| col at κ = 1 against zero written into the pole's own-sign inputs (E1's operator restricted to the sign) | max logit difference 4.8e-6 (fp32) |
| E⁺ against an independent loop; column norms | equal (rel 1e-5); equal |
| c = min of the two column norms, equal within a pair; κ = c / ‖w‖, κ = 1 on the smaller-norm member | yes |
| zero edit bitwise equal to base, ΔL exactly 0; every col edit differs | yes |
| the hook's E⁺ against pass 1's | rel error at most 1.1e-7; the neighbouring neuron's 71 against 849 |
| planted failures: wrong torch version, wrong snapshot, wrong E1 value, E1 failing input equal to the target, wrong K2 target, the bank at 0.7 × a, column norms 1.1 × L1's, a stage-1 pairs file given to L2 | each stops the job with the gate's message |
| stop rule | a budget of 0.55 × the full run's time parks the remaining edits; the job ends normally |
| L1 against L2 on the same edits | bit for bit |
| G-PC / G-DOSE logic | col > rand everywhere passes and its swap fails; col > rand in 16 of 32 fails |

The tiny model's own G-PC count (3 of 6) is reported only: a tiny model need not have a positive control that fires.

**Part C, freeze and verdict (17 checks).** Every label branch from planted records: SUPPORTED, SUPPORTED_UNSPECIFIC, REVERSED, FLAT,
UNDECIDED_NO_POWER, UNDECIDED, ABOVE_CEILING; DOES NOT COUNT from a failed gate and from a failing input that passed, with the realized
label kept; G-RES fails at a planted floor of 5e-5; the room at 35 test pairs (Wilson at 0.5: [0.343, 0.657]; 24 wins lift the lower
bound above 0.5); one parked pair listed, eight parked → G-PARK; H_eff reads collateral per unit efficacy; the practical saving equals
D/2 against a random pick; freeze reads only the freeze half.

## 4. What the banks cannot check, said plainly

- The pass side of K1, K2 and G-POLE on this job's forward. They assume same-device, same-version bit-identity, which E1 showed
  (a_ours equal to a_bank in `known_number.json`) and the two T_15 harvests show (K2 equal to the last digit).
- K3, G-FP32, G-ID, G-ZN, G-CACHE: no fp32 forward of O7 is banked.
- G-EPS on both sides: no forward with a rounding probe is banked. The evidence for the failing side is N12's measured bf16 noise of a
  window-summed loss (median 0.65 nats, `n12/READ_n12.md` §6). Whether the fp32 floor lies below 1e-6 nats per token on the 7B model is
  measured first by L1.
- G-NORM, G-INST and stage 2 of the pairing: the column norms and the energies on the eval windows are not banked. L1 banks them; the
  stage-2 count of pairs is unknown until then (a simulation gave 67).
- G-PC and G-DOSE: no edit's corpus loss is banked. Whether the layer's highest-energy pole beats a random direction of the same norm
  in 24 of 32 layers is the first thing L2 shows; a massive-activation pole may make both edits catastrophic in some layers.
- G-POLE's neighbour-neuron failing input on the real model: neuron n + 1 is not a grid pole.
- The fp32 rate on an L40S (25–60 TFLOP/s assumed) and the memory of 29.2 GB of weights plus 6.4 GB of kept residuals at batch 4.

## 5. Files

| file | what |
|---|---|
| `studies/next/07_PICKS/N6_SPEC.md` | the pre-registration |
| `studies/next/07_PICKS/N6_OFFLINE_GATE.md` | this record |
| `infra/pg/n6.py` | the pure logic and CLI |
| `infra/aws_pg/jobs/n6_edit.py` | the job |
| `infra/aws_pg/jobs/n6_dryrun.py` | the dry run |
| `studies/next/n6/diag/L1-01/{cause.md, A1_g_pole_197.tsv, metrics.json}` | L1-01's stop at G-POLE and the 197 values behind amendment A1 |
| `infra/aws_pg/steps/next_n6_L1.json`, `next_n6_L2a.json`, `next_n6_L2b.json` | the step lists (A5: L2a, the freeze half, and L2b, the test half, replace `next_n6_L2.json`) |
| `studies/next/n6/pairs.json` | stage-1 pairs, balance, the failing input, the poles' exemplars, the eval windows |
| `studies/next/n6/inputs/e1_known.json` | K1's banked value, its inputs and its failing input |
| `studies/next/n6/projection.json` | GPU-hours of L1 and L2 at three rates |
| `studies/next/n6/offline_gate/{gate_sides.py, gate_sides.json, gate_sides.log}` | both sides of the bank-held gates |
| `studies/next/n6/offline_gate/{dryrun.log, dryrun_summary.json, dryrun_tiny_L1_metrics.json, dryrun_tiny_L2_metrics.json, dryrun_tiny_raw_examples.jsonl}` | the dry run (summary and tiny metrics from the A2/A3 run; `dryrun.log` the draft's, `dryrun_freeze1.log`, `dryrun_freeze1a.log`, `dryrun_A2A3.log` the reruns) |
| `studies/next/n6/diag/{g_res_preview.py, L1-02/g_res_preview.json}` | G-RES's preview on L1-02's ten column edits, its path check and the eval-set noise of each L1 pair's contrast |
| `studies/next/n6/diag/{a4_preview.py, L1-02/a4_preview.json}` | amendment A4's two rules on L1-02's banks: G-RES at the factor 50; G-POWER at 96 windows and scaled to 1,024 |
| `studies/next/n6/projection_l1.py`, `projection_L1.json` | L2's projection at L1-02's rate on A4's eval set and roster, the GPU and host memory per phase |
| `studies/next/n6/samples/{raw_examples.py, raw_examples_zero_gpu.json, raw_examples_zero_gpu.md}` | the ten raw examples, zero-GPU part |

## 6. Amendment A1 (2026-11-06): the gap G-POLE found, and the rule

**The offline gate missed a gate whose two sides compare two computation paths.** G-POLE compares the bank (the harvest's full
2,049-token windows in batches of 8, bf16) with a recompute on another path (right-padded prefixes cut at each exemplar, sorted by
position, batches of 8, bf16; `reuse_ablate.readout`). Its pass side here was the bank against itself, 0.0 by construction (§2), so
it could not show the kernel rounding between the two paths. L1-01 measured that rounding: 0 to 0.01506 over 197 poles, 6 poles
above the freeze-1 tolerance 0.01; the job stopped (`studies/next/n6/diag/L1-01/cause.md`). The failing input sat at 0.923 or more.

**The rule, for every later gate of this kind:** a gate that compares two computation paths needs its tolerance frozen from a
control pair measured on those two paths, or it is reported rather than gated.

G-POLE does not meet the rule. By the head's decision it stays a gate, at 0.05 on the same statistic (spec §14 A1): 3.3 times the
worst pass-side value of L1-01, 18 times below the smallest failing-input value, 3.0 times below the offline T_14 failing input.
That bar was set after seeing the pass side of one run. All 197 values are in `studies/next/n6/diag/L1-01/A1_g_pole_197.tsv`.
The job now writes the recomputed g, u and a per pole and exemplar beside the bank's a (`g_pole.npz`) and the peak GPU memory per
phase (`metrics.json` → `measured.peak_mem_gib`).

The dry run on the amended code: `N6_DRYRUN_OK`, 83 checks, 0 failed (`studies/next/n6/offline_gate/dryrun_freeze1a.log`,
`dryrun_summary.json`). The five added checks: `g_pole.npz` has the shapes, positions and bank values of the pairs file, a =
act(g) u, and its statistic equals `per_pole`; the peak memory is recorded for each of the nine phases (None on the CPU); a planted
G-POLE value of 0.03 passes at 0.05; a planted 0.06 stops the job with "G_POLE FAILED" and its recorded value is 0.06. Other gates
of N6 that compare two paths: K1 and K2 compare this job's forward with banked forwards on the same paths (E1's readout, the
harvest's windows), and both were exact in L1-01. K3 compares fp32 with bf16 on the same windows at a tolerance of 0.02, set
before any fp32 number existed and not from a control pair. G-ZN and G-CACHE compare the progression (layers called one by one,
the head applied to one window at a time) with the model's own forward (the head applied to a batch of 4 windows) at a bitwise
bar; on the CPU they were bitwise equal, and L1-02 is their first measurement on the GPU. These three fall under the rule's gap
and are named here; none is changed by A1.

## 7. Amendments A2 and A3 (2026-11-06, before freeze 1b)

**A2 applies the two-path rule to G-ZN and G-CACHE.** L1-02 is their first measurement on the GPU and their control on the two paths:
the zero edit (δ = 0) read on the progression's path (head per window) against base pass 1 (the model's forward, head on a batch of
4) gave ΔL −3.015e-11 and a largest per-token difference of 9.537e-7 nats. The bars are now \|ΔL\| ≤ 1.221e-7 (G-ZN) and \|mean\| ≤
1.221e-7 with largest ≤ 1e-5 (G-CACHE), set after seeing those values (spec §14 A2). Both failing inputs are planted and fail: the first
column edit's ΔL; each window scored from another window's final residual (computed in the job). The job records the head's input
shapes per phase, so the two paths' shapes are measured, not assumed.

**A3 changes the stage-2 split** (spec §14 A3): 24 test pairs, 21 freeze pairs; the four pairs whose outcome L1 banked stay in the
freeze half.

The dry run on the amended code: `N6_DRYRUN_OK`, 99 checks, 0 failed (`studies/next/n6/offline_gate/dryrun_A2A3.log`); the 83 earlier
CHECK lines are unchanged. `gate_sides.py` prints `GATE_SIDES_OK` with output equal to the committed `gate_sides.json`.

**G-RES is not changed.** It compares the fp32 floor F_eps with the 10th percentile of \|ΔL_col\|. It is not a two-path gate: the
probe and the zero edit run the same path, and the zero edit's ΔL (the path offset) is 0.25 % of F_eps
(`studies/next/n6/diag/L1-02/g_res_preview.json`). Its preview on L1-02 fails by 1.14 because the column edits are small (median
\|ΔL_col\| 2.2e-6 nats per token), not because of a defect; the decision is the head's.

## 8. Amendment A4 (2026-11-06, before freeze 1b)

**A4 changes G-RES's factor from 100 to 50 and changes L2's design for power** (spec §14 A4): 1,024 eval windows instead of 96; G-POWER
at freeze 2 (τ_s ≤ 0.5 × the median \|D_col\| over the freeze pairs, else UNDECIDED_NO_POWER before the test half is read); the positive
control in the 8 layers 0, 4, …, 28 with G-PC and G-DOSE at 6 of 8. Two consequences in the code: the kept residual of 1,024 windows
(34.4 GB) lives in host memory and goes to the GPU in chunks of 96 windows; G-INST reads its E⁺ in its own pass over L1's 96 windows.

Both sides of every new or changed rule, in the dry run on the amended code (`N6_DRYRUN_OK`, 117 checks, 0 failed,
`studies/next/n6/offline_gate/dryrun_A4.log`):

| rule | pass side | failing side |
|---|---|---|
| G-RES, factor 50 | a floor at 0.015 × q10 passes (it would fail at the factor 100) | 0.05 × q10 fails; F_eps 5e-5 fails |
| G-POWER | sd at 0.4 × median \|D_col\| passes and the verdict reads the test half | 0.6 fails: UNDECIDED_NO_POWER for H_age, H_adj, H_eff, no test pair read (the same with every test record removed); with a failed gate, DOES NOT COUNT |
| G-PC, G-DOSE at 6 of 8 | 6 of 8 passes | 5 of 8 fails; the swapped labels fail |
| chunks | one chunk of 4 windows and two chunks of 2 give equal records, field by field | — |
| G-INST on L1's windows | an 8-window L2 eval set: G-INST reads L1's 4 windows and passes | G-INST read on L2's eval set stops the job |

On L1-02's banks (`studies/next/n6/diag/L1-02/a4_preview.json`): G-RES at the factor 50 passes (F_eps / bound 0.568); G-POWER reads 1.42
at 96 windows (fails) and 0.435 when τ_s is scaled to 1,024 independent windows with \|D_col\| held (passes), so the rule has room on both
sides. What the banks cannot show: the realized τ_s and \|D_col\| at 1,024 windows; the GPU and host memory of the chunked progression and
the host-device copy rate (projected in `projection_L1.json`; the job records `peak_mem_gib` per phase and `host_mem`).

Seven of the 99 earlier CHECK lines changed (A17's sha256, the peak-memory phases with `inst`, G-PC on the positive-control layers, the
tiny plan's 59 edits, the stop rule's counts, two lines' wording); the other 92 are identical. `gate_sides.py` prints `GATE_SIDES_OK`,
output equal to the committed `gate_sides.json`. pyflakes is clean on the changed files.

## 9. Amendment A5 (2026-11-06, before freeze 1c)

**A5 runs L2 as two launches** (spec §14 A5): L2a reads the freeze half and runs every gate; freeze 2 and G-POWER read its bank alone.
L2b reads the test half, after freeze 2 is tagged with G-POWER passed. The split is one argument of the job (`--half freeze|test`) on one
pairs file (`pairs_L2.json`, sha256 unchanged); `n6.plan(half=)` selects the roster. L2b checks L2a's `metrics.json` (`--against`)
before its first gate: stage L2, half freeze, complete with nothing parked, and the same eval windows, eval tokens and pairs content
(sha256). The banks are two directories, `banks/L2a-01/` and `banks/L2b-01/`; `n6 freeze --bank L2a` and `n6 verdict --bank L2a --bank
L2b` read them (`n6.read_banks`), L2b's gates under the prefix `L2b:`.

Both sides of every new rule, in the dry run on the amended code (`N6_DRYRUN_OK`, 136 checks, 0 failed,
`studies/next/n6/offline_gate/dryrun_A5.log`):

| rule | pass side | failing side |
|---|---|---|
| the split (real plan) | L2a 113 edits, L2b 99; in both only the two pair probes and the zero edit; together the 209-edit plan in its order | — |
| the split (tiny world) | every L2a and L2b record equals the one-launch record of that edit, field by field; L2b's 13 gates pass, each failing input fails; L2a's G-PC, G-DOSE, G-SAMPLES equal the one launch's | — |
| L2b identity | L2b against the tiny L2a passes and records L2a's four values | an L2b with another eval set stops before G-SAME (no gate, no record); an L2b against an L2a that parked stops |
| freeze 2 on L2a | `n6 freeze --bank L2a` equals freeze on the one launch | `n6 freeze` given L2b's bank stops |
| the verdict on both banks | `n6 verdict --bank L2a --bank L2b` (G-POWER planted to pass) equals the one launch's verdict in every field, 6 of 6 test pairs read | an L2b bank with another eval set stops; G-POWER passed with L2a alone stops; a planted failure of L2b's K3 makes every label DOES NOT COUNT |
| G-POWER failed | the verdict on L2a alone: UNDECIDED_NO_POWER, no test pair read | — |
| the job's CLI | both manifests pass the job's argument checks (`--half freeze`; `--half test --against` L2a-01's `metrics.json`), `--budget-min` equal to `projection_L1.json`'s job stop rules | stage L2 without `--half` stops; `--half test` without `--against` stops |

The 117 earlier CHECK lines are identical. `gate_sides.py` prints `GATE_SIDES_OK`, output equal to the committed `gate_sides.json`.
pyflakes is clean on the changed files. What the dry run cannot show: whether two L40S instances give bitwise-equal records (the three
edits that run in both launches show it on the GPU; K1 and K2 were exact across instances before).
