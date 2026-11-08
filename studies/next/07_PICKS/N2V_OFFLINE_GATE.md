# N2V offline gate (weight-free), 2026-11-06

**Go for the head's review; nothing launched.** The weight-free dry run printed `N2V_DRYRUN_OK` after 72 checks, 0 failed
(`studies/next/n2v/offline_gate/dryrun.log`, `dryrun_summary.json`). Every gate whose quantity the banks hold was
evaluated on both sides on N12's banked records (`studies/next/n2v/offline_gate/gate_sides.json`): each registered failing
input fails with room, and each pass side passes on a planted fp32 measurement. The job ran from its entry point in N12's
tiny world (N12's own tiny job wrote the records it reads), with every gate passing and every failing input failing, and
each instance gate stopped the job on its planted failing input. Pass sides that need a real fp32 loss of this model are
not in any bank and are stated as such.

Worktree `interp_research-bio2`, branch `bio2`, from `clean-neurons` at `ae8dc637f`. CPU torch 2.12.0, transformers 5.9.0
(`$SP/m1_venv`). Commands: `N2V_SPEC.md` §9.3.

## What was run

| step | output | result |
|---|---|---|
| `gate_sides.py` (10 s) | `gate_sides.json` | below |
| `mem_fp32.py` (25 s) | `mem_fp32.json` | fp32 math-kernel backward 1.145 GB per layer, 67.2 GB at layer 0, 48 of 153 real units fit the L40S; the method's control 36.92 GB against N12's measured 36.04 |
| `raw_examples.py` | `studies/next/n2v/samples/raw_examples_zero_gpu.{md,json}` | 40 pairs of N12's ten freeze example births |
| `n2v project` | `studies/next/n2v/projection.json` | 6.53–9.78 GPU-h, hard kill 763 min (core 2.61–3.56; core,fd 3.90–5.10) |
| `n2v_dryrun` (about 6 min) | `dryrun.log` | `N2V_DRYRUN_OK`, 72 of 72 (plus N12's tiny-world builder, 17 of 17) |

## Both sides of every gate, on the banks

| gate | failing side (registered input) | pass side | where |
|---|---|---|---|
| K-A | N12's banked dL with poles rolled inside each of the 10 K0 units: smallest gap 0.0166 nats against 1e-6 | not banked (no second bf16 run of these rows); tiny world: 40 of 40 pairs bitwise | `gate_sides.json` → `K_A`; dry run JOB |
| G-VAL window | N12's bf16 measurement: ρ 0.0996 (freeze), 0.0425 (test), fails (N12's values reproduce); the sign flip fails on the planted pass | planted fp32 (prediction × (1 + 5 % noise) + 1e-4): ρ 0.99978, 0.99973 | `halves.*.G_VAL_*` |
| N12's ceiling and the inspection | a copy of the prediction: n_equal 328 / 284, not cleared | the planted 5 % deviation: above 0.999, median relative deviation 0.035, cleared | `halves.*.G_VAL_planted_*` |
| G-FD | the sign flip fails; a copy reads above its ceiling | planted (0.1 × prediction × (1 + 1 % noise) + 1e-4): passes, not above its ceiling | `halves.*.G_FD_*` |
| G-ROOM | N12's bf16 measured change of the carrier as each unit's floor: median ρ 0.063 (freeze), 0.077 (test) | a planted floor of 1e-4: 0.99998 (window), 0.999 (fd); the largest uniform floor with room: 0.0169 / 0.0174 (window), 0.00169 / 0.00174 (fd) | `halves.*.G_ROOM_*`, `largest_floor_*` |
| the re-reading | N12's bf16 measurement: N12's own verdict, gradient labels DOES NOT COUNT; a planted fp32 pass above the ceiling and not cleared: DOES NOT COUNT | planted fp32 and cleared: `H_signal REFUTED; H_bridge UNDECIDED` with H_sel_signal_pb and _dt AT, the values the reader computed by hand (`READ_n12.md` §5) | `reread` |
| the re-reading's inputs | — | 442 of 442 records carry `pull`; 153 real carry `u_matched`; 289 twins carry `u_S`; `floor_u_freeze` present: no gradient run is needed | `reread_inputs` |
| G-SAME, INPUTS, G-FP32, G-CAUSAL, I-FWD, I-JVP | planted in the tiny world: each stops the job naming it | tiny world: pass | dry run FAIL, JOB |

## What the dry run shows beyond the banks

- INSTR (tiny fp32 Olmo 3, the job's own row functions): forward-mode prediction 0.1190132 against reverse-mode
  Σ −h·G 0.1190135; the neighbouring neuron's tangent differs; the 10 % finite difference 0.012200 against a tenth of the
  prediction 0.011901 (2.5 %); the probe 9.5e-7 nats against a zero ablation of 0.1335; G-CAUSAL bitwise before i0 for
  zero, fd and probe rows, and the own-loss tangent exactly 0 before i0.
- JOB: K-A 40 of 40 pairs bitwise against the tiny N12 job's banked dL (other batch companions); 18 failing inputs all
  fail; the fp32 zero ablation tracks the fp32 forward-mode prediction (ρ 0.959 over 160 pairs); a pair's dL recounts from
  the banked per-row losses; `metrics.json` carries no Spearman or label.
- FAIL: wrong versions, a zero units sha256, a hook one index early (G-CAUSAL: 162 of 288 rows bitwise), a second clean
  pass 1e-3 off (I-FWD), a second forward-mode pass × 1.001 (I-JVP), float32 precision "high" (G-FP32), N12's dL + 1e-3
  (K-A): each stops the job at the named gate; a tiny budget parks T_13 and T_12 and the job ends normally.
- VERDICT on synthetic banks over N12's real records: branches A (with the re-reading), B, C, D; a copy above the
  ceiling; parts core only (fd and own NOT_RUN, branch C); freeze refuses on a failed gate and on a G-ROOM failing input
  that passes; the verdict refuses a changed freeze file.

## Gaps, said plainly

1. No bank holds an fp32 loss of this model at these positions, so no pass side of G-VAL, G-FD or G-ROOM is shown on
   real data; the expected fp32 floor (about 1e-4 nats per unit) is an extrapolation from N6's probe (1.2e-8 per token).
2. The forward-mode rows' memory on the L40S is estimated (about 38 GB), not measured; the job records it at T_14.
3. The forward-mode row's cost (2× a plain row) is estimated.
4. G-CAUSAL and I-FWD assume a row's fp32 loss does not depend on its batch companions on the L40S; shown in bf16 by N12
   (G-ABL-ID 0.0) and on the CPU here, not in fp32 on the L40S.
5. The "usable units" refusal (< 50 per half) is asserted in code but was not planted.

## Freeze 1 (the head's decisions of 2026-11-06 19:25Z, applied 2026-11-07)

Spec §0.6 holds the decisions: forward-mode differentiation in fp32 is the registered way to an fp32 first-order
prediction; the launch runs parts core and fd (`--parts core,fd`); the own-loss readout does not ride on those rows
(its prediction needs the forward-mode rows), so G-VAL own and window_jvp read NOT_RUN; the ceiling inspection, K-A
(1e-6, bitwise expected) and G-ROOM as written. The steps file now reads `--parts core,fd --budget-min 360`; the
projection for these parts is `studies/next/n2v/projection_core_fd.json` (3.90–5.10 GPU-h, hard kill 398 min;
regenerated by `n2v project --parts core,fd`, byte-equal to the file the first executor wrote).

Dry run on the freeze-1 code, 2026-11-07T15:25:57Z–15:29:43Z (`$SP/bio2/n2v_dry_f1b`): `N2V_DRYRUN_OK`, 75 checks, 0
failed (`studies/next/n2v/offline_gate/dryrun_f1.log`, `dryrun_f1_summary.json`), plus N12's tiny-world builder, 17 of
17. Three checks are new; the 72 earlier CHECK lines are equal to the draft's run except the loading-speed text in the
value column of three JOB lines.

| new check | what it shows | value |
|---|---|---|
| REAL the keys the job reads from `units.json` exist in the registered file | N1 A1's lesson (a dry run's planted input carried a key the real file lacks): N12's `units.json` has `units` and `examples`, 10 per half, each a real unit of its half | keys `births_sha256, dt_missing, examples, n_units, rule, units` |
| JOB parts core,fd | the job from its entry point with `--parts core,fd`: exits 0, every gate passes, every failing input fails, no I-JVP and no forward-mode value; I-FWD runs; dL, dL_own, dL_fd, dL_fd_own and the probe equal the full run's bitwise | 160 pairs, 17 gates |
| VERDICT parts core,fd | synthetic banks without forward-mode values: own and window_jvp NOT_RUN; branch A on a tracking measurement, branch B on a noisy window with a passing finite difference | A; B |

## Freeze 1a (the coordinator's decision of 2026-11-07)

The forward-mode part is added so the own-next-token-loss readout has its prediction (spec §0.6 item 8). The manifest
reads `--parts core,fd,jvp --budget-min 700`; the launch's hard kill is 763 min (1.3 × 9.78 h). The full-parts
projection regenerates byte-equal to `studies/next/n2v/projection.json` (6.53–9.78 GPU-h).

Dry run on the freeze-1a code, 2026-11-07T16:01:27Z–16:05:19Z (`$SP/bio2/n2v_dry_f1a`): `N2V_DRYRUN_OK`, 76 checks, 0
failed (`studies/next/n2v/offline_gate/dryrun_f1a.log`, `dryrun_f1a_summary.json`). Against freeze 1's run: one check is
new (`REAL the manifest runs the registered parts and job budget`: `--parts core,fd,jvp --budget-min 700`); the two
core,fd checks are renamed from "the head's choice" to "a supported path", with the same values; every other CHECK line
is equal apart from the loading-speed text. The JOB section's main launch has always run parts core,fd,jvp: K-A, G-FP32,
G-CAUSAL, I-FWD and I-JVP pass and fail on their planted inputs, and the fp32 zero ablation tracks the forward-mode
prediction (ρ 0.959 over 160 pairs) in the tiny world.

## Freeze 1b (A1, 2026-11-07): the loss's alignment

score-01 stopped at `t14_G_CAUSAL` with 428 of 963 changed rows bitwise before i0. The 428 are exactly the rows at an
even slot distance from their clean row in the batches of 4 (`studies/next/n2v/diag/score-01/slot_count.json`, rebuilt
from the real plan). The fp32 loss read each row's logits in place; a position's 100,278 fp32 logits are 8 bytes past a
multiple of 16, so the row view's alignment alternates with the slot and the CUDA log-softmax sums in another order.
A1 (`N2V_SPEC.md` §14): `ce_pos` copies each row's logits before the cross-entropy; G-CAUSAL records per-row
differences (largest difference before i0, positions that differ, batch and slot of the row and its clean row, count
over the tolerance 0 by slot parity, the ten worst rows); the report `slot_control` runs one window's clean row in all
four slots through the copy and the row view at the first point; I-FWD and I-JVP record their largest difference. The
manifest is `infra/aws_pg/steps/next_n2v_score-02.json`, equal in content to score-01's. The projection regenerates
byte-equal (6.53–9.78 GPU-h, hard kill 763 min).

Dry run on the A1 code, 2026-11-07T18:42:47Z–18:46:56Z (`$SP/bio2/n2v_dry_a1`): `N2V_DRYRUN_OK`, 83 checks, 0 failed
(`studies/next/n2v/offline_gate/dryrun_a1.log`, `dryrun_a1_summary.json`). Against freeze 1a's run:

| new check | what it shows | value |
|---|---|---|
| INSTR A1 alignment | the tiny Olmo 3 has the real vocabulary; the row view's start mod 16 bytes alternates with the slot; each copy starts aligned | vocab 100278; view 0, 8, 0, 8; copy 0, 0, 0, 0 |
| INSTR A1 ce_pos | the copy does not change the loss on the CPU: bitwise equal to the row view's | equal |
| INSTR A1 slot_control | runs; four slots per path; every path 0 on the CPU | 0 |
| JOB A1 G-CAUSAL record | at every point: tolerance 0, largest 0, none over, ten worst rows with batch and slot | 3 points; 288, 297, 243 rows |
| JOB A1 slot_control | reported at the first point | copy and view 0 |
| JOB A1 I-FWD, I-JVP | record their largest difference | 0, 0 |
| FAIL causal_early record | the planted early hook still stops the job at `t14_G_CAUSAL`; the record shows the planted rows | 126 of 288 over; largest 0.0812; worst a zero row, batch 28, slot 1 |

The manifest check now names score-02. Every other CHECK line is equal to freeze 1a's, values included, apart from the
loading-speed text. Gap, said plainly: CPU kernels read misaligned rows in the same order, so no laptop check can show
the GPU difference or its removal; the instance's `slot_control` report is that check.
