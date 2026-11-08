# N12 offline gate (the record)

**Answer.** The weight-free dry run printed `N12_DRYRUN_OK`: **94 checks, 0 failed**
(`studies/next/n12/offline_gate/dryrun.log`; values in `dryrun_summary.json`). After amendment A1 (2026-11-06):
**100 checks, 0 failed** (`dryrun_a1.log`, `dryrun_summary_a1.json`; section "Amendment A1" below). Every gate whose quantity the banks
hold was evaluated on both sides at every point where it is applied, from the banked files, before any launch (the
method of M2's amendment A5): 12 gate families, pass side passing and registered failing input failing at every
point the bank covers (`offline_gate/gate_sides.json`, log `gate_sides.log`). Three gates have no bank at all, because
no gradient or ablation of this model was ever banked: G-VAL, G-ABL-ID and I-GRAD. Their two sides are shown only in
the dry run (an fp64 tiny model against finite differences, the tiny world, planted inputs). The cross-device G-EX at
the Blackwell-banked points T_7–T_14 is reported, not gated, and G-TOPK is the gate there (M2's A3).

`SP` below means
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad`.
`BANK` means `/Users/abaybektursun/projects/interp_research_banks/neurons`.

## What was run

```
cd /Users/abaybektursun/projects/interp_research-bio
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n12 units --out studies/next/n12/units.json          # twice, same sha256
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n12 project --units studies/next/n12/units.json > studies/next/n12/projection.json
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n12/samples/raw_examples.py --out studies/next/n12/samples --cache $SP/n12/stats_cache
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n12/offline_gate/gate_sides.py --out studies/next/n12/offline_gate/gate_sides.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.n12_dryrun --bank $BANK --hf $SP/m2/hf \
    --sides studies/next/n12/offline_gate/gate_sides.json --examples studies/next/n12/samples/raw_examples_zero_gpu.json \
    --out $SP/n12/dry
```
- Environment: the scratchpad venv `m1_venv` (Python 3.12, CPU torch 2.12.0, transformers 5.9.0, numpy 1.26.4,
  pyflakes), the one M1 and M2 used. pyflakes is clean on the five new files. Wall time of the dry run: 6 min 32 s.
  The log keeps every CHECK line and drops the progress bars.
- `$SP/m2/hf`: M2's tokenizer files and branch configs (`M2_OFFLINE_GATE.md`, "What was run").
- `units.json`: sha256 `c223dfee261c3f6019ad46d08cccc23859143e6f59698aa0ef24de769f95508a`, reproduced byte for byte by
  a second build and by the dry run; registered as `n12_score.UNITS_SHA`.
- No weights were downloaded. The tiny models are random tensors made on the laptop.

## Fetched (read-only)

- S3, profile `fuelos`, `us-east-1`: M2's job vectors (`results/next_m2/score-06/out/score/m2/vectors/`, and the same
  paths of score-05 and score-04): 20 files for the raw examples, 10 carriers' pre-role files at T_14 for K0, up to 6
  files per point per run for G-M2V; byte-range reads of the members `hist`, `hist_edges`, `n_pole_tokens`, `max_abs`,
  `q999` of 20 stats files (`results/neurons/w3/grid/<dir>/stats/layer_LL.npz`) for the ten raw examples; `aws s3 ls`
  of the vector prefixes. Nothing else.
- Laptop banks: the 16 grid points' `exemplar_a.npz` and the T_15 `contexts.jsonl` (sha256 registered in `m2.py`),
  `next/recount2/arrays.npz`, `w4/o7_s_v2/explain/clusters.jsonl`.
- Repo files: M2's `metrics.json`, ranks files and raw examples (`studies/next/m2/banks/score-06/m2/`), M2's
  `a5_gate_sides.json` (the harvest's G0 controls, G-REV against the HF refs) and `bank_scan.json`, the committed
  branch configs (`studies/next/m2/offline_gate/hf/`).
- Web: four searches (`N12_SPEC.md` §1.5).

## Both sides of every gate at every point, from the banks (the A5 table)

Per point (`gate_sides.json` → `gates`; score-06 is M2's completed L40S run; "second run" is score-05, and from T_10
also score-04, on the same windows):

| point | G0 pass side: harvest / score-06 max (≤ 0.01) | G0 failing input u × 1.1, min over layers | G-M2V pass side: second L40S run against score-06, exact share (entries) | G-M2V failing input (axis rolled), p99 against 0.021159 | G-EX: N12's entries; pass p99 in score-06 | G-EX failing input (neighbour's bank), times the tolerance | G-TOPK pass p99 / failing p99, score-06 |
|---|---|---|---|---|---|---|---|
| T_15 | 0.00511 / 0.00607 | 0.0944 | score-05 1.000 (154,112); score-04 1.000 | 1.199 | 1,285; 0.0000 (gated) | 62 | 0.0000 / 1.128 |
| T_14 | 0.00539 / 0.00503 | 0.0950 | score-05 1.000 (528,384); score-04 1.000 | 2.439 | 921; 0.0289 (reported) | 64 | 0.0142 / 1.218 |
| T_13 | 0.00428 / 0.00476 | 0.0948 | score-05 1.000 (616,448); score-04 1.000 | 3.480 | 1,345; 0.0519 (reported) | 139 | 0.0132 / 1.218 |
| T_12 | 0.00617 / 0.00521 | 0.0943 | score-05 1.000 (352,256); score-04 1.000 | 2.894 | 1,378; 0.0325 (reported) | 172 | 0.0158 / 1.156 |
| T_11 | 0.00591 / 0.00465 | 0.0964 | score-05 1.000 (418,304); score-04 1.000 | 2.848 | 1,152; 0.0640 (reported) | 238 | 0.0130 / 1.075 |
| T_10 | 0.00474 / 0.00501 | 0.0950 | score-05 1.000 (352,256); score-04 1.000 | 6.495 | 713; 0.0430 (reported) | 221 | 0.0115 / 1.111 |
| T_9 | 0.00435 / 0.00425 | 0.0953 | score-05 1.000 (880,640) | 475.4 | 370; 0.0981 (reported) | 161 | 0.0098 / 1.118 |
| T_8 | 0.00362 / 0.00371 | 0.0970 | score-05 1.000 (143,104) | 1,411.6 | 287; 0.0270 (reported) | 75 | 0.0113 / 1.403 |
| T_7 | 0.00311 / 0.00347 | 0.0962 | score-05 1.000 (286,208) | 3,061.6 | 174; 0.0864 (reported) | 433 | 0.0068 / 1.016 |
| T_6 | 0.00349 / 0.00352 | 0.0965 | not banked (assumed) | 62.6 | 10; 0.0000 (gated) | 28 | 0.0000 / 0.998 |
| T_5 | 0.00334 / 0.00335 | 0.0965 | not banked (assumed) | 101.0 | 10; 0.0000 (gated) | 496 | 0.0000 / 1.017 |

G-EX's tolerance is 0.021159 at the gated points (T_5, T_6, T_15) and 0.038718 on |a_bank| ≥ 0.1 at the reported
points; G-TOPK's is 0.021159 at T_5, T_6, T_15 and 0.044273 at T_7–T_14 (M2's A1, its revision and A2).

The other gates:

| gate | where | pass side (value, source) | failing side (value, source) | result |
|---|---|---|---|---|
| G-REV | 11 points | registered = banked manifest = HF ref, 11 of 11 (M2's `a5_gate_sides.json` → `rev_cfg`) | forty zeros equals no registered revision | pass, fail |
| G-CFG | 11 points | the committed configs pass with no differing key, 11 of 11 | model_type renamed fails, 11 of 11 | pass, fail |
| K0 | T_14, first | the 10 carriers' own activations bit-identical between score-05 and score-06 (10 of 10), so B is identical (tolerance 1e-6) | M2's birth-role B of the same carriers: smallest \|ΔB\| 8.995e-04, 900 times the tolerance | pass, fail |
| I-ID | T_13 | score-04, score-05, score-06: 5,415,936 activations exact, 0 flips | M2's birth-role vectors at T_14: p99 3.381 | pass, fail |
| degenerate poles | 11 points, every layer the plan reads | `bank_scan.json`: T_8 L07N640−, T_10 L24N9826+, none elsewhere; no unit's own pole is degenerate at any of its roles | (A4: such a pole gets B = 0, ranks last, is not matched) | as M2 |
| usable units | both halves | real 82 / 71, pb 79 / 70, dt 73 / 67, every count ≥ 50 | 30 units (dry run) | pass, fail |
| movement control | freeze half | mean log10 r0 − log10 r1 1.056 [0.839, 1.285] from M2's banked ranks (82 carriers) | the identity: 0 | pass, fail |
| the twins' construction control (dt's operator) | T_15, freeze half | M2's twins under B: 79 of 79 in the top 10 | the opposite pole: 0 of 79 | pass, fail |
| G-M2TOP10 (the competitor set from the banks) | the ten raw examples' 20 (layer, point) pairs | M2's top-10 under B recomputed from M2's vectors and the histograms: names and values equal, 10 of 10 at the pre role and 10 of 10 at the birth role | the neuron axis rolled: 0 of 10 reproduce | pass, fail |
| the decomposition's identity | the ten raw examples | r_c = r_o = r0 and F = 0 with T_k replaced by T_(k−1), 10 of 10 | (dry run: a broken record reads DOES NOT COUNT) | pass |

What the bank cannot check, said plainly:
- **G-VAL, G-ABL-ID, I-GRAD**: no gradient or ablation of OLMo-3-7B exists in any bank. Both sides are shown only on
  the tiny model and on planted records (sections INSTR, JOB, FAIL, VERDICT below). The gradient rate on the L40S is not
  measured; the stop rule acts on the first point's value.
- **G-M2V at T_5 and T_6**: no second L40S run reached them; the pass side is assumed from same-device bit-identity at
  the nine other points. The failing side is banked there.
- **G-EX's pass side on the dt entries**: the dt positions were never forwarded; at the gated points the pass side
  assumes the L40S forward reproduces the L40S bank bit for bit, as it did on every M2 entry. The failing side is
  computed on every N12 entry, dt included.
- **G-TOPK** is M2's instrument on M2's entries; N12 adds the dt positions. Its failing side is the job's own.
- **The backward's memory** at batch 1 from layer 0 (about 10 GB of saved activations beside 14.6 GB of weights) is an
  estimate.

## Results by section

Each line of the log is one `CHECK`.

### REAL: the laptop banks and the A5 table (29 checks)

- `units.json` reproducible and registered; M2's ranks files and births.json registered.
- Every real unit is its birth (153); every pb unit is M2's twin (149); every dt twin re-dated k, same layer, other
  neuron, same size, same distinct count, non-echo ranks (140 of 140).
- The reader's counts reproduce: 78 of the 149 births with a twin have distinct-token share below 0.67, 16 of 149
  twins. dt twins: 70 of 140 below 0.67.
- The plan: every (unit, role) once, points in the registered order, gradient windows inside the forward windows, 10
  real carriers at T_14 for K0. Every one of the real plan's windows keeps at least 16 loss-matched candidates after the
  8-token exclusions (the smallest window keeps 1,826 of 2,047).
- The raw examples' checks (G-M2TOP10 both sides, the identity, 20 competitors with B > 0).
- One CHECK per banked gate family of the table above (12), both sides.

### SCORE: the pure functions (15 checks)

Competitors (own neuron excluded, B > 0, stable order); the layer rank with ties against the pole; the counterfactual
ranks on planted moves (only the pole rises: F > 0, r_c = r1 = 1; only the others fall: F < 0, r_o = r1 = 1; the
identity: F = 0); the pull's sign convention and A4's NaN q999; the loss match (nearest losses, exclusions, ties by
index); u with ties against the pole; Spearman with ties; G-VAL on a faithful prediction (passes, its sign flip fails),
an unrelated one (fails) and an identical one (its ceiling); the room check (every label branch at n = 71); the
bootstrap; the matched floor of a constant.

### INSTR: the gradient instrument on a tiny fp64 model (4 checks)

- `n12_score.grad_capture` against central finite differences on the 8 largest-gradient neurons of two exemplar
  positions: largest relative error 1.28e-03 (the loss is summed in fp32 inside the capture).
- The failing input, the neighbouring neuron's gradient: smallest relative error 0.256.
- Zero ablation against −h·G on 24 (neuron, position) pairs: Spearman 0.990.
- The loss-matched positions are captured beside each exemplar, the exemplar's neighbourhood excluded.

### JOB: the tiny world (16 checks)

M2's tiny world (4 layers, hidden 64, MLP width 96, the real tokenizer, 64 windows of 128 tokens, checkpoints
T_12–T_15, 40 births). M2's job writes its banks; then `n12_score.main` from its entry point in a child process, with
the loss match set to 4 positions and a 1-token exclusion (the tiny windows are dense with unit positions; the real
values are checked on the real plan in REAL). Every gate passed and every failing input failed (33). K0 ran first at
T_14 on 10 carriers, |ΔB| 0, its failing input 1.9e-03. G-M2V exact at the four points. G-ABL-ID: clean rows exact,
another window's loss off by 0.22–0.85 nats. I-GRAD: Spearman 1.000000, exact; the rolled axis −0.006. I-ID exact, 0
flips. One N12 record per unit (112); 20 competitors, the pull, 4 ablation pairs, the matched u, the identity and the
drop in each real record; every carrier's r0 and r1 equal M2's banked ranks; the pull recomputes from the banked
gradient rows and the grid q999; the raw examples carry the competitors' contexts, the exemplars and the loss-matched
texts. `metrics.json` carries no rank, hit or label.

### FAIL: each instance gate on a planted input (13 checks)

The child exits non-zero naming the gate for: G_SAME (transformers 0.0.0), INPUTS (a zero units sha256), G_CORPUS,
t14_G_REV (a wrong revision), t14_G_CFG (width changed), t14_G0 (u × 1.0 in place of 1.1: "its failing input passed"),
K0 (M2's pre-role B moved by 1e-3), t15_G_M2V (M2's T_15 vectors × 1.1), t14_G_EX (T_13's bank in T_14's place),
t14_G_ABL_ID (the clean row zeroing the whole MLP: 3.84 nats), I_GRAD (a noisy second pass: Spearman 0.696, 29 flips),
I_ID (the second forward × 1.1). A 0.02-minute budget parks T_15, T_13 and T_12 and the job ends normally.

### VERDICT: freeze and verdict (16 checks)

On synthetic records of 71 units per kind and half, one case per branch: the base case (H_cf OWN_RISE, H_away_sel_pb
ABOVE, H_signal SUPPORTED, H_bridge FORETELLS, H_dt SUPPORTED); the others falling (OTHERS_FALL, BELOW, REFUTED,
CONTRARY, H_dt UNDECIDED with H_sel_dt AT); a mixed case (MIXED, AT, UNDECIDED, UNDECIDED, UNDECIDED); dt twins hitting
more than the carriers (H_sel_dt BELOW, H_dt REFUTED); the pull above its ceiling (ABOVE_CEILING); G-VAL failing (every
gradient label DOES NOT COUNT, the realized labels stand); G-VAL above its ceiling (DOES NOT COUNT until inspected);
30 dt twins (NOT_RUN); the dt construction control missing (H_dt DOES NOT COUNT); 30 real test units (DOES NOT COUNT,
missing); no movement (DOES NOT COUNT); a broken identity (DOES NOT COUNT); a failed gate (DOES NOT COUNT, gates); the
freeze records changed (the verdict refuses). The tiny job's freeze refuses (16 real freeze units < 50), the
usable-units gate's failing input on a real job output; its freeze half summarizes under the frozen definitions
(G-VAL Spearman 0.916 on the tiny model in bf16).

### PROJECT (1 check)

1.64–2.33 GPU-h; the high case fits 0.9 of the 182-minute hard kill.

## Spec issues found while writing (resolved in `N12_SPEC.md` as stated)

1. **The competitor set needs no forward.** The brief allowed that 20 might need the forward again; M2's banked vectors
   hold every neuron at every exemplar position at T_(k−1) and T_k, and B follows from them and the histograms
   (G-M2TOP10, 10 of 10). The launch still forwards everything so that one forward serves every number; G-M2V shows it
   equals M2's arrays.
2. **F depends on the starting rank.** A twin starts at rank 1 in 40 of 79 freeze births (the carrier in 21 of 82), so
   a real − twin contrast on F would compare different rooms to rise. The twin contrast is made on the competitors'
   drop instead; every competitor starts at a layer rank of 1–21 for every unit.
3. **G-VAL's minimum has no control data.** No ablation of this model was ever banked; ρ ≥ 0.5 was registered a
   priori in the draft, with the shuffle floor as the data-driven part. **Head's decision, 2026-11-06, before freeze
   1: ρ ≥ 0.7** (`n12.VAL_RHO_MIN`); the lower bound above the shuffle's 97.5th percentile, the ceiling 0.999 and the
   sign-flip failing input are unchanged. The dry run was run again after the change (section "Rerun at freeze 1").
4. **`loss_matched` read its constants at definition time** in the first draft, so the tiny world's smaller settings
   did not reach it. The JOB section's first run stopped on the 16-candidate assertion in a 128-token window; the
   function now reads the constants at call time. The real plan's windows keep at least 1,826 candidates (REAL).
5. **The first draft compared F between real units and twins**; replaced by the drop before the final dry run (item 2).
   The dry run of the first draft also printed `N12_DRYRUN_OK` (94 checks); the banked log is the final code's.

## Rerun at freeze 1 (2026-11-06)

After the head's decision (G-VAL's bar ρ ≥ 0.7, item 3 above) the dry run was run again with the same command and
inputs (output `$SP/n12/dry_f1`): **`N12_DRYRUN_OK`, 94 checks, 0 failed**, wall time 6 min 37 s. Every CHECK line
equals the draft's run, and `dryrun_summary.json` is equal as JSON: no check reads a G-VAL value between 0.5 and
0.7 (INSTR's zero ablation gives 0.990; the planted G-VAL failure gives 0.023 and 0.101). `dryrun.log` is the
rerun's log.

## Amendment A1 (2026-11-06): the deterministic backward

score-01 stopped at I-GRAD at T_13 (Spearman 0.999642, 4 of 98 u_S changed; rule 0). `N12_SPEC.md` §14 holds the
amendment, its sources and its cost.

**The gap in this gate, named.** The dry run's backward runs on the CPU, where two passes are bit-identical. Its only
planted second pass for I-GRAD was noisy (Spearman 0.696, 29 flips), far from the 0.9996 the L40S produced. No planted
input had a nearly exact second pass, so the "no u_S change" clause was never shown to act on one, and nothing here
could show whether the GPU backward is deterministic. The JOB section's "I-GRAD exact" says only that the code
compares the right arrays.

**Planted now, and the other A1 checks** (`infra/aws_pg/jobs/n12_dryrun.py`; 6 new checks, 100 in all):

| section | check | result (`offline_gate/dryrun_a1.log`) |
|---|---|---|
| SCORE | `grad_mode` turns on deterministic algorithms and the bf16 math reduction and turns both off after; without `CUBLAS_WORKSPACE_CONFIG` it refuses (failing input) | pass; the refusal names the variable |
| SCORE | `n12.u_resolution` on planted gaps: noise 0 counts exact ties only (1 of 2), noise 0.25 counts both; `order_ranks` | pass |
| INSTR | the attention operations the gradient pass dispatches (tiny fp32 model): under `grad_mode` the math kernel's `_safe_softmax` and `_softmax_backward_data`, no fused kernel; with `grad_mode` replaced by a null context the fused `_scaled_dot_product_flash_attention_for_cpu` and its backward (failing input) | pass |
| JOB | the tiny job records `grad_pass_settings` (`:4096:8`, deterministic, MATH); I-GRAD's largest \|ΔG\| and \|ΔS\| are 0 and both changed-unit files are empty (36 units, 9,984 entries) | pass |
| FAIL | **the near-exact second pass**: one entry of the reversed pass moved by 1e6, at the carrier neuron of the first T_13 unit with u_S below 1 (`L00N3-:c0`), in the direction that lowers its pull. Spearman over the 9,984 entries 0.999995 (the 0.999 clause passes), u_S changed in 1 unit, that unit, and every changed unit reads the moved entry; largest \|ΔG\| 1e6; the job exits non-zero naming I_GRAD | pass: the zero-change clause fails a pass that the Spearman clause lets through |
| VERDICT | the verdict writes `u_S_resolution` for both halves with the noise from I-GRAD's \|ΔS\| (synthetic banks: 71 real units, 0 within noise 0) | pass |

The INSTR finite-difference check now runs under `grad_mode` (the fp64 math kernel): largest relative error 1.28e-03,
as before. Every other CHECK line equals the freeze-1 run except two values that move with the gradient pass's
attention kernel (the tiny world is bf16): the planted noisy pass's Spearman 0.6960 → 0.6957, and the tiny job's
freeze-half mean u_S 0.531 → 0.559 over its 16 real units (G-VAL ρ 0.916 → 0.917). So on the tiny model a change of
attention kernel alone, at bf16 precision, moves mean u_S by 0.03; the u_S resolution report (§14) is about this
scale of effect.

**What still cannot be checked here.** Whether the math kernel's backward is bit-deterministic on the L40S, whether any
operation raises in deterministic mode on CUDA, the time cost and the memory on the GPU, and whether the 32 MiB cuBLAS
workspace changes the forward's bits. score-02 measures each: I-GRAD at T_13, the first gradient pass at T_14,
`measured.per_grad_s` and `points.<t>.grad_max_mem_gb`, and K0 at T_14.

**Memory, measured on the CPU** (`offline_gate/a1_memory.py`, `a1_memory.json`; `N12_SPEC.md` §14 "Cost"): saved for
the backward per full-width layer 0.388 GB (default kernel), 0.959 GB (math, fp32 upcast), 0.640 GB (math, bf16
reductions); estimated peak for a window whose lowest needed layer is 0: 28.7, 46.4 and 36.5 GB against the L40S's
48.3 GB. transformers 5.9 calls sdpa with no mask and `is_causal=True`.

**Run** (same command and inputs as above, output `$SP/n12/dry_a1`, on commit `1791c3cfa` with a clean tree,
2026-11-06 11:20:59–11:28:42Z, 7 min 43 s): `N12_DRYRUN_OK`, 100 checks, 0 failed. Every CHECK line equals the run of
11:11:51Z on the same code before its commit. `dryrun_a1.log` keeps every CHECK line and drops the progress bars.

## Not verified here, said plainly

- No GPU number exists. The gradient rate, the backward's memory on the L40S, the bf16 agreement of zero ablation with
  its first-order prediction in the 7B model, and the determinism of the sdpa backward are measured by the launch.
- The tiny world checks the code path and every gate; it is not a test of the science.
- The 2609.01170 and wdd e589 numbers are quoted from `04_LITERATURE.md`'s read, not re-read.

## Files created

| file | what |
|---|---|
| `studies/next/07_PICKS/N12_SPEC.md` | the pre-registration (draft) |
| `studies/next/07_PICKS/N12_OFFLINE_GATE.md` | this record |
| `infra/pg/n12.py` | units (dt twins), plan, competitors, counterfactual ranks, pull, loss match, records, labels, freeze, verdict, projection |
| `infra/aws_pg/jobs/n12_score.py` | the job |
| `infra/aws_pg/jobs/n12_dryrun.py` | the offline gate |
| `infra/aws_pg/steps/next_n12_score-01.json` | the step list |
| `studies/next/n12/units.json`, `projection.json` | the unit table; GPU-hours per point |
| `studies/next/n12/samples/{raw_examples.py, raw_examples_zero_gpu.json, raw_examples_zero_gpu.md}` | the ten raw examples at zero GPU |
| `studies/next/n12/offline_gate/{gate_sides.py, gate_sides.json, gate_sides.log, dryrun.log, dryrun_summary.json}` | the A5 table and the dry run |
