# N1 offline gate (weight-free), 2026-11-06

**Go for the head's review; nothing launched.** The weight-free dry run printed `N1_DRYRUN_OK` after 44 checks, 0 failed
(`studies/next/n1/offline_gate/dryrun.log`, `dryrun_summary.json`). The one weight statistic the banks hold, the
down-projection column norm, was checked on both sides on the banked files (`studies/next/n1/offline_gate/gate_sides.json`):
the bank's path is bitwise reproducible across three L40S runs, and K-B's registered failing input fails for every banked
neuron with room. The job ran from its entry point on 16 tiny checkpoints written in the real checkpoints' format, every gate
passed and every failing input failed; each instance gate stopped the job on its planted failing input. The input-row
statistics and the job's own fp64 column norms have no bank; their pass side is shown only in the tiny world.

Worktree `interp_research-bio2`, branch `bio2`, from `clean-neurons` at `ae8dc637f`. CPU torch 2.12.0 (`$SP/m1_venv`).
Commands: `N1_SPEC.md` §9.3.

## What was run

| step | output | result |
|---|---|---|
| `n1 units` | `studies/next/n1/units.json` (sha256 `ada35f38…`) | 442 units (real 82/71, dt 73/67, pb 79/70); 98–100 matched neurons per birth |
| `onsets.py` | `studies/next/n1/onsets/onsets.json` (sha256 `bb7158e5…`) | k reproduced for 442 of 442; concentrated 60 of 153 real, 51 of 140 dt, 64 of 149 pb; earlier crossings 20, 12, 24 |
| `gate_sides.py` | `gate_sides.json`; `studies/next/n1/kb_bank.json` (775 values, sha256 `401c6134…`); `studies/next/n1/inputs/n6_colnorms_t15.npy` (the LFS object `dff570e3…` of N6 L1-02) | below |
| `raw_examples.py` | `studies/next/n1/samples/raw_examples_zero_gpu.{md,json}` | the ten freeze example births and their twins |
| `n1 project` | `studies/next/n1/projection.json` | 0.82–1.71 instance-h on r7i.2xlarge; hard kill 134 min; 0 GPU-h |
| `n1_dryrun` (40 s) | `dryrun.log` | `N1_DRYRUN_OK`, 44 of 44 |

## Both sides of every gate, on the banks

| gate | failing side | pass side | where |
|---|---|---|---|
| K-B | the neighbouring point's banked value of the same neuron: smallest relative difference 0.00155 over 419 adjacent pairs (median 0.14), against the tolerance 1.3e-4; 0 within it | not banked for the CPU fp64 path; the bank's own path: M2 against N12 458 of 458 bitwise, M2 against N6 at T_15 184 of 184, N12 against N6 122 of 122; tiny world: 9 of 9 points, largest 1.9e-7 | `gate_sides.json` → `K_B`, `control_pairs_bank_path`; dry run JOB |
| G-SAME, INPUTS, G-REV, G-SHAPE, I-ID | planted in the tiny world: each stops the job at the named gate | tiny world: pass | dry run FAIL, JOB |
| PC | planted size 0: shares 0.03–0.09 below 0.9 at every bracket | planted 99th-percentile size: 1.0 at every bracket | dry run JOB (no bank holds weights) |
| room | — | every selection label and hypothesis reachable at 67 pairs | `gate_sides.json` → `room` |
| usable, ceiling, freeze unchanged | 30 dt pairs; a planted excess of 4.0; one byte appended | the synthetic cases | dry run VERDICT |

## What the dry run shows beyond the banks

- JOB: the statistics in `stats.npz` equal an independent numpy fp64 recount from the tiny safetensors (largest difference
  0.0); the neuron given a planted input-row change in bracket 10 reads e 13.78 on the input side and −0.41 on the output side,
  every other unit at most 1.07; 58 failing inputs fail; `metrics.json` carries no excess or label.
- FAIL: a version string, a zero units sha256, a zero revision at T_2, a transposed down tensor at T_3, the bank moved by 0.1 %
  (K-B at T_7), a second read of T_15 changed by 1 % in one tensor (I-ID): each stops the job at the named gate. A change of
  0.01 % in that plant rounded back to the same bf16 values and did not stop it; the plant was set to 1 %.
- VERDICT on synthetic records: ROWS_FIRST, COLUMN_FIRST, TOGETHER, NEITHER (H_0), NEITHER (a BELOW side), the floor label,
  the ceiling, the missing-twin case, and both refusals.

## Found and fixed during this gate

- G-TRI (our relative column change at least the banked relative norm change) was first written as a gate. Its failing input
  (other neurons' norm changes) passed in the tiny world: a vector's change is far larger than its norm's, so the triangle
  inequality holds for the wrong neuron too. It has no power, so it is reported (`tNN_TRI`), not gated.

## Gaps, said plainly

1. No bank holds an input-row statistic of any neuron, so H_rows' quantity has no offline look at all.
2. K-B compares an fp32 GPU reduction with a CPU fp64 one; its tolerance is the fp32 bound for any summation order, not a
   control pair on these two paths (none exists). The head may prefer it reported.
3. The CPU instance's rate (read and compute 60–180 s per checkpoint) is estimated, and its price is not banked.
4. Whether the AMI's torch imports on an r7i instance is assumed from earlier CPU jobs on the same AMI, not checked here.

## Freeze 1 (the head's decision of 2026-11-06 19:25Z)

K-B is reported, not gated: no control pair exists for an L40S fp32 reduction against a CPU fp64 one. The job now writes
K-B to `metrics.json` → `reports` → `tNN_K_B` (`R.report` in place of `R.gate` in `infra/aws_pg/jobs/n1_weights.py`).
Two dry-run checks changed with it: the JOB check reads K-B from `reports` and also asserts that no gate is named K_B
and that every K-B failing input failed; the FAIL case `kb_off` (the bank moved by 0.1 %) now asserts that the job
completes and that K-B reports a failure at every banked point.

Dry run again on the changed code, 2026-11-06T19:22:29Z–19:23:12Z (`$SP/bio2/n1_dry_f1`): `N1_DRYRUN_OK`, 44 checks,
0 failed (`studies/next/n1/offline_gate/dryrun.log`, `dryrun_summary.json`). Against the draft's run, three CHECK lines
differ and every other line is equal:

| check | draft | freeze 1 |
|---|---|---|
| JOB every registered failing input failed | 58 failing inputs (gates) | 49 (the 9 K-B failing inputs moved to `reports`) |
| JOB K-B | gated; t07 7.7e-08, t08 9.5e-08, t09 4.3e-08 | reported; the same values; every K-B failing input failed |
| FAIL kb_off | the job stops at `t07_K_B` | the job completes (exit 0); K-B reports a failure at 9 of 9 banked points, smallest max_rel 1.0e-03 |

The onset finding (93 of 153 births hold less of the rise in the dated bracket than in the two before it; median ratio
0.84 of the threshold one checkpoint before birth) is accepted as the basis of the registered readout and is stated in
`N1_SPEC.md` §8 row 10. The intermediate-checkpoint onset location is not registered.

## A1 (2026-11-07): the raw-examples writer

weights-01 stopped at its last step with `KeyError: 'examples'`: the job read the example births from `units.json`, which
carries only `units` and `n`. The dry run had planted a `units.json` with an `examples` key, so its JOB section never
met the real file's shape. Spec §14 A1 holds the record.

| run | code | result |
|---|---|---|
| the plant on the frozen job (ended 2026-11-07T15:19:47Z, `$SP/bio2/n1_dry_a1_plant`) | `n1-freeze-1` job; dry run with the tiny `units.json` reduced to the real keys (`n`, `units`) | JOB stops: `KeyError: 'examples'`, as weights-01 did (`studies/next/n1/offline_gate/dryrun_a1_plant_on_freeze1_job.log`) |
| the fix (2026-11-07T15:20:15Z–15:20:55Z, `$SP/bio2/n1_dry_a1`) | the writer reads `reg["examples"]`; `registry()` carries `n1.EXAMPLES` (N12's ten example births per half) | `N1_DRYRUN_OK`, 46 checks, 0 failed (`dryrun_a1.log`, `dryrun_a1_summary.json`) |

Two checks are new: `REAL A1` (the job's registered example births equal `studies/next/n12/units.json` → `examples`, 10
and 10, and each is a real unit of its half in N1's `units.json`) and `JOB A1` (the tiny `units.json` has the real file's
top-level keys). The other 44 CHECK lines are byte-equal to the freeze-1 run's (`diff` of the CHECK lines of
`dryrun.log` and `dryrun_a1.log`).
