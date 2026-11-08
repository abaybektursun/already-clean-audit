# L1-02 record (completed 2026-11-06T16:18:57Z; two gates that compare two paths failed)

**L1-02 ran to the end (exit 0, 13 of 13 edits, nothing parked) and banked the column norms and energies; G-ZN and G-CACHE
failed at their bitwise bar by fp32 rounding (zero edit ΔL −3.0e-11 nats per token; largest per-token loss difference 9.5e-7
nats).** Every other gate passed with its failing input failing. Source: `banks/L1-02/metrics.json` → `gates`,
`banks/L1-02/edits.jsonl`. This folder holds `job.log` (copied once after the end), `STATUS`, `heartbeat.jsonl`, `dmon.log`,
`pipeline_status.json`, `steps.json` and `g_pole_join.json`.

## G-ZN and G-CACHE

| gate | value | bar | failing input |
|---|---|---|---|
| G-ZN | zero edit (P002_late_zero, L03N7574−): ΔL −3.015e-11 nats per token, not bitwise equal to base | bitwise equal | first col edit differs: fails, as required |
| G-CACHE | base losses recomputed from the progression's final residual against base pass 1: largest per-token difference 9.537e-7 nats | bitwise equal | — (dry run) |
| G-ID (same path twice) | base pass 1 against base pass 2: 0.0 | bitwise equal | passes; its failing input fails |

Likely cause (a hypothesis; the job does not record which piece differs): base pass 1 and pass 2 run the model's own forward,
which applies the final norm and the head to a batch of 4 windows ([4, 2,049, 4,096] into the head). The progression and every
edit apply the final norm and the head to one window at a time (`Model.window_logits`, [1, 2,049, 4,096]). On the GPU the two
matrix shapes can select different fp32 GEMM kernels, which round differently. G-ID, which compares the same path twice, is exact.
On the CPU (the dry run) the two paths were bitwise equal, so the gap did not show offline. It would be refuted by a forward that
applies the head per window to the base pass too and still differs. Could it have been checked from banked data? No: no fp32
forward of O7 on the GPU was banked before L1-02 (`N6_OFFLINE_GATE.md` §4, §6 names G-ZN and G-CACHE as two-path gates).

Size of the offset: the zero edit's ΔL (−3.0e-11) is 400 times below the fp32 rounding floor F_eps 1.22e-8 and 22,000 times
below the smallest \|ΔL_col\| of the 10 column edits (6.74e-7). Every edit's ΔL carries this offset because it is read on the
progression's path against base pass 1.

What it means under the registered rules: §4 makes every label DOES NOT COUNT when a gate the verdict reads fails. L2 runs the
same code, so G-ZN and G-CACHE would fail again. The choice is the head's before freeze 1b (none applied): a tolerance frozen
from this control pair (the rule of `N6_OFFLINE_GATE.md` §6: L1-02 is a measurement on the two paths), the base pass computed on
the progression's path (head per window) so the two sides share one path, or report rather than gate.

## G-POLE, joined with the harvest's g and u

`g_pole_join.json` (`studies/next/n6/diag/g_pole_join.py` on `banks/L1-02/g_pole.npz` and the T_15 `contexts.jsonl`): 41 of 197
poles have a recomputed g or u that differs from the harvest's bf16 value, by 1 or 2 bf16 steps (36 exemplar g values, 41 u
values). They are exactly the 41 poles whose G-POLE value is above 1e-6; none of the 156 others has a g or u difference. The
worst pole, L08N3846+ (0.01506), has 2 g values and 1 u value one step away. The per-pole values equal L1-01's bit for bit on a
second instance. The join script was checked on a bank-equal input (0 differences) and on a planted one-step change (1 step found).

## A preview of G-RES (freeze 2), from the ten raw column edits

G-RES needs F_eps ≤ min(1e-6, 0.01 × the 10th percentile of \|ΔL_col\| over the freeze half's pair edits). On the ten L1 column
edits (five stage-1 freeze pairs; four of the five pairs are freeze pairs of `pairs_L2.json`, none is a test pair): the 10th
percentile is 1.075e-6, so the bound would be 1.075e-8, below F_eps 1.221e-8. G-RES would fail on this preview by a factor 1.14.
The registered G-RES reads the stage-2 freeze half's 56 column edits at freeze 2; this is a preview, not the gate.
