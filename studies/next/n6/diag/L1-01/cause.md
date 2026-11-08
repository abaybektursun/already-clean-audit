# L1-01 stopped at G-POLE (bf16 phase), 2026-11-06T15:29:08Z

**The job stopped at G-POLE: the worst pole's recomputed exemplar activations differ from the T_15 bank by 0.01506 of its
largest banked activation (L08N3846+), above the tolerance 0.01.** Six of 197 poles are above 0.01. The registered failing
input (neuron n + 1 at the same positions) failed as required: smallest value 0.923. Source: `metrics.json` →
`gates.G_POLE` (this folder) and `job.log` (the AssertionError at the end).

| quantity | value | rule |
|---|---|---|
| G-POLE value: max over 197 poles of max over the first 10 banked exemplars of \|a_new − a_bank\| / max \|a_bank\| | 0.015059 (L08N3846+) | ≤ 0.01: **fail** |
| failing input (neuron n + 1, same forward): smallest value over 197 poles | 0.9234 (median 1.0015) | must exceed 0.01: fails, as registered |
| K1 (E1's zero value) | 0.6742177605628967 against 0.6742178218025651 (difference 6.1e-8; a_before and a_after equal to E1's banked values, max difference 0.0) | pass; failing input fails |
| K2 (bf16 NLL, first 120,000 targets) | 2.675372037760417, equal to the banked value to the last digit | pass; failing input fails |
| G-SAME, G-REV | torch 2.12.0+cu130, transformers 5.9.0, NVIDIA L40S; snapshot `373bad25…` | pass; failing inputs fail |

The per-pole values fall in two groups (`per_pole` in `metrics.json`):

| per-pole value | poles |
|---|---|
| exactly 0 | 111 |
| (0, 1e-6] | 45 |
| (1e-6, 1e-4] | 0 |
| (1e-4, 1e-3] | 1 |
| (1e-3, 5e-3] | 15 |
| (5e-3, 1e-2] | 19 |
| above 1e-2 | 6: L08N3846+ 0.01506 (P059 late), L31N3149− 0.01197 (not in a pair), L18N2982+ 0.01152 (P049 early), L25N10599− 0.01100 (P031 late), L26N10050− 0.01029 (P015 early), L15N6531− 0.01026 (P039 late); the five paired ones are all in the freeze half |

By layer: layers 0–7 66 poles, 8 above 1e-3, 0 above 1e-2; layers 8–15 30, 11, 2; layers 16–23 35, 6, 1; layers 24–31 66, 15, 3.

What it shows. For 156 of 197 poles the recomputed g and u equal the bank's, and a differs only at fp32 arithmetic level
(≤ 1e-6). For 41 poles the recomputed activation differs at bf16 level: bf16 keeps 8 significand bits, so one rounding
step is 0.39–0.78 % of a value, and the 41 values (1e-4 to 0.015 of the pole's largest activation) are a fraction of one
step to about four steps. The forward in the harvest's shapes reproduces the bank exactly: K2 runs full 2,049-token
windows in batches of 8, as the harvest did, and equals the banked NLL to the last digit. G-POLE does not run the harvest's
shapes: `reuse_ablate.readout` (via `unite_route.pack`) forwards right-padded prefixes that end at each exemplar, sorted by
position in the window and batched 8 at a time, with the forward cut after the pole's layer. The likely cause is that
these different matrix shapes select different GPU kernels, and those kernels round bf16 differently. This is a hypothesis.
The job did not record g, u or a per exemplar, or which exemplar set the maximum. It would be refuted by a recompute in the
harvest's shapes (full windows, batches of 8) that still differs at bf16 level.

Could the cause have been checked from banked data? **No.** The offline gate's pass side for G-POLE compared two records
of the same harvest (`arrays.npz` against `contexts.jsonl`, 0.0 by construction), and `N6_OFFLINE_GATE.md` §4 lists the
pass side of G-POLE on this job's forward as not checkable from the banks. The closest banked forwards on the same
prefix path, E1's and E4's known numbers at T_15 (`studies/instruments/results/e{1,4}/1/t15/samples/known_number.json`),
report a median ratio (0.0 for each of the 3 poles) and two sample values per pole. A median hides a few entries that
differ, so they could not show this max-statistic difference. M2's and N12's T_15 G-EX was exact, but it ran full windows
in harvest batches (`m2_score.forward_capture`), not prefixes. The tolerance 1e-2 was set as if the recompute were
bit-identical (spec §3.1, §4). It was checked against the failing side only (the T_14 values, smallest 0.150).

What the gate guards and what the failure is not. G-POLE checks that each edited pole's identity on the input side matches
the bank. The failing input (the neighbouring neuron) sits at 0.92, about 61 times the worst pass-side value, so the
gate's discrimination is intact. Nothing after G-POLE ran: no column norm, energy, fp32 number or edit exists for this
attempt (`edits.jsonl` is empty; `colnorms.npy` and `energies.npy` were not written).

The run DOES NOT COUNT. No relaunch without the head. These choices are the head's, and none is applied: (a) a dated
amendment that sets the tolerance from this run's pass side, for example 0.05 (3.3 times the worst value seen, 3.0 times
below the T_14 failing input's smallest value 0.150, 18 times below the neighbour's 0.92); (b) recompute G-POLE in the
harvest's shapes (full windows, batches of 8), where K2 shows bit-identity; (c) use M2's A1 statistic
(`m2.values_gate`, p99 of \|new − bank\| / (0.05 + \|bank\|)) in place of the max. Any of these changes code or a
registered threshold after freeze 1, so the tag would need a new freeze (`n6-freeze-1a`) and the next attempt the prefix
`L1-02`.
