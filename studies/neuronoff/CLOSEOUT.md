# CLOSEOUT — NEURONOFF

Status: **CLOSED**, 2026-10-25. Freeze tag `neuronoff-freeze-1` = e5a5c33c (SPEC unchanged since). Final commit:
a1f70bb1 on branch `neurons/off` (closeout content; this line was added in the next commit).

The frozen rules gave no Arm A answer. The SPEC §6 positive control failed on 65 of 156 Q4B poles and 75 of 202 G4
poles (shares 0.4167 and 0.3713, above 0.30), which cancels both models' Arm A labels. Arm B: 0 of 7 poles SPECIAL.
Every number below resolves to the artifact named beside it. Paths are relative to `studies/neuronoff/`.

## 1. Artifacts

| artifact | path |
|---|---|
| spec, schema, plans | `SPEC.md`, `SCHEMA.md`, `plan/` (frozen at e5a5c33c) |
| journal (predictions, launches, gates, deviations) | `JOURNAL.md` |
| run handoff (gate tables, cost, deviations) | `handoff/2026-10-25_run.md` |
| result document | `results/RESULTS.md` |
| frozen thresholds | `results/thresholds_Q4B.json`, `results/thresholds_G4.json` |
| verdicts (frozen numbers, cancellation stamps propagated) | `results/verdict_Q4B_v2.json`, `results/verdict_G4_v2.json` |
| verdicts as first written by the frozen code | `results/verdict_Q4B.json`, `results/verdict_G4.json` |
| construct counts | `results/armb_counts_Q4B_v2.json`, `results/armb_counts_G4_v2.json` |
| grid | `results/grid_v2.md` (first write: `results/grid.md`) |
| exploratory E1 | `results/q4b/1/exploratory_E1_v2.json`, `results/g4/1/exploratory_E1_v2.json` |
| exploratory E2 | `results/q4b/1/exploratory_E2.json`, `results/g4/1/exploratory_E2.json` |
| committed cell copies | `results/{pre_q4b,q4b,g4}/1/` (`metrics.json`, `controls.json`, `manifest.json`, Arm B teacher and samples) |
| full banks (untracked) | `bank/off/{pre_q4b,q4b,g4}/1/`; S3 `s3://<s3-bucket>/results/neurons/off/<cell>/1/` |
| code | `infra/aws_pg/jobs/neuron_off.py`, `infra/pg/neuron_off_core.py`, `infra/pg/neuron_off_analysis.py`, `infra/pg/neuron_off_exploratory.py`, `tests/test_neuron_off_core.py` |
| infrastructure log | `../../infra/aws_pg/operations/2026-10-25_neuronoff_orphan_reapers.md` |

## 2. Claims record

Tags: FROZEN-CANCELLED (decided by a frozen rule, which cancels it), FROZEN (decided by a frozen rule),
EXPLORATORY (rule chosen after unblinding; never the headline), REPRODUCED-BY-REVIEWER (a different model
recomputed it from the raw banks, result review 2026-10-25).

| id | claim | artifact → key | status |
|---|---|---|---|
| N1 | Every model gate passed in Q4B and G4, and every registered failing input failed | `results/{q4b,g4}/1/controls.json`; `results/{q4b,g4}/1/metrics.json → gates` | FROZEN; REPRODUCED-BY-REVIEWER |
| N2 | G-pc failed on 65 of 156 Q4B and 75 of 202 G4 Arm A poles (0.4167, 0.3713 > 0.30) | `results/verdict_<M>_v2.json → pc_cancel_share` | FROZEN; REPRODUCED-BY-REVIEWER |
| N3 | Q4B Arm A own-context, collateral and energy labels: cancelled ("instrument not shown to fire"; G-n 26 / 4) | `results/verdict_Q4B_v2.json → headline_own, label_coll, label_energy` | FROZEN-CANCELLED; REPRODUCED-BY-REVIEWER |
| N4 | G4 Arm A own-context, collateral and energy labels: cancelled ("instrument not shown to fire"; G-n 40 / 9) | `results/verdict_G4_v2.json → headline_own, label_coll, label_energy` | FROZEN-CANCELLED; REPRODUCED-BY-REVIEWER |
| N5 | Every Arm A secondary row in both models is cancelled, whatever label the first-written verdict printed | `results/verdict_<M>_v2.json → secondary` | FROZEN-CANCELLED |
| N6 | Under the cancelled label: Q4B M −0.000101 [−0.004477, 0.010809] (δ 0.008075); G4 M −0.0000348 [−0.008433, 0.005257] (δ 0.013030) | `results/verdict_<M>_v2.json → stats.own`; `results/thresholds_<M>.json → arm_a.delta_own` | FROZEN-CANCELLED; REPRODUCED-BY-REVIEWER |
| N7 | Study label own-context and collateral: MIXED (no model counts) | `results/grid_v2.md` | FROZEN-CANCELLED |
| N8 | Arm B: 0 of 7 SPECIAL; 6 NOT-SPECIAL, 1 UNRESOLVED (L08N3398-) under the frozen rule | `results/verdict_<M>_v2.json → arm_b.*.verdict` | FROZEN; REPRODUCED-BY-REVIEWER |
| N9 | The Arm B intervals of L11N5238+ and L27N2924- (frozen NOT-SPECIAL) and L08N3398- (frozen UNRESOLVED) cross m_B*; `_v2` labels them "interval crosses margin" (D1, after review) | `results/verdict_<M>_v2.json → arm_b.<pole>.readouts.R1.D.hi, m_B_star, interval_vs_margin` | REPRODUCED-BY-REVIEWER |
| N10 | TeX L04N5165+ is not tested where it fires (≤ 0.0525 of banked max on every scored position); `_v2` reads "instrument not shown to fire" (D1, after review; frozen verdict NOT-SPECIAL) | `results/verdict_Q4B_v2.json → arm_b.L04N5165+.pole_activation_max_frac`; `results/q4b/1/arm_b/L04N5165p_teacher.jsonl → pole_a_inputs` | REPRODUCED-BY-REVIEWER |
| N11 | React useEffect (G4 L07N10202-): Δ_pole 0.00984, Δ_ctrl1 0.00575, Δ_layer 0.816, D 0.00408 [−0.0159, 0.0244], m_B* 0.0380, NOT-SPECIAL; construct in 321 / 321 / 320 of 400 | `results/verdict_G4_v2.json → arm_b.L07N10202-`; `results/armb_counts_G4_v2.json` | FROZEN; REPRODUCED-BY-REVIEWER |
| N12 | Behaviour changed (§5.4): 0 of 7 | `results/armb_counts_<M>_v2.json → poles.*.behaviour_changed` | FROZEN; REPRODUCED-BY-REVIEWER |
| N13 | One pole's ablation moves own-context NLL by a median 0.0026–0.0093 nats per side; per-context median ≤ 0.000037 | `bank/off/<cell>/1/out/poles/*.json → own.contexts[].dnll`; `results/RESULTS.md` §3 | REPRODUCED-BY-REVIEWER |
| N14 | E1 (pooled positive control, re-gated nulls): H_morcos in Q4B (M 0.00175 [−0.00202, 0.00553], δ 0.00747) and G4 (M 0.000234 [−0.00348, 0.00335], δ 0.01109); δ exceeds the side medians | `results/<cell>/1/exploratory_E1_v2.json → regated_nulls.result` | EXPLORATORY; REPRODUCED-BY-REVIEWER |
| N15 | E1 with the frozen nulls: cancelled by G-n | `results/<cell>/1/exploratory_E1_v2.json → result.label_own` | EXPLORATORY |
| N16 | E2 relative difference D_k / mean(\|own_one\|, \|own_ctrl\|): Q4B +0.255 [−0.489, +0.676] (70 pairs), G4 +0.083 [−0.490, +0.654] (89 pairs); both intervals cover 0 | `results/<cell>/1/exploratory_E2.json → relative_D` | EXPLORATORY |
| N17 | Pre-registered predictions missed: Arm B SPECIAL (≥ 3 of 7 → 0), G-pc pass (≥ 90% → 58.3%, 62.9%), M > 0 (→ negative in both) | `JOURNAL.md` PREDICTIONS; `results/RESULTS.md` §5 | FROZEN |
| N18 | Compute: 2.794 instance-hours, USD 6.44 | `handoff/2026-10-25_run.md` §6 | FROZEN |

## 3. Deviations

1. U+2028 parser fix (post-freeze, code only). The first unblind raised at `neuron_off_analysis.py:727`: the parent
   verdict file holds U+2028, U+2029 and U+0085 inside JSON strings, and `str.splitlines()` split there. Fix
   7f6f244d (split on "\n" only), merged at d6f234a9. No rule changed and no frozen output moved (`JOURNAL.md`).
2. E1: pooled per-model positive control in place of per-pole G-pc. Chosen after unblinding and never the headline.
   Code ee5a5fa0, merged at 35507ae4 (`JOURNAL.md` "E1 exploratory re-read").
3. E2: relative effect per pair and median absolute own-context ΔNLL per side. Chosen after unblinding and never the
   headline. Code and outputs 4a5de55a, merged at cb184543.
4. Stamp fix (code-only deviation D1, after review): the frozen code printed §7.4 secondary-row labels, E1
   frozen-nulls secondary labels and the grid's study row without the model-level cancellation
   (`neuron_off_analysis.py:432`, `subset_row`). The `_v2` copies carry the stamps; the study row prints "0 counted
   models". D1 also states where each Arm B interval lies against m_B* ("interval crosses margin" for L11N5238+,
   L27N2924-, L08N3398-) and prints "instrument not shown to fire" for a pole whose clean activation never exceeds
   0.1 of its banked maximum (TeX L04N5165+, 0.0525); that 0.1 line was set after review. Every shared numeric field
   of the `_v2` files equals the original (re-checked: 2,208 shared numeric fields, 0 differ; `_v2` adds 7
   `pole_activation_max_frac` fields). Code and outputs 4a5de55a, merged at cb184543; originals untouched.
5. O7 dropped by the frozen L4 budget rule. R = `progress.tokens_per_second` = 2,291 tokens/s, dominated by Arm B
   sampling; Arm A alone ran at 13,355 tokens/s. The projection was 24.41 h against the 11 h line.
6. G4 ran 90 pairs, not 70. The literal L4 index of pair 70's null partner (a control-control pair) is 202. Q4B
   ran 70 pairs (`--max-poles 156`).

Smaller runbook deviations (preflight on g6e.4xlarge, budgets from truncated token counts, `--plan` on unblind,
the held `armb_counts`) are listed in `handoff/2026-10-25_run.md` §7.

## 4. What a follow-up would need

1. A pooled positive control, frozen in advance and checked for room to pass as well as to fail. The per-pole
   0.05-nat floor sits above the per-context median whole-layer effect (0.0139 Q4B, 0.0207 G4).
2. At least 30 held-out contexts per pole in the positive control. This run used each pole's first 10.
3. O7 and the dropped pairs (Q4B pairs 71–96, G4 pairs 91–97) as a pre-registered second attempt, with the budget
   rule measured on Arm A throughput only.
4. Steering (X2) to test sufficiency. Necessity, measured here, is about 0.01 nats per pole on its own contexts
   (`results/RESULTS.md` §3).
