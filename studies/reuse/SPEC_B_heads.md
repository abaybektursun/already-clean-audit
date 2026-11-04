# SPEC B: does a neuron's new job run through the old job's attention heads? (draft)

Status: **draft, 2026-11-01 20:30 CST, not frozen.** Freeze is the commit after the stage's weight-free dry run passes and the verdict script fails on each failing input, on the user's go; no forward before it. Parents: `SPEC.md` (§1's "yes and specific" branch, read with R1b of `SPEC2.md`), `results/REVIEW_amend2.md`, `NEXT.md` step 1. Written by the coordinator (Fable 5.1), who has seen every REUSE number; every threshold below is set from the MLP-driver references already banked or from constructed controls, not from head data, which does not exist yet.

## 1. Question and decision

REUSE found that at a job's birth checkpoint the new job's activation runs in part through the old job's top-50 MLP drivers (I 0.226 against a same-checkpoint null of 0.031 and a same-job ceiling of 0.653). Attention was not measured; the MLP drivers were chosen by the direct path, and holding an MLP neuron removes every path through it. This spec asks the same question of attention heads: at the birth checkpoint, does P's activation on J depend on the heads that drive P on O*?

| outcome | next |
|---|---|
| I_head not refuted and specific, and I_head within 0.15 of the MLP I on the same jobs | both input channels are reused alike; the program's next question is the output side (Spec A) |
| I_head not refuted and specific, and I_head above the MLP I by 0.15 or more | heads carry the reuse; a joint head-and-neuron ablation spec follows |
| I_head refuted or not specific | reuse is an MLP-input fact; Spec A follows |

## 2. Hypotheses, each with the observation that refutes it

| hypothesis | refuted when |
|---|---|
| H_head: removing O*'s top-h heads lowers P's activation on J by a large fraction of what removing J's own heads does | the interval of median I_head at birth overlaps the interval of the head null at birth (R1h) |
| H_generic-head: the fall comes from heads every neuron depends on | median Δ(H(O*); J) − Δ(H_foreign; J) has an interval containing 0, or so does the activity-matched contrast (R2h); then "not specific" whatever R1h says |
| H_same-as-MLP: heads and MLP drivers carry the same share | the interval of the paired median I_head − I_MLP over live jobs excludes 0 (R5h); a positive value means heads carry more |
| instrument artifact | any gate of §5 fails |

## 3. Measurement

Model, checkpoints, cases, windows, readout a_P and the job-fall rule as SPEC §2; the same 150 primary cases, the same birth checkpoints and T_15. Every head of every layer below L is a candidate; OLMo 3 7B has 32 layers; head counts and grouped key-value layout are read from the config at load, never typed.

**Head attribution.** For context x, the direct contribution of head (l, h) to P's pre-activation at the marked token is the head's output vector at that position (the slice of the `o_proj` input belonging to h, times its slice of `W_o`) read through P's input rows, scaled by the inverse RMS and gain of layer l's own post-attention norm (OLMo 3 applies its norms after the attention and MLP blocks, and the MLP reads the residual directly; amended at freeze from "the layer-L RMSNorm scale"), as `unite_route.direct_A` reads an MLP neuron's contribution through its layer's post-feedforward norm; the worker extends the capture with a per-head hook on the `o_proj` input. `H(J)` = the h heads with the largest `Σ_{x∈J} A_{(l,h)}(x)`. **h is frozen on the sniper jobs:** the smallest of 10, 20, 40 whose median own fall is at least 0.25 and whose room (§5) is at least 0.20; stated before any primary forward.

**Ablation.** A head's output at every position of the window is replaced by its background mean, the mean over UNITE's e1 centring positions at that checkpoint (as the MLP replacement); identity, own `H(J)`, cross `H(O*)`, random-matched `R_H` (per head in H(O*), the head of the same layer, outside both sets, nearest in background mean output norm), foreign `H_foreign` (the heads of the foreign pole Q of REUSE, chosen the same way), and activity-matched `R_HJ` (nearest in mean output norm on J's contexts). Statistic `I_head(J) = Δ(H(O*); J) / Δ(H(J); J)`; cancel rule as SPEC §2.

**References, both at the birth checkpoints from the start.** Null: the 56 planted two-neuron sums of S3, each side's heads by the same rule, full sides capped at 12, labels and shuffled labels. Ceiling: the 138 own ceiling splits, side 2 read with own `H(side 2)` and cross `H(side 1)`. Weights and bootstraps as SPEC §6 and SPEC2 B1.

## 4. Rules (bootstrap over neurons, 2,000 draws, seed 0, 95% percentile)

- **R1h.** Median I_head at birth over live jobs against the head null at birth: overlap → refuted; real higher, no overlap → not refuted; real lower → its own outcome.
- **R2h.** Median Δ(H(O*); J) − Δ(H_foreign; J) and median Δ(H(O*); J) − Δ(R_HJ; J): both intervals above 0 → specific; otherwise not specific, and the post says which control it failed.
- **R5h.** Paired median I_head − I_MLP over the jobs live under both instruments, I_MLP from `results/primary/verdict.json`: interval above 0 → heads carry more; below 0 → less; contains 0 → alike.
- Reported, no rule: I_head at T_15 and its rise against the head ceiling's rise (as R3b); I_head by the clusterer's label (descriptive, the label named as the clusterer's, no reader labels); the head sets' overlap `s_head`; the share of H(J) in the last four layers below L.

## 5. Gates (each with value, failing input and identity value)

Known number and completeness as SPEC §5 at every load. Identity: |Δ| ≤ 1e-3 on every context; failing input own heads. Positive control: median own fall at the frozen h ≥ 0.25; failing input random-matched heads. Room: median head ceiling at birth minus the head null's 95th percentile at birth ≥ 0.20; failing input shuffled planted labels. Cancel share under 40%. Head capture check: the sum over heads of the per-head direct contributions equals the layer's attention-block contribution already captured by `unite_route` within 1e-2 relative (amended at freeze from 1e-3: the forward is bf16 and the captured block carries bf16 rounding, which puts the fp32 per-head sum at 1e-3 to 3e-3 on bf16 tiny models; 1e-2 is REUSE's completeness line, and the failing input sits at 0.45 or more), on every context of every job at every load; failing input, the heads of the wrong layer.

## 6. Point predictions, no rule attached

Own head fall at frozen h 0.35; head null at birth 0.05; head ceiling at birth 0.55; median I_head at birth 0.15 [0.08, 0.22], so R1h not refuted; R2h specific against foreign, not against activity-matched (interval containing 0); R5h: I_head below I_MLP by 0.08 [0.02, 0.14]; H(J) concentrated in the four layers below L for over half the jobs.

## 7. Budget and runbook

Sniper: 6 jobs, three h values, two checkpoints, about 0.3 GPU-h. Primary: 150 jobs × 6 arms × 12 contexts × 2 checkpoints, 138 ceilings, 56 planted pairs × 4 arms at 8 birth loads, about 60,000 forwards, 9 loads: about 1.0 GPU-h; cap 1.6. Total about 1.3 of the program's remaining 3.27 GPU-h; $0 API. Steps as SPEC §13: Opus worker writes the head capture, the stage and `verdict_heads.py` with a tiny-model dry run and failing inputs; coordinator reviews, freezes (tag `reuse-heads-freeze-1`); sniper first, h frozen in the journal before the primary launch; samples before any number; one unblinding; different-model read; results and post addendum.

## 8. Limits known now

Mean replacement of a head's output is an off-distribution intervention, bounded by the identity gate and the matched controls, not removed. Grouped key-value attention shares keys across heads; the ablation is of the head's output, so the sharing does not enter. The head null is read on full sides with T_15's planted construction (SPEC2 F7(d), (e)). The clusterer's labels are descriptive only (SPEC3 Phase 0). One model, one run.
