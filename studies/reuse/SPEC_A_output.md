# SPEC A: does a neuron's new job write to the old job's readers? (draft)

Status: **draft, 2026-11-01 20:30 CST, not frozen.** Freeze is the commit after the stage's weight-free dry run passes and the verdict script fails on each failing input, on the user's go; no forward before it. Parents: `SPEC.md` (§1's "no" branch, the output-side question), `results/REVIEW_amend2.md`, `NEXT.md` step 2. Written by the coordinator (Fable 5.1); thresholds set from constructed references, not from reader data, which does not exist yet.

## 1. Question and decision

REUSE measured the input side: whether a new job's activation runs through the old job's upstream drivers (it does, in part). This spec measures the output side: when P fires on J, does its effect land on the same downstream neurons that it drives when it fires on O*? A pole whose two jobs share inputs but not readers would be two functions sharing a detector; one whose jobs share readers but not inputs would be one function with two triggers.

| outcome | next |
|---|---|
| I_out not refuted and specific at birth | the pole's jobs share an output function; the growth story is about readers, and the next spec reads reader reuse over training against the input reuse already measured |
| I_out refuted or not specific | readers are job-specific even where inputs are shared; "new job" means a new output function on a partly shared detector |

## 2. Hypotheses, each with the observation that refutes it

| hypothesis | refuted when |
|---|---|
| H_out: holding P changes the old job's top readers on J's contexts by a large fraction of what it changes J's own readers | the interval of median I_out at birth overlaps the interval of the two-neuron null at birth (R1o) |
| H_generic-out: the effect lands on readers every pole writes to | median of the O*-reader effect minus the foreign pole's readers' effect has an interval containing 0 (R2o): not specific |
| H_in-out: input reuse and output reuse go together | Spearman of I_out and I_MLP over live jobs has a 95% interval containing 0 (R6o, reported with a rule only on sign) |
| instrument artifact | any gate of §5 fails |

## 3. Measurement

Model, checkpoints, cases, windows as SPEC §2; the forward now runs past L to the last layer. **Readers.** For context x, the direct contribution of P to downstream MLP neuron k (layer > L) at the marked token is P's normed output, `a_P(x)` times its down-projection column under the inverse RMS and gain of layer L's post-feedforward norm (OLMo 3's post-norm layout; amended at freeze from "the layer's RMSNorm scale"), read by k's gate and up rows, as `unite_route.direct_A` reads upstream contributions with the roles reversed; the 8 primary poles in the last layer have no MLP reader and are counted as not run, so this spec reads 142 jobs; the worker adds a downstream capture of every MLP neuron's activation at the marked token. `R(J)` = the r readers with the largest `Σ_{x∈J} |C_{P→k}(x)|`; **r is frozen on the sniper jobs** as h in SPEC B (smallest of 10, 20, 50 with own effect ≥ 0.25 of the total and room ≥ 0.20).

**Intervention.** Hold P's activation at its background mean at every position of the window (one forward per context, plus identity): every downstream neuron's activation is captured before and after, so all reader sets are read from the same two forwards. The effect on a reader set S on context x is `E(S; x) = Σ_{k∈S} |a_k(x) − a_k^P(x)|`, normalised by the identity-forward activation scale of S; job effect = mean over contexts with a_P(x) > 0. **Statistic** `I_out(J) = E(R(O*); J) / E(R(J); J)`; foreign readers `R_foreign` = the top readers of the foreign pole Q on Q's job; random-matched readers = per reader in R(O*), the neuron of the same layer outside both sets nearest in background |a|. Cancel rule: `E(R(J); J) < 2 · E(random; J)` or fewer than 3 positive contexts.

**References at the birth checkpoints.** Null: the 56 planted two-neuron sums; P is the sum, its readers on each side by the same rule; sides capped at 12; labels and shuffled labels. Ceiling: the 138 own ceiling splits, side 2 read with R(side 2) against R(side 1). Weights and bootstraps as SPEC §6 and SPEC2 B1.

## 4. Rules (bootstrap over neurons, 2,000 draws, seed 0)

- **R1o.** Median I_out at birth over live jobs against the null at birth: overlap → refuted; real higher → not refuted.
- **R2o.** Median E(R(O*); J) − E(R_foreign; J): interval above 0 → specific.
- **R6o.** Spearman of I_out with I_MLP (from `results/primary/verdict.json`) over live jobs, bootstrap over neurons: interval above 0 → input and output reuse go together.
- Reported, no rule: I_out at T_15 and its rise against the ceiling's; the reader sets' overlap; the depth of R(J) (median layer gap above L); logit readers (the top-50 vocabulary items by direct logit contribution of P on J against O*), as a second output channel with no rule.

## 5. Gates

Known number and completeness as SPEC §5 at every load; identity |Δ| ≤ 1e-3 on every downstream neuron read; positive control median own effect at frozen r ≥ 0.25 of the total downstream effect, failing input random-matched readers; room: median ceiling I_out at birth minus the null's 95th percentile ≥ 0.20, failing input shuffled planted labels; cancel share under 40%; reader capture check: the instrument's δg_k and δu_k over the MLP neurons of layer L+1 equal the gate and up projections of layer L+1 applied in fp32 to P's normed output computed from the weights alone, within 1e-2 relative (the line of REUSE's completeness gate, set at freeze with SPEC B's), on every context of every job at every load; failing input, the down column of the wrong layer.

## 6. Point predictions, no rule attached

Own reader effect at frozen r 0.40 of the total; null at birth 0.05; ceiling 0.60; median I_out at birth 0.30 [0.20, 0.40], so R1o not refuted; R2o specific; R6o positive, Spearman 0.35 [0.15, 0.55]; readers lie within four layers above L for over half the jobs; related cases (clusterer's label, descriptive) above unrelated by 0.15.

## 7. Budget and runbook

One forward pair per context: 150 jobs × 2 × 12 × 2 checkpoints, 138 ceilings × 2 × 12, 56 planted × 2 × 24 × 8 loads, about 30,000 forwards but run past L to the last layer, about 0.6 GPU-h; sniper 0.15; cap 1.0. $0 API. Steps as SPEC B §7; verdict `verdict_output.py`; tag `reuse-output-freeze-1`. SPEC A and SPEC B can share one pipeline on the same nine loads if both are frozen before the launch; the job then runs both stages per load, about 1.9 GPU-h together, under the program's remaining 3.27.

## 8. Limits known now

Holding P at its mean removes every path from P, direct and through attention; the reader sets are chosen by the direct path only. Readers are MLP neurons; attention heads downstream are not read except through the logit channel, which has no rule. The null and ceiling constructions are REUSE's, with their stated limits. One model, one run.
