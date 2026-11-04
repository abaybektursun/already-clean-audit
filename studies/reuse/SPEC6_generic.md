# SPEC6 — the generic term of mean-replacement controls: the norm-scale path, the cut, and a per-layer rule

Status: drafted 2026-11-02 20:50 CST (coordinator); pre-freeze amendments of 21:55 CST below (§5, §9, from the builder's notes `results/review6/BUILD_NOTES_generic.md`); frozen at tag `reuse-freeze-6` before any forward. Parents: `SPEC4_birth.md` (frozen `reuse-freeze-4`), `NOTE_high_background.md` with its addendum, `results/REVIEW_birth.md` addendum, `results/sanity/SANITY_science.md` item 4, `NEXT.md` items 2 and 6. Model `allenai/Olmo-3-1025-7B` at T_15 (`stage1-step1413814`), the final checkpoint only.

## 0. What SPEC4 found, in the words this spec uses

A control arm's fall has a generic term: the part that occurs for reasons unrelated to the two clusters. At T_15 the two-neuron null (56 planted pairs of unrelated neurons, read as one neuron with two clusters, SPEC2 B1 weighting) is 0.163 [0.110, 0.220] with every neuron eligible and 0.065 [0.034, 0.088] with the top 1% of upstream neurons by |ā| (B_t, 3,412 of 341,248) excluded from every set: a gap of 0.098 between medians (`results/birth/before_nulls.json`, `results/birth/verdict_birth.json → rules.M2`). The sanity check found that B_t by the global cut is mostly a depth effect (a layer's median |ā| rises about 100-fold from layer 0 to 30; the cut takes 7.7 to 14.3% of layer 30 and almost nothing below layer 20), that only 10 to 131 distinct members per load enter any set, at a median 129 to 639 times their own layer's median, mostly in layers 8 to 25, and that the null's pairs sit in layers 5, 15 and 26 only (`results/sanity/SANITY_science.md` item 4). The method note's mechanism is an inference, not a measurement: this model normalises each MLP's output (`post_feedforward_layernorm`, an RMSNorm) before adding it to the residual stream, so replacing a large-|ā| neuron by its mean changes that norm's divisor at the position and rescales every other neuron of the layer. The remainder of the null's rise after the exclusion, 0.053 [0.025, 0.085], has no named source.

## 1. Question, decision, hypotheses

**Question.** Through which path do the high-background neurons move the controls: the post-feedforward norm's scale, or their direct contribution to the residual stream? Is the 1% cut a boundary or the start of a tail? Does a per-layer outlier rule do the global cut's work?

**Decision.** If the norm path carries the term, every mean-replacement ablation in this program holds the post-feedforward norm scale fixed from now on, and the exclusion becomes optional; if the direct path carries it, the exclusion stays and is redefined per layer if N3 says the per-layer rule does the same work. If the 3% cut lowers the null further, the cut is a tail and the next spec sets it from the null's convergence; if not, 1% is a boundary.

| hypothesis | statement | refuted when |
|---|---|---|
| H_norm | The high-background neurons move the controls through the norm's scale: holding the scale at its unablated value removes most of the gap between the unexcluded and the excluded null | the paired fall of the unexcluded null from free to held scale, Δ_held, has its interval below 0.049 (half the gap) (N1) |
| H_direct | The neurons act through their direct contribution; holding the scale changes the unexcluded null little | Δ_held's interval above 0.049 (N1) |
| H_tail (boring) | The remainder after the 1% cut is the next tail of the same distribution: a 3% cut lowers the null further | the paired difference null′_1% − null′_3%, Δ_cut, has its interval below 0.02 (N2) |
| H_boundary | The 1% cut is a boundary: the remainder has another source | Δ_cut's interval above 0.02 (N2) |
| H_depth | The global cut's effect comes from the few mid-layer outliers, which a per-layer rule keeps while dropping the last-layer bulk | null′ under the per-layer cut exceeds null′_1% by more than 0.02 by its interval (N3) |
| H_reuse-held | The reuse ratio at T_15 survives the held scale | I′_held's interval overlaps null′_held's (N4) |
| H_weights | Driver lineage can be read from weights alone: the top 50 upstream neurons by the weights-only alignment ω overlap the attribution-ranked drivers D′(J) well above two unrelated neurons' overlap | the median overlap's interval lies below 0.12 (W1); prior knowledge disclosed: the sanity check's proxy read 0.08 at the birth checkpoints (`results/sanity/SANITY_science.md` item 7), so a refutation on the unseen T_15 data is expected |
| Instrument artifact | | any gate of §5 fails |

## 2. The measurement

The instrument is SPEC4's (`infra/pg/reuse_ablate.py`, `infra/pg/reuse_birth.py`): a set S of upstream MLP neurons is replaced at the input of `layers[l].mlp.down_proj` by its mean activation ā over UNITE's e1 positions at T_15, at positions from the window start, and the pole's activation a_P at the marked token of each context is read with the forward stopped after the pole's layer; a fall is 1 − a_P(ablated)/a_P(clean), the job fall its mean over contexts with a_P > 0; sets are the top 50 by direct attribution (D′), the energy-matched random set (random′), a foreign neuron's drivers (foreign′), all ranked with B_t excluded as SPEC4 did (the "excluded" condition) or with no exclusion (the "unexcluded" condition, SPEC2's construction), and under a 3% cut and a per-layer 1% cut (§3).

**New: the held scale.** During the clean forward of a context the job records, for each layer l that any arm ablates, the RMSNorm divisor of `layers[l].post_feedforward_layernorm` at every position: inv_l(x, p) = rsqrt(mean over the hidden dimension of y_l(x, p)² + eps), y_l the MLP output. During an ablated forward with scale held, the norm computes weight ⊙ y_l^ablated(x, p) ⊙ inv_l(x, p) with the recorded inv, so the replacement changes only the direct contribution of S to the residual stream and nothing else in the layer is rescaled. With scale free, the norm computes its own divisor as the model does (SPEC4's condition). Every arm is read under both.

**Quantities.** Per pair and per case: the fall under each (arm, exclusion, scale). The two-neuron null under (exclusion, scale) is the SPEC4 construction (cross fall of side 2's set on side 1's own fall, tercile thirds, inverted-CDF weighted quantiles, bootstrap over pairs within tercile, 2,000 draws, seed 0). Δ_held = null_unexcluded_free − null_unexcluded_held, paired per pair, median with its interval. Δ_cut = null′_1%_free − null′_3%_free, paired. I′_held = median over live cases of cross′_held/own′_held (SPEC4's R1c15 form under the held scale), against null′_1%_held. **Weights-only alignment (W1, from the queue's item 5).** For every live case at T_15 the job banks SPEC4's pre-birth arrays for J and O* (the pole's rows G_P, U_P; the weights-only loads G_P·d_k and U_P·d_k for every upstream neuron k, d_k = γ_l ⊙ W_down[:, k]; mean |a_k| over the contexts; the per-layer scale; the attribution top 200). Offline, ω_k = |c̄_g G_P·d_k + c̄_u U_P·d_k| with c̄_g, c̄_u the case's mean coefficients over J (SPEC4 §3.1's definition), B_t excluded; the top 50 by ω against D′(J): the overlap as a share of 50, median over live cases, bootstrap over neurons. Reported beside it: the proxy mean|a_k| · s_l · ω_k (the sanity check's construction) and the same for O*. Reported: the same arms for the 1%-excluded null under the held scale (expected near-unchanged), the members-alone arm (only the B_t members that the unexcluded cross set contains, replaced alone, free and held: the term's size and its norm share measured directly), own′_held (positive control), the ceiling′ under the held scale, and the count of distinct B members entering sets under each cut.

## 3. Design and arms

One load, T_15. Items: the 150 primary cases of SPEC4 (`inputs/birth_items.json`, live rule of `results/birth/verdict_birth.json` applied in the verdict) and the 56 planted pairs of SPEC2. Exclusion conditions: none; global 1% (B_t as SPEC4, `high_background.json` recomputed at this load from ā); global 3% (the top 3% by |ā|, 10,237); per-layer 1% (the top 1% by |ā| within each upstream layer, 110 per layer). Scales: free, held. Arms per case: own′, cross′, random′, foreign′ under (global 1%) × (free, held); the ceiling′ reads (the same-cluster halves, 129 in SPEC4) under (global 1%) × (free, held). Arms per pair: own and cross (both sides) under (none, 1%, 3%, per-layer) × (free, held); the members-alone arm under (none) × (free, held). Every arm's sets are ranked at this load from recomputed direct attribution, as SPEC4 did; SPEC4's banked sets are the known number for the ranking.

## 4. Data

Cases and pairs as in SPEC4 (`inputs/birth_items.json`, sha12 `31706fd1fdbe`; the pairs from `results/sniper/2/` planted sides). Contexts per cluster 12 (cap), the harvest's exemplars, 256-token window. The corpus for ā: UNITE's e1 positions at T_15 (as SPEC4's `abar.npz`). The ablation's mean values and the attribution ranks are recomputed on the instance; the comparison with SPEC4's T_15 bank is the known number.

## 5. Gates (each with its failing input and its identity value)

- **Known number.** (a) The clean a_P of each case's contexts at this load against SPEC4's T_15 bank, median relative error at most 0.01 (SPEC4 read 0 to 0.0031; failing input: SPEC5's dating bank at T_14 on the same contexts in place of T_15, since SPEC4 has no T_14 load; *amended 2026-11-02 21:55 CST before freeze*). (b) The free-scale own′ and cross′ falls of 20 cases against SPEC4's banked falls, median absolute error at most 0.01 (failing input: another case's falls). (c) The unexcluded null at T_15, free scale, against `before_nulls.json` 0.163 to within 0.03 of the median (failing input: the excluded null, 0.065).
- **Held-scale identity.** A held-scale forward with an empty set reproduces the clean a_P exactly (max absolute difference 0, in the model's dtype) on 20 contexts (the first primary case with L ≥ 2: its J contexts then its O* contexts, first 20; *amended 21:55 CST before freeze*) and on every identity-held arm of the load; failing input: the inv recorded from layer l−1 applied at layer l (nonzero).
- **Held-scale completeness.** On the same 20 contexts, the held-scale ablated norm output equals weight ⊙ y^ablated ⊙ inv_clean recomputed in fp32 by an independent probe and cast to the output dtype as the norm casts, to 1e-4 relative (the set: the case's own set ranked without exclusion, so the divisor moves; *amended 21:55 CST before freeze: the cast, since a bfloat16 output rounds by up to 2e-3 against an uncast fp32 value*); failing input: the free-scale output in its place.
- **Positive control.** own′_held median over live cases at least 0.25 (own′_free was 0.544 in SPEC4); failing input: random′_held.
- **Room.** Under the held scale, the ceiling′_held median minus the 95th percentile of null′_1%_held at least 0.2 (SPEC4's room-c15 was 0.296); failing input: the labels own and cross swapped within each pair with probability one half (offline, seed 0; *amended 21:55 CST before freeze*).
- **Cancel.** SPEC4's cancel rule on the live cases; cancelled cases are excluded and counted.
- **Completeness and memory** as SPEC4.

Every verdict reads its gates; a failed gate stamps every rule "not read". Every gate was run on its failing input in the dry run and must fail there.

## 6. Frozen rules

- **N1.** Δ_held over the 56 pairs (paired per pair: the tercile-weighted median of the per-pair differences, bootstrap over pairs within tercile, 2,000 draws, seed 0; the difference of the two weighted medians reported beside it; *wording fixed 21:55 CST before freeze*): interval above 0.049 → H_direct refuted (the norm path carries most of the gap); below 0.049 → H_norm refuted; containing → not separated. Reported beside it: the same Δ for the 1%-excluded null, and the members-alone arm's fall free and held (the term's size and its norm share: 1 − held/free).
- **N2.** Δ_cut over the 56 pairs (paired): interval above 0.02 → H_boundary refuted (the cut is a tail); below 0.02 → H_tail refuted (1% is a boundary); containing → not separated.
- **N3.** null′_perlayer_free − null′_1%_free (paired): interval above 0.02 → H_depth refuted; otherwise H_depth not refuted; reported: the count of distinct members entering sets under each cut and the share of the gap each cut removes.
- **N4.** I′_held (median over live cases) against null′_1%_held (both with intervals): no overlap and I′_held above → H_reuse-held not refuted; overlap → H_reuse-held refuted; reported beside it: I′_free on the same cases at this load (SPEC4 read 0.294) and the paired difference I′_free − I′_held.
- **W1.** The weights-only overlap (median over live cases, interval over neurons): interval below 0.12 → H_weights refuted; at or above 0.12 with the interval above 0.06 (the two-neuron s null′'s 95th percentile, SPEC4) → not refuted; otherwise not separated. Failing input: another case's ω ranking (overlap near 0).
- **Reported, no rule:** own′ and cross′ under free and held for every case; the per-tercile readings of N1 and N4; the ceiling′ under held; the single-largest-member arm (the one member with the largest |ā| in each unexcluded cross set, replaced alone, free and held).

## 7. Point predictions, before any forward

| quantity | predicted | refuted if |
|---|---|---|
| Δ_held, unexcluded null (N1) | 0.06 [0.03, 0.09] | outside [0.00, 0.15] |
| Δ_held, 1%-excluded null | 0.01 [0.00, 0.03] | outside [−0.02, 0.06] |
| Δ_cut (N2) | 0.015 [0.00, 0.03] | outside [−0.02, 0.06] |
| null′_perlayer_free − null′_1%_free (N3) | 0.00 [−0.02, 0.02] | outside [−0.05, 0.05] |
| own′_held | 0.50 [0.42, 0.56] | outside [0.25, 0.70] |
| I′_held (N4) | 0.27 [0.20, 0.33] | outside [0.10, 0.45] |
| null′_1%_held | 0.05 [0.03, 0.08] | outside [0.00, 0.12] |
| members-alone fall, free; norm share 1 − held/free | 0.08; 0.6 | outside [0.02, 0.20]; outside [0.2, 0.9] |
| W1 weights-only overlap at T_15 | 0.08 [0.05, 0.11] | outside [0.00, 0.25] |
| proxy overlap at T_15 (reported) | 0.60 [0.5, 0.7] | outside [0.3, 0.9] |
| GPU hours (one load) | 1.0 [0.7, 1.5] | over 2.5 |

The author expects the norm path to carry more than half of the gap (the note's inference) and the 1% cut to be close to a boundary.

## 8. Threats

| threat | effect | mitigation |
|---|---|---|
| The held scale is implemented on the wrong module or position range | every held number wrong | identity and completeness gates with failing inputs; the module name `post_feedforward_layernorm` checked against the loaded model's layer; the recorded inv asserted per position |
| Pairs in three layers only (5, 15, 26) | N1 to N3 read on a null whose layers do not match the cases' | stated as a limit; N4 reads on the cases' own layers; per-layer readings of N1 reported |
| bfloat16 resolution (falls under about 0.5%) | small Δ unreadable | the lines (0.049, 0.02) are above the resolution; Δ reported with the per-pair standard error |
| SPEC4's T_15 bank was read on a different card | known-number mismatch | tolerance 0.01 set from SPEC4's cross-card agreement (0 to 0.0031) |
| 3% cut removes members of the sets' top 50 beyond the generic term | the null falls for reasons other than the tail | the own′ and ceiling′ under 3% reported; a fall of the ceiling′ above 0.05 marks the cut as cutting signal |

## 9. Runbook

1. Worker (Opus, own worktree `../interp_research-normheld`, branch `research/normheld`): a new `infra/pg/reuse_generic.py` importing SPEC4's `reuse_birth.py` unchanged (*amended 21:55 CST: the frozen instrument is not edited*) with the held-scale context manager, the 3% and per-layer cuts, the members-alone and single-member arms; `infra/aws_pg/jobs/reuse_generic.py` (one stage, T_15); `infra/aws_pg/steps/reuse_generic.json`; `studies/reuse/scripts/verdict_generic.py` (gates with failing inputs `--failing-input known_t14 | held_wrong_layer | completeness_free | positive_random | room_shuffled`, rules N1 to N4 and W1, the grid `GRID6.md`); `jobs/reuse_dryrun.py` part 10 with tiny tensors where the held norm is checked against a hand computation. No model on the laptop.
2. Coordinator: freeze (tag `reuse-freeze-6`), dry run green, launch on AWS (`g6e.2xlarge,g6e.4xlarge`, `--no-weights`, fresh prefix `results/reuse/generic/1`, budget 150 min), `pg watch` under a background shell; nothing scored before STATUS; samples opened (clean and held a_P for three cases, the members-alone arm's set printed) before any number; §7 restated in the journal before the verdict runs once; four failing inputs; recount under seed 1 (`scripts/review/fable_generic_recount.py`); a different model reads before any result is called closed; figures regenerated after corrections; the claims register updated.
3. Cost: about 1 GPU-h on the AWS lane. The REUSE study's cap (5.0 GPU-h) is already exceeded by 3.2 h; this run is a further recorded deviation, the coordinator's decision under the standing instruction to run the queue autonomously.

## 10. Grid, filled once

| rule | reading | outcome |
|---|---|---|
| N1 | | |
| N2 | | |
| N3 | | |
| N4 | | |
| W1 | | |
| gates | | |
| GPU hours | | |

## 11. Literature

Stolfo, Wu, Sachan, Belinkov 2024 ("Confidence regulation neurons", NeurIPS 2024) §3.2 holds the final-norm scale fixed to separate an entropy neuron's norm-mediated effect from its direct effect (read for SPEC4, `results/review4/LIT_high_background.md`). Sun et al. 2024 ("Massive activations in large language models") and Yu et al. 2024 (super weights) describe the published early-layer objects; the sanity check places the acting members in the middle layers. No paper read reports the norm-scale path of mean replacement for upstream (non-final) layers; the next search is the literature agent's `results/review6/`.

## 12. Definition of done

Gates with failing inputs run and failed in the dry run; the known numbers reproduce SPEC4; the verdict runs once; the recount and the read agree; the grid is full; the register has rows N1 to N4 and W1; costs in `infra/aws_pg/operations/`.
