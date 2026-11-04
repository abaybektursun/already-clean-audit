# SPEC10 — does the newer task depend on the older task's drivers specifically, or on drivers the neuron's tasks share?

Status: drafted 2026-11-03 03:50 CST (coordinator); pre-freeze amendments of 04:15 CST below (§3, §5, §6, from the builder's notes `results/review10/BUILD_NOTES_sharedinputs.md`); frozen at tag `reuse-freeze-10` before any forward. Parents: `SPEC9_sameweights.md` and its read (`results/sameweights/read/READ_sameweights.md`: the pole's other clusters share drivers with the new cluster as much as the old one does, 0.125 against 0.140, d_third 0.000 [−0.020, 0.030] on 108 cases), `SPEC4_birth.md` (the exclusion, R1c15, the cancel rule), `SPEC6_generic.md` (the T_15 free-scale own′ and cross′ falls of the primary cases, the known number). Model `allenai/Olmo-3-1025-7B` at T_15 only.

## 0. What is known

The program's headline instrument is the ablation ratio I = Δ(D′(O*); J) / Δ(D′(J); J): the fall of the pole's activation on the new cluster J when the old cluster's drivers are mean-replaced, over the fall when J's own drivers are. At T_15 with the 1% exclusion it reads 0.294 [0.229, 0.345] on the 150 primary cases against a two-neuron null′ of 0.065 [0.034, 0.088] (SPEC4 R1c15; SPEC6 re-read 0.302 on 113 cases). The driver overlap between J and O* (0.14) equals the overlap between J and the pole's other clusters (0.125), so the drivers J shares with O* may be drivers J shares with every task of the neuron. The ablation has never been read with another cluster of the same pole in O*'s place.

## 1. Question, decision, hypotheses

**Question.** Is J's dependence on O*'s drivers larger than its dependence on the drivers of the pole's other clusters?

**Decision.** If the dependence is specific to O*, the post keeps "the newer task runs partly through the older task's inputs" with SPEC9's qualifier; if it is shared, the post's headline becomes "a neuron's tasks run through shared inputs" and the old–new pairing is dropped from the claim.

| hypothesis | statement | refuted when |
|---|---|---|
| H_old | J depends on O*'s drivers more than on the pole's other clusters' drivers | the paired difference I_O − I_K has its interval below 0.05 (X1f) |
| H_shared | J's dependence on O*'s drivers is the same as on any other cluster of the pole | the interval above 0.05 (X1f) |
| Instrument artifact | | any gate of §5 fails |

## 2. The measurement

At T_15, free scale, the 1% exclusion (SPEC4's instrument, `reuse_ablate` and `reuse_birth` unchanged): for each case with at least one other live cluster K on the same pole among the cases (K ≠ J and K ≠ O* by cluster index; every such K, deduplicated), the sets D′(J), D′(O*) and D′(K) ranked from captures on their own contexts at this load; the arms read on J's contexts: own′ (D′(J)), cross_O (D′(O*)), cross_K (D′(K), one arm per K), random′ (SPEC4's energy-matched set); job falls as SPEC4 (mean over contexts with a_P > 0 of 1 − a_P(ablated)/a_P(clean)). I_O = cross_O/own′; I_K = the mean over the case's K of cross_K/own′; d = I_O − I_K per case. Reported beside the rule: K's own fall (D′(K) on K's contexts) as K's positive control; the reverse arm D′(J) on K (symmetry); the sibling split (K that shares the case's O*, or not); cross_K per K.

## 3. Design and data

One load, T_15. Cases: SPEC4's live cases whose pole has at least one other live cluster among the cases: 107 (64 primary on 28 neurons, 43 fresh on 17; the read's 108th, `L10N4041+:c1`, has a cancelled sibling only; *amended 04:15 CST before freeze*), 135 K reads (1 or 2 per case), the list banked by the job (`cases.json`). Primary is discovery; fresh is confirmation and decides. Each arm has a clean and an ablated readout (the npz is written as each arm is read) and the parent's identity arm is read on J: about 16,300 readouts and 3,400 captures, 4.9 M tokens, about 0.2 GPU-h.

## 4. Known numbers

The primary cases' own′ and cross_O at this load against SPEC6's T_15 free-scale falls (`results/generic/1/generic/t15/falls/<case>__own__g1__free.npz` and `__cross__g1__free.npz`, the same instrument on the same card type): median absolute difference at most 0.01; failing input the next case's falls. The parent's known number against the harvest and the capture identity as SPEC4.

## 5. Gates

Known numbers (above; the job records the SPEC6 comparison and the verdict decides it); capture identity (20 contexts twice, 0; failing input the hook on L−1); exclusion (no set member in B_t; failing input the unexcluded ranking); positive control (own′ median at least 0.25, read per set; failing input random′); cancel (SPEC4's rule at this load; share under 0.40 per set; failing input own′ and random′ swapped); K validity (every K is a live cluster of the same pole, not J and not O* by index, with its own contexts; failing input K replaced by J). *Amended 04:15 CST before freeze: the positive control and the cancel share are read per set, as SPEC4's verdict reads them.* A failed gate stamps every rule "not read".

## 6. Frozen rules

- **X1 (primary, discovery) and X1f (fresh, confirmation; X1f decides).** The median over the listed cases not cancelled at this load by SPEC4's cancel rule (*amended 04:15 CST before freeze: "live" in §6 names this load's cancel rule; d over every listed case is reported beside the rule*) of d = I_O − I_K, bootstrap over neurons (2,000 draws, seed 0, 95% percentile): interval above 0.05 → H_shared refuted (the dependence names the older task); below 0.05 → H_old refuted (the dependence is shared); containing 0.05 → not separated. The line 0.05 is the smallest difference the program has called a distinction between two ratios on these cases (SPEC4's R2c read 0.088 and SPEC2's R2b 0.126 as "specific"; the ratio intervals are about ±0.06 wide), set before any forward.
- **Reported, no rule:** I_O and I_K with intervals on both sets; the share of cases with I_O > I_K; K's own fall; the reverse arm; the sibling split; I_O against SPEC6's re-read (0.302 on the finite cases).

## 7. Point predictions, before any forward

| quantity | predicted | refuted if |
|---|---|---|
| I_O primary; fresh | 0.29 [0.23, 0.35]; 0.26 [0.18, 0.34] | outside [0.10, 0.50] |
| I_K primary; fresh | 0.26 [0.18, 0.33]; 0.24 [0.15, 0.33] | outside [0.05, 0.50] |
| d primary (X1); fresh (X1f) | 0.03 [−0.03, 0.09]; 0.02 [−0.05, 0.09] | outside [−0.15, 0.25] |
| share with I_O > I_K, fresh | 0.55 [0.42, 0.68] | outside [0.3, 0.9] |
| K's own fall (positive control for K) | 0.50 [0.42, 0.58] | under 0.25 |
| GPU hours (one load) | 0.4 [0.25, 0.7] | over 1.2 |

The author expects shared dependence (d near 0), from SPEC9's overlap result.

## 8. Threats

| threat | effect | mitigation |
|---|---|---|
| K shares the case's O* (siblings on the same pole share the oldest cluster) | K's drivers are O*'s drivers in part | the sibling split reported; K ≠ O* by index enforced |
| Few cases per neuron, 46 neurons | wide intervals | the bootstrap over neurons; the fresh set decides |
| J and K share a document | cross_K inflated | the corpus rule's flag reported; SPEC9 found none between J and O*, 12 of 108 between J and K |
| The 1% exclusion's own effect | both ratios lowered alike | paired per case; the exclusion is the program's standing construction |

## 9. Runbook

Worker (Opus, own worktree `../interp_research-sharedinputs`, branch `research/sharedinputs`): `infra/aws_pg/jobs/reuse_sharedinputs.py` (a subclass of `jobs/reuse_birth.py`'s Job; one T_15 stage; the case list with its K's banked; `falls/<case>__<arm>.npz` as SPEC4's; `samples/` with three cases' prompts and sets; `metrics.json` with every gate), `steps/reuse_sharedinputs.json`, `scripts/verdict_sharedinputs.py` (`--failing-input known_next_case | capture_wrong_layer | positive_random | cancel_swapped | k_is_j`, X1 and X1f, `GRID10.md`), the dry run `jobs/reuse_dryrun_sharedinputs.py`. Coordinator: freeze (`reuse-freeze-10`), the dry run, launch (`g6e.2xlarge,g6e.4xlarge`, `--no-weights`, fresh prefix `results/reuse/sharedinputs/1`, budget 90 min), samples, §7 restated, verdict once, failing inputs, recount (seed 1), a different model's read, figures, the register. Cost about 0.4 GPU-h, a further recorded deviation; API $0.

## 10. Grid, filled once

| rule | reading | outcome |
|---|---|---|
| X1 | | |
| X1f | | |
| gates | | |
| GPU hours | | |
