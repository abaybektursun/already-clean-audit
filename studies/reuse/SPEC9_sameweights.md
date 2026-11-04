# SPEC9 — is the shared-driver overlap specific to the neuron's two clusters, or a property of its weights?

Status: drafted 2026-11-03 01:50 CST (coordinator); pre-freeze amendments of 02:40 CST below (§5, §8, from the builder's notes `results/review9/BUILD_NOTES_sameweights.md`); frozen at tag `reuse-freeze-9` before any forward. Parents: `SPEC4_birth.md` (S1, S3; the exclusion), `SPEC6_generic.md` (the T_15 bank of D′(J) and D′(O*) for the 150 primary cases, `results/generic/1/generic/t15/prebirth/`), `results/sanity/SANITY_science.md` item 7 and `results/sanity/s4_sameweights.py` (the proxy), `results/REVIEW_birth.md` addendum item 3. Model `allenai/Olmo-3-1025-7B` at T_15 only.

## 0. What is known

S1 (SPEC4, confirmatory on 91 live fresh cases): a pole's two clusters share 0.120 [0.080, 0.160] of their top-50 drivers against a two-neuron null whose 95th percentile is 0.060 and a same-cluster-halves ceiling of 0.38; the primary cases read 0.12 (S3). The outside reader's first objection (`results/sanity/SANITY_science.md` item 12): two context sets of one neuron may share drivers because they share that neuron's input weights, and the two-neuron null, which changes the weights as well, cannot answer that. The sanity check's proxy (a weights-times-activity score on SPEC4's Part A arrays, which reproduces D′(J) at 0.64) gave 0.02 for other cases' contexts read through the pole's weights against 0.10 and 0.12 for O*'s contexts, O* beating the swap in 82% and 78% of cases. A proxy, not the instrument; disclosed as prior knowledge.

## 1. Question, decision, hypotheses

**Question.** Read with the instrument (direct attribution at the marked token, the top 50 with B_t excluded), do the drivers of a pole's new cluster overlap the drivers of its old cluster more than they overlap the drivers of other neurons' contexts read through the same pole?

**Decision.** If the overlap is specific, S1 stands as a claim about the two clusters and the post's overlap paragraph keeps its wording; if the same-weights overlap matches O*'s, S1 is restated as a property of the pole's weights and the two-neuron null is retired as the reference for overlap.

| hypothesis | statement | refuted when |
|---|---|---|
| H_specific | D′(J) overlaps D′(O*) more than it overlaps D′ of other neurons' contexts read on the same pole | the paired difference s_O − s_swap has its interval below 0.03 (W2f) |
| H_weights | The overlap is the pole's weights: any two context sets read on it share drivers alike | the interval above 0.03 (W2f) |
| Instrument artifact | | any gate of §5 fails |

## 2. The measurement

At T_15, for each case (pole P, clusters J and O* with their 12 contexts each as SPEC4 capped them): a capture of P and every upstream neuron at the marked token of each context (SPEC4's `unite_route.capture`, 256-token window, g and u in fp32), the direct attribution of each upstream neuron to P summed over the set's contexts (`reuse_ablate.drivers_direct`), and D′(X) = the top 50 by attribution with B_t excluded (SPEC4's `top_m_excluded`, B_t recomputed at this load from ā over UNITE's e1 positions). Three swapped sets per case: the J contexts of three other cases on a different neuron, drawn once by a seeded rule (`numpy.random.default_rng(9)`, cases in item order, from the pool of all cases of both sets on another neuron), read on P. s_O = |D′(J) ∩ D′(O*)| / 50 (S1's statistic); s_swap = the mean over the three swaps of |D′(J) ∩ D′(swap_k)| / 50; d = s_O − s_swap per case. Live: SPEC4's live flags at birth (`results/birth/verdict_birth.json`, cases.primary_birth and fresh_birth, not cancelled). Reported beside the rule: s_swap against the two-neuron s null′'s 95th percentile (0.060, SPEC4); the per-swap overlaps; the overlap of D′(O*) with the swaps; the share of cases where s_O > s_swap; the distribution of s_O (the S1 statistic at T_15 on both sets).

## 3. Design and data

One load, T_15. Cases: SPEC4's 150 primary (discovery) and 93 fresh (confirmation) from `inputs/birth_items.json` (sha12 `31706fd1fdbe`); the fresh cases' T_15 captures are new (SPEC4 read them at birth only). Captures per case: 24 own contexts and 36 swapped; about 14,600 captures in all, no ablation.

## 4. Known numbers

The job's D′(J) and D′(O*) for the primary cases against SPEC6's banked sets at T_15 (`generic/1/generic/t15/prebirth/<case>__J.npz`, `__O.npz`, `D_layer`, `D_neuron`; on S3 and local): the median over cases of |D′ ∩ D′_SPEC6| / 50 at least 0.90 (the same instrument on the same card type; bfloat16 jitter can swap neighbours at the cut); failing input: the next case's SPEC6 set (near 0). The S1 statistic at T_15 is new on both sets (SPEC4 read s′ at the birth checkpoints: 0.120 on the fresh cases and 0.120 on the primary, `results/birth/verdict_birth.json → rules.S1, rules.S3`); its T_15 value is reported beside them, no rule. *(Wording fixed 01:55 CST before freeze.)*

## 5. Gates (each with its failing input and identity value)

- Known number (above); the parent job's known number against the harvest (`unite_stats.known_ratio` ≤ 0.01; failing input the previous checkpoint's bank) and completeness; capture identity (20 contexts twice, 0; failing input the hook on L−1).
- Exclusion: no member of any D′ is in B_t; cut count 3,412; failing input the unexcluded ranking against B_t.
- Swap validity: every swapped set's source case is on a different neuron from the pole's, and its contexts are its own J contexts (asserted against the items); failing input: a swap drawn from the same neuron.
- Room: the same-cluster-halves ceiling for the overlap, read on SPEC4's 129 primary ceiling splits at T_15 as the items give them (the two halves of O*, the construction behind SPEC4's 0.38; *amended 02:40 CST before freeze: §5 had written "halves of J"*), captured here, minus the two-neuron 95th percentile (0.060) at least 0.20 (on SPEC4's T_15 data the room is 0.32); failing input: the halves' labels shuffled across cases.
- A failed gate stamps every rule "not read".

## 6. Frozen rules

- **W2 (primary, discovery) and W2f (fresh, confirmation; W2f decides).** The median over live cases of d = s_O − s_swap, bootstrap over neurons (2,000 draws, seed 0, 95% percentile): interval above 0.03 → H_weights refuted (the overlap is specific to the two clusters); below 0.03 → H_specific refuted; containing 0.03 → not separated. The line is half the two-neuron null′'s 95th percentile (0.060), set on SPEC4's control data; the proxy's prior reading (0.10 − 0.02) is disclosed.
- **Reported, no rule:** s_swap's median with interval against 0.060 (if above, the two-neuron null under-states what two context sets on one pole share); the share of cases with s_O > s_swap; the per-swap spread; s_O on both sets at T_15.

## 7. Point predictions, before any forward

| quantity | predicted | refuted if |
|---|---|---|
| s_O primary at T_15; fresh | 0.12 [0.10, 0.14]; 0.12 [0.08, 0.16] | outside [0.04, 0.25] |
| s_swap primary; fresh | 0.02 [0.01, 0.04]; 0.02 [0.01, 0.04] | outside [0.00, 0.10] |
| d primary (W2); fresh (W2f) | 0.09 [0.06, 0.12]; 0.09 [0.05, 0.13] | outside [0.00, 0.20] |
| share of cases with s_O > s_swap, fresh | 0.80 [0.70, 0.88] | outside [0.5, 0.95] |
| known number (median overlap with SPEC6's sets) | 0.96 [0.92, 1.00] | under 0.90 |
| GPU hours (one load, captures only) | 0.25 [0.15, 0.45] | over 0.8 |

## 8. Threats

| threat | effect | mitigation |
|---|---|---|
| Swapped contexts come from poles whose contexts the pole hardly fires on | the attribution ranking on a set where P is near zero is noise, which lowers s_swap for the wrong reason | the pole's mean activation on each swapped set is banked and reported; the rule is also read on the swaps where P's mean activation is at least a tenth of its mean on J (reported) |
| Shared documents between J and O* | s_O inflated by the same passage on both sides | the items carry no flag; the verdict computes it from the corpus's document offsets with the third set's rule, which gives 0 of 150 primary and 0 of 93 fresh cases (*amended 02:40 CST before freeze*); reported |
| The three swaps are too few | s_swap noisy per case | the median over 91 live cases; the per-swap spread reported |
| bfloat16 rank jitter at the cut | the known number under 0.90 | the gate's line set at 0.90 with the rank jitter in mind |

## 9. Runbook

Worker (Opus, own worktree `../interp_research-sameweights`, branch `research/sameweights`; the same-neuron failing input is a fixed substitution of swap 1 by the first other case on the pole's neuron, and the capture-identity tolerance is SPEC5's 1e-5; *noted 02:40 CST before freeze*): `infra/aws_pg/jobs/reuse_sameweights.py` (one stage at T_15, captures only, reusing `jobs/reuse_birth.py`'s loading, ā, B_t, attribution and `top_m_excluded`; banks per case `sets/<case>.npz` with D′(J), D′(O*), D′(swap_1..3), the top-200 scores, P's mean activation per set, the swap sources; `samples/` with three cases' prompts and sets; `metrics.json` with every gate), `steps/reuse_sameweights.json`, `scripts/verdict_sameweights.py` (gates with failing inputs `--failing-input known_next_case | capture_wrong_layer | swap_same_neuron | room_shuffled`, W2 and W2f, `GRID9.md`), dry run in a new module `jobs/reuse_dryrun_sameweights.py`. Coordinator: freeze (`reuse-freeze-9`), the dry run, launch (`g6e.2xlarge,g6e.4xlarge`, `--no-weights`, fresh prefix `results/reuse/sameweights/1`, budget 60 min), samples opened, §7 restated, verdict once, failing inputs, recount (seed 1), a different model's read, figures, the register. Cost about 0.25 GPU-h on the AWS lane, a further recorded deviation; API $0.

## 10. Grid, filled once

| rule | reading | outcome |
|---|---|---|
| W2 | | |
| W2f | | |
| gates | | |
| GPU hours | | |
