# SPEC5, DATING: is a birth an event? The pole's activation on each cluster across the whole grid

Status: **frozen 2026-11-02 at the commit tagged `reuse-freeze-5`** (after the dry run passed and the verdict failed on each failing input); no forward before it. Earlier status: draft. Parents: `SPEC4_birth.md` (frozen, `reuse-freeze-4`), `results/REVIEW_birth.md` F1, Result 30, `results/birth/explore_rhoP_O.json`. Written by the coordinator (Fable 5.1). Budget note: the REUSE study stands at 5.847 GPU-hours against its 5.0 cap; this spec needs about 1.3 GPU-hours on the AWS lane. The coordinator runs it under the user's autonomy directive and the lane's own rule that AWS spend is not budgeted (`infra/vast/launch_preflight.sh`: "AWS side is unlimited by directive"); the overrun of the study's cap is recorded as a dated deviation in the journal. Nothing here is a rule of a parent; every rule carries its own letter.

## 0. What SPEC4 found, in the words this spec uses

UNITE dates a cluster J of pole P (one sign of one MLP neuron) as born at the first grid checkpoint t_b at which J's contexts sit among P's top 50. SPEC4 banked P's activation on J's and on the old cluster O*'s contexts at t_b and at t_b−1. The pole's mean activation on J at t_b−1 over that at t_b (ρ_P) is 0.95 at the median on 149 primary cases and 0.86 on 91 fresh; by birth load it is 2.2 to 2.7 for births at T_8 to T_10 and 0.7 to 0.8 at T_12 and T_13; 22 of 149 cases are under 0.5. The old cluster's own ratio is 1.27. Relative to O*, J stands at 0.57 of O*'s level at t_b−1 and 0.75 at t_b (ratio of ratios 0.77 [0.72, 0.85]; under 1 in 113 of 149 cases). The whole activation scale shrinks through training (the top-1% background cut falls from 25.6 at T_5 to 0.24 at T_15). So a birth under the top-50 dating is, at one grid step, a change of rank inside a shrinking scale, with the new cluster gaining about a third against the old one. Whether any birth is an event, a rise of the new cluster's relative activation by a factor of two or more between adjacent checkpoints at some point on the grid, was not measured: only one pair of checkpoints per case was read.

## 1. Question, decision, hypotheses

**Question.** Across all 16 grid checkpoints, does the pole's activation on the new cluster, relative to the pole's own scale at that checkpoint, rise by a factor of two or more between some adjacent pair of checkpoints (an event), or does it rise gradually (a drift)?

**Decision.** If births are events for most cases, Part A's question ("what arrives at a birth") is asked again on the event checkpoints with SPEC4's instrument; if births are drifts, the lineage program stops asking what arrives at a birth and asks instead how the relative activation grows, and the "birth" object is retired in favour of a dated activation curve.

| hypothesis | statement | refuted when |
|---|---|---|
| H_event | For most cases the relative activation of J doubles between some adjacent pair of checkpoints | fewer than half of the live cases have a doubling pair (D1) |
| H_drift | The relative activation of J rises without a doubling pair for most cases | at least half of the live cases have a doubling pair (D1) |
| H_old-decline (boring) | The rank change at t_b is the old cluster's decline, not the new one's rise: J's own relative activation is flat across the grid while O*'s falls | J's relative activation at t_b over that at t_b−2 (two grid steps) at least 1.2 by its interval, or O*'s at least 0.8 by its interval (D2) *(corrected 2026-11-02 16:15 CST before freeze: the draft stated the predicted pattern as the refutation)* |
| H_dating-stable | The top-50 dating agrees with an activation-based dating | the activation-based birth (the first checkpoint at which J's relative activation reaches half its final value) is within one grid step of t_b for at least two thirds of the cases (D3) |
| Instrument artifact | | any gate of §5 fails |

## 2. The measurement

For every case (150 primary and 93 fresh, J and O* at 12 contexts each by rank) and every grid checkpoint t in T_0 to T_15: one capture of P on J's contexts and on O*'s contexts (no ablation): a_P(x, t) at the marked token of each context. **Relative activation** r_J(t) = mean over J of a_P(x, t) divided by s_P(t), where s_P(t) is P's own scale at t: the median of a_P over P's 50 top contexts at T_15 (the UNITE harvest's exemplar contexts). The harvest banked a_P on those 50 contexts at every grid checkpoint (`<grid>/t<nn>/poles/exemplar_a.npz`, 600 poles × 50; every case's pole is among the 600), so s_P(t) is read from that bank and not re-captured; the known-number gate at each load is the equivalence check between this job's captures and that bank (it passed at 0 to 0.0031 in SPEC4). r_O(t) likewise. Banked per case: the raw means of a_P over J and over O* at the 16 checkpoints (from this job), s_P(t) (from the harvest bank, copied into the items by the builder), and r_J(t), r_O(t). *(Amended 2026-11-02 16:05 CST before freeze: the draft re-captured the scale contexts; the bank already holds them.)* **Validity (added 2026-11-02 16:15 CST before freeze, after the builder found 502 of the 3,888 case-checkpoint scales at or below zero, 126 cases at T_0 and 2 at T_10, none from T_11 on):** a (case, checkpoint) pair is valid when s_P(t) > ε with ε = 0.01 × s_P(T_15) for that pole; r_J and r_O are defined only there, every rule reads valid pairs only, and the share of valid pairs per checkpoint is reported. **Doubling pair:** adjacent valid checkpoints (t, t+1) with r_J(t+1) ≥ 2 r_J(t), r_J(t) > 0 and r_J(t+1) ≥ 0.1 (a floor against ratios of near-zero values). **Activation-based birth** b_act = the first valid t with r_J(t) ≥ 0.5 max over valid t of r_J(t).

The instrument is `unite_route.capture` at layer L with gu_layers {L} (as SPEC4's Part A banks), reading g and u of P at the marked token; a_P = σ silu(g) u. REUSED byte-identical: `unite_route.capture`, `pack`, `reuse_ablate.readout`'s conventions, `jobs/reuse_ablate.REVISION`, the known-number block for all 16 checkpoints (`sniper_items.known_block`, ts 0 to 15; T_0's failing input is T_1's bank). NEW: `infra/aws_pg/jobs/reuse_dating.py` (one load per call, captures only), `studies/reuse/scripts/build_items_dating.py` (the cases of `birth_items.json` plus, per case, s_P(t) for the 16 checkpoints from the harvest's `exemplar_a.npz` banks, with the bank files' sha256s), `studies/reuse/scripts/verdict_dating.py`, a dry-run part.

## 3. Design

Sixteen loads (T_0 to T_15), each: known number, then the captures for every case (243 cases × 2 sides × 6 to 12 contexts = 4,979 contexts), no readout forwards; the T_15 load adds the scale captures and the million-token pass. From SPEC4's rates, under 3 minutes of compute per load; the download dominates (the AWS lane's earlier REUSE runs took about 5 minutes per load including the download). AWS lane, `g6e.2xlarge`, `--no-weights`, one pipeline of 16 steps; estimate 1.3 to 2.6 h (the worker's count from SPEC4's rates); cap 3.0 GPU-h.

## 4. Data

`birth_items.json` (sha12 `31706fd1fdbe`) for the cases; the scale contexts from the harvest's positions bank for each pole (the 50 exemplar positions at T_15, `interp_research-neurons-explain/bank/w4/o7_s_v2/explain/positions.npz`, sha256 recorded by the builder); the known block for 16 checkpoints. Discovery and confirmation: the 150 primary cases are discovery for the thresholds of §6 (the doubling factor 2 and the floor 0.1 were set from SPEC4's one pair before this data); the 93 fresh cases are the confirmation set for D1 to D3, read once.

## 5. Gates

| gate | passes when | failing input |
|---|---|---|
| Known number | at each of 16 loads, as `SPEC.md` §5 | the previous grid checkpoint's bank (T_1's for T_0) |
| Capture identity | at every load, a second capture of 20 contexts equals the first to 1e-5 | the hook on the wrong layer (L−1) |
| Scale | s_P(T_15) > 0 for every case, and valid pairs (s_P(t) > ε) for at least 95% of cases at every checkpoint from T_9 on (set 2026-11-02 16:25 CST before freeze from the banked scales, which give 0.856, 0.872 and 0.926 valid at T_6 to T_8 and 0.955 at T_9; the shares at every checkpoint are reported); the job's own capture of P on its 50 top contexts at T_15 (once per distinct pole, 183 × 50 = 9,150 contexts, at the T_15 load only) gives a median within 5% of the bank's | the bank's median of a different pole |
| Anchor | at t_b and t_b−1, the mean of a_P over J at t_b−1 over that at t_b equals SPEC4's ρ_P per live case to 1e-2 (the known-number tolerance; SPEC4 ran on a different card), and the median absolute deviation is reported | SPEC4's ρ_P of a different case |

## 6. Frozen rules

- **D1.** The share of live cases (live: SPEC4's live rule at birth) with at least one doubling pair among their valid adjacent checkpoints, with a 95% bootstrap interval over neurons. Interval above 0.5 → H_event not refuted, H_drift refuted; below 0.5 → the reverse; containing 0.5 → not separated. Read on the primary set (discovery) and the fresh set (confirmation); the fresh set decides.
- **D2.** Medians over live cases with valid pairs at t_b and t_b−2 of r_J(t_b)/r_J(t_b−2) and r_O(t_b)/r_O(t_b−2), with intervals; H_old-decline refuted when the first's interval lies at or above 1.2 or the second's at or above 0.8; not refuted when the first's lies below 1.2 and the second's below 0.8; otherwise not separated. The count of cases without a valid pair at t_b−2 is reported.
- **D3.** The share of live cases with |b_act − t_b| ≤ 1 grid step; interval above 2/3 → the top-50 dating stands; below → an activation-based dating replaces it in the next spec.
- **Reported, no rule:** r_J(t) and r_O(t) medians per checkpoint on both sets (the curves), the distribution of the largest adjacent ratio per case, the share of doubling pairs that fall at t_b, and the same statistics with s_P replaced by the median |ā| of the layer (a global scale).
- **Reported, no rule (added 2026-11-02 16:20 CST before freeze, after the reviewing session's caveat that a 200-position mean can be set by one token):** at T_15 the job also takes ā over 2,000 windows of 512 corpus tokens (positions 1 to 511) and banks, per upstream neuron, the share of positions with a > 0 and the shares with |a| above half and above twice its 200-position ā; the verdict reports the Jaccard between B_15 (the 200-position top 1%) and the million-token top 1%, and for B_15's members the median share of positive positions and the median share above half ā. This is the stability check of SPEC4's exclusion set and the reviewing session's questions 2 and 3; no rule attaches.

## 7. Point predictions, before any forward

| quantity | predicted | refuted if |
|---|---|---|
| D1 share with a doubling pair, fresh | 0.35 [0.2, 0.5] | outside [0.1, 0.7] |
| D2 r_J(t_b)/r_J(t_b−2); r_O | 1.4 [1.1, 1.8]; 0.7 [0.5, 0.9] | outside [0.8, 2.5]; outside [0.3, 1.2] |
| D3 share within one step | 0.5 [0.35, 0.65] | outside [0.2, 0.8] |
| largest adjacent ratio, median | 1.6 [1.3, 2.0] | outside [1.1, 3] |
| GPU hours | 1.8 [1.2, 2.6] | over 3.0 |

The author expects drift: SPEC4's one pair gave a relative gain of about a third, and the grid's early steps are small.

## 8. Threats

| threat | effect | mitigation |
|---|---|---|
| The scale s_P is itself a rank-based quantity (top contexts at T_15) | r_J inherits the scale's own drift | the global scale (median |ā| of the layer) reported beside it |
| 12 contexts per cluster | noisy means | the cluster's contexts are the same at every checkpoint; the bootstrap is over neurons |
| Early checkpoints have huge activations | ratios of large numbers | the relative measure; the floor 0.1 on doubling pairs |
| The grid's late gaps | a doubling across 240,000 steps is not an event | D1 reports the step size of each doubling pair |

## 9. Runbook

1. Worker builds §2's NEW items in its own worktree; dry run on tiny models with a constructed case whose r_J doubles at one checkpoint and one whose r_J is flat; verdict green and failing on each §5 input. 2. Freeze `reuse-freeze-5`. 3. `./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args ' --steps-s3 s3://<s3-bucket>/results/reuse/dating/1/steps.json' --budget-min 185 --type g6e.2xlarge --no-weights --s3-prefix results/reuse/dating/1 --json --detach`; poll `metrics.json`. 4. Samples before any number: five cases' curves with their prompts and marked tokens; the anchor gate's per-case equality. 5. One verdict; a different model reads; Results as numbered sentences; `POST.md` where §1's decision says.

## 10. Grid, filled once

| row | predicted | observed | verdict |
|---|---|---|---|
| D1 fresh | 0.35 | | |
| D2 | 1.4; 0.7 | | |
| D3 | 0.5 | | |
| gates | pass | | |
| GPU hours | 1.8 | | |

## Amendment 5b (post-freeze, 2026-11-02 17:50 CST; the frozen text above is unchanged and its verdict stands as "rules not read")

The frozen run's per-load gates all passed; two instrument gates failed for definitional reasons found after the data: (1) the scale gate compared the job's capture of each pole's 50 scale contexts, taken with the 256-token window of the case captures, against the harvest bank's full-window exemplar activations; 16 of 183 poles differ by over 5% (one by 57%), so the bank's s_P(t) is not the job's own measurement; (2) the anchor gate required every live case's pre-birth ratio to reproduce SPEC4's banked ρ_P to 1e-2, but SPEC4 ran on a different card; the median absolute deviation is 2.6e-3 and 17 of 240 cases exceed 1e-2 (the birth-load means agree to 0.9% and the pre-birth means to 1.8%).

5b: a second pipeline of 16 loads captures, at every checkpoint, each of the 183 poles' 50 scale contexts with the same 256-token window and readout as the case captures, and banks per pole per load the 50 values and their median; s_P(t) is that median. The scale gate becomes: at T_15 the 5b medians equal the frozen run's T_15 scale medians to 1e-5 per pole (the same instrument twice; failing input the next pole's), s_P(T_15) > 0 for every case, and the validity shares from T_9 on at least 0.95 (reported at every checkpoint, under the 5b scales); the bank comparison is reported, not gated. The anchor gate becomes: the median absolute deviation over the live cases at most 1e-2, with the maximum and the count over 1e-2 reported (failing input unchanged). Every other definition and rule of §2 and §6 is unchanged and is read under the 5b scales, labelled "amendment 5b". Cost: 9,150 captures per load, about 1.3 GPU-h on the AWS lane; the study's cap overrun grows and is recorded. The §7 predictions stand as written for the 5b read.

## Post-freeze notes (2026-11-02 20:15 CST; the frozen text and the amendment above are unchanged)

1. Amendment 5b was filed after the frozen verdict had run and after the rules had been printed under the frozen scales by the recount (journal 17:40 CST); it redefines s_P(t), the input of every rule. By this repo's rule 5 it is an exploratory deviation, not a rule reading; the pre-registered outcome of this spec is the frozen verdict's: instrument artifact, rules not read. The amendment's grid, figure footer and verdict file carry the label "amendment 5b"; read them as exploratory.
2. §6's registered control, the layer's median |ā| at every checkpoint (the global scale), was not deliverable by the verdict code at the freeze (`available: False`); the spec should have been amended before the tag. Found by the process check (`results/sanity/SANITY_process.md` item 4).
3. §1 defines the activation-based birth by half its final value; §2 and the verdict use half the maximum over valid checkpoints. The two give the same D3 here (`results/dating/explore_scale_variants.json`).
4. §0 and the write-ups called UNITE's dating "the top-50 dating" and the birth "the cluster entering the top 50". UNITE's birth is the first checkpoint at which the cluster's median activation on its exemplars reaches the pole's corpus 99.9th percentile and does so again at the next checkpoint (`results/sanity/SANITY_science.md` item 1). The rule puts J under that line at t_b−1 by construction.
5. The hand-written CST times in this file and in the journal are approximate (0 to 57 minutes after the git commit they describe); the git commit and tag times are authoritative for the order of edits and freezes. Two "before freeze" stamps (16:20, 16:25 CST) and the amendment's "17:50 CST" fall after their tags by the clock but before them in git.
6. No literature search was recorded before this spec's compute; the next spec's Phase 0 records one.
