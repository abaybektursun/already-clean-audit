# SPEC8 — does a neuron's second cluster form at an event or by drift? A confirmatory set, a threshold-free dating, and a scale that is not the pole's own

Status: drafted 2026-11-02 21:50 CST (coordinator); pre-freeze amendments of 22:15 CST below (§2, §3, §5, §9, from the builder's notes `results/review8/BUILD_NOTES_birthobject.md`); frozen at tag `reuse-freeze-8` before any forward; the lines of §6 are set on the 240 cases SPEC5 read (control data), and every rule is confirmed on cases never read. Parents: `SPEC5_dating.md` with its post-freeze notes, `results/REVIEW_dating.md`, `results/sanity/SANITY_science.md` items 1 and 2, `results/review6/LIT_dating.md` (Phase 0), `NEXT.md` item 7. Model `allenai/Olmo-3-1025-7B`, the 16 public checkpoints T_0 to T_15 of the UNITE grid.

## 0. What is known, in the field's words

UNITE dates a cluster's birth by a threshold crossing: the first checkpoint at which the cluster's median activation on its exemplars reaches the pole's corpus 99.9th percentile at t and t+1 (`results/sanity/SANITY_science.md` item 1). Schaeffer, Miranda and Koyejo 2023 (arXiv 2304.15004) show that a discontinuous metric produces an apparent event on a smooth quantity; SPEC5 read a continuous quantity at that date and found no gain beyond the grid-wide trend on its confirmation set (J/O* gain into the birth over the case's gain elsewhere 0.97 [0.85, 1.06], Result 34), and its event-or-drift rules depended on the scale statistic. SAE-Track (Xu, Wang and Wang 2024, arXiv 2412.17626 §4.2), in the same backward design, reports that "most features undergo predominantly a gradual formation process"; Quirke et al. 2023 (arXiv 2311.00863) report one neuron's late function growing gradually. Three observations would refute the drift reading (Michaud et al. 2023 arXiv 2303.13506; Kangaslahti et al. 2025 arXiv 2506.15872; Chen et al. 2024 arXiv 2309.07311): single contexts jumping at nearly the same time while their mean ramps; a threshold-free break on the scale-free quantity landing at the dated birth more often than chance; a jump that survives a continuous x-axis. No paper measured a neuron's second cluster across checkpoints.

## 1. Question, decision, hypotheses

**Question.** On cases never read, does the pole's activation on its newer cluster, relative to its older cluster (scale-free) and relative to the layer's typical activation (a scale that is not the pole's own), form at an event that the dated birth marks, or by drift?

**Decision.** Event at the birth: Part A's question ("what arrives") is asked again on the event checkpoints with SPEC4's instrument on these cases. Drift, or events not at the birth: the program stops asking what arrives at a birth; the dated birth is kept as a rank event and the REUSE results are stated as measurements at it; the next instrument is the per-context developmental curve.

| hypothesis | statement | refuted when |
|---|---|---|
| H_signature | The threshold-free break of J/O* (Chen et al.'s break(f, Δ)) lands within one grid step of t_b more often than shuffled dates allow | the observed share's interval lies at or below the shuffled null's 95th percentile (E1) |
| H_context-event | The single contexts behind J's mean jump at nearly the same checkpoint (an event hidden by the mean), more than the same pole's other contexts do | J's concentration (the share of its contexts whose jump time is within one grid step of the case's median jump time) is not above the same pole's mixed J and O* contexts' concentration by its interval (E2) |
| H_old-decline-global (boring) | Under the layer's scale, J's level does not rise across the two steps into the birth while O*'s falls | J's two-step ratio under the global scale at or above 1.2 by its interval, or O*'s at or above 0.8 (E3; SPEC5's D2 under a scale that is not the pole's) |
| H_drift | J/O* rises without an event: the largest adjacent log-rate of J/O* per unit of log-tokens is not concentrated at any checkpoint | the share of cases whose largest rate falls at the modal checkpoint exceeds the shuffled null's 95th percentile (E4; Chen et al.'s continuous x-axis) |
| Instrument artifact | | any gate of §5 fails |

## 2. The measurement

For each case (pole P with sign, new cluster J, older cluster O*, UNITE birth t_b) and each grid checkpoint t: a_P(x, t) at the marked token of every context x of J and of O* (6 to 12 each, SPEC5's capture, 256-token window, fp32 g and u, a_P in float64); s_P(t), the median of a_P over P's 50 scale contexts captured with the same window (SPEC5 amendment 5b's stage); the global scale g_l(t), the median over the layer's neurons of |ā| at t over 200 corpus positions (SPEC4's `abar.npz` construction at every load); the token count of each checkpoint from the model card (steps × batch × sequence, a constant factor, so log-tokens is log-steps up to a constant). Quantities, per case: J/O*(t) = mean a_P over J / mean a_P over O* on pairs where mean a_P over O* > 0 and s_P(t) > 0.01 s_P(T_15) (SPEC5's validity); r_J^g(t) = mean a_P over J / g_l(t) and r_O^g likewise; per context x in J, c_x(t) = a_P(x, t) / g_l(t). The break: break(f, Δ=1) = argmax over t of [f(t+1) − f(t)] − [f(t) − f(t−1)] on f = log J/O* over valid t (Chen et al. 2024 eq. 2). The jump time of a context: argmax over valid adjacent pairs of log c_x(t+1) − log c_x(t) with c_x(t) > 0. The log-rate of J/O* at a pair: [log J/O*(t+1) − log J/O*(t)] / [log steps(t+1) − log steps(t)], pairs from T_1 on. *Amended 22:15 CST before freeze:* the model card states the stage-1 token total (5.93 T) but not the tokens per step, so the rate is per log-step (`infra/pg/olmo3_tokens.json` holds the steps and the card's lines); it equals the per-log-token rate only if tokens per step are constant, a limit.

## 3. Design

A third case set: jobs of the UNITE harvest whose pole's neuron is in neither SPEC4's primary set (104 neurons) nor its fresh set (59), dated by the same rule, with their J and O* contexts and scale contexts as `build_items_dating.py` builds them. *Amended 22:15 CST before freeze:* the harvest's REUSE pool holds exactly those 163 neurons, so the set is built from the harvest with UNITE's own filters plus |J| ≥ 6, |O*| ≥ 6, no shared marked token and L ≤ 30 (the ā pass covers layers 0 to 30), dropping SPEC4's L ≥ 4 and shared-document filters, which served its ablation: 127 cases on 65 neurons (80 signed poles), births T_4 to T_13, terciles 74/19/34 (`inputs/birthobject_items.json`, sha256 `f31c40420c0b…`, manifest `inputs/manifest_birthobject_7915654acb99.json`). It differs from the first two sets: 68 cases are at layers 0 to 3, where those sets had none, and in 76 cases J and O* share a document. Every rule is read on the whole set and reported on the strata (layers 0 to 3 against 4 to 30; shared document against not), and the §6 lines were set on cases from layers 4 to 31, which is stated as a limit. Two AWS stages per load as SPEC5 (dating captures; scale captures) plus the ā pass (200 positions) at every load; 16 loads, one pipeline. Live rule: a case is live when s_P(t_b) > 0.01 s_P(T_15) and mean a_P over O* at t_b > 0 (no ablation is run). Discovery and confirmation: the §6 lines are set on SPEC5's 240 cases (`results/dating/explore_jo.json`, `explore_scale_variants.json`, the bank), which are the control data; the rules are read once on the new set.

## 4. Data

Items `inputs/birthobject_items.json` with sha256; the harvest banks as SPEC5 (`positions.npz`, `poles.json`); the token-count table from the model card, banked with its source line. The corpus for ā: UNITE's e1 positions at each checkpoint.

## 5. Gates (each with its failing input and identity value)

- Known number per load: the pole's capture against the harvest's exemplar activation at T_15 (median relative difference over poles at most 0.05; SPEC5 read 1.3%; failing input: the previous checkpoint's bank). Capture identity per load (20 contexts twice, 0; failing input: the hook on layer L−1). Completeness as SPEC5.
- Global scale: g_l(T_15) against SPEC4's T_15 per-layer medians of |ā| (`birth/1/t15/background.json`, the tracked record of `abar.npz`; *amended 22:15 CST*), median relative difference over layers at most 0.05 (failing input: SPEC4's T_13 record); g_l(t) > 0 at every load.
- Scale: s_P(T_15) > 0 for every case; validity shares from T_9 on at least 0.90 (*amended 22:15 CST before freeze:* set on this set's harvest-bank shares, 0.945 at T_9 and 0.937 at T_10, under SPEC5's 0.95 line because of the layer-0 poles; the shares at every checkpoint are reported); the scale stage's T_15 medians against the dating stage's T_15 scale capture to 1e-5 per pole (SPEC5 amendment 5b's construction); failing input: the next pole's.
- Known number: the job's per-load check against the harvest's exemplar activations, and the T_15 scale medians against the harvest bank's within 0.05 (median over poles); failing input: the T_14 bank in place of T_15.
- Null power: the shuffled-dates null of E1 and E4 has a 95th percentile below 0.5 (room for the share to exceed it); failing input: the observed dates shuffled, which must land inside the null.
- A failed gate stamps every rule "not read".

## 6. Frozen rules (lines set on SPEC5's 240 cases before any forward; the control values, `results/dating/explore_birthobject_lines.json` by `scripts/review/explore_birthobject_lines.py`, seed 1, with the pole's own scale standing in for the global scale: E1 share 0.275 [0.197, 0.353] and 0.341 [0.236, 0.444] against shuffled-dates 95th percentiles 0.268 and 0.330; E2 concentration of J 0.426 [0.387, 0.463] and 0.466 [0.407, 0.530] against the same pole's mixed set 0.453 and 0.401; E4 share at the modal checkpoint 0.128 [0.073, 0.184] and 0.132 [0.069, 0.204] against 0.168 and 0.198; every rule has room to fire in either direction)

- **E1.** Share of live cases with |break(log J/O*, 1) − t_b| ≤ 1, interval by bootstrap over neurons (2,000 draws, seed 0); the null: the same share with each case's t_b replaced by another live case's t_b (2,000 shuffles), its 95th percentile. Interval above the null's 95th percentile → H_signature not refuted (and the Decision's event branch); at or below → H_signature refuted.
- **E2.** For each live case with at least 6 contexts in J and 6 in O* whose c_x has a valid jump: the concentration of J, the share of J's contexts whose jump time lies within one grid step of J's median jump time; the same-pole control, the concentration of a set of 3 J and 3 O* contexts of the same case (seeded draw); both as means over cases with bootstrap intervals over neurons (2,000 draws, seed 0), and their paired difference. The difference's interval above 0 → H_context-event not refuted; otherwise refuted. Reported: the old cluster's own concentration, and the null of contexts drawn from six different cases of the same tercile (which the pole's shared trajectory makes a wrong null: on SPEC5's cases J's concentration 0.43 and 0.47 exceeds it, 0.35, and equals the same pole's old cluster's, 0.42 and 0.45, and the mixed set's, 0.45 and 0.40).
- **E3.** r_J^g(t_b)/r_J^g(t_b−2) and r_O^g(t_b)/r_O^g(t_b−2), medians with intervals over live cases: J's interval at or above 1.2 or O*'s at or above 0.8 → H_old-decline-global refuted; J's below 1.2 and O*'s below 0.8 → not refuted; otherwise not separated (SPEC5 §6 D2's form).
- **E4.** The checkpoint of each live case's largest log-rate of J/O* per log-token; the share at the modal checkpoint, with interval; the null: the share at the mode under shuffled pair labels within case, 95th percentile. Share above the null's 95th percentile → H_drift refuted (events concentrate); otherwise H_drift not refuted. Reported: the modal checkpoint and whether it is t_b.
- **Reported, no rule:** J/O* curves aligned to t_b (SPEC5's figure on the new set); the two-step gain into t_b over the case's gain elsewhere (SPEC5's 0.97 on its fresh set, as the known comparison); D1 to D3 of SPEC5 under the median, mean and global scales, side by side; per-context curves for six cases in the samples.

## 7. Point predictions, before any forward (the coordinator's, from SPEC5 and the literature)

| quantity | predicted | refuted if |
|---|---|---|
| E1 share at the birth; shuffled null 95th percentile | 0.30 [0.22, 0.38]; 0.30 | share outside [0.05, 0.55] |
| E2 concentration of J; of the same pole's mixed set | 0.44 [0.38, 0.50]; 0.43 | outside [0.25, 0.65] |
| E3 J ratio; O* ratio under the global scale | 1.15 [1.0, 1.35]; 0.85 [0.7, 1.0] | outside [0.7, 2.0]; outside [0.4, 1.3] |
| E4 share at the modal checkpoint; null 95th percentile | 0.13 [0.07, 0.20]; 0.18 | outside [0.05, 0.5] |
| two-step gain into t_b over the case's gain elsewhere | 1.0 [0.9, 1.1] | outside [0.7, 1.5] |
| GPU hours (16 loads, two stages and the ā pass) | 1.8 [1.3, 2.6] | over 3.5 |

The author expects drift (E1 and E4 at chance, E2 at the same-pole control, as on SPEC5's cases), with E3 reading the old cluster's decline under the global scale as SPEC5's median scale did.

## 8. Threats

| threat | effect | mitigation |
|---|---|---|
| Selection built into the birth rule (J crosses the pole's 99.9th percentile at t_b) | a rise at t_b by construction | E1 and E4 compare against shuffled dates and shuffled pairs; the gain over the case's own gain elsewhere is reported; the threat is stated as a limit |
| Early near-zero scales (SPEC5's T_0 to T_4) | degenerate ratios | the validity rule; the global scale g_l(t) has no per-pole zero; per-context jumps require c_x(t) > 0 |
| The grid's late gaps (up to 440,000 steps) | an event inside one gap reads as one pair | E4 reads rates per log-token; stated as a limit |
| Few contexts per cluster (6 to 12) | E2 underpowered | the null is built with the same counts; cases under 6 contexts excluded from E2 and counted |
| The third set differs from the first two (later births, other layers) | the lines set on SPEC5's cases do not transfer | the set's birth and layer distributions reported against the first two; the lines are shares and intervals, not absolute levels |

## 9. Runbook

Worker (Opus, own worktree): the items builder for the third set; the ā pass added to the dating job's scale stage (`--stage dating+scale --abar`, one model load per checkpoint; the scale stage writes under `t<nn>/scale_stage/`); `scripts/verdict_birthobject.py` with five failing inputs (`known_shifted`, `capture_wrong_layer`, `scale_other_pole`, `global_other_checkpoint`, `dates_observed_shuffled`); the dry run in its own module `jobs/reuse_dryrun_birthobject.py`. *Amended 22:15 CST before freeze:* the pipeline runs T_15 first, so that a scale-gate failure shows after one load, then T_0 to T_14. Coordinator: the §6 lines' values on SPEC5's cases written in the journal, freeze (tag `reuse-freeze-8`), launch on AWS (one pipeline, 16 loads, budget 200 min, fresh prefix `results/reuse/birthobject/1`), samples opened (six cases' per-context curves) before any number, §7 restated, verdict once, failing inputs, recount under seed 1, a different model's read, figures, the register. Cost about 1.8 GPU-h on the AWS lane, a further recorded deviation from the study's cap; API $0. SPEC8 runs no ablation, so it launches beside SPEC6 (*amended 22:15 CST*).

## 10. Grid, filled once

| rule | reading | outcome |
|---|---|---|
| E1 | | |
| E2 | | |
| E3 | | |
| E4 | | |
| gates | | |
| GPU hours | | |

## 11. Literature

The five citations of `results/review6/LIT_dating.md` §6 (Quirke et al. 2023; Xu, Wang and Wang 2024; Ge et al. 2025 arXiv 2509.17196; Schaeffer et al. 2023; Chen et al. 2024), each read for the quoted claim; Moskvoretskii et al. 2026 (arXiv 2605.13329) for the per-layer scale on this model's checkpoints; Xu 2026 (arXiv 2606.02378) for the per-checkpoint null and the robust quantile; Michaud et al. 2023 and Kangaslahti et al. 2025 for the per-sample jump.
