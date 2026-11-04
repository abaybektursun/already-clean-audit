# REUSE: does a neuron's new job run through the old job's upstream inputs? An ablation test on OLMo-3 checkpoints

Status: **frozen 2026-11-01**. Freeze 1 is the commit that first adds this file with this status line, tag `reuse-freeze-1`. Amendment 1, 2026-11-01, §13, tag `reuse-amendment-1`: the tasks between the freeze and a closed result, two rows the grid lacked, and the limits that travel with the result; no rule, threshold, item or arm changed. Amendment 2, 2026-11-01, `SPEC2.md`, tag `reuse-amendment-2`: the pre-publication checks (birth-checkpoint planted reference, activity-matched control, ceilings at T_15, mixture and convergence tests, judge labels) with their own rules and point predictions; nothing here changed. Every value below was set from the sniper jobs S1–S3 and the planted and ceiling references only; no primary job has been forwarded. Written 2026-11-01 by the coordinator (Fable 5.1) after the UNITE review (`../../../interp_research-unite/studies/unite/results/REVIEW_2026-11-01.md`). Rules that changed between the draft and the freeze are marked where they stand, with the draft rule and its outcome kept. Parents: UNITE `PREREG.md`, `RESULTS.md`, `REVIEW_2026-11-01.md`; the essay "Neural Networks Are Already Clean"; Eliwat's recgen-checkpoints (Pythia-410m), which this study credits as the first input-side test.

Scope rule from the user, 2026-11-01: small volume, key points only. One decisive experiment, one primary set of 150 jobs, three rules, no secondary arms unless they are free.

## 0. What UNITE established and why this study exists

UNITE measured T, a Gram overlap between a new job's upstream attribution and the older job's. Every number reproduced. The review found that T is the same under another neuron's input rows (direct readout: T_foreign 0.332 against T 0.339), so T measures how alike the two jobs' contexts are in upstream-activation space, not whether this neuron reads them through the same inputs. It also found that 209 of 841 pairs share a document and those pairs carry T 0.56 against 0.285 for the rest. Correlational overlap cannot separate "the neuron reuses its inputs" from "the contexts are alike". An intervention can: remove the old job's drivers and see whether the new job's activation falls.

## 1. Question, decision, hypotheses

**Question.** When training gives a pole P (one sign of one MLP neuron of `allenai/Olmo-3-1025-7B`) a new job J (a cluster of its top-50 contexts, dated backward from T_15 as in UNITE), does P's activation on J at J's birth depend on the same upstream MLP neurons that drive P on its older job O*?

**Decision.** If the answer is yes and specific (R1 not refuted, R2 excludes 0): the essay's growth claim has causal support on a 7B SwiGLU model at top-50 activation, and the next study goes to attention heads as drivers. If no, or not specific: the input-reuse reading of growth fails on O7 at top-50, and the next study asks whether what is shared is the output side instead.

| hypothesis | statement | refuted when |
|---|---|---|
| H_reuse | Removing O*'s top-m drivers lowers P's activation on J by a large fraction of what removing J's own drivers does | The 95% interval of median I over jobs overlaps the interval of the planted median I (R1) |
| H_generic (boring) | The fall comes from upstream neurons every neuron depends on, not from P's wiring | The interval of median Δ(D(O*); J) − Δ(D_foreign; J) excludes 0 and is positive (R2). If it includes 0, H_reuse is reported as "not specific" whatever R1 says |
| H_consolidate | Reuse grows over training: I at T_15 exceeds I at birth on the same jobs | The interval of median (I_T15 − I_birth) includes 0 or is below 0 (R3) |
| Instrument artifact | The fall is from hook placement, bf16 noise or a selection effect | Any gate in §5 fails |

A low I does not prove packing: one broad feature built from separate detectors also shows separate drivers (Eliwat v23). Every claim is at top-50 activation and about upstream MLP neurons only; attention heads are not ablated.

## 2. The measurement

**Forward.** O7 at revision `stage1-step<n>`, bf16 weights, fp32 arithmetic in the hook, BOS prepended, forward stopped after P's layer L (`unite_route.upto_layer`), 256-token window ending at the marked token, as in UNITE.

**Ablation.** For a set S of upstream neurons k at layers l < L, the MLP activation `a_k(t) = silu(g_k(t)) u_k(t)` is replaced at every position t of the window by its background mean `ā_k`, the mean over the 200 centring positions of UNITE's set e1 at that checkpoint (`unite_route.background_positions`). This removes k's effect through the direct path and through attention. **Frozen on S1 (2026-11-01): mean replacement.** ā_k was available at T_15 and T_10; zero replacement gave an own fall of 0.542 against 0.548 for the mean at m 50 direct (`results/sniper/1/s1/metrics.json → summary`).

**Readout.** `a_P(x)`, P's signed activation at the marked token of context x, σ applied so that top contexts are positive. `Δ(S; x) = (a_P(x) − a_P^S(x)) / a_P(x)`. For a job, `Δ(S; J) = mean over x ∈ J`. Absolute `a_P(x)` and `a_P^S(x)` are banked beside every Δ.

**Drivers.** `D(J)` = the m upstream neurons with the largest `Σ_{x∈J} A_k(x)` at that checkpoint. Two rankings exist: direct attribution `A_k` (UNITE §2, MLP path only) and gradient × activation `a_k ∂a_P/∂a_k` summed over positions (UNITE gradient readout, all paths).

**Ranking, frozen on S1: direct.** The rule written before S1 said "the ranking with the larger median own fall", and S1 would have picked gradient × activation: own fall 0.470 against 0.356 at m 20. The rule was changed before freeze, on the null arms alone, because R2 needs the null arms low: under gradient ranking the random-matched fall is 0.055 at m 20 and 0.091 at m 50, against 0.002 and 0.030 under direct, and the foreign fall at m 50 is 0.187 against 0.120. Gradient ranking also cancelled 1 of 6 jobs (L27N8796+:c2) and gave one T_10 own fall of 0.075 against 0.203 under direct. Drivers chosen by the direct path still remove every path when ablated, so the readout keeps attention-routed effects; only the selection is direct. The original rule and its outcome stay here.

**m, frozen on S3: 50.** The rule: the smallest of 20 and 50 whose median own fall on the sniper jobs is at least 0.25 (S1, direct: 0.356 at 20, 0.548 at 50) and whose room on the S3 references is at least max(0.20, m_r). At m 20 room is 0.169 against a line of 0.21 (m_r 0.21–0.22 by seed), so m 20 fails; at m 50 room is 0.329 against 0.200 (m_r 0.10). Values recomputed by the coordinator from `results/sniper/2/t15/metrics.json`. The draft had provisionally named m 20 on S1's own-fall numbers alone.

**The statistic.** `I(J) = Δ(D(O*); J) / Δ(D(J); J)`, read at J's birth checkpoint and at T_15. Also `s(J) = |D(J) ∩ D(O*)| / m`, the sparse overlap share, reported beside I.

**Arms per job and checkpoint**, 5 forwards per context: identity (S empty), own `D(J)`, cross `D(O*)`, random-matched R, foreign `D_foreign`. No more.

- **R**: for each k in D(O*), the neuron of the same layer, not in D(J) ∪ D(O*), nearest in background mean |a| to k. Count- and energy-matched by construction. One draw; deterministic, so no seed.
- **D_foreign**: Q = the pole in the item set with P's layer and sign and the nearest ‖G‖² + ‖U‖² at T_15, P excluded; D_foreign = top-m drivers of Q's largest job on Q's own activation at the same checkpoint. Tests whether O*'s drivers are special or any busy neuron's drivers would do.
- **Cancel.** A job is cancelled at a checkpoint when `Δ(D(J); J) < 2 |Δ(R; J)|` (no room to read I) or fewer than 3 contexts have `a_P(x) > 0`. Cancelled jobs are counted and reported.

## 3. References

- **Ceiling.** UNITE's token- and document-disjoint split of O* (`ceiling` in the items). `I_ceil = Δ(D(side 1); side 2) / Δ(D(side 2); side 2)` at the job's birth checkpoint. Same job, so I should be high.
- **Planted packed pole.** UNITE's 56 kept pairs at T_15 (`planted_contexts.json`), `a_plant = a_A + λ a_B`, sides J (A's term larger) and O (the rest). `I_plant = Δ(D(O); J) / Δ(D(J); J)` on `a_plant`. Two neurons with separate inputs, so I should be near 0. T_15 only; stated as a limit at every birth-checkpoint number. Planted I is count-matched by subsampling sides to the primary set's |J| and |O*| (200 draws, seed 0, median), as in UNITE.
- **Planted band.** Per depth tercile (L 0–10, 11–21, 22–31), the 5th–95th percentile of count-matched planted I. Reported, not deciding: the share of live jobs inside the band, corrected by the ceiling share as UNITE's p. No rule attaches to it.
- **Frozen reference values, S3 at T_15, m 50 direct** (`results/sniper/2/t15/metrics.json`, `results/sniper/S3.json`): planted I medians by tercile 0.038, 0.273, 0.211 on 18, 20, 18 pairs; planted median weighted by the primary set's tercile counts (50, 50, 50) **0.174 [0.116, 0.223]**; bands [−0.188, 0.183], [0.014, 0.505], [0.024, 0.296]; 95th percentile over all pairs 0.459. Ceiling I over the 65 setting-set splits (60 neurons) **0.788 [0.700, 0.856]**; over the 50 splits that are not also a primary job's ceiling 0.772. Count-matching ran on the instance with 40 draws per pair (200 would have needed about 270,000 forwards); it moves the planted median little, since 39 of 56 pairs return their full-side value. The planted pairs at layer 15 carry the upper tail of the band. The planted reference exists at T_15 only; this limit stands beside every birth-checkpoint number.

## 4. Items

From UNITE's 841 primary jobs (`inputs/7076e7e5407b/j4c_items.json`, sha256 `96d8a72a…eea9a62`): keep jobs whose J and O* share no document (`doc_offsets.npy`) and no marked token, with |J| ≥ 6, |O*| ≥ 6, L ≥ 4. The pool, computed 2026-11-01 (`../../../interp_research-unite/studies/unite/results/reuse_candidates.json`): 312 jobs, 87 / 110 / 115 by depth tercile, of which 54 / 86 / 84 have a ceiling split of O*. J is capped at its first 12 contexts by rank. **Draw 50 per depth tercile with seed 0 from the pool jobs whose O* has a ceiling split**, so every primary job carries its own ceiling; the 6 sniper jobs (L09N7823+:c3, L07N7004-:c0, L21N3982+:c2, L12N8337-:c1, L27N8796+:c2, L27N7794+:c4) are excluded first. The pool jobs with a split that are not drawn form the **setting set**: their ceilings set the room gate in S3 and never enter the primary read. **Drawn 2026-11-01** (`inputs/manifest_b972b00ceff4.json`, `inputs/primary_items.json`, `inputs/setting_items.json`, uploaded to `s3://<s3-bucket>/results/reuse/inputs/b972b00ceff4/`): primary 50 / 50 / 50 from 52 / 84 / 82 candidates; labels **57 related, 93 unrelated** (the draft expected the reverse; the pool with a split holds 94 related and 130 unrelated, and the label reader reproduces the inventory's 374 / 467 on all 841); setting set 74 jobs including the 6 sniper jobs, 65 distinct splits on 60 neurons, of which 15 splits are also a primary job's ceiling because jobs sharing a pole and O* share one split. Room is recorded both ways in §3 and §5. **Primary set: 150 jobs, two checkpoints each, birth and T_15.** Counts, hashes and the pool size are written to `inputs/manifest_<sha12>.json` before freeze. Planted: all 56 kept pairs.

**Relatedness labels exist for every job**, so no new labels are bought: the clusterer's sample-0 pair marks label each J against its own O* as related (374 of 841) or unrelated (467), and sample 1 agrees on 586 (`../../../interp_research-unite/studies/unite/results/labels_inventory.md`). The label is read from the items at freeze and never from the falls. UNITE's `flagged` is a different quantity (judge decoy test) and is not used.

No discovery/confirmation split: the set is read once, and nothing is tuned on it. Thresholds come from S1–S2 (6 jobs, not in the primary set) and from the planted and ceiling references.

## 5. Gates, each with its failing input

| gate | passes when | failing input, must fail | on failure |
|---|---|---|---|
| Known number | UNITE's gate reused at each load: median `|a_ours − a_bank| / max|a_bank|` ≤ 1e-2 on 16 contexts | previous checkpoint's bank | one debug on a fresh prefix, then the load does not count |
| Identity | S empty gives `|Δ| ≤ 1e-3` on every context | own drivers | Arm cancelled |
| Positive control | median own fall at m 50 ≥ 0.25 on the primary set at birth. **Frozen on S1–S3:** own 0.548 at T_15 and 0.610 at T_10 (m 50 direct, 6 jobs); random 0.030 and 0.005, under half of own | random-matched R: median fall under half of own | study stops: no room |
| Room | median `I_ceil` − 95th percentile of planted I ≥ max(0.20, m_r). **Frozen on S3 at m 50 direct, T_15: room 0.329 (0.313 on the 50 clean splits), m_r 0.10, line 0.200. Passes.** The failing input, planted side labels shuffled within pair, gives a median shuffled I of 0.72 and a room of −0.20, so it fails as it must | planted side labels shuffled within pair | study stops: cannot separate same-job from packed |
| Cancel share | under 40% of primary jobs cancelled at birth | sets of 7 contexts from no job | study stops |

Each gate's three numbers travel with it: the value, the failing input's value, and the identity value.

## 6. Rules, frozen before the primary run

Bootstrap over neurons, 2,000 draws, seed 0, 95% percentile intervals; planted over pairs within tercile, weighted by the primary set's tercile job counts, as UNITE §13.

- **R1 H_reuse.** Median I at birth over live jobs against the planted median I. Overlap: refuted. No overlap and real higher: not refuted. No overlap and real lower: its own outcome, does not support H_reuse.
- **R2 H_generic.** Median over live jobs of `Δ(D(O*); J) − Δ(D_foreign; J)`. Interval excludes 0 and is positive: the cross effect is specific to O*'s drivers. Otherwise: not specific, and R1 is reported as uninterpretable about reuse.
- **R3 H_consolidate.** Median over jobs live at both checkpoints of `I_T15 − I_birth`. Above 0: consolidation; includes 0: no change; below 0: separation.
- **R4 H_related-share.** Related new jobs reuse the old job's drivers more than unrelated ones: median I at birth over related jobs minus median over unrelated jobs, bootstrap over neurons. Interval above 0: holds. Includes 0 or below: refuted. This is the O7 counterpart of Eliwat's "unrelated facets share no drivers at formation" and costs no forward. Labels per §4; about 90 related and 60 unrelated jobs are expected in the 150, read from the pool before the draw and written to the manifest.
- **Reported, no rule.** `s(J)` against planted and ceiling s (UNITE's banked top-20 gave medians 0.15 real, 0.35 ceiling, 0.00 planted at m 20, `driver_overlap.md`); the share of jobs inside the planted band and its ceiling-corrected form; I by tercile; absolute falls per arm; the position `(median I − planted median) / (median I_ceil − planted median)`; the linearity check of §7; I for unrelated jobs alone against the planted median.

A rule found wrong after the primary run gets a dated exploratory note; the frozen rule still decides.

## 7. Instrument checks in the sniper jobs

- **S1, T_15, 6 jobs (2 per tercile, filters of §4, not in the primary set), 3 planted pairs, 3 ceiling splits.** Arms: identity; own and cross at m 20, 50, 100 under both rankings; R at each m; D_foreign at m 50 under both rankings; zero replacement for own at m 50 direct. About 1,500 forwards of 257 tokens plus one load. Estimate 0.25 GPU-h, budget 60 min.
- **S2, T_10, the same jobs**, arms identity, own and cross at m 50 under both rankings, R, D_foreign. Checks the instrument at an early checkpoint where activations are small. Estimate 0.15 GPU-h. S1 and S2 run as one pipeline on one instance.
- **Linearity, reported.** For each ablated set and context, `Σ_{k∈S} A_k(x)` (direct attribution) against the observed change in P's pre-activations g and u and in a_P. Slope and R² over all pairs. If the direct sum predicts the fall poorly, that is itself a UNITE postmortem number: the Gram readout was missing the paths the ablation removes.

**S1 and S2 ran on 2026-11-01**, instance <instance-id>, prefix `results/reuse/sniper/1`, 0.146 GPU-h, `STATUS exit=0` (`results/sniper/SNIPER.md`, `results/sniper/1/s{1,2}/metrics.json`). Gates: known number 0.0 at T_15 and 7.7e-4 at T_10 against failing inputs 0.195 and 0.147; completeness 1.0e-3 and 1.3e-3; identity max |Δ| exactly 0.0 on 204 and 50 contexts while own drivers moved every context. Direct ranking, 6 jobs, medians: T_15 m 20 own 0.356, cross 0.154, random 0.002, I 0.52; m 50 own 0.548, cross 0.326, random 0.030, foreign 0.120, I 0.66; T_10 m 50 own 0.610, cross 0.276, random 0.005, foreign 0.016, I 0.46. References at m 50 direct, T_15: planted I 0.032, 0.292, 0.287 (shuffled labels 0.89, 1.58, 0.70); ceiling I 0.950, 0.996, 0.756. Linearity: the direct attribution sum explains the fall with slope 0.61 and R² 0.79 (scale-free slope 0.50, R² 0.83), so about half of each fall runs through paths the direct Gram did not see. Samples opened: `s1/samples/arms.jsonl` holds the forwarded prompt, marked token, a_P before and after, and the five largest ablated drivers per arm.

- **S3, T_15 and T_10, references at the candidate settings.** T_15: all 56 planted pairs and the ceiling splits of the setting set, arms identity, own, cross, random at m 20 and m 50 direct, planted labels shuffled at both m. T_10: the 6 sniper jobs at m 20 direct, arms identity, own, cross, random, foreign. Sides capped at 12 contexts by rank.

**S3 ran on 2026-11-01**, prefix `results/reuse/sniper/2`, launch 08:13:42Z to `STATUS exit=0` 08:36:22Z, **0.378 GPU-h** against the 0.2 estimate, because count-matching ran on the instance (66,216 forwards at T_15). Gates: known number 0.0 and 7.7e-4 against 0.195 and 0.147; completeness 1.0e-3 and 1.3e-3; identity exactly 0.0 on 1,803 and 50 contexts while own drivers moved 1,802 of 1,803 and 50 of 50. Reference values in §3; room in §5. T_10, m 20 direct, 6 jobs: own 0.388, cross 0.184, random 0.014, foreign 0.020, I 0.46, none cancelled. The coordinator recomputed the planted medians, bands, ceiling medians, room and m_r from `2/t15/metrics.json` under another seed; all agree with `S3.md` within 0.01.
- **Samples.** Per arm and job: the prompt as forwarded, the marked token, a_P before and after, the five ablated neurons with the largest |A_k| with their a_k. At least the job with the highest I and the one with the lowest per tercile. A number without these is not quotable.

Freeze decisions after S1–S2: m, ranking, replacement, the positive-control floor, the room margin. Each is written here with the S1 value that set it, then committed.

## 8. Threats

| threat | effect | mitigation |
|---|---|---|
| Shared documents between J and O* | fall inflated by document topic | items are document-disjoint |
| Generic high-activity upstream neurons | cross fall not about O* | D_foreign (R2) and energy-matched R |
| D(J) and D(O*) overlap by construction when m is large | I inflated | s reported; m is the smallest with room |
| Mean replacement moves the model off-distribution | fall from distribution shift | identity gate; R arm has the same shift at matched energy |
| Birth misdated, grid gaps | I read at the wrong checkpoint | births at T_7 and T_15 excluded already (≤ T_13); I at T_15 reported too |
| Planted reference at T_15 only | band shifts over training | stated beside every birth number; no fix in budget |
| Attention heads carry the job | drivers miss them | stated scope; gradient ranking counts attention-routed MLP paths |
| Few contexts per job | noisy Δ | cap 12, floor 6; cancel rule |

## 9. Runbook

1. Offline gate: `python -m infra.aws_pg.jobs.reuse_dryrun --out <scratch>` on tiny random configs; identity exact; own fall positive; samples written. Items script prints §4 counts or stops.
2. S1 + S2 pipeline, `--detach`, fresh prefix `results/reuse/sniper/<n>`, poll `metrics.json`. Journal starts here. Open samples before any number.
3. Freeze 1: fill every TO FREEZE, commit `SPEC.md`, `inputs/manifest_*.json`, journal, code. Named files only.
4. Primary run: the 150 primary jobs at birth and at T_15, arms identity, own, cross, random, foreign at m 50 direct, plus each job's own ceiling split at its birth checkpoint (identity, own, cross). Planted references are S3's and are not rerun. Births T_4–T_13 plus T_15: about 11 loads at 0.06 GPU-h and about 23,000 forwards. Estimate 1.0–1.5 GPU-h, budget 150 min, one or two pipelines. `verdict` runs once, offline, on the banked falls, at the freeze commit.
5. Results written after a different model reads the artifacts. Nothing is pushed or merged.

Stop rules: total use 5 GPU-h; a job over twice its estimate with no output advancing is killed and journaled; a failed gate gets one debug and never a relaxed threshold.

## 10. Artifacts

Per step: `metrics.json` (gates, per-arm falls per job, cancel flags), `falls/<pole>.npz` (per context and arm: a_P before, after, g, u, the ablated set), `samples/arms.jsonl`, `background.json`. Local: `studies/reuse/JOURNAL.md`, `results/sniper/`, `results/primary/`, `results/RESULTS.md`. Code, all NEW: `infra/pg/reuse_ablate.py`, `infra/aws_pg/jobs/reuse_ablate.py`, `jobs/reuse_dryrun.py`, `steps/reuse_*.json`, `studies/reuse/scripts/build_items.py`. REUSED, read-only: `unite_route` (attribution, background positions, upto_layer, Rows), `unite_stats` (bootstrap), `neurons_harvest` (MLPTap, forward, load).

## 11. Prediction grid, filled once

| prediction | predicted | observed | verdict |
|---|---|---|---|
| R1 H_reuse | no point registered | | |
| R2 H_generic | no point registered | | |
| R3 H_consolidate | no point registered | | |
| Positive control floor | ≥ 0.25 at frozen m | | |
| Room | ≥ 0.20 | | |

Coordinator's expectation, no rule attached, written before any sniper number: own fall at m 50 near 0.3–0.5; planted I near 0.1; ceiling I near 0.6–0.8; real I between, nearer planted; R2 positive but small; I at T_15 above I at birth.

Coordinator's expectation at freeze, after S1–S3 and before any primary forward, no rule attached: median I at birth near 0.45 against planted 0.174 and ceiling 0.788, so R1 not refuted and the position near 0.45; cross − foreign near 0.2, so R2 positive; I at T_15 near 0.6, so R3 positive; related minus unrelated near 0.1 with an interval that may include 0, so R4 is the one most likely to fail. The six sniper jobs gave I 0.66 at T_15 and 0.46 at T_10 at m 50, and that is what these guesses rest on.

## 12. Budget

GPU: sniper used 0.524 GPU-h (S1+S2 0.146, S3 0.378), primary ≤ 3.0, hard cap 5.0 GPU-h on `g6e.2xlarge`. The anthropic-direct session confirmed on 2026-11-01 that it runs no GPU instances, so the four-instance quota is this study's.

API credits ($200 key, expires 2026-11-15): **$0 planned.** The label inventory of 2026-11-01 found relatedness labels for every job (§4), so the $40 reserve for labels is released. The path below stays recorded in case a label control is needed later; nothing is submitted without the user's say. State of the credit on 2026-11-01 as reported by the anthropic-direct session: $7.76 spent, a $15 follow-up cap live, and about $70–118 of its own planned runs before the expiry.

Path, handed over by the anthropic-direct session on 2026-11-01 at head `2705eea8` (branch `infra/anthropic-direct`, now merged into `research/integrate`): module `infra/pg/neurons_llm.py`, subcommands `run-batch --settings infra/pg/settings/anthropic_sonnet55.json ...` (Message Batches, automatic chunking under 100,000 requests and 256 MB, `--dry-run` sends nothing) and `collect-batch`; failures that repeat to `--max-attempts` end in `responses/<batch>.declined.jsonl`. The key is read from the environment variable `<api-key-env-var>`, loaded only inside the command from the user's keys file; it is never written anywhere. Record: `infra/operations/anthropic-direct/2026-11-01_RUN.md`. Caveats carried from that record: the judge gate passed for Sonnet 5.5 at effort low on the S_gate task only (a_ctrl 0.897 [0.833, 0.955], n 39), so any new label task needs its own positive control and null; effort changes granularity (clusterer agreement 0.474 low, 0.526 medium, 0.646 high against validated self-agreement 0.744), so a task sensitive to how finely the model splits runs at effort high and says so; Haiku 5.5 is out (0.724 on the gate). That session had spent $10.60 of the $200 by the handoff.

Controls the labels need, because the judge gate on that path (Sonnet 5.5, a_ctrl 0.897 on the S_gate task; Haiku 5.5 fails at 0.724) covers a different task: a positive control of 30 planted-pair sides (unrelated by construction) and 30 ceiling-split sides (related by construction), labelled blind in the same batch; the labels are used only if at least 0.85 of each control group is labelled as constructed. Refusals are recorded as declines, not relabelled. Cache reads on Sonnet 5.5 cost 0.1× input.

## 13. Amendment 1, 2026-11-01: from the freeze to a closed result

Added after freeze 1 (`1fb5408a`) and before any primary forward. It changes no rule, threshold, item or arm. It adds the tasks that stand between the freeze and a closed result, the rows the grid in §11 lacks, the pre-specified responses to the failures the primary run can meet, and the limits that travel with the result. The amendment is the commit that adds this section, tag `reuse-amendment-1`. Sections 0–12 are not edited; where this section and an earlier sentence disagree on a count, §4 and this section are right and the earlier sentence is left as written.

### 13.1 Code that must exist, dry-run and be committed before the primary launch

At the freeze the job module `infra/aws_pg/jobs/reuse_ablate.py` has only `--stage sniper`, and no verdict script exists, although §9 step 4 names both. One Opus worker writes both in one session on this worktree; the coordinator reviews the diff; the commit SHA is written in the journal before `pg run`. The verdict logic is therefore fixed before the primary falls exist.

1. **Primary stage.** `--stage primary` in `jobs/reuse_ablate.py` and `infra/aws_pg/steps/reuse_primary.json`. Input: `inputs/primary_items.json` from `s3://<s3-bucket>/results/reuse/inputs/b972b00ceff4/`, manifest `b972b00ceff4`. Loads in ascending order: T_4 to T_13 and T_15, 11 loads, each behind the known-number gate of §5. At a job's birth checkpoint: arms identity, own, cross, random, foreign at m 50 direct on J's contexts, cap 12 by rank, plus the job's own ceiling split (identity, own, cross). At T_15: every job, arms identity, own, cross, random, foreign. Outputs per load as §10, `metrics.json` written inside the loop, samples as §7. The arm functions, m, ranking, replacement, background set and cancel rule are the sniper code's; the stage adds no computation.
2. **Verdict script.** `studies/reuse/scripts/verdict.py`. Reads the primary bank `results/primary/<n>/*/metrics.json` and `falls/`, the planted and ceiling references from `results/sniper/2/t15/metrics.json` as frozen in §3, and the labels from the manifest. Computes the §5 gates, each with its value, its failing-input value and its identity value; the §6 rules R1–R4 with the bootstrap of §6; every item under "Reported, no rule". Writes `results/primary/verdict.json` with the artifact path beside each number and `results/primary/GRID.md`, which is §11 plus §13.2 filled. It runs once on the primary bank at its committed SHA.
3. **Verdict dry run, before the launch, recorded in the journal.** (a) On the S3 bank `results/sniper/2/t15` and `t10`: the own, cross, random and foreign medians must match `S3.md` within 0.01. (b) Failing inputs, each must fail: own and random arm labels swapped, so the positive control fails; planted side labels shuffled within pair, so room fails (S3 gives −0.20 at m 50); cross and foreign swapped, so R2 reads not specific. A verdict script that passes on a swapped input is broken and does not launch anything.
4. **Offline gate.** `jobs/reuse_dryrun.py` extended to the primary stage on tiny random configs: two births at different checkpoints and one T_15; prints `REUSE_DRYRUN_OK` only with samples written for every arm and for the ceiling.

### 13.2 Rows the grid lacks

§11 has no row for R4 and none for the cancel-share, identity and known-number gates. The grid is filled as the union of §11 and this table. The R4 counts are 57 related and 93 unrelated as drawn (§4); the expectation of about 90 and 60 in §6 was written before the draw and is wrong; the rule is unchanged.

| prediction | predicted | observed | verdict |
|---|---|---|---|
| R4 H_related-share | no point registered; coordinator's expectation at freeze: near 0.1, interval may include 0 | | |
| Cancel share at birth | under 40% of 150 | | |
| Identity | `|Δ| ≤ 1e-3` on every context at every load | | |
| Known number | ≤ 1e-2 at each of 11 loads; the failing input fails at each | | |

### 13.3 The primary run, §9 step 4, on the user's go only

- Journal entry at launch, before `pg run`: code SHA, manifest hash, fresh prefix `results/reuse/primary/<n>`, instance ID, budget 150 min, and the §11 expectations restated unchanged.
- Launch `--detach`; follow with `pg watch` and `metrics.json`; never `job.log`. Not scored while running.
- The first load's `samples/arms.jsonl` is opened as soon as it syncs: the prompt as forwarded, the marked token, a_P before and after, the ablated set. Wrong input or wrong format: `pg kill`, journal, one debug, fresh prefix.
- Responses, fixed here: a load that fails the known-number gate gets one debug on a fresh prefix, then does not count, and the jobs born there are reported as not run; the live count over 150 is the first line of every report and no rule is conditioned on it. A job over twice its estimate with no `metrics.json` advancing is killed and journaled. Total use is capped at 5 GPU-h. A failed gate never gets a relaxed threshold.
- One executor session from launch to `STATUS`. If the session ends first, it writes `studies/reuse/HANDOFF.md`: prefix, instance, loads done, exact next action.

### 13.4 After `STATUS`, in this order

1. Pull the bank to `results/primary/<n>/`; check every artifact timestamp is after the launch time.
2. Open samples before any number: per arm and per tercile, the job with the highest I and the one with the lowest; answers in the journal as §7.
3. Journal entry with the §11 and §13.2 predictions restated, then `verdict.py` once at its committed SHA. That is the one unblinding.
4. A different model reads the artifacts, read-only, under the result-review method: recompute R1–R4 and the gates from `falls/` under another seed; confirm each gate's failing-input value is present; open the samples; one literature search under the field's terms (upstream neuron ablation, drivers, training checkpoints, OLMo). Opus executes 13.1–13.4.3, so the coordinator (Fable) reviews. Output `results/REVIEW_<date>.md`.
5. `results/RESULTS.md`, written after the review: numbered Results, one declarative sentence each, a null in the same register as a positive; the grid filled; every number with its artifact path; run, unrun and cancelled counts over 150 in the first line; the limits of §13.5 as their own sentences; Eliwat credited. A frozen rule found wrong gets a dated exploratory note with both readings; the frozen rule decides.
6. Three figures, each with the number, its interval and the null drawn on the panel: I per job against the planted and ceiling bands; cross against foreign; I at birth against I at T_15. Regenerated after any correction.
7. Commit of named files on the user's say; nothing pushed; `main` untouched; then merge into `research/integrate`.
8. Journal closeout; the review's findings become the next spec's Phase 0.

### 13.5 Limits that travel with the result

- One model and one checkpoint series, Olmo-3 7B. No replication at another scale, seed or model; the budget did not allow one.
- The planted reference is measured at T_15 only (§8); birth numbers are read against a band from a later checkpoint.
- Attention-routed computation is not measured (§8).
- Labels come from the UNITE inventory, not from a judge; with 57 related and 93 unrelated jobs R4 has little power, and its interval width says so.
- Mean replacement is an off-distribution intervention; the identity gate and the energy-matched random arm bound its effect and do not remove it.
- The ceiling splits are UNITE's; 15 of the 65 setting splits are also primary ceilings; room is reported both ways (§3, §5).
- A null on R1 or R2 is a finding in its own right and is reported as one.

### 13.6 Budget for the remainder

GPU: primary 1.0–1.5 GPU-h, cap 3.0 inside the 5.0 hard cap; verdict, dry runs and review 0 GPU-h. API: $0. Agent sessions: one Opus worker for §13.1, one Opus executor for §13.3 through §13.4.3, the coordinator for the diff review and §13.4.4–5; never more than two agents at once.

### 13.7 Outside this study, for the record

User decisions, none of which blocks REUSE: fast-forward `main` to `research/integrate`; merge or drop the four September branches `depthanchor`, `depthoff`, `depthopen`, `postcheck` (README union conflict); the dirty worktrees drafts, neurons-explore and neuronshow, and the untracked folders on main; the 15 files in `<local-agent-transcripts>/`. UNITE's open result still needs a write-up with the review beside it, crediting Eliwat; not scheduled.
