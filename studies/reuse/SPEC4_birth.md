# SPEC4, BIRTH: what changes when a neuron's activation reaches a new region, and what the new region's inputs share with the old one once the high-background neurons are set aside

Status: **draft, 2026-11-02, not frozen.** Freeze 4 is the commit after the §12 dry runs pass and the verdict script fails on each failing input, on the user's go, tag `reuse-freeze-4`; no forward before it. Parents: `SPEC.md` (frozen, `reuse-freeze-1`), `SPEC2.md` (`reuse-freeze-2`), `results/REVIEW_amend2.md`, `results/REVIEW_heads.md`, the fourth review (`REVIEW_2026-11-01_fourth.md`) and the exploratory hub split of 2026-11-01 23:30 CST that this spec turns into tests. Written by the reviewing session (Fable 5.1). No rule of the parents changes; every rule here is new and carries its own letter.

**Disclosure.** The author has read every REUSE bank, including the exploratory split of the 150 primary cases by whether their driver sets hold a high-background neuron (§0). The thresholds below are set from the reference distributions and from that look. Two parts of this spec read data that does not exist yet and are confirmatory: the 93 fresh cases of §4, which no instrument has forwarded, and the pre-birth captures of §3.1, which no run has taken. The re-read of the 150 primary cases under the changed instrument is a re-analysis and is labelled so wherever it is reported.

**Scope rule, carried from `SPEC.md`:** one pipeline, one unblinding, rules that a script can check, nothing secondary unless it is free.

## 0. What the program knows, in the words this spec uses

A **pole** P is one sign of one MLP neuron of `allenai/Olmo-3-1025-7B` at layer L. A **cluster** of P is a group among its top-50 contexts at T_15, as the Sonnet clusterer grouped them (REUSE called it a job; a reader finds most clusters not coherent, so the word task is not used). The **older cluster** O* is present at the checkpoint before the **new cluster** J first appears; J's **birth** t_b is that first checkpoint and t_b−1 the grid checkpoint before it (UNITE's dating, `t_before` in the items). A **driver set** D(C) is the 50 upstream MLP neurons with the largest summed direct attribution to P on C's contexts at a checkpoint; **I** is the fall of P's activation on J when D(O*) is held at its background mean, divided by the fall when D(J) is held; **s** is |D(J) ∩ D(O*)| / 50. The **two-neuron null** is 56 sums of two unrelated neurons read as one pole; the **same-cluster ceiling** is 138 document-disjoint halves of one cluster.

Established: at birth I is 0.226 against a null of 0.031 and a ceiling of 0.65; the cross fall beats a foreign neuron's drivers and an activity-matched set; I rises to T_15 as the ceiling's own I rises; the clusterer's relatedness label is the overlap s in another form; attention heads and downstream readers could not be read with their first instruments (Results 1–23).

Found on 2026-11-01 in the banks, not yet tested: **high-background neurons** carry the growth. Call B_t the 1% of upstream neurons with the largest mean activation magnitude over the 200 background positions at checkpoint t (the same ā the ablation already computes; at T_15 the cut is 0.241 against a median of 0.003). A driver set of 50 holds about three of them. They carry 3% of the own set's attribution and two thirds of a foreign set's. Splitting the primary cases by whether D(O*) holds one:

| quantity | birth, none | birth, some | T_15, none | T_15, some |
|---|---|---|---|---|
| foreign fall | 0.000 (38) | 0.017 (110) | 0.039 (44) | 0.094 (105) |
| I over cases | 0.156 (43) | 0.270 (105) | 0.230 (41) | 0.399 (108) |
| two-neuron null | 0.006 (123) | 0.038 (325) | 0.040 (23) | 0.233 (33) |
| ceiling | | | 0.722 (40) | 0.727 (89) |

Without them the T_15 null is the birth null. The split is post hoc and B_t membership correlates with layer, so it is a hypothesis, and §6 gives it a rule. Also found: s without B_t is 0.112 [0.079, 0.143] for the cases, 0.376 for the ceiling halves, and 0.053 at the null's 95th percentile.

## 1. Question, decision, hypotheses

**Question.** When training extends a pole's activation to a new region J, (a) what arrives at J's birth: the upstream features P reads there, P's reading of them, or neither; (b) how much of J's input is O*'s input, once the high-background neurons are set aside; (c) does that sharing hold on cases no instrument has seen.

**Decision.** (a) decides what the lineage program is about: if P's reading changes while the features stand (H_read), lineage is a story about input rows, and the next study follows rows; if the features arrive (H_arrive), lineage is a story about upstream births, and the next study follows them downstream; if neither changes (H_gate), births are threshold crossings and the next study measures pre-activations, not sets. (b) decides whether the post's "rise over training" paragraph is rewritten around the high-background neurons and whether the frozen rule R1's refutation is explained. (c) decides whether the label-free sentence "a pole's two clusters share about a seventh of their top drivers, above two neurons and below one cluster's halves" is confirmatory or stays exploratory.

| hypothesis | statement | refuted when |
|---|---|---|
| H_read | At t_b−1 the drivers of J are active on J's contexts, and P's input rows are not yet aligned with them | r_a below 0.5 or r_ω above 0.5 (L1) |
| H_arrive | At t_b−1 P's rows are aligned with J's drivers, and the drivers are not yet active on J | r_ω below 0.5 or r_a above 0.5 (L1) |
| H_gate (boring) | Both stand at t_b−1; P's pre-activation on J is near its birth value and the output is held down by the gate | r_a or r_ω below 0.5, or ρ_g below 0.5 (L1) |
| H_both | Neither stands at t_b−1 | r_a or r_ω above 0.5 (L1) |
| H_hub | The generic term of every arm is the high-background neurons: without them a foreign neuron's drivers are inert at T_15 and the two-neuron null at T_15 equals the birth null | foreign′ − random′ at T_15 above 0.03 (M1); or the T_15 null′ interval above the birth null′ interval by 0.05 or more (M2) |
| H_share | Without them, J's input still shares more with O*'s than two neurons' inputs share, on fresh cases | the fresh cases' s′ interval touches the null′'s 95th percentile (S1) |
| Instrument artifact | | any gate of §7 fails |

Primes mark quantities read with B_t excluded. r_a, r_ω, ρ_g are defined in §3.1.

## 2. The measurement, shared with the parents

Forward, window, readout a_P, mean replacement, the fall Δ(S; C), the cancel rule, the bootstrap over neurons (2,000 draws, seed 0, 95% percentile) and the weighting of the null by live count per birth checkpoint and one third per tercile are `SPEC.md` §2 and §6 and `SPEC2.md` §3 B1, unchanged. m is 50, ranking direct, as frozen. The instrument module `infra/pg/reuse_ablate.py` is not edited; a new module adds the three computations below.

**High-background set.** At each load t, `B_t` = the neurons of layers 0 to 30 whose |ā_k| lies in the top 1% over all 341,248 upstream neurons, ā from `reuse_ablate.background_mean_a` on UNITE's e1 positions (the same tensor the ablation writes back). `B_t`, its cut and the per-layer counts go to `high_background.json`. **Exclusion.** B_t is removed from the candidate pool before `top_m` ranks D(J), D(O*), the foreign set and the matched pools of R and R_J. Nothing else changes: the ablated values are still ā, the arms are the parents' arms. Every set is written with a flag per member saying it is outside B_t, and the verdict asserts the flag on every member of every set. **Sensitivity, offline, no rule:** each capture banks the 200 largest attributions with their ā, so s′ can be re-read with cuts of 0.2% and 3% without a forward; the falls are read at 1% only.

## 3. Three parts, one pipeline

### 3.1 Part A, what arrives at birth (new; confirmatory)

For every case, the contexts of J and of O* are forwarded at t_b−1 with capture only, no ablation: a_k(x) for every upstream neuron, P's g and u, and the layer scales s_l(x). The driver sets are those ranked at birth, D′_b(J) and D′_b(O*), and are held fixed across the two checkpoints.

For driver k and context set C at checkpoint t: **activity** ā_k(C, t) = mean over x ∈ C of |a_k(x)|; **alignment** ω_k(t) = |c̄_g G_P(t)·d_k(t) + c̄_u U_P(t)·d_k(t)| with d_k = γ_l ⊙ W_down[:, k] and c̄_g, c̄_u P's mean coefficients over J at birth, held fixed so that ω reads weights only (`unite_route.coefficients`, `Rows.loads`); the two are the factors of `unite_route.attribution`, a_k · s_l · (c_g G·d_k + c_u U·d_k). Per case:

- ρ_a(J) = median over k ∈ D′_b(J) of ā_k(J, t_b−1) / ā_k(J, t_b); ρ_a(O*) the same over D′_b(O*) on O*.
- ρ_ω(J) = median over k ∈ D′_b(J) of ω_k(t_b−1) / ω_k(t_b), a sign change counted as 0; ρ_ω(O*) the same.
- **r_a = ρ_a(J) / ρ_a(O*)** and **r_ω = ρ_ω(J) / ρ_ω(O*)**: J's factors relative to the old cluster's drift over the same interval.
- ρ_g = mean over J of g_P(x, t_b−1) / mean over J of g_P(x, t_b); ρ_u likewise; ρ_P the same for a_P (the dating quantity; expected small by construction).
- Reported beside them: the cosine of P's G and U rows between t_b−1 and t_b; the share of D′_b(J) that is also in D′_{b−1}(J) when J's contexts are ranked at t_b−1 (whether the same neurons already lead the attribution before birth).

**Rule L1.** Medians over the cases live at birth, bootstrap over neurons. A factor is *present* when its interval lies above 0.5 and *absent* when it lies below 0.5; otherwise *undecided*. H_read: r_a present and r_ω absent. H_arrive: r_ω present and r_a absent. H_gate: both present and ρ_g present. H_both: both absent. Any undecided factor: *not separated*, and the four medians are reported with their intervals. The rule is read on the 150 primary cases and on the 93 fresh cases separately; the two readings must agree for the §1 decision to be taken, and a disagreement is its own outcome.

**Gate L (baseline).** ρ_a(O*) and ρ_ω(O*) medians inside [0.5, 2.0], with the interval of each: the old cluster, present at both checkpoints, must read as present. Failing input: the baseline replaced by ρ_P, the pole's own mean activation on J at t_b−1 over that at t_b, which must read below 0.5 (the dating makes it small). If gate L fails, L1 is not read. *Amended before freeze, 2026-11-02 (coordinator): the draft's failing input was "J's own pre-birth capture in place of O*'s, must read below 0.5"; under the draft's own prediction r_a 0.75 that input reads about 0.75 and cannot be relied on to fail, so it could not serve as a failing input.*

### 3.2 Part B, the high-background term (re-analysis on the 150; new instrument reading)

At every birth load and at T_15: the 150 primary cases with arms identity, own′, cross′, random′, foreign′, R_J′ on J; their 138 ceiling halves (side 2: identity, own′, cross′) at birth and the 129 distinct halves at T_15; the 56 two-neuron sums (side 1: identity, own′, cross′, random′; labels and shuffled labels) at every birth load and at T_15. Everything B_t-excluded.

- **Gate room-c** (birth): median I′_ceil at birth − weighted 95th percentile of the null′ at birth ≥ 0.200; failing input, side labels shuffled within pair. **Gate room-c15** likewise at T_15 with the 129 halves and the 56 sums read there.
- **R1c.** Median I′ at birth over live cases against the null′ at birth: overlap → refuted; no overlap, real higher → not refuted; lower → its own outcome. **R1c15** the same at T_15 against the null′ at T_15. Each reads only if its room gate passes.
- **M1.** Median over live cases at T_15 of Δ(foreign′; J) − Δ(random′; J): interval inside [−0.03, 0.03] → the high-background neurons were the generic term; above 0.03 → a generic term remains and its size is reported; below −0.03 → its own outcome.
- **M2.** Null′ at T_15 against null′ at birth, both weighted as SPEC2 B1: medians within 0.05 and intervals overlapping → one null at both checkpoints; T_15 higher by 0.05 or more without overlap → the T_15 null has a term the high-background neurons do not explain.
- **R2c.** Median cross′ − foreign′ and cross′ − R_J′ at birth: both intervals above 0 → specific; else not, naming the control.
- **R3c.** Paired median I′_T15 − I′_birth over cases live at both, minus the same over the 138 ceiling halves, bootstrapped jointly over neurons: contains 0 → generic; above 0 → beyond the same-cluster rise; below 0 → the halves rise more.

### 3.3 Part C, the overlap on fresh cases (new; confirmatory)

The 93 fresh cases (§4) at their birth checkpoints: capture of J and O*, arms identity, own′, cross′, random′ on J (the positive control and the cancel rule need own′ and random′; I′ is reported but no rule attaches, since the fresh set is small and the question is the overlap); the ceiling halves of the 35 that have one, 31 distinct splits read once each (identity, own′, cross′). No T_15 reads. The foreign pole Q of a fresh case is chosen at T_15 by the parents' rule and recorded, but Part C has no foreign arm, so it is not forwarded (noted 2026-11-02).

- **S1.** Median s′ over the live fresh cases, bootstrap over neurons, against the weighted 95th percentile of s′ over the 56 sums at the birth loads (the null′ for s, weighted as the null for I): interval above the percentile → H_share not refuted; touching or below → refuted.
- **S2, reported.** Position (median s′ − null′ median) / (ceiling′ median − null′ median), the ceiling′ being the 169 reads (138 primary + 31 fresh) at birth; s′ by the clusterer's label, descriptive; s′ at the 0.2% and 3% cuts from the banked top-200.
- **S3, reported.** The same S1 statistic on the 150 primary cases (re-analysis), so the two sets stand side by side.

## 4. Items

**Primary set:** the 150 cases of `inputs/primary_items.json` (manifest `b972b00ceff4`), their ceiling halves, and the 56 sums and setting halves of `inputs/setting_items.json`; nothing drawn again.

**Fresh set:** every case in UNITE's pool of 312 (`../interp_research-unite/studies/unite/results/reuse_candidates.json`, filters of `SPEC.md` §4: no shared document or marked token between J and O*, |J| ≥ 6, |O*| ≥ 6, L ≥ 4) that is not a primary or sniper case and whose neuron carries no primary case. Counted 2026-11-02: **93 cases on 59 neurons**, terciles 24 / 33 / 36, births T_6 2, T_7 3, T_8 3, T_9 4, T_10 16, T_11 10, T_12 25, T_13 30; 35 with a ceiling half of O*. J capped at 12 contexts by rank; O* at 12; labels from the clusterer inventory (descriptive only); foreign candidates and `t_before` from UNITE's `j4c_items.json` by `build_items.py`'s rule. The builder writes `inputs/birth_items.json` and `inputs/manifest_birth_<sha12>.json` with counts and hashes, stops if any fresh case shares a neuron with a primary case, and uploads both to S3 before freeze. No draw, no seed: the set is the whole remainder on new neurons.

Loads: T_5 (step 8,000; it serves the two primary and two fresh cases born at T_6 as their t_b−1), T_6 to T_13, T_15: ten loads. T_15 first, because the foreign pole Q of each fresh case is chosen there by the parents' rule; the 150 primary cases keep their Q from `results/reuse/primary/1/out/t15/foreign.json`.

## 5. Point predictions, written before any forward, no rule attached

| quantity | predicted | refuted if |
|---|---|---|
| null′ at T_15 (M2 input) | 0.05 [0.02, 0.10] | outside [0.00, 0.12] |
| null′ at birth | 0.01 [0.00, 0.03] | outside [−0.02, 0.05] |
| room-c at birth | 0.45 | under 0.20 |
| I′ at birth, primary re-read | 0.17 [0.12, 0.23] | outside [0.08, 0.30] |
| I′ at birth, fresh set | 0.17 [0.10, 0.25] | outside [0.05, 0.35] |
| I′ at T_15, primary | 0.24 [0.18, 0.31] | outside [0.12, 0.40] |
| M1, foreign′ − random′ at T_15 | 0.02 [−0.01, 0.04] | above 0.05 |
| R2c | specific against both | either interval contains 0 |
| R3c | generic, difference 0.00 [−0.05, 0.05] | interval above 0.05 or below −0.05 |
| S1, fresh s′ | 0.11 [0.07, 0.15] against a null′ p95 of 0.05 and a ceiling′ of 0.38 | interval touches the p95 |
| L1 | H_read: r_a 0.75 [0.55, 0.95], r_ω 0.35 [0.15, 0.55], ρ_P 0.15 [0.05, 0.30] | r_a under 0.5 or r_ω over 0.5 |
| gate L baseline, ρ_a(O*), ρ_ω(O*) | 1.0 [0.8, 1.2] each | outside [0.5, 2.0] |
| peak GPU memory, deepest poles | 23 GB [19, 27] | outside |
| billed time on the rental | 0.75 h [0.5, 1.3]; cost $0.6 [0.4, 1.2] of credit | over 2 h or $2 |

The prediction for L1 rests on the ordinary finding that lower layers settle earlier in training than the rows that read them; it is the author's and may be wrong, which is the point of writing it. Every prediction for Part B comes from the post hoc split of §0 and is weaker evidence than a prediction written blind; the fresh set and Part A carry the confirmatory weight.

## 6. Decision rows

| outcome | next |
|---|---|
| H_read not refuted on both sets | lineage follows input rows: the next spec measures when and toward which upstream features G_P rotates, over the full grid |
| H_arrive not refuted on both sets | lineage follows upstream births: the next spec dates the drivers' own births and asks whether a new cluster waits for them |
| H_gate not refuted | births are threshold crossings: the next spec reads g_P and u_P on J across the grid, not driver sets |
| not separated, or the two sets disagree | reported as such; no next spec on this question before a larger fresh set |
| H_hub not refuted (M1 inside the band and M2 one null) | the post's "over training" paragraph is rewritten: the growth of every arm was the high-background neurons; R1's refutation is explained; a method note is written (§13) |
| H_hub refuted | the generic term has another source; the post keeps "the rise is generic" with the measured remainder |
| S1 not refuted | the overlap sentence becomes confirmatory and leads the post's label-free paragraph |
| S1 refuted | the sentence stays exploratory and the post says the fresh cases did not show it |

## 7. Gates, each with value, failing input and identity value

| gate | passes when | failing input, must fail |
|---|---|---|
| Known number | at each load, as `SPEC.md` §5 (≤ 1e-2 on 16 contexts) | the previous grid checkpoint's bank |
| Completeness | ≤ 1e-2 at each load | the column from the wrong layer |
| Identity | ∣Δ∣ ≤ 1e-3 on every context | own′ drivers |
| Exclusion | no member of any set is in B_t at that load; `high_background.json` present with 3,412 ± 10 members per load (1% of 341,248) | exclusion disabled in the verdict's reading of the bank: members of B_t found |
| Positive control | median own′ fall at birth ≥ 0.25 over the 150, and over the 93 | random′ in place of own′, under 0.25 |
| Room-c, room-c15 | ≥ 0.200 | side labels shuffled within pair |
| Cancel share | under 40% at birth, each set | own′ and random′ swapped |
| Gate L | §3.1 | ρ_P as the baseline (amended 2026-11-02, see §3.1) |
| Memory | `torch.cuda.max_memory_allocated` recorded per load in `metrics.json`; under the rental's GPU memory by at least 2 GB at every load | the T_15 load (deepest poles) on a 24 GB card in the dry run's estimate; reported, no stop |

Each gate's three numbers travel with it. A failed room gate leaves its rule unread and the readable items reported, as in SPEC B.

## 8. Threats

| threat | effect | mitigation |
|---|---|---|
| The 1% cut is arbitrary | B_t too small leaves the term; too large removes real drivers | own′ fall must hold the 0.25 floor; s′ at 0.2% and 3% reported; cut frozen before the data |
| Excluding B_t changes the sets, so I′ is a new quantity | not comparable with I | both reported; the 150 are a re-analysis by name; the fresh set is read on I′ only |
| Drift between t_b−1 and t_b inflates every ratio | r_a, r_ω read low | ratios are relative to the old cluster's drift (gate L) |
| The grid is coarse (320k → 560k steps) | "t_b−1" is far from the birth | reported per birth load; T_12 and T_13 births carry 55 of 93 fresh cases |
| Fresh cases share 28 poles' neurons with the primary set | not independent | excluded: the fresh set is on 59 new neurons |
| Mean replacement off-distribution | fall from shift | identity gate, random′ and R_J′ at matched energy, now without B_t on either side |
| Hub membership correlates with layer | Part B's effect is a layer effect | per-tercile readings reported for M1, M2, R1c |
| Mean-replacing a large-ā neuron changes its layer's post-feedforward norm scale s_l(x), which rescales every other neuron of that layer (Stolfo et al. 2024 §3.2 found such a path for entropy neurons) | the generic term of a hub is a norm effect, not a path through the hub's own direction | not separated here; excluding B_t removes the term either way; a replacement that holds s_l at its clean value is a next-spec instrument (added 2026-11-02 after the literature search) |
| One model, one run | | stated; the fresh set is the only replication in budget |

## 9. Compute: vast.ai, chosen by price, with the AWS lane as the fallback

**Why vast.** The job is ten checkpoint loads of about 15 GB each plus about 90,000 readout forwards and 25,000 captures, about 28 GPU-minutes on an L40S. On AWS that is one `g6e.2xlarge` hour at $2.24 in cash. On vast.ai the same work rents for about $0.5 to $1.2 of the account's $30.56 credit (read live 2026-11-02 05:42Z), so the user's instruction to use vast.ai is also the cheaper path. The run goes through the lane that already exists: `infra/vast/controller.py` (plan, rent, respawn), `watchdog.py` on the laptop under a session Monitor, the Lambda reaper as the money backstop, the deps-only image `shackleton:mdl-v1` (torch 2.12, CUDA 13.0, the AMI's pins), and the cell `infra/vast/cells/pipeline_job.py`, which runs `infra.aws_pg.jobs.pipeline` on a steps file unchanged. Weights come from the Hub in-run, as the AWS runs did (`download("allenai/Olmo-3-1025-7B", revision)`, the cache cleared after each load), with `HF_ONLINE=1`, `HF_HUB_DISABLE_XET=1` and `HF_HUB_CACHE=/workspace/hf/hub` in the manifest env and the read-scoped `<api-key-env-var>` shipped at rent time. Not from S3: pulling 150 GB of checkpoints out of S3 would cost more in egress than the compute.

**What the instrument needs of the card.** The model is 14.9 GB in bf16. `unite_route.Rows` caches γ ⊙ W_down in fp32 on the GPU for every layer below L, up to 5.6 GB for a layer-31 pole, and the capture at batch 32 adds about 1.5 GB, with 1 to 2 GB of allocator overhead: about 23 GB at the deepest poles. **Hard requirements:** GPU memory ≥ 32 GB; host RAM ≥ 32 GB; disk ≥ 80 GB; CUDA ≥ 13.0 (the image); compute capability ≥ 8.0 (bf16); one GPU; `verified`, `rentable`, reliability > 0.98; download ≥ 800 Mb/s; bandwidth billed at $0 both ways (the lane's rule: hosts that bill bytes would bill the checkpoints); storage ≤ $0.30 per GB-month; no mining card (CMP series excluded by name); no machine in `infra/vast/HOSTS.json` known_bad. 24 GB cards are excluded by the memory estimate, not by evidence; the memory gate of §7 records the number so the next spec can admit them if it is under 22 GB.

**Price rule, frozen.** Among offers passing the requirements, the executor computes for each

estimated minutes = 10 × (14.9 GB × 8 / min(inet_down, 1,500 Mb/s)) / 60 + 28 × max(0.5, 93.6 / dlperf) + 10,

that is the ten downloads at the host's rate capped at what the Hub serves, the L40S compute time scaled by vast's dlperf (L40S 93.6; speed-ups beyond 2× are not credited), and ten minutes of load, known numbers and uploads; and

estimated cost = (dph_total + storage_cost × 80 GB / 730 h) × minutes / 60.

The offer with the smallest estimated cost is rented. Tie-break within $0.10: the higher reliability, then the shorter estimated time. The full offer table with both columns is banked as `results/birth/<n>/offers.json` before the rent, so the choice is auditable. Reliability is vast's `reliability2`.

**Snapshot, 2026-11-02 05:42Z, illustrative; prices move and the rule is re-run at launch.** Offers passing the requirements with the strict base filter:

| offer | GPU | memory | $/h | down Mb/s | reliability | dlperf | est. min | est. $ |
|---|---|---|---|---|---|---|---|---|
| 44457595 (PR) | RTX 5090 | 32 GB | 0.760 | 1,175 | 0.9903 | 199 | 41 | 0.54 |
| 58380903 (TW) | RTX 5090 | 32 GB | 0.674 | 819 | 0.9883 | 199 | 48 | 0.56 |
| 58771762 (VN) | RTX 5090 | 32 GB | 0.741 | 874 | 0.9947 | 174 | 48 | 0.61 |
| 58763445 (WA, US) | RTX PRO 6000 WS | 96 GB | 1.622 | 2,078 | 0.9991 | 283 | 37 | 1.02 |
| 32252204 (ES) | RTX PRO 6000 Max-Q | 96 GB | 1.622 | 909 | 0.9953 | 284 | 46 | 1.26 |
| for comparison, excluded by memory: 8936322 (IE) | RTX 4090 | 24 GB | 0.550 | 13,831 | 0.9991 | 97 | 50 | 0.47 |
| for comparison, AWS `g6e.2xlarge` | L40S | 48 GB | 2.242 | | | 93.6 | 55 | 2.06 cash |

On that snapshot the cheapest offer is 44457595 at about $0.54, but the tie-break within $0.10 prefers the higher reliability, so the rule as frozen rents 58679921 at about $0.59 (`results/review4/` does not hold the raw offers; the launch-time offers are banked as `offers.json`). The rule is `studies/reuse/scripts/offers_birth.py`. A 5090 has not served this stack before; the first load (T_15, every pole, the deepest included) is the smoke test, and a failure there costs under $0.05 and triggers the respawn below. The RTX PRO 6000 at $1.02 is the choice if the user prefers a card class the lane has run (`HOSTS.json` known_good lists the WS).

**Lane mechanics.** (1) Inputs the box can read: the box IAM user reads only `weights/hf/*` and `results/mdl/*`, so the laptop copies, with the `<aws-profile>` profile, the items (0.6 MB), `birth_items.json`, `setting_items.json`, the corpus prefix (725 MB) and the primary T_15 `foreign.json` to `s3://<s3-bucket>/results/mdl/reuse_birth/inputs/`, and the steps file to `results/mdl/reuse_birth/<n>/steps.json`; outputs land under `results/mdl/reuse_birth/<n>/`. No AWS credential of the user's goes to the box. (2) Manifest `infra/vast/manifests/reuse_birth.json`: image mdl-v1, disk 80, `gpu_query` from the requirements above, one cell `pipeline_job` with `steps_s3` and the shipped `pg_git_sha`, env as above, `watch` {`max_billed_h`: 3, `load_backstop_min`: 30, `progress_stall_min`: 20, `progress_kill_min`: 45}. (3) Preflight: the static tier (`TIER=1 bash preflight.sh`) stamps `.preflight_pass` for HEAD; `rent` refuses without it. (4) `controller.py plan` prints the offers; the executor applies the price rule, banks `offers.json`, journals the choice, then `controller.py rent <manifest> <offer>`. (5) `python3 -u watchdog.py` runs under a session Monitor until `_rental_complete.json`; the reaper is deployed (`bash deploy_reaper.sh`) before the first rent. (6) Liveness and progress are read from `live_out/*/metrics.json` and `pipeline_status.json` on S3, never `job.log`. (7) After completion the laptop pulls the bank to `results/birth/<n>/` and `controller.py ledger` reconciles the realized dollars into the lane's spend record and `infra/aws_pg/operations/`.

**Respawn and fallback.** An infra death (host, network, net probe) respawns on the next offer by the price rule, at most twice; a traceback never respawns. If no offer passes the requirements under an estimated $2.00 within 30 minutes of planning, or after two infra deaths, the run moves to the AWS lane as frozen in the parents (`./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline … --type g6e.2xlarge --no-weights`, cap 1.5 GPU-h), on the user's say, with the inputs already in S3.

**Budget.** vast: estimate $0.6, cap $3.00 of the $30.56 credit (the lane's `budget.json` ceiling, $36, is not raised). AWS fallback: cap 1.5 GPU-h, about $3.40 cash, only on the user's say. API: $0. The REUSE study's 5.0 GPU-hour cap counts vast billed hours at face value; after this run the study stands at about 3.5 of 5.0.

## 10. Code: NEW and REUSED

NEW, one Opus worker, own worktree, one session: `infra/pg/reuse_birth.py` (B_t and `high_background.json`; the exclusion wrapper around `top_m`, `random_matched` and the foreign choice; the pre-birth capture statistics ρ_a, ρ_ω, ρ_g, ρ_u, ρ_P and the row cosines; the top-200 bank; the memory record), `infra/aws_pg/jobs/reuse_birth.py` (stages `birth_t15`, `birth_load`, `prebirth_load`, one process per load running the three parts), `infra/aws_pg/steps/reuse_birth.json`, `infra/vast/manifests/reuse_birth.json`, `studies/reuse/scripts/build_items_birth.py`, `studies/reuse/scripts/verdict_birth.py` (gates, L1, room-c, R1c, R1c15, M1, M2, R2c, R3c, S1–S3, the §5 table, `GRID4.md`), `jobs/reuse_dryrun.py` part 8.

REUSED, byte-identical, asserted by `git diff --stat` at freeze: `infra/pg/reuse_ablate.py` (`background_mean_a`, `ablation_dict`, `readout`, `direct_A`, `top_m`, `random_matched`, `fall`, `job_fall`, `cancel`), `infra/pg/unite_route.py` (`capture`, `Rows`, `attribution`, `coefficients`, `background_positions`), `infra/pg/unite_stats.py` (`boot_neurons`, `tercile`, `rule_median`), `infra/aws_pg/jobs/pipeline.py`, `infra/vast/controller.py`, `watchdog.py`, `policy.py`, `cells/pipeline_job.py`, `scripts/verdict.py` and `verdict2.py` (readers, `planted_ref`, `joint_rise`, `gate`).

## 11. Forwards and time

| load | cases | captures | readouts |
|---|---|---|---|
| T_15 | 150 primary, 129 halves, 56 sums | 3,600 + 1,550 + 1,350 | 10,800 + 2,300 + 2,700 |
| T_5 | pre-birth for 4 cases | 100 | 0 |
| T_6 to T_13 (eight loads) | births: primary cases there with 6 arms and their halves; fresh cases there with 4 arms and their halves; 56 sums × 4 arms; pre-birth captures for the cases born at the next load | 3,600 + 2,200 + 1,650 + 10,750 + 5,800 | 10,800 + 2,500 + 4,500 + 2,200 + 21,500 |
| total | | about 30,600 | about 57,300 |

About 88,000 forwards by the table above. *Counted from the built items on 2026-11-02 (`build_items_birth.py`, the job's arm lists): 87,974 readouts and 35,567 captures, about 123,500 forwards; the table leaves out the per-unit "before" readout and the shuffled planted arms that room-c's failing input needs; the largest load is T_15 with 19,852 readouts, and the planted pairs are 54,000 of the readouts.* The heads primary ran 76,200 readouts and 24,300 captures in 1,424 s of units on an L40S (`REVIEW_heads.md` §1), so about 29 minutes of compute here; on a 5090 by dlperf, about 15 to 20. Ten loads at about 2.5 minutes each of download and load on a 1 Gb/s host: 25 minutes. Billed estimate 0.8 h (the §5 row's 0.75 h [0.5, 1.3] stands).

## 12. Order, gates and the blind

1. Literature, before freeze, queries recorded in the journal: "massive activations" (Sun et al. 2024), "super weights" and "super activations" (Yu et al. 2024), "attention sinks" (Xiao et al. 2023), "emergent outlier features" (Dettmers et al. 2022), each read for the specific claim that a small set of neurons with extreme background activation exists and whether anyone has reported its effect on attribution-ranked ablation; "feature formation during pretraining checkpoints", "circuit emergence training dynamics", "neuron specialization over training", "Pythia checkpoints interpretability", "OLMo checkpoints". A published hub effect is cited and the method note of §13 becomes a confirmation of it on this model, not a contribution. *Done 2026-11-02, before freeze: `results/review4/LIT_high_background.md` (45 papers opened, 40 read in their relevant sections, queries listed there). Outcome: the extreme top of the set is the published object (super activations / super weights, Yu et al. 2024; massive weights, Oh et al. 2024; down-projection-input outliers, An et al. 2025; the source of the residual-stream massive activations of Sun et al. 2024), one to a few early-layer MLP neurons firing on the first token and delimiters; the 1% set by background mean is not published, the nearest rule being keystone neurons (Ji et al. 2026, under 0.2% of channels, high mean, low variation); no paper reports the effect on attribution-ranked sets or mean-ablation controls, and none recommends excluding or matching on such units, so §13 stands as a new report citing Ji et al. §3.3, Stolfo et al. §3.2 and Sun et al. §3 as the nearest results; Part A's order question is open, and the toy-model induction-head results disagree on the order (Bietti 2023 reading first; Ge 2025 and Dong 2026 upstream first; Singh 2024 co-evolving).*
2. The worker builds §10; `build_items_birth.py` prints §4's counts or stops; the dry run on tiny models passes with samples for every arm, the pre-birth statistics, `high_background.json` and the memory record; the verdict dry run stops on an incomplete bank, reproduces the amend2 planted medians on the amend2 bank within 0.01 when B_t is empty, and fails on each failing input of §7 and on a fresh set that shares a neuron with the primary set.
3. Coordinator reviews the diff against this document line by line, then **freeze 4** (`reuse-freeze-4`); the two spec status lines of SPEC B and SPEC A get their dated freeze line at the same commit, as the fourth review asked.
4. Inputs to S3 (§9.1), preflight stamp, reaper deployed, `plan`, price rule, `offers.json`, journal entry with §5 restated and the chosen offer, then `rent`, on the user's go.
5. Watchdog under a Monitor; the first load's `samples/arms.jsonl` and `high_background.json` opened as soon as they sync: the prompt as forwarded, the marked token, a_P before and after, the five largest ablated drivers with their ā, none in B_t. Wrong input or format: destroy, journal, one debug, fresh prefix.
6. After `_rental_complete.json`: pull; timestamps after the rent; samples before any number, for the highest and lowest I′ per tercile, one fresh case per tercile, one pre-birth case with its ratios; journal entry restating §5; `verdict_birth.py` once at its committed SHA. That is the one unblinding.
7. A different model reads the artifacts under the result-review method, recomputes every rule under another seed, checks the exclusion flag on every set, and files `results/REVIEW_birth.md`. Opus executes 2 to 6, so Fable reviews.
8. `results/RESULTS.md` addendum "SPEC4 results", numbered Results in one declarative sentence each, the §5 table filled whatever landed, the limits of §14 as their own sentences; `POST.md` rewritten where §6 says; figures regenerated; commit of named files on the user's say; nothing pushed before the user's word.

## 13. The method note, if H_hub holds

One page, `studies/reuse/NOTE_high_background.md`: in a trained 7B model about 1% of upstream neurons carry a hundred times the typical background activation (its extreme top is the published super-activation set; the 1% set is not a published object; see §12.1); attribution-ranked driver sets of 50 admit two or three of them; mean-replacing them lowers any downstream neuron's activation whatever its drivers are, so a two-neuron null, a foreign control and the rise over training all read high at the final checkpoint and not at birth; excluding them (or matching on them) restores the controls. The numbers of M1, M2 and R1c15 against R1 and R1b, with the figure of §0's table before and after exclusion. It travels with the literature result of §12.1.

## 14. Limits known now

One model, one run; the fresh set is 93 cases on 59 neurons, read at birth only. The high-background cut at 1% is a definition, tested at one value for the falls. Part A's ratios are read on direct attributions and weights; attention-routed paths are not decomposed. The grid's late gaps make t_b−1 a coarse "before". The two-neuron sums keep their T_15 construction (`SPEC2.md` F7). vast.ai hosts are not the AWS host: the known-number gate at every load is the equivalence check between the two lanes, and the device name, driver and torch build are banked per load.

## 15. Grid, filled once

| row | predicted (§5) | observed | verdict |
|---|---|---|---|
| gate L | 1.0 [0.8, 1.2] | | |
| L1, primary | H_read | | |
| L1, fresh | H_read | | |
| room-c, room-c15 | 0.45; ≥ 0.20 | | |
| R1c, R1c15 | not refuted, not refuted | | |
| M1 | 0.02, inside the band | | |
| M2 | one null | | |
| R2c | specific | | |
| R3c | generic | | |
| S1 | not refuted, 0.11 against 0.05 | | |
| positive control, cancel share, identity, known number, completeness, exclusion, memory | as §7 | | |
| billed hours, dollars, offer rented | 0.75 h, $0.6 | | |

## Work queue after this spec, status 2026-11-03 05:20 CST (post-freeze record; the frozen text above is unchanged; the queue of record is `NEXT.md`)

| queue item | spec | status | result |
|---|---|---|---|
| this spec (the high-background term, fresh cases, Part A) | `SPEC4_birth.md`, tag `reuse-freeze-4` | run and closed 2026-11-02 | `results/REVIEW_birth.md`, Results 24–31 |
| 1. activation-based dating across the grid | `SPEC5_dating.md`, tags `reuse-freeze-5`, `reuse-freeze-5b` | run 2026-11-02; instrument artifact under the frozen text; the amendment exploratory | `results/REVIEW_dating.md`, Results 32–37 |
| 2. mean replacement with the norm scale held | `SPEC6_generic.md`, tag `reuse-freeze-6` | run 2026-11-02; instrument artifact; the norm path withdrawn as the mechanism | `results/REVIEW_generic.md`, Results 38–43 |
| 3. Qwen3.5-0.8B loop neurons always-on? | `../alwayson/SPEC_qwen_alwayson.md`, tag `alwayson-freeze-1` | run 2026-11-02; refuted | `../alwayson/results/RESULTS.md` |
| 4. the edit on the high-background set | the conditional test not run (item 3 refuted); its replacement run as `../alwayson/SPEC2_edit.md`, tag `alwayson-freeze-2`, with the post-freeze exploratory amendment 2b | run 2026-11-03; the frozen loop flag saturated at the 4,096-token cap in every arm (positive control R_1 0.025 against 0.15), E1 and E2 not read, a spec defect; under the registered exploratory repetition reading the always-on edit lowers the repetition share by 0.09 [0.00, 0.18] against the editor's 0.31 and a norm-matched random set's 0.05, not separated on either line; closed on that reading together with the weight-space share at the random level and SPEC1's refuted H_on; a confirmatory re-run is a successor item in `NEXT.md`, not launched | `../alwayson/results/REVIEW_edit.md`, `../alwayson/results/RESULTS.md` Results 7–12 |
| 5. weights-only driver prediction | folded into `SPEC6_generic.md` as rule W1 | read 2026-11-02; at the line (0.12) | Result 42 |
| 6. the remainder at the 3% cut | read inside `SPEC6_generic.md` | read 2026-11-02; not separated; the global 1% cut stays | Results 40–41 |
| 7. the birth as an object (third case set) | `SPEC8_birthobject.md`, tag `reuse-freeze-8` | run 2026-11-03; the drift branch | `results/REVIEW_birthobject.md`, Results 44–49 |
| 8. the low layers | `SPEC12_lowlayer.md`, tags `reuse-freeze-12a`, `reuse-freeze-12b` (a new harvest of 64 low-layer neurons on the banked contexts; the W4 clusterer by Sonnet subagents; one 16-load capture) | run 2026-11-03; 84 cases on 41 neurons; E1 not separated (0.44 [0.33, 0.55] against a within-set null's 0.405), the gain into the birth holds (1.37 [1.09, 1.75]; 1.09 deeper), E2 not refuted on its line (0.073 [0.005, 0.143]); H_token-event not separated, H_context-event not refuted; against SPEC5's deeper cases E1 and the gain differ (+0.14 [0.02, 0.26], +0.28 [0.01, 0.67]) and E2 and the locking do not; the drift conclusion not overturned | `results/REVIEW_lowlayer.md`, Results 59–65 |
| 9. the same-weights null, then the ablation against the pole's other clusters | `SPEC9_sameweights.md` (tag `reuse-freeze-9`), `SPEC10_sharedinputs.md` (tag `reuse-freeze-10`) | run 2026-11-03; the headline restated in measured terms | `results/REVIEW_sameweights.md`, `results/REVIEW_sharedinputs.md`, Results 50–55 |
| 10. the shared dependence at the birth checkpoints (siblings) | `SPEC11_sharedbirth.md`, tag `reuse-freeze-11` | run 2026-11-03; H_old-at-birth refuted on both sets; the old–new pairing dropped at the birth too | `results/REVIEW_sharedbirth.md`, Results 56–58 |
