# REUSE amendment 2: the checks that stand between the closed result and a post

Status: **amendment 2, 2026-11-01, rules fixed; freeze 2 is the commit after the §3 code passes its dry run, tag `reuse-freeze-2`, before any forward or API call.** Parent: `SPEC.md` (frozen `reuse-freeze-1`, amended `reuse-amendment-1`), `results/RESULTS.md`, `results/REVIEW_2026-11-01.md`, `results/REVIEW_2026-11-01_second.md`. Written by the coordinator (Fable 5.1). No rule of `SPEC.md` is changed; every rule here is new and is named with a letter suffix. Disclosure: the coordinator had seen the primary bank before writing §2 (the related-job quantiles, the own-fall medians, the s medians); the thresholds in §2 are set from the reference distributions, not from the related jobs, and §2 is labelled exploratory with a pre-stated rule. §3 and §4 read data that does not exist yet.

## 1. Why, and what each outcome changes

The closed result has three sentences a post cannot carry as written. (a) R1 "refuted" is read at birth against a reference measured at T_15, and the room at birth is 0.194 against the 0.200 line (`REVIEW_2026-11-01_second.md` F3). (b) "Specific" rests on a foreign control that is inert on the new job's contexts (F2). (c) The related jobs are read as one population reusing at half strength, and their I runs from the planted band to the ceiling (F1). A fourth sentence, "driver concentration grows", is contradicted by the bank (F4) and is corrected, not tested.

| task | question | if yes | if no |
|---|---|---|---|
| A1 correction | — | — | — |
| A2 mixture | are related pairs two populations? | post says "a new task is either the same task or separate wiring" | post says "related pairs reuse at half strength on the median" |
| A3 convergence | do driver sets converge over training? | post carries it as an observation with a number | post drops the sentence |
| B1 birth reference | does the fake-neuron reference at birth leave room, and does R1 hold there? | post headline stays "no reuse at birth for unrelated pairs" | post headline becomes "the instrument cannot separate at birth; at T_15 real pairs sit above the reference" |
| B2 active control | do O*'s drivers beat neurons active on J's contexts? | "specific" stays | "specific" is dropped; R2 is reported as "beats random and foreign only" |
| B3 ceiling rise on all splits | is the R3 rise matched by the same-job rise? | R3 reported as generic | R3 reported as reuse growing over training |
| C judge labels | does R4 survive labels not from the clusterer? | R4 reported with both label sources | R4 reported as clusterer-dependent |

## 2. Offline tasks, 0 GPU, on the primary bank

**A1. Correction.** A dated note at the top of `results/RESULTS.md` Result 3 and an addendum line in `results/REVIEW_2026-11-01.md` D1: own fall 0.529 at birth and 0.526 at T_15 over the 147 jobs live at both (paired median −0.009, 70 of 147 rise); cross fall 0.112 → 0.182. "Driver concentration grows" is retracted; "what one context set's drivers do for another context set of the same neuron grows" replaces it. Figures unchanged (they plot I). Artifact: `results/REVIEW_2026-11-01_second.md` F4 and the script that recomputes it, copied into `scripts/review/second_review.py`.

**A2. Mixture test on the 56 related live jobs.** Split at I* = (planted median 0.174 + own-ceiling median at birth 0.653) / 2 = **0.414**, the midpoint of the two references, set before looking at where the related jobs fall relative to it. High group: I ≥ 0.414; low group: I < 0.414. Statistic: median s(J) in each group, bootstrap over neurons, 2,000 draws, seed 0. References: s of the 138 own ceilings at birth (median, interval over splits) and s of the 92 unrelated live jobs (median 0.10). Rule A2: **mixture** if the high group's s interval contains the ceilings' s median and the low group's s interval contains the unrelated s median; **partial reuse** if the high group's s interval lies below the ceilings' s interval; **neither** otherwise, reported as such. Second statistic, reported with no rule: Spearman correlation of I and s within the related set, with the same correlation within the unrelated set and within the ceilings. Failing input: the same rule on the 92 unrelated jobs must not return "mixture" (their high group is 8% of 92 and its s is the test).

**A3. Convergence.** Paired difference s_T15 − s_birth over the 147 jobs live at both, median, bootstrap over neurons; the same over the 17 ceiling reads (15 splits) with S3's T_15 side. Rule A3: **converge** if both intervals are above 0; **no change** if either contains 0; **diverge** if either is below 0. Reported beside it: the paired cross-fall rise 0.112 → 0.182 with its interval, and the own-fall difference of A1.

Both run from `scripts/review/second_review.py`, which prints each rule's input and verdict; outputs `results/primary/offline2.json`.

## 3. One GPU job: birth reference, active control, ceilings at T_15

Instrument: the primary stage of `infra/aws_pg/jobs/reuse_ablate.py` as frozen (m 50, direct ranking, mean replacement, cap 12 by rank, cancel rule, known-number and identity gates at every load). Three additions, no change to any existing arm:

**B1. Planted pairs at the birth checkpoints.** The 56 planted pairs of S3 (`inputs/setting_items.json → planted`), at each load T_6 to T_13: identity, own D(side 1), cross D(side 2), random-matched R, on full sides capped at 12, no count matching (its 40 draws per pair are what made S3 cost 0.378 GPU-h; the full-side and count-matched medians at T_15 are both in `results/sniper/S3.json` and the full-side value is the one paired here). Reference at birth: every pair's I at every birth load, weighted so each birth checkpoint carries the primary set's live-job count at that checkpoint (`verdict.json → count`) and each tercile carries one third, as SPEC §6; median with bootstrap over pairs within tercile; 95th percentile under the same weights. Gate room-b: median I_ceil over the 138 own birth splits (0.653) minus the planted 95th percentile at birth ≥ 0.200; failing input: planted side labels shuffled within pair at the same loads (own and cross only). Rule R1b: if room-b passes, R1 as frozen (overlap of the two medians' intervals) with the birth reference in place of the T_15 one; if room-b fails, R1b is **cannot separate at birth**, and the post reports the T_15 comparison (I 0.375 [0.303, 0.446] against 0.174 [0.117, 0.221], no overlap) as the only reading with room.

**B2. Active control R_J.** For each job at its birth load and at T_15: for each k in D(O*), the neuron of the same layer, not in D(J) ∪ D(O*), nearest to k in mean |a_k| at the marked token over J's contexts, read from the identity forward (all MLP layers below L at the marked position; one tensor per context). Count- and energy-matched on J's contexts by construction; deterministic. One arm, 12 forwards per job per load. Rule R2b: median over live jobs of Δ(D(O*); J) − Δ(R_J; J), bootstrap over neurons: interval above 0 → **specific beyond activity**; otherwise **not specific beyond activity**, and the post reports R2 as "beats random and foreign only". Failing input: R_J in place of own must fail the positive control (median fall under 0.25). Also read, no rule: R_J's absolute fall against the random-matched R's (0.007) and the foreign fall (0.010).

**B3. All 138 own ceiling splits at T_15.** Side 2 read with identity, own D(side 2), cross D(side 1). Rule R3b: paired median of I_ceil(T_15) − I_ceil(birth) over the 138 splits, bootstrap over neurons, against R3's 0.098 [0.068, 0.128]: if the interval of (R3 rise − ceiling rise), bootstrapped over neurons jointly, is above 0 → **reuse grows beyond the same-job rise**; contains 0 → **generic**; below 0 → **the same-job rise is larger**.

Loads: T_6 to T_13 and T_15, 9 loads at 0.06 GPU-h each; forwards about 56 × 4 × 24 × 8 (B1) + 150 × 12 × 2 (B2) + 138 × 3 × 12 (B3) ≈ 52,000. **Estimate 1.0 GPU-h, budget 120 min, hard cap 1.5 GPU-h.** Study total after this job at most 2.6 of 5.0.

Stage `--stage amend2`, steps file `infra/aws_pg/steps/reuse_amend2.json`, T_15 first (Q and the ceilings' T_15 reads), then T_6 to T_13. Verdict script `scripts/verdict2.py`: reads the new bank and the primary bank, computes room-b, R1b, R2b, R3b and the no-rule items, writes `results/amend2/verdict2.json` and `GRID2.md`, every number with its artifact path. Dry run as SPEC §13.1 item 3: on the primary bank with the new arms absent it must stop, not fill; on the sniper S3 bank it must reproduce the full-side planted median at T_15 within 0.01; failing inputs: shuffled planted labels must fail room-b; R_J in place of own must fail the positive control.

## 4. Judge labels, API, ≤ $15

Through `infra/pg/neurons_llm.py` on the anthropic-direct path, settings `infra/pg/settings/anthropic_sonnet55.json`, effort high (SPEC §12: granularity-sensitive tasks run at high), Message Batches, `--dry-run` first and its request count and token estimate written to the journal. Items: the 150 primary pairs (J's 12 contexts and O*'s 12 contexts as text with the marked token bracketed, order of the two sets randomized per item, seed 0) plus the controls SPEC §12 fixed: 30 planted-pair sides (unrelated by construction) and 30 ceiling-split sides (related by construction), blind in the same batch. Question, fixed: "Are these two sets of passages about the same kind of thing? Answer related or unrelated and one sentence why." Gate judge: at least 0.85 of each control group labelled as constructed; otherwise the labels are not used and the post says so. Rule R4c: R4 as frozen under the judge labels (median I related − unrelated, bootstrap over neurons, interval above 0 → holds). Reported: agreement with the clusterer labels over the 150, and R4 on the pairs where both sources agree. Refusals are declines, not relabelled. Nothing is submitted before the user's go; one batch; cap $15.

## 5. Point predictions, logged here before any new data, no rule attached

| quantity | predicted | refuted if |
|---|---|---|
| planted median at birth, pooled | 0.14 [0.09, 0.19] | outside [0.05, 0.25] |
| room-b | 0.25 | under 0.20 |
| R1b | refuted (overlap) | not refuted |
| R_J fall median | 0.02 | over 0.06 |
| R2b | specific beyond activity | interval contains 0 |
| ceiling rise over 138, paired median | 0.10 [0.05, 0.16] | under 0.03 or over 0.20 |
| R3b | generic | reuse grows beyond |
| A2 | mixture | partial or neither |
| A3 | converge, s rise 0.04 | no change |
| judge agreement with clusterer | ≥ 0.75 | under 0.65 |
| R4c | holds | interval contains 0 |

The post reports this table filled, landed or not. These are the program's first held-out predictions.

## 6. Order and gates

1. A1 now, by the coordinator; then A2 and A3 by script; `offline2.json` committed with the script. Journal entry.
2. One Opus worker, one session, own worktree: §3 stage, arms, steps file, `verdict2.py`, dry runs; §4 item builder and the `--dry-run` batch. Coordinator reviews the diff. **Freeze 2 commit**, tag `reuse-freeze-2`.
3. On the user's go: the GPU job (`--detach`, fresh prefix `results/reuse/amend2/<n>`, `metrics.json` only, first samples opened before the run continues, stop rules of SPEC §13.3) and the API batch. Journal entry at each launch with the §5 table restated.
4. After `STATUS`: samples before any number; `verdict2.py` once; `collect-batch`; R4c.
5. Different-model read (Opus executes, Fable reviews), `results/REVIEW_amend2.md`.
6. `results/RESULTS.md` addendum "Amendment 2 results", dated; §5 filled; fig1 regenerated with the birth reference band; one new figure, I_ceil at T_15 against birth over 138 splits with the diagonal null.
7. Post draft `studies/reuse/POST.md`, §7 below. Commit on the user's say; nothing pushed.

## 7. The post

One question, one figure (fig1 with both references at the checkpoint the jobs are read at), 600 to 900 words. In order: what was measured, in the plain words of the checked summary with its seven corrections applied; the unrelated-pair null with both references and the room number in the same sentence as the verdict; the related split with the A2 outcome named; the rise over training with A3 and R3b outcomes and no mechanism; the §5 table; limits in their own sentences (one model, one run, labels, attention not measured, mean replacement); Eliwat credited with "agrees with", not "reproduces"; a link to `SPEC.md`, `SPEC2.md`, the verdicts and the bank. No sentence a frozen rule did not produce or a limit does not qualify.

## 8. Budget

GPU: 1.0 estimate, 1.5 cap. API: ≤ $15 of the $200 (expires 2026-11-15); $0 until the user's go. Sessions: one Opus worker (§6.2), one Opus executor (§6.3–4), coordinator for review and the post; never more than two agents at once.
