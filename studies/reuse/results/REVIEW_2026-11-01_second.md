# REUSE second review, 2026-11-01 (closeout report checked against the bank)

Reviewer: Claude Fable 5.1, a later session than the one that wrote `REVIEW_2026-11-01.md`; same model, different context, own code (`scratchpad reuse_review/recompute.py`, reads `falls/*.npz` directly, bootstrap seed 7). Read-only: nothing launched, edited or acknowledged. Report read last.

**Verdict: every number in the closeout report reproduces from the banked falls, the frozen rules fired as written, and the process checks hold. The frozen verdicts stand: R1 refuted at birth, R2 specific, R3 consolidation, R4 holds. Two readings in the report go beyond what the bank shows: "related facets reuse at about half the ceiling" treats the related jobs as one population, and the bank shows a spread from the planted band to the ceiling; and "specific to O*'s drivers" rests on a foreign control that is as inert on J's contexts as the random control.**

## 1. Snapshot
Commits `ad2f8c82` (code, 15:55:35Z) → launch 15:58:32Z (`run.json`, EC2 LaunchTime) → first load 16:02:47Z → `STATUS exit=0` 16:29:18Z (S3 LastModified, checked) → `058265a0` closeout, `66a7bae5` operations, integrate `2667f612`. Journal order: predictions restated (entry 421) before the one unblinding (448). `pg audit` clean at review time. 9 loads is right: no primary job is born at T_4, T_5 or T_14 (`primary_items.json → birth`: 6:2, 7:3, 8:9, 9:18, 10:13, 11:28, 12:42, 13:35). T_15 ran first because §2 chooses the foreign neuron Q at T_15 and the birth loads read it from that step (`--from-step`, `foreign.json` only); drivers are re-ranked at every load (own-set overlap birth vs T_15 median 0.32, range 0.00–0.74).

## 2. Verified, reviewer's seed 7 against the report
| quantity | report | reviewer |
|---|---|---|
| live at birth, both, T_15; cancelled | 148, 147, 149; 2 | same (`L04N10274+:c2`, `L19N1554+:c5`) |
| R1 median I at birth | 0.226 [0.170, 0.308] | 0.226 [0.173, 0.305] |
| R2 cross − foreign | 0.084 [0.060, 0.122] | 0.084 [0.060, 0.121] |
| R3 I_T15 − I_birth | 0.098 [0.068, 0.128] | 0.098 [0.068, 0.128] |
| R4 related − unrelated; medians | 0.227 [0.133, 0.297]; 0.382, 0.155 | 0.227 [0.136, 0.290]; 0.382, 0.155 (56, 92) |
| positive control own fall; random; foreign; cross | 0.528; 0.0065; 0.010; 0.113 | 0.529; 0.0065; 0.010; 0.113 |
| identity max abs fall, 9 loads | 0.0 | 0.0 |
| I at T_15 | 0.375 | 0.375 (149) |
| I_ceil at birth, own splits | 0.653 (138) | 0.653, q25 0.428 |
| paired ceiling rise, 15 splits | 0.142, 13 of 17 | 0.711 → 0.788, 0.142, 13 of 17 |
| room at birth | 0.195 | 0.653 − 0.459 = 0.194 (p95 from `verdict.json`) |
| GPU-h | 0.513 | 1,846 s = 0.513 |
Planted 0.174 [0.117, 0.221]: verified at freeze from the S3 bank, not recomputed again. Known-number and completeness gates: read from `metrics.json`, not recomputed (need the model). API $0: a non-event, no artifact can show it; not verified.

## 3. Findings the report and the first review did not contain
**F1. The related jobs are not one population.** I at birth over the 56 related live jobs: q10 0.08, q25 0.22, median 0.38, q75 0.59, q90 0.71; 32% above 0.5 (where 67% of the own ceilings sit) and 34% inside their tercile's planted band. Unrelated: median 0.16, 8% above 0.5, 58% inside the band. s(J) related q25–q75 0.14–0.35 against the ceilings' 0.48. Two readings fit these medians equally: related facets reuse the old job's drivers at half strength, or a third of the related pairs are one job that UNITE's clusterer split in two and the rest are separate jobs. The sentence "related facets reuse at about half the same-job ceiling" picks the first without a test. Cheapest check, offline, no GPU: within the related set, I against s, and the shared-driver identities of the related jobs with I > 0.5 against their own ceiling split.

**F2. The foreign control had little to beat.** Foreign fall median 0.010 against random 0.007: Q's drivers, chosen on Q's contexts, are nearly inert on J's contexts, so R2 "cross − foreign > 0" carries the same information as "cross > random". R2 stands as frozen. What it does not test is whether O*'s drivers beat upstream neurons that are active on J's contexts but chosen for another neuron; that is the control UNITE's review asked for on the Gram side and it was not run here. Cost to run: one arm at 9 loads, about 0.8 GPU-h.

**F3. Room at the decision checkpoint is under the line.** Confirmed from the ingredients: ceiling at birth 0.653, planted 95th percentile 0.459, room 0.194 against the frozen 0.200. The room gate was read at T_15 (0.329) and passed as frozen; at birth, where R1 is read, the instrument's separation is at the line the spec set for stopping. The first review's D2 states this; it is the strongest caveat on R1 and belongs in the first sentence of any write-up beside "refuted".

**F4. The rise over training is not "driver concentration".** Own fall, the drop when a job's own 50 drivers are held at their means, is 0.529 at birth and 0.526 at T_15 over the 147 jobs live at both (paired median −0.009; 70 of 147 rise). What rises is the cross fall, 0.112 → 0.182, and the ceiling's cross fall. So `RESULTS.md` Result 3 ("the growth is in driver concentration generally") and the first review's D1 mechanism ("at birth the top-50 drivers of any job are less concentrated") are not what the bank shows: a job's own drivers carry the same share of its response at both checkpoints, and what grows is how much one context set's drivers do for another context set of the same neuron. The verdicts are unchanged; the mechanism sentence should be retracted in the write-up.

Also verified for the plain-language summary: related against unrelated median I holds in each tercile (0.33/0.21, 0.59/0.14, 0.34/0.15) and at each birth step where both labels occur (T_8 to T_13; T_6 has one job of each, T_7 has no unrelated job); 104 distinct neurons carry the 150 jobs; unrelated cross − foreign 0.059 with cross 0.085.

## 4. Reviewer's own errors
SPEC §13.1, written by this reviewer, says "loads in ascending order" and "11 loads"; §2 fixes Q at T_15, so T_15 must run first, and only 9 checkpoints have births. The executor followed §2. Noted here; §13 is not edited.

## 5. Publication check
One search, 2026-11-01, "ablating upstream neurons across training checkpoints, new feature forms, reuses existing inputs": nothing that ablates an older job's drivers and reads a new job at its birth checkpoint. Closest, arXiv 2605.18789 (feature life history across Pythia checkpoints, cross-layer ablation), already in the first review. Not swept: arXiv full text, X.

## 6. Disposition
Frozen verdicts R1–R4 and all gates: reproduced, stand. Report's "related facets reuse at half the ceiling": one of two readings, F1. Report's "specific": as frozen, with F2 beside it. Result status: closed under the frozen rules; the two readings above go to the next spec's Phase 0 with F1's offline check first.
