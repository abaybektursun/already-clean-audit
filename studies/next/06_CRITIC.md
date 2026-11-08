# 06 — Adversarial read of the picks (Program A, Program B, N22)

Reviewer: Claude Opus 5.5, a different model from the author. 2026-11-05. Read-only. No model was run. I wrote
nothing in the repository except this file. The computations used banked JSON and npz files on the laptop.
Small API reads (the W&B GraphQL API, Hugging Face refs, arXiv abstracts) went to the scratchpad.

Read in full:
- `README.md`, `02_FRAMES.md`, `03_IDEAS.md` (as of 23:15), `05_SCORING.md` and `04_LITERATURE.md`.
- `04_PRACTICAL.md` §5 and §6.
- `00_ASSETS.md` §1 to §6 and its corrections. `00b_DATA_ORDER.md`.
- `01_JLENS.md` §5 and §7.
- `studies/unite/CRITIC_2026-10-31.md` lines 1 to 60, and the repo rules.

`04_RECENT.md` and `04_WDD.md` did not exist when I started (checked 23:11). They existed when I reached the
scoring stage (23:19), and I read both in full. `05_SCORING.md` was not updated after they landed (file time
22:59).

**Answer.** Program A's unit, the dated birth, is mostly set by how many top-50 positions a cluster holds and
how high they rank, not by its job. I regrouped each pole's top-50 positions at random into clusters of the
real sizes. Three results:
- 96% of the birth histogram stays in place. The total-variation distance to the real histogram is 0.040;
  random regroupings differ from their own mean by at most 0.028 in 300 draws.
- The share of poles whose main job is born first rises from the real 0.476 to 0.530 [0.491, 0.568].
- Only 148 of the 1,238 primary late-born jobs (12%) are born later than 95% of random same-size groups from
  their own pole.

What follows from this, one line each:
- **N20.** The proposed rule has a free threshold that decides the headline. The share of poles "born" by step
  180,000 is 0.70, 0.18 or 0.009 for thresholds of 1, 10 or 25 of 50 positions.
- **N27.** The banked data predict that its own third refutation fires: mismatched births would predict as
  well. Its benchmark curves are already public in W&B, at 0 GPU-h.
- **N4.** It cannot name "the documents that made the job" at 7B. Stage 1 is one uniform shuffle, so the
  documents of one interval are exchangeable with the next interval's. The gradient cosine is forced by the
  job's own inputs. The lithium job, its first target, has no D1 birth date.
- **N1, N2, N12.** As written, they measure AdamW rotation rate, selection on activation and selection on
  the winner.
- **Program B.** N28 has no statistical power on about ten nested stages. N19's premise is unmeasured on
  OLMo-3. N13 is the card to run first.
- **N22.** It survives with three additions: an accuracy gate, a lens-free positive control and a slot
  comparison at matched depth.

I found no card number that disagrees with its artifact. The defects are in the designs. One closed result
(D1 birth order) used a reference distribution that is too weak (Finding 0.2).

**Scripts and outputs.** `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/crit06/`:
- The scripts: `load.py`, `pseudo.py`, `eta.py`, `tv.py`, `percluster.py`, `top50rule.py`,
  `age_energy.py` and `lithium.py`, each with its `*_out.txt`.
- The API reads: `wandb_keys.json`, `wandb_runs.json` and `hf_refs.json`.

The arXiv abstracts are in `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/crit06_dl/`.
`BANK/` means `/Users/abaybektursun/projects/interp_research_banks/`.

---

## Part 0. The birth instrument, which every Program A card uses

### Finding 0.1. Random regroupings of the same positions reproduce the birth histogram. Severity: blocks the plan.

**Evidence.** The D1 rule is in `studies/already-clean/SPEC_D1.md` §2.4 and in `presence` and `birth` of
`infra/pg/neurons_verdict.py`. It reads as follows:
- **Presence.** A cluster is present at point t when the median, over its T_15 exemplar positions, of the
  pole-direction activation at t is at least the pole's own 99.9th percentile at t.
- **Birth.** Birth is the first t at which the cluster is present at t and at t+1.

I recomputed every cluster's medians and births:
- **Inputs.** The exemplar activations are in `BANK/neurons/w3/grid/*/out/*/poles/exemplar_a.npz`. The `q999`
  values are in each point's `samples/contexts.jsonl`.
- **Check.** I compared the result with `BANK/neurons/w5/verdict/samples/q2_clusters.jsonl`. There were 0
  mismatches over 2,454 non-echo clusters, and b = 0.476 was reproduced.
- **Random groups.** For each pole, I shuffled its non-echo top-50 positions into groups of the real cluster
  sizes and applied the same rule. `pseudo.py` used 200 draws and `tv.py` used 300.

| point | step | share of real clusters born here | share of random groups born here |
|---|---|---|---|
| T_4 | 5,000 | 0.012 | 0.010 |
| T_5 | 8,000 | 0.020 | 0.020 |
| T_6 | 12,000 | 0.025 | 0.022 |
| T_7 | 20,000 | 0.040 | 0.044 |
| T_8 | 35,000 | 0.065 | 0.065 |
| T_9 | 60,000 | 0.072 | 0.077 |
| T_10 | 100,000 | 0.097 | 0.107 |
| T_11 | 180,000 | 0.141 | 0.159 |
| T_12 | 320,000 | 0.190 | 0.187 |
| T_13 | 560,000 | 0.199 | 0.200 |
| T_14 | 1,000,000 | 0.117 | 0.099 |
| T_15 | 1,413,814 | 0.015 | 0.005 |

The split by job type, from `pseudo_out.txt` (real counts against random means):
- **Main jobs, T_10 to T_14:** 74, 89, 97, 109, 47 against 71.4, 95.1, 105.0, 111.3, 47.4.
- **Other jobs, T_10 to T_14:** 164, 257, 370, 380, 239 against 191.8, 294.4, 352.8, 380.4, 195.9.

The real data do differ in one place. Real clusters are late-born more often than random groups: 1,535
against 1,282 [1,238, 1,329], counted before the B3 and B4 filters. The excess comes mostly from extra births
at T_14 and T_15: 268 non-main births at those two points, against 205 for random groups.

**Why it happens.** Exemplars are chosen at T_15 as the pole's largest activations, then read backward in
time.
1. Going back, each position's activation decorrelates from its T_15 value.
2. A group's median falls below the 99.9th percentile once enough of its members have decorrelated.
3. When that happens depends on the group's size and on how far above the threshold its members sat at T_15.
   It does not require the members to share a job.

The top-50 set itself turns over by about 80% between neighbouring banked points. The median overlap of two
adjacent points' top-50 sets is 9 to 13 of 50 at every interval from T_4 to T_13 (`top50_out.txt`).

**Consequence.** A count of births per point (N20, N27) cannot be read as a count of job-formation events.
The per-birth cards (N1, N2, N3, N4, N12) sample from all 1,238 late-born jobs. Most of those births are dated
where a random group of the same size would also be dated.

**Fix.** Make the random-group reference part of the instrument. Report each birth with its one-sided p
against same-size random groups from the same pole, and run the per-birth cards on the jobs that pass
(Finding 0.3).

### Finding 0.2. The closed D1 birth-order result falls below the random-group reference. Severity: changes a number.

**Evidence.** `BANK/neurons/w5/verdict/metrics.json` → `q2.birth_order` gives b 0.476 [0.436, 0.517] on 574
poles. It was read as "main is oldest" against a within-pole permutation of birth ranks (mean 0.414, 97.5th
percentile 0.449).

That permutation ignores how `main` is defined: the cluster that holds the pole's largest T_15 activation.
That cluster has the highest median, so the mechanism of 0.1 gives it the earliest birth.

Under random regrouping at the same sizes, b is 0.530 [0.491, 0.568] (`pseudo_out.txt`). The real value lies
below the 2.5th percentile.

**Consequence.**
- The support for "the main job is oldest" comes from how `main` is chosen.
- The claim's owner should retag it, per the repo's Provenance rule 2.
- Program A's story that "offshoots are younger" should not cite it.

**Fix.** Use the random-group reference for any birth-order statement.

### Finding 0.3. Job-specific timing exists, but it is the smaller part of the signal. Severity: changes a number.

**Evidence, by position (`eta.py`).** I dated each T_15 exemplar position on its own, with the same rule. I
then measured how much of the variance of these position births the real cluster labels explain (eta squared)
and compared that with random same-size labels. This covered 574 poles with two or more clusters, 200 draws
each.
- Real labels explain a mean of 0.221 (median 0.163).
- Random labels explain a mean of 0.066 (median 0.060).
- For 0.505 of poles, the real eta squared exceeds 95% of random labelings.

**Evidence, by cluster (`percluster.py`).** Of the 1,238 primary late-born clusters, 148 (0.120) are born
later than 95% of random same-size groups from their own pole. If there were no job timing at all, the
expected share would be at most 0.05, about 62 clusters. The share born later than the median random group is
0.519.

**Consequence.** Clusters carry real timing information. About 150 late-born jobs are candidates for a birth
event. The other 1,090 or so are dated about where a random group would be.

**Fix.**
- Draw Program A's "100 dated births" from the 148.
- Before freezing that set, apply the raw reading and the B4 match to the random groups as well. This review
  applied only the percentile reading to them.

### Finding 0.4. The README defines a birth with the wrong rule. Severity: cosmetic, but it propagates into N2 and N20.

`README.md` "Vocabulary" says a birth is "the first of the 16 training points at which a job's contexts are
among the pole's top 20". The rule actually used is the 99.9th-percentile rule of 0.1.
- Over 20.56M scored tokens, the 99.9th percentile is about rank 20,560.
- The rule also requires presence at the next point.

N2's phrase "activation already at its top-20 level" and N20's top-50 rule both inherit the wrong
definition.

### Finding 0.5. Activation scale falls about 500-fold across the grid. Severity: changes a number.

The fields `max_abs` and `q999` of `contexts.jsonl`, over the 640 grid poles:
- **Median max|a|:** 1,238 at T_3, 245 at T_8, 18.3 at T_10 and 2.46 at T_15.
- **Median 99.9th percentile:** 398 at T_2 and 0.355 at T_15.

In this architecture, the MLP input is the residual stream with no norm in front. The norm comes after the MLP
(`infra/pg/instruments_adapter.py` line 5: "the MLP reads h_L itself (no pre-MLP norm)"). So any raw
cross-point comparison of activation times
gradient, gradient size or weight-row angle mixes the job with this trend. Compare within each point instead,
by rank or percentile among the layer's poles at that point.

### Finding 0.6. The lithium neuron has no dated birth, and its onset bracket is 15,000 steps. Severity: blocks the first measurement of N1 and N4.

**The lithium count.** The count of " lithium" in the top-50 of pole L03N6372+ at each point
(`lithium_out.txt`):

| points | steps | " lithium" in top-50 |
|---|---|---|
| T_0 to T_5 | 0 to 8,000 | 0 |
| T_6, T_7 | 12,000, 20,000 | 1, 1 |
| T_8 | 35,000 | 25 |
| T_9 to T_12 | 60,000 to 320,000 | 47, 40, 44, 40 |
| T_13, T_14 | 560,000, 1,000,000 | 6, 7 |
| T_15 | 1,413,814 | 0 |

**Why it has no birth date.**
- At T_15 the clusterer found 15 jobs on this pole, and none is about lithium
  (`studies/already-clean/writeup/stories/A_training_dynamics.md` §2).
- The D1 rule reads only T_15 exemplars, so the lithium job has no D1 birth date.
- By the top-50 count, its onset lies in (20,000, 35,000]. It left the pole between steps 320,000 and 560,000.
- The pole's max|a| falls from 143.9 at T_5 to 0.246 at T_15.

**A second banked instance.** The same pole's top tokens include licence words at many points:

| points | token |
|---|---|
| T_2, T_3 | " licensed" |
| T_5 to T_9 | " Licensed" |
| T_11 | " Licensing" |
| T_13 | "/licenses" |
| T_15 | "/MIT" |

D1 dates this pole's main job, "MIT license / opensource URL tokens", at T_13. It does so because the rule
tracks the T_15 positions, not the token family. That these tokens form one lineage is my reading; I did not
measure it.

---

## Part 1. Program A

### N20 — the census of all poles

1. **Identifiability. Severity: blocks the plan.**
   - **What it measures.** The proposed quantity is the first point at which the T_15 top-50 positions appear
     in the point's top-50. On the 640 grid poles, the median overlap with T_15 is 0 of 50 through T_9. It is
     then 1, 2, 6, 11 and 20 at T_10 to T_14 (`top50_out.txt`). The overlap rises because each point is
     closer to the end, and a pole with no job event shows the same rise (0.1).
   - **The threshold decides the claim.** Share of poles born by T_11 (step 180,000, the last point before
     20% of training):

     | threshold (of 50) | share born by T_11 |
     |---|---|
     | 1 | 0.702 |
     | 10 | 0.181 |
     | 25 | 0.009 |
     | 40 | 0.000 |

     The claim "most poles have their final job before 20%" holds at threshold 1 and fails at 10. The
     refutation "more than a third change job in the last half" fires at 25, where 565 of 640 poles are
     "born" at T_14 or T_15. It does not fire at 1, where the count is 14 of 640.
   - **Depth order.** The depth order also has a known cause with no job event: layers converge to their
     final representations from the bottom up (SVCCA, https://arxiv.org/abs/1706.05806, abstract).
   - **The D1 rule cannot run on CPU for all poles.** It needs each pole's activation at its T_15 exemplar
     positions at every point. That array, `exemplar_a`, is banked for the 640 grid poles only
     (`00_ASSETS.md` §1.3). For the other 703,872 poles, S3 holds each point's own top-50 and the histograms.
     On a CPU, only the overlap rule above can be computed.
2. **Refuting observation. Severity: changes a number.** The refutation can fire, but whether it fires
   depends on a threshold that is not frozen. Three things are missing:
   - **An identity pair.** Harvest two adjacent late checkpoints, 1,000 steps apart, on the same corpus.
     Their overlap is the ceiling for "no event".
   - **The random-group reference** (0.1).
   - **A validation that does not come from the same rule family.** Calibrating against the D1 dates does not
     validate the rule. Both select at T_15 and read backward.
3. **The number. Severity: cosmetic.**
   - Divide births per point by each point's width in ln(step), which is 0.35 to 0.69 on this grid.
   - "The share that change job after step 100,000" uses the word "job". Jobs exist only for the 640
     clustered poles.
4. **Already done. Severity: changes a number.**
   - **In this repo.** The 600-pole sample is uniform and stratified by layer. It already estimates the
     population shares under the D1 rule: 340 of 600 main jobs are born by T_11, and 546 by T_13
     (`q2.births`). What the census adds is per-layer cells. The sample, at 16 to 20 poles per layer, cannot
     fill those.
   - **Elsewhere.** SAE features in Pythia-160M and 410M emerge and die about 40 times faster in the first 1%
     of training (https://arxiv.org/abs/2605.18789, abstract, per `04_RECENT.md` §4). wdd tracked every output
     vector of several blocks at 16 checkpoints between steps 1,000 and 16,000
     (https://github.com/sunmoonron/wdd-descendants, per `04_WDD.md` §6).
5. **The demo. Severity: blocks the plan.** A 32 × 16 map titled "births" would show the chosen threshold and
   each layer's drift rate. Fix:
   - Title it by its rule ("first point at which k of the final top-50 are in the point's top-50").
   - Print k.
   - Show it beside the identity-pair map and the random-group map.
6. **Cost. Severity: changes a number.**
   - **The overlap rule** costs 0 GPU-h: a 30.9 GB read from S3 on a CPU instance.
   - **The D1 rule on all poles** needs exemplar activations at 15 points. At the measured 3,234 to 4,623
     tokens per second on 20.56M tokens (`00_ASSETS.md` §1.1), each harvest is 1.2 to 1.8 GPU-h. That is
     about 22 GPU-h, plus 15 downloads of 14.6 GB.
   - **The identity pair** adds two harvests, about 3.5 GPU-h.

### N27 — births lead benchmark rises

1. **Identifiability. Severity: blocks the plan.**
   - **The third refutation is the expected outcome.** By 0.1, per-point birth counts are mostly set by cluster
     size and rank. So the cumulative curve of any subset of births follows the cumulative curve of all births.
     Matched and mismatched subsets would then lead or lag the benchmark by the same amount, which is the
     card's third refutation.
   - **"Lead" is undefined.** Between two monotone curves, "lead" has no fixed meaning until both are put on
     one scale, for example the time each takes to reach half its own range.
   - **The birth grid limits resolution.** The birth curve has 16 points. After step 100,000 its intervals are
     80,000 to 440,000 steps, so a lead shorter than one interval cannot be resolved.
   - **The content match has no agreement number.** The rule matching tasks to jobs would read the clusterer's
     job names, and no second-reader agreement exists for those names. The only agreement banked for O7 is on
     the pole classes: κ 0.070 (`metrics.json` → `q1.O7`). That is below the 0.2 floor in `02_FRAMES.md` §6.
2. **Refuting observation. Severity: changes a number.** The refutation can fire, and the third clause
   probably will. Missing:
   - the random-group births, registered as a comparison;
   - a permutation reference for the 0.05 R² bar (15 tasks give a wide spread of partial R²);
   - a ceiling.
3. **The number. Severity: changes a number.**
   - The card does not say what R² is computed on: per task over checkpoints, or across tasks.
   - Births are counts and the benchmark is in bits per byte. One normalization must be stated for both.
4. **Already done. Severity: changes a number.**
   - **The benchmark curves are already public, free.** The W&B project `ai2-llm/Olmo-3-1025-7B` logs 26
     downstream evaluation entries every 10,000 steps, from about step 10,000 to step 1,413,814 (the first
     run, which ends at step 9,809, logs none). Several benchmarks appear in two formats. Every entry is
     logged in bits per byte; the multiple-choice entries are also logged as CE loss and accuracy. The
     benchmarks are:
     - ARC;
     - MMLU by subject group;
     - HellaSwag;
     - six "basic skills" sets;
     - HumanEval;
     - MBPP and three multilingual MBPP sets;
     - Minerva MATH;
     - copycolors.

     The project also logs 11 validation-set CE losses and the learning rate. Source: GraphQL queries of the
     run list and history keys (`wandb_keys.json`, `wandb_runs.json`); the evals are 10,000 steps apart in
     run `zckrmllo`.
   - **The validation-loss baseline is strong.** Downstream scores follow pretraining loss across model and
     data sizes (https://arxiv.org/abs/2403.15796, abstract). Continuous likelihood metrics are predictable
     from small runs (https://arxiv.org/abs/2504.11393, abstract).
   - **There are named predictors to beat.** `04_PRACTICAL.md` names two from 2026
     (https://arxiv.org/abs/2604.01025, https://arxiv.org/abs/2604.08510).
5. **The demo. Severity: changes a number.** "Births climbing, then the benchmark rising" shows two monotone
   curves, and a reader will read cause. The mismatched-births curve and the random-group curve belong in the
   same chart.
6. **Cost. Severity: changes a number.** The 6 to 10 GPU-h estimate falls to about 0 for the benchmark
   curves. The lead also needs denser birth dating, at 1.2 to 1.8 GPU-h per extra harvested point.

### N4 — the documents that made the job

1. **Identifiability. Severity: blocks the plan.**
   1. **The documents are exchangeable.**
      - Stage 1 is one epoch of one seeded uniform shuffle of a fixed mix (`00b_DATA_ORDER.md`). Every
        1,000-step interval is a random draw of 8,192-token windows from the same population.
      - So the documents in the birth interval are exchangeable with the documents of the next interval.
      - "These documents made the job" cannot be told apart from "documents of this kind, seen once the model
        was ready, make the job".
      - The counterfactual then predicts a delay of about one interval. That is the published result for
        heads: deletion delays emergence, and saturation converges (https://arxiv.org/abs/2601.21996, TEXT,
        per `04_LITERATURE.md`).
   2. **The cosine is forced.**
      - The gradient of one document's loss on the gate and up rows is a sum over its positions: a scalar
        times the residual input at that position.
      - A document that contains the job's token therefore adds a term along that token's input direction.
      - If the realized change makes the pole fire more on that token, the change also has a component along
        that direction.
      - So job-token documents align with the realized change because both are built from the same input
        vectors, not because these documents caused it.
      - Gradient similarity also tracks format more than content (https://arxiv.org/abs/2609.19589, abstract,
        per `04_RECENT.md`).
   3. **The first target has no birth.** The first measurement names the lithium job, which has no D1 birth
      date (0.6).
2. **Refuting observation. Severity: blocks the plan.** Because of the forced cosine, the cosine clause has
   little room to fire. The retrain clause runs on a different, small model. Two matched controls from the
   same operator are missing:
   - documents that contain the token at positions where the pole does not fire;
   - the same token's documents in an interval after the job already exists. If the cosine is the same there,
     it does not measure the birth.
3. **The number. Severity: changes a number.**
   - **Two different objects.** The card compares one document's raw gradient at the earlier checkpoint with
     the sum of 15,000 AdamW updates at a changing learning rate.
   - **A precedent.** In wdd's Pythia measurement, a row's 1,000-step update follows its own gradient at
     cosine 0.012 to 0.013
     (https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/26_drift_cause.md, REPO-DOC, per
     `04_LITERATURE.md`).
   - **A fix is available.** The stage-1 checkpoints include optimizer state (`model_and_optim/` shards, per
     the corrections in `00b_DATA_ORDER.md`). That lets each document's term be put in update units.
4. **Already done. Severity: changes a number.**
   - Unit-level counterfactual retraining exists for heads at 14M to 160M (2601.21996).
   - A 7B counterfactual exists at behavior level: OLMo-3-7B mid-training retrained on edited data
     (https://arxiv.org/abs/2610.06851, full, per `04_RECENT.md`).
   - wdd's removal test on a proxy stream found no effect and was underpowered
     (https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/54_class_data_attribution.md).
5. **The demo. Severity: blocks the plan.** "The documents that taught it" would be a list of documents that
   contain " lithium". The next interval's list would look the same. The panel would imply a cause that the
   design cannot show.
6. **Cost. Severity: changes a number.** The "1 GPU-h per job" estimate assumes a 1,000-step bracket.
   - **The bracket is 15,000 steps for lithium.** That is 7,680,000 range requests, 251.7 GB, before any
     gradient is computed.
   - **Narrowing it.** Bisecting down to 1,000 steps needs about 4 checkpoint loads (14.6 GB each) and a
     forward pass on the lithium windows.
   - **The gradients.** The per-document backward pass scales with the number of instances that contain the
     token, which nobody has counted. At an assumed 1.2e14 FLOP/s on one L40S, 1,000 instances of 8,192
     tokens is about 0.8 GPU-h and 10,000 instances is about 8. The format-matched control set doubles this.
   - **Rate limits.** Cloudflare rate limits are untested (`00b_DATA_ORDER.md` risk 9).

### N1 — input first, readers later

1. **Identifiability. Severity: blocks the plan.**
   1. **AdamW sets the rotation.**
      - Under AdamW, the angle a weight vector turns per step is set mostly by the optimizer, not by the job.
        Weight decay drives neurons to a rotational equilibrium with similar angular updates across neurons
        and layers (https://arxiv.org/abs/2305.17212, abstract).
      - wdd measured that every row turns by a similar angle (consecutive cosine 0.82 to 0.88). It also found
        that a row's actual change has cosine 0.002 with the negative gradient
        (https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/25_training_motion.md, per
        `04_WDD.md` §2.2 and §2.5).
      - On OLMo-3-7B itself, stage-1 update windows from step 30,000 on have Fisher leverage at the
        random-direction level on prose (https://arxiv.org/abs/2610.37719, full, per `04_RECENT.md`).
      - So comparing the cosine of consecutive gate rows with that of consecutive down columns measures
        rotation rate. That rate differs between rows and columns by norm and fan-in.
   2. **The window misses the bracket.** The birth point is the upper end of a bracket. For births at T_11 to
      T_14, the bracket is 80,000 to 440,000 steps. The window, 1,000 steps before to 20,000 after the birth
      point, lies after most of the bracket. Together with 0.1, this means there may be no event in the
      window.
   3. **Norms drift.** Row norms and residual norms change with the 500-fold activation trend (0.5).
2. **Refuting observation. Severity: blocks the plan.**
   - **The problem.** If both ratios sit near 1 (equilibrium rotation), the majority rule reads at chance
     whichever side the job was written into. There is no floor, no ceiling and no positive control.
   - **Fix.** Measure the component of the realized change along job-defined directions:
     - For gate and up rows: the mean residual input at the job's exemplar positions, minus the mean at
       activation-matched non-job positions.
     - For the down column: the direction that the gradient of the job's readers favours.

     Express these in units of the same pole's non-birth intervals at the same learning rate (W&B logs the
     learning rate). Compare against 100 random directions, and use the public Adam second moment as a
     preconditioner.
   - **Positive control.** Plant a known rank-one change in one row.
3. **The number. Severity: changes a number.**
   - A gate row lives in input space and a down column in output space. Both have 4,096 entries, but their
     norms, gradient statistics and effective learning rates differ. Equal cosines do not mean equal change.
   - The same pole's "non-birth intervals" sit at different learning rates.
4. **Already done. Severity: changes a number.**
   - **wdd points both ways.** On Pythia the output vector is the moving part: row motion is 0.64 to 0.78 of a
     word's rise, and the later output vector makes a word in 0.59 of entrants against 0.18 for the input row.
     On OLMo-1B, the input side is selective eight intervals before entry (z 2.63)
     (https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/33_second_model.md, per `04_WDD.md`
     §6).
   - **The "readers later" part has an analogue.** A per-channel reader–writer sign forms in the first few
     percent of training, in a fixed depth order (https://arxiv.org/abs/2610.29340, full, per
     `04_RECENT.md`).
5. **The demo. Severity: changes a number.** Under equilibrium rotation, the three curves per neuron will look
   alike for every pole. With the birth marked at the end of the bracket, the marker invites reading noise as
   order.
6. **Cost. Severity: changes a number.**
   - **Columns are strided.** `down_proj.weight` is stored as [4,096, 11,008], so one neuron's column is 4,096
     values spaced 22,016 bytes apart. Each read takes the whole matrix of that layer, 90 MB in bf16.
   - **Readers are most of the file.** The readers are all later layers' gate, up and attention input
     matrices, which make up most of each 14.6 GB file.
   - **The windows.** 22 checkpoints for each of about 10 bracket windows is about 220 checkpoint reads, about
     3 TB, on a CPU instance.
   - **Locating onsets.** Bisecting inside a 240,000-step bracket takes about 8 checkpoint loads per job. For
     100 jobs that is about 800 loads, 12 TB and 25 to 40 instance-hours.

### N2 — silent or used

1. **Identifiability. Severity: blocks the plan.**
   - **The measure forces the reading.** Use is measured as activation times gradient, and on the job's
     contexts the activation is selected high:
     - The exemplars are the pole's largest activations at T_15.
     - At birth, their median is above the pole's 99.9th percentile by definition.
     - Random poles on the same contexts have small activations.

     So "use above the random-pole band at birth" follows from the selection alone, even if the gradient per
     unit activation is the same for both. The reading "no silent period" is produced by the measure.
   - **wdd has the same defect.** wdd's reader statistic multiplies in the word's own activation
     (https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/63_already_clean_growth.md, per
     `04_WDD.md` §4.7 item 5).
   - **Attribution patching is a poor estimate for MLP outputs.** It reaches Pearson 0.006 against activation
     patching in GPT-2 Large (RelP, https://arxiv.org/abs/2508.21258, abstract, per `04_LITERATURE.md`).
     Attribution patching has documented false negatives (https://arxiv.org/abs/2403.00745, abstract).
   - **Fix.**
     - Score the gradient per unit activation, which is the readers' sensitivity, on the job's contexts.
     - Compare it with the same pole on activation-matched non-job contexts, and with activation-matched other
       poles.
     - Before using any first-order score, validate it against real ablation on the 40 gate poles.
2. **Refuting observation. Severity: changes a number.** Freezing the share on control poles is good, but the
   band must be activation-matched, as above. Add a per-pole positive control: the pole's use must be
   detectable at T_15, or the pole is cancelled. Single-pole ablation failed its positive control on 0.417 of
   Q4B poles and 0.371 of G4 poles (`studies/neuronoff/results/verdict_{Q4B,G4}_v2.json`).
3. **The number. Severity: changes a number.** Raw activation times gradient changes by orders of magnitude
   across points (0.5). Compare within each point.
4. **Already done. Severity: cosmetic.** Neither result decides the neuron-level question. Both outcomes
   remain open.
   - Probe directions are readable before they are causal (https://arxiv.org/abs/2609.01048).
   - In wdd, recruitment is not a performance event, and the class direction already carries load before the
     row arrives (https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/41_six_directions.md).
5. **The demo. Severity: blocks the plan.** With activation times gradient, the use curve follows the fire
   curve by construction.
6. **Cost. Severity: changes a number.**
   - **Full design.** With full 2,048-token windows, 100 births × 50 contexts × 16 points is 80,000 windows. A
     forward plus backward pass is about 9.0e13 FLOP each, 7.2e18 FLOP in all: about 17 GPU-h at an assumed
     1.2e14 FLOP/s, plus the ablation forwards.
   - **Reduced design.** Three points per birth (before, at and after) is about 3 to 4 GPU-h.
   - **Truncation needs a check.** The banked contexts hold 64 tokens before the position and 16 after.
     Activations on an 80-token window differ from those on the harvest's 2,048-token window. Check them
     against the banked values before use.

### N3 — the route at birth

1. **Identifiability. Severity: changes a number.**
   - **Two causes need no route-building.** "Contributors are older than the job" follows from both:
     - Contributors sit in earlier layers, and earlier layers settle first (N20, item 1).
     - The job is a late-born secondary job, while each contributor would be dated by its main job, which the
       rule dates early (0.2).
   - **Matched controls.** Controls must match on layer and job type: contributors to late-born jobs of
     control poles in the same layer, dated by the same rule.
   - **Relatedness needs a measure.** "Related in content" needs a reader, and the only banked O7 agreement is
     κ 0.070 on pole classes. Use a mechanical measure instead: shared top-50 positions or tokens.
2. **Refuting observation. Severity: changes a number.**
   - **Churn alone fires the clause.** "The top contributors change wholesale between the birth and the next
     point" fires on churn alone. The pole's own top-50 turns over by about 80% between neighbouring points
     (0.1). The chance level is how much the same pole's contributors churn across non-birth intervals.
   - **The top 10 explain little.** The 16 largest upstream contributors carry 0.02 to 0.05 of a mid-network
     neuron's input energy (wdd e685,
     https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/62_already_clean_readers.md). In
     Pythia-410m, heads carry 0.77 of a facet's direct rise (Eliwat, https://github.com/Meliwat/recgen-checkpoints,
     per `04_LITERATURE.md` N3). Report the share of pre-activation that the top 10 explain, frozen on
     controls.
3. **The number. Severity: changes a number.**
   - Heads have no birth-dating instrument: there are no head banks (`00_ASSETS.md` §6, gap 2).
   - Neuron ages taken from the census inherit N20's threshold.
   - The exact input split must first reproduce the neuron in fp32. At Qwen2.5-7B it does in fp32 (error
     5e-6), and in bf16 it fails at 0.86 to 0.95 of the damage (wdd e700,
     https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/65_brake_bypass.md, per `04_WDD.md`
     §3 item 13).
4. **Already done. Severity: changes a number.**
   - This repo's unite Arm 1 measured upstream input overlap at birth for 841 OLMo-3 7B jobs: median T 0.356
     [0.307, 0.384]. Its status is "Not closed: awaiting review by a different model"
     (`studies/unite/results/RESULTS.md`).
   - Neuron routes over training with RelP exist for Pythia-1b (https://arxiv.org/abs/2610.33299, full, per
     `04_RECENT.md`).
5. **The demo. Severity: changes a number.** A ten-node graph labelled with job names, carrying a few percent
   of the pole's input, would read as "the circuit".
6. **Cost. Severity: cosmetic.**
   - 3 GPU-h is plausible if the work builds on the unite J4 banks (`s3://pg-lineage-769063704696/results/unite/j4{a,b,c}/1/out/`)
     and `infra/pg/unite_route.py`.
   - Head ages would need a new instrument.

### N12 — gradient starvation

1. **Identifiability. Severity: blocks the plan.**
   - **The winner is chosen after the fact.** Pole A is the pole that holds the job at the end. Its up-row
     gradient on the job's contexts is proportional to its own gate output there, which rises at birth by
     definition. So A's share rises.
   - **The error falls for everyone.** The other poles' gradients scale with the error at those positions.
     When the loss there falls for any reason (heads, other layers), every competitor's gradient falls.
   - **Together.** These two facts give "one rises, the rest fall" with no competition at all.
   - **A gradient is not an update under Adam.** In Pythia-410M, a 2.25- to 2.73-fold gradient advantage did
     not carry through to the updates or the weights (https://arxiv.org/abs/2609.01170, TEXT, per
     `04_LITERATURE.md`).
   - **Fix.**
     - Freeze the competitor set before birth: the 20 poles of the layer with the largest gradient per unit
       activation on the job's contexts at the point before birth.
     - Score A's rank among them at that point. This is a prediction; see M2.
     - Compare the competitors' share on the job's contexts with their share on loss-matched non-job contexts
       at the same point.
     - Report AdamW updates beside the gradients.
2. **Refuting observation. Severity: changes a number.** "Competitors unchanged across the birth" can fire,
   but nothing sets a floor for the effect of the falling error.
3. **The number. Severity: cosmetic.** The card does not say which parameters it means: gate row, up row or
   down column. Units also drift across points (0.5).
4. **Already done. Severity: cosmetic.** In wdd e589, the class positions supply 0.215 of the recruit's
   gradient, against 0.086 for matched non-recruits. The direction-cosine difference from matched controls is
   about 1.3 standard errors
   (https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/40_turn_and_paper.md, per `04_WDD.md`
   §4.7 item 2).
5. **The demo. Severity: blocks the plan.** Many curves with one rising is the picture that selection on the
   winner produces.
6. **Cost. Severity: changes a number.**
   - The gradients come free with N2's backward passes if all poles are recorded.
   - The updates need optimizer state at both ends of each interval: fp32 master weights and two moments for
     7.3e9 parameters, about 88 GB per checkpoint. That figure is inferred from the parameter count, not read
     from the bucket.

---

## Part 2. Program B

### N13 — what RL wrote

1. **Identifiability. Severity: changes a number.**
   1. **Energy alone can explain the result.** If the change is concentrated, the top k neurons by change norm
      hold more change energy than any random set of k. So "the top k recover the gain and a random k do not"
      can be energy alone. The energy-matched random set must come from outside the top k, with equal total
      change energy, which means more neurons.
   2. **DARE predicts the opposite of the card.** DARE drops 90 to 99% of SFT delta parameters at random,
      rescales the rest by 1/(1−p), and keeps the abilities (https://arxiv.org/abs/2311.03099, abstract). It
      predicts that a rescaled random subset can recover the gain. The card's "random recovers nothing" is
      expected only without rescaling. Register both.
   3. **bf16 rounding.** Sparsity counts on bf16 checkpoints are inflated by rounding
      (https://arxiv.org/abs/2511.08567). Rank neurons on the float difference, with the bf16 step as the
      floor.
   4. **Not only MLP neurons.** RL changes sit partly in attention. Swapping attention patterns between the RL
      and DPO checkpoints of OLMo-2-7B moves a cue effect "most of the way" (https://arxiv.org/abs/2610.22937,
      full, per `04_RECENT.md`). Add an arm for attention and norm weights, and an arm for everything except
      the top k.
2. **Refuting observation. Severity: changes a number.**
   - The clause "more than 5% of neurons needed" has room to fire.
   - **Missing gate.** The stage's own gain must be several times the evaluation noise. AIME's 30 problems
     cannot resolve a transplant; MATH-500 can.
   - **Ceiling:** the full delta.
   - **Floors:** a DARE-rescaled random set, an unrescaled random set and the zero delta.
3. **The number. Severity: changes a number.** "Gain recovered against k" is a fraction of a margin, and on a
   small stage gain it becomes unstable. Report absolute accuracy with intervals (repo rules, "Instruments and
   controls", item 9).
4. **Already done. Severity: changes a number.**
   - Weight Patching recovers instruction following with the top 5% of heads and neurons
     (https://arxiv.org/abs/2604.13694).
   - Steering the changed SAE features recovers about 80% of an RL gain at 1.5B
     (https://arxiv.org/abs/2609.15064).
   - RL updates on Olmo-3 have 388 to 1,508 times the Fisher leverage of random directions
     (https://arxiv.org/abs/2610.37719).

   These results predict localization at some k. What is open is the neuron basis on one model's chain, with
   the DARE control (`04_RECENT.md` §6).
5. **The demo. Severity: changes a number.** "The neurons RL rewrote, with their jobs before and after" needs
   jobs at the post-training base.
   - The chain starts from the released base `main`, after stages 2 and 3, not from `stage1-step1413814`
     (`01_JLENS.md` §7).
   - The stage-1 job labels do not transfer without harvests at `main` and at each stage, 1.2 to 1.8 GPU-h
     each.
6. **Cost. Severity: changes a number.** 3 to 6 GPU-h is low. `04_PRACTICAL.md` §5 estimates 20 to 25, and
   that fits: Think models write long chains of thought, and every arm needs a full MATH-500 generation.

### N28 — forgetting predicted from weights

1. **Identifiability. Severity: blocks the plan.**
   - **The stages are nested.** The 8 to 11 released stages are links of a few chains: Think (SFT, DPO, RL),
     Instruct and RL-Zero. Forgetting accumulates along a chain, and so does every distance from the base (L2
     change, step count). Any predictor that grows along the chain fits it.
   - **Too few independent branches.** With about 3 independent branches, leave-one-out R² cannot separate the
     age-weighted score from plain L2.
   - **Within-stage checkpoints do not help.** Think SFT has 43, Think RL 55 and RL-Zero 19. They add points
     along one trajectory each, not independent runs.
2. **Refuting observation. Severity: blocks the plan.**
   - **The floor will likely pass anyway.** L2 itself will probably meet the 0.5 floor on nested stages. So
     the clause "not above L2" decides, and at n ≈ 10 it has no power.
   - **Missing reference.** Shuffle the ages across poles. If the shuffled-age score fits as well, age adds
     nothing.
3. **The number. Severity: changes a number.**
   - **The literature R² is a different object.** The figures come from RL's Razor
     (https://arxiv.org/abs/2509.04259): 0.34 to 0.58 for weight-change metrics, and for forward KL 0.71 at 3B
     and 0.96 on MNIST. They are R² across many independent fine-tuning runs. Leave-one-out R² across about ten
     nested stages of one model measures something else.
   - **Ages cover almost no poles.** Ages exist for 640 of 704,512 poles (0.091%), dated at stage-1 points.
     Reader counts are not banked anywhere.
4. **Already done. Severity: cosmetic.**
   - RL's Razor uses forward KL.
   - Fisher leverage per stage has been measured on Olmo-3 (2610.37719).
5. **The demo. Severity: blocks the plan.** A scatter of about ten points along a chain looks like a line for
   any cumulative predictor.
6. **Cost. Severity: cosmetic.** 15 GPU-h is plausible for about ten stages and three evaluations.

### N6 — edit the young

1. **Identifiability. Severity: blocks the plan as designed; fixable.**
   - **Age is collinear with activation statistics.** On the 600 dated poles, the within-layer Spearman
     correlations between main-job birth index and T_15 statistics are (`age_energy_out.txt`):

     | T_15 statistic | Spearman with birth index |
     |---|---|
     | log max\|a\| | −0.592 |
     | log 99.9th percentile | −0.437 |
     | log density | +0.502 |

   - **The rule makes this so.** The birth rule is itself a threshold on exemplar activation relative to the
     99.9th percentile (Part 0). So a heavy-tailed pole is born earlier by construction.
   - **Energy dominates removal cost.** In wdd e733, price against activation energy is 0.94 to 0.97
     (https://github.com/sunmoonron/wdd-descendants, per `04_LITERATURE.md` N6).
   - **Regression cannot separate age.** A regression of collateral on age, energy, norm and layer over 300
     poles, at |ρ| ≈ 0.6, cannot assign the effect to age.
   - **Fix.**
     - Use pairs in the same layer, matched on max|a|, density and column norm, that differ in birth by at
       least three points.
     - Hold out pairs.
     - Adjust ages by the random-group reference.
2. **Refuting observation. Severity: changes a number.** "Flat after matching" can fire, but single-pole
   collateral sits near noise (neuronoff). Add two controls:
   - a positive control, an edit of known large effect such as each layer's highest-energy pole;
   - a random-direction edit of the same Frobenius norm.
3. **The number. Severity: changes a number.**
   - Zeroing columns of different norms is not an equal-size edit. Fix ‖ΔW‖_F per edit.
   - Report collateral in nats per token.
   - For a rule editors can use, add an efficacy readout. An edit on a young pole may cost less because it
     does less.
4. **Already done. Severity: cosmetic.** The card is OPEN (`04_LITERATURE.md`). Energy is the known dominant
   predictor.
5. **The demo. Severity: changes a number.** Unless poles are matched, the scatter of collateral against age
   will show the energy gradient.
6. **Cost. Severity: changes a number.**
   - 900 edits (300 poles × 3 types), each with a forward pass over a wikitext test set of about a
     quarter-million tokens, is about 3e18 FLOP. That is about 7 GPU-h at 1.2e14 FLOP/s. Caching the residual
     up to the edited layer halves it.
   - Getting under 2 GPU-h requires an eval set of about 50,000 tokens, which may not resolve single-pole
     effects.

### N19 — what RL kills

1. **Identifiability. Severity: blocks the plan.**
   1. **The premise is unmeasured here.**
      - RLVR models beat their base models at small k and lose to them at large k
        (https://arxiv.org/abs/2504.13837, abstract).
      - But across 13 RLVR checkpoints of OLMo-2-1B, coverage at 256 samples rose from 94.5% to 96.9%
        (https://arxiv.org/abs/2611.02574, TEXT, per `04_LITERATURE.md`).
      - Nobody knows whether Think RL lowers large-k pass rates relative to Think-DPO on OLMo-3 7B.
   2. **A partial revert always trades.** Any partial revert toward the pre-RL weights trades pass@1 for
      pass@k. The baselines are whole-model weight interpolation (WiSE-FT, https://arxiv.org/abs/2109.01903,
      abstract) and a sweep of sampling temperature. A restore must beat both at matched pass@1.
   3. **The readers are global.** For final-layer entropy poles, the readers are the final norm and the
      unembedding. Entropy neurons act through the final norm's scale, by writing into directions that the
      unembedding maps near zero (https://arxiv.org/abs/2406.16254, abstract). Restoring their readers is a
      global change.
2. **Refuting observation. Severity: blocks the plan.** Two things are missing:
   - a gate showing that the collapse exists on this chain;
   - the interpolation-and-temperature frontier as the baseline.
3. **The number. Severity: changes a number.**
   - Name k, the estimator (unbiased pass@k from n samples) and the problem set.
   - State "a small cost to pass@1" as a number.
4. **Already done. Severity: cosmetic.** Entropy collapse in RLVR is a named, studied problem
   (`04_PRACTICAL.md` §1). It has not been studied at the neuron level.
5. **The demo. Severity: blocks the plan.** A restored pass@k curve drawn above the RL curve misleads unless
   the interpolation and temperature curves are in the same chart.
6. **Cost. Severity: changes a number.** 2 to 4 GPU-h is low.
   - **Per arm.** 100 problems × 64 samples × about 4,000 tokens is 25.6M generated tokens. At an assumed
     2,000 to 3,000 tokens per second for a 7B model in a batched server on one L40S (not measured here), that
     is 2.4 to 3.6 GPU-h.
   - **In all.** Six arms (pre, post, restore, random restore, interpolation and temperature) come to 15 to
     25 GPU-h.

---

## Part 3. N22 — the model gets a variable

1. **Identifiability. Severity: changes a number.**
   1. **Slot is confounded with depth.**
      - The first intermediate becomes readable at earlier layers than the second (`01_JLENS.md` §5.1).
      - If each slot is swapped in its own layer band, a value injected earlier passes through more of the
        remaining operations.
      - So "f₁(v) from slot 1, f₂(v) from slot 2" also follows from "the model applies whatever operations
        remain after the injection depth".
      - Fix: compare the slots at a matched layer where both are readable, or state the claim as depending on
        depth.
   2. **Near values leak.** Number tokens share magnitude and digit features, and v + d is close to v. Score
      the exact target token against v ± d and against the other slot's target. That is the 2 × 2
      interaction.
   3. **A lens-free positive control is missing.**
      - Use an interchange intervention at the operand positions, taken from a prompt whose intermediate
        equals v. Causal mediation for arithmetic is the method (https://arxiv.org/abs/2305.15054, abstract).
      - If the interchange shows slot dependence and the lens swap does not, the variable exists outside the
        lens coordinates.
      - If neither does, the lens test never had a chance.
   4. **An alternative predicts no onset.** Arithmetic may run on a set of heuristic neurons, which are the
      main source of arithmetic accuracy early in training (https://arxiv.org/abs/2410.21272, abstract). That
      account predicts no separate "variable" onset.
2. **Refuting observation. Severity: changes a number.**
   - The two floors are good.
   - **Missing gate.** Accuracy without chain of thought must be gated at the final point and at each earlier
     point. There are about 1,000 single-token answers, so chance is about 0.1%.
   - **The order clause is close to impossible.** "Slot dependence appears before the model answers correctly"
     cannot hold as stated. Raising f₁(v) exactly needs the downstream multiply and add to be right on v, and
     that is the ability that gives correct answers. Keep "readout before slot dependence", and score each
     point only on the items it answers.
3. **The number. Severity: changes a number.** "Readout appears" (rank 1 at some layer) and "slot dependence
   appears" (an interaction above both floors) are different quantities. At each point they are measured on
   different item sets. Freeze both thresholds on the split-half lens.
4. **Already done. Severity: cosmetic.**
   - No across-training slot test was found (`04_RECENT.md`).
   - Open-model swaps failed or were weak at 27B (`01_JLENS.md` §5.2).
   - Third-party lenses for OLMo-3-7B exist for the post-training variants, not for stage-1 points
     (https://huggingface.co/mhough/olmo3-jacobian-lenses).
5. **The demo. Severity: cosmetic.** "At step X" is an interval of the 16-point grid, for example 100,000 to
   180,000. Print the interval.
6. **Cost. Severity: cosmetic.** 3 to 6 GPU-h becomes about 5 to 8 GPU-h: five lens fits at 0.5 to 1 GPU-h
   each (an estimate, `01_JLENS.md` §7), the split-half refit and five downloads. The swaps are cheap.

## The Program C gate (not asked; found while checking costs). Severity: changes a number.

The two sources disagree:
- `02_FRAMES.md` §2 says the LIBERO harness scores 200 episodes in about seven minutes of GPU time.
- `00_ASSETS.md` §4 gives a median of 27.62 s per episode, and about 7 GPU-h for 810 rollouts.

By the measured rate, 200 episodes take about 1.5 GPU-h. The gate is 600 episodes: blank and swapped
instructions on four suites at 50 episodes each, plus a 200-episode unmodified baseline. That is about 4.6
GPU-h, not about 1.

---

## Part 4. The scoring rubric

**R1. Edge and Novelty count one fact twice. Severity: changes a number.** For Program A cards, "nobody else
has 16 banked points at 7B" sets Edge to 5. It is also why the audit found no prior work, which gives Novelty
4. The product counts that one fact twice.

**R2. Decisiveness was scored without an identifiability check. Severity: changes a number.** Four cards
scored decisive 4 have a design problem that decides the reading:
- N2 uses a measure that forces its reading.
- N20 has a threshold that sets its answer.
- N12 selects on the winner.
- N28 has no power.

Decisiveness 5 asks that "the refuting observation has room to fire". Room is not enough when a confound fires
it either way.

**R3. The Wow anchors were not applied. Severity: changes a number.** N20 is a heatmap. Anchor 3 is "a
picture an ML researcher understands", and `02_FRAMES.md` §5 says a gallery, slider or count "does not pass on
its own". At Wow 3, N20 falls from 1,600 to 1,200, and N3 (a route picture, Wow 4) falls from 960 to 720.

**R4. The cost axis is small in money but large in score. Severity: cosmetic.**
- Cost 5 against cost 3 multiplies leverage by 1.67, the same factor as Wow 5 against Wow 3.
- At the quoted $2.24 to $3.00 per GPU-hour, the money between them is about $10 to $90.
- The estimates are also low for N2, N4, N6, N13 and N19, and high for N27 (Parts 1 and 2).

**The one change.** A card can be a pick only if one of its own outcomes has one of the three shapes in
`02_FRAMES.md` §5:
- a before/after with a matched control;
- a prediction made before the measurement;
- a number an engineer already tracks.

A card whose outcome is a picture or a description is scored as an instrument of the pick it serves. That
covers N20, N1, N3, N12, and N2 as designed. Its cost is added to that pick, and it has no leverage of its
own.

Under this change, Program A has one card that can be a pick, N27, and Part 1 predicts it fails its own
mismatched-births control. N13, N22 and N6 remain eligible.

---

## Part 5. What is missing (three ideas)

### M1. Stage 2 is a 7B data experiment that AI2 already ran.

**The asset.** Hugging Face lists three stage-2 runs from the same stage-1 checkpoint, at about 2T tokens.
Each runs 47,684 steps, on mixes named `round5`, `gen-mc-only` and `math-code-reasoning-web` (branches
`stage2-step47684-mix-*-from-2T-ckpt`). They sit beside the main stage-2 run, which has checkpoints every
1,000 steps (`hf_refs.json`, 52 stage-2 branches).

**The measurement.**
- Harvest all neurons at the shared start and at the three ends, on the fixed 20M corpus.
- Add two adjacent checkpoints of the main stage-2 run as the identity pair.
- That is five harvests, about 8 GPU-h.
- The quantity needs no judge. For each pole, take the change in the share of its top-50 positions that lie
  in code and math documents of the fixed corpus. The sources are in the corpus manifest.

**The picture.** A before/after with its control in the same chart: the same neurons, from the same weights,
under two mixes.

**Refuted if** either:
- the between-mix difference in these shifts is inside the identity pair's spread; or
- the math-code mix does not move more poles toward code and math documents than the general mix does.

**What else it fixes.** It removes the bracket problem of N1 and N4, because births inside stage 2 can be
dated to 1,000 steps on the main run. A third-party lens file already dates 34 of 64 layer-20 readout
directions at `stage2-step47684` (https://huggingface.co/jeeva2812/olmo3-jlens-checkpoints, per
`04_RECENT.md` §8).

**Not checked.**
- Whether the 2T start is a released stage-1 branch. 2T tokens is about step 476,837, which is not a multiple
  of 1,000.
- The composition of the mixes.

**Nearest prior work.** Mid-training edits on OLMo-3-7B, read at behavior level
(https://arxiv.org/abs/2610.06851).

### M2. Name the neuron that will take a job, before it does.

**The measurement.**
- Take the 148 jobs that pass the random-group test (0.3).
- At the point before each birth, run one forward pass on the job's exemplar windows.
- Rank all 11,008 poles of the job's layer by a score frozen in advance: the pole's within-point percentile on
  those positions, and that percentile times its gradient per unit activation.
- Define the carriers at the birth point as the poles above their own 99.9th percentile on those positions.
  This needs a second forward pass.
- The score is the share of jobs whose carrier ranks in the top 10. Chance is 10/11,008, about 0.0009.
  Compare against layer-matched, activation-matched random poles.

**The baseline to beat.** The gradient term must beat the pole's own current percentile. wdd's best forecast
lost to the row's own current usage, 0.964 against 0.927
(https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/55_forecast_vs_features_planted_row.md).

**Refuted if** precision at 10 is at the level of random poles, or the gradient term adds nothing over the
current-percentile baseline.

**Why it qualifies.** It is shape 2 of `02_FRAMES.md` §5, a prediction made before the measurement. The
headline would read: "at step 100,000 we named the neuron that carried this job at step 180,000".

**Prior work.**
- SAE carriers were predicted from onset firing in 4 of 5 cases in Pythia (https://arxiv.org/abs/2605.18789,
  abstract, per `04_RECENT.md`).
- wdd ranked the eventual row first in 0.52 of cases two checkpoints ahead, in Pythia-410m
  (https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/39_migration.md).

**Cost.** About 30 checkpoint loads and forward passes on about 1,500 windows: 5 to 8 GPU-h, or about 12 with
gradients.

### M3. Skipped updates, explained by their batch.

**The asset.** The run zeroed the update on outlier steps by rule (SkipStepAdamW, sigma factor 6, window
128, `00b_DATA_ORDER.md` risk 4; which statistic it thresholds was not checked). W&B logs `optim/step skipped` and `train/CE loss` for every
step. The batch order is computable, and pilot 0's offline gate passed (`JOURNAL.md`, 01:16).

**The measurement.** For each skipped step:
- score its batch on the nearest earlier checkpoint, at most 999 steps before, with a forward pass only;
- score 60 random batches from the same interval.

**The engineer's number.** The share of skipped updates whose batch is a loss outlier on that checkpoint.
These are the skips a data filter could have prevented. Name the documents and MLP channels that carry each
outlier.

**Refuted if**, for most skipped steps, the skipped batch's loss lies inside the 5–95% band of the random
batches. The skips would then be caused by the model's state, not by the data.

**Cost.**
- A forward pass over 4.19M tokens is about 6.1e16 FLOP, about 0.14 GPU-h at 1.2e14 FLOP/s.
- 30 skipped batches and 60 control batches come to about 13 GPU-h, plus 14.6 GB per checkpoint.

**Not audited.** I did not audit the literature on loss spikes for this idea.

---

## Part 6. Verdicts

| card | verdict | changes, or the reason |
|---|---|---|
| N20 | RUN WITH CHANGES | Run as an instrument, not a headline. Title the map by its rule and print k. Sweep the threshold. Harvest the identity pair (about 3.5 GPU-h). Add the random-group reference. Normalize per ln(step). The D1 rule on all poles costs about 22 GPU-h. |
| N27 | DO NOT RUN as designed | The banked data predict its third refutation, and the job names have no agreement number. If wanted, run a 0 GPU-h version on the W&B evals, with random-group births registered as the comparison, on the 148 births only. |
| N4 | DO NOT RUN at 7B | The interval's documents are exchangeable, the cosine is forced and the lithium job is undated. Ask the data question with M1 instead. |
| N1 | RUN WITH CHANGES | First locate onsets inside their brackets. Measure job-direction projections against random directions, Adam-preconditioned, at matched learning rate. Plant a positive control. Use the 148 births. Budget the transfer for strided columns (Part 1). |
| N2 | RUN WITH CHANGES | Use gradient per unit activation with activation-matched controls. Validate first-order scores against ablation on the gate poles. Require a positive control per pole at T_15. Normalize within each point. Use full windows, or a truncation checked against the banked values. |
| N3 | RUN WITH CHANGES | Close the unite Arm 1 review first and build on its banks. Match controls on layer and job type. Measure relatedness mechanically. Freeze the share explained on controls. Run the fp32 split check. Claim no head ages until an instrument exists. |
| N12 | RUN WITH CHANGES | Run only as the prediction in M2: freeze the competitor set before birth, compare against loss-matched contexts and report Adam updates beside the gradients. |
| N13 | RUN WITH CHANGES | Add DARE-rescaled and tail-energy-matched random sets, a bf16 floor and the attention-and-norm and complement arms. Gate the gain against MATH-500 noise. Report absolute accuracies. Take jobs from harvests at `main` and each stage. Budget 20 to 25 GPU-h. |
| N28 | DO NOT RUN | About ten nested stages give no power. It compares against an R² of a different object. Ages cover 0.091% of poles. |
| N6 | RUN WITH CHANGES | Use pairs matched on energy, density and norm. Use edits of equal Frobenius norm. Add a positive control, a random-direction edit and an efficacy readout. Budget 4 to 8 GPU-h. |
| N19 | DO NOT RUN until gated | First measure whether Think RL lowers large-k pass rates against Think-DPO. Then run only with interpolation and temperature baselines at matched pass@1. Budget 15 to 25 GPU-h. |
| N22 | RUN WITH CHANGES | Gate accuracy at each point. Add the lens-free interchange positive control. Compare slots at matched depth. Score the exact-target interaction. Drop "slot dependence before correct answering". Budget 5 to 8 GPU-h. |

Program A as a bundle should not be the first program. Its first step is the instrument check from Part 0:
- the random-group reference, at 0 GPU-h;
- the identity-pair harvest, about 3.5 GPU-h.

If fewer than about 100 births survive the check, Program A has no unit to study.

**Run first: N13.** It is the only pick whose readout is a benchmark before/after with matched random
controls in the same chart. Its identification does not rest on the birth dates, which Part 0 shows are
mostly set by cluster size and rank. The literature disagrees on its answer: sparse RL updates against a bf16
artifact, and DARE predicting that rescaled random subsets also work. So both outcomes are informative and
the refutation has room to fire. Start with the single RL stage that has the largest gain clearing the
MATH-500 noise gate: RL-Zero-Math from the base, or Think-DPO to Think RL. That costs about 20 GPU-h, and the
inputs only need downloading.

---

## Part 7. Limits of this review

- **The random-group test is partial.** It covers the 600 S poles and uses only the percentile reading. The
  primary count also requires agreement with the raw reading and a match in a second clustering (B3, B4), and
  I did not apply those to the random groups. My recount of late-born clusters before those filters is
  1,535; the banked primary set is 1,238.
- **It cannot rule out pole-wide events.** Random regroupings keep anything that moves a whole pole at once.
  The test shows that births are mostly a property of the pole and of the group's ranks. It does not show that
  no event happens.
- **Some costs are estimates.** Several are FLOP estimates at an assumed 1.2e14 FLOP/s on one L40S, about a
  third of its dense bf16 peak. Two rates are measured: the harvest, about 4.8e13 FLOP/s forward with hooks
  (from `00_ASSETS.md` §1.1), and the LIBERO episode rate.
- **Some citations are abstract-level.** The prior work I added was read as abstracts through the arXiv API on
  2026-11-05:
  - SVCCA;
  - Du et al.;
  - DataDecide;
  - AtP*;
  - rotational equilibrium;
  - DARE;
  - WiSE-FT;
  - Yue et al.;
  - Stolfo et al. 2023 and 2024;
  - Nikankin et al.

  Every other citation carries the read level of the file I cite it from.
- **The API reads are not in the repo.** The W&B and Hugging Face facts come from API reads saved in the
  scratchpad.
- **Unread.** I did not read `pilot0/` or the cards outside the brief.
