# Visuals plan for the walkthrough, 4 Nov 2026

Each picture follows the rule of "One neuron across training". Viewer text says neuron, group of contexts, training step and judge, never pole, p_u, H_SP, gate or unblinding. "Group" names the clustering model's groups, because line 720 and E_skeptic §3.5 show they are not always jobs. Line numbers are in `index.html` here.

## (a) Concepts the prose carries

1. Token, neuron, activation, positive and negative side: 308–317.
2. Top context and the corpus (50M tokens for explanations, 20.56M for the training points): 318–321, 500, 509, 678.
3. The essay's claim (packing against recursive generalization, 90 / 7 / 2–4), the two questions and the decision: 300–305, 327–331.
4. The explanation test (explainer, blind writer, positives, near misses, distant texts, blind maximum, score, strong, pass): 336–345, 352.
5. The line for "strong", set from texts scored at a random neuron: 347–350.
6. The class is computed, not claimed: 352–353, 427.
7. One pass and one fail: 357–382, Fig 4 384–411.
8. The explainer model changes the pass rate: 414–495, Figs 1–2.
9. Checkpoints and the 16 training points: 504–509, Fig 3 511–572.
10. Groups and the birth rule: 578–582.
11. The parent test, decoy, control, unrelated share and rule: 583–593, 699.
12. Predictions and outcomes: 596–610, 701.
13. The got and lithium neurons: 288, 293, 617–665, 710.
14. The mixed outcome, the time trend and its cautions: 289, 667–668, 700, 707–708.
15. What the judge's control can and cannot catch: 670–671.
16. The audit on three models against the essay, and the explainer dependence: 673, 702.
17. When the strongest groups first appear: 709.
18. Is the strongest group the oldest: 700.
19. "Cluster" replaces "job": 720.
20. Wiring (overlap score, switch-off score, fake two-neuron neurons, split halves): 290, 713–719, Fig 5.
21. Always-on neurons: 291, 746–748, 752–753, 765–767, 803–806, 815–816.
22. Shared inputs, the same-weights objection and the third group: 754, 769–773, Fig 6.
23. Birth is a line crossing, and activations shrink: 755, 758–763, 923.
24. The value written into switched-off neurons: 812–816, Fig 7.
25. Groups against inputs against the marked token: 292, 866–871, Fig 8.
26. Heads, downstream, the forecast, the input ranking and the side studies: 721, 906–918.
27. Where it stands, and costs: 574, 920–926.

Not pictured: 3, 8, 12, 24, 26, 27. They lack a per-neuron artifact or lie outside this study.

## (b) Pictures

P0, "One neuron across training" (676–681), is built.

### P1. What a top context is
- **Sentence:** A neuron's top contexts are the 50 tokens, out of 20,560,068 read, where it fired hardest.
- **Object:** one OLMo neuron at the end of training (default L03N7155+, "got").
- **Count:** 50 ticks on a strip of 20,560,068 positions, beside the neuron's 99.9th-percentile line.
- **Axis:** position in the corpus, in millions of tokens. **Control:** a rank slider from 1 to 50 that shows that context. **Random:** the slider's random-neuron button.
- **Data:** local, `interp_research_banks/neurons/w3/grid/1/out/t15_step1413814/samples/contexts.jsonl` (16,502,054 B) and the `metrics.json` beside it.
- **Claim:** the got neuron's strongest token reads 0.663, and its 99.9th percentile is 0.050 (that file).
- **Avoid:** taking the top 50 as all the neuron does (E_skeptic §3.4, "The parent side"). Name the corpus, because the explainer read a separate 50.5M-token run (A_training_dynamics Part 3).
- **Replaces:** 318–321, 509. **Size:** about 0.3 MB. **Compute:** none. Assert every tok < 20,560,068.

### P2. The 16 training points
- **Sentence:** The study saw 16 saved points of OLMo-3's training, packed into its start.
- **Object:** the 16 checkpoints. **Count:** dots; 7 of the 16 lie in the first 1%.
- **Axis:** training tokens, from 0 to 5.93T. **Control:** a switch between a linear and a log axis. **Random:** none applies.
- **Data:** local, the steps in `neuron_across_training/data.js`; SPEC_D1 §4.3.
- **Claim:** T0–T6 lie below 1%, and no point lies between 39.6% and 70.7% (SPEC_D1 §4.3).
- **Avoid:** dating a birth more finely than its interval (E_skeptic §3.2). Say that the slider spaces its points evenly.
- **Replaces:** 506, Fig 3. **Size:** under 2 KB. **Compute:** none.

### P3. One explanation, sixteen test texts
- **Sentence:** An explanation passes only if texts written from it alone fire the neuron and its near misses do not.
- **Object:** one OLMo neuron's explanation, 5 positives, 5 near misses, 5 distant texts and blind maximum.
- **Count:** filled squares per row (strong texts), against the limits of at least 3, at most 2 and at most 1.
- **Axis:** score (peak activation over the neuron's strongest corpus activation). **Control:** drag the line for "strong" (it starts at 0.3153). **Random:** score the same texts at a random neuron (the stored shuffled scores).
- **Data:** S3 only, `w5/verdict/samples/q1_O7_poles.jsonl` (24,420,010 B).
- **Claim:** on OLMo, 1,794 of 2,995 positives were strong; 23 of 8,985 texts at a random neuron (local `w5/verdict/metrics.json`, q1.O7.strong_rates).
- **Avoid:** a failure judges the explanation, not the neuron (line 382). Near-silent neurons can pass (E_skeptic §5 item 1).
- **Replaces:** 338–345, 352, 360–382, Fig 4. **Size:** about 1.8 MB.
- **Compute:** pass or fail from the stored scores. Assert that the stored class comes back at 0.3153 and class_theta_plus at 0.4153, for all 600.

### P4. How many neurons have one job
- **Sentence:** Of 600 neurons per model, this is how many have an explanation that passed the test.
- **Object:** 600 neurons each of Qwen3-4B base, Gemma-3 4B and OLMo-3 7B.
- **Count:** dots coloured one job, two related, unrelated, cannot tell; a line at 540 marks the essay's 90%.
- **Axis:** the model columns. **Control:** the line for "strong": three stops locally (0.2153, 0.3153, 0.4153), continuous with the q1 files. **Random:** classes recomputed from scores at a random neuron.
- **Data:** local, `w5/verdict/metrics.json` (136,032 B; q1.<model>.fractions, band_theta_±0.1). For the continuous line and the random button, S3 `q1_{Q4B,G4,O7}_poles.jsonl` (23.5, 23.7 and 24.4 MB).
- **Claim:** one job for 358, 338 and 311 of 600; 383, 373 and 370 at 0.2153; 313, 294 and 235 at 0.4153 (RESULTS.md Q1).
- **Avoid:** "cannot tell" is not "packed" (line 673). The shares move with the explainer (OLMo: Opus 54.7%, Sonnet 44.8%), and the explainer claimed 521 where the test confirmed 311 (E_skeptic §4, D9).
- **Replaces:** 304, 328, 673, 702. **Size:** 1 KB, or 350 KB continuous.
- **Compute:** port assign_class, explanation_passes and job_supported from `infra/pg/neurons_verdict.py`. Assert that the stops equal metrics.json and that the shuffled strong counts are 48, 26 and 23.

### P5. One neuron's 50 contexts, sorted into groups
- **Sentence:** A language model sorted each neuron's 50 strongest contexts into groups; some neurons got one group, some fifteen.
- **Object:** one OLMo neuron at the end of training.
- **Count:** one column of squares per group, each square labelled with its marked token. The viewer counts groups and the distinct tokens in each.
- **Axis:** none (columns by size). **Control:** a switch between the first and second sorting (the model asked twice). **Random:** a random neuron of the 600.
- **Data:** S3 only, `w4/o7_s_v2/explain/clusters.jsonl` (944,095 B; both samples; the verdict's input, sha256 afa666e2…). The text comes from the local T15 contexts.jsonl.
- **Claim:** 2,454 groups over 600 neurons; the two sortings agree within one for 67.8% (metrics.json q2.B4_m_within_1); lithium gets 15, then 3 (B_clutter Story 1).
- **Avoid:** a group is not a job (line 720: 109 of 129 not about one thing). The fair contrast with the explainer is 532 against 169–232, not 26 (E_skeptic §3.5).
- **Replaces:** 581, 720, the counting in 871. **Size:** about 0.5 MB for 65 neurons, or 3.6 MB for all 600.
- **Compute:** join ranks to contexts. Assert the sha256 and that each q2.gallery exemplar lies among its group's ranks. Drop the 11 echo groups.

### P6. When each group first appears
- **Sentence:** A group is born at the first point where more than half of its contexts reach the neuron's top 0.1%, and still do at the next point.
- **Object:** one OLMo neuron's groups (default "got").
- **Count:** one row of squares per group, filled for contexts at or above the 99.9th-percentile line at the chosen point.
- **Axis:** the 16 points. **Control:** the step slider, as in P0. **Random:** a random neuron.
- **Data:** S3 only. `w5/verdict/samples/q2_clusters.jsonl` (196,269,741 B); 16 × `w3/grid/{1,1b}/out/<step>/poles/exemplar_a.npz` (3,761,544 B in all); `t15_step1413814/poles/positions.npz` (120,603 B). The q999 per step is in the local contexts.jsonl.
- **Claim:** the got neuron's groups are born at steps 5,000, 20,000, 35,000 and 100,000 (A_training_dynamics §5).
- **Avoid:** a birth is a line crossing, not an event (lines 755, 923), and "born" is not "settled" (E_skeptic §3.2). Lithium "lost the top 20"; it was not "replaced" (§3.4).
- **Replaces:** 582, 659, 663–664, 617–656, 759–763. **Size:** about 0.3 MB.
- **Compute:** count exemplars with a⁺ ≥ q999. Assert that this gives back present_pct in all 2,454 × 16 cells, and fails with the wrong step's q999.

### P7. Did the new group grow from the old one
- **Sentence:** A judge sees a late group beside two earlier records, its own neuron's and a random neuron's, and picks its source.
- **Object:** OLMo's 1,238 late-born groups and 600 control groups (each neuron's strongest group against its own earlier record).
- **Count:** one dot per group with 4 marks, each filled when the judge picked the neuron's own record.
- **Axis:** marks filled, from 0 to 4 (not birth time). **Control** and **Random:** a "judge flipping a coin" button; under it, 5 of 16 dots show 0 or 1.
- **Data:** S3 only, `w4/o7_s_v2/explain/judgments.jsonl` (2,537,271 B; sha256 800fb2f7… in q2.inputs).
- **Claim:** the judge picked the true record in 81.2% of late judgments and 2,218 of 2,400 controls. 0 or 1 of 4 were right for 125 late groups and 19 controls. Untraceable share: 0.265 [0.211, 0.321] (RESULTS.md line 5).
- **Avoid:** "phase change", since there is no trend inside a neuron (299 against 309). Controls share a marked token more often (50.0% against 18.7%), and matched for that the share is 0.206 [0.132, 0.271] (E_skeptic §3.1). 4 of 4 is not descent (§4, A5).
- **Replaces:** 583–588, 668, 671, 699, 707–708. **Size:** about 40 KB.
- **Compute:** random-decoy judgments only. Assert 1,238; 0.8116; 2,218 of 2,400; 125 and 19. Judgment text needs q2_clusters.jsonl.

### P8. Is the strongest group the oldest
- **Sentence:** In how many neurons was the group strongest at the end also among the first to appear?
- **Object:** 574 OLMo neurons with two or more born groups. **Count:** a tally of rows whose strongest-group dot is leftmost.
- **Axis:** the 16 points. **Control** and **Random:** shuffle which group is strongest within each neuron.
- **Data:** S3 only, q2_clusters.jsonl.
- **Claim:** 273 of 574 with ties counted; shuffles average 238 and pass 258 in 2.5% of draws (metrics.json q2.birth_order). Alone at the front: 126 against a chance 17.5% (E_skeptic §4, A6).
- **Avoid:** "the main job is the oldest" (in 301 another came first), and "changed careers" (E_skeptic B7).
- **Replaces:** 700. **Size:** 30 KB. **Compute:** assert 273, and 258 from 10,000 shuffles.

### P9. Group births across all 600 neurons
- **Sentence:** This is how many groups first appeared at each training point.
- **Object:** OLMo's 2,454 groups. **Count:** stacked dots, with the strongest groups accented. **Axis:** the 16 points.
- **Control:** a switch between per point and per 1,000 steps. **Random:** none applies.
- **Data:** local, metrics.json q2.births.
- **Claim:** 546 of 600 strongest groups had crossed the line by step 560,000 (39.6%), and 1 of 2,454 at step 1,000.
- **Avoid:** "settled by 40%" and "a 190-fold fall". The fall is 25-fold against T14 (E_skeptic §3.2).
- **Replaces:** 709. **Size:** 2 KB. **Compute:** count over interval width.

### P10. Switch off 50 inputs
- **Sentence:** Switching off the 50 earlier neurons that feed one group most also lowers the neuron on its other group.
- **Object:** one case, a neuron and its newer group's contexts. **Count:** mean activation, as a bar before and after.
- **Axis:** activation. **Control:** which 50 are switched off: own, the other group's, random, another neuron's. **Random:** 50 random neurons, and a tick for the 56 fake two-neuron neurons.
- **Data:** local, `studies/reuse/results/birth/verdict_birth.json` (1,035,816 B; cases.primary_T15, pairs_T15).
- **Claim:** at the end of training the other group's inputs remove 0.294 [0.229, 0.345] of what its own remove; fake neurons 0.065 (reuse RESULTS.md 24–25).
- **Avoid:** "the newer group uses the older's inputs", since a third group's do as well (line 772).
- **Replaces:** 717–719, Figs 5–6. **Size:** 60 KB. **Compute:** assert 0.294 and 0.065.

### P11. The always-on neurons
- **Sentence:** A few dozen OLMo neurons have average activations far above the rest of their layer.
- **Object:** every neuron in layers 0–30 at the end of training. **Count:** dots above a 100× line.
- **Axis:** average activation over the layer median, on a log scale. **Control:** a layer slider. **Random:** Gemma and Qwen, if their per-neuron averages are banked (not confirmed).
- **Data:** local, `studies/instruments/results/e1/1/t15/abar.npz` (1,269,313 B; [31, 11,008]) and `high_background.json` (111,866 B).
- **Claim:** the 1% cut holds 3,412 neurons, 1,575 in layer 30 and none in layers 0–6; members entering input sets sit at 145× (instruments RESULTS.md Result 19).
- **Avoid:** the name fits only the few dozen (line 748). The normalization route was withdrawn (line 767).
- **Replaces:** 747–748. **Size:** 120 KB. **Compute:** assert 3,412 and the per-layer counts.

**Dropped:** the center birth curve (exploratory; E_skeptic §3.6; needs 1.5 GB from S3).

## (c) Ranking

Readiness: 3 is local, 2 is S3 files under 25 MB, 1 needs the 196 MB file. Clarity: 3 is a counted headline result, 2 is supporting.

| picture | readiness × clarity | build |
|---|---|---|
| P4 | 3 × 3 = 9 | **1** |
| P5 | 2 × 3 = 6 | **2** |
| P3 | 2 × 3 = 6 | **3** |
| P7 | 2 × 3 = 6 | **4** |
| P1 | 3 × 2 = 6 | **5** |
| P10, P11, P9 | 3 × 2 = 6 | |
| P6 | 1 × 3 = 3 | |
| P2 | 3 × 1 = 3 | |
| P8 | 1 × 2 = 2 | |

Ties go to clarity. P6 follows once q2_clusters.jsonl is local.

## (d) Files to bring down

From `s3://pg-lineage-769063704696/results/neurons/` into `/Users/abaybektursun/projects/interp_research_banks/neurons/`, at the same paths:

| file | bytes | for |
|---|---|---|
| w5/verdict/samples/q1_O7_poles.jsonl | 24,420,010 | P3, P4 |
| w5/verdict/samples/q1_G4_poles.jsonl | 23,685,393 | P4 |
| w5/verdict/samples/q1_Q4B_poles.jsonl | 23,450,997 | P4 |
| w4/o7_s_v2/explain/clusters.jsonl | 944,095 | P5 |
| w4/o7_s_v2/explain/judgments.jsonl | 2,537,271 | P7 |
| w5/verdict/samples/q2_clusters.jsonl | 196,269,741 | P6, P8 |
| w3/grid/{1,1b}/out/*/poles/exemplar_a.npz (16) | 3,761,544 | P6 |
| w3/grid/1/out/t15_step1413814/poles/positions.npz | 120,603 | P6 |

The total is 275,189,654 B; the top five need the first five rows (75,037,766 B). The local metrics.json matches RESULTS.md's sha256 a95bfa57… (run 7), and the S3 samples share its upload time.
