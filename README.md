# An independent, pre-registered audit of "Neural Networks Are Already Clean"

Neuron-pole explanations on three open models, and tests of how a neuron's activation clusters depend on each other on
OLMo-3-7B. Six studies, each pre-registered. Updated 2026-11-04.

A neuron's clusters depend on each other's inputs, but the older cluster is not special and no event marks a new
cluster's arrival. On OLMo-3-7B, replacing one cluster's upstream drivers lowers the neuron on another cluster by 0.23 to
0.29 of the fall from its own drivers. Two unrelated neurons summed give 0.004 to 0.007, and one cluster split in two
gives 0.65. So the clusters are neither independent nor one cluster counted twice.

## Results

A pole is one sign of one MLP neuron. A cluster is one group within a pole's top-50 contexts. The ratio is the fall in
the neuron on cluster A when cluster B's top-50 drivers are replaced, divided by the fall when A's own are replaced.

| question | result | status | study |
|---|---|---|---|
| How many poles have one job? | 0.597, 0.563, 0.518 on three models under a blind test; not understood 0.320 to 0.415; reader agreement kappa 0.07 to 0.18 | frozen rule | [`results/RESULTS.md`](results/RESULTS.md) |
| Are a neuron's clusters independent? | No. Ratio 0.293 under a zero write (two-neuron null 0.007) and 0.231 under a write at the marked token alone (null 0.004); 69% of cases clear the null's 95th percentile against 5% of pairs | pre-registered, OLMo-3-7B only | [REUSE](studies/reuse/README.md), [INSTRUMENTS](studies/instruments/results/RESULTS.md) |
| Does a newer cluster depend most on the older one? | Not separated from the neuron's other clusters: d −0.046 [−0.091, 0.030] at the end of training, −0.014 [−0.076, 0.025] at birth | pre-registered; 127 of 135 comparison clusters share its lineage, so a lineage-wide dependence is not excluded | [REUSE](studies/reuse/README.md) |
| Does anything happen at a cluster's birth? | No signature fired: break at the birth in 0.307 [0.225, 0.395] of cases, shuffled null's 95th percentile 0.283; gain 1.05 [0.98, 1.27] times elsewhere | pre-registered | [REUSE](studies/reuse/results/RESULTS.md) |
| Does a pole's label predict what it writes? | Mostly when the label names an output token: D_w 0.553 to 0.618 against 0.030 to 0.096 for other labels | exploratory; no frozen headline | [NEURONRW](studies/neuronrw/CLOSEOUT.md) |
| Does removing a labelled pole matter? | Removing one pole moves own-context NLL by a median 0.0026 to 0.0093 nats; 0 of 7 code-construct poles changed behaviour | main comparison cancelled by its positive control | [NEURONOFF](studies/neuronoff/CLOSEOUT.md) |
| Does pushing a code neuron write that code? | Refuted on 3 of 3 gemma-3-4b-pt neurons; 3 Qwen3-4B-Base neurons not scorable | frozen rule | [REALPROMPT](studies/x/realprompt/REPORT.md) |
| Do shared drivers forecast the collateral fall? | r 0.81 [0.76, 0.85] over 277 cases | exploratory | [INSTRUMENTS](studies/instruments/results/RESULTS.md) |
| Does the value an ablation writes matter? | On OLMo-3-7B the null's 95th percentile runs from 0.098 to 0.575 by written value and exclusion; the null's median is 0.004 on the other two models | pre-registered | [INSTRUMENTS](studies/instruments/results/RESULTS.md) |

**Limits on these results.** The ratio has been read on OLMo-3-7B only, in one run. The clusters come from a language-model
clusterer, and a separate reader called 109 of 129 of them not coherent, so read "job" below as "cluster". The 0.65 split
was measured at birth under the mean write. The first study's unrelated share, 26.5% [21.1%, 32.1%], labels packing under
its frozen rule, but a matched ceiling gives 20.6% [13.2%, 27.1%]. Its object is the same clusters, so it is not the
headline here. INSTRUMENTS stage E6b had not closed at this copy and is not included.

## What is in this repository

This is a public, read-only copy of six studies from Abay Bektursun's private working repository: pre-registrations,
results, verdict files, figures and review reports. The raw data, the code and the study journals are not here. The
files at the top level are the first study, the audit ([Audit details](#audit-details)). The later studies are under
[`studies/`](studies/). Local paths and infrastructure names are redacted ([`REDACTIONS.md`](REDACTIONS.md)).

## Start here

| you want | open |
|---|---|
| the answer in five minutes | [Results](#results) on this page |
| the audit, the first study | [Audit details](#audit-details) |
| every reported number, with footnotes on defects, sensitivities and provenance | [`results/RESULTS.md`](results/RESULTS.md) |
| the hypotheses and the observation that would refute each | [`SPEC_D1.md` §1](SPEC_D1.md#1-question-decision-hypotheses-refuting-observations) |
| the measurement, step by step | [`SPEC_D1.md` §2](SPEC_D1.md#2-the-measurement-exactly) |
| the decision rules, frozen before any confirmation number was read | [`SPEC_D1.md` §6](SPEC_D1.md#6-frozen-decision-rules-set-on-the-discovery-sets-at-w2-d-before-any-confirmation-number-is-read) |
| the prediction grid filled at unblind | [`SPEC_D1.md` §11](SPEC_D1.md#11-prediction-grid-filled-once-at-w5-whatever-the-result) |
| the six figures | [`results/figures/png/`](results/figures/png/) (SVG in `results/figures/`, PDF in `results/figures/pdf/`) |
| which script and inputs produced each figure and number | [`results/REGISTRY.md`](results/REGISTRY.md) |
| the checker reports on the record | [`checks/CHECKERS.md`](checks/CHECKERS.md), then [`checks/acceptance_final.md`](checks/acceptance_final.md) |
| every change made for this public copy | [`REDACTIONS.md`](REDACTIONS.md) |

![Bar chart of Q1 class fractions per model with Wilson 95% intervals: one job on 56.3% of G4 poles, 51.8% of O7 poles and 59.7% of Q4B poles. Dashed lines mark the essay's 90 / 7 / 2–4, labelled as its own thresholds, not this study's.](results/figures/png/F1_audit_classes.png)

*F1. Share of the 600 poles per model in each class, with Wilson 95% intervals. The dashed lines are the essay's
estimate, drawn as a labelled reference, never as this study's threshold. The footer names the `metrics.json` the figure
was drawn from.*

## Audit details

Everything from here to the end of the page describes the first study.

### What the study measures

The essay is "Neural Networks Are Already Clean" (Nick Cammarata, Goodfire;
https://already-clean.goodfire.com/essay.md). It reports ≈ 90 / 7 / 2–4 (one main job / two related jobs / not yet
understood) on Qwen3-32B. It also says that the later jobs of a neuron grew out of its earlier ones wherever it has
looked, and that the checkpoints to test this on a large language model did not exist. When the study was written, no
outsider had reproduced any number in the essay (`SPEC_D1.md` §1).

A pole is one sign of one MLP neuron's activation. The study reads 600 poles per model.

| key | model | used for |
|---|---|---|
| Q4B | Qwen3-4B-Base | Q1 |
| G4 | Gemma-3-4B-pt | Q1 |
| O7 | OLMo-3-1025-7B at its end-of-stage-1 checkpoint `stage1-step1413814`, not the released model | Q1; Q2 across 16 public stage-1 checkpoints from step 0 to step 1,413,814 |

**Q1, audit.** What fraction of poles have one job (ONE), two related jobs (TWO-REL), unrelated jobs (UNREL), or are
not understood (UNK)? An explanation counts only if it passes the frozen §6.1 test. A writer that sees only the
one-sentence explanation, never a context or an activation, writes five positives meant to drive the pole, five near
misses meant not to, and five texts on the most distant concept it can invent. A text's score is its strongest
activation at the pole, in the pole's sign, divided by the pole's largest absolute activation over the study corpus. Texts
written from the explanation must score at or above θ_s = 0.3153 (the 99th percentile of shuffled-pole scores on a
discovery set) on at least 3 of 5 positives, while at most 2 of 5 near misses and at most 1 of 5 distant texts do.
Explanations were written by Claude Opus 5.5 and Claude Sonnet 5.5; a second label per pole, for the agreement kappa,
came from DeepSeek-V4.1-Flash-1017.

**Q2, development.** Did the jobs an O7 pole has at the end of pretraining grow out of its earlier jobs? On the 600 O7
audit poles (the set S), a clusterer groups each pole's top-50 contexts at the last checkpoint into jobs. A job's birth
is the first checkpoint at which it is present under a frozen activation rule and stays present at the next one
(`SPEC_D1.md` §2.4). A blind judge (Claude Sonnet 5.5) picks each late-born job's true earlier record against a decoy.
The unrelated share `p_u` follows from the judge's accuracy on late-born jobs against its accuracy on a positive
control (§6.3):

```
p_u = (a_ctrl − A) / (a_ctrl − 0.5)
```

`A` is the judge's mean accuracy on late-born jobs against a random decoy. `a_ctrl` is its accuracy on the positive
control: each pole's main job, whose true record is the pole's own record at the previous grid checkpoint, a pair
related by construction. 0.5 is chance. The value is clipped to [0, 1].

| hypothesis | claim | the observation that refutes it |
|---|---|---|
| H_RG, recursive generalization | every job a pole has at the end of training traces to an earlier job of the same pole | `p_u` lower bound above 0.20 |
| H_SP, packing | a substantial fraction of later jobs have no antecedent in the pole | `p_u` upper bound below 0.10 |
| H_VEST, vestigial mixtures | extra jobs are leftovers from early training, born at or before the main job and shrinking | extra jobs mostly born after the main job, or mostly growing over the last half |
| H_MIG, replacement | the pole's job is replaced wholesale by an unrelated new one | consecutive dominant clusters pass the parent test |

The hypotheses, the observation that would refute each, and the decision rules are in `SPEC_D1.md` §1 and §6.

### Result

#### Q1: one-job fractions

Under explanations that pass the frozen §6.1 rule, the one-job fractions are 0.597 [0.557, 0.635] on Q4B,
0.563 [0.523, 0.602] on G4 and 0.518 [0.478, 0.558] on O7 (Wilson 95%, 600 poles each). Not understood (UNK) is
0.320–0.415 and unrelated jobs (UNREL) at most 0.013.

| class | Q4B | G4 | O7 |
|---|---|---|---|
| ONE: one job | 0.597 [0.557, 0.635] | 0.563 [0.523, 0.602] | 0.518 [0.478, 0.558] |
| TWO-REL: two or three related jobs | 0.057 [0.041, 0.078] | 0.060 [0.044, 0.082] | 0.060 [0.044, 0.082] |
| UNREL: unrelated jobs, each supported on its own | 0.013 [0.007, 0.026] | 0.012 [0.006, 0.024] | 0.007 [0.003, 0.017] |
| UNK: not understood | 0.320 [0.284, 0.358] | 0.365 [0.327, 0.404] | 0.415 [0.376, 0.455] |
| RARE: too few strong activations in the corpus to classify | 0.013 [0.007, 0.026] | 0.000 [0.000, 0.006] | 0.000 [0.000, 0.006] |
| label agreement with the second reader (kappa) | 0.180 | 0.167 | 0.070 |

The essay's ≈ 90 / 7 / 2–4 (one main job / two related jobs / not yet understood, on Qwen3-32B) is its own estimate
under its own thresholds; the essay says the number depends on where the threshold for weak activations is set. This
study's one-job fractions are under the thresholds frozen in §6.1, on different models, with this study's class
definitions, so the two sets of numbers are not the same quantity. This is an independent measurement, not a
refutation.

#### Q2: did the later jobs grow out of earlier ones?

The developmental outcome (Q2, O7, 600 poles) is mixed and is reported as mixed under §6.7. The unrelated share of
late-born jobs is p_u = 0.265 [0.211, 0.321]; its lower bound is above the frozen 0.20 rule, so the rule labels the
packing hypothesis H_SP (the post-closeout sensitivity under [Limits](#limits) gives 0.206 [0.132, 0.271]; the rule is
not applied to it). In the same data the main job is the oldest job more often than a permutation null allows
(b = 0.476 [0.436, 0.517], null 97.5th percentile 0.449, p = 0.0005, Holm 0.0010), which supports the
main-job-is-oldest form of recursive generalization (H_RG). The vestigial reading is not supported
(v = 0.035 [0.026, 0.044]). In 17 poles (2.8%) the main job at the last checkpoint is late-born and failed the parent
test (the H_MIG count).

| readout | frozen rule | observed | label |
|---|---|---|---|
| `p_u`, unrelated share of late-born jobs (§6.3) | H_RG if the upper bound < 0.10; H_SP if the lower bound > 0.20; mixed otherwise | 0.265 [0.211, 0.321]; A = 0.812, a_ctrl = 0.924, 1,238 late-born clusters | H_SP |
| `b`, main job is the earliest-born (§6.4) | supports the main-job-is-oldest form of H_RG if above the permutation null's 97.5th percentile | 0.476 [0.436, 0.517] over 574 poles; null 97.5th percentile 0.449; p = 0.0005, Holm 0.0010 | main oldest (above band) |
| `v`, vestigial share (§6.5) | H_VEST only if the lower bound > 0.5 | 0.035 [0.026, 0.044] | not H_VEST |
| H_MIG, replacement (§6.6) | a count, no rule | 17 poles (2.8% of S) | count |
| hard decoy, judge separates by domain alone (secondary) | not measurable if its control is below 0.80 | control 0.798 | not measurable |

Every number is in [`results/RESULTS.md`](results/RESULTS.md); the prediction grid filled at unblind is
`SPEC_D1.md` §11.

![Left: judge accuracy 0.812 on late-born clusters and 0.924 on controls against a random decoy, 0.697 and 0.798 against the hard decoy, with chance at 0.5. Right: p_u = 0.265 with its interval, above the H_SP line at 0.20 and the H_RG line at 0.10.](results/figures/png/F5_parent_test.png)

*F5. Parent-test accuracy on late-born clusters and on the positive controls, against random and hard decoys, with
chance drawn at 0.5. Right panel: `p_u` with its interval and the two frozen rule lines.*

### Limits

Limits stated in the record:

- **The margin of the H_SP label.** One sensitivity bears on the H_SP label: the judge's control items share a marked
  token with the true record in 50.0% of cases and the late-born test items in 18.7%, and re-weighting the control
  ceiling to the test items' mix gives p_u = 0.206 [0.132, 0.271]. Token overlap is partly the outcome being measured,
  so this is a sensitivity bound, not a corrected estimate. The frozen readout stands, but its margin above 0.20 rests
  on a ceiling measured on easier items. That sensitivity was computed after closeout and is recorded in the study
  journal (entry 2026-10-24T08:20Z), which is not part of this copy.
- **Hard decoy.** The hard-decoy control scored 0.798, below its 0.80 gate, so that row is not measurable.
- **Manual backend.** On S, every judge line and most clusterer answers came from Claude Sonnet 5.5 running as Claude
  Code subagents (the manual backend), not from the API that served the gate. The model and prompt match the gate; the
  sampling differs (`results/RESULTS.md` footnotes, with the manual-backend audit and its leave-out readings).
- **Label agreement.** Cross-model label agreement is low (kappa 0.07–0.18).
- **Explainer model.** On the 117-pole discovery set, 33 poles (28%) pass §6.1 under one explainer model and fail under
  the other. On the audit sets the explainer model also tracks depth (Opus for most shallow poles, Sonnet for most deep
  ones).
- **F41.** A defect found after closeout (F41, `results/RESULTS.md` footnotes) showed each negative pole the positive
  pole's output-vocabulary line in the explainer and second-reader prompts. The clusterer and judge never saw that line,
  so Q2 is untouched.
- **F42.** A second post-closeout finding (F42, `SPEC_D1.md` §15 item 18): in Gemma-3-4B layers 10–11, 25 windows split
  at a CRLF hold 3.44% and 5.19% of top-50 slots. Three sampled G4 poles are affected, all UNK, so no count moves.

### Freeze record

| event | time (UTC) | where it is recorded |
|---|---|---|
| Essay text fetched; the study cites it by line of this plain text | 2026-10-19 | `SPEC_D1.md` §0 |
| Predictions entry | heading 2026-10-20T02:58:11Z, committed 2026-10-20T03:13:32Z | `results/RESULTS.md` line 81 |
| Pre-registration frozen | 2026-10-20T16:50:32Z | annotated git tag `already-clean-freeze-2026-10-20`, which points at commit `dc18799af27c47631eafad2a6d2018ba578b6970` (dc18799a) on branch `drafts/already-clean` of the private working repository |
| Model-policy amendment | 2026-10-20 17:15Z | `SPEC_D1.md` §2.2 |
| First unblind | 2026-10-23T19:55:33Z | `results/RESULTS.md` provenance footnote, `SPEC_D1.md` §11 |
| Reported numbers: the seventh run | unblind 2026-10-24T04:06Z (2026-10-24T04:06:05.084615Z) | `metrics.json` sha256 `a95bfa57bfe65246b00950d094f9fee5e87743ec2945296d1337b59a1e717e16` (`results/RESULTS.md` line 3) |
| This copy taken | 2026-10-25T04:26:07Z | commit 85ad5e37 of `drafts/already-clean` |

- **What the freeze fixed.** The §6 rules, prompt v3 and θ_s = 0.3153 were set at the tag before any confirmation
  number was read. The audit harvests (GPU forward passes only, no explanation written and no number read) ran earlier
  that day. The clusterer and judge were not part of the freeze: they were chosen afterwards on the separate gate set
  `S_gate` (`SPEC_D1.md` §2.2, §4.4).
- **The amendment.** A dated amendment (§2.2, 2026-10-20 17:15Z) superseded the model half of the freeze: the explainer
  for the poles without a banked Opus 5.5 answer (456 of the 1,800 audit poles) and the writer and job writer for every
  pole moved to Sonnet 5.5, and the second reader to DeepSeek-V4.1-Flash-1017. Amendments after the freeze carry their
  dates in place; `SPEC_D1.md` §0 lists the main ones.
- **Re-runs.** Six re-runs followed the first unblind, to add readouts or after inputs were corrected in answer to
  checker and audit findings. p_u stayed within 0.260–0.265 and every label stayed the same across the seven runs
  (`results/RESULTS.md` provenance footnote, `SPEC_D1.md` §11).
- **The reported file.** The `metrics.json` file itself was hashed again on 2026-10-31 and matches the sha256 above.
- **Code at the reported run.** `neurons/explain` d832b80f, `neurons/harvest` bca491a9 (`INDEX.md`).
- **History.** This copy carries none of the private repository's history, so the tag and commits above cannot be
  checked from this repository alone.
- **Essay version.** The live page now carries a footnote dated 2026-10-30 that revises one sentence about an
  InceptionV1 neuron, so line numbers may have shifted. The 90 / 7 / 2–4 sentence was still on the page on 2026-10-31.

### Files

| path | what it is |
|---|---|
| [`README.md`](README.md) | this file |
| [`studies/`](studies/) | the five later studies: REUSE, INSTRUMENTS, NEURONOFF, NEURONRW, REALPROMPT |
| [`SPEC_D1.md`](SPEC_D1.md) | the pre-registration: question, hypotheses with refuting observations, measurement, frozen rules (§6), controls, runbook, prediction grid (§11, filled at unblind), budget, literature record, pre-mortem, closures (§15); amendments after the freeze carry their dates in place |
| [`DATA_CONTRACT.md`](DATA_CONTRACT.md) | file formats, paths and invariants of the harvest, explain and verdict stages |
| [`INDEX.md`](INDEX.md) | the study's own index of its files, written for the private working repository; several files it names are not in this copy |
| [`results/RESULTS.md`](results/RESULTS.md) | every reported number, written from `metrics.json` (sha256 on line 3), with footnotes on defects, sensitivities and provenance |
| [`results/REGISTRY.md`](results/REGISTRY.md) | each figure and number mapped to its generating script, input hashes, and DISCOVERY or CONFIRMATORY |
| [`results/README.md`](results/README.md) | where the result copies came from |
| [`results/figures/`](results/figures/) | F1–F6 as SVG, with `pdf/` and `png/` renders; each footer prints the metrics sha256 it was drawn from (table below) |
| [`checks/`](checks/) | read-only checker reports (table below) |
| [`REDACTIONS.md`](REDACTIONS.md) | every redaction (file, line, pattern class, replacement), the two reworded `SPEC_D1.md` lines, and the reviewer fixes applied |
| [`CITATION.cff`](CITATION.cff) | citation metadata, read by GitHub's "Cite this repository" |
| [`LICENSE`](LICENSE) | the license: CC BY 4.0 for text and figures |

The figures are byte-identical to the study's copies. The text files differ from the study's files only on the lines
listed in `REDACTIONS.md`.

#### Figures

| figure | what it shows |
|---|---|
| [F1](results/figures/png/F1_audit_classes.png) | Q1 class fractions, with the essay's estimate drawn as a labelled reference |
| [F2](results/figures/png/F2_strong_rates.png) | strong rates for positives, near misses, distant and shuffled texts |
| [F3](results/figures/png/F3_grid_population.png) | grid population |
| [F4](results/figures/png/F4_births.png) | cluster births, with the §6.4 null |
| [F5](results/figures/png/F5_parent_test.png) | parent-test accuracy and `p_u` |
| [F6](results/figures/png/F6_gallery.png) | a three-pole gallery |

#### Checker reports

The reports were written by Claude Opus checker agents, except `acceptance_final.md`, which is the output of a script.
The brief is in `checks/CHECKERS.md`. The study journal, which holds the answers to the flags, is not part of this copy.
W0–W5 are the study's waves (`SPEC_D1.md` §9.2).

| file | what it is |
|---|---|
| [`checks/CHECKERS.md`](checks/CHECKERS.md) | the brief and the list |
| [`checks/W1-W3.md`](checks/W1-W3.md) | check of W0–W3 and the explain side of W2-C |
| [`checks/W4-W5.md`](checks/W4-W5.md) | check of W4 (clustering and judge on the manual backend) and W5 (unblind) |
| [`checks/closeout_W5.md`](checks/closeout_W5.md) | closeout check |
| [`checks/closeout_W5_recheck.md`](checks/closeout_W5_recheck.md) | re-check of the answers to the closeout flags |
| [`checks/literature_W5.md`](checks/literature_W5.md) | literature read at source |
| [`checks/acceptance_final.md`](checks/acceptance_final.md) | mechanical acceptance check: 15 PASS, 2 SUBSTANCE, 0 FAIL |
| [`checks/closeout_slips_2026-10-25.md`](checks/closeout_slips_2026-10-25.md) | reviewer items found after closeout, checked against the files |
| [`checks/w5_rerun_2026-10-25.md`](checks/w5_rerun_2026-10-25.md) | re-derivation of the unblind from banked inputs |
| [`checks/substantive_25.txt`](checks/substantive_25.txt), [`checks/nonpassing_172.txt`](checks/nonpassing_172.txt) | chunk lists used in the manual-backend audit |

### What is not here

- The raw harvests (per-checkpoint top-50 activations and statistics for every pole), the explain directories
  (requests, responses, explanations, probe texts and scores, clusters, judgments, per-call records), the
  manual-backend chunk outputs and the 977 answering-agent transcripts. They are on S3 in a private bucket; `INDEX.md`
  gives their paths and object counts, with the bucket name redacted.
- The verdict outputs `metrics.json`, `grid.md` and the per-pole and per-cluster sample files (23–24 MB per Q1 model,
  196 MB for the Q2 clusters). They are in a local working directory and on S3; the sha256 above identifies
  `metrics.json`.
- The study journal, the session handoff notes, the runbook scripts, the W0 verification record, the parent documents
  (`DIRECTIONS.md`, `IDEAS.md`, `THEORIES.md`, `CHECKS.md`) and the code (`infra/pg/neurons_*.py`,
  `infra/aws_pg/jobs/neurons_*.py`). They stay in the private working repository; the copied files refer to them by
  name.
- The essay itself; read it at the link above.

## How to cite

Abay Bektursun (2026). An independent, pre-registered audit of "Neural Networks Are Already Clean": neuron-pole
explanations on three open models and a developmental test on OLMo-3-7B checkpoints.
https://github.com/abaybektursun/already-clean-audit. Pre-registration tag `already-clean-freeze-2026-10-20`
(commit dc18799a); results from `metrics.json` sha256 a95bfa57bfe65246….

```bibtex
@misc{bektursun2026alreadyclean,
  author       = {Bektursun, Abay},
  title        = {An independent, pre-registered audit of ``Neural Networks Are Already Clean'':
                  neuron-pole explanations on three open models and a developmental test on OLMo-3-7B checkpoints},
  year         = {2026},
  howpublished = {\url{https://github.com/abaybektursun/already-clean-audit}},
  note         = {Pre-registration tag already-clean-freeze-2026-10-20 (commit dc18799a); metrics.json sha256 a95bfa57bfe65246}
}
```

`CITATION.cff` carries the same metadata for GitHub's "Cite this repository".

## License

CC BY 4.0 for text and figures, https://creativecommons.org/licenses/by/4.0/. See [`LICENSE`](LICENSE).
