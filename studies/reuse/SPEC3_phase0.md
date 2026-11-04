# SPEC3 Phase 0: can the relatedness of two tasks be read by an independent judge? A two-stage label instrument

Status: **draft, 2026-11-01 19:10 CST, not frozen.** Freeze is the commit after the item builder's `--dry-run` passes, on the user's go; no API call before it. Parents: `SPEC.md` §12, `SPEC2.md` §4, `results/REVIEW_amend2.md` F6 and §7, `NEXT.md` step 0. Written by the coordinator (Fable 5.1), who has seen both judge batches' answers; every threshold below is set on constructed controls, not on the primary pairs.

## 1. Why, and what each outcome changes

Two judge questions (topic, SPEC2 §4; marked-token role, exploratory) each labelled 29 of 30 two-neuron sides unrelated and only 20 and 19 of 30 same-job splits related. The same-job control is "related by construction" only if the clusterer's job is coherent enough for its two document-disjoint halves to read as one kind of thing; for about a third of these jobs it is not. R4 (related − unrelated I at birth, 0.227 [0.133, 0.297]) therefore rests on the clusterer's labels alone.

| outcome | the post says |
|---|---|
| coherence gate passes and the relatedness gate passes on coherent pairs | R4 under independent labels, with the agreement to the clusterer; R4 on the agreeing pairs |
| coherence gate passes, relatedness gate fails on coherent pairs | relatedness is not a reader-recoverable relation at this window; R4 stays clusterer-only and is marked so |
| coherence gate fails | the instrument cannot read coherence either; the program drops relatedness labels until a judge passes constructed controls |

## 2. Hypotheses, each with the observation that refutes it

| hypothesis | refuted when |
|---|---|
| H_coh: most clusterer jobs are coherent to an independent reader | the share of primary O* jobs judged coherent is under 0.50 (both halves) |
| H_rel: among coherent pairs, an independent reader reproduces the clusterer's relatedness | agreement under 0.65 on coherent pairs, or the relatedness gate fails |
| H_R4: R4 holds under independent labels | the interval of median I related − unrelated under judge labels contains 0 |

## 3. Instrument

**Stage 1, coherence, one set per item.** Question, fixed: "Here is one set of passages, each with one marked token. Do the marked tokens play one kind of role across the set (the same kind of word or symbol, in the same kind of position, doing the same kind of work)? Ignore what the passages are about. Answer coherent or mixed and one sentence why." Window 64 tokens before the marked token, 16 after (the v2 render). Items: each primary O* job's two ceiling halves (129 splits × 2 = 258 sets); each primary J (150 sets, 6 to 12 passages); controls, blind in the same batch: 30 **same-token sets** (passages of one job whose marked token strings are all identical; coherent by construction; drawn with seed 0 from the jobs that have at least 6 such passages) and 30 **mixed sets** (6 passages from one neuron's job and 6 from another neuron's job, the two-neuron sides of S3, seed 0; mixed by construction). Gate coh: at least 0.85 of the same-token sets labelled coherent and 0.85 of the mixed sets labelled mixed. Failing input: labels permuted. A job is coherent when both of its halves are labelled coherent; J is coherent when its set is.

**Stage 2, relatedness, one pair per item, run only if gate coh passes.** The SPEC2 v2 question, unchanged, on the 64/16 window. Items: the primary pairs whose J and O* are both coherent; controls: the ceiling splits of coherent O* jobs (related by construction, now that coherence is established by a separate question; up to 30, seed 0) and 30 two-neuron sides (unrelated by construction). Gate rel: 0.85 of each control group as constructed. Failing input: labels permuted. Refusals are declines, in the denominator.

**Rule R4d.** If gate rel passes: R4 as frozen (SPEC §6) over the live coherent pairs under the judge's labels, bootstrap over neurons, 2,000 draws, seed 0; interval above 0 → holds. Reported: agreement with the clusterer on the coherent pairs; R4 on the pairs where both sources agree; the share of coherent jobs by tercile; R4 under the clusterer's labels restricted to coherent pairs (so the restriction's own effect is visible).

## 4. Point predictions, no rule attached

Share of O* jobs coherent 0.65; same-token sets coherent 0.95; mixed sets mixed 0.95; gate rel passes with the ceiling splits of coherent jobs at 0.90 related; agreement with the clusterer on coherent pairs 0.75; R4d 0.25 [0.10, 0.40]; R4 under clusterer labels on coherent pairs 0.25.

## 5. Budget and runbook

Stage 1 about 468 requests × 3k input tokens ≈ 1.4M tokens, $1.5 at Sonnet 5.5 Batch, effort high; Stage 2 at most 210 requests, $0.9; total under $3 on key 2 (register: `infra/operations/credits/ANTHROPIC_KEYS.md`). 0 GPU: every I is already banked. Runbook: an Opus worker extends `scripts/judge_items.py` with `--mode coherence` (one set per request, the two control constructions) and `scripts/judge_verdict.py` with `--stage coherence|relatedness`; dry runs on fabricated answers (controls as constructed → gates pass; permuted → fail); freeze commit; `run-batch --dry-run`, then the user's go; Stage 1 submitted and collected; coherence read; Stage 2 built from it, dry-run, submitted; R4d once; a different-model read; `RESULTS.md` Result 19; the post's label paragraph revised.

## 6. Limits known now

A reader's coherence is not the clusterer's grouping criterion; a job can be one driver pattern and still read as mixed, so a low coherence share is a statement about readability, not about the clusterer being wrong. Labels are read on T_15 passages at a 64-token window; the primary I is read at birth. The same-token positive control is easier than a typical job; the gate it sets is a floor on the instrument, not on the jobs.
