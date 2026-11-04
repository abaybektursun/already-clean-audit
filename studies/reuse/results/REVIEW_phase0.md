# REUSE SPEC3 Phase 0 review, 2026-11-01 (coordinator, Fable 5.1; instrument built by an Opus worker; this text checked by an Opus reader and corrected on 18 items)

Read-only on the bank. Own recount `../scripts/review/fable_phase0_recount.py`, output `phase0/recount_fable.txt`, from the raw answers, the keys and `primary/verdict.json`; it shares no code with `judge_verdict.py`.

**Verdict.** The coherence reader passed its constructed controls and found 109 of 129 clusterer jobs not coherent (78 with both halves mixed, 31 with one). Among the 20 coherent jobs the relatedness question failed its control again (13 of 20 related against 0.85). SPEC3 §1 row 2 applies: relatedness is not a reader-recoverable relation at this window, and R4 stays a claim under the clusterer's labels, marked so. Beyond the row, the coordinator's decision for the next specs: no further reader relatedness labels are bought. The REUSE ratios do not use the labels and do not change.

## 1. Snapshot (artifact times; CST is UTC−6)

Freeze 3 `60f6b33a` committed 19:31:25 (tag `reuse-freeze-3`; the file's status line still reads "draft", the freeze is the commit, and both manifests record the file's hash) → Stage 1 `msgbatch_019Ukz6kDxqxPUJAsS9N9uX3` submitted 19:31:29 (`submitted_utc` 01:31:29Z), 468 requests, answers landed 19:38:33, $1.022 → four answers printed, then `verdict_coherence.json` written 41 s after the answers, by one command in that order → Stage 2 `msgbatch_01Y6J4nnDEEVKntdHZQtZwr9` submitted 19:40:10, 62 requests, answers 19:46:13, $0.263 → three answers printed, then `verdict_relatedness.json` 38 s later. Phase 0 $1.286 on key 2; study API $4.04 (4.043); 0 GPU. The user's go for Phase 0 came in answer to the coordinator's question after the draft spec existed, at about 19:15; the journal entry of 19:40 wrote 18:25, the time of the earlier decisions, and is corrected in the journal. The journal headings "19:50" and "20:00" were written at about 19:44 and 19:49.

## 2. Verified, own recount against the verdict files

| quantity | verdict file | recount |
|---|---|---|
| same-token sets coherent; mixed sets mixed | 27 of 30; 30 of 30 | same |
| O* splits: both halves coherent; one; none | 20 of 129 (10/41, 4/45, 6/43); 31; 78 | same |
| J sets coherent | 43 of 150 | same |
| pairs with J and O* coherent | 12, all live | same |
| two-neuron sides unrelated; coherent jobs' halves related | 26 of 30; 13 of 20 | same |
| agreement with the clusterer on the 12 | 0.500 (3 of 9 related, 3 of 3 unrelated) | same |
| R4 under the clusterer's labels on the 12 | 0.466 [0.096, 0.588], seed 0, 44 of 2,000 draws non-finite | 0.466 [0.098, 0.588], seed 1, 39 non-finite; medians 0.565 (9) and 0.100 (3) |
| failing inputs | both gates fail under permuted labels | exit 1 both |
| costs | $1.022, $0.263 | usage logs at Batch prices: same |

## 3. Findings

**F1. The reader tells constructed one-role sets from constructed mixed sets, and calls most clusterer sets mixed.** Controls 0.90 and 1.00. Clusterer O* halves: 71 of 258 coherent; both halves coherent in 20 of 129 jobs (0.155); J sets 43 of 150 (0.287). H_coh (predicted 0.65, refuting line 0.50) is refuted. The reasons name parts of speech and positions. Caveat from the reader's keyword count: about 105 of the 294 "mixed" answers on clusterer sets say most tokens share one role with a few exceptions, so "mixed" under this question is a strict criterion.

**F2. Even coherent halves are often not "the same role" to the reader.** 13 of 20 pairs of coherent halves were called related (0.65). The example `L09N2031-__c2`, half A "majority" in voting thresholds and half B "honors" and "minimum", was called unrelated; its own ceiling read has I 0.207 and a shared-driver share of 0.08, the lowest of the 20, so here the reader agrees with the wiring. Across the 20, the reader's related calls have median shared share 0.58 and its unrelated calls 0.46: the label co-varies with shared drivers, weakly, on a small set. H_rel refuted; R4d not read.

**F3. What the three runs can and cannot separate.** They are two question wordings (topic; marked-token role, used in v2 and again here), on overlapping items: the same 30 two-neuron sides in all three, 7 of the 20 coherent splits among v2's controls, all 12 pairs among v2's 150. The clusterer and every reader are Sonnet 5.5 at temperature 1.0. The role question repeated on identical sets gave the same label on 26 of 30 two-neuron sides, 4 of 7 ceiling splits and 7 of 12 pairs (0.583). So the 0.500 agreement with the clusterer on the 12 is below the reader's agreement with itself, and the data cannot separate the relation from the model or from sampling noise. What they support: the same-job control fails at 0.63–0.67 in every run while the two-neuron sides pass at 0.87–0.97, and restricting to coherent jobs barely moves it (v2 on the 23 non-coherent controls 14 of 23, 0.61; Phase 0 on the coherent 13 of 20, 0.65).

**F4. The restricted R4 is not a quiet confirmation.** Under the clusterer's labels on the 12 coherent pairs, related − unrelated is 0.466 [0.096, 0.588] on 9 against 3; 44 of 2,000 bootstrap draws have no unrelated job and are dropped. Reported as the spec asked, not read as a rule.

**F5. Predictions (SPEC3 §4).** Coherent O* share 0.65 → 0.155, refuted (H_coh). Same-token sets coherent 0.95 → 0.90 and mixed sets mixed 0.95 → 1.00, no rule. Gate rel passes → fails, H_rel refuted. Agreement 0.75 → 0.50, refuted. R4d 0.25 [0.10, 0.40] → not read. R4 under the clusterer's labels on coherent pairs 0.25 → 0.466 [0.096, 0.588], no rule.

**F6. Errata in the frozen text.** SPEC3 §1 says both earlier questions labelled 29 of 30 two-neuron sides unrelated; v2 labelled 28 of 30. The spec file is left as hashed; the correction lives here.

## 4. Disposition

SPEC3 §1 row 2. Coordinator's decisions for `NEXT.md`: no further reader relatedness labels; the clusterer's label is kept as a descriptive split and named as the clusterer's; a neuron's "task" is a cluster of its top passages, which this reader mostly does not see as one role, and the post says so. Nothing in the REUSE ratios changes.
