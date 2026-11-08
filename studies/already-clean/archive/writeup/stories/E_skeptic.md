# E. Hostile but fair review of story reports A to D

- UTC span: 2026-10-24T07:46:40Z to 2026-10-24T08:10Z.
- Reviewer model: Claude Opus 5.5 (`claude-opus-5-5`).
- Method: read-only Python over the raw files. Nothing was written except this file.
- Paths: `E/` = `/Users/abaybektursun/projects/interp_research-neurons-explain/`. `S/` = `/Users/abaybektursun/projects/interp_research-drafts/studies/already-clean/`. `H/` = `/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/w3/grid/{1,1b}/out/<step>/stats/`.
- Main sources: `E/bank/w5/verdict/samples/q2_clusters.jsonl` (C), `q1_{Q4B,G4,O7}_poles.jsonl` (P), `E/bank/w5/verdict/metrics.json` (M), `E/bank/w4/o7_s_v2/explain/` (clusterer requests, clusters, judgments, calls), `E/infra/pg/neurons_verdict.py` lines 324 to 430.

## Bottom line

The arithmetic in all four reports is good. Of 68 recomputed claims, 54 match exactly. Every quote I looked up exists in the data, apart from small formatting changes. The trouble is interpretation. Three of the most quotable hooks fail a direct test: the birth-time ramp in A1 and B2, the Trello story in A4, and the "two clocks" story in A8. Three more misstate what the data shows: the timing in B4, the "career change" framing in B7, and the flag rate in B8. The "532 vs 26" contrast in B6 compares two prompts that define a job differently.

## 1. Recomputed claims

"Match" means my value equals theirs to the stated precision.

| report | claim | their value | my value | verdict |
|---|---|---|---|---|
| A1, B2 | late-born primary jobs, pooled judge accuracy | 1,238; 0.812 | 1,238; 0.8116 | match |
| A1 | a_ctrl on S | 0.924 | 2,218 of 2,400 = 0.9242 | match |
| A1 | p_u by birth bin T_2-9, T_10-11, T_12, T_13, T_14-15 | -0.013, 0.216, 0.272, 0.371, 0.420 | same | match |
| A1 | pole-bootstrap intervals (4,000 draws, a_ctrl fixed) | [-0.073, 0.051] to [0.322, 0.525] | same, all five bins | match |
| B2 | p_u windows T_4-9, T_10-12, T_13-15 with intervals | 0 [0, 0.059]; 0.242 [0.169, 0.317]; 0.395 [0.321, 0.472] | -0.013 clipped to 0 [0, 0.055]; 0.242 [0.171, 0.314]; 0.395 [0.317, 0.476] | match (bootstrap noise) |
| A1 | flag rates: 0 of 53 by 20k, then 4.3, 9.1, 11.3, 13.3, 14.6, 11.8% | as stated | same | match |
| B2 | flagged late jobs born at or before T_8 | 0 of 120 | 0 of 120 | match |
| A1 | ramp per depth tercile with own a_ctrl (0.882, 0.934, 0.960) | -0.03/0.36/0.34; 0.00/0.27/0.44; 0.11/0.22/0.35 | same | match |
| B2 | p_u for jobs with 11 or more contexts by window | 0.024, 0.110, 0.256 | same | match |
| A2, B1 | lithium records: T_9 20/20, T_11 19/20, T_12 19/20 | as stated | same | match |
| A2, B1 | T_13 record "/licenses" 8, " OpenSSL" 5, " Euler" 3, " License" 2 | as stated | same | match |
| A2 | T_14 record " ore" 4, " Cassandra" 3, " lithium" 3 | as stated | " Ore" 2 + " ore" 2, " Cassandra" 3, " lithium" 3 | match (case merged) |
| A2, B1 | 15 jobs, most of 600; runner-up 12; 9 of 12 late jobs flagged; next pole 4 | as stated | same | match |
| A3 | cumulative main births by T_11, T_12, T_13 | 56.7, 72.8, 91.0% | same | match |
| A3 | cumulative extra births by T_11, T_12, T_13 | 45.1, 65.0, 85.5% | same | match |
| A3 | jobs born at T_14 or T_15 | 322 of 2,454 = 13.1% | same | match |
| A3 | births per 1,000 steps, peak over last | 16.3 at T_5, 0.087 at T_15, about 188 | 16.33, 0.087, 187.7 | match. Peak over T_14 is 25 |
| A4 | Trello records T_8 and T_9 | "rello" 20 of 20 each | same | match |
| A4 | "At the end, Trello is gone" | gone | 1 of 50 T_15 top contexts is T⟦rello⟧; job c2 is named "... (Viber, Discord, Messenger, Notion, Trello)" | MISMATCH |
| A5 | poles with 3 or more late jobs, all 4/4 | 45 | 45 | match |
| A5 | "got" records: T_6 Got 18, T_7 onna 15 and Gott 5, T_9 onna 13, T_14 otta 14 | as stated | same (case merged) | match |
| A6 | single earliest job; main uniquely oldest; chance | 365 of 574; 126 (22.0%); 17.5% | same; permutation p = 0.0015, null 97.5th pct 117 | match |
| A6 | drift poles; gap of 8 or more checkpoints | 301 of 574; 10 | same | match |
| A7 | layers 0-10 record overlap, 100k vs 1M | median 0%, 55% none, 38 poles | 0.0, 0.553, 38 | match |
| A7 | layers 0-10 record overlap, 560k vs 1M | median 15%, 14% none, 78 poles | 0.15, 0.141, 78 | match |
| A7, B4 | L06N7759+ T_8 record | "/?" 11, "?" 6 | same | match |
| A7, B9 | L12N1003- T_9 "chemical" | A 13, B 12 | " chemical" 11, "chemical" 1, " Chemical" 1 | both reproduce under different case rules |
| A7 | L03N10451+ T_8 "Knight" 10 of 20, all "Dark Knight" | as stated | 11 marks (one lowercase), all Dark Knight | match |
| A8 | median first crossing of 0.3: jobs T_11, center T_3 | as stated | T_11; T_3 | match, but see section 3.6 |
| A8 | median normalized job value at 12k, 35k, 100k, 180k, 320k, 560k, end | 0.012 to 0.681 | same | match |
| A10 | zero-median share at T_0, 12k, 60k, 100k, 180k, 320k, 560k | 44.5, 25.1, 16.1, 11.0, 6.9, 3.0, 0.5% | same | match |
| A10 | jobs with a zero after init | 1,758 (71.6%) | same | match |
| A10 | zero share at 12k by tercile; main birth medians by tercile | 34.7/24.2/12.2%; T_12/T_11/T_10 | same | match |
| A10 | poles with no job until 1M; until 560k | 14; 78 | same | match |
| A11 | extra jobs at or before main; extra shrank; mains shrank; ever 2x final | 1,071; 92; 20; 2 | same | match |
| A Part 2 | median max\|a\| over final at init, 3k, 60k, 100k, 560k, 1M | 180, 392, 22, 7.3, 2.4, 1.6 | 180.3, 391.6, 22.4, 7.3, 2.4, 1.6 | match |
| A Part 2 | q999/max at init and 560k to 1M | 0.377; 0.178 | same | match |
| A Part 2 | r at birth p95; count at 0.9 or more; flagged vs 4/4 r; growth | 0.555; 0; 0.290 vs 0.291; 0.537 vs 0.382 | same | match |
| A Part 2 | ONE poles with one cluster; median clusters | 3.5%; 4 | 11 of 311; 4 | match |
| B3 | React T_11 record | Truly 9, uly 7, Seriously 2, Exactly 2 | same (case merged) | match |
| B3 | "The judge picked the loan sharks. It got 1 of 4 right." | 1 of 4 | 1 of 4. The loan-shark decoy was shown twice and picked once. The other two misses were against a "Writer / Created" decoy | partly misleading |
| B3 | React near misses; "Exactly" job strong texts | 0.784, 0.201, 0.023, 0.089; 2 of 5 | same | match |
| B4 | hook: "By step 60k it was the laughing neuron" | 60k | at 180k (T_11) its top-20 still had "?" 6 and "/?" 4 against "aha" 8; laugh job born T_9 at 0.19 of max | MISMATCH |
| B6 | explainer 1 job; second reader ONE; clusterer 1 job; mean; max | 532; 561; 26; 4.09; 15 | same | match (definitions differ, section 3.5) |
| B6 | sample-0 pairs marked unrelated | 3,051 of 4,829 | same | match |
| B6 | two samples give equal counts; sample-1 mean | 182 of 600; 3.375 | same | match |
| B7 | H_MIG mains scoring 0 of 4 | 11 of 17 | 11 | match |
| B8 | mean jobs; flagged share; p_u by layer quartile | 4.92/3.82/3.94/3.67; 0.143/0.102/0.085/0.051; 0.307/0.255/0.279/0.172 | same. p_u uses each quartile's own a_ctrl, not stated in the story | match |
| B8 | "fragment"-type job names; in layers 0-7 | 420; 213 | 390/199 to 449/231 by regex | fragile |
| B10 | hard controls right; on flagged; 4/4 random and 0/2 hard; mains both hard wrong; late jobs 0/4 | 958/1,200; 89/250; 109; 82; 70 | same | match |
| B11 | main not the biggest cluster; seven poles with main of 3 or fewer beside 30 or more | 279 of 574; 7 | same | match |
| B11 | "In 56 of them it beats the runner-up by less than 2%" | 56 | 36, 38, 43 or 58 under four definitions | does not reproduce |
| B5, B numbers | p_u ONE 0.242, UNK 0.277; UNREL 13 jobs A 0.577, p_u 0.795 | as stated | same | match |
| B numbers | poles with a "misc" job, 18 as main; poles with all late jobs flagged | 155, 18; 19 of 476, 3 with 2 or more | same | match |
| C1 | five near-silent Q4B maxima and densities; peaks of other layer-1 neurons | 0.002657 to 0.005143; 0.8966 to 17.22 | same | match |
| C1 | poles below 1% of model median max; cross-neuron pairs sharing 5+ contexts | 23/0/0; 22/1/1 | same | match |
| C1 | shared marked contexts | 27, 26, 25 | 28, 26, 26 | near match (method) |
| C2 | Sudoku repeat test | 27 of 28 vs 0 of 30 | same | match |
| C3 | identical output-vocabulary text for both poles | 300 of 300 per model | 300 of 300 in Q4B, G4, O7 | match |
| C4 | copy descriptions; median blind max vs others | 44; 0.128 vs 0.626 | 39 with my regex; 0.131 vs 0.624 | regex-dependent, direction holds |
| C5 | marks after a raw carriage return; texts with one; pos0 share | 25 of 30 in each; 0; 1.0 | same; none of the three descriptions mentions it | match |
| C6 | second-reader labels; raw agreement; explainer UNK, reader agreed, passed | as stated | same. Reader schema did allow UNREL | match |
| C8 | T_0 labels; confidence range and mean; T_15 mean | 18/20/2; 0.08 to 0.40, 0.171; 0.407 | same | match |
| C8 | long-document regex, T_0 vs T_15 | 28 vs 4 | 20 vs 3 with my regex | regex-dependent |
| C11 | bread poles; bread or volcano poles; bread texts | 757; 1,279 (71%); 2,572 | 757; 1,276 (71.3%); 2,579 | match within regex |
| C12, C numbers | X3 counts; essay wins with one positive at 0.30 or more | 41/358, 34/338, 35/311; 39/55/54 | same | match |
| C9 | last-layer ONE poles | 6/16, 1/16, 4/20 | same | match |
| D1 | calls, manual (judge, clusterer), API; judge rows by day; busiest hour; API-price estimate | 12,902; 12,585 (11,497, 1,088); 317; 9,166/1,868/463; 795; about $244 | same; 11,497 x 0.0162 + 1,088 x 0.053 = 243.9 | match |
| D2 | 375 protocol-breaking lines: old, fresh, other 10,659; answers changed | 81.6%, 82.1%, 80.9%; 24 (11 vs 13 correct) | same | match |
| D2 | chunk 028 answers before and after | ABBBBAAAABBABBA, 12 of 15 | same | match |
| D numbers | distinct ids; ids answered more than once | 12,312; 589 | same | match |
| D8 | failing clusterer answers; 6 or fewer whole lines; exactly 2; the judge line | 174; 165; 120; judge:w4j:991 | same | match |
| D10 | checker flags by severity | 5 high, 15 medium, 19 low | 2+0+3+0, 2+3+9+1, 8+6+2+3 | match |
| D7 | O7 explainer counts per tercile (Opus/Sonnet) | 208/12, 204/16, 14/186 | 208/0, 204/0, 14/174 on the 600 audit poles | report counts include the 40 poles left out of the audit |
| D5 | "the committed chk_full.py implements the tail rule" | tail rule | commit 66b2d558 (2026-10-24T07:39:35Z) replaced it with the every-line rule | stale |

## 2. Quote checks

"Yes" means found verbatim at that pole. "Yes, format" means the same words with a moved space, a curly apostrophe, or different quote marks.

| report | quote fragment | pole | found verbatim |
|---|---|---|---|
| A, B | "Fires on the word or word-piece that completes a well-known multi-part name or fixed compound term" | O7 L03N6372+ | yes |
| A | "Altura Mining is developing the Pilgangoora⟦ lithium⟧ project" | O7 L03N6372+ | yes |
| A | "SemiSolid⟦ lithium⟧-ion battery" | O7 L03N6372+ | yes |
| A | "opensource.org/licenses⟦/MIT⟧", "David⟦ Beckham⟧", "Palo⟦ Alto⟧" | O7 L03N6372+ | yes |
| B | "apps like ChatGPT, Hugging⟦Face⟧'s HuggingChat" | O7 L03N6372+ | yes, format (curly apostrophe) |
| A | "1. T⟦rello⟧" | O7 L04N6289+ | yes |
| A | "Fires on the name of a software app or online service, especially the second word or piece of a branded product name" | O7 L04N6289+ | yes |
| A | job name "(Viber, Discord, Messenger, Notion, T...)" | O7 L04N6289+:c2 | NO. The name ends "Notion, Trello)". The cut hides the word the story says is gone |
| A | "G⟦otta⟧ admit, it's a lot lighter"; "Get⟦tin⟧' tipsy" | O7 L03N7155+ | yes |
| A, B | "go.microsoft.com/fwlink⟦/?⟧LinkId=317598"; "wordpress.org⟦/?⟧v=2.8.4" and "v=4.6.6" | O7 L06N7759+ | yes |
| A, B | "Oh, ping-pong is an Olympic sport. H⟦aha⟧." | O7 L06N7759+ | yes |
| A | "basic⟦ chemical⟧ capacity will expand" | O7 L12N1003- | yes |
| A | "Beyond academic⟦ excellence⟧", "Engaging Learning⟦ Experience⟧", "Fast-Track Skill⟦ Acquisition⟧" | O7 L23N809- | yes |
| A | "San Gabriel⟦ Gul⟧ch", "Calouste⟦ Gul⟧benkian" | O7 L23N809- | yes |
| A | "Fires strongly on the word-piece 'Gul'" | O7 L23N809- | yes, format (double quotes in data) |
| A, B | "Fires at the last position before the model expects a next token starting with a capital O" | O7 L31N3362- | yes |
| A | T_14 record "holds Beginning, originates, 来源, Childhood, Starting" | O7 L01N4337- | yes, but the most common mark, " CZ" (5 of 20), is left out |
| B | `import React, { useState, ⟦useEffect⟧ } from "react"` | O7 L01N10284- | yes, format (data marks "⟦ useEffect⟧") |
| B | "the token "useEffect" inside a JavaScript React import line" and "chiefly "Exactly"" | O7 L01N10284- | yes |
| B | "the surname Harris or Haris"; "the 'useState' hook inside a React import line" | Q4B L07N4960+ | yes |
| B | "the type name Random immediately after the keyword new" | O7 L06N4452- | yes |
| B | "lock, latch, hinge, zipper, clip, clutch, gun slide"; "catch (Db⟦Update⟧ConcurrencyException)" | O7 L16N10459+ | yes |
| B | "In C# source files, the line break right after the opening brace of a namespace declaration" | G4 L22N9321- | yes |
| B | "sentence-ending periods inside incoherent, machine-generated spam prose (spun essay-writing / thesis-help SEO filler" | G4 L11N1220- | yes |
| B | "J. G. (⟦John⟧ Gadsby)", "I find myself⟦ second⟧ guessing", "Galligan, Richard", "g as in garden" | O7 L31N10276- | yes |
| B | "⟦Thanks⟧ for the interview George!"; "gnæt (gnat)" | O7 L31N10276- | yes, format ("⟦ Thanks⟧"; "gnæ t (gnat)") |
| B | "turned ⟦OFF⟧"; neighbour "Follow us on Twitter" | O7 L03N10451+, L03N10379+ | yes, format ("turned⟦ OFF⟧"); yes |
| C | "This pole never fires anywhere in the corpus (zero tokens with positive activation, max of 0) ..." | Q4B L01N7440+ | yes |
| C | "UWYGADFPVZBECKMTHXSLRINQO⟦J⟧"; "3 8 2 1 4 ⟦4⟧ 5 9 7"; "new int[] { 3, 0, 0, 4, 8, 1, ⟦1⟧, 7, 9} };" | Q4B L13N2197- and + | yes |
| C | "chipping a tooth in⟦ the⟧ process!" and "chipping a tooth⟦ in⟧ the process!" | Q4B L31N2783+ and - | yes |
| C | "In today's ever-evolving digital landscape, it is crucial to foster ..." (0.27) | Q4B L24N2529- | yes |
| C | "Hirschsprung disease affects the large intestine of newborns. Doctors diagnose Hirshsprung disease ..." | Q4B L16N1066- | yes |
| C | "Don't even reach for your wallet, tonight is my treat, I'm buying ..." | O7 L09N8024- | yes |
| C | "Mist⟦ral⟧ Large"; "We benchmarked GPT-4, Claude, Gemini and Mistral ..." | G4 L10N7669- | yes |
| C | "I think, that you are mistaken. Write to me⟦ in⟧ PM, we will communicate." | G4 L26N2364- | yes |
| C | "Mr Bamidele⟦ O⟧dugbesan"; "She scored the top of the boule with a razor blade ..." | G4 L01N2838+, L00N770- | yes |
| D | "DO NOT USE OPUS API HARD RULE"; "use my claude code tokens for sonnet and opus" | S/JOURNAL.md lines 450, 910 | yes |
| D | "The judge model, not the clusters, is the limit."; "wrong by a factor of three"; "the reject loop was paying for a prompt defect" | S/JOURNAL.md lines 821-822, 193, 737 | yes |
| D | "ran sed to rewrite the ids on lines 11-14 (4567->4459, ...)"; "pairing confirmed" | S/handoff/judge_incident_log.txt lines 9, 12 | yes |

No quote was invented. One was cut in a way that changes its meaning (A4). One record listing leaves out its top entry (A5).

## 3. Cross-cutting claims and the tests

### 3.1 "Unrelated share rises with birth time" (A1, B2)

**Size.** Later jobs are smaller (median 9.5 contexts at T_4-9, 6 at T_14-15). Weighting every bin to the same size mix gives p_u -0.001, 0.224, 0.270, 0.351, 0.438. The ramp survives. Inside each size band it also rises: 6 to 10 contexts go from -0.09 to 0.58; 11 or more go from 0.02 to 0.39.

**Judge difficulty, proxied by token overlap.** I parsed every `judge_saw`. The share of random-decoy judgments whose true record shares a marked token with the exemplars falls from 0.265 (T_4-9) to 0.085 (T_14-15). When there is overlap, accuracy is 0.90 to 0.97 in every bin, with no ramp. Without overlap the ramp stays: p_u 0.018, 0.284, 0.320, 0.454, 0.460. Inside quartiles of word-level Jaccard difference (true record minus decoy) the ramp holds in quartiles 1 to 3 and shrinks in quartile 4 (-0.10 to 0.10). Overlap explains part of the ramp, not all of it.

**No per-bin ceiling.** This is the serious gap. The controls do not match the test items. Control judgments share a marked token with their record in 50.0% of cases. Late-job judgments do so in 18.7%. The judge gets 0.975 right with overlap and 0.873 without. Matching the ceiling to each bin's overlap rate gives p_u -0.074, 0.158, 0.213, 0.321, 0.356. Pooled it gives 0.206 [0.132, 0.271] (2,000 pole draws, controls resampled in the same draw). Overlap is partly the outcome, so this is a sensitivity bound, not a corrected estimate. Two consequences still follow. The per-bin numbers ("4 in 10") are not identified. The pooled H_SP label also depends on the ceiling.

**Within one neuron (my added test).** This is the test that breaks the story. I took late-born jobs of the same pole born at different checkpoints: 1,029 pairs. The earlier job scored higher in 299, the later one in 309, and 421 tied. A pole-demeaned slope is -0.0096 accuracy per checkpoint [-0.027, +0.010]. The pooled slope is -0.029. Among the 46 poles with late jobs both at or before T_11 and at or after T_13, early minus late accuracy is -0.013 [-0.098, 0.070]. The ramp is a fact about which neurons have late-born jobs. Grouping poles by when their first job appears leaves a rise inside each group. So the ramp is not only about when a neuron starts. But no test shows a neuron that "grows family early and adds strangers late".

**Verdict.** The pooled ramp is real in the data. It is not a size effect. Its height depends on an unmatched ceiling, and inside a neuron it cannot be seen. "Phase change from growing families to packing in strangers" (A1) and "a model looks cleanest early in training" (B2) do not follow.

### 3.2 "Jobs settled by 40% of training" (A3)

- "Born" means the median exemplar first reaches the pole's 99.9th percentile and is still there at the next checkpoint (`neurons_verdict.py` lines 330 to 345). That bar is low. The median job is born at 0.295 of the pole's max. At 560k the median job sits at 0.404 of max, against 0.681 at the end. So 91% of final main jobs had crossed a low bar by 560k. They had not settled.
- The grid has no checkpoint between 560k (39.6%) and 1M (70.7%). A T_14 birth can fall anywhere in that window.
- At T_15 one present checkpoint is enough for birth (line 342). That bin has 36 births. The "190-fold" collapse rests on it. Peak over T_14 is 25-fold.
- Jobs are taken from the T_15 top 50, so "never born" is 0 by construction. A job that formed and died has no row. The records show such drop-outs: lithium, URL "?", "chemical", "Dark Knight". 44 jobs also drop out after birth. Neither total can be counted.
- "The shape you would expect from a learning-rate-limited process" (A3) is not tested.

**Verdict.** Safe if worded as "91% of the final main jobs had first entered the pole's top 0.1% by step 560k". "Settled" and "190-fold" need the caveats above.

### 3.3 "Shallow layers hold the most jobs" (B8)

- **Second clustering.** Sample 1 also gives more jobs in layers 0-7: 3.95 against 3.09 in layers 24-31.
- **Merge jobs that share a marked token.** The gap grows: 4.70 against 2.76.
- **Topic-splitting proxy.** Only 3.9% of job pairs in layers 0-7 share a marked token. In layers 24-31 it is 21.2%. Shallow poles also have fewer distinct marked tokens in their top 50 (24.5 against 29.4). The clusterer splits shallow poles by token, not by topic. B's reading that shallow tokens "share one token rule, so the clusterer splits them" is backwards about topic. It is right only in that the explainer joins different tokens under one rule.
- **Flag rate.** This half of the hook is an instrument effect. Control main jobs, judged against their own T_14 record, are flagged 5.9% of the time in layers 0-7 and 1.3% in layers 24-31. Control accuracy is 0.880 and 0.952. So 4.5x on controls is larger than the 2.8x B reports for late jobs. With each quartile's own ceiling, p_u is 0.307 and 0.172, and the intervals overlap. B says so in the text but not in the hook.

**Verdict.** The higher job count in shallow layers holds across both clustering samples. "Nearly three times the flag rate" should not be published.

### 3.4 Migration stories against "the top-20 window moved while the job persisted"

- **Lithium (A2, B1).** Lithium is 19 of 20 at T_12 and 1 of 20 at T_13. It comes back to 3 of 20 at T_14 and is 0 of 50 at T_15. The name-completion jobs were already above the 99.9th percentile during the lithium era. Palo Alto was present at T_9. The misc proper-noun job was present from T_10 on. UC Santa Cruz was present from T_5 to T_8. All 15 final jobs peak in a narrow band, 0.16 to 0.25. One reading fits as well as "replaced": lithium sat on top of a name-completion neuron and lost the top 20 to it. Lithium was still in the top 20 at 70.7% of training.
- **Trello (A4).** Fails. Four app-name jobs (c0, c1, c2, c5) were born at T_8, the same checkpoint where the record is "rello" 20 of 20. Trello is still in the T_15 top 50, and job c2's name lists it. Nothing shows a seed that came first, generalized, then left.
- **React (B3).** The adverb job was present from T_4 to T_15. At T_15 it holds 24 of the top 50 contexts against React's 12. The T_12 record still has "Exactly" 7 and "Truly" 3. The neuron added React. It did not switch to it.
- **Haha (B4, A7).** A real change, but later than B says. The pole's own record, shown as a decoy at T_11 (180k), has "?" 6 and "/?" 4 against "aha" 8. URL marks are gone from the T_13 and T_14 records and from the T_15 top 50. The switch falls between 180k and 560k. The laugh job was born at T_9 at 0.19 of max.
- **The parent side.** Before birth a new job's median exemplar is below the 99.9th percentile by rule. A minor parent below the top 20 is still invisible. A9 shows this with the Gul pole: the gulp job was present from T_5 but is missing from the T_8 record.
- **H_MIG depends on the main rule.** Main is the job with the highest final peak. If main is the job with the most contexts, 19 poles qualify and only 8 overlap the study's 17. In 9 of the 17, the earliest job holds more of the final top 50 than the "main" does. Examples: L05N2634+ 47 vs 3, L17N5407+ 43 vs 7, L14N4543+ 25 vs 8. Birth order is robust: b = 0.479 with main by size, against a null 97.5th percentile of 0.448.

### 3.5 "The explainer gives 1 job to 532 poles, the clusterer to 26" (B6)

The prompts define a job differently. The clusterer prompt says "Related offshoots of one job may form their own cluster; mark them related to it" and "Every context belongs to exactly one cluster". The explainer prompt says offshoots "are one job grown outward" and defines "ONE (one job, possibly with small related offshoots)". To compare like with like, join the clusterer's clusters through its own "related" links. 232 poles then have one job family, and 169 have no unrelated pair at all. The mean is 2.07 families, not 4.09 clusters. The fair contrast is 532 against 169 to 232. That is still a large disagreement, but not 20-fold. The self-disagreement numbers (182 of 600 exact, 67.8% within one) stand.

### 3.6 Extra test: "two clocks" (A8)

The 60-fold gap compares median first-crossing times. The center curve's first crossing is pulled toward random init: 111 poles cross at T_0 and 120 at T_1 (M `center_birth_curve.blind_max.first_t_counts`). Only 1% of center curves are non-decreasing. UNK poles cross earliest, at T_1, which points to noise. Comparing the median curves instead, the center crosses 0.3 at T_6 (12k) and jobs cross at T_12 (320k). The two curves also measure different things: the max over one probe sentence against the median over 50 exemplars. A says so, but the hook does not.

## 4. Tiers (46 stories)

### SAFE (19)
- A6 main job uniquely oldest in 22.0% vs 17.5% (p = 0.0015; robust to the main rule).
- A9 the Gul sibling the judge missed.
- A11 extra jobs are not leftovers, with its own selection caveat in the hook.
- B9 the G neuron false alarm (13 of 20 record contexts precede a G word; judge 1 of 4).
- B10 hard-decoy near miss, 958 of 1,200.
- C1 near-silent Qwen layer-1 neurons pass as ONE.
- C6 the second reader never said UNREL, though its schema allowed it.
- C8 random-init neurons get job titles at low confidence.
- C11 sourdough and lava.
- C12 small human moments; 41/358, 34/338, 35/311 beat the corpus max.
- D1 budget walls and the move to Claude Code agents.
- D2 annotator agents cut corners; labels barely moved.
- D3 judge failed the gate; fixed by a model change.
- D4 seven unblinds, p_u 0.260 to 0.265.
- D6 the pre-registration bet against its own result.
- D8 forty excerpt-read chunks, not one.
- D10 the checkers' 39 flags.
- D11 GPU costs (I checked only $25.60 and $30.27 in SPEC §12).
- D12 companion paper and the late novelty check (I checked only 2601.08383 in `checks/literature_W5.md`).

### SAFE WITH CAVEAT (19)
- A2 lithium: lithium was still 3 of 20 at step 1M, and the name-completion jobs were already above the 99.9th percentile during the lithium era. So "replaced" means "lost the top 20".
- A3 structure by 40%: "born" means first entry into the top 0.1% at 0.29 of max, not settled. The 188x rests on a final bin with a one-checkpoint rule (25x against T_14).
- A5 family trees: the judge also gives 4/4 with no visible parent (gulch against a gul-free record, Terminator against a "�"/Euler record), so 4/4 means "closer than a random decoy".
- A7 jobs that died: "died" means "left the top 20 or top 50". The turnover numbers come from records that exist only for poles that added a job.
- A10 late bloomers: zero medians are measured on exemplars chosen at T_15, so early zeros are partly expected by selection.
- B1 most cluttered neuron: the 15 jobs come from one clustering (the second finds 3), 12 of them have 2 to 5 contexts, and lithium was still 3 of 20 at T_14.
- B3 React: the adverb job never left (24 of 50 final contexts against React's 12). The neuron added React.
- B5 odd couples: "14 of 19 pair code with prose" is a reading of job names, and the UNREL p_u of 0.795 rests on 13 jobs.
- B11 main is a side gig: "56 within 2%" does not reproduce. Birth order is robust to the main rule. H_MIG is not.
- C2 Sudoku: a repeat heuristic on 30 contexts, not probed.
- C3 "in the process": the sign argument is direct-path arithmetic. The identical vocabulary text (900 of 900 neurons) is measured.
- C4 copy neurons: regex-defined. The effect holds inside UNK poles too (median blind max 0.109 against 0.313).
- C5 carriage return: the raw \r is in the prompt string. Say "not rendered", not "invisible".
- C7 machine-written prose: these are 8 explainer descriptions, 6 of which failed the test. Not verified detectors.
- C9 next-word language with depth: depth is confounded with explainer model, as C says.
- C10 different obsessions: regex counts, except "forward-looking" 41/8/8, which reproduces exactly.
- D5 Read truncation: drop the "loose end". Commit 66b2d558 at 2026-10-24T07:39:35Z made the code match the record.
- D7 budget-order confound: the O7 tercile counts quoted (208/12, 204/16, 14/186) include the 40 poles left out of the audit. On the audit set they are 208/0, 204/0, 14/174.
- D9 observer effect: "a factor of five" and 22.7% to 45.4% describe the replaced Flash instrument. On the final instrument, ONE labels exceed confirmed ONE by 1.5x to 1.7x (525/358, 520/338, 521/311).

### DO NOT PUBLISH as written (8)
- **A1** "Early jobs are family; late jobs are strangers". Inside one neuron no ramp shows (299 vs 309 pairs; slope CI [-0.027, +0.010]). The per-bin values also depend on a ceiling that does not match the test items. Publishable form: "late-born jobs, mostly on neurons whose jobs all form late, are judged unrelated more often".
- **A4** Trello. Trello is in the final top 50 and in a final job's name, and the report cut that name to "T...". The four app-name jobs were born at the same checkpoint as the Trello-only record.
- **A8** two clocks. The 60x comes from first-crossing statistics pulled to random init. On median curves the gap is 12k against 320k, and the two curves measure different things.
- **B2** "The mess arrives in the last stretch". Same problem as A1. "A model looks cleanest early in training" cannot come from a design that sees only the jobs that survive to T_15.
- **B4** "By step 60k it was the laughing neuron". At 180k, URL marks were still 10 of 20 of its top contexts. Use A7's dates instead.
- **B6** "532 vs 26". The two prompts define a job differently. The comparable figure is 532 against 169 to 232.
- **B7** "Seventeen neurons that changed careers". In 9 of the 17 the earliest job is still larger than the "main" at the end. Under a size-based main rule only 8 of the 17 remain. The count is fine as "main job late-born and flagged". "Changed careers" is not.
- **B8** "Nearly three times the flag rate". The judge's own control flag rate differs 4.5x between the same layer bands. The job-count half is publishable.

## 5. What a lab researcher would find interesting

1. **C1, near-silent neurons pass as clean (Qwen3-4B).** Five of eight sampled layer-1 neurons peak at 0.003 or less and sit above 1% of that peak on nearly every token. They still pass as ONE, because every score is relative to the pole's own max. This is a concrete, reproducible flaw that applies to any normalized-activation explainer pipeline.
2. **C3, the output vocabulary is sign-blind.** In all 900 sampled neurons both poles got identical "most promoted / most suppressed" text. Any pipeline that gives a negative pole a logit-lens summary invites a backwards "predicts X" story. This is cheap to fix and general.
3. **D2, auditing LLM-agent annotators.** 375 lines from protocol-breaking agents were re-answered by fresh agents. 24 flipped (6.4%), and accuracy was 81.6% before and 82.1% after, against 80.9% elsewhere. That is a real test-retest rate for an agent judge, and it shows a process audit and an outcome audit are different things.

## 6. The one overreach to quote back

A1: "One pooled p_u hides a phase change from growing families to packing in strangers, which is exactly what H_SP predicted."

Inside the same neuron, a later-born job is judged unrelated no more often than an earlier one: 299 pairs one way, 309 the other, 421 tied. The ceiling the per-bin numbers are scaled against was measured on controls that share a marked token with their record 2.7 times as often as the test items. Matched to the test items, the pooled estimate falls to 0.206 [0.132, 0.271]. The data show which neurons look cluttered. They do not show a phase of training.
