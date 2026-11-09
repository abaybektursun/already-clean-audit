# Audit of the public plan's chronology and decision chain

Written 2026-11-08 by an Opus auditor at the head's request. Object: `studies/already-clean/walkthrough/PLAN_PUBLIC_2026-11-08.md`
at clean-neurons 5cd349f61, its §1 and its §3 table (eleven chapters). This file reports; it does not rewrite the plan.

**Answer.** The plan's November dates and most of its November numbers match the record, but three of its decisions state the
opposite of the record, and its chain of "this result opened that question" is mostly not the record's chain. (1) Chapter 1
says the October study found that "later groups mostly grow from earlier ones", and chapter 2 that "the tools pass their checks;
the growth is real"; the record's pre-registered label is packing (p_u 0.265 [0.211, 0.321], above the 0.20 line), the birth-order
reading that pointed to growth was retagged on 5 November, the known-edit check had every rule cancelled, and the 4 November position
is "a line crossing, not an event". (2) Chapter 6 says the half-precision gradient *prediction* was noise; the record says the
half-precision *measurement* of the switch-off was the noisy side and the prediction was right (Spearman 0.995 against the fp32
prediction, 0.960 against the fp32 measurement). (3) Chapter 3 gives a choice order and a choice criterion that no planning
file contains: the record chose N13 first and M1 second, by a five-axis leverage score and a "before/after" rule. Five of the
ten "question it opened" cells name a link the record does not contain, and two more are partly right. Counts are in §6.

**Conventions.** `J:NN` is `studies/next/JOURNAL.md` line NN, followed by its CST stamp. Other paths are relative to the
repository root unless they start with `next/` (= `studies/next/`) or `ac/` (= `studies/already-clean/`). "Plan line NN" is a
line of the plan file. Every quotation is copied from the file named.

**Files read.** The plan in full. `next/JOURNAL.md` in full (lines 1–159). The study records under
`next/{m1,m2,n12,n2v,n1,n13,n6,n22,pilot0}`: M2, N6 (JOURNAL and STATE), N12 and N22 in full by a second Opus reader under this
auditor's brief, with its line references spot-checked; M1, N1 and N2V journals in full by this auditor; N1 and N2V STATE.md
searched for decisions and open items; N13 JOURNAL.md lines 1–120 and STATE.md lines 1–80 (the rest repeats the head's journal
entry by entry); pilot0's README and GPU_CHECK2.md verdict lines. `next/08_SUMMARY.md`, `next/09_PLAN.md`, `next/05_SCORING.md`,
`next/10_RETAG.md`, `next/20_ABSTRACT.md`, `next/21_REFLECTION.md`, `next/HANDOFF.md`, `next/HANDOFF_2026-11-07.md`, the head reads
and verdict files of every study. For October: `git show gh-pages:index.html` from "What happened next, 24 October to 4 November
2026" to the end, tags stripped, and `ac/JOURNAL.md`, `ac/README.md`, `ac/SPEC_D1.md`, `ac/results/`, `ac/archive/` (by a second
Opus reader under this auditor's brief, spot-checked). Because chapter 2's content comes from other studies, that reader also
read `studies/{reuse,unite,instruments,neurons}/` result files where the public page cites them.

## 0. The structural finding first

The plan's §1 (plan line 22–23) says the reader "should finish knowing ... why each experiment followed the one before it", and
§3 gives each chapter a "question it opened". The record does not support that shape for most links.

- **What was chosen, when.** At 01:35 CST on 5 November the picks were fixed as "1. N13, what RL wrote ... first", "2. M1, the
  three stage-2 mixes", "3. M2, name the carrier before the birth", "4. N22, the model gets a variable", "5. N6, edit the young",
  with "Program A's mechanism cards (N1, N2, N3)" after them (`next/05_SCORING.md` lines 111–135; `next/08_SUMMARY.md` lines 7,
  13, 18–19; `next/09_PLAN.md` lines 18–24 and the job list lines 26–60). All of these were chosen before any of them ran.
- **Links the record does contain.** The critic's attack on the birth dates led to the recount, the retag and the restriction
  to the births that beat chance (J:15–16, J:26–29; `next/05_SCORING.md` line 153). M2's result moved N12 up: "next under the
  decision table: N12 (competition) moves up in J12" (J:77, 2026-11-06 02:20 CST), and M2's reader set N12's distinct-token
  twin (J:78). N12's failed validation led to N2V: "the next instrument is the ablation's loss change in fp32 ... registered as
  the J12 follow-up before any gradient readout counts" (J:87, 08:10 CST); N12's own decision named both of its successors, "the
  weight change behind the carrier's rise is the next object (N1 ...), and the gradient readout needs the small-scale check before
  any further use" (`next/n12/READ_head.md` lines 78–79). N13's result led to the rank test (J:146) and to the withdrawal of the
  practical card (`next/HANDOFF.md` line 75). N6's inverted control left a design rule (`next/21_REFLECTION.md` line 66).
- **"Question it opened" cells the record does not contain** (ten cells; chapter 3 has none). Chapter 1: the question about the
  moment of birth was asked and answered inside chapter 2's period (§3, C-O4). Chapter 2: the "six questions" were not what
  chapter 3 chose from (C-O11). Chapter 4: M2 was chosen before M1 ran (C8). Chapter 7: N1 and N2V were started together as
  "J12's next two cards" (J:99, 2026-11-06 12:20 CST); N1's verdict, 2026-11-07 19:35Z, came before N2V's, 2026-11-08 00:53Z, so
  N1 could not have opened "now that the instrument works" (C18). Chapter 8: N13 was the first pick and had run since 5 November
  (C26). Partly right: chapter 5 (the record's next step was competition, C13) and chapter 11 (the second question has no source,
  C39). Recorded: chapters 6, 9 and 10. Also not causal, though no cell claims it: N6 was the fifth pick and its main run launched
  on 7 November "rather than holding the slot", before N13's verdict (J:119).

A chronological article can still be honest if it says that the program ran two tracks from one choice, and marks which
transitions were causal. The items below are written against the chapters as they stand.

## 1. Experiments missing from the chapters

Each line: date (CST unless marked Z); what it was; source; the chapter it belongs in. The October items are in §4.

### Chapter 3 (5 November: the choice)

1. 01:16 — Checkpoint content-hash check: the 16 stage-1 revisions have 16 distinct safetensors sha256 sets, so no
   duplicate-label defect (J:13). Belongs in chapter 3 or 1 as the check that the sixteen snapshots are distinct.
2. 01:16 to 16:05 — The data-order pilot. Offline reconstruction of two stage-1 batches, fingerprint equal to the trainer's
   (J:13; `next/pilot0/README.md` lines 1–8). GPU loss check: "Frozen rule on the registered quantity (every position, pads
   included): NOT CONFIRMED (+0.164 nats, 6.6 sd)" (J:21, 02:45). Second check (J8) with a pad-excluded rule frozen first: both
   steps in the pass branch, "the second data-order check is closed" (J:34, J:47, J:48, 16:05). Chapter 3. It was the asset behind
   the "documents that made the job" card, which the critic set aside the same night (item 3).
3. 01:35 — The critic's attack (`next/06_CRITIC.md`): "random regroupings of each pole's top-50 positions into groups of the real
   sizes reproduce the D1 birth histogram (TV 0.040) ... D1's b 0.476 is below the random-group reference 0.530" (J:15). The same
   entry sets aside "N4 at 7B, N28, N27-as-designed" and adds M1, M2, M3. Chapter 3; chapter 1 must point forward to it.
4. 01:50 — Independent verification of the critic (`next/06b_VERIFY_BIRTHS.md`): b below all 300 random draws (p 0.0067); "150 of
   1,238 late-born jobs beat 95% of random same-size groups against a selection-matched chance level of 28"; histogram TV 0.0406
   against a noise floor of at most 0.028 (J:16). Chapter 3.
5. 09:30 — Spec-only recount (`next/10_RECOUNT2.md`): 2,454 of 2,454 clusters match; b 0.476 [0.436, 0.516] against 0.530
   [0.495, 0.568]; "153 of 2,428 clusters born later than chance against 22.4 expected"; cause found in ties for the earliest date
   (J:26). Chapter 3. These 153 births are the unit of chapters 5, 6, 7 and 8 (J:60), and no chapter says so.
6. 09:58–10:12 — The retag of "the main job is the oldest" to "RETAGGED 2026-11-05, not supported", F4 and the walkthrough's
   births picture redrawn with the chance level, row D1-b added to `studies/neurons/CLAIMS.md` (J:28, J:29; `next/10_RETAG.md`
   lines 3–8). Chapter 3, with a forward note in chapter 1. The plan's chapter 1 still uses the births picture as if unchanged.
7. 01:56 — M1 asset check: the mixes are endpoints of 100B tokens from one stage-1 start; "the pre-stated ordering 0/40/65 as
   the test with room to fail"; a gate that fails if the loaded weights equal `main` (J:17). Chapter 4 (setup).
8. 02:13–02:50 — N13 pre-registration and the question of which MATH set the published 40.0 is; settled as Minerva MATH; the
   registered pair moved to `Olmo-3.1-7B-RL-Zero-Math` (J:18, J:19, J:22). Chapter 9.

### Chapter 4 (M1)

9. 2026-11-05 16:31Z — K0: "1,535 of 1,535 exemplar activations of T_15 bit-identical to the bank"; corpus NLL equal
   (`next/m1/JOURNAL.md` line 8; J:33). Chapter 4.
10. 2026-11-05 16:44Z — Freeze 2 on stage-1 references: the drift bar 0.0082, 0.0066, 0.0023, 0.0035 is the pass bar; both branches fire on
    reference pairs (`next/m1/JOURNAL.md` line 11; J:35–36). Chapter 4.
11. 2026-11-05 19:11Z — Identity pair: start harvested twice, `counts.npz` byte-identical (`next/m1/JOURNAL.md` line 17; J:39). Chapter 4.
12. 2026-11-05 17:40 — Independent read (131 of 131 fields) and the head's recount (J:51; `next/m1/JOURNAL.md` lines 23–24). Chapter 4.
13. 2026-11-05 17:45–18:25 — The Goodfire preprint read in full (`next/12_SHAPING_NOTE.md`) and the two zero-GPU checks it prompted: pole
    turnover ("a loss at fixed pole identity, not movement between poles") and a reading control ("the mix changes which of a
    feature's token families dominates its strongest instances") (J:52–54; `next/m1/READ_head.md` lines 74–94). Chapter 4.
14. 2026-11-08 05:54–05:58 — The abstract check corrected M1's size from "about one slot in fifty" to 0.6–0.9 of a slot per pole,
    and "the 65 % mix held them" to "at most 0.36 of a slot" (J:149, J:150). Chapter 4 must use the corrected numbers.

### Chapter 5 (M2)

15. 2026-11-05 19:40 — Pre-check: "153 births pass J1's recount ... on 140 carrier poles"; split by neuron parity; the pseudo-birth twins
    as the structured control; the head's prior "H_B SUPPORTED, H_sel_B AT_SELECTION more likely than ABOVE" (J:60). Chapter 5.
16. The five attempts that did not count, each with its own cause: score-01 stopped at the cross-device G-EX gate at T_14
    (J:61); score-02 at G-EX at T_13 (J:66); score-03 at G-EX at T_13 again (J:71); score-04 crashed in the scoring code on an
    assertion about poles with no own-sign token (J:72); score-05 stopped because G0's failing input had no power at T_6 (J:74).
    score-06 gave the verdict (J:76). Each was followed by an amendment (A1–A5). Chapter 5. M2 study-record lines are in §1b.
17. 2026-11-06 02:00 — Raw examples: "other poles of the layer already hold the job's exemplar positions one checkpoint before
    (NPR 'Copyright': L10N3510+ holds 5 of 5; SQL 'CHARSET': two poles hold 4 and 3, the carrier none)" (J:76); the head's read:
    "the gap behind the twins is in the competitors" (J:77). Chapter 5. It is why N12 was moved up.
18. 2026-11-06 02:40 — The independent reader's post hoc finding: "the twin gap sits on repeated-token births ... on varied-token births
    carriers and twins hit alike (57 against 58 of 71), on repeated-token births the carrier loses (22 against 53 of 78)" (J:78).
    Chapter 5, marked post hoc.

### Chapter 6 (N12)

19. 2026-11-06 03:50 — N12 pre-registration: the realized competition (the carrier's own move against the others' move, F),
    the leaders' drop against M2's twins, M2's follow-up twin matched on distinct tokens, and the gradient signal; the head raised
    G-VAL's bar from 0.5 to 0.7 (J:79). Chapter 6.
20. 2026-11-06 04:45 — score-01 does not count: "the batch-1 backward on the L40S is not bit-deterministic ... 4 of 98 units' u_S flipped";
    A1, a deterministic backward (J:80). Chapter 6.
21. 2026-11-06 07:35 — score-02's realized results, all counted: H_cf OWN_RISE (F 0.951 [0.738, 1.180]), H_away_sel_pb BELOW_SELECTION,
    H_dt SUPPORTED (real 42 of 71, distinct-token twins 41 of 67; real − dt −0.030 [−0.149, +0.090] AT_SELECTION) (J:84). Chapter 6.
    The plan moves the distinct-token result to chapter 5 and "own rise" to chapter 8.
22. 2026-11-06 08:10 — The independent read of N12: "OWN_RISE's sign is close to forced by the dating rule"; "G-VAL measured the resolution
    of a bf16 window-summed ablation, not the gradient's validity ... the earlier 'first-order breakdown' reading is withdrawn"
    (J:87). Chapter 6.
23. 2026-11-06 07:35 and 10:00 — The memory squeeze: score-02 needed "bf16 reductions in the math attention kernel to fit the L40S's memory" (J:84); the
    user then ruled "in cases like these they can use vast.ai" (J:92, 10:00). Chapter 6, because it shaped how N2V was built (item 27).

### Chapter 7 (N1)

24. 2026-11-06 13:25 — The onset pre-check at zero GPU: "births are gradual rises (93 of 153 hold less of the rise in the dated
    bracket than in the two before ...)", so the readout became the excess change over brackets k−2 to k (J:101;
    `next/n1/JOURNAL.md` lines 6–10). Chapter 7.
25. 2026-11-06 19:28Z to 2026-11-07 19:31Z — weights-01 crashed writing its examples (`KeyError: 'examples'`); weights-02 was
    killed by the executor, ruled a deviation; weights-03 completed, 49 gates pass (`next/n1/JOURNAL.md` lines 13–46; J:103, J:107,
    J:120). Chapter 7.
26. 2026-11-07 15:00 — The independent read's confound checks: "carriers' max |activation| is 1.39 times the twins' (adjusting
    widens to AT, −0.20 [−0.59, 0.21] ...)"; "the freeze half read AT"; "pooled 140 pairs −0.320 [−0.604, −0.035]" (J:127).
    Chapter 7.

### Chapter 8 (N2V)

27. 2026-11-06 13:25 — N2V pre-registration: "the fp32 reverse-mode backward does not fit the L40S (67 GB ...), so the spec
    registers forward-mode differentiation in fp32 ... which keeps one GPU model and needs no Vast image"; the head chose "core plus
    the 10 % finite difference" (J:101). On 7 November the head added the forward-mode part "so that readout runs" (J:107;
    `next/n2v/JOURNAL.md` lines 6–20). Chapter 6 or 8.
28. 2026-11-07 16:44Z — score-01 does not count at G-CAUSAL: "535 of 963 zero-ablation rows not bitwise equal to their baseline
    before the ablated position" (J:112); cause traced to the 16-byte alignment of fp32 logit rows, A1, score-02 (J:117;
    `next/n2v/JOURNAL.md` lines 23–46). Chapter 8.
29. 2026-11-08 00:53Z — What G-VAL compared: "N12's banked bf16 prediction against the fp32 zero ablation"; "The bf16 gradient
    against the fp32 forward-mode prediction: Spearman 0.995" (`next/n2v/JOURNAL.md` lines 107–115; J:134). Chapter 8; it is the fact
    that corrects chapter 6 (§3, item C15).
30. 2026-11-07 19:50 — The independent read of N2V (within-unit top pole 65 of 71; zero ablation first order at the median; "where the bf16
    and fp32 predictions disagree most the measurement sides with fp32 in 26 of 28") and the dated retag of N12's head-read
    sentence (J:136). Chapter 8.

### Chapter 9 (N13)

31. 2026-11-05 10:15 — The N13 offline gate: "the spec's grading rule ... judges 108 of 500 MATH-500 golds unequal to themselves"; the code's
    corrected rule became amendment A1 (J:30, J:37). Chapter 9.
32. 2026-11-05 — The known number took seven launches: attempts 1–5 failed in environment setup (J:35–37); attempt 6 ran at batch
    size 1 and was killed by the head (J:44); the projection gate A19 was built (J:46, J:49); attempt 7 passed, "39.96 against the
    published 40.0 ± 1.6" (J:55). Chapter 9.
33. 2026-11-05 to 06 — The smoke (attempts 8 and 9) measured a sustained rate of 236.6 tokens/s and long RL answers, and the
    projection said "no arm runs under the 35 ceiling on the L40S" (J:57, J:59, J:62, J:65, J:67). Chapter 9; it is the reason for
    the user's decision in §2, item D15.
34. 2026-11-05 22:08 — A26 changed answer extraction: "base 153 → 177 of 300 right (0.510 → 0.590)" on the smoke (J:69). Chapter 9; the
    base's 57.2 is measured under this rule.
35. 2026-11-05 23:45 — s3gate-01 failed in R1's stage 6 (A29); R1's gates showed "the RL model started from base `main`"; the
    per-block change profile was flat, "so the block-window transplant arms ... are retired" (J:73). R1 rerun bit-identical (J:81).
    Chapter 9.
36. 2026-11-06 05:00 — "the four rndE arms realize only 0.70–0.82 of their intended MLP change energy after bf16 rounding ... so
    as built they are not energy-matched"; A30, stochastic rounding (J:81, J:82). Chapter 9: the "random neurons of equal energy"
    were equal only after this fix.
37. 2026-11-06 12:15 — The S3 gate: base 57.2, rl 92.0, zero 55.4, id 91.2; G 34.8 against the bound 13.60 (J:98). Chapter 9.
38. 2026-11-07 09:20 — The outage: "the model's weekly limit on the account cut both running executors ... no agent ran for about
    20 hours"; two S4 launches failed on A30's exact float equality, "the second launch of the pair repeated it because nobody was
    watching"; A31 (J:103–106). Chapter 9 (it also delayed chapters 7 and 8).
39. 2026-11-08 05:00 — GSM8K held out at 1 %: "top − dare −23.7 [−26.5, −21.1] ... REPLICATES on 3 of 3"; the format count (NONE
    arms write an "Answer:" line in 0.41–0.46 of answers, REC arms in 0.00–0.04) (J:145). Chapter 9.
40. 2026-11-08 05:31 — The independent read of N13: "DARE's realized energy is 33 to 1,000 times the whole change's"; "format does not make the
    gap"; "the two GSM8K labels top − rndE and top − rndT are fragile"; "the mlp − attn +3.4 lies entirely in cap-hit problems"
    (J:146). Chapter 9.

### Chapter 10 (N6)

41. 2026-11-06 09:10 — N6 pre-registration: 74 matched pairs, age against ln max|a| ρ +0.027 after matching against −0.592
    before; E1's zero replacement restricted to one pole; losses in fp32 (J:89). Chapter 10.
42. 2026-11-06 09:40 — L1-01 does not count at G-POLE (0.015 against 0.01; two batch shapes round differently); A1 sets 0.05 (J:91).
    Chapter 10.
43. 2026-11-06 10:30 — L1-02: stage-2 pairing gives 45 balanced pairs; G-ZN and G-CACHE fail on two-path rounding; A2–A4 (J:93). Chapter 10.
44. 2026-11-06 11:00–11:55 — The power preview: at 96 windows "the window-bootstrap sd of the pair contrast ... exceeds the contrasts
    themselves"; the eval set raised to 1,024 windows; a power gate added; the positive control trimmed to 8 layers; L2 reprojected
    to 35.2–44.0 GPU-h, "7.9 times the old figure"; split into L2a and L2b (J:94–96). Chapter 10.
45. 2026-11-08 09:15 — The exploratory freeze-half readout "p_col 0.762 [0.549, 0.894]" (16 of 21) exists and is not a claim
    (J:153). Chapter 10 should say it exists and why it is not quoted.

### A study with no chapter: J10 (N22, the Jacobian-lens arithmetic study)

46. 2026-11-06 09:20 — pre-registered (J:90); 12:05 — "J10 (N22 with N25) ends at T_15 on its registered go/no-go: NO-GO ... OLMo-3
    7B at T_15 answers 7 of 1,200 items ... against the 100 the rule requires", cost $1.40 (J:97). It was the fourth pick
    (`next/05_SCORING.md` line 130). Belongs between chapters 6 and 7 (its date) or in chapter 11 (`next/21_REFLECTION.md` line 14
    counts it as one of "two pilots decided").

### Chapter 11 (what we know)

47. 2026-11-08 05:54–09:30 — The abstract checks found 4 mismatches in draft 3, 3 in draft 4, 1 in a head read during draft 5, 0
    in draft 6; for N6 "the independent recount covers G-PC and the R ratios but not τ_s itself"; "N6 has no READ files"
    (J:149–151, J:155). Chapter 11.
48. 2026-11-08 13:25 — The reflection, with "Where the head misjudged" (N6's control inside the main run; N12 in bf16; a practical card
    proposed before its deciding measurement) (`next/21_REFLECTION.md` lines 72–85; J:156). Chapter 11.

### 1b. Study-record lines for the attempts above

The head's journal cites each attempt; the study records give the exact stop. M2: score-01 stop on `t14_G_EX`, p99 0.2563
against 0.0444 (`next/m2/JOURNAL.md` line 14); A1 and its revision before score-02 (lines 17–30); score-02 stop on `t13_G_EX`
(lines 38–39); A2 (line 41); score-03 stop (line 54); A3, "if any gate stops score-04, M2 stops for good" (line 57); score-04
"crashed in the scoring code at T_10, not on a gate" (line 72); A4 (line 76); score-05 stop on `t06_G0`, "its failing input
passed", and "M2 ends without a verdict" (lines 90–92); A5 (line 93); score-06 complete, 52 gates (line 100); verdict (line 110).
N12: G-VAL bar raised (`next/n12/JOURNAL.md` lines 6–10); score-01 stop on I-GRAD (lines 51–58); A1 (lines 61–64); the executor's
bf16-reduction choice, "The fp32 upcast would very likely stop the job out of memory at T_14" (lines 71–77); freeze 2 with G-VAL
failing, "every gradient label ... will read DOES NOT COUNT at the verdict and is reported as exploratory" (lines 158–166); verdict
(lines 180–196). N6: L1-01 stop on G-POLE (`next/n6/JOURNAL.md` lines 26–34); A1 (lines 37–41); L1-02 (lines 62–76); A2–A4
(lines 93–125); A5 (lines 153–158); L2a complete with G-PC failed (lines 211–231); G-POWER failed at freeze 2 (lines 239–248);
close (lines 264–273). N22: l1-01 launched on try 16 (`next/n22/JOURNAL.md` line 37); "7 of 1,200" (lines 57–62); NO-GO, "L2 and
L3 are not launched" (lines 97–102); verdict (lines 105–111).

## 2. Decisions missing or misdated

Each line: date; the decision; the record's own reason (quoted); whether the plan states it, with the right reason, in the
right chapter.

| # | date (CST) | decision | the record's reason, quoted | in the plan? |
|---|---|---|---|---|
| D1 | 11-05 01:05 | Provisional picks: Program A (biography of a neuron) first | "It uses every asset only we have" (`next/05_SCORING.md` lines 83–84; J:11) | No. Needed because the critic reversed it 30 minutes later. |
| D2 | 11-05 01:35 | Picks v2: N13 first, M1 second, M2, N22, N6, then N1–N3; N4 at 7B, N28, N27 as designed not run | Rubric change: "Only a card with a before/after, a prediction made before the measurement, or an engineer's number can be a pick" (`next/05_SCORING.md` lines 97–99); N13 is "The only pick with a benchmark before/after and matched random controls that does not rest on the birth dates" (lines 113–114) | Stated wrongly: chapter 3 gives another order and another criterion (§3, C1–C2). |
| D3 | 11-05 01:50 | The unit of the birth studies shrinks to the births that beat random groups | "Program A's unit is those jobs" (`next/05_SCORING.md` line 153); M2 used "153 births" (J:60) | No. |
| D4 | 11-05 09:10 | User: retag approved; N13 and M1 approved; the second data-order check runs; the robot gate deferred | "go ahead. We want quality"; "the robot gate is deferred until the N13 and M1 headline results are in" (J:24) | No. |
| D5 | 11-05 09:58–10:12 | "The main job is the oldest" retagged, not supported | b 0.476 "below the 0.530 [0.495, 0.568] that random same-size groups give" (`next/10_RETAG.md` lines 5–8; J:28–29) | No. Belongs in chapter 3, with a forward note in chapter 1. |
| D6 | 11-05 10:15 | N13's grading rule replaced before freeze | the spec's rule "judges 108 of 500 MATH-500 golds unequal to themselves" (J:30) | No (chapter 9). |
| D7 | 11-05 15:05 | Attempt 6 of the known number killed; the pre-check must act on what it measures | "the path check found the defect before the big spend, but the job did not act on what it measured. The pre-check has to decide." (J:44) | No (chapter 9). |
| D8 | 11-05 16:05 | The data-order check closed, both steps pass | "the second data-order check is closed with both steps in the pass branch" (J:48) | No (chapter 3). |
| D9 | 11-05 17:40 | M1 closed; the per-pole analysis not run | "Under spec §1.2 the next step for this outcome is to report, not to run the per-pole analysis" (`next/m1/READ_head.md` lines 68–72) | No. The plan's chapter 4 states a different decision (§3, C7). |
| D10 | 11-05 18:05 | After the Goodfire preprint: two M1 checks now; a per-block profile added to N13; block-window arms decided after R1; an SFT→DPO harvest "a new question, not scheduled" | "their relocation finding is a stated threat to any single-basis, single-layer neuron measure of ours" (J:52); decisions listed in J:53 | No (chapters 4 and 9). |
| D11 | 11-05 19:15 | User: move a job to Vast when AWS capacity fails; head builds a Vast lane for N13 | "If it keeps running out of quota, just go to vast AI, get the GPU there." (J:58) | No. |
| D12 | 11-05 19:40 | M2: score B is the primary and H_B the headline | "the question is whether the carrier can be named before birth at all and whether that exceeds the selection" (J:60) | No (chapter 5). |
| D13 | 11-05 20:15 to 11-06 00:50 | M2 amendments A1–A5 after five stops; the cross-device gate made a report at T_7–T_14 | A2: "the head notes the pattern and holds the line at this principled form: no fourth attempt without a new reason" (J:66); A3: "a threshold that cannot be frozen on control data is a measurement, not a gate" (J:71); A4: "a different failure class, a loud assertion on a banked degeneracy" (J:72); A5: "no gate goes to the instance with a failing input that has no power where it is applied" (J:74) | Only as "five attempts failed their own checks" (chapter 5); the conversion of a gate into a report is not stated (§3, C11). |
| D14 | 11-05 21:05 | The H100-on-Vast plan withdrawn; the N13 Vast lane stopped before any rental | User: "is everything runnable on AWS L40S? Because I need to keep the vast resources for another agent." (J:63, J:64) | No. |
| D15 | 11-05 21:45 | User removes the per-study GPU-hour ceilings; N13 runs its full roster | "Remove the GPU limits." (J:68); without it "the park order cuts at P1: no arm runs under the 35 ceiling on the L40S" (J:67) | No. This decision is why chapter 9 has any arms at all. |
| D16 | 11-06 00:50 | Program rule: the offline gate evaluates both sides of every gate on banked controls before launch | "five attempts, five different defects ... each checkable in advance from banked data the offline gate did not read" (J:74) | No (chapter 5). |
| D17 | 11-06 02:20 | N12 moves up | "next under the decision table: N12 (competition) moves up in J12" (J:77) | Partly: chapter 5's question names "gradient, or ... weights", not competition (§3, C13). |
| D18 | 11-06 02:40 | M2's registered follow-up: a twin matched on distinct-token count, to be specified with N12 | "the registered follow-up is a twin matched on the number of distinct tokens ... specified with J12" (J:78) | No; its result is shown in chapter 5 without saying it came from N12 on 6 November. |
| D19 | 11-06 03:50 | N12: G-VAL bar 0.5 → 0.7; N3 (route at birth) out of scope | "N3 out of scope (the unite Arm 1 review is not closed)" (J:79) | No. |
| D20 | 11-06 04:45 | N12 A1: deterministic backward | "the zero-flip clause was never shown to have room: the gap is recorded" (J:80) | No (chapter 6). |
| D21 | 11-06 05:00 | N13 A30: rndE built with stochastic rounding so its energy is matched | "as built they are not energy-matched to the top-k arms" (J:81) | No (chapter 9). |
| D22 | 11-06 08:10 | N12's gradient reading withdrawn; next instrument the fp32 ablation | "G-VAL measured the resolution of a bf16 window-summed ablation, not the gradient's validity ... the earlier 'first-order breakdown' reading is withdrawn" (J:87) | Stated with the wrong reason: chapter 6 says the prediction was noise (§3, C15). |
| D23 | 11-06 09:40 | Program rule: a gate comparing two computation paths needs a tolerance from a control pair on those paths, or it is reported | "the same lesson as M2's G-EX and N13's A29" (J:91) | No. |
| D24 | 11-06 10:00 | User: a job whose memory exceeds the L40S runs on a Vast H100, not under a precision compromise | "in cases like these they can use vast.ai" (J:92) | No (chapter 6). |
| D25 | 11-06 10:30–11:55 | N6 A2–A5: the test half becomes the larger half; 1,024 eval windows; a power gate paid first; L2 split into L2a and L2b, held behind N13's S4 | "the design has no power at 96 eval windows" (J:94); "the power gate is a pre-check on the freeze half and is paid for first" (J:95) | No (chapter 10). |
| D26 | 11-06 12:05 | J10 (N22) closed NO-GO; a format screen not scheduled | "the critic's Part 3 item 1.3 was right that the lens test had to earn its chance with an accuracy gate first, and the gate did its job for $1.40"; "N13's S4, N6's L2a and J12's remaining cards come first" (J:97) | No. A chosen study closed by its own go/no-go has no chapter. |
| D27 | 11-06 12:20 | J12's next two cards started together: N2V and N1 | N2V: "if the gradient tracks fp32 ablation, N12's gradient labels are re-read ... if not, the instrument is retired" (J:99) | Misordered: chapters 7 and 8 present N1 then N2V as a sequence (§3, C18). |
| D28 | 11-06 13:25 | N2V uses forward mode in fp32, not a Vast H100; parts "core plus the 10 % finite difference" | forward mode "keeps one GPU model and needs no Vast image" (J:101) | Partly: chapter 6 has "a forward mode that fits the memory"; the choice against the user's Vast rule (D24) is not stated. |
| D29 | 11-07 09:20 | After the outage: new executors; A31 tolerance on the energy audit; relaunch of B1 and B2 | "a check that compares two computations bit for bit (the two-path pattern once more ...)" (J:103) | No (chapter 9). |
| D30 | 11-07 10:00 | N2V: add the forward-mode part | "so that readout runs" (J:107) | No. |
| D31 | 11-07 14:05 | N6 L2a launched before N13's verdict | "the slot would otherwise idle until N13's verdict on the 8th; the practical card ... is not yet written" (J:119) | No (chapter 10). |
| D32 | 11-07, proposed; 11-08, withdrawn | The practical card (add the RL model's top-1 % MLP delta to another post-trained OLMo-3 7B) proposed, then withdrawn | proposal: "run ahead of N6 L2a if N13 says localized" (`next/HANDOFF_2026-11-07.md` lines 163–166); withdrawal: "N13 did not read localized ... So the top-1 % transplant proposal is withdrawn (the head's decision, sent with this task; no earlier file states it)" (`next/HANDOFF.md` lines 71–76) | Withdrawal stated in chapter 9 with a reason that matches; the proposal date and the fact that the journal never records the withdrawal are not. |
| D33 | 11-07 19:50 | N12's head-read sentence retagged; rule: first-order predictions in fp32 | "the cause was bf16 loss resolution" (J:136) | No (chapter 8). |
| D34 | 11-08 09:15 | N6 closes; L2b not launched; the larger-eval follow-up not taken | "the frozen rule decides; L2b (19 GPU-h) would buy a result the rule has already declared does not count. The registered next step (a larger eval set or more pairs) is not taken now; it is lower leverage than the rank test of the RL delta." (J:153) | Partly: chapter 10 says the second half was not run; the reason and the ranking against the rank test are missing. |

Two decisions from the October record that bear on the chain:

| # | date | decision | the record's own words | in the plan? |
|---|---|---|---|---|
| D35 | 10-19 (written), not applied | The already-clean plan's rule for a packing outcome | "If it lands for packing, or the instrument cannot tell, D2–D6 keep decomposition methods and the neuron basis is demoted to one view among several" (`ac/SPEC_D1.md` lines 110–113) | No. The outcome was packing (`ac/results/RESULTS.md` line 18); no file records the rule as applied or set aside, and the November program used the neuron basis. |
| D36 | 11-04 to 11-05 | The 4 November open items and the six successors were not used; the 5 November brainstorm started from a new inventory | open items at `gh-pages:index.html` line 701; successors at `studies/neurons/NEXT.md` lines 29–40; "Started 2026-11-05" (`next/README.md` line 5) | No. Chapter 2 says "Six questions were left; which first?" (§3, C-O11). |

Decisions inside the studies that the head's journal states only in passing, with the study record's own sentence: N12's
executor chose bf16 reductions in the attention kernel to fit memory, "The fp32 upcast would very likely stop the job out of
memory at T_14" (`next/n12/JOURNAL.md` lines 71–77), which is what prompted D24; N12's head read named N1 as "the next object"
and the gradient readout as needing "the small-scale check before any further use" (`next/n12/READ_head.md` lines 78–82); N1's
executor killed weights-02, ruled a deviation (`next/n1/JOURNAL.md` lines 37–39).

## 3. Statements in the plan the record contradicts

C-numbers run through the plan in order. "Contradicted" means the record states otherwise; "qualified" means the record
limits or adds a condition the plan omits.

In this section `P:NN` is line NN of `git show gh-pages:index.html` (raw HTML lines; the text is read with tags stripped),
`AJ:NN` is `ac/JOURNAL.md` line NN, `R:NN` is `ac/results/RESULTS.md` line NN and `S:NN` is `ac/SPEC_D1.md` line NN.

**§1 and chapter 1 (plan lines 16 and 86).**

- **C-O1, qualified.** Plan line 16: "Most end up with one recognizable job"; chapter 1: "The claim holds for most neurons". Record:
  "On model O7, 51.8% of poles (47.8–55.8%, Wilson 95%) have one job under explanations that pass §6.1" (R:47); Q4B 59.7 %, G4
  56.3 % (R:68, R:26). For OLMo-3 7B, the model the site is about, the interval reaches below one half. The pre-registered warning
  row "fired on all three models (ONE 0.52–0.60 against 0.90)" (S:786), and "The essay's 90 / 7 / 2–4 is its own estimate under its
  own thresholds, not ours" (R:22).
- **C-O2, contradicted.** Chapter 1 decision: "later groups mostly grow from earlier ones". Record: "On O7 ... the parent test labels
  H_SP. The unrelated share of late-born clusters is p_u = 0.265 [0.211, 0.321], lower bound above the 0.20 rule" (R:18); the
  public page, "By the rule written in advance the result is labelled packing" (P:497). Of the two readings the page says point
  toward growth, birth order "on 5 November a check showed that it does not" (P:498; R:9); the other, "groups born before the
  strongest one that later shrank are rare, 0.035 [0.026, 0.044]", stands (P:499). About three quarters of late-born groups were
  traced, so "mostly" is true of the traced share, but the record's decision is the packing label, and its margin "depends on a
  ceiling measured on easier items" (sensitivity 0.206 [0.132, 0.271], AJ:1495–1503; P:506). The two growth predictions "were wrong
  by the rules written in advance, in opposite directions" (P:501).
- **C-O3, qualified.** Chapter 1: "Sorted one neuron's fifty strongest inputs into groups; had a judge name each group and tested the
  judge against a control". Record: the object is a pole's fifty strongest *contexts* (in chapter 2's studies "inputs" means
  upstream neurons, P:512); a clusterer model sorted and named the groups; the judge picked which earlier record a late-born group
  came from (the parent test, S:295–303); the first judge failed its control twice before a second judge passed (AJ:792, AJ:816,
  AJ:839).
- **C-O4, misplaced.** Chapter 1's question opened: "What is the moment a group appears made of?" Record: this question was asked and
  answered inside chapter 2's period, "Is anything special about the moment a group appears? 2 and 3 November" (P:547–549), with
  the answer "The birth of a group is a line crossing, not an event" (P:699) and "The moment of birth could not be read" (P:545).
  The November studies (M2, N1, N2V) returned to it.
- **C-O5, qualified (dates).** Chapter 1 "October" and chapter 2 "24 October to 4 November". Record: the already-clean study ran
  19–25 October (AJ:6; `ac/README.md` line 5, "closed 2026-10-25"); its first births, parent test and one-job counts landed
  23 October (AJ:949) and its final run on 24 October (AJ:1318). Chapter 2's "training-point test" is that same run (P:496, "The
  result of the training-point test, 24 October"). Everything else in chapter 2 is from seven other studies (neuronoff, neuronrw,
  unite, reuse, alwayson, instruments, mechanics; P:508–701), none of which the plan names.

**Chapter 2 (plan line 87).**

- **C-O6, contradicted.** Chapter 2 decision: "The tools pass their checks; the growth is real". Record: the known-edit check (E2):
  "Under the frozen room gate every rule is cancelled; the readings below are EXPLORATORY" (`studies/instruments/results/RESULTS.md`
  line 43); the page, "the direction was taken from the published edit itself, so they do not show that the columns can be found
  from the weights alone" (P:686). E0: "unblinded once, gate failed" (`studies/instruments/JOURNAL.md` line 10). The 4 November
  position: "a neuron's groups use inputs they hold in common, whichever came first" and "The birth of a group is a line crossing,
  not an event" (P:698–699). Input reuse: "H_reuse is refuted at birth under the frozen rule (R1)" (`studies/reuse/results/RESULTS.md`
  line 7), read as reuse only against a reference added by amendment (P:513).
- **C-O7, contradicted.** Chapter 2 decision: "the always-on neurons are set aside from every later count". Record: "This program's
  ablation instrument uses zero replacement with the global top-1 percent exclusion" (`studies/instruments/results/RESULTS.md` line
  24); the page, "Writing zero, with the few dozen always-on OLMo neurons excluded, gives a reference near zero, and that is now the
  program's rule" (P:700). The scope is the input-replacement instrument. The October counts (one job, p_u) never used it, and no
  November spec applies it; the only mention is N6's risk row, "the positive-control pole is a massive-activation neuron"
  (`next/07_PICKS/N6_SPEC.md` line 463).
- **C-O8, qualified.** Chapter 2: "checked the normalization step". Record: "the divisor is not the route and the sentence of
  2 November, that replacing one always-on neuron changes almost any later neuron, is withdrawn" (P:552). The chapter must keep the
  withdrawal.
- **C-O9, qualified.** Chapter 2: "and another model". Record: in Qwen3.5-0.8B "Only 11 of those 128 neurons are among the top 1 %
  ... against a line of 32 set in advance", and "The planned edit test could not be read, because 193 to 200 of 200 answers ran to
  the 4,096-token limit" (P:587).
- **C-O10, misplaced.** Chapter 2's "The training-point test" belongs to chapter 1's study and its result is packing (C-O2, C-O5).
- **C-O11, not in the record as stated.** Chapter 2's question opened: "Six questions were left; which first?" Record: the page lists
  five open items, "the balanced grouping on fresh neurons; the short OLMo run of the two-model construction; whether the shared
  dependence extends to groups the sorting model did not link; whether the next comparison uses the raw score or the score per
  shared input; and the independent review of the first wiring score" (P:701); the instruments study lists six "successors"
  (`studies/neurons/NEXT.md` lines 29–40, 2026-11-04 00:50 CST). The 5 November brainstorm chose from neither; it started from a new
  inventory and 29 new cards (`next/README.md` lines 5–8; `next/03_IDEAS.md`), and no planning file under `next/` cites NEXT.md.

**Chapter 3 (plan line 88).**

- **C1, contradicted.** Plan: "Chosen in this order: does the data mix move a neuron's favorites; can the carrier be named early;
  does the gradient foretell it; do the weights move at the birth; where does a later skill live; does an older neuron cost more
  to edit." Record: "1. **N13, what RL wrote** (Program B), first ... 2. **M1, the three stage-2 mixes** ... 3. **M2, name the
  carrier before the birth** ... 4. **N22, the model gets a variable** ... 5. **N6, edit the young** ... Program A's mechanism cards
  (N1, N2, N3) follow" (`next/05_SCORING.md` lines 113–135); "Run **what RL wrote** first" (`next/08_SUMMARY.md` line 7);
  `next/09_PLAN.md` line 21: "J5 opens the first pick". The plan's list is the order in which results landed. N12
  ("does the gradient foretell it") was not a pick; it "moves up" on 6 November after M2 (J:77). N22, the fourth pick, is absent.
- **C2, contradicted.** Plan: "Chosen by expected information per GPU-hour". Record: "Leverage = product of the five" (wow,
  novelty, decisiveness, edge, inverted cost; `next/05_SCORING.md` lines 9–19); after the critic, "Only a card with a before/after,
  a prediction made before the measurement, or an engineer's number can be a pick" (lines 97–99); the folder's own description,
  "leverage = wow × novelty × chance of a clear answer, divided by cost" (`next/README.md` line 31). No file uses "expected
  information per GPU-hour".
- **C3, qualified.** Plan: "a known number was reproduced on the cloud machine before anything new was trusted". Record: the first
  cloud run of 5 November, the data-order loss check, did not reproduce its number under its frozen rule, "NOT CONFIRMED (+0.164
  nats, 6.6 sd)" (J:21); it was closed later that day by a second check (J:48). Each study then reproduced its own known number
  (M1 J:33; N13 J:55; N6 J:91; N22 J:97).
- **C4, qualified.** Plan: "One night of inventory, reading and scoring; a critic argued against each idea". Record: the same night
  also produced the attack on the October study's birth dates and its verification (J:15–16), which the chapter does not name.

**Chapter 4 (plan line 89).**

- **C5, contradicted (quantity).** Plan picture: "three distributions of how many of the fifty changed". Record: "The measure, per
  neuron pole: the fraction of its 50 strongest-activation positions on the fixed 20.6M-token corpus that fall in the five code and
  math directories" (`next/m1/READ_head.md` lines 12–14). "How many changed" is the exploratory turnover quantity
  (`next/m1/EXPLORATORY_turnover.md`), not the registered one.
- **C6, contradicted.** Plan: "The favorites move ... in the direction and roughly the amount the mix predicts." Record: H_order
  UNDECIDED, "the ordering holds at the resolution of the bootstrap but not at the resolution of ordinary stage-1 drift"
  (`next/m1/READ_head.md` lines 35–38); "The contrast is mostly G moving away from code and math, not M moving toward it" (line 26);
  "λ landed above the linear expectation" (line 55). No amount was predicted. Also the plan's "by less than one slot of fifty" is
  right for G only after the 8 November correction, "0.6–0.9 of one slot" (line 30; J:149).
- **C7, not in the record.** Plan decision: "They are stable enough to be the object of the birth studies." Record decision: "Under
  spec §1.2 the next step for this outcome is to report, not to run the per-pole analysis" (`next/m1/READ_head.md` lines 68–72).
  No file links M1 to the birth studies' choice of object.
- **C8, not in the record.** Plan's question opened: "Can you tell which neuron will take a new job before it does?" M2 was adopted
  at 01:35 CST on 5 November (`next/05_SCORING.md` line 126), before M1's first harvest (16:57Z, `next/m1/JOURNAL.md` line 13).
- **C9, qualified.** Plan question: "Does changing the training data move what a neuron responds to?" Record: the measure answers
  "which documents hold a feature's strongest instances"; the qualified reading is that "the mix changes which of the feature's token
  families dominates its strongest instances" (`next/m1/READ_head.md` lines 47–49, 89–92).

**Chapter 5 (plan line 90).**

- **C10, contradicted.** Plan: "Scored all 22,016 neurons". Record: "rank the carrier among its layer's 22,016 poles" (J:60);
  "MLP width 11,008, so 22,016 poles per layer" (`next/07_PICKS/M2_SPEC.md` line 116). Poles of one layer, not neurons.
- **C11, qualified.** Plan: "five attempts failed their own checks before the sixth ran clean". Record: three stopped at the
  cross-device G-EX gate, one at G0 because its failing input had no power, and one "crashed in the scoring code at T_10, not on a
  gate" (`next/m2/JOURNAL.md` lines 14, 38, 54, 72, 90). The gate that stopped three runs was loosened twice and then made a report
  at T_7–T_14: "a threshold that cannot be frozen on control data is a measurement, not a gate" (J:71). The sixth run passed every
  gate that remained, and still reported the cross-device residual above tolerance at T_13, T_11, T_10, T_9 and T_7 (J:74, J:76).
- **C12, misattributed and not in the record.** Plan picture: "a slider over vocabulary breadth closes the gap"; decision: "the
  difference is how many distinct words the contexts contain. The score measures vocabulary breadth, not the job." Record: the M2
  reader's finding is post hoc, "a hypothesis for a registered test, not a result" (`next/m2/READ_m2.md` line 170); the registered
  test is N12's H_dt, 6 November, "token repetition explains M2's twin gap, as pre-registered with prior 0.50" (J:84); the record's
  reading, "repeated-token contexts are contested, and a cluster of distinct tokens is named about as well as a random group of the
  pole's own contexts" (`next/n12/READ_head.md` lines 45–46). No file says "vocabulary breadth" or that the score "measures ... not
  the job". The records count distinct tokens, not words.
- **C13, qualified.** Plan's question opened: "Is it special in its gradient, or in its weights?" Record: "next under the decision
  table: N12 (competition) moves up in J12" (J:77). The gradient and the weights became the next questions only in N12's own
  decision, "the weight change behind the carrier's rise is the next object (N1 ...), and the gradient readout needs the small-scale
  check" (`next/n12/READ_head.md` lines 78–79).

**Chapter 6 (plan line 91 and the F9 row, plan line 106).**

- **C14, qualified.** Plan: "agreement 0.10". Record: "Spearman ... 0.100 on the freeze half and 0.043 on the test half, bar 0.7"
  (J:84). The test half decides; quote both.
- **C15, contradicted.** Plan: "in the precision these tools usually run in, the prediction is noise"; F9 sentence: "In half
  precision the prediction is noise (0.10). In full precision it is nearly exact (0.96)." Record: "G-VAL measured the resolution of
  a bf16 window-summed ablation, not the gradient's validity ... the earlier 'first-order breakdown' reading is withdrawn" (J:87);
  "N12's banked bf16 first-order prediction tracks single-neuron zero ablation once the loss is read in fp32 ... the bf16 resolution
  of the summed loss was the whole problem"; "The bf16 prediction against the fp32 forward-mode prediction: Spearman 0.995" (J:134).
  The 0.96 is the half-precision prediction against the full-precision measurement (`next/n2v/JOURNAL.md` line 110). The noisy side
  was the measurement. The plan's wording matches `next/21_REFLECTION.md` line 64 ("first-order predictions in bf16 ranked zero
  ablations at Spearman 0.10"), which is itself contradicted by J:87 and J:134.
- **C16, contradicted.** Plan: "Its answers were set aside unread." Record: "exploratory, not counting: u_S 0.527 ..., H_bridge
  −0.013" (J:84), read in the head's read (J:85); N2V counted them later as "the values N12's exploratory read printed on the 6th"
  (J:134). They were read and not counted.
- **C17, qualified.** Plan: "We rebuilt it in full precision, in a forward mode that fits the memory." Record: the deciding readout
  compared "N12's banked bf16 prediction against the fp32 zero ablation" (`next/n2v/JOURNAL.md` line 110); forward mode was added for
  an fp32 prediction and the own-token readout (J:101, J:107). What was rebuilt in full precision was the measurement. The forward
  mode was chosen over the user's Vast rule because it "keeps one GPU model and needs no Vast image" (J:101).

**Chapter 7 (plan line 92 and the F11 row, plan line 107).**

- **C18, misordered.** Plan's question opened: "What does the gradient say, now that the instrument works?" Record: N1 and N2V were
  started together (J:99, 2026-11-06 12:20 CST). N1's verdict was at 2026-11-07 19:35Z (`next/n1/JOURNAL.md` line 73); N2V's
  instrument was shown to work at 2026-11-08 00:53Z (`next/n2v/JOURNAL.md` line 107). N1 could not have opened a question that
  assumed N2V's result.
- **C19, contradicted.** Plan: "For 67 births"; picture: "67 paired dots and the median with its band". Record: 67 pairs out of 71
  test births (`next/n1/JOURNAL.md` line 74); the statistic is the mean of the paired differences with a bootstrap interval:
  "Mean of x - y over paired units ... with a percentile bootstrap 95 % interval" (`infra/pg/m2.py` lines 546–553, called from
  `infra/pg/n1.py` line 227).
- **C20, qualified.** Plan: "compared the change in the carrier's input weights with a comparison neuron's dated alike". Record: input
  rows and output column, each against distinct-token twins, M2's twins and a matched-neuron floor; output column AT_SELECTION,
  −0.258 [−0.654, 0.130] (J:124).
- **C21, contradicted.** Plan: "The birth is in the activations, not in the weights." Record: "what is special about a real birth is
  not in the carrier's weight norms. The spec's decision table names the next instrument for NEITHER: a job-direction projection of
  the row change, not a norm" (`next/n1/READ_head.md` lines 46–48); the abstract, "the size of the carrier's weight change is not
  the event" (`next/20_ABSTRACT.md` line 16). The record limits the claim to the size of the change; the direction is untested.
- **C22, qualified.** Plan: "The margin is two births, so this is held lightly." Record adds two more limits: "the freeze half read
  AT" and "carriers' max |activation| is 1.39 times the twins' (adjusting widens to AT, −0.20 [−0.59, 0.21] ...)" (J:127).

**Chapter 8 (plan line 93).**

- **C23, qualified.** Plan: "Ran the full-precision instrument on 284 switch-offs". Record: 284 test pairs and 328 freeze pairs, 612
  in all (`next/n2v/JOURNAL.md` line 72). The run's first attempt did not count (J:112).
- **C24, misdated and qualified.** Plan decision: "The carrier wins by its own rise". Record: this is N12's realized label of
  6 November (J:84), and its reader found "OWN_RISE's sign is close to forced by the dating rule" (J:87). N2V re-read only N12's
  gradient labels (J:134).
- **C25, qualified.** Plan: "the gradient one snapshot earlier does not single it out". Record: H_signal REFUTED, "status a dated
  re-reading, not a prediction" (J:134).
- **C26, not in the record.** Plan's question opened: "Where does a skill learned after pre-training live?" N13 was the first pick
  (`next/05_SCORING.md` line 113) and had run since 5 November (J:30).

**Chapter 9 (plan line 94 and the F8 row, plan line 108).**

- **C27, contradicted.** Plan: "a random 0.1 % of all changed numbers scaled up a thousandfold". Record: "Here the pool is the MLP
  only ... This is a stated deviation" (`next/07_PICKS/N13_SPEC.md` lines 428–429); "a random 0.1 % of MLP entries scaled to estimate
  the whole change reads 85.2" (`next/20_ABSTRACT.md` line 20).
- **C28, qualified.** Plan: "attention only". Record: "all MLP alone 86.4 REC, attention and norms alone 83.0 REC" (J:145).
- **C29, qualified.** Plan: "scored 500 competition problems". Record: GSM8K (1,319 problems) was held out and scored at 1 %, "GSM8K at
  1 % REPLICATES on 3 of 3" (J:145).
- **C30, qualified.** Plan: "Three days of runs, each checked against a known number first". Record: the known number ran once,
  in R0 attempt 7, "39.96 against the published 40.0 ± 1.6" (J:55); each later launch was checked by tensor hashes, completeness and
  engine equality (J:104, J:145).
- **C31, qualified.** Plan: "The gain is spread across the network." Record: "No 0.1–3 % neuron subset carries the gain; a sparse
  unbiased estimate of the whole MLP change does" (J:150, the wording the abstract check imposed because an earlier sentence
  "claimed more than was tested"); "DARE is count-matched and not energy-matched ... so 'spread or scale' is open" (J:145);
  "DARE's realized energy is 33 to 1,000 times the whole change's" (J:146).
- **C32, qualified.** Plan: "We withdrew the plan to edit models by picking their most-changed neurons." Record: the withdrawn item is
  the card "add the RL model's top-1 % MLP delta to a different post-trained OLMo-3 7B model" (`next/HANDOFF_2026-11-07.md` lines
  163–166), withdrawn in `next/HANDOFF.md` lines 71–76, which says "no earlier file states it". The journal does not record it.

**Chapter 10 (plan line 95 and the F13 row, plan line 109).**

- **C33, qualified.** Plan: "Edited along a neuron's own direction against a random direction of equal size, 21 pairs". Record: the 21
  are early-born and late-born poles matched in one layer; the registered contrast is their difference in edit cost (H_age); the
  random-direction edit is a per-pair control (J:89; `next/07_PICKS/N6_SPEC.md` lines 110–134). The plan omits the age pairing,
  which is the study's question.
- **C34, contradicted.** Plan: "a control pair in eight layers had to order correctly first". Record: the control needed 6 of 8 layers
  (`next/07_PICKS/N6_SPEC.md` lines 310–313) and ran inside the same launch as the freeze-half edits: "N6 ran its positive control
  inside the main run ... Run first, on their own, they would have cancelled the study" (`next/21_REFLECTION.md` lines 74–77).
- **C35, qualified.** Plan: "the random edit moved the model 1.3 to 2.5 times more". Record: the 1.3–2.5 is the change in the residual
  stream, R (`next/n6/JOURNAL.md` line 225); the inversion itself is in loss, "the column edit cost more than the random edit of the
  same c in 2 of 8 layers (24, 28; need 6), the swapped labels in 6 of 8" (J:153). Name both.
- **C36, qualified.** Plan: "We set the study aside and did not run its second half." The reason is missing (§2, D34).

**Chapter 11 (plan line 96).**

- **C37, qualified.** Plan: "Five results stand, one study is set aside." Record: also "two pilots decided (J8 passed, J10 NO-GO)"
  (`next/21_REFLECTION.md` line 14).
- **C38, qualified.** Plan: "a second model re-derived every number from the files, four times". Record: for N6 "the independent
  recount covers G-PC and the R ratios but not τ_s itself"; "N6 has no READ files" (J:155); "3 design values without an artifact"
  (J:149).
- **C39, not in the record.** Plan's question opened: "whether a model can choose a weight edit by reasoning about what the neurons do,
  against the random control that beat us". No file states this question. The nearest line is the head's answer to the user's
  question "whether the program used LLM reasoning rather than classical methods" (J:156). If it is a new proposal of 8 November it
  should be dated and labelled as such.

**§1 of the plan (plan lines 12–23).**

- **C40, contradicted.** Plan line 22–23: "The reader should finish knowing ... why each experiment followed the one before it." See
  §0: most of the eleven chapters did not follow from the one before.
- **C41, qualified.** Plan line 14–15: "We kept sixteen snapshots ... from its random start to the end of its first training stage."
  Record: the sixteen are AI2's released stage-1 checkpoints, `stage1-step0` to `stage1-step1413814` (`next/00_ASSETS.md` lines 31,
  46); they were harvested, not kept by us. The range is right.

**Outside §1 and §3, noted because chapters 1, 5, 7 and 8 depend on them.**

- **C42, contradicted.** Plan §3a M5 (plan line 65): a birth is "the earliest snapshot at which its contexts are among the neuron's
  fifty". Record: "the first point t at which the median pole-direction activation over the job's final-point exemplar positions is
  at least the pole's own 99.9th percentile at t and at t+1" (`next/README.md` lines 48–50).
- **C43, contradicted.** Plan §3a M6 (plan line 66): "set the activation to zero at every token". Record: "zeroing a pole at its
  exemplar positions" (`next/07_PICKS/N2V_SPEC.md` line 152); the unit window is "2,049 tokens (BOS + 2,048)" (line 101).

## 4. The October chapter

The plan folds two different things into chapters 1 and 2: the already-clean study itself (19–25 October, closed 25 October)
and seven later studies run between 25 October and 4 November. For the chain to be honest the two chapters must keep the lines
below. Abbreviations as in §3. Each line: date; what happened; source.

**Chapter 1, the already-clean study (19–25 October).**

1. 19 Oct — The frozen plan states the stake: "If Q2 lands for recursive generalization, our next five studies (D2–D6) use the
   model's own neurons as the unit ... If it lands for packing, or the instrument cannot tell, D2–D6 keep decomposition methods and
   the neuron basis is demoted to one view among several" (S:110–113).
2. 20 Oct 02:58Z — Predictions written before the real run: for the parent test "H_RG ... Probability 0.65; mixed 0.30; H_SP 0.05"
   (AJ:77–78). Packing, the 0.05 branch, is what landed.
3. 20 Oct 07:50Z — The explanation prompt passed 22.5 % on the discovery set; "The prediction (55–75% pass) was wrong by a factor of
   three" (AJ:178–195).
4. 20 Oct 08:20Z–18:40Z — The explainer model changed from DeepSeek to Opus to Sonnet; the user's rule "no Opus through the API"
   stopped a wave with 1,351 answers banked; explanations ended mixed by model and the model is confounded with depth (AJ:250,
   AJ:443, AJ:483, AJ:510; R:147).
5. 20 Oct 16:48Z — Explainer prompt v3 frozen at 61.9 % (AJ:359–403).
6. 20–21 Oct — Sixteen checkpoints harvested on three GPU kinds, 26.9 billed GPU-h; a device check found 0 of 640 neurons flip at
   T_14, so the split stands (AJ:648–661, AJ:874–890).
7. 21 Oct 17:15Z–17:35Z — The first clusterer failed its two-clusterings agreement test (10 of 21); clusterer prompt v2; Sonnet chosen
   as clusterer at 29 of 39 (AJ:677, AJ:728, AJ:792).
8. 21 Oct 17:29Z–17:37Z — The first judge failed its control twice (0.673, 0.716); the Sonnet judge passed at 0.910; the hard-decoy
   control at 0.692 was later reported "not measurable" (AJ:792, AJ:816, AJ:839; R:149).
9. 21 Oct 21:47Z — API limit reached; the rest of the sorting and all judging ran as Claude Code subagents (AJ:892, AJ:908; P:502).
10. 23 Oct 20:02Z — First unblind: p_u 0.261 labels H_SP; birth order read "main oldest"; one job 0.597, 0.563, 0.518 (AJ:949).
11. 23 Oct 20:20Z–23:32Z — A centre birth curve added after the unblind as exploratory; "early for ONE" missed (AJ:1029, AJ:1070).
12. 24 Oct 00:12Z–02:30Z — Checker flags F13–F22; 375 judge lines and 200 clusterer answers re-answered; six unblind runs, p_u
    0.260–0.265, no label moved (AJ:1104–1348).
13. 24 Oct 08:20Z — Red team: the judge's controls are easier than its test items; p_u sensitivity 0.206 [0.132, 0.271]; the rise
    with birth time is between neurons, not within; "phase change" wording withdrawn (AJ:1491–1520; P:505–506).
14. 25 Oct — F41 (negative-pole explainer prompts showed the positive pole's vocabulary) and F42 (a Gemma artifact; three
    cross-model claims withdrawn); neither re-run; study closed (AJ:1522, AJ:1547; `ac/README.md` line 5).
15. Final readings to show: one job O7 51.8 % [47.8, 55.8], Q4B 59.7 %, G4 56.3 % (R:26, R:47, R:68); parent test p_u 0.265
    [0.211, 0.321], H_SP (R:18); groups that were born first and shrank 0.035 (P:499); 17 of 600 neurons ended with an untraced
    late-born strongest group (P:500).
16. 5 Nov 15:50Z — The birth-order reading retagged, not supported (AJ:1653; R:9; `next/10_RETAG.md`). Chapter 1's births picture
    must carry this.

**Chapter 2, the studies of 25 October to 4 November.**

17. 25–27 Oct — Three single-neuron studies: neuronoff's comparison cancelled because its removal check failed for 42 % and 37 % of
    neurons; neuronrw without a result; neuronshow a hand-run exploration (P:688–691).
18. 31 Oct–1 Nov — Wiring score 0.356 against 0.159 for fake neurons; a review found it "hardly changed" under another neuron's rows,
    left open (P:509; `studies/unite/results/reads/REVIEW_2026-11-01.md` line 5).
19. 1 Nov — Replacing the older group's 50 inputs with their average: 0.226 [0.170, 0.308]; "H_reuse is refuted at birth under the
    frozen rule" against 0.174; read as reuse against a birth-point reference 0.031 (`studies/reuse/results/RESULTS.md` lines 3, 7;
    P:512–513).
20. 1–2 Nov — The always-on neurons found (the top 1 % by average activation, 3,412) and excluded from 2 November (P:539–543).
21. 2 Nov — Normalization check: "the divisor is not the route", the 2 November sentence withdrawn (P:551–552).
22. 2–3 Nov — "Is anything special about the moment a group appears?": the ratio rises through all of training; the sharpest change
    sits near the birth in 0.307 of cases against 0.283 for shuffled births (P:547–549).
23. 2–3 Nov — The always-on hypothesis for a published Qwen3.5-0.8B edit: 11 of 128 neurons against a line of 32; the edit test
    could not be read (P:586–587).
24. 3 Nov — The older group is not preferred: −0.046 [−0.091, 0.030] (P:556).
25. 3 Nov — E0's gate failed (`studies/instruments/JOURNAL.md` line 10).
26. 3–4 Nov — Writing zero with the always-on neurons excluded adopted as the ablation instrument's rule (`studies/instruments/results/
    RESULTS.md` line 24; P:592–594, P:700).
27. 3–4 Nov — Groups from input profiles agree with the clusterer at 0.25 (P:644–647).
28. 3–4 Nov — The known-edit check: "Under the frozen room gate every rule is cancelled" (`studies/instruments/results/RESULTS.md`
    line 43; P:685–686).
29. 4 Nov — The position at the end: "whichever came first"; "a line crossing, not an event"; five open items (P:697–701); the
    instruments study's six successors (`studies/neurons/NEXT.md` lines 29–40).

## 5. Questions that were opened and never closed

The plan's last chapter names two open questions (plan line 96). The record holds these others, none answered by a later
chapter. Each: the question; where it is recorded.

**From the October record.**

1. Why random groups are first-born so often: ties, size or median activation, "not decided here" (R:15; P:498).
2. The packing label's margin rests on a judge ceiling measured on easier items; per-bin p_u is not identified (AJ:1495–1503).
3. The hard-decoy control was not measurable, so the boring reading "the judge separates true from decoy by domain alone" is not
   ruled out (R:103, R:149; S:122).
4. The frozen plan's decision for packing ("the neuron basis is demoted to one view among several", S:110–113) is not recorded as
   applied or set aside; the November program used the neuron basis throughout.
5. The explainer model is confounded with depth (R:147); F41 and F42 were not re-run (AJ:1522–1545, AJ:1547–1602).
6. The five open items of 4 November (P:701) and the six successors (`studies/neurons/NEXT.md` lines 29–40); none was taken up.
7. The wiring score's review (Arm 1) is not closed, which kept N3, route at birth, out of scope (J:79;
   `studies/unite/results/reads/REVIEW_2026-11-01.md` line 5).

**From the November record.**

8. What a birth is, beyond weight norms: "a job-direction projection of the row change, not a norm, and the readers of the column",
   with the post hoc hypothesis that the rise "is carried by a change in the residual stream the carrier reads" (`next/n1/READ_head.md`
   lines 47–51); "the winner is decided between the checkpoints, by something a single gradient at T_(k−1) does not show and the
   carrier's own weight norms do not carry" (`next/n2v/READ_head.md` lines 60–67).
9. N1's activation-size confound: "which side of the pair is unusual is not settled" (J:127).
10. Whether "coherent" and "late" births differ: "not separated by this run" (J:77); "the W4 judge's label accuracy as a second
    coherence measure points the other way" (J:78).
11. N12's H_bridge, UNDECIDED after the fp32 re-read (J:134); per-pole ablation for competitors, about 1 GPU-h, not run
    (`next/n12/READ_head.md` lines 71–72).
12. The location of onsets inside their brackets on intermediate checkpoints, 2.7–3.2 GPU-h, "not registered" (`next/n1/JOURNAL.md`
    line 10).
13. M1's ordering of the 40 % mix, H_order UNDECIDED; two instrument fixes "before the instrument is reused"; the per-pole analysis
    not run (J:51; `next/m1/READ_head.md` lines 60–72); the relocation threat from the Goodfire preprint (J:52).
14. N13's H_artifact INDETERMINATE (J:145); "What that leaves open is the shape of the change that the network tolerates such noise
    on" (`next/20_ABSTRACT.md` lines 67–68); the GSM8K small contrasts fragile, the mlp − attn gap in cap-hit problems (J:146). The
    rank test is the plan's first open question and is not specified or launched (`next/HANDOFF.md` lines 62–70).
15. N6's question, whether an older neuron costs more to edit, is unanswered; the registered next step ("a larger eval set or more
    pairs", about 5,760 windows) "is not taken now" (J:153); the effect-matched design is listed "as a finding, not yet as a spec
    rule" (`next/00_INDEX.md` lines 85–88); N6's τ_s not independently recounted (J:155).
16. N22: "A later format screen (easier expressions, about 0.3 GPU-h) is possible but not scheduled" (J:97); L2, L3 and N25's labels
    not run (`next/n22/JOURNAL.md` lines 102, 110–120).
17. The data order: the HF forward sits above the logged loss by +0.0055 to +0.0094, "cause unmeasured"; 32 known-number rows against
    the plan's 64 (J:34, J:48); its two remaining uses, skipped updates by batch (M3) and exposure counts (N29), never run
    (`next/05_SCORING.md` lines 140–141).
18. The robot grounding gate, deferred "until the N13 and M1 headline results are in" (J:24), never run (`next/HANDOFF.md` lines
    94–95).
19. The code-model project on RL-Zero-Code: "the practical card and the code project wait on the user" (J:137; `next/HANDOFF.md`
    lines 82–86).
20. The SFT → DPO harvest from the Goodfire note, "a new question, not scheduled" (J:53).
21. N2 as rewritten after the competitor read, "does the concept exist before its neuron" (J:14), and N27's zero-GPU version on
    public evaluation curves, which "may run later on verified births" (`next/05_SCORING.md` lines 139–140); neither ran.

## 6. Count

| section | items | of which |
|---|---|---|
| 1. Experiments missing from the chapters (November) | 48 | chapter 3: 8; chapter 4: 6; chapter 5: 4; chapter 6: 5; chapter 7: 3; chapter 8: 4; chapter 9: 10; chapter 10: 5; J10 with no chapter: 1; chapter 11: 2 |
| 2. Decisions missing or misdated | 39 | 36 in the two tables (D1–D36) and 3 in-study decisions in the note; of the 36, 28 not stated at all and 8 stated partly, with a wrong reason or in the wrong order |
| 3. Statements the record contradicts | 54 | 27 contradicted, misattributed, misdated, misplaced or without a source; 27 qualified. October (C-O1–C-O11): 11; November and §1 (C1–C41): 41; §3a, noted outside scope (C42–C43): 2 |
| 4. October experiments and decisions the two chapters must keep | 29 | chapter 1: 16; chapter 2: 13 |
| 5. Questions opened and never closed | 21 | October: 7; November: 14 |

The three findings that matter most:

1. Chapters 1 and 2 state the reverse of the October record's decisions (C-O2, C-O6, C-O7): the frozen label was packing, the
   birth-order reading was retagged, the known-edit check had every rule cancelled, and the always-on exclusion is one instrument's
   rule, not a rule of every later count. The frozen plan's own decision for a packing outcome is not recorded as applied (D35).
2. Chapter 6 and the F9 caption blame the half-precision prediction; the record blames the half-precision measurement and
   withdrew the "first-order breakdown" reading on 6 November (C15, C16, D22). The plan's wording matches
   `next/21_REFLECTION.md` line 64, which J:87 and J:134 contradict.
3. The decision chain is reconstructed rather than recorded (§0; C1, C2, C8, C18, C26, C39): the record chose N13 first by a
   leverage rubric and a "before/after" rule, and the chapters omit the 5 November recount and retag that fixed the unit of the
   birth studies (items 3–6), the user's "Remove the GPU limits" without which no N13 arm would have run (D15), M2's conversion of a
   failing gate into a report (C11, D13), and the N22 study closed by its own go/no-go (item 46, D26).
