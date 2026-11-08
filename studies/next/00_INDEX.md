# 00 — Index of the already-clean neuron program on OLMo-3 7B

Written 2026-11-08 by an Opus worker for the head. This file holds no result. Every number names the file it was copied from.
Paths are under `studies/next/` unless they start with `infra/`. `OPS/` is `infra/aws_pg/operations/`. "JOURNAL" is
`studies/next/JOURNAL.md`; an entry is named by its stamp as written and its line number.

## (a) Read in this order

| # | path | what it gives you | time |
|---|---|---|---|
| 1 | `HANDOFF.md` | what is closed, that nothing is running, the decisions open for the user, the standing rules, how to resume | 3 min |
| 2 | `20_ABSTRACT.md` | the final 300-word abstract, the evidence table (one row per load-bearing number, with its artifact and who recounted it), the next step and its cost | 4 min |
| 3 | `README.md` | the vocabulary: pole, job, birth, reader, route; the house rules for every file here | 1 min |
| 4 | §(b) below | each study's question, label, key number, and the paths of its spec, verdict, reads, figure and cost | 2 min |
| 5 | `n13/READ_head.md` | the head's read of the largest study (N13), claims N13-1 to N13-9 | 3 min |
| 6 | `20_ABSTRACT_CHECK.md`, first 20 lines | the fresh reader's check of every number in draft 3 (175 checked, 4 mismatched); later passes are in JOURNAL 2026-11-08 05:58 CST (line 150) and 09:30 CST (line 155) | 2 min |
| 7 | `08_SUMMARY.md` | why these studies were chosen (the brainstorm's one-page summary, 2026-11-05 02:00 CST) | 2 min |
| 8 | `JOURNAL.md` | the head's log, 153 entries (lines 3 to 155); search it by study id or stamp | about 60 min in full |

## (b) Studies

Status labels are copied from each verdict file's headline. Intervals are 95 %.

| study and question | status | key number | spec | STATE | thresholds; verdict | blind read; head read | claims | figure | GPU-h; dollars (note) | closed in JOURNAL |
|---|---|---|---|---|---|---|---|---|---|---|
| **M1**: does the stage-2 data mix (0, 40 or 65 % code and math) move where neurons' strongest activations land? | CLOSED: `H_lead SUPPORTED; H_order UNDECIDED; H_specific SUPPORTED` | M − G, code-and-math share of each pole's top-50 activations, band mean: +0.0109 [0.0104, 0.0114] (band 0) to +0.0196 [0.0192, 0.0200] (band 3), against drift bars 0.0082 and 0.0035 (`m1/verdict_m1.json` lead) | `07_PICKS/M1_SPEC.md` (also `M1_ASSETS.md`, `M1_OFFLINE_GATE.md`) | none; study log `m1/JOURNAL.md` | `m1/thresholds.json`; `m1/verdict_m1.json` | `m1/READ_m1.md`; `m1/READ_head.md` (exploratory: `m1/EXPLORATORY_turnover.md`, `m1/EXPLORATORY_reading_control.md`) | none | `m1/figures/fig_m1.png` | about 10.3; about $23 (`OPS/2026-11-05_next_m1_harvests.md`; K0 in `OPS/2026-11-05_next_m1_k0.md`) | 2026-11-05 17:40 CST (line 51) |
| **M2**: can the pole that will carry a new activation cluster be named one checkpoint before its birth? | CLOSED: `H_B SUPPORTED; H_sel_B BELOW_SELECTION; H_A SUPPORTED; H_AB LOSES` | carrier in its layer's top 10 of 22,016 poles: 42 of 71 = 0.592 [0.475, 0.698]; dated twins 53 of 70 = 0.757; carrier − twin −0.171 [−0.286, −0.057] (`m2/verdict_m2.json` labels, test.pb_pre_B) | `07_PICKS/M2_SPEC.md` | none; `m2/JOURNAL.md` | `m2/thresholds.json`; `m2/verdict_m2.json` | `m2/READ_m2.md` (blindness partly broken by commit subjects, disclosed); `m2/READ_head.md` | none | none | 3.324–3.467; $8.53–8.91 (`OPS/2026-11-05_next_m2_score.md`) | 2026-11-06 02:40 CST (line 78) |
| **N12**: at a birth, does the carrier win by its own rise or by its competitors falling, and does the gradient one checkpoint earlier single it out? | CLOSED: `H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal DOES NOT COUNT; H_bridge DOES NOT COUNT; H_dt SUPPORTED`; the gradient labels were re-read under N2V (`n2v/verdict_n2v.json` n12_reread) | own-rise F +0.951 [0.738, 1.180] decades of rank; competitors' drop, real − M2 twins −0.226 [−0.408, −0.056] (`n12/verdict_n12.json` labels) | `07_PICKS/N12_SPEC.md` | none; `n12/JOURNAL.md` | `n12/thresholds.json`; `n12/verdict_n12.json` | `n12/READ_n12.md` (blindness broken by a commit subject, disclosed); `n12/READ_head.md` (dated postscript after N2V) | none | none | 2.483–2.520; $5.56–5.64 (`OPS/2026-11-06_next_n12_score.md`) | 2026-11-06 08:10 CST (line 87) |
| **N2V**: does N12's first-order prediction (activation × gradient) track single-neuron zero ablation when the loss is read in fp32? | CLOSED: `G-VAL window PASS; G-FD PASS; G-VAL own PASS; branch A; N12 re-read: H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal REFUTED; H_bridge UNDECIDED; H_dt SUPPORTED` | test half Spearman 0.960 [0.915, 0.990], 284 pairs, 71 units (`n2v/verdict_n2v.json` test.window) | `07_PICKS/N2V_SPEC.md` | `n2v/STATE.md` | `n2v/thresholds.json`; `n2v/verdict_n2v.json` | `n2v/READ_n2v.md`; `n2v/READ_head.md` | N2V-1 to N2V-7 in `n2v/READ_head.md`; no CLAIMS.md | none | at most 6.59; $14.75 (`OPS/2026-11-07_next_n2v.md`) | 2026-11-07 19:50 CST (line 136) |
| **N1**: at a birth, do the carrier's input rows (gate, up) or its output column change first? | CLOSED: `NEITHER: in BELOW_SELECTION (NOT_ABOVE_FLOOR), out AT_SELECTION (NOT_ABOVE_FLOOR)` | input rows, real − distinct-token twin on e: −0.432 [−0.817, −0.059], 67 pairs; margin two births (`n1/verdict_n1.json` labels.sel_in; `n1/read_n1/fragility.json`) | `07_PICKS/N1_SPEC.md` | `n1/STATE.md` | `n1/thresholds.json`; `n1/verdict_n1.json` | `n1/READ_n1.md`; `n1/READ_head.md` | N1-1 to N1-6 in `n1/READ_head.md`; no CLAIMS.md | none | 0 GPU-h; CPU at most 4.68 instance-h, about $2.48 (`OPS/2026-11-06_next_n1.md`) | 2026-11-07 14:55 CST (line 129) |
| **N13**: is the MATH-500 gain of OLMo-3.1-7B-RL-Zero-Math over its base carried by its most-changed MLP neurons? | CLOSED: verdict `COMPLETE`; H_local REFUTED; H_diffuse SUPPORTED; H_attention REFUTED; H_artifact INDETERMINATE; necessity NOT NECESSARY at all four k | MATH-500 accuracy, n 500: top 1 % neurons transplanted 61.8 [57.5, 66.0] %, DARE at 1 % 85.8 [82.5, 88.6] %, top − DARE −24.0 [−28.0, −19.6] points; base 57.2, RL 92.0 (`n13/verdict_n13.json` arms, comparisons) | `07_PICKS/N13_SPEC.md` (also `N13_MATH_TASK.md`, `N13_OFFLINE_GATE.md`) | `n13/STATE.md` | `n13/thresholds.json`; `n13/verdict_n13.json` (also `n13/verdict_n13.md`) | `n13/READ_n13.md`; `n13/READ_head.md` | `n13/CLAIMS.md` (N13-C1 to C13); N13-1 to N13-9 in `n13/READ_head.md` | `n13/figures/fig_n13.png`, `fig_n13.pdf` | 102.555; about $249.16, see §(e) (`OPS/2026-11-05_next_n13_r0.md`, `OPS/2026-11-06_next_n13_s4.md`) | 2026-11-08 05:31 CST (line 146) |
| **N6**: does editing an early-born pole cost more corpus loss than editing a matched late-born pole? | DOES NOT COUNT: `H_age DOES NOT COUNT; H_adj DOES NOT COUNT; H_eff DOES NOT COUNT` | no test label. G-PC: the column edit beat the random edit in 2 of 8 layers (6 needed); G-POWER: τ_s 1.5096e-6 against the bound 6.363e-7 nats per token, ratio 1.186, bar 0.5 (`n6/verdict_n6.json`; `n6/banks/L2a-01/metrics.json` gates.G_PC) | `07_PICKS/N6_SPEC.md` | `n6/STATE.md` | `n6/thresholds.json`; `n6/verdict_n6.json` | no READ files; the head's read is JOURNAL 2026-11-08 09:15 CST (line 153); G-PC and R recounted by `abscheck/check_n6.py` | N6-1 to N6-3 in `n6/STATE.md`; no CLAIMS.md | none | 19.581–19.626; $43.86–43.96 (`OPS/2026-11-07_next_n6.md`; L1 in `OPS/2026-11-06_next_n6_edit.md`) | 2026-11-08 09:25 CST (line 154) |
| **J8** (`pilot0/`): do the reconstructed stage-1 batches reproduce the logged training loss? | CLOSED: both steps in the pass branch | pad-excluded loss minus logged loss, S − W: +0.0079 (step 61,001) and +0.0094 (step 100,001) nats per token, bar 0.0124; neighbour batches 0.0823 and 0.0483 away; one batch per step, no interval (`pilot0/GPU_CHECK2.md`) | rule frozen in `09_PLAN.md` line 517 (card line 158); code `pilot0/scripts/rule.py` | none; records `pilot0/README.md`, `pilot0/GPU_CHECK.md`, `pilot0/GPU_CHECK2.md` | rule as in the spec column; verdict at the top of `pilot0/GPU_CHECK2.md` | no blind read; the head's recount is in `pilot0/GPU_CHECK2.md` | none | none | J8 at most 1.164, about $2.85 (`OPS/2026-11-05_next_j8_loss_check2_b2_run.md`); the first pilot-0 GPU check at most 1.21, no dollar figure (`pilot0/GPU_CHECK.md`) | 2026-11-05 16:05 CST (line 48) |
| **J10/N22**: does OLMo-3 7B use an intermediate value of (a + b)·c + d as a variable that a Jacobian-lens edit can swap, and from which checkpoint? | NO-GO at T_15 | 7 of 1,200 items answered without chain of thought (freeze half 2, test half 5) against the 100 the go/no-go requires (`n22/verdict_n22_t15.json` 15.n_answered; JOURNAL line 97) | `07_PICKS/N22_SPEC.md` | none; `n22/JOURNAL.md` | `n22/thresholds.json`; `n22/verdict_n22_t15.json` | none; the head's decision is the journal entry | none | none | 0.622–0.627; $1.39–1.40 (`OPS/2026-11-06_next_n22_score.md`) | 2026-11-06 12:05 CST (line 97) |

## (c) Planning documents

| file | role | status |
|---|---|---|
| `00_ASSETS.md` | inventory of checkpoints, S3 harvests, banks and harnesses, 2026-11-04 | reference; corrected by its own section "Corrections, 2026-11-05" (line 300); its §6 gap 7 is superseded by `00b_DATA_ORDER.md` |
| `00b_DATA_ORDER.md` | how the OLMo-3 7B stage-1 batches are rebuilt from public files | reference; its open link (byte identity of the files) was tested on GPU by `pilot0/GPU_CHECK.md` and J8 (`pilot0/GPU_CHECK2.md`, closed) |
| `01_JLENS.md` | the Jacobian lens paper (arXiv 2607.15495) read in full for design | reference; the study it fed (J10/N22) ended NO-GO |
| `02_FRAMES.md` | the stance taken from the essay and the six-second test for a result | historical (brainstorm stage 2) |
| `03_IDEAS.md` | idea cards N1–N28, each with its refuting observation and first measurement | historical; the card status tags were not updated after the picks (26 DRAFT, 2 SET ASIDE); the picks are in `05_SCORING.md`, `06_CRITIC.md` Part 6, `08_SUMMARY.md` and `09_PLAN.md` |
| `04_LITERATURE.md` | already-done audit of cards N1–N13 under the field's terms | reference, searches dated 2026-11-05 |
| `04_PRACTICAL.md` | what capability engineers optimize and where a neuron view can move their numbers | reference; marked DRAFT in its header; its second intersection became N13 |
| `04_RECENT.md` | July–November 2026 recency and competitor scan | reference, searches through 2026-11-04 |
| `04_WDD.md` | the wdd-descendants program read in full | reference |
| `05_SCORING.md` | rubric, provisional scores, programs, picks | superseded for the order by `09_PLAN.md`; it carries its own revisions after the critic (line 95) and the verification (line 145) |
| `06_CRITIC.md` | adversarial read of the picks by a different model; Part 0 on the birth instrument; added M1, M2, M3 | reference; Part 0 was rechecked by `06b_VERIFY_BIRTHS.md` and `10_RECOUNT2.md`, which put the chance count at 22 to 28, not 62 (JOURNAL 2026-11-05 09:30 CST, line 26) |
| `06b_VERIFY_BIRTHS.md` (scripts in `06b_verify_births/`) | independent recomputation of the critic's birth finding | closed; carried forward by `10_RECOUNT2.md` |
| `07_PICKS/` | the pre-registrations: eight specs, eight offline gates, `M1_ASSETS.md`, `N13_MATH_TASK.md` | current; each spec carries its dated amendments and the frozen rules its verdict used |
| `08_SUMMARY.md` | one-page summary of the brainstorm and its picks | superseded by `09_PLAN.md` for the order and by `20_ABSTRACT.md` for the results |
| `09_PLAN.md` | job list J0–J14 with gates, costs and the head's decisions | partly superseded: the GPU-hour ceilings were removed by the user (JOURNAL 2026-11-05 21:45 CST, line 68); the order as run is in JOURNAL; J13 (robot gate) was never run; J14 became `20_ABSTRACT.md`; its line 517 is still the frozen J8 rule |
| `10_RECOUNT2.md` (scripts in `10_recount2/`) | third recount of the already-clean birth dates and of b | closed; the retag rests on it |
| `10_RETAG.md` | retag of the claim "the main job is the oldest" | closed; the record is `studies/already-clean/results/RESULTS.md` and `studies/neurons/CLAIMS.md` row D1-b (JOURNAL 2026-11-05 10:12 CST, line 29) |
| `11_HOUSEKEEPING.md` | housekeeping after the retag: figures, tests, dead worktrees | current; the head is updating it with the branch cleanup (read the copy on `clean-neurons`) |
| `12_SHAPING_NOTE.md` | the Goodfire shaping preprint read in full | reference; its §4 item 7 (block-window transplant arms) was retired after N13 R1's flat block profile (JOURNAL 2026-11-05 23:45 CST, line 73); item 8 is not scheduled (line 53) |

After the numbered files: `20_ABSTRACT.md` (FINAL), `20_ABSTRACT_CHECK.md` and `abscheck/` (the reader's check scripts and outputs).

## (d) Method rules learned here

Each rule names the incident that taught it and the file where it is written.

1. **Compute first-order predictions in fp32, by forward mode where the backward does not fit, and report the
   finite-difference check beside the zero ablation. Read small loss differences in fp32 with the identity pair's noise
   floor registered.** Incident: N12's G-VAL with a bf16 loss read Spearman 0.100 (freeze) and 0.043 (test)
   (`n12/verdict_n12.json` labels.H_signal); N2V read the same predictions against fp32 ablation at 0.960 [0.915, 0.990]
   (`n2v/verdict_n2v.json` test.window). Written: `n2v/READ_head.md` lines 51–53; JOURNAL 2026-11-06 08:15 CST (line 88).
2. **A gate that compares two computation paths (devices, batch shapes, kernels, summation orders) gets a tolerance
   frozen from a control pair measured on those paths, or it is reported and not gated. Keep a tensor hash check for the
   "one bf16 step" failing input, which no tolerance sees.** Incidents: M2's cross-device G-EX stopped three runs
   (JOURNAL lines 61, 66, 71; amendments A1–A3); N6's G-POLE (line 91) and G-ZN, G-CACHE (line 93); N13's arm-builder
   check (A29, line 73) and the A30 exact float equality of two sums in different orders (A31, line 104); N2V's
   t14_G_CAUSAL, where row alignment changed the log-softmax summation order (line 117). Written: JOURNAL 2026-11-06
   09:40 CST (line 91, "Rule added for every later spec"); `HANDOFF_2026-11-07.md` §2 and §7.
3. **The offline gate reads every banked control file and evaluates both sides of every gate on it before launch; a
   registered failing input must have power where its gate is applied.** Incident: M2's first five attempts stopped on
   defects the banks could have shown (cross-device agreement, degenerate poles, the silu-to-gelu_tanh failing input
   passing at T_6). Written: JOURNAL 2026-11-06 00:50 CST (line 74); `m2/READ_head.md` §Gates and limitations.
4. **Gradient passes use deterministic algorithms and the math attention kernel.** Incident: N12 score-01, where 21.3 %
   of 7.4 million gradient entries were equal between two passes and 4 of 98 units' u_S flipped. Written: JOURNAL
   2026-11-06 04:45 CST (line 80); `HANDOFF_2026-11-07.md` §2.
5. **Match a positive control on its realized effect, not on its weight size.** Incident: N6's G-PC inverted; the random
   edit's realized residual change was 1.3–2.5 times the column edit's in all 8 control layers, while the freeze-pair
   edits were matched (medians 60.8 and 60.7). Written as a finding, not yet as a spec rule: JOURNAL 2026-11-08 09:15 CST
   (line 153, "the control is matched on weight size, not on realized effect"); `n6/STATE.md` §What L2a measured.
6. **Assert an edit's realized energy after bf16 rounding.** Incident: N13's energy-matched random arms realized only
   0.70–0.82 of their intended MLP change energy (A30, stochastic rounding). Written: JOURNAL 2026-11-06 05:00 CST
   (line 81) and the rule list in 08:15 CST (line 88).
7. **The pre-check starts the engine, measures the rate the long step will see, and parks the job on what it measured.**
   Incidents: N13 attempt 6 ran at batch size 1 and its projection was off by more than ten times (JOURNAL line 44, "The
   pre-check has to decide"); attempt 7's arm engine died on a missing nvcc after the import check passed (line 55, "the
   first engine start of each venv belongs in the pre-check"); the opening-steps rate overstated the sustained rate 9–11
   times (line 67, amendment A22).
8. **Pay for the decisive cheap gate first and buy the expensive half only after it passes.** Incidents: N6 split L2
   into L2a and L2b so the power gate decided first (JOURNAL line 95); the gate failed and L2b's 19 GPU-h were not spent
   (line 153). J10's accuracy go/no-go ended the study at T_15 for $1.40 (line 97).
9. **Birth claims are compared against a structured control from the same rule: random same-size groups of a pole's own
   contexts dated alike, matched on the number of distinct tokens.** Incidents: the critic's random regrouping
   (`06_CRITIC.md` Part 0; `10_RECOUNT2.md`; retag in `10_RETAG.md`); M2's twins, added by the head as "a structured
   control the plan lacked" (JOURNAL line 60); the M2 reader found the twin gap on repeated-token births (line 78), and
   N12's distinct-token twins closed it (line 84). The general form is the repo's `CLAUDE.md`, Instruments rule 3.
10. **The blind-reader protocol.** A fresh Opus reader works in a worktree made from a tag whose commit subject names no
    label (or is told not to run `git log` before phase 1), commits its own numbers, then opens the verdict. Incident:
    blindness was broken by commit subjects in the M2 and N12 reads (disclosed in `m2/READ_m2.md`, `n12/READ_n12.md`).
    Written: JOURNAL 2026-11-06 08:10 CST (line 87); `HANDOFF_2026-11-07.md` §2; N13's blind tag `n13-s4-complete`
    (line 144). What each read caught beyond the verdict:
    - M1 (`m1/READ_m1.md`): 131 of 131 deterministic fields equal; the contrast comes mostly from G moving away from code
      and math. The head's read found the spec's §1.2 outcome sentence wrong for these data (JOURNAL line 51).
    - M2 (`m2/READ_m2.md`): the twin gap sits on repeated-token births (78 of 149 real births below a distinct-token share
      of 0.67).
    - N12 (`n12/READ_n12.md`, `n12/READ_head.md` addendum): OWN_RISE's sign is close to forced by the dating rule; G-VAL
      measured the resolution of a bf16 loss, not the gradient's validity; this led to N2V.
    - N2V (`n2v/READ_n2v.md`): the agreement is within units (a within-unit permutation leaves ρ 0.286); a defect in the
      reported field `sum_abs_h`.
    - N1 (`n1/READ_n1.md`): removing the two most negative births makes the label AT; carriers' max |activation| is 1.39
      times their twins'.
    - N13 (`n13/READ_n13.md`): DARE's realized energy is 33 to 1,000 times the full RL change's; the GSM8K labels top −
      rndE and top − rndT are fragile; the figure lacked the m_k labels (regenerated, JOURNAL line 148).
    - The abstract reader (`20_ABSTRACT_CHECK.md`): 4 mismatches in 175 numbers of draft 3, all corrected; a fourth pass
      found 31 of 31 equal (JOURNAL line 155).
11. **Journal stamps are read from the clock or taken from the commit time, never extrapolated.** Incident: six stamps
    written 0:05–1:27 ahead of the clock. Written: JOURNAL 2026-11-08 06:05 CST (line 151).
12. **Size a job's hard kill on the measured download rate.** Incidents: J8's Hugging Face downloads at 33–42 MB/s
    (JOURNAL line 34); N1 weights-02 and weights-03 slowed by hub downloads, with the S3 weight cache as the fix
    (line 107; `HANDOFF_2026-11-07.md` §3).

## (e) Cost

GPU-hours and dollars as each operations note states them. Where a note gives a range, the range is kept. "At most" and
"about" are the notes' words.

| study | GPU-h | dollars | file |
|---|---|---|---|
| pilot 0, first GPU loss check | at most 1.21 | none stated | `pilot0/GPU_CHECK.md` line 137 (no operations note) |
| J8, second data-order check | at most 1.164 (b1 0.851, b2 at most 0.313) | about $2.85 | `OPS/2026-11-05_next_j8_loss_check2_b2_run.md` |
| M1 | about 10.3 (K0 about 0.27, mix-a-01 about 6.0, mix-b-01 about 4.0) | about $23 | `OPS/2026-11-05_next_m1_harvests.md` line 113 |
| M2, six attempts | 3.324–3.467 | $8.53–8.91 | `OPS/2026-11-05_next_m2_score.md` line 470 |
| N12, two attempts | 2.483–2.520 | $5.56–5.64 | `OPS/2026-11-06_next_n12_score.md` line 149 |
| J10/N22 | 0.622–0.627 | $1.39–1.40 | `OPS/2026-11-06_next_n22_score.md` line 109 |
| N13 R0 and S2, attempts 1–9 | 7.062 | about $18.84 | `OPS/2026-11-05_next_n13_r0.md` line 375 |
| N13 S3, four launches | 18.982 | $42.51 | `OPS/2026-11-05_next_n13_r0.md` line 525; dollars in `OPS/2026-11-06_next_n13_s4.md` line 263 |
| N13 S4 | 76.511 | $187.81 | `OPS/2026-11-06_next_n13_s4.md` |
| N13 in all | 102.555 | about $249.16 (sum of the three N13 rows, computed here) | `OPS/2026-11-06_next_n13_s4.md` line 272 for the GPU-h |
| N2V, two attempts | at most 6.59 | $14.75 | `OPS/2026-11-07_next_n2v.md` line 73 |
| N1 | 0 (CPU: at most 4.68 instance-h) | about $2.48 | `OPS/2026-11-06_next_n1.md` line 93 |
| N6, three launches | 19.581–19.626 | $43.86–43.96 | `OPS/2026-11-07_next_n6.md` line 88 |
| **total, studies** | **147.83–148.06** | **$351.58–352.15**, pilot 0 not included | sum of the rows above, computed here; the upper bounds are used where a note says "at most" |
| N13 Vast lane, stopped before any run | not summed in the note | about $2.28 once ($0.007 Vast storage, about $0.67 EC2 builder, about $1.6 EC2 preflight), plus about $4.2 per month while the ECR image `shackleton:n13-v1` is kept | `infra/vast/operations/2026-11-05_n13_lane.md` §6b (on `clean-neurons` since merge 01ddd5ffe) |

With the Vast lane the one-off total is about $353.86–354.43. Prices used by the notes: g6e.2xlarge $2.24 per hour,
g6e.4xlarge $3.00424 per hour (`09_PLAN.md` line 13).

## (f) Branches and worktrees

The program's work now lives on branch `clean-neurons`, the head is cleaning up the merged branches and worktrees in
parallel, and the record of that cleanup is `studies/next/11_HOUSEKEEPING.md`.

## (g) Stale text found

Searched: every STATE.md and README under `studies/next/` (`README.md`, `pilot0/README.md`, `n1/STATE.md`,
`n2v/STATE.md`, `n13/STATE.md`, `n6/STATE.md`) for "running", "pending", "in progress", "awaits", "not yet", "to be
run", "L2b". Line numbers are those of this commit. No verdict text was changed. Where a note was added, it is one dated
line appended under the stale line; nothing was deleted.

Hits that contradict the closed state:

1. `README.md:7`: "Status: **in progress**, 2026-11-05 01:40 CST ... are running." Every study is closed (JOURNAL line
   155). Not annotated: this task allowed only the two pointer lines in the README (lines 3–4 now point to this file and
   to `HANDOFF.md`).
2. `n13/STATE.md:3`: "(S4 running)" in the top paragraph; the latest update below it (lines 15–20) still ends with "Next
   (the head): the independent reader". N13 closed (JOURNAL line 146). Annotated at line 6.
3. `n13/STATE.md:163–165`: the cost table, dated 18:00Z on 2026-11-07, lists `s4armsA1-01`, `s4armsB1-02` and
   `s4armsB2-02` as running. All three ended, the last at 10:47:04Z on 2026-11-08 (`OPS/2026-11-06_next_n13_s4.md`).
   Covered by the note at line 6.
4. `n13/STATE.md:90`: "the running steps A1 `top_k1_B`, B1-02 `rndE_k1`, B2-02 `rndE_k3` ... nothing else of them is
   banked". All three were banked later (JOURNAL lines 115, 117, 118). Covered by the note at line 6.

Found outside the search terms while reading, and annotated:

5. `n1/STATE.md:12–14`: "Open: spec §7 step 5, the fresh Opus reader ... the head spawns it." The read is done and N1
   closed (JOURNAL line 129). Annotated at line 15.
6. `pilot0/README.md:209` "## GPU loss check (not run)" and `pilot0/README.md:262` "`loss_check.py`: the GPU check
   above. Not run." The check ran (`pilot0/GPU_CHECK.md`, `pilot0/GPU_CHECK2.md`). Annotated at line 210.

Hits that do not contradict the closed state:

- `n1/STATE.md:23` "### weights-03 (running at 17:55Z)" and `:33` "EC2 running": dated snapshots inside the 2026-11-07
  handoff; the section above them says weights-03 ended.
- `n13/STATE.md:16, 45, 49–50, 101, 176, 207`: "0 pending", dated log entries, the poller's own text, "running 0".
- `n6/STATE.md`, 24 lines with "L2b" (4, 7, 52, 59, 74–155): every one says L2b was not launched or keeps its unused
  launch commands under the heading "(the head's; not run)".
- `n2v/STATE.md:138`: "depending" matches "pending"; not a status.
