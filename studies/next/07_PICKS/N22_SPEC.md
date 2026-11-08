# N22 with N25: does OLMo-3 7B use an intermediate value as a variable, and from which checkpoint? Pre-registration

**Answer.** Three launches on one L40S each answer the card's question, projected at 7.46–12.29 GPU-h in all
($16.71–27.53 at $2.24 per hour; `studies/next/n22/projection.json`), against the card's 8–13. L1 (1.05–1.76 GPU-h,
hard kill 138 min) does everything at the final stage-1 checkpoint T_15 and decides the go/no-go. L2 (1.01–1.58,
kill 124 min) is an accuracy scan of the other 15 points. L3 (5.39–8.94, run as two launches of about half each) fits
the lens at up to four earlier points chosen from L2 by a frozen rule, plus step 0 and one early point for N25.
The primary readout is a 2 × 2 interaction in fp32 log-probabilities: put the same value v into the slot of the
first intermediate (a + b) or of the second ((a + b) × c), at the same layers, and score the exact targets
f1(v) = v·c + d and f2(v) = v + d. It must exceed a matched-norm random-direction edit and a shuffle of the targets;
its ceiling is the same interaction when the input itself is changed. The ten raw examples are tokenized and checked
at zero GPU (`studies/next/n22/samples/raw_examples_zero_gpu.md`); every model output on them is PENDING and is the
first thing L1 writes. The weight-free dry run printed `N22_DRYRUN_OK` (`N22_OFFLINE_GATE.md`). The five gates whose
quantity a bank holds (the known number, the revision, the config, the corpus, the tokenizer) pass on their pass
side and fail on their registered failing input at all 16 points. The other gates have no bank and were shown on a
tiny model only.

Written 2026-11-06 by an Opus worker in worktree `interp_research-n22`, branch `n22` (from `clean-neurons` at
`0f42e91d2`). No model was run, no weights were downloaded, nothing was launched. Network reads: S3, read-only (the 16
grid harvests' `controls.json` and `manifest.json`, the T_14 L40S re-harvest, M1's T_15 known-number run, the corpus
manifest); Hugging Face (the refs of `allenai/Olmo-3-1025-7B`, the `config.json` of five branches); five web searches
(§1.5).

---

## 0. Status and provenance

| field | value |
|---|---|
| status | FROZEN at freeze 1. The head's review (2026-11-06 15:20Z) accepted the draft as written, nothing changed. The dry run was rerun on the tagged code 15:17:53Z–15:22:46Z: `N22_DRYRUN_OK`, 100 checks, 0 failed (`studies/next/n22/offline_gate/dryrun_freeze1.log`); `dryrun_summary.json` equal as JSON to the draft's; four CHECK lines differ only in values drawn without a seed or in elapsed seconds, all PASS. |
| freeze 1 | an annotated tag `n22-freeze-1` holding this spec, `infra/pg/n22.py`, `infra/aws_pg/jobs/n22_score.py`, `infra/aws_pg/jobs/n22_dryrun.py`, the step files, `studies/next/n22/items.json` (sha256 `dbf0b225…`, registered as `n22_score.ITEMS_SHA`), `studies/next/n22/offline_gate/gate_sides.py`, `studies/next/n22/samples/raw_examples.py`. **Made**: tag `n22-freeze-1` (tag object `a877b3390`) on commit `76fde0751db99a00fd0ad2a96142dab19ee93695`, 2026-11-06T15:23:29Z. |
| freeze 2 | after L1, from the freeze half only (§6.1): `studies/next/n22/thresholds.json`, tag `n22-freeze-2`. **Made**: tag `n22-freeze-2` (tag object `8da8b5881`) on commit `a083b357fe550793904e85347365bd2df3dd2144`, 2026-11-06T18:00:44Z. Go/no-go **NO-GO**: 7 of 1,200 answered at T_15 (needs 100), the interchange control not run (needs 50 answered per half). The T_15 verdict ran once at 18:00:50Z (`studies/next/n22/verdict_n22_t15.json`). L2 and L3 are not launched; the study ends here (§3.3). |
| freeze 3 | after L3, from the freeze halves of the L3 points (§6.1): `studies/next/n22/thresholds_stage2.json`, tag `n22-freeze-3`. |
| parents | `studies/next/09_PLAN.md` card J10; `studies/next/03_IDEAS.md` N22, N25; `studies/next/01_JLENS.md` §1, §5, §7 Seeds 1, 5, 6; `studies/next/06_CRITIC.md` Part 3; `studies/next/05_SCORING.md` rows N22, N25 and the revision; `studies/next/12_SHAPING_NOTE.md` §4 items 4–6 |
| method | repo `CLAUDE.md`; the `research-spec` skill; the format of `N12_SPEC.md` and `M1_SPEC.md`; M2's amendment A5 for the offline gate; the six lessons of the head's brief (§0.2) |
| vocabulary | `studies/next/README.md`. A run whose gate fails is labelled **DOES NOT COUNT**. |

### 0.1 The brief's requirements and where each sits

| requirement (head's brief, 2026-11-06) | where |
|---|---|
| the question in the card's words | §1.1 |
| accuracy gate at each checkpoint without chain of thought (about 1,000 single-token answers, chance about 0.1 %) | §3.1 step 1; L2 runs it at all 15 other points |
| the lens-free interchange positive control at the final checkpoint first | §5.1; it is in L1, and it decides the go/no-go |
| slots compared at matched depth | §2.3, §3.2 (LENS_M on the matched band) |
| the exact-target 2 × 2 score | §2.2 (the interaction), §5.3 (the shuffle of targets, the near values) |
| N25: a step-0 lens and the projection of the massive-activation direction | §3.4, §7 (N25), §5.5 |
| a split-half lens for thresholds | §5.4 (delta_read), §6 |
| the known number: corpus loss at the loaded checkpoint (`controls.json`) | §5.6 (K0) |
| go/no-go: past T_15 only if at least 100 expressions and the interchange control above both floors | §3.3 |
| hypotheses with refuting observations | §1.3 |
| thresholds frozen on control data; floors (shuffle, random-direction, step-0 lens); ceiling; does-not-count branches | §5, §6 |
| the identity pair | §5.4 |
| the ten raw examples first | §10; `studies/next/n22/samples/` |
| budget with a park order and a stop rule; which steps run at which checkpoints and what each costs | §9.6 |
| the three dimensions | §11 |
| lesson 1: small loss or logit differences read in fp32 with an identity pair showing the noise floor | §2.4; I-FWD, I-EDIT, G-IC-ID (§5.4) |
| lesson 2: any gradient pass deterministic, math attention kernel, identity pair a gate | §2.5; I-GRAD |
| lesson 3: cross-device bank comparisons at T_7–T_14 reported, not gated | §5.6 (K0 tiers) |
| lesson 4: a failing input must have power where it is applied, shown from the banked controls at every point | `N22_OFFLINE_GATE.md` (bank-side gates at 16 points; K0's smallest margin 3.4 times) |
| lesson 5: every reader of a failing linear readout runs the nonlinear probe beside it | §5.7 |
| lesson 6: read with one direction, edit along the other | §3.2 (DOM_M; the interchange), §2.6 |

### 0.2 Facts read for this draft

| fact | value | source |
|---|---|---|
| model, points | `allenai/Olmo-3-1025-7B`; T_0 … T_15 = stage-1 steps 0 … 1,413,814; revisions as `m2.POINTS`; banks T_0–T_6 and T_15 on L40S, T_7–T_14 on RTX PRO 5000 Blackwell | `infra/pg/m2.py` `POINTS`; the 16 banked `manifest.json` (S3) |
| architecture | 32 blocks, hidden 4,096, MLP 11,008, 32 heads, no grouped-query attention, untied embeddings, vocabulary 100,278; each block normalizes its attention and MLP outputs (post-norm: `h + norm(attn(h))`, then `h + norm(mlp(h))`) | `config.json`; `transformers` 5.9.0 `modeling_olmo3.py` `Olmo3DecoderLayer.forward` |
| tokenizer | the 1,000 strings "0"–"999" are single tokens; no 4-digit string is; a space before a digit is its own token `Ġ` (id 220); `" = "` ends in `Ġ=` then `Ġ`; `tokenizer.json` of `main` and of `stage1-step60000` are identical (sha256 `73fd5254…`); its vocabulary sha256 `2f3c015e…` equals the corpus record's | scratchpad `jlens_in/olmo_tok/tokenizer.json`, `pilot0_in/tokenizer.json`; `infra.pg.matrix_models.vocab_sha256`; S3 corpus manifest |
| known number per point | `nll_corpus_first_120k.mean_nll`: T_0 12.2233822265625 … T_15 2.675372037760417 (all 16 in `n22.K0_BANK`) | the 16 `controls.json` (S3, read 2026-11-06) |
| known-number control pairs | T_15: grid harvest and M1's K0 run, both L40S, equal; `stage1-step476838`: M1 start_a and start_b, both L40S, equal (2.781479736328125); T_14: Blackwell 2.714761865234375 against L40S 2.714700182088216, difference 6.1683e-05 | S3 `grid/1/out/t15…/controls.json`, `next_m1/k0-01/out/k0/controls.json`, `grid/1b` and `grid/1c` T_14; laptop `studies/next/m1/banks/start_{a,b}/controls.json` |
| measured rates | L40S forward 0.2156 s per 2,049-token window at batch 8 (M2 score-06); gradient pass 0.5832 s per 2,049-token window, deterministic math kernel (N12 score-02); fixed cost per point 158.6 s (N12); download 14.6 GB at 76.0 MB/s (M1 K0) to 178–321 MB/s (M2) | `studies/next/m2/banks/score-06/m2/metrics.json`; `studies/next/n12/banks/score-02/n12/metrics.json` → `measured`; `studies/next/m1/JOURNAL.md` |
| the lens code | vendored `jlens` @ 581d3986: per prompt one forward and ceil(d / dim_batch) activation-only backward passes; a one-hot cotangent at every valid target position (positions 16 … n−2); the mean over source positions; `target_layer` default the last block | `infra/pg/vendor/jlens/fitting.py`, `VENDOR.md` |
| the paper's arithmetic protocol | prompt "calc: ( 4 + 17 ) * 2 + 7 =" (Opus 4.5) and "(4+17)*2+7=" (Sonnet 4.5); 21 reaches rank 1 near layer 71, 42 about 8 layers later, 49 in the final layers; mean-difference patching "add the difference between these to the residual stream at a single layer L", effective at the same layers; coordinate swaps "at every token position over the upper half of the network's layers"; 42→63 raises 70, 21→63 raises 133 | the paper's text, §3.3 and §A.24.1–A.24.2 (scratchpad `jlens_text/paper.txt` lines 853–868, 4301–4420; read for these claims) |
| lessons of this program | the six rules of the head's brief, from N12 (G-VAL measured bf16 rounding of a summed loss, `studies/next/n12/READ_n12.md` §6), N12 A1 (the default backward was not deterministic), M2 A3 and A5 | the brief; the files named |

### 0.3 Facts to verify at run time

| # | fact | verified by | rule if it differs |
|---|---|---|---|
| 1 | the AMI has torch 2.12.0+cu130, transformers 5.9.0 | G-SAME | the job stops |
| 2 | each branch's tokenizer encodes the registered prompts | G-TOK at every point | the job stops |
| 3 | the L40S forward reproduces the banked corpus NLL (exactly at L40S-banked points; within 0.01 elsewhere) | K0 at every point | the job stops |
| 4 | the lens's backward is bit-deterministic with the math kernel | I-GRAD at every lens point | the job stops |
| 5 | an fp32 forward of the same prompts in another batch order agrees within 1e-4 nats | I-FWD at every full point | the job stops |
| 6 | 7.3 B parameters in fp32 (29.2 GB) plus the lens vectors (0.5 GB) fit in 48 GB | the first full point | out of memory is not caught; the job stops |
| 7 | the lens rate (78 s per prompt estimated, 120 s high) and the fp32 forward rate (0.018 s, 0.040 s high) | `measured` in `metrics.json` after the first point | the stop rule (§9.6) |
| 8 | OLMo-3 answers at least 100 of 1,200 pool items at T_15 without chain of thought | the accuracy gate | the go/no-go (§3.3) |

### 0.4 Already-done audit

| cell | prior measurement | path | what it gives N22 |
|---|---|---|---|
| a lens of OLMo-3 7B | none fitted in this repo. Third-party lenses exist for `main` (Neuronpedia, final-block target, 568 prompts) and for post-training variants; none for the 16 stage-1 points | `01_JLENS.md` §7; `04_RECENT.md` §8 | our checkpoints need their own fits |
| lenses across OLMo-3 pretraining | jeeva2812 (layer 20 only, 8 stage-1 steps, geometry); senku21x (OLMo-3 32B, six stage-1 points, 10 prompts each) | `04_RECENT.md` §8 | the timing with a step-0 floor and split-half noise is not done; the 7B is not done |
| arithmetic on OLMo-3 | none in this repo | `git ls-files` grep for arithmetic studies | — |
| the lens in this repo | `infra/pg/stagediff_lens.py` fits the vendored lens on Qwen3 and Molmo2 (STAGEDIFF) | that file | the import path of the vendored code; nothing about OLMo-3 |
| gradient determinism on the L40S | default sdpa backward not bit-deterministic; deterministic algorithms with the math kernel bit-identical | `studies/next/07_PICKS/N12_SPEC.md` §14 A1; `n12/banks/score-02/n12/metrics.json` | the lens fit runs under `n12_score.grad_mode` |
| bf16 loss differences | a summed-window bf16 loss moves 0.2–0.9 nats under any change at one position | `studies/next/n12/READ_n12.md` §6 | every edit readout here is an fp32 log-probability at one position |
| the known number at T_15 | reproduced exactly by M1 on an L40S | `studies/next/m1/JOURNAL.md` 16:31:38Z | K0's same-device tier is exact |
| the open-model swaps | Qwen 3.6 27B: arithmetic "failed to replicate"; answer swaps dominate intermediate swaps; J-lens intermediate swaps flip top-1 on 6.3–11.1 % of items (Qwen, Gemma) | `01_JLENS.md` §5.2 (Nanda commentary; Zeisler) | the prior for H_slot is low |

### 0.5 Objects, defined once

- **Item.** An expression (a + b) × c + d with s1 = a + b ∈ [11, 60], c ∈ [2, 9], d ∈ [1, 99], s2 = s1·c, y = s2 + d
  ≤ 639. The seven numbers a, b, c, d, s1, s2, y are distinct, and none of s1, s2, y appears in the three solved
  lines. The **pool** is 1,200 items (`n22.make_pool`, seed 20261106). Items with even index form the **freeze half**
  (600), odd the **test half** (600).
- **Prompt.** BOS, three fixed solved lines "(3 + 5) * 4 + 2 = 34", "(12 + 7) * 3 + 9 = 66", "(25 + 16) * 2 + 8 = 90",
  then the query "(a + b) * c + d = ". 63 tokens. The query occupies positions 49–62; a, b, c, d sit at 50, 53, 57, 60;
  the answer is predicted at position 62 (the token `Ġ` after `Ġ=`). No chain of thought.
- **Answered.** The greedy top-1 at position 62 is the token of y. Chance is about 1 in 1,000 if the model spreads its
  mass over number tokens.
- **v, f1, f2.** Five values per item, v ∈ [11, 99], drawn so that f1(v) = v·c + d ≤ 999, f2(v) = v + d, and v, f1(v),
  f2(v) differ from each other, from every number of the item and from the solved lines' numbers, and v = s1′·c′ for
  some c′ ∈ [2, 9] (so a counterfactual with product v exists).
- **Partner.** For each item, the next item of its half (cyclic) with another (c, d) whose targets for the five v are
  single tokens and differ from the item's. **Shuffled targets** f1x(v) = v·c_p + d_p and f2x(v) = v + d_p.
- **Lens.** J_ℓ for source block ℓ ∈ 0…29 into target block 30 (the penultimate block), the vendored estimator on 24
  prompts of BOS + 127 corpus tokens (seeded windows of the grid corpus). **A** is the mean over prompts 0–11, **B**
  over 12–23, the **full** lens (A + B)/2. Readout: lens_ℓ(h) = lm_head(norm(J_ℓ h)). The **logit lens** is J = I. The
  **lens vector** of token w at block ℓ is v_{w,ℓ} = J_ℓᵀ (g ⊙ u_w), with u_w the row of the unembedding and g the
  final norm's gain, the exact first-order direction of the readout's logit (the paper does not say whether it folds
  g; `01_JLENS.md` §1.2).
- **Readout hit.** s_k is in the top 10 of the full vocabulary of lens_ℓ at position 62. **Net readout** R_k(ℓ) = share
  of items with a hit for s_k minus the share with a hit for the partner's s_k.
- **Bands** (`n22.band_rule`, freeze half, answered items, full lens). ℓ1 = argmax R_1, ℓ2 = argmax R_2, ℓm = argmax
  min(R_1, R_2) (ties: the lower block); the **matched band** B_m = {ℓm − 1, ℓm, ℓm + 1}; the **upper half** U =
  blocks 16–29 (the paper's "upper half of the network's layers").
- **Edits** (the arms; all at fp32):
  - **LENS_U**: the paper's coordinate swap s_k ↔ v at every query position, clamped at every block of U.
    With V = [v_{s_k}, v_v], c = V⁺h, h′ = h + V(σ(c) − c), σ exchanging the two coordinates.
  - **LENS_M**: the same swap on the matched band B_m only. **The primary arm.**
  - **DOM_M**: the paper's mean-difference patch (§A.24.1) at position 62 of block ℓm: h′ = h + μ_ℓm^(k)[v] −
    μ_ℓm^(k)[s_k], with μ^(k)[x] the mean residual over 16 **class prompts** whose slot-k intermediate is x, every other
    operand random (`n22.class_prompts`).
  - **ANS_U, ANS_M**: the answer substitution y ↔ f_k(v), a coordinate swap as LENS_U, LENS_M.
  - **RAND_U, RAND_M, RAND_DOM**: the edit of LENS_U, LENS_M, DOM_M with its norm kept at each position and block and
    its direction replaced by an isotropic random unit vector (seeded).
- **Interaction.** For one item, one v and one arm X with its slot-1 edit X1 and slot-2 edit X2:
  **I = [lp(f1 | X1) − lp(f1 | X2)] + [lp(f2 | X2) − lp(f2 | X1)]**, lp the fp32 log-probability at position 62. The
  log-partitions cancel, so I equals the same expression in logits. I > 0: the target of the slot that was edited rose
  more than the other slot's target. The **shuffled interaction** I_x uses f1x, f2x. Per item: the mean over its v.
- **Interchange (IC).** For v (the first two of the item's five), the counterfactual prompt of slot 1 (a′ + b′ = v,
  the item's c and d; its answer is f1(v)) and of slot 2 ((a″ + b″)·c″ = v, the item's d; its answer f2(v)). The
  interchange at site s (the embedding output, or the output of block 0, 4, …, 28) writes the counterfactual's residual
  into the item's forward at the slot's operand positions (slot 1: a, b; slot 2: a, b, c). At the embedding it is the
  counterfactual prompt itself. **RAND_IC**: h + ‖h_cf − h‖·r at the same positions and site.

---

## 1. Question, decision, hypotheses

### 1.1 The question

In the card's words: does OLMo-3 use an intermediate value as a variable whose next operation depends on its
position, and from which checkpoint? Concretely: when the same value v is written into the representation of the
first intermediate (a + b) or of the second ((a + b) × c), does the model then output v·c + d or v + d respectively,
at the same depth, and from which of the 16 stage-1 checkpoints?

### 1.2 The decision this run makes

| outcome | reading | next |
|---|---|---|
| NO-GO at T_15 (fewer than 100 answered, or the interchange control at or below a floor) | the model cannot be tested on this task without chain of thought, or the harness does not move the answer; the lens test had no chance | stop; L2 and L3 are not spent; report the counts and the control |
| GO; H_slot SLOT_DEPENDENT at T_15 | a variable in the lens coordinates is passed to a slot-dependent operation at one depth | L2, L3; date it (H_order) |
| GO; H_slot NOT_SHOWN, DOM_M or the interchange shows it | the variable exists, outside the lens coordinates (critic 1.3) | L2, L3 with the dating arm DOM_M or LENS_U (§6.3); the lens is not the instrument for it |
| GO; LENS_U SLOT_DEPENDENT, LENS_M NOT_SHOWN | the paper's reading holds only with each slot edited at its own depth; the critic's depth alternative is not excluded | report; date with LENS_U and state the depth caveat |
| GO; nothing above the floors | the interchange moves the answer, the lens and DoM edits do not | report; H_order NOT_RUN; L3 still fits the lenses for N25 and the readout timing |
| N25: the readout or the band statistics move by more than split-half noise when the top-3 massive coordinates are projected out | the lens reads the sink | every N22 readout claim carries that caveat (05_SCORING: "after N25 says the lens is not reading the sink") |
| DOES NOT COUNT | no claim; the failed gate and its value are reported | fix, fresh attempt prefix |

### 1.3 Hypotheses and the observation that refutes each

Labels as `n22.point_verdict` and `n22.order_label` compute them on the test half (§6).

| id | hypothesis | predicts | refuted by |
|---|---|---|---|
| **H_slot** (the card's claim, matched depth) | At T_15 the value in the first-intermediate slot is multiplied by c and added to d, the value in the second slot only added to d, with both slots edited at the same blocks. | LENS_M: I above RAND_M and above the shuffle (both paired lower bounds > 0) | either paired upper bound < 0 (BELOW_FLOOR), or an interval containing 0 (NOT_SHOWN) |
| **H_depth** (critic 1.1, the alternative) | The downstream function depends on the depth of injection, not on the slot: whatever is injected at block ℓ goes through the operations remaining after ℓ. | LENS_U SLOT_DEPENDENT with each intermediate edited where it lives, LENS_M NOT_SHOWN | LENS_M SLOT_DEPENDENT (at matched depth the two slots give different functions) |
| **H_inject** (the boring one) | The edit only writes output tokens: the raised token is v, or the numbers near v and near the targets rise together. | top-1 = v as often as top-1 = target; I at the shuffle (near and other-item targets rise as much) | I above the shuffle (the exact item-specific target rises more than the other item's v·c′ + d′) and the exact-target margin (§5.3) |
| **H_answer** (Nanda's open-model result, `01_JLENS.md` §5.2) | Writing the answer directly does what the intermediate edit does. | I_LENS_M ≤ I_ANS_M at the same band | I_LENS_M − I_ANS_M lower bound > 0 (reported as the qualifier "above answer substitution") |
| **H_outside** (critic 1.3) | The variable exists but outside the lens coordinates. | the interchange fires; LENS_M and LENS_U not above the floors; DOM_M above them | LENS_M or LENS_U SLOT_DEPENDENT |
| **H_order** (the card's dating claim) | Across checkpoints, readout of the intermediates appears before slot dependence. | SUPPORTED: a scored point with readout PRESENT and slot dependence absent, before the first point with slot dependence | REFUTED_SAME (both first appear at the same scored point after the earliest), REFUTED_CONTRARY (slot dependence at a point without readout, or earlier) |
| **H_heuristic** (critic 1.4, Nikankin et al.) | Arithmetic runs on heuristic neurons with no separate variable: no slot-dependence onset distinct from accuracy. | slot dependence never above the floors at any answered point, or appearing exactly with accuracy | slot dependence at a point after accuracy's onset (H_order SUPPORTED with t_s after the first scored point) |
| **H_sink** (N25) | The intermediate readout is carried by the massive-activation coordinates. | READOUT_SINK: the net readout with the top-3 coordinates projected out differs from the full lens's by more than delta_read | READOUT_SURVIVES |
| **H_arch** (N25) | The lens's band statistics are architectural. | ARCHITECTURAL: the step-0 lens within split-half noise of T_15 on a statistic | LEARNED |
| **H_form** (N25) | The band statistics appear between two specific checkpoints, after the massive coordinates form. | FORMS_BETWEEN with the first interval at or after the massive-formation point | SMOOTH or NO_CHANGE; or the interval before the formation point |

What the labels can and cannot show. The interaction cancels everything common to the two edits: the injected v,
the clean distribution, the log-partition. It does not cancel what differs between them: which coordinate is removed
(s1 against s2). The random floor shows the size of interaction an arbitrary edit of the same norm makes; the shuffle
shows how much a target of the same form but another item's c, d rises. Neither floor tells whether the computation
that produced f1(v) is "multiplication": the 2 × 2 says the downstream function differs by slot in the way the
expression requires. The order clause "slot dependence before correct answering" of `03_IDEAS.md` is dropped, as the
critic asks (Part 3 item 2): raising f1(v) exactly needs the multiply and add to be right on v.

### 1.4 The author's prior, logged before any model output exists (2026-11-06)

| quantity | prior | basis |
|---|---|---|
| ≥ 100 of 1,200 answered at T_15 | 0.80 | a 7B base after 5.9 T tokens; two-digit addition, multiplication by one digit and addition without chain of thought; no measurement |
| the interchange control fires (GO) | 0.75 given ≥ 100 | at the embedding it is the counterfactual prompt; it fails only if the model rarely solves the counterfactuals |
| H_slot (LENS_M) SLOT_DEPENDENT | 0.20 | Qwen 3.6 27B: arithmetic swaps "failed to replicate" (Nanda); J-lens intermediate swaps flip 6.3–11.1 % (Zeisler); one curated Claude example |
| LENS_U (the paper's protocol) SLOT_DEPENDENT | 0.25 | as above; depth-specific editing is easier |
| DOM_M SLOT_DEPENDENT (and counts) | 0.35 | the paper's mean-difference patching localized every intermediate; the DoM guard may remove it |
| readout PRESENT for s1 at T_15 | 0.50; for s2 0.35 | the paper's Figure 17; open-model readouts weaker (J++ post: on OLMo 3 32B the logit lens 35.2 % beats the J-lens 32.6 % with filtering) |
| READOUT_SINK | 0.30 | Brauer (final-layer target dominated by one direction; we use the penultimate); jeeva2812's retraction |
| H_order | SUPPORTED 0.10, CENSORED 0.25, REFUTED_SAME 0.10, REFUTED_CONTRARY 0.05, NOT_RUN 0.50 | NOT_RUN whenever no arm is SLOT_DEPENDENT at T_15 |
| N25 H_arch: LEARNED on the CKA block statistic | 0.70 | senku21x: the first layer resembling the logit lens moves 14 → 41 over OLMo-3 32B training |
| I-GRAD bit-identical | 0.90 | N12 A1 on the same AMI |
| K0 exact at T_15 | 0.95 | M1 reproduced it exactly |

### 1.5 Literature (query terms recorded)

Searches of 2026-11-06 (WebSearch): (1) `interchange intervention multi-step arithmetic intermediate result language
model causal abstraction variable`; (2) `implicit multi-hop arithmetic reasoning intermediate value activation
patching "(a+b)" language models layers`; (3) `emergence of arithmetic circuits across pretraining checkpoints Pythia
OLMo mechanism training dynamics`; (4) `"Arithmetic Without Algorithms" bag of heuristics neurons emerge training
checkpoints Nikankin`; (5) `slot-dependent arithmetic Jacobian lens swap intermediate "133" "70" replication open
model OLMo Qwen workspace`. Earlier records: `04_RECENT.md` (N22 phrases: "multi-step arithmetic intermediate value
patching checkpoints", "slot dependence over training", "latent reasoning faithfulness along training trajectories",
"when does the model use intermediate result OLMo"; §8 lenses across checkpoints) and `01_JLENS.md` §6.

- The paper (Gurnee et al., arXiv 2607.15495): read for the protocol in §0.2, from its text. The swap 21→63 raises
  133 and 42→63 raises 70 on Sonnet 4.5 (§A.24.2); the swaps are at every position over the upper half of the layers.
- Nikankin et al., arXiv 2410.21272 ("Arithmetic Without Algorithms", ICLR 2025): abstract only. Its claim used here:
  a bag of heuristics "appears as the main source of arithmetic accuracy early in training". Not read in full.
- Stolfo et al., arXiv 2305.15054 (causal mediation for arithmetic): named by the critic; abstract only, not read by me.
- Matsumoto et al., "Tracing and Manipulating Intermediate Values in Neural Math Problem Solvers"; Kudo et al., arXiv
  2412.01113; arXiv 2509.09650 (three-operand arithmetic, middle-layer information transfer): seen in search results
  only; not read; no claim is taken from them.
- Not found in five searches: an across-training test of slot-dependent arithmetic on an open model, or any
  matched-depth version of the paper's slot test. Five searches are not a hole in the literature.

---

## 2. The measurement and the instrument

### 2.1 In plain words

- **Items.** The 1,200-item pool; at each point only the items that point answers enter the edits (critic Part 3
  item 2), at most the first 200 per half.
- **Model.** `allenai/Olmo-3-1025-7B` at its stage-1 branches, the weights as released (bf16), upcast to fp32 for
  every readout and edit.
- **Quantity.** Per (item, v, arm): the interaction I in nats. Per item: the mean over its five v (two for the
  interchange). Per arm and half: the mean over items with a percentile bootstrap 95 % interval over items (10,000
  resamples, seed 0). Paired differences against each floor by item (the same bootstrap).
- **Pairing.** Every arm and floor of an item uses the same prompt, the same v, the same blocks; the floors' random
  directions are seeded by (point, arm, slot, batch).
- **Readouts.** Per (item, block, lens variant): the rank of s1, s2, y, the partner's s1, s2, y and the four
  operands in the full vocabulary. Per half and block: the net readout share with Wilson intervals.

### 2.2 The instrument, exactly

`infra/aws_pg/jobs/n22_score.py`; its docstring lists the steps of a point. The parts that make the number:

1. `Point.pool_forward`: batches of 64 prompts, `model(input_ids, logits_to_keep=1)` in fp32; `log_softmax` of the
   last position; forward hooks on every block's output record the residual at position 62.
2. `Point.readout_logits(h, J)` = `lm_head(norm(h @ J.T))` with the model's own final norm and LM head, fp32.
3. `edit_rows`: per (arm, slot) and batch, forward hooks on the arm's blocks rewrite the block output:
   `lens_edit` (every query position, `n22.lens_swap_delta`), `dom_edit` (position 62); the random floors keep the
   norm per position and draw the direction from a seeded generator. One record per row:
   `{item, vi, arm, slot, lp: {f1, f2, v, y, s1, s2, f1±1, f2±1, f1x, f2x}, top1, top1_lp}`.
4. `ic_rows`: the counterfactual forward records the residual at every site; the patched forward writes it at the
   slot's operand positions through a mask; RAND_IC adds a random vector of the same norm per position.
5. `n22.item_interactions`, `n22.slot_label`, `n22.point_verdict` turn the records into labels (§6).

### 2.3 Matched depth (critic 1.1)

The paper edits both slots over the same layers (the upper half), but each swap acts where its token's coordinate is
present: 21 around layer 71, 42 around layer 79 (§A.24.1). A value injected earlier passes through more of the
remaining operations, so "f1 from slot 1, f2 from slot 2" also follows from "the remaining operations after the
injection depth". LENS_M edits both slots on one band, B_m, where both intermediates are readable by the frozen
rule; there the depth is the same for both rows of the 2 × 2. If no block has both readable (the matched net readout
at or below delta_read on the freeze half), LENS_M and DOM_M DO NOT COUNT ("no matched layer") and the claim can only
be stated as depending on depth, through LENS_U.

### 2.4 Precision (lesson 1)

Every forward after the lens fit is fp32 with matmul precision "highest" (no TF32). Each readout is one log-probability
at one position, never a summed window loss. The noise floor is shown by three identity pairs (§5.4): I-FWD (the pool
again in reversed batch order), I-EDIT (the swap with σ the identity), G-IC-ID (the patch at the embedding against the
counterfactual's own forward). Each must be within 1e-4 nats; the tiny world gives exactly 0.

### 2.5 The lens fit (lesson 2)

The fit is a gradient pass. It runs under `n12_score.grad_mode`: `torch.use_deterministic_algorithms(True)`,
the sdpa math kernel only, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA starts. I-GRAD recomputes prompt 0's
Jacobian after the fit; the relative Frobenius difference must be at most 1e-6 at every source block (expected 0;
failing input: prompt 1's Jacobian). The fit is in bf16 (the released weights' precision, as the reference code and
the public lenses), accumulated in fp32 by the vendored code.

### 2.6 What the lens can and cannot say; read and edit directions (lesson 6)

- **It can say** whether a token's direction, transported to block 30 by the average Jacobian, is among the top 10 of
  the readout. That is a linear probe per token, the same kind of fact as a logit-lens readout (`01_JLENS.md` §5.2).
- **It cannot say** how the value is represented outside the token directions, nor what computation produced it.
  The paper's own graphs put most of the influence in the non-J-space remainder (§A.24.2).
- **Read with one, edit along the other.** The lens vectors are read directions. The paper's swap writes along them;
  that is the paper's instrument and is kept (LENS_U, LENS_M). Rule 6 asks for the edit along a difference of means:
  DOM_M writes μ[v] − μ[s_k], the paper's own lens-free check (§A.24.1), read at the band the lens located. The
  interchange writes the counterfactual run's own state, the per-pair difference, not a probe direction.
- **A caveat of DOM_M that is a does-not-count branch.** With the other operands drawn at random, the class mean of
  "s2 = v" also moves the answer by v − s2 on average, since y = s2 + d; that alone would raise f2(v) = y + (v − s2).
  So DOM_M DOES NOT COUNT at a band where the answer y is already readable (its net readout at ℓm above delta_read on
  the freeze half). Balancing the answer across classes instead moves d by the same amount; no class construction
  isolates s2 when y = s2 + d.

---

## 3. Design

### 3.1 The pre-checks, in the card's order, and where each runs

| # | pre-check (the card) | what runs | where |
|---|---|---|---|
| 1 | accuracy gate at each checkpoint without chain of thought | the pool forward; answered counts per half; the partner's-answer rate beside it | every point (L1 at T_15, L2 at the other 15, again at L3's points) |
| 2 | the lens-free interchange positive control at the final checkpoint first | IC at 9 sites (freeze half) and at the embedding (test half), RAND_IC; the go/no-go reads the freeze half at the embedding | L1, then every N22 point of L3 |
| 3 | slots compared at matched depth | LENS_M and DOM_M on B_m | every full point |
| 4 | the exact-target 2 × 2 score | the interaction; the shuffle of targets; the near values f ± 1 and v | every full point |
| 5 | N25: a step-0 lens and the projection of the massive-activation direction | the massive scan at every point; the projected lens (top-3 coordinates of the source block and of block 30 zeroed) at every lens point; the lens at T_0 | L1, L2 (scan), L3 (T_0) |
| 6 | a split-half lens for thresholds | lenses A and B (12 prompts each) at every lens point; delta_read | every lens point |
| 7 | the known number | K0 | every point, before anything else is computed on the model |

### 3.2 The arms at a full point and what each adds

| arm | blocks, positions | adds |
|---|---|---|
| LENS_M | B_m, every query position | the primary: the paper's swap at matched depth |
| LENS_U | blocks 16–29, every query position | the paper's protocol as published; with LENS_M it separates slot from depth |
| DOM_M | ℓm, position 62 | rule 6: a difference-of-means write; the paper's lens-free method |
| ANS_U, ANS_M | as LENS_U, LENS_M | the answer substitution: the reference for H_answer; ANS_U is the positive control of the swap instrument |
| RAND_U, RAND_M, RAND_DOM | as the arm | the random-direction floor of matched norm |
| DOM sweep (reported) | each even block 0–28, position 62; freeze half, v index 0 | the paper's Figure 88: where the mean-difference patch acts |
| IC, RAND_IC | the embedding and blocks 0, 4, …, 28; operand positions | the lens-free positive control and its profile over depth |

### 3.3 Go/no-go (the card, Part 3 item 1.3)

After L1, from the freeze half and the pool counts (`n22.go_nogo`, written by freeze 2):
**GO** if at least 100 of the 1,200 items are answered at T_15 **and** the interchange control at the embedding is
above both floors (paired lower bounds of I_IC − I_RAND_IC and I_IC − I_IC,shuffled both > 0). Otherwise **NO-GO**:
"the lens test had no chance"; L2 and L3 are not launched, and the T_15 verdict is still computed and reported. The
test half of T_15 is unblinded after freeze 2 whatever the go/no-go.

### 3.4 Which points, chosen how

- **L1**: T_15 (full).
- **L2**: T_14, T_13, …, T_0 (accuracy): K0, the massive scan, the pool forward. 15 downloads.
- **L3** (`n22.plan_lens`, frozen): C = the points T_1–T_14 with at least 50 answered in both halves at L2. If C has
  more than four, keep its earliest, its latest and the two nearest the thirds of the index range between them. N25
  adds T_0 and the point just before the earliest of C (the early point). Visiting order: T_0, the N22 points from the
  latest to the earliest, then the early point. The two N25 points are lens-only (kind n25: the readout on every item,
  the probes, N25's statistics, no edits).
- Why T_0 and the early point: the step-0 lens is the architectural floor of every lens statistic; the early point
  brackets the readout before items are answered (reported on all items).

---

## 4. Data

| object | content | identity |
|---|---|---|
| pool | 1,200 items, 600 per half, five v each, two counterfactual pairs each, the partner | `studies/next/n22/items.json`, sha256 `dbf0b2251516edeb869053c72f2c9e6fe4bab2c9250799f308fbda489bba6e69` (3,840,903 bytes); regenerated byte for byte by `python -m infra.pg.n22 items` (dry run REAL) |
| class prompts | 75 slot-1 values and 241 slot-2 values, 16 prompts each (5,056) | inside `items.json` → `class_prompts`, with their token ids |
| token ids | every prompt's 63 ids; the ids of "0"–"999" | inside `items.json`; G-TOK re-encodes 20 items at every point; the dry run re-encodes all 1,200 items, 2,400 counterfactuals and 5,056 class prompts |
| lens corpus | 24 windows of 127 tokens of the grid corpus (`tokens.npy`, sha256 `2aca80ca…`), seeded starts (`n22.SEED_LENS`), BOS prepended | the starts are written to `tNN/massive.json` |
| the known number's corpus | the first 120,000 targets of the same `tokens.npy`, windows of 2,048 | `n22.K0_BANK` from the 16 `controls.json` |
| contamination | the items are generated; exact strings may occur in pretraining text. Not controlled; stated (§8). | — |

The split is fixed before any model output: even index → freeze, odd → test; an item's v, counterfactuals and partner
stay in its half. Everything frozen (bands, delta_read, the matched-layer and DoM-guard decisions, the positive
control's pass) comes from the freeze half; the labels from the test half.

---

## 5. Controls

Rule 4 of "Instruments and controls": fit on both sides (lens A and lens B; both halves of the items), a positive
control that fired (the interchange; ANS_U against RAND_U; LENS_FIDELITY), the value on an identity pair (I-FWD,
I-EDIT, G-IC-ID, I-GRAD; A against B for the lens statistics).

### 5.1 The lens-free positive control: the interchange (critic 1.3)

At the embedding the interchange is the counterfactual prompt itself, so its interaction is what the model does when
the input changes: the slot-1 counterfactual's answer is f1(v), the slot-2 one's is f2(v). Over the sites 0, 4, …, 28
it shows up to which depth the operand positions still carry what decides the answer. It fires when the embedding
site's I_IC is above both floors (RAND_IC and the shuffle) on the freeze half. If it does not fire, no edit at a later
site could show slot dependence: the lens test had no chance. A site whose interaction exceeds the embedding's upper
bound is flagged for inspection (the residual at operand positions cannot carry more slot-specific information than the
input).

### 5.2 Floors (the best achievable reference, never zero)

| floor | of | what it answers |
|---|---|---|
| RAND_U, RAND_M, RAND_DOM, RAND_IC | each edit | does any edit of the same norm at the same positions and blocks give this interaction? |
| the shuffle (I_x, partner's targets) | each edit | does a target of the same form, v·c′ + d′ and v + d′ with another item's c′, d′, rise as much? |
| the partner's readout | the readout | does any number of the same kind sit in the top 10 at that block? |
| the step-0 lens | N25's statistics; the readout at T_0 | is the structure present in the untrained model? |
| the rotated residual | LENS_FIDELITY | does the lens transport block 29's residual toward block 30's at all? |

### 5.3 The exact target (critic 1.2)

The interaction cancels whatever the injected v writes onto all numbers alike. The shuffle tests the next layer:
other items' targets of the same form. Reported beside each arm (`n22.edit_reports`): the share of edits whose target
outranks f ± 1 and v (the exact-target margin), the share whose top-1 is v (H_inject), and the top-1 double flip
(argmax after the slot-1 edit is f1 and after the slot-2 edit is f2, the paper's 133-and-70 event) with Wilson
intervals.

### 5.4 Identity pairs

| pair | pass | failing input (shown failing in the dry run) |
|---|---|---|
| I-GRAD | prompt 0's Jacobian twice, relative Frobenius ≤ 1e-6 at every block | prompt 1's Jacobian (tiny world: 0.544); a planted noisy repeat stops the job |
| A against B | the split-half lens: delta_read = the largest difference of net readout over blocks and slots, freeze half | — (it is the noise ceiling, not a gate) |
| I-FWD | the pool forward twice, the second in reversed batch order, max \|Δ lp\| over the 1,000 number tokens ≤ 1e-4 | each item against its partner's forward; a planted +0.01 stops the job |
| I-EDIT | the lens swap with σ the identity on the matched band, against the clean forward, ≤ 1e-4 | h + 0.01‖h‖r on the band; a planted nonzero zero-edit stops the job |
| G-IC-ID | the patch at the embedding against the counterfactual's own forward, ≤ 1e-4 | against the item's clean forward; patched positions shifted by one stop the job |

### 5.5 The massive-activation projection (N25)

Per point and block, the three coordinates with the largest max |h_i| over every position of the 24 lens prompts
(BOS included). The projected lens zeroes those coordinates on the input side of J_ℓ (the source block's top 3) and on
its output side (block 30's top 3). The readout and the three N25 statistics are recomputed with it. The scan also
records the ratio max |h_i| / median |h| per block at every point (L1, L2, L3), which dates the formation of the
massive coordinates at the 16-point resolution.

### 5.6 The known number K0 (lesson 3)

`n22_score.k0_nll`: the banked harvest's rule (`neurons_harvest.run`): windows of BOS + 2,048 corpus tokens in order,
batches of 8, all logits in bf16 cast to fp32, per-window cross-entropy, the first 120,000 targets. Frozen on the three
control pairs (§0.2):

| tier | where | pass | basis |
|---|---|---|---|
| same device | T_0–T_6, T_15 (banked on L40S) | \|NLL − bank\| ≤ 1e-6 | two L40S pairs differ by 0.0 |
| coarse (gate) | T_7–T_14 (banked on Blackwell) | \|NLL − bank\| ≤ 0.01 | 162 times the device pair (6.17e-5); 3.4 times below the smallest neighbour gap at those points (0.0339, T_11) |
| reported | T_7–T_14 | the difference against 2 × 6.17e-5 | M2 A3: cross-device comparisons are reported, not gated |

Failing input at every point: the neighbouring point's banked value. Its gap over the tier is 3.4–7.7 at the coarse
points and at least 39,390 at the same-device points (`studies/next/n22/offline_gate/gate_sides.json`).

### 5.7 The nonlinear probe beside every linear readout (lesson 5)

At every lens point, at every third source block and at ℓ1, ℓ2, ℓm, a linear softmax probe and a one-hidden-layer MLP
probe (256 ReLU units) classify the slot value from the class prompts' residual at position 62 (12 train, 4 test per
value; 300 Adam steps), beside the same two with the labels permuted. Reported: both test accuracies, the MLP's gain
over the linear probe and the same gain on the permuted labels. Where the lens readout is NOT_PRESENT and the MLP
probe's gain exceeds the permuted gain, the value is present but not on the token directions. A limit: an MLP probe can
compute a + b from operand features itself; its success says the value is recoverable from the state.

### 5.8 Positive controls of the instruments

- **PC-ANS** (the swap instrument): per item, the mean over (v, slot) of lp(f_k | ANS_U_k) − lp(f_k | RAND_U_k); fires
  if its lower bound is above 0, on the freeze half (frozen at freeze 2) and on the test half (at the verdict). If it
  does not fire on either, LENS_M and LENS_U DO NOT COUNT: the swap cannot write a token where the answer is written.
- **LENS_FIDELITY**: per item, cos(J_29 h_29, h_30) against the same lens fed a randomly rotated residual; paired lower
  bound above 0. Gated at full points, reported at n25 points (the step-0 lens). The cosine of the untransported
  residual is reported beside it.

---

## 6. Frozen rules

### 6.1 What is frozen where

| quantity | frozen on | when |
|---|---|---|
| K0's tiers | the three control pairs | now (`n22.K0_TOL_SAME`, `K0_TOL_COARSE`, `K0_REF_CROSS`) |
| bands ℓ1, ℓ2, ℓm, B_m at each point | the point's freeze half, answered items (n25 points: every freeze item), full lens | in-job, by the frozen rule `n22.band_rule` |
| delta_read at each point | lens A against lens B, freeze half | in-job |
| matched layer exists; the DoM guard | freeze half against delta_read | freeze 2 (T_15), freeze 3 (L3 points) |
| PC-ANS freeze-half pass; the interchange control; the go/no-go | freeze half; the pool counts | freeze 2 |
| the test files | sha256 of `edits_test.jsonl`, `ic_test.jsonl` at freeze; the verdict refuses a changed file | freeze 2, freeze 3 |

Registered a priori, with the reason: the 1e-4-nat tolerance of the fp32 identity pairs (no fp32 forward of this model
is banked; the tiny world gives 0); I-GRAD's 1e-6 (expected 0); K_READ = 10 (the paper's pass@10 evaluations);
N_MIN = 50 answered items per half; N_SWAP_MAX = 200 per half (cost); every paired label's 0; the 10 × step-0 ratio
that dates the massive coordinates; the CKA block windows (1–3 blocks apart against ≥ 15 apart). No threshold is set on
test-half data.

### 6.2 The labels at one point (test half; `n22.point_verdict`)

1. The test files' sha256 equal the frozen ones; every instance gate passed and every failing input failed.
2. **Readout** per slot at ℓ_k: PRESENT if the net share exceeds delta_read and the Wilson interval of the hit share
   lies above that of the partner's; else NOT_PRESENT. **READOUT_SINK** if the projected lens's net share differs from
   the full lens's by more than delta_read, else READOUT_SURVIVES. "Readout present" for the order test means both
   slots PRESENT.
3. **Each arm** (LENS_M with RAND_M, LENS_U with RAND_U, DOM_M with RAND_DOM), on items with both slots: fewer than 50
   items → DOES NOT COUNT; ABOVE_FLOORS if the paired lower bounds against the random floor and the shuffle are both
   > 0; BELOW_FLOOR if either upper bound < 0; AT_FLOOR otherwise. **Ceiling**: the test half's I_IC at the embedding;
   an arm whose lower bound exceeds the ceiling's upper bound DOES NOT COUNT until inspected.
4. LENS_M and LENS_U DO NOT COUNT if PC-ANS did not fire on both halves. LENS_M and DOM_M DO NOT COUNT if no matched
   layer. DOM_M DOES NOT COUNT if the answer is readable at ℓm.
5. **H_slot** = LENS_M: ABOVE_FLOORS → SLOT_DEPENDENT; AT_FLOOR → NOT_SHOWN; BELOW_FLOOR → BELOW_FLOOR. Qualifier
   "above answer substitution": I_LENS_M − I_ANS_M paired lower bound > 0. **Depth reading**: LENS_U ABOVE and LENS_M
   AT or BELOW → DEPTH_NOT_EXCLUDED.
6. Reported, no label: the edit reports (§5.3), the interchange profile (freeze half), the DoM sweep (freeze half), the
   probes, N25's statistics, the massive ratio, accuracy and the partner's-answer rate.

### 6.3 Across points (`n22.order_label`, `n22.dating_arm`)

- **The dating arm**: LENS_M if SLOT_DEPENDENT at T_15; else LENS_U if ABOVE_FLOORS; else DOM_M if ABOVE_FLOORS and
  counting; else none and H_order is NOT_RUN.
- Over the scored points (the N22 points of L3 and T_15) in time order: t_r = the first with readout present, t_s =
  the first with the dating arm ABOVE_FLOORS. SUPPORTED if t_r < t_s; REFUTED_CONTRARY if t_s < t_r or readout never
  present; if t_r = t_s: CENSORED when it is the earliest scored point (both precede what the grid can score),
  REFUTED_SAME otherwise. The interval is printed as "between T_x and T_y" (critic Part 3 item 5).
- **N25** (`n22.n25_labels`), per statistic S (CKA block, kurtosis maximum, MLP gain maximum), noise δ_S(t) = |S_A −
  S_B| at t: ARCHITECTURAL if |S(T_0) − S(T_15)| ≤ δ_S(T_0) + δ_S(T_15), else LEARNED; SINK_DEPENDENT at t if
  |S_proj − S_full| > δ_S(t); FORMS_BETWEEN the first consecutive pair of lens points whose change exceeds the sum of
  their noises, SMOOTH if no pair does but the total does, NO_CHANGE otherwise. The massive coordinates "form" at the
  first point whose ratio is at least 10 times T_0's (`n22.massive_formed`).

### 6.4 Room

At 50 items per half and a spread of 1 nat per edit, a planted interaction of 1 nat and of 0.5 nat reaches
ABOVE_FLOORS in 20 of 20 draws, 0.25 nat in 14 of 20 (`N22_OFFLINE_GATE.md`, VERDICT room; `dryrun_summary.json` →
`room`). The spread on OLMo-3 is unknown before L1; the freeze half's spread is reported at freeze 2 beside the
smallest interaction it resolves. Every label reaches each of its branches on planted records (VERDICT section).

---

## 7. N25 in this card

N25 is the instrument check of N22 (`05_SCORING.md`). Its three statistics are computed at every lens point for four
lens variants (A, B, full, projected), in `n22_score.n25_stats`:
- **CKA block structure**: linear CKA between the lens-vector matrices of 2,000 seeded tokens at every pair of source
  blocks; the mean over pairs 1–3 blocks apart minus the mean over pairs ≥ 15 apart.
- **Readout kurtosis**: the median over 128 sampled corpus positions of the excess kurtosis of the lens logits over the
  vocabulary; the maximum over blocks.
- **MLP gain**: for 256 lens vectors and 256 isotropic directions, at the block's median residual norm, the median norm
  of block ℓ+1's MLP output (before its output norm; OLMo-3 normalizes the MLP's output, so this is the MLP's raw
  response, not the update's size); the ratio; the maximum over blocks.
Its refutations are the card's: the step-0 lens within split-half noise (architectural); a move beyond split-half
noise under the projection (the band is the sink); everything smooth. Its resolution in time is the lens points: T_0,
the early point, up to four N22 points, T_15.

---

## 8. Threats to validity

| # | threat | effect | mitigation |
|---|---|---|---|
| 1 | only answered items are edited; answered sets differ by point | the populations differ across checkpoints | stated; each point scored on its own items (critic); the order test compares labels, not values |
| 2 | slot confounded with depth | LENS_U's reading ambiguous | LENS_M at matched depth; NO_MATCHED_LAYER branch |
| 3 | near values and number features | I could reflect magnitude | the 2 × 2 cancels v's own write; the shuffle with other items' targets; the exact-target margin |
| 4 | the edit removes different content in the two rows (s1 against s2) | part of I comes from removal | the random floor of the same norm; stated |
| 5 | DOM's class means carry the answer shift for slot 2 | DOM_M raises f2(v) directly | the DoM guard (§2.6); stated as a limit when it holds |
| 6 | the lens reads the massive coordinates | the readout and bands are the sink's | the projected lens; READOUT_SINK; N25 |
| 7 | the lens is fitted on 24 prompts | lens noise | split-half delta_read in every threshold; A and B reported |
| 8 | bf16 rounding | small differences are noise | fp32 forwards; three identity pairs |
| 9 | the prompts are synthetic, three fixed solved lines | format effects | the same format at every point and arm; stated |
| 10 | contamination of exact expressions | memorized answers | stated; the counterfactuals and v targets are mostly unseen combinations |
| 11 | the Blackwell-banked known number at T_7–T_14 | K0 there is coarse | G-REV, G-CFG, G-TOK still exact; the difference reported |
| 12 | one model, one run, one prompt format | no replicate | stated; OLMo-2 has no stage-1 grid here |
| 13 | the MLP probe can compute the value itself | its success over-reads presence | stated (§5.7) |
| 14 | "upper half" is our translation of the paper's layer range to blocks 16–29 | LENS_U differs from the paper | stated; the paper's layer indices are on a 0–100 scale |

---

## 9. Runbook

### 9.1 Checklist before any launch

- Read `.claude/skills/aws-pg/SKILL.md` and the docstrings of `infra/aws_pg/jobs/pipeline.py` and `n22_score.py`.
  Profile `fuelos`, region `us-east-1`, on every call.
- Worktree `../interp_research-n22`, branch `n22`; it needs its own `.pgvenv` for `./infra/aws_pg/pg` (`09_PLAN.md` J0).
  Commit named files only.
- Liveness: an instance id from `./infra/aws_pg/pg ps`, or `updated_utc` of `out/score/n22/metrics.json`. Never poll
  `job.log`.
- Cost and incidents: `infra/aws_pg/operations/2026-11-<dd>_next_n22.md`; science: `studies/next/n22/JOURNAL.md`.

### 9.2 Code (NEW or REUSED, checked against the tree at `0f42e91d2`)

| file | status | role |
|---|---|---|
| `infra/pg/n22.py` | NEW | the pool, prompts, scoring, labels, freeze, verdict, plan-lens, projection; CLI |
| `infra/aws_pg/jobs/n22_score.py` | NEW | the job |
| `infra/aws_pg/jobs/n22_dryrun.py` | NEW | the offline gate |
| `infra/aws_pg/steps/next_n22_l1-01.json`, `next_n22_l2-01.json` | NEW | the step lists of L1 and L2 (L3's are written by `n22 plan-lens`, §9.5) |
| `studies/next/n22/offline_gate/gate_sides.py` | NEW | both sides of every bank-held gate at 16 points |
| `studies/next/n22/samples/raw_examples.py` | NEW | the zero-GPU part of the ten raw examples |
| `infra/pg/vendor/jlens` (`fitting.jacobian_for_prompt`, `hf.from_hf`, `hooks.ActivationRecorder`) | REUSED, unchanged | the lens estimator |
| `infra/aws_pg/jobs/n12_score.py` (`grad_mode`) | REUSED, unchanged | deterministic gradient settings |
| `infra/aws_pg/jobs/m2_score.py` (`Run`, `ALLOW`, `VERSIONS`, `TOKENS_SHA`, `utc`, `dump`) | REUSED, unchanged | job pieces |
| `infra/aws_pg/jobs/m2_dryrun.py` (`make_corpus`, `tiny_model`, `make_remote`, `fake_hub`) | REUSED, unchanged | the tiny world |
| `infra/pg/m2.py` (`POINTS`, `REPO`, `BOS`, `wilson`, `sha256_file`, `read_jsonl`, `L40S`) | REUSED, unchanged | constants |
| `infra/pg/m1.py` (`config_gate`, `REF_CONFIG`), `infra/pg/neurons_harvest.py` (`load_model`, `text_tower`, `forward`, `run`), `infra/pg/neurons_corpus.py`, `infra/pg/matrix_models.py` (`vocab_sha256`), `infra/aws_pg/jobs/neurons_harvest.py` (`fetch`, `fetch_corpus`), `infra/pg/diff_harvest.py` (`hf_online_setup`), `infra/aws_pg/jobs/pipeline.py` | REUSED, unchanged | as M2 |

### 9.3 The offline gate

```
SP=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad
cd /Users/abaybektursun/projects/interp_research-n22
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n22 items --tok $SP/n22/hf/tok --out studies/next/n22/items.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.pg.n22 project > studies/next/n22/projection.json
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n22/samples/raw_examples.py --tok $SP/n22/hf/tok --out studies/next/n22/samples
PYTHONPATH=. $SP/m1_venv/bin/python studies/next/n22/offline_gate/gate_sides.py --tok $SP/n22/hf/tok \
    --tok2 $SP/pilot0_in/tokenizer.json --out studies/next/n22/offline_gate/gate_sides.json
PYTHONPATH=. $SP/m1_venv/bin/python -m infra.aws_pg.jobs.n22_dryrun --hf $SP/n22/hf \
    --sides studies/next/n22/offline_gate/gate_sides.json --out $SP/n22/dry
```
`$SP/n22/hf` holds `tok/` (the OLMo-3 tokenizer files) and `config_stage1-step476838.json`. It must end with
`N22_DRYRUN_OK`. The record is `N22_OFFLINE_GATE.md`.

### 9.4 S3 layout

```
results/next_n22/inputs/items.json
results/next_n22/<launch>-NN/steps.json  STATUS  job.log  heartbeat.jsonl
results/next_n22/<launch>-NN/out/score/n22/metrics.json
results/next_n22/<launch>-NN/out/score/n22/samples/raw_examples_tNN.jsonl
results/next_n22/<launch>-NN/out/score/n22/tNN/{design.json, accuracy.json, clean_lp_numbers.npy, readout.npz,
    massive.json, probes.json, n25.json, lens_AB.pt (about 2 GB), edits_{freeze,test}.jsonl, ic_{freeze,test}.jsonl,
    dom_sweep_freeze.jsonl, raw_example_edits.jsonl}
```

### 9.5 Sessions

```
B=s3://pg-lineage-769063704696
CORPUS=$B/results/mdl/neurons/w1/q4b/1/out/corpus/corpus
```

**S1. Freeze 1 (laptop).** §9.3 to `N22_DRYRUN_OK`; the head's review; commit; tag `n22-freeze-1`; record tag, SHA,
UTC in §0. `./infra/aws_pg/pg ps` and `audit`: no stray instance. `aws s3 ls $B/results/next_n22/ --profile fuelos
--region us-east-1` lists nothing. Upload: `aws s3 cp studies/next/n22/items.json $B/results/next_n22/inputs/items.json
--profile fuelos --region us-east-1`; download it back and compare sha256.

**S2. L1 (`l1-01`; 1.05–1.76 GPU-h; hard kill 138 min).**
```
aws s3 cp infra/aws_pg/steps/next_n22_l1-01.json $B/results/next_n22/l1-01/steps.json --profile fuelos --region us-east-1
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_n22/l1-01/steps.json" \
    --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_n22/l1-01 --budget-min 138 --json
./infra/aws_pg/pg watch results/next_n22/l1-01
```
The capacity rule of M2 (every 180 s for six tries, then every 600 s). The first gates to read: K0, then I-GRAD, then
`measured`. During the run only `n22/metrics.json` is read.

**S3. The first sample check, freeze 2, the T_15 verdict, the go/no-go (laptop).** Pull `n22/metrics.json`,
`samples/raw_examples_t15.jsonl`, `t15/` except `lens_AB.pt` to `studies/next/n22/banks/l1-01/`. Open the raw examples
first and answer in `studies/next/n22/JOURNAL.md`: is the prompt what §0.5 says; is the model's top-1 a number token;
for one answered and one unanswered item, are p(y), the lens top-10 at ℓ1, ℓ2, ℓm and the after-edit top-1 what the
records hold. Copy the author's prior (§1.4) into the journal. Then `python -m infra.pg.n22 freeze --banks
studies/next/n22/banks/l1-01 --out studies/next/n22/thresholds.json`; commit; tag `n22-freeze-2`. Then the verdict
once: `python -m infra.pg.n22 verdict --banks studies/next/n22/banks/l1-01 --thresholds studies/next/n22/thresholds.json
--items studies/next/n22/items.json --out studies/next/n22/verdict_t15.json`. Fill §12's T_15 rows. **The go/no-go is
`thresholds.json` → `points.15.go_nogo.go`.** NO-GO ends the study here.

**S4. L2 (`l2-01`; 1.01–1.58 GPU-h; hard kill 124 min).** As S2 with `next_n22_l2-01.json` and `--budget-min 124`.

**S5. Plan L3 (laptop).** `python -m infra.pg.n22 plan-lens --metrics studies/next/n22/banks/l2-01/n22/metrics.json >
studies/next/n22/plan_l3.json`. Write the step files of L3a (the first three points of `order`) and L3b (the rest),
each `--points` with kind n25 for `n25_points` and full otherwise; project each with `n22.project` and set each
launch's kill at 1.3 times its high projection. Commit them.

**S6. L3a and L3b** (together 5.39–8.94 GPU-h). Two launches, in parallel if capacity allows (the quota holds four).

**S7. Freeze 3, the stage-2 verdict (laptop).** `n22 freeze` on each L3 bank with its points → `thresholds_stage2.json`;
commit; tag `n22-freeze-3`; `n22 verdict` per bank; then `n22.order_label` over the per-point verdicts with
`n22.dating_arm(T_15)`, `n22.n25_labels` over the lens points' `n25.json`, `n22.massive_formed` over every point's
`massive.json`. Fill §12.

**S8. The independent recount** (a fresh Opus agent, its own code, from the records and `readout.npz`) and the head's
read. The result document comes after both.

### 9.6 Budget, stop rule, park order

| launch | points and kinds | GPU-h, center | GPU-h, high | hard kill |
|---|---|---|---|---|
| L1 `l1-01` | T_15 full | 1.05 | 1.76 | 138 min |
| L2 `l2-01` | T_14 … T_0 accuracy (15) | 1.01 | 1.58 | 124 min |
| L3a, L3b | up to four full points, T_0 and the early point (n25) | 5.39 | 8.94 | 1.3 × each launch's high projection (about 349 min each) |
| total | | 7.46 | 12.29 | |

Per point (`n22.project`): a full point 3,572 s center, 6,119 s high (download 82–192 s, load 60 s, K0 15 s, lens 25
prompts × 78–120 s, 63,056 fp32 forwards × 0.018–0.040 s, 300 s of statistics and writes); an accuracy point 229–365 s;
an n25 point 2,450–3,747 s; boot 224 s per launch. At $2.24 per hour: $16.71–27.53; on g6e.4xlarge ($3.00424):
$22.41–36.92. Split by card: N25 alone adds the two n25 points of L3 (1.36–2.08 GPU-h); the rest is N22. The lens
rate is the one unmeasured rate (an estimate scaled from N12's measured gradient pass by FLOPs); the fp32 forward rate
is an estimate.

**Stop rule, in the job**: before each later point, the elapsed time plus the remaining points at each kind's
measured wall time (the registered high estimate until one of that kind has run) must stay below 0.9 × the budget;
otherwise this and every later point are PARKED and the job ends normally (planted in the dry run). **Park order**:
the end of `order` first: the early N25 point, then the N22 points from the earliest; T_0 and the latest N22 point
are kept longest. If parking leaves no N22 point, H_order is scored on T_15 alone (NOT_RUN or CENSORED) and a
relaunch for the parked points is the head's decision.

### 9.7 Artifacts

- On S3 (§9.4): gates with their failing-input values, the records, the lenses, the examples.
- On the laptop: `studies/next/n22/{items.json, projection.json, samples/, offline_gate/, banks/, thresholds.json,
  thresholds_stage2.json, verdict_t15.json, verdict_stage2.json, JOURNAL.md}`.
- Figures (after S7): the interaction per arm with both floors drawn per point; the readout curves over blocks with
  the partner's line and delta_read; the interchange profile; N25's three statistics over the lens points with the
  split-half band and the step-0 value drawn. Each regenerated after any correction.

---

## 10. The ten raw examples

**Zero-GPU part, written now** (`studies/next/n22/samples/raw_examples_zero_gpu.{md,json}`, by
`studies/next/n22/samples/raw_examples.py`): the first ten freeze-half items as the model sees them.

| item | query | s1 | s2 | y | v | f1 | f2 | partner's f1, f2 | slot-1 counterfactual | slot-2 counterfactual |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | (2 + 20) * 4 + 62 = | 22 | 88 | 150 | 81 | 386 | 143 | 388, 145 | (8 + 73) * 4 + 62 = | (19 + 8) * 3 + 62 = |
| 2 | (12 + 1) * 4 + 64 = | 13 | 52 | 116 | 86 | 408 | 150 | 776, 174 | (27 + 59) * 4 + 64 = | (11 + 32) * 2 + 64 = |
| 4 | (2 + 12) * 8 + 88 = | 14 | 112 | 200 | 76 | 696 | 164 | 655, 123 | (30 + 46) * 8 + 88 = | (2 + 17) * 4 + 88 = |
| 6 | (10 + 39) * 8 + 47 = | 49 | 392 | 439 | 95 | 807 | 142 | 598, 123 | (58 + 37) * 8 + 47 = | (15 + 4) * 5 + 47 = |
| 8 | (27 + 9) * 6 + 28 = | 36 | 216 | 244 | 51 | 334 | 79 | 301, 148 | (36 + 15) * 6 + 28 = | (2 + 15) * 3 + 28 = |
| 10 | (40 + 18) * 4 + 97 = | 58 | 232 | 329 | 86 | 441 | 183 | 495, 151 | (19 + 67) * 4 + 97 = | (23 + 20) * 2 + 97 = |
| 12 | (3 + 11) * 5 + 65 = | 14 | 70 | 135 | 80 | 465 | 145 | 401, 161 | (60 + 20) * 5 + 65 = | (18 + 22) * 2 + 65 = |
| 14 | (25 + 16) * 4 + 81 = | 41 | 164 | 245 | 63 | 333 | 144 | 548, 107 | (52 + 11) * 4 + 81 = | (7 + 2) * 7 + 81 = |
| 16 | (34 + 13) * 8 + 44 = | 47 | 376 | 420 | 75 | 644 | 119 | 440, 140 | (46 + 29) * 8 + 44 = | (4 + 21) * 3 + 44 = |
| 18 | (2 + 12) * 5 + 65 = | 14 | 70 | 135 | 52 | 325 | 117 | 504, 140 | (44 + 8) * 5 + 65 = | (2 + 11) * 4 + 65 = |

What they show at zero GPU: every number is one token; the prompt has 63 tokens and the answer is predicted at
position 62 after `Ġ=` `Ġ`; f1(v) and f2(v) differ from each other, from v and from every number of the item, so a
raised target is neither the injected token nor a copied operand; the slot-1 counterfactual keeps c and d and its
answer is f1(v), the slot-2 one keeps d and its answer is f2(v). Item 14 shares a and b with the third solved line
(25 + 16), allowed by the rule (only s1, s2, y may not appear in the solved lines); its readout of 41 is read with
that in mind. Items 12 and 18 share s1 = 14, c = 5, d = 65 with different operands, so they share s2 and y: the
readout of the same intermediate from two operand pairs.

**PENDING, the first GPU step**: L1 writes `samples/raw_examples_t15.jsonl` right after the bands are chosen and before
any other edit: per item the model's top-1 and its probability, p(y), the lens top-10 with probabilities at ℓ1, ℓ2,
ℓm, the ranks of s1 and s2 at every block, and for v index 0 the top-1 and the scored log-probabilities after every
arm and floor. If the first ten agree in correctness, the first freeze item of the other kind is added. The executor
opens this file before any number (S3).

---

## 11. The three dimensions

- **Predictive.** H_slot is a prediction from one curated Claude example to an open 7B at matched depth, scored once
  on a held-out half under rules frozen on the other. H_order is a registered prediction across checkpoints. The prior
  for both is low (§1.4); a refutation is the expected outcome and is reported in the same register.
- **Novel.** No across-training test of slot-dependent arithmetic, and no matched-depth version of the paper's test,
  was found (§1.5). A NO-GO or a failure on OLMo-3 is itself new: the Qwen failure was blamed on digit splitting, and
  OLMo-3 keeps every number below 1,000 as one token.
- **Practical.** None claimed. The one usable output is an instrument fact: whether the lens coordinates, a
  difference-of-means write or neither carries an arithmetic variable on an open 7B, which tells a user of the lens
  whether its swap is a valid intervention there.

---

## 12. Prediction grid (filled at unblinding, whatever the result)

T_15 filled 2026-11-06T18:01Z from `studies/next/n22/verdict_n22_t15.json` (the verdict run once after tag `n22-freeze-2`); where the verdict holds no value the row names the file that does or says why none exists.

| row | quantity | prior | value (T_15) | label | later points |
|---|---|---|---|---|---|
| 1 | answered at T_15 (of 1,200); freeze, test | ≥ 100 at 0.80 | 7 of 1,200: freeze 2, test 5 (`verdict_n22_t15.json` → `15.n_answered`) | below 100 and below N_MIN 50 per half | L2 not launched (NO-GO) |
| 2 | the interchange control at the embedding (freeze): I_IC − I_RAND_IC, I_IC − I_IC,x | fires 0.75 | not in the verdict; not run (fewer than 50 answered per half); the go/no-go is `thresholds.json` → `points.15.go_nogo.go` = false | NO-GO | — |
| 3 | delta_read; ℓ1, ℓ2, ℓm; the matched net readout | — | delta_read 0.5, ℓ1 = 0, ℓ2 = 25 (`verdict_n22_t15.json` → `15.readout.*.delta`, `.layer`; set by 2 answered freeze items); ℓm = 0 and matched net readout 0.0 are in `thresholds.json`, not the verdict | no matched layer (`thresholds.json`) | — |
| 4 | readout s1, s2 at ℓ1, ℓ2 (test): net share | PRESENT 0.50, 0.35 | s1 at block 0: hit 0.0, partner 0.0, net 0.0 (n 5); s2 at block 25: hit 0.2, partner 0.0, net 0.2 (n 5) (`verdict_n22_t15.json` → `15.readout`) | s1 NOT_PRESENT, s2 NOT_PRESENT | — |
| 5 | READOUT_SINK s1, s2 | 0.30 | projected-lens net: s1 0.0, s2 0.2 (equal to the full lens's) | s1 READOUT_SURVIVES, s2 READOUT_SURVIVES (on 5 items, delta_read 0.5) | — |
| 6 | LENS_M: I, I − RAND_M, I − shuffle | SLOT_DEPENDENT 0.20 | NOT_RUN: no edits at this point (`verdict_n22_t15.json` → `15.slot`) | H_slot not labelled | — |
| 7 | LENS_M − ANS_M | — | NOT_RUN: no edits at this point (`verdict_n22_t15.json` → `15.slot`) | — | — |
| 8 | LENS_U: I and both differences | 0.25 | NOT_RUN: no edits at this point (`verdict_n22_t15.json` → `15.slot`) | not labelled | — |
| 9 | DOM_M: I and both differences; the DoM guard | 0.35 | NOT_RUN: no edits at this point (`verdict_n22_t15.json` → `15.slot`); the DoM guard (`thresholds.json`): answer not readable at ℓm (net 0.0) | not labelled | — |
| 10 | PC-ANS freeze, test | fires | not computed: no edit records on either half | — | — |
| 11 | the ceiling: test-half I_IC at the embedding | — | not computed: no interchange records | — | — |
| 12 | double flips, top-1 = v, exact-target margin per arm | — | NOT_RUN: no edits at this point (`verdict_n22_t15.json` → `15.slot`) | — | — |
| 13 | H_order | see §1.4 | — | NOT_RUN (no dating arm: no arm was scored at T_15) | L3 not launched |
| 14 | N25: H_arch, H_sink, H_form per statistic; the massive formation point | LEARNED 0.70 | not in the verdict: the N25 labels need T_0's lens (L3), not launched; T_15's statistics are reported in `banks/l1-01/n22/t15/n25.json` | not labelled | — |
| 15 | the probes: MLP gain over linear, against the permuted gain, at ℓ1, ℓ2, ℓm | — | not in the verdict; reported in `banks/l1-01/n22/t15/probes.json` | — | — |

---

## 13. References

- Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models", arXiv 2607.15495 (read:
  §3.3, §A.24.1, §A.24.2 for the claims in §0.2; the rest through `01_JLENS.md`).
- `01_JLENS.md` (the distillation, with the Nanda commentary, Zeisler, Brauer, Panchal, Ostrow et al., the J++ post);
  `04_RECENT.md` §8 (public OLMo-3 lenses).
- Nikankin, Reusch, Mueller, Belinkov, "Arithmetic Without Algorithms", arXiv 2410.21272 (abstract only).
- Stolfo, Belinkov, Sachan, arXiv 2305.15054 (abstract only, through `06_CRITIC.md`).
- Fel et al., "Shaping & Monitoring Models through their Internal Representations" (through `12_SHAPING_NOTE.md` §4).
