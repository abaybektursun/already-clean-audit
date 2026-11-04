# NEURONRW — do clean neurons write what they read?

Answer the study will give, in one line per arm: (Arm 0) per model, the share of poles whose output tokens a blind
judge says match their input label, minus the same share for shuffled labels, with its 95% interval; (self-repair)
the median ratio of a pole's measured ablation ΔNLL to its direct-path ΔNLL; (Arm 1) for seven code-construct poles,
the change in construct presence between the largest coherent positive and negative steering dose, against the
matched control's interval; (Arm 2) per model, the share of steered ONE poles whose continuations a blind judge
assigns to their own label out of three, against 1/3 and against the same judge on unsteered text.

## 0. Status and provenance

- Status: **DRAFT**, 2026-10-25; amended 2026-10-26 by A1–A19 (§14.2) after an adversarial review and two re-reviews, all
  accepted by the director. The body below carries every amendment. Frozen at tag `neuronrw-freeze-1` (§13 freeze record), placed only
  after the director reports the code merged and the review in. Nothing in §4–§7 changes after the tag except by a
  dated amendment in §13.
- Branch `neurons/rw`, worktree `../interp_research-neuronrw`, based on `neurons/off` (b98c9eb4). Code branch
  `neurons/rw-code`, worktree `../interp_research-neuronrw-code`. Banks are untracked and read by absolute path.
- Parents: "already clean" D1 (`../interp_research-drafts/studies/already-clean/SPEC_D1.md`, labels and classes);
  NEURONOFF (closed; `studies/neuronoff/SPEC.md`, `RESULTS.md`, `CLOSEOUT.md`, `checks/2026-10-25_result_review.md`,
  `LIT_steering.md`).
- Director's brief: `NEURONRW_BRIEF.md` (session scratchpad, 2026-10-25). Changes from it are listed in §14.
- Companion files: `SCHEMA.md` (file contract), `AUDIT.md` (already-done measurements, the output-token leak,
  literature read), `plan/` (frozen items, draws and prompts; built by `plan/make_plan.py`), `JOURNAL.md` (from the
  PREDICTIONS commit on).

## 1. Question, decision, hypotheses

**Object.** A neuron pole, as in the parents. For neuron i of layer l, `a = act(W_gate x) · (W_up x)` is the
down_proj input; the positive pole fires where a > 0, the negative pole where a < 0. When the pole fires it adds
`a · W_down[:, i]` to the MLP output, so it writes along `w = sign · u_i`, `u_i = W_down[:, i] / ||W_down[:, i]||₂`.
The parent gave each audit pole an input label (the explainer's `explanation`, what makes it fire) and a class
(ONE = one job, explanation passed the blind test; UNK = failed or declined).

**Question.** Does the vocabulary a pole writes toward agree with its input label? Is a pole's ablation effect
smaller than its direct path implies (self-repair)? Can pushing a pole along w make the model write its label?

**Closest prior work on input against output (A17).** Arad, Mueller & Belinkov 2025 ("SAEs Are Good for Steering — If
You Select the Right Features", arXiv 2505.20063, EMNLP 2025; read 2026-10-26 for these claims) define, for SAE
features, an input score (§3.1, Eq. 8: the fraction of a feature's top activating tokens found among its top logit-lens
tokens) and an output score (§3.2, Eq. 9–10: the change in rank-weighted probability of those tokens when the feature
is steered at factor 10 on a neutral prompt). High values of both "rarely co-occur in the same features" (abstract;
§5, Figure 6: most features lie near one axis); high input scores sit in early layers and high output scores at about
66–100% of depth in Gemma-2 (§3.4, Figure 3); keeping high-output-score features gives "2-3x improvements when
steering". Their input score is a token-identity form of Arm 0's agreement, and it is highest early, which points the
other way from Geva et al. 2021 and Gurnee et al. 2024 for H_depth; their output score is close to Arm 2. NEURONRW
differs in object (MLP neuron poles in base models, not SAE latents), in the read side (a tested natural-language input
label, not a token list), and in the judge.

**Decision.** If labels and write directions agree above shuffle (H_write) and steering produces the label (H_suff),
an input label is also a usable output label, and the next neuron study can steer by label. If they agree on paper
but steering does not produce the label, input labels describe what the model reads; the next study selects poles
by output-side tests (Arm 0 agreement, depth) before any steering claim. If neither holds, input labels say nothing
about what a pole writes, and the parent's legibility result is a statement about reading only.

**Hypotheses and refuting observations** (amended A3, A4, A5). Statistics are defined in §7. "lo" and "hi" are the
ends of a 95% percentile bootstrap interval (10,000 resamples, `numpy.random.default_rng(20261025)` fresh for each
interval). Arm 0 rows use the re-explained labels (§5.0) unless a row says otherwise.

| id | claim | supported when | refuted when | EQUIVALENT when the interval lies inside |
|---|---|---|---|---|
| H_write (Arm 0) | a pole's top-20 output tokens match its input label more often than a shuffled label, and more often than its label matches another pole's list | lo(D_w) > 0, D_w ≥ δ_w, and lo(D_f) > 0 | hi(D_w) ≤ 0 | ±0.10 |
| H_depth (Arm 0) | agreement rises with depth | lo(D_w,late − D_w,early) > 0 | hi ≤ 0 | ±0.15 |
| H_ONE (Arm 0) | ONE poles agree more than matched UNK poles | lo(paired mean of s_ONE − s_UNK) > 0 | hi ≤ 0 | ±0.10 |
| H_repair | the pole's own-position ablation ΔNLL is less than half its direct-path ΔNLL | hi(M_r) < 0.5 | lo(M_r) ≥ 1 | [0.8, 1.25] ("equivalent to no repair") |
| H_repair_ONE | ONE poles are repaired more (lower ratio) than matched UNK poles | hi(median paired difference) < 0 | lo ≥ 0 | ±0.20 |
| SWINGS (Arm 1, per pole) | steering the pole moves construct presence by ≥ 20 points across its coherent range, beyond its control | abs(S_t) ≥ 0.20 (count readout: ≥ 1.0) and the paired interval of S_t − S_c excludes 0 on the side of S_t | NO SWING: the paired interval of S_t − S_c contains 0 and lies inside ±0.10 (count: ±0.5) | — (otherwise UNRESOLVED) |
| H_suff (Arm 2) | steered ONE poles produce continuations a judge assigns to their own label above chance, at the primary coherent dose | lo(mean o) > 1/3 and lo(paired mean of o − o_α=0) > 0 | hi(mean o) ≤ 1/3 | 1/3 ± 0.08 ("equivalent to chance") |
| H_pred_agree (Arm 2) | Arm 0 agreement predicts steering success: the agree stratum's on-label rate is at least twice the disagree stratum's | lo(R) > 1 and R ≥ 2 | hi(R) ≤ 1 | [0.8, 1.25] |
| H_pred_conf (Arm 2; reported equivalence row) | explainer confidence does not predict steering success | the Spearman interval lies inside (−0.2, 0.2) | the Spearman interval excludes 0 | (the supported condition is the equivalence) |
| H_depth (Arm 2) | on-label rate rises with depth | lo(o_late − o_early) > 0 | hi ≤ 0 | ±0.15 |
| B1 leak (amended A16) | seeing the output tokens makes an explainer's label agree more with them (AUDIT §3) | lo(L) > 0, L = paired mean over ONE items, pooled over the three models, of s(label written with the vocabulary lines) − s(label written without them), both by the same re-explainer model | hi(L) ≤ 0 | ±0.05 |
| B2 instrument fault | the steering hook adds something other than α·RMS_l·w after the post-MLP norm | any gate of §6 fails | every gate passes, each registered failing input fails and each registered passing input passes | — |
| B3 judge bias | the judge prefers some labels or letters whatever the text | the α = 0 on-label rate exceeds 0.5 (G-judge2 ceiling) or an fc3 letter share differs from 1/3 by more than 0.1 | the α = 0 rate is ≤ 0.5 and every letter share is within 1/3 ± 0.1 | — |

A label is SUPPORTED, REFUTED or EQUIVALENT only by the columns above. An interval that contains the null value and
does not lie inside the margin reads **NOT SHOWN** (NEURONOFF A1: an interval covering 0 does not refute a direction).
In-between labels (for example "above shuffle, below margin", "repair, not below 0.5") are reported and never promoted.
Every printed label carries its room stamps from §7.5 beside it: the R-pass share for the supported label and the
R-equiv share for EQUIVALENT, at the realized n. The margins are stated, not fitted; §7.5 checks each has room.

## 2. The instrument

### 2.1 Where the norm sits (Hugging Face `transformers` main, commit c9b377c53abd6deb23ee32c8abe7ccdebb568c27, fetched 2026-10-25)

`src/transformers/models/gemma3/modeling_gemma3.py`, `Gemma3DecoderLayer.forward`, lines 412–416:
```
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
```
`Gemma3RMSNorm.forward`, line 148: `output = output * (1.0 + self.weight.float())` (gain is 1 + weight).

`src/transformers/models/olmo3/modeling_olmo3.py`, `Olmo3DecoderLayer.forward`, lines 262–265:
```
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
```
(Olmo 3 has no norm before the MLP; `Olmo3RMSNorm.forward` line 58 returns `(self.weight * hidden_states)`.)

`src/transformers/models/qwen3/modeling_qwen3.py`, `Qwen3DecoderLayer.forward`, lines 311–314:
```
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
```
(no norm after the MLP). In Gemma 3 and Olmo 3 a vector added to the down_proj output is renormalized by
`post_feedforward_layernorm`, so its realized size saturates (NEURONOFF `LIT_steering.md` §C1). The job runs the
`transformers` version on the pg AMI (5.9.0 in NEURONOFF's manifests); the code writer checks that the installed
version has the same three orderings and the same decoder-layer return type, and G-place (§6) checks the realized
vector on real weights.

### 2.2 Steering hook

A forward hook on the decoder layer module `decoder_layers(model)[l]` (`infra/pg/neuron_off_core.py:decoder_layers`)
adds, to the layer's output hidden state, `v = α · RMS_l · w` at every sequence position ≥ P − 1, where P is the
prompt length including BOS (the last prompt token and every generated token). The add is in fp32 (`h.float() + v`),
cast back to the model dtype. Rows of one batch may carry different poles and doses; a row's vector is zero at
layers other than its pole's. The add runs at α = 0 too (G-ident). Positions < P − 1 are never changed (G-floor-pos).
`w` is computed once per pole from the model's bf16 `W_down` in fp32. Sign: a negative pole pushes −u_i at α > 0.
Positions are absolute: the hook keeps a running offset over the forwards of one `generate` call (prefill, then one
position per decode step), as `neuron_off_core.PoleAblator` does; a hook keyed on the index within the current
forward would skip every decode step (G-gen's registered failing input). Output type (A11): in `transformers` main
the decoder layer returns a bare tensor (`Gemma3DecoderLayer.forward` ends `return hidden_states`, line 418); the
hook accepts a bare tensor, or a tuple whose element 0 is the hidden state, returns the same type, and raises on any
other type.

**Realized dose per row (A2).** Every steered sample row banks `realized_cos`, the smallest cosine between the
realized `bf16(h + v) − h` and w over position P − 1 of the prefill and decode steps 1–3. A (pole, dose) cell whose
median `realized_cos` is below 0.95 reads "instrument not shown to fire" in every arm.

**Dose units.** `RMS_l = sqrt(mean over positions p ≥ 1 and coordinates j of h_l[p, j]²)`, h_l the decoder layer l
output on corpus windows 0–15 (BOS prepended when defined; window position 0 excluded, as the harvest's mask
domain), fp32, banked once per model in `arm0/rms.json`. α is dimensionless; `α · RMS_l` is the L2 norm of the added
vector. For reference, the residual vector's L2 norm is about `sqrt(d) · RMS_l` (d = 2560 for Q4B and G4, 4096 for O7),
so α = 8 adds a vector of about 0.16 (Q4B, G4) or 0.125 (O7) of a typical residual norm. Arms 1 and 2 read RMS_l from
the same model's Arm 0 bank.

**Realized dose.** The residual stream is bf16; `bf16(h + v) − h` differs from v by rounding (relative spacing 2⁻⁷
per coordinate, about 0.1 · RMS_l in L2 norm for d = 2560). The job banks the realized vector's cosine with w and its
norm ratio at the smallest and largest dose of every stage (G-place).

**Sampling.** Temperature 0.7, `top_k = 0`, `top_p = 1.0`, `repetition_penalty = 1.0`, 128 new tokens, stop at EOS
allowed, every knob set explicitly (`neuron_off_core.generation_config`). Rows of one `generate` call share one prompt
(no padding) and at most 160 rows. Seed: `torch.manual_seed(20261025 + 1000 · prompt_i + c)` before the c-th call of
prompt_i, c counted from 0 in plan row order. Deterministic kernels throughout
(`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA starts; no
`warn_only`). Tokenization: every call `add_special_tokens=False`; the job prepends BOS when the tokenizer defines
one (G4 id 2, O7 id 100257, Q4B none).

**Coherence score.** `nll_clean_mean` = mean NLL of the completion's tokens under the unhooked model, prompt as
context, fp32 log-softmax (§5.2 uses it).

### 2.3 Arm 0 instruments (weights, plus one clean forward per model)

- **Projection.** `scores = W_U · (gain ⊙ w)` in fp32, with `gain = final_norm(ones(d))` in fp32 (the harvest's
  `vocab_table` rule) and `W_U = model.get_output_embeddings().weight`. Top 20 and bottom 20 over the attested
  vocabulary (token ids that occur at least once in the model's 50M-token corpus, minus tokenizer.json special ids:
  Q4B 105,000, G4 152,437, O7 91,951 ids; `plan/arm0_<M>.json → attested_vocab`), and the same over the full vocabulary.
  The judge sees the attested top 20 (§5.1).
- **Direct path (self-repair; amended A6).** At each NEURONOFF held-out context t of a measured pole: `delta_r` =
  the pole's contribution to the residual stream at position t only. Primary, with the scale frozen at both norms
  (Rushing & Nanda's convention extended to the post-MLP norm): Qwen 3 `delta_r = a_i(t) · W_down[:, i]`; Gemma 3
  and Olmo 3 `delta_r = gain_post ⊙ (a_i(t) · W_down[:, i]) · s_post`, with `gain_post` the layer's
  `post_feedforward_layernorm` gain and `s_post = 1 / sqrt(mean(y²) + eps)` of the clean down_proj output y at t.
  `r` = the clean input to the final norm at t; logits `W_U · (gain ⊙ (r − delta_r) · s)`, `s = 1 / sqrt(mean(r²) +
  eps)` of the clean r. Secondary rows: (a) the post-MLP norm recomputed, `delta_r = postnorm(y) − postnorm(y − a_i(t)
  · W_down[:, i])`, final scale frozen; (b) the final norm recomputed on `r − delta_r`. Gemma's
  `final_logit_softcapping` is applied when the config sets it. `direct_dnll = NLL(target | r − delta_r) − NLL(target
  | r)`, both through the same fp32 path, target = NEURONOFF's `target_id`.
- **Total effect at the context position only (A6, primary).** One extra forward per context: the pole zeroed at
  position t only (`PoleAblator` with its mask restricted to t), `tonly_dnll = NLL_ablated − NLL_clean` at t. The
  direct path and this total effect then remove the same thing, so their ratio isolates repair through later
  layers. NEURONOFF's banked `dnll` (pole zeroed at every firing position of the window) is a secondary row.

### 2.4 Before-model and after-model, pairing

Same checkpoint without and with the hook: Q4B `Qwen/Qwen3-4B-Base` @ 906bfd4b4dc7f14ee4320094d8b41684abff8539;
G4 `google/gemma-3-4b-pt` @ cc012e0a6d0787b4adcc0fa2c4da74402494554d; O7 `allenai/Olmo-3-1025-7B` branch
`stage1-step1413814` @ 373bad25002f1624757a73235c5ca844c6375c25. Weights bf16, SDPA attention, `inference_mode`.
Arm 0 pairs each pole's true label with its own shuffled labels (paired per pole) and ONE with UNK by NEURONOFF's
pairs. Arm 1 pairs a target with its matched control on the same 20 prompts. Arm 2 pairs each pole's steered judge
calls with its own α = 0 judge calls on the same candidate triple.

## 3. Design

| piece | what it adds | without it |
|---|---|---|
| Arm 0 judge, true against two shuffled labels per pole | H_write against the best null: the same list, a wrong label | "agreement" could be a judge that says yes |
| Arm 0 three-way forced choice | a check on yes-bias with a fixed chance line (1/3) | yes/no alone mixes agreement with leniency |
| RE-EXPLAIN (labels written without the output tokens) | H_write free of the leak (AUDIT §3, B1) | agreement could be copied tokens |
| fixed-label null (a pole's label against another pole's list) | removes labels that draw "yes" with any list | a generic label would look like agreement |
| self-repair ratio | sufficiency and necessity on the same poles without new ablations | NEURONOFF's small ΔNLL is uninterpretable |
| dose sweep and α_c | a dose grid set by measured collapse, the same rule in every model | a fixed grid could be all-coherent or all-collapsed |
| coherence gate, two-sided | steered text is read only where it is text | collapse would read as an on-label or off-label effect |
| Arm 1 matched controls | a pole's swing against what pushing any same-layer pole does | a swing could be generic damage |
| Arm 2 α = 0 judge null and label-prompted positive control | the best null (judge label bias) and proof the judge can fire | NEURONOFF lesson: a positive control without room |
| three models | replication across Qwen 3, Gemma 3, OLMo 3 | one family |

## 4. Data and items

### 4.1 Models, weights, corpus

Weights: Q4B and G4 from the S3 cache (`--weights models--Qwen--Qwen3-4B-Base`, `--weights models--google--gemma-3-4b-pt`;
`infra/pg/matrix_repo_facts.json:_meta.bucket_dirs`); O7 by in-job Hub download (`--no-weights`, `--revision
stage1-step1413814`). Corpus: `s3://<s3-bucket>/results/neurons/w1/q4b/1/out/corpus/corpus`, token arrays
`tok/<qwen3|gemma3|olmo3>/`; the job asserts each `tokens_sha256` equals the plan's.

### 4.2 Arm 0 items (`plan/roles_<M>.json`, `plan/arm0_<M>.json → poles`)

All class-ONE poles of the parent verdict file (Q4B 358, G4 338, O7 311; 1,007) and the `ctrl` poles of NEURONOFF's
`plan/pairs_<M>.json` (96, 97, 100; all UNK). Items: 454, 435, 411. Depth terciles of ONE poles (parent `tercile`):
Q4B 150 / 101 / 107, G4 146 / 98 / 94, O7 129 / 95 / 87. Explainer of ONE poles: Q4B 288 Opus / 70 Sonnet, G4 282 / 56,
O7 233 / 78. RE-EXPLAIN covers every ONE and every UNK pole of the audit: 550, 557, 560 requests (1,667).

### 4.3 Self-repair poles (`plan/arm0_<M>.json → repair`)

Every NEURONOFF Arm A pole in `bank/off/<q4b|g4>/1/out/poles/` that passed G-ident and G-fit: Q4B 156 poles (15,600
contexts in 9,540 windows), G4 201 poles (20,100 contexts in 11,549 windows; G4 L16N8452+ failed G-fit). O7 has no
NEURONOFF bank and no self-repair row. G-pc (NEURONOFF's per-pole positive control) is not used: its floor had no
room (NEURONOFF review finding 6).

### 4.4 Known-number poles (`plan/arm0_<M>.json → known`)

Q4B L31N6653+ and G4 L30N10175+: the NEURONOFF Arm A pole with the largest `own_dnll` mean among poles passing
G-ident, G-fit and G-pc. Banked over 100 contexts: own clean NLL mean 1.3216 / 2.8553 nats, own ΔNLL mean 0.3082 /
2.5112 nats. O7 L23N1402- (first Arm A pole of NEURONOFF's O7 plan): harvest `a` at its 100 held-out contexts, plus
the harvest's mean NLL over the first 120,000 corpus targets, 2.675372037760417 nats.

### 4.5 Arm 1 poles (`plan/roles_<M>.json → arm1`, `plan/arm1_<M>.json`)

Kept (NEURONOFF Arm B targets with their ctrl1, d as NEURONOFF §5.2): Q4B L11N5238+ (SQL FOREIGN KEY) / L11N4144+
(0.61); Q4B L27N2924- (Angular) / L27N7674+ (0.06); Q4B L08N3398- (HTML title attribute) / L08N9674- (1.43); G4
L07N10202- (React useEffect) / L07N5834- (0.46); G4 L06N599+ (Python import … as) / L06N8972- (0.45); G4 L08N7823+
(SQL REFERENCES) / L08N6441+ (0.43). Prompts: `studies/neuronoff/prompts/<pole>.jsonl` (20 each). TeX L04N5165+
never fires on its prompts (AUDIT §2) and is replaced by the §5.5 rule from this frozen candidate list (Q4B ONE poles
whose label matches `LaTeX|\bTeX\b|backslash`, ordered by the parent's `blind_max_score`, descending): L16N9243+
(0.786; control L16N2662-, d 0.34), L29N34+ (0.622; L29N8202-, 1.43), L35N1881- (0.564; L35N7677+, 0.38). Candidate
controls: the nearest unused UNK pole of the layer by NEURONOFF's distance, no caliper, never the other pole of the
same neuron (it writes −w). A control with d > log 2 is printed "poor control match" (L08N9674-, L29N8202-).

### 4.6 Arm 2 items (rule frozen now in `plan/arm2_rule.json`; drawn after the Arm 0 judge, §9.4 L9)

80 ONE poles and 40 matched UNK poles per model, drawn by the rule quoted in §5.7. The population excludes the
sweep poles, the Arm 1 targets and the TeX candidates.

### 4.7 Fixed prompt lists

Neutral prompts (Arm 2, the sweep, and the coherence calibration): `"Here is some text:\n"`, `"The following was
written yesterday.\n\n"`, `"Notes\n\n"`, `"Here is a document:\n\n"`, `"Text:\n"`. Judge prompts:
`plan/judge_prompts.json` (system and user templates, response schemas, render rules), quoted in §5.8.

### 4.8 Plans, hashes, reproducibility

`python3 -I studies/neuronrw/plan/make_plan.py --out <dir>` rebuilds every plan file byte for byte (checked
2026-10-26 after A1–A14: two runs, 17 of 17 files identical). `plan/budget.json → models.<M>.plan_sha256` records the hashes. The
builder reads the parent verdict file, the harvest banks, NEURONOFF's pairs and plans and banks, and the w2c
explanation files; each input's path and sha256 is in `roles_<M>.json → inputs`.

## 5. Arm measurements

### 5.0 RE-EXPLAIN (amended A1; laptop, API, before any GPU run)

The parent's explainer saw the output tokens (AUDIT §3), and no word-matching rule can tell copied tokens from real
agreement. So every ONE and every UNK pole of the audit (Q4B 550, G4 557, O7 560; 1,667 requests) is explained again
without them:
- Request: the parent's own explainer request for the pole (`../interp_research-neurons-explain/bank/w2c/<cell>/explain/
  requests/c1-explainer.jsonl`, id `explainer:<pole>:0`, the request whose answer is the verdict's label), copied
  verbatim with the same 30 contexts, statistics, schema, temperature (1.0) and `max_tokens` (16000), except: the two
  user lines that start with the strings in `USER_VOCAB` are deleted, the system clause `SYS_CLAUSE` is replaced by
  ".", and the system bullet `SYS_BULLET` is deleted, so the prompt never points at absent data. The exact strings
  are in `studies/neuronrw/plan/make_plan.py`, lines 48–52 (A16); `make_plan.py:reexplain_requests` makes these edits
  and asserts each is found exactly once.
- B1 variant (A16): for the 1,007 ONE poles a second request, the parent request verbatim with the vocabulary lines
  kept and only the model replaced (batch `rxv-<M>`, ids `explainer:rxv:<pole>`; Q4B 358, G4 338, O7 311). The main batch's ids are
  `explainer:rx:<pole>`: `neurons_llm.make_request` requires every id to start with its role (A19). Both labels then come
  from the same model, so B1 compares seeing against not seeing the tokens and nothing else. The original labels are
  80% Opus and were selected for passing the blind test, so original minus re-explained would mix explainer and
  regression to the mean into B1.
- Model: read from the parent's record file
  `../interp_research-neurons-explain/bank/w4/o7_gate_v2_sonnet_js/explain/responses/w4j-judge-s0.backend.json`
  (`reported_models[0]`, asserted equal to `settings.models.explainer` and `settings.models.judge`), never typed;
  backend `openai_compat` and base URL from the same file. The parent's Opus explainer is not callable through an API
  (parent policy, `infra/pg/neurons_llm.py` `OPUS_RULE`), so every re-explanation comes from that one model.
- Frozen record: `plan/reexplain_<M>.json` (request ids, the sha256 of the canonical request lines, the edits, the
  sources, and a `with_vocab` block with the variant's ids and its own sha256). `neuron_rw_judge build --step RX` rebuilds the requests by calling `reexplain_requests` and asserts the
  sha256. Requests, responses, sidecars and the call log are banked in `S/bank/rw/reexplain/` (neurons_llm layout) and
  copied to `s3://<s3-bucket>/results/neurons/rw/reexplain/1/`; the collected answers are committed as
  `results/reexplain_<M>.jsonl` and `results/reexplain_vocab_<M>.jsonl` (`pole`, `explanation`, `class`, `confidence`,
  `jobs`, `model`).
- Use: every Arm 0 and Arm 2 judge request shows the re-explained `explanation` as the pole's label (label set
  "reexplained"). The variant's label (set "with_vocab") enters only B1; the parent's label (set "original") enters
  only secondary rows. A pole with no
  re-explanation after neurons_llm's attempts is left out of the primary rows and counted. The re-explained class is
  reported against the parent's class; a secondary row keeps only poles re-explained as ONE.

### 5.1 Arm 0 — read against write (weights only; the judge runs on the laptop)

Per item p (frozen draws in `plan/judge_draws_<M>.json`; amended A8): five distinct ONE poles of p's parent tercile,
never p or the other pole of p's neuron, are drawn once: `shuf1`, `shuf2`, `fixed_list`, and two fc3 distractors.
- `yn_true`: p's attested top-20 list against p's label; `yn_shuf1`, `yn_shuf2`: the same list against the labels of
  `shuf1` and `shuf2`; `yn_fixed`: p's label against the top-20 list of `fixed_list` (the fixed-label null).
- `fc3`: p's list against three labels, p's and the two distractors', in the frozen letter order (`fc3_truth`; truth
  letters balanced A / B / C over items).
- Label sets: all five requests with re-explained labels (primary); `yn_true`, `yn_shuf1`, `yn_shuf2` again for ONE
  items with the with-vocabulary variant's labels (B1, A16); the same three again with the parent's original labels
  (secondary only).
- Retest: a frozen 10% of items (`retest`) repeat `yn_true` and `fc3` (re-explained) with new ids.
- Positive control G-judge0: for 100 frozen ONE poles (`judge0_pc`), the pole's own firing-token list (the first 20
  distinct decoded strings of the tokens at its top-1000 positions in rank order; `arm0/firing_tokens.jsonl`) against
  its re-explained label (registered passing input) and against `shuf1`'s (registered failing input).

Per item and label set: `y_true`, `y_s1`, `y_s2`, `y_fixed` ∈ {0, 1} (1 = "yes"); `s = y_true − (y_s1 + y_s2) / 2`;
`f = y_true − y_fixed`; `c3` = 1 if the fc3 choice is the truth letter. A missing or declined answer leaves the item
out of every row that needs it; counts are printed.

Literal-overlap row (secondary; A1). For each item and label set, `literal` is true if the label names a token of the
pole's judged top-20, by this rule (implemented once, `neuron_rw_core.literal_overlap`, with a test): label and token
are NFKC-normalized and casefolded; the token is stripped of whitespace and of leading and trailing characters of
Unicode categories P and S; a token with no character of category L or N is skipped; a token holding a character of a
script written without spaces (Unicode name containing CJK, HIRAGANA, KATAKANA, HANGUL, THAI, LAO, KHMER, MYANMAR or
TIBETAN) matches if it occurs anywhere in the label; otherwise the label is split into words (maximal runs of L and N
characters), and the token t matches a word w if t = w, or one suffix from (ations, ation, ings, ing, ers, er, ed, es,
ly, s) removed from each leaves the same string of at least 3 characters, or t has at least 4 characters and w starts
with t, or w has at least 4 characters and t starts with w; tokens of 1 character are skipped and tokens of 2–3
characters match only by t = w. D_w is printed for literal and non-literal items in every label set.

### 5.2 Coherence gate (Arms 1 and 2 and the sweep; amended A7)

Band per arm and model, frozen on control data before any steered sample is read. The arm's α = 0 samples are split
by `default_rng([20261025, model_index, 12]).permutation` (in file order): the first half calibrates, the second half
tests.
- `L` = 2.5th percentile of `nll_clean_mean` over the calibration half.
- `U` = the larger of the 97.5th percentile over the calibration half and the 97.5th percentile over the natural
  calibration spans (corpus windows 0–15, 16 spans of 32 prompt + 96 continuation tokens each, 256 spans).
- α = 0 samples per band: sweep — the sweep's 100; Arm 1 — that model's Arm 1 α = 0 samples (Q4B 800, G4 600); Arm 2 —
  the sweep's 100 and Arm 2's 50 pooled (same prompts, same sampling).

A sample is coherent if `n_new_tokens ≥ 8` and `L ≤ nll_clean_mean ≤ U`; otherwise it is collapsed. Collapsed samples
are counted per pole and dose and printed beside every readout as "coherent n / total"; they are never dropped
without the count. The floor L is a tightening of the brief's one-sided rule: repetition loops have low NLL under the
clean model and would pass a ceiling-only gate. U includes natural text because samples at temperature 0.7 have
lower NLL than human text, so a margin from α = 0 samples alone would mark coherent off-prompt text as collapsed.

Registered inputs, run by every generating job on real weights and banked in `samples/coherence_inputs.jsonl`:
failing input 1, neutral prompt 0 followed by the first sweep pole's attested top-1 token repeated 96 times, must be
collapsed; failing input 2, neutral prompt 0 followed by 96 attested ids drawn by `default_rng([20261025,
model_index, 10])`, must be collapsed; passing input 1, the natural test spans (windows 16–31, 256 spans), coherent
share ≥ 0.90; passing input 2, the test half of the arm's α = 0 samples, coherent share ≥ 0.90; passing input 3
(Arm 2 only), the 400 label-prompted continuations, coherent share ≥ 0.90. Any of them landing wrong cancels that
arm's coherent-only readouts for the model ("coherence gate not shown to work"); the all-sample curves are then
reported as exploratory.

### 5.3 Dose scale α_c (from the Arm 0 sweep)

Sweep: 10 frozen ONE poles per model (`arm0_<M>.json → sweep`), doses ±1, ±2, ±4, ±8, ±16, ±32, the 5 neutral prompts,
4 samples per prompt (200 samples per dose over the 10 poles), and 20 α = 0 samples per prompt. `C(α)` = collapsed
share at dose α pooled over the 10 poles (sweep band). Rule (`neuron_rw_analysis dose_scale`):
- `α_c` = the smallest positive sweep dose with `C(+α) ≥ 0.5`; `α_c = 1` if `C(+1) ≥ 0.5`; `α_c = 64` if no dose up to
  32 reaches 0.5.
- Arm 1 grid = `α_c × {−1, −1/2, −1/4, −1/8, 0, 1/8, 1/4, 1/2, 1}` (the brief's grid when α_c = 8).
- Arm 2 doses = `sorted({2, 4} ∪ {α_c / 2})`; primary dose `d_p = α_c / 2`, the largest sweep dose below the collapse
  point (the brief's "+2, +4 and the first dose above the knee if different", stated as a rule).
- Written to `plan/dose_scale_<M>.json`; `make_plan.py` then rewrites `arm1_<M>.json` (byte-identical when α_c = 8);
  both files are committed with a JOURNAL line before the arm's launch.
- The sweep is the first stage work of the Arm 0 job, right after the gates (A9), so a budget stop cannot lose it.

**Held-out prediction for α_c from Liu 2026 (A12).** Liu (arXiv 2606.19831, "Leverage Is Not Reach: A Control-Window
Law for Single-Neuron Steering in Language Models"), read 2026-10-26 for this claim: the intervention is
`r(t) = r_L + t·v` (Eq. 1); the budget is `g = t‖v‖ / ‖r_L‖ = t / B` with `B = ‖r_L‖ / ‖v‖` (Eq. 3); the alignment
follows `cos θ_v = g / sqrt(1 + g²)` (Eq. 5); collapse sits at `t* = m*·B`, with m* "the collapse magnitude in units of
‖r_L‖", measured per architecture (Table 3: Qwen3-4B mean m* = 1.37, Gemma-3-4B mean m* = 0.31; collapse located by a
distinct-bigram coherence ratio, not by this study's band). In this study's units the added norm is `α · RMS_l`, so
collapse is predicted at `α* = m* · ‖r_l‖ / RMS_l`. With `‖r_l‖ ≈ sqrt(d) · RMS_l` (d = 2560): α* ≈ 1.37 × 50.6 = 69
for Q4B and 0.31 × 50.6 = 15.7 for G4. On the sweep grid that predicts **α_c(Q4B) = 64** (no sweep dose up to 32 reaches
half collapse) and **α_c(G4) ∈ {16, 32}** (A17: the predicted 15.7 sits 2% below the grid dose 16, m* is a mean over
neurons, and Liu's collapse criterion differs from this study's NLL band, so the next grid dose is admitted). `dose_scale` also prints α* recomputed from the banked `mean_norm / rms` of the
sweep poles' layers. Liu gives no OLMo 3 value, so O7 has no α_c prediction. The prediction is held out: α_c is set by
the §5.3 rule, never by this formula.

### 5.4 Arm 1 — the volume knob

Per pair (target, control) and prompt set: 20 prompts × 10 samples at each of the 9 doses for each pole; the α = 0
samples are generated once and shared (3,400 samples per pair). Construct counts by NEURONOFF's frozen regexes
(`studies/neuronoff/SPEC.md` §5.4, copied into `roles_<M>.json → arm1[].regex`) on `prompt[-200:] + completion`; a
match counts only if it ends inside the completion. `R(α)` = share of coherent samples with at least one match (the
TeX prompt set: mean match count per coherent sample). `C(α)` = collapsed share.

`d*` = the largest positive grid dose d with `C(+d) ≤ 0.5` and `C(−d) ≤ 0.5` for both the target and the control.
Swing `S = R(+d*) − R(−d*)` for each pole. The brief's "swing across the coherent range" is read as the difference
between the two ends of the largest dose range coherent for both poles; a maximum minus minimum over doses would be
biased upward by noise. Every curve R(α), C(α) of both poles is banked and plotted.

Paired interval (amended A3): target and control share the 20 prompts, so each bootstrap replicate resamples the 20
prompts once, recomputes `S_t` and `S_c` from the coherent samples of the drawn prompts, and keeps `S_t − S_c`
(10,000 replicates). SWINGS needs abs(S_t) ≥ 0.20 (count readout: 1.0) and the interval of `S_t − S_c` on the side of
S_t (lo > 0 for S_t > 0, hi < 0 for S_t < 0). NO SWING needs the interval to contain 0 and lie inside ±0.10 (count:
±0.5). Everything else is UNRESOLVED.

Ceiling case. Construct presence at α = 0 is 0.76–0.985 for six of the seven prompt sets (AUDIT §2). If R ≥ 0.95 at
both +d* and −d* for both poles, an increase cannot be seen, and a registered input that plants an increase cannot
fail to read NO SWING. Such a pair prints "at ceiling: only a decrease could be seen" beside its label, and its room
check plants a decrease (§7.3).

### 5.5 TeX replacement rule

The Q4B Arm 0 job measures, for each Arm 1 target, each TeX candidate (§4.5) and L04N5165+, `sign · a` at every
position ≥ 1 of BOS + prompt + continuation for the 20 prompts of its set (`arm0/firing.jsonl`). A prompt fires if the
maximum is ≥ 0.3 of the pole's banked maximum `|a|` (NEURONOFF's R3 threshold). The first candidate in frozen order
with at least 15 firing prompts replaces L04N5165+ and takes the TeX prompt set and its count readout. If none
qualifies, the TeX slot is dropped and reported "dropped: no candidate fires on at least 15 of 20 prompts".
`neuron_rw_analysis replacement` writes `plan/arm1_replacement_Q4B.json`; `make_plan.py` adds the pair to
`arm1_Q4B.json`. Kept targets are not re-selected; their firing count is printed beside their label.

### 5.6 Arm 1 pre-registered predictions (written 2026-10-25, before any GPU run)

The readout is `S` (presence points; TeX set: macros per completion). "Control" = the matched control's |S|.

Predicted labels use the amended rule (A3): "NO SWING" below means abs(S) < 0.10 predicted; the label then reads
UNRESOLVED if the paired interval is wider than ±0.10.

| pole | label | predicted label | predicted abs(S) | predicted sign of S | control |
|---|---|---|---|---|---|
| Q4B L11N5238+ | SQL FOREIGN KEY | NO SWING | < 0.10 | — | < 0.10 |
| Q4B L27N2924- | Angular | SWINGS | ≥ 0.20 | + | < 0.10 |
| Q4B L08N3398- | HTML title attribute | NO SWING | < 0.10 | — | < 0.10 |
| G4 L07N10202- | React useEffect | NO SWING | < 0.10 | — | < 0.10 |
| G4 L06N599+ | Python import … as | NO SWING | < 0.10 | — | < 0.10 |
| G4 L08N7823+ | SQL REFERENCES | NO SWING | < 0.10 | — | < 0.10 |
| Q4B TeX replacement | TeX prompt set | SWINGS if L35N1881- is chosen (a last-layer pole whose label names LaTeX openers); NO SWING otherwise | ≥ 1.0 macro if L35N1881- | − (it suppresses backslash openers) | < 0.5 macro |

Summary predictions: at most 2 of 7 targets SWING (probability 0.6); every control has |S| < 0.10 (probability 0.5);
the dose that collapses half the sweep samples, α_c, is at least 16 in every model (probability 0.6).

**Missed predictions are reported.** The result document carries a section "Missed predictions" that lists, one row
each, every prediction in this section and in §11 whose observed value lands outside the predicted one, with the
artifact path and key (NEURONOFF review finding 5). A prediction is never dropped from the grid.

### 5.7 Arm 2 — sufficiency at scale (amended A8, A10, A11)

Draw (rule frozen in `plan/arm2_rule.json`; executed by `plan/make_plan.py` itself once `plan/dose_scale_<M>.json`,
`results/judge_a0-yn-<M>.jsonl` and `results/reexplain_<M>.jsonl` are committed, so the draw is byte-reproducible and
its asserts run in the same code as every other plan):
- Population: the model's ONE poles minus the sweep poles, the Arm 1 targets and the TeX candidates, each with an A0
  `yn_true` answer on its re-explained label and a re-explained label.
- Strata: that answer (agree = yes, disagree = no) × parent tercile, in the order agree-0, agree-1, agree-2,
  disagree-0, disagree-1, disagree-2. Quotas 14, 14, 13, 13, 13, 13 (80). A stratum short of its quota gives all its
  poles; the total shortfall is then given one pole at a time to the strata in stratum order, cycling from agree-0,
  skipping strata with no undrawn pole.
- Draw: `default_rng([20261025, model_index, 7]).permutation` over each stratum's poles sorted by id, strata in
  order; the first quota poles are drawn; draw order is round robin over strata.
- Controls: in draw order, each eligible drawn ONE pole takes the nearest unused eligible UNK pole of its layer
  (NEURONOFF distance, no caliper, never the other pole of the same neuron — asserted in `make_plan.py`); poles without
  one are skipped; stop at 40.
- Distractors (A8): two ONE poles of the pole's parent tercile, never the pole, its matched partner or the other pole
  of its neuron, drawn by `default_rng([20261025, model_index, 8])`; the same triple and letter at every dose and at
  α = 0; truth letters balanced by `default_rng([20261025, model_index, 9])`.
- Labels: re-explained (§5.0) in every request and label prompt. `model_index` Q4B 0, G4 1, O7 2.
- Tested on synthetic answers on 2026-10-26 (scratch only): 80 ONE and 40 UNK poles drawn, letters 40 / 40 / 40, the
  shortfall path exercised with 3% "yes" answers.

Samples: every Arm 2 pole at every Arm 2 dose (§5.3), 5 neutral prompts × 10 samples; α = 0: 5 × 10 (shared by all
poles); label-prompted positive control: the first 40 drawn ONE poles, prompt `"Here is some text about the
following: {label}\n\n"`, 10 samples at α = 0.

Judge calls: for each pole, dose (including α = 0) and prompt with at least 3 coherent samples, one `a2_fc3` request
showing every coherent sample of that cell (at most 10) and the pole's triple; at `d_p` also one `a2_guess` request on
the same texts, sent as a separate request without candidates. Retest: 10% of the `a2_fc3` cells drawn by
`default_rng([20261025, model_index, 13])` (A11). `o_p(α)` = share of the pole's judged prompts at dose α whose
choice is the truth letter. A pole contributes at α if at least one prompt was judged there.

Intervals (A10): two-way bootstrap. Each replicate resamples poles; for each drawn pole it resamples that pole's
judged prompts at the steered dose; and, because the α = 0 texts are shared by every pole, it resamples the 5 α = 0
prompts once per replicate (their own draw) and uses that draw for every pole's α = 0 term. Means, paired differences
and ratios of §7.4 are computed inside each replicate.

### 5.8 Judge protocol (all judge steps)

- Backend and model, read from the parent's own judge records, never typed (amended A1, A11): `plan/make_plan.py:
  judge_settings` reads
  `../interp_research-neurons-explain/bank/w4/o7_gate_v2_sonnet_js/explain/responses/w4j-judge-s0.backend.json`
  (`backend`, `settings.base_url`, `settings.key_env`, `settings.temperature`, `settings.response_format`, and the model
  as `reported_models[0]`, asserted equal to `settings.models.judge` and `settings.models.explainer`) and
  `…/explain/requests/w4j-judge-s0.jsonl` (`max_tokens`, one value over its 360 requests, asserted), and writes them with
  both files' sha256 into `plan/judge_prompts.json → judge`. As built on 2026-10-26: `openai_compat`,
  `https://api.deepinfra.com/v1/openai`, key env `NEURONS_API_KEY_1`, temperature 1.0, `max_tokens` 8000, model
  `anthropic/claude-sonnet-5-5` (720 judge calls in that parent cell). The parent's labels came from that model and
  from `anthropic/claude-opus-5-5` (`bank/w2c/<cell>/explain/calls.jsonl`); Opus is not callable through the API under
  the parent's policy. The code writer copies the record's `settings` object verbatim to
  `infra/pg/settings/deepinfra_neuronrw_judge.json`; `neuron_rw_judge build` asserts every request's model, temperature
  and `max_tokens` equal `judge_prompts.json → judge`.
- Model check before each judge step: the ID from `judge_prompts.json → judge.model` must appear in `GET
  https://api.deepinfra.com/v1/openai/models`, and every response must report the request's model (`neurons_llm`
  collect rule). Otherwise the step stops and the director is told; no other model is substituted. Checked by the spec
  writer at 2026-10-26T03:10Z: the public model list (175 ids) contains it.
- Client: `infra/pg/neurons_llm.py` from `neurons/explain` (unchanged): request, response, backend sidecar, usage and
  call-log formats as SCHEMA §4.
- Blind: the judge sees only token strings or continuations and label texts. No pole id, class, layer, dose, model
  name or study name. Request ids are opaque; request order is a seeded permutation; candidate letters are balanced.
- Prompts: `plan/judge_prompts.json`, frozen. Arm 0 yes/no system text: "You compare two descriptions of one unit
  inside a language model. The first is a list of 20 vocabulary tokens, strongest first, that the unit's output pushes
  the model to predict. … Answer "yes" if the token list clearly relates to the described inputs: the same words or
  word pieces, the same topic, the same language or syntax, or tokens that would naturally come right after the
  described inputs. Otherwise answer "no"." The file holds the full texts of `a0_yn`, `a0_fc3`, `a2_guess`, `a2_fc3`.
- Chance: 1/3 for every fc3 task. Null for yes/no: the same list with a shuffled label.
- Who runs it: the director, as separate steps (§9.5); the executor builds requests and collects answers.

## 6. Instrument readouts and gates

Every gate has a floor, a ceiling, a cancelled branch, a registered failing input (must fail) and a registered
passing input (must pass). Both inputs run on real data in the preflight cell `pre_q4b` (model gates, G-proj,
G-repair-fit, G-coh) or in the judge preflight batches J0 and J2 (judge gates), and again in every full job or judge
step; both results are banked beside the gate. In the Arm 0 job the gates run first, then the sweep (A9). NaN or a missing value takes the cancelled branch. A job aborts
(`status: aborted_gate`) if a model gate fails, a registered failing input passes, or a registered passing input
fails; pivot P1 (§9.6) applies.

| gate | scope | pass condition (floor / ceiling) | cancelled branch | registered failing input (must fail) | registered passing input (must pass) |
|---|---|---|---|---|---|
| G-det | every job | clean forward twice on corpus windows 0–1: max abs Δlogit = 0 / 0 | job aborts | window 1 with the token at position 100 replaced by (id + 1) mod V: max abs Δlogit > 0 after position 100 | the clean repeat itself |
| G-ident | every job | steering hook at α = 0 against no hook, natural calibration spans 0–1, steering from P − 1: max abs Δlogit = 0 / 0 | job aborts | the same hook at α = 1: max abs Δlogit > 0 | the hook at α = 0 |
| G-place | every job, first 3 pc poles, natural span 0 | realized `Δh = bf16(h + v) − h` at the hooked layer output, median over steered positions, at α = 1 and at the stage's largest dose: cos(Δh, w) ≥ 0.95 / ≤ 1 + 1e-6 and ‖Δh‖ / (α · RMS_l) in [0.9, 1.1] | job aborts | the same vector added to the MLP input (forward pre-hook on `mlp`): cos < 0.95 or ratio outside [0.9, 1.1] at the layer output | the decoder-layer output hook. Reported, not gating: the down_proj-output placement in G4 and O7 (the saturating location of LIT_steering §C1), with its ratio |
| G-gen (A2) | every generating job, first 3 pc poles, neutral prompt 0, 4 rows | realized Δh at the hooked layer output at decode steps 1, 2 and 3 of a cached `generate`, at α = 1 and at the stage's largest dose: cos(Δh, w) ≥ 0.95 / ≤ 1 + 1e-6 and ‖Δh‖ / (α · RMS_l) in [0.9, 1.1] at every step | job aborts | a hook keyed on the index within the current forward (it adds at prefill position P − 1 and never at a decode step): Δh = 0 at steps 1–3, so the gate fails | the absolute-position hook. Per (pole, dose) cell: median `realized_cos` < 0.95 → "instrument not shown to fire" (§2.2) |
| G-known | every job | Q4B, G4: median abs(a_ours / a_banked − 1) ≤ 0.01; abs(own clean NLL mean − banked) ≤ 0.002 nats; abs(own ΔNLL mean − banked) ≤ 0.002 nats, on the plan's known contexts (100 in Arm 0, first 20 later); O7: median rel. error of a ≤ 0.03 and abs(mean NLL over the first 120,000 targets − 2.675372037760417) ≤ 0.005; floor 0 (an exact match is allowed and flagged) | job aborts; P1 | layer 0's whole MLP output zeroed on the mask domain (`neuron_off_core.PoleAblator` mode `mlp_zero`): the clean NLL mean (O7: the 120,000-target mean) must move by at least 0.5 nats; a smaller shift aborts the job (NEURONOFF moved the 120,000-target mean by 1.852 and 1.712 nats) | the unhooked model |
| G-floor-dir | every job | a random unit direction (`default_rng(20261025)`, normal, normalized) at α = 0 on the natural spans 0–1: max abs Δlogit = 0 / 0 | job aborts | the same direction at the stage's largest dose: > 0 | the direction at α = 0 |
| G-floor-pos | every job | the real hook at the stage's largest dose: max abs Δlogit over positions < P − 1 = 0 / 0 | job aborts | the hook started at position 0: > 0 at positions < P − 1 | the real hook |
| G-pc (pooled positive control) | every job | mean continuation ΔNLL, steered at the stage's pc dose (Arm 0: 32; Arms 1, 2: α_c) minus unhooked, over 50 frozen audit poles × natural spans 0–9 × 96 positions: ≥ 0.05 / ≤ 15 nats | job aborts ("instrument not shown to fire") | α = 0: 0 < 0.05 | the pc dose; room to pass: the pc dose is the sweep dose at which half the samples collapse or above, so the logits move far |
| G-coh (A7) | every generating job | §5.2: both collapse inputs flagged; natural test spans ≥ 0.90 coherent; the test half of the α = 0 samples ≥ 0.90 coherent; Arm 2 also the 400 label-prompted continuations ≥ 0.90; L < U | the arm's coherent-only readouts cancelled for that model | repeated-token and random-token strings (must be marked collapsed) | natural test spans, the test half of the α = 0 samples, and (Arm 2) the label-prompted continuations |
| G-proj (Arm 0, analysis) | model | share of Arm 0 items whose full-vocabulary top-10 decoded strings equal the harvest `vocab.json` promoted list in order ≥ 0.99 / ≤ 1; median abs(score · ‖W_down[:, i]‖ / banked − 1) ≤ 1e-3 | Arm 0 labels cancelled | our bottom-10 (the opposite sign) against the banked promoted list: share < 0.99 | the correct sign |
| G-repair-fit (Arm 0, Q4B and G4) | pole | the t-only ablation forward run with the identity hook gives max abs Δlogit = 0 at t; median abs(a_ours / a_off − 1) ≤ 0.01; median abs(clean NLL − NEURONOFF clean NLL) ≤ 0.002; manual fp32 path on the clean final residual against the model's own NLL: median abs difference ≤ 0.01 and 95th percentile ≤ 0.05 nats | pole leaves the repair rows (counted); more than 30% of a model's poles out cancels its repair labels | the manual path applied to the residual entering layer n_layers − 2 instead of the final norm: median abs difference > 0.5 | the final residual |
| G-judge0 | model, Arm 0 | yes-rate of firing-token lists against their re-explained labels ≥ 0.60; ceiling: yes-rate of `yn_shuf1` and `yn_shuf2` in the main batch ≤ 0.50 | Arm 0 labels cancelled ("judge not shown to fire" or "judge not discriminating") | firing-token lists against `shuf1` labels: yes-rate < 0.60 | firing-token lists against true labels |
| G-judge2 | model, Arm 2 | on-label rate of the label-prompted continuations (40 poles) ≥ 0.60; ceiling: on-label rate of all α = 0 calls ≤ 0.50 | Arm 2 labels cancelled | α = 0 neutral continuations of the same 40 poles: rate < 0.60 | label-prompted continuations |
| G-retest | judge task (a0 yn, a0 fc3, a2 fc3) | Cohen's kappa between original and retest answers ≥ 0.50 / ≤ 1 | the task's labels stamped "judge unreliable (kappa k)" | the retest answers against the originals of a seeded derangement of the items (`default_rng([20261025, model_index, 11])`): kappa < 0.50 | the same items |
| G-n (A9) | model | Arm 0: ≥ 300 items with all five primary answers; repair: ≥ 20 qualifying poles (H_repair), ≥ 15 qualifying pairs pooled (H_repair_ONE); Arm 2: at least 0.7 of the drawn ONE poles (56 of 80; 42 of 60 after ladder step 5) contributing at d_p, at every budget ladder step | Arm 0 / repair: "underpowered"; Arm 2: "no coherent dose" (H_suff cancelled for that model) | the same check one below its line (299 items / 19 poles / 14 pairs / ceil(0.7 n) − 1 poles): must fail | the full set |

Readouts carried by every number before unblinding (CLAUDE.md Instruments §4): fit on both sides (G-known and
G-repair-fit for measured poles; G-proj for projections), a positive control that fired (G-pc for steering; G-judge0
and G-judge2 for judges), and the value on an identity pair (G-ident). G-place and G-gen together audit the realized
vector at prefill and at decode steps (CLAUDE.md Instruments §8). An arm missing any of the three for a model is
cancelled and stamped in its verdict file.

## 7. Frozen decision rules and statistics

All intervals: 95% percentile bootstrap, 10,000 resamples, `numpy.random.default_rng(20261025)` fresh per interval,
numpy-linear 2.5th and 97.5th percentiles. Units are stated on both sides of every comparison.

### 7.1 Arm 0 (per model; H_write and H_depth on ONE items; H_ONE on NEURONOFF pairs; re-explained labels)

- `A` = mean y_true (agreement rate), `Sh` = mean (y_s1 + y_s2) / 2 (shuffle rate), `Fx` = mean y_fixed, each with an
  interval: absolute rates.
- `D_w` = mean over ONE items of s; `D_f` = mean of f; intervals over ONE items.
- `δ_w` (frozen in `thresholds_arm0_<M>.json` from shuffled answers only) = 97.5th percentile over 10,000 bootstrap
  resamples of ONE items of `abs(mean(y_s1 − y_s2))`, without sign flips.
- Labels (A4): **H_WRITE** lo(D_w) > 0, D_w ≥ δ_w and lo(D_f) > 0; **ABOVE SHUFFLE, BELOW MARGIN** lo(D_w) > 0 and
  D_w < δ_w; **ABOVE SHUFFLE, NOT ABOVE FIXED-LABEL NULL** lo(D_w) > 0, D_w ≥ δ_w, lo(D_f) ≤ 0; **REFUTED** hi(D_w) ≤ 0;
  **EQUIVALENT** the D_w interval inside ±0.10; **NOT SHOWN** otherwise. A D_w interval containing δ_w prints "interval
  crosses margin".
- H_depth: `D_w(tercile 2) − D_w(tercile 0)` over ONE items, each tercile resampled separately. **SUPPORTED** lo > 0;
  **REFUTED** hi ≤ 0; **EQUIVALENT** inside ±0.15; **NOT SHOWN** otherwise.
- H_ONE: pairs = NEURONOFF `pairs`; `d_k = s(one_k) − s(ctrl_k)`. Primary row pooled over the three models (n = 293,
  resampled within model); per-model rows secondary. **SUPPORTED** lo > 0; **REFUTED** hi ≤ 0; **EQUIVALENT** inside
  ±0.10; **NOT SHOWN** otherwise. Secondary: the same without the 22 same-neuron pairs.
- B1 (leak; A16): `L` = paired mean over ONE items of s(with_vocab) − s(reexplained), pooled over the three models
  (resampled within model); labels as the hypotheses table.
- fc3: accuracy per model and class with interval, read against 1/3; letter shares of the choices (B3).
- Secondary rows (frozen; never the headline): the original and with-vocabulary label sets (A, Sh, D_w); the
  literal-overlap split of §5.1 in every label set; poles re-explained as ONE only; by pole sign (F41); by explainer model; by tercile; A, Sh and Fx per
  tercile; ONE against UNK in absolute A.
- Study label: SUPPORTED if at least two models give the supported label and no model gives REFUTED; REFUTED if at
  least two models give REFUTED and none gives the supported label; EQUIVALENT if at least two give EQUIVALENT and none
  gives a directional label; otherwise MIXED. Cancelled models do not count; with two counting models both must agree.

### 7.2 Self-repair (Q4B, G4; amended A6)

- Split: in each pole's held-out list (k = 0..99, rank order), odd k selects and even k estimates.
- Per pole, on the odd contexts: `D_odd` = mean `direct_dnll` (both scales frozen). Qualifying: the interval of
  `D_odd` over its 50 odd contexts has lo > 0.
- Per qualifying pole, on the even contexts: `T_p` = mean `tonly_dnll` (total effect, pole zeroed at the context
  position only, nats), `D_p` = mean `direct_dnll`; `r_p = T_p / D_p` when `D_p > 0`, and `r_p = +∞` when `D_p ≤ 0`
  (the direct effect did not replicate; counted against repair). `M_r` = median of r_p with interval over poles, per
  model and pooled (resampled within model); an infinite bootstrap median makes that end infinite.
- Labels: **H_REPAIR** hi(M_r) < 0.5; **REPAIR, NOT BELOW 0.5** 0.5 ≤ hi < 1; **REFUTED** lo(M_r) ≥ 1; **EQUIVALENT TO NO
  REPAIR** interval inside [0.8, 1.25]; **NOT SHOWN** otherwise.
- H_repair_ONE: NEURONOFF pairs with both poles measured and qualifying; `r_one − r_ctrl` (∞ − ∞ and finite − ∞ pairs
  are dropped and counted); median, pooled over Q4B and G4 (primary) and per model. **SUPPORTED** hi < 0; **REFUTED**
  lo ≥ 0; **EQUIVALENT** inside ±0.20; **NOT SHOWN** otherwise.
- Always printed: median T_p and D_p per class with intervals (absolute scores), the qualifying count out of measured,
  the count with D_p ≤ 0, the secondary rows (post-MLP norm recomputed; final norm recomputed; NEURONOFF's
  all-position total effect in place of the t-only one), and the share of G4 layer 10–11 contexts at window
  position 1 (F42).
- The lines 0.5 and 1 are the brief's, fixed, not tuned; the interval width is printed beside the label.

### 7.3 Arm 1 (per target; amended A2, A3)

- **INSTRUMENT NOT SHOWN TO FIRE**: the Arm 1 job's G-place, G-gen or G-pc failed; or a ±d* cell of either pole has a
  median `realized_cos` below 0.95; or no d* exists ("no coherent range"); or either pole has fewer than 50 coherent
  samples (of 200) at +d* or −d*. Never read as NO SWING.
- **SWINGS**, **NO SWING**, **UNRESOLVED**: §5.4, on the paired interval of `S_t − S_c`.
- Room (frozen in `thresholds_arm1_<M>.json` from control lines): R-pass plants a swing of 0.30 (count: 1.5) on a copy
  of the control's coherent samples (A18: a plant equal to the 0.20 threshold leaves a near-zero control's observed
  swing at the threshold about half the time, so R-pass would sit near 0.5 even with a working readout; 0.30 tests
  whether a swing clearly above the threshold is seen) — a decrease at −d* when the control's R(−d*) ≥ 0.5, otherwise an increase at +d*
  — and the SWINGS rule against the unplanted control must fire in at least 80% of 2,000 replicates; R-equiv: control
  against a second bootstrap copy of itself must read NO SWING in at least 80%. Registered failing input: R-pass on the
  first 2 prompts must fall below 80%. A pair failing R-pass prints "underpowered for a swing"; failing R-equiv,
  "NO SWING cannot be read".
- Printed beside every label: the firing count (k of 20 prompts), d*, the control's d and "poor control match" when
  d > log 2, coherent counts and median `realized_cos` at ±d*, R(0), the ceiling note of §5.4, and the room stamps.

### 7.4 Arm 2 (per model; ONE poles contributing at d_p; two-way bootstrap of §5.7)

- `o(d_p)` mean with interval; paired `o(d_p) − o(0)` mean with interval.
- H_suff: **SUPPORTED** lo(mean o) > 1/3 and lo(paired) > 0; **ABOVE CHANCE, NOT ABOVE α = 0** lo(mean o) > 1/3 and
  lo(paired) ≤ 0; **REFUTED** hi(mean o) ≤ 1/3; **EQUIVALENT TO CHANCE** interval inside 1/3 ± 0.08; **NOT SHOWN**
  otherwise; **CANCELLED: NO COHERENT DOSE** if G-n fails. At a true on-label rate of 0.40 with between-pole spread
  0.15, lo > 1/3 in 66% of simulated runs and SUPPORTED in 45% (the reviewer's simulation,
  `sim/sim.py` in the session scratchpad, not in git, re-run 2026-10-26; half-width of mean o 0.056).
- H_pred_agree: `R = mean o(d_p) over agree strata / mean o(d_p) over disagree strata`. **SUPPORTED** lo(R) > 1 and
  R ≥ 2; **AGREE ABOVE DISAGREE, BELOW 2×** lo > 1 and R < 2; **REFUTED** hi(R) ≤ 1; **EQUIVALENT** inside [0.8, 1.25];
  **NOT SHOWN** otherwise. The difference of the two means is printed beside R.
- H_pred_conf (A5; a reported equivalence row): Spearman ρ between explainer confidence and o(d_p) (average ranks for
  ties). **SUPPORTED** the interval lies inside (−0.2, 0.2); **REFUTED** the interval excludes 0 (sign printed);
  **NOT SHOWN** otherwise.
- H_depth (Arm 2): `o(d_p)` tercile 2 minus tercile 0; labels as H_depth in §7.1.
- Secondary rows: every other dose; ONE against matched UNK (paired over the pairs) at d_p; by explainer model; ten
  free-text guesses per model printed beside their true labels.
- Study label: as §7.1.

### 7.5 Thresholds frozen on control data, and room on both sides

`freezeK` writes `results/thresholds_armK_<M>.json` from control data only and refuses to read any other row (it
filters by role from `roles_<M>.json` before opening a file). Control data: Arm 0 — `yn_shuf1`, `yn_shuf2`, `yn_fixed`
answers, G-judge0 answers and retest pairs; Arm 1 — α = 0 samples, natural spans, control-pole samples, and both
poles' coherence flags and `realized_cos` (for d* and the fire rule; the target's regex counts are not read); Arm 2 —
α = 0 samples and judge answers, natural spans, label-prompted samples and answers. The executor commits each
thresholds file by name before `unblindK` runs; `unblindK` refuses otherwise.

Room checks, stamped in the same file and printed beside every label (A4). For each rule below, 2,000 outer draws of
size n from the null values (R-null), from the null values plus the planted effect E (R-pass), and from the null values
alone for the equivalence label (R-equiv); in each draw the label is computed with an inner bootstrap of 1,000.
R-null: the supported label fires in at most 5% of draws. R-pass: it fires in at least 80%. R-equiv: EQUIVALENT
fires in at least 50%. Registered failing input: R-pass at n = 5 must fail; if it does not, the room check is cancelled
("room check cannot fail") and the rule's supported label cannot be read.

| rule | null values | n | E |
|---|---|---|---|
| H_write | `y_s1 − y_s2` per ONE item | ONE items | +0.15 |
| H_depth (Arm 0) | `y_s1 − y_s2`, tercile 0 and tercile 2 items | the two tercile sizes | +0.20 on tercile 2 |
| H_ONE | `(y_s1 − y_s2)(one) − (y_s1 − y_s2)(ctrl)` per pair | 293 pooled | +0.15 |
| H_suff | `o(0) − mean o(0) + 1/3` per ONE pole | contributing ONE poles (80 planned) | +0.15 |
| H_pred_agree | `o(0)` in both groups | the two group sizes | agree values × 2.5 (A5) |
| H_pred_conf | `o(0)` against confidence, confidence permuted per draw | 80 | `+0.3 · (rank(confidence) / n − 0.5)` (the planted relation must make REFUTED fire; R-equiv uses the permuted data) |
| H_depth (Arm 2) | `o(0)`, terciles 0 and 2 | the two tercile sizes | +0.20 on tercile 2 |
| SWINGS (Arm 1) | §7.3 | 20 prompts | 0.30 (count 1.5; A18) |
| B1 (A16) | per ONE item, `(y_s1 − y_s2)` in the with-vocabulary set minus `(y_s1 − y_s2)` in the re-explained set | 1,007 pooled (expected half-width about 0.04 against the ±0.05 margin) | +0.10 |

A rule whose R-pass is below 0.8 at the real n is stamped "underpowered for a direction"; one whose R-null exceeds
0.05 is stamped "rule fires on the null" and its supported label cannot be read; one whose R-equiv is below 0.5 is
stamped "no room for equivalence" and its EQUIVALENT label cannot be read. `unblindK` repeats all three at the
realized n and stamps them. Self-repair has fixed lines (0.5, 1, [0.8, 1.25]) and no control set; its interval width
is printed instead.

### 7.6 Multiplicity

The decisive readouts are H_write (Arm 0) and H_suff (Arm 2), per model, combined by the study rule. Every other
hypothesis and row is secondary and labelled so; no p-values are used. Arm 1 gives seven case-study labels, each with
its own interval; no family-wise claim is made from them.

### 7.7 Stamps

Every printed label of a cancelled model, arm or pole carries its cancellation stamp in every file it is written to
(verdicts, secondary rows, `grid.md`, the result document). A label whose interval crosses its margin prints
"interval crosses margin". A study row with fewer than two counting models prints "N counted models" with N.

## 8. Bugs not to repeat, and threats to validity

### 8.1 Bugs not to repeat (brief), each with a named test in `tests/test_neuron_rw.py`

| # | bug (source) | rule in this study | test |
|---|---|---|---|
| 1 | `str.splitlines()` split records at U+2028 (NEURONOFF deviation 1) | every JSON-lines read splits on `"\n"` only | `test_jsonl_split_u2028`: a fixture line holding U+2028, U+2029 and U+0085 inside strings parses to the right record count in every reader of the code |
| 2 | cancelled labels printed without a stamp (NEURONOFF review finding 2) | §7.7 | `test_cancel_stamp_everywhere`: a cancelled model's labels carry the stamp in the verdict, every secondary row and `grid.md`; a study row with zero counting models prints "0 counted models"; an interval crossing its margin prints "interval crosses margin" |
| 3 | a pole that never fired read NOT-SPECIAL (finding 4) | §7.3 and §7.4 | `test_not_shown_to_fire_label`: Arm 1 with G-place failed, with no d*, and with 49 coherent samples each give INSTRUMENT NOT SHOWN TO FIRE, never NO SWING; Arm 2 with 55 contributing poles gives CANCELLED: NO COHERENT DOSE |
| 4 | per-pole positive control with no room (finding 6) | G-pc pooled over 50 poles × 10 spans; no gate kills an item on n ≤ 10 | `test_pc_pooled_room`: the pc gate takes the pooled array, its α = 0 input fails and a fixture with a planted shift passes; a code search asserts no per-item kill path with n ≤ 10 |
| 5 | double BOS from the tokenizer | `add_special_tokens=False`, BOS prepended by the job | `test_tokenize_one_bos`: a fixture tokenizer that adds BOS by default yields exactly one BOS through the job's encoder; plus `neuron_rw_analysis tokcheck` on the local tokenizer folders before freeze (F1) |
| 6 | non-deterministic kernels | deterministic mode, identity ceiling 0, no `warn_only` | `test_deterministic_mode`: after job setup `torch.are_deterministic_algorithms_enabled()` is true, `torch.is_deterministic_algorithms_warn_only_enabled()` is false, `CUBLAS_WORKSPACE_CONFIG` is set; G-ident at α = 0 gives exactly 0 on the tiny model |
| 7 | a known-number failing input that could stay inside the tolerance | G-known's failing input must move the number by ≥ 0.5 nats | `test_known_failing_input_shift`: on the tiny model the layer-0 MLP zero moves the readout by more than the tolerance, and a fixture whose shift is below `failing_min_shift_nats` makes the job abort |
| 8 | thresholds not frozen first; kill rules without room | §7.5 | `test_freeze_before_unblind_and_room`: `unblindK` refuses without a committed thresholds file and an earlier PREDICTIONS commit; on a fixture R-pass fails at n = 5, passes at the full n, and R-null is ≤ 0.05 |
| 9 | a mixed-throughput budget rule dropped a third of the plan (NEURONOFF deviation 5) | §12: each arm's own measured rate | `test_budget_uses_arm_rate`: the budget function reads generation rows/s for generation and forward tokens/s for teacher forcing, from the arm's own stage, and applies the ladder in order |
| 10 | a hook that saturates behind a post-MLP norm (LIT_steering §C1) | §2.2 hook location; G-place | `test_hook_after_postnorm`: on tiny Gemma 3 and Olmo 3 configs the realized Δh at the layer output equals v within bf16 rounding; the MLP-input placement fails G-place |
| 11 | the other pole of a neuron used as a steering control (found 2026-10-25 while building the plan) | §4.5, §5.7; asserted in `make_plan.py` for Arm 1 pairs, TeX candidates, Arm 2 controls and every draw | `test_no_same_neuron_control`: `make_plan.same_neuron` and the Arm 2 builder on a synthetic answers file never pair two poles of one neuron |
| 12 | a hook keyed on the per-forward position index skips every decode step (A2) | §2.2 absolute positions; G-gen | `test_gen_gate_step_index`: on the tiny model the absolute-position hook passes G-gen at decode steps 1–3 and the per-forward-index hook fails it |
| 13 | decoder layers return a bare tensor in current `transformers` (A11) | §2.2 | `test_hook_output_types`: the hook works on a layer returning a bare tensor and on one returning a tuple, returns the same type, and raises on anything else |
| 14 | a judge request that leaks the pole's identity (A11) | §5.8 blind | `test_judge_requests_blind`: across every request `build` writes for steps J0, A0, J2, A2, no system or user text contains a pole id, a layer number pattern `L\d+N\d+`, a class name, a dose, a model name or the study name, and ids are opaque |
| 15 | re-explain prompts edited beyond the vocabulary lines, or a typed model ID (A1) | §5.0 | `test_reexplain_requests`: on two parent requests the rebuilt request differs from the parent only by the two user lines and the two system passages, its model equals `reported_models[0]` read from the record file, and the canonical lines hash to the manifest's `requests_sha256` |
| 16 | Arm 1 swing intervals computed unpaired (A3); Arm 2 intervals over poles only (A10) | §5.4, §5.7 | `test_paired_and_two_way_bootstrap`: on fixtures the Arm 1 replicate resamples one prompt index set for both poles, and the Arm 2 replicate resamples poles, prompts within pole, and the shared α = 0 prompts once per replicate |

### 8.2 Threats to validity

| threat | effect | mitigation |
|---|---|---|
| T1 the explainer saw the top-10 output tokens (AUDIT §3) | H_write inflated by copying | RE-EXPLAIN labels without the tokens are primary (§5.0); B1 measures the leak; literal-overlap row (§5.1) |
| T1b every re-explanation comes from one model, unlike the parent's Opus and Sonnet labels | the re-explained labels may differ in style and class from the parent's | re-explained class reported against the parent's; row of poles re-explained as ONE; the original-label rows |
| T2 judge leniency (yes-bias) | agreement rate inflated | paired shuffle baseline; fc3 with chance 1/3; G-judge0 ceiling on the shuffle yes-rate |
| T3 judge label-choice bias (generic labels picked for any text) | on-label rate off 1/3 at α = 0 | paired α = 0 null for H_suff; letters balanced; G-judge2 ceiling |
| T4 the judge model also wrote some labels (Sonnet explained 20–25% of ONE poles) | the judge may favour its own phrasing | rows by explainer model (§7.1, §7.4) |
| T5 explainer split by depth (AUDIT §3) | H_depth partly Opus against Sonnet | rows by explainer model within tercile |
| T6 bf16 rounding of small additions | a small dose is not the intended vector | G-place at α = 1 and the largest dose; realized cosine banked at every dose; doses below 1 are printed with their realized cosine |
| T7 RMS inflated by a few large positions or coordinates | doses not comparable across layers | position 0 excluded; mean and median norms banked; α_c expresses doses relative to measured collapse |
| T8 coherence gate misclassifies | on-label text marked collapsed, or collapse marked text | two-sided band with a natural-text ceiling; four registered inputs; all-sample curves banked |
| T9 attested vocabulary changes the top-20 | judged list differs from the harvest's | full-vocabulary lists banked; G-proj on the full vocabulary |
| T10 Arm 1 presence near ceiling at α = 0 (AUDIT §2) | upward swing impossible | signed S between ±d*; room check on the control interval |
| T11 the total effect zeroes the pole at every firing position of the window, the direct effect only at the readout position | the ratio mixes self-repair with indirect paths from other positions | stated with every ratio; the renormalized variant printed |
| T12 batched rows draw different random numbers across doses | no per-sample pairing across doses | readouts are per-dose rates; no paired-sample claim |
| T13 10 sweep poles set α_c for the model | α_c may not fit every pole | per-pole collapse rates in Arms 1 and 2 are printed; Arm 1 uses d* per pair |
| T14 F41 (negative poles' explainer saw swapped vocabulary lines) | negative-pole labels may describe what the pole suppresses | row by pole sign |
| T15 the TeX replacement is chosen by a firing check | selection on the input side | the rule and candidate order are frozen; the readout (construct count) is not read by the selection |

## 9. Runbook (an agent executes this without asking)

Paths: `S` = `<local>/interp_research-neuronrw` (spec and launch worktree), `C` =
`<local>/interp_research-neuronrw-code`. Bucket `B` = `s3://<s3-bucket>`. AWS calls go
through `./infra/aws_pg/pg` (profile `<aws-profile>`, region `us-east-1`, fixed in `infra/aws_pg/config.py`) or carry
`--profile <aws-profile> --region us-east-1`. Never run a model on the laptop. Never poll `job.log`.

### 9.1 Code owed before freeze (code writer, in `C` on `neurons/rw-code`; full list, amended A14)

1. `infra/pg/neuron_rw_core.py`:
   a. steering hook (§2.2): decoder-layer output, per-row vectors, absolute positions from P − 1 with a running offset
      across prefill and decode steps, fp32 add, runs at α = 0, accepts a bare tensor or a tuple and returns the same
      type, raises otherwise (A11);
   b. G-place inputs (MLP-input and down_proj-output placements) and the G-gen failing input (a hook keyed on the index
      within the current forward) (A2);
   c. realized-Δ capture at prefill P − 1 and decode steps 1–3, giving each row's `realized_cos` (A2);
   d. projections (§2.3; attested and full vocabulary);
   e. direct path with both scales frozen (primary) and the two recomputed variants, fp32, per family, softcap (A6);
   f. `nll_clean_mean` scoring (at most 4,096 fp32 logit positions at once);
   g. firing check (§5.5); coherence band with the seeded half split and classifier (§5.2, A7);
   h. `literal_overlap` (§5.1, A1);
   i. bootstraps: one-way, paired-over-prompts for Arm 1 (A3), two-way for Arm 2 with a per-replicate α = 0 prompt
      draw (A10); room checks R-null, R-pass, R-equiv (§7.5, A4).
2. `infra/aws_pg/jobs/neuron_rw.py`: stages `arm0`, `arm1`, `arm2` and `--smoke` (SCHEMA §2). Gates in the order G-det,
   G-ident, G-place, G-gen, G-known, G-floor-dir, G-floor-pos, G-pc; in `arm0` the sweep runs next (A9), then the
   Arm 1 throughput probe, projections, RMS, natural spans, firing check, `judge0_pc` firing tokens, and the repair
   forwards: one clean pass per window plus one t-only ablation forward per context (A6); G-coh after the α = 0
   samples; outputs of SCHEMA §3; throughput split into forward, generation and scoring, plus the probe's rate.
3. `infra/aws_pg/jobs/neuron_rw_dryrun.py`: tiny random Qwen 3, Gemma 3 and Olmo 3 configs; the full pipeline minus
   real weights; every SCHEMA output written; each gate's registered inputs land as registered.
4. `infra/pg/neuron_rw_analysis.py`: `gates0`, `dose_scale` (also prints Liu's α*, §5.3), `replacement`,
   `freeze0|1|2`, `unblind0|1|2` with the A4 labels and room stamps beside every label, `repair` (odd/even split, A6),
   `budget` (§12 rule, A9), `tokcheck`. The Arm 2 draw is in `plan/make_plan.py`, not here.
5. `infra/pg/neuron_rw_judge.py`: `build` for steps RX (rebuilds the re-explain requests by importing
   `plan/make_plan.py:reexplain_requests` and asserts the manifest sha256; A1), J0, A0 (both label sets, `yn_fixed`),
   J2, A2 (retest drawn by `default_rng([20261025, model_index, 13])`, A11); every request's model, temperature and
   `max_tokens` taken from `plan/judge_prompts.json → judge` (read from the parent's files by `make_plan.py`, never
   typed); `collect` through `neurons_llm.collect`, writing `results/judge_<batch>.jsonl` and
   `results/reexplain_<M>.jsonl`.
6. `git checkout neurons/explain -- infra/pg/neurons_llm.py` (unchanged); `infra/pg/settings/deepinfra_neuronrw_judge.json`
   = the `settings` object of the parent record file, copied verbatim by a one-line script that reads it.
7. `tests/test_neuron_rw.py`: the tests of §8.1 (including `test_judge_requests_blind`, `test_gen_gate_step_index`,
   `test_hook_output_types`, `test_reexplain_requests`, `test_paired_and_two_way_bootstrap`), plus one per analysis
   subcommand, and the per-cell `realized_cos` rule (median < 0.95 → "instrument not shown to fire").
Reused unchanged: `infra/pg/neuron_off_core.py` (`PoleAblator`, `next_token`, `bos_prefix`, `decoder_layers`,
`generation_config`), `infra/pg/neurons_harvest.py` (`text_tower`, `dump`), `infra/aws_pg/jobs/neurons_harvest.py`
(`download`, `fetch`, `fetch_corpus`).

### 9.2 Offline gate and freeze (laptop, no weights)

- F1. In `C`: `python -m pytest -q tests/test_neuron_rw.py` passes; for arch in qwen3, gemma3, olmo3 and stage in arm0,
  arm1, arm2: `python -m infra.aws_pg.jobs.neuron_rw_dryrun --arch <arch> --stage <stage> --out /tmp/neuronrw_dry/<arch>_<stage>`
  runs green, every registered failing input fails and every passing input passes, `metrics.json` is rewritten during
  the run, every SCHEMA key is present, `samples/` is written. `python -m infra.pg.neuron_rw_analysis tokcheck` on the
  local `tok/<qwen3|gemma3|olmo3>/tokenizer/` folders (tokenizer files only): the prefix property holds for 139 of 140
  Arm 1 records with Q4B TeX prompt 7 failing (as NEURONOFF), and each neutral prompt encodes without a BOS from the
  tokenizer. Record the pass lines in `JOURNAL.md`.
- F2. In `S`: `python3 -I studies/neuronrw/plan/make_plan.py --out /tmp/neuronrw_plan_check`, then `cmp` each JSON file
  with the committed one; identical expected; a difference stops the run as a spec defect.
- F3. In `S`: `git merge --no-ff neurons/rw-code` (disjoint files). `.pgvenv` symlink present. `git status --short` empty
  apart from untracked bank folders.
- F4. `JOURNAL.md` holds the heading `## 2026-10-26T03:07:48Z — PREDICTIONS` (§5.6 and the predicted column of §11,
  verbatim), committed by the spec writer before any thresholds file exists. If the adversarial review changes a
  prediction, a new dated JOURNAL entry `PREDICTIONS AMENDMENT` is committed before the tag; the first entry is never
  edited. Then, only after the director reports the code merged and the adversarial review in: `git tag
  neuronrw-freeze-1`, and a dated freeze line in §13 (tag, SHA, UTC), committed by name.
- F5. `aws s3 cp studies/neuronrw/plan/arm0_<M>.json $B/results/neurons/rw/plans/1/arm0_<M>.json --profile <aws-profile>
  --region us-east-1` for M in Q4B, G4, O7.
- R0. RE-EXPLAIN (director, §9.5, before L1; A1): step RX for the three models; `collect`; commit
  `results/reexplain_<M>.jsonl` by name with a JOURNAL line (counts answered, declined, re-explained class against the
  parent's); `aws s3 sync S/bank/rw/reexplain/ $B/results/neurons/rw/reexplain/1/ --profile <aws-profile> --region us-east-1`.

### 9.3 Blind protocol

Job plans carry no class, label or role. During runs the executor opens `metrics.json`, `controls.json`, `samples/`
and the first pole's or prompt's lines of each cell, and logs which. `roles_<M>.json` is joined to per-pole results only
by `freezeK` (control rows), `unblindK`, `make_plan.py`'s Arm 2 draw (after `unblind0`) and `repair`. Judge requests
carry no pole id (`test_judge_requests_blind`).

### 9.4 Launch sequence

`M` and `m` name the model in upper and lower case. `CORPUS=$B/results/neurons/w1/q4b/1/out/corpus/corpus`.
Q4B args `--model-key Q4B --repo Qwen/Qwen3-4B-Base --tokname qwen3`, weights `--weights models--Qwen--Qwen3-4B-Base`;
G4 `--model-key G4 --repo google/gemma-3-4b-pt --tokname gemma3`, `--weights models--google--gemma-3-4b-pt`;
O7 `--model-key O7 --repo allenai/Olmo-3-1025-7B --revision stage1-step1413814 --tokname olmo3`, `--no-weights`.

- L1. Preflight (known number first):
  ```
  cd S && ./infra/aws_pg/pg run infra.aws_pg.jobs.neuron_rw \
    --args " --stage arm0 --smoke --model-key Q4B --repo Qwen/Qwen3-4B-Base --tokname qwen3 --corpus $CORPUS --plan $B/results/neurons/rw/plans/1/arm0_Q4B.json" \
    --s3-prefix results/neurons/rw/pre_q4b/1 --type g6e.2xlarge,g6e.4xlarge --budget-min 45 \
    --weights models--Qwen--Qwen3-4B-Base --detach --json
  ```
- L2. `./infra/aws_pg/pg watch results/neurons/rw/pre_q4b/1`; read `out/metrics.json` and `out/controls.json`. Pass:
  `status: complete`, every gate true, every registered failing input failed, every registered passing input passed.
- L3. Sample check, written to `JOURNAL.md` before any number: open `out/samples/*.jsonl`, `out/arm0/projections.jsonl`
  (first 3 lines), `out/arm0/repair/` (one file), `out/sweep/samples.jsonl` (one sample per dose). Answer: is the input
  the spec's (prompt as the model saw it, BOS once for G4 and O7, none for Q4B); is the output in format; is the scored
  quantity the model's own; does the realized dose match the intended one; are the collapse inputs marked collapsed.
- L3b. Budget minutes (A15): from the preflight's `out/metrics.json → throughput` (forward, generation and scoring
  rates; O7 at 0.56 of them) and its measured load and gate time plus 0.1 h, project each Arm 0 cell's hours from the
  work counts in `plan/budget.json`; `--budget-min` = max(60, ceil(60 × 1.3 × projected hours)). Write the three
  numbers in the JOURNAL. The pre-freeze estimates at planning rates (69, 82, 40 minutes) are not used to launch; at
  those rates O7's 40 minutes falls short of a measured need of about 52.
- L4. Arm 0 cells, launched together, each with the `--budget-min` of L3b:
  ```
  ./infra/aws_pg/pg run infra.aws_pg.jobs.neuron_rw --args " --stage arm0 <M args> --corpus $CORPUS --plan $B/results/neurons/rw/plans/1/arm0_<M>.json" \
    --s3-prefix results/neurons/rw/arm0_<m>/1 --type g6e.2xlarge,g6e.4xlarge --budget-min <L3b value> <M weights> --detach --json
  ```
- L5. After each `STATUS exit=0`: `aws s3 sync $B/results/neurons/rw/arm0_<m>/1/ S/bank/rw/arm0_<m>/1/ --profile <aws-profile>
  --region us-east-1`. Then `neuron_rw_analysis gates0 --model <M>` (G-proj, G-repair-fit), `dose_scale --model <M>`,
  and for Q4B `replacement`. Rerun `make_plan.py --out studies/neuronrw/plan` (rewrites `arm1_<M>.json` only); commit
  `dose_scale_<M>.json`, `arm1_replacement_Q4B.json` and the arm1 plans by name with a JOURNAL line; upload the arm1
  plans to `$B/results/neurons/rw/plans/2/`.
- L6. Arm 0 judge (director, §9.5; `neuron_rw_judge run`, A22): J0 preflight on Q4B, then A0 for the three models; `collect`; G-retest;
  `freeze0` per model → commit the thresholds files → `unblind0` per model. Then `repair --model Q4B` and `--model G4`.
- L7. Budget rule (§12) with each model's measured Arm 0 sweep rates; write the decision and any ladder cut in the
  JOURNAL.
- L8. Arm 1 cells (Q4B, G4), `--stage arm1`, plans `$B/results/neurons/rw/plans/2/arm1_<M>.json`, prefixes
  `results/neurons/rw/arm1_<m>/1`, `--budget-min` = max(60, ceil(60 × 1.3 × projected hours)) from §12's measured rates.
- L9. Rerun `python3 -I studies/neuronrw/plan/make_plan.py --out studies/neuronrw/plan` (it now finds the A0 answers
  and the re-explanations and writes `arm2_<M>.json` and the roles `arm2` blocks; §5.7) → commit them by name →
  upload to `$B/results/neurons/rw/plans/3/` → Arm 2 cells, `--stage arm2`, prefixes `results/neurons/rw/arm2_<m>/1`,
  budgets as L8.
- L10. Liveness: `./infra/aws_pg/pg ps` and each cell's `out/metrics.json → updated_utc`. The first line of every status
  report is the run / unrun / parked count over the roster (pre_q4b, arm0 × 3, arm1 × 2, arm2 × 3). Never score a
  running cell. Sync after `STATUS exit=0` as L5.
- L11. Arm 1: sample check (one sample per pole and dose) → `freeze1` → commit → `unblind1`.
- L12. Arm 2: sample check → judge (director; `neuron_rw_judge run`, A22): J2 preflight on Q4B, then A2 for the three models; `collect`; G-retest →
  `freeze2` → commit → `unblind2`.
- L13. A different model reads the artifacts (CLAUDE.md Provenance §3) before the result document is written.

### 9.5 Judge steps (run by the director)

```
set -a; . ~/.config/interp_research/keys.env; set +a          # exports NEURONS_API_KEY_1; never printed
MODEL=$(python3 -c "import json; print(json.load(open('studies/neuronrw/plan/judge_prompts.json'))['judge']['model'])")
curl -s https://api.deepinfra.com/v1/openai/models | python3 -c "import json,sys; ids={m['id'] for m in json.load(sys.stdin)['data']}; assert sys.argv[1] in ids, 'judge model not listed'" "$MODEL"
python -m infra.pg.neuron_rw_judge build --step <J0|A0|J2|A2> --model <M> --out studies/neuronrw/judge
python -m infra.pg.neuron_rw_judge run --step <J0|A0|J2|A2> --model <M> --out studies/neuronrw/judge --concurrency 8
python -m infra.pg.neuron_rw_judge collect --step <step> --model <M>
```
Amended 2026-10-26 (A22 addendum): judge calls go through `neuron_rw_judge run`, which prints the A22 cost projection
and refuses a step whose projected cumulative judge spend would pass $28; `neurons_llm run-many` is no longer used for
judge steps. RE-EXPLAIN, finished in R0, ran through `neurons_llm run-many` after `neuron_rw_judge reexplain --dry-run`.
RX: `--explain bank/rw/reexplain`, batches `rx-<M>`, the same settings file. J0: the first 20 Arm 0 items of Q4B
(all primary requests each) and 20 G-judge0 pairs; the gates of §6 are computed on
these real answers and must land as registered before A0 starts. J2: the first 20 Arm 2 ONE poles of Q4B, their
label-prompted and α = 0 calls. Preflight answers are kept and excluded from the readouts (counted).

### 9.6 Pivots (pre-specified)

- P1 a model gate fails: re-run the cell once at a fresh attempt on `--type g6e.2xlarge`. A second failure: that model
  leaves that arm, reported with both values; other cells proceed.
- P2 capacity: the launcher tries the next type; no capacity for 2 h → stop and hand off.
- P3 budget kill mid-cell: completed poles and generate calls stand (written as they finish); readouts use complete
  (pole, dose, prompt) cells; poles missing a dose are excluded and counted; G-n decides. A fresh-attempt re-launch of
  the missing poles only if the cap allows.
- P4 G-det, G-ident or G-floor fails: no re-run without a code fix and a dated amendment, then a fresh preflight.
- P5 deterministic mode raises on an op: a code change that selects a deterministic kernel, a dated amendment, a fresh
  preflight. `warn_only` and turning deterministic mode off are not allowed.
- P6 G-pc's passing input fails in the preflight: the pc dose is raised to 64 once by a dated amendment; a second
  failure makes that model's steering arms read "instrument not shown to fire".
- P7 the judge model is not listed or a response reports another model: stop and tell the director; no substitute.
- P8 G-coh fails: §5.2 cancelled branch; the run continues.
- P9 no TeX candidate qualifies: the slot is dropped (§5.5).
- P10 G-judge0 or G-judge2 lands wrong in its preflight: stop the judge step and hand off to the director; a changed
  judge prompt is a dated amendment made before the full batch, and the preflight items stay excluded.
- P11 the budget ladder (§12) reaches its last step: stop and hand off as a spec defect.
- P12 RE-EXPLAIN: a request that neurons_llm declines after its attempts leaves that pole out of the primary rows
  (counted). If more than 10% of a model's ONE poles have no re-explanation, that model's Arm 0 and Arm 2 primary
  labels are cancelled ("labels not re-explained") and the original-label rows are reported as exploratory.

### 9.7 Sessions and handoff

S1: §9.2, L1–L4; ends with `studies/neuronrw/handoff/S1.md` (open instance ids, prefixes, budgets, exact next action).
S2: L5–L9; ends with `handoff/S2.md`. S3: L10–L13. Each handoff starts with the run / unrun / parked count.

## 10. Artifacts

Per cell under `S/bank/rw/<cell>/<attempt>/out/` (SCHEMA §3). Study files: `results/thresholds_armK_<M>.json`,
`results/verdict_armK_<M>.json`, `results/repair_<M>.json`, `results/judge_<batch>.jsonl`, `results/grid.md`,
`results/REGISTRY.md` (every number → file, key, input sha256), `judge/` (requests, responses, keys, call log),
`results/reexplain_<M>.jsonl` and the banked RE-EXPLAIN folder `bank/rw/reexplain/` (S3 copy under
`results/neurons/rw/reexplain/1/`).
Figures, after the checker, each with its null drawn: F1 Arm 0 agreement, shuffle and fixed-label rates per model and
tercile, for re-explained and original labels; F2 self-repair T_p against D_p with the 0.5 and 1 lines; F3 sweep collapse curves with α_c;
F4 Arm 1 R(α) and C(α) per pair, target and control; F5 Arm 2 on-label rate per dose with the 1/3 line and the α = 0
rate; F6 Arm 2 agree against disagree strata.

## 11. Prediction grid (predicted column frozen before the tag; observed filled once by `unblindK`)

Amended 2026-10-26 (A4, A5, A12, A13): labels use the amended label set; the first version stays in `JOURNAL.md`
(PREDICTIONS, 2026-10-26T03:07:48Z) and this one is its PREDICTIONS AMENDMENT entry.

| readout | predicted | probability | observed | label |
|---|---|---|---|---|
| H_write (re-explained labels), study label | SUPPORTED | 0.5 (NOT SHOWN or ABOVE SHUFFLE, BELOW MARGIN 0.3; MIXED 0.15; REFUTED 0.05) | ____ | ____ |
| B1 leak, pooled (A16 definition) | SUPPORTED: labels written with the vocabulary lines agree more than labels written without them, L between 0.02 and 0.10 | 0.55 | ____ | ____ |
| Arm 0 absolute rates, every model, re-explained labels | A between 0.20 and 0.50; Sh and Fx between 0.05 and 0.25 | 0.6 | ____ | ____ |
| H_depth (Arm 0), study label | SUPPORTED (late above early; Geva et al. 2021, Gurnee et al. 2024); Arad et al. 2025's token-identity input score is highest early, so REFUTED is not unlikely | 0.45 (REFUTED or NOT SHOWN 0.45; MIXED 0.1) | ____ | ____ |
| H_ONE, pooled | NOT SHOWN (interval contains 0, wider than ±0.10) | 0.5 (EQUIVALENT 0.2; SUPPORTED 0.25; REFUTED 0.05) | ____ | ____ |
| fc3 accuracy, ONE, every model | above 1/3 with lo > 1/3, point between 0.40 and 0.60 | 0.6 | ____ | ____ |
| re-explained class | at least 80% of parent-ONE poles re-explained as ONE, every model | 0.5 | ____ | ____ |
| H_repair, pooled | REPAIR, NOT BELOW 0.5 | 0.35 (H_REPAIR 0.25; NOT SHOWN 0.25; REFUTED 0.15) | ____ | ____ |
| H_repair_ONE, pooled | NOT SHOWN | 0.65 | ____ | ____ |
| self-repair qualifying poles | fewer than half the measured poles qualify on odd contexts, each model | 0.6 | ____ | ____ |
| α_c (this study's guess) | ≥ 16 in every model | 0.6 | ____ | ____ |
| α_c held out from Liu 2026 Eq. 3, 5 and Table 3 (§5.3; A17) | Q4B 64; G4 16 or 32 | 0.4 (Q4B), 0.55 (G4) | ____ | ____ |
| Arm 1 SWINGS count | ≤ 2 of 7 (per pole: §5.6); most others UNRESOLVED rather than NO SWING | 0.6 | ____ | ____ |
| H_suff, study label | NOT SHOWN: on-label rate between 0.33 and 0.42 per model; at a true 0.40 the interval excludes 1/3 in about two thirds of runs (§7.4), so the label is NOT SHOWN or SUPPORTED by chance of the draw | 0.45 (SUPPORTED 0.3; ABOVE CHANCE, NOT ABOVE α = 0 0.1; EQUIVALENT TO CHANCE 0.1; REFUTED 0.05) | ____ | ____ |
| parent D1 X2 prediction ("steering-lens agreement above shuffled on poles that pass §6.1", p 0.7) | quoted for comparison; this study predicts H_suff is not SUPPORTED | 0.7 | ____ | ____ |
| H_pred_agree, study label | NOT SHOWN (agree above disagree in point estimate, interval of R contains 1) | 0.55 | ____ | ____ |
| H_pred_conf, study label | NOT SHOWN (interval contains 0 but is wider than ±0.2 at n = 80) | 0.6 (SUPPORTED 0.15; REFUTED 0.25) | ____ | ____ |
| H_depth (Arm 2), study label | NOT SHOWN | 0.6 | ____ | ____ |
| G-pc, G-place, G-gen, G-known pass in every job | yes | 0.8 | ____ | ____ |
| G-judge0 and G-judge2 pass | yes | 0.75 | ____ | ____ |
| α = 0 Arm 2 on-label rate, every model | between 0.25 and 0.42 | 0.8 | ____ | ____ |

## 12. Compute budget (amended A6, A9)

Hard cap 12 GPU-hours (instance-hours of every cell). Planning line 11.0 h; the rest covers one P1 re-run.

Planning throughput (`plan/budget.json → rates`; the rule below replaces it with measured rates): forward 13,355
tokens/s for the 4B models (NEURONOFF's Arm-A-only preflight rate), scoring 8,622 (Q4B) and 9,452 (G4) tokens/s
(NEURONOFF's main-cell rates), O7 at 0.56 of the 4B rate (parameter ratio 4.0B / 7.3B, as NEURONOFF §12); generation
12 rows/s (4B) and 6 rows/s (O7) at 160 rows per call, an assumption: NEURONOFF measured 3.0 (Q4B) and 2.6 (G4) rows/s at
20 rows per call (`studies/neuronoff/results/{q4b,g4}/1/manifest.json → stage_s.arm_b`: 1,609.7 s for 4 × 1,200
samples; 1,411.3 s for 3 × 1,200), and a larger batch reads the weights once for more rows. Fixed time per cell 0.25 h
(4B) and 0.35 h (O7, Hub download).

| cell | work (from `plan/budget.json`) | arithmetic | GPU-h |
|---|---|---|---|
| pre_q4b | smoke: gates, 20 projections, 3 repair poles, 2 sweep poles | 0.25 + small | 0.30 |
| arm0_q4b | 11.04M forward tokens + 15.45M t-only repair tokens (A6); 2,500 sweep + 320 probe rows; 0.48M scoring tokens | 0.25 + (26.49M / 13,355 + 2,820 / 12 + 0.48M / 8,622) / 3,600 | 0.88 |
| arm0_g4 | 13.83M + 20.47M; 2,820; 0.48M | 0.25 + (34.30M / 13,355 + 2,820 / 12 + 0.48M / 9,452) / 3,600 | 1.04 |
| arm0_o7 | 0.58M; 2,500; 0.35M | 0.35 + (0.58M / 7,479 + 2,500 / 6 + 0.345M / 5,293) / 3,600 | 0.51 |
| arm1_q4b | 4 pairs × 20 prompts × 17 doses-and-poles × 10 = 13,600 rows; 5.82M scoring tokens | 0.25 + (13,600 / 12 + 5.82M / 8,622) / 3,600 | 0.75 |
| arm1_g4 | 3 pairs: 10,200 rows; 4.37M | 0.25 + (10,200 / 12 + 4.37M / 9,452) / 3,600 | 0.61 |
| arm2_q4b | 120 poles × 3 doses × 5 × 10 + 50 + 400 = 18,450 rows; 2.55M | 0.25 + (18,450 / 12 + 2.55M / 8,622) / 3,600 | 0.76 |
| arm2_g4 | as Q4B | 0.25 + (18,450 / 12 + 2.55M / 9,452) / 3,600 | 0.75 |
| arm2_o7 | as Q4B | 0.35 + (18,450 / 6 + 2.55M / 5,293) / 3,600 | 1.34 |
| **total** | | | **6.95** |

The t-only total-effect forwards (A6) add 0.32 GPU-h (Q4B) and 0.43 GPU-h (G4); they are an upper bound (one full
prefix per context; reusing a window's clean cache for the positions before t would cost less). At NEURONOFF's
measured 20-row rates (3.0, 2.6, 1.46 rows/s) the same plan costs 15.0 GPU-h (`plan/budget.json →
total_hours_at_batch20_rate` 14.7 + preflight 0.3), over the cap; the ladder below then cuts it. Runs: 9 launches.

API calls (laptop, no GPU): RE-EXPLAIN 2,674 requests of about 40,000 characters each (1,667 without the vocabulary
lines + 1,007 with them, A16; §5.0); Arm 0 about 14,300 (1,300 items × 5 primary + 1,007 × 3 with-vocabulary + 1,300 × 3
original-label + 600 G-judge0 + 260 retest) plus about 120 in J0; Arm 2 about
9,850 (3 models × 120 poles × 4 dose cells × 5 prompts, 1,800 guesses at d_p, 120 label-prompted cells, about 730
retest) plus about 40 in J2.

**Budget rule (frozen; each arm's own measured throughput; amended A9).** After the Arm 0 cells, per model:
- Arm 1 (Q4B, G4): `gen1_M` = `metrics.json → throughput.gen_rows_per_s_arm1probe`, the rate of the two 160-row probe
  calls on an Arm 1 prompt inside the Arm 0 job; projected Arm 1 hours = fixed + Arm 1 rows / gen1_M + Arm 1 rows ×
  (Arm 1's mean prompt length + 128) / score_M, with the mean prompt length in tokens banked by the Arm 0 job from
  `prompt_sets` (`throughput.arm1_mean_prompt_tokens`).
- Arm 2: `gen2_M` = `throughput.gen_rows_per_s` of the model's own sweep (neutral prompts, the Arm 2 prompt length);
  Arm 2 is never projected from Arm 1's rate.
- `score_M` = `throughput.score_tokens_per_s`; fixed time = the cell's measured load and gate time plus 0.1 h.
- Spent = the instance-hours of every finished cell from `./infra/aws_pg/pg ps` records. If spent + projected Arm 1 +
  projected Arm 2 ≤ 11.0 h, everything runs. Otherwise apply, in order, until it fits: (1) Arm 2 doses := {d_p, d_p / 2};
  (2) Arm 2 samples per prompt 10 → 6; (3) Arm 1 samples per prompt 10 → 6; (4) Arm 2 drops O7; (5) Arm 2 keeps the first
  60 ONE and 30 UNK poles in draw order; (6) stop (P11). G-n stays at least 0.7 of the drawn ONE poles at every step.
  Each step is recomputed with the measured rates and written in the JOURNAL.
- `--budget-min` of every cell = max(60, ceil(60 × 1.3 × the cell's projected hours)) (A15): Arm 0 cells from the
  pre_q4b measured rates (§9.4 L3b), Arms 1 and 2 from the rates above. The planning-rate values (69, 82, 40 minutes
  for Arm 0) are a pre-freeze estimate only.

## 13. Amendments and freeze record

Freeze record: frozen 2026-10-26T05:06:42Z. The tag `neuronrw-freeze-1` is placed on the commit that adds this line (its SHA is read
from the tag: `git rev-parse neuronrw-freeze-1`). PREDICTIONS were committed before it at ef011487, with PREDICTIONS
AMENDMENT at a5adb20d and PREDICTIONS AMENDMENT 2 at c3e38c12 (`studies/neuronrw/JOURNAL.md`); no thresholds file
exists in any commit (`git log --all --diff-filter=A -- 'studies/neuronrw/results/thresholds_*'` is empty). The code
is `neurons/rw-code` f84fd1c0, merged with `--no-ff` at 144a5523 (the merged tree equals f84fd1c0's). Offline gate on
the merged tree (scratchpad venv `nvenv`, no weights): `tests/test_neuron_rw.py` 72 passed; `tests/test_neuron_off_core.py`
72 passed; nine dry runs (qwen3, gemma3, olmo3 × arm0, arm1, arm2) exit 0 with `NEURON_RW_DRYRUN_OK`; RE-EXPLAIN dry
run for Q4B, G4, O7 reproduces both `requests_sha256` values of each manifest (550 + 358, 557 + 338, 560 + 311
requests); `tokcheck` 139 of 140 Arm 1 records hold the prefix property (Q4B TeX prompt 7 fails, as NEURONOFF) and no
neutral prompt gets a BOS from the tokenizer; plans byte-reproducible (17 of 17 at 4b28e2c3, unchanged by the merge).
Amendments made before the tag: §14.2 (A1–A19, 2026-10-26).

Rules the code fixed where this spec is silent, ratified as written at f84fd1c0 (module docstrings): job J1–J17
(`infra/aws_pg/jobs/neuron_rw.py`), judge K1–K5 (`infra/pg/neuron_rw_judge.py`), analysis A1–A10
(`infra/pg/neuron_rw_analysis.py`; their letter is the code's, not this spec's amendment series). Two readings are
stated: (1) for Arm 2 the decisive coherence band is the analysis band over the sweep's and Arm 2's α = 0 rows pooled
(§5.2); the job's own Arm 2 band (J13) is a preview; (2) when the sweep's G-coh fails, `dose_scale` still writes α_c
and stamps it (analysis A2), and that stamp is printed beside every Arm 1 and Arm 2 label of the model. No GPU run, no
RE-EXPLAIN call and no judge call of this study exists before this line.

**Post-freeze amendments A20 and A21 (dated 2026-10-26; director's decisions after preflight attempt 1).** Both are
instrument fixes found before any arm ran and before any unblinding; no threshold, margin or decision rule moves. The
frozen body above is not edited; §14.2 carries the amended rules. Preflight attempt 1, `results/neurons/rw/pre_q4b/1`
(<instance-id>, g6e.4xlarge, about 0.25 instance-hours, counted in spent), passed all eight model gates with
every registered failing input failing (`bank/rw/pre_q4b/1/out/controls.json`; JOURNAL 2026-10-26T09:42:00Z): G-det
0.0 (failing input 21.04); G-ident 0.0 (0.4375); G-place cos 0.9963–0.999996, ratio 0.99994–1.0037 (MLP-input
placement cos −0.435 to 0.121); G-gen cos 0.9937–0.999997, ratio 0.99974–1.0069 (per-forward-index hook 0); G-known
exact (median |a/a_banked − 1| = 0.0 on 100 contexts, own clean NLL 1.3216137 and own ΔNLL 0.3082006 equal to the
banked values; layer-0 failing input shift 3.366 nats); G-floor-dir 0.0 (9.375); G-floor-pos 0.0 (23.44); G-pc
0.541 nats [0.335, 0.793] (α = 0 failing input 0.0). It then raised in stage work (`neuron_rw.py:829`, A20 below)
before G-coh, G-proj or G-repair-fit, so it did not pass L2. It is counted as attempt 1; the fresh preflight runs at
`results/neurons/rw/pre_q4b/2` on plans regenerated under A20 and uploaded to `results/neurons/rw/plans/2/` (so the
regenerated Arm 1 plans of §9.4 L5 go to `plans/3/` and the Arm 2 plans of L9 to `plans/4/`).

**Post-freeze amendment A22 (dated 2026-10-26; the user's decision; recorded before any judge call and before any
unblinding).** The judge steps J0, A0, J2 and A2 together may spend $30; their calls are cut from about 24,300 to
6,908 by the rules of §14.2 A22 (repeats per item first, then stratified seeded subsamples inside frozen strata; every
gate and the retest rate kept), with a stop rule at a projected $28. A22 changes n, not thresholds; the frozen rules
decide the headlines. Predictions left without power at the new n are marked underpowered in the JOURNAL before any
data. The A20/A21 code (`neurons/rw-code` 1b46ff2e) is merged at 949e6f4d; the `plans/2` hashes are in §14.2.

**Post-freeze amendment A23 (dated 2026-10-26; director's decision after preflight attempt 2; before any arm ran and
before any unblinding).** Preflight attempt 2, `results/neurons/rw/pre_q4b/2` (<instance-id>, `status: complete`;
JOURNAL 2026-10-26T10:56:38Z), passed every job gate with every registered failing input failing: G-det 0.0 (21.04);
G-ident 0.0 (0.4375); G-place at α = 1 and 32 as in attempt 1, and the new A21 entries at α = −1 (real hook cos
median 0.99636–0.99694, wrong-sign hook −0.9963 to −0.9969, each failing); G-gen 0.9937–1.0 (per-forward-index hook 0);
G-known exact again (100 of 100 a exact; own clean NLL 1.3216137015563436 and own ΔNLL 0.3082006178121082 equal to the
banked values; layer-0 failing input 3.366 nats); G-floor-dir 0.0 (9.375); G-floor-pos 0.0 (23.44); G-pc 0.541 nats
[0.335, 0.793] (α = 0 input 0.0); G-coh passed (natural test spans 0.934, α = 0 test half 0.900); A20 firing tokens
11–20 per pole, every count equal to the plan's and at least 10; G-proj 20 of 20. G-repair-fit landed wrong (A23
below), so attempt 2 did not clear the preflight. It counts as attempt 2; the next preflight runs at
`results/neurons/rw/pre_q4b/3`.

**Post-freeze amendment A24 (dated 2026-10-26; director's decision after the Arm 0 cells; before any judge call and
before any unblinding; no hypothesis threshold moves).** What passed (JOURNAL 2026-10-26T11:40:00Z, 12:50:00Z,
13:20:00Z): preflight attempt 3 (`pre_q4b/3`) green on every gate, the known number exact a third time; the A23 fit
read 0.0 / 0.0 on all three poles, the wrong-epsilon input failed on every pole, and the layer-18 shifts were 7.9–13.5
nats. Q4B Arm 0 (`arm0_q4b/1`): every gate passed (G-proj 453 of 454, G-repair-fit 0 of 156 out, G-coh passed);
α_c(Q4B) = 64, so Liu's held-out prediction lands; the TeX slot is dropped (no candidate fires on 15 of 20 prompts;
§5.5, P9). G4 Arm 0 (`arm0_g4/1`): G-known exact (own clean NLL 2.855343296620995, own ΔNLL 2.5111956053151516),
G-repair-fit 0 of 201 out, G-pc 12.318 nats [10.70, 14.14]; G-proj and G-coh landed wrong (A24 (1), (3)). O7 Arm 0
(`arm0_o7/1`): G-known's readings passed (activation median error 0.0043; 120,000-target NLL 2.6753136 against
2.6753720, Δ 5.8e-5), but its failing input lacked room and the cell aborted (A24 (2)). Amendments in §14.2.

**Post-freeze amendment A25 (dated 2026-10-26; director's decision after freeze0 and before unblind0; nothing
unblinded, no true-label yes-rate seen; δ_w and the H_write decision rule unchanged).** What passed (JOURNAL
2026-10-26T13:45:00Z–14:12:00Z): J0 G-judge0 PASS (firing-token lists against true labels 1.00, shuffle 0.025, failing
input pc_shuf 0.10); A0 2,640 calls answered with 0 declined; G-retest yn_true κ 1.0 on all three models; O7 Arm 0
attempt 2 (`arm0_o7/3`) green, with the A24 G-known failing input shifting 2.32 nats, and the A24 fallback running
α_c = 32 for O7 after its own G-coh failure (repeated-token NLL 0.252 ≥ L 0.126); judge spend $4.14. What landed
wrong: the §7.5 room check's registered failing input (R-pass at n = 5) read 1.00 / 0.95 / 0.83 (Q4B / G4 / O7), so it
cannot fail (A25 in §14.2).

**Post-freeze amendment A26 (dated 2026-10-26; director's decision after Arm 0 was unblinded and before any Arm 1 or
Arm 2 label was frozen or unblinded).** What happened (JOURNAL 2026-10-26T14:35:00Z–15:21:00Z): the A25 `freeze0`
re-run read the amended n = 5 R-pass at 0.01–0.26 on every row of every model, so the frozen stamp stands nowhere; O7
H_depth cannot be read (the +0.0 plant fires 0.097 at n = 5, rule fires on the null 0.39). `unblind0` ran once per
model: Q4B A 0.180 [0.110, 0.260], Sh 0.005, Fx 0.020, D_w 0.175 [0.105, 0.250] → H_WRITE, above the margin 0.03; G4 A
0.200 [0.120, 0.280], Sh 0.015, Fx 0.010, D_w 0.185 [0.110, 0.265] → H_WRITE, above the margin 0.05, and CANCELLED under
the frozen exact-order G-proj reading; O7 A 0.190 [0.120, 0.270], Sh 0.065, Fx 0.050, D_w 0.125 [0.035, 0.215] →
H_WRITE, interval crosses the margin 0.10. Study level: H_write SUPPORTED, H_depth MIXED, H_ONE pooled NOT SHOWN, B1
pooled EQUIVALENT. Self-repair CANCELLED, underpowered (16 and 15 qualifying poles against 20; M_r 3.94 [0.265, ∞) and
0.725 [0.196, 3.04]). Budget 5.51 h projected against 11 (no ladder step). Arm 2 drawn on `plans/5`. `arm1_g4/1`
completed with G-coh failing again (P8: coherent-only readouts cancelled, all-sample curves exploratory). `arm1_q4b/1`
aborted on G-known (A26 in §14.2).

**Post-freeze amendment A27 (dated 2026-10-26; director's decision before any Arm 1 or Arm 2 freeze or unblinding).**
What happened (JOURNAL 2026-10-26T15:46:00Z, 16:23:00Z): A26 worked — `arm1_q4b/2` reproduced G-known exactly on the
full Arm 0 windows (100 of 100 activations; own clean NLL 1.3216137 and own ΔNLL 0.3082006, the truncated-window reading
1.4825952 / 0.6278234 printed beside), and `arm2_q4b/1` completed with G-known exact. A24's own refutation fired: at
the fallback α_c 32 the steered continuations are mostly not text on G4 (all six Arm 1 poles at ±32: repetition loops,
mixed script, symbol runs; ±16 degraded) and on O7 in the Arm 0 sweep (±32 repetition); Q4B is fluent at ±32. G-coh's
repeated-token failing input has no room in Arm 1 on either model (NLL 1.339 inside the floor 0.077 on Q4B, 0.298 inside
0.064 on G4; the floor comes from code continuations, which repetition cannot undercut), and Q4B's Arm 2 job preview
band reads the label-prompted continuations 0.745 coherent. Cost so far: about 5.0 instance-hours, judge spend $4.14.

**Post-freeze amendment A28 (dated 2026-10-26; before any O7 Arm 2 judge call).** The independent result review is
committed as `studies/neuronrw/checks/2026-10-26_result_review.md` (ade57f08). A28 raises the judge budget for O7's
Arm 2, registers the held-out prediction P-lit on O7's Arm 2 before it is judged, and records the review findings
accepted for the closeout (§14.2). O7's Arm 2 answers have not been requested (no O7 Arm 2 key or response file exists).

## 14. Changes from the director's brief, and why

1. Judge: `anthropic/claude-sonnet-5-5` through the parent's `openai_compat` DeepInfra backend (the parent's W4 judge
   settings, §5.8). The parent's labels also came from `anthropic/claude-opus-5-5`, which the parent's own code refuses
   to call through an API since 2026-10-20.
2. The judged top-20 is taken over the attested vocabulary (tokens that occur in the corpus); Gemma's 262,208-row
   vocabulary holds unused tokens. Full-vocabulary lists are banked, and G-proj checks them against the harvest.
3. Leak control (tightening): the parent's explainer saw the top-10 output tokens (AUDIT §3). Superseded by A1: labels
   re-explained without those tokens are primary; the first draft's no-mention stratum is replaced by B1 and the
   literal-overlap row.
4. Coherence gate: two-sided (a floor catches repetition loops), and the ceiling also admits natural text (samples at
   temperature 0.7 have lower NLL than human text). Frozen on α = 0 samples and natural spans.
5. Doses: the brief's grid scaled by α_c, the measured collapse dose (identical to the brief when α_c = 8); Arm 2 doses
   {2, 4} plus α_c / 2, primary dose α_c / 2 — the brief's "first dose above the collapse knee" stated as a rule.
6. Arm 2: "10 samples each from 5 neutral prompts" read as 10 samples per prompt (50 per dose), one judge call per
   prompt; H_suff also requires beating the pole's own α = 0 judge rate (tightening); the free-text guess is a separate
   request at d_p only, so the candidates cannot leak into it.
7. Self-repair: the ratio is taken on poles whose direct effect is shown (interval above 0), because a ratio over a
   near-zero denominator explodes (CLAUDE.md Instruments §9); absolute T and D are printed. In Gemma 3 and Olmo 3 the
   direct path goes through `post_feedforward_layernorm`. The frozen-scale convention follows Rushing & Nanda 2024.
8. Floor gate: the brief's random direction at α = 0 is kept, and a position floor (no change before P − 1) added,
   because a zero dose cannot fail on any direction. G-place added: the realized vector is audited, not the intended one.
9. Known number: Q4B and G4 use the NEURONOFF pole with the largest own ΔNLL among poles passing every gate (a and own
   ΔNLL as the brief says); O7 has no NEURONOFF bank, so it uses the harvest's a and its 120,000-target NLL.
10. TeX: a frozen candidate list and rule; firing is measured on GPU over all positions in the Arm 0 job, because the
    brief's 15-of-20 test needs activations that only a forward gives.
11. Judge positive controls (G-judge0, G-judge2) and a retest kappa added: NEURONOFF's lesson that a positive control
    must be shown to pass on real data.
12. Arm 1 swing measured between ±d*; the refutation is the brief's (inside the control's interval).
13. H_pred_agree uses the ratio of raw on-label rates; a ratio of rates above chance would divide by a value near 0.
14. Arm 0 UNK items are NEURONOFF's pair controls only (293), as the brief says; same-neuron pairs are kept in H_ONE
    and a row without them is reported. No steering control is ever the other pole of the same neuron.
15. H_write and H_depth are computed on ONE items; UNK items enter H_ONE only.
16. Budget at the planning rates is 6.95 GPU-h (after A6), against the brief's ≈ 10 (Arm 0 ≈ 3, Arm 1 ≈ 1, Arm 2 ≈ 6);
    at NEURONOFF's measured 20-row rates it is 15.0 and the frozen ladder cuts it.

### 14.2 Amendments A1–A14 (2026-10-26, before freeze; adversarial review, every item accepted by the director)

Each is carried in the body; this list says what changed and where.

- **A1** (blocker; §1 B1 row, §5.0, §5.1, §5.7, §5.8, §8.1 test 15; `make_plan.py` `judge_settings`,
  `reexplain_requests`, `reexplain_manifest`). The literal leak detector missed short, non-ASCII and inflected tokens,
  so B1 could be refuted by a detector artifact. New pre-GPU step RE-EXPLAIN: every ONE and every UNK audit pole
  (1,667) is re-explained with the parent's explainer request minus the output-vocabulary lines and the two system
  passages that describe them, through the parent's `openai_compat` DeepInfra backend, model read from the parent
  record file. H_write uses the re-explained labels; the parent's labels give B1 and secondary rows; the literal rule
  is now a Unicode- and stem-aware secondary row checked against the whole judged top-20. Every request and response
  is banked. The coordinator named ONE poles; UNK poles are added so H_ONE and the Arm 2 controls compare labels made
  the same way.
- **A2** (blocker; §2.2, §6 G-gen, §7.3, §8.1 test 12). New gate G-gen at decode steps 1–3; registered failing input a
  hook keyed on the per-forward index; per-cell rule median `realized_cos` < 0.95 → "instrument not shown to fire".
- **A3** (blocker; §1, §5.4, §7.3). Arm 1 reads the paired interval of S_t − S_c over the 20 shared prompts; NO SWING
  needs it to contain 0 and lie inside ±0.10; otherwise UNRESOLVED; the ceiling case is stated.
- **A4** (§1, §7). REFUTED only when the interval is wholly on the wrong side (hi ≤ 0 or the mirror); an interval
  covering the null reads NOT SHOWN unless inside a stated margin (EQUIVALENT); R-pass and R-equiv stamps beside every
  label.
- **A5** (§1, §7.4, §7.5). H_pred_conf is a reported equivalence row (SUPPORTED only inside |ρ| < 0.2); the H_pred_agree
  room check plants ×2.5.
- **A6** (§2.3, §7.2, §12; SCHEMA §3.3). Total effect banked with the pole zeroed at the context position only (one
  forward per context; +0.32 and +0.43 GPU-h); direct path with the scale frozen at both norms (secondary rows
  recompute each norm); qualifying poles chosen on odd-rank contexts, ratios estimated on even-rank contexts.
- **A7** (§5.2, §6 G-coh). The band is frozen on a seeded half of the α = 0 samples and tested on the other half; the
  400 label-prompted continuations are an Arm 2 passing input.
- **A8** (§5.1, §5.7; `make_plan.py` judge draws, Arm 2 builder). Shuffled labels and Arm 2 distractors come from ONE
  poles of the same parent tercile; new fixed-label null (a pole's label against another pole's list), required for
  H_WRITE.
- **A9** (§5.3, §6 G-n, §9.4, §12). Arm 1 projected from a probe on its own prompts and its own row count; Arm 2 keeps
  the sweep's rate; G-n is ≥ 0.7 of drawn ONE poles at every ladder step; the sweep runs first in the Arm 0 job;
  `--budget-min` = ceil(60 × 1.3 × projection).
- **A10** (§5.7, §7.4). Arm 2 intervals bootstrap poles and prompts within pole; the shared α = 0 prompts get their
  own draw per replicate.
- **A11** (§2.2, §5.7, §5.8, §8.1 tests 11, 13, 14; `make_plan.py`). The hook handles a bare tensor output; Arm 2 retest
  seed `[20261025, model_index, 13]`; `test_judge_requests_blind`; "never the other pole of the same neuron" asserted in
  `make_plan.py` for Arm 1 pairs, TeX candidates, Arm 2 controls and every draw; the judge's `max_tokens` (8000) is
  read from the parent's judge request file.
- **A12** (§5.3, §11). Liu 2026 read for its collapse law (Eq. 1, 3, 5; Table 3 m*); held-out prediction α_c = 64 (Q4B)
  and 16 (G4).
- **A13** (§7.4, §11). The H_suff prediction now matches the interval width: at a true 0.40, lo > 1/3 in about two thirds
  of runs (66% in the re-run of the reviewer's simulation; the coordinator quoted 64%), so the predicted label is NOT
  SHOWN, not REFUTED.
- **A14** (§9.1). Full code delta list for the code writer.

**Amendments A15–A17 (2026-10-26, before freeze; re-review, every item accepted by the director).**

- **A15** (§9.4 L3b, L4, L8; §12). `--budget-min` of every cell = max(60, ceil(60 × 1.3 × projected hours)), projected
  from the pre_q4b measured rates (Arm 0) or the arm's own measured rates (Arms 1, 2); the planning-rate values are a
  pre-freeze estimate only (O7 Arm 0: 40 minutes at planning rates against a measured need of about 52).
- **A16** (§1 B1 row, §5.0, §5.1, §7.1, §7.5, §12; `make_plan.py` `reexplain_requests(keep_vocab=True)`, manifest
  `with_vocab` block). B1 could not lose as first defined: original labels are 80% Opus and were selected on passing
  the blind test, so original minus re-explained mixes explainer and regression to the mean. B1 is now the same
  re-explainer with the vocabulary lines minus without them, for the 1,007 ONE poles, pooled over the three models, with
  a room-table row. §5.0 points to `make_plan.py` lines 48–52 for the exact removed strings.
- **A17** (§1, §5.3, §11, AUDIT §4). The Liu held-out α_c is 64 for Q4B and {16, 32} for G4. Arad et al. 2025 is cited
  where input and output scores are discussed; its early-layer input scores lower this study's H_depth probability
  from 0.55 to 0.45 (a prediction change, written in the JOURNAL before any data).

**Amendments A18–A19 (2026-10-26, before freeze; accepted by the director).**

- **A18** (§7.3, §7.5). The Arm 1 R-pass plant is 0.30 (count 1.5), not 0.20: a plant equal to the SWINGS threshold
  reads "underpowered" at R-pass ≈ 0.5 on a near-zero control.
- **A19** (§5.0; SCHEMA §1.7, §4; `make_plan.py`). RE-EXPLAIN ids are `explainer:rx:<pole>` and `explainer:rxv:<pole>`,
  because `neurons_llm.make_request` raises unless an id starts with `<role>:`. The same rule applies to judge requests,
  so judge ids are `judge:<batch>:<6-digit index>` (SCHEMA §4).

Declined: none.

**Post-freeze amendments A20–A21 (2026-10-26, after preflight attempt 1, before any arm and any unblinding; director's
decisions).**

- **A20** (instrument fix; §5.1, §6 G-judge0, SCHEMA §1.2 `judge0_pc`, §3.3 `arm0/firing_tokens.jsonl`; `make_plan.py`
  line 194). Defect: §5.1 takes the G-judge0 firing tokens from a pole's top-1,000 harvested positions, but the frozen
  plan kept only the first 300 (`T[j][:300]`), and the job asserted exactly 20 distinct decoded tokens per pole, so
  Q4B L00N4968+ (3 distinct strings in 300 positions) raised `neuron_rw.py:829` before any arm. Amended rule: the
  positions are the pole's top 1,000 (as §5.1 already said); the firing-token set is the distinct decoded strings at
  those positions, ranked by frequency over the 1,000 (ties by best rank), the top 20 kept; floor 10 distinct strings.
  A `judge0_pc` pole with fewer than 10 is replaced by the next ONE pole in a frozen order
  (`default_rng([20261025, model_index, 14]).permutation` of the model's ONE poles outside the drawn 100) that has at
  least 10, and the replacement is recorded in the plan with its own `shuf1` drawn by the §5.1 rule (a ONE pole of
  its tercile, never the other pole of its neuron). The frozen draw (`choice` of 100, then sorted) has no order past
  its 100 poles, so the "next pole" order is this new frozen permutation (purpose index 14). Counts are made with the corpus tokenizer file the job
  decodes with (`tok/<tokname>/tokenizer/tokenizer.json`, sha256 recorded); the job asserts its count equals the
  plan's and is at least 10. The plan is regenerated as `plans/2` with new hashes. Expected effect from the runner's
  count (`bank/rw/checks/firing_scope.out.txt`): at 1,000 positions only G4 L01N5562+ (1 distinct) falls below 10 and
  is replaced; Q4B L00N4968+ (14), L03N8511+ (11), L33N1238+ (15) and G4 L29N4505+ (10) stay with fewer than 20
  tokens; the code writer confirms. Observation that would show the floor is too low: G-judge0 fails its 0.60 floor on
  poles with 10–12 distinct tokens while passing on poles with more. Checked at the new preflight (`pre_q4b/2`: the
  job's per-pole counts must equal the plan's) and in J0 and A0, where the G-judge0 yes-rate is printed separately for
  the 10–12 group and the rest; the frozen gate still reads all poles, and a failing 10–12 group is reported as "floor
  too low" beside the G-judge0 label.
- **A21** (instrument fix; §2.2 realized dose, §6 G-gen per-cell rule, §7.3, SCHEMA §3.4 `realized_cos`). Defect:
  `realized_cos` was computed against +w for every dose, so negative doses banked about −1.0 (sweep α −1 … −32:
  −0.9974 … −0.999997), and the rule "median `realized_cos` < 0.95 → instrument not shown to fire" would have cancelled
  every negative-dose Arm 1 cell. G-place and G-gen test only α = 1 and the largest positive dose, so no gate caught it.
  Amended: `realized_cos` is the cosine between the realized `Δh = bf16(h + v) − h` and the signed intended vector
  `v = α · RMS_l · w` (w = pole sign · u_i; the director's shorthand α·u_i), equal to cos(Δh, sign(α) · w). The 0.95
  rule is unchanged. Registered inputs, run at α = −1 on the G-place poles and inputs: the real hook must give a
  median cosine ≥ 0.95 (passing input), and a wrong-sign hook that adds −v must give about −1.0 and read "instrument
  not shown to fire" (failing input). Arm 2 is unaffected (positive doses only).
- **A20 addendum (2026-10-26, with the code at `neurons/rw-code` 1b46ff2e, merged at 949e6f4d).** The replacement's
  `shuf1` is drawn from the purpose-14 generator continued after its permutation (a ONE pole of the replacement's
  tercile, never the other pole of its neuron). `make_plan.py` now decodes with the corpus tokenizer file and needs the
  `tokenizers` package, so §9.2 F2 runs with the scratchpad venv's python (`nvenv/bin/python -I
  studies/neuronrw/plan/make_plan.py --out <dir>`), not the system python. The one replacement is G4 L01N5562+ (tercile
  0, 1 distinct string) → L24N8171- (tercile 2, 194 distinct), new `shuf1` L28N2702+ (tercile 2); no other pole of the
  300 is below 10. Plans of `plans/2` (sha256 of the files at 949e6f4d): `arm0_Q4B.json`
  19e5c7810c7145f77fa9fa73c68b94fa14af7da056aa6ac10a8ae624416a1826, `arm0_G4.json`
  54c1eecd76bb078f17279ab77c94c2d0fbd4882e67589220e0a55d74eed9e4ca, `arm0_O7.json`
  5bcb87cfa42fb768e3119fb4fbe29d73960e4c30e3dac8a56c0dad71e16b8cd7, `judge_draws_G4.json`
  c819c3b0ef78e0fcb5cd828260b199c982b6a494e22ca29d66848ddafd6f1561, `budget.json`
  d506fe6d4c315db6808dce336cf62218b7b60e5c218303c508ebad394bf2f5f8; the other plan files are unchanged.

**Post-freeze amendment A22 (2026-10-26; the user's decision; before any judge call and before any unblinding).**

- **A22** (judge-step API budget; deviation from the frozen §12 scale; §5.1, §5.7, §6 G-n and G-judge0, §7.1, §7.4,
  §12). The judge steps J0, A0, J2 and A2 together may spend $30 (the provider limit was raised by exactly $30; about
  $6 was left before), against about 24,300 frozen calls. A22 changes n, not thresholds, margins or decision rules; the
  frozen rule still decides every headline. Counts (worst case, three Arm 2 doses):

  | step and task | frozen calls | A22 calls | A22 rule |
  |---|---|---|---|
  | J0 (Q4B preflight) | 140 | 140 | unchanged |
  | A0 primary | 6,500 (1,300 items × 5) | 1,650 | 150 items per model; ONE items yn_true, yn_shuf1, yn_shuf2, yn_fixed (4); UNK items yn_true, yn_shuf1, yn_shuf2 (3); fc3 dropped |
  | A0 with-vocabulary (B1) | 3,021 (1,007 × 3) | 300 | the subsample's 100 ONE items per model, yn_true only |
  | A0 original labels | 3,900 (1,300 × 3) | 300 | the subsample's 100 ONE items per model, yn_true only |
  | A0 G-judge0 | 600 (100 poles × 2 × 3) | 300 | 50 poles per model |
  | A0 retest | 260 | 90 | 10% of the subsample (15 items per model), yn_true and yn_shuf1 |
  | J2 (Q4B preflight) | 120 | 120 | unchanged |
  | A2 fc3 | 7,200 (360 poles × 4 cells × 5 prompts) | 2,880 | every drawn pole and dose cell, 2 prompts per cell |
  | A2 guesses at d_p | 1,800 | 720 | 2 prompts |
  | A2 label-prompted (G-judge2) | 120 | 120 | unchanged |
  | A2 retest | 732 | 288 | 10% of A2 fc3, seed unchanged |
  | **total** | **24,393** | **6,908** | |

  Rules. (1) Every gate and the 10% retest rate stay; G-judge0 uses 50 of its 100 poles per model; G-retest runs for
  every task that is run (a0 yn, a2 fc3; a0 fc3 is not run). (2) Repeats per item are cut first: fc3 is dropped from
  Arm 0, and yn_fixed is kept only for ONE items, so every term of H_WRITE (D_w, δ_w, D_f) keeps its frozen formula;
  Arm 2 judges neutral prompts 0 and 1 (`default_rng([20261025, 15]).choice(5, 2, replace=False)`, sorted), for every
  pole, dose and α = 0. (3) Items: per model 50 NEURONOFF pairs (strata = the ONE pole's parent tercile, proportional
  quotas by largest remainder, `default_rng([20261025, model_index, 16]).permutation` within each stratum over pairs
  sorted by `pair_id`) plus 50 further ONE poles not in a drawn pair (strata = parent tercile, proportional,
  `default_rng([20261025, model_index, 17])`); the G-judge0 50 are drawn from its 100 within parent-tercile strata by
  `default_rng([20261025, model_index, 19])`. B1 becomes the paired mean of yn_true(with_vocab) − yn_true(reexplained)
  over those ONE items, pooled over the three models (margin ±0.05 unchanged). (4) Arm 2 keeps every pole and dose cell.
  (5) Cost rule: the runner prices each call from its `usage.jsonl` tokens at $2 per million input and $10 per million
  output. After J0 it projects the rest: Arm 0 calls at J0's mean yn cost, every Arm 2 call at J0's mean fc3 cost plus
  1,400 input tokens (ten 128-token continuations); if spent plus projection exceeds $28 it stops before Arm 0 and
  reports. No step (A0, J2, A2) starts if spent plus that step's projected cost would pass $28; after J2, Arm 2 is
  projected from J2's own mean cost. Estimate at about 350 input and 100 output tokens per Arm 0 call and 1,900 and 100
  per Arm 2 call: about $24. (6) G-n Arm 0 becomes at least 140 of the 150 subsample items with all their primary
  answers (failing input 139); Arm 2's G-n is unchanged (0.7 of drawn ONE poles). (7) Widening: a three-way hit rate
  pooled over the three models at d_p goes from about ±2.7 points (1,200 cells) to about ±4.2 points (480 cells); per
  model the Arm 2 on-label half-width goes from about 0.058 to about 0.083 (2 prompts per pole, between-pole spread
  0.15); the Arm 0 D_w half-width per model from about 0.06 (about 340 ONE items) to about 0.11 (100). Predictions
  that keep enough power at the new n: H_write (for D_w ≥ 0.2), the Arm 0 absolute rates, the α = 0 Arm 2 rate, the
  D1 X2 comparison (against an on-label rate ≥ 0.45), the gate rows, and every GPU-only row. Marked underpowered now,
  before unblinding (JOURNAL A22 entry): B1 (predicted L 0.02–0.10 against a pooled half-width about 0.07), H_depth
  (Arm 0; about 0.25), H_ONE (about 0.12; EQUIVALENT unreachable), H_suff's NOT SHOWN band (EQUIVALENT TO CHANCE
  unreachable; SUPPORTED needs about 0.45), H_pred_agree, H_pred_conf, H_depth (Arm 2). The fc3 accuracy row is not
  measured under A22.
- **A22 addendum (2026-10-26, before any judge call; rules the code fixed where A22 was silent, at `neurons/rw-code`
  d6b405c5, merged at 921586de).** (1) The A0 retest items (10% of the subsample, 15 per model) are drawn with
  `default_rng([20261025, model_index, 18])`. (2) The B1 room null is `y_s1 − y_s2` of the re-explained set, because the
  with-vocabulary set no longer has shuffled answers. (3) The with-vocabulary and original-label rows report only the
  yes-rate A, since only `yn_true` is judged for them. (4) The A2 retest is drawn from the fc3 cells only: 288 calls;
  the frozen rule, which also counts the label-prompted cells, would give 300. A22 adopts 288. Judge calls run through
  `neuron_rw_judge run`, which prints the projection and refuses a step past $28 (§9.5, §9.4 L6 and L12). Totals as
  implemented: J0 140, A0 2,640, J2 120, A2 4,008; 6,908 calls; Arm 0 plans byte-identical to `plans/2`.

**Post-freeze amendment A23 (2026-10-26, after preflight attempt 2, before any arm and any unblinding; director's
decision).**

- **A23** (instrument fix; §6 G-repair-fit, §7.2, SCHEMA §3.3 repair fields). Two defects, found on the three preflight
  poles of `pre_q4b/2`.
  (1) Precision mismatch. The frozen gate compared the fp32 manual path (bf16 final residual upcast, fp32 final norm and
  unembedding) with the model's own bf16 NLL. On L04N5557- the gap had median 0.021 and p95 0.070 nats, spread over 71
  of 100 contexts, and on all three poles it scaled with 1 − p(target) at about 0.03: bf16 rounding of logits of
  magnitude 16–32. On tiny fp32 models the manual path equals the model to 0.0, so the formula (epsilon, gain,
  softcap) is right. The frozen gate therefore measured how uncertain a pole's contexts are (L04N5557- had 77
  contexts at NLL ≥ 0.5, the other two poles 5 and 22), not whether the manual path is right. Amended: the gate
  compares the manual path computed in the model's dtype — the same formula (the final norm's epsilon and gain, the
  unembedding, the softcap) evaluated in bf16, new per-context field `clean_nll_manual_model_dtype` — with the model's
  NLL; limits median ≤ 0.01 and p95 ≤ 0.05 nats and the 30% model rule are unchanged. The fp32 manual path still
  produces every repair quantity. The frozen reading is printed beside the amended one, and every self-repair label
  carries both. Registered failing input of the amended comparison: the same model-dtype manual path with the
  final-norm epsilon replaced by the context's own mean(r²) (the normalized vector shrinks by about 1/√2), which must
  exceed the limits on every preflight pole. Observation that would show the amended gate is too loose: that planted
  wrong-epsilon path passing.
  (2) No room in the frozen failing input. The residual read at layer n_layers − 2 had to shift NLL by more than 0.5
  nats; it shifted 0.203 and 0.096 nats on two of the three poles. Amended: the read is at layer n_layers // 2; the
  0.5-nat rule stands; the per-pole shift is banked and must exceed 0.5 on every preflight pole at preflight 3
  (`pre_q4b/3`), else the gate is cancelled as designed.
  No threshold of a hypothesis moves.

**Post-freeze amendment A24 (2026-10-26, after the Arm 0 cells, before any judge call and any unblinding; director's
decision; no hypothesis threshold moves).**

- **A24 (1) G-proj** (§6). The frozen exact-order match read 0.880 on G4 against 0.99, while the numeric gate passed
  (median relative score error 2.8e-4 ≤ 1e-3). Of the 52 mismatches, 41 are order swaps among near-tied tokens and 11
  swap a token at the 10th/11th boundary; our scores sit a median 0.027% above the banked ones, a precision difference,
  not a wrong vector. Amended: a pole's list matches when its top-10 shares at least 9 of 10 tokens with the banked list,
  order ignored; the pass condition is the numeric gate (unchanged, the primary reading) and an overlap share ≥ 0.99.
  Registered failing input: the neighbouring neuron's column (neuron i + 1, same layer and sign), which must fail.
  The frozen exact-order share is printed beside; G4's Arm 0 labels carry both readings (frozen: cancelled; amended:
  as computed). Too loose if the neighbouring-column input reaches 9-of-10 overlap on more than 1% of poles.
- **A24 (2) G-known failing input** (§6). Zeroing layer 0's MLP output moved the known NLL by 3.366 / 1.132 / 0.114 nats
  (Q4B / G4 / O7): no room on OLMo 3. Amended: the MLP outputs of layers 0 through n_layers // 4 are zeroed; the 0.5-nat
  rule stands; the per-model shift is banked, and the layer-0-only shift is kept as a reported reading. It applies to
  O7's Arm 0 re-run and to every later job; the Q4B and G4 Arm 0 cells stand on their passing layer-0 inputs. O7 Arm 0
  re-runs as attempt 2 at the fresh prefix `results/neurons/rw/arm0_o7/3` (the `/2` prefix was named by a P1 re-run that
  created nothing). A P1 re-run may use g6e.4xlarge when g6e.2xlarge has no capacity (three tries found none); the
  preflight outputs were byte-identical across the two instance types. Too loose if the layer-0-only reading ever falls
  inside G-known's tolerance (the gate could then not see a removed first layer).
- **A24 (3) G-coh fallback** (§5.2, §5.3, P8). G-coh failed on G4 twice over (the repeated-token input's NLL 0.555 ≥ L =
  0.2155, not flagged; natural test spans 0.887 < 0.90). Under P8 the run continues with α_c = 64 stamped "coherence gate
  not shown to work", but G-pc already reads 12.32 nats at dose 32 against its 15-nat ceiling, so the G4 Arm 1 and
  Arm 2 jobs would abort at 64. Amended: when G-coh fails for a model, the running α_c is the largest sweep dose whose
  positive-control ΔNLL is below the ceiling, from the Arm 0 bank, else the G-pc dose that passed (G4: 32, the only
  positive-control dose banked); Arm 1, Arm 2 and their pc, place and floor doses use the running α_c, with the §5.3
  rules applied to it (G4: Arm 1 grid ±32, ±16, ±8, ±4, 0; Arm 2 doses {2, 4, 16}, d_p = 16). The frozen α_c
  stays the one Liu's prediction is judged on (G4: 64; prediction {16, 32} missed). That model's Arm 1 and Arm 2 results
  are stamped. Too loose if, at the running α_c, G-pc in the Arm 1 or Arm 2 job exceeds 15 nats or the sample check
  finds the ±α_c continuations mostly not text.
- **Runner rule.** A model whose Arm 0 cell fails twice is dropped from Arm 1 and Arm 2, and the run continues with the
  other models; the run stops only if Q4B drops.

**Post-freeze amendment A25 (2026-10-26, after freeze0, before unblind0; director's decision; nothing unblinded and no
true-label yes-rate seen; δ_w and the H_write decision rule unchanged).**

- **A25** (§7.5 Arm 0 room checks). Readings (`results/thresholds_arm0_{Q4B,G4,O7}.json`): shuffled-label yes-rates on
  the ONE items 0.005 / 0.015 / 0.065; the registered failing input of the room check (R-pass at n = 5) read 1.00 /
  0.95 / 0.83, so the frozen check cannot fail on any model and would stamp the supported labels of H_write, H_ONE and
  B1 (all three models) and of H_depth (Q4B) NOT READABLE. Cause: the frozen plant adds a constant +0.15 to the per-item
  null `y_s1 − y_s2`, which is almost all zeros because the judge almost never accepts a shuffled label, so the planted
  sample has near-zero variance and clears the interval at any n. That is a construction error in the check, not
  evidence about the test: the real statistic draws its variance from the true-label answers, which the check never
  varied. Amended: the plant replaces the true-label answers of the planted group by Bernoulli draws with probability
  `p_sh + E (1 − p_sh)`, with `p_sh` the model's pooled shuffled-label yes-rate on the ONE items and E the row's planted
  effect (H_write 0.15; H_depth 0.20 on tercile 2; H_ONE 0.15 on the ONE side; B1 0.10 on the with-vocabulary side); the
  unplanted group's true-label answers are Bernoulli(`p_sh`); shuffled answers stay as judged. The statistic, interval,
  replicates, n grid, pass bounds and δ_w are unchanged; R-null and R-equiv use the unplanted (+0.0) draws. Registered
  failing inputs: the amended plant at n = 5 must give R-pass < 0.80, and the +0.0 plant must fire the supported label
  at no more than the nominal 5% at n = 5 and fail the 0.80 power requirement at full n. The frozen constant-plant
  reading is printed beside the amended one, and every affected label carries both. Too loose if the amended R-pass at
  n = 5 still reads 0.80 or more on a model: the frozen stamp then stands for that model. Procedure: `freeze0` is
  re-run with the amended check, the frozen thresholds files are kept as `results/thresholds_arm0_<M>_preA25.json`, and
  `unblind0` then runs once.

**Post-freeze amendment A26 (2026-10-26, after Arm 0 was unblinded, before any Arm 1 or Arm 2 label was frozen or
unblinded; director's decision; no hypothesis threshold moves).**

- **A26** (§6 G-known in Arms 1 and 2). `arm1_q4b/1` aborted on G-known: own ΔNLL mean 0.6278 against the banked
  0.6312 (gap 0.00336 > 0.002), while the clean-NLL gap was 0.00082 and 19 of 20 activations were exact. Cause: the
  Arm 1 and Arm 2 stages pass the first 20 known contexts, and each known window is forwarded only up to the last
  context given, so windows are shorter than in Arm 0's 100-context forwards and the bf16 forward drifts at the same
  positions. `arm1_g4/1` passed at 3.5e-7 under the same rule; the drift depends on which windows are shortened. A P1
  re-run would fail the same way and drop Q4B from Arms 1 and 2, ending the run on a window-length artifact. Amended:
  every stage computes G-known on exactly the Arm 0 known-context windows (all 100 contexts, each window forwarded to its
  last Arm 0 context), loaded from the model's Arm 0 plan; the 0.002 tolerance and the A24 failing input are unchanged;
  the truncated-window reading is printed beside the amended one. `arm1_q4b` re-runs as attempt 2 at
  `results/neurons/rw/arm1_q4b/2` (the frozen P1 re-run, under the amended gate). `arm1_g4/1` stands because its gate
  passed; its record notes that the job predates A26 and that A26 changes only how the gate is computed. Observation that
  would show A26 wrong: the full-window G-known still misses 0.002 at `arm1_q4b/2`; Q4B then leaves Arm 1 under the
  frozen P1 rule.

**Post-freeze amendment A27 (2026-10-26, before any Arm 1 or Arm 2 freeze or unblinding; director's decision; no
hypothesis threshold moves).**

- **A27** (§5.2, §5.3, §7.3, §7.4, P8, A24 (3)).
  (1) No change to the coherence instrument mid-run. Under the frozen P8 the coherent-only Arm 1 readouts are CANCELLED
  on every model; the all-sample dose curves are exploratory, stamped, and never a headline.
  (2) G4 and O7 have no validated dose scale: the A24 fallback is refuted by its own check, and the frozen α_c 64 lies
  above their positive-control ceiling. Their Arm 2 cells still run at the A24 doses ({2, 4, 16}, d_p 16) so the samples
  and judgments are banked, but every G4 and O7 Arm 2 label is exploratory, stamped "dose scale not validated; coherence
  gate not shown to work", and excluded from the study-level H_suff verdict. That verdict is decided on Q4B alone (doses
  {2, 4, 32}, d_p 32; α_c 64 measured and matching Liu's held-out prediction).
  (3) If G-coh fails at `freeze2` for a model, the A2 judge runs on all samples of that model without the coherence
  filter, its labels carry the stamp, and the frozen coherent-only reading is printed beside where it can be computed.
  (4) Observation that would show A27 too loose: Q4B's own G-coh failing at `freeze2` with its natural spans or α = 0
  test half below 0.90; H_suff is then stamped on Q4B too, and the study has no headline sufficiency verdict.
  Cost so far: about 5.0 instance-hours of the 12-hour cap; judge spend $4.14, and every later judge step stays under
  the A22 $28 stop rule.

**Post-freeze amendment A28 (2026-10-26, before any O7 Arm 2 judge call; director's decision).**

- **A28 (1) Budget** (A22 cost rule). The user raised the provider limit by $20 for O7's Arm 2, so the A22 parameters
  become `limit_usd` 50 and `stop_usd` 48. Judge spend so far $22.50; O7 A2 projected about $9.55. The code writer
  changes `make_plan.py` accordingly.
- **A28 (2) Held-out prediction P-lit** (from the review's literal-pole finding, §6 of the review file). Definition,
  the reviewer's rule quoted: "These are items where the re-explained label names a token of the pole's own judged
  top-20, by the frozen `literal_overlap` rule; recomputed from the request texts." Computed from banked data only: the
  pole's attested top-20 (`bank/rw/arm0_<cell>/out/arm0/projections.jsonl → top`, the list the Arm 0 judge saw) and its
  re-explained label (`results/reexplain_<M>.jsonl`), through `neuron_rw_core.literal_overlap` (§5.1). Recomputed on
  Q4B's 80 Arm 2 ONE poles it gives the reviewer's 17 literal / 63 non-literal split (G4 16 / 64). O7's split, fixed now
  before any O7 Arm 2 answer: 16 literal of 80 — L01N8975+, L04N3936-, L07N7004-, L11N10245-, L12N9261-, L13N2406-,
  L13N2999-, L13N9773+, L14N671+, L16N1206-, L19N6189+, L21N8785-, L22N3469+, L26N6019+, L29N10781-, L31N1349+ — and 64
  non-literal.
  Prediction: on O7's Arm 2 at its d_p (16), the pole-level paired on-label gain o(d_p) − o(0) (the two-way bootstrap of
  §5.7) has a point estimate of at least 0.30 on the literal poles, and the non-literal poles' interval includes 0.10 or
  lies below it. Refuting observation: a non-literal gain whose interval lies entirely above 0.15. Any other outcome is
  printed "P-lit not confirmed" with both intervals.
  Built from: Q4B Arm 2 literal 17 poles gain 0.50 [0.29, 0.71], non-literal 63 poles 0.10 [−0.03, 0.24]; G4 Arm 2
  (exploratory dose) literal 0.50, non-literal 0.227 [0.086, 0.359]; Arm 0 literal D_w 0.55–0.62 against non-literal
  0.03–0.10 over the three models. O7's labels stay stamped under A27 and outside the H_suff study verdict; P-lit is a
  statement about where the effect sits.
- **A28 (3) Review findings accepted for the closeout.**
  - The frozen rule decides the headline; the amended readings are exploratory. H_write: frozen reading NOT READABLE on
    every model under the frozen room check, and MIXED with one counted model under A24 alone. H_suff: frozen reading
    CANCELLED under P8, and MIXED (one counted model) under the two-model study rule that A27 replaced.
  - The §11 predictions are added to the missed count: H_suff (predicted NOT SHOWN, observed SUPPORTED 0.531), the D1 X2
    comparison, B1 (observed EQUIVALENT 0.013), the Arm 0 absolute rates, H_depth (Arm 0; observed MIXED), the gate row
    (G-known failed in `arm0_o7/1` and `arm1_q4b/1`), and the α = 0 Arm 2 rate (G4 0.244). "11 of 18" was a count over a
    partial list.
  - Dose-32 repetition: about 29% of Q4B ONE cells by the reviewer's distinct-bigram metric (chosen after seeing the
    data), against 4.4% collapsed by the coherence band.
  - Liu's G4 verdict is reworded from "missed" to "not decidable with this coherence instrument" (the sweep band failed
    G-coh on G4).
  - The missing stamps are added to `results/grid.md` and `results/study_arm2.json` (and the G4 H_write row of
    `verdict_arm0_G4.json`).
  - Novelty is narrowed to Arad et al. 2025's token-identity measure: Arm 0's agreement sits in the token-identity overlap
    their input score counts; what is new is the object (MLP neuron poles in three base-model families) and the read
    side (labels written without the output tokens).

## 15. References

Read for the claims used (AUDIT §4 records queries and locations): Geva et al. 2021 (EMNLP), §4 and Figure 4; Gurnee
et al. 2024, §5.1; Rushing & Nanda 2024 (ICML), §2–3; Liu 2026 (arXiv 2606.19831), Eq. 1, 3, 5, §3.2 and Table 3
(read 2026-10-26 through the arXiv HTML for the collapse law and the Qwen3-4B and Gemma-3-4B m* values); Arad,
Mueller & Belinkov 2025 (arXiv 2505.20063, EMNLP 2025), abstract, §3.1–3.4, §5 and Figures 3 and 6 (read 2026-10-26 for
the input-score and output-score definitions, their co-occurrence and their depth). Cited only through `studies/neuronoff/LIT_steering.md`, as read
by the NEURONOFF literature agent: Gur-Arieh et al. 2025; McGrath et al. 2023; Kazemi 2026.
