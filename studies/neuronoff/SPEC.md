# NEURONOFF — does an input-clean neuron pole matter when it is turned off?

Answer the study will give, in one line: per model, the median paired difference in held-out next-token NLL
increase (nats) between ablating a pole whose explanation passed the parent's §6.1 test (class ONE) and ablating a
matched pole whose explanation failed it or was declined (class UNK), with its 95% interval, read against zero and
against a margin frozen on UNK-against-UNK pairs; plus seven code-construct case studies with samples.

## 0. Status and provenance

- Status: **FROZEN 2026-10-25T18:17:16Z**, tag `neuronoff-freeze-1` (§13 freeze record). It was a draft until the tag (§9.2 step F4),
  made after the offline gate passes and before the first GPU launch. Nothing in §4–§7 changes after the tag except
  by a dated amendment in §13.
- Branch `neurons/off`, worktree `../interp_research-neuronoff`, based on `neurons/harvest` (bca491a9) because
  `neurons/explain` lacks `infra/pg/neurons_harvest.py` (the hook code this study reuses). The banks are not under
  git; they are read from the two neurons worktrees by absolute path (§4.4).
- Parent: "already clean" D1, `../interp_research-drafts/studies/already-clean/SPEC_D1.md` (frozen 2026-10-20),
  `results/RESULTS.md` (unblind 2026-10-24T04:06Z, metrics sha256 a95bfa57…), `handoff/S6.md`.
- Director's brief: `NEURONOFF_BRIEF.md` (session scratchpad, 2026-10-25). Changes from it are listed in §14.
- Companion files: `LIT.md` (literature, queries recorded), `AUDIT.md` (prior measurements; none intervene on these
  neurons), `SCHEMA.md` (file contract, v4), `plan/` (frozen selection; built by `plan/make_plan.py`),
  `prompts/` (Arm B, blind writer).
- Code: `infra/pg/neuron_off_core.py`, `infra/aws_pg/jobs/neuron_off.py`, `infra/aws_pg/jobs/neuron_off_dryrun.py`,
  `tests/test_neuron_off_core.py` on branch `neurons/off-code` (0d239ab1, c5da7217, built to SCHEMA v2). The code
  delta owed before freeze is §9.1.
- Amended 2026-10-25 after an adversarial review accepted by the director: A1–A12, listed in §13 with what changed.
  The body below already carries every amendment.

## 1. Question, decision, hypotheses

**Object.** A neuron pole, as in the parent study. For neuron i of layer l, `a = act(g)·u` is the down_proj input
(`g = W_gate x`, `u = W_up x`, act = silu for Q4B and O7, gelu_tanh for G4). The positive pole is the set of tokens
with a > 0, the negative pole those with a < 0. The parent assigned each of 600 audit poles per model a class:
ONE (one job, explanation passed the blind test), TWO-REL, UNREL, UNK (UNK-F failed, UNK-N declined), RARE.

**The contrast, named exactly (A3).** "Explanation passed §6.1" (class ONE) against "explanation failed §6.1 or the
explainer declined" (class UNK). It is not "one job against illegible": in the kept pairs the explainer itself gave
the control a one-job label for 85% (Q4B), 85% (G4) and 79% (O7) of controls; those explanations then failed the blind
test. Only 45 (Q4B), 55 (G4) and 47 (O7) of the kept pairs keep both classes under θ_s ± 0.1; they form a frozen
secondary row (§7.4).

**Question.** When a ONE pole is turned off, does the model lose more on the pole's own strongest held-out contexts
than when a matched UNK pole is turned off? Does it lose less on general text? And what does turning off a
code-construct pole do to the code the model writes?

**Decision.** If ONE poles are more necessary on their own contexts (H_cause), the parent's legibility class is a
cheap proxy for causal role, and the next neuron study selects ONE poles for circuit tracing. If there is no
difference (H_morcos), legibility from inputs says nothing about function; the next study must measure each pole's
output role directly and the parent's Q1 fractions are read as statements about the observer only. If ONE poles are
less necessary (H_readout), clean poles are read-outs; the next study follows their downstream readers.

**Hypotheses and refuting observations** (Arm A, own-context readout; M, lo, hi, δ defined in §7.1–§7.2; amended A1):

| hypothesis | claim | supported when | refuted when |
|---|---|---|---|
| H_cause | ONE poles are more necessary on their own contexts than matched UNK poles | lo > 0 and M ≥ δ | hi ≤ 0 |
| (positive, below δ) | ONE poles are more necessary, by less than two matched UNK poles typically differ | lo > 0 and M < δ | — (a label, not a hypothesis) |
| H_morcos | passing the explanation test does not predict necessity (Morcos et al. 2018) | lo ≤ 0 ≤ hi, −δ < lo and hi < δ, room check R-b passed | the interval excludes 0 |
| H_readout | ONE poles are less necessary: they read the concept and do not write it | hi < 0 and M ≤ −δ | lo ≥ 0 |
| (negative, below δ) | ONE poles are less necessary, by less than δ | hi < 0 and M > −δ | — |
| H_spec (collateral readout) | ONE poles do less damage on general text | hi_c < 0 and M_c ≤ −δ_c | lo_c ≥ 0 |
| B1 energy confound | any difference comes from the size of the removed vector, not the class | the energy-normalized readout (§7.4) gives a different direction from the raw one, or G-bal fails | the two directions agree and G-bal passes |
| B2 instrument fault | the hook does something other than zero the pole | any gate of §6 fails | every gate of §6 passes, each with its registered failing input failing |
| B3 explainer confound | ONE against UNK is partly Opus against Sonnet (AUDIT §3) | the label differs between the two explainer strata (§7.4 row) | the strata agree in sign. Note: pairs are matched within layer, and the explainer model is nearly balanced in the kept pairs (Opus on the ONE side 75 / 75 / 70, on the control side 75 / 74 / 68) |

An interval that sits inside (−δ, δ) but excludes 0 is reported as "directional, below δ", not as H_morcos and not
as a refutation of H_cause. A directional label (H_cause, H_readout or either "below δ" label) is a headline only if
the energy-normalized readout has the same direction (A5); otherwise the model's headline reads "direction not
robust to energy normalization".

**Arm B hypotheses** (per pole; D, m_B* defined in §5.5): SPECIAL (the pole is causally special for its label):
mean D > m_B* and the 95% interval of D excludes 0. Refuted (NOT-SPECIAL) when mean D ≤ m_B*: the pole's drop is
within what a matched UNK pole's drop differs by from another matched UNK pole's.

## 2. The measurement, exactly

**Instrument (shared by both arms).** A forward pre-hook on `mlp.down_proj` of layer l receives the model's own
bf16 `a` (as `neurons_harvest.MLPTap._down` reads it). It returns `a'` with `a'[:, p, i] = 0` at every position p in
the mask domain where `sign(a[:, p, i])` equals the pole's sign, and every other entry bit-identical. Arithmetic, if
any, in fp32, cast back to bf16. Mask domain (SCHEMA §6): corpus or text positions with window (text) position ≥ 1;
BOS and window position 0 are never changed, as the harvest never scored them. Generated positions are inside the
domain. The other pole of the same neuron is untouched. The identity hook (G-ident) uses the same mask and path and writes the
original values back (A7). Weights bf16, SDPA attention, `inference_mode`, deterministic kernels (G-det),
`use_cache=False` for teacher forcing, cached generation for sampling. Zero ablation only (not mean ablation).

**Arm A per-pole quantities.**
- Held-out contexts of pole p: 100 global token positions t from the parent harvest's exact top-1000 (§4.3).
- Input for context t: the corpus window that holds t, from its first token to t inclusive, BOS prepended when
  the tokenizer defines one. This is the exact prefix the harvest forwarded, so the pole's activation at t is
  reproduced (§6.4). Contexts sharing a window share one forward up to the window's last held-out position.
- `dnll_t = NLL_ablated − NLL_clean` of the true next corpus token `tokens[t+1]` from the logits at t, in nats.
- **own_dnll(p)** = mean of dnll_t over the 100 contexts (nats). **coll_dnll(p)** = mean over the collateral targets
  (the first 16 corpus windows of that model's tokenization, scored by the harvest rule: targets `input[:, 1:]` from
  `logits[:, :-1]`, BOS prepended when defined; 32,752 targets for Q4B, 32,768 for G4 and O7) of the per-target NLL
  difference, ablated minus clean (nats). The clean collateral pass runs once per model.
- **removed norm** at context t = `|a_ours(t)| · ||W_down[:, i]||₂` (fp32), banked per context.
- Before-model and after-model: the same checkpoint (Q4B `Qwen/Qwen3-4B-Base` @ 906bfd4b…; G4
  `google/gemma-3-4b-pt` @ cc012e0a…; O7 `allenai/Olmo-3-1025-7B` branch `stage1-step1413814` @ 373bad25…) without and
  with the hook. Pairing: per context (clean and ablated on identical inputs), then per pair of poles (§4.2).

**Summary.** Per model: M = median over valid pairs k of `D_k = own_dnll(one_k) − own_dnll(ctrl_k)`, 95% percentile
bootstrap over pairs (10,000 resamples, numpy default_rng(20261025)). Same for `coll_dnll` (M_c). Also reported: the
median own_dnll and coll_dnll on each side with 95% bootstrap intervals (absolute scores), and the median clean NLL
on each side.

## 3. Design

| piece | what it adds | without it |
|---|---|---|
| Arm A, ONE against matched UNK, own contexts | the decisive contrast: necessity on the pole's own job | — |
| null pairs (UNK against matched UNK) | the margin δ: how much two matched non-ONE poles differ by chance and by mismatch | a "difference" could be the size of ordinary pole-to-pole variation |
| collateral readout | H_spec, and Morcos/Zhou's overall-damage readout | no view of off-target cost |
| matching on count and energy (§4.2), realized-norm balance gate | removes "ONE poles are simply stronger" (B1) | Dalvi et al. 2019's warning: an unmatched contrast measures magnitude |
| whole-layer positive control | shows the hook location and the readout register a known large effect | a null could mean a dead instrument |
| identity pair, floor coordinate, reproduction of banked a | B2 | a bug would read as a result |
| Arm B | what turning off a construct pole does to written code, with samples | the population number has no picture |
| three models | replication across families (Qwen3, Gemma 3, OLMo-3) | one family |

## 4. Data and items

### 4.1 Models, weights, corpus

Weights: Q4B and G4 from the S3 cache (`--weights models--Qwen--Qwen3-4B-Base`, `--weights models--google--gemma-3-4b-pt`;
both listed in `infra/pg/matrix_repo_facts.json:_meta.bucket_dirs`); O7 by in-job Hub download (`--no-weights
--revision stage1-step1413814`, as the parent's W1 harvest). Corpus: the parent's 50M-token corpus,
`s3://<s3-bucket>/results/neurons/w1/q4b/1/out/corpus/corpus`, tokenizer arrays `tok/<qwen3|gemma3|olmo3>/`.
The job asserts the corpus `tokens_sha256` equals the plan's (copied from each harvest manifest).

### 4.2 Arm A pole selection (frozen in `plan/pairs_<M>.json`, built by `plan/make_plan.py`)

Eligible: poles of the parent audit sample with class ONE or UNK (UNK-F or UNK-N) in
`../interp_research-neurons-explain/bank/w5/verdict/samples/q1_<M>_poles.jsonl`, a complete harvest top-1000, and
100 held-out contexts under §4.3. The control class is UNK, the illegible class, not "any non-ONE": TWO-REL and UNREL
poles passed the legibility test too (14–18% of non-ONE poles), and including them would blur the contrast (§14).

Matching variables, from banked arrays only (amended A5): `density` = the parent's stats `density` (tokens with
|a| ≥ 0.01·max|a|, "a few times a page") over scored tokens, the count of positions where the pole fires with any
strength; `meanabs` = mean |a| over the pole's sign tokens from the harvest's log-histogram, geometric bin centres
(energy on general text); `ho_med` = median |a_banked| over the pole's 100 held-out contexts (energy at the readout
positions). The sign share `rate` was dropped as a matching variable: it lies in 0.30–0.72 for every pole and never
differs by 1.5× within a candidate pair, so it matched nothing; it is still recorded. Distance
`d = max_k |log(x_one,k / x_ctrl,k)|` over the three variables; caliper `d ≤ log 1.5`; same layer.

Procedure (`make_plan.py`, seed 20261025): Arm B targets first take their two controls (§5.2); then within each layer
the ONE poles in seeded random order each take the nearest unused UNK pole within the caliper; the feasible pairs are
shuffled with the same generator and the first 100 kept; **the controls of the dropped pairs return to the pool
(A11)**; each kept control then takes the nearest unused UNK pole of its layer within the caliper as its null partner
(`source: extra`); kept controls left without a partner are paired with each other within a layer by the same rule
(`source: control-control`, no extra compute).

**Estimand (A4).** ONE poles inside UNK's energy and layer range: ONE poles that have, in their own layer, an UNK pole
within a factor 1.5 on all three variables. It is not all ONE poles. Counts (`pairs_<M>.json` → `estimand`):

| model | ONE eligible | ONE with a match | unmatched | kept pairs | null pairs (extra + control-control) | class-stable pairs |
|---|---|---|---|---|---|---|
| Q4B | 354 | 96 | 258 | 96 | 33 (20 + 13) | 45 |
| G4 | 335 | 97 | 238 | 97 | 38 (23 + 15) | 55 |
| O7 | 311 | 152 | 159 | 100 | 67 (56 + 11) | 47 |

Kept-against-eligible shift (medians; eligible ONE → kept ONE, eligible UNK beside): density Q4B 0.114 → 0.168 (UNK
0.184), G4 0.095 → 0.154 (0.162), O7 0.132 → 0.172 (0.177); meanabs Q4B 0.076 → 0.086 (0.103), G4 0.043 → 0.035
(0.040), O7 0.021 → 0.035 (0.038); ho_med Q4B 6.29 → 5.22 (5.19), G4 4.10 → 2.23 (2.27), O7 1.29 → 1.97 (1.59). The
kept ONE poles are denser-firing than the typical ONE pole in all three models and, in G4, weaker at their held-out
positions. Results speak to the kept set only.

Balance of kept pairs, median log ratio ONE/control: density +0.002 / +0.046 / −0.059, meanabs −0.031 / +0.042 /
−0.010, ho_med +0.046 / +0.067 / +0.076 (Q4B / G4 / O7). ho_med leans toward ONE in all three models, which is why a
directional headline needs the energy-normalized readout to agree (A5) and why the per-pair log density ratio is a
covariate row (§7.4). Depth terciles of kept pairs: Q4B 10 / 50 / 36, G4 21 / 49 / 27, O7 28 / 39 / 33; shallow Q4B
and G4 layers are under-represented because UNK poles are rare there (parent RESULTS). This is a scope limit, stated
with every result.

### 4.3 Held-out contexts (frozen in `plan/plan_<M>.json`)

The harvest bank holds, for every audit pole, the exact running top-1000 over the 50M corpus (`poles/top1000.npz`,
`tok` and `a`, window position 0 excluded; confirmed for all 1,800 poles: Q4B 720 rows, G4 600, O7 640, every audit
pole present; a few dead Q4B poles have fewer than 1,000 entries and are ineligible). The explainer saw ranks 0–29
(the top 30 of `topk/`). Rule: candidate ranks 30, 31, … of top1000 in order; drop a candidate whose token lies in
the same document as one of the 30 seen contexts (from either array) and within that context's rendered window
(256 tokens before, 16 after); drop a candidate at window position 2047 (its next token is outside the window); keep
the first 100. A pole with fewer than 100 such contexts is ineligible. The brief said "ranks 31–130"; the exclusion makes the
used ranks run past 130 for poles whose strongest firing clusters in a few documents (§14).

### 4.4 Inputs, by path and hash

`plan/pairs_<M>.json` and `plan/plan_<M>.json` record the sha256 of the verdict sample file, harvest manifest, top1000
and document offsets they were built from. `plan/budget.json` holds forward-token counts and plan sha256
(regenerated after A5): Q4B `7af5184d…`, G4 `b280de5c…`, O7 `cef5db7d…`. Re-running `python3 -I studies/neuronoff/plan/make_plan.py --out
studies/neuronoff/plan` must reproduce these hashes byte for byte; F2 in §9.2 checks it.

## 5. Arm B — code-construct case studies

### 5.1 Poles, verified against the verdict file (2026-10-25)

All seven IDs exist and are class ONE with a passing explanation (Opus explainer). No ID or label was wrong. Labels
as given to the blind writer, with the firing token from the parent explanation (not shown to the writer):

| pole | model | label | parent explanation, where it fires |
|---|---|---|---|
| L07N10202- | G4 | React useEffect | the "(()" right after useEffect in `useEffect(() => {`, useEffect in a react import list, componentWill… |
| L06N599+ | G4 | Python import … as | the keyword " as" in `import X as Y` / `from X import Z as Y` |
| L08N7823+ | G4 | SQL REFERENCES | REFERENCES right after `FOREIGN KEY (col)`; the idx suffix of a foreign-key index name |
| L11N5238+ | Q4B | SQL FOREIGN KEY | " KEY" right after FOREIGN in a constraint clause |
| L27N2924- | Q4B | Angular | the token right after "Angular" (version, "by", "@angular/", CLI, Material); AngularJS `$scope`, `$http` |
| L08N3398- | Q4B | HTML title attribute | " title" as an attribute name inside an open tag or `attr('title')` |
| L04N5165+ | Q4B | TeX | ConTeXt, "context", "environment", TeX macro-name pieces (`\setuphead`, `\documentclass`, `\normalfont`) |

Three are negative poles; F41 (AUDIT §3) touched only promoted-word clauses, which none of these labels uses.

### 5.2 Controls

Each target takes the two nearest UNK poles of its layer by the §4.2 distance (density, meanabs, ho_med), with no
caliper (layers hold 2–6 UNK poles; most targets have no UNK pole within log 1.5). ctrl1 is the matched control for
every readout; ctrl2 exists only to freeze the margin m_B. Pairs (d1, d2), regenerated after A5: Q4B L11N5238+ →
L11N4144+ (0.61), L11N4304+ (0.91); L27N2924- → L27N7674+ (0.06), L27N8528- (0.13); L08N3398- → L08N9674- (1.43),
L08N6028+ (1.43); L04N5165+ → L04N9078- (1.03), L04N5781- (1.68); G4 L07N10202- → L07N5834- (0.46), L07N4140+ (0.50);
L06N599+ → L06N8972- (0.45), L06N2449- (0.85); L08N7823+ → L08N6441+ (0.43), L08N7758- (2.71). A control with d > log 2
(0.69) differs by more than a factor 2 on some variable; that pole's verdict is printed with "poor control match" and
its d values (Q4B L08N3398-, L04N5165+, L11N5238+ ctrl2; G4 L06N599+ ctrl2, L08N7823+ ctrl2).

### 5.3 Prompts and readouts

Prompts: `prompts/<pole>.jsonl`, 20 per pole, written by one Opus subagent that saw only the seven label strings and
general instructions (never a context, explanation or score). Records carry `prompt`, `target` (the writer's guess of
the construct's first token), `target_position: "end"` and a fourth key, `continuation` (the construct text, scored
token by token). Tokenization (A9): every call is `tok(text, add_special_tokens=False)`, and the job prepends BOS
itself when the tokenizer defines one (Gemma's tokenizer would otherwise add a second BOS). Input: BOS (when defined)
+ prompt tokens + continuation tokens. Continuation token ids are `tok(prompt + continuation)[len(tok(prompt)):]`
when `tok(prompt)` is a prefix of `tok(prompt + continuation)`; a record without that prefix property is excluded
and listed (if more than 2 of 20 fail, the pole's Arm B verdict is cancelled). Checked 2026-10-25 against the local
`tokenizer.json` copies in the corpus folder (`tokenizers` 0.21.1, no weights): 139 of 140 records hold the prefix
property; Q4B L04N5165+ prompt 7 fails (`… 94.7` + `\%`) and is excluded. F1 repeats the check with the job's own
tokenizer. 17 of 140 continuations start with a whitespace-only token (L11N5238+ 7, L04N5165+ 6, L07N10202- 4).

Conditions (teacher forcing): clean, identity, ablate_pole, ablate_ctrl1, ablate_ctrl2, ablate_layer (whole MLP of the
target's layer). Readouts per prompt j, for condition x: `Δ_x,j = logp_clean,j − logp_x,j` (positive = the expected
continuation became less likely), in nats:
- **R1 (primary)**: sum of log-probabilities over all continuation tokens.
- R2: the brief's "target" readout: the summed log-probability of the continuation tokens up to and including the
  first token whose decoded string contains a non-whitespace character (A9). For a record whose first token is not
  whitespace-only this is the first token's log-probability; otherwise it adds the leading whitespace token(s).
- R3: mean over the continuation's scored positions whose input token carries the target pole's clean activation
  with the pole's sign and `|a| ≥ 0.3 · banked_max_abs` (`pole_a_inputs`, SCHEMA §3.4); prompts with no such position
  are excluded from R3 and counted. R3 asks whether the loss sits right after the pole fires.

Sampling: conditions clean, ablate_pole, ablate_ctrl1; 20 completions per prompt per condition; temperature 0.7,
`top_k = 0`, `top_p = 1.0`, `repetition_penalty = 1.0`, 128 new tokens, stop at EOS allowed; the model's own
`generation_config` sampling defaults are overridden explicitly and the used config is banked. Seeds:
`torch.manual_seed(20261025 + 1000·j)` before the batch of 20 for prompt j, identical across conditions, so a
clean and an ablated completion diverge only where the ablation changed the logits. Every completion is banked.

### 5.4 Construct counts (frozen regexes)

Applied to the string `prompt[-200:] + completion`; a match counts only if it ends inside the completion. Per
completion: presence (≥ 1 match) and the number of matches. Python `re`, flags as written.

```
L07N10202-  \buseEffect\b
L06N599+    \bimport\s+[\w.]+\s+as\s+\w+|\bfrom\s+[\w.]+\s+import\s+\w+\s+as\s+\w+
L08N7823+   (?i)\breferences\s+[\w.`"\[\]]+\s*\(
L11N5238+   (?i)\bforeign\s+key\b
L27N2924-   \bAngular(JS)?\b|@angular/
L08N3398-   \stitle\s*=\s*["']|\btitle\s+attribute\b|attr\(\s*['"]title
L04N5165+   \\[A-Za-z]+|\\[%&$#_{}]        (presence saturates; the per-completion count is this pole's readout)
```
Each line is the pole id, spaces, then the pattern exactly as passed to `re.compile` (a raw string). The TeX pattern
is a backslash followed by letters, or a TeX control symbol. The patterns were checked on the 140 written
prompt + continuation pairs before freeze (a match ending in the continuation: 20, 20, 20, 20, 20, 20, 20 of 20 per
pole), so no pattern is blind by construction.

Behaviour changed: the presence-rate drop (clean minus ablate_pole) exceeds the drop under ablate_ctrl1 by at least
0.10, and the 95% bootstrap interval over prompts of that difference excludes 0. For L04N5165+ the same rule on the
mean count per completion, with threshold 1.0 macro per completion.

### 5.5 Arm B decision rule (per pole, on R1; amended A8)

`D_j = Δ_pole,j − Δ_ctrl1,j`; null `D0_j = Δ_ctrl1,j − Δ_ctrl2,j`. `m_B` = the 97.5th percentile, over 10,000 bootstrap
resamples of the 20 prompts (default_rng(20261025)), of `|mean(D0)|`, **without sign flips**: a systematic difference
between the two controls stays in the margin. The margin used is `m_B* = max(m_B, 0.02 nats)`; 0.02 nats is the
frozen floor (a 2% change in the probability of the whole construct). Both are frozen from the control lines before
any target line is read (§9.3).
SPECIAL: mean(D) > m_B* and the 95% bootstrap interval of mean(D) has lower end > 0. NOT-SPECIAL: mean(D) ≤ m_B*.
UNRESOLVED: otherwise.
Room (floor): the pole's mean Δ_layer must be ≥ 5·m_B* and ≥ 0.1 nats, else the verdict reads "instrument not shown
to fire" (cancelled, not NOT-SPECIAL). Ceiling: mean Δ_pole ≤ mean Δ_layer (one coordinate removing more than its
whole layer points to a hook fault); above it the verdict is cancelled and the pole is flagged for the checker.

### 5.6 Pre-registered predictions (written 2026-10-25, before any GPU run)

| pole | R1 verdict | Δ_pole R1 (nats) | behaviour changed | where the loss sits |
|---|---|---|---|---|
| G4 L06N599+ (import as) | SPECIAL | 0.05–1.0 | no | on the alias token after " as" (R3 > R2) |
| Q4B L11N5238+ (FOREIGN KEY) | SPECIAL | 0.05–1.0 | no | on the token after " KEY" |
| G4 L08N7823+ (REFERENCES) | SPECIAL | 0.05–1.0 | no | on the table name after REFERENCES |
| Q4B L08N3398- (title attribute) | SPECIAL | 0.05–1.0 | no | on `="` after " title" |
| G4 L07N10202- (useEffect) | NOT-SPECIAL | < 0.1 | no | most continuations do not contain "(()" |
| Q4B L27N2924- (Angular) | NOT-SPECIAL | < 0.1 | no | the pole fires after "Angular", mostly past the scored span |
| Q4B L04N5165+ (TeX, layer 4) | NOT-SPECIAL | < 0.1 | no | — |

Summary predictions: SPECIAL for at least 3 of 7 poles, probability 0.6; behaviour changed for at least 1 of 7,
probability 0.2; where SPECIAL, no substitute construct dominates (the model still writes the construct; the loss is a
spread of probability, not a switch). A prediction that lands elsewhere is reported as missed.

## 6. Instrument readouts and gates

Every gate has a floor, a ceiling and a cancelled branch, and a registered input that makes it fail; the failing
input runs in the same job and its result is banked beside the gate. NaN or a missing value always takes the
cancelled branch.

| gate | scope | pass condition (floor / ceiling) | cancelled branch | registered failing input (must fail) |
|---|---|---|---|---|
| G-known (known number, preflight principle) | model, before any pole | `|ours − banked| ≤ 0.005` nats on the mean NLL over the first 120,000 corpus targets (banked: Q4B 3.0893311075846355, G4 3.002326672363281, O7 2.675372037760417); floor 0 | job aborts (`status: aborted_gate`); §9.5 pivot P1 | the same targets with layer 0's whole MLP zeroed on the mask domain; it moves the mean by far more than 0.005 (a removed first MLP layer is catastrophic); the coordinator's point stands: a one-window shift could move the mean by less than 0.005 and is not used |
| G-det (determinism; amended A7) | model | with deterministic kernels (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, SDPA math or flash forward), the clean forward twice on corpus windows 0–1: `max|Δlogit| = 0`; floor and ceiling 0 | model's Arm A and Arm B labels cancelled | the second forward with input token 100 of window 1 replaced by the next token id: `max|Δlogit| > 0` at the positions after it |
| G-ident (identity pair; amended A7) | pole (first held-out window) and Arm B prompt | the identity hook runs the pole's real firing mask through the same scatter-and-cast path as the ablation and writes the original values back (`a'[mask, i] = a[mask, i]` via fp32): `max|Δlogit|` against clean `= 0`; floor and ceiling 0 | pole cancelled (its pair cancelled) | `a := 1.5·a` at every neuron of the layer of the model's first Arm A pole, on its first window: `max|Δlogit| > 0` |
| G-floor (floor coordinate) | layer, once per model | on corpus window 0, the first slice of length 2048, 512, 128, 64 or 16 tokens in which some (neuron j, sign σ) of layer l has no σ-sign entry in the mask domain; lowest such j; ablating (j, σ) gives `max|Δlogit| = 0` exactly; floor and ceiling 0 | a layer with no such coordinate at 16 tokens is "not constructible"; the model's Arm A labels are cancelled if any constructible layer is nonzero or fewer than 80% of used layers are constructible | ablating (j, −σ), which has entries at every domain position of the slice: `max|Δlogit| > 0` |
| G-fit (reproduction of banked a, "fit on both sides"; amended A2) | pole, both sides of every pair | over the 100 held-out contexts, `|a_ours/a_banked − 1|`: median ≤ 0.03 and 95th percentile ≤ 0.15, zero sign mismatches; floor 0 (an exact match is allowed and flagged for the checker) | pole cancelled (its pair cancelled) | the same comparison of a_banked (neuron i) against a_ours of neuron i+1 at the same positions, read in the same forward. Expected: median relative error above 0.5 and sign mismatches near half the contexts (an unrelated coordinate), so the gate fails. The earlier input (a_banked shifted by one rank) is withdrawn: adjacent top-1000 ranks differ by 0.13–0.15%, and it passed on 727 of 727 poles (reviewer) |
| G-pc (positive control) | pole, first 10 held-out contexts | mean whole-layer ΔNLL `≥ 0.05` nats and `≤ 15` nats | pole cancelled; if more than 30% of a model's Arm A poles are cancelled by G-pc, that model's Arm A labels read "instrument not shown to fire" | the identity hook in the positive-control path gives 0 < 0.05 |
| G-bal (energy balance) | model | `B = median_k log(E_one,k / E_ctrl,k)` with `E` = mean removed norm over the held-out contexts: `−log 1.5 ≤ B ≤ log 1.5` | NaN: model cancelled | the same B with every `E_ctrl` divided by 1.6 when B ≥ 0 and multiplied by 1.6 when B < 0: |B| grows by log 1.6 = 0.47 > log 1.5, so it always leaves the band; the analysis runs it on the real data and reports it (A7) |
| G-n (sample size) | model | valid pairs `≥ 60` and valid null pairs `≥ 20` | model's Arm A labels cancelled ("underpowered after gates") | the same check on the first 59 valid pairs, and on the first 19 valid null pairs: must fail (analysis, on the real data; A7) |

Readouts carried by every Arm A number before unblinding (CLAUDE.md Instruments §4): fit on both sides (G-fit for
ONE and control poles), a positive control that fired (G-pc), and the identity-pair value (G-ident). A pair missing
any of the three is cancelled and stamped in `verdict_<M>.json`.

## 7. Frozen decision rules and statistical plan

### 7.1 Per-model statistics (Arm A)

Valid pair: both poles pass G-ident, G-fit and G-pc. Valid null pair: both poles pass the same three. For valid pairs,
`D_k` as §2; M = median, [lo, hi] = 95% percentile bootstrap interval over pairs (10,000 resamples, seed 20261025).
Collateral: `D^c_k = coll_dnll(one_k) − coll_dnll(ctrl_k)`, M_c, [lo_c, hi_c].

### 7.2 Thresholds frozen on control data

From valid null pairs only: `d_j = own_dnll(ctrl_j) − own_dnll(extra_j)` and `d^c_j` likewise. **δ = median_j |d_j|**,
**δ_c = median_j |d^c_j|**: the typical difference between two matched non-ONE poles (about 0.67 of the null
difference's spread for a normal null). δ is a size margin for "meaningful", not a significance threshold: a
95% interval of the median at n = 100 has a half-width of only about 0.24–0.36·δ (simulation, below), so the labels
of §7.3 read zero and δ separately (A1). Written to `results/thresholds_<M>.json` and committed before `unblind` runs.

Room checks, computed by `freeze` from control data and stamped in the same file:
- R-a: δ > 0 and finite; else the model's Arm A labels are cancelled ("no range").
- R-b (room for H_morcos): `hw` = half-width of the 95% bootstrap interval of the median of n resampled values of
  `s_j·d_j`, n = 100 (the planned pair count), fixed signs s_j from default_rng(20261025). H_morcos can be supported
  only if `hw < δ` (an interval that covers 0 must be able to fit inside ±δ); if not, H_morcos is stamped "no room"
  for that model and only directional labels can be read.
- R-c (room for H_cause and H_readout): add 1.5·δ to every `s_j·d_j`; draw 2,000 bootstrap samples of size n; the share
  in which the H_cause rule fires must be ≥ 0.8. If not, the model is stamped "underpowered for a direction".
- Registered failing input for R-b and R-c (A1): the same computation at n = 5 must fail both; if it does not, the
  room checks are cancelled for that model ("room check cannot fail") and H_morcos cannot be supported. The reviewer's
  n = 10 is also computed and reported, not used as the hard input: in simulation (`room_sim.py` logic, normal and
  lognormal-difference nulls of 33 and 67 pairs, 40 null sets each) n = 10 failed R-b in 83–100% and R-c in 77–100%
  of null sets, while n = 5 failed both in 100%, and n = 100 passed R-b in 100% (hw/δ median 0.24–0.36) and R-c in
  85–100%.
- `unblind` repeats R-b and R-c at the actual valid pair count, on the same control data, and stamps both.

The same checks run for δ_c and δ_e (§7.4). `freeze` reads no row of a ONE pole and no Arm B target line; it asserts so
by filtering pole ids from `pairs_<M>.json` roles before opening any per-pole file.

### 7.3 Labels (amended A1, A5)

Per model, own-context, from M, [lo, hi] and δ:
- H_cause: lo > 0 and M ≥ δ.
- POSITIVE, BELOW δ: lo > 0 and M < δ.
- H_readout: hi < 0 and M ≤ −δ.
- NEGATIVE, BELOW δ: hi < 0 and M > −δ.
- H_morcos: lo ≤ 0 ≤ hi and −δ < lo and hi < δ, with R-b passed.
- INCONCLUSIVE: lo ≤ 0 ≤ hi otherwise (the interval reaches ±δ, or R-b failed).
Refutations: H_cause is refuted only when hi ≤ 0; H_readout only when lo ≥ 0; H_morcos when the interval excludes 0.

Collateral, from M_c, [lo_c, hi_c], δ_c: H_spec (hi_c < 0 and M_c ≤ −δ_c); LESS COLLATERAL, BELOW δ (hi_c < 0 and
M_c > −δ_c); REVERSED (lo_c > 0 and M_c ≥ δ_c); MORE COLLATERAL, BELOW δ (lo_c > 0 and M_c < δ_c); EQUIVALENT
(lo_c ≤ 0 ≤ hi_c, −δ_c < lo_c, hi_c < δ_c, R-b passed for δ_c); otherwise INCONCLUSIVE.

Energy agreement (A5): the model's headline is its raw label only if the label is non-directional (H_morcos,
INCONCLUSIVE) or the energy-normalized readout (§7.4) has an interval on the same side of 0. Otherwise the headline is
"direction not robust to energy normalization", with both labels printed. If G-bal fails, the energy-normalized label
is the headline and the raw label is printed beside it.

Study label (own-context), over models whose gates pass: H_cause if at least two models give H_cause and no model
gives a negative label (H_readout or NEGATIVE, BELOW δ); POSITIVE, BELOW δ if at least two models give a positive
label (H_cause or POSITIVE, BELOW δ), no model gives a negative one, and fewer than two give H_cause; the mirror rules
for H_readout and NEGATIVE, BELOW δ; H_morcos if at least two models give H_morcos and none gives H_cause or
H_readout; otherwise MIXED. With only two valid models both must agree. Headlines, not raw labels, enter this rule.
The same rule for the collateral label. `verdict_<M>.json` holds: n pairs valid and cancelled (with reasons), M, lo,
hi, δ, δ_c, room stamps and their failing inputs, the raw label, the energy-normalized label, the headline, absolute
medians per side with intervals, clean-NLL medians per side, B (G-bal) and its failing input, and every §7.4 row.

### 7.4 Secondary rows (frozen, reported, never decide the headline unless stated)

- Energy-normalized: `e(p) = own_dnll(p) / mean removed norm`; M_e over pairs, δ_e from null pairs, labels by §7.3.
  Decides the headline only through the energy-agreement rule of §7.3.
- Class-stable pairs (A3): M, interval and label on the pairs whose ONE and control both keep their class under
  θ_s − 0.1 and θ_s + 0.1 (`pairs_<M>.json` → `pairs[].class_stable`; 45 / 55 / 47 pairs), δ unchanged.
- Density covariate (A5): per pair `log_density_ratio` (in `pairs_<M>.json`); Spearman correlation of D_k with it,
  and M on the pairs with |log density ratio| ≤ log 1.25.
- Predictable-target row (A6): own_dnll recomputed per pole over its held-out contexts whose clean p(target) =
  exp(−clean_nll) is below 0.9, on both sides of every pair (a pole left with fewer than 20 such contexts drops its
  pair from this row; counts reported); M, interval and label by §7.3 with the main δ.
- By explainer model of the ONE pole (Opus, Sonnet), by depth tercile, by pole sign: M and interval per stratum
  (B3 reads the explainer row: same sign in both strata, else B3 is flagged).
- ONE poles' own_dnll against coll_dnll ratio (selectivity of necessity), per side.
- Share of held-out contexts at G4 window position 1 for poles of layers 10–11 (F42; held-out only, A13).

### 7.5 Multiplicity

The decisive statistics are the three per-model own-context labels, combined by the frozen study rule of §7.3; no
p-values are used for labels. Collateral and every §7.4 row are secondary and labelled so. Arm B has seven per-pole
verdicts, reported as case studies, each with its own interval; no family-wise claim is made from them.

## 8. Threats to validity

| threat | effect | mitigation |
|---|---|---|
| T1 ONE poles fire more strongly than UNK poles | ONE looks more necessary by magnitude | matching on ho_med and meanabs; realized removed norm banked; G-bal; energy-normalized row |
| T2 ONE poles fire on predictable boilerplate (low clean NLL) | a near-certain target can lose little probability in absolute terms | clean NLL medians per side reported; the predictable-target row (A6) recomputes M on contexts with clean p(target) < 0.9 on both sides |
| T3 explainer model split by depth (AUDIT §3) | ONE vs UNK partly Opus vs Sonnet | explainer stratum row (B3) |
| T4 control pool thin per layer (2–6 UNK) | imperfect matches | caliper log 1.5 (Arm A); d recorded; null pairs built by the same rule carry the same mismatch into δ |
| T5 shallow layers under-represented (Q4B, G4) | result speaks mainly to middle and deep layers | depth row; stated scope |
| T6 held-out contexts from documents the explainer partly saw | "held-out" weaker than intended | rendered-window exclusion; ranks used are recorded |
| T7 zero ablation is off-distribution for poles with a large mean | effect includes an offset shift | zero ablation is the brief's instrument; Morcos 2018 found mean clamping more damaging; no claim beyond zero ablation |
| T8 bf16 batch-shape differences between harvest and job | a_ours ≠ a_banked | G-fit |
| T9 collateral sample is corpus text that may contain a pole's own contexts | collateral partly own-context | counted before freeze from banked arrays (`plan/t9_overlap.py` → `plan/t9_collateral_overlap.json`, A13): poles with any top-1000 position in the 16 windows Q4B 49 / 212, G4 58 / 217, O7 85 / 256; maximum 6 / 44 / 37 positions out of ≈ 32,760 collateral targets |
| T10 Arm B prompts written by one writer | prompt style shared across poles | the matched control sees the same prompts; per-prompt pairing |
| T11 sampling RNG under ablation | divergence from RNG, not logits | identical seeds across conditions; clean-vs-identity sampling equality checked on prompt 0 |

## 9. Runbook (an agent executes this without asking)

Paths: `S` = `<local>/interp_research-neuronoff` (spec and launch worktree), `C` =
`<local>/interp_research-neuronoff-code`. Bucket `B` = `s3://<s3-bucket>`. Every AWS
call goes through `./infra/aws_pg/pg` (profile <aws-profile> and region us-east-1 are fixed in `infra/aws_pg/config.py`).
Never run a model on the laptop. Never poll `job.log`.

### 9.1 Code delta owed before freeze (second code pass, in `C` on `neurons/off-code`; one list, amended A12)

The code was built to SCHEMA v2. Before F1 it must add:
1. `--max-arm-b K` and the Arm A iteration rule (SCHEMA §2, v3).
2. G-known's failing input = layer-0 whole-MLP ablation on the same 120,000 targets (SCHEMA §3.1, v3).
3. The memory rule (SCHEMA §2, v3).
4. `pole_a_inputs` on the clean Arm B teacher line (SCHEMA §3.4).
5. Identity hook on the pole's real firing mask, through the same scatter-and-cast path, writing the original values
   back; never an empty mask (A7; SCHEMA §6, v4).
6. G-fit failing input: a_banked of neuron i against a_ours of neuron i+1 at the same positions, from the same
   forward; banked as `reproduction.failing_input` with its median relative error and sign-mismatch count (A2).
7. G-det: deterministic kernels (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set
   before CUDA initialises), ceiling 0, failing input = token 100 of window 1 replaced by the next token id (A7).
8. Tokenization: `add_special_tokens=False` on every tokenizer call, BOS prepended by the job when defined (A9); the
   prefix-property check and R2's whitespace rule as §5.3.
9. Arm B null rule: m_B from `|mean(D0)|` bootstrap without sign flips, `m_B* = max(m_B, 0.02)`, the ceiling
   mean Δ_pole ≤ mean Δ_layer (A8).
10. Plan v2 fields: `poles[].match_vars` (density, meanabs, ho_med, rate) and `pairs` v2 fields (`log_density_ratio`,
    `class_stable`, `explainer_one`, `explainer_ctrl`, `estimand`); the job may ignore `match_vars` but must accept
    the v2 schema strings (A5, A3, A4).
11. The analysis subcommands of §9.6 in `infra/pg/neuron_off_core.py`, with one focused test each: `freeze` refuses
    to load a ONE row; `unblind` refuses without a committed thresholds file and a prior PREDICTIONS heading;
    `armb_counts` reproduces a hand-counted fixture; the §7.3 label function on fixed fixtures (one case per label,
    including "directional, below δ" and an interval inside ±δ that excludes 0); the room checks fail at n = 5.

The job consumes `plan/plan_<M>.json` from this study; `neuron_off_core.match_controls` is not the selection
authority and is not used to build any plan.

### 9.2 Offline gate and freeze (laptop, no model weights)

F1. In `C`: `python -m pytest -q tests/test_neuron_off_core.py` passes; for each arch in qwen3, gemma3, olmo3:
`python -m infra.aws_pg.jobs.neuron_off_dryrun --arch <arch> --out /tmp/neuronoff_dry/<arch>` runs green and its
outputs show: G-ident 0, G-floor 0 with its failing input > 0, G-fit pass and its shifted-rank input fail, G-known
pass with the dry-run tolerance and its failing input fail, `metrics.json` rewritten after every pole, every SCHEMA v3
key present, `samples/*.jsonl` and `arm_b/*` written. Also in F1: the prefix-property check of §5.3 run with
`transformers.AutoTokenizer` on each local `tok/<qwen3|gemma3>/tokenizer/` folder, `add_special_tokens=False`
(tokenizer files are allowed on the laptop; weights are not); expected 139 of 140 with Q4B L04N5165+ prompt 7
failing. Record the outputs' paths and the pass lines in `JOURNAL.md`.
F2. In `S`: `python3 -I studies/neuronoff/plan/make_plan.py --out /tmp/neuronoff_plan_check` and `cmp` each
`plan_<M>.json` and `pairs_<M>.json` with the committed ones (identical expected; a difference stops the run and is
reported as a spec defect).
F3. Merge: in `S`, `git merge --no-ff neurons/off-code` (disjoint files; no conflict expected). `ln -s
<local>/interp_research/.pgvenv S/.pgvenv` if missing. `git status --short` must be empty.
F4. Write `studies/neuronoff/JOURNAL.md` with a heading `## <UTC> — PREDICTIONS` that copies §5.6 and §11's predicted
column verbatim; commit it by name; then `git tag neuronoff-freeze-1` and record tag, SHA and UTC in §13 by a dated
line (commit by name).
F5. Upload plans: `aws s3 cp studies/neuronoff/plan/plan_<M>.json $B/results/neurons/off/plans/1/plan_<M>.json
--profile <aws-profile> --region us-east-1` for M in Q4B, G4, O7.

### 9.3 Blind protocol

The plan the job reads carries no class and no pair role. During the run the executor may open `metrics.json`,
`controls.json` and the sample files of the first Arm A pair and the first Arm B pole only, for the sample check
(§9.4 L3), and logs which ones were opened. `pairs_<M>.json` is not joined to per-pole results before
`thresholds_<M>.json` is committed. Arm B: `freeze` computes m_B and m_B* from ctrl1 and ctrl2 lines first; target lines are
read only by `unblind`.

### 9.4 Launch sequence

L1. Preflight (known number first, principle 12):
```
cd S && ./infra/aws_pg/pg run infra.aws_pg.jobs.neuron_off \
  --args " --model-key Q4B --repo Qwen/Qwen3-4B-Base --corpus $B/results/neurons/w1/q4b/1/out/corpus/corpus --tokname qwen3 --plan $B/results/neurons/off/plans/1/plan_Q4B.json --max-poles 3 --max-arm-b 1" \
  --s3-prefix results/neurons/off/pre_q4b/1 --type g6e.2xlarge,g6e.4xlarge --budget-min 60 \
  --weights models--Qwen--Qwen3-4B-Base --detach --json
```
L2. Watch with `./infra/aws_pg/pg watch results/neurons/off/pre_q4b/1`; read `$B/results/neurons/off/pre_q4b/1/out/metrics.json`
and `controls.json` only. Pass: `status: complete`, every gate true, every registered failing input failed.
L3. Sample check (principle 14), written to `JOURNAL.md` before any number: open `out/samples/own_ablation.jsonl`,
`positive_control.jsonl`, `collateral.jsonl`, and `out/arm_b/L11N5238p_teacher.jsonl` and `_samples.jsonl`; answer: is
the input the spec's (window prefix ends at the held-out token; prompt + continuation as written); is the output in
format; is the scored quantity the model's own next-token NLL; do clean and identity samples match on prompt 0.
L4. Throughput: `R = progress.tokens_per_second` from the preflight. Projected Arm A forward hours = Q4B
44,569,571/R + G4 46,304,362/R + O7 56,546,074/(0.56·R) (0.56 = the assumed O7/4B rate ratio of §12, about the
parameter ratio 4.0B/7.3B).
Projected total = that + 0.4 (Arm B) + 0.75 (three cells' fixed time). If ≤ 11 h: launch all three cells (L5).
If > 11 h: drop O7 (the brief's two-model fallback) and recompute; if still > 11 h, keep only the first 70 pairs of
each remaining model (`--max-poles` = the 1-based index in `plan.order`, Arm A poles only, of the last pole of
pair 70 and its null partner, computed by `python3 -I -c` over the plan; write the number in the journal).
L5. Main cells, launched together (quota allows four g6e.2xlarge), each with `--budget-min` =
ceil(60 × 1.3 × (tokens_M/R_M + ArmB_M + 0.25)):
```
./infra/aws_pg/pg run infra.aws_pg.jobs.neuron_off --args " --model-key Q4B --repo Qwen/Qwen3-4B-Base --corpus $B/results/neurons/w1/q4b/1/out/corpus/corpus --tokname qwen3 --plan $B/results/neurons/off/plans/1/plan_Q4B.json" \
  --s3-prefix results/neurons/off/q4b/1 --type g6e.2xlarge,g6e.4xlarge --budget-min <Q4B> --weights models--Qwen--Qwen3-4B-Base --detach --json
./infra/aws_pg/pg run infra.aws_pg.jobs.neuron_off --args " --model-key G4 --repo google/gemma-3-4b-pt --corpus $B/results/neurons/w1/q4b/1/out/corpus/corpus --tokname gemma3 --plan $B/results/neurons/off/plans/1/plan_G4.json" \
  --s3-prefix results/neurons/off/g4/1 --type g6e.2xlarge,g6e.4xlarge --budget-min <G4> --weights models--google--gemma-3-4b-pt --detach --json
./infra/aws_pg/pg run infra.aws_pg.jobs.neuron_off --args " --model-key O7 --repo allenai/Olmo-3-1025-7B --revision stage1-step1413814 --corpus $B/results/neurons/w1/q4b/1/out/corpus/corpus --tokname olmo3 --plan $B/results/neurons/off/plans/1/plan_O7.json --skip-arm-b" \
  --s3-prefix results/neurons/off/o7/1 --type g6e.2xlarge,g6e.4xlarge --budget-min <O7> --no-weights --detach --json
```
At R = 8,000 tokens/s the budgets are Q4B 156, G4 157, O7 292 minutes (Arm B 0.2 h for Q4B, 0.15 h for G4). O7 (14.6 GB bf16) fits one L40S 48 GB, as the
parent's W1 O7 harvest did on g6e.2xlarge.
L6. Liveness: `./infra/aws_pg/pg ps` and the `updated_utc` of each cell's `out/metrics.json`; status lines report
only from these. The first line of every status report is the run/unrun/parked count over the roster (pre_q4b, q4b,
g4, o7). Never score a running cell.
L7. After `STATUS exit=0`: `aws s3 sync $B/results/neurons/off/<cell>/<attempt>/ S/bank/off/<cell>/<attempt>/
--profile <aws-profile> --region us-east-1`; record the artifact timestamps.

### 9.5 Pivots (pre-specified)

- P1 G-known fails: re-run the cell once at a fresh attempt on `--type g6e.2xlarge` only. A second failure: that
  model is out of the study, reported with both values; the other cells proceed.
- P2 capacity: the launcher tries the next type and zone; no waiting. If no capacity for 2 h, stop and hand off.
- P3 budget kill mid-cell: the finished poles stand (saved per pole); re-launch at a fresh attempt with the plan's
  remaining poles only if the GPU-hour cap allows; else analyse the completed pairs (G-n decides).
- P4 G-det or G-floor fails: the model's labels are cancelled; no re-run without a code fix and a dated amendment.
- P5 more than 30% of a model's poles cancelled by any one gate: report the gate as the finding for that model.
- P6 deterministic mode raises (an op without a deterministic implementation under
  `torch.use_deterministic_algorithms(True)`): the preflight fails loud; the fix is a code change that selects a
  deterministic kernel for that op, recorded as a dated amendment, then a fresh preflight attempt. `warn_only=True`
  and turning deterministic mode off are not allowed (A13).

### 9.6 Analysis (laptop; `python -m infra.pg.neuron_off_core <subcommand>`; rules exactly as below)

- `freeze --model M --bank S/bank/off/<cell>/<attempt>/out --pairs studies/neuronoff/plan/pairs_M.json --out studies/neuronoff/results/thresholds_M.json`:
  loads per-pole files only for `ctrl` and `extra` poles of `null_pairs` and the `ctrl1`/`ctrl2` teacher lines of
  `arm_b`; raises if any other pole id is requested. Applies the per-pole gates (G-ident, G-fit, G-pc) to null-pair
  poles; computes δ, δ_c, δ_e, R-a, R-b, R-c and their n = 5 and n = 10 failing inputs (§7.2), and per Arm B pole m_B,
  m_B* and their control-only inputs (§5.5, no sign flips); writes
  the file with input sha256s. The executor commits it by name before `unblind`.
- `unblind --model M --bank … --pairs … --thresholds studies/neuronoff/results/thresholds_M.json --journal studies/neuronoff/JOURNAL.md --out studies/neuronoff/results/verdict_M.json`:
  refuses unless the thresholds file is committed in git with no working-tree change and the journal's PREDICTIONS
  heading was committed before it (`git log` times); computes §7.1, §7.3, §7.4 and the Arm B verdicts (§5.5 with the
  room rule), writes `verdict_M.json` and appends the model's rows to `results/grid.md`. Refuses an existing output.
- `armb_counts --bank … --plan studies/neuronoff/plan/plan_M.json --out studies/neuronoff/results/armb_counts_M.json`:
  applies §5.4 to every completion; per pole, condition and prompt: presence count out of 20 and mean matches;
  the behaviour-changed rule of §5.4 with a 95% bootstrap over prompts (10,000, seed 20261025).
- Study label: `unblind --study` reads every `verdict_M.json` and writes the §7.3 study rows into `grid.md`.

### 9.7 Sessions and handoff

S1: §9.1–§9.2 (if not already done), L1–L5. Ends after the main launches with `studies/neuronoff/handoff/S1.md`
(record state NEW/REUSED, open instance ids, prefixes, budgets, exact next action). S2: L6–L7, §9.6, the result
document after the checker. A different model reads the artifacts (CLAUDE.md Provenance §3) before any result is
written: one Opus checker in its own worktree, brief = this spec, the bank paths and `results/`.

## 10. Artifacts

Per cell under `S/bank/off/<cell>/<attempt>/out/` (SCHEMA §3): `metrics.json`, `controls.json`, `manifest.json`,
`poles/<stem>.json`, `arm_b/<stem>_teacher.jsonl`, `arm_b/<stem>_samples.jsonl`, `samples/own_ablation.jsonl`,
`samples/positive_control.jsonl`, `samples/collateral.jsonl`, plus the launcher's `job.log`, `heartbeat.jsonl`,
`STATUS`. Study files: `results/thresholds_<M>.json`, `results/verdict_<M>.json`, `results/armb_counts_<M>.json`,
`results/grid.md`, `results/REGISTRY.md` (every number → file, key, input sha256). Figures (after the checker): F1
per-model paired difference with δ band and the null-pair distribution drawn on the panel; F2 absolute own_dnll per
side; F3 collateral; F4 Arm B per-pole Δ with ctrl1, ctrl2 and m_B* drawn; F5 Arm B construct presence by condition.

## 11. Prediction grid (predicted column frozen at F4; observed filled once by `unblind`)

| readout | predicted (2026-10-25) | probability | observed | label |
|---|---|---|---|---|
| own-context, study label (amended A1 label set) | H_cause | 0.35 (POSITIVE, BELOW δ 0.2; H_morcos 0.25; H_readout or NEGATIVE, BELOW δ 0.1; MIXED 0.1) | ____ | ____ |
| own-context M, Q4B / G4 / O7 | > 0, absolute own_dnll on both sides below 0.05 nats median | — | ____ | ____ |
| collateral, study label | EQUIVALENT | 0.45 (H_spec or LESS, BELOW δ 0.25; REVERSED or MORE, BELOW δ 0.2; MIXED 0.1) | ____ | ____ |
| class-stable row (A3) same direction as the headline | yes | 0.7 | ____ | ____ |
| predictable-target row (A6) same direction as the headline | yes | 0.75 | ____ | ____ |
| energy-normalized direction agrees with raw (A5) | yes in every model | 0.75 | ____ | ____ |
| explainer strata same sign (B3) | yes | 0.7 | ____ | ____ |
| G-pc pass rate | ≥ 90% of poles per model | 0.7 | ____ | ____ |
| Arm B SPECIAL count | ≥ 3 of 7 (per pole: §5.6) | 0.6 | ____ | ____ |
| Arm B behaviour changed | 0 of 7 | 0.8 | ____ | ____ |

## 12. Compute budget

Forward tokens (from `plan/budget.json`, exact window prefixes): Arm A per model = two passes over the held-out
windows + positive control + identity + collateral = Q4B 44,569,571; G4 46,304,362; O7 56,546,074 (regenerated after
A5; Q4B 212, G4 217, O7 256 Arm A poles).
Assumed throughput (no plain-forward measurement exists in the repo; the harvest's 3,922 / 3,566 / 3,335 tokens/s
included a top-k over every neuron per layer, so a plain forward is expected faster): 8,000 tokens/s for the 4B models,
4,500 for O7. The preflight replaces the assumption (L4).

| item | arithmetic | GPU-h |
|---|---|---|
| Arm A Q4B | 44.57M / 8,000 / 3,600 | 1.55 |
| Arm A G4 | 46.30M / 8,000 / 3,600 | 1.61 |
| Arm A O7 | 56.55M / 4,500 / 3,600 | 3.49 |
| Arm B, 7 poles | 60 generations of 20 × 128 tokens per pole at ≈ 3.2 s each ≈ 0.05 h, plus teacher forcing | 0.4 |
| fixed per cell (boot, weight sync or 14.6 GB download, load, gates, collateral clean, known number ×2) | 3 × 0.25 | 0.75 |
| preflight cell | 0.25 fixed + 3 Arm A poles + 1 Arm B pole | 0.35 |
| **total** | | **8.2** |

At the harvest's own throughput (3,922 / 3,566 / 3,335 tokens/s) the same plan costs 3.16 + 3.61 + 4.71 + 1.5 = 13.0
GPU-h, over the 12 GPU-h cap; L4's frozen ladder then drops O7 (3.16 + 3.61 + 0.4 + 0.5 + 0.35 = 8.0 GPU-h) before
it reduces N. The remaining ≈ 3.8 GPU-h under the cap at the
assumed throughput covers one re-run of one cell (P1, P3). Runs: 4 launches (pre_q4b, q4b, g4, o7).

## 13. Amendments and freeze record

Freeze record: frozen 2026-10-25T18:17:16Z. The tag `neuronoff-freeze-1` is placed on the commit that adds this line (its SHA is read from the tag: `git rev-parse neuronoff-freeze-1`). PREDICTIONS were committed before it at 4b612779 (`studies/neuronoff/JOURNAL.md`); the code is `neurons/off-code` b744a6a3, merged at 175f6c49; offline gate: 64 tests passed, three dry runs green, tokenizer check 139 of 140 as expected. No GPU run of this study exists before this line.

**Amendments of 2026-10-25, before freeze, from an adversarial review accepted by the director.** Each is carried in
the body; this list says what changed and where.

- **A1** (blocker; §1 table, §7.2, §7.3, §11). δ is a spread (median |null difference|, ≈ 0.67 σ of the null) used as
  a margin on a median whose 95% half-width at n = 100 is ≈ 0.24–0.36·δ; the old rules labelled a true 0.3σ shift
  H_morcos most of the time and let an interval excluding 0 in H_cause's favour count as refuting it. Now: H_morcos
  requires lo ≤ 0 ≤ hi; H_cause is refuted only when hi ≤ 0 (H_readout only when lo ≥ 0); an interval inside ±δ that
  excludes 0 is "POSITIVE / NEGATIVE, BELOW δ"; the study rule counts these. Room checks R-b and R-c get a registered
  failing input: n = 5 must fail both (n = 10 also reported; see §7.2 for why n = 10 is not the hard input).
- **A2** (blocker; §6 G-fit). The shifted-rank failing input passed on 727 of 727 poles (adjacent ranks differ by
  0.13–0.15%). Replaced by a_banked of neuron i against a_ours of neuron i+1 at the same positions; expected median
  relative error above 0.5.
- **A3** (§1, §7.4). The contrast is named "explanation passed §6.1" against "explanation failed §6.1 or declined";
  85% / 85% / 79% of kept controls carried a one-job label from the explainer. New frozen row on class-stable pairs
  (45 / 55 / 47 after A5's re-matching; the reviewer's 49 / 57 / 47 were for the earlier pairs).
- **A4** (§4.2). Estimand stated: ONE poles inside UNK's energy and layer range. Unmatched ONE poles after A5: 258 of
  354 (Q4B), 238 of 335 (G4), 159 of 311 (O7) (the reviewer's 243 / 210 / 148 were for the earlier rate-based
  matching); kept-against-eligible shifts of density, meanabs and ho_med reported.
- **A5** (§4.2, §7.3, §7.4; `make_plan.py`). Matching on the parent's `density` instead of the sign share `rate`, plus
  meanabs and ho_med; pairs, plans and budget regenerated (Q4B 96 pairs / 33 null pairs, G4 97 / 38, O7 100 / 67;
  forward tokens 44.57M / 46.30M / 56.55M). A directional headline requires the energy-normalized readout on the same
  side of 0; per-pair log density ratio is a covariate row.
- **A6** (§7.4, §8 T2). Frozen row: M on held-out contexts with clean p(target) < 0.9 on both sides.
- **A7** (§2, §6, SCHEMA §6 v4). Identity hook writes the original values on the pole's real firing mask (the
  scatter-and-cast path runs); G-det ceiling 0 with deterministic kernels; registered failing inputs for G-det (one
  token changed), G-n (59 pairs / 19 null pairs) and G-bal (|B| pushed out by log 1.6).
- **A8** (§5.5, §5.6). Arm B null: bootstrap of |mean D0| without sign flips; frozen floor m_B* = max(m_B, 0.02 nats);
  ceiling mean Δ_pole ≤ mean Δ_layer, verdict cancelled above it; useEffect prediction is NOT-SPECIAL.
- **A9** (§5.3, §9.2 F1). `add_special_tokens=False` with BOS prepended by the job; R2 includes leading whitespace-only
  tokens up to the first non-whitespace token (17 of 140 continuations); prefix property checked against the local
  tokenizer files: 139 of 140 hold it (Q4B L04N5165+ prompt 7 excluded); F1 repeats it with the job's tokenizer.
- **A10** (`LIT.md`). Wang et al. 2022 (skill neurons), Nandan et al. 2025 (universal neurons) and Gong et al. 2025
  read and cited; "no LM paper runs this contrast" softened: Wang et al. ran the selectivity-to-importance question in
  RoBERTa; what survives is the explanation-tested class, single-pole zero ablation and count- and energy-matched
  controls.
- **A11** (§4.2). Controls of dropped pairs return to the pool, now stated.
- **A12** (§9.1). One list of code deltas for the second code pass.
- Arm B controls changed with A5's distance (§5.2); five targets now have a control with d > log 2, flagged.

**Amendment A13 (2026-10-25, before freeze): rules the code fixed where this spec was silent, ratified.** Source:
module docstrings of `infra/pg/neuron_off_core.py`, `infra/aws_pg/jobs/neuron_off.py` (job rules A1–A8) and
`infra/pg/neuron_off_analysis.py` (analysis rules B1–B13) at `neurons/off-code` b744a6a3. Every one is ratified as
written there; none is changed, so the code needs no further pass. The six the coordinator named, stated exactly:
1. R-c inner bootstrap: 2,000 outer draws of size n from `s_j·d_j + 1.5·δ`; in each draw M is the median and lo the
   2.5th percentile of 1,000 inner bootstrap medians; the rule fires when lo > 0 and M ≥ δ. The same for δ_c and δ_e.
   Random signs `s_j = default_rng(20261025).choice([-1, 1], size=len(d))`; every bootstrap uses its own fresh
   `default_rng(20261025)`; intervals are numpy-linear 2.5th and 97.5th percentiles (analysis B1, B2).
2. Room at a subset: for the class-stable and predictable-target rows, R-b and R-c are recomputed at that subset's
   valid pair count, with the main δ (B3).
3. Density-covariate row: Spearman correlation of D_k with the log density ratio, and M with its 95% interval on the
   pairs with |log density ratio| ≤ log 1.25; no label is assigned to this row (B13).
4. Collateral study rule: the own-context study rule of §7.3 applied to the collateral labels with the mapping
   lo_c > 0 and M_c ≥ δ_c → REVERSED; lo_c > 0 and M_c < δ_c → MORE COLLATERAL, BELOW δ; hi_c < 0 and M_c ≤ −δ_c →
   H_spec; hi_c < 0 and M_c > −δ_c → LESS COLLATERAL, BELOW δ; covering 0 inside ±δ_c with room → EQUIVALENT. A model
   counts when its label is not cancelled; with fewer than two counting models the study label is MIXED (B11).
5. F42 and T9: the analysis reports F42 on held-out contexts only. The collateral share at window position 1 is
   dropped: every collateral window contributes exactly one position-1 target, so the share is a constant (1/2047 or
   1/2048) and carries no information. T9 is computed spec-side before freeze (`plan/t9_overlap.py`, §8), not by the
   analysis.
6. Deterministic-mode CUDA risk: the job sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA starts and
   `torch.use_deterministic_algorithms(True)`; an op without a deterministic implementation raises at the preflight
   (L1), which is where it should surface; pivot P6 (§9.5).
Also ratified, with no change: a per-pole registered failing input that did not fail is counted and reported but
does not cancel the pole (B4); `freeze` reads Arm B lines of conditions clean, identity, ablate_ctrl1, ablate_ctrl2
and ablate_layer only, and `unblind` applies the Arm B ceiling (B5); Arm B records without the prefix property are
scored and flagged by the job and excluded by the analysis (job A7, B6); prompt 0 is also sampled under the identity
hook (job A7); the PREDICTIONS-before-thresholds check uses the earliest commit adding each (B9); the analysis
commands are `python -m infra.pg.neuron_off_analysis …`, also reachable as `python -m infra.pg.neuron_off_core …`
(§9.6), plus `tokcheck` for F1.

## 14. Changes from the director's brief, and why

1. Base branch `neurons/harvest`, not `neurons/explain`: the latter lacks `infra/pg/neurons_harvest.py`; banks are
   untracked and read by path from both worktrees.
2. Control class UNK, not every non-ONE class: TWO-REL and UNREL passed the legibility test.
3. Matching variables are the parent's density (count), mean |a| and the held-out median |a| (energy at the readout
   positions), not firing rate and mean |a| alone: in G4 the ONE poles' median |a| over ranks 31–130 is 1.8× the UNK
   poles' (4.19 against 2.30), which would confound own-context ΔNLL with magnitude; the sign share ("firing rate"
   under this ablation) never differs by 1.5× and matched nothing (A5); and the archived program's rule requires
   count- and energy-matched nulls.
4. "Ranks 31–130" became "the first 100 contexts from rank 31 on that lie outside the explainer's rendered windows".
5. Mask domain excludes BOS and window/text position 0, matching the harvest's scored domain; activity there is
   banked as a diagnostic.
6. Collateral sample: 16 windows (≈ 32.8k targets) instead of ≈ 100k, to fit three models under 12 GPU-h; the clean
   known number still uses the harvest's full 120,000 targets.
7. Arm B: prompts carry a `continuation` and the primary readout is the summed log-probability of the whole construct
   (R1). The first-token target alone (R2) would be scored at the position before the construct, where these poles
   do not fire (each fires on a token inside the construct), so it would be close to null by construction. R2 is
   still reported.
8. Arm B controls: two per target (ctrl2 freezes the margin), nearest in layer without a caliper, because four
   targets have no UNK pole within log 1.5 in their layer.
9. The meaningful-difference margin δ is the median absolute null-pair difference (a per-pair spread), not the
   standard error of the null median; with the latter the equivalence label could never fire (room check R-b).
10. Antverg & Belinkov's title in the brief is wrong; the paper is "On the Pitfalls of Analyzing Individual Neurons in
    Language Models" (ICLR 2022). LIT.md uses the correct one.
11. G-known's registered failing input is a layer-0 whole-MLP ablation, not a window shift (coordinator, 2026-10-25).

## 15. References

Read at source for the claims used; see `LIT.md` for what was read and where: Morcos et al. 2018; Zhou et al. 2018;
Leavitt & Morcos 2020; Gurnee et al. 2023; Bills et al. 2023; Bricken et al. 2023; Voita et al. 2023; Geva et al.
2022; Antverg & Belinkov 2022; Dalvi et al. 2019; Huang et al. 2023; Dai et al. 2022; Kargaran et al. 2025.
