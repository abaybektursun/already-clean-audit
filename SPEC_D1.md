# Direction 1 — does "already clean" hold, and did it grow that way? An independent audit on two 4B models and a developmental test of recursive generalization on OLMo-3-7B's public checkpoints

## 0. Status and provenance

- Status: **FROZEN 2026-10-20T16:50:32Z**, tag `already-clean-freeze-2026-10-20` at dc18799a on `drafts/already-clean`
  (§6 rules, prompt v3, θ_s = 0.3153), set before any confirmation number was read. The W1-C audit harvests (GPU only, no
  explanation and no number read) ran before the tag, on 2026-10-20 03:15–07:35Z; the first confirmation explain
  directory started at 16:50:36Z, after the tag. Dated amendments after the freeze: §2.2 model policy (2026-10-20 17:15Z),
  §6.6a declined answers at W4 (2026-10-21), the clusterer prompt v2 on S_gate (2026-10-21 17:45Z, allowed by §4.4), and
  the checker's corrections (2026-10-21, JOURNAL 18:00Z). Everything in §6 was frozen on the discovery sets named in §4.4.
- Parents: `DIRECTIONS.md` D1; `IDEAS.md` I0 (instrument), I10 (developmental test), I11 (audit);
  `THEORIES.md` P1, P6, P8, P12; `CHECKS.md` batches 1–4 (literature; queries recorded there). Essay cited
  as `essay.md` by line (plain text of https://already-clean.goodfire.com/essay.md, fetched 2026-10-19).
- Amendments 2026-10-19 (draft, before any code): `DATA_CONTRACT.md` fixes the file layout and amends
  §2.1 (`sparse/` replaced by `exemplar_a` at fixed positions and a top-1000 record; `pass_through.json` and
  `nll.json` merged into `controls.json`; 10M/20M readings taken as band snapshots inside one harvest) and §4.2
  (documents ordered in three bands so 10M ⊂ 20M ⊂ 50M are prefixes of one token stream). Where the two
  documents differ on file layout, `DATA_CONTRACT.md` wins; the measurement rules of §6 are unchanged.
- Spec defects this document must not repeat (`studies/stagediff/CLOSEOUT.md` §4): thresholds without a
  floor, gates with no room, a readout blind by construction, one-sided rules.
- Facts about the substrate below were read on 2026-10-19 from the Hugging Face API and the repos' own
  `config.json`, `README.md` and file trees (saved under the session scratchpad `olmo/`). Nothing is from
  memory. Facts about our infrastructure were read from `.claude/skills/aws-pg/SKILL.md`,
  `infra/aws_pg/jobs/pipeline.py`, `infra/aws_pg/bake_env.sh`, `infra/pg/diff_harvest.py`.
- Amendments 2026-10-19 (evening, after W0): N1–N5 are built and ran end to end once at toy scale
  (`verify/VERIFY.md`, `verify/HARVEST.md`, `verify/EXPLAIN.md`; branches `neurons/harvest` b0ec4364 and
  `neurons/explain` 4e38734e). Every amendment below marked **W0** comes from that run: §0b (stance), §2.1
  (exemplar record, context width), §2.2 (explainer motifs), §2.3 (UNK split), §4.1 (G1 model), §4.2 (corpus
  rule, duplicates, window position 0), §4.4 (sequencing, `S` = 600, the gate set), §5 (G0 power, G1), §6.3
  (conditional bootstrap), §7 (power), §9 (runbook), §12 (measured budget), §14, §15, §16.

## 0b. Stance: the essay as true north

Written 2026-10-19 after re-reading the essay in full, so that this study's defaults are the essay's and
do not drift back toward the field's. These are operating rules, checked at every write-up.

1. **Superposition is the last hypothesis, not the null** (essay.md 201, 209). When a pole resists
   explanation the suspects are tried in this order: our tools (context too short, corpus too small, weak
   activations read as jobs, the double-negative echo, a tokenizer or pull defect); our explanation (a list
   of nouns where the job was a shared pattern or a distinction, essay.md 97–99); the model's learning
   (size, architecture); recursive generalization not yet followed over training; and only then packing.
   UNK is a claim about the observer (essay.md 169) and is never read as evidence for packing.
2. **The asymmetry** (essay.md 521): assume a pole is clean and be wrong, and a messy pole is a lead the next
   experiment sharpens; assume it is packed and be wrong, and you stop. So the study looks harder before it
   concludes, and says plainly when it cannot. A failed explanation sends us back to the microscope first.
3. **A neuron is a verb**: a distinction with two poles, the negative pole a job in its own right and often
   a gate (essay.md 97, 219–229, 349). Poles are the unit. W0 bore this out: both poles of a neuron carried
   distinct readable jobs (one neuron's positive pole "a preparatory or trial stage before the main event",
   its negative pole "the venue someone goes to").
4. **Trust the pole's own range** (essay.md 323–325): strong activations in raw units, density in pages, no
   percentile standardization across neurons. Weak activations are followed only when something specific
   points at them, and then by tracing the circuit, not by relabelling the pole.
5. **Explanations must be specific enough to lose** (essay.md 387–397, 635–645): near misses, the most
   distant concept, the blind maximum. The distant-concept rate is the essay's own test for packing and is
   reported as such; the shuffled pole is our null. A test the explanation cannot fail is not a test.
6. **The cause matters more than the count** (essay.md 29, 783). Q2, whether the later job grew from the
   earlier one, is the decisive readout; Q1 is calibration and the field's stated demand. No cleanliness
   score is optimized or held tightly.
7. **Truthful optimism** (essay.md 779–789): report what is seen, with intervals; do not water a change of
   kind down to "cleaner than we thought", and do not inflate a toy into a readout. No rule moves after
   data; a wrong rule is stated as wrong and both readings are given.
8. **Expect motifs, then look for them** (essay.md 39–45, 187, 305, 487): summary on punctuation, the word
   before the cue, say-or-just-think suppression in the last layer, feeling and social-cue detectors,
   long-context dependence, next-token jobs. The explainer is told them (§2.2) because the map fills in
   faster once they are known, and because without them a monosemantic pole looks polysemantic.

## 0c. Exploration track: cheap moves with large reach

The confirmatory grid (§11) is frozen and small. Around it runs an exploration track in the essay's own
style (essay.md 475–477: start from questions, zoom in and out, follow the noise), where the point is to find
the one cheap observation that settles or overturns a lot. Rules: every move is labelled EXPLORATORY, runs on
the discovery sets or after W5, never changes a §6 rule, and is written up with what it cost and what would
have surprised us. Ranked by reach per unit cost.

| move | the question it answers | the cheapest decisive version | cost | what would surprise us |
|---|---|---|---|---|
| **X1 predict the offshoot** | the essay's strongest claim about recursive generalization is that a branch's next offshoot is predictable from the branch (essay.md 807, 815) | an agent sees a pole's record at `T_12` only and names, in one sentence, what the pole will also fire on by `T_15`; a blind judge scores the prediction against the pole's actual `T_15` late-born clusters and against a shuffled pole's; the hit rate over `S` is the readout | agent calls only, ≈ 2 per pole; no GPU | a hit rate near the shuffled rate: branches grow but not predictably, which weakens the "fractal explanation" claim; a hit rate far above it is the first forward prediction of a neuron's growth at LM scale |
| **X2 steer and guess** | does a lens that never sees the top-k agree with the explainer? the essay's convergence test (essay.md 43, 187) | add `α · W_down[:, i]` at the pole's layer and sample 20 continuations at three `α`; a blind agent writes the explanation from the generations alone; a judge scores agreement with Agent 1's explanation against a shuffled pairing | one short GPU job per model (≈ 0.3 GPU-h for 600 poles); ≈ 3 agent calls per pole | agreement no better than shuffled on poles that pass §6.1: the explanation predicts activations but not function |
| **X3 blind maximum against ten billion tokens** | the essay reports a sentence written from the explanation alone out-fires every corpus example (essay.md 389–391) | we already bank the `max` text per pole; report the share of poles whose blind maximum scores ≥ 1.0 of the corpus max, by class | zero: already in `texts.jsonl` and `scores.npz` | a share near zero on ONE poles: the explanation names the center less well than the essay's did, or 50M tokens already found the center |
| **X4 the five-second question** | how many UNK dissolve when asked the essay's way (essay.md 73, 99: "what one thing might this be, if I could not use nouns?") | every UNK-F pole on the discovery set is re-read by a fresh agent that sees both poles' top contexts and the other pole's explanation, and must answer in one sentence without a list of nouns; the new sentence goes through the same writer and probe | ≈ 3 agent calls per UNK pole; the discovery rescoring job already planned | most UNK stay UNK: the remainder is in the pole, not in our reading; most dissolve: the first-pass prompt was the problem, and the second pass becomes prompt v2 |
| **X5 where the step-1000 jobs went** | W0 saw step-1000 top contexts carry ≈ 1% of the final pole's activation; did those early features consolidate elsewhere, dissolve, or stay as weak offshoots? (essay.md 767: consolidation as a Schelling point) | for 20 poles, score the step-1000 top-50 positions at `T_15` against every pole of the same layer; the best-matching pole's explanation says whether the feature moved, merged or vanished | one small GPU pass (positions × one layer's activations); ≈ 1 agent call per pole | the early features reappear as the main jobs of other neurons in the layer: migration, not growth, and H_MIG becomes a population count worth its own readout |
| **X6 the negative pole as the gate** | the essay says the two poles usually draw one distinction (V vs O, taxi vs ambulance, essay.md 97, 229); W0 saw pairs that read as separate jobs | an agent sees only the two poles' explanations and labels the pair contrast, same-axis, unrelated, or echo; the distribution by layer is the readout | ≈ 1 agent call per neuron; zero GPU | most pairs unrelated in middle layers: SwiGLU poles are two jobs, not one distinction, which changes how poles should be counted in the audit |
| **X7 decode the noise** | the "pr" neuron fired on Base64 and Bengali that decoded to "pr" (essay.md 183) | a pre-pass flags contexts in non-Latin scripts, Base64 and URL encodings and appends a decoded or transliterated line to the explainer's view | ≈ 20 lines of code; zero GPU | the explainer's pass rate on flagged poles rises: a share of "polysemantic" was encoding |
| **X8 the learning curve of the microscope** | the essay says the map fills in faster as motifs are known (essay.md 39–45) | on the discovery set only, explainer pass rate as a function of the number of motifs in the prompt (none, the W0 six, the §2.2 v2 list) | the W2-D rescoring job and ≈ 120 calls per prompt version | no gain from motifs: the explainer already knows them, or the remainder is not about motifs |
| **X9 duplicates as a free determinism check** | the W0 corpus held up to 7 copies of one document | identical windows must give identical activations; any difference is a harvest defect, found for free | zero | any mismatch |
| **X10 dead poles** | Q4B has ≈ 5,500 poles in layers 1–3 that never take one sign in 2M tokens; OLMo has none | a weights-only check of whether `g` is one-signed on the corpus for those neurons, which would make early SwiGLU neurons behave as ReLUs | zero GPU (stats already banked) | the dead poles are dead by weight geometry: an architecture fact worth a sentence in the audit |

X1 and X2 are the moves with the largest reach: a correct forward prediction of a neuron's next branch, or
a second lens that agrees with the first without sharing its inputs, would each say more about whether the
network is legible than any fraction in §6.2. X3, X9 and X10 cost nothing and are reported with W1. X4 and
X8 run inside W2-D and feed prompt v2; whatever they change is frozen before any confirmation number is read.

## 1. Question, decision, hypotheses, refuting observations

**The object.** An MLP neuron pole in a gated-MLP transformer. For a neuron `i` in layer `ℓ` with gate
pre-activation `g = W_gate x`, up pre-activation `u = W_up x`, the activation is `a = act(g)·u`
(`act = silu` for Qwen3 and OLMo-3, `gelu_tanh` for Gemma 3). The positive pole is the set of tokens
where `a > 0`, the negative pole where `a < 0`. Each pole is one unit of analysis, as the essay counts
them (essay.md 29).

**The two questions.**
- Q1 (audit, descriptive with controls): on models no one at Goodfire has audited, what fraction of neuron
  poles have one job, two related jobs, unrelated jobs, or are not understood, under explanations that can
  lose? The essay reports roughly 90 / 7 / 2–4 on Qwen3-32B with stated caveats (essay.md 29–31, 369).
- Q2 (developmental, the decisive contrast): when a pole at the end of pretraining has more than one job,
  did the later jobs grow out of the earlier ones? The essay says yes, everywhere it has looked, on
  retrained InceptionV1 and on a few LM neurons, and says the checkpoints to test it on a large LM did not
  exist (essay.md 129–137, 681, 813–817). `allenai/Olmo-3-1025-7B` publishes 1,421 stage-1 checkpoints.

**The decision.** If Q2 lands for recursive generalization, our next five studies (D2–D6) use the
model's own neurons as the unit and the pipeline built here as the instrument. If it lands for packing,
or the instrument cannot tell, D2–D6 keep decomposition methods and the neuron basis is demoted to one
view among several. Q1 alone does not change our next action; it is the field's stated demand and the
calibration step for Q2's instrument, so it costs little beyond Q2.

| hypothesis | claim | the observation that refutes it |
|---|---|---|
| H_RG recursive generalization | every job a pole has at the end of training traces to an earlier job of the same pole: a blind judge given the pole's earlier record identifies the parent against a decoy far above chance. The essay's further expectations for a modern architecture, that the main job is the oldest and offshoots start weak (essay.md 129, 805), are supports, not the claim | the unrelated share `p_u` (§6.3) has lower bound above 0.20. A failed birth-order test (§6.4) is drift, the InceptionV1 pattern of essay.md 131, and is reported as such, not as a refutation |
| H_SP packing (superposition in the essay's sense, essay.md 35, 643) | a substantial fraction of later jobs have no antecedent in the pole: the judge is at chance on them; they appear at full strength; the share grows with training as capacity tightens | `p_u` upper bound below 0.10 |
| H_VEST vestigial mixtures (the question the 2021 paper left open, essay.md 801) | extra jobs are leftovers from early training: born at or before the main job and shrinking over the last half of training | the extra jobs are mostly born after the main job, or mostly growing over the last half |
| H_MIG replacement | the pole's job is replaced wholesale: a new dominant cluster appears while the old one vanishes, with no relation between them | consecutive dominant clusters pass the parent test; this is a sub-case of H_SP under the primary readout and is reported as its own count |
| the boring ones | (B1) the judge separates true from decoy by domain alone; (B2) births are artifacts of corpus size; (B3) early checkpoints' activation scale makes "present" ill-defined; (B4) the clustering agent is unreliable | (B1) the hard-decoy row (§5) stays far above chance; (B2) births are stable between the 10M and 20M token readings; (B3) the percentile and raw-ratio birth rules agree; (B4) two independent clusterings agree on the cluster count within one for most poles |

H_RG and H_SP are not exclusive over a population; the decision rule in §6.3 assigns the label from the
estimated unrelated share and reports mixed outcomes as mixed. The essay itself expects a small
not-understood remainder (essay.md 29), so H_RG is not "zero unrelated jobs"; it is "fewer than one in ten".

**Why the question decides something for the field.** No outsider has reproduced any number in the essay
(CHECKS.md, "already clean" critiques row; coverage of 2026-10-17). The essay names the developmental
observation as the one that makes recursive generalization hard to vary (essay.md 815) and says large-LM
checkpoints were missing (813). OLMo-3 supplies them. A positive result is the first independent,
checkpoint-level evidence for the essay's mechanism on a 2025 open model; a negative result is the first
evidence against it at scale. No neuron-level development study exists on OLMo or Pythia (CHECKS.md I10
rows: 2610.21025 is a crosscoder on the unembedding; 2606.02378 is attention circuits; 2605.13329 is
persona vectors). **(2026-10-23 correction, read at source, `checks/literature_W5.md` §4d:** arXiv 2601.08383 is a neuron-level, time-resolved attribution study over OLMo-7B and OLMoE checkpoints, ranking neurons by importance on relational-fact prompts; the claim that stands is that no study has explained neurons, their jobs, clusters, relatedness or births, across OLMo or Pythia checkpoints.)

## 2. The measurement, exactly

### 2.1 Harvest (GPU)

For one model checkpoint and the corpus `C` (§4.2), run the model forward over non-overlapping windows of
2,048 tokens with a BOS token prepended where the tokenizer defines one, batch of 8 windows, bf16, SDPA
attention, `inference_mode`. A forward hook on every MLP module records `g`, `u` and `a` for every token.
The instrument writes, per checkpoint:

- `topk/`: for every pole of every neuron, the 50 highest `|a|` tokens over `C` with `(doc_id, position,
  a, g, u)`. Raw values, not percentiles (essay.md 323: raw shape is information). The sign pair
  `(sign g, sign u)` of each top-50 entry gives the SwiGLU regime split; the double-negative share of a
  pole is the fraction of its top-50 with `g < 0, u < 0` (essay.md 373, 573–577).
- `stats/`: per pole, `max|a|`, the 99th, 99.9th and 99.99th percentiles of `|a|` over `C`, the count of
  tokens with `|a| ≥ 0.01·max|a|` (the pole's density, "a few times a page" against "a few times a
  book", essay.md 325), and the fraction of the top-50 that are the same token string.
- `sparse/` (only for the sampled poles `S`, §4.4, and only in W3): every `(doc_id, position, a)` with
  `|a| ≥ 0.05·max|a|` at that checkpoint, plus the top 1,000 regardless. This record gives any exemplar
  set's activation at that checkpoint without a second forward.
- `probe/` (W3 and W4): activations of every agent-written text (§2.2) at every pole in `S`, at that
  checkpoint.
- `nll.json`: mean next-token NLL over the first 120k tokens of `C` and over the wikitext-2 windows of
  §5 G1; `pass_through.json`: §5 G0.

A context is the 64 tokens before and 16 after the activating token, the activating token marked, with
the raw `a`, `g`, `u` and the pole's `max|a|` for that checkpoint. **W0 amendment:** the explainer's and
clusterer's rendered contexts are 256 tokens before and 16 after (the essay found that two paragraphs of
context made every long-context neuron look polysemantic, essay.md 191; 64 tokens is less than that). The
harvest stores positions, so this is an explain-side constant; the judge's exclusion window (§2.4) widens
with it. The judge's own records stay at 64 before so that two records fit one call.

**W0 amendment, exemplar record.** The `T_15` harvest cannot take `--positions` (its top-50 is known only
after its own harvest), so with `--poles` and no positions the harvest writes `poles/positions.npz` (its own
top-50 positions) and `poles/exemplar_a.npz` at them; `T_0 … T_14` take `--positions-from <T_15 dir>`. Every
grid directory then has the same 50 exemplar columns (DATA_CONTRACT §7). The grid's `--poles` is `S ∪ S_gate`
(§4.4) and each explain directory's positions list is its own subset.

### 2.2 Explain and test (API, no GPU)

Agents are Anthropic API calls with model id `claude-opus-5-5` (from this session's environment list,
2026-10-19; logged per call with temperature, prompt SHA-256 and response). Three roles, strictly
separated by what they see:
**W0/W1 amendment:** the run uses the open-weights backend on DeepInfra, every role on
`deepseek-ai/DeepSeek-V4.1-Flash-1017`, temperature 1.0, first live call logged 2026-10-20 (DATA_CONTRACT
§8); Agent 3 is an independent second call. The Anthropic backend stays available and is used only if a key
arrives; a change of model id mid-run is a dated amendment and a reason to redo the discovery calibration.
**W2-D amendment (2026-10-20, before the freeze):** Agents 1 and 3 (explainer, second reader) run on
`anthropic/claude-opus-5-5` through the same DeepInfra endpoint (`settings/deepinfra_opus_explainer*.json`,
temperature 1.0 and json_schema accepted there). Measured on the 120 discovery poles with prompt v2 and the same
writer: §6.1 pass 27/119 with DeepSeek-V4.1-Flash, 54/119 with Opus 5.5 (JOURNAL 08:20Z). The writer's and the
judge's model ids are decided the same way, one variable at a time on the discovery set, and recorded in the
explain manifest and `verdict discovery` report of the frozen version; the frozen version is a (prompt SHA-256,
model id per role) pair. The leak check on explanations refuses only the ⟦ ⟧ marker, activation values and
references to a shown context (DATA_CONTRACT §9); refused answers are re-asked by `explain reject`.
**Frozen 2026-10-20 (tag `already-clean-freeze-2026-10-20`, JOURNAL 16:48Z):** prompt v3 (explainer and second reader
SHA-256 `e36cf31d…`, writer `ed68dcec…`, job writer `3ab19637…`; explain 53cc6515), explainer, second reader, writer and
job writer on `anthropic/claude-opus-5-5`; discovery pass 73/118 = 61.9% against v2's 71/119 under the same probe rule;
the clusterer and judge are tuned on `S_gate` at W4 (§4.4) and are not part of this freeze.
**Model policy amendment (2026-10-20 17:15Z, the user's rule, supersedes the model half of the freeze):** no Opus model
(`anthropic/claude-opus-*`) is called through any API from now on; the LLM runner refuses a settings file or request that
names one on an API backend. Opus is available only as a Claude Code subagent through the manual backend, sparingly: at
most the `S_gate` judge gate or a re-read of a failing control, never a 600-pole batch. The API budget is $200 for the
study; $155 was spent by 17:15Z (§12). W2-C under the amendment: the 1,351 banked Opus explainer answers (1,344 in use after 7 leak rejects, checker F9; batch `c1`:
Q4B 465, G4 460, O7 426, the first poles of each cell in pole-id order, so lower layers) stay the explainer answer for
those poles; the remaining 489 poles' explainer, and the writer and job writer for every pole, run on the best non-Opus
candidate of the discovery rounds 6–9 (Sonnet 5.5, DeepSeek-V4-Pro, GLM-5.3, Gemini 3.7 Flash, each the four Q1 roles
with prompt v3, scored like for like with rounds 4 and 5; **decided 2026-10-20 18:40Z: `anthropic/claude-sonnet-5-5`, round 6
pass 81/118 = 68.6% (59.8–76.3) at its own θ_s = 0.2665, which is 71/118 at the frozen θ_s = 0.3153, against Opus v3's 73/118
(checker correction 2026-10-21: the rounds were compared under different θ_s; at the frozen θ_s the two are within two poles,
and the decision stands on cost); DeepSeek-V4-Pro 21/119, Gemini 3.7 Flash 49/119; GLM-5.3 gave no
answer in 15 minutes and was dropped; JOURNAL 18:40Z**); the second reader runs on `deepseek-ai/DeepSeek-V4.1-Flash-1017`
for every pole, so the agreement of §6.2 is cross-model and labelled so; clusterer and judge run on the same Flash model
(W4 may change them on `S_gate` within the budget). Every Q1 readout is also reported split by explainer model, and the
depth-tercile table carries the caveat of the explainer split by depth tercile (checker F3, 2026-10-21; counts written here 2026-10-23, Opus / Sonnet per tercile 0, 1, 2: Q4B 202/2, 196/0, 65/135; G4 210/2, 191/3, 54/140; O7 208/12, 204/16, 14/186), so the by-model split compares mostly shallow Opus poles with mostly deep Sonnet poles; on discovery 33 of 117 non-RARE poles pass under one explainer and fail under the other, reported beside kappa in RESULTS. Projected cost
≈ $31. The explain manifest records the model id per pole and role.

- **Agent 1, the explainer.** Sees, for one pole: the 30 strongest contexts, `stats/`, the SwiGLU regime
  split of the top-50, and the top-10 vocabulary items by `W_out[:, i]·W_U` (promoted and suppressed
  tokens, the output-centric view; last-layer suppression is the essay's "say or just think" motif,
  essay.md 305). Writes: a one-sentence explanation; the list of distinct jobs it sees (1, 2, 3 or more),
  with the relation between jobs (related / unrelated) and one line of reasoning; a class label from
  {ONE, TWO-REL, UNREL, UNK}; a confidence in [0, 1]. It is told the essay's caveats: weak activations
  usually matter less, the double-negative regime is an echo of the other pole, a neuron that predicts the
  next token is not polysemantic for doing so (essay.md 31). **W0 amendment (prompt v2, to be tuned on the
  discovery set at W2-D and frozen):** it is also told, in the essay's words, to describe the shared pattern
  or the distinction the pole draws rather than a list of nouns (essay.md 97–99: "if I couldn't use groups of
  nouns at all, how would I describe this neuron?"); that the summary motif puts the firing on punctuation
  or on the word before the cue (essay.md 41–43, 187); that long-context dependence is a job, not noise
  (essay.md 191); that UNREL is reached only after the shared-pattern search fails and each job can be
  written down on its own; and that UNK is the honest answer when no explanation is trusted (essay.md 209).
  W0's explainer confidence tracked the test (mean 0.50 on passing poles, 0.34 on failing), so confidence
  is kept as a banked field.
- **Agent 2, the blind writer.** Sees only Agent 1's one-sentence explanation. Writes five texts meant to
  drive the pole strongly, five near misses meant not to, five texts on the most distant concept it can
  invent, and one sentence meant to drive the pole maximally (blind maximum, essay.md 389–391). It never
  sees a context or an activation.
- **Agent 3, the second reader.** An independent sample of Agent 1's call (same inputs, temperature 1,
  different seed) for inter-rater agreement on the class label and job count.

Every text Agent 2 writes is scored on the GPU (§2.1 `probe/`) at the pole, with the score defined as
`max over the text's tokens of a` for the positive pole (`min` for the negative) divided by the pole's
corpus `max|a|`. This is the essay's quantity (essay.md 637: "share of the neuron's strongest example in
the corpus").
**W2-D amendment (2026-10-20):** the maximum is taken over the text's positions ≥ 1; position 0 has no context
and is excluded, as window position 0 is excluded from the corpus statistics (§4.2); a text under two tokens
is refused before upload. `max|a|` is the W1 50M-token harvest's, never the scoring job's own. An answer the
provider filters or the model refuses is re-asked up to three times and then recorded as declined; a pole
whose writer was declined has no texts and is UNK-N with the role in the reason (§2.3).

### 2.3 Classes (Q1 readout)

Per pole, after scoring, the pipeline assigns one of:

- RARE: fewer than 50 tokens in `C` with `|a| ≥ 0.01·max|a|`, or `max|a| = 0`. Not classified; counted.
- ONE: Agent 1 says one job and the explanation passes the legibility rule (§6.1).
- TWO-REL: Agent 1 says two or three related jobs and the explanation passes.
- UNREL: Agent 1 says two or more jobs it judges unrelated, and at least two of them are individually
  supported: for each such job, Agent 2 writes five positives from that job's sentence alone, and at
  least 3 of 5 are strong (§6.1's positive clause only). These extra calls are counted in §12.
- UNK: everything else, including a ONE or TWO-REL whose explanation fails. **W0 amendment:** UNK is
  reported in two parts, UNK-N (Agent 1 declined to explain) and UNK-F (an explanation was given and failed
  §6.1), because they say different things about the observer: UNK-N is a pole the explainer could not read
  with these inputs, UNK-F is an explanation the test rejected. Neither is evidence about the pole's number
  of jobs (§0b item 1). On W0's discovery set all 8 UNK poles were UNK-F.
  **W2-D amendment (2026-10-20):** UNK-N covers any agent that declined, with the role in the reason: the
  explainer giving no explanation it trusts, or the writer's answer filtered by the provider or refused by the
  model after three attempts (then the pole has no texts). The count per role is reported with every cell.

Also banked per pole, without a rule: the double-negative share; whether the pole is a single-token
suppressor (last layer only: the strongest `W_out·W_U` entry is negative and at least twice the second);
Agent 1's confidence; the share of the pole's top-50 at window position 0 (§4.2).

### 2.4 Clusters, births, parents (Q2 readout)

Only for the sampled poles `S` (600 poles, §4.4; W0 amendment) on OLMo-3, over the checkpoint grid
`T_0 … T_15` (§4.3).

- **Clusters at the end.** At `T_15`, Agent 1 (clustering role, a fourth prompt) groups the pole's top-50
  contexts from the 20M-token corpus into `m` clusters, names each, and marks each pair related or
  unrelated. Contexts in the double-negative regime (`g < 0, u < 0`) are the other pole's echo (essay.md 373):
  the clusterer puts them in one cluster flagged `echo`, which is reported but excluded from `m`, from the
  parent test and from every count (amended 2026-10-19; matches Agent 1's instruction). `E_c` is the exemplar set of cluster `c` (its contexts' `(doc_id, position)`). A second
  independent sample gives `m'`; agreement `|m − m'| ≤ 1` is the B4 control.
- **Presence and birth.** At checkpoint `t`, cluster `c` is present if the median exemplar in `E_c` has
  `a⁺_t ≥ q_{99.9}(a⁺_t)`, where `a⁺_t = max(±a_t, 0)` is the activation in the pole's own direction (an
  exemplar that fires the other pole does not count; amended 2026-10-19 after review, replacing `|a_t|`)
  and the percentile is the pole's own 99.9th percentile of that quantity at `t` (within-checkpoint normalization,
  so that activation scale drift cannot create or hide presence). `birth(c)` is the first `t` at which `c`
  is present and stays present at the next grid point; a cluster first present at `T_15` has `birth = T_15`
  (no next point exists; it is late-born with `t_before = T_14`; amended 2026-10-19). The raw-ratio reading (median exemplar
  `median_{E_c} a⁺_t ≥ 0.3·max|a_t|`, with the same a⁺ = max(s·a, 0) as the percentile reading and median > 0; **wording aligned to the code 2026-10-21**) is computed too; §6.3 requires the two readings to agree on the late/early
  split for a cluster to enter the primary count (B3).
- **Strength at birth.** `r(c) = median_{E_c}|a_{birth(c)}| / max|a_{birth(c)}|` (weak-first is `r`
  well below 1).
- **Main and late clusters.** `main` is the cluster with the largest `max|a|` at `T_15`. A cluster is
  **late-born** if `birth(c) > min_c' birth(c')` for that pole. Clusters whose `birth(c) = T_1` (the first
  trained checkpoint) have `T_0` (random init) as their only earlier record and are excluded from the
  parent test and counted as "formed before step 1000".
- **Parent test (two-alternative forced choice).** For a late-born cluster `c` with `t_b = birth(c)` and
  `t_before` the grid point before `t_b`: the judge (a fifth prompt, same model id) sees ten exemplars of
  `c` (contexts at `T_15`) and two unlabeled records, each the top-20 contexts of a pole at `t_before`:
  the true pole's and a decoy's. Any context whose activating token lies in the same document within 64 tokens before or 16 after an
  exemplar of `E_c` (the rendered window) is removed from both records (amended 2026-10-19 from exact
  positions only, after review). Contexts are shown as text with the activating token marked and no numbers, so activation
  scale cannot identify the checkpoint or the pole. The judge names the record the cluster grew from. Decoys: **random** (a pole drawn uniformly
  from the same layer, primary) and **hard** (the same-layer pole whose top-50 at `T_15` shares the most
  token strings with the true pole, secondary). Four judgments per cluster per decoy type: two judge
  samples × two decoy draws (the hard decoy is fixed, so two samples × one decoy, two judgments).
- **Positive control for the judge.** For each pole's `main` cluster, the same 2AFC with `c = main`,
  `t_before = T_14`, exemplars removed from both records. This measures the judge's ceiling on a pair that
  is related by construction. Its flag rate is `f_ctrl` (§6.3).
- **Birth order.** Per pole with `m ≥ 2`, whether `main` is the earliest-born cluster. Null: birth ranks
  permuted within the pole.
- **Growth over the last half.** For every non-main cluster, the sign of the change in
  `median_{E_c} a⁺_t / max|a_t|` from `T_12` (step 320k) to `T_15` (**2026-10-21:** a⁺ as in the percentile reading; undefined when either max|a_t| is 0, and such clusters are counted as excluded).
- **Center birth curve.** The blind-maximum sentence and the five positives for each pole, scored at every
  grid point: the first `t` at which the blind maximum reaches 0.3 of its `T_15` score.

## 3. Design and instruments

| id | instrument | NEW / REUSED | what it adds |
|---|---|---|---|
| N1 | `infra/pg/neurons_harvest.py`: hooks, top-k, stats, sparse, probe, NLL, pass-through | NEW; model loading through `transformers.AutoModelForCausalLM` with the AMI's pinned transformers (`bake_env.sh` line 29: `transformers>=4.47,<6`; `olmo3` and `gemma3` model types require ≥ 4.57 and ≥ 4.50; the installed version is read at W0 and logged) | the whole Q1 and Q2 data |
| N2 | `infra/pg/neurons_corpus.py`: fixed shard list, document order, token budget, SHA-256 of the token arrays | NEW | the corpus manifest (§4.2) |
| N3 | `infra/pg/neurons_explain.py`: Agents 1–3, clustering and judge prompts, call log | NEW; prompts seeded by the essay's protocol (essay.md 389–395, 637–645); Transluce's 2024 explainer pipeline and NeuronScope 2601.03671 to be read at source before freeze and reused where they are the same step (CHECKS.md I0 row) | explanations that can lose |
| N4 | `infra/pg/neurons_verdict.py`: classes, `p_u`, birth order, grid fill; refuses to run without a `PREDICTIONS` journal entry | NEW | the frozen rules, executable |
| N5 | `infra/aws_pg/jobs/neurons_harvest.py` (one checkpoint, `--repo --revision --corpus-tokens --sample-poles --probe-texts --out`) and `jobs/neurons_pipeline_steps.json` for the W3 loop through `jobs/pipeline.py` | NEW job module; REUSED runner (`pipeline.py`, read 2026-10-19: steps from an S3 JSON, each `python -m module args --out out/<name>/`, fail-loud, `pipeline_status.json`) | ephemeral, server-side sequencing |
| R1 | `diff_harvest.hf_online_setup()` | REUSED (reads the HF token from SSM `<ssm-parameter>`, `diff_harvest.py` lines 30–48) | in-job download of OLMo-3 revisions and the Dolma 3 shards |
| R2 | `text_eval.clean_text_perplexity` (STAGEDIFF B3) | REUSED for G1 | parity with a banked number |

**Decisive experiment.** W3's parent test on late-born clusters against the random decoy (`p_u`, §6.3).
Everything else corroborates or calibrates: Q1 calibrates the explainer and supplies the sample strata;
the hard decoy guards B1; birth order and weak-first are the essay's two further predictions (essay.md
815: "a specific shape in both semantic space and time"); the center birth curve is the cheapest
developmental readout and will exist for every sampled pole whatever the clustering does.

**What each extra piece adds.** Without Q1 we could not stratify Q2's sample toward the messy poles where
packing would hide. Without the sparse record we would need one extra forward per checkpoint per
exemplar set. Without the step-0 checkpoint we could not show that the top-50 of a random network is
noise (the degenerate control for "present"). Without the two 4B models the audit would be one family.

## 4. Data and substrate

### 4.1 Models

| key | repo | revision | hidden | MLP width | layers | act | neurons | role |
|---|---|---|---|---|---|---|---|---|
| Q4B | `Qwen/Qwen3-4B-Base` | `906bfd4b…` (bucket; `matrix_repo_facts.json`) | 2560 | 9,728 | 36 | silu | 350,208 | audit; discovery set |
| C0 | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f…` (bucket; STAGEDIFF C0, the model the banked 2.2444 came from) | 2560 | 9,728 | 36 | silu | 350,208 | preflight only (G1 parity). **W0 amendment:** replaces `Qwen/Qwen3-4B`, which is STAGEDIFF's P0 and has no banked B3 number; it read 2.549 on the same protocol, a correct measurement of a different model |
| G4 | `google/gemma-3-4b-pt` | `cc012e0a…` (bucket) | 2560 | 10,240 | 34 | gelu_tanh (the `gemma3` default; the bucket config sets no override; confirmed at W0 from the loaded module) | 348,160 | audit, second family |
| O7 | `allenai/Olmo-3-1025-7B` | `stage1-step<n>` for `n` in §4.3 | 4096 | 11,008 | 32 | silu | 352,256 | audit at `T_15`; development |

G4's text config was read from the bucket snapshot on 2026-10-19 (hidden 2560, intermediate 10,240,
34 layers). OLMo-3 facts read 2026-10-19: `config.json` at `stage1-step0`, `stage1-step500000` and `main` agree on
hidden 4096, intermediate 11008, 32 layers, silu, vocab 100,278, untied embeddings; `main` adds YaRN
rope scaling to 65,536 context, stage-1 checkpoints have `rope_scaling: null` at 8,192. `tokenizer.json`
has the same git object id (`5fe17212…`) at `stage1-step0`, `stage1-step1000` and `stage1-step1413814`;
every checkpoint is 3 safetensors shards totalling 14,596,063,712 bytes. The model card states "7B Model:
No merging" and stage 1 = 5.93T tokens on `dolma3_mix-6T-1025`. Stage-1 checkpoints exist at every
multiple of 1,000 steps from 0 to 1,413,000 plus 1,413,814 (1,421 branches). 5.93T / 1,413,814 steps
≈ 4.19M tokens per step, so one checkpoint interval is ≈ 4.2B tokens ≈ 0.07% of stage 1. We use
stage-1 checkpoints only; `main` (after stage 2 and 3) is out of scope.

Q4B, Q4 and G4 are in the S3 weight cache (`pg run --weights`). O7 revisions are downloaded in-job from the
Hub after `hf_online_setup()`, one at a time, deleted after harvest (14.6 GB each; the instance root is
200 GB). Pivot if Hub throughput is below 50 MB/s at W0: seed the 16 revisions into
`weights/hf/hub/` with `seed_weights.sh` (≈ 234 GB in S3) and switch N5 to `--weights`.

### 4.2 Corpus `C`

Source: `allenai/dolma3_mix-6T-1025-7B` (OLMo-3-7B's own stage-1 mix; not gated; 99,998 shard files
under `data/<source>-<topic>[-<part>]/shard_<n>.jsonl.zst`, read 2026-10-19). Rule, executed by N2 and
recorded as a manifest with SHA-256 of every shard file and of the final token arrays:

1. List the `data/*` directories (165 at commit `2ca900fb…`, read at W0), sort lexicographically, take
   the 50 at indices `floor(i · n_dirs / 50)` for `i = 0 … 49` (**W0 amendment:** the old `k = floor(n_dirs
   / 50)` rule left 55 directories), and from each take the lexicographically first shard file. Shards are
   `.jsonl.zst` or `.jsonl.gz` (about half each; a name may carry a digit suffix after `.zst`, which is
   excluded); both are read. When the first file ends before the directory's token budget, the next file
   in lexicographic order is opened (the PDF directories' first `.gz` files are 0.11–0.28 MB).
2. Read documents in file order. **W0 amendment:** documents are deduplicated by the SHA-256 of their text
   before selection; the W0 corpus held 2,364 selected documents with only 1,358 distinct texts (34% of the
   characters were copies, within-directory upsampling of the Dolma 3 mix), and copies took 15–20% of
   top-k slots. Document selection is fixed once with the OLMo-3 tokenizer: the
   documents that fill the first 1,000,000 tokens of each shard form the 50M corpus (50 sources), and
   those that fill the first 400,000 form the 20M corpus (a prefix under the same order). Every model
   tokenizes the same documents with its own tokenizer, so Q4B's and G4's token counts differ from 50M by
   a few percent; the manifest records each tokenizer's count and token-array hash.
3. Windows of 2,048 tokens, documents concatenated with the tokenizer's EOS between them. **W0 amendment:**
   the first token of a window has no preceding context; on G4 it held 1.5% of all top-k slots (31× a
   uniform spread) and 9,256 G4 poles and 1,632 Q4B poles had half or more of their top-50 there. Window
   position 0 is excluded from top-k and stats (a context the model never saw is not a job), and the share
   of a pole's pre-exclusion top-50 that sat there is banked per pole.

The 50M corpus is used at the final checkpoint of every model (Q1). The 20M corpus is used at every OLMo
grid point including `T_15` (Q2), so every developmental comparison is at equal corpus. The 10M prefix
gives the B2 sensitivity reading. For Q4B and G4 the corpus is in-distribution web, code, PDF and math
text but not their training data; this is stated with the result. The essay used 10B tokens for its
maximum examples (essay.md 389); our corpus is 200× smaller, so RARE is a class we report, not hide.

### 4.3 Checkpoint grid (O7)

| id | step | tokens (≈) | share of stage 1 |
|---|---|---|---|
| T_0 | 0 | 0 | 0 (random init; degenerate control) |
| T_1 | 1,000 | 4.2B | 0.07% |
| T_2 | 2,000 | 8.4B | 0.14% |
| T_3 | 3,000 | 12.6B | 0.21% |
| T_4 | 5,000 | 21B | 0.35% |
| T_5 | 8,000 | 34B | 0.57% |
| T_6 | 12,000 | 50B | 0.85% |
| T_7 | 20,000 | 84B | 1.4% |
| T_8 | 35,000 | 147B | 2.5% |
| T_9 | 60,000 | 252B | 4.2% |
| T_10 | 100,000 | 419B | 7.1% |
| T_11 | 180,000 | 755B | 12.7% |
| T_12 | 320,000 | 1.34T | 22.6% |
| T_13 | 560,000 | 2.35T | 39.6% |
| T_14 | 1,000,000 | 4.19T | 70.7% |
| T_15 | 1,413,814 | 5.93T | 100% |

Every branch named exists (checked against the saved refs listing). Early density is deliberate: persona
vectors were reported to form within 0.22% of OLMo-3 pretraining (CHECKS.md I10 row, 2605.13329), which
is between `T_3` and `T_4`. Pivot (pre-specified): if more than half of all clusters are born at `T_1`,
nothing finer than 1,000 steps exists for OLMo-3; report it as a resolution limit and, as an exploratory
extension, run the same pipeline on `EleutherAI/pythia-2.8b`, whose checkpoints include steps 1, 2, 4,
…, 512 (CHECKS.md I10 row: 154 checkpoints per model; verify at source before launching).

### 4.4 Samples, discovery and confirmation

- **Audit sample.** Per model, 300 neurons drawn uniformly at random with a logged seed, stratified
  uniformly over layers; both poles → 600 poles. **Checker note (2026-10-21):** after the random discovery holdout the
  Q4B audit sample holds 4–10 neurons per layer and O7 8–10; the depth terciles stay balanced, no readout changes. The essay's own "could lose" test used 200 random
  neurons (essay.md 637).
- **Discovery set.** 60 further neurons from Q4B (120 poles), disjoint from the audit sample, on which the
  prompts, `θ_s` (§6.1) and the class rules are tuned and frozen. Nothing tuned on it is re-tuned later.
- **Development sample `S`.** **W0 amendment: 600** OLMo poles at `T_15` (was 400; §15 item 7 decided for
  power: at 400 a true `p_u = 0.05` fires H_RG in only 70% of simulated datasets, at 600 near 90%), drawn
  from the audit sample's classified poles: every UNREL and UNK pole first (where packing would hide), then
  every TWO-REL pole, then ONE poles at random until 600; RARE poles are excluded. Population-weighted
  estimates are reported beside the stratified ones. A **development discovery set `S_gate`** of 40 further
  OLMo poles, disjoint from `S`, drawn the same way, is used to tune the clustering and judge prompts and to
  set the judge's validity gate (§6.3); the parent test on `S` is computed by N4 but not read until W5.
  **W0 amendment:** the grid is harvested once with `--poles` = `S ∪ S_gate`; `S_gate` and `S` live in two
  explain directories whose positions lists are their own subsets (the explain side accepts a grid pole list
  that is a superset); `judge_gate.json` must be written before the first `S` judge batch, and N4 refuses
  otherwise.
- **Sequencing (W0 amendment, enforced by N4).** `θ_s` is set on the discovery set, and the development
  (audit) explain directories must start after `θ_s` is written; N4 refuses a Q1 directory whose explain
  started earlier. In W0 this refusal fired on the first attempt and a development set was added.
- **Pairing.** Items are poles; every per-pole quantity is keyed by `(model, revision, layer, neuron,
  pole)`. Clusters are keyed by that plus the cluster index at `T_15`.

## 5. Controls, each tied to a claim

| control | what it checks | claim it protects | gate |
|---|---|---|---|
| G0 pass-through | the MLP output recomputed from the hooked `g`, `u` equals the module's output within bf16 tolerance (`max abs diff ≤ 1e-2` of the output's max abs) on 8 windows at every layer | every harvested activation is the one the model used | hard gate at W0 and at the start of every harvest job. **W0:** passed on every model and checkpoint (0.0034–0.0065). Its registered failing input (silu↔gelu_tanh swap) fails 31–33 layers on final checkpoints but only 3 of 32 at step 1000, where `|g|` is small; before W3 a second failing input with power at early checkpoints is registered (a shifted or scaled `u`) |
| G1 NLL parity | **W0 amendment:** C0's (`Qwen/Qwen3-4B-Instruct-2507`) wikitext-2 mean NLL from the harvest forward equals the banked STAGEDIFF B3 value, `studies/stagediff/results/w1/bat-C0/1/out/metrics.json` → `readouts/B3/clean/mean_nll = 2.2444`, within 0.02; the readout reuses `text_eval.shell_text_perplexity` through `StageForward(bos_id=None)`, B3's exact call (`clean_text_perplexity` asserts a BOS, which no Qwen3 tokenizer defines). W0 reproduced 2.244386524302183 bit for bit | the forward condition is the one our earlier studies used | hard gate at W0: passed |
| G2 random init | at `T_0`, the top-50 of a sampled pole has no cluster with `m ≥ 1` that Agent 1 labels with confidence above 0.5 in more than 10% of poles | "present" is not triggered by an untrained network | reported; failure means the presence rule is too loose and is tightened on the discovery set before freeze |
| shuffle (Q1) | every pole's positives, near misses and distant texts are also scored at a different random pole from the same layer | the strong-response rate is specific to the explained pole | the shuffled strong rate is the null for `θ_s` (§6.1) |
| distant texts (Q1) | the essay's test for packing (essay.md 643) | the explanation is not "anything" | reported as the packing rate |
| judge positive control (Q2) | `main` cluster against the pole's own `T_14` record | the judge can recognize a related pair | validity gate: accuracy ≥ 0.80 on the development discovery set, else the instrument does not run on `S` |
| judge chance (Q2) | 50% by construction | the null for the parent test | drawn on every panel |
| hard decoy (Q2) | the most token-overlapping same-layer pole | B1: separation is not by domain alone | secondary row; if accuracy against the hard decoy on the positive control is below 0.80 the hard row is reported as not measurable |
| seed (Q2) | two judge samples, two clustering samples | agent variance | reported as agreement; both samples enter the four-judgment rule |
| corpus size (Q2) | births recomputed on the 10M prefix | B2 | fraction of clusters whose late/early split changes; reported |
| width (Q1) | classes recomputed with `θ_s ± 0.1` | the audit numbers are not knife-edge | reported as a band |

## 6. Frozen decision rules (set on the discovery sets at W2-D, before any confirmation number is read)

6.1 **Strong response and legibility.** `θ_s` = the 99th percentile of the shuffled-pole scores on the
discovery set (the best achievable null for "fires on an unrelated text"). A text is strong if its score
≥ `θ_s`. An explanation **passes** if at least 3 of 5 positives are strong, at most 2 of 5 near misses
are strong, and at most 1 of 5 distant texts is strong. The essay's comparison reading (strong = 0.30;
win if strong positives outnumber strong near misses, tie if equal) is computed and reported beside ours,
labelled as the essay's rule; it is never used to decide. If `θ_s` on the discovery set exceeds 0.5 the
corpus is too small for the instrument and W1-C does not start. **Set 2026-10-20T16:44:57Z: θ_s = 0.3153**
(`bank/w1/q4b_disc_v3/verdict/theta.json`, 99th percentile of 1,785 shuffled-pole scores under the position-≥ 1 probe
rule; JOURNAL 16:48Z).

6.2 **Audit result (Q1).** Per model, the fractions ONE, TWO-REL, UNREL, UNK, RARE over the 600 poles with
Wilson 95% intervals, by depth tercile, and the inter-rater kappa (**2026-10-20 amendment:** between the explainer's label
and a second read by `DeepSeek-V4.1-Flash-1017`, so a cross-model agreement; reported by explainer model). There is no pass/fail; the result is
the table. The sentence we will write is of the form "on model X, p% of poles (interval) have one job
under explanations that pass §6.1". The essay's 90 / 7 / 2–4 is drawn as a reference line labelled as
its own estimate under its own thresholds, never as our threshold.

6.3 **Unrelated share (Q2 primary).** Model: a late-born cluster is either related to its pole's earlier
record, in which case the judge is correct with the positive-control accuracy `a_ctrl`, or unrelated, in
which case the judge is at chance, 0.5. With `A` the mean per-cluster accuracy over late-born clusters
against the random decoy (four judgments per cluster, §2.4), the estimated unrelated share is

    p_u = (a_ctrl − A) / (a_ctrl − 0.5),   clipped to [0, 1],

with a 95% bootstrap interval over poles (clusters resampled with their pole; `a_ctrl` resampled over its
own poles in the same draw; **W0 amendment:** a draw with no late-born cluster or no control is redrawn and
counted, the count is reported as `degenerate_draws`, and the estimate is refused if redraws exceed 1% of
the 10,000 draws; on W0's 4-pole grid 52 draws were redrawn). Rules: **H_RG** if the upper bound of `p_u` < 0.10; **H_SP** if the lower
bound > 0.20; **mixed** otherwise, reported as such with the point estimate. A cluster is **flagged** if
at most one of its four judgments is correct; flags are used for the H_MIG count (§6.6) and the gallery,
not for `p_u`. (The flag-rate estimator was considered and rejected before data: at `n = 400` clusters
its interval on `p_u` is about ±0.13, which cannot separate the two rules; the accuracy estimator's is
about ±0.04 to ±0.07, §7.) A cluster enters the count only if the percentile and raw-ratio birth readings
agree that it is late-born (B3). The 0.10 and 0.20 are judgments written before data; they bracket the
essay's own not-understood estimate (2–4%, essay.md 29) and what we would call "a substantial fraction".
Validity gate: the positive-control accuracy on the development discovery set must be ≥ 0.80, else the
parent test is reported as not valid and only the center birth curve and birth order are read.

6.4 **Birth order (Q2 secondary).** `b` = fraction of poles with `m ≥ 2` whose `main` is earliest-born.
Null: 10,000 permutations of birth ranks within each pole. Supports the "main job is the oldest" form of
H_RG (essay.md 129: the improved architecture "stays on the fields") if `b` exceeds the 97.5th percentile
of the null; otherwise the pole population shows drift, the InceptionV1 pattern (essay.md 131), which is
reported as drift and does not count against H_RG. `1 − b` is the drift share.

6.5 **Vestigial (H_VEST).** `v` = fraction of non-main clusters born at or before `main` **and** shrinking
over the last half. H_VEST is the label only if `v`'s lower bound > 0.5.

6.6 **Replacement (H_MIG).** Count of poles whose `main` at `T_15` is late-born and flagged. Reported as a
count with its share of `S`; no rule beyond §6.3.

6.6a **Declined agent answers at W4 (2026-10-21 amendment, before any W4 call).** A pole whose clusterer sample is
declined after the attempts cap leaves Q2 and is counted as `declined_clusterer` beside RARE and UNK-N; a cluster or
control whose judgment is declined leaves `p_u` and `a_ctrl` and is counted as `declined_judgments`; both counts and
ids are in `judge_gate.json` and the unblind metrics. The center birth curve (exploratory) is recorded as not
computed when only T_15 carries probe scores. A judge batch on `S` is refused before `judge_gate.json` exists.
**2026-10-23 amendment (checker F6):** the per-checkpoint probe job `x/center_probe/1` (the W2-C O7 texts at all 16
checkpoints, manifest prepared 2026-10-21) ran on 2026-10-23 after the first unblind; its outputs were read only at W5,
by `neurons_verdict unblind --center-probe`, and the §11 row is filled from them as the exploratory reading §2.4 defines
(first t at which the blind maximum reaches 0.3 of its own T_15 score in that run). No rule of §6.3–6.6 uses it.

6.7 **Mixed outcomes.** Every combination of 6.3–6.6 is reported in the grid (§11). No rule is
reinterpreted after unblinding; a wrong rule is stated as wrong and both readings are given.

## 7. Statistical plan

- Fractions: Wilson 95% intervals; `p_u`, `b`, `v`: bootstrap over poles, 10,000 resamples, seed logged.
- Multiplicity: three decisive statistics (`p_u`, `b`, `v`), reported together. `p_u` is decided by its
  bootstrap interval alone and carries no p-value; Holm correction over the two p-values of `b` and `v`,
  always both, never conditional on an outcome (amended 2026-10-19: a Holm set that shrinks after seeing
  a rule, or a p-value whose direction is picked after the rule, is not a correction). Everything in §2.4's "growth" and "center birth curve" and
  every by-layer or by-class breakdown is exploratory and labelled so.
- Effect size against the null: `p_u` is already a share above the chance-calibrated null; `b` is
  reported as the gap to the permutation mean.
- Stability: Q1 by depth tercile and by SwiGLU regime (poles with double-negative share ≥ 0.8 reported
  separately); Q2 by class stratum, reweighted to the population.
- Power (**W0 amendment:** `S` = 600 poles, §4.4; W0's four poles carried 1.25 late-born primary clusters
  each after exclusions, so 600 poles give ≈ 750 clusters before exclusions; simulation at 600 clusters puts
  the H_RG firing rate near 90% at a true `p_u = 0.05`, against 70% at 400): 400 poles with on average 1.5 non-main clusters give ≈ 600 late-born clusters before exclusions;
  the arithmetic below assumes 400 after exclusions. With `a_ctrl = 0.95`, a true `p_u = 0.05` gives
  `A ≈ 0.928` and a true `p_u = 0.30` gives `A ≈ 0.815`. The per-cluster accuracy is a two-point mixture,
  so its variance over clusters is ≈ 0.024 at `p_u = 0.05` and ≈ 0.07 at `p_u = 0.30`; the 95% half-width
  on `A` at `n = 400` is ≈ 0.015 and ≈ 0.026, which is ≈ 0.034 and ≈ 0.058 on `p_u`. So a true 0.05 has
  upper bound ≈ 0.09 and a true 0.30 has lower bound ≈ 0.24: the two rules of §6.3 are separable at the
  edges and the middle is labelled mixed by design. If exclusions leave fewer than 200 clusters, the
  result is reported with its wider interval and labelled underpowered; no rule changes.

## 8. Threats to validity

| threat | effect on the result | mitigation |
|---|---|---|
| corpus 200× smaller than the essay's | RARE inflated; some clusters missed; births late | report RARE; 10M/20M sensitivity; state the limit |
| first interval is 4.2B tokens | early births unresolved; many clusters at `T_1` | "formed before step 1000" is its own class; Pythia pivot (§4.3) |
| judge wins by domain, not relation (B1) | `p_u` underestimated | hard decoy row |
| the explainer sees top examples and writes an explanation that only re-describes them | Q1 inflated | Agent 2 never sees examples; near misses and distant texts; shuffle null sets `θ_s` |
| double-negative regime read as a second job | UNREL inflated | regime split shown to Agent 1; poles with share ≥ 0.8 reported separately (essay.md 373) |
| activation scale drift across checkpoints (B3) | false births | within-checkpoint percentile rule; raw-ratio agreement required |
| agent clustering variance (B4) | wrong `m`, wrong `E_c` | two samples; agreement reported; a sample-1 cluster enters the primary count only if a sample-2 cluster holds a strict majority of its exemplars (rule written 2026-10-19); the exclusion rate is reported |
| stratified sample over-represents messy poles | `p_u` for the population overstated | report stratified and population-weighted |
| API model drift between sessions | judgments not comparable | all judging for one readout in one session; model id and date per call |
| 2,048-token windows against 8,192 training context | long-range poles under-sampled | stated; no fix in this study |
| transformers version mismatch for `olmo3` | load failure or wrong rope | W0 smoke loads `stage1-step1000` and `T_15`, logs the version, checks G0 and NLL finite |
| gated-MLP activation for Gemma 3 (`gelu_tanh`) differs in its negative range | regime split not comparable to silu models | reported per model; the regime figure is per activation function |

## 9. Runbook and blind protocol

9.1 **Preflight (principle 12).** W0 job on Q4: G0 at every layer, G1 against the banked 2.2444, then
on Q4B a harvest of 2M tokens with samples opened (ten contexts for three poles, printed with their raw
values). Then
the same on O7 `stage1-step1000` and `T_15` (download, G0, NLL finite, 2M tokens). Throughput
(tokens/s) is measured here and sets every later `--budget-min`. **W0 done 2026-10-19** (`verify/VERIFY.md`):
G0 passed everywhere; G1 passed on C0 and showed Q4 was the wrong model; throughput 3,990–4,023 tokens/s on
the 4B models and 3,291–3,393 on O7 (an L40S at 100% SM); Hub download 103–126 MB/s, above the pivot; the
whole explain → write → probe → verdict loop and a 16-point mini grid with the judge ran end to end on O7
with the manual backend. Launch form is now `--type g6e.2xlarge,g6e.4xlarge` (six launches found no
capacity on the first try). Every run's `corpus/` pull must include `tok/<tokname>/tokenizer/`.

9.2 **Waves** (fresh S3 prefix per attempt: `results/neurons/<wave>/<cell>/<attempt>/`).

| wave | cells | instrument | gate to pass |
|---|---|---|---|
| W0 substrate | dry-run of N1–N4 on a 4-layer stub (`matrix_forward.stub_text_model`) with samples written; `seed`: corpus manifest built on a CPU instance; smoke+preflight on Q4; smoke on O7 two revisions | N1, N2, N5 | dry-run green with samples; G0, G1 passed; throughput logged; first-batch samples in the journal |
| W1-D discovery harvest | Q4B 50M corpus, all neurons top-50 + stats; discovery poles' probe texts come later | N1 | STATUS exit=0; samples opened |
| W2-D discovery explain | Agents 1–3 on the 120 discovery poles; Agent 2 texts; a W1-D rescoring job for the texts and the shuffle; set `θ_s`; freeze §6. **W0 amendment:** the explainer and writer prompts are iterated here, on the discovery set only, before `θ_s` is set: each version's SHA-256 and its pass rate are logged in `JOURNAL.md`; the target is the pass rate the essay reports for its own test (76% positives strong, 11% near misses, 1% distant, essay.md 637–643), the stop is three versions or no gain, and whatever version is frozen is the version for every confirmation cell. W0's first-pass prompt passed 12 of 20 on O7. **Done 2026-10-20:** v2 and v3 on Opus 5.5 scored like for like (59.7%, 61.9%); v3 frozen; θ_s = 0.3153; tag `already-clean-freeze-2026-10-20`; the two scoring jobs ran on vast.ai (`infra/vast`, cell `pipeline_job`) after a night without g6e capacity | N3, GPU rescoring (one short job) | `θ_s` set and written; freeze tag |
| W1-C audit harvests | G4 50M; O7 `T_15` 50M (Q4B's W1-D harvest is reused; its audit sample is disjoint from the discovery set) | N1 | exit=0; samples opened |
| W2-C audit explain + score | Agents 1–3 on 600 poles × 3 models; one scoring job per model (texts + shuffle). **2026-10-20:** started 16:50Z on Opus, stopped at 16:57Z (provider 402) and at 17:12Z (the user's no-Opus rule) with 1,351 explainer answers banked; resumes under the §2.2 model policy amendment once rounds 6–9 name the model | N3, N1 probe | all texts scored; classes computed by N4 but not read |
| W3 development | pipeline job(s) over `T_0 … T_15` on the 20M corpus; **W0 amendment:** `T_15` first with `--poles S ∪ S_gate` (it writes its own exemplar record), then `T_0 … T_14` with `--positions-from` the `T_15` directory; the Q1 scoring of `S`'s texts is its own short step on `T_15` with `--poles probe_poles.json --probe-texts`, because one directory cannot carry both pole lists; two jobs of 8 checkpoints each so no job exceeds one budget | N5 + `pipeline.py` | `pipeline_status.json` complete; per-checkpoint G0; samples opened at the first checkpoint |
| W4 clusters and judge | clustering at `T_15` on `S` and on `S_gate`, in two explain directories; validity gate on `S_gate` written first; parent tests on `S` (random and hard decoys); positive controls. **W0:** 48 and 54 judge lines ran on the 4-pole grid with a_ctrl 1.00 | N3 | validity gate recorded before any `S` judgment is read (N4 refuses otherwise) |
| W5 unblind | `neurons_verdict` on banked JSON; grid filled once | N4 | `PREDICTIONS` entry in `JOURNAL.md` precedes every confirmation read |

9.3 **Blind protocol.** Q1's confirmation classes and Q2's `S` judgments are computed by the jobs and
written to S3 but not read until W5. `neurons_verdict.py` refuses to run unless `JOURNAL.md` contains a
`PREDICTIONS` entry dated before the first W2-C artifact and before the first W4 artifact. The person who
tunes prompts on the discovery sets does not open confirmation samples before W5; sample checks on
confirmation cells are done on the first ten poles only and logged as such.

9.4 **Sessions.** S1 = W0. S2 = W1-D + W2-D + freeze. S3 = W1-C + W2-C. S4 = W3 (launch, then the two
pipeline jobs run on their own; the session ends at launch after writing the handoff). S5 = W4. S6 = W5
and closeout. Each session ends at a wave boundary after writing `handoff/S<n>.md` with the NEW/REUSED
record state, open jobs, exact next action and resume steps. The first line of every status report is the
run/unrun/parked count over the cell roster (W0: 4 cells; W1: 3; W2: 4 scoring jobs; W3: 2; W4–W5: 0 GPU).

9.5 **No questions mid-run.** Every fork is resolved in §14. Anything the run needs and cannot derive is
a spec defect to fix before freeze; §15 lists the ones open today.

9.6 **Checkers.** Read-only, non-blocking sanity checks after every wave, run as Opus agents only, never
more than two at once, each in its own worktree. Briefs in `checks/CHECKERS.md` (to write at W0): did the
samples match the spec; did any gate have room to fail; were any numbers written before samples were
opened.

## 10. Artifacts

- Samples, one file per readout × arm, `out/samples/<readout>_<arm>.jsonl`: for Q1, per pole the 30
  contexts as the explainer saw them, the explanation, every Agent 2 text with its score and the strong
  flag, the shuffled-pole scores, the class; for Q2, per cluster the exemplars, the two records as the
  judge saw them, the judge's answer and the truth, the birth table. At least one passing and one failing
  pole per model and per class in the gallery; PENDING readouts listed, not omitted.
- `out/metrics.json` (every number a figure cites), `out/controls.json`, `out/manifest.json` (corpus
  hashes, revisions, transformers and torch versions, model id of every agent call), `events.jsonl`,
  `run.log`, compute per stage.
- Registry: every figure and number → generating script, input hashes, DISCOVERY or CONFIRMATORY.
- Figures (PNG for review, PDF for the paper; title states the finding; null drawn on the panel):
  F1 audit class fractions per model with intervals, essay reference line labelled; F2 strong rates for
  positives, near misses, distant and shuffled texts per model; F3 double-negative share, dead share and
  density distribution over the OLMo grid; F4 histogram of cluster births over the grid, main and non-main
  separately; F5 parent-test accuracy on late-born clusters, positive control and chance, random and hard
  decoys, with `p_u` and its interval; F6 a gallery of three poles' branching trees over training (one
  ONE, one TWO-REL, one UNREL ∪ UNK), each node a cluster with its exemplars and `r(c)`.
- Code: `infra/pg/neurons_*.py`, `infra/aws_pg/jobs/neurons_*.py`, the corpus manifest, and the prompts
  as files with their SHA-256 in `manifest.json`. These are the open pipeline; release is a separate
  decision after closeout.
- Study files: `SPEC_D1.md` (this, frozen at the tag), `JOURNAL.md` (live), `RESULTS.md`, `figures/`,
  `handoff/`, `checks/`, an index saying what is live.

## 11. Prediction grid (filled once, at W5, whatever the result)

Filled 2026-10-23T19:56Z from `bank/w5/verdict/grid.md` (unblind of 2026-10-23T19:55:33Z, `metrics.json` sha256 9129a195…; re-run 20:06:45Z with the checker F2 item 3 sensitivity reading added, same grid, sha256 56886c88…; re-run 23:28Z with the center birth curve row filled from the per-checkpoint probe job `x/center_probe/1` of the same day, every other cell unchanged, `metrics.json` sha256 4b2f3c7758573f33887dfaf983c44e77ab2018a3fc6707e4d0a78030973fca1e; earlier runs kept as `bank/w5_run1_20261023T1955Z` and `bank/w5_run2_20261023T2006Z`); observed and label cells copied verbatim from the instrument. The Q1 predictions in the PREDICTIONS entry were written for the DeepSeek-Flash, prompt-v2 instrument, before the change to the Opus and Sonnet explainers with prompt v3 (checker F5). **Fourth run 2026-10-24T00:18Z** (`metrics.json` sha256 cf696d9e2278eaef6a155e934e19a0a5cb2d0ae380d538133ac800fad76dcad2): after the W4–W5 checker's F14, nine judge rows paired with a neighbouring request were re-answered and `judgments.jsonl` rebuilt (two answers changed); only the `p_u` cell moved, from 0.261 [0.206, 0.316] (run 3) to 0.260 [0.205, 0.315], every label unchanged; run 3 is kept as `bank/w5_run3_20261023T2328Z`. **Fifth run 2026-10-24T01:03Z** (`metrics.json` sha256 b020ea29bd01b1905058494826a579dfc164ca06d8456b7ae022612836e1baf8): after the W4–W5 checker's F13, the 375 judge lines of the 25 chunks whose agents had broken the reading protocol were re-answered under the hardened guard and `judgments.jsonl` rebuilt (24 answers changed); the `p_u` cell moved from 0.260 [0.205, 0.315] (run 4) to 0.262 [0.207, 0.316] with a_ctrl 0.924, and the hard-decoy cell to accuracy 0.697, on flagged 0.351, control 0.798; every label unchanged; run 4 is kept as `bank/w5_run4_20261024T0018Z`. **Sixth run 2026-10-24T02:28Z** (`metrics.json` sha256 002108c3ea15da9f…): after the W4–W5 checker's F15 and the lead's F22 audit, the 200 sample-1 clusterer answers read from excerpts plus five whose read could not be audited were re-answered under the hardened chain and `clusters.jsonl` rebuilt (205 rows changed); the late-born primary set moved from 1,239 to 1,238 clusters (13 entering, judged in the supplementary batch `w4j2`, 78 lines, seed 4113; 14 leaving, their 84 judge lines set aside), and one judge line whose input never reached its agent whole was re-answered with the same verdict. Four cells moved: `p_u` from 0.262 [0.207, 0.316] (run 5) to 0.265 [0.211, 0.321] with A 0.812 over 1,238 clusters; H_MIG from 16 poles (0.027) to 17 (0.028); median r at birth of late-born clusters from 0.2908 to 0.2913; hard-decoy accuracy on flagged clusters from 0.351 to 0.356 (125 flagged); `b`, `v`, the Q1 fractions and the center birth curve are unchanged and every label unchanged; the F2 sensitivity reading moved from 0.272 [0.216, 0.328] to 0.275 [0.220, 0.332]. Run 5 is kept as `bank/w5_run5_20261024T0103Z`. **Seventh run 2026-10-24T04:06Z** (`metrics.json` sha256 a95bfa57bfe65246…): the sixth run's inputs unchanged; every cell, every number and every label equal to run 6 (`runbook/fill_spec11.py`: 0 cells differ); run so that the sample files carry, per pole, the contexts as the explainer saw them and, per judgment, the records as the judge saw them (§10; closeout checker F23 and F29; explain d832b80f). Run 6 is kept as `bank/w5_run6_20261024T0228Z`.

| readout | H_RG predicts | H_SP predicts | H_VEST predicts | observed | label |
|---|---|---|---|---|---|
| `p_u` (§6.3) | upper bound < 0.10 | lower bound > 0.20 | — | p_u = 0.265 [0.211, 0.321]; A = 0.812, a_ctrl = 0.924, 1238 clusters; pole bootstrap conditional on a late-born cluster and a control in each draw, 0 of 10000 draws redrawn (refused above 1%) | H_SP |
| `b` birth order (§6.4) | above the permutation band | inside it | main not earliest in many poles | b = 0.476 [0.436, 0.517]; null mean 0.414, 97.5th 0.449; p = 0.0005 (Holm 0.0010); 574 poles | main oldest (above band) |
| `v` vestigial (§6.5) | low | — | lower bound > 0.5 | v = 0.035 [0.026, 0.044]; p = 1.0000 (Holm 1.0000); 1854 clusters, 0 excluded (growth undefined) | not H_VEST |
| H_MIG count (§6.6) | ≈ 0 | some | — | 17 poles (0.028 of S) | count |
| `r(c)` at birth (exploratory) | median well below 1 | no prediction | — | median r: late-born 0.2912783205072667, all born 0.295 | exploratory |
| hard-decoy accuracy (secondary) | far above 0.5 | near 0.5 on flagged clusters | — | accuracy 0.697; on flagged 0.356; control 0.798 | not measurable |
| Q1 fractions (descriptive) | in the essay's regime with stated thresholds | — | — | G4: ONE 0.563 [0.523, 0.602], TWO-REL 0.060 [0.044, 0.082], UNREL 0.012 [0.006, 0.024], UNK 0.365 [0.327, 0.404], RARE 0.000 [0.000, 0.006] (UNK-N 13, UNK-F 206); O7: ONE 0.518 [0.478, 0.558], TWO-REL 0.060 [0.044, 0.082], UNREL 0.007 [0.003, 0.017], UNK 0.415 [0.376, 0.455], RARE 0.000 [0.000, 0.006] (UNK-N 14, UNK-F 235); Q4B: ONE 0.597 [0.557, 0.635], TWO-REL 0.057 [0.041, 0.078], UNREL 0.013 [0.007, 0.026], UNK 0.320 [0.284, 0.358], RARE 0.013 [0.007, 0.026] (UNK-N 11, UNK-F 181) | descriptive |
| center birth curve (exploratory) | monotone, early for ONE poles | — | — | blind maximum reaches 0.3 of its T_15 score first at median T_3 over 598 poles (ONE T_4 (n = 311); TWO-REL T_6 (n = 36); UNK T_1 (n = 248); UNREL T_6 (n = 4)); positives' mean median T_2; ratio curve non-decreasing (tolerance 0.05) for 0.010 of poles; 1 poles without a positive T_15 score; 1 poles without texts left out; 0 dead pole-checkpoints | exploratory |

## 12. Compute budget (GPU-hours and run counts; API calls)

Lane: `./infra/aws_pg/pg`, on-demand `g6e.2xlarge` (1× L40S 48 GB), profile `<aws-profile>`, region
`us-east-1`, fresh prefix per attempt, `--budget-min` on every run, `pg audit` after every launch. Weights:
Q4B, Q4 and G4 by `--weights`; O7 by in-job download (`--no-weights`).

**W0 amendment: throughput is measured.** The derivation below assumed 15,900 tokens/s for 4B and 8,700
for 7B; W0 measured 3,990–4,023 and 3,291–3,393 on an L40S at 100% SM, 4.0× and 2.6× lower, with the final
array writes (single-threaded `savez_compressed`) at 116–163 s per checkpoint and O7 downloads at 103–126 MB/s.
The budget is re-derived from those rates (HARVEST.md):

| wave | runs | GPU-h each | GPU-h | old estimate |
|---|---|---|---|---|
| W0 (done: three verify attempts, grid, probe, dev, dev probe; 4.1 GPU-h measured) | 7 | | 4.1 | 1.3 |
| W1-D Q4B 50M harvest | 1 | 3.6 | 3.6 | 1.1 |
| W2-D rescoring (discovery texts + shuffle) | 1 | 0.2 | 0.2 | 0.2 |
| W1-C G4 50M; O7 `T_15` 50M | 2 | 3.8; 4.2 | 8.0 | 3.1 |
| W2-C scoring, one per model | 3 | 0.2 | 0.6 | 0.7 |
| W3 O7 16 checkpoints × 20M in two pipeline jobs (download + harvest + write ≈ 1.8 h each) | 2 | 14.3 | 28.5 | 14.4 |
| W3 Q1 scoring step for `S` on `T_15` | 1 | 0.2 | 0.2 | — |
| contingency 30% on the undone waves | | | 12.3 | 6.2 |
| **total** | **17 runs** | | **≈ 57 GPU-h** | 27 |

Derived before W0 (kept for the record): L40S bf16 dense peak 362 TFLOP/s (vendor sheet), assumed
35% utilization → 127 TFLOP/s; 7B at 2 × 7.3e9 FLOP/token → ≈ 8,700 tokens/s; 4B → ≈ 15,900 tokens/s;
hooks and top-k add ≈ 20%. Follow-up before W3: profile the hook cost against the bare forward and thread
the writes (HARVEST.md follow-up 7); either could cut W3 by a third.

API calls (model id `claude-opus-5-5`, or the open-weights backend once its first live call is logged):
Q1: 1,800 poles × (Agent 1 + Agent 2 + Agent 3) = 5,400, plus 120 × 3 × up to three prompt versions on the
discovery set, plus one extra Agent 2 call per job for UNREL candidates (a minority); Q2 with `S` = 600:
clustering 600 × 2 + 40 × 2 = 1,280; judge ≈ 750 clusters × 6 + 640 positive controls × 6 ≈ 8,300. Total
≈ 16,000 calls. W0 measured the explainer at about 5k input tokens per pole (30 contexts of 64 + 16 tokens)
and the judge at about 5k per line; 256-token contexts (§2.1) roughly triple the explainer and clusterer
input. Logged per call. W0 answered every line with Claude Code subagents (manual backend) at zero API cost.
**Measured at W2-D (DeepInfra, Opus 5.5 for the four Q1 roles):** 120 poles cost about $20 (3.9M input, 0.24M output
tokens, 388 calls), $0.17 per pole; W2-C's 1,840 poles project to about $310. The GPU lane gained a second provider:
vast.ai through `infra/vast` (one L40S rental for both W2-D scoring rounds, 12 billed minutes, $0.18).
**API budget record (2026-10-20 17:15Z):** budget $200. Spent ≈ $155 at list prices (checker F19: the dashboard figure at 17:05Z was lower, JOURNAL 2026-10-20 17:05Z, so "dashboard agrees" is withdrawn):
discovery rounds d1–d3 $40; W2-C Opus explainer answers, 1,351 poles, $86; discovery rounds 6–9 on four non-Opus
candidates ≈ $20; smoke tests and one aborted round ≈ $3. The plan as frozen would have cost ≈ $400–450 (audit $314,
judge $10, clusterer $8, discovery $40, exploration $20–40); with Opus in every role ≈ $850. The amended plan (§2.2) costs
≈ $31 more: 489 explainer calls ≈ $8, writer and job writer ≈ $4, second reader on Flash ≈ $3, clusterer and judge on
Flash ≈ $16. GPU is outside this budget (vast credit ≈ $36, AWS).

**Budget record (2026-10-21 19:50Z, after W3, round 13 and the W4 gate; dated post-freeze amendment).** API spend is
read from the provider's dashboard (the `calls.jsonl` logs record ids, models and hashes, not token counts): $176.48 at
01:04Z, when the limit was raised from $200 to $250; ≈ $222 after W2-C's writer and second-reader steps and the W4
clusterer rounds (Flash and Sonnet on S_gate, Flash on S); ≈ $229 after the Sonnet judge gate; ≈ $237 after round 13 and
the G2 check; ≈ $13 remain. W4 on S at the gate's measured rates costs ≈ $49 for clustering (1,200 Sonnet requests) and
≈ $90 for the judge (≈ 9,000 Sonnet lines), so S waits on a top-up; the alternatives are in JOURNAL 2026-10-21T17:37:11Z.
GPU, measured: vast 13 rentals, 30.9 billed GPU-h, $25.60 realized of the $36.00 ceiling, of which the W3 grid 28.9 GPU-h
and $22.68 (the table above estimated 28.5), W2-D rounds 1.0, W2-C G4 scoring 0.4, the T_14 device check 1.9. The AWS
lane ran W0, the three W1 harvests, W2-D rounds 1–5 and the Q4B and O7 W2-C scoring; its forward-pass hours from
`metrics.json` are W1 3.7 + 4.2 + 4.2 and ≈ 0.01 per probe (billed instance hours are not tallied in this record).
Forward-pass hours over every banked run, both lanes: 38.2.
Corrected the same day from the usage logs (JOURNAL 2026-10-21T19:50:09Z): S clustering started and stopped at the limit with 327 of
1,200 answers ($18); measured list prices are $0.053 per clusterer request and $0.0162 per judge line at 18 lines per
pole, so W4 on S needs ≈ $240 more, not the ≈ $139 estimated above.

**Closeout record (2026-10-23).** API: the provider's user-set limit of $250 was reached during S clustering on 2026-10-21
(≈ $237 at list prices through the W4 gate and round 13, then $18.38 for 327 S clusterer answers); after that, 883 clusterer
answers and all 11,034 judge lines on S ran as Claude Code Sonnet subagents through the manual backend at zero API dollars,
and W5 made no API call. GPU: the vast program closed at $30.27 of the $36 ceiling, the 13 rentals above ($25.60) plus the
center-probe run (RTX PRO 6000 WS, 2.31 billed hours at $2.01/h, $4.67; a slow-net host and a killed H100 respawn before
it were unbilled); the AWS lane is unchanged. Forward-pass hours over every banked run: 38.2 before W5, plus the center
probe's 16 checkpoints (a 131,072-token pass and 10,404 text forwards each) inside its 2.31 billed hours. **2026-10-24 addendum (F15, F22):** the same backend answered, at zero API dollars, the 205 sample-1 clusterer re-answers (42 chunks, three attempts discarded by the gate, one answer refused by the echo rule and re-answered), one judge line re-answered for F22 and the 78 supplementary judge lines of batch `w4j2` (15 chunks); the judge record now holds 11,028 lines (10,950 of w4j + 78 of w4j2), `calls.jsonl` 12,902 rows, every call Sonnet 5.5; no rental and no API call this round.

**Backend amendment (2026-10-21 21:40Z, the user's instruction to answer the Sonnet and Opus roles with Claude Code subagents).** W4 on S continues
with Sonnet 5.5 as Claude Code subagents through the manual backend after the provider's user-set limit stopped the API at
327 of 1,200 clusterer answers. The clusterer and judge identities of the passed gate (model `anthropic/claude-sonnet-5-5`,
clusterer prompt v2, judge prompt `11b5fd81…`) are unchanged; the serving differs (Claude Code's `sonnet` alias, its own
sampling and no constrained decoding) and is recorded per answer: manual rows carry `backend: manual`, `collect`/`reject`
run with `--backend mixed`, and `calls.jsonl` logs each row's backend. Validity on S is read from S's own 600 controls
(a_ctrl), as §6.3 already requires; the gate stays the selection record. Answering agents get chunks of requests
(`runbook/manual_chunks.py`) in which no two requests share a pole (clusterer) or a cluster (judge), so the judge never
sees a cluster's true record twice (the leak MANUAL_BACKEND.md's one-line-per-agent rule prevents) **(2026-10-24, checker W4–W5 F17: true for clusters; two clusters of one pole with the same `t_before` carry nearly the same record, 56 such lines in 96 chunks, judge accuracy 0.795 on them against 0.813 overall; future judge chunks group by pole and `t_before`)**; agents read nothing
but their chunk. No Opus role exists in W4, so Opus is not used.

## 13. Literature record (queries and status)

Queries and findings are in `CHECKS.md` rows I0, I10, I11, P6, P8 and the "already clean" critiques row
(all ABSTRACT level, 2026-10-19). Before freeze, read at source for the specific claim, with the quote and
its location recorded: Transluce 2024 neuron-description pipeline (reuse decision for Agent 1's input
format); NeuronScope 2601.03671 (polysemantic-case handling); GLUScope 2602.23826 (the four sign regimes,
to match their definition of the regime split); 2610.21025, 2606.02378, 2605.13329 (to state exactly
what has and has not been done on OLMo/Pythia checkpoints); Stolfo et al. 2406.16254 (entropy and
frequency neurons, for the last-layer suppressor definition); the essay's companion paper if it appears
before freeze (its tests replace ours where they are the same test). 2601.18699 stays DO NOT CITE.

**Status 2026-10-23 (closeout).** The read-at-source list was not completed before the freeze; no source read of these papers
is in the journal before W5. Completed at closeout, after the unblind, by an Opus reader: `checks/literature_W5.md` (quotes with
locations and every URL fetched). Five of the six items were read at source; no companion paper has appeared on arXiv as of
2026-10-23 (queries recorded there). Findings that bear on this text, recorded as dated notes and without changing any frozen
rule (§6.7): (1) §1's "no neuron-level development study" is narrowed, see the correction there (2601.08383); (2) GLUScope's
quadrant definition equals the study's `(sign g, sign u)` split, and its one worked example, a final-layer OLMo neuron whose
gate−/in− quadrant is the interpretable one, contradicts the essay's "double-negative is an echo" as a general rule; the
study's echo rule (§2.4) stands as frozen and this is stated as a limit, one neuron, not a rate; (3) Stolfo et al. define
entropy and frequency neurons by vocabulary-wide output effects found by mediated ablation; the study's single-token
suppressor rule (strongest `W_out·W_U` entry negative and at least twice the second) is the study's own, not theirs, and its
`W_U` is not centered as Stolfo's is; stated as a limit of the rule. Transluce's description step is reused only for
marking contexts with delimiters; NeuronScope clusters explanation sentences and shares no step with this study.

**Status 2026-10-24.** Companion paper searched again at closeout re-verification (web search, general and restricted to arxiv.org): no companion paper on arXiv; the one arXiv hit, 2606.11722 ("Non-Gaussianity Reveals Interpretable Directions in Language Models"), references a repository named `goodfire-already-clean` and is a third-party paper, not the companion; not read, recorded as a lead only. Item 6 of the read-at-source list is conditional on the companion appearing before the freeze; it did not, so nothing replaces the study's tests (§6.7 unchanged). The closeout checker (checks/closeout_W5.md §4, 2026-10-24T03:05Z) looked further: the essay page itself ("a technical companion paper will offer a more detailed account", "our forthcoming technical paper"), Goodfire's research index (newest post 2026-10-15), the arXiv API by both authors named in the essay's language-model section; none is the companion. The reading "closed as not applicable" is the checker's too.

## 14. Pre-mortem and pre-specified responses

| failure | likely cause | response |
|---|---|---|
| `θ_s` on the discovery set > 0.5 | corpus too small; shuffled poles fire on generic text | do not start W1-C; raise the corpus to 100M tokens (one more harvest, ≈ 2 GPU-h per model) and re-set `θ_s`; one retry only, then disclose and continue with the 50M reading |
| judge positive-control accuracy < 0.80 on the development discovery set | prompt, record length, or exemplar leakage | one prompt revision on the discovery set; if still < 0.80, the parent test is not valid; report birth order, vestigial and center birth curve only; say so |
| more than half of clusters born at `T_1` | formation faster than 4.2B tokens | report as a resolution limit of OLMo-3; exploratory Pythia extension (§4.3) as a separate dated amendment |
| Hub download < 50 MB/s or throttled | network | seed the 16 revisions to S3 via `seed_weights.sh`; switch to `--weights` |
| `olmo3` not loadable on the AMI's transformers | pin | build the pinned venv in-job with `matrix_models.ensure_transformers_pin` at a version ≥ 4.57 that imports; log it |
| a W3 pipeline job dies mid-grid | capacity or budget | the pipeline's `pipeline_status.json` names the completed steps; relaunch the remaining steps as a new manifest under a fresh prefix; completed checkpoints are not re-run |
| clustering samples disagree on `m` for most poles | prompt | one revision on the development discovery set; clusters present in only one sample are excluded from the primary count; the exclusion rate is reported |
| Q1 fractions far below the essay's on all three models | our thresholds or our corpus; or the regime differs by model size | report both readings (ours and the essay's rule); the result stands as measured; no threshold moves. **W0 amendment, the order of suspects (§0b item 1), applied before any sentence about the model is written:** the UNK-N/UNK-F split, the explainer's confidence on the failing poles, the context width, the double-negative share, the window-0 share, the corpus size (RARE), and the model size (the essay reports about 70% roughly one job on GPT-2 XL against 90% on Qwen3-32B, essay.md 401; a 4B or 7B base model sits between) are each read and reported as the observer-side account first |
| most discovery poles UNK on the first prompt | explainer inputs or prompt, not the poles | iterate on the discovery set only (§9.2 W2-D rule: three versions at most, every version logged); if the pass rate stays below half, say so and proceed; the confirmation sets never tune anything |
| W0's judge accuracy on parent tests was near 1.0 on 5 clusters | tiny n; or the parent test is easier than the control plan assumes | nothing is read from W0; at W4 the hard-decoy row and the flag rate carry the check that the judge is not separating by domain; if a_ctrl and A are both at 1.0 on `S`, the result is reported as the instrument's ceiling, with `p_u`'s interval |

**Closeout 2026-10-23: which rows fired, and the response taken.** θ_s = 0.3153 (no). Judge positive-control accuracy:
fired at the W4 gate, the Flash judge failed on both clusterings (JOURNAL 2026-10-21T17:29:44Z); the response was a second
gate with Sonnet 5.5 as judge on the same clusters (`explain fork-clusters`), a_ctrl 0.910, pass (17:37:11Z), a judge
change rather than a prompt revision, recorded as a dated amendment; on S, a_ctrl = 0.923. More than half born at T_1:
no (share 0.0004). Hub download: no (198–234 MB/s). `olmo3` loading: no. A W3 job dying mid-grid: job b's first host
failed the network probe before any step and was respawned; no completed checkpoint was re-run. Clustering samples
disagreeing on `m`: B4 m-within-1 is 0.668; the response taken before S was the clusterer prompt v2 on S_gate
(2026-10-21 17:45Z); clusters present in one sample only are excluded (108 of 2,454 clusters, unmatched share 0.044, reported; 52 of them otherwise late-born primary, status `unmatched_in_sample_1`). Q1
fractions far below the essay's: fired on all three models (ONE 0.52–0.60 against 0.90); both readings are reported
(`metrics.json q1.<cell>.essay_rule`), no threshold moved. Most discovery poles UNK on the first prompt: fired (d1 pass
22.7%); three prompt versions at most were used (v1–v3, every version logged), pass 61.9% at v3; the confirmation sets
tuned nothing. W0's judge near 1.0: on S, a_ctrl 0.923 and A 0.813, not both at 1.0; the hard-decoy row is not
measurable (control 0.796) and the flag rate is 0.099, both reported. **2026-10-24 addendum (run 6):** on the rebuilt inputs the same rows stand: a_ctrl = 0.924 and A = 0.812; clusters present in one sample only are 116 of 2,454 (unmatched share 0.047, reported), 53 of them otherwise late-born primary (status `unmatched_in_sample_1`); the hard-decoy row stays not measurable (control 0.798) and the flag rate is 0.101 (125 of 1,238); B4 m-within-1 0.678 (closeout: 0.668); no row fired that had not fired at closeout.

## 15. Open before freeze (spec defects today)

Status after W0 (2026-10-19 evening): items 3, 6, 7 and 8 are closed below; items 1, 2, 4 and 5 stay open;
items 9–14 are new.
**Status 2026-10-24 (closeout checker F35):** every item is closed with a dated line below; nothing is open.

1. The Anthropic API key for Agents 1–3 and the judge: N3 runs where the key is. Default: on the laptop
   as a control-plane script (API calls only, no models), reading `<api-key-env-var>` from the user's
   environment. Alternative: a CPU instance with a new SSM parameter beside `<ssm-parameter>`. The
   user decides; nothing is printed. **W0:** until a key is present, every line is answered by Claude Code
   subagents through the manual backend (`neurons_prompts/MANUAL_BACKEND.md`), which answered 300 lines at W0
   with zero rejections; at W2-C and W4 scale (≈ 2,200 writer lines, ≈ 8,300 judge lines) this needs either the
   key or a workflow runner, and the first live call per backend is still to be logged. **Closed at W1 by item 13** (the DeepInfra key, outside every repository); the manual backend's audit of 2026-10-22 to 2026-10-24 is in JOURNAL and `handoff/judge_incident_log.txt`.
9. **W0:** the corpus rule changes of §4.2 (dedupe by text hash, window position 0 excluded and banked, the
   evenly spaced directory rule, both shard formats, the next-file rule) are implemented on `neurons/harvest`
   before W1-D and their effect on the W0 numbers is noted in `JOURNAL.md`. **Closed at W1:** harvest 504fd4f5 (dedupe, window-0 exclusion with `pos0_share`), JOURNAL 2026-10-20.
10. **W0:** when `--sample-neurons` is given without `--poles`, the harvest writes `poles/` (vocab, top-1000,
    its own exemplar record) for the sampled poles, so a final-checkpoint audit harvest is one job. **Closed at W1:** every W1 harvest's `out/<cell>/poles/` exists (harvest 504fd4f5 "drawn poles get `poles/`").
11. **W0:** explainer prompt v2 (§2.2), 256-token contexts for the explainer and clusterer (§2.1), the UNK-N /
    UNK-F split (§2.3), X3 and X7 (§0c) on `neurons/explain` before W2-D; every prompt version's SHA-256 logged. **Closed at W1/W2-D:** explain d281852c (prompt v2, 256-token contexts, UNK-N/UNK-F, X3, X7), JOURNAL 2026-10-20;
    every prompt version's SHA-256 is in the journal and the explain manifests.
12. **W0:** a second registered failing input for G0 with power at early checkpoints (§5). **Closed at W1:** harvest 504fd4f5, second G0 failing input (u×1.1), JOURNAL 2026-10-20T03:15Z entry (re-check F37: the
    result is in that entry, not in `verify/VERIFY.md`, whose G0 row lists only the activation swap).
13. **Closed at W1:** a DeepInfra key arrived (kept outside every repository, mode 600); its OpenAI-compatible
    endpoint serves the open-weights models and the Anthropic models, so item 1's key question is closed too:
    N3 runs on the laptop as a control-plane script reading `<api-key-env-var>` from the environment.
14. **W0:** `S` = 600 (§4.4) raises W3's `--poles` list to ≈ 640 and the judge plan to ≈ 8,300 lines; the
    W4 session plan (§9.4) is one session per 2,000 judge lines with the backend of item 1. **Closed at W4:** the judge plan came to 11,034 lines (two random draws and the hard decoy, two samples, plus 600 controls);
    all answered on the manual backend across sessions S5 and S6 (judge calls by day: 9,166 on 2026-10-22, 1,868 on 2026-10-23,
    463 on 2026-10-24; re-check F37), the hardened chain from 2026-10-22T10:29Z; §12 holds the counts.
2. Read-at-source list in §13 completed and recorded. **Closed at closeout 2026-10-23, after the freeze:** `checks/literature_W5.md`; §13 status paragraph.
3. **Closed at W0:** throughput measured (§9.1); §12 re-derived (≈ 57 GPU-h).
4. Whether the judge should be a different model family from the explainer (principle 9). Default: same
   model id, different role and prompt; the critic at closeout is a different model. **Closed 2026-10-24 by the record:** the judge is Sonnet 5.5 (chosen at the W4 gate after the Flash judge failed,
   JOURNAL 2026-10-21T17:37Z); the explainers were Opus (most of terciles 0–1) and Sonnet (tercile 2) with the same prompt
   role split; the critics at closeout (checkers W1–W3, W4–W5, closeout) ran as Opus. The default was kept for the judge's
   role and prompt; the family question was settled by the gate, not by a rule.
5. `checks/CHECKERS.md` written. **Closed 2026-10-21T17:18Z, after W3** (re-check F37; not at W1 as first written): `checks/CHECKERS.md`, drafts f87f1b50.
6. **Closed at W0:** G1 names `Qwen/Qwen3-4B-Instruct-2507` (`cdbee75f…`), the model the banked value came
   from, and reproduced it bit for bit (§4.1, §5). `matrix_repo_facts.json` still lacks `Qwen/Qwen3-4B`; it
   is no longer needed by this study.
7. **Closed at W0, decided for power:** `S` = 600 poles (§4.4, §7). The simulation (branch `neurons/explain`,
   `test_neurons_verdict.py`): with `a_ctrl = 0.95` and a true `p_u = 0.05`, at `n = 400` H_RG fires in 139 of
   200 simulated datasets (70%); at 600 near 90%. The grid harvest cost does not change with the pole count;
   the judge plan grows to ≈ 8,300 lines (§12).
8. **Closed at W0:** `claude-opus-5-5` rejects `temperature`; Agent 3 is an independent call at the model's
   default sampling, and `neurons_llm.py` refuses any other temperature for that backend. Open-weights backends
   keep temperature 1.0.
15. **Found at closeout re-verification 2026-10-24, closed the same day:** §10 names `events.jsonl` and `run.log` per run, and no job
    ever wrote files of those names (the AWS lane wrote `job.log` and `STATUS`, the vast lane `vast/job.log`, `vast/timings.tsv`,
    `vast/_rental_complete.json`, both lanes `out/pipeline_status.json` and per-stage `manifest.json` with `started_utc`,
    `finished_utc`, `timing_s`), a spec defect this list should have carried. Closed by `runbook/write_run_logs.py`: beside every
    harvest run in S3 (and in the harvest bank where the run is banked) `run.log` is the job's `job.log` copied verbatim and
    `events.jsonl` holds the lifecycle events read from those files, its first line saying it is derived and naming the sources;
    nothing is invented and the originals stay. Scope (closeout checker F28): the GPU harvest and probe runs, 26 prefixes; the
    explain, clusterer, judge and verdict steps are not runs of that kind and keep their per-call record in each explain
    directory's `calls.jsonl` (utc, model, backend, response hash) and the W5 unblind in `metrics.json` (`created_utc`,
    `inputs`); the corpus stage's compute is in the events since the 03:17Z regeneration (its time read from the corpus manifest's
    `created_utc` since the 03:5xZ regeneration, re-check F38). JOURNAL 2026-10-24.
16. **§13 item 6 (the essay's companion paper), closed 2026-10-24 as not applicable:** the item was conditional on the paper
    appearing before the freeze; it did not (searched 2026-10-23 and 2026-10-24, general and arxiv.org-restricted). Should it appear
    later, comparing its same-test rows with this study's is a dated amendment after closeout and changes no frozen rule (§6.7).
17. **Found 2026-10-24 after closeout, F41, closed 2026-10-25 by the code fix and this disclosure:** the explainer and second-reader prompts showed every negative pole the positive
    pole's output-vocabulary line (the harvest file is per pole and already flipped; the explain code flipped it again). W2-D, W2-C
    and the T_0 control are affected in that one line; the clusterer and judge prompts carry no such line. ONE shares by sign
    are equal within noise in every depth third of every model; the casualty is the wording of negative-pole descriptions
    about promoted or suppressed words. Code fixed in `neurons/explain`; no wave re-run. JOURNAL 2026-10-25T03:50Z.
18. **Found 2026-10-25 after closeout, F42, closed the same day as a disclosed harvest limitation:** the W0 exclusion of window position 0 (§4.2) left G4's first-token pile one
    token to the right in layers 10 and 11: 25 windows whose position 0 is "\r" and position 1 is "\n" hold 3.44% and 5.19% of
    those layers' top-50 slots and dominate 1,391 and 2,109 poles. Three sampled G4 poles are affected (L10N1890-, L10N4243-,
    L11N9800-, 25 of 50 top positions each), all UNK, so no §11 count moves. A harvest limitation, disclosed, not re-run.
    JOURNAL 2026-10-25T04:15Z.


## 16. W0 verification record (2026-10-19), facts only

Full record: `verify/VERIFY.md`; harvest side `verify/HARVEST.md`; explain side `verify/EXPLAIN.md`,
`verify/REVIEW_EXPLAIN.md`. Nothing here is a readout; `S` was 4 poles and the Q1 sets 20 and 4.

- Instrument: explain → write → probe discriminates. Shuffled-pole scores on 20 O7 poles had median 0.05 and
  99th percentile 0.30 (`θ_s` = 0.2989); 12 of 20 first-pass explanations passed §6.1; on the 4-pole
  development set positives fired 11/20, near misses 3/20, distant 0/20, shuffled 0/60. Explainer confidence
  0.50 on passing poles against 0.34 on failing.
- Judge: 47 of 48 control trials correct across two directories, hard decoys included; a_ctrl 1.00.
- Observations on the model: both poles of a neuron carried distinct readable jobs; the four grid poles'
  main clusters first exist at steps 100k–560k of 1.41M; step-1000 top contexts carry a median 0.0104 of the
  final pole's max; Q4B has ≈ 5,500 one-signed poles in layers 1–3, OLMo none; G4 puts 1.5% of top-k slots on
  window position 0; 34% of the W0 corpus's characters were duplicate documents.
- Gates: G0 0.0034–0.0065 everywhere; G1 2.244386524302183 on C0; every N4 refusal fired when it should
  (θ_s timing, judge-gate timing, degenerate bootstrap).
- Cost: 4.1 GPU-h over seven launches; 300 manual-backend lines answered with zero rejections.
