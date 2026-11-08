# Literature record W5: read at source for SPEC_D1 §13

Date: 2026-10-23. Each page was fetched with curl (arXiv abstract and HTML pages, transluce.org, the arXiv API for
search), converted to text, and quoted from that text. Text in quotation marks is verbatim; where a quote contains
math that arXiv renders as MathML, the math is transcribed from the MathML alt text and marked as such. Nothing
below comes from memory. 2601.18699 was neither fetched nor cited. The essay page (already-clean.goodfire.com)
was not fetched, because it is outside the sites allowed for this task.

Study lines referred to below were read this session: SPEC_D1.md line 96 (`g = W_gate x`, `u = W_up x`),
145–148 (regime split of the top-50), 213–216 (Agent 1's inputs), 265–266 (single-token suppressor rule),
133–135 (§1 sentence "No neuron-level development study exists on OLMo or Pythia"); DATA_CONTRACT.md line 95
(`W_U @ (gamma_final ⊙ W_down[:, i])`, final norm gain folded in) and line 220 (contexts rendered 256 tokens
before, 16 after); verify/EXPLAIN.md line 46 (`render` prints a pole's top-n contexts with the marked token).

---

## 1. Transluce 2024, "Scaling Automatic Neuron Description" (Choi, Huang, Meng, Johnson, Steinhardt, Schwettmann)

- **What §13 needs:** the input format given to the describing model, for the reuse decision on Agent 1's input
  format (SPEC §3 N3; CHECKS I0).
- **Read at source:** yes. https://transluce.org/neuron-descriptions ("Published: October 23, 2024"). The
  collapsed "Expand to see sample prompt" box is not in the static HTML; its text was read from the page's own
  script bundle https://transluce.org/_next/static/chunks/pages/neuron-descriptions-08fa9dff1210b18b.js.
- **Quotes and locations:**
  - Section "Describing exemplars": "Prior work [9] represented each exemplar as a list of tab-separated (token,
    activation) pairs." Then: "We found that this paired format is difficult for the explainer to ingest, and
    instead replace per-token activation values with inline delimiters ({{}}) to highlight activations above a
    certain quantile, similarly to [21]."
  - Same section, footnote: "quantiles are determined dynamically so that a minimum of three highlighted tokens
    per prompt are shown to the explainer."
  - Same section, "Best-of-k sampling": "varying the number of exemplars (10-20) included in the prompt"; "Few-shot
    examples": "we sample one few-shot example from the 19 neuron puzzles".
  - Section "Generating exemplars": "Since activations can be both positive and negative, we construct separate
    exemplar sets [E+ and E−, math] for the most-positive and most-negative exemplars for each feature." Length:
    "We sample chat interactions of length 95 tokens"; Fineweb "Text sequences ... have a length of 65 contiguous
    tokens", each prefixed with the 30-token Llama chat prefix.
  - Sample prompt (bundle): "You will receive a list of text excerpts on which the neuron activates. Tokens
    causing activation will appear between delimiters like {{this}}." The few-shot answer is a single line,
    "[DESCRIPTION]: phrases in a passage that indicate a character has died".
  - Section "Limitations": "Our descriptions were scored on mostly top-activating exemplars which means that they
    are not representative of the full behavior of the neuron across all quantiles."
  - Figure 8 caption: "Using chain-of-thought prompting in the description-generating prompt results in worse
    scores."
- **What the explainer sees (from the quotes):** 10–20 whole excerpts of about 95 tokens, every token above a
  per-prompt quantile wrapped in {{ }}, no activation numbers, no statistics, no output-weight or vocabulary
  projection, no sign-regime information, one polarity per prompt, one few-shot example. It returns one
  description, with no job count and no class label.
- **What it means for the study:** Transluce's format covers one part of Agent 1's input, the marking of
  activating tokens with delimiters and the separate description of the positive and negative poles, which the
  study already does (one ⟦ ⟧-marked token per context in a 256-before/16-after window, not every above-quantile
  token in a 95-token excerpt); Agent 1's `stats/`, regime split, `W_out·W_U` vocabulary and job-count/class outputs
  have no counterpart in Transluce and are the study's own additions. Transluce's one measured finding on format
  (per-token numbers were harder for the explainer than delimiters) bears on Agent 1 only if its contexts also
  carry per-token values, which was not checked here.

## 2. NeuronScope, arXiv 2601.03671 (Liu, Miao, Zhao, Liu, Du; submitted 7 Jan 2026)

- **What §13 needs:** how polysemantic cases are handled.
- **Read at source:** yes. https://arxiv.org/abs/2601.03671 and https://arxiv.org/html/2601.03671v1 (§1–§5,
  Limitations, Appendices A–H).
- **Quotes and locations:**
  - §2.2: "Rather than prematurely collapsing behavior into one dominant concept, the proposer deliberately
    retains all plausible cues to support downstream decomposition and grouping."
  - §2.3: "Each cluster represents a unique semantic facet of the neuron. For instance, a single neuron may be
    resolved into distinct clusters corresponding to contextual causation and prepositional usage, which are
    subsequently treated as separate interpretation targets."
  - Appendix B: "The Clustering Agent employed text-embedding-3-small OpenAI et al. (2024) to compute explanation
    embeddings and applied HDBSCAN McInnes et al. (2017) with a minimum cluster size of 2."
  - Appendix F, Decomposition Agent, rule 6: "If there is only one semantic meaning, return only one sentence."
  - §3.1: "For semantic coverage, we use Number measures how many distinct semantic concepts are associated with a
    neuron." Table 1 reports mean Number 4.48–5.53 (Llama-3.1-8B), 2.22–2.49 (Gemma-2-2B), 2.26–2.42 (Qwen3-4B).
  - Appendix B: "For each selected layer, we analyzed 200 neurons with the highest activation frequency across the
    corpus."
  - Also read: the Hypothesis Agent prompt (Appendix F) opens with the same two sentences as Transluce's sample
    prompt ("You are a meticulous AI researcher conducting an important investigation into a specific neuron
    inside a language model that activates in response to text excerpts. ..."), and Appendix H's only baseline
    follows "Scaling Automatic Neuron Description". The Limitations section says the method "relies on
    activation-based clustering", while Appendix B clusters explanation embeddings; the paper is inconsistent on
    this point.
- **What it means for the study:** NeuronScope splits one recall-oriented explanation into sentences, clusters the
  sentences' text embeddings and refines one explanation per cluster; it never clusters the contexts, never asks
  whether two modes are related, and only high positive activations are used. Its "Number" therefore counts
  related facets as separate modes (Appendix G's final explanations [3] "the '#' character specifically marking the
  start of a comment within Python code" and [6] "the initial character of a new line that marks the commencement
  of a comment" are two of six), so its counts are not comparable to TWO-REL/UNREL. No NeuronScope step is the same
  step as the study's clustering role (top-50 contexts grouped, pairs marked related or unrelated), and the
  IDEAS.md plan to reuse NeuronScope "for polysemantic cases" has nothing to reuse beyond the Transluce preamble
  the two share.

## 3. GLUScope, arXiv 2602.23826 (Gerstner, Veloso, Schütze; v1 27 Feb 2026, v2 21 Oct 2026)

- **What §13 needs:** the four sign regimes, to match their definition of the regime split.
- **Read at source:** yes, v2. https://arxiv.org/abs/2602.23826 and https://arxiv.org/html/2602.23826v2 (§1–§6,
  Limitations). The abstract page's title is "GLUScope: A Tool for Analyzing GLU Neurons in Transformer Language
  Models"; the v2 HTML title reads "A Resource for Analyzing GLU Neurons ...".
- **Quotes and locations:**
  - §2 (math transcribed from MathML): "SwiGLU(x_gate, x_in) := Swish(x_gate) · x_in, where x_gate := ⟨w_gate, x⟩
    and x_in := ⟨w_in, x⟩". Footnote 6: "In other works, W_in and W_out are also called W_up and W_down,
    respectively."
  - §2.1: "For a given neuron the x_gate and x_in values can take four possible sign combinations, which we call
    activation quadrants. We denote these quadrants as gate+_in+ (for x_gate>0 and x_in>0), gate+_in-, etc."
  - §2.1: "For each neuron, we record not only the strongest activations overall, but the strongest activations
    within each activation quadrant."
  - §4.2: "we count how often each activation quadrant (Section 2.1) appears", with a top-k of "k=16" kept per
    quadrant.
  - §5.2, OLMo-7B-0424 neuron 31.9634 (final layer): "The gate-_in- activations are highly interpretable." The
    quadrant "only appears 17.34% of the time", and "The highly interpretable gate-_in- activations are much
    smaller (at most 1.58) and would therefore have been missed." The other quadrants' extrema are given as 3.48
    and −5.12.
  - §3.2, on Transluce: "Therefore they have separate pages for the positive and negative activations of each
    neuron. We go one step further and consider four different activation quadrants for each neuron."
- **What it means for the study:** the study's split by `(sign g, sign u)` with `g = W_gate x` and `u = W_up x` as
  pre-activations is GLUScope's quadrant definition exactly (W_in is W_up), so the regimes map one to one and the
  double negative is gate-_in-. Two differences have to be stated: the study splits only a pole's top-50 by |a|,
  while GLUScope counts quadrant frequency over all tokens and keeps a top-16 inside each quadrant. And GLUScope's
  one worked case, a final-layer OLMo neuron whose gate-_in- quadrant is the interpretable one and small in
  magnitude, contradicts as a general rule the essay's "the double-negative regime is an echo of the other pole",
  which the study gives Agent 1 and uses to drop `g < 0, u < 0` contexts from the parent test. This is one neuron,
  not a rate. Inference, not measured: a top-50 by |a| inside the positive pole of such a neuron would hold mostly
  gate+_in+ entries. CHECKS P8's "documented before the essay" holds, since v1 is dated 27 Feb 2026.

## 4. What has and has not been done on OLMo/Pythia training checkpoints

### 4a. arXiv 2610.21025, "Learning to Read Out: Unembedding Dynamics in Language Model Pretraining" (He, Shen, Iacob, Jovanovic, Qiu, Lane; 14 Oct 2026)

- **Read at source:** yes. https://arxiv.org/abs/2610.21025 and https://arxiv.org/html/2610.21025v1 (§1–§6, B.2,
  D.3, I.2 read; the rest scanned).
- **Has done:** §3: "We study readout development in Pythia-160M, 1B, and 6.9B (Biderman et al., 2023) and in
  OLMo-2-7B, which differs from Pythia in architecture, tokenizer, and training recipe (Team OLMo et al., 2025).
  The dictionaries for all four models cover the full vocabulary across 32 public checkpoints (Table C.1)." §2.1:
  "We fit the dictionary to readout parameters rather than activations, so it needs only checkpoint weights,
  without labels or an activation corpus." It also runs linear probes on the final hidden state and readout swaps
  (Pythia-1B and 6.9B) and pretrains four 31M models. The OLMo-2-7B checkpoints are 32 stage-1 revisions from step
  150 to step 928000 (B.2).
- **Has not done:** any MLP-neuron analysis (the words "neuron" and "MLP" occur 0 times in the HTML text), and
  OLMo-3. From I.2: "OLMo-2-7B is the only cross-family check". From Future work: "measuring weight drift in every
  layer (Figure C.6) would show whether each component, like the readout, settles on its own schedule."
- **What it means for the study:** this is a crosscoder on unembedding rows; it does not touch MLP neurons, and its
  0.4–1.1% reorganization window is a readout timing, not a neuron timing.

### 4b. arXiv 2606.02378, "When Do Attention Circuits Form? ..." (Xu; v2 9 Jun 2026)

- **Read at source:** yes. https://arxiv.org/abs/2606.02378 and https://arxiv.org/html/2606.02378v2 (§1, §2, §4,
  §12, §13).
- **Has done:** abstract: "we apply a participation-ratio (PR) spectral signal and an all-head capability-specific
  selectivity screen to track induction, previous-token, and BOS-attractor heads as they emerge." Models: "Pythia
  1B ..., OLMo 1B-0724-hf ..., and OLMoE 1B-7B-0924", at 10 log-spaced revisions each; the earliest OLMo revision
  is step1000, about 2B tokens. §12: "We sampled 8 OLMoE revisions rather than 10 for this reason", against the
  abstract's 30 runs.
- **Has not done:** MLP neurons ("neuron" and "MLP" occur 0 times), OLMo-3, or any 7B dense model. §1, Position:
  "This paper is the activation-space half of a broader study of emergence in pretraining."
- **What it means for the study:** it covers attention heads only, at 1B scale, and nothing at neuron level.

### 4c. arXiv 2605.13329, "Tracing Persona Vectors Through LLM Pretraining" (Moskvoretskii, Glandorf, Medina Moreira, Käser, West; 13 May 2026)

- **Read at source:** yes. https://arxiv.org/abs/2605.13329 and https://arxiv.org/html/2605.13329v1 (§2, §3,
  App. C, F.6).
- **Has done:** §2.2: "We run all experiments on OLMo-3-1025-7B (Olmo et al., 2026)", at 17 pretraining
  checkpoints. F.6 lists "4.2B, 8.4B, 13B, 21B, 29B, 38B, 42B, 63B, 84B, 126B, 210B, 415B, 1.2T, 3.0T, 5.9T, 6T and
  6.1T tokens". §3: "Persona vectors form very early, roughly at 0.22% of total training tokens." Method, §2.1: "we
  record the mean residual-stream activation over the generated tokens at layer l, and take the difference of the
  resulting per-persona means". It is replicated on Apertus-8B.
- **Has not done:** anything with neurons ("neuron" and "MLP" occur 0 times). App. C: "the vectors should be viewed
  as a representation-level summary rather than a complete account of the underlying mechanism".
- **What it means for the study:** it uses the same model, with the same 4.2B-token first checkpoint as the study's
  grid, but reads residual-stream directions at layers 16 and 20, not neurons. By its own statement, its onset is a
  lower bound set by how fluently early checkpoints generate text.

### 4d. Found while checking item 4 (not on §13's list): arXiv 2601.08383

- "Deconstructing Pre-training: Knowledge Attribution Analysis in MoE and Dense Models", v1, 13 Jan 2026. **Read
  partially at source** (abstract, Introduction, Experiment Settings, "Neuron-Level Dynamics", Related Work,
  Conclusion): https://arxiv.org/abs/2601.08383, https://arxiv.org/html/2601.08383.
- Introduction: "In this paper, we present a neuron-level, time-resolved comparison of MoE and dense Transformers
  throughout pre-training to fill this gap." Its models are OLMo-7B (Groeneveld et al. 2024; 600K steps, 2.5T
  tokens) and OLMoE-1B-7B. It ranks FFN and attention neurons by a gated log-probability-increase attribution on 906
  relational-fact prompts and tracks the Jaccard overlap of the top 1% between checkpoints: "OLMo never develops a
  durable core. Its Jaccard overlap peaks briefly at 10.2% around 70K steps, but then declines to 4.9% by 600K."
- **What it means for the study:** this contradicts, as written, the SPEC §1 sentence "No neuron-level development
  study exists on OLMo or Pythia" and CHECKS I10's "no neuron-level development study found on OLMo or Pythia".
  The papers read here support a narrower claim: no study has explained neurons (jobs, clusters, related or
  unrelated, births or branching) across OLMo or Pythia checkpoints. 2601.08383 tracks importance ranks on one task
  family, not what the neurons do.
- **Checked and set aside (abstract level):** 2609.29328 ("seven checkpoints" there are seven models, Qwen3 0.6B–8B,
  Pythia-410M, TinyLlama-1.1B and Llama-3.1-8B, not training steps; confirmed in its HTML); 2610.09600 (refusal
  circuits across alignment checkpoints); 2606.03990 (neuron populations across model scale). Leads not read, all
  described as feature-level in 2610.21025 §5 or in their abstracts: Bayazit, Mueller, Bosselut 2026 (ACL)
  "Crosscoding through time"; Ge et al. 2026 (ICLR, openreview qKmXz1UR0Z) "Evolution of concepts in language
  model pre-training"; Sawmya, Adler, Shavit, arXiv 2505.19440 (abstract read: sparse autoencoders across training
  checkpoints).

## 5. Stolfo et al., arXiv 2406.16254, "Confidence Regulation Neurons in Language Models" (v2, 8 Nov 2024)

- **What §13 needs:** the definitions of entropy neurons and token-frequency neurons, for the last-layer
  suppressor definition.
- **Read at source:** yes. https://arxiv.org/abs/2406.16254 and https://arxiv.org/html/2406.16254v2 (§1–§5, §8,
  Appendices A, C, G).
- **Quotes and locations:**
  - Abstract: "Entropy neurons are characterized by an unusually high weight norm and influence the final layer
    normalization (LayerNorm) scale to effectively scale down the logits."
  - Abstract: token frequency neurons "boost or suppress each token's logit proportionally to its log frequency,
    thereby shifting the output distribution towards or away from the unigram distribution."
  - §3.1, identification: "We do this by searching for neurons with a high weight norm and a minimal impact on the
    logits." Eq. 3 (MathML): LogitVar(w_out) = Var(W_U w_out / (‖W_U‖_dim=1 ‖w_out‖)).
  - §3.3: "all the entropy neurons write almost exclusively to directions within the effective null space."
  - §3.4: "As entropy neurons act by re-scaling the residual stream and have no effect that is specific to a subset
    of the tokens, their activation has a large effect on the entropy of the output distribution but little impact
    on the relative ranking of the tokens."
  - §4 (MathML): the frequency direction is v_freq,i = log(p_freq,i) − m(log(p_freq)). Token frequency neurons are
    found by mean ablation with the logit component along v_freq held fixed (TE against DE_freq, Eqs. 7–8).
  - §5, Pythia-410M neuron 23.417: "the projection of this neurons' w_out onto the frequency direction v_freq is
    negative, suggesting that this neuron is suppressing common tokens and promoting rare ones."
  - Footnote 3: "our study focuses on the final-layer MLPs due to their direct path to the final LayerNorm and
    unembedding".
  - Appendix C: "Since softmax is translation invariant, we also center W_U".
  - Appendix A: "in Pythia and Gemma, this fraction is around 40%, which is significantly less than the 80%
    observed in GPT-2, LLaMA2 7B, and Phi-2."
- **What it means for the study:** both classes are final-layer neurons defined by output effects across the whole
  vocabulary (writes into the near-null space that act through the final norm; a push along the log-frequency
  direction), and both are found by mediated ablation. Neither is a "single-token suppressor", so the study's rule
  (strongest `W_out·W_U` entry negative and at least twice the second) does not come from Stolfo and should say so.
  Stolfo is the reference for what that rule has to exclude. Two points for the rule are inferences, not
  measured. First, an entropy neuron's near-flat logit vector and a frequency neuron's vocabulary-wide push are
  unlikely to pass a 2× top-entry test. Second, Stolfo centers W_U because softmax ignores a constant shift; the
  study's `W_U @ (gamma_final ⊙ W_down[:, i])` states the final-norm fold but not centering, and a per-neuron
  constant added to every entry changes both the sign and the first-to-second ratio the rule tests.

## 6. The essay's companion paper

- **Result: none found.**
- **What was established:** according to search-result summaries (secondary coverage, not fetched), the essay is by
  Nick Cammarata (Goodfire), dated 15 October 2026, and its technical companion is described as forthcoming.
- **Queries run on 2026-10-23:**
  - WebSearch (extended): `"Neural Networks Are Already Clean" Goodfire paper arXiv`. Result: no arXiv listing,
    coverage only.
  - WebSearch (extended): `Cammarata Goodfire "already clean" companion paper arXiv neurons`. Result: no arXiv
    listing; coverage says the companion is pending.
  - WebSearch (standard): `arxiv 2610 neurons "already clean" legibility training checkpoints Goodfire technical
    report`. Result: none.
  - arXiv API (`http://export.arxiv.org/api/query`, sorted by submission date, newest first). A `cat:cs.LG` query
    returned submissions dated 2026-10-22, which shows the index was current to that date:
    - `au:Cammarata`: 45 results, none on neural-network neurons (physics, hardware, one vision paper).
    - `au:"Nick Cammarata"`: 1 result, 2606.25234 (23 Jun 2026, "Structuring Sparsity: Block-Sparse Featurizers
      Capture Visual Concept Manifolds"; abstract read: InceptionV1, DINOv3, SDXL). It is not a neuron-legibility
      or training-checkpoint paper, and it predates the essay.
    - `au:Cammarata_N`: 0.
    - `abs:"already clean"`: 2, both unrelated. `ti:"already clean"`: 0.
    - `abs:Goodfire`: 2, both unrelated.
    - `abs:"superposition" AND abs:"neurons" AND abs:"polysemantic"`: 16, none by the essay's author.
    - `abs:neuron AND abs:legible`: 6, none by the author.
    - `abs:neurons AND abs:"already" AND abs:"clean"`: 1, unrelated.
    - `ti:neurons AND abs:"blind" AND abs:explanation`: 1, unrelated.
    - `abs:"neuron explanations" AND abs:"training"`: 0.
    - `abs:"Qwen3-32B" AND abs:neurons`: 0. `abs:"neuron poles"`: 0.
    - `abs:monosemantic AND abs:neurons AND abs:pretraining`: 6, none by the author.
    - `abs:neurons AND abs:interpretable AND abs:checkpoints`: 12, none by the author.
- **What it means for the study:** as of 2026-10-23, no companion paper is on arXiv under the author's name or the
  essay's terms. None of the study's tests is replaced, so §13's rule (their tests replace ours where they are the
  same test) does not yet apply.

---

## Summary

| item | read at source | claim |
|---|---|---|
| 1 Transluce 2024 input format | yes (page, plus page bundle for the sample prompt) | supported, narrowed: CHECKS I0's reuse of Transluce for the description step holds for marking contexts with delimiters; the rest of Agent 1's input is not in Transluce |
| 2 NeuronScope 2601.03671 | yes | addressed; reuse not supported: it clusters explanation sentences, has no related/unrelated test, and no step is the same as the study's |
| 3 GLUScope 2602.23826 | yes (v2) | supported: the quadrant definition equals the study's `(sign g, sign u)`; its one worked example contradicts "double negative = echo" for a final-layer OLMo neuron |
| 4 2610.21025, 2606.02378, 2605.13329 | yes, all three | supported for these three (none is neuron-level); the broader SPEC §1 / CHECKS I10 sentence is contradicted by 2601.08383 (partial read) |
| 5 Stolfo 2406.16254 | yes | supported for the definitions as CHECKS gives them; the single-token suppressor rule is not addressed by the paper and is the study's own |
| 6 companion paper | no paper to read | none found |

Read at source: 5 of 6 items (item 6: none found).

## URLs fetched

- https://transluce.org/neuron-descriptions
- https://transluce.org/_next/static/chunks/pages/neuron-descriptions-08fa9dff1210b18b.js
- https://arxiv.org/abs/2601.03671 and https://arxiv.org/html/2601.03671v1
- https://arxiv.org/abs/2602.23826 and https://arxiv.org/html/2602.23826v2
- https://arxiv.org/abs/2610.21025 and https://arxiv.org/html/2610.21025v1
- https://arxiv.org/abs/2606.02378 and https://arxiv.org/html/2606.02378v2
- https://arxiv.org/abs/2605.13329 and https://arxiv.org/html/2605.13329v1
- https://arxiv.org/abs/2406.16254 and https://arxiv.org/html/2406.16254v2
- https://arxiv.org/abs/2601.08383 and https://arxiv.org/html/2601.08383
- https://arxiv.org/abs/2609.29328 and https://arxiv.org/html/2609.29328
- https://arxiv.org/abs/2610.09600, https://arxiv.org/abs/2505.19440, https://arxiv.org/abs/2606.03990,
  https://arxiv.org/abs/2606.25234
- http://export.arxiv.org/api/query with each query listed in item 6, plus `cat:cs.LG` for index freshness
- WebSearch result pages were not fetched; only the search summaries were used, and only in item 6.
