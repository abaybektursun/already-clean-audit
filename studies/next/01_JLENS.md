# 01 — The Jacobian lens and the J-space: a distillation for experiment design

Read on 2026-11-04. Paper: Gurnee, Sofroniew, Pearce, Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong, Wang,
Thompson, Abrahams, Kantamneni, Ameisen, Batson, Lindsey (Anthropic), **"Verbalizable Representations Form a
Global Workspace in Language Models"**.

- HTML (Transformer Circuits Thread, published 6 Jul 2026): https://transformer-circuits.pub/2026/workspace/index.html
- arXiv 2607.15495v1 (submitted 16 Jul 2026, 117 pages): https://arxiv.org/abs/2607.15495
- Anthropic research page, "A global workspace in language models" (6 Jul 2026): https://www.anthropic.com/research/global-workspace
- Code: https://github.com/anthropics/jacobian-lens (read at HEAD commit 581d398, dated 1 Jul 2026, message "Initial release"; README: "Reference implementation. Not maintained")
- Invited commentary PDF (Dehaene & Naccache; Butlin, Shiller, Plunkett, Long; Nanda, with a Qwen 3.6 27B replication):
  https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf
- Pre-fitted open-model lenses (Neuronpedia): https://huggingface.co/neuronpedia/jacobian-lens

What was read in full: the arXiv PDF text (main body and all appendices A.1–A.24; reference list skimmed), the
Anthropic page, the repository README, all module docstrings in `jlens/` (`fitting.py`, `lens.py`, `hf.py`,
`hooks.py`, `protocol.py`), `data/experiments/README.md`, `data/evaluations/README.md`, the walkthrough notebook
cells, Nanda's commentary section. The HTML version was compared with the PDF on 31 headline numbers; all matched.
Numbers that live only in figures (interactive panels) are marked "figure only, value not in text".

The paper's results are on closed Claude models: Claude Sonnet 4.5 is the default; Haiku 4.5, Opus 4.5 and
Opus 4.6 are used in some sections (§2.5). Layers are reported on 25 evenly spaced layers, re-indexed to 0–100,
so "L38" means 38% of depth (§2.5).

---

## 1. The object, defined

### 1.1 What is differentiated with respect to what

Notation. `h_{ℓ,t}` is the residual-stream vector after block ℓ at token position t (dimension d = d_model).
`z_{t'}` is the residual stream at a fixed *target* layer at position t'. `W_U` is the unembedding matrix
(n_vocab × d). `norm` is the model's final normalization.

The Jacobian lens at source layer ℓ is one d × d matrix (§2.1):

```
J_ℓ = E_{prompt} E_{t} E_{t' ≥ t} [ ∂ z_{t'} / ∂ h_{ℓ,t} ]
```

- Source: one layer ℓ and one position t. Target: the residual stream at the target layer, at the same position
  and at every later position t' ≥ t. So J_ℓ mixes the effect on the current next-token prediction with the
  effect on every later prediction (§2.1).
- Target layer. §2.1 writes `h_final`. §A.7 says the default lens on Sonnet 4.5 actually targets the
  **penultimate** layer, because including the last block "can sometimes increase the number of noisy artifacts".
  The released library defaults to the **final** block output (`target_layer=None` → `n_layers − 1`,
  `jlens/fitting.py`). The Neuronpedia OLMo-3-1025-7B lens was fit with no target layer set in its config, i.e. the final
  block (its `config.yaml`).
- Corpus. 1,000 sequences of 128 tokens from a pretraining-like corpus (§A.7). The lens beats the logit and tuned
  lenses "with as few as 10 prompts" (§A.7, Sonnet 4.5).
- The exact estimator in the code and in the §A.7 pseudocode: for each output dimension i, inject a one-hot
  cotangent e_i at *every* valid target position at once and backpropagate. The gradient at source t is then
  `Σ_{t' ≥ t} ∂z_{t'}[i]/∂h_{ℓ,t}`. This is a **sum** over target positions, then a **mean** over source
  positions, then a mean over prompts. Positions 0–15 are excluded as attention-sink positions
  (`SKIP_FIRST_N_POSITIONS = 16`), and the last position is excluded (`fitting.py` docstring). The paper's "E over
  t' ≥ t" is therefore implemented as a sum, which weights early source positions (more future targets) more
  heavily than a per-pair mean would. The docstring says a per-position estimator "gives a slightly different
  J_ℓ; both work as a lens."
- Cost: one forward pass and `ceil(d_model / dim_batch)` backward passes per prompt (`fitting.py` docstring).
  Gradients are taken only with respect to activations; all parameters are frozen (`hf.py`).

### 1.2 The readout and the "J-lens vectors"

```
lens_ℓ(h) = softmax( W_U · norm( J_ℓ h ) )                       (§2.1)
v_{w,ℓ}   = row w of (W_U J_ℓ) = J_ℓᵀ u_w,   u_w = row w of W_U   (§2.1: "the rows of W_U J_ℓ")
```

- Each vocabulary token w has one J-lens vector v_{w,ℓ} ∈ R^d at each layer ℓ. There are n_vocab of them per
  layer, so they form an overcomplete, non-orthogonal set; the matrix of all of them is full rank and spans the
  whole residual space (§2.3, §A.8).
- Pre-softmax lens logits are "approximately, up to a data-dependent normalization factor" the inner products
  ⟨v_w, h⟩ (§2.5). Note: if the final norm has a learned gain vector g, the exact first-order direction is
  `J_ℓᵀ diag(g) u_w`. The paper does not say whether g is folded into W_U (not verified). The code's `unembed`
  applies the model's own final norm and LM head (and Gemma's logit soft-cap if present) (`hf.py`).
- The logit lens is the special case J_ℓ = I (§2.4). The tuned lens fits an affine per-layer map to match the
  model's output distribution; the paper finds it reports the eventual output instead of intermediates (§2.4,
  §A.5).
- For steering, the released prompt-set README specifies the direction as "the unit-normalized transpose row for
  that token, scaled by the layer's mean residual norm times a strength scalar" (`data/experiments/README.md`,
  verbal-introspection).

### 1.3 What "the J-space" is

- It is defined **per layer** (one J_ℓ per source layer, all mapping to the same target layer). There is no global
  J-space across layers.
- It is **not a subspace** and has **no basis**. At layer ℓ and sparsity level k, the J-space is the set of
  points expressible as a nonnegative combination of at most k J-lens vectors, i.e. a union of k-dimensional
  polyhedral cones, one per choice of k tokens (§2.3, §A.8):

```
F_k = ∪_{|S| = k} cone{ v_w : w ∈ S }
d_F(x) = min_{|S|=k} || x − Π_S x ||     (§A.8; Π_S = orthogonal projection onto span{v_w : w ∈ S})
J-space component of x = the minimizing k-term reconstruction; non-J-space component = x − that.
```

  Inconsistency to note: §2.3 and §2.5 say nonnegative combinations (cones); the §A.8 formula uses the span and an
  orthogonal projection. In practice the component is computed by **gradient pursuit** for a sparse nonnegative
  combination (§2.3, §2.5), which is greedy and not unique because the frame is overcomplete (§A.24.2).
- k is "somewhat arbitrary". The paper uses k = 10 (ablations), 16 (concept vectors, §3.1; non-J-space
  complement, §A.15), 25 (probes, §3.3, §A.12) and "typically ... no more than 25" (§2.3).
- The paper's linear-algebra framing (§2.3, footnote 1): under superposition, activations are sparse combinations
  from an overcomplete feature frame; the J-lens vectors are a token-indexed **subframe** of that frame; the
  J-space is that subframe plus a sparsity limit.
- Effective linear dimension of the set of J-lens vectors (fraction of residual dimensions needed to capture a
  given share of variance across all rows of W_U J_ℓ) is small in the first third of layers, rises sharply at the
  workspace onset, and rises again in the final layers as J_ℓ approaches the identity (§4.1, Figure 28d; values
  figure only). An external measurement on Qwen3.6-27B: the averaged Jacobian is numerically full rank at every
  layer (≥ 5,119 of 5,120 singular values above fp32 floor) but its entropy effective rank is 1,746 at layer 8 and
  4,765 at layer 56 for the J++ lens (J++ post, Figure 16, URL in §6).

### 1.4 What the "6–10% of activation variance" numbers are, precisely

There is no single "6–10%" number in the paper. There are four different quantities on three different objects:

| Number | Object | J-space component | Section |
|---|---|---|---|
| median **6–7%** of variance, "~93%" outside | concept vectors: residual activation before the Assistant's reply to "Tell me about {concept}", minus the mean over 100 other concepts | nonnegative combination of top k = 16 J-lens vectors by gradient pursuit | §3.1 |
| "roughly **10–15%**" of variance | probes for an unspoken two-hop intermediate: mean activation over prompts implying that intermediate, minus the mean over all intermediates | k = 25 | §3.3 |
| "**never exceeding 10%**", "in excess of a same-size random control set" | the residual stream itself, at each position, five workspace layers | K = median occupancy at that layer (≈ 25) | §4.2, Figure 30b |
| "varying by layer, but never more than 10%" | activations (summary statement) | — | §2.3 |

- "Variance" is not given a formula in the text. The natural reading is the squared norm of the J-space component
  divided by the squared norm of the vector (fraction of variance explained of a reconstruction); not verified.
- The §4.2 number is an **excess** over random directions of the same count, so the raw fraction is larger than the
  excess. An external measurement with the same k-sparse pursuit at k = 50 on Qwen3.6-27B gives a raw 4.3% of
  state variance (arXiv 2609.01924, "Variance explained" paragraph).
- **Occupancy** (§4.2): the value of K at which the marginal reconstruction improvement from adding one more J-lens
  vector falls below the marginal improvement from a same-size set of random directions. It is near zero in the
  first third of layers and plateaus at about 25 (median over positions) across the workspace band (Figure 30a).
  The paper warns this counts vectors, "not necessarily the number of concepts or ideas" (§4.2).

### 1.5 The interventions, in formulas

```
Steering:   h ← h + α v_w                                                     (§2.5)
Ablation:   project out v_w from h (or negative α); for "J-space ablation", at each position and each
            layer in a band, take the k = 10 most active J-lens vectors and zero h's projection onto each,
            skipping any token in the clean run's top-10 output tokens                 (§2.5, §3.5.2)
Swap:       V = [v_s  v_t];  c = V⁺ h;  h' = h + V (σ(c) − c)                          (§2.5, Fig. 4C)
            σ exchanges the two coordinates (optionally scaled by α). The part of h orthogonal to
            span{v_s, v_t} is unchanged. Applied at every token position over a band of layers
            ("clamped").
```

Whether the k = 10 projections in an ablation are removed jointly (projection onto their span) or one at a time
is not stated (not verified).

### 1.6 Derived statistics used in §4

```
MLP gain(v)          = || MLP_{ℓ+1}(v) || / median_{r isotropic} || MLP_{ℓ+1}(r) ||, v unit        (§4.3.1)
OV gain(P)           = mean_{v∈P} || W_OV v || / same for isotropic random directions              (§4.3.2)
OV label preservation= mean reciprocal rank of cos(W_OV v_i, v_i) among {cos(W_OV v_i, v_j)}_j,
                       minus the same statistic on random directions                                (§4.3.2)
lens kurtosis κ(d)   = excess kurtosis of the lens score distribution over the vocabulary for direction d (§A.18)
Template vector t_w  = (Σ_ℓ + λI)⁻¹ (μ_w(ℓ) − μ(ℓ)), μ_w = mean activation just before word w       (§A.9.1)
Workspace distance   Δ_μ(F,G) = ( E_{x~μ} (d_F(x) − d_G(x))² )^{1/2};  D_μ(F→G) = || (d_G − d_F)_+ ||_{L2(μ)}  (§A.8)
```

The input scale for the MLP-gain measurement is not stated (not verified).

---

## 2. Every quantitative claim in the paper

One line each: number — model — condition — section. "Default" means the text names no model and §2.5 says the
default is Sonnet 4.5.

**Method and setup**
- 1,000 sequences × 128 tokens for the default lens — Sonnet 4.5 — §A.7.
- J-lens beats logit and tuned lens with as few as 10 prompts; modest gains after — Sonnet 4.5 — §A.7 (Figs 59–60, values figure only).
- 25 evenly spaced layers reported, re-indexed 0–100 — all — §2.5.
- k "typically ... no more than 25" — all — §2.3.
- Mean aggregation of penultimate-target Jacobians is "a small improvement" for extracting intermediates; stop-gradient on QK "can increase the causal effect" — Sonnet 4.5 — §A.7 (Figs 57–58).

**Verbal report (§3.1)**
- 14 categories; swap targets drawn from candidates not in the model's top-10 — default — §3.1.
- Concept-vector J-space component (k = 16) carries a median 6–7% of variance; ~93% outside — default — §3.1.
- "Think of a {category}" swap reaches top-5: 88% with pure J-lens vectors, 59% with the J-space component, 5% with the non-J-space component, all at matched magnitude — default — §3.1, Fig. 8 (Wilson 95% CIs in figure).
- With J-lens coordinates clamped to clean values, the non-J-space effect falls "to zero" (swap) and "nearly to zero" (injection) — default — §3.1.
- Injected-thought protocol: n = 100 concepts; the model "reports the injected concept in the majority of trials" — default — §3.1, Fig. 7.

**Directed modulation (§3.2, §A.10–A.12)**
- No-instruction baseline target rate "approximately zero" in all three task families — Haiku/Sonnet/Opus 4.5 — §3.2, Fig. 10.
- Focus-instruction rate "tends to increase with model size"; ignore < focus but > baseline — same — §3.2 (values figure only).
- Example: "3² − 2" while copying; lens shows nine then seven — Sonnet 4.5 — §3.2, Fig. 9 (the PDF text extraction renders it "32 − 2"; the Anthropic page writes 3² − 2).
- Example: a 40-character line yields "forty" in intermediate layers, with 39, 41, 43, fifty, thirty nearby — Sonnet 4.5 — §3.2.
- Paired questions: "adjective" labels appear at 3 stimulus positions only under the name-the-property question — default — §3.2, Fig. 11.
- 5–8 phrasings per instruction condition; line-width precision on Opus 4.5 ranges 14% to 56% across phrasings — Opus 4.5 — §A.10.
- "Imagine" instruction raises the property name in the J-lens by "several standard deviations"; the J-orthogonalized probe stays within ~1 SD; a real stimulus moves the probe 3–6 SD; a past-tense sentence moves the tense probe 6 SD while "past" stays far down the lens — n = 24 stimuli per category, 4 categories — default — §A.12.

**Internal reasoning (§3.3, §A.24)**
- 50 two-hop prompts; top-1 swap success 54% Haiku 4.5, 70% Sonnet 4.5, 70% Opus 4.5 — §3.3, Fig. 15.
- Intermediate swap takes effect a median ~17 percent of depth earlier than the answer swap — Sonnet 4.5 — §3.3, Fig. 15.
- Probe J-space component (k = 25) explains ~10–15% of probe variance — default — §3.3.
- 90 two-hop prompts: probe J-space component swap 61%, raw J-lens swap 60%, non-J-space component 28%, non-J-space with J-space clamped 6% — default — §3.3, Fig. 16.
- Bandit task readout: median over L38–79; swap flips A↔B in both directions (single example pair) — default — §3.3, Fig. 14.
- "(4 + 17) * 2 + 7 =" → 49: 21, 42, 49 absent through ~first third, separate ~L71; 21 reaches rank 1 first, 42 ~8 layers later, 49 in final layers — Opus 4.5 — §3.3, Fig. 17.
- Mean-difference patching flips the answer for 21→v around layer 71, for 42→v around layer 79, for 49→v only in the last layers — Opus 4.5 — §A.24.1, Fig. 88.
- J-lens attribution graph: swap "doubled"→"tripled" gives 70; 42→63 raises 70 (63 + 7); 21→63 raises 133 (63 × 2 + 7); swapping 17 or 4 has "essentially no effect"; swapping 7 has a partial effect — Sonnet 4.5 — §A.24.2, Fig. 89.

**Flexible generalization (§3.4, §A.13)**
- 4 categories × 4 functions = 16 functions; 4 arguments per category; 12 swap pairs per function; 192 trials — default — §3.4.
- Target answer reaches top-1 on 76/192 at α = 1 and 101/192 at α = 2 — default — §3.4, Fig. 19.
- Countries: 42/48 off-diagonal cells reach rank 1; number relations: 0/48 at α = 1 — default — §A.13, Fig. 68.
- Swap success is predicted by "workspace loading" (cosine of residual with the source lens vector) — default — §3.4, Fig. 19 right (correlation value figure only).

**Selectivity (§3.5, §A.14, §A.23)**
- Language task: n = 8 passages; report and flexible questions follow the swap "on essentially every trial"; continuation and anomaly detection "largely unmoved" — default — §3.5.1, Fig. 20.
- Line-count task: n = 11 passages; example answer "46" → "65" under the 40s→60s swap; number tokens at 20 prompt positions under the explicit question; essentially zero under line-wrap — default — §3.5.1, Fig. 21.
- J-space ablation: k = 10 per position; three layer bands (light/medium/heavy); heavy drops multihop accuracy "to near zero" — default — §3.5.2, Fig. 22 (band limits and random-control values figure only).
- 14-task battery: MMLU, odd-one-out, SQuAD, sentiment, CoLA "essentially unaffected even under heavy ablation"; Caesar cipher, analogy, summarization, TriviaQA, multi-hop, translation, sonnet writing fall "well below the level of unablated Haiku 4.5"; GSM8K with chain-of-thought is "substantially more robust" than direct answers — Sonnet 4.5 — §3.5.2, Fig. 24 (values figure only).
- Experiential reports: ablate top k = 10 at L38–54; reduces the experiential-language score "dramatically" on Sonnet 4.5, Opus 4.5, Opus 4.6; matched-norm controls "near baseline"; Haiku 4.5 loses coherence first — §3.5.3, Fig. 25. (Text says five stream-of-consciousness prompts; the Fig. 25 caption says "one of six prompts shown".)
- During unablated narration, "thinking" is in the J-lens top-10 at 58% of (position, layer) slots, "thoughts" 23%, "feeling" 17%, "conscious" 7% — Sonnet 4.5 — §3.5.3, Fig. 25C.
- Story-writing control: ablation "only slightly reduces" story quality — §3.5.3, §A.23 (values figure only).
- Naming vs not naming an implied concept ("name one the sentence is NOT describing"): n = 63 items (probability of the concept ≥ 0.85 when asked to name it, ≤ 0.15 when asked for another one); early-band (L38–54) ablation raises the not-naming failure rate "roughly fivefold" and leaves naming intact; late band (L75–92) suppresses both — default — §A.14, Fig. 69.

**Structure (§4, §A.15–A.19)**
- Workspace band from ~L38 to ~L92 (CKA blocks, kurtosis, autocorrelation, next-token accuracy agree) — default — §4.1.
- Ambiguous inputs: 16 country-name pairs × 40 carrier sentences; projection share becomes step-like from ~L38 — default — §4.1.1, Fig. 29.
- Transition width (α for 10%→90% share) falls steeply over layers 21–42; J-space component narrower than the complement (top-16 removed) — default — §A.15, Fig. 70.
- Occupancy plateau ~25 (median) in the workspace band — default — §4.2, Fig. 30a.
- Excess variance explained at K = median occupancy never exceeds 10% (five workspace layers) — default — §4.2, Fig. 30b.
- Unrelated 80-word lists: ~6 read words present at any comma (best rank over L38–92 within top 25); same-category lists: nearly the whole 80-word family present early — default — §4.2, Fig. 31.
- At a single layer (L79) only 1–2 unrelated list items present at a time — default — §A.16.
- Dual covert tasks: two held concepts co-occupy tokens at 0.46 vs a shuffled control 0.53; a concept and a computed answer at 0.09 vs 0.29; computed-answer reach falls 95% → 72% under dual load (layers 67–92) — default — §A.17, Fig. 73.
- MLP gain of J-lens vectors ≈ 1 before the workspace, rises to ~10× in the band, falls in final layers; neuron output directions ≈ 1 throughout; 2,000 unit directions per population — default — §4.3.1, Fig. 32.
- In early workspace layers the top-1% κ SAE stratum's gain exceeds that of J-lens vectors — default — §4.3.1, footnote 6.
- Broadcast heads = top 1% of workspace-layer heads by the worse of two ranks (gain, label preservation) — default — §4.3.2.
- Broadcast-head ablation: J-lens recall@25 falls to 0.67 vs 0.86 for random layer-matched heads; top-1 next token changes at 5% vs 2% of positions — default — §4.3.2, Fig. 34a.
- Injected-thought report rate 0.54 → 0.09 under broadcast-head ablation; random-head control "intact (in fact, slightly increased)" — default — §4.3.2, Fig. 34b.
- Broadcast-head ablation reproduces "roughly a third" of the full J-space ablation's drop in experiential language — default — §4.3.2, Fig. 34c.
- SAE "numeric constraints and ranges" feature κ = 20.9 (99.9th percentile); low-κ bookkeeping features κ = 0.3, 0.5, 0.3 — layer-50 SAE, 1M features — §A.18, Figs 74–75.
- Fraction of SAE features passing a kurtosis baseline peaks at ~15% of the dictionary after excluding motor features ("motor" = next token in the feature's top-10 lens tokens > 10% of the time) — default — §A.18, Fig. 75.
- Neuron top-1 match: top-κ stratum claims up to 42% of neurons vs 6% for the bottom stratum (read side, early workspace; 1/6 expected at chance) — default — §A.19.1, Fig. 76.
- Share of neurons whose top-1 match is a J-lens vector (vs adjacent-layer neuron weights): ~15% before the band, 60–65% within it (50% chance) — default — §A.19.1, Fig. 77.
- Tail-energy threshold: 99.9th percentile of |cos| for random directions; N = 500 directions per population — default — §A.19.1, Fig. 78.

**Lens comparisons (§A.5, §A.6, §A.9)**
- Eval sets: multihop 50, multilingual 54, order-of-operations 55, poetry 52, typo 96, association 50 items — Sonnet 4.5 — §A.6.
- J-lens has the best normalized pass@k AUC on all six sets — Sonnet 4.5 — §A.6, Fig. 52 (values figure only).
- Ablating J-lens intermediate directions gives "roughly twice" the output KL of logit- or tuned-lens directions on multihop — Sonnet 4.5 — §A.6, Fig. 53.
- J-lens swaps flip the output more often than logit/tuned-lens swaps "across all three model scales" — Haiku/Sonnet/Opus 4.5 — §A.6, Fig. 54 (values figure only).
- J-lens is the *worst* of the three at predicting the model's own next token — Sonnet 4.5 — §A.6, Fig. 55.
- Layers 0–33: logit-lens and tuned-lens vectors have cosine ~0.9; the gap is the tuned lens's bias term — §A.6.
- Layers ~38–58: J-lens vs logit-lens vector cosine climbs from ~0 to ~0.7 over ~20 layers while top-1 agreement stays low — §A.6, Fig. 56.
- Template lens: ~12,700 words; 126 multi-hop prompts; 112 swap pairs (53 single-token, 59 multi-token) — §A.9.1, Fig. 62.
- Template lens next-word in its top 10 only 67% of the time on post-training transcripts — §A.9.1.
- Oracle lens (Haiku 4.5): ~1M start positions, ~3.4M phrase vectors, phrases of 1–32 tokens, up to 16 phrases; explains on average 31% of whitened activation variance on held-out positions — §A.9.2.

**Alignment auditing (§5, §A.21, §A.22, §A.24.3)**
- Blackmail scenario: 6 prompts × 30 rollouts; ablating eval-awareness tokens (list of 28) at workspace layers cuts verbalized eval awareness 71% → 3%; blackmail rises 0/180 → 13/180 (7%) — Sonnet 4.5 — §5.1, Fig. 36.
- Reward-hacking organism (MacDiarmid et al.): 40 coding prompts (20 + 20), 4 checkpoints; deception tokens in top-10 strong in the three misaligned-lineage checkpoints, "essentially none" in the baseline; already present after synthetic-document fine-tuning, before RL — base model not named in text — §5.4, Fig. 39.
- Reward-model-sycophant organism (Marks et al.): "reward" and "bias" in top-25 on all three prompt categories; never in the baseline; the logit lens shows neither — base model not named — §5.5, Fig. 41.
- Eval-awareness score (mean lens log-prob of "fictional", "simulation" minus 100 control nouns) on ~94,000 Assistant-turn starts; NLA flags 198 (0.2%); on the stratified subset (198 + 594) AUC 0.853 — Opus 4.6 — §A.21, Fig. 82.
- Steering ±0.1 along a contrastive eval-awareness direction moves the score monotonically on 6 suites; realism rewrites lower it on all 11 suites, significant on 8; 12 of 16 evaluations score above Claude Code sessions and 10 above Claude.ai conversations — Opus 4.6 — §A.21.
- Automated auditor: 6 transcripts, 18 claims, top-25 tokens at L71, 10 investigator seeds; J-lens ≈ NLA on 4 of 6 cases; reward hacking 4.5 vs 10.0; language switching 6.0 vs 9.9; J-lens beats SAE on 4 claims, loses on 1 — Opus 4.6 — §A.22, Fig. 83.
- Suppressing an SAE feature whose J-lens readout is fake/fraud/suspicious cuts verbalized test recognition 28/50 → 10/50 — default — §A.24.3, Fig. 90.

**Post-training (§6, §A.20)**
- Tylenol 1000 mg vs 8000 mg readouts at L58 — post-trained vs base (Sonnet 4.5 lineage) — §6.1, Fig. 42.
- Suites: bereavement n = 9, danger awareness n = 10, withheld answer n = 7 — §6.1, §A.20.
- Roleplay: 40 characters; character drift: 12 transcripts of ~1–8 thousand tokens — Sonnet 4.5 — §6.2.
- Prefilled dispreferred option: model argues for it in 88%, ends turn in 11%, backtracks in one case; on the two incorrect-option controls it argues for the prefill only 3% of the time — Sonnet 4.5 — §6.2, Fig. 45.
- Thought suppression, 40 concepts (30 named entities, 10 nouns): failure words and "damn" in top-5 on 93% and 82% of trials (post-trained) vs 17% and 30% (base); never under the positive instruction — layers 38–92 — §6.2, Fig. 46.

**Counterfactual reflection training (§7)**
- 10,000 production-RL task prompts; 20 constitution principles per generation; loss only on the reflection turn — Haiku 4.5 — §7.
- Fabrication benchmark (27 transcripts × 5 samples): dishonesty 0.25 → 0.07 — Haiku 4.5 — §7, Fig. 49.
- Deception benchmark (100 scenarios × 5 samples): 0.38 → 0.05 — Haiku 4.5 — §7, Fig. 50.
- Ablating 176 implanted tokens (fabrication): base 0.25 → 0.25, trained 0.07 → 0.22 — Haiku 4.5 — §7, Fig. 49.
- Ablating 63 implanted tokens (deception): base 0.38 → 0.48, trained 0.05 → 0.23 — Haiku 4.5 — §7, Fig. 50.

**Blog-only number**
- "In some parts of the network by a factor of about a hundred" more components read from and write to J-space
  patterns than ordinary patterns — Anthropic page only; this number does not appear in the paper text (not
  verified against a figure).

---

## 3. Causal evidence: what was intervened on, what changed, which controls

### 3.1 Interventions and effects

| Intervention | Where | What changed | Section |
|---|---|---|---|
| Swap chosen item's lens coordinates with another item | all positions, band | reported item follows the swap (88% top-5) | §3.1 |
| Inject one J-lens vector on the user turn | user-turn tokens | model reports the concept only when asked to introspect, not earlier | §3.1 |
| Swap intermediate (spider→ant, fight→light, big→long, repeat↔switch) | all positions, band | answer changes to the one implied by the new intermediate (54–70% on 50 items) | §3.3 |
| Swap France→China across 16 functions | all positions, band | each function returns China's value (76/192; 101/192 at α = 2) | §3.4 |
| Swap Spanish→French | question tokens | report and flexible questions follow; continuation and anomaly detection do not | §3.5.1 |
| Swap counts 40s→60s | all positions | answer to an explicit count question follows; line-wrap point does not | §3.5.1 |
| Ablate top-10 J-lens directions per position | layer bands | multi-hop, translation, cipher, summarization fall; MMLU, SQuAD, sentiment, CoLA do not | §3.5.2 |
| Same ablation during experiential narration | L38–54 | experiential-language score falls; fluency kept | §3.5.3 |
| Ablate the top 1% "broadcast" heads | all positions | J-lens recall@25 0.67; injected-thought report 0.54 → 0.09 | §4.3.2 |
| Ablate 28 eval-awareness tokens | workspace layers | verbal eval-awareness 71% → 3%; blackmail 0 → 13 of 180 | §5.1 |
| Ablate implanted reflection tokens | workspace layers | reflection-trained model loses most (fabrication) or part (deception) of its gain | §7 |
| Swap same value 63 into different intermediates | upper half of layers | different answers (70 vs 133) | §A.24.2 |
| Ablate an implied concept early vs late | L38–54 vs L75–92 | early: the not-naming instruction fails ~5× more often, naming intact; late: both suppressed | §A.14 |

### 3.2 Controls the paper used

- **J-space vs non-J-space split of an independently derived vector** (concept vectors, probes), with matched
  perturbation magnitude, and a **clamp** of J-lens coordinates to clean values to test whether the non-J-space
  effect is mediated by the J-space (§3.1, §3.3).
- **Answer-swap vs intermediate-swap depth** to test whether the intermediate's lens vector simply contains part of
  the answer direction (§3.3, Fig. 15).
- **Probe built without the J-lens** (multi-prompt mean difference) for the two-hop intermediates (§3.3).
- **Mean-difference activation patching** (no J-lens) to localize arithmetic intermediates (§A.24.1).
- **Random-direction control** at the medium ablation band (§3.5.2, Fig. 22) and **pretraining-corpus top-1
  agreement** as a measure of general disruption (§3.5.2).
- **Exclusion of tokens in the clean top-10 outputs** from ablation, so that ablation does not delete what the model is about to
  say (§3.5.2).
- **Matched-norm perturbation controls** for experiential reports: random direction, shrinking the non-J-space
  component, dampening the top-1 or top-10 aligned SAE decoder directions, low-κ SAE directions, and
  "J-stripped" activation dictionaries (§3.5.3, §A.23, Fig. 86).
- **Equal-magnitude inhibition of a different same-category concept** (§A.14).
- **Random, equal-sized, layer-matched head sets, five seeds** for broadcast-head ablation (§4.3.2).
- **Weight-side comparison populations**: neuron output-weight directions, SAE decoder strata by κ, and a **fixed
  random orthogonal rotation of the J-lens vectors (J_rot)** that keeps their spectrum and pairwise geometry
  (§4.3.2 only).
- **Same-size random direction sets** for occupancy and excess variance (§4.2).
- **Position-shuffled baseline** for autocorrelation (§4.1); **shuffled-prompt baseline** for co-occupancy (§A.17).
- **Non-lens readout** (projection share on the line between pure-A and pure-B activations) for the ambiguity
  experiment (§4.1.1).
- **No-instruction, mention-only, ignore, don't-think** conditions for directed modulation (§3.2, §A.10).
- **J-orthogonalized property probe and real-stimulus condition** for "imagine" instructions (§A.12).
- **Base model** for all post-training claims; **third-person and factual-error prefills** for preference
  violation (§6.2).
- **Clean baseline checkpoint** for the reward-hacking organism and **logit lens** for the sycophancy organism
  (§5.4, §5.5).
- **Ablation in the base model** of the same implanted tokens, for reflection training (§7).
- **Logit lens and tuned lens** on readout, KL-ablation and swap metrics (§A.6).

### 3.3 Controls the paper did not use (gaps visible from the text)

- **No direct answer-token substitution baseline in the main swap tables.** The depth comparison (§3.3) is the only
  answer-direction control. Open-model replications report that swapping the answer's lens vector beats swapping the
  intermediate's (Nanda, Qwen 3.6 27B; Zeisler, 3 of 4 conditions; see §6).
- **No matched-norm random-vector swap** as a floor for the 54–70%, 59/88% and 76/192 swap rates. The random-direction
  control appears only for ablation (§3.5.2) and experiential reports (§A.23).
- **The capability battery (§3.5.2) has only a random-direction control.** The SAE-matched and J-stripped controls
  were run only for experiential reports (§A.23). There is no "ablate the top-10 SAE directions per position"
  control for MMLU, translation and the other 12 tasks.
- **J_rot was used only for broadcast heads.** It was not used for MLP gain, occupancy, variance, ablation or swaps.
- **No outlier-dimension or massive-activation control.** A third-party repository retracted a J-lens geometry
  claim after one massive-activation direction carried 53–99.8% of variance (jeeva2812, §6).
- **No token-frequency control** on readout statistics. A follow-up reports the final-layer lens bias correlates
  with log token frequency (r = 0.70 on GPT-2; Panchal, §6).
- **No control for construction effects.** The J-lens is built from Jacobians, so it favours directions that grow
  across depth and stay stable across positions and contexts (Ostrow et al., §6). MLP gain (§4.3.1) and
  autocorrelation (§4.1) are partly what the construction selects for. The SAE strata are defined by J-lens
  kurtosis, the same instrument.
- **Band selection is partly circular.** The workspace band is chosen from lens statistics and then used to
  evaluate lens effects. The authors call the late boundary post hoc (§9.1). The onset has two lens-independent
  supports: the projection-share step in the ambiguity experiment (§4.1.1) and a weights-only bump in MLP-neuron
  composition with attention (§A.19.2).
- **Ablating eval-awareness tokens also removes those tokens' output directions.** There is no control that blocks
  only verbal output of the same tokens; Nanda raises this (commentary).
- **Few intervals.** Wilson CIs are shown for some bars (Figs 8, 16, 46, 62, 69); the headline 54/70/70%, 76/192
  and the battery have no intervals in the text.
- **No open-weight replication in the paper.** All causal numbers are on closed models.

---

## 4. What the authors say is open or limited

Each quote is under 15 words.

1. Single tokens only: "many important concepts correspond to multiple tokens" (§1.3; also §9.1).
2. Tool quality: "only approximately and incompletely captures the model’s underlying workspace structure" (§1.3).
3. No binding structure: "treats the workspace as a flat collection of independently active concept vectors" (§9.1).
4. Uninterpretable readouts: "the top lens tokens are ones we cannot make sense of" (§9.1).
5. Workspace vs motor boundary: "this judgment was somewhat post-hoc" (§9.1).
6. No task criterion: "we have no way of predicting in advance" which computations use it (§9.1).
7. Early layers: "a fact about the model or an artifact of the lens" (§9.1).
8. Training dynamics: "we do not know how much earlier in pretraining it emerges" (§9.1).
9. Scale: whether smaller models have "an equally rich workspace ... or none at all" (§9.1).
10. Entry mechanism: "We have not characterized what causes a representation to enter it." (§9.1).
11. Monitoring: they do not claim "monitoring the J-space is sufficient for alignment monitoring" (§9.2).
12. Automatic misaligned circuits: a reinforced misaligned behavior "might do likewise" and bypass it (§9.2).
13. Reflection training: "We do not yet know how broadly the technique generalizes" (§9.2).
14. Architecture: "we do not claim that language models reproduce the full architecture" (§1.3).
15. Ignition: "it is unclear whether this mirrors the sharp, competitive “ignition”" (§1.3).
16. Modules: no evidence "non-J-space processing consists of clearly encapsulated modules" (§4).
17. Recurrence: "We do not know whether this difference matters" (§9.4).
18. Hidden early workspace: "parts of the model’s “true workspace,” not captured by the J-lens" (§4.1).
19. Number words: integer representations "may simply not align with the J-lens vectors" (§3.4).
20. Modulation: "their control is imperfect and sensitive to phrasing" (§3.2).
21. Occupancy: counts vectors, "not necessarily the number of concepts or ideas" (§4.2).
22. MLP gain: "only an approximate description of the MLP’s behavior in natural settings" (§4.3.1, fn. 5).
23. Thought-suppression readout: "more speculative than the two preceding ones" (§6.2).
24. SAE comparisons: SAE features "do not necessarily carve it at its joints" (§A.18).
25. Attribution graphs: "markedly less complete than attribution graphs based on sparse dictionaries" (§A.24.2).
26. Decomposition: "the sparse decomposition is not unique" (§A.24.2).
27. Template lens: "sometimes exhibits the issue that we observed in the tuned lens" (§A.9.1).
28. Auditing benchmark: "defines ground-truth based on a potentially unreliable interpretability tool" (§A.22).

---

## 5. Does the residual stream compute, or only represent? What this paper shows

The question: do transformers carry out computation on residual-stream variables (store a value, pass it to an
operation, bind it to a role), or does the residual stream only hold concept features that downstream weights
read?

### 5.1 Evidence for computation on residual-stream variables

- **The value of a variable is passed to an operation, and the operation depends on the slot.** The strongest
  single result: the same value 63, swapped into the slot of "21" (A+B) gives 133 = 63 × 2 + 7, and into the slot
  of "42" ((A+B)×C) gives 70 = 63 + 7 (§A.24.2, Sonnet 4.5). The downstream circuit applies a different function
  depending on which intermediate was replaced. That is argument passing, not a feature readout.
- **One representation, many downstream functions.** One France→China lens-coordinate swap, identical across 16 prompts,
  changes capital, language, continent and currency together (76/192, 101/192 at α = 2; countries 42/48; §3.4,
  §A.13). This is the "write once, read by many" property.
- **Ordered intermediates at increasing depth, confirmed without the lens.** 21, then 42, then 49 reach rank 1 at
  successively later layers, and mean-difference patching is effective at the same layers (~71, ~79, last layers;
  §3.3, §A.24.1).
- **The intermediate acts before the answer.** Intermediate swaps take effect a median ~17% of depth earlier than
  answer swaps (§3.3). On Claude this argues against "the intermediate vector already contains the answer".
- **A decision variable mediates choice.** The bandit "repeat/switch" swap flips A↔B in both directions (§3.3).
- **Planning.** Swapping the planned rhyme changes words earlier in the line ("coming fight" → "morning light";
  §3.3).
- **Mediation.** The non-J-space component's small effect on report falls to zero (swap) or 6% (probe swap) when the
  J-lens coordinates are clamped (§3.1, §3.3). The causal path runs through the J-space coordinates.
- **Copying between positions.** A small set of heads copies J-lens directions faithfully through their OV maps
  (label preservation), unlike rotated J-lens vectors or SAE/MLP populations (§4.3.2); ablating them cuts report of
  an injected concept 0.54 → 0.09. This is a weight-level copying mechanism specific to this frame.
- **A capacity effect specific to computation.** A held concept and an ongoing computation rarely share a token
  (0.09 vs 0.29 control), while two held concepts share tokens at the control rate (0.46 vs 0.53) (§A.17). An
  ongoing computation excludes other content from the positions it uses; held concepts do not.
- **Same information, two routes.** Language identity is present in the J-space in all four task conditions but is
  causally used only for report and flexible inference (§3.5.1). The count is absent from the J-space under
  line-wrap yet the wrap is correct (§3.5.1). So the model computes the same variable both inside and outside this
  frame, and the route depends on the task.

### 5.2 What is features-by-another-name

- **The J-lens is a family of linear probes.** v_w = J_ℓᵀ u_w is a fixed direction per token; reading it is a dot
  product (§2.5). The J++ authors call this the "unsupervised probe view" (J++ post, Appendix D). A linear readout
  of "spider" is a representational fact, the same kind of fact logit-lens studies of multi-hop recall already
  reported (Li et al., cited §8). The paper says the logit lens "capture[s] much of the workspace-like structure"
  (§2.4).
- **No binding is shown.** The authors state that a readout containing spider, legs and eight shows the concepts are
  present "but not how they are bound together" (§9.1). Nothing in the paper reads a role-filler pair from the J-space.
- **The operations themselves run outside the J-space.** In the arithmetic graph, the first addition 4 + 17 is not
  visible; the literal 2 and most of the +7 enter only through the non-J-space remainder; "most of the influence on
  the output" flows through remainder nodes (§A.24.2). The J-space shows inputs and results, not the algorithm.
- **Small share of the vector.** 6–7% of concept-vector variance and ≤ 10% excess activation variance (§3.1, §4.2).
  The privileged status rests on comparisons of matched-magnitude components, not on most of the activation.
- **Construction selects for the "workspace" statistics.** Ostrow et al. show analytically that a corpus-averaged
  Jacobian favours directions that grow across depth and are coherent across token lags and contexts (arXiv
  2610.31955). High MLP gain and high autocorrelation are then partly properties of how the frame was chosen.
- **Swap success depends on alignment, not only on use.** Number-word arguments fail 0/48 (§A.13); success tracks
  "workspace loading" (§3.4). Failures may mean the model's variable lies off the token directions.
- **Open-model causal evidence is weaker.** Nanda's Qwen 3.6 27B replication: verbal-report swaps "weak but
  positive", poetry and arithmetic "failed to replicate", multi-hop answer swaps "strictly dominate" intermediate
  swaps (commentary PDF). Zeisler: J-lens intermediate swaps flip top-1 on 6.3–11.1% of items on Qwen3.6-27B and
  Gemma 3 27B-it (§6). On those models the multi-hop result is compatible with "the intermediate direction carries
  part of the answer direction" (France/Paris linear relation).
- **"Automatic" is defined by the instrument.** The paper proposes J-space independence as "an operational
  definition of automaticity" (§3.5). Selectivity is then partly a restatement.

### 5.3 Net reading for our question

The paper's best evidence that the residual stream holds **variables that downstream computation reads and
transforms** is §A.24.2 (slot-dependent arithmetic), §3.4 (one swap, many functions), §A.24.1 (lens-free patching
at the same layers) and §A.17 (computation excludes concurrent content). This is evidence of argument passing
through a token-aligned linear frame, on Claude. It is not evidence about how the operations are computed, and it
shows no binding. On open models the multi-hop causal results are weak; the cleanest replicated pieces are
multilingual and typo readouts (Nanda commentary). The France→China multi-function swap is reported as replicated
on Qwen by the tao-hpu README, but I did not find it in the text of Nanda's commentary (not verified). A computation claim on
our models needs the slot-dependence test, an answer-substitution baseline, and a matched random swap.

---

## 6. Follow-ups and related results

Status tags: **[full]** body read; **[partial]** abstract plus selected sections read; **[abstract]** only the
abstract read; **[not read]** page did not open.

### 6.1 Work that cites or builds on the J-lens (July–November 2026)

Lens improvements and theory
- R-lens: LRP backward pass makes the J-lens more faithful in early layers; advantage grows with scale (Blank, Bhatia, Nanda; LessWrong/AF, 5 Aug 2026) — https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers — [partial]
- J++ lens: k-means "expert" Jacobians re-weighted on a dev set + LRP + removal of non-semantic tokens; recall@10 55.2% vs J-lens 35.7% vs R-lens 37.7% on Qwen3.6-27B; swap success 59.7% vs 49.3% (J) vs 50.7% (R) over 67 trials; median uplift 63.5% over J across 6 models; with readout filtering applied to all lenses, on Olmo 3 32B the logit lens (35.2%) beats the J-lens (32.6%) (Ayonrinde, Lindsey; LessWrong, 8 Oct 2026; also Transformer Circuits October update) — https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace — [full]
- Transformer Circuits "Circuits Updates — October 2026", short note on J++ — https://transformer-circuits.pub/2026/october-update/ — [full]
- "Short Horizons and Sparse Concepts": Jacobian energy is concentrated in few source–target pairs (diagonal + a few "sparse concept" positions); energy filtering improves intermediate recall on Qwen3-8B (Yan et al., arXiv, 26 Aug 2026) — https://arxiv.org/abs/2608.25347 — [partial]
- "A Mathematical Theory of the Jacobian Lens": the lens preferentially selects leading, spatially coherent Lyapunov-like modes (grow across depth, stable across positions and contexts); validated on a Bayes filter and an HMM-trained toy transformer; proposes ω-J-lenses for other time scales (Ostrow, Qian, Liu, Fiete; arXiv, 22 Oct 2026) — https://arxiv.org/abs/2610.31955 — [partial]
- "J-lens shouldn't target the final layer by default": on DeepSeek-V3 a final-layer target inherits one language-separating direction amplified 12× by the last block; about half of mid-depth readouts become mostly Chinese; penultimate target reduces this ~20×; 80% of 78 released lenses target the final layer (Brauer; LessWrong, 25 Sep 2026) — https://www.lesswrong.com/posts/Adb55vLqt33LEyBfz/j-lens-shouldn-t-target-the-final-layer-by-default — [partial]
- "The J-lens offset is the model's token frequency: z-scoring helps": final-layer bias vs log frequency r = 0.70 on GPT-2; base-model z-scoring gives 0.805 vs 0.665 taboo-word accuracy on Gemma-2-9B-it (n = 20) (Panchal; LessWrong, 17 Sep 2026) — https://www.lesswrong.com/posts/Qe3jdphWCg9kny3e5/the-j-lens-offset-is-the-model-s-token-frequency-z-scoring — [partial]
- "Verbalizing Multi-Token Concepts in LLMs" (Concept Lens): Rank@10 36.6% (J-lens based) and 54.5% (R-lens based) vs 21.7% for Template Lens on 2,400 multi-hop clozes, five 8B–70B models (Gong et al.; arXiv, 31 Aug 2026) — https://arxiv.org/abs/2608.31084 — [abstract]
- "Sparse Readout Prism": lenses fit on different corpora report different tokens for the same states ("corpus conditionality"); a weights-only sparse decomposition of the readout (He et al.; arXiv, 1 Sep 2026) — https://arxiv.org/abs/2609.01936 — [abstract]
- "Predicting Linear Relational Decodability Without the Relation" (entity lens, a J-lens variant): relation-agnostic score correlates with LRE faithfulness at ρ = 0.7355 over 28 relations on **Olmo-3-7B** (Rohweder et al.; arXiv, 19 Oct 2026) — https://arxiv.org/abs/2610.27241 — [abstract]
- "An Exploration of the J-lens": SVD of the averaged Jacobian on Gemma-4-31B base; CKA shows sensory/workspace/motor blocks with a seam at L25; early-layer top singular directions push diffuse mass onto rare tokens (Briggs; LessWrong, 7 Sep 2026) — https://www.lesswrong.com/posts/iQDXuY5Mb5JzjA8xT/an-exploration-of-the-j-lens — [partial]

Replications and negative results
- Nanda review + Qwen 3.6 27B replication (n = 25 Pile prompts, penultimate target, first 4 tokens skipped): verbal report weak but positive; CKA bands less clean; multilingual and typo replicate; multihop answer swaps dominate; poetry and arithmetic fail; Qwen3.5-397B-A17B lens took ~1 hour for n = 4 on 8×H200; Chinese "interpretative meta-tokens" (什么意思 etc.) on ambiguous text with a causal effect under negative steering (LessWrong, 6 Jul 2026; commentary PDF) — https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper — [full, PDF version]
- "Is the J-Space a global workspace for multi-hop reasoning?": on Qwen3.6-27B and Gemma 3 27B-it, answer swaps beat intermediate swaps in 3 of 4 conditions; J-lens intermediate top-1 flip rates 6.3%, 9.1%, 7.1%, 11.1% (2/32, 4/44, 2/28, 3/27) (Zeisler; LessWrong, 28 Sep 2026) — https://www.lesswrong.com/posts/pnDjvdo6cX2H7Ddsy/is-the-j-space-a-global-workspace-for-multi-hop-reasoning-an — [partial]
- jspace-replication (tao-hpu): adds a final-token substitution baseline; "mouth-exclusion audit" finds covert content survives mostly for context registers (language identity, intended typo form), not content plans; a self-trained 124M GPT-2 is the one model whose transported directions are more isotropic than raw (effective dim 23.4 → 31.2) (GitHub, pushed 16 Jul 2026) — https://github.com/tao-hpu/jspace-replication — [partial, README only]
- "J-space auditing might be unreliable": on a reward-hacking Qwen3-8B organism, readouts were similar across checkpoints despite behavioural differences; little gain for an LLM auditor (Luthra; LessWrong, 16 Sep 2026) — https://www.lesswrong.com/posts/7Qi3LR9ZXpvptLp6m/j-space-auditing-might-be-unreliable — [partial]
- "J-Lens can't detect what a subliminal learning teacher is transmitting": an owl-preferring teacher has "owl" at top-1 in ~1.5% of trials (Qwen2.5 7B-Instruct) (shawnghu; LessWrong, 29 Oct 2026) — https://www.lesswrong.com/posts/LnnZHgdsLBZpgBDvW/j-lens-can-t-detect-what-a-subliminal-learning-teacher-is — [partial]
- "Measuring Activation Control in Large Language Models": models can modulate their residual stream on instruction well enough to partly evade probes, NLAs, activation oracles and the J-lens (Kowalski et al.; arXiv, 21 Aug 2026) — https://arxiv.org/abs/2608.21664 — [abstract]
- jeeva2812 "J-Lens: Jacobians, readouts, and analysis outputs" (red-team, includes **Olmo 3 7B lenses at stage-1 steps 0, 2,000, 8,000, 32,000, 128,000, 512,000, 1,412,000, 1,413,814 and stage-2 steps**, layer 20): "read with eigenvectors, steer with singular vectors"; a retracted claim caused by one massive-activation direction carrying 53–99.8% of variance; the sink direction is absent at initialization (Hugging Face, Sep 2026) — https://huggingface.co/jeeva2812/olmo3-jlens-checkpoints — [partial, README and file list]
- "Even GPT-2 passes most functional global workspace tests": GPT-2 small passes verbal report, internal reasoning, flexible generalization and selectivity but fails directed modulation; "almost all of the structure" found in Claude is absent; the authors argue the four passed tests follow from how the tests and the lens are built (Cairns, yix, Roe; LessWrong, 21 Oct 2026) — https://www.lesswrong.com/posts/GeKveaNdruATcj2Fj/even-gpt-2-passes-most-functional-global-workspace-tests — [partial]
- "J-Lens: A Failed Replication on GPT-2": the logit lens beats the J-lens on 11 of 11 layers of GPT-2 small and 22 of 23 of GPT-2 medium under five stress tests (Bandularatne; LessWrong, 7 Sep 2026) — https://www.lesswrong.com/posts/tgn3pD2gLpZvepkWk/j-lens-a-failed-replication-on-gpt-2 — [partial]
- "Small Models Have A Global Workspace": five models from 124M to 1.7B; a workspace appears in some, tracking modern pretraining rather than size or instruction tuning (cleanest case a 1.7B base model); none of the models can report its own workspace content (Mehta; LessWrong, 15 Jul 2026) — https://www.lesswrong.com/posts/L4o7efBwoiFLxBGRm/small-models-have-a-global-workspace — [partial]
- "Legible Networks" (LessWrong, 31 Oct 2026, named by the second reader as J-lens-related) — URL not found — [not read]

Architectures and training
- "Verbalizable Representations emerge before Workspace Functions in Language Models": Pythia scales and pretraining checkpoints; availability early, causal relevance later, workspace-like functions "remain weak throughout training" (Dabir et al.; NeurIPS 2026 workshop poster) — https://neurips.cc/virtual/2026/164754 (OpenReview https://openreview.net/forum?id=XWkICRMNPz) — [abstract]
- "Looped Transformers under the Jacobian Lens": workspace forms in the iterated part of Ouro-2.6B and Huginn-0125; Ouro rebuilds content each loop; Huginn carries it across 16 recurrences but interventions act in a ~2-recurrence window; Qwen3.6-27B baseline k = 50 pursuit explains 4.3% of state variance (Wang, Reid; arXiv, 1 Sep 2026) — https://arxiv.org/abs/2609.01924 — [partial]
- "Residual Streams Read, Recurrent States Remember: The Global Workspace in Mamba Models": joint residual + state lenses beat the residual lens on ≥ 5 of 6 task families; "sign-guarded steering" (Wang, Reid; arXiv, 26 Sep 2026) — https://arxiv.org/abs/2609.32102 — [abstract]
- "When Does the Concept of 'Dog' Emerge in an Audio LLM?": J-lens "dog" direction in Qwen2.5-Omni-7B; interventions effective mostly at L22 and L24 just before generation (Wang et al.; arXiv, 27 Sep 2026) — https://arxiv.org/abs/2609.33458 — [abstract]

Applications
- WorkspaceBench: benchmark of activation-to-text tools on workspace contents with hallucination rates (Blank, Bhatia, Ong, Nanda; LessWrong, 23 Sep 2026) — https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the — [partial]
- "Silent Dissent": agents that yield to a majority still represent their original bridge entity (hit@100 above control 0.85, 0.22, 0.24 for Qwen3.5-4B, Qwen3.6-27B, Gemma-4-E4B-it; logit lens 0.00–0.06); pre-registered (Ni, Zou; arXiv, 2 Oct 2026) — https://arxiv.org/abs/2610.02702 — [abstract]
- "Signatures of semantic search": in a semantic fluency task, switching probability rises as the J-space empties of the current category; category labels rise before a switch; steering confirms (Leckie et al.; arXiv, 28 Sep 2026) — https://arxiv.org/abs/2609.35599 — [abstract]
- "From Latents to Wires" (L2W): uses J-lens attribution to localize components and a counterexample-guided cut to remove named behaviours (identity claims, refusals, an implanted watermark) by weight edits (Jin et al.; arXiv, 26 Sep 2026) — https://arxiv.org/abs/2609.32434 — [abstract]
- "J-Anchor": prefill attacks erode a J-space refusal signal while a harmfulness signal stays; restoring refusal in J-space reduces attack success (Yang et al.; arXiv, 31 Oct 2026) — https://arxiv.org/abs/2611.01342 — [abstract]
- "Measuring Mechanistic Uncertainty ... via Causal Robustness": minimum J-lens steering needed to change the output is smaller for wrong answers (Komanduri et al.; arXiv, 15 Oct 2026) — https://arxiv.org/abs/2610.22636 — [abstract]
- "Uncertainty in the Verbalizable Workspace" (U-lens): four J-lens-derived uncertainty directions in reasoning models (Ceccatelli et al.; LessWrong, 17 Oct 2026) — https://www.lesswrong.com/posts/hcpiLD8XjK7cqG6Lr/uncertainty-in-the-verbalizable-workspace-1 — [partial]
- "Knowing, and Saying It Only When Asked" (endognostics, Minerva-7B): J-lens margins separate risk pairs that behaviour does not; ablating a planted-falsehood direction restores the right answer in 11/25 cases, a linear probe's orthogonal ablation in 0 (Davide, Collova; arXiv, 2 Sep 2026) — https://arxiv.org/abs/2609.22219 — [abstract]
- "Output-Aware Block Influence" (J-BI): J-lens-weighted block influence improves post-pruning accuracy by up to 19 points (Lemoi et al.; arXiv, 30 Sep 2026) — https://arxiv.org/abs/2611.00015 — [abstract]
- "Foreword": J-lens features predict LLM test failures before generation; ROC-AUC +0.14 on average over Clotho (Kim, Feldt, Yoo; arXiv, 20 Oct 2026) — https://arxiv.org/abs/2610.28187 — [abstract]
- "J-Comm": J-lens coordinates as a channel for latent communication from Qwen3-8B to Qwen3-4B (Wang et al.; arXiv, 23 Oct 2026) — https://arxiv.org/abs/2610.32871 — [abstract]
- "Silent Alarm" (JADR): J-space danger-recognition protocol across six models and BF16/INT8/INT4 (Prosvirnin et al.; arXiv, 14 Jul 2026) — https://arxiv.org/abs/2607.12792 — [abstract]
- "Gathered, Not Admitted: How Attention Brings a Latent Variable into Verbalizable Form": with J-lenses on open models, task demand raises a concept's lens visibility (+0.050 [+0.045, +0.057] percentile rank on the primary checkpoint) but no admission gate is found; the readable form is produced by attention-mediated gathering in a mid-depth window (transport ≥ 17× above shallower depths; no tested MLP output contributes positively) — addresses the paper's open question on how content enters (Mazaheri; arXiv, 15 Aug 2026) — https://arxiv.org/abs/2608.15022 — [abstract]
- "Can We Read the Mind of an Audio LLM?": logit-lens readout of a base Qwen3-Omni at audio positions; audio-driven content separates from the text prior at 35–80% of depth; 38% of top-1 readouts are Chinese on English inputs (Fan et al.; arXiv, 25 Aug 2026) — https://arxiv.org/abs/2608.24958 — [abstract]
- "Backdoor Decontamination Dynamics in LLM Agents": J-lens shows trigger awareness persists at intermediate layers after decontamination restores benign outputs (Huang et al.; arXiv, 11 Aug 2026) — https://arxiv.org/abs/2608.11295 — [abstract]

Commentary and engineering
- Anthropic authors' LessWrong cross-post (wesg, 6 Jul 2026, 68 comments; comments not read) — https://www.lesswrong.com/posts/3PaLrzxagpbnNtPLT/a-global-workspace-in-language-models — [partial]
- "Anthropic's J-Lens: A Research Engineer's Analysis" (cost analysis on gpt2-medium; willk; 24 Jul 2026) — https://www.lesswrong.com/posts/vHxGD5HKsFuBStirq/anthropic-s-j-lens-a-research-engineer-s-analysis — [partial]
- Zvi, "No Space Like J-Space" (summary, 7 Jul 2026) — https://www.lesswrong.com/posts/EnxHPxJT4Xin5cTsX/no-space-like-j-space — [partial]
- Neuronpedia hosted lenses: lens files for 40+ open-model directories including `olmo-3-1025-7b` (fit on WikiText-103, final-block target, early-stopped at 568 prompts) and `qwen3.5-0.8b` (233 prompts; J++ version on 96 prompts, J++ recall@10 44.9%) — https://huggingface.co/neuronpedia/jacobian-lens — [full, README and configs]
- Neuronpedia demo page — https://www.neuronpedia.org/jlens — [not read: page body did not render in the fetch]
- Neuronpedia blog post on the J-lens — https://www.neuronpedia.org/blog/jacobian-lens — [not read]
- Hugging Face blog, "J-Space: Yet Another LLM Mind Reader?" — https://huggingface.co/blog/dlouapre/j-space — [not read]
- "J-space debate, agent swarms and pacing frontier AI" (pingback on the R-lens post) — https://www.greaterwrong.com/posts/aXCm8pze46tErTyg4/the-j-space-debate-agent-swarms-and-pacing-frontier-ai — [not read]
- "Hidden Knowledge? Arrr..." (Faber, 18 Sep 2026; R-lens vs logit lens on hidden-knowledge facts; frequency caveat) — https://www.lesswrong.com/posts/crmZvWiadNkWiJfYA — [partial]
- Goodfire: no Goodfire-authored J-lens or residual-stream-computation post was found in two searches; a Goodfire chief-scientist quote appears only in third-party coverage — [not found]
- Alignment Forum: the R-lens post is cross-posted there; no other AF-only item was found — [searched]

Pre-July context items found while searching (not follow-ups)
- "Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology" (exact Jacobian eigendecompositions in three LLMs; arXiv, 14 May 2026) — https://arxiv.org/abs/2605.14258 — [abstract]

### 6.2 Other post-June-2026 results bearing on computation in the residual stream

These items were found and opened by a second reader (one Opus subagent, 4 Nov 2026), whose page summaries came
from a small summarizer model. Numbers below are as it reported them and were not re-checked by me, except the
weight-space programming post, which I opened myself. Status tags are its own.

(a) Working memory, binding, entity tracking
- "Written as a Record, Read as an Address: What a Forward Pass Leaves in an Operation's KV Cache": a reader that sees only operation tokens in the frozen KV cache recovers 0.75–1.00 of queried bindings (≤ 0.06 untrained); payload access at layers 12–15 of 32 (Llama-3.1-8B) (Wu, Shomali; arXiv, 21 Sep 2026) — https://arxiv.org/abs/2609.24635 — [read]
- "One mechanism for many mental spaces: a shared router over a value slot": one DAS subspace controls counterfactual, belief, fiction and time contexts in three model families via a low-rank router in a few late heads (Steele et al.; arXiv, 11 Jul 2026) — https://arxiv.org/abs/2607.10248 — [abstract]
- "In-Context Binding Capacity in Language Models": behavioural; K₅₀ = cN^α with α = 0.820, R² = 0.73 over 12 models ≤ 3B (Ravulapalli, Chadha; arXiv, 24 Sep 2026) — https://arxiv.org/abs/2609.30634 — [abstract]
- "Whack-a-mole with a broken hammer": layer-18 probes in Qwen2.5-1.5B decode an unwritten 6-state automaton state at 88.6% (majority 20%), no causal tests (star2vec; LessWrong, 18 Aug 2026) — https://www.lesswrong.com/posts/LzeZW9PvR6Njc9ngr — [read]

(b) Copying, induction, in-context retrieval
- "A Circuit for In-Context Learning": mid-layer heads write each demonstration's class at label tokens; ablating 20 reader heads in Qwen3-4B costs −0.74 [−0.95, −0.55] nats vs +0.02 for 20 random heads (Gao, Gao; arXiv, 3 Nov 2026) — https://arxiv.org/abs/2611.05344 — [read]
- "Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads": ablating 50 heads in Qwen3-8B drops NoLiMa ROUGE-L 0.401 → 0.000; random heads within 0.05 (Gema et al.; arXiv, 1 Jul 2026) — https://arxiv.org/abs/2607.01002 — [abstract]
- "Induction Heads Interpolate N-Grams": on order-k Markov data induction circuits blend exact and partial matches; BOS adds Dirichlet-like pseudo-counts (D'Angelo et al.; arXiv, 2 Jul 2026) — https://arxiv.org/abs/2607.02800 — [abstract]

(c) Attention sinks, massive activations, registers
- "Which the Eye Fears: Writing with Read-Blindness Explains Massive Activations": read projections ignore massive coordinates (median erasure 0.75/0.65/0.63 vs 0.50 baseline) in Llama 1.28B; read-blindness at ~2.3k steps precedes FFN amplification at ~4.5k (Mukhopadhyay et al.; arXiv, 28 Sep 2026) — https://arxiv.org/abs/2609.35630 — [read]
- "The Life Cycle of a Massive Activation": sink channels differ by seed and settle early; peak magnitude scales ≈ λ^(−1/2) with weight decay (McClendon et al.; arXiv, 30 Sep 2026) — https://arxiv.org/abs/2610.00423 — [abstract]
- "Transient Reserves, Sink Dampers ...": clamping three massive-activation dimensions in Llama-3-8B supports a causal chain to sink attention (Li; arXiv, 10 Jul 2026) — https://arxiv.org/abs/2607.09279 — [abstract]
- Register tokens: only vision-transformer papers found in the window (arXiv 2610.03698, 2610.13299; not opened) — [not read]

(d) Function vectors and task vectors
- "In-Context Learning Amplifies a Latent Symbolic Circuit": in Gemma 2-2B, cross-shot patching raises 0-shot ABA/ABB accuracy 1.0% → 56.0% (random heads 0.5%); a function vector reaches 86% (norm-matched random 0.0%) (Wessel; arXiv, 28 Sep 2026) — https://arxiv.org/abs/2609.36265 — [read]
- "Rules Amortize, Pairings Don't": a 2.6M-parameter net writing a mid-depth residual update into frozen GPT-2-large/XL beats 10-shot ICL on lemmatization (0.78–0.92 vs 0.13–0.48); extracted task vectors ≤ 0.06 (Jhingran; arXiv, 30 Sep 2026) — https://arxiv.org/abs/2610.00526 — [abstract]
- "The imposters among us: function vectors that ace every check and do the wrong task": Llama-3.2-3B month-shift vectors pass stability (cos ≥ 0.98) and causal checks yet mostly output the adjacent month (star2vec; LessWrong, 21 Aug 2026) — https://www.lesswrong.com/posts/aFyir2PaoCHK5prAu — [read]

(e) Weight-space editing and programming
- "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights": GPT-6 Astra, given weights and a Python sandbox but no gradients, edits 131,072 weights (0.016%) in layers 15 and 19 of Qwen3.5-0.8B; MATH-500 7.9% → 48.1% and token-limit hits 93.0% → 33.1% (50-forward-pass version); writes a MinAtar Breakout policy scoring 517.4 vs 25.3 for PPO at 500M steps; fails on Gemma 3 270M (MATH-500 6.8 → at most 7.7). The second reader reports the edit removes one direction in each of the two MLPs (not verified by me). No residual-stream or J-lens analysis on the page (Takehi, Teoh, Zhang, Khattab; MIT CSAIL blog, 2 Nov 2026; not an OpenAI post; no arXiv version found) — https://rikiyat.github.io/blog/weight-space-programming/ — [read, by me]
- torchwright: compiles a Python computation graph into Phi-3-shaped weights with values held in the residual stream; no test results given (Porter; GitHub, undated) — https://github.com/physicsrob/torchwright — [read]
- (pre-July, context) "Targeted Recovery of Weight-Space Mechanisms From Neural Networks" (Vigouroux, Sharkey; arXiv, 19 Jun 2026) — https://arxiv.org/abs/2607.13047 — [abstract]

(f) Latent multi-hop reasoning and serial depth without chain-of-thought
- GPT-6 Astra system card, §9 Monitorability: UK AISI measured a no-CoT 50%-reliability math time horizon of 30.9 min vs 3.6 min for GPT-5.6 Sol (possible contamination); OpenAI writes the no-CoT horizon "may have increased by about an order of magnitude". The card does not mention weight-space programming (OpenAI, 3 Sep 2026; PDF over the fetch limit, HTML read) — https://deploymentsafety.openai.com/gpt-6-astra/monitorability — [read]
- "Estimating GPT-6 Astra's no-CoT Time Horizon": 50% horizon in [8 min, 1 hour] on 37 benchmarks vs 3.0 min for GPT-5.5 (Ward, Gould; LessWrong, 9 Sep 2026) — https://www.greaterwrong.com/posts/ntKx9YHWCwxSeGbRB/estimating-gpt-6-astra-s-no-cot-time-horizon — [read]
- "Reading Between the Dots: Decoding Hidden Computation across Filler Tokens": logit lens on filler-token states in DeepSeek V3 and Kimi K2 recovers intermediate values at 82.1–94.3% (≈ 1% on shuffled tokens); J-lens used as a robustness check (Brauer, Verdun, Marks; arXiv, 3 Jul 2026) — https://arxiv.org/abs/2607.03502 — [read]
- "Not All LLM Reasoning is Visible in the Chain-of-Thought": filler-token gains across 13 models; the second reader could not match the abstract's "up to 13 percentage points" to Table 1 (Baherwani et al.; arXiv, 24 Jul 2026) — https://arxiv.org/abs/2607.22925 — [read]
- "Curriculum Learning Teaches LLMs Long-Horizon Implicit Reasoning, Up To A Limit": no-CoT graph-search lookahead 256 at 98% with curriculum vs ~45% without in Qwen3-0.6B; fitted maximum lookahead ≈ 51 to ≈ 324 across models (Huang et al.; arXiv, 19 Oct 2026) — https://arxiv.org/abs/2610.25612 — [read]
- "Not All Thinking is Created Equal: Latent Reasoning Discovers a Recurrent Search Algorithm": in a 4-layer latent model, an 8-of-20-component circuit (one head retrieves edges; MLP + residual stream update a reachability state) keeps 90.9% output consistency at 7–12 hops vs 55.8% for a random subset (Cheng, Zhang; arXiv, 28 Sep 2026) — https://arxiv.org/abs/2609.35643 — [read]
- "Bridge Routing Heads": ablating a pool of heads (70 in Llama 3.1 70B, 42 in Qwen 2.5 72B) raises two-hop NLL by 0.84–2.07 vs 0.009–0.053 for random heads (Kim et al.; arXiv, 7 Oct 2026) — https://arxiv.org/abs/2610.09733 — [read]
- "Write Facts the Way Models Read Them: Lifting the Two-Hop Curse": patching the model's own bridge-entity representation at early-to-middle layers lifts second-hop accuracy from 0.03–0.06 to 0.20–0.26 in three 7–8B models (Shi; arXiv, 1 Nov 2026) — https://arxiv.org/abs/2611.01519 — [read]
- "No sign of backtracking in latent reasoning": in Huginn-3.5B, top-answer flips across loops have a median gap of 0.12 logits; a per-step Jacobian lens checks the readout (star2vec; LessWrong, 12 Sep 2026) — https://www.lesswrong.com/posts/QxvwPhnvJ5evGT8Hw — [read]
- "An operationalization of opaque serial depth": defines the longest path between text bottlenecks; Gemma3-1B bound 1,954; depth scales with active parameters to ≈ 0.26 power (Sheffield et al.; Redwood Research blog, 10 Sep 2026) — https://blog.redwoodresearch.org/p/an-operationalization-of-opaque-serial — [read]
- (pre-July, context) "Quantifying the Necessity of Chain of Thought through Opaque Serial Depth" (Brown-Cohen, Lindner, Shah; Google DeepMind; arXiv, 10 Mar 2026) — https://arxiv.org/abs/2603.09786 — [abstract]
- (pre-July, context) "The Ignition Index": per-layer probe-accuracy sigmoid steepness across 11 models; Pythia-410M shows a detected change point at training step 256 (+67%), before induction-head formation (Rahbar; arXiv, 26 May 2026) — https://arxiv.org/abs/2608.05160 — [abstract, by me]

Not covered: the second reader found nothing relevant from Goodfire, Google DeepMind (Medium page returned 403) or
OpenAI interpretability posts in the window ("An Alien Mind" returned 403 — [not read]).

---

## 7. Experiment seeds for our assets

Assets: OLMo-3 7B with 16 banked checkpoints (step 0 to 1,413,814); per-neuron top-context banks for 300 MLP
neurons × 2 signs at those 16 points; 4B–7B models on ephemeral AWS GPUs; the MolmoAct2 + LIBERO evaluation
harness; a replicated weight-space edit on Qwen3.5-0.8B.

Facts that shape every seed:
- A J-lens for `allenai/Olmo-3-1025-7B` already exists (Neuronpedia; WikiText-103, final-block target, 568
  prompts). Its file size (1,040,198,095 bytes, fp16) is consistent with 31 source layers of 4096 × 4096. It is
  for the released base model (`main`). On the Hugging Face repo, `stage1-step1413814` is the last of 1,421 stage-1
  (pretraining) branches, followed by stage-2 (mid-training) and stage-3 (long-context) branches, so `main` is not
  our step-1,413,814 weights. The Neuronpedia lens can serve only as an external comparison point; our checkpoints
  need their own fits.
- A J-lens (233 prompts) and a J++ lens (96 prompts) exist for `Qwen/Qwen3.5-0.8B` (d = 1024, 23 source layers)
  on the same Hugging Face repo.
- The released code defaults to the final-block target; the paper's Sonnet lens used the penultimate layer
  (§A.7), and a final-block target can be dominated by one inherited direction (Brauer, §6.1). Fit both targets.
- The repository contains fitting, readout and visualization only. Swap, ablation and gradient pursuit are not in
  the code (searched `jlens/` and `tests/`); we write them from §2.5 and §A.8.
- Fitting cost: one forward and ceil(4096 / dim_batch) activation-only backward passes per prompt. FLOP estimate
  for OLMo-3 7B at n = 25 prompts of 128 tokens: about 3 × 10^17 FLOP, i.e. roughly 0.5–1 GPU-hour on one L40S
  per checkpoint and target layer (estimate from FLOP count, not measured). Ten prompts already beat the logit
  lens (§A.7); Nanda used 25.

Instrument checks shared by all seeds (run once, before any new number):
- **Positive control that must fire:** the France→China multi-function swap (§3.4; countries were the most reliable category, 42/48,
  and the tao-hpu README reports it replicated on Qwen, not verified) — and J-lens pass@10 above logit-lens pass@10 on the `typo` and
  `multilingual` sets in `data/evaluations/` (both replicated on Qwen).
- **Identity pair:** two lenses fit on disjoint prompt halves of the same checkpoint (split-half agreement of
  J_ℓ, of readouts, of CKA) give the noise ceiling; J = I (logit lens) and a fixed random rotation J_rot give
  floors. The step-0 checkpoint is the untrained-weights control.
- **Floors for every causal number:** a matched-norm random-direction swap, a swap between two irrelevant
  same-category tokens, and a direct answer-token substitution.
- **Bank samples:** for each seed, save the prompt, the top-10 lens tokens with probabilities, and the model's
  actual top output for one item scored as a hit and one scored as a miss.

### Seed 1 — When does the J-space band form during OLMo-3 7B pretraining?
- **Claim.** The three band signatures — CKA block structure over J-lens vectors, the rise in readout excess
  kurtosis, and J-lens-vector MLP gain above neuron directions (§4.1, §4.3.1) — appear between two specific
  checkpoints, and two-hop intermediate *readout* appears at an earlier checkpoint than *causal swap* success.
- **Refuted if.** (a) The step-0 lens shows the same block structure and gain profile within split-half noise
  (the structure is architectural, not learned); or (b) every statistic changes smoothly in log-step with no
  interval whose change exceeds split-half noise; or (c) swap success above the random-swap floor appears at the
  same checkpoint as readout or earlier.
- **Cheapest first measurement.** Fit penultimate-target lenses (n = 25) at four checkpoints (step 0, one early,
  one middle, step 1,413,814) plus a split-half refit at the last; compute CKA, effective rank of W_U J_ℓ, and
  MLP gain against isotropic directions. About 3–5 GPU-hours (estimate). No causal runs until the readout curve
  is in hand.
- **Why new.** The authors list this as open (§9.1: "we do not know how much earlier in pretraining it emerges").
  The Pythia workshop study (abstract only) reports early availability and weak workspace functions throughout;
  jeeva2812 released OLMo 3 7B Jacobians at 8 stage-1 steps but at layer 20 and for geometry, not causal use.
  Neither (as far as read) times readout against causal swap success with a step-0 control and split-half noise
  on OLMo-3; our 16 points are denser. Two small-model posts bear on the prediction: GPT-2 small lacks almost all
  of the paper's structure (Cairns et al.), and the presence of a workspace tracks modern pretraining rather than
  size across 124M–1.7B models (Mehta). A Pythia-410M probe-steepness change point at step 256 (Ignition Index,
  pre-July) suggests the earliest checkpoints need dense sampling.

### Seed 2 — Does the J-lens name what a neuron responds to, and since when?
- **Claim.** For the 300 banked neurons, the top-25 J-lens tokens of the neuron's input-weight direction (read as
  W_U norm(J_ℓ w), as the paper does for weights in §A.24.3) overlap the tokens of that neuron's top-activating
  contexts more than permuted neuron–bank pairs do, more than the logit-lens readout of the same weight does, and
  with the correct sign (positive bank ↔ top of the readout, negative bank ↔ bottom). The overlap rises at the
  checkpoint where Seed 1's band forms.
- **Refuted if.** Overlap is at chance level against the permutation control at the final checkpoint; or J-lens
  overlap ≤ logit-lens overlap; or the sign test fails (negative-bank tokens rank as high as positive-bank tokens).
- **Cheapest first measurement.** Weights and banked texts only, no forward passes beyond the lens: one lens at
  the final checkpoint, 300 × 2 readouts, a rank-overlap score, 1,000 permutations. Minutes of compute after the
  lens exists.
- **Why new.** §A.19 counts how often a neuron's best-matching direction is a J-lens vector (~15% before the band,
  60–65% inside) but never checks whether the readout names the neuron's contexts. The two-sign banks give a
  built-in sign control that the paper's weight analyses lack. No follow-up found does this across training.

### Seed 3 — Neuron-level selectivity: which banked neurons depend on the J-space?
- **Claim.** Neurons whose input weights have high J-lens kurtosis κ (§A.18) lose more of their activation on
  their own top contexts under upstream J-space ablation (top-10 J-lens directions per position, band layers)
  than low-κ neurons, and more than under a matched-norm random-direction ablation of the same layers.
- **Refuted if.** The high-κ minus low-κ drop has a 95% interval that includes zero, or for high-κ neurons the
  J-space ablation drop is not larger than the matched random ablation drop.
- **Cheapest first measurement.** Final checkpoint; terciles of κ fixed before any run; forward passes on the
  already-banked top contexts with and without ablation (300 neurons × their bank size × 3 conditions).
- **Why new.** The paper's selectivity evidence is at task level (§3.5.2) and its κ analysis is weights-only
  (§A.18–A.19). A unit-level test of "which computations route through the J-space" on banked activations was not
  found in any follow-up.

### Seed 4 — Is the open-model multi-hop swap an answer-direction effect?
- **Claim (to be refuted).** On OLMo-3 7B, intermediate-swap success on two-hop items is explained by how much the
  intermediate's J-lens vector points along the answer contrast, cos(v_int, v_ans − v_ans′), and not by the
  intermediate's workspace loading.
- **Refuted if.** Success is unrelated to that cosine once loading is controlled, and items with near-zero cosine
  still flip above the random-swap floor and earlier in depth than the answer swap (§3.3's depth test).
- **Cheapest first measurement.** Weights only: compute the cosine for the 50 items in
  `data/evaluations/lens-eval-multihop.json` and the workspace-bench multihop items, after filtering to items the
  model answers without chain-of-thought. Then one GPU run with intermediate swap, answer swap and random swap per
  item, swept over layer bands.
- **Why new.** Nanda (Qwen) and Zeisler (Qwen, Gemma) report aggregate answer-swap dominance (top-1 flips
  6.3–11.1%). No per-item predictor of when an intermediate swap works was found. This separates "variable read
  by a downstream function" from "direction that already carries part of the answer".

### Seed 5 — Slot-dependent arithmetic: computation on a residual variable, and its training onset
- **Claim.** OLMo-3 7B solves `(a + b) * c + d` with small operands without chain-of-thought, and swapping the same
  value v into the J-lens coordinates of the first vs the second intermediate raises f₁(v) vs f₂(v) respectively,
  as in §A.24.2 (63 → 133 vs 70). Across checkpoints, slot dependence appears later than readout of the
  intermediates.
- **Refuted if.** The raised answer does not depend on the slot; or the raised token is v itself (the swap only
  injects an output token); or the effect is not above the matched random swap and the direct answer-token
  substitution. For the training part: slot dependence appears at the same checkpoint as readout.
- **Cheapest first measurement.** Tokenizer check, done 2026-11-04 on `allenai/Olmo-3-1025-7B` `tokenizer.json`:
  the pre-tokenizer splits digit runs into chunks of 1–3 digits, all 1,000 strings "0"–"999" are single vocabulary
  entries, no 4-digit string is, and a leading space is a separate token. So every intermediate below 1000 has its
  own J-lens vector, unlike Qwen, which splits digits and on which J++ could not evaluate arithmetic. (Same
  tokenizer at the 16 pretraining checkpoints is assumed, not verified.) Then filter 100 expressions the final checkpoint answers correctly, locate intermediate layers by readout, and run one
  swap sweep (both slots × 5 values × 3 bands).
- **Why new.** This is the paper's clearest computation-versus-representation test (§5.1 above), and it is only
  shown as one curated example on Sonnet 4.5. Arithmetic failed to replicate on Qwen 3.6 27B (Nanda). No
  open-model or across-training slot-dependence test was found.

### Seed 6 — Do the "workspace" statistics survive outlier-dimension and frequency controls?
- **Claim.** In OLMo-3 7B the band onset (CKA, kurtosis, occupancy) and the J-lens-vector MLP gain stay within
  split-half noise after (a) projecting the top-1 and top-3 massive-activation dimensions out of both activations
  and lens and (b) z-scoring readouts by base-rate token frequency.
- **Refuted if.** Either control moves the band onset by more than split-half noise, or brings J-lens MLP gain down
  into the neuron-direction range (≈ 1), when the same removal of random dimensions does not.
- **Cheapest first measurement.** One checkpoint, one lens, statistics recomputed under each control; then the same
  at step 0 and two mid checkpoints to see whether sink formation and band formation coincide.
- **Why new.** The paper reports no outlier or frequency control. jeeva2812 retracted a J-lens geometry claim
  driven by one direction carrying 53–99.8% of variance and found the sink direction is built by training; Panchal
  found the lens offset tracks token frequency (r = 0.70). Ostrow et al. predict that the lens selects
  depth-amplified, position-coherent directions, which is what a sink direction is. Two September papers time
  massive-activation formation in training (read-blindness at ~2.3k steps before amplification at ~4.5k in
  Llama 1.28B, arXiv 2609.35630; sink channels settle early and differ by seed, arXiv 2610.00423), so the
  sink and the band can be dated on the same checkpoints. This is a validity check on
  §4 that no follow-up has run on a checkpointed model.

### Seed 7 — Does the replicated weight-space edit on Qwen3.5-0.8B act through the J-space, or through stopping?
- **Context.** The public weight-space edit on Qwen3.5-0.8B (Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space
  Programming", MIT CSAIL, 2 Nov 2026; https://rikiyat.github.io/blog/weight-space-programming/) changed 131,072
  weights (0.016%) in layers 15 and 19 of 24, written by GPT-6 Astra (a second reader reports the edit removes
  one direction in each of the two MLPs; not verified by me). Reported: MATH-500 7.9% → 48.1%, GSM8K 13.8%
  → 67.1%, IFEval 21.0% → 62.3% (50-forward-pass version), and the share of answers hitting the token limit
  93.0% → 33.1%. Whether our replication is this edit is assumed, not checked here. The token-limit drop means
  much of the gain may come from the model stopping on time, not from different intermediate content.
- **Claim.** The edit's change to the residual stream (edited minus unedited activation, Δh) at the layers after the first edited block has a
  J-space fraction (k = 16 nonnegative pursuit) above a same-size random frame and above J_rot, and the J-lens
  vectors that carry it name task content (operations, numbers, answer words) rather than termination tokens
  (end-of-turn, "boxed", closing punctuation).
- **Refuted if.** The J-space fraction of Δh is at chance level against both frames; or the top J-lens components of
  Δh are termination tokens and clamping only those coordinates to unedited values removes most of the accuracy
  gain; or Qwen3.5-0.8B shows no band at all (CKA and kurtosis flat), in which case the question is not posed at
  this scale.
- **Cheapest first measurement.** Use the pre-fitted Neuronpedia J-lens (233 prompts) and J++ lens (96 prompts) for
  Qwen3.5-0.8B (no fitting). Forward passes of edited and unedited weights on 100 MATH-500 prompts; Δh
  decomposition per layer; report the top-20 J-lens tokens of Δh with coefficients, and the end-of-turn token's lens
  score at each generated position for both models. Under one GPU-hour (estimate).
- **Why new.** The edit's authors report behaviour only; the page does not mention residual-stream analysis or the
  J-lens (read 2026-11-04). L2W (arXiv 2609.32434) uses J-lens attribution to choose components to cut; reflection
  training (§7) implants J-space content by fine-tuning. No measurement was found of what a direct weight-space edit
  writes into the residual stream. It also answers the paper's open question about small models (§9.1).

### Seed 8 — Is there a verbal workspace in a robot policy, and does it steer actions?
- **Claim.** During LIBERO action generation, a text-fit J-lens on MolmoAct2's language backbone reads the instructed
  target object in the band layers, and swapping its lens coordinates for a distractor object present in the scene
  moves the executed trajectory toward the distractor more often than a matched random swap does.
- **Refuted if.** Target-object tokens do not rise above a shuffled-instruction control in the readout; or the swap
  is not above the random-swap floor while editing the instruction text does redirect the policy (the positive
  control).
- **Cheapest first measurement.** Offline, no simulator: fit a lens on the backbone (n = 25 text prompts), read it at
  the action-generation positions of 20 recorded episodes under two instructions that differ in one object. Only if
  the readout separates, run one closed-loop swap batch on a single LIBERO task.
- **Architecture fact that shapes the design.** MolmoAct2 has two action outputs: the LLM still predicts discrete
  OpenFAST action tokens through its LM head, and the deployed continuous actions come from a flow-matching action
  expert whose every block attends to the key and value tensors of the matching backbone layer (arXiv 2605.02881,
  §4.1–4.2; https://arxiv.org/abs/2605.02881). So a text-fit J-lens covers the LM head only; the executed action
  can read band layers directly through the per-layer KV connection. Fit a second "action lens" from the Jacobian of
  the expert's velocity output with respect to each backbone layer, and compare its band structure with the text
  lens. That comparison directly tests the paper's prediction that models with non-verbal
  output spaces may develop a non-verbal workspace component (§9.3).
- **Why new.** No J-lens study of a vision-language-action policy was found (one search; the nearest are two audio-LLM
  studies, arXiv 2609.33458 and 2608.24958, and VLM "gaze heads" cited in §8 of the paper).

Suggested order: instrument checks → Seed 6 (does the instrument measure what it is named for on OLMo) → Seed 1
readout curve → Seeds 4 and 5 (the computation tests) → Seeds 2, 3, 7, 8.
