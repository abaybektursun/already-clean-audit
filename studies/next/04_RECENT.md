# 04_RECENT — recency and competitor scan, July to November 2026

Opus agent for the Fable session, 2026-11-05. Scope: work that overlaps neuron development across public
pretraining checkpoints of a 7B-class model, dating a job's birth, the route at birth, attribution of the
birth to documents, and transplants between checkpoints or post-training stages.

**Answer first.** No one has published neuron-level job dating, route-at-birth, or neuron-level document
attribution on a 7B-class model across checkpoints. Searches ran through 2026-11-04. The Goodfire companion
paper is not out. The essay itself says it had no language-model checkpoints. The closest published work is
at 1B scale or below, or uses the whole model, the residual stream or attention heads rather than MLP neurons.
The gap is narrowing on three sides:
1. OLMo-3 checkpoints are now used at whole-model level. One paper measures stage-1 windows anchored at
   `stage1-step1413814`, the lab's own last checkpoint (2610.37719). Another retrains OLMo-3-7B mid-training
   on edited data (2610.06851). A third party has released per-checkpoint Jacobian lenses for OLMo-3-32B.
2. A mature head's parameters were transplanted into an early Pythia checkpoint. It recovers 35 to 68
   percent of the later gain (2610.04119).
3. A neuron-level circuit was tracked over training in Pythia-1b. Its final form is 18 neurons and one head
   (2610.33299).

**Correction to the brief.** "Neural Networks Are Already Clean" is by Nick Cammarata (Goodfire). Its byline
reads "With contributions from Curt Tigges (language neuron research)". Gabriel Goh does not appear as an
author. He appears only as a cited co-author of Multimodal Neurons (2021)
(https://already-clean.goodfire.com/, fetched 2026-11-04).

**How the scan was run.**
- arXiv API keyword queries restricted to submissions from 2026-06-15 to 2026-11-06: 30 queries.
- Semantic Scholar citation lists of 14 seed papers. Citing papers dated from 2026-06-01 were kept.
- Hugging Face API searches for lens repositories and Goodfire releases.
- GitHub API for repository dates.
- About 40 web searches.

**What the read tags mean.**
- *full*: methods and results sections were read from the arXiv HTML or PDF, and the rest was searched by
  keyword.
- *abstract only*: only the abstract or the repository card was read.
- *not read*: the source was seen only in search results.

Downloads are in
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/recent_in/`.

---

## 1. Goodfire: the companion paper and any neuron or checkpoint data

**Finding.** The technical companion paper has not been published, and Goodfire has released no neuron data
for OLMo or Qwen. Checks: Goodfire research page, arXiv author search for Tigges and Cammarata, Goodfire
Hugging Face organization, Goodfire GitHub organization, all on 2026-11-04. The essay says outright that it
watched no language-model neuron grow over training.

- *Neural Networks Are Already Clean* — Nick Cammarata, Goodfire; contributions from Curt Tigges and Tarun
  Gogineni — posted 2026-10-15; footnote revised 2026-10-30 — https://already-clean.goodfire.com/ (markdown:
  https://already-clean.goodfire.com/essay.md).
  - Models: Qwen3-32B neuron audit; Kimi K3 deferred to the companion paper; InceptionV1 and ConvNeXt
    retrained to watch neurons form.
  - Numbers: about 90 percent of Qwen3-32B poles have one main job, about 7 percent two related jobs, 2 to 4
    percent not understood. The double-negative SwiGLU echo "is the whole top of the list" for about 4 percent
    of neurons.
  - On language-model checkpoints: "for something like a huge language model we don't have the checkpoints to
    watch it happen", and "We haven't looked over training". The second sentence is about experts one scale up.
  - On the companion paper: "A technical companion paper will offer a more detailed account of language model
    neuron experiments, such as ... the different regimes of modern non-linearities and what kinds of circuit
    motifs show up at different depths."
  - The essay has neuron-only attribution graphs on Kimi K3 (section "Exploring Language Models").
  - [full]
- Cammarata on X, on the companion paper: "theres a much more standard technical paper coming soon" and "the
  technical paper we're still finishing up". Week of 2026-10-15; dates taken from a mirror that shows only
  weekday and time — https://twitter.modelconduct.org/s/67 (source posts, for example
  https://x.com/nickcammarata/status/2111158258748813817). [abstract only, via the mirror]
- *Are Neural Networks Already Clean?* — Evidence Press — 2026-10-20 —
  https://evidencepress.org/articles/neural-networks-already-clean/ — "Goodfire's companion technical paper has
  not yet been published". [abstract only]
- Goodfire research index — https://www.goodfire.com/research, fetched 2026-11-04. The last item is the essay
  (2026-10-15). It lists no paper on neurons, checkpoints or OLMo for July to November. [full]
- Goodfire Hugging Face: datasets `reward-hack-data` (2026-10-08), `molmobot-wrist-so2-activations`
  (2026-08-13), `r1-collect` (2025); models are SAEs from 2025 only. No neuron or checkpoint release —
  https://huggingface.co/Goodfire, API listing 2026-11-04. [full]
- Goodfire GitHub: 26 most recent repositories. None holds language-model neuron data. The org forked `olmes`
  and `datatrove` on 2026-06-10 and 2026-06-05 — https://github.com/goodfire-ai, API listing 2026-11-04. [full]
- *Anatomy of Post-Training: Using Interpretability to Characterize Data and Shape the Learning Signal* —
  Bergen, Bhalla, ..., Lubana (Goodfire) — 2026-06-10 — https://arxiv.org/abs/2606.12360; blog
  https://www.goodfire.com/research/predictive-data-debugging.
  - Models: DPO on the Dolci preference set from the SFT checkpoint, with Olmo 3 7B, Olmo 3.1 32B and Llama 3.1
    8B to 70B.
  - Measured: SAE concepts of the preference data predict what DPO promotes, R² = 0.9 (blog).
  - Post-training stage only; no neuron dating. [abstract only]
- Third-party test: *already-clean-replication* — GitHub user AMX07 — 2026-10-16 —
  https://github.com/AMX07/already-clean-replication.
  - Model: a 4-layer SwiGLU TinyStories model with checkpoints every 5,000 steps.
  - Measured: entrants to each neuron's top-30 contexts against the previous top-30.
  - Result: "From step 5k on, most changing neurons narrowed and sharpened instead". Branching is listed as
    "Not observed".
  - Echo-dominated neurons are 0.0 to 0.7 percent, against the essay's about 4 percent. [full]
- Third-party plan: *Track N: the MLP's neurons across training* — ZachData, PR #304 — 2026-11-02 —
  https://github.com/ZachData/MetastableStateAnalysis/pull/304 — Pythia-410m at 18 checkpoints. A plan only;
  "Not designed, downloaded or run". [full]
- Commentary with no new checkpoint data:
  - *Neural networks are not already clean* — leogao — 2026-10-17 —
    https://www.greaterwrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean — argues for
    downstream KL divergence simulation scoring. [full]
  - *Is GPT 2XL already clean?* — Jasper Blank — 2026-10-25 —
    https://www.greaterwrong.com/posts/drMv4SM2gaxk3oz3m/is-gpt-2xl-already-clean — about 8 percent of a
    50-neuron sample explained at correlation ≥ 0.8. [full]

## 2. Ai2 and the OLMo team

**Finding.** Ai2 has published no interpretability work on OLMo-3 checkpoints in this window (Ai2 blog
2026-10-07 to 2026-11-03; arXiv author queries). Third parties now use the OLMo-3 checkpoints, the stage
chain and the mid-training data, at whole-model, behavior or document level. This scan found no 2026 Ai2
release of the OLMo-3 batch order. The data-order question belongs to `00b_DATA_ORDER.md`.

- Ai2 blog, fetched 2026-11-04 — https://allenai.org/blog — nine visible posts from 2026-10-07 to 2026-11-03.
  None is on interpretability, OlmoTrace, checkpoints or data order. June to September is behind pagination
  and was not seen. [full for the visible page]
- *What a Weight Update Buys: Leverage Across Training Stages* — Shikhar Shiromani — 2026-10-27 —
  https://arxiv.org/abs/2610.37719.
  - Checkpoints: Olmo-3-7B stage-1 windows anchored at `stage1-step1413814` with widths from 814 steps to 1M
    steps; also `stage1-step1272000`→`main`, Instruct and Think SFT, DPO and RLVR, and five RL-Zero runs.
  - Measured: Fisher quadratic form of each update divided by that of random directions of the same size,
    called leverage.
  - Numbers: pretraining windows 0.78 to 0.95; SFT 5.4 to 5.7; RL-Zero and RLVR 388 to 1,508; DPO 3,304 to
    6,165 on Olmo-3. From 30k steps on, stage-1 windows are at random level on prose.
  - Narrow stage-1 deltas are 64 percent MLP, 26 percent attention and 10 percent embeddings by squared norm.
  - Whole-model only; nothing per neuron. [full]
- *Base Models Can Reason By Taking a Cue From Training Data* — Wang, Dravid, Shao, Farhat, Min, Efros —
  2026-10-05 — https://arxiv.org/abs/2610.06851.
  - Method: retrains Olmo-3-7B on the released 10B-token mid-training mix with words substituted; also
    SmolLM3-3B.
  - Numbers: ".\n\nOkay" raises base MATH-500 pass@1 from 42 to 78 percent. After the "Okay"→"Chicken" edit,
    ".\n\nChicken" raises MATH-500 from 2.4 to 37.2 percent.
  - A counterfactual data intervention at 7B, read at behavior level. [full]
- *Questioning Instrumental Convergence in RLVR: Checkpoint Evidence from OLMo-3* — Alparslan, Tanwisuth,
  Petterson — 2026-10-22 — https://arxiv.org/abs/2610.31624.
  - Checkpoints: OLMo-3-7B-Think SFT, DPO, 13 intermediate RLVR checkpoints and the final model.
  - Instrumental rate: 25.0 percent at SFT, 46.1 percent near RLVR step 200, 11.8 percent final.
  - Behavior only. [abstract only]
- *Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models* — Kou et al. — 2026-10-22 —
  https://arxiv.org/abs/2610.30882 — "wait" arrives with OLMo-3 mid-training data. Output-level indices only.
  [abstract only]
- *Capability Provenance in Language Models: A Case Study in Social Reasoning* — Matlin, ..., Quirke,
  Biderman, Riedl — 2026-06-17 — https://arxiv.org/abs/2606.19625.
  - Method: TrackStar via Bergson. Corpus gradients come from OLMo3-7B Base over a 10.5B-token Dolma3 working
    set, aggregated into 576 WebOrganizer bins; query gradients come from OLMo3-7B Instruct.
  - Unlearning the top bins hurts the matched benchmark more than random bins.
  - Document level; no intermediate checkpoints. [full]
- *Tracing Persona Vectors Through LLM Pretraining* — Moskvoretskii et al. — 2026-05-13 —
  https://arxiv.org/abs/2605.13329 — OLMo-3-1025-7B at 17 checkpoints; residual-stream directions form by 0.22
  percent of tokens. [read by the lab, `studies/already-clean/checks/literature_W5.md` §4c; not re-read]
- *When Probing Accuracy Saturates, Fragility Resolves* — Reblitz-Richardson — 2026-06-09 —
  https://arxiv.org/abs/2606.11375 — 20 stage-1 OLMo-3 7B checkpoints; linear probes plus noise fragility.
  [abstract only]
- *Analyzing Occupational Gender Bias across the LLM Training Lifecycle* — Gadhia, Hoyle, Stańczak —
  2026-10-16 — https://arxiv.org/abs/2610.23587. [not read]

## 3. Timaeus and developmental interpretability

**Finding.** Nothing at 7B scale and no neuron or feature birth-timing study appeared July to November 2026.
The one billion-parameter singular-learning-theory study in the window is on Pythia-1B and is not by Timaeus.

- Timaeus research index — https://timaeus.co/research, fetched 2026-11-04.
  - July to November items: *Elicitation without Backpropagation* (2026-07-21, https://arxiv.org/abs/2607.18804)
    and *Patterning the Toy Model of Superposition* (listed October 2026).
  - The largest language model in any 2026 item is Pythia-1.4B (*Spectroscopy at Scale*, April 2026).
  - [full for the index]
- *Patterning in Practice: Debiasing Reward Models with Susceptibilities* — Wang, Donoway, Murfet — 2026-09-01
  — https://arxiv.org/abs/2609.00699 — no checkpoints of a pretrained model. [not read]
- *Bridging Singular Learning Theory and Mechanistic Interpretability at the Billion-Parameter Scale* — Ji,
  Wang, Sun — 2026-10-31 — https://arxiv.org/abs/2611.00640.
  - Setup: Pythia-1B at all 154 checkpoints; a five-stage Dyck bracket circuit; Fisher spectrum of 144
    components at every checkpoint.
  - Number: changes of the Fisher trace track changes of causal effect with median partial rank correlation
    0.51.
  - [abstract plus model and checkpoint sections]
- *Pre-carved Niches: The Formation Dynamics of Modular Task Partitions in Early LLM Training* — Li, Li —
  2026-09-01 — https://arxiv.org/abs/2609.01170.
  - Setup: Pythia-410M trained from scratch, with attribution patching at every step over 14 tasks.
  - Numbers: two sharp jumps; winners receive 2.25 to 2.73 times the loser's gradient supply.
  - A 2.8B run is pre-registered. [abstract only]
- No Hoogland-authored arXiv paper on language-model checkpoints in the window (arXiv author query,
  2026-11-04).

## 4. Crosscoders and SAEs across checkpoints

**Finding.** Feature-birth work in 2026 is on Pythia up to 410M with SAEs. On OLMo-2-7B it covers readout
(unembedding) features. RL-checkpoint SAEs are at 1.5B. None works in the MLP neuron basis or at 7B over
pretraining.

- *Features have life history. And we should care* — Stecher, Radovanović, Sikimić, Kahle — 2026-05-07 —
  https://arxiv.org/abs/2605.18789.
  - Setup: SAE features matched across Pythia-160M and -410M checkpoints.
  - Numbers: features emerge, die and reorganize about 40 times faster in the first 1 percent of training.
    About 50 "carrier" features are fixed by then. They recruit 64 percent of active features by the end.
  - Future carriers are predicted from onset firing in 4 of 5 cases. [abstract only]
- *Learning to Read Out: Unembedding Dynamics in Language Model Pretraining* — He, Shen, Iacob, Jovanovic,
  Qiu, Lane — 2026-10-14 — https://arxiv.org/abs/2610.21025.
  - Setup: one dictionary fit to readout rows across 32 checkpoints of Pythia-160M, 1B and 6.9B and of
    OLMo-2-7B.
  - No MLP neurons. Search coverage summarizes it as "most OLMo-2-7B features emerge late".
  - [read by the lab, `checks/literature_W5.md` §4a; not re-read]
- *What Does an LLM Learn from Reinforcement Learning? ... Fixed-SAE Track* — Du, Tang, Duan, Liu —
  2026-09-14 — https://arxiv.org/abs/2609.15064.
  - Setup: one SAE on activations pooled over the base and 13 GRPO checkpoints, Qwen2.5-1.5B-Instruct layer 27.
  - Results: drift is small and late-layer. Steering the changed features into the base recovers about 80
    percent of the RL gain. [full]
- *Understanding On-Policy Distillation ... via Sparse Crosscoders* — Yu, Ye, Wang, Zou — 2026-09-28 —
  https://arxiv.org/abs/2609.35210 — on-policy distillation leaves over 98 percent of frequently used student
  features within 20 percent of their firing rate. [abstract only]
- *Comparing Latent Concept Formation in State Space Models and Transformers via Sparse Autoencoders* —
  Nagaraj et al. — 2026-09-21 — https://arxiv.org/abs/2609.24440 — Mamba-130m against Pythia-70m at the final
  checkpoint; 99.98 percent of features align. [abstract only]
- *Tracing Subliminal Learning Through Pretraining* — Tang, Anand, Paulo — 2026-10-22 —
  https://arxiv.org/abs/2610.30884 — Pythia, PolyPythia and DataDecide OLMo checkpoints; preference transfer is
  measured as behavior, not features. [abstract only]
- *Sparse Autoencoder Attribution Graphs Are Decided at the Margin* — Karl Li — 2026-10-31 —
  https://arxiv.org/abs/2611.00606 — on Pythia up to 160m, zeroing one feature moves a next-layer feature across
  the TopK cutoff in 0.74 or more of cases. [abstract only]
- Before the window, already known to the lab: *Crosscoding Through Time* (Bayazit, Mueller, Bosselut;
  https://arxiv.org/abs/2509.05291; Pythia-1B, OLMo-1B, BLOOM-1B) and *Evolution of Concepts in Language Model
  Pre-Training* (Ge et al., ICLR 2026; https://arxiv.org/abs/2509.17196; Pythia crosscoders). Semantic Scholar
  shows 3 and 1 citing papers since June 2026; none extends them to 7B. [abstract only]

## 5. Neuron-level studies across checkpoints

**Finding.** The 2026 neuron-level developmental papers are at 1B scale or below. One tracks a neuron-level
agreement circuit in Pythia-1b and 410m. One finds a reader-writer sign per MLP channel that forms within a
few percent of training. One tracks closed-gate damping over OLMo-1B training. No 2026 follow-up to Gurnee's
universal neurons was found: Semantic Scholar lists 16 citing papers since June 2026, and none is a
universality study.

- *The Emergence and Sparsification of a Syntactic Circuit* — Yao, Boguraev, Pimentel, Mahowald — 2026-10-23
  — https://arxiv.org/abs/2610.33299.
  - Setup: pythia-1b and ten seeds of pythia-410m across training. RelP neuron attribution plus neuron-to-head
    routing at each checkpoint.
  - Results: agreement is learned within 2 percent of training and is first carried by a diffuse set of
    neurons. The circuit then shrinks and reroutes between heads. Final pythia-1b circuit: 18 neurons and one
    head. [full]
- *Reader–Writer Coupling in Transformer MLPs* — Oh, Ahn, Yu — 2026-10-21 — https://arxiv.org/abs/2610.29340.
  - Setup: a per-channel pairing coordinate from the exact square Jacobian of the MLP update. GPT-2 small and
    medium trained with three seeds each, released OLMo-1B checkpoints, Pythia 70m to 6.9B.
  - Results: each block's strongest pairs are opposed; the writer points against its reader. The sign is set
    "within a few percent of training" in a fixed depth order shared across five GPT-2 seeds, before the
    coupling grows. [full]
- *Closed Gates Leak: Decomposing Adjacent-Layer Cancellation in Gated Language Models* — Dam, Gopalakrishnan
  — 2026-10-28 — https://arxiv.org/abs/2610.39068.
  - Setup: ten gated models plus GELU and ReLU controls; OLMo-1B checkpoints from 2B to 3T tokens.
  - Results: closed-gate neurons carry the damping. The adjacent-layer cosine turns negative between 33B and
    64B tokens. In OLMo-1B the closed side carries 66 percent of the damping.
  - This bears on the double-negative regime and on N15. [full]
- *Comparing Sparsified MLPs and Transcoders: Many Transcoder Features Are Neurons in Disguise* — Shabalin,
  Piotrowski — 2026-10-17 — https://arxiv.org/abs/2610.24521.
  - Setup: 13 open transcoders on Gemma, Qwen and Llama.
  - Results: "neuron features" are a large share of active late-layer features. They are the only features
    stable across transcoder training runs.
  - Final checkpoints only. [abstract plus result sections]
- *Grammatical "grandmother neurons" are rare in LLMs* — He, Mesgarani — 2026-09-24 —
  https://arxiv.org/abs/2609.29328 — the "seven checkpoints" are seven models, not training steps. [read by the
  lab, `checks/literature_W5.md` §4d]
- *Revenge of Monosemanticity: Neuron Specialization as a New Form of Feature Learning in MLPs* — Abedsoltan,
  Boix-Adsera, Kalogiannis, Belkin — 2026-08-25 — https://arxiv.org/abs/2608.24007 — theory and small-MLP
  regression; no language model. [abstract only]
- *Lagged Coupling: Internal Representations Become Readable Before They Become Causal* — Xining Xun —
  2026-09-01 — https://arxiv.org/abs/2609.01048.
  - Setup: Pythia 160M to 12B at eight checkpoints, four task families; residual-stream probe against steering
    on the probe direction.
  - Results: steering effects are at chance level in 43 of 48 cells; read-before-write in 11 of 11 units.
    A pre-registered OLMo-2 replication at two sizes is scored "weak". [full]
- Instrument note from two independent papers. Some public Pythia-2.8B checkpoint labels hold byte-identical
  weights: step54000 to step64000 carry the bytes of step143000 (2610.29340 Appendix B; 2609.01048 Appendix A).
  Both papers gate every checkpoint on content hashes.
- Before the window, known to the lab: *Deconstructing Pre-training* (Wang et al.,
  https://arxiv.org/abs/2601.08383, AAAI 2026). OLMo-7B and OLMoE neuron importance ranks on relational
  facts; dense OLMo top-1 percent Jaccard overlap peaks at 10.2 percent around 70K steps. [read by the lab,
  §4d]

## 6. Transplants between checkpoints or stages; what RL changes in the weights

**Finding.** Published transplants move attention heads (Pythia up to 1B; OLMo-2-7B and Tulu-3-8B RL against
DPO), LoRA updates (OLMo-3-32B stages), or SAE-feature steering vectors. None moves MLP neurons between
adjacent pretraining checkpoints or between OLMo-3-7B stages with a count- and energy-matched random control.
The RL weight-update papers agree on a few points:
- RL updates are small and sparse in coordinates, and most of the gain sits in a few late-layer features
  (2609.15064).
- They are not low-rank beyond a noise baseline (2610.22438).
- They have high leverage per unit norm: 388 to 1,508 times random on Olmo-3 (2610.37719).

Sources:
- *Copying Before Suppression: What Drives a Below-Chance Dip During Language Model Training?* — Dahiya,
  Blondin — 2026-10-02 — https://arxiv.org/abs/2610.04119.
  - Setup: one mature suppressing head's query, key, value and output weights copied into the early
    below-chance checkpoint (Pythia 160M, 410M, 1B). The control is a bundle of eight random heads. OLMo-1B is
    used only for an activation-level check.
  - Result: the copy recovers 35 to 68 percent of the end-of-training gain in logit difference. [full]
- *Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism* — Gupta, Zhang, Jin —
  2026-10-16 — https://arxiv.org/abs/2610.22937.
  - Base to aligned: copying the base model's query and output weights for the cue heads into the aligned
    model barely changes caving.
  - RL against DPO, in Tulu-3-8B and OLMo-2-7B: swapping the attention patterns of six middle layers shifts the
    suggested-answer cue effect "most of the way" to the other checkpoint's level. [full]
- *Pre-training interventions, ex post facto: Grafting model beliefs across checkpoints* — Nutter, Roytburg,
  Dumas, Ou, Feng — 2026-09-30 — https://arxiv.org/abs/2610.00767.
  - Setup: a LoRA adapter trained on synthetic documents at the base checkpoint is added to the post-trained
    model. Tested on Qwen3-14B, Llama-3.3-70B and DeepSeek-V4-Flash, and across the OLMo-3-32B Instruct and
    Think chain (SFT, DPO, RL-50, RL-1150, final).
  - Result: only the adapter fit before instruction tuning keeps coherence on every checkpoint. Adapters fit
    later lose 17 to 29 points of decisiveness (61.3 to 31.9–44.2). [full]
- *Understanding LLM Parameter Update Sparsity through the Lens of Fisher* — Zhang, Mukherjee, Peng —
  2026-09-28 — https://arxiv.org/abs/2609.36262 — fixed sparse masks chosen from the initial diagonal Fisher
  keep a large share of the RL and on-policy distillation gain. [abstract only]
- *GRRR: The Geometry of Reshaping, Rotation, and Routing in Decoder LLM post-training* — Qi, Tang, Zhu —
  2026-08-26 — https://arxiv.org/abs/2609.22146 — 12 SFT and RL chains; removing the change of singular values
  keeps most math gains. [abstract only]
- *Rank Measures of Fine-Tuning Updates Need a Noise Baseline* — Altaf — 2026-10-15 —
  https://arxiv.org/abs/2610.22438 — 13 released pairs up to 8B; counted rank of updates is only a little
  below a random matrix of the same shape. [abstract only]
- *Sparsity Curse: Understanding RLVR Model Parameter Space from Model Merging* — Wu et al. — 2026-06-16 —
  https://arxiv.org/abs/2606.18521 — sparse RLVR updates are near-orthogonal across runs, and standard merging
  degrades them. [abstract only]
- *Context-Grounding Gains Are Mediated by Pre-existing Machinery: Auditing GRPO, SFT, and DPO* — Gupta, Gupta
  — 2026-09-01 — https://arxiv.org/abs/2609.00925 — adding the starting model's grounding direction recovers 35
  percent of the DPO gain. [abstract only]
- *RLVR Sharpens the Pencil, Not the Mind* — Bangera, Gupta — 2026-11-02 — https://arxiv.org/abs/2611.02574 —
  OLMo-2-1B at 13 RLVR checkpoints; error-detection probes stay at AUROC 0.798 to 0.840. [abstract only]
- *Can Capability Enhancement be Localized?* — Nguyen, Bair, Kolter — 2026-10-15 —
  https://arxiv.org/abs/2610.21294 — small sets of heads whose removal raises task accuracy. [abstract only]
- *Where Do Distilled Students Store Thinking?* — Kim, Jung, Kim, Lee — 2026-11-03 —
  https://arxiv.org/abs/2611.04141. A LoRA subspace for thinking-like behavior is grafted at twice its weight.
  Same-energy random directions often nearly match it. [abstract only]
- *Localizing RL-Induced Tool Use to a Single Crosscoder Feature* — Shportko et al. — 2026-06-25 —
  https://arxiv.org/abs/2606.26474 — Qwen2.5-3B; passive transfer to base +6.8 ± 5.0 pp. [abstract only]
- Prior, before the window: Panigrahi et al. 2023 grafting (https://arxiv.org/abs/2302.06600). Semantic
  Scholar lists 12 citing papers since June 2026; none grafts across pretraining checkpoints.

## 7. Neuron-level or feature-level training-data attribution

**Finding.** The closest causal result is Mechanistic Data Attribution (January 2026, Pythia): removing
high-influence samples shifts when interpretable heads emerge. At 7B, the published work is behavior-level
(OLMo-3-7B mid-training word substitution) or document-bin level (OLMo3-7B TrackStar). Feature-to-document
links at scale are correlational. Two 2026 papers report that gradient-similarity attribution tracks answer
format more than content. That is a direct confound for N4's cosine test.

- *Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units* — Chen, Luo, Pan —
  2026-01-29 — https://arxiv.org/abs/2601.21996.
  - Setup: influence functions on the Pythia family.
  - Result: removing or augmenting a small set of high-influence samples changes the emergence of
    interpretable heads; random interventions do not. [abstract only]
- *Base Models Can Reason By Taking a Cue From Training Data* — 2026-10-05 — https://arxiv.org/abs/2610.06851.
  Counterfactual mid-training edits on Olmo-3-7B; see item 2. [full]
- *Capability Provenance in Language Models* — 2026-06-17 — https://arxiv.org/abs/2606.19625. OLMo3-7B,
  benchmark-level; see item 2. [full]
- *Data Attribution of Emergent Misalignment with Persona Features* — Vetter, Kaczér, Flek, Mai — 2026-08-11 —
  https://arxiv.org/abs/2608.11025.
  - Setup: SAE diffing on four open models; features attributed to one million web documents.
  - Result: fine-tuning on the retrieved documents does not reliably induce the behavior. [abstract only]
- *Measuring Task-Agnostic Training Data Influence Across Language Model Pretraining* — Nishida et al. —
  2026-08-13 — https://arxiv.org/abs/2608.13515 — 18 Pythia and PolyPythia configurations. Influence is
  measured as movement toward the final parameters; literature data leads early, STEM data later.
  [abstract only]
- *Form Over Content In Gradient-Based Data Attribution Methods* — Kim, Jung, Kim, Oh, Oh — 2026-09-17 —
  https://arxiv.org/abs/2609.19589.
  - Result: gradient alignment follows answer format. Cosine is near 0.4 for a shared format and near 0.0 for
    the same task in another format, from the earliest pretraining checkpoints on. [abstract only]
- *Robust LLM Data Selection via Content-Invariant Gradient Attribution* — Gu, Pang, Shen — 2026-10-27 —
  https://arxiv.org/abs/2610.36755 — gradient attribution scores change under surface-form edits.
  [abstract only]
- *Symbolic Mechanistic Data Attribution* — Habibi, Lee, Seif El-Nasr — 2026-06-28 —
  https://arxiv.org/abs/2606.29171 — 200 SFT pairs on Llama-3.2-3B attributed to an SAE-feature policy.
  [abstract only]
- *Tracing LLM Behavior to the Training Data with Empirical Next-Token Distributions* — Izzo — 2026-07-15 —
  https://arxiv.org/abs/2607.14306. [abstract only]
- *RL-IF: Tracking Non-local Training Data Influence for Online Reinforcement Learning* — Liu et al. —
  2026-10-20 — https://arxiv.org/abs/2610.28883 — control benchmarks and an RLHF toxicity setting.
  [abstract only]
- *Concept Influence* — FAR.AI — February 2026 — https://www.far.ai/blog/concept-data-attribution-02-2026 —
  attributes behavior through SAE features. [not read]
- No "OLMoTrace features" or neuron-level OlmoTrace extension was found.

## 8. Jacobian lenses across checkpoints

**Finding.** Two more OLMo-3 lens sets exist beside jeeva2812, and one of them spans training. The OLMo-3-32B
set covers six stage-1 checkpoints, `main`, and Think SFT, DPO and final, at every layer. A separate OLMo-3-7B
set covers the 11 post-training variants with no pretraining checkpoints. Pythia lenses at 14 training steps
are public for six sizes.

- `senku21x/olmo3-32b-jacobian-lens-checkpoints` — 2026-10-13 —
  https://huggingface.co/senku21x/olmo3-32b-jacobian-lens-checkpoints.
  - Checkpoints: OLMo-3-1125-32B at `stage1-step5000`, 10000, 30000, 100000, 300000 and 656000, plus `main`,
    think-sft, think-dpo and think.
  - Fit: source layers 0 to 62; 10 WikiText prompts per lens.
  - Observations, as stated in the card: the first layer whose lens resembles the logit lens (CKA ≥ 0.8) moves
    14 / 17 / 23 / 26 / 30 / 39 / 41 from 5k to `main`. Layer-16 top readout axis grows from about 2 to 46
    percent of readout energy. [full card]
- `mhough/olmo3-jacobian-lenses` — 2026-07-22, updated 2026-10-22 —
  https://huggingface.co/mhough/olmo3-jacobian-lenses.
  - Lenses: OLMo-3-1025-7B base plus Instruct SFT, DPO and final, Think SFT, DPO and final, and four RL-Zero
    models; layers 0, 3, ..., 30.
  - Claimed: a quantitative "post-training point of view" result and an ignition test. No pretraining
    checkpoints. [abstract only, card]
- `danorel/Olmo-3-7B-Instruct-jpp-lens` — 2026-10-23 — https://huggingface.co/danorel/Olmo-3-7B-Instruct-jpp-lens
  — J++ lens, Instruct only. [card]
- `vincentoh/olmo-3-7b-jlens` — 2026-07-07 — https://huggingface.co/vincentoh/olmo-3-7b-jlens — placeholder,
  "Nothing is uploaded yet". It states Neuronpedia's set has only the OLMo-3-1025-7B base. [card]
- `jeeva2812/olmo3-jlens-checkpoints` — 2026-09-02 — https://huggingface.co/jeeva2812/olmo3-jlens-checkpoints.
  - Files beyond the layer-20 lenses: `analysis/eigen_training.json` has layer 8, 16 and 24 eigen statistics at
    11 points from `stage1-step0` to `main`, including `stage2-step47684` and `stage3-step5000`.
    `analysis/birth.json` dates 64 layer-20 readout directions against chance 0.125.
  - My count from that file: first crossing at `stage1-step512000` for 12, `stage1-step1413814` for 16,
    `stage2-step47684` for 34, and `stage3-step5000` for 2. The mean "captured" value is 0.519 at the end of
    stage 1 and 1.0 at `main`.
  - The field meanings are inferred from names; the card does not define them. [full files]
- `jspace-emergence-paper/pythia-jlens-fits` — 2026-09-03 —
  https://huggingface.co/jspace-emergence-paper/pythia-jlens-fits — Pythia 70m to 6.9B at steps 0, 16, 256,
  512, 1000, 2000, 3000, 4000, 16000, 40000, 70000, 100000, 130000 and 143000. The card is empty. Probably the
  workshop study the lab lists in `01_JLENS.md` (Dabir et al.), not confirmed. [file list only]
- `AnonymousInterpScience/pythia-jlens-artifacts` — 2026-09-01 —
  https://huggingface.co/AnonymousInterpScience/pythia-jlens-artifacts — corpus and context dependence of the
  lens on the Pythia 70m to 2.8B ladder; final checkpoints. [card]
- `01_JLENS.md` already lists the arXiv Jacobian-lens follow-ups. No new arXiv paper fits lenses across
  pretraining checkpoints beyond the lab's list (arXiv query for "Jacobian lens", "J-space" and "global
  workspace", 2026-06-15 to 2026-11-06).

## 9. Named researchers and the Transformer Circuits thread

**Finding.** None of the named researchers published work on neuron or feature development over training
from July to November 2026. The Transformer Circuits thread in the window has no training-dynamics item.

- Transformer Circuits index, fetched 2026-11-04 — https://transformer-circuits.pub/:
  - June 2026 update (turn-averaged SAEs).
  - *Verbalizable Representations Form a Global Workspace* (Gurnee et al., 2026-07-16,
    https://arxiv.org/abs/2607.15495).
  - *Characterizing interference weights in a tiny language model* (Turner, Wu, Batson, 2026-08-21,
    https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html).
  - October 2026 update: the J++ lens (Ayonrinde, Lindsey,
    https://transformer-circuits.pub/2026/october-update/index.html).
  - None concerns checkpoints. [full for the index and the October update]
- Wes Gurnee: the workspace paper above, and *A Shared Subcircuit Lets LLMs Count Down Across Tasks*
  (Dunefsky, Gurnee, Ameisen; 2026-07-14; https://arxiv.org/abs/2607.12279; Llama-3.1-70B-Instruct, final
  model). [abstract only]
- Jack Lindsey: the J++ lens (October update). [full]
- Lee Sharkey: *Interpreting Language Model Parameters* (Bushnaq, ..., Sharkey; arXiv 2026-10-14,
  https://arxiv.org/abs/2610.20748; Goodfire blog 2026-05-05) and *Targeted Recovery of Weight-Space Mechanisms
  From Neural Networks* (Vigouroux, Sharkey; 2026-06-19; https://arxiv.org/abs/2607.13047). Final weights only.
  [abstract only]
- Jesse Hoogland, Timaeus: see item 3; nothing on language-model checkpoints in the window.
- Neel Nanda: no 2026 post on neuron development found. He said he would like to see the weak-activation work
  "in Goodfire's forthcoming paper" (https://x.com/NeelNanda5/status/2110875974296654062, week of 2026-10-15,
  via the mirror in item 1). [abstract only]
- Chris Olah: no 2026 post on watching features form found (transformer-circuits.pub index; web search).
- Gabriel Goh: no 2026 paper found (arXiv; web search). See the correction at the top.
- Related position paper: *Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics* —
  Biderman, Khan, Mireshghallah, Arnett, Barez, Saphra — 2026-06-03 — https://arxiv.org/abs/2606.06533.
  [abstract only]

---

## Overlap with our cards

Each line says whether anyone has published the same measurement on a 7B-class model.

- **N1, birth order in weights** (input rows, then output column, then readers, at dated births): none found.
  Nearest:
  - 2610.29340, per-channel reader-writer sign forming in the first few percent of training, up to Pythia-6.9B
    and OLMo-1B, not tied to job births (https://arxiv.org/abs/2610.29340).
  - 2610.37719, whole-model Fisher leverage of OLMo-3-7B stage-1 windows anchored at the lab's
    `stage1-step1413814` (https://arxiv.org/abs/2610.37719).
  - Phrases searched: "neuron evolution during pretraining", "input weights output weights over training",
    "reader writer neurons training", "weight drift checkpoints neurons", "row column weights checkpoints".
- **N2, use at birth** (attribution of a pole to the loss at its birth): none found at neuron level or at 7B.
  Nearest:
  - 2609.01048, residual-stream probes readable before steering is causal, Pythia 160M to 12B, eight
    checkpoints, 11 of 11 units, OLMo-2 replication "weak" (https://arxiv.org/abs/2609.01048).
  - 2608.15323, functional sufficiency of masked concepts over checkpoints, models up to 410M plus OLMo-1B
    (https://arxiv.org/abs/2608.15323).
  - Phrases searched: "readable before causal", "silent period neuron", "when do concepts become functionally
    sufficient", "attribution patching across checkpoints", "use versus readability over training".
- **N3, route at birth** (neuron-basis attribution graph over training): none at 7B. Nearest:
  - 2610.33299, RelP neuron attribution and neuron-to-head routing over training in pythia-1b and ten seeds of
    pythia-410m; final circuit 18 neurons and one head (https://arxiv.org/abs/2610.33299).
  - Neuron-basis circuits at a final checkpoint: 2601.22594 (https://arxiv.org/abs/2601.22594).
  - Phrases searched: "attribution graph across training checkpoints", "circuit formation pretraining
    neurons", "transcoder checkpoints", "circuit tracing over training".
- **N4, documents that made the job:** none found at neuron level. Nearest:
  - 2601.21996, influence functions shift head emergence in Pythia (https://arxiv.org/abs/2601.21996).
  - 2610.06851, counterfactual OLMo-3-7B mid-training edits, behavior-level (https://arxiv.org/abs/2610.06851).
  - Confound to control: 2609.19589, gradient similarity follows answer format
    (https://arxiv.org/abs/2609.19589).
  - Phrases searched: "which training examples created this feature", "feature-level influence", "data
    attribution for SAE features", "neuron-level training data attribution", "OLMoTrace features",
    "mechanistic data attribution".
- **N5 / N13 / N16, transplant** (k neurons between adjacent checkpoints or OLMo-3 stages, against a
  count- and energy-matched random set): none found. Nearest:
  - Head-parameter transplant from mature to early Pythia 160M to 1B, 35 to 68 percent, against eight random
    heads (https://arxiv.org/abs/2610.04119).
  - Attention-pattern swap between RL and DPO checkpoints of OLMo-2-7B and Tulu-3-8B
    (https://arxiv.org/abs/2610.22937).
  - LoRA graft across the OLMo-3-32B stages (https://arxiv.org/abs/2610.00767).
  - SAE-feature steering recovering about 80 percent of the RL gain at 1.5B (https://arxiv.org/abs/2609.15064).
  - Phrases searched: "transplant neurons between checkpoints", "checkpoint grafting", "skill localization
    pretraining checkpoints", "what RL changes in the weights 2026", "sparse subnetwork RL update", "RL updates
    localized neurons", "neuron-level merging".
- **N7, state neurons by invariance:** none found, at any scale with checkpoints. Nearest: a countdown
  subcircuit in Llama-3.1-70B at the final checkpoint (https://arxiv.org/abs/2607.12279). Phrases searched:
  "neurons track bracket depth quote state list index", "counting neurons over training", "positional neurons
  training dynamics", "content-invariant neurons".
- **N20, census of births for every pole of a 7B model:** none found. Nearest:
  - SAE feature life histories in Pythia-160M and 410M (https://arxiv.org/abs/2605.18789).
  - Readout features in OLMo-2-7B (https://arxiv.org/abs/2610.21025).
  - Neuron importance-rank overlap in OLMo-7B (https://arxiv.org/abs/2601.08383).
  - Phrases searched: "feature birth", "feature lineage", "feature genealogy", "when do features form", "neuron
    specialization over training", "monosemantic neurons emerge", "polysemantic neurons training dynamics".
- **N21, two architectures on the same data:** none found for development. Nearest:
  - Olmo 3 against Olmo Hybrid token-level loss at final checkpoints (Li, Merrill, 2026-06-18,
    https://arxiv.org/abs/2606.20936).
  - Behavior-level emergence order consistent across four model families, ρ = 0.81 (Liu et al., 2026-04-09,
    https://arxiv.org/abs/2604.08510).
  - Phrases searched: "Olmo Hybrid checkpoints interpretability", "universal neurons across checkpoints",
    "feature universality across architectures training", "transformer versus hybrid development".
- **N22, arithmetic variables over training on OLMo-3:** none found. Nearest: Pythia workshop study (Dabir et
  al., in `01_JLENS.md`) and 2609.01048, both residual-level and not arithmetic slots. Phrases searched:
  "multi-step arithmetic intermediate value patching checkpoints", "slot dependence over training", "latent
  reasoning faithfulness along training trajectories", "when does the model use intermediate result OLMo".
