# Ideas inventory — v1 after literature batches 1–4 (2026-10-19)

Status tags: OPEN (no prior work found on the specific claim), NARROWED (part is published; what remains is
stated), KILLED (published, or a verification rather than a direction), FINISH (already ours; complete as
planned). Cost classes: W weights only · F forward passes on cached checkpoints · T training runs · A needs
the agentic neuron-paper pipeline (I0). Every live idea names the observation that refutes it. All prior-work
statements are ABSTRACT-level unless CHECKS.md says otherwise; nothing is quotable as a citation until read at
source for the specific claim. Virality notes are judgments.

## I0 — Shared instrument: an agentic neuron-paper pipeline for 4B–7B open models (prerequisite)

Per neuron: top-k activations on a long-context corpus; raw activation range and sparsity (not percentiles);
negative pole; strongest upstream and downstream weights; explanation by agent 1; blind maximum by agent 2
who sees only the explanation; five positives and five near misses scored; a most-distant-concept control;
SwiGLU sign-regime split (GLUScope-style). Reuse: Transluce's explainer and pipeline (Llama-3.1-8B, 2024),
Neuronpedia autointerp, output-centric descriptions 2501.08319, NeuronScope 2601.03671 for polysemantic
cases, RelP attribution 2601.22594. Decision after reading those at source. Scale on our lineage:
Qwen3-4B has 36 × 9,728 = 350,208 MLP neurons (HF config read 2026-10-19).

## Live ideas

### I10 — Recursive generalization at LM scale with public checkpoints — OPEN — F, A
The essay's developmental claim (branches trace to one local origin and appear in order) was shown on
retrained InceptionV1 and on a few LM neurons; it says checkpoints for large LMs were lacking (essay.md 813).
`allenai/Olmo-3-1025-7B` exposes 1,421 stage-1 checkpoints (HF API read). Sample neurons at the final
checkpoint stratified by depth and by final-checkpoint legibility class; run I0 at a log-spaced subset of
checkpoints; for each neuron, cluster its top activations at every checkpoint and test whether later
clusters descend from earlier ones (shared upstream circuit, shared tokens or semantics) in order.
Refuted if a substantial fraction of neurons acquire top-activation clusters with no traceable antecedent
(the superposition prediction), or if branching order is random with respect to training time.
Nearest prior: crosscoder across Pythia checkpoints for the unembedding (2610.21025); attention-circuit
timing (2606.02378); persona vectors forming within 0.22% of OLMo-3 pretraining (2605.13329). No
neuron-level development study found on OLMo or Pythia.
Virality: very high. It is the test the essay calls decisive and the one critics (Nanda, Gorton) are asking
for, on a 2025 open 7B model, with a figure anyone can read (a neuron's branching tree over training).

### I11 — Independent audit of the cleanliness estimate on 4B models — OPEN — F, A
Random neurons from Qwen3-4B and gemma-3-4b-pt (two families); explanations that could lose; report
one-job / two-related / not-understood fractions with the essay's own caveats (thresholds, weak activations,
raw vs percentile); bank the double-negative share and the last-layer single-token-suppression share.
Refuted if the controls do not separate (near misses and most-distant texts fire as often as positives);
reported either way.
Nearest prior: none; explainers list "independent validation: none yet" four days after publication.
Virality: very high and short-lived; also the calibration step for I0, so it costs little beyond I0.

### I1 — Where the lost ability went: sparse repurposing vs diffuse erosion — NARROWED — F, A
Parent and child share neuron indices. Run I0 on both on plain text; rank neurons by change in explanation,
raw range and top readers (sibling pair Molmo2-4B vs Molmo2-ER as the floor on the robot hops). Then two
restorations at matched parameter mass: the top-k changed neurons (in/out rows or per-neuron activation
patch) vs the long tail. Theory A predicts the top-k recovers most of the lost plain-text ability and the
tail little; Locate-then-Merge's diffuse-erosion hypothesis (2505.16703) predicts the reverse.
Refuted (for A) if the tail restoration at matched mass recovers as much or more; refuted (as a method) if
restoration of either recovers less than the DROID logit bias already does (0.38).
Pairs in cache: Qwen3-4B-Base→Qwen3-4B→Instruct-2507; Qwen3-4B-Instruct→Molmo2-4B / -ER; MolmoAct2→DROID /
LIBERO; MolmoAct2-Pretrain→MolmoAct2; gemma-2-2b→-it; gemma-3-4b-pt→-it; paligemma lineage;
zephyr-7b-beta→Zephyr_RMU.
Published: the gained side (refusal gate from pre-existing structure 2605.12290; sparse MLP subsets
2604.13694; keystone neurons 2605.24846). Unpublished: the lost side with explanations attached and the
sparse-vs-diffuse contrast.
Virality: high. "Model diffing needs no crosscoder: the neurons are already shared" plus "forgetting is k
neurons, here are their names" is a one-figure claim every lab with a parent/child pair can check.

### I7 — Does the optimizer set the cleanliness regime? — OPEN — T, A
Same small LM, same data and token budget, AdamW vs Muon vs SGD(+momentum), compared at matched loss; score
one-job fraction (I0), double-negative share, and activation sparsity. Published hints: optimizers have
regularization effects on activations and differ in weight spectra (2605.06654); Muon reaches higher
effective rank than Adam at the same loss (2606.09658); optimizer mismatch forgets more (2605.10468). No
legibility measurement by optimizer exists. Our band result (Adam-conserved, SGD-mobile subspace) is the
mechanism candidate.
Refuted if one-job fraction is within seed noise across optimizers at matched loss.
Virality: high. Muon is the optimizer story of 2025–26; the essay never varies the optimizer; the result
travels either way ("Muon models are less legible" or "optimizer does not matter for legibility").

### I2 — The narrow-finetuning trace is recruited neurons resting on — OPEN — F, A
ADL mean difference on first tokens of unrelated text; remove the mean-activation (norm-scale) direction
(STAGEDIFF: cosine 0.96, projecting it out deepens the loss); project the residual onto the layer's neuron
write directions; the top neurons should be I1's changed set with task-related explanations; sibling floor.
Refuted if the projection spreads across hundreds of neurons with no overlap with I1's set.
Published: ADL (ICLR 2026) with the authors' overfitting hypothesis; no independent mechanism paper
(2605.25902 is grey-box recovery). Runs on I1's pairs.
Virality: medium-high; answers the stated open question in Neel's lane.

### I3 — Robot fine-tune flattening vs mid-layer gating — NARROWED (lowered prior) — W, F
DROID: +2.3–2.7 nats, bias-repairable 0.38; LIBERO: distributed over layers 12–31. Candidate carriers from
Stolfo et al. 2406.16254: entropy neurons (final layer, write into the unembedding's low-rank null space,
act through final LayerNorm scale) and token-frequency neurons (unigram prior). Published null for
instruct-tuning: 2504.02904 found entropy neurons nearly identical in base and post-trained models. The
robot hop is opposite in sign and far larger, so test rather than assume; the frequency-neuron component
matches the bias-repairable share.
Refuted if entropy and frequency neurons in the DROID child are unchanged against the sibling floor, or
restoring them moves wikitext NLL less than the calibration bias. Cheap first cell of I1 on the robot hops.
Virality: medium; clean figure (two neuron classes, two robot fine-tunes) if it lands; informative null if
not (it would push the loss into the mid-layer gates for DROID too).

### I5 — Modality as a generalization axis — NARROWED — F, A
Qwen3-4B-Instruct→Molmo2-4B. For the neurons most active on image tokens in Molmo2, compare the text
explanation in parent and child. Prediction (A7): same explanation with image-side activations added; new
image-active neurons that were dead or unrelated in the parent are rare.
Refuted if image-active neurons are mostly ones whose text explanation changed or vanished.
Published: multimodal neurons in a frozen LM (2308.01544) and in LLaVA/MiniGPT-4 (ACL 2024 Findings,
2406.11193, 2503.22941). None has the parent of a trained-LM VLM; we do.
Virality: medium-high; "image concepts are grown onto the text neurons that already meant them" is crisp.

### I4 — Watch the route form, then insulate it — NARROWED — T, A
NOMIX cell 3a re-fine-tunes MolmoAct2 on LIBERO. Save dense checkpoints; track I0 summaries per neuron; A8
predicts the plain-text loss appears in the window where I1's set is repurposed. Second run: freeze or
restore the set with no text rehearsal; score against the 1% text-mix arm at equal steps (2502.06042).
Refuted if the loss accrues smoothly across neurons, or the frozen arm retains no more text than no-mix, or
costs LIBERO success beyond the seed band.
Published: neuron freezing for safety (SafeNeuron 2602.12158; 2508.09190), Subspace Tuning 2610.22293
(weight singular directions), Locate-then-Merge (post hoc). Unpublished: diff-derived set on a VLA against a
compute-matched text-mix arm, and the timing claim.
Virality: medium; decision-relevant for VLA labs.

### I6 — Decompose the low-gain band into named neuron-level sources — OPEN — W
Overlap of the band with (a) the top-energy coordinate's input-space shadow, (b) the unembedding's bottom
singular subspace in late layers (Stolfo's null space), (c) write directions of double-negative-dominated
neurons, (d) write directions of entropy neurons; per-neuron reader gain vs activation energy as predictors
of ablation cost. Adjacent: Subspace Tuning 2610.22293 (different object).
Refuted if overlaps are at chance and reader gain predicts no better than energy.
Virality: low-medium; it is what turns the band paper's description into a mechanism.

### I14 — Architecture for legibility at equal loss — OPEN — T, A
Sign-separated SwiGLU vs SwiGLU on the small-LM stack at matched loss; score double-negative share and
one-job fraction. Nearest: GLUScope 2602.23826 (analysis), MemGLU 2608.07323 (closed-tail gating),
bilinear MLPs 2410.08417. Refuted if the variant costs loss or does not move legibility.
Virality: medium.

### I17 — Unlearning as reader disconnection (zephyr-7b-beta vs Zephyr_RMU) — OPEN — F, A
RMU is representation-level and updates few layers (WMDP paper, ABSTRACT); no neuron-level analysis found.
Prediction from A: forget-set neurons still fire; their readers moved. Refuted if the neurons stop firing.
Virality: medium.

## Low priority or finish as planned
- I8 DEPTHOPEN perturbation-in-training arm — design improvement, OPEN, T.
- I12 rare facts ride related circuits — essay's own prediction, not ours.
- I16 attention heads broad and recursively generalized — essay's own program.
- I21 DEPTHANCHOR/POSTCHECK — FINISH. Behavior-level precedents exist (DeepThinkVLA 2511.15669 mask-CoT
  barely matters; Ablate-to-Validate 2605.21642); our mechanism (gate at sigmoid(−4) never trained) is the
  contribution.

## Killed
- I9 EM as a race / predict susceptibility from the base model — published (2607.21356, 2606.20814,
  2602.00298). Theory A6 stands as corroborated.
- I15 MoE experts anti-superposition — published (2604.02178 ICML 2026).
- I3b post-training overconfidence = entropy neurons — published null (2504.02904).
- I19 weak-activation ablation — corroborated by TEAL/CATS; verification, not a direction.
- I13, I18, I20 — folded into I3, E3, I4.
