# Working theories — "Neural Networks Are Already Clean" × our program

Status: DRAFT v0, 2026-10-19. Working space, not a result. Every claim about our own record cites a file in
this repo; every claim about the essay cites `essay.md` by line (plain-text version fetched 2026-10-19 from
https://already-clean.goodfire.com/essay.md). The essay is taken as true by instruction. Our own results are
quoted with their artifact keys as the closeouts give them. Anything marked INFERENCE is ours and unverified.

Companion files: `IDEAS.md` (inventory, one block per idea, status tags), `CHECKS.md` (what was checked
against the literature and against our artifacts, with the query strings), `DIRECTIONS.md` (the ranked
list, written last).

## 0. How to read this

Five theories. A, B, C are substantive and testable on checkpoints we already cache. D is a generalization
of the essay with a cheap test on our small-LM stack. E is method. §7 gives the cross-links; §8 lists what
the essay does not cover, which is where a result of ours would be new rather than a replication.

## 1. Primitives taken from the essay

- P1 One learning principle: generalization, fractal (recursive generalization). Superposition in the
  "unrelated things packed because they rarely interfere" sense is not observed in any real network studied;
  it is the last hypothesis, not the null. (essay.md 33–35, 201–209)
- P2 Circuit ecology: circuits race for residual error; generalization and composition win because they are
  fastest; gradient starvation kills competitors; rich-get-richer. Generalization starts at step one on
  natural data. (653–671, 707–719)
- P3 New computation is a route between existing circuits, built as close to the output as the task allows.
  Umbrella→bee: nine steps, last three blocks, existing bee neurons switched on, umbrella neurons faded,
  swapping those blocks into the original suffices. Memorization sits near the output. (283, 287–293)
- P4 Rare facts ride related machinery; prediction: the neurons and heads that carry a rare fact are related
  to its content. Consolidation into a neuron is for buildability (readers), not for steering. (767–771)
- P5 A representation matters only through what reads it. Neurons are gates; the negative pole is a
  distinction, context-dependent. (97–101, 219–229, 465)
- P6 LM last layer largely suppresses single tokens ("say or just think"); weak activations matter most
  there and in early vision layers; elsewhere they can be removed at near-zero cost. (305)
- P7 The residual stream can be dominated by one unread coordinate that scales the rest through LayerNorm;
  remove it from the view, not the model. Neuron read/write directions are the right basis for the stream;
  a rotation away from the neuron basis is trivial. (417, 427, 583, 749)
- P8 SwiGLU double-negative regime: ~4% of neurons have their top activations from (−)×(−); ablating the
  regime network-wide raises most benchmarks, lowers some reasoning ones. (373, 571–577)
- P9 Architecture designed for learning (residual, LayerNorm) made late layers legible without any
  interpretability-specific change; small/simplified models are less legible; bigger models have narrower
  neurons. (119–125, 411)
- P10 Experts in MoE are far more specialized than reported and are structurally anti-packing. (379–385)
- P11 Everything of the "higher-level structure" kind (probes, dictionaries, the changes training makes,
  including on-policy self-distillation) decomposes into interpretable neurons. (509)
- P12 Method: neuron papers; blind maximum; explanations that could lose (near misses, most-distant
  concept); attribution graphs over the model's own neurons with no reconstruction loss; watch neurons grow
  over checkpoints; edge motifs; tethered abstraction. (339–345, 387–397, 635–651, 679–685)
- P13 Agents carry a 2023 default reading of "superposition" and could not read past a base64 figure in the
  one note that corrects it. (551, 557)

## 2. Theory A — Adaptation is colonization of existing routes

**Statement.** A fine-tune does not build mechanisms. It captures the new residual error through the
existing circuit that is already most active on the new inputs and closest to the output, by retuning that
circuit's readers (gates, last-layer suppressors). What the parent loses is exactly the original function of
the readers that were repurposed. The set is small, late, and semantically adjacent to the new task.

**Derived predictions.**
- A1 For any parent→child pair, the neurons whose explanation or raw activation range changed are few,
  concentrated in late blocks, and their original jobs are adjacent to the child's task content.
- A2 Restoring that set from the parent (weights or activations) recovers most of the parent's lost
  capability; restoring an equal-count random set or an equal-norm matched set recovers little.
- A3 The "readable trace" of narrow fine-tuning on unrelated text (Minder et al. 2510.13900) is the
  recruited neurons resting slightly on: the mean activation difference, after the norm-scale direction is
  removed, projects onto the write directions of the set in A1.
- A4 Output-format forgetting (flattening, unigram-prior shift) is carried by last-layer suppression,
  entropy and token-frequency neurons; knowledge forgetting is carried by mid-layer gates. The two separate
  by depth and by neuron class.
- A5 Knowledge insulation works because the stop-gradient keeps the race out of the VLM's neurons; a
  neuron-level version (freeze or restore the set in A1) should retain the parent's ability with no old data.
- A6 A narrow fine-tune whose fastest route is a broad pre-existing circuit generalizes broadly (emergent
  misalignment is this with a persona circuit); a context cue that makes a narrow route faster prevents it.
  Susceptibility is predictable from the base model: distance from the task to the nearest broad circuit.
- A7 A new modality grafted onto an LM is read by existing concept neurons (recursive generalization along
  a modality axis), not by new neurons.
- A8 Timing: the parent's loss appears in the step window in which the set in A1 is repurposed, not
  smoothly.

**What our record already says (STAGEDIFF/SINKPATCH, `studies/stagediff/CLOSEOUT.md`).**
- The robot-only fine-tunes lose plain text; DROID's loss is flattening (+2.3 to +2.7 nats) of which a
  logit bias repairs 0.38; LIBERO's is distributed across layers 12–31 and no cheap fix reaches it
  (`studies/nomix/NOTES.md` §3). Reads as A4: DROID mostly format-side, LIBERO mostly knowledge-side.
- Refuted at the subspace level: head and norm, additive activation trace, start-token state, first-content
  token state. None of these is a neuron-level object; A1–A3 are untested.
- The mean activation difference is a mean shift (cosine 0.96 with the child's mean) and projecting it out
  deepens the loss (R −1.11). Under P7 that is the norm-scale knob (Theory B), not a trace.
- Hackett et al. 2608.08904 (per our reading chain, abstract-level): late-layer cliff L28–35 localized to
  late MLP writes. Consistent with A1's "late".

**What refutes A.** Changed neurons many, uniform in depth, unrelated to the task; restoration of the set
recovering no more than an equal-count random set; the loss accruing smoothly across all neurons over
training.

## 3. Theory B — Unread energy

**Statement.** Networks carry large activation energy in directions nothing reads with gain: the LayerNorm
scale knob (P7), the unembedding null-space directions used by entropy neurons, the SwiGLU double-negative
echo (P8), broad early-layer dials. This energy is cheap to delete, conserved under Adam because its
gradient has no consistent sign, moved freely under SGD, and it confounds every energy-ranked instrument
(massive activations, PCA, mean-difference traces, lens fits).

**Our object.** The low-gain band: bottom eigenspace of the consensus Gram of 24 Adam fine-tune updates;
~24% of layer activation energy at width 32; 2.90× cheaper to delete than its energy-matched twin; reader
gain 0.13×/0.26×/0.72× on three objects; SGD moves it 16–22× random; the held-out split was never run
(`papers/highenergy-lowgain/DRAFT.md` abstract; `studies/lineage/STATE_2026-09-04.md` §1–2).

**Derived predictions.**
- B1 The band decomposes into named neuron-level sources: overlap with the norm-scale coordinate's shadow,
  with the bottom singular subspace of the unembedding (late layers), and with the write directions of
  double-negative-dominated neurons.
- B2 Reader gain, computed per neuron (the downstream weight norm applied to a neuron's write direction),
  predicts ablation cost better than activation energy, across layers.
- B3 Published energy-ranked importance claims flip sign under an energy-matched null (the paper's
  protocol), and the directions that flip are the unread ones in B1.
- B4 INFERENCE on Adam: a direction with large raw gradient and no consistent sign has large second moment
  and small first moment, so Adam's step there is small; SGD's is large. This is why the band is
  Adam-conserved and SGD-mobile. Testable with optimizer-state logging (the residual cheap test 3 in the
  lineage closeout §9).

**What refutes B.** The band's overlap with each named source at chance; per-neuron reader gain no better
than energy at predicting ablation cost.

## 4. Theory C — The optimizer referees the race

**Statement.** Which circuit wins the race for residual error depends on the optimizer's per-direction step
sizes. Adam equalizes per-coordinate steps; Muon equalizes per-matrix singular directions; SGD follows raw
gradient. The essay credits legibility to architecture and never varies the optimizer. Our band result shows
the optimizer changes what structure is conserved. INFERENCE throughout.

**Derived predictions.**
- C1 At matched loss, the fraction of neurons with one job (scored by explanations that could lose) differs
  across Adam, Muon and SGD on the same small LM and data.
- C2 Fine-tuning with Muon moves the unread band and other low-SNR directions more than Adam does;
  forgetting at matched task gain differs in a direction the band's ablation cost predicts (cheap-to-delete
  band ⇒ small effect on plain text; the effect, if any, comes from other low-SNR directions).
- C3 Recursive-generalization overgrowth (offshoots) is more common under optimizers that let a neuron
  pick up weak, inconsistent gradient.

**What refutes C.** One-job fraction within seed noise across optimizers; forgetting within seed noise.

## 5. Theory D — Legibility is training health

**Statement.** Anything that improves optimization (residual stream, normalization, scale, data quality,
a better optimizer) increases the fraction of neurons with one job; anything that harms it (no residual,
double-negative nonlinearity, noisy labels, tiny models) produces overgrowth (recursive generalization).
Cleanliness is a training-health diagnostic, and interpretability-motivated architecture changes should be
the ones that also improve loss (P9).

**Derived predictions.**
- D1 A sign-separated SwiGLU (gate and value signs handled so (−)×(−) cannot masquerade as (+)×(+)) trains
  to equal loss and has fewer double-negative-dominated neurons and a higher one-job fraction.
- D2 One-job fraction rises monotonically along a quality/scale ladder on one architecture.
- D3 Label noise lowers the one-job fraction before it raises loss noticeably.

**What refutes D.** D1 changes legibility only at a loss cost; D2 non-monotone beyond seed noise.

## 6. Theory E — Instrument discipline

Two rules the essay and our closeouts independently arrived at:
- E1 Match the null on the quantity the object has (our paper), and read the network in its own units
  (the essay). Every dead item in our value audit violated one of the two.
- E2 A radical hypothesis whose only symptom is "this doesn't make sense" is not a hypothesis (our CLAUDE.md
  "the observation that would refute it"; the essay's "superposition of the gaps").
- E3 Agents' priors are an instrument defect (P13): briefs must carry the essay's definitions; keep our
  no-metaphor rule (name the measure, not the motif) while adopting its claims.

## 7. Cross-links

- A×B: STAGEDIFF's mean shift is B's knob, which is why projecting it out hurt and why SINKPATCH had
  nothing to transplant (parent/child sink cosine ≥ 0.99989) while a random vector there destroyed the
  model (R −3 to −8).
- A×C: knowledge insulation and optimizer choice are two ways of deciding who wins the race; A5 and C2 are
  the same experiment with different levers.
- B×D: the double-negative echo is both an unread-energy source (B1) and a training-health defect (D1).
- A×P4: the "smallest thing to keep" (NOMIX) and "where does a rare fact live" are one question: the
  readers of the recruited route.

## 8. What the essay does not cover (where our results would be new)

- Optimizers (Theory C) — not mentioned once.
- Fine-tuning as the object of study: one experiment (umbrella→bee) at 7M; nothing on LM fine-tunes,
  model diffing, forgetting, or traces.
- Training checkpoints at LM scale: it says they lacked them (essay.md 813). OLMo-class releases have them.
- Unlearning and emergent misalignment: not mentioned.
- Any VLA or robot policy; any "interpretable intermediate" (DEPTHANCHOR's object).
- An independent audit of the cleanliness estimate by anyone outside Goodfire.

## 9. Corrections after verification (2026-10-19, see CHECKS.md)

- A6 (emergent misalignment as a race) is corroborated by the literature, not open: 2607.21356 (pre-existing
  persona subspace, removing it cuts EM to 0.0%), 2606.20814 (pre-fine-tuning activations predict post-FT
  alignment), 2602.00298 (base-model membership inference predicts domain susceptibility). Keep as support
  for A; drop as a direction.
- A4's entropy-neuron clause has a published null for instruct-tuning (2504.02904: overlapping entropy
  neurons nearly identical in base and post-trained models). The robot hops differ in sign and size
  (+2.3–2.7 nats of flattening), so the clause stays as a hypothesis with a lowered prior, and the
  token-frequency and null-space components are tested separately.
- A has a named competitor: diffuse erosion (Locate-then-Merge 2505.16703: many small changes erode the old
  ability). The discriminating observation is in I1: recovery from the few most-changed neurons versus from
  the long tail at matched parameter mass.
- A1's "few, late, pre-existing" is published for the *gained* behavior (refusal gate from pre-existing
  discrimination structure, 2605.12290; instruction following in sparse MLP subsets, 2604.13694). The
  unpublished half is the *lost* ability and its restoration.
- C2 (Muon fine-tunes forget more) is published (2605.10468; 2605.06654 optimizer-model consistency). C1
  (legibility by optimizer) remains open with two published hints (optimizers shape activation statistics and
  effective rank: 2605.06654, 2606.09658).
- P8's regime is documented before the essay (GLUScope 2602.23826); the benchmark effect of ablating it is
  the essay's.
- P10 is replicated in the literature (2604.02178, ICML 2026). Not a direction.
- P6 is corroborated by TEAL/CATS (2408.14690, 2404.08763). Not a direction.
- A7 has a precedent with a frozen LM (2308.01544); the open case is a trained LM with its parent available.
