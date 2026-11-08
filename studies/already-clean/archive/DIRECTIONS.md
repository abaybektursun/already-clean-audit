# Directions — ranked, 2026-10-19

Ranking criterion: asymmetry (cost × decisiveness × size of the update either way) and then reach in 2026
(judgment). Every item has a first decisive cell and the observation that kills it. Evidence levels per
CHECKS.md: all prior work ABSTRACT-level; our own numbers carry their artifact keys in the closeouts.

## Tier 1 — run first

**D1. The developmental test of "already clean" on a frontier-style open model (I10 + I11 + I0).**
Claim under test: neurons grow by recursive generalization, so every late offshoot has a traceable antecedent
and branches appear in order over training. Substrate: `allenai/Olmo-3-1025-7B`, 1,421 stage-1 checkpoints.
Deliverables: an open neuron-paper pipeline with the essay's tests; an independent cleanliness estimate on
Qwen3-4B-Base and gemma-3-4b-pt; per-neuron branching trees over training. Spec: `SPEC_D1.md` (draft,
2026-10-19): 300 neurons per model audited, 400 OLMo poles followed over 16 stage-1 checkpoints, a blind
parent test against a decoy as the decisive readout. Kill: the unrelated share of late-born clusters has lower
bound above 0.20 (rules in SPEC_D1.md §6.3). Why asymmetric: cheap relative to
any decomposition method (no training), and the update is large in either direction; it is the evidence the
field is explicitly asking for.
**Status 2026-10-19 evening: W0 done** (`SPEC_D1.md` §16, `verify/VERIFY.md`). The instrument is built,
tested with a failing input per gate, and ran end to end at toy scale on real OLMo checkpoints: the explain →
write → probe loop discriminates explained poles from shuffled ones, the judge's positive control is near
1.0, and every blind-protocol refusal fired when it should. Spec amended (§0b stance, §0c exploration track,
`S` = 600, corpus dedupe and window-0 exclusion, G1 model, measured budget ≈ 57 GPU-h). Next: W1-D/W1-C
harvests (three 50M runs), W2-D with prompt v2 on the discovery set, then the waves in order.

**D2. Where the lost ability went: sparse repurposing vs diffuse erosion (I1 + I2 + I3).**
Claim: forgetting after a fine-tune is the original function of a few repurposed readers, concentrated late,
semantically adjacent to the new task; the narrow-finetuning trace is those neurons resting on. Competitor
with a published name: diffuse erosion (Locate-then-Merge). First cell (weights + a few forwards, no agents):
on MolmoAct2→DROID, rank neurons by parameter and activation change; restore top-k vs the long tail at
matched mass; score wikitext NLL against the 0.38 the logit bias already recovers; in the same job read the
entropy and token-frequency neurons (Stolfo) against the sibling floor. Second cell adds I0 explanations and
the ADL projection. Kill: tail restoration ≥ top-k at matched mass; or neither beats the bias. Why
asymmetric: all checkpoints cached; one job decides between two named hypotheses; the positive result is a
no-crosscoder model-diffing method.

**D3. Does the optimizer set the cleanliness regime? (I7)**
Claim: at matched loss, AdamW, Muon and SGD produce different one-job fractions and double-negative shares.
Substrate: the small-LM stack. First cell: one architecture, one token budget, three optimizers, two seeds,
I0 on 300 neurons each. Kill: fractions within seed noise. Why asymmetric: the essay never varies the
optimizer; two published hints (activation regularization, effective rank) point to a difference; Muon is
what labs are adopting; a null is also publishable as "legibility is architecture, not optimizer".

## Tier 2 — run on the same instrument

**D4. Image concepts grow onto the text neurons that already meant them (I5).** Qwen3-4B-Instruct→Molmo2-4B;
identity of image-active neurons' text explanations across the hop. Kill: most image-active neurons changed
or vanished text meaning. First VLM-formation claim with the parent in hand.

**D5. Watch the route form, then insulate it (I4).** Dense checkpoints on NOMIX cell 3a; timing of the text
loss against repurposing of D2's set; freeze/restore arm vs 1% text mix. Kill: smooth accrual, or no
retention gain, or LIBERO cost beyond the seed band. Answers NOMIX's question with a data-free recipe.

**D6. Unlearning as reader disconnection (I17).** zephyr-7b-beta vs Zephyr_RMU, both cached. Kill:
forget-set neurons stop firing.

## Tier 3 — cheap completions

**D7. Decompose the low-gain band into named sources (I6).** Weights only; makes the band paper mechanistic.
**D8. Sign-separated SwiGLU at equal loss (I14).** Small-LM stack; architecture claim with matched loss.
**D9. Finish DEPTHANCHOR/POSTCHECK (I21)** with the one-line mechanism; add the perturbation-in-training arm
to DEPTHOPEN (I8) if it runs.

## Not directions (verified)
Emergent misalignment prediction from the base model (published 2607.21356, 2606.20814, 2602.00298);
MoE expert specialization (2604.02178); entropy neurons as the post-training confidence mechanism (null,
2504.02904); weak-activation ablation (TEAL/CATS); Muon fine-tuning forgets more (2605.10468, 2605.06654).

## Order of work
D2's first cell (no agents) and I0's calibration (I11's first 100 neurons) in parallel; then D1's first cell;
D3 once I0 is stable. Literature read at source before any spec is frozen, for the papers CHECKS.md names
on that idea.
