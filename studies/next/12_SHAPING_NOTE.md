# 12 — What the Goodfire shaping preprint gives our program

Literature note. Opus agent for Fable, 2026-11-05. No GPU, no model run, no commit.

Source: T. Fel, M. Kowal, M. Feiszli, S. Grant, A. Lee, C. Earls et al. (senior: T. McGrath, O. Lewis), "Shaping &
Monitoring Models through their Internal Representations", Goodfire preprint, 82 pages, read in full (§7). "p. N"
is the page number printed on the PDF page; it equals the PDF page index. A value marked "read from the plot" was
read off a figure and is approximate. Every other number is from the text, a table or a caption.

**Answer.** One finding bears on our work directly. In OLMo-3-7B after DPO, five consecutive transformer blocks (of
32) were copied from a penalized run into the unpenalized run. The copy recovered up to 76 % of the penalized run's
sycophancy reduction, and kept the six-benchmark mean within 0.6 points (pp. 60–61). The copied blocks were the ones
whose weights differed most between the two runs. This is a parallel to N13, not support for its premise: it is a
whole-block transplant between two DPO runs from one SFT checkpoint, with a penalty at a known layer and no random
control, and not a neuron-level transplant of an RL gain. None of the paper's experiments uses a pretraining or
mid-training checkpoint, a data mix, or the neuron basis (§6), so this paper does not cover M1.

## 1. What the paper establishes

**1.1 The shaping losses.** A specification is a constraint on a nonnegative violation ω(θ) measured on internal
representations. The model minimizes the task risk R(θ) subject to ω(θ) ≤ ε. In practice it minimizes
L(θ) = R(θ) + λ·ω(θ), where a finite λ need not enforce the constraint exactly (p. 4, Eq. 1).

- *Expression shaping* (p. 5, Eq. 2). ω is the expected value, over prompts and over trajectories sampled from the
  policy, of the sum over tokens t of γ^(t−1)·ρ(z_t). Here z_t is a reader's concept score at token t (a linear
  probe, a projection on a difference of means, or a featurizer latent), ρ is a nonnegative function, and
  γ ∈ (0, 1] discounts later tokens. The gradient (Eq. 3) has two terms. The policy-gradient term changes which
  trajectories are sampled. The second ("attached") term changes the score of a fixed trajectory, and exists only
  when gradients flow through the reader. Detaching the score keeps only the first term; with zero task reward
  this is RLFR (p. 5; Fig. 15, p. 19). The reader can be the current policy, a frozen copy of the initial policy,
  or a separate model (p. 5).
- *Reliance shaping* (p. 5, Eq. 4; p. 6). The model is split at one layer into an encoder z = f_θ(x) and a scalar
  readout s_θ(z), such as a class logit or a preference margin. ω is the mean of ν(z, v)², the squared attribution
  of the readout to a unit concept direction v. Seven choices of ν are given (p. 6): gradient ⟨∇s(z), v⟩;
  gradient × input ⟨∇s(z), v⟩⟨z, v⟩; integrated gradients; SmoothGrad; noise s(z + ξv) − s(z); shuffle, which
  replaces the v-coordinate with another example's; deletion s(z) − s(z without its v-coordinate). For an affine
  readout s = ⟨w, z⟩ + b, each vanishes if and only if ⟨w, v⟩ = 0, and all select the same task-optimal solutions
  (Proposition 5, pp. 6, 43).
- *Direction.* v is the normalized difference of means (DoM) of activations with and without the concept. It is
  recomputed every n steps ("refresh") as the representation moves (Fig. 2, p. 7; p. 8).
- *Conditional reliance* (p. 12). Each penalty term is multiplied by the indicator that a context latent is active.
- *Implementation.* Gradient penalties use double backpropagation; a PyTorch snippet is on p. 42.

**1.2 Models, checkpoints, layers.**

| setting | model | training | layer | pages |
|---|---|---|---|---|
| shortcut, vision | DINOv2 (384-d CLS, frozen or trainable); ResNet-50 in App. L | own Waterbirds from CUB and Places: 2,066 training birds, class and background perfectly correlated; 3 seeds | last layer and block 9 | 8, 80 |
| conditional rules | DINOv3 Large, frozen; BatchTopK SAE, 300 concepts, L0 16 | MLP readouts on ImageNette | 9 | 12, 52 |
| medical shortcut | RAD-DINO, frozen, 768-d; Top-k SAE, 12,288 features, k = 64 | 1,152 CheXpert radiographs, drain predicts label perfectly | output tokens | 13 |
| sycophancy | OLMo-3-7B (details below); OLMo 2 13B; OLMo 3 32B | DPO | 17 of 32 (7B) | 15–18, 58–61 |
| reward hacking | Qwen3.5-9B | GRPO on Countdown, 150 updates, 512 rollouts per update, groups of 32; chess transfer on 512 Lichess puzzles | 19 | 19–20, 62 |
| harmfulness | Mistral-7B-Instruct; Llama-3-8B-Instruct (abliterated) | LoRA rank 64, REINFORCE, 50 updates of 32 harmful prompts; one seed per configuration | 19 | 21, 63 |

OLMo-3-7B in detail:

- *Release.* The post-trained Instruct line at its SFT stage, named "OLMo-3-7B-Instruct-SFT" (p. 58). Not the
  base model. No revision or commit hash is given.
- *Training.* The authors' own DPO run. One epoch on the 124,942 single-turn `delta_learning` pairs of
  `allenai/Dolci-Instruct-DPO`; each pair has a Qwen3-32B chosen reply and a Qwen3-0.6B rejected reply. 975
  optimizer steps at global batch 128, TRL `sigmoid_norm` loss, β = 5, maximum length 4,096 tokens, seed 42,
  learning rate 10⁻⁶ with 10 % warmup and linear decay (p. 58). The Figure 14 run is a single run with the
  Gradient × Input penalty at λ = 0.3 (p. 58). Figure 13 sweeps λ ∈ {0.1, 0.3, 0.5, 0.7, 1} (p. 17). The block
  swap and the per-benchmark figure use λ = 0.5 (pp. 60–61).
- *Layer.* The residual stream after decoder block 17 of 32, on the first 1,024 tokens of the chosen and the
  rejected sequence, averaged over the top 10 % of completion tokens in each sequence (p. 58).
- *How the direction was found.* It was built from the SFT checkpoint only (p. 16). Three DoM candidates were
  compared by steering before use. The "Sycophancy Verbose DoM" at layer 17 moved the share of changed answers
  from 90 % to 0 % over the steering range; a random direction moved it only to 43 % (Fig. 38, p. 58). Page 59
  (Eq. 12) describes the penalty direction as an SFT behavioural DoM: the mean layer-17 residual over the final
  reply of "capitulating" conversations (n = 2,217) minus that of "holding" conversations (n = 2,489). The 4,706
  two-turn conversations were generated by the SFT model on 280 MMLU test questions with 17 pushback challenges;
  ‖d(θ₀)‖ = 3.711. The direction is refit every 10 steps and blended with the previous one by an exponential
  moving average of weight 0.5 (p. 58). The text does not reconcile the two descriptions (paired replies with
  added verbal agreement on p. 58; the model's own unpaired replies on p. 59).
- *Held-out monitor panel.* MMLU validation split, three disjoint partitions of 160 questions (fit, calibration,
  test), 96 pairs per partition. The readout is averaged over reply tokens strictly before the answer letter
  (p. 59, Table 3).

**1.3 Headline numbers.** Each number has its unit and its comparison.

Vision:
- Waterbirds, unfrozen DINOv2, last layer. Worst-group accuracy (WGA) was 93 % to 97.7 % under the reliance
  penalties, against 51.6 % for plain training (ERM) (p. 8).
- Block 9. WGA was 85.9 % (Gradient × Input) to 93.8 % (Shuffle), and every penalty kept the original-direction
  AUROC (L1) at 0.98–1.0. Projection, LEACE and INLP brought L1 to chance (p. 10).
- After LEACE at block 9, L1 was 0.44 AUROC and L2 0.59, while nonlinear probes stayed above 0.98. Both linear
  detectors recovered within two layers (Fig. 7, p. 11).
- Refresh period. WGA was 94.5 %, 94.2 % and 90.8 % with the direction refreshed every 50, 500 and 5,000 steps;
  10 seeds (Fig. 31, p. 50).
- Penalizing along the DoM gave about 90 % WGA, against about 80 % along a logistic-probe direction, in the frozen
  setting (p. 11).
- CheXpert pneumothorax with a drain shortcut. ERM reached 33.3 % WGA, against 56.0 % for balanced training. Some
  penalties reached or exceeded the balanced reference when the directions were spans built from agent-labelled
  SAE features (pp. 13, 15; Fig. 10).

Sycophancy, OLMo-3-7B. Sycophancy is the share of MMLU answers changed after a user challenge (p. 15).
- It was about 24 % at SFT and about 54 % after plain DPO (read from the plot, Fig. 13, p. 17).
- Gradient × Input reliance shaping lands near SFT sycophancy at full DPO loss and full six-benchmark mean (Fig.
  13, p. 17; Fig. 39, p. 60).
- Data filtering (10k, 30k or 60k pairs removed), reward shaping and inoculation prompting fall on the line traced
  by lowering the DPO learning rate (p. 17).
- An LLM judge flagged 191 chosen replies (0.15 %) and 884 rejected replies (0.71 %) of 124,942 as sycophantic
  (Fig. 12A, p. 16).
- Monitoring. The original DoM's AUROC fell from 0.981 to 0.931. A freely trained linear probe stayed
  near-perfect (p. 17).
- Entanglement with the preference-difference direction b_t: cos(v₀, b_t) < 0.097 and cos(v_t⊥, b_t) < 0.043
  (p. 18).
- Scale. OLMo 2 13B (λ = 0.1) and OLMo 3 32B (λ = 0.5) reach near-SFT sycophancy while keeping DPO's capability
  gain (Fig. 40, p. 60).
- Block swap. The weights of blocks in the DPO model were replaced with those of the shaped model (Fig. 41, p. 61;
  p. 60). A single block before layer 17 recovers up to 20 % of the shaped run's sycophancy reduction. Blocks
  after layer 17 recover nothing. Windows of five consecutive blocks ending at layer 17 or 18 recover up to 76 %.
  Those windows are where the shaped weights moved furthest from DPO's (relative weight change per block). Every
  hybrid keeps the DPO six-benchmark mean within 0.6 points. Page 18 states this as "reduce sycophancy to 75% of
  the shaped model" and page 60 as "a 75% reduction"; the figure's quantity is the share of the shaped run's
  reduction.

Reward hacking, Qwen3.5-9B:
- Under plain GRPO, the peek rate rose from about 0.15 to 1.00 within ten updates, and honest solving fell to 0.
  Every expression-shaping method suppressed peeking within about ten updates, and honest solving reached
  0.96–0.98. Filtering succeeded in one of three seeds (p. 19).
- With gradients through the score, the original probe's balanced accuracy fell from 0.98 to 0.57 by update 150.
  It stayed at 0.87–0.90 for the other methods. A refitted probe read 0.97–0.98 (p. 19).
- Chess transfer. The base model peeks in 30 % of episodes and solves 26 % honestly. GRPO on Countdown gives 0.99
  peeking and 0.01 honest solving. Shaped policies peek at most 0.004 of the time and solve 0.48–0.60 honestly with
  their own v, and 0.71 with the frozen reader's v (p. 20).

Harmfulness, Mistral-7B-Instruct and Llama-3-8B-Instruct:
- JailbreakBench harmfulness (StrongREJECT score) fell from 0.70–0.71 to 0.01 with a DoM reward on a frozen copy of
  the base, against 0.16 and 0.19 for the probe loss of Libon et al. L1 AUROC was 0.84 and 0.85 against 0.64 and
  0.52 (Table 2, p. 21).
- The frozen-base reward refuses 14–20 % of safe XSTest prompts, against 2–10 % for the probe loss (p. 22).
- With gradients through the probe, the original probe's AUROC fell from 0.87 to 0.58–0.64 within 50 updates. A
  refitted linear probe read 0.84–0.86 (p. 22).
- After 400 updates on Mistral, the original layer-19 probe reached AUROC 0.30. Refitted linear probes read
  0.83–0.87 across layers, and an MLP probe added 0.02–0.04 (p. 23).
- A refitted DoM rotated by about 50° (Mistral, frozen-base reward) while the original probe kept AUROC 0.84
  (pp. 22–23).
- Long runs. After 400 (Mistral) and 1,000 (Llama) updates, the original probe read 0.85 and 0.86 and JailbreakBench
  harmfulness stayed at 0.01. But answers grew to the 256-token limit as moralizing text, and only 0.40 and 0.58 of
  safe XSTest prompts were answered (p. 64; Table 4, p. 65).

**1.4 The theory's predictions, as testable sentences.** Setup (p. 24): input x = (x_c, x•), with core features x_c
and one unwanted feature x•; encoder z = Wx; a fixed projection P = I − AAᵀ removes the subspace A that encodes
x•; h = Pz; linear head ŷ = wᵀh; δ = ∂ℓ/∂ŷ. Residual version: Fig. 23, p. 25; Appendix K, pp. 72–79. Each item
gives the claim, the test, the observation that would refute it, and what was observed.

1. *Regrowth (re-encoding).* After a fixed projection, the removed subspace receives no gradient. The concept's
   encoding regrows from the first step, at a rate of about ‖Pw‖·β• (β• is the concept's coefficient in the best
   linear predictor), along a new direction outside the removed subspace (Proposition 1, p. 25; Propositions 7–9,
   p. 68; Fig. 48, p. 69). Test: after a fixed projection, a linear probe refitted at the same layer recovers the
   concept. Refuted if the model's use of the concept returns while a refitted linear probe at that layer stays
   at chance. Observed: toy residual network R² = 1.00 by step 100 (p. 73); in Waterbirds the DoM moves away from
   v₀ as WGA rises (Fig. 5, p. 10); in OLMo-3-7B the original DoM fell 0.981 → 0.931 while a refit probe stayed
   near-perfect (p. 17).
2. *Entanglement.* The regrown encoding lies along Pw, the direction the head already uses to read the core
   features (p. 25). Test: a fresh erasure along the regrown direction also removes part of the core, so task
   accuracy falls below the best accuracy available without x•. Refuted if accuracy after a fresh erasure stays at
   that level. Observed: in the toy, 0.49–0.50 against 0.652 (p. 81), and 0.42–0.56 against 0.608 in Appendix K.2
   (pp. 73–74). In Waterbirds under LEACE, cos(v_t, b_t) rose to 0.44 (ResNet-50) and 0.39 (DINOv2) (p. 82; Fig.
   53, p. 81). Counter-case: in OLMo-3-7B sycophancy the cosines stayed below 0.097 and 0.043 (pp. 18, 82).
3. *Relocation to other layers.* Every block below the intervention receives the same regrowth signal. Only the
   output map of the last block before the intervention keeps its frozen part (Proposition 2, p. 26; Propositions
   10–12, p. 72). Test: the weight change and the causal effect of a penalty at one layer concentrate in the
   blocks just below it, and a fixed probe moved to a lower layer fails too. Refuted if swapping blocks above the
   penalty layer carries the behaviour, or if the blocks below show no change. Observed: in Fig. 41 (p. 61),
   blocks after layer 17 recover nothing, single blocks before it up to 20 %, and five-block windows ending at 17
   or 18 up to 76 %. In harmfulness, original probes lose accuracy from about layer 6 onward (p. 23). In
   Waterbirds, linear detectors recover within two layers above the LEACE site (p. 11). The paper does not exclude
   a concept spreading across layers or token positions (p. 28).
4. *Nonlinear product code.* When the linear erasure is refitted often, the model keeps using x• through a
   degree-two code h_odd(x) = x•·g(x/‖x‖). This code is uncorrelated with x•, so no linear readout can use it,
   but a nonlinear reader decodes it (Propositions 3–4, p. 27; Lemma 1, p. 28; Eqs. 17–18, pp. 75–76). A ReLU layer
   already contains the code at random initialization; training strengthens it and teaches the block above to
   read it (p. 77). Test: under frequent refits, accuracy just after a fresh erasure returns to the Bayes level,
   while a linear probe fails and an MLP probe reads x•. Refuted if accuracy after a fresh erasure stays at the
   level available without x•, or if an MLP probe does no better than a linear one. Observed: in the toy with
   refits every ≤ 10 steps, linear R² ≤ 0.014, nonlinear share 0.04 → 0.29, accuracy after a fresh erasure 0.80
   against a best of 0.806, MLP probe R² 0.99 (p. 26). With refits every 20 steps (K.2): accuracy 0.788–0.790,
   linear R² 0.12–0.27, MLP probe R² 0.95–0.96 (p. 74). Also at Waterbirds block 9 under LEACE (p. 11). **Not
   observed in any language model:** there an MLP probe added only 0.02–0.04 AUROC, about the same as in the
   control without a harm penalty (pp. 23, 82).
5. *Erasure.* Penalties whose attribution contains the coordinate ⟨z, v⟩ (Gradient × Input, Integrated
   Gradients, Deletion) have a second optimum in which the encoder stops encoding the concept (p. 10). Test: scale
   the gradient through ⟨z, v⟩ by ρ. L1 should stay near 1 for small ρ and drop past a critical ρ, which shrinks
   as λ grows (Fig. 32, pp. 49, 51). Refuted if a stop-gradient on ⟨z, v⟩ (ρ = 0) still loses L1. Observed: under
   Gradient × Input with refresh in Waterbirds, the land and water centroids of each class collapse (Fig. 6, p.
   11), and the refitted nonlinear reader fails together with the original one (pp. 28, 82).
6. *Early stopping* (linear toy only). After an ablation, reliance on x• returns in a fast mode and a slow mode.
   Stopping between the two keeps about half of the reduction at a few percent of the excess loss. Trained to
   convergence, the model returns to its original reliance; "interventions suit fine-tuning more than
   pretraining" (pp. 68–71, 82).

**1.5 The monitorability hierarchy** (Table 1, p. 7). Five levels, ordered by which part of the original linear
monitor is kept.

| level | name | kept |
|---|---|---|
| L1 | reuse linear | direction, probe family, layer, model |
| L2 | linear refit | probe family, layer, model |
| L3 | nonlinear refit | layer, model |
| L4 | different layer | model |
| L5 | different model (an external judge of input or output) | nothing; does not show readability inside the shaped model |

L1–L3 are evaluated at the intervention layer, after the projection for erasure methods (p. 7). Each form of
circumvention calls for one level: L2 for re-encoding, L3 for the product code, and L4 after the next nonlinear
layer (p. 28). "Circumvention" is the paper's word for a monitor no longer reporting what training was meant to
remove (p. 3).

**1.6 Stated limitations.**
- Other mechanisms may arise in other settings or at larger scales (p. 1). The three forms are not claimed to be
  all the ways a model can get around a constraint. Moving the concept into higher-order interactions, or spreading
  it across layers or token positions, is not excluded (p. 28).
- Whether favourable monitorability holds at frontier scale is open. Training duration is left for future work. The
  same intervention can leave the monitor working over a short run and lose it over a longer one (p. 29).
- A frozen reader removes pressure on the representation but not on the scored text, so a KL or length constraint
  is still needed (pp. 24, 64).
- Language-model results are one seed per setting (Figs. 18–21, pp. 21–23; Figs. 43–47, pp. 63–66). The Figure 14
  OLMo run is a single run (p. 58).
- Under direct gradients, norms were tracked only for tokens below the exclusion cutoff (Fig. 21 caption, p. 23).
- The residual-network propositions assume block outputs are added directly to the stream ("pre-norm
  architectures, not post-norm"). They use first-order approximations (pp. 72–73).
- Why data filtering fails is given as a suspicion, not tested (p. 17).

**1.7 Code, data, weights.** Page 1 links https://github.com/goodfire-ai/reliance. The repository exists and is
public. GitHub API, checked 2026-11-05: created 2026-11-05T10:28:27Z, last push 21:02:07Z, top level `README.md`,
`pyproject.toml`, `reliance/`, `experiments/`, `tests/`, `assets/`, no license reported. I did not read its
contents, so which experiments it covers is unknown. The PDF does not say that trained weights or new datasets are
released. The training and evaluation data are public sets (Dolci-Instruct-DPO, MMLU, BeaverTails, JailbreakBench,
XSTest, ClearHarm, CheXpert, CUB, Places, ImageNette, Lichess puzzles).

**1.8 Inconsistencies inside the paper.**
- Toy reference accuracies. Pages 26 and 81 give a best accuracy of 0.806 and 0.652 without x•; Appendix K.2 gives
  a Bayes accuracy of 0.792 and 0.608 without x• (p. 73). Section 6 says one seed; K.2 says five seeds.
- The 32B model is "OLMo 3 32B" in the text (pp. 17, 59) and "Olmo-3.1 32B" in the legend of Fig. 40 (p. 60).
- Reward hacking. Page 19 describes the frozen reader as "a frozen copy of a reference model (Qwen3.6-35B)", as
  printed; Fig. 16 and page 20 call it the frozen base.
- The block-swap figure: 75 % (pp. 18, 60) against "up to 76%" (p. 61), with different wording (§1.3).
- The abstract lists SFT among the training paradigms (p. 1). The only SFT-like training in the body is the RAFT
  baseline (pp. 21, 63). This is my reading.

**1.9 Corrections to the 17:45 CST journal entry** (not edited here; one writer per file). Code is released (§1.7);
the entry says it is not. The entanglement cosine is 0.44 at most in the PDF (Waterbirds, ResNet-50, LEACE, p. 82),
not 0.45. It is a vision result; in OLMo it stays below 0.097 (p. 18).

## 2. Where it backs our results

| our claim | their finding | page | how far the support goes |
|---|---|---|---|
| (a) M1: a stage-2 mix with 0 % code+math moves neurons' strongest activations off code and math documents by about one slot in fifty; a 65 % mix holds them | No data-mix experiment and no neuron measurement. The nearest result: removing 10k–60k of 124,942 DPO pairs that a probe flagged moved sycophancy and DPO progress along the same line as a lower learning rate; "the model learns sycophancy from the data as a whole" | 16–18 | framing only. The paper shows a whole-dataset effect on a behaviour. The intervention removes pairs rather than changing a mix share, and the object is a residual direction, not neurons. |
| (a) M1: the feature's content (token-level in the one example read) is unchanged; its strongest instances move | A DoM rotated about 50° while the original probe kept AUROC 0.84; "direction drift alone does not establish a loss of monitoring". The direction along which activations move with a concept differs from the direction that detects it (Remark 1) | 22–23, 46 | framing only. Both say a summary of where a concept sits can change while what detects it does not. The objects differ: a residual direction against neuron top-50 document shares. |
| (b) N13 premise: the gain from RL on math lives in a small, identifiable part of the weights, so a transplant of the top changed weights could carry it | Copying five consecutive blocks (of 32) from the shaped run into the DPO run recovers up to 76 % of the shaped run's sycophancy reduction. These are the blocks with the largest relative weight change. Single blocks recover ≤ 20 %, and blocks after the penalty layer recover 0. The benchmark mean stays within 0.6 points | 18, 60–61 | parallel finding. A transplant of the most-changed weights carries a training-induced behaviour difference without loss of benchmark accuracy. Differences from N13: DPO with a penalty, not RL; sycophancy, not math; whole blocks (attention and MLP), not MLP neurons; 15.6 % of blocks; no random-window, DARE or energy-matched control; seeds not stated for the swap. |
| (b) N13: why the change would be localized | The theory places the change in the blocks below the penalty layer, because that is where the regrowth signal enters (Proposition 2) | 26, 72 | does not transfer. The location is set by the penalty layer, and RL-Zero has no penalty layer. The paper gives no reason to expect a small locus for an RL gain. |
| (c) Neuron study: the neuron basis is set early in stage 1 | Not studied. Toy: a ReLU layer holds the product code at random initialization, and training strengthens it (p. 77). Linear toy: "interventions suit fine-tuning more than pretraining" (p. 68) | 68, 77 | framing only. Both are toy models, and neither concerns a neuron basis in a trained LM. |
| (c) Ordinary drift between neighbouring checkpoints is the floor for any later change | Representation change is read against a run of the same length without the pressure. Without a penalty the direction "barely moves" (Fig. 5); the "No harm term" control (Figs. 19, 21); the MLP-probe gain is judged against the no-penalty control (p. 23). There is no checkpoint-to-checkpoint or seed-to-seed spread for the LMs (one seed) | 9–10, 22–23, 82 | same practice: a no-pressure reference of equal length. It is not evidence for the size of our floor. |
| (d) A probe that fails is not a negative result | "neither a lower training penalty, nor failure of a particular detector, establishes that the unwanted behavior or concept information is gone" (p. 7). L1 0.44 and L2 0.59 while L3 > 0.98 (p. 11). Original probe AUROC 0.30 while refits read 0.83–0.87 (p. 23) | 7, 11, 23 | direct evidence: stated as a rule and shown in four settings. |
| (d) Identity pairs and drift references before a change is read | Drift references: no-penalty runs; a random direction (p. 19); a random direction steers only to 43 % against 0 % for the DoM (Fig. 38). Positive control: each DoM was tested by steering before use (p. 58). In-training measurements run without gradients and restore the random-number state (p. 58). No identity pair (the same weights measured twice) | 19, 58 | same practice for drift references and a positive control. No identity pair, and the LM results are one seed, which our rules would stamp. |

## 3. Where it threatens our results

| our result or measure | their finding | page | how far it reaches |
|---|---|---|---|
| M1's per-pole measure: the share of a pole's top-50 positions in code and math documents, with the pole's identity fixed | Under pressure a concept regrows along a new direction (v_t⊥ grows from zero). A fixed readout then loses it while a refitted one finds it. This is the L1/L2 distinction | 7, 9–10, 25, 80 | Moderate, for the wording only. M1 reads each pole at fixed identity, which is the L1 type. "G moves neurons off code and math" could be a loss in the neuron basis, or a move of code and math to other poles. Band means average over all 176,128 poles of a band, so a move between poles of one band partly cancels in the mean (my reasoning). The banked counts can separate the two at 0 GPU-hours (§4 item 1). |
| Any single-layer measure: a lens or probe at one layer (J10), one-layer instruments | Relocation: the change spreads to every block below the pressure point; original probes fail from about layer 6 onward; detectors recover two layers up | 11, 23, 26, 61, 72 | Low for M1 and N13: M1 covers all 32 layers and N13 ranks neurons across all layers. Real for single-layer readouts: a failure at one layer says nothing about the next one. |
| Neuron measures read in one basis (top activations of single neurons) | Product code: the concept sits in degree-two statistics spread across units (B has effective rank 2.2–3.0 over 32 units), which no single linear readout sees | 26–28, 76–77 | Low in our setting. The code appeared under repeated linear erasure (refits every ≤ 10–20 steps) and at one vision layer under LEACE. It did not appear in any LM tested (MLP probe +0.02–0.04 AUROC, the same as the control, p. 23). M1 and N13 apply no erasure. A single neuron's activation is already nonlinear, but a code spread over many units would not show in any one unit's top-50. |
| Reading what a unit does from its top contexts: M1's example L04N3542−, J1's raw examples, J9 and J12 exemplar windows | The max-activating tokens of a sycophancy direction that steers 90 % → 0 % looked like those of a random direction. Automatic labels of every layer's DoM described sequence boundaries, boilerplate and formatting | 16, 58 | Moderate, for content claims. M1's verdict is a document-share claim and stands. The sentence "a token-level feature" is a reading from top contexts, which is the reading this figure shows can mislead. It needs a matched random-pole reading beside it (§4 item 3). |
| N13: citing Fig. 41 as support | Single-block transplants carry ≤ 20 %, and only five-block windows carry up to 76 %. The locus is set by the penalty layer | 60–61, 26 | Low for the design: N13 has the random, DARE and energy-matched controls that Fig. 41 lacks. Real for the write-up: Fig. 41 must not be cited as evidence that RL gains are localized in neurons. |
| M1's lasting reading: "G moves, M holds" | Linear toy: the effect of an intervention fades if training continues to convergence | 68–71 | Framing only. M1 is not an intervention on a direction, and the argument is a linear toy. It is a reason to state M1's result as "at the end of 100B tokens", not as permanent. |

## 4. What we can borrow at low cost

Ranked by value per GPU-hour. Items 1–5 cost 0 GPU-hours. M1 is closed and its mixes are unblinded, so every M1
item is a dated exploratory analysis, not a test of a frozen rule.

1. **M1 pole turnover (tests the relocation threat on a closed result).** *What:* per band, take the set of poles
   whose code+math share is at or above a threshold τ at the start. Count how many stay above τ at G, R and M
   (fixed identity, the L1 type). Count how many poles outside the set rise above τ (re-selection, the L2 type).
   Freeze τ on the stage-1 reference harvests before reading G, R or M again. Compute the same two counts on a
   stage-1 neighbour pair as the floor. *What it would show:* few leaving and few arriving beyond the floor means
   G's mean drop is a loss in the neuron basis. Arrivals at the rate of departures mean code and math moved
   between poles. It cannot see a move out of the neuron basis; that needs activations. *Cost:* 0 GPU-hours,
   numpy on the laptop. *Artifacts today:*
   `studies/next/m1/banks/{start_a,start_b,G,R,M}/m1/counts.npz` and
   `studies/next/m1/banks/refs/{t00,t12,t13,t14b,t14c,t15}/m1/counts.npz`. Each holds `counts` (32 × 11,008 × 2 ×
   50, top-50 positions per directory), `filled`, `cm_mask` (the code+math directories), `dirs` and `groups`. Use
   the share definition of `studies/next/m1/read_head/recount.py`.
2. **N13 per-block weight-change profile (informs N13 before money).** *What:* in R1's ranking output, add the
   per-block relative change ‖θ_RL − θ_base‖ / ‖θ_base‖ for the attention weights and the MLP weights of each of
   the 32 blocks. *What it would show:* whether the RL-Zero-Math change has a block-level peak like Fig. 41's
   middle row (p. 61), or is flat. A flat profile retires item 6. *Cost:* 0 extra GPU-hours; R1 already reads every
   tensor (`N13_SPEC.md` §3.1). It needs a code change on branch `n13`, and a dated amendment if it lands after
   `n13-freeze-1`. *Artifacts:* none banked yet; it rides on J5.
3. **Matched random-pole reading control** (Fig. 12C–D, p. 16). *What:* before writing what a pole responds to,
   decode the top contexts of 3–5 random poles from the same layer, matched on maximum activation, and read them
   blind beside it. *What it would show:* whether the reading picks out the pole or describes what any strongly
   active pole looks like. *Cost:* 0 GPU-hours where the matched poles' contexts are banked
   (`studies/next/m1/banks/*/samples/contexts.jsonl`); otherwise the poles are added to the sample list of the
   next harvest. *Applies to:* `raw_examples_m1.md` (L04N3542−), J1's ten raw examples, J9 and J12 exemplar windows.
4. **Read with one direction, edit along the other** (Remark 1, p. 46; Eq. 8; Proposition 6, p. 47). The paper
   states two rules. The difference of means is the direction to edit or erase along: rank-one erasure removes the
   linear information only along it. A probe is the direction to detect with. For a featurizer, the decoder atom
   is for specification and the encoder row for detection (p. 46). An MLP neuron has the same two parts: its
   down-projection column (the direction it writes) and its gate and up rows (the directions it reads). This
   mapping is ours, not the paper's. *Use:* J11 (N6) edits along the written direction; J12's "input-row and
   output-column cosines" kept as two separate quantities; J10 directions for interchange taken as DoM, not probe
   weights. *Cost:* 0 GPU-hours; spec text.
5. **State the monitoring level of each claim** (Table 1, p. 7). In the M1 and N13 write-ups and in
   `20_ABSTRACT.md`, say which level each measure is at. M1's per-pole share is the L1 type until item 1 runs. Use
   p. 7's sentence as the citation for the program's rule that a failed detector is not a negative result.
   *Cost:* 0.
6. **L3 check beside any failing linear readout in J10.** *What:* when the linear lens or a linear probe fails at a
   layer, fit a one-hidden-layer MLP probe on the same activations at the same layer. Report its gain over the
   linear probe beside the same gain on a control, as the paper does against its no-pressure run (p. 23). *What it
   would show:* whether the failure is "absent" or "not linear". *Cost:* minutes of GPU on activations J10 already
   collects, inside its 5–8 GPU-hours; one spec line.
7. **Block-window transplant arms for N13.** Only if item 2 shows a peak. *What:* two arms. One copies whole
   blocks (attention and MLP) from RL for the five-block window with the largest relative change; the other copies
   a random five-block window. *What it would show:* whether the RL-Zero gain sits in a block window, as the DPO
   penalty difference did (p. 61), beside N13's neuron arms. *Cost:* about 0.65–1.0 GPU-hours per arm, 1.3–2.0 for
   the two. This is our arithmetic: S4's 17.6–26.5 GPU-hours (`09_PLAN.md`, J6) over the roughly 27 arms of
   `N13_SPEC.md` §3.2, at equal cost per arm. It needs a dated amendment before `n13-freeze-2` and goes below the
   existing arms in the stop rule's priority.
8. **SFT → DPO neuron harvest on the paper's own object.** *What:* two harvests with the M1 instrument and corpus,
   on the public OLMo-3-7B Instruct SFT and DPO checkpoints. The exact repository names must be verified on Hugging
   Face before any spec. *What it would show:* whether a DPO stage moves neurons' top-activation document shares by
   more than the stage-1 drift bar. That puts a post-training stage beside M1's stage-2 mixes. *Cost:* 3.4–3.8
   GPU-hours on AWS L40S (1.7–1.9 per harvest, from `09_PLAN.md` J7: 8.4–9.5 for five). It is a new question and
   needs its own spec.

## 5. Sentences for the program's abstract, with this citation

Sentences we can write:

1. "A detector that stops reading a concept after training does not show that the concept is gone; in recent work
   on training against internal representations, a refitted linear probe recovered the concept in most settings,
   and a nonlinear probe recovered it where linear ones failed (Fel et al., 2026, pp. 7, 11, 17, 19, 22–23)."
2. "In OLMo-3-7B, copying the five consecutive transformer blocks whose weights differed most between a
   sycophancy-penalized DPO run and a plain DPO run into the plain run recovered up to 76 % of the penalized run's
   sycophancy reduction, with the six-benchmark mean within 0.6 points (Fel et al., 2026, pp. 60–61)." Use as
   background for N13 only, with its scope: block level, DPO, one penalty layer.
3. "Fel et al. (2026) measure how training pressure on a concept direction changes the representation during
   post-training (DPO, GRPO, REINFORCE) and classifier training; their study uses no pretraining or mid-training
   checkpoint and compares no data mixes (pp. 7–24, 58)."
4. "Training against a linear readout can move a concept into a degree-two code that only a nonlinear reader
   decodes; this was shown in a toy residual network under repeated re-erasure and at one intermediate layer of a
   vision model, while in the language models tested a nonlinear probe added only 0.02–0.04 AUROC over a linear
   one (Fel et al., 2026, pp. 11, 23, 26–28, 74)."
5. "The direction along which activations change with a concept (the difference of means) and the direction that
   best detects it (a linear probe) coincide only when the within-group noise is isotropic, and penalizing along
   the difference of means gave about 90 % against about 80 % worst-group accuracy on Waterbirds (Fel et al., 2026,
   pp. 11, 46)." Cite Fel et al. only; do not cite Haufe et al. (2014) for this unless its full text is read.

Sentences we must not write:

- That the paper shows training changes are localized in a small set of neurons or weights. Its transplant is
  whole blocks, between two DPO runs, at a location set by a penalty, with no random control (pp. 60–61).
- That RL gains, or math gains, can be transplanted. Not studied.
- That a data-mix change moves representations or neurons, citing this paper. Not studied; its data intervention is
  pair removal in DPO (pp. 16–17).
- That representations or bases are set early in pretraining, citing this paper. Only toy statements exist
  (pp. 68, 77).
- That product codes or circumvention occur in language models. In the LMs tested, the nonlinear gain matched the
  control (p. 23).
- That shaped concepts stay monitorable in general. It is one seed per LM setting, and frontier scale is left open
  (pp. 21–23, 29).
- That the paper releases weights or data. It releases a code repository (p. 1); weights are not mentioned.
- "Cosine up to 0.45" as an OLMo or LM number. The PDF gives 0.44, in a vision model (p. 82).

## 6. What it does not cover

- **No pretraining or mid-training checkpoint.** Every LM run starts from a post-trained model:
  OLMo-3-7B-Instruct-SFT (p. 58), Mistral-7B-Instruct and Llama-3-8B-Instruct (p. 21). The Qwen3.5-9B variant and
  the OLMo 2 13B and OLMo 3 32B variants are not named further (pp. 17, 19, 60).
- **No data-mix effect, and no neuron.** The only data intervention removes 10k, 30k or 60k DPO pairs ranked by a
  probe (pp. 16–17). Every readout is a residual-stream direction (DoM, linear or MLP probe) or, in vision, an SAE
  latent. No MLP neuron is read.
- **No intermediate released OLMo checkpoint**, and no stage-1 or stage-2 checkpoint. The only OLMo trajectory is the
  authors' own 975-step DPO run (p. 58).
- **No RL on math**, no RL-Zero, no transplant of an RL change, and no DARE or matched-random weight control. The
  one transplant (p. 61) has no random-block control.
- **No run-to-run spread for the LM results** (one seed) and no identity pair.
- **No corpus-scale top-activation statistics**, no births or ages of features, and no checkpoint-to-checkpoint
  drift reference.
- **No claim about when a basis forms during pretraining**, beyond the two toy statements (pp. 68, 77).

M1 (a data-mix effect on neurons' top-activation documents across released stage-2 checkpoints), N13 (a
neuron-level transplant of an RL-Zero math gain against DARE and energy-matched random arms) and the neuron study's
stage-1 timing are not covered by this paper. This note does not re-check the rest of the literature.

## 7. Provenance

- Paper link given: https://www.goodfire.com/preprints/mechanistic-shaping. One redirect, HTTP 200. Final URL:
  https://static.goodfire.ai/mechanistic-shaping/mechanistic-shaping.pdf.
- Downloaded 2026-11-05 at 15:47 local time with `curl -L` into an empty directory,
  `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/goodfire/pdf/mechanistic-shaping.pdf`.
  Nothing else is in that directory, and nothing was run from it.
- Size 21,573,758 bytes. sha256 `4d0cf520c18bead015af5f95345febc5834cb94580601b54c974c3a11f421755`. Header
  `%PDF-1.5`. 82 pages. No date is printed in the PDF.
- Pages read: all 82, as rendered pages through the Read tool, in five calls: 1–20, 21–40, 41–60, 61–80, 81–82.
  Main text pp. 1–29; references pp. 30–41; appendices A–L pp. 42–82.
- Pages not readable: none as a whole. Small figure text is partly illegible on p. 16 (Fig. 12C token highlights),
  p. 57 (Fig. 37 prompt text, mostly legible), p. 60 (Fig. 39 bar values) and p. 61 (Fig. 41 per-cell values and
  the scale factor of the weight-change axis). Values taken from plots are marked "read from the plot".
- Repository check: GitHub API metadata and top-level listing only (§1.7). Contents not read.
- Not read: the blog post (summarized in the 17:45 CST journal entry), any cited paper (Blank et al. 2026, Libon et
  al. 2026, Haufe et al. 2014 and others), and the repository's code.
- Program files read before the paper: `studies/next/09_PLAN.md`, `studies/next/m1/READ_head.md`,
  `../interp_research-n13/studies/next/07_PICKS/N13_SPEC.md` §1 and §3, and `studies/next/JOURNAL.md` (17:45 CST
  entry). Directory listings and array shapes (no values) of `studies/next/m1/banks/` were taken to name §4's
  artifacts.
