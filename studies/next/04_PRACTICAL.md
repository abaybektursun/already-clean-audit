# 04 — Practical: what capability engineers optimize, and where a neuron-level view can move their numbers

Opus research agent for the Fable session, 2026-11-05. Status: DRAFT. This is a map, not a literature audit;
novelty of every claim in (d) is checked in `04_LITERATURE.md` before any score. Nothing here has been run.

**Answer.** The highest-leverage intersection is the RL train–inference mismatch, where computing only the LM-head
logits in FP32 raised ScaleRL's fitted asymptotic pass rate from 0.52 to 0.61 (https://arxiv.org/html/2510.13786).
The fixes in use are global (FP16 everywhere, FP32 head, batch-invariant
kernels), and two searches found no paper that localizes the remaining mismatch to residual-stream channels or
neurons. The other four of the top five are in §5.

---

## 0. How to read this file

- About 100 pages were opened on 2026-11-04 and 11-05: arXiv abstracts or HTML full text, lab blog posts, model
  cards, GitHub READMEs. A number is quoted only from a page that was opened. An item seen only in a search-result
  snippet is tagged **not verified**.
- Quotes are verbatim and at most 15 words.
- GPU-hour figures in the (d) sections are this file's estimates, not sourced numbers. Basis: one L40S-class GPU
  (the default `g6e.2xlarge` of `./infra/aws_pg/pg`); 7B bf16 generation with vLLM at about 2,500 output
  tokens/s aggregate at batch 64 (assumption; the first run measures it); about 1 GPU-hour per 7B model variant
  for MATH-500 at one sample plus AIME at 16 samples with up to 16k-token answers (assumption).
- Thresholds inside "Refuted if" are proposals. Under the repo rules they are frozen on control data before
  unblinding.
- Every claim must survive the record of negatives in `02_FRAMES.md` §2: readable is not used; a label is not a
  function; one pole removed is near the noise. So every claim below reads out a realized effect (accuracy, loss,
  log-probability, FLOPs), acts on many neurons at once, and carries a matched random set.
- Assets used (from `02_FRAMES.md` §2; release details verified in `00_ASSETS.md`, not here): OLMo-3 7B with
  1,421 stage-1 checkpoints; 600 poles banked at 16 checkpoints; 1,238 dated late-born jobs; the replicated loop
  edit (128 columns of two down-projection matrices); the robot chain with the LIBERO harness. No claim here
  uses the robot chain; LIBERO success rate is covered by `03_IDEAS.md` N9.

**OLMo-3 post-training releases that the claims use** (model cards, opened):

- Think chain: Olmo-3-7B-Think-SFT → Olmo-3-7B-Think-DPO → Olmo-3-7B-Think. Post-training checkpoints are
  `step_XXXX` revisions. AIME 2025: 57.6 / 62.7 / 64.6. AIME 2024: 69.6 / 74.6 / 71.6. OMEGA: 45.0 / 40.5 / 37.8
  (https://huggingface.co/allenai/Olmo-3-7B-Think).
- RL-Zero family, RL from the base Olmo-3-1025-7B with no SFT stage named on the card: Math, Code, IF, General,
  Mix (names as listed on https://huggingface.co/allenai/Olmo-3-7B-RL-Zero-Math). The card does not say whether
  intermediate RL steps are released. To verify in `00_ASSETS.md`.
- Consequence: the DPO → RL step of the Think chain moves AIME 2025 by +1.9 points and AIME 2024 by −3.0. A
  transplant test on that step has little signal. Base → RL-Zero-Math is the stronger pair if its gain is large;
  that gain is measured first.

---

## 1. Post-training with reinforcement learning

**State, late 2026.** The open recipes read here are GRPO variants on verifiable rewards (DAPO, Dr. GRPO, CISPO
inside ScaleRL). Engineering work moved to
predictable compute scaling (ScaleRL, 400,000+ GPU-hours studied, https://arxiv.org/abs/2510.13786), numerical
agreement between the sampler and the trainer (https://arxiv.org/abs/2510.26788), and entropy control
(https://arxiv.org/abs/2505.22617). Analyses in 2026 describe RL as a sparse correction at 1–3% of token positions
(https://arxiv.org/abs/2605.06241) and as near-linear in weights over steps
(https://arxiv.org/html/2601.04537v1). Probes entered RL as monitors and as rewards
(https://arxiv.org/html/2609.19101, https://www.goodfire.com/research/rlfr). Reward hacking in production RL
generalizes to broader misalignment (https://arxiv.org/abs/2511.18397).

### 1(a). Numbers they report

- AIME 2024/2025 accuracy averaged over samples: DAPO reports 50 points on AIME 2024 from the Qwen2.5-32B base
  (https://arxiv.org/abs/2503.14476).
- MATH-500 accuracy: one training example lifts Qwen2.5-Math-1.5B from 36.0% to 73.6%
  (https://arxiv.org/abs/2504.20571).
- pass@k against the base model: RLVR wins at k = 1, the base wins at large k, six algorithms behave alike
  (https://arxiv.org/abs/2504.13837).
- Policy entropy over steps, fitted as R = −a·e^H + b: 73% of entropy and 76% of the gain are spent in the first
  200 steps (https://arxiv.org/html/2505.22617).
- Response length over training: an optimization bias in GRPO inflates the length of wrong answers
  (https://arxiv.org/abs/2503.20783).
- Compute–performance sigmoid with asymptote A, exponent B, midpoint C_mid: ScaleRL reaches A = 0.61; an 8B run
  was taken to 100,000 GPU-hours (https://arxiv.org/html/2510.13786).
- Sampler–trainer probability mismatch: FP32 logits alone move A from 0.52 to 0.61 (same URL); FP16 instead of
  BF16 removes the rounding mismatch (https://arxiv.org/abs/2510.26788).
- Forgetting against new-task gain: forward KL on the new task predicts forgetting with R² 0.71 on
  Qwen2.5-3B-Instruct; weight-change metrics reach R² 0.34–0.58 in the MNIST setting
  (https://arxiv.org/html/2509.04259).
- Reward-hacking rate and detector recall at fixed false-positive rate: GLM 5.2 hacks in 57.2% of DeepSWE
  rollouts; probe-or-monitor raises Kimi K3 recall from 41.1% to 80.2% for about 1 point of FPR
  (https://arxiv.org/html/2609.19101).
- Parameter footprint of the update: 5–30% of parameters move, across 7 algorithms and 10 models
  (https://arxiv.org/abs/2505.11711); rank-1 LoRA matches full fine-tuning in policy-gradient RL
  (https://thinkingmachines.ai/blog/lora/).
- Wall-clock per accuracy point: RL-Extra passes 0.38 on AIME24 after 180 RL steps against about 1,100 for
  plain RL, up to 6.1× faster (https://arxiv.org/html/2601.04537v1).

### 1(b). Open problems they name

| problem | their words | source |
|---|---|---|
| entropy collapse | "the collapse of policy entropy" | https://arxiv.org/abs/2505.22617 |
| predicting RL compute scaling | "there is no principled understanding" (of how to judge algorithmic changes) | https://arxiv.org/abs/2510.13786 |
| gains bounded by the base model | "This highlights the need for improved RL paradigms" | https://arxiv.org/abs/2504.13837 |
| length bias | "artificially increases response length (especially for incorrect outputs)" | https://arxiv.org/abs/2503.20783 |
| sampler–trainer mismatch | "often suffers from instability due to the numerical mismatch" | https://arxiv.org/abs/2510.26788 |
| dynamics of low-rank RL | "Our theoretical understanding of LoRA learning rates and training dynamics is limited." | https://thinkingmachines.ai/blog/lora/ |
| linearity at scale | "We have not yet verified whether the observed linearity generalizes to ultra-large-scale models" | https://arxiv.org/html/2601.04537v1 |
| exploration and credit assignment | "existing PSRL approaches suffer from limited exploration efficiency" | https://www.arxiv.org/abs/2509.26628 |
| spurious-reward gains | "The mechanism behind spurious-reward gains is contested." | https://arxiv.org/html/2610.17824v1 (survey by two independent researchers) |
| detectors inside the reward | "Prior work has studied obfuscation only in artificial settings" | https://arxiv.org/abs/2602.15515 |
| forgetting under off-policy training | "we didn't study online but off-policy algorithms" | https://arxiv.org/html/2509.04259 |

### 1(c). Interpretability-derived wins, and attempts that failed

Wins. The first four rows are analyses of where the update acts (token or weight level), not neuron-level
interpretability; they are listed because they turned an internal observation into a method.

| observation | win | size | source |
|---|---|---|---|
| entropy change is driven by the covariance of log-probability and advantage on few tokens | KL-Cov / Clip-Cov restrict updates on 10⁻⁴–10⁻³ of tokens | +2.0 average (7B), +6.4 average (32B); 32B AIME24 21.8 → 36.8 | https://arxiv.org/html/2505.22617 |
| high-entropy "forking" tokens carry the learning | gradient only on the top 20% entropy tokens | Qwen3-32B +11.04 AIME'25, +7.71 AIME'24; equal to full on 8B | https://arxiv.org/abs/2506.01939 |
| RL changes 1–3% of positions; the promoted token is in the base top-5 | ReasonMaxxer from a few hundred base rollouts | about three orders of magnitude less training cost (authors' claim) | https://arxiv.org/abs/2605.06241 |
| weights and log-probabilities move linearly with step | weight extrapolation (RL-Extra) | up to 6.1× wall-clock | https://arxiv.org/html/2601.04537v1 |
| a probe on a frozen copy reads hallucination | probe score as RL reward (RLFR, Gemma-3-12B-IT) | 58% fewer hallucinations on held-out LongFact++ (best-of-32 in harness); about 90× cheaper per intervention than an LLM judge; ~360 steps for about $2.5k | https://www.goodfire.com/research/rlfr |
| a difference-of-means direction separates hacking transcripts | probe as reward-hacking monitor | AUROC 0.93 (DeepSWE), 0.94 (ImpossibleBench); recall 41.1% → 80.2% with the monitor | https://arxiv.org/html/2609.19101 |
| a "toxic persona" SAE latent predicts emergent misalignment | detection; re-alignment with "just a few hundred benign samples" | no detection number on the page | https://arxiv.org/abs/2506.19823 |
| concept directions found by interpretability | ablate them during fine-tuning (CAFT) | 10× fewer misaligned responses, no loss on the training distribution | https://arxiv.org/abs/2507.16795 |
| persona vectors | predict and prevent trait shifts caused by fine-tuning data | no number in the abstract | https://arxiv.org/abs/2507.21509 |
| high-attention steps mark reasoning behaviors | branch rollouts there (AttnRL, ICLR 2026) | no number in the abstract; not verified | https://www.arxiv.org/abs/2509.26628 |

Failed or did not transfer.

| attempt | what happened | source |
|---|---|---|
| deception-probe penalty inside RLVR | models learn obfuscated policies; activations also drift so the probe stops firing, with or without the penalty; honesty needs high KL plus a strong penalty | https://arxiv.org/abs/2602.15515 |
| probe for a verifiable hack | AUC 0.998, but a string match scores 1.0 (one 0.5B model, one task, one seed) | https://onlychan.xyz/blog/probes-rlvr/ |
| weight-change size as a forgetting predictor | R² 0.34–0.58 against 0.96 for forward KL (MNIST); the update's sparsity disappears in fp32 (a bf16 artifact); updates are full-rank | https://arxiv.org/html/2509.04259 |
| "RL updates a small subnetwork" read literally | the precision gate "hides micro-updates in non-preferred regions", so an off-principal update looks sparse | https://arxiv.org/abs/2511.08567 |
| spurious rewards as a general method | random reward gives +21.4 on MATH-500 for Qwen2.5-Math-7B, no gain on Llama3 or OLMo2 | https://arxiv.org/abs/2506.10947 |

### 1(d). Three claims a neuron-level, checkpoint-aware study can test

**R1. The RL gain is carried by a k-neuron delta, and its neurons have known jobs.** (Extends N13.)
- Number moved: fraction of the base → RL-Zero-Math gain on MATH-500 and AIME 2025 recovered by a partial
  transplant.
- Measurement: per MLP neuron, the fp32 norm of the released change in its gate row, up row and down column.
  Transplant the full change of the top-k neurons into the base, for k = 0.1%, 0.3%, 1%, 3% of MLP neurons.
  Attention changes are excluded in arm A and included in arm B. Rank by norm, never by count of changed entries:
  released weights are bf16, and entry-level sparsity is a precision artifact (https://arxiv.org/html/2509.04259).
- Controls: k random neurons matched on change energy; DARE-style random entries at the same parameter count,
  rescaled by 1/(1−p) (random drop of 90–99% of SFT deltas keeps abilities, https://arxiv.org/abs/2311.03099);
  the complement (everything except the top-k); the identity transplant, which must reproduce the released score.
- Readout beside the score: the banked jobs of the top-k neurons that are among the 600 poles, and their layer
  distribution.
- Refuted if: the top 1% recovers less than half of the gain; or DARE at the same parameter count lands within
  the seed band of the top-k; or the energy-matched random set does.
- Cost: weights only for ranking (under 1 GPU-hour); about 20 model variants × 1 GPU-hour of evaluation = 20–25
  GPU-hours.
- Nearest prior: 5–30% of parameters (https://arxiv.org/abs/2505.11711); rank-1 LoRA suffices
  (https://thinkingmachines.ai/blog/lora/); grafting 0.01% of parameters recovers >95% of fine-tuning performance
  (https://arxiv.org/abs/2302.06600). None in the neuron basis with job labels, a DARE control and RL at 7B, as
  far as these reads show.

**R2. Forgetting is predicted from weights alone by change on old, much-read poles.** (Connects N6.)
- Number predicted: the drop on held-out evaluations (MMLU by log-likelihood, wikitext perplexity, IFEval) after
  each post-training stage, before those evaluations are run.
- Predictor S: for each pole, the change in its output column times its mean activation on a fixed text set,
  weighted by its reader count (or by age: poles born before the first banked point count as old). Sum over poles.
- Points: RL-Zero Math, Code, IF, General, Mix; Think SFT, DPO, RL; Instruct SFT, DPO, RL (Instruct stage
  releases to verify in `00_ASSETS.md`); plus any released intermediate steps. At least 8 points without the
  Instruct chain, 11 with it.
- Baselines: forward KL on the new task, which needs rollouts (R² 0.71 on Qwen2.5-3B,
  https://arxiv.org/html/2509.04259); L2 and Fisher-weighted weight change (R² 0.34–0.58, same URL); plain
  activation change.
- Refuted if: leave-one-out R² of S is not above that of L2 weight change, or is below 0.5.
- Cost: weight differences (about 1 GPU-hour); pole activations on 1M tokens per model (about 2–3 GPU-hours in
  total); forgetting evaluations (about 6 GPU-hours); the KL baseline (about 6 GPU-hours). About 15 GPU-hours.
- Nearest prior: RL forgets less than SFT because of on-policy data (https://arxiv.org/abs/2510.18874); RL keeps
  more of the base model's head-level circuit than SFT on Qwen2.5-3B (https://arxiv.org/abs/2605.28860). Neither
  predicts forgetting from neuron age or reader count.

**R3. Entropy collapse runs through a few late-layer poles at forking positions.**
- Number predicted: the per-position entropy drop between RL checkpoints; later, the AIME score of an RL run
  regularized only on those neurons.
- Measurement: 2,000 rollouts from the starting model on the RL prompt set. Per token, record the starting
  entropy and the activations of MLP neurons in the last 8 layers. At each released RL step, recompute entropy on
  the same tokens by teacher forcing. Fit two predictors of the drop on half the prompts: starting entropy alone,
  and starting entropy plus the top 50 neuron activations. Score on the other half.
- Refuted if: the neuron features add less than 0.02 AUROC over starting entropy on held-out prompts; or the 50
  neurons chosen on one half overlap those chosen on the other half no more than random splits do.
- Cost: about 1 GPU-hour of generation; about 1 GPU-hour of teacher-forced passes at 5 checkpoints; about 3
  GPU-hours for the predictive step. The regularized RL run is a follow-up, about 30–60 GPU-hours.
- Depends on released intermediate RL steps (to verify in `00_ASSETS.md`).
- Nearest prior: covariance on 10⁻⁴–10⁻³ of tokens (https://arxiv.org/html/2505.22617); forking tokens
  (https://arxiv.org/abs/2506.01939); sparse policy selection at 1–3% of positions
  (https://arxiv.org/abs/2605.06241). All at token level, none in the neuron basis.

---

## 2. Inference systems

**State, late 2026.** Production attention architectures changed: latent KV (MLA cut KV 93.3% against DeepSeek
67B, https://arxiv.org/abs/2405.04434); sparse attention in DeepSeek-V3.2 (https://arxiv.org/abs/2512.02556; its
indexer details come from secondary sources, not verified); a learned per-head sink in gpt-oss
(https://huggingface.co/blog/welcome-openai-gpt-oss); a sigmoid gate after attention in Qwen3-Next
(https://github.com/qiuzh20/gated_attention). FP4 formats made the older outlier fixes less effective
(https://arxiv.org/abs/2509.23202). KV compression during decoding, which long reasoning traces need, is still
experimental in the main benchmark library (https://github.com/NVIDIA/kvpress). Training-free activation sparsity
reached 70% of SwiGLU FFN channels (https://arxiv.org/abs/2607.27591), and MLP sparsity vanishes at large batch
(https://arxiv.org/abs/2505.14884).

### 2(a). Numbers they report

- KV bytes per token at matched accuracy: MLA −93.3% (https://arxiv.org/abs/2405.04434); KIVI 2-bit: 2.6× less
  peak memory, 4× batch, 2.35–3.47× throughput (https://arxiv.org/abs/2402.02750).
- KV-compression leaderboard: RULER at 4k context (https://github.com/NVIDIA/kvpress).
- Long-context speed and maximum context: DuoAttention up to 2.18× decode (MHA) and 3.3M tokens on one A100 with
  quantization (https://arxiv.org/abs/2410.10819); StreamingLLM 22.2× over sliding-window recomputation
  (https://arxiv.org/abs/2309.17453).
- W4A4KV4 zero-shot gap to full precision: SpinQuant 2.9 points on LLaMA-2 7B
  (https://arxiv.org/abs/2405.16406); PrefixQuant +3.08 over SpinQuant on Llama-3-8B
  (https://arxiv.org/abs/2410.05265).
- FP4 end-to-end speedup: 2.2× on B200, 4× on RTX 5090 (https://arxiv.org/abs/2509.23202).
- Activation sparsity and decode speedup: TEAL 40–50%, 1.53× and 1.8×
  (https://arxiv.org/abs/2408.14690); Prox 70% FFN sparsity, 1.99× (https://arxiv.org/abs/2607.27591).
- Speculative decoding speedup and throughput at batch: EAGLE-3 up to 6.5×, 1.38× in SGLang at batch 64
  (https://arxiv.org/abs/2503.01840); MagicDec 2.51× at batch 32–256 (https://arxiv.org/abs/2408.11049).
- MoE compression: REAP prunes 50% of experts with near-lossless code generation on Qwen3-Coder-480B and Kimi-K2
  (https://arxiv.org/abs/2510.13999).
- Long-context accuracy: gated attention gains more than 10 points on RULER (https://arxiv.org/html/2505.06708).

### 2(b). Open problems they name

| problem | their words | source |
|---|---|---|
| decode-phase KV compression | "Not all existing presses are fully compatible with DecodingPress" | https://github.com/NVIDIA/kvpress |
| query-agnostic KV compression | "a compression method that works whatever comes after the context" | https://github.com/NVIDIA/kvpress |
| activation sparsity at batch | "union of active neurons quickly approaching dense computation" | https://arxiv.org/abs/2505.14884 |
| FP4 outliers | "NVFP4's small group size provably neutralizes traditional outlier mitigation techniques" | https://arxiv.org/abs/2509.23202 |
| high sparsity on SwiGLU models | "suffer substantial model-quality degradation at high sparsity" | https://arxiv.org/abs/2607.27591 |
| speculative decoding at batch | "the conventional wisdom suggests that its efficacy is limited to small batch sizes" | https://arxiv.org/abs/2408.11049 |
| sinks are not memory | "No. The context window remains unchanged." | https://github.com/mit-han-lab/streaming-llm |

### 2(c). Interpretability-derived wins, and attempts that failed

This is the area where interpretability observations already became shipped methods. "Interp-born" marks
methods whose papers start from an observation about specific activations, heads or neurons.

| observation | method | win | source |
|---|---|---|---|
| interp-born: systematic emergent outlier features | LLM.int8() mixed-precision decomposition | >99.9% of values in 8-bit; up to 175B without degradation | https://arxiv.org/abs/2208.07339 |
| interp-born: "Protecting only 1% salient weights can greatly reduce quantization error" (salience from activations) | AWQ | >3× over HF FP16; MLSys 2024 best paper | https://arxiv.org/abs/2306.00978 |
| interp-born: heads that learn a no-op push softmax inputs up and cause outliers | clipped softmax, gated attention | full INT8 activation quantization on BERT, OPT, ViT | https://arxiv.org/abs/2306.12929 |
| interp-born: attention sinks on the first tokens | StreamingLLM | 4M tokens streamed; 22.2× over recomputation | https://arxiv.org/abs/2309.17453 |
| sinks | "a learned attention sink per-head" in the softmax denominator (gpt-oss) | architecture adopted; no number | https://huggingface.co/blog/welcome-openai-gpt-oss (Hugging Face post, 5 Aug 2025; not checked against OpenAI's card) |
| interp-born: sinks and massive activations | sigmoid gate after attention (Qwen) | first-token attention 46.7% → 4.8%; max activation 1053 → 94; perplexity 6.026 → 5.761; MMLU +2.03; RULER +10; in Qwen3-Next; NeurIPS 2025 best paper (repo README) | https://arxiv.org/html/2505.06708 ; https://github.com/qiuzh20/gated_attention |
| interp-born: massive activations about 100,000× larger than others, acting as biases | (observation) | — | https://arxiv.org/abs/2402.17762 |
| interp-born: one "super weight" | keep it, clip other outliers, round-to-nearest with larger blocks | pruning it raises perplexity about three orders of magnitude | https://arxiv.org/abs/2411.07191 |
| interp-born: token-wise outliers at specific tokens | PrefixQuant: put those tokens in the KV prefix | +3.08 over SpinQuant (W4A4KV4, Llama-3-8B); 2.16× decode | https://arxiv.org/abs/2410.05265 |
| interp-born: 3 of 6,144 experts in Qwen3-30B-A3B create the massive activations | protect them under MoE compression | pruning them gives repetitive output; sinks collapse | https://arxiv.org/abs/2507.23279 |
| interp-born: retrieval heads are under 5% of heads | DuoAttention: full cache only on retrieval heads | 2.55× memory (MHA), 1.67× (GQA) | https://arxiv.org/abs/2404.15574 ; https://arxiv.org/abs/2410.10819 |
| interp-born: a few heavy-hitter tokens get most attention | H2O keeps 20% of tokens | up to 29× throughput | https://arxiv.org/abs/2306.14048 |
| interp-born: heads fall into local, special-token and broad patterns | FastGen per-head cache policy | no number in the abstract | https://arxiv.org/abs/2310.01801 |
| interp-born: contextual sparsity | Deja Vu | >2× over FasterTransformer on OPT-175B | https://arxiv.org/abs/2310.17157 |
| interp-born: power-law "hot" and "cold" neurons | PowerInfer: hot neurons on GPU | up to 11.69× over llama.cpp on an RTX 4090 | https://arxiv.org/abs/2312.12456 |
| magnitude sparsity of hidden states | TEAL | 1.53–1.8× decode | https://arxiv.org/abs/2408.14690 |
| head sparsity is stable across batch | Polar Sparsity | up to 2.2× end-to-end | https://arxiv.org/abs/2505.14884 |
| early layers already predict many tokens (this file's reading; the authors do not frame it as interpretability) | LayerSkip self-speculation | 1.82–2.16× | https://arxiv.org/abs/2404.16710 |

Failed or did not transfer.

| attempt | what happened | source |
|---|---|---|
| MLP contextual sparsity at serving batch sizes | the union of active neurons approaches dense | https://arxiv.org/abs/2505.14884 |
| training-free sparsity on SwiGLU models at high sparsity | prior methods lose substantial quality (Prox's statement about baselines) | https://arxiv.org/abs/2607.27591 |
| sinks as a route to long context | perplexity stays flat; the context window and memory do not grow | https://github.com/mit-han-lab/streaming-llm |
| rotations against outliers | random rotations differ by up to 13 points (SpinQuant); NVFP4's small groups neutralize outlier mitigation | https://arxiv.org/abs/2405.16406 ; https://arxiv.org/abs/2509.23202 |
| merging MoE experts | pruning beats merging for generative tasks | https://arxiv.org/abs/2510.13999 |

### 2(d). Three claims

**S1. The remaining sampler–trainer mismatch concentrates in a few massive-activation channels and sink
positions.**
- Numbers moved: per-token |log p(sampler) − log p(trainer)|, mean and 99.9th percentile; the fraction removed by
  keeping a few channels in fp32. Downstream: ScaleRL's asymptote moved 0.52 → 0.61 from the FP32 head alone
  (https://arxiv.org/html/2510.13786).
- Measurement: 1,000 rollouts from Olmo-3-7B-Think sampled by vLLM in bf16. Recompute log-probabilities with the
  training-style bf16 forward and with an fp32 reference. vLLM internals are not hooked, so the per-layer
  attribution runs on the training-side forward: per layer, the residual difference between bf16 and fp32,
  attributed to residual channels and positions by gradient times difference. Gate: the per-token bf16-vs-fp32
  error must correlate with the per-token sampler-vs-trainer mismatch, or the proxy is cancelled. Intervention:
  keep the LM head plus the top c channels (c ≤ 16 of the hidden size) and the down-projection columns of the
  neurons that write them in fp32.
- Controls: c random channels; the LM head alone (the known fix); full fp32 (the ceiling); both engines in fp32
  (the identity pair, which must show near-zero mismatch, or the instrument is broken).
- Refuted if: beyond the LM-head fix, the top 16 channels and positions 0–3 explain less than 30% of the
  remaining logit-difference variance; or the few-channel fp32 fix removes less than 25% of the 99.9th-percentile
  mismatch while full fp32 removes more than 75%.
- Cost: rollouts about 1 GPU-hour; recomputation in three precisions about 1 GPU-hour; attribution about
  1 GPU-hour. About 3–4 GPU-hours. A small GRPO stability run with and without the fix is a follow-up, about
  20–40 GPU-hours on a 1.5B model.
- Nearest prior: BF16 rounding as the cause (https://arxiv.org/abs/2510.26788); a per-logit additive
  perturbation model with no channel analysis (https://arxiv.org/abs/2609.32444); the numerical part stays roughly
  constant over training (https://arxiv.org/abs/2610.40945); high-magnitude LM-head activations in MiniMax-M1
  (body text seen only in a snippet; not verified; https://arxiv.org/abs/2506.13585); sink-kernel mismatch in
  gpt-oss RL (snippet only; not verified; https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl). No
  channel-level localization found.

**S2. Neuron jobs keep activation sparsity alive in grouped RL rollouts.**
- Number moved: the union fraction of active MLP neurons per layer at batch 64, at TEAL-style thresholds set for
  50% per-token sparsity; then FLOPs per decode step and, with a gather kernel, tokens/s.
- Measurement: Olmo-3-7B-Think, 4,096 prompts from an RL mix, 1,000 generated tokens each. Three batch
  compositions: random prompts; GRPO groups (8 prompts × 8 samples, samples of one prompt share context); prompts
  clustered by their mean activation over the 600 banked poles.
- Refuted if: both the grouped and the job-clustered batches raise union sparsity by less than 10 percentage
  points over random batches. The ceiling is the per-token sparsity (50%).
- Cost: about 4M generated tokens with activation recording, about 2–3 GPU-hours.
- Nearest prior: batching inputs that activate similar neurons was proposed in LTE (snippet only; not verified;
  https://arxiv.org/abs/2402.06126); MLP sparsity vanishes at batch (https://arxiv.org/abs/2505.14884). The
  grouped-rollout structure of GRPO was not found in either.

**S3. Drop the directions nobody reads (N10).**
- Number moved: wikitext perplexity and RULER-4k at fixed KV bytes per token.
- Measurement: per layer, rank value-cache directions by reader gain (the loss change when the direction is
  removed, averaged over a fixed text set) and, separately, by variance. Keep the top r of the head dimension, r =
  25%, 50%, 75%. Compare with KIVI 2-bit and kvpress ExpectedAttention at the same bytes.
- Refuted if: gain-ranked projection is not better than variance-ranked beyond the sample band at any r; or both
  lose to KIVI 2-bit at every matched byte budget (then no practical value).
- Cost: forward passes only, 2–4 GPU-hours.
- Prior: large (MLA, low-rank KV methods, KIVI). The novelty prior is low, as `03_IDEAS.md` already states.

---

## 3. Pretraining and scaling

**State, late 2026.** Decisions are made from small proxies: data rankings at 150M parameters
(https://arxiv.org/abs/2504.11393), task predictions from ~1%-compute ladders (https://arxiv.org/abs/2412.04403),
and mixture regressions over proxy swarms (Olmix, built for OLMo 3, https://arxiv.org/abs/2602.12237). Optimizer
gains shrink with scale (https://arxiv.org/abs/2509.02046). Muon reached 1T parameters with a per-head logit clip
(https://arxiv.org/html/2507.20534). Two 2026 papers predict downstream curves from internal representations
(https://arxiv.org/abs/2604.08510, https://arxiv.org/abs/2604.01025). Developmental interpretability reached
Pythia-1.4B, not 7B (https://timaeus.co/research).

### 3(a). Numbers they report

- Decision accuracy of small data experiments: rankings at 150M pick the 1B winner about 80% of the time at 0.01%
  of the compute (https://arxiv.org/abs/2504.11393).
- Task prediction error: within 2 points absolute for 7B-4T and 13B-5T targets at about 1% of their compute
  (https://arxiv.org/abs/2412.04403).
- Mixing gain and cost: mixture reuse +11.6% over no mixing; 74% less compute than recomputing after five domain
  updates (https://arxiv.org/abs/2602.12237).
- Optimizer speedup over AdamW: about 1.4× at 0.1B, about 1.1× at 1.2B (https://arxiv.org/abs/2509.02046).
- Parametrization efficiency: CompleteP 12–34% compute efficiency with depth-wise transfer
  (https://arxiv.org/abs/2505.01618).
- Stability: Kimi K2, 15.5T tokens with zero loss spikes; maximum attention logits exceeded 1000 without the clip
  (https://arxiv.org/html/2507.20534).
- Emergence: loss threshold below which tasks leave chance level (https://arxiv.org/abs/2403.15796); emergence
  predicted up to 4× compute ahead by fine-tuning small models (https://arxiv.org/abs/2411.16035).
- Skill order and curves: emergence order agrees across models (ρ = .81 over 45 pairs); held-out task curves
  predicted with R² .68–.84 (https://arxiv.org/abs/2604.08510).
- In-training evaluation cost: probes on checkpoint states, AUROC above 0.75, evaluation 1 hour → 3 minutes
  (https://arxiv.org/abs/2604.01025).

### 3(b). Open problems they name

| problem | their words | source |
|---|---|---|
| data mixing | "the configuration space for developing a mixing method is not well understood" | https://arxiv.org/abs/2602.12237 |
| emergence | "the lack of understanding around emergent capabilities" | https://arxiv.org/abs/2411.16035 |
| loss is not the target | "simple metrics like training loss (perplexity) are not always correlated with downstream performance" | https://arxiv.org/abs/2604.01025 |
| small-scale wins shrink | "the speedup of matrix-based optimizers is inversely proportional to model scale" | https://arxiv.org/abs/2509.02046 |
| dating a capability | "a model's hidden states can encode task information before its own readout expresses it" | https://arxiv.org/abs/2610.21025 |

Note on 2610.21025: `03_IDEAS.md` N1 cites it as "crosscoders across checkpoints". It is "Learning to Read Out:
Unembedding Dynamics in Language Model Pretraining" (He et al., 14 Oct 2026). It crosscodes the readout rows of
Pythia over checkpoints, from weights alone. It finds an early window at 0.4–1.1% of pretraining where the
readout reorganizes. The crosscoder paper across checkpoints is https://arxiv.org/html/2509.05291v1 (Bayazit,
Mueller, Bosselut; ACL 2026; Pythia, OLMo, BLOOM).

### 3(c). Interpretability-derived wins, and attempts that failed

| observation | win | size | source |
|---|---|---|---|
| attention-logit growth and output-logit divergence appear in small models at high learning rate | mitigations used at large scale work at small scale; two instabilities predictable from activation and gradient norms | no number in the abstract | https://arxiv.org/abs/2309.14322 |
| in a few heads, maximum attention logits exceed 1000 under Muon | per-head QK-Clip at τ = 100 | 15.5T tokens, 1T parameters, zero loss spikes | https://arxiv.org/html/2507.20534 |
| sinks and massive activations | gated attention trained in from the start | see §2(c) | https://arxiv.org/html/2505.06708 |
| syntactic attention structure appears abruptly with a loss drop | briefly suppressing it improves final quality | no number in the abstract | https://arxiv.org/abs/2309.07311 |
| susceptibilities | "patterning": data reweighting that speeds or slows structure formation in a small LM | no number in the abstract | https://arxiv.org/abs/2601.13548 |
| function-vector similarity between tasks | predict held-out tasks' learning curves | R² .68–.84 | https://arxiv.org/abs/2604.08510 |
| probes on in-training states | cheap checkpoint evaluation | AUROC > 0.75; 1 hour → 3 minutes | https://arxiv.org/abs/2604.01025 |
| induction circuit screening | the final induction circuit is identifiable within 0.3–2% of training tokens | — | https://arxiv.org/abs/2606.02378 |

Failed or did not transfer.

| attempt | what happened | source |
|---|---|---|
| developmental interpretability at scale | largest model on the Timaeus research page is Pythia-1.4B; no 7B result found | https://timaeus.co/research |
| early circuits as the final circuit | the induction circuit at formation overlaps the final one by Jaccard 0.29–0.33 (snippet only; not verified) | https://arxiv.org/abs/2606.02378 |
| dating a skill by output behavior | readout learning delays expression; a higher readout learning rate moves the date | https://arxiv.org/abs/2610.21025 |
| geometry phases as a training signal | phases correlate with downstream gains; no intervention reported | https://arxiv.org/abs/2509.23024 |

### 3(d). Three claims

**P1. Job births lead benchmark rises, and predict them on held-out tasks.**
- Number predicted: the checkpoint at which a task's continuous log-likelihood metric (the DataDecide choice,
  https://arxiv.org/abs/2504.11393) crosses a threshold.
- Measurement: 15 tasks; log-likelihood metrics at the 16 banked points plus denser stage-1 checkpoints around
  each rise (about 40 checkpoints). A rule frozen before unblinding maps each task to the dated jobs whose top
  contexts share its content. Predictor: cumulative matched births up to step t.
- Baselines: validation loss threshold (https://arxiv.org/abs/2403.15796); probes
  (https://arxiv.org/abs/2604.01025); function-vector similarity (https://arxiv.org/abs/2604.08510).
- Refuted if: matched births do not lead the rise (median lead ≤ 0 checkpoints); or they add less than 0.05 R²
  over validation loss on held-out tasks; or births of mismatched jobs predict equally well (then births only
  measure training progress).
- Cost: about 40 checkpoints × 15 tasks of log-likelihood scoring, 6–10 GPU-hours. Births are already dated.
- Caution: births are activation-based, so they date encoding, not readout; the readout lag
  (https://arxiv.org/abs/2610.21025) predicts that births lead behavior. A lag of zero would be informative.

**P2. The documents that made a job (N4), as a mixing tool.**
- Number predicted: the cosine between each document's gradient on a pole's input rows at the earlier checkpoint
  and the weight change realized across the birth interval; then, at small scale, whether removing those
  documents prevents the job.
- Use to a mixing team: it says which domain made which job, a question Olmix leaves open (§3(b)).
- Refuted if: the job's documents sit inside the cosine band of random documents from the same batches; or the
  small-scale retrain without them produces the job anyway.
- Cost: backward passes, about 1 GPU-hour per job at 7B (20 jobs, about 20 GPU-hours); 3–6 GPU-hours per
  small-scale retrained interval.
- Depends on the training data order being released (to verify in `00_ASSETS.md`).
- Nearest prior: verbatim tracing to training data (https://arxiv.org/abs/2504.07096); susceptibility-based
  reweighting on a small LM (https://arxiv.org/abs/2601.13548). None at one neuron of a 7B model.

**P3. RL readiness of a pretraining checkpoint is readable from its neurons.**
- Number predicted: pass@32 on 100 MATH problems for a pretraining checkpoint. RL gains are bounded by base
  pass@k at large k (https://arxiv.org/abs/2504.13837), so this is a proxy for the RL-Zero gain.
- Measurement: 8 checkpoints spanning stage 1 and the final base. Target: pass@32 by few-shot generation.
  Predictor frozen before: activation, on a fixed text set, of the poles whose banked jobs are verification and
  backtracking text ("wait", "let me check"). Compare with validation loss and base pass@1.
- Refuted if: the predictor adds less than 0.05 R² over validation loss on held-out checkpoints, or its rank
  order of checkpoints disagrees with pass@32 (Spearman below 0.5).
- Cost: about 5 GPU-hours of generation plus 1 GPU-hour of forwards. Follow-up: RL-Zero runs from 3 checkpoints,
  about 90 GPU-hours.
- Why OLMo: random rewards help Qwen but not OLMo2 (https://arxiv.org/abs/2506.10947). This file infers that
  OLMo gains are less likely to come from pre-existing shortcuts.

---

## 4. Capability: distillation, merging, test-time compute, self-play

**State, late 2026.** On-policy distillation reports about 9× less compute than SFT for a reasoning student
(https://thinkingmachines.ai/blog/on-policy-distillation/). Reasoning models loop at low temperature, and
distilled students loop more than their teachers (ICML 2026, https://proceedings.mlr.press/v306/pipis26a.html).
A 2026 survey lists the theory of merging as open (https://arxiv.org/html/2603.09938). Two training-time methods
for capability removal beat post-hoc unlearning in their own papers: data filtering
(https://arxiv.org/abs/2508.06601) and gradient masking on a fixed set of neurons and heads
(https://arxiv.org/abs/2512.05648). Self-play with an executor as
verifier trains from zero data (https://arxiv.org/abs/2505.03335).

### 4(a). Numbers they report

- Distillation compute to a target: on-policy distillation reaches 70% on AIME'24 in about 150 steps from a 60%
  SFT checkpoint; about 9× less compute than extrapolated SFT, about 30× counting teacher sampling; IFEval
  85% → 45% after mid-training, restored to 83% (https://thinkingmachines.ai/blog/on-policy-distillation/).
- Test-time scaling: s1-32B AIME24 50% → 57% with budget forcing, from 1,000 examples
  (https://arxiv.org/abs/2501.19393).
- Tokens at fixed accuracy: SEAL −11.8% to −50.4% tokens, up to +11% accuracy (https://arxiv.org/abs/2504.07986);
  probe-based early exit −24% tokens (https://arxiv.org/abs/2504.05419).
- Loop rate at low temperature: reported in Pipis et al.; no rates in the abstract
  (https://proceedings.mlr.press/v306/pipis26a.html).
- Merge retention: DARE drops 90–99% of SFT deltas without loss; a merged 7B ranked first on the Open LLM
  Leaderboard (https://arxiv.org/abs/2311.03099).
- Localized grafting: about 0.01% of parameters carry over 95% of fine-tuning performance
  (https://arxiv.org/abs/2302.06600).
- Capability removal: tamper resistance up to 10,000 steps and 300M tokens of adversarial fine-tuning, more than
  10× post-training baselines (https://arxiv.org/abs/2508.06601); SGTM needs about 7× more adversarial steps
  than RMU (https://arxiv.org/abs/2512.05648); relearning recovers 88% of pre-unlearning accuracy for current
  unlearning methods (https://arxiv.org/abs/2410.08827).

### 4(b). Open problems they name

| problem | their words | source |
|---|---|---|
| mergeability and scale | "How model scale relates to mergeability properties presents another theoretically unresolved frontier." | https://arxiv.org/html/2603.09938 |
| interference | "the question of when interference becomes irreconcilable" | https://arxiv.org/html/2603.09938 |
| distillation with environment reward | "There could be advantages to combining distillation-based per-token rewards with sequence-level environment rewards." | https://thinkingmachines.ai/blog/on-policy-distillation/ |
| loops | "temperature is a stopgap rather than a holistic solution" | https://proceedings.mlr.press/v306/pipis26a.html |
| open-weight safeguards | "Currently, there is not yet a robust science of open-weight model risk management." | https://arxiv.org/abs/2508.06601 |
| labels for removal | "labeling whether data is harmful is expensive at scale" | https://arxiv.org/abs/2512.05648 |
| self-play | "raises concerns about the long-term scalability of relying on human supervision" | https://arxiv.org/abs/2505.03335 |

### 4(c). Interpretability-derived wins, and attempts that failed

| observation | win | size | source |
|---|---|---|---|
| ~4% of heads drive overly short reasoning | ThinkEdit: edit their output projections (0.2% of parameters) | +6.39% on short-reasoning cases, +3.34% overall | https://arxiv.org/abs/2503.22048 |
| reflection and transition thoughts separate in latent space | SEAL steering vector | up to +11% accuracy, 11.8–50.4% fewer tokens | https://arxiv.org/abs/2504.07986 |
| hidden states encode whether an intermediate answer is correct | probe as verifier for early exit | 24% fewer tokens, no accuracy loss | https://arxiv.org/abs/2504.05419 |
| fine-tuning skill sits in a sparse parameter set | grafting | ~0.01% of parameters, >95% of performance | https://arxiv.org/abs/2302.06600 |
| task-essential parameters (~1%) | Localize-and-Stitch merging (as described by the 2026 survey) | "little storage overhead" per the survey; no number read | https://arxiv.org/html/2603.09938 |
| assign heads and MLP neurons to a domain during training | SGTM capability removal | better forget/retain than data filtering under label noise; ~7× more adversarial steps than RMU | https://arxiv.org/abs/2512.05648 |
| fact lookup localized mechanistically | edits and unlearning on those components resist relearning better | no number in the abstract | https://arxiv.org/abs/2410.12949 |

Failed or did not transfer.

| attempt | what happened | source |
|---|---|---|
| localization as a requirement for merging | random dropping of 90–99% of deltas with rescaling works | https://arxiv.org/abs/2311.03099 |
| localization to choose the edit site | "which layer we edit is a far better predictor of performance" | https://arxiv.org/abs/2301.04213 |
| post-hoc unlearning | relearning recovers 88% | https://arxiv.org/abs/2410.08827 |
| post-hoc removal against data filtering | filtering resists more than 10× longer | https://arxiv.org/abs/2508.06601 |
| SAE steering | "For steering, we find that prompting outperforms all existing methods" | https://arxiv.org/abs/2501.17148 |
| SAE probes out of distribution | dense probe AUROC about 0.999; GDM: "we are deprioritising fundamental SAE research for the moment" | https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks |
| reading what fine-tuning wrote from activation differences | traces are clear after narrow fine-tuning, but mixing in pretraining data "largely removes them" | https://arxiv.org/abs/2510.13900 |

### 4(d). Three claims

**K1. A loop fix is a rank-1 change in a few down-projections, and it transfers to OLMo-3 Think.**
- Numbers moved: greedy loop rate on MATH-500 and AIME (responses that hit the token limit with a repeated
  n-gram), tokens per correct answer, and pass@1 held within its band.
- Asset: the replicated edit (128 columns of two down-projections; its direction is the top singular vector of
  a released weight change, `02_FRAMES.md` §2).
- Measurement: first, greedy loop rate on Olmo-3-7B-Think-SFT, -DPO and -Think. If the stages differ, take the
  SFT → DPO and DPO → RL down-projection changes; per matrix, the top singular component. On half of the
  problems, find the two matrices whose rank-1 component, added alone to SFT, cuts loops most. Test on the other
  half.
- Controls: random rank-1 changes of equal norm in the same matrices; the full stage change (the ceiling).
- Refuted if: two rank-1 components recover less than half of the stage's loop-rate reduction; or random rank-1
  changes do as well; or pass@1 drops by more than 2 points. If the stages do not differ in loop rate, the claim
  is cancelled for this chain; the next test is a distilled student, which loops more than its teacher
  (https://proceedings.mlr.press/v306/pipis26a.html).
- Cost: about 0.5 GPU-hour per configuration; about 20 configurations; about 10 GPU-hours.
- Nearest prior: ThinkEdit (0.2% of parameters, https://arxiv.org/abs/2503.22048); loop onset
  (https://arxiv.org/abs/2601.05693).

**K2. Merge by neuron, with the jointly trained model as the ceiling.**
- Number moved: mean of the math and code target scores for RL-Zero-Math merged with RL-Zero-Code.
- Ceiling: RL-Zero-Mix, trained on both. Baselines: task arithmetic, TIES, DARE.
- Measurement: per neuron, both deltas. Classify neurons as math-only, code-only or both. Neuron merge: each
  task's top-k neurons; for the overlap, a rule frozen beforehand (larger change wins). Matched parameter count
  across methods.
- Refuted if: the neuron merge is within the seed band of TIES or DARE; or the share of overlapping neurons does
  not track the interference measured by task arithmetic.
- Cost: weights plus about 8 merged models × 1 GPU-hour, about 10 GPU-hours.
- Use: a measured answer to "when interference becomes irreconcilable" (§4(b)) at the neuron level.

**K3. Revoke a narrow capability through its young poles.**
- Numbers moved: forget-domain loss and accuracy against wikitext perplexity and MMLU; steps for relearning to
  restore half of the lost accuracy.
- Measurement: a domain with many late-born jobs among the 1,238 dated jobs (for example the chemistry cluster
  around the lithium job). Remove the output columns of k of its late-born poles, k = 32, 128, 512 (k ≥ 32
  because one pole removed is near the noise). Then fine-tune on 100 held-out domain documents.
- Controls: k energy-matched random poles; k old poles with related labels; RMU at matched retain loss.
- Refuted if: the forget/retain ratio of the young-pole edit is not better than energy-matched random; or
  relearning recovers as fast as after RMU (88% recovery is the published level for current methods,
  https://arxiv.org/abs/2410.08827).
- Cost: edits and evaluations about 4 GPU-hours; 6 relearning runs about 6 GPU-hours; RMU about 2 GPU-hours.
  About 12 GPU-hours.
- Nearest prior: SGTM (https://arxiv.org/abs/2512.05648); mechanistic unlearning
  (https://arxiv.org/abs/2410.12949); the warning that localization need not pick the edit site
  (https://arxiv.org/abs/2301.04213).

---

## 5. The five highest-leverage intersections, ranked

Ranking: how much an RL or systems engineer would act on the answer, times the chance of a clear answer, divided
by cost. Novelty is not yet audited (`04_LITERATURE.md`).

**1. Sampler–trainer mismatch, localized to channels (S1).** RL engineers lose stability and asymptote to this
mismatch: ScaleRL's FP32 logits alone moved the fitted asymptote from 0.52 to 0.61
(https://arxiv.org/html/2510.13786), and FP16 everywhere is the current global fix
(https://arxiv.org/abs/2510.26788). The matching interpretability observation is reported across many LLMs: a
few channels carry activations about 100,000 times larger than the rest (https://arxiv.org/abs/2402.17762), and
a single weight can produce them (https://arxiv.org/abs/2411.07191). If the residual mismatch sits in those
channels and sink positions, a systems engineer gets a precision rule that costs a few fp32 columns instead of
a format change. If it is spread out, the same 3–4 GPU-hour measurement says so. The engineer reads it because it
names which part of the model must be numerically exact during rollouts.

**2. What RL wrote, as a transplant with a DARE control (R1).** The literature disagrees on whether RL's update
is local: 5–30% of parameters move (https://arxiv.org/abs/2505.11711), but that sparsity disappears in fp32 and
the updates are full-rank (https://arxiv.org/html/2509.04259), and rank-1 LoRA suffices
(https://thinkingmachines.ai/blog/lora/). A top-k neuron transplant into the base, scored on MATH-500 and AIME,
with energy-matched random and DARE controls beside it, settles whether an RL gain can be stored, merged or removed
as a neuron set. It also gives the list of neurons and their jobs. An RL engineer reads it to decide where to
place adapters and whether a gain can be shipped as a small delta. Cost about 20–25 GPU-hours.

**3. Forgetting predicted from weights (R2).** Forgetting is the named cost of post-training. The best predictor
so far, forward KL on the new task (R² 0.71 at 3B), needs rollouts; weight-size metrics fail
(https://arxiv.org/html/2509.04259). A weights-only score that weights change by the age and reader count of each
pole would rank recipes and checkpoints before any evaluation is run, using 8 to 11 released OLMo-3 stages
and variants. An RL engineer reads it because it predicts a recipe's forgetting before the evaluation sweep. Cost
about 15 GPU-hours.

**4. Loops as a rank-1 fix (K1).** Reasoning models loop at low temperature, distilled students loop more than
teachers, and the authors call temperature "a stopgap" (https://proceedings.mlr.press/v306/pipis26a.html). The
lab already holds a replicated two-matrix edit that stops verbatim loops, and its direction is the top singular
vector of a released weight change. Showing the same form in the OLMo-3 Think stages would give a measured drop
in loop rate and in tokens per correct answer at fixed pass@1. A test-time-compute engineer reads it because loops
cost tokens and truncate answers. Cost about 10 GPU-hours.

**5. Activation sparsity in grouped rollouts (S2).** Activation sparsity gives decode speedups (TEAL 1.8×,
https://arxiv.org/abs/2408.14690; Prox 1.99× at 70%, https://arxiv.org/abs/2607.27591), and MLP sparsity
disappears at serving batch sizes (https://arxiv.org/abs/2505.14884). RL rollouts are not ordinary serving:
GRPO samples many continuations of one prompt, and prompts can be grouped by their neuron signature. A 2–3
GPU-hour measurement of the union of active neurons under three batch compositions either shows that a rollout
speedup is available or rules it out. A systems engineer on an RL team reads it because every RL step generates
rollouts and RL runs are large (MiniMax-M1's full RL run: 512 H800 GPUs for about three weeks,
https://arxiv.org/abs/2506.13585); the share of that cost spent on rollouts is not verified here.

Runner-up, not ranked because its reader is a pretraining engineer: P1, job births lead benchmark rises. It is the
claim that uses the lab's rarest asset (1,238 dated births on a 7B model) and competes with 2026 probe and
function-vector predictors (https://arxiv.org/abs/2604.01025, https://arxiv.org/abs/2604.08510).

---

## 6. Not verified, and caveats

- Seen only in search snippets: MiniMax-M1's LM-head finding and 0.9 → 0.99 correlation; the gpt-oss RL
  sink-kernel mismatch (LinkedIn post on Hugging Face); LTE's batching proposal; the Jaccard 0.29–0.33 for
  induction circuits and the OLMo token counts (23B induction, 264B sink) in 2606.02378; the OLMo-2 retrieval-head
  counts over 45 stage-1 checkpoints in 2606.21249; DeepSeek-V3.2's top-k of 2048 and its 6–7× long-context cost
  reduction; OpenThinker3-1.5B's 30% greedy loop rate; TEAL's batch-size limitation stated in its body;
  the head-retention counts in 2605.28860; the claim that 2604.01025 used OLMo3-7B (its abstract does not say).
- The gpt-oss learned sink is quoted from the Hugging Face launch post (opened), not from OpenAI's model card.
- The Anthropic reward-hacking paper (https://arxiv.org/abs/2511.18397) was read at abstract level only; its
  full-text fetch was refused by the fetch tool.
- Model names in 2609.19101 (Kimi K3, GLM 5.2, Qwen 3.8 Max, the GPT-5.6 Sol monitor) are as the paper states
  them; they were not checked elsewhere.
- 2610.17824 (verifier-noise survey) is by two independent researchers; its quotes are used for problem
  statements only.
- The onlychan.xyz post is anonymous, one model, one task, one seed; it is listed as a weak negative.
- Pages fetched through a summarizing tool; quotes were requested verbatim, but a second reader should spot-check
  the five quotes in §5 against the source text before this file is cited.
