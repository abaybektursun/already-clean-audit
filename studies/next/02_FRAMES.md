# Frames — how to think about the next study

Fable, 2026-11-05. This is the abstract layer. Idea cards are in `03_IDEAS.md`; this file says where the
cards come from and what "wow" means as a test.

## 1. The stance we adopt from the essay

Four claims, taken as working assumptions until a measurement says otherwise.

1. A network's neurons are its parts. The neuron read and write directions are the right basis for the
   residual stream. No reconstruction model is needed to begin.
2. New computation is a route between existing parts, built as close to the output as the task allows.
   A neuron grows an offshoot into adjacent territory; it does not pick up unrelated jobs.
3. Growth is driven by readers. A weight only changes if something downstream multiplies its output and
   that product reaches the loss. So the parts that exist first shape what can be learned next.
4. You can watch all of this happen over training. The essay says it lacked checkpoints for large language
   models. We have them.

The essay's optimism is a method, not a mood: look at the parts directly, test an explanation until it can
lose, and let each explained part constrain the next.

## 2. Our edge, stated as assets nobody else has in one place

- OLMo-3 7B with 1,421 stage-1 checkpoints public, 16 of them banked with each of 600 poles' top-20 contexts
  out of 20.6 million tokens, and birth dates for 1,238 late-born jobs. (`00_ASSETS.md` verifies.)
- If the training data order is released (to verify), the documents between any two checkpoints are known.
- A robot policy chain (Qwen3-4B → Molmo2-ER → MolmoAct2 → LIBERO fine-tune) cached, with a LIBERO
  evaluation harness that scores 200 episodes in about seven minutes of GPU time.
- A record of negatives: readable is not used (depth tokens), a label is not a function (neuronrw), one pole
  removed is near the noise (neuronoff). Every new idea must survive these three.
- A replication of a weight-space edit (128 columns of two down-projection matrices) that stops a model's
  verbatim loops, and the finding that its direction is the top singular vector of the released weight change.

## 3. Five lenses

**Time.** The network at step t is a program. Each training step patches it. We can read 1,421 patches.
Questions a patch answers: which rows move, in what order (input side, output side, readers), how local the
change is, and which documents caused it.

**Computation against language.** Most of our labels are nouns: "lithium", "legal citations". But a large
share of poles have jobs that are states, not words: inside a quoted string, second item of a list, the
entity bound to a pronoun, a counter. A state job looks unrelated in content and is read as "not
understood" by a judge that reads text. The already-clean study left 32 to 42 percent of poles not
understood. Some of those are computation. A state job is defined by invariance: substitute the content and
the pole still fires. The residual stream holds the variables; attention moves them; MLP neurons test them.

**Representation.** A representation matters only through its readers. For every pole we can measure two
separate things: how readable it is (judge accuracy on its contexts) and how used it is (activation times
gradient of the loss on those contexts, or the loss change when it is removed). The two can come apart, and
their order in time is a fact about learning: is a job used from the step it exists, or is there a period
when the pole fires and nothing reads it?

**Transformation.** A weight edit is a program written by hand. The essay's growth rule says where the
network itself writes: into the inputs of parts that already have readers, near the output. That is a
recipe for where a human edit should go, and a testable claim about which edits cost the least elsewhere.
Transplanting a job's weights between checkpoints is the same test from the other side: if a job lives in
its neuron's rows and columns, it should move with them.

**Geometry.** How many directions of the residual stream are read at each layer, how that count changes
over training, and whether the subspace that carries computation (the Jacobian lens object, `01_JLENS.md`)
is built early and filled with content later. The record already holds one unread coordinate (the band:
about a quarter of layer energy read at low gain).

## 4. The practical lens: do what the capability engineers do, from inside the model

The people who move the field now optimize four things: post-training with reinforcement learning (sample
efficiency, reward hacking, entropy collapse, credit assignment over long horizons), inference systems
(KV-cache size, quantization, activation sparsity, speculative decoding, routing), pretraining scale (data
mixing, curriculum, learning-rate schedules, what to spend the next token on), and capability itself
(distillation, merging, test-time compute). Each has a number they report. Interpretability earns their
attention only by moving one of those numbers, or by predicting one before it is measured.

What a neuron-level, training-aware view can offer them, as claims to test:

- **Updates are patches.** If a post-training stage (SFT, DPO, RL) changes a few neurons' jobs and their
  readers, the gain is a small patch that can be copied, merged or revoked without the data. The measurement
  is a transplant with a matched random control. (`03_IDEAS.md` N16.)
- **Where to write.** Edits on young poles (late-born, few readers) should cost less elsewhere than edits
  on old poles (many readers). That is a rule for model editing and for parameter-efficient fine-tuning.
  (N6.)
- **What to drop.** Directions no reader uses can be dropped from the cache at inference. The record's band
  is a candidate. The claim must beat the existing low-rank cache methods at matched perplexity. (N10.)
- **What the data did.** With the data order, a job's birth can be traced to documents. That is data
  attribution at the resolution of one neuron, and a tool for curriculum and mixing decisions. (N4.)
- **What RL wrote.** The OLMo-3 chain releases every post-training stage. The question "what did RL write,
  and into which parts" can be answered in the neuron basis with no reconstruction model. (N16.)

The agent brief for this lens is in `JOURNAL.md`; its report lands in `04_PRACTICAL.md`.

## 5. What "wow" means, as a test

A result passes the six-second test if a reader sees one picture or one before/after and understands the
claim without text. It passes the thirty-second test if the same reader, having read the one sentence under
the picture, cannot think of the obvious confound, because the picture already shows the control.

Three shapes pass both:

1. A small, understood change produces a large, visible effect, with a matched random change beside it that
   does not. (A weight edit; a transplant; a robot that does something different.)
2. A prediction made before the measurement lands where nobody expected. (A birth predicted; a document
   named; a use curve.)
3. A number an engineer already cares about moves. (Success rate, perplexity at a cache size, accuracy
   after a patch.)

A gallery, a slider or a count of neurons with one job does not pass on its own. It is the picture under
which one of the three shapes is shown.

## 6. What we refuse to do

Replicate a published result as the headline. Depend on a judge whose agreement with a second reader is
below 0.2. Push a neuron far outside its natural range and call the output a mechanism. Report a number
without the saved input and output beside it.
