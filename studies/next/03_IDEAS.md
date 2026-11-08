# Idea cards — v1, 2026-11-05 (Fable; agents extend in later sections)

Each card: the claim, the observation that refutes it, the cheapest first measurement, the demo if it
lands, cost in GPU-hours, the asset that gives us an edge, and the nearest prior work I know of (to be
audited in `04_LITERATURE.md` before any score is final). Cost classes: W weights only; F forward passes;
B forward and backward passes; E robot evaluation; T training. "Dated births" means the 1,238 late-born
jobs with birth points from the already-clean study.

Status tags: DRAFT (not audited), AUDITED (literature read), SCORED, PICKED, SET ASIDE.

---

## N1 — Input first, readers later: the order in which a job's birth is written into the weights
Lens: time, transformation. Cost: W. Status: DRAFT.
**Claim.** When a pole acquires a job, its input rows (gate and up) move first; its output column barely
turns; the readers that multiply that column specialize afterwards. This is the essay's growth rule stated
in weights: a new job is written into the inputs of a part that already has readers.
**Refuted if** at the birth interval the output column's direction changes by more than the input rows',
relative to the same pole's drift in non-birth intervals, in the majority of dated births; or if the
readers' rows change before the input rows.
**First measurement.** For 100 dated births (and the lithium neuron L03N6372), at every stage-1 checkpoint
from 1,000 steps before to 20,000 steps after the birth point: cosine of consecutive gate rows, up rows,
down columns, and of the next layers' rows restricted to the direction of the down column. Baseline: the
same pole at matched non-birth intervals; random poles at the same intervals. Read only the needed rows
from each checkpoint (safetensors header offsets with range requests), no full downloads.
**Demo.** Three curves per neuron over training with the birth marked. The lithium neuron first.
**Edge.** Dated births and 1,421 checkpoints. Nobody has both on a 7B model.
**Nearest prior.** Studies of weight drift across Pythia checkpoints; crosscoders across checkpoints
(2610.21025); none at the resolution of a dated job.

## N2 — Silent or used: is a job read from the step it exists?
Lens: representation, time. Cost: B, about 5 GPU-h. Status: DRAFT.
**Claim.** The attribution of a pole to the loss on its job's contexts (activation times gradient, and the
loss change when the pole is zeroed on those contexts) is above the chance band at the birth point itself.
The essay's reader-driven growth implies there is no silent period.
**Refuted if** at birth the use measure sits inside the band of random poles on the same contexts while the
activation is already at its top-20 level, for more than a stated share of births; the share is frozen on
control poles before unblinding.
**First measurement.** 100 dated births, the banked contexts and positions at each of the 16 points,
forward and backward at each point, the pole's activation times the gradient of the next-token loss with
respect to it, and the zero-ablation loss change. Controls: random poles on the same contexts; the same
pole on contexts outside its job.
**Demo.** Two curves per neuron, fire and use, over the slider. A use curve that lags the fire curve is the
picture; one that does not is the essay confirmed at 7B.
**Edge.** Banked contexts with positions at every point; no other group has to find them.
**Nearest prior.** Attribution patching (AtP) as a method. Added 2026-11-05 (`04_WDD.md`): on Pythia-410m and
OLMo-1B, removing the future neuron's output direction from the residual stream already costs 0.0047 nats two
checkpoints before the neuron is recruited (random direction 0.0002), and the class's loss falls no faster
across recruitment than at matched control positions. So the job's function may exist as a direction before
any pole carries it, and the birth is a consolidation. The card therefore measures two things at every point:
the cost of zeroing the pole, and the cost of projecting its final output direction out of the stream. If the
second precedes the first, the concept exists before its neuron.

## N3 — The route at birth: an attribution graph in the neuron basis, over training
Lens: computation, time. Cost: B, about 3 GPU-h. Status: DRAFT.
**Claim.** The upstream contributors to a job at its birth (neurons and heads with the largest activation
times gradient into the pole, on the job's contexts) are older than the job and related to it in content.
New computation is a route between existing parts.
**Refuted if** the contributors at birth are themselves born in the same interval, or are judged unrelated,
more often than for matched control poles; or if the top contributors change wholesale between the birth
point and the next.
**First measurement.** 30 dated births. At each of the 16 points, the top-10 upstream neurons and heads by
direct contribution to the pole's pre-activation on its contexts. Each contributor carries its own birth
date from the banks (if among the 600 poles) or from a fresh top-context harvest at two points.
**Demo.** The slider, with the route drawn under the neuron: a circuit forming over training, every node a
named neuron with its own job.
**Edge.** Same as N2, plus the route is in the model's own neurons, no transcoder.
**Nearest prior.** Attribution graphs with transcoders (Anthropic 2025); the essay's own neuron-basis
graphs on vision models; `studies/unite/PROPOSAL.md` Arm 1 asked the input-weight version of this.

## N4 — The documents that made the job
Lens: time, practical (data attribution). Cost: B at 7B about 1 GPU-h per job; T at small scale about
3–6 GPU-h per interval. Status: DRAFT. `00b_DATA_ORDER.md` (2026-11-05): the stage-1 batch order is a seeded NumPy shuffle computable
from public files; one 1,000-step interval is 512,000 instances of 32,768 bytes (16.8 GB) by HTTP range requests;
one GPU loss check on batch 60,000 (expected 2.1240015, public training log) remains to be run, about 0.2 GPU-h.
**Claim.** The documents seen in the birth interval that contain the job's tokens carry a gradient on the
pole's input rows aligned with the weight change actually realized; documents without those tokens do not.
At small scale (a suite with checkpoints and data order), retraining the interval without those documents
prevents the birth, and with them reproduces it.
**Refuted if** the cosine between per-document gradients and the realized change is at the chance band for
the job's documents; or if the counterfactual retrain births the job anyway (the job was due regardless of
those documents). Added 2026-11-05 (`04_RECENT.md`): gradient-similarity attribution tracks answer format
more than content (arXiv 2609.19589), so the control set must be format-matched documents without the
job's tokens, not random documents.
**First measurement.** The lithium job. Enumerate the batches between the checkpoints that bracket its
birth; find documents with " lithium"; gradient of the loss on each with respect to the pole's gate and up
rows at the earlier checkpoint; cosine with the realized change. Control: documents without the token;
random poles.
**Demo.** The slider with a second panel: the documents that taught it, with the counterfactual curve.
**Edge.** Dated births plus a public data order. Data attribution is usually at the level of an output,
never of one neuron's job.
**Nearest prior.** TRAK, influence functions, datamodels (output-level); "feature-level data attribution"
to be searched.

## N5 — Move a newborn skill with k neurons
Lens: transformation. Cost: W plus evaluations, about 2 GPU-h per skill. Status: DRAFT.
**Claim.** For a skill that appears between two adjacent checkpoints (1,000 steps apart), copying the
weight change restricted to the k most-changed neurons (input rows and output column) from the later
checkpoint into the earlier installs most of the skill's accuracy gain, with k a small fraction of all
neurons; a count- and energy-matched random selection installs none.
**Refuted if** the k-neuron transplant recovers less than half the gain at any k below five percent of
neurons, or if the random transplant recovers as much.
**First measurement.** Choose skills from OLMo-3's published per-checkpoint evaluation curves (to find in
`00_ASSETS.md`): the sharpest rise between adjacent checkpoints. Rank neurons by the norm of their weight
change; transplant the top k for k in 100, 300, 1,000, 3,000; evaluate; controls as above; also the
transplant of everything except the top k.
**Demo.** One chart: accuracy of the earlier checkpoint as k grows, against the random control, with the
named neurons and their jobs under it.
**Edge.** Adjacent public checkpoints at 7B; the dating method from the already-clean study.
**Nearest prior.** Skill localization by grafting after fine-tuning (Panigrahi et al. 2023); wdd-descendants
e578 (Pythia-410m: every transplant closure is negative; the whole neuron −0.020, the readout −0.873, so the
transplant moves the model away from the donor; `04_WDD.md` §8 corrects the sign misread in `04_LITERATURE.md`); copying one mature attention head into an
early Pythia checkpoint recovers 35–68% of the later gain against eight random heads (arXiv 2610.04119,
`04_RECENT.md`). So the card needs three arms: neurons, heads, readout.

## N6 — Edit the young: collateral cost of a weight edit by the age of the pole
Lens: transformation, practical (editing). Cost: F, under 2 GPU-h. Status: DRAFT.
**Claim.** An equal-size edit (zeroing or scaling the output column, or a rank-one write of the same norm)
on an early-born pole costs more wikitext perplexity than on a late-born pole, because early-born poles
have more readers. Age predicts collateral better than activation energy or column norm.
**Refuted if** collateral by age is flat after matching on energy and norm.
**First measurement.** 300 poles with birth dates, three edit types, wikitext and a held-out code set.
Regress collateral on age, energy, norm, layer.
**Demo.** One scatter: collateral against age; and a rule an editor can use.
**Edge.** Birth dates.
**Nearest prior.** ROME, MEMIT and their collateral analyses; none conditions on training age.

## N7 — Computation neurons against content neurons, by invariance
Lens: computation against language. Cost: F, 1–2 GPU-h for 100 poles at 4 points. Status: DRAFT.
**Claim.** A measurable share of poles have state jobs: the activation survives substitution of the
content words in its top contexts (structure kept) but not substitution of the structure (content kept).
State jobs are born earlier than content jobs.
**Refuted if** the share is under five percent, or the birth-point distributions of the two kinds do not
separate.
**First measurement.** For 100 poles drawn from the not-understood set and 100 from the one-job set: 20
banked contexts each, two edited versions per context, activation at four training points.
**Demo.** A gallery of the model's registers (quote state, list index, bracket depth, bound entity) and
their birth order on the slider: the model builds its registers first.
**Edge.** The not-understood set with its contexts; the 16 points.
**Nearest prior.** Binding IDs (Feng and Steinhardt), positional and counting neurons, "structural"
probes; a developmental ordering is not known to me.

## N8 — When is the workspace built: the Jacobian lens subspace over training
Lens: geometry, computation. Cost: B, about 2 GPU-h. Status: SET ASIDE, superseded by N25 (the lens object is not a subspace; see `01_JLENS.md` §1.3).
**Claim.** The low-dimensional subspace that the Jacobian lens finds at the final checkpoint is already
present, with high overlap, early in training (by the point where induction behaviour exists), and the
neurons that write into it are disproportionately the state neurons of N7.
**Refuted if** the overlap with the final subspace is at the chance band for random subspaces of the same
dimension until late in training, or if content and state neurons write into it equally.
**First measurement.** The lens at the 16 points on one fixed text set; principal angles between the
subspaces; the projection of each pole's output column onto the subspace.
**Demo.** One curve: overlap with the final workspace over training.
**Edge.** The 16 points and the pole set.
**Nearest prior.** The paper itself and its follow-ups; to be read.

## N9 — Weight-space programming for a robot: rewire one instruction
Lens: transformation, demo. Cost: E plus B, about 5 GPU-h. Status: DRAFT.
**Claim.** In MolmoAct2-LIBERO's language tower, the neurons that distinguish two instructions (for example
"open the drawer" against "close the drawer", or two target objects) are few enough that a weight edit of
at most 256 output columns makes one instruction execute the other, with the rest of the suite unchanged.
**Refuted if** no such edit moves the target task by more than 30 points of success while the others stay
within the seed band, although activation patching at the same sites does.
**First measurement.** Activation patching between the two instructions at the text positions, per layer,
to find the site; then neuron-level attribution at that site; then the column edit; LIBERO evaluation of
the edited model on all tasks.
**Demo.** A video: the robot told to open the drawer closes it, every time; and the success table.
**Edge.** The cached chain and a harness that scores 200 episodes in about seven minutes.
**Nearest prior.** Steering of VLAs by activation; instruction-grounding probes; weight-level rewiring
not known to me.

## N10 — Drop what nobody reads: an inference win from reader gain
Lens: geometry, practical (systems). Status: SET ASIDE. Published: sensitivity-weighted residual pruning beats SliceGPT on Llama-3.1-8B (arXiv 2609.35579; `04_LITERATURE.md` N10).
**Claim.** The directions of the residual stream with the lowest reader gain (the band) can be dropped from
the KV cache or from the stream at inference with a smaller perplexity cost at a given memory saving than
rank-by-energy, and compete with published low-rank cache methods at the same budget.
**Refuted if** the gain-ranked drop is no better than energy-ranked, or both are beaten by the published
methods at every budget.
**Edge.** The band measurement exists. Prior art is large; audit first.

## N11 — Explain at birth: checkpoints make neurons easier to explain
Lens: time, practical (auto-interpretability). Cost: judge calls only. Status: DRAFT.
**Claim.** An explanation written from a pole's contexts at its birth point scores higher with the judge on
the final checkpoint's contexts than an explanation written from the final contexts, because a young job
has fewer offshoots. The lithium neuron is the example: 20 of 20 at step 60,000, 0 of 50 at the end.
**Refuted if** judge accuracy is equal or lower for birth-point explanations.
**First measurement.** 200 late-born jobs; two explanations each; the judge on held-out contexts.
**Demo.** A bar: accuracy of the two explanation sources; a tool recommendation.
**Edge.** The banks at 16 points.

## N12 — Gradient starvation measured
Lens: time, representation. Cost: B, like N2. Status: DRAFT.
**Claim.** Before a job is born in pole A, several poles receive gradient on the job's contexts; after the
birth, the gradient to the others falls toward chance while A's rises. The winner takes the signal.
**Refuted if** the competitors' gradient on those contexts is unchanged across the birth.
**First measurement.** Shares N2's passes; record the gradient to all poles in the layer on the job's
contexts at each point.
**Demo.** One picture: many candidate curves, one rises, the rest fall.
**Nearest prior.** Grokking circuit-formation studies; coherent gradients; not at the level of a dated job.

## N13 — What RL wrote: the OLMo-3 post-training chain in the neuron basis
Lens: practical (post-training), transformation. Cost: F plus evaluations, 3–6 GPU-h. Status: DRAFT;
`00_ASSETS.md` §2 verifies the chain: Think-SFT 43 checkpoints (every 1,000 steps), Think-DPO 1, Think RL 55 (every 25 steps to 1,375), Instruct 8, and RL-Zero-Math 19 (every 100 steps) on `allenai/Olmo-3-1025-7B` post-training repos.
**Claim.** Across Base → SFT → DPO → RL, each stage changes the jobs of a small share of poles and the
rows of their readers; the stage's benchmark gain transplants with the k most-changed neurons into the
previous stage, and a matched random set transplants nothing. RL's patch is the smallest and the most
concentrated near the output.
**Refuted if** the gain needs more than five percent of neurons, or the random set recovers as much, or
the job changes are spread uniformly across depth.
**First measurement.** Pairwise weight differences between stages; rank neurons; transplant top-k; evaluate
on the stage's own benchmark (math or instruction following) and on a held-out one.
**Demo.** One chart per stage: gain recovered against k; and the list of neurons RL rewrote, with their
jobs before and after.
**Edge.** Public stage checkpoints on a model whose neurons we already dated.
**Nearest prior.** "RL fine-tunes small subnetworks" (sparse RL updates, 2025); task arithmetic; not in the
neuron-job basis, to be audited.

## N14 — Which heads deliver the job
Folded into N3: the heads whose attention moves the job's tokens to the pole, and whether they exist before
the job. Not scored separately.

## N15 — Double-negative neurons over training
Lens: time. Cost: W plus banked activations. Status: DRAFT; low wow. When do poles whose top activations are
products of two negative pre-activations appear in OLMo-3 (0.42 percent at the end against 4.71 percent in
Qwen3-4B)? Cheap, chartable, not a headline.

## N16 — Neuron-level merging: move a capability between siblings by its neurons
Lens: practical (capability), transformation. Cost: W plus evaluations, 3–6 GPU-h. Status: DRAFT.
**Claim.** Between two models that share a parent (for example OLMo-3 Instruct and OLMo-3 Think, or two
post-training stages), the neurons whose jobs differ carry the capability difference. Copying the k
most-changed neurons (rows and column) from the donor into the recipient transfers more of the donor's
benchmark gain, at less loss on the recipient's own benchmarks, than whole-weight task arithmetic or sparse
merging at the same parameter budget.
**Refuted if** neuron-level transfer is no better than TIES or DARE style sparse merging at matched budget,
or if the transferred gain comes with an equal loss elsewhere.
**First measurement.** One sibling pair; rank neurons by job change; transplant at four budgets; evaluate
donor benchmark and recipient benchmark; baselines at the same budget.
**Demo.** One chart with two curves per method: gain on the donor's task, loss on the recipient's.
**Edge.** The job-change ranking from the already-clean instrument; nobody merges in the job basis.
**Nearest prior.** Task arithmetic, TIES, DARE, model grafting (parameter-level, not job-level); to audit.

## N17 — Repair forgetting after a robot fine-tune by restoring k neurons
Lens: practical, representation. Cost: W plus text and robot evaluations, 3–5 GPU-h. Status: DRAFT; prior
lowered by sinkpatch (LIBERO loss spread over layers 12–31).
**Claim.** The plain-text loss MolmoAct2-LIBERO shows against its parent is carried by the k neurons whose
jobs changed most; restoring their rows and columns from the parent recovers most of the text loss with no
loss of LIBERO success. Diffuse erosion predicts the reverse.
**Refuted if** restoring the long tail at matched parameter mass recovers as much or more, or if the
restore costs LIBERO success beyond the seed band.
**First measurement.** Rank neurons by weight change; restore top-k against tail at matched mass; wikitext
perplexity; LIBERO 200 episodes.
**Demo.** A robot that keeps its skills and gets its language back, with the k neurons named.
**Edge.** The cached chain and the fast harness.
**Nearest prior.** Locate-then-Merge (diffuse erosion), our D2 in `studies/already-clean/DIRECTIONS.md`.

## N18 — Curriculum from neuron births (speculative)
Lens: practical (pretraining), time. Cost: T at small scale, 5–10 GPU-h. Status: DRAFT; depends on N4.
**Claim.** If N4 can name the documents that create a job, then choosing the next interval's documents by
neuron-level attribution accelerates a target skill at small scale against a random mix of the same size.
**Refuted if** the attributed mix gives no speedup over random at matched tokens.
**Demo.** Two loss curves on the target skill. An engineer's number.
**Nearest prior.** Data selection by influence (output level); not neuron level.

## N19 — What RL kills: entropy and diversity in the neuron basis
Lens: practical (post-training). Cost: F plus evaluations, 2–4 GPU-h. Status: DRAFT; the RL stages have
intermediate checkpoints: Think RL 55 every 25 steps and RL-Zero-Math 19 every 100 steps (`00_ASSETS.md` §2).
**Claim.** The loss of output diversity after reinforcement learning (the pass-at-k collapse that RL
engineers name) is carried by a small set of poles whose readers RL disconnected; restoring those readers
from the pre-RL stage restores pass-at-k diversity with a small cost to pass-at-1.
**Refuted if** the diversity loss is spread across thousands of poles, or restoring the set costs as much
pass-at-1 as it returns in pass-at-k.
**First measurement.** Pre-RL and post-RL stage; entropy-related poles (final-layer poles writing into the
unembedding's low-rank subspace, per Stolfo) and the poles with the largest job change; restore; measure
pass-at-k on the RL benchmark.
**Demo.** A pass-at-k curve before, after, and after the restore. The number RL people want.
**Nearest prior.** Entropy neurons (Stolfo 2024), entropy collapse in RLVR (2025–26), sparse RL updates.

## N20 — The developmental census: every pole of a 7B model, dated
Lens: time. Cost: analysis of banked JSON on one CPU instance (30.9 GB on S3), no GPU. Status: DRAFT.
**Fact from `00_ASSETS.md`.** The all-neuron harvest exists at all 16 points: top-50 contexts and activation
histograms for all 704,512 poles. Births were dated only for 600 poles. The same rule applied to every pole
gives the birth distribution of a whole 7B model: births per layer per interval, the share of poles whose
final job is present at each point, the share that change job after step 100,000.
**Claim.** Birth times are ordered by depth (early layers settle first) and most poles have their final
job before 20 percent of training; the late intervals add jobs to a small share of poles concentrated in a
few layers.
**Refuted if** births are uniform over depth and time, or if more than a third of poles change job in the
last half of training.
**First measurement.** A mechanical birth rule (first point at which the final top-50 positions appear in
the point's top-50), applied to all poles; the same rule on the 600 poles against their clustered dates as
calibration.
**Demo.** One picture: 32 layers by 16 points, each cell the number of births. Nobody has drawn it for a 7B
model.
**Edge.** The harvest. It would cost another group 16 full-corpus harvests to repeat.
**Nearest prior.** Crosscoders across Pythia checkpoints; stage-wise development; none at all-neuron scale.

## N21 — Same data, two architectures: is development universal?
Lens: time, geometry. Cost: F, 16 harvests of a second model, about 20–40 GPU-h; or 4 points for 8–10 GPU-h.
Status: DRAFT; depends on `Olmo-Hybrid-7B` sharing the stage-1 data mix (to verify).
**Claim.** `allenai/Olmo-Hybrid-7B` (1,419 stage-1 checkpoints, created 2026-01-28) trained on the same
mix shows the same jobs born in the same order: for jobs matched by their contexts, birth points agree
beyond chance, and the depth ordering of N20 is the same.
**Refuted if** matched jobs' birth points correlate at chance level across the two models.
**Demo.** Two birth maps side by side, and one scatter of birth point against birth point.
**Nearest prior.** Universality of features across seeds (Olah; Gurnee's universal neurons); not across
architectures over training.

## N22 — The model's variables over training: slot-dependent arithmetic on OLMo-3
Lens: computation against language, time. Cost: B, 3–6 GPU-h. Status: DRAFT. From `01_JLENS.md` Seed 5.
**Fact.** The workspace paper's clearest computation result is one curated example on Sonnet 4.5: the value
63 written into the slot of the first intermediate of (a+b)×c+d gives 133, into the second gives 70; the
downstream function depends on the slot (paper §A.24.2). It failed to replicate on Qwen 3.6 27B, whose
tokenizer splits digits. OLMo-3's tokenizer keeps every number 0–999 as one token (checked 2026-11-04).
**Claim.** OLMo-3 7B at the final checkpoint shows slot dependence above a matched random swap and above
direct answer-token substitution; across the 16 points, readout of the intermediates appears before slot
dependence, and slot dependence appears before the model answers these expressions correctly.
**Refuted if** the raised answer does not depend on the slot at the final checkpoint; or slot dependence and
readout appear at the same point; or the effect is not above the two floors.
**First measurement.** 100 expressions the final checkpoint answers without chain of thought; lens fits at
the final point (penultimate target); locate the intermediate layers by readout; one swap sweep (two slots,
five values, three layer bands) with both floors; then the same at four earlier points.
**Demo.** A picture over the slider: the intermediate value becoming readable, then becoming a variable
the next operation uses. "The model gets a variable at step X."
**Edge.** The 16 points plus a tokenizer that makes the test possible on an open model.
**Nearest prior.** The paper's one example; Nanda's and Zeisler's open-model replications (answer swaps
dominate); no across-training test found by the reader.

## N23 — An action workspace in a robot policy
Lens: representation, demo. Cost: B plus E, 3–6 GPU-h. Status: DRAFT. From `01_JLENS.md` Seed 8.
**Fact.** MolmoAct2's continuous actions come from a flow-matching action expert whose every block attends
to the key and value tensors of the matching backbone layer; the LM head also predicts discrete action
tokens (arXiv 2605.02881 §4.1–4.2). So the executed action can read any backbone layer directly.
**Claim.** A lens fit from the Jacobian of the action expert's velocity output with respect to each backbone
layer (an "action lens") has band structure that differs from the text lens on the same backbone; swapping
the instructed object's coordinates for a distractor present in the scene moves the executed trajectory
toward the distractor above a matched random swap.
**Refuted if** the action lens has no band structure beyond the step-0 style floor; or the swap is at the
random floor while editing the instruction text does redirect the policy.
**First measurement.** Offline, no simulator: both lenses on 20 recorded episodes under two instructions
differing in one object; only if the readouts separate, one closed-loop swap batch on one LIBERO task.
**Demo.** A video: the robot reaches for the object we wrote into its workspace.
**Edge.** The cached chain and the fast harness. No lens study of a vision-language-action policy was found.

## N24 — What the famous edit wrote (a short note, not a study)
Lens: transformation. Cost: F, under 1 GPU-h. Status: DRAFT; the user has rejected replication pages, so
this card survives only as an explanation of a public result, with our own finding as the headline.
**Facts.** "Sparks of Weight-Space Programming" (MIT CSAIL, 2026-11-02) reports an edit of 131,072 weights
in layers 15 and 19 of Qwen3.5-0.8B: MATH-500 7.9 to 48.1 percent, and answers hitting the token limit
93.0 to 33.1 percent. Our instruments study found the released change is one direction per block (its top
singular vector) and that verbatim loops of 100+ characters fall from 169 to 11.
**Claim.** The edit's effect is termination: the residual-stream change it induces loads onto end-of-answer
tokens in the lens readout, and clamping only those coordinates removes most of the accuracy gain.
**Refuted if** the change loads on task content (numbers, operations) and clamping termination coordinates
leaves the gain.
**Demo.** One sentence and one bar chart: what 131,072 weights did.
**Edge.** The replication and the loop detector already exist. Low wow for the public; high for the field.

## N25 — When the workspace band forms, with the massive-activation control
Lens: geometry, time. Cost: B, 3–5 GPU-h. Status: DRAFT. From `01_JLENS.md` Seeds 1 and 6. Supersedes N8.
**Claim.** The band signatures (block structure of the lens vectors across layers, readout kurtosis, lens
vector gain through MLPs) appear between two specific checkpoints of OLMo-3 7B, after the massive-activation
direction forms, and survive projecting that direction out; the step-0 lens shows none of it.
**Refuted if** the step-0 lens shows the structure within split-half noise (architectural, not learned); or
the signatures move by more than split-half noise when the top massive-activation dimensions are projected
out (the band is the sink); or everything changes smoothly with no interval above noise.
**First measurement.** Lenses at step 0, one early, one middle, the final point, plus a split-half refit;
the three statistics with and without the outlier projection.
**Demo.** One curve over the slider.
**Edge.** The 16 points; a third party released layer-20-only lenses at 8 steps for geometry, not for timing
with controls. The authors list the timing as open (§9.1).

## N26 — Sampler–trainer precision mismatch, localized to channels (from `04_PRACTICAL.md` S1)
Lens: geometry, practical (RL systems). Cost: F, 3–4 GPU-h. Status: DRAFT.
**Claim.** The residual mismatch between the rollout sampler and the trainer in RL post-training sits in the
few massive-activation channels and the sink positions; keeping those columns in fp32 removes it as well as
an fp32 output layer does (ScaleRL: fitted asymptote 0.52 to 0.61 with fp32 logits).
**Refuted if** the mismatch is spread across channels, so that the few-column rule recovers less than half of
the fp32-logit gain.
**Edge.** Low: anyone can run it. Kept for the record because the number is one RL engineers act on.

## N27 — Job births lead benchmark rises, and predict them on held-out tasks (from `04_PRACTICAL.md` P1)
Lens: time, representation, practical (pretraining). Cost: F, 6–10 GPU-h. Status: DRAFT.
**Claim.** For a task whose content matches a set of dated jobs, the cumulative count of matched births leads
the rise of the task's log-likelihood metric across checkpoints, and adds predictive power over validation
loss on held-out tasks. Births date encoding; readout lags; so births should lead behaviour.
**Refuted if** the median lead is zero or negative; or matched births add under 0.05 R² over validation loss
on held-out tasks; or births of mismatched jobs predict as well (births then only measure progress).
**First measurement.** 15 tasks (DataDecide-style continuous metrics) at the 16 banked points plus about 24
denser stage-1 checkpoints around each rise; a frozen rule matching tasks to jobs by content; baselines:
validation loss threshold, probes, function-vector similarity.
**Demo.** One chart per task: births climbing, then the benchmark rising. A prediction before the measurement.
**Edge.** 1,238 dated births on a 7B model. Nobody else has them.

## N28 — Forgetting predicted from weights alone, by change on old, much-read poles (from `04_PRACTICAL.md` R2)
Lens: transformation, practical (post-training). Cost: F plus evaluations, about 15 GPU-h. Status: DRAFT.
Extends N6 from a rule to a predictor.
**Claim.** A weights-only score, the change of each pole's output column times its mean activation, weighted
by the pole's reader count or age, predicts the held-out drop (MMLU log-likelihood, wikitext perplexity,
IFEval) across 8 to 11 released OLMo-3 post-training stages better than L2 or Fisher-weighted weight change
(R² 0.34–0.58 in the literature) and approaches the forward-KL predictor that needs rollouts (R² 0.71 at 3B).
**Refuted if** leave-one-out R² of the score is not above L2 weight change, or is below 0.5.
**Demo.** One scatter: predicted against measured forgetting, one point per released stage.
**Edge.** Birth dates and reader counts for the 600 poles; the released chain.

## N29 — Exposure before birth: how many times must the model see a concept before a neuron takes it?
Lens: time, practical (data). Cost: data-order index (no GPU) plus the verified births. Status: DRAFT.
**Claim.** Across verified births, the number of training occurrences of a job's defining tokens seen before
the birth point varies far less across jobs than the birth step does; a job is born after a roughly fixed
exposure, not at a fixed time.
**Refuted if** the spread of log exposure at birth is as wide as the spread of log birth step, or exposure
does not predict birth step beyond token frequency alone.
**First measurement.** With the reconstructed batch order (`00b_DATA_ORDER.md`), count each job's tokens in
every batch up to each checkpoint (one pass over the 16.8 GB per 1,000 steps, on an instance, no GPU);
relate cumulative exposure to birth for the births that pass the random-group test.
**Demo.** One scatter: exposure at birth against birth step, one point per job.
**Edge.** The exact data order plus dated births; nobody has both.

---

## What is not on the list, and why
- Replication pages of published edits. Rejected by the user; the headline must be ours.
- More steering atlases and persona vectors. Doses outside the natural range.
- Anything whose decisive readout is a judge with second-reader agreement under 0.2 and no mechanical check.
