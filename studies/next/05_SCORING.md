# Scoring — rubric, scores, programs, picks

Fable, 2026-11-05 01:05 CST. Scores are PROVISIONAL until `04_RECENT.md` and `04_WDD.md` land; the two cards
they can move are marked. Inputs: `00_ASSETS.md`, `00b_DATA_ORDER.md`, `01_JLENS.md`, `04_LITERATURE.md`,
`04_PRACTICAL.md`.

## Rubric (fixed before any score was entered)

Each card gets five integers.

| axis | 1 | 3 | 5 |
|---|---|---|---|
| **Wow** (six-second test) | a number only interpretability people read | a picture an ML researcher understands without text | a visible before/after or a benchmark number an engineer already tracks, with the control in the same picture |
| **Novelty** (after the audit) | DONE elsewhere; we would replicate | PARTLY; the open part is ours | OPEN under the field's own terms |
| **Decisiveness** | the result could be read either way | one reading is clear, the other needs a follow-up | both outcomes are clear and publishable; the refuting observation has room to fire |
| **Edge** | anyone with a GPU could do it this week | needs one of our banks or harnesses | needs two or more of our assets together |
| **Cost** (inverted) | over 30 GPU-h or any 7B training | 5–30 GPU-h | under 5 GPU-h, weights or forward passes only |

Leverage = product of the five. A 1 on any axis sinks the card.

## Scores

| card | wow | novelty | decisive | edge | cost | leverage | note |
|---|---|---|---|---|---|---|---|
| N20 census of all 704,512 poles | 4 | 4 | 4 | 5 | 5 | **1600** | CPU only; the foundation picture for everything else |
| N27 births lead benchmark rises | 4 | 4 | 4 | 5 | 4 | **1280** | predictive; uses the rarest asset; may move after `04_RECENT.md` |
| N6 edit the young | 3 | 5 | 4 | 4 | 5 | **1200** | OPEN; cheap; a rule editors can use |
| N11 explain at birth | 2 | 5 | 4 | 5 | 5 | **1000** | OPEN; judge calls only; a tool recommendation |
| N3 route at birth | 4 | 3 | 4 | 5 | 4 | **960** | the circuit-formation picture under the slider |
| N4 documents that made the job | 4 | 4 | 4 | 5 | 3 | **960** | gated by pilot 0 (data order); the most legible result for a general reader |
| N24 what the famous edit wrote | 3 | 4 | 4 | 4 | 5 | 960 | explanation of a public result; the user may veto as replication |
| N22 the model gets a variable | 4 | 4 | 3 | 4 | 4 | **768** | the computation flagship; open-model failures are a risk |
| N25 workspace band timing, sink control | 3 | 4 | 4 | 4 | 4 | 768 | instrument check for N22 |
| N28 forgetting predicted from weights | 4 | 4 | 4 | 4 | 3 | **768** | RL engineers' number; extends N6 |
| N1 input first, readers later | 3 | 4 | 5 | 4 | 4 | 960 | after `04_WDD.md`: their classes are defined by the output vector, so their output-first reading is partly by construction; ours must use both definitions |
| N2 silent or used | 4 | 4 | 4 | 5 | 4 | **1280** | revised after `04_WDD.md`: the function may precede the neuron; two measures per point (zero the pole; project its direction out) |
| N5 newborn skill moves with k neurons | 5 | 3 | 4 | 3 | 4 | 720 | prior lowered: at 410M every transplant closure was negative (`04_WDD.md`); needs heads and readout arms |
| N12 gradient starvation | 3 | 3 | 4 | 5 | 4 | 720 | shares passes with N2 |
| N7 state against content jobs | 4 | 3 | 3 | 4 | 4 | 576 | Ge et al. did features on Pythia; ours is single-neuron, by invariance |
| N13 what RL wrote | 4 | 4 | 4 | 3 | 3 | 576 | contested literature settles with the DARE control |
| N19 what RL kills | 4 | 4 | 3 | 3 | 4 | 576 | depends on N13's ranking |
| N21 two architectures, 4 points | 4 | 4 | 4 | 3 | 3 | 576 | needs harvests of Olmo-Hybrid-7B |
| N23 action workspace in the robot | 5 | 4 | 2 | 4 | 3 | 480 | decisiveness low until the grounding check passes |
| N17 repair the robot's forgetting | 4 | 3 | 3 | 4 | 3 | 432 | prior lowered by sinkpatch |
| N9 rewire one robot instruction | 5 | 3 | 2 | 3 | 3 | 270 | crowded; LIBERO policies ignore instructions |
| N26 precision mismatch in channels | 4 | 4 | 4 | 1 | 4 | 256 | no edge |
| N16 neuron-level merging | 4 | 3 | 3 | 2 | 3 | 216 | |
| N18 curriculum from births | 4 | 3 | 2 | 4 | 2 | 192 | depends on N4 |
| N15 double-negative over training | 2 | 3 | 3 | 4 | 5 | 360 | not a headline |
| N8, N10, N14 | — | — | — | — | — | — | set aside |

## Reading the table

Four cards above 1,000 are cheap and ours alone, but each is a picture or a rule, not a before/after. The
before/after cards (N5, N9, N23) carry the lowest decisiveness because published small-scale results and the
grounding gap in robot policies give them many uninformative ways to fail. The way to the wow the user wants
is to stack the cheap, decisive cards into one object, and to earn the before/after on top of it.

## Programs, scored as bundles

**Program A — The complete biography of a neuron in a 7B model.** N20 (census), N27 (births lead behaviour),
N4 (documents), N1, N2, N3, N12 (the mechanism at birth). One object: the slider, with three new panels under
it: the weights (input rows against output column across the birth), the route (upstream contributors with
their own ages), and the documents (the training instances that pushed the input rows). One prediction: births
lead benchmark rises on held-out tasks. One claim nobody can make without the data order: these documents did
it. Cost: pilot 0 about 0.2 GPU-h; census 0 GPU-h; N27 6–10; N4 1 GPU-h per job; N1–N3 and N12 about 8
GPU-h for 100 births. About 25 GPU-h in all.
Six-second test: the slider already passes it. Thirty-second test: the documents and the prediction.

**Program B — Post-training in the neuron basis.** N13 (what RL wrote, with the DARE control), N28
(forgetting predicted from weights), N6, N19. The engineers' numbers. About 40 GPU-h. Shares the harvest
instrument and the birth dates with Program A.

**Program C — The robot.** N23 (action lens), N9, N17. Held behind one cheap gate: a blank-instruction and
swapped-instruction baseline on MolmoAct2-LIBERO per suite (about 1 GPU-h). If the policy follows the scene
and not the words, there is nothing to rewire and the program is set aside with that number recorded.

**Program D — The model gets a variable.** N22 with N25 as its instrument check, N7, N11. The
computation-against-language flagship. About 10 GPU-h.

## Picks (provisional)

1. **Program A** first. It uses every asset only we have, every card is decisive either way, and the demo is
   the object the user already loves, made complete.
2. **Program B** second, in parallel where GPU time allows, because it answers a question RL engineers are
   arguing about now and reuses Program A's instruments.
3. **Program D's N22** as the one computation study, after N25 says the lens is not reading the sink.
4. **Program C** waits for its gate.

Order of the first runs: pilot 0 (data-order loss check) → census → N27 → N4 on the lithium job → N1–N3 on
100 births → N13.

---

## Revision after the critic (`06_CRITIC.md`), 2026-11-05 01:35 CST

**Rubric change adopted.** Only a card with a before/after, a prediction made before the measurement, or an
engineer's number can be a pick. Descriptive cards (N20, N25, N11) are instruments, costed inside the pick
they serve.

**The finding that changes the plan.** The critic recomputed the already-clean birth rule from the banked
exemplar activations and reproduced every D1 birth (0 mismatches over 2,454 clusters). Then it regrouped
each pole's top-50 positions at random into groups of the real sizes and applied the same rule: the birth
histogram of random groups matches the real one (total-variation distance 0.040), only 148 of the 1,238
late-born jobs are born later than 95 percent of such random groups, and the closed D1 number "main job is
oldest" (b 0.476) lies below the random-group reference 0.530 [0.491, 0.568]. If this holds under an
independent recomputation, the birth date of a job is mostly set by its cluster's size and rank, the D1
birth-order claim must be retagged by its owner, and Program A's unit shrinks to the births that pass the
random-group test. Verification is launched; nothing in Program A runs before it returns.

**Picks, version 2.**

1. **N13, what RL wrote** (Program B), first. The only pick with a benchmark before/after and matched random
   controls that does not rest on the birth dates. Changes adopted: a DARE-rescaled random set, a
   tail-energy-matched random set, a bf16 floor on "changed", an attention-and-norm arm, the complement arm,
   a MATH-500 noise gate before any gain is read, absolute accuracies, jobs from harvests at `main` and each
   stage. Stage: RL-Zero-Math from the base, or Think-DPO to Think RL, whichever clears the noise gate.
   About 20–25 GPU-h. Pre-registration drafting launched (`07_PICKS/N13_SPEC.md`).
2. **M1, the three stage-2 mixes** (critic's idea, adopted). AI2 ran three mid-training mixes from one
   checkpoint: a 7B data experiment nobody has read at the neuron level. Five harvests on the fixed corpus,
   an identity pair, no judge: per pole, the change in the share of top-50 positions in code and math
   documents, mix against mix. A before/after with its control in the chart. Asset check done (`07_PICKS/M1_ASSETS.md`): the shared
   start `stage1-step476838` is released; the mixes contain 0, 40 and 65 percent code plus math; the test is
   the pre-stated ordering of neuron shifts 0 < 40 < 65 with gen-mc-only against math-code-reasoning-web as
   the lead contrast; 8.4–11.4 GPU-h; two loader fixes before any run.
3. **M2, name the carrier before the birth** (critic's idea, adopted as the surviving form of N12 and N27):
   at the point before each verified birth, rank all poles of the layer by a frozen score and test whether
   the eventual carrier is in the top 10 (chance about 0.0009), against the pole's own current percentile
   as the baseline that beat the competitor's forecast. Runs only on births that pass the random-group test.
4. **N22, the model gets a variable**, with the critic's gates (accuracy gate per point, lens-free
   interchange positive control, slots compared at matched depth; the "before correct answering" clause
   dropped). 5–8 GPU-h.
5. **N6, edit the young**, with energy-, density- and norm-matched pairs and equal-norm edits. 4–8 GPU-h.
   Program A's mechanism cards (N1, N2, N3) follow the verification and the instrument check (random-group
   reference at 0 GPU-h; identity-pair harvest about 3.5 GPU-h), on the surviving births only.

**Set aside by the critic, accepted:** N4 at 7B (an interval's documents are exchangeable under a uniform
shuffle, and the per-document gradient cosine is forced by the job's own inputs); N28 (ten nested stages give
no power); N27 as designed (births also measure progress; a 0 GPU-h version on the public evaluation curves
may run later on verified births). N19 behind its gate. The data order keeps two uses: M3 (which skipped
updates were caused by their batch) and exposure counts per concept (`03_IDEAS.md` N29).

**Cost corrections accepted:** the robot gate is about 4.6 GPU-h (27.6 s per episode), not about 1.

## Verification of the birth finding (`06b_VERIFY_BIRTHS.md`), 2026-11-05 01:50 CST

Holds in part, and the part that matters for the picks holds.
- The retag is supported: under random regrouping at the real cluster sizes, b is 0.530 [0.497, 0.563]; the
  real 0.476 lies below all 300 draws (one-sided p 0.0067). The registered reference (0.414, 97.5th 0.449)
  reproduces exactly but treats the main cluster as arbitrary; it is the main cluster's size that sets most of
  the gap (size fixed alone gives 0.512).
- Job-specific timing is real and larger than the critic counted: 150 of 1,238 late-born jobs beat 95 percent
  of random same-size groups, against a selection-matched chance level of 28. Program A's unit is those jobs.
- The histogram is close to, not equal to, the random one (TV 0.0406 against a noise ceiling of 0.028, with
  the real excess at the last two points). So counts of births per point measure mostly size and rank, and the
  census (N20) stays an instrument.

Consequences for the record, to be decided by the owner of the already-clean study (the user):
1. Retag the D1 claim "the main job is the oldest" (b 0.476 against the registered 0.449): contradicted by a
   structured control that keeps how main is chosen. The walkthrough's births picture states this number.
2. Add the random-group reference to the birth instrument and to the walkthrough's description of a birth.
3. Any later use of "dated births" names the 150 that pass, with their one-sided p, not the 1,238.
