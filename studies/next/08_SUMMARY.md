# Summary — what the brainstorm found and what to run

Fable, 2026-11-05 02:00 CST. One page. Everything below has a file behind it in this folder.

## The answer

Run **what RL wrote** first: transplant the k most-changed MLP neurons from OLMo-3 RL-Zero-Math into the
base model and score MATH-500 against a DARE-rescaled random subset, an energy-matched random set and an
attention-only arm. It is the one study on the list with a benchmark before/after, matched controls in the
same chart, and no dependence on the birth dates the critic weakened. About 25.5–37.7 GPU-hours against the registered 35-hour ceiling (corrected 2026-11-05 from the spec). The
pre-registration is `07_PICKS/N13_SPEC.md`.

Run **the three stage-2 mixes** second: AI2 mid-trained OLMo-3 7B from one released checkpoint on mixes
holding 0, 40 and 65 percent code plus math. Six harvests on our fixed corpus, no judge, and the pre-stated
ordering of per-neuron shifts is the test. A 7B data experiment nobody has read at the neuron level. About
8–11 GPU-hours. Assets verified in `07_PICKS/M1_ASSETS.md`.

Then the biography program on the births that survive the random-group test, the arithmetic-slot study of
computation across training, and the edit-by-age rule, in that order.

## What we learned about our own record

1. **The already-clean birth dates are mostly set by cluster size and rank.** The critic found it; an
   independent recomputation confirmed the part that matters. The closed claim "the main job is the oldest"
   (b 0.476 against a registered reference of 0.449) lies below a structured control that keeps how the main
   cluster is chosen (0.530 [0.497, 0.563]). It needs retagging by its owner, and the walkthrough's births
   picture states that number. Job-specific timing is real for 150 of the 1,238 late-born jobs (chance 28).
   Files: `06_CRITIC.md` Part 0, `06b_VERIFY_BIRTHS.md`.
2. **We hold the exact training order of a 7B model.** The stage-1 batches of OLMo-3 7B are a seeded shuffle
   computable from public files; the dataset fingerprint recomputed from 1,016 public token files equals the
   trainer state's; two real batches were rebuilt and decoded on the laptop. The GPU loss check against the
   public training log matches to 0.0055 nats once pad positions are excluded, an exploratory reading; the frozen all-position rule says not confirmed, and one more registered check (about 0.3 GPU-h) closes it (`pilot0/GPU_CHECK.md`, `pilot0/PAD_ANALYSIS.md`). The critic showed the data order
   does not give "the documents that made a job" at 7B (an interval's documents are exchangeable under a
   uniform shuffle), but it does give exposure counts per concept (N29) and skipped-step forensics (M3).
3. **The workspace paper's causal results are strong on Claude and weak on open models**, and its one clear
   computation example (a value in an intermediate's slot, transformed by a slot-dependent function) can be
   tested on OLMo-3 because its tokenizer keeps numbers up to 999 as single tokens (`01_JLENS.md`).
4. **A one-author program has run the growth questions at small scale** and found a concept's direction in
   use before any neuron is recruited for it (`04_WDD.md`). Our N2 now asks "does the concept exist before its
   neuron", with two measures per checkpoint.
5. **Nobody has published neuron-level job dating, routes at birth or transplants on a 7B model across
   checkpoints** (`04_RECENT.md`). The Goodfire companion paper is not out.

## Decisions for the user

- Retag the D1 birth-order claim, or ask for a second independent recount first.
- Approve the N13 run (25.5–37.7 GPU-h against the ceiling 35, with the park order; corrected 2026-11-05) and the M1 run (8–11 GPU-h). For N13, say which task the report's Table 3 MATH 40.0 is (Minerva MATH or MATH-500), or let an agent settle it from the OLMo-Eval task list.
- The robot program waits behind a 4.6 GPU-h grounding gate (does MolmoAct2-LIBERO follow the words at all);
  say if that gate should run.
- Approve the second data-order check (about 0.3 GPU-h, pad-excluded rule frozen first) if the data order is to be
  cited anywhere.

## The files

`README.md` process · `00_ASSETS.md`, `00b_DATA_ORDER.md` what we have · `01_JLENS.md` the workspace paper ·
`02_FRAMES.md` the lenses · `03_IDEAS.md` 29 cards · `04_LITERATURE.md`, `04_PRACTICAL.md`, `04_RECENT.md`,
`04_WDD.md` the audits · `05_SCORING.md` scores, revisions, picks · `06_CRITIC.md`, `06b_VERIFY_BIRTHS.md`
the attack and its check · `07_PICKS/` pre-registrations and asset checks · `pilot0/` the data-order gate ·
`JOURNAL.md` every event with its time.

## Pending at the time of writing
- `pilot0/GPU_CHECK.md`: done, ≤1.21 GPU-h, both instances terminated. By the rule frozen before the run (every position, pads included) the check is NOT CONFIRMED, because the registered quantity was wrong: the training log excluded pad positions. The exploratory pad-excluded reading matches the log to 0.0055 nats (0.22 per-step sd), and the neighbour batch sits 3.2 sd away, so there is no off-by-one. A second check at two new steps with the pad-excluded rule frozen in advance (about 0.3 GPU-h) would close it.
- `07_PICKS/N13_SPEC.md`: drafted. Before freeze the user settles one fact: whether the Olmo 3 report's Table 3 MATH 40.0 is Minerva MATH or MATH-500.
- 2026-11-05 15:50Z: the D1 birth-order retag is done ("main oldest" not supported: b 0.476 below the random-group 0.530 [0.495, 0.568]). `../already-clean/results/RESULTS.md` (Retag), `../already-clean/JOURNAL.md` 2026-11-05T15:50Z, F4 redrawn (`../already-clean/results/figures/`), walkthrough corrected (`../already-clean/walkthrough/index.html` `#birthorder`, `../already-clean/walkthrough/births/`); report `10_RETAG.md`.
