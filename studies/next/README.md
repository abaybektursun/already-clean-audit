# NEXT — brainstorm for the next high-leverage study

- The map: `00_INDEX.md` (every study, its files, its cost, the method rules).
- The state: `HANDOFF.md` (what is closed, what is open, how to resume).

Started 2026-11-05. Owner: the Fable session (strategy, synthesis, review). Labor: Opus subagents, two at a time.
Status 2026-11-08: **closed**. Six studies closed (M1, M2, N12 with N2V, N1, N13, N6), two pilots decided, nothing running; the abstract is `20_ABSTRACT.md`, the reflection `21_REFLECTION.md`. The line below is the 2026-11-05 status, kept as written.
Status: **in progress**, 2026-11-05 01:40 CST. Stages 0–6 written; verification of the critic's birth finding, the N13 pre-registration and the M1 asset check are running. One GPU job has run: the pilot-0 loss check on a reconstructed training batch (`pilot0/`).

## What this folder is for

One question: what study, run with what we already have, gives a result that a reader sees and believes in
six seconds and still believes after thirty? The essay "Neural Networks Are Already Clean" is the starting
stance: a network's own neurons are its legible parts, new computation is a route between existing parts,
and you can watch the parts grow over training. We take that stance seriously and test it where nobody has:
on a 7B model with its training checkpoints in hand.

## How the brainstorm is run (the process, stated once)

The process is the one strong research teams use: facts first, then frames, then many ideas, then scoring,
then an adversarial read, then two or three pre-registrations. Each stage is a file. Nothing is scored before
the facts are written down; nothing is pre-registered before the critic has read it.

| stage | file | who writes it | rule |
|---|---|---|---|
| 0 | `00_ASSETS.md`, `00b_DATA_ORDER.md` | inventory agent; data-order agent | every number carries its artifact path or URL |
| 1 | `01_JLENS.md` | reader agent | the Jacobian lens paper and its follow-ups, read in full |
| 2 | `02_FRAMES.md` | Fable | the abstract lenses: time, computation against language, representation, transformation, geometry; what "wow" means as a test |
| 3 | `03_IDEAS.md` | Fable, then agents extend | one card per idea: claim, what refutes it, cheapest first measurement, the demo, cost, edge |
| 4 | `04_LITERATURE.md`, `04_PRACTICAL.md`, `04_RECENT.md`, `04_WDD.md` | four agents | already-done audit; what capability engineers optimize; July–November 2026 work; the one competitor program read in full |
| 5 | `05_SCORING.md` | Fable | leverage = wow × novelty × chance of a clear answer, divided by cost |
| 6 | `06_CRITIC.md`, `06b_VERIFY_BIRTHS.md` | a different model; an independent verifier | adversarial read of the picks; independent recomputation of its most severe finding |
| 7 | `07_PICKS/` | spec agents | pre-registration drafts (N13 first), asset checks (M1) |
| — | `pilot0/` | engineer agents | the data-order gate: reconstructed batches, the GPU loss check |
| — | `JOURNAL.md` | everyone | dated log, one line per event |

## House rules for every file here

Plain technical English. Short sentences. No analogies, no metaphors. Use the real terms: matrix, rank,
subspace, Jacobian, neuron, column. Never the words "null", "void", "ledger" (say chance level, cancelled,
record). A number without an artifact or URL does not exist. Every idea names the observation that refutes
it. A result that is elegant gets more scrutiny, not less.

## Vocabulary used throughout

- **Pole**: one sign of one MLP neuron. A neuron has a positive pole and a negative pole.
- **Job**: one group of a pole's top-50 contexts, as sorted by the clusterer in the already-clean study.
- **Birth**: as coded in the already-clean study, the first point t at which the median pole-direction activation over
  the job's final-point exemplar positions is at least the pole's own 99.9th percentile at t and at t+1. The critic
  (`06_CRITIC.md` Part 0) found this date is mostly set by the group's size and rank; see `06b_VERIFY_BIRTHS.md`.
- **Reader**: any later weight that multiplies a neuron's output; readers are what make a representation matter.
- **Route**: the chain of neurons and heads through which a pole's activation reaches the loss.
