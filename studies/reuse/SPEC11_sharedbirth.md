# SPEC11 — at the newer cluster's birth, does it depend on the older cluster's drivers more than on a sibling cluster's?

Status: drafted 2026-11-03 05:25 CST (coordinator); pre-freeze amendments of 06:10 CST below (§2, §3, §4, §8, from the builder's notes `results/review11/BUILD_NOTES_sharedbirth.md`); frozen at tag `reuse-freeze-11` before any forward. Parents: `SPEC10_sharedinputs.md` and its read (at T_15 the dependence on the older cluster's drivers is not larger, by the 0.05 line, than on the other clusters' drivers, and follows the shared drivers), `SPEC4_birth.md` (R1c: I′ 0.207 [0.156, 0.291] at the birth checkpoints against null′ 0.012; the birth-load banks `results/birth/1/t<nn>/falls/`, the known number). Model `allenai/Olmo-3-1025-7B` at the birth checkpoints T_7 to T_13.

## 0. What is known

The restated headline carries a birth-checkpoint value, 0.21, read only against the older cluster's drivers; SPEC10's comparison with the pole's other clusters was made at T_15. Other clusters born after the newer cluster cannot be read at its birth; siblings and the older cluster's own older cluster can.

## 1. Question, decision, hypotheses

**Question.** At the newer cluster's birth checkpoint, is its dependence on the older cluster's drivers larger than on the drivers of another cluster of the same pole that already exists?

**Decision.** If larger by the line, the birth-checkpoint sentence keeps the older cluster as its object ("at the birth, the dependence is on the older cluster's drivers in particular"); if not, the birth value joins the T_15 reading as a dependence on shared drivers, and the old–new pairing is dropped at the birth too.

| hypothesis | statement | refuted when |
|---|---|---|
| H_old-at-birth | At t_b(J) the dependence on O*'s drivers exceeds the dependence on an existing other cluster's drivers | the paired difference I_O − I_K at t_b has its interval below 0.05 (Y1f) |
| H_shared-at-birth | At t_b(J) the two are the same | the interval above 0.05 (Y1f) |
| Instrument artifact | | any gate of §5 fails |

## 2. The measurement

SPEC10's instrument at the newer cluster's birth checkpoint t_b(J) (SPEC4's loads T_7 to T_13; the sets ranked from captures at that load with B_t recomputed there; free scale; the 1% exclusion): own′ (D′(J) on J), cross_O (D′(O*) on J), cross_K (D′(K) on J) for each K of the case that exists at t_b(J), random′; I_O, I_K, d = I_O − I_K as SPEC10. A K exists at t_b(J) when every case whose J it is has birth at or before t_b(J), and a K that is only some case's O* exists by construction (*amended 06:10 CST before freeze: a K that is one case's O* but another case's J born later is excluded, as §5's gate requires; one K at T_11 falls under this*); K born later are excluded and banked. Reported: K's own fall at t_b(J) (K's positive control there), the reverse arm, the sibling split, and I_O against SPEC4's banked R1c value for the same cases.

## 3. Design and data

Seven loads (T_7 to T_13; the one SPEC10 case born at T_6 has no K existing there). Cases: the SPEC10 cases with at least one K existing at t_b(J): 47 primary (discovery) and 36 fresh (confirmation, decides), 101 K reads, 34 K dropped as born later; births 3, 4, 4, 10, 8, 24, 30 over T_7 to T_13 (*amended 06:10 CST before freeze from 48/36/102*). About 84 × (12 × (3 + K)) ablated readouts with clean readouts, plus the ranking captures, spread over seven loads; about 0.7 GPU-h, mostly downloads.

## 4. Known numbers

The primary cases' own′ and cross_O at their birth load against SPEC4's banked birth-load falls (`results/birth/1/t<nn>/falls/<case>.npz`, the same instrument on another card; SPEC4's cross-card known number read 0 to 0.0031): median absolute difference over the 47 primary cases pooled across loads at most 0.01 (T_7 has no primary case; *amended 06:10 CST before freeze*); failing input the next case's falls. The parent's known number against the harvest and the capture identity per load as SPEC4.

## 5. Gates

As SPEC10 (known numbers; capture identity per load; exclusion; positive control per set; cancel per set at the case's birth load; K validity, with the existence condition of §2 asserted from the items' births; failing input: a K born after t_b(J)). A failed gate stamps every rule "not read".

## 6. Frozen rules

- **Y1 (primary) and Y1f (fresh, decides).** The median over live cases of d = I_O − I_K at t_b(J), bootstrap over neurons (2,000 draws, seed 0): interval above 0.05 → H_shared-at-birth refuted; below 0.05 → H_old-at-birth refuted; containing → not separated. The line is SPEC10's.
- **Reported, no rule:** I_O and I_K with intervals; the share with I_O > I_K; K's own fall; the reverse arm; the sibling split; I_O against SPEC4's R1c on the same cases; the pooled d.

## 7. Point predictions, before any forward

| quantity | predicted | refuted if |
|---|---|---|
| I_O primary; fresh (SPEC4 read 0.207 and 0.257 on all cases) | 0.21 [0.15, 0.28]; 0.25 [0.17, 0.34] | outside [0.05, 0.50] |
| I_K primary; fresh | 0.19 [0.12, 0.28]; 0.22 [0.13, 0.33] | outside [0.03, 0.50] |
| d primary (Y1); fresh (Y1f) | 0.02 [−0.05, 0.09]; 0.02 [−0.06, 0.10] | outside [−0.15, 0.25] |
| K's own fall at t_b(J) | 0.45 [0.35, 0.55] | under 0.25 |
| GPU hours (seven loads) | 0.7 [0.4, 1.2] | over 2.0 |

The author expects shared dependence at the birth too (d near 0), from SPEC10.

## 8. Threats

| threat | effect | mitigation |
|---|---|---|
| Most K are siblings sharing O*, so D′(K) overlaps D′(O*) | cross_K partly replays cross_O | the overlap of D′(K) with D′(O*) and with D′(J) reported per K; the non-sibling K reported separately where they exist |
| K's activation at t_b(J) may be small (K born recently) | K's own fall low, its ranking noisy | K's own fall reported; §6's rule reads every existing K, and the reading restricted to K with own fall at least 0.25 is reported beside it (*amended 06:10 CST before freeze: §6 governs*) |
| Few cases at the early loads | the per-load numbers unreadable | the rule pools over loads; births by load reported |

## 9. Runbook

Worker (Opus, own worktree `../interp_research-sharedbirth`, branch `research/sharedbirth`): extend `jobs/reuse_sharedinputs.py` with `--t` and a per-load case subset (the cases born at that load) and the K existence rule; `steps/reuse_sharedbirth.json` (seven steps T_7 to T_13); `scripts/verdict_sharedbirth.py` (`--failing-input known_next_case | capture_wrong_layer | positive_random | cancel_swapped | k_born_later`, Y1 and Y1f, `GRID11.md`); the dry run `jobs/reuse_dryrun_sharedbirth.py`. Coordinator: freeze (`reuse-freeze-11`), the dry run, launch (one pipeline, `g6e.2xlarge,g6e.4xlarge`, `--no-weights`, fresh prefix `results/reuse/sharedbirth/1`, budget 120 min), samples, §7 restated, verdict once, failing inputs, recount (seed 1), a different model's read, figures, the register, the SPEC4 file's queue table. Cost about 0.7 GPU-h, a further recorded deviation; API $0.

## 10. Grid, filled once

| rule | reading | outcome |
|---|---|---|
| Y1 | | |
| Y1f | | |
| gates | | |
| GPU hours | | |
