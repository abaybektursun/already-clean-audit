# 04 — wdd-descendants, read in full for the growth cards

Reader agent, 2026-11-05. Source: shallow clone of https://github.com/sunmoonron/wdd-descendants at commit
`8800361655608655ff8d9b6ef89451251508239b` (2026-10-31), into
`/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/wdd_in/wdd-descendants/`.
No code from the repository was run. Numbers marked "recount" were recomputed by me from the repository's own
result JSON with `python3 -I`; every other number is quoted from the page or script named beside it.

Read levels. Read line by line by me: README sections TL;DR, repository, replication, findings, Phase 2, Phase 3,
7B workspace, scope; `docs/PAPER.md` sections 1 to 8; atlas pages 01, 24, 25, 33, 36, 37, 38, 39, 40, 41, 48, 49,
54, 55, 56, 62, 63, 64 (Established block), 69 (Established block); scripts `e565`, `e577`, `e578`, `e589`, `e684`,
the headers of `e588`, `e591`, `e628`, `e632`, `e666`, `e674_download`, `e701`, and `s101_common.py`, `ex_common.py`
(definitions). Read line by line by two Opus reader agents, whose extracts are merged in sections 2 and 4 and
marked "(agent read)": the other 60 atlas pages, `docs/NEXT.md` and `docs/ERRATA.md`. The README's 92 KB
area table and its 201 KB "Training's hand" narrative restate the atlas pages; I read them by search for the
experiments named here, not line by line.

---

## 1. The program in ten lines

1. **What it is.** One author's program (README byline Ronish Bhatt, ORCID 0009-0000-8835-5380; GitHub user
   sunmoonron, profile name BCR), 744 experiments (`docs/atlas/index.tsv`, ids e00 to e734 with variants), run
   2026-09-19 to 2026-10-28, MIT code, CC BY 4.0 documents, no paper (a draft `docs/PAPER.md`; a note on the
   "brake" was published and withdrawn 2026-10-20). The record says experiments were pre-registered in
   `results/*_prereg.json` and several designs came from proposals by "Gemini 4" (`docs/REVIEW_POSITIVE_CLAIMS.md` §15 to §19).
2. **Instrument (WDD).** A residual state after block B (centred, unit length, attention-sink positions dropped)
   is decomposed by orthogonal matching pursuit with K = 16 atoms over the model's own unit-norm MLP output
   vectors of blocks 0 to B (`scripts/s101_common.py`, `scripts/ex_common.py`).
3. **Row.** One MLP neuron's output vector (down-projection column, the program's "write row" or "write
   column"). The neuron's input weight is its "read row". **Floor**: a per-state extreme-value bound
   calibrated so that a randomly rotated copy of the dictionary sits at ratio 1. **S**: a row's largest
   |projection| / floor over all positions.
4. **Word.** One of the 256 rows most used by the pursuit at a checkpoint (`wordset`, NWORD = 256 in
   `ex_common.py`; "a word is one of the 256 most used rows", `docs/PAPER.md` §4).
5. **Class.** A word's positions where its row's |projection| exceeds the floor, usually fixed at step 16000
   and used backward in time (`e589`, `e632`, `e684` scripts). Its "direction" is the unit mean of the class's
   unit states. A class is therefore defined by the output vector's geometry on residual states, not by what
   the neuron fires on.
6. **Recruitment.** The first checkpoint from which the row is a word at every later checkpoint up to 16000
   (`first_hold` in `scripts/e632_lottery.py`, `scripts/e684_readers.py`; `e591` docstring). Resolution 1,000
   steps, steps 1000 to 16000 only. **Descendant** (phase 1, e01 to e356): the footprint, at later depth, of
   ablating one write.
7. **Models and checkpoints for growth.** Pythia-410m block 12, steps 1000 to 16000 (16 checkpoints), 24 x 512
   Pile evaluation tokens; Pythia-160m block 6, seeds 0, 1, 2 (PolyPythias), steps 0 to 512 log-spaced and 1000
   to 16000; OLMo-1B-0724 block 8, steps 1000 to 16000, 8 x 256 tokens (2,040 positions). Final-model work
   adds GPT-2, SmolLM2, Qwen2.5 0.5B to 14B, Mamba, two ViTs, Ouro-1.4B.
8. **Instruments for growth.** Leave-one-checkpoint-out forecast AUC of which rows become words; rank of the
   eventual row among all rows by class ratio; loss gradient on one row's output vector split by position;
   parameter transplants between steps 8000 and 16000 of Pythia-410m; continued training from a checkpoint on
   a proxy stream (`NeelNanda/pile-10k`, fresh AdamW); downstream "reader" share by an exact split of later
   neurons' pre-activations; a per-checkpoint TopK SAE as the comparison dictionary.
9. **Models of scale.** No developmental run above 1B. 7B and 14B appear only as final models (Qwen2.5-7B,
   Qwen2.5-14B).
10. **Thesis, in the author's words** (`docs/PAPER.md` §6): "The data gives the class and its direction;
   training recruits the nearest row to it, by a small drift the class's own positions drive; thousands of
   writers implement it; the row is replaceable. WDD is the instrument that follows the row."

---

## 2. Every quantitative conclusion

Format: id | model and checkpoints | what was measured | number, with its control | page. "Closure" in the
transplant lines is `(KL_before − KL_after) / KL_before` to the donor model (`scripts/e578_coalition_transplant.py`):
positive moves the recipient toward the donor, negative moves it away. Sections 2.1 to 2.7 are complete for the
growth topics. Section 2.8 gives the other areas one line each with their headline number.

### 2.1 Recruitment: when a row becomes a word, and which row

- e584 | Pythia-410m b12, 16 ckpts | when 5 events occur for 241 eventual words | every class coheres in the states at step 1000; rows recruited at step 7.0 thousand (median); "coalition" coherent 5.0 checkpoints before recruitment for 0.84 | [atlas 38][a38]
- e584 | same | per-checkpoint TopK SAE (4096 latents) against the row | persistent SAE feature for 0.31 of classes; decoder cosine with the row at recruitment 0.11; points at the row 7.0 checkpoints after recruitment for 33 of 241 | [atlas 38][a38]
- e585 | same, 253 classes | the class direction over time and which row speaks it | class direction cosine with its step-16000 direction 0.85 / 0.93 / 0.86 at 4000 / 8000 / final, while the 16000 row speaks it for 0.06 / 0.35 / 0.38; one speaker at the median after a gap of 8 checkpoints; 183 of 187 handoffs through overlaps | [atlas 39][a39]
- e587 | same | rank of the eventual row among 53,248 rows by ratio over the class | two checkpoints before: first for 0.52, top ten 0.84 (by cosine top ten 0.78); five before top ten 0.72; eight before 0.62 | [atlas 39][a39]
- e588 | Pythia-410m and OLMo-1B, 16 ckpts | forecast of recruitment 2 / 4 / 8 checkpoints ahead from row-free k-means classes | class side 0.927 / 0.907 / 0.869 (OLMo 0.932 / 0.901 / 0.848); row's S 0.926 (0.965); norm 0.423 (0.475); precision at 100: 0.05 / 0.10; verdict "below the pre-registered level" | [atlas 39][a39]
- e588, same run, from `results/FINDINGS.log` | the row's own pursuit usage at t, the quantity that defines a word | 0.964 (Pythia-410m) and 0.968 (OLMo-1B), above the class side; precision at 100: usage 0.23 / 0.22 against class side 0.05 / 0.10; the first class-side scoring of the same run ("owned-cluster variant") gave 0.582 and 0.521 | `results/FINDINGS.log`, [atlas 55][a55]
- e629 | Pythia-160m seeds 0, 1, 2, blocks 6 and 9 | same forecast against SAE routes | class side 0.886 (seeds 1, 2: 0.884, 0.881; block 9: 0.923); row's usage 0.923 (0.917, 0.922; block 9 0.953); SAE decoder cosine 0.810; static step-1000 score 0.640; label shuffle 0.501; retrodiction 0.897 | [atlas 55][a55]
- e629 | same | rank of the recruit among 21,504 rows two checkpoints before | first by class ratio for 0.43, top ten 0.79 of 194; SAE decoder puts it top ten for 0.33 (decoder-row cosine 0.10) | [atlas 55][a55]
- e629b | same | forecast rule fitted on seed 0, scored on others | 0.892, 0.896 (seeds 1, 2), 0.937 (block 9) | [atlas 55][a55]
- e632 | Pythia-160m seeds 0, 1, 2, b6, steps 0 to 16000 | "lottery": the recruit's rank at step 0 by cosine with its final class direction | median normalised rank 0.15 to 0.21 (random row 0.51; another class's direction 0.40); top 100 of 21,504 for 0.05 to 0.08; step-0 rank does not set recruitment time (Spearman 0.07) | [atlas 56][a56]
- e632 | same | output-vector rotation from step 0 | cosine with the step-0 direction at recruitment 0.20, all rows 0.19 ("rotate as far as other rows"); at 16000, 0.13 against 0.15 for all rows (S128 correction) | [atlas 56][a56]
- e632, S128 correction | same | is the step-0 alignment real | withdrawn: persistence of the random initial direction alone predicts rank 0.17 for the own-write-subtracted direction, stronger than the observed 0.32; the direct test (`scripts/e632b_*`) "was written and not run because the box was released" | [atlas 56][a56]
- e632 | same | read-weight and write-norm of the recruit against its ten nearest competitors, two checkpoints before | recruit's input-weight norm larger for 0.49, output norm for 0.65 (seeds 1, 2: 0.48 / 0.53, 0.38 / 0.52); chance 0.5 | [atlas 56][a56]
- e666 | same | given the class, which score names the recruit two checkpoints before | median rank 1 by class ratio (top ten 0.77 to 0.84), 1 to 3 by direct write (0.57 to 0.65), 2 to 4 by activation selectivity (0.55 to 0.59), usage 111 to 194; on a class defined by firing instead, selectivity top ten 0.79 to 0.88 and class ratio 0.52 to 0.62 (definition decides the winner) | [atlas 56][a56]
- e666 | same | the recruit's own write as a share of the class states' projection onto its row | 0.18 to 0.21 two checkpoints before, 0.23 to 0.27 at 16000 | [atlas 56][a56]
- e591 | Pythia-410m (205 words), OLMo-1B (159) | class loss drop across recruitment minus token-matched positions | −0.0102 [−0.0292, +0.0116] (OLMo −0.0111 [−0.1104, +0.0512]); "recruitment is not a performance event" | [atlas 41][a41]
- e591 | Pythia-410m | KL from removing the row's direction at its class, k = −2 / +2 | 0.0047 / 0.0109 (random direction 0.0002); median rise ratio 1.87 | [atlas 41][a41]
- e591b | same | same against stronger controls | k = −2: strongest other row 0.0058 (row beats it for 0.34), row-free class direction 0.3861, row at control positions 0.0015; k = +2: 0.0109 / 0.0089 / 0.4160 / 0.0028 | [atlas 41][a41]
- e591c | both | controls matched on token and on loss one checkpoint before | Pythia +0.0210 [−0.0123, +0.0583], OLMo +0.0035 [−0.0332, +0.0530] | [atlas 41][a41]
- e595 | Pythia-410m, OLMo-1B at 8000 and 16000 | row-free clusters per word | 0.15 at 1000, 3.53 at 16000, 5.88 final (K = 512); K = 128 / 2048: 0.57 / 25.46 | [atlas 41][a41]
- e683 | Pythia-160m, three seeds | do new class members arrive next to recent offshoots ("chain") | enrichment 0.91 to 0.92 (chance level 1.0); jumps 0.01 to 0.02 below the random-position control | [atlas 63][a63]
- e672 | Pythia-160m b6, 300 neurons | routes by upstream attribution; offshoots cheaper? order? | 83 of 300 have two routes; offshoots cheaper in 0.45 (by the essay's grouping costlier, p 0.045); core first 0.53 (p 0.33) | [atlas 38][a38]
- e674 | Pythia-160m b3, b9; Pythia-410m b6, b12, b18; OLMo-1B b4, b8, b12 | the larger part of a neuron's top firings becomes top firings first | one-route 0.58, two-route core 0.56 (p 0.0017); offshoots cost more (pooled p 0.0017) | [atlas 38][a38]
- e676 | same nine runs | firings nearest the centre of the neuron's input attributions become top firings first | 0.63 to 0.80 of neurons in all nine runs (p < 0.0001 each); lead survives frequency and position in 8 of 9 | [atlas 38][a38]
- e679 | Pythia-160m vs 410m | does the larger model split a two-route neuron | split 0.34, 0.28, 0.49 against 0.19, 0.17, 0.24 for one-route; no 410m neuron covers more than about a third of a 160m neuron's top firings | [atlas 38][a38]

### 2.2 Input weight against output vector

- e565 | Pythia-410m b12, 119 entrants, step-16000 neuron parts into step 8000 | which part makes the row a word | output vector 0.59 (direction only 0.54, norm only 0.18), input row 0.18, bias 0.03, both 0.67, all three 0.68; no part 0.00 | `results/FINDINGS.log`, `scripts/e565_factorial_transplant.py`
- e522 | Pythia-410m b12 and b6, 4 origins | who moves when a row enters: row or states | row motion 0.64 to 0.78 of the rise at b12 (0.56 to 0.68 at b6), state motion 0.02 to 0.24; every row turns by a similar angle (consecutive cosine 0.82 to 0.88) and grows (norm ratio 1.19 to 1.57) | [atlas 25][a25]
- e524b | same | time course of an entering row at 1,000-step resolution | directed drift of five thousand steps or more (consecutive increment cosine 0.28 to 0.31 against 0.24 to 0.25 matched), then a crossing of two thousand steps (S 1.07 to 1.25) | [atlas 25][a25]
- e648 | Pythia-410m b12, every row | does a row's output vector move away from its own input weight | yes for 0.96, 0.91, 0.86 of rows at 4000, 8000, 12000 (mean cosine −0.119, −0.083, −0.058) | [atlas 25][a25]
- e552, e554 | OLMo-1B b8 | input-side selectivity of the entering neuron for its future contexts | by absolute activation z 2.63 eight intervals before entry, 6.53 at entry (random non-words 0.05 to 0.25); the first, signed measure (e552) had given 0.06 at −8 and was corrected; the neuron's own write is the top contributor (rank 0 of 73,728); near misses stay selective on OLMo (3.66 to 5.81) | [atlas 33][a33]
- e547b (agent read) | Pythia-410m | same quantity | entrants z 1.61 / 2.45 / 2.71 / 2.85 / 3.08 from k = −8 to +2; near misses 1.91 falling to 0.36; random non-words 0.02 to 0.09 | [atlas 31][a31]
- e526 (agent read) | Pythia-410m b12, origins 4000 / 8000 / 12000 | output vector's motion against its own input weight | the typical row's output vector drifts away from its input weight (cosine −0.10 / −0.07 / −0.05 per thousand steps); entrants' do not (−0.006 / −0.014 / 0.009) | [atlas 26][a26]
- e209 (agent read) | GPT-2, Pythia checkpoints | input-output cosine of every neuron | Pythia mean −0.22 (70% below −0.1); 0 at initialisation, −0.04 at step 1000, −0.18 at step 4000 | [atlas 05][a05]
- e558 (agent read) | Pythia-410m, 16000 into 8000 | entrant made a word by transplant | direction only 0.48, output vector 0.59, whole neuron 0.68 (0.80 singly); stayers 0.95; never-words 0.00 | [atlas 34][a34]
- e558b, e571 (agent read) | Pythia-410m | row part against state-cloud part of an entry | row +0.15 to +0.18, cloud −0.00 to +0.03 (e558b); at one-checkpoint resolution row +0.045, cloud +0.032 (e571) | [atlas 34][a34], [atlas 35][a35]
- e564 (agent read) | Pythia-160m, 300 fine-tuning steps | output-vector rotation under fine-tuning | 0.0001, while the word set changes (Jaccard 0.69 to 0.78) | [atlas 34][a34]
- e632 | Pythia-160m | read-side "ticket" at step 0 (activation selectivity at future class) | median rank 0.33 to 0.38 (top 100 for 0.03 to 0.07) | [atlas 56][a56]

### 2.3 Readers

- e684 | Pythia-160m seeds 0, 1, 2, b6 rows, readers in b7 to b11 | readers of a recruited row, two checkpoints before against two after | ten strongest readers' share gains more than norm-matched rows in 0.79 / 0.76 / 0.75 of words; reader mass 0.81 / 0.77 / 0.75; readers' alignment gain with the row exceeds that with the class direction in 0.97 / 0.94 / 0.91 | [atlas 63][a63], `results/e684/*.json`
- e684, same files | the pre-registered primary test R1 (count of readers) | failed in all three seeds: word beats control in 0.33 / 0.26 / 0.26; median readers 0 before and 0 after; the registered bar was a share above 1 standard deviation, the script lowered it to 0.2 and still found none | `results/e684/*.json`, `results/e680_e691_prereg.json`
- e684, same files | size of the reader gain | median gain of the ten strongest readers' share 0.037 / 0.025 / 0.031 of a pre-activation standard deviation, against 0.0006 / 0.0008 / 0.0005 for controls; reader mass ratio after/before 1.34 / 1.27 / 1.24 | `results/e684/*.json`
- e684 | same | sub-class specialists | 119 → 119.5, 127 → 127.5, 104 → 118; gain above the random-subset control in 0.46 / 0.49 / 0.41 | `results/e684/*.json`
- e691 | final models | does a neuron's activation label predict what its readers do | Spearman −0.02 to 0.04; weight-only reader vectors 0.58 to 0.87 | [atlas 62][a62]
- e685 | 12 model-blocks, final | upstream contributors to a mid-network neuron at strong firings | 1,874 to 15,033 effective contributors; the 16 largest carry 0.02 to 0.05; driving by them alone costs 0.43 to 2.11 of the ablation damage | [atlas 62][a62]
- e574 | Pythia-410m | are words read more than other rows | read energy words 479 against norm-matched 526, random rows 435, random directions 473 | [atlas 36][a36]

### 2.4 Transplant between adjacent checkpoints

All on Pythia-410m block 12, parts of the step-16000 model placed into the step-8000 model.

- e577 | 119 entrants, 1,151 class positions | KL gap to step 16000 at the class positions | 0.2864 nats (elsewhere 0.2639) | [atlas 36][a36], `results/e577_functional_transplant.json`
- e577 | same | closure by the output direction alone, all entrants at once | −0.005 (KL change 0.0021 at class, 0.0017 elsewhere) | [atlas 36][a36]
- e577 | same | closure by the whole neuron (input row, bias, output column) | −0.020: the predictions move 2.0% of the gap away from the donor | [atlas 36][a36]
- e577 | same | controls | random word's direction −0.023; entrants' directions into random neurons −0.004 | [atlas 36][a36]
- e578 | 30 entrants (a first run with 40 gave NaN closures) | share that become words | direction 0.60, 128-neuron "coalition" 0.47, both 0.93, 512 with direction 0.83, block-matched random 128 neurons 0.13 | [atlas 36][a36], `results/FINDINGS.log`
- e578 | same | closure at the class (elsewhere) | direction +0.000 (−0.000); coalition −0.020 (−0.008); both −0.022 (−0.008); 512 with direction −0.066 (−0.021); random −0.005 (−0.001) | [atlas 36][a36]
- e578 | all entrants at once | closure by the readout (blocks 13 to 23, final norm, unembedding) | −0.873 at the classes, −0.994 elsewhere: the gap to the donor grows by 87% and 99%; with all coalitions and directions −0.776 / −0.972; all coalitions and directions without the readout −0.096 / −0.167 | [atlas 36][a36], `results/e578_coalition_transplant.json`
- e578 | necessity, step-8000 parts into the 16000 model | KL change at class / elsewhere | coalition 0.0105 / 0.0015, k = 512 0.0282 / 0.0050, random 0.0005 / 0.0005 (larger at the class in 1.00 / 1.00 / 0.47) | `results/FINDINGS.log`
- e573 | Pythia-410m, 60 classes that change row between 16000 and final | does the new row keep the old direction and effect | direction cosine 0.133 (random 0.007); effect cosine 0.156; kept rows 0.449 | [atlas 36][a36]
- e576, e669 | grokking toy | frozen trained output vectors in a fresh network | usage advantage 0.55 against 0.05 for random frozen rows, classes not carried (profile correlation −0.01); grokking within 20,000 steps 6 of 10 with trained directions against 4 of 15 random (Fisher p 0.11) | [atlas 36][a36]

### 2.5 Gradient

- e589 | Pythia-410m b12, 189 recruits, two checkpoints before recruitment, loss on 24 x 512 Pile tokens | cosine of minus the gradient on the output vector with the class direction | mean +0.020, 3.8 s.e. (recount: mean 0.0204, s.e. 0.0053); matched non-recruits +0.008 (recount: s.e. 0.0074, n = 64); random rows +0.002; class-shuffled −0.002 | [atlas 40][a40], `results/e589_gradient_turn.json`
- e589 | same | share of the gradient's size from the class's positions | 0.215 (median) from 0.006 of positions; matched 0.086, random 0.039 | [atlas 40][a40]
- e589 | same | step against the row's own step-16000 direction | +0.002 | [atlas 40][a40]
- e589, registered M1 | same | median cosine 0.2 or more | median 0.017; over 0.2 for 0.01 of recruits | `scripts/e589_gradient_turn.py`, `results/FINDINGS.log`
- e522 | Pythia-410m b12, b6, origins 2000 to 16000 | does a one-step move along the gradient select next checkpoint's entrants | AUC 0.47 to 0.54 over all non-words; Spearman with actual change 0.00 | [atlas 25][a25]
- e522b | same | reliability of that gradient on 65k tokens | flow split-half 0.19 to 0.44; row gradient split-half 0.02 to 0.05; rows' actual change has cosine 0.002 with the origin's negative gradient | [atlas 25][a25]
- e590 | Pythia-410m, 30 words removed and frozen at 8000, 600 proxy steps (2.5M tokens) | substitute speakers | removed and trained 0.17; removal alone 0.23; random rows removed 0.30; unfrozen 0.30; class loss 3.583 → 3.588 → 3.636 | [atlas 40][a40]
- e631 | Pythia-160m from 8000, 5,000 steps, 10M tokens | planted quiet row near a removed class | over the floor at 0.00 of its class at start and end in every arm; the arm with direction and input side drifts 0.74 → 0.84 of the floor (fixed copy 0.74 → 0.75); continuation raises loss everywhere 3.852 → 4.072 | [atlas 55][a55]

### 2.6 Data attribution

- e628 | Pythia-160m b6, from step 8000, 2,500 steps x 8 windows of 256 tokens (5.12M tokens), `NeelNanda/pile-10k`, AdamW lr 5e-5 | remove the windows containing a group's class bigrams; recruitment of the real run's 50 recruits | full stream 0.28 (A) / 0.20 (B); A removed 0.20 / 0.24; B removed 0.28 / 0.32; random windows removed 0.20 / 0.32; one s.e. 0.09 at 25 rows per group; verdict "no effect; underpowered" | [atlas 54][a54], `scripts/e628_class_data_attribution.py`
- e628 | same | power gate | the full stream recruits 0.24 of the real run's recruits (registered gate 0.25); the targeted removal takes a third of the stream | [atlas 54][a54]

### 2.7 Other developmental numbers on the pages I read

- e517b | Pythia-410m b12, origins 2000 to 32000 | entry prediction with the position's 64 largest writes replaced | real 0.86 to 0.97; fake 0.85 to 0.95; removed 0.86 to 0.96 | [atlas 24][a24]
- e518 | GPT-2, Pythia, OLMo-1B final | criterion picks a disjoint text set's new words | 0.86 / 0.89 / 0.91 (GPT-2 b3/6/9), 0.97 / 0.97 / 0.95 (Pythia b6/12/18), 0.90 / 0.92 / 0.93 (OLMo b4/8/12) | [atlas 24][a24]
- e521, e521b | Pythia-410m b12 | sources of a future word's projection | same-position MLP writes 0.62 to 0.75; ablating top 10 / 100 / 1,000 / 10,000 attributed writes removes 0.09 to 0.18 / 0.25 to 0.30 / 0.51 to 0.57 / 0.89 to 0.92 | [atlas 24][a24]
- e551 | OLMo-1B b8, 1000 to 16000 | criterion, retention, entries, words' S against random atoms | 0.94 vs 0.80; 0.64 vs 0.19; 63 vs 182; 1.70 vs 0.98 | [atlas 33][a33]
- e552 | OLMo-1B b8 | coalition within-row coherence | 0.731 (across rows −0.000; random directions 0.042); persistence 0.969 vs selection-matched 0.511 | [atlas 33][a33]
- e553 | OLMo-1B b8 | WDD features against activation-side features, logistic at H = 4 | 0.92 vs 0.77; WDD adds +0.133, activation side adds −0.004 | [atlas 33][a33]
- e581 | Pythia-160m b6 | SAE features against words: coalition coherence, necessity | 0.692 vs 0.672; zeroing top 128 producers drops a feature 0.92, a word's projection 0.77; producer Jaccard 0.07 | [atlas 37][a37]
- e580b | grokking toy, 20 runs | predicting generalisation 1,000 steps ahead | best WDD 0.824 vs best classical signal 0.961 | [atlas 37][a37]
- e682 | Pythia-410m, Qwen2.5-0.5B, 200 taught facts | fact writers' relation to the fact | fire on related text 0.42 to 0.56 (random 0.14 to 0.16); same-value Jaccard 0.34 to 0.38, cross-relation 0.00 to 0.04; no output vector moves by more than 0.001 in cosine | [atlas 63][a63]
- e688 | Pythia-160m, 5 seeds | planted two-concept "knot" on one row, 6,000 steps of training | installs in 4 of 5, keeps both in 4, sheds one in 0 | [atlas 63][a63]
- e690 | Pythia-160m, 410m, 1B | writers of memorised text against low-loss text | depth 0.83 vs 0.83, count 310 vs 304 | [atlas 63][a63]
- e687 | final models | surface rules fitted to a class, held-out coverage at precision 0.5 | 0.00 to 0.01 | [atlas 63][a63]

### 2.8 The other areas

One line per atlas page, its headline number with its control. Pages 02 to 35 and 42 to 79 are "(agent read)":
an Opus reader agent read them line by line and I quote its extract; I did not re-open them.

- 01 | e00, e01, e04, e10 | 5 final models | WDD reproduces the GPT-2 paper within 0.01 (FVU32 0.339, rotated 0.529); random-init dictionaries sit at the rotated level; pursuit identifies the dominant write worst (GPT-2 0.64 against 0.80 and 0.85 for dual one-shot and LASSO) | [atlas 01][a01]
- 02 (agent read) | e05, e127, e129 | 5 models; Pythia ckpts | per-neuron recall is a stable trait (ICC 0.95 to 0.97; Pythia 0.68 at step 0, 0.90 by step 1000); a covariance-matched threshold predicts identification at corr 0.97 to 0.99 with no fitting | [atlas 02][a02]
- 03 (agent read) | e02, e156, e165 | 5 models | a block's increment reads its write (SmolLM2 0.73 to 0.92 against 0.30 to 0.66 from the state); the gradient-important write is the decomposition's top atom in 15 to 32% of tokens; OLMo reverses the identified-versus-unidentified removal cost (+0.57 against +3.16) | [atlas 03][a03]
- 04 (agent read) | e13, e145 | 5 models | static head atoms capture 1 to 19% of the dominant head's write; head subspaces hold MLP write directions at 0.09 to 0.29 against 0.08 to 0.15 for random subspaces | [atlas 04][a04]
- 05 (agent read) | e194, e207, e209, e217 | 5 models; Pythia ckpts | every trained block contracts a perturbation by a gain of −0.16 to −0.42 (random directions −0.14 to −0.40); the contraction is born at steps 512 to 2000 as warmup ends; input-output cosine of every neuron 0 at init, −0.18 by step 4000 (Pythia) | [atlas 05][a05]
- 06 (agent read) | e227, e239, e243 | 5 models; Pythia ckpts | a write's downstream footprint names its source neuron at 88 to 98% (chance 3 to 6%); final-checkpoint footprints classified through the step-16000 network at 1.00 | [atlas 06][a06]
- 07 (agent read) | e258, e268, e283 | 5 models; Pythia ckpts | the footprint predicts its own future two blocks on at 0.43 to 0.78 (the write 0.06 to 0.12); footprint-function coupling +0.84 at step 0, +0.14 at step 64000, +0.31 final | [atlas 07][a07]
- 08 (agent read) | e286, e304, e313 | 5 models | identity, future, logit and KL need 8 to 32 directions each; the same counts hold for random and other perturbation families (refuted as special) | [atlas 08][a08]
- 09 (agent read) | e317, e332, e349 | 5 models; Pythia ckpts | nine observables share one 16-dimensional subspace at overlap 0.37 to 0.45 (reliability 0.38 to 0.51); WDD atoms are no better a basis for it than random vectors | [atlas 09][a09]
- 10 (agent read) | e20, e432, e437b | 5 models; Qwen2.5-7B | one atom reconstructs each sink state (FVU 0.001 to 0.06); a 7B claim of "97% of variance in 8 directions" was one sink token (11% on evaluation text) | [atlas 10][a10]
- 11 (agent read) | s1, e175, e379 | synthetic data and toys | synthetic recall 1.00 without cancellation; toy circuits sub-additive after convergence (0.20 to 0.62) | [atlas 11][a11]
- 21 (agent read) | e494, e504 | GPT-2, Pythia-160m SAEs | an SAE feature's top native atom is an MLP row for 0.76 to 0.93 of features (rotated rows 0.00); a feature's top row predicts its firing at AUC 0.62 to 0.73 | [atlas 21][a21]
- 22 (agent read) | e503 to e511 | GPT-2, OLMo-1B, Pythia at 8 ckpts | the floor holds (rotated calibration 0.97 to 1.00 in 27 cells); the largest writes ("chord") centred predict GPT-2's maximum within 13% | [atlas 22][a22]
- 23 (agent read) | e512 to e516b | Pythia-410m, 18 ckpts | writer-win rate at block 12 0.02 → 0.73 from step 512 to the end; the largest projection over the floor predicts the next checkpoint's entrants at AUC 0.76 (step 1000) to 0.97 (16000); from step 1000 nothing predicts the end's vocabulary above 0.69 | [atlas 23][a23]
- 26 (agent read) | e525 to e530 | Pythia-410m b6, b12, 53,248 rows | a row's thousand-step update has cosine 0.013 with its negative gradient (positive for 0.63 to 0.65 of rows); the best fitted predictor recovers 0.049 to 0.072 of the update direction's energy; "the line closes here" | [atlas 26][a26]
- 27 (agent read) | e531, e535 | Pythia-410m b12, 16 ckpts | a frozen random or rotated dictionary gives the entry criterion at 0.86 / 0.81 against native 0.94 / 0.91: four fifths of the criterion is generic | [atlas 27][a27]
- 28 (agent read) | e536 to e538 | synthetic rows against a Gaussian cloud with the real covariance | rows diffusing with no training give the criterion at 0.80 and breadth-first entry; a pull into the cloud gives the native level (0.92) | atlas page 28 in [the atlas directory][adir]
- 29 (agent read) | e538b, e539, e541b | Pythia b12, 222 exits | leavers stay over the floor at exit (0.82); exit is loss of usage and breadth while the projection stays ("one-way gate"); the persistence is the covariance's (retention native 0.50, covariance-matched 0.29, random 0.24) | [atlas 29][a29]
- 30 (agent read) | e543, e544 | Pythia rows | Gaussian clouds with the real dynamics miss the native level; grafting each position's top 4 real projections restores it | [atlas 30][a30]
- 31 (agent read) | e545, e546, e547b | Pythia-410m, 53,248 writers | a word's extremes are written by about 9,232 effective writers (top-10 share 0.01); the writers' coherence precedes entry against S-matched near misses (similarity 0.42 to 0.51 against 0.17 to 0.20); the neuron's own selectivity precedes entry (z 1.61 at k = −8) | [atlas 31][a31]
- 32 (agent read) | e548, e549, e673 | Pythia-160m, 410m | WDD's projection predicts entry at 0.88 against 0.74 for activation statistics; single-neuron ablation damage equals activation x gradient (Spearman 1.00 at blocks 12, 17, 22), while WDD's row score is unrelated to damage (0.08, 0.03, −0.01) | [atlas 32][a32]
- 34 (agent read) | e557, e558, e564 | Pythia-160m, 410m, deduped; Mamba; ViT | context-set twins across scale 0.40 and across a data twin 0.39 to 0.43 (rotated control 0.03); transplanting the 16000 neuron into 8000 makes it a word 0.68 | [atlas 34][a34]
- 35 (agent read) | e565, e566, e569b, e571 | Pythia-160m seeds, 410m, Mamba-130m, OLMo-1B | the output vector carries wordhood (0.59 against 0.18 for the input row); twins across seeds 0.34 to 0.42 and Pythia-410m to OLMo-1B 0.41 (control 0.03); at K = 8 norm-matched random rows describe states as well as the words (0.706 / 0.872 both) | [atlas 35][a35]

- 12 (agent read) | e360, e360c, e362, e362d | Pythia 46 ckpts; OLMo-1B 19 ckpts | output vectors settle before function (Pythia aligned 0.60 to 0.72 against ceiling-corrected signature 0.33 to 0.48 at step 16000); OLMo-1B's output vectors keep changing through most of training (aligned with the final 0.52 at step 256000, 0.92 at 1.0M), where Pythia's settle by a third; the final top-5% important neurons enter late in OLMo-1B (median entry step 400000 to 550000 of 1.45M) | [atlas 12][a12]
- 12 (agent read) | e363, e369, e373 | 5 models; Pythia, OLMo ckpts | induction co-selection peaks near z +11 (Pythia steps 3000 to 16000), fades to +2.4 (Pythia) and +1.5 (OLMo); gradients carry 35 to 81% of the ablation signal during training | [atlas 12][a12]
- 13 (agent read) | e375, e381, e387 | 5 models incl. OLMo-1B | reader-weighted WDD puts the previous-token head first for 14 of 20 induction readers; as a general edge finder top-1 0.23 (GPT-2) to 0.02 (OLMo); the readers' preference for identified writes is 1.71 already at Pythia's initialisation, genuine only in GPT-2 and OLMo | [atlas 13][a13]
- 13 (agent read) | e378, e390 | Pythia step 63000 vs final | final upper half under the step-63000 lower half loses 0.28 to 0.42 nats, more than the late gain of 0.115; an affine map repairs about half (the halves co-adapt) | [atlas 13][a13]
- 14 (agent read) | e395, e401, e406, e407, e412 | Pythia ckpts; OLMo-1B ckpts to 256000 | the model's own rows describe its states better than a rotation from step 256, not at step 0 (5 models, within 0.004 to 0.014); OLMo own beats rotation at every trained checkpoint; gradient errors favour writers at step 1000 (0.059 vs 0.020) and readers from step 16000 (0.063 vs 0.019) | [atlas 14][a14]
- 15 (agent read) | e415 to e424 | Qwen2.5-7B; 5 small models | at 7B native words surface a two-hop bridge in the top 20 for 0.11 to 0.21 of prompts against 0.00 to 0.01 for the logit lens, with no causal effect at the subject token | [atlas 15][a15]
- 16 (agent read) | e444b, e448d | grokking toy; small LM; Qwen2.5-7B | with output vectors frozen at random, rows are used at the rotated level (0.69 to 0.75 against 0.74); 22 of 24 nouns get one language-independent row in Qwen2.5-7B | [atlas 16][a16]
- 17 (agent read) | e462, e469, e471 | Pythia ckpts; Qwen | one batch's gradient on a word against its change: cosine 0.000, sign a coin flip; importance and wordhood are independent over 8 checkpoints (lagged Spearman −0.01) | [atlas 17][a17]
- 18 (agent read) | e473, e477 | Qwen | two attributes compose from two rows (74%), task PCA equal (91%) | [atlas 18][a18]
- 19 (agent read) | e479, e653 | Qwen2.5-7B-Instruct; small models | native words hold a concept while copying at 0.30 (logit lens 0.10); the "token after next" draft refuted (0.002 to 0.025 top-1 against bigram 0.229 to 0.285) | [atlas 19][a19]
- 20 (agent read) | e483, e492, e498 | 5 models; Pythia 9 ckpts; OLMo-1B 7 ckpts to 256000 | the floor follows a Gumbel law (KS 0.02 to 0.07); the writers' part of the excess rises with sparsity through 256,000 OLMo steps (block 8: 1.27 to 2.70) | [atlas 20][a20]

- 42 (agent read) | e597, e600 | Pythia-160m, 410m unlearning | the domain writers' output vectors keep cosine 0.9991 to 1.0000 under every gradient method; writer activation predicts 20-step relearning at Spearman 0.82 over 39 conditions (domain probe 0.67); in fine-tuning 25 / 29 entrants keep their rows at cosine 1.00 | [atlas 42][a42]
- 43 (agent read) | e602, e604, e644 | Pythia, Qwen2.5-0.5B | activation-difference neurons predict relearning at 0.83 against the writers' 0.89 (overlap Jaccard 0.06); instruct-only writers' rows at cosine 0.9973 with base (all rows 0.9972); zeroing the 128 rows with the largest usage change keeps 0.78 to 1.03 of a stored margin (random 0.98 to 1.03), refuted | [atlas 43][a43]
- 44 (agent read) | e607, e609, e610 | Qwen2.5-0.5B, 1.5B, Pythia-1B | harmful-request writers' |cosine| with the refusal direction 0.034 / 0.026 (all rows 0.024 / 0.023); restoring silenced writers recovers 0.09 of the forget rise; Pythia-1B writers Spearman 0.90 (activation-difference set 0.87) | [atlas 44][a44]
- 45 (agent read) | e611 | Pythia-410m b4 | class-conditioned Jacobian transport, predicted vs actual logit change 0.283 (random-subset map 0.210) | [atlas 45][a45]
- 46 (agent read) | e616, e613 | 63 conditions; Qwen2.5-0.5B TOFU | leave-one-method-out writers AUC 0.93, transfer 0.89 (majority 0.57); on TOFU 2 of 256 words pass the threshold, no readout | [atlas 46][a46]
- 47 (agent read) | e617b | 255 held-out conditions | registered rule accuracy 0.62 / 0.51 / 0.73 on held-out domains / methods / architecture (majority 0.55); AUC 0.85 / 0.91 / 0.93; refuted at the fixed cut | [atlas 47][a47]
- 48 | e618a, e618e | same | writers' AUC 0.85 [0.76, 0.93] (domains); the "steps back" delay is a floor artefact: floor-corrected recovery 0.84 (writers) against 0.80 (random) and 0.80 (no silencing) | [atlas 48][a48]
- 49 | e619, e620, e621 | Qwen2.5-0.5B, Pythia-160m | 98 of 256 words shared by base and instruct; one negated row flips gender in 0.34 of items at +0.000 Pile nats (random rows 0.00) | [atlas 49][a49]
- 50 (agent read) | e622, e662 | Qwen2.5-0.5B, 1.5B-Instruct | one gender row flips 0.28 of held-out items (random 0.00); held-out flips 0.00 in 3 of 8 attribute cells | [atlas 50][a50]
- 51 (agent read) | e623, e667 | four small models | blind detection of a negated word row: top ten in 0.00 of 10 | [atlas 51][a51]
- 52 (agent read) | e625, e626 | Qwen2.5-0.5B, Pythia-160m | one row negated: agreement 0.93 → 0.58 (French), 0.93 → 0.58 (Spanish), 0.84 → 0.53 (German) at +0.0003 nats; random row 0.00; Qwen2.5-1.5B-Instruct unchanged | [atlas 52][a52]
- 53 (agent read) | e627 | Pythia-160m, 3 seeds | a gradient "pin" installs the vote (1.00) but the behaviour dials are −0.029, +0.027, −0.006 | [atlas 53][a53]
- 57 (agent read) | e633 to e639 | six to thirteen final models incl. OLMo-1B | rows for half the weekday ring: Pythia-160m 181 ... OLMo-1B 28; removing writers does not affect OLMo-1B month arithmetic (1.00 throughout) | [atlas 57][a57]
- 58 (agent read) | e640 to e650 | 22 model-calendar pairs | the class-mean subspace test passes every class kind in OPT and OLMo-1B, so it does not single out calendars there | [atlas 58][a58]
- 59 (agent read) | e641, e646 | six models incl. OLMo-1B; fine-tune snapshots | in-prompt classes decode at 0.85 to 0.96 but the nearest output vector sits at percentile 0.21 to 0.76; a class stored by fine-tuning gets a row at percentile 0.93 to 0.98 within 10 steps in Pythia-410m | [atlas 59][a59]
- 60 (agent read) | e656 to e658 | three small models | repeating a block never lowers wikitext loss (least +0.045 to +0.057 nats) | [atlas 60][a60]
- 61 (agent read) | e664b, e668 | Qwen2.5 cells | a plain direct-contribution ranking without pursuit matches the WDD vote within 0.05 in 16 of 16 cells | [atlas 61][a61]
- 64 | e693, e694 | 11 models; Qwen2.5-14B | 16 principal directions of the removed weak write recover 0.60 to 0.96 of the zeroing cost (per-token table 0.15 to 0.95); at 14B the sum of small terms is 0.96 of the resultant | [atlas 64][a64]
- 65 (agent read) | e701, e696, e700 | Pythia-160m b6, 3 seeds; 8 models; Qwen2.5-7B fp32 | the "brake" goes from +0.28 to +0.30 (step 1000) to −0.60 to −0.76 (final), Spearman −0.99 to −1.00; a rank-16 linear map recovers 0.56 to 0.96 of the zeroing cost; at 7B in fp32 the exact input split reproduces the neuron (split error 5e-6) where bf16 fails (0.86 to 0.95) | [atlas 65][a65]
- 66 (agent read) | e701 | OLMo-1B b4, b8, b12 | the same quantity −0.05 to −0.27 at step 1000, between −0.31 and +0.01 through step 16000, −0.23 / −0.56 / −0.69 final | [atlas 66][a66]
- 67 (agent read) | e701 late | OLMo-1B b8, b12 | passes −0.5 at step 512000 (b12) and 1,000,000 (b8); Spearman with step −1.00 | [atlas 67][a67]
- 68 (agent read) | e707 | 5 to 6 model-blocks | the output's Fisher form predicts the finite KL of removing a part at Spearman 0.83 to 0.92 | [atlas 68][a68]
- 69 | e713, e714 | Qwen2.5-7B fp32 b14, b21 | 16 strongest inputs alone cost 59.43 to 284.33 times the ablation damage, 1.06 to 2.70 with 16 inhibitory inputs added; Fisher-chosen atoms no better than Euclidean (0.68 to 0.97 against 0.71 to 0.98) | [atlas 69][a69]
- 70 (agent read) | e715 | GPT-2 small | 16 native rows recover 0.92 to 0.95 of mean-ablation KL, SAE top 16 0.95 to 0.97 | [atlas 70][a70]
- 71 (agent read) | e721, e720 | Pythia-410m b12 ckpts; Qwen2.5-7B | brake +0.44 (step 1000) → −0.76 (final), Spearman −1.00; weak-activation cost at 7B 1.566 nats as recorded, 0.32 after the erratum | [atlas 71][a71]
- 72 (agent read) | e723, e724 | GPT-2 small | at a matched budget of 16 directions: fields 0.24, SAE 0.23 to 0.31, random 0.25 (block 6) | [atlas 72][a72]
- 73 (agent read) | e725 | 12 models | corrected strongest-quarter cost 0.32 to 8.06 nats (recorded 0.24 to 8.68) | [atlas 73][a73]
- 74 (agent read) | e727, e726, e728 | 8 models; Qwen3-4B; Qwen3.5-4B | weak/random cost ratio 0.73 to 2.27, verdict "no support" for weak activations mattering beyond their energy; closed-form rank-16 map 0.332 against trained 0.210 | [atlas 74][a74]
- 75 (agent read) | e729 | Qwen3.5-4B | dose-scaled predictor Spearman 0.30 against 0.35 unscaled; refuted | [atlas 75][a75]
- 76 (agent read) | e730 | Qwen3.5-4B | a row's direct-path vocabulary matches 7 of 11 themes blind (shuffled 0 of 11) | [atlas 76][a76]
- 77, 78 (agent read) | e731, e732 | Qwen3.5-4B | a one-token prefix injection keeps a persona for 4 of 6 rows; zeroing the row after the prefix changes nothing (1 of 6) | [atlas 77][a77], [atlas 78][a78]
- 79 (agent read) | e733 | Pythia-410m L6, L12, L18 | sampled-Fisher price against removal cost 0.89 / 0.89 (rows / SAE, L12); price against activation energy 0.97, 0.94, 0.97 | [atlas 79][a79]

---

## 3. Controls and statistics worth reusing

Each control below is in an MIT-licensed script. The scripts hard-code the author's box paths
(`/workspace/wdd/...`) and read caches built by earlier scripts, so they are designs to copy, not tools to run.

1. **Matched non-recruit.** A row that is not a word at the same checkpoint, has the same cosine with the class
   direction within 0.02, and is never a word at 4000 to 16000; paired one per recruit. Found for 64 of 189
   recruits. `scripts/e589_gradient_turn.py`. For us: a pole whose top-context overlap with the job is matched at
   the earlier point and which is never born.
2. **Class-shuffled pairing.** Each recruit's statistic computed against another recruit's class direction.
   Same positions and same rows, only the pairing broken. `scripts/e589_gradient_turn.py`, `scripts/e632_lottery.py`.
3. **Norm-matched rows on the same positions.** Three non-word rows of the same block nearest in output-vector
   norm, scored at the word's own class positions, plus three random rows. `scripts/e684_readers.py`;
   `scripts/e572_norm_matched_words.py`. It controls the norm, not the activation at those positions (see 4.6).
4. **Block-histogram-matched random neurons.** A random set with the same count per block as the tested
   "coalition". `scripts/e578_coalition_transplant.py`.
5. **Transplant in both directions.** Later parts into the earlier model (sufficiency) and earlier parts into the
   later model (necessity); the docstring gives the reason: the necessity direction is free of "the
   checkpoint-mixing confound of a later readout on earlier states". `scripts/e578_coalition_transplant.py`.
   Its closure statistic `(KL_before − KL_after) / KL_before`, at class positions and elsewhere, is the right
   signed readout for N5.
6. **Classes carried from another seed by position.** The class is defined by seed 1's rows and the speaker is
   searched in seed 0, which removes the circularity of a class defined by the tested row.
   `scripts/e632_lottery.py` (`--ref`). The same idea for us: define the job on one checkpoint chain's banked
   contexts and test the pole in another analysis that did not use those contexts.
7. **Definition swap ("home advantage").** Score every candidate rule on a class defined by the output side
   and on a class defined by the input side; a rule that wins only on its own definition is not evidence.
   `scripts/e666_provenance_baselines.py`.
8. **Time-shuffled pseudo-birth and difficulty-matched positions.** Loss aligned on a random pseudo-recruitment;
   control positions matched on token and on loss within 0.25 nats one checkpoint before.
   `scripts/e591_recruitment_matters.py`, `scripts/e591c_matched_controls.py`. Bootstrap intervals over words.
9. **Forecast controls.** Leave-one-checkpoint-out AUC with a label shuffle (0.501), a static early-checkpoint
   score (0.640), retrodiction (0.897), a rotated-dictionary score (0.806) and, decisively, the autocorrelation
   baseline (the row's own current usage, 0.923 to 0.968). `scripts/e629_forecast_headtohead.py`,
   `scripts/e588_forecast.py`. Any birth forecast of ours must beat "the pole's own current score".
10. **Split-half reliability of a gradient.** The same gradient quantity on two halves of the text; if the
    halves disagree (row gradient split-half 0.02 to 0.05), a non-result is not evidence. `scripts/e522b_gradient_reliability.py`.
11. **Persistence-matched initial alignment.** The component of the initial row orthogonal to its final
    direction, against a control that keeps the persistence. Written, never run (`scripts/e632b_*`, the
    persistence-matched script).
12. **Exact pre-activation split through the norm.** A later neuron's pre-activation written as the sum of
    each upstream write's contribution through LayerNorm, checked to reproduce the neuron. `scripts/c11.py`,
    `scripts/e676_center_out.py`, `scripts/e685_readers_*` (the e685 script).
13. **The split's own identity check, in fp32.** Drive the neuron with every one of its inputs and require
    0.00 of its ablation damage before trusting any K-input number. At Qwen2.5-7B block 14 in fp32 the split
    error is 5e-6; in bf16 the same check fails at 0.86 to 0.95 of the damage (e700, [atlas 65][a65], agent read;
    `scripts/e700_*`). For N3 at 7B this is the gate the route measurement must pass first.
14. **Each intervention against its own floor.** When inputs are zeroed, re-score recovery against the loss the
    zeroed model can reach, not against the intact model; this turned the program's "silencing slows
    relearning" into 0.84 against 0.80 for random sets (e618e, [atlas 48][a48]; `scripts/e618e_floors.py`).
15. **Pre-registration as JSON.** Each test with an honest probability and a boolean verdict field written by
    the script (`results/e680_e691_prereg.json`, `results/e632_prereg.json`, `results/e628_prereg.json`). The
    reuse is the format; see 4.7 item 5 for one deviation inside it.

## 4. Weaknesses

### 4.1 Model scale

Every developmental result is on Pythia-160m, Pythia-410m or OLMo-1B. The largest developmental model has 1.2B
parameters. 7B (Qwen2.5-7B) and 14B (Qwen2.5-14B) appear only as final models, with no checkpoints (README,
"The workspace agenda on a 7B model"; [atlas 69][a69]; [atlas 64][a64]). The recruitment window is steps 1000
to 16000: 11% of Pythia's 143,000 steps and 1.1% of OLMo-1B's 1,454,000. Later checkpoints are used only coarsely
(19 OLMo-1B and 46 Pythia checkpoints for neuron drift in e360 and e362; 11 to 13 for e674, e676, e701). Nothing is
dated after step 16000 at 1,000-step resolution.

### 4.2 GeLU against SwiGLU

The core growth chain (e584 to e591, e565, e577, e578, e589, e632, e666, e684) is on Pythia, which is GeLU with a
parallel residual. Only the forecast (e588), the loss part of e591, the cache statistics (e551 to e555), the
routes (e674, e676) and the "brake" timing (e701) were run on a SwiGLU model (OLMo-1B). Results that differ by
architecture: a near miss keeps its input-side selectivity on OLMo-1B (z 3.66 to 5.81) and loses it on Pythia
(1.91 to 0.36) ([atlas 33][a33], [atlas 31][a31]); the input-output cosine of a neuron is positive early and
negative late in gated models, negative throughout in GeLU models (e209, [atlas 05][a05]); the negative pole is
near the positive pole in 0.91 to 0.97 of gated-model neurons and 0.00 to 0.48 in GeLU models ([atlas 62][a62]);
the "brake" passes −0.5 by step 4000 in Pythia-410m (e721) and 16000 in Pythia-160m, and not before step 512000
in OLMo-1B (e701; [atlas 65][a65], [atlas 67][a67], [atlas 71][a71]). The program never splits a SwiGLU
neuron's input into gate and up rows in a developmental test.

### 4.3 Number of neurons and positions

The tracked objects are the top-256 rows of one block per model: 241 words (e584), 253 classes (e585), 189
recruits for the gradient (e589, matched controls found for 64), 119 entrants for e577, 30 for e578, 139 to 151
per seed for e684, 50 for e628, 159 on OLMo-1B (e591). The states are 24 x 512 Pile evaluation tokens (about
12,250 positions) for Pythia and 8 x 256 (2,040) for OLMo-1B and for the e577/e578 transplants (`pile_ids` in
`scripts/s101_common.py`). A class is often a few positions: the 30 classes of e578 have 3 to 244 positions,
median about 12 (`results/e578_coalition_transplant.json`). `docs/PAPER.md` §5 states the limit itself:
"Twelve thousand positions of Pile text per checkpoint; sixteen checkpoints of one run at one depth".

### 4.4 How a class is defined

wdd's class is the set of evaluation positions where a row's output vector has |projection| over an
extreme-value floor on the residual state, at step 16000. Ours is an LLM clustering of a pole's top
activating contexts. Four consequences:
1. **It is an output-side definition.** "Word" and "class" are both read off the output vector. A test of
   whether the input or the output side moves first is then biased toward the output side. Example: the later
   output vector makes an entrant a word in 0.59 of cases and the later input row in 0.18 (e565), but the word
   criterion is computed on the output vector, which only the first transplant changes directly.
2. **The definition decides which predictor wins.** On a class defined by projection the class ratio names the
   recruit in the top ten for 0.77 to 0.84; on a class defined by firing, selectivity does (0.79 to 0.88) and the
   ratio falls to 0.52 to 0.62 (e666, [atlas 56][a56]).
3. **The class is fixed at the end and read backward.** The class positions used two checkpoints before
   recruitment are where the future row projects at 16000. The cross-seed control in e632 is the only test that
   removes this circularity.
4. **It is geometric, not semantic.** A class is predicted by the token, the previous token, the position and
   four flags at AUC 0.853 (e574, [atlas 36][a36]), while surface rules fitted to a neuron's own top contexts
   cover 0.00 to 0.01 of its held-out positions (e687, [atlas 63][a63]). wdd's classes and our jobs are
   therefore different objects; a wdd number about a class is not a number about a job.

### 4.5 No dated birth per neuron, no data order

"Recruitment" is entry into the top-256 pursuit usage at 1,000-step resolution inside steps 1000 to 16000. There
is no birth date per neuron over all of training and no job definition independent of the output vector. The only
data intervention (e628) used a proxy stream (`NeelNanda/pile-10k`), a fresh AdamW optimizer at lr 5e-5 and 5.12M
tokens, against a real interval of about 2B tokens; the author calls it underpowered and lists the real data order
as not done (`docs/NEXT.md`, session 123; `docs/PAPER.md` §5). The gradient test (e589) uses the raw loss gradient
on 12,288 evaluation tokens at one checkpoint, not the Adam update, and the row-gradient estimate has split-half
reliability 0.02 to 0.05 on 65k tokens (e522b, [atlas 25][a25]).

### 4.6 Replication across seeds

Three seeds (Pythia-160m PolyPythias 0, 1, 2) for e629, e632, e666, e683, e684, e688, e701. One seed, one block and
one checkpoint pair for every transplant (e565, e577, e578), the gradient test (e589), the removed speaker (e590),
the planted row (e631) and the data test (e628). Pythia-410m and OLMo-1B results are single runs.

### 4.7 Places where the numbers do not support the prose

1. **Transplant closures are negative, and the readout moves the model away.** Atlas 36: "nothing short of the
   readout moves the recipient's predictions at the class toward the donor (closure -0.066 for 512 producers with
   the direction, -0.873 for the readout alone)". By the script's definition a negative closure moves the
   recipient away from the donor. The readout transplant raises the KL to the step-16000 model by 87% at the
   class and 99% elsewhere (`results/e578_coalition_transplant.json`: −0.8732, −0.9937). The whole-neuron
   transplant moves it away by 2.0% (e577, −0.0202). So "the neuron closes 2% of the gap and the readout 87%" is
   wrong in sign: no condition closed more than 0.0001 of the gap (the best is the direction alone in e578,
   +0.00003). The e578 docstring itself names the reason a later readout on
   earlier states fails ("the checkpoint-mixing confound"). Our own `04_LITERATURE.md` N5 repeats the misreading
   ("the readout moves 87%", "the bottleneck ... in the readout"); see section 8.
2. **The gradient result is above zero, not above its matched control.** Atlas 40: "3.8 standard errors above
   zero and above matched non-recruits (+0.008)". Recount from `results/e589_gradient_turn.json`: recruits mean
   0.0204 (s.e. 0.0053, n 189), matched non-recruits 0.0081 (s.e. 0.0074, n 64); the difference 0.012 is about
   1.3 combined standard errors. The registered M1 ("median cosine 0.2 or more") failed: median 0.017, over 0.2
   for 0.01 of recruits. `docs/PAPER.md` §1 calls the matched controls "at zero".
3. **The forecast headline omits the baseline that beats it.** README TL;DR: "forecast at AUC 0.93 on
   Pythia-410m and OLMo-1B". The row's own current usage, the quantity that defines a word, forecasts at 0.964 and
   0.968, and the class side adds +0.006 / +0.003 to S (registered F2 required +0.03). Atlas 55 states this
   omission itself ("a comparison the paper's sentence left out"); the README written later still uses the 0.93.
   The class-side score that reaches 0.93 is "the largest mean ratio over any cluster"; the first logged run of the
   same script, seven minutes earlier, scored the "owned-cluster" version at 0.582 and 0.521
   (`results/FINDINGS.log`, 2026-09-29 01:44 and 01:51), and the committed docstring now calls that version
   degenerate.
4. **The step-0 "ticket" was withdrawn but stays in the README.** README TL;DR: "at initialisation the future
   recruit is already in the top fifth of rows by alignment with its class's final direction". Atlas 56's S128
   correction withdraws it: the persistence of the random initial direction accounts for it.
5. **The readers test passed only its secondary readouts.** The registered primary R1 counted readers above 1
   standard deviation (`results/e680_e691_prereg.json`). The script lowered the bar to 0.2 ("one std found
   nobody") and still found a median of 0 readers before and after; R1 is false in all three seeds (0.33 / 0.26 /
   0.26). The atlas reports the zero counts but leads with R1b and R1c. Two features of `scripts/e684_readers.py`
   weaken R1b, R1c and R2 further (my reading of the script, not stated by the author): the reader share
   multiplies the word's own activation at its class, which rises at recruitment while the norm-matched controls'
   activation is not matched; and the readers whose alignment is compared are selected by their response to the
   row after recruitment, which favours "alignment with the row rose" over "alignment with the fixed class
   direction rose".
6. **The "brake" timing is Pythia's, not OLMo's.** README TL;DR: "the crowd reinforces the wires at step 1000,
   the sign flips by step 2000". On OLMo-1B the registered "brake by step 8000" is false at blocks 4, 8 and 12;
   the opposition passes −0.5 only at step 512000 (block 12) and 1,000,000 (block 8) (`results/e701/`).
7. **Flags from the agent-read pages** (each checked by the agent against the page's own table):
   - Atlas 31: "the coalition precedes the word" is established against S-matched near misses only; against random
     non-words the writers' aggregate direction was as stable (0.75 against 0.79) and that reading was retracted
     on the same page (e547).
   - Atlas 32: single-neuron ablation damage is activation x gradient (Spearman 1.00) and WDD's row score is
     unrelated to it (0.08, 0.03, −0.01); the page header does not mention the second result (e673).
   - Atlas 42, 43, 44, 47: the "silencing slows relearning" step counts stay in the Established bullets after
     e618e marked them a floor artefact ([atlas 48][a48]).
   - Atlas 71: the 7B weak-activation cost is stated as 1.566 nats with no mark; `docs/ERRATA.md` corrects it to
     0.32 (e720).
   - Atlas 74 concludes that "the claim that the weak activations matter beyond the energy they carry has no
     support in these eight models" (e727), while README result 2 still presents the weak activations as a learned
     suppressor.
   - Atlas 05: "every trained block answers a perturbation along any direction with a linear gain of −0.16 to
     −0.42", against about 0 along the state's own direction (e214) and +0.70 at Pythia block 0 (e210).
8. **Dates.** The README says the experiments ran "between 2026-09-19 and 2026-09-24, one to five minutes each";
   the result timestamps run to 2026-10-28 (e733), and several runs took up to 48 minutes (e591: 2,909 s).

## 5. What it did on OLMo-1B

Model: `allenai/OLMo-1B-0724-hf` (SwiGLU, 16 blocks, width 2048), key `olmo1b` in `scripts/wdd_common.py`.
Three kinds of use.

**A. The developmental cache (e550), steps 1000 to 16000.** Sixteen checkpoints 1,000 steps apart (the
revisions `step1000-tokens2B` to `step16000-tokens33B`), block 8 of 16, the MLP output vectors of blocks 0 to 8
(73,728 rows of dimension 2048), 8 x 256 evaluation tokens = 2,040 positions, none a sink ([atlas 33][a33]).
On this cache it ran:
- e551: the word criterion, retention, entries and S against a fixed random dictionary (0.94 vs 0.80; 0.64 vs
  0.19; 63 vs 182; 1.70 vs 0.98) ([atlas 33][a33]).
- e552: "coalitions" of producers at step 8000 (within-row coherence 0.731; effective writers 3,175; MLP and
  attention shares 0.86 / 0.14); 275 entrants followed from 8 intervals before entry ([atlas 33][a33]).
- e553: WDD features against activation features for predicting entry (logistic 0.92 vs 0.77) ([atlas 33][a33]).
- e554: the entering neuron's input side by absolute activation: z 2.63 / 3.99 / 6.53 at 8 / 4 / 0 intervals
  before entry, its own write the top contributor at every interval ([atlas 33][a33]).
- e555: the same statistics on a Gaussian cloud with the real covariance ([atlas 33][a33]).
- e588: the recruitment forecast, K = 128 row-free clusters, class side 0.932 / 0.901 / 0.848 at 2 / 4 / 8
  ahead, the row's S 0.965, the row's usage 0.968, precision at 100 0.10 ([atlas 39][a39], `results/FINDINGS.log`).
- e591 and e591c, the loss part only: 159 tracked words, class minus token-matched −0.0111 [−0.1104, +0.0512],
  minus difficulty-matched −0.0140 [−0.0703, +0.0471] ([atlas 41][a41], `results/FINDINGS.log`). The removal
  test of e591 was run on Pythia-410m only.
- e595: row-free clusters per word at steps 8000 and 16000 ([atlas 41][a41]).

**B. Coarse checkpoints across all of training.**
- e362, e362b, e362d: per-neuron drift of output vectors and of a logit signature over 19 checkpoints,
  `step0-tokens0B` to `step1454000-tokens3048B` (`scripts/olmo_revs.txt`). Output vectors aligned with the final
  0.52 at step 256000 and 0.92 at 1.0M, "where Pythia's settle by a third"; ceiling-corrected signatures trail the
  output vectors everywhere (step 550000: 0.51 to 0.66 against 0.75 to 0.76) ([atlas 12][a12], agent read).
- e360c: the final model's top-5% important neurons enter the top 5% late in OLMo-1B, median step 400000 to 550000
  of 1.45M ([atlas 12][a12], agent read).
- e401, e406, e410: own rows describe states better than a rotation at every trained OLMo checkpoint; later rows
  read earlier states better up to step 256000 ([atlas 14][a14], agent read).
- e498: the "two clocks" on 7 OLMo checkpoints to step 256000; the writers' part rises with sparsity (block 8:
  1.27 to 2.70) ([atlas 20][a20], agent read).
- e674, e676: routes of a neuron's top firings by exact upstream attribution, blocks 4, 8 and 12, 300 neurons
  per block, 256 windows (65,024 positions), 11 checkpoints (steps 1000, 2000, 4000, 8000, 16000, 32000, 64000,
  128000, 256000, 512000 and final; `scripts/e674_download.py`). Growth from the centre of the input
  attributions in 0.63 / 0.67 / 0.63 of neurons (recount from `results/e676/e676_olmo1b_b{4,8,12}.json`;
  p below 1e-5 each); the bulk-first lead 0.56 / 0.59 / 0.44 (p 0.030 / 0.0035 / 0.97), so it fails at block 12.
- e701: when the "brake" forms (the summed sub-threshold inputs opposing a neuron's 16 largest inputs off its
  class), blocks 4, 8, 12, 100 neurons each, steps 1000 to 16000 and 32000 to 1,000,000 plus final. The
  registered "brake by step 8000" was false at all three blocks; the summed opposition passes −0.5 at step
  512000 (block 12), step 1,000,000 (block 8) and never (block 4; final −0.23) (`results/e701/e701_olmo1b_*.json`).

**C. The final model**, as one of the five phase-1 models (e00 to e356) and in later final-model tests: e518
(criterion across texts 0.90 / 0.92 / 0.93 at blocks 4 / 8 / 12, [atlas 24][a24]), e566 (context-set twins with
Pythia-410m at 0.41 by character spans, `docs/PAPER.md` §1), e686 (negative pole, [atlas 62][a62]), e689 (width
series), and the final-model pages in 2.8.

**Not done on OLMo-1B:** any transplant (e565, e577, e578 are Pythia-410m only), the gradient (e589, Pythia-410m),
readers across recruitment (e684, Pythia-160m), the step-0 test (e632, Pythia-160m), data removal (e628,
Pythia-160m), the direction-removal test (e591, Pythia-410m), and any analysis of the gate and up rows separately.

## 6. What it did not do that our assets allow

| card | our plan | wdd |
|---|---|---|
| N1 input rows vs output column at a dated birth | gate, up and down change of the same pole at its birth interval against its own non-birth drift, 7B, 16 points | **Did at small scale, indirectly, and the result points against "the output column barely turns".** The output vector is the moving part: a row's entry is row motion 0.64 to 0.78 of the rise, state motion 0.02 to 0.24 (e522, Pythia-410m, [atlas 25][a25]); recruits' output vectors rotate from step 0 as far as other rows (0.20 vs 0.19, e632, Pythia-160m, [atlas 56][a56]); the later output vector makes a row a word in 0.59 of entrants, the later input row in 0.18 (e565, Pythia-410m, `scripts/e565_factorial_transplant.py`). The input side is also selective early: the entering neuron's activation already prefers its future contexts 8 intervals before entry on both models (z 1.61 on Pythia-410m, e547b, [atlas 31][a31]; 2.63 on OLMo-1B, e554, [atlas 33][a33]); this is activation, not a measured change of the input weights. Entrants' output vectors, unlike typical rows', do not drift away from their own input weights (e526, [atlas 26][a26]). **Did not:** measure input-row change and output-column change of the same neuron per interval against its own baseline; separate gate from up; any birth after step 16000; any model above 1B. Caveat for reuse: wdd's "word" and "class" are defined on the output vector, so its input-versus-output comparisons favour the output side by construction (4.4). |
| N2 use at birth | activation x gradient and zero-ablation of the pole on its job's contexts at each point | **Did at small scale, other quantities.** Loss at the class across recruitment does not change beyond controls: −0.0102 [−0.0292, +0.0116] (Pythia-410m), −0.0111 [−0.1104, +0.0512] (OLMo-1B) (e591, [atlas 41][a41]). Removing the row's direction from the block-12 state at the class costs 0.0047 nats two checkpoints before recruitment and 0.0109 two after, random direction 0.0002 (e591); the strongest other row costs 0.0058 at −2 (e591b). Their reading: the direction is used before the row is recruited. Separately, on the final Pythia-410m, zeroing each neuron of three blocks one at a time gives damage equal to activation x gradient at Spearman 1.00, 1.00, 1.00 (e673, [atlas 32][a32], agent read), which supports activation x gradient as N2's use measure. **Did not:** activation x gradient or zero-ablation of one neuron across its birth; jobs defined by contexts; any 7B model. |
| N3 route at birth with contributors' ages | top-10 upstream neurons and heads per birth, each with its own birth date | **Did at small scale, without contributor ages.** The 128 largest producers along the eventual row cohere 5.0 checkpoints before recruitment for 0.84 of words (e584, [atlas 38][a38]); a future word's projection comes 0.62 to 0.75 from same-position MLP writes, and its five largest sources are non-words six to seven blocks below (e521, [atlas 24][a24]); routes of top firings by upstream attribution grow from the centre in 9 of 9 runs (e676). **Did not:** date each contributor; name heads in the route; follow a route past step 16000 at 1,000-step resolution. |
| N4 documents by exact data order | the documents in the birth interval, per-document gradient against the realized change, then a counterfactual retrain | **Did not.** The only data test (e628, Pythia-160m) removed windows from a proxy stream (`NeelNanda/pile-10k`, fresh AdamW, 5.12M tokens) and found no specific effect at one s.e. 0.09 ([atlas 54][a54]). The author lists the real-order version as future work: "the same intervention on the real Pythia data order inside the 8000 to 10000 window ... which needs the preshuffled Pile" (`docs/NEXT.md`, session 123). `docs/PAPER.md` §5: "not the real data order". |
| N5 skill transplant across adjacent 7B checkpoints | top-k most-changed neurons, benchmark accuracy, random set matched on count and energy | **Did at small scale with a negative result,** on next-token KL at class positions, not on a skill, between steps 8000 and 16000 of Pythia-410m (8,000 steps apart, not adjacent): whole neuron closure −0.020, direction −0.005 (e577); 512 producers with the direction −0.066; readout −0.873 (e578) ([atlas 36][a36]). Every closure is negative: no transplant moved the earlier model toward the later one. **Did not:** select by change norm; score a benchmark; transplant between 1,000-step neighbours; go above 410m. |
| N12 competitor gradients | gradient to every pole of a layer on one job's contexts, before and after birth, with the realized update | **Did at small scale for the winner only.** The recruit's gradient two checkpoints before: 0.215 of its size from its class's positions (matched non-recruits 0.086, random 0.039); direction cosine +0.020 (e589, Pythia-410m, [atlas 40][a40]). The one-step gradient does not select the next checkpoint's entrants (AUC 0.47 to 0.54; e522, [atlas 25][a25]); future words' gradient share never separates from matched non-words (AUC at most 0.61; e514, [atlas 23][a23]); over 3M tokens a row's thousand-step update has cosine 0.013 with its negative gradient, raw or Adam-whitened (e525, e525b, [atlas 26][a26]); one batch's gradient on a word has cosine 0.000 with its change (e462, [atlas 17][a17]). **Did not:** the gradient to every competing pole on one job's contexts across its birth; the realized update per pole on those contexts. |
| N20 all-neuron census | a mechanical birth rule on all 704,512 poles at 16 points | **Did at small scale for one block range and one window.** Word status, S and usage for every output vector of blocks 0 to B at each of 16 checkpoints (53,248 rows in Pythia-410m, 21,504 in Pythia-160m, 73,728 in OLMo-1B), but "word" is a top-256 cut and the window is steps 1000 to 16000. The census closest to N20 is e515 / e515b on Pythia-410m, 13 checkpoints: rows used as words fall from 12,309 at step 1000 to 9,435 at the end at block 12; 0.04 to 0.05 of step-1000 words are words at the end and 0.70 to 0.73 of the end's words are new after step 4000 ([atlas 23][a23], agent read); the drift of every neuron over 46 Pythia and 19 OLMo checkpoints (e360, e362). One result bears on N20's claim directly and against it on OLMo-1B: output vectors are aligned with the final only 0.52 at step 256000 of 1.45M, and the final important neurons enter the top 5% at a median step 400000 to 550000 (e362, e360c, [atlas 12][a12], agent read). **Did not:** a birth date per neuron over all of training; a birth count per layer per interval; jobs defined by contexts; any 7B model. |

## 7. Relationship risk

Facts as of 2026-11-05. The repository has 0 stars, 0 forks, no issues ever opened and discussions disabled
(GitHub API). The last research commit is 2026-10-28 (e733, "on a Quadro P4000"); the last commit is the README
TL;DR of 2026-10-31. Since then the author's public pushes are unrelated to interpretability (a Nostr client
build, an Amethyst Homebrew tap, a t-shirt store demo, a personal site), except `wdd-playground`, a GPT-2-small
provenance explorer created and last pushed on 2026-10-31. Hardware in the record: one rented A100 80 GB for most
rounds, a local Dell with one or two Quadro P4000 cards (8 GB, "a 2017 card"), and "the 27B does not fit the 40 GB
box". `docs/NEXT.md` (agent read, in full) says "The box was released on 2026-10-11 and its caches (e629,
e582-style) are gone; a new box must rebuild them"; the 7B work it lists is final-model only (the "brake" at
Qwen2.5-7B's late block, the two-field dictionary at 7B), with several items marked "needs a rental". Its
developmental items are on Pythia: "the same test on Pythia-410m and OLMo-1B, whose early checkpoints exist"
(session 127), "denser late checkpoints" for OLMo-1B block 12 (session 153), the real Pythia data order "which needs
the preshuffled Pile" (session 123), and pricing "the same rows ... across the Pythia-410m checkpoints the Dell
already holds" (session 171). The README's only new-model plan is "A second developmental model with public
checkpoints and chat variants (OLMo 2)". OLMo-3 is not mentioned anywhere in the repository (search of README,
`docs/`, `scripts/`; the one string match, `dl_olmo3b.log` in `scripts/jobs_s159.txt`, is a download log that the
OLMo-1B late-checkpoint runs of e701 wait on). No file loads a data order or any 7B checkpoint series. Its
training runs are toys, a grokking task and proxy continuations of Pythia-160m and 410m of at most 10M tokens.

---

## 8. Corrections this reading makes to our own record

1. `04_LITERATURE.md` N5 says the wdd transplant "is negative for predictions: the whole neuron moves 2% of the
   gap, the readout moves 87%", and that wdd puts "the bottleneck outside the new neuron ... in the readout". Both
   closures are negative (−0.020, −0.873): the whole neuron and the readout each move the earlier model away from
   the later one (section 4.7 item 1). wdd does not show that the readout carries the skill; it shows that no
   transplant between steps 8000 and 16000 moved predictions toward step 16000.
2. `04_LITERATURE.md` Corrections item 4 could not find the "write column 0.59, read row 0.18" figure. It is e565,
   the factorial transplant (`results/FINDINGS.log`, 2026-09-28; `scripts/e565_factorial_transplant.py`): the share
   of 119 entrants that become words when one part of the step-16000 neuron is put into the step-8000 model.
3. `04_LITERATURE.md` N1 quotes e589 as recruits +0.020 against matched non-recruits +0.008. The matched controls
   cover 64 of 189 recruits and the difference is about 1.3 standard errors (section 4.7 item 2).

[adir]: https://github.com/sunmoonron/wdd-descendants/tree/main/docs/atlas
[a01]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/01_method.md
[a02]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/02_readability_law.md
[a03]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/03_increments_depth_targets.md
[a04]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/04_attention.md
[a05]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/05_cancellation_contraction.md
[a06]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/06_descendants_fate_transport.md
[a07]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/07_descendants_function.md
[a08]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/08_causal_coordinate.md
[a09]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/09_quotient_program.md
[a10]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/10_sinks_huge_directions.md
[a11]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/11_toys_synthetic.md
[a12]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/12_circuits_coselection_drift.md
[a13]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/13_readers_readouts.md
[a14]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/14_native_vocabulary.md
[a15]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/15_workspace_lenses.md
[a16]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/16_vision_round.md
[a17]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/17_forensic_instrument.md
[a18]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/18_recursion_composition.md
[a19]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/19_workspace_reader.md
[a20]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/20_classical_calibration.md
[a21]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/21_sae_bridge.md
[a22]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/22_model_tested.md
[a23]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/23_training_sweep.md
[a26]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/26_drift_cause.md
[a27]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/27_native_axes.md
[a29]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/29_exits.md
[a30]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/30_tails.md
[a31]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/31_coalitions.md
[a32]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/32_necessity.md
[a34]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/34_map.md
[a35]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/35_mechanism_map.md
[a24]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/24_selection_counterfactual.md
[a25]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/25_training_motion.md
[a33]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/33_second_model.md
[a36]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/36_object.md
[a37]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/37_instrument_placed.md
[a38]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/38_born.md
[a39]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/39_migration.md
[a40]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/40_turn_and_paper.md
[a41]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/41_six_directions.md
[a54]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/54_class_data_attribution.md
[a55]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/55_forecast_vs_features_planted_row.md
[a56]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/56_lottery_or_recruitment.md
[a62]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/62_already_clean_readers.md
[a63]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/63_already_clean_growth.md
[a64]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/64_crowd_read.md
[a69]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/69_brake_7b_fisher.md
[a42]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/42_audit_users.md
[a43]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/43_audit_baselines.md
[a44]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/44_refusal_direction_rescue.md
[a45]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/45_jlens_connection.md
[a46]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/46_auditor_sota.md
[a47]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/47_heldout_test.md
[a48]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/48_stats_controls_perword.md
[a49]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/49_parameter_side.md
[a50]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/50_attribute_catalogue.md
[a51]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/51_edit_detection_free_text.md
[a52]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/52_crosslingual_pinning.md
[a53]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/53_pinning_checks.md
[a57]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/57_calendar_rings.md
[a58]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/58_name_token_readout.md
[a59]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/59_incontext_classes.md
[a60]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/60_repeated_blocks.md
[a61]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/61_attribute_writers.md
[a65]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/65_brake_bypass.md
[a66]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/66_field_dictionary.md
[a67]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/67_gated_strong_field.md
[a68]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/68_reads_defends.md
[a70]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/70_dictionary_vs_sae.md
[a71]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/71_robustness_round.md
[a72]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/72_matched_budget.md
[a73]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/73_corrected_readout.md
[a74]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/74_zoo_through_wdd.md
[a75]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/75_dose_scaled_predictor.md
[a76]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/76_vocabulary_finder.md
[a77]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/77_switches_from_weights.md
[a78]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/78_persona_loop.md
[a79]: https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/79_priced_atoms.md
