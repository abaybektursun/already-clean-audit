# 04 — Literature audit for idea cards N1 to N13

Literature agent, 2026-11-05. No model was run. This file says, for each card, whether the specific claim
has already been measured, and by whom, under the field's own terms. N14 and N15 are skipped as asked.
N16 to N19 were added to `03_IDEAS.md` while this audit ran; they are not audited here, but N5 and N13
below cover most of N16 and N17.

## Verdicts

| card | verdict | closest prior work | the part still open |
|---|---|---|---|
| N1 input first, readers later | PARTLY | wdd-descendants e684 and e589 (GitHub, Pythia-160m and 410m) | Gate and up rows against the down column at a dated birth, at 7B, against non-birth drift. Prior evidence points the other way on the output side. |
| N2 silent or used | PARTLY | Xun, "Lagged Coupling", arXiv 2609.01048 (Pythia 160M to 12B) | A single MLP pole's activation-times-gradient and zero-ablation use at its own dated birth. |
| N3 route at birth | PARTLY | Eliwat, recgen-checkpoints (Pythia-410m layer 12); this repo's unite Arm 1 on OLMo-3 7B | Top-k upstream neurons and heads per birth, with each contributor's own age, at 7B, over 16 points. |
| N4 documents that made the job | PARTLY | Chen, Luo, Pan, "Mechanistic Data Attribution", arXiv 2601.21996 (Pythia 14M to 160M, attention heads) | One MLP neuron's job, at 7B, from a released data order. |
| N5 move a newborn skill with k neurons | PARTLY | Panigrahi et al. 2023, arXiv 2302.06600 (fine-tuning); wdd-descendants e577/e578 (adjacent Pythia-410m checkpoints, negative) | A benchmark skill moved between adjacent pretraining checkpoints by k neurons at 7B. |
| N6 edit the young | OPEN | Nearest: Ma et al., arXiv 2402.05827 (popular facts are harder to edit robustly) | Everything: no study conditions edit collateral on a unit's training age. |
| N7 state neurons by invariance | PARTLY | Ge et al., arXiv 2509.17196 (Pythia-6.9B crosscoder: previous-token features before induction features) | A content-substitution test that sorts single neurons into state and content, and the birth order of the two sorts. |
| N8 workspace over training | OPEN (one line) | Gurnee et al., arXiv 2607.15495, finished models only | Any across-checkpoint use of the Jacobian lens. |
| N9 rewire one robot instruction | PARTLY | Swann et al., arXiv 2610.20750 (activation patching in MolmoAct2, pi0.5, MolmoBot) | A weight edit that makes one instruction execute another, scored in closed loop. |
| N10 drop what nobody reads | DONE | Thrash, Chen, Kolouri, arXiv 2609.35579 (sensitivity-weighted residual pruning beats SliceGPT) | Only the specific band as an object; the general claim is published. |
| N11 explain at birth | OPEN | Gao, Zhang, Chen, neuron-evolution blog (explanations at 40 Pythia-160M checkpoints, no transfer scored) | Everything: no study scores an early-checkpoint explanation on the final model. |
| N12 gradient starvation measured | PARTLY | Li and Li, "Pre-carved Niches", arXiv 2609.01170 (Pythia-410M, every step) | Per-pole gradient on one job's contexts, for all poles of a layer, across a dated birth at 7B. |
| N13 what RL wrote | PARTLY | Mukherjee et al., arXiv 2505.11711, contested as a bf16 effect by arXiv 2511.08567 and 2509.04259 | Base, SFT, DPO, RL of one model, in the neuron basis, with a transplant of the k most-changed neurons and a matched random set. |

## Corrections to the cards' "nearest prior" lines

1. **arXiv 2610.21025 is not a crosscoder over Pythia checkpoints of model activations.** Its title is
   "Learning to Read Out: Unembedding Dynamics in Language Model Pretraining" (He, Shen, Iacob,
   Jovanovic, Qiu, Lane; 14 Oct 2026). It fits one sparse dictionary to every token's unembedding row at
   every checkpoint ("parameter-trajectory crosscoding") on Pythia-160M, 1B, 6.9B and OLMo-2-7B, 32
   checkpoints each. It contains no MLP-neuron analysis. Its main result bears on N1 and N2: hidden states
   encode task information before the readout expresses it. https://arxiv.org/abs/2610.21025 (TEXT; also
   `studies/already-clean/checks/literature_W5.md` §4a).
2. **RelP is arXiv 2508.21258**, not 2601.22594. RelP: Rezaei Jafari, Eberle, Khakzar, Nanda, 28 Aug 2025,
   https://arxiv.org/abs/2508.21258. arXiv 2601.22594 is Arora, Wu, Steinhardt, Schwettmann, "Language
   Model Circuits Are Sparse in the Neuron Basis", 30 Jan 2026, https://arxiv.org/abs/2601.22594.
3. **"RL fine-tunes small subnetworks" is disputed.** Two later papers attribute most of the measured
   sparsity to bf16 rounding (see N13).
4. **The record's "write column 0.59, read row 0.18" transplant figure for wdd-descendants** (in
   `studies/unite/research_notes/Recent independent AI research projects/checkpoint_priority_check.md`) was
   not found in that repository's README this session. The transplant numbers that were found are under N5.

## Read levels

- **TEXT**: the arXiv HTML (or PDF) was downloaded with curl into
  `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/lit_in/n04/`,
  converted to text, and the passages for the specific claim were read in that text. Numbers and quotes
  below come from that text. Sections not tied to the claim were searched by keyword, not read line by
  line.
- **REPO-DOC**: a GitHub README or documentation page read directly. Not peer-reviewed. Numbers not
  recomputed by me.
- **RECORD**: read at source in an earlier record of this repository (file named); not re-read here.
- **ABSTRACT**: abstract only.
- **SEARCH**: a search-engine summary only. Not quotable.

On wdd-descendants: it is one author's GitHub program (GitHub user sunmoonron, profile name BCR; the record
`checkpoint_priority_check.md` names the author as Ronish Bhatt, not verified here; https://github.com/sunmoonron/wdd-descendants,
created 2026-09-21, last push 2026-10-31, MIT license, no paper). It runs on Pythia-160m, Pythia-410m and
OLMo-1B checkpoints, and it touches N1, N2, N3, N4, N5, N6 and N12. Its "write row" or "write column" is
an MLP neuron's output vector (the down-projection column). Its "word" is a neuron whose output vector is
used, by orthogonal matching pursuit, at a set of contexts it calls the neuron's "class". Its numbers are
quoted from its atlas pages, which I read; I did not rerun anything.

---

## N1 — Input first, readers later

**Claim checked.** At a job's birth the gate and up rows move first, the down column barely turns, and the
readers of the down column specialize afterwards.

**Search phrases used.**
1. `neuron input weights versus output weights which changes first during training "feature learning" language model checkpoints MLP up_proj down_proj drift` (WebSearch; nothing relevant)
2. `"early alignment" OR "silent alignment" first-layer weights align direction before norm growth two-layer network small initialization output weights grow later` (WebSearch)
3. `per-neuron weight trajectory across Pythia checkpoints cosine similarity of neuron weight vectors between consecutive checkpoints "neurons" stabilize early later layers` (WebSearch)
4. `feed-forward key-value memories training dynamics keys learned before values MLP input weights output weights learning order transformer "keys" "values" over training` (WebSearch)
5. `"downstream" layers adapt after upstream feature forms "readout lags" representation learned before readout uses it training dynamics probe vs unembedding Pythia neurons readers` (WebSearch)
6. arXiv API `abs:"weight dynamics" AND abs:neurons AND abs:checkpoints` (0 results)
7. arXiv API `abs:"up_proj" AND abs:"down_proj" AND abs:pretraining` (0 results)
8. arXiv API `abs:"incoming weights" AND abs:"outgoing weights" AND abs:training` (1 result, unrelated)
9. wdd-descendants README and atlas pages, keyword search for `reader`, `recruit`, `transplant`, `read row`

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| wdd-descendants, e684, https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/63_already_clean_growth.md | Oct 2026 | Pythia-160m checkpoints (the e684 script loads `pythia160`) | Downstream readers of a newly recruited output vector, before and after recruitment, against norm-matched rows at the same positions | Readers gain share in 0.75 to 0.79 of words (ten strongest readers) and 0.75 to 0.81 (reader mass); readers align with the row more than with the class direction in 0.91 to 0.97; "no reader ever depends on the row at a fifth of its own spread" | REPO-DOC |
| wdd-descendants, e589, https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/40_turn_and_paper.md | Oct 2026 | Pythia-410m | Gradient of the output vector two checkpoints before recruitment, against its class direction | Mean cosine +0.020 (3.8 s.e.); matched non-recruits +0.008, random rows +0.002, class-shuffled −0.002. The class's positions supply 0.215 of the row's gradient at 0.006 of positions | REPO-DOC |
| wdd-descendants, e632 and e666, https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/56_lottery_or_recruitment.md | Oct 2026 | Pythia (three seeds) | Whether the future output vector is already aligned at step 0 | Median normalized rank 0.15 to 0.21 at step 0 (random row 0.51); "the recruited rows rotate from their step-0 direction as far as other rows do" | REPO-DOC |
| Bietti, Cabannes, Bouchacourt, Jegou, Bottou, "Birth of a Transformer: A Memory Viewpoint", https://arxiv.org/abs/2306.00802 | 1 Jun 2023 | Two-layer toy transformer on synthetic bigram and induction data | Order in which weight matrices learn, by freezing layers | "training key-query matrices does not lead to any learning unless W_O^2 is learned first". Order is top-down: output side first | TEXT |
| He et al., "Learning to Read Out", https://arxiv.org/abs/2610.21025 | 14 Oct 2026 | Pythia-160M, 1B, 6.9B; OLMo-2-7B | When the unembedding (a reader) reorganizes, against when hidden states encode the variable | Readout reorganizes 0.4 to 1.1% into Pythia training; hidden-state probes decode task information the outputs do not yet show; raising the readout learning rate closes the gap sooner | TEXT |
| Sun et al., "Weight Patching", https://arxiv.org/abs/2604.13694 | 15 Apr 2026 | Llama-3.2-3B, 3.1-8B, 2-13B, base and instruct | Which SwiGLU branch carries a gained behaviour when patched from instruct into base | "down-only localization is markedly weaker; restoration is strongest when gate and up are combined, with up alone outperforming gate alone". Values in Fig. 10 only | TEXT |

Also relevant, not read beyond a search summary: early-alignment theory for two-layer ReLU nets,
https://arxiv.org/abs/2307.12851, and the silent alignment effect, https://arxiv.org/abs/2111.00034 (SEARCH).

**Verdict: PARTLY.** Measured: that readers redistribute toward a recruited output vector after recruitment
(Pythia-160m), and that the output vector is the thing that turns toward its class, with the gradient
visible two checkpoints ahead (Pythia-410m). In a toy transformer the output side learns first. In
fine-tuning the gained behaviour sits in gate and up more than down. Not measured: the order of change of
gate rows, up rows and the down column at a dated birth in a SwiGLU 7B model, against the same pole's
non-birth drift. Two prior results point against the card's middle clause ("the output column barely
turns"): wdd-descendants' recruited rows rotate as far as other rows, and Bietti's order is output first. The
fine-tuning result (2604.13694) points for it. The prediction can land either way, so the card has room to
fail.

**Reuse.** wdd-descendants e684 reader statistic (ten strongest readers' share, reader mass, against norm-matched
rows at the same positions) is a ready control design. Its e589 class-shuffled and matched non-recruit
controls fit the card's baseline. Scripts are MIT-licensed: `scripts/e684_readers.py`, `scripts/e589_gradient_turn.py`.

---

## N2 — Silent or used

**Claim checked.** A pole's use (activation times gradient, and zero-ablation loss change on its job's
contexts) is above chance at its birth point; there is no silent period.

**Search phrases used.**
1. `representation decodable before it is causally used over training checkpoints "causal" "probe" emergence language model dormant features used later` (WebSearch)
2. `attribution patching across training checkpoints neuron importance at emergence "activation times gradient" developmental language model when does a feature start affecting loss` (WebSearch)
3. `"downstream" layers adapt after upstream feature forms "readout lags" ...` (WebSearch, shared with N1)
4. `Timaeus developmental interpretability 2026 local learning coefficient language model stages Pythia OLMo susceptibilities "per-token" refined LLC` (WebSearch)
5. arXiv API `ti:"Stagewise Development" AND ti:Transformers`; `ti:"Refined Local Learning Coefficient"`
6. arXiv API `abs:"local learning coefficient" AND abs:Pythia` (2 results); `abs:"local learning coefficient" AND abs:OLMo` (0)
7. devinterp.com publications page, read for 2025 to 2026 entries

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Xun, "Lagged Coupling: Internal Representations Become Readable Before They Become Causal", https://arxiv.org/abs/2609.01048 | 1 Sep 2026 | Pythia 160M to 12B, 8 checkpoints (1k to 143k), 4 task families; OLMo-2 replication at two sizes | Probe readability, behavioural readability, and the effect of adding the unit-norm probe direction at the probed site, against random directions of the same norm | Probe AUROC ≥ 0.990 from the first checkpoint; the steering effect is indistinguishable from random directions in 43 of 48 model-checkpoint cells; ordering read-before-write in 11 of 11 units | TEXT (PDF) |
| wdd-descendants, e591, https://github.com/sunmoonron/wdd-descendants (README, "Six directions beyond the descendants") | Oct 2026 | Pythia-410m, OLMo-1B | Whether recruitment of a neuron for a class changes loss at the class; cost of removing the class direction before and after recruitment | Class loss drop across recruitment −0.1231 nats against −0.1380 for token-matched positions, difference −0.0102 [−0.0292, +0.0116]. Removing the direction costs 0.0047 nats two checkpoints before recruitment and 0.0109 two after (random direction 0.0002): "the direction is load-bearing before the row arrives" | REPO-DOC |
| Aljaafari, Carvalho, Freitas, "Emergence and Localisation of Semantic Role Circuits in LLMs", https://arxiv.org/abs/2511.20910 | 25 Nov 2025 | Pythia 14M, 410M, 1B | EAP-IG circuits at each checkpoint; structure against faithfulness | "Roles become indispensable within 0–512 steps"; "Circuits stabilise structurally many steps before becoming functionally indispensable" | TEXT |
| He et al., https://arxiv.org/abs/2610.21025 | 14 Oct 2026 | Pythia, OLMo-2-7B | Encoded against expressed | See N1 | TEXT |
| Li and Li, "Pre-carved Niches", https://arxiv.org/abs/2609.01170 | 1 Sep 2026 | Pythia-410M trained from scratch, bf16 and fp32 | Attribution patching at every step for 14 tasks, with first-order loss decomposition | See N12 | TEXT |
| Ji, Wang, Sun, "Bridging Singular Learning Theory and Mechanistic Interpretability at the Billion-Parameter Scale", https://arxiv.org/abs/2611.00640 | 31 Oct 2026 | Pythia-1B, all 154 checkpoints, Dyck bracket circuit, 144 components (heads and MLP layers) | Fisher trace and zero-ablation KL per component over training | Changes of the Fisher trace track changes of causal effect, median partial rank correlation 0.51 | ABSTRACT plus keyword search of TEXT |

Developmental interpretability (Timaeus and coauthors), checked as asked. Hoogland, Wang,
Farrugia-Roberts, Carroll, Wei, Murfet, "Loss Landscape Degeneracy and Stagewise Development in
Transformers", https://arxiv.org/abs/2402.02364 (4 Feb 2024, ABSTRACT): the local learning coefficient
(LLC) divides training of a small attention-only language model and an in-context linear regression
transformer into stages that coincide with changes in internal structure. Wang, Hoogland, van Wingerden,
Furman, Murfet, https://arxiv.org/abs/2410.02984 (3 Oct 2024, ABSTRACT plus keyword search): refined LLCs
per attention head in a two-layer attention-only model with no MLP layers. "Embryology of a Language
Model", https://arxiv.org/abs/2508.00331, and "Towards Spectroscopy: Susceptibility Clusters in Language
Models", https://arxiv.org/abs/2601.12703 (Pythia-14M, 510 clusters, 50% match SAE features; ABSTRACT)
extend this with susceptibilities. The largest-scale LLC-against-mechanism test found is Pythia-1B
(2611.00640, table above). None of these works at the level of a single MLP neuron, and none uses OLMo.

**Verdict: PARTLY.** "Readable before used" is published for probe directions (steering, Pythia), for
task circuits (EAP-IG, Pythia), and for the unembedding (Pythia, OLMo-2). wdd-descendants shows, at the neuron level
on Pythia-410m, that the class direction already matters before the neuron is assigned to it. Not
measured: one MLP pole's own activation-times-gradient and zero-ablation effect on its job's contexts, at
the pole's dated birth, against random poles, at 7B. The prior work leans toward a lag (2609.01048) or
toward use before assignment (wdd-descendants e591), so both outcomes of N2 are live.

**Two cautions from the prior work.** (1) 2609.01048 adds a unit-norm direction, which it calls "the
weakest and most transparent member of the steering family"; its negative may be a dose effect. N2 uses
natural activation and zero-ablation, so it does not inherit this. (2) This repo's neuronoff study found
single-pole removal effects close to the noise, and its positive control failed on 65 of 156 Q4B poles
and 75 of 202 G4 poles (`studies/neuronoff/CLOSEOUT.md`). N2's zero-ablation arm needs a noise check
before any threshold is frozen.

**Reuse.** 2609.01048's three-track design (internal readability, behavioural readability, causal effect,
each against a stratified control) and its content-hash gate on checkpoints. 2511.20910's faithfulness
definition per checkpoint. wdd-descendants' token-matched and difficulty-matched loss controls (e591c).

---

## N3 — The route at birth in the neuron basis

**Claim checked.** At birth, the top upstream neurons and heads into a pole on its job's contexts are older
than the job and related to it.

**Search phrases used.**
1. `attribution graph across training checkpoints circuit formation OLMo 7B neurons "over training" circuit tracing transcoders checkpoints how circuits form pretraining 2026` (WebSearch)
2. `new feature built from existing components during pretraining upstream features older than downstream feature "compositional" emergence crosscoder checkpoints Ge "Evolution of concepts"` (WebSearch)
3. arXiv API `ti:"Relevance Patching" OR abs:RelP`
4. arXiv API `abs:"neuron basis" AND abs:circuit` (found 2601.22594)
5. arXiv API `abs:"attribution graph" AND abs:checkpoints` (3 results, none relevant)
6. arXiv API `abs:transcoder AND abs:"training dynamics"` (0); `abs:"circuit formation" AND abs:pretraining AND abs:neurons` (0)
7. `per-neuron weight trajectory across Pythia checkpoints ...` (WebSearch, shared with N1; found SAE-Track and feature life-history papers)

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| This repo, unite Arm 1, `studies/unite/results/RESULTS.md` | 2026-10-31, status "Not closed: awaiting review by a different model" | OLMo-3 7B, 841 late-born jobs, 251 neurons | Overlap T between the upstream input contributions of a new job and of the pole's older job (the record's Arm 1: does the new job arrive through inputs the neuron already uses), at birth, against a planted packed pole and a split-job ceiling | Median T 0.356 [0.307, 0.384] against planted 0.159 [0.105, 0.216]; ceiling 0.805 | RECORD (read this session) |
| Eliwat, recgen-checkpoints, https://github.com/Meliwat/recgen-checkpoints | 2026-10-18 to 10-20 | Pythia-410m layer 12, 14 checkpoints, 40 neurons | Top-100 upstream drivers by direct term during a facet's formation window | v4: 0 of 68 unrelated pairs adjacent at formation (3.1 expected); 17 of 18 related pairs adjacent. Heads carry a median 0.77 of a facet's direct rise, MLP neurons 0.25 | RECORD (`checkpoint_priority_check.md`, data inspected there) |
| Arora, Wu, Steinhardt, Schwettmann, "Language Model Circuits Are Sparse in the Neuron Basis", https://arxiv.org/abs/2601.22594 | 30 Jan 2026 | Llama 3.1 8B (Gemma-2 2B and 9B in the appendix), finished models | Gradient-based attribution circuits over raw MLP neurons, no transcoder | A circuit of about 10^2 MLP neurons controls subject-verb agreement; small neuron sets encode steps of a multi-hop city-state-capital task | TEXT |
| wdd-descendants, e685, README "The already-clean debate tested" | Oct 2026 | 12 model-blocks across several models | How many upstream contributors a mid-network neuron has at its strongest firings | 1,874 to 15,033 effective upstream contributors; the 16 largest carry 0.02 to 0.05 of the energy; halving the ablation damage needs 16 inputs at best and more than 1,024 in 5 of 12 model-blocks | REPO-DOC |
| Ge et al., "Evolution of Concepts in Language Model Pre-Training", https://arxiv.org/abs/2509.17196 | 21 Sep 2025 | Pythia-160M, 6.9B, crosscoders over snapshots | Emergence time of feature types and feature attribution to task metrics over snapshots | Previous-token features appear around steps 1,000 to 5,000; induction and context-sensitive features mostly 10,000 to 100,000; complexity against peak emergence r = 0.309 | TEXT |
| Tigges, Hanna, Yu, Biderman, "LLM Circuit Analyses Are Consistent Across Training and Scale", https://arxiv.org/abs/2407.10827 | 15 Jul 2024 | Pythia 70M to 2.8B | Head-level circuits across 300B tokens | Components change while the algorithm holds; larger models have more stable node sets (Jaccard, figure only) | TEXT (abstract and the §4 node-stability passage) |
| Gao and Gao, "A Circuit for In-Context Learning", https://arxiv.org/abs/2611.05344 | 3 Nov 2026 | Qwen3-4B-Base, SmolLM3-3B, Llama-3.2-3B; OLMo-2 1B and 7B checkpoints | Order in which parts of a head-level ICL circuit appear | Copying a repeated label appears first, withholding a label on unique-label prompts second, the readers' class selectivity last (paraphrase); the 1B model never develops the readers | TEXT (App. F) |

RelP (https://arxiv.org/abs/2508.21258, ABSTRACT): for MLP outputs in GPT-2 Large, attribution patching
reaches Pearson 0.006 with activation patching, RelP 0.956. This is a method, measured on finished models.

**Verdict: PARTLY.** Neuron-basis attribution graphs exist on finished 8B models (2601.22594). Upstream
drivers at the moment a facet forms exist for one layer of Pythia-410m (Eliwat). This repo has the input
overlap version on OLMo-3 7B (unite Arm 1), not yet closed. Not found: a per-birth top-k route of neurons
and heads, with each contributor's own birth date, over training at 7B.

**What the prior work predicts against the card.** (1) Eliwat finds heads, not MLP neurons, carry most of
a facet's direct input rise on Pythia-410m (0.77 against 0.25). A route that lists only neurons may miss
most of the input. (2) wdd-descendants e685 says the 16 largest inputs carry 0.02 to 0.05 of the energy; a top-10
route may explain little of the pole. The card's refutation rule should be stated in the share of the
pole's pre-activation that the top-k explains, with the share frozen on control poles.

**Reuse.** The unite J4 banks (`B/j4{a,b,c}/1/out/`, Grams per pole and checkpoint) already hold the input
side. Arora et al.'s neuron-basis pipeline and its local replacement rules for Llama 3 nonlinearities.
Eliwat's `classification_v4.json` (no license file; check before reuse).

---

## N4 — The documents that made the job

**Claim checked.** Documents in the birth interval that contain the job's tokens carry a gradient on the
pole's input rows aligned with the realized change; at small scale, removing them prevents the birth.

**Search phrases used.**
1. `training data attribution for an SAE feature or neuron "which training examples" caused feature to form "feature-level" data attribution language model` (WebSearch)
2. `OLMo data order released batch indices "training batches" reproduce exact documents seen between checkpoints OLMo 2 OLMo 3 data loader order` (WebSearch)
3. `per-example gradient restricted to one neuron's weights alignment with actual weight update between checkpoints TracIn neuron level "which documents" taught neuron pretraining` (WebSearch)
4. `OLMoTrace verbatim span tracing training data; "data order" OLMo 3 Dolma 3 batch order released reproduce training batches per step` (WebSearch)
5. arXiv API `abs:"data attribution" AND abs:neuron AND abs:pretraining` (0)
6. arXiv API `abs:"influence functions" AND abs:"induction heads"` (1: 2601.21996)
7. arXiv API `ti:"Stagewise Data Attribution"` (2510.12071)
8. devinterp.com publications page

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Chen, Luo, Pan, "Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units", https://arxiv.org/abs/2601.21996 | 29 Jan 2026 | Pythia 14M, 31M, 70M, 160M | EK-FAC influence of training samples on a unit's metric (induction and previous-token heads), inside the unit's emergence window; then counterfactual retraining by deleting (gradient masking) or duplicating the top samples | Deletion "consistently suppresses or delays head emergence, whereas random exclusion yields negligible impact"; augmentation (at most 10% of samples) accelerates it; "models under both augmentation and deletion regimes eventually converge to comparable saturation scores". Top 10% of samples carry up to 50% of cumulative influence (Pythia-14M). An MLP-neuron unit is defined (parameters W_in[:, i] and W_out[i, :]) but no neuron experiment is reported; SAE features are qualitative only | TEXT |
| wdd-descendants, e628, https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/54_class_data_attribution.md | Oct 2026 | Pythia-160m, continuation from step 8000 | Remove the windows that carry a group's class contexts and see whether its rows are still recruited | Underpowered, no specific effect: group A 0.28 with the full stream, 0.20 with its contexts removed, 0.20 with as many random windows removed; group B 0.20, 0.32, 0.32; s.e. 0.09 at 25 rows per group; the removal takes a third of the stream | REPO-DOC |
| wdd-descendants, e525 to e530, https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/26_drift_cause.md | Oct 2026 | Pythia (block 12; size not named on the page) | Agreement of a row's loss gradient with its realized thousand-step update | Gradient agrees with itself at cosine 0.14, 0.11, 0.08 (steps 4000, 8000, 12000); the thousand-step update follows it at 0.012 to 0.013; Adam-whitened direction no better | REPO-DOC |
| Lee, Smith, Adam, Hoogland, "Influence Dynamics and Stagewise Data Attribution", https://arxiv.org/abs/2510.12071 | 14 Oct 2025 | Toy model; Pythia-14M | Bayesian influence functions at each checkpoint; token-class influence over training | Influence changes sign and peaks at developmental transitions (figures) | ABSTRACT plus keyword search of TEXT |

Data order. OLMo 1 released its batch order (OLMo paper, https://arxiv.org/abs/2402.00838; SEARCH only).
For OLMo 2 mid-training, a third party reconstructed the seed-42 order and checked it against the global
index file OLMo saved (https://arxiv.org/abs/2610.41732, TEXT, §3 and appendix). Whether OLMo-3 stage 1
releases its order is not settled here; that is for `00_ASSETS.md`. OLMoTrace
(https://arxiv.org/abs/2504.07096, SEARCH) is verbatim span matching, not attribution.

**Verdict: PARTLY.** Unit-level data attribution with a counterfactual retrain is published for attention
heads at 14M to 160M (2601.21996). Neuron-level data attribution is defined there but not run. wdd-descendants tried
a removal test for neuron recruitment on Pythia-160m and could not detect an effect. Not found: one MLP
neuron's job at 7B traced to the documents between two checkpoints.

**What the prior work predicts against the card.** (1) MDA's deletions delay emergence but the heads
reach the same final score. The card's "the job was due regardless" outcome is the published outcome for
heads. The refutation rule must say whether a delay counts as prevention. (2) wdd-descendants' 0.012 cosine between
a row's gradient and its thousand-step update means a per-document gradient cosine with the realized change
over a 1,000-step interval will be small. The card's chance band must come from documents without the
token, matched on count and gradient norm, not from zero.

**Reuse.** MDA's EK-FAC restricted to a unit's parameters, its emergence-window rule, and its random
deletion control (code linked from the paper). The 2610.41732 reconstruction of OLMo's seeded order.

---

## N5 — Move a newborn skill with k neurons

**Claim checked.** The k most-changed neurons between two adjacent pretraining checkpoints, copied into the
earlier one, install most of a skill's gain; a matched random set installs none.

**Search phrases used.**
1. `grafting parameters between pretraining checkpoints transfer emergent capability sparse subset weight difference consecutive checkpoints` (WebSearch)
2. `"checkpoint" "weight patching" OR "parameter transplant" capability emergence pretraining checkpoints OLMo Pythia localize where a new ability is stored in weights` (WebSearch)
3. `"emergent abilities" localized parameters "intermediate checkpoints" "task vector" pretraining delta transplant earlier checkpoint recovers accuracy` (WebSearch)
4. `swap weights from later pretraining checkpoint into earlier checkpoint which layers or neurons carry the improvement language model 2025 2026` (WebSearch)
5. arXiv API `abs:"skill localization" AND abs:checkpoints` (0)
6. arXiv API `abs:grafting AND abs:"pretraining checkpoints"` (0)
7. wdd-descendants atlas 36, keyword `transplant`

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Panigrahi, Saunshi, Zhao, Arora, "Task-Specific Skill Localization in Fine-tuned Language Models", https://arxiv.org/abs/2302.06600 | 13 Feb 2023 | RoBERTa-base on 13 tasks; GPT-2 small | A learned binary mask that grafts fine-tuned values onto the pre-trained model, no retraining | About 0.01% of parameters recover more than 95% of fine-tuned performance (RoBERTa); 0.05% for GPT-2 small. Regions chosen by largest movement \|θ_ft − θ_pre\| do worse than learned regions at low sparsity (Fig. 2). "fine-tuning with AdamW does not contain sparse grafted models with good performance" unless an ℓ1 penalty on movement is added | TEXT |
| wdd-descendants, e577 and e578, https://github.com/sunmoonron/wdd-descendants/blob/main/docs/atlas/36_object.md | Oct 2026 | Pythia-410m, block 12, step 8000 receiving parts from step 16000 | Transplant of a new neuron's output direction, the whole neuron, its 128 or 512 strongest upstream neurons, or the readout (blocks 13 to 23, final norm, unembedding), scored as the closure of the 0.2864-nat prediction gap at the class positions | Whole neuron closes −0.020 of the gap; direction alone −0.005; direction with 512 upstream neurons −0.066; random −0.005; readout alone −0.873. The new neuron becomes a used direction in the earlier model in 0.60 of cases (direction) and 0.93 (direction and upstream), but the predictions do not move | REPO-DOC |
| Baherwani, Chen, Qiu, Wilson, Izmailov, "Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns", https://arxiv.org/abs/2606.25010 | 23 Jun 2026 | Pythia 14M to 410M, several seeds | Activation patching of the attention patterns of the top 16 heads into the pre-emergence checkpoint | "patching learned attention patterns can elicit emergence in an earlier checkpoint" (numbers in figures) | TEXT |
| Nutter et al., "Pre-training interventions, ex post facto: Grafting model beliefs across checkpoints", https://arxiv.org/abs/2610.00767 | 30 Sep 2026 | Families up to 284B | A synthetic-document adapter trained on the base checkpoint, added to the post-trained model | Installs the target belief as strongly as training on the post-trained model, with "reality drift" and preference-coherence loss cut by more than half on average | ABSTRACT |
| Sun et al., "Weight Patching", https://arxiv.org/abs/2604.13694 | 15 Apr 2026 | Llama 3B to 13B | Top 5% of heads and neurons patched from instruct into base | "recovers a substantial portion of instruction-following performance" (Fig. 16 only) | TEXT |

**Verdict: PARTLY.** Grafting a tiny parameter subset is DONE for fine-tuning (2302.06600) and for
instruct over base (2604.13694). Across adjacent pretraining checkpoints, the only neuron-level transplant
found is the one in wdd-descendants on Pythia-410m, and it is negative for predictions: the whole neuron moves 2% of the gap,
the readout moves 87%. Attention-pattern patching does elicit capabilities early (2606.25010). Not found:
a benchmark skill moved by k neurons between adjacent checkpoints at 7B, with a matched random set.

**What the prior work predicts against the card.** Three published results lower the prior. (1) The
pretraining optimizer is AdamW, and Panigrahi found AdamW fine-tuning does not localize without ℓ1. (2)
Ranking by movement norm, which the card proposes, was the weaker selection rule in Panigrahi's Fig. 2. (3)
wdd-descendants and Baherwani both put the bottleneck outside the new neuron: in the readout, or in attention
patterns. The card should add a heads arm and a readout arm, so that "neurons carry it" is tested against
the two published alternatives.

**Reuse.** Panigrahi's mask optimization as the upper reference for any k
(https://github.com/abhishekpanigrahi1996/Skill-Localization-by-grafting). wdd-descendants' closure-of-the-gap
statistic and its random-neuron and readout arms (`scripts/e578_coalition_transplant.py`, MIT). This
repo's `studies/unite/LITERATURE_2026-10-31.md` M3 has the fine-tuning restoration literature at matched
mass (2605.20296, 2505.16703), which N16 and N17 also need.

---

## N6 — Edit the young

**Claim checked.** An equal-size edit on an early-born pole costs more wikitext perplexity than on a
late-born pole; age predicts collateral better than energy or column norm.

**Search phrases used.**
1. `knowledge editing side effects depend on when the fact was learned during pretraining checkpoints "editing" collateral damage fact frequency popularity ROME MEMIT ripple` (WebSearch)
2. `neuron ablation cost depends on neuron age or number of downstream readers "out-degree" connectivity predicts ablation damage language model MLP neurons` (WebSearch)
3. `knowledge editing difficulty across pretraining checkpoints OLMo "knowledge entropy" later checkpoints harder to edit more forgetting acquisition time of fact editability` (WebSearch)
4. `"contribution utility" outgoing weights age maturity threshold continual backpropagation units with fewer downstream connections cheaper to reset` (WebSearch)
5. arXiv API `abs:"knowledge editing" AND abs:checkpoints AND abs:pretraining` (0)
6. arXiv API `abs:"model editing" AND abs:neurons AND abs:"side effects" AND abs:frequency` (0)
7. The 2026-10-31 record's queries 8, 9, 31, 44 and 51 on reader gain against ablation cost (`studies/unite/LITERATURE_2026-10-31.md`)

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Ma et al., "On the Robustness of Editing Large Language Models", https://arxiv.org/abs/2402.05827 | 8 Feb 2024 | Editing with MEMIT and others | Editing robustness by knowledge popularity (frequency, connection, co-occurrence) | "Knowledge that is more popular is memorized better, easier to recall, and harder to edit robustly" | TEXT (§6 headings and findings) |
| Cohen et al., "Evaluating the Ripple Effects of Knowledge Editing", https://arxiv.org/abs/2307.12976 | 24 Jul 2023 | GPT-2, GPT-J, GPT-Neo, LLaMA, GPT-3 | RippleEdits by popular and random facts | Logical Generalization higher for Random facts; Preservation higher for Popular facts | TEXT |
| This repo, `studies/unite/LITERATURE_2026-10-31.md` M2 | 2026-10-31 | CNN, transcoder features, Llama-3-8B and others | Whether a reader statistic predicts single-unit ablation cost | PARTLY TAKEN there: Molchanov's Spearman with oracle loss change, weight 0.27, activation mean 0.56, Taylor 0.73 (VGG-16, https://arxiv.org/abs/1611.06440) | RECORD |
| wdd-descendants, e733, README "Priced atoms" | Oct 2026 | Pythia-410m, three layers | A sampled-Fisher price against the cost of removing a neuron's output vector | Spearman 0.88 to 0.95 over 40 rows, but the price is "nearly the activation energy" (0.94 to 0.97 over all 4,096 rows) | REPO-DOC |
| Kim et al., "Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition", https://arxiv.org/abs/2410.01380 | 2 Oct 2024 | OLMo 1B, 7B checkpoints | Knowledge entropy and acquisition by model checkpoint (the model's age, not a unit's) | Entropy declines through pretraining; later checkpoints acquire and retain less | ABSTRACT |

Continual backpropagation scores a unit by activation times the sum of its outgoing weights and protects
units below an age threshold (SEARCH only; primary paper not read). It uses age as an eligibility rule,
not as a predictor of damage.

**Verdict: OPEN.** No study found conditions edit or ablation collateral on a unit's training age. Fact
popularity has a published effect on editing; reader-weighted statistics against ablation cost are partly
published (record M2). wdd-descendants e684 (readers gain share after recruitment, under N1) supports the card's
premise that readers accumulate with age, weakly.

**What the prior work predicts against the card.** wdd-descendants e733 and Molchanov's table both say activation
energy carries most of the predictive power for removal cost. Age is likely correlated with energy. The
card's regression must show age adds to energy and norm, with a held-out set, or the result is energy
relabelled.

**Reuse.** RippleEdits (https://github.com/edenbiran/RippleEdits) for a fact-edit arm. Record M2's oracle
ranking design.

---

## N7 — Computation neurons against content neurons, by invariance

**Claim checked.** A measurable share of poles keep firing when content words are substituted and stop
when structure is substituted; these state jobs are born earlier than content jobs.

**Search phrases used.**
1. `syntactic or structural features emerge before semantic features during language model pretraining checkpoints probing "syntax before semantics" learning order Pythia` (WebSearch)
2. `neurons that track position, bracket depth, quotation state, list index or counting in language models "state neurons" "syntax neurons" invariant to content substitution` (WebSearch)
3. `"binding" mechanism emerges during pretraining checkpoints entity attribute binding ID when learned Pythia OLMo developmental study variable binding language model training` (WebSearch)
4. `"positional neurons" OR "position neurons" OR "content-independent" neurons language model counterfactual token substitution invariance classify neurons token-driven versus context-driven` (WebSearch)
5. arXiv API `ti:"How Syntax Specialization Emerges"` (2505.19548)
6. arXiv API `abs:"binding ID" AND abs:checkpoints` (0); `ti:"binding" AND ti:"training dynamics"` (0)
7. arXiv API `abs:"entity tracking" AND abs:"training"` (21; none a developmental study of binding)

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Ge et al., https://arxiv.org/abs/2509.17196 | 21 Sep 2025 | Pythia-6.9B crosscoder, 32,768 features | Emergence time by feature type, types found by rules | Previous-token features around steps 1,000 to 5,000; induction and context-sensitive features mostly 10,000 to 100,000 | TEXT |
| Bayazit, Mueller, Bosselut, "Crosscoding Through Time", https://arxiv.org/abs/2509.05291 | Sep 2025 | Pythia-1B, OLMo-1B, BLOOM-1B | Crosscoder features across checkpoints | Early checkpoints use token-specific detectors; later ones favour group-level features | SEARCH; RECORD (`LITERATURE_2026-10-31.md` M4) |
| Duan et al., "How Syntax Specialization Emerges in Language Models", https://arxiv.org/abs/2505.19548 | 26 May 2025 | GPT-2 and Pythia-160M checkpoints; Pythia-2.8B for ablation | A syntactic sensitivity index per layer and per neuron over training | Index near zero up to 2M tokens, then rises; ablating the top 5% high-index neurons (about 467 of 9,216 in GPT-2) raises perplexity on grammatical sentences more than random | TEXT (method and §5) |
| Wang, Baker, Gordon, Murfet, "Embryology of a Language Model", https://arxiv.org/abs/2508.00331 | 1 Aug 2025 | 3M two-layer attention-only model | Susceptibilities over training, UMAP | A "spacing fin" for counting spacing tokens begins to separate around 1,000 steps; stage LM1 ends at 900 steps | TEXT |
| Voita, Ferrando, Nalmpantis, "Neurons in LLMs: Dead, N-gram, Positional", https://arxiv.org/abs/2309.04827; Gurnee et al., "Universal Neurons in GPT2", https://arxiv.org/abs/2401.12181 | 2023, 2024 | OPT 125M to 66B; GPT-2 | Positional neurons by mutual information with position; position neurons in layers 0 to 2 | Finished models; no checkpoint analysis found in either text | TEXT (keyword search) |
| Feng and Steinhardt, "How do Language Models Bind Entities in Context?", https://arxiv.org/abs/2310.17191 | 26 Oct 2023 | Pythia and LLaMA families | Binding ID vectors | Finished models; no checkpoint analysis found in the text | TEXT (keyword search) |

**Verdict: PARTLY.** Developmental order of structural features before content features is published at
the feature level (previous-token before induction, Pythia-6.9B; token-specific before group-level,
Pythia-1B, OLMo-1B) and for syntax-sensitive units (GPT-2, Pythia-160M). Positional neurons and binding
IDs are documented on finished models. Not found: a content-substitution invariance test that sorts single
MLP neurons into state jobs and content jobs, and the birth-date comparison of the two sorts.

**What the prior work predicts.** The early-structural ordering is already reported, so "state jobs are
born earlier" is close to a replication at the feature level. The new part is the invariance sort on the
32 to 42 percent of poles the judge did not understand. The 5% floor in the refutation rule is likely to
pass; a better floor is the share among the one-job set, as a matched control.

**Reuse.** Ge et al.'s rule definitions for previous-token, induction and context-sensitive features
(App. D). Voita's mutual-information test for position as one extra invariance arm.

---

## N8 — The Jacobian lens subspace over training (one line)

No study applies the Jacobian lens (Gurnee et al., https://arxiv.org/abs/2607.15495, 16 Jul 2026) across
training checkpoints; the follow-ups found (2608.25347, 2610.31955, 2607.12792, 2609.33458,
2611.00015, and the chess RL paper 2607.20952, which fits the lens on one post-RL checkpoint only) all use
finished models. Searched: WebSearch `"Jacobian lens" language model 2607.15495 follow-up training
checkpoints workspace subspace over training`; arXiv API `abs:"Jacobian lens" OR abs:"J-lens"` (22
results, all titles scanned); arXiv API `abs:"Jacobian lens" AND (abs:checkpoints OR abs:"training dynamics"
OR abs:pretraining)` (0 results). Verdict: OPEN. The paper itself is for
`01_JLENS.md`.

---

## N9 — Rewire one robot instruction by a weight edit

**Claim checked.** In MolmoAct2-LIBERO, an edit of at most 256 output columns makes one instruction
execute another, with the rest of the suite unchanged.

**Search phrases used.**
1. `mechanistic interpretability vision-language-action model steering neurons language instruction OpenVLA pi0 "steering" FFN neurons robot behavior change` (WebSearch)
2. `VLA models ignore language instructions LIBERO instruction perturbation "language" sensitivity LIBERO-Plus LIBERO-PRO vision-language-action follow wrong instruction` (WebSearch)
3. `model editing vision-language-action policy weight edit change which task executed instruction "knowledge editing" robot policy VLA localize instruction grounding layers activation patching` (WebSearch)
4. arXiv API `abs:"vision-language-action" AND abs:"model editing"` (1: 2608.04692)
5. arXiv API `abs:"vision-language-action" AND abs:"activation patching"` (2: 2610.10178, 2610.20750)

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Swann, Papillon, Jacobs, Boppana, Haklay, Bigelow, Lee, Watts, Icard, Schwager, Lewis, Merullo, Lubana and others, "Causal Analysis of Robot Foundation Models Enables Efficiency Gains", https://arxiv.org/abs/2610.20750 | 14 Oct 2026 | pi0.5, **MolmoAct2**, MolmoBot | Over 775,000 activation-patching interventions on 13 object pairs in two layouts (DROID setup, 52 inputs), patching backbone positions and the action-expert cross-attention | "Early-to-mid layer representations of the target noun token determine which object is selected"; in MolmoAct2 "the model forms task-conditioned representations at the text positions that follow the instruction". In MolmoBot, patching action-expert layer 24 alone recovers over 50% of the instructed reach contrast, a three-layer window close to 70%; keeping 3 of 36 layers holds success 46.7% against 46.9% over 3,000 episodes | TEXT |
| Wang et al., "Suppression Sticks, Locality Is Fragile: A Closed-Loop Target-and-Control Audit of Task-Vector Negation in VLA Policies", https://arxiv.org/abs/2608.04692 | 5 Aug 2026 | MergeVLA-LIBERO-Goal (about 1.4B), pi0.5, OpenVLA-7B | Weight-space subtraction of one skill's task vector, scored in closed loop on target and control tasks | On all ten LIBERO-Goal skills: five separate cleanly, three resist, two collapse globally; suppressible targets stay at 0% success, but mean control retention is 52%, and every suppressing edit harms at least one unrelated control | ABSTRACT plus keyword search of TEXT |
| Jiang et al., "Encoded but Not in Control: Revealing the Grounding Gap in Vision-Language Robot Policies", https://arxiv.org/abs/2610.06235 | 5 Oct 2026 | pi0.5, GR00T N1.7, FastWAM, LingBot-VA | Target replacement in a fixed LIBERO-Object scene; linear probes; attention contribution | 86.5% to 98.3% of trials go to the original target, 0.5% to 13.0% to the newly instructed one; probes decode the instructed target; instruction tokens carry 25.6% to 27.9% of the measured contribution for pi0.5 on LIBERO | TEXT |
| Fei et al., "LIBERO-Plus", https://arxiv.org/abs/2510.13626 | 15 Oct 2025 | OpenVLA-OFT and others | Seven perturbation factors; blank instruction; replaced targets | With a blank instruction, OpenVLA-OFT on the Object suite is "largely unchanged"; with replaced targets success drops "nearly to zero" | TEXT |
| Häon, Stocking, Chuang, Tomlin, "Mechanistic interpretability for steering vision-language-action models", https://arxiv.org/abs/2509.00328 | 30 Aug 2025 | OpenVLA, pi0-FAST; LIBERO-Long; UR5 | FFN value vectors projected to tokens, clustered, upweighted at inference | Slow and fast clusters change end-effector displacement; "less than 25% of FFN neurons being rewired for action prediction" | TEXT |
| Wulff and Cangelosi, https://arxiv.org/abs/2610.10178 | 7 Oct 2026 | pi0.5, GR00T N1.7 on LIBERO | Activation and attribution patching under five instruction corruptions | Both models react strongly to empty and directional instructions, little to rephrasing | ABSTRACT |

**Verdict: PARTLY.** The card's first measurement, activation patching between two instructions to find
the site, is DONE on MolmoAct2 (on DROID scenes, not the LIBERO fine-tune) by 2610.20750. Weight-space
edits of a VLA scored in closed loop exist, by subtracting task vectors (2608.04692). Not found: a weight
edit that makes one instruction execute another.

**What the prior work predicts against the card.** LIBERO policies often follow the scene, not the
instruction. Where the scene fixes the task (Object, Spatial), there is little language signal to rewire.
LIBERO-Goal is the suite where the scene is shared across tasks, and it is also where task-vector
subtraction separated cleanly in 2608.04692. The first check before any edit is a blank-instruction and
swapped-instruction baseline on MolmoAct2-LIBERO, per suite. If swapped instructions already execute the
original task, the edit has nothing to act on.

**Reuse.** 2610.20750's counterfactual pairs design and its finding of where in MolmoAct2 the target noun
is read (early-to-mid layers, text positions after the instruction). LIBERO-Plus perturbations
(https://github.com/sylvestf/LIBERO-plus). 2608.04692's target-and-control retention metric, which is the
card's "rest of the suite unchanged" stated as a number.

---

## N10 — Drop what nobody reads

**Claim checked.** Directions with the lowest reader gain can be dropped from the KV cache or the residual
stream at a smaller perplexity cost than energy-ranked dropping.

**Search phrases used.**
1. `KV cache low-rank compression importance-aware projection "query-aware" channel pruning keys ThinK Palu Eigen Attention comparison rank by downstream sensitivity rather than variance` (WebSearch)
2. `residual stream dimension pruning hidden dimension reduction LLM rank by Fisher or gradient sensitivity versus PCA energy SliceGPT comparison perplexity at matched compression` (WebSearch)
3. `value cache compression aware of output projection W_O "value" channels pruned by downstream contribution KV cache low-rank perplexity Llama 8B 2025 2026` (WebSearch)
4. arXiv API `abs:"KV cache" AND abs:"low-rank" AND abs:"sensitivity"` (5 results)
5. arXiv API `abs:"Jacobian lens" OR abs:"J-lens"` (found 2611.00015, depth pruning)

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Thrash, Chen, Kolouri, "Output-aware Residual Stream Pruning for Large Language Models", https://arxiv.org/abs/2609.35579 | 28 Sep 2026 | Llama-3.2-3B, Llama-3.1-8B, Mistral-7B, Mistral-Nemo, Phi-3-Medium (instruct) | Residual subspace chosen from a sensitivity-weighted covariance (second-order KL) against SliceGPT's activation PCA, at 10 to 30% of the hidden dimension removed | Llama-3.1-8B-Instruct WikiText-2 perplexity, base 7.21: at 10%, SliceGPT 15.63 against 11.04; at 25%, 40.24 against 23.79; KL 1.6814 against 1.1569 at 25% | TEXT (Table 1) |
| Xu et al., "ThinK: Thinner Key Cache by Query-Driven Pruning", https://arxiv.org/abs/2407.21018 | 30 Jul 2024 | LLaMA-3-8B, Mistral | Key-cache channels pruned by query-key interaction rather than magnitude | Over 20% further KV memory reduction over eviction and quantization baselines; with KIVI 2.8× peak-memory reduction | ABSTRACT plus §3 |
| Lesens, Rakhshan, Rabusseau, "KQ-SVD", https://arxiv.org/abs/2512.05916 | 5 Dec 2025 | LLaMA, Mistral | Closed-form low-rank decomposition of the attention matrix itself, not of keys alone | "consistently delivers superior projection quality" | ABSTRACT |
| Chang et al., "Palu", https://arxiv.org/abs/2407.21118 | 30 Jul 2024 | Llama family | Low-rank KV projection with per-matrix rank chosen by Fisher information | 50% KV compression with "strong accuracy"; up to 1.19 lower perplexity than quantization-only | TEXT (rank-search section) |
| Devoto, Jeblick, Jégou, "Expected Attention", https://arxiv.org/abs/2510.00636; Benfenati et al., StiefAttention, https://arxiv.org/abs/2601.21686 | 2025, 2026 | Llama-3-8B and others | KV pairs scored by expected attention and output impact; projections fitted to decoder-layer output error | See abstracts | ABSTRACT |
| "Output-Aware Block Influence: Jacobian-Lens Weighting for Depth Pruning", https://arxiv.org/abs/2611.00015 | 30 Sep 2026 | Qwen3-8B, Llama-3.1-8B-IT, Gemma-3-12B-IT | Block pruning scores transported by the Jacobian lens | Up to 19 points higher post-pruning accuracy than Block Influence | ABSTRACT |

**Verdict: DONE.** "Rank directions by how much downstream computation reads them, not by energy, and
lose less" is measured for the residual stream on 3B to 14B models (2609.35579), and for KV channels and
projections by query-aware and output-aware methods (ThinK, KQ-SVD, Expected Attention, StiefAttention).
What remains is only whether this repo's band is a better-chosen subspace than a sensitivity-weighted
covariance. That is a narrow method comparison, with 2609.35579 as the baseline to beat.

**Reuse.** 2609.35579's evaluation table (WikiText-2 perplexity, KL, common-sense accuracy at 10, 20, 25,
30%) as the comparison format, and SliceGPT plus its method as the two baselines.

---

## N11 — Explain at birth

**Claim checked.** An explanation written from a pole's contexts at birth scores higher on the final
checkpoint's contexts than one written from the final contexts.

**Search phrases used.**
1. `neuron explanations generated at intermediate training checkpoints applied to final model automated interpretability explanation drift across checkpoints simulation score` (WebSearch)
2. `"neuron descriptions" OR "feature explanations" stability over training checkpoints explanation from earlier checkpoint scores on later checkpoint polysemanticity increases during training explainability` (WebSearch)
3. `"Pythia" neurons explanations every checkpoint LLM explainer "neuron evolution" dataset lineage labels recursive generalization superposition HeartyHaven` (WebSearch)
4. arXiv API `abs:"neuron explanations" AND abs:checkpoints` (0)
5. arXiv API `abs:"automated interpretability" AND abs:"training dynamics"` (0)
6. arXiv API `abs:"explanation" AND abs:"earlier checkpoint"` (1, unrelated: 2606.32038)

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Gao, Zhang, Chen, neuron-evolution, https://heartyhaven.github.io/blog/neuron-evolution/ | Oct 2026 | Pythia-160M, 40 checkpoints, 240 neurons | An agent explains each neuron's top contexts at every checkpoint; a second agent links them | No explanation from one checkpoint is scored on another; the two agents agree in about 40% of cases; "Superposition appeared more often early in training" | REPO-DOC (blog page read this session) |
| wdd-descendants, e687, atlas 63 | Oct 2026 | Pythia-160m | Recall of token, previous-token and bigram tables fitted to a neuron's class, on held-out positions | 0.00 to 0.01 of held-out positions covered at precision 0.5; 0.09 to 0.14 of positions have no training precedent | REPO-DOC |
| Xu, neuron-lineage, https://github.com/skydancerosel/neuron-lineage | 2026-10-27 | Pythia-410m, 1.4b, 11 checkpoints | New jobs and their nearest older job against random directions | Role-adjacent new jobs 10.4% (neurons) against 6.5% (random directions) on 410m | RECORD (`checkpoint_priority_check.md`) |

**Verdict: OPEN.** Explanations at many checkpoints exist (Gao et al.), but no study scores an early
explanation against the final model's activations. The card's lithium example (20 of 20 at step 60,000,
0 of 50 at the end) is this repo's own.

**What to watch.** A birth-point explanation is written from fewer, younger contexts. If the final
contexts drift, the birth explanation can score higher simply by being narrower and matching the old core.
The judge must score on held-out final contexts drawn the same way for both explanations, with recall and
precision both reported.

**Reuse.** Gao et al.'s `lineages.parquet` on Hugging Face
(https://huggingface.co/datasets/LangGao/pythia-160m-neuron-evolution) as a second model for the same test.

---

## N12 — Gradient starvation measured

**Claim checked.** Before a job is born in pole A, several poles receive gradient on the job's contexts;
after, the others fall toward chance while A rises.

**Search phrases used.**
1. `"winner-take-all" OR "rich-get-richer" neurons compete for a feature during training gradient concentrates on one neuron "incidental polysemanticity" language model measured` (WebSearch)
2. `gradient starvation coherent gradients empirical test large language model pretraining per-neuron gradient share before and after feature emergence` (WebSearch)
3. `attribution patching across training checkpoints neuron importance at emergence ...` (WebSearch, shared with N2; found 2609.01170)
4. arXiv API `abs:"gradient starvation" AND abs:"language model"` (4 results)
5. arXiv API `abs:"feature competition" AND abs:training AND abs:neurons` (0)
6. wdd-descendants README, keyword `gradient`, `substitute`, `recruit`

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Li and Li, "Pre-carved Niches: The Formation Dynamics of Modular Task Partitions in Early LLM Training", https://arxiv.org/abs/2609.01170 | 1 Sep 2026 | Pythia-410M trained from scratch, bf16 and fp32 runs, about the first 165 steps | Gradient supply (sum of per-neuron gradient norms) to the neuron sets of a winning task pair and a losing task, against random same-size sets | Winners receive 2.25 to 2.73 times the loser's gradient supply, 9.5 to 11.5 standard deviations from the random control; "effective updates and weight magnitudes show no corresponding asymmetry"; the loser's supply never falls below 0.59 of the random baseline, so no absolute starvation | TEXT |
| wdd-descendants, e589 and e590, atlas 40 | Oct 2026 | Pythia-410m | The future neuron's gradient share at its class before recruitment; substitutes after removing 30 recruited neurons at step 8000 | Class positions supply 0.215 of the recruit's gradient (matched non-recruits 0.086, random 0.039). After removal and 2.5M tokens, 0.17 of classes have a substitute (unfrozen 0.30) | REPO-DOC |
| Lecomte et al., "What Causes Polysemanticity?", https://arxiv.org/abs/2312.03096 | Dec 2023 | Toy model with L1 | Winner-take-all assignment of features to neurons from initialization | Theory and toy counts | ABSTRACT |
| Zhao and Du, "Syntax vs. Semantics: How Transformers Learn Deep Dependencies", https://arxiv.org/abs/2608.26139 | Jun 2026 | Toy transformers; Llama-3.1-8B, Qwen2.5-Coder-7B | A "Gradient Starvation" account of sparse semantic dependencies | Theory with validation; no per-neuron measurement found | ABSTRACT plus keyword search of TEXT |

Original references, not read: gradient starvation, https://arxiv.org/abs/2011.09468; coherent
gradients, https://arxiv.org/abs/2002.10657.

**Verdict: PARTLY.** Relative gradient deprivation between competing neuron sets is measured on
Pythia-410M in the first ~165 steps (2609.01170), and one neuron's gradient share before its own
recruitment is measured on Pythia-410m (wdd-descendants). Not found: the gradient to every pole of a layer on one
job's contexts, before and after a dated late birth, at 7B.

**What the prior work predicts against the card.** 2609.01170 found the gradient asymmetry did not reach
the Adam updates or the weights. N12 measures gradients only. A gradient that falls for the losers may
leave their updates unchanged under Adam, so the card should record the realized update to each pole as
well (the house rule: a gradient is not an update).

**Reuse.** 2609.01170's random same-size control for gradient supply and its update-ratio readout.

---

## N13 — What RL wrote, in the neuron basis

**Claim checked.** Across Base → SFT → DPO → RL, each stage changes a small share of poles; the stage's
gain transplants with the k most-changed neurons; RL's patch is the smallest and nearest the output.

**Search phrases used.**
1. `what does RLVR change in the weights 2026 neuron-level analysis reinforcement learning updates localized MLP neurons transplant RL delta` (WebSearch)
2. `"The Path Not Taken" RLVR learns off the principals weight update spectrum; OLMo 3 Think SFT DPO RLVR weight difference analysis neurons` (WebSearch)
3. `reinforcement learning post-training changes few neurons "neuron-level" analysis base vs RL model which neurons change reasoning LLM 2026 arXiv` (WebSearch)
4. `"o_proj" OR "output projection" RL reasoning gains localized "Who reasons in the large language models" transplant post-trained module into base model recovers reasoning` (WebSearch)
5. arXiv API `abs:"reinforcement learning" AND abs:"update sparsity"` (6 results)
6. arXiv API `abs:"RLVR" AND abs:neurons AND abs:weights` (0)

**Closest work.**

| work | date | model | what was measured | number | read level |
|---|---|---|---|---|---|
| Mukherjee, Yuan, Hakkani-Tur, Peng, "Reinforcement Learning Finetunes Small Subnetworks in Large Language Models", https://arxiv.org/abs/2505.11711 | 16 May 2025 | 10 models incl. Llama-3.1-Tulu-3-8B and 70B (SFT → DPO), DeepSeek-R1-Zero, Eurus-2-7B-PRIME | Share of bf16 parameters unchanged (tolerance 1e-5) after RL; fine-tuning only that subnetwork | 68.5% to 96.0% unchanged after RL (Tulu-3-8B DPO 81.4%, 70B 95.2%); SFT 6% to 15%. "nearly all parameter matrices receive similarly sparse updates"; updates "nearly full-rank" | TEXT |
| Zhu et al., "The Path Not Taken: RLVR Provably Learns Off the Principals", https://arxiv.org/abs/2511.08567 | 11 Nov 2025 | DS-R1-Distill-Qwen-1.5B and 7B, Qwen3-4B, 8B, 14B, 30B-A3B | bf16-aware sparsity; spectral drift; overlap with principal weights | RL sparsity 36% to 92%, SFT 0.6% to 18.8% (e.g., Qwen3-8B-Base → DAPO 79.7%); "bfloat16 precision further accentuates the apparent sparsity by attenuating micro-updates" | TEXT |
| Shenfeld, Pari, Agrawal, "RL's Razor", https://arxiv.org/abs/2509.04259 | 4 Sep 2025 | Qwen 2.5 3B-Instruct; robotic policies | Predictors of forgetting; sparsity check | Forward KL on the new task predicts forgetting (R² 0.96 on their MNIST task; weight-change L1 0.34). "Performing the same training with float32 resulted in models with identical performance but without any sparsity in their weight updates" | TEXT |
| Sun et al., "Weight Patching", https://arxiv.org/abs/2604.13694 | 15 Apr 2026 | Llama 3B to 13B, instruct into base | Top 5% heads and neurons patched in | "substantial portion" of IFEval recovered (figure only) | TEXT |
| Chen et al., "Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons", https://arxiv.org/abs/2406.14144 | 20 Jun 2024 | Llama2-7B, Mistral-7B, Gemma-7B, Qwen2.5-3B (per the record) | Activation patching of about 5% of neurons | Restores over 90% of safety performance | ABSTRACT; RECORD (`LITERATURE_2026-10-31.md` M3) |
| Lee et al., "A Mechanistic Understanding of Alignment Algorithms: DPO and Toxicity", https://arxiv.org/abs/2401.01967 | 3 Jan 2024 | GPT2-medium | Parameter change after DPO; toxic MLP vectors | Every parameter has cosine above 0.99 with the original; "toxic MLP vectors do not change from DPO"; the model learns a distributed offset that bypasses them | TEXT |
| Akgül et al., "Rethinking RL for LLM Reasoning: It's Sparse Policy Selection", https://arxiv.org/abs/2605.06241 | 7 May 2026 | Several families | Token positions RL changes; KL-LoRA distillation of the RL teacher | 1% to 3% of positions change; a rank-32 LoRA on QKVO distilled from 100 problems reproduces the teacher's accuracy; rank-8 output-projection adapter within a few points | TEXT |
| Zhang, Mukherjee, Peng, "Understanding LLM Parameter Update Sparsity through the Lens of Fisher", https://arxiv.org/abs/2609.36262 | 28 Sep 2026 | RL and on-policy distillation runs | Diagonal Fisher against where gradients concentrate; fixed sparse masks from initial Fisher | Fisher masks "retain a large proportion of the improvement from full training" | ABSTRACT |

Also: in a LoRA-trained Qwen3-14B chess model, the RL-induced weight change put 86.2% of its energy in the
MLP projections, 56.5% in gate_proj alone, against 13.8% in attention (https://arxiv.org/abs/2607.20952,
23 Jul 2026, TEXT; a LoRA delta, not full weights). OLMo-2-1B across 13 RLVR checkpoints, probes only: pass@1 58.0% to 61.3%, coverage at 256 samples
94.5% to 96.9% (Bangera and Gupta, https://arxiv.org/abs/2611.02574, 2 Nov 2026, TEXT).

**Verdict: PARTLY.** Update sparsity after RL is measured on many models and is disputed as mostly bf16
rounding. Neuron-level transplant of a post-training gain is measured for instruction tuning
(2604.13694) and, at the activation level, for DPO safety (2406.14144). Not found: one model's
Base → SFT → DPO → RL chain analysed in the neuron basis, with a top-k neuron transplant per stage and a
matched random set.

**What the prior work predicts against the card.** (1) If the OLMo-3 stage checkpoints are stored in bf16,
"most-changed neuron" rankings are partly rankings of which weights crossed a bf16 rounding step
(2511.08567). The ranking must be done on the float difference with the bf16 step size as the floor, or on
fp32 weights if released. (2) Mukherjee reports that "nearly all parameter matrices receive similarly sparse
updates": the set of changed weights is not confined to some layers. That is evidence against the card's
"most concentrated near the output" when concentration is counted in changed weights; concentration of
update magnitude near the output was not tested there. (3) DPO in GPT2-medium changes every parameter a little and leaves the toxic vectors in place;
the DPO stage may change readers and gates, not jobs, which the card can test directly.

**Reuse.** The bf16-aware unchanged-weight definition (2511.08567, Def. 2.1). Mukherjee's subnetwork
fine-tuning as a second test of locality. 2605.06241's 1% to 3% of positions as a way to choose which
tokens to score a transplant on.

---

## What could not be read, and limits

- Lecomte et al. 2312.03096, Gradient Starvation 2011.09468, Coherent Gradients 2002.10657, the early
  alignment papers 2307.12851 and 2111.00034, continual backpropagation, the OLMo 1 data-order statement
  (2402.00838) and OLMoTrace (2504.07096): not read beyond a search summary or abstract, as marked.
- 2606.25010, 2604.13694, 2601.21996 and 2509.17196 put most of their numbers in figures. I quote only
  sentences from the text.
- wdd-descendants is a large one-author program (README about 400k characters). I read six atlas pages and
  the README rows for the cited experiments. Its claims are not peer-reviewed and I did not rerun them.
- The record's "0.59 for the write column and 0.18 for the read row" figure was not located (see
  Corrections, item 4).
- X and other social posts were not searched. Unindexed work cannot be excluded.

## Files

- Downloads (arXiv HTML and PDF, wdd-descendants pages):
  `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/lit_in/n04/`
- Converted text used for reading:
  `/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/lit_text_n04/`
