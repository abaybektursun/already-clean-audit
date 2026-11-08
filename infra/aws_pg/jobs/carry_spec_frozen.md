# CARRY — pre-registration: which component of fine-tuning's write carries the gain — plus the literature-anchored controls, nulls, and theorem lines that the scan of 2026-07-12 makes mandatory

> **Instrument note (2026-09-03).** Blank-image MMLU numbers: the readout scores an off-format token
> surface in the VLM shell, but re-measurement on both surfaces gives identical accuracies, so nothing is
> retracted; the shell itself costs the healthy model ~0.16 accuracy vs text-only, which is the caveat that
> matters for interpretation. Crosscoder "features preserved" statements on the gemma-2→PaliGemma-2 pair
> are VOID (C6, FVU 8.6). See `studies/lineage/2026-09-03_INSTRUMENT_INCIDENT.md` and
> `studies/lineage/CLAIMS_LEDGER.md`.

**One sentence.** SPECTER measured that fine-tuning's writes concentrate in high-curvature (stiff)
activation directions while the conserved core is curvature-generic; the strongest relevant paper
(Song–Ahn–Yun, ICLR 2025) proves that exactly this kind of concentration can carry zero function;
CARRY splits each of the 24 banked writes into its stiff and flat components and lets the model's
own task gains say **which component carries the adaptation** — and, around that MDE, executes
every control, null, theorem anchor, and confound check the 2026-07-12 literature scan generated,
under one freeze, with the scan itself captured as a citable registry.

**Truth clause (binding).** The author's credence is genuinely split on the MDE (see §11 P1) —
there is no favourite to protect. Every §6 rule enumerates all branches with a **"what the
program learns"** entry; a VOID is an instrument-scoping measurement; confound arms (Arm P) are
empowered to RE-SCOPE prior SPECTER claims and that outcome is written below as a productive
landing, not a loss. Every rank/correlation statistic defines its degenerate-input completion
(all-tied, n < 3, zero-variance) **in this document** — the SPECTER E16 successor rule, now law.

---

## 0. Status + provenance + freeze mechanics

- **Status:** PRE-REGISTRATION, AUTHORED 2026-07-12 (not yet frozen). §0.5 amendments permitted,
  dated, until the first tag.
- **Freeze mechanics:** per-arm tags `carry-freeze-<arm>-<UTC-date>` on `science/carry`; every
  driver asserts tag existence + spec sha256 match pre-launch and **refuses to launch otherwise**.
  No retroactive tags.
- **Branch/worktree:** `science/carry` from the `science/specter` tip **recomputed at freeze**
  (`git rev-parse science/specter`; expected 1a4b23a, never transcribed into a rule). Worktree
  `../ntm-frozen-gemma-carry`; symlink `.pgvenv` from the main repo before any `pg run`.
- **Model:** PaliGemma-1 (`google/paligemma-3b-pt-224`, 18 L, d = 2048, MQA 1 KV head), **eager**;
  sha re-asserted at Gate-0 against the invcore-FUNC pin. The 24-fine-tune family is the banked
  PG1 corpus (weights on S3; snapshots deleted after use — the Arm-W disk lesson is a protocol
  line here, not a hope).
- **Parents (artifacts only):** SPECTER (`results/specter_*`: operator bank, stiffness rows,
  witness bank, `rho_bank`), ATLAS (`results/atlas_*`: dormant-gate census, corpus), BISPEC
  (`results/bispec_*`: Σ banks, draw seeds, Ω profiles), COMWRITE (LOO banks + per-FT measured
  gains), conserved_core. Every number in a §6 rule is recomputed from its named artifact at run
  time.
- **The literature registry is a first-class deliverable.** `2026-07-12_lit_registry.md` (built
  at authoring from the deep-research digest `2026-07-12_lit_digest.json`, sha256 prefix
  `2e9d361ad21970c5`) is the ONLY permitted citation source for the papers below. Papers are
  referenced here as S0–S24 + 4 pending, exactly as the registry numbers them. **House
  cite-verify gate:** no S-entry may be cited in any result doc until Arm L flips it to
  `verified-complete` (identity confirmed against the fetched source; its adversarial votes
  joined; quote re-checked at claim level).
- **Binding process inheritances:** per-stage `gpu_seconds` asserted non-null before `green`;
  BH families emit per-comparison p-values; `green` is each job's last statement; drivers halt
  on nonzero STATUS; journal (`2026-07-12_carry_journal.md`) live from day one; predictions
  journaled before every unblind (§9.3); construction-matched range references for every range
  guard.

## 0.5 Pre-freeze verification checklist

Code surfaces verified against the tree on 2026-07-12 (re-verify at freeze):

| claim | verified at |
|---|---|
| Swap-shell surgery REUSED: `project_delta(dW, U, mtype)` (side-aware ΔW projection), `build_decoder(cell, base_dec, ft_dec, bases, …, excluded, seed, roll)`, `is_target`, `grid_of`, `assert_full_grid` | `infra/pg/comwrite_surgery.py:38–108` |
| Readouts REUSED: `inherit_pg1.eval_pg1` (per-item soft-acc, scramble dead floor), `ppl_gate(n_seq=120)`, `load_pg1(sha=…)` | `infra/pg/inherit_pg1.py:35–223` |
| SPECTER banks: operator eigvecs/eigvals per (site × mode × loss); stiffness rows + gated spectral completion (fidelity ρ = 0.927 vs direct quadforms; K-FAC path INVALID at ρ = 0.103 and **must not be used**); witness bank (`witness_directions.pt`); `rho_bank.pt` | `results/specter_{o,k,h,w}` (paths + hashes re-listed at freeze) |
| Hooks/utilities REUSED: `atlas_core.make_hook_radial/tangential`, `rope_rotate`, quantile-shuffle + sign-flip error cells (A9 family), `bispec_core.make_hook_quant_permuted`, `saturation_guard`, `haar_frames`, `hook_at` | `infra/pg/atlas_core.py`, `infra/pg/bispec_core.py` |
| Lit digest present in-repo with hash | `research/2026-07-12_lit_digest.json` (`2e9d361a…`) |
| Registry built with claims verbatim | `research/2026-07-12_lit_registry.md` |

Executor items before tagging (each has a §9.4 response):

1. **Stiffness-metric scope.** Read `results/specter_k` and record exactly which reference metric
   its stiffness rows used (site, mode, loss) and which (layer, matrix) pairs it covers; the Arm-C
   split is defined in THAT metric, single-metric scope disclosed. If per-layer metrics are
   needed for coverage, one additional operator pass is pre-approved (≈ 2 GPU-h), frozen before
   Arm C launches.
2. **Per-position decomposition.** Check whether the operator bank stores per-position Rayleigh
   contributions; if not, Arm P's one-battery recompute (probe batches re-run with per-position
   logging) is pre-approved.
3. **Per-FT gain table.** Recompute each fine-tune's measured task gain from the COMWRITE
   artifacts (never from prose); the 6 discovery FTs are the first 6 by ascending |gain| **among
   FTs whose full-write swap-shell gain clears the range guard** — a frozen, mechanical selection.
4. **Gemma norm convention.** Verify the rotation-gauge theorem (S18) applies to Gemma's
   RMSNorm-with-(1+γ) convention before the Arm-W2 rotation check is trusted; the dry-run proves
   it on tiny tensors first.
5. **D-suite green** (§9.1), including degenerate-input completions for every rank statistic and
   the historical replays.

## Glossary (defined once)

| object | definition |
|---|---|
| **ΔW** | one fine-tune's weight change on a decoder matrix, from the banked 24-FT corpus |
| **stiffness metric Q** | the SPECTER weight-space curvature form (gated spectral completion, fidelity 0.927): how much behavioural curvature a unit-norm weight change in a given direction excites |
| **stiff / flat split** | eigendecompose Q per (layer, matrix); the stiff subspace = top eigvecs holding 50 % of Q's trace; flat = complement; ΔW_stiff / ΔW_flat = `project_delta` of ΔW onto each (side per the surgery convention, verified §0.5-1) |
| **swap-shell** | COMWRITE's instrument: install a projected ΔW into the frozen base and read the task gain with `eval_pg1` (the full-weight probe has no range on PG1 — the COMWRITE lesson; the shell is the only licensed readout) |
| **gain_frac_X** | (task gain of base + ΔW_X) / (task gain of base + ΔW_full), per FT |
| **Δcarry** | median over FTs of (gain_frac_flat − gain_frac_stiff) — the primary estimand |
| **spike / F̄ / witness** | as in SPECTER: curvature-matrix top eigenvectors / the measured operator / the banked adversarial worst-case directions |
| **S0–S24** | registry entries in `2026-07-12_lit_registry.md` — the only citable forms of the scanned papers |

---

## 1. Mission tie-in, question, hypotheses, falsification

**Mission tie-in.** The mission is what fine-tuning preserves and changes, with mechanism. CARRY
asks the mechanism question at the write itself: *is the stiff concentration of the write the
adaptation, or its exhaust?* Either answer reorganizes the preservation story — if the flat part
carries, "conserve the loud directions" is safe because the loud part was never the function; if
the stiff part carries, accumulated fine-tuning geometry genuinely departs from per-step SGD
geometry (S10) and the write's stiffness is itself the phenomenon to explain.

**Question.** Which component of fine-tuning's write — the high-curvature (stiff) projection or
the low-curvature (flat) projection — carries the behavioural task gain, and do the
literature-mandated controls (position confound, theorem nulls, sink controls, rotation
covariance, intruder census) leave the program's operator results standing at their current
scope?

**Program hypotheses (opposite predictions on Δcarry):**

- **H_flat-carries** — the flat component reproduces the gain; the stiff mass is functionally
  light (the S10 lesson transferred to accumulated writes).
- **H_stiff-carries** — the stiff component reproduces the gain; accumulated write geometry
  departs from per-step SGD geometry (bounds S10's transfer; promotes the parked lazy-dial arm).
- **H_distributed** — neither component alone reproduces it; the stiff/flat axis is not the
  functional axis (the split instrument is scoped, and the residual decomposition is banked).
- **H_no-range** (the boring one) — the swap-shell gains lack range on too many FTs; killed or
  confirmed by the Gate-0/discovery range guard before any sealed exposure.

**Falsification table (kill conditions named in advance):**

| hypothesis | killed by |
|---|---|
| H_flat-carries | Δcarry ≤ 0 with CI excluding +0.2, or flat inside the random-split null band |
| H_stiff-carries | Δcarry ≥ 0 with CI excluding −0.2, or stiff inside the random-split band |
| H_distributed | either component alone reaching gain_frac ≥ 0.6 beyond the band |
| "SPECTER spikes are real spikes" (standing claim under audit) | Arm P finding ≥ 0.7 of a stable spike's Rayleigh mass on sink/BOS positions — the claim is then RE-SCOPED to content-position curvature, disclosed as such |
| "witnesses exceed smooth-math bounds" (standing) | Arm N finding PG1 activations decisively non-isoperimetric — the √d line is then reported as not-binding and the claim re-anchored (also informative: the activation geometry itself violates the theorem's premise) |
| "operator unifies error families" (Arm E) | ρ(predicted, measured) < 0.5 across families with range — the escaping family is named and becomes the next mechanism target |

**Decision per world.** H_flat-carries → protect-the-loud-conserve-cheap becomes a testable
transfer rule and the conserved-core story is rewritten around functional emptiness of stiff
mass. H_stiff-carries → the lazy-dial successor (S7) is promoted to test whether regularization
moves the write out of stiff directions. H_distributed → the functional axis is sought in the
residual decomposition (banked). All three name their next step.

## 2. Estimands

**Primary (ICH E9):**
- **Population:** the 18 sealed fine-tunes (the 24-FT banked corpus minus the 6 frozen discovery
  FTs), each with its own measured-gain task.
- **Treatment/contrast:** swap-shell cells {base+ΔW_stiff, base+ΔW_flat, base+ΔW_full} per FT,
  plus the matched random-split null (3 seeds).
- **Variable:** per-FT task gain via `eval_pg1` on that FT's COMWRITE-recomputed gain task
  (paired by item id under the `carry|` salt); blank-image ppl at n_seq = 120 banked as the
  collateral row, never the adjudicator.
- **Intercurrent events:** an FT whose full-write gain fails the range guard (< 3× its
  construction-matched seed-noise) is VOID for scoring, counted, and reported; if > 9 of 18
  sealed FTs VOID, the verdict is a scope statement about swap-shell range, not about carrying.
- **Summary measure:** Δcarry with 10k FT-level pair-bootstrap 95 % CI; family {Δcarry_gain,
  Δcarry_ppl-collateral} BH q = 0.10 with emitted p-values; gain is the headline.

**Instrument identity:** `comwrite_surgery.project_delta` + `build_decoder` (the split install),
`inherit_pg1.eval_pg1` (readout), stiffness eigvecs from the `results/specter_k` completion
(recomputed from the artifact, never re-derived by K-FAC). Degenerate-input completions:
Spearman/Kendall on any all-tied vector := 0; a bootstrap over < 3 valid FTs returns VOID, not a
CI.

**Secondary estimands (one line each; rules in §6):** per-spike sink-position Rayleigh fraction
(Arm P); the G-vs-√d and Ford-bound overlays + isoperimetry diagnostics (Arm N); ρ(predicted,
measured) over ≥ 7 error families + the incoherence-processed quantization row (Arm E);
mean-substitution / RMS-preserving / zeroing triple and the Hadamard covariance gap (Arm W2);
intruder-dimension fraction across 24 FTs at ε ∈ {0.6, 0.7, 0.8, 0.9} (Arm D2); registry
completion state (Arm L); κ-bridge curve and targeted-witness gain (Arm B2, curiosity).

## 3. Design

### 3.1 Arms (ordered as run; every arm names what only it adds)

| arm | what it does | forwards | anchor papers (registry) |
|---|---|---|---|
| **Gate-0** | ppl anchor + model sha; **swap-shell wiring preflight** — reproduce one banked COMWRITE surgery cell's gain from its artifact before any new cell; RMSNorm-γ readout at the sink coordinates (minutes; the S11 check); MQA asserts; `carry|` re-salt audit | 1 small battery | S11 |
| **Arm L** (CPU/web) | complete the registry: extract the 4 pending papers; cast the 27 remaining adversarial votes; join all votes to claims; confirm the 25 inferred identities against fetched sources; flip statuses to `verified-complete`; regenerate the dossier artifact page with final numbers | 0 | all |
| **Arm P** (zero-forward*) | position-stratify every stable SPECTER spike: per-position Rayleigh contributions split {BOS/sink, content}; (*one pre-approved recompute battery if the bank lacks per-position terms) | 0–1 battery | S21 |
| **Arm N** | theorem anchoring: (a) replot Arm-H witnesses against the √d ≈ 45 Lipschitz line + the Ford −σ·Φ⁻¹(μ) bound computed from banked noise-damage rates; (b) isoperimetry diagnostics on banked activation rows (thin-shell concentration + Lipschitz-functional tail tests, frozen); (c) sink-head zero-QK-curvature check; (d) GGN-vs-full-Hessian comparison at the top eigenpair | (c,d) ≈ 1 small battery | S13, S8, S23, S24 |
| **Arm C** (MDE + confirmatory head) | stiff/flat split of each ΔW in the frozen stiffness metric; swap-shell triple + random-split null per FT; discovery on the 6 frozen FTs (rule rehearsal, no rule change after); sealed pass on the remaining 18, **once** | ≈ 24 FTs × 6 cells | S10, S20 |
| **Arm E** | error-family unification battery at L0: {SAE m=1 reconstruction error, quantization e(x), −e(x), within-quantile shuffle, full permutation, Gaussian iid, rotated-error, **incoherence-processed (Hadamard-rotated frame) quantization error**}; per-family F̄-spike alignment computed from the banked eigvecs; per-family damage predictions journaled BEFORE the battery; Lᵖ growth-exponent row fitted per family | ≈ 1 battery | S14, S12, S15, S18, S3 |
| **Arm W2** | sink/gauge controls: mean-substitution of sink coordinates (S17's exact null), RMS-preserving vs naive ablation, random-token input-agnosticity forward, Hadamard rotation-covariance instrument check (a banked statistic recomputed under the rotated gauge must be invariant) | ≈ 1 small battery | S17, S11, S16, S18 |
| **Arm D2** (CPU + 1 opt.) | intruder-dimension census: max cosine of each FT's top-10 post-FT singular vectors to the pretrained SVD, all 24 FTs × 7 matrix types × 18 layers; optional single λ-scaling intervention cell on the max-intruder FT | 0 (+1 cell) | S6 |
| **Arm B2** (curiosity, protected) | κ-family bridge from noise to one banked witness (damage vs κ curve); one targeted-witness search (install a named competing behaviour at matched norm) | ≈ 1 battery | S2, S9 |
| **parked** | the λ-regularization lazy dial (requires training runs); **promotion condition:** Arm C lands H_stiff-carries | — | S7 |

**The MDE is Arm C on the 6 discovery fine-tunes.** One person, one job: six swap-shell triples
with their nulls. Opposite outcomes force different conclusions and different successors; the 18
sealed FTs then give the headline its confirmation estimate.

### 3.2 What each non-MDE arm adds (the #11 discipline, one line each)

Arm L makes the scan citable (nothing else does). Arm P is the only check that can re-scope
SPECTER's spike claims before anything builds on them. Arm N converts two of our strongest
numbers into theorem-anchored numbers. Arm E is the only test that can unify four external
results with two of ours in one figure. Arm W2 closes the sink/gauge account with
literature-exact controls. Arm D2 is the only cross-check of our conserved-core story against
the external intruder-dimension mechanism. Arm B2 spends the curiosity budget on the two
instruments the scan handed us for free.

## 4. Data

1. **Items:** the VQA/task pools re-salted `carry|`; per-FT gain tasks recomputed from COMWRITE
   artifacts (§0.5-3). Discovery items 200/task; sealed cells run once on the sealed halves.
2. **FT-level split:** 6 discovery FTs (frozen mechanical selection, §0.5-3) / 18 sealed FTs.
   The rule is rehearsed on discovery and applied verbatim to sealed; no threshold moves after
   discovery is seen.
3. **Weights protocol:** one FT snapshot on disk at a time; delete after its cells complete;
   `HF_HUB_DISABLE_XET=1` in every job env.
4. **Manifests:** stiffness eigvecs, split projectors, per-FT ΔW hashes, error-family
   construction seeds — all content-hashed in `carry_manifest.json`; every exclusion counted.

## 5. Controls (each gates a specific claim)

| control | gates | construction |
|---|---|---|
| **swap-shell positive control** | the instrument detects a known gain | Gate-0 reproduces one banked COMWRITE surgery cell before any new cell runs |
| **random-split null** | Arm C's headline | matched-rank random eigenspace splits of the SAME ΔW (3 seeds/FT): both parts' gain_frac distributions form the band any real carrying must beat |
| **construction-matched seed noise** | every range guard | seed noise measured on the same cell family it gates (the ATLAS R-T1 repair, standing) |
| **saturation guards** | every cell | `saturation_guard` + dead-floor conventions |
| **shuffle tests** | Arm E, Arm D2 | family-label shuffle for the prediction ρ (breaks pairing, keeps marginals); pretrained-SVD column shuffle for the intruder census null |
| **theorem checks designed to fail** | Arm N (c), Arm W2 rotation | sink-head QK-curvature and Hadamard covariance are exact predictions; a miss is a fail-loud instrument event or a bounded-theory finding, adjudicated by magnitude |
| **input-agnosticity null** | Arm W2 | random-token forwards (S16's import): the sink structure must persist on nonsense input to be called content-free |

## 6. Metrics + frozen decision rules

Conventions: gains on each FT's own task scale; log(1+Δppl) for ppl rows; 10k pair bootstraps;
exact permutation where n ≤ 10; BH q = 0.10 within named families with emitted p-values; every
rank statistic's all-tied completion is ρ/τ := 0; status flags last.

**R-C (primary; family = {gain, ppl-collateral}).** Guards first: per-FT range guard (full-write
gain ≥ 3× construction-matched seed noise, else FT VOID, counted); split-rank parity assert;
random-split band from 3 seeds/FT.

| branch | condition (sealed 18, gain readout) | the program learns |
|---|---|---|
| `H_flat-carries` | Δcarry ≥ +0.2 with CI-lo > 0 AND median gain_frac_flat ≥ 0.6 AND flat exceeds the random-split band | the write's stiff mass is functionally light — S10's lesson holds for accumulated writes; the conserved-core story gains its mechanism candidate; protect-the-loud is licensed as a transfer rule to test |
| `H_stiff-carries` | mirror (Δcarry ≤ −0.2, CI-hi < 0, stiff ≥ 0.6 beyond band) | accumulated fine-tuning geometry departs from per-step SGD geometry — a clean, citable divergence from S10's regime; the lazy-dial successor is promoted |
| `H_distributed` | both parts inside the band, or both gain_frac ∈ [0.3, 0.7] with Δcarry CI covering 0 | the stiff/flat axis is not the functional axis — the split instrument is scoped, the per-FT residual decomposition is banked, and the functional axis is the successor's object |
| `VOID-range` | > 9 of 18 sealed FTs fail the range guard | the swap-shell's range on this corpus is the finding (a COMWRITE-lesson extension), reported with the per-FT table |

**R-P.** Per stable spike: sink_frac = Rayleigh mass on {BOS, sink} positions / total.
`H_spikes-content` iff sink_frac ≤ 0.3 for ≥ 6 of 8 promoted spikes (SPECTER's spike claims stand
at current scope); `H_spikes-sink-loaded` iff sink_frac ≥ 0.7 for ≥ 4 spikes (the claims are
RE-SCOPED to sink-position curvature — disclosed as a scoping correction in the registry and on
the artifact page; the gradient-sink literature S21 gains a live confirmation); `H_mixed`
otherwise (the per-spike map is the finding). All three branches are productive; none is a loss.

**R-N.** (a) Descriptive overlays — G values re-plotted against √2048 ≈ 45.3 and the Ford bound
(both recomputed in-driver). (b) Isoperimetry: two frozen diagnostics (norm thin-shell CV vs the
Gaussian reference; tail decay of 64 random 1-Lipschitz functionals vs the sub-Gaussian
envelope); `H_isoperimetric-enough` iff both within 2× of reference (the 45× line binds and our
witnesses' excess is theorem-anchored); `H_non-isoperimetric` otherwise (the line is reported
not-binding, and PG1's activation geometry violating the premise is itself a named finding that
bounds S13's applicability to transformer activations). (c) sink-head QK curvature: heads with
attention-to-sink ≥ 0.9 must show QK-block curvature ≤ 0.1× the median head → theorem-verified;
∈ (0.1, 0.5)× → measured leak, quantified; ≥ 0.5× → S23's one-hot collapse does not govern this
architecture — a bound on the theory, reported with the head census. (d) top-eig GGN/Hessian
ratio reported beside the banked trace gap (S24's caution, quantified at the point it warns
about).

**R-E.** Predictions for all ≥ 7 families journaled before the battery (from the operator
functional + spike alignments; the registry quotes the external predictions — S14's SAE-error
malignity, S18's incoherence mechanism). ρ_S(predicted, measured) with the label-shuffle null;
`VOID-no-range` iff measured damages span < 10×. `H_operator-unifies` iff ρ ≥ 0.8 AND shuffle
≤ 0.3 (four external results and two of ours land in one mechanism figure); `H_partial` iff
ρ ∈ [0.5, 0.8) (the escaping families are named — each is a mechanism lead); `H_families-escape`
iff ρ < 0.5 (the operator does not govern structured-error benignity — the strongest possible
scoping of the SPECTER atlas, and the residuals say where to look). QuaRot sub-row: incoherence-
processed quantization error's damage vs raw — sign and size reported against S18's mechanism
either way.

**R-W2.** Mean-substitution: `H_constant-bias` iff |Δ| ≤ 2× seed band while zeroing ≥ 10× band
(S17's exact pattern reproduced on PG1 — predicted-by credit recorded); any other pattern → the
deviation is the finding (PG1's sink differs from LLaMA's — a cross-architecture datum). RMS-
preserving vs naive ablation gap reported with CIs. Rotation covariance: the recomputed banked
statistic must match within fp tolerance → instrument certified; a miss is fail-loud (wiring or
the (1+γ) convention — §0.5-4 decides which before launch). Random-token persistence reported.

**R-D2.** Intruder fraction across the 24 full FTs at the frozen ε grid. `H_full-FT-clean` iff
median intruder count ≈ 0 at ε = 0.6–0.9 (S6's full-FT claim replicates on our corpus — and the
conserved-core account gains an independent spectral witness); `H_intruders-present` otherwise
(our 24-FT corpus departs from S6's regime — a corpus-level finding, and the intruder set is
immediately cross-referenced against U_unt's complement and the forgetting locus). Optional
λ-scaling cell adjudicated descriptively.

**R-L.** Registry gate: every S-entry `verified-complete` or explicitly `pending` with reason;
all 75 votes joined; any refutation discovered among the remaining 27 votes → the affected
entries re-tagged, the dossier artifact regenerated, and any spec text resting on that entry
amended by dated §0.5 note (the post-critic regeneration analog, applied to literature).

## 7. Statistical analysis plan

FT-level pair bootstraps (10k) for Δcarry; item-level bootstraps within cells; exact permutation
for n ≤ 10; BH q = 0.10 per named family with p-values emitted; all-tied rank statistics := 0 by
frozen completion; per-seed values banked; every effect stated against its matched null band,
never zero; discovery rehearsal may not alter any sealed threshold (no-rule-swap standing).

## 8. Threats-to-validity budget

| threat | effect | mitigation |
|---|---|---|
| swap-shell range on PG1 | Arm C starves | range guard + VOID accounting + the >9/18 scope branch; discovery FTs chosen among range-passing FTs |
| single-metric stiffness scope | split reflects one site/mode | §0.5-1 scope disclosure; pre-approved extra operator pass; random-split null bounds what any split can claim |
| accumulated-vs-per-step disanalogy (S10) | over-reading either landing | disclosed in §1; both landings' claims written to their own regime |
| snapshot disk overflow | mid-run death (the Arm-W event) | one-snapshot-at-a-time protocol; delete-after-use asserted in-driver |
| per-position terms absent | Arm P blocked | pre-approved recompute battery |
| registry identity errors | mis-citation | Arm L confirms every identity against the fetched source before `verified-complete` |
| isoperimetry diagnostics underpowered | R-N(b) inconclusive | two independent diagnostics, frozen; inconclusive → reported as bounded, the line labeled provisional |
| Gemma (1+γ) norm convention | rotation check invalid | §0.5-4 dry-run proof before launch |
| quantile-shuffle construction drift | Arm E families not comparable | REUSED atlas_b constructions with seeds re-verified against banked realized stats |
| sink-head census circularity | R-N(c) trivially passes | heads selected by attention telemetry only, curvature computed blind to selection |
| pool dry / Xet wedge | idle burn / stall | pg p5.4xlarge→g6e fallback; never wait; `HF_HUB_DISABLE_XET=1` |

Residual limitations, disclosed: one model; Arm C's gains are per-FT tasks of unequal difficulty
(the FT-level bootstrap absorbs scale, not task identity); Arm E is discovery-with-journaled-
predictions, not sealed; the registry's working tags originate from the scan and are only
authoritative after Arm L.

## 9. Blind protocol + execution runbook (autonomous)

**Strict order.** Offline gate + D-suite → Arm L (starts immediately, runs throughout) → Gate-0
→ Arm P, Arm N(a,b), Arm D2 (offline legs) → **Arm C discovery (6 FTs)** → journal the sealed
predictions → **Arm C sealed (18 FTs, once)** → Arm E (predictions journaled first) → Arm W2,
Arm N(c,d) → Arm B2 → unblind-per-arm → critic → registries + artifact refresh.

1. **Offline gate + D-suite (before any GPU):** split-projector algebra on tiny tensors
   (project_delta side conventions; stiff⊕flat = full reconstruction assert); rank-statistic
   degenerate completions constructed; the Hadamard/(1+γ) proof; error-family constructions
   replayed against banked realized stats; every §6 failure branch constructed, including
   R-C's VOID-range and R-E's VOID-no-range; historical replays (the saturated em-null → VOID;
   the R-T1 range-guard misfire → correctly classified; the all-tied Spearman corner → := 0).
2. **Preflight:** Gate-0's banked-cell reproduction is the first GPU act; any miss halts the
   arm, journal first.
3. **Predictions before unblinds:** §11 rows journaled per arm before its decisive read; Arm C's
   sealed predictions (per-FT expected direction under the discovery-rehearsed rule) journaled
   before any sealed FT runs; Arm E's per-family numeric predictions journaled before the
   battery.
4. **Pre-mortem responses:** OOM → microbatch ladder → g6e; missing per-position terms →
   recompute battery; stiffness coverage gap → extra operator pass then freeze; discovery FTs
   all-VOID → selection re-runs over the next 6 range-passing FTs (mechanical, journaled); any
   arm blocked → certificate + continue (arms are independent); Arm L web-source unfetchable →
   entry stays `pending` with reason, never guessed.
5. **Unblind once per arm; no rule swap; independence firewall** (external papers' numbers are
   comparison points and never thresholds — every threshold above is ours, frozen here).
6. **Critic-before-belief; post-critic regeneration** of every figure, the registry statuses,
   and the dossier artifact page.

## 10. Artifacts, logs, figure suite (maximum-visual mandate)

**Artifacts:** per-arm `out/metrics.json` (+ per-stage `gpu_seconds`, controls, p-values);
`carry_manifest.json`; split-projector and error-family banks (content-hashed); per-FT parquet
gain tables; the completed registry; `events.jsonl` + `run.log`; the artifact registry mapping
every figure/number → script + input hashes + DISCOVERY/CONFIRMATORY tag. All under
`s3://pg-lineage-769063704696/results/carry_*`.

**Figure suite — every figure: plain-language claim in the title backed by a number + CI + null
in `metrics.json`; a one-line "how to read"; a detailed caption saying what the eye should
check; the null/baseline overlaid; provenance footer (run id · SHA · input hash · tag);
colourblind-safe; PNG + PDF; full-sweep rule (every swept point and every component curve — the
standing house mandate). Regenerated post-critic.**

| id | figure | the claim it carries |
|---|---|---|
| **F-0** | program schema (pre-freeze): arms, dependencies, freeze points | the design itself, drawn before data |
| **F-C1 (headline)** | 24-FT slopegraph: gain_frac_stiff vs gain_frac_flat per FT, random-split band shaded, discovery/sealed marked | Δcarry with CI — the MDE's answer at a glance |
| **F-C2** | stiffness spectrum per (layer, matrix) with each ΔW's mass overlay | where the writes sit on the curvature axis |
| **F-C3** | per-FT gain waterfall: full → stiff-only → flat-only → random-split band | the decomposition FT by FT, no medians hiding tails |
| **F-P1** | per-spike stacked bars: sink/BOS vs content Rayleigh mass, 0.3/0.7 rule lines drawn | do the spikes survive the position confound |
| **F-N1** | the theorem ruler (log axis): typical 1× · √d = 45× line · Ford bound · witness band | our worst cases against published mathematics |
| **F-N2** | isoperimetry panel: norm thin-shell histogram + Lipschitz-functional tails vs Gaussian reference curves | does the theorem's premise hold on PG1 activations |
| **F-N3** | head scatter: attention-to-sink fraction vs QK-block curvature, predicted-zero line | the S23 collapse prediction, tested |
| **F-N4** | GGN vs full-Hessian at the top eigenpair, beside the banked trace gap | the S24 caution, quantified where it bites |
| **F-E1 (unification)** | error-family map: x = F̄-spike alignment, y = measured damage, 8 labeled families, operator-predicted curve + shuffle band | four external results and two of ours on one canvas |
| **F-E2** | Lᵖ growth exponents per family with p = 2 reference line | which corruptions are quadratic and which are not |
| **F-W1** | sink triptych per coordinate: mean-substitution / RMS-preserving / zeroing, seed band shaded | the constant-bias account, replicated or bounded on PG1 |
| **F-W2** | rotation-covariance identity plot (statistic pre vs post Hadamard gauge, y = x line) | the instrument's gauge certificate |
| **F-D1** | intruder heatmap: FT × layer max-cosine at ε grid, S6's full-FT expectation marked | our corpus against the intruder-dimension mechanism |
| **F-L1** | literature matrix: 29 papers × 8 banked results, cells colored by tag (predicted-by / consistent / bounds / tool / caution), verification status ringed | the whole scan, citable, in one view |
| **F-B1** | κ-bridge: damage vs κ from pure noise to one witness, with the E[cos θ] = κ/√(D+κ²) axis annotated | the graded path between typical and worst case |

**Post-execution artifact refresh (mandated):** the interactive dossier page
(`claude.ai/code/artifact/c582dbc4…`) is regenerated with final verdicts, the completed
registry statuses, and F-C1/F-E1/F-N1 as its lead visuals.

**On-disk lineage:** this prereg (frozen) → `2026-07-12_carry_journal.md` (live) → result doc →
`figures/carry/` + the registry. At close, update BOTH `studies/lineage/RESEARCH_INDEX.md` and
`studies/lineage/README.md`.

## 11. Pre-registered prediction grid (filled 1:1 at unblinding; author credences attached)

| # | quantity | author's prediction (credence) | result | verdict |
|---|---|---|---|---|
| P1 | **R-C sealed** | H_flat-carries, Δcarry ≥ +0.3 (medium — genuinely split; stiff-carries at 0.35) | | |
| P2 | R-C discovery→sealed stability | discovery direction persists on sealed (med-high) | | |
| P3 | R-P | spikes survive: ≥ 6/8 with sink_frac ≤ 0.3 (med-high) | | |
| P4 | R-N(b) | PG1 activations pass both isoperimetry diagnostics (med) | | |
| P5 | R-N(a) | witnesses stay ≥ 10× above the binding 45× line (high, conditional on P4) | | |
| P6 | R-N(c) | sink-heads show ≤ 0.1× median QK curvature (med-high) | | |
| P7 | R-N(d) | GGN underestimates the top Hessian eigenvalue by ≤ 2× (med) | | |
| P8 | **R-E** | H_operator-unifies, ρ ≥ 0.8; quantization benign BECAUSE low spike alignment (med-high) | | |
| P9 | R-E QuaRot row | incoherence-processing reduces quantization damage further (med) | | |
| P10 | R-W2 mean-sub | H_constant-bias reproduces on PG1 (high — S17 predicted-by) | | |
| P11 | R-W2 rotation | covariance check passes at fp tolerance (high — theorem, post §0.5-4) | | |
| P12 | R-D2 | H_full-FT-clean: near-zero intruders across the 24 (med-high) | | |
| P13 | R-L | zero refutations persist through the remaining 27 votes (med-high) | | |
| P14 | Arm B2 κ-bridge | damage stays near-noise until κ ≈ √D regime, then rises steeply (med) | | |
| P15 | Arm B2 targeted | a targeted witness installs the competing behaviour at matched norm (med-low) | | |
| P16 | program | at least one standing SPECTER/ATLAS claim gets RE-SCOPED by these controls (med — and that would be the process working) | | |

Misses stay visible and are scored as the information they are.

## 12. Compute budget + references

**Budget (GPU-h + run-count only; per-stage gpu_seconds logged by mandate).** Gate-0 ≈ 2;
Arm C ≈ 14 (24 FTs × 6 cells with snapshot cycling); Arm E ≈ 4; Arm W2 ≈ 3; Arm N(c,d) ≈ 2;
Arm P recompute (if needed) ≈ 2; Arm D2 optional cell ≈ 1; Arm B2 ≈ 4; margin ≈ 4.
**Total ≈ 36 H100-h across ≈ 9 pg jobs** (p5.4xlarge first, g6e fallback) + r7i CPU for
L/P/N(a,b)/D2. **Protected curiosity slice:** Arm B2 and the QuaRot sub-row run even if the
budget tightens.

**References.** The registry `2026-07-12_lit_registry.md` IS the reference section: S0–S24 with
verbatim claim-level quotes, four pending entries, the full 29-hit catalog, and per-entry
verification status. No paper is cited in any CARRY document except through it; no entry is
citable until `verified-complete` (Arm L). The digest (`2026-07-12_lit_digest.json`,
`2e9d361a…`) is the registry's raw source of record.

---

*Definition-of-done self-audit at authoring: decision named per world (§1); hypotheses opposite
on one primary readout (Δcarry) with the boring branch guarded first (§1, §6 R-C); MDE named and
one-person simple (Arm C discovery, 6 FTs); pre-mortems folded with responses (§9.4); estimand
5-attribute with CI and degenerate completions defined in-document (§2, §7); falsification table
includes the program's own standing claims as killable (§1); every §6 rule carries a ceiling or
VOID plus a failure branch the D-suite must construct (§9.1); control trinity per claim (§5);
discovery/confirmation split at FT level with a frozen mechanical selection (§4); no baked
thresholds — every parent number recomputed from its artifact; reuse ledger verified with file
+ line locations (§0.5); weight-free dry-run separated from the runtime preflight (§9.1–9.2);
journal + pre-unblind predictions (§9.3); sixteen figures each tied to number + CI + null with
reading guides and the full-sweep rule (§10); literature captured as a citable, claim-level,
verification-tracked registry with its own completion gate (§0, Arm L, R-L); independence
firewall, no-rule-swap, post-critic regeneration incl. the registry and the artifact page
(§9.5–9.6); freeze-tag-or-refuse per arm (§0). Nothing this program can observe is
uninformative: every branch of every rule names what the truth just taught us.*
