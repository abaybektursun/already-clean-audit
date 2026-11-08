# ATLAS — pre-registration: the susceptibility atlas — exact architectural symmetries, one estimable metric, and a sealed prediction test

> **Instrument note (2026-09-03).** Blank-image MMLU numbers: the readout scores an off-format token
> surface in the VLM shell, but re-measurement on both surfaces gives identical accuracies, so nothing is
> retracted; the shell itself costs the healthy model ~0.16 accuracy vs text-only, which is the caveat that
> matters for interpretation. Crosscoder "features preserved" statements on the gemma-2→PaliGemma-2 pair
> are VOID (C6, FVU 8.6). See `studies/lineage/2026-09-03_INSTRUMENT_INCIDENT.md` and
> `studies/lineage/CLAIMS_LEDGER.md`.

**One sentence.** ATLAS registers the closed-form symmetry structure of PG1's forward (three exact
invariances, computable from the weights), fits the one quadratic-plus-boundary object those
symmetries leave free, and holds the combined model to the strongest test this program can
administer: **predicting the damage of never-run perturbation cells, sealed, chosen adversarially
against it.**

---

## 0. Status + provenance + freeze mechanics

- **Status:** PRE-REGISTRATION, AUTHORED (not yet frozen). This document becomes a contract at
  freeze; until the freeze tag exists, §0.5 amendments are permitted and must be dated.
- **Freeze mechanics (binding, the BISPEC E13 successor rule):** each arm freezes as
  `atlas-freeze-<arm>-<UTC-date>` — a git tag on `science/atlas` — recorded here with tag + commit
  SHA + UTC timestamp at freeze time. **Every ATLAS driver asserts, before loading any model, that
  its arm's freeze tag exists and that the spec file's content hash at that tag matches the spec
  hash baked into the driver's config. A driver whose tag is absent refuses to launch.** No
  retroactive tags, ever.
- **Branch/worktree:** `science/atlas`, based on the `science/bispec` tip **as recomputed at freeze**
  (`git rev-parse science/bispec` — do not transcribe a SHA into this line before then). Worktree
  `../ntm-frozen-gemma-atlas`. Never the shared main worktree.
- **Model:** PaliGemma-1 (`google/paligemma-3b-pt-224`), 18 layers, d = 2048, **eager attention**,
  sha verified at Gate-0 against the value recorded in the REGRAIN result header artifact. All
  numbers PG1. No Gemma-2 contrast (retired object; standing 2026-07-01 ruling).
- **Parents (artifacts, not prose):** BISPEC (`2026-07-09_bispec_result.md` + `results/bispec_*`
  on S3), REGRAIN (`2026-07-09_regrain_result.md` + `results/regrain_*`), COMWRITE
  (`results/comwrite_*`), conserved_core / invariant_core banks. Every motivating number quoted in
  this spec carries its artifact name; **every number used in a decision rule is recomputed from
  its artifact at run time by the driver, never read from this document.**
- **Baseline discipline:** this program compares almost exclusively against its own lineage
  artifacts (`metrics.json` files named per rule). Where a task baseline is needed:
  `python scripts/compute_baselines.py > docs/BASELINES.md`, cite file + SHA.
- **Reporting rule (binding on the result doc).** Every decision rule in §6 enumerates *all* of its
  branches, and every branch carries a **"what the program learns"** entry. A VOID is an
  instrument-scoping measurement and is reported as one. A null is a boundary located, stated in
  the same plain register as a positive. The result doc contains no funnel/kill framing; findings
  are numbered declarative Results with caveats in their own sentences after the claim.
- **Journal:** `2026-07-10_atlas_journal.md`, live from day one, appended at every milestone.
  Predictions are journaled **before** each unblinding step (§9).

## 0.5 Pre-freeze verification checklist (executor completes; each line becomes a dated amendment if it changes anything)

Code surfaces verified against the tree on 2026-07-10 (re-verify at freeze; line numbers drift):

| claim | where verified |
|---|---|
| `inherit_pg1.eval_pg1` (VQA soft-acc, scramble dead floor via `scramble_seed`), `ppl_gate(n_seq=120, window=512)`, `harvest`, `ablate_at`, `load_pg1(sha=...)` | `infra/pg/inherit_pg1.py:35–223` |
| `bispec_core` REUSED surface: `accumulate_sigma`, `eig_bands`, `band_projector`, `make_hook_gaussian_group`, `make_hook_quant_permuted`, `make_hook_subspace_removal`, `make_hook_band_noise/deletion`, `make_hook_traverse`, `make_hook_gain_match`, `pertoken_subspace_pool`, `removed_energy_fraction/operator`, `energy_alignment/dispersion`, `sample_level_set(_raw)`, `haar_frames`, `principal_angles`, `n_eff_from_attention`, `band_profile`, `lazy_training_reference`, `random_write_reference`, `form_alignment`, `saturation_guard`, `hook_at`, adjudicators | `infra/pg/bispec_core.py:82–1173` |
| BISPEC Arm-K draw seeds are deterministic — `torch.Generator().manual_seed(11 + 1000*ss)` — so every banked coherent/iid draw vector is reconstructible offline, bit-exact | `infra/aws_pg/jobs/bispec_k.py:264–269` |
| Ω profiles banked (24-FT consensus write, 5 layers × 5 matrix types, per-band, both references) | `results/bispec_w/out/metrics.json` |
| v₁ (R18 direction) construction + σ₁ anchor (assert σ₁ reproduces the banked value) | `results/bispec_a/out/metrics.json`, built as `regrain_pipeB/out/f_steer` built it |
| MQA layout: `num_key_value_heads == 1`, `k_proj.shape[0] == v_proj.shape[0] == 256` | `2026-05-31_start_experiments_spec.md` §audit item 3; re-assert in Gate-0 |
| Σ banks at L0/L14 (1M rows, fp64, centered + uncentered, BOS-excluded) | `results/bispec_sigma/out/sigma.json` |

### §0.5.1 Pre-freeze amendments (2026-07-10, executor; full rationale in the journal E1–E2)

- **AM-1 (artifact-name correction).** `results/bispec_gate0b` does not exist; the pooled n_eff
  lives at `results/bispec_gate0/out/gate0.json` (`n_eff.pooled_median`). Drivers point there.
- **AM-2 (R-T4 leg B rule replaced — the frozen rule could not fire).** An exact permutation test
  on the ordering of 3 tercile statistics has min p = 1/6 > 0.05. Leg B is adjudicated on the
  per-window statistic instead: Spearman ρ between per-window n_eff and the per-window
  shared-minus-iid damage gap over the 120 ppl windows, permutation p ≤ 0.05 with ρ > 0.
  Terciles remain descriptive (F-T4).
- **AM-3 (R-B1 draw set defined).** "The 9 banked coherent draws" := the nine fully-coherent
  (g = all) P1 cells with seed formula `manual_seed(11 + 1000·ss)`: 3 @ benign (discovery),
  3 @ mid (discovery), 3 @ mid (sealed) — 6 distinct δ̄ vectors, 9 (δ̄, damage) pairs; Kendall
  τ-b handles the positive-part ties.
- **AM-4 (§0.5-1 sequencing).** The RoPE reproduction requires a GPU forward; per-arm freeze
  mechanics honour "verify before tagging" at the arm grain: Gate-0 runs under the gate0 tag;
  the T arm's tag is created only after Gate-0d passes.
- **AM-5 (freeze-assert mechanics).** The code tarball carries no `.git`/`studies/`; the driver
  half of freeze-or-refuse asserts sha256(`infra/aws_pg/jobs/atlas_spec_frozen.md`) against
  `infra/aws_pg/jobs/atlas_freeze.json` for its arm; the git half
  (`scripts/atlas_check_freeze.py`) runs on the control plane before every launch. Tag→commit
  pairs are journaled (a manifest cannot contain its own commit).
- **AM-6 (T-1 matched twin).** The radial field carries a ~0.82·ε DC component (mean(x̂)
  alignment); BISPEC proved coherent-DC injections carry the mean-shift mechanism. The
  ADJUDICATED tangential twin is therefore coherence-matched (one shared draw, orthogonalized
  per token); iid tangential and Haar-iid are support rows. Radial runs as ± (outward/inward),
  damage_radial = mean of the signed rows. Realized ‖δ̄‖ banked per cell.

Items the executor must verify before tagging (each has a §9 pre-mortem response):

1. **RoPE convention.** Reproduce one attention layer's output by manually applying HF Gemma's
   rotary embedding to hooked `q_proj`/`k_proj` outputs on one real batch; require max abs error
   < 1e-4 fp32. This certifies where "pre-RoPE" injection sits. (NEW code; the T-2 hook depends
   on it.)
2. **Draw reconstruction.** Regenerate all banked Arm-K draw vectors from the seed formula; verify
   the per-draw norm/energy invariants recorded in `results/bispec_k/out/metrics.json`.
3. **Corpus manifest.** Build `atlas_corpus.json` (§4.3) offline; every included cell resolves to
   an artifact path + content hash; every excluded cell is listed with its exclusion reason and
   counted — no silent caps.
4. **L8 harvest absent** (expected): grep S3 for an L8 Σ bank; if absent, Gate-0's harvest slot
   covers it (§3, Gate-0e).
5. **Weight-free dry-run green** (§9, D-suite) including the historical-event replays.

---

## 1. Mission tie-in, question, hypotheses, falsification

**Mission tie-in.** The program's charter is naming latent structure and mechanism in what
pretraining builds and fine-tuning changes. ATLAS names three candidate structures at once: the
**gauge distribution** G (the perturbation directions the architecture provably discounts), the
**susceptibility metric** (the quadratic form organizing damage on the quotient), and the
**boundary** (the one-sided direction Arm A of BISPEC left open). Every prior study sampled this
object cell by cell; ATLAS is the first to write the object down and let it stake predictions.

**Question.** Is behavioural susceptibility at a PG1 residual site organized by the architecture's
exact symmetries composed with one estimable quadratic form — and is that composite object
*predictive* of perturbation cells that have never been run?

**Competing hypotheses (program level):**

- **H_gauge** — the exact invariances (T1–T3 of §3.1) + a banded quadratic form organize the
  surface; a model that knows them predicts unseen cells materially better than energy alone.
- **H_energy** — Euclidean perturbation energy is sufficient within the measured domain (the
  incumbent: REGRAIN R1b's law, bounded across cells by BISPEC S6); gauge structure adds nothing
  predictive at this grain.
- **H_local** — susceptibility is cell-family-local; no transferable low-dimensional model
  predicts new cells (prior regularities were family-specific coincidences).

These make **opposite predictions on one primary readout**: the sealed prediction contrast Δρ (§2).
H_gauge ⇒ Δρ > 0 with M2 accurate; H_energy ⇒ M0 accurate and Δρ ≈ 0; H_local ⇒ both inaccurate.

**Falsification table** (each row names the observation that kills the hypothesis — including the
author's favourites):

| hypothesis | killed by |
|---|---|
| H_gauge (favourite) | radial ≈ tangential damage at matched energy (R-T1 ceiling branch); or RoPE ladder flat/non-monotone (R-T2); or sealed Δρ CI-hi ≤ 0 (R-M) |
| H_energy | M0 sealed rank-accuracy < 0.5 on cells *chosen adversarially to be easy for it to fail* — if energy survives that, it is strengthened, not merely unrefuted |
| H_local | either model clearing ρ ≥ 0.5 sealed |
| H_sqrt-neff (T3) | coherence ratio fails to track √n_eff across site and across item strata (R-T4, both legs) |
| H_one-boundary (T5a) | τ(damage, ⟨δ̄,v₁⟩₊) ≤ τ for the e₁ and norm regressors on the banked draws (R-B1) |
| H_dormant-gate (T5b) | clamping the flipped gate population recovers ≤ 25 % of the collapse while the clamp's own controls pass (R-B2) |
| H_visible-variance (T6) | λρ fits the banked write profile no better than λ (R-W) |
| H_maxent-null (T7) | tilt-model curve misses the held-out site's sampled curve beyond the band (R-N) |
| H_connection (T8) | Jacobian-conjugated transport closes < 50 % of the D_γ identity gap (R-X) |

Every kill lands somewhere informative: the branch tables in §6 assign each outcome its lesson.

---

## 2. Estimands

**Primary estimand (the program's headline; ICH E9 attributes):**

- **Population:** never-run perturbation cells at the L0 residual site (hook = residual entering
  `layers[0]`) of PG1's frozen forward, drawn from the pre-registered cell grammar (§3.4), applied
  at every position, eager attention.
- **Treatment/contrast:** predicted damage rank under model **M2** (gauge + banded metric) vs
  under model **M0** (energy-only), both frozen before the sealed battery (§6 R-M).
- **Variable:** blank-image wikitext Δppl at n_seq = 120 (primary readout; scored as
  log(1 + Δppl)); VQA soft-acc effect (secondary readout; dead-model floor from
  `scramble_seed`-patched images).
- **Intercurrent events:** a sealed cell that lands dead/saturated by the frozen guard
  (`saturation_guard`) is VOID for scoring and replaced by the next cell in the pre-registered
  ordered backup list (30 registered; the first 20 valid are scored). Every VOID is logged and
  reported.
- **Summary measure:** Δρ = ρ_Spearman(M2 predictions, measured) − ρ_Spearman(M0 predictions,
  measured) over the 20 valid sealed cells, with a 10k pair-bootstrap 95 % CI, per readout, BH
  across the 2-readout family.

**Instrument identity (estimand attribute):** `inherit_pg1.eval_pg1` (VQA; per-item soft-acc,
paired by item id under the `atlas|` salt) and `inherit_pg1.ppl_gate(n_seq=120, window=512)` (ppl),
both on the eager forward; hooks installed via `bispec_core.hook_at(layers, 0, hook)`.

**Secondary estimands** (one line each; full rules in §6): radial/tangential damage ratio at
matched per-token energy (R-T1); Spearman ρ of damage vs RoPE-plane frequency at matched injected
energy (R-T2); per-channel coherent-injection damage triple K/V/Q (R-T3); coherent/iid damage
ratio vs √n_eff across site and item strata (R-T4); Kendall τ of banked coherent-draw damage vs
⟨δ̄,v₁⟩₊ (R-B1); clamp-recovered fraction of the v₁ collapse (R-B2); Δfit of λρ vs λ against the
banked Ω profile (R-W); max curve gap at the held-out site (R-N); identity-gap closure fraction
(R-X); dispersion-dimension calibration error (R-D).

---

## 3. Design

### 3.1 The frozen theory block (registered now, so no fit can be accused of hindsight)

These eight statements are the program's registered mathematical content. T1–T2 are theorems of
the architecture (exact); T3–T8 are hypotheses-to-kill with derivations. Each is tested by a named
arm.

- **T1 (radial gauge — exact).** RMSNorm(x) = γ ⊙ x·√d/‖x‖ is exactly invariant to per-token
  positive rescaling, and its Jacobian (√d/‖x‖)·diag(γ)(I − x̂x̂ᵀ) annihilates the per-token radial
  direction x̂_t. At a residual site, the radial component of a perturbation reaches the network
  only through the raw skip path (it changes the skip-to-block mixing ratio), never through a
  normed block input. *Registered retro-explanations:* the gain-matched null's 0.4 % reproduction
  (`results/bispec_a`), band0's lowest per-energy sensitivity (`results/bispec_s`, S6), the
  second-order character of mean-shift damage.
- **T2 (softmax key gauge + RoPE ladder — exact + derived).** A position-coherent shift added to
  every **post-RoPE key** changes each logit row by a j-independent constant and therefore changes
  **no attention pattern, exactly**. A coherent shift on every **value** passes through attention
  at gain exactly 1 (it exits as a downstream mean shift); iid value noise is pooled down by
  ≈ 1/√n_eff. The **query** channel has no such invariance. RoPE grades the key gauge: a coherent
  **pre-RoPE** shift confined to rotary frequency plane f retains coherence
  |Σ_j a_j e^(i j ω_f)| — a Dirichlet-kernel-type factor, monotone-decreasing in ω_f over the
  context window, computable from the config with zero free parameters.
- **T3 (√n_eff transport law).** For mean-shift-free perturbations riding the value/transport
  path, the coherent/iid damage ratio is √n_eff. Motivating banked pair (both recomputed at run
  time): pooled-median n_eff = 3.93 (`results/bispec_gate0b`) ⇒ √n_eff = 1.98; sealed P3@r128
  ratio 14.98/7.39 = 2.03 (`results/bispec_k_sealed`). One matching point is not a law; R-T4 tests
  the scaling.
- **T4 (dispersion dimension).** For damage D(v) = vᵀMv over random unit directions,
  CV²(D) = 2/PR(M). Hence d_eff := 2/CV² estimates the participation ratio of the susceptibility
  form any multi-draw family probes — zero forwards. Banked reads (motivation): P1-coherent
  d_eff ≈ 1.1; REGRAIN R1b union-matched d_eff ≈ 25–37.
- **T5 (one-sided boundary; two parts).** (a) The banked coherent-draw damages are ordered by
  ⟨δ̄, v₁⟩₊ — the positive part of the overlap with the R18 direction — better than by |⟨δ̄, e₁⟩|
  or by norm; if so, Arm K's 110× spread and Arm A's asymmetry are one mechanism. (b) The
  boundary's substrate is a **dormant GeGLU gate population**: MLP gate pre-activations strictly
  negative on-distribution that +αv₁ switches on; −α pushes them deeper off, which yields the sign
  asymmetry for free, and the population's exhaustion yields a damage saturation a smooth cubic
  does not predict.
- **T6 (visible-variance write law).** ΔW accumulates Σ_t δ_t y_tᵀ, so mode-i write intensity
  ∝ λ_i ρ_i with ρ_i the squared error-coefficient correlation. The top mode is ≈ DC across tokens,
  so its gradient is the batch-mean of the error signal — near zero for a calibrated model —
  giving ρ₁ ≈ 0. Predicts Arm W's two-reference verdict shape: data-aligned in direction, starved
  on the top mode.
- **T7 (Bingham nulls).** The max-entropy distribution on Gr(k,d) at fixed expected removed energy
  is p(V) ∝ exp(θ·tr(P_V Σ)). It supplies a principled em-null sampler and expresses the
  level-set dispersion curve as a functional of the spectrum alone — a candidate closed form for
  BISPEC R5's per-site curve and an answer to its open site-leg question.
- **T8 (interventions transport by conjugation).** An ablation is an operator; the correct carry
  across sites is conjugation through the interlayer Jacobian (JVPs at rank k), not carrying the
  frame. Tested on the exact V1r case where frame transport left a 0.355 identity gap.

### 3.2 Arms (ordered as run; §9 fixes the strict sequence)

| arm | what it does | forwards | new code |
|---|---|---|---|
| **Gate-0** | anchors: recompute ppl anchor vs `invcore-FUNC` artifact; identity hook bit-exactness; `gauss_iid` R² vs REGRAIN R10 artifact; model sha; MQA shape asserts; **(0d)** RoPE convention reproduction (§0.5-1); **(0e)** L8 Σ harvest (1M rows, same recipe as `bispec_sigma`); `atlas|` re-salt audit | 1 small battery | RoPE check only |
| **Arm D0** (offline, CPU) | dispersion-dimension calibration on synthetic forms (PR ∈ {1, 4, 32, 256}, n_draws ∈ {3, 8}) + sweep over every banked multi-draw family; MQA K-read/band overlap from weights + banked Σ; corpus manifest build; **R-B1** banked-draw regression (draws regenerated from seeds); Bingham fit on banked L0/L14 curves + **prediction of the L8 curve written to the journal before L8's curve is sampled** | 0 | pure modules |
| **Arm T** (GPU) | **T-1 (MDE)**: radial vs tangential at matched per-token energy, ε ∈ {0.1, 0.2, 0.4}·rms, 3 seeds each; **T-2**: coherent K-path injection per RoPE frequency plane (8 planes, matched energy) + frequency-shuffled control; **T-3**: coherent injection decomposed by channel (post-RoPE K / V / Q; matched energy); **T-4**: P3-style shared-vs-iid removal at L14 (n_eff(L14) measured and journaled first) + per-stratum ratios at L0 (attention-entropy terciles) | ~50 cells | 3 hooks |
| **Arm B** (GPU) | dormant-gate capture along ±αv₁ (gate pre-activation sign census, layers 0–3); clamp traversal (flipped population zeroed) + random-clamp control + identity-at-α=0 check; curiosity cells **C-1** (sign-flipped quant error −e(x)) and **C-2** (within-cell shuffled error) | ~20 cells | clamp hook |
| **Arm W** (GPU) | backward battery on the frozen base: per-mode error-coefficient correlations ρ_i at the 5 banked layers × 5 matrix inputs, two proxy losses (VQA-answer CE on discovery items; blank-image wikitext CE); fit λρ vs λ vs shuffled-ρ against banked Ω profiles | 1 battery (fwd+bwd) | grad capture |
| **Arm X** (GPU) | JVP-conjugated transport of the U_unt ablation across the V1r site pair; re-run the identity check that D_γ failed | ~6 cells | JVP transport |
| **Arm M** (the confirmatory head) | assemble corpus (banked + T/B discovery cells); fit M0–M3 (§3.3) with leave-family-out CV; **freeze** fitted parameters + the adversarial selection algorithm's output (30 ordered cells) + journaled per-cell predictions; run the **sealed battery once** on the `atlas|` sealed half; unblind once | ~30 discovery fill-in cells + 20–30 sealed | fit module (pure) |

**The MDE is T-1.** One person, one job: radial vs tangential perturbations at matched per-token
energy. H_gauge and H_energy predict opposite outcomes (ratio ≪ 1 vs ≈ 1) on one readout, and
either branch redirects the rest of the program (§6 R-T1). Everything else is corroboration or
construction on top of it.

### 3.3 The model family (frozen; fit on discovery only)

All models predict log(1 + Δppl) and the VQA effect of a cell from its **cell descriptor** —
computable from the perturbation's construction, the banked Σ eigendecomposition, and the weights,
**never from a forward**:

- **M0 (energy-only):** damage = monotone spline of injected/removed per-token energy (add and
  delete fitted separately; isotone regression).
- **M1 (banded metric):** M0 + a 16-sub-band diagonal sensitivity vector in Σ's eigenbasis
  (sub-bands = frozen energy-quantile refinement of the 5 BISPEC bands).
- **M2 (gauge quotient):** M1 with the descriptor first decomposed through T1–T3: per-token radial
  fraction removed; sequence-coherent component split by channel reach (K post-RoPE / V / Q) with
  the K-coherent part discounted by the Dirichlet retention factor and the V-coherent part
  assigned transport gain 1 vs 1/√n_eff for iid; mean-shift component flagged.
- **M3:** M2 + a one-sided term c·⟨δ̄, v₁⟩₊² (v₁ from the banked Arm-A construction).

Fit: least squares on log-damage; model selection is **not** performed on fit quality — all four
are frozen after fitting and all four stake sealed predictions. Leave-family-out CV on the corpus
is reported as a descriptive robustness panel (which families each model transfers to — every
pattern is a finding about where structure lives).

### 3.4 The sealed-cell grammar + adversarial selection (frozen algorithm)

Cell grammar (the population the primary estimand quantifies over): {add, delete} ×
{sub-band ∈ 16} × {sequence mode: dc, sin4, white, shared-subspace, per-token-subspace} ×
{channel confinement: none, K post-RoPE plane (low/high), V, Q} × {radial/tangential projection} ×
amplitude ∈ {0.1, 0.2}·rms — minus already-run combinations (the corpus), minus combinations the
dry-run proves degenerate (e.g., radial × delete at a band with no radial mass).

Selection: rank candidate cells by |M2 − M0| predicted disagreement (in log-damage), enforce
coverage (≥ 3 cells per major axis; no more than 4 cells sharing a (band × mode) pair), take the
top 30 as the ordered sealed list. The algorithm is code, in the tree at freeze; its output is
committed and journaled **with per-cell numeric predictions from all four models** before any
sealed forward runs.

---

## 4. Data

1. **Items.** The VQA pool re-salted `atlas|` (fresh hash salt ⇒ fresh discovery/sealed split;
   audit at Gate-0 that no `atlas|`-sealed item id coincides with a `bispec|`-sealed exposure in
   the cells we reuse — reused *banked* cells were measured on `bispec|`/`grain|` splits and enter
   only as training data for the fit, which touches no `atlas|` sealed item). Discovery cells run
   on 200 discovery items; the sealed battery runs on 200 sealed items, **touched once**.
2. **ppl.** Blank-image wikitext, n_seq = 120, window 512 — the program-frozen convention
   (`ppl_gate`). Cells from any source at other n_seq are excluded from the corpus (counted).
3. **Corpus manifest** (`atlas_corpus.json`, built offline, content-hashed): every banked
   (perturbation → damage) cell at L0/L14 with n_seq = 120 readouts from `bispec_k`, `bispec_s`,
   `bispec_a`, `bispec_k_sealed` (as banked numbers — their sealed half is *their* program's, not
   ours), `regrain_gate0`, `regrain_pipeA/i_frontier`, `regrain_c_*` energy ladder. Each cell:
   descriptor, damage, artifact path, content hash, DISCOVERY/CONFIRMATORY tag from its parent.
   Exclusions (dead/saturated by the frozen guard; wrong n_seq; wrong site) are listed and counted
   in the manifest — the count appears in the result doc. No silent caps.
4. **Forward-condition parity.** Eager everywhere; hooks at the same site convention as parents
   (`hook_at(layers, L)` = residual entering layer L); Σ = the banked centered, BOS-excluded
   estimate (`bispec_sigma`), perturbations applied at every position.

---

## 5. Controls (each gates a specific claim)

| control | gates | construction |
|---|---|---|
| **positive control (program)** | the fit machinery can detect known structure | M1 must retro-fit the banked S6 band-sensitivity ordering (band2 > band1/3/4 > band0 per unit energy) on the corpus; if the fit cannot recover what the data plainly shows, the fitter is repaired before freeze (this is a Gate-0-class check on code, not a result) |
| **seed-noise band** | every new-cell damage claim | 3 seeds per stochastic cell; the identity-hook cell bounds hook overhead at exactly 0 |
| **matched-energy Haar null** | every directional claim in Arm T | same construction, Haar direction (T-1: random non-radial direction at same per-token energy; T-2: same energy spread over a Haar plane pair; T-3: Haar 256-dim channel-shaped null) |
| **shuffle tests** | R-T2, R-W | frequency-shuffled plane assignment (breaks the ladder, keeps marginals); mode-shuffled ρ profile (breaks λ–ρ pairing, keeps both marginals) |
| **saturation/over-shoot guards** | every cell, both readouts | `saturation_guard` with the dead floor + intact anchors; traversal cells also carry the BISPEC over-shoot guard (gain_frac-style, VOID above 1.25) |
| **clamp integrity pair** | R-B2 | identity-at-α=0 (clamp installed, no traversal: |ΔVQA| ≤ 2·seed-noise) and random-clamp (equal count, never-flipping units: must NOT recover damage) |
| **the null's null** | R-N | the tilt-model's L8 prediction is compared against the *sampled* L8 curve AND against the best constant-curve baseline (site-average of L0/L14); the model must beat the constant to claim sufficiency |

Every one of these can fail, and §6 names what each failure reports.

---

## 6. Metrics + frozen decision rules

Global conventions: damage scored as log(1 + Δppl) for ppl and effect (1 − RF convention) for VQA;
10k pair-bootstrap CIs; exact permutation tests where n ≤ 10; BH q = 0.10 **within the named
family of each rule** (family memberships listed here, in one place, so no clause can silently
land in the wrong family — the BISPEC Arm-K lesson). Every rule below states: statistic → branches
(each with threshold, and **ceiling or VOID guard** — no floor without a ceiling) → what the
program learns on each branch. Status flags are computed **last** in every driver, after all
writes (the REGRAIN E17 rule).

**R-T1 (MDE).** ratio_rt = damage_radial / damage_tangential at matched per-token energy, per
amplitude ε ∈ {0.1, 0.2} (0.4 is a disclosed high-amplitude row, not adjudicated).

| branch | condition (both ε unless noted) | the program learns |
|---|---|---|
| `H_radial-gauge` | ratio ≤ 0.25 AND tangential ≥ 3× seed-noise AND neither arm saturated | T1 is load-bearing at this site; M2's radial discount is licensed; band0's cheapness is mechanistically explained |
| `H_radial-partial` | 0.25 < ratio < 0.75 | the skip path carries a measurable share; the radial discount becomes a *fitted* parameter in M2 (pre-specified fallback), and the skip-mixing mechanism gets a named successor |
| `H_radial-active` (ceiling) | ratio ≥ 0.75 | the gauge picture is bounded at its cheapest point — the strongest possible scoping result for T1; M2 runs without the radial discount (pre-specified), and the theory block's error is itself the finding |
| `VOID-no-range` | tangential < 3× seed-noise at both ε | the site tolerates 0.2·rms tangential noise — itself a tolerance measurement; one pre-specified re-run at ε = 0.4, then adjudicate there or report the tolerance curve as the result |
| `VOID-saturated` | either arm past the dead floor | amplitude grid was too hot: report, drop to ε = 0.05 once (pre-specified), adjudicate there |

**R-T2.** ρ_freq = Spearman(frequency index, damage) over 8 K-path planes at matched injected
energy; gap = damage(highest plane) − damage(lowest plane), CI by pair bootstrap; shuffle control
must show |ρ_shuffle| ≤ 0.3.

| branch | condition | learns |
|---|---|---|
| `H_ladder` | ρ_freq ≥ 0.8 AND gap CI-lo > 0 AND shuffle passes | the addressing circuit's exposure is frequency-graded exactly as the derived retention curve predicts — a zero-parameter architectural law; the S3 sign reversal has its mechanism |
| `H_flat` | |ρ_freq| ≤ 0.3 AND gap CI straddles 0 | K-path damage is not addressing-decoherence-limited: the softmax gauge is real (T2 is a theorem) but not load-bearing at this energy — locates the damage carrier in V/Q channels, sharpening T-3's reading |
| `H_inverted` (ceiling) | ρ_freq ≤ −0.8 | low-frequency planes carry the load — a named structure (slow-rotating addressing subspace) nobody predicted; successor named |
| `H_graded` | anything else | the partial ordering is reported with the fitted retention curve; the deviation profile localizes where attention's read deviates from the kernel prediction |
| `VOID` | any plane cell saturated/dead | re-run that plane at half energy (pre-specified, once) |

**R-T3.** Damage triple (K-coherent post-RoPE, V-coherent, Q-coherent) at matched energy.
Registered prediction (T2): K ≈ 0 (exact gauge), V = mean-shift-class, Q active. Adjudicated
descriptively against the seed-noise band and the Haar channel null; the triple's ordering is the
finding whichever way it lands (each of the 6 orderings maps to a named statement about which
channel carries stream damage — table in the driver). The K-coherent cell doubles as a **theorem
check**: K-coherent damage > 3× seed-noise ⇒ the hook is not where the theorem applies ⇒
instrument bug, fail loud (this is a signal designed to be able to fail).

**R-T4 (√n_eff; family = {leg A, leg B}, BH q = 0.10).**
Leg A (cross-site): n_eff(L14) is measured and journaled **before** any L14 coherence cell runs;
prediction_law = √n_eff(L14), prediction_constant = the L0 ratio recomputed from
`results/bispec_k_sealed`. `H_law` iff the measured L14 ratio's CI excludes prediction_constant
and contains prediction_law; `H_constant` iff the reverse; `H_both/neither` reported as the
overlap/exclusion pattern — each of the four patterns has a lesson row in the driver. `VOID-degenerate`
iff |√n_eff(L14) − √n_eff(L0)| < 0.3 (no discrimination possible — itself a finding about
attention-geometry stability across depth; leg B then carries the family alone).
Leg B (within-site): per-tercile ratios must order with per-tercile √n_eff (exact permutation
p ≤ 0.05 on the ordering). Either leg alone clearing at BH q = 0.10 ⇒ the law survives its first
test; both failing ⇒ the factor-two is a site constant, not an attention-pooling quantity — which
redirects M2's sequence factor to a fitted constant (pre-specified) and names a new question
(what sets the constant?).

**R-B1 (banked draws; zero forwards).** τ_v1 = Kendall τ(damage, ⟨δ̄,v₁⟩₊) over the 9 regenerated
coherent draws; comparators τ_e1 = τ(damage, |⟨δ̄,e₁⟩|), τ_norm = τ(damage, ‖δ̄‖); exact
permutation p on τ_v1.

| branch | condition | learns |
|---|---|---|
| `H_one-boundary-lead` | τ_v1 ≥ 0.6, p ≤ 0.05, τ_v1 > max(τ_e1, τ_norm) | Arm K's spread and Arm A's asymmetry are one object; BISPEC's "two mechanisms" unifies correctly; M3's term is independently motivated |
| `H_two-mechanisms-confirmed` | τ_e1 ≥ 0.6 with the same margin over τ_v1 | the coherent spread is e₁-carried after all; the two-mechanism reading is *strengthened* by the test built to overturn it |
| `H_neither-orders` | both τ < 0.6 or p > 0.05 | the spread's carrier is outside the two candidate directions — the residual regression (damage vs the full δ̄ decomposition) is banked for the successor, and d_eff (R-D) bounds its dimensionality |

**R-B2 (dormant gates; family = {recovered_frac}, guards outside the family).**
recovered_frac = 1 − damage_clamped/damage_unclamped at +3σ₁.
Guards first: identity-at-α=0 within 2× seed-noise (else `VOID-instrument`: the clamp has its own
footprint — scoped, battery re-run with soft clamp once); random-clamp must recover ≤ 0.25 (else
`VOID-clamp-artifact`: *any* clamping helps — a finding about MLP fragility, reported, arm ends).
Then: `H_dormant-gate` iff recovered_frac ≥ 0.5 (mechanism nailed causally; the saturation-shaped
proxy replaces A3's monotone one program-wide); `H_elsewhere` iff ≤ 0.25 (the boundary is not
layer-local MLP gating — attention/later layers named as the successor's territory, and the gate
census still yields the first map of PG1's dormant-gate reservoir); `H_partial` otherwise (the
fraction is the finding; the boundary is distributed across mechanisms).

**R-W (family = {Δfit_ppl-proxy, Δfit_vqa-proxy}, BH q = 0.10).**
Δfit = ρ_Spearman(Ω, λρ) − ρ_Spearman(Ω, λ) over the banked profile points (bands × layers ×
matrix types), per proxy loss. `H_visible-variance` iff Δfit ≥ 0.15 AND ρ_Spearman(Ω, λρ) ≥ 0.6
AND the shuffled-ρ control shows Δfit_shuffle ≤ 0; `H_other-order` iff Δfit ≤ 0 (the error-input
correlation is not the missing factor — the residual profile Ω/λ is banked as the named object for
the successor, and the ρ battery itself is a new artifact: the first per-mode task-visibility map
of PG1); `H_partial` between (which layers/types clear is the stratification finding — reported as
the map it is). Scope disclosed up front: ρ comes from proxy losses on the frozen base, not the
24 FTs' own objectives; the fit is a transfer claim at exactly that scope.

**R-N.** gap = max over the E/E_max grid of |predicted(L8) − sampled(L8)|, where the prediction is
journaled before sampling. `H_maxent-sufficient` iff gap ≤ 0.10 AND the model beats the constant
baseline (§5); `H_deviates` otherwise — the deviation profile (which E/E_max region, which sign)
is the finding and names where the real ensemble is more/less constrained than max-entropy.
Either way the program gains the Bingham sampler, which upgrades every future em-null.

**R-X.** closure = 1 − gap_conjugated/gap_frame, where gap_frame is the D_γ identity gap
recomputed from `results/bispec_v1r/out/v1.json`. `H_connection` iff gap_conjugated ≤ 0.10
(transport identity restored; every cross-site move gains a licensed carry); `H_higher-order` iff
closure < 0.5 (first-order conjugation is insufficient — quantifies exactly how non-linear the
inter-site map is, a number nothing in the lineage has measured); `H_partial` between (closure
fraction reported; the residual is the successor's target).

**R-D (instrument calibration; gates the d_eff readings, not a hypothesis).** On synthetic
quadratic forms: recovered d_eff within [PR/2, 2·PR] for PR ∈ {1, 4, 32} at n = 8 draws, and the
n = 3 bias curve banked. If the factor-2 band fails at n = 3 (expected — quantifying *how* wide
n = 3 is, is the point), d_eff readings on n = 3 families are reported with the calibrated
interval, ordinal use only. The banked-family sweep table (every multi-draw family in the corpus
with its d_eff + calibrated interval) ships regardless — it is the program's first quantitative
map of susceptibility dimensionality by mechanism.

**R-M (primary; family = {Δρ_ppl, Δρ_vqa}, BH q = 0.10; ppl is the headline).**

| branch | condition (per readout) | learns |
|---|---|---|
| `H_metric` | Δρ CI-lo > 0 AND ρ_M2 ≥ 0.5 | the gauge-quotient metric is *predictive machinery* — the program's first sealed forward-prediction instrument; M3 vs M2 secondary contrast then reads whether the boundary term adds sealed skill |
| `H_energy-sufficient` | ρ_M0 ≥ 0.5 AND Δρ CI-hi < 0.1 | energy survives cells chosen adversarially to break it — the strongest form of the R1b law ever measured, and the gauge structure is scoped to the symmetry theorems themselves |
| `H_partial` | Δρ CI straddles 0 with both ρ ≥ 0.5 | both models carry real, overlapping structure; the per-cell residual decomposition (which grammar axes M2 wins on) is the finding and the successor's map |
| `H_neither` | both ρ < 0.5 | the surface holds structure neither grammar expresses at this grain — the 20 sealed cells + residuals become the best-characterized anomaly set in the lineage, each already tagged by descriptor |
| `VOID-battery` | > 10 of 30 backup cells VOID | the grammar's amplitude convention doesn't fit the sealed regime — re-freeze once at half amplitude with a dated label (the one pre-specified re-freeze, used only if the battery itself is unreadable) |

**Curiosity cells (C-1, C-2; exploratory, no adjudication).** C-1: −e(x) (sign-flipped true
quantization error — function-of-token preserved, lattice landing destroyed). C-2: within-cell
shuffle (errors permuted among tokens with matched ⟨x⟩-quantile — pairing broken, conditional
marginals kept). Registered readings: C-1 ≈ permuted-damage ⇒ lattice landing is the mechanism;
C-1 ≈ true-damage ⇒ the mechanism is the error–token statistical dependence, not the lattice;
C-2 interpolating ⇒ dependence-strength is the axis. Every landing assigns A9 a mechanism class.

---

## 7. Statistical analysis plan

- Bootstrap: 10k resamples, paired by item id (VQA) / sequence id (ppl); BCa where skewed.
- Multiplicity: BH q = 0.10 within each rule's named family (§6 lists every family; no clause may
  be scored in a family other than the one printed here).
- Small-n: exact permutation distributions (R-B1's n = 9; R-T4 leg B's terciles).
- Damage scale: log(1 + Δppl) frozen for all fitting/scoring (handles the multiplicative regime);
  raw Δppl always reported beside it.
- Stability: 3 seeds per stochastic cell; per-seed values in `metrics.json`; a claim whose sign
  flips across seeds is reported as seed-bounded (that is its scope, not a defect).
- Effect-vs-null: every directional claim is a gap to its matched null with a CI, never a gap to
  zero.

## 8. Threats-to-validity budget

| threat | effect if unhandled | mitigation |
|---|---|---|
| ppl nonlinearity/saturation | per-energy sensitivities become floor artifacts | log scale + `saturation_guard` VOIDs + amplitude grid capped at 0.2 for adjudicated cells |
| RoPE convention mismatch | T-2 injects into the wrong frame | Gate-0d numeric reproduction gate (< 1e-4) before any T-2 cell |
| draw non-reconstruction | R-B1 regresses on wrong vectors | §0.5-2 bit-check of norm/energy invariants vs the banked artifact, pre-freeze |
| corpus convention drift | fit trained on incomparable numbers | manifest inclusion rules (site, n_seq, guard) + per-cell content hashes; exclusion counts published |
| winner's curse in model fitting | M2 flattered by its own training data | all sealed scoring on never-run cells; selection algorithm frozen; predictions journaled pre-battery |
| proxy-ρ gap (Arm W) | λρ mis-scored against FT writes | scope stated in the estimand; two proxies; shuffled-ρ control isolates pairing information |
| e₁/radial conflation | T-1 read as a band claim | T-1 is built per-token from x̂_t, not from e₁; the two are 0.82-aligned, disclosed, and orthogonal in construction here |
| clamp footprint | R-B2 credits the clamp, not the mechanism | identity-at-α=0 + random-clamp integrity pair, adjudicated before the branch |
| n_eff estimator choice | T3 scored against the wrong pooled statistic | median (banked convention) primary; mean banked beside it as a frozen sensitivity row |
| sealed-item leakage | corpus indirectly touched `atlas|` sealed items | fresh salt + Gate-0 overlap audit; banked cells enter only as fit data measured under other salts |
| GPU pool dry | idle burn or stall | pg launcher, p5.4xlarge first, g6e fallback with batch reduction; never wait on a dry pool |
| HF Xet stall | silent download wedge | `HF_HUB_DISABLE_XET=1` in every job env |

Residual limitations disclosed, not hidden: one model (PG1), primary site L0 (L14 for T-4 leg A
only); Arm W's ρ is proxy-task; R-B1's n = 9 bounds its power (exact tests only).

## 9. Blind protocol + execution runbook (autonomous; no mid-run human decision)

**Strict order.** Offline gate → Gate-0 → D0 → T → B, W, X (any order; independent) → M-fit →
M-freeze (params + sealed list + journaled predictions) → **M-sealed (once)** → unblind → result
doc.

1. **Offline gate (before any GPU):** full pipeline minus model-forward on real data + tokenizers —
   corpus build, descriptor computation, fit module on banked cells, selection algorithm, all
   adjudicators. Plus the **weight-free dry-run suite (D-suite)**, which must be green and must
   include the historical replays: REGRAIN R1's saturated em-null through `saturation_guard`
   (must VOID); the A7 chordal event through the energy-alignment metric (must not saturate);
   the banked P1 draw triple through d_eff (must read ≈ 1); the A3 monotone-proxy event through
   the new saturation-shaped proxy on a rotated control (must not correlate); Bingham sampler
   moments vs the analytic k = 1 case; T1's Jacobian identity on tiny tensors; every §6 branch
   reachable and mutually exclusive on constructed inputs — **including each rule's failure
   branch** (a rule whose FAILURE branch the dry-run cannot construct is returned to the author
   as a spec defect).
2. **Preflight (first GPU step):** Gate-0 anchors — each recomputed from its named artifact at run
   time. Any anchor miss halts the arm, journal first.
3. **Journal predictions before each unblind:** the §11 grid rows for an arm are copied into the
   journal, timestamped, before that arm's decisive cell runs (D0's Bingham L8 prediction; T-4's
   n_eff(L14) prediction; M's 30-cell prediction table).
4. **Pre-mortem responses (each failure pre-answered):** T-1 no-range → ε = 0.4 once → tolerance
   curve is the result. T-2 plane saturation → half energy once. L8 harvest OOM → L4 (pre-named
   substitute). Backward battery OOM → halve batch, chunk heads. Draw reconstruction fails →
   re-run the 9 coherent cells with logged seeds (one battery, pre-approved). V1r artifact numbers
   fail recompute → Arm X re-derives both legs from `bispec_v1r` job code paths before comparing.
   Any arm fully blocked → journal, close with certificate, continue with remaining arms
   (independence is by design); M runs on whatever discovery cells exist, with the corpus deficit
   disclosed.
5. **Unblind once.** The sealed battery is scored by the frozen R-M rule; the grid is filled 1:1;
   no threshold moves after unblinding; if a rule proves mis-shaped, both readings are reported
   under the no-swap discipline.
6. **Critic-before-belief:** an adversarial review pass on the result doc against artifacts before
   any registry update; post-critic, every downstream figure regenerated.

## 10. Artifacts, logs, figure suite

**Artifacts:** `atlas_corpus.json` (+ hash); per-arm `out/metrics.json` with every number any
figure cites; `controls.json`; per-cell parquet tables keyed by descriptor hash + item id;
`events.jsonl` + `run.log`; realized GPU-h per stage; the artifact registry (figure/number →
script + input hashes + DISCOVERY/CONFIRMATORY tag). All under
`s3://pg-lineage-769063704696/results/atlas_*`. `green` is the last statement every job executes;
drivers halt on nonzero STATUS.

**Figure suite** (each: claim in the title backed by number + CI + null in `metrics.json`; a
one-line "how to read"; a detailed caption explaining what the eye should check; null overlaid;
provenance footer run-id · SHA · input-hash · tag; colourblind-safe; PNG + PDF; full-sweep rule —
every swept point and every component curve, never the headline diff alone):

- **F-M1 (headline):** sealed parity plot — predicted vs measured log-damage, 20 cells, M0 and M2
  side by side, per-cell descriptor glyphs, Δρ + CI in the title.
- **F-T1:** radial vs tangential damage across the amplitude sweep — both curves, seed band, Haar
  null, saturation ceiling marked.
- **F-T2:** the RoPE ladder — damage vs frequency index, all 8 planes, derived retention curve
  overlaid, frequency-shuffled control band.
- **F-T3:** the K/V/Q channel triple with seed + Haar bands.
- **F-T4:** coherence ratio vs √n_eff — L0 strata + L14 point, law line and constant line both
  drawn.
- **F-B1:** banked-draw damages vs ⟨δ̄,v₁⟩₊ and vs |⟨δ̄,e₁⟩| (two panels, same y-axis), τ + exact
  p per panel.
- **F-B2:** the clamp quartet — unclamped, clamped, random-clamp, identity — across α, crossing
  count on the twin axis.
- **F-W1:** the write profile overlay — measured Ω, λ (NTK), λρ, Haar, shuffled-ρ — per layer ×
  matrix small-multiples.
- **F-N1:** predicted vs sampled dispersion curves at L0/L14 (fit) and L8 (held out), constant
  baseline drawn.
- **F-D1:** d_eff calibration curve (synthetic PR vs recovered, n = 3 and n = 8) + the
  banked-family map.
- **F-X1:** transport identity gaps — frame vs conjugated, against the 0.10 band.
- **F-C1:** the A9 mechanism triptych (true / permuted / sign-flipped / within-cell-shuffled).

**On-disk lineage:** this prereg (frozen) → `2026-07-10_atlas_journal.md` (live) →
`2026-07-1X_atlas_result.md` → `figures/atlas/`. Registry updates at close: BOTH
`studies/lineage/RESEARCH_INDEX.md` and `studies/lineage/README.md`.

## 11. Pre-registered prediction grid (filled 1:1 at unblinding, whatever lands)

| # | quantity | author's prediction | result | verdict |
|---|---|---|---|---|
| P1 | R-T1 ratio_rt @ ε=0.2 | ≤ 0.15 (`H_radial-gauge`) | | |
| P2 | R-T2 ladder | ρ_freq ≥ 0.9, monotone, retention-curve fit within 2× (`H_ladder`) | | |
| P3 | R-T3 triple | K-coherent ≤ seed-noise; Q > V at matched energy | | |
| P4 | R-T4 leg A | L14 ratio within ±0.4 of √n_eff(L14), excludes the L0 constant | | |
| P5 | R-T4 leg B | terciles order with √n_eff, exact p ≤ 0.05 | | |
| P6 | R-B1 | τ_v1 ≥ 0.7, beats both comparators (`H_one-boundary-lead`) | | |
| P7 | R-B2 | recovered_frac ≥ 0.5 (`H_dormant-gate`) | | |
| P8 | R-W | Δfit ≥ 0.15 on the ppl proxy (`H_visible-variance`) | | |
| P9 | R-N | gap ≤ 0.10 (`H_maxent-sufficient`) | | |
| P10 | R-X | closure ≥ 0.5; gap_conjugated ≤ 0.10 (`H_connection`) | | |
| P11 | R-D | n=8 recovery within factor 2 at PR ∈ {1,4,32}; n=3 interval ≈ ±3× | | |
| P12 | R-M headline | Δρ_ppl CI-lo > 0 with ρ_M2 ≥ 0.7 (`H_metric`) | | |
| P13 | M3 vs M2 | M3 adds sealed skill iff P6 lands (`coupled prediction`) | | |
| P14 | C-1/C-2 | C-1 ≈ permuted (lattice mechanism); C-2 interpolates | | |

Misses stay visible and are scored as the information they are; B-grid misses in BISPEC were its
most productive rows, and this grid is written expecting the same.

## 12. Compute budget + references

**Compute (GPU-h and run-count only).** Gate-0 + L8 harvest ≈ 4 H100-h; Arm T ≈ 10; Arm B ≈ 4;
Arm W ≈ 6; Arm X ≈ 4; Arm M discovery fill-in ≈ 12; sealed battery ≈ 6. Total ≈ 46 H100-h across
≈ 8 pg jobs (p5.4xlarge first, g6e fallback), plus CPU r7i for D0/offline. **Curiosity reserve:**
C-1/C-2 and the d_eff family sweep are the protected slice — they run even if budget tightens,
because they are nearly free and answer the highest-leverage open mechanism questions (A9;
susceptibility dimensionality).

**References (cite-verify gate applies: claim-level check + quote location before any appears in
the result doc; all tagged consistent-with until verified otherwise):** RoFormer (rotary position
embedding construction — for T2's frequency convention); a matrix-Bingham/Statistics-on-manifolds
source (for T7's ensemble family); lazy-training/NTK sources already claim-checked in BISPEC Arm W
(reused at the same scope). No citation number is load-bearing in any §6 rule.

---

*Definition-of-done self-audit at authoring: decision named (§1); opposite predictions on one
readout (§1); MDE named and one-person simple (T-1); pre-mortems folded with responses (§9.4);
estimand 5-attribute (§2); falsification table incl. the author's favourites (§1); every §6 rule
has a ceiling or VOID and a failure branch the dry-run must construct (§9.1); controls trinity per
claim (§5); discovery/sealed split + frozen promotion (§3.4, §4); no baked baselines (recompute
clauses throughout); reuse ledger verified with locations (§0.5); weight-free dry-run + preflight
split (§9); journal + pre-unblind predictions (§9.3); figures tied to numbers + nulls (§10);
independence firewall + no-rule-swap + post-critic regeneration (§9.5–9.6, §0); freeze-tag-or-refuse
(§0). Every branch of every rule carries a lesson: nothing this program can observe is
uninformative.*
