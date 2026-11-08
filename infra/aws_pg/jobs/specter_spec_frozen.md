# SPECTER — pre-registration: measure the susceptibility operator itself — kernel, bulk, spikes, by sequence mode — and let it name the carriers, face a sealed rematch, and meet the conserved core

> **Instrument note (2026-09-03).** Blank-image MMLU numbers: the readout scores an off-format token
> surface in the VLM shell, but re-measurement on both surfaces gives identical accuracies, so nothing is
> retracted; the shell itself costs the healthy model ~0.16 accuracy vs text-only, which is the caveat that
> matters for interpretation. Crosscoder "features preserved" statements on the gemma-2→PaliGemma-2 pair
> are VOID (C6, FVU 8.6). See `studies/lineage/2026-09-03_INSTRUMENT_INCIDENT.md` and
> `studies/lineage/CLAIMS_LEDGER.md`.

**One sentence.** Every study since GRAIN has probed one latent object — the quadratic form mapping a
residual-stream perturbation to behavioural damage — with behavioural cells, one random projection
at a time; SPECTER computes that object directly, matrix-free, as a family of operators resolved by
sequence mode and site, and then asks three pre-registered questions of the measurement: does its
spike structure **name the lineage's unnamed carriers**, does it **predict never-run sealed cells
better than energy**, and does it **coincide with what fine-tuning preserves**.

**Truth clause (binding).** This program measures; it does not vindicate. The author's lead model
(M4, defined in §3.1) is labeled the **hypothesis-to-kill** and receives the most hostile treatment
the design can construct: the sealed cells are selected to maximize its disagreement with the
incumbent, its death branch (`H_energy-sufficient-again`) is written below as a full positive
result with its own lesson, and the falsification table lists M4's kill conditions first. Every
rule in §6 enumerates all of its branches, each with a **"what the program learns"** entry; a VOID
is an instrument-scoping measurement; a null is a boundary located, reported in the same register
as a positive.

---

## 0. Status + provenance + freeze mechanics

- **Status:** PRE-REGISTRATION, AUTHORED 2026-07-11 (not yet frozen). §0.5 amendments permitted,
  dated, until the first tag exists.
- **Freeze mechanics:** per-arm tags `specter-freeze-<arm>-<UTC-date>` on `science/specter`,
  recorded with tag + SHA + UTC. **Every driver asserts pre-launch that its arm's tag exists and
  that the spec sha256 baked into its config matches the tagged spec file; absent tag ⇒ refuse to
  launch.** No retroactive tags.
- **Branch/worktree:** `science/specter` from the `science/atlas` tip **recomputed at freeze**
  (`git rev-parse science/atlas`; expected c176cfb, never transcribed into a rule). Worktree
  `../ntm-frozen-gemma-specter`. Never the shared main worktree.
- **Model:** PaliGemma-1 (`google/paligemma-3b-pt-224`), 18 L, d = 2048, MQA (1 KV head,
  head_dim 256, rope_θ = 10⁴), **eager**; sha re-asserted at Gate-0 against the invcore-FUNC pin.
  All numbers PG1.
- **Parents (artifacts, never prose):** ATLAS (`2026-07-11_atlas_result.md`, `results/atlas_*`),
  BISPEC (`results/bispec_*`), REGRAIN, COMWRITE banks, conserved_core banks. Every number in a
  §6 rule is recomputed from its named artifact by the driver at run time.
- **Two ATLAS spec-promise repairs, made binding here:** (i) every `metrics.json` carries
  `gpu_seconds` per stage and the driver asserts the field is non-null before `green` is written;
  (ii) wherever a §6 family rule names BH, the driver **emits the per-comparison p-values** — BH
  may not be "satisfied vacuously."
- **Inherited discipline:** `green` is the last statement every job executes; drivers halt on
  nonzero STATUS; status flags computed after all writes; journal
  (`2026-07-11_specter_journal.md`) live from day one; predictions journaled before each
  unblinding step (§9.3).

## 0.5 Pre-freeze verification checklist

Code surfaces verified against the tree on 2026-07-11 (re-verify at freeze; lines drift):

| claim | verified at |
|---|---|
| `inherit_pg1.eval_pg1` (VQA soft-acc, scramble floor), `ppl_gate(n_seq=120, window=512)`, `harvest`, `load_pg1(sha=…)` | `infra/pg/inherit_pg1.py:35–223` |
| `bispec_core` REUSED: `accumulate_sigma`, `eig_bands`, `band_projector`, `envelope`, `haar_frames`, `principal_angles`, `n_eff_from_attention`, `saturation_guard`, `hook_at` | `infra/pg/bispec_core.py:82–1013` |
| `atlas_core` REUSED: `rmsnorm_jacobian`, `make_hook_radial/tangential`, `rope_freqs/rope_rotate/plane_vector/dirichlet_retention`, `delta_postrope_coherent`, `channel_delta_fn`, `GateCensus`/`make_gate_clamp_hook`, `d_eff_from_damages`, `calibrate_d_eff`, `make_hook_subband_noise/deletion`, `subband_pool` | `infra/pg/atlas_core.py:48–621` |
| `atlas_fit` (cell grammar, descriptor computation, isotone link, adversarial selection, sealed scoring) and `atlas_bingham` (exact Grassmann-Bingham sampler) REUSED | `infra/pg/atlas_fit.py`, `infra/pg/atlas_bingham.py` |
| Banked inputs: Σ banks L0/L14 (`results/bispec_sigma`), L8 Σ (ATLAS Gate-0e artifact), Ω profiles + input Grams (`results/bispec_w`), U_unt (conserved_core bank), U_wr (COMWRITE LOO banks), v₁ + σ₁ anchor (`results/bispec_a`, rebuilt as `regrain f_steer`), dormant-gate map (`results/atlas_b`), ρ task-visibility map (`results/atlas_w/rho_bank.pt`), 20-cell anomaly set + residuals (`results/atlas_m_sealed`), corpus manifest (`results/atlas_d0/atlas_corpus.json`), BISPEC draw seeds (`bispec_k.py` `manual_seed(11+1000·ss)`) | each grepped/read this session; executor re-verifies paths + hashes at freeze |

Executor items before tagging (each has a §9.4 response):

1. **Differentiation path.** Verify `torch.func` (`jvp`, `vjp`, `functional_call`) composes with
   the HF PaliGemma eager forward in the pg env; if brittle, the pure core's fallback path —
   manual double-backprop via `autograd.grad` twice — is the frozen alternative. Both paths are
   implemented in `specter_core`; Gate-0's finite-difference check certifies whichever is active.
2. **Memory smoke.** One GGN-vector product at full T = 120, microbatch 8, layers 0→17 graph
   (vision tower excluded from the differentiated graph), on the launch instance class.
3. **Vocab-scale Fisher product.** `H_out u = p⊙u − p(pᵀu)` at V ≈ 257k implemented streaming;
   verified against dense on a toy vocab in the D-suite.
4. **Corpus + library manifests** built offline, content-hashed; every exclusion counted.
5. **D-suite green** (§9.1), including all historical replays and every §6 failure branch
   constructed.

## Glossary (defined once, used everywhere)

| object | definition |
|---|---|
| **F̄** | the pullback Fisher/Gauss–Newton operator E_batch[JᵀH_out J] at a site, J = ∂logits/∂x; PSD; the susceptibility metric at quadratic order |
| **H_ω** | F̄ restricted to sequence-mode probes δ_t = env_ω(t)·v, one d×d operator per envelope ω ∈ Ω = {dc, sin1, sin4, white}; H_dc = the coherent-mode operator; H_white = the position-averaged diagonal-block operator (fixed seeded Rademacher sign-sets; two sign-sets for stability) |
| **spike** | an eigenpair of an H_ω whose eigenvector is stable across estimation halves and whose Rayleigh quotient generalizes to held-out data (rule R-O1) |
| **bulk / kernel** | the flat central spectrum / the near-zero slice predicted by the gauge theorems (radial family; counter-rotated key-coherent slice) |
| **u_dc** | the top stable eigenvector of H_dc at L0 (ppl loss) — the candidate carrier of the banked 110× coherent-draw spread |
| **r_quad** | per spike: the largest amplitude at which measured damage stays within 2× (on log-damage) of the quadratic prediction (α²/2)·λ |
| **Q(ΔW)** | the weight-space stiffness pullback tr(ΔW Σ_y ΔWᵀ F̄_z) — the K-FAC form assembled from banked input Grams Σ_y and measured F̄_z |
| **R(site)** | tr(H_dc)/tr(H_white) — the quadratic-order coherence dial |
| **M4 / M0** | damage predictors: M4 = frozen closed-form quadratic functional of {H_ω} composed with a 1-D isotone link (zero fitted directional parameters); M0 = isotone link of per-token energy alone |
| **d_eff** | 2/CV² of damages across draws, on the frozen log-damage convention (the ATLAS R12 lesson) — participation-ratio estimator of whatever form a draw family probes |

---

## 1. Mission tie-in, question, hypotheses, falsification

**Mission tie-in.** The program's charter is naming latent structure. Five anomalies now point at
one unnamed object — the direction-resolved structure inside bands (ATLAS R8), the coherent-spread
carrier (BISPEC R10b / ATLAS R5), the band1 catastrophe, the cross-site coherence dial (ATLAS R4),
and the exceptional directions already named (U_unt, v₁). SPECTER computes the object they live in.
The CORE⟂SPIKE arm ties it to the mission's central question: whether **what fine-tuning refuses
to touch** is **what behaviour cannot afford to move**.

**Question.** What is the measured quadratic structure of behavioural susceptibility at PG1
residual sites — kernel, bulk, and spikes, resolved by sequence mode and site — and does that
measured structure (a) name the lineage's unnamed carriers, (b) predict sealed never-run cells,
(c) coincide with the conserved core?

**Why measurement can succeed where sampling could not (registered mathematics).** For any
quadratic form and a uniformly random unit direction v, vᵀF̄v concentrates at tr(F̄)/d with O(1/√d)
fluctuations — so generic probes see only the trace (the energy law is a theorem of high
dimension, which ATLAS confirmed at sealed adversarial grade), and exceptional directions are
exponentially rare under sampling. Lanczos is the constructive inverse of that concentration: it
returns the extreme eigenstructure first, at a cost of one JVP + one VJP per iteration.

**Program-level hypotheses (opposite predictions on the primary readout and the MDE):**

- **H_quadratic-atlas** (the author's lead, hypothesis-to-kill): stable spikes exist, they name
  the carriers (R-C1 fires), and the measured operator predicts sealed cells (R-M1 Δρ > 0).
- **H_scale-separation:** the operator measures cleanly and its spikes are stable, but sealed
  prediction still belongs to energy and/or spikes fail behavioural validation — local quadratic
  geometry does not set damage at behavioural amplitudes.
- **H_field:** no stable average spikes exist (split-half fails) — the metric is item-conditional
  F̄(x), and the exceptional set moves with the token.
- **H_artifact** (the boring one): wiring or batch artifacts — killed at Gate-0 by the
  finite-difference check and at R-O1 by split-half/held-out discipline before any downstream arm
  is interpreted.

**Decision named (what changes per world).** H_quadratic-atlas ⇒ the successor programs target
the spike set constructively (CERT launches on the atlas). H_scale-separation ⇒ the program
pivots to boundary/global geometry with Arm H's witnesses as primary objects. H_field ⇒ the
per-item operator (token-field) becomes the primary object and fuses with the A9 axis. Different
worlds, different next actions — no outcome leaves the next step ambiguous.

**Falsification table (author's favourites first):**

| hypothesis | killed by |
|---|---|
| H_quadratic-atlas (favourite #1) | R-C1: u_dc fails to order the banked draws (τ < 0.33); or R-S1: promoted spikes fail behavioural validation against the Bingham null; or R-M1: Δρ CI-hi ≤ 0 |
| H_core-stiff (favourite #2, Arm K) | U_unt's stiffness quantile inside the matched-rank Haar band |
| H_scale-separation | M4 winning sealed (Δρ CI-lo > 0) with spikes behaviourally validated |
| H_field | any spike passing split-half ≥ 0.7 AND held-out Rayleigh ∈ [0.5, 2] |
| H_artifact | Gate-0 FD agreement + R-O2 kernel checks landing where the theorems require |

## 2. Estimands

**Primary estimand (ICH E9):**
- **Population:** never-run perturbation cells at the L0 residual site (hook = residual entering
  `layers[0]`, eager, all positions), drawn from the frozen grammar (ATLAS grammar + the new
  channel/radial productions, §3.4).
- **Treatment/contrast:** predicted damage rank under **M4** (measured-operator model) vs **M0**
  (energy-only), both frozen before the sealed battery.
- **Variable:** log(1+Δppl), blank-image wikitext, n_seq = 120 (primary); VQA soft-acc effect on
  the dead-floor convention (secondary).
- **Intercurrent events:** dead/saturated cells VOID by `saturation_guard` and are replaced from
  the pre-registered ordered backup list (30 registered; first 20 valid scored); every VOID
  logged and reported.
- **Summary measure:** Δρ = ρ_Spearman(M4, measured) − ρ_Spearman(M0, measured) over the 20 valid
  cells, 10k pair-bootstrap 95 % CI, per readout; BH q = 0.10 across the 2-readout family with
  emitted p-values.

**Instrument identity.** Readouts: `inherit_pg1.eval_pg1` (per-item, paired by item id under the
`specter|` salt) and `inherit_pg1.ppl_gate(n_seq=120)`. Operator: `specter_core.ggn_vp(slice,
batch, v, mode)` — NEW, pure — returning H_ω v for the frozen forward slice (site → logits, vision
tower excluded from the graph), with the batch manifest hash as the pairing key; `hessian_vp`
implemented beside it (gap measurement only).

**Secondary estimands (one line each; rules in §6):** top-64 eigenpairs per (site ∈ {L0, L8, L14},
ω ∈ Ω, loss ∈ {ppl-CE, vqa-CE}); kernel Rayleigh quotients for the two gauge families; per-band
traces (16 sub-bands + the 5 BISPEC bands); τ(banked-draw damage, ⟨δ̄,u_dc⟩²); R(site) triple vs
the two behaviourally measured coherence ratios; stiffness-quantile overlaps of {U_unt, U_wr, task
footprint}; r_quad per promoted spike; the hunt gap G and quadratic ratio; the 18-layer
dispersion-curve collapse statistic; d_eff of each H_ω spectrum.

## 3. Design

### 3.1 Frozen theory + model block

- **GGN = Fisher, pinned.** H_out at the logits is diag(p) − ppᵀ (model distribution), applied
  streaming at V ≈ 257k. F̄ is PSD by construction. The full-Hessian/GGN gap is *measured* (one
  Hutchinson trace pass), not assumed away.
- **Mode restriction.** Ω = {dc, sin1, sin4, white}, envelopes from `bispec_core.envelope`
  (frozen parent conventions). H_white uses two fixed seeded Rademacher sign-sets; agreement
  between them is a stability row, and Lanczos on each sign-set is exact for that realized
  operator (disclosed randomized-proxy construction).
- **Gauge-kernel predictions (theorems to re-verify at quadratic order):** the per-token radial
  family and the counter-rotated key-coherent 256-dim slice have near-zero quadratic forms; exact
  thresholds in R-O2, including the branch where the quadratic and finite-amplitude regimes
  disagree — pre-registered as the most interesting possible landing, not an error state.
- **M4, with zero fitted directions.** Each grammar production has a closed-form expected
  quadratic functional q(cell) computed from {H_ω} and the cell's own realized construction
  (examples, frozen in code: white-in-band-b at energy E → E·tr(H_white P_b)/k_b; a realized
  dc draw v → vᵀH_dc v; a shared-subspace removal with realized frame Q → Σ_t-normalized
  tr(QᵀH̃Q) with the deletion's realized coordinates). damage_pred = g(q) with g a **single 1-D
  isotone link** calibrated on the discovery corpus. M0 = g₀(per-token energy), same link class.
  The only fitted object in either model is a monotone scalar function; every directional claim
  in M4 is *measured*, not fitted. (This is the design repair of ATLAS's M2, whose band weights
  were fitted and lost.)
- **K-FAC pullback.** Q(ΔW) = tr(ΔW Σ_y ΔWᵀ F̄_z), Σ_y = the banked post-norm input Grams
  (`results/bispec_w`), F̄_z from Arm O. The independence approximation is validated against
  direct weight-space GGN products on a subsample before any Arm-K verdict (R-K1 gate).
- **Conventions frozen:** damage scale log(1+Δppl) everywhere; d_eff on log-damage; fp32 Lanczos
  vectors with full reorthogonalization over a bf16 forward (FD check bounds the composite
  error); construction-matched range references for every range guard (the ATLAS R-T1 lesson:
  a "seed-noise" floor must be measured on the same construction family it gates).

### 3.2 Arms

| arm | what it does | what it adds (nothing else does) |
|---|---|---|
| **Gate-0** | anchors (ppl anchor vs invcore-FUNC recompute; identity hook; model sha; MQA asserts); **FD wiring check** — GGN-vp directional value vs central finite differences of the loss gradient along 8 probe directions on a fixed micro-batch, rel. err ≤ 1e-2, on whichever differentiation path §0.5-1 selected; memory smoke at full T; gpu-h logging assert | proves the wiring on a known number before any operator claim exists |
| **Arm O** | Lanczos top-64 per (site, ω, loss), fixed batch A split A1/A2 + held-out batch B; kernel Rayleigh censuses (radial family, key-coherent slice); Hutchinson traces per band; GGN-vs-Hessian gap trace; d_eff of each spectrum | the object itself — kernel, bulk, spikes, by mode and site |
| **Arm C** | **C-1 (MDE):** u_dc vs the 9 regenerated banked coherent draws (τ, exact permutation p; the ppl leg adjudicated on its 6 independent pairs per the ATLAS R5 disclosure, the VQA leg on 9). C-2: band×mode trace tables vs the banked S3 reversal. C-3: R(site) vs the two behaviourally measured ratios (recomputed from `results/bispec_k_sealed` and `results/atlas_t` at run time). C-4: overlap census of stable spikes vs the frozen library {U_unt, U_wr, v₁, e₁, mean dir, γ dir, per-layer W_k/W_v rowspaces, dormant-gate weight vectors, band projectors, B\*-union, task footprint} against matched-rank Haar nulls | the naming: every unnamed carrier gets its identification test, mostly at zero forwards |
| **Arm K** | stiffness spectra of Σ_y⊗F̄_z per (layer, matrix-type); stiffness-quantile overlaps of U_unt/U_wr/task footprint vs matched-rank Haar bands; K-FAC fidelity gate via 32 direct weight-space GGN products at 2 layers | the mission unification — preservation vs stiffness — for the price of an integral |
| **Arm S** | top-8 promoted spikes (rule R-O1 → promotion frozen at k = 8) → behavioural cells ±α ∈ {0.05, 0.1, 0.2}·rms vs Bingham em-nulls + rotated controls, discovery items; r_quad per spike | the causal bar (#7): a spike is spectral structure until an intervention moves the readout past the matched null |
| **Arm H** | projected gradient ascent on battery damage at fixed energy (per-token field and coherent-v families; α ∈ {0.1, 0.2}; 16 restarts; 32-seq proxy objective, found directions re-measured on the standard battery — the re-measure is the number of record) | the non-quadratic exceptional set, constructively — or an empirical concentration certificate |
| **Arm M** | link calibration on the discovery corpus (banked cells + Arm S/T-class cells); freeze M0/M4 + adversarial selection output (30 ordered cells maximizing \|M4−M0\| rank disagreement under coverage caps) + per-cell numeric predictions journaled; sealed battery run **once** on the `specter|` sealed half; unblind once | the rematch: the measured operator vs the energy law, sealed |
| **Arm B** (curiosity, protected) | Σ harvest at all 18 layers (200k rows each, one job); level-set dispersion curves per layer (CPU); pairwise collapse matrix; hybrid-spectrum sensitivity (swap top-r between sites, recompute) to isolate the responsible spectral feature; whitening-exponent fit (F̄ spectrum vs Σ spectrum) | the R10 surprise (L8 = L14 to 0.015) becomes a mapped invariant or a counterexample map |

**The MDE is C-1.** One Lanczos run (H_dc, L0, ppl) plus a zero-forward regression against draws
regenerated from logged seeds. Its outcomes force different conclusions: carrier named (the
program's longest-standing unnamed object gets an identity) vs spike-exists-but-does-not-carry
(routes to H_field / Arm H per §6). One person, one GPU-day.

### 3.3 Ordering constraint

Arm O's L0 pass and C-1 run before any other GPU arm (the MDE gates nothing but informs
everything); Arm M's sealed battery is the **last** GPU event in the program; Arm B is
independent and may run anytime. Arms S, H, K, and the L8/L14 operator passes are mutually
independent.

### 3.4 Sealed-cell grammar

ATLAS's frozen grammar (add/delete × 16 sub-bands × {dc, sin4, white, shared-subspace,
per-token-subspace} × amplitude {0.1, 0.2}·rms) **plus** the channel productions (post-RoPE-K /
V / Q confinement via `channel_delta_fn`) and the radial/tangential split (`make_hook_radial/
tangential`), minus already-run combinations (the corpus), minus dry-run-degenerate combinations
(each exclusion counted). Selection: rank by |M4 − M0| predicted rank disagreement; coverage caps
(≥ 3 cells per major axis, ≤ 4 sharing a (band × mode) pair); top 30 ordered; algorithm frozen in
code; output + all four models' per-cell numbers journaled before any sealed forward.

## 4. Data

1. **Operator batches.** Batch A (split A1/A2) and held-out batch B: wikitext sequences (n_seq
   bank) and VQA items **from the discovery half only**; manifests content-hashed; seeds logged.
2. **Items.** VQA pool re-salted `specter|` (fresh discovery/sealed split; Gate-0 audits zero
   overlap of the new sealed half with any operator batch or discovery cell). Discovery cells:
   200 items. Sealed battery: 200 items, **touched once**.
3. **Corpus for link calibration.** `atlas_corpus.json` + ATLAS's 20 sealed cells (they were
   sealed under ATLAS's `atlas|` contract, which closed at its unblind; here they are training
   data, and this reuse is disclosed) + Arm S cells as they land. Inclusion rules unchanged
   (L0, n_seq = 120, guard-clean); every exclusion counted.
4. **Library manifest.** Every named object with artifact path + content hash + construction
   recompute clause (e.g., v₁ rebuilt with the σ₁ assert; U_unt rebuilt from the consensus Gram).

## 5. Controls (each gates a specific claim)

| control | gates | construction |
|---|---|---|
| **positive control (instrument)** | Lanczos believes nothing until it recovers known structure | live: the key-coherent kernel slice (a theorem-required near-zero family) must land where R-O2 requires; dry-run: planted-spike + planted-kernel synthetic operators recovered to tolerance |
| **split-half + held-out** | every spike claim | eigvecs from A1 vs A2 (overlap), Rayleigh quotients on B (generalization); near-degenerate clusters compared by principal angles on cluster spans (frozen eigengap clustering rule), not vector-by-vector |
| **envelope-shuffle null** | H_dc's spikes are dc-specific | Lanczos on a position-shuffled dc envelope (breaks coherence, keeps marginals): its top space must NOT reproduce u_dc (overlap below the Haar band) |
| **matched-rank Haar nulls** | every census overlap (C-4, K) | `haar_frames` bands at the same rank, p95 |
| **Bingham em-nulls + rotated controls** | every Arm S/H behavioural claim | `atlas_bingham` sampler at matched energy; same-vector-rotated for direction specificity |
| **construction-matched range references** | every range guard | the noise floor each guard uses is measured on the same construction family it gates (ATLAS R-T1 repair, encoded) |
| **saturation/over-shoot guards** | every cell | `saturation_guard` + the traversal over-shoot guard |
| **sign-set replication** | H_white claims | two Rademacher sign-sets; agreement row reported |

## 6. Metrics + frozen decision rules

Conventions: log(1+Δppl); 10k pair-bootstrap CIs; exact permutation for n ≤ 10; BH q = 0.10
within each named family **with emitted p-values**; every rule below has a ceiling or VOID and a
failure branch the D-suite must construct.

**R-O1 (spike existence; per site × ω × loss).** spike_i is *stable* iff split-half overlap
≥ 0.7 AND held-out Rayleigh ratio ∈ [0.5, 2.0] (two-sided by construction).

| branch | condition | the program learns |
|---|---|---|
| `H_spikes` | n_stable ≥ 1 | the operator has nameable extreme structure; Arms C/S proceed on the stable set |
| `H_field` | n_stable = 0, Lanczos converged (residuals logged) | the metric is item-conditional — the pre-specified per-item mini-battery runs (power iteration, 16 items, top-1 each; dispersion of u₁(item) is the finding) and the token-field probe is promoted from curiosity to named successor |
| `VOID-nonconverged` | Lanczos residuals above tolerance after the restart protocol (once, doubled reorthogonalization budget) | reported at reduced k with residuals; the spectrum estimate itself remains a deliverable |

**R-O2 (kernel/theorem checks; the checks designed to be able to fail).**
Key-coherent slice: mean Rayleigh ≤ 1e-3 × bulk-mean → theorem verified at quadratic order;
∈ (1e-3, 0.1) × bulk → a measured leak, quantified (the number is the finding — RoPE
counter-rotation precision and V-channel bleed are the named suspects, both measurable from the
same census); ≥ 0.1 × bulk → **fail-loud instrument halt** (the theorem is exact; this is wiring).
Radial family: ≤ 0.05 × bulk → gauge confirmed at quadratic order; ∈ (0.05, 0.25) → the skip-path
share is measured (feeds M4's radial production as a *measured* constant); ≥ 0.25 → 
`H_radial-active-quadratic` — the quadratic and finite-amplitude regimes disagree about the gauge
itself, pre-registered as the most interesting landing (scale-inconsistency becomes the successor's
opening question).

**R-C1 (the MDE; family = {ppl leg (6 pairs), vqa leg (9 pairs)}, BH q = 0.10, exact p emitted).**
Guard first: if ⟨δ̄,u_dc⟩² spans < 10× across draws → `VOID-no-range` — the draw family is
near-orthogonal to the spike axis, itself a measured alignment fact; the pre-specified alternative
regressor (top-8 spike-subspace energy ‖P₈δ̄‖²) then adjudicates instead.

| branch | condition | learns |
|---|---|---|
| `H_carrier-named` | τ ≥ 0.6 with BH-adjusted exact p ≤ 0.05 on either leg | the 110× spread has an identity; Arm K/S read it against the library; the program's #1 unnamed object closes |
| `H_spike-not-carrier` | u_dc stable, τ < 0.33 on both legs | the spread is off-average or non-quadratic — routes weight to Arm H and the H_field mini-battery; the spike remains real spectral structure, scoped |
| `H_partial` | between | reported with both legs; the alternative regressor's reading banked for the successor |

**R-C2 (sign-reversal at quadratic order).** band1: tr(H_white P_b1)/k_b1 ≥ 30 × median(vᵀH_dc v
over Haar v in band1) AND band4 reversed ≥ 3× → the S3 reversal is a quadratic-order fact with a
mechanism (the operator's mode structure); exactly one side reproducing → the failing side names
where non-quadratic structure lives; neither → the reversal is wholly non-quadratic — a clean,
large statement that bounds every quadratic account of the sequence axis.

**R-C3 (coherence dial).** Targets recomputed at run time from `results/bispec_k_sealed` (L0)
and `results/atlas_t` (L14). Both |R(site) − ratio(site)| ≤ 0.4 → the dial is explained at
quadratic order and **R(L8) is journaled as a registered prediction** for one confirmatory L8
battery (run within this program, pre-approved); one or neither within band → the miss pattern
localizes the non-quadratic transport component by site.

**R-C4 (census).** A library overlap is an identification iff ≥ 3 × matched-rank Haar p95. Any
identification → a named object acquires a mechanism seat. Zero identifications → the stable
spike set is **new structure unaligned with every object the lineage has named** — pre-registered
as an exciting branch: the set ships with its constructive handle (the eigvec bank) as the
successor's starting material.

**R-K1 (CORE⟂SPIKE; gated).** Gate: K-FAC fidelity ρ ≥ 0.8 between Q(ΔW) and direct weight-GGN
products on the 32-probe subsample; below → Arm K reports at direct-subsample scope only
(pre-specified de-scope, not a halt; the fidelity number itself is an instrument result about
K-FAC on this architecture). Then, with U_unt's mean stiffness quantile q̄ against the
matched-rank Haar band:

| branch | condition | learns |
|---|---|---|
| `H_core-stiff` | q̄ ≥ Haar p95, in the top quartile | fine-tuning preserves what behaviour cannot afford to move — curvature explains conservation; protect-the-stiff-set becomes a testable transfer rule |
| `H_core-sloppy` | q̄ ≤ Haar p05, bottom quartile | fine-tuning preserves the *free* directions — a flat-minima-flavoured inversion that re-reads the invariant-core literature; the ρ-map supplies the next mechanism candidate (gradient starvation) |
| `H_core-generic` | inside the band | conservation is not curvature-positioned at all — the strongest license yet for the data-side (visibility/starvation) account, with the ρ-map in hand |

U_wr and the task footprint fill the same table as secondary rows (family BH q = 0.10, p emitted).

**R-S1 (per promoted spike).** Behaviourally real iff the spike cell's damage exceeds the Bingham
em-null band (CI-lo > 0) AND the rotated control sits inside the band. Below-band → spectral-only
structure, honestly scoped (still a deliverable: the quadratic atlas marks it). r_quad reported
for every spike regardless — the amplitude map is a deliverable of its own. Family = 8 spikes ×
2 readouts, BH q = 0.10, p emitted.

**R-H1 (hunt).** G = found-max damage / Bingham-typical at matched energy (re-measured on the
standard battery); Q-ratio = found-max / quadratic prediction along the found direction.
`H_concentration-certified` iff G ≤ 2 (an empirical robustness-radius certificate — a positive
deliverable no perturbation study has issued); `H_quadratic-worst-case` iff G > 2 ∧ Q-ratio ∈
[0.5, 2]; `H_boundary-dominated` iff G > 2 ∧ Q-ratio > 2 (witness direction banked — the
non-quadratic exceptional set becomes constructive). `VOID-flat` iff the optimizer improves < 1.2×
over init on the proxy (the landscape is locally flat at this energy — reported as the tolerance
statement it is).

**R-M1 (primary; family {ppl, vqa}, BH q = 0.10, p emitted; ppl is the headline).**

| branch | condition (per readout) | learns |
|---|---|---|
| `H_operator-predicts` | Δρ CI-lo > 0 ∧ ρ_M4 ≥ 0.5 | the measured operator is predictive machinery from first principles — the surface at this amplitude is quadratic-plus-named-spikes, and CERT inherits an atlas that works |
| `H_energy-sufficient-again` | ρ_M0 ≥ 0.5 ∧ Δρ CI-hi < 0.1 | **scale separation established at sealed grade:** damage at behavioural amplitudes is not set by the exactly-measured local quadratic form — an unreachable conclusion by any cheaper route, and the strongest possible motivation for the boundary program (Arm H's witnesses become the primary objects) |
| `H_partial` | Δρ CI straddles 0, both ρ ≥ 0.5 | both carry overlapping structure; the per-descriptor residual decomposition maps which grammar axes the operator wins on — the successor's chart |
| `H_neither` | both ρ < 0.5 | anomaly set v2, now doubly characterized (descriptor + operator residual) — the best-instrumented open object the lineage has produced |
| `VOID-battery` | > 10 of 30 backups VOID | one labeled half-amplitude re-freeze (the single pre-specified re-freeze in the program) |

**R-B1 (bulk; registered predictions, then measurement).** Collapse iff pairwise max|Δ| ≤ 0.05
across layers 4–17; the peel-off boundary (first layer violating) is reported wherever it lands;
the hybrid-swap sensitivity table names which spectral feature moves the curve. Registered
predictions: collapse holds on 4–17; peel-off at L ≤ 3. Either landing yields the same deliverable
— the responsible feature — because the sensitivity analysis runs regardless.

**Curiosity rows (exploratory, no adjudication):** the whitening-exponent fit (F̄ spectrum vs Σ
spectrum; descriptive, with CI; the banked band-grain fits span α ∈ [0.1, 0.7], so the registered
expectation is that no single power law survives); d_eff of every H_ω spectrum beside the banked
behavioural d_eff values; the per-item u₁ dispersion mini-battery if H_field fires.

## 7. Statistical analysis plan

10k pair bootstraps (BCa where skewed); exact permutation distributions for n ≤ 10 (C-1's legs);
BH q = 0.10 within each §6-named family, per-comparison p-values emitted in `metrics.json`
(driver-asserted non-null); split-half + held-out for all operator claims; two sign-sets for
H_white; per-seed values banked for every stochastic cell; every directional effect reported as a
gap to its matched null with a CI, never to zero; log-damage everywhere with raw Δppl beside it.

## 8. Threats-to-validity budget

| threat | effect if unhandled | mitigation |
|---|---|---|
| double-backward memory at T=120 | Arm O cannot run at spec scale | §0.5-2 smoke gate; microbatch + checkpointing; torch.func ↔ manual-autograd dual path; g6e fallback with halved batch (manifests re-asserted) |
| GGN ≠ full Hessian | spikes mislabeled as curvature | gap trace measured (Hutchinson) and reported beside every spike table |
| batch non-representativeness | spikes are batch idiosyncrasies | A1/A2 split-half + held-out B Rayleigh; frozen thresholds |
| H_white randomized proxy | sign-set artifacts read as structure | two sign-sets, agreement row; Lanczos exact per realized operator, disclosed |
| near-degenerate eigenvalues | vector-overlap flaps on real structure | frozen eigengap clustering; principal angles on cluster spans |
| K-FAC independence approximation | Arm K over-claims | fidelity gate on direct products; de-scope branch pre-specified |
| link-calibration leakage | M4 flattered by its own corpus | link is 1-D isotone only; all directional content measured; sealed cells never touch calibration |
| proxy objective in Arm H | found "worst case" is proxy-specific | number of record is the standard-battery re-measure |
| bf16 forward under fp32 Lanczos | eigenpair noise floor | FD check bounds composite error; residual norms logged; thresholds set above the measured floor |
| range guards mis-referenced | the ATLAS R-T1 misfire repeats | construction-matched references frozen (§5); the misfire event replayed in the D-suite |
| item leakage | sealed contamination | fresh `specter|` salt; Gate-0 overlap audit incl. operator batches |
| pool dry / Xet wedge | idle burn / silent stall | pg launcher p5.4xlarge→g6e; never wait on a dry pool; `HF_HUB_DISABLE_XET=1` everywhere |

Residual limitations, disclosed: one model; operator claims are batch-averaged quadratic
statements at three sites; Arm K's Σ_y are the banked Grams (their harvest conventions inherited
and disclosed); Arm H is a lower bound on the worst case by construction.

## 9. Blind protocol + execution runbook (autonomous)

**Strict order.** Offline gate + D-suite → Gate-0 → Arm O (L0) → **C-1 (MDE)** → C-2..C-4, K
(offline) → S, H, O(L8/L14), B (independent) → M link + selection freeze + journaled predictions →
**M sealed (once)** → unblind → critic → registries.

1. **Offline gate (before any GPU):** corpus/library manifests; descriptor + q(cell) formulas on
   banked cells; selection algorithm; all adjudicators; **D-suite** — planted-spike/kernel Lanczos
   recovery; a planted item-conditional operator that must FAIL split-half (proving H_field
   reachable); streaming Fisher-product vs dense on toy vocab; every §6 branch constructed
   including each failure branch; historical replays: the ATLAS R-T1 range-guard misfire
   (now classified correctly under construction-matched references), REGRAIN R1's saturated
   em-null (→ VOID), the ATLAS R12 d_eff convention event (frozen convention reproduces the
   corrected banked values), the A3 monotone-proxy event (saturation-shaped proxies immune on
   rotated controls).
2. **Preflight (first GPU step):** Gate-0 — FD wiring check + anchors, each recomputed from its
   named artifact. Any miss halts the arm, journal first.
3. **Journal predictions before each unblind:** the §11 rows for an arm are copied into the
   journal, timestamped, before that arm's decisive read (C-1's regression; C-3's R(L8)
   prediction; K's overlap read; M's 30-cell prediction table).
4. **Pre-mortem responses:** OOM → microbatch/checkpoint ladder → g6e fallback; Lanczos
   nonconvergence → one restart with doubled reorthogonalization, then reduced-k report;
   torch.func brittleness → manual double-backprop path (both in tree at freeze); K-FAC gate
   fail → subsample scope; hunt divergence → frozen LR-halving schedule; L8 bank absent →
   pre-approved harvest job; draw regeneration mismatch → halt C-1, journal, adjudicate on the
   alternative regressor family only if the parent artifact confirms the mismatch is theirs
   (the ATLAS fp32/fp64 inconsistency is already documented). Any fully blocked arm closes with
   a certificate; the rest continue; Arm M runs on whatever discovery corpus exists, deficit
   disclosed.
5. **Unblind once; no rule swap; independence firewall** (no comparison system's number becomes a
   threshold — R-C3's targets are recomputed measurements, used as *comparison points*, never as
   pass/fail cuts on other claims).
6. **Critic-before-belief; post-critic regeneration** of every downstream figure and table.

## 10. Artifacts, logs, figure suite

Per-arm `out/metrics.json` (every cited number + per-stage `gpu_seconds`, asserted non-null);
the **operator bank** (eigvecs/eigvals fp32 per site × mode × loss, content-hashed — a first-class
deliverable other programs will consume); `controls.json`; per-cell parquet keyed by descriptor
hash + item id; `events.jsonl` + `run.log`; the artifact registry (figure/number → script + input
hashes + DISCOVERY/CONFIRMATORY tag). All under `s3://pg-lineage-769063704696/results/specter_*`.

**Figure suite** (each: claim in the title backed by number + CI + null; one-line "how to read";
detailed caption; null overlaid; provenance footer; PNG + PDF; colourblind-safe; full-sweep rule —
every swept point and every component curve):

- **F-O1 (headline of Arm O):** the spectrum panorama — eigenvalue ladders per (site × mode),
  bulk band, kernel bars, split-half overlay.
- **F-O2:** gauge-kernel Rayleigh censuses (log scale) vs bulk mean — the theorem checks as bars.
- **F-C1:** the carrier regression, both legs, τ + exact p, draw labels.
- **F-C2:** the band × mode trace heatmap (the quadratic bi-spectral surface) with the banked
  behavioural cells overlaid at their coordinates — measured theory vs measured behaviour on one
  canvas.
- **F-C3:** the coherence dial — R(site) vs behavioural ratios, both bands drawn.
- **F-C4:** census matrix (spikes × library) with Haar-null bands.
- **F-K1:** stiffness-quantile overlaps for U_unt/U_wr/task footprint with Haar bands.
- **F-S1:** per-spike amplitude scans — all 8 spikes, quadratic prediction band, em-null band,
  rotated control, r_quad marked.
- **F-H1:** the worst-vs-typical frontier at matched energy, with the quadratic prediction line.
- **F-M1:** sealed parity, M4 and M0 side by side, per-cell descriptor glyphs, Δρ + CI.
- **F-B1:** the 18-layer dispersion-curve fan + pairwise collapse matrix + hybrid-swap
  sensitivity panel.
- **F-G1:** whitening scatter (F̄ eigenvalues vs Σ eigenvalues, log–log), fitted exponent + CI.

On-disk lineage: this prereg (frozen) → `2026-07-11_specter_journal.md` (live) → result doc →
`figures/specter/`. At close, update BOTH `studies/lineage/RESEARCH_INDEX.md` and
`studies/lineage/README.md`.

## 11. Pre-registered prediction grid (filled 1:1 at unblinding; author credences attached)

| # | quantity | author's prediction (credence) | result | verdict |
|---|---|---|---|---|
| P1 | R-O2 key-coherent slice | ≤ 1e-3 × bulk (high) | | |
| P2 | R-O2 radial family | ≤ 0.05 × bulk (med-high) | | |
| P3 | R-O1 at (L0, dc, ppl) | ≥ 1 stable spike; d_eff(H_dc spectrum) ≤ 3 (med-high) | | |
| P4 | **C-1 (MDE)** | τ ≥ 0.6, carrier named (med; favourite #1 — hostile treatment noted) | | |
| P5 | R-C2 | both sides reproduce (med) | | |
| P6 | R-C3 | both sites within ±0.4 (low-med; the sharpest single test in the program) | | |
| P7 | R-C4 | u_dc identifies with the mean-dir/γ family; ≥ 1 spike identifies with U_unt (low-med) | | |
| P8 | R-K1 | `H_core-stiff` (med-low; favourite #2) | | |
| P9 | R-S1 | ≥ 5 of 8 spikes behaviourally real (med) | | |
| P10 | r_quad | < 0.2·rms for the majority of spikes (med-high) | | |
| P11 | R-H1 | G > 2 with `H_boundary-dominated` at α = 0.2 (med) | | |
| P12 | **R-M1 headline** | Δρ CI-lo > 0 with ρ_M4 ≥ 0.6 (med — genuinely uncertain after ATLAS; the rematch is honest) | | |
| P13 | R-B1 | collapse on 4–17; peel-off at L ≤ 3 (med-high) | | |
| P14 | whitening | no single exponent survives (med) | | |

Misses stay visible and are scored as information; the last two programs' most productive rows
were their misses, and this grid is written expecting the same.

## 12. Compute budget + references

**Budget (GPU-h and run-count only; realized per-stage gpu_seconds logged by mandate).**
Gate-0 ≈ 2 H100-h; Arm O ≈ 8 (≈ 3,600 matvecs across 3 sites × 4 modes × 2 losses + traces +
kernel censuses); Arm S ≈ 5 (~100 cells); Arm H ≈ 6; Arm K ≈ 2 (32 direct products + integrals);
Arm M ≈ 8 (discovery fill-in + 21–30 sealed cells); Arm B ≈ 3 (18-layer harvest; curves are CPU);
margin ≈ 6. **Total ≈ 40 H100-h across ≈ 9 pg jobs** (p5.4xlarge first, g6e fallback), plus r7i
CPU for offline arms. **Protected curiosity slice:** Arm B and the whitening/d_eff rows run even
if the budget tightens — they are cheap and answer standing open questions (R10's invariant; the
sensitivity-scaling law).

**References (cite-verify gate: claim-level check + quote location before any appears in the
result doc; all consistent-with until verified; none load-bearing in any §6 rule):** K-FAC
(Martens & Grosse) for the Kronecker pullback form; the deep-Hessian spectral literature
(Lanczos/stochastic estimation of Hessian spectra in deep nets — Papyan; Ghorbani et al.) for
bulk-and-spikes phenomenology; GGN/Fisher equivalence for CE (Martens); RoFormer (already
claim-checked in ATLAS, reused at the same scope).

---

*Definition-of-done self-audit at authoring: decision named per world (§1); hypotheses opposite
on one sealed readout + one MDE readout (§1, §3.2); MDE named and one-person simple (C-1);
pre-mortems folded with responses (§9.4); estimand 5-attribute with CI (§2); falsification table
leads with the author's favourites (§1); every §6 rule carries a ceiling or VOID plus a failure
branch the D-suite must construct (§9.1); control trinity per claim (§5); discovery/sealed split
with frozen promotion k = 8 and a fresh salt (§3.2, §4); no baked thresholds — every parent
number recomputed from its artifact (§0, §6); reuse ledger verified with locations (§0.5);
weight-free dry-run separated from the runtime preflight (§9.1–9.2); journal + pre-unblind
predictions (§9.3); figures tied to number + CI + null with reading guides (§10); independence
firewall, no-rule-swap, post-critic regeneration, recompute-don't-snapshot (§9.5–9.6, §0); freeze
tag-or-refuse per arm (§0); GPU-h logging and emitted p-values made binding (§0). Nothing this
program can observe is uninformative: each branch of each rule names what the truth just taught
us.*
