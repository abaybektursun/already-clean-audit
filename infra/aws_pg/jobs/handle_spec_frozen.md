# HANDLE — pre-registration: from map to handle — the compact-core ladder, the skill-module demonstration, the compression-benignity mechanism, the two unnamed objects, the boundary set, and a verdict-level second-model replication

**One sentence.** CARRY established that the direction-specific carrying of fine-tuning's write
lives in a 679-dimensional stiff subspace holding only chance-level write energy; HANDLE asks the
question that decides whether that map becomes a tool — **does a much smaller core inside the
stiff space carry most of the skill, and is that core a portable, task-specific module?** — and
around that MDE runs the four highest-leverage open hunts the lineage now owns: the mechanism of
compression-error benignity (the operator's named blind spot), the identities of the two unnamed
objects (the coherent-spread carrier and the L0 spikes), the structure of the boundary/witness
set, and a scoped, verdict-level replication of the core laws on a second frozen public model.

**Nature-of-work clause.** This is open interpretability science on frozen, publicly released
checkpoints (PaliGemma family), with pre-registered methods, artifacts on durable storage, and
results reported whichever way they land. Arm B studies the banked worst-case directions as
*objects to be named* — what they are, how many there are, what they overlap — because a model's
extreme-sensitivity structure is exactly the kind of latent structure this program exists to
name. No arm targets deployed systems, and every experiment runs against our own frozen weights.

**Truth clause (binding).** Every §6 rule enumerates all branches with a **"what the program
learns"** entry; VOIDs are instrument-scoping measurements; null landings are boundaries located,
reported in the same register as positives. The author's favourites are flagged in §11 and each
has its killer named in §1. Every rank/correlation statistic defines its degenerate-input
completion in this document (all-tied := 0; n < 3 → VOID) — standing law since SPECTER E16.
**Rule-shape inheritance (the CARRY R-C lesson, now law):** every primary decision statistic in
this spec is **null-referenced** (measured against its matched random band), never a raw
fraction — CARRY's frozen rule scored raw gain fractions and could not express the two-axis truth
its own pre-registered control revealed; HANDLE's rules adjudicate on band-relative statistics
from the start.

---

## 0. Status + provenance + freeze mechanics

- **Status:** PRE-REGISTRATION, AUTHORED 2026-07-13 (not yet frozen). §0.5 amendments permitted,
  dated, until the first tag.
- **Freeze mechanics:** per-arm tags `handle-freeze-<arm>-<UTC-date>` on `science/handle`; every
  driver asserts tag existence + spec sha256 match pre-launch and refuses to launch otherwise. No
  retroactive tags.
- **Branch/worktree:** `science/handle` from the `science/carry` tip **recomputed at freeze**
  (`git rev-parse science/carry`; expected e544b86, never transcribed into a rule). Worktree
  `../ntm-frozen-gemma-handle`; symlink `.pgvenv` from the main repo before any `pg run`.
- **Model:** PaliGemma-1 (`google/paligemma-3b-pt-224`, 18 L, d = 2048, MQA 1 KV head), **eager**;
  sha re-asserted at Gate-0 against the invcore-FUNC pin. Arm X's second model is named in §3.2
  with its own §0.5 verification item.
- **Parents (artifacts only):** CARRY (`results/carry_*`: the sealed per-FT swap-shell rows, the
  dense stiffness eigendecomposition in `carry_scope`, the massive-coordinate census 738/674/604/
  2015, the error-family battery + Lᵖ exponents, the head census), SPECTER (`results/specter_*`:
  operator bank, witness bank `witness_directions.pt`, top-8 spike eigvecs, banked
  spike-subspace-energy regressor), BISPEC (draw seeds, Σ banks, h̄), COMWRITE (LOO banks,
  swap-shell provenance), the literature registry `2026-07-12_lit_registry.md`
  (**verified-complete** as of CARRY Arm L; the ONLY permitted citation source — S-numbers below
  refer to it). Every number in a §6 rule is recomputed from its named artifact at run time.
- **Binding process inheritances (the full standing set):** per-stage `gpu_seconds` asserted
  non-null before `green`; BH families emit per-comparison p-values; `green` is each job's last
  statement; drivers halt on nonzero STATUS; construction-matched range references for every
  range guard; one FT snapshot on disk at a time, deleted after use; `HF_HUB_DISABLE_XET=1` in
  every job env; journal (`2026-07-13_handle_journal.md`) live from day one; predictions
  journaled before every unblind; independence firewall — no external or cross-lineage number
  becomes a threshold.

## 0.5 Pre-freeze verification checklist

Code surfaces verified against the tree on 2026-07-13 (re-verify at freeze; lines drift):

| claim | verified at |
|---|---|
| Swap-shell REUSED: `project_delta(dW, U, mtype)`, `build_decoder(cell, base_dec, ft_dec, bases, …, excluded, seed, roll)`, `is_target`, `grid_of`, `assert_full_grid` | `infra/pg/comwrite_surgery.py:38–108` |
| Readouts REUSED: `inherit_pg1.eval_pg1` (per-item soft-acc, scramble dead floor), `ppl_gate(n_seq=120)`, `load_pg1(sha=…)` | `infra/pg/inherit_pg1.py:35–223` |
| CARRY banks: `carry_scope` dense Q eigendecomposition (2048², fp64; stiff = top-679 @ 50 % trace); per-FT sealed rows with `range_ok`, per-cell `{ft,stiff,flat,randA@i,randB@i}` gains + `injected_fro2`; task-assignment table (each FT's gain task); massive-coordinate census | `results/carry_{scope,c_sealed,c_assign,gate0}` (paths + hashes re-listed at freeze) |
| SPECTER banks: witness directions + their per-witness G rows; top-8 spike eigvecs + Rayleigh terms; the banked-but-unadjudicated top-8-subspace-energy regressor for the coherent draws | `results/specter_{h,o,c}` |
| Error-family constructions REUSED: quant/−e(x)/quantile-shuffle/permute/gauss/rotated + the CARRY Arm-E family manifest with seeds and realized stats | `infra/pg/atlas_core.py`, `infra/pg/bispec_core.py`, `results/carry_e` |
| Literature registry verified-complete; S-numbers stable | `research/2026-07-12_lit_registry.md` |

Executor items before tagging (each has a §9.4 response):

1. **Q eigendecomposition reuse.** Load `carry_scope`'s banked eigvectors directly (never
   re-derive); assert the top-679 half-trace split reproduces CARRY's stiff/flat projectors to
   fp tolerance on 3 sampled (layer, matrix) pairs.
2. **Ladder projector algebra.** `project_delta` onto nested top-k slices must satisfy
   P_k ⊂ P_{k'} for k < k' (nesting assert) and P_679 ≡ CARRY's stiff projector bit-for-bit.
3. **CARRY FT roster.** Recompute the 6 discovery / 12 scored-sealed FT lists and each FT's
   assigned gain task from `carry_c_assign` + `carry_c_sealed` `range_ok` flags — mechanical,
   no discretion.
4. **Second-model inventory (Arm X).** Verify on S3/Modal storage: PaliGemma-2-3b-pt-224 weights
   (the pg-lineage volumes hold PG2-era artifacts per the banked manifests) and whether any PG2
   relearn adapters with measurable task gains exist. Record what exists; Arm X's write-side
   bonus row runs only if adapters verify, else it is disclosed absent. Fallback second model if
   PG2 weights unavailable: the PG1 text stack evaluated text-only (weakest form, disclosed).
5. **SigLIP-side library (Arm N-b).** Verify the vision-tower projector weights and patch-grid
   geometry (16×16 → 256 image tokens) are readable from the checkpoint for the harmonic
   library construction; dry-run the 2-D harmonic basis on tiny tensors.
6. **SAE decoder atoms (Arm Q property 2).** Verify the banked m=1 L0 dictionary
   (GRAIN/REGRAIN lineage) loads and its decoder atoms are extractable; if unusable, property 2
   falls back to the BISPEC contrastive-band construction (pre-named, frozen at §0.5 close).
7. **D-suite green** (§9.1) including every §6 failure branch and the degenerate completions.

## 0.5.1 Pre-freeze amendments (dated 2026-07-13; folded into this frozen copy)

Decided under §0's "amendments permitted, dated, until the first tag" clause; full rationale +
provenance in `2026-07-13_handle_journal.md`. Each is a degenerate-input completion or a
spec↔artifact reconciliation, and names what it changes. The offline gate (§9.1) is GREEN
(D-suite 27/27; real-data ladder cap_frac .003→.352 monotone; core_{k_stiff} reproduces CARRY's
banked stiff-cell energy to rel_err 7e-9).

**Foundation / MDE.**
- **H-A1** k_half(FT) := None when gain_frac(core_{k_stiff}) ≤ 0 (the stiff install does not carry);
  R-H medians over defined k_half and counts the None's (n_khalf_void), never imputing.
- **H-A2** §4.1's "zero overlap with prior sealed exposures" ≡ SPLIT-INDEPENDENCE: the handle|
  sealed mask agrees with each prior salt's sealed mask at 0.4–0.6 (chance), asserted per task.
- **H-A3** Gate-0 swap-shell positive control = identity-cell bitwise base (delta 0) AND one banked
  CARRY sealed `stiff` cell (= core_{k_stiff}) score reproduced to 0.01 (cross-hardware bf16);
  reproduction FT = the first scored-sealed FT (aokvqa-da), its assigned task.
- **H-A4** corpus-task (coco_captions) gain SE = the carry_c CI-combination convention (verbatim).
- band^stiff at k==k_stiff = the full stiff space (= core, tautological): band_basis returns the
  full pool, handle_h skips those 3 evals and sets band_stiff := gf_core; nesting tolerance = 1e-5
  (fp32-banked Q eigvecs; genuine non-nesting is O(1)).

**Arm T.** **H-A5** core-census directions = top-r (r=min(16,K*)) left-singular vecs of the projected
core ΔW at (0, o_proj); the matched-rank Haar null is drawn WITHIN the K*-dim stiff subspace (all
cores share span(U_stiff); the question is same-directions-beyond-chance). **H-A6** 5/6 discovery FTs
share task textvqa → the module matrix is a demonstration at n=6 (disclosed); the Haar-core floor
row anchors each column.

**Arm Q.** **H-A7** the 4 zero-forward benignity properties: dependence_fraction (mean_t cos² to the
token's own quant residual), feature_alignment (S3; energy on the token's top-32 active SAE atoms),
massive_coord (S17; energy on the L0 massive coords, recomputed), incoherence (S18; max|·|/rms).
≥12 families = CARRY's 8 (reconstructed bit-for-bit) + 5 δ-mixture rungs; off-token = permuted quant
residual (energy-matched); rotated twin = Haar-rotated off-token.

**Arm N.** **HN-a1** the two "9-draw" objects are DISTINCT (the reference_fperp_two_objects
discipline): R-N-a runs on the BISPEC g-sweep (110× spread; all 5 candidates vary), `spike_p8`
recomputed there (**HN-a2**); the literal banked reg_p8 (SPECTER c1_rows) is reported as P7
continuity, outside the BH family. **HN-a3** n_eff_interaction := (‖shift‖/rms)·|cos(·,e₁)|. SigLIP
library = image_mean + patch_harmonics(16) + projector_rowspace(16) + embedding_pca(16) +
massive_coords(4); census rank-16 vs matched Haar p95; causal cell remeasures the spike's own
operator Rayleigh (≥50% drop).

**Arm B.** **B-1/1b** witness direction set = 8 (2 coherent vecs + 2 field × top-3 SVD); the angle
spectrum fed to R-B uses the 4 top-directions (the field-SVD dirs are orthogonal by construction).
**B-2c** d_eff = the 4-witness family; with |W|=4 the d_eff≤10 clause is NON-BINDING → the angle-vs-Haar
clause carries H_low-dim-surface (disclosed). **B-3** the 8 install-tracking probes = CARRY B2's
κ-family (distinct from W, else degenerate alignment≈1). **B-4** census overlap = ‖EᵀW‖²_F/rank(W)
(heterogeneous element ranks) vs matched-rank Haar.

**Arm X.** **X-1** (§0.5-4) PG2-3b-pt-224 base seeded; NO per-task PG2 FT adapters exist (verified
across S3/HF/tree) → the write-side stiff/flat row is a disclosed VOID; mix-224 noted as a successor
write source. **X-2** census depths resolved dynamically [0, N//2, N−3] (PG1's 0/8/14 not
transferable to Gemma-2's 26 layers). **X-3** (GQA) the post-RoPE-K coherent slice replicated across
num_key_value_heads (delta_multihead; PG2 k_proj = 4×head_dim). **X-4** attn_logit_softcapping is the
key-gauge theorem-breaker; Leg C maps its ratio to a label and does NOT halt (unlike PG1, where the
theorem is exact) — a firewall-aware land-elsewhere. Every PG2 threshold is self-calibrated on PG2
control data; the `pg1` field is a categorical label from frozen PG1 docs, touching only the
`pg1==pg2` equality. TR_EPS_FRAC=0.25 (leg A amplitude); a both-regimes-saturated level VOIDs.

## Glossary (defined once)

| object | definition |
|---|---|
| **Q / stiff space** | CARRY's stiffness metric (SPECTER gated spectral completion at (L0, dc, ppl), fidelity 0.927) and its top-679 eigvectors (50 % of tr(Q)); `carry_scope` banked |
| **core_k(FT)** | `project_delta` of that FT's ΔW onto the top-k Q-eigvector slice (nested: k ∈ K_LADDER) |
| **gain_frac_X** | (swap-shell gain of base + ΔW_X) / (gain of base + ΔW_full), per FT, on that FT's CARRY-assigned task |
| **band_k** | the matched-rank random reference at rank k: max over 3 seeded draws of gain_frac for a rank-k random-eigenspace projection of the SAME ΔW (the CARRY null, per rung) |
| **band_k^stiff** | the within-stiff reference: same, but the k directions are drawn randomly FROM the 679-dim stiff space — separates "top-of-the-ordering matters" from "any stiff directions suffice" |
| **k_half(FT)** | smallest k on the ladder with gain_frac ≥ 0.5 · gain_frac(679) |
| **excess_k** | gain_frac(core_k) − band_k — the null-referenced primary statistic (rule-shape law) |
| **module matrix M** | M[i,j] = swap-shell gain of base + core_K*(FT_i) evaluated on task_j, K* frozen in §6 R-H |
| **dependence fraction δ** | for the Arm-Q mixture ladder: the fraction of the error vector that is the token's own quantization residual, the remainder being the matched off-token construction; δ ∈ {0, 0.25, 0.5, 0.75, 1} |
| **witness set W** | SPECTER Arm-H's banked adversarial directions with their standard-battery damage rows |
| **S-numbers** | entries in `2026-07-12_lit_registry.md` (S3 = super-quadratic feature directions; S10 = Dom/Bulk SGD; S17 = massive activations; S18 = rotation gauge/QuaRot; S2 = κ-family; S6 = intruder dimensions) |

---

## 1. Mission tie-in, question, hypotheses, falsification

**Mission tie-in.** The mission bar is predictive ∧ novel ∧ practical. The lineage has banked the
first two repeatedly; CARRY's closeout graded its mechanism finding DIAGNOSTIC because a 679-dim
subspace is a map, not a handle. HANDLE is the program pointed directly at the third leg: a
compact, portable, task-specific core would be the first result in the lineage to clear all three
at once — and every fallback landing (ordered-but-broad, flat-within-stiff, non-portable) is a
mechanism fact about how fine-tuning stores skill, which is the mission's central question.

**Questions.**
1. (MDE) Does a small core (k ≤ 50) inside the 679 stiff directions carry most of the
   direction-specific gain — and is the Q-eigenvalue ordering load-bearing, or is stiffness a
   set property?
2. Are per-FT cores task-specific portable modules (diagonal dominance of the module matrix),
   and do they overlap each other / U_wr / U_unt beyond matched-rank chance?
3. What property of a structured activation error predicts its behavioural benignity, given that
   the global operator does not (CARRY R9, ρ = 0.238)? Candidates raced under a frozen rule,
   with a dose-response ladder for the causal leg.
4. What are the two unnamed objects — the coherent-spread carrier (three candidates excluded)
   and the L0 spikes (matched nothing in the library)?
5. What is the structure of the witness set — its dimensionality, its overlaps, and whether
   targeted installability tracks it?
6. Do the core laws replicate at verdict level on a second frozen public model?

**Program hypotheses (opposite predictions on the MDE's primary readout, excess_k at small k):**

- **H_compact** — a small core carries: excess_k large at k ≤ 50, and top-k beats
  band_k^stiff (the ordering is load-bearing).
- **H_ordered-broad** — the ordering is real (top-k beats band_k^stiff) but the gain accrues
  slowly: k_half > 150. The skill is stored across the stiff space with a graded profile.
- **H_set-property** — top-k does NOT beat band_k^stiff at any rung: any k stiff directions do
  as well as the top k. Stiffness is a set property; the eigenvalue ordering carries nothing.
- **H_no-range** (the boring one) — the ladder's low rungs sit inside seed noise on too many
  FTs; killed or confirmed by the range guard before any sealed exposure.

**Falsification table (killers named in advance; author's favourites first):**

| hypothesis | killed by |
|---|---|
| H_compact (favourite #1) | median k_half > 150, or excess_50 CI-lo ≤ 0, or top-k inside band^stiff at every rung |
| H_dependence-mechanism (Arm Q favourite #2) | the δ-ladder dose-response non-monotone, or dependence fraction ranks families no better than the label-shuffle null |
| H_modules (Arm T) | module matrix diagonal dominance ≤ chance (frozen count rule), or cross-installs indistinguishable from own-installs |
| "cores are new objects" | core overlap census: cores ≡ U_wr slices at matched rank (then the compact core IS the consensus write, a unification, and COMWRITE's object gains a function) |
| H_witness-low-dim (Arm B) | pairwise principal angles among witnesses indistinguishable from Haar (the witness set is generic scatter) |
| verdict-replication (Arm X) | any core-law verdict flipping on model 2 — reported as the boundary of the law's scope, which is exactly what a replication arm is for |

**Decision per world.** H_compact → Arm T's module demonstration becomes the bar-clearing
result and the successor is an application program (protection/merging/deletion on real
workflows). H_ordered-broad → the storage profile (gain vs k curve) is the object; successor
asks what sets its shape. H_set-property → stiffness-as-set joins the energy law as a second
"the label is the level set, not the coordinates" result — and the search for the functional
ordering moves off the Q-spectrum. Every world names its next step.

## 2. Estimands

**Primary (ICH E9):**
- **Population:** the 12 CARRY-scored sealed fine-tunes (roster recomputed mechanically, §0.5-3),
  each on its CARRY-assigned gain task, items re-salted `handle|`.
- **Treatment/contrast:** swap-shell installs of core_k(FT) for k ∈ K_LADDER = {5, 10, 20, 50,
  150, 350, 679}, against band_k and band_k^stiff (3 seeds each per rung).
- **Variable:** per-FT task gain via `eval_pg1`, paired by item id; ppl banked as collateral.
- **Intercurrent events:** per-FT range guard as in CARRY (full-write gain ≥ 3× construction-
  matched seed noise, recomputed fresh under the `handle|` salt); a rung whose install saturates
  a guard is VOID for that FT, counted. If > 6 of 12 sealed FTs VOID entirely, the verdict is a
  scope statement about shell range.
- **Summary measure:** median excess_50 (the null-referenced statistic at the pre-named rung)
  with 10k FT-level pair-bootstrap 95 % CI; plus median k_half with bootstrap CI; family
  {excess_50 vs band, excess_50 vs band^stiff} BH q = 0.10, p-values emitted.

**Instrument identity:** `comwrite_surgery.project_delta` + `build_decoder` (installs),
`inherit_pg1.eval_pg1` (readout, eager forward), Q eigvectors loaded from `carry_scope` (never
re-derived). Degenerate completions: all-tied rank stats := 0; bootstrap over < 3 valid FTs →
VOID; a band with zero variance across seeds triggers the seed-count doubling response (§9.4).

**Secondary estimands (one line each; rules in §6):** module-matrix diagonal dominance count and
cross-install gap (Arm T); core-overlap census values vs matched-rank Haar (Arm T-c); the
property-race Spearman per candidate + the δ-ladder monotonicity statistic (Arm Q); carrier
regression τ per candidate feature (Arm N-a); L0-spike library-census max overlaps + the causal
ablation drop (Arm N-b); witness-set d_eff and principal-angle spectrum + install-tracking ρ
(Arm B); the verdict-agreement vector on model 2 (Arm X).

## 3. Design

### 3.1 Arms (ordered as run; each names what only it adds)

| arm | what it does | forwards | anchors |
|---|---|---|---|
| **Gate-0** | ppl anchor; swap-shell preflight reproducing one banked CARRY sealed cell bit-for-bit; Q-projector nesting asserts (§0.5-2); `handle|` salt audit; range-guard recalibration under the new salt | 1 small battery | CARRY R1 |
| **Arm H** (MDE + confirmatory) | the ladder: 7 rungs × {core, band ×3, band^stiff ×3} per FT; discovery on CARRY's 6 discovery FTs (rule rehearsal); sealed on the 12, **once**; per-FT gain-vs-k curves with both bands | ≈ 12 FTs × 7 × 7 cells (sealed) + discovery | S10, CARRY R2 |
| **Arm T** | the demonstration: (a) module matrix M on the 6 discovery FTs (6×6 cross-installs at K*); (b) deletion selectivity: FT_i minus core_i on task_i vs on ppl + one non-target task; (c) core census: pairwise principal angles among cores + overlaps vs U_wr/U_unt/task footprints, all against matched-rank Haar bands | ≈ 50 cells | COMWRITE, S6 |
| **Arm Q** | the benignity mechanism race: 12+ error families (CARRY's 8 + the δ-mixture ladder at 5 rungs) × 4 candidate properties, each property a frozen zero-forward computation on the error bank; damage measured once per family; the δ-ladder is the causal dose-response leg | ≈ 1 battery | S3, S17, S18, CARRY R9 |
| **Arm N** | naming round 2: (a) carrier regression on the banked 9 coherent draws vs 5 new candidate features (all zero-forward; exact permutation); (b) L0-spike census vs the SigLIP-side library (patch-position 2-D harmonics, image-mean direction, projector rowspace, embedding-cluster axes, massive-coordinate axes) + ONE causal cell ablating the best-matching structure | (b) ≈ 1 small battery | SPECTER R5/C4, CARRY R4 |
| **Arm B** | the witness set as an object: pairwise principal-angle spectrum + d_eff of W; overlap census vs {spikes, sink coords, v₁, U_wr, Q-stiff space}; per-witness dose-response (5 amplitudes × 3 witnesses); targeted-install tracking — does install success (CARRY B2's 0.43-vs-0.22 instrument) correlate with witness-subspace alignment across 8 probe directions | ≈ 1 battery | SPECTER R10, CARRY B2, S2 |
| **Arm X** | verdict-level replication on model 2 (PG2-3b per §0.5-4): Gate-0-class anchors; operator pass (Lanczos top-32 at the embedding-output site, both losses); kernel theorem checks; a 20-cell two-regime battery (energy law across levels; operator ordering within a level); mid-stack spike census at 3 depths; write-side stiff/flat cell ONLY if adapters verified | ≈ 12 GPU-h | SPECTER, ATLAS, firewall |

**The MDE is Arm H discovery: the 7-rung ladder on 6 fine-tunes with both bands.** One person,
one job. Its outcomes force different conclusions (compact / ordered-broad / set-property), and
the sealed 12 give the headline its confirmation estimate.

### 3.2 Arm X scoping (the firewall, stated)

Model 2 is `google/paligemma2-3b-pt-224` (frozen public checkpoint; §0.5-4 verifies weights on
our storage). Arm X compares **verdicts only** — does each frozen rule land the same branch —
and never imports a PG1 number as a PG2 threshold or vice versa; every PG2 rule is calibrated on
PG2 control data. PG2 is Gemma-2-based, so the standing cross-lineage firewall applies in full;
the retired Gemma-2 deletion object is not touched (no LATLOC/LATABLATE-class contrasts). The
deliverable is one row per core law: {two-regime energy leg, within-level operator ordering,
kernel theorems, mid-stack spike depletion, (bonus) stiff-carries} × {replicates / lands
elsewhere / VOID}, each landing informative — a law that lands elsewhere on PG2 acquires its
scope boundary, which no amount of PG1 work could supply.

### 3.3 What each non-MDE arm adds (#11 discipline)

Arm T converts a compact core (if any) into the bar-clearing demonstration — and its census
answers "are the cores new objects or COMWRITE's write in disguise" either way. Arm Q is the
only designed test of a mechanism the global operator provably cannot supply, and its mixture
ladder is the lineage's first *dose-response on the token-dependence axis* (the A9 thread's
causal leg). Arm N spends near-zero forwards on the two oldest unnamed objects with genuinely
new candidate libraries. Arm B turns the witness bank from a stunt into an object with measured
structure. Arm X buys the cheapest external-validity multiplier the program has ever had.

## 4. Data

1. **Items:** pools re-salted `handle|`; Gate-0 audits zero overlap of new sealed items with any
   prior program's sealed exposures on the same tasks. Discovery 200 items/task; sealed cells
   run once.
2. **FT split:** discovery = CARRY's 6 discovery FTs; sealed = CARRY's 12 scored FTs. The ladder
   rule is rehearsed on discovery and applied verbatim to sealed; no threshold moves after
   discovery unblinds.
3. **Error bank (Arm Q):** every family construction seeded + content-hashed; realized per-token
   stats banked so property computations are auditable offline.
4. **Manifests:** `handle_manifest.json` — Q eigvec hashes, ladder projector hashes, per-FT ΔW
   hashes, error-family seeds, witness hashes, PG2 weight sha. Every exclusion counted.

## 5. Controls (each gates a specific claim)

| control | gates | construction |
|---|---|---|
| **swap-shell positive control** | instrument wiring | Gate-0 reproduces one banked CARRY sealed cell at delta 0.0 |
| **band_k (matched-rank random)** | every ladder claim | 3 seeded rank-k random-eigenspace projections of the same ΔW per rung |
| **band_k^stiff (within-stiff random)** | the ORDERING claim specifically | 3 seeded rank-k draws from inside the 679-dim stiff space — the control that separates H_compact from H_set-property |
| **construction-matched seed noise** | every range guard | measured on the same cell family it gates, under the `handle|` salt |
| **module-matrix off-diagonal** | Arm T portability | cross-installs ARE the control; plus a Haar-core row (random rank-K* subspace installed) as the floor |
| **label-shuffle null** | Arm Q property race | family labels permuted against damages (breaks pairing, keeps marginals) |
| **δ-ladder rotated twin** | Arm Q causal leg | each mixture rung's off-token component rotated (destroys direction, keeps norm) — the dose-response must survive it |
| **matched-rank Haar bands** | every census overlap (T-c, N-b, B) | `haar_frames` at the same rank, p95 |
| **saturation guards** | every cell | `saturation_guard` + dead-floor conventions, VOIDs counted |

## 6. Metrics + frozen decision rules

Conventions: gains on each FT's assigned task; log(1+Δppl) for ppl rows; 10k pair bootstraps;
exact permutation for n ≤ 10; BH q = 0.10 within named families, p-values emitted; all-tied rank
statistics := 0; status flags last; every primary statistic null-referenced (§ Truth clause).

**R-H (primary; family = {excess_50 vs band, excess_50 vs band^stiff}).** Guards first: per-FT
range guard; rung-level saturation VOIDs; nesting asserts.

| branch | condition (sealed 12) | the program learns |
|---|---|---|
| `H_compact` | median k_half ≤ 50 AND excess_50 CI-lo > 0 vs BOTH bands | the map is a handle: a ≤ 50-direction core carries the majority of direction-specific skill; Arm T's demonstration adjudicates the practical leg; the mission bar's third leg is in reach of THIS program |
| `H_ordered-broad` | excess vs band^stiff CI-lo > 0 at ≥ 3 rungs but median k_half > 150 | the Q-ordering is functionally real and the storage is graded — the gain-vs-k profile is the named object; skill storage is broad but not flat, and the successor asks what sets the profile's shape |
| `H_set-property` | excess vs band^stiff CI straddles 0 at every rung while excess vs band clears (the CARRY 12/12 pattern must re-detect) | stiffness is a SET property: any k stiff directions do what the top k do — a second "level-set, not coordinates" law joining REGRAIN R1b; the functional ordering, if any, lives off the Q-spectrum, and the residual regressions are banked |
| `H_carry-fails-to-redetect` | excess_679 vs band CI straddles 0 | the CARRY substructure itself does not reproduce under the fresh salt — a replication event senior to every ladder question; reported first, ladder read descriptively |
| `VOID-range` | > 6 of 12 FTs range-VOID | the shell's range under the new salt is the finding |

**R-T (the demonstration; family = {diag-dominance, deletion-selectivity}, gated: runs its full
grid only if R-H lands H_compact or H_ordered-broad; under H_set-property the module matrix runs
at K* = 50 with random-within-stiff cores as the module definition — the set-property version of
portability, pre-specified).** K* = the smallest rung clearing excess CI-lo > 0 vs both bands
(frozen selector; under H_set-property, K* = 50).

| branch | condition | learns |
|---|---|---|
| `H_modules` | own-core beats other-core on own task in ≥ 27 of 30 ordered discovery pairs (exact binomial vs 0.5, p ≤ 0.01) AND deletion drops own-task gain ≥ 3× more than ppl/non-target | skills are compact portable modules — with H_compact this is the bar-clearing demonstration; the census then says whether modules are new objects or U_wr slices |
| `H_shared-substrate` | cross-installs ≈ own-installs (diag dominance ≤ 18/30) AND core census shows cross-FT overlap ≫ Haar | the cores are one shared object (the consensus write with a function) — COMWRITE's U_wr unifies with CARRY's carrier, a different but equally large result |
| `H_neither` | diag dominance ≤ 18/30 AND census at Haar | portability fails and cores are FT-idiosyncratic — skill storage is contextual (bound to the FT's full write), the strongest form of the non-additivity CARRY R3 logged |
| `VOID` | > 2 of 6 discovery FTs range-VOID on cross-cells | the cross-install shell's range is the finding |

**R-Q (mechanism race; family = 4 properties, BH q = 0.10).** Each property p_i yields a frozen
zero-forward score per family; the race statistic is ρ_S(score_i, damage) over the ≥ 12 families;
the causal leg is the δ-ladder: Kendall τ(δ, damage) with exact p, required NEGATIVE (more
dependence → less damage) and monotone (no interior reversal beyond seed noise), rotated twin
required to break the trend.

| branch | condition | learns |
|---|---|---|
| `H_dependence-mechanism` | dependence fraction wins the race (ρ ≥ 0.7, shuffle ≤ 0.3, BH-pass) AND the δ-ladder is monotone-negative with the twin broken | the A9 axis is THE mechanism: benignity is fine-grained error–token dependence, dose-responsive — a named, causal account of why quantization is safe, directly usable as a compression-scheme screen |
| `H_feature-alignment` / `H_massive-coord` | that property wins instead | the mechanism is S3's feature-direction geometry / the sink-coordinate loading — either names the protective property with its own practical screen |
| `H_multi-factor` | ≥ 2 properties clear with no winner by ≥ 0.15 ρ margin | benignity is a composite; the fitted shares + residuals are the object, and the ladder says which factor is causal |
| `H_none-ranks` | no property clears | the benignity carrier escapes all four named candidates — the anomaly is now quadruply instrumented, and the per-family residual table is the successor's map |

**R-N-a (carrier, round 2; zero forwards).** Candidates: massive-coordinate loading (738/674/
604/2015), h̄ overlap, top-8-spike-subspace energy (the banked SPECTER regressor, adjudicated at
last), γ-direction overlap, per-draw n_eff interaction. τ per candidate, exact permutation,
BH over 5. Any winner (τ ≥ 0.6, p ≤ 0.05) → the 110× spread carrier is NAMED after four
programs. All-miss → five more exclusions bank; the carrier is formally outside every
first-order structure the lineage has measured, and the per-item (field) hypothesis is promoted.

**R-N-b (L0 spikes).** Census max-overlap vs matched-rank Haar p95 per library element;
identification at ≥ 3× p95. Identified → the causal cell (ablate the matched structure, spike
Rayleigh must drop ≥ 50 %) converts the identification to a mechanism; census-miss → the spikes
are confirmed outside BOTH libraries (weight-side and vision-side), and their eigvec bank ships
with the strongest possible "genuinely new object" certificate.

**R-B (witness set).** d_eff and principal-angle spectrum vs Haar (frozen bands); overlap census
(family BH); install-tracking ρ_S(install success, witness alignment) over 8 probes with
label-shuffle null. `H_low-dim-surface` iff d_eff ≤ 10 and angles ≪ Haar (the boundary is a
compact named object — the control-surface reading); `H_scattered` iff angles ≈ Haar (worst
cases are direction-idiosyncratic — bounds every compact-boundary story); install-tracking
reported either way (its positive branch links the install phenomenon to the same structure; its
null localizes installability elsewhere).

**R-X (replication; verdicts only).** Per core law: `replicates` / `lands-elsewhere` / `VOID`
(instrument fails its own PG2 anchors). No aggregate score; the row vector IS the result. Any
`lands-elsewhere` row names the first measured scope boundary of that law across models — the
external-validity datum the whole lineage lacks.

## 7. Statistical analysis plan

FT-level pair bootstraps (10k, BCa where skewed) for ladder statistics; item-level within cells;
exact permutation for n ≤ 10 (N-a's 9 draws, Q's δ-ladder, T's binomial); BH q = 0.10 per named
family with p-values emitted; per-seed values banked; every effect stated against its matched
band; discovery rehearsal may not alter sealed thresholds; degenerate completions as declared in
§2; the gain-vs-k curves reported with all three curves (core, band, band^stiff) at every rung —
the full-sweep visualization mandate applies to the analysis, not just the figures.

## 8. Threats-to-validity budget

| threat | effect | mitigation |
|---|---|---|
| shell range at small k | low rungs unreadable | range guard per rung; VOID accounting; the ladder's readable span is itself reported |
| Q single-metric scope | core defined in one site/mode's metric | disclosed (CARRY inheritance); Arm X's PG2 operator pass probes metric-generality at verdict level |
| non-additivity (CARRY R3) | gain_frac(k) curves non-monotone in k | nesting asserts separate construction from behaviour; non-monotone curves reported as measured (they are data about interaction, not error) |
| cross-task difficulty confound (Arm T) | M[i,j] confounds task hardness with specificity | dominance is judged per ordered pair on the SAME task column; the Haar-core floor row anchors each column |
| δ-ladder construction leakage | mixture rungs differ in more than δ | realized per-token stats banked per rung; rotated twin at every rung; dry-run asserts marginal matching |
| SAE-dictionary quality (Arm Q p2) | property 2 mis-scored | §0.5-6 verification + pre-named fallback construction |
| PG2 infra drift | Arm X anchors fail | Gate-X anchors first; VOID row honestly reported; fallback model pre-named |
| snapshot disk | mid-run death | one-snapshot protocol, asserted in-driver |
| winner's curse on K* | K* flattered by discovery | K* frozen from discovery, sealed pass judges it once; sealed curves report all rungs regardless |
| pool dry / Xet wedge | idle burn / stall | pg p5.4xlarge→g6e fallback; never wait; `HF_HUB_DISABLE_XET=1` |

Residual limitations, disclosed: one primary model (Arm X is verdict-level, not a second full
program); Arm T's module matrix is discovery-half (6 FTs — portability at n=6 is a demonstration,
not a survey); Arm Q's damage readout is the house battery at L0 only; the witness set is one
search procedure's output (S2's lesson — worst-case-ness is (search, model)-relative — stands).

## 9. Blind protocol + execution runbook (autonomous)

**Strict order.** Offline gate + D-suite → Gate-0 → **Arm H discovery** → journal sealed
predictions → **Arm H sealed (once)** → Arm T (K* frozen from H) → Arm Q (property scores
journaled before damage cells) → Arm N, Arm B (independent) → Arm X → unblind-per-arm → critic →
registries + artifact refresh.

1. **Offline gate + D-suite (before any GPU):** projector nesting on tiny tensors; ladder
   band constructions; δ-mixture marginal-matching asserts; module-matrix scoring incl. the
   binomial rule's reachability; every §6 failure branch constructed (including
   H_carry-fails-to-redetect and each VOID); degenerate completions exercised; historical
   replays (the CARRY raw-fraction rule-shape event — the new null-referenced statistic must
   classify it correctly; the saturated em-null; the all-tied corner).
2. **Preflight:** Gate-0's banked-cell reproduction is the first GPU act; any miss halts, journal
   first.
3. **Predictions before unblinds:** §11 rows journaled per arm before its decisive read; Arm H's
   per-FT sealed predictions (expected k_half direction under the rehearsed rule) journaled
   before any sealed rung; Arm Q's four property score-vectors journaled before any damage cell.
4. **Pre-mortem responses:** zero-variance band → double seeds once (6) then adjudicate; low-rung
   universal VOID → ladder reads from its first readable rung, disclosed; Arm T under
   H_set-property → the pre-specified random-within-stiff module definition; PG2 weights absent →
   fallback model, disclosed; SAE atoms unusable → fallback property construction; any arm fully
   blocked → certificate + continue; Arm X never blocks the PG1 arms (fully independent).
5. **Unblind once per arm; no rule swap; independence firewall** (§3.2's verdict-only clause is
   part of this).
6. **Critic-before-belief; post-critic regeneration** of every figure, table, and the artifact
   page.

## 10. Artifacts, logs, figure suite (maximum-visual mandate)

**Artifacts:** per-arm `out/metrics.json` (+ `gpu_seconds`, controls, p-values); the ladder
parquet (FT × rung × {core, bands} with CIs); the module matrix parquet; the error-property
table; census banks; `handle_manifest.json`; the artifact registry (figure/number → script +
hashes + DISCOVERY/CONFIRMATORY). All under `s3://pg-lineage-769063704696/results/handle_*`.

**Figure suite — standing rules apply (claim-in-title backed by number + CI + null; how-to-read
line; detailed caption; null overlaid; provenance footer; PNG + PDF; colourblind-safe; full-sweep
rule; regenerated post-critic):**

| id | figure | the claim it carries |
|---|---|---|
| **F-0** | program schema: arms, gates, freeze points (drawn pre-freeze) | the design itself |
| **F-H1 (headline)** | the ladder: gain-vs-k curves for all 12 sealed FTs (small multiples) with band and band^stiff ribbons at every rung, k_half marked per FT | the MDE's answer, FT by FT — no medians hiding tails |
| **F-H2** | median curve + both bands + the three hypothesis regions shaded (compact / ordered-broad / set-property) | which world we are in, at a glance |
| **F-H3** | excess_k forest plot (per-rung CIs vs both bands) | the null-referenced statistic itself |
| **F-T1** | the module matrix M as a heatmap, diagonal outlined, Haar-core floor row appended | portability and specificity in one view |
| **F-T2** | deletion-selectivity paired dumbbells: own-task drop vs ppl/non-target drop per FT | surgical or entangled |
| **F-T3** | core census chord/overlap map: cores × {each other, U_wr, U_unt} with Haar band | are the cores new objects |
| **F-Q1** | the property race: 4 panels, score vs damage per family, ρ + shuffle band each | which ruler ranks benignity |
| **F-Q2** | the δ-ladder: damage vs dependence fraction with rotated-twin curve overlaid | the causal dose-response |
| **F-N1** | carrier regression round 2: 5 candidate panels on the 9 draws, τ + exact p | is the carrier named |
| **F-N2** | L0-spike census matrix vs the SigLIP-side library + the causal-ablation bar | what the spikes are |
| **F-B1** | witness-set geometry: principal-angle spectrum vs Haar + overlap census | is the boundary an object |
| **F-B2** | install-tracking scatter: install success vs witness alignment, shuffle band | does installability ride the same structure |
| **F-X1** | the replication scoreboard: core laws × {replicates / lands-elsewhere / VOID} on PG2, one row each with the PG2-calibrated evidence | the first cross-model scope map |
| **F-X2** | PG2 operator panorama beside PG1's (verdict annotations only, no numeric cross-thresholds) | the two models' spectra side by side under the firewall |

**Post-execution artifact refresh (mandated):** the interactive dossier page is regenerated with
HANDLE's verdicts; F-H1/F-T1/F-Q2 lead.

**On-disk lineage:** this prereg (frozen) → `2026-07-13_handle_journal.md` (live) → result doc →
`figures/handle/`. At close, update BOTH `studies/lineage/RESEARCH_INDEX.md` and
`studies/lineage/README.md`.

## 11. Pre-registered prediction grid (filled 1:1 at unblinding; author credences)

| # | quantity | author's prediction (credence) | result | verdict |
|---|---|---|---|---|
| P1 | **R-H sealed** | H_compact, median k_half ≤ 50 (0.45 — favourite #1, killer named; H_ordered-broad at 0.30, H_set-property at 0.20) | | |
| P2 | CARRY re-detect | excess_679 clears vs band on ≥ 10/12 FTs (high) | | |
| P3 | R-T modules | H_modules diag dominance (med — conditional on H_compact) | | |
| P4 | R-T census | cores overlap U_wr above Haar but are not identical (med) | | |
| P5 | **R-Q race** | H_dependence-mechanism wins (med-high — favourite #2, killer named) | | |
| P6 | R-Q ladder | monotone-negative dose-response, twin broken (med-high) | | |
| P7 | R-N-a | the banked spike-subspace regressor finally orders the draws (low-med; all-miss at 0.35) | | |
| P8 | R-N-b | spikes identify with patch-position harmonics (low-med) | | |
| P9 | R-B | H_low-dim-surface, d_eff ≤ 10 (med) | | |
| P10 | R-B install | install success tracks witness alignment (med-low) | | |
| P11 | R-X | two-regime + spike-depletion verdicts replicate on PG2; ≥ 1 law lands elsewhere (med) | | |
| P12 | program | ≥ 1 standing claim re-scoped by these controls (med — the process working) | | |

Misses stay visible and are scored as the information they are; the lineage's most productive
rows have consistently been its misses, and this grid expects the same.

## 12. Compute budget + references

**Budget (GPU-h + run-count; per-stage gpu_seconds by mandate).** Gate-0 ≈ 1; Arm H ≈ 16
(the big grid: ~590 sealed cells + discovery, snapshot cycling); Arm T ≈ 6; Arm Q ≈ 4;
Arm N-b ≈ 2; Arm B ≈ 4; Arm X ≈ 12; margin ≈ 5. **Total ≈ 50 H100-h across ≈ 10 pg jobs**
(p5.4xlarge first, g6e fallback) + r7i CPU for offline legs. **Protected curiosity slice:**
Arm N (both legs) and Arm B's install-tracking run even if the budget tightens — they are the
cheapest shots at the oldest open questions.

**References.** All citations route through `2026-07-12_lit_registry.md` (verified-complete):
S3 (super-quadratic feature directions — Arm Q property 2 and the p>2 anchor), S10 (Dom/Bulk —
the ordering-vs-set framing of R-H), S17 (massive activations — Arm Q property 3, Arm N-a
candidate), S18 (rotation gauge — instrument covariance, standing), S2 (κ-family + the
(search, model)-relativity caveat, Arm B), S6 (intruder dimensions — Arm T census context).
No external number is a threshold anywhere in §6.

---

*Definition-of-done self-audit at authoring: decision named per world (§1); hypotheses opposite
on one null-referenced primary readout with the boring branch guarded first (§1, §6 R-H); MDE
named and one-person simple (Arm H discovery); pre-mortems folded with responses (§9.4);
estimand 5-attribute with CI and degenerate completions in-document (§2); falsification table
leads with both favourites and includes the program's own standing claims as killable (§1);
every §6 rule carries a ceiling or VOID plus a failure branch the D-suite must construct,
including the parent-replication failure branch (§6 R-H); control trinity per claim with the
band^stiff control doing the load-bearing discrimination (§5); discovery/confirmation split at
FT level with frozen mechanical rosters (§4); no baked thresholds — every parent number
recomputed from its artifact; reuse ledger verified with file + line locations (§0.5);
weight-free dry-run separated from the runtime preflight (§9.1–9.2); journal + pre-unblind
predictions (§9.3); fifteen figures each tied to number + CI + null with reading guides and the
full-sweep rule (§10); citations routed exclusively through the verified registry (§12);
independence firewall incl. the cross-model verdict-only clause (§3.2, §9.5); no-rule-swap,
post-critic regeneration, recompute-don't-snapshot (§9.6, §0); freeze-tag-or-refuse per arm
(§0); rule-shape law — null-referenced primaries — inherited from CARRY and applied throughout.
Nothing this program can observe is uninformative: every branch of every rule names what the
truth just taught us.*
