"""DARWIN — the arm registry. ONE machine-readable source of truth for §3's coverage enumeration.

Every arm in §4 / §6 appears here exactly once with: its wave, its register, its REFUTER line
copied from the spec, one expected direction copied from the spec, its named incumbent (Register
B only), its dose knob, its cost cap, and its status.

Why a registry rather than prose. §3 requires "coverage enumeration at freeze: every arm x cell =
planned / pivot-skipped(reason) / VOID-expected; the completion report reproduces the table
filled". A table maintained by hand drifts from the code that fills it — HUMBOLDT's own ledger
needed a verify script for exactly that reason. Here the freeze table, the expectation store, and
the completion report are all rendered FROM this dict, so they cannot disagree with each other.

STATUS values:
  planned       — the arm is in scope and will run
  pivot-skipped — the arm's named pivot fired; `reason` says which and why
  VOID-expected — the arm is expected to be unpowered/undecidable and is recorded as such

`status` is set to `planned` at freeze for everything in scope; a later status change is an
ADDENDUM and must carry a reason (enforced by `set_status`).
"""
from __future__ import annotations

# (arm, wave, register, title, refuter, expected_direction, incumbent, dose_knob, cap)
_ARMS = [
    # ---------------- Wave 0 (§4) — free/CPU, cap 4 CPU-days ----------------
    # G1 is the §9 step-4 preflight GATE, not a science arm, but it writes atlas rows and so must
    # appear in the coverage enumeration — otherwise the atlas carries an arm the table cannot
    # account for, which is the silent-coverage failure §0.1 names.
    ("G1", "W0", "ATLAS", "Consensus-Gram re-derivation gate (§9 step 4)",
     "re-derived core32/anti32 do not match HUMBOLDT's banked subspaces",
     "the re-derived consensus eigenbasis reproduces the banked core32 and anti32 to float "
     "tolerance", None, None, "cpu"),
    ("G2", "W0", "ATLAS", "Battery preflight: reproduce the FUNC base-ppl anchor (§9 step 6)",
     "the reproduced base perplexity misses the FUNC anchor by more than 2% relative",
     "the same code path reproduces the banked base perplexity to well inside 2% relative",
     None, None, "1 L40S-h"),
    ("W0-0", "W0", "ATLAS", "Derive + publish the topsigma32 structured control",
     "re-derived bottomsigma32 does not match HUMBOLDT's banked bottomsigma32 subspace",
     "the co-derived bottomsigma32 reproduces the banked one to float tolerance, so topsigma32 "
     "is its honest structured counterpart", None, None, "cpu"),
    ("W0-1a", "W0", "ATLAS", "Bank-side top-leverage coordinate lists",
     "twin overlap bands indistinguishable",
     "core32 leverage is spikier than the haar twins, so its >5x-median coordinate set is "
     "smaller and more concentrated than a twin's", None, None, "cpu"),
    ("W0-1b", "W0", "ATLAS", "Massive-activation coordinate lists (median |h_i|)",
     "twin overlap bands indistinguishable",
     "a small set of coordinates per layer exceeds 5x the global median |h_i|, reproducing the "
     "known massive-activation phenomenon", None, None, "1 L40S-h"),
    ("W0-1c", "W0", "ATLAS", "Identity join: core coords vs outlier dimensions",
     "twin overlap bands indistinguishable",
     "core32's overlap with the massive-activation coordinate set exceeds every twin's overlap "
     "band", None, None, "cpu"),
    ("W0-2", "W0", "ATLAS", "Cross-layer core agreement",
     "twin overlap bands indistinguishable",
     "layers' core32 subspaces agree with each other above the independent-Haar reference k/n, "
     "and more than the twins do", None, None, "cpu"),
    ("W0-3", "W0", "ATLAS", "Family census (non-Gemma bases with >=12 key-aligned full FTs)",
     "no consensus-avoided, coordinate-spiky subspace exists outside the Gemma family",
     "at least one non-Gemma base shows a consensus-avoided bottom subspace with leverage "
     "spikiness above its matched-rotation null", None, None, "cpu"),
    ("W0-4", "W0", "ATLAS", "Scale census (gemma-7b zoo)",
     "the gemma-7b zoo shows no consensus-avoided spiky subspace",
     "the same bottom-subspace structure appears at 7b scale", None, None, "cpu"),
    ("W0-5", "W0", "ATLAS", "VLA ecosystem projections",
     "no non-pi0 VLA shows the image-into-core sign flip",
     "at least one other key-aligned VLA reproduces the F8 image-vs-text core energy signature",
     None, None, "cpu"),
    ("W0-6", "W0", "ATLAS", "gemma-2b family tree + odometer",
     "profile correlation carries no family structure (dendrogram indistinguishable from noise)",
     "derivative profiles cluster by lineage, and profile drift grows with training-time gap",
     None, None, "cpu"),
    ("W0-7", "W0", "ATLAS", "OOD ROC from the existing atlas",
     "AUC <= 0.6 everywhere",
     "band-deviation of the core32 e_U series separates repetition and code from wikitext above "
     "AUC 0.6 in at least some layers", None, None, "cpu"),
    ("W0-8", "W0", "ATLAS", "Delta-compression bound (energy vs basis rank)",
     "the universal basis captures no more delta energy than a matched-rank random basis",
     "the highway basis captures delta energy well above the k/n Haar reference at every rank",
     None, "basis rank 4..256", "cpu"),
    ("W0-9", "W0", "ATLAS", "Ontogeny movie (Pythia/OLMo checkpoint suites)",
     "the bottom subspace never stabilises, or stabilises immediately at step 0",
     "the consensus bottom subspace crystallises at an identifiable training step and is stable "
     "thereafter", None, None, "cpu"),
    ("W0-10", "W0", "ATLAS", "Recipe forensics (SFT vs DPO vs RLHF signatures)",
     "held-out accuracy <= chance + CI",
     "a frozen simple classifier on bank/spectral signatures separates recipe classes above "
     "chance on held-out checkpoints", None, None, "cpu"),
    ("W0-11", "W0", "ATLAS", "Loss-spike retrodiction (OLMo logs)",
     "core-band statistics show no relation to recorded spike steps",
     "core-band statistics move at or before recorded loss-spike steps", None, None, "cpu"),
    ("W0-12", "W0", "ATLAS", "Sonification of per-layer core energy",
     "no statistics are claimed by this arm (joy arm)",
     "audio artifacts are banked for every B1 document", None, None, "cpu"),
    ("W0-13", "W0", "ATLAS", "Core-relative frame for Gemma Scope features",
     "core frame no more stable than raw",
     "SAE feature coordinates expressed in the core frame are more stable across the 24 FTs than "
     "in raw coordinates or twin frames", None, None, "cpu"),

    # ---------------- Wave 1 (§4) — single L40S, cap 80 L40S-h ----------------
    ("W1-A", "W1", "GRAD", "Context burning into a core-confined bias",
     "in-core burn <= its own twin burns",
     "a core-confined bias trained to match 16-shot activations beats complement/haar-confined "
     "biases of equal size", "576-param unconstrained bias", "training strength lambda", "6 L40S-h"),
    ("W1-B", "W1", "ATLAS", "ICL amplifier (core-projected context gain)",
     "gain on the core channel moves 4-shot no closer to 16-shot than twin gains do",
     "scaling the core-projected context shift moves 4-shot toward 16-shot", None,
     "gain g in {1,1.5,2,4}", "6 L40S-h"),
    ("W1-C", "W1", "ATLAS", "Intercom capacity (core-pattern messages at L0)",
     "readout bits/token at matched ppl-delta no better than twin channels",
     "the core channel carries more bits/token at equal perplexity cost than twin channels",
     None, "amplitude a", "6 L40S-h"),
    ("W1-D", "W1", "GRAD", "Delta compression eval at highway rank r",
     "universal basis strictly dominated at all ranks",
     "the universal highway basis matches or beats equal-rank SVD-of-own-delta at some rank",
     "equal-rank SVD of the model's own delta", "rank r in {8,32,128}", "6 L40S-h"),
    ("W1-E", "W1", "ATLAS", "Adapter stacking (burned vector on ft-vqav2)",
     "interference beyond CI despite subspace disjointness",
     "both capabilities retain their solo scores within CI when stacked", None, None, "6 L40S-h"),
    ("W1-F", "W1", "GRAD", "KV-cache quantization, core components high-precision",
     "core-high-precision KV quantization no better than uniform at equal memory",
     "keeping core components in high precision beats uniform KV quantization at equal memory",
     "uniform KV-cache quantization", "bits per component", "6 L40S-h"),
    ("W1-G", "W1", "GRAD", "Repair-pi0 readers (rank-32 per-layer core readers)",
     "core readers <= twin readers (content not preferentially in core)",
     "core-bank readers recover more caption/VQA capability than equal-capacity twin readers",
     "linear-probe-only floor; full-fine-tune ceiling", "reader capacity", "6 L40S-h"),
    ("W1-H", "W1", "ATLAS", "Attention puppetry (eps-perturbation along core vs twins)",
     "attention-map JS response to core perturbation within the twin band",
     "core-direction perturbations move attention maps more than twin perturbations at matched "
     "next-token KL", None, "epsilon", "6 L40S-h"),
    ("W1-I", "W1", "ATLAS", "Core-resonant text search",
     "no statistics are claimed by this arm (joy arm)",
     "greedy/beam search finds token sequences with core energy above the twin-search controls",
     None, None, "6 L40S-h"),
    ("W1-J", "W1", "ATLAS", "Heartbeat redo (detrended autocorrelation/spectra)",
     "detrended spectra show no peak above the shuffled-series null",
     "the detrended e_U series carries autocorrelation structure above its shuffled null",
     None, None, "2 L40S-h"),
    ("W1-K", "W1", "GRAD", "Watermark (rank-1 in-core stamp surviving a full FT)",
     "recovery <= twin recovery after FT, or battery harm above CI at all amplitudes",
     "an in-core rank-1 stamp is recoverable after a full fine-tune at an amplitude that does "
     "not harm the battery", "twins-stamped controls", "stamp amplitude", "6 L40S-h"),
    ("W1-L", "W1", "GRAD", "Durable edits placed in core vs random vs highway",
     "edit survival in core no better than in random or highway coordinates",
     "edits placed in core coordinates survive a subsequent fine-tune better than elsewhere",
     "random-coordinate and highway placement", None, "6 L40S-h"),
    ("W1-M", "W1", "GRAD", "Reset-button FT (highway-restricted, then subtracted)",
     "coefficient subtraction does not return the model to base within CI",
     "subtracting the highway coefficients returns weights and battery to base and removes the "
     "task skill", "LoRA-delete", None, "6 L40S-h"),
    ("W1-N", "W1", "GRAD", "Conservedness quantization (INT4 + fp16 core)",
     "fp16-core INT4 no better than the fp16-outlier-list trick at equal memory",
     "keeping the conserved coordinates in fp16 beats uniform INT4 and matches or beats the "
     "standard outlier-list trick", "uniform INT4; fp16-outlier-list", None, "6 L40S-h"),
    ("W1-O", "W1", "GRAD", "Prune-by-conservedness vs magnitude pruning",
     "doubly-cold ranking no better than magnitude pruning at equal sparsity",
     "pruning by conservedness beats magnitude pruning at matched sparsity", "magnitude pruning",
     "sparsity", "6 L40S-h"),
    ("W1-P", "W1", "GRAD", "Highway-aware merging with explicit collision resolution",
     "no gain over TIES at any arity",
     "highway-aware merging beats naive averaging and TIES at some arity", "TIES-merging; naive "
     "averaging", "merge arity 2/3/4", "6 L40S-h"),
    ("W1-Q", "W1", "GRAD", "Gauge experiments (rotate-so-core-is-axes vs Hadamard-smeared)",
     "smeared arm moves the core no more than unrotated",
     "rotating so the core is axis-aligned changes realized core motion under Adam relative to "
     "the smeared and unrotated arms", "unrotated same-recipe control", None, "12 L40S-h"),
    ("W1-R", "W1", "ATLAS", "Reader/writer absorption (global or per-layer per W0-2)",
     "absorbed reader/writer recovers no more than its matched twin",
     "a reader/writer confined to the core absorbs more of the adaptation than a twin-confined "
     "one", None, "absorption rank", "6 L40S-h"),
    ("W1-S", "W1", "GRAD", "SGD tail + let-it-breathe optimizer variants",
     "neither variant changes core motion or battery beyond CI vs stock Adam",
     "an SGD tail or subspace-second-moment Adam moves the core measurably more than stock Adam "
     "at matched task score", "stock Adam", "breathing dose", "6 L40S-h"),
    ("W1-T", "W1", "GRAD", "Frozen-GaLore: highway-projected training + few-data ladder",
     "dominated by LoRA at every point of the data ladder",
     "highway-projected training matches or beats LoRA-r32 and GaLore at matched optimizer memory "
     "in the low-data regime", "LoRA-r32; GaLore; full Adam", "data {50,200,1000,full}", "6 L40S-h"),
    ("W1-U", "W1", "GRAD", "Geometric replay (core-activation anchor vs EWC-lite vs replay)",
     "anchor's retention <= EWC-lite control at matched task score",
     "a core-activation anchor retains more LM ability than EWC-lite and replay at matched task "
     "score", "EWC-lite on core coords; available-data replay", "anchor lambda", "6 L40S-h"),
    ("W1-V", "W1", "ATLAS", "Data conflict scores (per-example gradient-core projection)",
     "removing top-decile core-attackers does not change retention beyond CI",
     "retraining without the top-decile core-attacking examples improves retention", None,
     None, "6 L40S-h"),
    ("W1-W", "W1", "GRAD", "Distillation-32 (core-activation matching)",
     "core-activation matching no better than hidden-state or logit matching at equal budget",
     "a student distilled on core-activation matching beats hidden-state matching at equal budget",
     "hidden-state matching; logit matching", None, "8 L40S-h"),
    ("W1-X", "W1", "ATLAS", "Telemetry harness (streaming core projector callback)",
     "overhead exceeds the budget that would make streaming telemetry usable",
     "the streaming projector adds small measured overhead and produces a usable demo log",
     None, None, "2 L40S-h"),

    # ---------------- Wave 2 (§4) — cap 120 L40S-h ----------------
    ("W2-A", "W2", "ATLAS", "Failure envelope (long SGD, Muon, LR ladder, QAT)",
     "no regime breaks the invariant (core motion flat across all regimes)",
     "at least one optimizer/LR regime measurably breaks the core invariant", None,
     "LR x {1,4,16}", "24 L40S-h"),
    ("W2-B", "W2", "ATLAS", "Optimizer-regime probe (flow-matching head; small PPO)",
     "core motion and highway concentration identical across regimes",
     "imitation and RL regimes differ measurably in core motion and highway concentration",
     None, None, "24 L40S-h"),
    ("W2-C", "W2", "ATLAS", "Small pretrains: registers-in-width / evict-the-core / scrambled",
     "a core forms identically in every condition including the scrambled-position control",
     "reserving protected dimensions or freezing core directions changes whether and where a "
     "core forms", None, None, "40 L40S-h"),
    ("W2-D", "W2", "ATLAS", "Skeleton curriculum + transplant + data attribution",
     "core-coordinate transplant reaches target loss no faster than fresh init",
     "transplanting core coordinates reduces time-to-loss relative to fresh init", None,
     "curriculum switch point", "20 L40S-h"),
    ("W2-E", "W2", "ATLAS", "Two tenants + distillation inheritance",
     "a distilled student's own consensus core is unrelated to the teacher's coordinates",
     "an output-only distilled student inherits the teacher's core coordinates", None, None,
     "6 L40S-h"),
    ("W2-F", "W2", "GRAD", "Interaction matrix (anchor x restriction x breathing)",
     "no interaction term is distinguishable from zero",
     "at least one pairwise interaction term is non-zero beyond its CI", "additive baseline",
     "one dose each", "6 L40S-h"),

    # ---------------- Wave P (§4) — cap 30 L40S-h ----------------
    ("WP-1", "WP", "GRAD", "LIBERO bridge pilot (emits the stage-2 margin CIs)",
     "stack retains no more than naive at matched task success",
     "the retention stack retains more trunk battery than naive FT at matched LIBERO success",
     "available-data replay; naive FT", "anchor dose", "24 L40S-h"),

    # ---------------- Stage 2 (§6) — launches only past G0 ----------------
    ("S2-A", "S2", "STAGE2", "Forgetting-free robot (flagship)",
     "no dose satisfies both the task and retention predicates",
     "the graduated retention stack lands within delta_task of naive AND above naive by m_retain",
     "available-data replay; naive FT", "anchor dose", "280 L40S-h (re-anchored at G0)"),
    ("S2-B", "S2", "STAGE2", "Continual robot (LIBERO sequential, >=5 tasks)",
     "retention at or below EWC at matched memory",
     "the highway-partition + reset-button + burned-vector stack retains above EWC at matched "
     "memory", "EWC; replay; naive-sequential", None, "100 L40S-h (re-anchored at G0)"),
    ("S2-C", "S2", "STAGE2", "Sidecars: repair-at-large, serving stack, provenance atlas",
     "the reader recipe fails to recover any faculty on wild models",
     "the W1-G reader recipe recovers a documented lost faculty on at least one wild HF model",
     "8 separate LoRAs (serving demo)", None, "40 L40S-h"),
]

# ---------------------------------------------------------------------------
# POST-FREEZE ADDENDA. The frozen spec is NEVER edited — §0.2.2's fork check diffs the
# executor's copy against the director's, and a retrofit would trip it (the TESSERA incident is
# why that rule exists). A dated addendum file is the sanctioned side channel; its arms enter
# here tagged with their source so every row they emit is attributable to the addendum rather
# than to the frozen spec.
# ---------------------------------------------------------------------------
SPEC_SOURCE = "spec 2026-08-12_darwin_program_spec.md"
CEQ_ADDENDUM = "ADDENDUM 2026-08-12_darwin_addendum_ceq.md"

_ADDENDUM_ARMS = [
    ("W1-N4", "W1", "GRAD", "Core-exact quantization (CEQ): vault + projector-routed bulk",
     "arm b (INT4 + fp16 outlier-column list) >= CEQ everywhere AND the leak-law prediction "
     "misses by >2x",
     "d/e are EXPECTED to land close to b on ppl-at-matched-bits; the decisive novelty rides on "
     "f (vault reuse, predicted by the leak law) and the KV variant (the T7 mechanism)",
     "INT4 + fp16 outlier-column list (arm b, the standard trick); uniform INT4 (arm a)",
     "bit budget / list size k", "6 L40S-h", CEQ_ADDENDUM),
]

ARMS = {a[0]: {"arm": a[0], "wave": a[1], "register": a[2], "title": a[3], "refuter": a[4],
               "direction": a[5], "incumbent": a[6], "dose_knob": a[7], "cap": a[8],
               "source": (a[9] if len(a) > 9 else SPEC_SOURCE),
               "status": "planned", "reason": None}
        for a in _ARMS + _ADDENDUM_ARMS}

WAVE_ORDER = ("W0", "W1", "W2", "WP", "S2")
STATUSES = ("planned", "pivot-skipped", "VOID-expected")


def by_wave(wave: str):
    return [a for a in ARMS.values() if a["wave"] == wave]


def set_status(arm: str, status: str, reason: str):
    """Post-freeze status changes are ADDENDA and must carry a reason (§3, §0.1)."""
    assert arm in ARMS, f"unknown arm {arm}"
    assert status in STATUSES, f"unknown status {status}"
    assert reason and reason.strip(), f"{arm}: a status change requires a reason"
    ARMS[arm].update(status=status, reason=reason)
    return ARMS[arm]


def coverage_table() -> list:
    """§3's coverage enumeration, as rows. Rendered to markdown by scripts/darwin_freeze.py."""
    rows = []
    for w in WAVE_ORDER:
        for a in sorted(by_wave(w), key=lambda x: x["arm"]):
            rows.append({**a, "claimed_by": "darwin-executor"})
    return rows


def addendum_arms() -> list:
    return sorted(a["arm"] for a in ARMS.values() if a["source"] != SPEC_SOURCE)


def totals() -> dict:
    out = {"n_arms": len(ARMS), "n_addendum_arms": len(addendum_arms())}
    for w in WAVE_ORDER:
        out[w] = len(by_wave(w))
    for r in ("ATLAS", "GRAD", "STAGE2"):
        out[r] = sum(1 for a in ARMS.values() if a["register"] == r)
    out["n_with_incumbent"] = sum(1 for a in ARMS.values() if a["incumbent"])
    out["n_with_dose"] = sum(1 for a in ARMS.values() if a["dose_knob"])
    return out
