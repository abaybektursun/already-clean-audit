"""MERCATOR program registry — the arms, the §5 coverage contract, the §10 budget ledger.

Spec: studies/lineage/paligemma1-pi0/research/2026-08-14_mercator_program_spec.md
Queue (the ideation record every arm cites): 2026-08-14_successor_round_queue.md

This module is the machine-readable form of §4 + §5 + §10. It exists so that three failure modes
named in §0.1 become impossible rather than merely discouraged:

  * SILENT COVERAGE SHRINKAGE — `coverage_table()` enumerates EVERY queue item and refuses to
    build if any item has no disposition, so an arm cannot quietly vanish between spec and report.
  * AN UNBUDGETED WAVE — `WaveBudget` accounts REALIZED L40S-hours against the §10 wave cap and
    refuses a launch that would exceed it, so the cap binds mechanically instead of in prose.
  * AN OUT-OF-ORDER LAUNCH — `check_deps` enforces the §8 dependency edges.

FAIL LOUD: every inconsistency is an assertion, never a warning.
"""
from __future__ import annotations

import json
import pathlib

SPEC_SOURCE = "spec 2026-08-14_mercator_program_spec.md"
QUEUE_SOURCE = "queue 2026-08-14_successor_round_queue.md"

# ---------------------------------------------------------------------------
# §4 — the arms. (arm, wave, register, queue_item, title, refuter, cap_l40s_h, cap_note)
#
# `cap_l40s_h` is the §4 per-arm GPU cap verbatim from the spec; 0.0 means a CPU-only arm (its cap
# is the wave's r7i-day budget, tracked separately). `refuter` is the spec's REFUTER line verbatim
# — it is pre-logged into the expectation store before the arm runs (§3), so it is quoted here as
# the single source rather than retyped per job.
# ---------------------------------------------------------------------------
_ARMS = [
    # ---- Wave K — the DARWIN fold (gate; blocks freeze) ----
    ("M-K1", "K", "ATLAS", "gate",
     "Dispatch + verify the DARWIN stage-1 critic; fold findings; amend this spec pre-freeze",
     "n/a (gate arm; deliverable = the critic report + the fold)", 0.0,
     "cap: the critic's report + fold, no new science"),

    # ---- Wave 0 — from existing artifacts (CPU-dominant; cap 6 r7i-days + 4 L40S-h) ----
    ("M-D1", "W0", "ATLAS", "D1",
     "Regenerate W2-F summary (stale composition_passes) + audit ALL DARWIN summaries for "
     "verdict-field staleness",
     "n/a (hygiene arm; deliverable = the diff)", 0.0, "CPU"),
    ("M-A7", "W0", "ATLAS", "A7",
     "Fix the REFEREE dose-anchor lookup; re-derive the true helpful dose; log defect + corrected "
     "anchor row",
     "n/a (defect arm; deliverable = corrected anchor + defect-ledger row)", 0.0, "CPU"),
    ("M-1", "W0", "ATLAS", "1",
     "MQA prediction: k_proj avoidance in non-MQA/GQA families from EXISTING census banks",
     "k_proj at chance in a true-MHA family kills the MQA story", 0.0, "CPU"),
    ("M-B1", "W0", "ATLAS", "B1",
     "OUTPUT-SIDE census: consensus Gram dW dW^T over the 24 house FTs (7 types x 18 layers) + "
     "twins + spectra + leverage",
     "of 'second sink': bottom row-space at matched-rotation-null levels everywhere", 0.0, "CPU"),
    ("M-9a", "W0", "ATLAS", "9",
     "Universality-table CPU cells: cross-layer verticality, per-type scoping, leverage profiles, "
     "Llama published-outlier-index vs spike-list",
     "per-cell, pre-named in the universality table (a cell at chance refutes that property)",
     0.0, "CPU"),
    ("M-8", "W0", "ATLAS", "8",
     "Single-model sink predictor from BASE-model gradient magnitude + sign-consistency; validate "
     "against 5 measured banks",
     "predictor <= activation-magnitude baseline on overlap with measured banks", 2.0,
     "~2 L40S-h total (backward passes only)"),
    ("M-A6", "W0", "ATLAS", "A6",
     "DOME-nuisance vs sink overlap: streaming top-eigenspace of centered gradient covariance; "
     "principal angles vs sink/twins",
     "overlap within twin band", 2.0, "GPU-light"),
    ("M-mine", "W0", "ATLAS", "S-W2-A",
     "Mine W2-A's per-bank motion rows for the Muon sink behavior",
     "n/a (standing mining arm; expected-pattern row pre-logged: Muon moves the sink; damage "
     "follows the motion power law)", 0.0, "CPU"),
    ("M-V", "W0", "ATLAS", "S-W0-5",
     "VLA ecosystem census: mechanical roster search over the VLA roster for a base with >=12 "
     "key-aligned full-FT derivatives; consensus banks + avoidance + spikiness if found",
     "of sink universality in the VLA family: full-strength sink absent where the roster supports "
     "a census; if no roster qualifies, COVERAGE-IMPOSSIBLE with the count", 0.0,
     "CPU roster probe; census cells draw from Wave 1 if the roster qualifies "
     "(PRE-FREEZE AMENDMENT A3 — this standing item had no arm in the spec as written)"),
    ("M-C6", "W0", "ATLAS", "C6",
     "OOD sharpening: rescore existing B1 corpora with resonance-informed features vs the "
     "recomputed AUC baseline",
     "resonance-informed features no better than the recomputed baseline AUC", 0.0, "CPU"),
    ("M-D3", "W0", "ATLAS", "D3",
     "Loss-spike retrodiction retry against OLMo's published logs",
     "spike times unpredicted at the pre-set margin; pivot: logs genuinely absent -> "
     "COVERAGE-IMPOSSIBLE with evidence", 0.0, "CPU"),
    ("M-D4", "W0", "ATLAS", "D4",
     "Recipe-forensics roster probe: does ANY base now have >=2 held-out minority-recipe FTs?",
     "n/a (probe arm; deliverable = the roster count, unparking W0-10 or recording the probe)",
     0.0, "CPU"),
    ("M-D5", "W0", "ATLAS", "D5",
     "Ruscio et al. 2605.16600 claim-level read (quote + location per cited claim); tag "
     "predicted-by vs consistent-with",
     "n/a (cite-verify gate; deliverable = the claim table)", 0.0, "CPU"),

    # ---- Wave 1 — GPU-light science (cap 110 L40S-h) ----
    ("M-12", "W1", "ATLAS", "12",
     "Sink x induction heads (the semantics bridge): head identification, two-clock overlay, "
     "pos-vs-induction discriminator",
     "induction heads' read overlap with sink within twin band", 8.0, "runs FIRST in Wave 1"),
    ("M-G", "W1", "ATLAS", "S-W1-G",
     "Repair-pi0 readers: rank-32 per-layer readers U^T h -> text-readout path; COCO-caption + "
     "VQA recovery vs equal-capacity twins vs linear-probe floor vs small full-FT ceiling",
     "core readers <= twin readers", 10.0, "top priority; pi0.5 re-shell rides along (F8)"),
    ("M-Q", "W1", "ATLAS", "S-W1-Q",
     "Gauge rotation FT: gamma-fold + rotation function-preservation gate, then FT rotated-align "
     "vs rotated-Hadamard vs unrotated, same recipe/seeds",
     "of the TENANT reading: Hadamard arm moves the sink no more than unrotated", 12.0,
     "THE causal Adam-basis test; feeds M-11"),
    ("M-A4", "W1", "ATLAS", "A4",
     "Sink-vs-BOS timing: project sink banks onto EARLY OLMo checkpoints' activations",
     "sink-coordinate activation traffic does not predate crystallization", 3.0, ""),
    ("M-2", "W1", "ATLAS", "2",
     "Ontogeny replication: Pythia suites (>=2 sizes); windowed consensus Gram; crystallization "
     "fraction per cell",
     "of 'forms late': wild variance across families", 4.0, "CPU-heavy + 4 L40S-h"),
    ("M-3", "W1", "ATLAS", "3",
     "Muon natural experiment: mechanical roster search for an open Muon-trained base with >=12 "
     "key-aligned full-FT derivatives; consensus banks + avoidance + spikiness",
     "of TENANT-universality: full-strength sink in a Muon-trained base", 6.0,
     "pivot ladder: checkpoint-delta variant, else census-impossible with the count"),
    ("M-9b", "W1", "ATLAS", "9",
     "Universality GPU cells on ONE foreign base (Llama-3.2-3B): FUNC-style input ablation, "
     "B1-style activation share, W1-I-style addressability",
     "per cell: ablation within matched-random band; energy at chance; search fails to beat "
     "twin-search", 8.0, ""),
    ("M-B7", "W1", "ATLAS", "B7",
     "Language-variance anomaly: per-language sink-energy profiles + variance decomposition vs "
     "script/tokenization covariates + twins",
     "per-language variance within the twin band (the anomaly is instrument noise)", 3.0, ""),
    ("M-C4", "W1", "ATLAS", "C4",
     "Resonance fingerprint: W1-I search on each census base at a frozen budget; within-family vs "
     "cross-family profile stability",
     "profiles no more stable within family than across", 6.0, ""),
    ("M-A2", "W1", "ATLAS", "A2",
     "Breathe ceiling: extend W1-S's gain ladder until damage onset; retention readout; REFEREE "
     "power-law pickup; g4-anomaly replication cell",
     "no damage onset within the swept ladder (the ceiling is unbounded in range) OR damage onset "
     "not tracked by the REFEREE power law", 8.0, ""),
    ("M-A3", "W1", "ATLAS", "A3",
     "Settling experiments: per-coordinate Var(g)/m^2 vs subspace-projected consistency; "
     "scalar-denominator normalized-SGD arm; epsilon-sensitivity at the suppressed regime",
     "the settling diagnostic does not separate sink from twin coordinates", 6.0,
     "optional sub-arm: leak-corrected update-projection (the E5 unlock)"),
    ("M-4", "W1", "ATLAS", "4",
     "Erosion-curve program: mark-vs-function confound arm FIRST, then erosion rate vs subspace "
     "write-traffic across >=6 subspaces x >=3 durations x 2 recipes",
     "of the amplitude story: large stamps still die everywhere; of engineering: no monotone "
     "traffic-erosion relation", 12.0, "A1's matched-motion twins ride inside this arm"),
    ("M-E4", "W1", "ATLAS", "E4",
     "W1-V retry on the damaging recipe: per-example gradient-sink alignment -> drop top-decile "
     "-> retention delta",
     "dropping the top-decile-aligned examples does not change retention vs a random-decile drop",
     4.0, ""),
    ("M-AB", "W1", "ATLAS", "S-W1-AB",
     "ICL chain, four gated stages: task screening, context burning + amplifier, prompt priming "
     "(C5), adapter stacking (E1)",
     "per stage: (i) no task shows real ICL lift at n with power; (ii) burning/amplifying within "
     "twin band; (iii) resonant preamble no better than matched-length neutral; (iv) stacking "
     "interaction within CI", 10.0, "C5 and E1 ride inside this arm"),
    ("M-E2", "W1", "ATLAS", "E2",
     "Distillation-32: small student, sink-activation matching vs hidden-state vs logits at equal "
     "budget",
     "sink-activation matching no better than hidden-state matching at equal budget", 6.0, ""),
    ("M-E3", "W1", "GRAD", "E3",
     "KV-cache quantization base arm + long-context extension (7c)",
     "routed KV quantization no better than uniform KV quantization at matched bits", 6.0, ""),
    ("M-D2", "W1", "ATLAS", "D2",
     "Core-relative frame retry: find/train a gemma-1-compatible dictionary",
     "no dictionary direction aligns with the core above the twin band; pivot ladder: public "
     "gemma-1 SAEs, own small SAE, else COVERAGE-IMPOSSIBLE with evidence", 6.0, ""),

    # ---- Wave 2 — pretrains + regimes (cap 130 L40S-h) ----
    ("M-10", "W2", "ATLAS", "10",
     "Registers-in-width: switches (a) fixed gamma / (b) block second moment / (c) writer init "
     "ablated INDEPENDENTLY; adoption curve, leverage in/out, outlier magnitudes, matched loss",
     "adoption angle stays at the Haar band (the network ignores the sanctioned block)", 40.0,
     ">=2 seeds; LR sweep on switch (b)"),
    ("M-11", "W2", "ATLAS", "11",
     "Rotation-equivariant twin: the complete grid-aware inventory neutralized or declared; "
     "leverage spectrum, max|x_i|, INT4 gap, and THE SAFETY-RAIL TEST",
     "of the grid theory: spiky structure in the equivariant twin (then the leak hunt IS the "
     "result)", 45.0, "comparison at equal VALIDATION LOSS, never equal steps"),
    ("M-B2", "W2", "ATLAS", "B2",
     "Evict-the-core + scrambled-position pretrains (independent switches from M-10)",
     "eviction does not move the emergent sink off the evicted block", 20.0, ""),
    ("M-B3", "W2", "ATLAS", "B3",
     "Two-tenants + distillation inheritance: does an output-only student grow its sink at the "
     "SAME coordinates?",
     "student sink coordinates at chance overlap with teacher coordinates", 12.0, ""),
    ("M-B4", "W2", "ATLAS", "B4",
     "Core transplant at init",
     "transplanted core neither persists nor accelerates crystallization vs matched-random "
     "transplant", 0.0, "cap group B456 = 16 L40S-h combined"),
    ("M-B5", "W2", "ATLAS", "B5",
     "Curriculum REVERSED (quality-late), switch point parameterized by measured crystallization",
     "quality-late placement changes neither the sink nor the loss vs quality-early at matched "
     "budget", 0.0, "cap group B456 = 16 L40S-h combined"),
    ("M-B6", "W2", "ATLAS", "B6",
     "Per-batch gradient-sink attribution logged during all Wave-2 pretrains (free rider)",
     "no batch class scores above the shuffled-attribution null", 0.0,
     "cap group B456 = 16 L40S-h combined"),
    ("M-A5", "W2", "ATLAS", "A5",
     "Regime probe: flow-matching imitation head (1 LIBERO task) + small PPO text run; sink "
     "motion + highway concentration per regime",
     "non-supervised regimes show the same sink motion as supervised at matched realized motion",
     10.0, ""),

    # ---- Wave P — the practical instruments (Register B; cap 60 L40S-h + CPU) ----
    ("M-C1", "WP", "GRAD", "C1",
     "The fine-tune linter: highway profile, sink leak + leak-law forecast, damage forecast, "
     "recipe class, nearest checkpoint; leave-one-out on the 24 FTs + externals",
     "forecasts no closer than the no-model baseline at the pre-set factor", 6.0, ""),
    ("M-C2", "WP", "GRAD", "C2",
     "Merge pre-check: highway-collision profile -> predicted TIES-merge damage across >=10 "
     "pairs/triples",
     "prediction no better than the norm-product baseline", 6.0, ""),
    ("M-C3", "WP", "GRAD", "C3",
     "Progressive delivery demo: r-ladder streaming format on ft-vqav2 (+1 more FT with valid "
     "readout); capability-vs-bytes vs own-SVD incumbent",
     "capability-vs-bytes no better than the own-SVD incumbent", 8.0, ""),
    ("M-5", "WP", "GRAD", "5",
     "Sink-aware gradient compression: sink-excluded top-k vs naive top-k at matched bandwidth "
     "(+ random-excluded control)",
     "no convergence gain at matched bandwidth", 12.0,
     "single-node simulation of the communication is acceptable and is stated in-row"),
    ("M-6", "WP", "GRAD", "6",
     "Rate-of-change checkpoint format: spec + end-to-end demo (union-k stable block, "
     "leak-law-priced vault, highway motion file); round-trip equivalence gates",
     "format costs more bytes than the own-SVD incumbent at matched capability", 10.0,
     "uses M-Q's fold machinery"),
    ("M-7", "WP", "GRAD", "7",
     "Quantization continuation: protection-list replication on Llama/Qwen; composition-failure "
     "mechanism (interaction localization over shared coordinates)",
     "protection list no better than outlier-column selection on the foreign families", 10.0,
     "rides on M-E3"),
    ("M-C7", "WP", "GRAD", "C7",
     "Sampled telemetry: 1-in-N ladder; overhead + drift-detection power",
     "sampled telemetry cannot detect the planted drift at the overhead the ladder buys", 4.0, ""),
    ("M-C8", "WP", "ATLAS", "C8",
     "Observatory + basis cards release: full gemma-2b zoo scan, family tree, census banks "
     "packaged with derivation manifests + this spec's terminology",
     "n/a (release arm; deliverable = the packaged artifact + manifest)", 0.0,
     "CPU-only; cutoff logged"),
    ("M-C9", "WP", "GRAD", "C9",
     "Repair-at-large: 2-3 wild HF models with documented lost faculties",
     "core readers <= twin readers on the wild models", 4.0,
     "ONLY if M-G graduates its REFUTER"),
]

ARMS = {a[0]: {"arm": a[0], "wave": a[1], "register": a[2], "queue_item": a[3], "title": a[4],
               "refuter": a[5], "cap_l40s_h": a[6], "cap_note": a[7]} for a in _ARMS}
assert len(ARMS) == len(_ARMS), "duplicate arm id in _ARMS"

WAVE_ORDER = ("K", "W0", "W1", "W2", "WP")

# §4 arms whose GPU cap is shared across a group rather than held per arm.
CAP_GROUPS = {"B456": {"arms": ("M-B4", "M-B5", "M-B6"), "cap_l40s_h": 16.0}}

# §8 dependency edges, verbatim. `check_deps` refuses an out-of-order launch.
# §6 states the only GRADUATION gate is M-C9's (on M-G); these are EXECUTION-ORDER edges, which is
# a different thing — recorded here so the two cannot be confused (pre-freeze amendment A4).
DEPS = {
    "M-G": (),
    "M-12": (),                      # "run FIRST in this wave" — expressed as a wave rule below
    "M-11": ("M-Q",),                # M-Q before M-11
    "M-C9": ("M-G",),                # the one GRADUATION-gated arm (§6)
    "M-7": ("M-E3",),                # M-7 "rides on M-E3"
    "M-6": ("M-Q",),                 # M-6 "uses M-Q's fold machinery"
}
WAVE_FIRST = {"W1": "M-12"}          # §8: "M-12 first in W1"
# §8: "M-AB before M-E1" and "M-4 confound arm before curve" are INTRA-arm stage orders (E1 is
# stage iv of M-AB; the confound arm is stage 1 of M-4), enforced inside those jobs, not here.

# ---------------------------------------------------------------------------
# §5 — the coverage map. EVERY queue item, its disposition, no exceptions.
# ---------------------------------------------------------------------------
ARM = "ARM"                               # covered by one or more arms above
RIDES_INSIDE = "RIDES-INSIDE"             # covered as a named stage of another arm
PARKED = "PARKED"                         # explicitly not funded, with its unlock condition
CLOSEOUT = "CLOSEOUT"                     # a §8 runbook step, not an arm
DIRECTOR_DECISION = "DIRECTOR-DECISION"   # post-MERCATOR; explicitly NOT an arm
DISPOSITIONS = (ARM, RIDES_INSIDE, PARKED, CLOSEOUT, DIRECTOR_DECISION)

# The complete queue enumeration (§5: items 1-12, A1-A7, B1-B7, C1-C9, D1-D7, E1-E6, and all five
# standing items). Written out rather than generated, so that a typo in a range cannot silently
# shrink the contract.
QUEUE_ITEMS = (
    tuple(str(i) for i in range(1, 13))
    + tuple(f"A{i}" for i in range(1, 8))
    + tuple(f"B{i}" for i in range(1, 8))
    + tuple(f"C{i}" for i in range(1, 10))
    + tuple(f"D{i}" for i in range(1, 8))
    + tuple(f"E{i}" for i in range(1, 7))
    + ("S-W1-G", "S-W1-Q", "S-W1-AB", "S-W2-A", "S-W0-5")
)
assert len(QUEUE_ITEMS) == len(set(QUEUE_ITEMS)) == 53, \
    f"queue enumeration is {len(QUEUE_ITEMS)} items, expected 53 distinct"

# Non-arm dispositions, verbatim from §5.
NON_ARM_DISPOSITION = {
    "A1": (RIDES_INSIDE, "M-4",
           "motion-matched controls: M-4's matched-motion twins ARE the REFEREE successor (§5)"),
    "C5": (RIDES_INSIDE, "M-AB",
           "prompt priming is stage (iii) of the ICL chain (§4 M-AB)"),
    "E1": (RIDES_INSIDE, "M-AB",
           "adapter stacking is stage (iv) of the ICL chain, gated on a working burner (§4 M-AB)"),
    "E5": (PARKED, None,
           "W1-T ladder stays PARKED; unlock = leak-corrected update-projection built, a named "
           "optional sub-arm of M-A3 (§5)"),
    "E6": (DIRECTOR_DECISION, None,
           "folded into D7: the robotics charter is a post-MERCATOR director decision, NOT an arm "
           "(§5)"),
    "D6": (CLOSEOUT, None,
           "propagation is §8 runbook step 10, not an arm (§5)"),
    "D7": (DIRECTOR_DECISION, None,
           "robotics charter remains a post-MERCATOR director decision, NOT an arm (§5)"),
}


def arms_for(queue_item: str) -> list:
    """Every arm covering a queue item. §5's contract is 'no stone unturned', which is a
    surjectivity requirement (every item covered), not an injectivity one: item 9 is deliberately
    split into a CPU cell (M-9a) and a GPU cell (M-9b) by §4 itself.

    PRE-FREEZE AMENDMENT A2: §5's prose says 'maps to exactly one arm'; §4 as written maps item 9
    to two. The coverage table therefore lists ALL arms per item and asserts >=1, which is the
    contract §5 actually enforces ('silent shrinkage = defect').
    """
    return sorted(a for a in ARMS if ARMS[a]["queue_item"] == queue_item)


def coverage_table() -> list:
    """One row per queue item: its disposition and the arms (if any) that cover it.

    Refuses to build if any queue item has neither an arm nor a named non-arm disposition — the
    mechanical form of §5's no-stone-unturned contract.
    """
    rows = []
    for item in QUEUE_ITEMS:
        arms = arms_for(item)
        if arms:
            rows.append({"queue_item": item, "disposition": ARM, "arms": arms,
                         "note": "; ".join(ARMS[a]["title"] for a in arms)})
            continue
        assert item in NON_ARM_DISPOSITION, (
            f"queue item {item!r} has no arm and no named disposition — §5 requires every item to "
            f"map somewhere, and an unmapped item IS the silent shrinkage the spec calls a defect")
        d, host, note = NON_ARM_DISPOSITION[item]
        assert d in DISPOSITIONS, f"unknown disposition {d!r} for {item}"
        rows.append({"queue_item": item, "disposition": d,
                     "arms": [host] if host else [], "note": note})
    # And the converse: no arm may exist without a queue item (the M-K1 gate arm excepted, which
    # covers the queue's "Gate before all of this" line rather than a numbered item).
    orphans = sorted(a for a, m in ARMS.items()
                     if m["queue_item"] not in QUEUE_ITEMS and m["queue_item"] != "gate")
    assert not orphans, f"arms with no queue item: {orphans}"
    assert len(rows) == len(QUEUE_ITEMS)
    return rows


def uncovered() -> list:
    """Queue items with neither an arm nor a disposition. Empty by construction of the assertion in
    coverage_table(); exposed so a preflight can report it as a number rather than a traceback."""
    return [i for i in QUEUE_ITEMS if not arms_for(i) and i not in NON_ARM_DISPOSITION]


# ---------------------------------------------------------------------------
# §10 — the budget ledger. Caps bind on REALIZED hours, not on estimates.
# ---------------------------------------------------------------------------
WAVE_CAPS_L40S_H = {"K": 0.0, "W0": 4.0, "W1": 110.0, "W2": 130.0, "WP": 60.0}
CURIOSITY_RESERVE_L40S_H = 25.0     # §10: curiosity extensions + surprise follow-ups ONLY
WAVE_CAP_R7I_DAYS = {"W0": 6.0}     # §4 Wave 0: "cap 6 r7i-days"
TOTAL_CAP_L40S_H = sum(WAVE_CAPS_L40S_H.values()) + CURIOSITY_RESERVE_L40S_H   # ~330 (§10)


def arm_cap(arm: str) -> float:
    """The GPU cap that binds this arm: its own, or its cap group's share."""
    for g, spec in CAP_GROUPS.items():
        if arm in spec["arms"]:
            return spec["cap_l40s_h"]
    return ARMS[arm]["cap_l40s_h"]


def wave_cap_audit() -> list:
    """§10 vs §4: does the sum of per-arm caps fit inside the wave cap?

    It does NOT for W1 and W2 as the spec is written, and that is recorded here rather than
    silently reconciled: per-arm caps are CEILINGS, the wave cap is the BINDING budget, and §10's
    named degradation order ('drop seeds, then drop the largest-model cell, journaled') is what
    closes the gap if realized spend approaches the cap. Recording the oversubscription is the
    executor's job; choosing which science to drop is the director's.
    """
    rows = []
    for w in WAVE_ORDER:
        arms = [a for a in ARMS if ARMS[a]["wave"] == w]
        seen_groups, total = set(), 0.0
        for a in arms:
            g = next((g for g, s in CAP_GROUPS.items() if a in s["arms"]), None)
            if g is not None:
                if g in seen_groups:
                    continue
                seen_groups.add(g)
                total += CAP_GROUPS[g]["cap_l40s_h"]
            else:
                total += ARMS[a]["cap_l40s_h"]
        cap = WAVE_CAPS_L40S_H[w]
        rows.append({"wave": w, "n_arms": len(arms), "sum_arm_caps_l40s_h": round(total, 1),
                     "wave_cap_l40s_h": cap,
                     "oversubscribed_by_l40s_h": round(max(0.0, total - cap), 1),
                     "degradation_order": "§10 default: drop seeds, then drop the largest-model "
                                          "cell, journaled"})
    return rows


class WaveBudget:
    """Realized-hours ledger per wave (§10). Durable JSON so it survives a session boundary —
    §10 says multi-session is expected and the waves are the handoff seams.

    `spend()` records what an arm ACTUALLY consumed (instance hours from the launcher), and
    `check(arm, est)` refuses a launch whose estimate cannot fit in the wave's remaining budget.
    A refusal is a stop-and-journal event, not a silent skip.
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else []

    def _write(self):
        self.path.write_text(json.dumps(self.rows, indent=2, sort_keys=True))

    def spend(self, arm: str, hours: float, *, instance_type: str, run_id: str,
              curiosity: bool = False):
        assert arm in ARMS, f"unknown arm {arm}"
        assert hours >= 0, f"negative hours for {arm}"
        if curiosity:
            assert ARMS[arm]["register"] != "GRAD", (
                f"{arm} is a Register-B arm; §10 reserves the curiosity pot for curiosity "
                f"extensions and surprise follow-ups ONLY, 'never to rescue a GRAD arm'")
            spent = self.spent(curiosity=True) + hours
            assert spent <= CURIOSITY_RESERVE_L40S_H, (
                f"curiosity spend would reach {spent:.1f} h, past the §10 reserve of "
                f"{CURIOSITY_RESERVE_L40S_H} h")
        self.rows.append({"arm": arm, "wave": ARMS[arm]["wave"], "hours": float(hours),
                          "instance_type": instance_type, "run_id": run_id,
                          "curiosity": bool(curiosity)})
        self._write()
        return self.spent(ARMS[arm]["wave"])

    def spent(self, wave: str = None, *, curiosity: bool = None) -> float:
        rows = self.rows
        if wave is not None:
            rows = [r for r in rows if r["wave"] == wave]
        if curiosity is not None:
            rows = [r for r in rows if r["curiosity"] == curiosity]
        return float(sum(r["hours"] for r in rows))

    def spent_by_arm(self, arm: str) -> float:
        return float(sum(r["hours"] for r in self.rows if r["arm"] == arm))

    def remaining(self, wave: str) -> float:
        return WAVE_CAPS_L40S_H[wave] - self.spent(wave, curiosity=False)

    def curiosity_remaining(self) -> float:
        return CURIOSITY_RESERVE_L40S_H - self.spent(curiosity=True)

    def check(self, arm: str, est_hours: float) -> dict:
        """Refuse a launch that cannot fit. Also refuses an arm already over its OWN cap."""
        w = ARMS[arm]["wave"]
        rem = self.remaining(w)
        own = arm_cap(arm)
        arm_spent = self.spent_by_arm(arm)
        d = {"arm": arm, "wave": w, "est_hours": float(est_hours), "wave_remaining": rem,
             "arm_cap": own, "arm_spent": arm_spent}
        assert est_hours <= rem, (
            f"{arm}: {est_hours} L40S-h does not fit in wave {w}'s remaining {rem:.1f} h "
            f"(cap {WAVE_CAPS_L40S_H[w]}). §10's degradation order must be applied and journaled "
            f"BEFORE launching — a silent overrun is an unbudgeted wave")
        assert arm_spent + est_hours <= own or own == 0.0, (
            f"{arm}: {arm_spent:.1f} + {est_hours} h exceeds its §4 cap of {own} h")
        return d


def check_deps(arm: str, completed) -> dict:
    """§8 execution-order edges. `completed` is the set of arms that have finished."""
    assert arm in ARMS, f"unknown arm {arm}"
    done = set(completed)
    need = [d for d in DEPS.get(arm, ()) if d not in done]
    assert not need, (
        f"{arm} depends on {need}, which have not completed — §8 fixes this order and an "
        f"out-of-order launch would consume budget on an arm whose input does not exist yet")
    w = ARMS[arm]["wave"]
    first = WAVE_FIRST.get(w)
    if first and arm != first:
        assert first in done, (
            f"{arm} is in wave {w}, whose §8 order runs {first} FIRST; {first} has not completed")
    return {"arm": arm, "deps": list(DEPS.get(arm, ())), "wave_first": first, "ok": True}


def totals() -> dict:
    cov = coverage_table()
    return {"spec": SPEC_SOURCE, "queue": QUEUE_SOURCE,
            "n_arms": len(ARMS), "n_queue_items": len(QUEUE_ITEMS),
            "n_uncovered": len(uncovered()),
            "by_wave": {w: sum(1 for a in ARMS.values() if a["wave"] == w) for w in WAVE_ORDER},
            "by_register": {r: sum(1 for a in ARMS.values() if a["register"] == r)
                            for r in ("ATLAS", "GRAD")},
            "by_disposition": {d: sum(1 for r in cov if r["disposition"] == d)
                               for d in DISPOSITIONS},
            "total_cap_l40s_h": TOTAL_CAP_L40S_H,
            "curiosity_reserve_l40s_h": CURIOSITY_RESERVE_L40S_H}


# ---------------------------------------------------------------------------
# §5 / §10 — realized status, the handoff state waves are the seam of.
# ---------------------------------------------------------------------------
STATUSES = ("planned", "running", "complete", "pivot-skipped", "VOID-expected", "parked")


class ArmStatus:
    """Durable per-arm execution status + the S3 prefix that holds its artifact.

    §10 says the program is multi-session and that "the journal + coverage table" ARE the handoff
    state. A coverage table that only lists what was PLANNED cannot serve as handoff state, so the
    realized status lives beside it and the completion report renders both.
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else {}

    def set(self, arm: str, status: str, *, s3_prefix: str = None, note: str = None,
            outcome: str = None):
        assert arm in ARMS, f"unknown arm {arm}"
        assert status in STATUSES, f"unknown status {status!r} (expected one of {STATUSES})"
        if status == "complete":
            assert s3_prefix, f"{arm}: a complete arm must name the prefix holding its artifact"
        self.rows[arm] = {"arm": arm, "status": status, "s3_prefix": s3_prefix, "note": note,
                          "outcome": outcome, "wave": ARMS[arm]["wave"],
                          "queue_item": ARMS[arm]["queue_item"]}
        self.path.write_text(json.dumps(self.rows, indent=2, sort_keys=True))
        return self.rows[arm]

    def get(self, arm: str) -> dict:
        return self.rows.get(arm, {"arm": arm, "status": "planned", "s3_prefix": None,
                                   "note": None, "outcome": None,
                                   "wave": ARMS[arm]["wave"],
                                   "queue_item": ARMS[arm]["queue_item"]})

    def summary(self) -> dict:
        by = {s: 0 for s in STATUSES}
        for a in ARMS:
            by[self.get(a)["status"]] += 1
        return {"by_status": by, "n_arms": len(ARMS),
                "by_wave": {w: {s: sum(1 for a in ARMS
                                       if ARMS[a]["wave"] == w and self.get(a)["status"] == s)
                                for s in STATUSES if any(
                                    ARMS[a]["wave"] == w and self.get(a)["status"] == s
                                    for a in ARMS)}
                            for w in WAVE_ORDER}}


def render_coverage_md(status: "ArmStatus" = None) -> str:
    """The §5 coverage table, rendered for the freeze artifact and the completion report."""
    t = totals()
    out = ["# MERCATOR coverage table (§5)", "",
           f"Source: {SPEC_SOURCE} · {QUEUE_SOURCE}", "",
           f"{t['n_arms']} arms · {t['n_queue_items']} queue items · "
           f"{t['n_uncovered']} uncovered", "",
           "| queue item | disposition | arm(s) | status | note |", "|---|---|---|---|---|"]
    for r in coverage_table():
        st = "—"
        if status is not None and r["disposition"] == ARM:
            st = ", ".join(f"{a}:{status.get(a)['status']}" for a in r["arms"])
        out.append(f"| {r['queue_item']} | {r['disposition']} | {', '.join(r['arms']) or '—'} | "
                   f"{st} | {r['note']} |")
    out += ["", "## Wave budget audit (§10)", "",
            "| wave | arms | sum of per-arm caps (L40S-h) | wave cap | oversubscribed by |",
            "|---|---|---|---|---|"]
    for r in wave_cap_audit():
        out.append(f"| {r['wave']} | {r['n_arms']} | {r['sum_arm_caps_l40s_h']} | "
                   f"{r['wave_cap_l40s_h']} | {r['oversubscribed_by_l40s_h']} |")
    out += ["", f"Total cap {TOTAL_CAP_L40S_H} L40S-h "
                f"(including the {CURIOSITY_RESERVE_L40S_H} h curiosity reserve). Per-arm caps are "
                "ceilings; the wave cap is the binding budget, enforced on REALIZED hours by "
                "`WaveBudget`. §10's degradation order (drop seeds, then the largest-model cell) "
                "is applied and journaled if realized spend approaches a cap.", ""]
    return "\n".join(out)
