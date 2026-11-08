"""MENDEL — the arm registry (§2 of 2026-08-19_mendel_program_spec.md).

ONE place that names every arm, its queue item, its wave, its gates and its frozen verdict
tokens. Every job imports its own row from here rather than restating it, so `mendel_arms.tsv`,
the reports and the atlas rows cannot disagree about what an arm IS.

The verdict TOKENS are frozen here pre-freeze and are the only strings an arm may return as its
verdict (§6/e: no post-unblind rule swaps). A job that computes a verdict outside its arm's
`verdicts` tuple fails loud rather than inventing a new branch.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

# Waves, exactly as §2 names them.
WAVES = ("W0", "W1", "W2", "W3", "W4")

# Every arm's outcome is one of these plus its own arm-specific tokens. BLOCKED-WITH-UNLOCK is
# the spec's named branch for "the artifact is not as described" (§8 PLENUM row) and VOID-
# INSTRUMENT is §5's positive-control failure. Both are RESULTS, not errors.
UNIVERSAL_VERDICTS = ("BLOCKED-WITH-UNLOCK", "VOID-INSTRUMENT", "COVERAGE-IMPOSSIBLE")

# kind: GATE = wave-0 validity gate (no science verdict of its own)
#       MDE  = the program's minimum-detectable-effect arm (§3)
#       ARM  = an ordinary verdict-bearing arm
#       CHART= a charter/discovery-register arm (no confirmatory verdict)
ARMS = {
    # ---------------- Wave 0: gates ----------------
    "G1": dict(kind="GATE", item="27", wave="W0", gates=(),
               title="AMUNDSEN different-model critic pass",
               verdicts=("CONFIRMED", "DEMOTED", "MIXED"),
               kill="n/a (gate) — per-arm confirm/demote table is the deliverable"),
    "G2": dict(kind="GATE", item="13", wave="W0", gates=(),
               title="SINKFORM adversarial audit",
               verdicts=("CONFIRMED", "BROKEN", "MIXED"),
               kill="a broken step voids the T* riders in MN-4/5/14/16 and banks the break"),
    "G3": dict(kind="GATE", item="17-gate", wave="W0", gates=(),
               title="PLENUM matrix verification from artifacts",
               verdicts=("CONFIRMED", "NOT-AS-DESCRIBED"),
               kill="matrix not as described -> MN-5/14 BLOCKED-WITH-UNLOCK, ladder re-scopes"),
    "G4": dict(kind="GATE", item="12", wave="W0", gates=(),
               title="M-8 address-prediction preflight through the MN-1 instrument path",
               verdicts=("REPRODUCED", "NOT-REPRODUCED"),
               kill="the MN-1 path does not reproduce the banked M-8 number -> MN-1 is blocked"),

    # ---------------- Wave 1: CPU on banked artifacts ----------------
    "MN-1": dict(kind="MDE", item="26a", wave="W1", gates=("G4",),
                 title="Sink transport along the lineage vs the traffic-null",
                 verdicts=("INHERITED-BEYOND-TRAFFIC", "TRAFFIC-ONLY", "AMBIGUOUS"),
                 kill="no hop clears the traffic-predicted overlap -> TRAFFIC-ONLY"),
    "MN-2": dict(kind="ARM", item="26b", wave="W1", gates=("G4",),
                 title="Faculty-change co-location vs the energy-matched control",
                 verdicts=("SINK-COUPLED", "ENERGY-ONLY", "AMBIGUOUS"),
                 kill="sink fraction never exceeds the energy-matched band -> ENERGY-ONLY"),
    "MN-3": dict(kind="ARM", item="32", wave="W1", gates=(),
                 title="Frozen-probe transfer matrix",
                 verdicts=("TRANSFERS", "NO-TRANSFER", "AMBIGUOUS"),
                 kill="transfer no better than the structured (highway-projection) control"),
    "MN-4": dict(kind="ARM", item="16", wave="W1", gates=(),
                 title="T*-map + membership law + blind retrodiction battery",
                 verdicts=("LAW-HOLDS", "NO-REACH", "AMBIGUOUS"),
                 kill="membership adds nothing over activation-energy-only, or retrodictions miss"),
    "MN-5": dict(kind="ARM", item="17t0", wave="W1", gates=("G3",),
                 title="Survival-predictor ladder tiers 0 / 0.5 / 1",
                 verdicts=("TIER1-WINS", "TIER0.5-WINS", "NO-TIER-BEATS-DW", "BLOCKED-WITH-UNLOCK"),
                 kill="a tier that fails to beat the tier below out-of-sample"),
    "MN-6": dict(kind="ARM", item="18", wave="W1", gates=(),
                 title="The LoRA column: do BA deltas respect the core?",
                 verdicts=("LORA-RESPECTS-CORE", "LORA-DELIVERS-INTO-CORE", "AMBIGUOUS"),
                 kill="LoRA delivers into the core at matched-null rates -> full-FT qualifier"),
    "MN-7": dict(kind="ARM", item="15b", wave="W1", gates=(),
                 title="The SGD span signature on banked dW",
                 verdicts=("SPAN-CLEAN", "OUT-OF-SPAN", "AMBIGUOUS"),
                 kill="an SGD-trained checkpoint shows substantial out-of-span motion"),
    "MN-8": dict(kind="ARM", item="24+95", wave="W1", gates=(),
                 title="Magnitude structure as one object",
                 verdicts=("ONE-OBJECT", "CATALOG", "AMBIGUOUS"),
                 kill="case-by-case accounts win -> confound catalog, not an object"),
    "MN-9": dict(kind="ARM", item="19", wave="W1", gates=(),
                 title="What is L9",
                 verdicts=("EXCEPTIONAL", "UNEXCEPTIONAL", "AMBIGUOUS"),
                 kill="unexceptional vs matched-magnitude draws from the same spectrum tail"),
    "MN-10": dict(kind="ARM", item="20A", wave="W1", gates=(),
                  title="The Bures repair",
                  verdicts=("REPAIRED", "NO-VARIANT-PASSES", "AMBIGUOUS"),
                  kill="no cost variant passes its own split-half stability gate"),
    "MN-11": dict(kind="CHART", item="21", wave="W1", gates=(),
                  title="The flow-field mining day (curiosity reserve)",
                  verdicts=("DISCOVERY-REGISTERED", "NOTHING-CLEARS"),
                  kill="per pass: nothing clears matched within-model drift bands"),
    "MN-12": dict(kind="ARM", item="3", wave="W1", gates=(),
                  title="The width-8 kernel",
                  verdicts=("KERNEL-REPLICATES", "FLUKE", "AMBIGUOUS"),
                  kill="k-profile fails a third split, or coordinates unstable across re-draws"),

    # ---------------- Wave 2: gated composites ----------------
    "MN-13": dict(kind="ARM", item="26b-repr", wave="W2", gates=("MN-10",),
                  title="Arrangement/feature co-location per hop",
                  verdicts=("SINK-COUPLED", "ENERGY-ONLY", "AMBIGUOUS"),
                  kill="mirrors MN-2's rule on the arrangement/feature readouts"),
    "MN-14": dict(kind="ARM", item="17t2+26c", wave="W2", gates=("G2", "MN-4"),
                  title="T* at PLENUM grain and lineage grain",
                  verdicts=("THEORY-TIER-WINS", "THEORY-TIER-LOSES", "BLOCKED-WITH-UNLOCK"),
                  kill="the theory tier does not beat tier 1 -> H_wml stands"),

    # ---------------- Wave 3: GPU tail ----------------
    "MN-15": dict(kind="ARM", item="14", wave="W3", gates=("G1", "MN-4"),
                  title="The storage MDE",
                  verdicts=("STORAGE-HOLDS", "STORAGE-DIES", "AMBIGUOUS"),
                  kill="in-core payload erodes no slower than the top-energy twin"),
    "MN-16": dict(kind="ARM", item="22", wave="W3", gates=("G2",),
                  title="v-seeding with the computable dose",
                  verdicts=("BEATS-BOTH", "BEATS-EWC-ONLY", "NO-BETTER-THAN-REPLAY"),
                  kill="no better than replay, or realized horizons miss predicted ones"),
    "MN-17": dict(kind="ARM", item="25", wave="W3", gates=(),
                  title="REFEREE low-dose retest at matched realized motion",
                  verdicts=("GAIN-SURVIVES-MATCHING", "GENERIC-MOTION", "AMBIGUOUS"),
                  kill="matched controls show the same gains"),
    "MN-18": dict(kind="ARM", item="20B", wave="W3", gates=("MN-10",),
                  title="Is arrangement-destruction Adam-shaped?",
                  verdicts=("ADAM-SHAPED", "NOT-ADAM-SHAPED", "AMBIGUOUS"),
                  kill="destruction unchanged under both optimizer swaps"),

    # ---------------- Wave 4: closure ----------------
    "MN-19": dict(kind="ARM", item="9", wave="W4", gates=("G1", "G2"),
                  title="The mechanism seal",
                  verdicts=("SEALED", "NOT-SEALED", "PARTIAL"),
                  kill="per-claim kill branches named at the seal's own freeze"),
}

ROSTER = tuple(ARMS)                      # the 23 rows every report's first line counts
N_GATES = sum(1 for m in ARMS.values() if m["kind"] == "GATE")
N_ARMS = len(ARMS) - N_GATES


def arm(name: str) -> dict:
    assert name in ARMS, f"unknown arm {name!r} (roster: {sorted(ARMS)})"
    return ARMS[name]


def check_verdict(name: str, verdict: str) -> str:
    """§6/e made mechanical: an arm may only return a verdict its frozen row names.

    A new branch invented at scoring time is exactly the post-unblind rule swap the spec forbids,
    so it fails loud here rather than reaching a report.
    """
    m = arm(name)
    allowed = tuple(m["verdicts"]) + UNIVERSAL_VERDICTS
    assert verdict in allowed, (
        f"{name}: verdict {verdict!r} is not in the frozen set {allowed} — §6/e forbids inventing "
        f"a branch after unblinding; report BOTH readings under a frozen token instead")
    return verdict


def gates_for(name: str) -> tuple:
    return tuple(arm(name)["gates"])
