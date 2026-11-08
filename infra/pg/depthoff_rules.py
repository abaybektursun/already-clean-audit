"""DEPTHOFF — SPEC §4's decision rules as code, frozen before any rollout existed.

Every rule has a floor, a ceiling and a cannot-decide branch, and `selftest()` plants numbers that force
each branch in both directions. The clause that matters most is rule 1's `underpowered`: an interval wider
than the equivalence margin can never return `no_loss`, however centred it is. POSTCHECK's base-model cell
and DEPTHOPEN's three arms both died on the equivalent mistake in ratio form, and both were caught only
because the rule refused rather than rounded.

    .pgvenv/bin/python -c "from infra.pg import depthoff_rules as r; print(r.selftest())"
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- frozen constants (SPEC §4)
DELTA_POINTS = 5.0          # the equivalence margin, in success-rate points
ON_FLOOR = 0.90             # G-instrument: ON's pooled rate must clear this
BROKEN_CEILING = 0.50       # G-control: the positive control must fall below this
BOOTSTRAP_N = 10_000
REMOVAL_ARMS = ("ZERO", "NOGEN")
ALL_ARMS = ("ON", "ON2", "ZERO", "NOGEN", "BROKEN")


def wilson(k: int, n: int, z: float = 1.96) -> Dict[str, float]:
    """DEPTHANCHOR's interval, with exact ends at 0 and n so no figure gets a negative bar."""
    if n == 0:
        return {"rate": 0.0, "lo": 0.0, "hi": 0.0, "k": 0, "n": 0}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return {"rate": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4), "k": int(k), "n": int(n)}


def _bootstrap_diff(a: Sequence[int], b: Sequence[int], *, seed: int,
                    n_resamples: int = BOOTSTRAP_N) -> Dict[str, float]:
    """Paired bootstrap on rate(a) - rate(b), in POINTS. The episode is the resampling unit, which is
    what makes this paired: a resample takes the (a_i, b_i) pair together, never the two independently."""
    import numpy as np
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("paired bootstrap on zero episodes")
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    point = float((x.mean() - y.mean()) * 100.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    draws = (x[idx].mean(axis=1) - y[idx].mean(axis=1)) * 100.0
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return {"point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "half_width": round((hi - lo) / 2.0, 4), "n_pairs": len(x)}


def discordance_bound_points(n_discordant: int, n_pairs: int) -> float:
    """The 95 % upper bound on |rate(a) - rate(b)| in points, from the discordant count alone.

    Why this exists: the paired difference is (a_only - b_only)/n, so its magnitude can never exceed the
    discordance rate. A paired BOOTSTRAP on perfectly concordant data returns [0, 0] at ANY n — 12 pairs
    and 12,000 pairs look identical — so the bootstrap's own width cannot be the power check. The Wilson
    upper bound on the discordance rate can: at 0 of 12 it is 24.2 points, at 0 of 200 it is 1.9.
    """
    if n_pairs <= 0:
        return 100.0
    return round(wilson(int(n_discordant), int(n_pairs))["hi"] * 100.0, 4)


def discordance(a: Sequence[int], b: Sequence[int]) -> Dict[str, object]:
    """McNemar's exact test on the paired binary outcomes. `a_only` = a succeeded where b failed."""
    if len(a) != len(b):
        raise ValueError(f"discordance needs equal lengths, got {len(a)} and {len(b)}")
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n_disc = a_only + b_only
    # exact two-sided binomial on the discordant pairs against p = 1/2
    if n_disc == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(math.comb(n_disc, i) for i in range(0, k + 1)) / (2.0 ** n_disc)
        p = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n_disc,
            "rate": round(n_disc / len(a), 4) if a else 0.0,
            "mcnemar_exact_p": round(p, 6), "n_pairs": len(a)}


def _bootstrap_discordance_gap(on: Sequence[int], null_arm: Sequence[int], test_arm: Sequence[int], *,
                               seed: int, n_resamples: int = BOOTSTRAP_N) -> Dict[str, float]:
    """Interval on d_test - d_null, both measured against the SAME ON episodes, resampled together.
    Resampling the episode index once for all three arms is what keeps the comparison paired."""
    import numpy as np
    n = len(on)
    if not (len(null_arm) == len(test_arm) == n):
        raise ValueError("the discordance gap needs all three arms on the same episodes")
    if n == 0:
        raise ValueError("the discordance gap on zero episodes")
    o = np.asarray(on, dtype=bool)
    u = np.asarray(null_arm, dtype=bool)
    t = np.asarray(test_arm, dtype=bool)
    d_null = (o != u).astype(np.float64)
    d_test = (o != t).astype(np.float64)
    point = float((d_test.mean() - d_null.mean()) * 100.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    draws = (d_test[idx].mean(axis=1) - d_null[idx].mean(axis=1)) * 100.0
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return {"point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0), "n_pairs": n}


# ---------------------------------------------------------------- the gates
def g_instrument(on: Sequence[int]) -> Dict[str, object]:
    """ON must reproduce a known number before any new one is trusted (CLAUDE.md, offline gate 4)."""
    w = wilson(sum(on), len(on))
    ok = len(on) > 0 and w["rate"] >= ON_FLOOR
    return {"gate": "G-instrument", "floor": ON_FLOOR, "on": w,
            "verdict": "pass" if ok else "broken",
            "published_per_suite": [98.4, 100.0, 98.0, 96.6],
            "why": ("ON clears the floor, so the closed loop reproduces the released policy's behaviour"
                    if ok else
                    "ON is below the floor: the instrument, not the depth pathway, is what is being "
                    "measured. No other rule may be read.")}


def g_control(broken: Optional[Sequence[int]]) -> Dict[str, object]:
    """The registered input that makes the equivalence claim fail (CLAUDE.md, instruments 2)."""
    if broken is None or len(broken) == 0:
        return {"gate": "G-control", "ceiling": BROKEN_CEILING, "broken": None,
                "verdict": "not_run",
                "why": "the positive control did not run, so no equivalence verdict may be published"}
    w = wilson(sum(broken), len(broken))
    ok = w["rate"] <= BROKEN_CEILING
    return {"gate": "G-control", "ceiling": BROKEN_CEILING, "broken": w,
            "verdict": "pass" if ok else "insensitive",
            "why": ("swapping the image collapses the loop, so the loop is sensitive to its input"
                    if ok else
                    "swapping the image did NOT collapse the loop: it is not sensitive to what it sees, "
                    "and every equivalence verdict is cancelled")}


# ---------------------------------------------------------------- rule 1
def rule1(on: Sequence[int], arm: Sequence[int], *, arm_name: str, seed: int = 20260910,
          n_resamples: int = BOOTSTRAP_N) -> Dict[str, object]:
    """SPEC §4 rule 1. `underpowered` wins over `no_loss` whenever the interval is wider than delta."""
    iv = _bootstrap_diff(on, arm, seed=seed, n_resamples=n_resamples)
    disc = discordance(on, arm)
    bound = discordance_bound_points(disc["n_discordant"], disc["n_pairs"])
    lo, hi = iv["lo"], iv["hi"]
    # WHICH width decides power, and why it is not simply the larger of the two.
    #   n_discordant <= 1: the bootstrap is degenerate — with zero flips every resample gives exactly 0,
    #     so 12 pairs and 12,000 pairs both report [0, 0]. The discordance bound is the honest width here.
    #   n_discordant >= 2: the bootstrap is the correct uncertainty in the difference. The discordance
    #     bound is NOT an uncertainty — it is the largest |diff| that count could produce if every flip
    #     went one way — so using it here would call a genuinely tight result underpowered (15/15 evenly
    #     split at n=200 is +/-3.8 pts of real uncertainty against a 20.8 pt bound).
    half = bound if disc["n_discordant"] <= 1 else iv["half_width"]
    # ORDER MATTERS: a decisive interval is decisive whatever its width. Testing power first would have
    # called a 20-point drop `underpowered`, which the planted NOGEN case caught.
    if lo > DELTA_POINTS:
        verdict, why = "loss", (
            f"the whole interval [{lo:.2f}, {hi:.2f}] pts sits above +{DELTA_POINTS} pts: switching the "
            f"depth pathway off costs task success")
    elif hi < -DELTA_POINTS:
        verdict, why = "gain", (
            f"the whole interval [{lo:.2f}, {hi:.2f}] pts sits below -{DELTA_POINTS} pts: the depth "
            f"pathway HURTS on this suite set")
    elif half > DELTA_POINTS:
        verdict, why = "underpowered", (
            f"the deciding half-width {half:.2f} pts exceeds the margin {DELTA_POINTS} pts "
            f"(bootstrap {iv['half_width']:.2f}, discordance bound {bound:.2f} on "
            f"{disc['n_discordant']} of {disc['n_pairs']} pairs), so this arm cannot return no_loss "
            f"however centred it is. This is NOT evidence of no effect.")
    elif hi < DELTA_POINTS and lo > -DELTA_POINTS:
        verdict, why = "no_loss", (
            f"the interval [{lo:.2f}, {hi:.2f}] pts lies inside +/-{DELTA_POINTS} pts and the deciding "
            f"half-width {half:.2f} pts is tighter than the margin: success is equivalent with the depth "
            f"pathway off")
    else:
        verdict, why = "inconclusive", (
            f"the interval [{lo:.2f}, {hi:.2f}] pts straddles a margin edge without clearing it")
    return {"rule": 1, "arm": arm_name, "delta_points": DELTA_POINTS, "diff_on_minus_arm": iv,
            "discordance": disc, "discordance_bound_points": bound, "deciding_half_width": round(half, 4),
            "on": wilson(sum(on), len(on)), "arm_rate": wilson(sum(arm), len(arm)),
            "verdict": verdict, "why": why}


# ---------------------------------------------------------------- rule 2
def rule2(on: Sequence[int], on2: Sequence[int], arm: Sequence[int], *, arm_name: str,
          seed: int = 20260910, n_resamples: int = BOOTSTRAP_N) -> Dict[str, object]:
    """SPEC §4 rule 2: does this arm disagree with ON more often than ON disagrees with its own repeat?"""
    gap = _bootstrap_discordance_gap(on, on2, arm, seed=seed, n_resamples=n_resamples)
    d_null = discordance(on, on2)
    d_test = discordance(on, arm)
    real = bool(gap["excludes_zero"] and gap["point"] > 0)
    return {"rule": 2, "arm": arm_name, "d_null_on_vs_on2": d_null, "d_test_on_vs_arm": d_test,
            "gap_points": gap, "verdict": "real_effect" if real else "within_noise_floor",
            "why": ("this arm flips outcomes more often than re-running the identical configuration does"
                    if real else
                    "this arm flips outcomes no more often than re-running the identical configuration "
                    "does, so the manipulation is inside the instrument's own noise floor")}


# ---------------------------------------------------------------- rule 3
def rule3(r1: Dict[str, object], r2: Dict[str, object], control: Dict[str, object],
          instrument: Dict[str, object]) -> Dict[str, object]:
    """SPEC §4 rule 3: the headline for one removal arm, reading every gate above it."""
    if instrument["verdict"] != "pass":
        return {"rule": 3, "arm": r1["arm"], "verdict": "cancelled_instrument",
                "why": "G-instrument failed; no verdict is read", "reads": ["G-instrument"]}
    if control["verdict"] != "pass":
        return {"rule": 3, "arm": r1["arm"], "verdict": "cancelled_control",
                "why": f"G-control returned {control['verdict']!r}; equivalence cannot be claimed",
                "reads": ["G-instrument", "G-control"]}
    if r1["verdict"] == "no_loss" and r2["verdict"] == "within_noise_floor":
        v, why = "no_loss", ("the paired rate difference is inside the frozen margin and the arm flips no "
                             "more outcomes than an identical re-run, so the depth pathway can be switched "
                             "off at inference without losing task success")
    elif r1["verdict"] == "loss":
        v, why = "loss", "switching the depth pathway off costs task success beyond the frozen margin"
    elif r1["verdict"] == "gain":
        v, why = "gain", "switching the depth pathway off IMPROVES task success beyond the frozen margin"
    elif r1["verdict"] == "underpowered":
        v, why = "underpowered", ("the interval is wider than the margin. This is not a negative result "
                                  "and must not be written as one")
    else:
        v, why = "inconclusive", f"rule 1 returned {r1['verdict']!r}, rule 2 {r2['verdict']!r}"
    return {"rule": 3, "arm": r1["arm"], "verdict": v, "why": why,
            "reads": ["G-instrument", "G-control", "rule1", "rule2"],
            "rule1": r1["verdict"], "rule2": r2["verdict"]}


def decide(outcomes: Dict[str, Sequence[int]], *, seed: int = 20260910,
           n_resamples: int = BOOTSTRAP_N) -> Dict[str, object]:
    """The whole verdict from per-episode binary outcomes keyed by condition name.

    `outcomes` must carry ON, ON2 and at least one removal arm, all on the same episodes in the same
    order. BROKEN may be shorter — it is its own sample, not a paired one.
    """
    for need in ("ON", "ON2"):
        if need not in outcomes:
            raise KeyError(f"outcomes must carry {need!r}; got {sorted(outcomes)}")
    on, on2 = list(outcomes["ON"]), list(outcomes["ON2"])
    instrument = g_instrument(on)
    control = g_control(outcomes.get("BROKEN"))
    arms: Dict[str, object] = {}
    for name in REMOVAL_ARMS:
        if name not in outcomes:
            continue
        arm = list(outcomes[name])
        r1 = rule1(on, arm, arm_name=name, seed=seed, n_resamples=n_resamples)
        r2 = rule2(on, on2, arm, arm_name=name, seed=seed, n_resamples=n_resamples)
        arms[name] = {"rule1": r1, "rule2": r2,
                      "rule3": rule3(r1, r2, control, instrument)}
    headline = {name: arms[name]["rule3"]["verdict"] for name in arms}
    return {"frozen": {"delta_points": DELTA_POINTS, "on_floor": ON_FLOOR,
                       "broken_ceiling": BROKEN_CEILING, "bootstrap_n": n_resamples, "seed": seed},
            "gates": {"G-instrument": instrument, "G-control": control,
                      "G-floor": discordance(on, on2)},
            "arms": arms, "headline": headline,
            "rates": {k: wilson(sum(v), len(v)) for k, v in outcomes.items()}}


# ---------------------------------------------------------------- planted-number selftest
def _pattern(n: int, k_success: int) -> List[int]:
    """k successes first, then failures — deterministic, so a planted case is reproducible."""
    return [1] * k_success + [0] * (n - k_success)


def selftest() -> Dict[str, object]:
    """Force every branch of every rule, in both directions. Any 'FAIL' means the rules are not frozen."""
    out: Dict[str, object] = {}
    n, small = 200, 2000

    # --- G-instrument: pass and broken
    out["G-instrument pass"] = g_instrument(_pattern(n, 196))["verdict"]
    out["G-instrument broken"] = g_instrument(_pattern(n, 100))["verdict"]

    # --- G-control: pass, insensitive, not_run
    out["G-control pass"] = g_control(_pattern(10, 1))["verdict"]
    out["G-control insensitive"] = g_control(_pattern(10, 9))["verdict"]
    out["G-control not_run"] = g_control(None)["verdict"]

    # --- rule 1 no_loss: identical outcomes, so the interval is a point at 0
    on = _pattern(n, 196)
    out["rule1 no_loss"] = rule1(on, list(on), arm_name="ZERO", n_resamples=small)["verdict"]
    # --- rule 1 loss: the arm drops 15 points, far outside the margin
    out["rule1 loss"] = rule1(on, _pattern(n, 166), arm_name="ZERO", n_resamples=small)["verdict"]
    # --- rule 1 gain: the arm gains 15 points
    out["rule1 gain"] = rule1(_pattern(n, 166), _pattern(n, 196), arm_name="ZERO",
                              n_resamples=small)["verdict"]
    # --- rule 1 underpowered: 12 paired episodes cannot resolve 5 points, even agreeing perfectly
    out["rule1 underpowered"] = rule1(_pattern(12, 11), _pattern(12, 11), arm_name="ZERO",
                                      n_resamples=small)["verdict"]
    # --- rule 1 must NOT call an evenly-split-but-tight result underpowered. At n=200 the half-width is
    # 0.98*sqrt(n_discordant) points, so 10 flips is +/-3.1 pts against a 9.8 pt discordance bound: real
    # equivalence. An earlier version used max() of the two widths and threw this away.
    split_on = _pattern(n, 170)[:]
    split_arm = _pattern(n, 170)[:]
    for i in range(5):                         # 5 episodes ON wins
        split_arm[i] = 0
    for i in range(170, 175):                  # 5 episodes the arm wins
        split_arm[i] = 1
    r1_split = rule1(split_on, split_arm, arm_name="ZERO", n_resamples=small)
    out["rule1 tight_despite_discordance"] = r1_split["verdict"]
    out["rule1 split discordance n"] = r1_split["discordance"]["n_discordant"]
    # --- and the other side of that boundary: 30 flips at n=200 is +/-5.4 pts, which genuinely cannot
    # resolve the 5 pt margin. The rule must say so rather than rounding to equivalence.
    wide_arm = _pattern(n, 170)[:]
    for i in range(15):
        wide_arm[i] = 0
    for i in range(170, 185):
        wide_arm[i] = 1
    out["rule1 wide_discordance"] = rule1(split_on, wide_arm, arm_name="ZERO",
                                          n_resamples=small)["verdict"]

    # --- rule 2 within_noise_floor: the arm agrees with ON exactly as often as ON2 does
    on2 = _pattern(n, 196)[:]
    on2[0], on2[1] = 0, 0                      # two flips against ON
    arm_same = _pattern(n, 196)[:]
    arm_same[2], arm_same[3] = 0, 0            # two flips too, different episodes
    out["rule2 within_noise_floor"] = rule2(on, on2, arm_same, arm_name="ZERO",
                                            n_resamples=small)["verdict"]
    # --- rule 2 real_effect: the arm flips 40 episodes, ON2 flips two
    arm_bad = _pattern(n, 196)[:]
    for i in range(40):
        arm_bad[i] = 0
    out["rule2 real_effect"] = rule2(on, on2, arm_bad, arm_name="ZERO", n_resamples=small)["verdict"]

    # --- rule 3: the four ways it refuses, and the one way it agrees
    good_inst = g_instrument(on)
    bad_inst = g_instrument(_pattern(n, 100))
    good_ctl = g_control(_pattern(10, 1))
    bad_ctl = g_control(_pattern(10, 9))
    r1_ok = rule1(on, list(on), arm_name="ZERO", n_resamples=small)
    r2_ok = rule2(on, on2, arm_same, arm_name="ZERO", n_resamples=small)
    r1_loss = rule1(on, _pattern(n, 166), arm_name="ZERO", n_resamples=small)
    r1_under = rule1(_pattern(12, 11), _pattern(12, 11), arm_name="ZERO", n_resamples=small)
    r2_under = rule2(_pattern(12, 11), _pattern(12, 11), _pattern(12, 11), arm_name="ZERO",
                     n_resamples=small)
    out["rule3 no_loss"] = rule3(r1_ok, r2_ok, good_ctl, good_inst)["verdict"]
    out["rule3 cancelled_instrument"] = rule3(r1_ok, r2_ok, good_ctl, bad_inst)["verdict"]
    out["rule3 cancelled_control"] = rule3(r1_ok, r2_ok, bad_ctl, good_inst)["verdict"]
    out["rule3 loss"] = rule3(r1_loss, r2_ok, good_ctl, good_inst)["verdict"]
    out["rule3 underpowered"] = rule3(r1_under, r2_under, good_ctl, good_inst)["verdict"]
    # the clause the study exists to protect: a wide interval must NEVER read as no_loss
    out["rule3 underpowered is not no_loss"] = (out["rule3 underpowered"] != "no_loss")

    # --- decide() end to end
    full = decide({"ON": on, "ON2": on2, "ZERO": arm_same, "NOGEN": arm_bad,
                   "BROKEN": _pattern(10, 1)}, n_resamples=small)
    out["decide ZERO"] = full["headline"]["ZERO"]
    out["decide NOGEN"] = full["headline"]["NOGEN"]

    expect = {
        "G-instrument pass": "pass", "G-instrument broken": "broken",
        "G-control pass": "pass", "G-control insensitive": "insensitive",
        "G-control not_run": "not_run",
        "rule1 no_loss": "no_loss", "rule1 loss": "loss", "rule1 gain": "gain",
        "rule1 underpowered": "underpowered",
        "rule1 tight_despite_discordance": "no_loss", "rule1 split discordance n": 10,
        "rule1 wide_discordance": "underpowered",
        "rule2 within_noise_floor": "within_noise_floor", "rule2 real_effect": "real_effect",
        "rule3 no_loss": "no_loss", "rule3 cancelled_instrument": "cancelled_instrument",
        "rule3 cancelled_control": "cancelled_control", "rule3 loss": "loss",
        "rule3 underpowered": "underpowered", "rule3 underpowered is not no_loss": True,
        "decide ZERO": "no_loss", "decide NOGEN": "loss",
    }
    failures = {k: (out[k], v) for k, v in expect.items() if out.get(k) != v}
    out["FAILURES"] = failures
    out["ok"] = not failures
    return out


if __name__ == "__main__":
    import json
    report = selftest()
    print(json.dumps(report, indent=1))
    raise SystemExit(0 if report["ok"] else 1)
