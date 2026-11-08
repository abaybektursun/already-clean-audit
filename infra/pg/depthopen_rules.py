"""DEPTHOPEN §6 — the frozen decision rules, as code, written before any arm's numbers exist.

Each function takes measured numbers and returns a verdict with the clauses that produced it, so a
write-up cites a computed record rather than a reading. The rules are SPEC §6 verbatim:

  1. Reads      median over frames of shuffle ÷ RGB ceiling exceeds m_arm, the bootstrap interval
                excludes m_arm, and removal ÷ seed floor exceeds 3. An arm with removal ÷ seed floor
                under 3 is "no room", NOT "does not read" — POSTCHECK's base-model cell is why.
  2. H_open     holds if A3 or A4 reads and A0 does not. H_need holds if no arm reads. If A0 also reads,
                the released condition's inertness is a property of their scale or data, and H_better is
                still evaluated.
  3. H_better   for the reading arm against A0: the camera-dimension difference has a 95 % two-proportion
                interval excluding 0 in the positive direction, exceeds the banked run-to-run flip rate,
                and the standard-LIBERO pooled rate is within 8 points of A0. H_harm is the same rule
                with the opposite sign. Neither, if the interval covers 0: "no difference at n = 200".
  4. Layout and per-level results are descriptive.

The two-proportion interval is Newcombe's hybrid score interval — the difference of two Wilson intervals —
because at n = 200 and rates near the floor a Wald interval on a difference can cross an impossible bound
and would hand the rule a number it must not be given.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

Z95 = 1.959963984540054
REMOVAL_ROOM = 3.0          # SPEC §6 rule 1 and §5 "room, per arm"
LIBERO_BAND_POINTS = 8.0    # SPEC §3 R4 and §6 rule 3


def wilson(k: int, n: int, z: float = Z95) -> Dict[str, float]:
    if n <= 0:
        return {"rate": 0.0, "lo": 0.0, "hi": 0.0, "k": 0, "n": 0}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"rate": p, "lo": 0.0 if k == 0 else max(0.0, centre - half),
            "hi": 1.0 if k == n else min(1.0, centre + half), "k": int(k), "n": int(n)}


def newcombe_difference(k1: int, n1: int, k2: int, n2: int, z: float = Z95) -> Dict[str, float]:
    """A 95 % interval for p1 − p2 from the two Wilson intervals (Newcombe 1998, method 10)."""
    a = wilson(k1, n1, z)
    b = wilson(k2, n2, z)
    diff = a["rate"] - b["rate"]
    lo = diff - math.sqrt((a["rate"] - a["lo"]) ** 2 + (b["hi"] - b["rate"]) ** 2)
    hi = diff + math.sqrt((a["hi"] - a["rate"]) ** 2 + (b["rate"] - b["lo"]) ** 2)
    return {"difference": diff, "lo": max(-1.0, lo), "hi": min(1.0, hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "direction": "positive" if lo > 0 else ("negative" if hi < 0 else "covers zero"),
            "arm": a, "control": b}


def flip_rate(successes_run1: Sequence[bool], successes_run2: Sequence[bool]) -> Dict[str, float]:
    """The run-to-run noise floor SPEC §5 banks: how often the SAME task flips between two runs of the
    same checkpoint, and how far the two rates ended up apart."""
    if len(successes_run1) != len(successes_run2):
        raise ValueError(f"the two runs cover {len(successes_run1)} and {len(successes_run2)} tasks")
    n = len(successes_run1)
    flips = sum(1 for a, b in zip(successes_run1, successes_run2) if bool(a) != bool(b))
    r1 = sum(1 for a in successes_run1 if a) / n if n else 0.0
    r2 = sum(1 for b in successes_run2 if b) / n if n else 0.0
    return {"n_tasks": n, "n_flipped": flips, "task_flip_rate": flips / n if n else 0.0,
            "rate_run1": r1, "rate_run2": r2, "rate_difference_abs": abs(r1 - r2)}


def rule1_reads(*, shuffle_over_ceiling: Optional[float], shuffle_ci95: Optional[Sequence[float]],
                m_arm: Optional[float], removal_over_seed_floor: Optional[float],
                arm: str = "") -> Dict[str, object]:
    """Rule 1, with its own three clauses reported whatever the outcome."""
    missing = [name for name, value in (("shuffle_over_ceiling", shuffle_over_ceiling),
                                        ("m_arm", m_arm),
                                        ("removal_over_seed_floor", removal_over_seed_floor))
               if value is None]
    if missing:
        return {"arm": arm, "verdict": "CANCELLED", "why": f"missing {missing}"}
    lo = float(shuffle_ci95[0]) if shuffle_ci95 else None
    hi = float(shuffle_ci95[1]) if shuffle_ci95 else None
    above = float(shuffle_over_ceiling) > float(m_arm)
    interval_excludes = (lo is not None and hi is not None
                         and (lo > float(m_arm) or hi < float(m_arm)))
    has_room = float(removal_over_seed_floor) > REMOVAL_ROOM
    if not has_room:
        verdict = "no room"
    elif above and interval_excludes:
        verdict = "reads"
    else:
        verdict = "does not read"
    return {"arm": arm, "verdict": verdict,
            "shuffle_over_ceiling": float(shuffle_over_ceiling), "m_arm": float(m_arm),
            "shuffle_ci95": [lo, hi],
            "point_above_margin": bool(above),
            "interval_excludes_margin": bool(interval_excludes),
            "removal_over_seed_floor": float(removal_over_seed_floor),
            "removal_room_threshold": REMOVAL_ROOM, "has_room": bool(has_room),
            "reads": verdict == "reads"}


def rule2_hypothesis(reads_by_arm: Dict[str, str]) -> Dict[str, object]:
    """Rule 2 over rule 1's verdicts. `reads_by_arm` maps arm -> "reads" / "does not read" / "no room".

    H_need means "the channel stays unused whatever the initialization", which is a claim about arms that
    HAD room to show a difference. Rule 1's own frozen text says an arm below the removal-over-floor bar is
    "no room", NOT "does not read", so an implementation that lets those arms fall into H_need contradicts
    the rule above it. When no arm has room, neither clause fires and the answer is that this instrument
    cannot decide at this scale.
    """
    reading = sorted(a for a, v in reads_by_arm.items() if v == "reads")
    no_room = sorted(a for a, v in reads_by_arm.items() if v == "no room")
    with_room = sorted(a for a, v in reads_by_arm.items() if v in ("reads", "does not read"))
    a0 = reads_by_arm.get("A0")
    treatment_reads = any(a in reading for a in ("A3", "A4"))
    if not with_room:
        holds = "undecidable: no arm has room"
        why = ("every arm's removal over seed floor is below the bar rule 1 sets, so no arm could have "
               "shown the difference; H_need is a claim about arms that had room and did not read")
    elif not reading:
        holds = "H_need"
        why = (f"no arm reads the grid among the arms with room ({', '.join(with_room)}), so LIBERO cannot "
               f"decide the question")
    elif treatment_reads and a0 != "reads":
        holds = "H_open"
        why = f"{', '.join(a for a in reading if a in ('A3', 'A4'))} reads and A0 does not"
    elif a0 == "reads":
        holds = "H_open_with_A0_reading"
        why = ("A0 reads at our scale too, so the released condition's inertness is a property of their "
               "training scale or data; H_better is still evaluated")
    else:
        holds = "no clause fires"
        why = f"arms reading: {reading}; A0 = {a0}"
    return {"verdict": holds, "why": why, "arms_reading": reading,
            "arms_with_room": with_room, "arms_without_room": no_room,
            "reads_by_arm": dict(sorted(reads_by_arm.items()))}


def rule3_better(*, arm: str, camera_arm: Dict[str, int], camera_control: Dict[str, int],
                 libero_arm: Dict[str, int], libero_control: Dict[str, int],
                 banked_flip_rate: Optional[float]) -> Dict[str, object]:
    """Rule 3. `camera_*` and `libero_*` are {"k": successes, "n": trials}."""
    diff = newcombe_difference(camera_arm["k"], camera_arm["n"],
                               camera_control["k"], camera_control["n"])
    lib_arm = wilson(libero_arm["k"], libero_arm["n"])
    lib_ctl = wilson(libero_control["k"], libero_control["n"])
    band = abs(lib_arm["rate"] - lib_ctl["rate"]) * 100.0
    within_band = band <= LIBERO_BAND_POINTS
    beats_noise = (banked_flip_rate is None
                   or abs(diff["difference"]) > float(banked_flip_rate))
    if not diff["excludes_zero"]:
        verdict = "no difference at n = %d" % camera_arm["n"]
    elif diff["direction"] == "positive" and beats_noise and within_band:
        verdict = "H_better"
    elif diff["direction"] == "negative" and beats_noise:
        verdict = "H_harm"
    elif not beats_noise:
        verdict = "inside the banked run-to-run noise"
    else:
        verdict = "camera difference excludes zero but the LIBERO band clause fails"
    return {"arm": arm, "verdict": verdict, "camera_difference": diff,
            "libero_pooled_arm": lib_arm, "libero_pooled_control": lib_ctl,
            "libero_gap_points": band, "libero_band_points": LIBERO_BAND_POINTS,
            "within_libero_band": bool(within_band),
            "banked_flip_rate": banked_flip_rate, "exceeds_banked_flip_rate": bool(beats_noise)}


def prediction_grid(*, gate_sigmoid_median: Dict[str, Optional[float]],
                    reads: Dict[str, Dict[str, object]],
                    attention_share: Dict[str, Optional[float]],
                    camera_difference: Optional[Dict[str, float]],
                    libero_gap_points: Optional[float]) -> Dict[str, object]:
    """SPEC §11's grid, filled from the measurements. H_open and H_need predict different cells, and the
    observed column is computed, not typed."""
    def cell(observed, open_pred, need_pred):
        return {"H_open_predicts": open_pred, "H_need_predicts": need_pred, "observed": observed}

    a3 = gate_sigmoid_median.get("A3")
    return {
        "A3 gate after training": cell(a3, "median sigmoid stays above 0.3", "drifts below 0.1"),
        "A3 shuffle over ceiling vs m": cell(
            (reads.get("A3") or {}).get("verdict"), "above", "below"),
        "A4 shuffle over ceiling vs m": cell(
            (reads.get("A4") or {}).get("verdict"), "above", "below"),
        "A0 shuffle over ceiling vs m": cell(
            (reads.get("A0") or {}).get("verdict"), "below", "below"),
        "A3 attention share on depth vs A0": cell(
            {"A3": attention_share.get("A3"), "A0": attention_share.get("A0")}, "higher", "equal"),
        "A3 minus A0 camera success": cell(
            camera_difference, "H_better: positive, H_harm: negative", "0"),
        "A3 minus A0 standard LIBERO": cell(libero_gap_points, "within 8", "within 8"),
    }
