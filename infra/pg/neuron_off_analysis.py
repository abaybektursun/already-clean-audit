"""NEURONOFF analysis (SPEC.md §5.3-5.5, §6, §7, §9.3, §9.6 as amended A1-A12; SCHEMA.md v4 §4).

    python -m infra.pg.neuron_off_analysis freeze --model M --bank <cell>/out --pairs plan/pairs_M.json \
        --out results/thresholds_M.json
    python -m infra.pg.neuron_off_analysis unblind --model M --bank <cell>/out --pairs plan/pairs_M.json \
        --plan plan/plan_M.json --thresholds results/thresholds_M.json --journal JOURNAL.md --out results/verdict_M.json
    python -m infra.pg.neuron_off_analysis unblind --study --results results/
    python -m infra.pg.neuron_off_analysis armb_counts --bank <cell>/out --plan plan/plan_M.json \
        --out results/armb_counts_M.json [--patterns <json>]
    python -m infra.pg.neuron_off_analysis tokcheck --plan plan/plan_M.json --tokenizer <dir or tokenizer.json>
`python -m infra.pg.neuron_off_core <command> ...` runs the same commands (the module SPEC §9.6 names).

The statistics are pure functions of loaded JSON (freeze_compute, unblind_compute, study_rows, armb_counts_compute,
tokcheck_compute); the command wrappers do the reads (through the blind guard `Bank`), the git checks and the writes.

Rules fixed here where SPEC is silent (assumptions, for the spec writer to ratify):
 B1 Random signs s_j = numpy default_rng(20261025).choice([-1, 1], size=len(d)); every bootstrap draws from its own
    fresh default_rng(20261025); intervals are 2.5th and 97.5th percentiles (numpy linear).
 B2 R-c: 2,000 outer draws of size n from s_j·d_j + 1.5·δ; in each, M = median and lo = 2.5th percentile of 1,000
    inner bootstrap medians (SPEC gives no inner count); the rule fires when lo > 0 and M >= δ. Collateral and energy
    use the same positive-direction rule with δ_c, δ_e.
 B3 Room at a subset (class-stable, predictable-target rows) is recomputed at the subset's pair count.
 B4 Per-pole gates are recomputed from the per-pole file (G-ident: identity max |dlogit| == 0; G-fit from the
    contexts with the tolerances banked in the file; G-pc: mean of the banked contexts against the banked floor and
    ceiling) and must equal the job's booleans. A per-pole registered failing input that did not fail is counted
    and reported; it does not cancel the pole.
 B5 freeze reads Arm B teacher lines of conditions clean, identity, ablate_ctrl1, ablate_ctrl2 and ablate_layer and
    never ablate_pole; the ceiling (needs ablate_pole) is applied by unblind.
 B6 Arm B prompt validity: prefix_property true on the clean line. Arm B G-ident: every valid prompt's identity line
    has max_abs_dlogit_vs_clean == 0, else the pole is cancelled.
 B7 unblind takes --plan (banked_max_abs for R3). Depth tercile comes from the parent verdict file named in the
    pairs file (pairs v2 carries explainer_one but no tercile).
 B8 A pair whose per-pole file is missing is cancelled ("not measured"). The model's Arm A labels are cancelled when
    metrics.gates has a false entry or G-floor's 80% constructible rule fails; Arm B verdicts when G-known, G-det or
    G-ident's failing input fail.
 B9 Git: the thresholds file is tracked with no working-tree change, and the earliest commit adding the journal's
    "## ... PREDICTIONS" heading is a strict ancestor of the earliest commit adding the thresholds file.
 B10 F42 row: held-out contexts only (share at window position 1, poles of layers 10-11, G4).
 B11 Study rule: a model counts when its headline is not cancelled; with fewer than two such models, MIXED.
 B12 freeze, like unblind, refuses an existing output file.
 B13 The density-covariate row reports M and its interval on the |log density ratio| <= log 1.25 pairs, no label.

Code-only deviation D1, dated 2026-10-25 (after the result review; no rule and no number changes):
 - every printed label of a cancelled model carries "CANCELLED (<reasons>)": the main labels, the headline, the
   subset rows (class-stable, p(target) < 0.9; the label they would read is kept in label_if_not_cancelled) and the
   grid; the study row prints the counted models and their number;
 - Arm B verdicts state where the 95% interval of mean(D) lies against m_B* (", interval crosses margin",
   ", interval below margin", ", interval above margin");
 - an Arm B pole whose own clean activation never exceeds 0.1 of its banked maximum at any scored position (the
   pole_a_inputs of the valid prompts) reads "instrument not shown to fire" in place of SPECIAL / NOT-SPECIAL /
   UNRESOLVED;
 - unblind --grid and unblind --study --suffix write regenerated files beside the originals (defaults unchanged;
   --study matches verdict_<M><suffix>.json exactly, so a suffixed file never enters an unsuffixed study row).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess
import time

import numpy as np

SEED = 20261025
N_BOOT = 10_000
N_PLANNED = 100
RC_OUTER, RC_INNER, RC_SHIFT, RC_SHARE = 2_000, 1_000, 1.5, 0.8
ROOM_FAIL_N, ROOM_REPORT_N = 5, 10
BAL, BAL_FAIL = math.log(1.5), math.log(1.6)
GN_PAIRS, GN_NULL = 60, 20
PC_SHARE = 0.30
FLOOR_SHARE = 0.8
ARMB_MAX_EXCLUDED = 2
MB_FLOOR = 0.02
ROOM_X, ROOM_MIN = 5.0, 0.1
R3_FRAC = 0.3
P_TARGET_MAX, P_TARGET_MIN_CTX = 0.9, 20
LDR_MAX = math.log(1.25)
BEHAVIOUR_MIN, TEX_POLE, TEX_MIN = 0.10, "L04N5165+", 1.0
ARMB_FIRE_FRAC = 0.1
FREEZE_CONDITIONS = ("clean", "identity", "ablate_ctrl1", "ablate_ctrl2", "ablate_layer")
COUNT_CONDITIONS = ("clean", "ablate_pole", "ablate_ctrl1")
OWN = {"strong_pos": "H_cause", "weak_pos": "POSITIVE, BELOW δ", "strong_neg": "H_readout",
       "weak_neg": "NEGATIVE, BELOW δ", "null": "H_morcos"}
COLL = {"strong_pos": "REVERSED", "weak_pos": "MORE COLLATERAL, BELOW δ", "strong_neg": "H_spec",
        "weak_neg": "LESS COLLATERAL, BELOW δ", "null": "EQUIVALENT"}
PATTERNS = {                                  # SPEC §5.4, byte for byte
    "L07N10202-": r"\buseEffect\b",
    "L06N599+": r"\bimport\s+[\w.]+\s+as\s+\w+|\bfrom\s+[\w.]+\s+import\s+\w+\s+as\s+\w+",
    "L08N7823+": r'(?i)\breferences\s+[\w.`"\[\]]+\s*\(',
    "L11N5238+": r"(?i)\bforeign\s+key\b",
    "L27N2924-": r"\bAngular(JS)?\b|@angular/",
    "L08N3398-": r'''\stitle\s*=\s*["']|\btitle\s+attribute\b|attr\(\s*['"]title''',
    "L04N5165+": r"\\[A-Za-z]+|\\[%&$#_{}]",
}


def stem(pole: str) -> str:
    return pole.replace("+", "p").replace("-", "m")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def finite(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, np.generic):
        return finite(o.item())
    return o


def jsonl(text: str) -> list:
    """JSON lines split on "\n" only: str.splitlines() also splits on U+2028/U+2029, which JSON allows raw inside a
    string (the parent verdict file has them)."""
    return [json.loads(x) for x in text.split("\n") if x.strip()]


def cancel_stamp(reasons: list) -> str:
    return "CANCELLED (" + "; ".join(reasons) + ")"


def write_json(path, obj):
    path = pathlib.Path(path)
    assert not path.exists(), f"refusing to overwrite {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(obj), indent=1))


# ------------------------------------------------------------------------------------------ statistics
def boot(x, stat, size=None, n_boot=N_BOOT) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    vals = stat(x[rng.integers(0, len(x), (n_boot, len(x) if size is None else size))], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def median_ci(x) -> dict:
    if not len(x):
        return {"n": 0, "M": None, "lo": None, "hi": None}
    lo, hi = boot(x, np.median)
    return {"n": len(x), "M": float(np.median(x)), "lo": lo, "hi": hi}


def mean_ci(x) -> dict:
    if not len(x):
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    lo, hi = boot(x, np.mean)
    return {"n": len(x), "mean": float(np.mean(x)), "lo": lo, "hi": hi}


def signs(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).choice([-1.0, 1.0], size=n)


def ranks(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    r = np.empty(len(a))
    r[a.argsort(kind="stable")] = np.arange(len(a))
    for v in np.unique(a):
        r[a == v] = r[a == v].mean()
    return r


def spearman(x, y) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1]) if len(x) > 2 else float("nan")


def _room(d, delta, n: int) -> dict:
    """R-a, R-b, R-c of SPEC §7.2 on null differences d at pair count n (B1, B2)."""
    out = {"n": n, "R_a": num(delta) and delta > 0, "hw95": None, "morcos_room": False, "power": None,
           "direction_room": False}
    if not out["R_a"] or len(d) == 0 or n < 1:
        return out
    x = signs(len(d)) * np.asarray(d, dtype=np.float64)
    lo, hi = boot(x, np.median, size=n)
    y = x + RC_SHIFT * delta
    rng = np.random.default_rng(SEED)
    fires = 0
    for _ in range(RC_OUTER):
        s = y[rng.integers(0, len(y), n)]
        inner = np.median(s[rng.integers(0, n, (RC_INNER, n))], axis=1)
        fires += bool(np.percentile(inner, 2.5) > 0 and np.median(s) >= delta)
    out.update(hw95=(hi - lo) / 2, morcos_room=(hi - lo) / 2 < delta, power=fires / RC_OUTER,
               direction_room=fires / RC_OUTER >= RC_SHARE)
    return out


def room(d, delta, n: int) -> dict:
    """§7.2 room checks at n with their registered failing input (A1): the same checks at n = 5 must fail both R-b and
    R-c, else the room checks are cancelled ("room check cannot fail") and neither room holds; n = 10 is reported."""
    out = _room(d, delta, n)
    f5, f10 = _room(d, delta, ROOM_FAIL_N), _room(d, delta, ROOM_REPORT_N)
    can_fail = not f5["morcos_room"] and not f5["direction_room"]
    out["failing_input_n5"] = {**f5, "fails": can_fail}
    out["reported_n10"] = f10
    out["cancelled"] = not can_fail
    if not can_fail:
        out["morcos_room"] = out["direction_room"] = False
    return out


def stamps(rm: dict) -> dict:
    if rm["cancelled"]:
        return {"morcos": "room check cannot fail", "direction": "room check cannot fail"}
    return {"morcos": "room" if rm["morcos_room"] else "no room",
            "direction": "powered" if rm["direction_room"] else "underpowered for a direction"}


def label(s: dict, delta: float, rm: dict, names: dict) -> str:
    """SPEC §7.3 (A1): zero and δ read separately. names = OWN or COLL."""
    M, lo, hi = s["M"], s["lo"], s["hi"]
    if lo > 0:
        return names["strong_pos"] if M >= delta else names["weak_pos"]
    if hi < 0:
        return names["strong_neg"] if M <= -delta else names["weak_neg"]
    if -delta < lo and hi < delta and rm["morcos_room"]:
        return names["null"]
    return "INCONCLUSIVE"


def label_own(s, delta, rm):
    return label(s, delta, rm, OWN)


def label_coll(s, delta, rm):
    return label(s, delta, rm, COLL)


def refutations(s: dict) -> dict:
    """§7.3: H_cause is refuted only when hi <= 0; H_readout only when lo >= 0; H_morcos when the interval excludes 0."""
    if s["M"] is None:
        return {"H_cause": None, "H_readout": None, "H_morcos": None}
    return {"H_cause": s["hi"] <= 0, "H_readout": s["lo"] >= 0, "H_morcos": s["lo"] > 0 or s["hi"] < 0}


def headline(raw: str, energy_label: str, energy_stats: dict, bal_pass: bool) -> tuple[str, str]:
    """§7.3 energy agreement (A5)."""
    if raw.startswith("CANCELLED"):
        return raw, "cancelled"
    if not bal_pass:
        return energy_label, "energy (G-bal failed)"
    if raw in (OWN["strong_pos"], OWN["weak_pos"]) and not energy_stats["lo"] > 0 or \
            raw in (OWN["strong_neg"], OWN["weak_neg"]) and not energy_stats["hi"] < 0:
        return "direction not robust to energy normalization", "energy disagrees"
    return raw, "raw"


# ------------------------------------------------------------------------------------------ per pole
def own(rec) -> float:
    return float(np.mean([c["dnll"] for c in rec["own"]["contexts"]]))


def coll(rec) -> float:
    return rec["collateral"]["dnll"]


def energy(rec) -> float:
    return float(np.mean([c["removed_norm"] for c in rec["own"]["contexts"]]))


def own_e(rec) -> float:
    return own(rec) / energy(rec)


def own_predictable(rec):
    """own_dnll over the held-out contexts with clean p(target) < 0.9 (A6); None with fewer than 20 of them."""
    d = [c["dnll"] for c in rec["own"]["contexts"] if math.exp(-c["clean_nll"]) < P_TARGET_MAX]
    return float(np.mean(d)) if len(d) >= P_TARGET_MIN_CTX else None


def pole_gates(rec) -> dict:
    """G-ident, G-fit, G-pc recomputed from the per-pole file (B4)."""
    idn, rp, pc = rec["identity"], rec["reproduction"], rec["positive_control"]
    g_ident = num(idn["max_abs_dlogit_identity"]) and idn["max_abs_dlogit_identity"] == 0.0
    ctx = rec["own"]["contexts"]
    rel = [abs(c["a_ours"] / c["a_banked"] - 1.0) for c in ctx]
    mism = sum(not (c["a_ours"] > 0 if rec["sign"] == 1 else c["a_ours"] < 0) for c in ctx)
    g_fit = bool(rel) and float(np.percentile(rel, 50)) <= rp["median_rel_tol"] and \
        float(np.percentile(rel, 95)) <= rp["p95_rel_tol"] and mism == 0
    pm = float(np.mean([x["dnll"] for x in pc["contexts"]]))
    g_pc = num(pm) and pc["floor_nats"] <= pm <= pc["ceiling_nats"]
    assert (g_ident, g_fit, g_pc) == (idn["pass"], rp["pass"], pc["pass"]), \
        f"{rec['pole']}: recomputed gates {(g_ident, g_fit, g_pc)} differ from the job's"
    reasons = [n for n, g in (("G-ident", g_ident), ("G-fit", g_fit), ("G-pc", g_pc)) if not g]
    return {"valid": not reasons, "reasons": reasons, "fit_failing_input_failed": not rp["failing_input"]["pass"],
            "pc_failing_input_failed": not pc["failing_input"]["pass"]}


def pair_status(ids: tuple, poles: dict, gates: dict) -> list:
    reasons = []
    for q in ids:
        if poles.get(q) is None:
            reasons.append(f"{q}: not measured")
        else:
            reasons += [f"{q}: {r}" for r in gates[q]["reasons"]]
    return reasons


def null_ids(pairs: dict) -> set:
    return {x["ctrl"] for x in pairs["null_pairs"]} | {x["extra"] for x in pairs["null_pairs"]}


# ------------------------------------------------------------------------------------------ Arm B
def armb_prompts(lines: dict) -> tuple[list, list]:
    """(valid prompt ids, excluded prompt ids) by the prefix property of the clean line."""
    clean = lines["clean"]
    return sorted(i for i in clean if clean[i]["prefix_property"]), sorted(i for i in clean if not clean[i]["prefix_property"])


def r2(line: dict) -> float:
    """R2 (A9): summed log-probability up to and including the first continuation token with a non-whitespace char."""
    k = next((j for j, t in enumerate(line["cont_tokens"]) if t.strip()), len(line["cont_tokens"]) - 1)
    return float(sum(line["logp_per_token"][:k + 1]))


def delta(lines: dict, cond: str, i: int, readout: str = "R1") -> float:
    """Δ_x,j = logp_clean,j - logp_x,j (positive = the continuation became less likely)."""
    f = {"R1": lambda r: r["logp_sum"], "R2": r2}[readout]
    return f(lines["clean"][i]) - f(lines[cond][i])


def m_b(lines: dict, valid: list) -> tuple[float, float, list]:
    """§5.5 (A8): m_B = 97.5th percentile over 10,000 resamples of the prompts of |mean(D0)|, no sign flips;
    m_B* = max(m_B, 0.02 nats)."""
    d0 = np.array([delta(lines, "ablate_ctrl1", i) - delta(lines, "ablate_ctrl2", i) for i in valid])
    rng = np.random.default_rng(SEED)
    mb = float(np.percentile(np.abs(d0[rng.integers(0, len(d0), (N_BOOT, len(d0)))].mean(axis=1)), 97.5))
    return mb, max(mb, MB_FLOOR), d0.tolist()


def armb_verdict(e: dict, lines: dict, thr: dict, banked_max_abs: float, sign: int) -> dict:
    ok, excluded = armb_prompts(lines)
    out = {"pole": e["pole"], "ctrl1": e["ctrl1"], "ctrl2": e["ctrl2"], "n_prompts_valid": len(ok),
           "excluded_prompts": excluded, "m_B": thr["m_B"], "m_B_star": thr["m_B_star"], "pc_mean_drop": thr["pc_mean_drop"]}
    bad_ident = [i for i in ok if not lines["identity"][i]["max_abs_dlogit_vs_clean"] == 0.0]
    rows = {}
    for name in ("R1", "R2"):
        dp = [delta(lines, "ablate_pole", i, name) for i in ok]
        dc1 = [delta(lines, "ablate_ctrl1", i, name) for i in ok]
        rows[name] = {"delta_pole_mean": float(np.mean(dp)), "delta_ctrl1_mean": float(np.mean(dc1)),
                      "D": mean_ci([a - b for a, b in zip(dp, dc1)])}
    r3, n3 = [], 0
    for i in ok:
        a = lines["clean"][i]["pole_a_inputs"]
        ks = [k for k, x in enumerate(a) if (x > 0 if sign == 1 else x < 0) and abs(x) >= R3_FRAC * banked_max_abs]
        if not ks:
            n3 += 1
            continue
        lp = {c: lines[c][i]["logp_per_token"] for c in ("clean", "ablate_pole", "ablate_ctrl1")}
        r3.append([float(np.mean([lp["clean"][k] - lp[c][k] for k in ks])) for c in ("ablate_pole", "ablate_ctrl1")])
    rows["R3"] = {"n_prompts": len(r3), "n_excluded_no_position": n3,
                  "delta_pole_mean": float(np.mean([x[0] for x in r3])) if r3 else None,
                  "delta_ctrl1_mean": float(np.mean([x[1] for x in r3])) if r3 else None,
                  "D": mean_ci([x[0] - x[1] for x in r3])}
    out["readouts"] = rows
    layer_mean = float(np.mean([delta(lines, "ablate_layer", i) for i in ok]))
    out["ceiling"] = {"delta_pole_mean": rows["R1"]["delta_pole_mean"], "delta_layer_mean": layer_mean,
                      "pass": rows["R1"]["delta_pole_mean"] <= layer_mean}
    D, mB = rows["R1"]["D"], thr["m_B_star"]
    fire = [x * sign for i in ok for x in lines["clean"][i]["pole_a_inputs"]]
    out["pole_activation_max_frac"] = max(fire) / banked_max_abs if fire else None
    out["interval_vs_margin"] = "below" if D["hi"] <= mB else "above" if D["lo"] > mB else "crosses"
    if len(excluded) > ARMB_MAX_EXCLUDED:
        v = cancel_stamp(["prefix property fails on more than 2 prompts"])
    elif bad_ident:
        v = cancel_stamp([f"G-ident fails on prompts {bad_ident}"])
    elif not thr["room"]:
        v = "instrument not shown to fire"
    elif not out["ceiling"]["pass"]:
        v = cancel_stamp(["ceiling: the pole removes more than its whole layer; flagged for the checker"])
    elif out["pole_activation_max_frac"] is None or out["pole_activation_max_frac"] <= ARMB_FIRE_FRAC:
        v = f"instrument not shown to fire (pole activation at most {ARMB_FIRE_FRAC} of its banked maximum at every scored position)"
    else:
        v = ("SPECIAL" if D["mean"] > mB and D["lo"] > 0 else "NOT-SPECIAL" if D["mean"] <= mB else "UNRESOLVED") + \
            f", interval {out['interval_vs_margin']} margin"
    out["verdict"] = v
    return out


# ------------------------------------------------------------------------------------------ freeze
def freeze_compute(pairs: dict, poles: dict, teacher: dict, n_planned: int = N_PLANNED) -> dict:
    """Thresholds from control data only. poles: {control pole id: per-pole file or None}; teacher: {Arm B target:
    {condition: {prompt i: line}}} with FREEZE_CONDITIONS only."""
    allowed = null_ids(pairs)
    forbidden = {x["one"] for x in pairs["pairs"]} | {e["pole"] for e in pairs["arm_b"]}
    assert set(poles) <= allowed and not (set(poles) & forbidden), "freeze was handed a pole outside the null pairs"
    assert all(set(t) <= set(FREEZE_CONDITIONS) for t in teacher.values()), "freeze was handed an ablate_pole line"
    gates = {q: pole_gates(r) for q, r in poles.items() if r is not None}
    valid, cancelled = [], []
    for x in pairs["null_pairs"]:
        reasons = pair_status((x["ctrl"], x["extra"]), poles, gates)
        (cancelled.append({"pair_id": x["pair_id"], "reasons": reasons}) if reasons else valid.append(x))
    d = [own(poles[x["ctrl"]]) - own(poles[x["extra"]]) for x in valid]
    dc = [coll(poles[x["ctrl"]]) - coll(poles[x["extra"]]) for x in valid]
    de = [own_e(poles[x["ctrl"]]) - own_e(poles[x["extra"]]) for x in valid]
    dl = {k: (float(np.median(np.abs(v))) if v else float("nan")) for k, v in (("own", d), ("coll", dc), ("e", de))}
    arm_b = {}
    for e in pairs["arm_b"]:
        lines = teacher[e["pole"]]
        ok, excluded = armb_prompts(lines)
        mb, mbs, d0 = m_b(lines, ok)
        drop = float(np.mean([delta(lines, "ablate_layer", i) for i in ok]))
        arm_b[e["pole"]] = {"ctrl1": e["ctrl1"], "ctrl2": e["ctrl2"], "n_prompts_valid": len(ok),
                            "excluded_prompts": excluded, "cancelled": len(excluded) > ARMB_MAX_EXCLUDED,
                            "m_B": mb, "m_B_star": mbs, "null_diffs": d0, "pc_mean_drop": drop,
                            "room": drop >= ROOM_X * mbs and drop >= ROOM_MIN}
    return {"arm_a": {"n_null_pairs": len(pairs["null_pairs"]), "n_null_valid": len(valid), "cancelled": cancelled,
                      "delta_own": dl["own"], "delta_coll": dl["coll"], "delta_e": dl["e"],
                      "rule": "delta = median_j |d_j| over valid null pairs (SPEC §7.2)",
                      "room": {"own": room(d, dl["own"], n_planned), "coll": room(dc, dl["coll"], n_planned),
                               "energy": room(de, dl["e"], n_planned)},
                      "null_diffs_own": d, "null_diffs_coll": dc, "null_diffs_e": de,
                      "fit_failing_input_not_failed": sum(not g["fit_failing_input_failed"] for g in gates.values()),
                      "pc_failing_input_not_failed": sum(not g["pc_failing_input_failed"] for g in gates.values())},
            "arm_b": arm_b}


# ------------------------------------------------------------------------------------------ unblind
def model_gate_reasons(metrics: dict) -> list:
    g = metrics.get("gates")
    if g is None:
        return ["gates not run"]
    reasons = [k for k in ("known_number_pass", "determinism_pass", "identity_failing_input_fails") if g.get(k) is not True]
    f = g["floor"]
    built = f["n_layers"] - f["n_not_constructible"]
    if f["n_pass"] != built:
        reasons.append("G-floor: a constructible layer failed")
    if built < FLOOR_SHARE * f["n_layers"]:
        reasons.append("G-floor: fewer than 80% of layers constructible")
    return reasons


def subset_row(xs: list, D: list, delta: float, null_d: list, keep, model_reasons: list = ()) -> dict:
    """M, interval and §7.3 label on the valid pairs where keep(pair) holds, with the main δ (B3). A cancelled model's
    row reads "CANCELLED (<reasons>)"; the label it would read is kept in label_if_not_cancelled (D1)."""
    sub = [d for x, d in zip(xs, D) if keep(x)]
    s = median_ci(sub)
    rm = room(null_d, delta, len(sub)) if sub else None
    lab = label_own(s, delta, rm) if sub and rm["R_a"] else None
    row = {"stats": s, "label": cancel_stamp(list(model_reasons)) if model_reasons else lab, "stamps": stamps(rm) if rm else None}
    if model_reasons:
        row["label_if_not_cancelled"] = lab
    return row


def unblind_compute(pairs: dict, poles: dict, metrics: dict, thresholds: dict, verdict_rows: dict, teacher: dict,
                    plan_poles: dict) -> dict:
    model = pairs["model_key"]
    ta = thresholds["arm_a"]
    gates = {q: pole_gates(r) for q, r in poles.items() if r is not None}
    valid, cancelled = [], []
    for x in pairs["pairs"]:
        reasons = pair_status((x["one"], x["ctrl"]), poles, gates)
        (cancelled.append({"pair_id": x["pair_id"], "reasons": reasons}) if reasons else valid.append(x))
    measured = [q for q in poles if poles[q] is not None]
    pc_cancel_share = sum("G-pc" in gates[q]["reasons"] for q in measured) / max(len(measured), 1)
    D = [own(poles[x["one"]]) - own(poles[x["ctrl"]]) for x in valid]
    Dc = [coll(poles[x["one"]]) - coll(poles[x["ctrl"]]) for x in valid]
    De = [own_e(poles[x["one"]]) - own_e(poles[x["ctrl"]]) for x in valid]
    n, nn = len(valid), ta["n_null_valid"]
    st = {"own": median_ci(D), "coll": median_ci(Dc), "energy": median_ci(De)}
    dl = {"own": ta["delta_own"], "coll": ta["delta_coll"], "energy": ta["delta_e"]}
    rm = {"own": room(ta["null_diffs_own"], dl["own"], n), "coll": room(ta["null_diffs_coll"], dl["coll"], n),
          "energy": room(ta["null_diffs_e"], dl["energy"], n)}

    logs = [math.log(energy(poles[x["one"]]) / energy(poles[x["ctrl"]])) for x in valid]
    B = float(np.median(logs)) if logs else float("nan")
    bal = num(B) and -BAL <= B <= BAL
    B_fail = float(np.median([v + BAL_FAIL if B >= 0 else v - BAL_FAIL for v in logs])) if logs else float("nan")
    g_bal = {"B": B, "bound": BAL, "pass": bal,
             "failing_input": {"rule": "every E_ctrl divided by 1.6 when B >= 0, multiplied by 1.6 when B < 0",
                               "B": B_fail, "pass": num(B_fail) and -BAL <= B_fail <= BAL}}

    def gn(a, b):
        return a >= GN_PAIRS and b >= GN_NULL
    g_n = {"n_valid": n, "n_null_valid": nn, "pass": gn(n, nn),
           "failing_input": {"first_59_pairs": gn(min(n, GN_PAIRS - 1), nn),
                             "first_19_null_pairs": gn(n, min(nn, GN_NULL - 1))}}

    model_reasons = model_gate_reasons(metrics)
    if not num(B):
        model_reasons.append("G-bal: B is not a number")
    if pc_cancel_share > PC_SHARE:
        model_reasons.append("instrument not shown to fire (G-pc cancels more than 30% of poles)")
    if not g_n["pass"]:
        model_reasons.append(f"underpowered after gates ({n} valid pairs, {nn} valid null pairs)")

    def lab(kind, fn):
        if model_reasons:
            return cancel_stamp(model_reasons)
        if not rm[kind]["R_a"]:
            return cancel_stamp(["no range"])
        return fn(st[kind], dl[kind], rm[kind])
    labels = {"own": lab("own", label_own), "coll": lab("coll", label_coll), "energy": lab("energy", label_own)}
    head, head_rule = headline(labels["own"], labels["energy"], st["energy"], bal)

    def side(fn, role):
        return median_ci([fn(poles[x[role]]) for x in valid])
    clean_nll = lambda r: float(np.mean([c["clean_nll"] for c in r["own"]["contexts"]]))  # noqa: E731
    strata = {}
    for name, key in (("explainer", lambda x: x["explainer_one"]), ("tercile", lambda x: verdict_rows[x["one"]]["tercile"]),
                      ("sign", lambda x: poles[x["one"]]["sign"])):
        groups = {}
        for x, d in zip(valid, D):
            groups.setdefault(str(key(x)), []).append(d)
        strata[name] = {k: median_ci(v) for k, v in sorted(groups.items())}
    ms = [s["M"] for s in strata["explainer"].values() if s["M"] is not None]
    pred = [x for x in valid if own_predictable(poles[x["one"]]) is not None and own_predictable(poles[x["ctrl"]]) is not None]
    Dp = [own_predictable(poles[x["one"]]) - own_predictable(poles[x["ctrl"]]) for x in pred]
    ldr = [x["log_density_ratio"] for x in valid]
    secondary = {
        "energy_normalized": {"stats": st["energy"], "delta_e": dl["energy"], "room": rm["energy"],
                              "stamps": stamps(rm["energy"]), "label": labels["energy"]},
        "class_stable": subset_row(valid, D, dl["own"], ta["null_diffs_own"], lambda x: x["class_stable"], model_reasons),
        "density_covariate": {"spearman_D_vs_log_density_ratio": spearman(D, ldr),
                              "abs_ldr_le_log_1_25": median_ci([d for d, r in zip(D, ldr) if abs(r) <= LDR_MAX])},
        "predictable_target": {"n_pairs": len(pred), "n_pairs_dropped": n - len(pred),
                               **subset_row(pred, Dp, dl["own"], ta["null_diffs_own"], lambda x: True, model_reasons)},
        "strata": strata, "B3_explainer_same_sign": len(ms) == 2 and (ms[0] > 0) == (ms[1] > 0),
        "selectivity_own_over_coll": {r: {**median_ci([own(poles[x[r]]) / coll(poles[x[r]]) for x in valid
                                                       if coll(poles[x[r]]) != 0]),
                                          "n_zero_coll_excluded": sum(coll(poles[x[r]]) == 0 for x in valid)}
                                      for r in ("one", "ctrl")}}
    if model == "G4":
        ctx = [c for x in valid for r in ("one", "ctrl") if poles[x[r]]["layer"] in (10, 11)
               for c in poles[x[r]]["own"]["contexts"]]
        secondary["F42_heldout_share_pos1_layers_10_11"] = (sum(c["pos"] == 1 for c in ctx) / len(ctx)) if ctx else None
    arm_b = {e["pole"]: armb_verdict(e, teacher[e["pole"]], thresholds["arm_b"][e["pole"]],
                                     plan_poles[e["pole"]]["banked_max_abs"], plan_poles[e["pole"]]["sign"])
             for e in pairs["arm_b"]}
    b_reasons = [k for k in ("known_number_pass", "determinism_pass", "identity_failing_input_fails")
                 if (metrics.get("gates") or {}).get(k) is not True]           # B8: these also cancel Arm B
    if b_reasons:
        arm_b = {k: {**b, "verdict": cancel_stamp(b_reasons)} for k, b in arm_b.items()}
    est = pairs.get("estimand", {})
    return {"model_key": model, "estimand": est, "n_one_unmatched": est.get("n_one_unmatched"),
            "n_pairs": len(pairs["pairs"]), "n_valid": n, "cancelled": cancelled, "n_null_valid": nn,
            "model_reasons": model_reasons,
            "M": st["own"]["M"], "lo": st["own"]["lo"], "hi": st["own"]["hi"], "stats": st,
            "delta": dl["own"], "delta_c": dl["coll"], "delta_e": dl["energy"],
            "room": rm, "stamps": {k: stamps(v) for k, v in rm.items()},
            "label_own": labels["own"], "label_coll": labels["coll"], "label_energy": labels["energy"],
            "headline_own": head, "headline_rule": head_rule, "refutations": refutations(st["own"]),
            "B": B, "G_bal_pass": bal, "G_bal": g_bal, "G_n": g_n,
            "absolute": {"own_one": side(own, "one"), "own_ctrl": side(own, "ctrl"), "coll_one": side(coll, "one"),
                         "coll_ctrl": side(coll, "ctrl"), "clean_nll_one": side(clean_nll, "one"),
                         "clean_nll_ctrl": side(clean_nll, "ctrl")},
            "pc_cancel_share": pc_cancel_share,
            "fit_failing_input_not_failed": sum(not g["fit_failing_input_failed"] for g in gates.values()),
            "pc_failing_input_not_failed": sum(not g["pc_failing_input_failed"] for g in gates.values()),
            "secondary": secondary, "arm_b": arm_b}


def study_label(labels: list, names: dict = OWN) -> str:
    """§7.3 study rule over the headlines of models whose gates pass (B11)."""
    if len(labels) < 2:
        return "MIXED"
    c = labels.count
    pos = c(names["strong_pos"]) + c(names["weak_pos"])
    neg = c(names["strong_neg"]) + c(names["weak_neg"])
    if c(names["strong_pos"]) >= 2 and neg == 0:
        return names["strong_pos"]
    if pos >= 2 and neg == 0:
        return names["weak_pos"]
    if c(names["strong_neg"]) >= 2 and pos == 0:
        return names["strong_neg"]
    if neg >= 2 and pos == 0:
        return names["weak_neg"]
    if c(names["null"]) >= 2 and c(names["strong_pos"]) == 0 and c(names["strong_neg"]) == 0:
        return names["null"]
    return "MIXED"


def study_rows(verdicts: list) -> dict:
    own_l = [v["headline_own"] for v in verdicts if not v["headline_own"].startswith("CANCELLED")]
    coll_l = [v["label_coll"] for v in verdicts if not v["label_coll"].startswith("CANCELLED")]
    return {"models": [v["model_key"] for v in verdicts], "own": study_label(own_l, OWN),
            "coll": study_label(coll_l, COLL), "own_labels": own_l, "coll_labels": coll_l,
            "own_counted": [v["model_key"] for v in verdicts if not v["headline_own"].startswith("CANCELLED")],
            "coll_counted": [v["model_key"] for v in verdicts if not v["label_coll"].startswith("CANCELLED")]}


# ------------------------------------------------------------------------------------------ Arm B counts
def armb_counts_compute(entries: list, samples: dict, patterns: dict) -> dict:
    """SPEC §5.4 on every completion. entries: plan.arm_b; samples: {pole: [sample lines]}."""
    out = {}
    for e in entries:
        pid = e["pole"]
        pat = re.compile(patterns[pid])
        prompts = {p["i"]: p["prompt"] for p in e["prompts"]}
        per = {}
        for s in samples[pid]:
            if s["condition"] not in COUNT_CONDITIONS:
                continue
            pre = prompts[s["i"]][-200:]
            per.setdefault(s["condition"], {}).setdefault(s["i"], []).append(
                sum(1 for m in pat.finditer(pre + s["completion"]) if m.end() > len(pre)))
        table = {c: {str(i): {"n": len(v), "presence": sum(x > 0 for x in v), "mean_matches": float(np.mean(v))}
                     for i, v in sorted(per[c].items())} for c in COUNT_CONDITIONS}
        tex = pid == TEX_POLE
        rate = lambda c, i: table[c][i]["mean_matches"] if tex else table[c][i]["presence"] / table[c][i]["n"]  # noqa: E731
        ids = sorted(set.intersection(*(set(table[c]) for c in COUNT_CONDITIONS)), key=int)
        diff = [rate("ablate_ctrl1", i) - rate("ablate_pole", i) for i in ids]
        s = mean_ci(diff)
        thr = TEX_MIN if tex else BEHAVIOUR_MIN
        out[pid] = {"pattern": patterns[pid], "readout": "mean matches per completion" if tex else "presence rate",
                    "table": table, "difference": s, "threshold": thr,
                    "behaviour_changed": s["mean"] is not None and s["mean"] >= thr and s["lo"] > 0}
    return out


def tokcheck_compute(entries: list, encode) -> dict:
    """SPEC §5.3 / F1: for every Arm B prompt, tok(prompt + continuation) has tok(prompt) as a strict prefix, both
    without special tokens. encode(text) -> ids."""
    out = {}
    for e in entries:
        bad = []
        for pr in e["prompts"]:
            p, f = encode(pr["prompt"]), encode(pr["prompt"] + pr["continuation"])
            if not (len(f) > len(p) and f[:len(p)] == p):
                bad.append(pr["i"])
        out[e["pole"]] = {"n": len(e["prompts"]), "n_strict_prefix": len(e["prompts"]) - len(bad), "failing": bad}
    return out


# ------------------------------------------------------------------------------------------ files and guards
class Bank:
    """Reads a cell's out/ directory; with `allowed`, refuses any other pole id (SPEC §9.3 blind protocol)."""

    def __init__(self, root, allowed: set | None = None):
        self.root, self.allowed, self.read = pathlib.Path(root), allowed, {}

    def _load(self, path):
        self.read[str(path)] = sha256(path)
        return path.read_text()

    def json(self, name):
        return json.loads(self._load(self.root / name))

    def pole(self, pid):
        if self.allowed is not None and pid not in self.allowed:
            raise PermissionError(f"blind protocol: {pid} is not a null-pair control pole")
        path = self.root / "poles" / f"{stem(pid)}.json"
        return json.loads(self._load(path)) if path.exists() else None

    def teacher(self, target, conditions):
        out = {c: {} for c in conditions}
        for r in jsonl(self._load(self.root / "arm_b" / f"{stem(target)}_teacher.jsonl")):
            if r["condition"] in conditions:
                out[r["condition"]][r["i"]] = r
        return out

    def samples(self, target):
        return jsonl(self._load(self.root / "arm_b" / f"{stem(target)}_samples.jsonl"))


def git(cwd, *args) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def unblind_refusals(thresholds, journal):
    """B9: raises unless the thresholds file is committed and clean and the PREDICTIONS heading came strictly before."""
    thresholds, journal = pathlib.Path(thresholds).resolve(), pathlib.Path(journal).resolve()
    top = git(thresholds.parent, "rev-parse", "--show-toplevel")
    if git(top, "status", "--porcelain", "--", str(thresholds)) or not git(top, "ls-files", "--", str(thresholds)):
        raise PermissionError(f"{thresholds} is not committed or has a working-tree change")
    heads = [x for x in journal.read_text().splitlines() if x.startswith("## ") and "PREDICTIONS" in x]
    if not heads:
        raise PermissionError(f"{journal} has no PREDICTIONS heading")
    p = git(top, "log", "--format=%H", "-S", heads[0], "--", str(journal)).splitlines()
    t = git(top, "log", "--diff-filter=A", "--format=%H", "--", str(thresholds)).splitlines()
    if not p or not t:
        raise PermissionError("the PREDICTIONS heading or the thresholds file is not in git history")
    p0, t0 = p[-1], t[-1]
    ok = p0 != t0 and subprocess.run(["git", "-C", top, "merge-base", "--is-ancestor", p0, t0]).returncode == 0
    if not ok:
        raise PermissionError(f"PREDICTIONS commit {p0[:8]} is not strictly before thresholds commit {t0[:8]}")


def cmd_freeze(a):
    pairs = json.loads(pathlib.Path(a.pairs).read_text())
    assert pairs["model_key"] == a.model, f"pairs file is for {pairs['model_key']}"
    assert pairs["schema"] == "neuronoff/pairs/v2", pairs["schema"]
    allowed = null_ids(pairs)
    bank = Bank(a.bank, allowed)
    poles = {q: bank.pole(q) for q in sorted(allowed)}
    teacher = {e["pole"]: bank.teacher(e["pole"], FREEZE_CONDITIONS) for e in pairs["arm_b"]}
    thr = freeze_compute(pairs, poles, teacher)
    write_json(a.out, {"model_key": a.model, "created_utc": utc(),
                       "inputs": {a.pairs: sha256(a.pairs), **bank.read}, **thr})
    print("FREEZE_OK", json.dumps(finite({"delta_own": thr["arm_a"]["delta_own"], "delta_coll": thr["arm_a"]["delta_coll"],
                                          "n_null_valid": thr["arm_a"]["n_null_valid"],
                                          "room_own": {k: thr["arm_a"]["room"]["own"][k] for k in
                                                       ("hw95", "morcos_room", "power", "direction_room", "cancelled")},
                                          "room_own_n5_fails": thr["arm_a"]["room"]["own"]["failing_input_n5"]["fails"],
                                          "arm_b": {k: {x: v[x] for x in ("m_B", "m_B_star", "room")}
                                                    for k, v in thr["arm_b"].items()}})))


def grid_lines(v: dict) -> list:
    f = lambda x: "—" if x is None else f"{x:.4g}"  # noqa: E731
    s, c = v["stats"]["own"], v["stats"]["coll"]
    rows = [f"| {v['model_key']} | own | {v['headline_own']} ({v['headline_rule']}; raw {v['label_own']}; energy "
            f"{v['label_energy']}) | {f(s['M'])} [{f(s['lo'])}, {f(s['hi'])}] | δ {f(v['delta'])} | "
            f"{v['n_valid']}/{v['n_pairs']} pairs, {v['n_one_unmatched']} ONE unmatched |",
            f"| {v['model_key']} | collateral | {v['label_coll']} | {f(c['M'])} [{f(c['lo'])}, {f(c['hi'])}] | "
            f"δ_c {f(v['delta_c'])} | B {f(v['B'])} |"]
    for p, b in v["arm_b"].items():
        D = b["readouts"]["R1"]["D"]
        rows.append(f"| {v['model_key']} | Arm B {p} | {b['verdict']} | mean D {f(D['mean'])} [{f(D['lo'])}, "
                    f"{f(D['hi'])}] | m_B* {f(b['m_B_star'])} | {b['n_prompts_valid']} prompts |")
    return rows


def append_grid(path, rows):
    path = pathlib.Path(path)
    head = "" if path.exists() else "| model | readout | label | estimate | threshold | n |\n|---|---|---|---|---|---|\n"
    with open(path, "a") as fh:
        fh.write(head + "\n".join(rows) + "\n")


def cmd_unblind(a):
    if a.study:
        res = pathlib.Path(a.results)
        pat = re.compile(r"verdict_[A-Za-z0-9]+" + re.escape(a.suffix) + r"\.json")
        verdicts = [json.loads(p.read_text()) for p in sorted(res.glob("verdict_*.json")) if pat.fullmatch(p.name)]
        rows = study_rows(verdicts)
        append_grid(res / f"grid{a.suffix}.md", [
            f"| study | own | {rows['own']} | {rows['own_labels']} | | {len(rows['own_counted'])} counted models "
            f"{rows['own_counted']} of {rows['models']} |",
            f"| study | collateral | {rows['coll']} | {rows['coll_labels']} | | {len(rows['coll_counted'])} counted models "
            f"{rows['coll_counted']} of {rows['models']} |"])
        print("STUDY", json.dumps(rows, ensure_ascii=False))
        return
    assert not pathlib.Path(a.out).exists(), f"refusing to overwrite {a.out}"
    unblind_refusals(a.thresholds, a.journal)
    pairs = json.loads(pathlib.Path(a.pairs).read_text())
    plan = json.loads(pathlib.Path(a.plan).read_text())
    thresholds = json.loads(pathlib.Path(a.thresholds).read_text())
    assert pairs["model_key"] == plan["model_key"] == thresholds["model_key"] == a.model, "files name different models"
    vp = pairs["inputs"]["verdict_poles"]
    assert sha256(vp["path"]) == vp["sha256"], f"{vp['path']} changed since the pairs were built"
    verdict_rows = {r["pole"]: r for r in jsonl(pathlib.Path(vp["path"]).read_text())}
    bank = Bank(a.bank)
    ids = sorted({q for x in pairs["pairs"] for q in (x["one"], x["ctrl"])} | null_ids(pairs))
    poles = {q: bank.pole(q) for q in ids}
    teacher = {e["pole"]: bank.teacher(e["pole"], ("clean", "identity", "ablate_pole", "ablate_ctrl1", "ablate_ctrl2",
                                                   "ablate_layer")) for e in pairs["arm_b"]}
    v = unblind_compute(pairs, poles, bank.json("metrics.json"), thresholds, verdict_rows, teacher, plan["poles"])
    v = {"created_utc": utc(), "inputs": {a.pairs: sha256(a.pairs), a.plan: sha256(a.plan),
                                          a.thresholds: sha256(a.thresholds), vp["path"]: vp["sha256"], **bank.read}, **v}
    write_json(a.out, v)
    append_grid(a.grid or pathlib.Path(a.out).parent / "grid.md", grid_lines(v))
    print("UNBLIND_OK", json.dumps(finite({k: v[k] for k in ("model_key", "n_valid", "M", "lo", "hi", "delta",
                                                             "label_own", "label_energy", "headline_own", "label_coll",
                                                             "B", "refutations", "n_one_unmatched")}), ensure_ascii=False))


def cmd_armb_counts(a):
    plan = json.loads(pathlib.Path(a.plan).read_text())
    patterns = json.loads(pathlib.Path(a.patterns).read_text()) if a.patterns else PATTERNS
    bank = Bank(a.bank)
    samples = {e["pole"]: bank.samples(e["pole"]) for e in plan["arm_b"]}
    out = armb_counts_compute(plan["arm_b"], samples, patterns)
    write_json(a.out, {"model_key": plan["model_key"], "created_utc": utc(),
                       "inputs": {a.plan: sha256(a.plan), **bank.read}, "poles": out})
    print("ARMB_COUNTS_OK", json.dumps({p: {"behaviour_changed": r["behaviour_changed"],
                                            "difference": finite(r["difference"])} for p, r in out.items()}))


def cmd_tokcheck(a):
    from tokenizers import Tokenizer
    path = pathlib.Path(a.tokenizer)
    path = path / "tokenizer.json" if path.is_dir() else path
    tk = Tokenizer.from_file(str(path))
    plan = json.loads(pathlib.Path(a.plan).read_text())
    out = tokcheck_compute(plan["arm_b"], lambda t: tk.encode(t, add_special_tokens=False).ids)
    print("TOKCHECK", plan["model_key"], str(path), sha256(path)[:12], json.dumps(out))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neuron_off_analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for k in ("--model", "--bank", "--pairs", "--out"):
        f.add_argument(k, required=True)
    u = sub.add_parser("unblind")
    u.add_argument("--study", action="store_true")
    u.add_argument("--results")
    u.add_argument("--suffix", default="", help="--study: read verdict_<M><suffix>.json, write grid<suffix>.md")
    u.add_argument("--grid", default=None, help="grid file to append to (default: grid.md beside --out)")
    for k in ("--model", "--bank", "--pairs", "--plan", "--thresholds", "--journal", "--out"):
        u.add_argument(k)
    c = sub.add_parser("armb_counts")
    for k in ("--bank", "--plan", "--out"):
        c.add_argument(k, required=True)
    c.add_argument("--patterns", default=None, help="JSON {pole: regex}; default the frozen SPEC §5.4 table")
    t = sub.add_parser("tokcheck")
    t.add_argument("--plan", required=True)
    t.add_argument("--tokenizer", required=True, help="a tokenizer.json, or a directory holding one")
    a = ap.parse_args(argv)
    if a.cmd == "unblind" and not a.study:
        missing = [k for k in ("model", "bank", "pairs", "plan", "thresholds", "journal", "out") if getattr(a, k) is None]
        assert not missing, f"unblind needs --{', --'.join(missing)}"
    {"freeze": cmd_freeze, "unblind": cmd_unblind, "armb_counts": cmd_armb_counts, "tokcheck": cmd_tokcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
