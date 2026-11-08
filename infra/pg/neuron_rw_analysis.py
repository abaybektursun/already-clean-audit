"""NEURONRW analysis (studies/neuronrw/SPEC.md §5.1-§5.7, §6 analysis gates, §7, §12, amendments A1-A19; SCHEMA.md v5 §5).
Laptop only: banked JSON and local tokenizer files.

    python -m infra.pg.neuron_rw_analysis tokcheck --plan-dir P --tokenizer Q4B=<tok dir> --tokenizer G4=... --tokenizer O7=...
    python -m infra.pg.neuron_rw_analysis gates0 --model M --plan-dir P --bank <arm0 out> --results-dir R
    python -m infra.pg.neuron_rw_analysis dose_scale --model M --bank <arm0 out> --out P/dose_scale_M.json
    python -m infra.pg.neuron_rw_analysis replacement --plan-dir P --bank <Q4B arm0 out> --out P/arm1_replacement_Q4B.json
    python -m infra.pg.neuron_rw_analysis budget_min --plan-dir P --preflight <pre_q4b out>     # §9.4 L3b (A15)
    python -m infra.pg.neuron_rw_analysis budget --plan-dir P --bank Q4B=<arm0 out> ... --spent-hours H --out R/budget_<k>.json
    python -m infra.pg.neuron_rw_analysis judge0 --model M --plan-dir P --results-dir R          # J0 G-judge0 (A20 split)
    python -m infra.pg.neuron_rw_analysis freeze0 --model M --plan-dir P --results-dir R --out R/thresholds_arm0_M.json
    python -m infra.pg.neuron_rw_analysis unblind0 --model M --plan-dir P --bank <arm0 out> --results-dir R \
        --thresholds R/thresholds_arm0_M.json --journal JOURNAL.md --out R/verdict_arm0_M.json
    python -m infra.pg.neuron_rw_analysis study0 --results-dir R --bank Q4B=<arm0 out> --bank G4=... --bank O7=...
    python -m infra.pg.neuron_rw_analysis repair --model M --plan-dir P --bank <arm0 out> --results-dir R
    python -m infra.pg.neuron_rw_analysis repair --pooled --plan-dir P --results-dir R
    python -m infra.pg.neuron_rw_analysis freeze1 --model M --plan-dir P --bank <arm1 out> --arm0-bank <arm0 out> --out ...
    python -m infra.pg.neuron_rw_analysis unblind1 --model M --plan-dir P --bank <arm1 out> --arm0-bank <arm0 out> \
        --thresholds ... --journal JOURNAL.md --out R/verdict_arm1_M.json
    python -m infra.pg.neuron_rw_analysis freeze2 --model M --plan-dir P --bank <arm2 out> --arm0-bank <arm0 out> --results-dir R --out ...
    python -m infra.pg.neuron_rw_analysis unblind2 --model M ... --thresholds ... --journal JOURNAL.md --out R/verdict_arm2_M.json
    python -m infra.pg.neuron_rw_analysis study2 --results-dir R --plan-dir P
    python -m infra.pg.neuron_rw_analysis missed --results-dir R --plan-dir P --journal JOURNAL.md --bank-root bank/rw
    python -m infra.pg.neuron_rw_analysis grid --results-dir R --out R/grid.md          # regenerated from the results (C8)
    python -m infra.pg.neuron_rw_analysis literal_split --model M --plan-dir P --results-dir R --bank <arm0 out> [--arm2-bank <arm2 out>]
The Arm 2 draw is done by plan/make_plan.py (SCHEMA v5 §1.4), not here. B1 (A16) is per model in unblind0 (secondary)
and pooled over the counted models in study0 (the B1 row of the hypotheses table).

The statistics are pure functions of loaded JSON (`*_compute`); the commands read files, check git and write. Every
output file is new (never overwritten); grid rows are appended to <results>/grid.md. Bootstraps (one-way, paired over
prompts, two-way), room checks (R-null, R-pass, R-equiv) and the coherence band live in infra/pg/neuron_rw_core.py.

Labels (SPEC §1 table, §7, A4): SUPPORTED, REFUTED or EQUIVALENT only by their stated columns; an interval containing the
null and not inside the margin reads NOT SHOWN; in-between labels are printed as such. Every label carries its room
stamp ("[R-pass x, R-equiv y; stamps]") and states where its interval lies against its margin. Every label of a cancelled
model, arm or pole reads "CANCELLED (<reasons>)" in every file and in grid.md, the label it would read kept in
`label_if_not_cancelled`. An Arm 1 pole not shown to be pushed reads INSTRUMENT NOT SHOWN TO FIRE, never NO SWING.

Rules fixed here where SPEC is silent (for the spec writer to ratify):
- A1 Intervals: percentile bootstrap, 10,000 resamples, a fresh numpy default_rng(20261025) per interval; two groups
  draw from one generator; pooled rows resample within model; infinite ratios enter percentiles as +inf.
- A2 dose_scale banks the sweep's G-coh; a failed G-coh is stamped and alpha_c is still written. Liu's alpha* =
  m* x mean over the sweep poles' layers of mean_norm / rms (arm0/rms.json), m* = 1.37 (Q4B), 0.31 (G4), none (O7).
- A3 Room checks (§7.5): the simulated supported and EQUIVALENT conditions are the rule's primary columns
  (neuron_rw_core.rule_fires). Arm 1 room (§7.3): each of 2,000 outer replicates resamples the 20 prompts, makes a
  within-prompt bootstrap copy of the control's coherent samples at +-d*, plants the swing at the rate level (0.30,
  count 1.5, A18; clipped to [0, 1] for presence), and computes the SWINGS (R-pass) or NO SWING (R-equiv, unplanted copy)
  rule with an inner paired bootstrap of 1,000.
- A4 G-retest: Cohen's kappa over (original, retest) pairs; one category only gives null (cancelled branch); failing
  input: the originals of a derangement (default_rng([20261025, model_index, 11]), redrawn until no fixed point).
- A5 G-proj: `top_full[:10]` strings against the harvest vocab.json promoted strings in order; logit * w_down_norm
  against the banked values (median relative error).
- A6 Arm 1: C(alpha) per pole over all its rows at alpha (alpha = 0 rows shared); realized_cos cell = median over the
  steered rows of the (pole, dose); the threshold relation reads against +-0.20 (count 1.0) on S_t's side.
- A7 Arm 2: o_p(alpha) from fc3_steer (alpha > 0) and fc3_alpha0 (alpha = 0) answers; a (pole, dose) cell whose median
  realized_cos is below 0.95 leaves the pole out at that dose (counted); d_p = plan.primary_dose.
- A8 budget (§12, A9): fixed = arm0 manifest load_s + stage_s rms and gates + 0.1 h; Arm 1 rows / gen_rows_per_s_arm1probe
  + rows x (arm1_mean_prompt_tokens + 128) / score; Arm 2 rows / the sweep's gen_rows_per_s + rows x (plan/budget.json
  arm2 score tokens per row) / score; both plus the in-stage natural spans at the forward rate. budget_min (§9.4 L3b,
  A15): the preflight's load_s and stage_s rms + gates are read from its manifest.json (metrics.json carries the rates
  only); the same fixed hours for all three models, O7's rates only scaled by 0.56. Every cell's minutes =
  max(60, ceil(78 x hours)).
- A9 Repair: G-repair-fit's identity criterion reads tonly_identity_max_abs_dlogit where the job measured it (k = 0).
- A10 Missed predictions: SPEC §5.6 rows and the alpha_c rows of §11 (Liu's held-out values, A12).
Fail loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
import json
import math
import pathlib
import re
import subprocess
import time

import numpy as np

from infra.pg.neuron_rw_core import (EQUIV, N_BOOT, ROOM_INNER, ROOM_OUTER, ROOM_PASS_MIN, SEED, _pct, apply_room,
                                     ci, coherence_band, derangement, diff, g_coh, is_coherent, kappa, literal_overlap, num,
                                     paired_ratio_boot, pooled, read_jsonl, room, split_halves, two_group, two_way_boot)

KAPPA_MIN = 0.50
B1_EFFECT = 0.10              # SPEC §7.5 room table, B1 row (A16): planted +0.10 on the paired difference
JUDGE_FLOOR, JUDGE_CEIL = 0.60, 0.50
FIRING_LOW = (10, 12)        # A20: G-judge0 poles with 10-12 distinct firing tokens, printed apart from the rest
LETTER_TOL = 0.10
PROJ_SHARE, PROJ_REL, PROJ_OVERLAP = 0.99, 1e-3, 9            # A24: a top-10 list matches at >= 9 of 10 strings
REPAIR_FIT = {"a": 0.01, "clean": 0.002, "manual_med": 0.01, "manual_p95": 0.05, "early_min": 0.5, "out_share": 0.30}
GN = {"repair_poles": 20, "repair_pairs": 15, "arm2_share": 0.7}          # Arm 0's G-n: judge_draws a22.g_n_arm0 (A22)
ARM1_MIN_COHERENT, REALIZED_MIN = 50, 0.95
SWING = {"presence": 0.20, "count": 1.0}
NO_SWING = {"presence": 0.10, "count": 0.5}
ROOM1_PLANT = {"presence": 0.30, "count": 1.5}       # SPEC §7.3 R-pass plant (A18)
CEILING = 0.95
POOR_MATCH = math.log(2)
ALPHA_C_NONE = 64
N_LABEL_PROMPTS = 40
LIU_M = {"Q4B": 1.37, "G4": 0.31}
PLANNING_LINE = 11.0
LADDER = ("arm2 doses := {d_p, d_p/2}", "arm2 samples 10 -> 6", "arm1 samples 10 -> 6", "arm2 drops O7",
          "arm2 keeps the first 60 ONE and 30 UNK poles", "stop (P11)")
PREDICTIONS_5_6 = {   # SPEC §5.6: (predicted label, predicted |S|, predicted sign); other TeX replacements: NO SWING
    "L11N5238+": ("NO SWING", "< 0.10", None), "L27N2924-": ("SWINGS", ">= 0.20", "+"),
    "L08N3398-": ("NO SWING", "< 0.10", None), "L07N10202-": ("NO SWING", "< 0.10", None),
    "L06N599+": ("NO SWING", "< 0.10", None), "L08N7823+": ("NO SWING", "< 0.10", None),
    "L35N1881-": ("SWINGS", ">= 1.0", "-")}
ALPHA_C_PRED = {"Q4B": (64,), "G4": (16, 32)}        # SPEC §5.3 held-out prediction from Liu 2026 (A12, A17); none for O7


# ------------------------------------------------------------------------------------------ small helpers
def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


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


def write_json(path, obj):
    path = pathlib.Path(path)
    assert not path.exists(), f"refusing to overwrite {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(obj), indent=1, ensure_ascii=False))


def cancel_stamp(reasons) -> str:
    return "CANCELLED (" + "; ".join(reasons) + ")"


def relation(lo, hi, margin, name: str) -> str:
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)) or (isinstance(hi, float) and math.isnan(hi)):
        return f", interval not computed ({name})"
    if lo > margin:
        return f", interval above {name}"
    if hi < margin:
        return f", interval below {name}"
    return f", interval crosses {name}"


def stamped(label: str, reasons: list) -> dict:
    return {"label": cancel_stamp(reasons) if reasons else label, "label_if_not_cancelled": label if reasons else None}


def head(label: str) -> str:
    return label.split(",")[0].split(" [")[0]


def model_gate_reasons(metrics: dict, only=None) -> list:
    g = metrics.get("gates")
    if g is None:
        return ["job gates not run"]
    return [f"job gate {k} failed" for k, v in g.items() if k != "coherence" and v is not True and (only is None or k in only)]


def ok_interval(s: dict) -> bool:
    return s.get("lo") is not None and s.get("hi") is not None and not math.isnan(s["lo"]) and not math.isnan(s["hi"])


# ------------------------------------------------------------------------------------------ A4 label rules
def label_dir(s: dict, null: float = 0.0, eq=(-0.10, 0.10), flip: bool = False, name: str = "0") -> str:
    """SUPPORTED lo > null (flip: hi < null); REFUTED hi <= null (flip: lo >= null); EQUIVALENT inside eq; NOT SHOWN."""
    if not ok_interval(s):
        return "NOT COMPUTED"
    lo, hi = s["lo"], s["hi"]
    if (hi < null) if flip else (lo > null):
        lab = "SUPPORTED"
    elif (lo >= null) if flip else (hi <= null):
        lab = "REFUTED"
    elif eq[0] < lo and hi < eq[1]:
        lab = "EQUIVALENT"
    else:
        lab = "NOT SHOWN"
    return lab + relation(lo, hi, null, name)


def label_write(D: dict, Df: dict, delta_w: float) -> str:
    """§7.1: H_WRITE / ABOVE SHUFFLE, BELOW MARGIN / ABOVE SHUFFLE, NOT ABOVE FIXED-LABEL NULL / REFUTED / EQUIVALENT /
    NOT SHOWN."""
    if not ok_interval(D):
        return "NOT COMPUTED"
    lo, hi = D["lo"], D["hi"]
    if lo > 0 and D["est"] >= delta_w:
        lab = "H_WRITE" if ok_interval(Df) and Df["lo"] > 0 else "ABOVE SHUFFLE, NOT ABOVE FIXED-LABEL NULL"
    elif lo > 0:
        lab = "ABOVE SHUFFLE, BELOW MARGIN"
    elif hi <= 0:
        lab = "REFUTED"
    elif EQUIV["H_write"][0] < lo and hi < EQUIV["H_write"][1]:
        lab = "EQUIVALENT"
    else:
        lab = "NOT SHOWN"
    return lab + relation(lo, hi, delta_w, f"margin delta_w {delta_w:.4g}")


def label_repair(s: dict) -> str:
    if s.get("lo") is None:
        return "NOT COMPUTED"
    lo, hi = s["lo"], s["hi"]
    if hi < 0.5:
        lab = "H_REPAIR"
    elif 0.5 <= hi < 1:
        lab = "REPAIR, NOT BELOW 0.5"
    elif lo >= 1:
        lab = "REFUTED"
    elif 0.8 < lo and hi < 1.25:
        lab = "EQUIVALENT TO NO REPAIR"
    else:
        lab = "NOT SHOWN"
    return lab + relation(lo, hi, 0.5, "0.5") + relation(lo, hi, 1.0, "1") + f" (interval width {hi - lo:.3g})"


def label_suff(o: dict, paired: dict) -> str:
    if not ok_interval(o):
        return "NOT COMPUTED"
    lo, hi = o["lo"], o["hi"]
    if lo > 1 / 3:
        lab = "SUPPORTED" if ok_interval(paired) and paired["lo"] > 0 else "ABOVE CHANCE, NOT ABOVE ALPHA = 0"
    elif hi <= 1 / 3:
        lab = "REFUTED"
    elif EQUIV["H_suff"][0] < lo and hi < EQUIV["H_suff"][1]:
        lab = "EQUIVALENT TO CHANCE"
    else:
        lab = "NOT SHOWN"
    return lab + relation(lo, hi, 1 / 3, "chance 1/3") + relation(paired.get("lo"), paired.get("hi"), 0.0, "alpha = 0 (paired)")


def label_agree(R: dict) -> str:
    if not ok_interval(R):
        return "NOT COMPUTED"
    lo, hi = R["lo"], R["hi"]
    if lo > 1:
        lab = "SUPPORTED" if R["est"] >= 2 else "AGREE ABOVE DISAGREE, BELOW 2X"
    elif hi <= 1:
        lab = "REFUTED"
    elif EQUIV["H_pred_agree"][0] < lo and hi < EQUIV["H_pred_agree"][1]:
        lab = "EQUIVALENT"
    else:
        lab = "NOT SHOWN"
    return lab + relation(lo, hi, 1.0, "ratio 1") + relation(lo, hi, 2.0, "ratio 2")


def label_conf(s: dict) -> str:
    """A5: a reported equivalence row."""
    if not ok_interval(s):
        return "NOT COMPUTED"
    lo, hi = s["lo"], s["hi"]
    if -0.2 < lo and hi < 0.2:
        lab = "SUPPORTED"
    elif lo > 0 or hi < 0:
        lab = f"REFUTED ({'positive' if lo > 0 else 'negative'})"
    else:
        lab = "NOT SHOWN"
    return lab + relation(lo, hi, 0.0, "0")


def label_swing(S_t: float, d: dict, readout: str) -> str:
    """§5.4 (A3): SWINGS / NO SWING / UNRESOLVED on the paired interval d of S_t - S_c."""
    if not ok_interval(d):
        return "NOT COMPUTED"
    thr, ns = SWING[readout], NO_SWING[readout]
    side = d["lo"] > 0 if S_t > 0 else d["hi"] < 0 if S_t < 0 else False
    if abs(S_t) >= thr and side:
        lab = "SWINGS"
    elif d["lo"] <= 0 <= d["hi"] and -ns < d["lo"] and d["hi"] < ns:
        lab = "NO SWING"
    else:
        lab = "UNRESOLVED"
    m = thr if S_t >= 0 else -thr
    return lab + relation(d["lo"], d["hi"], 0.0, "0 (paired S_t - S_c)") + f", S_t {S_t:+.3g} against threshold {m:+g}"


def study_label(labels: list, supported: str) -> str:
    """§7.1 study rule; cancelled labels never count; "N counted models" when undecided (§7.7)."""
    heads = [head(l) for l in labels if not l.startswith("CANCELLED")]
    n = len(heads)
    sup, ref = sum(h == supported for h in heads), sum(h == "REFUTED" for h in heads)
    eqv = sum(h.startswith("EQUIVALENT") for h in heads)
    if n < 2:
        return f"MIXED ({n} counted models)"
    if n == 2:
        return supported if sup == 2 else "REFUTED" if ref == 2 else "EQUIVALENT" if eqv == 2 else "MIXED (2 counted models)"
    if sup >= 2 and ref == 0:
        return supported
    if ref >= 2 and sup == 0:
        return "REFUTED"
    if eqv >= 2 and sup == 0 and ref == 0:
        return "EQUIVALENT"
    return f"MIXED ({n} counted models)"


# ------------------------------------------------------------------------------------------ tokcheck (§8.1 bug 5)
def tokcheck_compute(prompt_sets: dict, neutral: list, encode, bos_id) -> dict:
    bad, n = [], 0
    for s, rows in prompt_sets.items():
        for r in rows:
            n += 1
            p, f = encode(r["prompt"]), encode(r["prompt"] + r["continuation"])
            if not (len(f) > len(p) and f[:len(p)] == p):
                bad.append({"prompt_set": s, "i": r["i"]})
    return {"n_records": n, "n_prefix_ok": n - len(bad), "prefix_failing": bad,
            "neutral_with_bos": [x for x in neutral if bos_id is not None and encode(x)[:1] == [bos_id]]}


# ------------------------------------------------------------------------------------------ gates0: G-proj, G-repair-fit
def g_proj(projections: dict, vocab: dict) -> dict:
    """§6 G-proj, amended A24: the numeric check (median relative error of our top-10 scores against the banked ones
    <= 1e-3) is primary; a pole's top-10 list matches when at least PROJ_OVERLAP of its 10 strings are in the banked
    top-10 (order ignored; multiset intersection, so repeated decoded strings count once each), and >= 0.99 of poles
    must match. Registered failing input: each pole's banked list against the top 10 of the neighbouring neuron's
    column (neuron i + 1, same layer and sign; projections.jsonl "neighbour", banked by the job from A24 on), which must
    fail the share rule (too loose if it matches on more than 1% of poles). A bank made before A24 carries no
    neighbour column; the input is then the next audited pole of another neuron in sorted order, labelled as such.
    The frozen exact-order share is printed beside."""
    poles = sorted(projections)
    neuron = lambda p: p[:-1]  # noqa: E731
    top = {p: [x["str"] for x in projections[p]["top_full"][:10]] for p in poles}
    bank = {p: [t for t, _ in vocab[p]["promoted"][:10]] for p in poles}
    overlap = lambda a, b: sum((Counter(a) & Counter(b)).values())  # noqa: E731
    exact = [top[p] == bank[p] for p in poles]
    ov = [overlap(top[p], bank[p]) for p in poles]
    has = {"neighbour" in projections[p] for p in poles}
    assert len(has) == 1, "some projection lines carry the neighbour column and some do not"
    if has == {True}:
        rule = "the banked list against our top 10 of the neighbouring neuron's column (neuron i + 1, same layer and sign)"
        ov_bad = [overlap(projections[p]["neighbour"]["top_full"][:10], bank[p]) for p in poles]
    else:
        rule = ("bank made before A24 (no neighbour column): the banked list against our top 10 of the next audited pole "
                "of another neuron in sorted order")
        nb = {p: next(poles[(i + j) % len(poles)] for j in range(1, len(poles)) if neuron(poles[(i + j) % len(poles)]) != neuron(p))
              for i, p in enumerate(poles)}
        ov_bad = [overlap(top[nb[p]], bank[p]) for p in poles]
    rel = [abs(x["logit"] * projections[p]["w_down_norm"] / v - 1.0) for p in poles
           for x, (_, v) in zip(projections[p]["top_full"][:10], vocab[p]["promoted"][:10]) if v != 0]
    share = float(np.mean([o >= PROJ_OVERLAP for o in ov]))
    fshare = float(np.mean([o >= PROJ_OVERLAP for o in ov_bad]))
    med = float(np.median(rel)) if rel else float("nan")
    numeric = num(med) and med <= PROJ_REL
    return {"n": len(poles), "median_rel_score_err": med, "numeric_pass": bool(numeric),
            "share_top10_overlap": share, "overlap_min": PROJ_OVERLAP, "overlap_histogram": {str(k): ov.count(k) for k in sorted(set(ov))},
            "pass": bool(numeric and share >= PROJ_SHARE),
            "frozen": {"rule": "top-10 lists equal in order (frozen reading)", "share_top10_equal": float(np.mean(exact)),
                       "pass": bool(numeric and float(np.mean(exact)) >= PROJ_SHARE)},
            "failing_input": {"rule": rule, "share": fshare, "median_overlap": float(np.median(ov_bad)),
                              "fails": fshare <= 1 - PROJ_SHARE}}


def repair_fit(rec: dict) -> dict:
    """G-repair-fit per pole (A23): the identity hook exact, a and the clean NLL reproduced, and the manual path run in
    the model dtype (clean_nll_manual_model_dtype) against the model's own NLL within 0.01 median / 0.05 p95; its
    registered failing input, the same path with the wrong epsilon (scale halved), must exceed those limits. The frozen
    reading (the fp32 manual path against the model's NLL, same limits) is printed beside it with its own pass/fail.
    The early-layer failing input (n_layers // 2, more than 0.5 nats) applies to both readings."""
    cs = rec["contexts"]
    F = REPAIR_FIT
    idn = [c["tonly_identity_max_abs_dlogit"] for c in cs if c["tonly_identity_max_abs_dlogit"] is not None]
    a = float(np.median([abs(c["a_ours"] / c["a_off"] - 1.0) for c in cs]))
    cl = float(np.median([abs(c["clean_nll_ours"] - c["clean_nll_off"]) for c in cs]))
    dev = lambda key: np.abs([c[key] - c["clean_nll_ours"] for c in cs])  # noqa: E731
    within = lambda d: float(np.median(d)) <= F["manual_med"] and float(np.percentile(d, 95)) <= F["manual_p95"]  # noqa: E731
    base = bool(idn) and max(idn) == 0.0 and a <= F["a"] and cl <= F["clean"]
    man, md, wrong = dev("clean_nll_manual"), dev("clean_nll_manual_model_dtype"), dev("clean_nll_manual_model_dtype_wrong_eps")
    early = float(np.median(dev("clean_nll_manual_early")))
    q = np.array([1 - math.exp(-c["clean_nll_ours"]) for c in cs])                               # 1 - p(target)
    ratio = man[q > 1e-3] / q[q > 1e-3]
    return {"tonly_identity_max": max(idn) if idn else None, "median_a_rel_err": a, "median_clean_abs": cl,
            "model_dtype_median_abs": float(np.median(md)), "model_dtype_p95_abs": float(np.percentile(md, 95)),
            "pass": bool(base and within(md)),
            "failing_input_amended": {"rule": "model-dtype manual path with epsilon 3 mean(h^2) + 4 eps (final-norm scale "
                                              "halved) must exceed 0.01 median or 0.05 p95 (A23)",
                                      "median_abs": float(np.median(wrong)), "p95_abs": float(np.percentile(wrong, 95)),
                                      "fails": not within(wrong)},
            "frozen": {"rule": "fp32 manual path against the model's NLL, 0.01 median / 0.05 p95 (frozen reading)",
                       "manual_median_abs": float(np.median(man)), "manual_p95_abs": float(np.percentile(man, 95)),
                       "pass": bool(base and within(man)),
                       "manual_err_over_1_minus_p_median": float(np.median(ratio)) if ratio.size else None},
            "failing_input": {"layer": rec["failing_input"]["layer"], "early_median_abs": early, "fails": early > F["early_min"]}}


def gates0_compute(projections: dict, vocab: dict, recs: dict) -> dict:
    fit = {p: repair_fit(r) for p, r in recs.items()}
    n_out = sum(not f["pass"] for f in fit.values())
    n_out_frozen = sum(not f["frozen"]["pass"] for f in fit.values())
    return {"g_proj": g_proj(projections, vocab), "g_repair_fit": fit,
            "repair_fit_summary": {"n": len(fit), "n_out": n_out, "share_out": n_out / len(fit) if fit else None,
                                   "failing_input_failed_everywhere": all(f["failing_input"]["fails"] for f in fit.values()),
                                   "amended_failing_input_failed_everywhere": all(f["failing_input_amended"]["fails"]
                                                                                  for f in fit.values()),
                                   "frozen_n_out": n_out_frozen, "frozen_share_out": n_out_frozen / len(fit) if fit else None}}


# ------------------------------------------------------------------------------------------ coherence (§5.2, A7)
def coherence_from(alpha0: list, cal_spans: list, inputs: list, model_key: str, label_rows=None) -> dict:
    """The arm's band from the calibration half of its alpha = 0 rows (file order) and the natural calibration spans;
    G-coh on the failing inputs and natural test spans of samples/coherence_inputs.jsonl, the test half and (Arm 2)
    the label-prompted rows."""
    cal, test = split_halves(alpha0, model_key)
    by = {r["input"]: r for r in inputs}
    nat = by["passing_1_natural_test_spans"]
    spans = [{"nll_clean_mean": v, "n_new_tokens": n} for v, n in zip(nat["nll_clean_mean"], nat["n_new_tokens"])]
    gc = g_coh(coherence_band(cal, cal_spans), {"repeated": by["failing_1_repeated_token"], "random": by["failing_2_random_attested"]},
               spans, test, label_rows)
    gc["split"] = {"n_calibration": len(cal), "n_test": len(test)}
    return gc


# ------------------------------------------------------------------------------------------ §5.3 dose scale
def dose_scale_compute(model_key: str, sweep: list, cal_spans: list, inputs: list, rms: dict, sweep_poles: list, pc: dict) -> dict:
    """§5.3; A24: when the sweep's G-coh failed, alpha_c := the largest sweep dose at which the positive-control dNLL
    stays below its ceiling. The Arm 0 bank holds one G-pc reading (controls.json pc, at the plan's pc.alpha), so the
    fallback is that dose when it passed below the ceiling; no passing G-pc reading raises. The frozen alpha_c (and the
    held-out predictions scored on it) is kept as alpha_c_frozen."""
    gc = coherence_from([r for r in sweep if r["alpha"] == 0], cal_spans, inputs, model_key)
    band = gc["band"]
    doses = sorted({r["alpha"] for r in sweep if r["alpha"] != 0})
    C = {d: float(np.mean([not is_coherent(r, band) for r in sweep if r["alpha"] == d])) for d in doses}
    pos = [d for d in doses if d > 0 and C[d] >= 0.5]
    a_frozen = min(pos) if pos else ALPHA_C_NONE
    a_c, fallback = a_frozen, None
    if not gc["pass"]:
        ok = pc["pass"] and pc["mean_dnll"] < pc["ceiling_nats"] and pc["alpha"] in doses
        assert ok, f"{model_key}: G-coh failed and no G-pc reading below the ceiling at a sweep dose: {pc['alpha']} {pc['mean_dnll']}"
        a_c = pc["alpha"]
        fallback = {"rule": "A24: G-coh failed; alpha_c := the largest sweep dose with the positive-control dNLL below its "
                            "ceiling (the bank's G-pc reading)", "pc_alpha": pc["alpha"], "pc_mean_dnll": pc["mean_dnll"],
                    "pc_ceiling_nats": pc["ceiling_nats"]}
    pred = ALPHA_C_PRED.get(model_key)
    lay = {x["layer"]: x for x in rms["layers"]}
    ratio_ = float(np.mean([lay[q["layer"]]["mean_norm"] / lay[q["layer"]]["rms"] for q in sweep_poles]))
    return {"schema": "neuronrw/dose-scale/v1", "model_key": model_key, "alpha_c": a_c, "alpha_c_frozen": a_frozen,
            "alpha_c_fallback": fallback,
            "liu_verdict_frozen": None if pred is None else ("lands" if a_frozen in pred else "missed") +
            f" (alpha_c_frozen {a_frozen} against {' or '.join(map(str, pred))})",
            "collapse_rate_by_dose": {str(d): C[d] for d in doses}, "band": {"lo": band["lo"], "hi": band["hi"]},
            "g_coh": gc, "stamp": None if gc["pass"] else "coherence gate not shown to work",
            "arm1_doses": [a_c * f for f in (-1, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1)],
            "arm2_doses": sorted({2, 4} | {a_c / 2}), "d_p": a_c / 2,
            "liu_alpha_star": None if model_key not in LIU_M else LIU_M[model_key] * ratio_,
            "liu_inputs": {"m_star": LIU_M.get(model_key), "mean_norm_over_rms_sweep_layers": ratio_,
                           "held_out_alpha_c_prediction": ALPHA_C_PRED.get(model_key)},
            "rule": "SPEC §5.3: alpha_c = the smallest positive sweep dose with C(+alpha) >= 0.5 (sweep band); 64 if none up to 32"}


# ------------------------------------------------------------------------------------------ §5.5 replacement
def firing_counts(firing: list) -> dict:
    out = {}
    for r in firing:
        out[r["pole"]] = out.get(r["pole"], 0) + bool(r["fires"])
    return out


def replacement_compute(roles: dict, firing: list, rule: dict) -> dict:
    k = firing_counts(firing)
    cands = sorted((e for e in roles["arm1"] if e["status"] == "candidate_replacement"), key=lambda e: e["order"])
    chosen = next((e["target"] for e in cands if k.get(e["target"], 0) >= rule["min_prompts"]), None)
    return {"schema": "neuronrw/arm1-replacement/v1", "model_key": roles["model_key"], "replacement": chosen, "firing": k,
            "candidates_in_order": [e["target"] for e in cands], "rule": f"SPEC §5.5: the first candidate in frozen order "
            f"with at least {rule['min_prompts']} prompts whose max sign*a >= {rule['threshold_frac']} of its banked max",
            "statement": None if chosen else f"dropped: no candidate fires on at least {rule['min_prompts']} of 20 prompts"}


# ------------------------------------------------------------------------------------------ §12 budget (A9, A15)
BUDGET_FLOOR_MIN, BUDGET_FACTOR, O7_RATE = 60, 1.3, 0.56


def budget_minutes(hours: float) -> int:
    """A15: --budget-min = max(60, ceil(60 x 1.3 x projected hours))."""
    return max(BUDGET_FLOOR_MIN, math.ceil(60 * BUDGET_FACTOR * hours))


def budget_min_compute(metrics: dict, manifest: dict, budget_plan: dict) -> dict:
    """§9.4 L3b (A15): each Arm 0 cell's hours from the pre_q4b preflight's measured forward, generation and scoring rates
    (metrics.json throughput; O7 at 0.56 of them) and its load and gate time (manifest.json load_s, stage_s rms and
    gates) plus 0.1 h, over the work counts of plan/budget.json (make_plan.hours' formula)."""
    assert (metrics["model_key"], metrics["stage"], metrics["status"]) == ("Q4B", "arm0", "complete"), \
        f"not a completed Q4B arm0 preflight: {metrics['model_key']} {metrics['stage']} {metrics['status']}"
    t = metrics["throughput"]
    fixed = (manifest["load_s"] + manifest["stage_s"].get("rms", 0.0) + manifest["stage_s"]["gates"]) / 3600 + 0.1
    cells = {}
    for m, b in budget_plan["models"].items():
        f = O7_RATE if m == "O7" else 1.0
        fwd, gen, score = (f * t[k] for k in ("forward_tokens_per_s", "gen_rows_per_s", "score_tokens_per_s"))
        h = fixed + ((b["arm0_forward_tokens"] + b["arm0_tonly_tokens"]) / fwd + b["arm0_sweep_rows"] / gen +
                     b["arm0_score_tokens"] / score) / 3600
        cells[m] = {"rate_factor": f, "projected_hours": h, "budget_min": budget_minutes(h)}
    return {"rule": "max(60, ceil(60 x 1.3 x projected hours)); SPEC §9.4 L3b, A15", "fixed_hours": fixed,
            "rates": {k: t[k] for k in ("forward_tokens_per_s", "gen_rows_per_s", "score_tokens_per_s")}, "cells": cells}


def budget_compute(cells: dict, budget_plan: dict, dose_scale: dict, spent: float, n_pairs: dict) -> dict:
    """cells: {M: {"metrics", "manifest", "plan"} of the model's arm0 cell}."""
    def fixed(m):
        man = cells[m]["manifest"]
        return (man["load_s"] + man["stage_s"].get("rms", 0.0) + man["stage_s"]["gates"]) / 3600 + 0.1

    def spans(m):
        ns, bos = cells[m]["plan"]["natural_spans"], int(cells[m]["plan"]["bos_id"] is not None)
        return (len(ns["calibration_windows"]) + len(ns["test_windows"])) * ns["spans_per_window"] * (ns["prompt_tokens"] + ns["cont_tokens"] + bos)

    def project(c):
        out = {}
        for m in cells:
            t, b = cells[m]["metrics"]["throughput"], budget_plan["models"][m]
            fwd = spans(m) / t["forward_tokens_per_s"]
            if n_pairs.get(m):
                rows = n_pairs[m] * 20 * 17 * c["n1"]
                out[f"arm1_{m}"] = fixed(m) + (rows / t["gen_rows_per_s_arm1probe"] +
                                               rows * (t["arm1_mean_prompt_tokens"] + 128) / t["score_tokens_per_s"] + fwd) / 3600
            if m in c["arm2_models"]:
                nd = 2 if c["two_doses"] else len(dose_scale[m]["arm2_doses"])
                rows = c["arm2_poles"] * nd * 5 * c["n2"] + 5 * c["n2"] + N_LABEL_PROMPTS * c["n2"]
                out[f"arm2_{m}"] = fixed(m) + (rows / t["gen_rows_per_s"] + rows * b["arm2_score_tokens"] / b["arm2_rows"] /
                                               t["score_tokens_per_s"] + fwd) / 3600
        return out
    c = {"two_doses": False, "n2": 10, "n1": 10, "arm2_models": sorted(cells), "arm2_poles": 120, "arm2_one": 80}
    edits = [lambda c: c.update(two_doses=True), lambda c: c.update(n2=6), lambda c: c.update(n1=6),
             lambda c: c.update(arm2_models=[m for m in c["arm2_models"] if m != "O7"]), lambda c: c.update(arm2_poles=90, arm2_one=60)]
    steps = []
    for k in range(len(LADDER)):
        pr = project(c)
        total = spent + sum(pr.values())
        steps.append({"step": k, "applied": list(LADDER[:k]), "config": dict(c), "projected_hours": pr, "total": total,
                      "budget_min": {cell: budget_minutes(h) for cell, h in pr.items()},
                      "fits": total <= PLANNING_LINE, "g_n_arm2_min_poles": math.ceil(GN["arm2_share"] * c["arm2_one"])})
        if total <= PLANNING_LINE or k == len(edits):
            break
        edits[k](c)
    last = steps[-1]
    return {"spent_hours": spent, "planning_line": PLANNING_LINE, "fixed_hours": {m: fixed(m) for m in cells}, "steps": steps,
            "decision": "run" if last["fits"] else "stop (P11): the ladder's last step does not fit",
            "applied": last["applied"] if last["fits"] else list(LADDER)}


# ------------------------------------------------------------------------------------------ Arm 0 (§5.1, §7.1)
def judge_rows(results_dir, batch: str) -> list:
    return read_jsonl(pathlib.Path(results_dir) / f"judge_{batch}.jsonl")


def yes(rows: list, task: str, label_set: str = "reexplained") -> dict:
    return {r["pole"]: (1 if r["answer"] == "yes" else 0) for r in rows
            if r["task"] == task and r["label_set"] == label_set and r["answer"] is not None and not r["preflight"]}


def retest_gate(orig: list, re_rows: list, model_key: str) -> dict:
    by_id = {r["id"]: r for r in orig}
    out = {}
    for task in sorted({by_id[r["retest_of"]]["task"] for r in re_rows}):
        pairs = [(by_id[r["retest_of"]]["answer"], r["answer"]) for r in re_rows if by_id[r["retest_of"]]["task"] == task
                 and r["answer"] is not None and by_id[r["retest_of"]]["answer"] is not None]
        a, b = [x for x, _ in pairs], [y for _, y in pairs]
        k = kappa(a, b)
        kf = kappa([a[i] for i in derangement(len(a), model_key)], b)
        ok = k is not None and k >= KAPPA_MIN
        out[task] = {"n": len(pairs), "kappa": k, "pass": ok, "failing_input": {"kappa": kf, "fails": kf is None or kf < KAPPA_MIN},
                     "stamp": None if ok else f"judge unreliable (kappa {k if k is None else round(k, 3)})"}
    return out


def judge0_gate(pt: dict, ps: dict, shuf: float, n_distinct: dict) -> dict:
    """§6 G-judge0 on {pole: 0/1} pc_true and pc_shuf answers and the shuffle yes-rate; A20: the pc_true yes-rate is
    printed for poles with 10-12 distinct firing tokens and for the rest, and "floor too low" is noted beside the label
    when the 10-12 group is below the floor while the rest is at or above it. The gate reads all poles."""
    assert set(pt) <= set(n_distinct), f"G-judge0 poles without a plan n_distinct: {sorted(set(pt) - set(n_distinct))[:5]}"
    rate = lambda v: float(np.mean(v)) if len(v) else float("nan")  # noqa: E731
    low = [v for p, v in pt.items() if FIRING_LOW[0] <= n_distinct[p] <= FIRING_LOW[1]]
    rest = [v for p, v in pt.items() if n_distinct[p] > FIRING_LOW[1]]
    j0 = {"pc_true_yes_rate": rate(list(pt.values())), "n_pc": len(pt), "floor": JUDGE_FLOOR, "shuffle_yes_rate": shuf,
          "ceiling": JUDGE_CEIL, "failing_input": {"pc_shuf_yes_rate": rate(list(ps.values())),
                                                   "fails": num(rate(list(ps.values()))) and rate(list(ps.values())) < JUDGE_FLOOR},
          "by_distinct": {"10-12": {"n": len(low), "pc_true_yes_rate": rate(low)},
                          "13-20": {"n": len(rest), "pc_true_yes_rate": rate(rest)}}}
    reasons = ([] if num(j0["pc_true_yes_rate"]) and j0["pc_true_yes_rate"] >= JUDGE_FLOOR else ["judge not shown to fire"]) + \
        ([] if num(shuf) and shuf <= JUDGE_CEIL else ["judge not discriminating"]) + \
        ([] if j0["failing_input"]["fails"] else ["G-judge0 failing input did not fail"])
    note = num(rate(low)) and num(rate(rest)) and rate(low) < JUDGE_FLOOR <= rate(rest)
    j0["floor_note"] = (f"floor too low (10-12 distinct: yes-rate {rate(low):.3f} on {len(low)} poles, below {JUDGE_FLOOR}; "
                        f"the rest {rate(rest):.3f} on {len(rest)})") if note else None
    j0["label"] = ("PASS" if not reasons else "FAIL (" + "; ".join(reasons) + ")") + (f" [{j0['floor_note']}]" if note else "")
    return {**j0, "pass": not reasons, "reasons": reasons}


def freeze0_compute(model_key: str, roles: dict, ctrl_rows: list, retest_rows: list, pc_rows: list, n_distinct: dict,
                    outer=ROOM_OUTER, inner=ROOM_INNER) -> dict:
    """Control data only (SPEC §7.5): yn_shuf1, yn_shuf2, yn_fixed and pc_* answers and the retest pairs. B1 (A22: the
    paired yn_true difference, with-vocabulary minus re-explained) takes as its null the paired difference of the two
    exchangeable shuffled answers y_s1 - y_s2 of the same ONE items."""
    ret = {x["retest_of"] for x in retest_rows}
    assert not any(r["task"] in ("yn_true", "fc3") and r["id"] not in ret for r in ctrl_rows), \
        "freeze0 was handed a yn_true or fc3 answer outside the retest pairs"
    s1, s2, fx = yes(ctrl_rows, "yn_shuf1"), yes(ctrl_rows, "yn_shuf2"), yes(ctrl_rows, "yn_fixed")
    one = [p for p in roles["one"] if p in s1 and p in s2]
    d = np.array([s1[p] - s2[p] for p in one], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    delta_w = float(np.percentile(np.abs(d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)), 97.5)) if len(d) else None
    terc = {t: [p for p in one if roles["poles"][p]["tercile"] == t] for t in (0, 2)}
    pairs = [(s1[x["one"]] - s2[x["one"]]) - (s1[x["ctrl"]] - s2[x["ctrl"]]) for x in roles["pairs"]
             if all(q in s1 and q in s2 for q in (x["one"], x["ctrl"]))]
    null = {"H_write": d.tolist(), "H_depth_t2": [s1[p] - s2[p] for p in terc[2]], "H_depth_t0": [s1[p] - s2[p] for p in terc[0]],
            "H_ONE": pairs, "B1": d.tolist()}
    psh = float(np.mean([s1[p] for p in one] + [s2[p] for p in one])) if one else float("nan")    # A25: p_sh on ONE items
    rec = lambda p: [s1[p], s2[p], psh]  # noqa: E731
    plant = {"p_sh": psh, "H_write": [rec(p) for p in one], "H_depth": ([rec(p) for p in terc[2]], [rec(p) for p in terc[0]]),
             "H_ONE": [[s1[x["one"]], s2[x["one"]], s1[x["ctrl"]], s2[x["ctrl"]], psh] for x in roles["pairs"]
                       if all(q in s1 and q in s2 for q in (x["one"], x["ctrl"]))],
             "B1": [rec(p) for p in one]}
    rooms = {"H_write": room("H_write", null["H_write"], len(d), 0.15, delta_w or 0.0, outer, inner, plant["H_write"]),
             "H_depth": room("H_depth", (null["H_depth_t2"], null["H_depth_t0"]), (len(terc[2]), len(terc[0])), 0.20, 0.0, outer,
                             inner, plant["H_depth"]),
             "H_ONE_model": room("H_ONE", pairs, len(pairs), 0.15, 0.0, outer, inner, plant["H_ONE"]),
             "B1_model": room("B1", null["B1"], len(null["B1"]), B1_EFFECT, 0.0, outer, inner, plant["B1"])}
    sh = list(s1.values()) + list(s2.values())
    g = judge0_gate(yes(pc_rows, "pc_true"), yes(pc_rows, "pc_shuf"), float(np.mean(sh)) if sh else float("nan"), n_distinct)
    return {"model_key": model_key, "arm": 0, "delta_w": delta_w, "n_one_shuffled": len(one),
            "rule_delta_w": "SPEC §7.1: 97.5th percentile of |mean(y_s1 - y_s2)| over 10,000 bootstrap resamples of ONE items",
            "shuffle_rate": ci([(s1[p] + s2[p]) / 2 for p in one]), "fixed_rate": ci([fx[p] for p in one if p in fx]),
            "null_values": null, "plant": plant, "room": rooms, "g_retest": retest_gate(ctrl_rows, retest_rows, model_key),
            "g_judge0": g}


def item_scores(roles: dict, yn_rows: list, fc3_rows: list, label_set: str) -> dict:
    t, s1, s2 = yes(yn_rows, "yn_true", label_set), yes(yn_rows, "yn_shuf1", label_set), yes(yn_rows, "yn_shuf2", label_set)
    fx = yes(yn_rows, "yn_fixed", label_set)
    c3 = {r["pole"]: (int(bool(r["correct"])), r["answer"]) for r in fc3_rows
          if r["task"] == "fc3" and r["label_set"] == label_set and r["answer"] is not None and not r["preflight"]}
    out = {}
    for p in roles["poles"]:
        if not all(p in d for d in (t, s1, s2)):
            continue
        out[p] = {"y_true": t[p], "y_s1": s1[p], "y_s2": s2[p], "s": t[p] - (s1[p] + s2[p]) / 2,
                  "y_fixed": fx.get(p), "f": None if p not in fx else t[p] - fx[p], "c3": c3.get(p, (None, None))[0],
                  "choice": c3.get(p, (None, None))[1]}
    return out


def unblind0_compute(model_key: str, roles: dict, thr: dict, gates0: dict, yn_rows: list, yn_vocab_rows: list, yn_orig_rows: list,
                     fc3_rows: list, metrics: dict, projections: dict, reexplain: dict, reexplain_vocab: dict, sub: dict,
                     outer=ROOM_OUTER, inner=ROOM_INNER) -> dict:
    """§7.1 on re-explained labels (primary); with-vocabulary labels for B1 (A22: paired yn_true difference; per model
    here, pooled in study0); original labels in secondary rows only. sub = judge_draws a22 block (task lists per class,
    g_n_arm0)."""
    it = item_scores(roles, yn_rows, fc3_rows, "reexplained")
    need = {c: [{"yn_fixed": "y_fixed", "fc3": "c3"}[t] for t in ts if t in ("yn_fixed", "fc3")] for c, ts in sub["tasks"].items()}
    full = {p: v for p, v in it.items() if all(v[k] is not None for k in need[roles["poles"][p]["class"]])}   # every primary answer
    yt = {"with_vocab": yes(yn_vocab_rows, "yn_true", "with_vocab"), "original": yes(yn_orig_rows, "yn_true", "original")}
    one = [p for p in roles["one"] if p in full]
    S = lambda ps, src=full: [src[p]["s"] for p in ps]  # noqa: E731
    D, Df = ci(S(one)), ci([full[p]["f"] for p in one])
    A, Sh, Fx = ci([full[p]["y_true"] for p in one]), ci([(full[p]["y_s1"] + full[p]["y_s2"]) / 2 for p in one]), \
        ci([full[p]["y_fixed"] for p in one])
    terc = {t: [p for p in one if roles["poles"][p]["tercile"] == t] for t in (0, 1, 2)}
    depth = two_group(S(terc[2]), S(terc[0]), diff)
    pairs = [x for x in roles["pairs"] if x["one"] in full and x["ctrl"] in full]
    pv = [full[x["one"]]["s"] - full[x["ctrl"]]["s"] for x in pairs]
    b1_vals = [yt["with_vocab"][p] - full[p]["y_true"] for p in one if p in yt["with_vocab"]]
    b1 = ci(b1_vals)
    gn = {"n_items": len(full), "need": sub["g_n_arm0"], "pass": len(full) >= sub["g_n_arm0"],
          "failing_input": {"n": sub["g_n_arm0"] - 1, "fails": sub["g_n_arm0"] - 1 < sub["g_n_arm0"]}}
    reasons = model_gate_reasons(metrics)
    gp = gates0["g_proj"]
    if not gp["pass"] or not gp["failing_input"]["fails"]:
        reasons.append("G-proj failed")
    reasons += thr["g_judge0"]["reasons"]
    if not gn["pass"]:
        reasons.append(f"underpowered ({len(full)} items with every primary answer, need {sub['g_n_arm0']})")
    rt = thr["g_retest"]
    sfx = lambda st: f" [{st}]" if st else ""  # noqa: E731
    yn_st, fc3_st = rt.get("yn_true", {}).get("stamp"), rt.get("fc3", {}).get("stamp")
    pl = thr["plant"]                                                    # A25: the Bernoulli plant's records
    rm_w = room("H_write", thr["null_values"]["H_write"], len(one), 0.15, thr["delta_w"], outer, inner, pl["H_write"])
    rm_d = room("H_depth", (thr["null_values"]["H_depth_t2"], thr["null_values"]["H_depth_t0"]), (len(terc[2]), len(terc[0])),
                0.20, 0.0, outer, inner, (pl["H_depth"][0], pl["H_depth"][1]))
    rm_o = room("H_ONE", thr["null_values"]["H_ONE"], len(pairs), 0.15, 0.0, outer, inner, pl["H_ONE"])
    rm_b = room("B1", thr["null_values"]["B1"], len(b1_vals), B1_EFFECT, 0.0, outer, inner, pl["B1"])
    lw = apply_room(label_write(D, Df, thr["delta_w"]), {"H_WRITE"}, rm_w) + sfx(yn_st)
    ld = apply_room(label_dir(depth, eq=EQUIV["H_depth"]), {"SUPPORTED"}, rm_d) + sfx(yn_st)
    hone = ci(pv)
    lo_ = apply_room(label_dir(hone, eq=EQUIV["H_ONE"]), {"SUPPORTED"}, rm_o) + sfx(yn_st)
    fc3_run = any("fc3" in ts for ts in sub["tasks"].values())
    letters = [full[p]["choice"] for p in full if full[p]["choice"] is not None]
    shares = {c: letters.count(c) / len(letters) for c in "ABC"} if letters else {}
    fc3 = {cls: ci([full[p]["c3"] for p in full if roles["poles"][p]["class"] == cls and full[p]["c3"] is not None])
           for cls in ("ONE", "UNK")}
    row = lambda s: {**stamped("ROW" + relation(s["lo"], s["hi"], 0.0, "0"), reasons), "D_w": s}  # noqa: E731
    top = {p: [x["str"] for x in projections[p]["top"]] for p in projections}
    labs = {"reexplained": {p: r["explanation"] for p, r in reexplain.items()},
            "with_vocab": {p: r["explanation"] for p, r in reexplain_vocab.items()},
            "original": {p: roles["poles"][p]["label"] for p in roles["poles"]}}
    val = {"reexplained": {p: full[p]["s"] for p in full}, **yt}   # D_w on re-explained labels; yn_true rate on the others
    lit = {ls: {p: literal_overlap(lab[p], top[p]) for p in one if p in lab} for ls, lab in labs.items()}
    rx_one = [p for p in one if p in reexplain and reexplain[p]["class"] == "ONE"]
    nsn = ci([v for x, v in zip(pairs, pv) if x["one"][:-1] != x["ctrl"][:-1]])
    rowa = lambda s: {**stamped("RATE" + relation(s["lo"], s["hi"], 0.0, "0"), reasons), "A": s}  # noqa: E731
    sec = {**{f"{ls}_label_set": {"A": rowa(ci([yt[ls][p] for p in one if p in yt[ls]]))} for ls in ("with_vocab", "original")},
           "literal_overlap": {ls: {"readout": "D_w" if ls == "reexplained" else "A (yn_true rate; A22 runs no shuffles here)",
                                    "literal": (row if ls == "reexplained" else rowa)(ci([val[ls][p] for p in one
                                                                                         if lit[ls].get(p) and p in val[ls]])),
                                    "non_literal": (row if ls == "reexplained" else rowa)(ci([val[ls][p] for p in one if p in lit[ls]
                                                                                             and not lit[ls][p] and p in val[ls]])),
                                    "n_literal": sum(lit[ls].values())} for ls in lit},
           "reexplained_as_ONE": row(ci(S(rx_one))),
           "by_sign": {str(sg): row(ci(S([p for p in one if roles["poles"][p]["sign"] == sg]))) for sg in (1, -1)},
           "by_explainer": {m: row(ci(S([p for p in one if roles["poles"][p]["explainer_model"] == m])))
                            for m in sorted({roles["poles"][p]["explainer_model"] for p in one})},
           "by_tercile": {str(t): {**row(ci(S(ps))), "A": ci([full[p]["y_true"] for p in ps]),
                                   "Sh": ci([(full[p]["y_s1"] + full[p]["y_s2"]) / 2 for p in ps]),
                                   "Fx": ci([full[p]["y_fixed"] for p in ps])} for t, ps in terc.items()},
           "A_one_vs_unk": {"ONE": A, "UNK": ci([full[p]["y_true"] for p in roles["unk"] if p in full])},
           "H_ONE_without_same_neuron": {**stamped(label_dir(nsn, eq=EQUIV["H_ONE"]), reasons), "d": nsn}}
    rows = {"H_write": {**stamped(lw, reasons), "D_w": D, "D_f": Df, "delta_w": thr["delta_w"], "room": rm_w},
            "H_depth": {**stamped(ld, reasons), "diff_t2_minus_t0": depth, "room": rm_d},
            "H_ONE_model": {**stamped(lo_, reasons), "d": hone, "n_pairs": len(pairs), "room": rm_o},
            "B1_model": {**stamped(apply_room(label_dir(b1, eq=EQUIV["B1"]), {"SUPPORTED"}, rm_b), reasons), "L": b1,
                         "n": len(b1_vals), "room": rm_b},
            "fc3": ({**stamped("ACCURACY" + relation(fc3["ONE"]["lo"], fc3["ONE"]["hi"], 1 / 3, "chance 1/3") + sfx(fc3_st), reasons),
                     "by_class": fc3, "letter_shares": shares,
                     "B3_letters_ok": bool(shares) and all(abs(v - 1 / 3) <= LETTER_TOL for v in shares.values())} if fc3_run else
                    {**stamped("NOT MEASURED (A22: Arm 0 fc3 is not run)", reasons), "by_class": None, "letter_shares": None,
                     "B3_letters_ok": None})}
    stamp_labels(rows, g_proj_frozen_stamp(gp))             # A28 (3), review A7: every Arm 0 label, primary and secondary
    stamp_labels(sec, g_proj_frozen_stamp(gp))
    return {"model_key": model_key, "arm": 0, "model_reasons": reasons, "n_items": len(full), "n_one": len(one), "A": A, "Sh": Sh,
            "Fx": Fx, "rows": rows, "secondary": sec, "g_proj": gp, "g_n": gn, "g_retest": rt, "g_judge0": thr["g_judge0"],
            "pair_values_H_ONE": pv, "null_values_H_ONE": thr["null_values"]["H_ONE"],
            "pair_values_B1": b1_vals, "null_values_B1": thr["null_values"]["B1"], "plant_H_ONE": pl["H_ONE"], "plant_B1": pl["B1"]}


# ------------------------------------------------------------------------------------------ self-repair (§7.2, A6)
def inf_ci(x) -> dict:
    """Median with a percentile bootstrap interval where values may be +inf (an infinite median makes that end inf)."""
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {"n": 0, "est": None, "lo": None, "hi": None}
    big = 1e300
    xs = np.where(np.isinf(x), big, x)
    m = np.median(xs[np.random.default_rng(SEED).integers(0, len(x), (N_BOOT, len(x)))], axis=1)
    lo, hi = _pct(m)
    back = lambda v: math.inf if v >= big / 10 else float(v)  # noqa: E731
    return {"n": len(x), "est": back(np.median(xs)), "lo": back(lo), "hi": back(hi), "n_inf": int(np.isinf(x).sum())}


def repair_compute(roles: dict, recs: dict, metrics: dict, direct: str = "direct_dnll", total: str = "tonly_dnll") -> dict:
    """§7.2 under the amended G-repair-fit (A23); the same computation under the frozen fp32 reading is printed beside
    every H_repair label ("frozen_reading")."""
    fit = {p: repair_fit(r) for p, r in recs.items()}
    out = repair_core(roles, recs, metrics, fit, True, direct, total)
    fr = repair_core(roles, recs, metrics, fit, False, direct, total)
    n_fr = sum(not f["frozen"]["pass"] for f in fit.values())
    note = (f" [frozen G-repair-fit reading (fp32 manual vs model NLL): {n_fr} of {len(fit)} poles out, "
            f"{'PASS' if not any('G-repair-fit' in x for x in fr['model_reasons']) else 'FAIL'}; label under it: {fr['H_repair']['label']}]")
    out["H_repair"] = {k: (v + note if isinstance(v, str) else v) for k, v in out["H_repair"].items()}
    out["frozen_reading"] = {"n_out": n_fr, "model_reasons": fr["model_reasons"], "H_repair": fr["H_repair"], "M_r": fr["M_r"],
                             "n_fit": fr["n_fit"], "n_qualifying": fr["n_qualifying"]}
    return out


def repair_core(roles: dict, recs: dict, metrics: dict, fit: dict, amended: bool, direct: str, total: str) -> dict:
    good = [p for p in recs if (fit[p]["pass"] if amended else fit[p]["frozen"]["pass"])]
    out_share = 1 - len(good) / len(recs) if recs else 1.0
    odd = {p: ci([c[direct] for c in recs[p]["contexts"] if c["k"] % 2 == 1]) for p in good}
    qual = [p for p in good if num(odd[p]["lo"]) and odd[p]["lo"] > 0]
    T = {p: float(np.mean([c[total] for c in recs[p]["contexts"] if c["k"] % 2 == 0])) for p in qual}
    Dp = {p: float(np.mean([c[direct] for c in recs[p]["contexts"] if c["k"] % 2 == 0])) for p in qual}
    r = {p: (T[p] / Dp[p] if Dp[p] > 0 else math.inf) for p in qual}
    M = inf_ci(list(r.values()))
    reasons = model_gate_reasons(metrics)
    if out_share > REPAIR_FIT["out_share"]:
        reasons.append(f"G-repair-fit removes {out_share:.0%} of poles (more than 30%)")
    if not all(f["failing_input"]["fails"] for f in fit.values()):
        reasons.append("G-repair-fit failing input did not fail on every pole")
    if amended and not all(f["failing_input_amended"]["fails"] for f in fit.values()):
        reasons.append("G-repair-fit wrong-epsilon failing input did not exceed the limit on every pole")
    gn = {"n_qualifying": len(qual), "pass": len(qual) >= GN["repair_poles"],
          "failing_input": {"n": GN["repair_poles"] - 1, "fails": True}}
    if not gn["pass"]:
        reasons.append(f"underpowered ({len(qual)} qualifying poles)")
    cls = {c: {"T": ci([T[p] for p in qual if roles["poles"][p]["class"] == c], np.median),
               "D": ci([Dp[p] for p in qual if roles["poles"][p]["class"] == c], np.median)} for c in ("ONE", "UNK")}
    ctx = [c for p in good if recs[p]["layer"] in (10, 11) for c in recs[p]["contexts"]]
    return {"model_key": roles["model_key"], "direct_key": direct, "total_key": total, "n_measured": len(recs), "n_fit": len(good),
            "n_qualifying": len(qual), "n_direct_not_replicated": sum(Dp[p] <= 0 for p in qual), "ratios": r, "T": T, "D": Dp,
            "M_r": M, "model_reasons": reasons, "H_repair": stamped(label_repair(M), reasons), "by_class": cls, "g_n": gn,
            "fit": fit, "F42_share_pos1_layers_10_11": (sum(c["tok"] % 2048 == 1 for c in ctx) / len(ctx)) if ctx else None}


def repair_pooled(per_model: dict, roles: dict) -> dict:
    counted = [m for m, r in per_model.items() if not r["model_reasons"]]
    M = pooled([[min(v, 1e300) for v in per_model[m]["ratios"].values()] for m in counted], np.median)
    for k in ("est", "lo", "hi"):
        M[k] = math.inf if M[k] is not None and M[k] >= 1e299 else M[k]
    pdiff, dropped = {}, 0
    for m, r in per_model.items():
        pdiff[m] = []
        for x in roles[m]["pairs"]:
            if x["one"] in r["ratios"] and x["ctrl"] in r["ratios"]:
                a, b = r["ratios"][x["one"]], r["ratios"][x["ctrl"]]
                if math.isinf(a) or math.isinf(b):
                    dropped += 1
                else:
                    pdiff[m].append(a - b)
    pp = pooled([pdiff[m] for m in counted], np.median)
    gn = {"n_pairs": pp["n"], "pass": pp["n"] >= GN["repair_pairs"], "failing_input": {"n": GN["repair_pairs"] - 1, "fails": True}}
    reasons = [] if len(counted) == len(per_model) else [f"models cancelled: {sorted(set(per_model) - set(counted))}"]
    return {"models": sorted(per_model), "counted": counted, "M_r": M, "H_repair": stamped(label_repair(M), reasons),
            "H_repair_ONE": stamped(label_dir(pp, eq=(-0.20, 0.20), flip=True), reasons + ([] if gn["pass"] else [f"underpowered ({pp['n']} pairs)"])),
            "paired_ratio_diff": pp, "pairs_dropped_infinite": dropped,
            "per_model_paired": {m: ci(v, np.median) for m, v in pdiff.items()}, "g_n": gn}


# ------------------------------------------------------------------------------------------ Arm 1 (§5.4, §7.3, A2, A3)
def construct_count(pattern: str, prompt: str, completion: str) -> int:
    """NEURONOFF §5.4: matches on prompt[-200:] + completion that end inside the completion."""
    pre = prompt[-200:]
    return sum(1 for m in re.finditer(pattern, pre + completion) if m.end() > len(pre))


def arm1_rows(samples: list, target: str, ctrl: str) -> dict:
    out = {target: {}, ctrl: {}}
    for r in samples:
        for p in ((target, ctrl) if r["pole"] is None else (r["pole"],) if r["pole"] in out else ()):
            out[p].setdefault(r["alpha"], []).append(r)
    return out


def d_star(rows: dict, band: dict, doses: list):
    C = {p: {a: float(np.mean([not is_coherent(r, band) for r in v])) for a, v in d.items()} for p, d in rows.items()}
    ok = [d for d in doses if d > 0 and all(C[p].get(d, 1.0) <= 0.5 and C[p].get(-d, 1.0) <= 0.5 for p in rows)]
    return (max(ok) if ok else None), C


def cell_realized(rows: list):
    v = [r["realized_cos"] for r in rows if r.get("realized_cos") is not None]
    return float(np.median(v)) if v else None


def per_prompt(rows_by_alpha: dict, alpha: float, band: dict, value, prompt_ids: list) -> tuple:
    """(num, den) per prompt over the coherent rows at alpha: num = sum of value(row), den = coherent count."""
    num_, den = np.zeros(len(prompt_ids)), np.zeros(len(prompt_ids))
    pos = {p: i for i, p in enumerate(prompt_ids)}
    for r in rows_by_alpha.get(alpha, []):
        if is_coherent(r, band):
            num_[pos[r["prompt_i"]]] += value(r)
            den[pos[r["prompt_i"]]] += 1
    return num_, den


def value_fn(readout: str, pattern: str, prompts: dict):
    def v(r):
        n = construct_count(pattern, prompts[r["prompt_i"]], r["completion"])
        return float(n) if readout == "count" else float(n > 0)
    return v


def swing_counts(rows: dict, t: str, c: str, d: float, band: dict, value, ids: list) -> dict:
    return {f"{p}{s}": per_prompt(rows[p], s * d, band, value, ids) for p in (t, c) for s in (1.0, -1.0)}


def swing_stat(t: str, c: str):
    return lambda R: (R[f"{t}1.0"] - R[f"{t}-1.0"]) - (R[f"{c}1.0"] - R[f"{c}-1.0"])


def arm1_room(ctrl_counts_vals: dict, readout: str, n_prompts=None, outer: int = 2_000, inner: int = 1_000) -> dict:
    """§7.3 room (A3 in the module docstring): R-pass (planted copy reads SWINGS against the control) and R-equiv
    (unplanted copy reads NO SWING); ctrl_counts_vals {+1/-1: [per-prompt arrays of the control's coherent values]}."""
    vals = {s: v[:n_prompts] if n_prompts else v for s, v in ctrl_counts_vals.items()}
    n = len(vals[1.0])
    thr, ns, plant = SWING[readout], NO_SWING[readout], ROOM1_PLANT[readout]
    plant_down = sum(map(sum, vals[-1.0])) / max(1, sum(map(len, vals[-1.0]))) >= 0.5 if readout == "presence" else False
    rng = np.random.default_rng(SEED)
    hit_pass = hit_eq = 0
    for _ in range(outer):
        idx = rng.integers(0, n, n)
        cn, cd, kn, kd = {}, {}, {}, {}
        for s in (1.0, -1.0):
            cn[s] = np.array([float(np.sum(vals[s][i])) for i in idx])
            cd[s] = np.array([float(len(vals[s][i])) for i in idx])
            kn[s] = np.array([float(np.sum(np.asarray(vals[s][i])[rng.integers(0, len(vals[s][i]), len(vals[s][i]))]))
                              if len(vals[s][i]) else 0.0 for i in idx])
            kd[s] = cd[s].copy()
        for planted in (True, False):
            pn = {s: kn[s].copy() for s in kn}
            if planted:
                s = -1.0 if plant_down else 1.0
                pn[s] = pn[s] + (-1 if plant_down else 1) * plant * kd[s]
                if readout == "presence":
                    pn[s] = np.clip(pn[s], 0, kd[s])
            counts = {"t1.0": (pn[1.0], kd[1.0]), "t-1.0": (pn[-1.0], kd[-1.0]), "c1.0": (cn[1.0], cd[1.0]), "c-1.0": (cn[-1.0], cd[-1.0])}
            with np.errstate(divide="ignore", invalid="ignore"):
                S_t = pn[1.0].sum() / kd[1.0].sum() - pn[-1.0].sum() / kd[-1.0].sum()
            d = paired_ratio_boot(counts, swing_stat("t", "c"), n_boot=inner)
            if not ok_interval(d):
                continue
            if planted:
                hit_pass += abs(S_t) >= thr and (d["lo"] > 0 if S_t > 0 else d["hi"] < 0)
            else:
                hit_eq += d["lo"] <= 0 <= d["hi"] and -ns < d["lo"] and d["hi"] < ns
    return {"n_prompts": n, "planted": "decrease at -d*" if plant_down else "increase at +d*", "r_pass": hit_pass / outer,
            "r_equiv": hit_eq / outer}


def freeze1_compute(model_key: str, roles: dict, plan: dict, samples_by_set: dict, cal_spans: list, inputs: list, replacement,
                    room_outer: int = 2_000, room_inner: int = 1_000) -> dict:
    """Control data only (SPEC §7.5): alpha = 0 rows, natural spans, control-pole rows, both poles' coherence flags and
    realized_cos (target rows arrive without completions)."""
    alpha0 = [r for s in plan["prompt_sets"] for r in samples_by_set[s] if r["pole"] is None]
    gc = coherence_from(alpha0, cal_spans, inputs, model_key)
    band = gc["band"]
    pairs = {}
    for e in [x for x in roles["arm1"] if x["status"] == "kept" or x["target"] == replacement]:
        rows = arm1_rows(samples_by_set[e["prompt_set"]], e["target"], e["ctrl"])
        prompts = {p["i"]: p["prompt"] for p in plan["prompt_sets"][e["prompt_set"]]}
        ids = sorted(prompts)
        ds, C = d_star(rows, band, [d for d in plan["doses"] if d > 0])
        out = {"target": e["target"], "ctrl": e["ctrl"], "readout": e["readout"], "d_star": ds,
               "collapse": {p: {str(a): c for a, c in sorted(v.items())} for p, v in C.items()},
               "realized_cos": {p: {str(a): cell_realized(v) for a, v in sorted(rows[p].items())} for p in rows}, "room": None}
        if ds is not None:
            val = value_fn(e["readout"], e["regex"], prompts)
            cv = {s: [[val(r) for r in rows[e["ctrl"]].get(s * ds, []) if r["prompt_i"] == i and is_coherent(r, band)] for i in ids]
                  for s in (1.0, -1.0)}
            full = arm1_room(cv, e["readout"], None, room_outer, room_inner)
            two = arm1_room(cv, e["readout"], 2, room_outer, room_inner)
            out["room"] = {**full, "failing_input_2_prompts": {**two, "fails": two["r_pass"] < ROOM_PASS_MIN},
                           "stamps": (["underpowered for a swing"] if full["r_pass"] < ROOM_PASS_MIN else []) +
                           (["NO SWING cannot be read"] if full["r_equiv"] < ROOM_PASS_MIN else []) +
                           ([] if two["r_pass"] < ROOM_PASS_MIN else ["room check cannot fail"])}
        pairs[e["target"]] = out
    return {"model_key": model_key, "arm": 1, "band": band, "g_coh": gc, "replacement": replacement, "pairs": pairs}


def unblind1_compute(model_key: str, roles: dict, plan: dict, thr: dict, samples_by_set: dict, metrics: dict, firing: list) -> dict:
    band, gc = thr["band"], thr["g_coh"]
    instrument = model_gate_reasons(metrics, {"place", "gen", "pc"})
    model_reasons = [r for r in model_gate_reasons(metrics) if r not in instrument]
    if not gc["pass"]:
        model_reasons.append("coherence gate not shown to work")
    k = firing_counts(firing)
    out = {}
    for t, fz in thr["pairs"].items():
        e = next(x for x in roles["arm1"] if x["target"] == t)
        c = e["ctrl"]
        rows = arm1_rows(samples_by_set[e["prompt_set"]], t, c)
        prompts = {p["i"]: p["prompt"] for p in plan["prompt_sets"][e["prompt_set"]]}
        ids = sorted(prompts)
        val = value_fn(e["readout"], e["regex"], prompts)
        rate = lambda rr: (sum(val(r) for r in rr if is_coherent(r, band)) / max(1, sum(is_coherent(r, band) for r in rr)))  # noqa: E731
        curves = {p: {str(a): {"R": rate(v), "C": float(np.mean([not is_coherent(r, band) for r in v])),
                               "coherent": sum(is_coherent(r, band) for r in v), "n": len(v), "realized_cos": cell_realized(v)}
                      for a, v in sorted(rows[p].items())} for p in rows}
        ds = fz["d_star"]
        info = {"firing_count": k.get(t), "d_star": ds, "control": c, "control_d": e["d"], "poor_control_match": e["d"] > POOR_MATCH,
                "curves": curves, "R0": curves[t].get("0.0", {}).get("R")}
        if not gc["pass"]:                        # P8 (A27): the coherent-only readouts cancel; all-sample curves, exploratory
            info["curves_all_samples"] = {"stamp": "exploratory (coherence gate not shown to work; P8, A27)",
                                          "curves": {p: {str(a): {"R": float(np.mean([val(r) for r in v])) if v else None, "n": len(v)}
                                                         for a, v in sorted(rows[p].items())} for p in rows}}
        why, S, dd = list(instrument), None, None
        if ds is None:
            why.append("no coherent range")
        else:
            low = sorted({p for p in rows for a in (ds, -ds) if sum(is_coherent(r, band) for r in rows[p].get(a, [])) < ARM1_MIN_COHERENT})
            if low:
                why.append(f"fewer than {ARM1_MIN_COHERENT} coherent samples at +-d* ({low})")
            weak = sorted({f"{p} {a:+g}" for p in rows for a in (ds, -ds)
                           if cell_realized(rows[p].get(a, [])) is None or cell_realized(rows[p][a]) < REALIZED_MIN})
            if weak:
                why.append(f"median realized_cos below {REALIZED_MIN} ({weak})")
            cnt = swing_counts(rows, t, c, ds, band, val, ids)
            dd = paired_ratio_boot(cnt, swing_stat(t, c))
            S = {p: float(cnt[f"{p}1.0"][0].sum() / max(1, cnt[f"{p}1.0"][1].sum()) - cnt[f"{p}-1.0"][0].sum() / max(1, cnt[f"{p}-1.0"][1].sum()))
                 for p in (t, c)}
        if why:
            lab = "INSTRUMENT NOT SHOWN TO FIRE (" + "; ".join(why) + ")"
        else:
            lab = label_swing(S[t], dd, e["readout"])
            rm = fz["room"]
            lab += f" [R-pass {rm['r_pass']:.2f}, R-equiv {rm['r_equiv']:.2f}" + (f"; {'; '.join(rm['stamps'])}" if rm["stamps"] else "") + "]"
            if head(lab) == "NO SWING" and rm["r_equiv"] < ROOM_PASS_MIN:
                lab = "NO SWING NOT READABLE" + lab[len("NO SWING"):]
            if all(curves[p].get(str(a), {}).get("R", 0) >= CEILING for p in (t, c) for a in (ds, -ds)):
                lab += " [at ceiling: only a decrease could be seen]"
        if info["poor_control_match"]:
            lab += " [poor control match]"
        out[t] = {**stamped(lab, model_reasons), "S": S, "paired_S_t_minus_S_c": dd, **info}
    return {"model_key": model_key, "arm": 1, "model_reasons": model_reasons, "g_coh": gc, "targets": out}


# ------------------------------------------------------------------------------------------ Arm 2 (§5.7, §7.4, A5, A10)
def o_cells(rows: list) -> dict:
    """{pole: {alpha: {prompt_i: 0/1}}} from fc3_steer and fc3_alpha0 answers (preflight never read)."""
    acc = {}
    for r in rows:
        if r["task"] in ("fc3_steer", "fc3_alpha0") and r["answer"] is not None and not r["preflight"]:
            acc.setdefault(r["pole"], {}).setdefault(float(r["alpha"]), {})[r["prompt_i"]] = int(bool(r["correct"]))
    return acc


def realized_cells(samples: list) -> dict:
    cells = {}
    for r in samples:
        if r["condition"] == "steer" and r["pole"] is not None:
            cells.setdefault((r["pole"], float(r["alpha"])), []).append(r)
    return {k: cell_realized(v) for k, v in cells.items()}


def freeze2_compute(model_key: str, roles: dict, ctrl_rows: list, retest_rows: list, alpha0: list, cal_spans: list, inputs: list,
                    label_rows: list, outer=ROOM_OUTER, inner=ROOM_INNER) -> dict:
    """Control data only (SPEC §7.5): alpha = 0 samples and answers, natural spans, label-prompted samples and answers."""
    ret = {x["retest_of"] for x in retest_rows}
    assert all(r["task"] in ("fc3_alpha0", "fc3_label_prompt") or r["id"] in ret for r in ctrl_rows), \
        "freeze2 was handed a steered answer outside the retest pairs"
    gc = coherence_from(alpha0, cal_spans, inputs, model_key, label_rows)
    cells = o_cells(ctrl_rows)
    o0 = {p: float(np.mean(list(v[0.0].values()))) for p, v in cells.items() if 0.0 in v}
    lab = {}
    for r in ctrl_rows:
        if r["task"] == "fc3_label_prompt" and r["answer"] is not None and not r["preflight"]:
            lab.setdefault(r["pole"], []).append(int(bool(r["correct"])))
    rate = lambda v: float(np.mean(v)) if len(v) else float("nan")  # noqa: E731
    lab_rate, a0_rate = rate([np.mean(v) for v in lab.values()]), rate(list(o0.values()))
    fail_rate = rate([o0[p] for p in lab if p in o0])
    reasons = ([] if num(lab_rate) and lab_rate >= JUDGE_FLOOR else ["judge not shown to fire"]) + \
        ([] if num(a0_rate) and a0_rate <= JUDGE_CEIL else ["judge not discriminating"]) + \
        ([] if num(fail_rate) and fail_rate < JUDGE_FLOOR else ["G-judge2 failing input did not fail"])
    letters = [r["answer"] for r in ctrl_rows if r["task"] == "fc3_alpha0" and r["answer"] is not None and not r["preflight"]]
    shares = {c: letters.count(c) / len(letters) for c in "ABC"} if letters else {}
    ent = {e["pole"]: e for e in roles["arm2"]}
    one = [p for p in o0 if ent[p]["role"] == "ONE"]
    v0 = np.array([o0[p] for p in one])
    ag = [o0[p] for p in one if ent[p]["stratum"].startswith("agree")]
    dg = [o0[p] for p in one if ent[p]["stratum"].startswith("disagree")]
    t = {k: [o0[p] for p in one if roles["poles"][p]["tercile"] == k] for k in (0, 2)}
    rooms = {"H_suff": room("H_suff", (v0 - v0.mean() + 1 / 3).tolist() if len(v0) else [], len(v0), 0.15, 0.0, outer, inner),
             "H_pred_agree": room("H_pred_agree", (ag, dg), (len(ag), len(dg)), 2.5, 0.0, outer, inner),
             "H_pred_conf": room("H_pred_conf", ([o0[p] for p in one], [roles["poles"][p]["confidence"] for p in one]), len(one),
                                 0.3, 0.0, outer, inner),
             "H_depth": room("H_depth", (t[2], t[0]), (len(t[2]), len(t[0])), 0.20, 0.0, outer, inner)}
    return {"model_key": model_key, "arm": 2, "band": gc["band"], "g_coh": gc,
            "g_judge2": {"label_prompted_rate": lab_rate, "n_label_poles": len(lab), "floor": JUDGE_FLOOR, "alpha0_rate": a0_rate,
                         "ceiling": JUDGE_CEIL, "failing_input": {"alpha0_rate_same_poles": fail_rate,
                                                                   "fails": num(fail_rate) and fail_rate < JUDGE_FLOOR},
                         "pass": not reasons, "reasons": reasons},
            "letter_shares_alpha0": shares, "B3_letters_ok": bool(shares) and all(abs(v - 1 / 3) <= LETTER_TOL for v in shares.values()),
            "g_retest": retest_gate(ctrl_rows, retest_rows, model_key), "o_alpha0": o0, "room": rooms}


def unblind2_compute(model_key: str, roles: dict, thr: dict, fc3_rows: list, guess_rows: list, metrics: dict, d_p: float,
                     samples: list) -> dict:
    """§7.4. A27: when freeze2's G-coh failed, the judge saw all samples of each cell (no coherence filter); the labels
    are computed on those answers and stamped "coherence gate not shown to work", and the frozen coherent-only reading
    (cells whose shown samples all lie inside the band, the frozen rule's CANCELLED label with its numbers) is printed
    beside as "frozen_coherent_only"."""
    base = model_gate_reasons(metrics) + thr["g_judge2"]["reasons"]
    if thr["g_coh"]["pass"]:
        return unblind2_core(model_key, roles, thr, fc3_rows, guess_rows, d_p, samples, base)
    v = unblind2_core(model_key, roles, thr, fc3_rows, guess_rows, d_p, samples, base)
    stamp_labels(v["rows"], " [coherence gate not shown to work]")
    band, groups = thr["band"], {}
    for r in samples:
        if r["condition"] == "steer":
            groups.setdefault((r["pole"], float(r["alpha"]), r["prompt_i"]), []).append(is_coherent(r, band))
    full = {k for k, v_ in groups.items() if all(v_)}
    keep = [r for r in fc3_rows if r["task"] not in ("fc3_steer", "fc3_alpha0") or
            ((None if r["task"] == "fc3_alpha0" else r["pole"]), float(r["alpha"]), r["prompt_i"]) in full]
    fr = unblind2_core(model_key, roles, thr, keep, guess_rows, d_p, samples, base + ["coherence gate not shown to work"])
    v["coherence_filter"] = False
    v["frozen_coherent_only"] = {"rule": "A27: the frozen coherent-only reading on the cells whose shown samples all lie inside "
                                         "the band (the judge saw every sample)", "n_cells": len(full),
                                 "rows": fr["rows"], "g_n": fr["g_n"]}
    return v


def unblind2_core(model_key: str, roles: dict, thr: dict, fc3_rows: list, guess_rows: list, d_p: float, samples: list,
                  reasons: list) -> dict:
    reasons = list(reasons)
    cells = o_cells(fc3_rows)
    rc = realized_cells(samples)
    ent = {e["pole"]: e for e in roles["arm2"]}
    weak = sorted(f"{p} {a:+g}" for (p, a), v in rc.items() if v is None or v < REALIZED_MIN)
    usable = lambda p, a: a == 0.0 or (rc.get((p, a)) is not None and rc[(p, a)] >= REALIZED_MIN)  # noqa: E731
    one_drawn = [p for p in ent if ent[p]["role"] == "ONE"]
    one = [p for p in one_drawn if p in cells and d_p in cells[p] and usable(p, d_p)]
    need = math.ceil(GN["arm2_share"] * len(one_drawn))
    gn = {"n_contributing": len(one), "n_drawn": len(one_drawn), "need": need, "pass": len(one) >= need,
          "failing_input": {"n": need - 1, "fails": need - 1 < need}}
    conf = {p: roles["poles"][p]["confidence"] for p in one}

    def stat(md, m0, poles):
        ag = np.array([ent[p]["stratum"].startswith("agree") for p in poles])
        t2 = np.array([roles["poles"][p]["tercile"] == 2 for p in poles])
        t0 = np.array([roles["poles"][p]["tercile"] == 0 for p in poles])
        with np.errstate(invalid="ignore", divide="ignore"):
            pair = np.nanmean(md - m0) if np.isfinite(m0).any() else np.nan
            R = md[ag].mean() / md[~ag].mean() if ag.any() and (~ag).any() else np.nan
            dep = md[t2].mean() - md[t0].mean() if t2.any() and t0.any() else np.nan
        from infra.pg.neuron_rw_core import spearman
        return {"o": float(md.mean()), "paired": float(pair), "R": float(R), "diff_agree_disagree":
                float(md[ag].mean() - md[~ag].mean()) if ag.any() and (~ag).any() else float("nan"),
                "rho": spearman([conf[p] for p in poles], md), "depth": float(dep)}
    steer = {p: list(cells[p][d_p].values()) for p in one}
    zero = {p: cells[p].get(0.0, {}) for p in one}
    prompts0 = sorted({i for p in one for i in zero[p]})          # review C10: resample the judged alpha = 0 prompts only (A22)
    tw = two_way_boot(steer, zero, prompts0, stat) if one else {k: {"est": None, "lo": None, "hi": None}
                                                                                for k in ("o", "paired", "R", "rho", "depth", "diff_agree_disagree")}
    rm = thr["room"]
    st = thr["g_retest"].get("fc3_steer", {}).get("stamp")
    ls = apply_room(label_suff(tw["o"], tw["paired"]), {"SUPPORTED", "ABOVE CHANCE, NOT ABOVE ALPHA = 0"}, rm["H_suff"]) + (f" [{st}]" if st else "")
    if not gn["pass"]:
        ls = f"CANCELLED: NO COHERENT DOSE ({len(one)} of {len(one_drawn)} ONE poles contributing at d_p, need {need})" + \
            relation(tw["o"]["lo"], tw["o"]["hi"], 1 / 3, "chance 1/3")
    rows = {"H_suff": {**stamped(ls, reasons), "o_dp": tw["o"], "paired_minus_alpha0": tw["paired"]},
            "H_pred_agree": {**stamped(apply_room(label_agree(tw["R"]), {"SUPPORTED", "AGREE ABOVE DISAGREE, BELOW 2X"}, rm["H_pred_agree"]),
                                       reasons), "R": tw["R"], "difference": tw["diff_agree_disagree"]},
            "H_pred_conf": {**stamped(apply_room(label_conf(tw["rho"]), {"REFUTED (positive)", "REFUTED (negative)"}, rm["H_pred_conf"]),
                                      reasons), "spearman": tw["rho"]},
            "H_depth": {**stamped(apply_room(label_dir(tw["depth"], eq=EQUIV["H_depth"]), {"SUPPORTED"}, rm["H_depth"]), reasons),
                        "diff_t2_minus_t0": tw["depth"]}}
    pairs = [(e["matched"], p) for p, e in ent.items() if e["role"] == "UNK" and e["matched"] in cells and p in cells
             and d_p in cells[e["matched"]] and d_p in cells[p]]
    other = sorted({a for p in cells for a in cells[p]} - {d_p})
    mean_o = lambda p, a: float(np.mean(list(cells[p][a].values())))  # noqa: E731
    sec = {"other_doses": {str(a): ci([mean_o(p, a) for p in one_drawn if p in cells and a in cells[p] and usable(p, a)]) for a in other},
           "one_minus_matched_unk_dp": ci([mean_o(a, d_p) - mean_o(b, d_p) for a, b in pairs]),
           "by_explainer": {m: ci([mean_o(p, d_p) for p in one if roles["poles"][p]["explainer_model"] == m])
                            for m in sorted({roles["poles"][p]["explainer_model"] for p in one})},
           "cells_not_shown_to_fire": weak,
           "guesses": [{"pole": r["pole"], "prompt_i": r["prompt_i"], "guess": r["answer"], "label": roles["poles"][r["pole"]]["label"]}
                       for r in sorted(guess_rows, key=lambda r: (r["pole"], r["prompt_i"])) if r["answer"] is not None
                       and not r["preflight"]][:10]}
    return {"model_key": model_key, "arm": 2, "d_p": d_p, "model_reasons": reasons, "g_n": gn, "rows": rows, "secondary": sec,
            "g_judge2": thr["g_judge2"], "g_coh": thr["g_coh"]}


def prediction_rows(journal: str) -> dict:
    """{readout: {"predicted", "probability"}} of the table in the last JOURNAL entry headed "PREDICTIONS" (the one the
    grid is scored against)."""
    heads = [m.start() for m in re.finditer(r"^## .*PREDICTIONS.*$", journal, re.M)]
    assert heads, "no PREDICTIONS entry in the journal"
    seg = journal[heads[-1]:]
    nxt = re.search(r"^## ", seg[3:], re.M)
    seg = seg[:nxt.start() + 3] if nxt else seg
    rows = {}
    for line in seg.split("\n"):
        if line.startswith("| ") and not line.startswith(("| readout", "|---")):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            rows[c[0]] = {"predicted": c[1], "probability": c[2]}
    assert rows, "the last PREDICTIONS entry has no table"
    return rows


STATUSES = ("landed", "missed", "not scorable", "not readable")


def status(x) -> str:
    """True = missed, False = landed, None = not scorable; a status string passes through."""
    return x if isinstance(x, str) else "not scorable" if x is None else "missed" if x else "landed"


def missed_full(preds: dict, res: dict) -> list:
    """Review B2: one row per prediction of the scored PREDICTIONS entry (§11 table) beyond the §5.6 rows, each with a
    status in STATUSES under the frozen rules and under the amendments. `res` holds the loaded results: s0, s2 (study
    files), v0, v1, v2 (verdicts by model), t0, t2 (thresholds by model), roles, rx_class ({model: {pole: class}}),
    repair ({model: repair file}), repair_pooled, gates ([(cell, gates)]), rows56 (the §5.6 rows, already scored).
    Every table row must be mapped; the bounds a scorer uses must appear in the predicted text."""
    sh = lambda x: re.split(r" \(|,| \[", x)[0]  # noqa: E731      the head of a label or of a study verdict string

    def nr(label, supported):                     # a supported label the room check cannot read
        return "not readable" if sh(label).endswith("NOT READABLE") else sh(label) != supported
    s0, s2, v0, v2, t0, t2 = res["s0"], res["s2"], res["v0"], res["v2"], res["t0"], res["t2"]
    fr0, fr2, am2 = s0["frozen_rule"], s2["frozen_rule"], s2["amended_exploratory"]
    rate = {m: v["rows"]["H_suff"]["o_dp"]["est"] for m, v in v2.items()}
    a0r = {m: float(np.mean([v for p, v in t2[m]["o_alpha0"].items()
                             if {e["pole"]: e["role"] for e in res["roles"][m]["arm2"]}[p] == "ONE"])) for m in sorted(t2)}
    bad = [f"{c} {g}" for c, gs in res["gates"] for g in ("pc", "place", "gen", "known") if gs.get(g) is False]
    share = {m: (sum(res["rx_class"][m].get(p) == "ONE" for p in res["roles"][m]["one"]), len(res["roles"][m]["one"]))
             for m in sorted(res["rx_class"])}
    q2 = v2["Q4B"]
    rp, rpp = res["repair"], res["repair_pooled"]
    fz_q = lambda r: r.get("frozen_reading", {}).get("n_qualifying")  # noqa: E731
    covered56 = {"α_c (this study's guess)": "alpha_c >= 16 in every model", "α_c held out from Liu": "Liu",
                 "Arm 1 SWINGS count": "at most 2 of 7"}
    scorers = {
        "H_write (re-explained labels), study label": lambda: (
            ("SUPPORTED",), fr0["H_write"], s0["H_write"], nr(fr0["H_write"], "H_WRITE"), nr(s0["H_write"], "H_WRITE"),
            "results/study_arm0.json frozen_rule.H_write, H_write"),
        "B1 leak, pooled": lambda: (
            ("SUPPORTED", "0.02", "0.10"),
            f"{sh(fr0['B1_pooled']['label'])}, {fr0['B1_pooled']['counted']} alone, L {fmt(fr0['B1_pooled']['value'])}",
            f"{sh(s0['B1_pooled']['label'])}, L {fmt(s0['B1_pooled']['L'])}",
            sh(fr0["B1_pooled"]["label"]) != "SUPPORTED" or not 0.02 <= fr0["B1_pooled"]["value"]["est"] <= 0.10,
            sh(s0["B1_pooled"]["label"]) != "SUPPORTED" or not 0.02 <= s0["B1_pooled"]["L"]["est"] <= 0.10,
            "results/study_arm0.json frozen_rule.B1_pooled, B1_pooled"),
        "Arm 0 absolute rates": lambda: (
            ("0.20", "0.50", "0.05", "0.25"),) + (lambda o, m_: (o, o, m_, m_, "results/verdict_arm0_*.json A, Sh, Fx"))(
            "; ".join(f"{m} A {v['A']['est']:.3f}, Sh {v['Sh']['est']:.3f}, Fx {v['Fx']['est']:.3f}" for m, v in sorted(v0.items())),
            any(not 0.20 <= v["A"]["est"] <= 0.50 or not 0.05 <= v["Sh"]["est"] <= 0.25 or not 0.05 <= v["Fx"]["est"] <= 0.25
                for v in v0.values())),
        "H_depth (Arm 0), study label": lambda: (
            ("SUPPORTED",), fr0["H_depth"], s0["H_depth"], nr(fr0["H_depth"], "SUPPORTED"), nr(s0["H_depth"], "SUPPORTED"),
            "results/study_arm0.json"),
        "H_ONE, pooled": lambda: (
            ("NOT SHOWN",), f"{fr0['H_ONE_pooled']['label']} {fmt(fr0['H_ONE_pooled']['value'])}",
            f"{s0['H_ONE_pooled']['label']} {fmt(s0['H_ONE_pooled']['d'])}",
            sh(fr0["H_ONE_pooled"]["label"]) != "NOT SHOWN", sh(s0["H_ONE_pooled"]["label"]) != "NOT SHOWN",
            "results/study_arm0.json frozen_rule.H_ONE_pooled, H_ONE_pooled"),
        "fc3 accuracy": lambda: ((), "not measured (A22: Arm 0 fc3 is not run)", "not measured (A22)", None, None, "SPEC A22"),
        "re-explained class": lambda: (
            ("80%",),) + (lambda o, m_: (o, o, m_, m_, "results/reexplain_*.jsonl class; roles one"))(
            "; ".join(f"{m} {a}/{n}" for m, (a, n) in share.items()), any(a < 0.80 * n for a, n in share.values())),
        "H_repair, pooled": lambda: (
            (), rpp["H_repair"]["label"], rpp["H_repair"]["label"],
            None if rpp["H_repair"]["label"].startswith("CANCELLED") else sh(rpp["H_repair"]["label"]) != "REPAIR",
            None if rpp["H_repair"]["label"].startswith("CANCELLED") else sh(rpp["H_repair"]["label"]) != "REPAIR",
            "results/repair_pooled.json"),
        "H_repair_ONE, pooled": lambda: (
            ("NOT SHOWN",), rpp["H_repair_ONE"]["label"], rpp["H_repair_ONE"]["label"],
            None if rpp["H_repair_ONE"]["label"].startswith("CANCELLED") else sh(rpp["H_repair_ONE"]["label"]) != "NOT SHOWN",
            None if rpp["H_repair_ONE"]["label"].startswith("CANCELLED") else sh(rpp["H_repair_ONE"]["label"]) != "NOT SHOWN",
            "results/repair_pooled.json"),
        "self-repair qualifying poles": lambda: (
            ("fewer than half",),
            "; ".join(f"{m} {fz_q(r)}/{r['n_measured']} (frozen fp32 fit)" for m, r in sorted(rp.items())),
            "; ".join(f"{m} {r['n_qualifying']}/{r['n_measured']}" for m, r in sorted(rp.items())),
            any(fz_q(r) >= 0.5 * r["n_measured"] for r in rp.values()), any(r["n_qualifying"] >= 0.5 * r["n_measured"] for r in rp.values()),
            "results/repair_*.json n_qualifying, frozen_reading.n_qualifying"),
        "H_suff, study label": lambda: (
            ("NOT SHOWN", "0.33", "0.42"), f"{fr2['H_suff']}; two-model rule on Q4B: {fr2['H_suff_two_model_rule_on_Q4B']}",
            f"{am2['H_suff']}; o(d_p) " + ", ".join(f"{m} {r:.3f}" for m, r in sorted(rate.items())),
            None if sh(fr2["H_suff"]).startswith(("CANCELLED", "MIXED")) else sh(fr2["H_suff"]) != "NOT SHOWN",
            sh(am2["H_suff"]) != "NOT SHOWN" or any(not 0.33 <= r <= 0.42 for r in rate.values()),
            "results/study_arm2.json; results/verdict_arm2_*.json rows.H_suff.o_dp"),
        "parent D1 X2": lambda: (
            ("not SUPPORTED",), fr2["H_suff"], am2["H_suff"], sh(fr2["H_suff"]) == "SUPPORTED", sh(am2["H_suff"]) == "SUPPORTED",
            "results/study_arm2.json"),
        "H_pred_agree, study label": lambda: (
            ("NOT SHOWN",), "CANCELLED (P8) on every model" if not q2["g_coh"]["pass"] else q2["rows"]["H_pred_agree"]["label"],
            f"Q4B alone (A27): {q2['rows']['H_pred_agree']['label']}", None if not q2["g_coh"]["pass"] else nr(q2["rows"]["H_pred_agree"]["label"], "NOT SHOWN"),
            nr(q2["rows"]["H_pred_agree"]["label"], "NOT SHOWN"), "results/verdict_arm2_Q4B.json rows.H_pred_agree"),
        "H_pred_conf, study label": lambda: (
            ("NOT SHOWN",), "CANCELLED (P8) on every model" if not q2["g_coh"]["pass"] else q2["rows"]["H_pred_conf"]["label"],
            f"Q4B alone (A27): {q2['rows']['H_pred_conf']['label']}", None if not q2["g_coh"]["pass"] else nr(q2["rows"]["H_pred_conf"]["label"], "NOT SHOWN"),
            nr(q2["rows"]["H_pred_conf"]["label"], "NOT SHOWN"), "results/verdict_arm2_Q4B.json rows.H_pred_conf"),
        "H_depth (Arm 2), study label": lambda: (
            ("NOT SHOWN",), fr2["H_depth"], am2["H_depth"],
            None if sh(fr2["H_depth"]).startswith(("CANCELLED", "MIXED")) else nr(fr2["H_depth"], "NOT SHOWN"),
            nr(am2["H_depth"], "NOT SHOWN"), "results/study_arm2.json frozen_rule.H_depth, amended_exploratory.H_depth"),
        "G-pc, G-place, G-gen, G-known pass in every job": lambda: (
            ("yes",), "every job passed" if not bad else "failed: " + ", ".join(bad),
            "every job passed" if not bad else "failed: " + ", ".join(bad), bool(bad), bool(bad), "bank/rw/*/*/out/metrics.json gates"),
        "G-judge0 and G-judge2 pass": lambda: (
            ("yes",),) + (lambda o, m_: (o, o, m_, m_, "results/thresholds_arm0_*.json g_judge0, thresholds_arm2_*.json g_judge2"))(
            "; ".join([f"G-judge0 {m} {t['g_judge0']['pass']}" for m, t in sorted(t0.items())] +
                      [f"G-judge2 {m} {t['g_judge2']['pass']}" for m, t in sorted(t2.items())]),
            not all(t["g_judge0"]["pass"] for t in t0.values()) or not all(t["g_judge2"]["pass"] for t in t2.values())),
        "α = 0 Arm 2 on-label rate": lambda: (
            ("0.25", "0.42"),) + (lambda o, m_: (o, o, m_, m_, "results/thresholds_arm2_*.json o_alpha0 (ONE poles)"))(
            "; ".join(f"{m} {r:.3f}" + (" (exploratory, A27)" if m != "Q4B" else "") for m, r in a0r.items()),
            any(not 0.25 <= r <= 0.42 for r in a0r.values()))}
    out = []
    for k, v in preds.items():
        hit = [x for x in covered56 if k.startswith(x)]
        if hit:
            assert any(covered56[hit[0]] in r["pred_id"] + r["prediction"] for r in res["rows56"]), f"{k}: no §5.6 row carries it"
            continue
        key = [x for x in scorers if k.startswith(x)]
        assert len(key) == 1, f"prediction {k!r} has no scorer (every row of the PREDICTIONS table must be scored)"
        bounds, fo, ao, fm, am, art = scorers[key[0]]()
        assert all(b in v["predicted"] for b in bounds), f"{k}: bounds {bounds} not in {v['predicted']!r}"
        out.append({"pred_id": k, "prediction": f"SPEC §11 {k}", "predicted": v["predicted"], "observed_frozen": fo,
                    "observed_amended": ao, "status_frozen": status(fm), "status_amended": status(am), "artifact": art})
    return out


def tally(rows: list, key: str) -> dict:
    """Totals over predictions (rows grouped by pred_id): missed if any row missed, else not readable, else not scorable,
    else landed."""
    by = {}
    for r in rows:
        by.setdefault(r["pred_id"], []).append(r[key])
    pick = lambda v: next(s for s in ("missed", "not readable", "not scorable", "landed") if s in v)  # noqa: E731
    out = {s: 0 for s in STATUSES}
    for v in by.values():
        out[pick(v)] += 1
    return {**out, "predictions": len(by)}


# ------------------------------------------------------------------------------------------ literal split (review 2026-10-26)
def distinct_bigram_ratio(text: str) -> float:
    """The review's repetition metric: distinct bigrams over bigrams of the word and punctuation tokens of a completion
    (regex \\w+|[^\\w\\s]); fewer than 3 tokens read 0.0. A sample with ratio < REPEAT_CUT counts as repetitive."""
    w = re.findall(r"\w+|[^\w\s]", text)
    if len(w) < 3:
        return 0.0
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg)


REPEAT_CUT = 0.5
P_LIT_MODEL = "O7"            # A28 (2): P-lit is registered on O7's Arm 2; on Q4B and G4 it is in sample (review C3)


def literal_split_compute(model_key: str, roles: dict, sub: dict, yn_rows: list, projections: dict, reexplain: dict,
                          fc3_rows: list | None = None, d_p: float | None = None, samples: list | None = None,
                          band: dict | None = None) -> dict:
    """Exploratory split (review 2026-10-26 §6): an item or pole is literal when its re-explained label names one of the
    pole's own top-20 promoted output tokens (the attested top-20 the Arm 0 judge saw; §5.1 literal_overlap). Arm 0:
    D_w per group over the ONE items with every primary answer (roles order). Arm 2: per ONE pole with answers at d_p,
    the paired gain mean o(d_p) - mean o(0) over its judged prompts, with an interval over poles (roles.arm2 order);
    plus, per dose, the share of ONE cells and of samples whose distinct-bigram ratio is below 0.5, beside the share the
    band marks collapsed."""
    lit = lambda p: literal_overlap(reexplain[p]["explanation"], [x["str"] for x in projections[p]["top"]])  # noqa: E731
    it = item_scores(roles, yn_rows, [], "reexplained")
    need = {c: [{"yn_fixed": "y_fixed"}[t] for t in ts if t == "yn_fixed"] for c, ts in sub["tasks"].items()}
    full = {p: v for p, v in it.items() if all(v[k] is not None for k in need[roles["poles"][p]["class"]])}
    one = [p for p in roles["one"] if p in full and p in reexplain]
    grp = {g: [p for p in one if lit(p) == (g == "literal")] for g in ("literal", "non_literal")}
    out = {"model_key": model_key, "rule": "literal: the re-explained label names one of the pole's own top-20 promoted output "
                                           "tokens (literal_overlap, SPEC §5.1); exploratory, made after the review",
           "arm0": {g: {"n": len(ps), "D_w": ci([full[p]["s"] for p in ps]), "A": ci([full[p]["y_true"] for p in ps])}
                    for g, ps in grp.items()}, "arm2": None, "repetition": None}
    if fc3_rows is not None:
        cells = o_cells(fc3_rows)
        ent = [e for e in roles["arm2"] if e["role"] == "ONE"]
        poles = [e["pole"] for e in ent if e["pole"] in cells and d_p in cells[e["pole"]] and e["pole"] in reexplain]
        gain = {p: float(np.mean(list(cells[p][d_p].values()))) - float(np.mean(list(cells[p][0.0].values()))) for p in poles}
        prompts0 = sorted({i for p in poles for i in cells[p].get(0.0, {})})

        def two_way(ps):                          # §5.7 two-way bootstrap of the paired difference (A28 (2), P-lit)
            if not ps:
                return {"est": None, "lo": None, "hi": None}
            with np.errstate(invalid="ignore"):
                return two_way_boot({p: list(cells[p][d_p].values()) for p in ps}, {p: cells[p].get(0.0, {}) for p in ps}, prompts0,
                                    lambda md, m0, q: {"paired": float(np.nanmean(md - m0))})["paired"]
        out["arm2"] = {"d_p": d_p, **{g: {"n": len(ps), "poles": ps, "paired_gain": ci([gain[p] for p in ps]),
                                          "paired_gain_two_way": two_way(ps),
                                          "o_dp": float(np.mean([np.mean(list(cells[p][d_p].values())) for p in ps])) if ps else None,
                                          "o_0": float(np.mean([np.mean(list(cells[p][0.0].values())) for p in ps])) if ps else None}
                                       for g, ps in {g: [p for p in poles if lit(p) == (g == "literal")]
                                                     for g in ("literal", "non_literal")}.items()}}
        lt, nl = out["arm2"]["literal"]["paired_gain_two_way"], out["arm2"]["non_literal"]["paired_gain_two_way"]
        ok = lt["est"] is not None and nl["lo"] is not None
        out["arm2"]["p_lit"] = {
            "rule": "A28 (2), registered on O7: literal two-way paired gain point >= 0.30 and the non-literal interval includes 0.10 "
                    "or lies below it; refuted if the non-literal interval lies entirely above 0.15",
            "reading": ("in sample (P-lit was built from these data), not a test" if model_key != P_LIT_MODEL else None if not ok else
                        "P-lit refuted" if nl["lo"] > 0.15 else
                        "P-lit confirmed" if lt["est"] >= 0.30 and nl["lo"] <= 0.10 else "P-lit not confirmed")}
    if samples is not None:
        role = {e["pole"]: e["role"] for e in roles["arm2"]}
        allp = {}
        for r in samples:
            if r["condition"] == "steer":
                allp.setdefault(float(r["alpha"]), []).append(not is_coherent(r, band))
        by = {}
        for r in samples:
            if r["condition"] == "steer" and (r["pole"] is None or role.get(r["pole"]) == "ONE"):
                by.setdefault(float(r["alpha"]), {}).setdefault((r["pole"], r["prompt_i"]), []).append(r)
        out["repetition"] = {"rule": f"distinct-bigram ratio < {REPEAT_CUT} (the review's metric, chosen after seeing the data)",
                             "by_dose": {str(a): {"n_cells": len(cs), "n_samples": sum(map(len, cs.values())),
                                                  "share_cells_repetitive": float(np.mean([np.mean([distinct_bigram_ratio(r["completion"])
                                                                                                    for r in v]) < REPEAT_CUT for v in cs.values()])),
                                                  "share_samples_repetitive": float(np.mean([distinct_bigram_ratio(r["completion"]) < REPEAT_CUT
                                                                                             for v in cs.values() for r in v])),
                                                  "band_collapse_share": float(np.mean([not is_coherent(r, band) for v in cs.values() for r in v])),
                                                  "band_collapse_share_all_poles": float(np.mean(allp[a]))}
                                         for a, cs in sorted(by.items())}}
    return out


# ------------------------------------------------------------------------------------------ missed predictions (A10)
NOT_SCORABLE_RULE = ("A prediction whose readout a gate cancels under a rule set (a CANCELLED label, or a number read from a "
                     "cancelled cell, such as a cancelled pair's S) is not scorable under that rule set (review N1).")


def missed_compute(verdicts1: dict, dose_scales: dict) -> list:
    """§5.6 rows; NOT_SCORABLE_RULE: a CANCELLED Arm 1 target makes its target and control rows not scorable, and a
    count over cancelled labels (at most 2 of 7 SWING) is not scorable."""
    rows = []
    why = "readout cancelled by a gate"
    for m, v in verdicts1.items():
        for t, r in v["targets"].items():
            pred = PREDICTIONS_5_6.get(t, ("NO SWING", "< 0.5", None))
            S = r["S"][t] if r["S"] else None
            gone = r["label"].startswith("CANCELLED")
            rows.append({"prediction": f"SPEC §5.6 {m} {t}", "predicted": f"{pred[0]}, |S| {pred[1]}, sign {pred[2]}",
                         "observed": f"{head(r['label'])}, S_t {S}", "artifact": f"results/verdict_arm1_{m}.json targets.{t}.label",
                         "missed": None if gone else head(r["label"]) != pred[0], **({"why": why} if gone else {})})
            cs = r["S"][r["control"]] if r["S"] else None
            rows.append({"prediction": f"SPEC §5.6 {m} {t} control", "predicted": "|S_c| < 0.10", "observed": f"S_c {cs}",
                         "artifact": f"results/verdict_arm1_{m}.json targets.{t}.S",
                         "missed": None if gone else not (cs is not None and abs(cs) < 0.10), **({"why": why} if gone else {})})
    labs = [r["label"] for v in verdicts1.values() for r in v["targets"].values()]
    n_sw = sum(head(x) == "SWINGS" for x in labs)
    gone = any(x.startswith("CANCELLED") for x in labs)
    rows.append({"prediction": "SPEC §5.6 summary: at most 2 of 7 targets SWING", "predicted": "<= 2", "observed": str(n_sw),
                 "artifact": "results/verdict_arm1_*.json", "missed": None if gone else n_sw > 2, **({"why": why} if gone else {})})
    for m, d in dose_scales.items():
        rows.append({"prediction": f"SPEC §5.6 summary: alpha_c >= 16 ({m})", "predicted": ">= 16", "observed": str(d["alpha_c_frozen"]),
                     "artifact": f"plan/dose_scale_{m}.json alpha_c_frozen", "missed": not d["alpha_c_frozen"] >= 16})
        if m in ALPHA_C_PRED:
            rows.append({"prediction": f"SPEC §5.3 / §11 held-out alpha_c from Liu 2026 ({m})", "predicted": " or ".join(map(str, ALPHA_C_PRED[m])),
                         "observed": str(d["alpha_c_frozen"]), "artifact": f"plan/dose_scale_{m}.json alpha_c_frozen",
                         "missed": d["alpha_c_frozen"] not in ALPHA_C_PRED[m]})
    return rows


# ------------------------------------------------------------------------------------------ git, grid
def git(cwd, *args) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def unblind_refusals(thresholds, journal):
    """SCHEMA §5: the thresholds file is committed with no working-tree change, and the journal's PREDICTIONS heading was
    committed strictly before it (earliest commit adding each)."""
    thresholds, journal = pathlib.Path(thresholds).resolve(), pathlib.Path(journal).resolve()
    if not thresholds.exists():
        raise PermissionError(f"{thresholds} is missing: run freeze first")
    top = git(thresholds.parent, "rev-parse", "--show-toplevel")
    if git(top, "status", "--porcelain", "--", str(thresholds)) or not git(top, "ls-files", "--", str(thresholds)):
        raise PermissionError(f"{thresholds} is not committed or has a working-tree change")
    heads = [x for x in journal.read_text().split("\n") if x.startswith("## ") and "PREDICTIONS" in x]
    if not heads:
        raise PermissionError(f"{journal} has no PREDICTIONS heading")
    p = git(top, "log", "--format=%H", "-S", heads[0], "--", str(journal)).split("\n")
    t = git(top, "log", "--diff-filter=A", "--format=%H", "--", str(thresholds)).split("\n")
    if not p[0] or not t[0]:
        raise PermissionError("the PREDICTIONS heading or the thresholds file is not in git history")
    ok = p[-1] != t[-1] and subprocess.run(["git", "-C", top, "merge-base", "--is-ancestor", p[-1], t[-1]]).returncode == 0
    if not ok:
        raise PermissionError(f"PREDICTIONS commit {p[-1][:8]} is not strictly before thresholds commit {t[-1][:8]}")


def fmt(s) -> str:
    f = lambda x: "—" if x is None else f"{x:.4g}"  # noqa: E731
    return "—" if not s else f"{f(s.get('est'))} [{f(s.get('lo'))}, {f(s.get('hi'))}]"


def append_grid(path, rows: list):
    path = pathlib.Path(path)
    top = "" if path.exists() else "| model | arm | readout | label | estimate [95% interval] | n |\n|---|---|---|---|---|---|\n"
    with open(path, "a") as fh:
        fh.write(top + "\n".join(rows) + "\n")


def grid0(v):
    r = v["rows"]
    return [f"| {v['model_key']} | 0 | H_write D_w | {r['H_write']['label']} | {fmt(r['H_write']['D_w'])} | {v['n_one']} ONE |",
            f"| {v['model_key']} | 0 | H_depth | {r['H_depth']['label']} | {fmt(r['H_depth']['diff_t2_minus_t0'])} | |",
            f"| {v['model_key']} | 0 | H_ONE (per model, secondary) | {r['H_ONE_model']['label']} | {fmt(r['H_ONE_model']['d'])} | {r['H_ONE_model']['n_pairs']} pairs |",
            f"| {v['model_key']} | 0 | B1 leak L (per model, secondary) | {r['B1_model']['label']} | {fmt(r['B1_model']['L'])} | {r['B1_model']['n']} ONE |",
            f"| {v['model_key']} | 0 | fc3 ONE | {r['fc3']['label']} | {fmt((r['fc3']['by_class'] or {}).get('ONE'))} | |"]


def grid1(v):
    return [f"| {v['model_key']} | 1 | {t} S_t - S_c | {r['label']} | {fmt(r['paired_S_t_minus_S_c'])} | firing {r['firing_count']} of 20, d* {r['d_star']} |"
            for t, r in v["targets"].items()]


def grid2(v):
    r = v["rows"]
    return [f"| {v['model_key']} | 2 | H_suff o(d_p) | {r['H_suff']['label']} | {fmt(r['H_suff']['o_dp'])} | {v['g_n']['n_contributing']} poles |",
            f"| {v['model_key']} | 2 | H_pred_agree R | {r['H_pred_agree']['label']} | {fmt(r['H_pred_agree']['R'])} | |",
            f"| {v['model_key']} | 2 | H_pred_conf rho | {r['H_pred_conf']['label']} | {fmt(r['H_pred_conf']['spearman'])} | |",
            f"| {v['model_key']} | 2 | H_depth | {r['H_depth']['label']} | {fmt(r['H_depth']['diff_t2_minus_t0'])} | |"]


def study_grid(arm: int, rows: dict, counted: list, models: list) -> list:
    return [f"| study | {arm} | {k} | {lab} | | {len(counted)} counted models {counted} of {models} |" for k, lab in rows.items()]


# ------------------------------------------------------------------------------------------ commands
def roles_of(a, m=None) -> tuple:
    pd = pathlib.Path(a.plan_dir)
    return pd, json.loads((pd / f"roles_{m or a.model}.json").read_text())


def load_reexplain(results_dir, model_key: str, variant: str = "") -> dict:
    """{pole: results/reexplain_<variant><M>.jsonl line}; variant "vocab_" is the B1 with-vocabulary set (A16)."""
    return {r["pole"]: r for r in read_jsonl(pathlib.Path(results_dir) / f"reexplain_{variant}{model_key}.jsonl")}


def cmd_tokcheck(a):
    from tokenizers import Tokenizer
    pd, out = pathlib.Path(a.plan_dir), {}
    for spec in a.tokenizer:
        m, d = spec.split("=", 1)
        plan = json.loads((pd / f"arm0_{m}.json").read_text())
        path = pathlib.Path(d) / "tokenizer.json" if pathlib.Path(d).is_dir() else pathlib.Path(d)
        tk = Tokenizer.from_file(str(path))
        arm1 = pd / f"arm1_{m}.json"
        if arm1.exists():          # the Arm 1 job's prompts are arm0's records without the continuation
            got = json.loads(arm1.read_text())["prompt_sets"]
            assert got == {k: [{"i": r["i"], "prompt": r["prompt"]} for r in plan["prompt_sets"][k]] for k in got}, \
                f"arm1_{m}.json prompts differ from arm0_{m}.json"
        out[m] = {"tokenizer": str(path), "sha256": sha256(path), "arm1_sets_covered": sorted(got) if arm1.exists() else [],
                  **tokcheck_compute(plan["prompt_sets"], plan["neutral_prompts"], lambda t: tk.encode(t, add_special_tokens=False).ids,
                                     plan["bos_id"])}
    print("TOKCHECK", json.dumps(out))


def cmd_gates0(a):
    _, roles = roles_of(a)
    b = pathlib.Path(a.bank)
    hv = roles["inputs"]["harvest_vocab"]
    assert sha256(hv["path"]) == hv["sha256"], f"{hv['path']} changed since the roles file was built"
    proj = {r["pole"]: r for r in read_jsonl(b / "arm0" / "projections.jsonl")}
    recs = {json.loads(p.read_text())["pole"]: json.loads(p.read_text()) for p in sorted((b / "arm0" / "repair").glob("*.json"))}
    out = {"model_key": a.model, "created_utc": utc(), **gates0_compute(proj, json.loads(pathlib.Path(hv["path"]).read_text()), recs),
           "inputs": {hv["path"]: hv["sha256"], str(b / "arm0" / "projections.jsonl"): sha256(b / "arm0" / "projections.jsonl")}}
    write_json(pathlib.Path(a.results_dir) / f"gates0_{a.model}.json", out)
    print("GATES0_OK", json.dumps(finite({"g_proj": out["g_proj"], "repair_fit": out["repair_fit_summary"]})))


def cmd_dose_scale(a):
    b = pathlib.Path(a.bank)
    cal = [r for r in read_jsonl(b / "arm0" / "natural_spans.jsonl") if r["set"] == "calibration"]
    sweep = read_jsonl(b / "sweep" / "samples.jsonl")
    sweep_poles = [{"pole": q, "layer": l} for q, l in sorted({(r["pole"], r["layer"]) for r in sweep if r["pole"] is not None})]
    out = dose_scale_compute(a.model, sweep, cal, read_jsonl(b / "samples" / "coherence_inputs.jsonl"),
                             json.loads((b / "arm0" / "rms.json").read_text()), sweep_poles,
                             json.loads((b / "controls.json").read_text())["pc"])
    out["inputs"] = {str(b / f): sha256(b / f) for f in ("sweep/samples.jsonl", "arm0/natural_spans.jsonl", "samples/coherence_inputs.jsonl",
                                                         "arm0/rms.json", "controls.json")}
    write_json(a.out, out)
    print("DOSE_SCALE_OK", json.dumps(finite({k: out[k] for k in ("model_key", "alpha_c", "alpha_c_frozen", "liu_verdict_frozen",
                                                                   "collapse_rate_by_dose", "stamp", "liu_alpha_star")})))


def cmd_replacement(a):
    pd, roles = roles_of(a, "Q4B")
    f = pathlib.Path(a.bank) / "arm0" / "firing.jsonl"
    out = {**replacement_compute(roles, read_jsonl(f), json.loads((pd / "arm0_Q4B.json").read_text())["firing_rule"]),
           "inputs": {str(f): sha256(f)}}
    write_json(a.out, out)
    print("REPLACEMENT_OK", json.dumps({k: out[k] for k in ("replacement", "firing", "statement")}))


def cmd_budget(a):
    pd, cells = pathlib.Path(a.plan_dir), {}
    for spec in a.bank:
        m, d = spec.split("=", 1)
        cells[m] = {"metrics": json.loads((pathlib.Path(d) / "metrics.json").read_text()),
                    "manifest": json.loads((pathlib.Path(d) / "manifest.json").read_text()),
                    "plan": json.loads((pd / f"arm0_{m}.json").read_text())}
    ds = {m: json.loads((pd / f"dose_scale_{m}.json").read_text()) for m in cells}
    n_pairs = {m: len(json.loads((pd / f"arm1_{m}.json").read_text())["poles"]) // 2 for m in cells if (pd / f"arm1_{m}.json").exists()}
    out = budget_compute(cells, json.loads((pd / "budget.json").read_text()), ds, a.spent_hours, n_pairs)
    write_json(a.out, out)
    print("BUDGET_OK", json.dumps(finite({"decision": out["decision"], "applied": out["applied"], "total": out["steps"][-1]["total"]})))


def cmd_budget_min(a):
    pre = pathlib.Path(a.preflight)
    out = {**budget_min_compute(json.loads((pre / "metrics.json").read_text()), json.loads((pre / "manifest.json").read_text()),
                                json.loads((pathlib.Path(a.plan_dir) / "budget.json").read_text())),
           "inputs": {str(pre / f): sha256(pre / f) for f in ("metrics.json", "manifest.json")}, "created_utc": utc()}
    if a.out:
        write_json(a.out, out)
    print("BUDGET_MIN", json.dumps({m: c["budget_min"] for m, c in out["cells"].items()}),
          json.dumps(finite({m: round(c["projected_hours"], 3) for m, c in out["cells"].items()})))


def plan_n_distinct(a) -> dict:
    """{pole: n_distinct} of plan/arm0_<M>.json judge0_pc (A20)."""
    return {e["pole"]: e["n_distinct"] for e in json.loads((pathlib.Path(a.plan_dir) / f"arm0_{a.model}.json").read_text())["judge0_pc"]}


def cmd_judge0(a):
    """§9.5 J0: the §6 G-judge0 gate on the J0 preflight answers (results/judge_j0-<M>.jsonl), with A20's split."""
    rd = pathlib.Path(a.results_dir)
    rows = [r for r in judge_rows(rd, f"j0-{a.model}") if r["preflight"] and r["answer"] is not None]
    pick = lambda task: {r["pole"]: int(r["answer"] == "yes") for r in rows if r["task"] == task}  # noqa: E731
    sh = list(pick("yn_shuf1").values()) + list(pick("yn_shuf2").values())
    g = judge0_gate(pick("pc_true"), pick("pc_shuf"), float(np.mean(sh)) if sh else float("nan"), plan_n_distinct(a))
    write_json(rd / f"j0_gates_{a.model}.json", {"model_key": a.model, "g_judge0": g, "created_utc": utc()})
    print("JUDGE0", a.model, json.dumps(finite({"label": g["label"], "by_distinct": g["by_distinct"]})))


def a22_of(a) -> dict:
    """judge_draws_<M>.json -> a22 (A22: subsample, task lists, g_n_arm0)."""
    return json.loads((pathlib.Path(a.plan_dir) / f"judge_draws_{a.model}.json").read_text())["a22"]


def a0_fc3_rows(a) -> list:
    """results/judge_a0-fc3-<M>.jsonl when the a22 task lists run fc3, else none (A22 drops Arm 0 fc3)."""
    run = any("fc3" in ts for ts in a22_of(a)["tasks"].values())
    return judge_rows(a.results_dir, f"a0-fc3-{a.model}") if run else []


def cmd_freeze0(a):
    _, roles = roles_of(a)
    rd = pathlib.Path(a.results_dir)
    retest = judge_rows(rd, f"a0-retest-{a.model}")
    ret = {r["retest_of"] for r in retest}
    ctrl = [r for r in judge_rows(rd, f"a0-yn-{a.model}") if r["task"] in ("yn_shuf1", "yn_shuf2", "yn_fixed") or r["id"] in ret] + \
        [r for r in a0_fc3_rows(a) if r["id"] in ret]                                       # filtered by role before use
    out = {**freeze0_compute(a.model, roles, ctrl, retest, judge_rows(rd, f"a0-pc-{a.model}"), plan_n_distinct(a)),
           "created_utc": utc()}
    write_json(a.out, out)
    print("FREEZE0_OK", json.dumps(finite({"delta_w": out["delta_w"], "g_judge0": out["g_judge0"]["pass"],
                                           "room": {k: v["stamps"] for k, v in out["room"].items()}})))


def cmd_unblind0(a):
    assert not pathlib.Path(a.out).exists(), f"refusing to overwrite {a.out}"
    unblind_refusals(a.thresholds, a.journal)
    _, roles = roles_of(a)
    rd, b = pathlib.Path(a.results_dir), pathlib.Path(a.bank)
    v = unblind0_compute(a.model, roles, json.loads(pathlib.Path(a.thresholds).read_text()),
                         json.loads((rd / f"gates0_{a.model}.json").read_text()), judge_rows(rd, f"a0-yn-{a.model}"),
                         judge_rows(rd, f"a0-yn-vocab-{a.model}"), judge_rows(rd, f"a0-yn-orig-{a.model}"),
                         a0_fc3_rows(a), json.loads((b / "metrics.json").read_text()),
                         {r["pole"]: r for r in read_jsonl(b / "arm0" / "projections.jsonl")}, load_reexplain(rd, a.model),
                         load_reexplain(rd, a.model, "vocab_"), a22_of(a))
    write_json(a.out, {**v, "created_utc": utc()})
    append_grid(rd / "grid.md", grid0(v))
    print("UNBLIND0_OK", json.dumps(finite({k: r["label"] for k, r in v["rows"].items()}), ensure_ascii=False))


def verdicts(rd, arm: int) -> dict:
    return {json.loads(p.read_text())["model_key"]: json.loads(p.read_text()) for p in sorted(pathlib.Path(rd).glob(f"verdict_arm{arm}_*.json"))
            if re.fullmatch(rf"verdict_arm{arm}_[A-Za-z0-9]+\.json", p.name)}


def cmd_study0(a):
    rd = pathlib.Path(a.results_dir)
    vs = verdicts(rd, 0)
    counted = [m for m, v in vs.items() if not v["model_reasons"]]
    hone = pooled([vs[m]["pair_values_H_ONE"] for m in counted])
    null = [x for m in counted for x in vs[m]["null_values_H_ONE"]]
    rm = room("H_ONE", null, hone["n"], 0.15, plant=[x for m in counted for x in vs[m]["plant_H_ONE"]]) if null else None
    b1 = pooled([vs[m]["pair_values_B1"] for m in counted])
    b1_null = [x for m in counted for x in vs[m]["null_values_B1"]]
    rm_b = room("B1", b1_null, b1["n"], B1_EFFECT, plant=[x for m in counted for x in vs[m]["plant_B1"]]) if b1_null else None
    cancelled = [] if len(counted) == len(vs) else [f"models cancelled: {sorted(set(vs) - set(counted))}"]
    known = {}
    for spec in a.bank:
        m, d = spec.split("=", 1)
        known[m] = json.loads((pathlib.Path(d) / "controls.json").read_text())["known"]["failing_input"]
    assert set(known) == set(vs), f"--bank given for {sorted(known)}, verdicts for {sorted(vs)}"
    fr = study0_frozen(vs, known)
    out = {"models": sorted(vs), "counted": counted, "frozen_rule": fr,
           "H_write": study_label([v["rows"]["H_write"]["label"] for v in vs.values()], "H_WRITE"),
           "H_depth": study_label([v["rows"]["H_depth"]["label"] for v in vs.values()], "SUPPORTED"),
           "H_ONE_pooled": {**stamped(apply_room(label_dir(hone, eq=EQUIV["H_ONE"]), {"SUPPORTED"}, rm),
                                      [] if len(counted) == len(vs) else [f"models cancelled: {sorted(set(vs) - set(counted))}"]),
                            "d": hone, "room": rm},
           "B1_pooled": {**stamped(apply_room(label_dir(b1, eq=EQUIV["B1"]), {"SUPPORTED"}, rm_b), cancelled), "L": b1, "room": rm_b}}
    out["amended_exploratory"] = {"H_write": out["H_write"], "H_depth": out["H_depth"],
                                  "note": "exploratory: holds only under A24 (1), A24 (2) and A25"}
    write_json(rd / "study_arm0.json", out)
    append_grid(rd / "grid.md", study0_rows(out))
    print("STUDY0_OK", json.dumps({**{k: out[k] for k in ("H_write", "H_depth", "counted")}, "B1": out["B1_pooled"]["label"]},
                                  ensure_ascii=False))


def cmd_repair(a):
    rd = pathlib.Path(a.results_dir)
    if a.pooled:
        res = {m: json.loads((rd / f"repair_{m}.json").read_text()) for m in ("Q4B", "G4")}
        for r in res.values():
            r["ratios"] = {p: (math.inf if v is None else v) for p, v in r["ratios"].items()}
        out = repair_pooled(res, {m: roles_of(a, m)[1] for m in res})
        write_json(rd / "repair_pooled.json", out)
        append_grid(rd / "grid.md", repair_pooled_rows(out))
        print("REPAIR_POOLED_OK", json.dumps(finite({"H_repair": out["H_repair"]["label"], "H_repair_ONE": out["H_repair_ONE"]["label"]}),
                                             ensure_ascii=False))
        return
    _, roles = roles_of(a)
    b = pathlib.Path(a.bank)
    recs = {json.loads(p.read_text())["pole"]: json.loads(p.read_text()) for p in sorted((b / "arm0" / "repair").glob("*.json"))}
    metrics = json.loads((b / "metrics.json").read_text())
    res = repair_compute(roles, recs, metrics)
    res["secondary"] = {name: {k: x[k] for k in ("M_r", "H_repair", "n_qualifying")} for name, x in
                        (("post_mlp_norm_recomputed", repair_compute(roles, recs, metrics, "direct_dnll_postrenorm")),
                         ("final_norm_recomputed", repair_compute(roles, recs, metrics, "direct_dnll_finalrenorm")),
                         ("neuronoff_all_position_total", repair_compute(roles, recs, metrics, total="dnll_off")))}
    write_json(rd / f"repair_{a.model}.json", res)
    append_grid(rd / "grid.md", repair_rows(a.model, res))
    print("REPAIR_OK", json.dumps(finite({"H_repair": res["H_repair"]["label"], "n_qualifying": res["n_qualifying"]}), ensure_ascii=False))


def arm1_inputs(a):
    pd, roles = roles_of(a)
    plan = json.loads((pd / f"arm1_{a.model}.json").read_text())
    b, b0 = pathlib.Path(a.bank), pathlib.Path(a.arm0_bank)
    sets = {s: read_jsonl(b / "arm1" / "samples" / f"{s}.jsonl") for s in plan["prompt_sets"]}
    cal = [r for r in read_jsonl(b0 / "arm0" / "natural_spans.jsonl") if r["set"] == "calibration"]
    rep = pd / "arm1_replacement_Q4B.json"
    replacement = json.loads(rep.read_text())["replacement"] if a.model == "Q4B" and rep.exists() else None
    return pd, roles, plan, sets, cal, b, b0, replacement


def cmd_freeze1(a):
    _, roles, plan, sets, cal, b, _, replacement = arm1_inputs(a)
    controls = {e["ctrl"] for e in roles["arm1"]}
    for s in sets:                                    # targets keep only coherence fields and realized_cos (SPEC §7.5)
        sets[s] = [r if r["pole"] in controls or r["pole"] is None else
                   {k: r[k] for k in ("pole", "alpha", "prompt_i", "k", "n_new_tokens", "nll_clean_mean", "realized_cos")} for r in sets[s]]
    out = {**freeze1_compute(a.model, roles, plan, sets, cal, read_jsonl(b / "samples" / "coherence_inputs.jsonl"), replacement),
           "created_utc": utc()}
    write_json(a.out, out)
    print("FREEZE1_OK", json.dumps(finite({"g_coh": out["g_coh"]["pass"], "pairs": {t: {"d_star": p["d_star"], "room": p["room"]}
                                                                                    for t, p in out["pairs"].items()}})))


def stamp_labels(obj, text: str):
    """Append `text` to every label string of a verdict (keys "label" and "label_if_not_cancelled"), in place."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("label", "label_if_not_cancelled") and isinstance(v, str):
                obj[k] = v + text
            else:
                stamp_labels(v, text)
    elif isinstance(obj, list):
        for v in obj:
            stamp_labels(v, text)
    return obj


def dose_stamp(plan_dir, model: str) -> str:
    """§13 reading 2 and A24 (3): the sweep's G-coh stamp and the running alpha_c, printed beside every Arm 1 and Arm 2
    label of the model ("" when its G-coh passed)."""
    d = json.loads((pathlib.Path(plan_dir) / f"dose_scale_{model}.json").read_text())
    if d["stamp"] is None:
        return ""
    if d.get("alpha_c_fallback"):
        return (f" [exploratory: dose scale not validated; {d['stamp']}; running alpha_c {d['alpha_c']} (A24), frozen alpha_c "
                f"{d['alpha_c_frozen']} (A27)]")
    return f" [{d['stamp']}]"


def cmd_unblind1(a):
    assert not pathlib.Path(a.out).exists(), f"refusing to overwrite {a.out}"
    unblind_refusals(a.thresholds, a.journal)
    _, roles, plan, sets, _, b, b0, _ = arm1_inputs(a)
    f = b0 / "arm0" / "firing.jsonl"
    v = unblind1_compute(a.model, roles, plan, json.loads(pathlib.Path(a.thresholds).read_text()), sets,
                         json.loads((b / "metrics.json").read_text()), read_jsonl(f) if f.exists() else [])
    stamp_labels(v, dose_stamp(a.plan_dir, a.model))
    write_json(a.out, {**v, "created_utc": utc()})
    append_grid(pathlib.Path(a.out).parent / "grid.md", grid1(v))
    print("UNBLIND1_OK", json.dumps(finite({t: r["label"] for t, r in v["targets"].items()}), ensure_ascii=False))


def arm2_inputs(a):
    pd, roles = roles_of(a)
    b, b0, rd = pathlib.Path(a.bank), pathlib.Path(a.arm0_bank), pathlib.Path(a.results_dir)
    samples = read_jsonl(b / "arm2" / "samples.jsonl")
    alpha0 = [r for r in read_jsonl(b0 / "sweep" / "samples.jsonl") if r["alpha"] == 0] + \
        [r for r in samples if r["condition"] == "steer" and r["pole"] is None]
    cal = [r for r in read_jsonl(b0 / "arm0" / "natural_spans.jsonl") if r["set"] == "calibration"]
    return pd, roles, b, rd, alpha0, cal, samples


def cmd_freeze2(a):
    _, roles, b, rd, alpha0, cal, samples = arm2_inputs(a)
    retest = judge_rows(rd, f"a2-retest-{a.model}")
    ret = {r["retest_of"] for r in retest}
    ctrl = [r for r in judge_rows(rd, f"a2-fc3-{a.model}") if r["task"] in ("fc3_alpha0", "fc3_label_prompt") or r["id"] in ret]
    out = {**freeze2_compute(a.model, roles, ctrl, retest, alpha0, cal, read_jsonl(b / "samples" / "coherence_inputs.jsonl"),
                             [r for r in samples if r["condition"] == "label_prompt"]), "created_utc": utc()}
    write_json(a.out, out)
    print("FREEZE2_OK", json.dumps(finite({"g_coh": out["g_coh"]["pass"], "g_judge2": out["g_judge2"]})))


def cmd_unblind2(a):
    assert not pathlib.Path(a.out).exists(), f"refusing to overwrite {a.out}"
    unblind_refusals(a.thresholds, a.journal)
    pd, roles, b, rd, _, _, samples = arm2_inputs(a)
    plan = json.loads((pd / f"arm2_{a.model}.json").read_text())
    v = unblind2_compute(a.model, roles, json.loads(pathlib.Path(a.thresholds).read_text()), judge_rows(rd, f"a2-fc3-{a.model}"),
                         judge_rows(rd, f"a2-guess-{a.model}"), json.loads((b / "metrics.json").read_text()), float(plan["primary_dose"]),
                         samples)
    stamp_labels(v, dose_stamp(pd, a.model))
    write_json(a.out, {**v, "created_utc": utc()})
    append_grid(rd / "grid.md", grid2(v))
    print("UNBLIND2_OK", json.dumps(finite({k: r["label"] for k, r in v["rows"].items()}), ensure_ascii=False))


def study2_compute(vs: dict, dose_scales: dict) -> dict:
    """A27 (2): models whose dose scale is the A24 fallback ("dose scale not validated") keep their printed Arm 2 rows,
    exploratory, but leave the study verdicts; with one model left (Q4B) its label decides. A27 (4): if a counted
    model's freeze2 G-coh failed on its natural test spans or its alpha = 0 test half, the study has no headline
    H_suff verdict."""
    excluded = sorted(m for m in vs if dose_scales[m].get("alpha_c_fallback"))
    incl = sorted(m for m in vs if m not in excluded)
    weak = [m for m in incl if not all(vs[m]["g_coh"]["passing_inputs"][k]["pass"] for k in ("natural_test", "alpha0"))]

    def verdict(key):
        if key == "H_suff" and weak:
            return f"NO HEADLINE (A27 (4): {weak} G-coh failed on natural test spans or the alpha = 0 test half)"
        labs = [vs[m]["rows"][key]["label"] for m in incl]
        if len(incl) == 1:
            return "MIXED (0 counted models)" if labs[0].startswith("CANCELLED") else f"{head(labs[0])} (decided on {incl[0]} alone, A27)"
        return study_label(labs, "SUPPORTED")
    counted = [m for m in incl if not vs[m]["rows"]["H_suff"]["label"].startswith("CANCELLED")]
    return {"models": sorted(vs), "counted": counted, "H_suff": verdict("H_suff"), "H_depth": verdict("H_depth"),
            "excluded": {m: "dose scale not validated; coherence gate not shown to work (A27)" for m in excluded},
            "excluded_rows": {m: {k: vs[m]["rows"][k]["label"] for k in ("H_suff", "H_depth")} for m in excluded}}


def study0_rows(out: dict) -> list:
    """grid.md rows of study_arm0.json: the frozen-rule readings first, then the amended ones, marked exploratory."""
    fr, n = out["frozen_rule"], len(out["counted"])
    return ([f"| {m} | 0 | frozen-rule reasons | {'; '.join(r) if r else 'none'} | | |" for m, r in fr["reasons"].items()] +
            [f"| study | 0 | {k} (frozen rule) | {fr[k]} | | |" for k in ("H_write", "H_depth")] +
            [f"| pooled | 0 | {k} (frozen rule) | {fr[f'{k}_pooled']['label']} | {fmt(fr[f'{k}_pooled']['value'])} | "
             f"counted under the frozen rules: {fr[f'{k}_pooled']['counted']} |" for k in ("H_ONE", "B1")] +
            [f"| study | 0 | {k} (amended, exploratory) | {out[k]} [exploratory: A24, A25] | | {n} counted models |" for k in ("H_write", "H_depth")] +
            [f"| pooled | 0 | H_ONE (amended, exploratory) | {out['H_ONE_pooled']['label']} | {fmt(out['H_ONE_pooled']['d'])} | "
             f"{out['H_ONE_pooled']['d']['n']} pairs, {n} counted models |",
             f"| pooled | 0 | B1 leak L (amended, exploratory) | {out['B1_pooled']['label']} | {fmt(out['B1_pooled']['L'])} | "
             f"{out['B1_pooled']['L']['n']} ONE items, {n} counted models |"])


def study2_rows(out: dict) -> list:
    """grid.md rows of study_arm2.json: frozen first, amended (exploratory) second, the Q4B H_suff row beside."""
    fr, am, n = out["frozen_rule"], out["amended_exploratory"], len(out["counted"])
    rows = [f"| study | 2 | H_suff (frozen rule) | {fr['H_suff']}; two-model rule on Q4B: {fr['H_suff_two_model_rule_on_Q4B']} | | |",
            f"| study | 2 | H_suff (amended, exploratory) | {am['H_suff']} [exploratory: A24, A27] | | {n} counted models |"]
    d = out["Q4B_H_suff"]
    if d is not None:
        fc = d["frozen_coherent_only"]
        rows.append(f"| Q4B | 2 | H_suff beside the study row | {d['label']} | {fmt(d['o_dp'])} | "
                    f"frozen coherent-only: {fc['label'] if fc else 'none'} {fmt(fc['o_dp']) if fc else ''} |")
    return rows + [f"| study | 2 | H_depth (frozen rule) | {fr['H_depth']} | | |",
                   f"| study | 2 | H_depth (amended, exploratory) | {am['H_depth']} [exploratory: A24, A27] | | {n} counted models |"]


def repair_rows(m: str, res: dict) -> list:
    return [f"| {m} | repair | H_repair M_r | {res['H_repair']['label']} | {fmt(res['M_r'])} | "
            f"{res['n_qualifying']}/{res['n_measured']} poles qualify, {res['n_direct_not_replicated']} with D_p <= 0 |"]


def repair_pooled_rows(out: dict) -> list:
    return [f"| pooled | repair | H_repair M_r | {out['H_repair']['label']} | {fmt(out['M_r'])} | {len(out['counted'])} counted models |",
            f"| pooled | repair | H_repair_ONE | {out['H_repair_ONE']['label']} | {fmt(out['paired_ratio_diff'])} | |"]


def grid_compute(rd) -> list:
    """Review C8: every grid.md row regenerated from the current results files (no stale rows)."""
    rd = pathlib.Path(rd)
    load = lambda f: json.loads((rd / f).read_text())  # noqa: E731
    rows = [r for v in verdicts(rd, 0).values() for r in grid0(v)]
    rows += study0_rows(load("study_arm0.json")) if (rd / "study_arm0.json").exists() else []
    rows += [r for f in sorted(rd.glob("repair_*.json")) if f.name != "repair_pooled.json"
             for r in repair_rows(load(f.name)["model_key"], load(f.name))]
    rows += repair_pooled_rows(load("repair_pooled.json")) if (rd / "repair_pooled.json").exists() else []
    rows += [r for v in verdicts(rd, 1).values() for r in grid1(v)]
    rows += [r for v in verdicts(rd, 2).values() for r in grid2(v)]
    rows += study2_rows(load("study_arm2.json")) if (rd / "study_arm2.json").exists() else []
    return rows


def cmd_grid(a):
    out = pathlib.Path(a.out)
    assert not out.exists(), f"refusing to overwrite {out}"
    rows = grid_compute(a.results_dir)
    append_grid(out, rows)
    print("GRID_OK", out, len(rows))


def cmd_study2(a):
    rd = pathlib.Path(a.results_dir)
    vs = verdicts(rd, 2)
    amended = study2_compute(vs, {m: json.loads((pathlib.Path(a.plan_dir) / f"dose_scale_{m}.json").read_text()) for m in vs})
    fr = study2_frozen(vs)
    counted = amended["counted"]
    q = vs.get("Q4B")
    detail = None if q is None else {
        "label": q["rows"]["H_suff"]["label"], "o_dp": q["rows"]["H_suff"]["o_dp"],
        "frozen_coherent_only": None if "frozen_coherent_only" not in q else
        {"label": q["frozen_coherent_only"]["rows"]["H_suff"]["label"], "o_dp": q["frozen_coherent_only"]["rows"]["H_suff"]["o_dp"],
         "paired_minus_alpha0": q["frozen_coherent_only"]["rows"]["H_suff"]["paired_minus_alpha0"]}}
    out = {"frozen_rule": fr, "amended_exploratory": {**amended, "note": "exploratory: holds only under A24 and A27"},
           "Q4B_H_suff": detail, "models": amended["models"], "counted": counted,
           "H_suff": f"frozen: {fr['H_suff']} (two-model rule on Q4B: {fr['H_suff_two_model_rule_on_Q4B']}); amended, exploratory: "
                     f"{amended['H_suff']}",
           "H_depth": f"frozen: {fr['H_depth']}; amended, exploratory: {amended['H_depth']}"}
    write_json(rd / "study_arm2.json", out)
    append_grid(rd / "grid.md", study2_rows(out))
    print("STUDY2_OK", json.dumps(out))


# ------------------------------------------------------------------------------------------ frozen-rule readings (review E2)
def g_proj_frozen_stamp(gp: dict) -> str:
    """The frozen exact-order G-proj reading printed beside every Arm 0 label of a model that fails it (A28 (3), review
    A7); "" when the frozen reading passes or is absent."""
    fz = gp.get("frozen")
    if fz is None or fz["pass"]:
        return ""
    return f" [frozen G-proj reading: exact-order share {fz['share_top10_equal']:.3f} < {PROJ_SHARE}, CANCELLED under the frozen rule]"


def frozen_arm0_reasons(v: dict, known_failing: dict) -> list:
    """The frozen rules that would cancel a model's Arm 0 labels: G-proj by exact-order lists (before A24 (1)) and the
    layer-0-only G-known failing input (before A24 (2)); `known_failing` = the Arm 0 job's controls.json known
    failing_input (a job made before A24 zeroed layer 0 only)."""
    out = []
    fz = v["g_proj"].get("frozen")
    if fz is not None and not fz["pass"]:
        out.append(f"frozen G-proj: exact-order share {fz['share_top10_equal']:.3f} < {PROJ_SHARE}")
    l0 = known_failing["layer0_only"]["shift"] if known_failing.get("layer0_only") else known_failing["shift"]
    if l0 < known_failing["min_shift_nats"]:
        out.append(f"frozen G-known failing input (layer 0 only) shifts {l0:.3f} < {known_failing['min_shift_nats']} nats")
    return out


def frozen_room_label(label: str, rm: dict, supported: set) -> str:
    """A supported label whose frozen constant-plant room check cannot fail reads NOT READABLE under the frozen rule."""
    h = head(label)
    fz = (rm or {}).get("frozen_constant_plant")
    if h in supported and fz is not None and not fz["failing_input_n5"]["fails"]:
        return f"{h} NOT READABLE (frozen constant-plant room check cannot fail)"
    return h


def study0_frozen(vs: dict, known_failing: dict, outer=ROOM_OUTER, inner=ROOM_INNER) -> dict:
    """§7.1 study labels under the frozen rules, printed before the amended ones (review E2); the pooled H_ONE and B1
    rows over the models the frozen rules count (review B1), with the frozen constant-plant room check."""
    per, lab = {}, {}
    for m, v in vs.items():
        rs = frozen_arm0_reasons(v, known_failing[m])
        per[m] = rs
        lab[m] = {k: (cancel_stamp(rs) if rs else frozen_room_label(v["rows"][k]["label"], v["rows"][k].get("room"), sup))
                  for k, sup in (("H_write", {"H_WRITE"}), ("H_depth", {"SUPPORTED"}))}
    room_only = {m: frozen_room_label(v["rows"]["H_write"]["label"], v["rows"]["H_write"].get("room"), {"H_WRITE"}) +
                 g_proj_frozen_stamp(v["g_proj"]) for m, v in vs.items()}
    fc = sorted(m for m in vs if not per[m] and not vs[m]["model_reasons"])

    def pool(vals, nulls, rule, effect):
        d = pooled([vs[m][vals] for m in fc])
        nl = [x for m in fc for x in vs[m][nulls]]
        rm = room(rule, nl, d["n"], effect, 0.0, outer, inner) if nl else None
        return {"label": apply_room(label_dir(d, eq=EQUIV[rule]), {"SUPPORTED"}, rm) if d["n"] else "MIXED (0 counted models)",
                "value": d, "counted": fc, "room": rm}
    a24_only = {m: cancel_stamp(per[m]) if per[m] else v["rows"]["H_write"]["label"] for m, v in vs.items()}
    return {"rule": "frozen: exact-order G-proj, layer-0 G-known failing input, constant-plant room check",
            "reasons": per, "labels": lab,
            "H_write": study_label([lab[m]["H_write"] for m in vs], "H_WRITE"),
            "H_depth": study_label([lab[m]["H_depth"] for m in vs], "SUPPORTED"),
            "H_write_room_check_only": {"labels": room_only, "study": study_label(list(room_only.values()), "H_WRITE")},
            "H_write_frozen_g_proj_g_known_only": {"labels": a24_only, "study": study_label(list(a24_only.values()), "H_WRITE")},
            "counted": fc, "H_ONE_pooled": pool("pair_values_H_ONE", "null_values_H_ONE", "H_ONE", 0.15),
            "B1_pooled": pool("pair_values_B1", "null_values_B1", "B1", B1_EFFECT)}


def study2_frozen(vs: dict) -> dict:
    """Arm 2 study labels under the frozen rules: P8 cancels a model whose freeze2 G-coh failed; §7.1 needs two counted
    models (one counted model reads MIXED (1 counted models))."""
    lab = {m: ("CANCELLED (P8: coherence gate not shown to work)" if not v["g_coh"]["pass"] else v["rows"]["H_suff"]["label"])
           for m, v in vs.items()}
    q = vs.get("Q4B")
    dep = {m: ("CANCELLED (P8: coherence gate not shown to work)" if not v["g_coh"]["pass"] else v["rows"]["H_depth"]["label"])
           for m, v in vs.items()}
    allc = lambda ls: ("CANCELLED (P8: coherence gate not shown to work on every model)"  # noqa: E731
                       if all(x.startswith("CANCELLED") for x in ls.values()) else study_label(list(ls.values()), "SUPPORTED"))
    return {"rule": "frozen: P8 (G-coh failed: CANCELLED) and the §7.1 two-model rule", "labels": lab, "labels_H_depth": dep,
            "H_depth": allc(dep),
            "H_suff": allc(lab),
            "H_suff_two_model_rule_on_Q4B": None if q is None else study_label([q["rows"]["H_suff"]["label"]], "SUPPORTED")}


def cmd_literal_split(a):
    pd, rd, b = pathlib.Path(a.plan_dir), pathlib.Path(a.results_dir), pathlib.Path(a.bank)
    roles = json.loads((pd / f"roles_{a.model}.json").read_text())
    proj = {r["pole"]: r for r in read_jsonl(b / "arm0" / "projections.jsonl")}
    kw = {}
    if a.arm2_bank:
        kw["samples"] = read_jsonl(pathlib.Path(a.arm2_bank) / "arm2" / "samples.jsonl")
        kw["band"] = json.loads((rd / f"thresholds_arm2_{a.model}.json").read_text())["band"]
        f = rd / f"judge_a2-fc3-{a.model}.jsonl"
        if f.exists():                               # an Arm 2 that was not judged (O7) has samples but no answers
            kw["fc3_rows"], kw["d_p"] = read_jsonl(f), float(json.loads((pd / f"arm2_{a.model}.json").read_text())["primary_dose"])
    out = literal_split_compute(a.model, roles, json.loads((pd / f"judge_draws_{a.model}.json").read_text())["a22"],
                                judge_rows(rd, f"a0-yn-{a.model}"), proj, load_reexplain(rd, a.model), **kw)
    write_json(rd / f"literal_split_{a.model}.json", {**out, "created_utc": utc()})
    show = {"arm0": {g: (x["n"], fmt(x["D_w"])) for g, x in out["arm0"].items()}}
    if out["arm2"]:
        show["arm2"] = {g: (out["arm2"][g]["n"], fmt(out["arm2"][g]["paired_gain"]), "two-way " + fmt(out["arm2"][g]["paired_gain_two_way"]))
                        for g in ("literal", "non_literal")}
        show["p_lit"] = out["arm2"]["p_lit"]["reading"]
    print("LITERAL_SPLIT", a.model, json.dumps(show, ensure_ascii=False))


def liu_not_decidable(rows: list, ds: dict) -> list:
    """A28 (3), review 5c: a Liu row of a model whose sweep band failed G-coh reads "not decidable with this coherence
    instrument", with the stamp; in place."""
    for r in rows:
        m = re.search(r"\((G4|O7|Q4B)\)$", r["prediction"])
        if m and "Liu" in r["prediction"] and ds[m.group(1)]["stamp"]:
            for k in ("observed_frozen", "observed_amended"):
                r[k] += f" [collapse instrument failed G-coh on {m.group(1)}]"
            r["missed_frozen"] = r["missed_amended"] = None
            r["why"] = "not decidable with this coherence instrument"
    return rows


def cmd_missed(a):
    rd, pd = pathlib.Path(a.results_dir), pathlib.Path(a.plan_dir)
    ds = {json.loads(p.read_text())["model_key"]: json.loads(p.read_text()) for p in sorted(pd.glob("dose_scale_*.json"))}
    v1 = verdicts(rd, 1)
    rows56 = liu_not_decidable([{**r, "observed_frozen": r["observed"], "observed_amended": r["observed"], "missed_frozen": r["missed"],
                                 "missed_amended": r["missed"]} for r in missed_compute(v1, ds)], ds)
    for r in rows56:
        r["pred_id"] = "alpha_c >= 16 in every model" if "alpha_c >= 16" in r["prediction"] else r["prediction"]
        r["status_frozen"], r["status_amended"] = status(r["missed_frozen"]), status(r["missed_amended"])
    tex = json.loads((pd / "arm1_replacement_Q4B.json").read_text())
    if tex["replacement"] is None:                  # §5.6 TeX row: target and control, not scorable when the slot dropped
        rows56 += [{"pred_id": f"SPEC §5.6 Q4B TeX replacement{c}", "prediction": f"SPEC §5.6 Q4B TeX replacement{c}",
                    "predicted": "SWINGS if L35N1881- is chosen; NO SWING otherwise" if not c else "< 0.5 macro",
                    "observed_frozen": tex["statement"], "observed_amended": tex["statement"], "status_frozen": "not scorable",
                    "status_amended": "not scorable", "artifact": "plan/arm1_replacement_Q4B.json"} for c in ("", " control")]
    cs = [r["S"][r["control"]] for v in v1.values() for r in v["targets"].values() if r["S"]]
    obs = ", ".join(f"{x:+.3f}" for x in cs)
    gone = any(r["label"].startswith("CANCELLED") for v in v1.values() for r in v["targets"].values())    # NOT_SCORABLE_RULE
    st = status(None if gone else any(abs(x) >= 0.10 for x in cs))
    rows56.append({"pred_id": "SPEC §5.6 summary: every control |S| < 0.10", "prediction": "SPEC §5.6 summary: every control |S| < 0.10",
                   "predicted": "every control has |S| < 0.10", "observed_frozen": obs, "observed_amended": obs,
                   "status_frozen": st, "status_amended": st, "artifact": "results/verdict_arm1_*.json targets.*.S",
                   **({"why": "readout cancelled by a gate"} if gone else {})})
    v2 = verdicts(rd, 2)
    load = lambda f: json.loads((rd / f).read_text())  # noqa: E731
    res = {"s0": load("study_arm0.json"), "s2": load("study_arm2.json"), "v0": verdicts(rd, 0), "v1": v1, "v2": v2,
           "t0": {m: load(f"thresholds_arm0_{m}.json") for m in verdicts(rd, 0)}, "t2": {m: load(f"thresholds_arm2_{m}.json") for m in v2},
           "roles": {m: json.loads((pd / f"roles_{m}.json").read_text()) for m in sorted(set(verdicts(rd, 0)) | set(v2))},
           "rx_class": {m: {r["pole"]: r["class"] for r in read_jsonl(rd / f"reexplain_{m}.jsonl")} for m in verdicts(rd, 0)},
           "repair": {load(f.name)["model_key"]: load(f.name) for f in sorted(rd.glob("repair_*.json")) if f.name != "repair_pooled.json"},
           "repair_pooled": load("repair_pooled.json"), "rows56": rows56,
           "gates": [(str(f.parent.parent.relative_to(pathlib.Path(a.bank_root))), json.loads(f.read_text())["gates"])
                     for f in sorted(pathlib.Path(a.bank_root).glob("*/*/out/metrics.json"))]}
    rows = rows56 + missed_full(prediction_rows(pathlib.Path(a.journal).read_text()), res)
    out = rd / "missed_predictions.md"
    assert not out.exists(), f"refusing to overwrite {out}"
    tf, ta = tally(rows, "status_frozen"), tally(rows, "status_amended")
    line = lambda t: ", ".join(f"{t[s]} {s}" for s in ("missed", "not scorable", "not readable", "landed")) + f", of {t['predictions']} predictions"  # noqa: E731
    out.write_text("# Missed predictions (SPEC §5.6, §11)\n\n"
                   f"Rule: {NOT_SCORABLE_RULE}\n\n"
                   f"Frozen rules: {line(tf)}. Amended (exploratory): {line(ta)}. Counted over predictions; a prediction with "
                   "several rows (alpha_c >= 16 in every model) counts once.\n\n"
                   "| prediction | predicted | observed (frozen rules) | observed (amended) | frozen | amended | artifact |\n"
                   "|---|---|---|---|---|---|---|\n" +
                   "\n".join("| " + " | ".join(str(x).replace("|", "\\|") for x in (
                       r["prediction"], r["predicted"], r["observed_frozen"], r["observed_amended"],
                       r["status_frozen"] + (f" ({r['why']})" if r.get("why") and r["status_frozen"] == "not scorable" else ""),
                       r["status_amended"] + (f" ({r['why']})" if r.get("why") and r["status_amended"] == "not scorable" else ""),
                       r["artifact"])) + " |" for r in rows) + "\n")
    print("MISSED_OK", json.dumps({"frozen": tf, "amended": ta}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neuron_rw_analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, *req, multi=()):
        s = sub.add_parser(name)
        for k in req:
            s.add_argument(k, required=True)
        for k in multi:
            s.add_argument(k, action="append", required=True)
        return s
    p("tokcheck", "--plan-dir", multi=("--tokenizer",))
    p("gates0", "--model", "--plan-dir", "--bank", "--results-dir")
    p("dose_scale", "--model", "--bank", "--out")
    p("replacement", "--plan-dir", "--bank", "--out")
    p("budget_min", "--plan-dir", "--preflight").add_argument("--out")
    b = p("budget", "--plan-dir", "--out", multi=("--bank",))
    b.add_argument("--spent-hours", type=float, required=True)
    p("judge0", "--model", "--plan-dir", "--results-dir")
    p("freeze0", "--model", "--plan-dir", "--results-dir", "--out")
    p("unblind0", "--model", "--plan-dir", "--bank", "--results-dir", "--thresholds", "--journal", "--out")
    p("study0", "--results-dir", multi=("--bank",))
    r = p("repair", "--plan-dir", "--results-dir")
    r.add_argument("--model")
    r.add_argument("--bank")
    r.add_argument("--pooled", action="store_true")
    p("freeze1", "--model", "--plan-dir", "--bank", "--arm0-bank", "--out")
    p("unblind1", "--model", "--plan-dir", "--bank", "--arm0-bank", "--thresholds", "--journal", "--out")
    p("freeze2", "--model", "--plan-dir", "--bank", "--arm0-bank", "--results-dir", "--out")
    p("unblind2", "--model", "--plan-dir", "--bank", "--arm0-bank", "--results-dir", "--thresholds", "--journal", "--out")
    p("study2", "--results-dir", "--plan-dir")
    p("missed", "--results-dir", "--plan-dir", "--journal", "--bank-root")
    p("grid", "--results-dir", "--out")
    p("literal_split", "--model", "--plan-dir", "--results-dir", "--bank").add_argument("--arm2-bank")
    a = ap.parse_args(argv)
    if a.cmd == "repair":
        assert a.pooled or (a.model and a.bank), "repair needs --model and --bank, or --pooled"
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
