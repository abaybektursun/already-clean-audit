"""NEURONOFF exploratory deviations E1 and E2, dated 2026-10-25. Analysis only, banked JSON only, not the headline.

    python -m infra.pg.neuron_off_exploratory e1 --cell-dir bank/off/<cell>/1/out \
        --thresholds studies/neuronoff/results/thresholds_<M>.json --plan studies/neuronoff/plan/plan_<M>.json \
        --out studies/neuronoff/results/<cell>/1/exploratory_E1.json

Rule E1 (coordinator, 2026-10-25), exactly: the per-pole G-pc gate is replaced by a per-model pooled positive control:
mean over ALL held-out positions of all Arm A poles of ΔNLL_layer must be >= 0.05 nats, and its registered failing
input (the same pooled statistic computed on the identity readout, which must be < 0.05) must fail; floor 0.05,
ceiling: pooled mean <= the known-number layer-0 pooled value (+1.71 G4 / +1.85 Q4B), cancelled above. Every other
rule (B1-B12, δ freeze from the same thresholds files, the labels, the subset rows, R-b/R-c room checks) is unchanged.
Poles are no longer cancelled individually by G-pc; poles cancelled by G-fit, G-ident or not-measured stay cancelled.

How the banked data meets the rule (stated here for ratification):
 E1.a "All held-out positions": the job banked ΔNLL_layer only at each pole's first 10 held-out contexts
      (poles/<stem>.json positive_control.contexts); the pooled mean is over those positions of every measured Arm A
      pole of the plan.
 E1.b The identity readout is banked at one position per pole (positive_control.failing_input.mean_dnll, the first
      held-out context); the failing input is the mean over those positions.
 E1.c The 95% interval of the pooled mean: percentile bootstrap over positions (10,000, default_rng(20261025)).
 E1.d The known-number layer-0 pooled value = controls.json known_number.failing_input.value - known_number.value.
 E1.e δ, δ_c, δ_e, the null differences and the valid null-pair count are taken from the thresholds file as frozen
      ("result"). The frozen null-pair validity applied the per-pole G-pc gate that E1 removes, so the file also holds
      "regated_nulls": the same freeze rule (δ = median |d| over valid null pairs, neuron_off_analysis B1-B12) with the
      null pairs re-gated under E1 (G-ident, G-fit, not measured) and the labels that follow. The rule text does not
      say which of the two E1 means.
 E1.f Arm B is not part of E1.
 E1.g The pairs file and the parent verdict file are located through the thresholds file's recorded inputs (paths
      relative to its repository; sha256 checked).

Rule E2 (coordinator, 2026-10-25; chosen after unblinding), exactly: per model, the median over valid E1 pairs of
D_k / mean(|own_one_k|, |own_ctrl_k|) with a 95% bootstrap interval, plus the median absolute own-context ΔNLL per side
(as the result review's §3 table).
 E2.a Valid E1 pairs: both poles measured and passing G-ident and G-fit (E1's pair rule; no per-pole G-pc).
 E2.b A pair whose denominator is 0 is left out and counted. Interval: percentile bootstrap over pairs (10,000,
      default_rng(20261025)).
 E2.c Per side: median and interquartile range of own_dnll (the review's §3 table: absolute scores, signed), and the
      median of |own_dnll|.

    python -m infra.pg.neuron_off_exploratory e2 --cell-dir ... --thresholds ... --plan ... --out .../exploratory_E2.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from infra.pg import neuron_off_analysis as A

STATUS = "EXPLORATORY E1 2026-10-25, not the headline"
E2_STATUS = "EXPLORATORY E2 2026-10-25, chosen after unblinding, not the headline"
PC_FLOOR = 0.05


def pole_reasons_e1(rec) -> list:
    """G-ident and G-fit as B4 (recomputed, checked against the job); G-pc no longer cancels a pole."""
    return [r for r in A.pole_gates(rec)["reasons"] if r != "G-pc"]


def pair_reasons_e1(ids, poles) -> list:
    out = []
    for q in ids:
        out += [f"{q}: not measured"] if poles.get(q) is None else [f"{q}: {r}" for r in pole_reasons_e1(poles[q])]
    return out


def pooled_control(poles: dict, arm_a_ids: list, controls: dict) -> dict:
    measured = [q for q in arm_a_ids if poles.get(q) is not None]
    pos = [c["dnll"] for q in measured for c in poles[q]["positive_control"]["contexts"]]
    ident = [poles[q]["positive_control"]["failing_input"]["mean_dnll"] for q in measured]
    kn = controls["known_number"]
    ceiling = kn["failing_input"]["value"] - kn["value"]
    s = A.mean_ci(pos)
    fi = float(np.mean(ident)) if ident else float("nan")
    fails = A.num(fi) and fi < PC_FLOOR
    if not A.num(s["mean"]) or s["mean"] < PC_FLOOR:
        status = "below floor"
    elif s["mean"] > ceiling:
        status = "above ceiling"
    elif not fails:
        status = "failing input did not fail"
    else:
        status = "pass"
    return {"n_poles": len(measured), "n_positions": len(pos), "mean_dnll_layer": s["mean"], "lo": s["lo"], "hi": s["hi"],
            "floor": PC_FLOOR, "ceiling": ceiling,
            "failing_input": {"readout": "identity", "n_positions": len(ident), "mean_dnll": fi, "fails": fails},
            "status": status, "pass": status == "pass"}


def null_regated(pairs: dict, poles: dict) -> dict:
    valid = [x for x in pairs["null_pairs"] if not pair_reasons_e1((x["ctrl"], x["extra"]), poles)]
    d = {"own": [A.own(poles[x["ctrl"]]) - A.own(poles[x["extra"]]) for x in valid],
         "coll": [A.coll(poles[x["ctrl"]]) - A.coll(poles[x["extra"]]) for x in valid],
         "e": [A.own_e(poles[x["ctrl"]]) - A.own_e(poles[x["extra"]]) for x in valid]}
    dl = {k: float(np.median(np.abs(v))) if v else float("nan") for k, v in d.items()}
    return {"n_null_pairs": len(pairs["null_pairs"]), "n_null_valid": len(valid), "delta_own": dl["own"],
            "delta_coll": dl["coll"], "delta_e": dl["e"], "null_diffs_own": d["own"], "null_diffs_coll": d["coll"],
            "null_diffs_e": d["e"]}


def arm_a(pairs: dict, poles: dict, metrics: dict, ta: dict, pc: dict, verdict_rows: dict) -> dict:
    """§7.1-§7.4 Arm A with the E1 gate, every other rule as neuron_off_analysis.unblind_compute."""
    valid, cancelled = [], []
    for x in pairs["pairs"]:
        reasons = pair_reasons_e1((x["one"], x["ctrl"]), poles)
        (cancelled.append({"pair_id": x["pair_id"], "reasons": reasons}) if reasons else valid.append(x))
    D = [A.own(poles[x["one"]]) - A.own(poles[x["ctrl"]]) for x in valid]
    Dc = [A.coll(poles[x["one"]]) - A.coll(poles[x["ctrl"]]) for x in valid]
    De = [A.own_e(poles[x["one"]]) - A.own_e(poles[x["ctrl"]]) for x in valid]
    n, nn = len(valid), ta["n_null_valid"]
    st = {"own": A.median_ci(D), "coll": A.median_ci(Dc), "energy": A.median_ci(De)}
    dl = {"own": ta["delta_own"], "coll": ta["delta_coll"], "energy": ta["delta_e"]}
    rm = {"own": A.room(ta["null_diffs_own"], dl["own"], n), "coll": A.room(ta["null_diffs_coll"], dl["coll"], n),
          "energy": A.room(ta["null_diffs_e"], dl["energy"], n)}
    logs = [math.log(A.energy(poles[x["one"]]) / A.energy(poles[x["ctrl"]])) for x in valid]
    B = float(np.median(logs)) if logs else float("nan")
    bal = A.num(B) and -A.BAL <= B <= A.BAL
    reasons = A.model_gate_reasons(metrics)
    if not A.num(B):
        reasons.append("G-bal: B is not a number")
    if not pc["pass"]:
        reasons.append(f"E1 pooled positive control: {pc['status']}")
    if not (n >= A.GN_PAIRS and nn >= A.GN_NULL):
        reasons.append(f"underpowered after gates ({n} valid pairs, {nn} valid null pairs)")

    def lab(kind, fn):
        if reasons:
            return A.cancel_stamp(reasons)
        if not rm[kind]["R_a"]:
            return A.cancel_stamp(["no range"])
        return fn(st[kind], dl[kind], rm[kind])
    labels = {"own": lab("own", A.label_own), "coll": lab("coll", A.label_coll), "energy": lab("energy", A.label_own)}
    head, rule = A.headline(labels["own"], labels["energy"], st["energy"], bal)
    pred = [x for x in valid if A.own_predictable(poles[x["one"]]) is not None
            and A.own_predictable(poles[x["ctrl"]]) is not None]
    Dp = [A.own_predictable(poles[x["one"]]) - A.own_predictable(poles[x["ctrl"]]) for x in pred]
    ldr = [x["log_density_ratio"] for x in valid]
    strata = {}
    for name, key in (("explainer", lambda x: x["explainer_one"]), ("tercile", lambda x: verdict_rows[x["one"]]["tercile"]),
                      ("sign", lambda x: poles[x["one"]]["sign"])):
        groups = {}
        for x, d in zip(valid, D):
            groups.setdefault(str(key(x)), []).append(d)
        strata[name] = {k: A.median_ci(v) for k, v in sorted(groups.items())}
    return {"n_pairs": len(pairs["pairs"]), "n_valid": n, "cancelled": cancelled, "n_null_valid": nn,
            "model_reasons": reasons, "stats": st, "delta": dl["own"], "delta_c": dl["coll"], "delta_e": dl["energy"],
            "room": rm, "stamps": {k: A.stamps(v) for k, v in rm.items()},
            "label_own": labels["own"], "label_coll": labels["coll"], "label_energy": labels["energy"],
            "headline_own": head, "headline_rule": rule, "refutations": A.refutations(st["own"]), "B": B, "G_bal_pass": bal,
            "secondary": {
                "energy_normalized": {"stats": st["energy"], "label": labels["energy"]},
                "class_stable": A.subset_row(valid, D, dl["own"], ta["null_diffs_own"], lambda x: x["class_stable"],
                                             reasons),
                "predictable_target": {"n_pairs": len(pred), "n_pairs_dropped": n - len(pred),
                                       **A.subset_row(pred, Dp, dl["own"], ta["null_diffs_own"], lambda x: True,
                                                      reasons)},
                "density_covariate": {"spearman_D_vs_log_density_ratio": A.spearman(D, ldr),
                                      "abs_ldr_le_log_1_25": A.median_ci([d for d, r in zip(D, ldr) if abs(r) <= A.LDR_MAX])},
                "strata": strata}}


def e1_compute(pairs: dict, poles: dict, metrics: dict, controls: dict, thresholds: dict, verdict_rows: dict,
               arm_a_ids: list) -> dict:
    pc = pooled_control(poles, arm_a_ids, controls)
    rg = null_regated(pairs, poles)
    return {"status": STATUS, "model_key": pairs["model_key"], "pooled_positive_control": pc,
            "result": arm_a(pairs, poles, metrics, thresholds["arm_a"], pc, verdict_rows),
            "regated_nulls": {**{k: rg[k] for k in ("n_null_pairs", "n_null_valid", "delta_own", "delta_coll", "delta_e")},
                              "result": arm_a(pairs, poles, metrics, rg, pc, verdict_rows)}}


def side_summary(x: list) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"n": len(x), "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75)), "median_abs": float(np.median(np.abs(x)))}


def e2_compute(pairs: dict, poles: dict) -> dict:
    valid = [x for x in pairs["pairs"] if not pair_reasons_e1((x["one"], x["ctrl"]), poles)]
    one = [A.own(poles[x["one"]]) for x in valid]
    ctrl = [A.own(poles[x["ctrl"]]) for x in valid]
    ratios = [(a - b) / ((abs(a) + abs(b)) / 2) for a, b in zip(one, ctrl) if abs(a) + abs(b) > 0]
    return {"status": E2_STATUS, "model_key": pairs["model_key"], "n_valid_pairs": len(valid),
            "n_zero_denominator": len(valid) - len(ratios),
            "relative_D": A.median_ci(ratios), "own_dnll_one": side_summary(one), "own_dnll_ctrl": side_summary(ctrl)}


def load_inputs(a):
    """(pairs, plan, verdict rows, bank, Arm A ids, inputs record) for the e1 and e2 commands (E1.g)."""
    thr_path = pathlib.Path(a.thresholds).resolve()
    thresholds = json.loads(thr_path.read_text())
    root = pathlib.Path(A.git(thr_path.parent, "rev-parse", "--show-toplevel"))
    pkey = next(k for k in thresholds["inputs"] if k.endswith(f"pairs_{thresholds['model_key']}.json"))
    ppath = root / pkey
    assert A.sha256(ppath) == thresholds["inputs"][pkey], f"{ppath} changed since the freeze"
    pairs = json.loads(ppath.read_text())
    plan = json.loads(pathlib.Path(a.plan).read_text())
    assert pairs["model_key"] == plan["model_key"] == thresholds["model_key"], "files name different models"
    bank = A.Bank(a.cell_dir)
    arm_a_ids = [q for q in plan["order"] if plan["poles"][q]["arm"] == "A"]
    inputs = {str(thr_path): A.sha256(thr_path), str(ppath): A.sha256(ppath), a.plan: A.sha256(a.plan)}
    return thresholds, pairs, plan, bank, arm_a_ids, inputs


def cmd_e2(a):
    _, pairs, _, bank, arm_a_ids, inputs = load_inputs(a)
    poles = {q: bank.pole(q) for q in arm_a_ids}
    out = e2_compute(pairs, poles)
    out = {"status": E2_STATUS, "rule": __doc__.split("Rule E2")[1].split("    python -m")[0].strip(),
           "created_utc": A.utc(), "inputs": {**inputs, **bank.read}, **out}
    A.write_json(a.out, out)
    print("E2", out["model_key"], json.dumps(A.finite({k: out[k] for k in ("n_valid_pairs", "n_zero_denominator", "relative_D",
                                                                          "own_dnll_one", "own_dnll_ctrl")})))


def cmd_e1(a):
    thr_path = pathlib.Path(a.thresholds).resolve()
    thresholds = json.loads(thr_path.read_text())
    root = pathlib.Path(A.git(thr_path.parent, "rev-parse", "--show-toplevel"))
    pkey = next(k for k in thresholds["inputs"] if k.endswith(f"pairs_{thresholds['model_key']}.json"))
    ppath = root / pkey
    assert A.sha256(ppath) == thresholds["inputs"][pkey], f"{ppath} changed since the freeze"
    pairs = json.loads(ppath.read_text())
    plan = json.loads(pathlib.Path(a.plan).read_text())
    assert pairs["model_key"] == plan["model_key"] == thresholds["model_key"], "files name different models"
    vp = pairs["inputs"]["verdict_poles"]
    assert A.sha256(vp["path"]) == vp["sha256"], f"{vp['path']} changed since the pairs were built"
    verdict_rows = {r["pole"]: r for r in A.jsonl(pathlib.Path(vp["path"]).read_text())}
    bank = A.Bank(a.cell_dir)
    arm_a_ids = [q for q in plan["order"] if plan["poles"][q]["arm"] == "A"]
    poles = {q: bank.pole(q) for q in arm_a_ids}
    out = e1_compute(pairs, poles, bank.json("metrics.json"), bank.json("controls.json"), thresholds, verdict_rows,
                     arm_a_ids)
    out = {"status": STATUS, "rule": __doc__.split("Rule E1")[1].split("How the banked")[0].strip(),
           "created_utc": A.utc(), "inputs": {str(thr_path): A.sha256(thr_path), str(ppath): A.sha256(ppath),
                                              a.plan: A.sha256(a.plan), vp["path"]: vp["sha256"], **bank.read}, **out}
    A.write_json(a.out, out)
    pc, r, g = out["pooled_positive_control"], out["result"], out["regated_nulls"]

    def row(x):
        s = x["secondary"]
        return {"n_valid": x["n_valid"], "n_null_valid": x["n_null_valid"], "delta": x["delta"], "M_own": x["stats"]["own"],
                "label_own": x["label_own"], "headline": x["headline_own"], "class_stable": s["class_stable"],
                "p_target_lt_0_9": {k: s["predictable_target"][k] for k in ("n_pairs", "stats", "label")},
                "energy": s["energy_normalized"]}
    print("E1", out["model_key"], json.dumps(A.finite({"pooled_pc": {k: pc[k] for k in ("mean_dnll_layer", "lo", "hi", "n_positions",
                                                                                         "ceiling", "status")},
                                                       "pc_failing_input": pc["failing_input"],
                                                       "frozen_thresholds": row(r),
                                                       "regated_nulls": row(g["result"])}), ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neuron_off_exploratory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("e1", "e2"):
        e = sub.add_parser(name)
        for k in ("--cell-dir", "--thresholds", "--plan", "--out"):
            e.add_argument(k, required=True)
    a = ap.parse_args(argv)
    {"e1": cmd_e1, "e2": cmd_e2}[a.cmd](a)


if __name__ == "__main__":
    main()
