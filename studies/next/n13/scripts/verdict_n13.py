"""N13 verdict (studies/next/07_PICKS/N13_SPEC.md §5 three numbers, §6.1 gates, §6.3 labels and the verdict, §7 statistics). Laptop only,
banked JSON in. Runs once on the complete banks (§9.5).

    python -m studies.next.n13.scripts.verdict_n13 --banks <dir> --thresholds thresholds.json --setup <setup out dir> --known <r0 metrics.json>
        --controls <r1 controls.json> --out verdict_n13.json [--failing-input GATE]

Exit codes: 0 verdict written (not cancelled); 3 CANCELLED by a failed gate; 5 CANCELLED because a gate's failing input did not fail.
With --failing-input GATE the gate's registered failing input replaces it; exit 0 only when that produces the registered effect
(CANCELLED by that gate, or the stamp for S-BAND, or "no room" for G-GATE-GSM), else 1.
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

from infra.pg import n13_arms as A
from infra.pg import n13_plan as P
from infra.pg import n13_score as S
from studies.next.n13.scripts.freeze_thresholds import find

KS = ("k0.1", "k0.3", "k1", "k3")
GATES_FI = ("G-GRADER", "G-TEMPLATE", "G-INIT", "G-PARTITION", "PC-RANK", "G-KNOWN", "G-HASH", "G-MASK", "G-COMPLETE", "G-GATE", "G-ID",
            "S-BAND", "G-GATE-GSM")
GATES_PASS = ("G-GRADER", "G-TEMPLATE", "G-INIT", "G-PARTITION", "PC-RANK", "G-KNOWN", "G-HASH", "G-MASK", "G-COMPLETE")
ENDPOINTS = ("base", "rl", "zero", "id")


def jl(p):
    return json.loads(pathlib.Path(p).read_text())


def load_bank(banks, arm, bench):
    p = find(banks, arm, bench)
    recs = sorted((json.loads(l) for l in p.read_text().splitlines() if l.strip()), key=lambda r: r["problem_index"])
    return recs, jl(p.parent / "metrics.json")


def present(banks, arm, bench) -> bool:
    return len([p for p in pathlib.Path(banks).rglob("metrics.json") if p.parent.name == bench and p.parent.parent.name == arm]) == 1


def gather_gates(a, T, banks) -> dict:
    setup = pathlib.Path(a.setup)
    gr = jl(setup / "grader.json")
    tp = jl(setup / "template.json")
    C, K = jl(a.controls), jl(a.known)
    g = {"G-GRADER": {"pass": gr["pass"], "failed": gr["rules"][gr["active_rule"]]["failing_input"]["failed"], "value": gr["rules"][gr["active_rule"]]["self_equal"]},
         "G-TEMPLATE": {"pass": tp["pass"], "failed": tp["failing_input"]["failed"]},
         "G-INIT": {"pass": C["G-INIT"]["pass"], "failed": C["G-INIT"]["failing_input"]["failed"], "value": C["G-INIT"]["b"]["difference"]},
         "G-PARTITION": {"pass": C["G-PARTITION"]["pass"], "failed": C["G-PARTITION"]["failing_input"]["failed"]},
         "PC-RANK": {"pass": C["PC-RANK"]["pass"], "failed": C["PC-RANK"]["failing_input"]["failed"], "value": C["PC-RANK"]["recovery"],
                     "failing_value": C["PC-RANK"]["failing_input"]["value"]},
         "G-KNOWN": {"pass": K["G-KNOWN"]["pass"], "failed": K["G-KNOWN"]["failing_input"]["failed"], "value": K["G-KNOWN"]["value"],
                     "failing_value": K["G-KNOWN"]["failing_input"]["value"]},
         "G-HASH": {"pass": C["G-HASH"]["pass"], "failed": C["G-HASH"]["failing_input"]["failed"]},
         "G-MASK": {"pass": C["G-MASK"]["pass"], "failed": C["G-MASK"]["failing_input"]["failed"]},
         "G-GATE": {"pass": T["G-GATE"]["pass"], "failed": T["G-GATE"]["failing_input"]["failed"], "value": T["G-GATE"]["G"],
                    "failing_value": T["G-GATE"]["failing_input"]["value"]},
         "G-ID": {"pass": T["G-ID"]["pass"], "failed": T["G-ID"]["failing_input"]["failed"], "value": [T["G-ID"]["L_minus_Lp"], T["G-ID"]["B_minus_Bp"]]}}
    if "S-BAND" in T:
        g["S-BAND"] = {"pass": T["S-BAND"]["pass"], "failed": T["S-BAND"]["failing_input"]["failed"], "value": T["S-BAND"]["value"],
                       "failing_value": T["S-BAND"]["failing_input"]["value"]}
    if "gsm8k" in T:
        gg = T["gsm8k"]["G-GATE-GSM"]
        g["G-GATE-GSM"] = {"pass": gg["pass"], "failed": gg["failing_input"]["failed"], "value": gg["G"]}
    comp, comp_fi, empty = {}, True, {}
    for r in P.roster():
        for b in ("math500",) + (("gsm8k",) if r["gsm8k"] else ()):
            if not present(banks, r["arm"], b):
                continue
            recs, m = load_bank(banks, r["arm"], b)
            if m["status"] in ("parked", "not_run_condition"):
                continue
            n = a.n_math if b == "math500" else a.n_gsm
            if m["status"] == "screened":
                comp[f"{r['arm']}/{b}"] = True
                continue
            v = S.g_complete(recs, n, r["seed_set"])
            comp[f"{r['arm']}/{b}"] = v["pass"]
            empty[f"{r['arm']}/{b}"] = v["n_empty"]          # A25: reported, not gated
            comp_fi &= not S.g_complete(recs[:-1], n, r["seed_set"])["pass"]
    g["G-COMPLETE"] = {"pass": all(comp.values()) and bool(comp), "failed": comp_fi, "value": comp, "n_empty": empty}
    return g


def label(acc, screened, T) -> str:
    if screened or acc < T["B"] - T["delta"]:
        return "DAMAGED"
    if acc > T["L"] + T["delta"]:
        return "ABOVE"
    if acc >= T["H"]:
        return "REC"
    if acc > T["Q"]:
        return "PART"
    return "NONE"


def arm_rows(banks, T) -> dict:
    rows = {}
    for r in P.roster():
        if not present(banks, r["arm"], "math500"):
            rows[r["arm"]] = {"status": "PENDING"}
            continue
        recs, m = load_bank(banks, r["arm"], "math500")
        if m["status"] in ("parked", "not_run_condition"):
            rows[r["arm"]] = {"status": m["status"]}
            continue
        if m["status"] == "screened":                       # §7: damaged arms stopped by the screen are read on problems 0-19
            recs = [q for q in recs if q["problem_index"] < S.SCREEN["first"]]
        x = [q["score"] for q in recs]
        n = len(x)
        acc = S.acc(x)
        scr = m["status"] == "screened"
        rows[r["arm"]] = {"status": m["status"], "acc": acc, "n": n, "wilson": S.wilson(int(sum(x)), n), "screened": scr,
                          "label": None if r["arm"] in ENDPOINTS else label(acc, scr, T), "cap_hit_share": float(np.mean([q["cap_hit"] for q in recs])),
                          "answer_line_share": float(np.mean([q["mode"] == "answer_line" for q in recs])),
                          "mean_tokens": float(np.mean([q["n_new"] for q in recs])), "scores": {q["problem_index"]: q["score"] for q in recs}}
    return rows


def paired(ra, rb) -> tuple:
    common = sorted(set(ra["scores"]) & set(rb["scores"]))
    return np.array([ra["scores"][i] for i in common]), np.array([rb["scores"][i] for i in common])


def family_tests(rows) -> list:
    t = []
    for k in KS:
        for c in ("dare", "rndE", "rndT"):
            t.append((f"top_{k}", f"{c}_{k}"))
    t += [("mlp", "attn"), ("top_k1", "comp_k1")]
    return t


def comparisons(rows, T) -> dict:
    fam = family_tests(rows)
    avail = [(a, b) for a, b in fam if rows.get(a, {}).get("acc") is not None and rows.get(b, {}).get("acc") is not None]
    ps = []
    res = {}
    for a, b in avail:
        xa, xb = paired(rows[a], rows[b])
        mc = S.mcnemar(xa, xb)
        res[f"{a}|{b}"] = {"diff": rows[a]["acc"] - rows[b]["acc"], "mcnemar_p": mc["p"], "n_paired": len(xa),
                           "bootstrap": S.paired_bootstrap(xa, xb) if len(xa) else None}
        ps.append(mc["p"])
    if ps:
        adj = S.bh(ps)["adjusted"]
        for (a, b), q in zip(avail, adj):
            res[f"{a}|{b}"]["bh_p"] = q
    res["_family_size_available"] = len(avail)
    res["_indeterminate_parked"] = [f"{a}|{b}" for a, b in fam if (a, b) not in avail]
    return res


def beats(cmp, rows, a, b, T):
    c = cmp.get(f"{a}|{b}")
    return bool(c and rows[a]["acc"] - rows[b]["acc"] > T["delta"] and c.get("bh_p", 1.0) < 0.05)


def within(rows, b, a, T):
    return bool(rows[b]["acc"] >= rows[a]["acc"] - T["delta"])


def hyps(rows, cmp, T, C) -> dict:
    lab = lambda x: rows.get(x, {}).get("label")
    have = lambda x: rows.get(x, {}).get("acc") is not None
    k_rec = next((k for k in KS if have(f"top_{k}") and lab(f"top_{k}") in ("REC", "ABOVE")), None)
    H = {"k_rec": k_rec}
    if k_rec is None:
        H["H_local"] = "REFUTED (k_rec NONE)"
    else:
        tk = f"top_{k_rec}"
        ctl = [f"{c}_{k_rec}" for c in ("dare", "rndE", "rndT")]
        if not all(have(x) for x in ctl):
            H["H_local"] = "INDETERMINATE (control arms parked)"
        elif any(within(rows, x, tk, T) for x in ctl):
            H["H_local"] = "REFUTED (a control within delta)"
        elif lab(f"rndT_{k_rec}") == "DAMAGED":
            H["H_local"] = "INDETERMINATE (rndT damaged)"
        elif all(beats(cmp, rows, tk, x, T) for x in ctl):
            H["H_local"] = "SUPPORTED" + (" (control damaged)" if any(lab(x) == "DAMAGED" for x in ctl) else "")
        else:
            H["H_local"] = "INDETERMINATE"
    dw = [k for k in KS if have(f"dare_{k}") and have(f"top_{k}") and lab(f"dare_{k}") in ("REC", "ABOVE") and within(rows, f"dare_{k}", f"top_{k}", T)]
    df = [k for k in KS if have(f"dare_{k}") and have(f"top_{k}") and (lab(f"dare_{k}") not in ("REC", "ABOVE") or beats(cmp, rows, f"top_{k}", f"dare_{k}", T))]
    H["H_diffuse"] = f"SUPPORTED (smallest k {dw[0]})" if dw else ("REFUTED" if len(df) == 4 else "INDETERMINATE")
    la, lm = lab("attn"), lab("mlp")
    if la is None or lm is None:
        H["H_attention"] = "INDETERMINATE (arm parked)"
    elif la in ("REC", "ABOVE") and lm in ("NONE", "DAMAGED"):
        H["H_attention"] = "SUPPORTED"
    elif lm in ("REC", "ABOVE") and la in ("NONE", "DAMAGED"):
        H["H_attention"] = "REFUTED"
    elif lm in ("REC", "ABOVE") and la in ("REC", "ABOVE"):
        H["H_attention"] = "REFUTED (BOTH)"
    else:
        H["H_attention"] = "INDETERMINATE (" + ("interaction" if la == lm == "NONE" else "PARTIAL") + ")"
    art = C["PC-ART"]                                  # N13_SPEC.md A2: the rule must give SUPPORTED on P+ and REFUTED on P-
    jf, jr = C["J_floor"]["k1"], C["J_round"]["k1"]
    if art["rule_P_plus"] != "SUPPORTED" or art["rule_P_minus"] != "REFUTED":
        H["H_artifact"] = f"CANCELLED (PC-ART: rule(P+) = {art['rule_P_plus']}, rule(P-) = {art['rule_P_minus']})"
    else:
        H["H_artifact"] = A.artifact_rule(jf, jr)
    H["H_artifact_values"] = {"J_floor_k1": jf, "J_round_k1": jr, "J_round_P_plus_k1": art["P_plus"]["J_round"]["k1"],
                              "J_sp_k1": C["J_sp"]["k1"], "J_sp_P_plus_k1": art["P_plus"]["J_sp"]["k1"],
                              "rule_P_plus": art["rule_P_plus"], "rule_P_minus": art["rule_P_minus"]}
    nec = {}
    for k in KS:
        c = f"comp_{k}"
        if have(c):
            nec[k] = "NECESSARY" if lab(c) in ("NONE", "PART", "DAMAGED") else ("NOT NECESSARY" if rows[c]["acc"] >= rows["rl"]["acc"] - T["delta"] else "neither")
    H["necessity"] = nec
    return H


def gsm_readout(banks, rows, T) -> dict:
    if "gsm8k" not in T or not T["gsm8k"]["G-GATE-GSM"]["pass"]:
        return {"status": "no room" if "gsm8k" in T else "not run"}
    out = {}
    for c in ("dare", "rndE", "rndT"):
        a, b = "top_k1", f"{c}_k1"
        if not (present(banks, a, "gsm8k") and present(banks, b, "gsm8k")):
            out[f"{a}|{b}"] = "INDETERMINATE (parked)"
            continue
        xa = np.array([r["score"] for r in load_bank(banks, a, "gsm8k")[0]])
        xb = np.array([r["score"] for r in load_bank(banks, b, "gsm8k")[0]])
        lo, hi = S.paired_bootstrap(xa, xb)
        d_m = rows[a]["acc"] - rows[b]["acc"]
        d_g = S.acc(xa) - S.acc(xb)
        same = np.sign(d_m) == np.sign(d_g)
        excl = lo > 0 or hi < 0
        out[f"{a}|{b}"] = {"diff_gsm8k": d_g, "ci": [lo, hi], "label": ("REPLICATES" if same and excl else "CONSISTENT" if same else
                                                                     "CONTRADICTS" if excl else "UNDEFINED (opposite sign, CI includes 0)")}
    return out


def three_numbers(gates, T, C) -> dict:
    tn = {"fit_both_sides": {"G-KNOWN": gates["G-KNOWN"].get("value"), "L_minus_Lprime": T.get("G-ID", {}).get("L_minus_Lp"),
                             "bound_3sqrt2sigma": T.get("G-ID", {}).get("bound"), "id_hash_equal_rl": C.get("G-HASH", {}).get("id", {}).get("pass"),
                             "S-BAND": T["S-BAND"]["value"] if "S-BAND" in T else "pair 1b: no published MATH-500 number"},
          "positive_control": {"G": T.get("G"), "bound_5sqrt2sigma": 5 * math.sqrt(2) * T["sigma"] if "sigma" in T else None,
                               "PC-RANK_recovery": C.get("PC-RANK", {}).get("recovery")},
          "identity_pair": {"L_minus_Lprime": T.get("L", 0) - T.get("L_prime", 0) if "L" in T else None,
                            "B_minus_Bprime": T.get("B", 0) - T.get("B_prime", 0) if "B" in T else None}}
    missing = [f"{g}.{k}" for g, d in tn.items() for k, v in d.items() if v is None]
    tn["missing"] = missing
    return tn


def verdict(a) -> tuple:
    T = jl(a.thresholds)
    C = jl(a.controls)
    gates = gather_gates(a, T, a.banks)
    fi = a.failing_input
    if fi:
        assert fi in gates, (fi, sorted(gates))
        gates[fi] = {**gates[fi], "pass": not gates[fi]["failed"], "substituted": "failing input"}
    V = {"gates": gates, "pair": T["pair"], "failing_input_run": fi, "stamps": []}
    for g in GATES_FI:
        if g in gates and not gates[g]["failed"]:
            V["verdict"] = f"CANCELLED ({g}: its failing input did not fail)"
            return V, 5
    for g in GATES_PASS:
        if not gates[g]["pass"]:
            V["verdict"] = f"CANCELLED ({g})"
            return V, 3
    if not gates["G-GATE"]["pass"]:
        V["verdict"] = "CANCELLED (G-GATE: gain under the noise gate; pair 2 not in these banks)"
        return V, 3
    if not gates["G-ID"]["pass"]:
        V["verdict"] = "CANCELLED (G-ID: identity transplant does not reproduce the released score)"
        return V, 3
    if T["pair"] == "pair1" and not gates["S-BAND"]["pass"]:
        V["stamps"].append("RL side outside the published 3.1 band under this harness")
    if T["pair"] == "pair1b":
        V["stamps"].append("pair 1b: no published MATH-500 number for the RL side")
    tn = three_numbers(gates, T, C)
    V["three_numbers"] = tn
    if tn["missing"]:
        V["verdict"] = f"CANCELLED (three numbers missing: {tn['missing']})"
        return V, 3
    rows = arm_rows(a.banks, T)
    cmp = comparisons(rows, T)
    V["hypotheses"] = hyps(rows, cmp, T, C)
    V["gsm8k"] = gsm_readout(a.banks, rows, T)
    if "G-GATE-GSM" in gates and not gates["G-GATE-GSM"]["pass"]:
        V["stamps"].append("GSM8K: no room")
    V["arms"] = {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in rows.items()}
    V["comparisons"] = cmp
    V["thresholds"] = {k: T[k] for k in ("C", "sigma", "delta", "B", "L", "G", "H", "Q", "B_prime", "L_prime")}
    V["parked"] = [k for k, v in rows.items() if v.get("status") in ("parked", "PENDING")]
    V["verdict"] = "COMPLETE"
    return V, 0


def table(V) -> str:
    L = [f"# N13 verdict: {V['verdict']}", ""]
    if V.get("stamps"):
        L += [f"Stamp: {s}" for s in V["stamps"]] + [""]
    L += ["| gate | pass | failing input failed | value |", "|---|---|---|---|"]
    for g, r in V["gates"].items():
        L.append(f"| {g} | {r['pass']} | {r['failed']} | {json.dumps(r.get('value'))[:60] if 'value' in r else ''} |")
    if "arms" in V:
        L += ["", "| arm | status | acc (%) | Wilson 95% | label | cap-hit share | answer_line share | mean tokens |", "|---|---|---|---|---|---|---|---|"]
        for k, r in V["arms"].items():
            if r.get("acc") is None:
                L.append(f"| {k} | {r['status']} | | | | | | |")
                continue
            w = r["wilson"]
            L.append(f"| {k} | {r['status']} | {r['acc']:.1f} | [{w[0]:.1f}, {w[1]:.1f}] | {r['label'] or '-'} | {r['cap_hit_share']:.2f} | "
                     f"{r['answer_line_share']:.2f} | {r['mean_tokens']:.0f} |")
        L += ["", "| hypothesis | verdict |", "|---|---|"] + [f"| {k} | {v} |" for k, v in V["hypotheses"].items()]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    for f in ("--banks", "--thresholds", "--setup", "--known", "--controls", "--out"):
        ap.add_argument(f, required=True)
    ap.add_argument("--failing-input", default=None)
    ap.add_argument("--n-math", type=int, default=500)
    ap.add_argument("--n-gsm", type=int, default=1319)
    a = ap.parse_args()
    V, code = verdict(a)
    if a.failing_input:
        g = a.failing_input
        ok = (V["verdict"].startswith(f"CANCELLED ({g}") or (g == "S-BAND" and any("outside the published" in s for s in V["stamps"]))
              or (g == "G-GATE-GSM" and "GSM8K: no room" in V["stamps"]))
        print(f"FAILING INPUT {g}: verdict '{V['verdict']}', stamps {V['stamps']}: {'registered effect' if ok else 'NOT the registered effect'}")
        pathlib.Path(a.out).write_text(json.dumps(V, indent=1))
        sys.exit(0 if ok else 1)
    pathlib.Path(a.out).write_text(json.dumps(V, indent=1))
    pathlib.Path(a.out).with_suffix(".md").write_text(table(V))
    print(table(V))
    sys.exit(code)


if __name__ == "__main__":
    main()
