"""N12 offline gate: both sides of every gate the job runs, at every point where it is applied, from banked files only
(the method of M2's amendment A5, studies/next/m2/offline_gate/a5_gate_sides.py). A gate goes to the instance only
if, where the bank holds the quantity it reads, its pass side passes and its registered failing input fails.

Inputs (read-only): M2's job outputs on the laptop (studies/next/m2/banks/score-06/m2/metrics.json, ranks files) and
on S3 (the vectors of score-04, score-05 and score-06, small files); M2's A5 record (a5_gate_sides.json: G0 from the
banked harvest controls, G-REV, the device facts); the committed branch configs (studies/next/m2/offline_gate/hf/);
the bank scan (bank_scan.json); the laptop exemplar arrays (exemplar_a.npz of the 16 grid points); N12's unit table.
Freeze-half units only where an N12 quantity of a unit is read; test-half carriers enter only through their own
banked activations (K0, G-EX), which M2's verdict already used.

    PYTHONPATH=. python studies/next/n12/offline_gate/gate_sides.py --out studies/next/n12/offline_gate/gate_sides.json
"""
import argparse
import io
import json
import pathlib
import sys
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies/next/m2/read"))
import s3range  # noqa: E402

from infra.pg import m1, m2, n12  # noqa: E402

BANK = pathlib.Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
M2O = ROOT / "studies/next/m2/offline_gate"
M2B = ROOT / "studies/next/m2/banks/score-06/m2"
PTS = list(range(5, 16))


def vec(attempt, t, name):
    url = s3range.presign(f"results/next_m2/{attempt}/out/score/m2/vectors/t{t:02d}/{name}.npz")
    with urllib.request.urlopen(url) as r:
        return np.load(io.BytesIO(r.read()), allow_pickle=False)["a"]


def exists(attempt, t, name):
    import subprocess
    r = subprocess.run(["aws", "s3", "ls", f"s3://pg-lineage-769063704696/results/next_m2/{attempt}/out/score/m2/vectors/"
                        f"t{t:02d}/{name}.npz", "--profile", "fuelos", "--region", "us-east-1"], capture_output=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    uj = json.loads((ROOT / "studies/next/n12/units.json").read_text())
    U = uj["units"]
    pl = n12.plan(U)
    jm = json.loads((M2B / "metrics.json").read_text())
    a5 = json.loads((M2O / "a5_gate_sides.json").read_text())
    scan = json.loads((M2O / "bank_scan.json").read_text())
    m2rec = {}
    for h in ("freeze", "test"):
        for r in m2.read_jsonl(M2B / f"ranks_{h}.jsonl"):
            m2rec[(r["unit"], r["role"])] = r
    gates = {}

    # G0: pass side from the banked harvest controls and score-06; u x 1.1 fails on every layer (M2's A5 record)
    g0 = {t: {"harvest_max": a5["G0"][str(t)]["pass_side_harvest"]["max_ratio"],
              "score06_max": jm["gates"][f"t{t:02d}_G0"]["max_ratio"],
              "u11_min_harvest": a5["G0"][str(t)]["u1.1_harvest"]["min_ratio"],
              "u11_min_score06": jm["gates"][f"t{t:02d}_G0"]["failing_input"]["min_ratio"]} for t in PTS}
    gates["G0"] = {"scope": "11 points; harvest controls and score-06, u x 1.1 as the failing input", "banked": True,
                   "points": g0,
                   "pass_side_ok": all(v["harvest_max"] <= 0.01 and v["score06_max"] <= 0.01 for v in g0.values()),
                   "fail_side_ok": all(min(v["u11_min_harvest"], v["u11_min_score06"]) > 0.01 for v in g0.values()),
                   "summary": f"pass max {max(max(v['harvest_max'], v['score06_max']) for v in g0.values()):.5f}; "
                              f"u x 1.1 min {min(min(v['u11_min_harvest'], v['u11_min_score06']) for v in g0.values()):.4f}"}

    # G-REV: registered = banked manifest = HF ref (M2's A5 read); the failing input (forty zeros) equals none
    rv = {t: a5["rev_cfg"]["points"][str(t)]["G_REV"] for t in PTS}
    gates["G_REV"] = {"scope": "11 points; registered revision = banked manifest = HF ref", "banked": True,
                      "pass_side_ok": all(v["pass"] and v["registered"] == m2.POINTS[t][3] for t, v in rv.items()),
                      "fail_side_ok": all(m2.POINTS[t][3] != "0" * 40 for t in PTS), "summary": "11 of 11"}

    # G-CFG: the committed configs of the 11 branches pass with no differing key; model_type renamed fails
    cf = {}
    for t in PTS:
        c = json.loads((M2O / "hf" / f"config_{m2.POINTS[t][2]}.json").read_text())
        g = m1.config_gate(c, m1.REF_CONFIG)
        f = m1.config_gate({**c, "model_type": "olmo2-retrofit"}, m1.REF_CONFIG)
        cf[t] = {"pass": bool(g["pass"] and g["diff_keys"] == []), "failing_pass": bool(f["pass"])}
    gates["G_CFG"] = {"scope": "11 points; committed branch configs", "banked": True,
                      "pass_side_ok": all(v["pass"] for v in cf.values()),
                      "fail_side_ok": not any(v["failing_pass"] for v in cf.values()), "summary": "11 of 11"}

    # K0 at T_14: the 10 carriers; pass side = two L40S runs (score-05, score-06) of their own column bit-identical,
    # so B is identical; failing side = their banked birth-role B
    k0 = sorted((u for u in U if u["kind"] == "real" and u["roles"]["pre"] == n12.KNOWN_T), key=lambda u: u["unit"])
    k0 = k0[:n12.K0_N]
    same, dB = [], []
    for u in k0:
        nm = f"{u['unit'].replace(':', '_')}_pre"
        x5, x6 = vec("score-05", 14, nm), vec("score-06", 14, nm)
        n = u["carrier"][0]
        same.append(bool(np.array_equal(x5[:, n], x6[:, n])))
        dB.append(abs(m2rec[(u["unit"], "birth")]["value"]["B"] - m2rec[(u["unit"], "pre")]["value"]["B"]))
    gates["K0"] = {"scope": "T_14, the 10 carriers (" + ", ".join(u["unit"] for u in k0) + ")", "banked": True,
                   "units": [u["unit"] for u in k0], "score05_eq_score06": same, "failing_abs_dB": dB,
                   "pass_side_ok": all(same), "fail_side_ok": min(dB) > n12.K0_TOL_B,
                   "summary": f"{sum(same)} of 10 bit-identical across two runs; failing min |dB| {min(dB):.3e} "
                              f"against {n12.K0_TOL_B}"}

    # G-M2V: two L40S runs of the same units at T_7-T_15 (score-05) and T_10-T_15 (score-04) against score-06;
    # failing side: the neuron axis rolled; T_5 and T_6 have no second run (assumed, as M2's A5)
    m2v = {}
    for t in PTS:
        here = [(uid, r) for uid, r in pl["points"][t]["roles"]]
        from infra.aws_pg.jobs.n12_score import m2_name
        names = {"freeze": [], "test": []}
        for uid, role in here:
            u = next(x for x in U if x["unit"] == uid)
            nm = m2_name(u, role, t)
            if nm is not None:
                names[u["half"]].append(nm)
        # freeze-half files; at a point with none (T_5, T_6) the test half's, for the aggregate statistic only
        half = "freeze" if names["freeze"] else "test"
        names = sorted(set(names[half]))[:6]
        row = {"n_files": len(names), "half": half}
        x6 = [vec("score-06", t, nm) for nm in names]
        new = np.concatenate([x.ravel() for x in x6])
        row["failing_p99"] = m2.values_gate(new, np.concatenate([np.roll(x, 1, axis=1).ravel() for x in x6]),
                                            m2.TOL["same"])["p99"]
        for att in ("score-05", "score-04"):
            if names and exists(att, t, names[0]):
                xo = np.concatenate([vec(att, t, nm).ravel() for nm in names])
                g = m2.values_gate(new, xo, m2.TOL["same"])
                row[att] = {"p99": g["p99"], "exact_share": g["exact_share"], "n": g["n"]}
        m2v[t] = row
    two = [t for t in PTS if "score-05" in m2v[t] or "score-04" in m2v[t]]
    gates["G_M2V"] = {"scope": "11 points (pass side at " + ", ".join(f"T_{t}" for t in two) +
                      "; T_5, T_6 assumed); up to 6 files per point, freeze half (test half at T_5, T_6, "
                      "aggregate only)", "banked": True, "points": m2v,
                      "pass_side_ok": all(m2v[t][k]["p99"] <= m2.TOL["same"]["p99"] for t in two
                                          for k in ("score-05", "score-04") if k in m2v[t]),
                      "fail_side_ok": all(v["failing_p99"] > m2.TOL["same"]["p99"] for v in m2v.values()),
                      "summary": "exact share " + ", ".join(
                          f"T_{t} {m2v[t].get('score-05', m2v[t].get('score-04'))['exact_share']:.3f}" for t in two) +
                                 f"; failing p99 min {min(v['failing_p99'] for v in m2v.values()):.3f}"}

    # G-EX: N12's entries (every unit's own pole at its own positions, every role at the point) against the bank;
    # failing side = the neighbouring point's bank, for every N12 entry including the dt units; pass side at the
    # gated points (T_5, T_6, T_15) from score-06 (bit-identical on M2's entries); reported at T_7-T_14
    exa = {t: np.load(BANK / "w3/grid" / m2.POINTS[t][1] / "poles/exemplar_a.npz")["a"] for t in range(16)}
    gx = {}
    for t in PTS:
        seen = {}
        for uid, role in pl["points"][t]["roles"]:
            u = next(x for x in U if x["unit"] == uid)
            for r in u["ranks"]:
                seen[(u["row"], r)] = 1
        rows = np.array([k[0] for k in seen])
        cols = np.array([k[1] for k in seen])
        nb = t + 1 if t < 15 else t - 1
        tol = m2.TOL["same"] if m2.POINTS[t][4] == m2.L40S else m2.TOL["ex_cross"]
        f = m2.values_gate(exa[nb][rows, cols], exa[t][rows, cols], tol)
        key = f"t{t:02d}_G_EX"
        jx = jm["gates"].get(key) or jm["reports"].get(key)
        gx[t] = {"n_entries_n12": len(seen), "n_entries_m2": jx["n"], "gated": m2.POINTS[t][4] == m2.L40S,
                 "failing_p99": f["p99"], "failing_times_tol": f["p99"] / tol["p99"], "score06_p99": jx["p99"],
                 "score06_exact_share": jx["exact_share"]}
    gated = [t for t in PTS if gx[t]["gated"]]
    gates["G_EX"] = {"scope": "gated at T_5, T_6, T_15, reported at T_7-T_14 (A3); failing side on N12's entries",
                     "banked": True, "points": gx,
                     "pass_side_ok": all(gx[t]["score06_p99"] <= m2.TOL["same"]["p99"] for t in gated),
                     "fail_side_ok": all(gx[t]["failing_p99"] > (m2.TOL["same"] if gx[t]["gated"] else
                                                                 m2.TOL["ex_cross"])["p99"] for t in PTS),
                     "summary": f"gated pass p99 {[round(gx[t]['score06_p99'], 4) for t in gated]}; failing "
                                f"{min(gx[t]['failing_times_tol'] for t in PTS):.0f}-"
                                f"{max(gx[t]['failing_times_tol'] for t in PTS):.0f} times the tolerance"}

    # G-TOPK: score-06 both sides at all 11 points (M2's instrument; N12 adds the dt positions)
    tk = {t: {"p99": jm["gates"][f"t{t:02d}_G_TOPK"]["p99"], "n": jm["gates"][f"t{t:02d}_G_TOPK"]["n"],
              "failing_p99": jm["gates"][f"t{t:02d}_G_TOPK"]["failing_input"]["p99"],
              "tol": m2.tol_for(t, m2.L40S)["p99"]} for t in PTS}
    gates["G_TOPK"] = {"scope": "11 points; score-06 (M2's entries)", "banked": True, "points": tk,
                       "pass_side_ok": all(v["p99"] <= v["tol"] for v in tk.values()),
                       "fail_side_ok": all(v["failing_p99"] > v["tol"] for v in tk.values()),
                       "summary": f"pass p99 max {max(v['p99'] for v in tk.values()):.4f}; failing p99 min "
                                  f"{min(v['failing_p99'] for v in tk.values()):.3f}"}

    # I-ID at T_13: score-04, -05, -06 bit-identical, 0 flips; failing side M2's birth-role vectors at T_14
    ii = jm["gates"]["I_ID"]
    gates["I_ID"] = {"scope": "T_13; score-04, score-05, score-06", "banked": True,
                     "pass_side_ok": ii["pass"] and ii["exact_share"] == 1.0 and ii["carrier_top10_flips_B"] == 0 and
                     all(a5["I_ID"][k]["pass"] for k in ("score-04", "score-05")),
                     "fail_side_ok": not ii["failing_input"]["pass"],
                     "summary": f"exact share {ii['exact_share']}, failing p99 {ii['failing_input']['p99']:.3f}"}

    # degenerate poles: the scan covers all 32 layers at T_5-T_15; no unit's own pole is degenerate at its roles
    deg = {int(t): set(v["degenerate_all_layers"]) for t, v in scan["points"].items()}
    own = [(u["unit"], r, t) for u in U for r, t in u["roles"].items() if u["pole"] in deg[t]]
    in_layers = {t: sorted(p for p in deg[t] if m2.parse_pole(p)[0] in
                           {lyr for w in pl["points"][t]["need"].values() for lyr, _, _ in w}) for t in PTS}
    gates["DEGENERATE"] = {"scope": "11 points, every layer the plan reads (bank_scan.json)", "banked": True,
                           "in_plan_layers": in_layers, "own_pole_degenerate": own,
                           "pass_side_ok": own == [], "fail_side_ok": True,
                           "summary": "; ".join(f"T_{t} {v}" for t, v in in_layers.items() if v) + "; no own pole"}

    # usable units per half and kind
    cnt = uj["n_units"]
    gates["USABLE"] = {"scope": "both halves; real, pb, dt", "banked": True, "counts": cnt,
                       "pass_side_ok": min(cnt.values()) >= n12.N_MIN, "fail_side_ok": 30 < n12.N_MIN,
                       "summary": json.dumps(cnt)}

    # the movement positive control, freeze half (M2's banked pre and birth ranks of the carriers)
    fz = [u for u in U if u["kind"] == "real" and u["half"] == "freeze"]
    mv = [np.log10(m2rec[(u["unit"], "pre")]["rank"]["B"]) - np.log10(m2rec[(u["unit"], "birth")]["rank"]["B"])
          for u in fz]
    bm = n12.boot_mean(mv)
    gates["MOVEMENT"] = {"scope": "freeze half (82); the test half is read only at the verdict", "banked": True,
                         "move": bm, "identity": n12.boot_mean(np.zeros(len(mv))),
                         "pass_side_ok": bm["lo"] > 0, "fail_side_ok": not n12.boot_mean(np.zeros(len(mv)))["lo"] > 0,
                         "summary": f"mean log10 r0 - log10 r1 {bm['mean']:.3f} [{bm['lo']:.3f}, {bm['hi']:.3f}]; "
                                    "identity 0"}

    # the twins' construction control at T_15 (M2's pb end records, the same operator as dt), freeze half
    pe = [m2rec[(u["unit"], "end")] for u in U if u["kind"] == "pb" and u["half"] == "freeze"]
    hit = np.mean([r["rank"]["B"] <= m2.TOP for r in pe])
    opp = np.mean([r["rank_opposite"]["B"] <= m2.TOP for r in pe])
    gates["CONSTRUCTION_TWIN"] = {"scope": "T_15, freeze half, M2's twins under B (the dt operator)", "banked": True,
                                  "n": len(pe), "hits": float(hit), "opposite": float(opp),
                                  "pass_side_ok": hit >= m2.CTRL_MIN, "fail_side_ok": opp < m2.CTRL_MIN,
                                  "summary": f"{hit:.3f} of {len(pe)}; opposite pole {opp:.3f}"}

    # not bank-checkable: no gradient or ablation of this model was ever banked
    for name, why in (("G_ABL_ID", "no ablation banked; dry run only (tiny world: clean rows exact, the planted "
                                   "clean-row ablation stops the job)"),
                      ("I_GRAD", "no gradient banked; dry run only (tiny world: the reversed pass and the planted "
                                 "noise)"),
                      ("G_VAL", "no ablation or gradient banked; dry run only (fp32 finite differences, the tiny "
                                "world, planted records)")):
        gates[name] = {"scope": why, "banked": False, "pass_side_ok": None, "fail_side_ok": None}
    res = {"what": __doc__.split("\n")[0], "gates": {k: v for k, v in gates.items() if v["banked"]},
           "not_banked": {k: v for k, v in gates.items() if not v["banked"]}}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    for k, v in gates.items():
        print(f"{k:18s} pass {v['pass_side_ok']} fail {v['fail_side_ok']}  {v.get('summary', v['scope'])}")


if __name__ == "__main__":
    main()
