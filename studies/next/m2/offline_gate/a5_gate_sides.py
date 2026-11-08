"""Offline gate, amendment A5 (M2_SPEC.md §3.8 "A5"): both sides of every gate the job runs, at every point T_5-T_15,
from banked files only. The pass side is what a correct run gives; the failing side is the gate's registered failing
input. A gate goes to the instance only if, where the bank holds the quantity it reads, the pass side passes and the
failing side fails at every point where the gate is applied. Where the bank does not hold it, the output says what the
gate assumes.

Reads: the 11 banked harvest controls.json and manifest.json files (S3, about 3 KB each); the HF refs of the repo (one
API read, no weights); the five attempts' metrics.json (studies/next/m2/banks/score-0N/); the laptop exemplar arrays and
the T_14 L40S repeat; the committed branch configs (offline_gate/hf/); and by byte-range reads of the banked topk files
only the members `tok` and `a` of layer 3 at T_5 and T_6, of the five layers T_7 scores (the positive control of the
G-TOPK count) and `tok` of the 26 layers that hold freeze-half births at T_15 (the construction control under A, freeze
half only). G-TOPK is written as aggregates only (count, median and p99), the same quantities the job writes to
metrics.json; no per-pole value of a test-half unit is computed or written. Nothing of score-01 to score-05 beyond
metrics.json is read.

    PYTHONPATH=. python studies/next/m2/offline_gate/a5_gate_sides.py --bank <BANK/neurons> \
        --births studies/next/m2/births.json --repeat <T_14 L40S repeat exemplar_a.npz> \
        --out studies/next/m2/offline_gate/a5_gate_sides.json
"""
import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(pathlib.Path.cwd()))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import bank_scan as S  # noqa: E402
from infra.pg import m1, m2  # noqa: E402
from infra.pg.neurons_harvest import G0_TOL, WRONG_U_SCALE  # noqa: E402

BANKS = pathlib.Path("studies/next/m2/banks")
HF = pathlib.Path("studies/next/m2/offline_gate/hf")
G0_DIAG = pathlib.Path("studies/next/m2/diag/score05_g0_fail.json")
PTS = list(range(5, 16))
AWS = ["--profile", "fuelos", "--region", "us-east-1"]
REFS_URL = f"https://huggingface.co/api/models/{m2.REPO}/refs"
EPS = 1e-5


def s3_json(key: str, d: pathlib.Path, name: str) -> tuple[dict, str]:
    f = d / name
    subprocess.run(["aws", "s3", "cp", f"s3://{S.BUCKET}/{key}", str(f), *AWS, "--only-show-errors"], check=True)
    return json.loads(f.read_text()), hashlib.sha256(f.read_bytes()).hexdigest()


def read_members(t: int, kind: str, lyr: int, names: tuple, size: int) -> dict:
    key = f"{S.ROOT}/{m2.POINTS[t][1]}/{kind}/layer_{lyr:02d}.npz"
    z, rf, _ = S.open_zip(key, size, names, ())
    return {n: np.load(io.BytesIO(z.read(n + ".npy"))) for n in names}


def jobs_gates() -> dict:
    return {a.name: json.loads((a / "metrics.json").read_text()) for a in sorted(BANKS.glob("score-0*"))}


# ------------------------------------------------------------------------------------------------------------- G0
def g0(d: pathlib.Path, jobs: dict) -> dict:
    """Pass side: the harvest's own G0 (first batch, windows 0-7) and the jobs' G0 (the plan's first 8 windows).
    Failing side under A5: u scaled by 1.1 (the harvest's failing_input_2), every layer. The activation swap (the
    freeze-1f failing input, reported from A5 on): harvest and jobs. The bound: for any input, per layer,
    ratio_s lies in [(s - 1) - s * r0, (s - 1) + s * r0], r0 the gate's own ratio on the same windows (triangle
    inequality on act(g) * s * u - a = s (act(g) * u - a) + (s - 1) a); so wherever the gate passes (r0 <= 0.01),
    u x 1.1 gives at least 0.1 - 0.011 = 0.089 on every layer."""
    reg = json.loads(G0_DIAG.read_text())["banked_harvest"]
    s = WRONG_U_SCALE
    out = {}
    for t in PTS:
        c, sha = s3_json(f"{S.ROOT}/{m2.POINTS[t][1]}/controls.json", d, f"controls_t{t:02d}.json")
        assert sha == reg[f"{t:02d}"]["sha256"], (t, sha)
        c = c["G0"]
        assert c["failing_input_2"]["u_scale"] == s and c["tol"] == G0_TOL
        r0, r1, r2 = (np.array(x, np.float64) for x in (c["ratio"], c["failing_input"]["ratio"],
                                                         c["failing_input_2"]["ratio"]))
        lo, hi = (s - 1) - s * r0, (s - 1) + s * r0
        row = {"controls_sha256": sha, "harvest_windows": f"0-{c['n_windows'] - 1}",
               "pass_side_harvest": {"max_ratio": float(r0.max()), "pass": bool((r0 <= G0_TOL).all())},
               "u1.1_harvest": {"min_ratio": float(r2.min()), "max_ratio": float(r2.max()),
                                "n_layers_over_tol": int((r2 > G0_TOL).sum()), "fails": bool((r2 > G0_TOL).any()),
                                "fails_every_layer": bool((r2 > G0_TOL).all())},
               "u1.1_bound": {"n_layers": int(r0.size), "n_within": int(((r2 >= lo - EPS) & (r2 <= hi + EPS)).sum()),
                              "lowest_lower_bound": float(lo.min())},
               "act_swap_harvest": {"max_ratio": float(r1.max()), "n_layers_over_tol": int((r1 > G0_TOL).sum()),
                                    "would_pass": bool((r1 <= G0_TOL).all())},
               "jobs": {}}
        for a, m in jobs.items():
            g = m["gates"].get(f"t{t:02d}_G0")
            if g is not None:
                row["jobs"][a] = {"max_ratio": g["max_ratio"], "pass": g["pass"],
                                  "act_swap_max_ratio": g["failing_input"]["max_ratio"],
                                  "act_swap_would_pass": g["failing_input"]["pass"],
                                  "u1.1_lower_bound": (s - 1) - s * g["max_ratio"]}
        out[t] = row
    return out


# ------------------------------------------------------------------------------------------------- G-REV, G-CFG
def rev_cfg(d: pathlib.Path, jobs: dict) -> dict:
    body = urllib.request.urlopen(REFS_URL, timeout=60).read()
    refs = {b["name"]: b["targetCommit"] for b in json.loads(body)["branches"]}
    out = {"hf_refs": {"url": REFS_URL, "sha256": hashlib.sha256(body).hexdigest()}, "points": {}}
    for t in PTS:
        man, sha = s3_json(f"{S.ROOT}/{m2.POINTS[t][1]}/manifest.json", d, f"manifest_t{t:02d}.json")
        _, _, branch, rev, dev = m2.POINTS[t]
        c = json.loads((HF / f"config_{branch}.json").read_text())
        g = m1.config_gate(c, m1.REF_CONFIG)
        gf = m1.config_gate({**c, "model_type": "olmo2-retrofit"}, m1.REF_CONFIG)
        jr = {a: m["gates"][f"t{t:02d}_G_REV"]["commit"] for a, m in jobs.items() if f"t{t:02d}_G_REV" in m["gates"]}
        out["points"][t] = {
            "G_REV": {"registered": rev, "manifest": man["revision"], "hf_ref_today": refs.get(branch),
                      "pass": rev == man["revision"] == refs.get(branch), "failing_input": "0" * 40,
                      "failing_input_fails": rev != "0" * 40, "jobs_commit_equal": {a: c == rev for a, c in jr.items()}},
            "G_CFG": {"pass": bool(g["pass"] and g["diff_keys"] == []), "failing_input_fails": not gf["pass"]},
            "manifest": {"sha256": sha, "device_name": man["device_name"], "registered_device": dev,
                         "torch": man["torch"], "transformers": man["transformers"], "pg_git_sha": man["pg_git_sha"],
                         "out": man["args"]["out"], "batch": man["batch"], "window": man["window"],
                         "dtype": man["dtype"], "attn": man["attn"]}}
    return out


# --------------------------------------------------------------------------------------------- K0, G-EX, I-ID
def known_number(bank: pathlib.Path, bj: dict, repeat: pathlib.Path, jobs: dict) -> dict:
    grid = bank / "w3" / "grid"
    ex = {t: np.load(grid / m2.POINTS[t][1] / "poles" / "exemplar_a.npz")["a"] for t in range(16)}
    rep = np.load(repeat)["a"]
    assert m2.sha256_file(repeat) == m2.REPEAT_T14["sha256"]
    us = sorted((u for u in m2.units(bj["births"]) if u["kind"] == "real"), key=lambda u: u["unit"])
    k0 = []
    for u in us:
        if u["pole"] not in {x["pole"] for x in k0}:
            k0.append(u)
        if len(k0) == 10:
            break
    rows = np.array([u["row"] for u in k0 for _ in u["ranks"]])
    cols = np.array([r for u in k0 for r in u["ranks"]])
    kf = m2.values_gate(ex[15][rows, cols], ex[14][rows, cols], m2.TOL["same"], 10, len(set(rows.tolist())))
    out = {"K0": {"n": kf["n"], "failing_input_T14": {"median": kf["median"], "p99": kf["p99"], "fails": not kf["pass"]},
                  "jobs": {a: {"p99": m["gates"]["K0"]["p99"], "exact_share": m["gates"]["K0"]["exact_share"],
                               "pass": m["gates"]["K0"]["pass"]} for a, m in jobs.items() if "K0" in m["gates"]}},
           "G_EX": {}}
    pl = m2.plan(bj["births"])
    um = {u["unit"]: u for u in pl["units"]}
    for t, p in pl["points"].items():
        ent = sorted({(um[uid]["row"], r) for uid, _ in p["roles"] for r in um[uid]["ranks"]})
        rr, cc = np.array([x[0] for x in ent]), np.array([x[1] for x in ent])
        nb = t + 1 if t < 15 else t - 1
        tol = m2.tol_ex(t, m2.L40S)
        f = m2.values_gate(ex[t][rr, cc], ex[nb][rr, cc], tol)
        row = {"gated": m2.ex_is_gate(t, m2.L40S), "tol": tol["p99"], "abs_min": tol.get("abs_min"), "n": len(ent),
               "n_poles": len(set(rr.tolist())), "n_gated": f["n_gated"],
               "failing_input": {"neighbour": nb, "median": f["median"], "p99": f["p99"], "fails": not f["pass"],
                                 "times_tol": f["p99"] / tol["p99"]}, "jobs": {}}
        if t == 14:
            g = m2.values_gate(rep[rr, cc], ex[14][rr, cc], tol)
            row["pass_side_repeat"] = {"p99": g["p99"], "n_gated": g["n_gated"], "pass": g["pass"]}
        for a, m in jobs.items():
            g = m["gates"].get(f"t{t:02d}_G_EX") or m.get("reports", {}).get(f"t{t:02d}_G_EX")
            if g is not None:
                row["jobs"][a] = {"p99": g["p99"], "pass": g["pass"], "exact_share": g["exact_share"],
                                  "failing_p99": g["failing_input"]["p99"]}
        out["G_EX"][t] = row
    out["I_ID"] = {a: {k: m["gates"]["I_ID"][k] for k in ("n", "p99", "exact_share", "carrier_top10_flips_B", "pass")}
                   | {"failing_p99": m["gates"]["I_ID"]["failing_input"]["p99"],
                      "failing_pass": m["gates"]["I_ID"]["failing_input"]["pass"]}
                   for a, m in jobs.items() if "I_ID" in m["gates"]}
    return out, pl, um


# ------------------------------------------------------------------------------------------------------- G-TOPK
def topk_point(t: int, p: dict, sizes: dict) -> dict:
    """The job's G-TOPK at t from the bank: the count of (pole, slot) entries whose banked top-50 position is a position
    the plan captures at t, and a lower bound on the failing input (the forward's neuron n - 1 at that position against
    the bank's neuron n): exact where the position is in neuron n - 1's own banked top-50, else the distance of the bank
    value to the interval neuron n - 1's top-50 thresholds allow. Aggregates only."""
    need = {}
    for w in p["need"].values():
        for lyr, _, e in w:
            need.setdefault(lyr, set()).add(e)
    n_all, lb_all, bank_abs, n_exact = 0, [], [], 0
    tol = m2.tol_for(t, m2.L40S)["p99"]
    for lyr, es in sorted(need.items()):
        z = read_members(t, "topk", lyr, ("tok", "a"), sizes[lyr])
        tok, a = z["tok"], z["a"].astype(np.float64)
        n_i, s_i, j_i = np.nonzero(np.isin(tok, np.array(sorted(es), np.int64)))
        n_all += n_i.size
        full = (tok >= 0).all(-1)                                        # [N, 2]: all 50 slots filled
        mplus = np.where(full[:, 0], np.where(tok[:, 0] >= 0, a[:, 0], np.inf).min(-1), 0.0)
        mminus = np.where(full[:, 1], np.where(tok[:, 1] >= 0, -a[:, 1], np.inf).min(-1), 0.0)
        for n, s, j in zip(n_i, s_i, j_i):
            e, ab = tok[n, s, j], a[n, s, j]
            q = (n - 1) % m2.N_NEURONS
            hit = np.argwhere(tok[q] == e)
            if hit.size:
                v = a[q, hit[0][0], hit[0][1]]
                lb = abs(v - ab)
                n_exact += 1
            else:
                lb = max(-mminus[q] - ab, ab - mplus[q], 0.0)
            lb_all.append(lb / (m2.ATOL + abs(ab)))
            bank_abs.append(abs(ab))
    lb_all = np.array(lb_all)
    row = {"tol": tol, "n_layers": len(need), "n_positions": sum(len(v) for v in need.values()), "n_entries": n_all}
    if n_all:
        row["failing_lower_bound"] = {"median": float(np.median(lb_all)), "p99": float(np.quantile(lb_all, 0.99)),
                                      "min": float(lb_all.min()), "n_exact": n_exact,
                                      "n_above_tol": int((lb_all > tol).sum()),
                                      "fails_for_certain": bool(np.quantile(lb_all, 0.99) > tol)}
        row["bank_abs_min"] = float(np.min(bank_abs))
    return row


# --------------------------------------------------------------------------------------- construction control A
def construction_a_freeze(bj: dict, sizes: dict) -> dict:
    """The verdict's construction control under A at T_15 on the freeze half (DISCOVERY half): the carrier's rank under
    A on E_c among its layer's 22,016 poles (ties against), and its failing input (the opposite pole's rank)."""
    fr = [u for u in m2.units(bj["births"]) if u["kind"] == "real" and u["half"] == "freeze"]
    layers = sorted({u["layer"] for u in fr})
    with cf.ThreadPoolExecutor(8) as ex:
        tok = dict(zip(layers, ex.map(lambda lyr: read_members(15, "topk", lyr, ("tok",), sizes[lyr])["tok"], layers)))
    rk, rk_opp, full, ties = [], [], 0, 0
    for u in fr:
        A = np.isin(tok[u["layer"]], np.asarray(u["tok"], np.int64)).sum(-1).astype(np.float64)
        n, s = u["carrier"]
        r = m2.ranks(A)
        rk.append(int(r[n, s]))
        rk_opp.append(int(r[n, 1 - s]))
        full += int(A[n, s] == len(u["tok"]))
        ties += int(r[n, s] > 1)
    rk, rk_opp = np.array(rk), np.array(rk_opp)
    return {"n": len(fr), "n_layers": len(layers), "carrier_A_equals_size": full,
            "precision_at_10": float((rk <= m2.TOP).mean()), "n_rank_1": int((rk == 1).sum()),
            "n_rank_2_10": int(((rk > 1) & (rk <= m2.TOP)).sum()), "n_rank_above_10": int((rk > m2.TOP).sum()),
            "max_rank": int(rk.max()), "n_tied": ties, "misses": [fr[i]["unit"] for i in np.flatnonzero(rk > m2.TOP)],
            "opposite_precision_at_10": float((rk_opp <= m2.TOP).mean()), "opposite_min_rank": int(rk_opp.min()),
            "ctrl_min": m2.CTRL_MIN, "pass": bool((rk <= m2.TOP).mean() >= m2.CTRL_MIN),
            "failing_input_fails": bool((rk_opp <= m2.TOP).mean() < m2.CTRL_MIN)}


def usable(pl: dict, bj: dict) -> dict:
    us = {u["unit"]: u for u in pl["units"]}
    out = {}
    for h in ("freeze", "test"):
        real = [u for u in us.values() if u["kind"] == "real" and u["half"] == h]
        pb = [u for u in us.values() if u["kind"] == "pb" and u["half"] == h]
        ok = [u for u in real if all(u["roles"][r] in pl["points"] for r in ("pre", "end"))]
        out[h] = {"births": len(real), "usable_if_nothing_parked": len(ok), "twins": len(pb), "n_min": m2.N_MIN}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--births", required=True)
    ap.add_argument("--repeat", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bj = json.loads(pathlib.Path(a.births).read_text())
    jobs = jobs_gates()
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        res = {"what": __doc__.split("\n\n")[0], "attempts_read": list(jobs), "G0": g0(d, jobs),
               "rev_cfg": rev_cfg(d, jobs)}
    a1 = {an: m for an, m in jobs.items() if an != "score-01"}      # score-01 ran the freeze-1 statistic (relative difference)
    kn, pl, um = known_number(pathlib.Path(a.bank), bj, pathlib.Path(a.repeat), a1)
    res.update(kn)
    sz = {t: S.sizes(t, "topk") for t in (5, 6, 7, 15)}
    res["G_TOPK"] = {"offline": {t: topk_point(t, pl["points"][t], sz[t]) for t in (7, 6, 5)}, "jobs": {}}
    for t in PTS:
        for an, m in a1.items():
            g = m["gates"].get(f"t{t:02d}_G_TOPK")
            if g is not None:
                res["G_TOPK"]["jobs"].setdefault(t, {})[an] = {
                    "n": g["n"], "p99": g.get("p99"), "pass": g["pass"], "tol": g.get("tol", {}).get("p99"),
                    "failing_median": g.get("failing_input", {}).get("median"),
                    "failing_p99": g.get("failing_input", {}).get("p99"),
                    "failing_pass": g.get("failing_input", {}).get("pass")}
    res["construction_A_freeze"] = construction_a_freeze(bj, sz[15])
    res["usable"] = usable(pl, bj)
    res["projection"] = {an: m.get("projection") for an, m in jobs.items()}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))

    # ---- tables
    print(f"G0 (tolerance {G0_TOL}; u x {WRONG_U_SCALE} is the failing input from A5; the swap is reported)")
    print("point | pass side: harvest max / score-05 max | u x1.1 harvest min-max (layers over tol) | bound within | "
          "act swap: harvest max / score-05 max")
    for t in sorted(res["G0"], reverse=True):
        r = res["G0"][t]
        j = r["jobs"].get("score-05", {})
        print(f"T_{t} | {r['pass_side_harvest']['max_ratio']:.5f} / {j.get('max_ratio', float('nan')):.5f} | "
              f"{r['u1.1_harvest']['min_ratio']:.4f}-{r['u1.1_harvest']['max_ratio']:.4f} "
              f"({r['u1.1_harvest']['n_layers_over_tol']}/32) | {r['u1.1_bound']['n_within']}/32 | "
              f"{r['act_swap_harvest']['max_ratio']:.5f} / {j.get('act_swap_max_ratio', float('nan')):.5f}")
    print("G-REV, G-CFG")
    for t, r in sorted(res["rev_cfg"]["points"].items(), reverse=True):
        print(f"T_{t} | registered = manifest = HF ref today: {r['G_REV']['pass']} | failing input fails: "
              f"{r['G_REV']['failing_input_fails']} | G-CFG pass {r['G_CFG']['pass']}, failing input fails "
              f"{r['G_CFG']['failing_input_fails']} | device {r['manifest']['device_name']} | run {r['manifest']['out']}")
    print(f"K0 failing input (T_14's values): p99 {res['K0']['failing_input_T14']['p99']:.4f}; jobs "
          f"{ {a: v['p99'] for a, v in res['K0']['jobs'].items()} }")
    print("G-EX: point | gated | entries | failing p99 (x tol) | pass side (score-05 / repeat)")
    for t, r in sorted(res["G_EX"].items(), reverse=True):
        j = r["jobs"].get("score-05", {})
        ps = r.get("pass_side_repeat", {}).get("p99")
        print(f"T_{t} | {r['gated']} | {r['n']} ({r['n_gated']} read) | {r['failing_input']['p99']:.3f} "
              f"({r['failing_input']['times_tol']:.0f}x) | {j.get('p99')} / {ps}")
    print("G-TOPK offline (aggregates):")
    for t, r in res["G_TOPK"]["offline"].items():
        print(f"T_{t} | {json.dumps(r)}")
    print("G-TOPK jobs (score-05):", {t: (v["score-05"]["n"], v["score-05"]["p99"], v["score-05"]["failing_p99"])
                                       for t, v in res["G_TOPK"]["jobs"].items() if "score-05" in v})
    print("I-ID:", json.dumps(res["I_ID"]))
    print("construction control under A, freeze half, T_15:", json.dumps(res["construction_A_freeze"]))
    print("usable:", json.dumps(res["usable"]))
    print("A5_SIDES_DONE", m2.sha256_file(a.out)[:16])


if __name__ == "__main__":
    main()
