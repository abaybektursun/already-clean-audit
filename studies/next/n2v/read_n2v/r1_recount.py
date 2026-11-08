"""N2V reader, step 1 and 2: recount every pair's readouts from losses/tNN.npz and check G-CAUSAL from the raw rows.

    python -I r1_recount.py --s3 <dir holding score-02/n2v> --repo <worktree> --out r1_recount.json

Own code (numpy only). The plan is rebuilt from N12's units.json and N12's records (abl order of poles), not from
infra.pg.n2v. Per pair, per SPEC section 0.5: dL = sum over the unit's windows of sum_j [l_zero(j) - l_clean(j)],
fp64 over fp32 per-position differences; dL_own the same at the unit's indices only; dL_fd and dL_fd_own from the fd
row; pred_jvp the sum of the tangent over all j; pred_jvp_own at the unit's indices; probe and probe_own from the probe
row (one per unit window, the carrier's output change added and subtracted). Two summation orders: the job's
(numpy sum per window, then a Python sum over windows ascending) and an exact one (math.fsum over every term).
"""
import argparse
import collections
import json
import math
import pathlib

import numpy as np

W = 2048


def read_jsonl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]


def pole_parse(s):
    # "L03N1326-" -> (3, 1326, 1)   (sign index 0 for +, 1 for -)
    lyr, rest = s[1:].split("N")
    return int(lyr), int(rest[:-1]), 0 if rest[-1] == "+" else 1


def my_plan(units, n12):
    by = {r["unit"]: r for r in n12 if r["kind"] == "real"}
    out = {}
    for u in units:
        if u["kind"] != "real":
            continue
        ws = collections.defaultdict(list)
        for e in u["tok"]:
            i = 1 + e % W
            if i <= W - 1:
                ws[e // W].append(i)
        r = by[u["unit"]]
        out[u["unit"]] = {"t": u["roles"]["pre"], "half": u["half"], "layer": u["layer"],
                          "poles": [a[0] for a in r["abl"]], "abl": r["abl"],
                          "uw": [(w, ws[w]) for w in sorted(ws)]}
        assert r["t_pre"] == u["roles"]["pre"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s3 = pathlib.Path(a.s3) / "score-02" / "n2v"
    repo = pathlib.Path(a.repo)
    units = json.loads((repo / "studies/next/n12/units.json").read_text())["units"]
    n12 = [r for h in ("freeze", "test") for r in read_jsonl(repo / f"studies/next/n12/banks/score-02/n12/n12_{h}.jsonl")]
    plan = my_plan(units, n12)
    recs = {h: read_jsonl(s3 / f"n2v_{h}.jsonl") for h in ("freeze", "test")}
    rec_by = {(p["unit"], p["pole"]): p for h in recs for p in recs[h]}
    assert len(rec_by) == 612, len(rec_by)
    pts = sorted({v["t"] for v in plan.values()})
    mine, exact = {}, {}
    causal = {"rows": 0, "bitwise_before_i0": 0, "differ_at_or_after_i0": 0, "differ_at_i0": 0,
              "equal_everywhere": 0, "by_kind": collections.Counter(), "examples": []}
    jvp_before = {"rows": 0, "exact_zero_before_i0": 0, "max_abs_before_i0": 0.0, "nonzero_at_or_after_i0": 0}
    window_check = {"units": 0, "windows_equal_plan": 0, "key_sets_complete": 0}
    clean_same_window = {"pairs": 0, "bitwise": 0, "max_abs": 0.0}
    ifwd_fail = {"n": 0, "equal": 0}
    ijvp_fail = {"n": 0, "equal": 0}
    shared_wl = []
    for t in pts:
        z = np.load(s3 / "losses" / f"t{t:02d}.npz")
        keys = set(z.keys())
        us = sorted(u for u, v in plan.items() if v["t"] == t)
        # windows per unit from the keys
        kw = collections.defaultdict(set)
        for k in keys:
            u, w, kind, q = k.split("|")
            kw[u].add(int(w))
        assert set(kw) == set(us), (t, set(kw) ^ set(us))
        clean_by_w = collections.defaultdict(list)
        wl = collections.defaultdict(list)
        for u in us:
            p = plan[u]
            window_check["units"] += 1
            window_check["windows_equal_plan"] += int(sorted(kw[u]) == [w for w, _ in p["uw"]])
            need = {f"{u}|{w}|clean|None" for w, _ in p["uw"]} | {f"{u}|{w}|probe|0" for w, _ in p["uw"]} | \
                {f"{u}|{w}|{k}|{q}" for w, _ in p["uw"] for k in ("zero", "fd", "jvp") for q in range(4)}
            window_check["key_sets_complete"] += int(need <= keys and len([k for k in keys if k.startswith(u + "|")]) == len(need))
            for w, ix in p["uw"]:
                wl[(w, p["layer"])].append(u)
            vals = {q: collections.defaultdict(float) for q in range(4)}
            terms = {q: collections.defaultdict(list) for q in range(4)}
            pr, prt = collections.defaultdict(float), collections.defaultdict(list)
            for w, ix in p["uw"]:
                lc = z[f"{u}|{w}|clean|None"]
                clean_by_w[w].append(lc)
                i0 = min(ix)
                ix = np.asarray(ix)
                for kind, qs in (("zero", range(4)), ("fd", range(4)), ("probe", [0])):
                    for q in qs:
                        lr = z[f"{u}|{w}|{kind}|{q}"]
                        causal["rows"] += 1
                        causal["by_kind"][kind] += 1
                        b = bool(np.array_equal(lr[:i0], lc[:i0]))
                        causal["bitwise_before_i0"] += int(b)
                        causal["differ_at_or_after_i0"] += int(not np.array_equal(lr[i0:], lc[i0:]))
                        causal["differ_at_i0"] += int(lr[i0] != lc[i0])
                        causal["equal_everywhere"] += int(np.array_equal(lr, lc))
                        if sum(x["t"] == t for x in causal["examples"]) < 8:
                            nz = np.flatnonzero(lr != lc)
                            causal["examples"].append({"t": t, "unit": u, "window": w, "kind": kind, "q": q, "i0": int(i0),
                                                       "bitwise_before_i0": b, "first_differing_j": int(nz[0]) if nz.size else None,
                                                       "n_differ": int(nz.size)})
                d = {}
                for kind in ("zero", "fd"):
                    for q in range(4):
                        x = z[f"{u}|{w}|{kind}|{q}"].astype(np.float64) - lc.astype(np.float64)
                        key = "dL" if kind == "zero" else "dL_fd"
                        vals[q][key] += x.sum()
                        vals[q][key + "_own"] += x[ix].sum()
                        terms[q][key] += x.tolist()
                        terms[q][key + "_own"] += x[ix].tolist()
                for q in range(4):
                    tg = z[f"{u}|{w}|jvp|{q}"]
                    jvp_before["rows"] += 1
                    jvp_before["exact_zero_before_i0"] += int(np.all(tg[:i0] == 0))
                    jvp_before["max_abs_before_i0"] = max(jvp_before["max_abs_before_i0"], float(np.abs(tg[:i0]).max()) if i0 > 0 else 0.0)
                    jvp_before["nonzero_at_or_after_i0"] += int(np.any(tg[i0:] != 0))
                    x = tg.astype(np.float64)
                    vals[q]["pred_jvp"] += x.sum()
                    vals[q]["pred_jvp_own"] += x[ix].sum()
                    terms[q]["pred_jvp"] += x.tolist()
                    terms[q]["pred_jvp_own"] += x[ix].tolist()
                x = z[f"{u}|{w}|probe|0"].astype(np.float64) - lc.astype(np.float64)
                pr["probe"] += x.sum()
                pr["probe_own"] += x[ix].sum()
                prt["probe"] += x.tolist()
                prt["probe_own"] += x[ix].tolist()
            for q, pole in enumerate(p["poles"]):
                mine[(u, pole)] = {**{k: float(v) for k, v in vals[q].items()}, **{k: float(v) for k, v in pr.items()},
                                   "pred_bf16_n12": p["abl"][q][1], "meas_bf16_n12": p["abl"][q][2], "t": t,
                                   "half": p["half"], "rank": q, "n_windows": len(p["uw"]),
                                   "n_later": int(sum(W - min(ix) for _, ix in p["uw"]))}
                exact[(u, pole)] = {**{k: math.fsum(v) for k, v in terms[q].items()},
                                    **{k: math.fsum(v) for k, v in prt.items()}}
        # the same window's clean row in different units (other batches, other companions): bitwise expected
        for w, rows in clean_by_w.items():
            for r in rows[1:]:
                clean_same_window["pairs"] += 1
                clean_same_window["bitwise"] += int(np.array_equal(r, rows[0]))
                clean_same_window["max_abs"] = max(clean_same_window["max_abs"], float(np.abs(r.astype(np.float64) - rows[0]).max()))
        shared_wl += [{"t": t, "window": w, "layer": l, "units": v} for (w, l), v in wl.items() if len(v) > 1]
        if t == 13:
            first16 = []
            for u in us:
                for w, ix in plan[u]["uw"]:
                    ifwd_fail["n"] += 1
                    ifwd_fail["equal"] += int(np.array_equal(z[f"{u}|{w}|clean|None"], z[f"{u}|{w}|zero|0"]))
                    if len(first16) < 16:
                        first16.append((u, w))
            for u, w in first16:
                ijvp_fail["n"] += 1
                ijvp_fail["equal"] += int(np.array_equal(z[f"{u}|{w}|jvp|0"], z[f"{u}|{w}|jvp|1"]))
    # compare with the records
    fields = ["dL", "dL_own", "dL_fd", "dL_fd_own", "pred_jvp", "pred_jvp_own", "probe", "probe_own"]
    cmp = {}
    for f in fields:
        ab = np.array([abs(mine[k][f] - rec_by[k][f]) for k in rec_by])
        rel = np.array([abs(mine[k][f] - rec_by[k][f]) / max(abs(rec_by[k][f]), 1e-300) for k in rec_by])
        abx = np.array([abs(exact[k][f] - rec_by[k][f]) for k in rec_by])
        relx = np.array([abs(exact[k][f] - rec_by[k][f]) / max(abs(rec_by[k][f]), 1e-300) for k in rec_by])
        cmp[f] = {"job_order_max_abs": float(ab.max()), "job_order_max_rel": float(rel.max()),
                  "job_order_n_bitwise": int((ab == 0).sum()), "fsum_max_abs": float(abx.max()),
                  "fsum_max_rel": float(relx.max()), "n": int(ab.size)}
    for f, g in (("pred_bf16", "pred_bf16_n12"), ("meas_bf16", "meas_bf16_n12")):
        ab = np.array([abs(mine[k][g] - rec_by[k][f]) for k in rec_by])
        cmp[f + "_vs_N12_abl"] = {"max_abs": float(ab.max()), "n_bitwise": int((ab == 0).sum()), "n": int(ab.size)}
    for f in ("t", "half", "rank", "n_windows", "n_later"):
        cmp[f + "_equal"] = int(sum(mine[k][f] == rec_by[k][f] for k in rec_by))
    out = {"s3_dir": str(s3), "n_pairs": len(rec_by), "points": pts, "window_check": window_check,
           "compare_with_records": cmp, "G_CAUSAL_from_npz": {**causal, "by_kind": dict(causal["by_kind"])},
           "jvp_tangent_before_i0": jvp_before, "identity_clean_same_window_other_unit": clean_same_window,
           "I_FWD_failing_input_from_npz": ifwd_fail, "I_JVP_failing_input_from_npz": ijvp_fail,
           "shared_window_layer_at_point": {"n": len(shared_wl), "items": shared_wl[:20]}}
    # per-pair recount, for the next scripts
    out["pairs"] = [{"unit": k[0], "pole": k[1], **mine[k], **{f + "_fsum": exact[k][f] for f in fields}}
                    for k in sorted(mine)]
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
