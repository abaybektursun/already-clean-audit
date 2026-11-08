"""N1 independent read, step 1-2: recompute r, z and e for every unit from stats.npz with code written for this read
(numpy only; nothing imported from infra.pg), and compare with the job's records n1_freeze.jsonl and n1_test.jsonl.
Also: the column norms in stats.npz against kb_bank.json (775 banked values, T_5..T_15) and N6's T_15 array (K-B), and
the internal consistency of stats.npz (law of cosines between norms, diff and dot), each with an input that must fail.

Definitions (N1_SPEC.md section 0.5):
  r_M(n, j) = ||M_n(T_j) - M_n(T_(j-1))|| / ||M_n(T_(j-1))||; in = [gate row; up row] (8,192), out = down column.
  z_M(n, j) = (log r - median over the layer's 11,008 neurons) / (1.4826 * median absolute deviation), per bracket, layer.
  e_M(n, k) = mean z over the window {k-2, k-1, k} - mean z over the reference {2, ..., k-3}; e_k uses {k} alone.
  ceiling = mean over the window of the layer's 99th percentile of z (linear interpolation).

    python3 -I recount.py --stats S3/stats.npz --test S3/n1_test.jsonl --freeze S3/n1_freeze.jsonl \
        --units studies/next/n1/units.json --kb studies/next/n1/kb_bank.json \
        --n6 studies/next/n1/inputs/n6_colnorms_t15.npy --out studies/next/n1/read_n1
"""
import argparse
import json
import pathlib

import numpy as np

MAD_K = 1.4826
SIDES = ("in", "out", "gate", "up")


def med_last(a):
    s = np.sort(a, axis=-1)
    n = s.shape[-1]
    return s[..., n // 2] if n % 2 else 0.5 * (s[..., n // 2 - 1] + s[..., n // 2])


def q_lin_sorted(s, p):
    n = s.shape[-1]
    h = p * (n - 1)
    lo = int(np.floor(h))
    return s[..., lo] + (h - lo) * (s[..., lo + 1] - s[..., lo])


def r_sides(norms, diff):
    p = norms[:-1]
    return {"in": np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2) / np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2),
            "out": diff[..., 2] / p[..., 2], "gate": diff[..., 0] / p[..., 0], "up": diff[..., 1] / p[..., 1]}


def zscore(r):
    x = np.log(r)
    m = med_last(x)[..., None]
    mad = MAD_K * med_last(np.abs(x - m))[..., None]
    assert np.isfinite(x).all() and (mad > 0).all()
    return (x - m) / mad


def unit_e(zz, k):
    """zz: the neuron's z over brackets 1..15 (index j - 1)."""
    ref = list(range(2, k - 2))
    assert ref, k
    base = zz[[i - 1 for i in ref]].mean()
    return {"e_window": float(zz[[k - 3, k - 2, k - 1]].mean() - base), "e_k": float(zz[k - 1] - base),
            "e_by_bracket": {str(j): float(zz[j - 1] - base) for j in range(k - 2, min(k + 1, 15) + 1)},
            "ref_mean": float(base), "n_ref": len(ref)}


def main():
    ap = argparse.ArgumentParser()
    for a in ("stats", "test", "freeze", "units", "kb", "n6", "out"):
        ap.add_argument(f"--{a}", required=True)
    a = ap.parse_args()
    S = np.load(a.stats)
    norms, diff, dot = S["norms"], S["diff"], S["dot"]
    assert norms.shape == (16, 32, 11008, 3) and diff.shape == dot.shape == (15, 32, 11008, 3)
    out = {}

    # internal consistency: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, per vector (gate row, up row, down column)
    cur, prev = norms[1:], norms[:-1]
    lhs = diff ** 2
    rhs = cur ** 2 + prev ** 2 - 2 * dot
    scale = cur ** 2 + prev ** 2
    out["law_of_cosines"] = {"max_abs_over_scale": float((np.abs(lhs - rhs) / scale).max()),
                             "failing_input_norms_shifted_one_bracket":
                                 float(np.median(np.abs(lhs[:-1] - (norms[2:] ** 2 + prev[:-1] ** 2 - 2 * dot[:-1])) / scale[:-1]))}
    out["min_diff"] = float(diff.min())
    out["n_zero_diff"] = int((diff == 0).sum())

    R = r_sides(norms, diff)
    Z = {s: zscore(R[s]) for s in SIDES}
    Zs = {s: np.sort(Z[s], axis=-1) for s in ("in", "out")}
    q99 = {s: q_lin_sorted(Zs[s], 0.99) for s in ("in", "out")}          # [15, 32]
    C = {"in": (dot[..., 0] + dot[..., 1]) / (np.sqrt(prev[..., 0] ** 2 + prev[..., 1] ** 2) *
                                               np.sqrt(cur[..., 0] ** 2 + cur[..., 1] ** 2)),
         "out": dot[..., 2] / (prev[..., 2] * cur[..., 2])}

    units = json.loads(pathlib.Path(a.units).read_text())["units"]
    U = {u["unit"]: u for u in units}
    recs = {h: [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]
            for h, p in (("test", a.test), ("freeze", a.freeze))}

    mine = {}
    for u in units:
        l, n, k = u["layer"], u["neuron"], u["k"]
        row = {x: u[x] for x in ("unit", "kind", "half", "birth", "layer", "neuron", "pole", "k")}
        for s in SIDES:
            row[s] = unit_e(Z[s][:, l, n], k)
        row["norm_prev_k"] = {"in": float(np.sqrt(norms[k - 1, l, n, 0] ** 2 + norms[k - 1, l, n, 1] ** 2)),
                              "out": float(norms[k - 1, l, n, 2])}
        row["r_window_mean_log"] = {s: float(np.log(R[s][[k - 3, k - 2, k - 1], l, n]).mean()) for s in ("in", "out")}
        if u["kind"] == "real":
            row["matched_e_window"] = {s: [unit_e(Z[s][:, l, m], k)["e_window"] for m in u["matched"]] for s in ("in", "out")}
            row["ceiling"] = {s: float(np.mean([q99[s][j - 1, l] for j in (k - 2, k - 1, k)])) for s in ("in", "out")}
        mine[u["unit"]] = row

    # compare with the job's records
    cmp = {"n_units": 0, "max_abs": {}, "worst": {}}
    def upd(key, d, who):
        if d > cmp["max_abs"].get(key, -1):
            cmp["max_abs"][key], cmp["worst"][key] = float(d), who
    for h in ("test", "freeze"):
        for r in recs[h]:
            m = mine[r["unit"]]
            assert (r["layer"], r["neuron"], r["k"], r["kind"], r["half"]) == (m["layer"], m["neuron"], m["k"], m["kind"], m["half"])
            cmp["n_units"] += 1
            l, n = r["layer"], r["neuron"]
            for s in SIDES:
                upd(f"e_window_{s}", abs(r[s]["e_window"] - m[s]["e_window"]), r["unit"])
                upd(f"e_k_{s}", abs(r[s]["e_k"] - m[s]["e_k"]), r["unit"])
                assert set(r[s]["e_by_bracket"]) == set(m[s]["e_by_bracket"]), r["unit"]
                upd(f"e_by_bracket_{s}", max(abs(r[s]["e_by_bracket"][j] - m[s]["e_by_bracket"][j]) for j in r[s]["e_by_bracket"]), r["unit"])
                upd(f"r_{s}", float(np.max(np.abs(np.array(r[s]["r"]) - R[s][:, l, n]) / R[s][:, l, n])), r["unit"])
                upd(f"z_{s}", float(np.max(np.abs(np.array(r[s]["z"]) - Z[s][:, l, n]))), r["unit"])
                if s in C:
                    upd(f"cos_{s}", float(np.max(np.abs(np.array(r[s]["cos"]) - C[s][:, l, n]))), r["unit"])
            if r["kind"] == "real":
                for s in ("in", "out"):
                    assert len(r["matched_e_window"][s]) == len(m["matched_e_window"][s])
                    upd(f"matched_e_{s}", float(np.max(np.abs(np.array(r["matched_e_window"][s]) - np.array(m["matched_e_window"][s])))), r["unit"])
                    upd(f"ceiling_{s}", abs(r["ceiling"][s] - m["ceiling"][s]), r["unit"])
    out["records_vs_recount"] = cmp

    # ten named test units, five real and five dt, spelled out
    test_real = [r for r in recs["test"] if r["kind"] == "real"]
    test_dt = [r for r in recs["test"] if r["kind"] == "dt"]
    pick = test_real[::14][:5] + test_dt[::13][:5]
    out["ten_units"] = []
    for r in pick:
        m = mine[r["unit"]]
        k, l, n = r["k"], r["layer"], r["neuron"]
        out["ten_units"].append({"unit": r["unit"], "kind": r["kind"], "layer": l, "neuron": n, "k": k,
                                 "window": [k - 2, k - 1, k], "reference": [2, k - 3],
                                 "z_in_window": [float(Z["in"][j - 1, l, n]) for j in (k - 2, k - 1, k)],
                                 "z_out_window": [float(Z["out"][j - 1, l, n]) for j in (k - 2, k - 1, k)],
                                 "e_in_mine": m["in"]["e_window"], "e_in_record": r["in"]["e_window"],
                                 "e_out_mine": m["out"]["e_window"], "e_out_record": r["out"]["e_window"],
                                 "abs_diff_in": abs(m["in"]["e_window"] - r["in"]["e_window"]),
                                 "abs_diff_out": abs(m["out"]["e_window"] - r["out"]["e_window"])})

    # K-B: our fp64 column norms against the banked fp32 values; failing input: T_(t-1) against the bank's T_t
    kb = json.loads(pathlib.Path(a.kb).read_text())["values"]
    n6 = np.load(a.n6).astype(np.float64)
    kbo = {}
    for t in range(5, 16):
        b = [x for x in kb if x[2] == t]
        ours = np.array([norms[t, x[0], x[1], 2] for x in b])
        bv = np.array([x[3] for x in b])
        fo = np.array([norms[t - 1, x[0], x[1], 2] for x in b])
        row = {"n_bank": len(b), "max_rel": float((np.abs(ours - bv) / bv).max()),
               "failing_input_min_rel": float((np.abs(fo - bv) / bv).min()),
               "examples": [{"layer": x[0], "neuron": x[1], "t": t, "bank": x[3], "stats_npz": float(norms[t, x[0], x[1], 2]),
                             "rel": float(abs(norms[t, x[0], x[1], 2] - x[3]) / x[3])} for x in b[:2]]}
        if t == 15:
            rel6 = np.abs(norms[15, :, :, 2] - n6) / n6
            row["n6"] = {"n": int(n6.size), "max_rel": float(rel6.max()),
                         "failing_input_T14_min_rel": float((np.abs(norms[14, :, :, 2] - n6) / n6).min()),
                         "failing_input_T14_median_rel": float(np.median(np.abs(norms[14, :, :, 2] - n6) / n6))}
        kbo[f"t{t:02d}"] = row
    out["K_B"] = kbo
    out["K_B_max_rel_all"] = max(max(v["max_rel"], v.get("n6", {}).get("max_rel", 0)) for v in kbo.values())

    od = pathlib.Path(a.out)
    (od / "units_recount.json").write_text(json.dumps(mine))
    (od / "recount.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "K_B"}, indent=1))
    print("K_B max_rel", out["K_B_max_rel_all"], {t: (v["n_bank"], v["max_rel"], v["failing_input_min_rel"]) for t, v in kbo.items()})


if __name__ == "__main__":
    main()
