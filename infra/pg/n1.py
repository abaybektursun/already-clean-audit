"""N1: the weight order at birth (studies/next/07_PICKS/N1_SPEC.md). Numpy only; the job imports torch itself.

Users:
  - the laptop: the onset pre-check (`onset_row`, studies/next/n1/onsets/onsets.py), the unit table (`units`), the
    projection (`project`), freeze 2 (`freeze`, from the freeze half only) and the verdict (`verdict`, once, on the test
    half);
  - the job infra/aws_pg/jobs/n1_weights.py (the per-neuron statistics of every checkpoint pair, the known number, the
    identity pair, the positive control);
  - the dry run infra/aws_pg/jobs/n1_dryrun.py.

    python -m infra.pg.n1 units   --out studies/next/n1/units.json
    python -m infra.pg.n1 project
    python -m infra.pg.n1 freeze  --bank studies/next/n1/banks/<attempt> --out studies/next/n1/thresholds.json
    python -m infra.pg.n1 verdict --bank studies/next/n1/banks/<attempt> --thresholds studies/next/n1/thresholds.json \
                                  --out studies/next/n1/verdict_n1.json

Objects (SPEC section 0.5). Grid points T_0..T_15 (m2.POINTS); bracket j is (T_(j-1), T_j], j = 1..15. For neuron n of
layer l and side M:
  in   the input rows, [gate_proj.weight[n, :], up_proj.weight[n, :]] (8,192 entries: the directions the neuron reads);
  out  the output column, down_proj.weight[:, n] (4,096 entries: the direction it writes).
  r_M(n, j)   = ||M_n(T_j) - M_n(T_(j-1))|| / ||M_n(T_(j-1))||, fp64 from the bf16 checkpoint (exact upcast).
  z_M(n, j)   = (log r_M(n, j) - median over the layer's 11,008 neurons) / (1.4826 * MAD over them): the neuron's place
                among its layer in that bracket (it removes the bracket's step count and learning rate, which every
                neuron of the bracket shares).
  e_M(n, j)   = z_M(n, j) - mean of z_M(n, i) over the reference brackets i = 2 .. j - 2: the excess over the neuron's own
                history (it removes a neuron's lasting scale, such as a small norm that makes every relative change large).
A unit (a birth's carrier, or a twin's pole) dated k reads e_in and e_out at the birth bracket k (and at k - 1 and k + 1
as reported order readouts).

Fail loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from infra.pg import m2

R_FLOOR = 1e-3              # onset pre-check: r floored before log10
SIDES = ("in", "out")
REF_FIRST = 2               # the reference brackets start at bracket 2 (bracket 1 is initialisation to step 1,000)
REF_GAP = 2                 # ... and end at k - 2 (bracket k - 1 is reported as the pre-birth bracket, not reference)
N_MIN = 50                  # usable units per half and kind
N_BOOT, SEED = 10000, 0
N_FLOOR_REPS = 1000
MAD_K = 1.4826
KB_TOL = 1.3e-4             # K-B: |ours - bank| / bank for a column norm; the bank is fp32 over 4,096 terms, whose
                            # rounding is at most (n + 1) eps / (1 - (n + 1) eps) = 2.44e-4 on the sum of squares in any
                            # order (eps = 2^-24), so 1.22e-4 on the norm, plus the final rounding
PC_N = 32                   # PC: planted neurons per layer and bracket
PC_SEED = 20261106
PC_MIN = 0.9                # PC: share of planted neurons above the layer's 95th percentile of z, both sides
# A1 (2026-11-07): the ten raw-example births per half, copied from studies/next/n12/units.json -> examples (N12's rule:
# freeze, 5 with A_own >= 1 and 5 with A_own = 0, default_rng(0) on sorted ids; test, 10 by default_rng(1) on sorted
# ids, no data read). N1's units.json does not carry them; the job reads them from its registry (n1_weights.registry).
EXAMPLES = {"freeze": ["L03N7574+:c6", "L03N7746+:c0", "L06N9844+:c1", "L16N1206+:c1", "L20N5782+:c1", "L07N8340+:c4",
                       "L10N9052+:c2", "L10N9794+:c2", "L19N7374-:c1", "L24N5066-:c1"],
            "test": ["L01N369+:c0", "L02N4649+:c3", "L03N9631+:c3", "L05N3483+:c3", "L10N10529-:c3", "L12N5641+:c0",
                     "L22N7483-:c5", "L24N8193-:c1", "L27N8999-:c1", "L28N4613+:c0"]}


# ----------------------------------------------------------------------------------------------- the onset pre-check
def trajectory(u: dict, z) -> np.ndarray:
    """r_t = median over the unit's T_15 exemplar slots of a+_t, divided by the pole's q999_t, t = 0..15."""
    i = u["row"]
    sgn = 1.0 if u["pole"].endswith("+") else -1.0
    ap = np.maximum(sgn * z["a"][:, i, :], 0.0)
    med = np.median(ap[:, u["ranks"]], axis=1)
    q = z["q999"][:, i]
    assert (q > 0).all(), f"{u['unit']}: q999 <= 0 at some point"
    return med / q


def onset_row(u: dict, z) -> dict:
    """One unit's onset readout on the grid (see studies/next/n1/onsets/onsets.py)."""
    i = u["row"]
    sgn = 1.0 if u["pole"].endswith("+") else -1.0
    ap = np.maximum(sgn * z["a"][:, i, :], 0.0)
    k = int(m2.births_matrix(ap, z["q999"][:, i], np.array([u["ranks"]]))[0])
    r = trajectory(u, z)
    pres = r >= 1.0
    lr = np.log10(np.maximum(r, R_FLOOR))
    trans = bool(any(pres[t] and not pres[t + 1] for t in range(0, k)))
    f = None
    if r[k - 1] > 0 and r[k] > 0 and r[k - 1] < 1.0 <= r[k]:
        s0, s1 = np.log(max(m2.STEPS[k - 1], 1)), np.log(m2.STEPS[k])
        f = float((0.0 - np.log(r[k - 1])) / (np.log(r[k]) - np.log(r[k - 1])))
        f = float((np.exp(s0 + f * (s1 - s0)) - m2.STEPS[k - 1]) / (m2.STEPS[k] - m2.STEPS[k - 1]))
    return {"unit": u["unit"], "kind": u["kind"], "half": u["half"], "pole": u["pole"], "k": k, "k_unit": u["k"],
            "bracket": [m2.STEPS[k - 1], m2.STEPS[k]], "bracket_steps": m2.STEPS[k] - m2.STEPS[k - 1],
            "r": {"pre3": float(r[k - 3]), "pre2": float(r[k - 2]), "pre": float(r[k - 1]), "birth": float(r[k]),
                  "post": float(r[k + 1]) if k < 15 else None},
            "r_all": [round(float(x), 5) for x in r], "transient": trans,
            "rise_k": float(lr[k] - lr[k - 1]), "rise_pre": float(lr[k - 1] - lr[k - 3]),
            "concentrated": bool(lr[k] - lr[k - 1] > lr[k - 1] - lr[k - 3]), "f_cross": f}


# ----------------------------------------------------------------------------------------------- the unit table
def units(n12_units: list, m2_ranks: list) -> dict:
    """N12's units (real, dt, pb: births and their twins with layer, pole and k) as neurons, with the activation-matched
    neurons of each real unit (M2's 100 matched poles at T_(k-1), the carrier's neuron excluded, as neurons, first
    occurrence kept)."""
    matched = {r["unit"]: r["matched"] for r in m2_ranks if r["role"] == "pre" and r["kind"] == "real"}
    out = []
    for u in n12_units:
        lyr, n, s = m2.parse_pole(u["pole"])
        row = {"unit": u["unit"], "kind": u["kind"], "half": u["half"], "birth": u["birth"], "layer": lyr, "neuron": n,
               "sign": s, "pole": u["pole"], "k": u["k"], "d": u["d"], "size": len(u["tok"])}
        if u["kind"] == "real":
            ms = []
            for p in matched[u["unit"]]:
                ln, nn, _ = m2.parse_pole(p)
                assert ln == lyr
                if nn != n and nn not in ms:
                    ms.append(nn)
            row["matched"] = ms
        out.append(row)
    return {"units": out, "n": {f"{k}_{h}": sum(u["kind"] == k and u["half"] == h for u in out)
                                for k in ("real", "dt", "pb") for h in ("freeze", "test")}}


# ----------------------------------------------------------------------------------------------- the statistics
def window(k: int) -> list:
    """The rise window: brackets k - 2, k - 1, k (the onset pre-check: most births rise over several brackets)."""
    return [k - 2, k - 1, k]


def reference(k: int) -> list:
    """The reference brackets: 2 .. k - 3 (bracket 1 is initialisation; the window is excluded)."""
    ref = list(range(REF_FIRST, k - REF_GAP))
    assert ref, f"k = {k}: no reference bracket"
    return ref


def side_r(norms: np.ndarray, diff: np.ndarray) -> dict:
    """norms [16, L, N, 3] (gate row, up row, down column), diff [15, L, N, 3] (bracket j at index j - 1) -> r per side,
    [15, L, N] each: in = [gate; up] as one 8,192-vector, out = the column; gate and up reported."""
    prev = norms[:-1]
    r = {"in": np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2) / np.sqrt(prev[..., 0] ** 2 + prev[..., 1] ** 2),
         "out": diff[..., 2] / prev[..., 2], "gate": diff[..., 0] / prev[..., 0], "up": diff[..., 1] / prev[..., 1]}
    for k, v in r.items():
        assert np.isfinite(v).all() and (v >= 0).all(), f"r_{k}: a non-finite or negative value"
    return r


def robust_z(r: np.ndarray) -> np.ndarray:
    """[15, L, N] -> z of log r within each (bracket, layer) over its N neurons: (x - median) / (1.4826 MAD)."""
    x = np.log(np.maximum(r, 1e-300))
    med = np.median(x, axis=-1, keepdims=True)
    mad = MAD_K * np.median(np.abs(x - med), axis=-1, keepdims=True)
    assert (mad > 0).all(), "a bracket and layer with MAD 0"
    return (x - med) / mad


def excess(z: np.ndarray, layer: int, n: int, k: int, brackets: list) -> float:
    """Mean z over `brackets` minus mean z over the reference brackets of k, for neuron n of `layer`."""
    zz = z[:, layer, n]
    return float(np.mean([zz[j - 1] for j in brackets]) - np.mean([zz[i - 1] for i in reference(k)]))


def unit_record(u: dict, Z: dict, R: dict, C: dict) -> dict:
    """One line of n1_<half>.jsonl. Z, R: per side z and r [15, L, N]; C: per side cosine of consecutive vectors."""
    lyr, n, k = u["layer"], u["neuron"], u["k"]
    rec = {x: u[x] for x in ("unit", "kind", "half", "birth", "layer", "neuron", "pole", "k", "d", "size")}
    for s in Z:
        rec[s] = {"e_window": excess(Z[s], lyr, n, k, window(k)), "e_k": excess(Z[s], lyr, n, k, [k]),
                  "e_by_bracket": {str(j): excess(Z[s], lyr, n, k, [j]) for j in range(k - 2, min(k + 1, 15) + 1)},
                  "r": [float(x) for x in R[s][:, lyr, n]], "z": [round(float(x), 5) for x in Z[s][:, lyr, n]],
                  "cos": [float(x) for x in C[s][:, lyr, n]] if s in C else None}
    if "matched" in u:
        rec["matched_e_window"] = {s: [excess(Z[s], lyr, m, k, window(k)) for m in u["matched"]] for s in ("in", "out")}
    rec["ceiling"] = {s: float(np.mean([np.quantile(Z[s][j - 1, lyr], 0.99) for j in window(k)])) for s in ("in", "out")}
    return rec


def cosines(norms: np.ndarray, dot: np.ndarray) -> dict:
    """cos of consecutive vectors per bracket: in from the summed gate and up dots, out from the column's."""
    p, c = norms[:-1], norms[1:]
    return {"in": (dot[..., 0] + dot[..., 1]) / (np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2) * np.sqrt(c[..., 0] ** 2 + c[..., 1] ** 2)),
            "out": dot[..., 2] / (p[..., 2] * c[..., 2]),
            "gate": dot[..., 0] / (p[..., 0] * c[..., 0]), "up": dot[..., 1] / (p[..., 1] * c[..., 1])}


def records(U: list, norms: np.ndarray, diff: np.ndarray, dot: np.ndarray) -> list:
    R = side_r(norms, diff)
    Z = {s: robust_z(v) for s, v in R.items()}
    C = cosines(norms, dot)
    return [unit_record(u, Z, R, C) for u in U]


# ----------------------------------------------------------------------------------------------- summaries and labels
def boot_mean(x, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    x = np.asarray(x, np.float64)
    assert x.size > 0
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"n": int(x.size), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def floor_reps(me: np.ndarray, reps: int = N_FLOOR_REPS, seed: int = SEED) -> dict:
    """me [units, K] matched neurons' e_window; replicate r draws one per unit; its mean. q975 is the floor's top."""
    rng = np.random.default_rng(seed)
    nu, k = me.shape
    v = np.array([me[np.arange(nu), rng.integers(0, k, nu)].mean() for _ in range(reps)])
    return {"mean": float(v.mean()), "q025": float(np.quantile(v, 0.025)), "q975": float(np.quantile(v, 0.975)),
            "reps": reps}


SEL = ("ABOVE_SELECTION", "AT_SELECTION", "BELOW_SELECTION")


def label_sign(stat: dict) -> str:
    return SEL[0] if stat["lo"] > 0 else (SEL[2] if stat["hi"] < 0 else SEL[1])


def half_summary(recs: list) -> dict:
    R = {r["birth"]: r for r in recs if r["kind"] == "real"}
    out = {"n": {k: sum(r["kind"] == k for r in recs) for k in ("real", "dt", "pb")}}
    for tw in ("dt", "pb"):
        T = {r["birth"]: r for r in recs if r["kind"] == tw}
        ids = sorted(b for b in R if b in T)
        for s in ("in", "out", "gate", "up"):
            out[f"pair_{s}_real_{tw}"] = m2.paired([R[b][s]["e_window"] for b in ids], [T[b][s]["e_window"] for b in ids]) \
                if ids else None
            out[f"pair_k_{s}_real_{tw}"] = m2.paired([R[b][s]["e_k"] for b in ids], [T[b][s]["e_k"] for b in ids]) \
                if ids else None
    for s in ("in", "out", "gate", "up"):
        out[f"real_{s}"] = boot_mean([r[s]["e_window"] for r in R.values()])
        for kind in ("dt", "pb"):
            out[f"{kind}_{s}"] = boot_mean([r[s]["e_window"] for r in recs if r["kind"] == kind])
        out[f"profile_{s}"] = {str(d): boot_mean([r[s]["e_by_bracket"][str(r["k"] + d)] for r in R.values()
                                                  if str(r["k"] + d) in r[s]["e_by_bracket"]]) for d in (-2, -1, 0, 1)}
    for s in ("in", "out"):
        out[f"floor_{s}"] = floor_reps(np.array([r["matched_e_window"][s][:min(len(x["matched_e_window"][s])
                                                                                  for x in R.values())]
                                                 for r in R.values()]))
        out[f"ceiling_{s}"] = float(np.median([r["ceiling"][s] for r in R.values()]))
    return out


def hypothesis(lab_in: str, lab_out: str) -> str:
    a, b = lab_in == SEL[0], lab_out == SEL[0]
    if a and b:
        return "TOGETHER"
    if a:
        return "ROWS_FIRST"
    if b:
        return "COLUMN_FIRST"
    return "NEITHER" if not (lab_in == SEL[1] and lab_out == SEL[1]) else "NEITHER (H_0: both at selection)"


def room_check(n: int = 67) -> dict:
    """Every branch of the frozen labels is reachable at the test half's twin count."""
    one, zero = np.ones(n), np.zeros(n)
    labs = {"above": label_sign(m2.paired(one, zero)), "below": label_sign(m2.paired(zero, one)),
            "at": label_sign(m2.paired(np.r_[one[: n // 2], -one[n // 2:]], zero))}
    hy = {hypothesis(a, b) for a in SEL for b in SEL}
    ok = labs == {"above": SEL[0], "below": SEL[2], "at": SEL[1]} and \
        {"TOGETHER", "ROWS_FIRST", "COLUMN_FIRST", "NEITHER", "NEITHER (H_0: both at selection)"} <= hy
    return {"pass": bool(ok), "labels": labs, "hypotheses": sorted(hy)}


def freeze(bank) -> dict:
    bank = pathlib.Path(bank)
    jm = json.loads((bank / "n1" / "metrics.json").read_text())
    g = m2.gates_ok(jm)
    assert g["pass"], f"freeze refused: gates {g}"
    fz = m2.read_jsonl(bank / "n1" / "n1_freeze.jsonl")
    s = half_summary(fz)
    assert s["n"]["real"] >= N_MIN and s["n"]["dt"] >= N_MIN, f"freeze half too small: {s['n']}"
    rc = room_check()
    assert rc["pass"], f"room check failed: {rc}"
    return {"created_from": str(bank / "n1" / "n1_freeze.jsonl"),
            "sha256_n1_freeze": m2.sha256_file(bank / "n1" / "n1_freeze.jsonl"),
            "sha256_job_metrics": m2.sha256_file(bank / "n1" / "metrics.json"),
            "floor_in_freeze": s["floor_in"], "floor_out_freeze": s["floor_out"], "room": rc, "gates": g,
            "discovery": s}


def verdict(bank, th: dict) -> dict:
    bank = pathlib.Path(bank)
    assert m2.sha256_file(bank / "n1" / "n1_freeze.jsonl") == th["sha256_n1_freeze"], "freeze records changed"
    jm = json.loads((bank / "n1" / "metrics.json").read_text())
    g = m2.gates_ok(jm)
    te = m2.read_jsonl(bank / "n1" / "n1_test.jsonl")
    s = half_summary(te)
    v = {"gates": g, "test": s, "labels": {}, "thresholds_sha256": th.get("_sha256")}
    reasons = [] if g["pass"] else [f"gates: {g['problems'] or 'incomplete'}"]
    if s["n"]["real"] < N_MIN:
        reasons.append(f"missing: {s['n']['real']} real test units < {N_MIN}")
    L = v["labels"]
    for side in ("in", "out"):
        pr = s[f"pair_{side}_real_dt"]
        L[f"sel_{side}"] = {"label": "NOT_RUN"} if pr is None or pr["n"] < N_MIN else {"label": label_sign(pr), **pr}
        bar = max(th[f"floor_{side}_freeze"]["q975"], s[f"floor_{side}"]["q975"])
        st = s[f"real_{side}"]
        L[f"floor_{side}"] = {"label": "ABOVE_FLOOR" if st["lo"] > bar else "NOT_ABOVE_FLOOR", "bar": bar, **st}
        if st["lo"] > s[f"ceiling_{side}"]:
            L[f"sel_{side}"]["label"] = "ABOVE_CEILING"
    if any(L[f"sel_{x}"]["label"] == "NOT_RUN" for x in ("in", "out")):
        reasons.append("fewer than 50 dt pairs: no selection control")
    if any(L[f"sel_{x}"]["label"] == "ABOVE_CEILING" for x in ("in", "out")):
        reasons.append("a side above its ceiling: DOES NOT COUNT until inspected")
    v["hypothesis"] = hypothesis(L["sel_in"]["label"], L["sel_out"]["label"])
    v["headline"] = ("DOES NOT COUNT: " + "; ".join(reasons)) if reasons else \
        (f"{v['hypothesis']}: in {L['sel_in']['label']} ({L['floor_in']['label']}), "
         f"out {L['sel_out']['label']} ({L['floor_out']['label']})")
    return v


# ----------------------------------------------------------------------------------------------- the projection
RATES = {"download_s": (90.0, 150.0),   # 14.6 GB per checkpoint: M2/N12 measured 45-82 s, N6 152 s, 368 s once (N12 T_11)
         "compute_s": (60.0, 180.0),    # read 96 MLP tensors, fp64 norms, differences and dots, the positive control
         "start_s": (300.0, 450.0), "end_s": (180.0, 300.0), "identity_s": (60.0, 120.0)}


def project() -> dict:
    out = {"checkpoints": 16, "rates": RATES}
    for j, name in ((0, "center"), (1, "high")):
        s = RATES["start_s"][j] + 16 * (RATES["download_s"][j] + RATES["compute_s"][j]) + RATES["identity_s"][j] + \
            RATES["end_s"][j]
        out[f"instance_h_{name}"] = round(s / 3600, 2)
    out["budget_min_hard_kill"] = int(np.ceil(1.3 * out["instance_h_high"] * 60))
    out["gpu_h"] = 0.0
    out["instance"] = "r7i.2xlarge (8 vCPU, 64 GB; the Standard family quota, not the G quota); price not banked"
    return out


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    root = pathlib.Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["units", "project", "freeze", "verdict"])
    ap.add_argument("--bank", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "units":
        n12u = json.loads((root / "studies/next/n12/units.json").read_text())["units"]
        rk = [r for h in ("freeze", "test") for r in m2.read_jsonl(root / f"studies/next/m2/banks/score-06/m2/ranks_{h}.jsonl")]
        r = units(n12u, rk)
        pathlib.Path(a.out).write_text(json.dumps(r, indent=1))
        print("N1_UNITS_OK", json.dumps(r["n"]))
    elif a.cmd == "project":
        print(json.dumps(project(), indent=1))
    elif a.cmd == "freeze":
        assert not pathlib.Path(a.out).exists(), f"{a.out} exists; a freeze is written once"
        th = freeze(a.bank)
        pathlib.Path(a.out).write_text(json.dumps(th, indent=1))
        print("N1_FREEZE_OK", m2.sha256_file(a.out))
    else:
        assert not pathlib.Path(a.out).exists(), f"{a.out} exists; the verdict runs once"
        th = json.loads(pathlib.Path(a.thresholds).read_text())
        th["_sha256"] = m2.sha256_file(a.thresholds)
        v = verdict(a.bank, th)
        pathlib.Path(a.out).write_text(json.dumps(v, indent=1))
        print(v["headline"])


if __name__ == "__main__":
    main()
