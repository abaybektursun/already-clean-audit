"""N2V: N2's instrument re-validated in fp32 (studies/next/07_PICKS/N2V_SPEC.md), N12's registered follow-up. Numpy only;
the job imports torch itself.

Users:
  - the laptop CLI below: `project` (GPU-hours from the plan), `freeze` (freeze half only, before the test half is
    read), `verdict` (once, on the test half) and `reread` (N12's gradient labels from N12's banked records, run by the
    verdict only on the branch that allows it);
  - the job infra/aws_pg/jobs/n2v_score.py (the plan, the per-pair records);
  - the dry run infra/aws_pg/jobs/n2v_dryrun.py.

    python -m infra.pg.n2v project
    python -m infra.pg.n2v freeze  --bank studies/next/n2v/banks/<attempt> --out studies/next/n2v/thresholds.json
    python -m infra.pg.n2v verdict --bank studies/next/n2v/banks/<attempt> --thresholds studies/next/n2v/thresholds.json \
                                   --out studies/next/n2v/verdict_n2v.json

Objects (SPEC section 0.5). The units are N12's 153 real units at their pre point T_(k-1); the pairs are N12's ablated
(unit, pole) pairs: the carrier and its top 3 competitors under B (n12_<half>.jsonl -> abl, 328 freeze, 284 test).
Per unit window w (a window holding a valid exemplar position; input index i = 1 + e mod 2048 <= 2047), rows of one
fp32 forward each, every row's per-position next-token loss l(j), j = 0..2047 (input j predicts input j + 1):
  clean   nothing changed;
  zero q  pole q's neuron: its down-projection input h set to 0 at the unit's indices in w;
  fd q    the same entries scaled by FD_SCALE (a 10 % reduction);
  probe   the carrier's output change v = -h W_down[:, n] added to the down projection's output at those indices and
          subtracted again: zero in exact arithmetic, so its loss change is the rounding floor of this path (N6's probe);
  jvp q   forward-mode derivative of every l(j) along the zero-ablation direction (tangent -h at those entries).
Per pair (unit, q), summed over the unit's windows, fp64 sums of fp32 per-position differences:
  dL      window readout: sum over j of l_zero(j) - l_clean(j);      dL_own  the same over the unit's indices only;
  dL_fd   sum over j of l_fd(j) - l_clean(j);                         dL_fd_own;
  pred_bf16   N12's banked first-order prediction sum(-h G) (bf16 forward, bf16 backward; abl[i][1]);
  pred_jvp    the fp32 forward-mode prediction of dL;                 pred_jvp_own  of dL_own;
  probe       the unit's probe change (window), probe_own (own), the same for its four pairs.

Fail loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile

import numpy as np

from infra.pg import m2, n12

FD_SCALE = 0.9              # the finite-difference row: 10 % reduction of h at the same entries
N_POLES = 1 + n12.N_ABL     # carrier + top 3 competitors under B, N12's ablated set
ROOM_RHO_MIN = 0.9          # G-ROOM: a perfect instrument plus the measured floor must reach this median Spearman
N_ROOM = 1000
K_A_TOL = 1e-6              # K-A: N12's bf16 ablation rows re-measured on the instance, |dL - banked dL| per pair, nats
K_A_N = 10                  # K-A units: N12's K0 carriers (the first 10 real units by id whose pre point is T_14)
KNOWN_T = 14
IDENTITY_T = 13
N_JVP_ID = 16               # I-JVP: the first 16 unit windows of T_13 again
ORDER = [14, 13, 12, 11, 10, 9, 8, 7, 5]
FD_CEIL_REL = 1e-6          # G-FD ceiling: every pair's dL_fd / (1 - FD_SCALE) equal to its prediction within this
INSPECT_REL = 1e-3          # the inspection of N12's ceiling: median |meas - pred| / |pred| at least this (not a copy)
SEED = 0
READOUTS = {                # readout -> (prediction key, measurement key, prediction scale)
    "window": ("pred_bf16", "dL", 1.0),
    "fd": ("pred_bf16", "dL_fd", 1.0 - FD_SCALE),
    "own": ("pred_jvp_own", "dL_own", 1.0),
    "window_jvp": ("pred_jvp", "dL", 1.0),
}
PROBE_KEY = {"window": "probe", "fd": "probe", "own": "probe_own", "window_jvp": "probe"}


# ----------------------------------------------------------------------------------------------- the plan
def unit_windows(u: dict) -> list:
    """[(window, input indices)] of the unit's valid exemplar positions, windows ascending (N12's ablation rows)."""
    byw = {}
    for e in u["tok"]:
        i = 1 + e % m2.WINDOW
        if i <= m2.WINDOW - 1:
            byw.setdefault(e // m2.WINDOW, []).append(i)
    return [(w, byw[w]) for w in sorted(byw)]


def plan(units: list, n12_recs: list) -> dict:
    """Per pre point: the real units, their four ablated poles (N12's abl order) and their unit windows."""
    by = {r["unit"]: r for r in n12_recs if r["kind"] == "real"}
    pts = {}
    for u in units:
        if u["kind"] != "real":
            continue
        r = by[u["unit"]]
        poles = [m2.parse_pole(p[0]) for p in r["abl"]]
        assert len(poles) == N_POLES and poles[0][1:] == tuple(u["carrier"]), u["unit"]
        assert all(p[0] == u["layer"] for p in poles), u["unit"]
        t = u["roles"]["pre"]
        assert t == r["t_pre"], u["unit"]
        pts.setdefault(t, []).append({"unit": u["unit"], "half": u["half"], "layer": u["layer"],
                                      "poles": [[p[1], p[2]] for p in poles], "uw": unit_windows(u)})
    order = [t for t in ORDER if t in pts]
    assert sorted(order) == sorted(pts), (order, sorted(pts))
    return {t: sorted(pts[t], key=lambda x: x["unit"]) for t in order}


def counts(pl: dict) -> dict:
    out = {}
    for t, items in pl.items():
        uw = sum(len(x["uw"]) for x in items)
        out[t] = {"units": len(items), "unit_windows": uw, "plain_rows": 10 * uw, "jvp_rows": N_POLES * uw}
    return out


# ----------------------------------------------------------------------------------------------- per-pair records
def pair_values(lc: np.ndarray, rows: dict, idxs_by_w: dict) -> dict:
    """lc: {w: clean losses [2048] fp32}; rows: {(kind, q, w): losses [2048] (or tangents for jvp)}; idxs_by_w: {w:
    input indices}. Returns per q the window and own sums (fp64 over fp32 differences), and the probe."""
    out = {}
    ws = sorted(lc)
    for q in range(N_POLES):
        v = {}
        for kind, key in (("zero", "dL"), ("fd", "dL_fd")):
            if (kind, q, ws[0]) not in rows:
                continue
            d = [rows[(kind, q, w)].astype(np.float64) - lc[w].astype(np.float64) for w in ws]
            v[key] = float(sum(x.sum() for x in d))
            v[key + "_own"] = float(sum(x[idxs_by_w[w]].sum() for x, w in zip(d, ws)))
        if ("jvp", q, ws[0]) in rows:
            v["pred_jvp"] = float(sum(rows[("jvp", q, w)].astype(np.float64).sum() for w in ws))
            v["pred_jvp_own"] = float(sum(rows[("jvp", q, w)].astype(np.float64)[idxs_by_w[w]].sum() for w in ws))
        out[q] = v
    d = [rows[("probe", 0, w)].astype(np.float64) - lc[w].astype(np.float64) for w in ws]
    pr = {"probe": float(sum(x.sum() for x in d)),
          "probe_own": float(sum(x[idxs_by_w[w]].sum() for x, w in zip(d, ws)))}
    return {"pairs": out, "probe": pr}


def causal_ok(lc: np.ndarray, lr: np.ndarray, i0: int) -> tuple[bool, bool]:
    """G-CAUSAL for one row: losses at input indices j < i0 bitwise equal to the clean row's (pass side); the failing
    input includes j = i0, whose loss the change reaches, so it must differ there."""
    return bool(np.array_equal(lc[:i0], lr[:i0])), bool(np.array_equal(lc[:i0 + 1], lr[:i0 + 1]))


# ----------------------------------------------------------------------------------------------- statistics
def as_recs(pairs: list, pk: str, mk: str, scale: float = 1.0) -> list:
    """N12's record shape for n12.validation: one record per unit with abl [[pole, prediction, measurement]]."""
    by = {}
    for p in pairs:
        by.setdefault(p["unit"], []).append([p["pole"], scale * p[pk], p[mk]])
    return [{"abl": v} for _, v in sorted(by.items())]


def validation(pairs: list, readout: str) -> dict:
    """N12's G-VAL, run by N12's own code (n12.validation: Spearman, bootstrap over units, shuffle floor, sign-flipped
    failing input, ceiling flag), on one readout. G-FD adds its own ceiling (the measurement equal to the prediction)."""
    pk, mk, sc = READOUTS[readout]
    v = n12.validation(as_recs(pairs, pk, mk, sc))
    v["readout"], v["prediction"], v["measurement"], v["scale"] = readout, pk, mk, sc
    if readout == "fd":
        pr = np.array([sc * p[pk] for p in pairs])
        me = np.array([p[mk] for p in pairs])
        rel = np.abs(me - pr) / np.maximum(np.abs(pr), 1e-30)
        v["above_ceiling"] = bool(rel.max() < FD_CEIL_REL)
        v["ceiling_rule"] = f"every |dL_fd - 0.1 pred| / |0.1 pred| < {FD_CEIL_REL}"
        v["max_rel_dev"] = float(rel.max())
    me = np.array([p[mk] for p in pairs], np.float64)
    pr = np.array([sc * p[pk] for p in pairs], np.float64)
    v["sign_agreement"] = float(np.mean(np.sign(me) == np.sign(pr)))
    v["median_abs_meas"], v["median_abs_pred"] = float(np.median(np.abs(me))), float(np.median(np.abs(pr)))
    return v


def room(pairs: list, readout: str, noise: dict | None = None, reps: int = N_ROOM, seed: int = SEED) -> dict:
    """G-ROOM: could this readout pass with the measured floor? A perfect instrument (measurement = prediction) plus each
    unit's own probe change with a random sign per pair; the median Spearman over `reps` replicates must reach
    ROOM_RHO_MIN. `noise` replaces the probe values (the registered failing input: N12's bf16 measured change of the
    carrier, a noise-dominated quantity)."""
    pk, _, sc = READOUTS[readout]
    pr = np.array([sc * p[pk] for p in pairs], np.float64)
    nz = np.array([(noise[p["unit"]] if noise is not None else p[PROBE_KEY[readout]]) for p in pairs], np.float64)
    rng = np.random.default_rng(seed)
    rho = np.array([n12.spearman(pr, pr + rng.choice([-1.0, 1.0], pr.size) * nz) for _ in range(reps)])
    return {"readout": readout, "median_rho": float(np.median(rho)), "q05": float(np.quantile(rho, 0.05)),
            "rho_min": ROOM_RHO_MIN, "pass": bool(np.median(rho) >= ROOM_RHO_MIN),
            "median_abs_noise": float(np.median(np.abs(nz))), "median_abs_pred": float(np.median(np.abs(pr))),
            "noise": "probe" if noise is None else "planted"}


def bf16_noise(pairs: list) -> dict:
    """G-ROOM's failing input: per unit, N12's banked bf16 measured change of the carrier (|dL| about 0.65 nats at the
    median against a prediction of 0.022)."""
    return {p["unit"]: abs(p["meas_bf16"]) for p in pairs if p["rank"] == 0}


# ----------------------------------------------------------------------------------------------- labels
def inspect(pairs: list, readout: str) -> dict:
    """The inspection N12's ceiling calls for (rho > 0.999: "DOES NOT COUNT until inspected"), frozen here before any
    fp32 number exists. In fp32 a near-linear effect can rank-agree above 0.999 without a defect, so the ceiling asks
    only whether the measurement is a copy of the prediction: cleared when no pair's measurement equals its
    prediction and the median relative deviation |meas - pred| / |pred| is at least INSPECT_REL (the bf16 prediction's
    own error is about 1 %, N12 A1). The reader's recount of dL from the banked per-row losses is the third check
    (SPEC section 7) and is not run here."""
    pk, mk, sc = READOUTS[readout]
    pr = np.array([sc * p[pk] for p in pairs], np.float64)
    me = np.array([p[mk] for p in pairs], np.float64)
    rel = np.abs(me - pr) / np.maximum(np.abs(pr), 1e-30)
    return {"n_equal": int((me == pr).sum()), "median_rel_dev": float(np.median(rel)), "rel_min": INSPECT_REL,
            "cleared": bool((me != pr).all() and np.median(rel) >= INSPECT_REL)}


def g_label(val: dict | None, rm: dict | None, insp: dict | None = None) -> str:
    """PASS, FAIL, DOES NOT COUNT (no room, a failing input that passed, or above the ceiling and not cleared by the
    inspection), NOT_RUN."""
    if val is None:
        return "NOT_RUN"
    if rm is not None and not rm["pass"]:
        return "DOES NOT COUNT: no room"
    if val["failing_input"]["pass"]:
        return "DOES NOT COUNT: failing input passed"
    if val["above_ceiling"] and not (insp is not None and insp["cleared"]):
        return "DOES NOT COUNT: above ceiling"
    return "PASS" if val["pass"] else "FAIL"


def both(a: str, b: str) -> str:
    """A label over two halves: PASS only if both pass; FAIL if either fails and neither does not count."""
    if "DOES NOT COUNT" in a or "DOES NOT COUNT" in b:
        return a if "DOES NOT COUNT" in a else b
    if "NOT_RUN" in (a, b):
        return "NOT_RUN"
    return "PASS" if a == b == "PASS" else "FAIL"


def decision(window: str, fd: str, own: str) -> dict:
    """The frozen decision table (SPEC section 1.2)."""
    if window.startswith("DOES NOT COUNT"):
        return {"branch": "D", "reading": "the fp32 readout has no room or failed a check; no claim about the instrument",
                "next": "report the floor; fp64 rows or the own-loss readout"}
    if window == "PASS":
        return {"branch": "A", "reading": "N12's banked bf16 first-order prediction tracks single-neuron zero ablation "
                "read in fp32", "next": "re-read N12's gradient labels from N12's banked records under N12's frozen rules"}
    if fd == "PASS":
        return {"branch": "B", "reading": "the gradient is right for a 10 % change but -h G does not track zero ablation "
                "at these positions; N12's labels stay DOES NOT COUNT", "next": "ablation per pole for removal claims"}
    return {"branch": "C", "reading": "the first-order instrument does not track ablation in fp32; retired",
            "next": "ablation per pole for the competitor comparison", "fd": fd, "own": own}


# ----------------------------------------------------------------------------------------------- freeze, verdict
def read_bank(bank, halves=("freeze", "test")) -> tuple[dict, dict]:
    """A2 (2026-11-08): freeze reads the freeze half only, so the test records need not exist before freeze 2."""
    bank = pathlib.Path(bank)
    jm = json.loads((bank / "n2v" / "metrics.json").read_text())
    return jm, {h: m2.read_jsonl(bank / "n2v" / f"n2v_{h}.jsonl") for h in halves}


def half_block(pairs: list) -> dict:
    out = {"n_pairs": len(pairs), "n_units": len({p["unit"] for p in pairs})}
    has_jvp = all("pred_jvp" in p for p in pairs)
    has_fd = all("dL_fd" in p for p in pairs)
    for ro in READOUTS:
        if ro in ("own", "window_jvp") and not has_jvp or ro == "fd" and not has_fd:
            out[ro] = None
            continue
        val = validation(pairs, ro)
        out[ro] = {"room": room(pairs, ro), "validation": val,
                   "inspection": inspect(pairs, ro) if val["above_ceiling"] and ro != "fd" else None}
    out["room_failing_input"] = room(pairs, "window", noise=bf16_noise(pairs))
    pr = [p for p in pairs if "pred_jvp" in p]
    if pr:
        a = np.array([p["pred_bf16"] for p in pr])
        b = np.array([p["pred_jvp"] for p in pr])
        out["bf16_vs_jvp_prediction"] = {"spearman": n12.spearman(a, b),
                                         "median_rel_diff": float(np.median(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))}
    out["probe_median_abs"] = float(np.median([abs(p["probe"]) for p in pairs if p["rank"] == 0]))
    out["probe_own_median_abs"] = float(np.median([abs(p["probe_own"]) for p in pairs if p["rank"] == 0]))
    return out


def labels(blk: dict) -> dict:
    return {ro: (g_label(blk[ro]["validation"], blk[ro]["room"], blk[ro]["inspection"]) if blk[ro] is not None
                 else "NOT_RUN") for ro in READOUTS}


def freeze(bank) -> dict:
    bank = pathlib.Path(bank)
    jm, recs = read_bank(bank, ("freeze",))
    g = m2.gates_ok(jm)
    assert g["pass"], f"freeze refused: gates {g}"
    assert len({p["unit"] for p in recs["freeze"]}) >= n12.N_MIN, "fewer than 50 freeze units"
    blk = half_block(recs["freeze"])
    assert not blk["room_failing_input"]["pass"], "G-ROOM's failing input passed on the freeze half"
    return {"created_from": str(bank / "n2v" / "n2v_freeze.jsonl"),
            "sha256_n2v_freeze": m2.sha256_file(bank / "n2v" / "n2v_freeze.jsonl"),
            "sha256_job_metrics": m2.sha256_file(bank / "n2v" / "metrics.json"), "gates": g,
            "freeze": blk, "labels_freeze": labels(blk), "rules": {"val_rho_min": n12.VAL_RHO_MIN,
                                                                   "val_ceiling": n12.VAL_RHO_CEIL,
                                                                   "room_rho_min": ROOM_RHO_MIN}}


def verdict(bank, th: dict, n12_banks, n12_thresholds) -> dict:
    """Once, on the test half; on branch A the N12 re-reading. Every value is written whatever the headline."""
    bank = pathlib.Path(bank)
    assert m2.sha256_file(bank / "n2v" / "n2v_freeze.jsonl") == th["sha256_n2v_freeze"], "freeze records changed"
    jm, recs = read_bank(bank)
    g = m2.gates_ok(jm)
    blk = half_block(recs["test"])
    lf, lt = th["labels_freeze"], labels(blk)
    lab = {ro: both(lf[ro], lt[ro]) for ro in READOUTS}
    v = {"gates": g, "test": blk, "labels_freeze": lf, "labels_test": lt, "labels": lab,
         "thresholds_sha256": th.get("_sha256")}
    reasons = [] if g["pass"] else [f"gates: {g['problems'] or 'incomplete'}"]
    if len({p["unit"] for p in recs["test"]}) < n12.N_MIN:
        reasons.append("missing: fewer than 50 test units")
    if blk["room_failing_input"]["pass"]:
        reasons.append("G-ROOM's failing input passed on the test half")
    if reasons:
        v["headline"] = "DOES NOT COUNT: " + "; ".join(reasons)
        return v
    d = decision(lab["window"], lab["fd"], lab["own"])
    v["decision"] = d
    if d["branch"] == "A":
        cleared = any(b is not None and b["window"]["inspection"] is not None and b["window"]["inspection"]["cleared"]
                      for b in (th["freeze"], blk))
        v["n12_reread"] = reread(n12_banks, n12_thresholds, recs["freeze"], recs["test"], cleared)
    v["headline"] = (f"G-VAL window {lab['window']}; G-FD {lab['fd']}; G-VAL own {lab['own']}; branch {d['branch']}"
                     + (f"; N12 re-read: {v['n12_reread']['headline']}" if "n12_reread" in v else ""))
    return v


# ----------------------------------------------------------------------------------------------- the N12 re-reading
def reread(n12_banks, n12_thresholds, n2v_freeze: list, n2v_test: list, ceiling_cleared: bool = False) -> dict:
    """N12's verdict code, unchanged (n12.verdict), on N12's banked records, with one input replaced: the measured
    ablation of each validation pair (abl[i][2]) becomes N2V's fp32 window measurement, on both halves. N12's freeze
    records are left byte-identical (n12.verdict checks their sha256); the freeze half's validation enters through
    the thresholds, as N12's freeze wrote it. No gradient is recomputed: the labels read the banked pull. If N2V's
    inspection cleared the ceiling on the window readout, N12's ceiling is lifted for this call only (its code reads
    n12.VAL_RHO_CEIL at call time), which is what "until inspected" provides for."""
    n12_banks = pathlib.Path(n12_banks)
    th = json.loads(pathlib.Path(n12_thresholds).read_text())
    th["_sha256"] = m2.sha256_file(n12_thresholds)
    fp = {h: {(p["unit"], p["pole"]): p["dL"] for p in ps} for h, ps in (("freeze", n2v_freeze), ("test", n2v_test))}

    def swap(r, h):
        if r["kind"] != "real":
            return r
        r = dict(r)
        r["abl"] = [[a[0], a[1], fp[h][(r["unit"], a[0])]] for a in r["abl"]]
        return r
    ceil = n12.VAL_RHO_CEIL
    if ceiling_cleared:
        n12.VAL_RHO_CEIL = 1.0
    fz = [swap(r, "freeze") for r in m2.read_jsonl(n12_banks / "n12" / "n12_freeze.jsonl")]
    th["validation_freeze"] = n12.validation([r for r in fz if r["kind"] == "real" and "pull" in r])
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="n2v_reread_"))
    (tmp / "n12").mkdir()
    shutil.copy(n12_banks / "n12" / "metrics.json", tmp / "n12" / "metrics.json")
    shutil.copy(n12_banks / "n12" / "n12_freeze.jsonl", tmp / "n12" / "n12_freeze.jsonl")
    with open(tmp / "n12" / "n12_test.jsonl", "w") as f:
        for r in m2.read_jsonl(n12_banks / "n12" / "n12_test.jsonl"):
            f.write(json.dumps(swap(r, "test")) + "\n")
    v = n12.verdict(tmp, th)
    n12.VAL_RHO_CEIL = ceil
    shutil.rmtree(tmp)
    return {"what": "N12's gradient labels re-read with the fp32 window measurement in place of the bf16 one; N12's "
                    "frozen rules and code; status: a dated re-reading (N12's test half was read on 2026-11-06)",
            "ceiling_cleared_by_inspection": ceiling_cleared, "headline": v["headline"], "labels": {k: v["labels"][k] for k in ("H_signal", "H_sel_signal_pb",
                                                                               "H_sel_signal_dt", "H_bridge")},
            "validation_freeze": th["validation_freeze"], "validation_test": v["test"]["validation"]}


# ----------------------------------------------------------------------------------------------- GPU projection
RATES = {"plain_s": (1.18, 1.40),    # fp32 forward row, batch 4, L40S: N6 L1-02 measured 113.4 s for 96 windows
         "jvp_factor": (2.0, 3.0),   # a forward-mode row against a plain row (one extra matmul per matmul)
         "bf16_s": (0.2293, 0.25),   # N12 score-02 measured per forward or ablation row (bf16, batch 8)
         "fixed_s": (200.0, 420.0),  # per point: download, load, fp32 cast (N12 measured 56-416 s fixed per point)
         "start_s": (344.0, 450.0)}  # boot 164, corpus 60, records 120


def project(units: list, n12_recs: list, parts=("core", "fd", "jvp")) -> dict:
    pl = plan(units, n12_recs)
    c = counts(pl)
    per_row = {"core": 6, "fd": 4}
    tot = [RATES["start_s"][0], RATES["start_s"][1]]
    out = {"points": {}, "parts": list(parts)}
    for t, x in c.items():
        uw = x["unit_windows"]
        rows = uw * (per_row["core"] + (per_row["fd"] if "fd" in parts else 0))
        jv = uw * N_POLES if "jvp" in parts else 0
        row = {"unit_windows": uw, "plain_rows": rows, "jvp_rows": jv}
        for j in (0, 1):
            s = RATES["fixed_s"][j] + rows * RATES["plain_s"][j] + jv * RATES["plain_s"][j] * RATES["jvp_factor"][j]
            if t == KNOWN_T:
                ka = sum(len(i["uw"]) for i in pl[t][:K_A_N])
                s += ka * (N_POLES + 2) * RATES["bf16_s"][j]
            if t == IDENTITY_T:
                s += uw * RATES["plain_s"][j] + (N_JVP_ID * N_POLES * RATES["plain_s"][j] * RATES["jvp_factor"][j]
                                                 if "jvp" in parts else 0)
            tot[j] += s
            row[("s_center", "s_high")[j]] = round(s)
        out["points"][t] = row
    out["unit_windows"] = sum(x["unit_windows"] for x in c.values())
    out["gpu_h_center"], out["gpu_h_high"] = round(tot[0] / 3600, 2), round(tot[1] / 3600, 2)
    out["usd_center_g6e2x"], out["usd_high_g6e2x"] = round(out["gpu_h_center"] * 2.24, 2), round(out["gpu_h_high"] * 2.24, 2)
    out["budget_min_hard_kill"] = int(np.ceil(1.3 * out["gpu_h_high"] * 60))
    out["rates"] = RATES
    return out


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    root = pathlib.Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["project", "freeze", "verdict"])
    ap.add_argument("--units", default=str(root / "studies/next/n12/units.json"))
    ap.add_argument("--n12-banks", default=str(root / "studies/next/n12/banks/score-02"))
    ap.add_argument("--n12-thresholds", default=str(root / "studies/next/n12/thresholds.json"))
    ap.add_argument("--parts", default="core,fd,jvp")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "project":
        nb = pathlib.Path(a.n12_banks) / "n12"
        recs = m2.read_jsonl(nb / "n12_freeze.jsonl") + m2.read_jsonl(nb / "n12_test.jsonl")
        r = project(json.loads(pathlib.Path(a.units).read_text())["units"], recs, tuple(a.parts.split(",")))
        print(json.dumps(r, indent=1))
    elif a.cmd == "freeze":
        assert not pathlib.Path(a.out).exists(), f"{a.out} exists; a freeze is written once"
        th = freeze(a.bank)
        pathlib.Path(a.out).write_text(json.dumps(th, indent=1))
        print("N2V_FREEZE_OK", m2.sha256_file(a.out), json.dumps(th["labels_freeze"]))
    else:
        assert not pathlib.Path(a.out).exists(), f"{a.out} exists; the verdict runs once"
        th = json.loads(pathlib.Path(a.thresholds).read_text())
        th["_sha256"] = m2.sha256_file(a.thresholds)
        v = verdict(a.bank, th, a.n12_banks, a.n12_thresholds)
        pathlib.Path(a.out).write_text(json.dumps(v, indent=1))
        print(v["headline"])


if __name__ == "__main__":
    main()
