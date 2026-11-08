"""N12: competition at birth (studies/next/07_PICKS/N12_SPEC.md), with M2's registered follow-up (the twin matched on
the number of distinct exemplar tokens). Numpy only; the job imports torch itself.

Users:
  - the laptop CLI below: `units` (the unit table, before any GPU), `project` (GPU-hours from the plan), `freeze`
    (from the freeze half only, before the test half is read) and `verdict` (once, on the test half);
  - the job `infra/aws_pg/jobs/n12_score.py` (plan, competitor sets, pull, counterfactual ranks, records);
  - the dry run `infra/aws_pg/jobs/n12_dryrun.py`.

    python -m infra.pg.n12 units   --bank <BANK/neurons> --births studies/next/m2/births.json --out studies/next/n12/units.json
    python -m infra.pg.n12 project --units studies/next/n12/units.json
    python -m infra.pg.n12 freeze  --banks studies/next/n12/banks --out studies/next/n12/thresholds.json
    python -m infra.pg.n12 verdict --banks studies/next/n12/banks --thresholds studies/next/n12/thresholds.json \
                                   --out studies/next/n12/verdict_n12.json

Objects (SPEC section 0.5). The births, the halves, E_c, a+, B and the layer rank are M2's (infra/pg/m2.py). Units: each
birth's carrier on E_c ("real"), M2's pseudo-birth twin ("pb") and the new distinct-token twin ("dt"). For a unit at
its pre point T_(k-1):
  C       the competitor set: the 20 poles of the layer with the highest B, B > 0, the unit pole's neuron excluded.
  G_n(i)  dL/dh_n at input index i, L the summed next-token loss of the window, h_n the down-projection input.
  pull    s_q(i) = -sign_q * G_n(i) * q999_q, nats per own-q999 unit of activation (sign +1 for the + pole).
  P_q     mean over E_c of s_q; P'_q the same over the loss-matched positions; S_q = P_q - P'_q (specific pull).
  u       share of C at or above the unit pole under a score (0 = above every competitor).
Counterfactual ranks (layer rank under B, ties against the pole): r0 at T_(k-1), r1 at T_k, r_c with only the pole at
T_k (others at T_(k-1)), r_o with only the others at T_k; F = log10 r_o - log10 r_c (> 0: the pole's own move buys
more rank than the others' move). drop = the median over C of log10 rank(T_k) - log10 rank(T_(k-1)) (> 0: the
competitors lose rank; every competitor starts at a layer rank of 1-21, for real units and twins alike).

Fail loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from infra.pg import m2

K_COMP = 20                 # competitors per unit
N_ABL = 3                   # competitors ablated per real unit, beside the carrier (the top 3 under B)
M_LM = 16                   # loss-matched positions per exemplar
EXCL = 8                    # no loss-matched position within EXCL tokens of any unit position of the window
SEED_DT = 20261106
DT_DRAWS = 2000
N_MIN = 50                  # usable units per half and kind, else DOES NOT COUNT (real) or NOT_RUN (twins)
VAL_RHO_MIN = 0.7           # G-VAL: Spearman of the first-order prediction against the ablation, registered a priori
                            # (head's decision, 2026-11-06, before freeze 1: raised from 0.5)
VAL_RHO_CEIL = 0.999        # above it: DOES NOT COUNT until inspected (the ablation would equal its prediction)
ABL_ID_TOL = 1e-3           # G-ABL-ID: |clean row - main pass| per window, nats
IGRAD_RHO_MIN = 0.999       # I-GRAD: Spearman of the two gradient passes over every captured entry
KNOWN_T = 14                # K0 runs first, at T_14, on the first 10 real carriers whose pre point is T_14
K0_N = 10
K0_TOL_B = 1e-6             # K0: |B_new - B_M2| for each of the 10 carriers (same device, expected bit-identical)
ORDER = [14, 15, 13, 12, 11, 10, 9, 8, 7, 6, 5]
N_BOOT, SEED = 10000, 0
N_SHUF = 1000


# ----------------------------------------------------------------------------------------------- the unit table
def contexts_t15(bank) -> dict:
    """pole -> {T_15 rank: (global position, decoded current token)} from the grid's banked T_15 contexts.jsonl."""
    p = pathlib.Path(bank) / "w3" / "grid" / m2.POINTS[15][1] / "samples" / "contexts.jsonl"
    assert m2.sha256_file(p) == m2.CONTEXTS_SHA[15], "T_15 contexts.jsonl sha256"
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[r["pole"]] = {c["rank"]: (int(c["tok"]), c["token"]) for c in r["topk"]}
    return out


def distinct(ctx: dict, pole: str, ranks: list) -> int:
    """The number of distinct decoded current tokens among a group's positions (the reader's measure, READ_m2 s5)."""
    return len({ctx[pole][r][1] for r in ranks})


def dt_twin(b: dict, d: int, B: dict, ctx: dict, rng) -> dict | None:
    """The twin matched on the number of distinct tokens: a group of another grid pole's own T_15 top-50 (non-echo
    ranks) in the birth's layer, of the birth's size, with exactly d distinct current tokens, dated k by the same rule.
    Candidates as M2's pseudo-birth (sample-0 clustered grid poles of the layer, another neuron, sorted, then a seeded
    permutation). Per candidate DT_DRAWS draws: d distinct tokens uniformly; one position of each; the rest uniformly
    from the chosen tokens' other positions. The first draw dated k is taken. None if no candidate yields one."""
    cands = sorted(q for q in B["clusters"] if m2.parse_pole(q)[0] == b["layer"] and m2.parse_pole(q)[1] != b["neuron"])
    for q in rng.permutation(np.array(cands, dtype=object)):
        ranks = m2.nonecho_ranks(B["clusters"][q])
        if len(ranks) < b["size"]:
            continue
        groups = collections.defaultdict(list)
        for r in ranks:
            groups[ctx[q][r][1]].append(int(r))
        names = sorted(groups)
        if len(names) < d:
            continue
        draws = []
        for _ in range(DT_DRAWS):
            pick = [groups[names[j]] for j in rng.choice(len(names), d, replace=False)]
            if sum(len(p) for p in pick) < b["size"]:
                continue
            first = [p[int(rng.integers(len(p)))] for p in pick]
            rest = [r for p in pick for r in p if r not in first]
            more = [int(x) for x in rng.choice(rest, b["size"] - d, replace=False)] if b["size"] > d else []
            draws.append(sorted(first + more))
        if not draws:
            continue
        g = np.array(draws)
        i = B["row"][q]
        hit = np.flatnonzero(m2.births_matrix(B["a_plus"][:, i, :], B["q999"][:, i], g) == b["k"])
        if hit.size:
            rq = g[hit[0]].tolist()
            lq, nq, sq = m2.parse_pole(q)
            return {"pole": q, "layer": lq, "neuron": nq, "sign": sq, "row": i, "ranks": rq,
                    "tok": B["tok15"][i, rq].tolist(), "d": distinct(ctx, q, rq), "draw": int(hit[0] + 1)}
    return None


def build_units(bank, births_json: dict) -> dict:
    B = m2.load_bank(bank)
    ctx = contexts_t15(bank)
    units, n_dt = [], collections.Counter()
    for b in births_json["births"]:
        for r, t in zip(b["ranks"], b["tok"]):
            assert ctx[b["pole"]][r][0] == t, f"{b['id']}: contexts.jsonl position differs from births.json"
        d = distinct(ctx, b["pole"], b["ranks"])
        base = {"half": b["half"], "layer": b["layer"], "birth": b["id"], "k": b["k"], "size": b["size"], "d_real": d}
        units.append({**base, "unit": b["id"], "kind": "real", "pole": b["pole"], "carrier": [b["neuron"], b["sign"]],
                      "row": b["row"], "tok": b["tok"], "ranks": b["ranks"], "d": d,
                      "roles": {"pre": b["k"] - 1, "birth": b["k"]}})
        if b["pb"] is not None:
            p = b["pb"]
            units.append({**base, "unit": b["id"] + ":pb", "kind": "pb", "pole": p["pole"],
                          "carrier": [p["neuron"], p["sign"]], "row": p["row"], "tok": p["tok"], "ranks": p["ranks"],
                          "d": distinct(ctx, p["pole"], p["ranks"]), "roles": {"pre": b["k"] - 1, "birth": b["k"]}})
        t = dt_twin(b, d, B, ctx, np.random.default_rng([SEED_DT, b["layer"], b["neuron"], b["sign"], b["c"]]))
        n_dt[b["half"], t is not None] += 1
        if t is not None:
            assert t["d"] == d and len(t["tok"]) == b["size"] and t["layer"] == b["layer"] and t["neuron"] != b["neuron"]
            units.append({**base, "unit": b["id"] + ":dt", "kind": "dt", "pole": t["pole"],
                          "carrier": [t["neuron"], t["sign"]], "row": t["row"], "tok": t["tok"], "ranks": t["ranks"],
                          "d": t["d"], "draw": t["draw"], "roles": {"pre": b["k"] - 1, "birth": b["k"], "end": 15}})
    for u in units:
        assert all(0 < e < m2.N_WINDOWED and e % m2.WINDOW != 0 for e in u["tok"]), u["unit"]
    cnt = collections.Counter((u["kind"], u["half"]) for u in units)
    return {"rule": "M2's 153 births (studies/next/m2/births.json); real = the carrier on E_c; pb = M2's twin; dt = the "
                    "twin matched on size, layer, birth point and the number of distinct current tokens",
            "births_sha256": births_json["_sha256"], "n_units": {f"{k}_{h}": v for (k, h), v in sorted(cnt.items())},
            "dt_missing": sorted(b["id"] for b in births_json["births"]
                                 if not any(u["unit"] == b["id"] + ":dt" for u in units)),
            "examples": births_json["examples"], "units": units}


# ----------------------------------------------------------------------------------------------- the plan
def plan(units: list) -> dict:
    """Per point: forward windows (every (unit, role) at t) with the (layer, input index, position) entries to capture;
    gradient windows (pre roles) with their entries; the exclusion positions per window (every unit position in it,
    any role); the ablation row count (real pre units: N_ABL + 1 poles per window, plus one clean row per unit)."""
    excl = collections.defaultdict(set)
    for u in units:
        for e in u["tok"]:
            excl[e // m2.WINDOW].add(1 + e % m2.WINDOW)
    pts = {}
    for u in units:
        for role, t in u["roles"].items():
            p = pts.setdefault(t, {"need": {}, "grad": {}, "roles": [], "abl_rows": 0})
            p["roles"].append([u["unit"], role])
            for e in u["tok"]:
                p["need"].setdefault(e // m2.WINDOW, set()).add((u["layer"], 1 + e % m2.WINDOW, e))
                if role == "pre":
                    p["grad"].setdefault(e // m2.WINDOW, set()).add((u["layer"], 1 + e % m2.WINDOW, e))
            if role == "pre" and u["kind"] == "real":
                p["abl_rows"] += (N_ABL + 1) * len({e // m2.WINDOW for e in u["tok"]}) + 1
    for t, p in pts.items():
        p["windows"] = sorted(p["need"])
        p["grad_windows"] = sorted(p["grad"])
        p["need"] = {w: sorted(v) for w, v in p["need"].items()}
        p["grad"] = {w: sorted(v) for w, v in p["grad"].items()}
    order = [t for t in ORDER if t in pts]
    assert sorted(order) == sorted(pts), (order, sorted(pts))
    return {"units": units, "points": {t: pts[t] for t in order},
            "excl": {w: sorted(v) for w, v in excl.items()}}


# ----------------------------------------------------------------------------------------------- per-unit quantities
def competitors(Bs: np.ndarray, n: int, k: int = K_COMP) -> list:
    """The k poles of the layer with the highest B (stable order), B > 0, neuron n (both poles) excluded."""
    s = Bs.astype(np.float64).copy()
    s[n, :] = -1.0
    idx = np.argsort(-s.ravel(), kind="stable")[:k]
    out = [[int(i // 2), int(i % 2)] for i in idx if s.ravel()[i] > 0]
    assert len(out) == k, f"only {len(out)} poles with B > 0"
    return out


def layer_rank(Bs: np.ndarray, n: int, s: int, value: float | None = None) -> int:
    """1 + the number of the layer's other poles with B >= value (value defaults to the pole's own B)."""
    v = Bs[n, s] if value is None else value
    return int((Bs >= v).sum() - (Bs[n, s] >= v)) + 1


def cf_ranks(B_pre: np.ndarray, B_birth: np.ndarray, n: int, s: int) -> dict:
    """r0, r1 and the two one-sided counterfactual ranks; F = log10 r_o - log10 r_c."""
    r0 = layer_rank(B_pre, n, s)
    r1 = layer_rank(B_birth, n, s)
    rc = layer_rank(B_pre, n, s, B_birth[n, s])
    ro = layer_rank(B_birth, n, s, B_pre[n, s])
    return {"r0": r0, "r1": r1, "r_c": rc, "r_o": ro, "F": float(np.log10(ro) - np.log10(rc)),
            "move": float(np.log10(r0) - np.log10(r1))}


def pull(G: np.ndarray, q999: np.ndarray) -> np.ndarray:
    """G [m, N] = dL/dh at m positions -> s [m, N, 2] = -sign * G * q999 (sign +1 for the + pole, -1 for the - pole);
    a pole with no token of its sign (q999 NaN, A4) gets 0."""
    q = np.nan_to_num(q999.astype(np.float64), nan=0.0)
    G = G.astype(np.float64)
    return np.stack([-G * q[None, :, 0], G * q[None, :, 1]], -1)


def loss_matched(ell: np.ndarray, i: int, excl: list, m: int | None = None, ex: int | None = None) -> list:
    """The m (M_LM) input indices j in [1, 2047] nearest the exemplar's next-token loss ell[i], none within ex (EXCL)
    of any unit position of the window; ties by smaller j. ell[j] = CE of predicting input j + 1 from input index j."""
    m = M_LM if m is None else m
    ex = EXCL if ex is None else ex
    n = ell.size                                  # 2048 losses for input indices 0..2047
    ok = np.ones(n, bool)
    ok[0] = False
    for p in excl:
        ok[max(0, p - ex):min(n, p + ex + 1)] = False
    cand = np.flatnonzero(ok)
    assert cand.size >= m, f"{cand.size} loss-matched candidates < {m}"
    o = np.lexsort((cand, np.abs(ell[cand] - ell[i])))
    return cand[o[:m]].tolist()


def u_share(score: np.ndarray, comp: list, pole: list) -> float:
    """The share of the competitors (other than the pole) whose score is at or above the pole's."""
    n, s = pole
    others = [c for c in comp if c != [n, s]]
    return float(np.mean([score[a, b] >= score[n, s] for a, b in others]))


def order_ranks(v) -> list:
    """Rank of each entry, 1 = the highest value; ties by position (stable). A1: I-GRAD's S order of a unit."""
    o = np.argsort(-np.asarray(v, np.float64), kind="stable")
    r = np.empty(o.size, np.int64)
    r[o] = np.arange(1, o.size + 1)
    return r.tolist()


def u_resolution(recs: list, noise: float) -> dict:
    """A1: u_S does not resolve a carrier whose specific pull S lies within the backward's noise of its nearest
    competitor's. Reported, not a gate: of the real units with a pull record, how many have that gap <= noise."""
    R = [r for r in recs if r["kind"] == "real" and "pull" in r]
    gap = np.array([min(abs(c["S"] - r["pull"]["S"]) for c in r["comp"]) for r in R])
    return {"noise_max_abs_dS": float(noise), "n_real": len(R), "n_within_noise": int((gap <= noise).sum()),
            "min_gap": float(gap.min()), "median_gap": float(np.median(gap))}


SCORE01_IGRAD = ("score-01, nondeterministic backward at T_13: 4 of 98 pre-role units changed u_S, Spearman 0.999642, "
                 "exact share 0.2129; the size of the change was not recorded and its second pass was not banked "
                 "(studies/next/n12/diag/score-01/metrics.json -> gates.I_GRAD)")


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    o = np.argsort(x, kind="stable")
    r = np.empty(x.size)
    r[o] = np.arange(x.size, dtype=np.float64)
    xs = x[o]
    _, start, cnt = np.unique(xs, return_index=True, return_counts=True)
    for s0, c in zip(start, cnt):
        if c > 1:
            r[o[s0:s0 + c]] = s0 + (c - 1) / 2.0
    return r


def unit_record(u: dict, pre: dict, birth: dict, grad: dict | None, abl: list | None, matched: list | None) -> dict:
    """One line of n12_<half>.jsonl. pre/birth: {"t", "B" [N, 2]}; grad: {"P", "Pp" [N, 2], "n_valid", "dell"};
    abl: [[pole, dL_pred, dL_meas]]; matched: the K_MATCH matched poles [[n, s]] (real units)."""
    n, s = u["carrier"]
    lyr = u["layer"]
    comp = competitors(pre["B"], n)
    rec = {"unit": u["unit"], "kind": u["kind"], "half": u["half"], "birth": u["birth"], "layer": lyr, "k": u["k"],
           "pole": u["pole"], "size": len(u["tok"]), "d": u["d"], "d_real": u["d_real"],
           "t_pre": pre["t"], "t_birth": birth["t"], "cf": cf_ranks(pre["B"], birth["B"], n, s),
           "B_pre": float(pre["B"][n, s]), "B_birth": float(birth["B"][n, s]),
           "bar_pre": float(np.sort(np.delete(pre["B"].ravel(), 2 * n + s))[-m2.TOP])}
    rows = []
    for a, b in comp:
        rows.append({"pole": m2.pole_id(lyr, a, b), "B_pre": float(pre["B"][a, b]), "B_birth": float(birth["B"][a, b]),
                     "rank_pre": layer_rank(pre["B"], a, b), "rank_birth": layer_rank(birth["B"], a, b)})
    rec["comp"] = rows
    rec["gain"] = [float(np.log10(r["rank_pre"]) - np.log10(r["rank_birth"])) for r in rows]
    rec["drop"] = float(-np.median(rec["gain"]))      # the competitors' median loss of layer rank, in decades
    rec["gain_pole"] = float(np.log10(rec["cf"]["r0"]) - np.log10(rec["cf"]["r1"]))
    if grad is not None:
        P, Pp = grad["P"], grad["Pp"]
        S = P - Pp
        for r, (a, b) in zip(rows, comp):
            r.update(P=float(P[a, b]), Pp=float(Pp[a, b]), S=float(S[a, b]))
        rec["pull"] = {"P": float(P[n, s]), "Pp": float(Pp[n, s]), "S": float(S[n, s]),
                       "u_S": u_share(S, comp, [n, s]), "u_P": u_share(P, comp, [n, s]),
                       "u_Pp": u_share(Pp, comp, [n, s]), "n_valid": grad["n_valid"], "dell": grad["dell"],
                       "rho_S_gain": spearman([r["S"] for r in rows], rec["gain"]),
                       "rho_S_Bpre": spearman([r["S"] for r in rows], [r["B_pre"] for r in rows])}
        # the ceiling: the realized gain as the score (the oracle the pull tries to foretell)
        g = np.full(P.shape, -np.inf)
        for r, (a, b) in zip(rows, comp):
            g[a, b] = np.log10(r["rank_pre"]) - np.log10(r["rank_birth"])
        g[n, s] = rec["gain_pole"]
        rec["pull"]["u_gain"] = u_share(g, comp, [n, s])
        if matched is not None:
            rec["pull"]["u_matched"] = [u_share(S, comp, m) for m in matched]
    if abl is not None:
        rec["abl"] = abl
    return rec


# ----------------------------------------------------------------------------------------------- statistics
def boot_mean(x, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    x = np.asarray(x, np.float64)
    assert x.size > 0, "nothing to summarize"
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"n": int(x.size), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def floor_u(um: np.ndarray, reps: int = m2.N_FLOOR_REPS, seed: int = SEED) -> dict:
    """um [units, K] = u of each matched pole; replicate r draws one per unit; its mean u. Lower is better, so the
    floor's 2.5th percentile is the bar."""
    rng = np.random.default_rng(seed)
    nu, k = um.shape
    v = np.array([um[np.arange(nu), rng.integers(0, k, nu)].mean() for _ in range(reps)])
    return {"mean": float(v.mean()), "q025": float(np.quantile(v, 0.025)), "min": float(v.min()), "reps": reps}


def validation(recs: list, n_shuf: int = N_SHUF, seed: int = SEED) -> dict:
    """G-VAL over the (real unit, ablated pole) pairs: Spearman of the first-order prediction -h*G against the
    measured ablation; bootstrap over units; the shuffle floor; the failing input (the prediction with its sign
    flipped). Pass: rho >= VAL_RHO_MIN and the lower bound above the shuffle's 97.5th percentile."""
    pairs = [(i, p[1], p[2]) for i, r in enumerate(recs) for p in r.get("abl", [])]
    assert pairs, "no ablation pairs"
    ui = np.array([x[0] for x in pairs])
    pr = np.array([x[1] for x in pairs], np.float64)
    me = np.array([x[2] for x in pairs], np.float64)
    rho = spearman(pr, me)
    assert rho is not None, "constant prediction or ablation"
    rng = np.random.default_rng(seed)
    shuf = np.array([spearman(rng.permutation(pr), me) for _ in range(n_shuf)])
    units = np.unique(ui)
    bs = []
    for _ in range(1000):
        pick = rng.choice(units, units.size)
        idx = np.concatenate([np.flatnonzero(ui == j) for j in pick])
        v = spearman(pr[idx], me[idx])
        if v is not None:
            bs.append(v)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    slope = float((pr * me).sum() / (pr * pr).sum()) if (pr * pr).sum() > 0 else None
    out = {"n_pairs": len(pairs), "n_units": int(units.size), "rho": rho, "lo": float(lo), "hi": float(hi),
           "shuffle_q975": float(np.quantile(shuf, 0.975)), "slope_through_origin": slope,
           "rho_min": VAL_RHO_MIN, "ceiling": VAL_RHO_CEIL}
    out["pass"] = bool(rho >= VAL_RHO_MIN and lo > out["shuffle_q975"])
    out["above_ceiling"] = bool(rho > VAL_RHO_CEIL)
    fr = spearman(-pr, me)
    out["failing_input"] = {"what": "the prediction with its sign flipped", "rho": fr,
                            "pass": bool(fr >= VAL_RHO_MIN)}
    return out


def by_birth(recs: list, kind: str) -> dict:
    return {r["birth"]: r for r in recs if r["kind"] == kind}


def half_summary(recs: list, half: str) -> dict:
    """Every number one half gives under the frozen definitions. No label."""
    assert all(r["half"] == half for r in recs), "records of another half"
    R, T, D = by_birth(recs, "real"), by_birth(recs, "pb"), by_birth(recs, "dt")
    ids = sorted(R)
    out = {"half": half, "n": {"real": len(R), "pb": len(T), "dt": len(D)}}
    F = np.array([R[i]["cf"]["F"] for i in ids])
    out["F_real"] = boot_mean(F)
    out["move_real"] = boot_mean([R[i]["cf"]["move"] for i in ids])
    out["dc_real"] = boot_mean([np.log10(R[i]["cf"]["r0"]) - np.log10(R[i]["cf"]["r_c"]) for i in ids])
    out["do_real"] = boot_mean([np.log10(R[i]["cf"]["r0"]) - np.log10(R[i]["cf"]["r_o"]) for i in ids])
    out["hits"] = {k: {role: m2.wilson(int(sum(r["cf"][key] <= m2.TOP for r in X.values())), max(len(X), 1))
                       for role, key in (("pre", "r0"), ("birth", "r1"), ("own_only", "r_c"), ("others_only", "r_o"))}
                   for k, X in (("real", R), ("pb", T), ("dt", D))}
    out["drop_real"] = boot_mean([R[i]["drop"] for i in ids])
    out["drop_pb"] = boot_mean([T[i]["drop"] for i in sorted(T)]) if T else None
    out["drop_dt"] = boot_mean([D[i]["drop"] for i in sorted(D)]) if D else None
    out["F_pb"] = boot_mean([T[i]["cf"]["F"] for i in sorted(T)]) if T else None
    out["F_dt"] = boot_mean([D[i]["cf"]["F"] for i in sorted(D)]) if D else None
    tw = [i for i in ids if i in T]
    out["pair_F_real_pb"] = m2.paired(F[[ids.index(i) for i in tw]], [T[i]["cf"]["F"] for i in tw]) if tw else None
    dw = [i for i in ids if i in D]
    out["pair_F_real_dt"] = m2.paired(F[[ids.index(i) for i in dw]], [D[i]["cf"]["F"] for i in dw]) if dw else None
    out["pair_drop_real_pb"] = m2.paired([R[i]["drop"] for i in tw], [T[i]["drop"] for i in tw]) if tw else None
    out["pair_drop_real_dt"] = m2.paired([R[i]["drop"] for i in dw], [D[i]["drop"] for i in dw]) if dw else None
    # the gradient readouts (real units with a pull record)
    gr = [i for i in ids if "pull" in R[i]]
    if gr:
        out["u_S_real"] = boot_mean([R[i]["pull"]["u_S"] for i in gr])
        out["u_P_real"] = boot_mean([R[i]["pull"]["u_P"] for i in gr])
        out["u_Pp_real"] = boot_mean([R[i]["pull"]["u_Pp"] for i in gr])
        out["u_gain_real"] = boot_mean([R[i]["pull"]["u_gain"] for i in gr])
        out["floor_u"] = floor_u(np.array([R[i]["pull"]["u_matched"] for i in gr]))
        rho = [R[i]["pull"]["rho_S_gain"] for i in gr if R[i]["pull"]["rho_S_gain"] is not None]
        out["rho_S_gain_real"] = boot_mean(rho)
        out["rho_S_gain_real"]["n_undefined"] = len(gr) - len(rho)
        rb = [R[i]["pull"]["rho_S_Bpre"] for i in gr if R[i]["pull"]["rho_S_Bpre"] is not None]
        out["rho_S_Bpre_real"] = boot_mean(rb)
        tg = [i for i in gr if i in T and "pull" in T[i]]
        out["pair_u_pb_real"] = (m2.paired([T[i]["pull"]["u_S"] for i in tg], [R[i]["pull"]["u_S"] for i in tg])
                                 if tg else None)
        dg = [i for i in gr if i in D and "pull" in D[i]]
        out["pair_u_dt_real"] = (m2.paired([D[i]["pull"]["u_S"] for i in dg], [R[i]["pull"]["u_S"] for i in dg])
                                 if dg else None)
        rt = [T[i]["pull"]["rho_S_gain"] for i in sorted(T) if "pull" in T[i] and T[i]["pull"]["rho_S_gain"] is not None]
        out["rho_S_gain_pb"] = boot_mean(rt) if rt else None
        out["validation"] = validation([R[i] for i in gr])
        out["dell_median"] = float(np.median([R[i]["pull"]["dell"] for i in gr]))
    # part B: the distinct-token twin under B at the pre role (exactly as M2's twins), and its construction control
    out["pair_hit_real_dt"] = (m2.paired([R[i]["cf"]["r0"] <= m2.TOP for i in dw], [D[i]["cf"]["r0"] <= m2.TOP for i in dw])
                               if dw else None)
    bw = [i for i in dw if i in T]
    out["pair_hit_dt_pb"] = (m2.paired([D[i]["cf"]["r0"] <= m2.TOP for i in bw], [T[i]["cf"]["r0"] <= m2.TOP for i in bw])
                             if bw else None)
    out["pair_hit_real_pb"] = (m2.paired([R[i]["cf"]["r0"] <= m2.TOP for i in tw], [T[i]["cf"]["r0"] <= m2.TOP for i in tw])
                               if tw else None)
    out["bar_median"] = {k: float(np.median([r["bar_pre"] for r in X.values()])) if X else None
                         for k, X in (("real", R), ("pb", T), ("dt", D))}
    ends = [r for r in D.values() if "end" in r]
    out["dt_end_hits"] = (m2.wilson(int(sum(r["end"]["rank"] <= m2.TOP for r in ends)), len(ends))
                          if ends else None)
    out["dt_end_opposite_hits"] = (m2.wilson(int(sum(r["end"]["rank_opposite"] <= m2.TOP for r in ends)), len(ends))
                                   if ends else None)
    return out


def label_lower(stat: dict, bar: float) -> dict:
    """Lower is better (u). SUPPORTED: upper bound below the bar; REFUTED: point at or above it; else UNDECIDED."""
    lab = "SUPPORTED" if stat["hi"] < bar else ("REFUTED" if stat["mean"] >= bar else "UNDECIDED")
    return {"label": lab, **stat, "bar": float(bar)}


def label_sign(stat: dict, names: tuple) -> str:
    lo, hi = stat["lo"], stat["hi"]
    return names[0] if lo > 0 else (names[2] if hi < 0 else names[1])


SEL = ("ABOVE_SELECTION", "AT_SELECTION", "BELOW_SELECTION")


def room_check() -> dict:
    """The frozen labels reach every branch on planted outcomes of the test half's size (71 units)."""
    n = 71
    up, dn, eq = np.ones(n), -np.ones(n), np.r_[np.ones(n // 2), -np.ones(n - n // 2)]
    r = {"F_up": label_sign(boot_mean(up), ("OWN_RISE", "MIXED", "OTHERS_FALL")),
         "F_down": label_sign(boot_mean(dn), ("OWN_RISE", "MIXED", "OTHERS_FALL")),
         "F_mixed": label_sign(boot_mean(eq), ("OWN_RISE", "MIXED", "OTHERS_FALL")),
         "u_low": label_lower(boot_mean(np.zeros(n)), 0.4)["label"],
         "u_high": label_lower(boot_mean(np.full(n, 0.6)), 0.4)["label"],
         "pair_above": label_sign(m2.paired(np.ones(n), np.zeros(n)), SEL),
         "pair_below": label_sign(m2.paired(np.zeros(n), np.ones(n)), SEL),
         "pair_at": label_sign(m2.paired(np.ones(n), np.ones(n)), SEL)}
    ok = r == {"F_up": "OWN_RISE", "F_down": "OTHERS_FALL", "F_mixed": "MIXED", "u_low": "SUPPORTED",
               "u_high": "REFUTED", "pair_above": "ABOVE_SELECTION", "pair_below": "BELOW_SELECTION",
               "pair_at": "AT_SELECTION"}
    return {"pass": bool(ok), "labels": r}


def identity_ok(recs: list) -> dict:
    """The identity of the decomposition: every record's identity readout (T_k replaced by T_(k-1)) is r_c = r_o = r0,
    F = 0, written by the job from the pre arrays alone."""
    bad = [r["unit"] for r in recs if r["cf_identity"] != {"r_c": r["cf"]["r0"], "r_o": r["cf"]["r0"], "F": 0.0}]
    return {"pass": not bad, "bad": bad[:10], "n": len(recs)}


def gates_ok(jm: dict) -> dict:
    return m2.gates_ok(jm)


def freeze(banks) -> dict:
    banks = pathlib.Path(banks)
    jm = json.loads((banks / "n12" / "metrics.json").read_text())
    g = gates_ok(jm)
    assert g["pass"], f"freeze refused: gates {g}"
    fz = m2.read_jsonl(banks / "n12" / "n12_freeze.jsonl")
    s = half_summary(fz, "freeze")
    assert s["n"]["real"] >= N_MIN, f"freeze half has {s['n']['real']} real units, needs {N_MIN}"
    assert s["move_real"]["lo"] > 0, f"no movement to decompose on the freeze half: {s['move_real']}"
    assert identity_ok(fz)["pass"], "the decomposition's identity failed on the freeze half"
    rc = room_check()
    assert rc["pass"], f"room check failed: {rc}"
    s["u_S_resolution"] = {**u_resolution(fz, jm["gates"]["I_GRAD"]["max_abs_dS"]), "score01": SCORE01_IGRAD}
    return {"created_from": str(banks / "n12" / "n12_freeze.jsonl"),
            "sha256_n12_freeze": m2.sha256_file(banks / "n12" / "n12_freeze.jsonl"),
            "sha256_job_metrics": m2.sha256_file(banks / "n12" / "metrics.json"),
            "floor_u_freeze": s.get("floor_u"), "validation_freeze": s.get("validation"),
            "n_min": N_MIN, "val_rho_min": VAL_RHO_MIN, "room": rc, "gates": g, "discovery": s}


def verdict(banks, th: dict) -> dict:
    """Once, on the test half, with the frozen thresholds. Every value is written whatever the headline."""
    banks = pathlib.Path(banks)
    assert m2.sha256_file(banks / "n12" / "n12_freeze.jsonl") == th["sha256_n12_freeze"], "freeze records changed"
    jm = json.loads((banks / "n12" / "metrics.json").read_text())
    g = gates_ok(jm)
    te = m2.read_jsonl(banks / "n12" / "n12_test.jsonl")
    s = half_summary(te, "test")
    v = {"gates": g, "test": s, "thresholds_sha256": th.get("_sha256"), "labels": {}}
    reasons = [] if g["pass"] else [f"gates: {g['problems'] or 'incomplete'}"]
    if s["n"]["real"] < N_MIN:
        reasons.append(f"missing: {s['n']['real']} real test units < {N_MIN}")
    if not s["move_real"]["lo"] > 0:
        reasons.append(f"no movement to decompose: {s['move_real']}")
    if not identity_ok(te)["pass"]:
        reasons.append("the decomposition's identity failed")
    L = v["labels"]
    L["H_cf"] = {"label": label_sign(s["F_real"], ("OWN_RISE", "MIXED", "OTHERS_FALL")), **s["F_real"]}
    for k, key in (("pb", "pair_drop_real_pb"), ("dt", "pair_drop_real_dt")):
        pr = s[key]
        L[f"H_away_sel_{k}"] = ({"label": "NOT_RUN"} if pr is None or pr["n"] < N_MIN else
                                {"label": label_sign(pr, SEL), **pr})
    # the gradient labels: only with the instrument validated on both halves and nothing above its ceiling
    val_f, val_t = th.get("validation_freeze"), s.get("validation")
    grad_reason = None
    if val_f is None or val_t is None:
        grad_reason = "no pull records"
    elif not (val_f["pass"] and val_t["pass"]) or val_f["failing_input"]["pass"] or val_t["failing_input"]["pass"]:
        grad_reason = f"G-VAL failed (freeze rho {val_f['rho']:.3f}, test rho {val_t['rho']:.3f})"
    elif val_f["above_ceiling"] or val_t["above_ceiling"]:
        grad_reason = "G-VAL above its ceiling: DOES NOT COUNT until inspected"
    if grad_reason:
        for k in ("H_signal", "H_sel_signal_pb", "H_sel_signal_dt", "H_bridge"):
            L[k] = {"label": "DOES NOT COUNT", "reason": grad_reason}
    else:
        bar = min(0.5, th["floor_u_freeze"]["q025"], s["floor_u"]["q025"])
        lab = label_lower(s["u_S_real"], bar)
        if s["u_S_real"]["hi"] < s["u_gain_real"]["lo"]:
            lab["label"] = "ABOVE_CEILING"
        L["H_signal"] = lab
        for k, key in (("pb", "pair_u_pb_real"), ("dt", "pair_u_dt_real")):
            pr = s[key]
            L[f"H_sel_signal_{k}"] = ({"label": "NOT_RUN"} if pr is None or pr["n"] < N_MIN else
                                      {"label": label_sign(pr, SEL), **pr})
        L["H_bridge"] = {"label": label_sign(s["rho_S_gain_real"], ("FORETELLS", "UNDECIDED", "CONTRARY")),
                         **s["rho_S_gain_real"]}
    # part B
    pr, pb = s["pair_hit_real_dt"], s["pair_hit_dt_pb"]
    if pr is None or pr["n"] < N_MIN:
        L["H_sel_dt"] = L["H_dt_pb"] = L["H_dt"] = {"label": "NOT_RUN"}
    else:
        L["H_sel_dt"] = {"label": label_sign(pr, SEL), **pr}
        L["H_dt_pb"] = {"label": label_sign(pb, ("HIGHER", "SAME", "LOWER")), **pb}
        sup = L["H_sel_dt"]["label"] in ("AT_SELECTION", "ABOVE_SELECTION") and L["H_dt_pb"]["label"] == "LOWER"
        L["H_dt"] = {"label": "SUPPORTED" if sup else ("REFUTED" if L["H_sel_dt"]["label"] == "BELOW_SELECTION"
                                                       else "UNDECIDED")}
        if s["dt_end_hits"] is None or s["dt_end_hits"][0] < m2.CTRL_MIN or s["dt_end_opposite_hits"][0] >= m2.CTRL_MIN:
            L["H_dt"] = {"label": "DOES NOT COUNT", "reason": f"dt construction control {s['dt_end_hits']}"}
    noise = jm["gates"]["I_GRAD"]["max_abs_dS"]
    v["u_S_resolution"] = {"freeze": u_resolution(m2.read_jsonl(banks / "n12" / "n12_freeze.jsonl"), noise),
                           "test": u_resolution(te, noise), "score01": SCORE01_IGRAD}
    v["cross_device_gex"] = m2.cross_device_gex(jm)
    v["degenerate_poles"] = m2.degenerate_counts(jm)
    if reasons:
        v["headline"] = "DOES NOT COUNT: " + "; ".join(reasons)
    else:
        v["headline"] = (f"H_cf {L['H_cf']['label']}; H_away_sel_pb {L['H_away_sel_pb']['label']}; "
                         f"H_signal {L['H_signal']['label']}; H_bridge {L['H_bridge']['label']}; "
                         f"H_dt {L['H_dt']['label']}")
    return v


# ----------------------------------------------------------------------------------------------- GPU projection
def project(units: list, per_window=(0.2156, 0.25), grad_window=(0.54, 1.0), fixed=(110.0, 160.0)) -> dict:
    """GPU-hours of one launch from the plan, from M2's measured rates (score-06 metrics.json -> measured: 0.2156 s per
    window forward in batches of 8, fixed cost 61-130 s per point). Forward windows at the forward rate; ablation rows
    at the same rate per row; gradient windows (batch 1, forward and backward from the lowest needed layer) at an
    estimated 0.54 s (2.5 x the forward rate) or, in the high case, 1.0 s; the identity re-runs at T_13; boot 164 s,
    corpus 60 s, the S3 reads of the M2 vectors 60 s, the final records 120 s."""
    pl = plan(units)
    out = {"points": {}}
    tot = [164.0 + 60 + 60 + 120] * 2
    for t, p in pl["points"].items():
        nb = -(-len(p["windows"]) // 8)
        na = -(-p["abl_rows"] // 8)
        row = {"windows": len(p["windows"]), "grad_windows": len(p["grad_windows"]), "abl_rows": p["abl_rows"]}
        for j in (0, 1):
            sec = fixed[j] + (nb + na) * 8 * per_window[j] + len(p["grad_windows"]) * grad_window[j]
            if t == m2.IDENTITY_T:
                sec += nb * 8 * per_window[j] + len(p["grad_windows"]) * grad_window[j]
            tot[j] += sec
            row[("s_center", "s_high")[j]] = round(sec)
        out["points"][t] = row
    out["gpu_h_center"] = round(tot[0] / 3600, 2)
    out["gpu_h_high"] = round(tot[1] / 3600, 2)
    out["window_forwards"] = sum(r["windows"] for r in out["points"].values())
    out["grad_windows"] = sum(r["grad_windows"] for r in out["points"].values())
    out["abl_rows"] = sum(r["abl_rows"] for r in out["points"].values())
    out["usd_center_g6e2x"] = round(out["gpu_h_center"] * 2.24, 2)
    out["usd_high_g6e2x"] = round(out["gpu_h_high"] * 2.24, 2)
    out["budget_min_hard_kill"] = int(np.ceil(1.3 * out["gpu_h_high"] * 60))
    return out


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["units", "project", "freeze", "verdict"])
    ap.add_argument("--bank", default="/Users/abaybektursun/projects/interp_research_banks/neurons")
    ap.add_argument("--births", default="studies/next/m2/births.json")
    ap.add_argument("--units", default=None)
    ap.add_argument("--banks", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out", required=False)
    ap.add_argument("--per-window", type=float, nargs=2, default=None, help="project: center, high s per forward row")
    ap.add_argument("--grad-window", type=float, nargs=2, default=None, help="project: center, high s per gradient window")
    ap.add_argument("--fixed", type=float, nargs=2, default=None, help="project: center, high fixed s per point")
    a = ap.parse_args(argv)
    if a.cmd == "units":
        bj = json.loads(pathlib.Path(a.births).read_text())
        bj["_sha256"] = m2.sha256_file(a.births)
        r = build_units(a.bank, bj)
        pathlib.Path(a.out).write_text(json.dumps(r, indent=1))
        print("N12_UNITS_OK", json.dumps(r["n_units"]), "dt missing", len(r["dt_missing"]))
    elif a.cmd == "project":
        kw = {k: tuple(v) for k, v in (("per_window", a.per_window), ("grad_window", a.grad_window),
                                         ("fixed", a.fixed)) if v is not None}
        r = project(json.loads(pathlib.Path(a.units).read_text())["units"], **kw)
        r["rates"] = {"per_window": kw.get("per_window", (0.2156, 0.25)), "grad_window": kw.get("grad_window", (0.54, 1.0)),
                      "fixed": kw.get("fixed", (110.0, 160.0))}
        print(json.dumps(r, indent=1))
    elif a.cmd == "freeze":
        assert not pathlib.Path(a.out).exists(), f"{a.out} exists; a freeze is written once"
        th = freeze(a.banks)
        pathlib.Path(a.out).write_text(json.dumps(th, indent=1))
        print("N12_FREEZE_OK", m2.sha256_file(a.out))
    elif a.cmd == "verdict":
        assert not pathlib.Path(a.out).exists(), f"{a.out} exists; the verdict runs once"
        th = json.loads(pathlib.Path(a.thresholds).read_text())
        th["_sha256"] = m2.sha256_file(a.thresholds)
        v = verdict(a.banks, th)
        pathlib.Path(a.out).write_text(json.dumps(v, indent=1))
        print(v["headline"])


if __name__ == "__main__":
    main()
