"""UNITE Arm 1 offline analysis (studies/unite/PREREG.md §6): gates and freeze values from the J1 banks, and the frozen verdict from the J4 banks. NumPy only; no model.

    python -m infra.pg.unite_analysis pilot --items <j1_items.json> --t15 <dir> --t13 <dir> --t10 <dir> --t7 <dir> --out pilot.json [--field-dir grams_grad]
    python -m infra.pg.unite_analysis verdict --items <j4c_items.json> --freeze <PREREG_FREEZE2.json> --pilot <pilot JSON of the readout> --dirs T_4=<dir> ... --out verdict.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from infra.pg import unite_stats as US

P_BAND = (5, 95)
ENERGY_TOL = 0.25


def layer_of(pole: str) -> int:
    return int(pole[1:3])


def neuron_of(pole: str) -> str:
    return pole[:-1]


class Grams:
    def __init__(self, d, sub: str = "grams"):
        self.d, self.sub, self.c = pathlib.Path(d), sub, {}

    def get(self, unit: str):
        if unit not in self.c:
            p = self.d / self.sub / f"{unit}.npz"
            if not p.exists():
                self.c[unit] = None
            else:
                z = np.load(p)
                self.c[unit] = {k: z[k] for k in z.files}
                self.c[unit]["idx"] = {int(r): i for i, r in enumerate(z["ranks"])}
        return self.c[unit]

    def T(self, unit: str, J, O, field: str = "M"):
        z = self.get(unit)
        return None if z is None or field not in z else US.stat_T(z[field], [z["idx"][r] for r in J], [z["idx"][r] for r in O])


def o_star(job: dict) -> dict:
    return next(o for o in job["older"] if o["c"] == job["o_star"])


def share(z, ranks, restricted_layer=None) -> float:
    """Sum over a job's contexts of the MLP part of the rise, over the sum of all parts. restricted_layer: count only that layer's MLP terms."""
    i = [z["idx"][r] for r in ranks]
    num = z["rise"][i, 0].sum() if restricted_layer is None else z["rise_by_layer"][i, restricted_layer].sum()
    return float(num / z["rise"][i].sum())


# ------------------------------------------------------------------------------------------ J1
def planted_rows(g: Grams, planted: dict, counts: list, shuffle_seed=None) -> list:
    rows = []
    for u, r in planted.items():
        if not r["kept"]:
            continue
        z = g.get(u)
        if z is None:
            continue
        idx = z["idx"]
        J, O = [idx[x] for x in r["J"] if x in idx], [idx[x] for x in r["O"] if x in idx]
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed + r["order"])
            both = rng.permutation(np.array(J + O))
            J, O = both[:len(J)].tolist(), both[len(J):].tolist()
        M = z["M"]
        full = US.stat_T(M, J, O)
        rows.append({"unit": u, "layer": r["layer"], "half": r["half"], "T_count_matched": US.planted_count_matched(M, J, O, counts), "T_full": full["T"], "X": full["X"], "WJ": full["WJ"], "WO": full["WO"],
                     "energy_J": US.mean_energy(M, J), "energy_O": US.mean_energy(M, O), "T_foreign": US.stat_T(z["M_foreign"], J, O)["T"] if "M_foreign" in z else None, "n_J": len(J), "n_O": len(O)})
    return rows


def bands(rows: list, key: str = "T_count_matched") -> dict:
    out = {}
    for ter in (0, 1, 2):
        v = np.array([r[key] for r in rows if US.tercile(r["layer"]) == ter and np.isfinite(r[key])])
        out[str(ter)] = {"lo": float(np.percentile(v, P_BAND[0])), "hi": float(np.percentile(v, P_BAND[1])), "n": int(v.size), "median": float(np.median(v))} if v.size else {"lo": None, "hi": None, "n": 0, "median": None}
    return out


def delta(ceilings: np.ndarray, planted: np.ndarray) -> float:
    return float(np.median(ceilings) - np.percentile(planted, 95))


def room_margin(ceil_by_neuron: dict, planted_vals: np.ndarray, n_boot: int = US.N_BOOT_ROOM, seed: int = 0) -> float:
    """m = 2 * bootstrap SD of the room difference over half A: neurons for the ceilings, pairs for the planted values."""
    keys = sorted(ceil_by_neuron)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        c = np.concatenate([ceil_by_neuron[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        d[b] = delta(c, rng.choice(planted_vals, planted_vals.size))
    return float(2 * d.std(ddof=1))


def pilot(items: dict, dirs: dict, sub: str = "grams", planted_from=None) -> dict:
    """Every Arm 1 gate read from the banks. `sub`='grams_grad' reads the gradient readout's Grams for W noise and room; MLP share always comes from the direct readout's table."""
    half = items["half"]
    g = {t: Grams(d, sub) for t, d in dirs.items()}
    direct = {t: Grams(d, "grams") for t, d in dirs.items()}
    T15 = g["T_15"]
    jobs = items["jobs"]
    planted = json.loads((pathlib.Path(planted_from or dirs["T_15"]) / "planted_contexts.json").read_text())

    # W noise: half A sets; failing input: half B sets, from no job
    noise_poles = [p for p in items["tok"] if layer_of(p) >= 1 and T15.get(p) is not None and "M_noise" in T15.get(p)]
    A_m = [T15.get(p)["M_noise"] for p in noise_poles if half[p] == "A"]
    B_m = [T15.get(p)["M_noise"] for p in noise_poles if half[p] == "B"]
    w_noise = float(np.percentile(US.noise_W(A_m, seed=0), 95))

    def signed_noise(Ms, seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(200):
            M = Ms[rng.integers(len(Ms))]
            out.append(US.within(M, rng.choice(M.shape[0], 7, replace=False)))
        return np.array(out)
    share_no_job = float((signed_noise(B_m, 1) <= w_noise).mean())

    # jobs at T_15 and at birth
    def job_row(job, grams, t, field="M"):
        o = o_star(job)
        r = grams.T(job["pole"], job["ranks"], o["ranks"], field)
        return None if r is None else {"job": job["job"], "pole": job["pole"], "neuron": neuron_of(job["pole"]), "t": t, "half": half[job["pole"]], "layer": layer_of(job["pole"]), **r}
    t15_rows = [r for r in (job_row(j, T15, "T_15") for j in jobs) if r]
    birth_rows = [r for j in jobs if f"T_{j['birth']}" in g and j["birth"] != 15 for r in [job_row(j, g[f"T_{j['birth']}"], f"T_{j['birth']}")] if r]
    B15 = [r for r in t15_rows if r["half"] == "B" and r["layer"] >= 1]
    share_cancelled = float(np.mean([US.cancelled(r, w_noise) for r in B15]))
    noise = US.gate_noise(share_cancelled, share_no_job, w_noise)

    # the room gate
    ceil_rows = []
    for p, c in items["room_ceiling"].items():
        r = T15.T(p, c["sides"][0], c["sides"][1])
        if r is not None and layer_of(p) >= 1:
            ceil_rows.append({"pole": p, "neuron": neuron_of(p), "half": half[p], **r})
    finite = [r for r in ceil_rows if np.isfinite(r["T"])]
    plant = planted_rows(T15, planted, items["job_counts"])
    shuffled = planted_rows(T15, planted, items["job_counts"], shuffle_seed=7)

    def vals(rows, h, key="T_count_matched"):
        return np.array([r[key] for r in rows if r["half"] == h and np.isfinite(r[key])])
    ceil_A = {}
    for r in finite:
        if r["half"] == "A":
            ceil_A.setdefault(r["neuron"], []).append(r["T"])
    ceil_A = {k: np.array(v) for k, v in ceil_A.items()}
    ceil_B = np.array([r["T"] for r in finite if r["half"] == "B"])
    m = room_margin(ceil_A, vals(plant, "A"))
    d_B = delta(ceil_B, vals(plant, "B"))
    d_shuf = delta(ceil_B, vals(shuffled, "B"))
    room = US.gate_room(d_B, m, float(np.median(ceil_B)), d_shuf)
    room.update({"delta_A": delta(np.concatenate(list(ceil_A.values())), vals(plant, "A")), "n_ceilings": {"A": int(sum(map(len, ceil_A.values()))), "B": int(ceil_B.size)},
                 "n_planted": {"A": int(vals(plant, "A").size), "B": int(vals(plant, "B").size)}, "p95_planted_B": float(np.percentile(vals(plant, "B"), 95)), "n_ceiling_W_not_positive": len(ceil_rows) - len(finite)})

    # MLP share (always the direct readout's table). One value per job: at birth when that checkpoint is in the pilot, else at T_15.
    shares, restricted = [], []
    for j in jobs:
        t = f"T_{j['birth']}" if f"T_{j['birth']}" in direct else "T_15"
        z = direct[t].get(j["pole"])
        L = layer_of(j["pole"])
        if z is None or L < 1:
            continue
        shares.append({"job": j["job"], "half": half[j["pole"]], "layer": L, "t": t, "share": share(z, j["ranks"])})
        if L >= 8:
            restricted.append({"job": j["job"], "half": half[j["pole"]], "layer": L, "t": t, "share": share(z, j["ranks"], L - 1)})
    med = float(np.median([s["share"] for s in shares if s["half"] == "B"]))
    res = float(np.median([s["share"] for s in restricted]))
    mlp = US.gate_mlp(med, res)
    mlp.update({"n_jobs_half_B": sum(s["half"] == "B" for s in shares), "n_jobs_restricted": len(restricted), "restricted_share_half_B": float(np.median([s["share"] for s in restricted if s["half"] == "B"])) if any(s["half"] == "B" for s in restricted) else None,
                "share_all_jobs": float(np.median([s["share"] for s in shares]))})

    band = bands(plant)
    energy = {}
    for ter in (0, 1, 2):
        real = [np.sqrt(max(r["X"], 0)) for r in t15_rows if US.tercile(r["layer"]) == ter and r["half"] == "A"]
        ref = float(np.median(real)) if real else float("nan")
        sub_rows = [r for r in plant if US.tercile(r["layer"]) == ter and abs(r["energy_J"] - ref) <= ENERGY_TOL * ref and abs(r["energy_O"] - ref) <= ENERGY_TOL * ref]
        energy[str(ter)] = {"reference": ref, **bands(sub_rows)[str(ter)]}
    by_t = {t: bands(planted_rows(g[t], planted, items["job_counts"])) for t in dirs if t != "T_15"}
    primary = "direct" if mlp["direct_primary"] and room["pass"] else "gradient"
    gates = {"W_noise": noise, "room": room, "MLP_share": mlp}
    return {"gates": gates, "band_count_matched": band, "band_energy_matched": energy, "planted_band_at_other_checkpoints_exploratory": by_t, "planted_rows": plant, "ceiling_rows": ceil_rows,
            "t15_job_rows": t15_rows, "birth_job_rows": birth_rows, "shares": shares, "restricted_shares": restricted, "readout_read": sub, "room_passes": room["pass"], "primary_if_all_pass": primary,
            "freeze2": {"m": m, "W_noise": w_noise, "band_edges": {k: [v["lo"], v["hi"]] for k, v in band.items()}, "primary_readout": primary}}


# ------------------------------------------------------------------------------------------ J4
def planted_by_tercile(rows: list, key: str = "T_count_matched") -> dict:
    out = {0: [], 1: [], 2: []}
    for r in rows:
        if np.isfinite(r[key]):
            out[US.tercile(r["layer"])].append(r[key])
    return {k: np.array(v) for k, v in out.items()}


def planted_median(by_ter: dict, weights: dict, n_boot: int, seed: int = 0) -> tuple:
    """PREREG §13: planted tercile medians weighted by real job counts. Point, and draws from a bootstrap over pairs, within each tercile, so the weights stay the job counts."""
    rng = np.random.default_rng(seed)
    point, draws = 0.0, np.zeros(n_boot)
    for t, w in sorted(weights.items()):
        if w > 0:
            v = by_ter[t]
            assert v.size, f"no planted pair in tercile {t}, which holds real jobs"
            point += w * float(np.median(v))
            draws += w * np.median(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)
    return point, draws


def median_comparison(sel: list, csel: list, by_ter: dict, n_boot: int) -> dict:
    """PREREG §13: median T at birth and at T_15 against the planted median, and (median T - planted) / (median ceiling - planted) with its interval.
    Neurons and pairs are resampled independently, seed 0 each; the ratio's draws pair them by draw number."""
    weights = {t: sum(r["tercile"] == t for r in sel) / len(sel) for t in (0, 1, 2)}
    point, draws = planted_median(by_ter, weights, n_boot)
    planted = {"point": point, "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)), "weights": weights, "n_jobs_weighted": len(sel),
               "tercile_medians": {str(t): float(np.median(v)) if v.size else None for t, v in by_ter.items()}, "n_pairs": {str(t): int(v.size) for t, v in by_ter.items()}, "n_boot": n_boot, "seed": 0}
    live = [r for r in sel if not r["cancelled"]]
    end = [r for r in sel if not r["cancelled_T15"]]
    ceil = [r for r in csel if not r["cancelled"]]

    def group(rows):
        out = {}
        for r in rows:
            out.setdefault(r["neuron"], []).append(r)
        return out
    birth = US.boot_neurons(group(live), lambda rs: float(np.median([r["T"] for r in rs])), n_boot)
    t15 = US.boot_neurons(group(end), lambda rs: float(np.median([r["T_at_T15"] for r in rs])), n_boot)
    res = {"planted_median": planted, "median_T_birth": birth, "median_T_T15": t15,
           "birth_vs_planted": US.rule_median(birth["lo"], birth["hi"], planted["lo"], planted["hi"]), "T15_vs_planted": US.rule_median(t15["lo"], t15["hi"], planted["lo"], planted["hi"])}
    if not ceil:
        res["position_between_planted_and_ceiling"] = None
        return res
    med_c = float(np.median([r["T"] for r in ceil]))
    jobs_by, ceil_by = group(live), group(ceil)
    keys = sorted(set(jobs_by) | set(ceil_by))
    rng = np.random.default_rng(0)
    ratio = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        j = [r["T"] for k in pick for r in jobs_by.get(k, [])]
        c = [r["T"] for k in pick for r in ceil_by.get(k, [])]
        if j and c:
            ratio[b] = (np.median(j) - draws[b]) / (np.median(c) - draws[b])
    ok = ratio[np.isfinite(ratio)]
    res["median_ceiling"] = med_c
    res["position_between_planted_and_ceiling"] = {"point": (birth["point"] - point) / (med_c - point), "lo": float(np.percentile(ok, 2.5)), "hi": float(np.percentile(ok, 97.5)), "n_boot": n_boot, "n_nonfinite_draws": int(n_boot - ok.size),
                                                   "rule": "(median T - planted median) / (median ceiling - planted median), at birth"}
    return res


def verdict(items: dict, dirs: dict, freeze: dict, sub: str = "grams", s1_jobs=None, n_boot: int = US.N_BOOT_VERDICT, planted_rows=None) -> dict:
    """The frozen rules on the 841 primary jobs, with PREREG §13's median comparison deciding H_grown and H_own-route; the secondaries beside them. Uses frozen values only."""
    assert planted_rows, "PREREG §13 needs this readout's own T_15 planted values"
    by_ter = planted_by_tercile(planted_rows)
    g = {t: Grams(d, sub) for t, d in dirs.items()}
    w_noise = freeze["W_noise"]
    band = {int(k): v for k, v in freeze["band_edges"].items()}
    t15 = g["T_15"]
    rows, ceil_seen, ceil_rows = [], set(), []
    for j in items["jobs"]:
        t = f"T_{j['birth']}"
        L = layer_of(j["pole"])
        o = o_star(j)
        own = g[t].T(j["pole"], j["ranks"], o["ranks"])
        assert own is not None, (j["job"], t)
        f = g[t].T(j["pole"], j["ranks"], o["ranks"], "M_foreign")
        x = g[t].T(j["pole"], j["ranks"], o["ranks"], "M_fixed")
        end = t15.T(j["pole"], j["ranks"], o["ranks"])
        all_o = [g[t].T(j["pole"], j["ranks"], oo["ranks"]) for oo in j["older"]]
        pool = sorted({r for oo in j["older"] for r in oo["ranks"]})
        rows.append({"job": j["job"], "pole": j["pole"], "neuron": neuron_of(j["pole"]), "layer": L, "tercile": US.tercile(L), "birth": j["birth"], "primary": j["primary"], "flagged": j["flagged"], "n_J": len(j["ranks"]), "n_O": len(o["ranks"]),
                     **own, "cancelled": US.cancelled(own, w_noise), "T_foreign": f["T"] if f else None, "T_fixed": x["T"] if x else None, "T_at_T15": end["T"], "WJ_T15": end["WJ"], "WO_T15": end["WO"], "cancelled_T15": US.cancelled(end, w_noise),
                     "T_max_over_older": float(np.nanmax([a["T"] for a in all_o])) if np.isfinite([a["T"] for a in all_o]).any() else float("nan"), "T_pooled_older": g[t].T(j["pole"], j["ranks"], pool)["T"], "shallow": L < 3})
        c = j["ceiling"]
        key = (j["pole"], t, c["c"]) if c else None
        if c and key not in ceil_seen:
            ceil_seen.add(key)
            r = g[t].T(j["pole"], c["sides"][0], c["sides"][1])
            ceil_rows.append({"pole": j["pole"], "neuron": neuron_of(j["pole"]), "tercile": US.tercile(L), "t": t, "primary": j["primary"], **r, "cancelled": US.cancelled(r, w_noise)})

    def by_neuron(sel):
        out = {}
        for r in sel:
            out.setdefault(r["neuron"], []).append(r)
        return out

    def inside(r):
        return US.in_band(r["T"], band[r["tercile"]])

    def fshare(rs):
        rs = [r for r in rs if not r["cancelled"]]
        return float(np.mean([inside(r) for r in rs])) if rs else float("nan")

    def package(sel, csel):
        if not [r for r in sel if not r["cancelled"]] or not [r for r in sel if not r["cancelled_T15"]]:
            return {"n_jobs": len(sel), "n_cancelled": sum(r["cancelled"] for r in sel), "rules": {}, "note": "no job left in this subset"}
        tagged = {}
        for r in sel:
            tagged.setdefault(r["neuron"], []).append(("job", r))
        for r in csel:
            tagged.setdefault(r["neuron"], []).append(("ceil", r))

        def p_stat(xs):
            j = [r for k, r in xs if k == "job"]
            c = [r for k, r in xs if k == "ceil"]
            f, fc = fshare(j), fshare(c)
            return US.packed_share(f, fc) if fc < 1 else float("nan")
        live = [r for r in sel if not r["cancelled"]]
        med = US.boot_neurons(by_neuron(live), lambda rs: float(np.median([r["T"] for r in rs])), n_boot)
        diffs = [r for r in live if r["T_foreign"] is not None and np.isfinite(r["T_foreign"])]
        d_for = US.boot_neurons(by_neuron(diffs), lambda rs: float(np.median([r["T"] - r["T_foreign"] for r in rs])), n_boot)
        p = US.boot_neurons(tagged, p_stat, n_boot)
        lo_off = float(np.median([r["T"] - band[r["tercile"]][0] for r in live]))
        hi_off = float(np.median([r["T"] - band[r["tercile"]][1] for r in live]))
        live_end = [r for r in sel if not r["cancelled_T15"]]
        end_lo = float(np.median([r["T_at_T15"] - band[r["tercile"]][0] for r in live_end]))
        end_hi = float(np.median([r["T_at_T15"] - band[r["tercile"]][1] for r in live_end]))
        cmp_ = median_comparison(sel, csel, by_ter, n_boot)
        own = cmp_["birth_vs_planted"]
        end_cmp = cmp_["T15_vs_planted"]
        grown = ("refuted: the interval of median T overlaps the planted median's" if own["overlap"] else
                 "not refuted: intervals do not overlap and real T is higher" if own["side"] == "real T higher" else
                 "not refuted by this rule: intervals do not overlap and real T is lower, its own outcome, does not support H_grown")
        return {"n_jobs": len(sel), "n_cancelled": len(sel) - len(live), "n_neurons": len(by_neuron(sel)), "median_T": med, "median_comparison": cmp_, "median_X": float(np.median([r["X"] for r in live])), "median_WJ": float(np.median([r["WJ"] for r in live])), "median_WO": float(np.median([r["WO"] for r in live])),
                "f": fshare(sel), "f_c": fshare(csel), "n_ceilings": len(csel), "n_ceilings_cancelled": sum(r["cancelled"] for r in csel), "p": p, "median_T_minus_T_foreign": d_for,
                "median_T_minus_band_lo": lo_off, "median_T_minus_band_hi": hi_off, "median_T_at_T15": float(np.median([r["T_at_T15"] for r in live_end])), "n_T15_cancelled": len(sel) - len(live_end), "T15_minus_band_lo": end_lo, "T15_minus_band_hi": end_hi,
                "rules": {"H_grown": grown, "no_shared_input": US.rule_packed(p["lo"], p["hi"]), "H_context": bool(d_for["lo"] <= 0 <= d_for["hi"]),
                          "H_own_route": bool(own["overlap"] and end_cmp["overlap"]),
                          "H_grown_original_rule_superseded_before_unblinding": "refuted: median T inside the packed band" if lo_off >= 0 and hi_off <= 0 else ("not refuted: median T above the band" if hi_off > 0 else "not refuted by that rule: median T below the band"),
                          "H_own_route_original_rule_superseded_before_unblinding": bool(lo_off >= 0 and hi_off <= 0 and end_lo >= 0 and end_hi <= 0)}}

    prim, cprim = [r for r in rows if r["primary"]], [r for r in ceil_rows if r["primary"]]
    out = {"primary": package(prim, cprim), "all_1049": package(rows, ceil_rows), "freeze": freeze}
    for ter in (0, 1, 2):
        out[f"tercile_{ter}"] = package([r for r in prim if r["tercile"] == ter], [r for r in cprim if r["tercile"] == ter])
    fl = [r for r in prim if r["flagged"] and not r["cancelled"]]
    un = [r for r in prim if not r["flagged"] and not r["cancelled"]]
    tag = {}
    for r in fl + un:
        tag.setdefault(r["neuron"], []).append(r)
    assert fl and un and tag, "a flag group is empty"

    def contrast(rs):
        a, b = [r["T"] for r in rs if r["flagged"]], [r["T"] for r in rs if not r["flagged"]]
        return float(np.median(a) - np.median(b)) if a and b else float("nan")
    c = US.boot_neurons(tag, contrast, n_boot)
    out["flagged_vs_unflagged"] = {"median_T_flagged": float(np.median([r["T"] for r in fl])), "median_T_unflagged": float(np.median([r["T"] for r in un])), "n_flagged": len(fl), "n_unflagged": len(un), "difference": c,
                                   "difference_corrected_by_0.58": c["point"] / US.FLAG_SCALE, "scale": US.FLAG_SCALE}
    live = [r for r in prim if not r["cancelled"]]
    out["sensitivity"] = {"median_T_fixed_coefficients": float(np.nanmedian([r["T_fixed"] for r in live if r["T_fixed"] is not None])),
                          "median_T_max_over_older": float(np.nanmedian([r["T_max_over_older"] for r in live])), "median_T_pooled_older": float(np.nanmedian([r["T_pooled_older"] for r in live])),
                          "excluding_births_T7_T15": float(np.median([r["T"] for r in live if r["birth"] not in (7, 15)])), "shallow_L_under_3": {"n": sum(r["shallow"] for r in live), "median_T": float(np.median([r["T"] for r in live if r["shallow"]])) if any(r["shallow"] for r in live) else None}}
    if s1_jobs:
        s1 = []
        for j in s1_jobs:
            t = f"T_{j['birth']}"
            if t in g and g[t].get(j["pole"]) is not None:
                r = g[t].T(j["pole"], j["ranks"], j["o_ranks"])
                s1.append({"neuron": neuron_of(j["pole"]), "tercile": US.tercile(layer_of(j["pole"])), **r, "cancelled": US.cancelled(r, w_noise)})
        s1l = [r for r in s1 if not r["cancelled"]]
        out["sample_1"] = {"n_jobs": len(s1), "n_cancelled": len(s1) - len(s1l), "median_T": float(np.median([r["T"] for r in s1l])) if s1l else None, "f": fshare(s1)}
    out["job_rows"], out["ceiling_rows"] = rows, ceil_rows
    return out


# ------------------------------------------------------------------------------------------ Arm 2
PREDICTIONS = {"H5": 0.37, "H5p": 0.57, "H4": 0.77}
F3_NLL_BAR, F3_MASS_SHARE = 2.8585, 0.25
CEILING_ROOM = 0.25


def per_window(d, hop, arm):
    return json.loads((pathlib.Path(d) / "per_item" / f"{hop}__{arm}.json").read_text())["nll_per_window"]


def swap_report(d, hops=("H4", "H5", "H5p"), n_boot=US.N_BOOT_VERDICT) -> dict:
    """F1: absolute NLL and R with intervals for every forward arm, the predictions and their rule, the gates, the F3 trigger."""
    m = json.loads((pathlib.Path(d) / "metrics.json").read_text())
    out = {"status": m.get("status"), "hops": {}}
    for hop in hops:
        ro = m["readouts"][hop]
        c, p = per_window(d, hop, "child"), per_window(d, hop, "parent")
        arms = {a: {**US.boot_windows(c, p, per_window(d, hop, a), n_boot), "mass_share": ro[a]["mass_share"]} for a, r in ro.items() if isinstance(r, dict) and r.get("kind") == "forward"}
        pred, a = PREDICTIONS[hop], arms["last_4_head"]
        fail = per_window(d, hop, "failing_input")
        ctl = m["controls"]
        out["hops"][hop] = {"gates": ctl[f"{hop}__gates"], "failing_input": {**ctl[f"{hop}__failing_input"], "R_interval": US.boot_windows(c, p, fail, n_boot)}, "arms": arms,
                            "prediction_last_4_head": {"predicted_R": pred, "R": a["R"], "lo": a["R_lo"], "hi": a["R_hi"], "refuted": not a["R_lo"] <= pred <= a["R_hi"]},
                            "reverse_top1_agreement_with_child": ro.get("reverse_top1_agreement_with_child")}
    if "H5" in out["hops"]:
        out["F3_trigger"] = [{"arm": a, **{k: v[k] for k in ("nll", "nll_hi", "R", "R_lo", "mass_share")}} for a, v in out["hops"]["H5"]["arms"].items() if v["nll_hi"] <= F3_NLL_BAR and v["mass_share"] <= F3_MASS_SHARE]
        out["F3_baseline"] = "the parent itself recovers everything if text may be sent to it"
    return out


def topk_report(d, ks=(10, 100, 1000, 10000), n_boot=US.N_BOOT_VERDICT) -> dict:
    """F2: the mass table, the arms with intervals and the frozen reading per k, read only where the all-MLP ceiling shows room."""
    m = json.loads((pathlib.Path(d) / "metrics.json").read_text())
    out = {"status": m.get("status"), "hops": {}}
    for hop, ro in m["readouts"].items():
        c, p = per_window(d, hop, "child"), per_window(d, hop, "parent")
        arms = {a: {**US.boot_windows(c, p, per_window(d, hop, a), n_boot), **{x: r[x] for x in ("mass_restored", "mass_share", "mass_intended", "kind", "matched", "n_components") if x in r}} for a, r in ro.items() if isinstance(r, dict) and r.get("kind") in ("f2", "ceiling", "sv_control")}
        ceil = arms["all_mlp"]["R"]
        read = {}
        table = m["controls"][f"{hop}__mass_table"]["k"]
        for k in ks:
            if not table[str(k)]["tail_exists"]:
                read[str(k)] = {"reading": "skipped: no least-changed set within 5% of the top-k mass"}
                continue
            if ceil < CEILING_ROOM:
                read[str(k)] = {"reading": "not read: ceiling R under 0.25, no room", "ceiling_R": ceil}
                continue
            per = {}
            for side in ("input", "output", "both"):
                if f"top{k}_{side}" not in arms:
                    continue
                t, b = arms[f"top{k}_{side}"], arms[f"tail{k}_{side}"]
                diff = US.boot_window_diff(c, p, per_window(d, hop, f"top{k}_{side}"), per_window(d, hop, f"tail{k}_{side}"), n_boot)
                reading = "diffuse erosion: tail at least equal to top k" if b["R"] >= t["R"] else "sparse repurposing: top k recovers most" if t["R"] >= 0.5 * ceil and diff["lo"] > 0 else "neither"
                per[side] = {"R_top": t["R"], "R_tail": b["R"], "top_minus_tail": diff, "reading": reading}
            read[str(k)] = {"ceiling_R": ceil, "sides": per}
        out["hops"][hop] = {"mass_table": m["controls"][f"{hop}__mass_table"], "arms": arms, "reading_by_k": read}
    if "H5" in out["hops"]:
        out["F3_trigger"] = [{"arm": a, **{k: v[k] for k in ("nll", "nll_hi", "R", "R_lo", "mass_share")}} for a, v in out["hops"]["H5"]["arms"].items() if v["nll_hi"] <= F3_NLL_BAR and v["mass_share"] <= F3_MASS_SHARE]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "verdict", "swap", "topk"])
    ap.add_argument("--items")
    ap.add_argument("--dirs", nargs="+", help="T_15=<dir> T_13=<dir> ...; for swap and topk, one directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub", default="grams")
    ap.add_argument("--freeze")
    ap.add_argument("--freeze-section", help="a section of the freeze file, for the secondary readout")
    ap.add_argument("--pilot", help="the readout's own pilot JSON, for its T_15 planted values (PREREG §13)")
    ap.add_argument("--planted-from")
    a = ap.parse_args()
    if a.cmd in ("swap", "topk"):
        res = {"swap": swap_report, "topk": topk_report}[a.cmd](a.dirs[0])
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print("ANALYSIS_OK", a.cmd, a.out)
        return
    items = json.loads(pathlib.Path(a.items).read_text())
    dirs = dict(x.split("=", 1) for x in a.dirs)
    if a.cmd == "pilot":
        res = pilot(items, dirs, a.sub, a.planted_from)
    else:
        freeze = json.loads(pathlib.Path(a.freeze).read_text())
        freeze = freeze[a.freeze_section] if a.freeze_section else freeze
        res = verdict(items, dirs, freeze, a.sub, items.get("s1_jobs"), planted_rows=json.loads(pathlib.Path(a.pilot).read_text())["planted_rows"])
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print("ANALYSIS_OK", a.cmd, a.out)


if __name__ == "__main__":
    main()
