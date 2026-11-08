"""UNITE statistics and gate rules (studies/unite/PREREG.md). NumPy only: the jobs, the dry run and the offline analysis share it."""
from __future__ import annotations

import numpy as np

N_BOOT_VERDICT = 10_000
N_BOOT_ROOM = 2_000
KNOWN_TOL = COMPLETE_TOL = 1e-2
ROOM_FLOOR, CEILING_MAX = 0.20, 1.2
MLP_LINE, MLP_SUSPECT = 0.50, 1.5
CANCEL_MAX = 0.40
WINDOW_MEDIAN, WINDOW_P90 = 0.05, 0.25
ID_LO, ID_HI, ID_FAIL = 0.9, 1.1, 0.5
TERCILE_EDGES = (11, 22)          # L 0-10, 11-21, 22-31 of 32 layers; planted layers 5, 15, 26 sit one in each
FLAG_SCALE = 0.58


def tercile(layer: int) -> int:
    return int(layer >= TERCILE_EDGES[0]) + int(layer >= TERCILE_EDGES[1])


# ------------------------------------------------------------------------------------------ the statistic
def within(M, J) -> float:
    J = np.asarray(J)
    assert len(J) >= 2, "W needs at least two contexts"
    S = M[np.ix_(J, J)]
    return float((S.sum() - np.trace(S)) / (len(J) * (len(J) - 1)))


def cross(M, J, O) -> float:
    return float(M[np.ix_(np.asarray(J), np.asarray(O))].mean())


def stat_T(M, J, O) -> dict:
    """T = X / sqrt(W(J) W(O)); NaN when either W is not positive. X, W(J), W(O) stay beside it."""
    wj, wo, x = within(M, J), within(M, O), cross(M, J, O)
    return {"T": x / np.sqrt(wj * wo) if wj > 0 and wo > 0 else float("nan"), "X": x, "WJ": wj, "WO": wo}


def identity_T(M, i: int, j: int) -> float:
    """One pair of contexts: M_ij / sqrt(M_ii M_jj), NaN when either diagonal is not positive."""
    a, b = M[i, i], M[j, j]
    return float(M[i, j] / np.sqrt(a * b)) if a > 0 and b > 0 else float("nan")


def noise_W(Ms: list, n_draws: int = 200, size: int = 7, seed: int = 0) -> np.ndarray:
    """|W| over random sets of `size` contexts from no job: each draw picks one pole's noise Gram at random."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for d in range(n_draws):
        M = Ms[rng.integers(len(Ms))]
        out[d] = abs(within(M, rng.choice(M.shape[0], size, replace=False)))
    return out


def cancelled(row: dict, w_noise: float) -> bool:
    return row["WJ"] <= w_noise or row["WO"] <= w_noise or not np.isfinite(row["T"])


def planted_count_matched(M, J, O, counts, draws: int = 200, seed: int = 0) -> float:
    """Median T over subsamples of the planted sides to real job counts [(n_J, n_O)], capped at the side's size."""
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(draws):
        nj, no = counts[rng.integers(len(counts))]
        j = rng.choice(J, min(max(nj, 2), len(J)), replace=False)
        o = rng.choice(O, min(max(no, 2), len(O)), replace=False)
        ts.append(stat_T(M, j, o)["T"])
    ts = np.array(ts)
    return float(np.median(ts[np.isfinite(ts)])) if np.isfinite(ts).any() else float("nan")


def mean_energy(M, J) -> float:
    """||mean A|| of a side, from the Gram: sqrt of the mean over all i, j in J of M_ij, clipped at 0."""
    return float(np.sqrt(max(M[np.ix_(np.asarray(J), np.asarray(J))].mean(), 0.0)))


# ------------------------------------------------------------------------------------------ intervals
def boot_neurons(rows_by_neuron: dict, stat, n_boot: int = N_BOOT_VERDICT, seed: int = 0) -> dict:
    """Percentile bootstrap over neurons. `rows_by_neuron`: {neuron: [row]}; `stat(rows)` returns a float."""
    keys = sorted(rows_by_neuron)
    rng = np.random.default_rng(seed)
    point = stat([r for k in keys for r in rows_by_neuron[k]])
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = stat([r for i in rng.integers(0, len(keys), len(keys)) for r in rows_by_neuron[keys[i]]])
    ok = draws[np.isfinite(draws)]
    if ok.size < 2:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sd": float("nan"), "n_units": len(keys), "n_rows": sum(len(v) for v in rows_by_neuron.values()),
                "n_boot": n_boot, "n_nonfinite_draws": int(n_boot - ok.size), "seed": seed}
    return {"point": float(point), "lo": float(np.percentile(ok, 2.5)), "hi": float(np.percentile(ok, 97.5)),
            "sd": float(ok.std(ddof=1)), "n_units": len(keys), "n_rows": sum(len(v) for v in rows_by_neuron.values()),
            "n_boot": n_boot, "n_nonfinite_draws": int(n_boot - ok.size), "seed": seed}


def recovery(child: float, parent: float, arm: float) -> float:
    return (child - arm) / (child - parent)


def boot_windows(child, parent, arm, n_boot: int = N_BOOT_VERDICT, seed: int = 0) -> dict:
    """Absolute NLL and R over paired windows, percentile bootstrap, 95%."""
    c, p, a = (np.asarray(x, float) for x in (child, parent, arm))
    assert c.shape == p.shape == a.shape and c.ndim == 1, (c.shape, p.shape, a.shape)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c.size, (n_boot, c.size))
    cm, pm, am = c[idx].mean(1), p[idx].mean(1), a[idx].mean(1)
    r = (cm - am) / (cm - pm)
    return {"nll": float(a.mean()), "nll_lo": float(np.percentile(am, 2.5)), "nll_hi": float(np.percentile(am, 97.5)),
            "R": recovery(c.mean(), p.mean(), a.mean()), "R_lo": float(np.percentile(r, 2.5)), "R_hi": float(np.percentile(r, 97.5)),
            "n_windows": int(c.size), "n_boot": n_boot, "seed": seed}


def boot_window_diff(child, parent, a, b, n_boot: int = N_BOOT_VERDICT, seed: int = 0) -> dict:
    """R(a) - R(b) over the same resampled windows."""
    c, p, x, y = (np.asarray(v, float) for v in (child, parent, a, b))
    idx = np.random.default_rng(seed).integers(0, c.size, (n_boot, c.size))
    cm, pm = c[idx].mean(1), p[idx].mean(1)
    d = (cm - x[idx].mean(1)) / (cm - pm) - (cm - y[idx].mean(1)) / (cm - pm)
    return {"point": recovery(c.mean(), p.mean(), x.mean()) - recovery(c.mean(), p.mean(), y.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


# ------------------------------------------------------------------------------------------ gates
def known_ratio(a_ours, a_bank) -> float:
    a_ours, a_bank = np.asarray(a_ours, float), np.asarray(a_bank, float)
    return float(np.median(np.abs(a_ours - a_bank)) / np.abs(a_bank).max())


def gate_known(ratio: float, wrong_ratio: float) -> dict:
    return {"ratio": ratio, "wrong_ratio": wrong_ratio, "tol": KNOWN_TOL, "pass": ratio <= KNOWN_TOL, "failing_input_failed": wrong_ratio > KNOWN_TOL}


def gate_complete(err: float, wrong_err: float) -> dict:
    return {"median_err_over_pole_max": err, "wrong_layer": wrong_err, "tol": COMPLETE_TOL, "pass": err <= COMPLETE_TOL,
            "failing_input_failed": wrong_err > COMPLETE_TOL}


def gate_identity(ts, cross_ts) -> dict:
    t, c = float(np.nanmedian(ts)), float(np.nanmedian(cross_ts))
    return {"median_T": t, "n": len(ts), "two_pole_median_T": c, "n_two_pole": len(cross_ts), "pass": ID_LO <= t <= ID_HI, "failing_input_failed": c < ID_FAIL}


def gate_noise(share_cancelled: float, share_no_job: float, w_noise: float) -> dict:
    return {"W_noise": w_noise, "share_cancelled_half_B": share_cancelled, "share_cancelled_no_job": share_no_job, "line": CANCEL_MAX,
            "pass": share_cancelled < CANCEL_MAX, "failing_input_failed": share_no_job > CANCEL_MAX}


def gate_window(rel, rel_16) -> dict:
    m, p = float(np.median(rel)), float(np.percentile(rel, 90))
    m16, p16 = float(np.median(rel_16)), float(np.percentile(rel_16, 90))
    return {"median": m, "p90": p, "median_16": m16, "p90_16": p16, "lines": [WINDOW_MEDIAN, WINDOW_P90], "pass": m <= WINDOW_MEDIAN and p <= WINDOW_P90,
            "failing_input_failed": not (m16 <= WINDOW_MEDIAN and p16 <= WINDOW_P90), "window": 256 if m <= WINDOW_MEDIAN and p <= WINDOW_P90 else "full"}


def gate_room(delta_B: float, m: float, med_ceiling_B: float, delta_shuffled: float) -> dict:
    line = max(m, ROOM_FLOOR)
    return {"delta_B": delta_B, "m": m, "line": line, "median_ceiling_B": med_ceiling_B, "delta_labels_shuffled": delta_shuffled, "ceiling_max": CEILING_MAX,
            "pass": delta_B >= line and med_ceiling_B <= CEILING_MAX, "failing_input_failed": delta_shuffled < line}


def gate_mlp(share: float, restricted: float) -> dict:
    """share: median per-job MLP share on half B. restricted: the same counting only layer L-1's MLP terms, gate poles at L >= 8."""
    if share > MLP_SUSPECT:
        branch = "decomposition suspect: Arm 1 cancelled"
    elif share < MLP_LINE:
        branch = "gradient readout primary, must pass room"
    else:
        branch = "direct readout primary"
    has_failing = restricted < MLP_LINE
    if not has_failing and share <= MLP_SUSPECT:
        branch = "no working failing input: direct readout not cleared, gradient readout primary, must pass room"
    return {"share": share, "restricted_share": restricted, "line": MLP_LINE, "suspect_above": MLP_SUSPECT, "failing_input_failed": has_failing, "branch": branch,
            "direct_primary": branch == "direct readout primary", "cancelled": branch.startswith("decomposition")}


# ------------------------------------------------------------------------------------------ verdict rules
def in_band(t: float, band) -> bool:
    return band[0] <= t <= band[1]


def packed_share(f: float, f_c: float) -> float:
    return (f - f_c) / (1.0 - f_c)


def rule_median(lo: float, hi: float, planted_lo: float, planted_hi: float) -> dict:
    """PREREG §13: a median over real jobs against the planted median, each with its 95% interval. Overlap refutes."""
    overlap = bool(lo <= planted_hi and planted_lo <= hi)
    side = "overlap" if overlap else "real T higher" if lo > planted_hi else "real T lower"
    return {"overlap": overlap, "side": side}


def rule_packed(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "does not count: interval undefined"
    return "refuted" if hi < 0.10 else "holds" if lo > 0.20 else "mixed"
