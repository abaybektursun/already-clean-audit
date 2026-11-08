"""GUTENBERG pure instrument core — every frozen rule and aggregation as a pure function.

Spec: studies/lineage/paligemma1-pi0/research/2026-08-19_gutenberg_program_spec.md (§0.5
amendments AM-0.5.4/5/7/8/9). Jobs under infra/aws_pg/jobs/gutenberg_*.py orchestrate; the
decisions live HERE and are unit-tested on tiny tensors (test_gutenberg_core.py) before any
compute is spent. No S3, no model loads, no filesystem in this module. FAIL LOUD.
"""
import numpy as np


def provenance():
    """Run provenance for GUTENBERG artifacts. `pg run` ships only the working-tree infra/
    (no .git on the instance — the GB-A/GB-2 first-launch failure, 2026-08-19), so the stamp
    is a FILE written at launch time by the control plane and shipped with the code. FAIL LOUD
    if absent: an unstamped artifact is the D41 staleness defect, not a fallback case."""
    import json as _json
    import pathlib as _pathlib
    p = _pathlib.Path(__file__).resolve().parent / "gutenberg_provenance.json"
    d = _json.loads(p.read_text())
    assert d.get("program") == "gutenberg", d
    return d


# ===========================================================================
# GB-A — object identity: subspace Jaccard vs the matched-rotation null (AM-0.5.7)
# ===========================================================================

def subspace_jaccard(U, V):
    """J = t / (2k - t) with t = ||U^T V||_F^2 = tr(P_U P_V), for two (n x k) orthonormal bases.
    J in [0, 1]; J = 1 iff span(U) = span(V); the subspace analogue of set Jaccard with t playing
    |A∩B| and 2k - t playing |A∪B| (equal dims)."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    assert U.shape == V.shape and U.ndim == 2, (U.shape, V.shape)
    k = U.shape[1]
    t = float(((U.T @ V) ** 2).sum())
    assert -1e-9 <= t <= k + 1e-6, t
    return t / (2.0 * k - t)


def rotation_null_median_jaccard(n, k, n_layers, n_draws, seed):
    """Null distribution of the DECISION statistic (median over layers of J) under the
    matched-rotation null: each layer's second basis replaced by an independent Haar k-frame.
    By rotation invariance J's null law depends only on (n, k), so the fixed frame is E_k
    (first-k coordinates) and t is the squared mass of the Haar frame's first k rows.
    Returns (median_draws (n_draws,), per_layer_draws (n_draws * n_layers,))."""
    rng = np.random.default_rng(seed)
    per_layer = np.empty(n_draws * n_layers, dtype=np.float64)
    for i in range(n_draws * n_layers):
        H, _ = np.linalg.qr(rng.standard_normal((n, k)))
        t = float((H[:k, :] ** 2).sum())
        per_layer[i] = t / (2.0 * k - t)
    medians = np.median(per_layer.reshape(n_draws, n_layers), axis=1)
    return medians, per_layer


def gba2_ceiling_verdict(per_layer_J_AB, gba_median_J):
    """Frozen rule AM-0.6.1 (registered before the ceiling datum existed).
    per_layer_J_AB: subspace Jaccard between the two disjoint 12-fine-tune halves' bottom-32
    eigenspaces, per layer — the same-object ceiling at family size 12.
    gba_median_J: GB-A's BANKED median Jaccard between the 24-FT core and the census core.

    CEILING-CLEAR      median J(A,B) >= 0.30 and gba_median_J < 0.5 * median J(A,B)
    CEILING-COLLAPSE   median J(A,B) < 0.15  -> GB-A re-registers as AMBIGUOUS-BY-CEILING
    INTERMEDIATE       otherwise             -> DISTINCT stands, carries the ceiling as limitation
    """
    med = float(np.median(np.asarray(per_layer_J_AB, dtype=np.float64)))
    if med < 0.15:
        verdict, gba = "CEILING-COLLAPSE", "AMBIGUOUS-BY-CEILING"
    elif med >= 0.30 and gba_median_J < 0.5 * med:
        verdict, gba = "CEILING-CLEAR", "DISTINCT (object difference)"
    else:
        verdict, gba = "INTERMEDIATE", "DISTINCT (with ceiling as stated limitation)"
    return {"verdict": verdict, "gba_reregistration": gba,
            "median_jaccard_halves": med, "gba_median_jaccard": float(gba_median_J),
            "ratio_gba_over_ceiling": float(gba_median_J / med) if med > 0 else float("inf")}


def gba_verdict(per_layer_J, null_median_draws):
    """Frozen rule (spec §3 GB-A + AM-0.5.7): SAME-OBJECT iff median_L J >= 0.5 AND above the
    rotation null's p95 (p95 of the null MEDIAN statistic — the null matched to the decision
    statistic); DISTINCT iff median_L J < 0.3; else AMBIGUOUS."""
    medJ = float(np.median(np.asarray(per_layer_J, dtype=np.float64)))
    p95 = float(np.percentile(np.asarray(null_median_draws, dtype=np.float64), 95))
    if medJ >= 0.5 and medJ > p95:
        verdict = "SAME-OBJECT"
    elif medJ < 0.3:
        verdict = "DISTINCT"
    else:
        verdict = "AMBIGUOUS"
    return {"verdict": verdict, "median_jaccard": medJ, "rotation_null_median_p95": p95}


# ===========================================================================
# GB-2 — reader gain (AM-0.5.4). Raw mean squared gain per direction, the banked convention:
# gain(U) = ||W U||_F^2 / k; allmean = ||W||_F^2 / n (the mean over a full orthonormal basis).
# ===========================================================================

def reader_gain(W, U):
    W = np.asarray(W, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    assert W.shape[1] == U.shape[0], (W.shape, U.shape)
    k = U.shape[1]
    return float(((W @ U) ** 2).sum()) / k


def allmean_gain(W):
    W = np.asarray(W, dtype=np.float64)
    return float((W ** 2).sum()) / W.shape[1]


def coordinate_bank(n, idx):
    """(n x k) axis-aligned orthonormal bank for a coordinate index set (topenergy arms)."""
    idx = np.asarray(idx)
    E = np.zeros((n, idx.size), dtype=np.float64)
    E[idx, np.arange(idx.size)] = 1.0
    return E


def haar_frame(n, k, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return Q


def overlap_with_singular_subspace(U, V_sing, k_sing):
    """Mean squared projection of U's columns onto the span of k_sing singular vectors
    (columns of V_sing[:, :k_sing]) — the banked overlap convention (chance = k_sing/n)."""
    B = np.asarray(V_sing, dtype=np.float64)[:, :k_sing]
    U = np.asarray(U, dtype=np.float64)
    return float(((B.T @ U) ** 2).sum()) / U.shape[1]


# ===========================================================================
# safetensors range-read (GB-2's transport; pure given a fetch(start, stop) -> bytes)
# ===========================================================================

def safetensors_header(fetch):
    """fetch(start, stop) returns bytes [start, stop). Returns (header dict, data_start)."""
    import json as _json
    import struct
    (hlen,) = struct.unpack("<Q", fetch(0, 8))
    assert 0 < hlen < 100_000_000, hlen
    hdr = _json.loads(fetch(8, 8 + hlen).decode("utf-8"))
    return hdr, 8 + hlen


def range_read_tensor(fetch, hdr, data_start, name):
    """One tensor by name as fp64 ndarray, reading only its byte range."""
    meta = hdr[name]
    s0, s1 = meta["data_offsets"]
    raw = fetch(data_start + s0, data_start + s1)
    dt = meta["dtype"]
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dt == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dt == "F16":
        arr = np.frombuffer(raw, dtype=np.float16)
    else:
        raise AssertionError(f"unhandled safetensors dtype {dt} for {name}")
    return np.asarray(arr, dtype=np.float64).reshape(meta["shape"])


# ===========================================================================
# GB-1 / GB-3 — paired per-window statistics + the frozen inversion rule
# ===========================================================================

def dnll_per_window(nll_arm, nll_none, ntok):
    """Per-token ΔNLL per window (paired by window id — same order, same corpus)."""
    a = np.asarray(nll_arm, dtype=np.float64)
    b = np.asarray(nll_none, dtype=np.float64)
    t = np.asarray(ntok, dtype=np.float64)
    assert a.shape == b.shape == t.shape and t.min() > 0
    return (a - b) / t


def boot_mean_ci(x, n_boot=1000, seed=0):
    """(mean, lo95, hi95) window bootstrap."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    N = x.size
    reps = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        reps[i] = x[rng.integers(0, N, N)].mean()
    return float(x.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def paired_ratio_ci(dn_num, dn_den, n_boot=1000, seed=0):
    """Ratio of mean per-window ΔNLL (num/den), bootstrapped over the SAME window indices in
    both arms per replicate (paired). Returns (ratio, lo95, hi95)."""
    a = np.asarray(dn_num, dtype=np.float64)
    b = np.asarray(dn_den, dtype=np.float64)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    N = a.size
    reps = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, N, N)
        reps[i] = a[idx].mean() / b[idx].mean()
    return float(a.mean() / b.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def inversion_verdict(core_ci, ematch_ci):
    """Frozen rule (spec §3 GB-1): INVERSION-HOLDS iff the energy-matched cost exceeds core's
    with disjoint 95% CIs; REVERSED iff core's point cost >= energy-matched's; else AMBIGUOUS.
    Each ci = (mean, lo95, hi95) of per-token ΔNLL."""
    core_mean, _, core_hi = core_ci
    em_mean, em_lo, _ = ematch_ci
    if em_lo > core_hi:
        return "INVERSION-HOLDS"
    if core_mean >= em_mean:
        return "REVERSED"
    return "AMBIGUOUS"


def am1b_confirm_verdict(sc_ratio_ci):
    """Frozen rule (AM-0.5.5, GB-3 site i): AM1B-CONFIRMED iff the fresh-draw S/C ratio's 95%
    paired-bootstrap CI excludes 1 (above it); REVERSED if the CI sits entirely below 1; else
    AMBIGUOUS. ci = (ratio, lo95, hi95)."""
    _, lo, hi = sc_ratio_ci
    if lo > 1.0:
        return "AM1B-CONFIRMED"
    if hi < 1.0:
        return "REVERSED"
    return "AMBIGUOUS"


# ===========================================================================
# GB-4 — the two-factor surface (AM-0.5.9)
# ===========================================================================

def _loglin_fit(logx, logy):
    A = np.stack([np.ones_like(logx), logx], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, logy, rcond=None)
    return coef  # (a, b)


def _loglin_pred(coef, logx):
    return coef[0] + coef[1] * logx


def two_factor_analysis(arms, heldout_names, n_boot=1000, seed=0,
                        max_discard_frac=0.2, min_fit_banks=6):
    """The GB-4 frozen analysis. arms: {name: {"dnll_w": per-window ΔNLL (paired order),
    "E": removed-energy median share, "G": median reader gain}}. heldout_names: frozen held-out
    banks (never fit). Fits log(mean dnll) = a + b*log(E*G) (two-factor) vs = a + b*log(E)
    (energy-alone), same parameter count. Δ = med|err_E| − med|err_2F| over held-out banks
    (log-space absolute prediction errors). LAW iff the window-bootstrap 95% CI of Δ is > 0.
    Arms with point mean dnll <= 0 are excluded from the fit (log-undefined) and disclosed.
    Auto-CURVE if fewer than min_fit_banks fit banks remain or if more than max_discard_frac of
    bootstrap replicates are discarded (a replicate is discarded when any surviving fit or
    held-out arm's replicate mean is <= 0). Pure; jobs feed it artifact-derived arrays."""
    heldout_names = list(heldout_names)
    for h in heldout_names:
        assert h in arms, f"held-out arm {h} missing from inputs"
    fit_names = [a for a in sorted(arms) if a not in heldout_names]
    excluded = [a for a in fit_names if float(np.mean(arms[a]["dnll_w"])) <= 0.0]
    fit_names = [a for a in fit_names if a not in excluded]
    result = {"fit_banks": fit_names, "heldout_banks": heldout_names,
              "excluded_nonpositive": excluded, "n_fit_banks": len(fit_names)}
    if len(fit_names) < min_fit_banks:
        result.update({"verdict": "CURVE",
                       "reason": f"auto-CURVE: {len(fit_names)} fit banks < {min_fit_banks}"})
        return result

    def _errors(means):
        lx2 = np.log([arms[a]["E"] * arms[a]["G"] for a in fit_names])
        lxe = np.log([arms[a]["E"] for a in fit_names])
        ly = np.log([means[a] for a in fit_names])
        c2, ce = _loglin_fit(lx2, ly), _loglin_fit(lxe, ly)
        e2 = [abs(float(_loglin_pred(c2, np.log(arms[h]["E"] * arms[h]["G"]))
                        - np.log(means[h]))) for h in heldout_names]
        ee = [abs(float(_loglin_pred(ce, np.log(arms[h]["E"]))
                        - np.log(means[h]))) for h in heldout_names]
        return float(np.median(ee) - np.median(e2)), (c2.tolist(), ce.tolist())

    point_means = {a: float(np.mean(arms[a]["dnll_w"])) for a in arms}
    for h in heldout_names:
        assert point_means[h] > 0, f"held-out arm {h} has non-positive cost — rule cannot run"
    delta_point, coefs = _errors(point_means)

    rng = np.random.default_rng(seed)
    N = len(next(iter(arms.values()))["dnll_w"])
    for a in arms.values():
        assert len(a["dnll_w"]) == N, "arms not window-aligned"
    deltas, discarded = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        means = {a: float(np.mean(np.asarray(arms[a]["dnll_w"])[idx]))
                 for a in fit_names + heldout_names}
        if any(m <= 0 for m in means.values()):
            discarded += 1
            continue
        deltas.append(_errors(means)[0])
    frac_disc = discarded / n_boot
    result.update({"delta_point": delta_point, "coef_two_factor": coefs[0],
                   "coef_energy_alone": coefs[1], "boot_discard_frac": frac_disc,
                   "n_boot_used": len(deltas)})
    if frac_disc > max_discard_frac:
        result.update({"verdict": "CURVE",
                       "reason": f"auto-CURVE: {frac_disc:.2f} of replicates discarded "
                                 f"(> {max_discard_frac})"})
        return result
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    result.update({"delta_lo95": float(lo), "delta_hi95": float(hi)})
    result["verdict"] = "LAW" if lo > 0 else "CURVE"
    result["reason"] = ("held-out |err| improvement CI clear of parity" if lo > 0
                       else "held-out improvement CI not clear of parity")
    return result


# ===========================================================================
# GB-1 aggregation (pure — the dry run executes this on synthetic cells)
# ===========================================================================

def build_gb1_metrics(rows_by_site, wnll_by_site, gates, ematch_diag, arm_shares,
                      n_boot=1000, seed=20260819):
    """Assemble GB-1's verdict artifact from raw per-arm outputs.
    rows_by_site: {site: {arm: ppl}}; wnll_by_site: {site: {arm: {"nll": [...], "ntok": [...]}}};
    gates: dict of gate booleans/values (asserted upstream, recorded here);
    ematch_diag / arm_shares: passed through for the match-verification record.
    Confirmatory verdict lives on site 'q' (AM-1's instrument); 'qk' is the bridge rider."""
    out = {"gates": gates, "energy_match_per_layer": ematch_diag,
           "arm_shares": arm_shares, "sites": {}}
    for site, wnll in wnll_by_site.items():
        none = wnll["none"]
        dn = {arm: dnll_per_window(w["nll"], none["nll"], none["ntok"])
              for arm, w in wnll.items() if arm != "none"}

        def pool(prefix):
            seeds = sorted(a for a in dn if a.startswith(prefix))
            assert seeds, f"no arms with prefix {prefix} at site {site}"
            return np.mean([dn[a] for a in seeds], axis=0), seeds

        core = dn["core32"]
        em, em_arms = pool("energymatched32")
        ha, ha_arms = pool("haar32")
        core_ci = boot_mean_ci(core, n_boot, seed)
        em_ci = boot_mean_ci(em, n_boot, seed + 1)
        site_res = {
            "ppl": rows_by_site[site],
            "dnll": {"core32": core_ci, "energymatched32": em_ci,
                     "haar32": boot_mean_ci(ha, n_boot, seed + 2)},
            "pooled_arms": {"energymatched32": em_arms, "haar32": ha_arms},
            "E_over_C": paired_ratio_ci(em, core, n_boot, seed + 3),
            "core_cheaper_in_n_windows": int((core < em).sum()),
            "n_windows": int(core.size),
            "verdict": inversion_verdict(core_ci, em_ci),
        }
        if any(a.startswith("specmatched32") for a in dn):
            sp, sp_arms = pool("specmatched32")
            site_res["dnll"]["specmatched32"] = boot_mean_ci(sp, n_boot, seed + 4)
            site_res["S_over_C"] = paired_ratio_ci(sp, core, n_boot, seed + 5)
            site_res["pooled_arms"]["specmatched32"] = sp_arms
        for extra in ("bottomsigma32", "topsigma32", "topenergy32"):
            if extra in dn:
                site_res["dnll"][extra] = boot_mean_ci(dn[extra], n_boot, seed + 6)
        out["sites"][site] = site_res
    assert "q" in out["sites"], "confirmatory site 'q' missing"
    out["verdict"] = out["sites"]["q"]["verdict"]
    return out


# ===========================================================================
# GB-3 aggregation (pure — the dry run executes this on synthetic cells).
# MENDEL lesson: the post-loop block that assembles results and writes metrics is the most
# expensive place in a job to hold a bug, because you only reach it after all the compute.
# ===========================================================================

def build_gb3_metrics(wnll, wnll_qk, rows, seed=20260819, n_boot=1000):
    """GB-3's verdict from one run's per-window NLLs.
    wnll / wnll_qk: {arm: {"nll": [...], "ntok": [...]}} at the q site and the q+k site, both
    sharing the same `none` arm. rows: the arm/ppl table (its reproduction_gate is carried
    through). Confirmatory statistic is the spectrum-matched ratio at the q site (AM-0.5.5)."""
    none = wnll["none"]
    ntok = np.asarray(none["ntok"], dtype=np.float64)

    def dn(arm, table):
        return dnll_per_window(table[arm]["nll"], none["nll"], ntok)

    def pooled(prefix, table):
        ks = sorted(k for k in table if k.startswith(prefix) and k != "none")
        assert ks, f"no arms with prefix {prefix}"
        return np.mean([dn(k, table) for k in ks], axis=0), ks

    core = dn("core32", wnll)
    sp, sp_arms = pooled("specmatched32", wnll)
    em, em_arms = pooled("energymatched32", wnll)
    sc = paired_ratio_ci(sp, core, n_boot, seed)
    ec = paired_ratio_ci(em, core, n_boot, seed + 1)
    core_ci = boot_mean_ci(core, n_boot, seed + 2)
    em_ci = boot_mean_ci(em, n_boot, seed + 3)
    out = {
        "verdict": am1b_confirm_verdict(sc),
        "S_over_C": sc, "E_over_C": ec,
        "dnll": {"core32": core_ci, "specmatched32": boot_mean_ci(sp, n_boot, seed + 4),
                 "energymatched32": em_ci},
        "core_cheaper_than_spec_in_n_windows": int((core < sp).sum()),
        "core_cheaper_than_ematch_in_n_windows": int((core < em).sum()),
        "n_windows": int(core.size),
        "fresh_draw_arms": {"spec": sp_arms, "ematch": em_arms},
        "inversion_verdict_fresh": inversion_verdict(core_ci, em_ci),
        "reproduction_gate": rows.get("reproduction_gate"),
    }
    if wnll_qk:
        core_qk = dn("core32", wnll_qk)
        em_qk, em_qk_arms = pooled("energymatched32", wnll_qk)
        out["qk_bridge_exploratory"] = {
            "E_over_C": paired_ratio_ci(em_qk, core_qk, n_boot, seed + 5),
            "dnll_core32": boot_mean_ci(core_qk, n_boot, seed + 6),
            "dnll_energymatched32": boot_mean_ci(em_qk, n_boot, seed + 7),
            "pooled_arms": em_qk_arms,
            "note": "FUNC's q+k instrument on this substrate — bridge row, no confirmatory weight",
        }
    return out


# ===========================================================================
# GB-A aggregation
# ===========================================================================

def build_gba_metrics(J_by_layer, null_median_draws, null_per_layer_draws, provenance):
    layers = sorted(J_by_layer)
    Js = [J_by_layer[L] for L in layers]
    v = gba_verdict(Js, null_median_draws)
    v.update({
        "per_layer_jaccard": {str(L): float(J_by_layer[L]) for L in layers},
        "n_layers": len(layers),
        "rotation_null_per_layer_p95":
            float(np.percentile(np.asarray(null_per_layer_draws), 95)),
        "n_layers_above_per_layer_null_p95":
            int(sum(j > np.percentile(np.asarray(null_per_layer_draws), 95) for j in Js)),
        "definition": "J = t/(2k−t), t = ||U1^T U2||_F^2, k=32; verdict per AM-0.5.7",
        "provenance": provenance,
    })
    return v
