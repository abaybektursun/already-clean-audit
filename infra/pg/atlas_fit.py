"""ATLAS fit — descriptors, the frozen model family M0–M3, the sealed-cell grammar + adversarial
selection, and the confirmatory adjudicators (R-B1, R-W, R-N, R-X, R-M).

The estimand chain (spec §2, §3.3–3.4): every perturbation cell gets a DESCRIPTOR computable from
its construction + the banked Σ eigendecomposition + banked activation rows — never from a forward.
Four models map descriptor → predicted log(1+Δppl) (and, fit separately, VQA effect):

  M0  energy-only        isotone regression on log per-token energy (add / delete separate)
  M1  banded metric      M0 + a 16-sub-band diagonal sensitivity (LS on the iso residual, backfit)
  M2  gauge quotient     M1 on the T1–T3-TRANSFORMED descriptor (radial discount, RoPE retention,
                         V-transport √n_eff, mean-shift regressor). The radial-discount switch is
                         SET BY R-T1's branch (pre-specified: gauge → 1.0, partial → fitted,
                         active → 0.0) — the theory block is load-bearing only where it survived.
  M3  boundary           M2 + c·⟨δ̄, v₁⟩₊²

Model selection is NOT performed on fit quality: all four freeze after fitting and all four stake
sealed predictions (winner's-curse mitigation is structural, not statistical).

Weight-free; unit-tested by the D-suite. No try/except hiding.
"""
import hashlib
import itertools
import json
import math

import numpy as np
import torch

from infra.pg.bispec_core import band_projector

# ---------------------------------------------------------------------------
# Frozen constants (spec §3.3–3.4, §6).
# ---------------------------------------------------------------------------
SUBBAND_COUNTS = (1, 3, 4, 4, 4)        # energy-quantile refinement of the 5 BISPEC bands → 16
GRAMMAR_AMPS_ADD = (0.1, 0.2)           # ×rms (per-token); the sealed grammar's add amplitudes
GRAMMAR_FRACS_DEL = (0.25, 1.0)         # delete-side envelope mean-square (partial, full)
SEALED_N_REGISTERED = 30
SEALED_N_SCORED = 20
COVERAGE_MIN = 3                        # ≥3 cells per major axis value
COVERAGE_MAX_BANDMODE = 4               # ≤4 cells sharing a (sub-band × mode) pair
RADIAL_DEGENERATE_FRAC = 1e-3           # radial × sub-band with no radial mass ⇒ degenerate
BACKFIT_ROUNDS = 3
RIDGE_LAM_FIT = 0.1     # frozen ridge on the linear block: plain LS blew single-support sub-bands
                        # to e^880-scale predictions; λ=1.0 over-shrank dilute band-grain features
                        # (XᵀX diag ≈ 0.6 → >60 % shrinkage) and failed the §5 S6 top-cell check.
                        # λ=0.1 keeps single-support coefficients bounded (sanity guard backstops)
                        # while letting the S6 structure through. Tuned ONLY against the §5
                        # positive control, never against any sealed or discovery fit metric.
PRED_SANITY_MARGIN = 3.0                # candidate predictions may not leave the observed range
                                        # by more than this many log1p units — else the fitter is
                                        # degenerate and the driver HALTS before any freeze


# ===========================================================================
# Sub-bands: the frozen 16-way refinement of the banked 5 bands.
# ===========================================================================
def subband_slices(band_slices, spectrum, counts=SUBBAND_COUNTS):
    """Split each banked band into `counts[b]` sub-bands at equal-ENERGY quantiles within the band
    (index cuts would put estimator noise on the boundary — the eig_bands lesson, inherited).
    A band whose rank is below its requested count contributes rank sub-bands (counted, disclosed).
    Returns list of (lo, hi) index pairs into the DESCENDING spectrum."""
    ev = torch.as_tensor(spectrum, dtype=torch.float64)
    out = []
    for (lo, hi), c in zip(band_slices, counts):
        rank = hi - lo
        c_eff = min(c, rank)
        seg = ev[lo:hi]
        cum = torch.cumsum(seg, 0) / seg.sum()
        prev = lo
        for j in range(1, c_eff + 1):
            if j == c_eff:
                cut = hi
            else:
                cut = lo + int(torch.searchsorted(cum, torch.tensor(j / c_eff, dtype=torch.float64)).item()) + 1
                cut = min(max(cut, prev + 1), hi - (c_eff - j))
            out.append((prev, cut))
            prev = cut
    assert out[0][0] == 0 and out[-1][1] == band_slices[-1][1]
    assert all(a < b for a, b in out)
    return out


def subband_projector(basis, sb_slices, s):
    lo, hi = sb_slices[s]
    return basis[:, lo:hi].contiguous()


def subband_fractions_of_vector(v, basis, sb_slices):
    """Energy fraction of a residual-space vector per sub-band (sums to ≤1; the remainder is
    outside the banked spectrum's span — zero for a full-rank basis)."""
    v = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
    tot = float((v ** 2).sum())
    assert tot > 0
    coef = basis.transpose(0, 1) @ v
    return [float((coef[lo:hi] ** 2).sum() / tot) for lo, hi in sb_slices]


# ===========================================================================
# Descriptors. One flat numeric schema for every cell, banked and new.
# ===========================================================================
DESCRIPTOR_FIELDS = (
    "is_delete",           # 0 add / 1 delete
    "log_energy",          # log10 per-token energy: add = (eps/rms-units)²; delete = removed frac E
    "dc_frac",             # DC power fraction of the sequence arrangement (mean-shift carrier)
    "shared_dir",          # 1 if one direction/subspace is shared across positions
    "mean_shift",          # expected ‖δ̄‖ in σ·√d units (0 on the delete side, disclosed)
    "radial_frac",         # expected per-token radial energy fraction (measured on banked rows)
    "chan_none", "chan_k_low", "chan_k_high", "chan_v", "chan_q",
    "retention",           # RoPE retention factor for K-channel cells (1.0 elsewhere)
    "v1_overlap_pos",      # ⟨δ̄, v₁⟩₊ in the same σ·√d units (0 when δ̄ ⊥̸ defined)
)


def descriptor_vector(desc, sb_fracs):
    """Flatten a descriptor dict + its 16 sub-band fractions into the fixed numeric vector the
    models consume. Channel cells carry zero sub-band mass (they inject inside attention, not on
    the residual — disclosed; M1 sees them as energy-only)."""
    base = [float(desc[k]) for k in DESCRIPTOR_FIELDS]
    sb = list(map(float, sb_fracs))
    assert len(sb) == 16 or (len(sb) < 16 and desc.get("subband_deficit_ok")), \
        f"expected 16 sub-band fractions, got {len(sb)}"
    sb = sb + [0.0] * (16 - len(sb))
    return np.asarray(base + sb, dtype=float)


# ===========================================================================
# M0 — isotone regression (PAVA, no sklearn dependency).
# ===========================================================================
def pava_fit(x, y):
    """Weighted-free pool-adjacent-violators on (x, y): returns (bx, by) step function knots
    (x sorted ascending, fitted values non-decreasing). Ties in x are pre-pooled."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    # pool exact ties first
    ux, inv = np.unique(xs, return_inverse=True)
    ym = np.zeros(ux.size)
    w = np.zeros(ux.size)
    for i, v in zip(inv, ys):
        ym[i] += v
        w[i] += 1.0
    ym /= w
    # PAVA
    vals, wts, idx = [], [], []
    for i in range(ux.size):
        vals.append(ym[i]); wts.append(w[i]); idx.append(i)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop(); idx.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
            wts[-1] += w2
    by = np.zeros(ux.size)
    start = 0
    for v, i_end in zip(vals, idx):
        by[start:i_end + 1] = v
        start = i_end + 1
    return ux, by


def pava_predict(bx, by, xq):
    """Step interpolation, flat extension beyond the fitted range."""
    xq = np.asarray(xq, float)
    j = np.clip(np.searchsorted(bx, xq, side="right") - 1, 0, len(bx) - 1)
    return by[j]


# ===========================================================================
# The model family.
# ===========================================================================
def _t123_transform(D, *, radial_discount, n_eff_l0):
    """The M2 gauge quotient applied to the descriptor matrix (rows = cells, cols = DESCRIPTOR
    layout). Returns (adjusted log-energy column, extra feature columns).

      * radial share discounted by `radial_discount` ∈ [0,1] (set by R-T1's branch)
      * K-channel energy × retention² (amplitude retention → energy retention squared)
      * V-channel iid energy × 1/n_eff (coherent V passes at gain 1)
      * mean-shift and dc-coherence enter as explicit regressors
    """
    d = {k: D[:, i] for i, k in enumerate(DESCRIPTOR_FIELDS)}
    energy = 10.0 ** d["log_energy"]
    eff = energy * (1.0 - radial_discount * d["radial_frac"])
    # K-channel gauge quotient: the RETAINED (row-mean) post-RoPE component is the gauge direction
    # and does nothing; the live energy is ε²·(1 − retention²) (floored so log stays finite)
    k_chan = d["chan_k_low"] + d["chan_k_high"]
    eff = eff * np.where(k_chan > 0, np.maximum(1.0 - d["retention"] ** 2, 1e-6), 1.0)
    v_iid = (d["chan_v"] > 0) & (d["dc_frac"] < 0.5)
    eff = eff * np.where(v_iid, 1.0 / max(n_eff_l0, 1.0), 1.0)
    log_eff = np.log10(np.maximum(eff, 1e-12))
    extras = np.stack([np.log1p(d["mean_shift"]), d["dc_frac"] * d["shared_dir"]], axis=1)
    return log_eff, extras


def fit_model(rows, level, *, radial_discount=1.0, n_eff_l0=3.93, readout="ppl"):
    """Fit one model level on corpus rows (list of dicts with 'descriptor_vec', 'subband16',
    'damage_ppl'/'damage_vqa', 'sign'). Backfit: iso on the energy axis ↔ LS on the linear block.
    Returns a frozen, JSON-serializable predictor spec."""
    y_key = {"ppl": "damage_ppl", "vqa": "damage_vqa"}[readout]
    spec = {"level": int(level), "readout": readout, "radial_discount": float(radial_discount),
            "n_eff_l0": float(n_eff_l0), "signs": {}}
    for sign in ("add", "delete"):
        sub = [r for r in rows if r["sign"] == sign and not r.get("void")]
        if not sub:
            spec["signs"][sign] = None
            continue
        D = np.stack([r["descriptor_vec"] for r in sub])
        y = np.asarray([r[y_key] for r in sub], float)
        if level >= 2:
            log_e, extras = _t123_transform(D, radial_discount=radial_discount, n_eff_l0=n_eff_l0)
        else:
            log_e = D[:, DESCRIPTOR_FIELDS.index("log_energy")]
            extras = np.zeros((len(sub), 0))
        feats = []
        if level >= 1:
            feats.append(D[:, len(DESCRIPTOR_FIELDS):])                      # 16 sub-band fracs
        if level >= 2:
            feats.append(extras)
        if level >= 3:
            v1p = D[:, DESCRIPTOR_FIELDS.index("v1_overlap_pos")]
            feats.append((v1p ** 2).reshape(-1, 1))
        X = np.concatenate(feats, axis=1) if feats else np.zeros((len(sub), 0))
        coef = np.zeros(X.shape[1])
        bx, by = pava_fit(log_e, y)
        for _ in range(BACKFIT_ROUNDS if X.shape[1] else 1):
            resid = y - pava_predict(bx, by, log_e)
            if X.shape[1]:
                A = X.T @ X + RIDGE_LAM_FIT * np.eye(X.shape[1])
                coef = np.linalg.solve(A, X.T @ resid)
                bx, by = pava_fit(log_e, y - X @ coef)
        spec["signs"][sign] = {"iso_x": bx.tolist(), "iso_y": by.tolist(),
                               "coef": coef.tolist(), "n": len(sub)}
    return spec


def predict_model(spec, rows):
    """Predict damage for rows under a frozen model spec. Rows lacking a sign the model never saw
    predict at the other sign's iso floor — counted, never silent."""
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        s = spec["signs"].get(r["sign"]) or next(v for v in spec["signs"].values() if v)
        D = r["descriptor_vec"].reshape(1, -1)
        if spec["level"] >= 2:
            log_e, extras = _t123_transform(D, radial_discount=spec["radial_discount"],
                                            n_eff_l0=spec["n_eff_l0"])
        else:
            log_e = D[:, DESCRIPTOR_FIELDS.index("log_energy")]
            extras = np.zeros((1, 0))
        feats = []
        if spec["level"] >= 1:
            feats.append(D[:, len(DESCRIPTOR_FIELDS):])
        if spec["level"] >= 2:
            feats.append(extras)
        if spec["level"] >= 3:
            v1p = D[:, DESCRIPTOR_FIELDS.index("v1_overlap_pos")]
            feats.append((v1p ** 2).reshape(-1, 1))
        X = np.concatenate(feats, axis=1) if feats else np.zeros((1, 0))
        pred = pava_predict(np.asarray(s["iso_x"]), np.asarray(s["iso_y"]), log_e)
        if X.shape[1] and len(s["coef"]):
            pred = pred + X @ np.asarray(s["coef"])
        out[i] = float(pred[0])
    return out


def leave_family_out(rows, level, families=None, **fit_kw):
    """The descriptive robustness panel: per held-out family, Spearman ρ of the model's
    predictions on that family when fit on all others. Every pattern is a finding about where
    structure lives; nothing here gates anything."""
    fams = families or sorted({r["family"] for r in rows if not r.get("void")})
    panel = {}
    for f in fams:
        train = [r for r in rows if r["family"] != f and not r.get("void")]
        test = [r for r in rows if r["family"] == f and not r.get("void")]
        if len(test) < 3 or len(train) < 8:
            panel[f] = {"n_test": len(test), "rho": None, "reason": "too few cells"}
            continue
        spec = fit_model(train, level, **fit_kw)
        pred = predict_model(spec, test)
        y = np.asarray([r["damage_ppl"] for r in test])
        panel[f] = {"n_test": len(test), "rho": spearman(pred, y)}
    return panel


# ===========================================================================
# Rank statistics (no scipy dependency for the frozen rules).
# ===========================================================================
def _ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1, dtype=float)
    # midranks for ties
    uv = np.unique(v)
    if uv.size < v.size:
        for u in uv:
            m = v == u
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    assert den > 0, "degenerate ranking (all-tied input)"
    return float((ra * rb).sum() / den)


def kendall_tau_b(a, b):
    """Vectorized τ-b (the exact permutation test enumerates 9! = 362 880 permutations; a
    pure-Python pair loop costs minutes per call — this is the O(n²)-numpy version)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    iu = np.triu_indices(n, 1)
    da = np.sign(a[:, None] - a[None, :])[iu]
    db = np.sign(b[:, None] - b[None, :])[iu]
    ta = int((da == 0).sum())
    tb = int((db == 0).sum())
    prod = da * db
    conc = int((prod > 0).sum())
    disc = int((prod < 0).sum())
    n0 = n * (n - 1) / 2
    den = math.sqrt((n0 - ta) * (n0 - tb))
    assert den > 0, "degenerate tau (one variable fully tied)"
    return (conc - disc) / den


def perm_p_tau(a, b, n_perm=None, seed=1337):
    """Exact permutation p for Kendall τ when n ≤ 9 (9! = 362 880 — enumerate); one-sided
    P[τ_perm ≥ τ_obs]."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    obs = kendall_tau_b(a, b)
    n = len(a)
    if n <= 9:
        cnt = tot = 0
        for perm in itertools.permutations(range(n)):
            t = kendall_tau_b(a, b[list(perm)])
            cnt += (t >= obs - 1e-12)
            tot += 1
        return obs, cnt / tot
    rng = np.random.default_rng(seed)
    n_perm = n_perm or 10000
    cnt = sum(kendall_tau_b(a, b[rng.permutation(n)]) >= obs - 1e-12 for _ in range(n_perm))
    return obs, (cnt + 1) / (n_perm + 1)


def perm_p_spearman(a, b, n_perm=10000, seed=1337):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    obs = spearman(a, b)
    rng = np.random.default_rng(seed)
    cnt = sum(spearman(a, b[rng.permutation(len(b))]) >= obs - 1e-12 for _ in range(n_perm))
    return obs, (cnt + 1) / (n_perm + 1)


def bh_pass(pvals, q=0.10):
    """Benjamini–Hochberg within one named family. Returns the boolean pass vector."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    passed = np.zeros(m, bool)
    thresh = 0.0
    for rank, idx in enumerate(order, start=1):
        if p[idx] <= q * rank / m:
            thresh = p[idx]
    passed[p <= thresh] = thresh > 0
    return passed


def pair_bootstrap_drho(pred_a, pred_b, y, n_boot=10000, seed=1337):
    """Δρ = ρ(pred_a, y) − ρ(pred_b, y) with a 10k pair bootstrap (resampling cells, keeping the
    triplet together). Returns (drho, lo, hi, rho_a, rho_b)."""
    pa, pb, yy = map(np.asarray, (pred_a, pred_b, y))
    n = len(yy)
    rng = np.random.default_rng(seed)
    obs = spearman(pa, yy) - spearman(pb, yy)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(yy[idx]).size < 3 or np.unique(pa[idx]).size < 3 or np.unique(pb[idx]).size < 3:
            continue
        boots.append(spearman(pa[idx], yy[idx]) - spearman(pb[idx], yy[idx]))
    b = np.asarray(boots)
    return (float(obs), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
            spearman(pa, yy), spearman(pb, yy))


# ===========================================================================
# The sealed-cell grammar + adversarial selection (frozen algorithm; deterministic).
# ===========================================================================
ADD_MODES = ("dc", "sin4", "white")
DEL_MODES = ("shared_subspace", "pertoken_subspace")
CHANNELS = ("none", "k_low", "k_high", "v", "q")
RT_PROJ = ("none", "radial", "tangential")


def enumerate_grammar(sb_ranks, radial_mass_by_sb, corpus_keys, degenerate_keys=()):
    """Every structurally valid grammar cell, minus already-run corpus cells, minus dry-run-proven
    degenerates. Every exclusion carries a reason and is COUNTED — no silent caps.

    Structural validity (frozen):
      * add + channel=none: sub-band ∈ 16 × mode ∈ {dc, sin4, white} × rt ∈ {none, radial,
        tangential} × amp ∈ {0.1, 0.2}
      * add + channel≠none: injected inside attention → sub-band/rt do not apply
        (mode ∈ {dc, white}, amp ∈ {0.1, 0.2})
      * delete: sub-band ∈ 16 × mode ∈ {shared, per-token subspace} × frac ∈ {0.25, 1.0};
        per-token twin needs sub-band rank ≥ 2 (rank-1 per-token ≡ shared: degenerate)
      * radial × sub-band with radial mass < 1e-3: degenerate (the spec's own example)
    """
    cells, excluded = [], []

    def push(c):
        k = cell_key(c)
        if k in corpus_keys:
            excluded.append({"key": k, "reason": "already-run (corpus)"})
        elif k in degenerate_keys:
            excluded.append({"key": k, "reason": "dry-run degenerate"})
        else:
            cells.append(c)

    for sb in range(len(sb_ranks)):
        for mode in ADD_MODES:
            for rt in RT_PROJ:
                if rt == "radial" and radial_mass_by_sb[sb] < RADIAL_DEGENERATE_FRAC:
                    excluded.append({"key": f"add/sb{sb}/{mode}/{rt}", "reason": "no radial mass in sub-band"})
                    continue
                for amp in GRAMMAR_AMPS_ADD:
                    push({"sign": "add", "subband": sb, "mode": mode, "channel": "none",
                          "rt": rt, "amp": amp})
    for ch in CHANNELS[1:]:
        for mode in ("dc", "white"):
            for amp in GRAMMAR_AMPS_ADD:
                push({"sign": "add", "subband": None, "mode": mode, "channel": ch,
                      "rt": "none", "amp": amp})
    for sb in range(len(sb_ranks)):
        for mode in DEL_MODES:
            if mode == "pertoken_subspace" and sb_ranks[sb] < 2:
                excluded.append({"key": f"del/sb{sb}/{mode}", "reason": "rank<2: per-token ≡ shared"})
                continue
            for frac in GRAMMAR_FRACS_DEL:
                push({"sign": "delete", "subband": sb, "mode": mode, "channel": "none",
                      "rt": "none", "amp": frac})
    return cells, excluded


def cell_key(c):
    return f"{c['sign']}/sb{c['subband']}/{c['mode']}/{c['channel']}/{c['rt']}/a{c['amp']}"


def select_sealed(cands, pred_m2, pred_m0, n=SEALED_N_REGISTERED):
    """Rank by |M2−M0| predicted log-damage disagreement; greedy under the (sub-band × mode) cap;
    then repair coverage minima (≥3 per major-axis value present among candidates) by swapping in
    the highest-disagreement cells of the missing value. Deterministic. Returns the ordered list
    of indices into `cands`."""
    dis = np.abs(np.asarray(pred_m2) - np.asarray(pred_m0))
    order = list(np.argsort(-dis, kind="stable"))

    def bm_of(i):
        return (cands[i]["subband"], cands[i]["mode"])

    chosen, bandmode = [], {}
    for i in order:
        if bandmode.get(bm_of(i), 0) >= COVERAGE_MAX_BANDMODE:
            continue
        chosen.append(i)
        bandmode[bm_of(i)] = bandmode.get(bm_of(i), 0) + 1
        if len(chosen) == n:
            break

    def axis_vals(axis):
        return sorted({str(c[axis]) for c in cands})

    # coverage repair MAINTAINS the (sub-band × mode) cap — the first D-suite run constructed a
    # repair that silently violated it (the check exists precisely to catch this)
    for axis in ("sign", "mode", "channel", "rt"):
        for val in axis_vals(axis):
            have = [i for i in chosen if str(cands[i][axis]) == val]
            pool = [i for i in order if str(cands[i][axis]) == val and i not in chosen]
            while len(have) < COVERAGE_MIN and pool:
                inc = pool.pop(0)
                if bandmode.get(bm_of(inc), 0) >= COVERAGE_MAX_BANDMODE:
                    continue
                surplus = [j for j in reversed(chosen)
                           if len([k for k in chosen if str(cands[k][axis]) == str(cands[j][axis])])
                           > COVERAGE_MIN and j not in have]
                if not surplus:
                    break
                out_j = surplus[0]
                chosen.remove(out_j)
                bandmode[bm_of(out_j)] -= 1
                chosen.append(inc)
                bandmode[bm_of(inc)] = bandmode.get(bm_of(inc), 0) + 1
                have.append(inc)
    return chosen[:n]


def grammar_cell_descriptor(cell, env):
    """Descriptor for a GRAMMAR cell — from its construction + banked Σ objects only, never a
    forward. `env` supplies: sb_slices, sb_ranks, radial_mass_by_sb, retention_by_channel (from
    the config-derived Dirichlet factor at T=271), spectrum, d.

    Returns (descriptor dict, subband16 fractions)."""
    d = env["d"]
    dd = {k: 0.0 for k in DESCRIPTOR_FIELDS}
    dd["retention"] = 1.0
    n_sb = len(env["sb_slices"])
    if cell["sign"] == "add" and cell["channel"] == "none":
        sbi = cell["subband"]
        fr = [0.0] * n_sb
        fr[sbi] = 1.0
        eps = float(cell["amp"])
        coh = 1.0 if cell["mode"] in ("dc", "sin4") else 0.0
        dc = 1.0 if cell["mode"] == "dc" else 0.0
        rmass = env["radial_mass_by_sb"][sbi]
        if cell["rt"] == "radial":
            radial_frac = 1.0
        elif cell["rt"] == "tangential":
            radial_frac = 0.0
        else:
            radial_frac = rmass
        dd.update({"log_energy": math.log10(eps ** 2), "dc_frac": dc, "shared_dir": coh,
                   "mean_shift": eps * dc, "radial_frac": radial_frac, "chan_none": 1.0})
        return dd, fr
    if cell["sign"] == "add":
        ch = cell["channel"]
        eps = float(cell["amp"])
        dc = 1.0 if cell["mode"] == "dc" else 0.0
        dd.update({"log_energy": math.log10(eps ** 2), "dc_frac": dc, "shared_dir": dc,
                   "chan_none": 0.0, f"chan_{ch}": 1.0,
                   "retention": env["retention_by_channel"].get(ch, 1.0) if dc else 0.0,
                   "mean_shift": 0.0})
        return dd, [0.0] * n_sb
    # delete
    sbi = cell["subband"]
    fr = [0.0] * n_sb
    fr[sbi] = 1.0
    lo, hi = env["sb_slices"][sbi]
    seg = env["spectrum"][lo:hi]
    kp = max(1, (hi - lo) // 2)
    E_sb = float(seg.sum() / env["spectrum"].sum())
    E_removed = E_sb * (kp / max(hi - lo, 1)) * float(cell["amp"])     # frac = amp axis on delete
    dd.update({"is_delete": 1.0, "log_energy": math.log10(max(E_removed, 1e-12)),
               "dc_frac": 1.0 if cell["mode"] == "shared_subspace" else 0.0,
               "shared_dir": 1.0 if cell["mode"] == "shared_subspace" else 0.0,
               "chan_none": 1.0})
    return dd, fr


# ===========================================================================
# Confirmatory adjudicators. Failure branches all constructible (D-suite asserts).
# ===========================================================================
def adjudicate_rb1(*, damages, v1_overlap_pos, e1_overlap_abs, norms):
    """R-B1 (zero forwards). FAILS H_one-boundary when τ_e1 wins with the same margin
    (`H_two-mechanisms-confirmed` — the reading built to be overturned is STRENGTHENED) or when
    neither orders (`H_neither-orders`, the residual regression is banked for the successor).

    A fully-tied one-sided regressor (every banked draw on the dead side of the boundary,
    ⟨δ̄,v₁⟩₊ ≡ 0) is a REACHABLE, informative landing, not an error: the boundary term predicts no
    differential damage among these draws, so it cannot carry their spread — τ_v1 := 0, p := 1."""

    def _tau(a, b):
        if np.unique(np.asarray(b, float)).size < 2:
            return 0.0, 1.0, True
        t, p = perm_p_tau(a, b)
        return t, p, False

    tau_v1, p_v1, v1_degen = _tau(damages, v1_overlap_pos)
    tau_e1, _, e1_degen = _tau(damages, e1_overlap_abs)
    tau_nrm, _, nrm_degen = _tau(damages, norms)
    if tau_v1 >= 0.6 and p_v1 <= 0.05 and tau_v1 > max(tau_e1, tau_nrm):
        v = "H_one-boundary-lead"
    elif tau_e1 >= 0.6 and tau_e1 > max(tau_v1, tau_nrm):
        v = "H_two-mechanisms-confirmed"
    else:
        v = "H_neither-orders"
    return {"verdict": v, "tau_v1": tau_v1, "p_v1": p_v1, "tau_e1": tau_e1, "tau_norm": tau_nrm,
            "v1_regressor_degenerate": v1_degen, "e1_regressor_degenerate": e1_degen,
            "norm_regressor_degenerate": nrm_degen}


def adjudicate_rw(*, dfit, rho_lamrho, dfit_shuffle):
    """R-W (per proxy; BH across the 2-proxy family upstream). FAILS H_visible-variance when
    Δfit ≤ 0 (`H_other-order`: the residual profile Ω/λ becomes the named successor object; the ρ
    battery itself is a new artifact either way)."""
    if dfit >= 0.15 and rho_lamrho >= 0.6 and dfit_shuffle <= 0:
        return {"verdict": "H_visible-variance", "dfit": dfit, "rho": rho_lamrho}
    if dfit <= 0:
        return {"verdict": "H_other-order", "dfit": dfit,
                "reading": "the error-input correlation is not the missing factor; Ω/λ is the "
                           "named residual object"}
    return {"verdict": "H_partial", "dfit": dfit,
            "reading": "which layers/types clear is the stratification finding — a map, reported "
                       "as one"}


def adjudicate_rn(*, gap, model_gap, constant_gap):
    """R-N. FAILS H_maxent-sufficient when the tilt model misses the held-out curve beyond 0.10
    OR does not beat the constant baseline (the null's null, §5) — beating zero was never the bar."""
    beats_constant = model_gap < constant_gap
    if gap <= 0.10 and beats_constant:
        return {"verdict": "H_maxent-sufficient", "gap": gap}
    return {"verdict": "H_deviates", "gap": gap, "beats_constant": bool(beats_constant),
            "reading": "the deviation profile (E/E_max region, sign) names where the real ensemble "
                       "is more/less constrained than max-entropy; the sampler is kept either way"}


def adjudicate_rx(*, gap_frame, gap_conjugated):
    """R-X. FAILS H_connection when closure < 0.5 (`H_higher-order`: quantifies exactly how
    non-linear the inter-site map is — a number nothing in the lineage has measured)."""
    assert gap_frame > 0, "frame gap must be recomputed from the artifact and positive"
    closure = 1.0 - gap_conjugated / gap_frame
    if gap_conjugated <= 0.10:
        return {"verdict": "H_connection", "closure": closure, "gap_conjugated": gap_conjugated}
    if closure < 0.5:
        return {"verdict": "H_higher-order", "closure": closure, "gap_conjugated": gap_conjugated}
    return {"verdict": "H_partial", "closure": closure, "gap_conjugated": gap_conjugated}


def adjudicate_rm(*, drho, drho_cilo, drho_cihi, rho_m2, rho_m0, n_void_backups):
    """R-M (primary, per readout; BH across the 2-readout family upstream). All five branches.
    FAILS H_metric when energy survives cells CHOSEN ADVERSARIALLY to break it
    (`H_energy-sufficient` — the strongest form of the R1b law ever measured)."""
    if n_void_backups > 10:
        return {"verdict": "VOID-battery",
                "response": "re-freeze once at half amplitude with a dated label (the one "
                            "pre-specified re-freeze)"}
    if drho_cilo > 0 and rho_m2 >= 0.5:
        return {"verdict": "H_metric", "drho": drho, "rho_m2": rho_m2, "rho_m0": rho_m0}
    if rho_m0 >= 0.5 and drho_cihi < 0.1:
        return {"verdict": "H_energy-sufficient", "drho": drho, "rho_m2": rho_m2, "rho_m0": rho_m0}
    if rho_m2 >= 0.5 and rho_m0 >= 0.5:
        return {"verdict": "H_partial", "drho": drho, "rho_m2": rho_m2, "rho_m0": rho_m0,
                "reading": "both models carry real overlapping structure; the per-cell residual "
                           "decomposition is the successor's map"}
    return {"verdict": "H_neither", "drho": drho, "rho_m2": rho_m2, "rho_m0": rho_m0,
            "reading": "the surface holds structure neither grammar expresses at this grain; the "
                       "sealed cells + residuals are the best-characterized anomaly set in the lineage"}


# ===========================================================================
# Corpus manifest utilities.
# ===========================================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_entry(*, cell_id, source, family, tag, sign, damage_ppl, damage_vqa, descriptor,
                   subband16, artifact_path, artifact_sha256, void=False, exclude_reason=None):
    """One corpus row. `damage_ppl` is log(1+Δppl) (frozen scale); raw Δppl is stored inside
    `descriptor['raw_dppl']` so nothing is lost to the transform."""
    return {"cell_id": cell_id, "source": source, "family": family, "tag": tag, "sign": sign,
            "damage_ppl": damage_ppl, "damage_vqa": damage_vqa, "descriptor": descriptor,
            "subband16": subband16, "artifact_path": artifact_path,
            "artifact_sha256": artifact_sha256, "void": void, "exclude_reason": exclude_reason}


def manifest_hash(rows):
    blob = json.dumps(rows, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
