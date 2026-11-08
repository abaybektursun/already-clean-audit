"""HUMBOLDT — the per-arm planted-structure smoke tests (§3, gate G5).

G5 (§1 kill-table): "Any recorded statistic is BROKEN if its planted-structure smoke test does not
move the statistic by ≥10× its twin spread." Each function below plants a structure the statistic
MUST see, measures the statistic on the planted and on a matched unstructured control, and returns
the evidence dict that `gates.json` records. They are pure numpy so the SAME code runs in the
control-plane dry-run and inside the job — a smoke test that only exists offline proves nothing
about the job that shipped.

FAIL LOUD. No try/except.
"""
from __future__ import annotations

G5_SEPARATION = 10.0        # the §1 factor. Frozen.


def smoke_a1_planted(n: int = 256, k0: int = 40, k: int = 32, n_ft: int = 8, m: int = 128,
                     seed: int = 1337) -> dict:
    """A1 smoke (§3/A1). Plant a k0-dim SHARED NULL subspace S: every ΔW_i = M_i (I − S Sᵀ), so
    every fine-tune leaves S exactly untouched. core32 of the consensus Gram must then sit inside S
    (all principal angles ≈ 0). The iid-random control (no shared null) must not."""
    import numpy as np
    from infra.pg.humboldt_core import principal_angles

    rng = np.random.default_rng(seed)
    S = np.linalg.qr(rng.standard_normal((n, k0)))[0][:, :k0]
    Pp = np.eye(n) - S @ S.T

    def _gram(dws):
        G = np.zeros((n, n), dtype=np.float64)
        for A in dws:
            G += (A.T @ A) / float((A ** 2).sum())
        return G

    planted = [rng.standard_normal((m, n)) @ Pp for _ in range(n_ft)]
    control = [rng.standard_normal((m, n)) for _ in range(n_ft)]

    out = {}
    for name, dws in (("planted", planted), ("control", control)):
        w, V = np.linalg.eigh(_gram(dws))
        ang = principal_angles(V[:, :k], S)
        out[name] = {"max_angle_rad": float(ang.max()), "mean_angle_rad": float(ang.mean()),
                     "bottom_eig_mean": float(np.clip(w, 0, None)[:k].mean())}
    sep = out["control"]["max_angle_rad"] / max(out["planted"]["max_angle_rad"], 1e-12)
    out.update({"separation": float(sep), "required": G5_SEPARATION,
                "ok": bool(sep >= G5_SEPARATION), "n": n, "k0": k0, "k": k, "n_ft": n_ft,
                "seed": seed,
                "statistic": "max principal angle between core32 and the planted shared null S"})
    return out


def smoke_a2_band(head_dim: int = 8, n: int = 4, seed: int = 1337) -> dict:
    """A2 smoke (§3/A2). A hand-built W that maps a chosen input direction ONLY onto band-0 dims
    (j = 0 and j = head_dim/2, the rotate_half pair sharing θ_0) must give coupling(0) ≈ 1 for that
    direction; a W that maps it onto band-1 dims must give coupling(0) ≈ 0."""
    import numpy as np
    from infra.pg.humboldt_core import rope_coupling

    nb = head_dim // 2
    U = np.zeros((n, 1)); U[0, 0] = 1.0
    out = {}
    for band in (0, 1):
        W = np.zeros((head_dim, n))
        W[band, 0] = 1.0
        W[band + nb, 0] = 1.0
        c = rope_coupling(W, U, head_dim=head_dim, n_out_heads=1)
        assert abs(float(c.sum()) - 1.0) < 1e-12, f"coupling does not sum to 1: {c.sum()}"
        out[f"planted_band{band}"] = {"coupling0": float(c[0]), "coupling1": float(c[1]),
                                      "argmax": int(np.argmax(c))}
    sep = out["planted_band0"]["coupling0"] / max(out["planted_band1"]["coupling0"], 1e-12)
    out.update({"separation": float(sep), "required": G5_SEPARATION,
                "ok": bool(out["planted_band0"]["argmax"] == 0
                           and out["planted_band1"]["argmax"] == 1
                           and sep >= G5_SEPARATION),
                "head_dim": head_dim, "n": n, "seed": seed,
                "statistic": "coupling(band 0) of a direction planted into band 0 vs band 1"})
    return out


def smoke_a3_haar(n: int = 512, k: int = 32, n_draw: int = 100, seed: int = 1337) -> dict:
    """A3 smoke (§3/A3). Principal angles between two INDEPENDENT random k-subspaces of ℝⁿ must
    land inside the recomputed Haar band (the band is recomputed here, never hardcoded)."""
    import numpy as np
    from infra.pg.humboldt_core import haar_angle_band, haar_subspace, principal_angles

    band = haar_angle_band(n, k, n_draw=n_draw, seed=seed)
    A = haar_subspace(n, k, seed + 10_001)
    B = haar_subspace(n, k, seed + 20_002)
    ang = principal_angles(A, B)
    lo = np.asarray(band["p2.5"]); hi = np.asarray(band["p97.5"])
    inside = (ang >= lo) & (ang <= hi)
    frac = float(inside.mean())
    # the negative half: a subspace compared against ITSELF must sit far below the band.
    self_ang = principal_angles(A, A)
    return {"frac_inside_band": frac, "ok": bool(frac >= 0.90),
            "self_max_angle_rad": float(self_ang.max()),
            "band_mean_first": float(band["mean"][0]), "band_mean_last": float(band["mean"][-1]),
            "observed_first": float(ang[0]), "observed_last": float(ang[-1]),
            "n": n, "k": k, "n_draw": n_draw, "seed": seed,
            "statistic": "fraction of principal angles inside the recomputed Haar [p2.5, p97.5]"}


def smoke_b1_projector(n: int = 512, k: int = 32, T: int = 64, seed: int = 1337) -> dict:
    """B1 smoke. Tokens planted ENTIRELY inside a bank must read e_U ≈ 1; tokens drawn isotropically
    must read e_U ≈ k/n. The readout must separate the two by ≥10×."""
    import numpy as np
    from infra.pg.humboldt_core import energy_fraction, haar_subspace

    rng = np.random.default_rng(seed)
    U = haar_subspace(n, k, seed)
    H_in = (rng.standard_normal((T, k)) @ U.T)
    H_iso = rng.standard_normal((T, n))
    e_in = energy_fraction(H_in, U)
    e_iso = energy_fraction(H_iso, U)
    sep = float(e_in.mean() / e_iso.mean())
    return {"e_planted_mean": float(e_in.mean()), "e_isotropic_mean": float(e_iso.mean()),
            "expected_isotropic": k / n, "separation": sep, "required": G5_SEPARATION,
            "ok": bool(sep >= G5_SEPARATION and abs(float(e_in.mean()) - 1.0) < 1e-9),
            "n": n, "k": k, "T": T, "seed": seed,
            "statistic": "e_U of in-bank tokens vs isotropic tokens"}


def smoke_b2_series(T: int = 2048, period: int = 16, seed: int = 1337) -> dict:
    """B2 smoke. A series with a planted period must show that period as the top spectral peak and
    a large autocorrelation at that lag; matched white noise must not."""
    import numpy as np
    from infra.pg.humboldt_core import autocorr, spectral_peaks

    rng = np.random.default_rng(seed)
    t = np.arange(T)
    planted = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(T)
    control = rng.standard_normal(T)
    out = {}
    for name, x in (("planted", planted), ("control", control)):
        pk = spectral_peaks(x, top=3)
        ac = autocorr(x, max_lag=64)
        out[name] = {"top_period": float(pk["period"][0]), "top_power": float(pk["power"][0]),
                     "acf_at_period": float(ac[period - 1])}
    sep = abs(out["planted"]["acf_at_period"]) / max(abs(out["control"]["acf_at_period"]), 1e-12)
    return {**out, "separation": float(sep), "required": G5_SEPARATION,
            "ok": bool(abs(out["planted"]["top_period"] - period) < 1.0 and sep >= G5_SEPARATION),
            "T": T, "period": period, "seed": seed,
            "statistic": "top spectral period + |ACF at the planted lag| vs white noise"}


def smoke_a4_projection(n: int = 256, k: int = 32, m: int = 128, seed: int = 1337) -> dict:
    """A4 smoke, structural half (the model half runs in the job on a held-out HOUSE fine-tune).
    A ΔW built to AVOID a bank must project near-zero onto it; a Haar-rotated copy of the SAME ΔW
    (identical spectrum, no alignment) must project at the chance level k/n."""
    import numpy as np
    from infra.pg.humboldt_core import haar_subspace

    rng = np.random.default_rng(seed)
    U = haar_subspace(n, k, seed)
    dW = rng.standard_normal((m, n)) @ (np.eye(n) - U @ U.T)
    R = np.linalg.qr(rng.standard_normal((n, n)))[0]
    dW_rot = dW @ R

    def frac(A):
        return float((np.linalg.norm(A @ U) ** 2) / (np.linalg.norm(A) ** 2))

    f_real, f_rot = frac(dW), frac(dW_rot)
    sep = f_rot / max(f_real, 1e-15)
    return {"frac_avoiding": f_real, "frac_haar_rotated": f_rot, "chance": k / n,
            "separation": float(sep), "required": G5_SEPARATION,
            "ok": bool(f_real < 1e-12 and abs(f_rot - k / n) < 5 * (k / n) and sep >= G5_SEPARATION),
            "n": n, "k": k, "m": m, "seed": seed,
            "statistic": "‖ΔW U‖²/‖ΔW‖² for an avoiding ΔW vs its Haar-rotated copy"}


ALL_OFFLINE_SMOKES = {
    "G5_A1_planted_shared_null": smoke_a1_planted,
    "G5_A2_rope_band": smoke_a2_band,
    "G5_A3_haar_angle_band": smoke_a3_haar,
    "G5_A4_projection": smoke_a4_projection,
    "G5_B1_activation_projector": smoke_b1_projector,
    "G5_B2_series": smoke_b2_series,
}


def run_all_offline(gates=None) -> dict:
    """Run every weight-free smoke; record into a Gates object if given. Returns {name: evidence}."""
    out = {}
    for name, fn in ALL_OFFLINE_SMOKES.items():
        ev = fn()
        out[name] = ev
        if gates is not None:
            # `ev` carries its own "ok"; passing it through as evidence would collide with the
            # positional argument. Strip it — the gate's verdict IS that value.
            gates.require(name, ev["ok"], **{k: v for k, v in ev.items() if k != "ok"})
        else:
            assert ev["ok"], f"{name} FAILED: {ev}"
    return out
