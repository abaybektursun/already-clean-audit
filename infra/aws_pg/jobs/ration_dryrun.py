"""RATION D-suite (CPU, weight-free) — spec §9.2: every mathematical claim in `ration_core` proven
on synthetic tensors, EVERY §6 verdict branch constructed (including each VOID/halt branch), and the
one-sided-thresholds replays (a known VOID event and a known halt event run through the guards). Run
on a pg r7i instance (never the laptop); D-suite green is the freeze precondition.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ration_dryrun --type r7i.2xlarge \
      --no-weights --s3-prefix results/ration_dryrun --budget-min 30
"""
import argparse
import json
import os
import pathlib
import time

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ===========================================================================
# Water-filling: holds b̄ exactly, monotone, tied→uniform, pinned.
# ===========================================================================
@check
def water_fill_holds_average_exactly():
    import numpy as np

    from infra.pg import ration_core as RC
    rng = np.random.default_rng(0)
    for d in (64, 512, 2048):
        w = rng.lognormal(0, 2.0, d)
        for b_bar in RC.BUDGET_GRID:
            b = RC.water_fill(w, b_bar, b_max=RC.B_MAX)
            assert b.sum() == round(b_bar * d), (d, b_bar, b.sum(), round(b_bar * d))
            assert b.min() >= 0 and b.max() <= RC.B_MAX
    # monotone: a coordinate with larger importance gets ≥ as many bits (rank correlation ~ +1)
    w = np.sort(rng.lognormal(0, 1.5, 256))
    b = RC.water_fill(w, 4, b_max=RC.B_MAX)
    rx = np.argsort(np.argsort(w))
    ry = np.argsort(np.argsort(b))
    rho = float(np.corrcoef(rx, ry)[0, 1])                          # Spearman via rank-Pearson
    assert rho > 0.95, f"water-fill not monotone in importance: rho={rho}"


@check
def water_fill_tied_is_uniform():
    import numpy as np

    from infra.pg import ration_core as RC
    b = RC.water_fill(np.full(128, 3.7), 5, b_max=RC.B_MAX)
    assert np.all(b == 5), "tied importances must give uniform bits (E16 rule)"
    assert b.sum() == 5 * 128


@check
def water_fill_pins_sink_and_holds_average():
    import numpy as np

    from infra.pg import ration_core as RC
    d = 2048
    w = np.random.default_rng(1).lognormal(0, 2.0, d)
    pinned = {c: RC.B_MAX for c in RC.MASSIVE_COORDS}
    b = RC.water_fill(w, 4, b_max=RC.B_MAX, pinned=pinned)
    for c in RC.MASSIVE_COORDS:
        assert b[c] == RC.B_MAX, f"sink coord {c} not pinned to b_max"
    assert b.sum() == round(4 * d), "average not held after pinning"


# ===========================================================================
# In-basis quantiser: identity lossless, b=0 → centre, rotation round-trips.
# ===========================================================================
@check
def quantiser_bmax_is_lossless_and_b0_is_centre():
    import torch

    from infra.pg import ration_core as RC
    gen = torch.Generator().manual_seed(2)
    d, n = 64, 200
    R, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    X = torch.randn(n, d, generator=gen, dtype=torch.float64) * 3.0 + 1.5
    centers = X.mean(0)
    stds = X.std(0)
    # b_max on every coord → near-lossless. NOT exact: the ±K_SIGMA·σ clip discards the rare tail
    # (~1e-4 rel floor), which is the quantiser's inherent behaviour. EXACT losslessness is the
    # PASSTHROUGH path (spec=None), checked in substitutor_passthrough_and_substitute — that is what
    # the GPU preflight uses to prove hook transparency, NOT b_max.
    bits = torch.full((d,), RC.B_MAX, dtype=torch.int64)
    Xh = RC.quantize_in_basis(X, R, bits, centers, stds)
    rel = float(((X - Xh) ** 2).sum() / (X ** 2).sum())
    assert rel < 1e-3, f"b_max quantiser not near-lossless (clip-limited): rel={rel}"
    # b=0 on every coord → every reconstruction is the centre (in basis R), i.e. X̂ = R·centres
    bits0 = torch.zeros(d, dtype=torch.int64)
    Xh0 = RC.quantize_in_basis(X, R, bits0, centers, stds)
    target = (centers @ R.transpose(0, 1)).expand(n, d)
    assert torch.allclose(Xh0, target, atol=1e-9), "b=0 must collapse to the centre level"


@check
def basis_stats_match_direct_projection():
    import torch

    from infra.pg import ration_core as RC
    gen = torch.Generator().manual_seed(3)
    d, n = 48, 5000
    R, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    X = torch.randn(n, d, generator=gen, dtype=torch.float64) @ torch.diag(
        torch.linspace(0.5, 4.0, d, dtype=torch.float64)) + 2.0
    mean_x = X.mean(0)
    C_x = torch.cov(X.T, correction=0)
    centers, stds = RC.basis_stats(R, mean_x, C_x)
    C = X @ R
    assert torch.allclose(centers, C.mean(0), atol=1e-8), "centres wrong"
    assert torch.allclose(stds, C.std(0, correction=0), atol=1e-6), "stds wrong"


# ===========================================================================
# The mirror's core: variance (KLT + water-fill) is MSE-optimal — beats a random
# basis AND the susceptibility (Q) basis on reconstruction MSE at matched budget.
# ===========================================================================
@check
def variance_water_filling_minimises_mse():
    import numpy as np
    import torch

    from infra.pg import ration_core as RC
    gen = torch.Generator().manual_seed(4)
    d, n = 32, 20000
    # anisotropic Gaussian: sharply spread spectrum so the KLT advantage is large
    sig = torch.logspace(1.0, -1.0, d, dtype=torch.float64)          # σ from 10 → 0.1
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    C_x = U @ torch.diag(sig ** 2) @ U.transpose(0, 1)
    X = torch.randn(n, d, generator=gen, dtype=torch.float64) @ (U * sig).transpose(0, 1)
    mean_x = X.mean(0)

    def mse(R, w, b_bar, pinned=None):
        bits = RC.water_fill(np.asarray(w, float), b_bar, pinned=pinned)
        centers, stds = RC.basis_stats(R, mean_x, C_x)
        Xh = RC.quantize_in_basis(X, torch.as_tensor(R, dtype=torch.float64), bits, centers, stds)
        return float(((X - Xh) ** 2).mean())

    sev, sea = RC._sigma_eig(C_x)
    # a Q-like basis with a DIFFERENT ordering (random rotation of the eigvecs) + arbitrary w
    Qrot, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    for b_bar in (2, 3, 4):
        mse_var = mse(sev, sea.numpy(), b_bar)                       # KLT + variance water-fill
        mse_rand = mse(Qrot, np.ones(d), b_bar)                      # random basis, uniform bits
        mse_qbad = mse(Qrot, sea.numpy(), b_bar)                     # mismatched basis+w
        assert mse_var < mse_rand, (b_bar, mse_var, mse_rand)
        assert mse_var < mse_qbad, (b_bar, mse_var, mse_qbad)


# ===========================================================================
# Policy builders, sink detection, oracle expansion.
# ===========================================================================
@check
def policy_builders_and_sink_detection():
    import numpy as np
    import torch

    from infra.pg import ration_core as RC
    gen = torch.Generator().manual_seed(5)
    d = 64
    Q_evecs, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    Q_evals = torch.linspace(10.0, 0.1, d, dtype=torch.float64).numpy()
    # plant a sink: make eigvec 0 concentrate on a chosen coord
    sink_coord = 7
    e = torch.zeros(d, dtype=torch.float64)
    e[sink_coord] = 1.0
    sev = Q_evecs.clone()
    sev[:, 0] = e                                                    # eigvec 0 = pure sink coordinate
    sev, _ = torch.linalg.qr(sev)                                   # re-orthonormalise
    dirs = RC.sink_eigendirections(sev, massive_coords=(sink_coord,), frac=0.5)
    assert len(dirs) >= 1, "planted sink direction not detected"
    banks = {"Q_evecs": Q_evecs, "Q_evals": Q_evals, "sigma_evecs": sev,
             "sigma_evals": np.linspace(5.0, 0.2, d), "mean_x": np.zeros(d),
             "absmean_x": np.abs(np.random.default_rng(0).normal(0, 1, d)),
             "C_x": np.eye(d), "sink_dirs": dirs, "seed": 1337}
    for name in RC.POLICIES:
        if name == "oracle":
            banks["oracle_w"] = np.linspace(1.0, 0.01, d)
        spec = RC.make_spec(name, 4, banks)
        assert spec["bits"].sum() == round(4 * d), (name, spec["bits"].sum())
    # sink-aware pins the detected directions
    sa = RC.make_spec("sink_aware_variance", 4, banks)
    for j in dirs:
        assert sa["bits"][j] == RC.B_MAX, "sink-aware did not pin the sink directions"
    # random == shuffled susceptibility λ (same basis, permuted importance)
    su = RC.make_spec("susceptibility", 4, banks)
    ra = RC.make_spec("random", 4, banks)
    assert torch.allclose(su["R"], ra["R"]), "random must share the Q basis"
    assert not np.array_equal(su["bits"], ra["bits"]), "random must permute the ranking"


@check
def oracle_importance_expands_bands():
    import numpy as np

    from infra.pg import ration_core as RC
    d, nb = 256, RC.N_ORACLE_BANDS
    dmg = np.linspace(1.0, 0.0, nb)                                  # top bands hurt most
    w = RC.oracle_importance(dmg, d, nb)
    assert w.shape == (d,) and w[0] > w[-1], "oracle importance not top-heavy"
    edges = RC.oracle_band_edges(d, nb)
    bb = RC.band_bits(d, edges[3], edges[4])
    assert (bb[edges[3]:edges[4]] == 0).all() and (bb[:edges[3]] == RC.B_MAX).all()
    # all-zero band damage → uniform (degenerate-safe)
    assert np.all(RC.oracle_importance(np.zeros(nb), d, nb) == 1.0)


# ===========================================================================
# Paired-excess bootstrap: two-sided (positive, negative, CI-contains-0).
# ===========================================================================
@check
def paired_excess_two_sided():
    import numpy as np

    from infra.pg import ration_core as RC
    rng = np.random.default_rng(6)
    n = 1000
    base = rng.integers(0, 2, n).astype(float)
    a = base.copy()
    b = base.copy()
    degrade = (base == 1) & (rng.random(n) < 0.15)                  # only worsen correct items
    b[degrade] = 0.0                                                # b strictly ≤ a → E>0
    r = RC.paired_excess(a, b)
    assert r["ci_lo"] > 0, f"A clearly better → E CI-lo must exceed 0: {r}"
    r2 = RC.paired_excess(b, a)
    assert r2["ci_hi"] < 0, "sign flips under argument swap (paired antisymmetry)"
    # equal policies → CI straddles 0 (cannot spuriously fire)
    r3 = RC.paired_excess(base, base.copy())
    assert r3["ci_lo"] <= 0 <= r3["ci_hi"], f"identical policies must straddle 0: {r3}"


# ===========================================================================
# adjudicate_ration — EVERY §6 branch constructed, guards two-sided.
# ===========================================================================
def _pe(ci_lo, ci_hi=None):
    return {"E": (ci_lo + (ci_hi if ci_hi is not None else ci_lo + 0.1)) / 2,
            "ci_lo": ci_lo, "ci_hi": ci_hi if ci_hi is not None else ci_lo + 0.1, "se": 0.01, "n": 500}


@check
def adjudicate_all_branches():
    from infra.pg import ration_core as RC
    dr = 0.02                                                       # frozen delta_range
    uni = {"E": 0.10, "se": 0.006, "ci_lo": 0.08, "ci_hi": 0.12}   # measurable range
    ok_head = {"E": 0.25}                                           # oracle ≫ uniform
    mir_ok = {"val": 0.4}                                           # variance wins MSE

    # H_prescriptive
    r = RC.adjudicate_ration(E_vs_sinkaware=_pe(0.03), E_vs_naive=_pe(0.05),
                             uniform_loss=uni, oracle_headroom=ok_head, delta_range=dr,
                             random_saturated=False, mirror_mse_excess=mir_ok)
    assert r["verdict"] == "H_prescriptive", r
    # H_sink-only (beats naive, not sink-aware)
    r = RC.adjudicate_ration(E_vs_sinkaware=_pe(-0.01, 0.02), E_vs_naive=_pe(0.03),
                             uniform_loss=uni, oracle_headroom=ok_head, delta_range=dr,
                             random_saturated=False, mirror_mse_excess=mir_ok)
    assert r["verdict"] == "H_sink-only", r
    # H_energy-suffices (beats neither)
    r = RC.adjudicate_ration(E_vs_sinkaware=_pe(-0.02, 0.01), E_vs_naive=_pe(-0.01, 0.02),
                             uniform_loss=uni, oracle_headroom=ok_head, delta_range=dr,
                             random_saturated=False, mirror_mse_excess=mir_ok)
    assert r["verdict"] == "H_energy-suffices", r
    # VOID-no-range: uniform below delta_range (the floor did not clear its noise)
    r = RC.adjudicate_ration(E_vs_sinkaware=_pe(0.03), E_vs_naive=_pe(0.05),
                             uniform_loss={"E": 0.005, "se": 0.006, "ci_lo": -0.01, "ci_hi": 0.02},
                             oracle_headroom=ok_head, delta_range=dr,
                             random_saturated=False, mirror_mse_excess=mir_ok)
    assert r["verdict"] == "VOID-no-range", r
    # VOID-no-range: oracle headroom below the ceiling (no dynamic range to allocate)
    r = RC.adjudicate_ration(E_vs_sinkaware=_pe(0.03), E_vs_naive=_pe(0.05), uniform_loss=uni,
                             oracle_headroom={"E": 0.01}, delta_range=dr,
                             random_saturated=False, mirror_mse_excess=mir_ok)
    assert r["verdict"] == "VOID-no-range", r
    # VOID-no-range: saturation (random == uniform == lossless)
    r = RC.adjudicate_ration(E_vs_sinkaware=_pe(0.03), E_vs_naive=_pe(0.05), uniform_loss=uni,
                             oracle_headroom=ok_head, delta_range=dr,
                             random_saturated=True, mirror_mse_excess=mir_ok)
    assert r["verdict"] == "VOID-no-range", r
    # halt-miswired: variance does NOT beat susceptibility on MSE (guard precedes the hypotheses)
    r = RC.adjudicate_ration(E_vs_sinkaware=_pe(0.03), E_vs_naive=_pe(0.05), uniform_loss=uni,
                             oracle_headroom=ok_head, delta_range=dr,
                             random_saturated=False, mirror_mse_excess={"val": -0.1})
    assert r["verdict"] == "halt-miswired", r


@check
def guard_order_void_precedes_prescriptive():
    """One-sided-thresholds replay: a would-be H_prescriptive with NO range must land VOID, never
    the headline — the range/saturation ceiling dominates a positive floor (feedback_one_sided...)."""
    from infra.pg import ration_core as RC
    r = RC.adjudicate_ration(E_vs_sinkaware=_pe(0.05), E_vs_naive=_pe(0.08),
                             uniform_loss={"E": 0.001, "se": 0.006, "ci_lo": -0.01, "ci_hi": 0.01},
                             oracle_headroom={"E": 0.30}, delta_range=0.02,
                             random_saturated=False, mirror_mse_excess={"val": 0.5})
    assert r["verdict"] == "VOID-no-range", "a positive floor must NOT override a dead range guard"


# ===========================================================================
# The substitutor hook: passthrough is exact identity; spec SUBSTITUTES (not adds).
# ===========================================================================
@check
def substitutor_passthrough_and_substitute():
    import torch

    from infra.pg import ration_core as RC

    class Blk(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states                                    # identity block: output == input

    layers = [Blk()]
    install, holder = RC.make_residual_substitutor(layers, 0)
    gen = torch.Generator().manual_seed(7)
    d, T = 32, 200                                                  # T large → stable per-coord stats
    hs = torch.randn(1, T, d, generator=gen, dtype=torch.float64)

    # passthrough (spec=None) → exact identity
    holder["spec"] = None
    with install():
        out = layers[0](hs)
    assert torch.equal(out, hs), "passthrough must be the exact identity (no delta, no quantise)"

    # substitute: b_max spec → near-lossless but NOT an additive delta (it REPLACES)
    R, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    centers = hs.reshape(-1, d).mean(0)
    stds = hs.reshape(-1, d).std(0)
    holder["spec"] = {"R": R, "bits": torch.full((d,), RC.B_MAX, dtype=torch.int64),
                      "centers": centers, "stds": stds}
    holder["collect_mse"] = True
    with install():
        out = layers[0](hs)
    rel = float(((out - hs) ** 2).sum() / (hs ** 2).sum())
    assert rel < 1e-3, f"b_max substitute not near-lossless (clip-limited): rel={rel}"
    # a coarse spec REPLACES the residual (output != input + a small delta of the SAME residual):
    holder["spec"]["bits"] = torch.full((d,), 2, dtype=torch.int64)
    with install():
        out2 = layers[0](hs)
    # the reconstruction is a quantised version of hs, so it differs and is NOT hs + f(prev delta)
    assert not torch.allclose(out2, hs, atol=1e-3), "coarse spec must visibly change the residual"
    assert holder.get("mse_cnt", 0) > 0, "MSE accumulator did not record"


@check
def capture_prehook_is_transparent():
    """A Σ_x capture forward_pre_hook must NOT modify the forward — it returns None. Returning a value
    would REPLACE the layer inputs (the run-5 bug: `return x.shape[0]` corrupted the forward)."""
    import torch

    class Blk(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 2.0                       # non-identity → corruption is visible

    layers = [Blk()]
    state = {"n": 0}

    def hook(_m, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        state["n"] += hs.reshape(-1, hs.shape[-1]).shape[0]
        # returns None → transparent (the capture pattern of _ration_shared.accumulate_sigma)

    h = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    x = torch.randn(1, 5, 8)
    out = layers[0](x)
    h.remove()
    assert torch.equal(out, x * 2.0), "capture pre-hook corrupted the forward (returned non-None?)"
    assert state["n"] == 5, "capture pre-hook did not accumulate row count"


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "ration_dryrun_v1", "checks": []}
    t00 = time.time()
    for fn in CHECKS:
        t0 = time.time()
        fn()
        rep["checks"].append({"name": fn.__name__, "ok": True, "seconds": round(time.time() - t0, 2)})
        (out / "dryrun.json").write_text(json.dumps(rep, indent=2))
        print(f"[ration-dryrun] {fn.__name__} OK", flush=True)
    rep["n_checks"] = len(rep["checks"])
    rep["wall_seconds"] = round(time.time() - t00, 1)
    rep["gpu_seconds"] = {"total": 0.0, "note": "CPU job"}
    rep["green"] = True
    (out / "dryrun.json").write_text(json.dumps(rep, indent=2))
    print("RATION_DRYRUN_OK", json.dumps({"n": rep["n_checks"]}))


if __name__ == "__main__":
    main()
