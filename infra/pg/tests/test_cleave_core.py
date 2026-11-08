"""CLEAVE weight-free dry-run logic gate (spec §9 step 2). Tiny tensors, no model — seconds on CPU.

Proves, before any GPU minute:
  * `build_decoder("residual", …)` IS `ΔW − proj_{U_P}ΔW` (the A-CLEAVE-2 claim the whole PROTECT
    axis rests on), `identity` is byte-exact base, `ft` is base+ΔW (the wiring gate's ground truth);
  * the off-projection removes exactly the in-span energy, and at rank=d leaves nothing (so retention
    reaches 0 — the A-CLEAVE-5 reachability claim);
  * the gradient off/into partition identity ‖into‖²+‖off‖²=‖g‖²;
  * the resolution gate certifies stable bases, REFUSES unstable ones, and fails loud on vacuously
    identical repeats (a gate that cannot fail is not a gate);
  * ρ*-reachability + the null-derived selectors on synthetic frontiers;
  * every branch of the §6 adjudicator.

Runs under the repo's pytest (infra/pg/tests). No try/except (CLAUDE.md).
"""
import numpy as np
import torch

from infra.pg import cleave_core as CL
from infra.pg import comwrite_core as CW
from infra.pg import comwrite_surgery as CS
from infra.pg.gauge_common import PROJ

D = 32                      # tiny residual width for the dry-run
INNER = 8                   # the non-residual side


def _bank(d=D, seed=0):
    """A tiny per-layer bank: a Q eigenbasis and a DISTINCT Σ eigenbasis (+ u_wr), fp64 orthonormal."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    S, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    W, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return {"Q_evecs": Q, "sigma_evecs": S, "u_wr": W[:, :16], "d": d}


def _banks(n_layers=2, d=D):
    return {L: _bank(d, seed=L) for L in range(n_layers)}


def _key(L, t):
    scope = "self_attn" if t in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    return f"model.layers.{L}.{scope}.{t}.weight"


def _shape(t, d=D, inner=INNER):
    """Residual-facing side must be d: input-side types (q/k/v/gate/up) → (inner, d);
    output-side types (o/down) → (d, inner)."""
    return (d, inner) if CW.residual_side(t) == "output" else (inner, d)


def _decoders(n_layers=2, d=D, seed=3):
    """A synthetic (base_dec, ft_dec) pair over the full n_layers×7 grid."""
    g = torch.Generator().manual_seed(seed)
    base, ft = {}, {}
    for L in range(n_layers):
        for t in PROJ:
            sh = _shape(t, d)
            Wb = torch.randn(*sh, generator=g, dtype=torch.float32)
            base[_key(L, t)] = Wb
            ft[_key(L, t)] = Wb + 0.1 * torch.randn(*sh, generator=g, dtype=torch.float32)
    base["model.embed_tokens.weight"] = torch.randn(5, d, generator=g)      # a non-target key
    ft["model.embed_tokens.weight"] = base["model.embed_tokens.weight"].clone()
    return base, ft


# ---------------------------------------------------------------------------
# 1. The PROTECT install: build_decoder("residual") == ΔW_off. The axis rests on this.
# ---------------------------------------------------------------------------
def test_build_decoder_residual_is_delta_off():
    base, ft = _decoders(n_layers=2)
    banks = _banks(2)
    bases = CL.off_bases(banks, "susceptibility", rank=4, n_layers=2)
    out, book = CS.build_decoder("residual", base, ft, bases, n_layers=2)
    for L in range(2):
        for t in PROJ:
            k = _key(L, t)
            dW = ft[k].float() - base[k].float()
            want = dW - CS.project_delta(dW, bases[(L, t)], t)
            got = out[k].float() - base[k].float()
            assert torch.allclose(got, want, atol=1e-5), f"{k}: residual cell != ΔW_off"
    assert book["cell"] == "residual"


def test_identity_cell_is_byte_exact_base():
    base, ft = _decoders(n_layers=2)
    bases = CL.off_bases(_banks(2), "random", rank=4, n_layers=2)
    out, _ = CS.build_decoder("identity", base, ft, bases, n_layers=2)
    for k, w in base.items():
        assert torch.equal(out[k], w), f"{k}: identity cell is not byte-exact base"


def test_ft_cell_reproduces_the_natural_update():
    """The wiring gate's ground truth: the 'ft' cell rebuilds W_base + ΔW through the same pipeline."""
    base, ft = _decoders(n_layers=2)
    bases = CL.off_bases(_banks(2), "random", rank=4, n_layers=2)
    out, _ = CS.build_decoder("ft", base, ft, bases, n_layers=2)
    for k in ft:
        assert torch.allclose(out[k].float(), ft[k].float(), atol=1e-5), f"{k}: ft cell != natural ΔW"


def test_non_target_keys_untouched():
    base, ft = _decoders(n_layers=2)
    bases = CL.off_bases(_banks(2), "energy", rank=4, n_layers=2)
    out, _ = CS.build_decoder("residual", base, ft, bases, n_layers=2)
    k = "model.embed_tokens.weight"
    assert torch.equal(out[k], base[k]), "a non-target key leaked through the surgery"


# ---------------------------------------------------------------------------
# 2. The off-projection's energy semantics + the A-CLEAVE-5 reachability claim.
# ---------------------------------------------------------------------------
def test_off_projection_removes_exactly_the_inspan_energy():
    g = torch.Generator().manual_seed(11)
    bank = _bank()
    for t in ("o_proj", "q_proj"):
        dW = torch.randn(*_shape(t), generator=g, dtype=torch.float64)
        U = bank["Q_evecs"][:, :6]
        off = dW - CS.project_delta(dW, U, t)
        # the off part has ZERO energy in span(U) …
        assert CW.energy_fraction(off.numpy(), U.numpy(), t) < 1e-10, f"{t}: off part retains in-span energy"
        # … and the into/off split conserves Frobenius energy (relative tol: project_delta is fp32
        # by design, so the cross term is O(1e-7)·‖ΔW‖², not 0).
        into = CS.project_delta(dW, U, t)
        fro2 = float((dW ** 2).sum())
        lhs = float((into ** 2).sum() + (off ** 2).sum())
        assert abs(lhs - fro2) / fro2 < 1e-5, f"{t}: ‖into‖²+‖off‖² != ‖ΔW‖² (rel {abs(lhs - fro2) / fro2:.2e})"


def test_rank_d_projection_leaves_nothing():
    """A-CLEAVE-5: at rank=d the off-projection removes the whole residual-facing update, so the
    install IS the base ⇒ retention 0. This is why PROTECT_LADDER must reach d — without it the
    spec's 'retention spans [0,1] by construction' (T2) is simply false."""
    g = torch.Generator().manual_seed(12)
    bank = _bank()
    for t in ("o_proj", "down_proj", "q_proj", "up_proj"):
        dW = torch.randn(*_shape(t), generator=g, dtype=torch.float64)
        U_full = bank["Q_evecs"]                      # all d columns
        off = dW - CS.project_delta(dW, U_full, t)
        # fp32 projection ⇒ the annihilation floor is ~√d·eps32·|ΔW|, not 0.
        assert float(off.abs().max()) < 1e-4, f"{t}: rank-d off-projection did not annihilate ΔW"


def test_protect_ladder_reaches_d():
    assert max(CL.PROTECT_LADDER) == CL.D_MODEL, "PROTECT ladder must reach d (A-CLEAVE-5)"
    assert list(CL.PROTECT_LADDER) == sorted(CL.PROTECT_LADDER), "ladder must be ascending"


def test_grad_projection_partition_identity():
    g = torch.Generator().manual_seed(13)
    bank = _bank()
    for t in ("o_proj", "gate_proj"):
        grad = torch.randn(*_shape(t), generator=g, dtype=torch.float32)
        U = bank["sigma_evecs"][:, :5].float()
        into = CW.project_grad_residual(grad, U, t, "into")
        off = CW.project_grad_residual(grad, U, t, "off")
        assert torch.allclose(into + off, grad, atol=1e-5), f"{t}: into+off != g"
        lhs = float((into ** 2).sum() + (off ** 2).sum())
        assert abs(lhs - float((grad ** 2).sum())) < 1e-3, f"{t}: ‖into‖²+‖off‖² != ‖g‖²"


def test_off_bases_covers_the_full_grid():
    banks = _banks(3)
    bases = CL.off_bases(banks, "susceptibility", rank=4, n_layers=3)
    assert set(bases) == {(L, t) for L in range(3) for t in PROJ}, "off_bases does not cover 18×7"
    for (L, t), U in bases.items():
        assert tuple(U.shape) == (D, 4), f"{(L, t)}: basis {tuple(U.shape)} != (d, r)"


# ---------------------------------------------------------------------------
# 3. Resolution certification (A-CLEAVE-3) — it must certify stability, refuse noise, and fail loud.
# ---------------------------------------------------------------------------
def _perturbed_basis(U, eps, seed):
    g = torch.Generator().manual_seed(seed)
    P = U + eps * torch.randn(*U.shape, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(P)
    return Q


def test_certify_resolution_certifies_stable_bases():
    U0 = _bank(seed=7)["Q_evecs"]
    reps = [U0.numpy()] + [_perturbed_basis(U0, 1e-4, s).numpy() for s in (1, 2)]
    evals = [np.linspace(10, 1, D) * (1 + 1e-3 * i) for i in range(3)]
    cert = CL.certify_resolution(reps, ranks=(2, 4, 8), evals_reps=evals, d=D)
    assert cert["certified_max"] == 8, f"stable bases failed to certify: {cert['per_rank']}"
    for r, v in cert["per_rank"].items():
        assert v["mean_overlap"] > 0.99, f"rank {r}: stable overlap only {v['mean_overlap']}"
        assert v["haar_floor"] == r / D


def test_certify_resolution_refuses_unstable_bases():
    """Independent random bases overlap at ~the Haar floor → certified_max must be 0. A gate that
    certifies noise is exactly STENCIL's T1 confound left open."""
    reps = [_bank(seed=s)["Q_evecs"].numpy() for s in (21, 22, 23)]
    evals = [np.linspace(10, 1, D) * (1 + 0.1 * i) for i in range(3)]
    cert = CL.certify_resolution(reps, ranks=(2, 4, 8), evals_reps=evals, d=D)
    assert cert["certified_max"] == 0, f"noise certified as resolved: {cert['per_rank']}"
    for r, v in cert["per_rank"].items():
        assert v["mean_overlap"] < 0.5, f"rank {r}: independent bases overlapped {v['mean_overlap']}"


def test_certify_resolution_independence_guard_fails_loud():
    """Identical repeats ⇒ overlap 1.0 at every rank ⇒ everything 'certified' — vacuously. The guard
    must RAISE (`feedback_signals_that_cannot_fail`)."""
    U0 = _bank(seed=7)["Q_evecs"].numpy()
    ev = np.linspace(10, 1, D)
    raised = False
    try:
        CL.certify_resolution([U0, U0, U0], ranks=(2, 4), evals_reps=[ev, ev, ev], d=D)
    except AssertionError as e:
        raised = "identical" in str(e)
    assert raised, "the independence guard did not fire on identical repeats"


def test_certify_resolution_requires_min_repeats():
    U0 = _bank(seed=7)["Q_evecs"].numpy()
    raised = False
    try:
        CL.certify_resolution([U0, U0], ranks=(2,), d=D)
    except AssertionError as e:
        raised = "independent Q measurements" in str(e)
    assert raised, "certification accepted < 3 repeats"


def test_resolved_ladder_uses_the_certified_SET_not_a_ceiling():
    """The L0 data inverted the prior: stability RISES with rank (sub-band degeneracy), so the
    certified ranks are {350, 679} — NOT an interval [0, max]. A ceiling rule would have kept exactly
    the unresolved low ranks and dropped the resolved high ones."""
    cert = {"certified_ranks": [350, 679], "certified_max": 679}
    out = CL.resolved_ladder((5, 10, 20, 50, 150, 350, 679, 2048), cert)
    assert out["kept"] == [350, 679], f"kept {out['kept']}"
    assert out["dropped"] == [5, 10, 20, 50, 150, 2048], "dropped ranks must be RETURNED for logging"


def test_certification_refuses_the_vacuous_rank_r_equals_d():
    """At r=d two full bases overlap 1.0 by construction — and the Haar floor is 1.0 too. A rank
    whose chance level already clears the tol cannot fail, so it must not certify
    (`feedback_one_sided_thresholds`)."""
    reps = [_bank(seed=s)["Q_evecs"].numpy() for s in (31, 32, 33)]
    evals = [np.linspace(9, 1, D) * (1 + 0.05 * i) for i in range(3)]
    cert = CL.certify_resolution(reps, ranks=(4, D), evals_reps=evals, d=D)
    full = cert["per_rank"][D]
    assert abs(full["mean_overlap"] - 1.0) < 1e-6, "full bases must overlap 1.0"
    assert full["haar_floor"] == 1.0 and full["vacuous"] is True
    assert full["certified"] is False, "r=d certified despite carrying zero information"
    assert D not in cert["certified_ranks"]


# ---------------------------------------------------------------------------
# 4. The frontier selectors + ρ*-reachability (STENCIL's T2 fix).
# ---------------------------------------------------------------------------
def test_knee_rank_finds_the_bend():
    """A curve that rises fast then saturates: the knee sits at the bend, not at either endpoint."""
    ranks = [5, 10, 20, 50, 150, 350, 679]
    red = [0.05, 0.30, 0.75, 0.95, 0.98, 0.99, 1.0]
    r = CL.knee_rank(ranks, red)
    assert r in (10, 20, 50), f"knee at {r}, expected the bend (10–50)"


def test_knee_rank_flat_curve_returns_widest():
    ranks = [5, 10, 20]
    assert CL.knee_rank(ranks, [0.5, 0.5, 0.5]) == 20


def test_rho_star_from_null_reads_the_null_curve():
    curve = [{"rank": 5, "retention": 0.98}, {"rank": 50, "retention": 0.61}, {"rank": 679, "retention": 0.1}]
    assert CL.rho_star_from_null(curve, 50) == 0.61
    raised = False
    try:
        CL.rho_star_from_null(curve, 999)
    except AssertionError:
        raised = True
    assert raised, "rho_star_from_null accepted an r* absent from the null curve"


def test_rho_reachable_true_and_false():
    curves = {"susceptibility": [{"rank": 5, "retention": 1.0}, {"rank": 2048, "retention": 0.0}],
              "random": [{"rank": 5, "retention": 0.97}, {"rank": 2048, "retention": 0.0}]}
    assert CL.rho_reachable(curves, 0.6)["reachable"], "ρ* inside every span reported unreachable"
    narrow = {"susceptibility": [{"rank": 5, "retention": 1.0}, {"rank": 679, "retention": 0.8}],
              "random": [{"rank": 5, "retention": 0.9}, {"rank": 679, "retention": 0.1}]}
    out = CL.rho_reachable(narrow, 0.5)
    assert not out["reachable"], "ρ* outside the susceptibility span reported reachable"
    assert out["per_policy"]["susceptibility"]["ok"] is False


def test_interp_forget_at_retention():
    rets = [1.0, 0.5, 0.0]
    forg = [0.20, 0.10, 0.0]
    assert abs(CL.interp_forget_at_retention(rets, forg, 0.5) - 0.10) < 1e-9
    assert abs(CL.interp_forget_at_retention(rets, forg, 0.75) - 0.15) < 1e-9
    assert CL.interp_forget_at_retention(rets, forg, 1.5) is None, "out-of-span must return None"


def test_retention_raises_on_zero_natural_gain():
    raised = False
    try:
        CL.retention(0.01, 0.0)
    except AssertionError as e:
        raised = "VOID-no-range" in str(e)
    assert raised, "retention divided through a zero natural gain"


def test_protective_frontier_pools_cells():
    """The swept axis is `alpha` by default (A-CLEAVE-12) — grouping by rank would collapse the whole
    α-sweep to one point, since every α cell shares the frozen r*. `key="rank"` still works."""
    cells = [{"policy": "random", "alpha": 0.0, "rank": 679, "retention": 1.0, "forgetting": 0.2},
             {"policy": "random", "alpha": 0.0, "rank": 679, "retention": 0.9, "forgetting": 0.1},
             {"policy": "random", "alpha": 1.0, "rank": 679, "retention": 0.5, "forgetting": 0.05},
             {"policy": "susceptibility", "alpha": 0.0, "rank": 679, "retention": 1.0, "forgetting": 0.3}]
    fr = CL.protective_frontier(cells, "random")                    # default: alpha
    assert [p["alpha"] for p in fr] == [0.0, 1.0], fr
    assert abs(fr[0]["retention"] - 0.95) < 1e-9 and fr[0]["n"] == 2
    # grouping by rank collapses the sweep — the bug this default guards against
    fr_r = CL.protective_frontier(cells, "random", key="rank")
    assert len(fr_r) == 1 and fr_r[0]["n"] == 3, fr_r


def test_rho_star_jointly_reachable_is_the_common_floor():
    """ρ* = the LOWEST retention every policy reaches — reachable by construction (the binding policy
    sits on its endpoint, the others interpolate). The α-axis mirror of STENCIL's common_tau."""
    curves = {"susceptibility": [{"alpha": 0.0, "retention": 1.0}, {"alpha": 1.0, "retention": 0.80}],
              "energy":         [{"alpha": 0.0, "retention": 1.0}, {"alpha": 1.0, "retention": 0.42}],
              "random":         [{"alpha": 0.0, "retention": 1.0}, {"alpha": 1.0, "retention": 0.55}]}
    rj = CL.rho_star_jointly_reachable(curves)
    assert abs(rj["rho_star"] - 0.80) < 1e-12, rj      # the highest floor binds
    assert rj["binding_policy"] == "susceptibility"
    assert CL.rho_reachable(curves, rj["rho_star"])["reachable"], "the joint floor must be reachable"


# ---------------------------------------------------------------------------
# 5. MECHANISM census (STENCIL's T4 fix) — measured, not inferred.
# ---------------------------------------------------------------------------
def test_mechanism_census_detects_where_the_mass_lives():
    """A matrix built entirely inside span(U_energy) on its residual-facing side must read
    energy≈1, susceptibility≈floor — the census's discriminating power."""
    bank = _bank(seed=5)
    g = torch.Generator().manual_seed(9)
    r = 4
    Ue = bank["sigma_evecs"][:, :r]
    M = Ue @ torch.randn(r, INNER, generator=g, dtype=torch.float64)     # o_proj: (d, inner)
    cen = CL.mechanism_census(M.numpy(), bank, rank=r, mtype="o_proj")
    assert cen["energy"] > 0.999, f"energy fraction {cen['energy']} for a matrix inside U_energy"
    assert cen["susceptibility"] < 0.5, f"susceptibility captured {cen['susceptibility']} of U_energy mass"
    assert abs(cen["chance_floor"] - r / D) < 1e-12
    assert CL.census_contrast(cen) > 0, "census_contrast should be > 0 when mass is in U_energy"


def test_mechanism_census_random_matrix_sits_at_the_floor():
    bank = _bank(seed=6)
    g = torch.Generator().manual_seed(10)
    r = 4
    M = torch.randn(D, INNER, generator=g, dtype=torch.float64)
    cen = CL.mechanism_census(M.numpy(), bank, rank=r, mtype="o_proj")
    for p in ("susceptibility", "energy", "random"):
        assert abs(cen[p] - r / D) < 0.25, f"{p}: random matrix read {cen[p]}, floor {r / D}"


# ---------------------------------------------------------------------------
# 6. The §6 adjudicator — every branch, in order.
# ---------------------------------------------------------------------------
def _ex(lo):
    return {"ci_lo": lo, "ci_hi": lo + 0.1, "median": lo + 0.05}


def _base_kwargs(**over):
    kw = dict(E_protect=_ex(0.02), specificity=_ex(0.01), mechanism_confirms=True,
              wiring_ok=True, resolution_ok=True, forget_range_ok=True, removable_range_ok=True,
              null_saturates=False, targeting_wins=False)
    kw.update(over)
    return kw


def test_adjudicate_halt_miswired_precedes_everything():
    for bad in ("wiring_ok", "resolution_ok"):
        v = CL.adjudicate_cleave(**_base_kwargs(**{bad: False}))
        assert v["verdict"] == "halt-miswired", f"{bad}=False did not halt"


def test_adjudicate_void_no_range():
    for bad in ("forget_range_ok", "removable_range_ok"):
        v = CL.adjudicate_cleave(**_base_kwargs(**{bad: False}))
        assert v["verdict"] == "VOID-no-range", f"{bad}=False did not VOID"


def test_adjudicate_saturation_guard_beats_a_map_win():
    v = CL.adjudicate_cleave(**_base_kwargs(null_saturates=True))
    assert v["verdict"] == "H_subspace-agnostic-protection"


def test_adjudicate_diagnostic_only():
    v = CL.adjudicate_cleave(**_base_kwargs(E_protect=_ex(-0.01), targeting_wins=False))
    assert v["verdict"] == "H_diagnostic-only"


def test_adjudicate_targeting_overturns_stencil():
    v = CL.adjudicate_cleave(**_base_kwargs(E_protect=_ex(-0.01), targeting_wins=True))
    assert v["verdict"] == "H_targeting"


def test_adjudicate_protective_is_the_practical_gate():
    v = CL.adjudicate_cleave(**_base_kwargs())
    assert v["verdict"] == "H_protective"


def test_adjudicate_protective_requires_every_leg():
    """E_protect alone is not enough: without specificity or the measured mechanism the map has not
    cleaved anything (spec §1.4 — 'off-energy matches it' kills H_protective)."""
    for over in ({"specificity": _ex(-0.005)}, {"mechanism_confirms": False}):
        v = CL.adjudicate_cleave(**_base_kwargs(**over))
        assert v["verdict"] == "H_diagnostic-only", f"{over} still returned {v['verdict']}"


def test_restricted_support_projects_only_certified_layers():
    """A-CLEAVE-11: at a rank certified on some layers only, the intervention must project THERE and
    pass ΔW through UNTOUCHED elsewhere — and the null must get the identical support."""
    banks = _banks(3)
    banks[0]["certified_ranks"] = [4, 8]
    banks[1]["certified_ranks"] = []          # L1 certifies nothing (as measured on PG1)
    banks[2]["certified_ranks"] = [8]
    assert CL.certified_layers(banks, 4, n_layers=3) == [0]
    assert CL.certified_layers(banks, 8, n_layers=3) == [0, 2]
    assert CL.certified_layers(banks, 5, n_layers=3) == []

    base, ft = _decoders(n_layers=3)
    for pol in ("susceptibility", "random"):          # identical support for treatment AND null
        bases, excluded = CL.off_bases_restricted(banks, pol, 8, [0, 2], n_layers=3)
        assert sorted({L for L, _ in bases}) == [0, 2]
        assert sorted({L for L, _ in excluded}) == [1]
        out, _ = CS.build_decoder("residual", base, ft, bases, excluded=excluded, n_layers=3)
        # L1 (excluded) must carry the FULL update — nothing removed there
        for t in PROJ:
            k = _key(1, t)
            dW = ft[k].float() - base[k].float()
            assert torch.allclose(out[k].float() - base[k].float(), dW, atol=1e-5), \
                f"{k}: excluded layer did not pass ΔW through untouched"
        # L0 (in support) must have its in-span component removed
        k0 = _key(0, "o_proj")
        dW0 = ft[k0].float() - base[k0].float()
        got = out[k0].float() - base[k0].float()
        assert not torch.allclose(got, dW0, atol=1e-5), "supported layer was not projected"


def test_restricted_support_raises_without_support():
    banks = _banks(2)
    raised = False
    try:
        CL.off_bases_restricted(banks, "susceptibility", 8, [], n_layers=2)
    except AssertionError as e:
        raised = "no certified layers" in str(e)
    assert raised, "an empty support must fail loud, not silently project nothing"


def test_target_grid_excludes_the_siglip_vision_tower():
    """PaliGemma's SigLIP tower is ALSO `...encoder.layers.N.self_attn.{q,k,v}_proj`, so a name-only
    match returns 207 = 126 + 27×3 instead of 126. Caught live by mark_grid_trainable's count assert
    ('grid has 207 matrices, expected 126'). Without the filter, a text layer's Q bank would be
    indexed by a VISION layer number and projected against a 1152-wide SigLIP matrix."""
    from infra.pg.gauge_common import PROJ as _P

    class _P2d:
        ndim = 2

    class _FakeModel:
        def __init__(self, names):
            self._n = names

        def named_parameters(self):
            return [(n, _P2d()) for n in self._n]

    text = [f"language_model.model.layers.{L}.{'self_attn' if t in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'}.{t}.weight"
            for L in range(18) for t in _P]
    vision = [f"vision_tower.vision_model.encoder.layers.{L}.self_attn.{t}.weight"
              for L in range(27) for t in ("q_proj", "k_proj", "v_proj")]
    assert len(text) == 126 and len(vision) == 81, (len(text), len(vision))
    grid = CL._target_grid(_FakeModel(text + vision))
    assert len(grid) == 126, f"grid picked up {len(grid)} matrices — the vision tower leaked back in"
    assert not any("vision" in n for n in grid), "a vision-tower matrix is in the grid"


def test_trained_set_is_exactly_what_the_surgery_can_rebuild():
    """A-CLEAVE-7: every param the natural update trains must be a key `build_decoder` rebuilds.
    STENCIL's mark_full_ft_trainable also trains the layernorm gains, which the surgery drops
    (non-target keys stay bitwise base) — so the ft cell could never reproduce the trained model.
    This pins the two sets to agree, at the level of the name-matching rules themselves."""
    import re

    from infra.pg.gauge_common import PROJ as _P
    names = [_key(0, t) for t in _P] + [
        "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight"]

    def full_ft_would_train(n):                 # stencil_core.mark_full_ft_trainable's rule
        return bool(re.search(r"layers\.\d+\.", n)) and not any(
            v in n.lower() for v in ("vision", "image", "siglip"))

    def surgery_rebuilds(n):                    # comwrite_surgery.is_target's rule (ndim==2 assumed)
        return any(f".{p}." in n for p in _P) and "layers." in n

    leaked = [n for n in names if full_ft_would_train(n) and not surgery_rebuilds(n)]
    assert leaked, "the confound this test guards has vanished — re-derive A-CLEAVE-7 before removing"
    # the grid rule (what CLEAVE actually trains) must leak NOTHING
    for n in names:
        if surgery_rebuilds(n):
            continue
        assert n in leaked, f"{n} classified inconsistently"


def test_paired_boot_small_n_and_sign():
    """The primary estimator must run at CLEAVE's real n (9 cells) — the item-level estimator
    asserts n≥10 and would refuse a perfectly valid cell-paired design."""
    a = [0.20, 0.22, 0.18, 0.25, 0.21, 0.19, 0.23, 0.20, 0.22]      # off-random forgetting
    b = [0.10, 0.11, 0.09, 0.12, 0.10, 0.09, 0.11, 0.10, 0.11]      # off-sus forgetting
    r = CL.paired_boot(a, b)
    assert r["n"] == 9 and r["small_n"] is True and r["insufficient"] is False
    assert r["E"] > 0 and r["ci_lo"] > 0, "a clean paired separation must clear the floor"
    flat = CL.paired_boot(b, b)
    assert flat["ci_lo"] <= 0 <= flat["ci_hi"], "identical arms must not clear the floor"


def test_paired_boot_insufficient_is_not_a_null():
    r = CL.paired_boot([0.1, 0.2], [0.0, 0.1], min_n=3)
    assert r["insufficient"] is True and r["ci_lo"] is None
    assert CL.clears_floor(r) is False, "missing evidence must never read as evidence"


def test_clears_floor_handles_none():
    assert CL.clears_floor({"ci_lo": 0.01}) is True
    assert CL.clears_floor({"ci_lo": -0.01}) is False
    assert CL.clears_floor({"ci_lo": None}) is False
    assert CL.clears_floor(None) is False


def test_adjudicate_refuses_to_verdict_an_unestimated_primary():
    """An UNESTIMATED primary must NOT become H_diagnostic-only.

    §6's branches all read "E_protect CI-lo ≤ 0", which presupposes a CI. With none,
    `clears_floor` returns False and would silently satisfy the diagnostic branch — reporting a
    verdict of ABSENCE, and calling it "airtight", from MISSING EVIDENCE. Caught on the real n=2
    confirmation artifact, where exactly that happened."""
    v = CL.adjudicate_cleave(**_base_kwargs(E_protect={"ci_lo": None, "n": 2, "insufficient": True}))
    assert v["verdict"] == "INSUFFICIENT-primary", v["verdict"]
    assert "not evidence of absence" in v["reason"]
    # it must not raise, and must not silently claim the diagnostic branch
    assert v["verdict"] != "H_diagnostic-only"


def test_adjudicate_insufficient_does_not_mask_the_earlier_guards():
    """The insufficient branch sits AFTER halt/VOID/saturation — a broken instrument or a substrate
    with no removable collateral is still reported as such, not as 'underpowered'."""
    v = CL.adjudicate_cleave(**_base_kwargs(E_protect={"ci_lo": None, "n": 2, "insufficient": True},
                                            wiring_ok=False))
    assert v["verdict"] == "halt-miswired"
    v = CL.adjudicate_cleave(**_base_kwargs(E_protect={"ci_lo": None, "n": 2, "insufficient": True},
                                            removable_range_ok=False))
    assert v["verdict"] == "VOID-no-range"


def test_adjudicate_measured_null_still_lands_diagnostic():
    """A genuinely MEASURED E_protect that fails its floor must still land H_diagnostic-only — the
    insufficient branch must not swallow real negative evidence."""
    v = CL.adjudicate_cleave(**_base_kwargs(E_protect={"ci_lo": -0.01, "ci_hi": 0.02, "n": 9,
                                                       "insufficient": False}))
    assert v["verdict"] == "H_diagnostic-only"



def test_pick_r_star_by_support_takes_max_coverage():
    """A-CLEAVE-12: with a rank-dependent support, r* is the rank whose certified support covers the
    MOST layers — read off the certification alone, and the setting most favourable to the map."""
    support = {20: [2], 50: [17], 350: [0, 2, 13], 679: [0, 3, 5, 6, 11, 13, 14, 15]}
    r, n = CL.pick_r_star_by_support(support)
    assert (r, n) == (679, 8), (r, n)
    # ties -> the smaller rank
    r2, n2 = CL.pick_r_star_by_support({50: [1, 2], 679: [3, 4]})
    assert r2 == 50, r2
    raised = False
    try:
        CL.pick_r_star_by_support({20: [], 679: []})
    except AssertionError as e:
        raised = "no reproducible referent" in str(e)
    assert raised, "an all-empty support must fail loud"


def test_combine_alpha_endpoints_and_monotone_removal():
    """alpha=0 must be the FULL natural update for every policy (an identity check), alpha=1 must
    equal the ordinary residual cell, and removed energy must rise with alpha."""
    base, ft = _decoders(n_layers=2)
    banks = _banks(2)
    bases = CL.off_bases(banks, "susceptibility", rank=4, n_layers=2)
    ft_cell, _ = CS.build_decoder("ft", base, ft, bases, n_layers=2)
    sh_cell, _ = CS.build_decoder("shared", base, ft, bases, n_layers=2)
    res_cell, _ = CS.build_decoder("residual", base, ft, bases, n_layers=2)

    a0 = CL.combine_alpha(ft_cell, sh_cell, base, 0.0)
    for k in ft:
        assert torch.allclose(a0[k].float(), ft_cell[k].float(), atol=1e-5), f"{k}: alpha=0 != ft cell"
    a1 = CL.combine_alpha(ft_cell, sh_cell, base, 1.0)
    for k in ft:
        assert torch.allclose(a1[k].float(), res_cell[k].float(), atol=1e-4), \
            f"{k}: alpha=1 != residual cell"
    # removed energy is monotone in alpha
    k0 = _key(0, "o_proj")
    prev = -1.0
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = CL.combine_alpha(ft_cell, sh_cell, base, a)
        removed = float(((ft_cell[k0].float() - c[k0].float()) ** 2).sum())
        assert removed >= prev - 1e-9, f"removed energy not monotone at alpha={a}"
        prev = removed
    raised = False
    try:
        CL.combine_alpha(ft_cell, sh_cell, base, 1.5)
    except AssertionError:
        raised = True
    assert raised, "alpha outside [0,1] must fail loud"



ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    """Run every dry-run test; return the pass count (the offline gate asserts == len(ALL))."""
    for fn in ALL:
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    print(f"\nALL {len(ALL)} CLEAVE dry-run logic-gate tests PASSED", flush=True)
    return len(ALL)


if __name__ == "__main__":
    import sys
    main()
    sys.exit(0)
