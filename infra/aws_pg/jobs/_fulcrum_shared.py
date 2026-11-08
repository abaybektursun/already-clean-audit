"""FULCRUM shared job glue — the causal θ-sweep engine (Leg A), built ENTIRELY on QUORUM/CLEAVE
surfaces (no re-banking, no new install path).

The intervention family ΔW(θ,P) = ΔW − θ·P(ΔW), P = the r=679 subspace of policy P on the 8
R1-certified layers, is the whole-decoder `combine_alpha(ft_cell, shared_cell, base, θ)` (θ ≡ CLEAVE's
α): the `ft` cell is W_base+ΔW, the `shared` cell is W_base+P(ΔW), so
combine_alpha = W_base + ΔW − θ·P(ΔW). For each θ we install it and read three things off the SAME
swap-shell QUORUM uses:
  * gain(θ)               — the capability's own wired-task gain (`_quorum_shared.gain_of`);
  * n*_energy(θ)          — a `rank_sweep(policy="energy")` over all 18 layers (R1-exempt) → n*;
  * footprint(θ)          — the energy-weighted curvature/energy census over the certified support
                            (`fulcrum_core.footprint_census`), to verify the intervention MOVED it.

The curvature intervention is restricted to r=679 / CERT8 (the one R1-safe curvature subspace, overlap
0.974); the n* readout is on the energy basis at all 18 layers (R1-exempt). The matched null is a Haar
subspace at the same rank+support; the shuffle control destroys the ordering; the energy-projection arm
is exploratory (confounded by readout-basis overlap, disclosed).

Heavy imports inside functions so the module py_compiles on the control plane. No try/except (CLAUDE.md).
"""
from __future__ import annotations

# The whole QUORUM bank/task/swap-shell surface, verbatim (no re-derivation).
from infra.aws_pg.jobs._quorum_shared import (  # noqa: F401
    N_ITEMS,
    N_LAYERS,
    N_PPL_BATTERY,
    S3_BUCKET,
    SEED,
    assert_grid,
    build_task_map,
    decoder_sd,
    drop_ft_cache,
    fetch_s3,
    ft_decoder_sd,
    gain_of,
    hash_split,
    install_full_delta,
    install_pg,
    lm_bundle,
    load_capability_bank,
    load_model,
    make_installer,
    rank_sweep,
    reset_model,
    score,
    snapshot_base,
    task_rows,
)

# The retention-denominator screen (a capability enters only if its full ΔW learns) — QUORUM's.
from infra.pg.quorum_core import MIN_GAIN_FULL, RANK_LADDER
from infra.pg.fulcrum_core import CERT8, CURV_RANK, THETA_LADDER


def verify_cert_support(banks, rank=CURV_RANK, expect=CERT8, n_layers=N_LAYERS):
    """Re-verify the R1-safe curvature support against the banks at job time (baseline discipline: the
    frozen CERT8 is confirmed from the artifact, never trusted from prose). Returns the measured list
    and asserts it equals `expect`."""
    from infra.pg import cleave_core as CL
    measured = CL.certified_layers(banks, int(rank), n_layers=n_layers)
    assert list(measured) == list(expect), \
        f"certified layers at r={rank} are {measured}, spec froze {list(expect)} — banks changed?"
    return measured


def _shared_ft_cells(base_lm_sd, ft_lm_sd, banks, policy, cert_layers, rank, seed, device):
    """The `ft` cell (policy-independent) and the `shared` cell (W_base + P(ΔW) at rank/CERT8) for one
    projection — built ONCE, then combine_alpha traces θ by pure arithmetic (CLEAVE's alpha_sweep)."""
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_surgery as SURG
    bases, excluded = CL.off_bases_restricted(banks, policy, int(rank), cert_layers, seed=seed,
                                              n_layers=N_LAYERS, device=device)
    ft_cell, _ = SURG.build_decoder("ft", base_lm_sd, ft_lm_sd, bases, excluded=excluded, n_layers=N_LAYERS)
    shared_cell, book = SURG.build_decoder("shared", base_lm_sd, ft_lm_sd, bases, excluded=excluded,
                                           n_layers=N_LAYERS)
    captured = (book["total_captured_fro2"] / book["total_fro2"]) if book["total_fro2"] > 0 else 0.0
    return ft_cell, shared_cell, captured


def theta_sweep(base_lm_sd, ft_lm_sd, banks, *, policy, cert_layers, install, read, reset,
                gain_full, seed, device, curv_rank=CURV_RANK, thetas=THETA_LADDER,
                energy_ladder=RANK_LADDER, rho_break, min_gain=None, log=print):
    """Walk θ over one projection's path; at each θ install ΔW(θ,P) and read gain, n*_energy, footprint.

    Returns per-θ [{policy, theta, gain, gain_retention, n_star, censored, curvature_fraction,
    energy_fraction, chance_floor, captured_energy_fraction, energy_gain_full}].

    `gain_full` = gain(θ=0) (the full ΔW's wired-task gain) — the gain-retention denominator (spec §2).
    `rho_break` = the frozen n*-crossing threshold (QUORUM's 0.9460). n*_energy(θ) is the
    `necessary_rank` of an ENERGY rank-sweep of ΔW(θ,P) — its own retention denominator is gain(θ)
    (gain_off_energy(r)/gain(θ)), so it measures how energy-removable ΔW(θ,P) is. A θ whose gain(θ) has
    fallen below `min_gain` yields no reliable n* (a ratio to ~0): n_star=None, recorded and reported."""
    import torch

    from infra.pg import cleave_core as CL
    from infra.pg import fulcrum_core as FC
    from infra.pg import quorum_core as QC
    min_gain = MIN_GAIN_FULL if min_gain is None else min_gain

    # Build the two GPU cells, materialize every ΔW(θ,P) onto CPU, then FREE the cells + ft — so the
    # per-θ measurement (install ftθ + energy rank-sweep) holds only base+model+ftθ+sweep-cell on the
    # GPU (~26GB peak on an L40S), not base+ft+ft_cell+shared_cell+ftθ+sweep-cell (~42GB, OOM-risky).
    ft_cell, shared_cell, captured = _shared_ft_cells(base_lm_sd, ft_lm_sd, banks, policy,
                                                      cert_layers, curv_rank, seed, device)
    ft_thetas = {}
    for th in thetas:
        ftθ = CL.combine_alpha(ft_cell, shared_cell, base_lm_sd, float(th))
        ft_thetas[float(th)] = {k: v.detach().to("cpu") for k, v in ftθ.items()}
        del ftθ
    del ft_cell, shared_cell
    torch.cuda.empty_cache()

    rows = []
    for th in thetas:
        reset()
        ftθ = {k: v.to(device) for k, v in ft_thetas[float(th)].items()}  # W_base + ΔW(θ,P)
        install(ftθ, f"{policy}/th{th}")
        gθ = read(f"{policy}/th{th}")["gain"]
        # footprint of ΔW(θ,P) over the certified support (verify the intervention moved it)
        fp = FC.footprint_census(ftθ, base_lm_sd, banks, rank=curv_rank, layers=cert_layers)
        row = {"policy": policy, "theta": float(th), "gain": float(gθ),
               "gain_retention": float(gθ / gain_full) if abs(gain_full) > 1e-9 else 0.0,
               "curvature_fraction": fp["curvature"], "energy_fraction": fp["energy"],
               "chance_floor": fp["chance_floor"], "captured_energy_fraction": captured}
        if abs(gθ) < min_gain:
            row["n_star"] = None
            row["censored"] = None
            row["nstar_note"] = f"gain(θ)={gθ:+.4f} < min_gain {min_gain} — n* undefined (ratio to ~0)"
            log(f"[theta] {policy:>22s} θ={th:<4.2f} gain={gθ:+.4f} ret={row['gain_retention']:+.3f} "
                f"Qfrac={fp['curvature']:.3f} Efrac={fp['energy']:.3f} n*=SKIP(low gain)")
        else:
            sweep = rank_sweep(base_lm_sd, ftθ, banks, policy="energy", ladder=energy_ladder,
                               install=install, read=read, reset=reset, gain_full=gθ, seed=seed,
                               device=device, log=lambda *a, **k: None)
            ns = QC.necessary_rank([s["retention"] for s in sweep], [s["rank"] for s in sweep],
                                   rho_break=rho_break)
            row["n_star"] = ns["n_star"]
            row["censored"] = ns["censored"]
            row["energy_sweep"] = sweep
            log(f"[theta] {policy:>22s} θ={th:<4.2f} gain={gθ:+.4f} ret={row['gain_retention']:+.3f} "
                f"Qfrac={fp['curvature']:.3f} Efrac={fp['energy']:.3f} "
                f"n*={ns['n_star']:.1f}{'(cens)' if ns['censored'] else ''}")
        rows.append(row)
        del ftθ
        ft_thetas[float(th)] = None                # free the CPU staging copy as we go
        torch.cuda.empty_cache()
    return rows


def control_e_curv(base_lm_sd, ft_pos_sd, banks, *, cert_layers, install, read, reset, gain_full,
                   device, rho_break, seeds=(SEED, SEED + 1000), log=print):
    """E_curv of a SYNTHETIC control capability at its OWN jointly-reachable g*, PLUS its own
    random-vs-random null band (the noise floor against which E_curv is judged — NOT an n0-scaled
    threshold: the matched-gain reachability confound compresses E_curv far below n0, so the correct
    reference is random-vs-random). Runs the curv θ-sweep + one θ-sweep per random `seed`.

    Returns {e_curv (curv vs median-random @ g*), null_band (|randomA−randomB| @ g*), g_star, n0,
    n_curv, n_rands, curv_rows, random_rows}. The floor control (energy-only) must return |e_curv| within
    its null band; the ceiling (curvature-only) must return e_curv BELOW −null_band (spec §5)."""
    import numpy as np

    from infra.pg import fulcrum_core as FC
    curv = theta_sweep(base_lm_sd, ft_pos_sd, banks, policy="susceptibility", cert_layers=cert_layers,
                       install=install, read=read, reset=reset, gain_full=gain_full, seed=seeds[0],
                       device=device, rho_break=rho_break, log=log)
    rands = [theta_sweep(base_lm_sd, ft_pos_sd, banks, policy="random", cert_layers=cert_layers,
                         install=install, read=read, reset=reset, gain_full=gain_full, seed=s,
                         device=device, rho_break=rho_break, log=log) for s in seeds]
    cv = paths_to_curves({"susceptibility": curv, **{f"r{i}": r for i, r in enumerate(rands)}})
    gc, nc = cv["susceptibility"]["gain"], cv["susceptibility"]["nstar"]
    assert gc, "control curv θ-path produced no reliable n* (all θ below min_gain)"
    rand_keys = [f"r{i}" for i in range(len(rands)) if cv[f"r{i}"]["gain"]]
    gbt = {"susceptibility": gc, **{k: cv[k]["gain"] for k in rand_keys}}
    gs = FC.g_star_jointly_reachable({"c": gbt}, projections=list(gbt))["g_star"]
    n_curv = FC.interp_nstar_at_gstar(nc, gc, gs)
    n_rands = [FC.interp_nstar_at_gstar(cv[k]["nstar"], cv[k]["gain"], gs) for k in rand_keys]
    n_rand_ok = [r["n_star"] for r in n_rands if r["reachable"]]
    reachable = n_curv["reachable"] and len(n_rand_ok) >= 1
    e_curv = (n_curv["n_star"] - float(np.median(n_rand_ok))) if reachable else None
    null_band = float(abs(n_rand_ok[0] - n_rand_ok[1])) if len(n_rand_ok) >= 2 else None
    return {"e_curv": e_curv, "reachable": reachable, "null_band": null_band, "g_star": gs,
            "n0": nc.get(0.0), "n_curv": n_curv.get("n_star"), "n_rands": n_rand_ok,
            "curv_rows": curv, "random_rows": rands[0]}


def paths_to_curves(theta_rows_by_policy):
    """Reshape theta_sweep rows into {policy: {'gain': {θ:gain}, 'nstar': {θ:n*}}} for the fulcrum_core
    interpolators — dropping θ-points whose n* is undefined (low gain), so the interpolation span is
    over reliable points only."""
    out = {}
    for policy, rows in theta_rows_by_policy.items():
        gain = {r["theta"]: r["gain"] for r in rows if r.get("n_star") is not None}
        nstar = {r["theta"]: r["n_star"] for r in rows if r.get("n_star") is not None}
        out[policy] = {"gain": gain, "nstar": nstar}
    return out
