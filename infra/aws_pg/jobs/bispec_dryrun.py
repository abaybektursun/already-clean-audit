"""BISPEC dry-run (CPU, weight-free) — spec §12.1.

Nothing here touches the model. Every assert below is an invariant of the INSTRUMENT: if one
fails, the corresponding GPU cell would produce a number that means something other than what its
name says. The suite is the gate that must be green before a single GPU-hour is spent.

Sections:
  1. injector invariants     — band leak, permutation marginals, matched per-token energy,
                               envelope mean-square, coherence structure, position tracking
  2. level-set sampler       — hits the constraint; dispersion telemetry is non-degenerate
  3. profile / surface math  — planted spectra, planted band profiles, overlap coefficient
  4. cancellation-law fit    — recovers a planted (a, b, p); the bare power law does NOT
  5. adjudicator reachability— every §1.2 branch reachable AND mutually exclusive per arm
  6. saturation guards       — each guard actually fires on the case it was written for

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_dryrun --type r7i.2xlarge --no-weights \
      --s3-prefix results/bispec_dryrun --budget-min 30
"""
import argparse
import json
import pathlib

import torch
import torch.nn as nn

from infra.pg import bispec_core as BC

D = 64          # tiny d — every invariant here is dimension-free
K = 6


class _Layer(nn.Module):
    """A decoder-layer stand-in that records the hidden states its pre-hook handed it."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, hidden_states, **kw):
        self.seen.append(hidden_states.detach().clone())
        return hidden_states


def _run(layers, hook, xs):
    """Drive one prefill + len(xs)-1 decode steps through the hook, return the perturbed states."""
    layers[0].seen.clear()
    with BC.hook_at(layers, 0, hook):
        for x in xs:
            layers[0](x)
    return list(layers[0].seen)


def _planted_sigma(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    ev = torch.tensor([1.0 / (i + 1) ** 2 for i in range(d)], dtype=torch.float64)
    return (Q * ev) @ Q.transpose(0, 1), Q, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "bispec_dryrun_v1", "d": D, "checks": {}}

    def _w():
        (out / "dryrun.json").write_text(json.dumps(rep, indent=2, default=str))

    def check(name, ok, **extra):
        rep["checks"][name] = {"pass": bool(ok), **extra}
        _w()                                    # a failing assert must still leave the evidence
        print(f"[dryrun] {name}: {'PASS' if ok else 'FAIL'} {extra if extra else ''}", flush=True)
        assert ok, f"dry-run invariant FAILED: {name} {extra}"

    layers = nn.ModuleList([_Layer()])
    Sigma, Qtrue, ev = _planted_sigma(D)
    bands = BC.eig_bands(Sigma)
    hbar = torch.zeros(D, dtype=torch.float64)

    # ---------------------------------------------------------------- 1. injectors
    # 1a. position tracking across the prefill/decode boundary
    trk = BC._PositionTracker()
    p0 = trk.positions(5).tolist()
    p1 = trk.positions(1).tolist()
    p2 = trk.positions(1).tolist()
    check("position_tracker", p0 == [0, 1, 2, 3, 4] and p1 == [5] and p2 == [6],
          prefill=p0, decode=[p1, p2])

    # 1b. coherence structure: g=None → one draw for all tokens; g=2 → pairs; g=1 → all distinct
    T = 8
    x0 = torch.zeros(1, T, D)
    for g, want in ((None, 1), (2, 4), (1, 8)):
        h = BC.make_hook_gaussian_group(sigma=1.0, seed=11, g=g, d=D)
        y = _run(layers, h, [x0])[0][0]                 # (T,D) — x was zero, so y IS δ
        n_distinct = len(torch.unique(y, dim=0))
        check(f"coherence_groups_g{g}", n_distinct == want, distinct=n_distinct, want=want)

    # 1c. matched per-token energy across g (the matched-fidelity requirement).
    # Averaged over seeds: a fully-coherent cell's per-token energy is ONE χ²_D sample, so at
    # d=64 a single draw carries ~18% relative spread. The instrument matches the per-token
    # energy in EXPECTATION; the realized value is measured per cell on the real forward.
    energies = {}
    for g in (None, 4, 1):
        vals = []
        for s in range(12):
            h = BC.make_hook_gaussian_group(sigma=1.0, seed=11 + 1000 * s, g=g, d=D)
            y = _run(layers, h, [torch.zeros(1, 256, D)])[0][0]
            vals.append(float((y ** 2).sum(-1).mean()))
        energies[str(g)] = sum(vals) / len(vals)
    spread = max(energies.values()) / min(energies.values())
    check("matched_per_token_energy", spread < 1.15, energies=energies, spread=spread,
          note="matched in expectation (= D·σ² = 64); realized value measured per GPU cell")

    # 1d. coherence survives the decode boundary (the whole reason for the tracker)
    h = BC.make_hook_gaussian_group(sigma=1.0, seed=11, g=None, d=D)
    ys = _run(layers, h, [torch.zeros(1, 4, D), torch.zeros(1, 1, D)])
    check("coherence_across_decode",
          torch.allclose(ys[0][0, 0], ys[1][0, 0], atol=1e-6),
          delta=float((ys[0][0, 0] - ys[1][0, 0]).abs().max()))

    # 1e. quantization-error permutation preserves the error MARGINALS exactly
    x = torch.randn(1, 16, D)
    step = 0.8 * float(x.pow(2).mean().sqrt())
    e_true = BC.quant_error(x.float(), step)
    h = BC.make_hook_quant_permuted(step, seed=5, permute=True)
    y = _run(layers, h, [x])[0]
    e_perm = y - x
    same_marginals = torch.allclose(torch.sort(e_true.flatten())[0],
                                    torch.sort(e_perm.flatten())[0], atol=1e-5)
    moved = not torch.allclose(e_true, e_perm, atol=1e-6)
    check("quant_permutation_marginals", same_marginals and moved,
          same_marginals=same_marginals, actually_permuted=moved)

    # 1f. band noise: purity is exact, and the complement is untouched
    for b in range(len(bands["band_slices"])):
        U = BC.band_projector(bands, b)
        h = BC.make_hook_band_noise(U, sigma=1.0, seed=3, mode="dc")
        d_ = _run(layers, h, [torch.zeros(1, 8, D)])[0][0]
        pur = BC.band_purity(d_.double(), U)
        check(f"band_purity_b{b}", pur > 1.0 - BC.LEAK_TOL, purity=pur, k=U.shape[1])

    # 1f2. per-token INJECTED ENERGY must not depend on the band's rank. Without the unit-sphere
    # normalization the energy is sigma^2 * k, and at L0 the band ranks are [1,10,575,880,582] --
    # the surface would measure band rank, not where the energy sits.
    en = {}
    for b in range(len(bands["band_slices"])):
        U = BC.band_projector(bands, b)
        h = BC.make_hook_band_noise(U, sigma=1.0, seed=3, mode="dc", d_total=D)
        dlt = _run(layers, h, [torch.zeros(1, 6, D)])[0][0]
        en[b] = float((dlt ** 2).sum(-1).mean())
    spread_e = max(en.values()) / min(en.values())
    check("band_noise_energy_matched_across_ranks", spread_e < 1.02,
          per_band_energy={k: round(v, 3) for k, v in en.items()}, want=float(D), spread=spread_e,
          ranks=[hi - lo for lo, hi in bands["band_slices"]])

    # 1g. envelopes carry unit mean-square (so every ω has matched per-token energy)
    pos = torch.arange(512)
    for mode, freq in (("dc", None), ("sin", 1.0), ("sin", 4.0)):
        c = BC.envelope(mode if mode == "dc" else "sin", pos, freq=freq)
        ms = float((c ** 2).mean())
        check(f"envelope_ms_{mode}{freq or ''}", abs(ms - 1.0) < 0.02, mean_square=ms)

    # 1h. FULL deletion removes exactly the band component and nothing else
    U = BC.band_projector(bands, 0)
    x = torch.randn(1, 8, D)
    h = BC.make_hook_band_deletion(U, hbar, mode="full")
    y = _run(layers, h, [x])[0]
    resid = ((y.double() - x.double()) + ((x.double() @ U) @ U.T)).abs().max()
    check("deletion_full_exact", float(resid) < 1e-4, max_resid=float(resid))

    # 1h2. the ω_del contrast: dc/sin/white share the frozen mean-square and stay inside [0,1].
    # ("full" is c=1 and is deliberately NOT mean-square-matched to them — a removal fraction is
    # bounded by 1, so nothing modulated can reach the mean-square of full removal.)
    pos = torch.arange(512)
    ms = {}
    for mode, freq in (("dc", None), ("sin", 1.0), ("white", None)):
        c = BC.deletion_envelope(mode, pos, freq=freq, seed=17)
        ms[mode] = float((c ** 2).mean())
        assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0, f"{mode} envelope left [0,1]"
    check("delete_envelope_ms_matched",
          all(abs(v - BC.DEL_MS_TARGET) < 0.02 for v in ms.values()),
          realized_ms={k: round(v, 4) for k, v in ms.items()}, target=BC.DEL_MS_TARGET)
    check("delete_envelope_full_is_one",
          float(BC.deletion_envelope("full", pos).min()) == 1.0)
    # white must actually vary across positions; dc must not
    cw = BC.deletion_envelope("white", pos, seed=17)
    cd = BC.deletion_envelope("dc", pos)
    check("delete_envelope_white_varies_dc_does_not",
          float(cw.std()) > 0.05 and float(cd.std()) < 1e-12,
          white_std=float(cw.std()), dc_std=float(cd.std()))

    # 1i. per-token subspace removal really differs per token; shared really does not
    Ush = torch.linalg.qr(torch.randn(D, K, dtype=torch.float64))[0]
    pool = BC.pertoken_subspace_pool(D, K, 8, seed=9, device="cpu")
    check("pool_frames_orthonormal",
          float((torch.einsum("nik,nil->nkl", pool, pool)
                 - torch.eye(K).expand(8, K, K)).abs().max()) < 1e-5)
    for per_token, want_equal in ((False, True), (True, False)):
        h = BC.make_hook_subspace_removal(Ush, hbar, per_token, seed=9, d=D, k=K,
                                          pool=pool if per_token else None)
        x = torch.ones(1, 4, D)
        y = _run(layers, h, [x])[0][0]
        rows_equal = bool(torch.allclose(y[0], y[1], atol=1e-5))
        check(f"subspace_removal_pertoken={per_token}", rows_equal == want_equal,
              rows_equal=rows_equal)
    # the pool's DISCLOSED period: position p uses frame p mod n_pool, so rows p and p+n_pool match
    h = BC.make_hook_subspace_removal(None, hbar, True, 0, D, K, pool=pool)
    x = torch.ones(1, 12, D)
    y = _run(layers, h, [x])[0][0]
    check("pool_period_is_n_pool",
          torch.allclose(y[0], y[8], atol=1e-5) and not torch.allclose(y[0], y[1], atol=1e-5),
          n_pool=8, note="frames repeat with period n_pool; n_pool=512 >= every window this program uses")
    # the pooled removal must leave each token orthogonal to ITS OWN frame (hbar = 0 here)
    resid = max(float((y[t] @ pool[t % 8]).abs().max()) for t in range(12))
    check("pool_removal_is_a_projection", resid < 1e-4, max_resid=resid,
          note="y_t must be orthogonal to the frame used at position t")

    # 1j. gain-match reproduces the traversal's per-token norm change, carrying no direction
    x = torch.randn(1, 8, D)
    v = torch.randn(D, dtype=torch.float64)
    alpha = 2.0
    gains = BC.gains_for_alpha(x[0].double(), v, alpha)
    h = BC.make_hook_gain_match(lambda nrm: gains.float().unsqueeze(0))
    y = _run(layers, h, [x])[0]
    got = y[0].norm(dim=-1).double()
    want = (x[0].double() + alpha * v / v.norm()).norm(dim=-1)
    check("gain_match_norms", float((got - want).abs().max()) < 1e-3,
          max_err=float((got - want).abs().max()))

    # ---------------------------------------------------------------- 2. level set
    frames, realized = [], []
    for s in range(6):
        Q, E = BC.sample_level_set(Sigma, k=8, target_E=0.25, seed=100 + s)
        frames.append(Q)
        realized.append(E)
    check("level_set_hits_target", all(abs(e - 0.25) < 1e-3 for e in realized), realized=realized)
    disp = BC.grassmann_diameter(frames)
    ref = BC.grassmann_diameter(BC.haar_frames(D, 8, 6, seed=7))
    check("level_set_dispersion_nondegenerate", disp["diameter"] > 0 and ref["diameter"] > 0,
          constrained=disp["diameter"], unconstrained=ref["diameter"])
    check("principal_angles_self_zero",
          float(BC.principal_angles(frames[0], frames[0]).max()) < 1e-6)
    # angles must come back ASCENDING (the Gate-0a bug: a flipped `s` reversed them, so `th[-1]`
    # documented as the largest angle was returning the smallest)
    th = BC.principal_angles(BC.haar_frames(D, 8, 2, seed=31)[0], BC.haar_frames(D, 8, 2, seed=31)[1])
    check("principal_angles_ascending", bool((th[1:] >= th[:-1] - 1e-12).all()) and float(th[-1]) >= float(th[0]),
          first_deg=float(th[0]) * 57.2958, last_deg=float(th[-1]) * 57.2958)
    # the top-k eigenbasis must remove exactly the top-k cumulative energy
    Etop = BC.removed_energy_fraction(Qtrue[:, :8], Sigma)
    check("removed_energy_topk", abs(Etop - float(bands["cum"][7])) < 1e-8,
          got=Etop, want=float(bands["cum"][7]))
    check("E_max_is_topk_energy",
          abs(BC.max_removable_energy(Sigma, 8) - float(bands["cum"][7])) < 1e-10)

    # A8/conditioning: on a STEEP spectrum, a large positive tilt must still converge to the top-k
    # subspace. The one-shot matrix power collapses to rank 1 here (lambda2/lambda1 ** t underflows);
    # subspace iteration does not. Planted L0-like spectrum: one direction at 70% of the trace.
    Sig_steep, Qs, _ = _planted_sigma(D, seed=9)
    evs = torch.zeros(D, dtype=torch.float64)
    evs[0] = 0.70
    evs[1:] = 0.30 * torch.tensor([1.0 / (i + 1) ** 2 for i in range(D - 1)], dtype=torch.float64)
    evs[1:] = evs[1:] / evs[1:].sum() * 0.30
    Sig_steep = (Qs * evs) @ Qs.transpose(0, 1)
    eig_s = torch.linalg.eigh(Sig_steep)
    E_lo_r, E_hi_r, E_an = BC.reachable_energy_range(Sig_steep, 8, eig=eig_s)
    check("subspace_iteration_reaches_topk", E_hi_r > 0.98 * E_an,
          E_max_reachable=E_hi_r, E_max_analytic=E_an, E_min_reachable=E_lo_r,
          note="one-shot Sigma^t QR would stall at lambda1's share (0.70); subspace iteration must not")

    # A7: the two dispersion metrics disagree, and the disagreement is the point. Build a spectrum
    # with one dominant direction (like the real L0), constrain the ensemble to a high energy, and
    # show the CHORDAL diameter barely moves while the ENERGY alignment jumps.
    Sig1, _Q1, _e1 = _planted_sigma(D, seed=4)
    ev1 = torch.zeros(D, dtype=torch.float64)
    ev1[0] = 0.70
    ev1[1:] = 0.30 / (D - 1)
    Sig1 = (_Q1 * ev1) @ _Q1.transpose(0, 1)
    eig1 = torch.linalg.eigh(Sig1)
    tight = [BC.sample_level_set(Sig1, 8, 0.30, seed=300 + s, eig=eig1)[0] for s in range(5)]
    haar5 = BC.haar_frames(D, 8, 5, seed=41)
    ch_t, ch_h = BC.grassmann_diameter(tight), BC.grassmann_diameter(haar5)
    en_t, en_h = BC.energy_dispersion(tight, Sig1), BC.energy_dispersion(haar5, Sig1)
    chordal_ratio = ch_t["diameter"] / ch_h["diameter"]
    energy_ratio = en_t["mean_distance"] / en_h["mean_distance"]
    check("chordal_metric_saturates_energy_metric_does_not",
          chordal_ratio > 0.85 and energy_ratio < chordal_ratio,
          chordal_ratio=chordal_ratio, energy_ratio=energy_ratio,
          constrained_energy_alignment=en_t["mean_alignment"],
          haar_energy_alignment=en_h["mean_alignment"],
          note="B9's chordal statistic cannot report thinness in high ambient dim")

    # ---------------------------------------------------------------- 3. profile / surface math
    check("bands_cover_spectrum",
          bands["band_slices"][0][0] == 0 and bands["band_slices"][-1][1] == D
          and abs(sum(bands["band_energy"]) - 1.0) < 1e-8,
          slices=bands["band_slices"], energy=bands["band_energy"])
    # a matrix planted entirely inside band 2 must profile as band 2
    U2 = BC.band_projector(bands, 2)
    M = U2 @ torch.randn(U2.shape[1], 5, dtype=torch.float64)
    prof = BC.band_profile(M, bands)
    check("band_profile_planted", prof[2] > 0.999, profile=[round(p, 4) for p in prof])
    check("lazy_reference_is_band_energy",
          BC.lazy_training_reference(bands) == bands["band_energy"])
    # A Haar-random dW profiles onto the DIMENSION fractions, not the energy fractions. Without this
    # reference, "the write avoids the data's top band" is a restatement of "most dimensions are not
    # the top one" -- bands 2-4 hold 2037 of 2048 dims at the real L0.
    rnd_ref = BC.random_write_reference(bands)
    check("random_write_reference_is_dimension_fraction",
          abs(sum(rnd_ref) - 1.0) < 1e-9
          and all(abs(rnd_ref[b] - (bands["band_slices"][b][1] - bands["band_slices"][b][0]) / D) < 1e-12
                  for b in range(len(rnd_ref))),
          rnd_ref=[round(x, 4) for x in rnd_ref], ntk_ref=[round(x, 4) for x in bands["band_energy"]])
    gen = torch.Generator().manual_seed(21)
    dW_rand = torch.randn(D, 40, generator=gen, dtype=torch.float64)
    prof_rand = BC.band_profile(dW_rand, bands)
    check("random_dW_lands_on_random_reference",
          max(abs(prof_rand[b] - rnd_ref[b]) for b in range(len(rnd_ref))) < 0.06,
          profile=[round(x, 4) for x in prof_rand], reference=[round(x, 4) for x in rnd_ref])
    # ... and it must be SEPARABLE from the NTK reference (they are different curves)
    check("ntk_and_random_references_differ",
          max(abs(bands["band_energy"][b] - rnd_ref[b]) for b in range(len(rnd_ref))) > 0.2)
    # the structure adjudicator must be able to report BOTH branches
    iso = BC.adjudicate_write_structure({"m": list(rnd_ref)}, rnd_ref)
    stru = BC.adjudicate_write_structure({"m": [1.0] + [0.0] * (len(rnd_ref) - 1)}, rnd_ref)
    check("write_structure_branches_reachable",
          iso["verdict"] == "H_write-isotropic" and stru["verdict"] == "H_write-structured",
          got=(iso["verdict"], stru["verdict"]))
    S = torch.rand(5, 3).numpy()
    check("overlap_self_is_one", abs(BC.support_overlap(S, S) - 1.0) < 1e-9)
    check("form_alignment_self_is_one", abs(BC.form_alignment(Sigma, Sigma) - 1.0) < 1e-9)

    # n_eff of a uniform attention row over n keys is exactly n; of a one-hot row, exactly 1
    unif = torch.full((1, 1, 10), 0.1)
    onehot = torch.zeros(1, 1, 10)
    onehot[0, 0, 0] = 1.0
    check("n_eff_uniform", abs(float(BC.n_eff_from_attention(unif)[0]) - 10.0) < 1e-6)
    check("n_eff_onehot", abs(float(BC.n_eff_from_attention(onehot)[0]) - 1.0) < 1e-6)

    # ---------------------------------------------------------------- 4. cancellation-law fit
    g_vals = [1, 4, 16, 64, 256]
    a_t, b_t, p_t = 0.20, 0.60, 0.5
    gen = torch.Generator().manual_seed(3)
    damage = [[a_t + b_t * g ** (-p_t) + 0.002 * float(torch.randn(1, generator=gen))
               for _ in range(3)] for g in g_vals]
    fit = BC.fit_cancellation_law(g_vals, damage, n_boot=200)
    check("fit_recovers_planted",
          abs(fit["a"] - a_t) < 0.03 and abs(fit["p"] - p_t) < 0.08,
          a=fit["a"], b=fit["b"], p=fit["p"], p_ci=fit["p_ci95"])
    # A4's whole point: with a floor present, the bare power law is badly biased toward 0
    check("bare_powerlaw_is_biased", abs(fit["p_bare_powerlaw"] - p_t) > 0.15,
          p_bare=fit["p_bare_powerlaw"], p_true=p_t,
          note="a bare slope would be mis-read as 'flatter than -1/2 => structured routing'")
    rep["fit_demo"] = fit

    # ---------------------------------------------------------------- 5. adjudicator reachability
    # Arm K — all three branches reachable, mutually exclusive
    kb = {
        "H_coherence": BC.adjudicate_coherence(
            {"P1": {"gap_cilo": 0.9, "levels_cleared": 2}, "P3": {"gap_cilo": 0.8, "levels_cleared": 2}}, 0.1),
        "H_spectrum-suffices": BC.adjudicate_coherence(
            {"P1": {"gap_cilo": 0.0, "levels_cleared": 0}, "P3": {"gap_cilo": 0.01, "levels_cleared": 1}}, 0.1),
        "H_conditional": BC.adjudicate_coherence(
            {"P1": {"gap_cilo": 0.9, "levels_cleared": 2}, "P3": {"gap_cilo": 0.0, "levels_cleared": 0}}, 0.1),
    }
    check("armK_branches_reachable", all(v["verdict"] == k for k, v in kb.items()),
          got={k: v["verdict"] for k, v in kb.items()})
    # A9: P2 must NOT be able to swing the sequence-axis verdict.
    with_p2 = BC.adjudicate_coherence(
        {"P1": {"gap_cilo": 0.9, "levels_cleared": 2}, "P3": {"gap_cilo": 0.8, "levels_cleared": 2},
         "P2": {"gap_cilo": -9.0, "levels_cleared": 0}}, 0.1)
    check("armK_p2_excluded_from_sequence_axis",
          with_p2["verdict"] == "H_coherence" and with_p2["excluded_from_axis"] == ["P2"],
          verdict=with_p2["verdict"], excluded=with_p2["excluded_from_axis"])
    # A9: the alignment axis is two-sided and its THREE branches are reachable, including the branch
    # Gate-0b actually observed (aligned corruption strictly LESS damaging than its shuffled twin).
    al = {
        "aligned_more_damaging": BC.adjudicate_alignment(gap=0.5, gap_cilo=0.4, gap_cihi=0.6, delta_K=0.1),
        "aligned_LESS_damaging": BC.adjudicate_alignment(gap=-0.5, gap_cilo=-0.6, gap_cihi=-0.4, delta_K=0.1),
    }
    inert = BC.adjudicate_alignment(gap=0.0, gap_cilo=-0.05, gap_cihi=0.05, delta_K=0.1)
    check("alignment_axis_two_sided",
          all(v["verdict"] == "H_alignment-matters" and v["sign"] == k for k, v in al.items())
          and inert["verdict"] == "H_alignment-inert",
          got={k: (v["verdict"], v.get("sign")) for k, v in al.items()}, inert=inert["verdict"],
          note="the observed Gate-0b branch (permuted quant 4.6x worse) must be expressible")

    # Arm A — all four branches reachable
    base = dict(gain_frac_threshold=0.8, rand_asym_hi=0.05)
    ab = {
        "H_mean-shift": BC.adjudicate_asymmetry(gain_frac=0.9, orth_asym=0.01, kink_score=0, cubic_score=0,
                                                proxy_rho=0, proxy_rho_cilo=0, **base),
        "H_boundary": BC.adjudicate_asymmetry(gain_frac=0.1, orth_asym=0.5, kink_score=0.9, cubic_score=0.2,
                                              proxy_rho=0.8, proxy_rho_cilo=0.4, **base),
        "H_cubic": BC.adjudicate_asymmetry(gain_frac=0.1, orth_asym=0.5, kink_score=0.2, cubic_score=0.9,
                                           proxy_rho=0.0, proxy_rho_cilo=-0.1, **base),
        "H_direction-primitive": BC.adjudicate_asymmetry(gain_frac=0.1, orth_asym=0.5, kink_score=0.9,
                                                         cubic_score=0.2, proxy_rho=0.0,
                                                         proxy_rho_cilo=-0.1, **base),
    }
    check("armA_branches_reachable", all(v["verdict"] == k for k, v in ab.items()),
          got={k: v["verdict"] for k, v in ab.items()})

    # Arm V2 — both branches reachable
    check("armV_branches_reachable",
          BC.adjudicate_levelset(constrained_diameter=0.1, unconstrained_diameter=1.0)["verdict"] == "H_thin-ball"
          and BC.adjudicate_levelset(constrained_diameter=0.9, unconstrained_diameter=1.0)["verdict"] == "H_isotropy")

    # Arm W — all three branches reachable
    refc = [0.5, 0.3, 0.15, 0.04, 0.01]
    wb = {
        "H_ntk-aligned": BC.adjudicate_write({("l0", "mlp"): [0.51, 0.3, 0.14, 0.04, 0.01]}, refc),
        "H_offset": BC.adjudicate_write({("l0", "mlp"): [0.20, 0.50, 0.25, 0.04, 0.01],
                                         ("l1", "mlp"): [0.22, 0.48, 0.25, 0.04, 0.01]}, refc),
        "H_stratified": BC.adjudicate_write({("l0", "mlp"): [0.20, 0.50, 0.25, 0.04, 0.01],
                                             ("l0", "attn"): [0.80, 0.10, 0.05, 0.04, 0.01]}, refc),
    }
    check("armW_branches_reachable", all(v["verdict"] == k for k, v in wb.items()),
          got={k: v["verdict"] for k, v in wb.items()})

    # ---------------------------------------------------------------- 6. saturation guards fire
    dead = BC.saturation_guard(damage_treat=0.0, damage_null=0.0, floor=0.0, intact=0.54, name="both-dead")
    intactish = BC.saturation_guard(damage_treat=0.54, damage_null=0.539, floor=0.0, intact=0.54, name="both-intact")
    live = BC.saturation_guard(damage_treat=0.30, damage_null=0.50, floor=0.0, intact=0.54, name="live")
    check("saturation_guard_fires_on_dead_pair", dead["void"], reason=dead["reason"])
    check("saturation_guard_fires_on_intact_pair", intactish["void"], reason=intactish["reason"])
    check("saturation_guard_passes_live_cell", not live["void"], **{k: v for k, v in live.items() if k != "void"})
    # the REGRAIN R1 event, replayed: an em-span control that obliterated the model would have been
    # scored identically to one that faithfully reproduced B*. The guard must VOID it.
    regrain_r1 = BC.saturation_guard(damage_treat=0.0, damage_null=0.0, floor=0.0, intact=0.5408,
                                     name="regrain_r1_em_reproduces")
    check("guard_would_have_caught_regrain_R1", regrain_r1["void"], reason=regrain_r1["reason"])

    n_pass = sum(1 for c in rep["checks"].values() if c["pass"])
    rep["n_checks"] = len(rep["checks"])
    rep["n_pass"] = n_pass
    print("BISPEC_DRYRUN_OK", json.dumps({"n_checks": rep["n_checks"], "n_pass": n_pass}))
    # green LAST: a crash above must not leave a green artifact behind
    rep["green"] = n_pass == len(rep["checks"])
    _w()


if __name__ == "__main__":
    main()
