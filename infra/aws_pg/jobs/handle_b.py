"""HANDLE Arm B (GPU) — the witness set as an object (spec §3.1 Arm B, §6 R-B, §11 P9/P10).

SPECTER Arm-H banked FOUR adversarial "witnesses" in `specter_h/out/witness_directions.pt`:
  coherent@0.1, coherent@0.2  — each a (d=2048,) vector at fixed norm eps_abs = α·rms·√d,
  field@0.1,    field@0.2     — each a (T_field≈770, 2048) per-token field at fixed per-token energy,
each with a standard-battery damage row in `specter_h.remeasure[key]['battery_log1p_dppl']`.

Arm B studies that set as an OBJECT:
  (a) witness geometry (zero-forward): pairwise principal-angle spectrum + grassmann_diameter of the
      witness DIRECTION set vs a matched-rank Haar band (mean_min_angle both sides); d_eff via
      specter_core.d_eff_log_damage on the witness damage family.
  (b) overlap census (zero-forward): witness subspace vs {8 spikes, sink coords 738/674/604/2015,
      v₁, U_wr, Q-stiff space} — projection-energy overlap vs matched-rank Haar p95, BH in-family.
  (c) per-witness dose-response (GPU): 3 witnesses × 5 amplitudes, coherent injection at the L0
      residual (specter_h's site), safe_ppl_gate damage → the dose-response curves.
  (d) install-tracking (GPU): CARRY B2's κ-family + yes-install instrument — across 8 κ-probe
      directions spanning graded witness-subspace alignment, ρ_S(install success, alignment) with
      carry_core.shuffle_null_rho.
Feeds handle_core.adjudicate_rb(d_eff, mean_min_angle, haar_mean_min_angle, overlap_census,
install_rho, install_shuffle_p). No try/except hiding (CLAUDE.md).

===========================================================================================
§0.5 AMENDMENTS (pre-freeze, flagged for the lead — the bank has 4 witnesses, 2 of them matrices;
a "witness SET" needs a well-defined DIRECTION set for the angle spectrum / d_eff / census / the
"8 probe directions"; these are the constructions this driver commits to):

  B-1  WITNESS DIRECTION SET D (census + install-alignment reference). Each coherent witness is one
       unit direction ŵ = w/‖w‖ (2 dirs). Each (T,2048) field witness contributes its top-`m_field`
       right-singular directions (SVD of the field matrix; the activation-space axes the field most
       varies along), m_field=3 ⇒ 6 dirs. |D| = 8. Wframe = QR(D) (asserted full column rank).

  B-1b ANGLE SPECTRUM SET (fed to adjudicate_rb as mean_min_angle). Represent each of the 4 banked
       witnesses by its single top direction (coherent → its vector; field → its top right-singular
       vector) ⇒ 4 rank-1 frames. Pairwise principal angles among the 4 (6 pairs) via
       grassmann_diameter; matched-rank Haar band = haar_frames(d,1) sets of 4. Rationale: this asks
       "do the independent worst-case searches find NEARBY directions?" — the honest low-dimensionality
       question. Using the full |D|=8 set would inject the SVD-orthogonality of each field's own
       directions (artificial 90° pairs) and bias toward scatter; the 8-dir spectrum is banked
       separately for the lead but does NOT feed the verdict.

  B-2  d_eff DAMAGE FAMILY. specter_core.d_eff_log_damage expects RAW Δppl and applies log1p itself;
       the banked battery_log1p_dppl values (and safe_log1p_dppl dose values) are ALREADY log1p, so we
       feed raw = expm1(log1p) (exact round-trip: log1p(expm1(x))=x). Family = the 4 banked witness
       damages + the 15 dose-response damages (per the lead's proposed construction). FLAG: the dose
       amplitudes are same-direction rescalings (dose axis, not angular set-spread) and the 4 banked
       rows use SPECTER's battery while the 15 dose rows use HANDLE's safe_ppl_gate battery — an
       instrument mix. We therefore ALSO bank `d_eff_witnesses_only` (the 4 banked rows) so the lead
       can choose which feeds the verdict; PRIMARY (fed to the adjudicator) = the augmented family.

  B-3  THE 8 PROBE DIRECTIONS (install-tracking) = CARRY B2's κ-family. probe_j = carry_core.kappa_draw(
       ŵ_headline, κ_j, d) for κ_j ∈ {0, .25, .5, 1, 2, 4, 8}·√d ∪ {∞ (pure witness)} — 8 draws
       spanning measured witness-subspace alignment from ≈d⁻¹ (pure noise) to ≈1 (the witness). The
       probes are DISTINCT from D on purpose: if the probes were D itself their alignment would be ≈1
       for all and the correlation degenerate. "install success" per probe = the CARRY-B2 yes-install
       instrument (`carry_b2._yes_fraction` reused verbatim): yes-fraction on VQA with the probe
       injected coherently at matched energy eps0, minus the base rate (the 0.43-vs-0.22 signal).
       alignment = ‖Wframeᵀ·probe‖² ∈ [0,1] (energy in the witness subspace). ρ_S over the 8 with a
       label-shuffle null.

  B-4  CENSUS OVERLAP STATISTIC. The census elements have heterogeneous ranks (v₁:1, sink:4, spikes:8,
       U_wr:128, Q-stiff:679) so comwrite_core.subspace_overlap (equal-k) cannot cover them; we use its
       generalization proj_energy(E,W)=‖EᵀW‖²_F/rank(W) = the fraction of the WITNESS subspace lying in
       span(E) (equals subspace_overlap when rank(E)=rank(W)). Matched-rank Haar null replaces W by a
       Haar rank-r_W frame; p95 + one-sided p; emit_family BH over the 5 elements.
===========================================================================================

  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_b --type g6e.8xlarge,p5.4xlarge --no-weights \
      --s3-prefix results/handle_b --budget-min 240
"""
import argparse
import contextlib
import json
import math
import os
import pathlib

# ---- amendment constants (all flagged above) -------------------------------------------------
M_FIELD = 3                                   # B-1: top singular directions kept per field witness
SINK_COORDS = (738, 674, 604, 2015)           # canonical persistent-channel set (carry_gate0)
DOSE_AMPS = (0.25, 0.5, 1.0, 2.0, 4.0)        # B-2/(c): amplitude multipliers × eps0
KAPPA_UNITS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, None)   # B-3: 8 κ-probes (×√d; None = pure witness)
N_HAAR_ANGLE = 200                            # Haar draws for the angle band
N_HAAR_CENSUS = 500                           # Haar draws for each census null
HEADLINE = "coherent@0.1"                     # SPECTER rh1 headline witness (G=9032)
WITNESS_KEYS = ("coherent@0.1", "coherent@0.2", "field@0.1", "field@0.2")


def _unit(v):
    import torch
    v = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
    return v / v.norm().clamp_min(1e-12)


def _field_svd_dirs(field, m):
    """Top-m right-singular directions (d,) of a (T,d) field — the activation-space axes it varies
    along most (B-1). Returns a (d, m) fp64 tensor of unit columns."""
    import torch
    F = torch.as_tensor(field, dtype=torch.float64)
    _U, _S, Vh = torch.linalg.svd(F, full_matrices=False)     # Vh: (min(T,d), d)
    return Vh[:m].transpose(0, 1).contiguous()                # (d, m)


def _proj_energy(E, W):
    """B-4 census statistic: ‖EᵀW‖²_F / rank(W) — fraction of the witness subspace W (d,r) lying in
    span(E) (d,k). Both columns orthonormal. Generalizes comwrite_core.subspace_overlap to k≠r."""
    import torch
    E = torch.as_tensor(E, dtype=torch.float64)
    W = torch.as_tensor(W, dtype=torch.float64)
    return float((E.transpose(0, 1) @ W).pow(2).sum() / W.shape[1])


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    ap.add_argument("--m-field", type=int, default=M_FIELD)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs import _handle_shared as HS

    freeze = HS.assert_frozen("b")
    HS.install_pg()
    rep = {"object": "handle_b_v1", "freeze": freeze}
    tm = HS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    import numpy as np
    import torch

    from infra.pg import bispec_core as BC
    from infra.pg import carry_core as CC
    from infra.pg import handle_core as HC
    from infra.pg import specter_core as SC

    wd = pathlib.Path("/tmp/handle_b")
    wd.mkdir(parents=True, exist_ok=True)

    # ---- load the 4 witnesses + their banked damage rows -------------------------------------
    wit = torch.load(HS.fetch(f"{HS.SPECTER_H}/witness_directions.pt", wd / "witness_directions.pt"),
                     map_location="cpu", weights_only=False)
    for k in WITNESS_KEYS:
        assert k in wit, f"witness bank missing {k!r}"
    sh = json.loads(HS.fetch(f"{HS.SPECTER_H}/metrics.json", wd / "specter_h.json").read_text())
    assert sh.get("green"), "specter_h not green"
    remeas = sh["remeasure"]
    banked_log1p = {k: float(remeas[k]["battery_log1p_dppl"]) for k in WITNESS_KEYS}
    rep["banked_witness_log1p_dppl"] = banked_log1p

    d = int(wit[HEADLINE].reshape(-1).shape[0])
    assert d == 2048, f"unexpected witness dim {d}"

    # coherent unit vectors; per-witness banked injection norm (eps_abs); headline unit
    coh01 = torch.as_tensor(wit["coherent@0.1"], dtype=torch.float64).reshape(-1)
    coh02 = torch.as_tensor(wit["coherent@0.2"], dtype=torch.float64).reshape(-1)
    eps0 = float(coh01.norm())                                   # the coherent@0.1 energy convention
    w_head = _unit(coh01)                                        # headline direction (for κ-probes)
    fld01_dirs = _field_svd_dirs(wit["field@0.1"], a.m_field)    # (d, m)
    fld02_dirs = _field_svd_dirs(wit["field@0.2"], a.m_field)    # (d, m)

    # B-1 direction set D (|D| = 2 + 2*m); B-1b angle-spectrum set (4 top dirs)
    D_cols = [_unit(coh01), _unit(coh02)] + list(fld01_dirs.transpose(0, 1)) \
        + list(fld02_dirs.transpose(0, 1))
    D_mat = torch.stack([c.reshape(-1) for c in D_cols], dim=1)  # (d, 2+2m)
    sv = torch.linalg.svdvals(D_mat)
    assert float(sv.min()) > 1e-6, f"witness direction set is rank-deficient (σ_min={float(sv.min()):.2e})"
    Wframe = torch.linalg.qr(D_mat)[0]                           # (d, r_W)
    r_W = int(Wframe.shape[1])
    top_dirs = [_unit(coh01), _unit(coh02), fld01_dirs[:, 0], fld02_dirs[:, 0]]  # 4 rank-1 frames
    rep["direction_set"] = {"n_dirs": len(D_cols), "r_W": r_W, "m_field": a.m_field,
                            "eps0": eps0, "sigma_min": float(sv.min())}
    _w()

    # ---- (a) witness geometry (zero-forward): angle spectrum vs matched-rank Haar -------------
    with tm.stage("geometry"):
        frames4 = [c.reshape(d, 1) for c in top_dirs]
        gd = BC.grassmann_diameter(frames4)
        mean_min_angle = float(gd["mean_min_angle"])
        haar_mm = []
        for s in range(N_HAAR_ANGLE):
            hf = BC.haar_frames(d, 1, 4, seed=4200 + 4 * s)
            haar_mm.append(BC.grassmann_diameter(hf)["mean_min_angle"])
        haar_mm = np.asarray(haar_mm, float)
        haar_mean_min_angle = float(haar_mm.mean())
        # the |D|=8 spectrum (banked for the lead, NOT fed to the verdict — B-1b)
        gd8 = BC.grassmann_diameter([c.reshape(d, 1) for c in D_cols])
        rep["geometry"] = {
            "mean_min_angle": mean_min_angle, "mean_max_angle": float(gd["mean_max_angle"]),
            "diameter": float(gd["diameter"]), "n_pairs": int(gd["n_pairs"]),
            "haar_mean_min_angle": haar_mean_min_angle,
            "haar_mean_min_angle_p05": float(np.quantile(haar_mm, 0.05)),
            "haar_mean_min_angle_p50": float(np.quantile(haar_mm, 0.50)),
            "angle_ratio": mean_min_angle / max(haar_mean_min_angle, 1e-9),
            "full8_mean_min_angle": float(gd8["mean_min_angle"]),
            "full8_n_pairs": int(gd8["n_pairs"])}
        print(f"[handle-b] angles: mean_min={mean_min_angle:.4f} rad  haar={haar_mean_min_angle:.4f} "
              f"ratio={rep['geometry']['angle_ratio']:.3f}", flush=True)
    _w()

    # ---- (b) overlap census (zero-forward) ---------------------------------------------------
    with tm.stage("census"):
        qb = HS.load_q_bank(str(wd))
        evecs = torch.as_tensor(qb["evecs"], dtype=torch.float64)
        k_stiff = int(qb["k_stiff"])
        lib = torch.load(HS.fetch(f"{HS.SPECTER_O}/library_bank.pt", wd / "library_bank.pt"),
                         map_location="cpu", weights_only=False)
        spikes = HS.load_promoted_spikes(str(wd))                # 8 dicts with evec (d,)

        E_spk = torch.linalg.qr(torch.stack([_unit(s["evec"]) for s in spikes], dim=1))[0]
        E_sink = torch.zeros(d, len(SINK_COORDS), dtype=torch.float64)
        for j, c in enumerate(SINK_COORDS):
            E_sink[c, j] = 1.0
        E_v1 = _unit(lib["v1"]).reshape(d, 1)
        E_uwr = torch.linalg.qr(torch.as_tensor(lib["U_wr_o_proj"], dtype=torch.float64))[0]
        E_qs = evecs[:, :k_stiff].contiguous()                  # eigenvectors: already orthonormal
        elems = {"spikes8": E_spk, "sink_coords": E_sink, "v1": E_v1,
                 "U_wr": E_uwr, "Q_stiff": E_qs}

        haar_pool = [BC.haar_frames(d, r_W, 1, seed=9100 + i)[0] for i in range(N_HAAR_CENSUS)]
        overlap_census, pvals, labels = {}, [], []
        for name, E in elems.items():
            obs = _proj_energy(E, Wframe)
            null = np.array([_proj_energy(E, H) for H in haar_pool])
            p95 = float(np.quantile(null, 0.95))
            p = float(((null >= obs).sum() + 1) / (N_HAAR_CENSUS + 1))
            overlap_census[name] = {"overlap": obs, "haar_p95": p95, "p_value": p,
                                    "k_elem": int(E.shape[1]), "above_p95": bool(obs > p95)}
            pvals.append(p)
            labels.append(name)
            print(f"[handle-b] census {name}: overlap={obs:.4f} p95={p95:.4f} p={p:.4f}", flush=True)
        HS.emit_family(rep, "rb_overlap_census", pvals, labels, q=0.10)
        rep["overlap_census"] = overlap_census
    _w()

    # ==========================================================================================
    # GPU legs: dose-response (c) + install-tracking (d).
    # ==========================================================================================
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from infra.pg import inherit_pg1 as IP
    from infra.pg.projector_ablation import build_tasks

    with tm.stage("load_base"):
        anat = HS.load_roster(str(wd))
        model, proc = IP.load_pg1(sha=anat["base_sha"])
        layers = IP.decoder_layers(model)
        dev = "cuda"
        g0 = json.loads(HS.fetch(f"{HS.HANDLE_GATE0}/gate0.json", wd / "g0.json").read_text())
        assert g0.get("green"), "handle gate0 not green"
        base = HS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
        base_ppl = base["ppl"]
        assert abs(base_ppl - g0["ppl_anchor"]["ppl"]) <= 0.5, "base ppl drifted vs gate0"
        rep["base_ppl"] = base_ppl
    _w()

    def coherent_ctx(direction, amp):
        """Factory (carry_b2.dir_ctx shape): coherent pre-hook on layers[0] adding a fixed direction
        at norm `amp` to every residual position. Reusable by carry_b2._yes_fraction."""
        v = SC.project_coherent(direction.to(dev).float().reshape(-1), float(amp))

        @contextlib.contextmanager
        def ctx():
            def hook(_m, args, kwargs):
                from infra.pg.inherit_core import _split_hidden
                hs, rebuild = _split_hidden(args, kwargs)
                return rebuild((hs.float() + v.unsqueeze(0)).to(hs.dtype))
            h = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
            yield
            h.remove()
        return ctx

    # ---- (c) per-witness dose-response -------------------------------------------------------
    # 3 witnesses = the 3 highest banked-damage witness DIRECTIONS (field → top-SVD dir), injected
    # coherently at 5 shared absolute norms {amp × eps0} (B-2). Damage = safe_log1p_dppl.
    dose_source = {"coherent@0.1": _unit(coh01), "coherent@0.2": _unit(coh02),
                   "field@0.1": fld01_dirs[:, 0], "field@0.2": fld02_dirs[:, 0]}
    dose_keys = sorted(WITNESS_KEYS, key=lambda k: banked_log1p[k], reverse=True)[:3]
    dose_damages_log1p = []
    with tm.stage("dose_response"):
        dose = {}
        for key in dose_keys:
            direction = dose_source[key]
            curve = []
            for amp in DOSE_AMPS:
                with coherent_ctx(direction, amp * eps0)():
                    pg = HS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
                dmg = HS.safe_log1p_dppl(pg["mean_nll"], base_ppl)
                curve.append({"amp": amp, "abs_norm": amp * eps0, "log1p_dppl": dmg,
                              "saturated_ppl": pg["saturated"]})
                dose_damages_log1p.append(dmg)
                print(f"[handle-b] dose {key} amp={amp}: dppl={dmg:+.4f}", flush=True)
            dose[key] = curve
            rep["dose_response"] = dose
            _w()
    _w()

    # ---- d_eff on the witness damage family (B-2) --------------------------------------------
    from infra.pg.atlas_core import d_eff_from_damages
    fam_log1p = [banked_log1p[k] for k in WITNESS_KEYS] + dose_damages_log1p
    # d_eff on the log1p family DIRECTLY: d_eff_log_damage(expm1(x)) == d_eff_from_damages(x), but
    # expm1 OVERFLOWS for adversarial witnesses whose log1p_dppl ~10^3 (Python float e^709 caps).
    # The log convention is the ATLAS-R12 one either way; this just skips the overflowing round-trip.
    d_eff_full = d_eff_from_damages(fam_log1p)
    d_eff_wit = d_eff_from_damages([banked_log1p[k] for k in WITNESS_KEYS])
    # B-2c (lead ratification): "d_eff of W" is the 4-witness family (the spec's estimand); the
    # augmented family conflates a same-direction dose axis with witness-set dimensionality. With
    # only 4 SPECTER witnesses d_eff ≤ 4 always, so the d_eff ≤ 10 clause is NON-BINDING and the
    # angle-vs-Haar clause carries H_low-dim-surface — disclosed, not papered over.
    d_eff = float(d_eff_wit["d_eff"])
    rep["d_eff_detail"] = {"augmented": d_eff_full, "witnesses_only": d_eff_wit,
                           "n_family": len(fam_log1p), "fed_to_verdict": "witnesses_only",
                           "deff_clause_binding": False,
                           "note": "|W|=4 ⇒ d_eff≤10 non-binding; angle clause carries the verdict"}
    print(f"[handle-b] d_eff(augmented n={len(fam_log1p)})={d_eff:.3f} "
          f"d_eff(witnesses n=4)={d_eff_wit['d_eff']:.3f}", flush=True)
    _w()

    # ---- (d) install-tracking: CARRY-B2 κ-family + yes-install instrument ---------------------
    with tm.stage("install_tracking"):
        from infra.aws_pg.jobs.carry_b2 import _yes_fraction     # reuse the exact B2 instrument
        tasks = {t.name: t for t in build_tasks()}
        hashes, n_stream = HS.load_handle_items(str(wd), "vqav2", "disc", n=a.n_items)
        vqa_rows = HS.rows_for_banked(tasks["vqav2"], hashes, n_stream)
        base_rate = _yes_fraction(model, proc, tasks["vqav2"], vqa_rows, dev, None)

        probes, alignments, installs = [], [], []
        for j, u in enumerate(KAPPA_UNITS):
            kappa = None if u is None else float(u) * math.sqrt(d)
            probe = CC.kappa_draw(w_head.float(), kappa, d, seed=8000 + j).double()
            # alignment = ‖P_W probe‖² ∈ [0,1]; proj_energy normalizes by the 2nd arg's rank, so
            # pass the rank-1 probe SECOND (Wframe first) to divide by 1, not by r_W.
            align = _proj_energy(Wframe, probe.reshape(d, 1))   # energy of the probe in span(W)
            rate = _yes_fraction(model, proc, tasks["vqav2"], vqa_rows, dev,
                                 coherent_ctx(probe, eps0))
            success = rate - base_rate
            alignments.append(align)
            installs.append(success)
            probes.append({"kappa_units": u, "expected_cos": (1.0 if u is None
                           else CC.expected_cos(kappa, d)), "alignment": align,
                           "yes_rate": rate, "install_success": success})
            print(f"[handle-b] probe κ={u}: align={align:.4f} yes={rate:.3f} "
                  f"install={success:+.3f}", flush=True)
            rep["install_tracking"] = {"base_rate": base_rate, "probes": probes}
            _w()
        null = CC.shuffle_null_rho(alignments, installs)
        install_rho = float(null["rho_obs"])
        install_shuffle_p = float(null["p_value"])
        rep["install_tracking"].update({"rho": install_rho, "shuffle_p": install_shuffle_p,
                                        "shuffle_p95": float(null["p95"])})
        print(f"[handle-b] install-tracking: rho={install_rho:+.3f} shuffle_p={install_shuffle_p:.3f}",
              flush=True)
    _w()

    # ---- R-B verdict --------------------------------------------------------------------------
    rep["rb"] = HC.adjudicate_rb(
        d_eff=d_eff, mean_min_angle=mean_min_angle, haar_mean_min_angle=haar_mean_min_angle,
        overlap_census=overlap_census, install_rho=install_rho, install_shuffle_p=install_shuffle_p)

    HS.assert_gpu_seconds(rep)
    print("HANDLE_B_OK", json.dumps({"verdict": rep["rb"]["verdict"], "d_eff": round(d_eff, 2),
                                     "mean_min_angle": round(mean_min_angle, 4),
                                     "install_tracks": rep["rb"]["install_tracking"]["tracks"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
