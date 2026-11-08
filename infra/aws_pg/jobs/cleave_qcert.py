"""CLEAVE Q-certification + banks (GPU) — spec §9.3 / §5, A-CLEAVE-3.

Measures the per-layer susceptibility Q **≥3 times independently** and certifies, per rank, whether
`U_sus` is the map or noise. This is STENCIL's T1 confound (≤8 stable spikes; 0–4 at most layers)
closed as a *measured gate* rather than an asserted caveat: map arms afterwards run ONLY inside the
certified-resolved rank range, and everything dropped is logged.

Each repeat i varies three independent sources of measurement noise:
  * the operator ITEMS   — a disjoint wikitext window draw (`skip = i · n_op_ppl`), a real data bootstrap;
  * the LANCZOS seed     — the Krylov start vector;
  * the Hutchinson TRACE seed — the sub-band bulk probes.
The Σ eigenbasis is harvested ONCE and shared across repeats: the gate therefore bounds **operator**
noise with the Σ basis held fixed. That scope is stated, not silently folded in — Σ-harvest noise is
out of scope and named as such in the report (`cert_scope`).

Q recipe per repeat is CARRY/SPECTER verbatim (dc/ppl operator; `stencil_banks.measure_q`):
Σ → `bispec_core.eig_bands` → `atlas_fit.subband_slices` → `specter_core.make_residual_injector` →
`lanczos(m=72, restart 144)` → `stable_spikes(A1, A2, ray_b)` → 16×8 Hutchinson sub-band traces →
`carry_core.materialize_q` (fidelity < 1e-8) → `stiff_split`.

  # A-CLEAVE-4 cheap-first pilot: ONE layer, full 3 repeats — measures per-layer cost + the curve shape
  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_qcert --args "--layers 0 --repeats 3 --validate-l0" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/cleave_qcert_pilot --budget-min 180
  # then the full sweep
  ./infra/aws_pg/pg run infra.aws_pg.jobs.cleave_qcert --args "--layers 0-17 --repeats 3" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/cleave_qcert --budget-min 1400
"""
import argparse
import hashlib
import json
import os
import pathlib

CONV_RESID_FRAC = 0.10
M_RESTART = 144
TOP_EVAL_HELDOUT = 16
N_SPIKE = 8
FIDELITY_TOL = 1e-8


def _parse_layers(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-17")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--validate-l0", action="store_true")
    ap.add_argument("--n-op-ppl", type=int, default=24)
    ap.add_argument("--n-op-held", type=int, default=12)
    ap.add_argument("--n-sigma-ppl", type=int, default=160)
    ap.add_argument("--n-sigma-vqa", type=int, default=200)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert a.repeats >= 3, "the certification needs ≥3 independent measurements (A-CLEAVE-3)"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _cleave_shared as CLS
    from infra.aws_pg.jobs import _ration_shared as RATS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    from infra.pg import carry_core as CC
    from infra.pg import cleave_core as CL
    from infra.pg import comwrite_core as CW
    from infra.pg import inherit_pg1 as IP
    from infra.pg import ration_core as RC
    from infra.pg import specter_core as SC

    CLS.install_pg()
    device, d = "cuda", IP.D
    path = "func"
    layers_want = _parse_layers(a.layers)
    rep = {"object": "cleave_qcert_v1", "layers": layers_want, "repeats": int(a.repeats),
           "gpu_seconds": {},
           "cert_scope": ("repeats vary (operator-item draw, lanczos seed, Hutchinson trace seed); "
                          "the Σ eigenbasis is harvested once and shared — this gate bounds OPERATOR "
                          "noise, not Σ-harvest noise"),
           "cert_tol": CL.CERT_TOL, "cert_ranks": list(CL.CERT_RANKS)}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = CLS.load_model("/tmp/cleave")
    rep["pinned_sha"], rep["base_ppl"] = sha, base_ppl
    _w()

    # ---- Σ harvest at ALL layers, ONCE (energy C_x + the Q-completion Σ) -------------------
    with tm.stage("sigma_harvest"):
        sig_windows = SPS.wikitext_windows(proc, a.n_sigma_ppl)
        vqa_task, _coco = IP.pg1_tasks()
        vqa_rows, _ = IP.load_items(vqa_task, a.n_sigma_vqa)
        sigma_state = RATS.accumulate_sigma_all_layers(model, proc, layers, sig_windows,
                                                       vqa_rows=vqa_rows, task=vqa_task, device=device)
    _w()

    def measure_q(L, rep_idx):
        """One INDEPENDENT measurement of Q at layer L (repeat `rep_idx`). The three noise sources
        the certification bounds are all keyed off rep_idx."""
        st = sigma_state[L]
        Sigma = (torch.as_tensor(st["gram"], dtype=torch.float64) / st["n"])
        bands = BC.eig_bands(Sigma)
        sb = AF.subband_slices(bands["band_slices"], bands["spectrum"])
        basis = torch.as_tensor(bands["basis"], dtype=torch.float64)
        basis32 = basis.to(device, torch.float32)
        rms = float((torch.diagonal(Sigma).sum() / d).sqrt())
        eps = 0.05 * rms

        # (1) an INDEPENDENT operator-item draw per repeat (a real data bootstrap, not a reseed)
        skip_a = rep_idx * (a.n_op_ppl + a.n_op_held)
        wins_a = SPS.wikitext_windows(proc, a.n_op_ppl, skip=skip_a)
        wins_b = SPS.wikitext_windows(proc, a.n_op_held, skip=skip_a + a.n_op_ppl)

        install, holder = SC.make_residual_injector(layers, L)
        A = [SPS.build_ppl_item(model, proc, device, wd, install, holder) for wd in wins_a]
        A1, A2 = A[:len(A) // 2], A[len(A) // 2:]
        B = [SPS.build_ppl_item(model, proc, device, wd, install, holder) for wd in wins_b]

        def mvf(its):
            def mv(v):
                return SC.mode_matvec(its, v.to(device) * eps, "dc", sign_seed=None, path=path) / eps
            return mv

        # (2) an INDEPENDENT Lanczos start per repeat
        lz_seed = 17 + 101 * rep_idx
        runs = {}
        for name, its in (("A1", A1), ("A2", A2)):
            res = SC.lanczos(mvf(its), d, m=SC.LANCZOS_M, seed=lz_seed, device=device)
            if not bool((res["resid"][:8] <= CONV_RESID_FRAC * max(float(res["evals"][0]), 1e-30)).all()):
                res = SC.lanczos(mvf(its), d, m=M_RESTART, seed=lz_seed, device=device)
                res["restarted"] = True
            runs[name] = res

        def ray_b(u):
            return SC.mode_quadform(B, u.to(device) * eps, "dc", sign_seed=None, path=path) / (eps ** 2)
        spikes, _clusters = SC.stable_spikes(runs["A1"], runs["A2"], ray_b, k=TOP_EVAL_HELDOUT)

        # (3) an INDEPENDENT Hutchinson probe stream per repeat
        gen = torch.Generator(device="cpu").manual_seed(310 + L + 1009 * rep_idx)
        traces_sb = []
        for (lo, hi) in sb:
            kb = hi - lo
            qs = []
            for _ in range(SC.N_TRACE_PROBES):
                g = torch.randn(kb, generator=gen)
                z = (basis32[:, lo:hi].to(device) @ (g / g.norm()).to(device))
                qs.append(SC.mode_quadform(A, z * eps, "dc", sign_seed=None, path=path) / (eps ** 2))
            traces_sb.append((sum(qs) / len(qs)) * kb)

        sp = spikes[:N_SPIKE]
        if sp:
            U = runs["A1"]["evecs"][:, [s["idx"] for s in sp]].double().cpu()
            lam = [float(s["eval_a1"]) for s in sp]
        else:
            U, lam = None, []
        bulk = SC.band_bulk_from_traces([float(t) for t in traces_sb], sb, U, lam, basis)
        Q = CC.materialize_q(U, lam, bulk, basis, sb)

        vg = torch.Generator(device="cpu").manual_seed(4242)
        rels = []
        for _ in range(16):
            v = torch.randn(d, generator=vg, dtype=torch.float64)
            v = v / v.norm()
            qd = float(v @ Q @ v)
            qf = SC.spectral_completion_quadform(v, U, lam, bulk, basis, sb)
            rels.append(abs(qd - qf) / max(abs(qf), 1e-300))
        fid = max(rels)
        assert fid < FIDELITY_TOL, f"L{L} rep{rep_idx}: dense Q disagrees with closed form ({fid:.2e})"

        evals, evecs, k_stiff = CC.stiff_split(Q)
        return {"Q_evals": evals, "Q_evecs": evecs, "k_stiff": int(k_stiff),
                "n_spikes": len(sp), "fidelity_err": fid,
                "restarted": bool(runs["A1"].get("restarted")), "lanczos_seed": lz_seed,
                "op_skip": skip_a}

    banks = {}
    rep["per_layer"] = {}
    for L in layers_want:
        # ---- energy Σ_x (RATION convention: the CENTERED covariance eigenbasis) ------------
        with tm.stage(f"L{L}/energy"):
            st = sigma_state[L]
            mean_x, C_x, absmean = RC.finalize_sigma(st["gram"], st["sum"], st["abs"], st["n"])
            sev, sea = RC._sigma_eig(C_x)

        # ---- ≥3 independent Q measurements → certification --------------------------------
        reps_q = []
        for i in range(a.repeats):
            with tm.stage(f"L{L}/operator_rep{i}"):
                reps_q.append(measure_q(L, i))
            print(f"[qcert] L{L} rep{i}: k_stiff={reps_q[-1]['k_stiff']} "
                  f"n_spikes={reps_q[-1]['n_spikes']} fid={reps_q[-1]['fidelity_err']:.2e}", flush=True)
            _w()

        with tm.stage(f"L{L}/certify"):
            cert = CL.certify_resolution([r["Q_evecs"].double().numpy() for r in reps_q],
                                         ranks=CL.CERT_RANKS,
                                         evals_reps=[r["Q_evals"].double().numpy() for r in reps_q],
                                         d=d)
        # repeat 0 is the canonical bank (deterministic choice, fixed before any read).
        Q_evecs, Q_evals, k_stiff = reps_q[0]["Q_evecs"], reps_q[0]["Q_evals"], reps_q[0]["k_stiff"]
        banks[L] = {"Q_evecs": Q_evecs.float(), "Q_evals": Q_evals.double(),
                    "sigma_evecs": sev.float(), "sigma_evals": sea.double(),
                    "sigma_mean": mean_x.float(), "k_stiff": k_stiff, "d": d,
                    "certified_max": cert["certified_max"],
                    "certified_ranks": list(cert["certified_ranks"]),
                    "metric_key": f"L{L}/dc/ppl"}

        # contrast range (T3 / STENCIL R3): Q ⊥ energy must hold for the arms to differ at all
        cr = {}
        for r in CL.CERT_RANKS:
            cr[r] = float(CW.subspace_overlap(Q_evecs.double()[:, :r].numpy(), sev.double()[:, :r].numpy()))
        rep["per_layer"][L] = {
            "k_stiff": k_stiff, "n_spikes": [r["n_spikes"] for r in reps_q],
            "fidelity_err": [r["fidelity_err"] for r in reps_q],
            "restarted": [r["restarted"] for r in reps_q],
            "certification": cert, "certified_max": cert["certified_max"],
            "contrast_overlap_sus_vs_energy": cr, "sigma_n": int(st["n"])}
        print(f"[qcert] L{L}: CERTIFIED_MAX={cert['certified_max']} "
              f"stability@50={cert['per_rank'][50]['mean_overlap']:.4f} "
              f"(floor {50 / d:.4f}) Q⊥E@50={cr[50]:.4f}", flush=True)
        torch.save(banks, out / "cleave_banks.pt")      # incremental — a late crash keeps done layers
        _w()

    # ---- L0 recipe self-check vs the authoritative carry_q_bank ---------------------------
    if a.validate_l0 and 0 in layers_want:
        with tm.stage("validate_l0"):
            qb = RATS.load_q_bank("/tmp/cleave")
            banked = torch.as_tensor(qb["evecs"]).double()
            measured = banks[0]["Q_evecs"].double()
            ov = {K: float(CW.subspace_overlap(measured[:, :K].numpy(), banked[:, :K].numpy()))
                  for K in (8, 20, 50, 150, 350, 679)}
        rep["l0_validation"] = {"subspace_overlap_measured_vs_banked": ov,
                                "banked_k_stiff": int(qb["k_stiff"]),
                                "measured_k_stiff": banks[0]["k_stiff"]}
        print(f"[qcert] L0 self-check vs carry_q_bank: {ov}", flush=True)
        _w()

    torch.save(banks, out / "cleave_banks.pt")
    rep["banks_sha256"] = hashlib.sha256((out / "cleave_banks.pt").read_bytes()).hexdigest()
    # the headline the whole map-arm design hangs on — reported whichever way it lands.
    rep["certified_max_by_layer"] = {L: rep["per_layer"][L]["certified_max"] for L in layers_want}
    rep["certified_ranks_by_layer"] = {L: banks[L]["certified_ranks"] for L in layers_want}
    rep["resolution_ok_any"] = any(v > 0 for v in rep["certified_max_by_layer"].values())
    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("CLEAVE_QCERT_OK", json.dumps({"certified_max_by_layer": rep["certified_max_by_layer"],
                                         "resolution_ok_any": rep["resolution_ok_any"]},
                                        default=str), flush=True)


if __name__ == "__main__":
    main()
