"""STENCIL banks (GPU) — measure & bank the per-layer susceptibility Q and energy Σ_x, and every
policy basis the fixed-basis arms pin to. Spec §2/§9.3.

Q at a fresh layer reproduces the CARRY/SPECTER recipe VERBATIM (dc/ppl operator): per layer L —
  1. harvest Σ (activation covariance) → `bispec_core.eig_bands` basis + `atlas_fit.subband_slices`;
  2. `specter_core.make_residual_injector`; build 24 ppl operator items (A1/A2) + 12 held-out B;
  3. `specter_core.lanczos`(m=72) on A1,A2 (restart 144) → `stable_spikes` (U, λ);
  4. 16×8 Hutchinson sub-band traces → `band_bulk_from_traces` (bulk);
  5. `carry_core.materialize_q` (verify vᵀQv == closed-form < 1e-8) → `carry_core.stiff_split`.
Energy Σ_x = the CENTERED covariance C_x eigenbasis (RATION convention). L0's Q is the authoritative
`carry_q_bank.pt`; L1–17 measured here. --validate-l0 measures L0 too and checks subspace_overlap vs
the banked L0 Q (the recipe self-check) — run that FIRST (cheap) before the full L1–17 sweep.

  # self-check the recipe on L0 (~0.5 GPU-h):
  ./infra/aws_pg/pg run infra.aws_pg.jobs.stencil_banks --args "--layers 0 --validate-l0" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/stencil_banks_l0 --budget-min 90
  # then the full sweep L1..L17:
  ./infra/aws_pg/pg run infra.aws_pg.jobs.stencil_banks --args "--layers 1-17" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/stencil_banks --budget-min 900
"""
import argparse
import json
import os
import pathlib

CONV_RESID_FRAC = 0.10
M_RESTART = 144
TOP_EVAL_HELDOUT = 16
N_SPIKE = 8                    # completion spikes kept (specter/carry convention)
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
    ap.add_argument("--layers", default="1-17")
    ap.add_argument("--validate-l0", action="store_true")
    ap.add_argument("--n-op-ppl", type=int, default=24)
    ap.add_argument("--n-op-held", type=int, default=12)
    ap.add_argument("--n-sigma-ppl", type=int, default=160)
    ap.add_argument("--n-sigma-vqa", type=int, default=200)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _ration_shared as RATS
    from infra.aws_pg.jobs import _specter_shared as SPS
    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    from infra.pg import carry_core as CC
    from infra.pg import comwrite_core as CW
    from infra.pg import inherit_pg1 as IP
    from infra.pg import ration_core as RC
    from infra.pg import specter_core as SC

    SS.install_pg()
    device, d = "cuda", IP.D
    path = "func"                                   # torch.func jvp/vjp (SPECTER Gate-0 certified)
    layers_want = _parse_layers(a.layers)
    rep = {"object": "stencil_banks_v1", "layers": layers_want, "gpu_seconds": {}}
    tm = SPS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    with tm.stage("load"):
        model, proc, layers, sha, base_ppl = SS.load_model("/tmp/stencil")
    rep["pinned_sha"] = sha
    _w()

    # ---- Σ harvest at ALL 18 layers (one pass): energy C_x + Q-completion Σ ----------------
    with tm.stage("sigma_harvest"):
        sig_windows = SPS.wikitext_windows(proc, a.n_sigma_ppl)
        vqa_task, _coco = IP.pg1_tasks()
        vqa_rows, _ = IP.load_items(vqa_task, a.n_sigma_vqa)
        sigma_state = RATS.accumulate_sigma_all_layers(model, proc, layers, sig_windows,
                                                       vqa_rows=vqa_rows, task=vqa_task, device=device)
    _w()

    # ---- operator batches (ppl only; A1/A2 + held-out B) ----------------------------------
    with tm.stage("op_items_windows"):
        wins_a = SPS.wikitext_windows(proc, a.n_op_ppl, skip=0)
        wins_b = SPS.wikitext_windows(proc, a.n_op_held, skip=a.n_op_ppl)

    def measure_q(L):
        """The full dc/ppl susceptibility Q at layer L → (Q_evals, Q_evecs, k_stiff, n_spikes,
        fidelity_err). Reproduces carry_scope's materialization on a freshly measured operator."""
        st = sigma_state[L]
        Sigma = (torch.as_tensor(st["gram"], dtype=torch.float64) / st["n"])          # uncentered
        bands = BC.eig_bands(Sigma)
        sb = AF.subband_slices(bands["band_slices"], bands["spectrum"])
        basis = torch.as_tensor(bands["basis"], dtype=torch.float64)
        basis32 = basis.to(device, torch.float32)
        rms = float((torch.diagonal(Sigma).sum() / d).sqrt())
        eps = 0.05 * rms

        install, holder = SC.make_residual_injector(layers, L)
        A = [SPS.build_ppl_item(model, proc, device, wd, install, holder) for wd in wins_a]
        A1, A2 = A[:len(A) // 2], A[len(A) // 2:]
        B = [SPS.build_ppl_item(model, proc, device, wd, install, holder) for wd in wins_b]

        def mvf(its):
            def mv(v):
                return SC.mode_matvec(its, v.to(device) * eps, "dc", sign_seed=None, path=path) / eps
            return mv

        runs = {}
        for name, its in (("A1", A1), ("A2", A2)):
            res = SC.lanczos(mvf(its), d, m=SC.LANCZOS_M, seed=17, device=device)
            if not bool((res["resid"][:8] <= CONV_RESID_FRAC * max(float(res["evals"][0]), 1e-30)).all()):
                res = SC.lanczos(mvf(its), d, m=M_RESTART, seed=17, device=device)
                res["restarted"] = True
            runs[name] = res

        def ray_b(u):
            return SC.mode_quadform(B, u.to(device) * eps, "dc", sign_seed=None, path=path) / (eps ** 2)
        spikes, _clusters = SC.stable_spikes(runs["A1"], runs["A2"], ray_b, k=TOP_EVAL_HELDOUT)

        # 16×8 Hutchinson sub-band traces on the full batch A
        gen = torch.Generator(device="cpu").manual_seed(310 + L)
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

        # materialization fidelity (carry_scope gate): dense vᵀQv == closed-form functional
        vg = torch.Generator(device="cpu").manual_seed(4242)
        rels = []
        for _ in range(16):
            v = torch.randn(d, generator=vg, dtype=torch.float64)
            v = v / v.norm()
            qd = float(v @ Q @ v)
            qf = SC.spectral_completion_quadform(v, U, lam, bulk, basis, sb)
            rels.append(abs(qd - qf) / max(abs(qf), 1e-300))
        fid = max(rels)
        assert fid < FIDELITY_TOL, f"L{L}: dense Q disagrees with closed form ({fid:.2e})"

        evals, evecs, k_stiff = CC.stiff_split(Q)
        return {"Q_evals": evals, "Q_evecs": evecs, "k_stiff": int(k_stiff),
                "n_spikes": len(sp), "spike_evals": lam, "fidelity_err": fid,
                "restarted": bool(runs["A1"].get("restarted"))}

    banks = {}
    rep["per_layer"] = {}
    for L in layers_want:
        with tm.stage(f"L{L}/energy"):
            st = sigma_state[L]
            mean_x, C_x, absmean = RC.finalize_sigma(st["gram"], st["sum"], st["abs"], st["n"])
            sev, sea = RC._sigma_eig(C_x)
        if L == 0 and not a.validate_l0:
            # authoritative banked L0 Q (dc/ppl) — reuse, do not re-measure
            qb = RATS.load_q_bank("/tmp/stencil")
            Q_evecs = torch.as_tensor(qb["evecs"]); Q_evals = torch.as_tensor(qb["evals"])
            k_stiff, n_spikes, fid, restarted = int(qb["k_stiff"]), None, None, None
        else:
            with tm.stage(f"L{L}/operator"):
                mq = measure_q(L)
            Q_evecs, Q_evals, k_stiff = mq["Q_evecs"], mq["Q_evals"], mq["k_stiff"]
            n_spikes, fid, restarted = mq["n_spikes"], mq["fidelity_err"], mq["restarted"]
        banks[L] = {"Q_evecs": Q_evecs.float(), "Q_evals": Q_evals.double(),
                    "sigma_evecs": sev.float(), "sigma_evals": sea.double(),
                    "sigma_mean": mean_x.float(), "k_stiff": k_stiff, "d": d,
                    "metric_key": f"L{L}/dc/ppl"}
        # contrast-range table (T3): subspace_overlap(U_sus, U_energy) across the rank ladder
        from infra.pg import stencil_core as STC
        cr = {}
        for r in STC.RANK_LADDER:
            if r <= d:
                cr[r] = STC.contrast_range({"Q_evecs": Q_evecs, "sigma_evecs": sev, "d": d}, r)["overlap"]
        rep["per_layer"][L] = {"k_stiff": k_stiff, "n_spikes": n_spikes, "fidelity_err": fid,
                               "restarted": restarted, "contrast_overlap": cr,
                               "sigma_n": int(st["n"])}
        print(f"[banks] L{L}: k_stiff={k_stiff} n_spikes={n_spikes} "
              f"overlap@50={cr.get(50):.4f} fid={fid}", flush=True)
        torch.save(banks, out / "stencil_banks.pt")     # incremental save — a late crash keeps done layers
        _w()

    # ---- L0 recipe self-check (measured vs banked) ----------------------------------------
    if a.validate_l0 and 0 in layers_want:
        with tm.stage("validate_l0"):
            qb = RATS.load_q_bank("/tmp/stencil")
            banked = torch.as_tensor(qb["evecs"]).double()
            measured = banks[0]["Q_evecs"].double()
            ov = {K: float(CW.subspace_overlap(measured[:, :K].numpy(), banked[:, :K].numpy()))
                  for K in (8, 20, 50, 150, 350, 679)}
        rep["l0_validation"] = {"subspace_overlap_measured_vs_banked": ov,
                                "banked_k_stiff": int(qb["k_stiff"]),
                                "measured_k_stiff": banks[0]["k_stiff"]}
        print(f"[banks] L0 self-check overlap vs carry_q_bank: {ov}", flush=True)
        _w()

    torch.save(banks, out / "stencil_banks.pt")
    import hashlib
    rep["banks_sha256"] = hashlib.sha256((out / "stencil_banks.pt").read_bytes()).hexdigest()
    SPS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()
    print("STENCIL_BANKS_OK", json.dumps({"layers": layers_want,
          "k_stiff": {L: rep["per_layer"][L]["k_stiff"] for L in layers_want}}, default=str), flush=True)


if __name__ == "__main__":
    main()
