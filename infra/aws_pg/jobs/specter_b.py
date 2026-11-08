"""SPECTER Arm B (GPU harvest + in-job CPU curves; the protected curiosity slice) — the ATLAS
R10 surprise (L8 = L14 dispersion collapse to 0.015) becomes a mapped invariant or a
counterexample map:

  1. Σ at ALL 18 residual sites in ONE pass (Gram accumulators, fp64; 300 wikitext windows +
     60 VQA items ≈ 170k rows — the bispec BOS-exclusion convention does not apply here because
     accumulation is over ALL positions of B=1 forwards, disclosed).
  2. Per layer: the house level-set dispersion curve (rank 25, rungs r·E_max(25),
     r ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}, 5 draws/rung, A7 energy-dispersion metric).
  3. R-B1: pairwise collapse matrix (tol 0.05, layers 4–17) + the peel-off boundary.
     Registered predictions journaled BEFORE the curves are computed.
  4. Hybrid-spectrum sensitivity: swap top-r eigenvalues (r ∈ {1, 8, 64}) between the collapse
     representative and the worst outlier, recompute — names the responsible spectral feature.
  5. Whitening-exponent fit: F̄ top-64 eigenvalues (operator bank) vs Σ eigenvalues at the same
     ranks, log–log slope + CI per site — the registered expectation is NO single exponent.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_b --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/specter_b --budget-min 240
"""
import argparse
import json
import os
import pathlib

N_WIN = 300
N_VQA = 60
RUNGS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
RANK = 25
N_DRAWS = 5
SWAP_R = (1, 8, 64)


def dispersion_curve(spectrum, evecs, rungs=RUNGS, k=RANK, n_draws=N_DRAWS, seed0=500):
    """The house level-set dispersion curve from a spectrum. Everything runs in ASCENDING
    eigenvalue order (the eigh convention max_removable_energy flips internally — passing a
    descending spectrum silently computes E_max from the BOTTOM-k, the run-2 bug). Rung targets
    are clamped into the sampler's numerically reachable range (flat deep-layer spectra have
    narrow ranges); realized targets returned beside the curve."""
    import torch

    from infra.pg.bispec_core import (energy_dispersion, max_removable_energy,
                                      sample_level_set, sample_level_set_raw)
    lam_asc = torch.as_tensor(spectrum, dtype=torch.float64).clamp_min(0).sort().values
    d_ = lam_asc.shape[0]
    Sigma_diag = torch.diag(lam_asc)
    eig = (lam_asc, torch.eye(d_, dtype=torch.float64))
    E_max = max_removable_energy(Sigma_diag, k, eig=(lam_asc, None))
    _, E_lo = sample_level_set_raw(Sigma_diag, k, -30.0, seed=0, eig=eig)
    _, E_hi = sample_level_set_raw(Sigma_diag, k, 30.0, seed=0, eig=eig)
    curve, realized = [], []
    for r in rungs:
        target = min(max(r * E_max, E_lo * 1.02), E_hi * 0.999)
        frames = [sample_level_set(Sigma_diag, k, target, seed=seed0 + s, eig=eig)[0]
                  for s in range(n_draws)]
        disp = energy_dispersion(frames, Sigma_diag)
        curve.append(disp["mean_distance"])
        realized.append(target)
    return curve, E_max, realized


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("b")
    RS.install_pg()
    rep = {"object": "specter_b_v1", "freeze": freeze}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # registered predictions FIRST (spec §6 R-B1 + P13/P14)
    (out / "pre_unblind_b.json").write_text(json.dumps(
        {"rows": ["P13", "P14"],
         "P13": "collapse holds on layers 4-17 (pairwise max|Δ| ≤ 0.05); peel-off at L ≤ 3",
         "P14": "no single whitening exponent survives across sites"}, indent=2))

    with tm.stage("load"):
        sha, _ = RS.pinned_pg1_sha("/tmp/specter")
        model, proc = IP.load_pg1(sha=sha)
        layers = IP.decoder_layers(model)
    d = IP.D
    dev = "cuda"

    # ---- 18-site Σ harvest in one pass ---------------------------------------------
    with tm.stage("harvest"):
        sums = {L: torch.zeros(d, dtype=torch.float64) for L in range(18)}
        gram = {L: torch.zeros(d, d, dtype=torch.float64) for L in range(18)}
        cnt = {L: 0 for L in range(18)}

        def _mk(L):
            def hook(_m, args, kwargs):
                hs = args[0] if args else kwargs["hidden_states"]
                x = hs.detach().reshape(-1, d).double().cpu()
                sums[L] += x.sum(0)
                gram[L] += x.T @ x
                cnt[L] += x.shape[0]
            return hook
        handles = [layers[L].register_forward_pre_hook(_mk(L), with_kwargs=True)
                   for L in range(18)]
        wins = SS.wikitext_windows(proc, N_WIN, verify_proc=True)
        rep["n_windows_dropped_unlocatable"] = wins[-1]["n_dropped_unlocatable"]
        import contextlib as _ctx

        @_ctx.contextmanager
        def _noop():
            yield
        with torch.no_grad():
            for wd in wins:
                item = SS.build_ppl_item(model, proc, dev, wd, _noop, {})
                item["forward_fn"]()
        vqa_task, _ = IP.pg1_tasks()
        from infra.aws_pg.jobs._regrain_shared import GRAIN_GATE0
        it = json.loads(SS.fetch(f"{GRAIN_GATE0}/items_vqa.json",
                                 "/tmp/specter/items_vqa.json").read_text())
        rows, hashes = IP.load_items(vqa_task, it["stream_window"])
        split = SS.specter_split(hashes)
        with torch.no_grad():
            for i in list(split["disc_idx"])[:N_VQA]:
                item = SS.build_vqa_item(model, proc, dev, rows[i], vqa_task, _noop, {})
                item["forward_fn"]()
        for h in handles:
            h.remove()
        spectra = {}
        for L in range(18):
            mu = sums[L] / cnt[L]
            Sig = gram[L] / cnt[L] - torch.outer(mu, mu)
            ev = torch.linalg.eigvalsh(0.5 * (Sig + Sig.T)).flip(0).clamp_min(0)
            spectra[L] = ev
        rep["harvest"] = {"n_rows": cnt[0], "n_windows": N_WIN, "n_vqa": N_VQA}
        torch.save({L: spectra[L] for L in spectra}, out / "spectra_18.pt")
    _w()

    # ---- curves + collapse + peel-off ------------------------------------------------
    with tm.stage("curves"):
        curves, emax, realized_t = {}, {}, {}
        for L in range(18):
            c, em, rt = dispersion_curve(spectra[L], None)
            curves[L] = c
            emax[L] = em
            realized_t[L] = rt
            print(f"[b] L{L}: curve={[round(x, 4) for x in c]} E_max={em:.3f}", flush=True)
        rep["curves"] = {str(L): curves[L] for L in curves}
        rep["E_max_rank25"] = {str(L): emax[L] for L in emax}
        rep["realized_targets"] = {str(L): realized_t[L] for L in realized_t}
        rep["rb1"] = SC.adjudicate_rb1_collapse(curves)
        print(f"[b] R-B1 {rep['rb1']['verdict']} peel={rep['rb1']['peel_off_layer']}",
              flush=True)
    _w()

    # ---- hybrid-spectrum sensitivity ---------------------------------------------------
    with tm.stage("hybrid"):
        Ls = sorted(curves)
        ref_L = 14
        dev_of = {L: float(np.max(np.abs(np.asarray(curves[L]) - np.asarray(curves[ref_L]))))
                  for L in Ls if L != ref_L}
        out_L = max(dev_of, key=dev_of.get)
        rows_h = {}
        for r in SWAP_R:
            hy = spectra[out_L].clone()
            hy[:r] = spectra[ref_L][:r]
            c_hy, _, _ = dispersion_curve(hy, None)
            rows_h[f"top{r}_from_L{ref_L}_into_L{out_L}"] = {
                "curve": c_hy,
                "dev_to_ref": float(np.max(np.abs(np.asarray(c_hy) -
                                                  np.asarray(curves[ref_L])))),
                "dev_before": dev_of[out_L]}
        rep["hybrid_swap"] = {"outlier_layer": out_L, "reference_layer": ref_L, "rows": rows_h}
    _w()

    # ---- whitening-exponent fit (F̄ vs Σ, log–log) --------------------------------------
    with tm.stage("whitening"):
        o_metrics = json.loads(SS.fetch(f"{SS.SPECTER_O}/metrics.json",
                                        "/tmp/specter/o_metrics.json").read_text())
        bank = torch.load(SS.fetch(f"{SS.SPECTER_O}/operator_bank.pt",
                                   "/tmp/specter/operator_bank.pt"),
                          map_location="cpu", weights_only=False)
        bank2 = torch.load(SS.fetch(f"{SS.S3_BUCKET}/results/specter_o_l8l14/out/operator_bank.pt",
                                    "/tmp/specter/operator_bank2.pt"),
                           map_location="cpu", weights_only=False)
        bank2.pop("_meta", None)
        bank.update(bank2)
        fits = {}
        for key in bank:
            if not key.endswith("/dc/ppl") or key.startswith("_"):
                continue                       # filter BEFORE parsing (the '_meta' crash)
            site = key.split("/")[0]
            L = int(site[1:])
            fe = np.asarray(bank[key]["evals_A1"][:64], float)
            se = np.asarray(spectra[L][:64], float)
            ok = (fe > 0) & (se > 0)
            if ok.sum() < 8:
                fits[site] = {"n": int(ok.sum()), "slope": None}
                continue
            x, y = np.log(se[ok]), np.log(fe[ok])
            slope, intercept = np.polyfit(x, y, 1)
            rng = np.random.default_rng(3)
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, ok.sum(), int(ok.sum()))
                if np.unique(x[idx]).size < 3:
                    continue
                boots.append(np.polyfit(x[idx], y[idx], 1)[0])
            fits[site] = {"n": int(ok.sum()), "slope": float(slope),
                          "ci95": [float(np.percentile(boots, 2.5)),
                                   float(np.percentile(boots, 97.5))]}
        rep["whitening"] = fits
    _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_B_OK", json.dumps({"rb1": rep["rb1"]["verdict"],
                                      "peel": rep["rb1"]["peel_off_layer"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
