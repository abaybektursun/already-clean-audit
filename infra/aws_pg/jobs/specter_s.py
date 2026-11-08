"""SPECTER Arm S (GPU) — the causal bar: the top-8 promoted spikes become behavioural cells
(±α ∈ {0.05, 0.1, 0.2}·rms coherent traversals at the L0 site) against Bingham em-nulls
(Σ-energy-matched directions) + same-norm rotated controls, on discovery items; r_quad per spike
from the ppl amplitude scan vs the operator's own quadratic prediction.

Frozen protocol (journal E4): promotion = pool stable spikes across (mode, loss) at L0, dedup at
|⟨u_i,u_j⟩| > 0.9 keeping the larger eigenvalue, rank by eval_a1, take 8. VQA measured at ±0.1
cells + all nulls/controls; ppl at every amplitude. Null band = [min, max] of the 4 Bingham
draws; gap CI + one-sided bootstrap p per readout; BH q=0.10 over the 8×2 family, p emitted.
Quadratic prediction: Δnll = ½·eps_abs²·λ·T̄ (T̄ = 512, the ppl window), mapped through
ppl = base·exp(Δnll); r_quad = largest α with |log-damage measured − predicted| ≤ log 2.

Optionally (--l8-battery, pre-approved by R-C3): the L8 coherent-vs-iid contrast whose R(L8)
prediction was registered by C-3.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_s --type p5.4xlarge,g6e.8xlarge \
      --no-weights --s3-prefix results/specter_s --budget-min 300
"""
import argparse
import json
import math
import os
import pathlib

SPEC_ALPHA_DOC = 0.05          # the frozen-grid rung, kept as the saturation-documentation row
LADDER = (0.5, 1.0, 2.0)       # ×α_i, the per-spike calibrated ladder (½·λ_i·ε²·T̄ = 1 at rung 1)
TARGET_DNLL = 1.0
N_BINGHAM = 4
N_VQA_ITEMS = 200
PROMOTE_DEDUP = 0.9


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--l8-battery", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import atlas_bingham as AB
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("s")
    RS.install_pg()
    rep = {"object": "specter_s_v1", "freeze": freeze}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    g0 = json.loads(SS.fetch(f"{SS.SPECTER_GATE0}/gate0.json", "/tmp/specter/g0.json").read_text())
    o_metrics = json.loads(SS.fetch(f"{SS.SPECTER_O}/metrics.json",
                                    "/tmp/specter/o_metrics.json").read_text())
    assert g0.get("green") and o_metrics.get("green"), "parents not green"

    with tm.stage("load"):
        sha, _ = RS.pinned_pg1_sha("/tmp/specter")
        model, proc = IP.load_pg1(sha=sha)
        layers = IP.decoder_layers(model)
        bank = torch.load(SS.fetch(f"{SS.SPECTER_O}/operator_bank.pt",
                                   "/tmp/specter/operator_bank.pt"),
                          map_location="cpu", weights_only=False)
        lib = torch.load(SS.fetch(f"{SS.SPECTER_O}/library_bank.pt",
                                  "/tmp/specter/library_bank.pt"),
                         map_location="cpu", weights_only=False)
        sig = SS.load_sigma_bank("L0", workdir="/tmp/specter")
    d = IP.D
    dev = "cuda"
    rms = g0["identity"]["site_rms"]
    base_ppl_anchor = SS.FUNC_BASE_PPL

    # ---- promotion (frozen) --------------------------------------------------------
    with tm.stage("promotion"):
        pool = []
        for key, e in bank.items():
            if not key.startswith("L0/"):
                continue
            for s in e.get("spikes", []):
                pool.append({"key": key, "idx": s["idx"], "eval": s["eval_a1"],
                             "u": e["evecs_A1"][:, s["idx"]].double()})
        pool.sort(key=lambda r: -r["eval"])
        promoted = []
        for cand in pool:
            if any(abs(float(cand["u"] @ p["u"])) > PROMOTE_DEDUP for p in promoted):
                continue
            promoted.append(cand)
            if len(promoted) == SC.PROMOTE_K:
                break
        rep["promoted"] = [{"key": p["key"], "idx": p["idx"], "eval": p["eval"]}
                           for p in promoted]
        assert promoted, "no stable spikes to promote — Arm S has no subjects (report + close)"
        # λ is normalized over the FULL item length (image prefix + text): use the banked
        # realized T, never the 512-token text window
        T_bar = bank["_meta"]["T_mean_ppl"]
        rep["T_bar"] = T_bar
        # AMENDMENT (journal E6, pre-unblind for this arm — it crashed before writing any cell
        # number): the frozen α-grid {0.05,0.1,0.2}·rms was inherited from random-direction
        # grammar cells; on spike-ALIGNED directions the operator's own quadratic prediction
        # says those amplitudes obliterate the model (½λ₁ε²T ≈ 250 nats at α=0.05 — measured:
        # mean NLL > 709, the parent instrument overflowed). Per spike i the ladder is
        # calibrated to the quadratic boundary: α_i solves ½·λ_i·(α·rms·√d)²·T̄ = 1, rungs
        # {0.5,1,2}·α_i (±), plus the frozen 0.05 rung as a saturation-documentation row.
        # r_quad's rule is unchanged (largest rung within 2× of the prediction on log-damage).
        preds = []
        for i, p in enumerate(promoted):
            a_cal = math.sqrt(2.0 * TARGET_DNLL / (p["eval"] * T_bar)) / (rms * math.sqrt(d))
            alphas = [round(r * a_cal, 6) for r in LADDER] + [SPEC_ALPHA_DOC]
            row = {"spike": i, "eval": p["eval"], "alpha_cal": a_cal, "alphas": alphas}
            for al in alphas:
                eps_abs = al * rms * math.sqrt(d)
                dnll = 0.5 * (eps_abs ** 2) * p["eval"] * T_bar
                mean_nll_pred = math.log(base_ppl_anchor) + dnll
                row[f"pred_log1p_dppl@{al}"] = SS.safe_log1p_dppl(mean_nll_pred,
                                                                  base_ppl_anchor)
            preds.append(row)
        rep["quadratic_predictions"] = preds
        (out / "pre_unblind_s.json").write_text(json.dumps(
            {"row": "P9/P10", "predictions": preds,
             "rule": ">=5 of 8 behaviourally real (med); r_quad rule unchanged on the "
                     "calibrated ladder"}, indent=2))
    _w()

    # ---- eval scaffold (CellRunner) ------------------------------------------------
    with tm.stage("scaffold"):
        vqa_task, _ = IP.pg1_tasks()
        from infra.aws_pg.jobs._regrain_shared import GRAIN_GATE0
        it = json.loads(SS.fetch(f"{GRAIN_GATE0}/items_vqa.json",
                                 "/tmp/specter/items_vqa.json").read_text())
        rows, hashes = IP.load_items(vqa_task, it["stream_window"])
        split = SS.specter_split(hashes)
        disc_rows = [rows[i] for i in split["disc_idx"][SS.N_OP_VQA + SS.N_OP_HELD:]][:N_VQA_ITEMS]
        intact = IP.eval_pg1(model, proc, vqa_task, disc_rows, dev)
        floor_r = IP.eval_pg1(model, proc, vqa_task, disc_rows, dev, scramble_seed=1234)
        floor = floor_r["score"]
        base_ppl = IP.ppl_gate(model, proc, dev, n_seq=SS.N_PPL_SEQ)["ppl"]
        assert abs(base_ppl - SS.FUNC_BASE_PPL) < SS.PPL_TOL
        rep["reference"] = {"intact_vqa": intact["score"], "floor": floor, "base_ppl": base_ppl}
        runner = SS.SpecterCellRunner(model, proc, vqa_task, disc_rows, intact, floor, base_ppl,
                                      rep, _w)
    _w()

    # ---- cells ----------------------------------------------------------------------
    def dir_cell_ctx(u, eps_abs):
        hook, _ = BC.make_hook_traverse(u.float(), eps_abs)
        return lambda: BC.hook_at(layers, 0, hook)

    Sigma = torch.as_tensor(sig["Sigma"], dtype=torch.float64)
    eigres = torch.linalg.eigh(0.5 * (Sigma + Sigma.T))
    spectrum = eigres.eigenvalues.flip(0).clamp_min(0)
    evecs_desc = eigres.eigenvectors.flip(1).double()

    with tm.stage("cells"):
        gen = torch.Generator(device="cpu").manual_seed(1501)
        for i, p in enumerate(promoted):
            u = p["u"] / p["u"].norm()
            sig_energy = float(u @ Sigma @ u)
            # Bingham em-null: rank-1 draws whose Σ-energy matches the spike's (θ from the
            # sampler's own energy fit; matched on expectation, realized values reported)
            tr = float(spectrum.sum())
            target_E = min(max(sig_energy / tr, float(spectrum.mean() / tr) * 1.001), 0.95 *
                           float(spectrum.max() / tr))
            theta, E_real, saturated = AB.fit_theta_for_energy(spectrum, 1, target_E, seed=11)
            nulls = []
            for s in range(N_BINGHAM):
                V = AB.sample_grassmann_bingham(spectrum, theta, 1, seed=1600 + 100 * i + s)
                nulls.append((evecs_desc @ V[:, 0]).float())
            rep.setdefault("null_meta", {})[f"spike{i}"] = {
                "sigma_energy": sig_energy, "target_E": target_E, "theta": theta,
                "mc_E": E_real, "saturated": bool(saturated)}
            gR = torch.randn(d, generator=gen, dtype=torch.float64)
            rot = (gR - (gR @ u) * u)
            rot = (rot / rot.norm()).float()               # same-norm rotated control
            prow = rep["quadratic_predictions"][i]
            a_cal = prow["alpha_cal"]
            alpha_adj = prow["alphas"][1]                  # the 1.0·α_i rung — adjudicated
            for al in prow["alphas"]:
                doc_rung = abs(al - SPEC_ALPHA_DOC) < 1e-12
                for sign in ((+1, -1) if not doc_rung else (+1,)):
                    eps_abs = al * rms * math.sqrt(d)
                    runner.run(f"spike{i}/a{sign * al}", dir_cell_ctx(sign * u.float(), eps_abs),
                               family="spike", arm="spike", draw=i,
                               skip_vqa=(abs(al - alpha_adj) > 1e-12),
                               alpha=sign * al, spike=i, doc_rung=doc_rung)
            eps_adj = alpha_adj * rms * math.sqrt(d)
            for s, nv in enumerate(nulls):
                runner.run(f"spike{i}/null{s}", dir_cell_ctx(nv, eps_adj),
                           family="spike", arm="bingham_null", draw=s, alpha=alpha_adj, spike=i)
            runner.run(f"spike{i}/rotated", dir_cell_ctx(rot, eps_adj),
                       family="spike", arm="rotated", draw=0, alpha=alpha_adj, spike=i)
    _w()

    # ---- R-S1 + r_quad ---------------------------------------------------------------
    with tm.stage("adjudicate"):
        cells = rep["cells"]

        def sel(spike, arm, alpha=None):
            return [c for c in cells if c.get("spike") == spike and c["arm"] == arm and
                    (alpha is None or abs(c.get("alpha", 9) - alpha) < 1e-9)]
        rs1, pvals, labels = {}, [], []
        for i in range(len(promoted)):
            alpha_adj = rep["quadratic_predictions"][i]["alphas"][1]
            sp_cell = sel(i, "spike", alpha_adj)[0]
            nul = sel(i, "bingham_null")
            rot = sel(i, "rotated")[0]
            for ro, key in (("ppl", "log1p_dppl"), ("vqa", "effect")):
                x = sp_cell.get(key)
                nvals = [c.get(key) for c in nul]
                rv = rot.get(key)
                if x is None or any(v is None for v in nvals) or rv is None:
                    rs1[f"spike{i}/{ro}"] = {"verdict": "VOID", "reason": "missing readout"}
                    pvals.append(1.0)
                    labels.append(f"spike{i}/{ro}")
                    continue
                band_lo, band_hi = min(nvals), max(nvals)
                gap = x - band_hi
                rot_inside = band_lo <= rv <= band_hi
                # one-sided p: normal-approx on the null draws (n=4, disclosed)
                mu, sd = float(np.mean(nvals)), float(np.std(nvals, ddof=1) + 1e-12)
                z = (x - mu) / sd
                pp = float(1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2))))
                pvals.append(pp)
                labels.append(f"spike{i}/{ro}")
                guard = BC.saturation_guard(
                    damage_treat=sp_cell.get("vqa", 0.0) if ro == "vqa" else x,
                    damage_null=np.mean([c.get("vqa", 0.0) if ro == "vqa" else c.get(key, 0.0)
                                         for c in nul]),
                    floor=rep["reference"]["floor"] if ro == "vqa" else 0.0,
                    intact=rep["reference"]["intact_vqa"] if ro == "vqa" else
                    math.log1p(1000.0), name=f"spike{i}/{ro}") if ro == "vqa" else {"void": False}
                rs1[f"spike{i}/{ro}"] = dict(SC.adjudicate_rs1(
                    damage_cilo_above_null=bool(gap > 0 and pp <= 0.05),
                    rotated_inside_band=bool(rot_inside), void=bool(guard.get("void"))),
                    damage=x, null_band=[band_lo, band_hi], rotated=rv, gap=gap, p=pp)
        SS.emit_family(rep, "s_family", pvals, labels)
        rep["rs1"] = rs1
        rquad = {}
        for i, prow in enumerate(rep["quadratic_predictions"]):
            best = 0.0
            for al in prow["alphas"]:
                cs = [c for c in sel(i, "spike", al) + sel(i, "spike", -al)]
                if not cs:
                    continue
                meas = float(np.mean([c["log1p_dppl"] for c in cs]))
                predv = prow[f"pred_log1p_dppl@{al}"]
                if abs(meas - predv) <= math.log(2.0) and predv > 0:
                    best = max(best, al)
            rquad[f"spike{i}"] = best
        rep["r_quad"] = rquad
        n_real = sum(1 for k, v in rs1.items() if v["verdict"] == "H_behaviourally-real")
        rep["rs1_summary"] = {"n_rows": len(rs1), "n_behaviourally_real": n_real}
    _w()

    # ---- optional pre-approved L8 confirmatory battery --------------------------------
    if a.l8_battery:
        with tm.stage("l8_battery"):
            c_metrics = json.loads(SS.fetch(f"{SS.S3_BUCKET}/results/specter_c/out/metrics.json",
                                            "/tmp/specter/c_metrics.json").read_text())
            pred_R = c_metrics["c3"].get("registered_prediction_R_L8")
            assert pred_R is not None, "L8 battery not registered by C-3 — refuse"
            res = {}
            for arm, g in (("coherent", None), ("iid", 1)):
                vals = []
                for ss in range(3):
                    hook = BC.make_hook_gaussian_group(0.1 * rms, 21 + 1000 * ss, g, d)
                    with BC.hook_at(layers, 8, hook):
                        pv = SS.safe_ppl_gate(model, proc, dev, n_seq=SS.N_PPL_SEQ)
                    vals.append(SS.safe_log1p_dppl(pv["mean_nll"], base_ppl))
                res[arm] = vals
            ratio = float(np.mean(res["coherent"]) / max(np.mean(res["iid"]), 1e-12))
            rep["l8_battery"] = {"registered_prediction_R_L8": pred_R,
                                 "measured_ratio": ratio, "cells": res,
                                 "abs_miss": abs(ratio - pred_R)}
        _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_S_OK", json.dumps(rep["rs1_summary"]))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
