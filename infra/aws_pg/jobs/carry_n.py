"""CARRY Arm N — theorem anchoring (spec §3.1 Arm N, §6 R-N). Four legs:

--legs ab  (CPU): (a) the theorem ruler — Arm-H witnesses vs the √d≈45 Lipschitz line and
                  the Ford −σΦ⁻¹(μ) bound; (b) isoperimetry diagnostics on banked PG1
                  activation rows (norm thin-shell CV + 1-Lipschitz functional tails).
--legs cd  (GPU): (c) sink-head zero-QK-curvature check (S23); (d) GGN-vs-full-Hessian at
                  the top eigenpair (S24), beside the banked trace gap.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_n --type r7i.4xlarge \
      --no-weights --s3-prefix results/carry_n_ab --budget-min 90 --args "--legs ab"
  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_n --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/carry_n_cd --budget-min 180 --args "--legs cd"
"""
import argparse
import json
import math
import os
import pathlib

D = 2048
SQRT_D = math.sqrt(2 * D)          # ≈ 45.25 — the S13 √d worst/typical gap for d=2048


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--legs", required=True, choices=("ab", "cd"))
    ap.add_argument("--n-rows", type=int, default=100_000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()          # cd leg streams wikitext + loads the gated PG1 (needs HF online)
    import torch

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import carry_core as CC

    freeze = CS.assert_frozen("n")
    CS.install_pg()
    rep = {"object": f"carry_n_{a.legs}_v1", "freeze": freeze}
    tm = CS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = pathlib.Path("/tmp/carry_n")
    wd.mkdir(parents=True, exist_ok=True)
    scope = json.loads(CS.fetch(f"{CS.CARRY_SCOPE}/metrics.json", wd / "scope.json").read_text())
    assert scope.get("green"), "carry_scope not green"

    if a.legs == "ab":
        rep["gpu_seconds"] = {"total": 0.0, "note": "CPU job"}
        _run_ab(rep, scope, wd, CS, CC, tm, _w)
    else:
        _run_cd(rep, scope, wd, CS, CC, tm, _w, torch)

    if a.legs == "ab":
        print("CARRY_N_AB_OK", json.dumps({"iso_verdict": rep["rn_b"]["L0_all"]["verdict"]}))
    else:
        print("CARRY_N_CD_OK", json.dumps({"rn_c": rep["rn_c"]["verdict"],
                                           "rn_d_ratio": rep["rn_d"]["ratio"]}))
    rep["green"] = True
    _w()


# ===========================================================================
def _run_ab(rep, scope, wd, CS, CC, tm, _w):
    import numpy as np
    import torch

    # (a) the theorem ruler ---------------------------------------------------------------
    with tm.stage("ruler"):
        h_rows = scope["arm_h_rows"]
        ruler = {"sqrt_d_line": SQRT_D, "typical": 1.0, "rows": {}}
        for name, r in h_rows.items():
            draws = r["null_draws"]
            n = len(draws)
            n_sat = sum(1 for x in draws if x >= 30.0)
            mu_hat = (n_sat + 1) / (n + 1)                      # (c+1)/(n+1) floor
            ford_mult = CC.ford_bound(1.0, mu_hat)
            ruler["rows"][name] = {
                "G": r["G"], "battery_log1p_dppl": r["battery_log1p_dppl"],
                "null_typical": r["null_typical"], "Q_ratio": r["Q_ratio"],
                "mu_hat": mu_hat, "ford_multiplier": ford_mult,
                "exceeds_sqrt_d": bool(r["G"] >= SQRT_D)}
        ruler["ford_note"] = ("mu_hat is the (c+1)/(n+1) floor because NO null draw "
                              "obliterates (all < 30 log-units), so the Ford line is a "
                              "LOWER bound on the bound — a descriptive overlay, not a gate")
        rep["ruler"] = ruler
        rep["ggn_gap_L0"] = scope["ggn_gap_L0"]
    _w()

    # (b) isoperimetry diagnostics --------------------------------------------------------
    with tm.stage("isoperimetry"):
        meta = json.loads(CS.fetch(f"{CS.REGRAIN_HARVEST}/harvest_meta.json",
                                   wd / "harvest_meta.json").read_text())
        rep["harvest_meta"] = {"sites": list(meta.get("shards", {}).keys())
                               if isinstance(meta.get("shards"), dict) else meta.get("sites"),
                               "n_rows": meta.get("n_rows")}
        cv_ref = 1.0 / math.sqrt(2 * D)
        rn_b = {}
        for site in ("L0", "L14"):
            shard = CS.fetch(f"{CS.REGRAIN_HARVEST}/shard_{site}_0.pt", wd / f"shard_{site}.pt")
            sd = torch.load(shard, map_location="cpu", weights_only=False)
            X = sd["X"].float().numpy()
            labels = sd["labels"].numpy() if "labels" in sd else None
            for pop, sel in (("all", slice(None)),
                             ("text", labels == 1 if labels is not None else slice(None))):
                Xp = X[sel] if not isinstance(sel, slice) else X
                cap = 300_000
                if Xp.shape[0] > cap:
                    Xp = Xp[:cap]
                if Xp.shape[0] < 10_000:
                    rn_b[f"{site}_{pop}"] = {"verdict": "VOID-too-few-rows",
                                             "n_rows": int(Xp.shape[0])}
                    continue
                norms = np.linalg.norm(Xp, axis=1)
                cv = CC.thin_shell_cv(norms)
                tails = CC.lipschitz_tail_chat(Xp, n_func=64, seed=1337)
                v = CC.adjudicate_rn_b(cv=cv, cv_ref=cv_ref, c_hat=tails["c_hat"])
                v.update({"n_rows": int(Xp.shape[0]), "c_hat_median": tails["c_hat_median"],
                          "norm_mean": float(norms.mean())})
                rn_b[f"{site}_{pop}"] = v
                print(f"[carry-n] iso {site}/{pop}: cv_ratio={v['cv_ratio']:.3f} "
                      f"c_hat={v['c_hat']:.3f} -> {v['verdict']}", flush=True)
                _w()
        # the §6 R-N(b) primary verdict = L0 all-rows
        rn_b["L0_all"] = rn_b["L0_all"]
        rep["rn_b"] = rn_b
    _w()


# ===========================================================================
def _run_cd(rep, scope, wd, CS, CC, tm, _w, torch):
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC

    anat = CS.load_roster(str(wd))
    with tm.stage("load"):
        model, proc = IP.load_pg1(sha=anat["base_sha"])
        layers = IP.decoder_layers(model)
        dev, d = "cuda", IP.D
        n_layers = len(layers)
        n_heads = model.config.text_config.num_attention_heads
        head_dim = model.config.text_config.head_dim
        rep["arch"] = {"n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim}

    # (c) sink-head QK curvature ----------------------------------------------------------
    with tm.stage("head_census"):
        wins = CS.wikitext_windows(proc, 6)
        bos_id = proc.tokenizer.bos_token_id
        # telemetry: all-layer attentions + the L0 sink set per item
        att_all, sink_masks = [], []
        for wdict in wins:
            inp = proc(text=[wdict["text"]], images=[wdict["blank"]],
                       return_tensors="pt", padding=False).to(dev)
            inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
            with torch.no_grad():
                atts = model(input_ids=inp["input_ids"],
                             attention_mask=inp["attention_mask"],
                             pixel_values=inp["pixel_values"],
                             output_attentions=True).attentions
            T = atts[0].shape[-1]
            received0 = atts[0].float().mean(dim=(0, 1, 2))
            mask = (received0 >= 10.0 / T)
            mask[0] = True
            for p in (inp["input_ids"][0] == bos_id).nonzero(as_tuple=True)[0].tolist():
                mask[p] = True
            sink_masks.append(mask.cpu())          # att_all is on CPU — index must match device
            att_all.append([atr.float().cpu() for atr in atts])
        # attn_sink_frac[L,h] = mean over items, query positions of mass on sink keys
        rows = []
        for L in range(n_layers):
            q_items = None
            install_w, holder_w = SC.make_weight_injector(layers[L].self_attn.q_proj)
            q_items = [CS.build_ppl_item(model, proc, dev, wdict, install_w, holder_w)
                       for wdict in wins[:4]]
            d_out = layers[L].self_attn.q_proj.out_features
            for h in range(n_heads):
                sink_fracs = []
                for att, mask in zip(att_all, sink_masks):
                    a_lh = att[L][0, h]                          # (T,T) query×key
                    sink_fracs.append(float(a_lh[:, mask].sum(dim=1).mean()))
                attn_sink_frac = sum(sink_fracs) / len(sink_fracs)
                gen = torch.Generator(device="cpu").manual_seed(2000 + n_heads * L + h)
                curvs = []
                for _p in range(2):
                    u = torch.randn(head_dim, generator=gen, dtype=torch.float64)
                    y = torch.randn(d, generator=gen, dtype=torch.float64)
                    block = torch.outer(u / u.norm(), y / y.norm())      # (head_dim, d)
                    dW = torch.zeros(d_out, d, dtype=torch.float64)
                    dW[h * head_dim:(h + 1) * head_dim, :] = block
                    dW = dW / dW.norm() * 1e-3
                    dWf = dW.float().to(dev)
                    q = sum(SC.ggn_quadform(it, dWf) / it["T"] for it in q_items) / len(q_items)
                    curvs.append(q / (1e-3 ** 2))
                rows.append({"layer": L, "head": h, "attn_sink_frac": attn_sink_frac,
                             "qk_curv": sum(curvs) / len(curvs)})
            print(f"[carry-n] head census L{L} done", flush=True)
            rep["head_rows"] = rows
            _w()
        rep["rn_c"] = CC.adjudicate_rn_c(rows)
    _w()

    # (d) GGN vs full-Hessian at the top eigenpair ----------------------------------------
    with tm.stage("ggn_vs_hessian"):
        install, holder = SC.make_residual_injector(layers, 0)
        items = [CS.build_ppl_item(model, proc, dev, wdict, install, holder)
                 for wdict in CS.wikitext_windows(proc, 12)]

        def matvec_ggn(v):
            return SC.mode_matvec(items, v.to(dev), "dc")

        def matvec_hess(v):
            acc = torch.zeros(d, dtype=torch.float64, device=dev)
            for it in items:
                T = it["T"]
                c = SC.env_coeffs("dc", T, device=dev)
                delta = c.unsqueeze(1) * v.to(dev).unsqueeze(0)
                w = SC.hessian_vp_item(it, delta)
                acc += (c.unsqueeze(1) * w).sum(0).double() / T
            return (acc / len(items)).float()

        rg = SC.lanczos(matvec_ggn, d, m=24, seed=0, device=dev)
        rh = SC.lanczos(matvec_hess, d, m=24, seed=0, device=dev)
        top_ggn = float(rg["evals"][0])
        top_hess = float(rh["evals"][0])
        rep["rn_d"] = CC.adjudicate_rn_d(top_ggn=top_ggn, top_hess=top_hess)
        rep["rn_d"]["banked_ggn_gap_L0"] = scope["ggn_gap_L0"]
        rep["rn_d"]["top_ggn_lanczos"] = top_ggn
        rep["rn_d"]["top_hess_lanczos"] = top_hess
    _w()


if __name__ == "__main__":
    main()
