"""SPECTER Arm H (GPU) — the non-quadratic exceptional set, constructively: projected gradient
ascent on damage at FIXED per-token energy, two families (coherent d-vector; absolute-position-
keyed per-token field), α ∈ {0.1, 0.2}·rms, 16 restarts each, on a 32-window ppl PROXY; every
found direction is re-measured on the STANDARD battery (the number of record) against
Bingham-typical damage at matched energy, and against its own quadratic prediction (live
quadform along the found direction).

R-H1 adjudicated on the max-G row (the hunt's claim is existential); per-row verdicts reported;
VOID-flat only if every row's optimizer improved < 1.2×. Frozen ascent: Adam lr=0.05·rms,
40 steps, lr halved after 3 consecutive decreases (the pre-registered divergence response).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_h --type p5.4xlarge,g6e.8xlarge \
      --no-weights --s3-prefix results/specter_h --budget-min 300
"""
import argparse
import json
import math
import os
import pathlib

N_RESTARTS = 16
N_STEPS = 40
N_PROXY = 32
ALPHAS = (0.1, 0.2)
N_NULL = 4


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
    from infra.pg import atlas_bingham as AB
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("h")
    RS.install_pg()
    rep = {"object": "specter_h_v1", "freeze": freeze}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    g0 = json.loads(SS.fetch(f"{SS.SPECTER_GATE0}/gate0.json", "/tmp/specter/g0.json").read_text())
    assert g0.get("green"), "Gate-0 not green"
    path = g0["diff_path"]

    with tm.stage("load"):
        sha, _ = RS.pinned_pg1_sha("/tmp/specter")
        model, proc = IP.load_pg1(sha=sha)
        layers = IP.decoder_layers(model)
        sig = SS.load_sigma_bank("L0", workdir="/tmp/specter")
    d = IP.D
    dev = "cuda"
    rms = g0["identity"]["site_rms"]
    L = 0

    with tm.stage("proxy_items"):
        wins = SS.wikitext_windows(proc, N_PROXY, skip=SS.N_OP_PPL + SS.N_OP_HELD)
        install, holder = SC.make_residual_injector(layers, L)
        items = [SS.build_ppl_item(model, proc, dev, wd, install, holder) for wd in wins]
        base_ppl = IP.ppl_gate(model, proc, dev, n_seq=SS.N_PPL_SEQ)["ppl"]
        assert abs(base_ppl - SS.FUNC_BASE_PPL) < SS.PPL_TOL

    def item_nll(it, delta_fn):
        """One window's weighted NLL with δ = delta_fn(T) installed; differentiable."""
        holder["delta"] = delta_fn(it["T"])
        logits = it["forward_fn"]()[0]
        lab = it["labels"].to(logits.device)
        live = lab != -100
        lse = torch.logsumexp(logits[live].float(), dim=-1)
        nll = lse - logits[live].float().gather(1, lab[live].unsqueeze(1)).squeeze(1)
        return (nll * it["loss_w"][live]).sum()

    def proxy_loss(delta_fn, subset=None):
        """Mean NLL over proxy windows (no_grad callers only — one graph per item otherwise)."""
        its = subset if subset is not None else items
        return sum(float(item_nll(it, delta_fn)) for it in its) / len(its)

    with torch.no_grad():
        clean_nll = float(proxy_loss(lambda T: torch.zeros(T, d, device=dev)))
    rep["proxy_clean_nll"] = clean_nll
    # the per-token field covers the FULL site sequence (image prefix + text): T ≈ 771, not 512
    T_field = max(it["T"] for it in items)
    rep["T_field"] = T_field
    _w()

    # ---- the hunt ---------------------------------------------------------------------
    found = {}
    with tm.stage("hunt"):
        for fam in ("coherent", "field"):
            for al in ALPHAS:
                eps_abs = al * rms * math.sqrt(d)
                best = {"damage": -1.0}
                inits = []
                for r in range(N_RESTARTS):
                    gen = torch.Generator(device="cpu").manual_seed(1700 + r)
                    if fam == "coherent":
                        theta = torch.randn(d, generator=gen).to(dev)
                    else:
                        theta = torch.randn(T_field, d, generator=gen).to(dev)
                    theta = theta.requires_grad_(True)
                    opt = torch.optim.Adam([theta], lr=0.05 * rms)
                    lr_halvings, down_run, prev = 0, 0, None
                    if fam == "coherent":
                        def dfn(T, th=theta):
                            v = SC.project_coherent(th, eps_abs)
                            return v.unsqueeze(0).expand(T, d)
                    else:
                        def dfn(T, th=theta):
                            f = SC.project_pertoken_energy(th, eps_abs)
                            assert T <= f.shape[0], f"field shorter than item ({T})"
                            return f[:T]
                    with torch.no_grad():
                        init_dam = max(float(proxy_loss(dfn)) - clean_nll, 1e-9)
                    for step in range(N_STEPS):
                        opt.zero_grad(set_to_none=True)
                        mb = [items[(step * 8 + j) % len(items)] for j in range(8)]
                        acc = 0.0
                        for it in mb:                     # per-item backward: ONE graph at a time
                            l_i = item_nll(it, dfn) / len(mb)
                            (-l_i).backward()
                            acc += float(l_i)
                        dam = acc - clean_nll
                        opt.step()
                        if prev is not None and dam < prev:
                            down_run += 1
                            if down_run >= 3:
                                for gparam in opt.param_groups:
                                    gparam["lr"] *= 0.5
                                lr_halvings += 1
                                down_run = 0
                        else:
                            down_run = 0
                        prev = dam
                    with torch.no_grad():
                        final = float(proxy_loss(dfn)) - clean_nll
                    inits.append({"restart": r, "init_dam": init_dam, "final_dam": final,
                                  "lr_halvings": lr_halvings})
                    if final > best["damage"]:
                        with torch.no_grad():
                            if fam == "coherent":
                                keep = SC.project_coherent(theta.detach(), eps_abs).cpu()
                            else:
                                keep = SC.project_pertoken_energy(theta.detach(), eps_abs).cpu()
                        best = {"damage": final, "restart": r, "delta": keep}
                    print(f"[h] {fam}@{al} r{r}: init={init_dam:.4f} final={final:.4f}",
                          flush=True)
                improved = best["damage"] / max(min(x["init_dam"] for x in inits), 1e-9)
                found[f"{fam}@{al}"] = {"best_proxy_damage": best["damage"],
                                        "restarts": inits, "improved_factor": improved,
                                        "delta": best["delta"], "eps_abs": eps_abs,
                                        "alpha": al, "family": fam}
                rep.setdefault("hunt", {})[f"{fam}@{al}"] = {
                    k: v for k, v in found[f"{fam}@{al}"].items() if k != "delta"}
                _w()

    # ---- re-measure on the standard battery + nulls + quadratic prediction --------------
    with tm.stage("remeasure"):
        Sigma = torch.as_tensor(sig["Sigma"], dtype=torch.float64)
        eig = torch.linalg.eigh(0.5 * (Sigma + Sigma.T))
        spectrum = eig.eigenvalues.flip(0).clamp_min(0)
        evecs_desc = eig.eigenvectors.flip(1).double()
        rows = {}
        for key, f in found.items():
            eps_abs = f["eps_abs"]
            if f["family"] == "coherent":
                hook, _ = BC.make_hook_traverse(f["delta"].float(), eps_abs)
                ctx = lambda h=hook: BC.hook_at(layers, L, h)
            else:
                fld = f["delta"].float()

                class _FieldHook:
                    def __init__(self, fl):
                        self.fl = fl
                        self.tracker = BC._PositionTracker()

                    def __call__(self, _m, args, kwargs):
                        from infra.pg.inherit_core import _split_hidden
                        hs, rebuild = _split_hidden(args, kwargs)
                        B, T, D = hs.shape
                        pos = self.tracker.positions(T).clamp_max(self.fl.shape[0] - 1)
                        dl = self.fl.to(hs.device)[pos]
                        return rebuild((hs.float() + dl.unsqueeze(0)).to(hs.dtype))
                hook = _FieldHook(fld)
                ctx = lambda h=hook: BC.hook_at(layers, L, h)
            with ctx():
                pg_res = SS.safe_ppl_gate(model, proc, dev, n_seq=SS.N_PPL_SEQ)
            dam = SS.safe_log1p_dppl(pg_res["mean_nll"], base_ppl)
            # matched-energy null typical
            nulls = []
            for s in range(N_NULL):
                if f["family"] == "coherent":
                    tr = float(spectrum.sum())
                    u = f["delta"].double() / f["delta"].double().norm()
                    tE = min(max(float(u @ Sigma @ u) / tr,
                                 float(spectrum.mean() / tr) * 1.001),
                             0.95 * float(spectrum.max() / tr))
                    th, _, _sat = AB.fit_theta_for_energy(spectrum, 1, tE, seed=11)
                    V = AB.sample_grassmann_bingham(spectrum, th, 1, seed=1800 + s)
                    nv = (evecs_desc @ V[:, 0]).float()
                    nhook, _ = BC.make_hook_traverse(nv, eps_abs)
                else:
                    nhook = BC.make_hook_gaussian_group(f["alpha"] * rms, 1900 + s, 1, d)
                with BC.hook_at(layers, L, nhook):
                    pn = SS.safe_ppl_gate(model, proc, dev, n_seq=SS.N_PPL_SEQ)
                nulls.append(SS.safe_log1p_dppl(pn["mean_nll"], base_ppl))
            # quadratic prediction along the found direction: live quadform (discovery items)
            if f["family"] == "coherent":
                dvec = f["delta"].float().to(dev)
                qn = SC.mode_quadform(items[:8], dvec, "dc", path=path) / \
                    float(dvec.norm() ** 2)
                dnll = 0.5 * float(dvec.norm() ** 2) * qn * T_field
            else:
                fields = [f["delta"].float().to(dev)[:it["T"]] for it in items[:8]]
                qraw = SC.field_quadform(items[:8], fields, path=path)
                dnll = 0.5 * qraw * T_field
            pred = math.log1p(base_ppl * (math.exp(min(dnll, 30.0)) - 1.0))
            typical = float(np.mean(nulls))
            G = dam / max(typical, 1e-9)
            Q_ratio = dam / max(pred, 1e-9)
            rows[key] = {"battery_log1p_dppl": dam, "null_typical": typical,
                         "null_draws": nulls, "quad_pred_log1p": pred, "G": G,
                         "Q_ratio": Q_ratio,
                         "improved_factor": found[key]["improved_factor"]}
            print(f"[h] {key}: dam={dam:.4f} typical={typical:.4f} G={G:.2f} "
                  f"Q={Q_ratio:.2f}", flush=True)
        rep["remeasure"] = rows
        all_flat = all(r["improved_factor"] < 1.2 for r in rows.values())
        head_key = max(rows, key=lambda k: rows[k]["G"])
        hr = rows[head_key]
        rep["rh1_per_row"] = {k: SC.adjudicate_rh1(G=r["G"], Q_ratio=r["Q_ratio"],
                                                   improved_factor=r["improved_factor"])
                              for k, r in rows.items()}
        rep["rh1"] = dict(SC.adjudicate_rh1(
            G=hr["G"], Q_ratio=hr["Q_ratio"],
            improved_factor=(1.0 if all_flat else max(r["improved_factor"]
                                                      for r in rows.values()))),
            headline_row=head_key)
        # bank the witness directions (constructive deliverable)
        torch.save({k: f["delta"] for k, f in found.items()}, out / "witness_directions.pt")
    _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_H_OK", json.dumps({"rh1": rep["rh1"]["verdict"],
                                      "headline": rep["rh1"]["headline_row"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
