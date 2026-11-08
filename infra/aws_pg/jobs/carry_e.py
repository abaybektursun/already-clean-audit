"""CARRY Arm E (GPU) — error-family unification battery at L0 (spec §3.1 Arm E, §6 R-E,
§9.3). Eight error families injected at the residual entering layers[0]; per-family damage
predictions journaled BEFORE the battery (enforced by stage order: the `predictions` stage
writes pre_unblind_e.json + rep before any battery cell). The measured operator (the L0/dc/ppl
completion) predicts per-family damage from each family's realized error field; the battery
measures it; R-E adjudicates ρ vs the label-shuffle null. The QuaRot sub-row (S18) is the
incoherence-processed quantization family, reported against raw either way.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_e --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/carry_e --budget-min 300
"""
import argparse
import json
import os
import pathlib

AMP_MULTS = (0.25, 0.5, 1.0)      # ×the family's headline construction (the Lᵖ ladder)
HEADLINE = 0.8                    # ×rms — the frozen headline quant/gaussian amplitude
N_PRED_ROWS = 512                 # rows sampled for the per-family predicted stiffness


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
    ap.add_argument("--n-vqa", type=int, default=200)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import bispec_core as BC
    from infra.pg import carry_core as CC
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg import specter_core as SC
    from infra.pg.inherit_core import _split_hidden

    freeze = CS.assert_frozen("e")
    CS.install_pg()
    rep = {"object": "carry_e_v1", "freeze": freeze}
    tm = CS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = pathlib.Path("/tmp/carry_e")
    wd.mkdir(parents=True, exist_ok=True)
    g0 = json.loads(CS.fetch(f"{CS.CARRY_GATE0}/gate0.json", wd / "gate0.json").read_text())
    assert g0.get("green"), "carry gate0 not green"
    anat = CS.load_roster(str(wd))
    U, lam, bulk, basis, sb, _ = CS.load_completion_env(str(wd))
    D = IP.D
    H = CC.hadamard_matrix(D).float()

    with tm.stage("load"):
        model, proc = IP.load_pg1(sha=anat["base_sha"])
        layers = IP.decoder_layers(model)
        dev = "cuda"
        X = torch.load(CS.fetch(f"{CS.REGRAIN_HARVEST}/shard_L0_0.pt", wd / "shard_L0.pt"),
                       map_location="cpu", weights_only=False)["X"][:8192].float().cuda()
        rms = float(X.pow(2).mean().sqrt())
        step = HEADLINE * rms
        Xr = X @ H.cuda()
        rms_r = float(Xr.pow(2).mean().sqrt())
        step_r = HEADLINE * rms_r
        ae, _cfg, _q, _sha = GB.load_dict_bank(
            CS.fetch(f"{CS.GRAIN_DICTS_L0}/dict_primary_m1_s0.pt", wd / "dict_m1.pt"))
        ae = ae.cuda()
        Rrot = RC.haar_rotation(D, 4242, device="cuda", dtype=torch.float32)
        rep["site"] = {"rms": rms, "step": step, "rms_rotated": rms_r, "step_r": step_r}
    _w()

    # ---- the family constructions (each returns an error field e(x), (T,D)) -------------
    def e_sae(x):
        return x - ae.decode(ae.encode(x.float(), use_threshold=True))    # SUBSTITUTION sign

    def e_quant(x, st):
        return BC.quant_error(x.float(), st)

    def _qshuffle_error(x, st, seed):
        e = BC.quant_error(x.float(), st)
        T = x.shape[0]
        if T <= 1:
            return e
        stat = x.float().mean(-1)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        q = torch.quantile(stat.double().cpu(),
                           torch.linspace(0, 1, 11, dtype=torch.float64))
        binid = torch.bucketize(stat.double().cpu(), q[1:-1])
        perm = torch.arange(T)
        for b in range(10):
            idx = torch.nonzero(binid == b).flatten()
            if idx.numel() > 1:
                perm[idx] = idx[torch.randperm(idx.numel(), generator=gen)]
        return e[perm.to(e.device)]

    # sign convention: SAE substitutes x→x−e (so hook adds −amp·e); all others add +amp·e
    FAMILIES = {
        "sae_error": ("sub", lambda x, m: e_sae(x)),
        "quant": ("add", lambda x, m: e_quant(x, m * step)),
        "quant_signflip": ("sub", lambda x, m: e_quant(x, m * step)),
        "quant_qshuffle": ("add", lambda x, m, s=0: _qshuffle_error(x, m * step, 40 + s)),
        "quant_perm": ("permadd", lambda x, m, s=0: e_quant(x, m * step)),
        "gaussian_iid": ("gauss", None),
        "rotated_error": ("add", lambda x, m: e_quant(x, m * step) @ Rrot.T),
        "quant_incoherent": ("incoh", None),
    }
    MULTIDRAW = {"quant_qshuffle", "quant_perm", "gaussian_iid"}

    def build_field(name, x, mult, seed=0):
        """The realized (T,D) error field a hook would add at this amplitude/seed."""
        kind, fn = FAMILIES[name]
        x = x.float()
        if name == "gaussian_iid":
            gen = torch.Generator(device="cpu").manual_seed(11 + 1000 * seed)
            g = (mult * HEADLINE * rms) * torch.randn(x.shape, generator=gen).to(x.device)
            return g
        if name == "quant_incoherent":
            xr = x @ H.to(x.device)
            er = BC.quant_error(xr, mult * step_r)
            return er @ H.to(x.device).T
        if name == "quant_qshuffle":
            return fn(x, mult, seed)
        if name == "quant_perm":
            e = BC.quant_error(x, mult * step)
            T = x.shape[0]
            if T > 1:
                gen = torch.Generator(device="cpu").manual_seed(5 + seed)
                e = e[torch.randperm(T, generator=gen).to(e.device)]
            return e
        f = fn(x, mult)
        return -f if kind == "sub" else f

    def make_hook(name, mult, seed=0):
        def hook(_m, args, kwargs):
            hs, rebuild = _split_hidden(args, kwargs)
            field = build_field(name, hs[0], mult, seed)
            return rebuild((hs.float() + field.unsqueeze(0)).to(hs.dtype))
        return hook

    # ---- predictions BEFORE the battery (§9.3) ------------------------------------------
    with tm.stage("predictions"):
        def q_form(vec):
            return SC.spectral_completion_quadform(vec, U, lam, bulk, basis, sb)
        Uc = None if U is None else U.double().cuda()
        pred, spike_align = {}, {}
        for name in FAMILIES:
            fld = build_field(name, X, 1.0, seed=0)           # (8192, D) headline
            idx = torch.randperm(X.shape[0])[:N_PRED_ROWS]
            qs, als = [], []
            for i in idx.tolist():
                e = fld[i].double()
                nrm2 = float((e ** 2).sum())
                if nrm2 <= 0:
                    continue
                eh = (e / nrm2 ** 0.5)
                qs.append(q_form(eh.cpu()) * nrm2)
                if Uc is not None:
                    als.append(float(((Uc.T @ eh.cuda()) ** 2).sum()))
            pred[name] = float(np.mean(qs)) if qs else 0.0
            spike_align[name] = float(np.mean(als)) if als else 0.0
        rep["predictions"] = {"q_pred": pred, "spike_alignment": spike_align}
        preunblind = {"P8": "H_operator-unifies, rho >= 0.8; quantization benign BECAUSE "
                            "low spike alignment (med-high)",
                      "P9": "incoherence-processing reduces quantization damage further (med)",
                      "q_pred": pred, "spike_alignment": spike_align}
        (out / "pre_unblind_e.json").write_text(json.dumps(preunblind, indent=2))
        rep["pre_unblind"] = preunblind
    _w()

    # ---- battery ------------------------------------------------------------------------
    with tm.stage("battery"):
        base = CS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
        rep["base_mean_nll"] = base["mean_nll"]
        base_ppl = base["ppl"]
        hashes, n_stream = CS.load_gate0_items(wd, "vqav2", "disc", n=a.n_vqa)
        from infra.pg.projector_ablation import build_tasks
        tasks = {t.name: t for t in build_tasks()}
        vqa_rows = CS.rows_for_banked(tasks["vqav2"], hashes, n_stream)
        base_vqa = IP.eval_pg1(model, proc, tasks["vqav2"], vqa_rows, dev)["score"]
        rep["base_vqa"] = base_vqa
        cells = {}
        for name in FAMILIES:
            for mult in AMP_MULTS:
                seeds = range(3) if (name in MULTIDRAW and mult == 1.0) else [0]
                dmgs, dvqas = [], []
                for s in seeds:
                    # BC.hook_at is itself a @contextmanager (fresh one per `with`)
                    with BC.hook_at(layers, 0, make_hook(name, mult, s)):
                        pg = CS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
                    dmgs.append(CS.safe_log1p_dppl(pg["mean_nll"], base_ppl))
                    if mult == 1.0:
                        with BC.hook_at(layers, 0, make_hook(name, mult, s)):
                            vq = IP.eval_pg1(model, proc, tasks["vqav2"], vqa_rows, dev)
                        dvqas.append(vq["score"] - base_vqa)
                cells[f"{name}@{mult}"] = {
                    "family": name, "amp_mult": mult,
                    "log1p_dppl": float(np.mean(dmgs)), "per_draw": dmgs,
                    "dvqa": (float(np.mean(dvqas)) if dvqas else None),
                    "saturated": bool(np.mean(dmgs) >= 30.0)}
                print(f"[carry-e] {name}@{mult}: dppl={np.mean(dmgs):+.3f}", flush=True)
                rep["cells"] = cells
                _w()
    _w()

    # ---- adjudicate ---------------------------------------------------------------------
    with tm.stage("adjudicate"):
        labels = list(FAMILIES)
        meas = [cells[f"{n}@1.0"]["log1p_dppl"] for n in labels]
        predv = [rep["predictions"]["q_pred"][n] for n in labels]
        rep["re"] = CC.adjudicate_re(pred=predv, meas=meas, labels=labels)
        rep["lp_exponents"] = {}
        for n in labels:
            amps = [m * HEADLINE for m in AMP_MULTS]
            dmg = [cells[f"{n}@{m}"]["log1p_dppl"] for m in AMP_MULTS]
            rep["lp_exponents"][n] = CC.lp_growth_exponent(amps, dmg)
        raw = cells["quant@1.0"]["log1p_dppl"]
        incoh = cells["quant_incoherent@1.0"]["log1p_dppl"]
        rep["quarot"] = {"raw": raw, "incoherent": incoh, "reduction": raw - incoh}
    _w()

    CS.assert_gpu_seconds(rep)
    print("CARRY_E_OK", json.dumps({"verdict": rep["re"]["verdict"], "rho": rep["re"]["rho"],
                                    "quarot_reduction": rep["quarot"]["reduction"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
