"""CARRY Arm B2 (GPU, protected curiosity) — the κ-bridge + one targeted-witness search
(spec §3.1 Arm B2, §11 P14/P15). Spends the curiosity budget on the two instruments the scan
handed us for free (S2's graded κ-family; a named-behaviour install at matched norm).

  kappa_bridge   damage vs κ from pure noise (κ=0) through the pure witness (κ=∞), tracing
                 the S2 graded path E[cosθ] = κ/√(D+κ²) between typical and worst case.
  targeted       install "answer yes" on VQA at matched per-token energy by coherent ascent
                 (the specter_h loop shape, minimizing the teacher-forced CE of "yes");
                 measure the install rate vs base and matched-norm random directions.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_b2 --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/carry_b2 --budget-min 300
"""
import argparse
import contextlib
import json
import math
import os
import pathlib

N_STEPS = 150
N_RESTARTS = 2
N_ASCENT_ITEMS = 24


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import carry_core as CC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC
    from infra.pg.projector_ablation import build_tasks

    freeze = CS.assert_frozen("b2")
    CS.install_pg()
    rep = {"object": "carry_b2_v1", "freeze": freeze}
    tm = CS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = pathlib.Path("/tmp/carry_b2")
    wd.mkdir(parents=True, exist_ok=True)
    g0 = json.loads(CS.fetch(f"{CS.CARRY_GATE0}/gate0.json", wd / "gate0.json").read_text())
    assert g0.get("green"), "carry gate0 not green"
    anat = CS.load_roster(str(wd))
    sg0 = json.loads(CS.fetch(f"{CS.SPECTER_GATE0}/gate0.json", wd / "specter_g0.json").read_text())
    rms = sg0["identity"]["site_rms"]
    D = IP.D
    eps_abs = 0.1 * rms * math.sqrt(D)                 # the specter_h coherent@0.1 convention

    with tm.stage("load"):
        model, proc = IP.load_pg1(sha=anat["base_sha"])
        layers = IP.decoder_layers(model)
        dev = "cuda"
        wit = torch.load(CS.fetch(f"{CS.SPECTER_H}/witness_directions.pt",
                                  wd / "witness.pt"), map_location="cpu", weights_only=False)
        w = wit["coherent@0.1"].float().reshape(-1)
        install, holder = SC.make_residual_injector(layers, 0)
        items = [CS.build_ppl_item(model, proc, dev, wdict, install, holder)
                 for wdict in CS.wikitext_windows(proc, 12)]
        base = CS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
        rep["base_mean_nll"] = base["mean_nll"]
        base_ppl = base["ppl"]

    def coherent_ctx(vec):
        v = SC.project_coherent(vec.to(dev), eps_abs)

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

    # ---- κ-bridge ----------------------------------------------------------------------
    with tm.stage("kappa_bridge"):
        kappas = [u * math.sqrt(D) for u in CC.KAPPA_GRID_UNITS] + [None]
        bridge = []
        for kappa in kappas:
            draws = [None] if kappa is None else range(3)
            dmgs = []
            for i in draws:
                v = CC.kappa_draw(w, kappa, D, seed=5000 + (i or 0))
                with coherent_ctx(v)():
                    pg = CS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
                dmgs.append(CS.safe_log1p_dppl(pg["mean_nll"], base_ppl))
            bridge.append({"kappa": kappa,
                           "expected_cos": (1.0 if kappa is None
                                            else CC.expected_cos(kappa, D)),
                           "log1p_dppl": float(np.mean(dmgs)), "per_draw": dmgs})
            print(f"[carry-b2] kappa={kappa}: dppl={np.mean(dmgs):+.3f}", flush=True)
            rep["kappa_bridge"] = bridge
            _w()
    _w()

    # ---- targeted witness: install "answer yes" on VQA ---------------------------------
    with tm.stage("targeted"):
        tasks = {t.name: t for t in build_tasks()}
        hashes, n_stream = CS.load_gate0_items(wd, "vqav2", "disc", n=200)
        vqa_rows = CS.rows_for_banked(tasks["vqav2"], hashes, n_stream)
        yes_items = [_yes_item(model, proc, dev, r, tasks["vqav2"], install, holder)
                     for r in vqa_rows[:N_ASCENT_ITEMS]]

        def item_ce(it, dfn):
            holder["delta"] = dfn(it["T"])
            logits = it["forward_fn"]()[0]
            lab = it["labels"].to(logits.device)
            live = lab != -100
            lse = torch.logsumexp(logits[live].float(), dim=-1)
            nll = lse - logits[live].float().gather(1, lab[live].unsqueeze(1)).squeeze(1)
            return (nll * it["loss_w"][live]).sum()

        best = {"ce": 1e9, "theta": None}
        for r in range(N_RESTARTS):
            gen = torch.Generator(device="cpu").manual_seed(1900 + r)
            theta = torch.randn(D, generator=gen).to(dev).requires_grad_(True)
            opt = torch.optim.Adam([theta], lr=0.05 * rms)
            down, prev = 0, None
            for step in range(N_STEPS):
                opt.zero_grad(set_to_none=True)

                def dfn(T, th=theta):
                    return SC.project_coherent(th, eps_abs).unsqueeze(0).expand(T, D)
                mb = [yes_items[(step * 8 + j) % len(yes_items)] for j in range(8)]
                acc = 0.0
                for it in mb:
                    l_i = item_ce(it, dfn) / len(mb)
                    l_i.backward()
                    acc += float(l_i)
                opt.step()
                if prev is not None and acc > prev:
                    down += 1
                    if down >= 3:
                        for gp in opt.param_groups:
                            gp["lr"] *= 0.5
                        down = 0
                else:
                    down = 0
                prev = acc
            with torch.no_grad():
                fce = sum(float(item_ce(it, lambda T, th=theta:
                                        SC.project_coherent(th, eps_abs).unsqueeze(0).expand(T, D)))
                          for it in yes_items) / len(yes_items)
            if fce < best["ce"]:
                best = {"ce": fce, "theta": theta.detach().clone()}
            print(f"[carry-b2] targeted restart {r}: yes-CE={fce:.4f}", flush=True)

        found_dir = SC.project_coherent(best["theta"], eps_abs)

        def dir_ctx(vec):
            @contextlib.contextmanager
            def ctx():
                def hook(_m, args, kwargs):
                    from infra.pg.inherit_core import _split_hidden
                    hs, rebuild = _split_hidden(args, kwargs)
                    return rebuild((hs.float() + vec.to(dev).unsqueeze(0)).to(hs.dtype))
                h = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
                yield
                h.remove()
            return ctx
        base_rate = _yes_fraction(model, proc, tasks["vqav2"], vqa_rows, dev, None)
        found_rate = _yes_fraction(model, proc, tasks["vqav2"], vqa_rows, dev, dir_ctx(found_dir))
        null_rates = []
        for i in range(3):
            gen = torch.Generator(device="cpu").manual_seed(7000 + i)
            rv = torch.randn(D, generator=gen)
            rvd = SC.project_coherent(rv.to(dev), eps_abs)
            null_rates.append(_yes_fraction(model, proc, tasks["vqav2"], vqa_rows, dev,
                                            dir_ctx(rvd)))
        with dir_ctx(found_dir)():
            coll = CS.safe_ppl_gate(model, proc, dev, n_seq=a.n_ppl_seq)
        rep["targeted"] = {
            "yes_ce_after": best["ce"], "install_rate": found_rate,
            "base_rate": base_rate, "null_rates": null_rates,
            "collateral_log1p_dppl": CS.safe_log1p_dppl(coll["mean_nll"], base_ppl),
            "eps_abs": eps_abs}
        print(f"[carry-b2] install_rate={found_rate:.3f} base={base_rate:.3f} "
              f"null={np.mean(null_rates):.3f}", flush=True)
    _w()

    CS.assert_gpu_seconds(rep)
    kmax = max(b["log1p_dppl"] for b in bridge if b["kappa"] is not None)
    print("CARRY_B2_OK", json.dumps({"max_kappa_damage": kmax,
                                     "witness_damage": bridge[-1]["log1p_dppl"],
                                     "install_rate": rep["targeted"]["install_rate"]}))
    rep["green"] = True
    _w()


def _yes_item(model, proc, dev, row, task, install, holder):
    """A teacher-forced 'yes' item (specter_shared.build_vqa_item shape, ans fixed to 'yes')."""
    import torch
    img = task.image_fn(row).convert("RGB")
    prompt = task.prompt_fn(row)
    inp = proc(text=[prompt], images=[img], suffix=["yes"], return_tensors="pt",
               padding=False).to(dev)
    inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
    labels = inp["labels"][0].cpu()
    T = int(inp["input_ids"].shape[1])
    loss_w = torch.zeros(T, dtype=torch.float32)
    shift = torch.full((T,), -100, dtype=torch.long)
    shift[:-1] = labels[1:]
    lw = shift != -100
    loss_w[lw] = 1.0 / int(lw.sum())

    def forward_fn():
        with install():
            return model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         pixel_values=inp["pixel_values"]).logits
    return {"forward_fn": forward_fn, "holder": holder, "loss_w": loss_w.to(dev),
            "labels": shift, "T": T}


def _yes_fraction(model, proc, task, rows, dev, ctx, batch_size=16):
    """Fraction of greedy-decoded predictions that are 'yes' (case/strip-normalized)."""
    import contextlib as _c

    import torch
    ctx = ctx() if ctx is not None else _c.nullcontext()
    n_yes, n = 0, 0
    with ctx:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images = [task.image_fn(r).convert("RGB") for r in batch]
            prompts = [task.prompt_fn(r) for r in batch]
            inp = proc(text=prompts, images=images, return_tensors="pt",
                       padding="longest").to(dev)
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            in_len = inp["input_ids"].shape[-1]
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=task.max_new_tokens, do_sample=False)
            dec = proc.batch_decode(gen[:, in_len:], skip_special_tokens=True)
            for pred in dec:
                n += 1
                if pred.strip().lower() == "yes":
                    n_yes += 1
    return n_yes / max(n, 1)


if __name__ == "__main__":
    main()
