"""RATION Arm K (GPU) — the labelled KV-cache corroboration (spec §3, NOT in the MDE). Applies the
water-filling quantiser to EVERY layer's k_proj/v_proj outputs (the whole KV cache) and sweeps the
average KV-bit budget, comparing the energy axes (variance / magnitude / uniform) on real behavior —
the high-practical-value long-context-memory use case. k_proj output is PRE-RoPE, so this is
RoPE-aware by construction.

HONEST SCOPE (Principle #10; T7): no KV susceptibility operator is banked, so this arm demonstrates
the allocation MACHINERY reaching the KV setting — it is NOT a susceptibility-vs-energy map-transfer
claim on KV. The weight/KV-space pullback of Q is the named follow-up.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ration_kv --type p5.4xlarge,g6e.8xlarge \
      --s3-prefix results/ration_kv --budget-min 90
"""
import argparse
import json
import os
import pathlib
import time


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import contextlib

    import numpy as np
    import torch

    from infra.aws_pg.jobs import _ration_shared as RS
    from infra.aws_pg.jobs._specter_shared import build_ppl_item
    from infra.pg import inherit_pg1 as IP
    from infra.pg import ration_core as RC

    t00 = time.time()
    rep = {"object": "ration_kv_v1", "scope": "machinery-reach on KV (no banked KV-Q; map-transfer is a follow-up)"}

    def dump():
        (out / "ration_kv.json").write_text(json.dumps(rep, indent=2, default=float))

    def stage(m):
        print(f"[ration-kv] {m} (+{time.time() - t00:.0f}s)", flush=True)

    model, proc, layers, sha, base_ppl = RS.load_model()
    ppl_disc, ppl_conf = RS.wikitext_split(proc)
    task, vqa_disc, vqa_conf = RS.vqa_task_and_split()

    # the KV modules: every layer's k_proj and v_proj (outputs are the cached K/V, pre-RoPE for K)
    modules = []
    for L in range(len(layers)):
        modules.append(layers[L].self_attn.k_proj)
        modules.append(layers[L].self_attn.v_proj)
    widths = [m.weight.shape[0] for m in modules]
    rep["n_kv_modules"] = len(modules)
    rep["kv_width"] = widths[0]
    stage(f"{len(modules)} KV modules, width={widths[0]}")

    # ---- accumulate per-module output Σ over discovery ppl windows (one pass) ----
    st = [{"gram": torch.zeros(w, w, dtype=torch.float64, device="cuda"),
           "sum": torch.zeros(w, dtype=torch.float64, device="cuda"),
           "abs": torch.zeros(w, dtype=torch.float64, device="cuda"), "n": 0} for w in widths]

    def mk(i):
        def hook(_m, _a, o):
            x = o.reshape(-1, o.shape[-1]).detach().double()
            st[i]["gram"] += x.transpose(0, 1) @ x
            st[i]["sum"] += x.sum(0)
            st[i]["abs"] += x.abs().sum(0)
            st[i]["n"] += x.shape[0]
        return hook

    hs = [m.register_forward_hook(mk(i)) for i, m in enumerate(modules)]
    try:
        for wd in ppl_disc:
            item = build_ppl_item(model, proc, "cuda", wd, contextlib.nullcontext, {})
            with torch.no_grad():
                item["forward_fn"]()
    finally:
        for h in hs:
            h.remove()
    stage(f"KV Σ accumulated ({st[0]['n']} rows/module)")

    # per-module banks (variance basis + coord-axis policies)
    module_banks = []
    for i, w in enumerate(widths):
        mean_x, C_x, absmean_x = RC.finalize_sigma(st[i]["gram"].cpu(), st[i]["sum"].cpu(),
                                                   st[i]["abs"].cpu(), st[i]["n"])
        sev, sea = RC._sigma_eig(C_x)
        module_banks.append({"Q_evecs": torch.eye(w, dtype=torch.float64),  # unused for KV policies
                             "Q_evals": np.ones(w), "sigma_evecs": sev, "sigma_evals": sea.numpy(),
                             "mean_x": mean_x, "C_x": C_x, "absmean_x": absmean_x.numpy(), "seed": 1337})

    # ---- clean (passthrough) reference + the readout under the CURRENT spec ----
    sub_i, sub_h = RC.make_proj_substitutor(modules)

    def kv_ppl_per_window():
        pw = []
        with sub_i():
            for wd in ppl_conf:
                item = build_ppl_item(model, proc, "cuda", wd, contextlib.nullcontext, {})
                with torch.no_grad():
                    logits = item["forward_fn"]()[0]
                lab = item["labels"].to(logits.device)
                live = lab != -100
                lse = torch.logsumexp(logits[live].float(), dim=-1)
                nll = lse - logits[live].float().gather(1, lab[live].unsqueeze(1)).squeeze(1)
                pw.append([float(nll.sum()), int(live.sum())])
        return pw

    def kv_vqa():
        return IP.eval_pg1(model, proc, task, vqa_conf[:RS.N_VQA_CELL], "cuda", ctx=sub_i())["per_item"]

    sub_h["spec"] = None                                 # passthrough clean
    g_clean_ppl = RC.ppl_goodness(kv_ppl_per_window())
    g_clean_vqa = [float(x) for x in kv_vqa()]
    rep["clean"] = {"vqa_acc": float(np.mean(g_clean_vqa))}
    stage(f"clean vqa={rep['clean']['vqa_acc']:.4f}")

    # ---- KV-bit sweep: variance / magnitude / uniform ----
    sweep = {}
    for policy in ("variance", "magnitude", "uniform"):
        sweep[policy] = {}
        # precompute per-module (R, centers, stds, w, pinned) once
        bases = [RS.precompute_basis_dev(policy, mb) for mb in module_banks]
        for b in RC.BUDGET_GRID:
            spec = []
            for (R_dev, cen, std, w_imp, pinned) in bases:
                bits = RC.water_fill(w_imp, b, pinned=pinned)
                spec.append({"R": R_dev, "bits": torch.as_tensor(bits).to(R_dev.device),
                             "centers": cen, "stds": std})
            sub_h["spec"] = spec
            g_ppl = RC.ppl_goodness(kv_ppl_per_window())
            vqa = kv_vqa()
            sweep[policy][str(b)] = {"ppl_loss": RC.loss_ci(g_clean_ppl, g_ppl),
                                     "vqa_loss": RC.loss_ci(g_clean_vqa, [float(x) for x in vqa]),
                                     "vqa_acc": float(np.mean(vqa))}
            stage(f"  {policy}@{b}b ppl_loss={sweep[policy][str(b)]['ppl_loss']['E']:+.4f} "
                  f"vqa_loss={sweep[policy][str(b)]['vqa_loss']['E']:+.4f}")
        dump()
    rep["kv_sweep"] = sweep
    rep["wall_seconds"] = round(time.time() - t00, 1)
    dump()
    print("RATION_KV_OK", json.dumps({"policies": list(sweep)}))


if __name__ == "__main__":
    main()
