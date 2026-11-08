"""TENANT Wing-X replay — ONE {optimizer, seed} instrumented FULL-FT run on the primary task
(spec §2 X-replay + the journaled per-step fallback for X-paired/X-orbit).

Trains the LM's 2-D weights directly (no adapters — the deadlock is a HOST-weight phenomenon)
for N_STEPS, logging per step:
  - in-core vs matched-random gradient shares, all 18 q_proj matrices
  - RAW in-core step components at the 3 designated layers (lag-autocorr, X-orbit)
  - q/k paired product-space cancellation index at the 3 layers (X-paired)
  - in-core vs matched-random DISPLACEMENT shares every LOG_EVERY steps
SGD arm: 100-step LR pilots over PILOT_LRS first, picks the loss-progress match to the banked
Adam pilot (both matchings reported), then the full run. P4 band-gate on the Adam arm:
in-core gradient ratio > 5x AND displacement ratio < 0.5x (ERLANGEN-scale continuity).

    python -m infra.aws_pg.jobs.tenant_x --out <dir> --optimizer adamw --seed 0 [--smoke]
    python -m infra.aws_pg.jobs.tenant_x --out <dir> --optimizer sgd --seed 0 \
        --adam-pilot-s3 s3://.../tenant_x_adamw_s0/out/metrics.json [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

BASIS_S3 = "s3://pg-lineage-769063704696/results/erlangen_m_textvqa/out/erlangen_core_basis.npz"
TASK = "textvqa"
N_STEPS = 500
LOG_EVERY = 25
LAG_LAYERS = (5, 9, 14)
N_RAND = 3
LR_ADAM = 2e-5
PILOT_LRS = (3e-5, 1e-4, 3e-4, 1e-3)
PILOT_STEPS = 100
BATCH = 4                       # fp32 full-FT memory budget (48GB L40S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--optimizer", choices=["adamw", "sgd", "adamfrozen"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--adam-pilot-s3", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from transformers import get_scheduler

    from infra.pg import erlangen_tasks as ET
    from infra.pg import lacuna_shell as LS
    from infra.pg import tenant_core as TC
    from infra.pg.lacuna_shell import canon

    assert torch.cuda.is_available()
    device = "cuda"
    n_steps = 30 if a.smoke else N_STEPS
    pilot_steps = 10 if a.smoke else PILOT_STEPS
    rep = {"optimizer": a.optimizer, "seed": a.seed, "smoke": a.smoke, "task": TASK,
           "n_steps": n_steps, "grad_shares": [], "disp": [], "qk_index": [],
           "torch": torch.__version__}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    local = out / "core_basis.npz"
    subprocess.run(["aws", "s3", "cp", BASIS_S3, str(local), "--region", "us-east-1"],
                   check=True)
    z = np.load(local)
    qkeys = sorted(k for k in z.files if "q_proj" in k)
    kkeys = sorted(k for k in z.files if "k_proj" in k)
    assert len(qkeys) == 18 and len(kkeys) == 18

    model, proc = LS.load_pg1(device=device)
    model = model.to(torch.float32)     # bf16 steps would swallow 2e-5-scale displacement
    lm = model.get_submodule("model.language_model")
    for p in model.parameters():
        p.requires_grad_(False)
    train_params = {}
    for n, p in lm.named_parameters():
        if p.ndim == 2 and "embed" not in n:
            p.requires_grad_(True)
            train_params[n] = p
    rep["n_trainable"] = sum(int(p.numel()) for p in train_params.values())

    qmods, kmods = {}, {}
    for n, m in lm.named_modules():
        for suffix, store, keys in (("self_attn.q_proj", qmods, qkeys),
                                    ("self_attn.k_proj", kmods, kkeys)):
            if n.endswith(suffix):
                key = canon(f"language_model.{n}.weight").replace(".", "__")
                if key in set(keys):
                    store[key] = m
    assert len(qmods) == 18 and len(kmods) == 18
    layer_of = {k: int(k.split("__")[2]) for k in qkeys}
    lag_keys = [k for k in qkeys if layer_of[k] in LAG_LAYERS]
    Qc = {k: torch.from_numpy(z[k]).float().to(device) for k in qkeys}
    Kc = {k: torch.from_numpy(z[k]).float().to(device) for k in kkeys}
    rng = np.random.default_rng(20260806 + a.seed)
    Qr = {k: [torch.linalg.qr(torch.from_numpy(
        rng.standard_normal(z[k].shape)).float())[0].to(device)
        for _ in range(N_RAND)] for k in qkeys}
    W0 = {k: qmods[k].weight.detach().clone() for k in qkeys}
    W0k = {k: kmods[k].weight.detach().clone() for k in kkeys}
    Wprev = {k: W0[k].clone() for k in lag_keys}
    Wprev_k = {kk: W0k[kk].clone() for kk in kkeys if layer_of[kk.replace("k_proj", "q_proj")]
               in LAG_LAYERS}
    lag_V = {k: [] for k in lag_keys}                          # per-step in-core components
    lag_Vr = {k: [] for k in lag_keys}                         # matched-random reference (F3)

    tr = ET.items(TASK, "train", 512 if a.smoke else 4096, n_eval_skip=128)

    def collate(rows):
        b = proc(text=[r["prompt"] for r in rows], images=[r["image"] for r in rows],
                 suffix=[r["target"] for r in rows], return_tensors="pt", padding="longest")
        b["pixel_values"] = b["pixel_values"].to(torch.float32)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    def make_opt(lr):
        ps = [p for p in model.parameters() if p.requires_grad]
        if a.optimizer in ("adamw", "adamfrozen"):
            return torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
        return torch.optim.SGD(ps, lr=lr, momentum=0.9)

    def run_steps(opt, sched, k_steps, log):
        losses = []
        v_snap = {}                            # adamfrozen: exp_avg_sq frozen at step 10
        g = torch.Generator().manual_seed(a.seed)
        order = torch.randperm(len(tr), generator=g).tolist()
        for step in range(k_steps):
            rows = [tr[order[(step * BATCH + j) % len(tr)]] for j in range(BATCH)]
            batch = collate(rows)
            model.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            loss.backward()
            if log:
                with torch.no_grad():
                    shares = {}
                    for k in qkeys:
                        gw = qmods[k].weight.grad.float()
                        tot = float(gw.pow(2).sum())
                        core = float((gw @ Qc[k]).pow(2).sum())
                        rand = [float((gw @ B).pow(2).sum()) for B in Qr[k]]
                        shares[k] = {"tot": tot, "core": core, "rand": rand}
                    rep["grad_shares"].append({"step": step, "shares": shares})
            if a.optimizer == "adamfrozen" and v_snap:
                with torch.no_grad():
                    for pp, vs in v_snap.items():
                        opt.state[pp]["exp_avg_sq"].copy_(vs.to(pp.device))
            opt.step()
            if a.optimizer == "adamfrozen" and step == 10 and not v_snap:
                with torch.no_grad():
                    for pp in opt.param_groups[0]["params"]:
                        v_snap[pp] = opt.state[pp]["exp_avg_sq"].detach().cpu().clone()
            if sched is not None:
                sched.step()
            losses.append(float(loss.item()))
            if log:
                with torch.no_grad():
                    qk_row = {"step": step}
                    for k in lag_keys:
                        dW = (qmods[k].weight.detach() - Wprev[k]).float()
                        lag_V[k].append((dW @ Qc[k]).reshape(-1).cpu().numpy()
                                        .astype(np.float32))
                        lag_Vr[k].append((dW @ Qr[k][0]).reshape(-1).cpu().numpy()
                                        .astype(np.float32))
                        kk = k.replace("q_proj", "k_proj")
                        dWk = (kmods[kk].weight.detach() - Wprev_k[kk]).float()
                        # head-sliced product (instrument contract): 8 q-heads x 256 vs the
                        # ONE shared MQA k-head; index per head, mean+min logged
                        hd = dWk.shape[0]
                        dWq_np, dWk_np = dW.cpu().numpy(), dWk.cpu().numpy()
                        Wq0_np, Wk0_np = W0[k].float().cpu().numpy(), \
                            W0k[kk].float().cpu().numpy()
                        Qq_np, Qk_np = Qc[k].cpu().numpy(), Kc[kk].cpu().numpy()
                        idxs, nqs, nks = [], [], []
                        for h in range(dWq_np.shape[0] // hd):
                            sl = slice(h * hd, (h + 1) * hd)
                            r = TC.qk_product_cancellation(
                                dWq_np[sl], dWk_np, Wq0_np[sl], Wk0_np, Qq_np, Qk_np)
                            if r["index"] == r["index"]:
                                idxs.append(r["index"])
                                nqs.append(r["norm_q"])
                                nks.append(r["norm_k"])
                        # warmup step 0 has lr=0 -> zero dW -> every head index NaN; logged
                        # as None rather than crashing (the analysis skips None rows)
                        qk_row[k] = ({"mean": float(np.mean(idxs)), "min": float(np.min(idxs)),
                                      "n_heads": len(idxs),
                                      "norm_q": float(np.mean(nqs)), "norm_k": float(np.mean(nks))}
                                     if idxs else {"mean": None, "min": None, "n_heads": 0})
                        Wprev[k].copy_(qmods[k].weight.detach())
                        Wprev_k[kk].copy_(kmods[kk].weight.detach())
                    rep["qk_index"].append(qk_row)
                if (step + 1) % LOG_EVERY == 0 or step == k_steps - 1:
                    with torch.no_grad():
                        disp = {}
                        for k in qkeys:
                            D = (qmods[k].weight.detach() - W0[k]).float()
                            tot = float(D.pow(2).sum())
                            disp[k] = {"tot": tot,
                                       "core": float((D @ Qc[k]).pow(2).sum()),
                                       "rand": [float((D @ B).pow(2).sum()) for B in Qr[k]]}
                        rep["disp"].append({"step": step + 1, "disp": disp})
                    rep["losses_so_far"] = losses[-5:]
                    _w()
                    print(f"[X] {a.optimizer} s{a.seed} step {step + 1} "
                          f"loss={losses[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return losses

    if a.optimizer == "sgd":
        assert a.adam_pilot_s3, "sgd arm needs --adam-pilot-s3 (loss-progress matching)"
        if a.adam_pilot_s3.startswith("s3://"):
            ap_local = out / "adam_pilot.json"
            subprocess.run(["aws", "s3", "cp", a.adam_pilot_s3, str(ap_local),
                            "--region", "us-east-1"], check=True)
        else:                                   # same-pipeline local path (steps share disk)
            ap_local = pathlib.Path(a.adam_pilot_s3)
        adam_ref = json.loads(ap_local.read_text())["pilot_loss_at_end"]
        pilots = {}
        for lr in PILOT_LRS:
            sd0 = {k: v.detach().cpu().clone() for k, v in train_params.items()}
            losses = run_steps(make_opt(lr), None, pilot_steps, log=False)
            pilots[str(lr)] = float(np.mean(losses[-5:]))
            with torch.no_grad():
                for k, v in train_params.items():
                    v.copy_(sd0[k].to(device))
            del sd0
            torch.cuda.empty_cache()
            print(f"[X] sgd pilot lr={lr}: loss_end={pilots[str(lr)]:.4f} "
                  f"(adam ref {adam_ref:.4f})", flush=True)
        lr_pick = float(min(PILOT_LRS, key=lambda lr: abs(pilots[str(lr)] - adam_ref)))
        rep["sgd_pilots"] = {"grid": pilots, "adam_ref": adam_ref, "picked": lr_pick}
        assert abs(pilots[str(lr_pick)] - adam_ref) <= 0.5 * max(adam_ref, 0.2), \
            f"SGD sanity FAIL: no pilot lr matches Adam loss progress ({pilots} vs {adam_ref})"
        lr = lr_pick
    else:
        lr = LR_ADAM

    opt = make_opt(lr)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=int(0.03 * n_steps),
                          num_training_steps=n_steps)
    losses = run_steps(opt, sched, n_steps, log=True)
    rep["lr"] = lr
    rep["final_loss_mean5"] = float(np.mean(losses[-5:]))
    if a.optimizer == "sgd":
        g_end = rep["grad_shares"][-1]["shares"]
        g_start = rep["grad_shares"][0]["shares"]
        tot_end = sum(v["tot"] for v in g_end.values())
        tot_start = sum(v["tot"] for v in g_start.values())
        assert tot_end >= 1e-4 * tot_start, \
            f"SGD sanity FAIL: gradients vanished (end/start {tot_end/tot_start:.2e})"
    if a.optimizer == "adamw":
        rep["pilot_loss_at_end"] = float(np.mean(losses[:pilot_steps][-5:]))

    # ---- lag-autocorr + summary ratios -------------------------------------------------------
    rep["lag_autocorr"] = {k: TC.lag_autocorr(np.stack(lag_V[k]).astype(np.float64), 5)
                           for k in lag_keys}
    rep["lag_autocorr_rand"] = {k: TC.lag_autocorr(np.stack(lag_Vr[k]).astype(np.float64), 5)
                                for k in lag_keys}
    gsh = rep["grad_shares"]
    core_ratio = float(np.median([r["shares"][k]["core"] / (np.mean(r["shares"][k]["rand"]) + 1e-30)
                                  for r in gsh for k in qkeys]))
    d_last = rep["disp"][-1]["disp"]
    disp_ratio = float(np.median([d_last[k]["core"] / (np.mean(d_last[k]["rand"]) + 1e-30)
                                  for k in qkeys]))
    rep["summary"] = {"grad_core_over_rand_median": core_ratio,
                      "disp_core_over_rand_median": disp_ratio}
    if a.optimizer == "adamw" and not a.smoke:
        p4 = core_ratio > 5.0 and disp_ratio < 0.5
        rep["p4_band_gate"] = {"grad_ratio": core_ratio, "disp_ratio": disp_ratio, "pass": p4}
        assert p4, f"P4 FAIL: Adam arm outside the ERLANGEN-scale band ({core_ratio:.2f}, {disp_ratio:.2f})"
    np.savez(out / "lag_bank.npz", **{k.replace("__", "."): np.stack(lag_V[k])
                                      for k in lag_keys})
    np.savez(out / "lag_bank_rand.npz", **{k.replace("__", "."): np.stack(lag_Vr[k])
                                           for k in lag_keys})
    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"TENANT_X_OK {a.optimizer} s{a.seed} grad_ratio={core_ratio:.2f} "
          f"disp_ratio={disp_ratio:.3f}", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # GESTALT D7
