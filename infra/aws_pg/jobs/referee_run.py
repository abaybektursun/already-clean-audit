"""REFEREE dose-ladder cell — ONE {arm, lambda, seed, task} full-FT run (spec §2).
The TENANT X-replay recipe REUSED verbatim (fp32, LM 2-D weights, AdamW 2e-5 wd 0, cosine,
batch 4, 500 steps, deterministic order), plus for dose arms the §2 update replacement on
the 18 q_proj matrices:  u_total = u_AdamW + P_sub(−η_c·λ·g_t − u_AdamW).

ETA_C is FROZEN with provenance (journal 2026-08-06; tenant_x4/out/adamw_s0). Readouts:
endpoint fc (conf items, NaN-in-place) + battery (the other anchor task) + ppl, each at
step 250 (mid-run, spec §8 slow-damage clause) and endpoint. P4: realized in-subspace /
out-of-subspace displacement banked every 25 steps. P5: final loss banked; the divergence
verdict + retry orchestration live outside the job (--eta-scale 0.5 is the retry knob).

    python -m infra.aws_pg.jobs.referee_run --out <dir> --arm incore --lam 16 --seed 0 \
        [--task textvqa] [--eta-scale 1.0] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time

BASIS_S3 = "s3://pg-lineage-769063704696/results/erlangen_m_textvqa/out/erlangen_core_basis.npz"
ETA_C = 0.029192947184804106          # v2, P3-recalibrated: v1 (0.0086415, banked-log
                                      # arithmetic) delivered 0.30x realized in-sub motion
                                      # on the smoke (raw in-core steps accumulate
                                      # sub-linearly); factor 3.378 measured, journal
CHOICE_SEED = 20260728
FC_NAME = {"textvqa": "textvqa", "vqav2": "vqav2", "cococap": "coco_captions"}
N_STEPS = 500
LOG_EVERY = 25
MID_STEP = 250
END_N, BAT_N = 400, 200
LR, WARMUP, BATCH = 2e-5, 0.03, 4
K_SUB = 32


def sub_seed(seed, arm, idx):
    return int(hashlib.sha1(f"referee:{seed}:{arm}:{idx}".encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["baseline", "incore", "leverage", "haar", "vfreeze"],
                    required=True)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--task", default="textvqa", choices=["textvqa", "cococap", "vqav2"])
    ap.add_argument("--battery-task", default="cococap",
                    choices=["textvqa", "cococap", "vqav2"])
    ap.add_argument("--eta-scale", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert (a.arm in ("baseline", "vfreeze")) == (a.lam == 0.0), \
        "lambda>0 iff dose arm (spec §2; vfreeze is the R1 rider, no dose)"
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
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC_fc
    from infra.pg import tenant_core as TC
    from infra.pg.lacuna_shell import canon

    assert torch.cuda.is_available()
    device = "cuda"
    n_steps = 30 if a.smoke else N_STEPS
    end_n, bat_n = (24, 12) if a.smoke else (END_N, BAT_N)
    eta_c = ETA_C * a.eta_scale
    cell_id = f"{a.task}_{a.arm}_l{a.lam:g}_s{a.seed}"
    rep = {"cell_id": cell_id, "arm": a.arm, "lam": a.lam, "seed": a.seed, "task": a.task,
           "battery_task": a.battery_task, "eta_c": eta_c, "eta_scale": a.eta_scale,
           "n_steps": n_steps, "smoke": a.smoke, "motion_audit": [], "evals": {},
           "torch": torch.__version__}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    # ---- eval items --------------------------------------------------------------------------
    tasks_fc = LS.mm_tasks()
    evals = {}
    for t, n_want in ((a.task, end_n), (a.battery_task, bat_n)):
        obj = tasks_fc[FC_NAME[t]]
        rows, hashes = IP.load_items(obj, 1900 if t == "refcoco" else 4000)
        disc, _ = LC.parity_split(hashes)
        c_rows = [r for r, m_ in zip(rows, disc) if not m_][:n_want]
        evals[t] = (obj, c_rows, PC_fc.choices_for_task(FC_NAME[t], c_rows, CHOICE_SEED))

    model, proc = LS.load_pg1(device=device)
    model = model.to(torch.float32)
    lm = model.get_submodule("model.language_model")
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in lm.named_parameters():
        if p.ndim == 2 and "embed" not in n:
            p.requires_grad_(True)

    def fc_eval(t):
        obj, rows, choices = evals[t]
        per = []
        model.eval()
        for r, (cands, gold) in zip(rows, choices):
            img = obj.image_fn(r).convert("RGB")
            s, _ = PC_fc.score_item_fc(model, proc, device, img,
                                       PC_fc.fc_prompt(obj.prompt_fn(r)), cands)
            if s is None or any(x != x for x in s):
                per.append(float("nan"))
                continue
            per.append(int(int(np.argmax(s)) == gold))
        model.train()
        return per

    def full_evals(tag):
        model.eval()
        ppl = IP.ppl_gate(model, proc, device)
        model.train()
        rep["evals"][tag] = {
            "task_per_item": fc_eval(a.task),
            "battery_per_item": fc_eval(a.battery_task),
            "ppl": float(ppl["ppl"] if isinstance(ppl, dict) else ppl)}
        rep["evals"][tag]["task_acc"] = float(np.nanmean(rep["evals"][tag]["task_per_item"]))
        rep["evals"][tag]["battery_acc"] = float(
            np.nanmean(rep["evals"][tag]["battery_per_item"]))
        _w()
        print(f"[REF] {cell_id} {tag}: task={rep['evals'][tag]['task_acc']:.4f} "
              f"bat={rep['evals'][tag]['battery_acc']:.4f} "
              f"ppl={rep['evals'][tag]['ppl']:.3f}", flush=True)

    # ---- subspaces + hook state --------------------------------------------------------------
    local = out / "core_basis.npz"
    subprocess.run(["aws", "s3", "cp", BASIS_S3, str(local), "--region", "us-east-1"],
                   check=True)
    z = np.load(local)
    qkeys = sorted(k for k in z.files if "q_proj" in k)
    qmods = {}
    for n, m in lm.named_modules():
        if n.endswith("self_attn.q_proj"):
            key = canon(f"language_model.{n}.weight").replace(".", "__")
            if key in set(qkeys):
                qmods[key] = m
    assert len(qmods) == 18
    Qs = {}
    if a.arm == "vfreeze":
        for k in qkeys:
            Qs[k] = torch.from_numpy(np.ascontiguousarray(
                z[k].astype(np.float64))).float().to(device)
    elif a.arm != "baseline":
        for i, k in enumerate(qkeys):
            Qc = z[k].astype(np.float64)
            if a.arm == "incore":
                Q = Qc
            elif a.arm == "leverage":
                Q, err = TC.leverage_matched_basis(TC.leverage_profile(Qc), K_SUB,
                                                   seed=sub_seed(a.seed, a.arm, i))
            else:
                rng = np.random.default_rng(sub_seed(a.seed, a.arm, i))
                Q = np.linalg.qr(rng.standard_normal((Qc.shape[0], K_SUB)))[0]
            Qs[k] = torch.from_numpy(np.ascontiguousarray(Q)).float().to(device)
    W0 = {k: qmods[k].weight.detach().clone() for k in qkeys}

    tr = ET.items(a.task, "train", 512 if a.smoke else 4096, n_eval_skip=128)

    def collate(rows):
        b = proc(text=[r["prompt"] for r in rows], images=[r["image"] for r in rows],
                 suffix=[r["target"] for r in rows], return_tensors="pt", padding="longest")
        b["pixel_values"] = b["pixel_values"].to(torch.float32)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                            weight_decay=0.0)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=int(WARMUP * n_steps),
                          num_training_steps=n_steps)
    losses = []
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(len(tr), generator=g).tolist()
    model.train()
    for step in range(n_steps):
        rows = [tr[order[(step * BATCH + j) % len(tr)]] for j in range(BATCH)]
        batch = collate(rows)
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        loss.backward()
        if Qs:
            Wpre = {k: qmods[k].weight.detach().clone() for k in qkeys}
        opt.step()
        if a.arm == "vfreeze" and step == 100:
            v_snap = {k: opt.state[qmods[k].weight]["exp_avg_sq"].detach().clone()
                      for k in qkeys}
            rep["vfreeze_snapshot_step"] = 101
        if Qs:
            with torch.no_grad():
                for k in qkeys:
                    W = qmods[k].weight
                    u = W.detach() - Wpre[k]
                    Q = Qs[k]
                    if a.arm == "vfreeze":
                        if step <= 100:
                            continue                     # standard Adam until the snapshot
                        st = opt.state[W]
                        t_ = st["step"] if isinstance(st["step"], (int, float)) \
                            else float(st["step"].item())
                        mh = st["exp_avg"].float() / (1 - 0.9 ** t_)
                        vh = v_snap[k].float() / (1 - 0.999 ** t_)
                        lr_t = opt.param_groups[0]["lr"]
                        u_alt = -lr_t * mh / (vh.sqrt() + 1e-8)
                        corr = (u_alt - u.float()) @ Q @ Q.T
                    else:
                        gr = W.grad.float()
                        corr = ((-eta_c * a.lam) * gr - u.float()) @ Q @ Q.T
                    W.add_(corr.to(W.dtype))
        sched.step()
        losses.append(float(loss.item()))
        if (step + 1) % LOG_EVERY == 0 or step == n_steps - 1:
            with torch.no_grad():
                audit = {}
                for k in qkeys:
                    D = (qmods[k].weight.detach() - W0[k]).float()
                    row = {"tot": float(D.pow(2).sum())}
                    if Qs:
                        row["in_sub"] = float((D @ Qs[k]).pow(2).sum())
                    Qc_t = torch.from_numpy(z[k].astype(np.float32)).to(device)
                    row["in_core"] = float((D @ Qc_t).pow(2).sum())
                    audit[k] = row
                rep["motion_audit"].append({"step": step + 1, "audit": audit})
            rep["losses_tail"] = losses[-5:]
            _w()
            print(f"[REF] {cell_id} step {step + 1} loss={losses[-1]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step + 1 == (10 if a.smoke else MID_STEP):
            full_evals("mid")

    rep["final_loss_mean5"] = float(np.mean(losses[-5:]))
    full_evals("end")
    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"REFEREE_RUN_OK {cell_id} final_loss={rep['final_loss_mean5']:.4f}", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # GESTALT D7
