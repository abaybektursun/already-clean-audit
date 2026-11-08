"""TENANT Wing-P cell — ONE {task, arm, seed, rank} placement run on the PG1 VLM (spec §2
P-grid). Marries PRAGNANZ M1's model/data/training path with M5's confined-adapter machinery,
evaluated by the PLENUM forced-choice readout (chance 0.25 by construction).

Arms (input-side k=32 subspace on the 18 q_proj adapter B-factors, mode='into', B init = 0):
  incore        the inherited consensus core (banked ERLANGEN basis)
  leverage      leverage-matched random (tenant_core sampler; the confound-killer, P1)
  random        Haar k=32 (fresh draw per run — the PZ-D5 lesson)
  topsigma      top-32 right-singular vectors of the frozen q_proj weight (PiSSA-family twin)
  otherlayer    the core of layer (i+9) mod 18, transported naively (specificity control)
  unconstrained no projector (the practitioner baseline)

Per run: trajectory fc every EVAL_EVERY steps (TRAJ_N conf items) · endpoint fc (END_N conf
items) · cross-task battery (BAT_N conf items of the other two tasks) · wikitext-path ppl
census · P3 confinement per checkpoint · Q5 rider (captured in-core signal share, streaming).

    python -m infra.aws_pg.jobs.tenant_p --out <dir> --task cococap --arm incore \
        --seed 1337 [--rank 512] [--steps 3000] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time

BASIS_S3 = "s3://pg-lineage-769063704696/results/erlangen_m_textvqa/out/erlangen_core_basis.npz"
CHOICE_SEED = 20260728
FC_NAME = {"textvqa": "textvqa", "vqav2": "vqav2", "cococap": "coco_captions"}
ALL_TASKS = ["textvqa", "cococap", "vqav2"]
TRAJ_N, END_N, BAT_N = 100, 400, 200
EVAL_EVERY = 250
K_SUB = 32
LR, WARMUP, BATCH = 2e-5, 0.03, 8
TRAIN_POOL = 8192


def sub_seed(seed, task, arm, idx):
    return int(hashlib.sha1(f"{seed}:{task}:{arm}:{idx}".encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=ALL_TASKS, required=True)
    ap.add_argument("--arm", choices=["incore", "leverage", "random", "topsigma",
                                      "otherlayer", "unconstrained"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
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
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC_fc
    from infra.pg import pragnanz_core as PC
    from infra.pg import relearn as R
    from infra.pg import tenant_core as TC
    from infra.pg.lacuna_shell import canon

    assert torch.cuda.is_available()
    device = "cuda"
    n_steps = 30 if a.smoke else a.steps
    traj_n, end_n, bat_n = (12, 24, 12) if a.smoke else (TRAJ_N, END_N, BAT_N)
    eval_every = 10 if a.smoke else EVAL_EVERY
    cell_id = f"{a.task}_{a.arm}_r{a.rank}_s{a.seed}"
    rep = {"cell_id": cell_id, "task": a.task, "arm": a.arm, "seed": a.seed, "rank": a.rank,
           "steps": n_steps, "smoke": a.smoke, "k_sub": K_SUB, "trajectory": [],
           "confinement": [], "torch": torch.__version__}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    # ---- eval items (fc; conf half; frozen order) --------------------------------------------
    tasks_fc = LS.mm_tasks()
    evals = {}
    for t in ALL_TASKS:
        obj = tasks_fc[FC_NAME[t]]
        rows, hashes = IP.load_items(obj, 4000)
        disc, _ = LC.parity_split(hashes)
        c_rows = [r for r, m_ in zip(rows, disc) if not m_]
        n_want = end_n if t == a.task else bat_n
        evals[t] = (obj, c_rows[:max(n_want, traj_n)],
                    PC_fc.choices_for_task(FC_NAME[t], c_rows[:max(n_want, traj_n)],
                                           CHOICE_SEED))
    rep["eval_ns"] = {t: len(evals[t][1]) for t in ALL_TASKS}

    model, proc = LS.load_pg1(device=device)
    lm = model.get_submodule("model.language_model")

    def fc_eval(rows_choices, n):
        # unscorable -> NaN-in-place (plenum contract); stats over scorable, counts banked
        obj, rows, choices = rows_choices
        per = []
        model.eval()
        for r, (cands, gold) in zip(rows[:n], choices[:n]):
            img = obj.image_fn(r).convert("RGB")
            s, _ = PC_fc.score_item_fc(model, proc, device, img,
                                       PC_fc.fc_prompt(obj.prompt_fn(r)), cands)
            if s is None or any(x != x for x in s):
                per.append(float("nan"))
                continue
            per.append(int(int(np.argmax(s)) == gold))
        model.train()
        return per

    # ---- arm bases on the 18 q_proj matrices -------------------------------------------------
    local = out / "core_basis.npz"
    subprocess.run(["aws", "s3", "cp", BASIS_S3, str(local), "--region", "us-east-1"],
                   check=True)
    z = np.load(local)
    qkeys = sorted(k for k in z.files if "q_proj" in k)
    assert len(qkeys) == 18
    qmods = {}
    for n, m in lm.named_modules():
        if n.endswith("self_attn.q_proj"):
            key = canon(f"language_model.{n}.weight").replace(".", "__")
            if key in set(qkeys):
                qmods[key] = m
    assert len(qmods) == 18, f"q_proj module map incomplete ({len(qmods)})"

    def arm_basis(i, key):
        Qc = z[key].astype(np.float64)
        if a.arm == "incore":
            return Qc, {}
        if a.arm == "leverage":
            Q, err = TC.leverage_matched_basis(TC.leverage_profile(Qc), K_SUB,
                                               seed=sub_seed(a.seed, a.task, a.arm, i))
            ov = TC.subspace_overlap(Q, Qc)
            return Q, {"lev_err": err, "overlap_core": ov}
        if a.arm == "random":
            rng = np.random.default_rng(sub_seed(a.seed, a.task, a.arm, i))
            return np.linalg.qr(rng.standard_normal((Qc.shape[0], K_SUB)))[0], {}
        if a.arm == "topsigma":
            W = qmods[key].weight.detach().float().cpu().numpy()
            _, _, Vt = np.linalg.svd(W, full_matrices=False)
            return Vt[:K_SUB].T.astype(np.float64), {}
        if a.arm == "otherlayer":
            other = qkeys[(i + 9) % 18]
            return z[other].astype(np.float64), {"src": other}
        raise AssertionError(a.arm)

    info = R.attach_adapters(lm, rank=a.rank, seed=a.seed, alpha=float(a.rank))
    model.to(device)
    rep["adapter"] = {k: info[k] for k in ("n_modules", "trainable_params", "rank", "alpha")}
    amods = {}                                  # post-attach wrappers (qmods holds the frozen
    for n, m in lm.named_modules():             # pre-attach Linears for topsigma weight reads)
        if n.endswith("self_attn.q_proj") and hasattr(m, "B"):
            key = canon(f"language_model.{n}.weight").replace(".", "__")
            if key in set(qkeys):
                amods[key] = m
    assert len(amods) == 18, f"adapter wrapper map incomplete ({len(amods)})"
    pname_by_id = {id(p): n for n, p in lm.named_parameters()}
    bases, prov = {}, {}
    if a.arm != "unconstrained":
        for i, key in enumerate(qkeys):
            Q, meta = arm_basis(i, key)
            m = amods[key]
            assert Q.shape == (m.B.shape[1], K_SUB)
            bases[pname_by_id[id(m.B)]] = torch.from_numpy(np.ascontiguousarray(Q)).float()
            prov[key] = meta
    rep["basis_prov"] = prov
    params_by_name = dict(lm.named_parameters())

    if bases:                                                  # P5 sanity (mode='into')
        pn, Q = next(iter(sorted(bases.items())))
        G_in = torch.randn(a.rank, Q.shape[0]) @ (Q @ Q.T)
        G_out = torch.randn(a.rank, Q.shape[0])
        G_out = G_out - (G_out @ Q) @ Q.T
        assert float((PC.project_into_grad(G_in, Q) - G_in).norm()) <= 1e-4 * float(G_in.norm())
        assert float(PC.project_into_grad(G_out, Q).norm()) <= 1e-4 * float(G_out.norm())
        rep["p5"] = "into-mode sanity OK"
    _w()

    # ---- data + training loop (M1 recipe) ----------------------------------------------------
    tr = ET.items(a.task, "train", 24 * max(1, n_steps // 3) if a.smoke else TRAIN_POOL,
                  n_eval_skip=128)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                            weight_decay=0.0)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=int(WARMUP * n_steps),
                          num_training_steps=n_steps)

    def collate(rows):
        b = proc(text=[r["prompt"] for r in rows], images=[r["image"] for r in rows],
                 suffix=[r["target"] for r in rows], return_tensors="pt", padding="longest")
        b["pixel_values"] = b["pixel_values"].to(torch.bfloat16)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    q5 = {"sum_captured": 0.0, "sum_total": 0.0}               # Q5 rider: in-subspace share
    losses = []

    def checkpoint_eval(step):
        pi = fc_eval(evals[a.task], traj_n)
        row = {"step": step, "acc": float(np.nanmean(pi)), "t_s": round(time.time() - t0)}
        if bases:
            # A-TENANT-1 (pre-freeze amendment): Adam's elementwise preconditioner does NOT
            # preserve subspaces (CLEAVE A-6) — a projected gradient is not a projected update,
            # so the parent anomaly's recipe (grad-confined, P2 bit-exact) LEAKS parameter
            # span by construction. P3 is therefore a CENSUS, not an assert: per-checkpoint
            # leakage is banked and reported; the arms are gradient-subspace-confined.
            leaks = []
            for pn2, Q in bases.items():
                B = params_by_name[pn2].detach().float().cpu().numpy()
                leaks.append(TC.confinement_frac(B, Q.numpy()))
            row["confinement_max"] = float(max(leaks))
            rep["confinement"].append({"step": step, "max_leak": float(max(leaks)),
                                       "median_leak": float(np.median(leaks))})
        rep["trajectory"].append(row)
        _w()
        print(f"[P] {cell_id} step {step}: acc={row['acc']:.4f}", flush=True)

    checkpoint_eval(0)
    model.train()
    for step in range(n_steps):
        rows = [tr[(step * BATCH + j) % len(tr)] for j in range(BATCH)]
        batch = collate(rows)
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        loss.backward()
        if bases:
            with torch.no_grad():
                for pn2, Q in bases.items():
                    g = params_by_name[pn2].grad
                    tot = float(g.float().pow(2).sum())
                    kept = PC.project_into_grad(g, Q.to(g.device))
                    q5["sum_captured"] += float(kept.float().pow(2).sum())
                    q5["sum_total"] += tot
                    g.copy_(kept)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if (step + 1) % eval_every == 0:
            checkpoint_eval(step + 1)

    rep["losses_head_tail"] = losses[:5] + losses[-5:]
    rep["q5_captured_share"] = (q5["sum_captured"] / q5["sum_total"]
                                if q5["sum_total"] > 0 else None)

    # ---- endpoint + battery + ppl census -----------------------------------------------------
    end_pi = fc_eval(evals[a.task], end_n)
    rep["endpoint"] = {"acc": float(np.nanmean(end_pi)), "per_item": end_pi,
                       "n": len(end_pi),
                       "n_unscorable": int(np.isnan(np.asarray(end_pi, float)).sum())}
    rep["battery"] = {}
    for t in ALL_TASKS:
        if t == a.task:
            continue
        pi = fc_eval(evals[t], bat_n)
        rep["battery"][t] = {"acc": float(np.nanmean(pi)), "per_item": pi, "n": len(pi),
                             "n_unscorable": int(np.isnan(np.asarray(pi, float)).sum())}
    model.eval()
    ppl = IP.ppl_gate(model, proc, device)
    rep["ppl"] = float(ppl["ppl"] if isinstance(ppl, dict) else ppl)
    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"TENANT_P_OK {cell_id} end={np.nanmean(end_pi):.4f} ppl={rep['ppl']:.3f}", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # GESTALT D7
