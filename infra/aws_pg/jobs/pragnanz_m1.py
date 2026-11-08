"""PRAGNANZ M1/M2/M3 — one instrumented 500-step run per destination (the spec's priced M1
population; banked erlangen_m logs lack per-batch vectors). q_proj-scoped per the §0.2 branch.

Per training step (frozen erlangen_m recipe: PG1 + rank-512 adapters, AdamW 2e-5, batch 8):
  1. normal training step (adapters only);
  2. PROBE backward (all LM 2-D base weights requires_grad):
     - M3: per-matrix ||G||^2 scalars (ALL matrices — the census denominators);
     - M1: per-q_proj-matrix in-core coefficients C = G @ Q_core and G @ Q_rand[j] (3 draws,
       per-matrix seeds) accumulated STREAMING (sum_c, sum_norm2, per 100-step segments) —
       mean2/var is exact from the accumulators, no per-step banking;
  3. M2: four quarter-batch backwards (q_proj grads only) -> DomeEstimator updates (flattened,
     k=32) on three representative q_proj matrices (L5/L9/L14); end-of-run: per-DOME-direction
     core-subspace projection fractions vs a sampled rotation null (N1 width printed).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.pragnanz_m1 \
      --args "--dst textvqa --basis-s3 s3://pg-lineage-769063704696/results/erlangen_m_textvqa/out/erlangen_core_basis.npz" \
      --type g6e.2xlarge,g6e.4xlarge,g6e.8xlarge,p5.4xlarge --s3-prefix results/pragnanz_m1_textvqa --budget-min 300

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

N_STEPS = 500
SEG = 100
N_RAND = 3
DOME_LAYERS = (5, 9, 14)
DOME_K = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--basis-s3", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--smoke", action="store_true", help="20 steps")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    import numpy as np
    import torch
    from transformers import get_scheduler

    from infra.pg import erlangen_tasks as ET
    from infra.pg import pragnanz_core as PC
    from infra.pg import relearn as R
    from infra.pg.lacuna_shell import canon, load_pg1, pg1_baseline_sd

    assert torch.cuda.is_available()
    device = "cuda"
    t0 = time.time()
    n_steps = 20 if a.smoke else a.n_steps
    rep = {"object": "PRAGNANZ M1/M2/M3 instrumented run", "dst": a.dst,
           "recipe": {"rank": a.rank, "lr": a.lr, "batch": a.batch, "optimizer": "adamw",
                      "scheduler": "cosine", "n_steps": n_steps},
           "basis": a.basis_s3, "n_rand": N_RAND, "dome": {"layers": list(DOME_LAYERS),
                                                           "k": DOME_K}}

    def bank():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    bank()

    model, proc = load_pg1(device=device)
    pg1_sd = pg1_baseline_sd(model)
    lm = model.get_submodule("model.language_model")
    tr = ET.items(a.dst, "train", 1024, n_eval_skip=128)

    # ---- q_proj core bases (the §0.2 branch scope) + matched-random draws -------------------
    local = out / "core_basis.npz"
    subprocess.run(["aws", "s3", "cp", a.basis_s3, str(local), "--region", "us-east-1"],
                   check=True)
    z = np.load(local)
    rng = np.random.default_rng(20260805)
    basis, rands = {}, {}
    for k in z.files:
        if "q_proj" not in k:
            continue                                   # q_proj scope, pre-registered
        U = torch.from_numpy(z[k]).float().to(device)
        basis[k] = U
        rands[k] = [torch.linalg.qr(torch.from_numpy(
            rng.standard_normal(tuple(U.shape))).float())[0].to(device)
            for _ in range(N_RAND)]
    assert basis, "no q_proj bases in the core file"
    rep["n_qproj"] = len(basis)
    r_core = int(next(iter(basis.values())).shape[1])
    rep["r_core"] = r_core
    bank()

    # name maps: canonical basis key -> live module
    qmods = {}
    for n, m in lm.named_modules():
        if n.endswith("self_attn.q_proj"):
            key = canon(f"language_model.{n}.weight").replace(".", "__")
            if key in basis:
                qmods[key] = m
    assert len(qmods) == len(basis), f"basis/module mismatch {len(qmods)} vs {len(basis)}"
    all2d = {n: p for n, p in lm.named_parameters()
             if p.ndim == 2 and "embed" not in n}
    dome_keys = [k for k in basis if any(f"__{i}__" in k or f"layers__{i}__" in k
                                         for i in DOME_LAYERS)]
    if len(dome_keys) != len(DOME_LAYERS):
        dome_keys = sorted(basis)[:len(DOME_LAYERS)]   # fallback: first three, recorded
    rep["dome_keys"] = dome_keys
    dome = {k: PC.DomeEstimator(int(qmods[k].weight.numel()), DOME_K, seed=9) for k in dome_keys}

    info = R.attach_adapters(lm, rank=a.rank, seed=1337, alpha=float(a.rank))
    lm.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=int(0.03 * n_steps),
                          num_training_steps=n_steps)

    def collate(rows):
        b = proc(text=[r["prompt"] for r in rows], images=[r["image"] for r in rows],
                 suffix=[r["target"] for r in rows], return_tensors="pt", padding="longest")
        b["pixel_values"] = b["pixel_values"].to(torch.bfloat16)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    # streaming accumulators: per (matrix, basis-arm) — sum_c, sum_n2, per segment
    n_seg = max(1, n_steps // SEG)
    cdim = {k: int(qmods[k].weight.shape[0]) * r_core for k in basis}
    acc = {k: {arm: {"sum": torch.zeros(cdim[k], dtype=torch.float64),
                     "n2": [0.0] * n_seg,
                     "sums_seg": torch.zeros(n_seg, cdim[k], dtype=torch.float32)}
               for arm in (["core"] + [f"rand{j}" for j in range(N_RAND)])}
           for k in basis}
    m3_energy = {"qproj_core": [0.0] * n_seg, "total": [0.0] * n_seg}
    losses = []

    for step in range(n_steps):
        rows = [tr[(step * a.batch + j) % len(tr)] for j in range(a.batch)]
        batch = collate(rows)
        # 1. training step
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        seg = min(step // SEG, n_seg - 1)
        # 2. probe backward (base weights)
        for p in all2d.values():
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        tot_e = 0.0
        core_e = 0.0
        for n, p in all2d.items():
            if p.grad is None:
                continue
            g = p.grad.float()
            tot_e += float((g * g).sum())
        for k, m in qmods.items():
            g = m.weight.grad
            assert g is not None, f"no base grad at {k}"
            gf = g.float()
            for arm, Q in ([("core", basis[k])]
                           + [(f"rand{j}", rands[k][j]) for j in range(N_RAND)]):
                c = (gf @ Q).reshape(-1)
                A = acc[k][arm]
                A["sum"] += c.double().cpu()
                A["sums_seg"][seg] += c.float().cpu()
                A["n2"][seg] += float((c * c).sum())
            core_e += float((gf @ basis[k]).pow(2).sum())
        m3_energy["total"][seg] += tot_e
        m3_energy["qproj_core"][seg] += core_e
        # 3. DOME microbatch updates (quarter batches; q_proj grads only)
        for p in all2d.values():
            p.requires_grad_(False)
        for m in qmods.values():
            m.weight.requires_grad_(True)
        micro_g = {k: [] for k in dome_keys}
        q = max(1, a.batch // 4)
        for mb in range(0, a.batch, q):
            mrows = rows[mb:mb + q]
            model.zero_grad(set_to_none=True)
            model(**collate(mrows)).loss.backward()
            for k in dome_keys:
                micro_g[k].append(qmods[k].weight.grad.float().reshape(-1).cpu().clone())
        for k in dome_keys:
            dome[k].update(torch.stack(micro_g[k]))
        for m in qmods.values():
            m.weight.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print(f"[M1] step {step}/{n_steps} loss={losses[-1]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            rep["progress"] = {"step": step, "loss": losses[-1]}
            bank()

    # ---- M1 statistics from accumulators ----------------------------------------------------
    T = n_steps
    m1 = {}
    for k in basis:
        row = {}
        for arm in acc[k]:
            A = acc[k][arm]
            mean = A["sum"].numpy() / T
            mean2 = float((mean * mean).sum())
            var = float(sum(A["n2"]) / T - mean2)
            seg_ratios = []
            for s_ in range(n_seg):
                sm = A["sums_seg"][s_].numpy() / SEG
                m2s = float((sm * sm).sum())
                vs = float(A["n2"][s_] / SEG - m2s)
                seg_ratios.append(m2s / vs if vs > 0 else float("inf"))
            row[arm] = {"mean2": mean2, "var": var,
                        "ratio": mean2 / var if var > 0 else float("inf"),
                        "seg_ratios": seg_ratios}
        rr = [row["core"]["ratio"] / max(row[f"rand{j}"]["ratio"], 1e-30)
              for j in range(N_RAND)]
        row["ratio_of_ratios"] = rr
        m1[k] = row
    rep["m1"] = m1
    # ---- M3 census --------------------------------------------------------------------------
    rep["m3"] = {"deadlocked_fraction_per_seg":
                 [c / t if t > 0 else None
                  for c, t in zip(m3_energy["qproj_core"], m3_energy["total"])],
                 "note": "in-core q_proj energy / TOTAL all-matrix gradient energy"}
    # ---- M2 angles --------------------------------------------------------------------------
    m2 = {}
    for k in dome_keys:
        U = dome[k].U                                   # (numel, DOME_K)
        Q = basis[k].cpu()
        W = qmods[k].weight
        fracs = []
        for j in range(DOME_K):
            u = U[:, j].reshape(W.shape)
            fracs.append(float((u @ Q).norm() / u.norm().clamp(min=1e-30)))
        g_ = torch.Generator().manual_seed(77)
        null = []
        for _ in range(24):
            v = torch.randn(W.numel(), generator=g_)
            vm = v.reshape(W.shape)
            null.append(float((vm @ Q).norm() / v.norm()))
        print(f"[M1][N1] M2 null width {k[-40:]}: [{min(null):.4f}, {max(null):.4f}]",
              flush=True)
        m2[k] = {"dome_core_fracs": fracs, "lam": [float(x) for x in dome[k].Lam],
                 "null": {"lo": min(null), "hi": max(null), "n": 24}}
    rep["m2"] = m2
    rep["losses_head_tail"] = losses[:5] + losses[-5:]
    rep["elapsed_s"] = time.time() - t0
    bank()
    np.savez_compressed(out / "m1_accumulators.npz",
                        **{f"{k}|{arm}|sumseg": acc[k][arm]["sums_seg"].numpy()
                           for k in basis for arm in acc[k]})
    print("PRAGNANZ_M1_OK", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # GESTALT D7
