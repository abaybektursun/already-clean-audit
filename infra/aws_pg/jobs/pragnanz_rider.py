"""PRAGNANZ rider (spec §3, EXPLORATORY census — run because budget remained): do items whose
in-core gradient signs oppose occupy different splat components at site 9? Connects the wings:
if signs are coherent within a component and oppose across components, the deadlock's
cancellation would live BETWEEN components — structure M1's run-level mean cannot see.
Census only: Cramér's V + permutation p; no confirmatory weight, no family membership.

    python -m infra.aws_pg.jobs.pragnanz_rider --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

BASIS_S3 = "s3://pg-lineage-769063704696/results/erlangen_m_textvqa/out/erlangen_core_basis.npz"
ATLAS_S3 = "s3://pg-lineage-769063704696/results/gestalt_arm_d/out/atlas_pg1/site9/full_s0.npz"
SITE = 9
N_ITEMS = 256
N_PERM = 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import os
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch

    from infra.pg import erlangen_tasks as ET
    from infra.pg import gestalt_splat as GS
    from infra.pg.lacuna_shell import canon, load_pg1

    assert torch.cuda.is_available()
    device = "cuda"
    n_items = 16 if a.smoke else N_ITEMS
    rep = {"object": "rider deadlock-boundary census", "site": SITE, "n_items": n_items,
           "smoke": a.smoke}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    for rel, dst in ((BASIS_S3, "basis.npz"), (ATLAS_S3, "atlas.npz")):
        subprocess.run(["aws", "s3", "cp", rel, str(out / dst), "--region", "us-east-1"],
                       check=True)
    z = np.load(out / "basis.npz")
    key = next(k for k in z.files if "q_proj" in k and "__9__" in k)
    Q = torch.from_numpy(z[key]).float().to(device)                      # (2048, 32) IN axis
    az = np.load(out / "atlas.npz")
    atlas = {k: torch.from_numpy(az[k]).to(device)
             for k in ("log_pi", "mu", "U", "lam", "sigma2")}

    model, proc = load_pg1(device=device)
    lm = model.get_submodule("model.language_model")
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    qw = None
    for n, p in lm.named_parameters():
        if canon(f"language_model.{n}").replace(".", "__") == key:
            qw = p
            break
    assert qw is not None, f"q_proj weight for {key} not found"
    for p in model.parameters():
        p.requires_grad_(False)
    qw.requires_grad_(True)

    tr = ET.items("textvqa", "train", n_items + 64, n_eval_skip=128)[:n_items]
    cap = {}
    hook = layers[SITE].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    C_rows, modal = [], []
    for i, r in enumerate(tr):
        b = proc(text=[r["prompt"]], images=[r["image"]], suffix=[r["target"]],
                 return_tensors="pt", padding="longest")
        b["pixel_values"] = b["pixel_values"].to(torch.bfloat16)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        model.zero_grad(set_to_none=True)
        loss = model(**b).loss
        loss.backward()
        g = qw.grad.detach().float()
        C_rows.append((g @ Q).reshape(-1).cpu())                          # (2048*32,)
        X = cap["h"].reshape(-1, cap["h"].shape[-1]).float()
        lr, _ = GS._log_resp(X, atlas)
        comp = lr.argmax(dim=1)
        modal.append(int(torch.mode(comp).values))
        if (i + 1) % 32 == 0:
            print(f"[RIDER] {i + 1}/{n_items} ({time.time() - t0:.0f}s)", flush=True)
    hook.remove()

    C = torch.stack(C_rows)                                               # (N, 65536)
    Cc = C - C.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(Cc, q=2)
    score = (Cc @ V[:, 0]).numpy()
    sign = (score > 0).astype(int)
    modal = np.asarray(modal)

    def cramers_v(x, y):
        cats = sorted(set(x.tolist()))
        tab = np.zeros((len(cats), 2))
        for xi, yi in zip(x, y):
            tab[cats.index(xi), yi] += 1
        n = tab.sum()
        exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum((tab - exp) ** 2 / np.where(exp > 0, exp, np.nan))
        k = min(len(cats) - 1, 1)
        return float(np.sqrt(chi2 / (n * max(k, 1)))), tab.tolist()

    v_obs, tab = cramers_v(modal, sign)
    rng = np.random.default_rng(20260806)
    null = np.asarray([cramers_v(modal, rng.permutation(sign))[0] for _ in range(N_PERM)])
    p = float((1 + (null >= v_obs).sum()) / (N_PERM + 1))
    rep.update({"key": key, "cramers_v": v_obs, "perm_p": p,
                "null_p95": float(np.percentile(null, 95)),
                "contingency_component_x_sign": tab,
                "n_components_occupied": int(len(set(modal.tolist()))),
                "pc1_var_frac": float((Cc @ V[:, 0]).pow(2).sum()
                                      / Cc.pow(2).sum()),
                "elapsed_s": time.time() - t0,
                "reading": "EXPLORATORY: V high + p low => opposing in-core gradient signs "
                           "segregate by splat component (between-component cancellation "
                           "candidate); else the census is flat"})
    np.savez(out / "rider_bank.npz", score=score, sign=sign, modal=modal)
    _w()
    print("PRAGNANZ_RIDER_OK", json.dumps({"V": v_obs, "p": p}), flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # GESTALT D7
