"""W13 §6C — random-subspace-at-equal-N necessity null (the behavioral structure control).

The decisive control that makes a small-N result EVIDENCE rather than method impotence: at the
graft rank/atom-count N (set to the arm's N_behav after the discovery runs), compare the TOP-N real
ΔW directions against M RANDOM rank-N subspaces drawn INSIDE ΔW's range (prereg §3.3, §5). The
necessity gap `R_cap(top-N) − R_cap(random-N)` must have a CI > 0 before any small-N is read as a
basis (§6C). Anchored by the trained-exploit result: random-init rank-512/1024 installs ~0% while
top-of-ΔW reaches ceiling — so the gap should be large and positive if the top directions carry the
capability.

Random rank-N subspace inside ΔW's range: Q = U_r · R where U_r spans ΔW's left-singular space and
R is a random orthonormal (rank × N) — a random N-dim subspace of the range; ΔW_rand = Q Qᵀ ΔW.
Top-N = U[:, :N] U[:, :N]ᵀ ΔW (the rank-N truncation, == the low-rank arm at N).

    pg run infra.aws_pg.jobs.w13_necessity --type g6e.xlarge \
        --args="--rank=1024 --draws=10 --n=300" --s3-prefix results/w13_necessity --budget-min 120
"""
from __future__ import annotations

import argparse
import json
import pathlib

SEED = 1337
N_BOOT = 2000

from infra.aws_pg.jobs.e4_capability import _eval_rows, _generate, _per_item_scores, _boot_ci
from infra.aws_pg.jobs.e4_compress_curve import _set_decoder, _decoder_keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, required=True, help="N = the arm's N_behav (top-N vs random-N)")
    ap.add_argument("--draws", type=int, default=10, help="random-subspace draws")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--q", type=int, default=1700)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.pg import diff_harvest as H
    from infra.pg import battery_tasks as B

    H.hf_online_setup()
    device = "cuda"
    assert torch.cuda.is_available(), "necessity null needs a GPU"
    n = 16 if a.smoke else a.n
    draws = 2 if a.smoke else a.draws
    N = a.rank
    print(f"[W13-NEC] gpu={torch.cuda.get_device_name(0)} rank=N={N} draws={draws} n={n}", flush=True)

    pg_after, pg_before, proc, paths, _it, gate = H.build_pair("base-pt", device)
    lm_edit = pg_before.get_submodule(paths["language_model"])
    lm_after = pg_after.get_submodule(paths["language_model"])
    base_sd = {k: v.detach().clone().cpu() for k, v in lm_edit.state_dict().items()}
    after_sd = {k: v.detach().clone().cpu() for k, v in lm_after.state_dict().items()}
    dkeys = _decoder_keys()
    del pg_after; torch.cuda.empty_cache()

    tasks = {"refcoco": B.REFCOCO_TASK, "textvqa": B.TEXTVQA_TASK}
    rows = {name: _eval_rows(t, n, SEED) for name, t in tasks.items()}

    def score():
        return {name: _boot_ci(_per_item_scores(t, rows[name], _generate(pg_before, proc, t, rows[name], device, a.batch)), N_BOOT, SEED)
                for name, t in tasks.items()}

    # full-rank SVD factors per matrix (generous q, covers N)
    svd = {}
    for k in dkeys:
        dW = (after_sd[k].float() - base_sd[k].float()).to(device)
        U, S, V = torch.svd_lowrank(dW, q=min(a.q + 256, min(dW.shape)), niter=8)
        svd[k] = (U.cpu(), S.cpu(), V.cpu())
        del dW, U, S, V; torch.cuda.empty_cache()

    def graft_topN():
        d = {}
        for k in dkeys:
            U, S, V = svd[k]; kk = min(N, S.shape[0])
            dWk = (U[:, :kk].to(device) * S[:kk].to(device)) @ V[:, :kk].to(device).T
            d[k] = (base_sd[k].float().to(device) + dWk).to(torch.bfloat16)
        return d

    def graft_randomN(gen):
        """Project ΔW onto a random N-dim subspace INSIDE its range: Q = U_r R (R random
        orthonormal), ΔW_rand = Q Qᵀ ΔW."""
        d = {}
        for k in dkeys:
            U, S, V = svd[k]
            dW = (after_sd[k].float() - base_sd[k].float()).to(device)
            r = S.shape[0]; kk = min(N, r)
            Ur = U[:, :r].to(device)                                # range basis (out × r)
            R = torch.randn(r, kk, generator=gen, device=device)
            Q, _ = torch.linalg.qr(R)                               # (r × kk) orthonormal columns
            Qd = Ur @ Q                                             # (out × kk) random subspace of range
            dW_rand = Qd @ (Qd.T @ dW)                              # project ΔW onto it (rank kk)
            d[k] = (base_sd[k].float().to(device) + dW_rand).to(torch.bfloat16)
            del dW, Ur, R, Q, Qd, dW_rand; torch.cuda.empty_cache()
        return d

    # top-N
    _set_decoder(lm_edit, base_sd, graft_topN(), device)
    top = score()
    print(f"[W13-NEC] top-{N}: refcoco={top['refcoco']['mean']:.3f} textvqa={top['textvqa']['mean']:.3f}", flush=True)

    # random-N draws
    gen = torch.Generator(device=device).manual_seed(SEED)
    rand = {name: [] for name in tasks}
    for di in range(draws):
        _set_decoder(lm_edit, base_sd, graft_randomN(gen), device)
        sc = score()
        for name in tasks:
            rand[name].append(sc[name]["mean"])
        print(f"[W13-NEC] random-{N} draw {di+1}/{draws}: "
              f"refcoco={sc['refcoco']['mean']:.3f} textvqa={sc['textvqa']['mean']:.3f}", flush=True)

    import numpy as np
    gap = {}
    for name in tasks:
        rmean = float(np.mean(rand[name])); rstd = float(np.std(rand[name]))
        gap[name] = {"top_N": top[name]["mean"], "random_N_mean": rmean, "random_N_std": rstd,
                     "random_N_draws": rand[name], "necessity_gap": top[name]["mean"] - rmean,
                     "gap_positive": bool(top[name]["mean"] - rmean > 0)}
        print(f"[W13-NEC] {name}: necessity_gap = {gap[name]['necessity_gap']:.3f} "
              f"(top={top[name]['mean']:.3f} − random={rmean:.3f}±{rstd:.3f})", flush=True)

    results = {"experiment": "w13_necessity", "rank_N": N, "draws": draws, "n_per_task": n,
               "swap_gate": gate, "necessity": gap}
    (out / "w13_necessity.json").write_text(json.dumps(results, indent=2))
    print("[W13-NEC] done → w13_necessity.json", flush=True)


if __name__ == "__main__":
    main()
