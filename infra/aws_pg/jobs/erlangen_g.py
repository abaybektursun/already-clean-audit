"""ERLANGEN Arm G — the edit-grammar compression loop (spec §2-G, methods-exploratory).

For each lineage hop, find the SHORTEST edit program (in bits) that reproduces the child's
behaviour within ε, and compare it against a structure-free baseline at MATCHED fidelity. No
confirmatory claim is made this round; the deliverable is the engine plus an honest table.

Grammar v0 terms
  rank r     ΔW ≈ top-r SVD                      dl = r(m+n)·bits
  sparse p   ΔW ≈ top-p|·| entries               dl = nnz·(bits + log2 mn)   (the index is charged)
  gain       W_child ≈ g·W_parent                dl = bits
  interface  embedding / projector copied whole   dl = dense
  dense      ΔW exactly                          dl = mn·bits

Fidelity is BEHAVIOURAL, not weight-space: install parent+reconstruction and measure the same
readout battery, ε = mean_r |NLL_r(recon) − NLL_r(child)|. A reconstruction that is close in
Frobenius norm but far in behaviour does not count as a reproduction (W12: energy ≠ function).

P6 gate: a PLANTED rank-8 + interface edit must be recovered — the search must select r ≤ 8 and
reach ε ≈ 0. If it does not, the engine is VOID and no compression table is reported.

    python -m infra.aws_pg.jobs.erlangen_g --out DIR --ft-max-repo … --ft-min-repo … [--eps 0.05]

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

RANKS = (1, 4, 16, 64, 256)
SPARSE = (0.001, 0.01, 0.1)
R_MAX = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.05, help="nats of mean NLL drift allowed")
    ap.add_argument("--ft-max-repo", default="")
    ap.add_argument("--ft-min-repo", default="")
    ap.add_argument("--hops", default="install,pi0,pi05,ft_max,ft_min")
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

    from infra.pg import erlangen_core as EC
    from infra.pg import erlangen_probe as P
    from infra.pg.lacuna_shell import install_canon, load_pg1, pg1_baseline_sd

    assert torch.cuda.is_available(), "Arm G needs a GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "erlangen_g_v1", "eps": a.eps, "ranks": list(RANKS), "sparse": list(SPARSE),
           "bits_per_param": EC.BITS_PER_PARAM, "hops": a.hops.split(",")}

    def bank():
        (out / "erlangen_g.json").write_text(json.dumps(rep, indent=2))

    bank()
    model, proc = load_pg1(device=device)
    pg1_sd = pg1_baseline_sd(model)
    gemma1 = P.donor_gemma1(pg1_sd)
    packs = P.build_items(n=a.n_items)
    rep["items_sha256"] = P.items_hash(packs)
    LM = sorted(k for k in pg1_sd if k.startswith("language_model.layers.") and pg1_sd[k].ndim == 2)
    IFACE = sorted(k for k in pg1_sd
                   if k.startswith("multi_modal_projector.") or "embed_tokens" in k)
    print(f"[G] {len(LM)} LM matrices, {len(IFACE)} interface tensors ({time.time()-t0:.0f}s)",
          flush=True)

    def hop_sds(hop):
        """(parent_sd, child_sd) for a hop, both as canonical shell state-dicts."""
        if hop == "install":
            return {**pg1_sd, **{k: v for k, v in gemma1.items()
                                 if k.startswith("language_model.") and "embed" not in k}}, pg1_sd
        if hop in ("pi0", "pi05"):
            return pg1_sd, P.unit_sd(hop, pg1_sd)
        if hop == "ft_max":
            return pg1_sd, P.conform(P.donor_repo(a.ft_max_repo), pg1_sd, "G/ft_max")
        if hop == "ft_min":
            return pg1_sd, P.conform(P.donor_repo(a.ft_min_repo), pg1_sd, "G/ft_min")
        if hop == "planted":
            return pg1_sd, planted_child()
        raise AssertionError(f"unknown hop {hop!r}")

    def planted_child(rank=8, seed=7):
        """P6: parent + a rank-8 LM edit + a wholly-replaced projector. The grammar must find it."""
        g = torch.Generator().manual_seed(seed)
        child = dict(pg1_sd)
        for k in LM:
            W = pg1_sd[k]
            m, n = W.shape
            U = torch.randn(m, rank, generator=g) * (0.02 / rank ** 0.5)
            V = torch.randn(rank, n, generator=g) * (0.02 / rank ** 0.5)
            child[k] = (W.float() + U @ V).to(W.dtype)
        for k in IFACE:
            if k.startswith("multi_modal_projector."):
                child[k] = (pg1_sd[k].float() * 1.05).to(pg1_sd[k].dtype)
        return child

    def battery():
        return P.vec(P.readout_vector(model, proc, device, packs, batch_size=a.batch))

    def fidelity(v, ref):
        return float(np.mean(np.abs(np.array(v) - np.array(ref))))

    rep["hop_results"] = {}
    for hop in ["planted"] + rep["hops"]:
        parent, child = hop_sds(hop)
        install_canon(model, child, f"G/{hop}/child")
        ref = battery()
        install_canon(model, parent, f"G/{hop}/parent")
        base = battery()
        res = {"hop": hop, "child_vec": ref, "parent_vec": base,
               "parent_child_gap": fidelity(base, ref), "programs": []}
        rep["hop_results"][hop] = res
        bank()
        print(f"[G] hop={hop} parent↔child battery gap={res['parent_child_gap']:.4f}", flush=True)

        # cached truncated SVD factors of ΔW, one pass per hop (CPU, fp32)
        fac = {}
        for k in LM:
            dW = (child[k].float() - parent[k].float())
            r = min(R_MAX, *dW.shape)
            U, S, Vh = torch.linalg.svd(dW.to(device), full_matrices=False)
            fac[k] = (U[:, :r].cpu(), S[:r].cpu(), Vh[:r].cpu())
            del U, S, Vh, dW
        torch.cuda.empty_cache()
        print(f"[G] hop={hop} SVD cache built ({time.time()-t0:.0f}s)", flush=True)

        def evaluate(term_kind, param, use_interface):
            # All reconstruction math runs on the GPU: a 16384x2048 truncated-SVD product or a
            # 33M-element topk on CPU costs ~2 s per matrix, i.e. ~4 min per evaluation across the
            # 126-matrix grid — the search would have been dominated by numpy, not by science.
            upd, dl = {}, []
            for k in LM:
                Wp = parent[k].to(device).float()
                dW = child[k].to(device).float() - Wp
                m, n = dW.shape
                if term_kind == "rank":
                    U, S, Vh = fac[k]
                    r = min(param, S.numel())
                    rec = (U[:, :r].to(device) * S[:r].to(device)) @ Vh[:r].to(device)
                    dl.append(EC.dl_rank(m, n, r))
                elif term_kind == "sparse":
                    nnz = max(1, int(param * m * n))
                    thr = torch.topk(dW.abs().flatten(), nnz, largest=True).values.min()
                    rec = torch.where(dW.abs() >= thr, dW, torch.zeros_like(dW))
                    dl.append(EC.dl_sparse(int((rec != 0).sum()), m, n))
                elif term_kind == "gain":
                    gg = float((child[k].to(device).float() * Wp).sum()
                               / Wp.pow(2).sum().clamp_min(1e-30))
                    rec = (gg - 1.0) * Wp
                    dl.append(EC.dl_gain())
                else:
                    raise AssertionError(term_kind)
                upd[k] = (Wp + rec).to(parent[k].dtype)
                del Wp, dW, rec
            terms = [{"kind": term_kind if term_kind != "gain" else "gain",
                      "dl": float(sum(dl))}]
            if use_interface:
                for k in IFACE:
                    upd[k] = child[k]
                terms.append({"kind": "interface",
                              "dl": float(sum(EC.dl_dense(*tuple(child[k].shape))
                                              if child[k].ndim == 2 else
                                              EC.dl_dense(child[k].shape[0], 1) for k in IFACE))})
            install_canon(model, upd, f"G/{hop}/{term_kind}{param}")
            del upd
            torch.cuda.empty_cache()
            v = battery()
            install_canon(model, parent, f"G/{hop}/reset")
            return {"term": term_kind, "param": param, "interface": use_interface,
                    "dl": EC.program_dl(terms), "terms": terms,
                    "fidelity": fidelity(v, ref), "vec": v}

        # BOTH grammars get the interface-copy option and are charged for it IDENTICALLY, so the
        # comparison isolates the one thing at issue: is low-rank/gain structure worth its bits
        # against raw sparse entries? (Giving the typed grammar a free interface term — the first
        # version of this loop — would have measured the handicap instead.)
        for use_if in (True, False):
            for r in RANKS:
                res["programs"].append(evaluate("rank", r, use_if))
                print(f"[G] {hop} rank{r} iface={use_if}: "
                      f"ε={res['programs'][-1]['fidelity']:.4f} "
                      f"dl={res['programs'][-1]['dl']:.3e}", flush=True)
                bank()
            for pp in SPARSE:
                res["programs"].append(evaluate("sparse", pp, use_if))
                print(f"[G] {hop} sparse{pp} iface={use_if}: "
                      f"ε={res['programs'][-1]['fidelity']:.4f} "
                      f"dl={res['programs'][-1]['dl']:.3e}", flush=True)
            res["programs"].append(evaluate("gain", 0, use_if))
            bank()

        ok = [p for p in res["programs"] if p["fidelity"] <= a.eps]
        typed = [p for p in ok if p["term"] in ("rank", "gain")]
        free = [p for p in ok if p["term"] == "sparse"]
        res["best_typed"] = min(typed, key=lambda p: p["dl"]) if typed else None
        res["best_structure_free"] = min(free, key=lambda p: p["dl"]) if free else None
        res["compression_ratio"] = (
            EC.compression_ratio(res["best_typed"]["dl"], res["best_structure_free"]["dl"])
            if (res["best_typed"] and res["best_structure_free"]) else None)
        res["dense_dl"] = float(sum(EC.dl_dense(*tuple(parent[k].shape)) for k in LM))
        print(f"[G] hop={hop} best_typed={res['best_typed'] and res['best_typed']['term']} "
              f"ratio={res['compression_ratio']}", flush=True)
        bank()
        del fac
        torch.cuda.empty_cache()

    # ---- P6 gate, computed LAST from numbers already banked -----------------------------------
    pl = rep["hop_results"]["planted"]
    hit = [p for p in pl["programs"] if p["term"] == "rank" and p["fidelity"] <= a.eps]
    rep["p6"] = {"recovered_rank": min((p["param"] for p in hit), default=None),
                 "planted_rank": 8,
                 "passes": bool(hit and min(p["param"] for p in hit) <= 16),
                 "failure_means": "the grammar cannot recover a planted low-rank+interface edit "
                                  "→ engine VOID, no compression table is reported"}
    rep["ENGINE_VALID"] = rep["p6"]["passes"]
    rep["total_secs"] = time.time() - t0
    rep["DONE"] = True
    bank()
    print(f"[G] DONE P6={rep['p6']['passes']} recovered_rank={rep['p6']['recovered_rank']} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
