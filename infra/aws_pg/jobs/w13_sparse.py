"""W13 — sparse-dictionary arm (the SOLE confirmatory arm). Behavioral graft of an overcomplete
sparse-dictionary reconstruction of ΔW, scored on RefCOCO + TextVQA → R_cap → N_behav.

The genuinely open question (prereg §1): does the eff-rank-heavy ΔW (eff-rank≈1440, recompute at
results/w13_real_dw) admit a SMALL sparse parameter basis — N_behav(τ*) < N_sparse=min(64,0.1·eff)
— or is it irreducible (N_behav > N_dense=0.5·eff)? Faithfulness BEHAVIORAL (R_cap), never Frobenius.

Method (prereg §2.3, §3.2; refined pre-freeze in §9-2 develop — see below): per decoder matrix,
orient to (samples=max dim, vec-dim=min dim); learn one K_max-atom dict ONCE (FISTA + MOD, fp32 GPU —
discovery precision; the frozen run asserts fp32↔fp64). The SWEEP AXIS is the dict SIZE k (the atom
budget), via the top-k most-used atoms (the §9-2 finding: "atoms used by ≥1 row" saturates to ~K at
LM row counts, so it is λ-insensitive and degenerate; truncating to the top-k atoms is the decisive,
λ-free readout, directly comparable to the low-rank rank-ladder — k ↔ rank). Each k is ONE
whole-decoder graft (base gemma-2-2b decoder + the k-atom sparse reconstruction on all matrices).
N_behav = the smallest atom budget k meeting R_cap ≥ τ*, vs the frozen N_sparse=64 / N_dense=720.

    pg run infra.aws_pg.jobs.w13_sparse --type g6e.xlarge \
        --args="--n=300 --layers=all --k-ladder=32,64,128,256,512,1024 --k-max=1024" \
        --s3-prefix results/w13_sparse --budget-min 240
"""
from __future__ import annotations

import argparse
import json
import pathlib

SEED = 1337
N_BOOT = 2000
TAU_STAR = 0.9

from infra.aws_pg.jobs.e4_capability import _eval_rows, _generate, _per_item_scores, _boot_ci
from infra.aws_pg.jobs.e4_compress_curve import _set_decoder, _decoder_keys, _decoder_key, N_LAYERS, LM_MATRIX_TYPES


def _soft(z, thr):
    import torch
    return torch.sign(z) * torch.clamp(z.abs() - thr, min=0.0)


def _specnorm_sq(D, n_iter=15):
    """σ_max(D)² = λ_max(DᵀD) via power iteration (cheap; replaces a full 4096² SVD per FISTA call,
    the sparse arm's dominant cost). D:(K,d) → iterate the d-dim eigenvector of DᵀD."""
    import torch
    v = torch.randn(D.shape[1], device=D.device, dtype=D.dtype)
    v = v / v.norm().clamp_min(1e-30)
    for _ in range(n_iter):
        v = D.T @ (D @ v)
        v = v / v.norm().clamp_min(1e-30)
    return (D @ v).pow(2).sum().item()


def _fista(X, D, lam, n_iter):
    """min_A 0.5‖X − A Dᵀ‖² + λ‖A‖₁ via FISTA. X:(R,d) D:(K,d) → A:(R,K). On X's device/dtype."""
    import torch
    R, K = X.shape[0], D.shape[0]
    L = _specnorm_sq(D)
    eta = 1.0 / max(L, 1e-12)
    A = torch.zeros(R, K, dtype=X.dtype, device=X.device)
    Y = A.clone(); t = 1.0
    for _ in range(n_iter):
        grad = (Y @ D - X) @ D.T
        A_next = _soft(Y - eta * grad, eta * lam)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        Y = A_next + ((t - 1.0) / t_next) * (A_next - A)
        A, t = A_next, t_next
    return A


def _lam_max(X, D):
    return (X @ D.T).abs().max().item()


def _learn_dict(X, n_atoms, lam_frac, n_outer, n_iter, gen):
    """MOD dict learning on rows of X (FISTA inner). Returns (D, A): unit-norm D:(n_atoms,d) and the
    final sparse codes A:(R,n_atoms). X on GPU fp32."""
    import torch
    R, d = X.shape
    idx = torch.randperm(R, generator=gen, device=X.device)[:n_atoms]
    D = X[idx].clone()
    if D.shape[0] < n_atoms:
        D = torch.cat([D, torch.randn(n_atoms - D.shape[0], d, generator=gen, device=X.device, dtype=X.dtype)], 0)
    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
    A = None
    for _ in range(n_outer):
        A = _fista(X, D, lam_frac * _lam_max(X, D), n_iter)
        D = torch.linalg.pinv(A) @ X
        nrm = D.norm(dim=1, keepdim=True)
        dead = (nrm < 1e-9).squeeze(1)
        if dead.any():
            worst = torch.topk((X - A @ D).norm(dim=1), int(dead.sum().item())).indices
            D[dead] = X[worst]; nrm = D.norm(dim=1, keepdim=True)
        D = D / nrm.clamp_min(1e-12)
    A = _fista(X, D, lam_frac * _lam_max(X, D), n_iter)
    return D, A


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--layers", default="all", help="all | odd | even | comma list e.g. 5,12,20")
    ap.add_argument("--k-ladder", default="64,128,256,512,1024,2048,4608", help="atom-budget ladder")
    ap.add_argument("--dict-mult", type=float, default=2.0, help="OVERCOMPLETE factor: K=mult·(atom-space dim)")
    ap.add_argument("--k-max", type=int, default=8192, help="hard cap on the learned dict size")
    ap.add_argument("--learn-lam", type=float, default=0.10, help="λ_frac for dict learning")
    ap.add_argument("--read-lam", type=float, default=0.02, help="λ_frac for the top-k readout re-code")
    ap.add_argument("--n-outer", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.pg import diff_harvest as H
    from infra.pg import battery_tasks as B

    H.hf_online_setup()
    device = "cuda"
    assert torch.cuda.is_available(), "sparse-arm graft needs a GPU"
    n = 16 if a.smoke else a.n
    k_ladder = [int(x) for x in a.k_ladder.split(",")]
    if a.smoke:
        k_ladder = [64, 1024, 4608]; a.n_outer = 12; a.n_iter = 40
    layer_sel = a.layers
    layers = (list(range(N_LAYERS)) if layer_sel == "all" else
              list(range(1, N_LAYERS, 2)) if layer_sel == "odd" else
              list(range(0, N_LAYERS, 2)) if layer_sel == "even" else
              [int(x) for x in layer_sel.split(",")])
    dkeys = [_decoder_key(L, mt) for L in layers for mt in LM_MATRIX_TYPES]
    print(f"[W13-SP] gpu={torch.cuda.get_device_name(0)} n={n} layers={layer_sel}({len(layers)}) "
          f"matrices={len(dkeys)} dict_mult={a.dict_mult} (OVERCOMPLETE) k_ladder={k_ladder}", flush=True)

    pg_after, pg_before, proc, paths, _it, gate = H.build_pair("base-pt", device)
    lm_edit = pg_before.get_submodule(paths["language_model"])
    lm_after = pg_after.get_submodule(paths["language_model"])
    base_sd = {k: v.detach().clone().cpu() for k, v in lm_edit.state_dict().items()}
    after_sd = {k: v.detach().clone().cpu() for k, v in lm_after.state_dict().items()}
    all_dkeys = _decoder_keys()
    del pg_after; torch.cuda.empty_cache()

    tasks = {"refcoco": B.REFCOCO_TASK, "textvqa": B.TEXTVQA_TASK}
    rows = {name: _eval_rows(t, n, SEED) for name, t in tasks.items()}

    def score():
        return {name: _boot_ci(_per_item_scores(t, rows[name], _generate(pg_before, proc, t, rows[name], device, a.batch)), N_BOOT, SEED)
                for name, t in tasks.items()}

    # FLOOR / CEILING (full 2-D ΔW graft over ALL matrices; the achieved ceiling, §2.4d)
    _set_decoder(lm_edit, base_sd, {}, device); floor = score()
    _set_decoder(lm_edit, base_sd, {k: after_sd[k].to(torch.bfloat16) for k in all_dkeys}, device); ceiling = score()
    print(f"[W13-SP] FLOOR refcoco={floor['refcoco']['mean']:.3f} textvqa={floor['textvqa']['mean']:.3f} | "
          f"CEILING refcoco={ceiling['refcoco']['mean']:.3f} textvqa={ceiling['textvqa']['mean']:.3f}", flush=True)

    # ---- learn one K_max-atom dict per matrix; rank atoms by usage (GPU fp32) ----
    # The dict-size LADDER is the sweep axis (analog of the low-rank rank-ladder): a SPARSE BASIS
    # exists iff few REUSED atoms suffice behaviorally. The per-matrix "atoms used by ≥1 row" metric
    # saturates to ~K at LM row counts (smoke finding) → degenerate; sweeping the atom budget k and
    # truncating to the top-k most-used atoms is the decisive, λ-free, low-rank-comparable readout.
    gen = torch.Generator(device=device).manual_seed(SEED)
    dicts, usage, orient = {}, {}, {}
    for ki, k in enumerate(dkeys):
        dW = (after_sd[k].float() - base_sd[k].float()).to(device)        # (out, in)
        rows_as_samples = dW.shape[0] >= dW.shape[1]
        X = dW if rows_as_samples else dW.T                              # (samples, dim=min)
        # OVERCOMPLETE dict (critic fix): K = mult · (atom-space dim), so the dictionary genuinely
        # exceeds the embedding dim on every matrix type (the prior flat K=1280 was UNDERCOMPLETE on
        # Q/O dim 2048 and MLP dim 2304 — the load-bearing matrices — voiding the "overcomplete reuse
        # SVD misses" that defines this arm). Capped by #samples (MOD pinv) and the hard k_max.
        K = min(int(a.dict_mult * X.shape[1]), X.shape[0] - 1, a.k_max)
        D, A = _learn_dict(X, K, a.learn_lam, a.n_outer, a.n_iter, gen)
        dicts[k] = D.cpu(); usage[k] = A.norm(dim=0).cpu(); orient[k] = rows_as_samples
        if ki % 30 == 0:
            print(f"[W13-SP] learned dict {ki+1}/{len(dkeys)} {k} K={K} dim={X.shape[1]} samples={X.shape[0]}", flush=True)
        del dW, X, D, A; torch.cuda.empty_cache()

    def recon_topk(k):
        """Keep each matrix's top-k most-used atoms, re-code rows sparsely against that k-atom
        sub-dict → {key: base+recon bf16}. The atom budget IS k (the N for this arm)."""
        edited = {}
        for key in dkeys:
            dW = (after_sd[key].float() - base_sd[key].float()).to(device)
            X = dW if orient[key] else dW.T
            D = dicts[key].to(device)
            kk = min(k, D.shape[0])
            top = torch.topk(usage[key].to(device), kk).indices
            Dk = D[top]
            A = _fista(X, Dk, a.read_lam * _lam_max(X, Dk), a.n_iter)
            R = A @ Dk
            recon = R if orient[key] else R.T
            edited[key] = (base_sd[key].float().to(device) + recon).to(torch.bfloat16)
            del dW, X, D, Dk, A, R, recon; torch.cuda.empty_cache()
        return edited

    def rcap(sc):
        return {name: {"score": sc[name]["mean"], "ci": [sc[name]["lo"], sc[name]["hi"]],
                       "R_cap": min(1.0, (sc[name]["mean"] - floor[name]["mean"]) /
                                    max(ceiling[name]["mean"] - floor[name]["mean"], 1e-9))}
                for name in tasks}

    curve = {}
    for k in k_ladder:
        _set_decoder(lm_edit, base_sd, recon_topk(k), device)
        c = rcap(score())
        curve[k] = c
        print(f"[W13-SP] k={k:5d} atoms/matrix  refcoco R_cap={c['refcoco']['R_cap']:.3f} "
              f"(acc={c['refcoco']['score']:.3f})  textvqa R_cap={c['textvqa']['R_cap']:.3f}", flush=True)

    # N_behav = smallest atom-budget k meeting R_cap >= tau* (monotone in k). RefCOCO is the reliable
    # binary axis; TextVQA overshoots (low CEILING) so it is reported but not decisive.
    n_behav = {}
    for name in tasks:
        hit = [k for k in k_ladder if curve[k][name]["R_cap"] >= TAU_STAR]
        n_behav[name] = min(hit) if hit else None

    # INSTRUMENT-VALIDITY GATE (critic fix): the FULL overcomplete-dict reconstruction (largest k)
    # must install capability (R_cap ≈ CEILING). If it does NOT, the sparse instrument cannot even
    # represent ΔW behaviorally → a small-k null is method impotence, not absence of a sparse basis,
    # and the verdict is UNINTERPRETABLE. This is the in-experiment analog the prereg's HALT gate
    # demands (a full behavioral positive-control-#2 with a planted sparse edit is still owed).
    k_full = max(k_ladder)
    validity = {name: curve[k_full][name]["R_cap"] for name in tasks}
    instrument_potent = validity["refcoco"] >= 0.5
    print(f"[W13-SP] INSTRUMENT VALIDITY: full-dict k={k_full} RefCOCO R_cap={validity['refcoco']:.3f} → "
          f"{'POTENT (small-k null is interpretable)' if instrument_potent else 'IMPOTENT → null uninterpretable'}", flush=True)

    results = {
        "experiment": "w13_sparse", "arm": "sparse-dictionary (SOLE confirmatory)", "status": "discovery",
        "n_per_task": n, "layers": layer_sel, "n_matrices": len(dkeys), "k_ladder": k_ladder,
        "dict_mult": a.dict_mult, "learn_lam": a.learn_lam, "read_lam": a.read_lam, "tau_star": TAU_STAR,
        "thresholds": {"N_sparse": 64, "N_dense": 720.2, "eff_rank": 1440.3, "note": "recompute @ results/w13_real_dw"},
        "floor": floor, "ceiling": ceiling, "curve": {str(k): curve[k] for k in k_ladder},
        "N_behav_atoms": n_behav,
        "instrument_validity": {"k_full": k_full, "refcoco_R_cap": validity["refcoco"], "potent": instrument_potent,
                                "energy_ranking_caveat": "top-k selects highest-energy atoms; a tail-localized "
                                "sparse basis could still be missed — a behavioral positive-control-#2 is owed"},
        "verdict_hint": "ONLY if instrument_potent: N_behav<64 → sparse basis (H_invisible FALSIFIED); "
                        ">720 → no sparse basis. If NOT potent → INCONCLUSIVE (instrument impotent).",
    }
    (out / "w13_sparse.json").write_text(json.dumps(results, indent=2))
    print(f"[W13-SP] N_behav(atom budget k, τ*={TAU_STAR}): {n_behav} | thresholds N_sparse=64 N_dense=720", flush=True)
    print("[W13-SP] done → w13_sparse.json", flush=True)


if __name__ == "__main__":
    main()
