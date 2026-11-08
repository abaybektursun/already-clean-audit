"""Invariant-core TRIVIALITY test — is the shared-untouched INPUT subspace trivial or meaningful?

Follow-up to `invariant_core.py` (same branch family). The prior run CONFIRMED a shared-untouched
input subspace across the 24 same-init PaliGemma-1 fine-tunes: for q_proj (18/18 layers) and k_proj,
the BOTTOM eigenvectors of the per-ft-unit-Frobenius consensus Gram
    G_M = Σ_i ΔW_iᵀ ΔW_i / ‖ΔW_i‖²_F        (n×n, the "untouched input directions")
sit FAR below a matched-random rotation null. THE OPEN QUESTION (journal open-Q#2): is that
subspace TRIVIAL (just the base weight's barely-used input directions — small singular values, no
signal flows there so no gradient writes there by construction) or MEANINGFUL (directions the base
DOES use, large/mid σ, yet every fine-tune preserves — a genuine invariant)?

DISCOVERY-grade, exploratory (no pre-reg). CPU, fp64, FAIL LOUD (no try/except, no synthetic
fallback). Reuses invariant_core.py's load + difference path VERBATIM (imported, not re-implemented).

For q_proj AND k_proj, all 18 layers:
  U_unt(M, k) = span of G_M's k SMALLEST eigenvectors (the untouched input subspace), k∈{8,32,128}.
  Base singular structure: W0_M = pt-224's weight. SVD W0_M = U Σ Vᵀ; the INPUT directions are the
    right-singular vectors V (columns of Vᵀᵀ). bottom-k' V = smallest σ ("barely-used"); top-k' V =
    largest σ ("most-used").

TEST A (dead-weight triviality), per matrix, per k:
  squared-cosine subspace overlap between U_unt(M,k) and (a) base BOTTOM-k right-singular subspace,
  (b) base TOP-k right-singular subspace. Random baseline: two random k-subspaces of ℝⁿ overlap
  ≈ k/n (n=2048 ⇒ k/2048). HIGH overlap with base-BOTTOM ⇒ trivial; overlap with base-TOP ≫ k/n ⇒
  striking/non-trivial. We also report the FULL base-σ profile inside U_unt: the energy-weighted
  mean base singular value of the untouched subspace vs the global mean — a one-number triviality
  readout (≪ global ⇒ untouched lives in small-σ land ⇒ trivial).

TEST B (one shared residual subspace?), per layer + a couple cross-layer pairs:
  squared-cosine overlap U_unt(q_proj,L) vs U_unt(k_proj,L), and U_unt(q,L) vs U_unt(q,L') —
  are the untouched subspaces the SAME across projections/layers (one shared residual subspace) or
  matrix-specific? vs the same k/n random baseline.

  # smoke (one layer, q+k):
  pg run infra.aws_pg.jobs.invariant_core_triv --type r7i.4xlarge --no-weights \
      --args "--layers 0 --k-grid 8,32,128" --budget-min 30
  # full (q+k, all 18 layers):
  pg run infra.aws_pg.jobs.invariant_core_triv --type r7i.4xlarge --no-weights \
      --args "--k-grid 8,32,128" --budget-min 120 --s3-prefix results/invcore_triv
"""
import argparse
import json
import pathlib

# Reuse the load + difference path VERBATIM (imported, not re-implemented).
from infra.aws_pg.jobs.invariant_core import (
    PG1_REPO,
    N_LAYERS,
    _load_decoder,
    _decoder_matrices,
    _ftype,
    _delete_cache,
)

TYPES = ("q_proj", "k_proj")


def _bottom_eigvecs(G, k):
    """fp64 symmetric eig (ascending eigenvalues). Returns the n×k matrix of the k SMALLEST
    eigenvectors (U_unt) + the corresponding eigenvalues. eigh gives orthonormal columns."""
    import numpy as np
    w, V = np.linalg.eigh(G)                 # ascending; V columns orthonormal
    kk = int(min(k, w.size))
    return V[:, :kk].astype(np.float64), w[:kk].astype(np.float64)


def _base_input_dirs(W0):
    """Right-singular vectors V (INPUT directions, n-dim) and singular values σ of the base weight
    W0 (m×n), fp64, DESCENDING σ. svd(full_matrices=False) gives Vh = Vᵀ (r×n, r=min(m,n)). Columns
    of Vh.T are the input directions; Vh.T[:, j] pairs with σ[j]. For an m<n matrix (MQA k_proj:
    256×2048) only r=256 input dirs have nonzero σ — the remaining n−r form the base's exact null
    space (σ=0, the most-barely-used directions of all)."""
    import numpy as np
    import torch
    U, s, Vh = torch.linalg.svd(W0.detach().to(torch.float64), full_matrices=False)
    return Vh.transpose(0, 1).cpu().numpy().astype(np.float64), s.cpu().numpy().astype(np.float64)


def _overlap(A, B):
    """Squared-cosine subspace overlap between two orthonormal-column bases A (n×a), B (n×b).
    raw = ‖AᵀB‖_F² = Σ cos²θ_i over principal angles (ranges 0..min(a,b)); normalised = raw/min(a,b)
    (ranges 0..1; = mean squared principal cosine). For two random k-subspaces of ℝⁿ the expected
    normalised overlap is k/n."""
    import numpy as np
    M = A.T @ B
    raw = float((M * M).sum())
    return raw, float(raw / min(A.shape[1], B.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="",
                    help="comma slugs; empty ⇒ enumerate all -224 (the same N5c cloud as invariant_core)")
    ap.add_argument("--layers", default="",
                    help="comma layer indices; empty ⇒ all 18")
    ap.add_argument("--k-grid", default="8,32,128", help="untouched-subspace dims to sweep")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k_grid = [int(x) for x in a.k_grid.split(",")]
    want_layers = ([int(x) for x in a.layers.split(",") if x.strip()]
                   if a.layers.strip() else list(range(N_LAYERS)))

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                       # FIRST — gated repos 401 otherwise
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    api = HfApi()

    # Same cloud as invariant_core: enumerate all -224 (matches N5c) or pin an explicit list.
    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha

    rep = {"object": "invariant_core_triviality_v1", "base_repo": PG1_REPO, "base_sha": base_sha,
           "cloud_repos": repos, "cloud_shas": shas, "n_ft": len(repos),
           "types": list(TYPES), "layers": want_layers, "k_grid": k_grid, "seed": a.seed,
           "U_unt": "bottom-k eigvecs of G=Σ_i ΔWᵢᵀΔWᵢ/‖ΔWᵢ‖²_F (per-ft unit-Frobenius consensus)",
           "base_input_dirs": "right-singular vectors V of base W0 (pt-224); top/bottom by σ",
           "overlap": "squared-cosine subspace overlap ‖AᵀB‖_F² (raw 0..k) + normalised /min(a,b); "
                      "random baseline k/n with n=2048"}
    print(f"[triv] base={PG1_REPO}@{base_sha[:8]} cloud={len(repos)} ft types={TYPES} "
          f"layers={want_layers} k={k_grid}", flush=True)

    # ---- base decoder: keep the q/k weights we need (for SVD), and the consensus Gram accumulator.
    w_base = _load_decoder(PG1_REPO, base_sha)
    base_W0 = {}        # base_W0[(type,layer)] = W0 (m×n) fp32
    for t in TYPES:
        for L in want_layers:
            key = f"layers.{L}.self_attn.{t}.weight"
            assert key in w_base, f"missing base weight {key}"
            base_W0[(t, L)] = w_base[key]

    # consensus Gram per (type,layer), accumulated over the 24 ft (one matrix at a time, no scratch:
    # q/k are 2048-input and small — the full G fits trivially; we never hold all 24 ΔW at once).
    n_in = {(t, L): base_W0[(t, L)].shape[1] for t in TYPES for L in want_layers}
    for (t, L), nn in n_in.items():
        assert nn == 2048, f"unexpected input dim {nn} for {t} L{L} (q/k input must be 2048)"
    G = {(t, L): np.zeros((2048, 2048), dtype=np.float64) for t in TYPES for L in want_layers}
    n_terms = {(t, L): 0 for t in TYPES for L in want_layers}

    for ridx, repo_id in enumerate(repos):
        sha = shas[repo_id]
        print(f"[triv] diff {repo_id}@{sha[:8]} ({ridx+1}/{len(repos)}) ...", flush=True)
        w_ft = _load_decoder(repo_id, sha)
        matched = _decoder_matrices(w_base, w_ft)
        for t in TYPES:
            for L in want_layers:
                key = f"layers.{L}.self_attn.{t}.weight"
                wb, wf = matched[key]
                assert _ftype(key) == t
                A = (wf.to(torch.float64) - wb.to(torch.float64))     # ΔW, m×n
                fro2 = float((A ** 2).sum())
                assert fro2 > 0, f"zero ΔW {key} in {repo_id} (frozen/degenerate ft)"
                term = (A.transpose(0, 1) @ A).cpu().numpy().astype(np.float64) / fro2
                G[(t, L)] += term
                n_terms[(t, L)] += 1
        del w_ft, matched
        _delete_cache(repo_id)

    for key, nt in n_terms.items():
        assert nt == len(repos), f"{key} got {nt} terms, expected {len(repos)}"

    # ---- TEST A: per (type,layer), overlap(U_unt, base-bottom-k) and overlap(U_unt, base-top-k).
    n = 2048
    U_unt = {}          # U_unt[(type,layer,k)] = n×k orthonormal (cached for TEST B)
    base_V = {}         # base_V[(type,layer)] = (V_full n×r, sigma r) ; r=2048 for q, 256 for k
    test_a = []
    for t in TYPES:
        for L in want_layers:
            V_full, sigma = _base_input_dirs(base_W0[(t, L)])
            base_V[(t, L)] = (V_full, sigma)
            r = sigma.shape[0]
            rec = {"type": t, "layer": L, "n_in": n, "base_rank": int(r),
                   "base_sigma_max": float(sigma.max()), "base_sigma_min": float(sigma.min()),
                   "k": {}}
            for k in k_grid:
                Uk, eigk = _bottom_eigvecs(G[(t, L)], k)
                U_unt[(t, L, k)] = Uk
                # base bottom-k': smallest-σ input dirs. If r<n (k_proj MQA), the n−r exact null-space
                # dirs (σ=0) are the most-barely-used; build the base BOTTOM subspace from the true
                # smallest-σ directions, padding with null-space basis when k exceeds r is unneeded
                # (k≤128 < r for q=2048; for k=256-input k_proj r=256 so k≤128<r too) — bottom/top
                # both drawn from the r real right-singular vectors.
                assert k <= r, f"k={k} exceeds base rank r={r} for {t} L{L}"
                base_bot = V_full[:, r - k:]          # smallest-σ k input directions
                base_top = V_full[:, :k]              # largest-σ k input directions
                raw_bot, nm_bot = _overlap(Uk, base_bot)
                raw_top, nm_top = _overlap(Uk, base_top)
                # energy-weighted base-σ profile INSIDE U_unt: project each base input dir onto U_unt,
                # weight σ by captured energy. mean_sigma_in_unt = Σ_j σ_j ‖proj_Uk v_j‖² / Σ_j ‖proj_Uk v_j‖².
                proj_e = (V_full.T @ Uk)              # r×k ; row j = coords of base dir j in U_unt
                cap = (proj_e * proj_e).sum(axis=1)   # r ; captured energy of base dir j by U_unt
                mean_sigma_in_unt = float((sigma * cap).sum() / cap.sum())
                rec["k"][str(k)] = {
                    "eig_min": float(eigk.min()), "eig_max": float(eigk.max()),
                    "overlap_base_bottom_raw": raw_bot, "overlap_base_bottom_norm": nm_bot,
                    "overlap_base_top_raw": raw_top, "overlap_base_top_norm": nm_top,
                    "random_baseline_norm": float(k / n),
                    "mean_sigma_in_unt": mean_sigma_in_unt,
                }
            rec["base_sigma_global_mean"] = float(sigma.mean())
            test_a.append(rec)
            k0 = str(k_grid[0])
            print(f"[triv][A] {t} L{L}: k={k_grid[0]} ov(base-bot)={rec['k'][k0]['overlap_base_bottom_norm']:.3f} "
                  f"ov(base-top)={rec['k'][k0]['overlap_base_top_norm']:.4f} rand={k_grid[0]/n:.4f} "
                  f"σ̄_unt={rec['k'][k0]['mean_sigma_in_unt']:.3e} (σ̄_global={rec['base_sigma_global_mean']:.3e})",
                  flush=True)
            (out / "invariant_core_triv.json").write_text(json.dumps({**rep, "test_a": test_a}, indent=2))

    # ---- TEST B: q-vs-k same layer + cross-layer q-vs-q. Are the untouched subspaces SHARED?
    test_b = []
    def _pair(t1, L1, t2, L2):
        rec = {"a": f"{t1}_L{L1}", "b": f"{t2}_L{L2}", "k": {}}
        for k in k_grid:
            raw, nm = _overlap(U_unt[(t1, L1, k)], U_unt[(t2, L2, k)])
            rec["k"][str(k)] = {"overlap_raw": raw, "overlap_norm": nm, "random_baseline_norm": float(k / n)}
        return rec

    # q-vs-k same layer, every layer present in both.
    for L in want_layers:
        if ("q_proj", L, k_grid[0]) in U_unt and ("k_proj", L, k_grid[0]) in U_unt:
            r = _pair("q_proj", L, "k_proj", L)
            test_b.append(r)
            print(f"[triv][B] q_L{L} vs k_L{L}: k={k_grid[0]} overlap_norm={r['k'][str(k_grid[0])]['overlap_norm']:.3f} "
                  f"(rand {k_grid[0]/n:.4f})", flush=True)
    # cross-layer q-vs-q (a couple of pairs: adjacent + far).
    qlayers = [L for L in want_layers if ("q_proj", L, k_grid[0]) in U_unt]
    cross_pairs = []
    if len(qlayers) >= 2:
        cross_pairs.append((qlayers[0], qlayers[1]))                 # adjacent
        cross_pairs.append((qlayers[0], qlayers[len(qlayers) // 2])) # mid
        cross_pairs.append((qlayers[0], qlayers[-1]))                # far
        cross_pairs.append((qlayers[len(qlayers) // 2], qlayers[-1]))
    for (L1, L2) in sorted(set(cross_pairs)):
        if L1 == L2:
            continue
        r = _pair("q_proj", L1, "q_proj", L2)
        test_b.append(r)
        print(f"[triv][B] q_L{L1} vs q_L{L2}: k={k_grid[0]} overlap_norm={r['k'][str(k_grid[0])]['overlap_norm']:.3f} "
              f"(rand {k_grid[0]/n:.4f})", flush=True)
    # one cross-layer k-vs-k for completeness.
    klayers = [L for L in want_layers if ("k_proj", L, k_grid[0]) in U_unt]
    if len(klayers) >= 2:
        r = _pair("k_proj", klayers[0], "k_proj", klayers[-1])
        test_b.append(r)
        print(f"[triv][B] k_L{klayers[0]} vs k_L{klayers[-1]}: k={k_grid[0]} "
              f"overlap_norm={r['k'][str(k_grid[0])]['overlap_norm']:.3f} (rand {k_grid[0]/n:.4f})", flush=True)

    # ---- AGGREGATE: across the q_proj layers (+ k_proj) at each k, mean overlaps + the verdict number.
    def _agg_a(records, ks):
        ob = [r["k"][ks]["overlap_base_bottom_norm"] for r in records]
        ot = [r["k"][ks]["overlap_base_top_norm"] for r in records]
        rs = [r["k"][ks]["mean_sigma_in_unt"] / r["base_sigma_global_mean"] for r in records]
        return {"n": len(records),
                "overlap_base_bottom_norm_mean": float(np.mean(ob)),
                "overlap_base_bottom_norm_min": float(np.min(ob)),
                "overlap_base_bottom_norm_max": float(np.max(ob)),
                "overlap_base_top_norm_mean": float(np.mean(ot)),
                "overlap_base_top_norm_max": float(np.max(ot)),
                "random_baseline_norm": float(int(ks) / n),
                "sigma_ratio_unt_over_global_mean": float(np.mean(rs)),
                "sigma_ratio_unt_over_global_min": float(np.min(rs))}
    by_type = {t: {ks: _agg_a([r for r in test_a if r["type"] == t], ks) for ks in map(str, k_grid)}
               for t in TYPES}
    rep["test_a"] = test_a
    rep["test_b"] = test_b
    rep["aggregate_test_a"] = by_type
    (out / "invariant_core_triv.json").write_text(json.dumps(rep, indent=2))

    print("=" * 78)
    for t in TYPES:
        for k in k_grid:
            ks = str(k)
            agg = by_type[t][ks]
            print(f"[triv][AGG] {t} k={k}: overlap(base-BOTTOM)={agg['overlap_base_bottom_norm_mean']:.3f} "
                  f"overlap(base-TOP)={agg['overlap_base_top_norm_mean']:.4f} rand={k/n:.4f} "
                  f"| σ̄_unt/σ̄_global={agg['sigma_ratio_unt_over_global_mean']:.3f}", flush=True)
    print(f"INVARIANT_CORE_TRIV_OK n_ft={len(repos)} types={TYPES} layers={len(want_layers)} k={k_grid}")


if __name__ == "__main__":
    main()
