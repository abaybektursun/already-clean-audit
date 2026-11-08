"""Invariant core — is there a weight subspace essentially NONE of the same-init fine-tunes
write to, larger / more coherent than a matched-random (rotation) null predicts?

Discovery-grade, exploratory (no pre-reg). CPU, fp64, FAIL LOUD (no try/except, no synthetic
fallback). Reuses the N5c cloud recipe end-to-end:
  base  W1 = google/paligemma-3b-pt-224 language_model decoder (the init all 24 ft share)
  ft_i  W2 = google/paligemma-3b-ft-{task}-224 language_model decoder
  ΔW_i  = W2 − W1   per decoder weight matrix (18 layers × 7 types = 126 matrices)
(differs from N5c's ΔW₁ = pt-224 − gemma-2b: THAT is the PG1 CREATION step; here the object is
the FINE-TUNES of pt-224, the "what do all adaptations refuse to touch" question.)

ESTIMAND (PER weight matrix M, shape m×n; the "untouched INPUT directions" live in ℝ^n, the
right-singular / row space of ΔW). For fine-tune i let ΔW_i = U_i Σ_i V_iᵀ. The directions M is
left ALONE by fine-tune i are where ‖ΔW_i v‖ is small — the BOTTOM right-singular subspace.
Normalise each fine-tune to unit Frobenius so EVERY fine-tune contributes equally (scicap/refcoco
are 30–75× bigger than okvqa; without this the "intersection across ALL" degenerates to "what the
2 biggest leave alone"). Consensus Gram of the SAME matrix M across the N fine-tunes:

    G_M = Σ_i  Â_i ,   Â_i = ΔW_iᵀ ΔW_i / ‖ΔW_i‖²_F      (n×n, trace(Â_i)=1 ∀ i ⇒ trace(G_M)=N)

A direction v that EVERY normalised fine-tune leaves alone has small vᵀ G_M v. The candidate
invariant core of M at dimension k = the k-dim subspace minimising the consensus write-energy =
the span of G_M's k SMALLEST eigenvectors; its score is the mean of the bottom-k eigenvalues:

    core_k(M) = mean of the k smallest eigvals of G_M    (small ⇒ a jointly-untouched k-subspace)

Done PER (type,layer) matrix — NOT pooled across layers (different layers have different input
semantics; pooling would conflate them). 126 matrices, each with its own null + verdict; the
overall verdict aggregates the per-matrix z's (Stouffer) by type and across all.

THE MATCHED-RANDOM NULL (rotation, same per-matrix norm AND singular spectrum — the W12/N5c
rotation-invariance reduction). Replace each ΔW_i by a random matrix of the SAME shape and SAME
spectrum: V_i → R_i, a Haar-random orthonormal n×r frame (r=rank). Â_i^null = R_i diag(p_i) R_iᵀ,
p_i = σ_i²/Σσ_i² (REAL normalised squared spectrum, sums to 1). G_null = Σ_i Â_i^null; read
core_k(G_null) over many draws → a null band; report REAL core_k vs band as z + empirical percentile.

Interpretation. core_k BELOW the null band ⇒ the fine-tunes share a more-coherent untouched subspace
than chance (an invariant core beyond null). core_k INSIDE the band ⇒ no excess structure: whatever
each fine-tune leaves alone, the leftovers don't COMMONLY align beyond matched-random rotations.

Resumable: writes invariant_core.json after EACH matrix type. Deletes each ft's HF snapshot after
differencing (24×~12 GB fp32 overflows the 200 GB root). down_proj's n=16384 Gram makes its null
eig the cost driver → its draw count is separately tunable (--n-null-bigmlp).

  # smoke (preflight on ONE ft, ONE type):
  pg run infra.aws_pg.jobs.invariant_core --type r7i.4xlarge --no-weights \
      --args "--ft-tasks vqav2 --types q_proj --k-grid 8,32,128 --n-null 64" --budget-min 60
  # full (24 ft, all 7 types):
  pg run infra.aws_pg.jobs.invariant_core --type r7i.8xlarge --no-weights \
      --args "--k-grid 8,32,128 --n-null 200" --budget-min 600
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess

BUCKET = "pg-lineage-769063704696"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PG1_REPO = "google/paligemma-3b-pt-224"     # the shared init (= N5c's W1 / pt-224.LM)
N_LAYERS = 18


def _norm_keys(sd):
    return {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}


def _lm_decoder_sd(model):
    from infra.pg.entities import resolve_submodule_paths
    lm = model.get_submodule(resolve_submodule_paths(model)["language_model"])
    if hasattr(lm, "lm_head") and hasattr(lm, "model"):
        lm = lm.model
    return _norm_keys(lm.state_dict())


def _decoder_matrices(w_base, w_ft):
    """{key: (w_base, w_ft)} for the 18×7 2-D decoder weight matrices present in both, by key.
    Fail-loud on a missing matrix or shape mismatch (architecture regression)."""
    out = {}
    for k in sorted(set(w_base) & set(w_ft)):
        if any(t in k for t in PROJ) and w_base[k].ndim == 2:
            assert w_base[k].shape == w_ft[k].shape, f"shape mismatch {k}"
            out[k] = (w_base[k], w_ft[k])
    expect = {f"layers.{i}.{('self_attn' if p.startswith(('q','k','v','o')) else 'mlp')}.{p}.weight"
              for i in range(N_LAYERS) for p in PROJ}
    missing = sorted(expect - set(out))
    assert not missing, f"18×7 incomplete: {len(missing)} missing (e.g. {missing[:5]})"
    assert len(out) == N_LAYERS * 7, f"expected {N_LAYERS*7}, got {len(out)}"
    return out


def _ftype(key):
    return next(t for t in PROJ if t in key)


def _delete_cache(repo_id):
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", ""), "hub")
    d = os.path.join(hub, "models--" + repo_id.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d)            # no try/except — a cleanup failure must fail loud


def _spectrum_normsq(dW):
    """fp64 normalised squared spectrum p_i = σ²/Σσ² (sums to 1) of one ΔW. Length r=min(m,n)."""
    import torch
    s = torch.linalg.svdvals(dW.detach().to(torch.float64))
    s2 = s ** 2
    return (s2 / s2.sum()).cpu().numpy()


def _consensus_gram(dW_list):
    """G = Σ_i ΔW_iᵀΔW_i / ‖ΔW_i‖²_F   (n×n, fp64). Each term has trace 1 ⇒ trace(G)=len(list)."""
    import numpy as np
    import torch
    n = dW_list[0].shape[1]
    G = torch.zeros(n, n, dtype=torch.float64)
    for dW in dW_list:
        A = dW.detach().to(torch.float64)
        fro2 = (A ** 2).sum()
        assert fro2 > 0, "zero ΔW — a frozen/degenerate ft slipped the FT-scale gate"
        G += (A.transpose(0, 1) @ A) / fro2
    return G.cpu().numpy().astype(np.float64)


def _bottom_topk_eigs(G, k_grid):
    """fp64 symmetric eigvals (ascending). Returns the eigvals array + {k: mean of bottom-k} and
    {k: mean of top-k}. The bottom-k mean is the invariant-core score; top-k is its mirror (the
    most-WRITTEN consensus subspace) reported for context."""
    import numpy as np
    w = np.linalg.eigvalsh(G)              # ascending
    w = np.clip(w, 0.0, None)              # tiny negatives are fp roundoff on a PSD matrix
    out_bot, out_top = {}, {}
    for k in k_grid:
        kk = int(min(k, w.size))
        out_bot[str(k)] = float(w[:kk].mean())
        out_top[str(k)] = float(w[-kk:].mean())
    return w, out_bot, out_top


def _haar_gram(spectra, n, rng):
    """One matched-random consensus Gram draw: G_null = Σ_i R_i diag(p_i) R_iᵀ, R_i a Haar n×r
    orthonormal frame, p_i the REAL normalised squared spectrum of ft i (rotation-invariance: only
    the spectrum + ambient dim matter, never the real singular vectors). No SVD here — QR of a
    Gaussian gives the Haar frame; we build only the r columns that carry weight (r=len(p_i))."""
    import numpy as np
    G = np.zeros((n, n), dtype=np.float64)
    for p in spectra:
        r = p.shape[0]
        Z = rng.standard_normal((n, r))
        Q, _ = np.linalg.qr(Z)             # n×r, columns orthonormal (Haar frame)
        G += (Q * p[None, :]) @ Q.T        # = Q diag(p) Qᵀ
    return G


def _null_band(spectra, n, k_grid, n_null, seed):
    """Matched-random null band for the bottom-k core score (and the top-k mirror). Returns, per k,
    the null samples' mean/std/percentiles for BOTH bottom-k and top-k. n_null full eigendecomps of
    n×n — the cost driver at n=16384, so n_null is caller-tunable per type."""
    import numpy as np
    rng = np.random.default_rng(seed)
    bot = {str(k): [] for k in k_grid}
    top = {str(k): [] for k in k_grid}
    for _ in range(n_null):
        G0 = _haar_gram(spectra, n, rng)
        w = np.clip(np.linalg.eigvalsh(G0), 0.0, None)
        for k in k_grid:
            kk = int(min(k, w.size))
            bot[str(k)].append(float(w[:kk].mean()))
            top[str(k)].append(float(w[-kk:].mean()))

    def _stat(xs):
        a = np.asarray(xs, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size),
                "p0.5": float(np.percentile(a, 0.5)), "p2.5": float(np.percentile(a, 2.5)),
                "p50": float(np.percentile(a, 50.0)), "p97.5": float(np.percentile(a, 97.5)),
                "p99.5": float(np.percentile(a, 99.5))}
    return ({str(k): _stat(bot[str(k)]) for k in k_grid},
            {str(k): _stat(top[str(k)]) for k in k_grid},
            {str(k): bot[str(k)] for k in k_grid})


Z_MARGIN = 3.0      # an invariant core worth CLAIMING must be FAR outside the null (Case-B planted
                    # cores read z≈−400), not at its 2.5% edge — a marginal z≈−2 at small n is null
                    # finite-sample fluctuation (offline gate: an iid-random control grazed p2.5 at
                    # bottom-8 with z=−2.1). So BEYOND_NULL requires BOTH the percentile cut AND |z|≥3.


def _verdict(real_bot, null_bot, null_bot_samples, k_grid):
    """Per-k: z = (real − null_mean)/null_std (NEGATIVE z ⇒ real core BELOW null ⇒ more coherent
    untouched subspace than chance) + empirical lower-tail p (fraction of null ≤ real). Verdict at
    each k: BEYOND_NULL_below iff real < null-p2.5 AND z ≤ −Z_MARGIN (a real, far-outside-null core,
    not a 2.5%-edge graze); ABOVE_NULL iff real > null-p97.5 AND z ≥ +Z_MARGIN; else AT_NULL."""
    import numpy as np
    out = {}
    for k in k_grid:
        ks = str(k)
        r = real_bot[ks]
        nm, ns = null_bot[ks]["mean"], null_bot[ks]["std"]
        z = (r - nm) / ns if ns > 0 else float("nan")
        samp = np.asarray(null_bot_samples[ks], dtype=np.float64)
        p_lower = float((np.sum(samp <= r) + 1) / (samp.size + 1))
        below = (r < null_bot[ks]["p2.5"]) and (z <= -Z_MARGIN)
        above = (r > null_bot[ks]["p97.5"]) and (z >= Z_MARGIN)
        v = "BEYOND_NULL_below" if below else ("ABOVE_NULL" if above else "AT_NULL")
        out[ks] = {"real": r, "null_mean": nm, "null_std": ns, "z": z,
                   "p_lower": p_lower, "verdict": v, "z_margin": Z_MARGIN,
                   "null_p2.5": null_bot[ks]["p2.5"], "null_p97.5": null_bot[ks]["p97.5"]}
    return out


def _load_decoder(repo_id, sha=None):
    import torch
    from transformers import PaliGemmaForConditionalGeneration
    kw = dict(torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    if sha:
        kw["revision"] = sha
    m = PaliGemmaForConditionalGeneration.from_pretrained(repo_id, **kw)
    return _lm_decoder_sd(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="",
                    help="comma slugs (e.g. vqav2,gqa); empty ⇒ enumerate all -224 (the N5c cloud)")
    ap.add_argument("--types", default=",".join(PROJ),
                    help="weight-matrix types to analyse; default all 7")
    ap.add_argument("--k-grid", default="8,32,128", help="bottom-k subspace dims to sweep")
    ap.add_argument("--n-null", type=int, default=200, help="matched-random null draws per type")
    ap.add_argument("--n-null-bigmlp", type=int, default=0,
                    help="override n-null for the n=16384 down_proj (0 ⇒ use --n-null)")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k_grid = [int(x) for x in a.k_grid.split(",")]
    want_types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    assert all(t in PROJ for t in want_types), f"unknown type in {want_types}"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                       # FIRST — gated repos 401 otherwise
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    api = HfApi()

    # the cloud: enumerate all -224 (matches N5c) or pin an explicit --ft-tasks list. Resolve+pin SHAs.
    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha

    rep = {"object": "invariant_core_v1", "base_repo": PG1_REPO, "base_sha": base_sha,
           "cloud_repos": repos, "cloud_shas": shas, "n_ft": len(repos),
           "types": list(want_types), "k_grid": k_grid, "n_null": a.n_null,
           "n_null_bigmlp": a.n_null_bigmlp or a.n_null, "seed": a.seed,
           "estimand": "mean of bottom-k eigvals of G=Σ_i ΔWᵢᵀΔWᵢ/‖ΔWᵢ‖²_F (per-ft unit-Frobenius)",
           "null": "rotation-invariant matched-random: Vᵢ→Haar Rᵢ, real normalised squared spectrum"}
    print(f"[invcore] base={PG1_REPO}@{base_sha[:8]}  cloud={len(repos)} ft  types={want_types} "
          f"k={k_grid} n_null={a.n_null}", flush=True)

    # FIXED base decoder.
    w_base = _load_decoder(PG1_REPO, base_sha)

    # ---- PASS 1: difference every ft, accumulate per-TYPE the list of ΔW (one per ft) + spectra.
    # We hold ΔW per type in fp32 (memory): per ft, the 18 layers of a type. Across 24 ft that is, for
    # the worst type (gate/up 16384×2048 fp32 ≈ 134 MB each × 18 × 24 ≈ 58 GB) — too big to hold ALL
    # types for ALL ft at once. So we process ONE type at a time across all ft (re-load each ft per
    # type would re-download 24×12GB×7). Instead: load each ft ONCE, extract+store the wanted types'
    # ΔW to local fp16 .npy scratch keyed (type,layer,ft), delete the HF cache, then per type assemble.
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    spectra = {t: {} for t in want_types}          # spectra[type][repo] = list of per-layer p arrays
    preflight = {}

    smoke_done = False
    for ridx, repo_id in enumerate(repos):
        sha = shas[repo_id]
        print(f"[invcore] diff {repo_id}@{sha[:8]} ({ridx+1}/{len(repos)}) ...", flush=True)
        w_ft = _load_decoder(repo_id, sha)
        matched = _decoder_matrices(w_base, w_ft)
        # frame preflight (control): MQA k/v single-head rows + fp64 ΔW well-formed on q_proj L0.
        if ridx == 0:
            k0 = matched["layers.0.self_attn.k_proj.weight"]
            v0 = matched["layers.0.self_attn.v_proj.weight"]
            preflight["kv_rows"] = [int(k0[1].shape[0]), int(v0[1].shape[0])]
            assert k0[1].shape[0] == v0[1].shape[0] == 256 and k0[1].shape[1] == 2048, \
                f"MQA misaligned: k {tuple(k0[1].shape)} v {tuple(v0[1].shape)}"
        for key, (wb, wf) in matched.items():
            t = _ftype(key)
            if t not in want_types:
                continue
            layer = int(key.split("layers.")[1].split(".")[0])
            dW = (wf.to(torch.float64) - wb.to(torch.float64))
            spectra[t].setdefault(repo_id, {})[layer] = _spectrum_normsq(dW)
            np.save(scratch / f"{t}__L{layer}__{ridx}.npy",
                    dW.to(torch.float16).cpu().numpy())     # fp16 scratch: differencing already fp64
            # ---- SMOKE: on the very first (type,layer,ft), run the WHOLE pipeline on that one matrix
            if not smoke_done and t == want_types[0] and layer == 0:
                dWn = dW / dW.norm()
                Gs = (dWn.transpose(0, 1) @ dWn).cpu().numpy().astype(np.float64)
                ws, sb, st_ = _bottom_topk_eigs(Gs, k_grid)
                rng = np.random.default_rng(a.seed)
                G0 = _haar_gram([spectra[t][repo_id][layer]], dWn.shape[1], rng)
                w0 = np.clip(np.linalg.eigvalsh(G0), 0.0, None)
                print(f"[invcore][SMOKE] {key} shape={tuple(dW.shape)} trace(G)={ws.sum():.4f} "
                      f"(want ≈1.0) bottom-{k_grid[0]}={sb[str(k_grid[0])]:.3e} "
                      f"null bottom-{k_grid[0]}={w0[:k_grid[0]].mean():.3e}", flush=True)
                assert abs(ws.sum() - 1.0) < 1e-6, f"single-ft normalised Gram trace {ws.sum()} != 1"
                smoke_done = True
        del w_ft, matched
        _delete_cache(repo_id)

    # ---- PASS 2: per (type,layer) MATRIX, consensus Gram across the N fine-tunes → eig + null + verdict.
    import numpy as np
    per_matrix = []                                       # one record per (type,layer)
    for t in want_types:
        layers = sorted(next(iter(spectra[t].values())).keys())
        n_in = None
        for layer in layers:
            # consensus Gram G_M = Σ_i normalised ΔWᵢᵀΔWᵢ over the N fine-tunes of THIS matrix.
            G = None
            spec_terms = []
            for ridx, repo_id in enumerate(repos):
                f = scratch / f"{t}__L{layer}__{ridx}.npy"
                if not f.exists():
                    continue                              # a dropped ft (none expected, but be safe)
                A = np.load(f).astype(np.float64)         # m×n
                fro2 = (A ** 2).sum()
                assert fro2 > 0, f"zero ΔW scratch {f}"
                term = (A.T @ A) / fro2
                G = term if G is None else G + term
                spec_terms.append(spectra[t][repo_id][layer])
                if n_in is None:
                    n_in = A.shape[1]
            n_terms = len(spec_terms)
            eigs, real_bot, real_top = _bottom_topk_eigs(G, k_grid)
            n_null = (a.n_null_bigmlp or a.n_null) if n_in >= 8192 else a.n_null
            seed_m = (a.seed * 1_000_003 + hash(f"{t}|{layer}") % 1_000_000) % (2 ** 31)
            nb, nt, nb_samples = _null_band(spec_terms, n_in, k_grid, n_null, seed_m)
            verdict = _verdict(real_bot, nb, nb_samples, k_grid)
            per_matrix.append({
                "type": t, "layer": layer, "n_in": int(n_in), "n_terms": int(n_terms),
                "trace_G": float(eigs.sum()), "eig_min": float(eigs.min()), "eig_max": float(eigs.max()),
                "real_bottom_k": real_bot, "real_top_k": real_top,
                "null_bottom_k": nb, "n_null_used": int(n_null), "verdict_by_k": verdict,
            })
            k0 = str(k_grid[0])
            print(f"[invcore] {t} L{layer} n={n_in} N={n_terms} bottom-{k_grid[0]}: "
                  f"real={real_bot[k0]:.4e} null={nb[k0]['mean']:.4e}±{nb[k0]['std']:.1e} "
                  f"z={verdict[k0]['z']:+.2f} → {verdict[k0]['verdict']}", flush=True)
            (out / "invariant_core.json").write_text(json.dumps({**rep, "per_matrix": per_matrix}, indent=2))

    # ---- AGGREGATE: per-type and overall Stouffer z over the per-matrix z's, + BEYOND_NULL counts.
    def _agg(records, ks):
        zs = [r["verdict_by_k"][ks]["z"] for r in records if r["verdict_by_k"][ks]["z"] == r["verdict_by_k"][ks]["z"]]
        n_beyond = sum(1 for r in records if r["verdict_by_k"][ks]["verdict"] == "BEYOND_NULL_below")
        n_above = sum(1 for r in records if r["verdict_by_k"][ks]["verdict"] == "ABOVE_NULL")
        return {"stouffer_z": float(sum(zs) / (len(zs) ** 0.5)) if zs else float("nan"),
                "mean_z": float(sum(zs) / len(zs)) if zs else float("nan"),
                "n_matrices": len(records), "n_beyond_null": n_beyond, "n_above_null": n_above}

    by_type = {t: {ks: _agg([r for r in per_matrix if r["type"] == t], ks) for ks in map(str, k_grid)}
               for t in want_types}
    overall = {ks: _agg(per_matrix, ks) for ks in map(str, k_grid)}
    for k in k_grid:
        ks = str(k)
        o = overall[ks]
        print(f"[invcore] OVERALL bottom-{k}: Stouffer z={o['stouffer_z']:+.2f} mean_z={o['mean_z']:+.2f} "
              f"({o['n_beyond_null']}/{o['n_matrices']} matrices BEYOND_NULL_below, "
              f"{o['n_above_null']} ABOVE_NULL)", flush=True)
    rep["per_matrix"] = per_matrix
    rep["by_type"] = by_type
    rep["overall"] = overall
    (out / "invariant_core.json").write_text(json.dumps(rep, indent=2))

    shutil.rmtree(scratch)
    print("=" * 78)
    print(f"INVARIANT_CORE_OK n_ft={len(repos)} n_matrices={len(per_matrix)} "
          f"overall_stouffer={ {k: round(overall[k]['stouffer_z'],2) for k in map(str,k_grid)} }")


if __name__ == "__main__":
    main()
