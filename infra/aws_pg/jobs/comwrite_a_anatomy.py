"""COMWRITE Arm A — anatomy of the common write U_wr (comwrite spec §4). CPU-only (r7i),
one streaming pass over the 24-FT ΔW family (+ the π0 hop), the invariant_core two-pass
spill pattern. All Grams live on the RESIDUAL-FACING d=2048 side (comwrite_core), so every
eigendecomposition is 2048² regardless of matrix type.

  A1 structure — top-k trace share vs the invariant_core Haar band (the "context" numbers,
       now verdicted); cos(top-1 eigvec, family-mean direction); LOO stability at k_loo;
       LOO↔disc-12 overlap (pre-mortem iv).
  A3 rank ladder — sealed-half (sha1 12/12) energy capture in the discovery basis at
       k ∈ K_GRID; pooled ladder → k* (frozen ≥90%-of-k128 selector, §5.1).
  A2 star test — GAUGE-TREE CCC + triple statistic on the RESIDUAL family
       {ΔWᵢ − P_wr^LOO(i)(ΔWᵢ)} at k*, same nulls; frozen rule: residual CCC inside the null
       band ⇒ the tree-likeness was the star around U_wr (GAUGE-R3 re-scoped).
  A4 origin — overlap(U_wr(k*), residual-side top-k* eigenspace of W_PG1 / W_gemma-2b)
       vs the k*/d floor.
  A5 hop row — energy fraction of ΔW_hop = π0 − PG1 into U_wr(k*) vs matched-random spans,
       layer-bootstrap (endow_writefrac convention). Exploratory (n=1 hop), labeled.

Arm-A outputs NO behavioral claims; frozen rules gate labels only (§4). Fail-loud, no
try/except; the hop is skipped ONLY via the explicit --no-hop pivot (§10.3, disclosed).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.comwrite_a_anatomy --type r7i.4xlarge --no-weights \
      --s3-prefix results/comwrite_a_anatomy --budget-min 600 \
      --args "--ft-tasks <the frozen 24 slugs, comma-sep>"
"""
import argparse
import json
import pathlib

from infra.pg.gauge_common import PROJ, sha1_split


def _det_seed(base_seed, tag):
    """Deterministic per-(tag) seed — endow_transplant's sha1 recipe, NEVER hash()
    (process-salted, breaks re-runs)."""
    from infra.pg.endow_transplant import _seed_for
    return _seed_for(base_seed, str(tag))


def _null_top_draw(args):
    """One Haar-Gram null draw's TOP-k eigenvalue means (worker; single-threaded BLAS —
    the draws are process-parallel, the per-draw QRs must not oversubscribe)."""
    spectra, n, k_grid, seed = args
    import numpy as np
    from threadpoolctl import threadpool_limits
    from infra.aws_pg.jobs.invariant_core import _haar_gram
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(seed)
        G0 = _haar_gram(spectra, n, rng)
        w = np.clip(np.linalg.eigvalsh(G0), 0.0, None)
    return {int(k): float(w[-min(int(k), w.size):].mean()) for k in k_grid}


def null_top_band(pool, spectra, n, k_grid, n_null, seed):
    """The A1 Haar band for the TOP-k statistic, n_null draws in parallel over `pool`.
    Statistically identical to invariant_core._null_band's top output (same _haar_gram
    construction, deterministic spawned seeds); parallel because 64 draws × 24 QRs per
    (L,t) dominated the sequential run (the 2026-07-06 exit-137 restart, journal entry 6)."""
    import numpy as np
    seeds = [int(s.generate_state(1)[0] % (2 ** 31))
             for s in np.random.SeedSequence(seed).spawn(n_null)]
    draws = pool.map(_null_top_draw, [(spectra, n, k_grid, s) for s in seeds])
    band = {}
    for k in k_grid:
        v = np.asarray([d[int(k)] for d in draws], dtype=np.float64)
        band[str(k)] = {"mean": float(v.mean()), "std": float(v.std()),
                        "p99.5": float(np.percentile(v, 99.5)), "n": int(v.size)}
    return band

K_REF = 16          # the frozen reference k for the A1 existence verdict + LOO stability
Z_MARGIN = 3.0      # invariant_core convention: beyond-band needs BOTH the percentile cut and |z|≥3


def _existence(share, band):
    """A1 existence verdict for one (L,t): the top-k trace share clears the Haar band's upper
    edge with margin (types inside the band are EXCLUDED from surgery, §8)."""
    z = (share - band["mean"]) / band["std"] if band["std"] > 0 else float("inf")
    return {"share": share, "null_mean": band["mean"], "null_p99.5": band["p99.5"],
            "z": float(z), "beyond": bool(share > band["p99.5"] and z >= Z_MARGIN)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", required=True,
                    help="the frozen 24-FT roster slugs, comma-sep (GAUGE convention: explicit)")
    ap.add_argument("--types", default=",".join(PROJ))
    ap.add_argument("--k-grid", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--k-loo", type=int, default=K_REF)
    ap.add_argument("--kstar-frac", type=float, default=0.9)
    ap.add_argument("--n-null", type=int, default=32, help="Haar draws per (L,t) for the A1 band")
    ap.add_argument("--null-procs", type=int, default=0,
                    help="processes for the parallel null band (0 ⇒ cpu_count)")
    ap.add_argument("--n-null-tree", type=int, default=200, help="A2 tree null draws")
    ap.add_argument("--n-rand-hop", type=int, default=8, help="A5 matched-random spans per (L,t)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-hop", action="store_true",
                    help="§10.3 pivot: hop sd unreachable → A5 skipped + DISCLOSED")
    ap.add_argument("--no-origin-g2b", action="store_true",
                    help="skip the gemma-2b half of A4 (PG1 half always runs)")
    ap.add_argument("--smoke", action="store_true",
                    help="3 FTs, 2 types, layers 0-1, tiny nulls — pipeline shakeout only")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                        # FIRST — gated repos 401 otherwise
    import numpy as np
    from huggingface_hub import HfApi

    from infra.aws_pg.jobs.invariant_core import (
        PG1_REPO, _decoder_matrices, _delete_cache, _ftype, _load_decoder,
        _spectrum_normsq)
    from infra.pg import comwrite_core as CC
    from infra.pg import endow_writefrac as EW
    from infra.pg import gauge_tree as GT

    k_grid = tuple(int(x) for x in a.k_grid.split(","))
    want_types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    assert all(t in PROJ for t in want_types), f"unknown type in {want_types}"

    api = HfApi()
    repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    if a.smoke:
        repos = repos[:3]
        want_types = tuple(t for t in want_types if t in ("q_proj", "down_proj")) or want_types[:2]
        k_grid = tuple(k for k in k_grid if k <= 4) or (1, 2, 4)
    else:
        assert len(repos) == 24, f"frozen roster must be 24 FTs, got {len(repos)}"
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha
    disc_repos, seal_repos = sha1_split(repos, n_discovery=len(repos) // 2)
    disc_idx = [repos.index(r) for r in disc_repos]
    seal_idx = [repos.index(r) for r in seal_repos]

    rep = {"object": "comwrite_a_anatomy_v1", "base_repo": PG1_REPO, "base_sha": base_sha,
           "roster": repos, "shas": shas, "discovery": disc_repos, "sealed": seal_repos,
           "types": list(want_types), "k_grid": list(k_grid), "k_loo": a.k_loo,
           "kstar_frac": a.kstar_frac, "n_null": a.n_null, "seed": a.seed,
           "smoke": bool(a.smoke),
           "hop": "SKIPPED_DISCLOSED (--no-hop pivot, §10.3)" if a.no_hop else "enabled",
           "estimand": ("U_wr(L,t;k) = top-k eigvecs of the residual-facing per-FT-unit-Frobenius "
                        "consensus Gram; input side q/k/v/gate/up, output side o/down")}
    print(f"[comwrite-a] base={PG1_REPO}@{base_sha[:8]} roster={len(repos)} "
          f"types={want_types} k={k_grid}", flush=True)

    # ---- PASS 1: difference every FT (+ hop, + origin bases), spill fp16 ΔW per (t,L,ft).
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    import torch
    spectra = {}                             # (t, L) -> [per-FT normalised squared spectrum]
    w_base = _load_decoder(PG1_REPO, base_sha)
    layers_seen = set()
    for ridx, repo_id in enumerate(repos):
        print(f"[comwrite-a] diff {repo_id}@{shas[repo_id][:8]} ({ridx + 1}/{len(repos)})",
              flush=True)
        w_ft = _load_decoder(repo_id, shas[repo_id])
        for key, (wb, wf) in _decoder_matrices(w_base, w_ft).items():
            t = _ftype(key)
            if t not in want_types:
                continue
            L = int(key.split("layers.")[1].split(".")[0])
            if a.smoke and L > 1:
                continue
            layers_seen.add(L)
            dW = wf.to(torch.float64) - wb.to(torch.float64)
            spectra.setdefault((t, L), []).append(_spectrum_normsq(dW))
            np.save(scratch / f"{t}__L{L}__{ridx}.npy", dW.to(torch.float16).cpu().numpy())
        del w_ft
        _delete_cache(repo_id)
    layers = sorted(layers_seen)

    # the π0 hop ΔW (A5) — the same matched-matrix differencing against the SAME base.
    if not a.no_hop:
        from infra.aws_pg.jobs.endow_r_r0 import _extract_decoder
        pi0_dec, pi0_sha = _extract_decoder("pi0")
        rep["pi0_sha"] = pi0_sha
        for key, (wb, wf) in _decoder_matrices(w_base, pi0_dec).items():
            t = _ftype(key)
            L = int(key.split("layers.")[1].split(".")[0])
            if t in want_types and L in layers_seen:
                dW = wf.to(torch.float64) - wb.to(torch.float64)
                np.save(scratch / f"{t}__L{L}__hop.npy", dW.to(torch.float16).cpu().numpy())
        del pi0_dec

    # A4 origin bases: spill the BASE matrices (PG1 always; gemma-2b optionally).
    for key, wb in w_base.items():
        if any(p in key for p in want_types) and wb.ndim == 2:
            t = _ftype(key)
            L = int(key.split("layers.")[1].split(".")[0])
            if L in layers_seen:
                np.save(scratch / f"base_pg1__{t}__L{L}.npy",
                        wb.to(torch.float16).cpu().numpy())
    del w_base
    if not a.no_origin_g2b:
        from infra.pg.entities import ENTITIES
        from transformers import AutoModelForCausalLM
        ent = ENTITIES["L0_g1"]
        g2b = AutoModelForCausalLM.from_pretrained(ent.repo_id, revision=ent.revision,
                                                   torch_dtype=torch.float32)
        rep["g2b_repo"] = ent.repo_id
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in g2b.model.state_dict().items()}
        for key, wb in sd.items():
            if any(p in key for p in want_types) and wb.ndim == 2:
                t = _ftype(key)
                L = int(key.split("layers.")[1].split(".")[0])
                if L in layers_seen:
                    np.save(scratch / f"base_g2b__{t}__L{L}.npy",
                            wb.to(torch.float16).cpu().numpy())
        del g2b, sd
        _delete_cache(ent.repo_id)

    # ---- PASS 2: per (L,t) — A1 structure + A3 ladder + LOO stability; pooled ladder → k*.
    def _load_dws(t, L):
        return [np.load(scratch / f"{t}__L{L}__{i}.npy").astype(np.float64)
                for i in range(len(repos))]

    import multiprocessing as mp
    import os
    pool = mp.Pool(a.null_procs or os.cpu_count())
    per_matrix = []
    pooled_ladder = {int(k): [] for k in k_grid}
    for t in want_types:
        for L in layers:
            dws = _load_dws(t, L)
            terms = [CC.gram_term(dW, t) for dW in dws]
            G = terms[0].copy()
            for T in terms[1:]:
                G += T
            d = G.shape[0]
            k_top = min(max(k_grid), d)

            # A1: top-k trace share vs the Haar band (share = mean-top-k·k/trace ↔ invariant_core
            # top-k mean; we band the SHARE directly from the same _null_band draws).
            seed_m = _det_seed(a.seed, f"{t}|{L}")
            nt_band = null_top_band(pool, spectra[(t, L)], d, k_grid, a.n_null, seed_m)
            wG, VG = np.linalg.eigh(G)                   # ONE eigendecomposition serves A1+U_full
            prof = np.clip(wG, 0.0, None)[::-1]
            tr = float(prof.sum())
            shares = {int(k): float(prof[:k].sum()) / tr for k in k_grid}
            n_ft = len(dws)
            a1 = {}
            for k in k_grid:
                b = nt_band[str(k)]
                # convert the null's mean-top-k eigenvalue band to a trace-share band (trace = n_ft).
                band = {"mean": b["mean"] * k / n_ft, "std": b["std"] * k / n_ft,
                        "p99.5": b["p99.5"] * k / n_ft}
                a1[int(k)] = _existence(shares[int(k)], band)
            kr = max([k for k in k_grid if k <= K_REF] or [min(k_grid)])   # a grid member ≤ 16
            exists = a1[kr]["beyond"]

            # mean-direction cosine (H_star-rank1 comparator). The SAME eigh serves U_full.
            U_full_lt = VG[:, ::-1][:, :k_top].copy()
            np.save(scratch / f"ufull__{t}__L{L}.npy", U_full_lt.astype(np.float32))
            mcos = CC.abs_cos(U_full_lt[:, 0], CC.mean_direction(dws, t))

            # A3: sealed capture in the DISCOVERY basis.
            G_disc = terms[disc_idx[0]].copy()
            for i in disc_idx[1:]:
                G_disc += terms[i]
            U_disc = CC.u_wr(G_disc, k_top)
            ladder = CC.concentration_ladder([(dws[i], t) for i in seal_idx], U_disc,
                                             k_grid=[k for k in k_grid if k <= k_top])
            for k, v in ladder.items():
                pooled_ladder[int(k)].append(v)

            # LOO stability at k_loo + LOO↔disc-12 (pre-mortem iv). Spill LOO bases for PASS 3.
            k_loo = min(a.k_loo, k_top)
            loo = []
            for i in range(n_ft):
                Ui = CC.u_wr(CC.gram_minus(G, terms[i]), k_top)
                np.save(scratch / f"loo__{t}__L{L}__{i}.npy", Ui.astype(np.float32))
                loo.append(Ui[:, :k_loo])
            pair = [CC.subspace_overlap(loo[i], loo[j])
                    for i in range(n_ft) for j in range(i + 1, n_ft)]
            loo_disc = [CC.subspace_overlap(U_disc[:, :k_loo], Ui) for Ui in loo]

            per_matrix.append({
                "type": t, "layer": L, "d": int(d), "n_ft": n_ft,
                "a1_share_by_k": {str(k): v for k, v in a1.items()},
                "exists_at_kref": bool(exists), "k_ref": int(kr),
                "mean_dir_cos_top1": float(mcos),
                "a3_sealed_ladder": {str(k): v for k, v in ladder.items()},
                "chance_floor": {str(k): CC.chance_floor(k, d)
                                 for k in k_grid if k <= k_top},
                "loo_stability_mean": float(np.mean(pair)),
                "loo_stability_floor": CC.chance_floor(k_loo, d),
                "loo_vs_disc12_mean": float(np.mean(loo_disc)), "k_loo": int(k_loo)})
            print(f"[comwrite-a] {t} L{L} share@{kr}={shares[kr]:.3f} "
                  f"(null p99.5={a1[kr]['null_p99.5']:.3f} z={a1[kr]['z']:+.1f}) "
                  f"exists={exists} mcos={mcos:.3f} loo={np.mean(pair):.3f}", flush=True)
            (out / "metrics.json").write_text(json.dumps(
                {**rep, "per_matrix": per_matrix}, indent=2))

    pool.close()
    pool.join()
    ladder_mean = {str(k): float(np.mean(v)) for k, v in pooled_ladder.items() if v}
    kstar = CC.k_star({int(k): v for k, v in ladder_mean.items()}, frac=a.kstar_frac)
    excluded = [[m["layer"], m["type"]] for m in per_matrix if not m["exists_at_kref"]]
    rep.update({"a3_pooled_ladder": ladder_mean, "k_star": int(kstar),
                "excluded_lt": excluded})
    print(f"[comwrite-a] pooled ladder {ladder_mean} → k*={kstar}; "
          f"excluded {len(excluded)}/{len(per_matrix)}", flush=True)

    # ---- PASS 3: A2 residual-family tree (LOO at k*) + A4 origin + A5 hop.
    n = len(repos)
    G_raw = np.zeros((n, n))
    G_res = np.zeros((n, n))
    a4, a5 = [], []
    for t in want_types:
        for L in layers:
            dws = _load_dws(t, L)
            k_use = min(kstar, dws[0].shape[1], dws[0].shape[0])
            lt_excluded = [L, t] in excluded
            R = []
            for i in range(n):
                if lt_excluded:
                    R.append(dws[i])                       # no commons here (§8): residual = ΔW
                else:
                    Ui = np.load(scratch / f"loo__{t}__L{L}__{i}.npy").astype(np.float64)[:, :k_use]
                    if CC.residual_side(t) == "input":
                        R.append(dws[i] - (dws[i] @ Ui) @ Ui.T)
                    else:
                        R.append(dws[i] - Ui @ (Ui.T @ dws[i]))
            dflat = [dW.ravel() for dW in dws]           # views — np.dot avoids the 256MB
            rflat = [Ri.ravel() for Ri in R]             # elementwise-product temporaries
            for i in range(n):
                for j in range(i, n):
                    G_raw[i, j] = G_raw[j, i] = G_raw[i, j] + float(np.dot(dflat[i], dflat[j]))
                    G_res[i, j] = G_res[j, i] = G_res[i, j] + float(np.dot(rflat[i], rflat[j]))

            if not lt_excluded:
                U_full = np.load(scratch / f"ufull__{t}__L{L}.npy").astype(np.float64)[:, :k_use]
                d = U_full.shape[0]
                # A4 origin: residual-side top-k* eigenspace of the base matrices.
                rec = {"type": t, "layer": L, "k": int(k_use),
                       "floor": CC.chance_floor(k_use, d)}
                for tag in ("pg1",) + (() if a.no_origin_g2b else ("g2b",)):
                    Wb = np.load(scratch / f"base_{tag}__{t}__L{L}.npy").astype(np.float64)
                    Ub = CC.u_wr(CC.gram_term(Wb, t), k_use)
                    rec[f"overlap_{tag}"] = CC.subspace_overlap(U_full, Ub)
                a4.append(rec)
                # A5 hop row.
                if not a.no_hop:
                    hop = np.load(scratch / f"{t}__L{L}__hop.npy").astype(np.float64)
                    rng_seed = _det_seed(a.seed, f"hop|{t}|{L}")
                    rnd = [CC.energy_fraction(
                        hop, np.linalg.qr(np.random.default_rng(rng_seed + i)
                                          .standard_normal((d, k_use)))[0], t)
                        for i in range(a.n_rand_hop)]
                    a5.append({"type": t, "layer": L, "k": int(k_use),
                               "hop_frac": CC.energy_fraction(hop, U_full, t),
                               "rand_mean": float(np.mean(rnd)),
                               "rand_draws": [float(x) for x in rnd],
                               "floor": CC.chance_floor(k_use, d)})
        print(f"[comwrite-a] pass3 {t} done", flush=True)

    # A2 verdict: intact vs residual family tree under the SAME nulls (frozen rule §4-A2).
    # Three-way on the residual: inside BOTH bands → star confirmed; beyond BOTH → tree
    # structure beyond the commons; beyond exactly ONE → AMBIGUOUS, logged as such (never
    # collapsed into the narrative-friendly star branch).
    tree_raw = GT.ccc_with_nulls(GT.gram_to_distance(G_raw), n_null=a.n_null_tree, seed=a.seed)
    tree_res = GT.ccc_with_nulls(GT.gram_to_distance(G_res), n_null=a.n_null_tree,
                                 seed=a.seed + 17)
    res_beats = int(tree_res["beats_cloud"]) + int(tree_res["beats_perm"])
    if not GT.beats_both(tree_raw):
        verdict = "intact_family_not_tree_like (A2 premise absent)"
    elif res_beats == 0:
        verdict = "H_star_topology_CONFIRMED (residual CCC inside both null bands; R3 re-scoped)"
    elif res_beats == 2:
        verdict = "tree_structure_beyond_commons (exploratory object, logged)"
    else:
        verdict = ("AMBIGUOUS_ONE_NULL (residual beats exactly one null — star NOT "
                   "confirmed, structure NOT established; logged)")
    rep["a2"] = {"intact": tree_raw, "residual": tree_res, "verdict": verdict}

    # A5 layer-bootstrap per type (endow_writefrac convention) — the J3 companion row.
    if a5:
        hop_rows = {}
        for t in want_types:
            per_layer = {r["layer"]: r["hop_frac"] for r in a5 if r["type"] == t}
            rnd_layer = {r["layer"]: r["rand_mean"] for r in a5 if r["type"] == t}
            if len(per_layer) >= 2:
                hop_rows[t] = {"hop": EW.layer_bootstrap(per_layer, seed=a.seed),
                               "rand": EW.layer_bootstrap(rnd_layer, seed=a.seed)}
        rep["a5"] = {"per_matrix": a5, "by_type": hop_rows,
                     "label": "EXPLORATORY (n=1 hop, §4-A5)"}

    rep["a4"] = a4
    rep["per_matrix"] = per_matrix
    (out / "metrics.json").write_text(json.dumps(rep, indent=2))
    print(f"[comwrite-a] DONE k*={kstar} a2={rep['a2']['verdict']}", flush=True)


if __name__ == "__main__":
    main()
