"""GAUGE-CS (P1, the MDE) — basis-coherence scan over the 24-FT PG1 ΔW family (CPU, --no-weights).

Streams the 24 fine-tunes exactly as invariant_core.py does (enumerate -224 live, pin SHAs, load
base once, difference per decoder matrix in fp64, fp16 scratch, delete cache). Per (matrix-type,
layer): builds the consensus bases on the DISCOVERY 12 only, then per FT computes concentration in
each basis class + the spectrum-matched null band + the entry-permutation control, accumulates the
per-FT excesses, and adjudicates discovery-clear (§4.4). The sealed-12 confirmation of promoted
cells + the L1 recovery leg run only if a cell clears (autonomous; frozen rules).

  # smoke (2 ft, q_proj only):
  pg run infra.aws_pg.jobs.gauge_cs --type r7i.4xlarge --no-weights \
      --args "--ft-tasks vqav2,gqa --types q_proj --n-null 32" --budget-min 60
  # full (default = the 4 attention types, max dim 2048; MLP-wide types need --dim-cap 16384):
  pg run infra.aws_pg.jobs.gauge_cs --type r7i.8xlarge --no-weights \
      --args "--types q_proj,k_proj,v_proj,o_proj --n-null 48" --budget-min 600
"""
import argparse
import json
import pathlib

from infra.pg import gauge_common as GC
from infra.pg import gauge_cs as CS
from infra.pg import relaw as RL          # bh_fdr for the sealed-confirmation BH step (§4.4)
# REUSED streaming spine (invariant_core.py): base/ft decoder load, matrix matching, cache delete.
from infra.aws_pg.jobs.invariant_core import (PG1_REPO, PROJ, _load_decoder, _decoder_matrices,
                                              _ftype, _delete_cache)

FIXED_CLASSES = ("entry", "base_svd", "consensus", "random_k")   # dW_svd is reference-only (§4.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="", help="comma slugs; empty ⇒ enumerate all -224 (24-FT)")
    # Default = the four attention types (max dim 2048): both singular bases are fully defined and
    # the two-sided projection is affordable. The 16384-wide MLP types (gate/up/down) are excluded
    # by default because a full orthogonal completion at 16384 is a 2 GB factor + an arbitrary
    # large-side basis (W_base is rank ≤ min-dim there); they need the economy-basis variant and an
    # explicit --dim-cap opt-in (staged like invariant_core did q/k/v/o then the MLP arm). Disclosed.
    ap.add_argument("--types", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--dim-cap", type=int, default=4096,
                    help="fail loud if a matrix's max dim exceeds this (explicit two-sided bases)")
    ap.add_argument("--n-null", type=int, default=48, help="spectrum-matched null draws per matrix")
    ap.add_argument("--n-random", type=int, default=8, help="K seeded Haar probes (random_k class)")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want_types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    assert all(t in PROJ for t in want_types), f"unknown type in {want_types}"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    api = HfApi()

    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha
    # deterministic 12/12 split (Gate-0-frozen; depends only on repo-ids).
    discovery, confirmation = GC.sha1_split(repos) if len(repos) >= GC.N_DISCOVERY else (repos, [])
    disc_idx = {r: i for i, r in enumerate(repos)}
    print(f"[gauge-cs] base={PG1_REPO}@{base_sha[:8]} n_ft={len(repos)} "
          f"discovery={len(discovery)} sealed={len(confirmation)} types={want_types}", flush=True)

    rep = {"object": "gauge_cs_v1", "generation": "gemma1_18L", "base_repo": PG1_REPO,
           "base_sha": base_sha, "cloud_repos": repos, "cloud_shas": shas,
           "discovery": discovery, "confirmation": confirmation,
           "classes": list(CS.BASIS_CLASSES), "fixed_classes": list(FIXED_CLASSES),
           "n_null": a.n_null, "n_random": a.n_random, "seed": a.seed,
           "estimand": "family-median excess of coefficient-Gini over the spectrum-matched null, "
                       "per (matrix-type, basis class); null basis-independent (rotation invariance)"}

    w_base = _load_decoder(PG1_REPO, base_sha)

    # ---- PASS 1: difference every FT, store per-(type,layer,ft) fp16 ΔW scratch + spectrum. ----
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    for ridx, repo_id in enumerate(repos):
        print(f"[gauge-cs] diff {repo_id}@{shas[repo_id][:8]} ({ridx+1}/{len(repos)})", flush=True)
        w_ft = _load_decoder(repo_id, shas[repo_id])
        matched = _decoder_matrices(w_base, w_ft)
        if ridx == 0:
            GC.assert_gemma1(list(matched.keys()))          # firewall (18-layer)
        for key, (wb, wf) in matched.items():
            t = _ftype(key)
            if t not in want_types:
                continue
            layer = GC.layer_index(key)
            dW = (wf.to(torch.float64) - wb.to(torch.float64)).cpu().numpy()
            np.save(scratch / f"{t}__L{layer}__{ridx}.npy", dW.astype(np.float16))
        del w_ft, matched
        _delete_cache(repo_id)

    # base-SVD bases depend only on W_base — cache per (type,layer).
    base_np = {k: v.to(torch.float64).cpu().numpy() for k, v in w_base.items()}

    # ---- PASS 2: per (type,layer) MATRIX — consensus (discovery only), then per-FT concentration. ----
    per_cell = []                       # one record per (type, layer, class)
    for t in want_types:
        layers = sorted({int(f.stem.split("__L")[1].split("__")[0])
                         for f in scratch.glob(f"{t}__L*__0.npy")})
        for layer in layers:
            key = f"layers.{layer}.{'self_attn' if t[0] in 'qkvo' else 'mlp'}.{t}.weight"
            def _load(ridx):
                f = scratch / f"{t}__L{layer}__{ridx}.npy"
                return np.load(f).astype(np.float64) if f.exists() else None
            disc_dws = [(_load(disc_idx[r]), r) for r in discovery]
            disc_dws = [(A, r) for A, r in disc_dws if A is not None]
            m0, n0 = disc_dws[0][0].shape
            assert max(m0, n0) <= a.dim_cap, (
                f"{t} L{layer} max dim {max(m0, n0)} > --dim-cap {a.dim_cap}: the explicit two-sided "
                f"basis classes need the economy-basis MLP variant; raise --dim-cap to opt in")
            U_cons, V_cons = CS.consensus_bases([A for A, _ in disc_dws])   # discovery-only
            Wb = base_np[key]
            # cell-level objects, computed ONCE (FT-independent — §4.3 null is per (type,class), and
            # base-SVD / random-K bases do not vary across fine-tunes): the pooled null band over the
            # discovery FTs' real spectra, the base-SVD bases, and the K seeded Haar probe bases.
            seed_cell = (a.seed * 1_000_003 + hash(f"{t}|{layer}") % 1_000_000) % (2 ** 31)
            disc_specs = [np.linalg.svd(A, compute_uv=False) for A, _ in disc_dws]
            band = CS.null_band_pooled(disc_specs, m0, n0, a.n_null, seed_cell)
            U_bsvd, V_bsvd = CS.svd_bases(Wb)
            rand_bases = [(CS.random_ortho(m0, seed_cell + 2 * j + 1),
                           CS.random_ortho(n0, seed_cell + 2 * j + 2)) for j in range(a.n_random)]
            cell_kw = dict(base_svd_bases=(U_bsvd, V_bsvd), consensus=(U_cons, V_cons),
                           random_bases=rand_bases)

            # per-class per-FT excess vs the shared cell null-band median (§4.2 ĉ_B summand).
            def _excesses(repo_ids):
                acc = {c: {"g": [], "p": []} for c in FIXED_CLASSES}
                perm = {c: {"g": [], "p": []} for c in FIXED_CLASSES}
                for r in repo_ids:
                    A = _load(disc_idx[r])
                    if A is None:
                        continue
                    Ap = CS.permute_entries(A, seed_cell + hash(r) % 9973 + 1)
                    for c in FIXED_CLASSES:
                        cr = CS.concentration_in_class(A, c, **cell_kw)
                        cp = CS.concentration_in_class(Ap, c, **cell_kw)
                        acc[c]["g"].append(CS.excess(cr["gini"], band["gini"]["p50"]))
                        acc[c]["p"].append(CS.excess(cr["pr"], band["pr"]["p50"]))
                        perm[c]["g"].append(CS.excess(cp["gini"], band["gini"]["p50"]))
                        perm[c]["p"].append(CS.excess(cp["pr"], band["pr"]["p50"]))
                return acc, perm, band

            acc, perm, band = _excesses(discovery)
            # δ_CS = widest per-cell null half-width on discovery controls (band p97.5−p50).
            delta_cs = float(band["gini"]["p97.5"] - band["gini"]["p50"])
            for c in FIXED_CLASSES:
                real_ci = CS.median_excess_ci(acc[c]["g"])
                perm_ci = CS.median_excess_ci(perm[c]["g"])
                sec = CS.median_excess_ci(acc[c]["p"])       # PR excess (secondary; sign check)
                cleared = CS.discovery_clear(real_ci["ci_lo"], delta_cs, perm_ci["ci_lo"])
                per_cell.append({
                    "type": t, "layer": layer, "class": c, "n_disc": len(acc[c]["g"]),
                    "delta_cs": delta_cs, "gini_excess": real_ci, "pr_excess_secondary": sec,
                    "perm_gini_excess": perm_ci, "discovery_clear": cleared,
                    "null_gini_p50": band["gini"]["p50"], "null_gini_p97.5": band["gini"]["p97.5"]})
                print(f"[gauge-cs] {t} L{layer} {c:9s} gini_excess CI-lo={real_ci['ci_lo']:+.4f} "
                      f"δ={delta_cs:.4f} perm_lo={perm_ci['ci_lo']:+.4f} → clear={cleared}", flush=True)
            (out / "gauge_cs.json").write_text(json.dumps({**rep, "per_cell": per_cell}, indent=2))

    # ---- promote ≤3 discovery-cleared cells → sealed confirmation (§4.4). ----
    cleared_cells = [c for c in per_cell if c["discovery_clear"]]
    cleared_cells.sort(key=lambda c: -c["gini_excess"]["ci_lo"])
    promoted = cleared_cells[:3]
    rep["per_cell"] = per_cell
    rep["n_discovery_cleared"] = len(cleared_cells)
    rep["promoted"] = [{"type": c["type"], "layer": c["layer"], "class": c["class"]} for c in promoted]

    # ---- SEALED confirmation of the promoted cells (§4.4): recompute their excess on the SEALED 12
    #      FTs (out-of-sample — the consensus basis is fit on discovery only, so this is the leakage
    #      test), then DETECTION = sealed CI-lo > δ_CS under BH(q=0.10) over the ≤3 promoted cells,
    #      secondary (PR) agreeing in sign. Reuses the in-run scratch (no re-download). ----
    sealed_records, sealed_p, sealed_meta = [], [], []
    for cell in (promoted if confirmation else []):     # sealed needs a non-empty held-out half
        t, layer, cls = cell["type"], cell["layer"], cell["class"]
        key = f"layers.{layer}.{'self_attn' if t[0] in 'qkvo' else 'mlp'}.{t}.weight"

        def _ld(r):
            f = scratch / f"{t}__L{layer}__{disc_idx[r]}.npy"
            return np.load(f).astype(np.float64) if f.exists() else None
        disc_A = [A for A in (_ld(r) for r in discovery) if A is not None]
        m0, n0 = disc_A[0].shape
        seed_cell = (a.seed * 1_000_003 + hash(f"{t}|{layer}") % 1_000_000) % (2 ** 31)
        U_cons, V_cons = CS.consensus_bases(disc_A)                 # discovery-only (leakage guard)
        U_bsvd, V_bsvd = CS.svd_bases(base_np[key])
        rand_bases = [(CS.random_ortho(m0, seed_cell + 2 * j + 1),
                       CS.random_ortho(n0, seed_cell + 2 * j + 2)) for j in range(a.n_random)]
        cell_kw = dict(base_svd_bases=(U_bsvd, V_bsvd), consensus=(U_cons, V_cons), random_bases=rand_bases)
        band = CS.null_band_pooled([np.linalg.svd(A, compute_uv=False) for A in disc_A],
                                   m0, n0, a.n_null, seed_cell)
        delta_cs = float(band["gini"]["p97.5"] - band["gini"]["p50"])
        g_exc, p_exc = [], []
        for r in confirmation:
            A = _ld(r)
            if A is None:
                continue
            cr = CS.concentration_in_class(A, cls, **cell_kw)
            g_exc.append(CS.excess(cr["gini"], band["gini"]["p50"]))
            p_exc.append(CS.excess(cr["pr"], band["pr"]["p50"]))
        sci, sec = CS.median_excess_ci(g_exc), CS.median_excess_ci(p_exc)
        sealed_p.append(CS.bootstrap_p_le(g_exc, delta_cs))
        sealed_meta.append((cell, sci, sec, delta_cs, len(g_exc)))
    reject = RL.bh_fdr(sealed_p, q=0.10).tolist() if sealed_p else []
    for i, (cell, sci, sec, delta_cs, n_seal) in enumerate(sealed_meta):
        # secondary (participation-ratio) AGREES with Gini on concentration when it is confidently
        # NEGATIVE: concentration raises Gini but LOWERS PR (they are anti-correlated), so the earlier
        # check `sec ci_lo > 0` was degenerate (could never fire for a real concentration). Corrected:
        # PR-excess ci_hi < 0 ⇒ effective participation dropped ⇒ concentrated, same as Gini > δ.
        det = CS.detection(sci["ci_lo"], delta_cs, bool(reject[i]), sec["ci_hi"] < 0)
        sealed_records.append({**cell, "n_sealed": n_seal, "sealed_gini_excess": sci,
                               "sealed_pr_excess_secondary": sec, "delta_cs": delta_cs,
                               "bh_p": sealed_p[i], "bh_ok": bool(reject[i]), "detection": det})
        print(f"[gauge-cs][SEALED] {cell['type']} L{cell['layer']} {cell['class']}: "
              f"sealed CI-lo={sci['ci_lo']:+.4f} δ={delta_cs:.4f} p={sealed_p[i]:.3f} "
              f"bh_ok={bool(reject[i])} → DETECTION={det}", flush=True)
    n_detect = sum(1 for r in sealed_records if r["detection"])
    rep["sealed_confirmation"] = sealed_records
    rep["n_detections"] = n_detect
    rep["verdict"] = ("H_hidden-basis (≥1 confirmed detection)" if n_detect >= 1 else
                      "H_no-hidden-basis (0 confirmed detections over the scanned classes at n=24/12-12)")
    rep["note"] = ("detection = discovery-clear ∧ sealed CI-lo>δ_CS ∧ BH(q=.10) ∧ secondary-sign; "
                   "consensus cells test family-basis structure OUT-OF-SAMPLE (leakage guard)")
    (out / "gauge_cs.json").write_text(json.dumps(rep, indent=2))
    print("=" * 78)
    print(f"GAUGE_CS_OK n_ft={len(repos)} n_cells={len(per_cell)} "
          f"discovery_cleared={len(cleared_cells)} promoted={len(promoted)} detections={n_detect} "
          f"verdict={rep['verdict']}")


if __name__ == "__main__":
    main()
