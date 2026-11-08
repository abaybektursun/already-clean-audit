"""ERLANGEN Arm B prerequisite — the consensus untouched-core basis, recomputed here (CPU).

Arm B contrasts the training gradient's energy inside the subspace adaptations avoid against a
matched-random subspace. That needs Π_core, and the 24-child core from `invariant_core` is not
banked as a basis — only its per-ft ΔW scratch is. Rather than depend on an artifact that does not
exist in the form needed, the basis is RECOMPUTED here from the five M-task fine-tunes, by the
invariant_core estimand verbatim:

    G_m = Σ_i ΔW_{i,m}ᵀ ΔW_{i,m} / ‖ΔW_{i,m}‖²_F     (per q/k matrix m, per-ft unit-Frobenius)
    U_unt(m) = the bottom-k eigenvectors of G_m       (the input directions every ft avoids)

SCOPE, stated in the artifact and carried into every downstream claim: this is a **5-fine-tune
consensus core**, not the 24-child core. It is the same estimand over a smaller, explicitly-named
population — never described as the invariant core itself.

The matched-random null is drawn per matrix (per-matrix seeds, N1) and the BASE-matrix control is
computed alongside: the same statistic on PG1's own q/k matrices, so a band the base also fails is
visible as census-only.

    python -m infra.aws_pg.jobs.erlangen_basis --out DIR --fts repo1,repo2,... [--k 32]

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

QK = ("self_attn.q_proj", "self_attn.k_proj")
N_LAYERS = 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fts", required=True, help="comma-separated PG1 fine-tune repos")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-null", type=int, default=64)
    ap.add_argument("--c6a", action="store_true",
                    help="CORE-PROVENANCE C6a: project the INSTALL write (PG1 − gemma-2b) onto the "
                         "consensus core and compare against a Frobenius-matched random band plus "
                         "a base-matrix control")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration

    from infra.pg.lacuna_shell import PG1_REPO, canon_map

    t0 = time.time()
    repos = [r.strip() for r in a.fts.split(",") if r.strip()]
    api = HfApi()
    rep = {"object": "erlangen_basis_v1", "k": a.k, "fts": repos,
           "shas": {r: api.model_info(r).sha for r in repos},
           "pg1_sha": api.model_info(PG1_REPO).sha,
           "estimand": "bottom-k eigvecs of Σ_i ΔWᵀΔW/‖ΔW‖²_F over the listed fine-tunes",
           "scope": "a 5-fine-tune consensus core — NOT the 24-child invariant core"}

    def bank():
        (out / "erlangen_basis.json").write_text(json.dumps(rep, indent=2))

    bank()

    def purge(repo):
        """Delete a checkpoint's HF cache once it has been differenced. 24 PaliGemma checkpoints at
        ~6 GB each overflow the 200 GB root — the first full-roster run died at 14/24 with ENOSPC.
        `invariant_core` carries the same discipline for the same reason."""
        import os
        import shutil
        cache = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")
        d = pathlib.Path(cache) / ("models--" + repo.replace("/", "--"))
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    def qk_of(repo, sha=None):
        m = PaliGemmaForConditionalGeneration.from_pretrained(
            repo, revision=sha, torch_dtype=torch.float32, device_map="cpu",
            attn_implementation="eager")
        sd = canon_map({k: v.detach().cpu() for k, v in m.state_dict().items()})
        del m
        keys = [f"language_model.layers.{L}.{t}.weight" for L in range(N_LAYERS) for t in QK]
        got = {k: sd[k].clone() for k in keys}
        del sd
        return got

    base = qk_of(PG1_REPO, rep["pg1_sha"])
    print(f"[basis] PG1 q/k loaded: {len(base)} matrices ({time.time()-t0:.0f}s)", flush=True)

    d = base[next(iter(base))].shape[1]
    grams = {k: torch.zeros(d, d, dtype=torch.float64) for k in base}
    n_ft = 0
    rep["per_ft_frob"] = {}
    for r in repos:
        w = qk_of(r, rep["shas"][r])
        fr = {}
        for k, v in base.items():
            dW = (w[k] - v).double()
            f2 = float((dW ** 2).sum())
            assert f2 > 0, f"{r}/{k}: zero ΔW — this is not a fine-tune of PG1"
            grams[k] += (dW.T @ dW) / f2
            fr[k] = f2 ** 0.5
        rep["per_ft_frob"][r] = {"total": float(sum(x ** 2 for x in fr.values()) ** 0.5)}
        n_ft += 1
        del w
        purge(r)
        print(f"[basis] {r} differenced ({n_ft}/{len(repos)}, {time.time()-t0:.0f}s)", flush=True)
        bank()

    # bottom-k eigvectors per matrix + the matched-random and base-matrix controls
    rng = np.random.default_rng(20260803)
    U, T = {}, {}
    stats = {}
    rep["stats"] = stats
    for k, G in grams.items():
        ev, evec = torch.linalg.eigh(G)               # ascending
        U[k] = evec[:, : a.k].contiguous().float()    # bottom-k = the avoided directions
        T[k] = evec[:, -a.k:].contiguous().float()
        lam = ev.numpy()
        # the band: mean bottom-k eigenvalue vs matched-random k-subspaces of the SAME Gram
        rand = []
        for s in range(a.n_null):
            A = torch.from_numpy(rng.standard_normal((d, a.k))).double()
            Q, _ = torch.linalg.qr(A)
            rand.append(float(torch.einsum("ij,jk,ki->", Q.T, G, Q) / a.k))
        # CRITIC FINDING 4. Under MQA k_proj is [256, 2048], so each ΔWᵀΔW has rank ≤ 256 and the
        # 5-FT sum has rank ≤ 1280 < 2048 ⇒ ≥768 exactly-zero eigenvalues. The "bottom-k" is then
        # an arbitrary LAPACK-order selection from an exact KERNEL, and `below_null` has no failure
        # mode. Counting the numerical kernel makes that visible instead of inferable.
        tol = float(max(lam.max(), 1e-300)) * d * 2.220446049250313e-16
        n_kernel = int((lam <= tol).sum())
        stats[k] = {"n_kernel_eigs": n_kernel, "kernel_tol": tol,
                    "bottom_k_inside_kernel": bool(lam[: a.k].max() <= tol),
                    "gate_can_fail": bool(lam[: a.k].max() > tol),
                    "bottom_mean": float(lam[: a.k].mean()), "top_mean": float(lam[-a.k:].mean()),
                    "trace_over_d": float(lam.mean()),
                    "rand_mean": float(np.mean(rand)), "rand_p2_5": float(np.percentile(rand, 2.5)),
                    "below_null": bool(lam[: a.k].mean() < np.percentile(rand, 2.5)),
                    "top_over_bottom": float(lam[-a.k:].mean() / max(lam[: a.k].mean(), 1e-30))}
    rep["n_matrices"] = len(U)
    rep["n_below_null"] = sum(1 for v in stats.values() if v["below_null"])
    rep["n_gate_can_fail"] = sum(1 for v in stats.values() if v["gate_can_fail"])
    rep["n_bottom_k_inside_kernel"] = sum(1 for v in stats.values() if v["bottom_k_inside_kernel"])
    rep["by_projection"] = {}
    for proj in ("q_proj", "k_proj"):
        sub = {k: v for k, v in stats.items() if k.endswith(proj + ".weight")}
        rep["by_projection"][proj] = {
            "n": len(sub),
            "n_gate_can_fail": sum(1 for v in sub.values() if v["gate_can_fail"]),
            "median_bottom_mean": float(np.median([v["bottom_mean"] for v in sub.values()])),
            "median_rand_mean": float(np.median([v["rand_mean"] for v in sub.values()])),
            "median_n_kernel_eigs": float(np.median([v["n_kernel_eigs"] for v in sub.values()]))}
    rep["unfailable_null_note"] = (
        "where bottom_k_inside_kernel is true the below-null gate CANNOT fail — the bottom-k is "
        "drawn from an exact numerical kernel, not from a measured low-energy band "
        "(critic finding 4; fourth unfailable-null instance in this lineage)")
    print(f"[basis] gate can fail on {rep['n_gate_can_fail']}/{len(U)} matrices; "
          f"{rep['n_bottom_k_inside_kernel']} have their bottom-k inside a numerical kernel",
          flush=True)
    print(f"[basis] {rep['n_below_null']}/{len(U)} matrices have a bottom-k band BELOW the "
          f"matched-random null", flush=True)

    np.savez_compressed(out / "erlangen_core_basis.npz",
                        **{k.replace(".", "__"): U[k].numpy() for k in U})
    np.savez_compressed(out / "erlangen_top_basis.npz",
                        **{k.replace(".", "__"): T[k].numpy() for k in T})
    rep["basis_file"] = "erlangen_core_basis.npz"
    rep["top_basis_file"] = "erlangen_top_basis.npz"
    rep["top_basis_note"] = ("the TOP-k eigenspace of the same consensus Gram — a STRUCTURED "
                             "control for Arm B: a Haar-random null only shows that the core is "
                             "activation-aligned, not that it differs from other aligned subspaces")
    # ---- CORE-PROVENANCE C6a: does the INSTALL write avoid the core too? -----------------------
    # Never executed in three attempts (critic finding 19). H_inherited predicts the gemma-2b→PG1
    # install ALSO avoids the core (energy below the matched-random band); H_settled predicts it
    # wrote there. FIX C4's flatness note cuts the safe way: a spectrally flat write puts ≈k/d of
    # its energy in ANY fixed k-subspace, so flatness pushes the ratio TOWARD the random band —
    # a BELOW-band reading is conservative against that confound.
    if a.c6a:
        from transformers import AutoModelForCausalLM
        g1 = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype=torch.float32)
        g1sd = {f"language_model.{k}": v.detach().cpu()
                for k, v in g1.state_dict().items()}
        del g1
        c6a, ratios = {}, []
        rep["c6a"] = c6a
        for k, Wb in base.items():
            src = k.replace("language_model.", "language_model.model.")
            cand = [src, k, k.replace("language_model.", "language_model.")]
            key = next((c for c in cand if c in g1sd), None)
            if key is None:
                key = next((x for x in g1sd if x.endswith(k.split("language_model.")[1])), None)
            assert key is not None, f"gemma-2b lacks a counterpart for {k}; have e.g. {sorted(g1sd)[:3]}"
            dW = (Wb - g1sd[key]).double()
            f2 = float((dW ** 2).sum())
            assert f2 > 0, f"{k}: zero install write"
            Uk = U[k].double()
            e_core = float((dW @ Uk).pow(2).sum()) / f2
            # Frobenius-matched random band on the SAME write, per-matrix seeds (N1)
            band = []
            gsd = np.random.default_rng(4242 + abs(hash(k)) % 1000)
            for _ in range(32):
                A = torch.from_numpy(gsd.standard_normal((d, a.k)))
                Q, _ = torch.linalg.qr(A)
                band.append(float((dW @ Q).pow(2).sum()) / f2)
            # BASE-MATRIX control: the same statistic on PG1's own weight matrix
            b2 = float((Wb.double() ** 2).sum())
            e_base = float((Wb.double() @ Uk).pow(2).sum()) / b2
            c6a[k] = {"install_energy_frac_in_core": e_core,
                      "rand_band_mean": float(np.mean(band)),
                      "rand_band_p2_5": float(np.percentile(band, 2.5)),
                      "below_band": bool(e_core < np.percentile(band, 2.5)),
                      "base_matrix_energy_frac": e_base,
                      "base_also_below": bool(e_base < np.percentile(band, 2.5)),
                      "ratio_install_over_rand": e_core / max(float(np.mean(band)), 1e-30)}
            ratios.append(c6a[k]["ratio_install_over_rand"])
        n_below = sum(1 for v in c6a.values() if v["below_band"])
        n_base_below = sum(1 for v in c6a.values() if v["base_also_below"])
        rep["c6a_summary"] = {
            "n_matrices": len(c6a), "n_below_band": n_below,
            "n_base_matrices_also_below": n_base_below,
            "median_ratio_install_over_rand": float(np.median(ratios)),
            "ratio_of_medians": float(np.median([v["install_energy_frac_in_core"] for v in c6a.values()])
                                      / max(np.median([v["rand_band_mean"] for v in c6a.values()]), 1e-30)),
            "verdict": ("H_inherited-leg (the install AVOIDS the core too)" if n_below > len(c6a) / 2
                        else "H_settled-leg (the install WROTE into the core)"),
            "census_only_guard": ("if n_base_matrices_also_below is comparable to n_below the band "
                                  "is not discriminating and this is census-only")}
        print(f"[basis] C6a: install below band on {n_below}/{len(c6a)} "
              f"(base matrices also below on {n_base_below}); median ratio "
              f"{rep['c6a_summary']['median_ratio_install_over_rand']:.4f}", flush=True)
        bank()
    rep["total_secs"] = time.time() - t0
    rep["DONE"] = True
    bank()
    print(f"[basis] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
