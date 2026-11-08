"""CORE-PROVENANCE C6c + C6a — is the untouched core INHERITED from pretraining, or SETTLED by PG1?

Spec `2026-08-03_core_provenance_addendum_spec.md`, tag `core-provenance-freeze-2026-08-03`. This
job runs **C6c** (the gate, first) and **C6a**; **C6b** is a separate GPU job that C6c can VOID.

**NOT a FIX-program result.** C6 was added to the FIX spec's authoring copy after the FIX program
was frozen and executed; per that spec's fork notice it was never part of it. Registered prediction
is **ABSTAINED** (§1): both parents' evidence is consistent with either hypothesis.

THE QUESTION. Result 3 says a q/k input subspace exists that all 26 children of PG1 leave untouched.
Two readings, and TESSERA cannot tell them apart:

  H_core-inherited   the core is a PRETRAINING-era invariant — gemma-2b already had it, the install
                     write avoided it too, and PG1's children inherited a property of the base LM.
  H_core-settled     the core is a property of the SETTLED checkpoint — PG1's own training created
                     it, and it says nothing about anything upstream.

The discriminator is the INSTALL write. If `PG1 − gemma-2b` also avoids the core, the core predates
PG1's children. If the install wrote *into* it, the core is PG1's own.

(c) CORRESPONDENCE GATE, FIRST — and it can VOID (b).
The core basis lives in a 2048-dim residual-input space. Ablating it *from gemma-2b* presumes the two
models' residual coordinates mean the same thing, which is an assumption, not a fact. **AF-7, the
operationalisation, recorded before the run:** transport across the hop is the identity on coordinate
indices (same width, same embedding lineage), and the gate tests whether that identity is meaningful
by asking whether PG1 and gemma-2b's q_proj actually READ comparable input subspaces — principal
angles between their top-r right-singular subspaces, per layer, against the analytic Haar null r/n.
If the two models read near-orthogonal subspaces, "the same core subspace" is ill-posed across the
hop and part (b) is VOID. The spec names the gate; this is one defensible reading of it, and the
alternatives are recorded in the artifact rather than silently chosen.

(a) INSTALL ENERGY IN THE CORE.
U_unt(L,k) = the k smallest eigenvectors of the 26-child consensus Gram (Result 3's own object,
rebuilt here because `tessera_s_full2` banks eigenVALUES only). Energy fraction of the install write
inside it:  ‖ΔW_install · U‖²_F / ‖ΔW_install‖²_F,  against the matched-random expectation k/n and a
drawn Haar band. **Below the band ⇒ the install avoided the core too (inherited). At/above ⇒ the
install wrote into it (settled).**

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_c6 --type r7i.2xlarge --no-weights \\
      --args "--s s3://.../tessera_s_full2/out/metrics.json" \\
      --s3-prefix results/tessera_c6 --budget-min 960

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

PG1_REPO = "google/paligemma-3b-pt-224"
GEMMA1_REPO = "google/gemma-2b"
K_GRID = (8, 32, 128)
R_CORR = 64          # subspace rank for the correspondence gate (Arm S's R_SUBSPACE)
N_DRAW = 200


def _read(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--s", required=True, help="banked tessera_s_full2 metrics.json")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from infra.aws_pg.jobs.tessera_s_core import (CORE_TYPES, _delete_cache, _ftype, _hf_lm_sd,
                                                  _policy_lm_sd, unit_sources)
    from infra.pg import tessera_core as TC

    s = _read(a.s)
    roster26 = s["roster26"]
    assert len(roster26) == 26, f"roster is {len(roster26)}, want 26"

    w_pg1 = _hf_lm_sd(PG1_REPO)
    w_g1 = _hf_lm_sd(GEMMA1_REPO)
    core_keys = sorted(k for k in w_pg1 if _ftype(k) in CORE_TYPES)
    assert len(core_keys) == 36, f"{len(core_keys)} core keys, want 36"

    rep = {"object": "TESSERA-FIX C6 — is the untouched core INHERITED or SETTLED?",
           "spec": "core-provenance-freeze-2026-08-03 §2 C6a/C6c",
           "registered_prediction": "ABSTAINED (§1) — no favorite between H_inherited and H_settled",
           "amendment_note": "THE SPEC'S LITERAL C6c IS DEGENERATE, which is why AF-7 substitutes. "
                             "§2 asks for principal angles between 'the core basis in PG1 "
                             "coordinates and its transport to gemma-2b coordinates', but no "
                             "transport map is defined, and the only natural one between two "
                             "2048-wide residual streams of the same lineage is the IDENTITY on "
                             "coordinate indices -- under which the comparison is a basis against "
                             "itself and returns 1.0 by construction. AF-7 therefore tests the "
                             "assumption that identity transport is MEANINGFUL, which is the "
                             "non-degenerate question the gate exists to ask. This is a SUBSTITUTE "
                             "statistic, not the spec's, and is labelled as one.",
           "amendment": "AF-7 — the spec names a correspondence gate without fixing its statistic. "
                        "Operationalised as principal angles between PG1's and gemma-2b's top-r "
                        "q/k right-singular subspaces per layer, against the analytic Haar null "
                        "r/n: if the two models read near-orthogonal input subspaces, 'the same "
                        "core subspace' is ill-posed across the hop and part (b) is VOID. Recorded "
                        "before the run; alternative readings noted in `gate_alternatives`.",
           "gate_alternatives": ["compare the core basis to gemma-2b's own bottom-sigma band "
                                 "(needs a gemma-2b child family, which does not exist)",
                                 "transport by the install write's own singular vectors (circular "
                                 "— that is what part (a) measures)"],
           "k_grid": list(K_GRID), "r_correspondence": R_CORR, "n_draws": N_DRAW}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    # ---- (c) correspondence gate ---------------------------------------------------------------
    haar = {}
    corr = []
    for k in core_keys:
        A = w_pg1[k].to(torch.float64).numpy()
        B = w_g1[k].to(torch.float64).numpy()
        n = A.shape[1]
        if n not in haar:
            haar[n] = TC.haar_overlap_null(n, R_CORR, R_CORR, n_draws=N_DRAW, seed=a.seed)
        U = TC.top_right_subspace(A, R_CORR)
        V = TC.top_right_subspace(B, R_CORR)
        o = TC.subspace_overlap(U, V)
        corr.append({"key": k, "type": _ftype(k), "overlap": o,
                     "haar_mean": haar[n]["mean"], "haar_p97_5": haar[n]["p97.5"],
                     "corresponds": bool(o > haar[n]["p97.5"])})
        rep["correspondence"] = corr
    n_corr = sum(1 for c in corr if c["corresponds"])
    med_o = float(np.median([c["overlap"] for c in corr]))
    gate_pass = bool(n_corr >= len(corr) // 2)
    rep["correspondence_gate"] = {
        "n_corresponding": n_corr, "n_matrices": len(corr), "median_overlap": med_o,
        "haar_p97_5": corr[0]["haar_p97_5"],
        "verdict": ("PASS — PG1 and gemma-2b read overlapping q/k input subspaces, so 'the same "
                    "core subspace' is meaningful across the hop and part (b) is well-posed"
                    if gate_pass else
                    "VOID — PG1 and gemma-2b read essentially unrelated q/k input subspaces; "
                    "transporting the core basis across the install hop is ill-posed, so part (b) "
                    "does not run and only part (a) reports")}
    print(f"[C6c] correspondence {n_corr}/{len(corr)} above Haar p97.5 "
          f"(median overlap {med_o:.4f}) -> {'PASS' if gate_pass else 'VOID'}", flush=True)
    _w()

    # ---- rebuild the 26-child consensus Gram, then (a) ------------------------------------------
    gram = {k: None for k in core_keys}
    for uid, (kind, ref) in unit_sources(roster26):
        print(f"[C6 unit] {uid}", flush=True)
        w = _hf_lm_sd(ref) if kind == "hf" else _policy_lm_sd(ref)
        for k in core_keys:
            A = (w[k].to(torch.float64) - w_pg1[k].to(torch.float64)).numpy()
            fro2 = float((A ** 2).sum())
            assert fro2 > 0, f"{uid}/{k}: zero ΔW"
            G = (A.T @ A) / fro2
            gram[k] = G if gram[k] is None else gram[k] + G
            del G, A
        del w
        if kind == "hf":
            _delete_cache(ref)
        rep.setdefault("units_streamed", []).append(uid)
        _w()

    rng = np.random.default_rng(a.seed)
    rows, basis = [], {}
    for k in core_keys:
        dW = (w_pg1[k].to(torch.float64) - w_g1[k].to(torch.float64)).numpy()   # the INSTALL write
        fro2 = float((dW ** 2).sum())
        evals, evecs = np.linalg.eigh(gram[k])
        n = gram[k].shape[0]
        # Bank the core BASIS, not just the verdict. Part (b) ablates this exact subspace from
        # gemma-2b, and without it that job would have to re-stream all 26 units (~3 GPU-h of
        # pure duplication). `tessera_s_full2` banking eigenVALUES only is what forced C3 and this
        # job to rebuild the Gram in the first place -- not repeating that.
        basis[k] = torch.from_numpy(np.ascontiguousarray(evecs[:, :max(K_GRID)]))
        rec = {"key": k, "type": _ftype(k), "install_fro2": fro2}
        for kk in K_GRID:
            U = evecs[:, :kk]                                   # the untouched core basis
            frac = float(((dW @ U) ** 2).sum() / fro2)
            draws = []
            for _ in range(50):
                Q, _r = np.linalg.qr(rng.standard_normal((n, kk)))
                draws.append(float(((dW @ Q) ** 2).sum() / fro2))
            d = np.asarray(draws)
            # BASE-MATRIX CONTROL (§2 + the house rule on any band statistic). If the BASE weight
            # matrices also fall below the band, the band does not discriminate and this is
            # census-only -- the LACUNA unfailable-null lesson, applied before the verdict.
            Wb = w_pg1[k].to(torch.float64).numpy()
            base_frac = float(((Wb @ U) ** 2).sum() / float((Wb ** 2).sum()))
            rec[str(kk)] = {"install_energy_in_core": frac,
                            "matched_random_mean": float(d.mean()),
                            "matched_random_p2_5": float(np.percentile(d, 2.5)),
                            "analytic_expectation": kk / n,
                            "ratio_to_random": frac / float(d.mean()),
                            "below_random_band": bool(frac < np.percentile(d, 2.5)),
                            "base_matrix_energy_in_core": base_frac,
                            "base_matrix_below_band": bool(base_frac < np.percentile(d, 2.5))}
        rows.append(rec)
        print(f"[C6a] {k} k=8 install-in-core {rec['8']['install_energy_in_core']:.5f} "
              f"vs random {rec['8']['matched_random_mean']:.5f} "
              f"(ratio {rec['8']['ratio_to_random']:.3f})", flush=True)
        rep["install_in_core"] = rows
        _w()

    summ = {}
    for kk in K_GRID:
        r = [x[str(kk)]["ratio_to_random"] for x in rows]
        nb = sum(1 for x in rows if x[str(kk)]["below_random_band"])
        # BOTH forms per §2 -- median-of-ratios AND ratio-of-medians.
        summ[str(kk)] = {"median_of_ratios": float(np.median(r)),
                         "ratio_of_medians": float(
                             np.median([x[str(kk)]["install_energy_in_core"] for x in rows])
                             / np.median([x[str(kk)]["matched_random_mean"] for x in rows])),
                         "min": float(min(r)), "max": float(max(r)),
                         "n_below_random_band": nb,
                         "n_base_matrices_below_band": sum(
                             1 for x in rows if x[str(kk)]["base_matrix_below_band"]),
                         "n_matrices": len(rows)}
    # H_core-inherited (a): the install must AVOID the core -- below the matched-random band.
    k8 = summ["8"]
    base_fails = k8["n_base_matrices_below_band"] >= len(rows) // 2
    inherited = bool(k8["median_of_ratios"] < 1.0 and k8["n_below_random_band"] >= len(rows) // 2)
    rep["summary"] = summ
    rep["C6a_verdict"] = {
        "verdict": ("CENSUS-ONLY — the BASE matrices also fall below this band, so it does not "
                    "discriminate and no verdict is read from it (§3)"
                    if base_fails else
                    ("INHERITED-leg — the install write also avoids the core"
                     if inherited else
                     "SETTLED — the install write does NOT avoid the core; the core is an object "
                     "of PG1's children rather than a pretraining-era invariant")),
        "median_of_ratios_k8": k8["median_of_ratios"],
        "ratio_of_medians_k8": k8["ratio_of_medians"],
        "n_below_random_band_k8": k8["n_below_random_band"],
        "n_base_matrices_below_band_k8": k8["n_base_matrices_below_band"],
        "n_matrices": len(rows),
        "flatness_note": "FIX C4 measured hop writes as spectrally FLATTER than FT writes. A flat "
                         "write puts ~k/d of its energy in ANY fixed k-subspace, so flatness pushes "
                         "this ratio TOWARD 1 and cannot manufacture avoidance: a below-band "
                         "reading is CONSERVATIVE against that confound (§2)."}
    rep["part_b_status"] = ("well-posed — the correspondence gate passed; the gemma-2b FUNC "
                            "ablation can be run" if gate_pass else
                            "VOID by the correspondence gate — not run")
    torch.save({"basis": basis, "k_grid": list(K_GRID), "roster26": roster26,
                "note": "columns are the ASCENDING eigenvectors of the 26-child consensus Gram; "
                        "U_unt(k) = basis[key][:, :k]"},
               out / "core_basis.pt")
    rep["core_basis"] = {"file": "core_basis.pt", "n_matrices": len(basis),
                         "max_k": max(K_GRID),
                         "why": "banked so part (b) can ablate this exact subspace from gemma-2b "
                                "without re-streaming the 26 units"}
    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"\n(c) correspondence: {rep['correspondence_gate']['verdict'][:60]}")
    print(f"(a) install-in-core k=8: median-of-ratios {k8['median_of_ratios']:.4f}, "
          f"{k8['n_below_random_band']}/{len(rows)} below the random band")
    print(f"    -> {rep['H_core_inherited_part_a']['verdict']}")
    print("TESSERA_C6_OK", flush=True)


if __name__ == "__main__":
    main()
