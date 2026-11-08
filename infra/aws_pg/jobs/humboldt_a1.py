"""HUMBOLDT A1 — complete avoidance-spectrum atlas (+ gates G1, G2a, G2b, G5-offline).

Spec §3/A1. CPU. For all 7 types × 18 layers of the PG1 LM decoder, over the 24 official PG1
fine-tunes: build the fp64 consensus Gram, record the FULL eigenvalue spectrum, bank core32/anti32
and the whole twin battery, and record per-bank coordinate leverage, participation ratio, and the
all-pairs cross-type principal angles.

Deviations from the literal spec text, all pre-freeze and journaled:
  C1  cross-type angles cover 15 type-pairs, not 21 — down_proj's input space is ℝ^16384 while the
      other six are ℝ^2048, so angles across it are undefined. Same-ambient pairs that nonetheless
      read DIFFERENT activation tensors carry cross_space=true.
  C3  the Gram is fp64-from-checkpoint, never from the cached fp16 ΔW scratch.
  C4  G2 splits into G2a (bit-determinism against the cached fp16-derived artifact) and G2b (the
      fp16-vs-fp64 precision audit, no threshold — evidence only).
  C5  bottomsigma32 uses the ECONOMY right-singular basis for rank-deficient types.

  pg run infra.aws_pg.jobs.humboldt_a1 --type r7i.4xlarge --no-weights \
      --s3-prefix results/humboldt_a1 --budget-min 480
"""
import argparse
import itertools
import json
import pathlib
import subprocess

from infra.pg import humboldt_core as H
from infra.pg import humboldt_smoke as S
from infra.pg import humboldt_weights as W

CACHED_INVCORE_S3 = "s3://pg-lineage-769063704696/results/invcore_full2/out/invariant_core.json"
CACHED_SCRATCH_S3 = "s3://pg-lineage-769063704696/results/invcore_full2/out/scratch"
K_GRID = (8, 32, 128)                      # invariant_core's grid — G2a compares against it
ANGLE_BANK_PAIRS = (("core32", "core32"), ("anti32", "anti32"), ("core32", "anti32"),
                    ("haar32_s0", "haar32_s0"), ("bottomsigma32", "bottomsigma32"),
                    ("complement32", "complement32"), ("core32", "haar32_s0"))
# G2a branches, pre-registered BEFORE the run (one-sided-threshold discipline: name the fail
# observation in advance).
#   PASS-A  bitwise identical to the cached artifact  → the pipeline is bit-deterministic.
#   PASS-B  not bitwise, but the deviation is inside the accumulation-order noise floor MEASURED in
#           the same job (the identical Gram re-accumulated in REVERSED fine-tune order), times
#           G2A_FLOOR_SLACK  → the difference is BLAS/summation ordering, not the pipeline.
#   FAIL    anything larger.
#
# Why a measured floor rather than a fixed tolerance: the bottom-k eigenvalues are ~2.6e6 times
# smaller than ‖G‖ (trace 24 vs a bottom-8 mean of ~9.2e-6), so a single fp64 rounding at the scale
# of G moves them by ~6e-10 RELATIVE, and accumulation over a 2048-length dot product and 24 terms
# puts the realistic ordering noise near 1e-8. Any fixed constant is either too tight (fails for a
# reason that is not the pipeline) or arbitrary. Re-accumulating in reversed order measures exactly
# the quantity the branch is trying to excuse. A genuine pipeline error is O(1), not O(1e-8).
G2A_FLOOR_SLACK = 10.0
G2A_FLOOR_MIN = 1e-12


def _same_site(ta, tb):
    return H.SITE[ta] == H.SITE[tb]


def _g2(out, gates, repos, core32_fp64_q, eig_fp64_q, layers, seed):
    """G2a — recompute q_proj bottom-k eigenvalues from the CACHED fp16 scratch through
    invariant_core's own PASS-2 code path and require bit-identity with the cached artifact.
    G2b — the fp16-vs-fp64 precision audit that motivated C3 (evidence only, no threshold)."""
    import numpy as np
    from infra.aws_pg.jobs.invariant_core import _bottom_topk_eigs

    cache = H.cache_dir(out, "invcore")
    subprocess.run(["aws", "s3", "cp", CACHED_INVCORE_S3, str(cache / "invariant_core.json"),
                    "--region", "us-east-1"], check=True)
    cached = json.loads((cache / "invariant_core.json").read_text())
    assert cached["cloud_repos"] == list(repos), \
        "cached invariant_core used a DIFFERENT fine-tune cloud — G2a would compare two objects"
    scratch = cache / "scratch"
    scratch.mkdir(exist_ok=True)
    subprocess.run(["aws", "s3", "sync", CACHED_SCRATCH_S3, str(scratch), "--region", "us-east-1",
                    "--exclude", "*", "--include", "q_proj__*"], check=True)

    def _accumulate(L, order):
        """invariant_core's PASS-2 accumulation, verbatim, over the fine-tune indices in `order`."""
        G = None
        for ridx in order:
            f = scratch / f"q_proj__L{L}__{ridx}.npy"
            assert f.exists(), f"cached fp16 scratch missing {f.name}"
            A = np.load(f).astype(np.float64)
            fro2 = (A ** 2).sum()
            term = (A.T @ A) / fro2
            G = term if G is None else G + term
        return G

    by_layer = {(r["type"], r["layer"]): r for r in cached["per_matrix"]}
    bitwise, rel_deltas, floor_deltas, angle_rows = {}, [], [], []
    for L in layers:
        ref = by_layer.get(("q_proj", L))
        assert ref is not None, f"cached artifact has no q_proj L{L} row"
        fwd = list(range(len(repos)))                        # SAME order as invariant_core PASS 2
        G16 = _accumulate(L, fwd)
        eigs16, bot16, _top16 = _bottom_topk_eigs(G16, list(K_GRID))
        # the noise floor: the identical computation, summed in the opposite order.
        _e_r, bot_rev, _t_r = _bottom_topk_eigs(_accumulate(L, fwd[::-1]), list(K_GRID))
        for k in K_GRID:
            ks = str(k)
            same = bot16[ks] == ref["real_bottom_k"][ks]
            bitwise[f"L{L}_k{k}"] = bool(same)
            denom = abs(ref["real_bottom_k"][ks])
            rel_deltas.append(abs(bot16[ks] - ref["real_bottom_k"][ks]) / denom if denom else 0.0)
            floor_deltas.append(abs(bot16[ks] - bot_rev[ks]) / denom if denom else 0.0)
        # G2b: the same layer's fp16-derived core32 vs the fp64-derived core32 shipped in the atlas.
        V16 = np.linalg.eigh(G16)[1]
        ang = H.principal_angles(V16[:, :H.BANK_K], core32_fp64_q[L])
        rel_eig = float(abs(eigs16[:H.BANK_K].mean() - eig_fp64_q[L][:H.BANK_K].mean())
                        / max(eig_fp64_q[L][:H.BANK_K].mean(), 1e-300))
        angle_rows.append({"layer": int(L), "max_angle_rad": float(ang.max()),
                           "mean_angle_rad": float(ang.mean()),
                           "bottom32_eig_rel_delta_fp16_vs_fp64": rel_eig})
        del G16, V16

    all_bitwise = all(bitwise.values())
    max_rel = float(max(rel_deltas))
    floor = max(float(max(floor_deltas)) * G2A_FLOOR_SLACK, G2A_FLOOR_MIN)
    ok = all_bitwise or max_rel <= floor
    gates.require("G2a_consensus_gram_pipeline", ok,
                  bitwise_identical=all_bitwise, n_cells=len(bitwise),
                  n_bitwise_mismatch=int(sum(1 for v in bitwise.values() if not v)),
                  max_rel_delta=max_rel,
                  measured_order_noise=float(max(floor_deltas)), slack=G2A_FLOOR_SLACK,
                  effective_tolerance=floor,
                  bitwise_nondeterminism=bool(not all_bitwise),
                  cached_artifact=CACHED_INVCORE_S3,
                  note=("PASS-A bitwise identical; PASS-B deviation inside the measured "
                        "accumulation-order noise floor (same Gram, reversed fine-tune order) × 10. "
                        "Both branches pre-registered before the run."))
    gates.record("G2b_fp16_vs_fp64_precision_audit", True,
                 threshold="none — evidence only (journal C3)",
                 per_layer=angle_rows,
                 max_core32_angle_rad=float(max(r["max_angle_rad"] for r in angle_rows)),
                 max_bottom32_eig_rel_delta=float(max(
                     r["bottom32_eig_rel_delta_fp16_vs_fp64"] for r in angle_rows)),
                 banks_shipped="fp64")
    return {"g2a_bitwise": bitwise, "g2b": angle_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default=",".join(H.TYPES))
    ap.add_argument("--layers", default="", help="comma idxs; empty ⇒ all 18")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-id", default="humboldt_a1")
    ap.add_argument("--skip-g2", action="store_true", help="smoke only; never in the real run")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    assert all(t in H.TYPES for t in types), f"unknown type in {types}"
    layers = ([int(x) for x in a.layers.split(",") if x.strip()] if a.layers.strip()
              else list(range(H.N_LAYERS)))

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.7)
    import numpy as np
    from huggingface_hub import HfApi

    gates = H.Gates(out)
    S.run_all_offline(gates)                            # G5, every arm, in-job (not only offline)

    api = HfApi()
    repos = W.ft_cloud(api)
    if a.smoke:
        repos = repos[:3]
        layers = layers[:2]
        types = tuple(t for t in types if t != "down_proj")
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(W.PG1_REPO).sha

    manifest = {"arm": "A1", "run_id": a.run_id, "commit_sha": H.commit_sha(),
                "base_repo": W.PG1_REPO, "base_sha": base_sha, "cloud_repos": repos,
                "cloud_shas": shas, "n_ft": len(repos), "types": list(types), "layers": layers,
                "bank_k": H.BANK_K, "k_grid": list(K_GRID), "seed": a.seed, "smoke": a.smoke,
                "precision": "fp64 consensus Gram accumulated from fp32 checkpoints (journal C3)",
                "bank_ids": list(H.BANK_IDS), "angle_bank_pairs": [list(p) for p in ANGLE_BANK_PAIRS]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[A1] base={base_sha[:8]} n_ft={len(repos)} types={types} layers={layers}", flush=True)

    atlas = H.Atlas(out, arm="A1", run_id=a.run_id)     # bound BEFORE any loop (§0.2.4)
    resid_types_all = [t for t in types if t in H.TYPES_RESID]

    G, W0, fro2_by = W.stream_consensus_grams(repos, shas, base_sha, types, layers)
    print(f"[A1] grams built: {len(G)} cells", flush=True)

    # ---- per (type, layer): eig → banks → statistics. Free each Gram immediately (down_proj is
    # 2.1 GB per cell; holding all 18 plus their eigenvector matrices would not fit).
    V_by = {}                       # kept only for q_proj (G2b) and for the per-layer angle pass
    eig_by = {}
    g1_resid = []
    for t in types:
        for L in layers:
            key = (t, L)
            Gm = G.pop(key)
            lam, V = W.eig_ascending(Gm)
            base = W0[key]
            banks = H.build_banks(V, base.numpy(), k=H.BANK_K,
                                  seed_base=H.cell_seed(a.seed, t, L))
            W.save_banks(out, t, L, banks)
            if t == "q_proj":
                # G3 (in B1) ablates the FUNC band — the bottom-128 eigvecs of THIS Gram — so the
                # 128-wide block ships alongside the 32-wide banks. Publishing it here is what makes
                # G3 an identification of the SHIPPED object rather than of a re-derivation.
                np.save(pathlib.Path(out) / "banks" / W.bank_name(t, L, "gramV_bottom128"),
                        np.ascontiguousarray(V[:, :128]))
            if t in H.TYPES_RESID:
                V_by[key] = {bid: U for bid, U in banks.items()}
            if t == "q_proj":
                eig_by[L] = lam
            fl = H.type_flags(t)
            # G1 (§1 kill-table row 1) — ONE residual per matrix, computed by splitting the REAL
            # eigenbasis into U and U⊥. It is the key-alignment / basis-completeness check.
            g1_resid.append({"type": t, "layer": int(L),
                             "resid": H.reconstruction_residual(base.numpy(), V, H.BANK_K)})

            atlas.add("gram_spectrum", vector_ref=atlas.vector(f"A1__gram_spectrum__{t}__L{L}", lam),
                      layer=L, type=t, model_id=W.PG1_REPO, model_sha=base_sha, n=int(lam.size),
                      flags={**fl, "trace_G": float(lam.sum()), "n_ft": len(repos)})
            atlas.add("dW_fro2_by_ft",
                      vector_ref=atlas.vector(f"A1__dW_fro2__{t}__L{L}",
                                              [fro2_by[key][r] for r in repos]),
                      layer=L, type=t, model_id=W.PG1_REPO, model_sha=base_sha, n=len(repos),
                      flags={**fl, "order": "manifest.cloud_repos"})
            for k in K_GRID:
                # These are properties of the GRAM's spectrum, not of a bank. Labelling them with a
                # bank_id made them look like bank statistics missing their twins; the twin-complete
                # object is `gram_bank_energy` below.
                atlas.add(f"gram_bottom{k}_mean", value=float(lam[:k].mean()), layer=L, type=t,
                          model_id=W.PG1_REPO, model_sha=base_sha, flags=fl)
                atlas.add(f"gram_top{k}_mean", value=float(lam[-k:].mean()), layer=L, type=t,
                          model_id=W.PG1_REPO, model_sha=base_sha, flags=fl)

            atlas.add("reconstruction_residual", value=g1_resid[-1]["resid"], layer=L, type=t,
                      model_id=W.PG1_REPO, model_sha=base_sha,
                      flags={**fl, "split": f"V[:, :{H.BANK_K}] | V[:, {H.BANK_K}:]"})
            for bid, U in banks.items():
                # ADDENDUM (§6, post-freeze addition, tagged as such): mean consensus write-energy
                # per direction of the bank, tr(UᵀGU)/k. For core32 this equals the bottom-k
                # eigenvalue mean and for anti32 the top-k mean BY CONSTRUCTION — which is exactly
                # why it is the right twin-comparable form: the same number the spectrum statistics
                # report for the real banks, now defined identically for every twin, so "how much do
                # the fine-tunes write here" can be read against its own chance column.
                atlas.add("gram_bank_energy",
                          value=float(np.trace(U.T @ (Gm @ U)) / U.shape[1]), layer=L, type=t,
                          bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha, tag=H.ADDENDUM,
                          flags={**fl, "flat_reference": float(lam.sum() / lam.size),
                                 "definition": "tr(UᵀGU)/k"})
                lev = H.leverage(U)
                atlas.add("bank_energy_fraction_of_W",
                          value=H.bank_energy_fraction(base.numpy(), U), layer=L, type=t,
                          bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**fl, "haar_expectation": H.BANK_K / H.IN_DIM[t]})
                atlas.add("leverage_profile",
                          vector_ref=atlas.vector(f"A1__leverage__{t}__L{L}__{bid}", lev),
                          layer=L, type=t, bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          n=int(lev.size), flags=fl)
                atlas.add("leverage_max", value=float(lev.max()), layer=L, type=t, bank_id=bid,
                          model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**fl, "haar_expectation": H.BANK_K / H.IN_DIM[t]})
                atlas.add("participation_ratio", value=H.participation_ratio(lev), layer=L, type=t,
                          bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**fl, "n_coords": int(lev.size)})
            del V, lam, banks, Gm
        print(f"[A1] {t}: {len(layers)} layers recorded ({atlas.n_rows} rows)", flush=True)
        H.s3_sync(out, f"humboldt/{a.run_id}")
        if not a.smoke:
            # Publish per TYPE, not once at the end. down_proj's 18 × 16384² eigendecompositions are
            # the only part of this arm that could plausibly hit the wall-clock cap; if it does, the
            # six residual types are already a usable bank library and the run resumes with
            # `--types down_proj --layers <missing>`.
            W.publish_banks(out)

    # ---- G1 (§1 kill-table row 1): the bank↔matrix key alignment, over every cell recorded.
    max_resid = max(r["resid"] for r in g1_resid)
    gates.require("G1_bank_loading_key_alignment", max_resid <= 1e-5,
                  max_residual=max_resid, threshold=1e-5, n_cells=len(g1_resid),
                  worst=max(g1_resid, key=lambda r: r["resid"]))

    # ---- G4 (§1 kill-table row 4): a Haar twin must NOT be statistically indistinguishable from the
    # real bank on leverage max. Both sides recomputed here; nothing quoted.
    # The GATE is scoped to q_proj — the type where the known object lives and where the spec's
    # "0.77-class spike vs 1/64-class" expectation comes from. Scoping matters: pooling all types
    # would let the gate fail for a SCIENTIFIC reason (some type's core32 happens to be flat), and a
    # plumbing gate that can fail for a non-plumbing reason is not a plumbing gate. The full
    # per-type table is recorded as evidence either way.
    per_type = {}
    for t in resid_types_all:
        lr = [float(H.leverage(V_by[(t, L)]["core32"]).max()) for L in layers]
        lh = [float(H.leverage(V_by[(t, L)]["haar32_s0"]).max()) for L in layers]
        per_type[t] = {"real": H.mean_ci(lr, seed=a.seed), "haar": H.mean_ci(lh, seed=a.seed)}
    gq = per_type.get("q_proj")
    assert gq is not None, "G4 requires q_proj in --types"
    gates.require("G4_twin_plumbing", gq["real"]["ci_lo"] > gq["haar"]["ci_hi"],
                  gate_scope="q_proj", q_proj_real_leverage_max=gq["real"],
                  q_proj_haar_leverage_max=gq["haar"], haar_expectation=H.BANK_K / 2048,
                  n_cells=len(layers), per_type_evidence=per_type,
                  criterion="q_proj core32 leverage-max CI strictly above the haar32_s0 CI")

    # ---- cross-type principal angles within each layer (C1: 15 pairs over the n=2048 types).
    for L in layers:
        for ta, tb in itertools.combinations(resid_types_all, 2):
            for ba, bb in ANGLE_BANK_PAIRS:
                ang = H.principal_angles(V_by[(ta, L)][ba], V_by[(tb, L)][bb])
                fl = {"type_a": ta, "type_b": tb, "bank_a": ba, "bank_b": bb,
                      "cross_space": not _same_site(ta, tb),
                      "site_a": H.SITE[ta], "site_b": H.SITE[tb],
                      "mqa_kernel_caveat": (ta in H.MQA_KERNEL_TYPES or tb in H.MQA_KERNEL_TYPES)}
                nm = f"A1__angles__L{L}__{ta}_{ba}__{tb}_{bb}"
                atlas.add("cross_type_principal_angles", vector_ref=atlas.vector(nm, ang),
                          layer=L, type=f"{ta}|{tb}", bank_id=f"{ba}|{bb}", model_id=W.PG1_REPO,
                          model_sha=base_sha, n=int(ang.size), flags=fl)
                atlas.add_ci("cross_type_principal_angle_mean", ang, layer=L, type=f"{ta}|{tb}",
                             bank_id=f"{ba}|{bb}", model_id=W.PG1_REPO, model_sha=base_sha,
                             seed=a.seed, flags=fl)
        print(f"[A1] L{L}: cross-type angles done ({atlas.n_rows} rows)", flush=True)
    # the chance reference for every angle cell, recomputed (never hardcoded).
    band = H.haar_angle_band(2048, H.BANK_K, n_draw=200, seed=a.seed)
    atlas.add("haar_angle_band_mean",
              vector_ref=atlas.vector("A1__haar_angle_band_mean", band["mean"]),
              n=H.BANK_K, seed=a.seed, flags={"n": 2048, "k": H.BANK_K, "n_draw": band["n_draw"],
                                              "role": "chance column for every angle cell"})
    (out / "haar_angle_band.json").write_text(json.dumps(band, indent=2))

    # ---- G2 (needs the fp64 q_proj core32, so it runs after the per-cell pass). The fp64 side of
    # the comparison is the SHIPPED bank, not a re-derivation of it.
    if not a.skip_g2:
        g2 = _g2(out, gates, repos, {L: V_by[("q_proj", L)]["core32"] for L in layers},
                 eig_by, layers, a.seed)
        (out / "g2_detail.json").write_text(json.dumps(g2, indent=2))

    if a.smoke:
        # a smoke's banks come from a 3-fine-tune cloud: same filenames, different object. Never let
        # them into the canonical library that every downstream arm loads.
        print("[A1][smoke] banks NOT published to the shared library", flush=True)
    else:
        W.publish_banks(out)                            # final catch-all (idempotent)
    atlas.close()
    rep = atlas.to_parquet()
    (out / "a1_summary.json").write_text(json.dumps(
        {"manifest": manifest, "atlas": rep, "gates": gates.gates,
         "n_banks": len(list((out / "banks").glob("*.npy")))}, indent=2, default=str))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_A1_OK rows={rep['n_rows']} vectors={rep['n_vectors']} "
          f"banks={len(list((out / 'banks').glob('*.npy')))} gates={ {k: v['ok'] for k, v in gates.gates.items()} }")


if __name__ == "__main__":
    main()
