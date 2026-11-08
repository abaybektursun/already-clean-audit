"""DARWIN G1 + W0-8 — consensus-Gram re-derivation and the delta-compression frontier. CPU.

Two jobs in one pass because both need the same expensive object: a full stream of the 24 official
PG1 fine-tunes, differenced against the base in fp64.

  G1  (§9 step 4, REQUIRED gate). Re-derive the consensus Gram G = Σ ΔWᵀΔW/‖ΔW‖²_F from the
      cached fine-tune deltas and check that its bottom-32 eigenspace is HUMBOLDT's banked
      core32, and its top-32 HUMBOLDT's anti32. This is the gate that proves core32 is the object
      the spec says it is — W0-0's G1a only verified the base-matrix SVD path (bottomsigma32).
      Gated on the SUBSPACE, recorded on the elements, for the reason in darwin_w0.bank_identity:
      an eigenbasis is defined only up to sign and up to rotation within a degenerate eigenspace,
      so literal bit-identity is not a well-posed requirement (journal defect D2).

  W0-8 (§4/W0-8). Energy captured vs basis rank over the highway rank ladder 4..256, per
      fine-tune: how much of a real ΔW lives in the top-r consensus ("highway") subspace, against
      the bottom-r (core) subspace and a matched-rank Haar control in the same table. This is the
      weights-only ESTIMATE; W1-D is its eval confirmation.

Memory (§0.2.7's stated escape). Σ n² x 8 bytes over the requested cells: 6 residual types x 18
layers x 2048² = 3.6 GB, which fits r7i.2xlarge. down_proj is 16384² x 8 x 18 = 38.7 GB and does
NOT — it is recorded as a coverage gap with this reason rather than silently dropped, and can be
run separately with --types down_proj --layers <subset> on a larger box.

  pg run infra.aws_pg.jobs.darwin_w0_gram --args "--arm G1" --type r7i.2xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

RESIDUAL_TYPES = H.TYPES_RESID          # 6 types at n = 2048; down_proj excluded on memory
DOWN_PROJ_GAP = ("down_proj's Gram is 16384^2 x 8 bytes = 2.1 GB per cell, 38.7 GB over 18 "
                 "layers, which exceeds r7i.2xlarge; §0.2.7 names r7i.4xlarge as the escape. "
                 "Recorded as a coverage gap, not dropped.")

SPEC_EXPECTATIONS = {
    "G1": ("re-derived core32/anti32 do not match HUMBOLDT's banked subspaces",
           "the re-derived consensus eigenbasis reproduces the banked core32 and anti32 to float "
           "tolerance"),
    "W0-8": ("the universal basis captures no more delta energy than a matched-rank random basis",
             "the highway basis captures delta energy well above the k/n Haar reference at every "
             "rank"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="G1", choices=("G1", "W0-8"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--types", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bank-rank", type=int, default=256,
                    help="W0-8 banks the top/bottom this-many eigenvectors so the rank-ladder "
                         "arms (W1-D) need not re-stream the 24-checkpoint cloud")
    ap.add_argument("--max-fts", type=int, default=0,
                    help="truncate the fine-tune cloud (0 = all). A truncated cloud makes the "
                         "G1 identity comparison RECORDED, not REQUIRED — see the note in main().")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    types = [t.strip() for t in a.types.split(",") if t.strip()] or list(RESIDUAL_TYPES)
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    for t in types:
        assert t in H.TYPES, f"unknown type {t}"
    if a.smoke:
        types, layers = types[:1], layers[:2]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import numpy as np
    from huggingface_hub import HfApi

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    api = HfApi()
    base_sha = api.model_info(W.PG1_REPO).sha
    repos_full = W.ft_cloud(api)
    repos = repos_full if a.max_fts <= 0 else repos_full[:a.max_fts]
    shas = {r: api.model_info(r).sha for r in repos}
    # The consensus Gram is defined BY the cloud it is summed over. HUMBOLDT's core32 is the
    # bottom-32 eigenspace of the Gram over ALL 24 fine-tunes; the Gram over a subset is a
    # different matrix with a different eigenbasis, so comparing the two is not a re-derivation
    # check at all. `--smoke` therefore truncates types/layers (each cell is independent) and
    # never the cloud. If a caller does truncate it via --max-fts, the identity comparison is
    # RECORDED rather than REQUIRED, because a gate on incomparable objects cannot mean anything.
    # Caught by the first G1 smoke, which truncated to 3 FTs and duly reported ~pi/2 (journal D4).
    cloud_complete = len(repos) == len(repos_full)
    print(f"[gram] {len(repos)}/{len(repos_full)} fine-tunes x {len(types)} types x "
          f"{len(layers)} layers (cloud_complete={cloud_complete})", flush=True)

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W0",
                    expected_ref=ref)
    surprises = D.SurpriseLog(out)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W0", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": W.PG1_REPO, "base_sha": base_sha, "n_fine_tunes": len(repos),
                "n_fine_tunes_full": len(repos_full), "cloud_complete": cloud_complete,
                "fine_tunes": repos, "fine_tune_shas": shas, "types": types, "layers": layers,
                "rank_ladder": DW.rank_ladder(), "expected_ref": ref, "started_utc": utc,
                "seed": a.seed, "coverage_gap_down_proj": DOWN_PROJ_GAP}
    D.write_json(out, "manifest.json", manifest)

    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    G, W0, fro2_by = W.stream_consensus_grams(repos, shas, base_sha, types, layers)

    identities, ladder = [], DW.rank_ladder()
    for t in types:
        n = H.IN_DIM[t]
        for L in layers:
            lam, V = W.eig_ascending(G[(t, L)])          # ASCENDING: core = first, anti = last
            core = np.ascontiguousarray(V[:, :H.BANK_K])
            anti = np.ascontiguousarray(V[:, -H.BANK_K:])
            fl = {**H.type_flags(t), "n_fine_tunes": len(repos), "trace_G": float(lam.sum())}

            for bid, U in (("core32", core), ("anti32", anti)):
                ident = DW.bank_identity(U, W.load_bank(bank_dir, t, L, bid))
                identities.append({"type": t, "layer": L, "bank_id": bid, **ident})
                atlas.add("gram_rederivation_max_principal_angle_rad",
                          value=ident["max_principal_angle_rad"], layer=L, type=t, bank_id=bid,
                          model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**fl, "vs": f"humboldt banked {bid}",
                                 "tol": DW.SUBSPACE_ANGLE_TOL,
                                 "bit_identical": ident["bit_identical"],
                                 "max_abs_diff_sign_aligned": ident["max_abs_diff_sign_aligned"]})
            atlas.add("gram_eigenvalue_spectrum",
                      vector_ref=atlas.vector(f"{a.arm}__lam__{t}__L{L}", lam), layer=L, type=t,
                      bank_id="core32", model_id=W.PG1_REPO, model_sha=base_sha, n=int(lam.size),
                      flags=fl)

            # --- W0-8 also BANKS the wide eigenbasis. W1-D's rank ladder runs to 256 and the
            # published banks stop at 32, so without this every downstream rank-ladder arm would
            # have to re-stream all 24 checkpoints to recover a basis this job already holds.
            if a.arm == "W0-8":
                wide = min(a.bank_rank, V.shape[1])
                W.save_banks(out, t, L, {
                    f"highway_top{wide}": np.ascontiguousarray(V[:, -wide:]),
                    f"core_bottom{wide}": np.ascontiguousarray(V[:, :wide])})
            if a.arm == "W0-8":
                w_base = W0[(t, L)].numpy().astype(np.float64)
                for r in ladder:
                    if r > n:
                        continue
                    bases = {"highway_topr": np.ascontiguousarray(V[:, -r:]),
                             "core_bottomr": np.ascontiguousarray(V[:, :r]),
                             "haar_r": H.haar_subspace(n, r, H.cell_seed(a.seed, t, L, r))}
                    for bid, U in bases.items():
                        e_base = DW.energy_captured(w_base, U)
                        atlas.add("base_energy_captured", value=e_base, layer=L, type=t,
                                  bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                                  dose=float(r),
                                  flags={**fl, "rank": r, "haar_reference": r / n})
            print(f"[{a.arm}] {t} L{L}: rederived (angle "
                  f"{identities[-1]['max_principal_angle_rad']:.2e})", flush=True)
        D.s3_sync(out, a.s3_prefix)

    if a.arm == "W0-8":
        D.publish_banks(out)
    worst = max(i["max_principal_angle_rad"] for i in identities)
    n_bit = sum(1 for i in identities if i["bit_identical"])
    ev = dict(worst_max_principal_angle_rad=worst, tol=DW.SUBSPACE_ANGLE_TOL,
              n_cells=len(identities), n_bit_identical=n_bit, n_fine_tunes=len(repos),
              n_fine_tunes_full=len(repos_full), cloud_complete=cloud_complete,
              worst_cell=max(identities, key=lambda i: i["max_principal_angle_rad"]))
    if cloud_complete:
        gates.require("G1_gram_rederivation_subspace_identity",
                      worst < DW.SUBSPACE_ANGLE_TOL, **ev)
    else:
        gates.record("G1_gram_rederivation_subspace_identity",
                     worst < DW.SUBSPACE_ANGLE_TOL, **ev,
                     note="RECORDED not REQUIRED: the cloud was truncated, so this Gram is not "
                          "the matrix HUMBOLDT's banks were derived from and the comparison is "
                          "not a re-derivation check")

    # Per-fine-tune write energy is free to record and is the denominator W0-8's eval arm needs.
    for (t, L), by_repo in fro2_by.items():
        atlas.add_ci("fine_tune_write_energy_fro2", list(by_repo.values()), layer=L, type=t,
                     model_id=W.PG1_REPO, model_sha=base_sha,
                     flags={**H.type_flags(t), "n_fine_tunes": len(by_repo)})

    atlas.close()
    rep = atlas.to_parquet()
    srep = surprises.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "surprises": srep, "gates": gates.gates,
                  "result": {"worst_angle_rad": worst, "n_bit_identical": n_bit,
                             "n_cells": len(identities)},
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} worst_angle={worst:.3e} "
          f"bit_identical={n_bit}/{len(identities)}")


if __name__ == "__main__":
    main()
