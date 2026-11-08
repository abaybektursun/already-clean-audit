"""DARWIN Wave 0 — the free/CPU arms (§4 Wave 0). One module, `--arm` selects the arm.

Arms implemented here (all CPU, all `--no-weights`; PG1 is fetched from the Hub by hf_setup):

  W0-0   PREFLIGHT (journal amendment A1). Derive the topsigma32 bank for all 7x18 cells from the
         base matrices and publish it to s3://…/darwin/banks/. HUMBOLDT banked bottomsigma32 but
         not topsigma32, and §3/§7 require the structured top-k/bottom-k alternative in the same
         table as every core-specialness contrast. Gate G1a: the CO-DERIVED bottomsigma32 must
         land on the same subspace as HUMBOLDT's banked one — that is what proves the derivation
         path is the same path, and therefore that topsigma32 is its honest counterpart.
  W0-1a  Bank-side coordinate lists: per (type, layer, bank) leverage vector, the >5x-median
         coordinate set, and fixed top-k lists. Feeds the W0-1c overlap join.
  W0-2   Cross-layer core agreement: all-pairs principal angles between layers' banks, per type,
         plus shared-coordinate fraction. Emits the W1-R routing decision (§4/W1-R).

  pg run infra.aws_pg.jobs.darwin_w0 --args "--arm W0-0" --type r7i.2xlarge --no-weights \
      --s3-prefix results/darwin_w0 --budget-min 180
"""
import argparse
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

ARMS = ("W0-0", "W0-1a", "W0-2")


def _expectations(out):
    """§11: the expectation store. Wave-0 rows are pre-logged by scripts/darwin_freeze.py and the
    store is carried into the job; if it is absent the job builds it from the spec's REFUTER lines
    at first use, BEFORE any statistic is computed, which preserves the ordering §11 requires."""
    return D.Expectations(pathlib.Path(out).parent / "expectations.jsonl")


SPEC_EXPECTATIONS = {
    # arm: (REFUTER copied from §4, ONE expected direction copied from §4)
    "W0-0": ("re-derived bottomsigma32 does not match HUMBOLDT's banked bottomsigma32 subspace",
             "the co-derived bottomsigma32 reproduces the banked one to float tolerance, so "
             "topsigma32 is its honest structured counterpart"),
    "W0-1a": ("twin overlap bands indistinguishable",
              "core32 leverage is spikier than the haar twins, so its >5x-median coordinate set "
              "is smaller and more concentrated than a twin's"),
    "W0-2": ("twin overlap bands indistinguishable",
             "layers' core32 subspaces agree with each other above the independent-Haar reference "
             "k/n, and more than the twins do"),
}


def _log_expectation(exp, arm, utc):
    if exp.by_arm(arm):
        return exp.require(arm)
    refuter, direction = SPEC_EXPECTATIONS[arm]
    return exp.log(arm, refuter=refuter, direction=direction, utc=utc)


# ---------------------------------------------------------------------------
# W0-0 — derive + publish topsigma32, gate the derivation path.
# ---------------------------------------------------------------------------
def run_w0_0(out, atlas, gates, args, base_sha):
    import numpy as np

    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"), require_topsigma=False)
    w_base = W._load_decoder(W.PG1_REPO, base_sha)
    layers = args.layers or list(range(H.N_LAYERS))
    types = args.types or list(H.TYPES)

    identities, n_pub = [], 0
    for t in types:
        for L in layers:
            W0 = w_base[W.matrix_key(t, L)].numpy().astype(np.float64)
            bot, top, sigma = H.sigma_subspaces(W0, H.BANK_K)
            H.orthonormal_check(top)
            H.orthonormal_check(bot)
            ref = W.load_bank(bank_dir, t, L, "bottomsigma32")
            ident = DW.bank_identity(bot, ref)
            identities.append({"type": t, "layer": L, **ident})

            W.save_banks(out, t, L, {"topsigma32": top})
            n_pub += 1
            fl = {**H.type_flags(t), "derived_by": "darwin W0-0",
                  "sigma_top": float(sigma[0]), "sigma_bottom_used": float(sigma[H.BANK_K - 1]),
                  "economy_rank": int(sigma.size)}
            atlas.add("bank_derivation_max_principal_angle_rad",
                      value=ident["max_principal_angle_rad"], layer=L, type=t,
                      bank_id="bottomsigma32", model_id=W.PG1_REPO, model_sha=base_sha,
                      flags={**fl, "vs": "humboldt banked bottomsigma32",
                             "tol": DW.SUBSPACE_ANGLE_TOL,
                             "bit_identical": ident["bit_identical"],
                             "max_abs_diff_sign_aligned": ident["max_abs_diff_sign_aligned"]})
            atlas.add("bank_energy_fraction", value=H.bank_energy_fraction(W0, top), layer=L,
                      type=t, bank_id="topsigma32", model_id=W.PG1_REPO, model_sha=base_sha,
                      flags={**fl, "haar_reference": H.BANK_K / H.IN_DIM[t]})
            atlas.add("sigma_spectrum",
                      vector_ref=atlas.vector(f"W0-0__sigma__{t}__L{L}", sigma),
                      layer=L, type=t, bank_id="topsigma32", model_id=W.PG1_REPO,
                      model_sha=base_sha, n=int(sigma.size), flags=fl)
            del W0
        print(f"[W0-0] {t}: {len(layers)} layers derived", flush=True)
        D.s3_sync(out, args.s3_prefix)

    worst = max(i["max_principal_angle_rad"] for i in identities)
    n_bit = sum(1 for i in identities if i["bit_identical"])
    # GATED on the subspace, RECORDED on the elements — see darwin_w0.bank_identity for why the
    # spec's literal "bit-identity" is not a well-posed requirement on a basis.
    gates.require("G1a_bank_derivation_subspace_identity",
                  worst < DW.SUBSPACE_ANGLE_TOL,
                  worst_max_principal_angle_rad=worst, tol=DW.SUBSPACE_ANGLE_TOL,
                  n_cells=len(identities), n_bit_identical=n_bit,
                  worst_cell=max(identities, key=lambda i: i["max_principal_angle_rad"]))
    D.publish_banks(out)
    return {"n_published": n_pub, "worst_angle_rad": worst, "n_bit_identical": n_bit,
            "n_cells": len(identities)}


# ---------------------------------------------------------------------------
# W0-1a — bank-side coordinate lists.
# ---------------------------------------------------------------------------
def run_w0_1a(out, atlas, gates, args, base_sha):
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    layers = args.layers or list(range(H.N_LAYERS))
    types = args.types or list(H.TYPES)
    seen_banks, summary = set(), {}

    for t in types:
        n = H.IN_DIM[t]
        for L in layers:
            for bid in D.bank_ids_for(t):
                U = W.load_bank(bank_dir, t, L, bid)
                lev = H.leverage(U)
                sel = DW.threshold_coords(lev, DW.LEVERAGE_MULT)
                seen_banks.add(bid)
                fl = {**H.type_flags(t), "haar_expected_leverage": H.BANK_K / n,
                      "rule": f"leverage > {DW.LEVERAGE_MULT}x median"}
                atlas.add("leverage",
                          vector_ref=atlas.vector(f"W0-1a__lev__{t}__L{L}__{bid}", lev),
                          layer=L, type=t, bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          n=int(lev.size), flags=fl)
                atlas.add("leverage_participation_ratio", value=H.participation_ratio(lev),
                          layer=L, type=t, bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**fl, "flat_reference": n})
                atlas.add("leverage_spike_count", value=float(sel["n_selected"]), layer=L, type=t,
                          bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**fl, **{k: sel[k] for k in
                                          ("threshold", "median", "fraction_selected")}})
                atlas.add("leverage_spike_coords",
                          vector_ref=atlas.vector(f"W0-1a__spike__{t}__L{L}__{bid}",
                                                  sel["coords"], dtype="int64"),
                          layer=L, type=t, bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          n=sel["n_selected"], flags=fl)
                for k in DW.FIXED_TOP_K:
                    atlas.add("leverage_topk_coords",
                              vector_ref=atlas.vector(f"W0-1a__top{k}__{t}__L{L}__{bid}",
                                                      DW.top_k_coords(lev, k), dtype="int64"),
                              layer=L, type=t, bank_id=bid, model_id=W.PG1_REPO,
                              model_sha=base_sha, n=k, flags={**fl, "k": k})
                summary.setdefault(bid, []).append(sel["n_selected"])
        print(f"[W0-1a] {t}: {len(layers)} layers x {len(D.bank_ids_for(t))} banks "
              f"({atlas.n_rows} rows)", flush=True)
        D.s3_sync(out, args.s3_prefix)

    D.check_controls(seen_banks, arm="W0-1a")
    return {"spike_counts_by_bank": {k: sum(v) / len(v) for k, v in summary.items()},
            "n_banks": len(seen_banks)}


# ---------------------------------------------------------------------------
# W0-2 — cross-layer agreement + the W1-R routing decision.
# ---------------------------------------------------------------------------
def run_w0_2(out, atlas, gates, args, base_sha):
    import numpy as np

    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))
    layers = args.layers or list(range(H.N_LAYERS))
    types = args.types or list(H.TYPES)
    seen_banks, core_shared = set(), []

    for t in types:
        n = H.IN_DIM[t]
        for bid in D.bank_ids_for(t):
            banks = {L: W.load_bank(bank_dir, t, L, bid) for L in layers}
            spikes = {L: DW.threshold_coords(H.leverage(banks[L]), DW.LEVERAGE_MULT)["coords"]
                      for L in layers}
            seen_banks.add(bid)
            m_cos2, shared = [], []
            for i, Li in enumerate(layers):
                for Lj in layers[i + 1:]:
                    ag = DW.subspace_agreement(banks[Li], banks[Lj])
                    sf = DW.shared_coordinate_fraction(spikes[Li], spikes[Lj])
                    m_cos2.append(ag["mean_cos2"])
                    if sf == sf:
                        shared.append(sf)
                    atlas.add("cross_layer_mean_cos2", value=ag["mean_cos2"], type=t, bank_id=bid,
                              model_id=W.PG1_REPO, model_sha=base_sha,
                              flags={**H.type_flags(t), "layer_i": Li, "layer_j": Lj,
                                     "haar_reference": H.BANK_K / n,
                                     "shared_coordinate_fraction": sf,
                                     "min_angle_rad": ag["min_angle_rad"]})
            atlas.add_ci("cross_layer_mean_cos2_summary", m_cos2, type=t, bank_id=bid,
                         model_id=W.PG1_REPO, model_sha=base_sha,
                         flags={**H.type_flags(t), "haar_reference": H.BANK_K / n,
                                "n_pairs": len(m_cos2)})
            if shared:
                atlas.add_ci("cross_layer_shared_coordinate_fraction", shared, type=t,
                             bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                             flags={**H.type_flags(t), "n_pairs": len(shared)})
                if bid == "core32":
                    core_shared.extend(shared)
            else:
                # A bank with no spiky coordinates anywhere yields no shared fraction. That is a
                # RECORDED outcome (it is what a flat, unspiky bank looks like), not a skip.
                atlas.add("cross_layer_shared_coordinate_fraction_undefined", value=1.0, type=t,
                          bank_id=bid, model_id=W.PG1_REPO, model_sha=base_sha,
                          flags={**H.type_flags(t),
                                 "reason": "no coordinate exceeded the leverage threshold"})
        print(f"[W0-2] {t}: {len(D.bank_ids_for(t))} banks x {len(layers)*(len(layers)-1)//2} "
              f"pairs ({atlas.n_rows} rows)", flush=True)
        D.s3_sync(out, args.s3_prefix)

    D.check_controls(seen_banks, arm="W0-2")
    # §4/W1-R's mechanical branch. Recorded as a gate so the choice is frozen in the artifact
    # BEFORE W1-R runs and cannot be re-read afterwards.
    routing = DW.w1r_routing(float(np.mean(core_shared)) if core_shared else 0.0)
    gates.record("W1R_routing", True, **routing,
                 n_pairs=len(core_shared), basis="core32 spiky-coordinate overlap, all types")
    D.write_json(out, "w1r_routing.json", routing)
    return {"w1r_routing": routing, "n_core_pairs": len(core_shared)}


RUNNERS = {"W0-0": run_w0_0, "W0-1a": run_w0_1a, "W0-2": run_w0_2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--layers", default="")
    ap.add_argument("--types", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    a.layers = [int(x) for x in a.layers.split(",") if x.strip()] or None
    a.types = [x.strip() for x in a.types.split(",") if x.strip()] or None
    if a.smoke:
        a.layers = (a.layers or list(range(H.N_LAYERS)))[:2]
        a.types = a.types or ["q_proj", "down_proj"]
    for t in (a.types or []):
        assert t in H.TYPES, f"unknown type {t}"

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H.hf_setup()                       # FIRST — XET off + gated-repo auth (§0.2.8)
    from huggingface_hub import HfApi
    base_sha = HfApi().model_info(W.PG1_REPO).sha

    utc = D.utc_now()
    exp = _expectations(out)
    ref = _log_expectation(exp, a.arm, utc)          # §11 — BEFORE any statistic is computed

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W0",
                    expected_ref=ref)                # bound BEFORE any loop (§0.2.5)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W0", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": W.PG1_REPO, "base_sha": base_sha,
                "layers": a.layers or list(range(H.N_LAYERS)), "types": a.types or list(H.TYPES),
                "seed": a.seed, "expected_ref": ref, "started_utc": utc,
                "rules": {"LEVERAGE_MULT": DW.LEVERAGE_MULT, "OUTLIER_MULT": DW.OUTLIER_MULT,
                          "FIXED_TOP_K": list(DW.FIXED_TOP_K),
                          "SUBSPACE_ANGLE_TOL": DW.SUBSPACE_ANGLE_TOL,
                          "W1R_SHARED_FRACTION_GATE": DW.W1R_SHARED_FRACTION_GATE}}
    D.write_json(out, "manifest.json", manifest)

    result = RUNNERS[a.arm](out, atlas, gates, a, base_sha)

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates, "result": result,
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} vectors={rep['n_vectors']} "
          f"{json.dumps(result, default=str)[:300]}")


if __name__ == "__main__":
    main()
