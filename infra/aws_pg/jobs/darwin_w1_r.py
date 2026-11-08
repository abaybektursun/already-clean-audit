"""DARWIN W1-R — one global reader/writer, or eighteen. CPU. (§4/W1-R.)

§4/W1-R is the only arm in the program whose SHAPE is decided by another arm's number:

    "W1-R depends on W0-2: if cross-layer agreement is weak, per-layer reader/writer absorption
     variant (documented in the arm, chosen mechanically by a frozen angle threshold: mean
     shared-coordinate fraction >= 0.5 -> global R, else per-layer)."

So this arm does two things, in this order and no other:

  1. READS the routing decision out of W0-2's banked artifact. It does not recompute the
     threshold and it does not recompute the fraction — `darwin_w0.w1r_routing` already ran
     inside W0-2 and its output is on that summary. Re-deriving a frozen branch at consumption
     time is how a rule quietly becomes a choice (feedback_no_post_unblind_rule_swap).

  2. MEASURES what the chosen variant costs. The question a global R has to answer is whether one
     32-dim subspace can stand in for all eighteen per-layer cores, so the readout is ABSORPTION:
     the Grassmann overlap between each layer's own core32 and the single global basis built from
     all of them. Per-layer absorption is 1.0 by construction, which is exactly why the global
     number is the informative one and why storage is recorded next to it — the trade this arm
     exists to price is 32x2048 floats once against 18 times that.

THE CONTROL, and it is not optional. A stacked basis absorbs its own inputs to some degree no
matter what they are, so the same construction is run over every twin bank. If a global basis
built from haar32 absorbs the per-layer haar32s as well as the core global R absorbs the per-layer
cores, then absorption is a property of stacking 18 subspaces and says nothing about the core.

FAIL LOUD: no try/except, no silent fallback.

  pg run infra.aws_pg.jobs.darwin_w1_r --args "--arm W1-R" --type r7i.2xlarge --no-weights ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_ceq as C
from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W

SPEC_EXPECTATIONS = {
    "W1-R": ("a single global basis absorbs the per-layer cores no better than it absorbs the "
             "per-layer twins — absorption is a property of stacking, not of the core",
             "one global reader/writer absorbs the per-layer cores well enough to replace "
             "eighteen of them, and does so more than for any twin"),
}


def global_basis(bank_dir, t, layers, bank_id, rank):
    """The single basis standing in for every layer: top-`rank` left singular directions.

    `union_basis` returns the FULL span (rank = shared + private, up to 18x32 = 576 dimensions),
    which is the worst case a union has to budget for but is not a 32-dim reader. The arm wants
    the best rank-32 stand-in, so the stack's dominant `rank` left singular vectors are taken —
    the Eckart-Young optimum for approximating the stacked column space at that rank.
    """
    import numpy as np
    bases = [W.load_bank(bank_dir, t, L, bank_id) for L in layers]
    M = np.concatenate([np.asarray(B, dtype=np.float64) for B in bases], axis=1)
    Uu, s, _ = np.linalg.svd(M, full_matrices=False)
    assert Uu.shape[1] >= rank, f"stack has {Uu.shape[1]} directions, need {rank}"
    full = C.union_basis(bases)
    return np.ascontiguousarray(Uu[:, :rank]), s, int(full.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="W1-R", choices=("W1-R",))
    ap.add_argument("--run-id", default="darwin_w1_r")
    ap.add_argument("--s3-prefix", default="darwin/darwin_w1_r")
    ap.add_argument("--w0-2-prefix", default="darwin/darwin_w0_2")
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--rank", type=int, default=H.BANK_K)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    if a.smoke:
        types, layers = types[:1], layers[:4]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()

    import numpy as np
    from huggingface_hub import HfApi
    from infra.aws_pg.jobs.invariant_core_func import PG1_REPO

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    base_sha = HfApi().model_info(PG1_REPO).sha
    bank_dir = D.fetch_all_banks(H.cache_dir(out, "banks"))

    # (1) The routing decision, READ from W0-2 rather than recomputed.
    d = D.fetch_prior_arm(H.cache_dir(out, "w0_2"), a.w0_2_prefix)
    hits = sorted(pathlib.Path(d).rglob("w0_2_summary.json"))
    assert hits, f"W0-2 summary not found under {a.w0_2_prefix} — W1-R's shape is undecidable"
    w0_2 = json.loads(hits[0].read_text())
    routing = w0_2["result"]["w1r_routing"]
    assert routing["gate"] == "W1R_routing", routing
    # Re-running the frozen rule on the banked input is an AUDIT, not a re-decision: it must
    # agree, and disagreement is a defect rather than a new answer.
    recheck = DW.w1r_routing(routing["mean_shared_fraction"])
    assert recheck["variant"] == routing["variant"], (recheck, routing)
    variant = routing["variant"]
    print(f"[W1-R] routing from W0-2: shared_fraction={routing['mean_shared_fraction']:.4f} "
          f">= {routing['threshold']} -> {variant}", flush=True)

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W1",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W1", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "base_repo": PG1_REPO, "base_sha": base_sha, "types": list(types),
                "layers": layers, "rank": a.rank, "routing": routing,
                "routing_source": f"{a.w0_2_prefix}/w0_2_summary.json (read, not recomputed)",
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)
    gates.record("W1R_routing_read_from_W0_2", True, **routing,
                 source=manifest["routing_source"], audit_agrees=True)

    seen, summary = set(), {}
    for t in types:
        n_dim = H.IN_DIM[t]
        for bid in D.bank_ids_for(t):
            R, svals, union_rank = global_basis(bank_dir, t, layers, bid, a.rank)
            overlaps = []
            for L in layers:
                U = W.load_bank(bank_dir, t, L, bid)
                ag = DW.subspace_agreement(U, R)
                overlaps.append(ag["mean_cos2"])
                atlas.add("global_absorption", value=ag["mean_cos2"], layer=L, type=t,
                          bank_id=bid, dose=float(a.rank), incumbent_id="per_layer",
                          model_id=PG1_REPO, model_sha=base_sha,
                          flags={"variant": variant, "routing": routing,
                                 "per_layer_absorption": 1.0,
                                 "union_rank_full": union_rank,
                                 "n_layers": len(layers),
                                 "haar_expected": a.rank / n_dim,
                                 "storage_floats_global": a.rank * n_dim,
                                 "storage_floats_per_layer": a.rank * n_dim * len(layers),
                                 "definition": "mean cos^2 of principal angles between this "
                                               "layer's bank and the single global rank-32 "
                                               "stand-in built from all layers' banks"})
            atlas.add_ci("global_absorption_mean", overlaps, type=t, bank_id=bid,
                         dose=float(a.rank), incumbent_id="per_layer", model_id=PG1_REPO,
                         model_sha=base_sha,
                         flags={"variant": variant, "n_layers": len(layers),
                                "union_rank_full": union_rank,
                                "singular_values": svals[:64].tolist(),
                                "spectrum_note": "singular values of the 18-block stack; a flat "
                                                 "spectrum means the layers share nothing and a "
                                                 "global reader cannot exist at this rank",
                                "haar_expected": a.rank / n_dim})
            summary.setdefault(t, {})[bid] = {"mean": float(np.mean(overlaps)),
                                              "min": float(np.min(overlaps)),
                                              "max": float(np.max(overlaps)),
                                              "union_rank_full": union_rank}
            seen.add(bid)
            print(f"[W1-R] {t} {bid}: absorption mean={np.mean(overlaps):.4f} "
                  f"min={np.min(overlaps):.4f} union_rank={union_rank}", flush=True)
        D.s3_sync(out, a.s3_prefix)

    D.check_controls(seen, arm="W1-R")
    core_mean = float(np.mean([summary[t]["core32"]["mean"] for t in types]))
    twin_mean = max(float(np.mean([summary[t][x]["mean"] for t in types]))
                    for x in H.TWIN_IDS if x in summary[types[0]])
    gates.record("W1R_absorption_measured", True, variant=variant, banks=sorted(seen),
                 core_absorption=core_mean, best_twin_absorption=twin_mean,
                 refuter_fires=bool(core_mean <= twin_mean))
    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, "w1_r_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates,
                  "routing": routing, "variant": variant, "summary": summary,
                  "core_absorption": core_mean, "best_twin_absorption": twin_mean,
                  "refuter_fires": bool(core_mean <= twin_mean),
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_W1_R_OK rows={rep['n_rows']} variant={variant} "
          f"core={core_mean:.4f} best_twin={twin_mean:.4f}")


if __name__ == "__main__":
    main()
