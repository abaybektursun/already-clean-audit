"""DARWIN W0-3 / W0-4 — the family and scale censuses (§4). CPU.

  W0-3  Family census. For each external base with >=12 key-aligned full-FT derivatives:
        consensus Gram, spectrum, bottom/top-32 banks, leverage profiles, matched-rotation null.
        Deliverable: does a consensus-avoided, coordinate-spiky subspace exist OUTSIDE the Gemma
        family? Pivot (frozen): <12 usable FTs -> record census-impossible for that base WITH the
        count, and continue to the next base.
  W0-4  Scale census. The same pipeline on gemma-7b's zoo.

Both gate §8's "Gemma-only phenomenon" threat, which is why the pivot records a count rather than
a silence: "we could not census this base" and "this base has no such subspace" are different
findings and must not collapse into one another.

--probe-only runs SELECTION ALONE (config reads, no weight downloads) and writes the roster
sizes. Selection is cheap and the census is not: walking into a 200 GB download to discover the
roster was 4 models deep is avoidable, and the probe's output is itself the pivot's evidence.

  pg run infra.aws_pg.jobs.darwin_w0_census --args "--arm W0-3 --probe-only" \
      --type r7i.2xlarge --no-weights --s3-prefix results/darwin_w0_3_probe --budget-min 60
"""
import argparse
import json
import pathlib

from infra.pg import darwin_census as CEN
from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_core as H

# Frozen rosters (§2 external rosters). Ordered; each base is censused independently.
BASES = {
    "W0-3": ("meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B", "mistralai/Mistral-7B-v0.3",
             "google/gemma-2b"),
    "W0-4": ("google/gemma-7b",),
    # SHACKLETON V1(a)/(d): weight-side census banks for the 14B rung, computed on
    # CPU from HF deltas before any GPU rental (spec §2; roster gate PASSed 12/12)
    "SHK-14B": ("Qwen/Qwen2.5-14B",),
    # SHACKLETON V3: gemma-2b banks for the batch-dial substrate (spec §2 V3)
    "SHK-G2B": ("google/gemma-2b",),
}
SPEC_EXPECTATIONS = {
    "SHK-G2B": ("the gemma-2b zoo yields no usable banks for the V3 dial",
                "gemma-2b census banks exist and feed the V3 batch ladder"),
    "SHK-14B": ("the 14B zoo shows no consensus-avoided bottom subspace above its "
                "matched-rotation null (census yields no usable banks for V1(a)/(d))",
                "a consensus-avoided, coordinate-spiky subspace exists at 14B and its "
                "banks feed the V1 ablation cells"),
    "W0-3": ("no consensus-avoided, coordinate-spiky subspace exists outside the Gemma family",
             "at least one non-Gemma base shows a consensus-avoided bottom subspace with leverage "
             "spikiness above its matched-rotation null"),
    "W0-4": ("the gemma-7b zoo shows no consensus-avoided spiky subspace",
             "the same bottom-subspace structure appears at 7b scale"),
}
# gemma-2b appears in W0-3's roster as the POSITIVE CONTROL: the census pipeline must recover the
# known structure on the family it was developed on, or a null on a foreign base is uninterpretable.
POSITIVE_CONTROL = "google/gemma-2b"


def census_one(base_repo, api, out, atlas, gates, args):
    """Select, then (unless probing) stream the cloud and record the census."""
    import numpy as np
    from transformers import AutoConfig
    from infra.aws_pg.jobs.invariant_core import _delete_cache

    base_sha = api.model_info(base_repo).sha
    cfg = AutoConfig.from_pretrained(base_repo, revision=base_sha)
    geom = CEN.base_geometry(cfg)
    accepted, log = CEN.select_derivatives(api, base_repo, geom, n_accept=args.n_accept,
                                           n_examine=args.n_examine, out=out)
    n_full = len(accepted)
    print(f"[census] {base_repo}: {n_full} accepted of {log['n_candidates']} candidates "
          f"(geometry {geom})", flush=True)

    atlas.add("census_roster_size", value=float(n_full), model_id=base_repo,
              model_sha=base_sha,
              flags={"geometry": geom, "n_candidates": log["n_candidates"],
                     "min_for_census": CEN.MIN_FTS_FOR_CENSUS,
                     "census_possible": log["census_possible"],
                     "n_rejected_adapter": sum(1 for r in log["log"]
                                               if r.get("is_adapter")),
                     "is_positive_control": base_repo == POSITIVE_CONTROL})

    if not log["census_possible"]:
        # The frozen pivot. Recorded WITH the count: census-impossible is not the same finding as
        # census-ran-and-found-nothing, and §0.1 forbids letting the two read alike.
        gates.record(f"W0_census_possible__{base_repo.replace('/', '_')}", False,
                     n_full_ft=n_full, required=CEN.MIN_FTS_FOR_CENSUS,
                     n_candidates=log["n_candidates"],
                     note="pivot fired: census-impossible for this base, recorded with its count")
        return {"base": base_repo, "status": "census-impossible", "n_full_ft": n_full}

    gates.record(f"W0_census_possible__{base_repo.replace('/', '_')}", True,
                 n_full_ft=n_full, required=CEN.MIN_FTS_FOR_CENSUS)
    if args.probe_only:
        return {"base": base_repo, "status": "probe-only", "n_full_ft": n_full}

    layers = list(range(geom["num_hidden_layers"]))[:args.max_layers] or [0]
    types = list(args.types or CEN.ATTN_TYPES)
    shas = {r["id"]: api.model_info(r["id"]).sha for r in accepted}

    # HEADER-ONLY PREFLIGHT (defects D8/D11/D12/D13). The file listing says what a repo is named;
    # the safetensors headers say what it contains. Validating here turns a class of expensive
    # mid-census aborts into cheap recorded rejections, before a single weight is downloaded.
    base_shapes = CEN.decoder_shapes(base_repo, base_sha, types, layers)
    prevalid, prevalid_log = [], []
    for rec in accepted:
        ok, why = CEN.validate_candidate(rec["id"], shas[rec["id"]], base_shapes, types, layers)
        prevalid_log.append({"id": rec["id"], "ok": ok, "reason": why})
        print(f"[census] preflight {'OK  ' if ok else 'DROP'} {rec['id']}: {why}", flush=True)
        if ok:
            prevalid.append(rec)
    D.write_json(out, f"preflight__{base_repo.replace('/', '_')}.json", prevalid_log)
    atlas.add("census_preflight_passed", value=float(len(prevalid)), model_id=base_repo,
              model_sha=base_sha,
              flags={"n_accepted": len(accepted), "n_failed_preflight":
                     len(accepted) - len(prevalid), "log": prevalid_log,
                     "min_for_census": CEN.MIN_FTS_FOR_CENSUS})
    if len(prevalid) < CEN.MIN_FTS_FOR_CENSUS:
        gates.record(f"W0_census_possible__{base_repo.replace('/', '_')}", False,
                     n_full_ft=len(prevalid), required=CEN.MIN_FTS_FOR_CENSUS,
                     n_failed_preflight=len(accepted) - len(prevalid), log=prevalid_log,
                     note="pivot fired at header preflight: roster thinned below the threshold")
        return {"base": base_repo, "status": "census-impossible-at-preflight",
                "n_full_ft": len(prevalid), "preflight": prevalid_log}
    accepted = prevalid

    G, base, fro2_by, n_terms, dropped = CEN.census_grams(
        base_repo, base_sha, accepted, shas, types, layers, delete_cache=_delete_cache)
    n_used = len(accepted) - len(dropped)
    # A roster thinned at LOAD time (undeclared packing the config prefilter could not see) is
    # re-checked against the frozen threshold here, so it becomes a recorded census-impossible
    # rather than a quiet small-n census (defect D11).
    atlas.add("census_roster_used", value=float(n_used), model_id=base_repo, model_sha=base_sha,
              flags={"n_accepted": len(accepted), "n_dropped_at_load": len(dropped),
                     "dropped": dropped, "min_for_census": CEN.MIN_FTS_FOR_CENSUS})
    if n_used < CEN.MIN_FTS_FOR_CENSUS:
        gates.record(f"W0_census_possible__{base_repo.replace('/', '_')}", False,
                     n_full_ft=n_used, required=CEN.MIN_FTS_FOR_CENSUS,
                     n_dropped_at_load=len(dropped), dropped=dropped,
                     note="pivot fired AFTER load: roster thinned below the threshold by "
                          "derivatives whose tensors disagreed with the base")
        return {"base": base_repo, "status": "census-impossible-after-load",
                "n_full_ft": n_used, "dropped": dropped}

    seed = args.seed
    for t in types:
        for L in layers:
            lam, V = np.linalg.eigh(G[(t, L)])
            lam = np.clip(lam, 0.0, None)                       # ascending
            n = V.shape[0]
            core = np.ascontiguousarray(V[:, :H.BANK_K])
            anti = np.ascontiguousarray(V[:, -H.BANK_K:])
            # SHACKLETON V1(a): persist the banks themselves — leverage profiles
            # alone cannot drive an ablation; the Gram dies with the instance
            bdir = pathlib.Path(args.out) / "banks"
            bdir.mkdir(exist_ok=True)
            np.save(bdir / f"{t}__L{L}__core32.npy", core)
            np.save(bdir / f"{t}__L{L}__anti32.npy", anti)
            fl = {"base": base_repo, "n_full_ft": n_terms[(t, L)], "geometry": geom,
                  "n_dropped_at_load": len(dropped),
                  "n_ambient": int(n), "trace_G": float(lam.sum())}
            atlas.add("census_gram_spectrum",
                      vector_ref=atlas.vector(f"{args.arm}__lam__{base_repo.replace('/','_')}__{t}__L{L}",
                                              lam),
                      layer=L, type=t, bank_id="core32", model_id=base_repo,
                      model_sha=base_sha, n=int(lam.size), flags=fl)
            # Real banks AND their matched nulls, in the same table (§7).
            banks = {"core32": core, "anti32": anti}
            for s in H.HAAR_SEEDS:
                banks[f"haar32_s{s}"] = H.haar_subspace(n, H.BANK_K,
                                                        H.cell_seed(seed, base_repo, t, L, s))
            for bid, U in banks.items():
                sp = CEN.spikiness(U)
                atlas.add("census_leverage_participation_ratio", value=sp["participation_ratio"],
                          layer=L, type=t, bank_id=bid, model_id=base_repo, model_sha=base_sha,
                          flags={**fl, **sp})
                atlas.add("census_leverage_max", value=sp["max_leverage"], layer=L, type=t,
                          bank_id=bid, model_id=base_repo, model_sha=base_sha,
                          flags={**fl, **sp})
                sel = DW.threshold_coords(H.leverage(U), DW.LEVERAGE_MULT)
                atlas.add("census_leverage_spike_count", value=float(sel["n_selected"]), layer=L,
                          type=t, bank_id=bid, model_id=base_repo, model_sha=base_sha,
                          flags={**fl, **{k: sel[k] for k in ("threshold", "median",
                                                              "fraction_selected")}})
            # Consensus avoidance: how little of the cloud's write energy the bottom bank carries.
            atlas.add("census_bottom_eigenvalue_fraction",
                      value=float(lam[:H.BANK_K].sum() / lam.sum()), layer=L, type=t,
                      bank_id="core32", model_id=base_repo, model_sha=base_sha,
                      flags={**fl, "uniform_reference": H.BANK_K / n})
        D.s3_sync(out, args.s3_prefix)
    return {"base": base_repo, "status": "censused", "n_full_ft": n_used,
            "n_accepted": len(accepted), "n_dropped_at_load": len(dropped), "dropped": dropped,
            "n_types": len(types), "n_layers": len(layers)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(BASES))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--bases", default="")
    ap.add_argument("--types", default="")
    ap.add_argument("--n-accept", type=int, default=CEN.MIN_FTS_FOR_CENSUS)
    ap.add_argument("--n-examine", type=int, default=CEN.N_EXAMINE_DEFAULT)
    ap.add_argument("--max-layers", type=int, default=8,
                    help="layer budget per base; the census statistic is per-cell so a layer "
                         "subset is a coverage choice, recorded in the manifest, not a shortcut")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    a.types = [t.strip() for t in a.types.split(",") if t.strip()] or None
    bases = [b.strip() for b in a.bases.split(",") if b.strip()] or list(BASES[a.arm])

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H.hf_setup()
    from huggingface_hub import HfApi
    api = HfApi()

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W0",
                    expected_ref=ref)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W0", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "bases": bases, "n_accept": a.n_accept, "n_examine": a.n_examine,
                "max_layers": a.max_layers, "types": a.types or list(CEN.ATTN_TYPES),
                "min_for_census": CEN.MIN_FTS_FOR_CENSUS, "probe_only": a.probe_only,
                "positive_control": POSITIVE_CONTROL, "expected_ref": ref, "started_utc": utc,
                "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    results = []
    for b in bases:
        results.append(census_one(b, api, out, atlas, gates, a))
        D.s3_sync(out, a.s3_prefix)

    atlas.close()
    rep = atlas.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "gates": gates.gates, "results": results,
                  "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} "
          f"{json.dumps(results, default=str)[:400]}")


if __name__ == "__main__":
    main()
