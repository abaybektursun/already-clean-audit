"""MERCATOR re-census — the foreign-family census, WITH the eigenvectors banked. CPU.

WHY THIS EXISTS. DARWIN's census computed the consensus Gram for five families and banked, per
cell, only the eigenvalue vector (`W0-3__lam__<base>__<type>__L<k>.npy`). Every statistic that
needs to know WHERE in the coordinates the structure sits therefore cannot be computed from the
existing artifacts, and M-9a recorded two of them as COVERAGE-IMPOSSIBLE with exactly that
evidence. This job removes that limitation once, and four arms depend on it:

    M-9a   cross-layer verticality; the published-outlier-index comparison (needs spike INDICES,
           and the census banked a spike COUNT)
    M-8    four of its five ground truths — the predictor can only be scored against a MEASURED
           address, and only gemma-2b/PG1 has one banked today
    M-9b   the universality GPU cells, which project onto a foreign base's banks
    M-C4   within-family versus cross-family resonance profile stability

Nothing about the estimand changes: same selection rule, same Gram, same normalisation, same
`darwin_census` code. The only difference is that the eigenvectors are written out instead of
discarded, plus the two coordinate statistics that need them.

The selection is re-run rather than reused, because a roster is a live query and the census is
DEFINED by the cloud it sums over — pairing today's eigenvectors with yesterday's roster would
produce banks for a Gram nobody computed. The realized roster is recorded per base so the two can
be compared.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_census as CEN
from infra.pg import darwin_w0 as W0
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P
from infra.pg import mercator_rowgram as RG

ARM = "M-9a"          # the re-census serves M-9a's impossible cells; it is that arm's second stage
BASES = ("google/gemma-2b", "meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B",
         "mistralai/Mistral-7B-v0.3", "google/gemma-7b")
POSITIVE_CONTROL = "google/gemma-2b"
ATTN_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
SPIKE_MULT = 5.0


def bank_name(base_repo: str, t: str, layer: int, bank_id: str) -> str:
    return f"{base_repo.replace('/', '__')}__{t}__L{int(layer)}__{bank_id}.npy"


def spike_coords(U, mult: float = SPIKE_MULT):
    """The coordinate INDICES whose leverage exceeds `mult` x the flat expectation k/n.

    DARWIN's census banked the COUNT of these. The indices are what a comparison against a
    published outlier list needs, and they are what cross-layer verticality is computed from.
    """
    lev = H.leverage(U)
    n = lev.size
    k = float(lev.sum())
    return sorted(int(i) for i in np.flatnonzero(lev > mult * (k / n)))


def verticality(coords_by_layer: dict) -> dict:
    """Cross-layer shared-coordinate fraction: how much the spike sets of different layers agree.

    Reuses DARWIN's `shared_coordinate_fraction` so the number means the same thing it does on the
    house family. Reported as the mean over ALL layer pairs together with the pairwise matrix, so a
    single dominant pair cannot carry the summary.
    """
    layers = sorted(coords_by_layer)
    pairs, vals = {}, []
    for i, a in enumerate(layers):
        for b in layers[i + 1:]:
            f = W0.shared_coordinate_fraction(coords_by_layer[a], coords_by_layer[b])
            pairs[f"L{a}-L{b}"] = float(f)
            # `shared_coordinate_fraction` returns NaN when either layer has NO spiky coordinates,
            # which is the correct signal for "undefined", not a value. Averaging it in makes the
            # whole summary NaN and the atlas refuses the row — which is how the re-census died on
            # a foreign base after a clean 15/15 census (defect M-D72). Undefined pairs are counted
            # and excluded, the same way M-D71 handles an empty sink set in Wave 2.
            if f == f:
                vals.append(f)
    assert pairs, "verticality needs at least two layers"
    n_undef = len(pairs) - len(vals)
    return {"mean_shared_fraction": (float(np.mean(vals)) if vals else None),
            "min": (float(np.min(vals)) if vals else None),
            "max": (float(np.max(vals)) if vals else None),
            "n_pairs": len(pairs), "n_defined_pairs": len(vals),
            "n_undefined_pairs": n_undef, "pairwise": pairs, "layers": layers,
            "note": ("an undefined pair is one where a layer had no spiky coordinates at all; "
                     "mean_shared_fraction is None when no pair is defined")}


def census_one(base_repo, api, out, at, gates, a) -> dict:
    from transformers import AutoConfig
    from infra.aws_pg.jobs.invariant_core import _delete_cache

    base_sha = api.model_info(base_repo).sha
    cfg = AutoConfig.from_pretrained(base_repo, revision=base_sha)
    geom = CEN.base_geometry(cfg)
    accepted, log = CEN.select_derivatives(api, base_repo, geom, n_accept=a.n_accept,
                                           n_examine=a.n_examine, out=out)
    # A ROSTER IS A DETERMINISTIC HUB QUERY, WHICH IS WHY RELAUNCHING IS NOT A HEDGE. The same
    # base selects the same repos in the same order every time, so a census that wedges on a repo's
    # download wedges there again on a fresh instance — "just run it again" fixes nothing. A hang is
    # also not an exception: `HF_HUB_DOWNLOAD_TIMEOUT` is a per-read socket timeout and a connection
    # that trickles never trips it, so `retry_transport` never sees a failure to retry.
    #
    # `--exclude-repos` makes the second attempt actually DIFFER from the first. The excluded set is
    # recorded on the atlas row, because a census over a hand-trimmed cloud is a different estimand
    # from a census over the selected one and the artifact has to say so.
    excluded = [r.strip() for r in a.exclude_repos.split(",") if r.strip()]
    if excluded:
        before = len(accepted)
        accepted = [r for r in accepted if r["id"] not in excluded]
        hit = [r for r in excluded if r not in {x["id"] for x in accepted}]
        print(f"[recensus] {base_repo}: excluded {before - len(accepted)} of {before} "
              f"({excluded})", flush=True)
        at.add("recensus_roster_excluded", value=float(before - len(accepted)),
               model_id=base_repo, model_sha=base_sha,
               flags={"excluded": excluded, "matched": hit, "n_before": before,
                      "why": ("operator exclusion: the census is over a hand-trimmed cloud, not "
                              "the selected one")})
    n_full = len(accepted)
    print(f"[recensus] {base_repo}: {n_full} accepted of {log['n_candidates']} candidates",
          flush=True)
    at.add("recensus_roster_size", value=float(n_full), model_id=base_repo, model_sha=base_sha,
           flags={"n_candidates": log["n_candidates"], "min_for_census": CEN.MIN_FTS_FOR_CENSUS,
                  "geometry": geom})
    if n_full < CEN.MIN_FTS_FOR_CENSUS:
        gates.record(f"recensus_possible__{base_repo.replace('/', '_')}", False,
                     n_full_ft=n_full, required=CEN.MIN_FTS_FOR_CENSUS)
        return {"base": base_repo, "outcome": M.COVERAGE_IMPOSSIBLE, "n_full_ft": n_full,
                "why": "roster below the frozen census threshold"}

    shas = {r["id"]: api.model_info(r["id"]).sha for r in accepted}
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or \
        list(range(0, int(geom["num_hidden_layers"]), max(1, int(geom["num_hidden_layers"]) // 8)))
    types = [t.strip() for t in a.types.split(",") if t.strip()]

    # HEADER-ONLY PREFLIGHT, reused from the census verbatim: the file listing says what a repo is
    # NAMED, the safetensors headers say what it CONTAINS. Validating here turns a class of
    # expensive mid-census aborts into cheap recorded rejections before a single weight is
    # downloaded, and the roster is re-checked against the threshold afterwards so a thinned roster
    # becomes a recorded census-impossible rather than a quiet small-n census.
    base_shapes = CEN.decoder_shapes(base_repo, base_sha, types, layers)
    prevalid, prevalid_log = [], []
    for rec in accepted:
        ok, why = CEN.validate_candidate(rec["id"], shas[rec["id"]], base_shapes, types, layers)
        prevalid_log.append({"id": rec["id"], "ok": ok, "reason": why})
        print(f"[recensus] preflight {'OK  ' if ok else 'DROP'} {rec['id']}: {why}", flush=True)
        if ok:
            prevalid.append(rec)
    M.write_json(out, f"preflight__{base_repo.replace('/', '_')}.json", prevalid_log)
    at.add("recensus_preflight_passed", value=float(len(prevalid)), model_id=base_repo,
           model_sha=base_sha,
           flags={"n_accepted": len(accepted), "n_failed": len(accepted) - len(prevalid)})
    if len(prevalid) < CEN.MIN_FTS_FOR_CENSUS:
        gates.record(f"recensus_possible__{base_repo.replace('/', '_')}", False,
                     n_full_ft=len(prevalid), required=CEN.MIN_FTS_FOR_CENSUS,
                     note="roster thinned below the threshold at header preflight")
        return {"base": base_repo, "outcome": M.COVERAGE_IMPOSSIBLE,
                "n_full_ft": len(prevalid), "why": "thinned below threshold at header preflight",
                "preflight": prevalid_log}
    accepted = prevalid
    n_full = len(accepted)

    G, base_mats, fro2_by, n_terms, dropped = CEN.census_grams(
        base_repo, base_sha, accepted, shas, types, layers, delete_cache=_delete_cache)
    n_used = n_full - len(dropped)
    at.add("recensus_roster_used", value=float(n_used), model_id=base_repo, model_sha=base_sha,
           flags={"n_accepted": n_full, "n_dropped_at_load": len(dropped), "dropped": dropped})
    if n_used < CEN.MIN_FTS_FOR_CENSUS:
        gates.record(f"recensus_possible__{base_repo.replace('/', '_')}", False,
                     n_full_ft=n_used, required=CEN.MIN_FTS_FOR_CENSUS,
                     note="roster thinned below the threshold at load")
        return {"base": base_repo, "outcome": M.COVERAGE_IMPOSSIBLE, "n_full_ft": n_used,
                "why": "roster thinned below the threshold at load"}
    gates.record(f"recensus_possible__{base_repo.replace('/', '_')}", True, n_full_ft=n_used)

    bankdir = pathlib.Path(out) / "banks"
    bankdir.mkdir(parents=True, exist_ok=True)
    per_type, n_banked = {}, 0
    for t in types:
        coords_by_layer = {}
        for L in layers:
            g = G[(t, L)]
            n = g.shape[0]
            seed = H.cell_seed(a.seed, "recensus", base_repo, t, L)
            lb, Vb, lt, Vt, trace, _full = RG.eig_extremes(g, H.BANK_K)
            banks = {"core32": Vb, "anti32": Vt,
                     "complement32": H.complement_subspace(Vb, seed + 900)}
            for s in H.HAAR_SEEDS:
                banks[f"haar32_s{s}"] = H.haar_subspace(n, H.BANK_K, seed + s)
            for bid, U in banks.items():
                H.orthonormal_check(U)
                np.save(bankdir / bank_name(base_repo, t, L, bid), np.asarray(U, dtype=np.float64))
            n_banked += 1

            coords = spike_coords(banks["core32"])
            coords_by_layer[L] = coords
            np.save(bankdir / bank_name(base_repo, t, L, "spike_coords"),
                    np.asarray(coords, dtype=np.int64))
            av = RG.row_avoidance_from_extremes(lb, trace, n, H.BANK_K)
            at.add("recensus_avoidance_ratio", value=av["ratio"], layer=L, type=t,
                   bank_id="core32", model_id=base_repo, model_sha=base_sha,
                   flags={**av, "n_full_ft": n_used})
            for bid, U in banks.items():
                at.add("recensus_leverage_max", value=float(H.leverage(U).max()), layer=L, type=t,
                       bank_id=bid, model_id=base_repo, model_sha=base_sha, basis=M.BASIS_COORD,
                       flags={"n": int(n)})
                at.add("recensus_spike_count", value=float(len(spike_coords(U))), layer=L, type=t,
                       bank_id=bid, model_id=base_repo, model_sha=base_sha, basis=M.BASIS_COORD,
                       flags={"mult": SPIKE_MULT, "n": int(n)})
        if len(coords_by_layer) >= 2:
            v = verticality(coords_by_layer)
            per_type[t] = {"verticality": v,
                           "spike_counts": {L: len(c) for L, c in coords_by_layer.items()}}
            # A base whose layers have no spiky coordinates has no verticality to record. The
            # per-type entry is still banked in the result body with its undefined-pair count.
            if v["mean_shared_fraction"] is None:
                print(f"[recensus] {base_repo} {t}: verticality undefined "
                      f"({v['n_undefined_pairs']}/{v['n_pairs']} pairs have an empty layer)",
                      flush=True)
                continue
            at.add("recensus_verticality", value=v["mean_shared_fraction"], type=t,
                   model_id=base_repo, model_sha=base_sha, bank_id="core32",
                   flags={k2: v[k2] for k2 in ("min", "max", "n_pairs", "layers")})
    return {"base": base_repo, "outcome": None, "n_full_ft": n_used, "layers": layers,
            "types": types, "n_cells_banked": n_banked, "per_type": per_type,
            "roster": [r["id"] for r in accepted], "dropped": dropped, "geometry": geom}


def main():
    ap = argparse.ArgumentParser(description="MERCATOR re-census with eigenvector banking")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bases", default="")
    ap.add_argument("--types", default="q_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--exclude-repos", default="",
                    help="comma-list of derivative repo ids to drop from the selected roster; the "
                         "exclusion is recorded on the atlas because it changes the estimand")
    ap.add_argument("--n-accept", type=int, default=16)
    ap.add_argument("--n-examine", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bases = [b.strip() for b in a.bases.split(",") if b.strip()] or list(BASES)
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w0") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_9a_recensus", register=meta["register"],
                 wave=meta["wave"], expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    from huggingface_hub import HfApi
    api = HfApi()
    results = {}
    for base in bases:
        results[base] = census_one(base, api, out, at, gates, a)
        M.write_json(out, "recensus_partial.json", results)      # incremental save (§0.2.5)
        M.s3_sync(out, f"{M.PROGRAM}/recensus")

    # The positive control: the pipeline must recover the known structure on the family it was
    # developed on, or a foreign-base number is uninterpretable.
    if POSITIVE_CONTROL in results and results[POSITIVE_CONTROL].get("outcome") is None:
        pt = results[POSITIVE_CONTROL]["per_type"]
        # `bool(pt)` was the whole test, and a non-empty dict is not a recovered structure — the
        # gate could not fail on the one base where failing is the point (M-D73). A positive
        # control that cannot fail certifies nothing about the foreign bases it licenses. The
        # pipeline must produce a DEFINED verticality for at least one type: undefined means the
        # layers had no spiky coordinates, which on the house family is instrument failure.
        vert = {t: v["verticality"]["mean_shared_fraction"] for t, v in pt.items()}
        defined = {t: x for t, x in vert.items() if x is not None}
        gates.require("recensus_positive_control_recovers_structure", bool(defined),
                      base=POSITIVE_CONTROL, verticality=vert,
                      n_types=len(vert), n_types_defined=len(defined),
                      undefined_pairs={t: v["verticality"]["n_undefined_pairs"]
                                       for t, v in pt.items()},
                      note=("the family the pipeline was developed on must yield a defined "
                            "cross-layer verticality, or a foreign-base number has nothing to "
                            "be interpreted against"))

    at.check_controls_per_statistic([])       # the battery here is 6 of 8 by construction (no
    #                                           sigma twins: those need the base matrix per family)
    info = at.to_parquet()
    M.write_json(out, "recensus_result.json",
                 {"arm": ARM, "stage": "recensus", "queue_item": meta["queue_item"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "bases": bases, "results": results,
                  "unlocks": ["M-9a cross-layer verticality", "M-9a outlier-index comparison",
                              "M-8 foreign-family ground truths", "M-9b", "M-C4"],
                  "battery_note": ("six of the eight standing banks are built here; the two sigma "
                                   "twins need each family's own base matrix and are added by the "
                                   "arm that consumes them")})
    print("RECENSUS_OK " + json.dumps({"rows": info["n_rows"],
                                       "bases": {b: r.get("outcome") or "ok"
                                                 for b, r in results.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
