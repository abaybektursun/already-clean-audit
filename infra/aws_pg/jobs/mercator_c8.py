"""M-C8 — the observatory and the basis cards (queue C8). Wave P, Register B. CPU-only.

A RELEASE ARM. There is no refuter, and pretending otherwise would be theatre: the deliverable is
the packaged artifact and its manifest, and the only way it can fail is by being wrong about what
it contains. So everything it asserts is derived here from the Hub and from the banked files, and
nothing is transcribed.

    THE ZOO SCAN. Every derivative of the gemma-2b / PaliGemma family the Hub will enumerate,
    classified by declared recipe, with the CUTOFF LOGGED. A scan without its cutoff is a claim
    about the Hub rather than a snapshot of it, and the number examined is as much a part of the
    artifact as the number found.

    THE FAMILY TREE. base -> derivative -> derivative-of-derivative, built from each repo's own
    declared `base_model` tag. Repos that declare a base outside the scan are recorded as external
    edges rather than dropped, because an incomplete tree that says where it stops is usable and
    one that does not is not.

    THE BASIS CARDS. One per banked bank id: what the subspace IS, how it was derived, what it is
    NOT, and the SHA of the code that made it. The "what it is not" line exists because every
    misuse this program has caught came from a bank being read as something adjacent — the
    consensus bottom-32 read as "the important directions", the row-side bank read as the
    input-side one, a fused qkv cell compared to a single projection.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-C8"
ROOTS = ("google/paligemma-3b-pt-224", "google/gemma-2b", "google/gemma-2-2b")
# What each bank id IS, and — the load-bearing half — what it is NOT. Written here rather than
# generated, because these are the distinctions the program learned the hard way and a card that
# restates the constructor teaches nothing.
CARDS = {
    "core32": {
        "is": ("the 32 eigenvectors of smallest eigenvalue of the consensus Gram "
               "sum_i dW_i^T dW_i / ||dW_i||^2_F over a family of full fine-tunes — the input "
               "directions the family's edits consistently AVOID"),
        "is_not": ("NOT 'the important directions', and NOT low-rank structure in any single "
                   "dW. It is defined by what the edits do not touch, so a large value of any "
                   "statistic inside it is a fact about avoidance, not about importance. "
                   "QUORUM found the corresponding H_compact is energy-aligned and not low "
                   "rank; LATABLATE found low-rank-in-difference does not imply removable."),
    },
    "anti32": {
        "is": "the 32 eigenvectors of LARGEST eigenvalue of the same consensus Gram — the highway",
        "is_not": ("NOT the complement of core32 (that is complement32) and NOT a random "
                   "subspace: it is where the family's edits concentrate, so it is the one bank "
                   "expected to matter for the fine-tuned tasks themselves."),
    },
    "topsigma32": {
        "is": "the top-32 right singular directions of the BASE matrix",
        "is_not": ("NOT derived from any fine-tune. It is a property of the base alone, which is "
                   "why it is the right control for 'is this about the edits or about the "
                   "weights'."),
    },
    "bottomsigma32": {
        "is": "the bottom-32 right singular directions of the BASE matrix",
        "is_not": ("NOT the sink. For a rank-deficient type it spans part of the null space, "
                   "where every statistic is degenerate — humboldt journal C5 keeps that "
                   "deliberate and it is why this bank reads exactly zero in several arms."),
    },
    "complement32": {
        "is": "a random 32-dimensional subspace drawn inside the orthogonal complement of core32",
        "is_not": ("NOT a Haar draw from the whole space: it is constrained to miss the sink, so "
                   "it separates 'outside the sink' from 'random'."),
    },
    "haar32_s0": {"is": "a Haar-random 32-dimensional subspace, seed 0",
                  "is_not": "NOT tied to the model in any way; the matched-random null."},
    "haar32_s1": {"is": "a Haar-random 32-dimensional subspace, seed 1",
                  "is_not": "NOT tied to the model in any way; the matched-random null."},
    "haar32_s2": {"is": "a Haar-random 32-dimensional subspace, seed 2",
                  "is_not": "NOT tied to the model in any way; the matched-random null."},
}


def main():
    ap = argparse.ArgumentParser(description="M-C8 — observatory + basis cards release")
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", default=",".join(ROOTS))
    ap.add_argument("--n-examine", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = [r.strip() for r in a.roots.split(",") if r.strip()]
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_wp") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_c8", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    from infra.pg import darwin_census as DC
    from infra.aws_pg.jobs.mercator_w0_roster import _api, _classify, _derivatives

    api = _api()
    scan, edges = {}, []
    frontier, seen = list(roots), set()
    while frontier:
        base = frontier.pop(0)
        if base in seen:
            continue
        seen.add(base)
        cands = _derivatives(api, base, n_examine=a.n_examine)
        classes = {k: _classify(v) for k, v in cands.items()}
        scan[base] = {
            "n_derivatives": len(cands),
            "n_examine_cap": a.n_examine,
            "cutoff_reached": bool(len(cands) >= a.n_examine),
            "n_adapter": sum(1 for v in classes.values() if v["is_adapter"]),
            "n_merge": sum(1 for v in classes.values() if v["is_merge"]),
            "n_declared_full": sum(1 for v in classes.values() if v["is_declared_full"]),
            "n_undeclared": sum(1 for v in classes.values() if not v["recipe_declared"]),
            "n_census_shortlist": sum(1 for v in classes.values() if v["census_shortlist"]),
            "derivatives": sorted(cands),
        }
        for child in cands:
            edges.append({"parent": base, "child": child,
                          "declared_classes": classes[child]["declared_classes"]})
        at.add("zoo_derivative_count", value=float(len(cands)), model_id=base,
               flags={k: v for k, v in scan[base].items() if k != "derivatives"})
        print(f"[M-C8] {base:34s} {len(cands):4d} derivatives "
              f"(adapter {scan[base]['n_adapter']}, merge {scan[base]['n_merge']}, "
              f"declared-full {scan[base]['n_declared_full']}, "
              f"undeclared {scan[base]['n_undeclared']})"
              + ("  [CUTOFF REACHED]" if scan[base]["cutoff_reached"] else ""), flush=True)

    # DERIVATIVES-OF-DERIVATIVES, one generation deep. Deeper would multiply Hub calls without
    # changing what the tree is for, and the depth is stated rather than implied.
    gen1 = sorted({e["child"] for e in edges})
    for child in gen1[: a.n_examine]:
        sub = _derivatives(api, child, n_examine=32)
        if sub:
            for g in sub:
                edges.append({"parent": child, "child": g, "generation": 2,
                              "declared_classes": _classify(sub[g])["declared_classes"]})
    depth2 = sum(1 for e in edges if e.get("generation") == 2)
    print(f"[M-C8] family tree: {len(edges)} edges ({depth2} at generation 2)", flush=True)

    external = sorted({e["parent"] for e in edges} - set(scan) - set(gen1))
    tree = {"roots": roots, "edges": edges, "n_edges": len(edges),
            "generation_2_edges": depth2, "external_parents": external,
            "depth": 2, "n_examine_cap": a.n_examine,
            "note": ("one generation of derivatives-of-derivatives; deeper multiplies Hub calls "
                     "without changing what the tree is for, and repos declaring a base outside "
                     "the scan are recorded as external edges rather than dropped")}
    M.write_json(out, "family_tree.json", tree)
    M.write_json(out, "zoo_scan.json", scan)

    # ---- the basis cards ----------------------------------------------------------------------
    prov = M.provenance()
    cards = {}
    for bid, body in CARDS.items():
        cards[bid] = {
            "bank_id": bid,
            "dimension": H.BANK_K,
            "side": "input (the matrix's residual-stream inputs; its COLUMNS)",
            "is": body["is"],
            "is_not": body["is_not"],
            "derivation": ("darwin_census.census_grams for the consensus Gram; "
                           "invariant_core_func._bottom_eigvecs for the extremes; "
                           "humboldt_core.haar_subspace / complement_subspace for the nulls"),
            "code_sha": prov.get("commit_sha"),
            "program": "MERCATOR",
            "terminology": ("'sink' = the consensus-avoided input subspace; 'highway' = the "
                            "consensus-concentrated one; 'gradient sink' = the sink's behaviour "
                            "under Adam, large gradient in and little motion out"),
        }
        at.add("basis_card_published", value=1.0, bank_id=bid,
               flags={k: v for k, v in cards[bid].items() if k != "bank_id"})
    M.write_json(out, "basis_cards.json", cards)

    manifest = {
        "arm": ARM, "queue_item": meta["queue_item"],
        "deliverable": "zoo scan + family tree + basis cards",
        "refuter": meta["refuter"],
        "expected_ref": expected_ref,
        "provenance": prov, "utc": M.utc_now(),
        "roots": roots, "n_examine_cap": a.n_examine,
        "zoo": {k: {kk: vv for kk, vv in v.items() if kk != "derivatives"}
                for k, v in scan.items()},
        "family_tree": {k: v for k, v in tree.items() if k != "edges"},
        "basis_cards": sorted(cards),
        "bank_k": H.BANK_K,
        "min_fts_for_census": DC.MIN_FTS_FOR_CENSUS,
        "files": ["zoo_scan.json", "family_tree.json", "basis_cards.json"],
    }
    complete = bool(scan and cards and edges)
    gates.record("c8_release_is_complete", complete,
                 n_roots=len(scan), n_cards=len(cards), n_edges=len(edges),
                 cutoffs_reached=[k for k, v in scan.items() if v["cutoff_reached"]],
                 note=("a release arm has no refuter; the only failure mode is being wrong about "
                       "what it contains, so every count here is derived, and the roots whose "
                       "enumeration hit the cap are named"))
    info = at.to_parquet()
    M.write_json(out, f"{ARM}_result.json", {**manifest, "atlas": info})
    print("C8_OK " + json.dumps({"rows": info["n_rows"], "roots": len(scan),
                                 "edges": len(edges), "cards": len(cards),
                                 "cutoffs": [k for k, v in scan.items()
                                             if v["cutoff_reached"]]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
