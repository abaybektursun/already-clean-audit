"""M-3 — the Muon natural experiment (queue item 3). Wave 1.

    THE QUESTION. TENANT's reading is that the gradient sink is made by Adam's per-coordinate
    second moment, and M-Q stage 2 and M-A3 stage 2 both moved it by editing that term. If that is
    right, a base model trained with an optimizer that has NO per-coordinate denominator should not
    have the sink. Muon orthogonalises the momentum, so its update has a flat singular spectrum by
    construction — no coordinate can be singled out by a second moment that does not exist.

    REFUTER of TENANT-universality (pre-logged): a full-strength sink in a Muon-trained base.

THE ARM IS A SEARCH FIRST AND A MEASUREMENT SECOND, and the search is MECHANICAL so that its
outcome is a fact about the Hub rather than about who looked:

    stage roster   enumerate candidate Muon-trained open bases, count each one's key-aligned
                   full-weight derivatives, and record the count. A base with >= 12 supports the
                   consensus-Gram census this program's banks are built from.
    stage census   for the best base found, build consensus banks from its derivatives and measure
                   the two sink statistics — avoidance ratio and leverage spikiness — against the
                   same matched-random nulls the house uses everywhere else.

THREE OUTCOMES, ALL RECORDED, AND THE THIRD IS NOT A FAILURE:
  * a base with enough derivatives  -> the census runs and the refuter is decidable;
  * a base with checkpoints but not derivatives -> the CHECKPOINT-DELTA variant, which builds the
    consensus from consecutive training checkpoints instead of from fine-tunes (W0-9's estimand,
    same normalisation, and the arm says which one it used on every row);
  * neither -> COVERAGE-IMPOSSIBLE with the counts that make it impossible, so the next program
    inherits the search rather than repeating it.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_census as DC
from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-3"
# The mechanical candidate list: open bases whose model cards or papers declare Muon (or its
# Moonlight/Kimi lineage) as the pretraining optimizer, plus the two suites that publish
# per-step checkpoints so the checkpoint-delta variant has somewhere to go. Enumerated here so the
# search is auditable; each is probed, and the misses are recorded with the reason.
MUON_CANDIDATES = (
    "moonshotai/Moonlight-16B-A3B",
    "moonshotai/Moonlight-16B-A3B-Instruct",
    "moonshotai/Kimi-K2-Instruct",
    "essential-ai/rope-8b",
    "Motif-Technologies/Motif-2.6B",
    "PleIAs/Pleias-350m-Preview",
)
# Adam-trained references, probed by the SAME code path, because "Muon bases have few
# derivatives" is only informative next to how many an Adam base of similar popularity has.
ADAM_REFERENCES = ("google/gemma-2-2b", "meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B")
MIN_FTS = DC.MIN_FTS_FOR_CENSUS
# SIZE IS A COST, NOT A GATE (removed 2026-08-20, director).
# This probe used to reject any base above MAX_CENSUS_PARAMS = 1.2e10, a constant justified in
# comment as "twice the largest census this program has streamed" — an arithmetic that did not
# even match its own number (twice ~3e9 is 6e9). It rejected Kimi-K2-Instruct, the only Muon
# base above the derivative floor, and so turned a bandwidth question into a scientific dead end.
# Model size does not bound feasibility: forward passes stream layer by layer, and weight-side
# statistics range-read one tensor at a time. What size actually bounds is transfer and disk,
# which is an operator decision. So the probe now REPORTS the estimated census transfer instead
# of vetoing on it, and census_possible turns only on whether the family is large enough for the
# consensus Gram to be well posed.


def probe(api, repo: str) -> dict:
    """Existence, config shape, derivative count and checkpoint-revision count for one base.

    Every field is read from the Hub rather than assumed, and a repo that does not resolve is
    recorded as absent with the reason instead of dropped — an unavailable candidate is part of
    the search's result.
    """
    import re
    from infra.aws_pg.jobs.mercator_w0_roster import _classify, _derivatives
    # EVERY DERIVED KEY GETS A DEFAULT HERE, so the absent-repo early return below cannot drift
    # away from the full path. It already had: adding `blocked_by_unknown_size` to the tail left
    # the early return without it, and one nonexistent Muon candidate (essential-ai/rope-8b) then
    # took the whole roster down with a KeyError after every base had been probed (M-D93).
    rec = {"repo": repo, "exists": True, "n_derivatives": 0, "n_census_shortlist": 0,
           "n_declared_full": 0, "n_step_revisions": 0, "shortlist": [], "weight_bytes": 0,
           "n_params": 0, "est_params": 0.0, "size_source": None, "size_known": False,
           "size_feasible": False, "census_possible": False,
           "blocked_by_unknown_size": False, "checkpoint_variant_possible": False}
    # A CANDIDATE THAT DOES NOT RESOLVE IS DATA. `repo_exists` answers without raising, so the
    # search records the miss and keeps going instead of aborting the roster on the first gated or
    # renamed repo — which is the M-D3 failure mode, and the outcome of a mechanical search must
    # not depend on the order the candidates happen to be listed in.
    if not api.repo_exists(repo):
        return {**rec, "exists": False,
                "why": "repo does not resolve on the Hub (absent, renamed, or gated)"}
    info = api.model_info(repo)
    rec["exists"] = True
    rec["sha"] = info.sha
    rec["downloads"] = int(getattr(info, "downloads", 0) or 0)
    rec["tags"] = [t for t in (getattr(info, "tags", []) or [])][:40]
    cands = _derivatives(api, repo)
    cls = {k: _classify(v) for k, v in cands.items()}
    rec["n_derivatives"] = len(cands)
    rec["n_census_shortlist"] = sum(1 for v in cls.values() if v["census_shortlist"])
    rec["n_declared_full"] = sum(1 for v in cls.values() if v["is_declared_full"])
    rec["shortlist"] = sorted(k for k, v in cls.items() if v["census_shortlist"])[:64]
    refs = api.list_repo_refs(repo)
    steps = [b.name for b in refs.branches if re.search(r"step\d+", b.name)]
    rec["n_step_revisions"] = len(steps)
    # Repo size in bytes, summed over the weight shards, read from the Hub rather than guessed
    # from a name. `safetensors` metadata gives the parameter count when the repo publishes it.
    files = api.model_info(repo, files_metadata=True).siblings or []
    rec["weight_bytes"] = int(sum(f.size or 0 for f in files
                                  if f.rfilename.endswith((".safetensors", ".bin"))))
    st = getattr(api.model_info(repo), "safetensors", None)
    rec["n_params"] = int(getattr(st, "total", 0) or 0) if st is not None else 0
    # bf16 shards are 2 bytes per parameter; the byte count is the fallback when the repo does not
    # publish a parameter total.
    est_params = rec["n_params"] or rec["weight_bytes"] / 2
    rec["size_source"] = ("safetensors_total" if rec["n_params"] else
                          ("weight_bytes" if rec["weight_bytes"] else None))
    if not est_params:
        # THE PROBE RETURNED NOTHING FOR EVERY BASE in the first roster run — n_params 0 and
        # weight_bytes 0 across all six Muon repos — so `size_feasible` was False everywhere and
        # the arm would have concluded "no Muon base can be censused" from a FAILED PROBE rather
        # than from the Hub's contents (M-D89). A filter fed by a probe that always returns zero
        # rejects everything and looks like a finding. The config is the fallback: dense parameter
        # arithmetic, which under-counts an MoE and is still worth orders of magnitude more than 0.
        try:
            from transformers import AutoConfig
            c = AutoConfig.from_pretrained(repo, trust_remote_code=False)
            h = int(getattr(c, "hidden_size", 0) or 0)
            L = int(getattr(c, "num_hidden_layers", 0) or 0)
            v = int(getattr(c, "vocab_size", 0) or 0)
            i = int(getattr(c, "intermediate_size", 0) or 4 * h)
            est_params = (v * h) + L * (4 * h * h + 3 * h * i)
            rec["size_source"] = "config_dense_arithmetic"
        except Exception as e:                       # noqa: BLE001 — a Hub metadata boundary
            rec["size_probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    rec["est_params"] = float(est_params)
    rec["size_known"] = bool(est_params > 0)
    # bf16 weights, one base + MIN_FTS derivatives: the operator-facing cost of this census.
    rec["census_transfer_bytes_est"] = float(est_params) * 2.0 * (MIN_FTS + 1) if est_params > 0 else 0.0
    rec["census_possible"] = bool(rec["n_census_shortlist"] >= MIN_FTS)
    # A base with enough derivatives whose SIZE could not be determined is an open question, not a
    # rejection: it is reported separately so "no base qualifies" can never be a probe failure
    # wearing a coverage verdict.
    rec["blocked_by_unknown_size"] = bool(rec["n_census_shortlist"] >= MIN_FTS
                                          and not rec["size_known"])
    rec["checkpoint_variant_possible"] = bool(rec["n_step_revisions"] >= 4)
    return rec


def main():
    ap = argparse.ArgumentParser(description="M-3 — the Muon natural experiment")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="roster", choices=("roster", "census"))
    ap.add_argument("--base", default="", help="census stage: the base the roster stage selected")
    ap.add_argument("--adam-reference", default="google/gemma-2b",
                    help="census stage: an Adam-trained base put through the SAME pipeline")
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--n-fts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id=f"m_3_{a.stage}", register=meta["register"],
                 wave=meta["wave"], expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    from infra.aws_pg.jobs.mercator_w0_roster import _api
    api = _api()

    if a.stage == "roster":
        rows = {}
        for repo in MUON_CANDIDATES + ADAM_REFERENCES:
            group = "muon" if repo in MUON_CANDIDATES else "adam_reference"
            r = probe(api, repo)
            r["group"] = group
            rows[repo] = r
            at.add("muon_roster_derivative_count", value=float(r["n_census_shortlist"]),
                   model_id=repo, model_sha=r.get("sha"),
                   flags={**{k: v for k, v in r.items() if k != "shortlist"},
                          "min_for_census": MIN_FTS})
            print(f"[M-3] {group:14s} {repo:42s} shortlist={r['n_census_shortlist']:4d} "
                  f"declared_full={r['n_declared_full']:3d} steps={r['n_step_revisions']:4d} "
                  f"census={r['census_possible']}", flush=True)
        muon = {k: v for k, v in rows.items() if v["group"] == "muon"}
        adam = {k: v for k, v in rows.items() if v["group"] == "adam_reference"}
        censusable = sorted((k for k, v in muon.items() if v["census_possible"]),
                            key=lambda k: -muon[k]["n_census_shortlist"])
        ckpt = sorted((k for k, v in muon.items() if v["checkpoint_variant_possible"]),
                      key=lambda k: -muon[k]["n_step_revisions"])
        route = ("census" if censusable else ("checkpoint_delta" if ckpt else None))
        by_count = sorted(k for k, v in muon.items()
                          if v["n_census_shortlist"] >= MIN_FTS)
        unsized = sorted(k for k, v in muon.items() if v["blocked_by_unknown_size"])
        gates.record("m3_no_qualifying_base_was_dropped_for_an_unknown_size", not unsized,
                     blocked_by_unknown_size=unsized,
                     size_sources={k: v.get("size_source") for k, v in muon.items()},
                     est_params={k: v.get("est_params") for k, v in muon.items()},
                     outcome=None if not unsized else M.VOID_INSTRUMENT,
                     note=("a base with enough derivatives but no readable size is an open "
                           "question; if any appears here, 'no Muon base is censusable' is a "
                           "probe failure and not a fact about the Hub (M-D89)"))
        gates.record("m3_a_muon_base_supports_the_census", bool(censusable),
                     censusable=censusable,
                     above_derivative_floor=by_count,
                     census_transfer_bytes_est={k: muon[k]["census_transfer_bytes_est"]
                                                for k in by_count},
                     muon_counts={k: v["n_census_shortlist"] for k, v in muon.items()},
                     adam_counts={k: v["n_census_shortlist"] for k, v in adam.items()},
                     min_for_census=MIN_FTS,
                     note=("the Adam references are probed by the same code path, so 'Muon bases "
                           "have few derivatives' can be read against how many a comparable Adam "
                           "base has"))
        gates.record("m3_route_available", route is not None, route=route,
                     checkpoint_variant_bases=ckpt,
                     outcome=None if route else M.COVERAGE_IMPOSSIBLE,
                     note=("neither route means the natural experiment cannot be run on today's "
                           "open weights, recorded with the counts that make it so"))
        M.write_json(out, f"{ARM}_roster_result.json",
                     {"arm": ARM, "stage": "roster", "queue_item": meta["queue_item"],
                      "refuter": meta["refuter"], "expected_ref": expected_ref,
                      "provenance": M.provenance(), "utc": M.utc_now(),
                      "atlas": at.to_parquet(), "candidates": list(MUON_CANDIDATES),
                      "adam_references": list(ADAM_REFERENCES), "min_for_census": MIN_FTS,
                      "rows": rows, "censusable": censusable,
                      "checkpoint_variant_bases": ckpt, "route": route,
                      "outcome": None if route else M.COVERAGE_IMPOSSIBLE,
                      "selected_base": (censusable[0] if censusable else
                                        (ckpt[0] if ckpt else None))})
        print("M3_ROSTER_OK " + json.dumps({"route": route, "censusable": censusable,
                                            "checkpoint_bases": ckpt}))
        return 0

    # ---- census stage -------------------------------------------------------------------------
    # DELEGATED, NOT REIMPLEMENTED, AND RUN ON BOTH ARMS. `mercator_recensus.census_one` is the
    # verified path from a base repo to consensus banks and an avoidance ratio — the same estimand
    # every other census in this program used. Re-deriving it here would make a Muon-vs-Adam
    # difference indistinguishable from a difference of census code, and running it on ONLY the
    # Muon base would leave the comparison to a number quoted from another artifact.
    assert a.base, "the census stage needs --base, which the roster stage selects"
    from infra.aws_pg.jobs.mercator_recensus import census_one

    class _Args:
        pass

    ra = _Args()
    ra.types, ra.layers, ra.seed = a.types, a.layers, a.seed
    ra.n_accept, ra.n_examine = a.n_fts, 200

    def avoidance_of(model_id):
        """Mean avoidance ratio over the cells this run banked for one base.

        The ratio is already normalised against the flat expectation (bottom-k energy fraction
        divided by k/m), so 1.0 IS the matched-random null and a value below it is avoidance. That
        is why no separate null column appears here.
        """
        rows = [json.loads(ln) for ln in pathlib.Path(at.path).read_text().splitlines() if ln]
        vals = [r["value"] for r in rows
                if r["statistic"] == "recensus_avoidance_ratio" and r["model_id"] == model_id]
        return (float(np.mean(vals)) if vals else None), len(vals)

    census = {}
    for label, repo in (("muon", a.base), ("adam_reference", a.adam_reference)):
        print(f"[M-3] census ({label}) on {repo} via the re-census pipeline", flush=True)
        res = census_one(repo, api, out, at, gates, ra)
        av, n_cells = avoidance_of(repo)
        census[label] = {"repo": repo, "result": res, "avoidance_ratio_mean": av,
                         "n_cells": n_cells, "outcome": res.get("outcome")}
        print(f"[M-3] {label:15s} {repo:42s} avoidance={av} over {n_cells} cells "
              f"(outcome {res.get('outcome')})", flush=True)

    muon_av = census["muon"]["avoidance_ratio_mean"]
    adam_av = census["adam_reference"]["avoidance_ratio_mean"]
    impossible = any(c["outcome"] == M.COVERAGE_IMPOSSIBLE for c in census.values())
    # FULL STRENGTH means the Muon base avoids at least as much as the Adam reference does, both
    # measured against the same flat-expectation null of 1.0 by the same code.
    full_strength = (None if impossible or muon_av is None or adam_av is None
                     else bool(muon_av <= adam_av))
    at.add("muon_vs_adam_avoidance_ratio",
           value=float(muon_av) if muon_av is not None else 0.0, model_id=a.base,
           bank_id="core32",
           flags={"muon_avoidance": muon_av, "adam_avoidance": adam_av,
                  "adam_reference": a.adam_reference, "flat_null": 1.0,
                  "full_strength_sink": full_strength})
    gates.record("m3_muon_base_has_a_full_strength_sink", bool(full_strength),
                 base=a.base, muon_avoidance=muon_av, adam_reference=a.adam_reference,
                 adam_avoidance=adam_av, flat_null=1.0,
                 outcome=M.COVERAGE_IMPOSSIBLE if impossible else None,
                 note=("this gate PASSING is the refuter of TENANT-universality firing: an "
                       "optimizer with no per-coordinate denominator still produced the sink. "
                       "The ratio is normalised so 1.0 is the matched-random null."))
    M.write_json(out, f"{ARM}_census_result.json",
                 {"arm": ARM, "stage": "census", "queue_item": meta["queue_item"],
                  "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": at.to_parquet(),
                  "muon_base": a.base, "adam_reference": a.adam_reference,
                  "delegated_to": "mercator_recensus.census_one", "census": census,
                  "muon_avoidance_ratio": muon_av, "adam_avoidance_ratio": adam_av,
                  "flat_null": 1.0, "full_strength_sink": full_strength,
                  "outcome": M.COVERAGE_IMPOSSIBLE if impossible else None})
    print("M3_CENSUS_OK " + json.dumps({"muon": muon_av, "adam": adam_av,
                                        "full_strength_sink": full_strength}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
