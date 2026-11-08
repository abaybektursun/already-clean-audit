"""MERCATOR Wave 0 — the three roster/availability probes.

  M-V  (standing W0-5)  VLA ecosystem census. Mechanical roster search over the §2 VLA roster for
                        a base with >= 12 key-aligned full-FT derivatives. If one qualifies, the
                        census runs; if none does, the deliverable is COVERAGE-IMPOSSIBLE WITH THE
                        COUNT — the count is the point, because "we looked and there were N" is a
                        different statement from "we did not look".
  M-D4 (queue D4)       Recipe-forensics roster probe: does ANY base now have >= 2 held-out
                        minority-recipe fine-tunes? If yes, W0-10's VOID-underpowered verdict can
                        be unparked; if no, the probe itself is the record.
  M-D3 (queue D3)       Loss-spike retrodiction retry against OLMo's published training logs.
                        Pivot named in §4: logs genuinely absent -> COVERAGE-IMPOSSIBLE with
                        evidence.

Every one of these can legitimately end in COVERAGE-IMPOSSIBLE, and each records what it searched
and what it found, so a null here is an observation with a denominator rather than an absence.

FAIL LOUD: no try/except around the science. The ONE guarded boundary is a per-repo Hub metadata
fetch inside the roster walk, where a single deleted/gated repo must not abort a survey of
hundreds — and every skip is COUNTED and REPORTED, never silent.
"""
import argparse
import json
import pathlib
import sys

from infra.pg import darwin_census as DC
from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARMS = ("M-V", "M-D4", "M-D3")

# §2's VLA roster. Bases are probed in this order; the search is mechanical and the log is kept.
VLA_BASES = ("openvla/openvla-7b", "lerobot/smolvla_base", "physical-intelligence/fast",
             "openvla/openvla-7b-prismatic")
# Public suites whose training logs would carry loss spikes (M-D3).
OLMO_LOG_CANDIDATES = (
    ("allenai/OLMo-2-0425-1B", "training log / wandb export in the repo"),
    ("allenai/OLMo-2-1124-7B", "training log / wandb export in the repo"),
    ("allenai/OLMo-2-0425-1B-stage1", "per-stage checkpoint suite"),
)
LOG_FILE_HINTS = ("loss", "log", "wandb", "metrics", "train")
MIN_FTS = DC.MIN_FTS_FOR_CENSUS          # 12, reused so M-V and M-3 share one threshold


def _api():
    """An authenticated, ONLINE Hub client.

    `run_job.sh` exports HF_HUB_OFFLINE=1, and `huggingface_hub` freezes that value into a module
    CONSTANT at import time — so flipping the environment variable after the package is imported
    has no effect. `H.hf_setup()` must therefore run before the first import of the package
    anywhere in the process, which is why main() calls it as its very first statement. The
    assertions below turn a regression of that ordering into a loud failure at the call site
    instead of an `OfflineModeIsEnabled` traceback three frames into the Hub client.
    """
    import os
    assert os.environ.get("HF_HUB_OFFLINE") == "0", (
        "HF_HUB_OFFLINE is not 0 — H.hf_setup() has not run, and this arm needs the Hub")
    # `constants` must be imported as a SUBMODULE: the package's lazy-attribute machinery does not
    # expose it as `huggingface_hub.constants`.
    from huggingface_hub import HfApi, constants
    assert constants.HF_HUB_OFFLINE is False, (
        "huggingface_hub was imported BEFORE hf_setup() flipped HF_HUB_OFFLINE, so the package "
        "froze offline mode into its constants. Move the hf_setup() call earlier in main().")
    return HfApi()


def _derivatives(api, base_repo: str, n_examine: int = 200) -> dict:
    """Every model that declares `base_repo` as its base, by each of the three tag forms DARWIN
    uses. Returns the raw candidate list plus the counts, with no filtering applied yet — the
    filtering is the next step and must be visible separately from the search."""
    seen = {}
    for form in DC.TAG_FORMS:
        tag = form.format(base_repo)
        for m in api.list_models(filter=tag, limit=n_examine):
            seen[m.id] = {"id": m.id, "tag_form": form,
                          "tags": list(getattr(m, "tags", []) or []),
                          "downloads": int(getattr(m, "downloads", 0) or 0)}
    return seen


ADAPTER_TAGS = ("peft", "lora", "adapter", "qlora")
MERGE_TAGS = ("merge", "mergekit")
FULL_FT_TAGS = ("full-finetune", "full_finetune", "fully-finetuned", "sft", "full-parameter")


def _classify(cand: dict) -> dict:
    """Three declared classes plus UNDECLARED — and the fourth one is the important one.

    An earlier version of this function called anything not tagged adapter-or-merge a
    "full_ft_candidate". That silently equated "declares nothing" with "declares a full
    fine-tune", which made M-D4's minority-recipe count enormous on every base and therefore made
    its criterion one that could not fail. Absence of a tag is absence of evidence; it gets its
    own bucket, and any statement about recipe classes is made over the DECLARED models only.
    """
    tags = {t.lower() for t in cand["tags"]}
    is_adapter = any(any(a in t for a in ADAPTER_TAGS) for t in tags)
    is_merge = any(any(a in t for a in MERGE_TAGS) for t in tags)
    is_full = any(any(a in t for a in FULL_FT_TAGS) for t in tags)
    declared = [n for n, b in (("adapter", is_adapter), ("merge", is_merge), ("full", is_full))
                if b]
    return {"is_adapter": is_adapter, "is_merge": is_merge, "is_declared_full": is_full,
            "declared_classes": declared,
            "recipe_declared": bool(declared),
            # For a CENSUS the question is different: which repos are plausibly full-weight
            # derivatives worth shape-validating. Untagged repos qualify for that, and the label
            # says so — it is a shortlist for validation, not a claim about the recipe.
            "census_shortlist": not (is_adapter or is_merge)}


def run_mv(at, gates) -> dict:
    """M-V — the VLA roster probe."""
    api = _api()
    log, qualifying = {}, []
    for base in VLA_BASES:
        found = _derivatives(api, base)
        cls = {k: _classify(v) for k, v in found.items()}
        short = sorted(k for k, c in cls.items() if c["census_shortlist"])
        log[base] = {"n_declaring_this_base": len(found),
                     "n_census_shortlist": len(short),
                     "n_adapters": sum(1 for c in cls.values() if c["is_adapter"]),
                     "n_merges": sum(1 for c in cls.values() if c["is_merge"]),
                     "n_recipe_declared": sum(1 for c in cls.values() if c["recipe_declared"]),
                     "census_shortlist": short[:50],
                     "passes_shortlist_threshold": len(short) >= MIN_FTS}
        at.add("vla_roster_shortlist_count", value=float(len(short)), model_id=base,
               flags={"n_declaring_this_base": len(found), "threshold": MIN_FTS,
                      "passes_shortlist_threshold": len(short) >= MIN_FTS,
                      "meaning": ("repos declaring this base and NOT tagged adapter/merge; a "
                                  "shortlist for shape validation, not a validated count")})
        if len(short) >= MIN_FTS:
            qualifying.append(base)
        print(f"[M-V] {base}: {len(found)} declare this base, {len(short)} on the census "
              f"shortlist ({'passes' if len(short) >= MIN_FTS else 'below'} threshold {MIN_FTS})",
              flush=True)

    outcome = None if qualifying else M.COVERAGE_IMPOSSIBLE
    gates.record("mv_roster_passes_shortlist_threshold", bool(qualifying), qualifying=qualifying,
                 threshold=MIN_FTS, bases_probed=list(VLA_BASES))
    at.add("vla_roster_qualifying_bases", value=float(len(qualifying)),
           flags={"bases_probed": list(VLA_BASES), "threshold": MIN_FTS, "outcome": outcome})
    return {"arm": "M-V", "bases_probed": list(VLA_BASES), "roster_log": log,
            "qualifying_bases": qualifying, "threshold": MIN_FTS,
            "outcome": outcome,
            "stage": "roster shortlist only",
            "what_this_does_not_establish": (
                "the shortlist counts repos that DECLARE this base and are not tagged "
                "adapter/merge. The census threshold the other five families were measured at is "
                "KEY-ALIGNED full-FT derivatives, which requires per-repo shape validation "
                "(darwin_census.validate_candidate). A base passing this shortlist has NOT been "
                "shown to support a census; it has been shown to be worth validating."),
            "note": ("if no base passes, COVERAGE-IMPOSSIBLE is a measured statement: N bases "
                     "probed with a per-base count, not an absence of looking")}


def run_md4(at, gates) -> dict:
    """M-D4 — the recipe-forensics roster probe.

    "Minority recipe" is defined from the roster itself rather than assumed: for each base, group
    its derivatives by declared recipe class (adapter / merge / full), and a class is a MINORITY
    recipe when it holds fewer than half the derivatives. W0-10 needs >= 2 HELD-OUT minority-recipe
    models on one base, so the probe reports the per-base minority counts and whether any base
    reaches 2.
    """
    api = _api()
    bases = [DC.__dict__.get("PG1_REPO", "google/paligemma-3b-pt-224"),
             "google/gemma-2b", "meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B",
             "mistralai/Mistral-7B-v0.3", "google/gemma-7b"]
    log, reached = {}, []
    for base in bases:
        found = _derivatives(api, base)
        cls = {k: _classify(v) for k, v in found.items()}
        n_total = len(cls)
        declared = {k: c for k, c in cls.items() if c["recipe_declared"]}
        counts = {"adapter": sum(1 for c in declared.values() if c["is_adapter"]),
                  "merge": sum(1 for c in declared.values() if c["is_merge"]),
                  "full": sum(1 for c in declared.values() if c["is_declared_full"])}
        n_declared = len(declared)
        # THE MINORITY CLASS IS THE RAREST NON-EMPTY DECLARED CLASS — one class, not the sum of
        # every class under half. Summing "everything below 50%" made the count enormous and the
        # criterion unfailable; W0-10 was underpowered because the RARE class had too few members,
        # so the rare class is what has to be counted.
        nonempty = {k: v for k, v in counts.items() if v > 0}
        rarest = min(nonempty, key=nonempty.get) if nonempty else None
        n_rarest = nonempty[rarest] if rarest else 0
        log[base] = {"n_total_derivatives": n_total, "n_recipe_declared": n_declared,
                     "declared_counts": counts, "rarest_declared_class": rarest,
                     "n_in_rarest_class": n_rarest, "reaches_two": n_rarest >= 2,
                     "undeclared_fraction": (1.0 - n_declared / n_total) if n_total else None}
        at.add("recipe_rarest_declared_class_count", value=float(n_rarest), model_id=base,
               flags={"declared_counts": counts, "n_total_derivatives": n_total,
                      "n_recipe_declared": n_declared, "rarest_class": rarest,
                      "criterion": "W0-10 needs >=2 models in the rarest DECLARED recipe class"})
        if n_rarest >= 2:
            reached.append(base)
        print(f"[M-D4] {base}: {n_total} derivatives, {n_declared} declare a recipe, "
              f"counts {counts}, rarest class {rarest}={n_rarest}", flush=True)
    gates.record("md4_unparks_w0_10", bool(reached), bases_reaching_two=reached,
                 criterion=">=2 models in the rarest DECLARED recipe class on one base")
    return {"arm": "M-D4", "bases_probed": bases, "roster_log": log,
            "bases_reaching_two_minority": reached,
            "w0_10_unparked": bool(reached),
            "criterion": ">=2 models in the rarest DECLARED recipe class",
            "note": ("recipe class is read from DECLARED tags only. A repo with no recipe tag is "
                     "counted as undeclared, never as a full fine-tune — treating silence as a "
                     "declaration is what made an earlier version of this criterion unfailable. "
                     "The undeclared fraction is reported per base so the metadata's coverage is "
                     "visible alongside the counts.")}


def run_md3(at, gates) -> dict:
    """M-D3 — are OLMo's published training logs reachable at all?

    The arm's science needs a per-step loss series to retrodict spikes against. This probe walks
    each candidate repo's FILE LIST for anything that could carry one, and reports what it found.
    §4's named pivot applies when nothing does.
    """
    from huggingface_hub.errors import RepositoryNotFoundError
    api = _api()

    # Discover rather than only guess: the hand-written candidates are a floor, and whatever the
    # Hub currently publishes under the OLMo-2 name is added to them. A guessed repo id that does
    # not exist is a PROBE RESULT, not an arm-ending error — the same boundary darwin_census draws
    # for candidate enumeration over third-party repos — and every miss is counted and named.
    discovered = sorted({m.id for m in api.list_models(author="allenai", search="OLMo-2",
                                                       limit=200)})
    probe = [(r, why) for r, why in OLMO_LOG_CANDIDATES]
    probe += [(r, "discovered by Hub search author=allenai search=OLMo-2") for r in discovered
              if r not in {x for x, _ in OLMO_LOG_CANDIDATES}]
    print(f"[M-D3] probing {len(probe)} repos ({len(discovered)} discovered)", flush=True)

    found, log, absent = {}, {}, []
    for repo, why in probe:
        try:
            files = [f.rfilename for f in api.model_info(repo, files_metadata=False).siblings]
        except RepositoryNotFoundError:
            absent.append(repo)
            log[repo] = {"why_probed": why, "exists": False}
            continue
        hits = sorted(f for f in files
                      if any(h in f.lower() for h in LOG_FILE_HINTS)
                      and f.rsplit(".", 1)[-1].lower() in ("csv", "json", "jsonl", "txt", "log",
                                                           "parquet"))
        log[repo] = {"why_probed": why, "exists": True, "n_files": len(files),
                     "candidate_log_files": hits}
        at.add("olmo_log_candidate_files", value=float(len(hits)), model_id=repo,
               flags={"n_files": len(files), "hits": hits[:20], "hints": list(LOG_FILE_HINTS),
                      "why_probed": why})
        if hits:
            found[repo] = hits

    outcome = None if found else M.COVERAGE_IMPOSSIBLE
    # The outcome belongs ON THE GATE, not only in the atlas flags and the result body: the
    # report and the critic read gates.json, and M-D3's refuter pre-registers this exact pivot
    # ("logs genuinely absent -> COVERAGE-IMPOSSIBLE with evidence"). Without the stamp the run
    # reads as an undecided FAIL rather than the coverage verdict it was designed to reach.
    gates.record("md3_logs_reachable", bool(found), repos_with_candidates=sorted(found),
                 n_probed=len(probe), n_absent=len(absent), outcome=outcome)
    at.add("olmo_logs_reachable", value=float(len(found)),
           flags={"n_probed": len(probe), "n_absent": len(absent), "absent": absent,
                  "outcome": outcome})
    return {"arm": "M-D3", "repos_probed": [r for r, _ in probe], "n_probed": len(probe),
            "repos_absent": absent, "n_discovered": len(discovered),
            "probe_log": log, "repos_with_candidate_logs": sorted(found), "outcome": outcome,
            "note": ("this probe establishes REACHABILITY only. A repo with candidate files still "
                     "needs the series parsed and aligned to the checkpoint grid before any "
                     "retrodiction claim; that is the arm's next stage and it does not run if this "
                     "probe returns nothing")}


def main():
    ap = argparse.ArgumentParser(description="MERCATOR Wave 0 — roster / availability probes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    a = ap.parse_args()
    # FIRST, before anything can import huggingface_hub: the package freezes HF_HUB_OFFLINE into a
    # module constant at import time, and run_job.sh sets it to 1.
    H.hf_setup()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = a.arm
    meta = P.ARMS[arm]

    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w0") / "expectations.jsonl")
    expected_ref = exp.require(arm)
    at = M.Atlas(out, arm=arm, run_id=arm.lower().replace("-", "_"), register=meta["register"],
                 wave=meta["wave"], expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    res = {"M-V": lambda: run_mv(at, gates),
           "M-D4": lambda: run_md4(at, gates),
           "M-D3": lambda: run_md3(at, gates)}[arm]()

    info = at.to_parquet()
    M.write_json(out, f"{arm}_result.json",
                 {"arm": arm, "queue_item": meta["queue_item"], "register": meta["register"],
                  "wave": meta["wave"], "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": info, "result": res})
    print(f"{arm}_OK " + json.dumps({"rows": info["n_rows"],
                                     "outcome": res.get("outcome")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
