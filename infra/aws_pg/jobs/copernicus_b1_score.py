"""CO-B1 scoring — apply §6.3's three instruments to CO-B1's banked quantities. CPU, no weights.

WHY A SCORER RATHER THAN A RERUN. `results/copernicus_b1` was SIGKILLed by its own in-instance
budget (`run_job.sh` wraps the job in `timeout --signal=KILL ${BUDGET}s`) one draw from the end of
the descriptive subsample block, so it never reached its verdict line. Everything the frozen gate
reads was already banked by the incremental saves: the 24 LOO folds, the disjoint half-split, the
Haar band, and the positive control. §6.5.8 exists precisely so a rule can be re-applied from the
artifact — re-running four CPU-hours of eigendecompositions to recompute numbers that are already
on disk would be waste, not rigour.

TWO THINGS THIS SCORER DOES THAT THE JOB'S OWN VERDICT LINE DID NOT.

(1) IT GATES INSTRUMENT (i) ON ITS POSITIVE CONTROL. §6.3 says the planted carrier "must be caught
    by (i)". `copernicus_b1.py` records that control as its own gate row and then composes the
    verdict as `REFUTER_FIRED if carriers else (REFUTER_DID_NOT_FIRE if structural else AMBIGUOUS)`
    — never consulting it. Had the job survived its budget it would have banked a clean
    REFUTER-DID-NOT-FIRE while its own positive control read False. That is the CO-M4 defect class:
    an implementation that does not match the frozen letter.

(2) IT SCORES THE THREE INSTRUMENTS SEPARATELY. §6.3 calls CO-B1 "three instruments, LOO alone is
    underpowered at 1/24 mass" — the spec anticipated (i) being weak, which is why (iii) exists. A
    single composite token hides which instrument carried the answer.

    python3 -m infra.aws_pg.jobs.copernicus_b1_score --out out
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import copernicus_core as C

ARM = "CO-B1"
QUEUE_ITEM = "22+75"
SOURCE = "s3://pg-lineage-769063704696/results/copernicus_b1/out/copernicus_b1.json"
JACKKNIFE_SD = 4.0                      # §6.3 (i) — frozen, and identical to the job's constant


def s3_json(uri):
    r = subprocess.run(["aws", "s3", "cp", uri, "-", "--region", "us-east-1"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout, f"unreadable: {uri}"
    return json.loads(r.stdout)


def instrument_i(d):
    """(i) jackknife influence. CONTAMINATION fires iff any fold sits > 4 jackknife SDs below the
    24-fold mean. VALID only if the planted carrier was caught."""
    loo, pc = d["loo"], d["positive_control"]
    vals = [f["mean_cos2"] for f in loo["folds"]]
    floor = loo["mean"] - JACKKNIFE_SD * loo["sd"]
    carriers = [f["held_out"] for f in loo["folds"] if f["mean_cos2"] < floor]
    required = JACKKNIFE_SD * loo["sd"]
    observed = abs(pc["with_plant"]["mean_cos2"] - pc["plant_held_out"]["mean_cos2"])
    return {
        "n_folds": len(vals), "mean": loo["mean"], "sd": loo["sd"], "floor": floor,
        "min_fold": min(vals), "max_fold": max(vals),
        "carriers_rederived": carriers, "carriers_recorded": loo["carriers"],
        "agrees_with_record": carriers == loo["carriers"],
        "positive_control": {
            "planted_repo": pc["planted_repo"], "caught": bool(pc["caught"]),
            "plant_moved_statistic_by": observed,
            "detection_needs": required,
            "shortfall_factor": (required / observed) if observed > 0 else None},
        "valid": bool(pc["caught"]),
        "verdict": ("VOID-INSTRUMENT" if not pc["caught"]
                    else ("CONTAMINATION" if carriers else "NO-CARRIER")),
    }


def instrument_iii(d):
    """(iii) disjoint half-split. STRUCTURAL requires both 12-repo consensuses above the Haar band.

    Independent of (i) by construction, and that independence is the point: a single-repo carrier
    can sit in only ONE half, so both halves clearing the band rules out single-repo contamination
    without relying on the jackknife at all."""
    hs, hb = d["half_split"], d["haar_band"]
    h1, h2 = hs["half1"]["mean_cos2"], hs["half2"]["mean_cos2"]
    lo = min(h1, h2)
    structural = bool(lo > hb["hi"])
    return {"half1": h1, "half2": h2, "min": lo,
            "haar_band": [hb["lo"], hb["hi"]], "n_haar_draws": hb["n_draws"],
            "margin_over_haar_hi": lo / hb["hi"],
            "verdict": "STRUCTURAL" if structural else "NOT-SEPARATED"}


def compose(i, iii, sub_complete):
    """§6.3's composite, refusing to rest on an instrument its own positive control invalidated."""
    if i["verdict"] == "CONTAMINATION":
        return C.REFUTER_FIRED, "a named fold drops below the jackknife floor"
    if i["valid"] and iii["verdict"] == "STRUCTURAL":
        return C.REFUTER_DID_NOT_FIRE, "both instruments agree and (i) is validated"
    if not i["valid"] and iii["verdict"] == "STRUCTURAL":
        return ("REFUTER-DID-NOT-FIRE-ON-(iii)-ALONE",
                "(i) is VOID by its own positive control; (iii) carries the answer unaided, and "
                "it rules out single-repo contamination on its own because a lone carrier can sit "
                "in only one half")
    return C.AMBIGUOUS, "(iii) did not separate from the Haar band"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=SOURCE)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = s3_json(a.source)
    i, iii = instrument_i(d), instrument_iii(d)
    sub_complete = "subsample_curve" in d
    verdict, why = compose(i, iii, sub_complete)

    rep = {"object": "copernicus_b1_score_v1", "arm": ARM, "program": "copernicus",
           "queue_item": QUEUE_ITEM, "source": a.source,
           "source_utc": d.get("utc"), "source_provenance": d.get("provenance"),
           "layer": d.get("layer"), "types": d.get("types"), "k": d.get("k"),
           "n_ft": d.get("n_ft"), "manifest_sha256": d.get("manifest_sha256"),
           "full24": d.get("full24"),
           "instrument_i_jackknife": i,
           "instrument_ii_subsample": {
               "complete": sub_complete,
               "note": ("descriptive row (§6.3 (ii)); the source run was SIGKILLed by its "
                        "in-instance budget during this block, so the curve is NOT banked and no "
                        "partial curve is recovered from the log")},
           "instrument_iii_halfsplit": iii,
           "verdict": verdict, "verdict_basis": why,
           "job_verdict_line_would_have_said": (
               "REFUTER-DID-NOT-FIRE — copernicus_b1.py composes its verdict without consulting "
               "the positive control, so it would have banked a clean token while its own control "
               "read False"),
           "provenance": C.provenance(), "utc": C.utc_now()}
    (out / "copernicus_b1_score.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps({"verdict": verdict, "i": i["verdict"], "iii": iii["verdict"],
                      "pc_caught": i["positive_control"]["caught"],
                      "pc_shortfall_x": i["positive_control"]["shortfall_factor"]},
                     indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
