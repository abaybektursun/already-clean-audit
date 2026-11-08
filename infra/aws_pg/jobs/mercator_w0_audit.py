"""MERCATOR Wave 0 — the two defect arms.

  M-D1 (queue D1)  Audit EVERY DARWIN summary/gates JSON for verdict-field staleness, then
                   regenerate the one the queue names. The known instance is W2-F's
                   `composition_passes: true` sitting beside an interaction of 2.5033 and a CI
                   half-width of 0.3648, i.e. beside inputs from which the same code computes
                   False. The arm's deliverable is the DIFF, and it is only worth anything if the
                   audit is exhaustive rather than targeted — so every verdict-like field in every
                   summary is enumerated, and each one is marked recomputable or not.

  M-A7 (queue A7)  Fix the REFEREE dose-anchor lookup, re-derive the true dose, and log the
                   defect. `isinstance(True, int)` is True in Python, so DARWIN's hint matcher
                   accepted a boolean and reported dose = 0.0 under the label "REFEREE banked
                   aggregate" — a fabricated provenance claim, which is worse than the wrong
                   number. The corrected anchor is REFEREE's own lambda ladder.

Both arms read artifacts only. CPU. FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

DARWIN_PREFIX = f"s3://{M.BUCKET}/darwin/"
REFEREE_S3 = f"s3://{M.BUCKET}/results/referee_aggregate4/out/metrics.json"
ARMS = ("M-D1", "M-A7", "M-D5")
CLAIMS_FILE = pathlib.Path(__file__).resolve().parents[2] / "pg" / "mercator_d5_claims.json"
CLAIM_TAGS = ("predicted-by", "consistent-with", "scope-limit", "absent")

# Field names that assert a VERDICT rather than record a measurement. A field matching one of
# these is a claim, and a claim beside its own inputs is checkable.
VERDICT_TOKENS = ("passes", "passed", "_ok", "ok_", "fires", "graduated", "monotone", "valid",
                  "confirmed", "holds", "significant", "refuted", "supported")
# The numeric roles a recomputable comparison needs, in the names the house uses for them.
POINT_KEYS = ("interaction", "point", "estimate", "contrast", "d", "delta", "value", "mean")
CI_LO_KEYS = ("ci_lo", "lo95", "lo", "ci_low")
CI_HI_KEYS = ("ci_hi", "hi95", "hi", "ci_high")
HW_KEYS = ("ci_halfwidth", "halfwidth", "hw")


def _s3_ls(prefix: str):
    r = subprocess.run(["aws", "s3", "ls", prefix, "--recursive", "--region", "us-east-1"],
                       check=True, capture_output=True, text=True)
    return [ln.split(None, 3)[3] for ln in r.stdout.splitlines() if ln.strip()]


def _s3_read_json(key: str):
    r = subprocess.run(["aws", "s3", "cp", f"s3://{M.BUCKET}/{key}", "-", "--region", "us-east-1"],
                       check=True, capture_output=True, text=True)
    return json.loads(r.stdout)


def _dicts(obj, path=""):
    """Every dict in a JSON tree, with its dotted path."""
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from _dicts(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _dicts(v, f"{path}[{i}]")


def _first(d: dict, names):
    for n in names:
        for k, v in d.items():
            if k.lower() == n and isinstance(v, (int, float)) and not isinstance(v, bool):
                return k, float(v)
    for n in names:
        for k, v in d.items():
            if n in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                return k, float(v)
    return None, None


def recompute_claim(d: dict, field: str):
    """Recompute a verdict-like boolean from the numbers sitting in the SAME dict.

    The house's composition/interaction rule is `estimate <= ci_halfwidth` (a term fails when it
    sits below -1 x its own half-width, which for the "not worse than additive" direction is the
    same comparison). If a point and a half-width (or a CI to derive one from) are both present,
    the claim is checkable; otherwise it is recorded as not-recomputable rather than guessed.
    """
    pk, point = _first(d, POINT_KEYS)
    if point is None:
        return None
    hk, hw = _first(d, HW_KEYS)
    if hw is None:
        lk, lo = _first(d, CI_LO_KEYS)
        hik, hi = _first(d, CI_HI_KEYS)
        if lo is None or hi is None or hi < lo:
            return None
        hw, hk = (hi - lo) / 2.0, f"({hik}-{lk})/2"
    return {"field": field, "declared": bool(d[field]), "point_key": pk, "point": point,
            "halfwidth_key": hk, "halfwidth": hw,
            "recomputed": bool(point <= hw),
            "agrees": bool(d[field]) == bool(point <= hw)}


def run_md1(at, gates) -> dict:
    keys = [k for k in _s3_ls(DARWIN_PREFIX)
            if k.endswith(".json") and ("summary" in k.rsplit("/", 1)[-1] or
                                        k.rsplit("/", 1)[-1] == "gates.json")]
    assert keys, f"no summary/gates JSON under {DARWIN_PREFIX}"
    print(f"[M-D1] auditing {len(keys)} summary/gates files", flush=True)

    n_fields, checkable, disagreements, per_file = 0, 0, [], {}
    for key in sorted(keys):
        doc = _s3_read_json(key)
        found = []
        for path, d in _dicts(doc):
            for f, v in d.items():
                if not isinstance(v, bool):
                    continue
                if not any(t in f.lower() for t in VERDICT_TOKENS):
                    continue
                n_fields += 1
                rec = recompute_claim(d, f)
                entry = {"path": f"{path}.{f}" if path else f, "declared": v,
                         "recomputable": rec is not None}
                if rec is not None:
                    checkable += 1
                    entry.update({k2: rec[k2] for k2 in
                                  ("point_key", "point", "halfwidth", "recomputed", "agrees")})
                    if not rec["agrees"]:
                        disagreements.append({"file": key, **entry})
                found.append(entry)
        if found:
            per_file[key] = found
    for i, dis in enumerate(disagreements):
        at.add("stale_verdict_field", value=1.0, item_id=f"{dis['file']}#{dis['path']}",
               flags={k: dis[k] for k in ("declared", "recomputed", "point", "halfwidth",
                                          "point_key", "file", "path")})
    at.add("summaries_audited", value=float(len(keys)))
    at.add("verdict_fields_found", value=float(n_fields))
    at.add("verdict_fields_recomputable", value=float(checkable))
    at.add("verdict_fields_disagreeing", value=float(len(disagreements)))
    gates.record("md1_audit_is_exhaustive", True, n_files=len(keys), n_fields=n_fields,
                 n_recomputable=checkable, n_disagreeing=len(disagreements))
    # The queue names one instance. Whether the audit re-finds it is the arm's own positive
    # control: an audit that cannot re-find a known defect has not audited anything.
    found_w2f = [d for d in disagreements if "w2_f" in d["file"].lower()]
    gates.require("md1_refinds_the_known_w2f_defect", bool(found_w2f),
                  known_instance="W2-F composition_passes", found=len(found_w2f),
                  note=("the queue names this defect; an audit that misses it is not evidence "
                        "about the ones it did not name"))
    return {"arm": "M-D1", "n_files": len(keys), "n_verdict_fields": n_fields,
            "n_recomputable": checkable, "n_disagreeing": len(disagreements),
            "disagreements": disagreements,
            "per_file": {k: v for k, v in per_file.items() if v},
            "not_recomputable_note": ("a verdict field with no point+CI in its own dict cannot be "
                                      "checked from the artifact; those are counted, not judged")}


def run_ma7(at, gates) -> dict:
    d = _s3_read_json(REFEREE_S3.split(f"{M.BUCKET}/", 1)[1])
    dr = d.get("dose_response")
    assert dr, f"{REFEREE_S3} has no dose_response block"
    lambdas = sorted(int("".join(ch for ch in k if ch.isdigit()))
                     for k in d["contrasts"] if k.startswith("incore")
                     and any(ch.isdigit() for ch in k))
    assert lambdas, "no incore<lambda> contrasts"
    curve = {}
    for lam in lambdas:
        c = d["contrasts"][f"incore{lam}"]
        curve[lam] = {"task_d": [float(x["d"]) for x in c["task"]],
                      "battery_d": [float(x["d"]) for x in c["battery"]],
                      "n_seeds": len(c["task"])}
        for readout in ("task", "battery"):
            vals = [float(x["d"]) for x in c[readout]]
            at.add_ci(f"referee_dose_{readout}_delta", vals, dose=float(lam),
                      flags={"lambda": lam, "source": REFEREE_S3, "n_seeds": len(vals)})
    # §3's ">= 4 log-spaced doses" governs what MERCATOR RUNS, not what it READS. REFEREE's own
    # ladder is 3 points (lambda 1/4/16, ratio 4 — log-spaced but short of MERCATOR's minimum), so
    # the shortfall is RECORDED and travels with the corrected anchor rather than asserted away.
    # Asserting here would make an inherited artifact fail a rule it was never bound by; ignoring
    # it would let a 3-point curve be cited as if it met the standard.
    ratios = np.diff(np.log(np.array(lambdas, dtype=np.float64)))
    inherited_grid = {"lambdas": lambdas, "n_doses": len(lambdas),
                      "mercator_min_doses": M.MIN_DOSES,
                      "meets_mercator_minimum": len(lambdas) >= M.MIN_DOSES,
                      "log_spaced": bool(len(ratios) > 0
                                         and float(np.std(ratios) / np.mean(ratios)) < 0.01),
                      "note": ("inherited from REFEREE; §3's dose-grid minimum binds MERCATOR's "
                               "own sweeps, and any citation of this curve carries this row")}
    gates.record("ma7_inherited_dose_grid_meets_mercator_minimum",
                 inherited_grid["meets_mercator_minimum"], **inherited_grid)
    at.add("referee_dose_grid_n_doses", value=float(len(lambdas)), flags=inherited_grid)

    # The corrected anchor: the largest lambda whose task delta is positive on the seed MEAN.
    means = {lam: float(np.mean(v["task_d"])) for lam, v in curve.items()}
    helpful = [lam for lam, m in means.items() if m > 0]
    at.add("referee_helpful_dose_max", value=float(max(helpful)) if helpful else 0.0,
           flags={"task_delta_means": means, "helpful_lambdas": helpful,
                  "definition": "largest lambda whose seed-mean task delta is > 0"})

    # The defect, re-derived from the artifact rather than asserted.
    leaves = {}

    def _walk(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                _walk(v, f"{p}.{k}" if p else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                _walk(v, f"{p}[{i}]")
        else:
            leaves[p] = o

    _walk(d)
    hints = ("dose", "helpful", "gain", "lam")
    matched = {k: v for k, v in leaves.items()
               if any(h in k.lower() for h in hints) and isinstance(v, (int, float))}
    bools = {k: v for k, v in matched.items() if isinstance(v, bool)}
    nums = {k: v for k, v in matched.items() if not isinstance(v, bool)}
    gates.require("ma7_defect_reproduced", bool(bools),
                  n_bool_keys_matched=len(bools), n_numeric_keys_matched=len(nums),
                  note=("isinstance(True, int) is True, so a hint matcher gated on "
                        "isinstance(v, (int, float)) accepts booleans; a bool can never be a dose"))
    at.add("referee_dose_lookup_bool_matches", value=float(len(bools)),
           flags={"bool_keys": sorted(bools), "numeric_keys": sorted(nums),
                  "darwin_reported_dose": 0.0,
                  "darwin_reported_label": "REFEREE banked aggregate"})
    return {"arm": "M-A7", "source": REFEREE_S3, "lambdas": lambdas, "dose_curve": curve,
            "inherited_grid": inherited_grid,
            "task_delta_means": means, "helpful_lambdas": helpful,
            "corrected_anchor": {"helpful_dose_max": max(helpful) if helpful else None,
                                 "definition": "largest lambda with seed-mean task delta > 0"},
            "defect": {"bool_keys_matched": bools, "numeric_keys_matched": sorted(nums),
                       "darwin_reported_dose": 0.0,
                       "darwin_reported_label": "REFEREE banked aggregate",
                       "root_cause": "isinstance(True, int) is True in Python"}}


def run_md5(at, gates) -> dict:
    """M-D5 — the cite-verify gate.

    The claim table is authored by READING the paper (that is the whole arm), and lives in the
    shipped tree so this job can verify its SHAPE and emit its rows: every claim carries a quote,
    a location, and a tag from the frozen vocabulary, and a claim tagged `predicted-by` must name
    what it predicts. A table row with no location is not a cite-verified claim, and the gate here
    exists so that cannot reach a result document.
    """
    assert CLAIMS_FILE.exists(), f"no claim table at {CLAIMS_FILE}"
    doc = json.loads(CLAIMS_FILE.read_text())
    claims = doc["claims"]
    assert claims, "empty claim table"
    for c in claims:
        for f in ("id", "what_the_paper_says", "location", "quote", "bearing_on_mercator",
                  "tag", "consequence"):
            assert c.get(f) and str(c[f]).strip(), f"{c.get('id')}: field {f} is empty"
        assert c["tag"] in CLAIM_TAGS, f"{c['id']}: unknown tag {c['tag']!r}"
        if c["tag"] == "absent":
            assert c["quote"] == "ABSENT", f"{c['id']}: an absent claim must quote ABSENT"
        else:
            assert c["quote"] != "ABSENT", f"{c['id']}: tagged {c['tag']} but quotes ABSENT"
        at.add("cite_verified_claim", value=1.0, item_id=c["id"],
               corpus_id=f"arxiv:{doc['source']['arxiv_id']}{doc['source']['version']}",
               flags={"tag": c["tag"], "location": c["location"],
                      "title": doc["source"]["title"]})
    counts = {t: sum(1 for c in claims if c["tag"] == t) for t in CLAIM_TAGS}
    for t, n in counts.items():
        at.add(f"claims_tagged_{t.replace('-', '_')}", value=float(n),
               corpus_id=f"arxiv:{doc['source']['arxiv_id']}{doc['source']['version']}")
    gates.require("md5_every_claim_has_a_location_and_a_tag", True, n_claims=len(claims),
                  counts=counts)
    gates.record("md5_any_finding_predicted_by_this_source", counts["predicted-by"] > 0,
                 n_predicted_by=counts["predicted-by"], n_claims=len(claims), counts=counts,
                 note=("a `predicted-by` tag is the strong claim; zero of them means the paper may "
                       "be cited only as consistent-with, and the scope limits say why"))
    return {"arm": "M-D5", "source": doc["source"], "gate": doc["gate"],
            "counts": counts, "verdict": doc["verdict"],
            "claims": [{k: c[k] for k in ("id", "tag", "location")} for c in claims]}


def main():
    ap = argparse.ArgumentParser(description="MERCATOR Wave 0 — the defect and cite-verify arms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = a.arm
    meta = P.ARMS[arm]

    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w0") / "expectations.jsonl")
    expected_ref = exp.require(arm)
    at = M.Atlas(out, arm=arm, run_id=arm.lower().replace("-", "_"), register=meta["register"],
                 wave=meta["wave"], expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    res = {"M-D1": lambda: run_md1(at, gates),
           "M-A7": lambda: run_ma7(at, gates),
           "M-D5": lambda: run_md5(at, gates)}[arm]()
    info = at.to_parquet()
    M.write_json(out, f"{arm}_result.json",
                 {"arm": arm, "queue_item": meta["queue_item"], "register": meta["register"],
                  "wave": meta["wave"], "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": info, "result": res})
    print(f"{arm}_OK " + json.dumps({"rows": info["n_rows"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
