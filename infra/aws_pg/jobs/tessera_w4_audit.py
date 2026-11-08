"""TESSERA §2.2 follow-through — read the two banked W4 numbers TESSERA named in advance.

The TESSERA result doc committed at 2026-08-02T01:11 (`8ae5c32`) closed Result 2 with:

    **Open and untested:** whether PG2's W4 "no new features" verdict — which used this trainer at
    this same `CC_LR = 1e-4` — shares the degenerate-dictionary signature. Two banked numbers settle
    it (`rel_norm_hist`, `n_live`, Modal `pg-results` volume). TESSERA asserts nothing about W4
    until they are read.

This job reads exactly those two numbers. Both the QUANTITIES and the DIAGNOSTIC were fixed before
the read: the quantities by the sentence above, the diagnostic by P1 (already frozen) and by
`delta_crosscoder`'s own classification thresholds. Nothing here is invented after the fact.

THE RULE, assembled only from already-frozen material:

  Leg 1 — P1's degenerate signature. P1 established that at `steps=4000, lr=1e-4` the crosscoder
          cannot recover a PLANTED 0.1111 at any width, and that the signature is a dictionary that
          stays nearly all-live with decoder norms piled at r ≈ 0.5. Operationalised with
          `delta_crosscoder`'s OWN cuts (`RELNORM_EMERGENT_MIN = 0.90`, `RELNORM_LOST_MAX = 0.10`):
          if no live latent sits beyond either cut, then `emergent_fraction = 0` is forced by the
          dictionary's shape, not measured from the data.
  Leg 2 — the null cannot fire. A shuffle-pairing null destroys token correspondence and therefore
          MUST manufacture one-sided latents. If it returns ~0, the instrument has no demonstrated
          ability to produce a positive reading at all, and a real reading of 0 carries no
          information. The reference is the same statistic from a P1-certified instrument.

  verdict INSTRUMENT-VOID  iff Leg 1 AND Leg 2 fire
          INSTRUMENT-OK    otherwise — the verdict under audit stands

The rule must be able to return INSTRUMENT-OK, so TESSERA's own A14 cells are run through the
IDENTICAL rule as a positive control. If they do not come back INSTRUMENT-OK the rule is broken and
this job says so instead of reporting on W4.

  python -m infra.aws_pg.jobs.tessera_w4_audit --out d --w4 w4.json[,w4b.json,...] --x <merged_x.json>

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import subprocess

from infra.pg.delta_crosscoder import RELNORM_EMERGENT_MIN, RELNORM_LOST_MAX

# The certified-instrument reference for Leg 2: TESSERA's own shuffle-emergent values sit in this
# range across every pair measured at the A14 setting. A null two orders of magnitude below the
# bottom of it has not shown it can fire.
NULL_FIRES_MIN = 0.01


def _read(p):
    if p is None:
        return None
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def tail_mass(hist):
    """Fraction of LIVE latents beyond each classification cut, from a banked rel-norm histogram."""
    c, e = hist["counts"], hist["edges"]
    tot = sum(c)
    assert tot > 0, "empty histogram — nothing to audit"
    i_hi = bisect.bisect_left(e, RELNORM_EMERGENT_MIN)
    i_lo = bisect.bisect_left(e, RELNORM_LOST_MAX)
    return {"n_live_in_hist": tot,
            "frac_above_emergent_cut": sum(c[i_hi:]) / tot,
            "frac_below_lost_cut": sum(c[:i_lo]) / tot,
            "emergent_cut": RELNORM_EMERGENT_MIN, "lost_cut": RELNORM_LOST_MAX}


def diagnose(live_frac, tails, shuffle_vals):
    leg1 = bool(tails["frac_above_emergent_cut"] == 0.0 and tails["frac_below_lost_cut"] < 1e-3)
    sh_max = max(shuffle_vals) if shuffle_vals else None
    leg2 = bool(sh_max is not None and sh_max < NULL_FIRES_MIN)
    return {"live_fraction": live_frac, **tails,
            "shuffle_null_max": sh_max, "null_fires_min": NULL_FIRES_MIN,
            "leg1_classification_tails_empty": leg1,
            "leg2_shuffle_null_cannot_fire": leg2,
            "verdict": "INSTRUMENT-VOID" if (leg1 and leg2) else "INSTRUMENT-OK"}


def audit_w4(d, src):
    rows = []
    for L, l in d["layers"].items():
        sd = l["seed_details"][0]
        cfg = sd["cc_cfg"]
        h = l.get("rel_norm_hist_seed0")
        assert h, f"{src} L{L}: no banked rel_norm_hist — the audit needs it"
        rows.append({"source": src, "layer": int(L), "steps": cfg["steps"], "lr": cfg["lr"],
                     "dict_size": cfg["dict_size"], "n_live": sd["n_live"],
                     "reported_emergent": sd["emergent_fraction"],
                     "reported_lost": sd["lost_fraction"],
                     "banked_verdict": l["verdict"]["verdict"],
                     **diagnose(sd["n_live"] / cfg["dict_size"], tail_mass(h),
                                l.get("shuffle_null_emergent_fractions") or [])})
    return rows


def audit_tessera(x):
    """The positive control: the SAME rule on TESSERA's own A14 cells. Must read INSTRUMENT-OK."""
    rows = []
    for pn, pr in x["pairs"].items():
        for L, s in pr.get("sites", {}).items():
            if "seeds" not in s:
                continue
            q = s["seeds"][0]
            cfg = q["cfg"]
            rows.append({"source": f"TESSERA/{pn}", "layer": int(L), "steps": cfg["steps"],
                         "lr": cfg["lr"], "dict_size": cfg["dict_size"], "n_live": q["n_live"],
                         "reported_emergent": q["emergent_fraction"],
                         "reported_lost": q["lost_fraction"], "banked_verdict": None,
                         **diagnose(q["n_live"] / cfg["dict_size"], tail_mass(q["rel_norm_hist"]),
                                    [z["emergent_fraction"] for z in s.get("shuffle_null", [])])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w4", required=True, help="comma list of banked W4 delta_crosscoder JSONs")
    ap.add_argument("--x", required=True, help="merged TESSERA Arm X artifact (positive control)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ctrl = audit_tessera(_read(a.x))
    ctrl_ok = all(r["verdict"] == "INSTRUMENT-OK" for r in ctrl)
    rep = {"object": "TESSERA §2.2 — the two banked W4 numbers, read",
           "prereg": "quantities named in the TESSERA result doc @ 8ae5c32 (2026-08-02T01:11-05:00) "
                     "BEFORE the read; diagnostic assembled from P1 (frozen) and "
                     "delta_crosscoder's own classification cuts",
           "positive_control": {"n_cells": len(ctrl), "all_INSTRUMENT_OK": ctrl_ok, "rows": ctrl}}
    for r in ctrl:
        print("[ctrl] %-22s L%-3d live=%.3f tail>%.2f=%.5f shuf_max=%s -> %s"
              % (r["source"], r["layer"], r["live_fraction"], r["emergent_cut"],
                 r["frac_above_emergent_cut"],
                 ("%.4f" % r["shuffle_null_max"]) if r["shuffle_null_max"] is not None else "None",
                 r["verdict"]))
    if not ctrl_ok:
        rep["verdict"] = "RULE-BROKEN — the positive control did not read INSTRUMENT-OK, so this " \
                         "rule cannot be applied to W4"
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
        print("\nRULE-BROKEN: refusing to report on W4.", flush=True)
        return

    rows = []
    for uri in a.w4.split(","):
        rows += audit_w4(_read(uri.strip()), pathlib.Path(uri.strip()).name)
    print()
    for r in rows:
        print("[W4]   %-28s L%-3d steps=%d lr=%g live=%.3f tail>%.2f=%.5f tail<%.2f=%.5f "
              "shuf_max=%.2e reported_em=%.4f banked=%s -> %s"
              % (r["source"], r["layer"], r["steps"], r["lr"], r["live_fraction"],
                 r["emergent_cut"], r["frac_above_emergent_cut"], r["lost_cut"],
                 r["frac_below_lost_cut"], r["shuffle_null_max"], r["reported_emergent"],
                 r["banked_verdict"], r["verdict"]))
    n_void = sum(1 for r in rows if r["verdict"] == "INSTRUMENT-VOID")
    rep["w4_rows"] = rows
    rep["summary"] = {"n_cells": len(rows), "n_instrument_void": n_void,
                      "verdict": ("INSTRUMENT-VOID" if n_void == len(rows) else
                                  "MIXED" if n_void else "INSTRUMENT-OK"),
                      "scope": "This bounds what W4's MEASUREMENT could detect. It is NOT a claim "
                               "that PG2 gained features — an instrument that cannot see a planted "
                               "signal is equally silent whether or not one is there. The correct "
                               "re-scope of the banked verdict is UNMEASURED, not REVERSED."}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"\nTESSERA_W4_AUDIT_OK {n_void}/{len(rows)} cells INSTRUMENT-VOID", flush=True)


if __name__ == "__main__":
    main()
