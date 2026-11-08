"""TESSERA-FIX C1 re-verdict — re-read every Arm X cell against a band that CAN fail. Spec §6.

WRITTEN BEFORE C1'S DATA EXISTS. The rule below is spec §6 verbatim; implementing it after seeing
the sibling band would let the band's shape influence the rule, which is the whole thing a frozen
rule prevents.

WHY. Arm X verdicted every cell against the identity pair (PG1, PG1). C0 showed that band is [0,0]
not because the two decoder halves are identical — they are not — but because identity's entire live
population sits within ±0.02 of r = 0.5 and would have to reach 0.40 to touch either classification
cut. **It is 20× short of being able to report a nonzero value.** "Clears the identity band" therefore
means "is nonzero", not "exceeds an achievable null".

C1 measures the seventh pair — `(ft-aokvqa-da, ft-refcoco-seg)`, two PG1 siblings with NO descent
relation — at the same setting, 2 seeds × 2 sites. Two real, differently-trained models whose
difference is not descent. That gives a nonzero-width reference band every Arm-X reading must clear.

THE FROZEN RULE (spec §6, unaltered):

    band(statistic) = [min, max] over {2 seeds × 2 sites}
    a TESSERA reading is REAL   iff it exceeds the band AT ITS OWN SITE (paired)
                    otherwise   WITHIN-REFERENCE-NOISE
    applied mechanically to all 12 Arm-X cells

Paired-at-its-own-site matters: the shuffle nulls already showed these statistics are strongly
site-dependent (identity's shuffle-emergent is 0.187 at L9 and 0.267 at L14), so pooling sites would
compare a reading against a band built partly at the wrong depth.

PRE-COMMITTED CONSEQUENCES (spec §1.2), so no result can be re-read after the fact:
  H_pi0-loss-real       π0's LOST inside the band at EITHER site ⇒ Result 6 downgraded to
                        within-reference-noise.
  H_install-emergent-real  install-emergent @L14 inside the band ⇒ Result 5 fully retracted to
                        "no measurement above reference noise".
And the instrument gate: C1's own shuffle range must fire (≥ the TESSERA floor 0.01) or the band is
VOID and C0's reasoning stands alone.

  python -m infra.aws_pg.jobs.tessera_c1_verdict --out v --x <merged_x.json> --c1 <c1_sibling.json>

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

SHUFFLE_FIRES_MIN = 0.01          # the TESSERA floor, spec §5
STATS = ("emergent_fraction", "lost_fraction")


def _read(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def band_from(c1, stat):
    """{site: [min, max]} over that site's seeds, plus the pooled band, from the sibling pair."""
    sib = c1["pairs"]["sibling"]["sites"]
    per_site, pooled = {}, []
    for L, s in sib.items():
        vals = [q[stat] for q in s.get("seeds", [])]
        assert vals, f"sibling L{L}: no seeds — the reference band has no data"
        per_site[str(L)] = [min(vals), max(vals)]
        pooled += vals
    return {"per_site": per_site, "pooled": [min(pooled), max(pooled)],
            "n": len(pooled)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--x", required=True, help="merged Arm X artifact")
    ap.add_argument("--c1", required=True, help="tessera_c1_sibling artifact")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    x, c1 = _read(a.x), _read(a.c1)
    sib = c1["pairs"]["sibling"]["sites"]

    # ---- instrument gate: the reference pair's own shuffle must fire -------------------------
    shuf = [q["emergent_fraction"] for s in sib.values() for q in s.get("shuffle_null", [])]
    shuf_max = max(shuf) if shuf else None
    fires = bool(shuf_max is not None and shuf_max >= SHUFFLE_FIRES_MIN)
    rep = {"object": "TESSERA-FIX C1 — Arm X re-verdicted against the sibling reference band",
           "rule": "band = [min, max] over {2 seeds × 2 sites}; REAL iff the reading exceeds the "
                   "band AT ITS OWN SITE (paired); else WITHIN-REFERENCE-NOISE (spec §6)",
           "reference_pair": c1["pairs"]["sibling"]["units"],
           "instrument_gate": {"shuffle_max": shuf_max, "floor": SHUFFLE_FIRES_MIN,
                               "fires": fires,
                               "why": "a reference pair whose own shuffle cannot fire is a "
                                      "degenerate pair, and its band would repeat the identity "
                                      "band's defect one level up"}}
    if not fires:
        rep["verdict"] = ("BAND VOID — the reference pair's shuffle did not fire; C0's reasoning "
                          "stands alone and no Arm-X cell is re-verdicted")
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
        print("BAND VOID: sibling shuffle_max "
              f"{shuf_max} < {SHUFFLE_FIRES_MIN}", flush=True)
        return

    bands = {st: band_from(c1, st) for st in STATS}
    rep["bands"] = bands

    rows = []
    for pn, pr in x["pairs"].items():
        if pn == "identity":
            continue
        for L, s in sorted(pr.get("sites", {}).items(), key=lambda kv: int(kv[0])):
            if "seeds" not in s:
                continue
            for st in STATS:
                m = sum(q[st] for q in s["seeds"]) / len(s["seeds"])
                # The frozen rule is "exceeds the band AT ITS OWN SITE (paired)". v1 fell back to
                # the POOLED band when a site was missing, silently scoring a cell against a band
                # built at a different depth -- and these statistics are strongly site-dependent
                # (identity's shuffle-emergent is 0.187 @L9 vs 0.267 @L14). A missing site VOIDs
                # the cell; it does not borrow another site's band. Caught by dry-running this job
                # against a PARTIAL C1 artifact before the real one existed.
                site_band = bands[st]["per_site"].get(str(L))
                if site_band is None:
                    rows.append({"pair": pn, "site": int(L), "statistic": st, "measured": m,
                                 "band_at_site": None, "old_verdict": None,
                                 "verdict": "VOID-NO-BAND-AT-SITE",
                                 "why": f"the reference pair has no measured band at L{L}; the "
                                        f"frozen rule is paired-at-site"})
                    continue
                lo, hi = site_band
                real = bool(m > hi)
                rows.append({"pair": pn, "site": int(L), "statistic": st, "measured": m,
                             "band_at_site": [lo, hi],
                             "old_verdict": (s.get("emergent_verdict") if st == "emergent_fraction"
                                             else s.get("lost_verdict") or {}).get("verdict")
                             if isinstance(s.get("emergent_verdict"), dict) else None,
                             "verdict": "REAL" if real else "WITHIN-REFERENCE-NOISE"})

    def _v(pair, st, site):
        for r in rows:
            if r["pair"] == pair and r["statistic"] == st and r["site"] == site:
                return r
        return None

    pi0 = [_v("pi0", "lost_fraction", L) for L in (9, 14)]
    ins = _v("install", "emergent_fraction", 14)
    rep["per_cell"] = rows
    rep["hypotheses"] = {
        "H_pi0-loss-real": {
            "rule": "π0's LOST must exceed the band at BOTH sites; inside at either ⇒ Result 6 "
                    "downgraded to within-reference-noise",
            "per_site": {f"L{r['site']}": r["verdict"] for r in pi0 if r},
            "verdict": ("SUPPORTED" if all(r and r["verdict"] == "REAL" for r in pi0)
                        else ("VOID — a site has no reference band"
                              if any(r and r["verdict"] == "VOID-NO-BAND-AT-SITE" for r in pi0)
                              else "NOT SUPPORTED — Result 6 downgrades"))},
        "H_install-emergent-real": {
            "rule": "install-emergent @L14 must exceed the band; inside ⇒ Result 5 fully retracted",
            "verdict": ("SUPPORTED" if ins and ins["verdict"] == "REAL"
                        else ("VOID — no reference band at L14"
                              if ins and ins["verdict"] == "VOID-NO-BAND-AT-SITE"
                              else "NOT SUPPORTED — Result 5 retracts to 'no measurement above "
                                   "reference noise'")),
            "measured": ins["measured"] if ins else None,
            "band": ins["band_at_site"] if ins else None}}
    n_real = sum(1 for r in rows if r["verdict"] == "REAL")
    n_void = sum(1 for r in rows if r["verdict"] == "VOID-NO-BAND-AT-SITE")
    rep["summary"] = {"n_cells": len(rows), "n_real": n_real, "n_void_no_band": n_void,
                      "n_within_noise": len(rows) - n_real - n_void,
                      "complete": bool(n_void == 0)}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    print("%-10s %-5s %-18s %10s %22s  %s"
          % ("pair", "site", "statistic", "measured", "band@site", "verdict"))
    for r in rows:
        if r["band_at_site"] is None:
            print("%-10s L%-4d %-18s %10.4f  %22s  %s"
                  % (r["pair"], r["site"], r["statistic"], r["measured"], "(no band at site)",
                     r["verdict"]))
        else:
            print("%-10s L%-4d %-18s %10.4f  [%8.4f, %8.4f]  %s"
                  % (r["pair"], r["site"], r["statistic"], r["measured"],
                     r["band_at_site"][0], r["band_at_site"][1], r["verdict"]))
    print(f"\nH_pi0-loss-real          : {rep['hypotheses']['H_pi0-loss-real']['verdict']}")
    print(f"H_install-emergent-real  : {rep['hypotheses']['H_install-emergent-real']['verdict']}")
    print(f"\nTESSERA_C1_VERDICT_OK {n_real}/{len(rows)} REAL", flush=True)


if __name__ == "__main__":
    main()
