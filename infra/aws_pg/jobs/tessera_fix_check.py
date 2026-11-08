"""TESSERA-FIX acceptance checks — one mechanical check per critic finding. Spec §3, §1.3.

Arm F's own falsifiable claim (spec §1.3) is that the fold can be made MECHANICAL. Every row of the
spec's §3 ledger carries an acceptance check; this script evaluates them. A finding whose check
cannot be made mechanical must be rebutted in writing instead, with its reason — and this script
reports that state rather than passing silently.

Checks run against the FILES, not against a summary of them. `--strict` exits non-zero on any
unfolded finding, which is what gates Arm G.

  python -m infra.aws_pg.jobs.tessera_fix_check --root . [--strict]

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

RES = "studies/lineage/paligemma1-pi0/research/2026-08-02_tessera_result.md"
FIXRES = "studies/lineage/paligemma1-pi0/research/2026-08-03_tessera_fix_result.md"
JRN = "studies/lineage/paligemma1-pi0/research/2026-07-31_tessera_journal.md"
CRIT = "studies/lineage/paligemma1-pi0/research/2026-08-03_tessera_critic_report.md"
XC = "infra/pg/tessera_crosscoder.py"
SC = "infra/aws_pg/jobs/tessera_s_core.py"
AGG = "infra/aws_pg/jobs/tessera_aggregate.py"
AGG_ART = "studies/lineage/paligemma1-pi0/research/artifacts/tessera_agg_metrics.json"
MAP = "studies/lineage/MAP.md"
IDX = "studies/lineage/RESEARCH_INDEX.md"
RDM = "studies/lineage/README.md"


def _t(root, rel):
    p = pathlib.Path(root) / rel
    return p.read_text() if p.exists() else ""


def grid_row(res_text, n):
    """The §7 grid row `| n | ... |` as one string, or '' if absent."""
    m = re.search(rf"^\|\s*{n}\s*\|.*$", res_text, re.M)
    return m.group(0) if m else ""


def section(res_text, heading):
    """The text of one `### Result N` section, so a check cannot be satisfied by a string that
    happens to appear elsewhere in the document.

    Scoping matters: an unscoped `"TESSERA" in index` passed at BASELINE because the W4 re-scope
    row mentions TESSERA, and an unscoped `"overlap" in res` passed because Result 4 is about
    overlaps. Three of the first three baseline passes were spurious for exactly this reason.
    """
    i = res_text.find(heading)
    if i < 0:
        return ""
    j = res_text.find("\n### ", i + len(heading))
    return res_text[i:] if j < 0 else res_text[i:j]


NUMWORD = {21: "Twenty-one", 22: "Twenty-two", 23: "Twenty-three", 24: "Twenty-four",
           25: "Twenty-five", 26: "Twenty-six", 27: "Twenty-seven", 28: "Twenty-eight",
           29: "Twenty-nine", 30: "Thirty"}


def _catalogue_ok(fixres):
    """Rows are contiguous 1..N AND the prose says N — both, computed, never typed."""
    i = fixres.find("## 3b. The label-drift catalogue")
    j = fixres.find("\n## 4. Prediction grid")
    if i < 0 or j < 0:
        return False
    sec = fixres[i:j]
    rows = re.findall(r"^\| (\d+) \| ", sec, re.M)
    n = len(rows)
    return (rows == [str(k) for k in range(1, n + 1)]
            and NUMWORD.get(n, "\u0000") in sec)


def build_checks(root):
    res, jrn, crit = _t(root, RES), _t(root, JRN), _t(root, CRIT)
    xc, sc, agg = _t(root, XC), _t(root, SC), _t(root, AGG)
    agg_art = _t(root, AGG_ART)
    fixres = _t(root, FIXRES)
    mp, idx, rdm = _t(root, MAP), _t(root, IDX), _t(root, RDM)
    r9 = grid_row(res, 9)
    r2 = grid_row(res, 2)
    r3 = section(res, "### Result 3")
    r9s = section(res, "### Result 9")
    r10 = section(res, "### Result 10")
    r11 = section(res, "### Result 11")
    # §8 IFF bar — a summary section, not a `### Result N` section, so `section()` cannot reach it.
    _i = res.find("## 8. IFF bar")
    iff = res[_i:res.find("\n## ", _i + 5)] if _i >= 0 else ""

    def erasure_ok():
        """No 'erasure term' claim without an adjacent N4/C2 citation (spec §3 row 1)."""
        for m in re.finditer(r"erasure term", res):
            w = res[max(0, m.start() - 400):m.end() + 400]
            if not re.search(r"\bN4\b|\bC2\b", w):
                return False
        return True

    return [
        (1, "Result 6 erasure wording gated on N4/C2; grid row 9 names N4",
         erasure_ok() and "N4" in r9),
        (2, "Result 5 headline = 'only pair clearing'; latent counts printed",
         "only pair" in res.lower() and re.search(r"n_live|latent count", res) is not None),
        # Substance, not a phrase. v1 matched the literal string "C1 reference band", which a
        # correct fold can satisfy in a dozen wordings and an incorrect one can satisfy by
        # mentioning it in passing. What must be true: Results 5 and 6 quote the band's ACTUAL
        # numbers, use the C1 verdict vocabulary, and no longer rest on ABOVE-IDENTITY-BAND.
        (3, "Results 5/6 are re-verdicted against the C1 band by its numbers, not by its name",
         (all(x in section(res, "### Result 5") for x in ("0.00014", "REAL"))
          and all(x in section(res, "### Result 6") for x in ("0.00028", "REAL"))
          and "ABOVE-IDENTITY-BAND" not in section(res, "### Result 5")
          and "ABOVE-IDENTITY-BAND" not in section(res, "### Result 6")
          and re.search(r"H_pi0-loss-real.{0,40}SUPPORTED", section(res, "### Result 6"), re.S)
          is not None
          and re.search(r"H_install-emergent-real.{0,40}SUPPORTED", section(res, "### Result 5"),
                        re.S) is not None)),
        (4, "grid row 9 records N3 FIRED; Result 8 states the permuted value +0.0337",
         "N3" in r9 and "FIRED" in r9 and "+0.0337" in res),
        (5, "Result 8 / grid row 2 say '≈0 bits'; the outcome-was-public reason is restored",
         ("≈0 bits" in res or "~0 bits" in res) and re.search(r"outcome.{0,40}public", res) is not None),
        (6, "Result 9 headline is a band, not a bare 580; full 13-point ladder present",
         all(x in r9s for x in ("0.924", "0.711", "0.145", "0.996", "0.941"))
         and re.search(r"high hundreds|±\s*8\s*%|\bband\b", r9s) is not None),
        (7, "Result 10: pre-registration scoped to Leg 1; assembly + NULL_FIRES_MIN + ordering disclosed",
         "Leg 1" in r10 and "NULL_FIRES_MIN" in r10 and "assembl" in r10.lower()
         and "ordering" in r10.lower()),
        (8, "Result 3 prints both median-of-ratios and ratio-of-medians (or the C3 supersession)",
         ("ratio-of-medians" in r3 or "ratio of medians" in r3)
         and ("0.453" in r3 or "same-base" in r3)),
        (9, "Result 11 labels the two shuffle values as shuffle-emergent; non-independence disclosed",
         r11.count("shuffle-emergent") >= 2
         and re.search(r"not independent|non-independent|same crosscoder|overlapping cells",
                       r11) is not None),
        (10, "survival_fraction semantics corrected in doc AND tessera_crosscoder docstring",
         ("1 − emergent" in res or "1 - emergent" in res)
         and ("NOT survival" in xc or "not survival" in xc)),
        (11, "§7 F-K membership accounting stated (declared m vs produced); K2 tagged F-K in the ARTIFACT",
         (re.search(r"(3 of (the )?5|two of (the )?five|2 of 5).{0,140}(produced|member)"
                    r"|(produced|member).{0,140}(3 of (the )?5|2 of 5)", res, re.S) is not None
          and re.search(r'"quantity":\s*"K2[^"]*".{0,400}?"family":\s*"F-K"', agg_art, re.S)
          is not None
          and "members_present_in_grid" in agg_art)),
        (12, "install-pair forward deviation disclosed as a post-freeze deviation",
         "deviation" in res.lower() and "install" in res.lower()),
        (13, "A12 symmetric BELOW-FT-REFERENCE branch in tessera_s_core; Result 4 cites the recount",
         "BELOW-FT-REFERENCE" in sc and "BELOW-FT-REFERENCE" in res),
        (14, "MAP TESSERA cells verdicted; an actual TESSERA ROW in RESEARCH_INDEX and README ledger",
         (re.search(r"^\|\s*\*\*TESSERA\*\*", idx, re.M) is not None
          and re.search(r"^\|\s*\*\*TESSERA\*\*", rdm, re.M) is not None
          and re.search(r"TESSERA.{0,200}(SURVIVES|ENERGY-ALIGNED|verdict)", mp, re.S) is not None)),
        (15, "P3 initial VOID + scale-ladder amendment in §9; plant-gain clause in Result 9",
         "P3" in res and "VOID" in res and "0.139" in r9s),
        (16, "Result 3 states n_null and the shared-seed caveat",
         "n_null" in r3 and re.search(r"shared seed|shared-seed", r3) is not None),
        (17, "Result 3 discloses the low-risk nature of its pre-registration",
         re.search(r"low[- ]risk", r3) is not None),
        # Not a critic finding — a HEADING/BODY desync check added after the fold re-synced five
        # headings that still carried pre-critic claims. Headings are the same artifact-desync
        # class as figures: a reader who skims them gets the retracted story.
        # ---- 19-23: THE SUMMARY SECTIONS ----------------------------------------------------
        # The independent re-audit found five already-"PASSED" findings still asserting their
        # pre-critic form in the §7 grid and the §8 IFF bar. NONE of checks 1-18 could see them:
        # every one is scoped to a `### Result N` section, and the grid and IFF bar are neither.
        # Those two sections are the ones a reader is most likely to read in isolation. Same
        # disease as the headings, one layer further out.
        (19, "grid row 1 does not present H_nothing-still as a finding about curvature",
         "INSTRUMENT-SUSPECT" in grid_row(res, 1)),
        # Retightened after C3: the magnitude is no longer flagged-pending, it is MEASURED on the
        # same-base control. The row must name C3 rather than quote 0.388 bare on the old
        # denominator -- the check tracks the claim's state, it does not freeze one wording.
        (20, "grid row 6 attributes the magnitude to the C3 same-base control",
         ("0.3883" in grid_row(res, 6) and "same-base" in grid_row(res, 6))),
        (21, "grid row 8 does not quote a bare n*=580",
         not re.search(r"\*\*580\*\*|\bn\\?\*\s*=?\s*580\b", grid_row(res, 8))),
        (22, "the IFF bar does not reintroduce 'erasure' for π0",
         not re.search(r"erasure", iff) or "not licensed" in iff),
        (23, "the IFF bar does not credit K2 with one bit",
         "carries one bit" not in iff and ("≈0 bits" in iff or "~0 bits" in iff)),
        # Tightened twice: first to require the confound be disclosed, then again once the
        # confound was MEASURED and the avoidance reading retracted. A check has to track the
        # claim's current state, or it certifies a sentence that is no longer there.
        # Tightened three times, tracking the claim's actual state each time: disclose the
        # confound -> retract avoidance -> and now record that flatness explains the MAIN EFFECT
        # but not the within-class ordering, which the install census showed. A check that keeps
        # certifying a superseded sentence is worse than no check.
        (24, "Result 4 reports the measured confound, does not assert avoidance, AND records that "
             "flatness does not explain the within-class ordering",
         (re.search(r"1\.6301", section(res, "### Result 4")) is not None
          and re.search(r"NOT SUPPORTED|not supported", section(res, "### Result 4")) is not None
          and re.search(r"residual open|not explained by flatness|within-class ordering",
                        section(res, "### Result 4")) is not None
          and "NOT SUPPORTED" in section(res, "### Result 4").split("\n")[0])),
        # The catalogue went stale by three entries within an hour of being written, and its own
        # prose count is the kind of hand-typed number it exists to catalogue. Count it here.
        (25, "the label-drift catalogue's row count matches its prose count and is contiguous",
         _catalogue_ok(fixres)),
        # Result 3's clause was RETRACTED (finding 8) and then RE-ESTABLISHED by C3's same-base
        # control at 0.3883. So the heading may carry 0.388x again -- but only if it says which
        # control the number came from. "Does the heading match the body's current state" is the
        # invariant; "this string is banned" was only ever a proxy for it.
        (18, "no result HEADING still carries a claim its folded body retracts",
         ((("0.388×" not in section(res, "### Result 3").split("\n")[0])
           or "same-base" in section(res, "### Result 3").split("\n")[0])
          and ("returns one bit" not in section(res, "### Result 8").split("\n")[0])
          and ("does not predict" not in section(res, "### Result 7").split("\n")[0])
          and ("loses features" not in section(res, "### Result 6").split("\n")[0])
          and ("not a lineage-specific channel"
               not in section(res, "### Result 4").split("\n")[0]))),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    crit = _t(a.root, CRIT)
    assert crit, f"the critic finding register is missing at {CRIT} — the fold has nothing to cite"

    rows = []
    for n, desc, ok in build_checks(a.root):
        rows.append({"finding": n, "check": desc, "passed": bool(ok)})
        print(f"[{'PASS' if ok else 'FAIL'}] finding {n:>2}: {desc}")
    n_pass = sum(1 for r in rows if r["passed"])
    rep = {"object": "TESSERA-FIX Arm F acceptance checks (spec §3)",
           "n_findings": len(rows), "n_passed": n_pass,
           "all_folded": bool(n_pass == len(rows)), "checks": rows,
           "note": "a check evaluates the FILES, not a summary of them; a finding that cannot be "
                   "made mechanical must be rebutted in writing instead of passed silently"}
    if a.out:
        o = pathlib.Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "metrics.json").write_text(json.dumps(rep, indent=1))
    print(f"\nTESSERA_FIX_CHECK {n_pass}/{len(rows)} folded")
    if a.strict and n_pass != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
