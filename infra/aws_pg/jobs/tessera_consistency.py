"""TESSERA-FIX Arm G — the map/registry consistency gate. Spec §4.1; critic finding 14.

"Comprehensive" has to be a CHECKED PREDICATE, not a mood. This script is the predicate. Exit ≠ 0
blocks DONE.

The four documents drift because they are maintained by hand at different times:

  MAP.md               the per-workstream × per-probe cell grid
  RESEARCH_INDEX.md    the registry of workstream → verdict → authoritative doc
  README.md            the ledger of what is live vs historical
  the result docs       where the numbers actually live

The gate checks CONSISTENCY between them, and deliberately does NOT check that any number is right —
numbers are the artifacts' job. It also enforces the house rule that registries carry POINTERS, not
numbers, since a number copied into a registry is a number that will rot there.

Checks:
  G1  every workstream row in RESEARCH_INDEX names an authoritative doc that EXISTS on disk
  G2  every workstream in RESEARCH_INDEX appears in the README ledger and vice versa
  G3  MAP.md renders with pending = 0 (no unfilled cell claimed as complete)
  G4  every VOID in MAP.md carries a gate id, so no cell is void "just because"
  G5  no bare numeric result in a registry row that is not also a pointer (the anti-rot rule)

  python -m infra.aws_pg.jobs.tessera_consistency --root . [--strict]

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

IDX = "studies/lineage/RESEARCH_INDEX.md"
RDM = "studies/lineage/README.md"
MAP = "studies/lineage/MAP.md"


def _t(root, rel):
    p = pathlib.Path(root) / rel
    return p.read_text() if p.exists() else ""


def expand_braces(pat):
    """`a_{x,y}.md` -> ['a_x.md', 'a_y.md'].

    The registry writes filesets in brace notation. A checker that globs the literal string reports
    every such row as a missing file — which is the checker being wrong, not the repo. Caught on the
    gate's own first run: 10 of 10 G1 'failures' were brace patterns.
    """
    m = re.search(r"\{([^}]*)\}", pat)
    if not m:
        return [pat]
    out = []
    for opt in m.group(1).split(","):
        out += expand_braces(pat[:m.start()] + opt.strip() + pat[m.end():])
    return out


def ledger_covers(code, rdm):
    """Is `code` visible in the ledger, allowing RANGE notation (`W1–W15` covers `W5`)?"""
    if re.search(rf"\b{re.escape(code)}\b", rdm):
        return True
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", code)
    if not m:
        return False
    stem, num = m.group(1), int(m.group(2))
    for r in re.finditer(rf"\b{stem}(\d+)\s*[–\-]\s*{stem}?(\d+)\b", rdm):
        if int(r.group(1)) <= num <= int(r.group(2)):
            return True
    return False


def registry_rows(idx):
    """[(code, row_text)] for every `| **CODE** ...` row in the registry."""
    out = []
    for m in re.finditer(r"^\|\s*\*\*([A-Za-z0-9_\-]+)\*\*\s*[^|]*\|(.*)$", idx, re.M):
        out.append((m.group(1), m.group(0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    idx, rdm, mp = _t(a.root, IDX), _t(a.root, RDM), _t(a.root, MAP)
    assert idx and rdm and mp, "one of RESEARCH_INDEX / README / MAP is missing"

    rows = registry_rows(idx)
    checks = []

    # G1 — every registry row points at a doc that exists.
    missing = []
    for code, text in rows:
        for pat in re.findall(r"`([^`]*\.md)`", text):
            for d in expand_braces(pat):
                if not list(root.rglob(pathlib.Path(d).name)):
                    missing.append((code, d))
    checks.append({"id": "G1", "desc": "registry rows point at docs that exist on disk",
                   "passed": not missing, "detail": missing[:10],
                   "n_rows": len(rows)})

    # G2 — registry ↔ README ledger agreement.
    codes = {c for c, _ in rows}
    in_rdm = {c for c in codes if ledger_covers(c, rdm)}
    checks.append({"id": "G2", "desc": "every registry workstream appears in the README ledger",
                   "passed": bool(codes) and codes == in_rdm,
                   "detail": sorted(codes - in_rdm)[:10]})

    # G3 — MAP is ACTUALLY pending-free, not merely declared so.
    # v1 string-matched the "pending = 0" headline. The re-audit found three rows tagged with the
    # legend's own pending glyph while that headline read 0 — a check that reads a claim instead of
    # counting the thing the claim is about. Count the glyphs.
    pend = re.findall(r"pending\s*[:=]\s*(\d+)", mp, re.I)
    glyph_rows = [ln for ln in mp.splitlines()
                  if re.match(r"^\|.*\|\s*·\s*\|", ln)]
    checks.append({"id": "G3", "desc": "MAP.md is pending-free: headline says 0 AND no row carries "
                                       "the pending glyph",
                   "passed": (bool(pend) and all(int(x) == 0 for x in pend)
                              and not glyph_rows),
                   "detail": {"declared": pend[:5],
                              "rows_tagged_pending": [r[:90] for r in glyph_rows[:5]],
                              "n_tagged": len(glyph_rows)}})

    # G4 — every VOID in the MAP names a gate.
    bad_void = []
    for m in re.finditer(r"VOID", mp):
        w = mp[max(0, m.start() - 200):m.end() + 200]
        if not re.search(r"gate|G\d|N\d|P\d|MIN_|no range|no-range|VOID-", w):
            bad_void.append(mp[max(0, m.start() - 60):m.end() + 60].replace("\n", " "))
    checks.append({"id": "G4", "desc": "every MAP VOID carries a gate id",
                   "passed": not bad_void, "detail": bad_void[:5]})

    # G5 — registries carry pointers, not numbers that will rot.
    # A row is flagged only when it carries a bare decimal WITHOUT any adjacent artifact pointer,
    # which is the shape that actually rots (a number with a doc/artifact beside it is citable).
    rot = []
    for code, text in rows:
        if re.search(r"\d\.\d{3,}", text) and not re.search(r"`[^`]*\.(md|json|py)`", text):
            rot.append(code)
    checks.append({"id": "G5", "desc": "no bare numeric result in a registry row without a pointer",
                   "passed": not rot, "detail": rot[:10]})

    # G6 — every S3 artifact a doc cites must actually resolve.
    # The re-audit found `results/tessera_w4_audit` cited with nothing at that path: the job had run
    # locally against Modal inputs and its output was never synced. A citation that resolves to
    # nothing is worse than no citation, because it reads as provenance.
    # NOTE: an S3 URI stem is a PREFIX, not a directory. Checking it with a trailing slash reports
    # legitimate family pointers as broken -- `results/plenum_fc` matches 23 objects and
    # `results/plenum_ctl` 8. That false alarm is why this check matches prefixes.
    # ONE listing, matched locally. The first version made an S3 call per cited prefix and took
    # minutes, which is how a gate stops being run.
    import subprocess as _sp
    # Match only inside an s3:// URI. A bare `results/X` also appears for the MODAL `pg-results`
    # volume (W4's banked crosscoders live there, not on S3), and counting those as broken S3
    # citations is the checker being wrong about what it checks -- the third time in this program.
    S3CITE = re.compile(r"s3://pg-lineage-769063704696/results/([a-zA-Z0-9_]+)")
    cited = set()
    for txt in (mp, idx, rdm):
        cited |= set(S3CITE.findall(txt))
    for d in sorted(pathlib.Path(a.root, "studies/lineage/paligemma1-pi0/research").glob("*.md")):
        cited |= set(S3CITE.findall(d.read_text()))
    listing = _sp.run(["aws", "s3", "ls", "s3://pg-lineage-769063704696/results/",
                       "--profile", "fuelos", "--region", "us-east-1"],
                      capture_output=True, text=True)
    assert listing.returncode == 0, f"could not list the results bucket: {listing.stderr[:200]}"
    existing = {ln.split()[-1].rstrip("/") for ln in listing.stdout.splitlines() if ln.strip()}
    broken = [c for c in sorted(cited) if not any(e.startswith(c) for e in existing)]
    checks.append({"id": "G6", "desc": "every cited S3 artifact prefix resolves to at least one "
                                       "object",
                   "passed": not broken, "detail": {"n_cited": len(cited), "broken": broken[:10]}})

    # G7 — every TESSERA artifact the result docs cite is also banked IN THE REPO.
    # G6 proves the S3 copy exists; this proves a reader without bucket credentials can still
    # check the numbers. The re-audit could only cross-check Result 10 against the journal's
    # transcription because its inputs lived on a Modal volume it had no access to.
    RES_DIR = pathlib.Path(a.root, "studies/lineage/paligemma1-pi0/research")
    ART = RES_DIR / "artifacts"
    cited_t = set()
    for d in ("2026-08-02_tessera_result.md", "2026-08-03_tessera_fix_result.md"):
        f = RES_DIR / d
        if f.exists():
            cited_t |= set(re.findall(r"results/(tessera_[a-zA-Z0-9_]+)", f.read_text()))
    banked = {q.name.replace("_metrics.json", "").replace("_controls.json", "")
              for q in ART.glob("*.json")} if ART.exists() else set()
    unbanked = sorted(c for c in cited_t if c not in banked)
    checks.append({"id": "G7", "desc": "every cited TESSERA artifact is banked in-repo, so the "
                                       "result docs are checkable without S3 credentials",
                   "passed": not unbanked,
                   "detail": {"n_cited": len(cited_t), "unbanked": unbanked[:10]}})

    # G8 — every FROZEN spec copy still matches its source.
    # The C6 amendment was added to the source 23 minutes after I copied and froze it, and I never
    # re-checked: the frozen hash is exactly what made the staleness invisible. A freeze protects
    # the analysis from the data, not the analyst from an amended contract.
    import hashlib as _h
    MAIN = pathlib.Path("/Users/abaybektursun/projects/ntm-frozen-gemma/studies/lineage/"
                        "paligemma1-pi0/research")
    # §5.1 of the core-provenance addendum allows exactly two states for a mismatch:
    #   (a) an amendment adopted with its OWN new tag and freeze, before the data it governs, or
    #   (b) a FORK, disclosed IN BOTH COPIES with a notice naming which is authoritative.
    # So the gate fails on UNDISCLOSED drift, not on drift as such. Anything else would make the
    # only way to pass be "never fork", which is not the rule the director wrote.
    OWNED = ("tessera_program_spec", "tessera_followup_fix_spec", "core_provenance_addendum_spec")
    undisclosed, disclosed, other = [], [], []
    for spec in sorted(RES_DIR.glob("*_spec.md")):
        src = MAIN / spec.name
        if not src.exists():
            continue
        ta, tb = spec.read_text(), src.read_text()
        if _h.sha256(ta.encode()).hexdigest() == _h.sha256(tb.encode()).hexdigest():
            continue
        rec = {"spec": spec.name,
               "notice_in_executor_copy": "FORK NOTICE" in ta,
               "notice_in_source_copy": "FORK NOTICE" in tb}
        if not any(spec.stem.endswith(o) for o in OWNED):
            other.append(rec)                      # another program's spec: informational only
        elif rec["notice_in_executor_copy"] and rec["notice_in_source_copy"]:
            disclosed.append(rec)
        else:
            undisclosed.append(rec)
    checks.append({"id": "G8", "desc": "no UNDISCLOSED drift between a frozen spec copy and its "
                                       "source (a disclosed fork, noticed in both copies, passes)",
                   "passed": not undisclosed,
                   "detail": {"undisclosed": undisclosed, "disclosed_forks": disclosed,
                              "other_programs_drifted_informational": [x["spec"] for x in other]}})

    # G9 — fork rule CLAUSE 3: one question, one execution owner.
    # Clauses 1-2 (blessed copy, sha-compare) stopped divergent COPIES; they did not stop the
    # duplicate C6 execution, which cost a GPU cycle and produced two contradictory C6b verdicts.
    # Every registry row for a program THIS repo dispatched must name its execution owner, so a
    # second executor verifies-and-cross-references instead of re-running.
    DISPATCHED = ("TESSERA", "CORE-PROVENANCE")
    unclaimed = []
    for code in DISPATCHED:
        m = re.search(rf"^\|\s*\*\*{re.escape(code)}\*\*.*$", idx, re.M)
        if not m:
            unclaimed.append({"code": code, "why": "no registry row"})
        elif not re.search(r"claimed-by|worktree|execution[s]? \(|executions?\b", m.group(0)):
            unclaimed.append({"code": code, "why": "row names no execution owner"})
    checks.append({"id": "G9", "desc": "every dispatched program's registry row names its execution "
                                       "owner (fork rule clause 3)",
                   "passed": not unclaimed, "detail": unclaimed})

    n_pass = sum(1 for c in checks if c["passed"])
    rep = {"object": "TESSERA-FIX Arm G — map/registry consistency gate",
           "n_checks": len(checks), "n_passed": n_pass,
           "consistent": bool(n_pass == len(checks)), "checks": checks,
           "scope": "consistency BETWEEN documents; correctness of any number is the artifacts' job"}
    if a.out:
        o = pathlib.Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['id']}: {c['desc']}"
              + (f"  -> {c['detail']}" if not c["passed"] else ""))
    print(f"\nTESSERA_CONSISTENCY {n_pass}/{len(checks)}")
    if a.strict and n_pass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
