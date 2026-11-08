"""N13 samples coverage (studies/next/07_PICKS/N13_SPEC.md §10.1): every arm x benchmark of the roster, present or PENDING. Laptop only.

    python -m studies.next.n13.scripts.samples_coverage --banks <dir> --out studies/next/n13/results/SAMPLES_COVERAGE.md
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.pg import n13_plan as P


def coverage(banks) -> list:
    rows = []
    for r in P.roster():
        for b in ("math500",) + (("gsm8k",) if r["gsm8k"] else ()):
            hits = [p for p in pathlib.Path(banks).rglob("samples.jsonl") if p.parent.name == b and p.parent.parent.name == r["arm"]]
            assert len(hits) <= 1, hits
            if not hits:
                rows.append((r["arm"], b, "PENDING", 0, ""))
                continue
            s = [json.loads(l) for l in hits[0].read_text().splitlines() if l.strip()]
            roles = sorted({x for q in s for x in q["roles"]})
            ok = {"lowest_right", "lowest_wrong"} <= set(roles)
            rows.append((r["arm"], b, "present" if ok else "present, missing a right or a wrong sample", len(s), ",".join(roles)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = coverage(a.banks)
    md = ["# N13 samples coverage", "", "| arm | benchmark | samples | rows | roles |", "|---|---|---|---|---|"]
    md += [f"| {a_} | {b} | {s} | {n} | {r} |" for a_, b, s, n, r in rows]
    pathlib.Path(a.out).write_text("\n".join(md) + "\n")
    print(f"{sum(r[2].startswith('present') for r in rows)} present, {sum(r[2] == 'PENDING' for r in rows)} PENDING")


if __name__ == "__main__":
    main()
