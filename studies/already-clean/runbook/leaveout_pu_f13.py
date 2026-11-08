"""Leave-out p_u for the W4-W5 checker's F13 (2026-10-24), computed with the instrument's own estimator.

usage (from the explain worktree, PYTHONPATH=.): python leaveout_pu_f13.py <explain dir> <chunk root (holds w4j_s0, w4j_s1)> <audit_poles.json> <list file>...
Each list file holds lines `<batch> <chunk>` (e.g. `s0 016`). For each list, the judge lines answered in those chunks (ids from
the chunks' *.imported.jsonl files) are left out of judgments.jsonl, and p_u is recomputed exactly as q2_readouts does
(neurons_verdict.judged, then p_u_bootstrap over the 600 poles of S with seed 0 and 10,000 draws; a cluster or control
with no remaining line drops out). The primary (nothing left out) is printed first and must reproduce metrics.json.
"""
import json
import sys
from pathlib import Path

import infra.pg.neurons_verdict as v


def ids_in_chunks(chunk_root: Path, pairs):
    ids = set()
    for batch, chunk in pairs:
        f = chunk_root / f"w4j_{batch}" / f"chunk_{int(chunk):03d}.imported.jsonl"
        ids |= {json.loads(l)["id"] for l in open(f)}
    return ids


def p_u_leaving_out(judgments, poles, left_out):
    rows = [j for j in judgments if j["id"] not in left_out and j["answer"] is not None and j["decoy_type"] == "random"]
    tests, ctrls = {}, {}            # as neurons_verdict.judged groups them, without its full-plan count check
    for j in rows:
        if j["control"]:
            ctrls.setdefault(j["cluster"].split(":c")[0], []).append(bool(j["correct"]))
        else:
            tests.setdefault(j["cluster"], []).append(bool(j["correct"]))
    by_pole = {p: [] for p in poles}
    for cl, t in tests.items():
        by_pole[cl.split(":c")[0]].append(float(sum(t) / len(t)))
    ctrl = [float(sum(ctrls[p]) / len(ctrls[p])) if p in ctrls else None for p in poles]
    pu = v.p_u_bootstrap([by_pole[p] for p in poles], ctrl, v.N_BOOT, 0)
    return {"p_u": round(pu["point"], 4), "ci95": [round(x, 4) for x in pu["ci95"]], "rule": v.p_u_rule(pu["ci95"]),
            "n_clusters": pu["n_clusters"], "n_controls": sum(c is not None for c in ctrl), "lines_left_out": len(left_out),
            "degenerate_draws": pu["degenerate_draws"]}


def main(explain, chunk_root, poles_path, *lists):
    judgments = v.read_jsonl(Path(explain) / "judgments.jsonl")
    poles = sorted(json.load(open(poles_path)))
    print(json.dumps({"primary": p_u_leaving_out(judgments, poles, set()), "judgments_sha256": v.file_sha256(Path(explain) / "judgments.jsonl")[:16]}))
    for lf in lists:
        pairs = [l.split() for l in open(lf) if l.strip()]
        out = p_u_leaving_out(judgments, poles, ids_in_chunks(Path(chunk_root), pairs))
        print(json.dumps({"list": Path(lf).name, "chunks": len(pairs), **out}))


if __name__ == "__main__":
    main(*sys.argv[1:])
