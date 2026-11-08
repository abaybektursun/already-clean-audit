"""Rebuild clusters.jsonl after re-answering clusterer rows (W4-W5 checker F15, 2026-10-24), with the instrument's own code.

usage (from the explain worktree, PYTHONPATH=.): python recollect_clusters_f15.py <explain root holding explain/> <batch prefix, e.g. w4c> @<file of expected ids>

`neurons_explain collect-clusters` collects once (it refuses ids already in calls.jsonl). After the 200 excerpt-read sample-1
answers were removed from the responses and re-answered, this script validates both clusterer batches exactly as
collect-clusters does (neurons_llm.validate_many with the clusterer partition and echo checks), rebuilds every row with
collect-clusters' row logic, and replaces clusters.jsonl only if the rows that changed are exactly among the expected ids'
poles and every expected id's response changed hash. The previous file is kept as clusters.before_recollect_<utc>.jsonl.
calls.jsonl gets one new row per re-answered id; the earlier rows stay as the record of the first answers.
"""
import datetime
import json
import os
import sys
from pathlib import Path

import infra.pg.neurons_explain as ne
import infra.pg.neurons_llm as llm


def main(root, prefix, expected_arg):
    explain = Path(root) / "explain"
    expected = set(Path(expected_arg[1:]).read_text().strip().split(",")) if expected_arg.startswith("@") else set(expected_arg.split(","))
    plans = {p["id"]: p for p in llm.read_jsonl(explain / "plans" / f"{prefix}-clusterer.jsonl")}
    batches, check, cross_check = ne.validation(explain, prefix, ["clusterer"])
    out, call_rows, problems = llm.validate_many(explain, batches, "mixed", check, cross_check)
    if problems:
        raise ValueError(f"{len(problems)} validation problem(s)\n  " + "\n  ".join(map(llm.problem_text, problems)))
    rows = []
    for batch in out.values():
        for rid, x in ne.answers_in_plan_order(batch, plans):
            rows.append({"pole": plans[rid]["pole"], "sample": plans[rid]["sample"],
                         "clusters": [{"c": c["c"], "name": c["name"], "echo": c["echo"], "exemplar_ranks": c["exemplar_ranks"]}
                                      for c in sorted(x["clusters"], key=lambda c: c["c"])],
                         "pairs": [[p["a"], p["b"], p["relation"]] for p in x["pairs"]]})
    for b in batches:
        for rid, row in sorted(llm.declined(explain, b).items()):
            rows.append({"pole": plans[rid]["pole"], "sample": plans[rid]["sample"], "declined": llm.decline_text(row)})
    cpath = explain / "clusters.jsonl"
    old = llm.read_jsonl(cpath)
    assert len(old) == len(rows), (len(old), len(rows))
    assert [(r["pole"], r["sample"]) for r in old] == [(r["pole"], r["sample"]) for r in rows], "row order changed"
    expected_keys = {(plans[i]["pole"], plans[i]["sample"]) for i in expected}
    changed = {(o["pole"], o["sample"]) for o, n in zip(old, rows) if o != n}
    assert changed <= expected_keys, f"rows changed outside the expected ids: {sorted(changed - expected_keys)[:5]}"
    logged = {r["id"]: r for r in llm.read_jsonl(explain / "calls.jsonl")}
    relog = [r for r in call_rows if r["id"] in expected]
    assert {r["id"] for r in relog} == expected, "re-answered ids missing from the validated rows"
    rehashed = {r["id"] for r in relog if logged[r["id"]]["response_sha256"] != r["response_sha256"]}
    assert rehashed == expected, f"ids whose response did not change: {sorted(expected - rehashed)[:5]}"
    keep = cpath.with_name("clusters.before_recollect_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl")
    assert not keep.exists()
    os.replace(cpath, keep)
    llm.write_jsonl_new(cpath, rows)
    llm.append_jsonl(explain / "calls.jsonl", relog)
    ne.record_models(explain)
    print(json.dumps({"rows": len(rows), "rows_changed": len(changed), "responses_rehashed": len(rehashed),
                      "calls_appended": len(relog), "kept": keep.name}))


if __name__ == "__main__":
    main(*sys.argv[1:])
