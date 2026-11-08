"""Rebuild judgments.jsonl after correcting judge rows (W4-W5 checker F14, 2026-10-24), with the instrument's own code.

usage (from the explain worktree, with its uv environment):
  python recollect_judge_f14.py <explain root holding explain/> <batch prefix, e.g. w4j> <expected changed ids, comma-separated>

`neurons_explain collect-judge` collects once: it appends to judgments.jsonl and refuses ids already in calls.jsonl. After
nine misattached rows were removed from the responses and re-answered, this script validates every batch exactly as
collect-judge does (neurons_llm.validate_many with the judge checks), rebuilds every judgments row with collect-judge's row
logic, and replaces judgments.jsonl only if the rows that changed are exactly the expected ids (answer or correct differ,
or the response changed). The previous file is kept as judgments.before_recollect_<utc>.jsonl. calls.jsonl gets one new row
per re-answered id (its new response hash, backend and utc); the earlier rows stay as the record of the first answers.
Reads no response text beyond what the instrument's validation reads.
"""
import json
import os
import sys
from pathlib import Path

import infra.pg.neurons_explain as ne
import infra.pg.neurons_llm as llm


def main(root, prefix, expected_csv):
    explain = Path(root) / "explain"
    if expected_csv.startswith("@"):            # @<file>: the comma-separated ids are read from that file
        expected_csv = Path(expected_csv[1:]).read_text().strip()
    expected = set(expected_csv.split(","))
    plans = {p["id"]: p for p in llm.read_jsonl(explain / "plans" / f"{prefix}-judge.jsonl")}
    batches, check, cross_check = ne.validation(explain, prefix, ["judge"])
    out, call_rows, problems = llm.validate_many(explain, batches, "mixed", check, cross_check)
    if problems:
        raise ValueError(f"{len(problems)} validation problem(s)\n  " + "\n  ".join(map(llm.problem_text, problems)))

    def row(rid):
        p = plans[rid]
        return {"id": rid, **{k: p[k] for k in ("cluster", "control", "decoy_type", "decoy_pole", "draw", "t_before",
                                               "sample", "shown_order")}}
    rows = []
    for batch in out.values():
        for rid, x in ne.answers_in_plan_order(batch, plans):
            truth = "A" if plans[rid]["shown_order"][0] == "true" else "B"
            rows.append({**row(rid), "answer": x["answer"], "correct": x["answer"] == truth})
    for b in batches:
        for rid, d in sorted(llm.declined(explain, b).items()):
            rows.append({**row(rid), "answer": None, "correct": None, "declined": llm.decline_text(d)})

    jpath = explain / "judgments.jsonl"
    old = llm.read_jsonl(jpath)
    assert len(old) == len(rows) == len(plans), (len(old), len(rows), len(plans))
    assert [r["id"] for r in old] == [r["id"] for r in rows], "id order changed"
    changed = {o["id"] for o, n in zip(old, rows) if o != n}
    calls = explain / "calls.jsonl"
    logged = {}
    for r in llm.read_jsonl(calls):
        logged[r["id"]] = r
    relog = [r for r in call_rows if r["id"] in expected]
    assert {r["id"] for r in relog} == expected, "re-answered ids missing from the validated rows"
    rehashed = {r["id"] for r in relog if logged[r["id"]]["response_sha256"] != r["response_sha256"]}
    assert changed <= expected, f"rows changed outside the expected ids: {sorted(changed - expected)[:5]}"
    assert rehashed == expected, f"ids whose response did not change: {sorted(expected - rehashed)}"
    import datetime
    keep = jpath.with_name("judgments.before_recollect_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl")
    assert not keep.exists()
    os.replace(jpath, keep)
    llm.append_jsonl(jpath, rows)
    llm.append_jsonl(calls, relog)
    ne.record_models(explain)
    print(json.dumps({"judgments": len(rows), "rows_changed": sorted(changed), "n_changed": len(changed),
                      "responses_rehashed": len(rehashed), "calls_appended": len(relog), "kept": keep.name}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
